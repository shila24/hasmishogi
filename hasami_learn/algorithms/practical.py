"""Masked practical-score and slip learning from calibrated proxy continuations."""

from __future__ import annotations

import copy
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..batched_selfplay import BatchEvaluator, DecisionRequest
from ..checkpoint_bundle import sha256_file
from ..config import TrainingConfig
from ..devices import resolve_inference_batch
from ..encoding import masked_log_softmax, sample_from_policy
from ..games import action_encoding, default_rules
from ..games.base import encode_player, require_search_game, terminal_rewards
from ..manifest import (ModelManifest, atomic_write_json, check_manifest_matches_spec,
                        load_checkpoint, save_checkpoint)
from ..parallel import selfplay_batches
from ..practical import (NetworkEvaluator, PracticalAdvisor, PracticalAgent, ProxyAgent,
                         load_levels, mover, prior_distribution, proxy_distributions,
                         sample_distribution, validate_levels)
from ..registry import ALGORITHMS
from ..search.mcts import MCTS, MctsConfig, batched_evaluator
from ..selfplay import augment, draw_value_for_iteration
from .base import CollectStats, Learner, UpdateStats


@dataclass
class PracticalParams:
    init: str = "engine"
    simulations: int = 64
    recalibrate_every: int = 10
    calibration_tiny: bool = False
    levels_file: str | None = None
    engine_checkpoint: str | None = None
    games_per_iteration: int = 64
    rollouts_per_start: int = 2
    on_policy_rate: float = 0.25
    slip_delta: float = 0.2
    distill_weight: float = 0.25
    slip_weight: float = 0.5
    default_level: int = 3
    batch_games: int = 8
    engine_store: str | None = None
    store_games: int = 64
    bootstrap_tiny: bool = False

    def __post_init__(self):
        if self.init not in ("engine", "joint"):
            raise ValueError("init must be engine or joint")
        if self.init == "joint" and (self.engine_checkpoint or self.levels_file or self.bootstrap_tiny or self.engine_store):
            raise ValueError("joint training starts from scratch and cannot use pretrained inputs or bootstrap_tiny")
        for name in ("games_per_iteration", "rollouts_per_start", "default_level", "batch_games", "store_games",
                     "simulations", "recalibrate_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("on_policy_rate", "slip_delta", "distill_weight", "slip_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.on_policy_rate > 1:
            raise ValueError("on_policy_rate must be at most one")


def masked_practical_loss(logits, own_levels, opponent_levels, targets):
    """BCE on one zero-based labelled cell per row; all other cells have zero gradient."""
    rows = torch.arange(logits.shape[0], device=logits.device)
    return F.binary_cross_entropy_with_logits(logits[rows, own_levels, opponent_levels], targets)


def joint_loss(logits, value, az, practical, slip, proxy, *, slip_weight):
    """Combine visit/outcome targets and the two labelled practical outputs."""
    logp = masked_log_softmax(logits.float(), az["mask"]).masked_fill(az["mask"] <= 0, 0)
    rows = torch.arange(len(proxy["own"]), device=logits.device)
    terms = dict(policy=-(az["policy"] * logp).sum(1).mean(),
                 value=F.mse_loss(value.float(), az["value"]),
                 practical=masked_practical_loss(practical.float(), proxy["own"], proxy["other"], proxy["target"]),
                 slip=F.binary_cross_entropy_with_logits(slip[rows, proxy["own"]].float(), proxy["slip"]))
    return terms["policy"] + terms["value"] + terms["practical"] + slip_weight * terms["slip"], terms


class ProxySeat:
    """Search-free proxy decision through the shared BatchSeat evaluator contract."""

    records = True

    def __init__(self, agent, evaluator, encoder):
        self.agent, self.evaluator, self.encoder = agent, evaluator, encoder

    def needs_evaluation(self, request):
        return True

    def decide(self, request, logits, value):
        legal = np.flatnonzero(request.mask > 0).tolist()
        distribution = self.agent.from_prior(prior_distribution(logits, legal))
        return sample_distribution(distribution, self.agent.rng), None, {}


@ALGORITHMS.register("practical")
class PracticalLearner(Learner):
    name = "practical"
    required_head = "policy_value"
    conformance_tiny = False
    replay_kind = None

    def params_dataclass(self):
        return PracticalParams

    def __init__(self, ctx):
        super().__init__(ctx)
        require_search_game(self.spec)
        if self.config.opponents.snapshot_every:
            raise ValueError("practical requires opponents.snapshot_every=0: pool snapshots lack the auxiliary-file hook")
        self.samples = []
        if self.params.init == "joint":
            self._init_joint()
            return
        if self.params.bootstrap_tiny:
            if self.params.engine_checkpoint or self.params.levels_file:
                raise ValueError("bootstrap_tiny cannot override supplied artifacts")
            self._bootstrap()
        if not self.params.engine_checkpoint or not self.params.levels_file:
            raise ValueError("practical training requires engine_checkpoint and levels_file")
        loaded = load_checkpoint(self.params.engine_checkpoint, self.device, expect_game_id=self.spec.game_id)
        check_manifest_matches_spec(loaded.manifest, self.spec, self.config.game.params)
        self.engine_path = loaded.directory.resolve()
        self.engine_hash = sha256_file(self.engine_path / "weights.pt")
        self.levels = load_levels(self.params.levels_file)
        if self.levels["engine_sha256"] != self.engine_hash:
            raise ValueError("calibration engine hash differs from training engine")
        k = self.levels["k"]
        required = {"practical": {"shape": [k, k], "activation": "sigmoid"},
                    "slip": {"shape": [k], "activation": "sigmoid"}}
        if any(getattr(self.network, "aux_heads", {}).get(n) != v for n, v in required.items()):
            raise ValueError("configure practical/slip sigmoid aux_heads matching levels k")
        engine_params = dict(loaded.manifest.network["params"])
        learner_params = dict(self.config.network.params)
        engine_params.pop("aux_heads", None)
        learner_params.pop("aux_heads", None)
        if (loaded.manifest.network["name"] != self.config.network.name or engine_params != learner_params
                or loaded.manifest.encoding_id != self.encoder.encoding_id
                or loaded.manifest.encoder_params != self.config.encoder.params):
            raise ValueError("practical network and encoder must match engine architecture")
        own_state = self.network.state_dict()
        base_state = loaded.network.state_dict()
        if any(key not in own_state or own_state[key].shape != value.shape for key, value in base_state.items()):
            raise ValueError("engine weights do not match practical network")
        missing = sorted(key for key in own_state.keys() - base_state.keys() if not key.startswith("aux."))
        if missing:
            raise ValueError(f"unexpected engine weight mapping: missing {missing}")
        # Trainer already loaded the complete warm-start state, including auxiliary heads.
        # The calibrated engine remains separate and must not overwrite those weights.
        if self.config.run.init_checkpoint is None:
            self.network.load_state_dict(base_state, strict=False)
        loaded.network.requires_grad_(False)
        self.engine = NetworkEvaluator(loaded.network, self.encoder, self.spec, self.device,
                                       self.config.device.inference_batch)
        self.engine.game_params = self.config.game.params
        self.engine.manifest = loaded.manifest
        self.practical = NetworkEvaluator(self.network, self.encoder, self.spec, self.device,
                                          self.config.device.inference_batch)
        self.advisor = PracticalAdvisor(self.engine, self.practical, self.levels,
                                        checkpoint_root=Path(self.params.levels_file).resolve().parent)
        self.advisor._level(self.params.default_level)
        self.available = [row.level for row in self.advisor.levels if any(
            name in row.styles for name in self.levels["train_styles"])]
        if not self.available:
            raise ValueError("no level has an available training style")
        if self.params.default_level not in self.available:
            raise ValueError("default_level has no available training proxy")
        self.starts = self._store_starts()

    def _init_joint(self):
        heads = getattr(self.network, "aux_heads", {})
        shape = heads.get("slip", {}).get("shape", [])
        if len(shape) != 1 or heads.get("practical") != {"shape": [shape[0], shape[0]], "activation": "sigmoid"} or heads["slip"]["activation"] != "sigmoid":
            raise ValueError("joint training requires matching practical/slip sigmoid aux_heads")
        self.k = shape[0]
        if not 1 <= self.params.default_level <= self.k:
            raise ValueError("default_level is outside the auxiliary head range")
        if self.params.calibration_tiny and (str(self.device) != "cpu" or self.k != 1):
            raise ValueError("calibration_tiny requires CPU and one level")
        if self.config.selfplay.workers != 1:
            raise ValueError("joint training currently requires selfplay.workers=1")
        self.engine = self.practical = NetworkEvaluator(self.network, self.encoder, self.spec, self.device,
                                                        self.config.device.inference_batch)
        self.engine.game_params = self.config.game.params
        self.search = MCTS(batched_evaluator(self.network, self.device, self.config.device.amp), self.encoder,
                           MctsConfig(simulations=self.params.simulations, batch_size=self.params.batch_games,
                                      truncation_value=self.config.selfplay.truncation_value), self.rng, spec=self.spec)
        self.az_samples, self.starts = [], []
        self.levels = None
        self.calibration_iteration = -1

    def _engine_manifest(self, iteration):
        return ModelManifest(algorithm=self.name, network=dict(name=self.config.network.name,
            params=dict(self.config.network.params), head=self.required_head), encoding_id=self.encoder.encoding_id,
            num_planes=self.encoder.num_planes, encoder_params=dict(self.config.encoder.params),
            game_id=self.spec.game_id, game_params=dict(self.config.game.params), num_players=self.spec.num_players,
            board_shape=self.spec.board_shape, ruleset_id=self.spec.ruleset_id, rules=default_rules(self.spec),
            action_size=self.spec.action_size, action_encoding=action_encoding(self.spec),
            algorithm_params=dict(self.config.algorithm.params), run_id=self.config.run.run_id, iteration=iteration,
            train_device=str(self.device), inference_hint=dict(mode="mcts", simulations=self.params.simulations),
            notes=self.config.run.notes)

    def _joint_advisor(self):
        self.advisor = PracticalAdvisor(self.engine, self.practical, self.levels)
        self.available = [row.level for row in self.advisor.levels if any(
            style in row.styles for style in self.levels["train_styles"])]
        if self.params.default_level not in self.available:
            raise ValueError("default_level has no available current-network training proxy")

    def _recalibrate(self, iteration):
        from ..practical_calibrate import calibrate

        # The snapshot is only calibration input; training always uses the live network.
        with tempfile.TemporaryDirectory(prefix="joint-calibration-") as temporary:
            root = Path(temporary)
            save_checkpoint(root / "engine", self.network, self._engine_manifest(iteration))
            levels = calibrate(root / "engine", output=root / "levels.json", device=str(self.device),
                backend=self.backend, k=self.k, seed=self.config.run.seed + iteration,
                max_plies=self.config.selfplay.max_plies, train_styles=("qre", "temperature"),
                heldout_styles=("alphabeta",), joint=True, tiny_budget=self.params.calibration_tiny,
                smoke=self.params.calibration_tiny)
        levels["notes"] += f" Calibration iteration={iteration}; measured engine sha256={levels['engine_sha256']}."
        levels["engine_checkpoint"] = "engine"
        self.levels = levels
        self.calibration_iteration = iteration
        self._joint_advisor()
        # Labels from the previous proxy population must not be mixed with new level meanings.
        self.samples = []

    def _collect_alphazero(self, iteration):
        cfg = self.config.selfplay
        def decision(result, mask, ply):
            policy = result.visits / result.visits.sum()
            temperature = cfg.temperature if ply < cfg.temperature_plies else 0.0
            return sample_from_policy(policy, mask, self.rng, temperature), policy.astype(np.float32), {}
        stats = CollectStats()
        wins = [0, 0, 0]
        for trajectories in selfplay_batches(self.search, self.encoder, decision, spec=self.spec,
                games=cfg.games_per_iteration, batch_games=self.params.batch_games, backend=self.backend,
                max_plies=cfg.max_plies, seed=self.config.run.seed + iteration * cfg.games_per_iteration,
                draw_value=draw_value_for_iteration(cfg, iteration), draw_reasons=cfg.draw_reasons):
            for trajectory in trajectories:
                values = trajectory.values(cfg.truncation_value)
                for planes, mask, policy, value in zip(trajectory.planes, trajectory.masks, trajectory.policies, values):
                    self.az_samples.append(dict(planes=planes, mask=mask, policy=policy, value=float(value)))
                    if cfg.symmetry_augmentation:
                        for p, m, pi, _ in augment(self.spec, planes, mask, policy, None):
                            self.az_samples.append(dict(planes=p, mask=m, policy=pi, value=float(value)))
                stats.games += 1
                stats.plies += trajectory.plies
                stats.truncated += int(trajectory.truncated)
                if not trajectory.truncated:
                    wins[trajectory.result_index()] += 1
        self.az_samples = self.az_samples[-self.config.replay.capacity:]
        stats.wins = tuple(wins)
        return stats

    def _bootstrap(self):
        """Opt-in CPU smoke fixture: two engine iterations and synthetic calibration."""
        from ..practical_calibrate import synthetic_levels
        from ..trainer import train_from_config

        if str(self.device) != "cpu" or self.config.run.iterations > 2:
            raise ValueError("bootstrap_tiny is limited to CPU runs of at most two iterations")
        raw = self.config.to_dict()
        engine_run = Path(tempfile.mkdtemp(prefix="smoke-engine-", dir=self.run_dir))
        raw["run"].update(output_dir=str(self.run_dir), run_id=engine_run.name, iterations=2,
                          keep_checkpoints=3, eval_every=0, init_checkpoint=None)
        # The auxiliary-free smoke engine is a fresh run, not the practical warm-start target.
        raw["algorithm"] = {"name": "alphazero", "params": {"simulations": 2, "batch_games": 2}}
        raw["network"]["params"].pop("aux_heads", None)
        raw["selfplay"].update(games_per_iteration=2, max_plies=4, symmetry_augmentation=False)
        raw["replay"].update(min_size=1, capacity=128, batch_size=8)
        raw["datastore"].update(enabled=True, path=None)
        raw["opponents"]["snapshot_every"] = 0
        raw["evaluation"]["games"] = 0
        torch_state = torch.get_rng_state()
        try:
            checkpoint = train_from_config(TrainingConfig.from_dict(raw), echo=False)
        finally:
            torch.set_rng_state(torch_state)
        self.params.engine_checkpoint = str(checkpoint)
        earlier = checkpoint.parent / "iter_000001"
        levels = synthetic_levels(checkpoint, earlier, k=self.network.aux_heads["slip"]["shape"][0],
                                  seed=self.config.run.seed)
        path = self.run_dir / "smoke-levels.json"
        atomic_write_json(path, levels)
        self.params.levels_file = str(path)
        self.params.engine_store = str(checkpoint.parent.parent / "selfplay")

    def _store_starts(self):
        root = Path(self.params.engine_store) if self.params.engine_store else self.engine_path.parent.parent / "selfplay"
        if not root.is_dir():
            return []
        # A read-only traversal avoids recovery/eviction side effects of opening SelfplayStore.
        starts = []
        import json
        for path in sorted((root / "games").glob("*.npz"))[-self.params.store_games:]:
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"].item()))
                if (metadata.get("game_id") != self.spec.game_id
                        or metadata.get("game_params", {}) != self.config.game.params
                        or metadata.get("ruleset_id") != self.spec.ruleset_id
                        or metadata.get("rules") != self.engine.manifest.rules):
                    raise ValueError("engine store contains an incompatible game")
                joints = metadata.get("joint_actions")
                if not joints:
                    moves = archive["moves"].tolist()
                    joints = None
                else:
                    moves = None
                starts.append((metadata.get("seed"), joints, moves))
        return starts

    def _start(self):
        game = None
        try:
            if self.starts and self.rng.random() < 0.5:
                seed, joints, moves = self.starts[int(self.rng.integers(len(self.starts)))]
                game = self.spec.create_game(self.backend, seed=seed)
                length = len(joints if joints is not None else moves)
                # Uniform bucket, then uniform ply within that bucket, prevents long games dominating.
                buckets = [list(range(a, min(length, a + 20))) for a in range(0, length, 20)]
                bucket = buckets[int(self.rng.integers(len(buckets)))] if buckets else [0]
                stop = int(self.rng.choice(bucket))
                for ply in range(stop):
                    actions = ({int(p): int(a) for p, a in joints[ply].items()} if joints is not None
                               else {mover(self.spec, game): int(moves[ply])})
                    outcome = self.spec.step(game, actions)
                    if outcome.terminal or outcome.truncated:
                        game.close()
                        game = self.spec.create_game(self.backend)
                        break
            else:
                game = self.spec.create_game(self.backend)
            if self.rng.random() < self.params.on_policy_rate:
                level = int(self.rng.choice(self.available))
                advice = self.advisor.advise(game, level, top_k=1)[0]
                outcome = self.spec.step(game, {mover(self.spec, game): advice.action})
                if outcome.terminal or outcome.truncated:
                    game.close()
                    game = self.spec.create_game(self.backend)
            return game
        except BaseException:
            if game is not None:
                game.close()
            raise

    def _proxy(self, level, seed):
        styles = [name for name in self.levels["train_styles"] if name in self.advisor.levels[level - 1].styles]
        return self.advisor.proxy_agent(level, style=str(self.rng.choice(styles)), seed=seed)

    def collect(self, iteration):
        started = time.perf_counter()
        az_stats = None
        if self.params.init == "joint":
            if self.levels is None:
                self._recalibrate(0)
            az_stats = self._collect_alphazero(iteration)
        stats = CollectStats()
        wins = [0, 0, 0]
        self.network.eval()
        new_samples = []
        base = None
        states = []
        try:
            for index in range(self.params.games_per_iteration):
                if index % self.params.rollouts_per_start == 0:
                    if base is not None:
                        base.close()
                    base = self._start()
                levels = [int(self.rng.choice(self.available)) for _ in range(2)]
                agents = [self._proxy(levels[p], int(self.rng.integers(2**31))) for p in range(2)]
                states.append(dict(game=base.clone(), levels=levels, agents=agents, rows=[], plies=0, done=False))
            cap = self.config.selfplay.max_plies
            evaluators = {}
            for offset in range(0, len(states), self.params.batch_games):
                batch = states[offset:offset + self.params.batch_games]
                while any(not state["done"] for state in batch):
                    active = [state for state in batch if not state["done"]]
                    games = [state["game"] for state in active]
                    players = [mover(self.spec, g) for g in games]
                    agents = [s["agents"][p] for s, p in zip(active, players)]
                    q_values = self.engine.one_ply_many(games)
                    actions = {}
                    search_indices = [i for i, agent in enumerate(agents) if agent.style == "qre"]
                    if search_indices:
                        distributions = proxy_distributions([agents[i] for i in search_indices],
                                                             [games[i] for i in search_indices],
                                                             q_values=[q_values[i] for i in search_indices])
                        for i, distribution in zip(search_indices, distributions):
                            actions[i] = sample_distribution(distribution, agents[i].rng)
                    groups = {}
                    for i, agent in enumerate(agents):
                        if agent.style == "qre":
                            continue
                        engine = agent.policy_engine
                        evaluator = evaluators.setdefault(engine, BatchEvaluator(engine.network, engine.device,
                                                            "off", resolve_inference_batch(self.config.device.inference_batch, self.device),
                                                            fixed_shape=self.config.device.fixed_batch_shape))
                        groups.setdefault(evaluator, []).append(i)
                    for evaluator, indices in groups.items():
                        requests = []
                        for i in indices:
                            requests.append(DecisionRequest(i, players[i], active[i]["plies"],
                                np.array(encode_player(self.encoder, games[i], players[i]), copy=True),
                                np.array(self.spec.legal_mask(games[i], players[i]), copy=True)))
                        logits, values = evaluator.forward(np.stack([r.planes for r in requests]))
                        for row, (i, request) in enumerate(zip(indices, requests)):
                            seat = ProxySeat(agents[i], evaluator, self.encoder)
                            actions[i] = seat.decide(request, logits[row], values[row])[0]
                    for i, (state, game, player) in enumerate(zip(active, games, players)):
                        action = actions[i]
                        if type(action) is not int or action not in self.spec.legal_actions(game, player):
                            raise ValueError("proxy produced an illegal action")
                        state["rows"].append(dict(planes=np.array(encode_player(self.encoder, game, player), copy=True),
                            mask=np.array(self.spec.legal_mask(game, player), copy=True),
                            own=state["levels"][player] - 1, other=state["levels"][1 - player] - 1,
                            slip=float(max(q_values[i].values()) - q_values[i][action] > self.params.slip_delta), player=player))
                        outcome = self.spec.step(game, {player: action})
                        state["plies"] += 1
                        truncated = outcome.truncated or (not outcome.terminal and cap > 0 and state["plies"] >= cap)
                        if outcome.terminal or truncated:
                            rewards = [0.0, 0.0] if truncated else terminal_rewards(self.spec, game)
                            for row in state["rows"]:
                                row["target"] = (rewards[row.pop("player")] + 1) / 2
                                new_samples.append(row)
                            stats.games += 1
                            stats.plies += state["plies"]
                            stats.truncated += int(truncated)
                            if not truncated:
                                wins[0 if rewards[0] > 0 else 1 if rewards[1] > 0 else 2] += 1
                            state["done"] = True
            self.samples.extend(new_samples)
            self.samples = self.samples[-self.config.replay.capacity:]
        finally:
            if base is not None:
                base.close()
            for state in states:
                state["game"].close()
        stats.wins = tuple(wins)
        stats.seconds = time.perf_counter() - started
        stats.extra = dict(samples=len(new_samples), truncation_score=0.5,
                           start_store_games=len(self.starts), synthetic_smoke=self.params.bootstrap_tiny)
        if az_stats is not None:
            stats.extra.update(init="joint", alphazero_games=az_stats.games, proxy_games=stats.games,
                               alphazero_samples=len(self.az_samples), calibration_iteration=self.calibration_iteration,
                               smoke=self.params.calibration_tiny, collection_order=["alphazero", "proxy"])
            stats.games += az_stats.games
            stats.plies += az_stats.plies
            stats.truncated += az_stats.truncated
            stats.wins = tuple(a + b for a, b in zip(stats.wins, az_stats.wins))
        return stats

    def update(self, iteration):
        if self.params.init == "joint":
            return self._update_joint(iteration)
        start = time.perf_counter()
        if not self.samples:
            return UpdateStats()
        losses = []
        self.network.train()
        try:
            for _ in range(self.config.optimizer.updates_per_iteration):
                indices = self.rng.integers(len(self.samples), size=self.config.replay.batch_size)
                selected = [self.samples[i] for i in indices]
                def tensor(name, dtype=torch.float32):
                    return torch.as_tensor(np.stack([row[name] for row in selected]), device=self.device, dtype=dtype)
                planes, mask = tensor("planes"), tensor("mask")
                with torch.no_grad():
                    teacher_logits, teacher_value = self.engine.network(planes)
                    teacher_logp = masked_log_softmax(teacher_logits.float(), mask).masked_fill(mask <= 0, 0)
                    teacher_p = teacher_logp.exp().masked_fill(mask <= 0, 0)
                with self.autocast():
                    logits, value, aux = self.network.forward_aux(planes)
                own, other = tensor("own", torch.long), tensor("other", torch.long)
                practical_loss = masked_practical_loss(aux["practical"].float(), own, other, tensor("target"))
                rows = torch.arange(len(selected), device=self.device)
                slip_loss = F.binary_cross_entropy_with_logits(aux["slip"][rows, own].float(), tensor("slip"))
                logp = masked_log_softmax(logits.float(), mask).masked_fill(mask <= 0, 0)
                distill = (teacher_p * (teacher_logp - logp)).sum(1).mean() + F.mse_loss(value.float(), teacher_value)
                loss = practical_loss + self.params.slip_weight * slip_loss + self.params.distill_weight * distill
                losses.append(self.clip_and_step(loss))
        finally:
            self.network.eval()
        return UpdateStats(steps=len(losses), loss=float(np.mean(losses)) if losses else None,
                           seconds=time.perf_counter() - start)

    def make_agent(self, mode="eval"):
        if self.params.init == "joint" and mode in ("eval", "engine", "selfplay"):
            from ..inference import MctsAgent
            return MctsAgent(self.network, self.encoder, self.device, simulations=self.params.simulations,
                             rng=self.rng, amp=self.config.device.amp, spec=self.spec,
                             mcts_params={"truncation_value": self.config.selfplay.truncation_value})
        return PracticalAgent(self.advisor, self.params.default_level)

    def inference_hint(self):
        if self.params.init == "joint":
            return dict(mode="practical", levels=self.k, engine_checkpoint="engine", engine_sha256="0" * 64)
        checkpoint_parent = self.run_dir / "checkpoints" / "iter_placeholder"
        return dict(mode="practical", levels=self.levels["k"],
                    engine_checkpoint=os.path.relpath(self.engine_path, checkpoint_parent.resolve()), engine_sha256=self.engine_hash)

    def on_checkpoint(self, directory):
        import json
        if self.params.init == "joint":
            if self.levels is None:
                self._recalibrate(0)
            manifest = json.loads((directory / "manifest.json").read_text())
            # The existing loader requires a nonrecursive engine dependency. This is an
            # inference view of the very same joint weights, not a second trained model.
            engine_dir = directory / "engine"
            engine_dir.mkdir()
            shutil.copyfile(directory / "weights.pt", engine_dir / "weights.pt")
            engine_manifest = copy.deepcopy(manifest)
            engine_manifest["inference_hint"] = dict(mode="mcts", simulations=self.params.simulations)
            atomic_write_json(engine_dir / "manifest.json", engine_manifest)
            manifest["inference_hint"]["engine_sha256"] = manifest["weights_sha256"]
            atomic_write_json(directory / "manifest.json", manifest)
            levels = copy.deepcopy(self.levels)
            levels.update(engine_checkpoint="engine", engine_sha256=manifest["weights_sha256"])
            atomic_write_json(directory / "levels.json", validate_levels(levels))
            return
        levels = copy.deepcopy(self.levels)
        for row in levels["levels"]:
            entry = row["styles"].get("checkpoint")
            if entry and entry["available"]:
                source = (self.advisor.checkpoint_root / entry["params"]["checkpoint"]).resolve()
                digest = sha256_file(source / "weights.pt")
                relative = Path("proxy_engines") / digest
                target = directory / relative
                if not target.exists():
                    target.mkdir(parents=True)
                    for name in ("manifest.json", "weights.pt"):
                        shutil.copyfile(source / name, target / name)
                entry["params"]["checkpoint"] = relative.as_posix()
        reference = os.path.relpath(self.engine_path, directory.resolve())
        levels["engine_checkpoint"] = reference
        validate_levels(levels)
        atomic_write_json(directory / "levels.json", levels)
        path = directory / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["inference_hint"]["engine_checkpoint"] = reference
        atomic_write_json(path, manifest)

    def extra_state(self):
        extra = {"samples": [{key: torch.from_numpy(value.copy()) if isinstance(value, np.ndarray) else value
                             for key, value in row.items()} for row in self.samples],
                "engine_sha256": getattr(self, "engine_hash", None)}
        if self.params.init == "joint":
            extra.update(init="joint", levels=self.levels, calibration_iteration=self.calibration_iteration,
                         az_samples=[{key: torch.from_numpy(value.copy()) if isinstance(value, np.ndarray) else value
                                      for key, value in row.items()} for row in self.az_samples])
        return extra

    def load_extra_state(self, extra):
        if self.params.init == "joint":
            if extra.get("init") != "joint":
                raise ValueError("joint resume requires joint learner state")
            self.levels = validate_levels(extra["levels"]) if extra.get("levels") else None
            self.calibration_iteration = extra["calibration_iteration"]
            self.az_samples = [{key: value.cpu().numpy().copy() if isinstance(value, torch.Tensor) else value
                                for key, value in row.items()} for row in extra["az_samples"]]
            self.samples = [{key: value.cpu().numpy().copy() if isinstance(value, torch.Tensor) else value
                             for key, value in row.items()} for row in extra.get("samples", [])]
            if self.levels is not None:
                self._joint_advisor()
            return
        if extra.get("engine_sha256", self.engine_hash) != self.engine_hash:
            raise ValueError("resume engine differs from checkpoint engine")
        self.samples = [{key: value.cpu().numpy().copy() if isinstance(value, torch.Tensor) else value
                         for key, value in row.items()} for row in extra.get("samples", [])]

    def _update_joint(self, iteration):
        start = time.perf_counter()
        losses, terms_by_step = [], []
        if self.samples and len(self.az_samples) >= max(1, self.config.replay.min_size):
            self.network.train()
            try:
                for _ in range(self.config.optimizer.updates_per_iteration):
                    def batch(samples):
                        selected = [samples[i] for i in self.rng.integers(len(samples), size=self.config.replay.batch_size)]
                        return {key: torch.as_tensor(np.stack([row[key] for row in selected]), device=self.device,
                            dtype=torch.long if key in ("own", "other") else torch.float32) for key in selected[0]}
                    az, proxy = batch(self.az_samples), batch(self.samples)
                    with self.autocast():
                        logits, value = self.network(az["planes"])
                        _, _, aux = self.network.forward_aux(proxy["planes"])
                    loss, terms = joint_loss(logits, value, az, aux["practical"], aux["slip"], proxy,
                                             slip_weight=self.params.slip_weight)
                    losses.append(self.clip_and_step(loss))
                    terms_by_step.append({key: float(term.detach()) for key, term in terms.items()})
            finally:
                self.network.eval()
        if iteration % self.params.recalibrate_every == 0:
            self._recalibrate(iteration)
        means = {key: float(np.mean([row[key] for row in terms_by_step]))
                 for key in ("policy", "value", "practical", "slip")} if terms_by_step else {}
        return UpdateStats(steps=len(losses), loss=float(np.mean(losses)) if losses else None,
            policy_loss=means.get("policy"), value_loss=means.get("value"), seconds=time.perf_counter() - start,
            extra=dict(practical_loss=means.get("practical"), slip_loss=means.get("slip"), init="joint",
                       calibration_iteration=self.calibration_iteration, evaluation_role="joint_engine_mcts",
                       smoke=self.params.calibration_tiny))
