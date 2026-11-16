"""AlphaZero self-play learner using batched PUCT and policy/value replay."""

from __future__ import annotations

from ..games.base import as_v2, require_search_game

import math
import time
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np
import torch

from ..datastore import SelfplayStore
from ..parallel import SelfplayWorkers, selfplay_batches
from ..encoding import masked_log_softmax
from ..registry import ALGORITHMS
from ..selfplay import (Trajectory, augment, draw_value_for_iteration, search_decision,
                        current_draw_value, current_draw_reasons)
from ..search.draw import SEARCH_SEMANTICS_VERSION, draw_reasons_for
from ..config import ConfigError
from .base import CollectStats, Learner, UpdateStats
from ..search.mcts import MCTS, MctsConfig, MctsResult, _integer_budget, batched_evaluator


@dataclass
class AlphaZeroParams:
    simulations: int = 64
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature_plies: int = 20
    batch_games: int = 8
    eval_simulations: int | None = None
    value_loss_weight: float = 1.0
    search_backend: str = "python"
    search_threads: int | None = None
    root_selection: str = "puct"
    gumbel_m: int = 16
    gumbel_c_visit: float = 50.0
    gumbel_c_scale: float = 1.0
    value_target_mix: float = 0.0

    def __post_init__(self) -> None:
        if self.search_backend not in ("python", "cpp"):
            raise ValueError("search_backend must be 'python' or 'cpp'")
        if self.search_threads is not None:
            _integer_budget(self.search_threads, "search_threads", minimum=1)
        _integer_budget(self.simulations, "simulations", minimum=1)
        _integer_budget(self.batch_games, "batch_games", minimum=1)
        _integer_budget(self.temperature_plies, "temperature_plies")
        if self.eval_simulations is None:
            self.eval_simulations = self.simulations
        _integer_budget(self.eval_simulations, "eval_simulations", minimum=1)
        if not math.isfinite(self.value_loss_weight) or self.value_loss_weight < 0:
            raise ValueError("value_loss_weight must be finite and nonnegative")
        if not math.isfinite(self.value_target_mix) or not 0 <= self.value_target_mix <= 1:
            raise ValueError("value_target_mix must be finite and within [0, 1]")
        MctsConfig(c_puct=self.c_puct, dirichlet_alpha=self.dirichlet_alpha,
                   dirichlet_epsilon=self.dirichlet_epsilon, root_selection=self.root_selection,
                   gumbel_m=self.gumbel_m, gumbel_c_visit=self.gumbel_c_visit,
                   gumbel_c_scale=self.gumbel_c_scale)


@ALGORITHMS.register("alphazero")
class AlphaZeroLearner(Learner):
    name = "alphazero"
    required_head = "policy_value"
    replay_kind = "pv"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        version = self.config.search_semantics_version
        if type(version) is not int or version != SEARCH_SEMANTICS_VERSION:
            raise ConfigError("AlphaZero requires search_semantics_version=2 (symmetric terminal draw contempt)")
        require_search_game(self.spec)
        if self.replay is None:
            raise ValueError("AlphaZero requires a policy/value replay buffer")
        search_class, search_options = MCTS, {}
        if self.params.search_backend == "cpp":
            from ..search.native_mcts import NativeMCTS

            if self.backend != "cpp":
                raise ValueError("search_backend='cpp' requires run.backend='cpp' (or auto resolving to cpp)")
            search_class = NativeMCTS
            search_options["threads"] = self.params.search_threads
        self.search = search_class(batched_evaluator(self.network, self.device, self.config.device.amp),
                                   self.encoder, MctsConfig(**self._mcts_params(True)), self.rng,
                                   spec=self.spec, **search_options)

        self._iteration = 0
        cfg = self.config.datastore
        self.store = (SelfplayStore(cfg.path or self.run_dir / "selfplay", cfg.max_games, cfg.max_bytes, spec=self.spec,
                                    game_params=dict(self.config.game.params))
                      if cfg.enabled else None)
        if self.store is not None:
            self.store.truncation_value = self.config.selfplay.truncation_value

    def params_dataclass(self):
        return AlphaZeroParams

    def _mcts_params(self, add_noise: bool, iteration: int | None = None) -> dict:
        return {"simulations": self.params.simulations, "c_puct": self.params.c_puct,
                "dirichlet_alpha": self.params.dirichlet_alpha,
                "dirichlet_epsilon": self.params.dirichlet_epsilon,
                "add_noise": add_noise, "batch_size": self.params.batch_games,
                "root_selection": self.params.root_selection, "gumbel_m": self.params.gumbel_m,
                "gumbel_c_visit": self.params.gumbel_c_visit, "gumbel_c_scale": self.params.gumbel_c_scale,
                "truncation_value": self.config.selfplay.truncation_value,
                "draw_value": (0.0 if iteration is None else
                               draw_value_for_iteration(self.config.selfplay, iteration)),
                "draw_reasons": (() if iteration is None else
                                 draw_reasons_for(self.spec, self.config.selfplay.draw_reasons))}

    def _decision(self, result: MctsResult, mask: np.ndarray, ply: int) -> tuple:
        return search_decision(result, mask, ply, self.rng, root_selection=self.params.root_selection,
                               temperature=self.config.selfplay.temperature,
                               temperature_plies=self.params.temperature_plies,
                               value_target_mix=self.params.value_target_mix)

    def actor_fn(self):
        def actor(game, planes, mask, ply):
            # Adversarial collectors use this hook and stamp the trainer's current draw state.
            self.search.config = replace(self.search.config, draw_value=current_draw_value(),
                                         draw_reasons=current_draw_reasons(self.spec))
            action, policy, extra = self._decision(self.search.run(game), mask, ply)
            if self.store is not None:
                # Both backends append to this list, including subsequent opponent moves.
                # The store copies the finished history when the trajectory is ingested.
                extra.update(game_moves=game.moves, decision_ply=ply)
            return action, policy, extra

        return actor

    def ingest_trajectories(self, trajectories: list[Trajectory], iteration: int | None = None) -> int:
        cfg = self.config.selfplay
        added = 0
        for trajectory in trajectories:
            for index, extra in enumerate(trajectory.extras):
                extra["value_target_mix"] = self.params.value_target_mix
                extra["outcome_value"] = trajectory.outcome_value_for(index, cfg.truncation_value)
                extra["value_target"] = trajectory.value_for(index, cfg.truncation_value)
            if self.store is not None:
                self.store.truncation_value = cfg.truncation_value
                self.store.add(trajectory, self._iteration if iteration is None else iteration)
            values = trajectory.values(cfg.truncation_value)
            for index in range(len(trajectory)):
                planes, mask = trajectory.planes[index], trajectory.masks[index]
                policy = trajectory.policies[index]
                if policy is None:
                    raise ValueError("AlphaZero trajectories require MCTS policy targets")
                self.replay.add(planes=planes, mask=mask, policy=policy, value=values[index])
                added += 1
                if cfg.symmetry_augmentation:
                    for planes, mask, policy, _ in augment(self.spec, planes, mask, policy, None):
                        self.replay.add(planes=planes, mask=mask, policy=policy, value=values[index])
                        added += 1
        return added

    def _selfplay_job(self, iteration: int) -> dict:
        cfg = self.config.selfplay
        games, remainder = divmod(cfg.games_per_iteration, cfg.workers)
        return {
            "game_id": self.spec.game_id, "num_players": self.spec.num_players, "game_params": dict(self.config.game.params),
            "state_dict": {key: value.detach().cpu().clone() for key, value in self.network.state_dict().items()},
            "network_name": self.config.network.name, "network_params": dict(self.config.network.params),
            "head": self.required_head, "encoder_name": self.config.encoder.name,
            "encoder_params": dict(self.config.encoder.params), "mcts_params": self._mcts_params(True, iteration),
            "search_backend": self.params.search_backend, "search_threads": self.params.search_threads,
            "backend": self.backend, "max_plies": cfg.max_plies, "batch_games": self.params.batch_games,
            "seeds": [self.config.run.seed + iteration * 1000 + i for i in range(cfg.workers)],
            "games_per_worker": [games + int(i < remainder) for i in range(cfg.workers)],
            "draw_value": draw_value_for_iteration(cfg, iteration), "draw_reasons": list(cfg.draw_reasons),
            "temperature": cfg.temperature, "temperature_plies": self.params.temperature_plies,
            "value_target_mix": self.params.value_target_mix,
            "threads": self.config.device.threads, "deterministic": self.config.device.deterministic,
            "amp": self.config.device.amp,
        }

    def collect(self, iteration: int) -> CollectStats:
        start = time.perf_counter()
        self._iteration = iteration
        cfg = self.config.selfplay
        self.search.config = MctsConfig(**self._mcts_params(True, iteration))
        stats = CollectStats()
        reasons = Counter()
        wins = [0] * (self.spec.num_players + 1)
        samples = 0
        if cfg.workers > 1:
            with SelfplayWorkers(cfg.workers, str(self.device)) as workers:
                batches = [workers.run(self._selfplay_job(iteration))]
        else:
            batches = selfplay_batches(
                self.search, self.encoder, self._decision, games=cfg.games_per_iteration,
                batch_games=self.params.batch_games, spec=self.spec, backend=self.backend, max_plies=cfg.max_plies, seed=self.config.run.seed + iteration * cfg.games_per_iteration + 0,
                draw_value=draw_value_for_iteration(cfg, iteration), draw_reasons=cfg.draw_reasons)
        for trajectories in batches:
            samples += self.ingest_trajectories(trajectories)
            for trajectory in trajectories:
                stats.games += 1
                stats.plies += trajectory.plies
                stats.truncated += int(trajectory.truncated)
                reasons[trajectory.reason] += 1
                wins[trajectory.result_index()] += 1
        stats.wins = tuple(wins)
        stats.seconds = time.perf_counter() - start
        stats.extra = {"samples": samples, "terminal_reasons": dict(sorted(reasons.items()))}
        if self.store is not None:
            stats.extra["datastore"] = self.store.stats()
        return stats

    def update(self, iteration: int) -> UpdateStats:
        start = time.perf_counter()
        if len(self.replay) < max(1, self.config.replay.min_size):
            return UpdateStats(seconds=time.perf_counter() - start)
        losses, policy_losses, value_losses = [], [], []
        self.network.train()
        try:
            for _ in range(self.config.optimizer.updates_per_iteration):
                batch = self.to_device(self.replay.sample(self.config.replay.batch_size))
                with self.autocast():
                    logits, value = self.network(batch["planes"])
                logp = masked_log_softmax(logits.float(), batch["mask"])
                # Zero illegal log probabilities before multiplication, including -inf.
                logp = logp.masked_fill(batch["mask"] <= 0, 0.0)
                policy_loss = -(batch["policy"] * logp).sum(dim=1).mean()
                value_loss = torch.nn.functional.mse_loss(value.float(), batch["value"])
                loss = policy_loss + self.params.value_loss_weight * value_loss
                losses.append(self.clip_and_step(loss))
                policy_losses.append(float(policy_loss.detach().item()))
                value_losses.append(float(value_loss.detach().item()))
        finally:
            self.network.eval()
        return UpdateStats(steps=len(losses), loss=float(np.mean(losses)) if losses else None,
                           policy_loss=float(np.mean(policy_losses)) if losses else None,
                           value_loss=float(np.mean(value_losses)) if losses else None,
                           seconds=time.perf_counter() - start)

    def make_agent(self, mode: str = "eval"):
        from ..inference import MctsAgent

        params = self._mcts_params(False)
        if mode == "selfplay":
            params.update(draw_value=current_draw_value(), draw_reasons=current_draw_reasons(self.spec))
        return MctsAgent(self.network, self.encoder, self.device,
                         simulations=self.params.eval_simulations,
                         mcts_params=params, rng=self.rng,
                         amp=self.config.device.amp, spec=self.spec)

    def extra_state(self) -> dict:
        return {"search_semantics_version": SEARCH_SEMANTICS_VERSION}

    def load_state_dict(self, state: dict) -> None:
        if (state.get("extra") or {}).get("search_semantics_version") != SEARCH_SEMANTICS_VERSION:
            raise ConfigError("Cannot resume AlphaZero checkpoint with old search semantics: "
                              "version 2 applies symmetric draw contempt inside search. "
                              "Start a new run; run.init_checkpoint can reuse weights without old replay/optimizer state.")
        super().load_state_dict(state)

    def inference_hint(self) -> dict:
        # Search settings resolved from the training config (not algorithm params) travel with the
        # checkpoint so create_agent / pooled checkpoints search exactly like the learner did.
        return {"mode": "mcts", "simulations": self.params.eval_simulations,
                "mcts_params": {"truncation_value": float(self.config.selfplay.truncation_value)}}
