"""Shared N-seat PPO, duplicate-deal PFSP and isolated archetype exploiters.

games_per_iteration counts tables; duplicate mode requires complete N-table
groups. Each group samples N-1 opponents once and rotates the entire assignment
over the same seed. Only the on-policy learner's decisions enter PPO.

Collection follows docs/METHOD_ja.md#methods section 3: opponent
assignments come from a dedicated stream and the pool statistics as of the start
of the collection; results are applied afterwards in canonical table order;
every stochastic decision has its own stream. Batched and reference collection
therefore produce identical data.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from hasami.agents import Decision, SearchLimits, check_cancel
from ..auxiliary import auxiliary_loss, trajectory_targets
from ..batched_selfplay import (BatchEvaluator, CollectionTiming, PolicySeat, SeatActor, SeatAgent, TableRequest,
                                agent_seed, decision_rng, opponent_rng, run_table_games, stream_prefix)
from ..config import ConfigError, OpponentsConfig
from ..devices import resolve_inference_batch
from ..games import action_encoding, default_rules
from ..inference import PolicyAgent, QAgent
from ..manifest import ModelManifest, save_checkpoint
from ..opponents import BoundAgent, OpponentPool, PoolEntry
from ..registry import ALGORITHMS
from ..selfplay import run_table_game
from .base import CollectStats, UpdateStats
from .ppo import PpoLearner, PpoParams, _policy_log_probs


@dataclass
class PpoLeagueParams(PpoParams):
    duplicate_deals: bool = True
    hand_reward_coef: float = 0.01
    hand_reward_scale: float = 10000.0
    aux_coef: float = 0.1
    aux_weights: dict[str, float] = field(default_factory=dict)
    exploiter_every: int = 0
    exploiter_iterations: int = 1
    exploiter_games: int = 4
    archetypes: list[str] = field(default_factory=lambda: ["defensive", "aggressive", "fast"])
    collection: str = "batched"  # or "reference": same tables one after another, one row per forward
    parallel_tables: int = 0  # tables in flight per batched chunk; 0 = all tables of the collection
    skip_forced_decisions: bool = True  # opponents' single-legal-action decisions skip the network

    def __post_init__(self):
        super().__post_init__()
        if self.collection not in ("batched", "reference"):
            raise ConfigError("collection must be batched or reference")
        if type(self.parallel_tables) is not int or self.parallel_tables < 0:
            raise ConfigError("parallel_tables must be a nonnegative integer")
        if type(self.skip_forced_decisions) is not bool:
            raise ConfigError("skip_forced_decisions must be boolean")
        if type(self.duplicate_deals) is not bool:
            raise ConfigError("duplicate_deals must be boolean")
        for name in ("hand_reward_coef", "aux_coef", "hand_reward_scale"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or (name == "hand_reward_scale" and value == 0):
                raise ConfigError(f"{name} must be finite and {'positive' if name.endswith('scale') else 'nonnegative'}")
        for name in ("exploiter_every", "exploiter_iterations", "exploiter_games"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "exploiter_every" else 1):
                raise ConfigError(f"invalid {name}")
        if (not isinstance(self.archetypes, (list, tuple)) or not self.archetypes
                or any(name not in ("defensive", "aggressive", "fast") for name in self.archetypes)):
            raise ConfigError("archetypes must contain defensive, aggressive or fast")
        if (not isinstance(self.aux_weights, dict)
                or any(name not in ("tenpai", "waits", "hand", "placement", "danger")
                       or not np.isfinite(weight) or weight < 0 for name, weight in self.aux_weights.items())):
            raise ConfigError("aux_weights must map supported heads to nonnegative finite weights")


def league_gae(trajectory, values, params, truncation_value):
    """Attribute each hand payment to the last same-seat decision preceding it."""
    values = np.asarray(values, dtype=np.float32)
    rewards = np.zeros(len(trajectory), dtype=np.float32)
    advantages = np.zeros_like(rewards)
    for player in set(trajectory.players):
        indices = [i for i, seat in enumerate(trajectory.players) if seat == player]
        rewards[indices[-1]] = trajectory.value_for(indices[-1], truncation_value)
        if params.hand_reward_coef:
            for ply, info in enumerate(trajectory.step_info):
                if "hand" not in info or "score_delta" not in info:
                    continue
                preceding = [i for i in indices if trajectory.extras[i]["decision_ply"] <= ply]
                if preceding:
                    rewards[preceding[-1]] += (params.hand_reward_coef * info["score_delta"][player]
                                              / params.hand_reward_scale)
        next_value = advantage = 0.0
        for i in reversed(indices):
            delta = rewards[i] + params.gamma * next_value - values[i]
            advantage = delta + params.gamma * params.gae_lambda * advantage
            advantages[i] = advantage
            next_value = values[i]
    return advantages, advantages + values


class ArchetypeAgent:
    """Public-information opponents; generic games use frozen policy temperatures.

    Riichi fast minimizes shanten; aggressive additionally takes legal calls;
    defensive prefers tiles already discarded by threatening riichi opponents.
    These are training heuristics, not claims of expert playing strength.
    """

    def __init__(self, name, spec, fallback):
        self.name, self.spec, self.fallback = name, spec, fallback

    def choose(self, game, limits, cancel=None, *, player=None):
        check_cancel(cancel)
        if self.spec.game_id != "riichi":
            return self.fallback.choose(game, limits, cancel, player=player)
        from riichi.actions import PASS, RON, TSUMO
        from riichi.agari import Meld
        from riichi.shanten import shanten
        obs = game.observation(player)
        legal = list(obs["legal_actions"])
        if not legal:
            raise ValueError("no legal actions")
        for action in (TSUMO, RON):
            if action in legal:
                return Decision(action)
        if PASS in legal:
            calls = [a for a in legal if 68 <= a <= 106]
            return Decision(calls[0] if self.name == "aggressive" and calls else PASS)
        melds = tuple(Meld(tuple(t // 4 for t in m["tiles"]), m["kind"], m["open"])
                      for m in obs["melds"][0])
        candidates = []
        threats = [row for row in range(1, 4) if obs["riichi"][row]]
        for action in legal:
            if action >= 68:
                continue
            counts = list(obs["hand_counts"])
            counts[action % 34] -= 1
            unsafe = sum(action % 34 not in {d["tile"] // 4 for d in obs["discards"][row]}
                         for row in threats) if self.name == "defensive" else 0
            candidates.append((unsafe, shanten(counts, melds), action < 34, action))
        return Decision(int(min(candidates)[-1] if candidates else legal[0]))


class LearnerSeat:
    """The PPO learner's decision rule (``PpoLearner._actor``) as a recording batch seat.

    Every decision is evaluated, forced ones included: their values enter GAE and their planes the auxiliary
    losses. Sampling uses the decision's own stream instead of the learner RNG.
    """

    records = True

    def __init__(self, evaluator: BatchEvaluator, encoder, temperature_for, rng_for) -> None:
        self.evaluator, self.encoder = evaluator, encoder
        self.temperature_for, self.rng_for = temperature_for, rng_for

    def needs_evaluation(self, request) -> bool:
        return True

    def decide(self, request, logits, value):
        mask = request.mask
        if not np.any(mask > 0):
            raise ValueError("PPO cannot act without legal actions")
        temperature = self.temperature_for(request.ply)
        logp = _policy_log_probs(torch.from_numpy(np.asarray(logits, dtype=np.float32)[None]),
                                 torch.as_tensor(mask).unsqueeze(0),
                                 torch.tensor([temperature], dtype=torch.float32))[0].numpy()
        legal = np.flatnonzero(mask > 0)
        probs = np.exp(logp[legal].astype(np.float64))
        rng = self.rng_for(request.table, request.player, request.ply)
        action = int(rng.choice(legal, p=probs / probs.sum()))
        return action, None, {"logp": float(logp[action]), "value": float(value), "temperature": float(temperature)}


@dataclass(frozen=True)
class _TablePlan:
    table: int
    group: int
    rotation: int
    seed: int
    learner_seat: int
    entries: tuple  # opponent entries in offset order (seat learner_seat + 1 + offset)


@ALGORITHMS.register("ppo-league")
class PpoLeagueLearner(PpoLearner):
    name = "ppo-league"

    def params_dataclass(self):
        return PpoLeagueParams

    def __init__(self, ctx):
        super().__init__(ctx)
        self._exploiters = 0
        self._namespace = 0  # stream namespace: 0 for the main learner, unique per exploiter
        if self.config.selfplay.symmetry_augmentation and self.spec.symmetries():
            raise ConfigError("ppo-league requires symmetry_augmentation=false: belief targets need explicit transforms")
        if (self.params.duplicate_deals
                and self.config.selfplay.games_per_iteration % self.spec.num_players):
            raise ConfigError("duplicate deals require games_per_iteration divisible by num_players")
        if (self.params.exploiter_every and self.params.duplicate_deals
                and self.params.exploiter_games % self.spec.num_players):
            raise ConfigError("duplicate deals require exploiter_games divisible by num_players")
        supported = {"tenpai", "waits", "hand", "placement", "danger"}
        heads = getattr(self.network, "aux_heads", {})
        if set(heads) - supported:
            raise ConfigError("ppo-league has no supervision for the configured auxiliary head")

    def _clear_rollout(self):
        super()._clear_rollout()
        self._aux_targets = []

    def ingest_trajectories(self, trajectories):
        added = 0
        for trajectory in trajectories:
            if not len(trajectory):
                continue
            targets = trajectory_targets(trajectory)
            if len(targets) != len(trajectory):
                raise ValueError("league trajectories require metadata for every decision")
            if any("logp" not in e or "value" not in e or "temperature" not in e
                   for e in trajectory.extras):
                raise ValueError("PPO league requires on-policy rollout metadata")
            values = [extra["value"] for extra in trajectory.extras]
            advantages, returns = league_gae(trajectory, values, self.params,
                                             self.config.selfplay.truncation_value)
            for i, action in enumerate(trajectory.actions):
                mask = np.array(trajectory.masks[i], copy=True)
                if type(action) is not int or not 0 <= action < len(mask) or mask[action] <= 0:
                    raise ValueError("illegal PPO league action")
                extra = trajectory.extras[i]
                self._planes.append(np.array(trajectory.planes[i], dtype=np.float32, copy=True))
                self._masks.append(mask)
                self._actions.append(action)
                self._logp_old.append(float(extra["logp"]))
                self._value_old.append(float(values[i]))
                self._temperatures.append(float(extra["temperature"]))
                self._advantages.append(float(advantages[i]))
                self._returns.append(float(returns[i]))
                self._aux_targets.append(targets[i])
                added += 1
        return added

    def _plan(self, iteration, prefix):
        """Every table's seed, learner seat and opponents, drawn before any game from the opponent stream."""
        cfg = self.config.selfplay
        n = self.spec.num_players
        rotations = n if self.params.duplicate_deals else 1
        stream = opponent_rng(prefix)
        plans = []
        for group in range(cfg.games_per_iteration // rotations):
            entries = tuple(self.opponents.sample(stream) for _ in range(n - 1))
            seats_to_train = range(n) if self.params.duplicate_deals else [int(stream.integers(n))]
            seed = self.config.run.seed + iteration * cfg.games_per_iteration + group
            for index, seat in enumerate(seats_to_train):
                plans.append(_TablePlan(group * rotations + index, group, index, seed, seat, entries))
        return plans

    def _seats(self, plan, prefix, evaluator_for, learner_seat):
        """Fresh seats of one table; stochastic agents get per-(table, seat) instances or decision streams."""
        n = self.spec.num_players
        rng_for = lambda table, seat, ply: decision_rng(prefix, table, seat, ply)
        seats = {plan.learner_seat: learner_seat}
        for offset, entry in enumerate(plan.entries):
            seat = (plan.learner_seat + offset + 1) % n
            if entry.kind == "classical":
                agent, limits = self.spec.baseline_agent(entry.ref, seed=agent_seed(prefix, plan.table, seat))
                seats[seat] = BoundAgent(agent, limits or SearchLimits(max_depth=1, time_ms=200, node_limit=20000))
                continue
            agent = self.opponents.agent_for(entry)
            if type(agent) is PolicyAgent:
                seats[seat] = PolicySeat.from_agent(agent, evaluator_for(agent.network), rng_for,
                                                    self.params.skip_forced_decisions)
            elif isinstance(agent, ArchetypeAgent) and self.spec.game_id != "riichi":
                fallback = agent.fallback
                seats[seat] = PolicySeat(evaluator_for(fallback.network), fallback.encoder, fallback.temperature,
                                         rng_for, self.params.skip_forced_decisions)
            elif isinstance(agent, ArchetypeAgent) or (type(agent) is QAgent and agent.epsilon == 0):
                seats[seat] = agent  # deterministic: sharing the instance cannot depend on scheduling
            else:
                raise ConfigError(f"opponent {entry.entry_id} ({type(agent).__name__}) is stochastic and cannot be "
                                  "made independent of table scheduling")
        return seats

    def collect(self, iteration):
        start = time.perf_counter()
        self._clear_rollout()
        cfg = self.config.selfplay
        n = self.spec.num_players
        stats = CollectStats(wins=tuple([0] * (n + 1)))
        wins = list(stats.wins)
        prefix = stream_prefix(self.config.run.seed, self._namespace, iteration)
        plans = self._plan(iteration, prefix)
        timing = CollectionTiming()
        batch_size = resolve_inference_batch(self.config.device.inference_batch, self.device)
        if self.device.type == "cpu" and self.config.device.fixed_batch_shape:
            # Fixed shapes alone do not fix CPU GEMM row-position rounding:
            # x86 Linear can use different arithmetic for rows 0 and 6 at B=8.
            # Both collectors and all policy seats use the same canonical row.
            batch_size = 1
        evaluators = {}

        def evaluator_for(network):
            if id(network) not in evaluators:
                evaluators[id(network)] = BatchEvaluator(network, self.device, self.config.device.amp, batch_size,
                                                         self.config.device.fixed_batch_shape, timing)
            return evaluators[id(network)]

        temperature_for = lambda ply: cfg.temperature if ply < cfg.temperature_plies else 1.0
        learner = LearnerSeat(evaluator_for(self.network), self.encoder, temperature_for,
                              lambda table, seat, ply: decision_rng(prefix, table, seat, ply))
        seats = [self._seats(plan, prefix, evaluator_for, learner) for plan in plans]
        trajectories = []
        if self.params.collection == "batched":
            chunk = self.params.parallel_tables or len(plans) or 1
            for offset in range(0, len(plans), chunk):
                requests = [TableRequest(seats[i], plans[i].seed, plans[i].table)
                            for i in range(offset, min(offset + chunk, len(plans)))]
                trajectories.extend(run_table_games(requests, self.encoder, spec=self.spec, backend=self.backend,
                                                    max_plies=cfg.max_plies, timing=timing))
        else:
            tick = time.perf_counter()
            for plan, table_seats in zip(plans, seats):
                adapted = {player: (SeatActor(seat, plan.table) if getattr(seat, "records", False)
                                    else SeatAgent(seat, plan.table, self.spec) if isinstance(seat, PolicySeat)
                                    else seat) for player, seat in table_seats.items()}
                trajectories.append(run_table_game(adapted, self.encoder, spec=self.spec, backend=self.backend,
                                                   seed=plan.seed, max_plies=cfg.max_plies))
            timing.total += time.perf_counter() - tick
            # ``run_table_game`` has no phase counters: only the shared evaluators' inference time is measured.
            timing.unmeasured = ("encode", "decide", "env", "agents")
        # Every seat's decisions, counted the same way in both modes (the reference runner has no counter).
        timing.decisions = sum(len(step) for trajectory in trajectories for step in trajectory.joint_actions)
        tables = []
        for plan, trajectory in zip(plans, trajectories):
            self.ingest_trajectories([trajectory])
            stats.games += 1
            stats.plies += trajectory.plies
            stats.truncated += int(trajectory.truncated)
            if not trajectory.truncated:
                wins[trajectory.result_index()] += 1
                score = (trajectory.rewards[plan.learner_seat] + 1.0) / 2.0
                # Applied only now, in canonical order: no assignment of this collection saw these results.
                # Repeated identities at one table get one result, not N-1 votes.
                for entry in {entry.entry_id: entry for entry in plan.entries}.values():
                    self.opponents.record_result(entry, score)
            tables.append({"table": plan.table, "group": plan.group, "rotation": plan.rotation, "seed": plan.seed,
                           "learner_seat": plan.learner_seat,
                           "opponents": [entry.entry_id for entry in plan.entries],
                           "seats": {str((plan.learner_seat + offset + 1) % n): entry.entry_id
                                     for offset, entry in enumerate(plan.entries)}})
        stats.wins = tuple(wins)
        stats.seconds = time.perf_counter() - start
        stats.extra = {"samples": len(self._actions), "tables": tables,
                       "aux_labelled": sum(bool(target) for target in self._aux_targets),
                       "pool": self.opponents.stats(), "collection": self.params.collection,
                       "inference_batch": batch_size, "fixed_batch_shape": self.config.device.fixed_batch_shape,
                       "timing": timing.to_dict()}
        return stats

    def _manifest(self, iteration):
        cfg = self.config
        return ModelManifest(
            game_id=self.spec.game_id, num_players=self.spec.num_players, board_shape=self.spec.board_shape,
            game_params=dict(cfg.game.params), ruleset_id=self.spec.ruleset_id, rules=default_rules(self.spec),
            action_size=self.spec.action_size, action_encoding=action_encoding(self.spec), algorithm=self.name,
            network={"name": cfg.network.name, "params": copy.deepcopy(cfg.network.params), "head": self.required_head},
            encoding_id=self.encoder.encoding_id, encoder_params=dict(cfg.encoder.params),
            num_planes=self.encoder.num_planes, algorithm_params=copy.deepcopy(cfg.algorithm.params),
            run_id=cfg.run.run_id, iteration=iteration, train_device=str(self.device),
            inference_hint=self.inference_hint(), notes=cfg.run.notes)

    def _train_exploiters(self, iteration):
        results = []
        for index, name in enumerate(self.params.archetypes):
            network = copy.deepcopy(self.network)
            frozen = copy.deepcopy(self.network).eval().requires_grad_(False)
            rng = np.random.default_rng(self.config.run.seed + iteration * 100003 + index)
            directory = self.run_dir / "pool" / f"exploiter_{iteration:06d}_{self._exploiters:06d}_{name}"
            cfg = copy.deepcopy(self.config)
            cfg.selfplay.games_per_iteration = self.params.exploiter_games
            cfg.algorithm.params["exploiter_every"] = 0
            cfg.opponents = OpponentsConfig(strategy="self", classical=[], classical_weight=0.0)
            fallback = PolicyAgent(frozen, self.encoder, self.device, spec=self.spec, rng=rng,
                                   temperature={"defensive": 0.0, "aggressive": 1.5, "fast": 0.5}[name])
            opponent = ArchetypeAgent(name, self.spec, fallback)
            pool = OpponentPool(cfg.opponents, self.device, self.encoder, rng, directory,
                                current_agent_fn=lambda: opponent, spec=self.spec)
            learner = type(self)(replace(self.ctx, config=cfg, network=network, rng=rng,
                                        opponents=pool, replay=None, metrics=None, run_dir=directory))
            learner._namespace = 1 + self._exploiters  # unique per exploiter over the run (persisted counter)
            steps = games = truncated = 0
            for cycle in range(self.params.exploiter_iterations):
                collected = learner.collect(iteration * self.params.exploiter_iterations + cycle)
                updated = learner.update(cycle + 1)
                steps += updated.steps
                games += collected.games
                truncated += collected.truncated
            save_checkpoint(directory, network, learner._manifest(iteration), learner.state_dict())
            entry = PoolEntry(f"pool:{directory.name}", "checkpoint", str(directory), iteration)
            state = self.opponents.state_dict()
            state["entries"].append(entry.to_dict())
            if self.opponents.cfg.pool_size:
                state["entries"] = state["entries"][-self.opponents.cfg.pool_size:]
            self.opponents.load_state_dict(state)
            self._exploiters += 1
            results.append({"archetype": name, "entry": entry.entry_id, "games": games,
                            "truncated": truncated, "steps": steps})
        return results

    def extra_state(self):
        return {"exploiters": self._exploiters}

    def load_extra_state(self, extra):
        self._exploiters = int(extra.get("exploiters", 0))

    def update(self, iteration: int) -> UpdateStats:
        start = time.perf_counter()
        count = len(self._actions)
        if not count:
            return UpdateStats(seconds=time.perf_counter() - start)
        arrays = {"planes": np.stack(self._planes).astype(np.float32),
                  "mask": np.stack(self._masks).astype(np.float32),
                  "action": np.asarray(self._actions, dtype=np.int64),
                  "logp_old": np.asarray(self._logp_old, dtype=np.float32),
                  "value_old": np.asarray(self._value_old, dtype=np.float32),
                  "advantage": np.asarray(self._advantages, dtype=np.float32),
                  "return": np.asarray(self._returns, dtype=np.float32),
                  "temperature": np.asarray(self._temperatures, dtype=np.float32)}
        if self.params.normalize_advantage and count > 1:
            advantage = arrays["advantage"]
            arrays["advantage"] = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        # Keep BatchNorm/dropout identical to infer(); eval mode still allows gradients.
        self.network.eval()
        totals = np.zeros(5, dtype=np.float64)
        seen = steps = 0
        early_stop = False
        aux_totals = {}
        for _ in range(self.params.epochs):
            order = self.rng.permutation(count)
            for index in np.array_split(order, min(count, self.params.minibatches)):
                batch = self.to_device({name: array[index] for name, array in arrays.items()})
                with self.autocast():
                    if getattr(self.network, "aux_heads", {}):
                        logits, value, raw = self.network.forward_aux(batch["planes"])
                    else:
                        logits, value = self.network(batch["planes"])
                        raw = {}
                logp = _policy_log_probs(logits.float(), batch["mask"], batch["temperature"])
                chosen = logp.gather(1, batch["action"].unsqueeze(1)).squeeze(1)
                log_ratio = chosen - batch["logp_old"]
                ratio = log_ratio.exp()
                advantage = batch["advantage"]
                surrogate = torch.minimum(ratio * advantage,
                                          ratio.clamp(1 - self.params.clip, 1 + self.params.clip) * advantage)
                policy_loss = -surrogate.mean()
                error = (value.float() - batch["return"]).square()
                if self.params.value_clip > 0:
                    clipped_value = batch["value_old"] + (value.float() - batch["value_old"]).clamp(
                        -self.params.value_clip, self.params.value_clip)
                    error = torch.maximum(error, (clipped_value - batch["return"]).square())
                value_loss = 0.5 * error.mean()
                safe_logp = logp.masked_fill(~torch.isfinite(logp), 0)
                entropy = -(logp.exp() * safe_logp).sum(dim=1).mean()
                loss = policy_loss + self.params.value_coef * value_loss - self.params.entropy_coef * entropy
                if raw and self.params.aux_coef:
                    aux, metrics = auxiliary_loss(raw, [self._aux_targets[i] for i in index], self.params.aux_weights)
                    loss = loss + self.params.aux_coef * aux
                    for name, number in metrics.items():
                        aux_totals[name] = aux_totals.get(name, 0.0) + number * len(index)
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fraction = ((ratio - 1).abs() > self.params.clip).float().mean()
                    totals += len(index) * np.array([loss.item(), policy_loss.item(), value_loss.item(),
                                                     approx_kl.item(), clip_fraction.item()])
                seen += len(index)
                if self.params.target_kl is not None and approx_kl.item() > self.params.target_kl:
                    early_stop = True
                    break
                self.clip_and_step(loss)
                steps += 1
            if early_stop:
                break
        means = (totals / seen).tolist()
        self._clear_rollout()
        stats = UpdateStats(steps=steps, loss=means[0], policy_loss=means[1], value_loss=means[2],
                           seconds=time.perf_counter() - start,
                           extra={"samples": count, "policy_loss": means[1], "value_loss": means[2],
                                  "approx_kl": means[3], "clip_fraction": means[4], "early_stop": early_stop,
                                  "aux_loss": {name: value / seen for name, value in aux_totals.items()}})
        if self.params.exploiter_every and iteration % self.params.exploiter_every == 0:
            stats.extra["exploiters"] = self._train_exploiters(iteration)
        stats.seconds = time.perf_counter() - start
        return stats
