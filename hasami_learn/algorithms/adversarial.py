"""Legal-policy league training around a learner's actor and ingestion hooks.

Trainer builds the network and replay from class defaults before resolving the
inner algorithm. Callers supplying a LearnerContext must match the inner head
and replay kind; the current Trainer supports policy/value inners without replay.
The encoder retains its existing approximation of repetition history.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field, replace
from typing import Any

from hasami.agents import SearchLimits
from ..config import ConfigError, NamedConfig, OpponentsConfig
from ..inference import PolicyAgent, QAgent
from ..games import default_rules, action_encoding
from ..manifest import ModelManifest, save_checkpoint
from ..opponents import OpponentPool, PoolEntry
from ..registry import ALGORITHMS
from ..replay import ReplayBuffer, fields_for
from ..selfplay import Trajectory, run_vs_opponent_game
from .base import CollectStats, Learner, UpdateStats


@dataclass
class AdversarialParams:
    inner_algorithm: str = "reinforce"
    inner_params: dict[str, Any] = field(default_factory=dict)
    snapshot_every: int = 2
    exploiter_every: int = 0
    exploiter_iterations: int = 2
    exploiter_games: int = 4
    main_games_fraction: float = 1.0
    learner_colour: str = "random"
    record_pool_stats: bool = True

    def __post_init__(self) -> None:
        for name in ("snapshot_every", "exploiter_every", "exploiter_iterations", "exploiter_games"):
            value = getattr(self, name)
            minimum = 1 if name in ("exploiter_iterations", "exploiter_games") else 0
            if type(value) is not int or value < minimum:
                raise ConfigError(f"adversarial: {name} must be an integer >= {minimum}")
        if not isinstance(self.inner_algorithm, str) or not isinstance(self.inner_params, dict):
            raise ConfigError("adversarial: inner_algorithm must be a name and inner_params a dict")
        if self.learner_colour not in ("random", "black", "white"):
            raise ConfigError("adversarial: learner_colour must be random, black, or white")
        if (type(self.main_games_fraction) not in (int, float)
                or not 0 < self.main_games_fraction <= 1):
            raise ConfigError("adversarial: main_games_fraction must be in (0, 1]")
        if type(self.record_pool_stats) is not bool:
            raise ConfigError("adversarial: record_pool_stats must be a boolean")


@ALGORITHMS.register("adversarial")
class AdversarialLearner(Learner):
    name = "adversarial"
    required_head = "policy_value"
    replay_kind = None

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        try:
            inner_cls = ALGORITHMS.get(self.params.inner_algorithm)
        except KeyError as exc:
            raise ConfigError(f"adversarial: {exc}") from exc
        self.required_head = inner_cls.required_head
        self.replay_kind = inner_cls.replay_kind
        self._validate_resources()
        config = replace(copy.deepcopy(self.config), algorithm=NamedConfig(
            self.params.inner_algorithm, copy.deepcopy(self.params.inner_params)))
        self.inner = inner_cls(replace(ctx, config=config))
        self.required_head = self.inner.required_head
        self.replay_kind = self.inner.replay_kind
        self._validate_resources()
        for hook in ("actor_fn", "ingest_trajectories"):
            if getattr(type(self.inner), hook) is getattr(Learner, hook):
                raise ConfigError(f"adversarial: {self.inner.name} must implement {hook}()")
        self._snapshots = 0
        self._exploiters = 0
        self._games = 0
        self._pool_truncated: dict[str, int] = {}

    def _validate_resources(self) -> None:
        if self.network.head != self.required_head:
            raise ConfigError(f"adversarial: network head {self.network.head!r} must match "
                              f"inner head {self.required_head!r}; Trainer uses class defaults")
        if self.replay_kind is not None:
            expected = fields_for(self.spec, self.encoder.num_planes, self.replay_kind)
            if self.replay is None or self.replay.fields != expected:
                raise ConfigError("adversarial: supply a replay buffer matching the inner learner; "
                                  "Trainer uses the wrapper's replay_kind=None class default")

    def params_dataclass(self):
        return AdversarialParams

    @property
    def optimizer(self):
        return self.inner.optimizer

    def actor_fn(self):
        return self.inner.actor_fn()

    def ingest_trajectories(self, trajectories: list) -> int:
        return self.inner.ingest_trajectories(trajectories)

    def _colour(self, rng) -> int:
        if self.params.learner_colour == "random":
            return int(rng.integers(self.spec.num_players))
        return 0 if self.params.learner_colour == "black" else 1

    @staticmethod
    def _score(trajectory: Trajectory, player: int) -> float | None:
        if trajectory.truncated:
            return None
        if trajectory.winner == -2:
            raise ValueError("league runner returned a nonterminal, nontruncated trajectory")
        return (trajectory.rewards[player] + 1.0) / 2.0

    def _play(self, learner, opponent, player: int, seed: int | None = None) -> Trajectory:
        return run_vs_opponent_game(
            learner.actor_fn(), opponent, player, self.encoder,
            spec=self.spec, backend=self.backend, max_plies=learner.config.selfplay.max_plies,
            limits=SearchLimits(max_depth=1, time_ms=200, node_limit=20000), seed=seed,
        )

    def collect(self, iteration: int) -> CollectStats:
        """Play the whole budget; ingest ceil(fraction * games) for the main learner.

        Remaining games only measure pool results. Truncations train using the
        inner learner's explicit target but never update PFSP win/draw/loss counts.
        """
        start = time.perf_counter()
        cfg = self.config.selfplay
        stats = CollectStats()
        wins, learner_wins = [0] * (self.spec.num_players + 1), [0, 0, 0]
        trajectories = []
        training_games = math.ceil(cfg.games_per_iteration * self.params.main_games_fraction)
        for index in range(cfg.games_per_iteration):
            entry = self.opponents.sample()
            player = self._colour(self.rng)
            # Collection uses the wrapper's current self-play budget.
            trajectory = run_vs_opponent_game(
                self.actor_fn(), self.opponents.agent_for(entry), player, self.encoder,
                spec=self.spec, backend=self.backend, max_plies=cfg.max_plies, seed=self.config.run.seed + iteration * cfg.games_per_iteration + index,
                limits=SearchLimits(max_depth=1, time_ms=200, node_limit=20000),
            )
            score = self._score(trajectory, player)
            if score is None:
                self._pool_truncated[entry.entry_id] = self._pool_truncated.get(entry.entry_id, 0) + 1
            else:
                self.opponents.record_result(entry, score)
                wins[trajectory.result_index()] += 1
                learner_wins[0 if score == 1 else 1 if score == 0 else 2] += 1
            if index < training_games:
                trajectories.append(trajectory)
            stats.games += 1
            stats.plies += trajectory.plies
            stats.truncated += int(trajectory.truncated)
        samples = self.ingest_trajectories(trajectories)
        self._games += stats.games
        stats.wins = tuple(wins)
        stats.seconds = time.perf_counter() - start
        stats.extra = {"samples": samples, "learner_wins": learner_wins,
                       "training_games": len(trajectories), "completed_games": sum(learner_wins)}
        if self.params.record_pool_stats:
            stats.extra.update(pool=self.opponents.stats(), pool_truncated=dict(self._pool_truncated))
        return stats

    def _manifest(self, learner, iteration: int) -> ModelManifest:
        cfg = learner.config
        return ModelManifest(
            game_id=self.spec.game_id, num_players=self.spec.num_players, board_shape=self.spec.board_shape,
            game_params=dict(cfg.game.params), ruleset_id=self.spec.ruleset_id, rules=default_rules(self.spec),
            action_size=self.spec.action_size, action_encoding=action_encoding(self.spec),
            algorithm=learner.name,
            network={"name": cfg.network.name, "params": copy.deepcopy(cfg.network.params),
                     "head": learner.required_head},
            encoding_id=self.encoder.encoding_id, encoder_params=dict(cfg.encoder.params), num_planes=self.encoder.num_planes,
            algorithm_params=copy.deepcopy(cfg.algorithm.params), run_id=cfg.run.run_id,
            iteration=int(iteration), train_device=str(self.device),
            inference_hint=learner.inference_hint(), notes=cfg.run.notes,
        )

    def _train_exploiter(self, iteration: int) -> dict[str, Any]:
        # Neither network buffers, optimizer state, replay, nor numpy RNG are shared.
        network = copy.deepcopy(self.network)
        frozen = copy.deepcopy(self.network).eval().requires_grad_(False)
        rng = copy.deepcopy(self.rng)
        agent_cls = QAgent if self.required_head == "q" else PolicyAgent
        opponent = agent_cls(frozen, self.encoder, self.device, rng=rng,
                             amp=self.config.device.amp, spec=self.spec)
        directory = self.run_dir / "pool" / f"exploiter_{iteration:06d}_{self._exploiters + 1:06d}"
        config = copy.deepcopy(self.inner.config)
        config.selfplay.games_per_iteration = self.params.exploiter_games
        config.opponents = OpponentsConfig(strategy="self", classical=[], classical_weight=0.0)
        pool = OpponentPool(config.opponents, self.device, self.encoder, rng, directory, game_params=dict(config.game.params),
                            current_agent_fn=lambda: opponent, spec=self.spec)
        replay = (ReplayBuffer(config.replay.capacity, fields_for(self.spec, self.encoder.num_planes, self.replay_kind), rng)
                  if self.replay_kind else None)
        exploiter = type(self.inner)(replace(
            self.inner.ctx, config=config, network=network, replay=replay, opponents=pool,
            rng=rng, metrics=None, run_dir=directory,
        ))
        wins = [0, 0, 0]
        truncated = 0
        exploiter_base = self.config.run.seed * 1_000_003 + iteration * 100_003  # deterministic per iteration
        for cycle in range(1, self.params.exploiter_iterations + 1):
            trajectories = []
            for game_index in range(self.params.exploiter_games):
                player = self._colour(rng)
                trajectory = self._play(exploiter, opponent, player, seed=exploiter_base + cycle * 1009 + game_index)
                score = self._score(trajectory, player)
                if score is None:
                    truncated += 1
                else:
                    wins[0 if score == 1 else 1 if score == 0 else 2] += 1
                    pool.record_result(pool.current, score)
                trajectories.append(trajectory)
            # Use league hooks: an inner collect() may otherwise play both sides.
            exploiter.ingest_trajectories(trajectories)
            exploiter.update(cycle)
        save_checkpoint(directory, network, self._manifest(exploiter, iteration))
        entry = PoolEntry(f"pool:{directory.name}", "checkpoint", str(directory), iteration)
        state = self.opponents.state_dict()
        state["entries"].append(entry.to_dict())
        if self.opponents.cfg.pool_size:
            state["entries"] = state["entries"][-self.opponents.cfg.pool_size:]
        self.opponents.load_state_dict(state)
        self._exploiters += 1
        completed = sum(wins)
        return {"exploiter_win_rate": wins[0] / completed if completed else None,
                "exploiter_wins": wins, "exploiter_truncated": truncated,
                "exploiter_games": completed + truncated, "exploiter_entry": entry.entry_id}

    def update(self, iteration: int) -> UpdateStats:
        start = time.perf_counter()
        stats = self.inner.update(iteration)
        if self.params.exploiter_every and iteration % self.params.exploiter_every == 0:
            stats.extra.update(self._train_exploiter(iteration))
        if (not self.config.opponents.snapshot_every and self.params.snapshot_every
                and iteration % self.params.snapshot_every == 0
                and not any(entry.entry_id == f"pool:iter_{iteration:06d}"
                            for entry in self.opponents.entries)):
            # Nested wrappers share a pool and must not overwrite its snapshot.
            entry = self.opponents.add_snapshot(self.network, self._manifest(self.inner, iteration), iteration)
            self._snapshots += 1
            stats.extra["snapshot"] = entry.entry_id
        if self.params.record_pool_stats:
            stats.extra["pool"] = self.opponents.stats()
        stats.seconds = time.perf_counter() - start
        return stats

    def make_agent(self, mode: str = "eval"):
        return self.inner.make_agent(mode)

    def inference_hint(self) -> dict[str, Any]:
        return self.inner.inference_hint()

    def extra_state(self) -> dict[str, Any]:
        return {"inner": self.inner.extra_state(), "snapshots": self._snapshots,
                "exploiters": self._exploiters, "games": self._games,
                "pool_truncated": dict(self._pool_truncated)}

    def load_extra_state(self, extra: dict[str, Any]) -> None:
        self.inner.load_extra_state(extra.get("inner", {}))
        self._snapshots = int(extra.get("snapshots", 0))
        self._exploiters = int(extra.get("exploiters", 0))
        self._games = int(extra.get("games", 0))
        self._pool_truncated = dict(extra.get("pool_truncated", {}))

    def state_dict(self) -> dict[str, Any]:
        return {"inner": self.inner.state_dict(),
                "adversarial": {key: value for key, value in self.extra_state().items() if key != "inner"}}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        counters = state.get("adversarial", {})
        self.inner.load_state_dict(state.get("inner", {}))
        self.load_extra_state({**counters, "inner": self.inner.extra_state()})

    def on_checkpoint(self, directory) -> None:
        self.inner.on_checkpoint(directory)
