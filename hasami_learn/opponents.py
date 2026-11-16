"""Opponent pool for adversarial / league style training."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from hasami.agents import Agent, Decision, SearchLimits
from .config import OpponentsConfig
from .games import Encoder
from .games import default_rules, get_spec
from .manifest import ModelManifest, save_checkpoint
from .registry import OPPONENT_STRATEGIES


@dataclass
class PoolEntry:
    entry_id: str
    kind: str  # classical | checkpoint | current
    ref: str
    added_iteration: int = 0
    wins: int = 0  # from the learner's perspective
    losses: int = 0
    draws: int = 0
    points: float | None = None

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def learner_win_rate(self) -> float:
        if self.games == 0:
            return 0.5
        return (self.points if self.points is not None else self.wins + 0.5 * self.draws) / self.games

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PoolEntry":
        return cls(**data)


class BoundAgent:
    """Wrap an agent with fixed search limits (used for classical baselines)."""

    def __init__(self, agent: Agent, limits: SearchLimits) -> None:
        self.agent = agent
        self.limits = limits

    def choose(self, game, limits: SearchLimits, cancel: threading.Event | None = None, player=None) -> Decision:
        from .selfplay import choose_agent
        return choose_agent(self.agent, game, self.limits, cancel, player)


class OpponentPool:
    def __init__(
        self,
        cfg: OpponentsConfig,
        device: torch.device,
        encoder: Encoder,
        rng: np.random.Generator,
        run_dir: Path,
        current_agent_fn: Callable[[], Agent] | None = None,
        seed: int = 0,
        spec=None,
        game_params: dict | None = None,
    ) -> None:
        self.spec = spec or get_spec()
        self.game_params = dict(game_params or {})  # variant identity every pooled checkpoint must match
        self.cfg = cfg
        self.device = device
        self.encoder = encoder
        self.rng = rng
        self.run_dir = Path(run_dir)
        self.seed = seed
        self._current_fn = current_agent_fn
        self.current = PoolEntry("current", "current", "current")
        self.classical: list[PoolEntry] = [PoolEntry(f"classical:{name}", "classical", name) for name in (cfg.classical if cfg.classical is not None else getattr(self.spec, "default_baselines", ("random",)))]
        self.entries: list[PoolEntry] = []  # checkpoint snapshots, oldest first
        self._agent_cache: dict[str, Agent] = {}

    # ------------------------------------------------------------ management
    def set_current(self, fn: Callable[[], Agent]) -> None:
        self._current_fn = fn

    def add_snapshot(self, network: torch.nn.Module, manifest: ModelManifest, iteration: int) -> PoolEntry:
        directory = self.run_dir / "pool" / f"iter_{iteration:06d}"
        save_checkpoint(directory, network, manifest)
        entry = PoolEntry(f"pool:iter_{iteration:06d}", "checkpoint", str(directory), added_iteration=iteration)
        self.entries.append(entry)
        while self.cfg.pool_size and len(self.entries) > self.cfg.pool_size:
            dropped = self.entries.pop(0)
            self._agent_cache.pop(dropped.entry_id, None)
        return entry

    def all_entries(self) -> list[PoolEntry]:
        return [self.current, *self.classical, *self.entries]

    def find(self, entry_id: str) -> PoolEntry:
        for entry in self.all_entries():
            if entry.entry_id == entry_id:
                return entry
        raise KeyError(f"unknown pool entry {entry_id}")

    # ------------------------------------------------------------ sampling
    def sample(self, rng: np.random.Generator | None = None) -> PoolEntry:
        """One opponent entry from the current statistics; ``rng`` overrides the pool's own stream."""
        rng = self.rng if rng is None else rng
        if self.classical and self.cfg.classical_weight > 0 and rng.random() < self.cfg.classical_weight:
            return self.classical[int(rng.integers(len(self.classical)))]
        strategy = OPPONENT_STRATEGIES.get(self.cfg.strategy)
        return strategy(self, rng)

    def agent_for(self, entry: PoolEntry, limits: SearchLimits | None = None) -> Agent:
        if entry.kind == "current":
            if self._current_fn is None:
                raise RuntimeError("OpponentPool has no current agent provider")
            return self._current_fn()
        cached = self._agent_cache.get(entry.entry_id)
        if cached is not None:
            return cached
        if entry.kind == "classical":
            agent, base_limits = self.spec.baseline_agent(entry.ref, seed=self.seed + len(self._agent_cache))
            agent = BoundAgent(agent, limits or base_limits or SearchLimits(max_depth=1, time_ms=200, node_limit=20000))
        elif entry.kind == "checkpoint":
            from .inference import create_agent

            agent = create_agent(
                entry.ref,
                device=self.device,
                mode=str(self.cfg.params.get("mode", "auto")),
                simulations=self.cfg.params.get("simulations"),
                temperature=float(self.cfg.params.get("temperature", 0.0)),
                seed=self.seed, game_id=self.spec.game_id, rules=default_rules(self.spec),
                # Pool opponents only play; belief diagnostics would cost an extra head pass per decision.
                aux_outputs=False,
            )
            from .manifest import check_manifest_matches_spec

            # The loader only validated the checkpoint against its own manifest; the pool plays one variant.
            check_manifest_matches_spec(agent.manifest, self.spec, self.game_params)
        else:
            raise ValueError(f"unknown pool entry kind {entry.kind}")
        self._agent_cache[entry.entry_id] = agent
        return agent

    def record_result(self, entry: PoolEntry, learner_score: float) -> None:
        if not np.isfinite(learner_score) or not 0 <= learner_score <= 1:
            raise ValueError("learner_score must be finite and in [0, 1]")
        if entry.points is None:
            entry.points = entry.wins + 0.5 * entry.draws
        entry.points += learner_score
        if learner_score >= 1.0:
            entry.wins += 1
        elif learner_score <= 0.0:
            entry.losses += 1
        else:
            entry.draws += 1

    def stats(self) -> list[dict[str, Any]]:
        return [dict(entry.to_dict(), learner_win_rate=round(entry.learner_win_rate, 3)) for entry in self.all_entries()]

    # ------------------------------------------------------------ persistence
    def state_dict(self) -> dict[str, Any]:
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "classical": [entry.to_dict() for entry in self.classical],
            "current": self.current.to_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.entries = [PoolEntry.from_dict(item) for item in state.get("entries", []) if Path(item["ref"]).exists()]
        classical = {item["ref"]: PoolEntry.from_dict(item) for item in state.get("classical", [])}
        self.classical = [classical.get(entry.ref, entry) for entry in self.classical]
        if state.get("current"):
            self.current = PoolEntry.from_dict(state["current"])
        self._agent_cache.clear()


# ---------------------------------------------------------------- strategies


@OPPONENT_STRATEGIES.register("self")
def _self(pool: OpponentPool, rng: np.random.Generator) -> PoolEntry:
    return pool.current


@OPPONENT_STRATEGIES.register("latest")
def _latest(pool: OpponentPool, rng: np.random.Generator) -> PoolEntry:
    return pool.entries[-1] if pool.entries else pool.current


@OPPONENT_STRATEGIES.register("uniform")
def _uniform(pool: OpponentPool, rng: np.random.Generator) -> PoolEntry:
    candidates = pool.entries or [pool.current]
    return candidates[int(rng.integers(len(candidates)))]


@OPPONENT_STRATEGIES.register("pfsp")
def _pfsp(pool: OpponentPool, rng: np.random.Generator) -> PoolEntry:
    """Prioritized fictitious self-play: favour opponents the learner loses to."""
    candidates = pool.entries or [pool.current]
    weights = np.array([(1.0 - entry.learner_win_rate) ** 2 + 0.05 for entry in candidates], dtype=np.float64)
    weights /= weights.sum()
    return candidates[int(rng.choice(len(candidates), p=weights))]
