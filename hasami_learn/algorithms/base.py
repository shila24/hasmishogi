"""Learner contract shared by every algorithm (see docs/TRAINING_ja.md)."""

from __future__ import annotations

import abc
import contextlib
import dataclasses
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from torch import nn

from hasami.agents import Agent
from ..config import ConfigError, OptimizerConfig, TrainingConfig
from ..devices import autocast_context, to_device
from ..games import Encoder, GameSpec, get_spec
from ..metrics import MetricsLogger
from ..replay import ReplayBuffer


@dataclass
class CollectStats:
    games: int = 0
    plies: int = 0
    wins: tuple[int, ...] = (0, 0, 0)  # one win count per seat followed by draws
    truncated: int = 0
    seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class UpdateStats:
    steps: int = 0
    loss: float | None = None
    policy_loss: float | None = None
    value_loss: float | None = None
    seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LearnerContext:
    config: TrainingConfig
    network: nn.Module
    encoder: Encoder
    device: torch.device
    rng: np.random.Generator
    replay: ReplayBuffer | None
    opponents: Any  # OpponentPool; typed loosely to avoid an import cycle
    metrics: MetricsLogger | None
    backend: str
    run_dir: Path
    spec: GameSpec | None = None


def build_optimizer(parameters, cfg: OptimizerConfig) -> torch.optim.Optimizer:
    params = [p for p in parameters if p.requires_grad]
    if cfg.name == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.name == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.name == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay, nesterov=cfg.momentum > 0)
    raise ConfigError(f"unknown optimizer {cfg.name}")


class Learner(abc.ABC):
    """One algorithm = collect experience, update the network, expose an Agent."""

    name: ClassVar[str] = "base"
    required_head: ClassVar[str] = "policy_value"
    # False for learners that need artifacts prepared outside a bare tiny config (e.g. a trained engine checkpoint
    # and calibrated levels); the agent conformance suite then leaves them to their own tests.
    conformance_tiny: ClassVar[bool] = True
    replay_kind: ClassVar[str | None] = "pv"  # "pv", "q" or None when the learner keeps its own data

    def __init__(self, ctx: LearnerContext) -> None:
        self.ctx = ctx
        self.config = ctx.config
        from ..games.base import as_v2
        self.spec = as_v2(ctx.spec or get_spec(ctx.config.game.name, ctx.config.game.params))
        ctx.spec = self.spec
        self.network = ctx.network
        self.encoder = ctx.encoder
        self.device = ctx.device
        self.rng = ctx.rng
        self.replay = ctx.replay
        self.opponents = ctx.opponents
        self.metrics = ctx.metrics
        self.backend = ctx.backend
        self.run_dir = ctx.run_dir
        self._optimizer: torch.optim.Optimizer | None = None
        self.params = self._parse_params()

    # ------------------------------------------------------------ params
    def params_dataclass(self) -> type | None:
        return None

    def _parse_params(self) -> Any:
        raw = dict(self.config.algorithm.params or {})
        cls = self.params_dataclass()
        if cls is None:
            return raw
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ConfigError(f"{self.name}: unknown algorithm params {unknown}; known: {sorted(known)}")
        return cls(**raw)

    # ------------------------------------------------------------ helpers
    @property
    def optimizer(self) -> torch.optim.Optimizer:
        if self._optimizer is None:
            self._optimizer = build_optimizer(self.network.parameters(), self.config.optimizer)
        return self._optimizer

    def autocast(self):
        return autocast_context(self.device, self.config.device.amp)

    def to_device(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        return to_device(batch, self.device)

    def log(self, kind: str, iteration: int, **fields: Any) -> None:
        if self.metrics is not None:
            self.metrics.log(kind, iteration, **fields)

    def clip_and_step(self, loss: torch.Tensor) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip = self.config.optimizer.grad_clip
        if clip and clip > 0:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=clip)
        self.optimizer.step()
        return float(loss.detach().item())

    @contextlib.contextmanager
    def timed(self):
        start = time.perf_counter()
        holder = {"seconds": 0.0}
        try:
            yield holder
        finally:
            holder["seconds"] = time.perf_counter() - start

    @torch.no_grad()
    def infer(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Run the network on a batch of planes; returns numpy (B, action_size) and (B,) or None."""
        was_training = self.network.training
        self.network.eval()
        try:
            x = torch.as_tensor(np.ascontiguousarray(planes), dtype=torch.float32, device=self.device)
            if x.ndim == 3:
                x = x.unsqueeze(0)
            with self.autocast():
                head, value = self.network(x)
            out = head.float().cpu().numpy()
            values = None if value is None else value.float().cpu().numpy()
            return out, values
        finally:
            if was_training:
                self.network.train()

    # ------------------------------------------------------------ contract
    @abc.abstractmethod
    def collect(self, iteration: int) -> CollectStats: ...

    @abc.abstractmethod
    def update(self, iteration: int) -> UpdateStats: ...

    # ------------------------------------------------------------ league hooks
    def actor_fn(self):
        """Return an ActorFn (game, planes, mask, ply) -> (action, policy_target, extra) for data collection.

        Learners that support opponent-pool / league training (adversarial wrapper)
        must implement this together with ``ingest_trajectories``.
        """
        raise NotImplementedError(f"{self.name} does not expose actor_fn()")

    def ingest_trajectories(self, trajectories: list) -> int:
        """Store recorded trajectories (list[Trajectory]) as training samples; return sample count."""
        raise NotImplementedError(f"{self.name} does not implement ingest_trajectories()")

    def make_agent(self, mode: str = "eval") -> Agent:
        """``mode="selfplay"`` serves opponents inside training: no auxiliary (belief) diagnostics."""
        from ..inference import PolicyAgent, QAgent

        aux = mode != "selfplay"
        if self.required_head == "q":
            return QAgent(self.network, self.encoder, self.device, spec=self.spec, aux_outputs=aux)
        return PolicyAgent(self.network, self.encoder, self.device, temperature=0.0, rng=self.rng, spec=self.spec,
                           aux_outputs=aux)

    def inference_hint(self) -> dict[str, Any]:
        return {"mode": "q" if self.required_head == "q" else "policy", "simulations": 0}

    def extra_state(self) -> dict[str, Any]:
        return {}

    def load_extra_state(self, extra: dict[str, Any]) -> None:
        return None

    def state_dict(self) -> dict[str, Any]:
        return {"optimizer": self.optimizer.state_dict(), "extra": self.extra_state()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        optimizer_state = state.get("optimizer")
        if optimizer_state:
            try:
                self.optimizer.load_state_dict(optimizer_state)
            except (ValueError, KeyError) as exc:
                print(f"warning: optimizer state not restored ({exc}); continuing with a fresh optimizer")
        self.load_extra_state(state.get("extra", {}) or {})

    def on_checkpoint(self, directory: Path) -> None:
        return None
