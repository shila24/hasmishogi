"""One-step Double DQN over learner decisions with opponent replies folded in.

The shared encoder includes current repetition features, but does not encode the
full repetition history: its planes are an approximation to a Markov state.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..config import ConfigError
from ..inference import QAgent
from ..registry import ALGORITHMS
from ..selfplay import ActorFn, Trajectory, run_vs_opponent_game
from .base import CollectStats, Learner, LearnerContext, UpdateStats


@dataclass
class DqnParams:
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_iterations: int = 50
    gamma: float = 0.99
    target_update_every: int = 200
    n_step: int = 1
    double: bool = True
    huber_delta: float = 1.0
    opponent_epsilon: float = 0.1

    def __post_init__(self) -> None:
        if type(self.n_step) is not int or self.n_step != 1:
            raise ConfigError("dqn: only n_step=1 is supported")
        for name in ("epsilon_decay_iterations", "target_update_every"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ConfigError(f"dqn: {name} must be a positive integer")
        for name in ("epsilon_start", "epsilon_end", "opponent_epsilon", "gamma"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ConfigError(f"dqn: {name} must be in [0, 1]")
        if self.epsilon_end > self.epsilon_start:
            raise ConfigError("dqn: epsilon_end must not exceed epsilon_start")
        if not isinstance(self.huber_delta, (int, float)) or not math.isfinite(self.huber_delta) or self.huber_delta <= 0:
            raise ConfigError("dqn: huber_delta must be finite and positive")
        if type(self.double) is not bool:
            raise ConfigError("dqn: double must be a boolean")


@ALGORITHMS.register("dqn")
class DqnLearner(Learner):
    name = "dqn"
    required_head = "q"
    replay_kind = "q"

    def __init__(self, ctx: LearnerContext) -> None:
        super().__init__(ctx)
        if self.replay is None:
            raise ConfigError("dqn requires a Q replay buffer")
        self.target_network = copy.deepcopy(self.network).to(self.device)
        self.target_network.requires_grad_(False)
        self.target_network.eval()
        self.gradient_steps = 0
        self.last_epsilon = float(self.params.epsilon_start)

    def params_dataclass(self) -> type:
        return DqnParams

    def _actor(self, game, planes, mask, ply):
        legal = np.flatnonzero(mask > 0)
        if not len(legal):
            raise ValueError("No legal action is available")
        if self.rng.random() < self.last_epsilon:
            action = int(self.rng.choice(legal))
        else:
            q, _ = self.infer(planes)
            action = int(np.argmax(np.where(mask > 0, q[0], -np.inf)))
        return action, None, {"epsilon": self.last_epsilon}

    def actor_fn(self) -> ActorFn:
        return self._actor

    def ingest_trajectories(self, trajectories: list[Trajectory]) -> int:
        """Store one transition per learner decision (plus optional mirrors).

        ``done`` marks the end of the training trajectory, including an explicit
        truncation target; it does not change the engine's terminal outcome.
        """
        cfg = self.config.selfplay
        added = 0
        for trajectory in trajectories:
            next_indices = {}
            last_by_player = {}
            for index in reversed(range(len(trajectory))):
                player = trajectory.players[index]
                next_indices[index] = last_by_player.get(player)
                last_by_player[player] = index
            for index, action in enumerate(trajectory.actions):
                next_index = next_indices[index]
                done = next_index is None
                sample = {
                    "planes": trajectory.planes[index],
                    "mask": trajectory.masks[index],
                    "action": action,
                    "reward": trajectory.value_for(index, cfg.truncation_value) if done else 0.0,
                    "next_planes": np.zeros_like(trajectory.planes[index]) if done else trajectory.planes[next_index],
                    "next_mask": np.zeros_like(trajectory.masks[index]) if done else trajectory.masks[next_index],
                    "done": float(done),
                }
                self.replay.add(**sample)
                added += 1
                if cfg.symmetry_augmentation:
                    for symmetry in self.spec.symmetries():
                        mirrored = dict(sample)
                        for field in ("planes", "next_planes"):
                            mirrored[field] = symmetry.planes(sample[field])
                        for field in ("mask", "next_mask"):
                            mirrored[field] = symmetry.policy(sample[field])
                        mirrored["action"] = symmetry.action(action)
                        self.replay.add(**mirrored)
                        added += 1
        return added

    def collect(self, iteration: int) -> CollectStats:
        start = time.perf_counter()
        cfg = self.config.selfplay
        # Trainer iterations are one-based; the first collection uses start.
        progress = min(1.0, max(0, iteration - 1) / self.params.epsilon_decay_iterations)
        self.last_epsilon = float(self.params.epsilon_start + progress * (self.params.epsilon_end - self.params.epsilon_start))
        stats = CollectStats()
        wins = [0] * (self.spec.num_players + 1)
        added = 0
        actor = self.actor_fn()
        for game_index in range(cfg.games_per_iteration):
            entry = self.opponents.sample()
            if entry.kind == "current":
                opponent = QAgent(self.network, self.encoder, self.device,
                                  epsilon=self.params.opponent_epsilon, rng=self.rng,
                                  amp=self.config.device.amp, spec=self.spec)
            else:
                opponent = self.opponents.agent_for(entry)
            learner_player = int(self.rng.integers(self.spec.num_players))
            trajectory = run_vs_opponent_game(
                actor, opponent, learner_player, self.encoder,
                spec=self.spec, backend=self.backend, max_plies=cfg.max_plies, seed=self.config.run.seed + iteration * cfg.games_per_iteration + game_index,
            )
            added += self.ingest_trajectories([trajectory])
            stats.games += 1
            stats.plies += trajectory.plies
            stats.truncated += int(trajectory.truncated)
            if not trajectory.truncated:
                wins[trajectory.result_index()] += 1
            # The pool only accepts W/D/L scores. Caps receive a neutral sampling
            # score, reported separately below; they are not rule-level draws.
            score = 0.5 if trajectory.truncated else (trajectory.rewards[learner_player] + 1.0) / 2.0
            self.opponents.record_result(entry, score)
        stats.wins = tuple(wins)
        stats.seconds = time.perf_counter() - start
        stats.extra = {"samples": added, "epsilon": self.last_epsilon,
                       "pool_truncated_neutral": stats.truncated}
        return stats

    @torch.no_grad()
    def _targets(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        was_training = self.network.training
        self.network.eval()
        self.target_network.eval()
        try:
            with self.autocast():
                target_q, _ = self.target_network(batch["next_planes"])
                if self.params.double:
                    online_q, _ = self.network(batch["next_planes"])
            legal = batch["next_mask"] > 0
            masked_target = target_q.float().masked_fill(~legal, -torch.inf)
            if self.params.double:
                actions = online_q.float().masked_fill(~legal, -torch.inf).argmax(dim=1)
                bootstrap = masked_target.gather(1, actions.unsqueeze(1)).squeeze(1)
            else:
                bootstrap = masked_target.max(dim=1).values
            # Select zero before arithmetic: multiplying 0 by -inf gives NaN.
            bootstrap = torch.where(legal.any(dim=1) & (batch["done"] == 0), bootstrap, 0.0)
            return batch["reward"].float() + self.params.gamma * (1 - batch["done"].float()) * bootstrap
        finally:
            self.network.train(was_training)

    def update(self, iteration: int) -> UpdateStats:
        start = time.perf_counter()
        if len(self.replay) < max(1, self.config.replay.min_size):
            return UpdateStats(seconds=time.perf_counter() - start)
        losses = []
        self.network.train()
        try:
            for _ in range(self.config.optimizer.updates_per_iteration):
                batch = self.to_device(self.replay.sample(self.config.replay.batch_size))
                targets = self._targets(batch)
                with self.autocast():
                    q, _ = self.network(batch["planes"])
                chosen = q.float().gather(1, batch["action"].long().unsqueeze(1)).squeeze(1)
                loss = torch.nn.functional.huber_loss(chosen, targets, delta=self.params.huber_delta)
                losses.append(self.clip_and_step(loss))
                self.gradient_steps += 1
                if self.gradient_steps % self.params.target_update_every == 0:
                    self.target_network.load_state_dict(self.network.state_dict())
        finally:
            self.network.eval()
        return UpdateStats(steps=len(losses), loss=float(np.mean(losses)) if losses else None,
                           seconds=time.perf_counter() - start,
                           extra={"gradient_steps": self.gradient_steps})

    def extra_state(self) -> dict[str, Any]:
        return {"target_network": self.target_network.state_dict(),
                "gradient_steps": self.gradient_steps, "last_epsilon": self.last_epsilon}

    def load_extra_state(self, extra: dict[str, Any]) -> None:
        self.target_network.load_state_dict(extra.get("target_network", self.network.state_dict()))
        self.target_network.to(self.device)
        self.target_network.requires_grad_(False)
        self.target_network.eval()
        self.gradient_steps = int(extra.get("gradient_steps", 0))
        self.last_epsilon = float(extra.get("last_epsilon", self.params.epsilon_start))

    def make_agent(self, mode: str = "eval") -> QAgent:
        return QAgent(self.network, self.encoder, self.device, epsilon=0.0,
                      rng=self.rng, amp=self.config.device.amp, spec=self.spec)

    def inference_hint(self) -> dict[str, Any]:
        return {"mode": "q", "simulations": 0}
