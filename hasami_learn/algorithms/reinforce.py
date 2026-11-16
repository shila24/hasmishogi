"""Reference learner: REINFORCE with a value baseline over self-play games.

Kept intentionally small so it doubles as the executable example of the Learner
contract for the other algorithms (AlphaZero, DQN, PPO, supervised, adversarial).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from ..encoding import masked_log_softmax, sample_from_policy
from ..registry import ALGORITHMS
from ..selfplay import augment, run_selfplay_game
from .base import CollectStats, Learner, UpdateStats


@dataclass
class ReinforceParams:
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    epochs: int = 1
    minibatch: int = 256
    normalize_advantage: bool = False
    advantage_clip: float = 5.0
    logp_floor: float = -20.0


@ALGORITHMS.register("reinforce")
class ReinforceLearner(Learner):
    name = "reinforce"
    required_head = "policy_value"
    replay_kind = None  # on-policy: keeps the last iteration's samples only

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._planes: list[np.ndarray] = []
        self._masks: list[np.ndarray] = []
        self._actions: list[int] = []
        self._values: list[float] = []

    def params_dataclass(self):
        return ReinforceParams

    # ------------------------------------------------------------ collect
    def _actor(self, game, planes, mask, ply):
        logits, _ = self.infer(planes)
        legal = mask > 0
        z = np.where(legal, logits[0].astype(np.float64), -np.inf)
        z -= z[legal].max()
        probs = np.exp(z)
        probs /= probs.sum()
        cfg = self.config.selfplay
        temperature = cfg.temperature if ply < cfg.temperature_plies else max(cfg.temperature * 0.5, 0.25)
        action = sample_from_policy(probs, mask, self.rng, temperature)
        return action, None, {}

    def actor_fn(self):
        return self._actor

    def ingest_trajectories(self, trajectories: list) -> int:
        cfg = self.config.selfplay
        added = 0
        for trajectory in trajectories:
            values = trajectory.values(cfg.truncation_value)
            for index in range(len(trajectory)):
                self._planes.append(trajectory.planes[index])
                self._masks.append(trajectory.masks[index])
                self._actions.append(trajectory.actions[index])
                self._values.append(float(values[index]))
                added += 1
                if cfg.symmetry_augmentation:
                    for planes, mask, _, action in augment(self.spec, trajectory.planes[index], trajectory.masks[index], None, trajectory.actions[index]):
                        self._planes.append(planes)
                        self._masks.append(mask)
                        self._actions.append(int(action))
                        self._values.append(float(values[index]))
                        added += 1
        return added

    def collect(self, iteration: int) -> CollectStats:
        start = time.perf_counter()
        cfg = self.config.selfplay
        stats = CollectStats()
        wins = [0] * (self.spec.num_players + 1)
        self._planes, self._masks, self._actions, self._values = [], [], [], []
        for game_index in range(cfg.games_per_iteration):
            trajectory = run_selfplay_game(self._actor, self.encoder, spec=self.spec, backend=self.backend, max_plies=cfg.max_plies, seed=self.config.run.seed + iteration * cfg.games_per_iteration + game_index)
            self.ingest_trajectories([trajectory])
            stats.games += 1
            stats.plies += trajectory.plies
            stats.truncated += int(trajectory.truncated)
            wins[trajectory.result_index()] += 1
        stats.wins = tuple(wins)
        stats.seconds = time.perf_counter() - start
        stats.extra = {"samples": len(self._actions)}
        return stats

    # ------------------------------------------------------------ update
    def update(self, iteration: int) -> UpdateStats:
        start = time.perf_counter()
        count = len(self._actions)
        if count == 0:
            return UpdateStats(seconds=time.perf_counter() - start)
        planes = np.stack(self._planes).astype(np.float32)
        masks = np.stack(self._masks).astype(np.float32)
        actions = np.asarray(self._actions, dtype=np.int64)
        values = np.asarray(self._values, dtype=np.float32)
        self.network.train()
        losses: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []
        steps = 0
        for _ in range(max(1, self.params.epochs)):
            order = self.rng.permutation(count)
            for begin in range(0, count, self.params.minibatch):
                index = order[begin : begin + self.params.minibatch]
                batch = self.to_device({"planes": planes[index], "mask": masks[index], "action": actions[index], "value": values[index]})
                with self.autocast():
                    logits, value = self.network(batch["planes"])
                logp = masked_log_softmax(logits.float(), batch["mask"])
                chosen = logp.gather(1, batch["action"].unsqueeze(1)).squeeze(1)
                target = batch["value"]
                advantage = target - value.float().detach()
                if self.params.normalize_advantage and advantage.numel() > 1:
                    std = advantage.std()
                    # Skip normalisation when almost every sample is a draw; dividing by a tiny std
                    # would turn the few decisive samples into enormous, destabilising weights.
                    if float(std) > 1e-3:
                        advantage = (advantage - advantage.mean()) / (std + 1e-8)
                if self.params.advantage_clip > 0:
                    advantage = advantage.clamp(-self.params.advantage_clip, self.params.advantage_clip)
                # Floor the log-probability so already-suppressed actions stop receiving gradient;
                # otherwise negative advantages push logits towards -inf without bound.
                chosen = chosen.clamp(min=self.params.logp_floor)
                policy_loss = -(chosen * advantage).mean()
                value_loss = torch.nn.functional.mse_loss(value.float(), target)
                entropy = -(torch.exp(logp) * logp * (batch["mask"] > 0)).sum(dim=1).mean()
                loss = policy_loss + self.params.value_coef * value_loss - self.params.entropy_coef * entropy
                losses.append(self.clip_and_step(loss))
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                steps += 1
        self.network.eval()
        self._planes, self._masks, self._actions, self._values = [], [], [], []
        return UpdateStats(steps, float(np.mean(losses)), float(np.mean(policy_losses)), float(np.mean(value_losses)),
                           time.perf_counter() - start, {"samples": count})
