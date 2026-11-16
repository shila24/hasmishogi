"""On-policy PPO with terminal-reward GAE over opponent-pool games."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from ..config import ConfigError
from ..encoding import masked_log_softmax
from ..inference import PolicyAgent
from ..registry import ALGORITHMS
from ..selfplay import Trajectory, augment, run_vs_opponent_game
from .base import CollectStats, Learner, UpdateStats


@dataclass
class PpoParams:
    clip: float = 0.2
    epochs: int = 4
    minibatches: int = 4
    gae_lambda: float = 0.95
    gamma: float = 1.0
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    value_clip: float = 0.2
    target_kl: float | None = None
    normalize_advantage: bool = True

    def __post_init__(self) -> None:
        for name in ("epochs", "minibatches"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ConfigError(f"ppo: {name} must be a positive integer")
        for name in ("gamma", "gae_lambda"):
            if not 0 <= getattr(self, name) <= 1:
                raise ConfigError(f"ppo: {name} must be in [0, 1]")
        for name in ("clip", "entropy_coef", "value_coef", "value_clip"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ConfigError(f"ppo: {name} must be finite and nonnegative")
        if self.target_kl is not None and (not np.isfinite(self.target_kl) or self.target_kl <= 0):
            raise ConfigError("ppo: target_kl must be positive or null")


def compute_gae(
    trajectory: Trajectory,
    values,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    truncation_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute advantages per learner move, with zero bootstrap at the episode boundary.

    A truncated episode uses the explicit configured target, not a game draw.
    Separate colour sequences also support trajectories recording both players.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (len(trajectory),):
        raise ValueError("one old value is required for every trajectory decision")
    advantages = np.zeros(len(trajectory), dtype=np.float32)
    players = np.asarray(trajectory.players)
    for player in np.unique(players):
        indices = np.flatnonzero(players == player)
        next_value = 0.0
        advantage = 0.0
        for index in reversed(indices):
            reward = trajectory.value_for(int(index), truncation_value) if index == indices[-1] else 0.0
            delta = reward + gamma * next_value - float(values[index])
            advantage = delta + gamma * gae_lambda * advantage
            advantages[index] = advantage
            next_value = float(values[index])
    return advantages, advantages + values


def _policy_log_probs(logits, mask, temperature):
    """Use the same temperature and legal support during rollout and update."""
    temperature = temperature.reshape(-1, 1)
    positive = temperature > 0
    scaled = logits / torch.where(positive, temperature, torch.ones_like(temperature))
    # Temperature zero is a point mass at the first legal argmax.
    best = logits.masked_fill(mask <= 0, -torch.inf).argmax(dim=1, keepdim=True)
    greedy_mask = torch.zeros_like(mask).scatter(1, best, 1)
    support = torch.where(positive, mask, greedy_mask)
    return masked_log_softmax(scaled, support)


@ALGORITHMS.register("ppo")
class PpoLearner(Learner):
    name = "ppo"
    required_head = "policy_value"
    replay_kind = None

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._clear_rollout()

    def params_dataclass(self):
        return PpoParams

    def _clear_rollout(self) -> None:
        self._planes: list[np.ndarray] = []
        self._masks: list[np.ndarray] = []
        self._actions: list[int] = []
        self._logp_old: list[float] = []
        self._value_old: list[float] = []
        self._advantages: list[float] = []
        self._returns: list[float] = []
        self._temperatures: list[float] = []

    def _actor(self, game, planes, mask, ply):
        if not np.any(mask > 0):
            raise ValueError("PPO cannot act without legal actions")
        logits, values = self.infer(planes)
        cfg = self.config.selfplay
        temperature = cfg.temperature if ply < cfg.temperature_plies else 1.0
        logp = _policy_log_probs(torch.from_numpy(logits), torch.as_tensor(mask).unsqueeze(0),
                                 torch.tensor([temperature], dtype=torch.float32))[0].numpy()
        legal = np.flatnonzero(mask > 0)
        probs = np.exp(logp[legal].astype(np.float64))
        action = int(self.rng.choice(legal, p=probs / probs.sum()))
        return action, None, {"logp": float(logp[action]), "value": float(values[0]),
                              "temperature": float(temperature)}

    def actor_fn(self):
        return self._actor

    def ingest_trajectories(self, trajectories: list[Trajectory]) -> int:
        cfg = self.config.selfplay
        added = 0
        for trajectory in trajectories:
            if not len(trajectory):
                continue
            extras = [trajectory.extras[i] if i < len(trajectory.extras) else {}
                      for i in range(len(trajectory))]
            missing = [i for i, extra in enumerate(extras) if "logp" not in extra or "value" not in extra]
            old_logp = [extra.get("logp") for extra in extras]
            old_values = [extra.get("value") for extra in extras]
            # Without metadata, the wrapper's policy is approximated by the current
            # untempered policy; its historical behaviour cannot be reconstructed.
            temperatures = [float(extra.get("temperature", 1.0)) for extra in extras]
            if missing:
                logits, values = self.infer(np.stack([trajectory.planes[i] for i in missing]))
                masks = np.stack([trajectory.masks[i] for i in missing])
                logp = _policy_log_probs(torch.from_numpy(logits), torch.from_numpy(masks),
                                         torch.tensor([temperatures[i] for i in missing]))
                for row, i in enumerate(missing):
                    if old_logp[i] is None:
                        old_logp[i] = float(logp[row, trajectory.actions[i]])
                    if old_values[i] is None:
                        old_values[i] = float(values[row])
            advantages, returns = compute_gae(trajectory, old_values, self.params.gamma,
                                              self.params.gae_lambda, cfg.truncation_value)
            for i, action in enumerate(trajectory.actions):
                if not 0 <= action < len(trajectory.masks[i]) or trajectory.masks[i][action] <= 0:
                    raise ValueError("PPO trajectory contains an illegal action")
                samples = [(trajectory.planes[i], trajectory.masks[i], None, action)]
                if cfg.symmetry_augmentation:
                    samples.extend(augment(self.spec, trajectory.planes[i],
                                                       trajectory.masks[i], None, action))
                for planes, mask, _, sample_action in samples:
                    self._planes.append(planes)
                    self._masks.append(mask)
                    self._actions.append(int(sample_action))
                    self._logp_old.append(float(old_logp[i]))
                    self._value_old.append(float(old_values[i]))
                    self._advantages.append(float(advantages[i]))
                    self._returns.append(float(returns[i]))
                    self._temperatures.append(temperatures[i])
                    added += 1
        return added

    def collect(self, iteration: int) -> CollectStats:
        start = time.perf_counter()
        cfg = self.config.selfplay
        self._clear_rollout()
        stats = CollectStats()
        wins = [0] * (self.spec.num_players + 1)
        for game_index in range(cfg.games_per_iteration):
            learner_player = int(self.rng.integers(self.spec.num_players))
            entry = self.opponents.sample()
            opponent = (PolicyAgent(self.network, self.encoder, self.device,
                                    temperature=cfg.temperature, rng=self.rng, spec=self.spec)
                        if entry.kind == "current" else self.opponents.agent_for(entry))
            trajectory = run_vs_opponent_game(self.actor_fn(), opponent, learner_player, self.encoder,
                                              spec=self.spec, backend=self.backend, max_plies=cfg.max_plies, seed=self.config.run.seed + iteration * cfg.games_per_iteration + game_index)
            self.ingest_trajectories([trajectory])
            stats.games += 1
            stats.plies += trajectory.plies
            stats.truncated += int(trajectory.truncated)
            if trajectory.truncated:
                # The pool has only win/loss/draw buckets. Explicitly assign a
                # neutral sampling score, while retaining separate game statistics.
                score = 0.5
            else:
                wins[trajectory.result_index()] += 1
                score = (trajectory.rewards[learner_player] + 1.0) / 2.0
            self.opponents.record_result(entry, score)
        stats.wins = tuple(wins)
        stats.seconds = time.perf_counter() - start
        stats.extra = {"samples": len(self._actions), "pool_truncated_neutral_scores": stats.truncated}
        return stats

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
        for _ in range(self.params.epochs):
            order = self.rng.permutation(count)
            for index in np.array_split(order, min(count, self.params.minibatches)):
                batch = self.to_device({name: array[index] for name, array in arrays.items()})
                with self.autocast():
                    logits, value = self.network(batch["planes"])
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
                safe_logp = logp.masked_fill(batch["mask"] <= 0, 0)
                entropy = -(logp.exp() * safe_logp).sum(dim=1).mean()
                loss = policy_loss + self.params.value_coef * value_loss - self.params.entropy_coef * entropy
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
        return UpdateStats(steps=steps, loss=means[0], policy_loss=means[1], value_loss=means[2],
                           seconds=time.perf_counter() - start,
                           extra={"samples": count, "policy_loss": means[1], "value_loss": means[2],
                                  "approx_kl": means[3], "clip_fraction": means[4], "early_stop": early_stop})

    def make_agent(self, mode: str = "eval") -> PolicyAgent:
        """``mode="selfplay"`` serves opponents inside training: no auxiliary (belief) diagnostics."""
        return PolicyAgent(self.network, self.encoder, self.device, temperature=0.0, rng=self.rng, spec=self.spec,
                           aux_outputs=mode != "selfplay")

    def inference_hint(self) -> dict:
        return {"mode": "policy", "simulations": 0}
