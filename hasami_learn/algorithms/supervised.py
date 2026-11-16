"""Imitation learning from validated records with legal policy and value targets."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch

from ..datasets import ARRAY_FIELDS, build_dataset, generate_games, load_dataset, save_dataset
from ..encoding import masked_log_softmax
from ..registry import ALGORITHMS
from ..selfplay import augment
from .base import CollectStats, Learner, UpdateStats


@dataclass
class SupervisedParams:
    dataset_path: str = ""
    generate_games: int = 200
    generator: str = ""
    random_opening_plies: int = 4
    augment: bool = True
    validation_fraction: float = 0.1
    label_smoothing: float = 0.0

    def __post_init__(self):
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0,1)")
        if not 0 <= self.label_smoothing <= 1:
            raise ValueError("label_smoothing must be in [0,1]")
        for name in ("generate_games", "random_opening_plies"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


def split_augmentation_groups(count: int, group_size: int, validation_fraction: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Split ``count`` consecutive samples into train/validation index arrays by whole groups.

    Samples are stored as ``[original, variant_1, ..., variant_k]`` blocks of ``group_size``,
    so a block never straddles the split (no symmetry leakage into validation).
    """
    if group_size < 1:
        raise ValueError("group_size must be positive")
    if count % group_size:
        raise ValueError(f"dataset of {count} samples is not a multiple of the augmentation group size {group_size}")
    groups = count // group_size
    order = rng.permutation(groups)
    held_out = min(groups - 1, max(1, int(groups * validation_fraction))) if validation_fraction and groups > 1 else 0
    indices = (order[:, None] * group_size + np.arange(group_size)).reshape(-1)
    return indices[held_out * group_size:], indices[:held_out * group_size]


@ALGORITHMS.register("supervised")
class SupervisedLearner(Learner):
    name = "supervised"
    required_head = "policy_value"
    replay_kind = "pv"

    def __init__(self, ctx):
        super().__init__(ctx)
        if self.replay is None:
            raise ValueError("supervised requires policy/value replay")
        self._collected = False
        self._dataset_path = ""
        self._train_indices = np.empty(0, np.int64)
        self._validation_indices = np.empty(0, np.int64)
        self._validation = None
        self._counts = {}

    def params_dataclass(self):
        return SupervisedParams

    def _install_dataset(self, data, refill=True):
        if data["planes"].shape[1] != self.encoder.num_planes:
            raise ValueError("dataset plane count does not match encoder")
        if data["meta"].get("encoding_id", self.encoder.encoding_id) != self.encoder.encoding_id:
            raise ValueError("dataset encoding does not match encoder")
        from ..datasets import _identity_params

        saved_encoder = _identity_params(data["meta"], "encoder_params")  # missing = default parameters
        if saved_encoder != dict(self.config.encoder.params):
            raise ValueError(f"dataset encoder_params {saved_encoder} do not match the configured encoder {dict(self.config.encoder.params)}")
        saved_game = _identity_params(data["meta"], "game_params")
        if saved_game != dict(self.config.game.params):
            raise ValueError(f"dataset game_params {saved_game} do not match the configured game {dict(self.config.game.params)}")
        self._validation = {key: data[key][self._validation_indices] for key in ARRAY_FIELDS}
        if refill:
            self.replay.clear()
            for begin in range(0, len(self._train_indices), self.config.replay.batch_size):
                indices = self._train_indices[begin:begin + self.config.replay.batch_size]
                self.replay.add_batch(**{key: data[key][indices] for key in ARRAY_FIELDS})
        self._counts = {"samples": len(data["value"]), "training_samples": len(self._train_indices),
                        "validation_samples": len(self._validation_indices), "replay_samples": len(self.replay)}

    def collect(self, iteration: int) -> CollectStats:
        start = time.perf_counter()
        if self._collected:
            return CollectStats(extra=dict(self._counts))
        generated = 0
        if self.params.dataset_path:
            self._dataset_path = self.params.dataset_path
            data = load_dataset(self._dataset_path, spec=self.spec, game_params=dict(self.config.game.params), encoder_params=dict(self.config.encoder.params))
        else:
            directory = self.run_dir / "data"
            paths = generate_games(
                directory / "games", self.params.generate_games,
                generator=self.params.generator or getattr(self.spec, "default_teacher", "random"), backend=self.backend, spec=self.spec,
                max_plies=self.config.selfplay.max_plies, seed=self.config.run.seed,
                random_opening_plies=self.params.random_opening_plies,
            )
            generated = len(paths)
            data = build_dataset(paths, self.encoder, self.backend, self.params.augment, spec=self.spec, game_params=dict(self.config.game.params), encoder_params=dict(self.config.encoder.params))
            self._dataset_path = str(save_dataset(directory / "dataset.npz", data, spec=self.spec).resolve())
        count = len(data["value"])
        if count == 0:
            raise ValueError("supervised training requires at least one sample")
        # Keep an original position and all of its symmetry variants on the same side of the split.
        # The group size is persisted by build_dataset; older datasets fall back to the spec's symmetries.
        meta = data["meta"]
        group_size = int(meta.get("group_size") or ((1 + len(self.spec.symmetries())) if meta.get("augment") else 1))
        self._train_indices, self._validation_indices = split_augmentation_groups(
            count, group_size, self.params.validation_fraction, self.rng)
        self._install_dataset(data)
        self._collected = True
        meta = data["meta"]
        return CollectStats(
            games=generated, plies=int(meta.get("plies", 0)) if generated else 0,
            wins=tuple(meta.get("wins", [0] * (self.spec.num_players + 1))) if generated else (0,) * (self.spec.num_players + 1),
            truncated=int(meta.get("ongoing", 0)) if generated else 0,
            seconds=time.perf_counter() - start, extra=dict(self._counts),
        )

    @torch.no_grad()
    def _validation_metrics(self):
        count = 0 if self._validation is None else len(self._validation["value"])
        if count == 0:
            return {"validation_samples": 0, "validation_policy_accuracy": None, "validation_value_mse": None}
        correct, squared_error = 0, 0.0
        was_training = self.network.training
        self.network.eval()
        try:
            for begin in range(0, count, self.config.replay.batch_size):
                batch = self.to_device({key: value[begin:begin + self.config.replay.batch_size] for key, value in self._validation.items()})
                with self.autocast():
                    logits, value = self.network(batch["planes"])
                actions = logits.float().masked_fill(batch["mask"] == 0, -torch.inf).argmax(1)
                correct += int((actions == batch["policy"].argmax(1)).sum().item())
                squared_error += float(((value.float() - batch["value"]) ** 2).sum().item())
        finally:
            self.network.train(was_training)
        return {"validation_samples": count, "validation_policy_accuracy": correct / count,
                "validation_value_mse": squared_error / count}

    def update(self, iteration: int) -> UpdateStats:
        start = time.perf_counter()
        losses, policy_losses, value_losses = [], [], []
        if len(self.replay) >= max(1, self.config.replay.min_size):
            self.network.train()
            for _ in range(self.config.optimizer.updates_per_iteration):
                batch = self.to_device(self.replay.sample(self.config.replay.batch_size))
                with self.autocast():
                    logits, value = self.network(batch["planes"])
                mask = batch["mask"]
                logp = masked_log_softmax(logits.float(), mask).masked_fill(mask == 0, 0)
                smoothing = self.params.label_smoothing
                target = (1 - smoothing) * batch["policy"] + smoothing * mask / mask.sum(1, keepdim=True)
                policy_loss = -(target * logp).sum(1).mean()
                value_loss = torch.nn.functional.mse_loss(value.float(), batch["value"])
                losses.append(self.clip_and_step(policy_loss + value_loss))
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
            self.network.eval()
        extra = {"samples": len(self.replay), **self._validation_metrics()}
        return UpdateStats(
            steps=len(losses), loss=float(np.mean(losses)) if losses else None,
            policy_loss=float(np.mean(policy_losses)) if losses else None,
            value_loss=float(np.mean(value_losses)) if losses else None,
            seconds=time.perf_counter() - start, extra=extra,
        )

    def actor_fn(self):
        def actor(game, planes, mask, ply):
            legal = np.flatnonzero(mask > 0)
            if not len(legal):
                raise ValueError("No legal action is available")
            logits, _ = self.infer(planes)
            action = int(legal[np.argmax(logits[0, legal])])
            return action, None, {}
        return actor

    def ingest_trajectories(self, trajectories: list) -> int:
        added = 0
        for trajectory in trajectories:
            for index, action in enumerate(trajectory.actions):
                planes, mask = trajectory.planes[index], trajectory.masks[index]
                if type(action) is not int or not 0 <= action < self.spec.action_size or mask[action] != 1:
                    raise ValueError("trajectory contains an illegal move label")
                policy = np.zeros(self.spec.action_size, np.float32)
                policy[action] = 1
                value = trajectory.value_for(index, self.config.selfplay.truncation_value)
                self.replay.add(planes=planes, mask=mask, policy=policy, value=value)
                added += 1
                if self.config.selfplay.symmetry_augmentation:
                    for planes, mask, policy, _ in augment(self.spec, planes, mask, policy, action):
                        self.replay.add(planes=planes, mask=mask, policy=policy, value=value)
                        added += 1
        return added

    def extra_state(self):
        return {"collected": self._collected, "dataset_path": self._dataset_path,
                "train_indices": torch.from_numpy(self._train_indices.copy()),
                "validation_indices": torch.from_numpy(self._validation_indices.copy())}

    def load_extra_state(self, extra):
        self._collected = bool(extra.get("collected", False))
        self._dataset_path = extra.get("dataset_path", "")
        if self._collected:
            self._train_indices = extra["train_indices"].cpu().numpy().copy()
            self._validation_indices = extra["validation_indices"].cpu().numpy().copy()
            self._install_dataset(load_dataset(self._dataset_path, spec=self.spec, game_params=dict(self.config.game.params), encoder_params=dict(self.config.encoder.params)))
