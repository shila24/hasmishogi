"""Hand-end supervision; unrevealed labels are absent, never negative examples.

Waits and danger describe structural waits at the end of the current hand,
ignoring yaku and furiten. They are hindsight risk proxies, not exact deal-in
probabilities at the earlier decision. Hand targets are tile-type distributions.
Placement is the observer's final match rank and is masked on truncation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def hand_targets(summary: dict, player: int, num_players: int) -> dict[str, np.ndarray]:
    opponents = [(player + offset) % num_players for offset in range(1, num_players)]
    result = {}
    if "tenpai" in summary:
        result["tenpai"] = np.array(summary["tenpai"], dtype=np.float32)[opponents].copy()
    if "waits" in summary:
        waits = np.array(summary["waits"], dtype=np.float32)[opponents].copy()
        result["waits"] = waits
        result["danger"] = np.max(waits, axis=0)
    if "hand_counts" in summary:
        counts = np.array(summary["hand_counts"], dtype=np.float32)[opponents].copy()
        result["hand"] = counts / np.maximum(counts.sum(axis=-1, keepdims=True), 1)
    return result


def trajectory_targets(trajectory) -> list[dict[str, np.ndarray]]:
    summaries = [(index, info["hand"]) for index, info in enumerate(trajectory.step_info)
                 if info.get("hand")]
    final = (trajectory.step_info[-1].get("hand", {})
             if trajectory.step_info and not trajectory.truncated else {})
    result = []
    for player, extra in zip(trajectory.players, trajectory.extras):
        ply = extra["decision_ply"]
        summary = next((hand for index, hand in summaries if index >= ply), None)
        target = hand_targets(summary, player, trajectory.num_players) if summary else {}
        if "placement" in final:
            placement = np.zeros(trajectory.num_players, dtype=np.float32)
            placement[int(final["placement"][player]) - 1] = 1
            target["placement"] = placement
        result.append(target)
    return result


def auxiliary_loss(raw: dict[str, torch.Tensor], targets: list[dict], weights=None):
    """Mean each head over labelled samples only; return loss and detached metrics."""
    if not raw:
        raise ValueError("auxiliary_loss requires raw head outputs")
    total = next(iter(raw.values())).sum() * 0
    metrics = {}
    for name, output in raw.items():
        indices = [i for i, target in enumerate(targets) if name in target]
        weight = float((weights or {}).get(name, 1.0))
        if not indices or weight == 0:
            continue
        expected = torch.as_tensor(np.stack([targets[i][name] for i in indices]),
                                   device=output.device, dtype=torch.float32)
        predicted = output[indices].float()
        if predicted.shape != expected.shape:
            raise ValueError(f"auxiliary head {name}: shape {predicted.shape} != target {expected.shape}")
        if name in ("tenpai", "waits", "danger"):
            loss = F.binary_cross_entropy_with_logits(predicted, expected)
        elif name in ("hand", "placement"):
            loss = -(expected * F.log_softmax(predicted, dim=-1)).sum(-1).mean()
        else:
            raise ValueError(f"no auxiliary loss defined for {name}")
        total = total + weight * loss
        metrics[name] = float(loss.detach())
    return total, metrics
