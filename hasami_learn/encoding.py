"""Generic policy helpers and legacy Hasami encoder exports."""

from __future__ import annotations
import numpy as np
from .games.hasami import (PlaneEncoder, ENCODING_ID, NUM_PLANES, BOARD_BYTES,
                           FLIP_SQUARE, FLIP_ACTION, board_to_plane,
                           terminal_value_for_side_to_move)

def outcome_value(rewards: list[float] | tuple[float, ...], player: int) -> float:
    """Fixed-player env rewards -> value for ``player``."""
    return float(rewards[player])


def masked_log_softmax(logits, mask):
    """Log-softmax over legal actions only; rows without legal actions return zeros."""
    import torch

    mask_bool = mask > 0
    neg = torch.finfo(logits.dtype).min
    masked = logits.masked_fill(~mask_bool, neg)
    logp = torch.log_softmax(masked, dim=-1)
    has_legal = mask_bool.any(dim=-1, keepdim=True)
    return torch.where(has_legal, logp, torch.zeros_like(logp))


def masked_softmax(logits, mask):
    import torch

    return torch.exp(masked_log_softmax(logits, mask)) * (mask > 0).to(logits.dtype)


def sample_from_policy(probs: np.ndarray, mask: np.ndarray, rng: np.random.Generator, temperature: float = 1.0) -> int:
    """Pick a legal action from a probability vector; temperature 0 = argmax."""
    legal = np.flatnonzero(mask > 0)
    if legal.size == 0:
        raise ValueError("No legal action is available")
    weights = np.asarray(probs, dtype=np.float64)[legal]
    if temperature <= 0:
        return int(legal[int(np.argmax(weights))])
    if temperature != 1.0:
        weights = np.power(np.maximum(weights, 0.0), 1.0 / temperature)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return int(rng.choice(legal))
    return int(rng.choice(legal, p=weights / total))
