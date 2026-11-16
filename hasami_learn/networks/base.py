"""Shared head modules and conventions for policy-value and Q networks.

Auxiliary (belief) heads are optional extras declared through the network parameter ``aux_heads``
(``{name: shape}`` or ``{name: {"shape": [...], "activation": "sigmoid" | "softmax" | "none"}}``).
A network that has them keeps ``forward(x) -> (head, value)`` unchanged, exposes the normalised
declaration as ``network.aux_heads`` (``{}`` when none) and adds
``forward_aux(x) -> (head, value, {name: raw (B, *shape)})``. Losses use the raw outputs;
``activate_aux`` turns them into the probabilities that agents publish through ``Decision.extra``.
"""

from __future__ import annotations

import math
import re
from typing import Any

import torch
from torch import nn

from ..games import get_spec

HEADS = ("policy_value", "q")
AUX_ACTIVATIONS = ("none", "sigmoid", "softmax")
_AUX_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def normalize_aux_heads(aux_heads: Any) -> dict[str, dict[str, Any]]:
    """Validate an ``aux_heads`` declaration into ``{name: {"shape": [...], "activation": str}}``."""
    if aux_heads is None:
        return {}
    if not isinstance(aux_heads, dict):
        raise ValueError("aux_heads must map head names to shapes")
    result: dict[str, dict[str, Any]] = {}
    for name, declared in aux_heads.items():
        if not isinstance(name, str) or not _AUX_NAME.match(name):
            raise ValueError(f"aux head name {name!r} must be an identifier")
        if isinstance(declared, dict):
            unknown = sorted(set(declared) - {"shape", "activation"})
            if unknown:
                raise ValueError(f"aux head {name}: unknown keys {unknown}")
            shape = declared.get("shape")
            activation = declared.get("activation", "none")
        else:
            shape, activation = declared, "none"
        if type(shape) is int:
            shape = [shape]
        if (not isinstance(shape, (list, tuple)) or not shape
                or any(type(n) is not int or n <= 0 for n in shape)):
            raise ValueError(f"aux head {name}: shape must be a non-empty list of positive integers")
        if activation not in AUX_ACTIVATIONS:
            raise ValueError(f"aux head {name}: activation must be one of {AUX_ACTIVATIONS}")
        result[name] = {"shape": [int(n) for n in shape], "activation": str(activation)}
    return result


def activate_aux(specs: dict[str, dict[str, Any]], raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Apply each head's declared activation (softmax over the last axis) to raw outputs."""
    out = {}
    for name, value in raw.items():
        activation = specs.get(name, {}).get("activation", "none")
        if activation == "sigmoid":
            out[name] = torch.sigmoid(value)
        elif activation == "softmax":
            out[name] = torch.softmax(value, dim=-1)
        else:
            out[name] = value
    return out


class AuxiliaryHeads(nn.Module):
    """Named auxiliary outputs computed from the shared feature vector."""

    def __init__(self, features: int, hidden: int, aux_heads: Any) -> None:
        super().__init__()
        self.specs = normalize_aux_heads(aux_heads)
        if not self.specs:
            raise ValueError("AuxiliaryHeads needs at least one declared head")
        self.layers = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(features, hidden), nn.ReLU(inplace=True),
                                nn.Linear(hidden, math.prod(spec["shape"])))
            for name, spec in self.specs.items()})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: layer(x).reshape(x.shape[0], *self.specs[name]["shape"])
                for name, layer in self.layers.items()}

    def activate(self, raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return activate_aux(self.specs, raw)


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class PolicyValueHead(nn.Module):
    """Map a (B, features) tensor to (policy_logits (B, action_size), value (B,))."""

    def __init__(self, features: int, hidden: int = 256, action_size: int | None = None) -> None:
        super().__init__()
        action_size = get_spec().action_size if action_size is None else action_size
        self.policy = nn.Sequential(nn.Linear(features, hidden, action_size), nn.ReLU(inplace=True), nn.Linear(hidden, action_size))
        self.value = nn.Sequential(nn.Linear(features, hidden // 2), nn.ReLU(inplace=True), nn.Linear(hidden // 2, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy(x), self.value(x).squeeze(-1)


class QHead(nn.Module):
    """Map a (B, features) tensor to (q_values (B, action_size), None)."""

    def __init__(self, features: int, hidden: int = 256, action_size: int | None = None) -> None:
        super().__init__()
        action_size = get_spec().action_size if action_size is None else action_size
        self.q = nn.Sequential(nn.Linear(features, hidden, action_size), nn.ReLU(inplace=True), nn.Linear(hidden, action_size))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.q(x), None


def make_head(head: str, features: int, hidden: int, action_size: int | None = None) -> nn.Module:
    if head == "policy_value":
        return PolicyValueHead(features, hidden, action_size)
    if head == "q":
        return QHead(features, hidden, action_size)
    raise ValueError(f"unknown head {head}; expected one of {HEADS}")


def resolve_dimensions(num_planes, board_shape, action_size):
    """Fill legacy defaults only when omitted, then validate network dimensions."""
    if num_planes is None or board_shape is None or action_size is None:
        default = get_spec()
        num_planes = default.make_encoder("").num_planes if num_planes is None else num_planes
        board_shape = default.board_shape if board_shape is None else board_shape
        action_size = default.action_size if action_size is None else action_size
    board_shape = tuple(board_shape)
    if (len(board_shape) != 2 or any(type(n) is not int or n <= 0 for n in board_shape)
            or type(num_planes) is not int or num_planes <= 0
            or type(action_size) is not int or action_size <= 0):
        raise ValueError("network dimensions must be positive integers with two board axes")
    return num_planes, board_shape, action_size
