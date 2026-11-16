"""Small fully-connected network; fast enough for CPU smoke tests."""

from __future__ import annotations

import torch
from torch import nn

from ..registry import NETWORKS
from .base import AuxiliaryHeads, init_weights, make_head, resolve_dimensions


class MlpNet(nn.Module):
    def __init__(self, head: str = "policy_value", num_planes: int | None = None, hidden: int = 256, layers: int = 2, board_shape=None, action_size=None, aux_heads=None) -> None:
        super().__init__()
        num_planes, board_shape, action_size = resolve_dimensions(num_planes, board_shape, action_size)
        self.board_shape = board_shape
        self.action_size = action_size
        self.head = head
        self.num_planes = num_planes
        blocks: list[nn.Module] = [nn.Flatten()]
        features = num_planes * board_shape[0] * board_shape[1]
        for _ in range(max(1, layers)):
            blocks += [nn.Linear(features, hidden), nn.ReLU(inplace=True)]
            features = hidden
        self.body = nn.Sequential(*blocks)
        self.out = make_head(head, features, hidden, self.action_size)
        self.aux = AuxiliaryHeads(features, hidden, aux_heads) if aux_heads else None
        self.aux_heads = dict(self.aux.specs) if self.aux is not None else {}
        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        # x: (B, C, H, W) -> (B, C*H*W) -> heads
        return self.out(self.body(x))

    def forward_aux(self, x: torch.Tensor):
        features = self.body(x)
        head, value = self.out(features)
        return head, value, (self.aux(features) if self.aux is not None else {})


NETWORKS.register("mlp", MlpNet)
