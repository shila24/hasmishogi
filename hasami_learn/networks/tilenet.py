"""Conv1d features over tiles, retaining observer-relative rows and tile identity."""

from __future__ import annotations

import math

from torch import nn

from ..registry import NETWORKS
from .base import AuxiliaryHeads, init_weights, make_head, resolve_dimensions


@NETWORKS.register("tilenet")
class TileNet(nn.Module):
    def __init__(self, head="policy_value", num_planes=None, board_shape=None,
                 action_size=None, channels=32, hidden=128, layers=2, aux_heads=None):
        super().__init__()
        self.num_planes, self.board_shape, self.action_size = resolve_dimensions(
            num_planes, board_shape, action_size)
        if any(type(n) is not int or n < 1 for n in (channels, layers)) or type(hidden) is not int or hidden < 2:
            raise ValueError("channels/layers must be positive integers and hidden >= 2")
        self.head = head
        blocks = []
        inputs = self.num_planes * self.board_shape[0]
        for _ in range(layers):
            blocks.extend([nn.Conv1d(inputs, channels, 3, padding=1), nn.ReLU()])
            inputs = channels
        blocks.extend([nn.Flatten(), nn.Linear(channels * self.board_shape[1], hidden), nn.ReLU()])
        self.body = nn.Sequential(*blocks)
        self.out = make_head(head, hidden, hidden, self.action_size)
        self.aux = AuxiliaryHeads(hidden, hidden, aux_heads) if aux_heads else None
        self.aux_heads = dict(self.aux.specs) if self.aux is not None else {}
        self.apply(init_weights)
        for layer in self.body:
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))

    def _features(self, x):
        if tuple(x.shape[1:]) != (self.num_planes, *self.board_shape):
            raise ValueError("input dimensions do not match TileNet")
        return self.body(x.reshape(x.shape[0], -1, self.board_shape[1]))

    def forward(self, x):
        return self.out(self._features(x))

    def forward_aux(self, x):
        features = self._features(x)
        policy, value = self.out(features)
        return policy, value, self.aux(features) if self.aux is not None else {}
