"""AlphaZero-style residual tower for the game board."""

from __future__ import annotations

import torch
from torch import nn

from ..registry import NETWORKS
from .base import AuxiliaryHeads, init_weights, make_head, resolve_dimensions


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(x + y)


class ResNet(nn.Module):
    def __init__(self, head: str = "policy_value", num_planes: int | None = None, channels: int = 64, blocks: int = 4, head_channels: int = 4, hidden: int = 256, board_shape=None, action_size=None, aux_heads=None) -> None:
        super().__init__()
        num_planes, board_shape, action_size = resolve_dimensions(num_planes, board_shape, action_size)
        self.board_shape = board_shape
        self.action_size = action_size
        self.head = head
        self.num_planes = num_planes
        self.stem = nn.Sequential(nn.Conv2d(num_planes, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(max(1, blocks))])
        self.reduce = nn.Sequential(nn.Conv2d(channels, head_channels, 1, bias=False), nn.BatchNorm2d(head_channels), nn.ReLU(inplace=True), nn.Flatten())
        features = head_channels * board_shape[0] * board_shape[1]
        self.out = make_head(head, features, hidden, self.action_size)
        self.aux = AuxiliaryHeads(features, hidden, aux_heads) if aux_heads else None
        self.aux_heads = dict(self.aux.specs) if self.aux is not None else {}
        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        # x: (B, C, H, W) -> tower (B, channels, H, W) -> (B, head_channels*H*W) -> heads
        return self.out(self.reduce(self.tower(self.stem(x))))

    def forward_aux(self, x: torch.Tensor):
        features = self.reduce(self.tower(self.stem(x)))
        head, value = self.out(features)
        return head, value, (self.aux(features) if self.aux is not None else {})


NETWORKS.register("resnet", ResNet)
