"""Sparse shared first layer and deterministic float32 NNUE reference forward."""
from __future__ import annotations

import torch
from torch import nn
from .features import FEATURE_COUNT, FEATURE_SET, KING_STRIDE, LAYOUT, MAX_ACTIVE


def architecture_id(width=256, hidden1=32, hidden2=32):
    if any(type(v) is not int or not 1 <= v <= cap
           for v, cap in ((width, 512), (hidden1, 128), (hidden2, 128))):
        raise ValueError("Invalid NNUE architecture dimensions")
    return f"{FEATURE_SET}:w{width}:{hidden1}:{hidden2}:f32:cp600"


class NNUE(nn.Module):
    """Inputs (B,2,39) sorted indices, output (B,) centipawns for the mover.

    ``forward`` is the ordered export reference. ``forward_training`` uses
    reductions and GEMM; its float32 tolerance is 0.002 cp + 1e-5 relative.
    The optional king-independent table is training-only and folded at export.
    """
    def __init__(self, width=256, hidden1=32, hidden2=32, *, factorised=False):
        super().__init__()
        self.architecture = architecture_id(width, hidden1, hidden2)
        self.width, self.hidden1, self.hidden2 = width, hidden1, hidden2
        self.embedding = nn.Embedding(FEATURE_COUNT, width)
        self.factor = nn.Embedding(KING_STRIDE, width, _weight=torch.zeros(KING_STRIDE, width)) if factorised else None
        self.bias = nn.Parameter(torch.full((width,), 0.25))
        self.fc1 = nn.Linear(2 * width, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.constant_(self.fc1.bias, 0.1)
        nn.init.constant_(self.fc2.bias, 0.1)
        nn.init.zeros_(self.fc3.bias)

    def feature_values(self, indices):
        if indices.ndim != 3 or tuple(indices.shape[1:]) != (2, MAX_ACTIVE):
            raise ValueError("NNUE requires indices shaped (batch,2,39)")
        safe = indices.clamp_min(0)
        values = self.embedding(safe)
        if self.factor is not None:
            values = values + self.factor(safe.remainder(KING_STRIDE))
        return values * (indices >= 0).unsqueeze(-1)

    def forward_training(self, indices):
        """Vectorised optimisation/validation path; never used for native parity."""
        x = (self.feature_values(indices).sum(dim=2) + self.bias).flatten(1).clamp(0, 1)
        x = self.fc1(x).clamp(0, 1)
        x = self.fc2(x).clamp(0, 1)
        return self.fc3(x)[:, 0].float() * LAYOUT["output_scale"]

    def export_state(self):
        """Fold the factor table into each king block without changing this model."""
        state = {name: tensor.detach().cpu() for name, tensor in self.state_dict().items()
                 if name != "factor.weight"}
        if self.factor is not None:
            embedding = state["embedding.weight"].reshape(-1, KING_STRIDE, self.width)
            state["embedding.weight"] = (embedding + self.factor.weight.detach().cpu()).flatten(0, 1)
        return state

    def forward(self, indices):
        values = self.feature_values(indices)
        active = indices >= 0
        accumulator = self.bias.expand(indices.shape[0], 2, -1)
        compensation = torch.zeros_like(accumulator)
        for slot in range(MAX_ACTIVE):
            adjusted = values[:, :, slot] - compensation
            updated = accumulator + adjusted
            present = active[:, :, slot:slot+1]
            compensation = torch.where(present, (updated-accumulator)-adjusted, compensation)
            accumulator = torch.where(present, updated, accumulator)
        x = accumulator.flatten(1).clamp(0, 1)
        for i, layer in enumerate((self.fc1, self.fc2, self.fc3)):
            y = layer.bias.expand(x.shape[0], -1)
            for column in range(x.shape[1]):
                y = y + x[:, column:column+1] * layer.weight[:, column]
            x = y.clamp(0, 1) if i < 2 else y
        return x[:, 0] * LAYOUT["output_scale"]
