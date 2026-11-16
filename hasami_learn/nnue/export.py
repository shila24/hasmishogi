"""hlnnw1 float32 weights; no pickle, executable code or external dependencies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import torch

from .data import atomic_write
from .features import FEATURE_COUNT, FEATURE_SET, LAYOUT
from .model import NNUE, architecture_id

MAGIC = b"hlnnw1\0\0"
HEADER = struct.Struct("<8sI16s48sIIIIfQ")
TENSORS = ("embedding.weight", "bias", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc3.weight", "fc3.bias")


def shapes(width, hidden1, hidden2):
    return ((FEATURE_COUNT, width), (width,), (hidden1, 2*width), (hidden1,),
            (hidden2, hidden1), (hidden2,), (1, hidden2), (1,))


def export_weights(model, path):
    path = Path(path)
    dimensions = shapes(model.width, model.hidden1, model.hidden2)
    count = sum(int(np.prod(shape)) for shape in dimensions)
    header = HEADER.pack(MAGIC, 1, FEATURE_SET.encode(), model.architecture.encode(),
                         FEATURE_COUNT, model.width, model.hidden1, model.hidden2,
                         LAYOUT["output_scale"], count)
    # Bounded by the architecture limits (under 377 MiB for the largest first layer).
    content = bytearray(header)
    state = model.export_state()
    for name, shape in zip(TENSORS, dimensions):
        tensor = state[name].detach().to(device="cpu", dtype=torch.float32).numpy()
        if tensor.shape != shape or not np.isfinite(tensor).all() or (np.abs(tensor) > 1e6).any():
            raise ValueError("Invalid NNUE tensor")
        content.extend(tensor.astype("<f4", copy=False).tobytes())
    checksum = hashlib.sha256(content).digest()
    content.extend(checksum)
    atomic_write(path, content)
    manifest = {"format": "hlnnw1", "version": 1, "feature_set": FEATURE_SET,
                "architecture": model.architecture, "sha256": checksum.hex(),
                "bytes": len(content), "tensors": {name: list(shape) for name, shape in zip(TENSORS, dimensions)}}
    atomic_write(str(path) + ".json", (json.dumps(manifest, sort_keys=True) + "\n").encode())
    return manifest


def load_weights(path, *, device="cpu"):
    from hasami_learn.devices import resolve_device

    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(HEADER.size)
        if len(header) != HEADER.size:
            raise ValueError("Truncated hlnnw1 header")
        magic, version, feature, architecture, features, width, h1, h2, scale, count = HEADER.unpack(header)
        expected_architecture = architecture_id(width, h1, h2)
        dimensions = shapes(width, h1, h2)
        expected_count = sum(int(np.prod(s)) for s in dimensions)
        if (magic != MAGIC or version != 1 or feature != FEATURE_SET.encode().ljust(16, b"\0")
                or architecture != expected_architecture.encode().ljust(48, b"\0") or features != FEATURE_COUNT
                or scale != LAYOUT["output_scale"] or count != expected_count
                or path.stat().st_size != HEADER.size + 4*count + 32):
            raise ValueError("hlnnw1 feature/architecture/size mismatch")
        payload = stream.read(4*count)
        digest = hashlib.sha256(header)
        digest.update(payload)
        if stream.read(32) != digest.digest():
            raise ValueError("hlnnw1 sha256 mismatch")
    values = np.frombuffer(payload, dtype="<f4")
    if not np.isfinite(values).all() or (np.abs(values) > 1e6).any():
        raise ValueError("Invalid hlnnw1 tensor")
    model = NNUE(width, h1, h2)
    offset = 0
    state = {}
    for name, shape in zip(TENSORS, dimensions):
        size = int(np.prod(shape))
        state[name] = torch.from_numpy(values[offset:offset+size].reshape(shape).astype(np.float32, copy=True))
        offset += size
    model.load_state_dict(state)
    return model.to(resolve_device(device))
