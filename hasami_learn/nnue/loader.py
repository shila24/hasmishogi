"""Bounded shuffled hlnn1 streaming with native batch feature extraction.

No derived files are needed: at most one raw shuffle buffer per worker and two
prefetched batches are resident. Spawn workers never receive a model or GPU.
"""
from __future__ import annotations

import ctypes as C
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from shogi.model import SQUARES

from .data import HEADER, MAGIC, RECORD, Record, digest_file, verify_shard
from .features import MAX_ACTIVE, active_features

RAW_DTYPE = np.dtype([("board", "u1", (SQUARES,)), ("hands", "u1", (14,)),
                      ("turn", "u1"), ("move_number", "<u4"), ("score", "<i4"),
                      ("depth", "<u2"), ("nodes", "<u8"), ("best_move", "<u2"),
                      ("result", "i1"), ("ply", "<u4"), ("flags", "<u4"),
                      ("teacher", "<f4"), ("game_id", "<u8")])
assert RAW_DTYPE.itemsize == RECORD.size


def validation_game(game_id, seed, fraction):
    digest = hashlib.sha256(f"hlnn-split-v1:{seed}:{game_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little") / 2**64 < fraction


@lru_cache(maxsize=1)
def native_encoder():
    from shogi.native_search import _load_path, find_library

    lib = _load_path(str(find_library()))
    try:
        encode = lib.ss_nnue_encode_records
    except AttributeError as exc:
        raise RuntimeError("Rebuild native shogi search for batch NNUE features") from exc
    encode.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t]
    encode.restype = C.c_int32
    return lib, encode


def resolve_backend(backend):
    if backend == "auto":
        try:
            native_encoder()
            return "native"
        except RuntimeError:
            return "python"
    if backend == "native":
        native_encoder()
    elif backend != "python":
        raise ValueError("Invalid NNUE feature backend")
    return backend


def encode_records(raw, backend="native"):
    """Validate canonical packed records and return contiguous int32 features."""
    raw = np.ascontiguousarray(raw, dtype=RAW_DTYPE)
    indices = np.full((len(raw), 2, MAX_ACTIVE), -1, dtype=np.int32)
    if backend == "native":
        lib, encode = native_encoder()
        if encode(raw.ctypes.data, raw.nbytes, indices.ctypes.data, indices.size):
            raise ValueError(lib.ss_last_error().decode("utf-8", errors="replace"))
    elif backend == "python":
        for row, packed in enumerate(raw):
            for side, features in enumerate(active_features(Record.unpack(packed.tobytes()).position)):
                indices[row, side, :len(features)] = features
    else:
        raise ValueError("Invalid NNUE feature backend")
    return indices


def raw_chunks(entry, chunk_size=8192):
    path, expected_hash, expected_sidecar = entry
    manifest = verify_shard(path)
    if manifest["sha256"] != expected_hash or digest_file(str(path)+".json") != expected_sidecar:
        raise ValueError("Training shard changed since configuration was recorded")
    games, count = set(), 0
    with Path(path).open("rb") as stream:
        if stream.read(HEADER.size) != HEADER.pack(MAGIC, 1, RECORD.size):
            raise ValueError("hlnn1 header mismatch")
        while content := stream.read(chunk_size * RECORD.size):
            if len(content) % RECORD.size:
                raise ValueError("Partial hlnn1 record")
            rows = np.frombuffer(content, dtype=RAW_DTYPE)
            games.update(map(int, np.unique(rows["game_id"])))
            count += len(rows)
            yield rows
    if count != manifest["records"] or len(games) != manifest["games"]:
        raise ValueError("hlnn1 record/game count mismatch")


class FeatureBatches(IterableDataset):
    """Disjoint worker shards, shuffled shard order and cross-game block shuffle."""
    def __init__(self, entries, *, validation, seed, fraction, batch_size, epoch,
                 shuffle_buffer, backend):
        super().__init__()
        self.entries = entries
        self.validation, self.seed, self.fraction = validation, seed, fraction
        self.batch_size, self.epoch = batch_size, epoch
        self.shuffle_buffer, self.backend = shuffle_buffer, backend

    def __iter__(self):
        worker = get_worker_info()
        worker_id, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        order = np.arange(len(self.entries))
        rng = np.random.default_rng(self.seed + self.epoch)
        if not self.validation:
            rng.shuffle(order)
        rng = np.random.default_rng([self.seed, self.epoch, worker_id])
        capacity = max(self.batch_size, self.shuffle_buffer)
        buffer = np.empty(capacity, dtype=RAW_DTYPE)
        used = 0
        for index in order[worker_id::workers]:
            for rows in raw_chunks(self.entries[index]):
                ids, inverse = np.unique(rows["game_id"], return_inverse=True)
                selected = np.array([validation_game(int(g), self.seed, self.fraction)
                                     == self.validation for g in ids])[inverse]
                rows = rows[selected]
                offset = 0
                while offset < len(rows):
                    size = min(capacity-used, len(rows)-offset)
                    buffer[used:used+size] = rows[offset:offset+size]
                    used += size
                    offset += size
                    if used == capacity:
                        yield from self.emit(buffer, rng)
                        used = 0
        if used:
            yield from self.emit(buffer[:used], rng)

    def emit(self, rows, rng):
        if not self.validation:
            rng.shuffle(rows)
        for offset in range(0, len(rows), self.batch_size):
            batch = rows[offset:offset+self.batch_size]
            yield {"indices": encode_records(batch, self.backend),
                   **{name: batch[name].astype(np.float32 if name != "game_id" else np.uint64)
                      for name in ("score", "result", "flags", "teacher", "game_id")}}


def numpy_batch(batch):
    """Keep worker transport on pipes, without torch shared-memory sockets."""
    return batch


def feature_batches(entries, *, workers=0, pin_memory=False, **options):
    dataset = FeatureBatches(entries, **options)
    kwargs = {"multiprocessing_context": "spawn", "prefetch_factor": 2} if workers else {}
    # The private generator prevents worker startup from consuming the model RNG.
    generator = torch.Generator().manual_seed(options["seed"] + options["epoch"])
    loader = DataLoader(dataset, batch_size=None, num_workers=workers,
                        collate_fn=numpy_batch, generator=generator, **kwargs)
    for batch in loader:
        tensors = {name: torch.from_numpy(value) for name, value in batch.items()}
        yield {name: value.pin_memory() for name, value in tensors.items()} if pin_memory else tensors
