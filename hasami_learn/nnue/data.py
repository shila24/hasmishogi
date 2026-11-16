"""Streaming hlnn1 shards; fixed little-endian records and verified JSON sidecars."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile

from shogi.model import Position, SQUARES, require_int
from .features import validate_position

MAGIC = b"hlnn1\0\0\0"
RECORD = struct.Struct("<81B14BBIiHQHbIIfQ")
HEADER = struct.Struct("<8sII")
IN_CHECK, CAPTURE_BEST, TRUNCATED = 1, 2, 4
FLAGS_MASK = 0x00FFFF07


@dataclass(frozen=True)
class Record:
    position: Position
    score: int
    depth: int
    nodes: int
    best_move: int
    result: int
    ply: int
    game_id: int
    flags: int = 0
    teacher: float = math.nan

    def pack(self):
        validate_position(self.position)
        for name, low, high in (("score", -2**31, 2**31-1), ("depth", 0, 65535),
                                ("nodes", 0, 2**64-1), ("best_move", 0, 13689),
                                ("result", -1, 1), ("ply", 0, 1_000_000),
                                ("game_id", 0, 2**64-1), ("flags", 0, FLAGS_MASK)):
            require_int(getattr(self, name), name, low, high)
        if self.flags & ~FLAGS_MASK:
            raise ValueError("Reserved hlnn1 flags")
        if not math.isnan(self.teacher) and not (math.isfinite(self.teacher) and -1 <= self.teacher <= 1):
            raise ValueError("Teacher must be a value in [-1,1] or NaN")
        p = self.position
        data = bytearray(RECORD.pack(*p.board, *p.hands[0], *p.hands[1], p.turn, p.move_number,
                                    self.score, self.depth, self.nodes, self.best_move, self.result,
                                    self.ply, self.flags, self.teacher, self.game_id))
        if math.isnan(self.teacher):
            data[125:129] = struct.pack("<I", 0x7FC00000)
        return bytes(data)

    @classmethod
    def unpack(cls, data):
        values = RECORD.unpack(data)
        position = Position(tuple(values[:SQUARES]), (tuple(values[SQUARES:SQUARES+7]), tuple(values[88:95])), values[95], values[96])
        score, depth, nodes, best, result, ply, flags, teacher, game_id = values[97:]
        record = cls(position, score, depth, nodes, best, result, ply, game_id, flags, teacher)
        if record.pack() != data:
            raise ValueError("Noncanonical hlnn1 record")
        return record


def sync_directory(path):
    """Persist rename metadata where directory fsync is available."""
    if hasattr(os, "O_DIRECTORY"):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic_write(path, content):
    """Publish a fully flushed file by replacement in its destination directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_shard(path, records):
    """Stream records; publish a sidecar last. Existing shards are never overwritten."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if Path(str(path) + ".json").exists():
        raise FileExistsError("hlnn1 sidecar already exists")
    count, games = 0, set()
    with path.open("xb") as stream:
        stream.write(HEADER.pack(MAGIC, 1, RECORD.size))
        for record in records:
            stream.write(record.pack())
            count += 1
            games.add(record.game_id)
        stream.flush()
        os.fsync(stream.fileno())
    manifest = {"format": "hlnn1", "version": 1, "record_size": RECORD.size, "records": count,
                "games": len(games), "sha256": digest_file(path), "bytes": path.stat().st_size}
    atomic_write(str(path) + ".json", (json.dumps(manifest, sort_keys=True) + "\n").encode())
    return manifest


def _unique_fields(pairs):
    fields = {}
    for key, value in pairs:
        if key in fields:
            raise ValueError("Duplicate hlnn1 sidecar field")
        fields[key] = value
    return fields


def verify_shard(path):
    """Verify immutable shard bytes and sidecar before any records are consumed."""
    path = Path(path)
    sidecar = Path(str(path) + ".json")
    if sidecar.stat().st_size > 4096:
        raise ValueError("Oversized hlnn1 sidecar")
    manifest = json.loads(sidecar.read_text(encoding="utf-8"), object_pairs_hook=_unique_fields)
    expected = {"format", "version", "record_size", "records", "games", "sha256", "bytes"}
    if (not isinstance(manifest, dict) or set(manifest) != expected or manifest["format"] != "hlnn1"
            or manifest["version"] != 1 or manifest["record_size"] != RECORD.size
            or any(type(manifest[k]) is not int or manifest[k] < 0 for k in ("version", "record_size", "records", "games", "bytes"))
            or manifest["games"] > manifest["records"]
            or manifest["bytes"] != HEADER.size + manifest["records"] * RECORD.size
            or manifest["bytes"] != path.stat().st_size or manifest["sha256"] != digest_file(path)):
        raise ValueError("hlnn1 sidecar/count/sha256 mismatch")
    return manifest


def read_shard(path, *, verify=True):
    """Verify the whole shard before yielding, then decode one record at a time."""
    path = Path(path)
    manifest = verify_shard(path) if verify else None
    games, count = set(), 0
    with path.open("rb") as stream:
        if stream.read(HEADER.size) != HEADER.pack(MAGIC, 1, RECORD.size):
            raise ValueError("hlnn1 header mismatch")
        while data := stream.read(RECORD.size):
            if len(data) != RECORD.size:
                raise ValueError("Truncated hlnn1 record")
            record = Record.unpack(data)
            count += 1
            games.add(record.game_id)
            yield record
    if verify and (manifest["records"] != count or manifest["games"] != len(games)):
        raise ValueError("hlnn1 record/game count mismatch")
