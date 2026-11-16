"""Versioned JSON records, replay validation, and atomic saves."""
from __future__ import annotations
import json
import os
from pathlib import Path
import tempfile
from . import create_game
from .model import Rules, Position, RULESET_ID, GameLike

MAX_RECORD_BYTES = 8 * 1024 * 1024
MAX_RECORD_PLIES = 20000


def record_dict(game: GameLike) -> dict:
    return {
        "format": "hasami-record-v1",
        "ruleset_id": RULESET_ID,
        "rules": game.rules.to_dict(),
        "initial": game.initial.to_dict(),
        "moves": game.moves.copy(),
        "final": {"position": game.position.to_dict(), "winner": game.outcome.winner,
                  "reason": game.outcome.reason},
    }


def from_record(data: dict, backend="python"):
    if (not isinstance(data, dict) or data.get("format") != "hasami-record-v1"
            or data.get("ruleset_id") != RULESET_ID):
        raise ValueError("Record format or ruleset version mismatch")
    expected = {"format", "ruleset_id", "rules", "initial", "moves", "final"}
    if set(data) != expected:
        raise ValueError("Unexpected record fields")
    if not isinstance(data["rules"], dict) or set(data["rules"]) != set(Rules().to_dict()):
        raise ValueError("Invalid rule fields")
    moves = data["moves"]
    if not isinstance(moves, list) or len(moves) > MAX_RECORD_PLIES:
        raise ValueError("Invalid or oversized move sequence")
    game = create_game(backend, Rules(**data["rules"]), Position.from_dict(data["initial"]))
    try:
        for index, action in enumerate(moves):
            try:
                game.play(action)
            except (ValueError, RuntimeError) as exc:
                raise ValueError(f"Invalid record move at ply {index + 1}: {exc}") from exc
        if record_dict(game)["final"] != data["final"]:
            raise ValueError("Record final state does not match replay")
        return game
    except Exception:
        game.close()
        raise


def save_record(game: GameLike, path: str | Path):
    target = Path(path)
    payload = json.dumps(record_dict(game), ensure_ascii=False, indent=2)
    if len(game.moves) > MAX_RECORD_PLIES or len(payload.encode("utf-8")) > MAX_RECORD_BYTES:
        raise ValueError("Record exceeds the interchange size limit")
    name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent,
                                         prefix=".hasami-", suffix=".tmp", delete=False) as stream:
            name = stream.name
            stream.write(payload + "\n")
        os.replace(name, target)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def load_record(path: str | Path, backend="python"):
    target = Path(path)
    if target.stat().st_size > MAX_RECORD_BYTES:
        raise ValueError("Record file is too large")
    data = json.loads(target.read_text(encoding="utf-8"))
    return from_record(data, backend)
