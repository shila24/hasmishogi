"""Append-only JSONL metrics following hasami_learn/metrics.py."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ALLOWED_KEYS = {
    "seconds", "games", "plies", "wins", "truncated", "steps", "loss", "policy_loss", "value_loss",
    "replay_size", "opponent", "win_rate", "elo_delta", "checkpoint", "extra", "terminal_reasons",
}
KINDS = {"collect", "update", "evaluation", "checkpoint", "opponent_pool", "resume", "info"}


def _plain(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, (list, tuple, dict)):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive
            return str(value)
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    return value


class MetricsLogger:
    def __init__(self, path: Path | str, run_id: str, device: str = "", echo: bool = True) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.device = device
        self.echo = echo
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def log(self, kind: str, iteration: int, **fields: Any) -> dict[str, Any]:
        if kind not in KINDS:
            raise ValueError(f"unknown metrics kind {kind}")
        row: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "run_id": self.run_id,
            "iteration": int(iteration),
            "kind": kind,
        }
        if self.device:
            row["device"] = self.device
        extra: dict[str, Any] = {}
        for key, value in fields.items():
            if key in ALLOWED_KEYS and key != "extra":
                row[key] = _plain(value)
            elif key == "extra" and isinstance(value, dict):
                extra.update(_plain(value))
            else:
                extra[key] = _plain(value)
        if "wins" in row and len(row["wins"]) != 3:
            extra["seat_wins"] = row.pop("wins")
        if extra:
            row["extra"] = extra
        self._handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._handle.flush()
        if self.echo:
            summary = {k: v for k, v in row.items() if k not in ("timestamp", "run_id", "device")}
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        return row

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
