"""Display-free helpers used by the Tk GUI (importable without tkinter)."""
from __future__ import annotations
from collections.abc import Iterable
from .agents import SearchLimits


def compute_limits(depth: int, time_ms: int, sims: int, family: str) -> SearchLimits:
    """Validate GUI budgets and select the node limit for the agent family."""
    if type(sims) is not int or not 1 <= sims <= 4000:
        raise ValueError("Sims は 1〜4000 の整数で指定してください。")
    limits = SearchLimits(depth, time_ms, 200000 if family == "classical" else sims)
    if limits.time_ms > 10000:
        raise ValueError("GUI search time must be at most 10000 ms")
    return limits


def unique_agent_id(base: str, existing_names: Iterable[str]) -> str:
    """Return the base name or its first available numbered suffix."""
    existing = set(existing_names)
    name, suffix = base, 2
    while name in existing:
        name = f"{base}-{suffix}"
        suffix += 1
    return name
