"""Import every algorithm module so they register with ALGORITHMS.

A module that fails to import (missing optional piece, syntax error while a
worker is still editing it) is recorded in ``IMPORT_ERRORS`` instead of
breaking the whole package.
"""

from __future__ import annotations

import importlib

from .base import CollectStats, Learner, LearnerContext, UpdateStats, build_optimizer

MODULES = ("reinforce", "alphazero", "dqn", "ppo", "supervised", "adversarial", "ppo_league", "ismcts", "practical")
IMPORT_ERRORS: dict[str, Exception] = {}

for _name in MODULES:
    try:
        importlib.import_module(f"{__name__}.{_name}")
    except Exception as exc:  # noqa: BLE001 - keep other algorithms usable
        IMPORT_ERRORS[_name] = exc

__all__ = ["CollectStats", "Learner", "LearnerContext", "UpdateStats", "build_optimizer", "IMPORT_ERRORS", "MODULES"]
