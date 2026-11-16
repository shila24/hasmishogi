"""Catalogue of every registry, for the training CLI.

Kept out of ``hasami_learn.registry`` because it imports every learner; runtime code (inference, the advisor, the
GUIs) imports the registry and must not pull the learners in (docs/CODE_GUIDE_ja.md, import closure).
"""

from __future__ import annotations

from .registry import ALGORITHMS, ENCODERS, GAMES, NETWORKS, OPPONENT_STRATEGIES


def describe_all() -> dict[str, list[str]]:
    """Return every registry's names; algorithms are imported lazily first."""
    from . import algorithms  # noqa: F401  (registers algorithm modules)
    from . import networks  # noqa: F401
    from . import encoding  # noqa: F401
    from . import opponents  # noqa: F401

    return {
        "games": GAMES.names(),
        "algorithms": ALGORITHMS.names(),
        "networks": NETWORKS.names(),
        "encoders": ENCODERS.names(),
        "opponent_strategies": OPPONENT_STRATEGIES.names(),
        "algorithm_import_errors": {k: str(v) for k, v in algorithms.IMPORT_ERRORS.items()},
    }
