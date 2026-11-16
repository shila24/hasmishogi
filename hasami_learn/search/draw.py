"""Shared draw utility for search leaves and training targets; no rule changes."""

SEARCH_SEMANTICS_VERSION = 2


def draw_reasons_for(spec, reasons) -> tuple[str, ...]:
    """Only reasons the adapter identifies as eligible draws receive contempt."""
    return tuple(reason for reason in reasons if reason in getattr(spec, "draw_reasons", ()))


def draw_utility(winner: int, reason: str, value: float, reasons) -> float | None:
    """Return symmetric utility for a selected draw, or None for other outcomes."""
    return float(value) if winner == -1 and reason in reasons else None
