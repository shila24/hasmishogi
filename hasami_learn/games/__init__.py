"""Explicitly registered game adapters; never discover or import external plugins."""

from .base import Encoder, GameSpec, Symmetry, StepOutcome, as_v2
from ..registry import GAMES
from . import hasami  # Register the built-in adapter.
from . import riichi  # Register the built-in four-player adapter.
from . import shogi  # Register the built-in standard shogi adapter.

# Adapters still being built (poker adapter poker, werewolf adapter werewolf) register when their module exists; any other import
# error inside them still propagates.
import importlib as _importlib

for _optional in ("poker", "werewolf"):
    try:
        _importlib.import_module(f"{__name__}.{_optional}")
    except ModuleNotFoundError as _exc:
        if _exc.name != f"{__name__}.{_optional}":
            raise


def get_spec(name: str = "hasami", params: dict | None = None) -> GameSpec:
    return as_v2(GAMES.get(name)(**(params or {})))


def default_rules(spec: GameSpec) -> dict:
    game = spec.create_game()
    try:
        return spec.rules_dict(game)
    finally:
        game.close()


def action_encoding(spec: GameSpec) -> str:
    return getattr(spec, "action_encoding", "index")
