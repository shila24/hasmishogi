"""Contracts for player observations, joint decisions, and game adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable
import numpy as np
from hasami.agents import Agent, SearchLimits
from hasami.model import GameLike
from ..signature_cache import accepts


class Encoder(Protocol):
    encoding_id: str
    num_planes: int
    def encode(self, game: GameLike, player: int | None = None) -> np.ndarray: ...
    def legal_mask(self, game: GameLike, player: int | None = None) -> np.ndarray: ...


@dataclass(frozen=True)
class StepOutcome:
    """A joint step result; final rewards retain fixed seat identity."""

    terminal: bool
    truncated: bool = False
    rewards: list[float] | None = None
    info: dict[str, Any] = field(default_factory=dict)


class StepLike(Protocol):
    rewards: list[float]
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class TurnBasedEnvLike(Protocol):
    game: GameLike
    truncated: bool
    def step(self, action: int) -> StepLike: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class Symmetry:
    """A rule-preserving transform applied together to all sample fields."""
    name: str
    planes: Callable[[np.ndarray], np.ndarray]
    policy: Callable[[np.ndarray], np.ndarray]
    action: Callable[[int], int]


@runtime_checkable
class GameSpec(Protocol):
    """Game plug-in contract (v2).

    Optional class attributes (defaults apply when absent):

    - ``zero_sum`` (True): declare ``False`` for games whose terminal values need not sum to zero;
      alternating search (MCTS/AlphaZero) refuses them.
    - ``supports_undo`` (False): declare ``True`` when ``game.undo()`` exactly reverses one
      ``step`` call, so tree search may descend with step/undo instead of cloning per simulation.
    - ``has_chance`` (True): declare ``False`` for games without chance events. A chance game is
      replayable only from its creation seed, so records and dataset extraction of a live game
      require a recoverable seed (``game.seed``, the trajectory, or a loaded record's metadata).
    - ``history_actions(game) -> list[dict[int, int]]`` (absent): the complete decision history in
      adapter action space. Required when the engine's scalar ``moves`` are not the adapter's action
      ids (``step`` translates them); records and datasets otherwise treat scalar moves as actions
      and refuse histories that replay differently.
    """

    game_id: str
    ruleset_id: str
    board_shape: tuple[int, int]
    action_size: int
    default_max_plies: int
    num_players: int
    perfect_information: bool
    def create_game(self, backend: str = "python", rules: dict | None = None,
                    position: dict | None = None, seed: int | None = None) -> GameLike: ...
    def actors(self, game: GameLike) -> list[int]: ...
    def legal_actions(self, game: GameLike, player: int) -> list[int]: ...
    def observe(self, game: GameLike, player: int) -> np.ndarray: ...
    def legal_mask(self, game: GameLike, player: int) -> np.ndarray: ...
    def step(self, game: GameLike, actions: dict[int, int]) -> StepOutcome: ...
    def terminal_values(self, game: GameLike) -> list[float]: ...
    def resolve_backend(self, spec: str) -> str: ...
    def rules_dict(self, game: GameLike) -> dict: ...
    def make_encoder(self, name: str, **params) -> Encoder: ...
    def symmetries(self) -> list[Symmetry]: ...
    def baseline_agent(self, spec: str, seed: int = 0) -> tuple[Agent, SearchLimits | None]: ...
    def action_name(self, action: int) -> str: ...


def require_search_game(spec) -> None:
    """Alternating zero-sum search requires two players, perfect information, and zero-sum values."""
    if spec.num_players != 2 or not spec.perfect_information:
        raise ValueError("MCTS/AlphaZero requires a two-player perfect-information game")
    if getattr(spec, "zero_sum", True) is False:
        raise ValueError("MCTS/AlphaZero back up values by negation per ply, so they require a zero-sum game; "
                         f"game {getattr(spec, 'game_id', '?')} declares zero_sum=False")


class LegacySpec:
    """Preserve explicitly registered v1 adapters through the v2 boundary."""

    num_players = 2
    perfect_information = True
    zero_sum = True
    supports_undo = True  # step() is the engine's own play(), which undo() reverses
    has_chance = False  # v1 adapters take no seed: their games are deterministic given the history

    def __init__(self, spec):
        self.original = spec

    def __getattr__(self, name):
        return getattr(self.original, name)

    def create_game(self, backend="python", rules=None, position=None, seed=None):
        return self.original.create_game(backend, rules, position)

    def actors(self, game):
        return [] if game.outcome.terminal else [game.position.turn]

    def legal_actions(self, game, player):
        return game.legal_actions() if player in self.actors(game) else []

    def legal_mask(self, game, player):
        mask = np.zeros(self.action_size, dtype=np.float32)
        mask[self.legal_actions(game, player)] = 1
        return mask

    def observe(self, game, player):
        return self.make_encoder("").encode(game)

    def step(self, game, actions):
        if set(actions) != set(self.actors(game)) or len(actions) != 1:
            raise ValueError("Joint actions must match the current actors")
        game.play(next(iter(actions.values())))
        return StepOutcome(game.outcome.terminal, False,
                           self.terminal_values(game) if game.outcome.terminal else None,
                           {"reason": game.outcome.reason})

    def terminal_values(self, game):
        if not game.outcome.terminal:
            raise ValueError("terminal_values requires a terminal game")
        return [0.0 if game.outcome.winner == -1 else 1.0 if p == game.outcome.winner else -1.0
                for p in range(self.num_players)]


def as_v2(spec):
    return spec if hasattr(spec, "num_players") else LegacySpec(spec)


def encode_player(encoder, game, player):
    """Support legacy encoders without catching exceptions from inside encode.

    Whether ``encode`` accepts a player is cached (``hasami_learn.signature_cache``; ``inspect.signature`` costs
    ~30 us and runs for every decision).
    """
    method = encoder.encode
    return method(game, player) if accepts(method, "encode", (game, player), {}) else method(game)


def terminal_rewards(spec, game):
    """Validate the adapter's final fixed-player value vector."""
    values = [float(value) for value in spec.terminal_values(game)]
    if len(values) != spec.num_players or not all(np.isfinite(v) and -1 <= v <= 1 for v in values):
        raise ValueError("terminal_values must provide one finite value in [-1, 1] per player")
    return values
