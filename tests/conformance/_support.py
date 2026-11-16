"""Shared deterministic fixtures and thin adapters to the independent oracle."""

from collections import Counter
from contextlib import closing
import random

from hasami import create_game
from hasami.model import GameLike, Position
from hasami.records import record_dict
from tests import oracle
from tests.test_rules import mask, native_available, setup


def available_backends() -> tuple[str, ...]:
    """Use the existing guard, including its required-native failure behavior."""
    return ("python", "cpp") if native_available() else ("python",)


def random_position(rng: random.Random, turn: int | None = None) -> Position:
    """Generate a validated, ongoing default-rules setup with varied material."""
    for _ in range(1000):
        black_count, white_count = rng.randint(5, 9), rng.randint(5, 9)
        squares = rng.sample(range(81), black_count + white_count)
        lead = black_count - white_count
        pending = (0 if lead > 0 else 1) if abs(lead) >= 3 else -1
        try:
            position = setup(
                squares[:black_count], squares[black_count:],
                rng.randrange(2) if turn is None else turn, pending,
            )
            with closing(create_game("python", position=position)) as game:
                if not game.outcome.terminal:
                    return position
        except ValueError:
            continue
    raise AssertionError("Could not generate a consistent ongoing position")


def oracle_successor(position: Position, action: int, edge: bool = True):
    """Adapt the array oracle's board and capture set to the public result format."""
    board, taken = oracle.apply(oracle.array_board(position), position.turn, action, edge)
    return board, mask(taken)


def game_snapshot(game: GameLike) -> dict:
    """Copy all observable state needed to detect mutation and undo drift."""
    position = game.position
    history = game.history_keys()
    return {
        "record": record_dict(game),
        "position": position,
        "outcome": game.outcome,
        "captures": (position.captures(0), position.captures(1)),
        "last_capture": game.last_capture,
        "history": history,
        "counts": Counter(history),
        "legal": game.legal_actions(),
    }


def swap_players(position: Position) -> Position:
    """Exchange player identities without changing square or action coordinates."""
    return Position(
        position.white, position.black, 1 - position.turn,
        -1 if position.pending == -1 else 1 - position.pending,
    )


def perft(game: GameLike, depth: int) -> int:
    """Count exact-depth paths through the public API, restoring the full game."""
    if depth == 0:
        return 1
    actions = game.legal_actions()
    if depth == 1:
        return len(actions)
    count = 0
    for action in actions:
        game.play(action)
        try:
            count += perft(game, depth - 1)
        finally:
            if not game.undo():
                raise AssertionError("Perft could not undo a played move")
    return count
