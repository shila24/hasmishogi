"""GameSpec adapter for standard shogi (hasami_learn/games/shogi.py).

Actions are the side-to-move policy labels of ``shogi.encoding`` (``ACTION_SIZE`` = 2188), not the engine's absolute
move codes: ``step`` maps a label to the unique legal move code and ``history_actions`` replays the engine's codes back
into labels, so records and datasets stay in adapter action space.
"""

from __future__ import annotations

import importlib

import numpy as np

from dataclasses import replace
from pathlib import Path

from hasami.agents import SearchLimits
from shogi import create_game as create_shogi_game
from shogi.agents import AlphaBetaAgent, GreedyAgent, RandomAgent
from shogi.encoding import ACTION_SIZE, ENCODING_ID, NUM_PLANES, encode, label_to_move, legal_labels, move_label
from shogi.model import REASONS, RULESET_ID, SIZE, SQUARES, Position, Rules
from shogi.notation import move_to_usi, to_sfen
from ..registry import ENCODERS, GAMES
from .base import StepOutcome

BOARD_SHAPE = (SIZE, SIZE)


def _native():
    return importlib.import_module("shogi.native")


def _native_available() -> bool:
    try:
        return bool(_native().available())
    except (ImportError, OSError, RuntimeError):
        return False


class PlaneEncoder:
    """``shogi-enc-v1``: 49 side-to-move planes of 9x9 (spec section 7)."""

    encoding_id = ENCODING_ID
    num_planes = NUM_PLANES
    action_size = ACTION_SIZE

    def __init__(self, **params) -> None:
        if params:
            raise ValueError(f"{ENCODING_ID} takes no parameters, got {sorted(params)}")

    @staticmethod
    def _player(game, player):
        turn = game.position.turn
        if player is None:
            return turn
        if player != turn:
            raise ValueError("The shogi encoder observes the side to move only")
        return player

    def encode(self, game, player=None) -> np.ndarray:
        player = self._player(game, player)
        fast = getattr(game, "encode_into", None)
        if fast is not None:
            planes = np.empty((NUM_PLANES, *BOARD_SHAPE), dtype=np.float32)
            fast(planes)
            return planes
        return np.asarray(encode(game, player), dtype=np.float32).reshape(NUM_PLANES, *BOARD_SHAPE)

    def legal_mask(self, game, player=None) -> np.ndarray:
        mask = np.zeros(ACTION_SIZE, dtype=np.float32)
        if game.outcome.terminal or (player is not None and player != game.position.turn):
            return mask
        fast = getattr(game, "legal_mask_into", None)
        if fast is not None:
            fast(mask)
            return mask
        labels = legal_labels(game)
        if labels:
            mask[np.asarray(labels, dtype=np.int64)] = 1.0
        return mask


ENCODERS.register(ENCODING_ID, PlaneEncoder)


def _position_to_sfen(position) -> str:
    if isinstance(position, str):
        return position
    if isinstance(position, dict):
        position = Position(board=tuple(position["board"]),
                            hands=tuple(tuple(hand) for hand in position["hands"]),
                            turn=position["turn"], move_number=position["move_number"])
    if not isinstance(position, Position):
        raise ValueError("shogi positions are SFEN strings, Position objects or their dicts")
    return to_sfen(position)


class LabelAgent:
    """A classical shogi agent (engine move codes) speaking the adapter's action space (policy labels)."""

    def __init__(self, agent) -> None:
        self.agent = agent

    def choose(self, game, limits, cancel=None, *, player=None):
        work = game.clone()  # owned here: closed on success, error and cancellation (native handles)
        try:
            decision = self.agent.choose(work, limits, cancel)
        finally:
            work.close()
        return replace(decision, action=move_label(game, decision.action))


@GAMES.register("shogi")
class ShogiSpec:
    game_id = "shogi"
    ruleset_id = RULESET_ID
    board_shape = BOARD_SHAPE
    action_size = ACTION_SIZE
    action_encoding = "shogi-label-2188-v1"
    default_max_plies = 320  # spec section 9; the rule's max_moves (also 320) ends games as a draw at the same ply
    default_encoder = ENCODING_ID
    num_players = 2
    perfect_information = True
    zero_sum = True
    supports_undo = True  # step() plays through the engine, whose undo() reverses it exactly
    has_chance = False
    draw_reasons = ("repetition", "max_moves")
    default_baselines = ("random", "greedy")
    default_teacher = "alphabeta:2"

    RULE_PARAMS = frozenset(("max_moves", "declaration"))

    def __init__(self, **params) -> None:
        unknown = set(params) - self.RULE_PARAMS
        if unknown:
            raise ValueError(f"shogi game params must be rule fields, got {sorted(unknown)}")
        self.rules = Rules(**params)

    def _rules(self, rules):
        if rules is None:
            return self.rules
        if isinstance(rules, Rules):
            return rules
        if isinstance(rules, dict):
            return Rules(**{**self.rules.to_dict(), **rules})
        raise ValueError("shogi rules must be a Rules object or a dict of rule fields")

    def create_game(self, backend="python", rules=None, position=None, seed=None):
        backend = self.resolve_backend(backend)
        sfen = None if position in (None, {}) else _position_to_sfen(position)
        return create_shogi_game(backend, self._rules(rules), sfen)

    def resolve_backend(self, backend):
        if backend == "python":
            return "python"
        if backend == "cpp":
            if not _native_available():
                raise RuntimeError("native shogi engine unavailable; run scripts/build_native.py")
            return "cpp"
        if backend == "auto":
            return "cpp" if _native_available() else "python"
        raise ValueError(f"unknown shogi backend {backend!r}")

    def rules_dict(self, game):
        return game.rules.to_dict()

    def actors(self, game):
        return [] if game.outcome.terminal else [game.position.turn]

    def legal_actions(self, game, player):
        return legal_labels(game) if player in self.actors(game) else []

    def observe(self, game, player):
        return PlaneEncoder().encode(game, player)

    def legal_mask(self, game, player):
        return PlaneEncoder().legal_mask(game, player)

    def step(self, game, actions):
        if set(actions) != set(self.actors(game)) or len(actions) != 1:
            raise ValueError("Joint actions must contain exactly the current shogi actor")
        game.play(label_to_move(game, next(iter(actions.values()))))
        return StepOutcome(game.outcome.terminal, False,
                           self.terminal_values(game) if game.outcome.terminal else None,
                           {"reason": game.outcome.reason})

    def terminal_values(self, game):
        if not game.outcome.terminal:
            raise ValueError("terminal_values requires a terminal game")
        winner = game.outcome.winner
        return [0.0 if winner == -1 else 1.0 if player == winner else -1.0 for player in range(self.num_players)]

    def terminal_value_for_side_to_move(self, game):
        winner = game.outcome.winner
        if not game.outcome.terminal:
            raise ValueError("terminal_value_for_side_to_move requires a terminal game")
        return 0.0 if winner == -1 else (1.0 if winner == game.position.turn else -1.0)

    def history_actions(self, game):
        """Every step's ``{player: label}``, replaying the engine's move codes from the initial position."""
        replay = create_shogi_game("python", game.rules, to_sfen(game.initial))
        try:
            history = []
            for code in game.moves:
                history.append({replay.position.turn: move_label(replay, code)})
                replay.play(code)
            return history
        finally:
            replay.close()

    def action_to_move(self, game, action):
        """The engine move code of a policy label in ``game`` (practical advisor hook)."""
        return label_to_move(game, action)

    def move_text(self, game, action):
        """USI text of a policy label in ``game`` (practical advisor and GUI hook)."""
        return move_to_usi(label_to_move(game, action))

    @staticmethod
    def action_name(action):
        """Position-free description of a label; use ``move_text`` for real notation."""
        label, square = divmod(int(action), SQUARES)
        return "win" if action == ACTION_SIZE - 1 else f"label{label}@{square}"

    def make_encoder(self, name="", **params):
        if name in ("", ENCODING_ID):
            return PlaneEncoder(**params)
        raise ValueError(f"unknown shogi encoder: {name}")

    def symmetries(self):
        return []  # the start position is not mirror-symmetric (bishop and rook)

    def baseline_agent(self, spec, seed=0):
        name, _, rest = spec.partition(":")
        if name == "random":
            return LabelAgent(RandomAgent(seed)), None
        if name == "greedy":
            return LabelAgent(GreedyAgent()), None
        if name == "alphabeta-cpp":
            from shogi.agents import native_baseline
            agent, limits = native_baseline(spec)
            return LabelAgent(agent), limits
        if name == "alphabeta":
            if ":" in spec and (not rest or any(not part for part in rest.split(":"))):
                raise ValueError("alphabeta requires a nonempty depth")
            parts = [int(x) for x in rest.split(":") if x]
            depth = parts[0] if parts else 2
            time_ms = parts[1] if len(parts) > 1 else 1000
            return LabelAgent(AlphaBetaAgent()), SearchLimits(max_depth=depth, time_ms=time_ms, node_limit=2_000_000)
        raise ValueError(f"unknown shogi baseline {spec}; use random, greedy or alphabeta:<depth>[:<ms>]")

    def save_record(self, game, path):
        """Write ``shogi/records.py``'s lossless JSON format (spec section 9); resignations round-trip."""
        from shogi.records import save_record

        path = Path(path)
        if path.suffix.lower() != ".json":
            raise ValueError("learning records of shogi games are JSON (.json)")
        save_record(getattr(game, "game", game), path)
        return path

    def load_record(self, path, backend="python"):
        from shogi.records import load_record

        game = load_record(path)
        if self.resolve_backend(backend) == "python":
            return game
        native = self.create_game(backend, game.rules, game.initial)
        try:
            for code in game.moves:
                native.play(code)
            if game.outcome.reason == "resign":
                native.resign()
            return native
        except BaseException:
            native.close()
            raise

    def validate_result_reasons(self, data):
        reasons = data["reasons"]
        if any(key not in (*REASONS[1:], "max_plies") for key in reasons):
            raise ValueError("Invalid saved pair result reasons")
