"""Hasami adapter: engine, encoding, symmetries, and classical baselines."""

from __future__ import annotations

import numpy as np
from hasami import create_game
from hasami.model import ACTION_SIZE, RULESET_ID, GameLike, Rules, Position, move_name as action_name
from hasami.env import TurnBasedEnv
from hasami.agents import Agent, SearchLimits, RandomAgent, GreedyAgent, AlphaBetaAgent
from ..registry import ENCODERS, GAMES
from .base import Symmetry, StepOutcome

ENCODING_ID = "hasami-enc-v1"
NUM_PLANES = 9
BOARD_BYTES = 11  # 81 bits fit in 11 little-endian bytes

FLIP_SQUARE = np.array([(s // 9) * 9 + (8 - s % 9) for s in range(81)], dtype=np.int64)
FLIP_ACTION = np.array([FLIP_SQUARE[a // 81] * 81 + FLIP_SQUARE[a % 81] for a in range(ACTION_SIZE)], dtype=np.int64)


def board_to_plane(board: int) -> np.ndarray:
    """Convert an 81-bit integer bitboard into a (9, 9) float32 plane."""
    raw = np.frombuffer(int(board).to_bytes(BOARD_BYTES, "little"), dtype=np.uint8)
    bits = np.unpackbits(raw, bitorder="little")[:81]
    return bits.reshape(9, 9).astype(np.float32)


class PlaneEncoder:
    """Nine 9x9 planes: mine, theirs, pending flags, turn, captures, repetition, bias."""

    encoding_id = ENCODING_ID
    num_planes = NUM_PLANES
    action_size = ACTION_SIZE

    def __init__(self, **params) -> None:
        if params:
            raise ValueError(f"{ENCODING_ID} takes no parameters, got {sorted(params)}")

    def encode(self, game: GameLike, player: int | None = None) -> np.ndarray:
        p = game.position
        if player is not None and player != p.turn:
            raise ValueError("Hasami encoding requires the current actor")
        mine, theirs = p.pieces(p.turn), p.pieces(1 - p.turn)
        planes = np.zeros((NUM_PLANES, 9, 9), dtype=np.float32)
        planes[0] = board_to_plane(mine)
        planes[1] = board_to_plane(theirs)
        planes[2] = 1.0 if p.pending == p.turn else 0.0
        planes[3] = 1.0 if p.pending == 1 - p.turn else 0.0
        planes[4] = float(p.turn)
        planes[5] = p.captures(p.turn) / 9.0
        planes[6] = p.captures(1 - p.turn) / 9.0
        limit = game.rules.repetition_limit
        if limit:
            planes[7] = game.history_keys().count(p.key()) / float(limit)
        planes[8] = 1.0
        return planes

    def legal_mask(self, game: GameLike, player: int | None = None) -> np.ndarray:
        mask = np.zeros(ACTION_SIZE, dtype=np.float32)
        legal = game.legal_actions() if player is None or player == game.position.turn else []
        if legal:
            mask[np.asarray(legal, dtype=np.int64)] = 1.0
        return mask

    @staticmethod
    def flip_horizontal(planes: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(planes[..., ::-1])

    @staticmethod
    def flip_horizontal_policy(vector: np.ndarray) -> np.ndarray:
        out = np.empty_like(vector)
        out[FLIP_ACTION] = vector
        return out

    @staticmethod
    def flip_horizontal_action(action: int) -> int:
        return int(FLIP_ACTION[int(action)])


ENCODERS.register(ENCODING_ID, PlaneEncoder)


def terminal_value_for_side_to_move(game: GameLike) -> float:
    """+1 if the player to move at a terminal position is the winner, -1 if the loser, 0 draw."""
    outcome = game.outcome
    if not outcome.terminal:
        raise ValueError("terminal_value_for_side_to_move requires a terminal game")
    if outcome.winner == -1:
        return 0.0
    return 1.0 if outcome.winner == game.position.turn else -1.0



@GAMES.register("hasami")
class HasamiSpec:
    num_players = 2
    perfect_information = True
    draw_reasons = ("repetition",)
    default_baselines = ("random", "greedy")
    zero_sum = True  # terminal values are +1/-1 or 0/0; MCTS relies on this
    supports_undo = True  # step() plays through the engine, whose undo() reverses it exactly
    has_chance = False  # no chance events: a record replays from the initial position and moves alone
    default_teacher = "alphabeta:2"
    game_id = "hasami"
    ruleset_id = RULESET_ID
    board_shape = (9, 9)
    action_size = ACTION_SIZE
    action_encoding = "from*81+to"
    default_max_plies = 300
    default_encoder = ENCODING_ID

    def __init__(self, **params):
        if params:
            raise ValueError(f"hasami takes no parameters, got {sorted(params)}")

    def create_game(self, backend="python", rules=None, position=None, seed=None):
        if isinstance(rules, dict):
            rules = Rules(**rules)
        if isinstance(position, dict):
            position = Position.from_dict(position)
        return create_game(backend, rules, position)

    def actors(self, game):
        return [] if game.outcome.terminal else [game.position.turn]

    def legal_actions(self, game, player):
        return game.legal_actions() if player in self.actors(game) else []

    def observe(self, game, player):
        return PlaneEncoder().encode(game, player)

    def legal_mask(self, game, player):
        return PlaneEncoder().legal_mask(game, player)

    def step(self, game, actions):
        if set(actions) != set(self.actors(game)) or len(actions) != 1:
            raise ValueError("Joint actions must contain exactly the current Hasami actor")
        game.play(next(iter(actions.values())))
        return StepOutcome(game.outcome.terminal, False,
                           self.terminal_values(game) if game.outcome.terminal else None,
                           {"reason": game.outcome.reason})

    def terminal_values(self, game):
        if not game.outcome.terminal:
            raise ValueError("terminal_values requires a terminal game")
        return [0.0 if game.outcome.winner == -1 else 1.0 if p == game.outcome.winner else -1.0
                for p in range(self.num_players)]

    def resolve_backend(self, spec):
        if spec in ("python", "cpp"):
            return spec
        if spec != "auto":
            raise ValueError(f"unknown backend {spec}")
        try:
            game = self.create_game("cpp")
            game.close()
            return "cpp"
        except (RuntimeError, ValueError, OSError):
            return "python"

    def rules_dict(self, game):
        return game.rules.to_dict()

    def make_encoder(self, name="", **params):
        return ENCODERS.get(name or self.default_encoder)(**params)

    terminal_value_for_side_to_move = staticmethod(terminal_value_for_side_to_move)
    action_name = staticmethod(action_name)

    def symmetries(self):
        return [Symmetry("horizontal", PlaneEncoder.flip_horizontal,
                         PlaneEncoder.flip_horizontal_policy, PlaneEncoder.flip_horizontal_action)]

    def baseline_agent(self, spec, seed=0):
        name, _, rest = spec.partition(":")
        if name == "random":
            return RandomAgent(seed), None
        if name == "greedy":
            return GreedyAgent(), None
        if name == "alphabeta":
            if ":" in spec and (not rest or any(not part for part in rest.split(":"))):
                raise ValueError("alphabeta requires a nonempty depth")
            parts = [int(x) for x in rest.split(":") if x]
            depth = parts[0] if parts else 2
            time_ms = parts[1] if len(parts) > 1 else 200
            return AlphaBetaAgent(), SearchLimits(max_depth=depth, time_ms=time_ms, node_limit=200000)
        raise ValueError(f"unknown baseline spec {spec}; use random, greedy or alphabeta:<depth>")

    def env(self, backend="python", max_plies=None, rules=None):
        if isinstance(rules, dict):
            rules = Rules(**rules)
        return TurnBasedEnv(backend, rules, self.default_max_plies if max_plies is None else max_plies)

    def save_record(self, game, path):
        from hasami.records import save_record
        return save_record(game, path)

    def load_record(self, path, backend):
        from hasami.records import load_record
        return load_record(path, backend)

    def validate_result_reasons(self, data):
        from hasami.model import REASONS
        reasons = data["reasons"]
        if (any(key not in (*REASONS[1:], "max_plies") for key in reasons)
                or reasons.get("repetition", 0) + data["truncated"] > data["draws"]
                or reasons.get("five_captures", 0) + reasons.get("lead_confirmed", 0)
                > data["wins"] + data["losses"]):
            raise ValueError("Invalid saved pair result reasons")


def augment_horizontal(encoder: PlaneEncoder, planes: np.ndarray, mask: np.ndarray, policy: np.ndarray | None, action: int | None):
    """Mirror one sample left-right; returns (planes, mask, policy, action)."""
    flipped_policy = None if policy is None else encoder.flip_horizontal_policy(policy)
    flipped_action = None if action is None else encoder.flip_horizontal_action(action)
    return encoder.flip_horizontal(planes), encoder.flip_horizontal_policy(mask), flipped_policy, flipped_action
