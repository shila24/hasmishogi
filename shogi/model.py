"""Immutable values and absolute move codes for shogi-std-v1."""
from dataclasses import asdict, dataclass

SIZE = 9
SQUARES = 81
P, L, N, S, G, B, R, K = range(1, 9)
PP, PL, PN, PS, PB, PR = range(9, 15)
SENTE, GOTE = 0, 1
HAND_TYPES = (P, L, N, S, G, B, R)
PROMOTED = {P: PP, L: PL, N: PN, S: PS, B: PB, R: PR}
UNPROMOTED = {value: key for key, value in PROMOTED.items()}
PIECE_LIMITS = (18, 4, 4, 4, 4, 2, 2)
DROP_BASE = 13122
DECLARE = 13689
MOVE_SPACE = 13690
RULESET_ID = "shogi-std-v1"
# Implementation limits shared with the native engine's 32-bit ABI fields (shogi/model.py):
# far beyond any real game, they keep every move number exactly representable.
MAX_SFEN_MOVE_NUMBER = 1_000_000
MAX_GAME_MOVES = 1_000_000
START_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
REASONS = ("ongoing", "checkmate", "no_legal_moves", "declaration", "perpetual_check",
           "repetition", "max_moves", "resign")


def require_int(value: object, name: str, lower: int, upper: int | None = None) -> int:
    if type(value) is not int or value < lower or (upper is not None and value > upper):
        raise ValueError(f"Invalid {name}")
    return value


def encode_move(from_sq: int, to_sq: int, promote: bool = False) -> int:
    require_int(from_sq, "source", 0, 80)
    require_int(to_sq, "destination", 0, 80)
    if type(promote) is not bool:
        raise ValueError("Promotion must be boolean")
    return (from_sq * 81 + to_sq) * 2 + int(promote)


def encode_drop(hand_index: int, to_sq: int) -> int:
    require_int(hand_index, "hand index", 0, 6)
    require_int(to_sq, "destination", 0, 80)
    return DROP_BASE + hand_index * 81 + to_sq


@dataclass(frozen=True)
class Move:
    kind: str
    from_sq: int = -1
    to_sq: int = -1
    promote: bool = False
    piece: int = -1


def decode_move(code: int) -> Move:
    require_int(code, "move code", 0, DECLARE)
    if code == DECLARE:
        return Move("declare")
    if code >= DROP_BASE:
        hand, target = divmod(code - DROP_BASE, 81)
        return Move("drop", to_sq=target, piece=HAND_TYPES[hand])
    source, target = divmod(code // 2, 81)
    return Move("move", source, target, bool(code % 2))


@dataclass(frozen=True)
class Rules:
    max_moves: int = 320
    declaration: str = "csa27"

    def __post_init__(self):
        require_int(self.max_moves, "max_moves", 0, MAX_GAME_MOVES)
        if self.declaration not in ("csa27", "none"):
            raise ValueError("Invalid declaration rule")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Rules":
        if not isinstance(data, dict) or set(data) != {"max_moves", "declaration"}:
            raise ValueError("Invalid rules object")
        return cls(**data)


@dataclass(frozen=True)
class Position:
    board: tuple[int, ...]
    hands: tuple[tuple[int, ...], tuple[int, ...]]
    turn: int
    move_number: int

    def __post_init__(self):
        if not isinstance(self.board, tuple) or len(self.board) != 81:
            raise ValueError("Board must be an immutable tuple of 81 cells")
        if any(type(p) is not int or p not in (0, *range(1, 15), *range(17, 31)) for p in self.board):
            raise ValueError("Invalid board piece")
        if (not isinstance(self.hands, tuple) or len(self.hands) != 2
                or any(not isinstance(h, tuple) or len(h) != 7 for h in self.hands)):
            raise ValueError("Hands must be two immutable seven-count tuples")
        for hand in self.hands:
            for count, limit in zip(hand, PIECE_LIMITS):
                require_int(count, "hand count", 0, limit)
        require_int(self.turn, "turn", 0, 1)
        require_int(self.move_number, "move number", 1, MAX_SFEN_MOVE_NUMBER + MAX_GAME_MOVES)

    def to_dict(self) -> dict:
        """JSON-serialisable form (used by learning records); ``Position(**...)`` with tuples restores it."""
        return {"board": list(self.board), "hands": [list(hand) for hand in self.hands],
                "turn": self.turn, "move_number": self.move_number}

    def key(self) -> str:
        from .notation import position_key
        return position_key(self)

    def sfen(self) -> str:
        return f"{self.key()} {self.move_number}"


@dataclass(frozen=True)
class Outcome:
    winner: int = -2
    reason: str = "ongoing"

    def __post_init__(self):
        require_int(self.winner, "winner", -2, 1)
        if self.reason not in REASONS or (self.winner == -2) != (self.reason == "ongoing"):
            raise ValueError("Invalid outcome")
        if (self.winner == -1) != (self.reason in ("repetition", "max_moves")):
            raise ValueError("Invalid draw outcome")

    @property
    def terminal(self) -> bool:
        return self.winner != -2


@dataclass(frozen=True)
class MoveResult:
    """A played code, original captured board cell (zero if none), and check flag."""
    code: int
    captured: int
    check: bool
