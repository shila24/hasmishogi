"""Shared value objects, validation, and stable wire conventions."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Protocol

SIZE = 9
SQUARES = 81
ACTION_SIZE = 6561
BOARD_MASK = (1 << SQUARES) - 1
LEFT = sum(1 << (9 * r) for r in range(9))
RIGHT = LEFT << 8
EDGE = LEFT | RIGHT | 511 | (511 << 72)
RULESET_ID = "hasami-lab-v1"
REASONS = ("ongoing", "five_captures", "lead_confirmed", "repetition", "no_moves")


def require_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def bits(board: int):
    """Yield occupied square indices in ascending order."""
    while board:
        low = board & -board
        yield low.bit_length() - 1
        board ^= low


def neighbours(board: int) -> int:
    return (((board & ~RIGHT) << 1) | ((board & ~LEFT) >> 1)
            | (board << 9) | (board >> 9)) & BOARD_MASK


def make_ray(square: int, dr: int, dc: int) -> tuple[int, ...]:
    r, c = divmod(square, 9)
    out = []
    r, c = r + dr, c + dc
    while 0 <= r < 9 and 0 <= c < 9:
        out.append(r * 9 + c)
        r, c = r + dr, c + dc
    return tuple(out)


RAYS = tuple(tuple(make_ray(s, dr, dc) for dr, dc in
                   ((-1, 0), (1, 0), (0, -1), (0, 1))) for s in range(81))


def encode_action(source: int, target: int) -> int:
    require_int(source, "source")
    require_int(target, "target")
    if not (0 <= source < 81 and 0 <= target < 81):
        raise ValueError("Square index must be in [0, 80]")
    return source * 81 + target


def decode_action(action: int) -> tuple[int, int]:
    require_int(action, "action")
    if not 0 <= action < ACTION_SIZE:
        raise ValueError("Action must be in [0, 6560]")
    return divmod(action, 81)


def square_name(square: int) -> str:
    r, c = divmod(square, 9)
    return f"{chr(65 + c)}{r + 1}"


def move_name(action: int) -> str:
    a, b = decode_action(action)
    return f"{square_name(a)}-{square_name(b)}"


@dataclass(frozen=True)
class Rules:
    edge_capture: bool = True
    capture_target: int = 5
    lead_target: int = 3
    repetition_limit: int = 4
    no_moves_draw: bool = False

    def __post_init__(self):
        if type(self.edge_capture) is not bool or type(self.no_moves_draw) is not bool:
            raise ValueError("Rule switches must be booleans")
        for name in ("capture_target", "lead_target", "repetition_limit"):
            require_int(getattr(self, name), name)
        if not 0 <= self.capture_target <= 9 or not 0 <= self.lead_target <= 9:
            raise ValueError("Capture and lead targets must be in [0, 9]; zero disables")
        if self.repetition_limit != 0 and not 2 <= self.repetition_limit <= 100:
            raise ValueError("Repetition limit must be zero or in [2, 100]")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Position:
    black: int = 511 << 72
    white: int = 511
    turn: int = 0
    pending: int = -1

    def __post_init__(self):
        for name in ("black", "white", "turn", "pending"):
            require_int(getattr(self, name), name)
        if self.black < 0 or self.white < 0 or (self.black | self.white) & ~BOARD_MASK:
            raise ValueError("Bitboards must be nonnegative 81-bit values")
        if self.black & self.white:
            raise ValueError("Bitboards must not overlap")
        if self.black.bit_count() > 9 or self.white.bit_count() > 9:
            raise ValueError("Each player has at most nine pieces")
        if self.turn not in (0, 1) or self.pending not in (-1, 0, 1):
            raise ValueError("Invalid turn or pending leader")
        if self.pending == self.turn:
            raise ValueError("The pending leader cannot be the player to reply")

    def pieces(self, player: int) -> int:
        return self.black if player == 0 else self.white

    def captures(self, player: int) -> int:
        return 9 - self.pieces(1 - player).bit_count()

    def key(self) -> tuple[int, int, int, int]:
        return self.black, self.white, self.turn, self.pending

    def to_dict(self) -> dict:
        return dict(black=hex(self.black), white=hex(self.white),
                    turn=self.turn, pending=self.pending)

    @classmethod
    def from_dict(cls, data: dict) -> Position:
        if not isinstance(data, dict) or set(data) != {"black", "white", "turn", "pending"}:
            raise ValueError("Invalid position object")
        if not all(isinstance(data[x], str) and len(data[x]) <= 23
                   and data[x].startswith("0x") for x in ("black", "white")):
            raise ValueError("Bitboards must be hexadecimal strings")
        return cls(int(data["black"], 16), int(data["white"], 16),
                   data["turn"], data["pending"])


@dataclass(frozen=True)
class Outcome:
    winner: int = -2
    reason: str = "ongoing"

    @property
    def terminal(self) -> bool:
        return self.winner != -2


@dataclass(frozen=True)
class MoveResult:
    action: int
    captured: int
    outcome: Outcome


class GameLike(Protocol):
    rules: Rules
    initial: Position
    moves: list[int]
    @property
    def position(self) -> Position: ...
    @property
    def outcome(self) -> Outcome: ...
    @property
    def last_capture(self) -> int: ...
    def legal_actions(self) -> list[int]: ...
    def play(self, action: int) -> MoveResult: ...
    def undo(self) -> bool: ...
    def clone(self) -> GameLike: ...
    def history_keys(self) -> list[tuple[int, int, int, int]]: ...
    def close(self) -> None: ...
