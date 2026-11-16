"""History-complete deterministic shogi reference engine."""
from collections import Counter

from .model import MAX_GAME_MOVES
from .model import (Rules, Position, Outcome, MoveResult, START_SFEN, DECLARE, K,
                    require_int)
from .mechanics import legal_moves, apply_move, attackers, declarable
from .notation import parse_sfen


def repetition_outcome(checks) -> Outcome:
    """Adjudicate (mover, gave_check) events spanning four occurrences.

    Keeping this decision separate permits the specified mutual-check case to
    be tested with event sequences independently of geometric reachability.
    """
    perpetual = [side for side in (0, 1)
                 if any(actor == side for actor, _ in checks)
                 and all(check for actor, check in checks if actor == side)]
    return (Outcome(1 - perpetual[0], "perpetual_check") if len(perpetual) == 1
            else Outcome(-1, "repetition"))


class ShogiGame:
    backend = "python"

    def __init__(self, rules: Rules | None = None, sfen: str | None = None):
        self.rules = rules if rules is not None else Rules()
        if not isinstance(self.rules, Rules):
            raise ValueError("Expected Rules")
        self.initial = parse_sfen(START_SFEN if sfen is None else sfen)
        self._position = self.initial
        self._outcome = Outcome()
        self.moves: list[int] = []
        self._keys = [self.initial.key()]
        self._counts = Counter(self._keys)
        self._checks: list[tuple[int, bool]] = []
        self._undo = []
        self._legal = None
        self.comments: list[str] = []
        self._finish_no_moves()

    @property
    def position(self) -> Position:
        return self._position

    @property
    def outcome(self) -> Outcome:
        return self._outcome

    @property
    def ply(self) -> int:
        return len(self.moves)

    def _actions(self):
        if self._legal is None:
            p = self.position
            actions = sorted(legal_moves(list(p.board), p.hands, p.turn))
            if self.rules.declaration == "csa27" and declarable(p.board, p.hands, p.turn):
                actions.append(DECLARE)
            self._legal = tuple(actions)
        return self._legal

    def legal_actions(self) -> list[int]:
        return [] if self.outcome.terminal else list(self._actions())

    def is_legal(self, code: int) -> bool:
        return type(code) is int and not self.outcome.terminal and code in self._actions()

    def _finish_no_moves(self):
        if not self._actions():
            self._outcome = Outcome(1 - self.position.turn, "checkmate" if self.in_check() else "no_legal_moves")

    def play(self, code: int) -> MoveResult:
        if self.outcome.terminal:
            raise ValueError("Game is over")
        if not self.is_legal(code):
            raise ValueError("Illegal move")
        if len(self.moves) >= MAX_GAME_MOVES:
            raise ValueError("Move limit exceeded")
        p = self.position
        self._undo.append((p, self.outcome, self._legal, False))
        captured, check = 0, False
        if code == DECLARE:
            self._position = Position(p.board, p.hands, p.turn, p.move_number + 1)
            self._outcome = Outcome(p.turn, "declaration")
        else:
            board, hands = list(p.board), [list(h) for h in p.hands]
            captured = apply_move(board, hands, p.turn, code)
            self._position = Position(tuple(board), tuple(tuple(h) for h in hands), 1 - p.turn, p.move_number + 1)
            self._legal = None
            check = self.in_check()
        self.moves.append(code)
        self._checks.append((p.turn, check))
        key = self.position.key()
        self._keys.append(key)
        self._counts[key] += 1
        if code != DECLARE:
            self._finish_no_moves()
            if not self.outcome.terminal and self._counts[key] >= 4:
                start = self._keys.index(key)
                self._outcome = repetition_outcome(self._checks[start:])
            if not self.outcome.terminal and self.rules.max_moves and self.ply == self.rules.max_moves:
                self._outcome = Outcome(-1, "max_moves")
        return MoveResult(code, captured, check)

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._position, self._outcome, self._legal, resignation = self._undo.pop()
        if not resignation:
            key = self._keys.pop()
            self._counts[key] -= 1
            if not self._counts[key]:
                del self._counts[key]
            self.moves.pop()
            self._checks.pop()
        return True

    def resign(self) -> None:
        if self.outcome.terminal:
            raise ValueError("Game is over")
        self._undo.append((self.position, self.outcome, self._legal, True))
        self._outcome = Outcome(1 - self.position.turn, "resign")

    def clone(self) -> "ShogiGame":
        other = object.__new__(type(self))
        other.__dict__ = self.__dict__.copy()
        for name in ("moves", "_keys", "_checks", "_undo", "comments"):
            setattr(other, name, getattr(self, name).copy())
        other._counts = self._counts.copy()
        return other

    def history_keys(self) -> list[str]:
        return self._keys.copy()

    def in_check(self, player=None) -> bool:
        side = self.position.turn if player is None else require_int(player, "player", 0, 1)
        p = self.position
        return bool(attackers(p.board, p.board.index(K + 16 * side), 1 - side))

    def checkers(self) -> list[int]:
        p = self.position
        return sorted(attackers(p.board, p.board.index(K + 16 * p.turn), 1 - p.turn))

    def can_declare(self) -> bool:
        p = self.position
        return (not self.outcome.terminal and self.rules.declaration == "csa27"
                and declarable(p.board, p.hands, p.turn))

    def sfen(self) -> str:
        return self.position.sfen()

    def close(self) -> None:
        """Match the resource-owning native interface; Python owns no handles."""

    @classmethod
    def replay(cls, rules, sfen, codes):
        game = cls(rules, sfen)
        for code in codes:
            game.play(code)
        return game


def create_game(backend: str = "python", rules: Rules | None = None,
                sfen: str | None = None) -> ShogiGame:
    if backend != "python":
        raise ValueError(f"Unsupported shogi backend: {backend}")
    return ShogiGame(rules, sfen)
