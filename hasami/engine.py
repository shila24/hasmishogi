"""Independent Python rules engine with bitboards and exact history."""
from __future__ import annotations
from collections import Counter
from .model import (Position, Rules, Outcome, MoveResult, bits, neighbours,
                    EDGE, BOARD_MASK, RAYS, decode_action)


def geometric_actions(position: Position) -> list[int]:
    occupied = position.black | position.white
    actions = []
    for source in bits(position.pieces(position.turn)):
        for ray in RAYS[source]:
            for target in ray:
                if occupied & (1 << target):
                    break
                actions.append(source * 81 + target)
    actions.sort()
    return actions


def captured_after_move(me: int, enemy: int, target: int, edge_capture: bool) -> int:
    """Compute all captures on one pre-removal snapshot, without cascades."""
    removed = 0
    for ray in RAYS[target]:
        run = 0
        for sq in ray:
            bit = 1 << sq
            if enemy & bit:
                run |= bit
            else:
                if me & bit:
                    removed |= run
                break
    if edge_capture:
        unseen = enemy
        while unseen:
            component = unseen & -unseen
            while True:
                expanded = component | (neighbours(component) & enemy)
                if expanded == component:
                    break
                component = expanded
            unseen &= ~component
            boundary = neighbours(component) & ~component & BOARD_MASK
            if (component & EDGE and boundary & (1 << target)
                    and not (boundary & ~me)):
                removed |= component
    return removed


class PythonGame:
    backend = "python"

    def __init__(self, rules: Rules | None = None, position: Position | None = None):
        self.rules = rules or Rules()
        self.initial = position or Position()
        self._position = self.initial
        self.moves: list[int] = []
        self._outcome = Outcome()
        self._last_capture = 0
        self._undo: list[tuple[Position, Outcome, int]] = []
        self._validate_setup()
        self._adjudicate_setup()
        self._keys = [self._position.key()]
        self._counts = Counter(self._keys)

    @property
    def position(self) -> Position:
        return self._position

    @property
    def outcome(self) -> Outcome:
        return self._outcome

    @property
    def last_capture(self) -> int:
        return self._last_capture

    def _validate_setup(self):
        p, r = self.position, self.rules
        lead = p.captures(0) - p.captures(1)
        if p.pending != -1:
            signed = lead if p.pending == 0 else -lead
            if not r.lead_target or signed < r.lead_target:
                raise ValueError("Pending leader is inconsistent with the material lead")
        if (r.capture_target and p.captures(0) >= r.capture_target
                and p.captures(1) >= r.capture_target):
            raise ValueError("Both players cannot already have a capture victory")
        immediate = bool(r.capture_target and max(p.captures(0), p.captures(1)) >= r.capture_target)
        if r.lead_target and abs(lead) >= r.lead_target and not immediate:
            if p.pending != (0 if lead > 0 else 1):
                raise ValueError("A setup with a decisive lead requires an explicit reply state")

    def _adjudicate_setup(self):
        p = self.position
        if self.rules.capture_target:
            for player in (0, 1):
                if p.captures(player) >= self.rules.capture_target:
                    self._finish(player, "five_captures")
                    return
        if not geometric_actions(p):
            if p.pending != -1:
                self._finish(p.pending, "lead_confirmed")
            else:
                self._finish(-1 if self.rules.no_moves_draw else 1 - p.turn, "no_moves")

    def _finish(self, winner: int, reason: str):
        p = self.position
        self._position = Position(p.black, p.white, p.turn, -1)
        self._outcome = Outcome(winner, reason)

    def legal_actions(self) -> list[int]:
        return [] if self.outcome.terminal else geometric_actions(self.position)

    def play(self, action: int) -> MoveResult:
        source, target = decode_action(action)
        if self.outcome.terminal:
            raise ValueError("The game has already ended")
        p, r = self.position, self.rules
        me, enemy = p.pieces(p.turn), p.pieces(1 - p.turn)
        if not (me & (1 << source)) or (me | enemy) & (1 << target):
            raise ValueError("The source must be friendly and the destination empty")
        valid = False
        for ray in RAYS[source]:
            for sq in ray:
                if (me | enemy) & (1 << sq):
                    break
                if sq == target:
                    valid = True
                    break
            if valid:
                break
        if not valid:
            raise ValueError("The move must follow an unobstructed orthogonal ray")
        new_me = (me ^ (1 << source)) | (1 << target)
        captured = captured_after_move(new_me, enemy, target, r.edge_capture)
        new_enemy = enemy & ~captured
        black, white = (new_me, new_enemy) if p.turn == 0 else (new_enemy, new_me)
        self._undo.append((p, self.outcome, self._last_capture))
        self._position = Position(black, white, 1 - p.turn, -1)
        self._last_capture = captured
        self.moves.append(action)
        self._outcome = Outcome()
        now = self.position
        if r.capture_target and now.captures(p.turn) >= r.capture_target:
            self._finish(p.turn, "five_captures")
        elif p.pending != -1 and (now.captures(p.pending) - now.captures(1 - p.pending)
                                  >= r.lead_target):
            self._finish(p.pending, "lead_confirmed")
        else:
            lead = now.captures(p.turn) - now.captures(1 - p.turn)
            if r.lead_target and lead >= r.lead_target:
                self._position = Position(black, white, 1 - p.turn, p.turn)
        # Register the position before draw adjudication; terminal keys have no pending leader.
        key = self.position.key()
        self._keys.append(key)
        self._counts[key] += 1
        if not self.outcome.terminal:
            if r.repetition_limit and self._counts[key] >= r.repetition_limit:
                self._finish(-1, "repetition")
            elif not geometric_actions(self.position):
                if self.position.pending != -1:
                    self._finish(self.position.pending, "lead_confirmed")
                else:
                    self._finish(-1 if r.no_moves_draw else p.turn, "no_moves")
        # Terminal normalization may clear a pending leader after no-move adjudication.
        final_key = self.position.key()
        if final_key != key:
            self._counts[key] -= 1
            if self._counts[key] == 0:
                del self._counts[key]
            self._keys[-1] = final_key
            self._counts[final_key] += 1
        return MoveResult(action, captured, self.outcome)

    def undo(self) -> bool:
        if not self._undo:
            return False
        key = self._keys.pop()
        self._counts[key] -= 1
        if self._counts[key] == 0:
            del self._counts[key]
        self._position, self._outcome, self._last_capture = self._undo.pop()
        self.moves.pop()
        return True

    def clone(self) -> PythonGame:
        other = object.__new__(PythonGame)
        other.rules, other.initial = self.rules, self.initial
        other._position, other._outcome = self.position, self.outcome
        other._last_capture = self._last_capture
        other.moves = self.moves.copy()
        other._undo = self._undo.copy()
        other._keys = self._keys.copy()
        other._counts = self._counts.copy()
        return other

    def history_keys(self) -> list[tuple[int, int, int, int]]:
        return self._keys.copy()

    def close(self) -> None:
        pass
