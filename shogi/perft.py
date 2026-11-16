"""Conventional move-tree counts, independent of repetition and move-limit draws.

Declaration is a terminal leaf, contributing no descendants. The traversal uses
the engine's legal generator and never mutates the supplied game or its history.
"""
from .mechanics import legal_moves, apply_move, declarable
from .model import DECLARE, require_int


def _count(board, hands, side, depth, declaration):
    if depth == 0:
        return 1
    actions = list(legal_moves(board, hands, side))
    can_declare = declaration and declarable(board, hands, side)
    if depth == 1:
        return len(actions) + int(can_declare)
    count = 0
    for action in actions:
        next_board, next_hands = board.copy(), [h.copy() for h in hands]
        apply_move(next_board, next_hands, side, action)
        count += _count(next_board, next_hands, 1 - side, depth - 1, declaration)
    return count


def perft(game, depth: int) -> int:
    require_int(depth, "depth", 0)
    if depth and game.outcome.terminal:
        return 0
    p = game.position
    return _count(list(p.board), [list(h) for h in p.hands], p.turn, depth,
                  game.rules.declaration == "csa27")


def divide(game, depth: int) -> dict[int, int]:
    require_int(depth, "depth", 1)
    p = game.position
    result = {}
    for code in game.legal_actions():
        if code == DECLARE:
            result[code] = int(depth == 1)
            continue
        board, hands = list(p.board), [list(h) for h in p.hands]
        apply_move(board, hands, p.turn, code)
        result[code] = _count(board, hands, 1 - p.turn, depth - 1, game.rules.declaration == "csa27")
    return result
