"""Movement tables and king-safe legal generation, independent of game history."""
from .model import (P, L, N, S, G, B, R, K, PP, PL, PN, PS, PB, PR,
                    PROMOTED, UNPROMOTED, DROP_BASE)

DIRECTIONS = ((-1, 0), (-1, -1), (-1, 1), (0, -1), (0, 1), (1, 0), (1, -1), (1, 1))
GOLD = ((-1, 0), (-1, -1), (-1, 1), (0, -1), (0, 1), (1, 0))
STEPS = {P: ((-1, 0),), N: ((-2, -1), (-2, 1)),
         S: ((-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)), G: GOLD, K: DIRECTIONS,
         PP: GOLD, PL: GOLD, PN: GOLD, PS: GOLD,
         PB: ((-1, 0), (1, 0), (0, -1), (0, 1)),
         PR: ((-1, -1), (-1, 1), (1, -1), (1, 1))}
SLIDES = {L: ((-1, 0),), B: ((-1, -1), (-1, 1), (1, -1), (1, 1)),
          R: ((-1, 0), (1, 0), (0, -1), (0, 1))}
SLIDES[PB], SLIDES[PR] = SLIDES[B], SLIDES[R]


def ray(square, dr, dc):
    row, col = divmod(square, 9)
    result = []
    row, col = row + dr, col + dc
    while 0 <= row < 9 and 0 <= col < 9:
        result.append(row * 9 + col)
        row, col = row + dr, col + dc
    return tuple(result)


RAYS = tuple(tuple(ray(sq, *direction) for direction in DIRECTIONS) for sq in range(81))


def _tables():
    movement, attacks = [], []
    for side in (0, 1):
        sign = 1 if side == 0 else -1
        piece_moves = [()] * 15
        for piece in range(1, 15):
            entries = []
            for sq in range(81):
                steps = tuple(line[0] for dr, dc in STEPS.get(piece, ())
                              if (line := ray(sq, sign * dr, sign * dc)))
                slides = tuple(ray(sq, sign * dr, sign * dc) for dr, dc in SLIDES.get(piece, ()))
                entries.append((steps, slides))
            piece_moves[piece] = tuple(entries)
        movement.append(tuple(piece_moves))
        masks = []
        for dr, dc in DIRECTIONS:
            toward = (-dr * sign, -dc * sign)
            step_mask = sum(1 << p for p, steps in STEPS.items() if toward in steps)
            slide_mask = sum(1 << p for p, slides in SLIDES.items() if toward in slides)
            masks.append((step_mask | slide_mask, slide_mask))
        attacks.append(tuple(masks))
    return tuple(movement), tuple(attacks)


MOVEMENT, ATTACK_MASKS = _tables()
KNIGHT_SOURCES = tuple(tuple(tuple(line[0] for dc in (-1, 1)
                                  if (line := ray(sq, 2 if side == 0 else -2, dc)))
                            for sq in range(81)) for side in (0, 1))


def attackers(board, square, by):
    """Return attacking squares, including attacks by pinned enemy pieces."""
    found = []
    for line, (near, far) in zip(RAYS[square], ATTACK_MASKS[by]):
        for distance, source in enumerate(line):
            piece = board[source]
            if piece:
                if piece >> 4 == by and (near if distance == 0 else far) & (1 << (piece & 15)):
                    found.append(source)
                break
    found.extend(sq for sq in KNIGHT_SOURCES[by][square] if board[sq] == N + 16 * by)
    return found


def attacked(board, square, by):
    for line, (near, far) in zip(RAYS[square], ATTACK_MASKS[by]):
        for distance, source in enumerate(line):
            piece = board[source]
            if piece:
                if piece >> 4 == by and (near if distance == 0 else far) & (1 << (piece & 15)):
                    return True
                break
    return any(board[sq] == N + 16 * by for sq in KNIGHT_SOURCES[by][square])


def pinned_sources(board, king, side):
    pins = set()
    for line, (_, mask) in zip(RAYS[king], ATTACK_MASKS[1 - side]):
        candidate = -1
        for square in line:
            piece = board[square]
            if not piece:
                continue
            if candidate < 0 and piece >> 4 == side:
                candidate = square
                continue
            if candidate >= 0 and piece >> 4 != side and mask & (1 << (piece & 15)):
                pins.add(candidate)
            break
    return pins


def in_zone(square, side):
    return square < 27 if side == 0 else square >= 54


def dead_square(piece, square, side):
    row = square // 9 if side == 0 else 8 - square // 9
    return (piece in (P, L) and row == 0) or (piece == N and row < 2)


def legal_moves(board, hands, side, *, drops=True):
    """Yield legal board/drop codes; temporarily edit and always restore board cells.

    Only king moves, moves from a pinned square, and check evasions need a new
    attack query. A checking pawn drop can only be answered by a board move:
    the adjacent pawn cannot be interposed against by a drop.
    """
    king = board.index(K + 16 * side)
    checked = attacked(board, king, 1 - side)
    pins = pinned_sources(board, king, side)
    for source, cell in enumerate(board):
        if not cell or cell >> 4 != side:
            continue
        piece = cell & 15
        steps, slides = MOVEMENT[side][piece][source]
        destinations = list(steps)
        for line in slides:
            for target in line:
                destinations.append(target)
                if board[target]:
                    break
        for target in destinations:
            captured = board[target]
            if captured and (captured >> 4 == side or captured & 15 == K):
                continue
            if checked or piece == K or source in pins:
                board[source], board[target] = 0, cell
                try:
                    safe = not attacked(board, target if piece == K else king, 1 - side)
                finally:
                    board[source], board[target] = cell, captured
                if not safe:
                    continue
            code = (source * 81 + target) * 2
            if not dead_square(piece, target, side):
                yield code
            if piece in PROMOTED and (in_zone(source, side) or in_zone(target, side)):
                yield code + 1
    if not drops:
        return
    pawn_files = {sq % 9 for sq, piece in enumerate(board) if piece == P + 16 * side}
    enemy_king = board.index(K + 16 * (1 - side))
    for hand, count in enumerate(hands[side]):
        if not count:
            continue
        piece = hand + 1
        for target, cell in enumerate(board):
            if cell or dead_square(piece, target, side) or (piece == P and target % 9 in pawn_files):
                continue
            pawn_check = piece == P and target + (-9 if side == 0 else 9) == enemy_king
            if checked or pawn_check:
                board[target] = piece + 16 * side
                try:
                    safe = not checked or not attacked(board, king, 1 - side)
                    if safe and pawn_check:
                        safe = next(legal_moves(board, hands, 1 - side, drops=False), None) is not None
                finally:
                    board[target] = 0
                if not safe:
                    continue
            yield DROP_BASE + hand * 81 + target


def apply_move(board, hands, side, code):
    """Apply a previously validated board/drop code to mutable working arrays."""
    if code >= DROP_BASE:
        hand, target = divmod(code - DROP_BASE, 81)
        hands[side][hand] -= 1
        board[target] = hand + 1 + 16 * side
        return 0
    source, target = divmod(code // 2, 81)
    cell, captured = board[source], board[target]
    if captured:
        piece = captured & 15
        hands[side][UNPROMOTED.get(piece, piece) - 1] += 1
    board[source] = 0
    board[target] = PROMOTED[cell & 15] + 16 * side if code & 1 else cell
    return captured


def declarable(board, hands, side):
    king = board.index(K + 16 * side)
    if not in_zone(king, side) or attacked(board, king, 1 - side):
        return False
    pieces = [cell & 15 for sq, cell in enumerate(board)
              if cell and cell >> 4 == side and cell & 15 != K and in_zone(sq, side)]
    points = sum(5 if piece in (B, R, PB, PR) else 1 for piece in pieces)
    points += sum(count * (5 if hand >= 5 else 1) for hand, count in enumerate(hands[side]))
    return len(pieces) >= 10 and points >= (28 if side == 0 else 27)
