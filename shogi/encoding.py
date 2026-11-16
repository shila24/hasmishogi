"""Side-to-move shogi-enc-v1 planes and injective legal policy labels."""
from .model import DECLARE, PIECE_LIMITS, decode_move, require_int

ENCODING_ID = "shogi-enc-v1"
NUM_PLANES = 49
ACTION_SIZE = 2188
DECLARE_INDEX = 2187
_DIRECTIONS = {(-1, 0): 0, (-1, -1): 1, (-1, 1): 2, (0, -1): 3,
               (0, 1): 4, (1, 0): 5, (1, -1): 6, (1, 1): 7,
               (-2, -1): 8, (-2, 1): 9}


def encode(game, player: int) -> list[float]:
    require_int(player, "player", 0, 1)
    if player != game.position.turn:
        raise ValueError("The encoder player must be the side to move")
    result = [0.0] * (NUM_PLANES * 81)
    p = game.position
    for square, cell in enumerate(p.board):
        if cell:
            plane = (cell & 15) - 1 + (14 if cell >> 4 != player else 0)
            relative = 80 - square if player else square
            result[plane * 81 + relative] = 1.0
    for owner, base in ((player, 28), (1 - player, 35)):
        for hand, divisor in enumerate(PIECE_LIMITS):
            value = p.hands[owner][hand] / divisor
            result[(base + hand) * 81:(base + hand + 1) * 81] = [value] * 81
    earlier = game.history_keys().count(p.key()) - 1
    values = (float(player == 1), float(game.in_check()), float(earlier >= 1),
              float(earlier >= 2), float(earlier >= 3),
              float(min(1, game.ply / game.rules.max_moves)) if game.rules.max_moves else 0.0, 1.0)
    for plane, value in enumerate(values, 42):
        result[plane * 81:(plane + 1) * 81] = [value] * 81
    return result


def _label(side, code):
    if code == DECLARE:
        return DECLARE_INDEX
    move = decode_move(code)
    target = 80 - move.to_sq if side else move.to_sq
    if move.kind == "drop":
        label = 20 + move.piece - 1
    else:
        source = 80 - move.from_sq if side else move.from_sq
        dr, dc = target // 9 - source // 9, target % 9 - source % 9
        direction = (dr, dc) if (dr, dc) in ((-2, -1), (-2, 1)) else ((dr > 0) - (dr < 0), (dc > 0) - (dc < 0))
        label = _DIRECTIONS[direction] + 10 * int(move.promote)
    return label * 81 + target


def move_label(game, code: int) -> int:
    if not game.is_legal(code):
        raise ValueError("Illegal move")
    return _label(game.position.turn, code)


def legal_labels(game) -> list[int]:
    return [_label(game.position.turn, code) for code in game.legal_actions()]


def label_to_move(game, index: int) -> int:
    require_int(index, "policy label", 0, ACTION_SIZE - 1)
    for code in game.legal_actions():
        if _label(game.position.turn, code) == index:
            return code
    raise ValueError("No legal move has this policy label")
