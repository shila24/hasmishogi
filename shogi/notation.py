"""Validated SFEN, USI, CSA and Japanese move text."""
import re

from .model import MAX_SFEN_MOVE_NUMBER
from .model import (P, L, N, S, G, B, R, K, PROMOTED, UNPROMOTED, PIECE_LIMITS,
                    Position, DECLARE, decode_move, encode_move, encode_drop)
from .mechanics import attacked, dead_square, in_zone

SFEN_PIECES = " PLNSGBRK"
CSA_PIECES = ("", "FU", "KY", "KE", "GI", "KI", "KA", "HI", "OU",
              "TO", "NY", "NK", "NG", "UM", "RY")
JAPANESE_PIECES = ("", "歩", "香", "桂", "銀", "金", "角", "飛", "玉",
                   "と", "成香", "成桂", "成銀", "馬", "龍")
FILES = "９８７６５４３２１"
RANKS = "一二三四五六七八九"


def square_to_usi(square: int) -> str:
    if type(square) is not int or not 0 <= square < 81:
        raise ValueError("Invalid square")
    row, col = divmod(square, 9)
    return f"{9 - col}{chr(ord('a') + row)}"


def usi_to_square(text: str) -> int:
    if not isinstance(text, str) or not re.fullmatch(r"[1-9][a-i]", text):
        raise ValueError("Invalid USI square")
    return (ord(text[1]) - ord("a")) * 9 + 9 - int(text[0])


def position_key(position: Position) -> str:
    ranks = []
    for row in range(9):
        rank, empty = [], 0
        for cell in position.board[row * 9:row * 9 + 9]:
            if not cell:
                empty += 1
                continue
            if empty:
                rank.append(str(empty))
                empty = 0
            piece = cell & 15
            letter = SFEN_PIECES[UNPROMOTED.get(piece, piece)]
            if cell >> 4:
                letter = letter.lower()
            rank.append(("+" if piece in UNPROMOTED else "") + letter)
        if empty:
            rank.append(str(empty))
        ranks.append("".join(rank))
    hand_text = []
    for side in (0, 1):
        for hand in reversed(range(7)):
            count = position.hands[side][hand]
            if count:
                letter = SFEN_PIECES[hand + 1]
                hand_text.append((str(count) if count > 1 else "") + (letter.lower() if side else letter))
    return f"{'/'.join(ranks)} {'w' if position.turn else 'b'} {''.join(hand_text) or '-'}"


def validate_position(position: Position) -> None:
    board = position.board
    totals = [sum(h[i] for h in position.hands) for i in range(7)]
    pawn_files = (set(), set())
    for side in (0, 1):
        if board.count(K + 16 * side) != 1:
            raise ValueError("Exactly one king per side is required")
    for square, cell in enumerate(board):
        if not cell:
            continue
        side, piece = cell >> 4, cell & 15
        if dead_square(piece, square, side):
            raise ValueError("Dead unpromoted piece")
        if piece == P:
            if square % 9 in pawn_files[side]:
                raise ValueError("Two unpromoted pawns on a file")
            pawn_files[side].add(square % 9)
        if piece != K:
            totals[UNPROMOTED.get(piece, piece) - 1] += 1
    if any(count > limit for count, limit in zip(totals, PIECE_LIMITS)):
        raise ValueError("Too many pieces")
    other_king = board.index(K + 16 * (1 - position.turn))
    if attacked(board, other_king, position.turn):
        raise ValueError("The side not to move is in check")


def parse_sfen(text: str) -> Position:
    if not isinstance(text, str):
        raise ValueError("SFEN must be text")
    fields = text.split()
    if len(fields) != 4:
        raise ValueError("SFEN must have four fields")
    board_text, turn_text, hand_text, number = fields
    if (turn_text not in ("b", "w") or not re.fullmatch(r"[1-9][0-9]*", number)
            or len(number) > 7 or int(number) > MAX_SFEN_MOVE_NUMBER):
        raise ValueError("Invalid SFEN turn or move number")
    ranks = board_text.split("/")
    if len(ranks) != 9:
        raise ValueError("SFEN must have nine ranks")
    board = []
    for rank in ranks:
        cells, index = [], 0
        while index < len(rank):
            char = rank[index]
            index += 1
            if char in "123456789":
                cells.extend([0] * int(char))
                continue
            promoted = char == "+"
            if promoted:
                if index == len(rank):
                    raise ValueError("Incomplete promotion marker")
                char = rank[index]
                index += 1
            if char.upper() not in SFEN_PIECES[1:]:
                raise ValueError("Invalid SFEN piece")
            piece = SFEN_PIECES.index(char.upper())
            if promoted:
                if piece not in PROMOTED:
                    raise ValueError("Invalid promoted piece")
                piece = PROMOTED[piece]
            cells.append(piece + (16 if char.islower() else 0))
        if len(cells) != 9:
            raise ValueError("SFEN rank must have nine cells")
        board.extend(cells)
    hands = [[0] * 7 for _ in range(2)]
    if hand_text != "-":
        tokens = re.findall(r"([1-9][0-9]*)?([PLNSGBRplnsgbr])", hand_text)
        if not tokens or "".join(count + piece for count, piece in tokens) != hand_text:
            raise ValueError("Invalid SFEN hands")
        for count, letter in tokens:
            side, hand = int(letter.islower()), SFEN_PIECES.index(letter.upper()) - 1
            if hands[side][hand]:
                raise ValueError("Duplicate hand piece")
            hands[side][hand] = int(count) if count else 1
    position = Position(tuple(board), tuple(tuple(h) for h in hands), int(turn_text == "w"), int(number))
    validate_position(position)
    return position


def to_sfen(position: Position) -> str:
    return position.sfen()


def move_to_usi(code: int) -> str:
    move = decode_move(code)
    if move.kind == "declare":
        return "win"
    if move.kind == "drop":
        return f"{SFEN_PIECES[move.piece]}*{square_to_usi(move.to_sq)}"
    return square_to_usi(move.from_sq) + square_to_usi(move.to_sq) + ("+" if move.promote else "")


def usi_to_move(game, text: str) -> int:
    if not isinstance(text, str):
        raise ValueError("Invalid USI move")
    if text == "win":
        code = DECLARE
    elif re.fullmatch(r"[PLNSGBR]\*[1-9][a-i]", text):
        code = encode_drop(SFEN_PIECES.index(text[0]) - 1, usi_to_square(text[2:]))
    elif re.fullmatch(r"[1-9][a-i][1-9][a-i]\+?", text):
        code = encode_move(usi_to_square(text[:2]), usi_to_square(text[2:4]), text.endswith("+"))
    else:
        raise ValueError("Invalid USI move")
    if not game.is_legal(code):
        raise ValueError("Illegal move")
    return code


def _digits(square):
    row, col = divmod(square, 9)
    return f"{9 - col}{row + 1}"


def move_to_csa(game, code: int) -> str:
    move = decode_move(code)
    if not game.is_legal(code):
        raise ValueError("Illegal move")
    if move.kind == "declare":
        return "%KACHI"
    piece = move.piece if move.kind == "drop" else game.position.board[move.from_sq] & 15
    if move.promote:
        piece = PROMOTED[piece]
    side = "-" if game.position.turn else "+"
    source = "00" if move.kind == "drop" else _digits(move.from_sq)
    return side + source + _digits(move.to_sq) + CSA_PIECES[piece]


def csa_to_move(game, text: str) -> int:
    if not isinstance(text, str):
        raise ValueError("Invalid CSA move")
    if text == "%KACHI":
        return usi_to_move(game, "win")
    if not re.fullmatch(r"[+-](00|[1-9]{2})[1-9]{2}[A-Z]{2}", text):
        raise ValueError("Invalid CSA move")
    if text[0] != ("-" if game.position.turn else "+"):
        raise ValueError("Wrong CSA player")
    try:
        piece = CSA_PIECES.index(text[5:])
    except ValueError as exc:
        raise ValueError("Invalid CSA piece") from exc
    target = (int(text[4]) - 1) * 9 + 9 - int(text[3])
    if text[1:3] == "00":
        code = encode_drop(piece - 1, target)
    else:
        source = (int(text[2]) - 1) * 9 + 9 - int(text[1])
        before = game.position.board[source] & 15
        if piece != before and PROMOTED.get(before) != piece:
            raise ValueError("Invalid CSA promotion")
        code = encode_move(source, target, piece != before)
    if not game.is_legal(code):
        raise ValueError("Illegal move")
    return code


def _previous_target(game):
    return decode_move(game.moves[-1]).to_sq if game.moves else -1


def _qualifier(game, move, piece):
    origins = []
    for code in game.legal_actions():
        candidate = decode_move(code)
        if (candidate.kind == "move" and candidate.to_sq == move.to_sq
                and candidate.promote == move.promote
                and game.position.board[candidate.from_sq] & 15 == piece):
            origins.append(candidate.from_sq)
    if len(origins) < 2:
        return ""
    rel = (lambda sq: 80 - sq) if game.position.turn else (lambda sq: sq)
    target = rel(move.to_sq)
    source = rel(move.from_sq)
    origins = [rel(sq) for sq in origins]
    left, right = min(sq % 9 for sq in origins), max(sq % 9 for sq in origins)

    def vertical(sq):
        return "上" if sq // 9 > target // 9 else "引" if sq // 9 < target // 9 else "寄"

    def horizontal(sq):
        if sq % 9 == target % 9 and vertical(sq) == "上":
            return "直"
        if left == right:
            return ""
        return "右" if sq % 9 == right else "左" if sq % 9 == left else ""

    h, v = horizontal(source), vertical(source)
    if h and sum(horizontal(sq) == h for sq in origins) == 1:
        return h
    if sum(vertical(sq) == v for sq in origins) == 1:
        return v
    return h + v


def move_to_japanese(game, code: int, *, kif: bool = False) -> str:
    if not game.is_legal(code):
        raise ValueError("Illegal move")
    move = decode_move(code)
    prefix = "" if kif else "△" if game.position.turn else "▲"
    if move.kind == "declare":
        return prefix + "入玉勝ち"
    destination = ("同　" if move.to_sq == _previous_target(game)
                   else FILES[move.to_sq % 9] + RANKS[move.to_sq // 9])
    piece = move.piece if move.kind == "drop" else game.position.board[move.from_sq] & 15
    result = prefix + destination + JAPANESE_PIECES[piece]
    if move.kind == "drop":
        return result + "打"
    if not kif:
        result += _qualifier(game, move, piece)
    if move.promote:
        result += "成"
    elif piece in PROMOTED and (in_zone(move.from_sq, game.position.turn)
                               or in_zone(move.to_sq, game.position.turn)):
        result += "不成"
    if kif:
        result += f"({_digits(move.from_sq)})"
    return result


def _normalize_japanese(text):
    return re.sub(r"\s", "", text).replace("竜", "龍").replace("王", "玉")


def japanese_to_move(game, text: str) -> int:
    if not isinstance(text, str):
        raise ValueError("Invalid Japanese move")
    candidate = _normalize_japanese(text)
    matches = []
    for code in game.legal_actions():
        forms = [move_to_japanese(game, code), move_to_japanese(game, code, kif=True)]
        forms += [form.lstrip("▲△") for form in forms]
        # KIF readers accept omitted optional non-promotion markers.
        forms += [form.replace("不成", "") for form in forms if "不成" in form]
        if candidate in {_normalize_japanese(form) for form in forms}:
            matches.append(code)
    if len(matches) != 1:
        raise ValueError("Illegal or ambiguous Japanese move")
    return matches[0]
