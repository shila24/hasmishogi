"""Validated JSON, KIF and CSA records, replayed exclusively by the engine.

KIF explicit starts use an SFEN position header; conventional board diagrams
and hand lines are also accepted. Rules and initial move numbers are carried
in extension headers/comments so our exports preserve nondefault games.
Unknown headers are retained in ``game.comments``. Text imports reject moves
with their physical line number and never execute code from a record.
"""
from dataclasses import asdict
import json
from pathlib import Path
import re

from .engine import ShogiGame
from .model import Rules, Position, Outcome, START_SFEN, DECLARE, K
from .notation import (CSA_PIECES, JAPANESE_PIECES, move_to_usi, usi_to_move,
                       move_to_csa, csa_to_move, move_to_japanese, japanese_to_move,
                       parse_sfen)

FORMAT = "shogi-record-v1"
MAX_BYTES = 8 * 1024 * 1024
MAX_MOVES = 20000
_KIF_END = {"checkmate": "詰み", "no_legal_moves": "合法手なし", "repetition": "千日手",
            "perpetual_check": "連続王手の千日手", "max_moves": "最大手数", "resign": "投了"}
_CSA_END = {"checkmate": "%TSUMI", "no_legal_moves": "%NO_LEGAL_MOVES", "repetition": "%SENNICHITE",
            "perpetual_check": "%PERPETUAL_CHECK", "max_moves": "%JISHOGI", "resign": "%TORYO"}


def _read_comment(text):
    if text.startswith('comment:"'):
        return json.loads(text[8:])
    return text


def record_dict(game) -> dict:
    result = {"format": FORMAT, "rules": game.rules.to_dict(), "sfen": game.initial.sfen(),
              "moves": [move_to_usi(code) for code in game.moves], "outcome": asdict(game.outcome)}
    if game.comments:
        result["comments"] = game.comments.copy()
    return result


def from_record(data: dict) -> ShogiGame:
    required = {"format", "rules", "sfen", "moves", "outcome"}
    if (not isinstance(data, dict) or not required <= set(data)
            or set(data) - required - {"comments"} or data["format"] != FORMAT):
        raise ValueError("Invalid shogi record")
    if not isinstance(data["sfen"], str) or not data["sfen"].strip():
        # ShogiGame treats a missing SFEN as the start position; a record must name its start explicitly.
        raise ValueError("Invalid record SFEN")
    moves = data["moves"]
    if not isinstance(moves, list) or len(moves) > MAX_MOVES or any(not isinstance(m, str) for m in moves):
        raise ValueError("Invalid record moves")
    outcome = data["outcome"]
    if not isinstance(outcome, dict) or set(outcome) != {"winner", "reason"}:
        raise ValueError("Invalid record outcome")
    expected = Outcome(**outcome)
    comments = data.get("comments", [])
    if not isinstance(comments, list) or any(not isinstance(c, str) for c in comments):
        raise ValueError("Invalid record comments")
    game = ShogiGame(Rules.from_dict(data["rules"]), data["sfen"])
    for index, text in enumerate(moves, 1):
        try:
            game.play(usi_to_move(game, text))
        except ValueError as exc:
            raise ValueError(f"Move {index}: {exc}") from exc
    if expected.reason == "resign":
        game.resign()
    if game.outcome != expected:
        raise ValueError("Record outcome does not match replay")
    game.comments = comments.copy()
    return game


def _check_text(text):
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_BYTES:
        raise ValueError("Invalid or oversized record text")


def to_json(game) -> str:
    return json.dumps(record_dict(game), ensure_ascii=False, indent=2) + "\n"


def from_json(text: str) -> ShogiGame:
    _check_text(text)
    try:
        return from_record(json.loads(text))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid JSON record") from exc


def to_kif(game) -> str:
    lines = ["#KIF version=2.0", "手合割：平手", f"局面：{game.initial.sfen()}",
             "規則：" + json.dumps(game.rules.to_dict(), ensure_ascii=False), "手数----指手---------消費時間--"]
    lines.extend("*comment:" + json.dumps(comment, ensure_ascii=False) for comment in game.comments)
    replay = ShogiGame(game.rules, game.initial.sfen())
    for index, code in enumerate(game.moves, 1):
        lines.append(f"{index:4d} {move_to_japanese(replay, code, kif=True)}")
        replay.play(code)
    if game.outcome.terminal and game.outcome.reason != "declaration":
        lines.append(f"{game.ply + 1:4d} {_KIF_END[game.outcome.reason]}")
    return "\n".join(lines) + "\n"


def _end(game, reason):
    if reason == "resign":
        game.resign()
    elif game.outcome.reason != reason:
        raise ValueError("Terminal marker does not match replay")


def _hand_number(text):
    if not text:
        return 1
    if text.isascii() and text.isdigit():
        return int(text)
    digits = "一二三四五六七八九"
    if text in digits and len(text) == 1:
        return digits.index(text) + 1
    if text == "十":
        return 10
    if len(text) == 2 and text[0] == "十" and text[1] in digits:
        return 10 + digits.index(text[1]) + 1
    raise ValueError("Invalid hand count")


def _kif_hands(text):
    text = text.strip()
    if text in ("なし", "無し", "-", ""):
        return (0,) * 7
    result = [0] * 7
    for token in text.split():
        if token[:1] not in JAPANESE_PIECES[1:8]:
            raise ValueError("Invalid KIF hand piece")
        hand = JAPANESE_PIECES.index(token[0]) - 1
        if result[hand]:
            raise ValueError("Duplicate KIF hand piece")
        result[hand] = _hand_number(token[1:])
    return tuple(result)


def from_kif(text: str) -> ShogiGame:
    _check_text(text)
    rules, sfen, game = Rules(), START_SFEN, None
    comments, rows = [], {}
    hands, turn = [(0,) * 7, (0,) * 7], 0
    ended = False
    for number, raw in enumerate(text.lstrip("\ufeff").splitlines(), 1):
        line = raw.strip()
        try:
            if not line or line.startswith("#") or line.startswith("手数----"):
                continue
            if line.startswith("*"):
                comments.append(_read_comment(line[1:]))
                continue
            if line.split() in ([str(n) for n in range(9, 0, -1)], list("９８７６５４３２１")):
                continue
            match = re.match(r"^([0-9]+)\s+(.+)$", line)
            if match:
                if game is None:
                    if rows:
                        if len(rows) != 9:
                            raise ValueError("Incomplete KIF board")
                        sfen = Position(tuple(cell for row in range(9) for cell in rows[row]), tuple(hands), turn, 1).sfen()
                    game = ShogiGame(rules, sfen)
                if ended or int(match[1]) != game.ply + 1 or game.ply >= MAX_MOVES:
                    raise ValueError("Invalid move number or move after end")
                move_text = re.sub(r"\s+\(\s*\d+:.*$", "", match[2]).strip()
                if move_text in _KIF_END.values():
                    _end(game, next(reason for reason, value in _KIF_END.items() if value == move_text))
                    ended = True
                else:
                    game.play(japanese_to_move(game, move_text))
                    ended = game.outcome.reason == "declaration"
                continue
            if line[0] in "0123456789":
                raise ValueError("Malformed numbered move")
            if game is not None:
                if line.startswith("まで"):
                    continue
                comments.append(raw)
                continue
            if line.startswith(("局面：", "局面:")):
                sfen = line[3:].strip()
                parse_sfen(sfen)
            elif line.startswith(("規則：", "規則:")):
                rules = Rules.from_dict(json.loads(line[3:]))
            elif line.startswith("手合割"):
                if re.split("[：:]", line, maxsplit=1)[-1].strip() != "平手":
                    raise ValueError("Only even starts or explicit positions are supported")
            elif line.startswith(("先手の持駒", "後手の持駒")):
                side = int(line.startswith("後手"))
                hands[side] = _kif_hands(re.split("[：:]", line, maxsplit=1)[-1])
            elif line == "後手番":
                turn = 1
            elif line.startswith("|"):
                row_text, separator, suffix = line[1:].partition("|")
                if not separator or suffix not in "一二三四五六七八九" or len(suffix) != 1:
                    raise ValueError("Invalid KIF board row")
                # Conventional diagrams use two characters per cell: side marker and piece.
                cells = []
                aliases = {"杏": 10, "圭": 11, "全": 12, "竜": 14, "王": 8}
                if len(row_text) != 18:
                    raise ValueError("Invalid KIF board width")
                for index in range(0, 18, 2):
                    marker, letter = row_text[index:index + 2]
                    if marker not in (" ", "v"):
                        raise ValueError("Invalid KIF side marker")
                    if letter == "・":
                        cells.append(0)
                    else:
                        piece = aliases[letter] if letter in aliases else JAPANESE_PIECES.index(letter)
                        cells.append(piece + (16 if marker == "v" else 0))
                row = "一二三四五六七八九".index(suffix)
                if row in rows:
                    raise ValueError("Duplicate KIF board row")
                rows[row] = cells
            elif line.startswith("+") or re.fullmatch(r"[１２３４５６７８９\s]+", line):
                continue
            else:
                comments.append(raw)
        except (ValueError, TypeError, IndexError) as exc:
            raise ValueError(f"Line {number}: {exc}") from exc
    if game is None:
        if rows:
            if len(rows) != 9:
                raise ValueError("Line end: incomplete KIF board")
            sfen = Position(tuple(cell for row in range(9) for cell in rows[row]), tuple(hands), turn, 1).sfen()
        game = ShogiGame(rules, sfen)
    game.comments = comments
    return game


def to_csa(game) -> str:
    p = game.initial
    lines = ["V2.2", "'rules:" + json.dumps(game.rules.to_dict()), f"'move_number:{p.move_number}"]
    lines.extend("'comment:" + json.dumps(comment, ensure_ascii=False) for comment in game.comments)
    for row in range(9):
        cells = [" * " if not cell else ("-" if cell >> 4 else "+") + CSA_PIECES[cell & 15]
                 for cell in p.board[row * 9:row * 9 + 9]]
        lines.append(f"P{row + 1}" + "".join(cells))
    for side in (0, 1):
        lines.append(("P-" if side else "P+") + "".join("00" + CSA_PIECES[hand + 1]
                     for hand, count in enumerate(p.hands[side]) for _ in range(count)))
    lines.append("-" if p.turn else "+")
    replay = ShogiGame(game.rules, p.sfen())
    for code in game.moves:
        lines.append(move_to_csa(replay, code))
        replay.play(code)
    if game.outcome.terminal and game.outcome.reason != "declaration":
        lines.append(_CSA_END[game.outcome.reason])
    return "\n".join(lines) + "\n"


def from_csa(text: str) -> ShogiGame:
    _check_text(text)
    rules, move_number = Rules(), 1
    board, hands = [0] * 81, [[0] * 7 for _ in range(2)]
    rows, comments, game, ended = set(), [], None, False
    for number, raw in enumerate(text.lstrip("\ufeff").splitlines(), 1):
        line = raw.strip()
        try:
            if not line or line == "V2.2" or re.fullmatch(r"T\d+", line):
                continue
            if line.startswith("'rules:") and game is None:
                rules = Rules.from_dict(json.loads(line[7:]))
            elif line.startswith("'move_number:") and game is None:
                move_number = int(line[13:])
            elif line.startswith("'"):
                comments.append(_read_comment(line[1:]))
            elif line in ("+", "-"):
                if game is not None or len(rows) != 9:
                    raise ValueError("Missing board or repeated turn")
                initial = Position(tuple(board), tuple(tuple(h) for h in hands), int(line == "-"), move_number)
                game = ShogiGame(rules, initial.sfen())
            elif line.startswith(("+", "-", "%")):
                if game is None or ended or game.ply >= MAX_MOVES:
                    raise ValueError("Move outside a game")
                if line in _CSA_END.values():
                    _end(game, next(reason for reason, value in _CSA_END.items() if value == line))
                    ended = True
                else:
                    game.play(csa_to_move(game, line))
                    ended = game.outcome.reason == "declaration"
            elif line == "PI":
                if rows or game is not None or any(any(h) for h in hands):
                    raise ValueError("Conflicting initial board")
                board = list(parse_sfen(START_SFEN).board)
                rows = set(range(9))
            elif re.match(r"P[1-9]", line):
                row = int(line[1]) - 1
                if game is not None or row in rows:
                    raise ValueError("Duplicate or late board row")
                # Preserve trailing spaces in the last empty cell.
                payload = raw.lstrip()[2:]
                if len(payload) != 27:
                    raise ValueError("Invalid CSA board width")
                for col in range(9):
                    token = payload[col * 3:col * 3 + 3]
                    if token == " * ":
                        continue
                    if token[0] not in ("+", "-"):
                        raise ValueError("Invalid CSA board piece")
                    piece = CSA_PIECES.index(token[1:])
                    board[row * 9 + col] = piece + (16 if token[0] == "-" else 0)
                rows.add(row)
            elif line.startswith(("P+", "P-")):
                if game is not None or (len(line) - 2) % 4:
                    raise ValueError("Invalid CSA hand list")
                side = int(line[1] == "-")
                for index in range(2, len(line), 4):
                    token = line[index:index + 4]
                    piece = CSA_PIECES.index(token[2:])
                    if token[:2] == "00":
                        if not 1 <= piece <= 7:
                            raise ValueError("Invalid CSA hand piece")
                        hands[side][piece - 1] += 1
                    elif re.fullmatch("[1-9]{2}", token[:2]):
                        square = (int(token[1]) - 1) * 9 + 9 - int(token[0])
                        if board[square]:
                            raise ValueError("Duplicate CSA placement")
                        board[square] = piece + 16 * side
                    else:
                        raise ValueError("Invalid CSA placement")
            elif line.startswith("P"):
                raise ValueError("Unsupported CSA position")
            elif line[0] in "0123456789":
                raise ValueError("Malformed CSA move: missing player sign")
            else:
                comments.append(raw)
        except (ValueError, TypeError, IndexError) as exc:
            raise ValueError(f"Line {number}: {exc}") from exc
    if game is None:
        raise ValueError("Line end: missing CSA side to move")
    game.comments = comments
    return game


def save_record(game, path) -> None:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".kif", ".kifu"):
        text = to_kif(game)
    elif suffix == ".csa":
        text = to_csa(game)
    elif suffix == ".json":
        text = to_json(game)
    else:
        raise ValueError("Unknown record extension")
    path.write_text(text, encoding="cp932" if suffix == ".kif" else "utf-8")


def load_record(path) -> ShogiGame:
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Record is too large")
    data = path.read_bytes()
    suffix = path.suffix.lower()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        if suffix != ".kif":
            raise ValueError("Record is not UTF-8") from None
        try:
            text = data.decode("cp932")
        except UnicodeDecodeError as exc:
            raise ValueError("Invalid KIF encoding") from exc
    if suffix in (".kif", ".kifu"):
        return from_kif(text)
    if suffix == ".csa":
        return from_csa(text)
    if suffix == ".json":
        return from_json(text)
    raise ValueError("Unknown record extension")
