"""Lossless record replay, interoperability examples and malformed input tests."""
import copy
from pathlib import Path
import random
import tempfile
import unittest

from shogi import create_game
from shogi.model import Rules, K, DECLARE
from shogi.records import (record_dict, from_record, to_json, from_json, to_kif,
                           from_kif, to_csa, from_csa, save_record, load_record)
from tests.shogi_suite.support import setup, play_text, snapshot
from tests.shogi_suite.test_perft import CASES, ROWS


class RecordTests(unittest.TestCase):
    def assert_roundtrip(self, game):
        for write, read in ((to_json, from_json), (to_kif, from_kif), (to_csa, from_csa)):
            with self.subTest(format=write.__name__, outcome=game.outcome):
                restored = read(write(game))
                self.assertEqual(snapshot(restored), snapshot(game))
                self.assertEqual(restored.rules, game.rules)
                self.assertEqual(restored.comments, game.comments)
                self.assertEqual(record_dict(restored), record_dict(game))

    def test_seeded_games_and_explicit_starts(self):
        rng = random.Random(113)
        for sfen in (None, ROWS[7]["sfen"], ROWS[21]["sfen"]):
            game = create_game(rules=Rules(70, "none"), sfen=sfen)
            game.comments = ["先手：テスト", "unknown-header: preserved"]
            for _ in range(70):
                if game.outcome.terminal:
                    break
                game.play(rng.choice(game.legal_actions()))
            self.assert_roundtrip(game)

    def test_every_end_reason_and_ongoing(self):
        self.assert_roundtrip(create_game())
        resigned = create_game()
        play_text(resigned, ["7g7f"])
        resigned.resign()
        self.assert_roundtrip(resigned)
        repeated = setup({"5i": K, "5a": K + 16})
        play_text(repeated, ["5i6i", "5a6a", "6i5i", "6a5a"] * 3)
        self.assert_roundtrip(repeated)
        perpetual = setup({"5i": K, "5a": K + 16, "4c": 7})
        play_text(perpetual, ["4c5c", "5a4a", "5c4c", "4a5a"] * 3)
        self.assert_roundtrip(perpetual)
        mate = setup({"9a": K + 16, "5i": K, "8c": 5, "7c": 3, "8d": 7})
        play_text(mate, ["8d9d"])
        self.assert_roundtrip(mate)
        trapped = setup({"9a": K + 16, "5i": K, "8c": 5, "8e": 3})
        play_text(trapped, ["8e7c"])
        self.assert_roundtrip(trapped)
        declared = create_game(sfen=CASES["declaration_cases"][0]["sfen"])
        declared.play(DECLARE)
        self.assert_roundtrip(declared)

    def test_record_start_must_be_an_explicit_sfen(self):
        # Review finding: a null SFEN used to fall back to the start position and replay silently.
        game = create_game(sfen="lnsgkgsnl/1r5b1/pp1pppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
        from shogi.notation import usi_to_move
        game.play(usi_to_move(game, "7g7f"))
        data = record_dict(game)
        self.assertEqual(from_record(copy.deepcopy(data)).sfen(), game.sfen())
        for bad in (None, "", "   ", 7):
            corrupted = copy.deepcopy(data)
            corrupted["sfen"] = bad
            with self.assertRaisesRegex(ValueError, "SFEN"):
                from_record(corrupted)

    def test_comments_cannot_become_configuration(self):
        game = create_game()
        game.comments = ['rules:{"max_moves":1,"declaration":"none"}',
                         "move_number:99", "comment:literal text", "first line\nsecond line"]
        self.assert_roundtrip(game)

    def test_files_utf8_and_cp932(self):
        game = create_game(rules=Rules(3))
        game.comments = ["棋譜の読込試験"]
        play_text(game, ["7g7f", "3c3d", "8h2b+"])
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            for suffix in (".kifu", ".kif", ".csa", ".json"):
                path = Path(directory) / ("game" + suffix)
                save_record(game, path)
                if suffix == ".kif":
                    self.assertIn("棋譜", path.read_bytes().decode("cp932"))
                self.assertEqual(record_dict(load_record(path)), record_dict(game))
            path = Path(directory) / "bad.kifu"
            path.write_bytes(b"\xff")
            with self.assertRaises(ValueError):
                load_record(path)
            with self.assertRaises(ValueError):
                save_record(game, Path(directory) / "game.txt")

    def test_external_even_start_examples(self):
        kif = "先手：Alice\n後手：Bob\n手合割：平手\n手数----指手---------消費時間--\n1 ７六歩(77) ( 0:01/00:00:01)\n2 ３四歩(33)\n3 ２二角成(88)\n4 同　銀(31)\n5 投了\n"
        csa = "V2.2\nN+Alice\nN-Bob\nPI\n+\n+7776FU\nT1\n-3334FU\n+8822UM\n-3122GI\n%TORYO\n"
        left, right = from_kif(kif), from_csa(csa)
        self.assertEqual(snapshot(left), snapshot(right))
        self.assertIn("先手：Alice", left.comments)
        self.assertIn("N+Alice", right.comments)
        self.assert_roundtrip(left)
        self.assert_roundtrip(right)

    def test_kif_board_diagram(self):
        text = """先手の持駒：歩二 香
後手の持駒：角
  ９ ８ ７ ６ ５ ４ ３ ２ １
+---------------------------+
| ・ ・ ・ ・v玉 ・ ・ ・ ・|一
| ・ ・ ・ ・ ・ ・ ・ ・ ・|二
| ・ ・ ・ ・ ・ ・ ・ ・ ・|三
| ・ ・ ・ ・ ・ ・ ・ ・ ・|四
| ・ ・ ・ ・ ・ ・ ・ ・ ・|五
| ・ ・ ・ ・ ・ ・ ・ ・ ・|六
| ・ ・ ・ ・ ・ ・ ・ ・ ・|七
| ・ ・ ・ ・ ・ ・ ・ ・ ・|八
| ・ ・ ・ ・ 玉 ・ ・ ・ ・|九
+---------------------------+
1 ５五歩打
"""
        game = from_kif(text)
        expected = setup({"5i": K, "5a": K + 16}, hands=((2, 1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1, 0)))
        play_text(expected, ["P*5e"])
        self.assertEqual(snapshot(game), snapshot(expected))

    def test_json_tampering(self):
        game = create_game(rules=Rules(2))
        play_text(game, ["7g7f", "3c3d"])
        base = record_dict(game)
        mutations = [("format", "other"), ("moves", ["7g7e"]), ("moves", "7g7f"),
                     ("outcome", {"winner": 0, "reason": "resign"}),
                     ("outcome", {"winner": -2, "reason": "ongoing"}),
                     ("rules", {"max_moves": True, "declaration": "none"}),
                     ("comments", [123]), ("sfen", "bad")]
        for field, value in mutations:
            data = copy.deepcopy(base)
            data[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                from_record(data)
        with self.assertRaises(ValueError):
            from_json("{")
        with self.assertRaises(ValueError):
            from_record({**base, "extra": True})

    def test_line_number_errors_and_false_terminal_markers(self):
        for read, text, line in ((from_kif, "手合割：平手\n1 ７五歩(77)\n", 2),
                                 (from_kif, "手合割：平手\n1 千日手\n", 2),
                                 (from_kif, "1 ７六歩(77)\n3 ３四歩(33)\n", 2),
                                 (from_kif, "手合割：平手\n1７六歩(77)\n", 2),
                                 (from_csa, "PI\n+\n+7775FU\n", 3),
                                 (from_csa, "PI\n+\n%SENNICHITE\n", 3),
                                 (from_csa, "PI\n+\n%TORYO\n+7776FU\n", 4)):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, f"Line {line}:"):
                read(text)

    def test_csa_malformed_moves_are_not_unknown_headers(self):
        for move in ("3334FU", "3334", "3334XX", "-3334", "-3334XX"):
            with self.subTest(move=move), self.assertRaisesRegex(ValueError, r"^Line 4:"):
                from_csa(f"PI\n+\n+7776FU\n{move}\n%TORYO\n")

    def test_csa_valid_moves_and_unknown_headers_preserve_result(self):
        game = from_csa("PI\n+\n+7776FU\n-3334FU\n%TORYO\n")
        self.assertEqual(record_dict(game)["moves"], ["7g7f", "3c3d"])
        self.assertEqual(game.outcome.winner, 1)
        self.assertEqual(game.outcome.reason, "resign")
        headers = ["N+Alice", "$EVENT:Regression", "X-UNKNOWN:preserved"]
        text = "\n".join(headers) + "\nPI\n+\n+7776FU\n-3334FU\n%TORYO\n"
        restored = from_csa(text)
        self.assertEqual(snapshot(restored), snapshot(game))
        self.assertEqual(restored.comments, headers)
