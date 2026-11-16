"""Wire conventions, strict validation and random-play move-text round trips."""
from dataclasses import FrozenInstanceError
import random
import unittest

from shogi import create_game
from shogi.model import (Rules, Position, Outcome, DECLARE, MOVE_SPACE, START_SFEN,
                         K, encode_move, encode_drop, decode_move)
from shogi.notation import (parse_sfen, square_to_usi, usi_to_square, move_to_usi,
                           usi_to_move, move_to_csa, csa_to_move, move_to_japanese,
                           japanese_to_move)
from tests.shogi_suite.support import setup, play_text, snapshot


class ModelNotationTests(unittest.TestCase):
    def test_all_move_codes(self):
        for code in range(MOVE_SPACE):
            move = decode_move(code)
            if move.kind == "move":
                rebuilt = encode_move(move.from_sq, move.to_sq, move.promote)
                self.assertEqual(move.piece, -1)
            elif move.kind == "drop":
                rebuilt = encode_drop(move.piece - 1, move.to_sq)
                self.assertEqual(move.from_sq, -1)
            else:
                rebuilt = DECLARE
                self.assertEqual((move.from_sq, move.to_sq, move.piece, move.promote), (-1, -1, -1, False))
            self.assertEqual(rebuilt, code)
        for square in range(81):
            self.assertEqual(usi_to_square(square_to_usi(square)), square)
        self.assertEqual(usi_to_square("7g"), 56)
        self.assertEqual(usi_to_square("5i"), 76)
        self.assertEqual(usi_to_square("1a"), 8)

    def test_invalid_codes_and_rules(self):
        for code in (-1, MOVE_SPACE, True, 1.0, "1", None):
            with self.assertRaises(ValueError):
                decode_move(code)
        for args in ((-1, 0, False), (0, 81, False), (True, 0, False), (0, 1, 1)):
            with self.assertRaises(ValueError):
                encode_move(*args)
        for args in ((-1, 0), (7, 0), (0, 81), (True, 0)):
            with self.assertRaises(ValueError):
                encode_drop(*args)
        for kwargs in ({"max_moves": -1}, {"max_moves": True}, {"max_moves": 1.0}, {"declaration": "other"}):
            with self.assertRaises(ValueError):
                Rules(**kwargs)
        for rules in (Rules(), Rules(0, "none"), Rules(12)):
            self.assertEqual(Rules.from_dict(rules.to_dict()), rules)
        for data in ({}, {"max_moves": 1, "declaration": "none", "extra": 0}, None):
            with self.assertRaises(ValueError):
                Rules.from_dict(data)
        with self.assertRaises(ValueError):
            create_game("unknown")

    def test_backend_selection(self):
        # native shogi bridge wired the native engine into shogi.create_game: "cpp" requires it, "auto" prefers it.
        from unittest import mock
        import shogi.native as native
        from shogi.engine import ShogiGame

        self.assertIs(type(create_game("python")), ShogiGame)
        with mock.patch.object(native, "available", return_value=False):
            self.assertIs(type(create_game("auto")), ShogiGame)
        if native.available():
            self.assertIs(type(create_game("cpp")), native.NativeShogiGame)
            self.assertIs(type(create_game("auto")), native.NativeShogiGame)

    def test_frozen_values_and_invalid_position(self):
        game = create_game()
        with self.assertRaises(FrozenInstanceError):
            game.position.turn = 1
        with self.assertRaises(FrozenInstanceError):
            game.rules.max_moves = 0
        p = game.position
        invalid = ((p.board[:-1], p.hands, 0, 1), (tuple([16] + list(p.board[1:])), p.hands, 0, 1),
                   (list(p.board), p.hands, 0, 1), (p.board, ([0] * 7, [0] * 7), 0, 1),
                   (p.board, p.hands, True, 1), (p.board, p.hands, 0, 0))
        for args in invalid:
            with self.assertRaises(ValueError):
                Position(*args)
        for args in ((-2, "resign"), (-1, "ongoing"), (0, "max_moves"), (True, "resign")):
            with self.assertRaises(ValueError):
                Outcome(*args)

    def test_sfen_roundtrip_and_validation(self):
        self.assertEqual(parse_sfen(START_SFEN).sfen(), START_SFEN)
        valid = "4k4/9/9/2+B6/9/9/9/9/4K4 w R2Pbs 123"
        self.assertEqual(parse_sfen(valid).sfen(), valid)
        invalid = ["", START_SFEN.replace(" b ", " x "), START_SFEN[:-1] + "0",
                   START_SFEN[:-1] + "-1", START_SFEN[:-1] + "1.0",
                   "9/9/9/9/9/9/9/9/9 b - 1", "4k4/9/9/9/9/9/9/9/3KK4 b - 1",
                   "4k4/9/9/9/9/9/9/9/4+K4 b - 1", "4k4/9/9/9/9/9/9/9/4K3+ b - 1",
                   "4k4/9/9/9/9/9/9/9/4X4 b - 1", "4k4/9/9/9/9/9/9/9/4K5 b - 1",
                   "4k4/9/9/9/9/9/9/9/4K4 b 19P 1", "4k4/9/9/9/9/9/9/9/4K4 b 3R 1",
                   "4k4/9/9/9/9/9/9/9/4K4 b P2P 1", "4k4/9/9/9/9/9/9/9/4K4 b 0P 1",
                   "4k4/9/9/9/9/9/9/9/4K4 b K 1", "4k4/9/9/9/9/9/9/9/4K4 b +P 1",
                   "P3k4/9/9/9/9/9/9/9/4K4 b - 1", "4k4/N8/9/9/9/9/9/9/4K4 b - 1",
                   "4k4/9/9/9/9/9/9/9/l3K4 b - 1", "4k4/9/9/9/9/9/9/n8/4K4 b - 1",
                   "4k4/9/4P4/9/4P4/9/9/9/4K4 b - 1",
                   "4k4/9/4R4/9/9/9/9/9/4K4 b - 1"]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_sfen(text)

    def test_random_notation_roundtrips(self):
        rng = random.Random(260113)
        for _ in range(3):
            game = create_game()
            for _ in range(100):
                if game.outcome.terminal:
                    break
                before = snapshot(game)
                self.assertEqual(parse_sfen(game.sfen()), game.position)
                for code in rng.sample(game.legal_actions(), min(3, len(game.legal_actions()))):
                    self.assertEqual(usi_to_move(game, move_to_usi(code)), code)
                    self.assertEqual(csa_to_move(game, move_to_csa(game, code)), code)
                    self.assertEqual(japanese_to_move(game, move_to_japanese(game, code)), code)
                    self.assertEqual(japanese_to_move(game, move_to_japanese(game, code, kif=True)), code)
                self.assertEqual(snapshot(game), before)
                game.play(rng.choice(game.legal_actions()))

    def test_known_text_and_ambiguity(self):
        game = create_game()
        code = usi_to_move(game, "7g7f")
        self.assertEqual(move_to_japanese(game, code), "▲７六歩")
        self.assertEqual(move_to_japanese(game, code, kif=True), "７六歩(77)")
        self.assertEqual(move_to_csa(game, code), "+7776FU")
        game = setup({"5i": K, "5a": K + 16, "4g": 5, "6g": 5})
        self.assertEqual(move_to_japanese(game, usi_to_move(game, "4g5f")), "▲５六金右")
        self.assertEqual(move_to_japanese(game, usi_to_move(game, "6g5f")), "▲５六金左")
        with self.assertRaises(ValueError):
            japanese_to_move(game, "▲５六金")
        game = setup({"5i": K, "5a": K + 16, "5d": 5, "5f": 5})
        self.assertEqual(move_to_japanese(game, usi_to_move(game, "5d5e")), "▲５五金引")
        self.assertEqual(move_to_japanese(game, usi_to_move(game, "5f5e")), "▲５五金直")
        game = setup({"5i": K, "9a": K + 16, "2d": 6})
        self.assertEqual(move_to_japanese(game, usi_to_move(game, "2d3c")), "▲３三角不成")
        self.assertEqual(move_to_japanese(game, usi_to_move(game, "2d3c+")), "▲３三角成")
        game = setup({"5i": K, "5a": K + 16, "5e": 1, "5c": 17})
        play_text(game, ["5e5d"])
        self.assertEqual(move_to_japanese(game, usi_to_move(game, "5c5d")), "△同　歩")

    def test_bad_move_text_is_atomic(self):
        game = create_game()
        before = snapshot(game)
        for parse, texts in ((usi_to_move, ("resign", "7g7e", "P*5e", "0a1b", "7g7f++", "win")),
                             (csa_to_move, ("-7776FU", "+7776TO", "+7776XX", "+0055OU", "%KACHI", "%TORYO")),
                             (japanese_to_move, ("▲７六飛", "７六歩(76)", "投了", "同歩"))):
            for text in texts:
                with self.subTest(text=text), self.assertRaises(ValueError):
                    parse(game, text)
                self.assertEqual(snapshot(game), before)
