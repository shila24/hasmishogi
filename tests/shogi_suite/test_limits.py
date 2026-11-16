"""Implementation limits shared with the native engine's 32-bit ABI fields (shogi/model.py)."""
import unittest
from unittest import mock

from shogi import create_game
from shogi.model import MAX_GAME_MOVES, MAX_SFEN_MOVE_NUMBER, Rules
from shogi.notation import usi_to_move

BOARD = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b -"


class LimitTests(unittest.TestCase):
    def test_sfen_move_numbers_are_bounded(self):
        self.assertEqual(create_game(sfen=f"{BOARD} {MAX_SFEN_MOVE_NUMBER}").position.move_number, MAX_SFEN_MOVE_NUMBER)
        for number in (MAX_SFEN_MOVE_NUMBER + 1, 2**31, 2**63):
            with self.assertRaisesRegex(ValueError, "move number"):
                create_game(sfen=f"{BOARD} {number}")

    def test_max_moves_rule_is_bounded(self):
        self.assertEqual(Rules(max_moves=MAX_GAME_MOVES).max_moves, MAX_GAME_MOVES)
        with self.assertRaisesRegex(ValueError, "max_moves"):
            Rules(max_moves=MAX_GAME_MOVES + 1)

    def test_a_game_never_exceeds_the_move_limit(self):
        game = create_game(rules=Rules(max_moves=0))
        with mock.patch("shogi.engine.MAX_GAME_MOVES", 2):
            game.play(usi_to_move(game, "7g7f"))
            game.play(usi_to_move(game, "3c3d"))
            before = game.sfen()
            with self.assertRaisesRegex(ValueError, "Move limit exceeded"):
                game.play(usi_to_move(game, "8h2b+"))
            self.assertEqual(game.sfen(), before)

    def test_move_numbers_fit_the_native_32_bit_fields(self):
        self.assertLess(MAX_SFEN_MOVE_NUMBER + MAX_GAME_MOVES, 2**31)


if __name__ == "__main__":
    unittest.main()
