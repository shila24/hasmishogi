"""Every oracle position and random trajectories exercise the policy bijection."""
import random
import unittest

from shogi import create_game
from shogi.encoding import (encode, move_label, label_to_move, legal_labels,
                            ACTION_SIZE, DECLARE_INDEX, ENCODING_ID)
from shogi.model import K, DECLARE, Rules
from shogi.notation import usi_to_move, usi_to_square
from tests.shogi_suite.test_perft import ROWS, CASES
from tests.shogi_suite.support import setup, play_text


class EncodingTests(unittest.TestCase):
    def assert_bijection(self, game):
        actions, labels = game.legal_actions(), legal_labels(game)
        self.assertEqual(labels, [move_label(game, code) for code in actions])
        self.assertEqual(len(labels), len(set(labels)))
        self.assertTrue(all(type(label) is int and 0 <= label < ACTION_SIZE for label in labels))
        self.assertEqual([label_to_move(game, label) for label in labels], actions)

    def test_every_vector_bijection(self):
        for row in ROWS:
            with self.subTest(sfen=row["sfen"]):
                self.assert_bijection(create_game(sfen=row["sfen"]))

    def test_random_play_bijection(self):
        game, rng = create_game(), random.Random(102)
        for _ in range(160):
            self.assert_bijection(game)
            if game.outcome.terminal:
                break
            game.play(rng.choice(game.legal_actions()))

    def test_hand_computed_planes(self):
        game = setup({"5i": K, "5a": K + 16, "7f": 9, "3d": 22},
                     turn=1, hands=((2, 1, 0, 0, 0, 0, 1), (3, 0, 1, 0, 0, 1, 0)), rules=Rules(4))
        values = encode(game, 1)
        expected = [0.0] * (49 * 81)
        for plane, square in ((7, "5a"), (5, "3d"), (21, "5i"), (22, "7f")):
            expected[plane * 81 + 80 - usi_to_square(square)] = 1.0
        for plane, value in ((28, 3 / 18), (30, 1 / 4), (33, 1 / 2),
                             (35, 2 / 18), (36, 1 / 4), (41, 1 / 2), (42, 1.0), (48, 1.0)):
            expected[plane * 81:(plane + 1) * 81] = [value] * 81
        self.assertEqual(values, expected)
        self.assertTrue(all(type(value) is float for value in values))
        with self.assertRaises(ValueError):
            encode(game, 0)
        with self.assertRaises(ValueError):
            encode(game, True)
        game = setup({"5i": K, "5a": K + 16, "4c": 7})
        play_text(game, ["4c5c"])
        self.assertEqual(encode(game, 1)[43 * 81:44 * 81], [1.0] * 81)

    def test_repetition_and_ply_planes(self):
        game = setup({"5i": K, "5a": K + 16}, rules=Rules(12))
        cycle = ["5i6i", "5a6a", "6i5i", "6a5a"]
        for occurrence in range(4):
            values = encode(game, game.position.turn)
            for previous in range(1, 4):
                self.assertEqual(values[(43 + previous) * 81:(44 + previous) * 81],
                                 [float(occurrence >= previous)] * 81)
            self.assertEqual(values[47 * 81:48 * 81], [occurrence / 3] * 81)
            if occurrence < 3:
                play_text(game, cycle)
        self.assertEqual(game.outcome.reason, "repetition")
        self.assertEqual(legal_labels(game), [])
        unlimited = create_game(rules=Rules(0))
        unlimited.play(unlimited.legal_actions()[0])
        self.assertEqual(encode(unlimited, 1)[47 * 81:48 * 81], [0.0] * 81)

    def test_labels_by_hand_and_errors(self):
        game = create_game()
        code = usi_to_move(game, "7g7f")
        self.assertEqual(move_label(game, code), 47)
        play_text(game, ["7g7f"])
        self.assertEqual(move_label(game, usi_to_move(game, "3c3d")), 47)
        game = setup({"5i": K, "5a": K + 16, "5d": 3}, hands=((1, 0, 0, 0, 0, 0, 0), (0,) * 7))
        self.assertEqual(move_label(game, usi_to_move(game, "5d6b+")), 18 * 81 + 12)
        self.assertEqual(move_label(game, usi_to_move(game, "P*5e")), 20 * 81 + 40)
        game = create_game(sfen=CASES["declaration_cases"][0]["sfen"])
        self.assertEqual(move_label(game, DECLARE), DECLARE_INDEX)
        self.assertEqual(label_to_move(game, DECLARE_INDEX), DECLARE)
        self.assertEqual(ENCODING_ID, "shogi-enc-v1")
        for index in (-1, ACTION_SIZE, True, 1.0):
            with self.assertRaises(ValueError):
                label_to_move(game, index)
        with self.assertRaises(ValueError):
            label_to_move(create_game(), DECLARE_INDEX)
        with self.assertRaises(ValueError):
            move_label(create_game(), DECLARE)
