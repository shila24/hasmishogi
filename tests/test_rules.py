import os
import random
import unittest
from hasami import create_game
from hasami.model import (Position, Rules, BOARD_MASK, neighbours, encode_action,
                          decode_action, bits)
from hasami.records import record_dict, from_record
from tests import oracle


def mask(squares):
    return sum(1 << s for s in squares)


def setup(black, white, turn=0, pending=-1):
    return Position(mask(black), mask(white), turn, pending)


MECHANICS = Rules(capture_target=0, lead_target=0, repetition_limit=0)


class RuleTests:
    backend = "python"

    def game(self, position=None, rules=None):
        game = create_game(self.backend, rules, position)
        self.addCleanup(game.close)
        return game

    def capture_case(self, black, white, source, target, taken, edge=True):
        rules = Rules(edge_capture=edge, capture_target=0, lead_target=0, repetition_limit=0)
        game = self.game(setup(black, white), rules)
        before = game.position
        action = encode_action(source, target)
        expected_board, expected_taken = oracle.apply(oracle.array_board(before), 0, action, edge)
        result = game.play(action)
        self.assertEqual(result.captured, mask(taken))
        self.assertEqual(set(bits(result.captured)), expected_taken)
        self.assertEqual(oracle.array_board(game.position), expected_board)
        after = game.position
        self.assertTrue(game.undo())
        self.assertEqual(game.position, before)
        self.assertEqual(game.last_capture, 0)
        self.assertEqual(game.play(action).captured, mask(taken))
        self.assertEqual(game.position, after)
        return game

    def test_opening_and_move_count(self):
        game = self.game()
        self.assertEqual(game.position.black.bit_count(), 9)
        self.assertEqual(game.position.white.bit_count(), 9)
        self.assertEqual(len(game.legal_actions()), 63)
        self.assertEqual(game.legal_actions(), oracle.legal(oracle.array_board(game.position), 0))

    def test_opening_depth_two_count(self):
        game = self.game()
        count = 0
        for action in game.legal_actions():
            game.play(action)
            count += len(game.legal_actions())
            game.undo()
        self.assertEqual(count, 3717)

    def test_empty_rook_has_sixteen_destinations(self):
        game = self.game(setup([40], [0]), MECHANICS)
        self.assertEqual(len(game.legal_actions()), 16)

    def test_illegal_moves_are_atomic(self):
        game = self.game()
        original = record_dict(game)
        for action in (-1, 6561, 1 << 50, True, 1.5, 72*81+72, 0*81+9,
                       72*81+73, 72*81+0, 72*81+62):
            with self.subTest(action=action):
                with self.assertRaises(ValueError):
                    game.play(action)
                self.assertEqual(record_dict(game), original)

    def test_horizontal_capture(self):
        self.capture_case([38,68], [39,40], 68, 41, [39,40])

    def test_vertical_capture(self):
        self.capture_case([13,44], [22,31], 44, 40, [22,31])

    def test_multi_direction_simultaneous_capture(self):
        self.capture_case([13,58,42,37], [22,31,49,41], 37, 40, [22,31,49,41])

    def test_diagonal_not_captured(self):
        self.capture_case([20,38], [30], 38, 40, [])

    def test_gap_breaks_capture(self):
        self.capture_case([37,68], [39], 68, 41, [])

    def test_self_insertion_survives(self):
        game = self.capture_case([67], [39,41], 67, 40, [])
        self.assertTrue(game.position.black & (1 << 40))

    def test_old_sandwich_is_not_swept(self):
        self.capture_case([39,41,80], [40], 80, 71, [])

    def test_preexisting_edge_enclosure_is_not_swept(self):
        self.capture_case([1,9,80], [0], 80, 71, [])

    def test_color_symmetry(self):
        black = [38,67,72,73,74,75,76,77,78]
        white = [39,37,3,0,1,2,4]
        game = self.game(setup(white, black, turn=1))
        game.play(67*81+40)
        self.assertEqual(game.position.pending, 1)
        game.play(3*81+39)
        self.assertFalse(game.outcome.terminal)
        self.assertEqual(game.position.pending, -1)

    def test_edge_alone_is_not_a_piece(self):
        self.capture_case([18], [0], 18, 9, [])

    def test_corner_capture(self):
        self.capture_case([1,18], [0], 18, 9, [0])

    def test_edge_group_capture(self):
        self.capture_case([3,9,10,20], [0,1,2], 20, 11, [0,1,2])

    def test_edge_group_gap_survives(self):
        self.capture_case([3,9,20], [0,1,2], 20, 11, [])

    def test_edge_rule_can_be_disabled(self):
        self.capture_case([3,9,10,20], [0,1,2], 20, 11, [], edge=False)

    def test_bent_edge_component_project_definition(self):
        self.capture_case([2,9,11,28], [0,1,10], 28, 19, [0,1,10])

    def test_no_interior_group_capture_or_cascade(self):
        self.capture_case([39,31,49,32,42,59], [40,41], 59, 50, [41])

    def test_upper_word_move_and_capture(self):
        self.capture_case([63,74], [64], 74, 65, [64])

    def test_fifth_capture_is_immediate(self):
        game = self.game(setup([38,67,72,73,74,75], [39,0,1,2,3]))
        game.play(67*81+40)
        self.assertEqual(game.outcome.winner, 0)
        self.assertEqual(game.outcome.reason, "five_captures")
        self.assertEqual(game.legal_actions(), [])
        with self.assertRaises(ValueError):
            game.play(0*81+9)

    def make_lead_game(self):
        return self.game(setup([38,67,72,73,74,75,76,77,78], [39,37,3,0,1,2,4]))

    def test_three_piece_lead_grants_one_reply(self):
        game = self.make_lead_game()
        game.play(67*81+40)
        self.assertFalse(game.outcome.terminal)
        self.assertEqual(game.position.pending, 0)
        self.assertEqual(game.position.turn, 1)
        game.play(0*81+9)
        self.assertEqual(game.outcome.winner, 0)
        self.assertEqual(game.outcome.reason, "lead_confirmed")
        game.undo()
        self.assertEqual(game.position.pending, 0)

    def test_reply_rescues_below_three(self):
        game = self.make_lead_game()
        game.play(67*81+40)
        result = game.play(3*81+39)
        self.assertEqual(result.captured, mask([38]))
        self.assertFalse(game.outcome.terminal)
        self.assertEqual(game.position.pending, -1)
        self.assertEqual(game.position.captures(0) - game.position.captures(1), 2)

    def test_four_to_three_still_loses(self):
        game = self.game(setup([38,68,72,73,74,75,76,77,78], [39,40,37,3,0,1,2]))
        game.play(68*81+41)
        self.assertEqual(game.position.pending, 0)
        game.play(3*81+39)
        self.assertEqual(game.outcome.winner, 0)
        self.assertEqual(game.outcome.reason, "lead_confirmed")

    def test_repetition_four_occurrences_and_undo(self):
        game = self.game()
        cycle = [72*81+63, 0*81+9, 63*81+72, 9*81+0]
        for i in range(12):
            game.play(cycle[i % 4])
            self.assertEqual(game.outcome.terminal, i == 11)
        self.assertEqual(game.outcome.winner, -1)
        self.assertEqual(game.outcome.reason, "repetition")
        self.assertEqual(game.history_keys().count(game.position.key()), 4)
        game.undo()
        self.assertFalse(game.outcome.terminal)
        game.play(cycle[-1])
        self.assertEqual(game.outcome.reason, "repetition")

    def test_repetition_disabled(self):
        game = self.game(rules=Rules(repetition_limit=0))
        cycle = [72*81+63, 0*81+9, 63*81+72, 9*81+0]
        for i in range(20):
            game.play(cycle[i % 4])
        self.assertFalse(game.outcome.terminal)

    def test_no_moves_policy(self):
        game = self.game(setup([0], [1,9]), MECHANICS)
        self.assertEqual(game.outcome.winner, 1)
        game = self.game(setup([0], [1,9]), Rules(capture_target=0, lead_target=0, no_moves_draw=True))
        self.assertEqual(game.outcome.winner, -1)

    def test_clone_keeps_history_and_is_independent(self):
        game = self.game()
        for a in [72*81+63, 0*81+9, 63*81+72, 9*81+0]:
            game.play(a)
        clone = game.clone()
        self.addCleanup(clone.close)
        self.assertEqual(clone.history_keys(), game.history_keys())
        clone.play(72*81+63)
        self.assertEqual(len(game.moves), 4)
        self.assertEqual(len(clone.moves), 5)

    def test_setup_validation(self):
        with self.assertRaises(ValueError):
            self.game(setup(range(9), range(20,26)))
        with self.assertRaises(ValueError):
            self.game(Position(511 << 72, 511, 1, 0))

    def test_random_positions_against_independent_array_oracle(self):
        rng = random.Random(20260912)
        for index in range(250):
            squares = rng.sample(range(81), 18)
            p = setup(squares[:9], squares[9:], index % 2)
            game = self.game(p)
            board = oracle.array_board(p)
            actions = game.legal_actions()
            self.assertEqual(actions, oracle.legal(board, p.turn))
            for action in rng.sample(actions, min(3, len(actions))):
                expected_board, expected_taken = oracle.apply(board, p.turn, action)
                result = game.play(action)
                self.assertEqual(set(bits(result.captured)), expected_taken)
                self.assertEqual(oracle.array_board(game.position), expected_board)
                game.undo()


class PythonRulesTests(RuleTests, unittest.TestCase):
    backend = "python"


def native_available():
    try:
        game = create_game("cpp")
        game.close()
        return True
    except (OSError, ValueError, RuntimeError):
        if os.environ.get("HASAMI_REQUIRE_NATIVE") == "1":
            raise
        return False


@unittest.skipUnless(native_available(), "Native library is not built")
class NativeRulesTests(RuleTests, unittest.TestCase):
    backend = "cpp"


class ModelTests(unittest.TestCase):
    def test_invalid_bitboards(self):
        for args in ((-1, 0), (1 << 81, 0), (1, 1), ((1 << 10) - 1, 0), (True, 0)):
            with self.assertRaises(ValueError):
                Position(*args)

    def test_neighbour_masks(self):
        for s in range(81):
            self.assertEqual(set(bits(neighbours(1 << s))), set(oracle.adjacent(s)))
        self.assertFalse(neighbours(1 << 8) & (1 << 9))
        self.assertTrue(neighbours(1 << 63) & (1 << 64))
        self.assertFalse(neighbours(BOARD_MASK) & ~BOARD_MASK)

    def test_encoding(self):
        for a in range(6561):
            self.assertEqual(encode_action(*decode_action(a)), a)

    def test_rule_validation(self):
        for kwargs in ({"repetition_limit": 1}, {"lead_target": -1}, {"edge_capture": 1},
                       {"capture_target": 10}, {"lead_target": True}):
            with self.assertRaises(ValueError):
                Rules(**kwargs)
