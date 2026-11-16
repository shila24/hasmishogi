"""Game lifecycle, complete history, repetition and terminal precedence."""
import random
import unittest
from unittest.mock import patch

from shogi import create_game, ShogiGame
from shogi.engine import repetition_outcome
from shogi.model import K, Rules, Outcome, Position, DECLARE, decode_move, encode_move
from shogi.notation import move_to_usi, usi_to_move, parse_sfen
from tests.shogi_suite.support import setup, play_text, snapshot
from tests.shogi_suite.test_perft import CASES


class HistoryTests(unittest.TestCase):
    def test_mutual_check_event_sequence_draws(self):
        # This is an adjudicator test, not a claim of a reachable board cycle.
        # The specification explicitly defines this defensive branch.
        mutual = [(side, True) for _ in range(6) for side in (0, 1)]
        self.assertEqual(repetition_outcome(mutual), Outcome(-1, "repetition"))
        neither = [(side, False) for _ in range(6) for side in (0, 1)]
        self.assertEqual(repetition_outcome(neither), Outcome(-1, "repetition"))
        for side in (0, 1):
            unilateral = [(actor, actor == side) for actor, _ in mutual]
            self.assertEqual(repetition_outcome(unilateral), Outcome(1 - side, "perpetual_check"))
            unilateral[side + 4] = (side, False)
            self.assertEqual(repetition_outcome(unilateral), Outcome(-1, "repetition"))
        self.assertEqual(repetition_outcome([(0, True)]), Outcome(1, "perpetual_check"))

    def test_random_undo_clone_replay(self):
        rng = random.Random(2626)
        for _ in range(3):
            game = create_game()
            snapshots = [snapshot(game)]
            for ply in range(90):
                if game.outcome.terminal:
                    break
                code = rng.choice(game.legal_actions())
                before = game.position
                move = decode_move(code)
                expected_capture = before.board[move.to_sq] if move.kind == "move" else 0
                result = game.play(code)
                self.assertEqual(result.code, code)
                self.assertEqual(result.captured, expected_capture)
                self.assertEqual(result.check, game.in_check())
                self.assertEqual(game.position.turn, 1 - before.turn)
                self.assertEqual(game.ply, ply + 1)
                self.assertEqual(len(game.history_keys()), game.ply + 1)
                self.assertEqual(game.history_keys()[-1], game.position.key())
                self.assertEqual(game.position.move_number, game.initial.move_number + game.ply)
                snapshots.append(snapshot(game))
                clone = game.clone()
                self.assertEqual(snapshot(clone), snapshot(game))
                clone.undo()
                self.assertEqual(snapshot(clone), snapshots[-2])
                self.assertEqual(snapshot(game), snapshots[-1])
            replay = ShogiGame.replay(game.rules, game.initial.sfen(), game.moves)
            self.assertEqual(snapshot(replay), snapshot(game))
            for expected in reversed(snapshots[:-1]):
                self.assertTrue(game.undo())
                self.assertEqual(snapshot(game), expected)
            self.assertFalse(game.undo())

    def test_illegal_moves_are_atomic(self):
        game = create_game()
        before = snapshot(game)
        for code in (None, True, 1.0, -1, 13690, 10**20, "123", 0, DECLARE):
            self.assertFalse(game.is_legal(code))
            with self.assertRaisesRegex(ValueError, "Illegal move"):
                game.play(code)
            self.assertEqual(snapshot(game), before)
        detached = game.history_keys()
        detached.clear()
        game.legal_actions().clear()
        self.assertEqual(snapshot(game), before)

    def test_repetition_initial_counts_clone_and_undo(self):
        game = setup({"5i": K, "5a": K + 16})
        cycle = ["5i6i", "5a6a", "6i5i", "6a5a"]
        play_text(game, cycle * 2)
        self.assertFalse(game.outcome.terminal)
        clone = game.clone()
        for index, text in enumerate(cycle):
            game.play(usi_to_move(game, text))
            clone.play(usi_to_move(clone, text))
            self.assertEqual(game.outcome.terminal, index == 3)
        self.assertEqual(game.outcome, Outcome(-1, "repetition"))
        self.assertEqual(snapshot(clone), snapshot(game))
        self.assertEqual(game.history_keys().count(game.initial.key()), 4)
        self.assertTrue(game.undo())
        self.assertFalse(game.outcome.terminal)
        play_text(game, [cycle[-1]])
        self.assertEqual(snapshot(clone), snapshot(game))

    def test_perpetual_check_both_colours(self):
        base = setup({"5i": K, "5a": K + 16, "4c": 7})
        cycle = ["4c5c", "5a4a", "5c4c", "4a5a"]
        for checking_side in (0, 1):
            if checking_side:
                board = tuple(0 if cell == 0 else (cell & 15) + 16 * (1 - (cell >> 4))
                              for cell in reversed(base.initial.board))
                game = create_game(sfen=Position(board, base.initial.hands, 1, 1).sfen())
                moves = []
                work = base.clone()
                for text in cycle:
                    move = decode_move(usi_to_move(work, text))
                    moves.append(move_to_usi(encode_move(80 - move.from_sq, 80 - move.to_sq, move.promote)))
                    work.play(usi_to_move(work, text))
            else:
                game, moves = base.clone(), cycle
            for _ in range(3):
                for text in moves:
                    side = game.position.turn
                    result = game.play(usi_to_move(game, text))
                    self.assertEqual(result.check, side == checking_side)
            self.assertEqual(game.outcome, Outcome(1 - checking_side, "perpetual_check"))
            before = snapshot(game)
            game.undo()
            self.assertFalse(game.outcome.terminal)
            play_text(game, [moves[-1]])
            self.assertEqual(snapshot(game), before)

    def test_interrupted_checks_draw_from_first_occurrence(self):
        game = setup({"5i": K, "5a": K + 16, "4c": 7})
        # The checking side misses check exactly once; later cycles cannot make
        # the whole repetition span perpetual check.
        results = play_text(game, ["4c3c", "5a4a", "3c4c", "4a5a"])
        self.assertEqual([result.check for result in results], [False, False, True, False])
        play_text(game, ["4c5c", "5a4a", "5c4c", "4a5a"] * 2)
        self.assertEqual(game.outcome, Outcome(-1, "repetition"))

    def test_perpetual_check_passes_exact_span_to_adjudicator(self):
        cycle = ["4c5c", "5a4a", "5c4c", "4a5a"]
        prefix = ["3d4d", "5a6a", "4d4c", "6a5a"]
        expected_events = [(0, True), (1, False)] * 6
        for lead_in in ([], prefix):
            with self.subTest(prefix=lead_in):
                rook_square = "3d" if lead_in else "4c"
                game = setup({"5i": K, "5a": K + 16, rook_square: 7})
                with patch("shogi.engine.repetition_outcome", wraps=repetition_outcome) as adjudicate:
                    results = play_text(game, lead_in)
                    self.assertEqual([result.check for result in results], [False] * len(lead_in))
                    repeated_key = game.position.key()
                    self.assertEqual(game.history_keys().index(repeated_key), len(lead_in))
                    adjudicate.assert_not_called()
                    for index, (text, event) in enumerate(zip(cycle * 3, expected_events)):
                        self.assertEqual(game.position.turn, event[0])
                        result = game.play(usi_to_move(game, text))
                        self.assertEqual(result.check, event[1])
                        self.assertEqual(game.in_check(), event[1])
                        if index < 11:
                            self.assertFalse(game.outcome.terminal)
                            adjudicate.assert_not_called()
                    adjudicate.assert_called_once_with(expected_events)
                occurrences = [index for index, key in enumerate(game.history_keys()) if key == repeated_key]
                self.assertEqual(occurrences, [len(lead_in) + offset for offset in (0, 4, 8, 12)])
                self.assertEqual(game.outcome, Outcome(1, "perpetual_check"))

    def test_max_moves_and_resign(self):
        game = create_game(rules=Rules(2))
        play_text(game, ["7g7f", "3c3d"])
        self.assertEqual(game.outcome, Outcome(-1, "max_moves"))
        before = snapshot(game)
        for action in (lambda: game.play(0), game.resign):
            with self.assertRaisesRegex(ValueError, "Game is over"):
                action()
        self.assertEqual(snapshot(game), before)
        game.undo()
        ongoing = snapshot(game)
        game.resign()
        self.assertEqual(game.outcome, Outcome(0, "resign"))
        self.assertEqual(game.ply, 1)
        clone = game.clone()
        clone.undo()
        self.assertEqual(snapshot(clone), ongoing)
        self.assertTrue(game.undo())
        self.assertEqual(snapshot(game), ongoing)
        self.assertTrue(game.undo())
        self.assertEqual(game.ply, 0)

    def test_mate_and_stalemate_precede_move_limit(self):
        game = setup({"9a": K + 16, "5i": K, "8c": 5, "7c": 3, "8d": 7}, rules=Rules(1))
        result = game.play(usi_to_move(game, "8d9d"))
        self.assertTrue(result.check)
        self.assertEqual(game.outcome, Outcome(0, "checkmate"))
        self.assertEqual(game.checkers(), [27])
        game.undo()
        self.assertFalse(game.outcome.terminal)
        # A corner king can be trapped without a checking piece on its square.
        game = setup({"9a": K + 16, "5i": K, "8c": 5, "8e": 3}, rules=Rules(1))
        game.play(usi_to_move(game, "8e7c"))
        self.assertEqual(game.outcome, Outcome(0, "no_legal_moves"))
        self.assertFalse(game.in_check())

    def test_declaration_precedes_cap_and_sfen_number(self):
        case = CASES["declaration_cases"][0]
        game = create_game(rules=Rules(1), sfen=case["sfen"].rsplit(" ", 1)[0] + " 78")
        game.play(DECLARE)
        self.assertEqual(game.outcome, Outcome(0, "declaration"))
        self.assertEqual(game.position.move_number, 79)
        self.assertEqual(parse_sfen(game.sfen()), game.position)
        self.assertEqual(game.history_keys(), [game.initial.key()] * 2)
        game.undo()
        self.assertEqual(game.position, game.initial)
