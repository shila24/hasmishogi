"""Compare the engine with immutable, independently generated oracle facts."""
import json
import os
from pathlib import Path
import unittest

from shogi import create_game
from shogi.notation import move_to_usi, usi_to_move
from shogi.perft import perft

VECTORS = Path(__file__).with_name("vectors")
ROWS = [json.loads(line) for line in (VECTORS / "perft_vectors.jsonl").read_text().splitlines()]
CASES = json.loads((VECTORS / "rule_cases.json").read_text())


class OracleTests(unittest.TestCase):
    def test_all_legal_lists_and_depths_one_two(self):
        for row in ROWS:
            with self.subTest(sfen=row["sfen"]):
                game = create_game(sfen=row["sfen"])
                self.assertEqual(sorted(map(move_to_usi, game.legal_actions())), row["legal_usi"])
                self.assertEqual(perft(game, 1), row["perft"][0])
                self.assertEqual(perft(game, 2), row["perft"][1])

    def test_forty_depth_three_vectors(self):
        # Smallest trees keep the default pure-Python suite within its time budget.
        for row in sorted(ROWS, key=lambda row: row["perft"][2])[:40]:
            with self.subTest(sfen=row["sfen"]):
                self.assertEqual(perft(create_game(sfen=row["sfen"]), 3), row["perft"][2])

    def test_start_depths_one_three(self):
        game = create_game()
        for depth, count in enumerate((1, 30, 900, 25470)):
            self.assertEqual(perft(game, depth), count)
        self.assertEqual(game.ply, 0)

    @unittest.skipUnless(os.environ.get("SHOGI_SLOW") == "1", "Set SHOGI_SLOW=1 for start perft(4)")
    def test_start_depth_four(self):
        self.assertEqual(perft(create_game(), 4), 719731)

    def test_move_cases(self):
        for case in CASES["move_cases"]:
            with self.subTest(case=case["name"]):
                game = create_game(sfen=case["sfen"])
                self.assertEqual(len(game.legal_actions()), case["perft1"])
                if case["legal"]:
                    self.assertTrue(game.is_legal(usi_to_move(game, case["move"])))
                else:
                    with self.assertRaises(ValueError):
                        usi_to_move(game, case["move"])

    def test_declaration_cases(self):
        from shogi.model import DECLARE, Outcome, Rules
        for case in CASES["declaration_cases"]:
            with self.subTest(case=case["name"]):
                game = create_game(sfen=case["sfen"])
                self.assertEqual(game.can_declare(), case["declarable"])
                self.assertEqual(game.is_legal(DECLARE), case["declarable"])
                disabled = create_game(rules=Rules(declaration="none"), sfen=case["sfen"])
                self.assertFalse(disabled.can_declare())
                if case["declarable"]:
                    before = game.position
                    game.play(DECLARE)
                    self.assertEqual(game.position.board, before.board)
                    self.assertEqual(game.position.hands, before.hands)
                    self.assertEqual(game.position.turn, before.turn)
                    self.assertEqual(game.position.move_number, before.move_number + 1)
                    self.assertEqual(game.history_keys(), [before.key(), before.key()])
                    self.assertEqual(game.outcome, Outcome(before.turn, "declaration"))
                    self.assertEqual(game.moves, [DECLARE])
                    self.assertEqual(game.legal_actions(), [])
                    self.assertTrue(game.undo())
                    self.assertEqual(game.position, before)
