"""Bit-exact native/reference lockstep, including complete scaled random games."""
from array import array
import json
import os
from pathlib import Path
import random
import time
import unittest

from shogi import ShogiGame
from shogi.encoding import encode, legal_labels
from shogi.model import DECLARE, Outcome, Position, Rules, encode_move, decode_move
from shogi.native import NativeShogiGame, available
from shogi.notation import usi_to_move
from shogi.perft import perft
from tests.shogi_suite.support import setup

NATIVE = available()
if not NATIVE and os.environ.get("HASAMI_REQUIRE_NATIVE") == "1":
    raise RuntimeError("Native shogi library is required")
VECTORS = Path(__file__).with_name("vectors")
ROWS = [json.loads(line) for line in (VECTORS / "perft_vectors.jsonl").read_text().splitlines()]
CASES = json.loads((VECTORS / "rule_cases.json").read_text())


@unittest.skipUnless(NATIVE, "Native shogi library is unavailable")
class NativeParityTests(unittest.TestCase):
    def assert_parity(self, python, native):
        self.assertEqual(native.rules, python.rules)
        self.assertEqual(native.initial, python.initial)
        self.assertEqual(native.position, python.position)
        self.assertEqual(native.outcome, python.outcome)
        self.assertEqual(native.ply, python.ply)
        self.assertEqual(native.moves, python.moves)
        self.assertEqual(native.history_keys(), python.history_keys())
        self.assertEqual(native.sfen(), python.sfen())
        actions = python.legal_actions()
        self.assertEqual(native.legal_actions(), actions)
        labels = legal_labels(python)
        self.assertEqual(native.legal_labels(), labels)
        self.assertEqual(len(labels), len(set(labels)))
        self.assertEqual(native.in_check(), python.in_check())
        for side in (0, 1):
            self.assertEqual(native.in_check(side), python.in_check(side))
        self.assertEqual(native.checkers(), python.checkers())
        self.assertEqual(native.can_declare(), python.can_declare())
        planes = array("f", [0]) * (49 * 81)
        native.encode_into(planes)
        self.assertEqual(planes.tobytes(), array("f", encode(python, python.position.turn)).tobytes())
        mask = array("f", [0]) * 2188
        native.legal_mask_into(mask)
        expected = array("f", [0]) * 2188
        for label in labels:
            expected[label] = 1
        self.assertEqual(mask.tobytes(), expected.tobytes())
        # Invalid codes must preserve every observable part of the native state.
        before = (native.position, native.outcome, native.moves, native.history_keys(),
                  native.legal_actions(), native.legal_labels(), native.sfen())
        for code in (-1, 0):  # Code zero is in range and reaches sg_play (a same-square move).
            self.assertFalse(native.is_legal(code))
            for game in (python, native):
                with self.assertRaisesRegex(ValueError, "^Game is over$" if game.outcome.terminal else "^Illegal move$"):
                    game.play(code)
        fresh = native.clone()
        try:
            self.assertEqual(before, (fresh.position, fresh.outcome, fresh.moves, fresh.history_keys(),
                                      fresh.legal_actions(), fresh.legal_labels(), fresh.sfen()))
        finally:
            fresh.close()
        return actions

    def test_every_vector_and_rule_position(self):
        rng = random.Random(2701)
        rows = ROWS + CASES["move_cases"] + CASES["declaration_cases"]
        for row in rows:
            with self.subTest(sfen=row["sfen"]):
                python = ShogiGame(sfen=row["sfen"])
                native = NativeShogiGame(sfen=row["sfen"])
                try:
                    for _ in range(5):
                        actions = self.assert_parity(python, native)
                        if not actions:
                            break
                        code = rng.choice(actions)
                        self.assertEqual(native.play(code), python.play(code))
                    self.assert_parity(python, native)
                    while python.undo():
                        self.assertTrue(native.undo())
                        self.assert_parity(python, native)
                    self.assertFalse(native.undo())
                finally:
                    native.close()

    def test_forty_depth_three_vectors(self):
        for row in sorted(ROWS, key=lambda row: row["perft"][2])[:40]:
            with self.subTest(sfen=row["sfen"]):
                python = ShogiGame(sfen=row["sfen"])
                native = NativeShogiGame(sfen=row["sfen"])
                try:
                    expected = perft(python, 3)
                    self.assertEqual(expected, row["perft"][2])
                    self.assertEqual(native.perft(3), expected)
                    self.assert_parity(python, native)
                finally:
                    native.close()

    def test_seeded_complete_random_games(self):
        scale = int(os.environ.get("SHOGI_PARITY_SCALE", "1"))
        self.assertGreaterEqual(scale, 1)
        games, plies, outcomes = 200 * scale, 0, {}
        start = time.perf_counter()
        for index in range(games):
            rng = random.Random(270000 + index)
            # All games run to a rule ending, including a cohort with the full 320-ply cap.
            rules = Rules((8, 16, 32, 64, 320)[index % 5], "none" if index % 3 == 0 else "csa27")
            sfen = None if index % 2 == 0 else ROWS[rng.randrange(len(ROWS))]["sfen"]
            python, native = ShogiGame(rules, sfen), NativeShogiGame(rules, sfen)
            try:
                while True:
                    actions = self.assert_parity(python, native)
                    if not actions:
                        break
                    if python.ply == 3:
                        clone = native.clone()
                        reference = python.clone()
                        try:
                            code = rng.choice(actions)
                            self.assertEqual(clone.play(code), reference.play(code))
                            self.assert_parity(reference, clone)
                            self.assert_parity(python, native)
                            self.assertTrue(clone.undo())
                            self.assert_parity(python, clone)
                        finally:
                            clone.close()
                    code = rng.choice(actions)
                    self.assertTrue(native.is_legal(code))
                    self.assertEqual(native.play(code), python.play(code))
                    plies += 1
                self.assertTrue(python.outcome.terminal)
                outcomes[python.outcome.reason] = outcomes.get(python.outcome.reason, 0) + 1
                replay = NativeShogiGame.replay(rules, python.initial.sfen(), python.moves)
                try:
                    self.assert_parity(python, replay)
                finally:
                    replay.close()
            finally:
                native.close()
        print(json.dumps({"shogi_parity_games": games, "plies": plies, "outcomes": outcomes,
                          "seconds": round(time.perf_counter() - start, 3)}, sort_keys=True), flush=True)

    def test_repetition_and_perpetual_check_both_sides(self):
        bases = [
            (setup({"5i": 8, "5a": 24}), ["5i6i", "5a6a", "6i5i", "6a5a"] * 3, Outcome(-1, "repetition")),
            (setup({"5i": 8, "5a": 24, "4c": 7}), ["4c5c", "5a4a", "5c4c", "4a5a"] * 3,
             Outcome(1, "perpetual_check")),
            (setup({"5i": 8, "5a": 24, "4c": 7}),
             ["4c3c", "5a4a", "3c4c", "4a5a"] + ["4c5c", "5a4a", "5c4c", "4a5a"] * 2,
             Outcome(-1, "repetition")),
            (setup({"5i": 8, "5a": 24, "3d": 7}),
             ["3d4d", "5a6a", "4d4c", "6a5a"] + ["4c5c", "5a4a", "5c4c", "4a5a"] * 3,
             Outcome(1, "perpetual_check")),
        ]
        for base, texts, outcome in bases:
            codes = []
            for text in texts:
                code = usi_to_move(base, text)
                codes.append(code)
                base.play(code)
            for rotate in (False, True):
                initial = base.initial
                expected = outcome
                actions = codes
                if rotate:
                    board = tuple(0 if not cell else (cell & 15) + 16 * (1 - (cell >> 4))
                                  for cell in reversed(initial.board))
                    initial = Position(board, initial.hands[::-1], 1 - initial.turn, initial.move_number)
                    actions = [encode_move(80 - decode_move(c).from_sq, 80 - decode_move(c).to_sq,
                                           decode_move(c).promote) for c in codes]
                    if expected.winner >= 0:
                        expected = Outcome(1 - expected.winner, expected.reason)
                rules = Rules(len(actions))  # Repetition must beat the simultaneous move cap.
                python, native = ShogiGame(rules, initial.sfen()), NativeShogiGame(rules, initial.sfen())
                try:
                    self.assert_parity(python, native)
                    for code in actions:
                        self.assertEqual(native.play(code), python.play(code))
                        self.assert_parity(python, native)
                    self.assertEqual(native.outcome, expected)
                    self.assertTrue(native.undo())
                    python.undo()
                    self.assert_parity(python, native)
                    self.assertEqual(native.play(actions[-1]), python.play(actions[-1]))
                    self.assert_parity(python, native)
                finally:
                    native.close()

    def test_terminal_precedence_declaration_and_resign(self):
        games = [
            (setup({"9a": 24, "5i": 8, "8c": 5, "7c": 3, "8d": 7}, rules=Rules(1)), "8d9d", "checkmate"),
            (setup({"9a": 24, "5i": 8, "8c": 5, "8e": 3}, rules=Rules(1)), "8e7c", "no_legal_moves"),
        ]
        for case in CASES["declaration_cases"]:
            if case["declarable"]:
                games.append((ShogiGame(Rules(1), case["sfen"]), "win", "declaration"))
        for python, text, reason in games:
            native = NativeShogiGame(python.rules, python.sfen())
            try:
                self.assert_parity(python, native)
                code = usi_to_move(python, text)
                self.assertEqual(native.play(code), python.play(code))
                self.assertEqual(native.outcome.reason, reason)
                self.assert_parity(python, native)
                self.assertTrue(native.undo())
                python.undo()
                self.assert_parity(python, native)
                python.resign()
                native.resign()
                self.assert_parity(python, native)
                clone = native.clone()
                try:
                    self.assertTrue(clone.undo())
                    python.undo()
                    self.assert_parity(python, clone)
                    self.assertEqual(native.outcome.reason, "resign")
                    self.assertTrue(native.undo())
                    self.assert_parity(python, native)
                finally:
                    clone.close()
            finally:
                native.close()
