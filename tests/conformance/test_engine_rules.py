"""E1-E12 conformance checks for the hasami-lab-v1 engine contract."""

from contextlib import closing, ExitStack
from copy import deepcopy
import json
import random
import unittest

from hasami import create_game
from hasami.model import Outcome, Position, Rules
from hasami.observation import observe
from hasami.records import from_record, record_dict
from tests import oracle
from tests.test_rules import MECHANICS, mask, native_available, setup
from tests.conformance._support import (
    available_backends, game_snapshot, oracle_successor, perft,
    random_position, swap_players,
)


POSITION_COUNT = 240
GAME_COUNT = 60
PLY_CAP = 240
SEED = 20260914
CYCLE = (72 * 81 + 63, 0 * 81 + 9, 63 * 81 + 72, 9 * 81 + 0)


class EngineConformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backends = available_backends()
        rng = random.Random(SEED)
        cls.positions = tuple(random_position(rng, i % 2) for i in range(POSITION_COUNT))

    def assert_step(self, game, action):
        before = game.position
        board, captured = oracle_successor(before, action, game.rules.edge_capture)
        result = game.play(action)
        self.assertEqual(result.action, action)
        self.assertEqual(result.captured, captured)
        self.assertEqual(game.last_capture, captured)
        self.assertEqual(oracle.array_board(game.position), board)
        self.assertEqual(game.position.turn, 1 - before.turn)
        self.assertEqual(result.outcome, game.outcome)
        if result.outcome.terminal:
            self.assertEqual(game.position.pending, -1)
            self.assertEqual(game.legal_actions(), [])
        return result

    def assert_terminal(self, game, winner, reason):
        self.assertEqual(game.outcome, Outcome(winner, reason))
        self.assertEqual(game.position.pending, -1)
        self.assertEqual(game.legal_actions(), [])
        self.assertEqual(game.history_keys()[-1], game.position.key())

    def test_e1_no_jumping(self):
        """E1: Every legal action follows an empty orthogonal path on 240 setups."""
        for index, position in enumerate(self.positions):
            board = oracle.array_board(position)
            reachable = set(oracle.legal(board, position.turn))
            for backend in self.backends:
                with self.subTest(backend=backend, position=index):
                    with closing(create_game(backend, position=position)) as game:
                        actions = game.legal_actions()
                        self.assertTrue(actions)
                        for action in actions:
                            source, target = divmod(action, 81)
                            self.assertNotEqual(source, target)
                            self.assertTrue(source // 9 == target // 9 or source % 9 == target % 9)
                            self.assertEqual(board[source], position.turn)
                            self.assertEqual(board[target], -1)
                            # Oracle reachability checks every intervening square independently.
                            self.assertIn(action, reachable)

    def test_e2_slide_completeness(self):
        """E2: Legal actions exactly equal oracle rays, including board/word edges."""
        positions = [(position, Rules()) for position in self.positions]
        positions.extend((setup([square], [40]), MECHANICS) for square in (0, 8, 63, 64, 72, 80))
        for index, (position, rules) in enumerate(positions):
            expected = oracle.legal(oracle.array_board(position), position.turn)
            for backend in self.backends:
                with self.subTest(backend=backend, position=index):
                    with closing(create_game(backend, rules, position)) as game:
                        self.assertEqual(game.legal_actions(), expected)
                        self.assertLessEqual(len(expected), 16 * position.pieces(position.turn).bit_count())
        for backend in self.backends:
            with self.subTest(backend=backend, case="sixteen destinations"):
                with closing(create_game(backend, MECHANICS, setup([40], [0]))) as game:
                    self.assertEqual(len(game.legal_actions()), 16)

    def test_e3_random_capture_exactness(self):
        """E3: Every legal move on 240 setups removes exactly the oracle capture set."""
        self.assertEqual({p.turn for p in self.positions}, {0, 1})
        self.assertEqual({p.pending for p in self.positions}, {-1, 0, 1})
        self.assertGreater(len({p.black.bit_count() for p in self.positions}), 1)
        for index, position in enumerate(self.positions):
            for backend in self.backends:
                with self.subTest(backend=backend, position=index):
                    with closing(create_game(backend, position=position)) as game:
                        for action in oracle.legal(oracle.array_board(position), position.turn):
                            with self.subTest(action=action):
                                self.assert_step(game, action)
                                self.assertTrue(game.undo())
                                self.assertEqual(game.position, position)

    def test_e3_capture_boundaries(self):
        """E3: Ordinary/edge captures share a snapshot and remove their union once."""
        cases = (
            ("horizontal", [38, 68], [39, 40], 68, 41, [39, 40]),
            ("vertical", [13, 44], [22, 31], 44, 40, [22, 31]),
            ("multiple rays", [13, 58, 42, 37], [22, 31, 49, 41], 37, 40, [22, 31, 49, 41]),
            ("corner", [1, 18], [0], 18, 9, [0]),
            ("edge group", [3, 9, 10, 20], [0, 1, 2], 20, 11, [0, 1, 2]),
            ("edge gap", [3, 9, 20], [0, 1, 2], 20, 11, []),
            ("bent group", [2, 9, 11, 28], [0, 1, 10], 28, 19, [0, 1, 10]),
            ("edge and ordinary overlap", [0, 3, 10, 20], [1, 2], 20, 11, [1, 2]),
            ("no interior sweep or cascade", [39, 31, 49, 32, 42, 59], [40, 41], 59, 50, [41]),
            ("word boundary", [63, 74], [64], 74, 65, [64]),
            ("diagonal", [20, 38], [30], 38, 40, []),
            ("ordinary gap", [37, 68], [39], 68, 41, []),
            ("edge alone", [18], [0], 18, 9, []),
        )
        for name, black, white, source, target, taken in cases:
            for swap in (False, True):
                position = setup(black, white)
                if swap:
                    position = swap_players(position)
                for backend in self.backends:
                    with self.subTest(backend=backend, case=name, swap=swap):
                        with closing(create_game(backend, MECHANICS, position)) as game:
                            result = self.assert_step(game, source * 81 + target)
                            self.assertEqual(result.captured, mask(taken))
        for backend in self.backends:
            with self.subTest(backend=backend, case="edge disabled"):
                rules = Rules(edge_capture=False, capture_target=0, lead_target=0, repetition_limit=0)
                with closing(create_game(backend, rules, setup([3, 9, 10, 20], [0, 1, 2]))) as game:
                    self.assertEqual(self.assert_step(game, 20 * 81 + 11).captured, 0)

    def test_e4_only_moved_piece_captures(self):
        """E4: Old sandwiches/enclosures remain and self-insertion never captures self."""
        cases = (
            ("sandwich", setup([39, 41, 80], [40]), 80 * 81 + 71),
            ("enclosure", setup([1, 9, 80], [0]), 80 * 81 + 71),
            ("self insertion", setup([67], [39, 41]), 67 * 81 + 40),
        )
        for name, original, action in cases:
            for swap in (False, True):
                position = swap_players(original) if swap else original
                for backend in self.backends:
                    with self.subTest(backend=backend, case=name, swap=swap):
                        with closing(create_game(backend, MECHANICS, position)) as game:
                            self.assertEqual(self.assert_step(game, action).captured, 0)
                            self.assertEqual(game.position.pieces(1 - position.turn), position.pieces(1 - position.turn))
                            self.assertTrue(game.position.pieces(position.turn) & (1 << (action % 81)))

    def test_e5_turn_alternation(self):
        """E5: Quiet, capturing and terminal plays all switch the player to move."""
        for index, position in enumerate(self.positions):
            for backend in self.backends:
                with self.subTest(backend=backend, position=index):
                    rng = random.Random(SEED + index)
                    with closing(create_game(backend, position=position)) as game:
                        for _ in range(8):
                            if game.outcome.terminal:
                                break
                            self.assert_step(game, rng.choice(game.legal_actions()))
        for backend in self.backends:
            with self.subTest(backend=backend, case="terminal turn"):
                with closing(create_game(backend)) as game:
                    for index in range(12):
                        self.assert_step(game, CYCLE[index % 4])
                    self.assert_terminal(game, -1, "repetition")
                    self.assertEqual(game.position.turn, 0)

    def test_e6_illegal_moves_are_atomic(self):
        """E6: Invalid types, IDs, paths, ownership and terminal plays reject atomically."""
        position = setup([40, 42], [58])
        cases = {
            "friendly destination": 40 * 81 + 42,
            "enemy destination": 40 * 81 + 58,
            "friendly jump": 40 * 81 + 44,
            "enemy jump": 40 * 81 + 67,
            "diagonal": 40 * 81 + 50,
            "wrong colour": 58 * 81 + 59,
            "empty source": 0 * 81 + 9,
            "same square": 40 * 81 + 40,
            "row wrap": 42 * 81 + 45,
            "negative": -1,
            "upper bound": 6561,
            "large ID": 1 << 50,
            "boolean": True,
            "float": 1.5,
        }
        for backend in self.backends:
            for swap in (False, True):
                start = swap_players(position) if swap else position
                with closing(create_game(backend, MECHANICS, start)) as game:
                    before = game_snapshot(game)
                    for name, action in cases.items():
                        with self.subTest(backend=backend, case=name, swap=swap):
                            with self.assertRaises(ValueError):
                                game.play(action)
                            self.assertEqual(game_snapshot(game), before)
                    self.assertFalse(game.undo())
                    self.assertEqual(game_snapshot(game), before)
            with self.subTest(backend=backend, case="rejection after capture"):
                start = setup([38, 68], [39, 40, 0])
                with closing(create_game(backend, MECHANICS, start)) as game:
                    initial = game_snapshot(game)
                    self.assertEqual(game.play(68 * 81 + 41).captured, mask([39, 40]))
                    before = game_snapshot(game)
                    with self.assertRaises(ValueError):
                        game.play(38 * 81 + 37)
                    self.assertEqual(game_snapshot(game), before)
                    self.assertTrue(game.undo())
                    self.assertEqual(game_snapshot(game), initial)
            with self.subTest(backend=backend, case="terminal with history"):
                with closing(create_game(backend)) as game:
                    for index in range(12):
                        game.play(CYCLE[index % 4])
                    before = game_snapshot(game)
                    with self.assertRaises(ValueError):
                        game.play(CYCLE[0])
                    self.assertEqual(game_snapshot(game), before)

    def test_e7_undo_exactness(self):
        """E7: Undo and replay restore all public state, including pending and captures."""
        for index, position in enumerate(self.positions):
            for backend in self.backends:
                with self.subTest(backend=backend, position=index):
                    rng = random.Random(SEED + index)
                    with closing(create_game(backend, position=position)) as game:
                        snapshots, actions = [game_snapshot(game)], []
                        for _ in range(6):
                            if game.outcome.terminal:
                                break
                            action = rng.choice(game.legal_actions())
                            result = self.assert_step(game, action)
                            after = game_snapshot(game)
                            self.assertTrue(game.undo())
                            self.assertEqual(game_snapshot(game), snapshots[-1])
                            self.assertEqual(game.play(action), result)
                            self.assertEqual(game_snapshot(game), after)
                            snapshots.append(after)
                            actions.append(action)
                        for before in reversed(snapshots[:-1]):
                            self.assertTrue(game.undo())
                            self.assertEqual(game_snapshot(game), before)
                        self.assertFalse(game.undo())
                        for action in actions:
                            game.play(action)
                        self.assertEqual(game_snapshot(game), snapshots[-1])

    def test_e7_repetition_counts_after_undo_and_clone(self):
        """E7: Undo removes repetition counts and full-history clones retain them."""
        for backend in self.backends:
            with self.subTest(backend=backend):
                with closing(create_game(backend)) as game:
                    for index in range(8):
                        game.play(CYCLE[index % 4])
                    saved = game_snapshot(game)
                    with closing(game.clone()) as clone:
                        self.assertEqual(game_snapshot(clone), saved)
                        for index in range(4):
                            clone.play(CYCLE[index])
                            self.assertEqual(clone.outcome.terminal, index == 3)
                        self.assert_terminal(clone, -1, "repetition")
                        self.assertEqual(game_snapshot(game), saved)
                    for _ in range(4):
                        self.assertTrue(game.undo())
                    # Replaying discarded visits must not produce an early draw.
                    for index in range(8):
                        before = game_snapshot(game)
                        result = game.play(CYCLE[index % 4])
                        self.assertEqual(game.outcome.terminal, index == 7)
                        self.assertTrue(game.undo())
                        self.assertEqual(game_snapshot(game), before)
                        self.assertEqual(game.play(CYCLE[index % 4]), result)
                    self.assert_terminal(game, -1, "repetition")

    def test_e8_fifth_capture_is_immediate(self):
        """E8: Reaching or exceeding five captures wins immediately for either player."""
        cases = (
            (setup([38, 67, 72, 73, 74, 75], [39, 0, 1, 2, 3]), 67 * 81 + 40, 5),
            (setup([38, 68, 72, 73, 74, 75], [39, 40, 0, 1, 2]), 68 * 81 + 41, 6),
        )
        for position, action, count in cases:
            for swap in (False, True):
                start = swap_players(position) if swap else position
                for backend in self.backends:
                    with self.subTest(backend=backend, captures=count, swap=swap):
                        with closing(create_game(backend, position=start)) as game:
                            self.assert_step(game, action)
                            self.assertEqual(game.position.captures(start.turn), count)
                            self.assert_terminal(game, start.turn, "five_captures")

    def test_e8_lead_reply_adjudication(self):
        """E8: A lead grants exactly one reply; only a reply below three rescues it."""
        three = setup([38, 67, 72, 73, 74, 75, 76, 77, 78], [39, 37, 3, 0, 1, 2, 4])
        four = setup([38, 68, 72, 73, 74, 75, 76, 77, 78], [39, 40, 37, 3, 0, 1, 2])
        cases = (
            ("unanswered three", three, 67 * 81 + 40, 0 * 81 + 9, 3, 3),
            ("rescued three", three, 67 * 81 + 40, 3 * 81 + 39, 3, 2),
            ("four to three", four, 68 * 81 + 41, 3 * 81 + 39, 4, 3),
        )
        for name, position, first, reply, initial_lead, final_lead in cases:
            for swap in (False, True):
                start = swap_players(position) if swap else position
                for backend in self.backends:
                    with self.subTest(backend=backend, case=name, swap=swap):
                        with closing(create_game(backend, position=start)) as game:
                            self.assert_step(game, first)
                            self.assertEqual(game.outcome, Outcome())
                            self.assertEqual(game.position.pending, start.turn)
                            self.assertEqual(game.position.captures(start.turn) - game.position.captures(1 - start.turn), initial_lead)
                            self.assertIn(reply, game.legal_actions())
                            pending = game_snapshot(game)
                            self.assert_step(game, reply)
                            self.assertEqual(game.position.captures(start.turn) - game.position.captures(1 - start.turn), final_lead)
                            if final_lead >= 3:
                                self.assert_terminal(game, start.turn, "lead_confirmed")
                            else:
                                self.assertEqual(game.outcome, Outcome())
                                self.assertEqual(game.position.pending, -1)
                                self.assertTrue(game.legal_actions())
                            self.assertTrue(game.undo())
                            self.assertEqual(game_snapshot(game), pending)

    def test_e8_fourfold_repetition(self):
        """E8: The initial key counts toward four occurrences; disabling draws works."""
        for backend in self.backends:
            for limit in (4, 0):
                with self.subTest(backend=backend, repetition_limit=limit):
                    with closing(create_game(backend, Rules(repetition_limit=limit))) as game:
                        self.assertEqual(game.history_keys(), [Position().key()])
                        for index in range(12 if limit else 20):
                            self.assert_step(game, CYCLE[index % 4])
                            self.assertEqual(game.outcome.terminal, bool(limit and index == 11))
                        if limit:
                            self.assert_terminal(game, -1, "repetition")
                            self.assertEqual(game.history_keys().count(game.position.key()), 4)
                        else:
                            self.assertEqual(game.outcome, Outcome())

    def test_e8_no_moves_policy(self):
        """E8: No-move setups/plays lose or draw; pending leads take precedence."""
        for backend in self.backends:
            for draw in (False, True):
                for swap in (False, True):
                    rules = Rules(edge_capture=False, capture_target=0, lead_target=0, no_moves_draw=draw)
                    blocked = setup([0], [1, 9])
                    closing_move = setup([1, 18], [0])
                    if swap:
                        blocked, closing_move = swap_players(blocked), swap_players(closing_move)
                    with self.subTest(backend=backend, draw=draw, swap=swap):
                        with closing(create_game(backend, rules, blocked)) as game:
                            self.assert_terminal(game, -1 if draw else 1 - blocked.turn, "no_moves")
                        with closing(create_game(backend, rules, closing_move)) as game:
                            self.assert_step(game, 18 * 81 + 9)
                            self.assert_terminal(game, -1 if draw else closing_move.turn, "no_moves")
                        pending = setup([0], [1, 9, 40, 50], pending=1)
                        if swap:
                            pending = swap_players(pending)
                        with closing(create_game(backend, Rules(capture_target=0, no_moves_draw=draw), pending)) as game:
                            self.assert_terminal(game, pending.pending, "lead_confirmed")

    @unittest.skipUnless(native_available(), "Native library is not built")
    def test_e9_backend_parity(self):
        """E9: Sixty seeded trajectories with a 240-ply cap agree at every transition."""
        for index in range(GAME_COUNT):
            rng = random.Random(SEED + index)
            with self.subTest(game=index), ExitStack() as stack:
                games = {backend: stack.enter_context(closing(create_game(backend))) for backend in self.backends}
                self.assertEqual(game_snapshot(games["python"]), game_snapshot(games["cpp"]))
                for ply in range(PLY_CAP):
                    reference = games["python"]
                    if reference.outcome.terminal:
                        break
                    expected_legal = oracle.legal(oracle.array_board(reference.position), reference.position.turn)
                    action = rng.choice(expected_legal)
                    results = []
                    for backend, game in games.items():
                        with self.subTest(backend=backend, ply=ply, action=action):
                            self.assertEqual(game.legal_actions(), expected_legal)
                            results.append(self.assert_step(game, action))
                    self.assertEqual(results[0], results[1])
                    self.assertEqual(game_snapshot(games["python"]), game_snapshot(games["cpp"]))

    def test_e10_random_record_roundtrips(self):
        """E10: Sixty random games replay across both backends with full history."""
        for index in range(GAME_COUNT):
            for source_backend in self.backends:
                with self.subTest(game=index, source=source_backend):
                    rng = random.Random(SEED + index)
                    with closing(create_game(source_backend)) as game:
                        for _ in range(PLY_CAP):
                            if game.outcome.terminal:
                                break
                            game.play(rng.choice(game.legal_actions()))
                        before = game_snapshot(game)
                        data = json.loads(json.dumps(record_dict(game)))
                        for target_backend in self.backends:
                            with self.subTest(target=target_backend):
                                with closing(from_record(data, target_backend)) as restored:
                                    self.assertEqual(game_snapshot(restored), before)
                                    if game.moves:
                                        last = game.moves[-1]
                                        self.assertTrue(restored.undo())
                                        restored.play(last)
                                        self.assertEqual(game_snapshot(restored), before)
                        self.assertEqual(game_snapshot(game), before)

    def test_e10_record_restores_repetition_adjudication(self):
        """E10: Replayed history triggers the fourth occurrence at the same ply."""
        for source_backend in self.backends:
            with closing(create_game(source_backend)) as game:
                for index in range(8):
                    game.play(CYCLE[index % 4])
                before = game_snapshot(game)
                for target_backend in self.backends:
                    with self.subTest(source=source_backend, target=target_backend):
                        with closing(from_record(record_dict(game), target_backend)) as restored:
                            self.assertEqual(game_snapshot(restored), before)
                            for index, action in enumerate(CYCLE):
                                self.assert_step(restored, action)
                                self.assertEqual(restored.outcome.terminal, index == 3)
                            self.assert_terminal(restored, -1, "repetition")
                            self.assertTrue(restored.undo())
                            self.assert_step(restored, CYCLE[-1])
                            self.assert_terminal(restored, -1, "repetition")
                self.assertEqual(game_snapshot(game), before)

    def test_e10_tampered_records_are_rejected(self):
        """E10: Illegal moves, altered history/results and incompatible records reject."""
        with closing(create_game("python")) as game:
            for index in range(12):
                game.play(CYCLE[index % 4])
            valid = record_dict(game)
        variants = {}
        for field, value in (("format", "bad"), ("ruleset_id", "bad")):
            variants[field] = deepcopy(valid)
            variants[field][field] = value
        for field, value in (("winner", 0), ("reason", "ongoing")):
            variants[field] = deepcopy(valid)
            variants[field]["final"][field] = value
        variants["illegal action"] = deepcopy(valid)
        variants["illegal action"]["moves"][0] = 72 * 81 + 0
        variants["missing cycle"] = deepcopy(valid)
        variants["missing cycle"]["moves"] = valid["moves"][4:]
        variants["post terminal"] = deepcopy(valid)
        variants["post terminal"]["moves"].append(CYCLE[0])
        variants["final board"] = deepcopy(valid)
        variants["final board"]["final"]["position"]["black"] = "0x0"
        variants["pending"] = deepcopy(valid)
        variants["pending"]["final"]["position"]["pending"] = 0
        for backend in self.backends:
            for name, data in variants.items():
                with self.subTest(backend=backend, tampering=name):
                    with self.assertRaises(ValueError):
                        with closing(from_record(data, backend)):
                            pass

    def test_e11_observation_matches_state(self):
        """E11: All mask entries and five planes reflect current and terminal state."""
        for index, position in enumerate(self.positions):
            for backend in self.backends:
                with self.subTest(backend=backend, position=index):
                    with closing(create_game(backend, position=position)) as game:
                        self.assert_observation(game)
                        self.assert_step(game, game.legal_actions()[0])
                        self.assert_observation(game)
        for backend in self.backends:
            with self.subTest(backend=backend, case="terminal observation"):
                with closing(create_game(backend)) as game:
                    for index in range(12):
                        game.play(CYCLE[index % 4])
                    self.assert_observation(game)

    def assert_observation(self, game):
        before = game_snapshot(game)
        observation = observe(game)
        position = game.position
        board = oracle.array_board(position)
        self.assertEqual(len(observation["action_mask"]), 6561)
        self.assertLessEqual(set(observation["action_mask"]), {0, 1})
        self.assertEqual([i for i, value in enumerate(observation["action_mask"]) if value], game.legal_actions())
        self.assertEqual(observation["legal_actions"], game.legal_actions())
        expected = (
            [int(piece == position.turn) for piece in board],
            [int(piece == 1 - position.turn) for piece in board],
            [int(position.pending == position.turn)] * 81,
            [int(position.pending == 1 - position.turn)] * 81,
            [position.turn] * 81,
        )
        self.assertEqual(len(observation["planes"]), 5)
        for plane, flat in zip(observation["planes"], expected):
            self.assertEqual(len(plane), 9)
            self.assertTrue(all(len(row) == 9 for row in plane))
            self.assertEqual([value for row in plane for value in row], flat)
        self.assertEqual(observation["state"], position.to_dict())
        self.assertEqual(observation["captures"], list(before["captures"]))
        self.assertEqual(observation["history"], [Position(*key).to_dict() for key in before["history"]])
        self.assertEqual(observation["history_counts"], [dict(position=Position(*key).to_dict(), count=count) for key, count in sorted(before["counts"].items())])
        self.assertEqual(observation["current_repetitions"], before["counts"][position.key()])
        self.assertEqual(game_snapshot(game), before)

    def test_e12_perft_anchors(self):
        """E12: Initial legal count is 63, perft(2) is 3717, and depth three agrees."""
        counts = {}
        for backend in self.backends:
            with self.subTest(backend=backend):
                with closing(create_game(backend)) as game:
                    before = game_snapshot(game)
                    self.assertEqual(len(game.legal_actions()), 63)
                    self.assertEqual(perft(game, 2), 3717)
                    counts[backend] = perft(game, 3)
                    self.assertGreater(counts[backend], 3717)
                    if backend == "cpp":
                        self.assertEqual(game.perft(2), 3717)
                        self.assertEqual(game.perft(3), counts[backend])
                    self.assertEqual(game_snapshot(game), before)
        if "cpp" in counts:
            self.assertEqual(counts["python"], counts["cpp"])


if __name__ == "__main__":
    unittest.main()
