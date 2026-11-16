"""Standard shogi through the learning stack: GameSpec adapter, encoder, records, search and a tiny AlphaZero run."""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from tests.learn._common import requires_torch

ROOT = Path(__file__).resolve().parents[2]


@requires_torch
class ShogiAdapterTests(unittest.TestCase):
    def setUp(self):
        from hasami_learn.games import get_spec

        self.spec = get_spec("shogi")

    def random_game(self, plies=60, seed=3, sfen=None):
        game = self.spec.create_game("python", position=sfen)
        self.addCleanup(game.close)
        rng = random.Random(seed)
        labels = []
        for _ in range(plies):
            actors = self.spec.actors(game)
            if not actors:
                break
            action = rng.choice(self.spec.legal_actions(game, actors[0]))
            labels.append({actors[0]: action})
            self.spec.step(game, {actors[0]: action})
        return game, labels

    def test_contract_attributes_and_encoder_match_the_engine(self):
        import numpy as np
        from shogi.encoding import ACTION_SIZE, NUM_PLANES, encode, legal_labels

        self.assertEqual((self.spec.num_players, self.spec.action_size, self.spec.board_shape), (2, ACTION_SIZE, (9, 9)))
        self.assertTrue(self.spec.zero_sum and self.spec.supports_undo and not self.spec.has_chance)
        game, _ = self.random_game(plies=37)
        player = game.position.turn
        planes = self.spec.observe(game, player)
        self.assertEqual(planes.shape, (NUM_PLANES, 9, 9))
        self.assertEqual(planes.dtype, np.float32)
        np.testing.assert_array_equal(planes.reshape(-1), np.asarray(encode(game, player), dtype=np.float32))
        mask = self.spec.legal_mask(game, player)
        self.assertEqual(sorted(np.flatnonzero(mask)), sorted(legal_labels(game)))
        self.assertEqual(self.spec.legal_mask(game, 1 - player).sum(), 0)
        with self.assertRaises(ValueError):
            self.spec.observe(game, 1 - player)

    def assert_same_state(self, game, before):
        self.assertEqual((game.position, game.outcome, list(game.moves), game.history_keys(),
                          self.spec.history_actions(game)), before)

    def state(self, game):
        return (game.position, game.outcome, list(game.moves), game.history_keys(), self.spec.history_actions(game))

    def test_step_and_undo_restore_the_exact_state(self):
        game, labels = self.random_game(plies=40, seed=11)
        self.assertFalse(game.outcome.terminal)
        self.assertEqual(self.spec.history_actions(game), labels)
        before = self.state(game)
        actions = self.spec.legal_actions(game, game.position.turn)
        self.assertGreaterEqual(len(actions), 5)
        undone = 0
        for action in actions:
            self.spec.step(game, {game.position.turn: action})
            self.assertTrue(game.undo())
            undone += 1
            self.assert_same_state(game, before)
        self.assertEqual(undone, len(actions))  # every legal action was stepped and undone
        with self.assertRaises(ValueError):
            self.spec.step(game, {0: 0, 1: 0})

    def test_terminal_transition_values_and_undo(self):
        from shogi.encoding import DECLARE_INDEX

        # CSA 27-point declaration (tests/shogi_suite/vectors/rule_cases.json): sente may declare and wins at once.
        game = self.spec.create_game(position="GGGG1SSSS/RB7/4K4/9/9/9/9/9/4k4 b RB 1")
        self.addCleanup(game.close)
        self.assertIn(DECLARE_INDEX, self.spec.legal_actions(game, 0))
        before = self.state(game)
        outcome = self.spec.step(game, {0: DECLARE_INDEX})
        self.assertTrue(outcome.terminal)
        self.assertEqual(outcome.info["reason"], "declaration")
        self.assertEqual(self.spec.terminal_values(game), [1.0, -1.0])
        self.assertEqual(self.spec.actors(game), [])
        self.assertTrue(game.undo())
        self.assert_same_state(game, before)

    def test_label_hooks_give_engine_codes_and_usi(self):
        from shogi.notation import move_to_usi

        game = self.spec.create_game()
        self.addCleanup(game.close)
        for label in self.spec.legal_actions(game, 0):
            code = self.spec.action_to_move(game, label)
            self.assertIn(code, game.legal_actions())
            self.assertEqual(self.spec.move_text(game, label), move_to_usi(code))
        self.assertIn("7g7f", [self.spec.move_text(game, a) for a in self.spec.legal_actions(game, 0)])

    def test_records_use_the_shogi_json_format_and_keep_resignations(self):
        from hasami_learn.games.records import load_record, save_record
        from shogi.records import from_json

        sfen = "1nrk3nb/3s2g1p/lppg1spp1/4pP3/1P7/3PP2Pl/2P3P1P/L1R1G2s1/BN1GK1SNL w 3Pp 56"
        for start, resign in ((None, False), (sfen, False), (None, True)):
            game, labels = self.random_game(plies=30, seed=5, sfen=start)
            if resign:
                game.resign()
            with tempfile.TemporaryDirectory() as tmp:
                path = save_record(self.spec, game, Path(tmp) / "game.json")
                interop = from_json(Path(path).read_text(encoding="utf-8"))  # the shogi record format itself
                self.assertEqual((interop.sfen(), interop.outcome), (game.sfen(), game.outcome))
                loaded = load_record(self.spec, path)
                try:
                    self.assertEqual((loaded.sfen(), loaded.outcome), (game.sfen(), game.outcome))
                    self.assertEqual(self.spec.history_actions(loaded), labels)
                finally:
                    loaded.close()
            with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
                save_record(self.spec, game, Path(tmp) / "game.kif")

    def test_baselines_play_through_the_learning_runner_in_label_space(self):
        from hasami.agents import SearchLimits
        from hasami_learn.selfplay import run_table_game

        encoder = self.spec.make_encoder()
        for name in ("random", "greedy", "alphabeta:1"):
            agent, limits = self.spec.baseline_agent(name, seed=1)
            opponent, _ = self.spec.baseline_agent("random", seed=2)
            trajectory = run_table_game({0: agent, 1: opponent}, encoder, spec=self.spec, max_plies=12,
                                        limits=limits or SearchLimits(1, 1000, 20000))
            self.assertEqual(trajectory.plies, 12, name)
            for step in trajectory.joint_actions:
                for action in step.values():
                    self.assertTrue(0 <= action < self.spec.action_size)

    def test_label_agent_closes_its_clone_on_success_and_cancellation(self):
        import threading
        from hasami.agents import AgentCancelled, SearchLimits

        game = self.spec.create_game()
        self.addCleanup(game.close)
        created, closed = [], []
        original = type(game).clone

        def tracking_clone(source):
            copy = original(source)
            created.append(copy)
            close = copy.close
            copy.close = lambda: (closed.append(copy), close())[1]
            return copy

        type(game).clone = tracking_clone
        self.addCleanup(setattr, type(game), "clone", original)

        def every_clone_closed():
            return created and all(any(c is copy for c in closed) for copy in created)

        for name in ("random", "greedy", "alphabeta:1"):
            agent, limits = self.spec.baseline_agent(name, seed=1)
            created.clear(); closed.clear()
            agent.choose(game, limits or SearchLimits(1, 1000, 20000))
            self.assertTrue(every_clone_closed(), name)
        cancel = threading.Event()
        cancel.set()
        agent, limits = self.spec.baseline_agent("alphabeta:2")
        created.clear(); closed.clear()
        with self.assertRaises(AgentCancelled):
            agent.choose(game, limits, cancel)
        self.assertTrue(every_clone_closed())

    def test_default_cap_and_rules_params(self):
        from hasami_learn.games import get_spec

        self.assertEqual(self.spec.default_max_plies, 320)
        self.assertEqual(get_spec("shogi", {"max_moves": 0}).rules.max_moves, 0)
        with self.assertRaises(ValueError):
            get_spec("shogi", {"board": 5})

    def test_mcts_returns_legal_labels(self):
        import numpy as np
        from hasami_learn.search.mcts import MCTS, MctsConfig

        def uniform(planes, masks):
            return masks / masks.sum(axis=1, keepdims=True), np.zeros(len(planes))

        game = self.spec.create_game()
        self.addCleanup(game.close)
        search = MCTS(uniform, self.spec.make_encoder(), MctsConfig(add_noise=False), np.random.default_rng(0),
                      spec=self.spec)
        result = search.run(game, simulations=12)
        visited = set(int(a) for a in np.flatnonzero(result.visits))
        self.assertTrue(visited)
        self.assertTrue(visited <= set(self.spec.legal_actions(game, 0)))

    def test_tiny_alphazero_trains_and_its_agent_plays_legally(self):
        from hasami_learn.config import TrainingConfig
        from hasami_learn.inference import create_agent
        from hasami_learn.trainer import train_from_config

        config = TrainingConfig.load(ROOT / "configs" / "tiny" / "alphazero-shogi.json")
        with tempfile.TemporaryDirectory() as tmp:
            config.run.output_dir = tmp
            config.datastore.path = str(Path(tmp) / "selfplay")
            train_from_config(config, echo=False)
            checkpoint = Path(tmp) / config.run.run_id / "checkpoints"
            agent = create_agent(str(checkpoint), device="cpu", simulations=4)
            game = self.spec.create_game()
            try:
                from hasami.agents import SearchLimits

                decision = agent.choose(game.clone(), SearchLimits(1, 5000, 4))
                self.assertIn(decision.action, self.spec.legal_actions(game, 0))
            finally:
                game.close()


if __name__ == "__main__":
    unittest.main()
