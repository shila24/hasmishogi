"""PUCT legality, history, batching, and resource-lifetime regressions."""

import os
import threading
import time
import unittest
from unittest.mock import Mock, patch

from hasami import create_game
from hasami.records import record_dict
from tests.learn._common import LearnTestCase, requires_torch
from tests.test_rules import setup


@requires_torch
class MctsTests(LearnTestCase):
    def setUp(self):
        super().setUp()
        import numpy as np
        from hasami_learn.encoding import PlaneEncoder

        self.np = np
        self.encoder = PlaneEncoder()
        self.batch_sizes = []

    def uniform(self, planes, masks):
        self.batch_sizes.append(len(planes))
        return masks / masks.sum(axis=1, keepdims=True), self.np.zeros(len(planes))

    def search(self, evaluator=None, **params):
        from hasami_learn.algorithms.mcts import MCTS, MctsConfig

        return MCTS(evaluator or self.uniform, self.encoder,
                    MctsConfig(add_noise=False, **params), self.np.random.default_rng(7))

    def game(self, position=None, backend="python"):
        game = create_game(backend, position=position)
        self.addCleanup(game.close)
        return game

    def assert_result(self, game, result, simulations):
        self.assertEqual(result.visits.shape, (6561,))
        self.assertEqual(result.visits.dtype, self.np.float32)
        self.assertEqual(result.simulations, simulations)
        self.assertEqual(float(result.visits.sum()), simulations)
        self.assertTrue(set(self.np.flatnonzero(result.visits)).issubset(game.legal_actions()))
        self.assertIn(int(result.visits.argmax()), game.legal_actions())

    def test_root_q_values_are_mean_backed_up_values_of_visited_edges(self):
        game = self.game()

        def constant(planes, masks):
            return masks / masks.sum(axis=1, keepdims=True), self.np.full(len(planes), 0.5)

        result = self.search(constant).run(game, simulations=10)
        legal = set(game.legal_actions())
        visited = set(int(a) for a in self.np.flatnonzero(result.visits))
        self.assertEqual(result.q_values.shape, (6561,))
        self.assertTrue(visited and visited < legal)  # some legal edges stay unvisited
        for action in range(6561):
            if action in visited:
                # every leaf is worth 0.5 to its own mover, so -0.5 to the root mover
                self.assertAlmostEqual(float(result.q_values[action]), -0.5, places=6)
            else:
                self.assertTrue(self.np.isnan(result.q_values[action]))

    def test_capture_preferred_and_root_unchanged(self):
        game = self.game(setup([38, 67, 72, 73, 74, 75, 76, 77, 78],
                               [39, 37, 3, 0, 1, 2, 4]))
        before = record_dict(game)
        result = self.search().run(game, simulations=200)
        self.assert_result(game, result, 200)
        self.assertEqual(record_dict(game), before)
        work = game.clone()
        try:
            self.assertNotEqual(work.play(int(result.visits.argmax())).captured, 0)
        finally:
            work.close()

    def test_run_many_batches_three_games(self):
        games = [self.game() for _ in range(3)]
        self.assertEqual(len(games[0].legal_actions()), 63)
        work = games[0].clone()
        try:
            count = 0
            for action in work.legal_actions():
                work.play(action)
                count += len(work.legal_actions())
                work.undo()
            self.assertEqual(count, 3717)
        finally:
            work.close()
        before = [record_dict(game) for game in games]
        results = self.search().run_many(games, simulations=12)
        self.assertEqual(len(results), 3)
        for game, result in zip(games, results):
            self.assert_result(game, result, 12)
        self.assertEqual(self.batch_sizes, [3] * 13)
        self.assertEqual([record_dict(game) for game in games], before)

    def test_cancel_deadline_and_zero_budget_return_prior(self):
        game = self.game()
        cancel = threading.Event()
        cancel.set()
        for kwargs in ({"cancel": cancel}, {"deadline": time.perf_counter() - 1},
                       {"simulations": 0}):
            with self.subTest(kwargs=kwargs):
                result = self.search().run(game, **kwargs)
                self.assertEqual(result.simulations, 0)
                self.assertAlmostEqual(float(result.visits.sum()), 1.0, places=6)
                self.assertIn(int(result.visits.argmax()), game.legal_actions())

    def test_cancel_during_search_keeps_completed_visits(self):
        game = self.game()
        cancel = threading.Event()

        def evaluate(planes, masks):
            output = self.uniform(planes, masks)
            if len(self.batch_sizes) == 3:
                cancel.set()
            return output

        result = self.search(evaluate).run(game, simulations=20, cancel=cancel)
        self.assert_result(game, result, 2)

    def test_deadline_between_simulations_keeps_completed_visits(self):
        game = self.game()
        with patch("hasami_learn.algorithms.mcts.time.perf_counter", side_effect=[0, 0, 2]):
            result = self.search().run(game, simulations=20, deadline=1)
        self.assert_result(game, result, 2)

    def test_root_noise_is_legal_seeded_and_optional(self):
        from hasami_learn.algorithms.mcts import MCTS, MctsConfig

        game = self.game()
        clean = self.search().run(game, simulations=0)
        noisy = []
        for _ in range(2):
            search = MCTS(self.uniform, self.encoder, MctsConfig(), self.np.random.default_rng(3))
            noisy.append(search.run(game, simulations=0).visits)
        self.np.testing.assert_array_equal(noisy[0], noisy[1])
        self.assertFalse(self.np.array_equal(clean.visits, noisy[0]))
        self.np.testing.assert_array_equal(noisy[0][self.encoder.legal_mask(game) == 0], 0)
        self.assertAlmostEqual(float(noisy[0].sum()), 1.0, places=6)

    def test_batch_size_caps_evaluator_calls(self):
        games = [self.game() for _ in range(5)]
        results = self.search(batch_size=2).run_many(games, simulations=3)
        self.assertEqual(self.batch_sizes, [2, 2, 1] * 4)
        for game, result in zip(games, results):
            self.assert_result(game, result, 3)

    def test_native_parity_and_clone_cleanup(self):
        try:
            native = self.game(backend="cpp")
        except (RuntimeError, OSError) as exc:
            if os.environ.get("HASAMI_REQUIRE_NATIVE") == "1":
                raise
            self.skipTest(str(exc))
        python = self.game()
        for game in (python, native):
            for action in [72 * 81 + 63, 0 * 81 + 9, 63 * 81 + 72, 9 * 81 + 0] * 2:
                game.play(action)
        before = record_dict(native)
        clone = native.clone()
        close = Mock(wraps=clone.close)
        clone.close = close
        with patch.object(native, "clone", return_value=clone):
            result = self.search().run(native, simulations=20)
        close.assert_called_once()
        with self.assertRaises(RuntimeError):
            clone.legal_actions()
        reference = self.search().run(python, simulations=20)
        self.np.testing.assert_array_equal(result.visits, reference.visits)
        self.assertEqual(result.root_value, reference.root_value)
        self.assertEqual(record_dict(native), before)

    def test_repetition_uses_callers_history_and_alternating_values(self):
        game = self.game()
        cycle = [72 * 81 + 63, 0 * 81 + 9, 63 * 81 + 72, 9 * 81 + 0]
        for action in cycle * 2:
            game.play(action)
        before = record_dict(game)
        calls = 0

        def evaluate(planes, masks):
            nonlocal calls
            policy = self.np.zeros_like(masks)
            policy[:, cycle[calls % 4]] = 1.0
            calls += 1
            return policy, self.np.full(len(planes), 0.75)

        result = self.search(evaluate, c_puct=100).run(game, simulations=8)
        self.assert_result(game, result, 8)
        self.assertEqual(calls, 4)
        self.assertAlmostEqual(result.root_value, -0.75 / 8)
        self.assertEqual(record_dict(game), before)

    def test_clones_closed_on_evaluator_error(self):
        game = self.game()
        before = record_dict(game)
        clone = game.clone()
        close = Mock(wraps=clone.close)
        clone.close = close
        evaluate = Mock(side_effect=[self.uniform(
            self.encoder.encode(game)[None], self.encoder.legal_mask(game)[None]),
            RuntimeError("evaluation failed")])
        with patch.object(game, "clone", return_value=clone):
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                self.search(evaluate).run(game)
        close.assert_called_once()
        self.assertEqual(record_dict(game), before)
        self.assertEqual(record_dict(clone), before)

    def test_terminal_root_skips_network(self):
        game = self.game(setup([38, 67, 72, 73, 74, 75, 76],
                               [39, 37, 3, 0, 1]))
        game.play(67 * 81 + 40)
        self.assertTrue(game.outcome.terminal)
        result = self.search().run(game, simulations=10)
        self.assertEqual(result.simulations, 0)
        self.assertEqual(result.root_value, -1.0)
        self.assertEqual(float(result.visits.sum()), 0.0)
        self.assertEqual(self.batch_sizes, [])

    def test_config_rejects_unknown_and_invalid_parameters(self):
        from hasami_learn.algorithms.mcts import MctsConfig

        with self.assertRaises(TypeError):
            MctsConfig(unknown=1)
        for params in ({"simulations": -1}, {"batch_size": 0}, {"c_puct": -1},
                       {"dirichlet_alpha": 0}, {"dirichlet_epsilon": 2}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                MctsConfig(**params)

    def test_batched_evaluator_masks_and_restores_training_mode(self):
        import torch
        from hasami_learn.algorithms.mcts import batched_evaluator

        class Network(torch.nn.Module):
            def forward(inner, planes):
                self.assertFalse(inner.training)
                self.assertFalse(torch.is_grad_enabled())
                return torch.zeros((len(planes), 6561)), torch.ones(len(planes)) * 0.25

        network = Network().train()
        game = self.game()
        planes = self.np.stack([self.encoder.encode(game)] * 2)
        masks = self.np.stack([self.encoder.legal_mask(game)] * 2)
        masks[1] = 0
        probs, values = batched_evaluator(network, "cpu")(planes, masks)
        self.assertTrue(network.training)
        self.assertEqual(probs.dtype, self.np.float32)
        self.np.testing.assert_array_equal(probs[masks == 0], 0)
        self.np.testing.assert_allclose(probs.sum(1), [1, 0], atol=2e-7)
        self.np.testing.assert_allclose(values, [0.25, 0.25])


@requires_torch
class SearchContractTests(LearnTestCase):
    """multiplayer search review findings: refuse non-zero-sum games and back up by player identity."""

    def setUp(self):
        super().setUp()
        import numpy as np
        self.np = np

    def uniform(self, planes, masks):
        probs = masks / self.np.maximum(masks.sum(axis=1, keepdims=True), 1)
        return probs, self.np.zeros(len(masks), dtype=self.np.float32)

    def test_non_zero_sum_terminal_values_are_rejected(self):
        from hasami_learn.algorithms.mcts import MCTS, MctsConfig
        from hasami_learn.games.base import require_search_game
        from hasami_learn.games.hasami import HasamiSpec

        class Skewed(HasamiSpec):
            def terminal_values(self, game):
                return [0.9, 0.8]

        spec = Skewed()
        encoder = spec.make_encoder("")
        search = MCTS(self.uniform, encoder, MctsConfig(simulations=10, add_noise=False), self.np.random.default_rng(0), spec=spec)
        game = create_game("python", None, setup([38, 67, 72, 73, 74, 75, 76], [39, 37, 3, 0, 1]))
        try:
            game.play(67 * 81 + 40)  # terminal root: the search reads terminal_values directly
            self.assertTrue(game.outcome.terminal)
            with self.assertRaises(ValueError):
                search.run(game, simulations=10)
        finally:
            game.close()

        class Declared(HasamiSpec):
            zero_sum = False

        with self.assertRaises(ValueError):
            MCTS(self.uniform, encoder, MctsConfig(), self.np.random.default_rng(0), spec=Declared())
        with self.assertRaises(ValueError):
            require_search_game(Declared())

    def test_extra_turn_win_is_credited_to_the_mover(self):
        """Player 0 may take an extra turn (action 1) and then win with a second action 1.

        Negating once per ply would credit that win to player 1 at the root; identity-based backup
        keeps it with player 0, so the search prefers the forced win over the drawing pass.
        """
        from types import SimpleNamespace
        from hasami_learn.algorithms.mcts import MCTS, MctsConfig
        from hasami_learn.games.base import StepOutcome
        np = self.np

        class ExtraTurnGame:
            def __init__(self):
                self.moves = []
                self.initial = {}

            @property
            def position(self):
                # Turn order: 0 -> (1: extra turn for 0 | 0: player 1) -> terminal. After a win the
                # engine reports the loser as the side to move, like the hasami engine does.
                if not self.moves:
                    return SimpleNamespace(turn=0)
                if self.moves[0] == 0:
                    return SimpleNamespace(turn=1)
                if len(self.moves) == 1:
                    return SimpleNamespace(turn=0)
                return SimpleNamespace(turn=1)

            @property
            def outcome(self):
                terminal = len(self.moves) == 2
                winner = 0 if terminal and self.moves == [1, 1] else -1
                return SimpleNamespace(terminal=terminal, winner=winner, reason="finished" if terminal else "ongoing")

            def legal_actions(self):
                return [] if self.outcome.terminal else [0, 1]

            def play(self, action):
                self.moves.append(int(action))

            def undo(self):
                self.moves.pop()

            def clone(self):
                other = ExtraTurnGame()
                other.moves = list(self.moves)
                return other

            def close(self):
                pass

        class ExtraTurnSpec:
            game_id = "extra-turn-test"
            ruleset_id = "extra-turn-v1"
            board_shape = (1, 1)
            action_size = 2
            default_max_plies = 4
            num_players = 2
            perfect_information = True
            zero_sum = True

            def create_game(self, backend="python", rules=None, position=None, seed=None):
                return ExtraTurnGame()

            def actors(self, game):
                return [] if game.outcome.terminal else [game.position.turn]

            def legal_actions(self, game, player):
                return game.legal_actions() if player in self.actors(game) else []

            def legal_mask(self, game, player):
                mask = np.zeros(2, dtype=np.float32)
                mask[self.legal_actions(game, player)] = 1
                return mask

            def step(self, game, actions):
                if set(actions) != set(self.actors(game)):
                    raise ValueError("wrong actors")
                game.play(int(next(iter(actions.values()))))
                return StepOutcome(game.outcome.terminal, False,
                                   self.terminal_values(game) if game.outcome.terminal else None, {})

            def terminal_values(self, game):
                winner = game.outcome.winner
                return [0.0, 0.0] if winner == -1 else [1.0 if p == winner else -1.0 for p in range(2)]

        class Encoder:
            def encode(self, game, player=None):
                return np.zeros((1, 1, 1), dtype=np.float32)

            def legal_mask(self, game, player=None):
                mask = np.zeros(2, dtype=np.float32)
                mask[game.legal_actions()] = 1
                return mask

        search = MCTS(self.uniform, Encoder(), MctsConfig(simulations=200, add_noise=False, c_puct=1.0),
                      np.random.default_rng(0), spec=ExtraTurnSpec())
        result = search.run(ExtraTurnGame(), simulations=200)
        self.assertEqual(int(result.visits.argmax()), 1, result.visits)
        self.assertGreater(result.root_value, 0.0)

    def test_search_uses_the_adapter_transition_not_raw_play(self):
        """An adapter may translate action ids in step(); the search must simulate what step() does."""
        from types import SimpleNamespace
        from hasami_learn.algorithms.mcts import MCTS, MctsConfig
        from hasami_learn.games.base import StepOutcome
        np = self.np

        class RawGame:
            """One raw move decides the game: raw 1 -> player 0 wins, raw 0 -> player 1 wins."""

            def __init__(self):
                self.moves = []
                self.initial = {}

            @property
            def position(self):
                return SimpleNamespace(turn=0 if not self.moves else 1 - self.outcome.winner)

            @property
            def outcome(self):
                terminal = bool(self.moves)
                winner = (0 if self.moves[0] == 1 else 1) if terminal else -2
                return SimpleNamespace(terminal=terminal, winner=winner, reason="finished" if terminal else "ongoing")

            def legal_actions(self):
                return [] if self.outcome.terminal else [0, 1]

            def play(self, raw):
                self.moves.append(int(raw))

            def clone(self):
                other = RawGame()
                other.moves = list(self.moves)
                return other

            def close(self):
                pass

        class TranslatingSpec:
            """Adapter action a plays raw move 1 - a (no supports_undo: the search must clone)."""
            game_id = "translating-test"
            ruleset_id = "translating-v1"
            board_shape = (1, 1)
            action_size = 2
            default_max_plies = 2
            num_players = 2
            perfect_information = True

            def create_game(self, backend="python", rules=None, position=None, seed=None):
                return RawGame()

            def actors(self, game):
                return [] if game.outcome.terminal else [game.position.turn]

            def legal_actions(self, game, player):
                return game.legal_actions() if player in self.actors(game) else []

            def legal_mask(self, game, player):
                mask = np.zeros(2, dtype=np.float32)
                mask[self.legal_actions(game, player)] = 1
                return mask

            def observe(self, game, player):
                return np.zeros((1, 1, 1), dtype=np.float32)

            def step(self, game, actions):
                if set(actions) != set(self.actors(game)):
                    raise ValueError("wrong actors")
                game.play(1 - int(next(iter(actions.values()))))
                return StepOutcome(game.outcome.terminal, False,
                                   self.terminal_values(game) if game.outcome.terminal else None, {})

            def terminal_values(self, game):
                return [1.0 if p == game.outcome.winner else -1.0 for p in range(2)]

        class Encoder:
            def encode(self, game, player=None):
                return np.zeros((1, 1, 1), dtype=np.float32)

        spec = TranslatingSpec()
        search = MCTS(self.uniform, Encoder(), MctsConfig(simulations=60, add_noise=False), np.random.default_rng(0), spec=spec)
        game = RawGame()
        result = search.run(game, simulations=60)
        self.assertEqual(game.moves, [])  # the caller's state is untouched
        self.assertEqual(int(result.visits.argmax()), 0, result.visits)  # adapter action 0 -> raw 1 -> player 0 wins
        self.assertGreater(result.root_value, 0.0)
        probe = RawGame()
        self.assertEqual(spec.step(probe, {0: int(result.visits.argmax())}).rewards, [1.0, -1.0])

    def test_clone_and_undo_descents_agree_on_hasami(self):
        """Hasami declares supports_undo; forcing the clone path must produce identical search results."""
        from hasami_learn.algorithms.mcts import MCTS, MctsConfig
        from hasami_learn.games.hasami import HasamiSpec
        np = self.np

        class NoUndo(HasamiSpec):
            supports_undo = False

        position = setup([38, 67, 72, 73, 74, 75, 76, 77, 78], [39, 37, 3, 0, 1, 2, 4])
        results = []
        for spec in (HasamiSpec(), NoUndo()):
            self.assertEqual(MCTS(self.uniform, spec.make_encoder(""), MctsConfig(), np.random.default_rng(0), spec=spec).undoable,
                             spec.supports_undo)
            game = create_game("python", None, position)
            try:
                search = MCTS(self.uniform, spec.make_encoder(""), MctsConfig(simulations=40, add_noise=False),
                              np.random.default_rng(3), spec=spec)
                results.append(search.run(game, simulations=40))
                self.assertEqual(record_dict(game)["moves"], [])
            finally:
                game.close()
        np.testing.assert_array_equal(results[0].visits, results[1].visits)
        self.assertEqual(results[0].root_value, results[1].root_value)

    def test_adapter_truncation_stops_the_descent(self):
        """An adapter may stop a game (StepOutcome.truncated) while the engine outcome stays ongoing; the
        search values that state like a capped self-play game and never steps beyond it."""
        from hasami_learn.algorithms.mcts import MCTS, MctsConfig
        from hasami_learn.games.base import StepOutcome
        from hasami_learn.games.hasami import HasamiSpec
        np = self.np

        class Capped(HasamiSpec):
            def step(self, game, actions):
                if len(game.moves) >= 1:
                    raise RuntimeError("step called after adapter truncation")
                outcome = super().step(game, actions)
                return StepOutcome(outcome.terminal, True, outcome.rewards, {"reason": "resource_limit"})

        class CappedClone(Capped):
            supports_undo = False

        with self.assertRaises(ValueError):
            MctsConfig(truncation_value=2.0)
        for spec in (Capped(), CappedClone()):
            for truncation_value in (0.0, -0.4):
                game = create_game("python")
                try:
                    search = MCTS(self.uniform, spec.make_encoder(""),
                                  MctsConfig(simulations=30, add_noise=False, truncation_value=truncation_value),
                                  np.random.default_rng(0), spec=spec)
                    result = search.run(game, simulations=30)
                    self.assertEqual(list(game.moves), [])
                    self.assertEqual(result.simulations, 30)
                    self.assertEqual(float(result.visits.sum()), 30.0)
                    self.assertAlmostEqual(result.root_value, truncation_value)
                finally:
                    game.close()

    def test_batched_roots_keep_independent_inputs_with_a_reusing_encoder(self):
        """run_many queues inputs from several trees before one evaluation; a reusable encoder buffer
        must not make every tree evaluate the last root's position or mask."""
        from hasami_learn.algorithms.mcts import MCTS, MctsConfig
        from hasami_learn.encoding import PlaneEncoder
        from hasami_learn.games.hasami import HasamiSpec
        np = self.np

        class ReusingEncoder(PlaneEncoder):
            def __init__(self):
                super().__init__()
                self.buffer = None
                self.mask_buffer = None

            def encode(self, game, player=None):
                fresh = super().encode(game, player)
                if self.buffer is None:
                    self.buffer = np.zeros_like(fresh)
                self.buffer[...] = fresh
                return self.buffer

            def legal_mask(self, game, player=None):
                fresh = super().legal_mask(game, player)
                if self.mask_buffer is None:
                    self.mask_buffer = np.zeros_like(fresh)
                self.mask_buffer[...] = fresh
                return self.mask_buffer

        received = []

        def recording(planes, masks):
            received.append((planes.copy(), masks.copy()))
            return self.uniform(planes, masks)

        fresh = PlaneEncoder()
        first = create_game("python")
        second = create_game("python")
        try:
            second.play(second.legal_actions()[0])
            self.assertEqual((first.position.turn, second.position.turn), (0, 1))
            spec = HasamiSpec()
            search = MCTS(recording, ReusingEncoder(), MctsConfig(simulations=4, add_noise=False, batch_size=8),
                          np.random.default_rng(0), spec=spec)
            results = search.run_many([first, second], simulations=4)
            planes, masks = received[0]  # the two roots are evaluated in one batch
            self.assertEqual(planes.shape[0], 2)
            np.testing.assert_array_equal(planes[0], fresh.encode(first))
            np.testing.assert_array_equal(planes[1], fresh.encode(second))
            np.testing.assert_array_equal(masks[0], fresh.legal_mask(first))
            np.testing.assert_array_equal(masks[1], fresh.legal_mask(second))
            self.assertFalse(np.array_equal(planes[0], planes[1]))
            for game, result in zip((first, second), results):
                self.assertTrue(set(np.flatnonzero(result.visits)).issubset(game.legal_actions()))
        finally:
            first.close()
            second.close()
