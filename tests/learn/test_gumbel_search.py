"""Hand-computed Gumbel search and frozen default-search regressions."""

import hashlib
import json
import unittest

from tests.learn._common import requires_torch


@requires_torch
class GumbelMathTests(unittest.TestCase):
    def test_completed_q_and_deterministic_interior_selection(self):
        import numpy as np
        from hasami_learn.search.mcts import _Node, MctsConfig

        node = _Node()
        node.expand(np.ones(3), np.array([.2, .3, .5]), .4)
        node.visits[:] = [2, 1, 0]
        node.value_sum[:] = [1., -.5, 0]
        node.total = 3
        # Visited-prior weighted Q is (.2*.5 + .3*-.5)/.5 = -.1.
        mixed = (.4 + 3 * -.1) / 4
        np.testing.assert_allclose(node.completed_q(), [.5, -.5, mixed], atol=1e-15)
        config = MctsConfig(root_selection="gumbel", gumbel_c_visit=0, gumbel_c_scale=1)
        expected = np.array([.2, .3, .5]) * np.exp(2 * np.array([.5, -.5, mixed]))
        expected /= expected.sum()
        np.testing.assert_allclose(node.improved_policy(config), expected, atol=1e-15)
        self.assertEqual(node.select_improved(config), int(np.argmax(expected - np.array([2, 1, 0]) / 4)))

    def test_equal_q_reduces_to_prior_including_zeros(self):
        import numpy as np
        from hasami_learn.search.mcts import _Node, MctsConfig

        node = _Node()
        node.expand(np.ones(4), np.array([.2, 0, .3, .5]), .25)
        np.testing.assert_allclose(node.improved_policy(MctsConfig()), node.prior, atol=1e-15)
        node.visits[:] = [1, 0, 2, 0]
        node.value_sum[:] = [.25, 0, .5, 0]
        node.total = 3
        np.testing.assert_allclose(node.completed_q(), .25, atol=1e-15)
        np.testing.assert_allclose(node.improved_policy(MctsConfig()), node.prior, atol=1e-15)

    def test_sequential_halving_schedule_and_total_budget(self):
        import numpy as np
        from hasami_learn.search.mcts import _Node, _SequentialHalving, MctsConfig

        config = MctsConfig(root_selection="gumbel", gumbel_c_scale=0)
        for count, budget, expected in ((4, 8, [0, 1, 2, 3, 0, 1, 0, 1]),
                                         (5, 7, [0, 1, 2, 3, 4, 0, 1]),
                                         (1, 3, [0, 0, 0]), (6, 2, [0, 1])):
            with self.subTest(count=count, budget=budget):
                node = _Node()
                node.expand(np.ones(count), np.ones(count), 0)
                schedule = _SequentialHalving(node, config, np.zeros(count), budget)
                selected = []
                for _ in range(budget):
                    edge = schedule.select(node)
                    selected.append(edge)
                    node.visits[edge] += 1
                    node.total += 1
                self.assertEqual(selected, expected)
                self.assertEqual(schedule.winner(node), 0)
                self.assertEqual(node.total, budget)

    def test_invalid_configuration(self):
        from hasami_learn.search.mcts import MctsConfig
        for kw in ({"root_selection": "bad"}, {"gumbel_m": 0}, {"gumbel_m": True},
                   {"gumbel_c_visit": -1}, {"gumbel_c_scale": float("nan")}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                MctsConfig(**kw)


@requires_torch
class DefaultReferenceTests(unittest.TestCase):
    def test_head_reference_vectors(self):
        import numpy as np
        from hasami_learn.games import get_spec
        from hasami_learn.search.mcts import MCTS, MctsConfig
        from tests.learn.test_native_mcts import NativeMctsTests

        for name, noise, budget, digest in DEFAULT_VECTORS:
            with self.subTest(game=name, noise=noise, budget=budget):
                spec = get_spec(name)
                game = spec.create_game("python")
                try:
                    for ply in range(5):
                        game.play(game.legal_actions()[ply])
                    rng = np.random.default_rng(48)
                    result = MCTS(NativeMctsTests.hash_evaluator, spec.make_encoder(),
                                  MctsConfig(simulations=budget, add_noise=noise), rng, spec).run(game)
                    actual = hashlib.sha256(result.visits.tobytes() + result.q_values.tobytes()
                                            + np.float64(result.root_value).tobytes()
                                            + json.dumps(rng.bit_generator.state, sort_keys=True).encode()).hexdigest()
                    self.assertEqual(actual, digest)
                    np.testing.assert_array_equal(result.policy_target, result.visits / result.visits.sum())
                    self.assertIsNone(result.selected_action)
                finally:
                    game.close()


# Captured from HEAD before Gumbel search edits; covers visits, Q, root value and RNG state.
DEFAULT_VECTORS = [
    ('hasami', False, 0, '08cf8bceb56265f82503acb91e465ede84a872c9a005b27cddb01dfaf977f299'),
    ('hasami', False, 1, '29da0426e6d8cc0e4b0baa2a44bf054f4cc3ba973b719522076c4f30118d2539'),
    ('hasami', False, 16, '8be7c4d9a26a773193ef33ebd8a97879e933bf0b4276994c96db696a181ee2df'),
    ('hasami', False, 64, '6753fd766c9163bb06865468f111566f42819f1d91cb47bfc255bfd1941c77d0'),
    ('hasami', True, 0, '3135ad521ee3e4846d1709a74e9faf2a3364890b1a0872dbdbe6fddebc1896aa'),
    ('hasami', True, 1, 'eb502f3930dbcd94038354264665dff191d9435bc5c71c8efae0587108484d58'),
    ('hasami', True, 16, '4b3a2b0784608d6bb2430fd7889ec8dea5add198e70a91d6ad6df51c1d1040d6'),
    ('hasami', True, 64, 'a6df5f931f3f81f1e597811e5a210f4397091339066655904b96cba16883fd46'),
    ('shogi', False, 0, '2d44bddaef95b9bc7851d54b49dd0c51969abfeb4cc62fd1e4c9823b84a40ce6'),
    ('shogi', False, 1, 'db303dc30d587ad4edd87b3399c8415e12e0fd52e4f3bc3dfe5f1fcace63e159'),
    ('shogi', False, 16, '04658fb8c1ff11eca329eff751c816db1457aaf62ebe7719da4ced299551b7ec'),
    ('shogi', False, 64, '0f54e2533a1863308b40f9d58e6c73677e5da06a8d35cee33e48bb1fffd3baef'),
    ('shogi', True, 0, '7e396eb24992bd2131aa770e3ee0f09b12b813e38178e9ed9b74390b8187a772'),
    ('shogi', True, 1, 'f1c720c7731377e7da7feb3cca6e8224389f1e7e841a7ac73ff07b660db6de29'),
    ('shogi', True, 16, '8428e1b5bbfcecff6efdcfb88870aea38f5c38d0246ac12fc70d4ed9a2825646'),
    ('shogi', True, 64, '990989a3853cb18a24e93697d496fb98fdd8a357d5642704fa57c055c52e7139'),
]


def toy_spec(actions=128, extra_turn=False, truncated=False, undoable=True):
    """A deterministic two-ply bandit: action 7 wins, every other action draws."""
    import copy
    from types import SimpleNamespace
    import numpy as np
    from hasami_learn.games.base import StepOutcome

    class Game:
        def __init__(self):
            self.moves = []

        @property
        def position(self):
            return SimpleNamespace(turn=0 if extra_turn else len(self.moves) % 2)

        @property
        def outcome(self):
            done = len(self.moves) == 2
            return SimpleNamespace(terminal=done, winner=(0 if self.moves[0] == 7 else -1) if done else -2)

        def clone(self):
            return copy.deepcopy(self)

        def undo(self):
            self.moves.pop()

        def close(self):
            pass

    class Spec:
        game_id, ruleset_id = "exact-bandit", "test-only"
        num_players, perfect_information, has_chance = 2, True, False
        supports_undo = undoable
        action_size, board_shape = actions, (1, 1)
        num_planes = 1

        def actors(self, game):
            return [game.position.turn]

        def legal_mask(self, game, player=None):
            mask = np.zeros(actions)
            if not game.outcome.terminal:
                mask[:actions if not game.moves else 1] = 1
            return mask

        def encode(self, game, player=None):
            if truncated and game.moves:
                raise AssertionError("evaluated after adapter truncation")
            return np.array([[[game.moves[0] + 1 if game.moves else 0]]], np.float32)

        def step(self, game, joint):
            if truncated and game.moves:
                raise AssertionError("stepped after adapter truncation")
            player, action = next(iter(joint.items()))
            assert player == game.position.turn and self.legal_mask(game)[action]
            game.moves.append(action)
            return StepOutcome(game.outcome.terminal, truncated,
                               self.terminal_values(game) if game.outcome.terminal else None)

        def terminal_values(self, game):
            return [1., -1.] if game.moves[0] == 7 else [0., 0.]

        def evaluate(self, planes, masks):
            prior = masks.copy()
            values = np.zeros(len(planes))
            for i, plane in enumerate(planes):
                first = int(plane.item())
                if first == 0 and actions > 1:
                    prior[i] = .03 / (actions - 1)
                    prior[i, 0] = .97
                    values[i] = float(actions > 7)
                elif first == 8:
                    values[i] = 1. if extra_turn else -1.
            return prior, values

    return Spec(), Game()


@requires_torch
class GumbelTreeTests(unittest.TestCase):
    def test_exact_game_finds_optimal_in_eight_of_128_actions(self):
        import numpy as np
        from hasami_learn.search.mcts import MCTS, MctsConfig

        spec, game = toy_spec()
        results = {}
        for mode in ("puct", "gumbel"):
            config = MctsConfig(simulations=8, add_noise=False, root_selection=mode, gumbel_m=8)
            results[mode] = MCTS(spec.evaluate, spec, config, np.random.default_rng(48), spec).run(game)
        puct, gumbel = results["puct"], results["gumbel"]
        prior = np.full(128, .03 / 127)
        prior[0] = .97
        self.assertLessEqual(np.abs(puct.policy_target - prior).sum(), .061)
        self.assertEqual(int(puct.policy_target.argmax()), 0)
        self.assertEqual(gumbel.selected_action, 7)
        self.assertEqual(int(gumbel.policy_target.argmax()), 7)
        self.assertGreater(gumbel.policy_target[7], .99)
        self.assertLess(gumbel.simulations, spec.action_size / 8)
        self.assertEqual(game.moves, [])
        self.assertEqual(gumbel.q_values[7], 1.)

    def test_budget_legality_seed_and_batching(self):
        import numpy as np
        from hasami_learn.search.mcts import MCTS, MctsConfig

        for count in (1, 3, 17):
            for budget in (0, 1, 2, 7, 64):
                with self.subTest(actions=count, budget=budget):
                    spec, game = toy_spec(actions=count)
                    config = MctsConfig(root_selection="gumbel", simulations=budget, batch_size=2)
                    def run():
                        return MCTS(spec.evaluate, spec, config, np.random.default_rng(48), spec).run_many([game] * 3)
                    first, second = run(), run()
                    for a, b in zip(first, second):
                        np.testing.assert_array_equal(a.policy_target, b.policy_target)
                        np.testing.assert_array_equal(a.visits, b.visits)
                        self.assertEqual(a.selected_action, b.selected_action)
                        self.assertEqual(a.simulations, budget)
                        self.assertAlmostEqual(a.policy_target.sum(), 1, places=14)
                        self.assertTrue(np.isfinite(a.policy_target).all())
                        self.assertTrue((a.policy_target >= 0).all())
                        self.assertIn(a.selected_action, range(count))

    def test_extra_turns_adapter_stop_terminal_and_ownership(self):
        import numpy as np
        from hasami_learn.search.mcts import MCTS, MctsConfig

        for extra in (False, True):
            for undoable in (False, True):
                for truncated in (False, True):
                    spec, game = toy_spec(extra_turn=extra, truncated=truncated, undoable=undoable)
                    search = MCTS(spec.evaluate, spec, MctsConfig(root_selection="gumbel", simulations=24,
                                  gumbel_m=8, add_noise=False, truncation_value=.4), np.random.default_rng(1), spec)
                    result = search.run(game)
                    self.assertEqual(game.moves, [])
                    self.assertEqual(result.simulations, 24)
                    if truncated:
                        self.assertAlmostEqual(result.root_value, .4)
                        np.testing.assert_allclose(result.q_values[np.isfinite(result.q_values)], .4)
                    else:
                        self.assertEqual(result.selected_action, 7)
                        self.assertEqual(result.q_values[7], 1.)
                spec, game = toy_spec(extra_turn=extra, undoable=undoable)
                game.moves[:] = [7, 0]
                search = MCTS(spec.evaluate, spec, MctsConfig(root_selection="gumbel"), np.random.default_rng(1), spec)
                result = search.run(game)
                self.assertEqual(result.simulations, 0)
                self.assertEqual(result.root_value, 1.)
                self.assertIsNone(result.selected_action)
                self.assertEqual(result.policy_target.sum(), 0)

    def test_cancelled_root_evaluation_and_noiseless_rng(self):
        import threading
        import numpy as np
        from hasami_learn.search.mcts import MCTS, MctsConfig

        spec, game = toy_spec()
        rng = np.random.default_rng(48)
        before = rng.bit_generator.state
        cancel = threading.Event()
        cancel.set()
        search = MCTS(spec.evaluate, spec, MctsConfig(root_selection="gumbel", add_noise=False), rng, spec)
        result = search.run(game, cancel=cancel)
        self.assertEqual(result.simulations, 0)
        self.assertEqual(result.selected_action, 0)
        self.assertAlmostEqual(result.policy_target[0], .97)
        self.assertEqual(rng.bit_generator.state, before)
