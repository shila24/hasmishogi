"""AlphaZero training, replay targets, and checkpoint inference regressions."""

import json
import math
import time

from hasami import create_game
from hasami.agents import RandomAgent, SearchLimits
from tests.learn._common import LearnTestCase, requires_torch


@requires_torch
class AlphaZeroTests(LearnTestCase):
    def test_registry_and_tiny_training_checkpoint_matches(self):
        import hasami_learn.algorithms
        from hasami_learn.evaluation import play_matches
        from hasami_learn.inference import MctsAgent, create_agent
        from hasami_learn.registry import ALGORITHMS
        from hasami_learn.trainer import train_from_config

        self.assertIn("alphazero", ALGORITHMS)
        start = time.perf_counter()
        checkpoint = train_from_config(self.tiny("alphazero", simulations=8, batch_games=2), echo=False)
        self.assertLess(time.perf_counter() - start, 90)
        self.assertTrue((checkpoint / "weights.pt").is_file())
        run_dir = checkpoint.parent.parent
        rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
        self.assertIn("collect", [row["kind"] for row in rows])
        update = next(row for row in rows if row["kind"] == "update")
        self.assertGreater(update["steps"], 0)
        for key in ("loss", "policy_loss", "value_loss"):
            self.assertTrue(math.isfinite(update[key]))
        agent = create_agent(str(run_dir), device="cpu")
        self.assertIsInstance(agent, MctsAgent)
        self.assertEqual(agent.simulations, 8)
        match = play_matches(agent, RandomAgent(7), 2, backend="python", max_plies=30,
                             limits=SearchLimits(1, 1000, 8))
        self.assertEqual(match.games, 2)
        self.assertGreater(match.plies, 0)

    def trainer(self, **params):
        from hasami_learn.trainer import Trainer

        trainer = Trainer(self.tiny("alphazero", simulations=2, batch_games=2, **params), echo=False)
        self.addCleanup(trainer.close)
        return trainer

    def test_collect_batches_and_replay_threshold_update(self):
        import numpy as np
        import torch
        from unittest.mock import patch

        trainer = self.trainer()
        cfg = trainer.config
        cfg.selfplay.games_per_iteration = 3
        cfg.selfplay.max_plies = 2
        cfg.selfplay.symmetry_augmentation = True
        cfg.selfplay.truncation_value = 0.4
        learner = trainer.learner
        self.assertEqual(learner.update(1).steps, 0)
        with patch.object(learner.search, "run_many", wraps=learner.search.run_many) as run_many:
            stats = learner.collect(1)
        self.assertEqual([len(call.args[0]) for call in run_many.call_args_list], [2, 2, 1, 1])
        self.assertEqual((stats.games, stats.plies, stats.truncated), (3, 6, 3))
        data = learner.replay.all()
        self.assertEqual(len(learner.replay), 12)
        np.testing.assert_allclose(data["value"], 0.4)
        np.testing.assert_allclose(data["policy"].sum(1), 1)
        np.testing.assert_array_equal(data["policy"][data["mask"] == 0], 0)
        np.testing.assert_array_equal(data["planes"][1], learner.encoder.flip_horizontal(data["planes"][0]))
        np.testing.assert_array_equal(data["policy"][1], learner.encoder.flip_horizontal_policy(data["policy"][0]))
        cfg.replay.min_size = 12
        before = [param.detach().clone() for param in learner.network.parameters()]
        update = learner.update(1)
        self.assertEqual(update.steps, cfg.optimizer.updates_per_iteration)
        self.assertTrue(math.isfinite(update.loss))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, learner.network.parameters())))

    def test_actor_and_ingest_against_opponent(self):
        import numpy as np
        from hasami_learn.selfplay import run_vs_opponent_game

        trainer = self.trainer(temperature_plies=0, eval_simulations=5)
        learner = trainer.learner
        trajectory = run_vs_opponent_game(learner.actor_fn(), RandomAgent(1), 1, learner.encoder,
                                          backend="python", max_plies=6)
        self.assertEqual(trajectory.players, [1, 1, 1])
        for action, policy, mask in zip(trajectory.actions, trajectory.policies, trajectory.masks):
            self.assertIs(type(action), int)
            self.assertGreater(mask[action], 0)
            self.assertEqual(action, int(np.argmax(policy)))
        self.assertEqual(learner.ingest_trajectories([trajectory]), 3)
        trajectory.truncated = False
        trajectory.rewards = (1.0, -1.0)
        self.assertEqual(learner.ingest_trajectories([trajectory]), 3)
        np.testing.assert_array_equal(learner.replay.all()["value"][3:], -1)
        self.assertEqual(learner.inference_hint(), {"mode": "mcts", "simulations": 5, "mcts_params": {"truncation_value": 0.0}})
        agent = learner.make_agent()
        self.assertFalse(agent.config.add_noise)
        self.assertEqual(agent.simulations, 5)
        game = create_game("python")
        self.addCleanup(game.close)
        decision = agent.choose(game, SearchLimits(1, 1000, 2))
        self.assertIs(type(decision.action), int)
        self.assertIn(decision.action, game.legal_actions())
        self.assertEqual(decision.nodes, 2)


@requires_torch
class TruncationValuePersistenceTests(LearnTestCase):
    """multiplayer search review: the search's truncation value must survive checkpointing and pooled inference."""

    def test_checkpoint_inference_restores_truncation_value(self):
        from hasami_learn.inference import MctsAgent, create_agent
        from hasami_learn.manifest import validate_manifest_dict
        from hasami_learn.trainer import Trainer, train_from_config

        config = self.tiny("alphazero", simulations=2, batch_games=2)
        config.selfplay.truncation_value = -0.4
        trainer = Trainer(config, echo=False)
        try:
            self.assertEqual(trainer.learner.search.config.truncation_value, -0.4)
            self.assertEqual(trainer.learner.make_agent().config.truncation_value, -0.4)
            manifest = trainer.manifest(0)
            self.assertEqual(manifest.inference_hint["mcts_params"], {"truncation_value": -0.4})
        finally:
            trainer.close()
        checkpoint = train_from_config(config, echo=False)
        saved = json.loads((checkpoint / "manifest.json").read_text())
        self.assertEqual(validate_manifest_dict(saved), [])  # schema admits the persisted search settings
        self.assertEqual(saved["inference_hint"]["mcts_params"], {"truncation_value": -0.4})
        agent = create_agent(str(checkpoint.parent.parent), device="cpu")
        self.assertIsInstance(agent, MctsAgent)
        self.assertEqual(agent.config.truncation_value, -0.4)
        override = create_agent(str(checkpoint.parent.parent), device="cpu", mcts_params={"truncation_value": 0.1})
        self.assertEqual(override.config.truncation_value, 0.1)


@requires_torch
class GumbelValueMixTests(LearnTestCase):
    def test_decision_boundary_and_stored_policy(self):
        import numpy as np
        from hasami_learn.selfplay import search_decision
        from hasami_learn.search.mcts import MctsResult

        result = MctsResult(np.array([8., 2., 0.]), .25, 10,
                            policy_target=np.array([.1, .9, 0.]), selected_action=0)
        rng = np.random.default_rng(48)
        before = rng.bit_generator.state
        for ply, expected in ((0, 0), (1, 0), (2, 1), (3, 1)):
            action, policy, extra = search_decision(result, np.array([1, 1, 0]), ply, rng,
                root_selection="gumbel", temperature=0, temperature_plies=2, value_target_mix=.5)
            self.assertEqual(action, expected)
            np.testing.assert_array_equal(policy, result.policy_target.astype(np.float32))
            self.assertEqual(extra["root_value"], .25)
        self.assertEqual(rng.bit_generator.state, before)

    def test_value_mix_outcome_draw_truncation_and_symmetry(self):
        import numpy as np
        from hasami_learn.algorithms.alphazero import AlphaZeroParams
        from hasami_learn.selfplay import Trajectory
        from hasami_learn.trainer import Trainer

        for mix in (-.1, 1.1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                AlphaZeroParams(value_target_mix=mix)
        config = self.tiny("alphazero", simulations=2, value_target_mix=.5)
        config.selfplay.symmetry_augmentation = True
        config.selfplay.truncation_value = .4
        trainer = Trainer(config, echo=False)
        self.addCleanup(trainer.close)
        learner = trainer.learner
        game = learner.spec.create_game("python")
        self.addCleanup(game.close)
        mask = learner.spec.legal_mask(game, 0)
        for kind, outcome in (("win", 1.), ("loss", -1.), ("draw", -.2), ("cap", .4)):
            trajectory = Trajectory(winner=0 if kind == "win" else 1 if kind == "loss" else -1,
                                    rewards=[1., -1.] if kind == "win" else [-1., 1.],
                                    reason="repetition", draw_value=-.2, truncated=kind == "cap")
            trajectory.append(learner.encoder.encode(game), mask.copy(), int(np.flatnonzero(mask)[0]),
                              0, mask / mask.sum(), {"root_value": -.6})
            learner.ingest_trajectories([trajectory])
            self.assertAlmostEqual(trajectory.extras[0]["outcome_value"], outcome)
            self.assertEqual(trajectory.extras[0]["root_value"], -.6)
            expected = .5 * outcome + .5 * -.6
            self.assertAlmostEqual(trajectory.value_for(0, .4), expected)
            self.assertAlmostEqual(trajectory.extras[0]["value_target"], expected)
            np.testing.assert_allclose(learner.replay.all()["value"][-2:], expected, atol=1e-7)
        for mix in (0., 1.):
            trajectory.extras[0]["value_target_mix"] = mix
            self.assertEqual(trajectory.value_for(0, .4), .4 if not mix else -.6)

    def test_playtime_is_noiseless_argmax_even_with_temperature(self):
        import numpy as np
        from unittest.mock import patch
        from hasami_learn.inference import MctsAgent
        from hasami_learn.search.mcts import MctsResult
        from hasami_learn.trainer import Trainer

        trainer = Trainer(self.tiny("alphazero", simulations=2), echo=False)
        self.addCleanup(trainer.close)
        game = trainer.spec.create_game("python")
        self.addCleanup(game.close)
        agent = MctsAgent(trainer.network, trainer.encoder, "cpu", simulations=2,
                         mcts_params={"root_selection": "gumbel", "add_noise": True}, temperature=2,
                         spec=trainer.spec)
        self.assertFalse(agent.config.add_noise)
        legal = trainer.spec.legal_actions(game, 0)
        visits = np.zeros(trainer.spec.action_size)
        policy = visits.copy()
        visits[legal[0]], policy[legal[-1]] = 2, 1
        result = MctsResult(visits, .2, 2, policy_target=policy, selected_action=int(legal[0]))
        before = agent.rng.bit_generator.state
        with patch.object(agent.search, "run", return_value=result):
            self.assertEqual(agent.choose(game, SearchLimits(1, 1000, 2)).action, legal[-1])
        self.assertEqual(agent.rng.bit_generator.state, before)
