"""PPO rollout, objective, and checkpoint regression tests."""

from __future__ import annotations

import math
import time
from unittest.mock import Mock, patch

from hasami import create_game
from hasami.agents import RandomAgent, SearchLimits
from tests.learn._common import LearnTestCase, requires_torch


@requires_torch
class PpoTests(LearnTestCase):
    def trainer(self, **params):
        from hasami_learn.trainer import Trainer

        config = self.tiny("ppo", **{"epochs": 1, "minibatches": 1, **params})
        trainer = Trainer(config, echo=False)
        self.addCleanup(trainer.close)
        return trainer

    def trajectory(self, player=0):
        from hasami_learn.encoding import PlaneEncoder
        from hasami_learn.selfplay import Trajectory

        game = create_game("python")
        self.addCleanup(game.close)
        encoder = PlaneEncoder()
        trajectory = Trajectory(rewards=(1.0, -1.0), winner=0, reason="five_captures")
        for value in (0.2, 0.4, 0.6):
            trajectory.append(encoder.encode(game), encoder.legal_mask(game),
                              game.legal_actions()[0], player, None,
                              {"logp": -math.log(63), "value": value, "temperature": 1.0})
        return trajectory

    def test_registration_and_defaults(self):
        from hasami_learn.algorithms.ppo import PpoLearner, PpoParams
        from hasami_learn.registry import ALGORITHMS

        self.assertIs(ALGORITHMS.get("ppo"), PpoLearner)
        self.assertEqual(PpoLearner.required_head, "policy_value")
        self.assertIsNone(PpoLearner.replay_kind)
        self.assertEqual(PpoParams().epochs, 4)
        self.assertEqual(PpoParams().minibatches, 4)
        learner = self.trainer().learner
        self.assertEqual(learner.inference_hint(), {"mode": "policy", "simulations": 0})
        self.assertEqual(learner.make_agent().temperature, 0)

    def test_gae_terminal_reward_closed_form_and_colour(self):
        import numpy as np
        from hasami_learn.algorithms.ppo import compute_gae

        trajectory = self.trajectory()
        advantage, returns = compute_gae(trajectory, [0.2, 0.4, 0.6], gamma=0.9, gae_lambda=0.8)
        # Deltas are 0.16, 0.14, 0.4; gamma * lambda is 0.72.
        np.testing.assert_allclose(advantage, [0.46816, 0.428, 0.4], rtol=1e-6)
        np.testing.assert_allclose(returns, [0.66816, 0.828, 1.0], rtol=1e-6)
        learner = self.trainer(gamma=0.9, gae_lambda=0.8).learner
        learner.ingest_trajectories([trajectory])
        np.testing.assert_allclose(learner._advantages, advantage)
        np.testing.assert_allclose(learner._returns, returns)
        _, white_returns = compute_gae(self.trajectory(player=1), [0.2, 0.4, 0.6], gae_lambda=1)
        np.testing.assert_allclose(white_returns, [-1, -1, -1], atol=1e-6)

    def test_truncation_target_and_empty_trajectory(self):
        import numpy as np
        from hasami_learn.algorithms.ppo import compute_gae
        from hasami_learn.selfplay import Trajectory

        trajectory = self.trajectory()
        trajectory.truncated = True
        _, returns = compute_gae(trajectory, [0.2, 0.4, 0.6], gae_lambda=1, truncation_value=0.3)
        np.testing.assert_allclose(returns, [0.3, 0.3, 0.3], atol=1e-6)
        self.assertEqual(compute_gae(Trajectory(), [])[0].size, 0)
        learner = self.trainer().learner
        self.assertEqual(learner.ingest_trajectories([Trajectory()]), 0)
        self.assertEqual(learner.update(1).steps, 0)

    def test_collect_legal_finite_and_current_opponent(self):
        import numpy as np
        from hasami_learn.selfplay import run_vs_opponent_game

        trainer = self.trainer()
        learner = trainer.learner
        trainer.config.selfplay.temperature = 0.7
        with patch("hasami_learn.algorithms.ppo.run_vs_opponent_game", wraps=run_vs_opponent_game) as runner:
            stats = learner.collect(1)
        self.assertEqual(stats.games, 2)
        self.assertGreater(stats.extra["samples"], 0)
        self.assertEqual(runner.call_count, 2)
        for call in runner.call_args_list:
            self.assertEqual(call.args[1].temperature, 0.7)
            self.assertIn(call.args[2], (0, 1))
        for mask, action in zip(learner._masks, learner._actions):
            self.assertIs(type(action), int)
            self.assertEqual(mask[action], 1)
        for values in (learner._logp_old, learner._value_old, learner._advantages, learner._returns):
            self.assertTrue(np.isfinite(values).all())
        self.assertEqual(stats.wins[2], 0)
        self.assertEqual(stats.truncated, 2)
        self.assertEqual(trainer.opponents.current.games, 2)
        self.assertEqual(stats.extra["pool_truncated_neutral_scores"], 2)

    def test_actor_temperature_and_single_forward(self):
        import numpy as np

        trainer = self.trainer()
        learner = trainer.learner
        game = create_game("python")
        self.addCleanup(game.close)
        self.assertEqual(len(game.legal_actions()), 63)
        leaves = 0
        for action in game.legal_actions():
            game.play(action)
            leaves += len(game.legal_actions())
            game.undo()
        self.assertEqual(leaves, 3717)
        planes, mask = trainer.encoder.encode(game), trainer.encoder.legal_mask(game)
        logits = np.zeros((1, 6561), np.float32)
        legal = np.flatnonzero(mask)
        logits[0, legal] = np.linspace(-2, 2, len(legal))
        trainer.config.selfplay.temperature = 0.5
        for ply, temperature in ((0, 0.5), (10, 1.0)):
            with patch.object(learner, "infer", return_value=(logits, np.array([0.25]))) as infer:
                action, _, extra = learner.actor_fn()(game, planes, mask, ply)
            infer.assert_called_once()
            z = logits[0, legal].astype(np.float64) / temperature
            expected_logp = z - np.log(np.exp(z).sum())
            self.assertAlmostEqual(extra["logp"], expected_logp[list(legal).index(action)], places=5)
            self.assertEqual(extra["value"], 0.25)
            self.assertIn(action, game.legal_actions())
        trainer.config.selfplay.temperature = 0
        with patch.object(learner, "infer", return_value=(logits, np.array([0.25]))):
            action, _, extra = learner.actor_fn()(game, planes, mask, 0)
        self.assertEqual(action, int(legal[-1]))
        self.assertEqual(extra["logp"], 0)

    def test_ingest_missing_extras_and_symmetry(self):
        import numpy as np

        trainer = self.trainer()
        learner = trainer.learner
        trainer.config.selfplay.symmetry_augmentation = True
        trajectory = self.trajectory()
        trajectory.extras = []
        self.assertEqual(learner.ingest_trajectories([trajectory]), 6)
        self.assertTrue(np.isfinite(learner._logp_old).all())
        for i in range(0, 6, 2):
            self.assertEqual(learner._actions[i + 1], trainer.encoder.flip_horizontal_action(learner._actions[i]))
            np.testing.assert_array_equal(learner._planes[i + 1], trainer.encoder.flip_horizontal(learner._planes[i]))
            np.testing.assert_array_equal(learner._masks[i + 1], trainer.encoder.flip_horizontal_policy(learner._masks[i]))
            for values in (learner._logp_old, learner._value_old, learner._advantages, learner._returns):
                self.assertEqual(values[i], values[i + 1])
        trajectory.extras = [{"logp": -2.0}, {"value": 0.75}, {}]
        learner.ingest_trajectories([trajectory])
        self.assertEqual(learner._logp_old[6], -2.0)
        self.assertEqual(learner._value_old[8], 0.75)

    def test_update_finite_and_on_policy_ratio(self):
        import torch

        trainer = self.trainer()
        learner = trainer.learner
        trainer.config.selfplay.temperature = 0.5
        learner.collect(1)
        before = [p.detach().clone() for p in learner.network.parameters()]
        stats = learner.update(1)
        for loss in (stats.loss, stats.policy_loss, stats.value_loss, stats.extra["approx_kl"]):
            self.assertTrue(math.isfinite(loss))
        self.assertEqual(stats.steps, 1)
        self.assertAlmostEqual(stats.extra["approx_kl"], 0, places=6)
        self.assertGreaterEqual(stats.extra["clip_fraction"], 0)
        self.assertLessEqual(stats.extra["clip_fraction"], 1)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, learner.network.parameters())))
        self.assertTrue(all(torch.isfinite(p).all() for p in learner.network.parameters()))
        self.assertEqual(learner.update(2).steps, 0)

    def test_clipped_objective_and_kl_stop(self):
        import torch

        class FixedPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.zeros(6561))
                self.value = torch.nn.Parameter(torch.tensor(0.75))

            def forward(self, planes):
                return self.logits.expand(len(planes), -1), self.value.expand(len(planes))

        for value_clip, target_kl in ((0.2, None), (0.0, None), (0.2, 0.01)):
            with self.subTest(value_clip=value_clip, target_kl=target_kl):
                trainer = self.trainer(epochs=3, minibatches=4, entropy_coef=0,
                                       normalize_advantage=False, value_clip=value_clip,
                                       target_kl=target_kl)
                learner = trainer.learner
                learner.network = FixedPolicy()
                trainer.config.optimizer.lr = 0
                trajectory = self.trajectory()
                for mask, extra in zip(trajectory.masks, trajectory.extras):
                    legal = mask.nonzero()[0][:2]
                    mask[:] = 0
                    mask[legal] = 1
                    extra.update(logp=math.log(0.25), value=0)
                learner.params.gae_lambda = 1
                learner.ingest_trajectories([trajectory])
                stats = learner.update(1)
                self.assertAlmostEqual(stats.policy_loss, -1.2, places=6)
                self.assertAlmostEqual(stats.value_loss, 0.32 if value_clip else 0.03125, places=6)
                self.assertAlmostEqual(stats.extra["approx_kl"], 1 - math.log(2), places=6)
                self.assertEqual(stats.extra["clip_fraction"], 1)
                self.assertEqual(stats.steps, 0 if target_kl else 9)
                self.assertEqual(stats.extra["early_stop"], target_kl is not None)

    def test_pool_result_perspective_and_classical_agent(self):
        trainer = self.trainer()
        learner = trainer.learner
        entry = trainer.opponents.classical[0]
        learner.rng = Mock(wraps=trainer.rng)
        learner.rng.integers.side_effect = [0, 1]
        trajectories = [self.trajectory(player=0), self.trajectory(player=1)]
        with patch.object(trainer.opponents, "sample", return_value=entry), \
                patch.object(trainer.opponents, "agent_for", wraps=trainer.opponents.agent_for) as agent_for, \
                patch("hasami_learn.algorithms.ppo.run_vs_opponent_game", side_effect=trajectories):
            stats = learner.collect(1)
        self.assertEqual(agent_for.call_count, 2)
        self.assertEqual((entry.wins, entry.losses, entry.draws), (1, 1, 0))
        self.assertEqual(stats.wins, (2, 0, 0))
        self.assertEqual(stats.truncated, 0)

    def test_resnet_update_preserves_rollout_normalization(self):
        from hasami_learn.trainer import Trainer

        config = self.tiny("ppo", epochs=1, minibatches=1)
        config.network.name = "resnet"
        config.network.params = {"channels": 8, "blocks": 1, "hidden": 16}
        config.selfplay.max_plies = 4
        trainer = Trainer(config, echo=False)
        self.addCleanup(trainer.close)
        trainer.learner.collect(1)
        stats = trainer.learner.update(1)
        self.assertTrue(math.isfinite(stats.loss))
        self.assertAlmostEqual(stats.extra["approx_kl"], 0, places=6)
        self.assertEqual(stats.extra["clip_fraction"], 0)

    def test_invalid_parameters_and_illegal_ingestion(self):
        from hasami_learn.algorithms.ppo import PpoParams
        from hasami_learn.config import ConfigError

        for params in ({"epochs": 0}, {"minibatches": 0}, {"gamma": 1.1},
                       {"gae_lambda": -1}, {"clip": float("nan")}, {"target_kl": 0}):
            with self.subTest(params=params), self.assertRaises(ConfigError):
                PpoParams(**params)
        trajectory = self.trajectory()
        trajectory.actions[0] = 0
        learner = self.trainer().learner
        with self.assertRaisesRegex(ValueError, "illegal action"):
            learner.ingest_trajectories([trajectory])

    def test_checkpoint_load_legal_game_and_resume(self):
        from hasami.env import TurnBasedEnv
        from hasami_learn.inference import create_agent
        from hasami_learn.trainer import Trainer, train_from_config

        config = self.tiny("ppo")
        start = time.perf_counter()
        final = train_from_config(config, echo=False)
        self.assertLess(time.perf_counter() - start, 60)
        self.assertTrue((final / "weights.pt").is_file())
        run_dir = final.parent.parent
        agent = create_agent(run_dir, device="cpu")
        random = RandomAgent(9)
        env = TurnBasedEnv("python", max_plies=100)
        self.addCleanup(env.close)
        while not env.game.outcome.terminal and not env.truncated:
            actor = agent if env.game.position.turn == 0 else random
            snapshot = env.game.clone()
            try:
                decision = actor.choose(snapshot, SearchLimits(1, 100, 10))
            finally:
                snapshot.close()
            self.assertIs(type(decision.action), int)
            self.assertIn(decision.action, env.game.legal_actions())
            env.step(decision.action)
        config.run.iterations = 2
        resumed = Trainer(config, resume=run_dir, echo=False)
        self.addCleanup(resumed.close)
        self.assertEqual(resumed.iteration, 1)
        self.assertTrue(resumed.learner.optimizer.state)
        self.assertEqual(resumed.run().name, "iter_000002")
