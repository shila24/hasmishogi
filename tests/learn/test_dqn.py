"""DQN regression tests, optional when torch is unavailable."""

from __future__ import annotations

import math
from unittest.mock import patch

from tests.learn._common import LearnTestCase, ROOT, requires_torch


@requires_torch
class DqnTests(LearnTestCase):
    def trainer(self, config=None, **params):
        from hasami_learn.trainer import Trainer

        trainer = Trainer(config or self.tiny("dqn", **params), echo=False)
        self.addCleanup(trainer.close)
        return trainer

    def test_registration_and_configs(self):
        from hasami_learn.algorithms.dqn import DqnLearner, DqnParams
        from hasami_learn.config import TrainingConfig
        from hasami_learn.registry import ALGORITHMS

        self.assertIs(ALGORITHMS.get("dqn"), DqnLearner)
        self.assertEqual(DqnLearner.required_head, "q")
        self.assertEqual(DqnLearner.replay_kind, "q")
        self.assertEqual(DqnParams().n_step, 1)
        for path in ("configs/dqn.json", "configs/tiny/dqn.json"):
            config = TrainingConfig.load(ROOT / path)
            self.assertEqual(config.algorithm.name, "dqn")

    def test_collect_legal_same_player_transitions(self):
        import numpy as np
        from hasami_learn.config import TrainingConfig
        from hasami_learn.selfplay import run_vs_opponent_game

        config = TrainingConfig.load(ROOT / "configs/tiny/dqn.json")
        config.run.output_dir = str(self.tmp)
        trainer = self.trainer(config)
        trajectories = []

        def record(*args, **kwargs):
            trajectory = run_vs_opponent_game(*args, **kwargs)
            trajectories.append(trajectory)
            return trajectory

        with patch("hasami_learn.algorithms.dqn.run_vs_opponent_game", side_effect=record):
            stats = trainer.learner.collect(1)
        data = trainer.replay.all()
        self.assertEqual(stats.games, 2)
        self.assertGreaterEqual(len(trainer.replay), config.replay.min_size)
        self.assertEqual(int(data["done"].sum()), stats.games)
        self.assertTrue(np.all(data["mask"][np.arange(len(trainer.replay)), data["action"]] == 1))
        self.assertEqual(set(data["done"]), {0.0, 1.0})
        offset = 0
        for trajectory in trajectories:
            self.assertEqual(len(set(trajectory.players)), 1)
            count = len(trajectory)
            np.testing.assert_array_equal(data["next_planes"][offset:offset + count - 1], trajectory.planes[1:])
            np.testing.assert_array_equal(data["next_mask"][offset:offset + count - 1], trajectory.masks[1:])
            self.assertTrue(np.all(data["reward"][offset:offset + count - 1] == 0))
            self.assertEqual(data["reward"][offset + count - 1], trajectory.value_for(-1, config.selfplay.truncation_value))
            offset += count
        done = data["done"] == 1
        self.assertFalse(data["next_planes"][done].any())
        self.assertFalse(data["next_mask"][done].any())
        self.assertEqual(sum(stats.wins) + stats.truncated, stats.games)

    def test_actor_masks_greedy_and_exploratory_actions(self):
        import numpy as np
        from hasami import create_game
        from scripts.benchmark import python_perft

        learner = self.trainer().learner
        game = create_game("python")
        self.addCleanup(game.close)
        self.assertEqual(len(game.legal_actions()), 63)
        self.assertEqual(python_perft(game, 2), 3717)
        planes, mask = learner.encoder.encode(game), learner.encoder.legal_mask(game)
        best = game.legal_actions()[-1]
        q = np.full((1, 6561), 1e6, np.float32)
        q[0, mask > 0] = -2
        q[0, best] = -1
        actor = learner.actor_fn()
        learner.last_epsilon = 0.0
        with patch.object(learner, "infer", return_value=(q, None)):
            action, policy, _ = actor(game, planes, mask, 0)
        self.assertEqual(action, best)
        self.assertIs(type(action), int)
        self.assertIsNone(policy)
        learner.last_epsilon = 1.0
        for _ in range(20):
            action, _, _ = actor(game, planes, mask, 0)
            self.assertIn(action, game.legal_actions())
            self.assertIs(type(action), int)
        with self.assertRaises(ValueError):
            actor(game, planes, np.zeros_like(mask), 0)

    def test_epsilon_schedule(self):
        learner = self.trainer().learner
        learner.config.selfplay.games_per_iteration = 0
        for iteration, expected in ((0, 1.0), (1, 1.0), (26, 0.525), (51, 0.05), (100, 0.05)):
            self.assertAlmostEqual(learner.collect(iteration).extra["epsilon"], expected)

    def test_ingest_terminal_rewards_and_horizontal_augmentation(self):
        import numpy as np
        from hasami_learn.selfplay import Trajectory

        learner = self.trainer().learner
        learner.config.selfplay.symmetry_augmentation = True
        learner.config.selfplay.truncation_value = 0.25
        for player, truncated, expected in ((0, False, -1), (1, False, 1), (1, True, 0.25)):
            with self.subTest(player=player, truncated=truncated):
                learner.replay.clear()
                trajectory = Trajectory(rewards=(-1, 1), truncated=truncated)
                for index, action in enumerate((72 * 81 + 45, 45 * 81 + 46)):
                    planes = np.zeros((9, 9, 9), np.float32)
                    planes[0, index, 0] = 1
                    mask = np.zeros(6561, np.float32)
                    mask[action] = 1
                    trajectory.append(planes, mask, action, player, None, {})
                self.assertEqual(learner.ingest_trajectories([Trajectory(), trajectory]), 4)
                data = learner.replay.all()
                np.testing.assert_array_equal(data["done"], [0, 0, 1, 1])
                np.testing.assert_array_equal(data["reward"], [0, 0, expected, expected])
                for original, mirrored in ((0, 1), (2, 3)):
                    for field in ("planes", "next_planes"):
                        np.testing.assert_array_equal(data[field][mirrored], learner.encoder.flip_horizontal(data[field][original]))
                    for field in ("mask", "next_mask"):
                        np.testing.assert_array_equal(data[field][mirrored], learner.encoder.flip_horizontal_policy(data[field][original]))
                    self.assertEqual(data["action"][mirrored], learner.encoder.flip_horizontal_action(int(data["action"][original])))
                np.testing.assert_array_equal(data["next_planes"][0], trajectory.planes[1])

    def test_update_waits_then_returns_finite_loss(self):
        trainer = self.trainer()
        learner = trainer.learner
        self.assertEqual(learner.update(1).steps, 0)
        learner.collect(1)
        trainer.config.replay.min_size = len(trainer.replay) + 1
        self.assertEqual(learner.update(1).steps, 0)
        trainer.config.replay.min_size = len(trainer.replay)
        stats = learner.update(1)
        self.assertEqual(stats.steps, trainer.config.optimizer.updates_per_iteration)
        self.assertTrue(math.isfinite(stats.loss))

    def test_targets_use_legal_actions_and_zero_terminal_bootstrap(self):
        import torch

        learner = self.trainer(gamma=0.5).learner
        batch = learner.to_device({
            "next_planes": torch.zeros(4, 9, 9, 9),
            "next_mask": torch.zeros(4, 6561),
            "reward": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "done": torch.tensor([0.0, 0.0, 1.0, 1.0]),
        })
        batch["next_mask"][[0, 2], 1:3] = 1
        online = torch.full((4, 6561), 1000.0, requires_grad=True)
        target = torch.full((4, 6561), 2000.0, requires_grad=True)
        with torch.no_grad():
            online[:, 1:3] = torch.tensor([-2.0, -1.0])
            target[:, 1:3] = torch.tensor([8.0, 4.0])
        for double, expected in ((True, [3, 2, 3, 4]), (False, [5, 2, 3, 4])):
            learner.params.double = double
            with patch.object(learner.network, "forward", return_value=(online, None)), patch.object(learner.target_network, "forward", return_value=(target, None)):
                targets = learner._targets(batch)
            torch.testing.assert_close(targets, torch.tensor(expected, dtype=torch.float32))
            self.assertFalse(targets.requires_grad)

    def test_target_network_changes_only_on_hard_update(self):
        import torch

        config = self.tiny("dqn", target_update_every=2)
        config.network.name = "resnet"
        config.network.params = {"channels": 4, "blocks": 1, "hidden": 8}
        config.optimizer.updates_per_iteration = 1
        learner = self.trainer(config).learner
        initial = {k: v.clone() for k, v in learner.target_network.state_dict().items()}
        self.assertIsNot(learner.network, learner.target_network)
        self.assertTrue(all(not p.requires_grad and p.device == learner.device for p in learner.target_network.parameters()))
        learner.collect(1)
        learner.update(1)
        self.assertTrue(all(torch.equal(v, learner.target_network.state_dict()[k]) for k, v in initial.items()))
        self.assertTrue(any(not torch.equal(v, learner.network.state_dict()[k]) for k, v in initial.items()))
        learner.update(2)
        self.assertEqual(learner.gradient_steps, 2)
        self.assertTrue(all(torch.equal(v, learner.target_network.state_dict()[k]) for k, v in learner.network.state_dict().items()))
        self.assertTrue(any(not torch.equal(v, learner.target_network.state_dict()[k]) for k, v in initial.items()))
        self.assertFalse(learner.target_network.training)

    def test_train_load_legal_matches_and_resume(self):
        import torch
        from hasami.agents import RandomAgent, SearchLimits
        from hasami_learn.evaluation import play_matches
        from hasami_learn.inference import QAgent, create_agent
        from hasami_learn.manifest import load_checkpoint
        from hasami_learn.trainer import Trainer, train_from_config

        config = self.tiny("dqn")
        config.replay.persist = True
        checkpoint = train_from_config(config, echo=False)
        run_dir = checkpoint.parent.parent
        agent = create_agent(run_dir, device="cpu")
        self.assertIsInstance(agent, QAgent)
        self.assertEqual(agent.epsilon, 0)
        match = play_matches(agent, RandomAgent(3), 2, backend="python", max_plies=30, limits=SearchLimits(1, 200, 20000))
        self.assertEqual(match.games, 2)
        self.assertGreater(match.plies, 0)
        saved = load_checkpoint(checkpoint, "cpu").learner_state["extra"]
        trainer = Trainer(config, resume=run_dir, echo=False)
        self.addCleanup(trainer.close)
        self.assertEqual(trainer.learner.gradient_steps, config.optimizer.updates_per_iteration)
        self.assertEqual(trainer.learner.gradient_steps, saved["gradient_steps"])
        self.assertEqual(trainer.learner.last_epsilon, saved["last_epsilon"])
        self.assertEqual(trainer.learner.inference_hint(), {"mode": "q", "simulations": 0})
        self.assertIsInstance(trainer.learner.make_agent(), QAgent)
        for key, value in trainer.learner.target_network.state_dict().items():
            torch.testing.assert_close(value, saved["target_network"][key])
        self.assertGreater(len(trainer.replay), 0)
        config.run.iterations = 2
        trainer.run()
        self.assertEqual(trainer.learner.gradient_steps, 2 * config.optimizer.updates_per_iteration)

    def test_rejects_unsupported_n_step(self):
        from hasami_learn.algorithms.dqn import DqnParams
        from hasami_learn.config import ConfigError

        with self.assertRaisesRegex(ConfigError, "n_step"):
            DqnParams(n_step=2)

    def test_pool_selection_scores_and_truncation_are_explicit(self):
        from hasami.agents import RandomAgent
        from hasami_learn.inference import QAgent
        from hasami_learn.selfplay import Trajectory

        trainer = self.trainer()
        learner = trainer.learner
        trainer.config.selfplay.games_per_iteration = 4
        pool = trainer.opponents
        entries = [pool.current, pool.classical[0], pool.current, pool.classical[0]]
        colours = []
        outcomes = [(1, False), (-1, False), (0, False), (0, True)]

        def game(actor, opponent, player, encoder, **kwargs):
            index = len(colours)
            colours.append(player)
            if index % 2 == 0:
                self.assertIsInstance(opponent, QAgent)
                self.assertEqual(opponent.epsilon, learner.params.opponent_epsilon)
            reward, truncated = outcomes[index]
            rewards = [0, 0]
            rewards[player], rewards[1 - player] = reward, -reward
            winner = player if reward == 1 else 1 - player if reward == -1 else -2 if truncated else -1
            return Trajectory(rewards=tuple(rewards), winner=winner, truncated=truncated)

        with patch.object(pool, "sample", side_effect=entries), patch.object(pool, "agent_for", return_value=RandomAgent(0)) as agent_for, patch.object(pool, "record_result", wraps=pool.record_result) as record, patch("hasami_learn.algorithms.dqn.run_vs_opponent_game", side_effect=game):
            stats = learner.collect(1)
        self.assertEqual(agent_for.call_count, 2)
        self.assertEqual([call.args[1] for call in record.call_args_list], [1.0, 0.0, 0.5, 0.5])
        self.assertTrue(all(player in (0, 1) for player in colours))
        self.assertEqual(stats.truncated, 1)
        self.assertEqual(sum(stats.wins), 3)
        self.assertEqual(stats.extra["pool_truncated_neutral"], 1)

    def test_update_uses_huber_delta(self):
        import torch

        trainer = self.trainer(huber_delta=2.0)
        learner = trainer.learner
        learner.collect(1)
        trainer.config.optimizer.updates_per_iteration = 1
        with patch.object(learner, "_targets", side_effect=lambda batch: torch.full_like(batch["reward"], 10.0)), patch.object(learner.network, "forward", side_effect=lambda planes: (torch.zeros(len(planes), 6561, requires_grad=True), None)), patch.object(learner, "clip_and_step", side_effect=lambda loss: float(loss.detach())):
            stats = learner.update(1)
        self.assertAlmostEqual(stats.loss, 18.0)
