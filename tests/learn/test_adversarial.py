"""League collection, exploiter isolation, and checkpoint regression tests."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from unittest.mock import patch

from tests.learn._common import LearnTestCase, requires_torch


@requires_torch
class AdversarialTests(LearnTestCase):
    def trainer(self, **params):
        from hasami_learn.trainer import Trainer

        config = self.tiny("adversarial", **params)
        config.opponents.strategy = "pfsp"
        config.opponents.classical_weight = 0.5
        trainer = Trainer(config, echo=False)
        self.addCleanup(trainer.close)
        return trainer

    def test_registration_and_context_delegation(self):
        from hasami_learn.algorithms.adversarial import AdversarialLearner
        from hasami_learn.registry import ALGORITHMS

        self.assertIs(ALGORITHMS.get("adversarial"), AdversarialLearner)
        trainer = self.trainer(inner_params={"entropy_coef": 0.02})
        learner = trainer.learner
        self.assertEqual(learner.required_head, learner.inner.required_head)
        self.assertEqual(learner.replay_kind, learner.inner.replay_kind)
        self.assertEqual(learner.inner.config.algorithm.name, "reinforce")
        self.assertEqual(trainer.config.algorithm.name, "adversarial")
        self.assertEqual(learner.inner.params.entropy_coef, 0.02)
        for name in ("network", "encoder", "replay", "opponents", "device", "rng", "metrics"):
            self.assertIs(getattr(learner, name), getattr(learner.inner, name))
        self.assertIs(learner.optimizer, learner.inner.optimizer)

    def test_rejects_incompatible_head(self):
        from hasami_learn.algorithms.adversarial import AdversarialLearner
        from hasami_learn.algorithms.reinforce import ReinforceLearner
        from hasami_learn.config import ConfigError

        trainer = self.trainer()
        with patch.object(ReinforceLearner, "required_head", "q"):
            with self.assertRaisesRegex(ConfigError, "head"):
                AdversarialLearner(trainer.learner.ctx)

    def test_collect_records_colour_correct_results_and_truncation(self):
        from hasami_learn.selfplay import Trajectory

        trainer = self.trainer(learner_colour="white")
        trainer.config.selfplay.games_per_iteration = 4
        entry = trainer.opponents.classical[0]
        trajectories = [
            Trajectory(winner=1, rewards=(-1.0, 1.0), plies=10),
            Trajectory(winner=0, rewards=(1.0, -1.0), plies=11),
            Trajectory(winner=-1, plies=12),
            Trajectory(winner=-2, truncated=True, plies=40),
        ]
        with patch.object(trainer.opponents, "sample", return_value=entry), patch(
            "hasami_learn.algorithms.adversarial.run_vs_opponent_game",
            side_effect=trajectories,
        ) as run, patch.object(trainer.learner.inner, "ingest_trajectories", return_value=7) as ingest:
            stats = trainer.learner.collect(1)
        self.assertEqual((entry.wins, entry.losses, entry.draws), (1, 1, 1))
        self.assertEqual(stats.wins, (1, 1, 1))
        self.assertEqual(stats.truncated, 1)
        self.assertEqual(stats.extra["learner_wins"], [1, 1, 1])
        self.assertEqual(stats.extra["samples"], 7)
        self.assertEqual(stats.extra["pool_truncated"], {entry.entry_id: 1})
        self.assertEqual(ingest.call_args.args[0], trajectories)
        self.assertTrue(all(call.args[2] == 1 for call in run.call_args_list))

    def test_real_collect_and_update_have_legal_samples_and_finite_loss(self):
        trainer = self.trainer()
        learner = trainer.learner
        stats = learner.collect(1)
        self.assertEqual(stats.games, 2)
        self.assertGreater(stats.extra["samples"], 0)
        for action, mask in zip(learner.inner._actions, learner.inner._masks):
            self.assertIs(type(action), int)
            self.assertEqual(mask[action], 1)
        update = learner.update(1)
        self.assertGreater(update.steps, 0)
        self.assertTrue(math.isfinite(update.loss))
        self.assertEqual(learner.inner._actions, [])

    def test_truncation_targets_are_explicit_and_not_pool_draws(self):
        trainer = self.trainer(learner_colour="black")
        trainer.config.selfplay.max_plies = 1
        trainer.learner.inner.config.selfplay.max_plies = 1
        trainer.learner.inner.config.selfplay.truncation_value = 0.25
        stats = trainer.learner.collect(1)
        self.assertEqual(stats.truncated, 2)
        self.assertEqual(stats.wins, (0, 0, 0))
        self.assertEqual(sum(e.games for e in trainer.opponents.all_entries()), 0)
        self.assertEqual(trainer.learner.inner._values, [0.25, 0.25])

    def test_pfsp_prefers_opponent_with_low_learner_win_rate(self):
        import numpy as np
        from hasami_learn.opponents import PoolEntry

        trainer = self.trainer()
        pool = trainer.opponents
        pool.cfg.classical_weight = 0.0
        pool.rng = np.random.default_rng(123)
        hard = PoolEntry("hard", "checkpoint", "unused-hard", losses=20)
        easy = PoolEntry("easy", "checkpoint", "unused-easy", wins=20)
        pool.entries = [easy, hard]
        hard_draws = sum(pool.sample() is hard for _ in range(200))
        self.assertGreater(hard_draws, 165)
        self.assertLess(hard_draws, 200)

    def test_exploiter_leaves_main_weights_optimizer_and_rng_untouched(self):
        import torch
        from hasami_learn.manifest import load_checkpoint

        trainer = self.trainer(exploiter_every=1, exploiter_iterations=1, exploiter_games=1)
        learner = trainer.learner
        learner.collect(1)
        learner.inner.update(1)
        weights = copy.deepcopy(trainer.network.state_dict())
        state = copy.deepcopy(learner.inner.state_dict())
        rng = copy.deepcopy(trainer.rng.bit_generator.state)
        mode = trainer.network.training
        with patch.object(learner.inner, "update", wraps=learner.inner.update):
            stats = learner.update(1)
        self.assertEqual(stats.extra["exploiter_games"], 1)
        self.assertIn("exploiter_win_rate", stats.extra)
        self.assertEqual(trainer.rng.bit_generator.state, rng)
        self.assertEqual(trainer.network.training, mode)
        for key, value in weights.items():
            self.assertTrue(torch.equal(value, trainer.network.state_dict()[key]))
        self.assert_nested_equal(state, learner.inner.state_dict())
        entry = trainer.opponents.entries[0]
        self.assertIn("exploiter", entry.entry_id)
        loaded = load_checkpoint(entry.ref, "cpu")
        self.assertEqual(loaded.manifest.algorithm, "reinforce")
        self.assertTrue(any(not torch.equal(weights[k], v) for k, v in loaded.network.state_dict().items()))

    def assert_nested_equal(self, left, right):
        import torch

        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_trainer_snapshot_schedule_avoids_duplicate_main_snapshots(self):
        trainer = self.trainer(snapshot_every=1)
        trainer.config.opponents.snapshot_every = 1
        trainer.run()
        self.assertEqual(len(trainer.opponents.entries), 1)
        self.assertEqual(trainer.learner.extra_state()["snapshots"], 0)

    def test_tiny_checkpoint_pool_resume_and_legal_game(self):
        from scripts.benchmark import python_perft
        from hasami import create_game
        from hasami.agents import RandomAgent, SearchLimits
        from hasami_learn.config import TrainingConfig
        from hasami_learn.inference import create_agent
        from hasami_learn.trainer import Trainer

        config = TrainingConfig.load("configs/tiny/adversarial.json")
        config.run.output_dir = str(self.tmp)
        trainer = Trainer(config, echo=False)
        self.addCleanup(trainer.close)
        checkpoint = trainer.run()
        self.assertTrue((checkpoint / "weights.pt").is_file())
        entries = trainer.opponents.entries
        self.assertEqual(len(entries), 2)
        self.assertEqual(len({e.ref for e in entries}), 2)
        self.assertTrue(all(Path(e.ref).is_relative_to(trainer.run_dir / "pool") for e in entries))
        self.assertTrue(all((Path(e.ref) / "manifest.json").is_file() for e in entries))
        self.assertGreater(sum(row["wins"] + row["losses"] + row["draws"] for row in trainer.opponents.stats()), 0)
        agent = create_agent(str(trainer.run_dir), device="cpu")
        opponent = RandomAgent(7)
        game = create_game("python")
        self.addCleanup(game.close)
        self.assertEqual(len(game.legal_actions()), 63)
        self.assertEqual(python_perft(game, 2), 3717)
        limits = SearchLimits(1, 100, 10)
        for _ in range(300):
            if game.outcome.terminal:
                break
            snapshot = game.clone()
            try:
                decision = (agent if game.position.turn == 0 else opponent).choose(snapshot, limits)
            finally:
                snapshot.close()
            self.assertIs(type(decision.action), int)
            self.assertIn(decision.action, game.legal_actions())
            game.play(decision.action)
        self.assertTrue(game.outcome.terminal)
        resumed = Trainer(config, resume=trainer.run_dir, echo=False)
        self.addCleanup(resumed.close)
        self.assertEqual(resumed.opponents.state_dict(), trainer.opponents.state_dict())
        self.assert_nested_equal(resumed.learner.state_dict(), trainer.learner.state_dict())

    def test_wrappers_can_nest(self):
        trainer = self.trainer(inner_algorithm="adversarial", snapshot_every=1,
                               inner_params={"snapshot_every": 1})
        stats = trainer.learner.collect(1)
        self.assertGreater(stats.extra["samples"], 0)
        self.assertTrue(math.isfinite(trainer.learner.update(1).loss))
        self.assertEqual(len(trainer.opponents.entries), 1)
        state = copy.deepcopy(trainer.learner.state_dict())
        trainer.learner._games = 0
        trainer.learner.inner._games = 17
        trainer.learner.load_state_dict(state)
        self.assert_nested_equal(state, trainer.learner.state_dict())

    def test_missing_hooks_and_replay_are_rejected(self):
        from hasami_learn.algorithms.adversarial import AdversarialLearner
        from hasami_learn.algorithms.base import Learner
        from hasami_learn.algorithms.reinforce import ReinforceLearner
        from hasami_learn.config import ConfigError

        trainer = self.trainer()
        with patch.object(ReinforceLearner, "actor_fn", Learner.actor_fn):
            with self.assertRaisesRegex(ConfigError, "actor_fn"):
                AdversarialLearner(trainer.learner.ctx)
        with patch.object(ReinforceLearner, "replay_kind", "pv"):
            with self.assertRaisesRegex(ConfigError, "replay buffer"):
                AdversarialLearner(trainer.learner.ctx)

    def test_invalid_params_fail_before_training(self):
        from hasami_learn.algorithms.adversarial import AdversarialParams
        from hasami_learn.config import ConfigError

        for params in ({"snapshot_every": -1}, {"exploiter_every": True},
                       {"exploiter_iterations": 0}, {"exploiter_games": 0},
                       {"main_games_fraction": 0}, {"main_games_fraction": float("nan")},
                       {"learner_colour": "purple"}, {"record_pool_stats": 1}):
            with self.subTest(params=params), self.assertRaises(ConfigError):
                AdversarialParams(**params)

    def test_main_fraction_keeps_full_game_budget_and_optional_stats(self):
        trainer = self.trainer(main_games_fraction=0.5, record_pool_stats=False)
        stats = trainer.learner.collect(1)
        self.assertEqual(stats.games, 2)
        self.assertEqual(stats.extra["training_games"], 1)
        self.assertGreater(stats.extra["samples"], 0)
        self.assertNotIn("pool", stats.extra)
        self.assertNotIn("pool", trainer.learner.update(1).extra)
