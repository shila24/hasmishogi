"""Spawned self-play legality, deterministic ordering, and resource cleanup."""

import multiprocessing
import pickle
from unittest.mock import patch

from hasami import create_game
from tests.learn._common import LearnTestCase, requires_torch


@requires_torch
class ParallelTests(LearnTestCase):
    def trainer(self, workers=2, games=3):
        from hasami_learn.trainer import Trainer

        cfg = self.tiny("alphazero", simulations=2, batch_games=2)
        cfg.selfplay.workers = workers
        cfg.selfplay.games_per_iteration = games
        cfg.selfplay.max_plies = 3
        trainer = Trainer(cfg, echo=False)
        self.addCleanup(trainer.close)
        return trainer

    def assert_legal(self, trajectories):
        import numpy as np

        for trajectory in trajectories:
            game = create_game("python")
            try:
                self.assertEqual(len(game.legal_actions()), 63)
                nodes = 0
                for opening in game.legal_actions():
                    game.play(opening)
                    nodes += len(game.legal_actions())
                    game.undo()
                self.assertEqual(nodes, 3717)
                self.assertEqual(trajectory.plies, len(trajectory))
                self.assertTrue(trajectory.truncated)
                for i, action in enumerate(trajectory.actions):
                    self.assertIs(type(action), int)
                    self.assertIn(action, game.legal_actions())
                    self.assertEqual(trajectory.players[i], game.position.turn)
                    policy, mask = trajectory.policies[i], trajectory.masks[i]
                    self.assertAlmostEqual(float(policy.sum()), 1)
                    np.testing.assert_array_equal(policy[mask == 0], 0)
                    game.play(action)
            finally:
                game.close()

    def test_two_workers_order_seeds_and_cleanup(self):
        import numpy as np
        from hasami_learn.parallel import SelfplayWorkers, _run_worker

        learner = self.trainer().learner
        job = learner._selfplay_job(2)
        self.assertEqual(job["seeds"], [2000, 2001])
        self.assertEqual(job["games_per_worker"], [2, 1])
        before = {p.pid for p in multiprocessing.active_children()}
        with SelfplayWorkers(2, "cpu") as workers:
            first = workers.run(job)
            second = workers.run(job)
        workers.close()
        self.assertEqual({p.pid for p in multiprocessing.active_children()}, before)
        self.assertEqual(len(first), 3)
        self.assert_legal(pickle.loads(pickle.dumps(first)))
        expected = []
        for i in range(2):
            expected.extend(_run_worker(dict(job, seed=job["seeds"][i],
                                             games=job["games_per_worker"][i], device_spec="cpu")))
        for a, b, c in zip(first, second, expected):
            self.assertEqual(a.actions, b.actions)
            self.assertEqual(a.actions, c.actions)
            np.testing.assert_array_equal(a.policies, b.policies)
            self.assertEqual(a.draw_value, job["draw_value"])
        with self.assertRaises(RuntimeError):
            workers.run(job)

    def test_worker_error_closes_pool(self):
        from hasami_learn.parallel import SelfplayWorkers

        job = self.trainer().learner._selfplay_job(1)
        job["network_name"] = "missing-network"
        before = {p.pid for p in multiprocessing.active_children()}
        workers = SelfplayWorkers(2, "cpu")
        with self.assertRaises(KeyError):
            workers.run(job)
        self.assertEqual({p.pid for p in multiprocessing.active_children()}, before)
        workers.close()

    def test_abrupt_worker_exit_raises_and_reaps_pool(self):
        from threading import Timer
        from hasami_learn.parallel import SelfplayWorkers

        job = self.trainer().learner._selfplay_job(1)
        job["max_plies"] = 1000
        before = {p.pid for p in multiprocessing.active_children()}
        workers = SelfplayWorkers(2, "cpu")
        self.addCleanup(workers.close)
        timer = Timer(0.2, workers._pool._pool[0].terminate)
        timer.start()
        try:
            with self.assertRaisesRegex(RuntimeError, "worker exited"):
                workers.run(job)
        finally:
            timer.cancel()
            timer.join()
        self.assertEqual({p.pid for p in multiprocessing.active_children()}, before)

    def test_zero_games_and_invalid_job_cleanup(self):
        from hasami_learn.parallel import SelfplayWorkers

        job = self.trainer(games=1).learner._selfplay_job(1)
        self.assertEqual(job["games_per_worker"], [1, 0])
        with SelfplayWorkers(2, "cpu") as workers:
            self.assertEqual(len(workers.run(job)), 1)
            job["games_per_worker"] = 0
            self.assertEqual(workers.run(job), [])
        with SelfplayWorkers(2, "cpu") as workers:
            job["seeds"] = [0]
            with self.assertRaises(ValueError):
                workers.run(job)
        for count in (0, -1, True):
            with self.assertRaises(ValueError):
                SelfplayWorkers(count, "cpu")

    def test_collect_and_single_worker_fallback(self):
        for count in (1, 2):
            with self.subTest(workers=count):
                learner = self.trainer(count).learner
                with patch.object(learner, "ingest_trajectories", wraps=learner.ingest_trajectories) as ingest:
                    stats = learner.collect(1)
                self.assertEqual((stats.games, stats.plies, stats.truncated), (3, 9, 3))
                trajectories = [t for call in ingest.call_args_list for t in call.args[0]]
                self.assert_legal(trajectories)
        learner = self.trainer(1).learner
        with patch("hasami_learn.algorithms.alphazero.SelfplayWorkers", side_effect=AssertionError):
            self.assertEqual(learner.collect(1).games, 3)
