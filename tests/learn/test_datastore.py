import unittest
"""Sparse self-play persistence, recovery, caps, and reconstructed datasets."""

import json
from unittest.mock import patch

from tests.learn._common import LearnTestCase, requires_torch


@requires_torch
class DatastoreTests(LearnTestCase):
    def trajectory(self):
        from hasami_learn.encoding import PlaneEncoder
        from hasami_learn.selfplay import run_selfplay_game

        def actor(game, planes, mask, ply):
            policy = mask / mask.sum()
            return game.legal_actions()[0], policy, {"root_value": 0.123, "simulations": 2}

        return run_selfplay_game(actor, PlaneEncoder(), max_plies=4)

    def store(self, **kwargs):
        from hasami_learn.datastore import SelfplayStore

        return SelfplayStore(self.tmp / "store", **kwargs)

    def test_sparse_roundtrip_and_dataset(self):
        import numpy as np
        from hasami_learn.encoding import PlaneEncoder

        trajectory = self.trajectory()
        store = self.store()
        path = store.add(trajectory, 7)
        with np.load(path, allow_pickle=False) as saved:
            self.assertNotIn("planes", saved.files)
            self.assertEqual(saved["moves"].dtype, np.int16)
            self.assertEqual(saved["policy_index"].dtype, np.int16)
            self.assertEqual(saved["policy_prob"].dtype, np.float16)
            self.assertEqual(saved["root_value"].dtype, np.float16)
        game = next(store.iter_games())
        np.testing.assert_array_equal(game["moves"], trajectory.actions)
        np.testing.assert_allclose(game["policy"], trajectory.policies, atol=2e-4)
        self.assertEqual(game["reason"], trajectory.reason)
        self.assertEqual(game["truncated"], trajectory.truncated)
        self.assertEqual(store.stats(), {"games": 1, "plies": 4, "bytes": path.stat().st_size})
        data = store.to_dataset(PlaneEncoder())
        self.assertEqual(data["planes"].shape, (4, 9, 9, 9))
        self.assertEqual(data["policy"].shape, (4, 6561))
        self.assertEqual(data["mask"].shape, (4, 6561))
        self.assertEqual(data["value"].shape, (4,))
        np.testing.assert_array_equal(data["planes"], trajectory.planes)
        np.testing.assert_array_equal(data["mask"], trajectory.masks)
        np.testing.assert_array_equal(data["policy"][data["mask"] == 0], 0)
        np.testing.assert_allclose(data["policy"].sum(1), 1, atol=1e-6)

    def test_count_eviction_and_append_only_index(self):
        store = self.store(max_games=2)
        trajectory = self.trajectory()
        paths = [store.add(trajectory, i) for i in range(3)]
        self.assertFalse(paths[0].exists())
        self.assertTrue(all(p.exists() for p in paths[1:]))
        rows = [json.loads(line) for line in store.index_path.read_text().splitlines()]
        self.assertEqual([r["iteration"] for r in rows], [0, 1, 2])
        self.assertEqual([g["iteration"] for g in store.iter_games()], [2, 1])
        self.assertEqual([g["iteration"] for g in store.iter_games(1, False)], [1])
        self.assertEqual(list(store.iter_games(0)), [])
        self.assertEqual(self.store(max_games=2).stats(), store.stats())

    def test_byte_eviction_and_oversized_game(self):
        store = self.store()
        trajectory = self.trajectory()
        first = store.add(trajectory, 0)
        store.max_bytes = first.stat().st_size + 100
        store.add(trajectory, 1)
        self.assertFalse(first.exists())
        self.assertLessEqual(store.stats()["bytes"], store.max_bytes)
        store.max_bytes = 1
        store.add(trajectory, 2)
        self.assertEqual(store.stats()["games"], 0)
        self.assertEqual(list(store.games_dir.glob("*.npz")), [])

    def test_partial_file_and_index_recovery(self):
        from hasami_learn.datastore import SelfplayStore

        store = self.store()
        trajectory = self.trajectory()
        first = store.add(trajectory, 1)
        with patch("hasami_learn.datastore.os.replace", side_effect=OSError("partial write")):
            with self.assertRaises(OSError):
                store.add(trajectory, 2)
        self.assertTrue(first.exists())
        self.assertEqual(store.stats()["games"], 1)
        with patch.object(store, "_append_index", side_effect=OSError("index interrupted")):
            with self.assertRaises(OSError):
                store.add(trajectory, 3)
        with store.index_path.open("ab") as output:
            output.write(b'{"file": "games/')
        partial = store.games_dir / ".selfplay-partial.tmp"
        partial.write_bytes(b"partial npz")
        recovered = SelfplayStore(store.root)
        self.assertFalse(partial.exists())
        recovered.add(trajectory, 4)
        self.assertEqual([g["iteration"] for g in recovered.iter_games()], [4, 3, 1])
        again = SelfplayStore(store.root)
        self.assertEqual(again.stats(), recovered.stats())

    def test_recovery_reapplies_caps_and_does_not_reuse_evicted_names(self):
        store = self.store()
        trajectory = self.trajectory()
        paths = [store.add(trajectory, 1) for _ in range(3)]
        recovered = self.store(max_games=1)
        self.assertEqual(recovered.stats()["games"], 1)
        self.assertTrue(paths[-1].exists())
        latest = recovered.add(trajectory, 1)
        self.assertNotIn(latest, paths)

    def test_rejects_incomplete_history_and_illegal_policy_on_replay(self):
        from hasami_learn.encoding import PlaneEncoder

        store = self.store()
        trajectory = self.trajectory()
        trajectory.plies += 1
        with self.assertRaisesRegex(ValueError, "full move history"):
            store.add(trajectory, 1)
        self.assertEqual(store.stats()["games"], 0)
        trajectory.plies -= 1
        trajectory.policies[0][:] = 0
        trajectory.policies[0][0] = 1
        store.add(trajectory, 1)
        with self.assertRaisesRegex(ValueError, "policy"):
            store.to_dataset(PlaneEncoder())

    def test_value_targets_and_empty_shapes(self):
        import numpy as np
        from hasami_learn.encoding import PlaneEncoder

        store = self.store()
        self.assertEqual(store.to_dataset(PlaneEncoder())["planes"].shape, (0, 9, 9, 9))
        trajectory = self.trajectory()
        store.truncation_value = 0.4
        store.add(trajectory, 1)
        np.testing.assert_allclose(store.to_dataset(PlaneEncoder())["value"], 0.4)
        trajectory.truncated = False
        trajectory.winner = -1
        trajectory.reason = "repetition"
        trajectory.draw_value = -0.2
        store.add(trajectory, 2)
        np.testing.assert_allclose(next(store.iter_games())["value"], -0.2)

    def test_enabled_learner_ingest_and_opponent_histories(self):
        import numpy as np
        from hasami.agents import RandomAgent
        from hasami_learn.selfplay import run_vs_opponent_game
        from hasami_learn.trainer import Trainer

        cfg = self.tiny("alphazero", simulations=2, batch_games=2)
        cfg.datastore.enabled = True
        cfg.selfplay.max_plies = 4
        trainer = Trainer(cfg, echo=False)
        self.addCleanup(trainer.close)
        learner = trainer.learner
        stats = learner.collect(5)
        self.assertEqual(stats.extra["datastore"]["games"], 2)
        self.assertEqual(learner.store.root, learner.run_dir / "selfplay")
        for player in (0, 1):
            trajectory = run_vs_opponent_game(learner.actor_fn(), RandomAgent(1), player,
                                              learner.encoder, max_plies=4)
            self.assertEqual(learner.ingest_trajectories([trajectory]), 2)
            latest = next(learner.store.iter_games())
            self.assertEqual(len(latest["moves"]), 4)
        data = learner.store.to_dataset(learner.encoder)
        self.assertEqual(len(data["value"]), 12)
        np.testing.assert_array_equal(data["policy"][data["mask"] == 0], 0)


class ZeroDecisionTrajectoryTests(unittest.TestCase):
    """Review finding on data ingestion: opponent games without learner decisions must not break ingestion."""

    def test_add_skips_trajectories_without_decisions(self):
        import tempfile
        from pathlib import Path

        try:
            import numpy  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("numpy not installed")
        from hasami.agents import RandomAgent, SearchLimits
        from hasami_learn.datastore import SelfplayStore
        from hasami_learn.encoding import PlaneEncoder
        from hasami_learn.selfplay import run_vs_opponent_game

        def never_called(game, planes, mask, ply):
            raise AssertionError("the learner must not move in a one-ply game where it plays second")

        trajectory = run_vs_opponent_game(never_called, RandomAgent(0), 1, PlaneEncoder(), backend="python", max_plies=1, limits=SearchLimits(1, 50, 10))
        self.assertEqual(len(trajectory), 0)
        self.assertTrue(trajectory.truncated)
        with tempfile.TemporaryDirectory() as directory:
            store = SelfplayStore(Path(directory), max_games=10, max_bytes=10 ** 6)
            self.assertIsNone(store.add(trajectory, 1))
            self.assertEqual(store.stats()["games"], 0)
