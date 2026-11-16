"""Record datasets and supervised learner integration tests."""

from __future__ import annotations

import contextlib
import io
import json
import math
import time
import unittest

from hasami import create_game
from hasami.agents import RandomAgent, SearchLimits
from hasami.records import load_record, record_dict, save_record
from tests.learn._common import LearnTestCase, requires_torch
from tests.test_rules import setup


@requires_torch
class SupervisedTests(LearnTestCase):
    def record(self, decisive=False):
        position = setup([38, 67, 72, 73, 74, 75], [39, 0, 1, 2, 3]) if decisive else None
        game = create_game("python", position=position)
        self.addCleanup(game.close)
        moves = [72 * 81 + 63, 0 * 81 + 9, 67 * 81 + 40] if decisive else [72 * 81 + 63, 0 * 81 + 9, 63 * 81 + 72]
        for action in moves:
            game.play(action)
        path = self.tmp / ("win.json" if decisive else "ongoing.json")
        save_record(game, path)
        loaded = load_record(path)
        self.addCleanup(loaded.close)
        return path, loaded

    def test_record_samples_masks_values_and_mirrors(self):
        import numpy as np
        from hasami_learn.datasets import iter_record_samples
        from hasami_learn.encoding import PlaneEncoder

        encoder = PlaneEncoder()
        for decisive in (False, True):
            _, game = self.record(decisive)
            before = record_dict(game)
            samples = list(iter_record_samples(game, encoder))
            self.assertEqual(len(samples), 3)
            self.assertEqual([s[3] for s in samples], [1, -1, 1] if decisive else [0, 0, 0])
            if decisive:
                self.assertEqual(game.outcome.reason, "five_captures")
            replay = create_game("python", game.rules, game.initial)
            self.addCleanup(replay.close)
            for planes, mask, action, _ in samples:
                np.testing.assert_array_equal(planes, encoder.encode(replay))
                np.testing.assert_array_equal(mask, encoder.legal_mask(replay))
                self.assertIs(type(action), int)
                self.assertIn(action, replay.legal_actions())
                replay.play(action)
            mirrored = list(iter_record_samples(game, encoder, augment=True))
            self.assertEqual(len(mirrored), 6)
            for original, flipped in zip(mirrored[::2], mirrored[1::2]):
                np.testing.assert_array_equal(flipped[0], encoder.flip_horizontal(original[0]))
                np.testing.assert_array_equal(flipped[1], encoder.flip_horizontal_policy(original[1]))
                self.assertEqual(flipped[2], encoder.flip_horizontal_action(original[2]))
                self.assertEqual(flipped[1][flipped[2]], 1)
                self.assertEqual(flipped[3], original[3])
            self.assertEqual(record_dict(game), before)

    def test_repetition_history_and_rule_anchors(self):
        from hasami_learn.datasets import iter_record_samples
        from hasami_learn.encoding import PlaneEncoder

        game = create_game("python")
        self.addCleanup(game.close)
        self.assertEqual(len(game.legal_actions()), 63)
        count = 0
        for action in game.legal_actions():
            game.play(action)
            count += len(game.legal_actions())
            game.undo()
        self.assertEqual(count, 3717)
        cycle = [72 * 81 + 63, 0 * 81 + 9, 63 * 81 + 72, 9 * 81 + 0]
        for action in cycle * 3:
            game.play(action)
        samples = list(iter_record_samples(game, PlaneEncoder()))
        self.assertEqual(game.outcome.reason, "repetition")
        self.assertEqual(len(samples), 12)
        self.assertEqual([float(s[0][7, 0, 0]) for s in samples], [0.25] * 4 + [0.5] * 4 + [0.75] * 4)
        self.assertTrue(all(s[3] == 0 for s in samples))

    def test_dataset_roundtrip_and_reject_illegal_labels(self):
        import numpy as np
        from hasami_learn.datasets import build_dataset, load_dataset, save_dataset
        from hasami_learn.encoding import PlaneEncoder

        path, _ = self.record(True)
        data = build_dataset([path], PlaneEncoder())
        target = self.tmp / "dataset.npz"
        save_dataset(target, data)
        loaded = load_dataset(target)
        self.assertEqual(loaded["meta"], data["meta"])
        for key, shape in {"planes": (6, 9, 9, 9), "mask": (6, 6561), "policy": (6, 6561), "value": (6,)}.items():
            self.assertEqual(loaded[key].shape, shape)
            self.assertEqual(loaded[key].dtype, np.float32)
            np.testing.assert_array_equal(loaded[key], data[key])
        np.testing.assert_array_equal(data["policy"].sum(1), np.ones(6))
        data["policy"][0] = 0
        data["policy"][0, 0] = 1
        with self.assertRaises(ValueError):
            save_dataset(self.tmp / "bad.npz", data)
        empty = build_dataset([], PlaneEncoder())
        save_dataset(self.tmp / "empty.npz", empty)
        self.assertEqual(load_dataset(self.tmp / "empty.npz")["planes"].shape, (0, 9, 9, 9))

    def test_generate_records_and_dataset_cli(self):
        from hasami_learn.datasets import generate_games, load_dataset, main

        paths = generate_games(self.tmp / "games", 2, generator="random", max_plies=12, seed=7)
        again = generate_games(self.tmp / "again", 2, generator="random", max_plies=12, seed=7)
        self.assertEqual(len(paths), 2)
        for path, repeat in zip(paths, again):
            game = load_record(path)
            self.addCleanup(game.close)
            self.assertEqual(len(game.moves), 12)
            self.assertFalse(game.outcome.terminal)
            self.assertEqual(path.read_text(), repeat.read_text())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["--records-dir", str(self.tmp / "games"), "--out", str(self.tmp / "cli.npz"), "--no-augment"])
        self.assertEqual(result, 0)
        self.assertIn("24", output.getvalue())
        self.assertEqual(load_dataset(self.tmp / "cli.npz")["value"].tolist(), [0] * 24)

    def test_train_checkpoint_and_legal_game(self):
        from hasami_learn.inference import create_agent
        from hasami_learn.trainer import train_from_config

        start = time.perf_counter()
        checkpoint = train_from_config(self.tiny("supervised", generate_games=2, generator="random"), echo=False)
        self.assertLess(time.perf_counter() - start, 60)
        agent = create_agent(checkpoint.parent.parent, device="cpu")
        rows = [json.loads(line) for line in (checkpoint.parent.parent / "metrics.jsonl").read_text().splitlines()]
        update = next(row for row in rows if row["kind"] == "update")
        self.assertTrue(math.isfinite(update["loss"]))
        self.assertGreater(update["steps"], 0)
        self.assertTrue(math.isfinite(update["extra"]["validation_value_mse"]))
        game = create_game("python")
        self.addCleanup(game.close)
        players = [agent, RandomAgent(1)]
        for _ in range(60):
            if game.outcome.terminal:
                break
            snapshot = game.clone()
            try:
                decision = players[game.position.turn].choose(snapshot, SearchLimits(1, 100, 10))
            finally:
                snapshot.close()
            self.assertIs(type(decision.action), int)
            self.assertIn(decision.action, game.legal_actions())
            game.play(decision.action)

    def test_collect_once_deterministic_split_and_resume(self):
        import numpy as np
        from hasami_learn.datasets import build_dataset, save_dataset
        from hasami_learn.encoding import PlaneEncoder
        from hasami_learn.trainer import Trainer

        path, _ = self.record(True)
        dataset = save_dataset(self.tmp / "train.npz", build_dataset([path], PlaneEncoder()))
        config = self.tiny("supervised", dataset_path=str(dataset), validation_fraction=0.34)
        config.replay.min_size = 1
        trainer = Trainer(config, echo=False)
        self.addCleanup(trainer.close)
        learner = trainer.learner
        stats = learner.collect(1)
        self.assertEqual(stats.extra["training_samples"], 4)
        self.assertEqual(stats.extra["validation_samples"], 2)
        before = {key: array.copy() for key, array in trainer.replay.all().items()}
        self.assertEqual(learner.collect(2).games, 0)
        for key, array in trainer.replay.all().items():
            np.testing.assert_array_equal(array, before[key])
        held_out = learner._validation_indices.copy()
        self.assertEqual(held_out[0] // 2, held_out[1] // 2)
        self.assertTrue(set(held_out).isdisjoint(learner._train_indices))
        config.run.run_id = "same-seed"
        other = Trainer(config, echo=False)
        self.addCleanup(other.close)
        other.learner.collect(1)
        np.testing.assert_array_equal(other.learner._validation_indices, held_out)
        trainer.iteration = 1
        checkpoint = trainer.checkpoint(1)
        resumed = Trainer(config, resume=checkpoint, echo=False)
        self.addCleanup(resumed.close)
        self.assertEqual(resumed.learner.collect(2).games, 0)
        np.testing.assert_array_equal(resumed.learner._validation_indices, held_out)
        for key, array in resumed.replay.all().items():
            np.testing.assert_array_equal(array, before[key])
        self.assertGreater(resumed.learner.update(2).steps, 0)

    def test_legal_smoothing_loss_validation_and_actor(self):
        import numpy as np
        import torch
        from hasami_learn.datasets import build_dataset, save_dataset
        from hasami_learn.encoding import PlaneEncoder
        from hasami_learn.trainer import Trainer

        game = create_game("python")
        self.addCleanup(game.close)
        action = game.legal_actions()[0]
        game.play(action)
        path = self.tmp / "one.json"
        save_record(game, path)
        data = build_dataset([path], PlaneEncoder(), augment=False)
        dataset = save_dataset(self.tmp / "single.npz", data)
        config = self.tiny("supervised", dataset_path=str(dataset), validation_fraction=0, label_smoothing=0.2)
        config.replay.min_size = 1
        config.optimizer.updates_per_iteration = 1
        trainer = Trainer(config, echo=False)
        self.addCleanup(trainer.close)
        learner = trainer.learner
        learner.collect(1)
        learner._validation = {key: data[key] for key in ("planes", "mask", "policy", "value")}
        with torch.no_grad():
            for parameter in trainer.network.parameters():
                parameter.zero_()
            trainer.network.out.policy[-1].bias[0] = 1000  # Illegal even though it has the highest logit.
            trainer.network.out.policy[-1].bias[action] = 2
        metrics = learner._validation_metrics()
        self.assertEqual(metrics["validation_policy_accuracy"], 1)
        self.assertEqual(metrics["validation_value_mse"], 0)
        expected = math.log(math.exp(2) + 62) - 2 * (0.8 + 0.2 / 63)
        stats = learner.update(1)
        self.assertAlmostEqual(stats.policy_loss, expected, places=5)
        self.assertEqual(stats.value_loss, 0)
        self.assertTrue(all(torch.isfinite(p).all() for p in trainer.network.parameters()))
        fresh = create_game("python")
        self.addCleanup(fresh.close)
        planes, mask = learner.encoder.encode(fresh), learner.encoder.legal_mask(fresh)
        chosen, _, _ = learner.actor_fn()(fresh, planes, mask, 0)
        self.assertIs(type(chosen), int)
        self.assertIn(chosen, fresh.legal_actions())
        self.assertEqual(learner.inference_hint(), {"mode": "policy", "simulations": 0})
        with self.assertRaises(ValueError):
            learner.actor_fn()(fresh, planes, np.zeros_like(mask), 0)

    def test_ingest_trajectories_uses_moves_and_explicit_truncation(self):
        import numpy as np
        from hasami_learn.selfplay import Trajectory
        from hasami_learn.trainer import Trainer

        config = self.tiny("supervised")
        config.selfplay.symmetry_augmentation = True
        config.selfplay.truncation_value = 0.25
        trainer = Trainer(config, echo=False)
        self.addCleanup(trainer.close)
        game = create_game("python")
        self.addCleanup(game.close)
        trajectory = Trajectory(rewards=(1, -1))
        for _ in range(2):
            action = game.legal_actions()[0]
            trajectory.append(trainer.encoder.encode(game), trainer.encoder.legal_mask(game), action,
                              game.position.turn, None, {})
            game.play(action)
        learner = trainer.learner
        self.assertEqual(learner.ingest_trajectories([trajectory]), 4)
        samples = trainer.replay.all()
        np.testing.assert_array_equal(samples["value"], [1, 1, -1, -1])
        np.testing.assert_array_equal(samples["policy"].sum(1), np.ones(4))
        self.assertFalse((samples["policy"][samples["mask"] == 0] != 0).any())
        trajectory.truncated = True
        trainer.replay.clear()
        learner.ingest_trajectories([trajectory])
        np.testing.assert_array_equal(trainer.replay.all()["value"], [0.25] * 4)
        trajectory.actions[0] = 0
        with self.assertRaises(ValueError):
            learner.ingest_trajectories([trajectory])

    def test_empty_training_data_and_invalid_parameters(self):
        from hasami_learn.algorithms.supervised import SupervisedParams
        from hasami_learn.datasets import build_dataset, generate_games, save_dataset
        from hasami_learn.encoding import PlaneEncoder
        from hasami_learn.trainer import Trainer

        for params in ({"validation_fraction": 1}, {"label_smoothing": -0.1}, {"generate_games": -1}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                SupervisedParams(**params)
        for params in ({"games": -1}, {"games": 1, "max_plies": 0}, {"games": 1, "random_opening_plies": -1}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                generate_games(self.tmp / "invalid", **params)
        dataset = save_dataset(self.tmp / "empty.npz", build_dataset([], PlaneEncoder()))
        trainer = Trainer(self.tiny("supervised", dataset_path=str(dataset)), echo=False)
        self.addCleanup(trainer.close)
        self.assertEqual(trainer.learner.update(1).steps, 0)
        with self.assertRaisesRegex(ValueError, "at least one sample"):
            trainer.learner.collect(1)


if __name__ == "__main__":
    unittest.main()
