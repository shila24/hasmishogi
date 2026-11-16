"""Vectorised training, native streaming, selection and checkpoint regressions."""
import glob
import importlib.util
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH, "torch is optional")
class FastTrainingTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        torch.set_num_threads(1)
        torch.manual_seed(5157)

    def native(self):
        from hasami_learn.nnue.loader import native_encoder
        try:
            return native_encoder()
        except RuntimeError:
            if os.environ.get("HASAMI_REQUIRE_NATIVE") == "1":
                self.fail("Native shogi library is required")
            self.skipTest("Native shogi library is unavailable")

    def positions(self, count=160):
        from shogi import create_game
        rng = random.Random(5157)
        result = []
        game = create_game("python")
        try:
            while len(result) < count:
                result.append(game.position)
                game.play(rng.choice(game.legal_actions()))
                if game.outcome.terminal:
                    game.close()
                    game = create_game("python")
        finally:
            game.close()
        return result

    def data(self, shards=4):
        from hasami_learn.nnue.data import Record, write_shard
        positions = self.positions(24)
        paths = []
        for shard in range(shards):
            path = self.root / f"shard-{shard}.hlnn1"
            rows = [Record(p, (i-12)*100 if i % 5 else 999999, 1, 30, 0,
                           i % 3-1, i, shard*100+i, flags=4 if i % 7 == 0 else 0)
                    for i, p in enumerate(positions)]
            write_shard(path, rows)
            paths.append(path)
        return paths

    def literal_shards(self, *, sibling):
        from dataclasses import replace
        from hasami_learn.nnue.data import read_shard, write_shard
        source = self.root / "shard-0.hlnn1"
        records = list(read_shard(source if source.exists() else self.data(1)[0]))
        base = self.root / f"case-{sibling}"
        path = base / "run[1]" / "positions[2].hlnn1"
        path.parent.mkdir(parents=True)
        write_shard(path, records)
        if sibling:
            other = base / "run1" / "positions2.hlnn1"
            other.parent.mkdir(exist_ok=True)
            write_shard(other, [replace(row, score=-321, game_id=row.game_id+100) for row in records])
        return path, records

    def test_frozen_paths_are_literal_with_and_without_matching_sibling(self):
        import torch
        from hasami_learn.nnue.control import atomic_json
        from hasami_learn.nnue.data import digest_file
        from hasami_learn.nnue.loader import raw_chunks
        from hasami_learn.nnue.train import train
        from hasami_learn.nnue.window import window_paths
        for sibling in (False, True):
            with self.subTest(sibling=sibling):
                path, records = self.literal_shards(sibling=sibling)
                entry = [str(path.resolve()), digest_file(path), digest_file(str(path)+".json")]
                snapshot = path.parent / "training-data.json"
                atomic_json(snapshot, dict(format="hlnn-window-v1", cycle=1,
                    policy=dict(window_positions=10000, max_cycles=2, data_byte_budget=65536),
                    positions=len(records), shards=[dict(path=str(path), sha256=entry[1],
                    sidecar_sha256=entry[2], records=len(records))]))
                self.assertEqual(window_paths(snapshot), [str(path)])
                options = dict(data=None, data_list=snapshot, epochs=1, width=2, hidden1=2,
                               hidden2=2, batch_size=16, seed=5, validation_fraction=.3,
                               feature_backend="python")
                out = self.root / f"literal-{sibling}"
                history = train(out=out, **options)
                checkpoint = torch.load(out / "checkpoint.pt", weights_only=True)
                self.assertEqual(checkpoint["config"]["data"], [entry])
                self.assertEqual(json.loads((out / "training.json").read_text())["data"], [entry])
                rows = [row for chunk in raw_chunks(checkpoint["config"]["data"][0]) for row in chunk]
                self.assertEqual([int(row["game_id"]) for row in rows], [row.game_id for row in records])
                self.assertEqual([int(row["score"]) for row in rows], [row.score for row in records])
                self.assertEqual(train(out=out, resume=out / "checkpoint.pt", **options), history)
                pattern = glob.escape(str(path.parent))+"/*.hlnn1"
                self.assertEqual(train(out=self.root / f"glob-{sibling}",
                    **{**options, "data_list": None, "data": [pattern]}), history)
                with self.assertRaisesRegex(ValueError, "not both"):
                    train(out=out, **{**options, "data": [glob.escape(str(path))]})
                path.write_bytes(path.read_bytes()[:-1]+b"x")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    train(out=out, resume=out / "checkpoint.pt", **options)

    def test_benchmark_does_not_expand_resolved_shards_again(self):
        from scripts import benchmark_nnue
        from hasami_learn.nnue.model import NNUE
        self.native()
        path, _ = self.literal_shards(sibling=True)

        class TrainingReached(Exception):
            pass

        def check_training_paths(data, *args, **kwargs):
            resolved = [p for pattern in data for p in glob.glob(str(pattern))]
            self.assertEqual(resolved, [str(path)])
            raise TrainingReached

        with patch("hasami_learn.nnue.model.NNUE", side_effect=lambda: NNUE(2, 2, 2)), \
                patch.object(benchmark_nnue, "train", side_effect=check_training_paths):
            with self.assertRaises(TrainingReached):
                benchmark_nnue.training_comparison(self.root / "benchmark",
                    data=[glob.escape(str(path.parent))+"/*.hlnn1"], device="cpu",
                    epochs=1, games=2, workers=0)

    def test_vectorised_reference_factor_folding_and_gradients(self):
        import torch
        from hasami_learn.nnue.features import FEATURE_COUNT, MAX_ACTIVE, batch_features
        from hasami_learn.nnue.model import NNUE
        from hasami_learn.nnue.export import export_weights, load_weights
        # Random features also exercise padding and arbitrary king blocks.
        random_features = torch.randint(FEATURE_COUNT, (128, 2, MAX_ACTIVE)).sort(dim=-1).values
        random_features[::2, :, 20:] = -1
        real = batch_features(self.positions())
        for factorised in (False, True):
            model = NNUE(factorised=factorised)
            if factorised:
                with torch.no_grad():
                    model.factor.weight.normal_(0, .02)
            for indices in (random_features, real):
                with torch.no_grad():
                    fast, reference = model.forward_training(indices), model(indices)
                torch.testing.assert_close(fast, reference, atol=.002, rtol=1e-5)
            path = self.root / f"folded-{factorised}.hlnnw1"
            export_weights(model, path)
            folded = load_weights(path)
            with torch.no_grad():
                self.assertTrue(torch.equal(model(real), folded(real)))
                torch.testing.assert_close(model.forward_training(real), folded(real), atol=.002, rtol=1e-5)
            model.forward_training(real).square().mean().backward()
            self.assertGreater(model.embedding.weight.grad.abs().sum().item(), 0)
            if factorised:
                self.assertGreater(model.factor.weight.grad.abs().sum().item(), 0)

    def test_native_batch_encoding_integrity_and_capacity(self):
        self.native()
        import numpy as np
        from hasami_learn.nnue.loader import RAW_DTYPE, encode_records
        from hasami_learn.nnue.data import HEADER, read_shard
        from hasami_learn.nnue.features import batch_features
        path = self.data(1)[0]
        raw = np.frombuffer(path.read_bytes()[HEADER.size:], dtype=RAW_DTYPE)
        reference = batch_features([r.position for r in read_shard(path)]).numpy()
        np.testing.assert_array_equal(encode_records(raw), reference)
        np.testing.assert_array_equal(encode_records(raw, "python"), reference)
        lib, encode = self.native()
        self.assertEqual(encode(None, 137, None, 78), -1)
        self.assertEqual(encode(raw.ctypes.data, raw.nbytes-1, None, 0), -1)
        self.assertEqual(encode(raw.ctypes.data, raw.nbytes, None, 1), -1)
        self.assertEqual(encode(None, 0, None, 0), 0)
        corrupt = raw.copy()
        corrupt[0]["board"][0] = 255
        with self.assertRaises(ValueError):
            encode_records(corrupt)
        corrupt = raw.copy()
        corrupt[0]["teacher"] = float("inf")
        with self.assertRaises(ValueError):
            encode_records(corrupt)
        self.assertTrue(lib.ss_last_error())

    def test_trained_factor_export_remains_bit_exact_in_native(self):
        self.native()
        from types import SimpleNamespace
        import torch
        from shogi.model import Rules
        from shogi.native_search import NativeSearch
        from hasami_learn.nnue.features import batch_features
        from hasami_learn.nnue.model import NNUE
        from hasami_learn.nnue.export import export_weights, load_weights
        positions = self.positions(96)
        indices = batch_features(positions)
        model = NNUE(factorised=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        loss = (model.forward_training(indices)/600-.3).square().mean()
        loss.backward(); optimizer.step()
        self.assertGreater(model.factor.weight.abs().sum().item(), 0)
        path = self.root / "trained.hlnnw1"
        export_weights(model, path)
        folded = load_weights(path)
        with torch.no_grad():
            reference = folded(indices)
        native_values = []
        with NativeSearch(nnue=path, hash_mb=1) as native:
            for position in positions:
                native.set_position(SimpleNamespace(initial=position, moves=[], rules=Rules(),
                                                     outcome=SimpleNamespace(reason="ongoing")))
                native_values.append(native.evaluate_float())
        self.assertTrue(torch.equal(reference, torch.tensor(native_values)))

    def test_streaming_workers_shuffle_coverage_split_and_hash(self):
        self.native()
        from hasami_learn.nnue.data import digest_file
        from hasami_learn.nnue.loader import feature_batches, validation_game
        paths = self.data()
        entries = [[str(p), digest_file(p), digest_file(str(p)+".json")] for p in paths]
        options = dict(seed=5, fraction=.3, batch_size=8, shuffle_buffer=32, backend="native")

        def collect(workers, epoch=0, validation=False):
            return [int(g) for batch in feature_batches(entries, workers=workers, epoch=epoch,
                    validation=validation, **options) for g in batch["game_id"].tolist()]

        for workers in (0, 2):
            training, validation = collect(workers), collect(workers, validation=True)
            self.assertEqual(len(training), len(set(training)))
            self.assertEqual(set(training) | set(validation), {s*100+i for s in range(4) for i in range(24)})
            self.assertFalse(set(training) & set(validation))
            self.assertTrue(all(validation_game(g, 5, .3) for g in validation))
            self.assertEqual(training, collect(workers))
            self.assertNotEqual(training, collect(workers, epoch=1))
            self.assertGreater(len({g//100 for g in training[:32]}), 1)
        path = paths[0]
        path.write_bytes(path.read_bytes()[:-1]+b"x")
        with self.assertRaisesRegex(ValueError, "sha256"):
            collect(0)

    def test_best_epoch_early_stop_resume_and_factor_checkpoint(self):
        import torch
        from hasami_learn.nnue.train import train, run_partition
        from hasami_learn.nnue.export import load_weights
        from hasami_learn.nnue.model import NNUE
        paths = self.data(1)
        options = dict(data=paths, width=4, hidden1=4, hidden2=4, batch_size=16,
                       seed=5, validation_fraction=.3, factorised=True, patience=1,
                       lr_schedule="cosine", schedule_epochs=4, feature_backend="python")

        def increasing_loss(*args, **kwargs):
            row, timing = run_partition(*args, **kwargs)
            if kwargs["validation"]:
                row["validation_loss"] = .5+kwargs["epoch"]
            return row, timing

        with patch("hasami_learn.nnue.train.run_partition", side_effect=increasing_loss):
            history = train(out=self.root / "full", epochs=4, **options)
            train(out=self.root / "resume", epochs=1, **options)
            checkpoint_path = self.root / "resume/checkpoint.pt"
            resumed = train(out=self.root / "resume", resume=checkpoint_path, epochs=4, **options)
        self.assertEqual(history, resumed)
        self.assertEqual(len(history), 2)
        self.assertTrue(history[-1]["early_stopped"])
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        self.assertEqual(checkpoint["best_epoch"], 1)
        self.assertIn("factor.weight", checkpoint["model"])
        self.assertFalse(torch.equal(checkpoint["model"]["factor.weight"], checkpoint["best_model"]["factor.weight"]))
        best = NNUE(4, 4, 4, factorised=True)
        best.load_state_dict(checkpoint["best_model"])
        exported = load_weights(self.root / "resume/weights.hlnnw1")
        for key, value in best.export_state().items():
            self.assertTrue(torch.equal(value, exported.state_dict()[key]))
        with patch("hasami_learn.nnue.train.run_partition") as run:
            again = train(out=self.root / "recovered", resume=checkpoint_path, epochs=4, **options)
            run.assert_not_called()
        self.assertEqual(again, history)
        for change in ({"factorised": False}, {"schedule_epochs": 5}, {"correlation_clip": 1000.}):
            with self.assertRaisesRegex(ValueError, "mismatch"):
                train(out=self.root / "bad", resume=checkpoint_path, epochs=4, **{**options, **change})

    def test_mate_targets_and_robust_metrics(self):
        import torch
        from hasami_learn.nnue.train import batch_targets, Correlation
        batch = dict(score=torch.tensor([999999., -999999., 0.]),
                     result=torch.tensor([-1., 1., -1.]), flags=torch.tensor([4, 4, 4]),
                     teacher=torch.full((3,), float("nan")))
        torch.testing.assert_close(batch_targets(batch, scale=600, mix=.8, mu=0), torch.tensor([1., 0., .5]))
        x, y = torch.tensor([1., 2., 3.], dtype=torch.float64), torch.tensor([1., 2., 3.], dtype=torch.float64)
        stats = Correlation()
        stats.add(x[:1], y[:1]); stats.add(x[1:], y[1:])
        self.assertAlmostEqual(stats.value(), 1.)
        stats = Correlation(); stats.add(x, torch.ones_like(y))
        self.assertIsNone(stats.value())

    def test_reported_correlations_clip_mates_and_use_sigmoid_space(self):
        import numpy as np
        import torch
        from hasami_learn.nnue.control import TRAINING
        from hasami_learn.nnue.data import digest_file
        from hasami_learn.nnue.loader import feature_batches
        from hasami_learn.nnue.model import NNUE
        from hasami_learn.nnue.train import run_partition
        paths = self.data(1)
        entries = [[str(p), digest_file(p), digest_file(str(p)+".json")] for p in paths]
        config = {**TRAINING, "seed": 5, "validation_fraction": .3, "feature_backend": "python"}
        host = next(iter(feature_batches(entries, validation=True, epoch=0, seed=5, fraction=.3,
                        batch_size=1024, shuffle_buffer=65536, backend="python")))
        y = host["score"].double()
        prediction = torch.linspace(-5000, 5000, len(y))
        model = NNUE(2, 2, 2)
        with patch.object(model, "forward_training", return_value=prediction):
            row, _ = run_partition(model, None, entries, validation=True, epoch=0,
                                   config=config, device=torch.device("cpu"))
        expected = np.corrcoef(prediction.double().clamp(-3000, 3000), y.clamp(-3000, 3000))[0, 1]
        self.assertAlmostEqual(row["teacher_correlation"], expected)
        sigmoid = np.corrcoef(torch.sigmoid(prediction.double()/600), torch.sigmoid(y/600))[0, 1]
        self.assertAlmostEqual(row["sigmoid_correlation"], sigmoid)
        self.assertNotAlmostEqual(expected, np.corrcoef(prediction, y)[0, 1])

    def test_multiprocess_training_resume_is_exact(self):
        self.native()
        from hasami_learn.nnue.train import train
        options = dict(data=self.data(2), width=2, hidden1=2, hidden2=2, workers=2,
                       batch_size=16, shuffle_buffer=32, feature_backend="native", seed=5,
                       validation_fraction=.3, factorised=True)
        full = train(out=self.root / "full-workers", epochs=2, **options)
        train(out=self.root / "resume-workers", epochs=1, **options)
        resumed = train(out=self.root / "resume-workers", epochs=2,
                        resume=self.root / "resume-workers/checkpoint.pt", **options)
        self.assertEqual(full, resumed)
        self.assertEqual((self.root / "full-workers/weights.hlnnw1").read_bytes(),
                         (self.root / "resume-workers/weights.hlnnw1").read_bytes())
