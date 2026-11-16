"""NNUE cross-language contracts and deterministic learning regressions."""
import importlib.util
import os
import unittest

TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH, "torch is optional")
class NNUERegressionTests(unittest.TestCase):
    def setUp(self):
        from pathlib import Path
        import tempfile
        from shogi import create_game
        from hasami_learn.nnue.data import Record, write_shard

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.base = Path(directory.name)
        game = create_game()
        self.addCleanup(game.close)
        data = self.base / "train.hlnn1"
        write_shard(data, [Record(game.position, (i-20)*20, 1, 30,
                                 game.legal_actions()[0], (i % 3)-1, 0, i) for i in range(40)])
        self.options = dict(data=[data], width=2, hidden1=2, hidden2=2,
                            batch_size=16, seed=5, validation_fraction=.3, device="cpu")

    def test_result_correlation_excludes_experimental_truncation(self):
        from dataclasses import replace
        from hasami_learn.nnue.data import TRUNCATED, read_shard, write_shard
        from hasami_learn.nnue.train import train, validation_game

        records = list(read_shard(self.options["data"][0]))
        records = [replace(r, flags=TRUNCATED if r.game_id % 2 else 0) for r in records]
        path = self.base / "truncated.hlnn1"
        write_shard(path, records)
        history = train(out=self.base / "truncation", epochs=1, **{**self.options, "data": [path]})
        expected = sum(not r.flags & TRUNCATED and validation_game(r.game_id, 5, .3) for r in records)
        self.assertEqual(history[0]["result_correlation_records"], expected)
        self.assertIn("result_correlation", history[0])
        self.assertEqual(history[0]["teacher_correlation_basis"],
                         "scores and predictions clipped to +/-3000 cp")
        self.assertIn("sigmoid_correlation", history[0])

    def test_validation_statistics_support_float32_only_device(self):
        from unittest.mock import Mock, patch
        from hasami_learn.nnue.model import NNUE
        from hasami_learn.nnue.train import train

        forward = NNUE.forward_training
        detached_predictions = []

        def device_forward(model, indices):
            prediction = forward(model, indices)
            if not model.training:
                # Model a device that forbids float64 while retaining real CPU training/loss.
                detached = Mock()
                detached.cpu.return_value = prediction.detach().cpu()
                detached.double.side_effect = TypeError("Device does not support float64")
                prediction.detach = lambda: detached
                detached_predictions.append(detached)
            return prediction

        with patch.object(NNUE, "forward_training", device_forward):
            history = train(out=self.base / "cpu", epochs=1, **self.options)
        self.assertGreater(history[0]["validation_records"], 0)
        self.assertTrue(detached_predictions)
        for prediction in detached_predictions:
            prediction.cpu.assert_called_once_with()
        self.assertTrue((self.base / "cpu/checkpoint.pt").is_file())

    def test_validation_on_mps_and_auto_saves_checkpoint(self):
        import math
        import torch
        from unittest.mock import patch
        from hasami_learn.devices import mps_available
        from hasami_learn.nnue.train import train

        if not mps_available():
            self.skipTest("Apple MPS device is not available")
        for device in ("mps", "auto"):
            with self.subTest(device=device), patch("torch.cuda.is_available", return_value=False):
                output = self.base / device
                history = train(out=output, epochs=1, **{**self.options, "device": device})
                checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
                self.assertEqual(checkpoint["config"]["device"], "mps")
                self.assertEqual(checkpoint["metrics"], history)
                self.assertGreater(history[0]["validation_records"], 0)
                self.assertTrue(math.isfinite(history[0]["validation_loss"]))

    def test_completed_resume_recovers_metrics_after_publication_interruption(self):
        import json
        import torch
        from unittest.mock import patch
        from hasami_learn.nnue.data import atomic_write
        from hasami_learn.nnue.train import train

        output = self.base / "interrupted"

        def interrupt_final_metrics(path, content):
            if path.name == "metrics.jsonl" and len(content.splitlines()) == 2:
                raise OSError("Interrupted after final checkpoint")
            return atomic_write(path, content)

        with patch("hasami_learn.nnue.train.atomic_write", side_effect=interrupt_final_metrics):
            with self.assertRaisesRegex(OSError, "Interrupted after final checkpoint"):
                train(out=output, epochs=2, **self.options)
        checkpoint_path = output / "checkpoint.pt"
        checkpoint_bytes = checkpoint_path.read_bytes()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["epoch"], 2)
        self.assertEqual(len((output / "metrics.jsonl").read_text().splitlines()), 1)
        expected = "".join(json.dumps(row, sort_keys=True)+"\n" for row in checkpoint["metrics"])
        for boundary in ("stale", "missing", "new_directory"):
            with self.subTest(boundary=boundary):
                destination = self.base / "recovered" if boundary == "new_directory" else output
                if boundary == "missing":
                    (output / "metrics.jsonl").unlink()
                with patch("hasami_learn.nnue.train.feature_batches") as batches:
                    history = train(out=destination, resume=checkpoint_path, epochs=2, **self.options)
                    batches.assert_not_called()
                self.assertEqual(history, checkpoint["metrics"])
                self.assertTrue((destination / "metrics.jsonl").is_file())
                self.assertEqual((destination / "metrics.jsonl").read_text(), expected)
                self.assertEqual(checkpoint_path.read_bytes(), checkpoint_bytes)
                self.assertTrue((destination / "weights.hlnnw1").is_file())

    def test_cross_language_native_availability_policy(self):
        import subprocess
        import sys

        cases = (
            ("SHOGI_SEARCH_LIB", "test_data_and_weights_cross_language"),
            ("SHOGI_SEARCH_LIB", "test_forward_and_features_on_10000_random_and_search_positions"),
            ("SHOGI_LIB", "test_forward_and_features_on_10000_random_and_search_positions"),
        )
        for variable, method in cases:
            for required in (False, True):
                with self.subTest(library=variable, method=method, required=required):
                    env = dict(os.environ, **{variable: str(self.base / "missing-library"),
                                              "HASAMI_REQUIRE_NATIVE": "1" if required else "0"})
                    result = subprocess.run(
                        [sys.executable, "-m", "unittest", f"tests.learn.test_nnue.NNUEContracts.{method}", "-v"],
                        env=env, capture_output=True, text=True, timeout=60)
                    if required:
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn("Native shogi library is required", result.stderr)
                        self.assertNotIn("skipped", result.stderr)
                    else:
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertIn("skipped 'Native shogi library is unavailable'", result.stderr)


@unittest.skipUnless(TORCH, "torch is optional")
class NNUEContracts(unittest.TestCase):
    def require_native(self, *, game=False):
        from shogi.native_search import available as search_available
        from shogi.native import available as game_available
        if not search_available() or (game and not game_available()):
            if os.environ.get("HASAMI_REQUIRE_NATIVE") == "1":
                self.fail("Native shogi library is required")
            self.skipTest("Native shogi library is unavailable")

    def test_initial_features_and_architecture(self):
        from hasami_learn.nnue.features import active_features, FEATURE_COUNT
        from hasami_learn.nnue.model import NNUE
        from shogi import create_game
        pairs = active_features(create_game().position)
        self.assertEqual([len(p) for p in pairs], [39, 39])
        self.assertEqual(FEATURE_COUNT, 183303)
        self.assertEqual(NNUE(width=4).architecture, "hlnn-kp-v1:w4:32:32:f32:cp600")

    def test_generated_layout(self):
        from pathlib import Path
        from hasami_learn.nnue.features import generated_header
        root = Path(__file__).resolve().parents[2]
        self.assertEqual((root / "cpp/shogi_search/include/shogi_search/nnue_layout.hpp").read_text(), generated_header())

    def test_data_and_weights_cross_language(self):
        self.require_native()
        import ctypes as C
        from dataclasses import replace
        import hashlib
        import math
        from pathlib import Path
        import tempfile
        import torch
        from shogi import create_game
        from shogi.native_search import NativeSearch
        from hasami_learn.nnue.data import Record, write_shard, read_shard
        from hasami_learn.nnue.export import export_weights, load_weights, HEADER
        from hasami_learn.nnue.model import NNUE
        torch.set_num_threads(1)
        torch.manual_seed(50)
        game = create_game()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            original, copied = base / "original.hlnn1", base / "copied.hlnn1"
            records = [Record(game.position, -200, 4, 123456, game.legal_actions()[0], -1, 0, 7),
                       Record(game.position, 300, 12, 2**64-1, game.legal_actions()[-1], 1, 42, 9,
                              flags=0x012303, teacher=0.25)]
            write_shard(original, records)
            self.assertEqual([r.pack() for r in read_shard(original)], [r.pack() for r in records])
            model = NNUE(width=4)
            weights = base / "weights.hlnnw1"
            export_weights(model, weights)
            loaded = load_weights(weights)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.state_dict()[key]))
            with NativeSearch(game, nnue=weights) as native:
                self.assertEqual(native.evaluator, model.architecture)
                native.save_nnue(base / "native.hlnnw1")
                self.assertEqual(weights.read_bytes(), (base / "native.hlnnw1").read_bytes())
                status = native._lib.ss_data_roundtrip(str(original).encode(), str(copied).encode())
                native._check(status)
                self.assertEqual(original.read_bytes(), copied.read_bytes())
                self.assertEqual([r.pack() for r in read_shard(copied)], [r.pack() for r in records])
                old_score = native.evaluate_float()
                for offset in (0, 12, 28, 80, HEADER.size, -1):
                    content = bytearray(weights.read_bytes())
                    content[offset] ^= 1
                    # Rehash ID mismatches: refusal cannot rely only on checksum failure.
                    if offset < HEADER.size and offset >= 0:
                        content[-32:] = hashlib.sha256(content[:-32]).digest()
                    bad = base / "bad.hlnnw1"
                    bad.write_bytes(content)
                    with self.subTest(offset=offset):
                        with self.assertRaises(ValueError):
                            native.load_nnue(bad)
                        with self.assertRaises(ValueError):
                            load_weights(bad)
                        self.assertEqual(native.evaluate_float(), old_score)
                native.load_nnue()
                self.assertEqual(native.evaluator, "bootstrap-v1")
                self.assertEqual(native.evaluate_float(), native.evaluate())
                with self.assertRaises(ValueError):
                    native.nnue_features(2)
                self.assertEqual(native._lib.ss_nnue_features(native._handle, 0, None, 0), 39)
                tiny = (C.c_int32 * 1)()
                self.assertEqual(native._lib.ss_nnue_features(native._handle, 0, tiny, 1), -1)
                self.assertEqual(native._lib.ss_evaluate_float(native._handle, None), -1)
                self.assertEqual(native._lib.ss_load_nnue(None, None), -1)
                content = bytearray(original.read_bytes()); content[-1] ^= 1
                original.write_bytes(content)
                with self.assertRaisesRegex(ValueError, "sha256"):
                    list(read_shard(original))
                self.assertEqual(native._lib.ss_data_roundtrip(str(original).encode(), str(base / "bad.hlnn1").encode()), -1)
            from hasami_learn.games.shogi import ShogiSpec
            spec = ShogiSpec()
            agent, limits = spec.baseline_agent(f"alphabeta-cpp:nodes=100,nnue={weights}")
            try:
                decision = agent.choose(game, limits)
                self.assertIn(decision.action, spec.legal_actions(game, game.position.turn))
            finally:
                agent.agent.close()
            for changes in ({"teacher": math.inf}, {"flags": 8}, {"result": 2}, {"best_move": 20000}):
                with self.assertRaises(ValueError):
                    replace(records[0], **changes).pack()

    def test_forward_and_features_on_10000_random_and_search_positions(self):
        self.require_native(game=True)
        from pathlib import Path
        import random
        import tempfile
        from types import SimpleNamespace
        import torch
        from shogi import create_game
        from shogi.model import Rules, DECLARE, DROP_BASE
        from shogi.native_search import NativeSearch
        from hasami_learn.nnue.features import active_features, batch_features
        from hasami_learn.nnue.model import NNUE
        from hasami_learn.nnue.export import export_weights
        torch.set_num_threads(1)
        torch.manual_seed(50050)
        rng = random.Random(50050)
        model = NNUE().eval()
        positions, coverage = [], set()
        with NativeSearch(hash_mb=1, features=0) as teacher:
            while len(positions) < 10000:
                game = create_game("cpp", Rules(max_moves=128))
                try:
                    while not game.outcome.terminal and len(positions) < 10000:
                        position = game.position
                        positions.append(position)
                        coverage.add("white" if position.turn else "black")
                        if any(any(hand) for hand in position.hands):
                            coverage.add("hands")
                        if any(cell & 15 > 8 for cell in position.board):
                            coverage.add("promoted")
                        legal = game.legal_actions()
                        if len(positions) % 11 == 0:
                            # Evaluation-only snapshots need no repetition history; moves still come from the actual game.
                            teacher.set_position(SimpleNamespace(initial=position, moves=[], rules=game.rules, outcome=game.outcome))
                            move = teacher.go(depth=1)["move"]
                            coverage.add("search")
                        else:
                            move = rng.choice(legal)
                        if DROP_BASE <= move < DECLARE:
                            coverage.add("drops")
                        game.play(move)
                finally:
                    game.close()
        self.assertEqual(coverage, {"white", "black", "hands", "promoted", "search", "drops"})
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "parity.hlnnw1"
            export_weights(model, weights)
            native_values = []
            with NativeSearch(nnue=weights, hash_mb=1) as native:
                for position in positions:
                    native.set_position(SimpleNamespace(initial=position, moves=[], rules=Rules(),
                                                        outcome=SimpleNamespace(reason="ongoing")))
                    pairs = active_features(position)
                    self.assertEqual(native.nnue_features(position.turn), pairs[0])
                    self.assertEqual(native.nnue_features(1-position.turn), pairs[1])
                    native_values.append(native.evaluate_float())
            worst = 0.0
            with torch.no_grad():
                for start in range(0, len(positions), 128):
                    features = batch_features(positions[start:start+128])
                    reference = model(features)
                    cpp = torch.tensor(native_values[start:start+128])
                    self.assertTrue(torch.equal(reference, cpp), (reference-cpp).abs().max().item())
                    worst = max(worst, (reference-cpp).abs().max().item())
            print(f"NNUE parity: 10000 positions, W=256, forward bit-exact; max_abs_cp={worst:.9g}", flush=True)

    def test_training_resume_split_targets_and_cpu_determinism(self):
        from pathlib import Path
        import tempfile
        import torch
        from shogi import create_game
        from hasami_learn.nnue.data import Record, TRUNCATED, write_shard
        from hasami_learn.nnue.train import train, targets, validation_game, batches
        from hasami_learn.nnue.export import load_weights
        game = create_game()
        records = [Record(game.position, (i-20)*20, 1, 30, game.legal_actions()[0], (i % 3)-1, 0, i,
                          teacher=0.5 if i % 2 else float("nan")) for i in range(40)]
        self.assertEqual(len({validation_game(7, 5, .3) for _ in range(10)}), 1)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            data = base / "train.hlnn1"
            write_shard(data, records)
            second = base / "second.hlnn1"
            write_shard(second, records[::2])
            partitions = []
            for is_validation in (False, True):
                rows = [row for batch in batches([data, second], validation=is_validation, seed=5,
                        fraction=.3, batch_size=7, epoch=0) for row in batch]
                partitions.append({row.game_id for row in rows})
            self.assertTrue(partitions[0] and partitions[1])
            self.assertFalse(partitions[0] & partitions[1])
            self.assertEqual(partitions[0] | partitions[1], set(range(40)))
            from unittest.mock import patch
            from hasami_learn.nnue.data import atomic_write
            marker = base / "atomic.bin"
            atomic_write(marker, b"old")
            with patch("hasami_learn.nnue.data.os.replace", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    atomic_write(marker, b"new")
            self.assertEqual(marker.read_bytes(), b"old")
            self.assertFalse(list(base.glob("atomic.bin.*.tmp")))
            options = dict(data=[data], width=4, hidden1=4, hidden2=4, batch_size=16, seed=5,
                           validation_fraction=.3, device="cpu", mu=.2)
            full = train(out=base / "full", epochs=2, **options)
            train(out=base / "resume", epochs=1, **options)
            resumed = train(out=base / "resume", resume=base / "resume/checkpoint.pt", epochs=2, **options)
            self.assertEqual(full, resumed)
            self.assertEqual((base / "full/weights.hlnnw1").read_bytes(), (base / "resume/weights.hlnnw1").read_bytes())
            with self.assertRaisesRegex(ValueError, "mismatch"):
                train(out=base / "resume", resume=base / "resume/checkpoint.pt", epochs=3, **{**options, "seed": 6})
            init_options = {k: v for k, v in options.items() if k not in ("width", "hidden1", "hidden2")}
            train(out=base / "init", init=base / "full/weights.hlnnw1", epochs=1, **init_options)
        target = targets([Record(game.position, 0, 0, 0, 0, -1, 0, 0, flags=TRUNCATED)],
                         scale=600, mix=.8, mu=0, device="cpu")
        self.assertEqual(target.item(), .5)
