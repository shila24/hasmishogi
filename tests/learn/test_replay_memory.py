"""Replay storage footprints and startup memory diagnostics."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from tests.learn._common import LearnTestCase, ROOT, requires_torch


@requires_torch
class ReplayMemoryTests(unittest.TestCase):
    def test_exact_storage_bytes_for_each_game_and_layout(self):
        import numpy as np
        from hasami_learn.games import get_spec
        from hasami_learn.replay import ReplayBuffer, fields_for

        # float32 planes/scalars, uint8 masks, float16 policy, int64 actions.
        cases = (
            ("hasami", "pv", 9 * 9 * 9 * 4 + 6561 + 6561 * 2 + 4),
            ("hasami", "q", 2 * 9 * 9 * 9 * 4 + 2 * 6561 + 8 + 4 + 4),
            ("shogi", "pv", 49 * 9 * 9 * 4 + 2188 + 2188 * 2 + 4),
            ("shogi", "q", 2 * 49 * 9 * 9 * 4 + 2 * 2188 + 8 + 4 + 4),
        )
        for game, kind, expected in cases:
            for capacity in (1, 3):
                with self.subTest(game=game, kind=kind, capacity=capacity):
                    fields = fields_for(get_spec(game), kind=kind)
                    replay = ReplayBuffer(capacity, fields)
                    self.assertEqual(len(replay), 0)
                    self.assertEqual(replay.sample_nbytes, expected)
                    self.assertEqual(replay.capacity_nbytes, capacity * expected)
                    self.assertEqual(replay.capacity_nbytes, replay.nbytes())
                    self.assertIs(type(replay.sample_nbytes), int)
                    self.assertIs(type(replay.capacity_nbytes), int)
                    sample = {name: np.zeros(shape, dtype=dtype)
                              for name, (shape, dtype) in fields.items()}
                    for _ in range(capacity + 1):
                        replay.add(**sample)
                    self.assertEqual(len(replay), capacity)
                    self.assertEqual(replay.sample_nbytes, expected)
                    self.assertEqual(replay.capacity_nbytes, capacity * expected)
                    replay.clear()
                    self.assertEqual(replay.sample_nbytes, expected)
                    self.assertEqual(replay.capacity_nbytes, capacity * expected)

    def test_custom_shapes_and_dtypes(self):
        import numpy as np
        from hasami_learn.replay import ReplayBuffer

        replay = ReplayBuffer(5, {"vector": ((2, 3), np.float64),
                                  "scalar": ((), np.uint16)})
        self.assertEqual(replay.sample_nbytes, 2 * 3 * 8 + 2)
        self.assertEqual(replay.capacity_nbytes, 250)

    def test_warning_threshold_is_strict_and_fraction_is_configurable(self):
        from hasami_learn.replay import memory_warning

        with patch("hasami_learn.replay._total_ram_bytes") as reader:
            for fraction, threshold in ((0.5, 500), (0.25, 250)):
                with self.subTest(fraction=fraction):
                    self.assertIsNone(memory_warning(threshold - 1, 1000, fraction))
                    self.assertIsNone(memory_warning(threshold, 1000, fraction))
                    warning = memory_warning(threshold + 1, 1000, fraction)
                    self.assertIsInstance(warning, str)
                    self.assertIn("replay", warning.lower())
                    self.assertIn(str(threshold + 1), warning)
                    self.assertIn("1000", warning)
            self.assertIsNotNone(memory_warning(501, 1000))
            reader.assert_not_called()

    def test_memtotal_is_read_in_kibibytes(self):
        from hasami_learn.replay import memory_warning

        with patch("hasami_learn.replay.Path.read_text", return_value=(
            "MemFree: 1 kB\nMemTotal:       2048 kB\nMemAvailable: 8 kB\n"
        )):
            self.assertIsNone(memory_warning(1024 * 1024))
            self.assertIsNotNone(memory_warning(1024 * 1024 + 1))

    def test_unknown_ram_does_not_guess_or_warn(self):
        from hasami_learn.replay import memory_warning

        for contents in ("", "MemAvailable: 1000 kB\n", "MemTotal:\n",
                         "MemTotal: unknown kB\n", "MemTotal: 2048 MB\n",
                         "MemTotal: 0 kB\n", "MemTotal: -1 kB\n"):
            with self.subTest(contents=contents):
                with patch("hasami_learn.replay.Path.read_text", return_value=contents):
                    self.assertIsNone(memory_warning(10**15))
        for error in (FileNotFoundError(), PermissionError(), UnicodeError()):
            with self.subTest(error=type(error).__name__):
                with patch("hasami_learn.replay.Path.read_text", side_effect=error):
                    self.assertIsNone(memory_warning(10**15))


@requires_torch
class ReplayMemoryTrainerTests(LearnTestCase):
    def config(self, algorithm):
        params = {"simulations": 2, "batch_games": 1} if algorithm == "alphazero" else {}
        config = self.tiny(algorithm, **params)
        config.selfplay.games_per_iteration = 1
        config.selfplay.max_plies = 12
        config.replay.capacity = 32
        config.replay.min_size = 1
        config.replay.batch_size = 4
        config.replay.persist = True
        return config

    def info_rows(self, run_dir):
        import jsonschema

        schema = json.loads((ROOT / "hasami_learn/schemas/training-metrics.schema.json").read_text())
        rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
        for row in rows:
            jsonschema.validate(row, schema)
        return [row for row in rows if row["kind"] == "info"]

    def test_info_rows_validate_without_private_schema(self):
        import jsonschema

        packaged = self.tmp / "hasami_learn" / "schemas" / "training-metrics.schema.json"
        packaged.parent.mkdir(parents=True)
        packaged.write_bytes((ROOT / "hasami_learn" / "schemas" / packaged.name).read_bytes())
        row = {"timestamp": "2026-11-16T00:00:00Z", "run_id": "test", "iteration": 0, "kind": "info"}
        metrics = self.tmp / "metrics.jsonl"
        metrics.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with patch(__name__ + ".ROOT", self.tmp):
            self.assertEqual(self.info_rows(self.tmp), [row])
            metrics.write_text(json.dumps({**row, "iteration": -1}) + "\n", encoding="utf-8")
            with self.assertRaises(jsonschema.ValidationError):
                self.info_rows(self.tmp)

    def test_tiny_runs_and_resumes_report_storage_and_warn_once(self):
        from hasami_learn.trainer import Trainer

        for algorithm, sample_bytes in (("alphazero", 22603), ("dqn", 18970)):
            with self.subTest(algorithm=algorithm):
                config = self.config(algorithm)
                run_dir = self.tmp / "runs" / config.run.run_id
                for resume, iteration in ((None, 0), (run_dir, 1)):
                    config.run.iterations = iteration + 1
                    stderr, stdout = io.StringIO(), io.StringIO()
                    with patch("hasami_learn.replay._total_ram_bytes", return_value=1024) as reader:
                        with redirect_stderr(stderr), redirect_stdout(stdout):
                            trainer = Trainer(config, resume=resume, echo=False)
                            try:
                                self.assertEqual(trainer.iteration, iteration)
                                self.assertEqual(len(trainer.replay) > 0, resume is not None)
                                checkpoint = trainer.run()
                                self.assertTrue(checkpoint.is_dir())
                                self.assertEqual(trainer.iteration, iteration + 1)
                            finally:
                                trainer.close()
                        reader.assert_called_once_with()
                    lines = stderr.getvalue().splitlines()
                    self.assertEqual(len(lines), 1, stderr.getvalue())
                    self.assertIn("replay", lines[0].lower())
                    self.assertEqual(stdout.getvalue(), "")
                    rows = self.info_rows(run_dir)
                    self.assertEqual(len(rows), iteration + 1)
                    extra = rows[-1]["extra"]
                    self.assertEqual(rows[-1]["iteration"], iteration)
                    self.assertEqual(extra["replay_capacity"], 32)
                    self.assertEqual(extra["replay_sample_bytes"], sample_bytes)
                    self.assertEqual(extra["replay_capacity_bytes"], 32 * sample_bytes)

    def test_no_warning_when_ram_is_unknown_or_buffer_is_small(self):
        from hasami_learn.trainer import Trainer

        for ram in (None, 2**40):
            with self.subTest(ram=ram):
                stderr = io.StringIO()
                with patch("hasami_learn.replay._total_ram_bytes", return_value=ram):
                    with redirect_stderr(stderr):
                        trainer = Trainer(self.config("dqn"), echo=False)
                        trainer.close()
                self.assertEqual(stderr.getvalue(), "")

    def test_learner_without_replay_is_unaffected(self):
        from hasami_learn.trainer import Trainer

        stderr = io.StringIO()
        with patch("hasami_learn.replay._total_ram_bytes", return_value=1) as reader:
            with redirect_stderr(stderr):
                trainer = Trainer(self.config("reinforce"), echo=False)
                try:
                    self.assertIsNone(trainer.replay)
                    extra = self.info_rows(trainer.run_dir)[0]["extra"]
                    self.assertFalse(any(key.startswith("replay_") for key in extra))
                finally:
                    trainer.close()
            reader.assert_not_called()
        self.assertEqual(stderr.getvalue(), "")

    def test_cli_quiet_keeps_warning_on_stderr(self):
        from hasami_learn.cli import main

        config = self.config("alphazero")
        config_path = self.tmp / "config.json"
        config.save(config_path)
        stderr, stdout = io.StringIO(), io.StringIO()
        with patch("hasami_learn.replay._total_ram_bytes", return_value=1024):
            with redirect_stderr(stderr), redirect_stdout(stdout):
                result = main(["train", "--config", str(config_path), "--quiet"])
        self.assertEqual(result, 0)
        self.assertEqual(len(stderr.getvalue().splitlines()), 1)
        self.assertIn("replay", stderr.getvalue().lower())
        self.assertNotIn("replay", stdout.getvalue().lower())


if __name__ == "__main__":
    unittest.main()
