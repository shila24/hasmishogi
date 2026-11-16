"""Regression tests for checkpoint publication, recovery, and graceful stops."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from unittest.mock import patch

from tests.learn._common import LearnTestCase, ROOT, requires_torch


@requires_torch
class ResilienceTests(LearnTestCase):
    def config(self, iterations=2):
        config = self.tiny("alphazero", simulations=2, batch_games=1)
        config.run.iterations = iterations
        config.run.keep_checkpoints = 5
        config.selfplay.games_per_iteration = 1
        config.selfplay.max_plies = 12
        config.replay.capacity = 64
        config.replay.persist = True
        return config

    def trainer(self, config, **kwargs):
        from hasami_learn.trainer import Trainer

        trainer = Trainer(config, echo=False, **kwargs)
        self.addCleanup(trainer.close)
        return trainer

    def rows(self, trainer):
        return [json.loads(line) for line in (trainer.run_dir / "metrics.jsonl").read_text().splitlines()]

    def test_failed_write_never_publishes_partial_checkpoint(self):
        from hasami_learn.manifest import load_checkpoint

        trainer = self.trainer(self.config())
        first = trainer.run()
        pointer = (first.parent / "latest.json").read_bytes()
        with patch("hasami_learn.manifest.torch.save", side_effect=OSError("power loss")):
            with self.assertRaises(OSError):
                trainer.checkpoint(3)
        self.assertFalse((first.parent / "iter_000003").exists())
        self.assertEqual((first.parent / "latest.json").read_bytes(), pointer)
        self.assertEqual(load_checkpoint(first).trainer_state["iteration"], 2)

    def test_resume_cleans_strays_and_falls_back_from_corrupt_weights(self):
        from hasami_learn.manifest import load_checkpoint

        config = self.config()
        trainer = self.trainer(config)
        latest = trainer.run()
        (latest / "weights.pt").write_bytes(b"corrupted")
        stray = latest.parent / ".tmp_iter_000003"
        stray.mkdir()
        (stray / "weights.pt").write_bytes(b"partial")
        resumed = self.trainer(config, resume=trainer.run_dir)
        self.assertEqual(resumed.iteration, 1)
        self.assertFalse(stray.exists())
        row = [row for row in self.rows(resumed) if row["kind"] == "resume"][-1]
        self.assertIn("sha256", str(row["extra"]["skipped"]))
        self.assertIn(stray.name, str(row["extra"]["skipped"]))
        final = resumed.run()
        self.assertEqual(load_checkpoint(final).trainer_state["iteration"], 2)

    def test_replay_roundtrips_at_run_root(self):
        import numpy as np

        config = self.config(1)
        trainer = self.trainer(config)
        final = trainer.run()
        self.assertTrue((trainer.run_dir / "replay.npz").is_file())
        self.assertFalse((final / "replay.npz").exists())
        resumed = self.trainer(config, resume=trainer.run_dir)
        self.assertEqual(resumed.replay.total_added, trainer.replay.total_added)
        self.assertEqual(resumed.replay.position, trainer.replay.position)
        self.assertGreater(len(resumed.replay), 0)
        for key, value in trainer.replay.all().items():
            np.testing.assert_array_equal(resumed.replay.all()[key], value)

    def test_retention_keeps_recovered_checkpoint_with_newer_corrupt_generations(self):
        from hasami_learn.manifest import load_checkpoint

        config = self.config(3)
        trainer = self.trainer(config)
        latest = trainer.run()
        for iteration in (2, 3):
            (latest.parent / f"iter_{iteration:06d}" / "weights.pt").write_bytes(b"corrupt")
        config.run.keep_checkpoints = 1
        config.run.iterations = 2
        resumed = self.trainer(config, resume=trainer.run_dir)
        self.assertEqual(resumed.iteration, 1)
        recovered = resumed.run()
        self.assertTrue(recovered.is_dir())
        self.assertEqual(load_checkpoint(trainer.run_dir).trainer_state["iteration"], 2)

    def test_broken_pointer_and_missing_state_recover(self):
        config = self.config()
        trainer = self.trainer(config)
        latest = trainer.run()
        (latest.parent / "latest.json").write_text("{broken")
        resumed = self.trainer(config, resume=trainer.run_dir)
        self.assertEqual(resumed.iteration, 2)
        (latest / "trainer_state.json").unlink()
        resumed = self.trainer(config, resume=trainer.run_dir)
        self.assertEqual(resumed.iteration, 1)

    def test_auto_resume_fresh_start_cleans_first_unfinished_checkpoint(self):
        from hasami_learn.cli import main

        config = self.config(1)
        config_file = self.tmp / "config.json"
        config.save(config_file)
        run_dir = self.tmp / "runs" / config.run.run_id
        stray = run_dir / "checkpoints" / ".tmp_iter_000001"
        stray.mkdir(parents=True)
        self.assertEqual(main(["train", "--config", str(config_file), "--resume", "auto", "--quiet"]), 0)
        self.assertFalse(stray.exists())
        self.assertTrue((run_dir / "checkpoints" / "iter_000001" / "manifest.json").is_file())

    def test_time_checkpoint_and_zero_disables_timer(self):
        from hasami_learn.algorithms.base import CollectStats, UpdateStats

        for seconds, expected in ((1, [1, 2, 3]), (0, [3])):
            with self.subTest(seconds=seconds):
                config = self.config(3)
                config.run.run_id += f"-{seconds}"
                config.run.checkpoint_every = 100
                config.run.checkpoint_every_seconds = seconds
                trainer = self.trainer(config)
                clock = [0.0]

                def update(iteration):
                    clock[0] += 2
                    return UpdateStats()

                with patch.object(trainer.learner, "collect", return_value=CollectStats()), patch.object(
                    trainer.learner, "update", side_effect=update
                ), patch("hasami_learn.trainer.time.monotonic", side_effect=lambda: clock[0]):
                    trainer.run()
                actual = [row["iteration"] for row in self.rows(trainer) if row["kind"] == "checkpoint"]
                self.assertEqual(actual, expected)

    def test_signal_finishes_update_and_restores_handlers(self):
        from hasami_learn.algorithms.base import CollectStats, UpdateStats

        for sig in (signal.SIGTERM, signal.SIGINT):
            with self.subTest(signal=sig):
                config = self.config(5)
                config.run.run_id += f"-{sig.name}"
                trainer = self.trainer(config)
                previous = signal.getsignal(sig)

                def update(iteration):
                    os.kill(os.getpid(), sig)
                    return UpdateStats(steps=1)

                with patch.object(trainer.learner, "collect", return_value=CollectStats()), patch.object(
                    trainer.learner, "update", side_effect=update
                ) as updates:
                    final = trainer.run()
                updates.assert_called_once_with(1)
                self.assertEqual(final.name, "iter_000001")
                self.assertEqual(signal.getsignal(sig), previous)
                self.assertEqual(self.rows(trainer)[-1]["extra"]["stop_reason"], sig.name)

    def test_cli_rejects_conflicts_without_overwriting_saved_config(self):
        from hasami_learn.cli import build_parser, cmd_train
        from hasami_learn.config import ConfigError

        config = self.config(1)
        trainer = self.trainer(config)
        trainer.run()
        saved = trainer.run_dir / "config.json"
        before = saved.read_bytes()
        for override in (["--seed", "9"], ["--games", "9"], ["--iterations", "0"],
                         ["--set", "optimizer.lr=0.5"]):
            with self.subTest(override=override):
                args = build_parser().parse_args(["train", "--resume", str(trainer.run_dir), *override])
                with self.assertRaises(ConfigError):
                    cmd_train(args)
                self.assertEqual(saved.read_bytes(), before)

    def test_resume_uses_saved_config_and_accepts_only_allowed_changes(self):
        from hasami_learn.cli import build_parser, resolve_train_request
        from hasami_learn.config import ConfigError

        config = self.config(1)
        trainer = self.trainer(config)
        trainer.run()
        command = ["train", "--resume", str(trainer.run_dir)]
        resolved, resume = resolve_train_request(build_parser().parse_args(
            [*command, "--device", "cpu", "--iterations", "3", "--quiet"]))
        self.assertEqual(resolved.run.iterations, 3)
        self.assertEqual(resolved.algorithm.params, config.algorithm.params)
        self.assertEqual(resume, trainer.run_dir)
        changed = config.with_overrides(optimizer__lr=0.5)
        source = self.tmp / "conflict.json"
        changed.save(source)
        with self.assertRaisesRegex(ConfigError, "optimizer.lr"):
            resolve_train_request(build_parser().parse_args([*command, "--config", str(source)]))

    def test_handlers_restored_when_update_raises(self):
        from hasami_learn.algorithms.base import CollectStats

        trainer = self.trainer(self.config())
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        with patch.object(trainer.learner, "collect", return_value=CollectStats()), patch.object(
            trainer.learner, "update", side_effect=RuntimeError("update failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "update failed"):
                trainer.run()
        for sig, handler in previous.items():
            self.assertEqual(signal.getsignal(sig), handler)

    def test_supervisor_retries_with_capped_backoff_and_stops_on_completion(self):
        import textwrap

        config = self.config(1)
        config_file = self.tmp / "config.json"
        config.save(config_file)
        fake_python = self.tmp / "training-python"
        fake_python.write_text(f"#!{sys.executable}\n" + textwrap.dedent('''\
            import json
            import os
            import sys
            from pathlib import Path

            if sys.argv[1] != "-m":
                os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
            root = Path(os.environ["SUPERVISOR_TEST_ROOT"])
            counter = root / "attempts"
            attempt = int(counter.read_text()) + 1 if counter.exists() else 1
            counter.write_text(str(attempt))
            if attempt < 9:
                sys.exit(1)
            config = json.loads((root / "config.json").read_text())
            run = Path(config["run"]["output_dir"]) / config["run"]["run_id"]
            (run / "config.json").write_text(json.dumps(config))
            (run / "trainer_state.json").write_text(json.dumps({"iteration": 1}))
        '''))
        fake_python.chmod(0o755)
        fake_sleep = self.tmp / "sleep"
        fake_sleep.write_text('#!/bin/bash\nprintf "%s\\n" "$1" >> "${SUPERVISOR_TEST_ROOT}/delays"\n')
        fake_sleep.chmod(0o755)
        env = {**os.environ, "HASAMI_TRAIN_PYTHON": str(fake_python),
               "SUPERVISOR_TEST_ROOT": str(self.tmp), "PATH": f"{self.tmp}{os.pathsep}{os.environ['PATH']}"}
        result = subprocess.run(["bash", "scripts/train_supervisor.sh", str(config_file)],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.tmp / "attempts").read_text(), "9")
        self.assertEqual((self.tmp / "delays").read_text().splitlines(),
                         ["5", "10", "20", "40", "80", "160", "300", "300"])
        log = self.tmp / "runs" / config.run.run_id / "supervisor.log"
        self.assertIn("training complete", log.read_text())

    def test_subprocess_sigterm_and_auto_resume(self):
        self._subprocess_sigterm_and_auto_resume()

    def test_supervisor_forwards_sigterm_and_waits_for_checkpoint(self):
        self._subprocess_sigterm_and_auto_resume(supervisor=True)

    def _subprocess_sigterm_and_auto_resume(self, supervisor=False):
        from hasami_learn.manifest import load_checkpoint

        config = self.config(5)
        config.selfplay.max_plies = 40
        config.algorithm.params["simulations"] = 8
        config_file = self.tmp / "config.json"
        config.save(config_file)
        run_dir = self.tmp / "runs" / config.run.run_id
        command = [sys.executable, "-m", "hasami_learn.cli", "train", "--config", str(config_file),
                   "--resume", "auto", "--quiet"]
        launch = ["bash", "scripts/train_supervisor.sh", str(config_file), "--quiet"] if supervisor else command
        with (self.tmp / "child.log").open("w+") as log:
            child = subprocess.Popen(launch, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     env={**os.environ, "HASAMI_TRAIN_PYTHON": sys.executable})
            try:
                deadline = time.monotonic() + 90
                while not (run_dir / "checkpoints" / "latest.json").is_file():
                    self.assertIsNone(child.poll(), "training exited before its first checkpoint")
                    self.assertLess(time.monotonic(), deadline, "first checkpoint timed out")
                    time.sleep(0.01)
                child.send_signal(signal.SIGTERM)
                self.assertEqual(child.wait(timeout=90), 0)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()
            stopped = load_checkpoint(run_dir).trainer_state["iteration"]
            self.assertGreaterEqual(stopped, 1)
            self.assertLess(stopped, 5)
            rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
            self.assertTrue(any(row.get("extra", {}).get("stop_reason") == "SIGTERM" for row in rows))
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=90)
            log.seek(0)
            self.assertEqual(result.returncode, 0, log.read())
        self.assertEqual(load_checkpoint(run_dir).trainer_state["iteration"], 5)
        self.assertEqual(json.loads((run_dir / "trainer_state.json").read_text())["iteration"], 5)


import unittest  # noqa: E402  (regression block appended by the orchestrator)


class ReviewFindingRegressionTests(unittest.TestCase):
    """Regressions for the review findings on checkpoint recovery (interrupted replacement, legacy replay layout)."""

    def test_interrupted_replacement_restores_the_retired_checkpoint(self):
        import shutil
        import tempfile
        from pathlib import Path

        try:
            import torch  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("torch not installed")
        from hasami_learn.manifest import MANIFEST_FILE, ModelManifest, cleanup_checkpoint_temps, resolve_checkpoint_dir, save_checkpoint
        from hasami_learn.registry import NETWORKS
        import hasami_learn.networks  # noqa: F401

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            target = run / "checkpoints" / "iter_000001"
            network = NETWORKS.get("mlp")(head="policy_value", num_planes=9, hidden=8, layers=1)
            manifest = ModelManifest(algorithm="reinforce", network={"name": "mlp", "params": {"hidden": 8, "layers": 1}, "head": "policy_value"}, encoding_id="hasami-enc-v1", num_planes=9)
            save_checkpoint(target, network, manifest, {"optimizer": {}, "extra": {}}, {"iteration": 1})
            # Simulate a crash after the first rename: the committed directory was retired to a backup.
            backup = run / "checkpoints" / ".tmp_replaced_iter_000001"
            shutil.move(str(target), str(backup))
            unfinished = run / "checkpoints" / ".tmp_iter_000001"
            unfinished.mkdir()
            (unfinished / "weights.pt").write_bytes(b"partial")
            skipped = cleanup_checkpoint_temps(run)
            self.assertTrue((target / MANIFEST_FILE).is_file(), skipped)
            self.assertFalse(backup.exists())
            self.assertFalse(unfinished.exists())
            self.assertEqual(resolve_checkpoint_dir(target), target.resolve())

    def test_legacy_replay_inside_checkpoint_is_restored(self):
        import tempfile
        from pathlib import Path

        try:
            import torch  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("torch not installed")
        from hasami_learn.config import tiny_config
        from hasami_learn.trainer import Trainer, train_from_config

        with tempfile.TemporaryDirectory() as directory:
            config = tiny_config("alphazero", run_id="legacy-replay", simulations=4, batch_games=1)
            config.run.output_dir = str(Path(directory) / "runs")
            config.replay.persist = True
            final = train_from_config(config, echo=False)
            run_dir = final.parent.parent
            root_replay = run_dir / "replay.npz"
            self.assertTrue(root_replay.is_file())
            # Move the replay into the checkpoint directory to mimic the pre-change layout.
            root_replay.replace(final / "replay.npz")
            trainer = Trainer(config, resume=run_dir, echo=False)
            try:
                self.assertGreater(len(trainer.replay), 0)
            finally:
                trainer.close()


    def test_auto_resume_recovers_a_retired_only_checkpoint(self):
        """Review finding on checkpoint recovery: --resume auto must not start fresh when the only checkpoint is retired."""
        import json
        import shutil
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        try:
            import torch  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("torch not installed")
        from hasami_learn.config import tiny_config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = tiny_config("reinforce", run_id="retired-auto")
            config.run.iterations = 1
            config.run.keep_checkpoints = 1
            config.run.output_dir = str(root / "runs")
            config_path = root / "config.json"
            config.save(config_path)
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
            subprocess.run([sys.executable, "-m", "hasami_learn.cli", "train", "--config", str(config_path), "--quiet"], check=True, env=env)
            run_dir = root / "runs" / "retired-auto"
            committed = run_dir / "checkpoints" / "iter_000001"
            shutil.move(str(committed), str(run_dir / "checkpoints" / ".tmp_replaced_iter_000001"))
            (run_dir / "checkpoints" / ".tmp_iter_000001").mkdir()
            subprocess.run([sys.executable, "-m", "hasami_learn.cli", "train", "--config", str(config_path), "--resume", "auto", "--iterations", "2", "--quiet"], check=True, env=env)
            rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
            resumes = [row for row in rows if row["kind"] == "resume"]
            self.assertTrue(resumes, "auto resume did not resume")
            self.assertEqual(resumes[-1]["iteration"], 1)
            self.assertEqual([row["iteration"] for row in rows if row["kind"] == "collect"], [1, 2])
            self.assertTrue((run_dir / "checkpoints" / "iter_000002" / "manifest.json").is_file())
