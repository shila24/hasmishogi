"""Crash recovery must clean only owned staging files under an exclusive writer lock."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests.learn._common import LearnTestCase, ROOT, TORCH_AVAILABLE, requires_torch


@unittest.skipUnless(os.name == "posix", "writer path matrix requires POSIX links")
class WriterPathTests(unittest.TestCase):
    """Every path-taking writer must check the original path before deriving a lock."""

    def operations(self, root):
        from types import SimpleNamespace
        from hasami_learn import runlock
        from hasami_learn.progress import write_html
        from hasami_learn.checkpoint_bundle import bundle_checkpoint
        from hasami_learn.autoloop.plan import Plan
        from hasami_learn.autoloop.runner import Controller
        from hasami_learn.autoloop.stages import Stages, publish_checkpoint
        from hasami_learn.autoloop.state import StateStore, atomic_json, cleanup_json_temps, fsync_dir
        from hasami_learn.autoloop.__main__ import main as loop_main
        from scripts.kill_resume_smoke import main as kill_resume_main

        stage = {"id": "first", "kind": "module", "module": "hasami_learn.autoloop",
                 "args": [], "done_file": "done.json"}
        config = {"schema_version": 1, "loop_id": "test", "output_dir": str(root / "output"),
                  "require_validation": False, "cycles": 1, "stages": [stage]}
        plan_file = root / "plan.json"
        plan_file.write_text(json.dumps(config))

        def plan_output(path):
            plan_file.write_text(json.dumps({**config, "output_dir": str(path)}))
            try:
                return Plan(plan_file)
            finally:
                plan_file.write_text(json.dumps(config))

        def plan_at(path):
            plan = Plan(plan_file)
            plan.loop_dir = path
            return plan

        def plan_with_output(path):
            plan = Plan(plan_file)
            plan.output = path
            return plan

        def module_done(path):
            plan = Plan(plan_file)
            if not path.is_absolute():
                # Completion paths are stage-relative; keep the unnormalized
                # suffix while translating the fixture's cwd-relative prefix.
                path = Path(os.path.relpath(Path.cwd(), plan.stage_dir(stage, 1))) / path
            return Stages(plan).command({**stage, "done_file": str(path / "done.json")}, 1)

        def lock(path, filename="run.lock"):
            with runlock.run_lock(path, filename):
                pass

        def lock_filename(path):
            if not path.is_absolute():
                # Lock filenames are directory-relative, while this matrix supplies
                # cwd-relative paths. Preserve the suffix to exercise every link.
                path = Path(os.path.relpath(Path.cwd(), root)) / path
            return lock(root, str(path / "run.lock"))

        def acquire(path):
            store = StateStore(path)
            try:
                store.acquire()
            finally:
                store.close()

        def verify_run(path):
            with patch.object(sys, "argv", ["kill_resume_smoke.py", "--verify-run", str(path)]):
                return kill_resume_main()

        # Each tuple names an entry point, the linked component, the operation,
        # and whether its documented cleanup behavior is a harmless no-op.
        entries = [
            ("writer_path", "linked", runlock.writer_path, False),
            ("writer_directory", "linked", runlock.writer_directory, False),
            ("plain_directory", "linked", lambda p: self.assertFalse(runlock.plain_directory(p)), True),
            ("run_lock", "linked", lock, False),
            ("run_lock.filename", "linked", lock_filename, False),
            ("cleanup_temps", "linked", lambda p: runlock.cleanup_temps(p, files=(r"\.tmp_.*",)), True),
            ("cleanup_training_files", "linked", runlock.cleanup_training_files, False),
            ("cleanup_progress_files", "linked", runlock.cleanup_progress_files, True),
            ("cleanup_json_files", "linked", lambda p: runlock.cleanup_json_files(p, ("state.json",)), True),
            ("progress.write_html", "linked", lambda p: write_html(p / "report.html", "new"), False),
            ("bundle_checkpoint", "linked", lambda p: bundle_checkpoint(root / "source", p / "bundle"), False),
            ("publish_checkpoint", "linked", lambda p: publish_checkpoint(root / "source", p / "bundle"), False),
            ("atomic_json", "linked", lambda p: atomic_json(p / "state.json", {}), False),
            ("cleanup_json_temps", "linked", lambda p: cleanup_json_temps(p, ("state.json",)), True),
            ("fsync_dir", "linked", fsync_dir, False),
            ("StateStore.acquire", "linked", acquire, False),
            ("StateStore.save", "linked", lambda p: StateStore(p).save({}), False),
            ("Plan.output", "linked", plan_output, False),
            ("Plan.stage_dir", "linked", lambda p: plan_at(p).stage_dir(stage, 1), False),
            ("Controller.event", "linked", lambda p: Controller(plan_at(p)).event("probe"), False),
            ("Controller.run", "linked", lambda p: Controller(plan_at(p)).run(), False),
            ("Controller.execute", "linked", lambda p: Controller(plan_at(p)).execute(stage, 1), False),
            ("Stages.cleanup", "linked", lambda p: Stages(plan_at(p)).cleanup(stage, 1), False),
            ("Stages.cleanup_existing", "linked", lambda p: Stages(plan_at(p)).cleanup_existing(), False),
            ("Stages.cleanup_artifacts", "linked", lambda p: Stages(SimpleNamespace(output=p)).cleanup_artifacts(), False),
            ("Stages.local", "linked", lambda p: Stages(SimpleNamespace(output=p)).local(stage, 1), False),
            ("Stages.command", "linked", lambda p: Stages(plan_at(p)).command(stage, 1), False),
            ("Stages.command.done_file", "linked", module_done, False),
            ("Stages.command.output", "linked", lambda p: Stages(Plan(plan_file)).command({**stage, "args": ["--out", str(p / "out.json")]}, 1), False),
            ("Stages.finalize", "linked", lambda p: Stages(plan_at(p)).finalize({"kind": "evaluate", "id": "eval", "source": "first"}, 1), False),
            ("Stages.run_dir", "linked", lambda p: Stages(plan_with_output(p)).run_dir(stage, 1), False),
            ("Stages.release_dir", "linked", lambda p: Stages(plan_with_output(p)).release_dir(1), False),
            ("autoloop.stop", "linked", lambda p: loop_main(["stop", "--loop", str(p)]), False),
            ("kill_resume_smoke.verify_run", "linked", verify_run, False),
        ]
        if TORCH_AVAILABLE:
            from hasami_learn import manifest, cli
            from hasami_learn.config import tiny_config
            from hasami_learn.datastore import SelfplayStore
            from hasami_learn.replay import ReplayBuffer
            from hasami_learn.trainer import Trainer, train_from_config
            from hasami_learn.autoloop.evaluate import main as evaluate_main
            import numpy as np

            def train(path, factory):
                cfg = tiny_config("reinforce")
                cfg.run.output_dir, cfg.run.run_id = str(path.parent), path.name
                return factory(cfg, echo=False)

            def command(name, path):
                argv = {"train": ["train", "--algorithm", "reinforce", "--tiny", "--output-dir", str(path), "--run-id", "run", "--resume", "auto"],
                        "config": ["config-template", "--algorithm", "reinforce", "--tiny", "--out", str(path / "config.json")],
                        "export": ["export-agents-config", "--checkpoint", "unused", "--out", str(path / "agents.json")],
                        "eval": ["eval", "--checkpoint", "unused", "--out", str(path / "eval.json")],
                        "records": ["eval", "--checkpoint", "unused", "--records-dir", str(path)]}[name]
                args = cli.build_parser().parse_args(argv)
                return args.func(args)

            entries += [
                ("save_checkpoint.checkpoints", "checkpoints", lambda p: manifest.save_checkpoint(p / "iter_000001", None, None), False),
                ("save_checkpoint.pool", "pool", lambda p: manifest.save_checkpoint(p / "iter_000001", None, None), False),
                ("atomic_write_json", "linked", lambda p: manifest.atomic_write_json(p / "trainer_state.json", {}), False),
                ("write_latest_pointer", "checkpoints", lambda p: manifest.write_latest_pointer(p.parent, p / "iter_000001", 1), False),
                ("fsync_directory", "linked", manifest.fsync_directory, False),
                ("resume_run_dir", "linked", manifest.resume_run_dir, False),
                ("find_resume_checkpoint", "linked", manifest.find_resume_checkpoint, False),
                ("recover_retired_checkpoints", "linked", manifest.recover_retired_checkpoints, False),
                ("cleanup_checkpoint_temps", "linked", manifest.cleanup_checkpoint_temps, False),
                ("SelfplayStore", "linked", SelfplayStore, False),
                ("ReplayBuffer.save", "linked", lambda p: ReplayBuffer(1, {"value": ((), np.float32)}).save(p / "replay.npz"), False),
                ("Trainer", "linked", lambda p: train(p, Trainer), False),
                ("train_from_config", "linked", lambda p: train(p, train_from_config), False),
                ("autoloop.evaluate", "linked", lambda p: evaluate_main(["--checkpoint", "unused", "--out", str(p / "eval.json")]), False),
                ("cli.dataset", "linked", lambda p: cli.cmd_dataset(SimpleNamespace(rest=["--out", str(p / "dataset.npz")])), False),
                ("cli.dataset.games", "linked", lambda p: cli.cmd_dataset(SimpleNamespace(rest=["--out-games", str(p)])), False),
                *[("cli." + name, "linked", lambda p, name=name: command(name, p), False)
                  for name in ("train", "config", "export", "eval", "records")],
            ]
        return entries

    def test_all_writer_entry_points_reject_linked_destinations(self):
        for form in ("absolute", "relative"):
            for variant in ("leaf", "intermediate", "dotdot", "below_root_dotdot", "missing_prefix"):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    operations = self.operations(root)
                    for name, component, operation, no_op in operations:
                        with self.subTest(entry=name, form=form, variant=variant):
                            with tempfile.TemporaryDirectory(dir=root) as case:
                                base = Path(case)
                                actual = base / "other"
                                (actual / "child" / "nested").mkdir(parents=True)
                                owner = base / "owner"
                                owner.mkdir()
                                if variant == "below_root_dotdot":
                                    (owner / component).mkdir()
                                    (owner / component / "link").symlink_to(actual / "child", target_is_directory=True)
                                else:
                                    (owner / component).symlink_to(actual / "child", target_is_directory=True)
                                for directory in (actual, actual / "child", actual / "child" / "nested"):
                                    (directory / ".tmp_state.json_abcdefgh").write_text("keep")
                                before = {p.relative_to(actual): p.read_bytes() for p in actual.rglob("*") if p.is_file()}
                                original = owner if form == "absolute" else Path(os.path.relpath(owner))
                                if variant == "missing_prefix":
                                    original = original / "missing" / ".."
                                destination = original / component
                                if variant == "intermediate":
                                    destination /= "nested"
                                elif variant == "dotdot":
                                    destination /= ".."
                                elif variant == "below_root_dotdot":
                                    destination = destination / "link" / ".."
                                if no_op:
                                    operation(destination)
                                else:
                                    with self.assertRaisesRegex(RuntimeError, "symlink"):
                                        operation(destination)
                                after = {p.relative_to(actual): p.read_bytes() for p in actual.rglob("*") if p.is_file()}
                                self.assertEqual(after, before)
                                self.assertEqual(list(owner.iterdir()), [owner / component])

    def test_checkpoint_publication_receives_the_validated_path_and_matching_lock(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch/numpy not installed (use .venv)")
        from hasami_learn.manifest import save_checkpoint
        from hasami_learn.runlock import WriterBusyError, run_lock
        from concurrent.futures import ThreadPoolExecutor

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "checkpoints" / "child").mkdir(parents=True)
            original = Path(os.path.relpath(run)) / "checkpoints" / "child" / ".." / "iter_000001"

            def publication(target, *args, **kwargs):
                self.assertEqual(target, run / "checkpoints" / "iter_000001")

                def compete():
                    with self.assertRaises(WriterBusyError):
                        with run_lock(target.parent.parent):
                            self.fail("publication did not hold its destination's lock")

                with ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(compete).result()
                return target

            with patch("hasami_learn.manifest._save_checkpoint", side_effect=publication) as publish:
                self.assertEqual(save_checkpoint(original, None, None), run / "checkpoints" / "iter_000001")
                publish.assert_called_once()

    def test_best_alias_is_resolved_before_locking_and_rejects_other_links(self):
        from hasami_learn.autoloop.state import atomic_json
        from hasami_learn.runlock import writer_path

        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "models"
            candidate = models / "candidate"
            candidate.mkdir(parents=True)
            best = models / "best"
            best.symlink_to("candidate", target_is_directory=True)
            target = best / "eval.json"
            self.assertEqual(writer_path(target, allow_best_alias=True), candidate / "eval.json")
            with patch("hasami_learn.autoloop.state._atomic_json") as publish:
                atomic_json(target, {})
                publish.assert_called_once_with(candidate / "eval.json", {})
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                atomic_json(best / ".." / "eval.json", {})
            (candidate / "linked").symlink_to(Path(tmp), target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                atomic_json(best / "linked" / "eval.json", {})
            best.unlink()
            best.symlink_to("../elsewhere", target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                atomic_json(target, {})


class CleanupTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "symlink test requires POSIX")
    def test_directory_checks_reject_symlinks_hidden_by_dotdot(self):
        from hasami_learn.runlock import cleanup_temps, cleanup_training_files, plain_directory, run_lock

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            actual = root / "actual"
            (actual / "child").mkdir(parents=True)
            (root / "link").symlink_to(actual / "child", target_is_directory=True)
            (root / "dangling").symlink_to(root / "missing", target_is_directory=True)
            for directory in (root, actual):
                (directory / ".tmp_trainer_state.json_abcdefgh").write_text("keep")
            for link in ("link", "dangling"):
                alias = root / link / ".."
                for path in (alias, Path(os.path.relpath(root)) / link / ".."):
                    with self.subTest(path=path):
                        self.assertFalse(plain_directory(path))
                        with self.assertRaisesRegex(RuntimeError, "symlink"):
                            with run_lock(path):
                                self.fail("a symlink directory was locked")
                        self.assertEqual(cleanup_temps(path, files=(r"\.tmp_.*",)), [])
                        with self.assertRaisesRegex(RuntimeError, "symlink"):
                            cleanup_training_files(path)
                        for directory in (root, actual):
                            self.assertEqual((directory / ".tmp_trainer_state.json_abcdefgh").read_text(), "keep")
                            self.assertFalse((directory / "run.lock").exists())

    def test_plain_dotdot_directory_still_supports_cleanup(self):
        from hasami_learn.runlock import cleanup_training_files, plain_directory

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "child").mkdir()
            staging = root / ".tmp_trainer_state.json_abcdefgh"
            staging.write_text("partial")
            directory = root / "child" / ".."
            self.assertTrue(plain_directory(directory))
            cleanup_training_files(directory)
            self.assertFalse(staging.exists())

    def test_bundle_inventory_helpers_load_without_package_context(self):
        import importlib.util
        import hashlib

        spec = importlib.util.spec_from_file_location("standalone_bundle", ROOT / "hasami_learn/checkpoint_bundle.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "weights.pt").write_bytes(b"weights")
            self.assertEqual(module.file_hashes(tmp), {"weights.pt": hashlib.sha256(b"weights").hexdigest()})

    def test_fresh_run_cleans_disabled_store_staging_but_preserves_committed_data(self):
        from hasami_learn.runlock import cleanup_training_files

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            games = run / "selfplay/games"
            games.mkdir(parents=True)
            staging = games / ".selfplay-partial.tmp"
            staging.write_text("partial")
            committed = games / "000001_000000000000.npz"
            committed.write_text("committed")
            report = run / ".progress-abcdefgh.tmp"
            report.write_text("dead reporter without a writer lock")
            cleanup_training_files(run)
            self.assertFalse(staging.exists())
            self.assertEqual(committed.read_text(), "committed")
            self.assertFalse(report.exists())

    def test_lock_symlink_is_rejected_without_touching_target(self):
        from hasami_learn.runlock import cleanup_training_files

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            target = root / "important"
            target.write_text("keep")
            (run / "run.lock").symlink_to(target)
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                cleanup_training_files(run)
            self.assertEqual(target.read_text(), "keep")

    def test_known_files_only_and_no_symlink_traversal(self):
        from hasami_learn.runlock import cleanup_training_files

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            owned = run / ".tmp_trainer_state.json_abcdefgh"
            owned.write_text("partial")
            replay = run / ".tmp_replay_abcdefgh.npz"
            replay.write_text("partial")
            unknown = run / ".tmp_user_notes"
            unknown.write_text("keep")
            outside = root / "outside"
            outside.mkdir()
            protected = outside / ".tmp_latest.json_abcdefgh"
            protected.write_text("keep")
            (run / "checkpoints").symlink_to(outside, target_is_directory=True)
            link = run / ".tmp_config.json_abcdefgh"
            link.symlink_to(protected)
            cleanup_training_files(run)
            self.assertFalse(owned.exists())
            self.assertFalse(replay.exists())
            self.assertEqual(unknown.read_text(), "keep")
            self.assertEqual(protected.read_text(), "keep")
            self.assertTrue(link.is_symlink())

    def test_live_writer_prevents_cleanup_and_kernel_releases_lock(self):
        from hasami_learn.runlock import cleanup_training_files

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            temporary = root / ".tmp_trainer_state.json_abcdefgh"
            temporary.write_text("active")
            code = ("import sys; from hasami_learn.runlock import run_lock; "
                    "\nwith run_lock(sys.argv[1]):\n print('locked', flush=True)\n sys.stdin.read()\n")
            child = subprocess.Popen([sys.executable, "-c", code, tmp], cwd=ROOT,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(), "locked")
                with self.assertRaisesRegex(RuntimeError, "writer"):
                    cleanup_training_files(root)
                self.assertEqual(temporary.read_text(), "active")
            finally:
                child.kill()
                child.wait(timeout=10)
                child.stdin.close()
                child.stdout.close()
            cleanup_training_files(root)
            self.assertFalse(temporary.exists())


@requires_torch
class CheckpointBoundaryTests(LearnTestCase):
    def test_resume_rejects_symlink_dotdot_before_recovery(self):
        from hasami_learn.manifest import find_resume_checkpoint, resume_run_dir

        actual = self.tmp / "actual"
        (actual / "child").mkdir(parents=True)
        (self.tmp / "link").symlink_to(actual / "child", target_is_directory=True)
        staging = actual / ".tmp_trainer_state.json_abcdefgh"
        staging.write_text("keep")
        alias = self.tmp / "link" / ".."
        for path in (alias, Path(os.path.relpath(self.tmp)) / "link" / ".."):
            for operation in (resume_run_dir, find_resume_checkpoint):
                with self.subTest(path=path, operation=operation.__name__):
                    with self.assertRaisesRegex(RuntimeError, "symlink"):
                        operation(path)
                    self.assertEqual(staging.read_text(), "keep")
                    self.assertFalse((actual / "run.lock").exists())

    def config(self, algorithm, run_id):
        config = self.tiny(algorithm)
        config.run.run_id = run_id
        config.run.iterations = 3
        config.run.keep_checkpoints = 5
        config.run.checkpoint_every = 1
        config.run.eval_every = 0
        config.opponents.snapshot_every = 0
        config.selfplay.games_per_iteration = 1
        config.selfplay.max_plies = 6
        config.replay.capacity = 16
        config.replay.persist = True
        config.datastore.enabled = False
        if algorithm == "alphazero":
            config.algorithm.params.update(simulations=2, batch_games=1)
        return config

    def assert_state_equal(self, left, right):
        import numpy as np
        import torch

        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(left, right)

    def snapshot(self, trainer):
        import copy
        import torch

        return copy.deepcopy({"network": trainer.network.state_dict(),
                              "learner": trainer.learner.state_dict(),
                              "rng": trainer.rng.bit_generator.state,
                              "torch_rng": torch.get_rng_state(),
                              "replay": None if trainer.replay is None else
                              (trainer.replay.all(), trainer.replay.position, trainer.replay.total_added)})

    def test_retired_generation_recovers_before_staging_cleanup(self):
        from hasami_learn.manifest import cleanup_checkpoint_temps, load_checkpoint
        from hasami_learn.trainer import train_from_config

        cfg = self.config("reinforce", "retired")
        target = train_from_config(cfg, echo=False)
        run = target.parent.parent
        backup = target.with_name(".tmp_replaced_" + target.name)
        target.rename(backup)
        unfinished = target.with_name(".tmp_" + target.name)
        unfinished.mkdir()
        (unfinished / "weights.pt").write_bytes(b"partial")
        unknown = target.parent / ".tmp_user_work"
        unknown.mkdir()
        outside = self.tmp / "outside"
        outside.mkdir()
        sentinel = outside / "sentinel"
        sentinel.write_text("keep")
        (target.parent / ".tmp_iter_999999").symlink_to(outside, target_is_directory=True)
        cleanup_checkpoint_temps(run)
        self.assertEqual(load_checkpoint(target).manifest.iteration, 3)
        self.assertFalse(backup.exists())
        self.assertFalse(unfinished.exists())
        self.assertTrue(unknown.is_dir())
        self.assertEqual(sentinel.read_text(), "keep")
        self.assertTrue((target.parent / ".tmp_iter_999999").is_symlink())

    def test_corrupt_snapshot_falls_back_to_matching_replay_and_explicit_selection(self):
        from hasami_learn.trainer import Trainer

        cfg = self.config("alphazero", "fallback")
        trainer = Trainer(cfg, echo=False)
        try:
            final = trainer.run()
        finally:
            trainer.close()
        (final / "replay_snapshot.npz").write_bytes(b"corrupt")
        second = final.parent / "iter_000002"
        resumed = Trainer(cfg, resume=trainer.run_dir, echo=False)
        try:
            self.assertEqual(resumed.iteration, 2)
            import numpy as np
            with np.load(second / "replay_snapshot.npz") as data:
                self.assertEqual(resumed.replay.total_added, int(data["__total"]))
        finally:
            resumed.close()
        explicit = Trainer(cfg, resume=final.parent / "iter_000001", echo=False)
        try:
            self.assertEqual(explicit.iteration, 1)
        finally:
            explicit.close()

    def test_initialized_trainer_cannot_overwrite_newer_writer(self):
        from hasami_learn.trainer import Trainer, train_from_config

        cfg = self.config("reinforce", "stale-trainer")
        trainer = Trainer(cfg, echo=False)
        try:
            train_from_config(cfg, echo=False)
            with self.assertRaisesRegex(RuntimeError, "run changed"):
                trainer.run()
        finally:
            trainer.close()

    def test_recovery_preserves_only_good_backup_when_committed_state_is_broken(self):
        import shutil
        from hasami_learn.manifest import cleanup_checkpoint_temps, find_resume_checkpoint
        from hasami_learn.trainer import train_from_config

        target = train_from_config(self.config("alphazero", "backup-state"), echo=False)
        backup = target.with_name(".tmp_replaced_" + target.name)
        shutil.copytree(target, backup)
        (target / "replay_snapshot.npz").write_bytes(b"broken")
        cleanup_checkpoint_temps(target.parent.parent)
        loaded, _ = find_resume_checkpoint(target.parent.parent)
        self.assertEqual(loaded.manifest.iteration, 3)
        self.assertFalse(backup.exists())
        self.assertNotEqual((target / "replay_snapshot.npz").read_bytes(), b"broken")

    def test_live_training_rejects_competing_cli_before_cleanup(self):
        from hasami_learn.trainer import Trainer

        cfg = self.config("reinforce", "live-writer")
        cfg.run.iterations = 1
        trainer = Trainer(cfg, echo=False)
        attempts = []

        def compete(name, iteration):
            if name != "during_trainer_state_write":
                return
            staging = list(trainer.run_dir.glob(".tmp_trainer_state.json_*"))
            self.assertEqual(len(staging), 1)
            child = subprocess.run([sys.executable, "-m", "hasami_learn.cli", "train",
                                    "--config", str(trainer.run_dir / "config.json"),
                                    "--resume", "auto", "--quiet"], cwd=ROOT,
                                   capture_output=True, text=True, timeout=30)
            self.assertEqual(child.returncode, 1, child.stdout + child.stderr)
            self.assertIn("another writer", child.stderr)
            self.assertTrue(staging[0].is_file())
            attempts.append(iteration)

        try:
            with patch("hasami_learn.runlock.checkpoint_boundary", side_effect=compete):
                trainer.run()
        finally:
            trainer.close()
        self.assertEqual(attempts, [1])

    def test_all_checkpoint_boundaries_resume_without_duplicate_or_skipped_updates(self):
        from hasami_learn.trainer import Trainer

        class Killed(BaseException):
            pass

        for algorithm in ("alphazero", "reinforce"):
            baseline = Trainer(self.config(algorithm, algorithm + "-baseline"), echo=False)
            try:
                baseline.run()
                expected = self.snapshot(baseline)
            finally:
                baseline.close()
            for boundary in ("after_checkpoint_publish", "during_trainer_state_write",
                             "during_replay_write", "before_metrics_append"):
                with self.subTest(algorithm=algorithm, boundary=boundary):
                    cfg = self.config(algorithm, algorithm + "-" + boundary)
                    trainer = Trainer(cfg, echo=False)
                    captured = []
                    leftovers = []
                    unlink = Path.unlink

                    def keep_unpublished(path, *args, **kwargs):
                        if captured and path.name.startswith(".tmp_"):
                            return None  # SIGKILL does not execute finally blocks.
                        return unlink(path, *args, **kwargs)

                    def fault(name, iteration):
                        if name == boundary and iteration == 2:
                            captured.append(self.snapshot(trainer))
                            leftovers.extend(trainer.run_dir.glob(".tmp_*"))
                            raise Killed()

                    try:
                        with patch("hasami_learn.runlock.checkpoint_boundary", side_effect=fault), \
                                patch.object(Path, "unlink", keep_unpublished):
                            with self.assertRaises(Killed):
                                trainer.run()
                    finally:
                        trainer.close()
                    self.assertEqual(len(captured), 1)
                    if boundary == "during_trainer_state_write" or (boundary == "during_replay_write" and algorithm == "alphazero"):
                        self.assertTrue(leftovers)
                        self.assertTrue(all(path.exists() for path in leftovers))
                    report_temp = trainer.run_dir / ".progress-abcdefgh.tmp"
                    report_temp.write_text("dead reporter")
                    resumed = Trainer(cfg, resume=trainer.run_dir, echo=False)
                    try:
                        self.assertFalse(report_temp.exists())
                        self.assertEqual(resumed.iteration, 2)
                        self.assert_state_equal(self.snapshot(resumed), captured[0])
                        resumed.run()
                        self.assert_state_equal(self.snapshot(resumed), expected)
                    finally:
                        resumed.close()
                    rows = [json.loads(line) for line in (trainer.run_dir / "metrics.jsonl").read_text().splitlines()]
                    for kind in ("collect", "update"):
                        self.assertEqual([row["iteration"] for row in rows if row["kind"] == kind], [1, 2, 3])
                    self.assertEqual([row["iteration"] for row in rows if row["kind"] == "checkpoint"], [1, 3])
                    self.assertEqual(list(trainer.run_dir.rglob(".tmp_*")), [])


if __name__ == "__main__":
    unittest.main()
