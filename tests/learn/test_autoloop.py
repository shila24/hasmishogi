"""Autoloop section 7: deterministic failures and real process recovery."""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hasami_learn.autoloop.health import SuspendDetector, probe
from hasami_learn.autoloop.plan import Plan, ROOT, validation_gate
from hasami_learn.autoloop.runner import Controller
from hasami_learn.autoloop.stages import Stages, checkpoint, verify_release
from hasami_learn.autoloop.state import LoopError, StateStore, atomic_json, read_json, validate
from tests.learn._common import requires_torch

FAKES = Path(__file__).with_name("autoloop_fakes")


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.jump = 0
        self.sleeps = []
        self.hook = lambda: None

    def wall(self): return self.now + self.jump
    def monotonic(self): return self.now
    def boot(self): return self.now + self.jump
    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds
        self.hook()


class FakeProcess:
    pid = 99999999
    stdin = None

    def __init__(self, ticks=1, code=0, finish=lambda: None):
        self.ticks, self.code, self.finish = ticks, code, finish
        self.finished = False

    def poll(self):
        if self.ticks > 0:
            self.ticks -= 1
            return None
        if not self.finished:
            self.finish()
            self.finished = True
        return self.code

    def wait(self):
        self.ticks = 0
        return self.code


class AutoloopTests(unittest.TestCase):
    def test_plan_and_control_reject_symlink_dotdot_output_paths(self):
        from hasami_learn.autoloop.__main__ import main

        actual = self.root / "actual"
        (actual / "child").mkdir(parents=True)
        (self.root / "link").symlink_to(actual / "child", target_is_directory=True)
        alias = self.root / "link" / ".."
        self.data["output_dir"] = str(alias)
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            self.write()
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            main(["stop", "--loop", str(alias)])
        self.assertFalse((actual / "control").exists())

    def test_symlink_dotdot_rejected_by_json_and_checkpoint_publication(self):
        from hasami_learn.autoloop.stages import publish_checkpoint
        from hasami_learn.autoloop.state import cleanup_json_temps
        from hasami_learn.checkpoint_bundle import bundle_checkpoint

        actual = self.root / "actual"
        (actual / "child").mkdir(parents=True)
        (self.root / "link").symlink_to(actual / "child", target_is_directory=True)
        alias = self.root / "link" / ".."
        for directory in (self.root, actual):
            (directory / "status.json").write_text("previous")
            (directory / ".tmp_status.json_abcdefgh").write_text("keep")
        for directory in (alias, Path(os.path.relpath(self.root)) / "link" / ".."):
            with self.subTest(directory=directory):
                with self.assertRaisesRegex(RuntimeError, "symlink"):
                    atomic_json(directory / "status.json", {"new": True})
                self.assertEqual(cleanup_json_temps(directory, ("status.json",)), [])
                for publish in (bundle_checkpoint, publish_checkpoint):
                    with self.subTest(publish=publish.__name__):
                        with self.assertRaisesRegex(RuntimeError, "symlink"):
                            publish(self.root / "unused-source", directory / "published")
                for parent in (self.root, actual):
                    self.assertEqual((parent / "status.json").read_text(), "previous")
                    self.assertEqual((parent / ".tmp_status.json_abcdefgh").read_text(), "keep")
                    self.assertFalse((parent / ".publish.lock").exists())

    def test_parallel_json_writers_serialize_without_rejecting_independent_files(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier

        barrier = Barrier(8)

        def write(index):
            barrier.wait(timeout=10)
            for turn in range(8):
                atomic_json(self.root / f"worker-{index}.json", {"turn": turn})

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(write, range(8)))
        for index in range(8):
            self.assertEqual(read_json(self.root / f"worker-{index}.json"), {"turn": 7})
        self.assertEqual(list(self.root.glob('.tmp_*')), [])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hasami-autoloop-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "plan.json"
        self.data = {"schema_version": 1, "loop_id": "test", "output_dir": str(self.root / "output"),
                     "require_validation": False, "cycles": 1,
                     "limits": {"min_free_gb": 0, "heartbeat_s": 0.02, "backoff_seconds": [0.02, 0.08],
                                "stall_timeout_s": 1, "stop_grace_s": 0.1, "max_attempts": 3},
                     "stages": [self.module("first")]}
        self.write()

    @staticmethod
    def module(name, mode="done"):
        return {"id": name, "kind": "module", "module": "hasami_learn.autoloop", "args": [mode], "done_file": "done.json"}

    def write(self):
        atomic_json(self.path, self.data)
        return Plan(self.path)

    def controller(self, **kwargs):
        return Controller(self.write(), **kwargs)

    def state(self):
        return read_json(self.root / "output/test/loop_state.json")

    def events(self):
        return [json.loads(line) for line in (self.root / "output/test/events.jsonl").read_text().splitlines()]

    def test_startup_cleans_owned_temps_only_after_guardian_lock_is_free(self):
        plan = self.write()
        stage = plan.stage_dir(plan.stage(0, 1), 1)
        models = plan.output / 'models'
        release = plan.output / 'release'
        control = plan.loop_dir / 'control'
        for directory in (stage, models, release, control):
            directory.mkdir(parents=True, exist_ok=True)
        files = [plan.loop_dir / '.tmp_loop_state.json_abcdefgh',
                 plan.loop_dir / '.tmp_status.json_abcdefgh',
                 control / '.tmp_stop_abcdefgh', stage / '.tmp_abcdefgh',
                 stage / '.tmp_done.json_abcdefgh']
        for path in files:
            path.write_text('interrupted')
        scratch = [models / '.tmp_abcdefgh', release / '.tmp_abcdefgh']
        for path in scratch:
            path.mkdir()
            (path / 'partial').write_text('interrupted')
        outside = self.root / 'outside'
        outside.mkdir()
        protected = outside / '.tmp_abcdefgh'
        protected.write_text('keep')
        (stage / '.tmp_eval.json_abcdefgh').symlink_to(protected)
        pointer = models / ('.best-' + 'a' * 32)
        pointer.symlink_to(outside, target_is_directory=True)
        unknown = models / '.tmp_user_work'
        unknown.mkdir()
        backup = plan.loop_dir / 'loop_state.json.bak'
        backup.write_text('preserve recovery evidence')
        guardian = StateStore(plan.loop_dir)
        self.assertTrue(guardian.acquire())
        try:
            self.assertEqual(Controller(plan).run(), 3)
            self.assertTrue(all(path.exists() for path in files + scratch))
        finally:
            guardian.close()
        controller = Controller(plan)
        with patch.object(controller.store, 'load', return_value=None), patch.object(controller, '_run', return_value=0):
            self.assertEqual(controller.run(), 0)
        self.assertTrue(all(not path.exists() for path in files + scratch))
        self.assertFalse(pointer.is_symlink())
        self.assertTrue(unknown.is_dir())
        self.assertEqual(protected.read_text(), 'keep')
        self.assertEqual(backup.read_text(), 'preserve recovery evidence')
        self.assertTrue((stage / '.tmp_eval.json_abcdefgh').is_symlink())

    def test_stage_retry_cleans_temporary_created_by_previous_attempt(self):
        plan = self.write()
        stage = plan.stage(0, 1)
        directory = plan.stage_dir(stage, 1)
        directory.mkdir(parents=True)
        temporary = directory / '.tmp_done.json_abcdefgh'
        attempts = []
        controller = Controller(plan)

        def execute(*_):
            self.assertFalse(temporary.exists())
            attempts.append(1)
            if len(attempts) == 1:
                temporary.write_text('interrupted')
                return 1, False
            atomic_json(directory / 'done.json', {'status': 'complete'})
            return 0, False

        with patch.object(controller, 'execute', side_effect=execute):
            self.assertEqual(controller.run(), 0)
        self.assertEqual(len(attempts), 2)
        self.assertFalse(temporary.exists())

    def test_module_report_temps_match_repository_relative_output_arguments(self):
        plan = self.write()
        directory = plan.stage_dir(plan.stage(0, 1), 1)
        directory.mkdir(parents=True)
        report = directory / 'results.json'
        self.data['stages'][0]['args'] = ['--out', os.path.relpath(report, ROOT)]
        plan = self.write()
        for name in ('.tmp_results.json_abcdefgh', '.tmp_results.done.json_abcdefgh'):
            (directory / name).write_text('interrupted')
        stages = Stages(plan)
        stages.cleanup(plan.stage(0, 1), 1)
        self.assertEqual(list(directory.glob('.tmp_*')), [])

    def test_nested_module_outputs_recover_on_startup_and_retry(self):
        self.data['stages'][0]['done_file'] = 'reports/results.done.json'
        plan = self.write()
        directory = plan.stage_dir(plan.stage(0, 1), 1)
        report = directory / 'reports/results.json'
        self.data['stages'][0]['args'] = ['--out', os.path.relpath(report, ROOT)]
        plan = self.write()
        report.parent.mkdir(parents=True)
        temporary = report.with_name('.tmp_results.json_abcdefgh')
        done_temp = report.with_name('.tmp_results.done.json_abcdefgh')
        for path in (temporary, done_temp):
            path.write_text('interrupted before startup')
        attempts = []
        controller = Controller(plan)

        def execute(*_):
            self.assertFalse(temporary.exists())
            self.assertFalse(done_temp.exists())
            attempts.append(1)
            if len(attempts) == 1:
                # Preserve a real writer temporary as SIGKILL would.
                with patch('hasami_learn.autoloop.state.os.replace', side_effect=RuntimeError('killed')), \
                        patch.object(Path, 'unlink'):
                    with self.assertRaisesRegex(RuntimeError, 'killed'):
                        atomic_json(report, {'partial': True})
                self.assertTrue(list(report.parent.glob('.tmp_*')))
                return 1, False
            self.assertEqual(list(report.parent.glob('.tmp_*')), [])
            atomic_json(report, {'result': 'complete'})
            atomic_json(report.with_suffix('.done.json'), {'status': 'complete'})
            return 0, False

        with patch.object(controller, 'execute', side_effect=execute):
            self.assertEqual(controller.run(), 0)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(list(directory.rglob('.tmp_*')), [])

    def test_command_and_cleanup_share_all_output_argument_forms(self):
        plan = self.write()
        stage = plan.stage(0, 1)
        directory = plan.stage_dir(stage, 1)
        outputs = [directory / f'reports {index}/result={index}.json' for index in range(4)]
        values = [str(outputs[0]), os.path.relpath(outputs[1], ROOT),
                  '~/' + os.path.relpath(outputs[2], Path.home()), str(outputs[3])]
        stage = {**stage, 'args': ['--out', values[0], '--out=' + values[1],
                                   '--output', values[2], '--output=' + values[3]]}
        stages = Stages(plan)
        self.assertEqual(stages.command(stage, 1)[3:],
                         ['--out', str(outputs[0]), '--out=' + str(outputs[1]),
                          '--output', str(outputs[2]), '--output=' + str(outputs[3])])
        for output in outputs:
            output.parent.mkdir(parents=True)
            for path in (output, output.with_suffix('.done.json')):
                path.with_name('.tmp_' + path.name + '_abcdefgh').write_text('interrupted')
            (output.parent / '.tmp_unrelated.json_abcdefgh').write_text('keep')
        stages.cleanup(stage, 1)
        self.assertEqual(sorted(p.name for p in directory.rglob('.tmp_*')),
                         ['.tmp_unrelated.json_abcdefgh'] * len(outputs))

    def test_nested_output_cleanup_preserves_undeclared_paths_and_links(self):
        plan = self.write()
        directory = plan.stage_dir(plan.stage(0, 1), 1)
        outside = self.root / 'outside'
        outside.mkdir()
        directory.mkdir(parents=True)
        (directory / 'linked').symlink_to(outside, target_is_directory=True)
        unknown = directory / 'undeclared'
        unknown.mkdir()
        protected = [outside / '.tmp_results.json_abcdefgh', unknown / '.tmp_results.json_abcdefgh']
        for path in protected:
            path.write_text('keep')
        for output in (directory / 'linked/results.json', directory / '../outside/results.json',
                       outside / 'results.json', directory / 'linked/../undeclared/results.json'):
            with self.subTest(output=output):
                self.data['stages'][0].update(done_file=str(output), args=['--output', str(output)])
                plan = self.write()
                Stages(plan).cleanup(plan.stage(0, 1), 1)
                self.assertTrue(all(path.read_text() == 'keep' for path in protected))
        self.data['stages'][0]['args'] = [
            '--output', os.path.relpath(directory, ROOT) + '/linked/../undeclared/results.json']
        plan = self.write()
        Stages(plan).cleanup(plan.stage(0, 1), 1)
        self.assertTrue(all(path.read_text() == 'keep' for path in protected))

    def test_live_standalone_bundle_excludes_cleanup_and_publication(self):
        from hasami_learn.autoloop.stages import publish_checkpoint

        plan, stages, cp = self.bundle_plan()
        code = '''import sys
from hasami_learn import checkpoint_bundle as bundle
copy = bundle._copy_files
def pause(*args, **kwargs):
    print('staged', flush=True)
    sys.stdin.readline()
    return copy(*args, **kwargs)
bundle._copy_files = pause
bundle.bundle_checkpoint(sys.argv[1], sys.argv[2])
'''
        for parent in ('models', 'release'):
            for operation in ('cleanup', 'publish'):
                with self.subTest(parent=parent, operation=operation):
                    target = plan.output / parent / ('baseline' if operation == 'cleanup' else 'snapshot')
                    child = subprocess.Popen([sys.executable, '-c', code, str(cp), str(target)], cwd=ROOT,
                                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, text=True)
                    try:
                        self.assertEqual(child.stdout.readline().strip(), 'staged')
                        staging = target.with_name('.tmp_' + target.name)
                        self.assertTrue(staging.is_dir())
                        with self.assertRaisesRegex(RuntimeError, 'another writer'):
                            if operation == 'cleanup':
                                stages.cleanup_artifacts()
                            else:
                                publish_checkpoint(cp, target.with_name('published'))
                        self.assertTrue(staging.is_dir())
                        _, error = child.communicate('\n', timeout=15)
                        self.assertEqual(child.returncode, 0, error)
                        self.assertEqual((target / 'weights.pt').read_bytes(), (cp / 'weights.pt').read_bytes())
                    finally:
                        if child.poll() is None:
                            child.kill()
                        child.communicate(timeout=15)
                    stages.cleanup_artifacts()
                    self.assertFalse(staging.exists())

    def test_nested_output_cleanup_excludes_live_json_writer(self):
        self.data['stages'][0]['done_file'] = 'reports/done.json'
        plan = self.write()
        stage = plan.stage(0, 1)
        output = Stages(plan).done_file(stage, 1)
        code = '''import sys
from hasami_learn.autoloop import state
def pause(*args):
    print('staged', flush=True)
    sys.stdin.read()
state.os.replace = pause
state.atomic_json(sys.argv[1], {'status': 'complete'})
'''
        child = subprocess.Popen([sys.executable, '-c', code, str(output)], cwd=ROOT,
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), 'staged')
            temporary = list(output.parent.glob('.tmp_*'))
            self.assertEqual(len(temporary), 1)
            with self.assertRaisesRegex(RuntimeError, 'another writer'):
                Stages(plan).cleanup(stage, 1)
            self.assertTrue(temporary[0].exists())
        finally:
            child.kill()
            child.communicate(timeout=10)
        Stages(plan).cleanup(stage, 1)
        self.assertFalse(temporary[0].exists())

    @requires_torch
    def test_manifest_module_writers_exclude_stage_and_training_cleanup(self):
        from concurrent.futures import ThreadPoolExecutor
        from hasami_learn import manifest, practical_calibrate, practical_eval
        from hasami_learn.runlock import WriterBusyError, cleanup_training_files

        plan = self.write()
        stage = plan.stage(0, 1)
        directory = plan.stage_dir(stage, 1)
        report = directory / 'reports/results.json'
        stage = {**stage, 'args': ['--out', str(report)]}
        stages = Stages(plan)
        run = self.root / 'training'
        cases = [(report, lambda: stages.cleanup(stage, 1)),
                 (report.with_suffix('.done.json'), lambda: stages.cleanup(stage, 1))]
        cases += [(run / name, lambda: cleanup_training_files(run)) for name in
                  ('config.json', 'trainer_state.json', 'smoke-levels.json', 'checkpoints/latest.json')]
        replace = os.replace
        # Use the exact writer imported by both practical modules, with real
        # kernel locks in a competing thread at the pre-publication boundary.
        for writer in (atomic_json, practical_eval.atomic_write_json, practical_calibrate.atomic_write_json):
            for target, cleanup in cases:
                with self.subTest(writer=writer.__module__, target=target):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text('previous')

                    def before_replace(source, destination):
                        self.assertEqual(Path(destination), target)
                        self.assertTrue(Path(source).is_file())
                        with ThreadPoolExecutor(max_workers=1) as pool:
                            with self.assertRaises(WriterBusyError):
                                pool.submit(cleanup).result(timeout=10)
                        self.assertTrue(Path(source).is_file())
                        self.assertEqual(target.read_text(), 'previous')
                        return replace(source, destination)

                    with patch.object(manifest.os, 'replace', side_effect=before_replace):
                        writer(target, {'status': 'complete'})
                    cleanup()
                    self.assertEqual(read_json(target), {'status': 'complete'})
                    self.assertEqual(list(target.parent.glob('.tmp_*')), [])

    @requires_torch
    def test_module_output_forms_recover_real_practical_reports(self):
        from hasami_learn.config import tiny_config
        from hasami_learn.practical_eval import main as evaluate
        from hasami_learn.trainer import train_from_config

        config = tiny_config('reinforce')
        config.run.output_dir = str(self.root / 'training')
        config.run.iterations = 1
        config.run.eval_every = 0
        config.selfplay.games_per_iteration = 1
        config.selfplay.max_plies = 2
        checkpoint = train_from_config(config, echo=False)
        replace, unlink = os.replace, Path.unlink

        for flag in ('--out', '--output'):
            for equals in (False, True):
                for relative in (False, True):
                    with self.subTest(flag=flag, equals=equals, relative=relative):
                        self.data['loop_id'] = f'test-{flag[2:]}-{int(equals)}-{int(relative)}'
                        plan = self.write()
                        directory = plan.stage_dir(plan.stage(0, 1), 1)
                        report = directory / 'reports/results.json'
                        value = os.path.relpath(report, ROOT) if relative else str(report)
                        output_args = [flag + '=' + value] if equals else [flag, value]
                        self.data['stages'][0].update(
                            module='hasami_learn.practical_eval', done_file='done.json',
                            args=['E7', '--checkpoint', str(checkpoint), '--depth', '1',
                                  '--positions', '1', '--rollouts', '1', '--max-plies', '2',
                                  '--level', '1', '--smoke', '--backend', 'python', *output_args])
                        plan = self.write()
                        command = Stages(plan).command(plan.stage(0, 1), 1)

                        def kill_report(source, destination):
                            if Path(destination) == report:
                                raise RuntimeError('injected kill before report publication')
                            return replace(source, destination)

                        def keep_report_temp(path, *args, **kwargs):
                            if path.parent == report.parent and path.name.startswith('.tmp_results.json_'):
                                return None  # SIGKILL does not execute the writer's finally.
                            return unlink(path, *args, **kwargs)

                        def interrupted():
                            with patch('hasami_learn.manifest.os.replace', side_effect=kill_report), \
                                    patch.object(Path, 'unlink', keep_report_temp):
                                with self.assertRaisesRegex(RuntimeError, 'injected kill'):
                                    evaluate(command[3:])
                            self.assertEqual(len(list(report.parent.glob('.tmp_results.json_*'))), 1)

                        interrupted()
                        attempts = []
                        controller = Controller(plan, clock=FakeClock())

                        def execute(*_):
                            self.assertEqual(list(report.parent.glob('.tmp_*')), [])
                            attempts.append(1)
                            if len(attempts) == 1:
                                interrupted()
                                return 1, False
                            self.assertEqual(evaluate(command[3:]), 0)
                            atomic_json(directory / 'done.json', {'status': 'complete'})
                            return 0, False

                        with patch.object(controller, 'execute', side_effect=execute):
                            self.assertEqual(controller.run(), 0)
                        self.assertEqual(len(attempts), 2)
                        self.assertEqual(read_json(report)['status'], 'complete')
                        self.assertEqual(read_json(report.with_suffix('.done.json')), read_json(report))
                        self.assertEqual(list(directory.rglob('.tmp_*')), [])

    def launch(self, command=None):
        log = (self.root / "process.log").open("a+")
        self.addCleanup(log.close)
        child = subprocess.Popen(command or [sys.executable, str(FAKES / "controller.py"), str(self.path)],
                                 cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        self.addCleanup(self.reap, child)
        return child

    @staticmethod
    def reap(child):
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()

    def await_file(self, path, child, timeout=15):
        end = time.monotonic() + timeout
        while not path.exists():
            self.assertIsNone(child.poll(), (self.root / "process.log").read_text())
            self.assertLess(time.monotonic(), end, str(path))
            time.sleep(0.01)

    def test_config_schema_strict_and_missing_plans_warn_only(self):
        for mutation in [lambda d: d.update(cycles=0), lambda d: d.update(cycles=True),
                         lambda d: d.update(output_dir="/mnt/c/data"), lambda d: d.update(unknown=1),
                         lambda d: d['stages'][0].update(module='os'), lambda d: d['limits'].update(backoff_seconds=[2,1])]:
            saved = json.loads(json.dumps(self.data))
            mutation(self.data)
            with self.assertRaises(LoopError): self.write()
            self.data = saved
        plan = Plan(ROOT / "configs/autoloop/shogi.json")
        self.assertTrue(plan.warnings)
        with self.assertRaisesRegex(LoopError, "missing"):
            Plan(plan.path, running=True)

    def test_controller_imports_no_torch(self):
        result = subprocess.run([sys.executable, '-c', 'import sys; import hasami_learn.autoloop.runner; assert "torch" not in sys.modules; assert "numpy" not in sys.modules'], cwd=ROOT, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_completed_status_reports_last_cycle_and_no_current_stage(self):
        for cycles in (1, 3):
            with self.subTest(cycles=cycles):
                self.data['cycles'] = cycles
                self.data['loop_id'] = f'complete-{cycles}'
                plan = self.write()

                def launch(*args, **kwargs):
                    done = kwargs['env']['HASAMI_AUTOLOOP_DONE_FILE']
                    return FakeProcess(finish=lambda: atomic_json(done, {'status': 'complete'}))

                for restart in (False, True):
                    with self.subTest(restart=restart):
                        controller = Controller(plan, clock=FakeClock(), launcher=launch)
                        self.assertEqual(controller.run(), 0)
                        status = read_json(plan.loop_dir / 'status.json')
                        self.assertEqual(status['status'], 'complete')
                        self.assertEqual(status['cycle'], cycles)
                        self.assertIsNone(status['stage'])
                        self.assertIsNone(status['run_id'])
                        command = [sys.executable, '-m', 'hasami_learn.autoloop',
                                   'status', '--loop', str(plan.loop_dir)]
                        text = subprocess.run(command, cwd=ROOT, capture_output=True,
                                              text=True, timeout=10)
                        self.assertEqual(text.returncode, 0, text.stderr)
                        self.assertEqual(text.stdout.strip(),
                                         f'complete cycle={cycles} stage=None iteration=None')
                        structured = subprocess.run([*command, '--json'], cwd=ROOT,
                                                    capture_output=True, text=True, timeout=10)
                        self.assertEqual(structured.returncode, 0, structured.stderr)
                        self.assertEqual(json.loads(structured.stdout), status)

    def test_packaged_schemas_validate_configs_and_state(self):
        for path in (ROOT / "configs/autoloop").glob("*.json"):
            validate(read_json(path), 'config')
        c = self.controller(clock=FakeClock(), launcher=lambda *a, **k: FakeProcess(code=1))
        self.assertEqual(c.run(), 1)
        validate(self.state(), 'state')
        if importlib.util.find_spec('jsonschema') is not None:
            import jsonschema
            for path in (ROOT / 'configs/autoloop').glob('*.json'):
                jsonschema.validate(read_json(path), read_json(ROOT / 'hasami_learn/schemas/autoloop-config.schema.json'))
            jsonschema.validate(self.state(), read_json(ROOT / 'hasami_learn/schemas/autoloop-state.schema.json'))

    def test_crash_backoff_blocks_and_exit_75_does_not_count(self):
        clock = FakeClock()
        codes = iter([75, 75, 7, 7, 7])
        controller = self.controller(clock=clock, launcher=lambda *a, **k: FakeProcess(code=next(codes)))
        self.assertEqual(controller.run(), 1)
        self.assertEqual(self.state()['stages']['1:first']['attempts'], 3)
        self.assertEqual(self.state()['status'], 'blocked')
        failures = [e for e in self.events() if e['event'] == 'stage_failed']
        self.assertEqual([e['delay'] for e in failures], [0.02, 0.04, 0.02, 0.04, 0.08])
        self.assertEqual([e['transient'] for e in failures], [True, True, False, False, False])

    def test_retry_pause_handles_heartbeat_exceeding_remaining_backoff(self):
        clock = FakeClock()
        controller = self.controller(clock=clock)
        with patch.object(controller, 'heartbeat', side_effect=lambda: setattr(clock, 'now', clock.now + 1)):
            controller.pause(0.02)
        self.assertTrue(all(seconds >= 0 for seconds in clock.sleeps), clock.sleeps)

    def test_real_crashing_and_temporary_commands(self):
        self.data['stages'][0]['args'] = ['temporary']
        # Process startup competes with the full parity suite on busy CPUs;
        # this test checks exit semantics, not the separately injected watchdog.
        self.data['limits']['stall_timeout_s'] = 10
        self.write()
        child = self.launch()
        self.assertEqual(child.wait(timeout=30), 0, (self.root / 'process.log').read_text())
        self.assertEqual(self.state()['stages']['1:first']['attempts'], 0)
        self.assertEqual((self.root / 'output/test/c1/first/launches').read_text(), '3')

    def test_lock_second_instance_and_stale_file(self):
        store = StateStore(self.root / 'output/test')
        self.assertTrue(store.acquire())
        try:
            self.assertEqual(self.controller().run(), 3)
        finally:
            store.close()
        self.assertTrue((store.directory / 'loop.lock').exists())
        self.assertTrue(store.acquire())
        store.close()

    def test_state_fallback_and_config_change(self):
        c = self.controller(clock=FakeClock(), launcher=lambda *a, **k: FakeProcess(code=1))
        self.assertEqual(c.run(), 1)
        state = self.state()
        state['status'] = 'idle'
        state['stages'] = {}
        c.store.save(state)
        c.store.save(state)
        c.store.path.write_text('{broken')
        self.assertEqual(c.store.load(), state)
        c.store.save(state)
        self.assertEqual(read_json(c.store.backup), state)
        self.data['cycles'] = 2
        with self.assertRaisesRegex(LoopError, 'config changed'):
            self.controller().run()
        c = self.controller(clock=FakeClock(), launcher=lambda *a, **k: FakeProcess(code=1))
        self.assertEqual(c.run(accept_config_change=True), 1)
        self.assertEqual(len(self.state()['config_changes']), 1)
        c.store.path.write_text('{}')
        c.store.backup.write_text('{}')
        with self.assertRaises(LoopError): c.store.load()

    def test_suspend_clock_and_marker(self):
        clock = FakeClock()
        detector = SuspendDetector(clock, 120)
        marker = self.root / 'resume-check'
        clock.jump = 300
        self.assertIsNotNone(detector.check(marker))
        self.assertIsNone(detector.check(marker))
        marker.touch()
        self.assertTrue(detector.check(marker)['external'])
        self.assertFalse(marker.exists())

    def test_probe_timeout_is_a_failure(self):
        with patch('hasami_learn.autoloop.health.subprocess.run', side_effect=subprocess.TimeoutExpired('probe', 1)):
            self.assertFalse(probe('cuda', self.root, 1)['ok'])

    def test_nonobject_artifacts_never_claim_completion(self):
        plan = self.write()
        stages = Stages(plan)
        stage = plan.stage(0, 1)
        atomic_json(stages.done_file(stage, 1), [])
        self.assertIsNone(stages.complete(stage, 1))
        release = stages.release_dir(1)
        atomic_json(release / 'release.json', [])
        self.assertFalse(verify_release(release))
        with self.assertRaises(LoopError):
            stages.score({'results': []})

    def test_failed_suspend_probes_exit_75_with_marker(self):
        clock = FakeClock()
        calls = []
        def launch(*args, **kwargs):
            (self.root / 'output/test/control/resume-check').touch()
            return FakeProcess(ticks=1000)
        controller = self.controller(clock=clock, launcher=launch,
                                     health_probe=lambda *a: calls.append(a) or {'ok':False})
        self.assertEqual(controller.run(), 75)
        self.assertEqual(len(calls), 3)
        self.assertEqual(self.state()['status'], 'needs_wsl_restart')
        self.assertTrue((controller.control / 'needs_wsl_restart').exists())
        self.assertEqual(self.state()['stages']['1:first']['attempts'], 0)

    def test_probe_recovers_after_backoff_and_restarts_child(self):
        clock = FakeClock()
        probes = iter([{'ok':False}, {'ok':True}])
        launched = []
        def launch(*args, **kwargs):
            launched.append(1)
            if len(launched) == 1:
                (self.root / 'output/test/control/resume-check').touch()
                return FakeProcess(ticks=1000)
            return FakeProcess(finish=lambda: atomic_json(self.root / 'output/test/c1/first/done.json', {'status':'complete'}))
        c = self.controller(clock=clock, launcher=launch, health_probe=lambda *a: next(probes))
        self.assertEqual(c.run(), 0)
        self.assertEqual(len(launched), 2)
        self.assertEqual(self.state()['stages']['1:first']['attempts'], 0)

    def test_successful_suspend_probe_continue_then_short_watchdog(self):
        clock = FakeClock()
        self.data['limits']['post_resume_progress_s'] = 0.06
        self.data['limits']['max_attempts'] = 1
        def launch(*args, **kwargs):
            (self.root / 'output/test/control/resume-check').touch()
            return FakeProcess(ticks=1000)
        c = self.controller(clock=clock, launcher=launch, health_probe=lambda *a: {'ok':True})
        self.assertEqual(c.run(), 1)
        self.assertTrue(any(e['event'] == 'stall' and e['post_resume'] for e in self.events()))
        self.assertLess(clock.now, 1001)

    def test_progress_after_resume_disarms_short_deadline(self):
        clock = FakeClock()
        self.data['limits']['post_resume_progress_s'] = 0.06
        def launch(*args, **kwargs):
            (self.root / 'output/test/control/resume-check').touch()
            return FakeProcess(ticks=20, finish=lambda: atomic_json(self.root / 'output/test/c1/first/done.json', {'status':'complete'}))
        clock.hook = lambda: (self.root / 'output/test/c1/first/progress').write_text(str(clock.now))
        c = self.controller(clock=clock, launcher=launch, health_probe=lambda *a: {'ok':True})
        self.assertEqual(c.run(), 0)
        self.assertFalse(any(e['event'] == 'stall' for e in self.events()))

    def test_stall_restarts_and_blocks(self):
        clock = FakeClock()
        self.data['limits']['stall_timeout_s'] = 0.05
        c = self.controller(clock=clock, launcher=lambda *a, **k: FakeProcess(ticks=10000))
        self.assertEqual(c.run(), 1)
        self.assertEqual(len([e for e in self.events() if e['event'] == 'stall']), 3)

    def test_disk_pause_resume_and_graceful_stop_file(self):
        clock = FakeClock()
        self.data['limits']['min_free_gb'] = 2
        def launch(*args, **kwargs):
            (self.root / 'output/test/control/stop').touch()
            return FakeProcess(ticks=10000)
        c = self.controller(clock=clock, launcher=launch, disk=lambda p: 1 if clock.now < 1060 else 3)
        self.assertEqual(c.run(), 0)
        self.assertEqual(self.state()['status'], 'stopped')
        self.assertTrue(any(e['event'] == 'paused' for e in self.events()))
        self.assertEqual(read_json(c.directory / 'status.json')['status'], 'stopped')

    def test_timed_stop_blocks_generated_service_condition_until_explicit_resume(self):
        clock = FakeClock()
        self.data['stop_after_hours'] = 0.05 / 3600
        controller = self.controller(clock=clock, launcher=lambda *a, **k: FakeProcess(ticks=10000))
        self.assertEqual(controller.run(), 0)
        self.assertEqual(self.state()['status'], 'stopped')

        ready = self.install_condition()
        condition = lambda: subprocess.run(['bash', str(ready)], capture_output=True, timeout=10).returncode
        self.assertEqual(condition(), 1)
        marker = controller.control / 'stop'
        self.assertEqual(read_json(marker)['reason'], 'stop_after_hours')
        self.assertEqual(self.events()[-1]['event'], 'stop')

        # A fresh controller's timer cannot bypass the service's persisted stop.
        resumed = Controller(controller.plan, clock=FakeClock(), launcher=lambda *a, **k: FakeProcess(
            finish=lambda: atomic_json(controller.directory / 'c1/first/done.json', {'status': 'complete'})))
        self.assertEqual(condition(), 1)
        marker.unlink()  # Documented explicit systemd resume action.
        self.assertEqual(condition(), 0)
        atomic_json(marker, {'reason': 'stop_after_hours'})
        self.assertEqual(resumed.run(), 0)  # Explicit CLI run also clears the marker.
        self.assertFalse(marker.exists())
        self.assertEqual(self.state()['status'], 'complete')

    def install_condition(self):
        # Generate the actual installed ExecCondition without invoking systemd.
        spec = importlib.util.spec_from_file_location('autoloop_install', ROOT / 'scripts/autoloop/install.py')
        installer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(installer)
        home = self.root / 'home'
        with patch.object(sys, 'argv', ['install.py', str(self.path)]), \
                patch.object(sys, 'platform', 'linux'), \
                patch.object(installer.os, 'uname', return_value=SimpleNamespace(release='microsoft')), \
                patch.object(installer.Path, 'home', return_value=home), \
                patch.object(installer.configparser.ConfigParser, 'getboolean', return_value=True), \
                patch.object(installer.subprocess, 'run'):
            self.assertEqual(installer.main(), 0)
        unit = (home / '.config/systemd/user/hasami-autoloop@.service').read_text()
        self.assertIn('ExecCondition=/bin/bash "%h/.config/hasami-autoloop/%i/ready.sh"', unit)
        ready = home / '.config/hasami-autoloop/test/ready.sh'
        return ready

    def test_generated_condition_recovers_valid_backup_for_invalid_primary(self):
        controller = self.controller(clock=FakeClock(), launcher=lambda *a, **k: FakeProcess(code=1))
        self.assertEqual(controller.run(), 1)
        state = self.state()
        state['status'] = 'running'
        controller.store.save(state)
        controller.store.save(state)
        ready = self.install_condition()
        for malformed in ([], None, {'status': 'complete'}, {'status': 'blocked'}, '{broken'):
            with self.subTest(primary=malformed):
                if malformed == '{broken':
                    controller.store.path.write_text(malformed)
                else:
                    atomic_json(controller.store.path, malformed)
                self.assertEqual(controller.store.load(), state)
                result = subprocess.run(['bash', str(ready)], cwd=self.root,
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
        for status in ('complete', 'blocked'):
            state['status'] = status
            controller.store.save(state)
            for corrupt in (False, True):
                with self.subTest(status=status, corrupt=corrupt):
                    if corrupt:
                        controller.store.save(state)
                        atomic_json(controller.store.path, [])
                    result = subprocess.run(['bash', str(ready)], cwd=self.root,
                                            capture_output=True, timeout=10)
                    self.assertEqual(result.returncode, 1, result.stderr)

    def assert_candidate_contract(self, directory):
        from datetime import datetime, timezone
        from hasami_learn.autoloop.state import sha256, validate_schema

        for filename, schema_name in [('eval.json', 'evaluation-evidence'),
                                      ('release.json', 'release-candidate')]:
            data = read_json(directory / filename)
            schema = read_json(ROOT / 'hasami_learn/schemas' / (schema_name + '.schema.json'))
            validate_schema(data, schema)
            if importlib.util.find_spec('jsonschema') is not None:
                import jsonschema
                jsonschema.validate(data, schema)
            self.assertEqual(datetime.fromisoformat(data['created_at']).utcoffset(),
                             timezone.utc.utcoffset(None))
        evaluation = read_json(directory / 'eval.json')
        release = read_json(directory / 'release.json')
        self.assertNotEqual(evaluation['device'], 'auto')
        self.assertEqual(evaluation['checkpoint_sha256'], sha256(directory / 'weights.pt'))
        self.assertEqual(release['weights_sha256'], evaluation['checkpoint_sha256'])
        self.assertEqual(release['game_id'], read_json(directory / 'manifest.json')['game_id'])
        self.assertEqual(release['training_data'], {'kind': 'self-play', 'human_game_records': False})

    def test_duplicate_wins_cannot_pass_gate_and_legacy_stays_readable(self):
        from hasami_learn.autoloop.evidence import validate_evaluation, wilson_interval
        plan, stages, cp = self.bundle_plan()
        path = plan.stage_dir(plan.stage(1, 1), 1) / 'eval.json'
        data = self.evaluation(cp, checkpoint(cp)[1], 100)
        row = data['results']['random']
        row['extra'].update(distinct_games=2, distinct_openings=1,
                            effective_games=2, effective_successes=2.)
        row['ci_low'], row['ci_high'] = wilson_interval(2., 2)
        self.assertAlmostEqual(row['ci_low'], 0.3423719528896193)
        atomic_json(path, data)
        validate_evaluation(data)
        stages.local(plan.stage(2, 1), 1)
        self.assertFalse(read_json(plan.stage_dir(plan.stage(2, 1), 1) / 'gate.json')['promoted'])
        self.assertFalse((plan.output / 'models/best').exists())
        # The old interval can be read, but its raw count is not independent gate evidence.
        row.pop('extra')
        row['ci_low'], row['ci_high'] = wilson_interval(100, 100)
        validate_evaluation(data)
        self.assertLess(stages.score(data)[1], .6)

    def test_effective_sample_evidence_refuses_incomplete_or_invalid_counts(self):
        import copy
        from hasami_learn.autoloop.evidence import validate_evaluation
        plan, stages, cp = self.bundle_plan()
        original = read_json(plan.stage_dir(plan.stage(1, 1), 1) / 'eval.json')
        mutations = [lambda e: e.pop('effective_games'), lambda e: e.update(effective_games=101),
                     lambda e: e.update(effective_games=True), lambda e: e.update(effective_successes=101),
                     lambda e: e.update(effective_successes=float('nan')),
                     lambda e: e.update(distinct_games=1), lambda e: e.update(sample_unit='raw_games')]
        for mutation in mutations:
            data = copy.deepcopy(original)
            mutation(data['results']['random']['extra'])
            with self.subTest(data=data), self.assertRaises(LoopError):
                validate_evaluation(data)

    def test_opening_stage_defaults_override_and_warm_start_best_placeholder(self):
        plan, stages, cp = self.bundle_plan()
        training = read_json(cp.parent.parent / 'config.json')
        training['evaluation'] = {'opening_plies': 3}
        atomic_json(cp.parent.parent / 'config.json', training)
        stage = plan.stage(1, 1)
        for override, expected in (({}, 3), ({'opening_plies': 5}, 5)):
            command = stages.command({**stage, **override}, 1)
            self.assertEqual(command[command.index('--opening-plies') + 1], str(expected))
            self.assertEqual(command[command.index('--opening-seed') + 1], '42')
        directory = plan.stage_dir(stage, 1)
        atomic_json(directory / 'results.json', read_json(directory / 'eval.json')['results'])
        atomic_json(directory / 'results-device.json',
                    {'device': 'cpu', 'created_at': '2026-11-13T00:00:00+00:00'})
        stages.finalize(stage, 1)
        protocol = read_json(directory / 'eval.json')['protocol']
        self.assertEqual(protocol['opening_plies'], 3)
        self.assertEqual(protocol['opening_seed'], 42)
        self.assertEqual(protocol['sample_unit'], 'opening_actions_and_seating')
        self.data['stages'][0]['set'] = {'run.init_checkpoint': '{best}'}
        self.data['stages'][0]['cycle_overrides'] = {'2': {'run.init_checkpoint': '{best}/checkpoints/iter_{cycle}'}}
        plan = self.write()
        stages = Stages(plan)
        for cycle, suffix in ((1, ''), (2, '/checkpoints/iter_2')):
            command = stages.command(plan.stage(0, cycle), cycle)
            expected = str(plan.output / 'models/best') + suffix
            self.assertIn('run.init_checkpoint=' + json.dumps(expected), command)

    def test_evaluation_finalize_writes_release_evidence(self):
        plan, stages, cp = self.bundle_plan()
        stage = plan.stage(1, 1)
        directory = plan.stage_dir(stage, 1)
        existing = read_json(directory / 'eval.json')
        atomic_json(directory / 'results.json', existing['results'])
        atomic_json(directory / 'results-device.json',
                    {'device': 'cpu', 'created_at': '2026-11-13T00:00:00+00:00'})
        stages.finalize(stage, 1)
        from hasami_learn.autoloop.state import validate_schema
        evidence = read_json(directory / 'eval.json')
        validate_schema(evidence, read_json(ROOT / 'hasami_learn/schemas/evaluation-evidence.schema.json'))
        self.assertEqual(evidence['device'], 'cpu')
        self.assertEqual(evidence['game_id'], 'hasami')
        self.assertIsNotNone(stages.complete(stage, 1))
        for metadata in ({'device': 'auto', 'created_at': '2026-11-13T00:00:00+00:00'},
                         {'device': 'cpu', 'created_at': 'not-a-date'}):
            with self.subTest(metadata=metadata):
                atomic_json(directory / 'results-device.json', metadata)
                with self.assertRaises(LoopError):
                    stages.finalize(stage, 1)
                self.assertEqual(read_json(directory / 'eval.json'), evidence)

    def test_effective_sample_evidence_survives_finalize_gate_and_export(self):
        from hasami_learn.autoloop.evidence import validate_evaluation, wilson_interval

        plan, stages, cp = self.bundle_plan()
        stage = plan.stage(1, 1)
        directory = plan.stage_dir(stage, 1)
        results = self.evaluation(cp, checkpoint(cp)[1], 100)['results']
        results['random']['extra'].update(distinct_games=20, distinct_openings=10,
                                          effective_games=20, effective_successes=20.)
        atomic_json(directory / 'results.json', results)
        atomic_json(directory / 'results-device.json',
                    {'device': 'cpu', 'created_at': '2026-11-15T00:00:00+00:00'})
        stages.finalize(stage, 1)
        evidence = read_json(directory / 'eval.json')
        validate_evaluation(evidence)
        row = evidence['results']['random']
        self.assertEqual((row['games'], row['wins'], row['score_rate']), (100, 100, 1.))
        self.assertEqual(row['extra'], results['random']['extra'])
        self.assertEqual(row['extra']['sample_unit'], evidence['protocol']['sample_unit'])
        self.assertEqual((row['ci_low'], row['ci_high']), wilson_interval(20, 20))
        self.assertAlmostEqual(row['ci_low'], .8388698745050667)
        self.assertNotEqual(row['ci_low'], wilson_interval(100, 100)[0])
        self.assertIn('group', evidence['notes'])
        stages.local(plan.stage(2, 1), 1)
        gate = read_json(plan.stage_dir(plan.stage(2, 1), 1) / 'gate.json')
        self.assertTrue(gate['promoted'])
        self.assertEqual(gate['lower'], row['ci_low'])
        stages.local(plan.stage(3, 1), 1)
        release = stages.release_dir(1)
        self.assertTrue(verify_release(release))
        self.assertEqual(read_json(release / 'eval.json'), evidence)

    def test_finalize_labels_legacy_raw_interval_alongside_effective_interval(self):
        from hasami_learn.autoloop.evidence import validate_evaluation, wilson_interval

        plan, stages, cp = self.bundle_plan()
        stage = plan.stage(1, 1)
        directory = plan.stage_dir(stage, 1)
        results = self.evaluation(cp, checkpoint(cp)[1], 100)['results']
        results['greedy'] = dict(results['random'])
        results['greedy'].pop('extra')
        results['random']['extra'].update(distinct_games=20, distinct_openings=10,
                                          effective_games=20, effective_successes=20.)
        atomic_json(directory / 'results.json', results)
        atomic_json(directory / 'results-device.json',
                    {'device': 'cpu', 'created_at': '2026-11-15T00:00:00+00:00'})
        stages.finalize(stage, 1)
        evidence = read_json(directory / 'eval.json')
        validate_evaluation(evidence)
        self.assertEqual(evidence['results']['random']['ci_low'], wilson_interval(20, 20)[0])
        self.assertEqual(evidence['results']['greedy']['ci_low'], wilson_interval(100, 100)[0])
        self.assertIn('without sample metadata retain their raw interval', evidence['notes'])
        self.assertEqual(stages.score(evidence)[1], wilson_interval(22, 22)[0])

    def test_non_object_manifest_is_invalid_for_completion_and_heartbeat(self):
        plan, stages, cp = self.bundle_plan()
        controller = Controller(plan)
        controller.state = {'stage_index': 0, 'cycle': 1, 'status': 'running'}
        for malformed in ([], None, 'manifest', 42, False):
            with self.subTest(manifest=malformed):
                atomic_json(cp / 'manifest.json', malformed)
                with self.assertRaisesRegex(LoopError, 'manifest.*object'):
                    checkpoint(cp)
                self.assertIsNone(stages.complete(plan.stage(0, 1), 1))
                controller.heartbeat()
                self.assertIsNone(read_json(plan.loop_dir / 'status.json')['latest_iteration'])

    @unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'torch not installed')
    def test_malformed_latest_manifest_recovers_older_checkpoint_with_auto_resume(self):
        for malformed in ([], None):
            with self.subTest(manifest=malformed):
                self.data = read_json(ROOT / 'configs/autoloop/tiny-hasami.json')
                self.data.update(loop_id='bad-list' if malformed == [] else 'bad-null',
                                 output_dir=str(self.root / 'output'))
                self.data['stages'] = self.data['stages'][:1]
                self.data['stages'][0]['set'].update({'run.iterations': 2, 'run.keep_checkpoints': 3})
                plan = self.write()
                stages = Stages(plan)
                stage = plan.stage(0, 1)
                command = stages.command(stage, 1)
                initial = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
                self.assertEqual(initial.returncode, 0, initial.stderr)
                run = stages.run_dir(stage, 1)
                latest, _ = checkpoint(run)
                self.assertTrue((run / 'checkpoints/iter_000001/manifest.json').is_file())
                atomic_json(latest / 'manifest.json', malformed)
                self.assertEqual(Controller(plan).run(), 0)
                rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
                resumed = [row for row in rows if row['kind'] == 'resume'][-1]
                self.assertEqual(resumed['iteration'], 1)
                self.assertTrue(any(Path(row['path']).name == latest.name for row in resumed['extra']['skipped']))
                self.assertEqual(read_json(run / 'trainer_state.json')['iteration'], 2)
                self.assertIsNotNone(stages.complete(stage, 1))
                self.assertEqual(read_json(plan.loop_dir / 'loop_state.json')['stages']['1:train']['attempts'], 0)

    def assert_guardian_cleans_descendant(self, parent_eof):
        directory = self.root / 'guardian'
        directory.mkdir()
        descendant_code = '''import os, signal, sys, time
from pathlib import Path
p = Path(sys.argv[1])
signal.signal(signal.SIGTERM, lambda *_: (p / "term").touch())
(p / "descendant").write_text(str(os.getpid()))
while True: time.sleep(0.01)
'''
        stage_code = '''import os, subprocess, sys, time
from pathlib import Path
p = Path(sys.argv[1])
subprocess.Popen([sys.executable, "-c", sys.argv[2], str(p)])
while not (p / "descendant").exists(): time.sleep(0.01)
(p / "ready").write_text(str(os.getpgrp()))
while not (p / "exit").exists(): time.sleep(0.01)
sys.exit(7)
'''
        store = StateStore(directory)
        self.assertTrue(store.acquire())
        log = (directory / 'worker.log').open('w')
        self.addCleanup(log.close)
        worker = subprocess.Popen([sys.executable, '-m', 'hasami_learn.autoloop.worker', '0.5',
                                   sys.executable, '-c', stage_code, str(directory), descendant_code],
                                  cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                  stderr=log, start_new_session=True,
                                  pass_fds=(store.lock.fileno(),))
        store.close()
        group = None
        try:
            deadline = time.monotonic() + 10
            while not (directory / 'ready').exists():
                self.assertIsNone(worker.poll())
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            group = int((directory / 'ready').read_text())
            descendant = int((directory / 'descendant').read_text())
            if parent_eof:
                worker.stdin.close()
            else:
                (directory / 'exit').touch()
            while not (directory / 'term').exists():
                self.assertIsNone(worker.poll(), 'guardian exited before cleaning descendants')
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            # The direct child has exited, but the stubborn descendant still owns work.
            time.sleep(0.05)
            contender = StateStore(directory)
            acquired = contender.acquire()
            contender.close()
            self.assertFalse(acquired, 'guardian released lock before escalation')
            self.assertEqual(worker.wait(timeout=5), 143 if parent_eof else 7,
                             (directory / 'worker.log').read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(descendant, 0)
            self.assertTrue(store.acquire())
        finally:
            for pgid in {worker.pid, group} - {None}:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            worker.wait(timeout=5)
            worker.stdin.close()
            store.close()

    def test_guardian_cleans_descendants_after_stage_exit(self):
        self.assert_guardian_cleans_descendant(parent_eof=False)

    def test_guardian_escalates_after_parent_eof_and_direct_child_exit(self):
        self.assert_guardian_cleans_descendant(parent_eof=True)

    def test_guardian_permission_denied_does_not_mean_group_is_gone(self):
        from hasami_learn.autoloop.worker import signal_group

        for signum in (0, signal.SIGTERM, signal.SIGKILL):
            with self.subTest(signal=signum), patch('os.killpg', side_effect=[
                    PermissionError(1, 'Operation not permitted'), ProcessLookupError()]):
                self.assertTrue(signal_group(99999999, signum))
                self.assertFalse(signal_group(99999999, signum))

    def test_validation_gate_missing_fail_other_commit_dirty_and_pass(self):
        repo = self.root / 'git'
        repo.mkdir()
        def git(*args):
            return subprocess.check_output(['git','-C',str(repo),*args], text=True, stderr=subprocess.PIPE).strip()
        git('init')
        git('config','user.name','Test')
        git('config','user.email','test')
        (repo / 'tracked').write_text('initial')
        git('add','tracked')
        git('commit','-m','initial')
        old = git('rev-parse','HEAD')
        (repo / 'tracked').write_text('current')
        git('commit','-am','current')
        commit = git('rev-parse','HEAD')
        config = {'require_validation':True,'validation_results':str(self.root / 'validation')}
        with self.assertRaises(LoopError): validation_gate(config, root=repo)
        for gate, identity, dirty, live_dirty in [('FAIL',commit,False,False),('PASS',old,False,False),('PASS',commit,True,False),('PASS',commit,False,True)]:
            atomic_json(Path(config['validation_results'])/'summary.json', {'gate':gate})
            atomic_json(Path(config['validation_results'])/'env.json', {'checkout':{'commit':identity,'dirty':dirty}})
            if live_dirty: (repo / 'tracked').write_text('dirty')
            with self.assertRaises(LoopError): validation_gate(config, root=repo)
        (repo / 'tracked').write_text('current')
        atomic_json(Path(config['validation_results'])/'env.json', {'checkout':{'commit':commit[:12],'dirty':False}})
        self.assertEqual(validation_gate(config, root=repo)['gate'], 'PASS')
        with self.assertRaises(LoopError): validation_gate(config, True, root=repo)

    def test_validation_failure_is_persisted_and_rechecked(self):
        def refuse(*args):
            raise LoopError('missing PASS')
        c = self.controller(gate=refuse)
        self.assertEqual(c.run(), 1)
        self.assertEqual(self.state()['validation']['gate'], 'FAIL')
        self.assertEqual(self.state()['status'], 'blocked')
        c = self.controller(clock=FakeClock(), launcher=lambda *a, **k: FakeProcess(
            finish=lambda: atomic_json(self.root / 'output/test/c1/first/done.json', {'status':'complete'})))
        self.assertEqual(c.run(), 0)

    def test_failed_probe_marker_requires_probe_before_restarting(self):
        marker = self.root / 'output/test/control/needs_wsl_restart'
        atomic_json(marker, {'probe': {'ok': False}})
        calls = []
        c = self.controller(clock=FakeClock(), health_probe=lambda *a: calls.append(a) or {'ok':False},
                            launcher=lambda *a, **k: self.fail('must not launch with broken device'))
        self.assertEqual(c.run(), 75)
        self.assertEqual(len(calls), 3)

    def test_cycle_override_and_placeholders(self):
        self.data['stages'][0]['set'] = {'algorithm.params': {'engine_checkpoint':'{best}'}}
        self.data['stages'][0]['cycle_overrides'] = {'2': {'run.iterations':9}}
        plan = self.write()
        stage = plan.stage(0,2)
        self.assertEqual(stage['set']['run.iterations'],9)
        self.assertEqual(stage['set']['algorithm.params']['engine_checkpoint'],str(plan.output/'models/best'))
        self.assertEqual(plan.run_id(stage,2),'test-first-c2')

    def test_train_completion_requires_root_state_and_valid_weights(self):
        from hasami_learn.autoloop.state import sha256
        self.data['stages'] = [{'id':'train','kind':'train','config':'configs/tiny/alphazero.json'}]
        plan = self.write()
        stages = Stages(plan)
        stage = plan.stage(0,1)
        run = stages.run_dir(stage,1)
        cp = run/'checkpoints/iter_000001'
        cp.mkdir(parents=True)
        (cp/'weights.pt').write_bytes(b'weights')
        atomic_json(cp/'manifest.json', {'weights_file':'weights.pt','weights_sha256':sha256(cp/'weights.pt')})
        atomic_json(cp/'trainer_state.json', {'iteration':1})
        atomic_json(run/'checkpoints/latest.json', {'path':'checkpoints/iter_000001','iteration':1})
        atomic_json(run/'trainer_state.json', {'iteration':0})
        self.assertIsNone(stages.complete(stage,1))
        atomic_json(run/'trainer_state.json', {'iteration':1})
        self.assertIsNotNone(stages.complete(stage,1))
        (cp/'weights.pt').write_bytes(b'bad')
        self.assertIsNone(stages.complete(stage,1))

    def test_gate_threshold_rejection_atomic_replacement_and_hashes(self):
        from hasami_learn.autoloop.state import sha256
        self.data['stages'] = [
            {'id':'train','kind':'train','config':'configs/tiny/alphazero.json'},
            {'id':'eval','kind':'evaluate','source':'train','games':100,'seed':42,'opponents':['random']},
            {'id':'gate','kind':'gate','source':'eval','min_score':0.6,'min_lower':0.6},
            {'id':'export','kind':'export'}]
        plan = self.write()
        stages = Stages(plan)
        gate = plan.stage(2,1)
        directory = plan.stage_dir(gate,1)
        directory.mkdir(parents=True)
        def candidate(cycle, wins):
            cp = stages.run_dir(plan.stage(0,cycle),cycle)/'checkpoints/iter_000001'
            cp.mkdir(parents=True)
            (cp/'weights.pt').write_bytes(str(cycle).encode())
            digest = sha256(cp/'weights.pt')
            atomic_json(cp/'manifest.json',{'game_id':'hasami','weights_file':'weights.pt','weights_sha256':digest})
            atomic_json(cp.parent.parent/'config.json',{'run': {'iterations':1}})
            evaluation = self.evaluation(cp, digest, wins)
            atomic_json(plan.stage_dir({'id':'eval'},cycle)/'eval.json',evaluation)
            return cp, digest
        candidate(1,50)
        stages.local(gate,1)
        self.assertFalse(stages.complete(gate,1)['promoted'])
        self.assertFalse((plan.output/'models/best').exists())
        cp,digest = candidate(2,80)
        stages.local(gate,2)
        best = plan.output/'models/best'
        self.assertTrue(best.is_symlink())
        self.assertEqual(checkpoint(best)[1],digest)
        old_target = best.resolve()
        cp,digest = candidate(3,90)
        stages.local(gate,3)
        self.assertEqual(checkpoint(best)[1],digest)
        self.assertTrue(old_target.is_dir())
        stages.local(plan.stage(3,3),3)
        self.assertTrue(verify_release(stages.release_dir(3)))
        (best/'weights.pt').write_bytes(b'corrupt')
        with self.assertRaises(LoopError): checkpoint(best)

    @staticmethod
    def evaluation(cp, digest, wins):
        p = wins / 100
        center = (p + 1.96**2 / 200) / (1 + 1.96**2 / 100)
        margin = 1.96 * math.sqrt(p * (1 - p) / 100 + 1.96**2 / 40000) / (1 + 1.96**2 / 100)
        return {'schema_version': 1, 'checkpoint': str(cp.resolve()), 'checkpoint_sha256': digest,
                'seed': 42, 'created_at': '2026-11-13T00:00:00+00:00', 'device': 'cpu', 'game_id': 'hasami',
                'results': {'random': {'games': 100, 'wins': wins, 'losses': 100 - wins,
                                       'draws': 0, 'truncated': 0,
                                       'extra': {'distinct_games': 100, 'distinct_openings': 50,
                                                 'effective_games': 100, 'effective_successes': wins,
                                                 'sample_unit': 'opening_actions_and_seating'},
                                       'ci_low': max(0, center - margin), 'ci_high': min(1, center + margin)}}}

    def test_gate_refuses_invalid_evidence_and_wrong_checkpoint(self):
        plan, stages, cp = self.bundle_plan()
        directory = plan.stage_dir(plan.stage(1, 1), 1)
        original = read_json(directory / 'eval.json')
        mutations = [lambda d: d.update(device='auto'),
                     lambda d: d.update(checkpoint_sha256='0' * 64),
                     lambda d: d.update(created_at='not-a-date'),
                     lambda d: d.update(created_at='2026-11-13T00:00:00'),
                     lambda d: d.update(schema_version=True),
                     lambda d: d.update(results={}),
                     lambda d: d['results']['random'].pop('ci_high'),
                     lambda d: d['results']['random'].update(ci_low=0.99)]
        for mutation in mutations:
            evidence = json.loads(json.dumps(original))
            mutation(evidence)
            with self.subTest(evidence=evidence):
                atomic_json(directory / 'eval.json', evidence)
                self.assertIsNone(stages.complete(plan.stage(1, 1), 1))
                with self.assertRaises(LoopError):
                    stages.local(plan.stage(2, 1), 1)
                self.assertFalse((plan.output / 'models/best').exists())

    def test_export_rejects_invalid_evidence_before_publishing(self):
        plan, stages, cp = self.bundle_plan()
        stages.local(plan.stage(2, 1), 1)
        path = plan.output / 'models/best/eval.json'
        data = read_json(path)
        data['device'] = 'auto'
        atomic_json(path, data)
        with self.assertRaises(LoopError):
            stages.local(plan.stage(3, 1), 1)
        self.assertFalse(stages.release_dir(1).exists())
        data['device'] = 'cpu'
        atomic_json(path, data)
        manifest_path = path.parent / 'manifest.json'
        manifest = read_json(manifest_path)
        manifest['game_id'] = 'Invalid/Game'
        atomic_json(manifest_path, manifest)
        with self.assertRaises(LoopError):
            stages.local(plan.stage(3, 1), 1)
        self.assertFalse(stages.release_dir(1).exists())

    def test_release_metadata_is_validated_and_name_is_safe(self):
        self.data['loop_id'] = 'A' * 80
        plan, stages, cp = self.bundle_plan()
        stages.local(plan.stage(2, 1), 1)
        stages.local(plan.stage(3, 1), 1)
        directory = stages.release_dir(1)
        self.assert_candidate_contract(directory)
        release = read_json(directory / 'release.json')
        self.assertRegex(release['name'], r'^[a-z0-9][a-z0-9._-]{0,63}$')
        from hasami_learn.autoloop.evidence import model_name
        self.assertNotEqual(model_name('A' * 80, 1), model_name('A' * 79 + 'B', 1))
        self.assertNotEqual(model_name('A' * 80, 1), model_name('A' * 80, 2))
        for mutation in [lambda d: d.pop('name'),
                         lambda d: d.update(weights_sha256='0' * 64),
                         lambda d: d.update(game_id='shogi'),
                         lambda d: d['training_data'].update(human_game_records=True),
                         lambda d: d.update(name='../escape')]:
            data = json.loads(json.dumps(release))
            mutation(data)
            atomic_json(directory / 'release.json', data)
            self.assertFalse(verify_release(directory))

    @unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'torch not installed')
    def test_evaluation_subprocess_records_resolved_device(self):
        import torch
        from hasami_learn.autoloop.evaluate import main

        for resolved, expected in [('cpu', 'cpu'), ('cuda', 'cuda:2')]:
            with self.subTest(device=resolved), \
                    patch('hasami_learn.devices.resolve_device', return_value=torch.device(resolved)), \
                    patch('torch.cuda.current_device', return_value=2), \
                    patch('hasami_learn.autoloop.evaluate.subprocess.run',
                          return_value=SimpleNamespace(returncode=0)) as launch:
                self.assertEqual(main(['--checkpoint', str(self.root), '--device', 'auto',
                                       '--out', str(self.root / 'results.json')]), 0)
                command = launch.call_args.args[0]
                self.assertEqual(command[:4], [sys.executable, '-m', 'hasami_learn.cli', 'eval'])
                self.assertEqual(command[-2:], ['--device', expected])
                self.assertEqual(read_json(self.root / 'results-device.json')['device'], expected)

    def bundle_plan(self, practical=False):
        from hasami_learn.autoloop.state import sha256

        self.data['stages'] = [
            {'id': 'train', 'kind': 'train', 'config': 'configs/tiny/alphazero.json'},
            {'id': 'eval', 'kind': 'evaluate', 'source': 'train', 'games': 100,
             'seed': 42, 'opponents': ['random']},
            {'id': 'gate', 'kind': 'gate', 'source': 'eval', 'min_score': 0.6, 'min_lower': 0.6},
            {'id': 'export', 'kind': 'export'}]
        self.data['limits'].update(max_attempts=1, stall_timeout_s=10)
        plan = self.write()
        stages = Stages(plan)
        run = stages.run_dir(plan.stage(0, 1), 1)
        cp = run / 'checkpoints/iter_000001'
        cp.mkdir(parents=True)
        (cp / 'weights.pt').write_bytes(b'candidate weights')
        digest = sha256(cp / 'weights.pt')
        manifest = {'game_id': 'hasami', 'weights_file': 'weights.pt', 'weights_sha256': digest,
                    'inference_hint': {'mode': 'policy'}}
        if practical:
            engine = self.root / 'engine-run/checkpoints/iter_000010'
            engine.mkdir(parents=True)
            (engine / 'weights.pt').write_bytes(b'engine weights')
            engine_digest = sha256(engine / 'weights.pt')
            atomic_json(engine / 'manifest.json', {'weights_sha256': engine_digest})
            manifest['inference_hint'] = {'mode': 'practical', 'levels': 6,
                                          'engine_checkpoint': os.path.relpath(engine.resolve(), cp),
                                          'engine_sha256': engine_digest}
            atomic_json(cp / 'levels.json', {'levels': [1, 2, 3, 4, 5, 6]})
        atomic_json(cp / 'manifest.json', manifest)
        atomic_json(cp / 'trainer_state.json', {'iteration': 1})
        atomic_json(cp / 'extra/nested.json', {'preserved': True})
        atomic_json(run / 'config.json', {'run': {'iterations': 1}})
        atomic_json(run / 'trainer_state.json', {'iteration': 1})
        atomic_json(run / 'checkpoints/latest.json', {'path': 'checkpoints/iter_000001', 'iteration': 1})
        evaluation = self.evaluation(cp, digest, 90)
        atomic_json(plan.stage_dir({'id': 'eval'}, 1) / 'eval.json', evaluation)
        return plan, stages, cp

    def test_practical_promotion_and_export_bundle_engine_and_extra_files(self):
        from hasami_learn.checkpoint_bundle import engine_dependency, file_hashes

        plan, stages, cp = self.bundle_plan(practical=True)
        original = file_hashes(cp)
        stages.local(plan.stage(2, 1), 1)
        best = plan.output / 'models/best'
        self.assertEqual(file_hashes(cp), original)
        hint = read_json(best / 'manifest.json')['inference_hint']
        self.assertEqual(hint['engine_checkpoint'], 'engine')
        self.assertEqual(engine_dependency(best), (best / 'engine').resolve())
        self.assertEqual((best / 'engine/weights.pt').read_bytes(), b'engine weights')
        self.assertEqual(read_json(best / 'levels.json'), read_json(cp / 'levels.json'))
        self.assertEqual(read_json(best / 'extra/nested.json'), {'preserved': True})
        # Export must work after both original training locations have moved.
        cp.rename(cp.with_name('moved-checkpoint'))
        (self.root / 'engine-run').rename(self.root / 'moved-engine-run')
        stages.local(plan.stage(3, 1), 1)
        release = stages.release_dir(1)
        self.assertTrue(verify_release(release))
        self.assert_candidate_contract(release)
        self.assertEqual(read_json(release / 'manifest.json')['inference_hint'], hint)
        self.assertEqual(engine_dependency(release), (release / 'engine').resolve())
        hashes = file_hashes(release)
        hashes.pop('release.json')
        self.assertEqual(read_json(release / 'release.json')['files'], hashes)
        for name, digest in file_hashes(best).items():
            self.assertEqual(hashes[name], digest, name)
        (release / 'engine/weights.pt').write_bytes(b'tampered engine')
        self.assertFalse(verify_release(release))

    def test_wrong_engine_hash_counts_attempt_and_preserves_previous_best(self):
        from hasami_learn.checkpoint_bundle import file_hashes

        plan, stages, cp = self.bundle_plan(practical=True)
        gate = plan.stage(2, 1)
        stages.local(gate, 1)
        best = plan.output / 'models/best'
        previous = best.resolve()
        previous_files = file_hashes(best)
        manifest = read_json(cp / 'manifest.json')
        manifest['inference_hint']['engine_sha256'] = '0' * 64
        atomic_json(cp / 'manifest.json', manifest)
        # Retry the same evaluated weights: cached candidates must not bypass bundling.
        (plan.stage_dir(gate, 1) / 'gate.json').unlink()
        controller = Controller(plan)
        self.assertEqual(controller.run(), 1)
        state = self.state()
        self.assertEqual(state['status'], 'blocked')
        self.assertEqual(state['stages']['1:gate']['attempts'], 1)
        self.assertEqual(best.resolve(), previous)
        self.assertEqual(file_hashes(best), previous_files)
        self.assertFalse((plan.stage_dir(gate, 1) / 'gate.json').exists())
        self.assertFalse(stages.release_dir(1).exists())
        failures = [event for event in self.events() if event['event'] == 'stage_failed']
        self.assertEqual(len(failures), 1)
        self.assertFalse(failures[0]['transient'])
        self.assertIn('engine_sha256', (plan.stage_dir(gate, 1) / 'stage.log').read_text())

    def test_local_retry_removes_unpublished_pointer_without_touching_its_target(self):
        plan, stages, _ = self.bundle_plan()
        models = plan.output / 'models'
        models.mkdir()
        outside = self.root / 'kept-candidate'
        outside.mkdir()
        (outside / 'sentinel').write_text('keep')
        stale = models / ('.best-' + 'b' * 32)
        stale.symlink_to(outside, target_is_directory=True)
        stages.local(plan.stage(2, 1), 1)
        self.assertFalse(stale.is_symlink())
        self.assertEqual((outside / 'sentinel').read_text(), 'keep')
        self.assertTrue(stages.complete(plan.stage(2, 1), 1)['promoted'])

    def test_plain_promotion_and_export_preserve_all_checkpoint_files(self):
        from hasami_learn.checkpoint_bundle import file_hashes

        plan, stages, cp = self.bundle_plan()
        stages.local(plan.stage(2, 1), 1)
        stages.local(plan.stage(3, 1), 1)
        best = plan.output / 'models/best'
        release = stages.release_dir(1)
        self.assertTrue(verify_release(release))
        self.assert_candidate_contract(release)
        self.assertFalse((release / 'engine').exists())
        hashes = read_json(release / 'release.json')['files']
        actual = file_hashes(release)
        actual.pop('release.json')
        self.assertEqual(hashes, actual)
        for name, digest in file_hashes(cp).items():
            self.assertEqual(file_hashes(best)[name], digest, name)
            self.assertEqual(hashes[name], digest, name)

    def test_installer_dry_run_has_no_side_effects(self):
        home = self.root / 'home'
        home.mkdir()
        result = subprocess.run(['bash','scripts/autoloop/install_wsl.sh','--dry-run','configs/autoloop/tiny-hasami.json'], cwd=ROOT,
                                env={**os.environ,'HOME':str(home),'HASAMI_TRAIN_PYTHON':sys.executable}, capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('enable-linger',result.stdout)
        loop_id = json.loads((ROOT / 'configs/autoloop/tiny-hasami.json').read_text())['loop_id']
        self.assertIn(f'hasami-autoloop@{loop_id}.service',result.stdout)
        self.assertEqual(list(home.iterdir()),[])

    def test_sigkill_restart_three_stages_complete_once(self):
        self.data['stages'] = [self.module(name) for name in ('first','second','third')]
        self.write()
        rng = random.Random(31)
        for index, stage in enumerate(self.data['stages']):
            directory = self.root / 'output/test/c1' / stage['id']
            # After a SIGKILL the previous stage's guardian legitimately holds the lock until its checkpoint finishes;
            # a controller started meanwhile exits with 3, whenever it gets to the lock (load-dependent), so relaunch
            # on 3 instead of assuming a fixed startup delay.
            deadline = time.monotonic() + 30
            while True:
                child = self.launch()
                while not (directory / 'progress').exists() and child.poll() is None:
                    self.assertLess(time.monotonic(), deadline, str(directory / 'progress'))
                    time.sleep(0.01)
                if child.poll() == 3:
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.05)
                    continue
                break
            self.await_file(directory/'progress',child)
            time.sleep(rng.uniform(0.005,0.03))
            child.kill()
            child.wait(timeout=5)
        deadline = time.monotonic() + 15
        while True:
            child = self.launch()
            code = child.wait(timeout=15)
            if code != 3: break
            self.assertLess(time.monotonic(),deadline)
            time.sleep(0.05)
        self.assertEqual(code,0,(self.root/'process.log').read_text())
        state = self.state()
        validate(state,'state')
        self.assertEqual(state['status'],'complete')
        for stage in self.data['stages']:
            done = read_json(self.root/'output/test/c1'/stage['id']/'done.json')
            self.assertEqual(done,{'status':'complete','value':42})
            self.assertEqual(len([e for e in self.events() if e['event']=='stage_done' and e['stage']==stage['id']]),1)
        self.assertEqual(self.launch().wait(timeout=10),0)
        self.assertEqual(len([e for e in self.events() if e['event']=='stage_done']),3)

    @unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'torch not installed')
    def test_real_tiny_train_sigkill_resume_evaluate_gate_export(self):
        self.data = read_json(ROOT/'configs/autoloop/tiny-hasami.json')
        self.data['output_dir'] = str(self.root/'output')
        self.data['loop_id'] = 'test'
        self.data['stages'][0]['set']['run.iterations'] = 12
        self.write()
        command = [sys.executable,'-m','hasami_learn.autoloop','run','--config',str(self.path)]
        child = self.launch(command)
        run = self.root/'output/runs/test-train-c1'
        self.await_file(run/'checkpoints/latest.json',child,40)
        before = read_json(run/'checkpoints/latest.json')['iteration']
        self.assertLess(before,12)
        child.kill()
        child.wait(timeout=5)
        deadline = time.monotonic()+50
        while True:
            child = self.launch(command)
            code = child.wait(timeout=50)
            if code != 3: break
            self.assertLess(time.monotonic(),deadline)
            time.sleep(0.05)
        self.assertEqual(code,0,(self.root/'process.log').read_text())
        rows = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
        resumed = [row['iteration'] for row in rows if row['kind']=='resume']
        self.assertTrue(resumed)
        self.assertGreaterEqual(resumed[-1],before)
        self.assertLess(resumed[-1],12)
        self.assertEqual(read_json(run/'trainer_state.json')['iteration'],12)
        release = self.root/'output/release/test-c1'
        self.assertTrue(verify_release(release))
        self.assert_candidate_contract(release)
        self.assertEqual(read_json(release / 'eval.json')['device'], 'cpu')
        self.assertEqual(self.state()['status'],'complete')
        from hasami_learn.manifest import load_checkpoint
        self.assertEqual(load_checkpoint(release,device='cpu').manifest.iteration,12)
        (release/'weights.pt').write_bytes(b'corrupted')
        self.assertFalse(verify_release(release))


if __name__ == '__main__':
    unittest.main()
