"""Reporter crash recovery stays independent of training and optional packages."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from hasami_learn.progress import write_html
from hasami_learn.runlock import cleanup_progress_files, cleanup_training_files, run_lock


ROOT = Path(__file__).resolve().parents[2]


class ProgressRecoveryTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'posix', 'directory locks require POSIX')
    def test_symlink_dotdot_cannot_clean_a_live_reporters_directory(self):
        import fcntl

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            actual = root / 'actual'
            (actual / 'child').mkdir(parents=True)
            (root / 'link').symlink_to(actual / 'child', target_is_directory=True)
            alias = root / 'link' / '..'
            active = actual / '.progress-abcdefgh.tmp'
            active.write_text('live reporter')
            lexical = root / active.name
            lexical.write_text('other directory')
            fd = os.open(actual, os.O_RDONLY | os.O_DIRECTORY)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                for path in (alias, Path(os.path.relpath(root)) / 'link' / '..'):
                    with self.subTest(path=path):
                        self.assertEqual(cleanup_progress_files(path), [])
                        self.assertEqual(active.read_text(), 'live reporter')
                        self.assertEqual(lexical.read_text(), 'other directory')
            finally:
                os.close(fd)

    @unittest.skipUnless(os.name == 'posix', 'symlink test requires POSIX')
    def test_reporter_rejects_symlink_dotdot_before_cleanup_or_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            actual = root / 'actual'
            (actual / 'child').mkdir(parents=True)
            (root / 'link').symlink_to(actual / 'child', target_is_directory=True)
            for directory in (root, actual):
                (directory / 'progress.html').write_text('previous report')
                (directory / '.progress-abcdefgh.tmp').write_text('keep')
            alias = root / 'link' / '..'
            for path in (alias, Path(os.path.relpath(root)) / 'link' / '..'):
                with self.subTest(path=path):
                    with self.assertRaisesRegex(RuntimeError, 'symlink'):
                        write_html(path / 'progress.html', 'new report')
                    for directory in (root, actual):
                        self.assertEqual((directory / 'progress.html').read_text(), 'previous report')
                        self.assertEqual((directory / '.progress-abcdefgh.tmp').read_text(), 'keep')

    def test_stale_reporter_temporary_recovered_by_resume_and_reporter(self):
        for recovery in ('resume', 'reporter'):
            with self.subTest(recovery=recovery), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                stale = directory / '.progress-abcdefgh.tmp'
                stale.write_text('dead reporter')
                output = directory / 'progress.html'
                output.write_text('previous report')
                unknown = directory / '.progress-user-notes.tmp'
                unknown.write_text('keep')
                if recovery == 'resume':
                    cleanup_training_files(directory)
                    self.assertEqual(output.read_text(), 'previous report')
                else:
                    write_html(output, 'new report')
                    self.assertEqual(output.read_text(), 'new report')
                self.assertFalse(stale.exists())
                self.assertEqual(unknown.read_text(), 'keep')

    def test_live_reporter_survives_resume_then_dead_reporter_is_recovered(self):
        code = '''import sys
from hasami_learn import progress
def pause(*args):
    print('staged', flush=True)
    sys.stdin.read()
progress.os.replace = pause
progress.write_html(sys.argv[1], 'new report')
'''
        for recovery in ('resume', 'reporter'):
            with self.subTest(recovery=recovery), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                output = directory / 'progress.html'
                output.write_text('previous report')
                child = subprocess.Popen([sys.executable, '-c', code, str(output)], cwd=ROOT,
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True)
                try:
                    self.assertEqual(child.stdout.readline().strip(), 'staged')
                    staging = list(directory.glob('.progress-*.tmp'))
                    self.assertEqual(len(staging), 1)
                    # This must return while the live reporter stays paused.
                    cleanup_training_files(directory)
                    self.assertTrue(staging[0].is_file())
                    self.assertEqual(output.read_text(), 'previous report')
                finally:
                    child.kill()
                    child.communicate(timeout=10)
                self.assertTrue(staging[0].is_file())
                if recovery == 'resume':
                    cleanup_training_files(directory)
                else:
                    write_html(output, 'recovered report')
                self.assertFalse(staging[0].exists())

    def test_reporter_never_takes_training_lock(self):
        code = 'import sys; from hasami_learn.progress import write_html; write_html(sys.argv[1], "report")'
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'progress.html'
            with run_lock(tmp):
                result = subprocess.run([sys.executable, '-c', code, str(output)], cwd=ROOT,
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(output.read_text(), 'report')

    @unittest.skipUnless(os.name == 'posix', 'symlink test requires POSIX')
    def test_cleanup_never_follows_reporter_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            outside = directory / 'outside'
            outside.mkdir()
            protected = outside / '.progress-abcdefgh.tmp'
            protected.write_text('keep')
            run = directory / 'run'
            run.mkdir()
            link = run / protected.name
            link.symlink_to(protected)
            (run / 'linked').symlink_to(outside, target_is_directory=True)
            cleanup_training_files(run)
            write_html(run / 'progress.html', 'report')
            self.assertTrue(link.is_symlink())
            self.assertEqual(protected.read_text(), 'keep')
            with self.assertRaisesRegex(RuntimeError, 'symlink'):
                write_html(run / 'linked/progress.html', 'report')


if __name__ == '__main__':
    unittest.main()
