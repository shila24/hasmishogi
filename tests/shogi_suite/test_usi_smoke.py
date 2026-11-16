"""Exercise the public protocol driver without optional inference dependencies."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts import usi_smoke


class USISmokeTests(unittest.TestCase):
    def test_real_classical_subprocess(self):
        report = usi_smoke.exercise()
        self.assertTrue(report['ok'])
        self.assertEqual(report['mode'], 'classical')
        self.assertEqual(tuple(report['options']), usi_smoke.declared_options(usi_smoke.ROOT))

    def test_cli_reports_absent_models_explicitly(self):
        result = subprocess.run([sys.executable, str(usi_smoke.ROOT / 'scripts/usi_smoke.py'),
                                 '--classical-only'], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('not run', json.loads(result.stdout)['model_status'])

    def test_missing_model_is_an_error_not_ready_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(usi_smoke.ProtocolError, 'model error'):
                usi_smoke.exercise(model=Path(tmp) / 'missing')

    def test_nonresponsive_child_is_terminated(self):
        session = usi_smoke.Session([sys.executable, '-c', 'import time; time.sleep(30)'],
                                    usi_smoke.ROOT, .1)
        try:
            with self.assertRaisesRegex(usi_smoke.ProtocolError, 'Timed out'):
                session.until('usiok')
        finally:
            session.close()
        self.assertIsNotNone(session.process.returncode)

    def test_missing_options_fail(self):
        command = [sys.executable, '-u', '-c', "input(); print('usiok'); input()"]
        with self.assertRaisesRegex(usi_smoke.ProtocolError, 'options'):
            usi_smoke.exercise(command=command)

    def test_illegal_bestmove_fails(self):
        program = ("import sys\n"
                   f"options = {usi_smoke.OPTIONS!r}\n"
                   "for line in sys.stdin:\n"
                   " if line.strip() == 'usi':\n"
                   "  print('\\n'.join(options)); print('usiok', flush=True)\n"
                   " elif line.strip() == 'isready': print('readyok', flush=True)\n"
                   " elif line.startswith('go'): print('bestmove 7g7f', flush=True)\n")
        with self.assertRaisesRegex(usi_smoke.ProtocolError, 'illegal'):
            usi_smoke.exercise(command=[sys.executable, '-u', '-c', program])

    def test_model_discovery_distinguishes_absent_and_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'models').mkdir()
            (root / 'models/index.json').write_text(json.dumps({'models': [
                {'name': 'example', 'game': 'shogi'}, {'name': 'other', 'game': 'hasami'}]}))
            self.assertEqual(usi_smoke.installed_models(root), [])
            model = root / 'models/example'
            model.mkdir()
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                usi_smoke.installed_models(root)
            for name in ('manifest.json', 'weights.pt'):
                (model / name).write_text('fixture')
            self.assertEqual(usi_smoke.installed_models(root), [model])


if __name__ == '__main__':
    unittest.main()
