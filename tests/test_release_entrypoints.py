"""Check runnable release entry points using only the standard library."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReleaseEntryTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('riichi'), 'Riichi is not exported')
    def test_riichi_score_cli(self):
        result = subprocess.run([sys.executable, '-m', 'riichi.cli', 'score',
                                 '123m123p123s45s77p', '--win', '6s', '--tsumo', '--riichi'],
                                cwd=ROOT, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        score = json.loads(result.stdout)
        self.assertEqual(score['payment']['total'], 8000)
        self.assertGreater(score['han'], 0)

    @unittest.skipUnless((ROOT / 'scripts/usi_smoke.py').exists(), 'USI is not exported')
    def test_usi_subprocess(self):
        result = subprocess.run([sys.executable, 'scripts/usi_smoke.py', '--classical-only'],
                                cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertTrue(json.loads(result.stdout)['ok'])
