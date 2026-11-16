"""Exercise the public module entry points in a fresh interpreter."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

from shogi import create_game
from shogi.notation import usi_to_move


class CliTests(unittest.TestCase):
    def run_cli(self, *args, module="shogi.cli"):
        return subprocess.run([sys.executable, "-m", module, *args],
                              cwd=Path(__file__).resolve().parents[2],
                              env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                              text=True, capture_output=True, timeout=10)

    def test_perft_and_divide(self):
        result = self.run_cli("perft", "--depth", "3")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "25470")
        result = self.run_cli("perft", "--depth", "2", "--divide")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.splitlines()), 31)
        self.assertEqual(result.stdout.splitlines()[-1], "total: 900")

    def test_sfen_and_random_reproducibility(self):
        game = create_game()
        game.play(usi_to_move(game, "7g7f"))
        result = self.run_cli("sfen", "--moves", "7g7f", module="shogi")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), game.sfen())
        first = self.run_cli("play-random", "--seed", "17", "--max-moves", "12")
        second = self.run_cli("play-random", "--seed", "17", "--max-moves", "12")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(first.stdout, second.stdout)
        self.assertEqual(first.stdout.splitlines()[-1], "winner=-1 reason=max_moves")

    def test_invalid_arguments_report_errors(self):
        for args in (("perft", "--depth", "-1"), ("sfen", "--sfen", "bad"),
                     ("sfen", "--moves", "7g7e")):
            result = self.run_cli(*args)
            self.assertEqual(result.returncode, 2)
            self.assertIn("error:", result.stderr)

    def test_discovery_after_production_import(self):
        code = ("import shogi, unittest; loader = unittest.TestLoader(); "
                "suite = loader.discover('tests', pattern='test_perft.py'); "
                "assert not loader.errors, loader.errors; "
                "assert suite.countTestCases() == 6, suite.countTestCases()")
        result = subprocess.run([sys.executable, "-c", code],
                                cwd=Path(__file__).resolve().parents[2],
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
