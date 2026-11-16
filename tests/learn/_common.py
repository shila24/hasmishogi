"""Shared helpers for hasami_learn tests; everything skips when torch is missing."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import numpy  # noqa: F401
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False

requires_torch = unittest.skipUnless(TORCH_AVAILABLE, "torch/numpy not installed (use .venv)")


class LearnTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def tiny(self, algorithm: str, **params):
        from hasami_learn.config import tiny_config

        config = tiny_config(algorithm, run_id=f"test-{algorithm}", **params)
        config.run.output_dir = str(self.tmp / "runs")
        return config
