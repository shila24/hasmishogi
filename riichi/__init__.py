"""Dependency-free four-player Riichi hand analysis and scoring."""
from .rules import RiichiRules

__all__ = ['RiichiRules']


def load_tests(loader, standard_tests, pattern):
    """Support repository unittest discovery when this package was imported first.

    The mandated `discover -s tests` command otherwise confuses tests/riichi with
    the production package. Test imports occur only when unittest calls this hook.
    """
    from pathlib import Path
    from unittest import TestLoader

    root = Path(__file__).resolve().parent.parent
    test_dir = root / 'tests' / 'riichi'
    if test_dir.is_dir():
        standard_tests.addTests(TestLoader().discover(str(test_dir), pattern or 'test*.py', str(root)))
    return standard_tests
