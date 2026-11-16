"""Standard-library shogi reference engine (shogi-std-v1)."""
from .engine import ShogiGame
from .model import Rules


__all__ = ["create_game", "ShogiGame", "Rules"]


def create_game(backend: str = "python", rules: Rules | None = None,
                sfen: str | None = None):
    """Construct the selected engine, with native preference for auto."""
    if backend in ("cpp", "auto"):
        from .native import NativeShogiGame, available
        if backend == "cpp" or available():
            return NativeShogiGame(rules, sfen)
    elif backend != "python":
        raise ValueError(f"Unsupported shogi backend: {backend}")
    return ShogiGame(rules, sfen)


def load_tests(loader, standard_tests, pattern):
    """Support repository discovery when the production package is imported first.

    unittest alone invokes this hook. Normal package imports never load tests.
    This follows the existing riichi package's handling of discover -s tests.
    """
    from pathlib import Path
    from unittest import TestLoader

    root = Path(__file__).resolve().parent.parent
    test_dir = root / "tests" / "shogi_suite"
    if test_dir.is_dir():
        standard_tests.addTests(TestLoader().discover(str(test_dir), pattern or "test*.py", str(root)))
    return standard_tests
