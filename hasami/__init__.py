"""Hasami Lab: deterministic game rules and pluggable agents."""
from .model import Rules, Position, Outcome, encode_action, decode_action
from .engine import PythonGame

__version__ = "1.0.0"


def create_game(backend="python", rules=None, position=None):
    if backend == "python":
        return PythonGame(rules, position)
    if backend == "cpp":
        from .native import NativeGame
        return NativeGame(rules, position)
    raise ValueError("Backend must be 'python' or 'cpp'")
