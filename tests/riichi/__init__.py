"""Riichi tests, including compatibility with discovery without an explicit top."""
from pathlib import Path

# `discover -s tests` imports this directory as `riichi`. Make the production
# submodules available in that namespace without copying or replacing the rules.
if __name__ == 'riichi':
    __path__.append(str(Path(__file__).resolve().parents[2] / 'riichi'))
    from .rules import RiichiRules
