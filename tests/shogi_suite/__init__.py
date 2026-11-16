"""Standard shogi reference-engine conformance tests."""
from pathlib import Path

# Discovery without -t imports this directory as shogi, as for tests/riichi.
# Extend its package path to the production modules and re-export their factory.
if __name__ == "shogi":
    __path__.append(str(Path(__file__).resolve().parents[2] / "shogi"))
    from .engine import ShogiGame, create_game
    from .model import Rules
