"""Check one installed model on its initial game without mutating the game."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from hasami.agents import SearchLimits
    from hasami_learn.inference import create_agent
    from hasami_learn.games import get_spec

    name = sys.argv[1]
    index = json.loads((ROOT / "models/index.json").read_text(encoding="utf-8"))
    entry = next(row for row in index["models"] if row["name"] == name)
    agent = create_agent(ROOT / "models" / name, device="cpu", simulations=2)
    spec = get_spec(entry["game"])
    game = spec.create_game(backend="python")
    try:
        player, = spec.actors(game)
        work = game.clone()
        try:
            choice = agent.choose(work, SearchLimits(max_depth=1, time_ms=1000, node_limit=2), player=player)
        finally:
            work.close()
        assert type(choice.action) is int and choice.action in spec.legal_actions(game, player)
    finally:
        game.close()


if __name__ == "__main__":
    main()
