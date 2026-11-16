"""Check one installed model on its initial game without mutating the game."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def nnue_smoke(directory):
    """Load the installed weights in native search and in a real USI subprocess."""
    from hasami.agents import SearchLimits
    from shogi import create_game
    from shogi.gui_support import load_strong_agent
    from shogi.notation import usi_to_move
    from usi_smoke import Session

    agent = load_strong_agent(directory)
    game = create_game('python')
    try:
        before = game.position
        choice = agent.choose(game, SearchLimits(max_depth=2, time_ms=1000, node_limit=1000))
        assert type(choice.action) is int and choice.action in game.legal_actions()
        assert game.position == before
        session = Session([sys.executable, '-m', 'shogi.usi'], ROOT, 60)
        try:
            session.send('usi')
            session.until('usiok')
            session.send('setoption name Engine value alphabeta-cpp')
            session.send('setoption name EvalFile value ' + str(directory / 'weights.hlnnw1'))
            session.send('isready')
            session.until('readyok')
            session.send('position startpos')
            session.send('go nodes 1000 depth 2')
            reply = session.until('bestmove')[-1].split()[1]
            assert usi_to_move(game, reply) in game.legal_actions()
            session.quit()
        finally:
            session.close()
    finally:
        agent.close()
        game.close()


def main():
    from hasami.agents import SearchLimits

    name = sys.argv[1]
    index = json.loads((ROOT / "models/index.json").read_text(encoding="utf-8"))
    entry = next(row for row in index["models"] if row["name"] == name)
    if entry.get('asset_kind') == 'nnue':
        nnue_smoke(ROOT / 'models' / name)
        return
    from hasami_learn.inference import create_agent
    from hasami_learn.games import get_spec
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
