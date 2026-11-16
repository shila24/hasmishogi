"""Run reproducible evaluation games without training any model."""
import argparse
import json
from pathlib import Path
from .agents import AgentRegistry, SearchLimits
from .env import TurnBasedEnv
from .records import save_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("python", "cpp"), default="python")
    parser.add_argument("--black", default="alphabeta")
    parser.add_argument("--white", default="random")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--swap", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--time-ms", type=int, default=200)
    parser.add_argument("--nodes", type=int, default=200000)
    parser.add_argument("--max-plies", type=int, default=400)
    parser.add_argument("--agents-config")
    parser.add_argument("--output", default="arena_output")
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be positive")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    limits = SearchLimits(args.depth, args.time_ms, args.nodes)
    summary = []
    for index in range(args.games):
        registry = AgentRegistry(args.seed + index)
        if args.agents_config:
            registry.load_config(args.agents_config)
        names = [args.black, args.white]
        if args.swap and index % 2:
            names.reverse()
        players = [registry.create(name) for name in names]
        env = TurnBasedEnv(args.backend, max_plies=args.max_plies)
        try:
            status = None
            while not env.game.outcome.terminal and not env.truncated:
                turn = env.game.position.turn
                snapshot = env.game.clone()
                try:
                    decision = players[turn].choose(snapshot, limits)
                finally:
                    snapshot.close()
                status = env.step(decision.action)
            record = {"game": index, "black": names[0], "white": names[1],
                      "winner": env.game.outcome.winner, "plies": len(env.game.moves),
                      "reason": status.info["reason"] if status else env.game.outcome.reason,
                      "terminated": env.game.outcome.terminal, "truncated": env.truncated,
                      "seed": args.seed + index, "backend": args.backend,
                      "limits": vars(limits)}
            save_record(env.game, out / f"game_{index:04d}.json")
            summary.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
        finally:
            env.close()
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
