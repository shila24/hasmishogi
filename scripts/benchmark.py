"""Measure a fixed perft workload, not playing strength or bitboard-only speedup."""
import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hasami import create_game


def python_perft(game, depth):
    if depth == 0:
        return 1
    if game.outcome.terminal:
        return 0
    nodes = 0
    for action in game.legal_actions():
        game.play(action)
        try:
            nodes += python_perft(game, depth - 1)
        finally:
            game.undo()
    return nodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output")
    args = parser.parse_args()
    if not 0 <= args.depth <= 4 or args.rounds < 1:
        parser.error("Use depth 0..4 and positive rounds")
    results = {"platform": platform.platform(), "python": sys.version,
               "depth": args.depth, "rounds": args.rounds, "results": []}
    leaf_counts = []
    for backend in ("cpp", "python"):
        game = create_game(backend)
        try:
            run = (lambda: game.perft(args.depth)) if backend == "cpp" else (lambda: python_perft(game, args.depth))
            run()
            timings = []
            for _ in range(args.rounds):
                start = time.perf_counter()
                leaves = run()
                timings.append(time.perf_counter() - start)
            median = statistics.median(timings)
            leaf_counts.append(leaves)
            results["results"].append({"backend": backend, "leaves": leaves,
                                       "median_seconds": median, "leaf_nodes_per_second": leaves / median,
                                       "all_seconds": timings})
        finally:
            game.close()
    if len(set(leaf_counts)) != 1:
        raise AssertionError("Engine perft counts differ")
    text = json.dumps(results, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
