"""CPU-only Family N measurements and optional paired-opening baseline matches.

The match uses 50 seeded six-ply openings twice (both colours) per opponent by
default. Native search has a fixed node budget. The Python baseline completes
its configured depth; any interrupted depth is an error, not a weaker opponent.
The game's max_moves rule ends long games; no experimental cap becomes a draw.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def wilson(group_scores):
    """Wilson score interval with one effective observation per paired opening."""
    n = len(group_scores)
    if not n:
        return [0.0, 1.0]
    p, z = sum(group_scores) / n, 1.959963984540054
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return [max(0, center-radius), min(1, center+radius)]


def play_group(job):
    from hasami.agents import SearchLimits
    from shogi import create_game
    from shogi.agents import AlphaBetaAgent
    from shogi.model import Rules
    from shogi.native_search import NativeSearch

    opponent_depth, group, seed, native_nodes, max_moves = job
    rng = random.Random(seed + group)
    opening = create_game("cpp", Rules(max_moves))
    try:
        for _ in range(6):
            opening.play(rng.choice(opening.legal_actions()))
        opening_codes, opening_sfen = opening.moves, opening.sfen()
    finally:
        opening.close()
    results = []
    for native_side in (0, 1):
        game = create_game("cpp", Rules(max_moves))
        python_agent = AlphaBetaAgent()
        start = time.monotonic()
        try:
            for code in opening_codes:
                game.play(code)
            decisions = []
            with NativeSearch(hash_mb=4) as search:
                while not game.outcome.terminal:
                    side = game.position.turn
                    if side == native_side:
                        search.set_position(game)
                        row = search.go(nodes=native_nodes)
                        action, depth, nodes = row["move"], row["depth"], row["nodes"]
                        if nodes != native_nodes and row["depth"] != 64:
                            raise RuntimeError("Native fixed-node budget was not completed")
                    else:
                        decision = python_agent.choose(game, SearchLimits(opponent_depth, 3600000, 2**64-1))
                        if not decision.completed:
                            raise RuntimeError("Python baseline failed to complete its configured depth")
                        action, depth, nodes = decision.action, decision.depth, decision.nodes
                    if not game.is_legal(action):
                        raise RuntimeError("Match agent returned illegal move")
                    decisions.append({"side": side, "move": action, "depth": depth, "nodes": nodes})
                    game.play(action)
            score = .5 if game.outcome.winner == -1 else float(game.outcome.winner == native_side)
            results.append({"opening_group": group, "seed": seed+group, "opening_moves": opening_codes,
                            "opening_sfen": opening_sfen, "native_side": native_side,
                            "opponent": f"alphabeta:{opponent_depth}", "native_nodes": native_nodes,
                            "rules": game.rules.to_dict(), "winner": game.outcome.winner,
                            "reason": game.outcome.reason, "native_score": score, "plies": game.ply,
                            "moves": game.moves, "decisions": decisions,
                            "elapsed_seconds": time.monotonic()-start})
        finally:
            game.close()
    return results


def executable(name):
    suffix = ".exe" if os.name == "nt" else ""
    for parent in (ROOT / "build", ROOT / "build/Release"):
        path = parent / (name + suffix)
        if path.is_file():
            return str(path)
    raise RuntimeError("Native search tools are not built; run scripts/build_native.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--games", type=int, default=0, help="Games PER opponent; 100 for acceptance evidence")
    parser.add_argument("--native-nodes", type=int, default=2000)
    parser.add_argument("--opponents", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--seed", type=int, default=490049)
    parser.add_argument("--max-moves", type=int, default=320)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--game-log", type=Path, help="Append completed opening groups for resumable evidence")
    args = parser.parse_args()
    if args.games < 0 or args.games % 2 or args.native_nodes < 1 or args.max_moves < 7:
        parser.error("games must be nonnegative/even, nodes positive, max-moves >= 7")
    bench = json.loads(subprocess.check_output([executable("shogi_search_bench"), str(args.depth)], text=True))
    fixed = json.loads(subprocess.check_output([executable("shogi_search_bench"), "64", str(args.native_nodes)], text=True))
    mates = json.loads(subprocess.check_output([executable("shogi_search_mates"),
        str(ROOT / "cpp/shogi_search/tests/data/mates-v1.tsv")], text=True))
    cpu = platform.processor()
    if Path("/proc/cpuinfo").is_file():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    result = {"format": "shogi-search-evidence-v1", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "machine": {"cpu": cpu, "platform": platform.platform(), "python": sys.version.split()[0]},
              "depth_benchmark": bench, "fixed_node_benchmark": fixed, "mate_suite": mates,
              "matches": [], "workers": args.workers,
              "interval_method": "Wilson with paired-opening mean scores; effective n = opening groups"}
    if args.games:
        games = []
        if args.game_log and args.game_log.exists():
            games = [json.loads(line) for line in args.game_log.read_text().splitlines() if line]
        for row in games:
            if row["opponent"] not in {f"alphabeta:{depth}" for depth in args.opponents} or not 0 <= row["opening_group"] < args.games//2:
                raise ValueError("Existing game log has a different requested match set")
            if row["native_nodes"] != args.native_nodes or row["rules"]["max_moves"] != args.max_moves or row["seed"] != args.seed+row["opening_group"]:
                raise ValueError("Existing game log has incompatible match settings")
        completed = {(row["opponent"], row["opening_group"]) for row in games}
        for key in completed:
            paired = [row for row in games if (row["opponent"], row["opening_group"]) == key]
            if sorted(row["native_side"] for row in paired) != [0, 1]:
                raise ValueError("Existing game log contains an incomplete or duplicated opening pair")
        jobs = [(depth, group, args.seed, args.native_nodes, args.max_moves)
                for depth in args.opponents for group in range(args.games // 2)
                if (f"alphabeta:{depth}", group) not in completed]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for rows in pool.map(play_group, jobs):
                games.extend(rows)
                if args.game_log:
                    with args.game_log.open("a", encoding="utf-8") as output:
                        for row in rows:
                            output.write(json.dumps(row, sort_keys=True) + "\n")
                        output.flush()
                print(f"completed {rows[0]['opponent']} opening {rows[0]['opening_group']}: "
                      + ", ".join(f"{row['native_score']} ({row['reason']}, {row['plies']} plies)" for row in rows), file=sys.stderr, flush=True)
        for depth in args.opponents:
            rows = [row for row in games if row["opponent"] == f"alphabeta:{depth}"]
            group_scores = [sum(row["native_score"] for row in rows if row["opening_group"] == group)/2
                            for group in range(args.games//2)]
            result["matches"].append({"opponent": f"alphabeta:{depth}", "native_nodes": args.native_nodes,
                "games": rows, "opening_groups": len(group_scores), "group_scores": group_scores,
                "wins": sum(row["native_score"] == 1 for row in rows),
                "draws": sum(row["native_score"] == .5 for row in rows),
                "losses": sum(row["native_score"] == 0 for row in rows),
                "score_rate": sum(group_scores)/len(group_scores), "wilson95": wilson(group_scores)})
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(str(args.output))
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
