"""Bounded single-thread NNUE learning sanity experiment; not the NNUE generation generator.

Only project self-play supplies data. Rule max_moves ends games; unfinished games
are never relabelled as draws. Fixed-node matches pair colours by seeded opening.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import glob
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shogi import create_game
from shogi.model import Rules, DECLARE, DROP_BASE
from shogi.native_search import NativeSearch
from hasami_learn.nnue.data import Record, IN_CHECK, CAPTURE_BEST, write_shard, atomic_write
from hasami_learn.nnue.train import train


def generate_records(count, *, seed, cap=96, depth=1):
    """Minimal one-thread helper using the existing searcher and its complete history."""
    rng = random.Random(seed)
    emitted, game_id = 0, 0
    with NativeSearch(hash_mb=1, features=0) as teacher:
        while emitted < count:
            game = create_game("cpp", Rules(max_moves=cap))
            rows = []
            try:
                while not game.outcome.terminal:
                    teacher.set_position(game)
                    result = teacher.go(depth=depth)
                    if not result["completed"]:
                        raise RuntimeError("Teacher depth did not complete")
                    p = game.position
                    code = result["move"]
                    capture = code < DROP_BASE and p.board[(code//2) % 81] != 0
                    flags = int(game.in_check()) * IN_CHECK + int(capture) * CAPTURE_BEST + (1 << 8)
                    rows.append(Record(p, result["score"], result["depth"], result["nodes"],
                                       code, 0, game.ply, game_id, flags=flags))
                    # Diversify opening and subsequent play using only legal self-play moves.
                    actual = rng.choice(game.legal_actions()) if game.ply < 8 or rng.random() < .15 else code
                    game.play(actual)
                winner = game.outcome.winner
                for row in rows:
                    value = 0 if winner == -1 else (1 if winner == row.position.turn else -1)
                    if emitted < count:
                        yield replace(row, result=value)
                        emitted += 1
                game_id += 1
            finally:
                game.close()
            if game_id % 100 == 0:
                print(f"generated {emitted}/{count} positions from {game_id} completed games", flush=True, file=sys.stderr)


def wilson_group(scores):
    n, z = len(scores), 1.959963984540054
    p = sum(scores)/n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n))/denominator
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator
    return [center-radius, center+radius]


def match(weights, *, games=100, nodes=128, seed=50, cap=128):
    if games < 2 or games % 2:
        raise ValueError("Matches require complete colour-swapped opening pairs")
    rng = random.Random(seed)
    rows, groups = [], []
    with NativeSearch(hash_mb=1) as baseline, NativeSearch(hash_mb=1, nnue=weights) as learned:
        for group in range(games//2):
            opening_game = create_game("cpp", Rules(max_moves=cap))
            try:
                opening = []
                for _ in range(6):
                    if opening_game.outcome.terminal:
                        raise RuntimeError("Opening ended early")
                    move = rng.choice([m for m in opening_game.legal_actions() if m != DECLARE])
                    opening.append(move); opening_game.play(move)
                scores = []
                for learned_side in range(2):
                    game = opening_game.clone()
                    try:
                        while not game.outcome.terminal:
                            search = learned if game.position.turn == learned_side else baseline
                            search.set_position(game)
                            result = search.go(nodes=nodes)
                            if result["move"] not in game.legal_actions():
                                raise RuntimeError("Illegal NNUE match move")
                            game.play(result["move"])
                        score = .5 if game.outcome.winner == -1 else float(game.outcome.winner == learned_side)
                        scores.append(score)
                        rows.append({"opening_group": group, "opening": opening, "nnue_side": learned_side,
                                     "score": score, "reason": game.outcome.reason, "plies": game.ply,
                                     "moves": game.moves})
                    finally:
                        game.close()
                groups.append(sum(scores)/2)
            finally:
                opening_game.close()
            if (group+1) % 10 == 0:
                print(f"match completed {2*(group+1)}/{games} games", file=sys.stderr, flush=True)
    return {"games": games, "opening_groups": len(groups), "score": sum(groups)/len(groups),
            "wilson95_opening_groups": wilson_group(groups), "nodes_per_move": nodes,
            "rule_max_moves": cap, "seed": seed, "rows": rows}


def throughput(weights, *, nodes=3000, rounds=3):
    positions = [None, "4k4/9/9/4p4/9/4P4/9/9/4K4 b RBGSLNP2p 1"]
    rows = []
    for weights_path, label in ((None, "bootstrap"), (weights, "nnue")):
        with NativeSearch(nnue=weights_path, hash_mb=1) as search:
            for sfen in positions:
                game = create_game("cpp", sfen=sfen)
                try:
                    search.set_position(game)
                    for mode in ("fixed_nodes", "fixed_depth"):
                        observations = [search.go(**({"nodes": nodes} if mode == "fixed_nodes" else {"depth": 2}))
                                        for _ in range(rounds)]
                        for result in observations:
                            if result["move"] not in game.legal_actions():
                                raise RuntimeError("Illegal benchmark move")
                        seconds = sum(r["elapsed_ms"] for r in observations)/1000
                        total = sum(r["nodes"] for r in observations)
                        rows.append({"evaluator": label, "sfen": game.sfen(), "mode": mode,
                                     "nodes": [r["nodes"] for r in observations],
                                     "completed_depths": [r["depth"] for r in observations],
                                     "seconds": seconds, "nodes_per_second": total/seconds})
                finally:
                    game.close()
    return rows


def loop_experiment(output, *, sweep_games=24, large_games=0, device="cuda", epochs=3):
    """Measure separate native runs; CUDA unavailability is recorded, never relabelled."""
    import subprocess
    import torch
    from hasami_learn.nnue.generate import generate, tool_path, validate_manifest
    from hasami_learn.nnue.control import GENERATION, atomic_json
    from hasami_learn.nnue.match import match as paired_match

    output.mkdir(parents=True, exist_ok=True)
    report = dict(format="hlnn-loop-benchmark-v1", cuda_available=torch.cuda.is_available(),
                  generation_sweep=[], limitations=[])
    for nodes in (1000, 2000, 5000):
        for threads in (8, 16, 24):
            directory = output / f"sweep-n{nodes}-t{threads}"
            options = {**GENERATION, "games": sweep_games, "nodes": nodes, "threads": threads,
                       "hash_mb": 8, "max_hash_mb": 192, "seed": 51}
            command = ["/usr/bin/time", "-f", "%M", "-o", str(output / "rss.txt"), str(tool_path()), "--out", str(directory)]
            for key, value in options.items():
                command += ["--"+key.replace("_", "-"), str(int(value) if type(value) is bool else value)]
            with (output / "generation.log").open("a") as log:
                subprocess.run(command, check=True, stdout=log)
            manifest = validate_manifest(directory)
            row = {k: manifest[k] for k in ("completed_games", "positions", "session_seconds", "positions_per_hour", "games_per_hour")}
            row.update(nodes_per_move=nodes, threads=threads, peak_rss_kib=int((output / "rss.txt").read_text()))
            report["generation_sweep"].append(row)
            atomic_json(output / "report.json", report)
            print(json.dumps(row), flush=True)
    if large_games:
        data = output / "large-data"
        started = time.monotonic()
        manifest = generate(data, dict(games=large_games, nodes=1000, threads=24, hash_mb=8, max_hash_mb=192, seed=5100))
        report["large_generation"] = {k:manifest[k] for k in ("completed_games", "positions", "positions_per_hour", "games_per_hour")}
        report["large_generation"]["seconds"] = time.monotonic()-started
        atomic_json(output / "report.json", report)
        if device.startswith("cuda") and not torch.cuda.is_available():
            report["limitations"].append("CUDA unavailable; default-architecture GPU training and trained-weight matches not run")
        else:
            started = time.monotonic()
            pattern = str(Path(glob.escape(str(data))) / "*.hlnn1")
            report["training"] = train([pattern], output / "train", epochs=epochs, device=device)
            report["training_seconds"] = time.monotonic()-started
            weights = output / "train/weights.hlnnw1"
            report["throughput"] = throughput(weights, nodes=2000)
            for label, budget in (("equal_nodes", dict(nodes=2000, time_ms=0)), ("equal_time", dict(nodes=0, time_ms=50))):
                evidence = paired_match(weights, None, output / label, dict(games=400, threads=8, seed=5000, **budget))
                report[label] = evidence["results"]
    atomic_json(output / "report.json", report)
    return report


def training_comparison(output, *, data, device, epochs, games, max_shards=0, workers=2):
    """Reproducible factor ablation with fixed protocols; never promotes a model."""
    import torch
    from hasami_learn.devices import resolve_device
    from hasami_learn.nnue.data import digest_file
    from hasami_learn.nnue.loader import feature_batches, raw_chunks, encode_records
    from hasami_learn.nnue.model import NNUE
    from hasami_learn.nnue.control import MATCH, atomic_json
    from hasami_learn.nnue.match import match as paired_match
    from hasami_learn.nnue.loop import gate_decision

    target_device = resolve_device(device)
    matches = [glob.glob(pattern) for pattern in data]
    if any(not group for group in matches):
        raise ValueError("Benchmark data pattern matched no shards")
    paths = sorted({path for group in matches for path in group})
    if max_shards:
        paths = paths[:max_shards]
    report = dict(format="hlnn-throughput-v1", device=str(target_device),
                  cuda_available=torch.cuda.is_available(), shards=len(paths),
                  positions=sum(json.loads(Path(p+".json").read_text())["records"] for p in paths),
                  seed=5151, variants={}, forward_steps={}, limitations=[])
    entries = [[p, digest_file(p), digest_file(p+".json")] for p in paths[:min(32, len(paths))]]
    import numpy as np
    raw = np.concatenate([chunk for entry in entries for chunk in raw_chunks(entry)])[:1024]
    encode_records(raw)
    started = time.monotonic()
    for _ in range(20):
        encode_records(raw)
    seconds = time.monotonic()-started
    report["native_feature_extraction"] = dict(positions=20*len(raw), seconds=seconds,
                                                positions_per_second=20*len(raw)/seconds)
    host = next(iter(feature_batches(entries, validation=False, seed=5151, fraction=.1,
                batch_size=1024, epoch=0, shuffle_buffer=1024, backend="native")))
    indices = host["indices"].long().to(target_device)
    torch.set_num_threads(1)
    for mode in ("ordered", "vectorised"):
        torch.manual_seed(5151)
        model = NNUE().to(target_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        forward = model if mode == "ordered" else model.forward_training
        for step in range(7):
            if target_device.type == "cuda":
                torch.cuda.synchronize()
            if step == 2:
                started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss = forward(indices).square().mean()/600**2
            loss.backward()
            optimizer.step()
        if target_device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.monotonic()-started
        report["forward_steps"][mode] = dict(positions=5*len(indices), seconds=seconds,
                                             positions_per_second=5*len(indices)/seconds)
        del model, optimizer, loss
    for factorised in (False, True):
        name = "factorised" if factorised else "plain"
        directory = output / name
        checkpoint = directory / "checkpoint.pt"
        started = time.monotonic()
        metrics = train([glob.escape(p) for p in paths], directory, epochs=epochs, device=device, factorised=factorised,
                        workers=workers, feature_backend="native", seed=5151, patience=2,
                        resume=checkpoint if checkpoint.exists() else None)
        weights = directory / "weights.hlnnw1"
        variant = dict(metrics=metrics, seconds=time.monotonic()-started,
                       throughput=[json.loads(line) for line in (directory / "throughput.jsonl").read_text().splitlines()],
                       selection={k:json.loads((directory / "training.json").read_text())[k]
                                  for k in ("selected_epoch", "best_validation_loss", "early_stopped")},
                       search=throughput(weights, nodes=30000, rounds=5), matches={})
        report["variants"][name] = variant
        atomic_json(output / "report.json", report)
        for label, budget in (("nodes1000", dict(nodes=1000, time_ms=0)),
                              ("nodes10000", dict(nodes=10000, time_ms=0)),
                              ("time10ms", dict(nodes=0, time_ms=10))):
            settings = {**MATCH, "games": games, "threads": 8, "hash_mb": 8,
                        "seed": 5151, "opening_plies": 8, **budget}
            evidence_path = output / name / label
            evidence = paired_match(weights, None, evidence_path, settings)
            config = dict(match=settings, gate=dict(min_score=.55, min_lower=.5))
            decision = gate_decision(directory, evidence_path / "eval.json", config,
                                     opponent_sha256=evidence["protocol"]["opponent_sha256"])
            variant["matches"][label] = dict(results=evidence["results"], gate=decision)
            atomic_json(output / "report.json", report)
    if max_shards:
        report["limitations"].append("Bounded CPU-scale subset; not full-dataset or GPU strength evidence")
    if games < 400:
        report["limitations"].append("Fewer than 400 match games per budget; promotion thresholds unchanged")
    atomic_json(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-comparison", action="store_true")
    parser.add_argument("--data", nargs="+")
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--loop-sweep", action="store_true")
    parser.add_argument("--sweep-games", type=int, default=24)
    parser.add_argument("--large-games", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quick", action="store_true", help="200000 positions, width 16, two epochs, 100 matched games")
    parser.add_argument("--positions", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--nodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.positions is not None and args.positions < 1:
        parser.error("positions must be positive")
    output = args.out or Path(tempfile.mkdtemp(prefix="hasami-nnue-benchmark-"))
    output.mkdir(parents=True, exist_ok=True)
    if args.training_comparison:
        if not args.data or args.max_shards < 0:
            parser.error("training comparison requires --data and nonnegative --max-shards")
        training_comparison(output, data=args.data, device=args.device, epochs=args.epochs or 6,
                            games=args.games, max_shards=args.max_shards, workers=args.workers)
        return
    if args.loop_sweep:
        loop_experiment(output, sweep_games=args.sweep_games, large_games=args.large_games, device=args.device, epochs=args.epochs or 3)
        return
    data = output / "positions.hlnn1"
    start = time.monotonic()
    manifest = write_shard(data, generate_records(args.positions or (200000 if args.quick else 300000), seed=args.seed))
    generation_seconds = time.monotonic()-start
    metrics = train([glob.escape(str(data))], output / "train", epochs=args.epochs or (2 if args.quick else 3),
                    width=args.width or (16 if args.quick else 256), batch_size=2048,
                    seed=args.seed, device="cpu", threads=1)
    weights = output / "train/weights.hlnnw1"
    speed = throughput(weights)
    result = match(weights, games=args.games, nodes=args.nodes, seed=args.seed)
    report = {"format": "hlnn-benchmark-v1", "device": "cpu", "threads": 1,
              "generation": {**manifest, "seconds": generation_seconds, "depth": 1, "rule_max_moves": 96},
              "training": metrics, "throughput": speed, "match": result,
              "output": str(output), "total_seconds": time.monotonic()-start}
    atomic_write(output / "report.json", (json.dumps(report, indent=2)+"\n").encode())
    summary = {**report, "match": {k:v for k,v in result.items() if k != "rows"}}
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
