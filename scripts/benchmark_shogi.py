"""Compare identical random playouts through Python, ctypes, and the C++ core.

Encoding includes float32 planes and the float32 legal policy mask. Timings include
creation/reset, legal actions, encoding when requested, RNG choice, and play.
Each round restarts the same seed across backends; checksums guard workload parity.
Output contains no host/user names or absolute paths. No third-party imports.
"""
from array import array
import argparse
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shogi import create_game
from shogi.encoding import encode, legal_labels
from shogi.model import Rules
from shogi.native import available


def run_python(backend, steps, seed, encoding, maximum):
    rng, checksum, games = seed, 0, 1
    feature_checksum = 0.0
    rules = Rules(maximum)
    planes = array("f", [0]) * 3969
    mask = array("f", [0]) * 2188
    start = time.perf_counter()
    game = create_game(backend, rules)
    try:
        for step in range(steps):
            if game.outcome.terminal:
                game.close()
                game = create_game(backend, rules)
                games += 1
            actions = game.legal_actions()
            if encoding:
                if backend == "cpp":
                    game.encode_into(planes)
                    game.legal_mask_into(mask)
                else:
                    planes = array("f", encode(game, game.position.turn))
                    mask = array("f", [0]) * 2188
                    for label in legal_labels(game):
                        mask[label] = 1
                feature_checksum += planes[step % 3969] + mask[step % 2188]
            rng = (1664525 * rng + 1013904223) & 0xFFFFFFFF
            action = actions[rng % len(actions)]
            game.play(action)
            checksum = (checksum * 31 + action) & 0xFFFFFFFF
        seconds = time.perf_counter() - start
    finally:
        game.close()
    return {"backend": "native_bridge" if backend == "cpp" else "python", "steps": steps,
            "seed": seed, "encoding": encoding, "games_started": games, "checksum": checksum,
            "feature_checksum": feature_checksum, "seconds": seconds, "steps_per_second": steps / seconds}


def find_tool():
    name = "shogi_bench_shogi" + (".exe" if sys.platform == "win32" else "")
    for directory in (ROOT / "build", ROOT / "build/Release", ROOT / "build/Debug"):
        tool = directory / name
        if tool.is_file():
            return tool
    raise RuntimeError("C++ benchmark tool is absent; run python3 scripts/build_native.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=270027)
    parser.add_argument("--max-moves", type=int, default=320)
    args = parser.parse_args()
    if not 1 <= args.steps <= 0xFFFFFFFF or not 1 <= args.rounds or not 0 <= args.seed <= 0xFFFFFFFF:
        parser.error("steps and rounds must be positive; steps and seed must fit uint32")
    if not 1 <= args.max_moves <= 1000000:
        parser.error("max-moves must be in 1..1000000")
    if not available():
        raise RuntimeError("Native shogi library is required; run python3 scripts/build_native.py")
    tool = find_tool()
    samples = []
    # Warm up each path before collecting measurements.
    for backend in ("python", "cpp"):
        run_python(backend, 100, args.seed, True, args.max_moves)
    subprocess.run([str(tool), "100", str(args.seed), "1", str(args.max_moves)], check=True, capture_output=True)
    for encoding in (False, True):
        for round_index in range(args.rounds):
            seed = (args.seed + round_index) & 0xFFFFFFFF
            round_samples = []
            # Alternate the order to reduce a systematic warmup/thermal ordering bias.
            order = ("python", "cpp", "tool") if round_index % 2 == 0 else ("tool", "cpp", "python")
            for backend in order:
                if backend == "tool":
                    result = subprocess.run([str(tool), str(args.steps), str(seed), str(int(encoding)),
                                             str(args.max_moves)], check=True, capture_output=True, text=True)
                    sample = json.loads(result.stdout)
                else:
                    sample = run_python(backend, args.steps, seed, encoding, args.max_moves)
                sample["round"] = round_index
                round_samples.append(sample)
            signatures = {(s["steps"], s["games_started"], s["checksum"], s["feature_checksum"])
                          for s in round_samples}
            if len(signatures) != 1:
                raise RuntimeError("Benchmark trajectories or encoded checksums differ")
            samples.extend(round_samples)
    medians = {str(encoding).lower(): {
        backend: statistics.median(s["steps_per_second"] for s in samples
                                   if s["backend"] == backend and s["encoding"] == encoding)
        for backend in ("python", "native_bridge", "cpp_tool")} for encoding in (False, True)}
    speedup = medians["true"]["native_bridge"] / medians["true"]["python"]
    print(json.dumps({"benchmark": "shogi-random-play-v1", "platform": sys.platform,
                      "architecture": platform.machine(), "python_version": platform.python_version(),
                      "steps_per_round": args.steps, "rounds": args.rounds, "max_moves": args.max_moves,
                      "encoding_includes": ["float32_planes", "float32_legal_mask"],
                      "samples": samples, "median_steps_per_second": medians,
                      "native_bridge_encoding_speedup": speedup,
                      "meets_10x_threshold": speedup >= 10}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
