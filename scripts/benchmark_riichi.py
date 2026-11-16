#!/usr/bin/env python3
"""Measure complete random matches through both adapters and the C++ tool."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def measure(spec, backend, games, encoding):
    rng = random.Random(24024)
    steps = 0
    start = time.perf_counter()
    for seed in range(games):
        game = spec.create_game(backend, seed=seed)
        try:
            while not game.outcome.terminal:
                actions = {}
                for p in spec.actors(game):
                    legal = spec.legal_actions(game, p)
                    if encoding:
                        spec.observe(game, p)
                        spec.legal_mask(game, p)
                    actions[p] = rng.choice(legal)
                spec.step(game, actions)
                steps += 1
        finally:
            game.close()
    seconds = time.perf_counter() - start
    return dict(backend=backend, encoding=encoding, games=games, steps=steps,
                seconds=seconds, steps_per_second=steps / seconds)


def main():
    from hasami_learn.games import get_spec
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=int, default=10)
    args = parser.parse_args()
    if args.games < 1:
        parser.error('--games must be positive')
    spec = get_spec('riichi')
    # Warm both engines and their shanten tables before measured runs.
    for backend in ('python', 'cpp'):
        measure(spec, backend, 1, True)
    rates = [measure(spec, backend, args.games, encoding)
             for encoding in (False, True) for backend in ('python', 'cpp')]
    root = Path(__file__).resolve().parents[1]
    tool = next((root / path for path in ('build/riichi_selfplay_bench',
                'build/Release/riichi_selfplay_bench.exe', 'build/riichi_selfplay_bench.exe')
                 if (root / path).is_file()), None)
    if tool is None:
        raise RuntimeError('build the C++ benchmark with scripts/build_native.py')
    rates.extend(json.loads(subprocess.check_output([str(tool), str(args.games)], text=True)))
    encoded = {row['backend']: row['steps_per_second'] for row in rates if row['encoding']}
    print(json.dumps(dict(results=rates, native_encoding_speedup=encoded['cpp'] / encoded['python']), indent=2))


if __name__ == '__main__':
    main()
