"""Isolated decision latency and evaluator throughput per device (docs/METHOD_ja.md#methods section 4).

1. ``PolicyAgent.choose`` latency (mean/p50/p95/p99 ms) on positions sampled from native riichi games, production
   network of the config, auxiliary outputs on and off.
2. ``BatchEvaluator`` rows/s for batch sizes 1..512 with fixed and variable shapes, and the smallest batch size that
   reaches 90 % of the best fixed-shape throughput (a starting point for ``device.inference_batch``).
Prints JSON without host names, user names or absolute paths.

    .venv/bin/python scripts/benchmark_decisions.py --devices cpu mps [--quick]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def _positions(spec, count, seed):
    """Clones of native riichi games at random decision points (random legal play)."""
    rng = random.Random(seed)
    positions = []
    game_seed = seed
    while len(positions) < count:
        game = spec.create_game("cpp", seed=game_seed)
        game_seed += 1
        stop = rng.randrange(1, 400)
        for _ in range(stop):
            actors = spec.actors(game)
            if not actors:
                break
            spec.step(game, {p: rng.choice(spec.legal_actions(game, p)) for p in actors})
        actors = spec.actors(game)
        if actors:
            positions.append((game.clone(), rng.choice(actors)))
        game.close()
    return positions


def _percentiles(values):
    values = sorted(values)
    pick = lambda q: values[min(len(values) - 1, int(round(q * (len(values) - 1))))]
    return {"mean_ms": round(sum(values) / len(values), 3), "p50_ms": round(pick(0.50), 3),
            "p95_ms": round(pick(0.95), 3), "p99_ms": round(pick(0.99), 3), "n": len(values)}


def main() -> int:
    import numpy as np
    import torch
    from hasami.agents import SearchLimits
    from hasami_learn.batched_selfplay import BatchEvaluator
    from hasami_learn.config import TrainingConfig
    from hasami_learn.devices import DEFAULT_INFERENCE_BATCH, available_devices, resolve_device
    from hasami_learn.games import get_spec
    from hasami_learn.inference import PolicyAgent
    from hasami_learn.registry import NETWORKS
    import hasami_learn.networks  # noqa: F401

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(ROOT / "configs" / "riichi-ppo-league-tonpuu.json"))
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--quick", action="store_true", help="fewer positions and rows")
    parser.add_argument("--amp", default=None, choices=["auto", "off", "on"],
                        help="mixed precision; default: the config's device.amp (the production path)")
    args = parser.parse_args()
    config = TrainingConfig.load(args.config)
    amp = args.amp or config.device.amp
    spec = get_spec(config.game.name, config.game.params)
    encoder = spec.make_encoder(config.encoder.name, **config.encoder.params)
    have = {d.split(":")[0] for d in available_devices()}
    devices = [d for d in (args.devices or sorted(have)) if d.split(":")[0] in have]
    positions = _positions(spec, 60 if args.quick else 300, 11)
    rows = 512 if args.quick else 4096
    planes = np.stack([np.asarray(encoder.encode(game, player), dtype=np.float32) for game, player in positions])
    planes = np.concatenate([planes] * (rows // len(planes) + 1))[:rows]
    report = {"environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "machine": platform.machine(), "logical_cpus": os.cpu_count(),
                              "torch_threads": torch.get_num_threads(),
                              "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
              "config": Path(args.config).name, "network": config.network.name, "amp": amp,
              "num_planes": encoder.num_planes, "skipped_devices": [d for d in (args.devices or []) if d not in devices],
              "devices": {}}
    for device_name in devices:
        device = resolve_device(device_name)
        torch.manual_seed(0)
        network = NETWORKS.get(config.network.name)(head="policy_value", num_planes=encoder.num_planes,
                                                    board_shape=spec.board_shape, action_size=spec.action_size,
                                                    **config.network.params).to(device).eval()
        entry = {"latency": {}, "evaluator": {}}
        limits = SearchLimits(1, 1000, 1000)
        for aux in (True, False):
            agent = PolicyAgent(network, encoder, device, spec=spec, aux_outputs=aux, amp=amp)
            for game, player in positions[:10]:
                agent.choose(game, limits, player=player)
            samples = []
            for game, player in positions:
                start = time.perf_counter()
                agent.choose(game, limits, player=player)
                samples.append((time.perf_counter() - start) * 1000)
            entry["latency"]["aux_on" if aux else "aux_off"] = _percentiles(samples)
        best = 0.0
        for fixed in (True, False):
            series = {}
            for batch in BATCHES:
                evaluator = BatchEvaluator(network, device, amp, batch, fixed)
                evaluator.forward(planes[: batch * 2])
                start = time.perf_counter()
                evaluator.forward(planes)
                rate = rows / (time.perf_counter() - start)
                series[str(batch)] = round(rate, 1)
                if fixed:
                    best = max(best, rate)
            entry["evaluator"]["fixed_shape" if fixed else "variable_shape"] = series
        fixed_series = entry["evaluator"]["fixed_shape"]
        entry["recommended_inference_batch"] = next(int(b) for b in fixed_series if fixed_series[b] >= 0.9 * best)
        entry["default_inference_batch"] = DEFAULT_INFERENCE_BATCH.get(device.type)
        report["devices"][device_name] = entry
        print(f"[bench] {device_name}: p95 {entry['latency']['aux_on']['p95_ms']} ms, "
              f"recommended batch {entry['recommended_inference_batch']}", file=sys.stderr, flush=True)
    for game, _ in positions:
        game.close()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
