"""Compare Python/native PUCT with identical CPU/CUDA evaluation batches.

JSON on stdout reports observed wall time, evaluator time, and their difference
(tree work plus bridge/import/allocation overhead). There is no playing-strength
claim. Both search implementations use the C++ game backend and a seeded MLP.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def learning_comparison(directory, seed):
    """Two bounded CPU runs; report all observations without a strength claim."""
    import torch
    from hasami.agents import SearchLimits
    from hasami_learn.config import TrainingConfig
    from hasami_learn.diagnostics import diagnose
    from hasami_learn.evaluation import play_matches
    from hasami_learn.inference import create_agent
    from hasami_learn.trainer import train_from_config

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    rows = []
    for mode, mix in (("puct", 0.), ("gumbel", .5)):
        cfg = TrainingConfig.load(ROOT / "configs/alphazero.json")
        cfg.run.output_dir = str(directory)
        cfg.run.run_id = mode
        cfg.run.seed = seed
        cfg.run.iterations = 10
        cfg.run.backend = "cpp"
        cfg.run.eval_every = 0
        cfg.run.checkpoint_every = 10
        cfg.run.keep_checkpoints = 10
        cfg.device.device, cfg.device.amp, cfg.device.threads = "cpu", "off", 1
        cfg.device.deterministic = True
        cfg.network.params = {"channels": 8, "blocks": 1}
        cfg.algorithm.params = {"simulations": 64, "batch_games": 2, "search_backend": "cpp",
                                "search_threads": 1, "root_selection": mode, "value_target_mix": mix}
        cfg.selfplay.games_per_iteration = 2
        cfg.selfplay.max_plies = 48
        cfg.selfplay.workers = 1
        cfg.replay.capacity, cfg.replay.batch_size, cfg.replay.min_size = 4096, 16, 16
        cfg.optimizer.updates_per_iteration = 2
        cfg.datastore.enabled = True
        cfg.replay.persist = True
        print(f"Training {mode}: 10 CPU iterations", file=sys.stderr, flush=True)
        start = time.perf_counter()
        checkpoint = train_from_config(cfg, echo=False)
        seconds = time.perf_counter() - start
        report = diagnose(checkpoint, checkpoint.parent.parent, "cpu", 32, [0, 64], seed)
        (directory / f"{mode}-diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
        matches = {}
        for opponent in ("random", "greedy", "alphabeta:1"):
            agent = create_agent(str(checkpoint), device="cpu", seed=seed, simulations=64, amp="off")
            baseline, _ = agent.spec.baseline_agent(opponent, seed)
            result = play_matches(agent, baseline, 4, spec=agent.spec, backend="cpp", max_plies=80,
                                  limits=SearchLimits(1, 3600000, 64),
                                  opponent_limits=SearchLimits(1, 3600000, 1000000),
                                  seed=seed, opening_seed=seed, opening_plies=6)
            matches[opponent] = result.to_dict()
        rows.append({"mode": mode, "value_target_mix": mix, "training_seconds": seconds,
                     "checkpoint": str(checkpoint), "config": cfg.to_dict(),
                     "target_statistics": report["target_statistics"],
                     "target_audit": report["target_audit"], "tactical_probes": report["tactical_probes"],
                     "seeded_opening_evaluation": matches})
        print(f"Completed {mode}: {seconds:.3f} training seconds", file=sys.stderr, flush=True)
    report = {"device": "cpu", "seed": seed, "iterations": 10,
              "scaled_from": "configs/autoloop/hasami-debug.json with configs/alphazero.json",
              "scaling": "64 simulations retained; CPU/AMP off/one torch and search thread; ResNet 8 channels/1 block, 2 games and batch_games=2, cap 48 plies, replay 4096/batch 16, 2 updates per iteration; other base settings retained.",
              "evaluation": "4 games/opponent, 6 seeded opening plies, cap 80, 64 simulations/decision, sides swapped, effective sample grouped by opening actions and seating; caps remain truncations with legacy half scores.",
              "limitations": "Small CPU smoke comparison; no strength or improvement claim. Tactical probes use 32 correlated sampled positions; missing classes are reported.",
              "rows": rows}
    (directory / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    import numpy as np
    import torch
    from hasami_learn.devices import configure_threads, resolve_device, seed_everything
    from hasami_learn.games import get_spec
    from hasami_learn.networks.mlp import MlpNet
    from hasami_learn.search.mcts import MCTS, MctsConfig, batched_evaluator
    from hasami_learn.search.native_mcts import NativeMCTS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=("hasami", "shogi"), default="hasami")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--games", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=39)
    parser.add_argument("--root-selection", choices=("puct", "gumbel"), default="puct")
    parser.add_argument("--learning-comparison-dir", help="New directory for the fixed two-run CPU learning comparison")
    args = parser.parse_args()
    if args.learning_comparison_dir:
        print(json.dumps(learning_comparison(args.learning_comparison_dir, args.seed), indent=2))
        return
    if min(args.games + args.threads + [args.simulations, args.batch_size, args.rounds, args.torch_threads]) < 1:
        parser.error("counts must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        print(json.dumps({"game": args.game, "device": "cuda", "status": "unavailable", "rows": []}))
        return
    device = resolve_device(args.device)
    configure_threads(args.torch_threads)
    seed_everything(args.seed, True)
    spec = get_spec(args.game)
    encoder = spec.make_encoder()
    network = MlpNet(num_planes=encoder.num_planes, board_shape=spec.board_shape,
                     action_size=spec.action_size, hidden=128, layers=2).to(device).eval()
    evaluator = batched_evaluator(network, device, "off")
    rows = []
    for count in args.games:
        with ExitStack() as stack:
            games = []
            rng = np.random.default_rng(args.seed)
            for i in range(count):
                game = spec.create_game("cpp")
                stack.callback(game.close)
                for _ in range(i % 16):
                    if not game.outcome.terminal:
                        game.play(int(rng.choice(game.legal_actions())))
                games.append(game)
            configs = [("python", None)] + [("cpp", threads) for threads in args.threads]
            expected = None
            for backend, threads in configs:
                elapsed, inference, completed = [], [], []
                for repetition in range(args.rounds + 1):
                    eval_seconds = 0.0
                    def timed(planes, masks):
                        nonlocal eval_seconds
                        # batched_evaluator copies outputs to CPU, synchronizing CUDA.
                        begin = time.perf_counter()
                        result = evaluator(planes, masks)
                        eval_seconds += time.perf_counter() - begin
                        return result
                    search_class = MCTS if backend == "python" else NativeMCTS
                    kwargs = {} if backend == "python" else {"threads": threads}
                    search = search_class(timed, encoder, MctsConfig(simulations=args.simulations,
                                          batch_size=args.batch_size, root_selection=args.root_selection),
                                          np.random.default_rng(args.seed), spec, **kwargs)
                    start = time.perf_counter()
                    result = search.run_many(games)
                    seconds = time.perf_counter() - start
                    if expected is None:
                        expected = result
                    for a, b in zip(expected, result):
                        np.testing.assert_array_equal(a.visits, b.visits)
                        np.testing.assert_allclose(a.q_values, b.q_values, atol=1e-12, rtol=0, equal_nan=True)
                        np.testing.assert_allclose(a.policy_target, b.policy_target, atol=1e-12, rtol=0)
                        if a.selected_action != b.selected_action:
                            raise RuntimeError("Selected-action parity failed during benchmark")
                        if abs(a.root_value - b.root_value) > 1e-12 or a.simulations != b.simulations:
                            raise RuntimeError("Search parity failed during benchmark")
                    if repetition:
                        elapsed.append(seconds)
                        inference.append(eval_seconds)
                        completed.append(sum(r.simulations for r in result))
                rows.append({"backend": backend, "threads": threads, "games": count,
                             "seconds_per_run_many": statistics.median(elapsed),
                             "simulations_per_second": statistics.median(n / t for n, t in zip(completed, elapsed)),
                             "evaluator_seconds": statistics.median(inference),
                             "tree_seconds": statistics.median(t - e for t, e in zip(elapsed, inference)),
                             "completed_simulations": completed[0], "parity": True,
                             "samples": [{"seconds": t, "evaluator_seconds": e} for t, e in zip(elapsed, inference)]})
    print(json.dumps({"game": args.game, "device": str(device), "cuda_available": torch.cuda.is_available(),
                      "network": {"name": "mlp", "hidden": 128, "layers": 2},
                      "game_backend": "cpp", "simulations": args.simulations, "batch_size": args.batch_size,
                      "root_selection": args.root_selection,
                      "torch_threads": args.torch_threads, "rounds": args.rounds, "seed": args.seed,
                      "tree_timing_includes": "bridge, full-history import, encoding, allocation and tree work",
                      "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
