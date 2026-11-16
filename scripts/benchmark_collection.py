"""End-to-end league collection throughput (docs/METHOD_ja.md#methods section 4).

Times a full ``collect()`` (engine, encoding, inference, bookkeeping) of the riichi PPO league for each device, table
count, mode and batch size, after a warm-up collection. Optionally measures the collection of an older commit
(``--legacy-rev``) in a temporary git worktree with the same config, tables, device and seeds. Batched and reference
rows of the same setting are compared by a SHA-256 digest of everything the collection produced (every trajectory,
the rollout buffers, auxiliary targets, assignments and pool statistics). Prints JSON without host names, user names
or absolute paths.

    .venv/bin/python scripts/benchmark_collection.py --tables 32 --devices cpu mps --legacy-rev <comparison-commit>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NEW_DEVICE_KEYS = ("fixed_batch_shape",)
NEW_ALGORITHM_KEYS = ("collection", "parallel_tables", "skip_forced_decisions")
BUFFERS = ("_planes", "_masks", "_actions", "_logp_old", "_value_old", "_temperatures", "_advantages", "_returns")

# Runs inside the legacy worktree: same config edits, time one collect after a warm-up. Decisions are counted the
# same way as the new collection does (every seat's action of every joint step).
LEGACY_SNIPPET = r'''
import json, sys, time
from hasami_learn.config import TrainingConfig
from hasami_learn.trainer import Trainer
cfg_path, tables, warmup = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
def run(n, iteration):
    config = TrainingConfig.load(cfg_path)
    config.selfplay.games_per_iteration = n
    trainer = Trainer(config, echo=False)
    try:
        learner, decisions = trainer.learner, [0]
        ingest = learner.ingest_trajectories
        def counting(trajectories):
            decisions[0] += sum(len(step) for t in trajectories for step in t.joint_actions)
            return ingest(trajectories)
        learner.ingest_trajectories = counting
        start = time.perf_counter()
        stats = learner.collect(iteration)
        return time.perf_counter() - start, stats, decisions[0]
    finally:
        trainer.close()
run(warmup, 999)
seconds, stats, decisions = run(tables, 1)
print(json.dumps({"seconds": seconds, "plies": stats.plies, "games": stats.games, "decisions": decisions,
                  "samples": stats.extra.get("samples")}))
'''


def _config_dict(path: Path, output: Path, device: str, tables: int, batch, mode: str, legacy: bool, amp) -> dict:
    data = json.loads(path.read_text())
    data["run"].update(output_dir=str(output), run_id="bench", iterations=1, backend="cpp")
    data["device"]["device"] = device
    if amp is not None:
        data["device"]["amp"] = amp
    data["selfplay"]["games_per_iteration"] = tables
    data["evaluation"]["games"] = 0
    if legacy:
        for key in NEW_DEVICE_KEYS:
            data["device"].pop(key, None)
        for key in NEW_ALGORITHM_KEYS:
            data["algorithm"]["params"].pop(key, None)
        data["device"]["inference_batch"] = 64  # unused by the legacy collection; must be an integer there
    else:
        data["device"]["inference_batch"] = batch
        data["algorithm"]["params"]["collection"] = mode
    return data


def _digest(trajectories, learner, extra: dict, tmp: Path) -> str:
    """SHA-256 over the exact data of one collection (arrays by dtype, shape and bytes; the rest as JSON)."""
    import numpy as np

    digest = hashlib.sha256()

    def add(value):
        if isinstance(value, (np.ndarray, np.generic)):
            array = np.ascontiguousarray(value)
            digest.update(f"{array.dtype}{array.shape}".encode())
            digest.update(array.tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True, default=repr).encode())

    for trajectory in trajectories:
        for array in (*trajectory.planes, *trajectory.masks):
            add(np.asarray(array))
        add([trajectory.actions, trajectory.players, trajectory.extras, trajectory.rewards, trajectory.joint_actions,
             trajectory.step_info, trajectory.truncated, trajectory.winner, trajectory.reason, trajectory.plies])
        for policy in trajectory.policies:
            add(None if policy is None else np.asarray(policy))
    for name in BUFFERS:
        for value in getattr(learner, name):
            add(np.asarray(value))
    for target in learner._aux_targets:
        for key in sorted(target):
            add(key)
            add(np.asarray(target[key]))
    kept = {key: value for key, value in extra.items() if key not in ("timing", "collection")}
    add(json.loads(json.dumps(kept, default=repr).replace(json.dumps(str(tmp))[1:-1], "<tmp>")))
    return digest.hexdigest()


def _collect(config_path: Path, iteration: int, tmp: Path):
    from hasami_learn.config import TrainingConfig
    from hasami_learn.trainer import Trainer

    trainer = Trainer(TrainingConfig.load(config_path), echo=False)
    try:
        learner, trajectories = trainer.learner, []
        ingest = learner.ingest_trajectories
        learner.ingest_trajectories = lambda batch: (trajectories.extend(batch), ingest(batch))[1]
        start = time.perf_counter()
        stats = learner.collect(iteration)
        seconds = time.perf_counter() - start
        return seconds, stats, len(learner._actions), _digest(trajectories, learner, stats.extra, tmp)
    finally:
        trainer.close()


def _rates(seconds: float, plies: int, decisions: int | None, samples: int) -> dict:
    return {"seconds": round(seconds, 3), "joint_steps": plies, "joint_steps_per_s": round(plies / seconds, 1),
            "decisions": decisions, "decisions_per_s": round(decisions / seconds, 1) if decisions else None,
            "learner_samples": samples, "samples_per_s": round(samples / seconds, 1)}


def measure(config: Path, device: str, tables: int, batch, mode: str, warmup: int, amp) -> dict:
    with tempfile.TemporaryDirectory(prefix="hasami-bench-") as tmp:
        tmp = Path(tmp)
        for name, n in (("warmup.json", warmup), ("config.json", tables)):
            data = _config_dict(config, tmp / "runs", device, n, batch, mode, False, amp)
            (tmp / name).write_text(json.dumps(data))
        _collect(tmp / "warmup.json", 999, tmp)
        seconds, stats, samples, digest = _collect(tmp / "config.json", 1, tmp)
    timing = stats.extra["timing"]
    return {"device": device, "tables": tables, "mode": mode, "amp": data["device"]["amp"],
            "inference_batch": stats.extra["inference_batch"],
            **_rates(seconds, stats.plies, timing["decisions"] or None, samples), "timing": timing,
            "_digest": digest}


def _sanitize(text: str, *paths: Path) -> str:
    for path in sorted({str(p) for p in (*paths, ROOT, Path.home())}, key=len, reverse=True):
        text = text.replace(path, "<path>")
    return text


def measure_legacy(rev: str, config: Path, device: str, tables: int, warmup: int, amp) -> dict:
    lib = ROOT / "build" / ("libriichi_native.dylib" if sys.platform == "darwin" else "libriichi_native.so")
    row = {"device": device, "tables": tables, "mode": f"legacy@{rev}"}
    with tempfile.TemporaryDirectory(prefix="hasami-legacy-") as tmp:
        tree = Path(tmp) / "tree"
        added = subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--detach", str(tree), rev],
                               capture_output=True, text=True)
        if added.returncode:
            return {**row, "error": "git worktree add failed",
                    "stderr_tail": _sanitize(added.stderr, Path(tmp)).strip().splitlines()[-3:]}
        try:
            cfg = Path(tmp) / "config.json"
            data = _config_dict(config, Path(tmp) / "runs", device, tables, None, "", True, amp)
            cfg.write_text(json.dumps(data))
            env = dict(os.environ, PYTHONPATH=str(tree), RIICHI_LIB=str(lib))
            out = subprocess.run([sys.executable, "-c", LEGACY_SNIPPET, str(cfg), str(tables), str(warmup)],
                                 cwd=tree, env=env, capture_output=True, text=True)
            if out.returncode:
                row.update(error=f"legacy collection exited with {out.returncode}",
                           stderr_tail=_sanitize(out.stderr, Path(tmp)).strip().splitlines()[-5:])
            else:
                result = json.loads(out.stdout.strip().splitlines()[-1])
                row.update(amp=data["device"]["amp"],
                           **_rates(result["seconds"], result["plies"], result["decisions"], result["samples"]))
        finally:
            removed = subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force", str(tree)],
                                     capture_output=True, text=True)
            if removed.returncode:
                subprocess.run(["git", "-C", str(ROOT), "worktree", "prune"], capture_output=True)
                row["worktree_cleanup"] = "remove failed; pruned"
                print(f"[bench] warning: git worktree remove failed: {_sanitize(removed.stderr, Path(tmp)).strip()}",
                      file=sys.stderr, flush=True)
    return row


def compute_speedups(results: list[dict]) -> list[dict]:
    """Ratios of every batched row to the reference at the same inference batch and to the legacy run."""
    speedups = []
    for row in results:
        if row.get("mode") != "batched":
            continue
        for other in results:
            if (other.get("device"), other.get("tables")) != (row["device"], row["tables"]) or \
                    not other.get("mode") or other["mode"] == "batched" or other.get("error"):
                continue
            # The reference is compared at the same inference batch; the legacy run is the common baseline.
            if other["mode"] == "reference" and other["inference_batch"] != row["inference_batch"]:
                continue
            ratio = lambda key: round(row[key] / other[key], 2) if row.get(key) and other.get(key) else None
            speedups.append({"device": row["device"], "tables": row["tables"], "batch": row["inference_batch"],
                             "versus": other["mode"], "versus_batch": other.get("inference_batch"),
                             "decisions_per_s_ratio": ratio("decisions_per_s"),
                             "samples_per_s_ratio": ratio("samples_per_s"),
                             "joint_steps_per_s_ratio": ratio("joint_steps_per_s")})
    return speedups


def main() -> int:
    import torch
    from hasami_learn.devices import available_devices

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(ROOT / "configs" / "riichi-ppo-league-tonpuu.json"))
    parser.add_argument("--tables", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--modes", nargs="+", default=["batched", "reference"], choices=["batched", "reference"])
    parser.add_argument("--inference-batch", type=int, nargs="+", default=None, help="default: the device default")
    parser.add_argument("--amp", default=None, choices=["auto", "off", "on"],
                        help="mixed precision; default: the config's device.amp (the production path)")
    parser.add_argument("--legacy-rev", default=None, help="also time this commit's collection in a git worktree")
    parser.add_argument("--warmup-tables", type=int, default=4)
    args = parser.parse_args()
    config = Path(args.config)
    have = {d.split(":")[0] for d in available_devices()}
    devices = [d for d in (args.devices or sorted(have)) if d.split(":")[0] in have]
    skipped = [d for d in (args.devices or []) if d not in devices]
    results = []
    for device in devices:
        for tables in args.tables:
            digests = {}
            for batch in (args.inference_batch or [None]):
                for mode in args.modes:
                    row = measure(config, device, tables, batch, mode, args.warmup_tables, args.amp)
                    digests.setdefault(batch, {})[mode] = row.pop("_digest")
                    results.append(row)
                    print(f"[bench] {device} tables={tables} {mode} batch={row['inference_batch']}: "
                          f"{row['decisions_per_s']} decisions/s", file=sys.stderr, flush=True)
            for batch, modes in digests.items():
                if len(modes) == 2:
                    # Exact data equality (section 5.1) is asserted by tests/learn/test_league_collection_parity.py;
                    # this reports it for the benchmarked setting.
                    results.append({"device": device, "tables": tables, "inference_batch": batch,
                                    "modes_identical": modes["batched"] == modes["reference"],
                                    "compared": "sha256 of trajectories, rollout buffers, aux targets, assignments, pool"})
            if args.legacy_rev:
                results.append(measure_legacy(args.legacy_rev, config, device, tables, args.warmup_tables, args.amp))
    speedups = compute_speedups(results)
    report = {"environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "machine": platform.machine(), "logical_cpus": os.cpu_count(),
                              "torch_threads": torch.get_num_threads(),
                              "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
              "config": config.name, "skipped_devices": skipped, "results": results, "speedups": speedups}
    print(json.dumps(report, indent=2))
    return 1 if any(row.get("error") for row in results) else 0


if __name__ == "__main__":
    sys.exit(main())
