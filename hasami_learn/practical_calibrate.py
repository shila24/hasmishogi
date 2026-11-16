"""Calibrate internal proxy levels with side-swapped games and Elo intervals."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

from hasami.agents import SearchLimits
from .checkpoint_bundle import sha256_file
from .devices import resolve_device
from .evaluation import MatchResult, play_matches
from .games import get_spec
from .inference import MctsAgent, PolicyAgent
from .ladder import fit_elo
from .manifest import atomic_write_json, load_checkpoint
from .practical import (HELDOUT_STYLES, TRAIN_STYLES, NetworkEvaluator, ProxyAgent,
                        validate_levels)


def bisect_strength(measure, target, low=0.0, high=1.0, tolerance=50.0, max_probes=12):
    """Probe a bounded increasing strength coordinate; do not stretch unreachable levels."""
    if not low < high or tolerance < 0 or max_probes < 2:
        raise ValueError("invalid calibration search bounds")
    probes = {}
    def probe(x):
        if x not in probes:
            row = dict(measure(x))
            elo, ci = row["elo"], row["ci"]
            if not np.isfinite(elo) or not ci[0] <= elo <= ci[1] or row["games"] < 1:
                raise ValueError("invalid calibration measurement")
            probes[x] = row
        return probes[x]
    lower, upper = probe(low), probe(high)
    if lower["elo"] > upper["elo"]:
        raise ValueError("strength parameter is not monotone")
    best_x = min(probes, key=lambda x: abs(probes[x]["elo"] - target))
    if target < lower["elo"] - tolerance or target > upper["elo"] + tolerance:
        return {"available": False, "params": {}, "elo": None, "ci": None,
                "games": sum(p["games"] for p in probes.values()), "note": "target outside measured reach"}
    while abs(probes[best_x]["elo"] - target) > tolerance and len(probes) < max_probes:
        middle = (low + high) / 2
        measured = probe(middle)
        best_x = min(probes, key=lambda x: abs(probes[x]["elo"] - target))
        if measured["elo"] < target:
            low = middle
        else:
            high = middle
    best = probes[best_x]
    available = abs(best["elo"] - target) <= tolerance
    return {"available": available, "params": best.get("params", {"strength": best_x}),
            "elo": best["elo"] if available else None, "ci": best["ci"] if available else None,
            "games": sum(p["games"] for p in probes.values()),
            "note": "bisection tolerance reached" if available else "probe budget exhausted"}


def elo_intervals(names, pairs, seed, samples=200):
    """Parametric pair bootstrap of the same Bradley-Terry fit used by the ladder."""
    estimate = fit_elo(names, pairs, "random")
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in names}
    for _ in range(samples):
        resampled = []
        for a, b, result in pairs:
            counts = rng.multinomial(result.games, np.array([result.wins, result.losses, result.draws]) / result.games)
            resampled.append((a, b, MatchResult(games=result.games, wins=int(counts[0]),
                                               losses=int(counts[1]), draws=int(counts[2]))))
        ratings = fit_elo(names, resampled, "random")
        for name in names:
            draws[name].append(ratings[name])
    return {name: (estimate[name], [min(estimate[name], float(np.quantile(draws[name], 0.025))),
                                   max(estimate[name], float(np.quantile(draws[name], 0.975)))]) for name in names}


def style_params(style, strength, earlier_checkpoints):
    if style == "qre":
        return {"simulations": round(256 * strength), "beta": 16 * strength, "epsilon": 0.2 * (1 - strength)}
    if style == "temperature":
        return {"temperature": float(4 * (0.025 ** strength))}
    if style == "alphabeta":
        return {"depth": min(4, 1 + int(4 * strength)), "epsilon": 1 - strength}
    if not earlier_checkpoints:
        raise ValueError("checkpoint style needs earlier engine checkpoints in training order")
    index = min(len(earlier_checkpoints) - 1, int(strength * len(earlier_checkpoints)))
    return {"checkpoint": str(Path(earlier_checkpoints[index]).resolve())}


def calibrate(engine_checkpoint, *, output, device="cpu", backend="cpp", k=6, seed=0,
              games_per_probe=200, max_plies=None, earlier_checkpoints=(), centres=None,
              train_styles=TRAIN_STYLES, heldout_styles=HELDOUT_STYLES, smoke=False,
              max_probes=12, bootstrap_samples=200, tiny_budget=False, joint=False):
    if tiny_budget:
        if not smoke or k != 1:
            raise ValueError("tiny_budget requires smoke=True and k=1; it cannot establish a multi-level ladder")
        games_per_probe, max_probes, bootstrap_samples = 2, 2, 8
        max_plies = min(max_plies or 4, 4)
    if k < 1 or games_per_probe < 2 or games_per_probe % 2 or (games_per_probe < 200 and not smoke):
        raise ValueError("calibration needs even probes, >=200 games unless explicitly smoke")
    loaded = load_checkpoint(engine_checkpoint, resolve_device(device))
    spec = get_spec(loaded.manifest.game_id, loaded.manifest.game_params)
    encoder = spec.make_encoder(loaded.manifest.encoding_id, **loaded.manifest.encoder_params)
    engine = NetworkEvaluator(loaded.network, encoder, spec, device)
    engine.game_params, engine.manifest = loaded.manifest.game_params, loaded.manifest
    limits = SearchLimits(4, 60000, 1000000)
    names = ["random", "greedy", "alphabeta:1", "alphabeta:2", "alphabeta:3", "prior", "mcts16", "mcts64", "mcts256"]
    if tiny_budget:
        names = ["random", "prior", "mcts2"]
    def anchor(name, agent_seed):
        if name == "prior":
            return PolicyAgent(engine.network, encoder, engine.device, temperature=1,
                               rng=np.random.default_rng(agent_seed), spec=spec), limits
        if name.startswith("mcts"):
            return MctsAgent(engine.network, encoder, engine.device, simulations=int(name[4:]),
                             rng=np.random.default_rng(agent_seed), spec=spec), limits
        return spec.baseline_agent(name, agent_seed)
    pairs = []
    for index, (a, b) in enumerate(combinations(names, 2)):
        first, first_limits = anchor(a, seed + index * 2)
        second, second_limits = anchor(b, seed + index * 2 + 1)
        result = play_matches(first, second, games_per_probe, spec=spec, backend=backend, max_plies=max_plies,
                              limits=first_limits or limits, opponent_limits=second_limits or limits,
                              seed=seed + index, swap_sides=True)
        pairs.append((a, b, result))
    ratings = elo_intervals(names, pairs, seed, bootstrap_samples)
    if centres is None:
        if joint or tiny_budget:
            # A new network need not beat a classical anchor yet. Use its measured range.
            low, high = min(r[0] for r in ratings.values()), max(r[0] for r in ratings.values())
        else:
            low, high = ratings["alphabeta:1"][0], ratings["mcts64"][0]
        if k > 1 and high <= low:
            raise ValueError("engine MCTS64 anchor must be stronger than alphabeta:1")
        centres = np.linspace(low, high, k).tolist()
    if len(centres) != k or any(b <= a for a, b in zip(centres, centres[1:])):
        raise ValueError("centres must increase and match k")
    levels = []
    for index, centre in enumerate(centres):
        styles = {}
        nearest = sorted(names, key=lambda n: abs(ratings[n][0] - centre))[:2]
        for style_index, style in enumerate((*train_styles, *heldout_styles)):
            if style == "checkpoint" and not earlier_checkpoints:
                styles[style] = dict(available=False, params={}, elo=None, ci=None, games=0,
                                     note="no earlier checkpoints supplied")
                continue
            def measure(strength):
                params = style_params(style, strength, earlier_checkpoints)
                if tiny_budget:
                    if style == "qre":
                        params["simulations"] = 0
                    elif style == "alphabeta":
                        params["depth"] = 1
                measured_pairs = list(pairs)
                for offset, name in enumerate(nearest):
                    probe_seed = seed + 10000 * (index + 1) + style_index * 1000 + offset
                    proxy = ProxyAgent(engine, style, params, probe_seed)
                    opponent, budget = anchor(name, probe_seed + 1)
                    result = play_matches(proxy, opponent, games_per_probe, spec=spec, backend=backend,
                                          max_plies=max_plies, limits=limits, opponent_limits=budget or limits,
                                          seed=probe_seed, swap_sides=True)
                    measured_pairs.append(("proxy", name, result))
                elo, interval = elo_intervals(names + ["proxy"], measured_pairs, seed, bootstrap_samples)["proxy"]
                return dict(params=params, elo=elo, ci=interval, games=2 * games_per_probe)
            styles[style] = bisect_strength(measure, centre, max_probes=max_probes)
        levels.append(dict(level=index + 1, elo_center=float(centre), styles=styles))
    data = dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(), game_id=spec.game_id,
                game_params=loaded.manifest.game_params, engine_checkpoint=str(loaded.directory.resolve()),
                engine_sha256=sha256_file(loaded.directory / "weights.pt"), k=k, seed=seed,
                games_per_probe=games_per_probe, train_styles=list(train_styles), heldout_styles=list(heldout_styles),
                anchors=[dict(name=name, elo=ratings[name][0], ci=ratings[name][1],
                              games=sum(r.games for a, b, r in pairs if name in (a, b))) for name in names],
                levels=levels, notes=("SMOKE: not strength evidence. " if smoke else "") +
                "Pair bootstrap 95% Elo intervals. Caps score one half and remain experimental truncations. " +
                json.dumps(dict(max_plies=max_plies, backend=backend, max_probes=max_probes,
                                bootstrap_samples=bootstrap_samples, tiny_budget=tiny_budget, joint=joint)))
    validate_levels(data)
    atomic_write_json(output, data)
    return data


def synthetic_levels(engine_checkpoint, earlier_checkpoint, *, k=3, seed=0):
    """Explicit synthetic ladder for tiny integration tests; never strength evidence."""
    loaded = load_checkpoint(engine_checkpoint, "cpu")
    levels = []
    for level in range(1, k + 1):
        styles = {}
        for style in TRAIN_STYLES + HELDOUT_STYLES:
            def measure(strength):
                params = {"qre": {"simulations": 0, "beta": strength * 4},
                          "temperature": {"temperature": 2 - strength},
                          "checkpoint": {"checkpoint": str(Path(earlier_checkpoint).resolve())},
                          "alphabeta": {"depth": 1, "epsilon": 1 - strength}}[style]
                return dict(elo=1000 * strength, ci=[1000 * strength - 100, 1000 * strength + 100],
                            games=2, params=params)
            styles[style] = bisect_strength(measure, 1000 * level / (k + 1), tolerance=1)
        levels.append(dict(level=level, elo_center=1000 * level / (k + 1), styles=styles))
    return validate_levels(dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(),
                                game_id=loaded.manifest.game_id, game_params=loaded.manifest.game_params,
                                engine_checkpoint=str(loaded.directory.resolve()),
                                engine_sha256=sha256_file(loaded.directory / "weights.pt"), k=k, seed=seed,
                                games_per_probe=2, anchors=[dict(name="random", elo=0, ci=[0, 0], games=0)],
                                levels=levels, train_styles=list(TRAIN_STYLES), heldout_styles=list(HELDOUT_STYLES),
                                notes="SYNTHETIC SMOKE: monotone function measurements, not observed playing strength."))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-checkpoint", required=True)
    parser.add_argument("--output", "--out", required=True)
    parser.add_argument("--earlier-checkpoint", action="append", default=[])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", default="cpp")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--games-per-probe", type=int, default=200)
    parser.add_argument("--max-plies", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tiny-budget", action="store_true")
    args = vars(parser.parse_args(argv))
    args["earlier_checkpoints"] = args.pop("earlier_checkpoint")
    calibrate(**args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
