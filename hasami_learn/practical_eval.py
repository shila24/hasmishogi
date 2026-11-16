"""Reproducible E1-E8 proxy experiments; tiny reports are explicitly smoke-only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .checkpoint_bundle import sha256_file
from .evaluation import wilson_interval
from .games.base import terminal_rewards
from .manifest import atomic_write_json, check_manifest_matches_spec
from .practical import (PracticalAdvisor, create_advisor, mover, sample_distribution, validate_artifact)
from .search.mcts import MCTS, MctsConfig, batched_evaluator

# Keep the original single-advisor experiment group stable for E1-E7 callers.
EXPERIMENTS = ("single_intervention", "robustness", "tag_team", "difficulty", "calibration", "levels", "cost")


def decision_rng(seed, player, ply, stream=0):
    """Common random numbers indexed by continuation ply/seat, independent of arm."""
    return np.random.default_rng([int(seed), int(player), int(ply), int(stream)])


def bootstrap_metric(metric, values, *, seed, statistic=np.mean, samples=500, **labels):
    data = np.asarray(values, dtype=np.float64)
    if not len(data):
        return dict(metric=metric, value=None, ci_low=None, ci_high=None, n=0,
                    ci_method="paired bootstrap 95% (no observations)", **labels)
    rng = np.random.default_rng(seed)
    draws = [float(statistic(data[rng.integers(len(data), size=len(data))])) for _ in range(samples)]
    value = float(statistic(data))
    return dict(metric=metric, value=value, ci_low=min(value, float(np.quantile(draws, 0.025))),
                ci_high=max(value, float(np.quantile(draws, 0.975))), n=len(data),
                ci_method="paired bootstrap 95%", **labels)


def wilson_metric(metric, scores, **labels):
    if not scores:
        return dict(metric=metric, value=None, ci_low=None, ci_high=None, n=0, ci_method="Wilson 95%", **labels)
    low, high = wilson_interval(sum(scores), len(scores))
    return dict(metric=metric, value=float(np.mean(scores)), ci_low=low, ci_high=high,
                n=len(scores), ci_method="Wilson 95%", **labels)


def _agents(advisor, levels, seed, style):
    agents = []
    for seat, level in enumerate(levels):
        if style == "mixture":
            available = [name for name in advisor.level_data["train_styles"]
                         if name in advisor.levels[level - 1].styles]
            name = str(decision_rng(seed, seat, 0, 2).choice(available))
        else:
            name = style
        agents.append(advisor.proxy_agent(level, style=name, seed=seed + seat))
    return agents


def continuation(advisor, game, levels, seed, *, max_plies, style="mixture", intervention=None,
                 rho=0.0, team_player=0, trace=None):
    """Fixed identity rewards; caps score half but are counted independently of draws."""
    spec = advisor.spec
    if getattr(advisor, "general", False):
        if style not in ("mixture", "temperature") or intervention is not None:
            raise ValueError("general continuations support temperature proxies and E1 interventions only")
        return _general_continuation(advisor, game, levels, seed, max_plies=max_plies, trace=trace)
    work = game.clone()
    agents = _agents(advisor, levels, seed, style)
    truncated = False
    ended = None  # the adapter's terminal StepOutcome; its result stands even if the engine outcome stays ongoing
    try:
        for ply in range(max_plies):
            if work.outcome.terminal:
                break
            player = mover(spec, work)
            rng = decision_rng(seed, player, ply)
            if trace is not None:
                trace.append((seed, player, ply))
            distribution = agents[player].distribution(work)
            action = sample_distribution(distribution, rng)
            if intervention is not None and player == team_player and decision_rng(seed, player, ply, 1).random() < rho:
                action = intervention(work, levels[player], levels[1 - player])
            outcome = spec.step(work, {player: action})
            if outcome.truncated:
                truncated = True
                break
            if outcome.terminal:
                ended = outcome
                break
        truncated = truncated or (ended is None and not work.outcome.terminal)
        if truncated:
            scores = [0.5, 0.5]
        else:
            rewards = ended.rewards if ended is not None and ended.rewards is not None else terminal_rewards(spec, work)
            scores = [(float(v) + 1) / 2 for v in rewards]
        return scores, truncated
    finally:
        work.close()


def paired_intervention(advisor, game, levels, actions, seeds, *, max_plies, style="mixture", traces=None,
                        player=None):
    if getattr(advisor, "general", False):
        if style not in ("mixture", "temperature"):
            raise ValueError("general interventions support temperature proxies only")
        return _general_intervention(advisor, game, levels, actions, seeds, max_plies=max_plies,
                                     traces=traces, player=player)
    player = mover(advisor.spec, game)
    samples, truncated = {}, {}
    for arm, action in actions.items():
        samples[arm], truncated[arm] = [], []
        child = game.clone()
        try:
            outcome = advisor.spec.step(child, {player: action})
            for seed in seeds:
                if outcome.truncated or outcome.terminal:
                    score = 0.5 if outcome.truncated else (terminal_rewards(advisor.spec, child)[player] + 1) / 2
                    cap = outcome.truncated
                else:
                    trace = [] if traces is not None else None
                    scores, cap = continuation(advisor, child, levels, seed, max_plies=max_plies, style=style, trace=trace)
                    score = scores[player]
                    if traces is not None:
                        traces[(arm, seed)] = trace
                samples[arm].append(score)
                truncated[arm].append(float(cap))
        finally:
            child.close()
    return samples, truncated


def reference_move(advisor, game, simulations=800):
    engine = advisor.engine
    search = MCTS(batched_evaluator(engine.network, engine.device, "off"), engine.encoder,
                  MctsConfig(simulations=simulations, add_noise=False), np.random.default_rng(0), spec=advisor.spec)
    result = search.run(game)
    legal = advisor.spec.legal_actions(game, mover(advisor.spec, game))
    return min(legal, key=lambda a: (-result.visits[a], a))


@dataclass(frozen=True)
class SampledPosition:
    game: object
    phase: str
    index: int
    seed: int


def sampled_positions(advisor, levels, seed, *, count, max_plies, backend):
    """Sample by ply bucket from independent held-out proxy games, preserving history."""
    spec = advisor.spec
    positions = []
    try:
        for index in range(count):
            game = spec.create_game(backend, seed=seed + index)
            agents = _agents(advisor, levels, seed + index, "mixture")
            buckets = [(start, min(end, max_plies)) for start, end in ((0, 20), (20, 60), (60, max_plies))
                       if start < min(end, max_plies)]
            low, high = buckets[index % len(buckets)]
            ply = int(decision_rng(seed, 0, index, 5).integers(low, high))
            try:
                outcome = None
                for turn in range(ply):
                    player = mover(spec, game)
                    action = sample_distribution(agents[player].distribution(game), decision_rng(seed + index, player, turn))
                    outcome = spec.step(game, {player: action})
                    if outcome.terminal or outcome.truncated:
                        break
                if not game.outcome.terminal and (outcome is None or not (outcome.terminal or outcome.truncated)):
                    phase = "early" if ply < 20 else "middle" if ply < 60 else "late"
                    positions.append(SampledPosition(game.clone(), phase, index, seed + index))
            finally:
                game.close()
        return positions
    except BaseException:
        for sample in positions:
            sample.game.close()
        raise


def _calibration_metrics(prefix, observations, seed, **labels):
    if not observations:
        return [bootstrap_metric(prefix + metric, [], seed=seed, **labels)
                for metric in ("brier", "ece", "log_loss")]
    data = np.asarray(observations)
    def measure(rows, metric):
        p, y = rows[:, 0], rows[:, 1]
        if metric == "brier":
            return np.mean((p - y) ** 2)
        if metric == "log_loss":
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p))
        bin_count = 10
        bins = np.minimum((p * bin_count).astype(int), bin_count - 1)
        return sum(np.mean(bins == b) * abs(np.mean(p[bins == b]) - np.mean(y[bins == b]))
                   for b in range(bin_count) if np.any(bins == b))
    return [bootstrap_metric(prefix + metric, np.arange(len(data)), seed=seed,
                             statistic=lambda indices, m=metric: measure(data[indices.astype(int)], m), **labels)
            for metric in ("brier", "ece", "log_loss")]


def training_compute(checkpoint):
    """Read measured phase wall time through this checkpoint; missing data stays null."""
    directory = Path(checkpoint).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    metrics = next((root / "metrics.jsonl" for root in (directory, *list(directory.parents)[:3])
                    if (root / "metrics.jsonl").is_file()), None)
    result = dict(checkpoint=str(directory), metrics=str(metrics) if metrics else None,
                  through_iteration=manifest["iteration"], wall_clock_seconds=None, device_seconds={}, gpu_hours=None,
                  coverage="sum of logged collect/update/evaluation wall times; excludes checkpoint I/O and setup")
    if metrics is None:
        return result
    rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    measured = [row for row in rows if row["iteration"] <= manifest["iteration"]
                and row["kind"] in ("collect", "update", "evaluation") and "seconds" in row]
    if not measured:
        return result
    by_device = defaultdict(float)
    for row in measured:
        seconds = float(row["seconds"])
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("training metrics contain invalid wall time")
        by_device[row.get("device", "unknown")] += seconds
    gpu_seconds = sum(seconds for device, seconds in by_device.items() if device.startswith(("cuda", "mps")))
    result.update(wall_clock_seconds=sum(by_device.values()), device_seconds=dict(by_device),
                  gpu_hours=gpu_seconds / 3600 if "unknown" not in by_device else None)
    return result


def _compare_difficulty(advisor, practical, game, action, pair, seed, style):
    """E4 uses the same engine and continuation policies for every arm."""
    player = mover(advisor.spec, game)
    agents = _agents(advisor, pair, seed, style)
    work = game.clone()
    shares, slips, errors, only = [], [], [], 0
    try:
        outcome = advisor.spec.step(work, {player: action})
        for ply in range(4):
            if outcome.terminal or outcome.truncated:
                break
            seat = mover(advisor.spec, work)
            distribution = agents[seat].distribution(work)
            if seat == player:
                q = advisor.engine.one_ply(work)
                bad = {a: max(q.values()) - value > 0.2 for a, value in q.items()}
                shares.append(float(np.mean(list(bad.values()))))
                errors.append(sum(distribution.get(a, 0) for a in q if bad[a]))
                only += int(sum(not value for value in bad.values()) == 1)
                if practical is not None:
                    slips.append(float(practical.evaluate([work])[0].slip[pair[player] - 1]))
            move = sample_distribution(distribution, decision_rng(seed, seat, ply))
            outcome = advisor.spec.step(work, {seat: move})
        return dict(error_share=float(np.mean(shares)) if shares else None,
                    predicted_slip=float(np.mean(slips)) if slips else None,
                    proxy_error_probability=float(np.mean(errors)) if errors else None,
                    only_moves_4_plies=only, own_decisions_4_plies=len(shares))
    finally:
        work.close()


def match_head_levels(reference_levels, training_levels):
    """Match by nearest centre within 100 Elo, requiring every training style."""
    matches = []
    for reference in reference_levels["levels"]:
        eligible = [row for row in training_levels["levels"]
                    if abs(row["elo_center"] - reference["elo_center"]) <= 100
                    and all(row["styles"].get(style, {}).get("available", False)
                            for style in training_levels["train_styles"])]
        closest = min(eligible, key=lambda row: (abs(row["elo_center"] - reference["elo_center"]),
                                                row["level"]), default=None)
        matches.append(dict(level=reference["level"], elo_center=reference["elo_center"],
            head_level=closest["level"] if closest else None,
            head_elo_center=closest["elo_center"] if closest else None,
            note="nearest covered head within 100 Elo; ties use lower index" if closest else
                 "missing: no head within 100 Elo with all training styles available"))
    return matches


class _MappedHead:
    """Translate scalar head lookups; missing indices must never be substituted."""

    def __init__(self, values, indices):
        self.values, self.indices = values, indices

    def __getitem__(self, key):
        indices = tuple(self.indices[index] for index in (key if isinstance(key, tuple) else (key,)))
        if None in indices:
            raise ValueError("comparison requested an unmatched learned level")
        return self.values[indices if isinstance(key, tuple) else indices[0]]


class _AlignedHeads:
    def __init__(self, evaluator, matches):
        self.evaluator = evaluator
        self.indices = [row["head_level"] - 1 if row["head_level"] is not None else None for row in matches]

    def evaluate(self, games, cancel=None):
        return [replace(row, practical=_MappedHead(row.practical, self.indices),
                        slip=_MappedHead(row.slip, self.indices))
                for row in self.evaluator.evaluate(games, cancel)]


def run_comparison(checkpoint, *, output, b_checkpoint=None, c_checkpoint=None,
                   device="cpu", backend="cpp", seed=100000, positions=8, rollouts=8,
                   max_plies=300, smoke=False, level=None, opponent_level=None,
                   guard=0.3, reference_simulations=800):
    """E8: shared positions, guard engine, proxies and per-seat/ply random streams."""
    if (min(positions, rollouts, max_plies, reference_simulations) < 1
            or not math.isfinite(guard) or guard < 0):
        raise ValueError("invalid comparison budget or guard")
    if reference_simulations != 800 and not smoke:
        raise ValueError("comparison engine reference is MCTS800; smaller budgets are smoke-only")
    started = time.perf_counter()
    reference = create_advisor(checkpoint, device, method="search")
    if getattr(reference, "general", False) or reference.levels_path is None:
        raise ValueError("compare requires calibrated two-player search-game levels")
    models = {"A_depth2": (reference, 2), "A_depth4": (reference, 4)}
    original_models = {}
    head_matches = {}
    for arm, path in (("B", b_checkpoint), ("C", c_checkpoint)):
        if path is None:
            continue
        loaded = create_advisor(path, device, method="learned")
        check_manifest_matches_spec(loaded.manifest, reference.spec, reference.engine.game_params)
        expected_init = "engine" if arm == "B" else "joint"
        if loaded.manifest.algorithm_params.get("init", "engine") != expected_init:
            raise ValueError(f"{arm} checkpoint must have init={expected_init}")
        if arm == "B" and sha256_file(loaded.engine_checkpoint / "weights.pt") != sha256_file(reference.engine_checkpoint / "weights.pt"):
            raise ValueError("B must share the common engine checkpoint used by A")
        original_models[arm] = loaded
        head_matches[arm] = match_head_levels(reference.level_data, loaded.level_data)
        aligned = _AlignedHeads(loaded.practical, head_matches[arm])
        models[arm] = (PracticalAdvisor(reference.engine, aligned, reference.level_data,
                                       checkpoint_root=reference.checkpoint_root), 1)
    for advisor in (reference, *original_models.values()):
        if ("SMOKE" in advisor.level_data.get("notes", "").upper()
                or "SMOKE" in advisor.manifest.notes.upper()) and not smoke:
            raise ValueError("smoke checkpoints/calibration require smoke evaluation")
    arms = ["own", "engine", *models]
    levels = [level] if level is not None else [row.level for row in reference.levels
        if "qre" in row.styles and any(style in row.styles for style in reference.level_data["train_styles"])]
    if not levels:
        raise ValueError("no common calibrated search levels are available")
    for own in levels:
        for value in (own, own if opponent_level is None else opponent_level):
            reference._search_proxy(value)
            reference.proxy_agent(value)
    sources = {"engine": training_compute(reference.engine_checkpoint)}
    sources.update({arm: training_compute(model.checkpoint_dir) for arm, model in original_models.items()})
    arm_sources = {arm: (["engine", "B"] if arm == "B" else ["C"] if arm == "C" else ["engine"])
                   for arm in arms}
    rows = [dict(role="engine", path=str(reference.engine_checkpoint),
                 sha256=sha256_file(reference.engine_checkpoint / "weights.pt"))]
    rows.extend(dict(role=arm, path=model.checkpoint_dir,
                     sha256=sha256_file(Path(model.checkpoint_dir) / "weights.pt"))
                for arm, model in original_models.items())
    buckets, latencies, used_seeds = defaultdict(list), defaultdict(list), {seed}
    results, cases, skipped_styles, missing_arms = [], [], [], []
    def add(metric, value, **labels):
        key = (metric, tuple(sorted(labels.items())))
        # Retain an empty bucket for an unavailable prediction, never fabricate zero.
        if value is not None:
            buckets[key].append(value)
        else:
            buckets[key]
    def choose(arm, game, own, other, choice_seed):
        tick = time.perf_counter()
        fallback = None
        if arm == "own":
            player = mover(reference.spec, game)
            action = sample_distribution(reference.proxy_agent(own).distribution(game),
                                         decision_rng(choice_seed, player, 0, 3))
        elif arm == "engine":
            action = reference_move(reference, game, reference_simulations)
        else:
            model, depth = models[arm]
            advice = model.advise(game, own, other, top_k=1, guard=guard, depth=depth)[0]
            action, fallback = advice.action, float(advice.guard_fallback)
        latencies[arm].append((time.perf_counter() - tick) * 1000)
        return action, fallback
    for i in levels:
        j = i if opponent_level is None else opponent_level
        pair = (i, j)
        active_arms = []
        for arm in arms:
            missing = [x for x in dict.fromkeys(pair) if arm in head_matches
                       and head_matches[arm][x - 1]["head_level"] is None]
            if missing:
                note = f"missing head match for reference levels {missing}: requires centre within 100 Elo and all training styles"
                missing_arms.append(dict(arm=arm, level=i, opponent_level=j, missing_levels=missing, note=note))
                results.append(dict(metric="missing_level", arm=arm, level=i, opponent_level=j,
                                    value=None, ci_low=None, ci_high=None, n=0, ci_method=note))
            else:
                active_arms.append(arm)
        position_seed = seed + i * 10000 + j * 1000000
        used_seeds.update(position_seed + n for n in range(positions))
        styles = ["mixture", *dict.fromkeys(reference.level_data["train_styles"] + reference.level_data["heldout_styles"])]
        available_styles = []
        for style in styles:
            if style != "mixture" and any(style not in reference.levels[x - 1].styles for x in pair):
                skipped_styles.append(dict(level=i, opponent_level=j, style=style, reason="unavailable calibration"))
            else:
                available_styles.append(style)
        sampled = sampled_positions(reference, pair, position_seed, count=positions, max_plies=max_plies, backend=backend)
        try:
            for sample in sampled:
                game, phase, index = sample.game, sample.phase, sample.index
                player = mover(reference.spec, game)
                own, other = pair[player], pair[1 - player]
                labels = dict(level=own, opponent_level=other, phase=phase)
                seeds = [position_seed + 1000 + index * rollouts + r for r in range(rollouts)]
                used_seeds.update(seeds)
                actions, fallbacks = {}, {}
                for arm in active_arms:
                    actions[arm], fallbacks[arm] = choose(arm, game, own, other, seeds[0])
                q = reference.engine.one_ply(game)
                eligible = len(q) > 2 and abs(reference.engine.evaluate([game])[0].value) <= 0.9
                history = game.history_keys() if hasattr(game, "history_keys") else game.moves
                cases.append(dict(position_id=hashlib.sha256(repr(history).encode()).hexdigest(),
                    position_seed=sample.seed, sample_index=index, levels=list(pair), player=player, phase=phase,
                    rollout_seeds=seeds, actions=actions, guard=guard, styles=available_styles, e1_eligible=eligible))
                for arm in active_arms:
                    add("value_loss", max(q.values()) - q[actions[arm]], arm=arm, **labels)
                    add("guard_fallback_rate", fallbacks[arm], arm=arm, **labels)
                for style in available_styles:
                    if eligible:
                        scores, caps = paired_intervention(reference, game, pair, actions, seeds,
                                                           max_plies=max_plies, style=style)
                        for arm in active_arms:
                            add("score", float(np.mean(scores[arm])), arm=arm, style=style, **labels)
                            add("truncation_rate", float(np.mean(caps[arm])), arm=arm, style=style, **labels)
                            for versus in ("own", "engine"):
                                add("score_difference", float(np.mean(np.asarray(scores[arm]) - scores[versus])),
                                    arm=arm, versus=versus, style=style, **labels)
                    for arm in active_arms:
                        practical = models[arm][0].practical if arm in original_models else None
                        measurements = _compare_difficulty(reference, practical, game, actions[arm], pair, seeds[0], style)
                        for metric, value in measurements.items():
                            add(metric, value, arm=arm, style=style, **labels)
        finally:
            for sample in sampled:
                sample.game.close()
        for style in available_styles:
            for rho in (0.1, 0.3, 1.0):
                for arm in active_arms:
                    scores, caps = [], []
                    for index in range(2 * rollouts):
                        team = index % 2
                        seat_levels = (i, j) if team == 0 else (j, i)
                        match_seed = position_seed + 200000 + index // 2
                        used_seeds.add(match_seed)
                        game = reference.spec.create_game(backend, seed=match_seed)
                        try:
                            def intervention(work, own, other):
                                return choose(arm, work, own, other, match_seed)[0]
                            score, cap = continuation(reference, game, seat_levels, match_seed, max_plies=max_plies,
                                style=style, intervention=None if arm == "own" else intervention,
                                rho=rho, team_player=team)
                            scores.append(score[team])
                            caps.append(float(cap))
                        finally:
                            game.close()
                    results.append(wilson_metric(f"tag_team_score_rho={rho}", scores, arm=arm,
                                                  style=style, level=i, opponent_level=j))
                    results.append(wilson_metric(f"tag_team_truncation_rate_rho={rho}", caps, arm=arm,
                                                  style=style, level=i, opponent_level=j))
    for (metric, labels), values in buckets.items():
        results.append(bootstrap_metric(metric, values, seed=seed, **dict(labels)))
        if metric == "value_loss":
            for q in (0.5, 0.9, 0.95, 1.0):
                results.append(bootstrap_metric(f"value_loss_q{q}", values, seed=seed,
                    statistic=lambda x, q=q: np.quantile(x, q), **dict(labels)))
    for arm in arms:
        for q, name in ((0.5, "median"), (0.95, "p95")):
            results.append(bootstrap_metric(f"advice_latency_{name}", latencies[arm], seed=seed, arm=arm,
                                            statistic=lambda x, q=q: np.quantile(x, q), unit="ms"))
        dependencies = [sources[key] for key in arm_sources[arm]]
        # The same run can be referenced through several checkpoint directories.
        unique = {entry["metrics"] or entry["checkpoint"]: entry for entry in sorted(
            dependencies, key=lambda entry: entry["through_iteration"])}
        seconds = [entry["wall_clock_seconds"] for entry in unique.values()]
        measured = all(value is not None for value in seconds)
        results.append(dict(metric="training_wall_clock_seconds", arm=arm,
            value=sum(seconds) if measured else None, ci_low=None, ci_high=None,
            n=len(seconds) if measured else 0, ci_method="logged phase wall times; no sampling interval", unit="s"))
        hours = [entry["gpu_hours"] for entry in unique.values()]
        measured_gpu = all(value is not None for value in hours)
        results.append(dict(metric="training_gpu_hours", arm=arm, value=sum(hours) if measured_gpu else None,
            ci_low=None, ci_high=None, n=len(hours) if measured_gpu else 0,
            ci_method="logged GPU-device phase wall times, not device utilization", unit="h"))
    # Evaluate C's own policy/value engine separately from the common guard engine.
    if "C" in original_models:
        from hasami.agents import SearchLimits
        from .evaluation import play_matches
        from .inference import MctsAgent
        model = original_models["C"]
        agent = MctsAgent(model.engine.network, model.engine.encoder, model.engine.device,
                          simulations=reference_simulations, rng=np.random.default_rng(seed), spec=reference.spec)
        opponent, limits = reference.spec.baseline_agent("random", seed)
        match = play_matches(agent, opponent, 2 * rollouts, spec=reference.spec, backend=backend,
            max_plies=max_plies, limits=SearchLimits(1, 60000, reference_simulations),
            opponent_limits=limits, seed=seed, swap_sides=True)
        results.append(wilson_metric("own_engine_score", [1.] * match.wins + [0.] * match.losses + [0.5] * match.draws,
                                     arm="C", versus="random"))
        results.append(wilson_metric("own_engine_truncation_rate", [1.] * match.truncated + [0.] * (match.games - match.truncated),
                                     arm="C", versus="random"))
    report = dict(schema_version=1, experiment="compare", status="complete", smoke=smoke,
        created_at=datetime.now(timezone.utc).isoformat(), game_id=reference.spec.game_id,
        device=str(reference.engine.device), checkpoints=rows, levels_sha256=sha256_file(reference.levels_path),
        config=dict(arms=arms, positions=positions, rollouts=rollouts, max_plies=max_plies, backend=backend,
            levels=levels, opponent_level=opponent_level, guard=guard, depths={arm: depth for arm, (_, depth) in models.items()},
            reference_simulations=reference_simulations, train_styles=reference.level_data["train_styles"],
            heldout_styles=reference.level_data["heldout_styles"], common_levels=reference.level_data,
            head_training_levels={arm: model.level_data for arm, model in original_models.items()},
            head_level_matches=head_matches, missing_arms=missing_arms,
            matched_cases=cases, skipped_styles=skipped_styles, training_compute=sources, arm_compute_sources=arm_sources,
            common_random_numbers="seed, player, continuation ply, stream; arm-independent",
            tag_team_rhos=[0.1, 0.3, 1.0], tag_team_seed_offset=200000, tag_team_swap_sides=True,
            truncation_score=0.5, bootstrap_samples=500, own_engine_reference="random; side-swapped MCTS"),
        seeds=sorted(used_seeds), results=results, seconds=time.perf_counter() - started,
        notes=("SMOKE ONLY; not strength or practical-benefit evidence. " if smoke else "") +
            "All arms use the same guard engine, reference levels and continuation proxies. Learned head axes and slip "
            "indices are matched by calibrated centre within 100 Elo with all training styles available. "
            "Unmatched arms are missing for that level pair (null missing_level rows; details in config.missing_arms). "
            "E1 excludes decided/forced positions and bootstraps independent positions with paired rollout means. "
            "E4 measures four-ply continuations; unavailable slip predictions remain null. Caps score half separately. "
            "Compute reports measured phase wall time by device, not accelerator utilization or assumed equal budgets.")
    validate_artifact(report, "eval")
    output = Path(output)
    atomic_write_json(output, report)
    atomic_write_json(output.with_suffix(".done.json"), report)
    return report


def run_experiment(experiment, checkpoint, *, output, device="cpu", backend="cpp", seed=100000,
                   positions=8, rollouts=8, max_plies=300, smoke=False, level=None, opponent_level=None,
                   guard=0.3, depth=None, reference_simulations=800, leave_one_out=None, method="search",
                   b_checkpoint=None, c_checkpoint=None):
    aliases = {**{f"E{i + 1}": name for i, name in enumerate(EXPERIMENTS)}, "E8": "compare"}
    experiment = aliases.get(experiment, experiment)
    if experiment == "compare":
        return run_comparison(checkpoint, output=output, b_checkpoint=b_checkpoint, c_checkpoint=c_checkpoint,
            device=device, backend=backend, seed=seed, positions=positions, rollouts=rollouts, max_plies=max_plies,
            smoke=smoke, level=level, opponent_level=opponent_level, guard=guard,
            reference_simulations=reference_simulations)
    if b_checkpoint is not None or c_checkpoint is not None:
        raise ValueError("B/C checkpoints are only supported by compare")
    if experiment not in EXPERIMENTS or min(positions, rollouts, max_plies, reference_simulations) < 1:
        raise ValueError("unknown experiment or nonpositive experiment budget")
    if reference_simulations != 800 and not smoke:
        raise ValueError("E1 engine reference is MCTS800; smaller budgets are smoke-only")
    if method not in ("search", "learned"):
        raise ValueError("evaluation method must be search or learned")
    if method == "search" and experiment not in ("single_intervention", "tag_team", "difficulty", "cost"):
        raise ValueError("search evaluation supports E1/E3/E4/E7; use method=learned for E2/E5/E6")
    started = time.perf_counter()
    advisor = create_advisor(checkpoint, device, method=method)
    if getattr(advisor, "general", False):
        return _general_experiment(advisor, experiment, output=output, backend=backend, seed=seed,
            positions=positions, rollouts=rollouts, max_plies=max_plies, level=level,
            opponent_level=opponent_level, guard=guard, depth=1 if depth is None else depth)
    depth = (4 if method == "search" else 1) if depth is None else depth
    advice_times = []
    advisor_arm = "search" if method == "search" else "practical"

    def advise(model, game, own, other):
        start = time.perf_counter()
        result = model.advise(game, own, other, top_k=1, guard=guard, depth=depth)[0]
        advice_times.append((time.perf_counter() - start) * 1000)
        return result
    if "SMOKE" in advisor.level_data.get("notes", "").upper() and not smoke:
        raise ValueError("synthetic/smoke calibration requires smoke evaluation")
    levels = [level] if level is not None else [entry.level for entry in advisor.levels
        if (method != "search" or "qre" in entry.styles) and any(
            style in entry.styles for style in advisor.level_data["train_styles"])]
    if not levels:
        raise ValueError("no levels have the policies required for this evaluation method")
    for i in levels:
        advisor._level(i)
    config = dict(positions=positions, rollouts=rollouts, max_plies=max_plies, backend=backend,
                  levels=levels, opponent_level=opponent_level, guard=guard, depth=depth,
                  reference_simulations=reference_simulations, bootstrap_samples=500,
                  truncation_score=0.5, train_styles=advisor.level_data["train_styles"],
                  heldout_styles=advisor.level_data["heldout_styles"], leave_one_out=leave_one_out or {},
                  method=method, calibration_notes=advisor.level_data.get("notes", ""))
    if method == "search":
        search_levels = sorted(set(levels) | ({opponent_level} if opponent_level is not None else set()))
        config["search_qre_params"] = {str(i): advisor._search_proxy(i).params for i in search_levels}
        config["level_elo"] = {str(row.level): row.elo for row in advisor.levels}
        config["continuation_branching"] = 6
        config["search_prior_prefilter"] = 8
        config["search_preserve_terminal_moves"] = True
        config["difficulty"] = "reach-weighted mean prefiltered-QRE error probability at own nodes; delta=0.2"
    results, used_seeds = [], {seed}
    checkpoint_rows = [dict(role=advisor_arm, path=str(advisor.checkpoint_dir),
                           sha256=sha256_file(Path(advisor.checkpoint_dir) / "weights.pt")),
                       dict(role="engine", path=str(advisor.engine_checkpoint),
                            sha256=sha256_file(advisor.engine_checkpoint / "weights.pt"))]
    models = [("mixture", advisor)]
    for omitted, path in (leave_one_out or {}).items():
        model = create_advisor(path, device, method=method)
        if omitted in model.level_data["train_styles"]:
            raise ValueError("leave-one-style-out checkpoint still trains on omitted style")
        if model.level_data["engine_sha256"] != advisor.level_data["engine_sha256"] or model.level_data["k"] != advisor.level_data["k"]:
            raise ValueError("transfer models need matching engines and calibrated levels")
        models.append((f"without_{omitted}", model))
        checkpoint_rows.append(dict(role=f"without_{omitted}", path=str(model.checkpoint_dir),
                                    sha256=sha256_file(Path(model.checkpoint_dir) / "weights.pt")))
    if experiment == "robustness" and not leave_one_out:
        raise ValueError("E2 requires leave-one-style-out checkpoints for the transfer matrix")
    buckets = defaultdict(list)
    def add(metric, value, **labels):
        buckets[(metric, tuple(sorted(labels.items())))].append(value)
    pairs = ([(i, j) for i in levels for j in levels] if experiment == "calibration" and opponent_level is None
             else [(i, i if opponent_level is None else opponent_level) for i in levels])
    for i, j in pairs:
        advisor._level(j)
        pair = (i, j)
        position_seed = seed + i * 10000 + j * 1000000
        used_seeds.update(position_seed + n for n in range(positions))
        sampled = sampled_positions(advisor, pair, position_seed, count=positions, max_plies=max_plies, backend=backend)
        try:
            for sample in sampled:
                game, phase, index = sample.game, sample.phase, sample.index
                player = mover(advisor.spec, game)
                own, other = pair[player], pair[1 - player]
                labels = dict(level=own, opponent_level=other, phase=phase)
                rollout_seeds = [position_seed + 1000 + index * rollouts + r for r in range(rollouts)]
                used_seeds.update(rollout_seeds)
                if experiment in ("single_intervention", "robustness"):
                    if len(advisor.spec.legal_actions(game, player)) <= 2 or abs(advisor.engine.evaluate([game])[0].value) > 0.9:
                        continue
                    for train_style, model in models if experiment == "robustness" else models[:1]:
                        styles = (list(dict.fromkeys([*advisor.level_data["train_styles"], *advisor.level_data["heldout_styles"]]))
                                  if experiment == "robustness" else ["mixture"])
                        for style in styles:
                            if style != "mixture" and any(style not in advisor.levels[x - 1].styles for x in pair):
                                continue
                            actions = dict(own=sample_distribution(advisor.proxy_agent(own).distribution(game),
                                                                   decision_rng(rollout_seeds[0], player, 0, 3)),
                                           engine=reference_move(advisor, game, reference_simulations))
                            actions[advisor_arm] = advise(model, game, own, other).action
                            samples, caps = paired_intervention(advisor, game, pair, actions, rollout_seeds,
                                                                 max_plies=max_plies, style=style)
                            row_labels = dict(labels, style=style, train_style=train_style)
                            # Resample independent starting positions, keeping their paired rollout means together.
                            for versus in ("engine", "own"):
                                difference = np.asarray(samples[advisor_arm]) - samples[versus]
                                add("score_difference", float(np.mean(difference)), arm=advisor_arm, versus=versus, **row_labels)
                            for arm in samples:
                                add("score", float(np.mean(samples[arm])), arm=arm, **row_labels)
                                add("truncation_rate", float(np.mean(caps[arm])), arm=arm, **row_labels)
                elif experiment == "difficulty":
                    actions = {advisor_arm: advise(advisor, game, own, other).action,
                               "engine": reference_move(advisor, game, reference_simulations)}
                    for arm, action in actions.items():
                        work = game.clone()
                        try:
                            outcome = advisor.spec.step(work, {player: action})
                            only_moves, shares, slips = 0, [], []
                            for ply in range(4):
                                if outcome.terminal or outcome.truncated:
                                    break
                                seat = mover(advisor.spec, work)
                                q = advisor.engine.one_ply(work)
                                errors = [max(q.values()) - value > 0.2 for value in q.values()]
                                if seat == player:
                                    shares.append(float(np.mean(errors)))
                                    only_moves += int(sum(not error for error in errors) == 1)
                                    if method == "search":
                                        probabilities = advisor.proxy_agent(own, style="qre").distribution(work)
                                        slips.append(advisor.error_probability(q, probabilities))
                                    else:
                                        slips.append(float(advisor.practical.evaluate([work])[0].slip[own - 1]))
                                distribution = advisor.proxy_agent(pair[seat]).distribution(work)
                                move = sample_distribution(distribution, decision_rng(rollout_seeds[0], seat, ply))
                                outcome = advisor.spec.step(work, {seat: move})
                            if shares:
                                add("error_share", float(np.mean(shares)), arm=arm, **labels)
                                add("predicted_slip", float(np.mean(slips)), arm=arm, **labels)
                            add("only_moves_4_plies", only_moves, arm=arm, **labels)
                        finally:
                            work.close()
                elif experiment == "calibration":
                    prediction = advisor.practical.evaluate([game])[0]
                    for rollout_seed in rollout_seeds:
                        scores, cap = continuation(advisor, game, pair, rollout_seed, max_plies=max_plies)
                        add("practical_observation", (float(prediction.practical[own - 1, other - 1]), scores[player]),
                            level=own, opponent_level=other)
                        agent = _agents(advisor, pair, rollout_seed, "mixture")[player]
                        action = sample_distribution(agent.distribution(game), decision_rng(rollout_seed, player, 0))
                        q = advisor.engine.one_ply(game)
                        add("slip_observation", (float(prediction.slip[own - 1]), float(max(q.values()) - q[action] > 0.2)), level=own)
                        add("truncation_rate", float(cap), **labels)
                elif experiment == "levels":
                    prediction = advisor.practical.evaluate([game])[0].practical
                    for lower, upper in zip(levels, levels[1:]):
                        add("ps_own_monotone", float(prediction[upper - 1, j - 1] >= prediction[lower - 1, j - 1]), level=lower, opponent_level=j)
                        add("ps_opponent_monotone", float(prediction[i - 1, upper - 1] <= prediction[i - 1, lower - 1]), level=i, opponent_level=lower)
                        a = advisor.advise(game, lower, j, top_k=1, guard=guard)[0]
                        b = advisor.advise(game, upper, j, top_k=1, guard=guard)[0]
                        add("chosen_engine_value_monotone", float(b.engine_value >= a.engine_value), level=lower, opponent_level=j)
                        add("chosen_practical_score_monotone", float(b.practical_score >= a.practical_score), level=lower, opponent_level=j)
                elif experiment == "cost":
                    advice = advise(advisor, game, own, other)
                    add("value_loss", advice.value_loss, **labels)
                    add("guard_fallback_rate", float(advice.guard_fallback), **labels)
        finally:
            for sample in sampled:
                sample.game.close()
        if experiment == "tag_team":
            for rho in (0.1, 0.3, 1.0):
                for arm in (advisor_arm, "engine"):
                    scores, caps = [], []
                    for index in range(2 * rollouts):
                        team = index % 2
                        seat_levels = [i, j] if team == 0 else [j, i]
                        match_seed = position_seed + index // 2
                        used_seeds.add(match_seed)
                        game = advisor.spec.create_game(backend, seed=match_seed)
                        try:
                            def intervention(work, own, other):
                                return (advise(advisor, work, own, other).action
                                        if arm == advisor_arm else reference_move(advisor, work, reference_simulations))
                            score, cap = continuation(advisor, game, seat_levels, match_seed, max_plies=max_plies,
                                                      intervention=intervention, rho=rho, team_player=team)
                            scores.append(score[team])
                            caps.append(float(cap))
                        finally:
                            game.close()
                    results.append(wilson_metric("score", scores, arm=f"{arm}:rho={rho}", level=i, opponent_level=j))
                    results.append(wilson_metric("truncation_rate", caps, arm=f"{arm}:rho={rho}", level=i, opponent_level=j))
    for (metric, labels_tuple), values in buckets.items():
        labels = dict(labels_tuple)
        if metric.endswith("_observation"):
            results.extend(_calibration_metrics(metric.replace("observation", ""), values, seed, **labels))
        else:
            results.append(bootstrap_metric(metric, values, seed=seed, **labels))
            if metric == "value_loss":
                for quantile in (0.5, 0.9, 0.95, 1.0):
                    results.append(bootstrap_metric(f"value_loss_q{quantile}", values, seed=seed,
                                                     statistic=lambda x, q=quantile: np.quantile(x, q), **labels))
    if experiment == "levels":
        for row in advisor.level_data["levels"]:
            for style, entry in row["styles"].items():
                results.append(dict(metric="proxy_elo", level=row["level"], style=style, value=entry["elo"],
                                    ci_low=entry["ci"][0] if entry["ci"] else None,
                                    ci_high=entry["ci"][1] if entry["ci"] else None,
                                    n=entry["games"], ci_method="calibration artifact 95%"))
    for quantile, name in ((0.5, "median"), (0.95, "p95")):
        results.append(bootstrap_metric(f"advice_latency_{name}", advice_times, seed=seed, arm=advisor_arm,
                                        statistic=lambda x, q=quantile: np.quantile(x, q), unit="ms"))
    report = dict(schema_version=1, experiment=experiment, status="complete", smoke=smoke,
                  created_at=datetime.now(timezone.utc).isoformat(), game_id=advisor.spec.game_id,
                  device=str(advisor.engine.device), checkpoints=checkpoint_rows,
                  config=config, seeds=sorted(used_seeds), results=results, seconds=time.perf_counter() - started,
                  notes=("SMOKE ONLY; not evidence of practical benefit. " if smoke else "") +
                  "Caps score half and are reported separately. E1/E2 bootstrap independent positions with paired "
                  "rollout means. E4 counts own-turn only-moves along sampled four-ply continuations. "
                  "E6 chosen-move monotonicity means nondecreasing engine/practical value, not action id.")
    if advisor.levels_path is not None:
        report["levels_sha256"] = sha256_file(advisor.levels_path)
    validate_artifact(report, "eval")
    output = Path(output)
    atomic_write_json(output, report)
    atomic_write_json(output.with_suffix(".done.json"), report)
    return report


def _general_actions(advisor, game, levels, seed, ply, *, forced=None, trace=None):
    actions = {}
    for player in advisor.spec.actors(game):
        if trace is not None:
            trace.append((seed, player, ply))
        if forced is not None and player == forced[0]:
            actions[player] = forced[1]
        else:
            distribution = advisor.proxy_agent(levels[player]).distribution(game, player=player)
            actions[player] = sample_distribution(distribution, decision_rng(seed, player, ply))
    return actions


def _general_continuation(advisor, game, levels, seed, *, max_plies, trace=None, forced=None):
    from .practical_general import outcome_score
    spec = advisor.spec
    if len(levels) != spec.num_players:
        raise ValueError("one continuation level per seat is required")
    for level in levels:
        advisor._level(level)
    work = game.clone()
    ended = None
    try:
        for ply in range(max_plies):
            if work.outcome.terminal:
                break
            outcome = spec.step(work, _general_actions(advisor, work, levels, seed, ply,
                forced=forced if ply == 0 else None, trace=trace))
            if outcome.terminal or outcome.truncated:
                ended = outcome
                break
        capped = (ended.truncated if ended is not None else not work.outcome.terminal)
        if capped:
            return [0.5] * spec.num_players, True
        values = ended.rewards if ended is not None and ended.rewards is not None else terminal_rewards(spec, work)
        return [outcome_score(spec, values, p) for p in range(spec.num_players)], False
    finally:
        work.close()


def _general_intervention(advisor, game, levels, actions, seeds, *, max_plies, traces, player):
    from .practical_general import active_player
    player = active_player(advisor.spec, game, player)
    samples, caps = {}, {}
    for arm, action in actions.items():
        if action not in advisor.spec.legal_actions(game, player):
            raise ValueError("intervention must be legal for the advised seat")
        samples[arm], caps[arm] = [], []
        for seed in seeds:
            trace = [] if traces is not None else None
            scores, capped = _general_continuation(advisor, game, levels, seed, max_plies=max_plies,
                                                   trace=trace, forced=(player, action))
            samples[arm].append(scores[player])
            caps[arm].append(float(capped))
            if traces is not None:
                traces[(arm, seed)] = trace
    return samples, caps


def _general_positions(advisor, levels, seed, count, max_plies, backend, own_level):
    """Yield owned positions, rotating the preferred observer over active seats."""
    spec = advisor.spec
    for index in range(count):
        game = spec.create_game(backend, seed=seed + index)
        try:
            player = index % spec.num_players
            seat_levels = list(levels)
            seat_levels[player] = own_level
            horizon = 0 if index == 0 else int(decision_rng(seed, index, 0).integers(min(max_plies, 60)))
            for ply in range(max_plies):
                if game.outcome.terminal:
                    break
                actors = spec.actors(game)
                if ply >= horizon and player in actors:
                    yield game, player, "early" if ply < 20 else "middle" if ply < 60 else "late"
                    break
                outcome = spec.step(game, _general_actions(advisor, game, seat_levels, seed + index, ply))
                if outcome.terminal or outcome.truncated:
                    break
        finally:
            game.close()


def _general_experiment(advisor, experiment, *, output, backend, seed, positions, rollouts,
                        max_plies, level, opponent_level, guard, depth):
    """E1/E4/E7 general-game proof; all reports are explicitly uncalibrated smoke."""
    if experiment not in ("single_intervention", "difficulty", "cost"):
        raise ValueError("general-game evaluation supports E1/E4/E7 only")
    started = time.perf_counter()
    levels = [level] if level is not None else [row.level for row in advisor.levels]
    buckets, times, seeds = defaultdict(list), [], {seed}
    def add(metric, value, **labels):
        buckets[(metric, tuple(sorted(labels.items())))].append(value)
    for own in levels:
        other = own if opponent_level is None else opponent_level
        advisor._level(own)
        advisor._level(other)
        position_seed = seed + own * 10000 + other * 1000000
        seats = [other] * advisor.spec.num_players
        seeds.update(position_seed + n for n in range(positions))
        sampled = _general_positions(advisor, seats, position_seed, positions, max_plies, backend, own)
        try:
            for index, (game, player, phase) in enumerate(sampled):
                labels = dict(level=own, opponent_level=other, phase=phase)
                seat_levels = [other] * advisor.spec.num_players
                seat_levels[player] = own
                if experiment == "single_intervention" and (
                        len(advisor.spec.legal_actions(game, player)) <= 2 or
                        abs(advisor.engine.evaluate([game], players=[player])[0].value) > 0.9):
                    continue
                advice_seed = position_seed + 1000 + index * rollouts
                seeds.add(advice_seed)
                tick = time.perf_counter()
                advice = advisor.advise(game, own, other, player=player, seed=advice_seed,
                                        guard=guard, depth=depth, top_k=1)[0]
                times.append((time.perf_counter() - tick) * 1000)
                if experiment == "cost":
                    add("value_loss", advice.value_loss, **labels)
                    add("guard_fallback_rate", float(advice.guard_fallback), **labels)
                    continue
                engine = advisor.engine_best(game, player=player, level=own, opponent_level=other,
                                             seed=advice_seed)
                actions = dict(search=advice.action, engine=engine.action)
                if experiment == "single_intervention":
                    actions["own"] = sample_distribution(
                        advisor.proxy_agent(own).distribution(game, player=player),
                        decision_rng(advice_seed, player, 0, 3))
                    rollout_seeds = [advice_seed + r for r in range(rollouts)]
                    seeds.update(rollout_seeds)
                    samples, caps = paired_intervention(advisor, game, seat_levels, actions, rollout_seeds,
                                                        max_plies=max_plies, player=player)
                    for versus in ("engine", "own"):
                        add("score_difference", float(np.mean(np.asarray(samples["search"]) - samples[versus])),
                            arm="search", versus=versus, **labels)
                    for arm in samples:
                        add("score", float(np.mean(samples[arm])), arm=arm, **labels)
                        add("truncation_rate", float(np.mean(caps[arm])), arm=arm, **labels)
                else:
                    for arm, action in actions.items():
                        shares, slips, only = [], [], 0
                        work = game.clone()
                        try:
                            outcome = advisor.spec.step(work, _general_actions(advisor, work, seat_levels,
                                advice_seed, 0, forced=(player, action)))
                            for ply in range(4):
                                if outcome.terminal or outcome.truncated:
                                    break
                                if player in advisor.spec.actors(work):
                                    slip, share, unique = advisor.decision_difficulty(work, player, own, other,
                                        decision_rng(advice_seed, player, ply, 4))
                                    slips.append(slip)
                                    shares.append(share)
                                    only += unique
                                outcome = advisor.spec.step(work, _general_actions(
                                    advisor, work, seat_levels, advice_seed, ply + 1))
                            if shares:
                                add("error_share", float(np.mean(shares)), arm=arm, **labels)
                                add("predicted_slip", float(np.mean(slips)), arm=arm, **labels)
                            add("only_moves_4_plies", only, arm=arm, **labels)
                            add("own_decisions_4_plies", len(shares), arm=arm, **labels)
                        finally:
                            work.close()
        finally:
            sampled.close()
    results = []
    for (metric, labels), values in buckets.items():
        results.append(bootstrap_metric(metric, values, seed=seed, **dict(labels)))
        if metric == "value_loss":
            for q in (0.5, 0.9, 0.95, 1.0):
                results.append(bootstrap_metric(f"value_loss_q{q}", values, seed=seed,
                    statistic=lambda x, q=q: np.quantile(x, q), **dict(labels)))
    for q, name in ((0.5, "median"), (0.95, "p95")):
        results.append(bootstrap_metric(f"advice_latency_{name}", times, seed=seed, arm="search",
                                        statistic=lambda x, q=q: np.quantile(x, q), unit="ms"))
    report = dict(schema_version=1, experiment=experiment, status="complete", smoke=True,
        created_at=datetime.now(timezone.utc).isoformat(), game_id=advisor.spec.game_id,
        device=str(advisor.engine.device),
        checkpoints=[dict(role="engine", path=str(advisor.engine_checkpoint),
                          sha256=sha256_file(Path(advisor.engine_checkpoint) / "weights.pt"))],
        config=dict(positions=positions, rollouts=rollouts, max_plies=max_plies, backend=backend,
            levels=levels, opponent_level=opponent_level, guard=guard, depth=depth,
            determinizations=advisor.determinizations, search_rollouts=advisor.rollouts,
            K=advisor.determinizations, M=advisor.rollouts, D=depth, seed=seed, method="search",
            reference="sampled one-step engine values; no MCTS reference for general games",
            truncation_score=0.5, bootstrap_samples=500, level_elo={str(i): None for i in levels},
            calibration_notes=advisor.level_data["notes"]), seeds=sorted(seeds), results=results,
        seconds=time.perf_counter() - started,
        notes="SMOKE ONLY; no evidence of practical benefit or human strength. Temperature continuations use each "
              "actor's own observation; all non-advised seats use opponent_level. E1 pairs rollout seeds per "
              "seat/step and bootstraps independent starting positions. E4 measures own decisions in four joint "
              "steps; absent own decisions are reported, not treated as evidence of low difficulty. "
              "Caps score half and are separate from terminal outcomes.")
    validate_artifact(report, "eval")
    output = Path(output)
    atomic_write_json(output, report)
    atomic_write_json(output.with_suffix(".done.json"), report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=(*EXPERIMENTS, "compare",
                                               *(f"E{i}" for i in range(1, len(EXPERIMENTS) + 2))))
    parser.add_argument("--checkpoint", required=True, help="common engine/advisor checkpoint")
    parser.add_argument("--b-checkpoint", help="optional method B practical checkpoint")
    parser.add_argument("--c-checkpoint", help="optional method C joint checkpoint")
    parser.add_argument("--output", "--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--method", choices=("search", "learned"), default="search")
    parser.add_argument("--backend", default="cpp")
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--positions", type=int, default=8)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--level", type=int)
    parser.add_argument("--opponent-level", type=int)
    parser.add_argument("--guard", type=float, default=0.3)
    parser.add_argument("--depth", type=int, help="continuation depth (default: search 4, learned 1)")
    parser.add_argument("--reference-simulations", type=int, default=800)
    parser.add_argument("--leave-one-out", action="append", default=[], metavar="STYLE=CHECKPOINT")
    parser.add_argument("--smoke", action="store_true")
    args = vars(parser.parse_args(argv))
    args["leave_one_out"] = dict(item.split("=", 1) for item in args["leave_one_out"])
    run_experiment(**args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
