"""Side-swapped round robin with Wilson score intervals and Bradley-Terry Elo.

Run ``python -m hasami_learn.ladder --entry random --entry model=CHECKPOINT``.
Finished pairs are saved atomically and can be reused with ``--resume``.
Fixed seeds reproduce scores when searches complete their depth/simulation budget;
classical wall-clock cutoffs and nondeterministic device kernels can still affect moves.
Learned MCTS uses only its configured simulation count during evaluation.
Elapsed seconds are measurements and are not expected to reproduce.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from itertools import combinations, permutations

import numpy as np
from pathlib import Path
from typing import Any

from hasami.agents import Agent, SearchLimits
from .games import get_spec, default_rules
from .evaluation import MatchResult, play_matches


def fit_elo(
    names: list[str],
    pairs: list[tuple[str, str, MatchResult]],
    anchor: str | None = None,
) -> dict[str, float]:
    """Fit Bradley-Terry strengths by iterative minorization-maximization.

    Add half a point for each side of each played pair to keep undefeated or
    winless entries finite. Draws count half, as in MatchResult. No evidence
    gives equal ratings. The anchor sets the additive Elo origin only.
    """
    if not names or len(set(names)) != len(names):
        raise ValueError("Elo names must be nonempty and unique")
    anchor = anchor if anchor is not None else ("random" if "random" in names else names[0])
    if anchor not in names:
        raise ValueError("Elo anchor must be an entry")
    index = {name: i for i, name in enumerate(names)}
    scores = [0.0] * len(names)
    edges = []
    for first, second, result in pairs:
        if first == second or first not in index or second not in index:
            raise ValueError("Invalid Elo pair")
        if (any(type(value) is not int or value < 0
                for value in (result.games, result.wins, result.losses, result.draws))
                or result.games != result.wins + result.losses + result.draws):
            raise ValueError("Invalid Elo result totals")
        if not result.games:
            continue
        i, j = index[first], index[second]
        scores[i] += result.wins + 0.5 * result.draws + 0.5
        scores[j] += result.losses + 0.5 * result.draws + 0.5
        edges.append((i, j, result.games + 1.0))
    strengths = [1.0] * len(names)
    for _ in range(10000):
        denominators = [0.0] * len(names)
        for i, j, games in edges:
            weight = games / (strengths[i] + strengths[j])
            denominators[i] += weight
            denominators[j] += weight
        updated = [scores[i] / d if d else 1.0 for i, d in enumerate(denominators)]
        scale = math.exp(sum(math.log(value) for value in updated) / len(updated))
        updated = [value / scale for value in updated]
        change = max(abs(math.log(a / b)) for a, b in zip(updated, strengths))
        strengths = updated
        if change < 1e-10:
            break
    origin = strengths[index[anchor]]
    return {name: 400.0 * math.log10(strengths[i] / origin) for i, name in enumerate(names)}



def fit_plackett_luce(names, tables, anchor=None):
    """Fit partial Plackett-Luce rankings with a weak symmetric prior.

    Each table supplies entry names and final points. Tied rank likelihoods
    sum over all internal orders, using their posterior weights in each MM
    update. Fully tied and truncated tables provide no ranking evidence.
    """
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("Rating names must be nonempty and unique")
    anchor = anchor or ("random" if "random" in names else names[0])
    if anchor not in names:
        raise ValueError("Rating anchor must be an entry")
    index = {name: i for i, name in enumerate(names)}
    choices = []
    for table in tables:
        entries, values = table["entries"], table["values"]
        if (len(entries) < 2 or len(set(entries)) != len(entries) or
                any(name not in index for name in entries) or len(entries) != len(values) or
                any(type(v) not in (int, float) or not math.isfinite(v) or not -1 <= v <= 1 for v in values)):
            raise ValueError("Invalid ranking table")
        if table.get("truncated", False):
            continue
        remaining = {index[name]: value for name, value in zip(entries, values)}
        while len(remaining) > 1:
            best = max(remaining.values())
            group = [i for i, value in remaining.items() if value == best]
            if len(group) < len(remaining):
                choices.append((group, list(remaining), 1.0))
            for i in group:
                del remaining[i]
    # Symmetric pseudo-results keep undefeated and disconnected entries finite.
    for i, j in combinations(range(len(names)), 2):
        choices.extend([([i], [i, j], 0.5), ([j], [i, j], 0.5)])
    strengths = np.ones(len(names), dtype=np.float64)
    wins = np.zeros(len(names), dtype=np.float64)
    for group, _, weight in choices:
        wins[group] += weight
    for _ in range(10000):
        denominator = np.zeros(len(names), dtype=np.float64)
        for group, candidates, weight in choices:
            paths = []
            for order in permutations(group):
                remaining = list(candidates)
                probability = 1.0
                terms = []
                for winner in order:
                    total = strengths[remaining].sum()
                    probability *= strengths[winner] / total
                    terms.append((list(remaining), total))
                    remaining.remove(winner)
                paths.append((probability, terms))
            normalizer = sum(probability for probability, _ in paths)
            for probability, terms in paths:
                for remaining, total in terms:
                    denominator[remaining] += weight * probability / normalizer / total
        updated = wins / denominator
        updated /= np.exp(np.log(updated).mean())
        change = np.max(np.abs(np.log(updated / strengths)))
        strengths = updated
        if change < 1e-10:
            break
    return {name: float(400 * np.log10(strengths[i] / strengths[index[anchor]]))
            for i, name in enumerate(names)}


def _table_report(config, tables):
    names = [entry["name"] for entry in config["entries"]]
    anchor = next((entry["name"] for entry in config["entries"] if entry["spec"] == "random"), names[0])
    rankings = [dict(game, entries=table["entries"]) for table in tables for game in table["games"]]
    ratings = fit_plackett_luce(names, rankings, anchor)
    standings = []
    for name in sorted(names, key=lambda n: -ratings[n]):
        played = [game for game in rankings if name in game["entries"]]
        completed = [game for game in played if not game["truncated"]]
        points = [game["values"][game["entries"].index(name)] for game in completed]
        ranks = [1 + sum(v > point for v in game["values"]) for game, point in zip(completed, points)]
        standings.append(dict(name=name, elo=round(ratings[name], 6), games=len(played),
                              truncated=len(played) - len(completed), placements=dict(Counter(ranks)),
                              mean_points=sum(points) / len(points) if points else 0.0))
    count = math.comb(len(names), config["num_players"])
    return dict(schema_version=1, config=config, anchor=anchor, complete=len(tables) == count,
                elo_method={"name": "Plackett-Luce MM", "ties": "marginalize internal tied orders", "pseudo_points_per_pair": 0.5},
                tables=tables, standings=standings)


def _run_tables(parsed, config, out, resume, spec, game_params):
    scheduled = list(combinations(parsed, spec.num_players))
    if not scheduled:
        raise ValueError("At least num_players unique entries are required")
    saved = {}
    if resume and (out / "ladder.json").exists():
        previous = json.loads((out / "ladder.json").read_text(encoding="utf-8"))
        if previous.get("schema_version") != 1 or dict({"opening_plies": 0, "opening_seed": None}, **previous.get("config", {})) != config:
            raise ValueError("Resume configuration differs from the saved ladder")
        allowed = {tuple(entry["name"] for entry in table) for table in scheduled}
        for table in previous.get("tables", []):
            key = tuple(table["entries"])
            if key not in allowed or key in saved or len(table["games"]) != config["games"]:
                raise ValueError("Invalid or duplicate saved table")
            for i, game in enumerate(table["games"]):
                if (type(game.get("truncated")) is not bool or game.get("rotation") != i % spec.num_players or
                        type(game.get("seed")) is not int or len(game.get("values", [])) != spec.num_players):
                    raise ValueError("Invalid saved table game")
            fit_plackett_luce(list(key), [dict(game, entries=list(key)) for game in table["games"]])
            saved[key] = table
    tables = []
    for entries in scheduled:
        key = tuple(entry["name"] for entry in entries)
        if key in saved:
            tables.append(saved[key])
        else:
            material = json.dumps([config["seed"], entries], sort_keys=True).encode("utf-8")
            table_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
            agents = [_make_agent(entry, table_seed + i, config["time_ms"], config["simulations"],
                                  config["device"], spec, game_params) for i, entry in enumerate(entries)]
            result = play_matches(agents[0][0], [a for a, _ in agents[1:]], config["games"],
                                  spec=spec, backend=config["backend"], max_plies=config["max_plies"],
                                  limits=agents[0][1], opponent_limits=[limit for _, limit in agents[1:]], seed=table_seed,
                                  opening_plies=config["opening_plies"], opening_seed=config["opening_seed"])
            # The report is JSON data: normalize joint-action seat keys before
            # returning it so a saved/resumed table is exactly the same value.
            tables.append(dict(entries=list(key), games=json.loads(json.dumps(result.tables))))
        available = {**saved, **{tuple(table["entries"]): table for table in tables}}
        ordered = [available[tuple(entry["name"] for entry in table)] for table in scheduled
                   if tuple(entry["name"] for entry in table) in available]
        report = _table_report(config, ordered)
        _save(out, report)
    return report

def _parse_entries(entries: list[str], game_spec=None) -> list[dict[str, str]]:
    game_spec = game_spec or get_spec()
    parsed = []
    for entry in entries:
        name, separator, spec = entry.partition("=")
        if not separator:
            spec = name
        name, spec = name.strip(), spec.strip()
        if not name or not spec or any(c in name for c in "\r\n"):
            raise ValueError("Each entry must be a nonempty [name=]baseline or checkpoint")
        item = {"name": name, "spec": spec}
        try:
            game_spec.baseline_agent(spec)
        except (ValueError, KeyError):
            from .manifest import resolve_checkpoint_dir, sha256_file

            directory = resolve_checkpoint_dir(spec)
            item.update(spec=str(directory),
                        manifest_sha256=sha256_file(directory / "manifest.json"),
                        weights_sha256=sha256_file(directory / "weights.pt"))
        parsed.append(item)
    if len(parsed) < 2 or len({entry["name"] for entry in parsed}) != len(parsed):
        raise ValueError("At least two uniquely named entries are required")
    return parsed


def _make_agent(entry: dict[str, str], seed: int, time_ms: int,
                simulations: int | None, device: str, game_spec=None, game_params: dict | None = None) -> tuple[Agent, SearchLimits]:
    game_spec = game_spec or get_spec()
    spec = entry["spec"]
    if "manifest_sha256" not in entry:
        agent, baseline_limits = game_spec.baseline_agent(spec, seed=seed)
        depth = baseline_limits.max_depth if baseline_limits else 1
        # Simulations are a learned-agent budget, not a classical node budget.
        return agent, SearchLimits(depth, time_ms, 200000)
    from .inference import create_agent

    agent = create_agent(spec, device=device, simulations=simulations, seed=seed,
                         rules=default_rules(game_spec), game_id=game_spec.game_id)
    from .manifest import check_manifest_matches_spec

    # load_checkpoint only validated the checkpoint against its own manifest; the ladder selected a game.
    check_manifest_matches_spec(agent.manifest, game_spec, game_params if game_params is not None else {})
    return agent, SearchLimits(1, time_ms, simulations or 200000)


def _restore_result(data: dict[str, Any], games: int, spec=None) -> MatchResult:
    """Validate stored observations; derived statistics are always recomputed."""
    fields = ("games", "wins", "losses", "draws", "truncated", "plies")
    if not isinstance(data, dict) or any(type(data.get(k)) is not int or data[k] < 0 for k in fields):
        raise ValueError("Invalid saved pair result counts")
    if (data["games"] != games or data["wins"] + data["losses"] + data["draws"] != games
            or data["truncated"] > data["draws"]):
        raise ValueError("Invalid saved pair result totals")
    reasons = data.get("reasons")
    if (not isinstance(reasons, dict)
            or any(not isinstance(k, str) or not k or type(v) is not int or v < 0
                   for k, v in reasons.items())
            or sum(reasons.values()) != games
            or reasons.get("max_plies", 0) > data["truncated"]):
        # Truncations are counted separately; a GameSpec v2 adapter may report its own truncation
        # reasons (e.g. "resource_limit"), so only the built-in cap is cross-checked against the count.
        raise ValueError("Invalid saved pair result reasons")
    spec = spec or get_spec()
    if hasattr(spec, "validate_result_reasons"):
        spec.validate_result_reasons(data)
    seconds = data.get("seconds")
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Invalid saved pair result seconds")
    diversity = {}
    if "extra" in data:
        from .autoloop.evidence import effective_sample

        effective_sample(data)
        diversity = {key: data["extra"][key] for key in
                     ("distinct_games", "distinct_openings", "effective_games", "effective_successes")}
    return MatchResult(**{k: data[k] for k in fields}, seconds=seconds, reasons=Counter(reasons), **diversity)


def _report(config: dict[str, Any], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    names = [entry["name"] for entry in config["entries"]]
    anchor = next((entry["name"] for entry in config["entries"] if entry["spec"] == "random"), names[0])
    matches = [(pair["first"], pair["second"], _restore_result(pair["result"], config["games"], get_spec(config.get("game_id", "hasami"), config.get("game_params"))))
               for pair in pairs]
    ratings = fit_elo(names, matches, anchor)
    totals = {name: MatchResult() for name in names}
    legacy = {name: Counter() for name in names}
    for first, second, result in matches:
        if result.effective_games is None:
            # Legacy pairs lack openings and per-seat scores. Match the gate's
            # conservative fallback: at most two seatings at the raw score rate.
            count = min(2, result.games)
            successes = result.win_rate * count
        else:
            successes, count = result.sample
        for name, wins, losses in ((first, result.wins, result.losses),
                                   (second, result.losses, result.wins)):
            total = totals[name]
            total.games += result.games
            total.wins += wins
            total.losses += losses
            total.draws += result.draws
            total.truncated += result.truncated
            total.plies += result.plies
            total.reasons.update(result.reasons)
            total.effective_games = (total.effective_games or 0) + count
            points = successes if name == first else count - successes
            total.effective_successes = (total.effective_successes or 0.0) + points
            if result.effective_games is None:
                legacy[name].update(pairs=1, games=result.games, effective_games=count)
            else:
                total.distinct_games = (total.distinct_games or 0) + result.distinct_games
                total.distinct_openings = (total.distinct_openings or 0) + result.distinct_openings
    standings = []
    for name in sorted(names, key=lambda name: -ratings[name]):
        if legacy[name]:
            # Complete sequence diversity cannot be recovered from old totals.
            totals[name].distinct_games = totals[name].distinct_openings = None
        row = totals[name].to_dict()
        if legacy[name]:
            row["extra"].update({f"legacy_{key}": value for key, value in legacy[name].items()})
        row.pop("seconds")
        row.pop("elo_delta")
        row.update(name=name, elo=round(ratings[name], 6), score_rate=row["win_rate"])
        standings.append(row)
    return {
        "schema_version": 1,
        "config": config,
        "anchor": anchor,
        "complete": len(pairs) == len(names) * (len(names) - 1) // 2,
        "scoring": "wins + 0.5 draws; caps reported separately; Wilson uses opening/seating group means. "
                   "Standings sum samples within opponent pairs; legacy pairs contribute at most two seatings "
                   "at their raw score rate, with unknown distinct counts. Saved legacy pair intervals remain raw.",
        "elo_method": {"name": "Bradley-Terry MM", "pseudo_points_per_side_per_pair": 0.5},
        "pairs": pairs,
        "standings": standings,
    }


def _markdown(report: dict[str, Any]) -> str:
    def label(name: str) -> str:
        return name.replace("\\", "\\\\").replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;")

    if "tables" in report:
        lines = ["# Evaluation ladder", "", "Plackett-Luce ratings; tied ranks are marginalized and caps are excluded.",
                 "", "| Entry | Rating | Games | Caps | Mean points | Placements |",
                 "| --- | ---: | ---: | ---: | ---: | --- |"]
        for row in report["standings"]:
            lines.append(f"| {label(row['name'])} | {row['elo']:.1f} | {row['games']} | "
                         f"{row['truncated']} | {row['mean_points']:.4f} | {row['placements']} |")
        return "\n".join(lines) + "\n"
    lines = ["# Evaluation ladder", "", f"Elo anchor: {label(report['anchor'])} = 0.",
             "Bradley-Terry fit uses 0.5 pseudo-points per side per played pair for finite ratings.",
             "Raw scores use wins + 0.5 draws; Wilson intervals use opening/seating group means. Draws include",
             "max-plies caps; caps are experimental truncations, not terminal rule draws.",
             "Standings credit legacy pairs with at most two seatings at their raw score rate; their saved pair intervals remain raw.",
             "Fixed seeds reproduce scores when searches finish their budgets; time cutoffs may vary.",
             "", "| Entry | Elo | Games | W/L/D | Caps | Score rate | Wilson 95% |",
             "| --- | ---: | ---: | --- | ---: | ---: | --- |"]
    for row in report["standings"]:
        lines.append(f"| {label(row['name'])} | {row['elo']:.1f} | {row['games']} | "
                     f"{row['wins']}/{row['losses']}/{row['draws']} | {row['truncated']} | "
                     f"{row['score_rate']:.1%} | {row['ci_low']:.1%}–{row['ci_high']:.1%} |")
    lines.extend(["", "## Pairs (score from the first entry's perspective)", "",
                  "| First | Second | Games | Score rate | Wilson 95% | End reasons |",
                  "| --- | --- | ---: | ---: | --- | --- |"])
    for pair in report["pairs"]:
        row = pair["result"]
        reasons = ", ".join(f"{key}: {count}" for key, count in sorted(row["reasons"].items()))
        lines.append(f"| {label(pair['first'])} | {label(pair['second'])} | {row['games']} | "
                     f"{row['win_rate']:.1%} | {row['ci_low']:.1%}–{row['ci_high']:.1%} | {reasons} |")
    return "\n".join(lines) + "\n"


def _save(out: Path, report: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for filename, content in (("ladder.json", json.dumps(report, indent=2, allow_nan=False) + "\n"),
                              ("ladder.md", _markdown(report))):
        temporary = out / f".{filename}.tmp"
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, out / filename)


def run_ladder(
    entries: list[str], *, games: int = 100, backend: str = "cpp", max_plies: int | None = None,
    time_ms: int = 1000, simulations: int | None = None, device: str = "auto",
    opening_plies: int = 0, opening_seed: int | None = None,
    seed: int = 0, out: str | Path = "runs/ladder", resume: bool = False, game_id: str = "hasami", game_params=None, spec=None,
) -> dict[str, Any]:
    """Play each unordered pair, alternating sides, and save after every pair.

    An even positive game count provides equal black/white allocation. Resume
    requires identical entries, checkpoint contents, seeds, and match settings.
    Recreate agents per pair so skipped pairs do not advance any random streams.
    """
    spec = spec or get_spec(game_id, game_params)
    max_plies = spec.default_max_plies if max_plies is None else max_plies
    if type(games) is not int or games <= 0 or games % spec.num_players:
        raise ValueError("games must be a positive multiple of num_players for equal seat rotation")
    if type(max_plies) is not int or max_plies < 0:
        raise ValueError("max_plies must be nonnegative; zero disables the cap")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(opening_plies) is not int or opening_plies < 0 or (max_plies and opening_plies > max_plies):
        raise ValueError("opening_plies must be nonnegative and not exceed max_plies")
    if opening_seed is not None and (type(opening_seed) is not int or opening_seed < 0):
        raise ValueError("opening_seed must be a nonnegative integer or None")
    if backend not in ("python", "cpp"):
        raise ValueError("backend must be python or cpp")
    SearchLimits(1, time_ms, simulations if simulations is not None else 200000)
    spec = spec or get_spec(game_id, game_params)
    parsed = _parse_entries(entries, spec)
    config = dict(entries=parsed, games=games, backend=backend, max_plies=max_plies,
                  time_ms=time_ms, simulations=simulations, device=device, seed=seed,
                  opening_plies=opening_plies, opening_seed=opening_seed,
                  swap_sides=True, ruleset_id=spec.ruleset_id, rules=default_rules(spec),
                  game_id=spec.game_id, game_params=game_params or {})
    out = Path(out)
    if spec.num_players > 2:
        config["num_players"] = spec.num_players
        return _run_tables(parsed, config, out, resume, spec, game_params)
    scheduled = list(combinations(parsed, 2))
    saved = {}
    if resume and (out / "ladder.json").exists():
        previous = json.loads((out / "ladder.json").read_text(encoding="utf-8"))
        previous_config = dict(previous.get("config") or {}) if isinstance(previous, dict) else {}
        # Ladders written before the game abstraction carry no game identity: they are Hasami ladders.
        previous_config.setdefault("game_id", "hasami")
        previous_config.setdefault("game_params", {})
        previous_config.setdefault("opening_plies", 0)
        previous_config.setdefault("opening_seed", None)
        if (not isinstance(previous, dict) or previous.get("schema_version") != 1
                or previous_config != config):
            raise ValueError("Resume configuration differs from the saved ladder")
        allowed = {(a["name"], b["name"]) for a, b in scheduled}
        if not isinstance(previous.get("pairs"), list):
            raise ValueError("Invalid saved pairs")
        for pair in previous["pairs"]:
            if not isinstance(pair, dict):
                raise ValueError("Invalid saved pair")
            key = (pair.get("first"), pair.get("second"))
            if any(not isinstance(name, str) for name in key) or key not in allowed or key in saved:
                raise ValueError("Invalid or duplicate saved pair")
            result = _restore_result(pair.get("result"), games, spec)
            saved[key] = dict(first=key[0], second=key[1], result=result.to_dict())
    pairs = []
    for first, second in scheduled:
        key = (first["name"], second["name"])
        if key in saved:
            pairs.append(saved[key])
        else:
            material = json.dumps([seed, first, second], sort_keys=True).encode("utf-8")
            pair_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
            agent, limits = _make_agent(first, pair_seed, time_ms, simulations, device, spec, game_params=game_params)
            opponent, opponent_limits = _make_agent(second, pair_seed + 1, time_ms, simulations, device, spec, game_params=game_params)
            result = play_matches(agent, opponent, games, backend=backend, max_plies=max_plies,
                                  limits=limits, opponent_limits=opponent_limits, swap_sides=True, spec=spec, seed=pair_seed,
                                  opening_plies=opening_plies, opening_seed=opening_seed)
            pairs.append(dict(first=key[0], second=key[1], result=result.to_dict()))
        # Preserve later saved pairs if resuming a report with a gap in its schedule.
        available = {**saved, **{(pair["first"], pair["second"]): pair for pair in pairs}}
        ordered = [available[(a["name"], b["name"])] for a, b in scheduled
                   if (a["name"], b["name"]) in available]
        report = _report(config, ordered)
        _save(out, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", dest="game_id", default="hasami")
    parser.add_argument("--entry", action="append", required=True, help="[name=]checkpoint|random|greedy|alphabeta:N")
    parser.add_argument("--games", type=int, default=100, help="Even game count per pair")
    parser.add_argument("--backend", choices=("cpp", "python"), default="cpp")
    parser.add_argument("--max-plies", type=int)
    parser.add_argument("--time-ms", type=int, default=1000)
    parser.add_argument("--simulations", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opening-plies", type=int, default=0)
    parser.add_argument("--opening-seed", type=int)
    parser.add_argument("--out", type=Path, default=Path("runs/ladder"))
    parser.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args(argv))
    args["entries"] = args.pop("entry")
    try:
        run_ladder(**args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
