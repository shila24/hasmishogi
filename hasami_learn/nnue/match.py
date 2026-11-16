"""Seeded, colour-paired native evaluator matches with effective-sample evidence."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import math
import os
from pathlib import Path
import random
import time

from hasami_learn.autoloop.evidence import effective_sample, utc_now, validate_evaluation, wilson_interval
from shogi import create_game
from shogi.model import Rules
from shogi.native_search import NativeSearch
from shogi.records import save_record
from .data import sync_directory
from .control import MATCH, Stop, atomic_json, identity, lock, read_json, sha256, validate_section


def elo(score):
    """Finite conventional Elo estimate, capped at +/-800 for endpoints."""
    return max(-800., min(800., 400 * math.log10(max(1e-9, score) / max(1e-9, 1-score))))


def summarize(rows):
    if not rows:
        raise ValueError("Match has no games")
    groups = defaultdict(list)
    openings, distinct = set(), set()
    counts = Counter()
    for row in rows:
        score = row["score"]
        if score not in (0., .5, 1.):
            raise ValueError("Invalid match score")
        opening = tuple(row["opening"])
        seat = row["candidate_side"]
        groups[(opening, seat)].append(score)
        openings.add(opening)
        distinct.add((tuple(row["moves"]), seat))
        counts["wins" if score == 1 else "losses" if score == 0 else "draws"] += 1
    successes = sum(sum(values)/len(values) for values in groups.values())
    low, high = wilson_interval(successes, len(groups))
    score = successes/len(groups)
    return dict(games=len(rows), wins=counts["wins"], losses=counts["losses"], draws=counts["draws"],
                truncated=0, score_rate=(counts["wins"]+.5*counts["draws"])/len(rows),
                ci_low=low, ci_high=high,
                extra=dict(sample_unit="opening_actions_and_seating", effective_games=len(groups),
                           effective_successes=successes, distinct_games=len(distinct), distinct_openings=len(openings),
                           effective_score=score, elo_difference=elo(score), elo_low=elo(low), elo_high=elo(high),
                           opening_groups=[dict(opening=list(o), candidate_side=s, games=len(v), score=sum(v)/len(v))
                                           for (o, s), v in sorted(groups.items())]))


def verify_evidence(evidence):
    validate_evaluation(evidence)
    rows = evidence["protocol"]["games_detail"]
    if evidence["results"] != {"opponent": summarize(rows)}:
        raise ValueError("Match evidence aggregates do not match game details")
    settings = evidence["protocol"]["settings"]
    if evidence["seed"] != settings["seed"] or evidence["checkpoint_sha256"] != evidence["protocol"]["candidate_sha256"]:
        raise ValueError("Match evidence identity mismatch")
    if len(rows) != settings["games"]:
        raise ValueError("Incomplete match evidence")
    for i, row in enumerate(rows):
        if len(row["opening"]) != settings["opening_plies"]:
            raise ValueError("Match opening length mismatch")
        winner, reason = row["winner"], row["reason"]
        if winner not in (-1, 0, 1) or reason not in ("checkmate", "no_legal_moves", "declaration", "perpetual_check", "repetition", "max_moves"):
            raise ValueError("Invalid completed match outcome")
        expected = .5 if winner == -1 else float(winner == row["candidate_side"])
        if row["score"] != expected or row["moves"][:len(row["opening"])] != row["opening"]:
            raise ValueError("Match score or opening disagrees with recorded game")
        if not 1 <= len(row["moves"]) <= settings["max_moves"]:
            raise ValueError("Invalid completed match length")
        if row["index"] != i or row["candidate_side"] != i % 2:
            raise ValueError("Missing or duplicate match games")
        if i % 2 and row["opening"] != rows[i-1]["opening"]:
            raise ValueError("Unpaired opening")
    return effective_sample(evidence["results"]["opponent"])


def draw_opening(options, pair):
    """Return a live game after at most 128 deterministic nonterminal draws."""
    seed = f'nnue-match-v1:{options["seed"]}:{pair}'
    for attempt in range(128):
        # Preserve existing first draws so previously completed games still resume.
        rng = random.Random(seed if attempt == 0 else f"{seed}:redraw:{attempt}")
        game = create_game("cpp", Rules(max_moves=options["max_moves"]))
        keep = False
        try:
            for _ in range(options["opening_plies"]):
                if game.outcome.terminal:
                    break
                game.play(rng.choice(game.legal_actions()))
            if not game.outcome.terminal and len(game.moves) == options["opening_plies"]:
                keep = True
                return game
        finally:
            if not keep:
                game.close()
    raise ValueError(f"Could not draw a nonterminal opening after 128 attempts (pair {pair})")


def match(candidate, opponent, out, settings=None, *, stop=None):
    options = {**MATCH, **(settings or {})}
    validate_section(options, "match")
    if options["games"] % 2 or options["opening_plies"] >= options["max_moves"]:
        raise ValueError("Invalid paired match configuration")
    if bool(options["nodes"]) == bool(options["time_ms"]):
        raise ValueError("Choose fixed nodes or fixed time separately")
    directory = Path(out)
    with lock(directory):
        protocol = dict(settings=options, candidate_sha256=identity(candidate), opponent_sha256=identity(opponent))
        path = directory / "protocol.json"
        if path.exists() and read_json(path) != protocol:
            raise ValueError("Match resume protocol/weights mismatch")
        atomic_json(path, protocol)
        completed = directory / "eval.json"
        if completed.exists():
            evidence = read_json(completed)
            verify_evidence(evidence)
            if any(evidence["protocol"][key] != value for key, value in protocol.items()):
                raise ValueError("Completed match protocol mismatch")
            if options["save_records"]:
                for row in evidence["protocol"]["games_detail"]:
                    if sha256(directory / "records" / f'game-{row["index"]:06d}.json') != row["record_sha256"]:
                        raise ValueError("Saved match record hash mismatch")
            return evidence
        started = time.monotonic()
        def worker(worker_id):
            rows = []
            with NativeSearch(nnue=candidate, hash_mb=options["hash_mb"]) as a, NativeSearch(nnue=opponent, hash_mb=options["hash_mb"]) as b:
                for pair in range(worker_id, options["games"]//2, options["threads"]):
                    if stop is not None and stop.requested:
                        break
                    opening_game = draw_opening(options, pair)
                    try:
                        opening = list(opening_game.moves)
                        for seat in (0, 1):
                            index = pair*2 + seat
                            row_path = directory / f"game-{index:06d}.json"
                            if row_path.exists():
                                row = read_json(row_path)
                                if row["opening"] != opening or row["candidate_side"] != seat or row["index"] != index:
                                    raise ValueError("Match resume game mismatch")
                                if options["save_records"] and sha256(directory / "records" / f"game-{index:06d}.json") != row["record_sha256"]:
                                    raise ValueError("Cached saved record hash mismatch")
                                replay = create_game("cpp", Rules(max_moves=options["max_moves"]))
                                try:
                                    for move in row["moves"]:
                                        replay.play(move)
                                    if (not replay.outcome.terminal or replay.outcome.winner != row["winner"]
                                            or replay.outcome.reason != row["reason"]):
                                        raise ValueError("Cached match record outcome mismatch")
                                finally:
                                    replay.close()
                                rows.append(row)
                                continue
                            game = opening_game.clone()
                            try:
                                nodes, seconds = [0, 0], [0., 0.]
                                while not game.outcome.terminal:
                                    player = game.position.turn
                                    search = a if player == seat else b
                                    search.set_position(game)
                                    result = search.go(nodes=options["nodes"] or None, time_ms=options["time_ms"] or None)
                                    if result["move"] not in game.legal_actions():
                                        raise ValueError("Illegal match action")
                                    nodes[int(player != seat)] += result["nodes"]
                                    seconds[int(player != seat)] += result["elapsed_ms"]/1000
                                    game.play(result["move"])
                                score = .5 if game.outcome.winner == -1 else float(game.outcome.winner == seat)
                                row = dict(index=index, opening=opening, candidate_side=seat, score=score,
                                           winner=game.outcome.winner, reason=game.outcome.reason,
                                           moves=game.moves, nodes=nodes, search_seconds=seconds)
                                if options["save_records"]:
                                    records = directory / "records"
                                    records.mkdir(exist_ok=True)
                                    record_path = records / f"game-{index:06d}.json"
                                    save_record(game, record_path)
                                    with record_path.open("rb") as stream:
                                        os.fsync(stream.fileno())
                                    sync_directory(records)
                                    row["record_sha256"] = sha256(record_path)
                                atomic_json(row_path, row)
                                rows.append(row)
                            finally:
                                game.close()
                    finally:
                        opening_game.close()
            return rows
        with ThreadPoolExecutor(max_workers=options["threads"]) as pool:
            rows = sorted((row for group in pool.map(worker, range(options["threads"])) for row in group), key=lambda r:r["index"])
        if len(rows) != options["games"]:
            raise InterruptedError("Match stopped after current opening pairs")
        evidence = dict(schema_version=1, checkpoint=str(Path(candidate).resolve()) if candidate else "bootstrap-v1",
                        checkpoint_sha256=identity(candidate), created_at=utc_now(), device="cpu", game_id="shogi",
                        seed=options["seed"], results={"opponent": summarize(rows)},
                        protocol={**protocol, "games_detail": rows, "swap_sides": True,
                                  "sample_unit": "opening_actions_and_seating", "seconds": time.monotonic()-started,
                                  "budget_kind": "nodes" if options["nodes"] else "time"})
        verify_evidence(evidence)
        atomic_json(directory / "eval.json", evidence)
        return evidence


def main(argv=None):
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate")
    parser.add_argument("--opponent")
    parser.add_argument("--out", required=True)
    parser.add_argument("--settings", default="{}")
    args = parser.parse_args(argv)
    try:
        with Stop() as stop:
            match(args.candidate, args.opponent, args.out, json.loads(args.settings), stop=stop)
    except InterruptedError:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
