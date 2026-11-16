"""Read-only, bounded-memory diagnostics for two-player AlphaZero checkpoints.

Expected targets are derived from fresh engine histories and resolved run settings,
never from trainer, trajectory-target, or datastore reconstruction helpers. Archive
policies retain their float16 storage bytes for exact replay matching. No store is
instantiated: its recovery constructor is a writer. Only the CLI's output is written.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from zipfile import BadZipFile, ZipFile

import numpy as np


CLASSES = ("win_in_one", "forced_defence", "capture")
CHECKS = ("archive", "variant", "history", "outcome", "mover", "planes", "mask",
          "policy", "value_target", "draw_value", "draw_reasons", "truncation_value",
          "replay_value", "mixed_value_target", "value_target_mix", "root_value")
SPEC_SILENCES = [
    "learning-system-spec.md defines mover targets; robust-training-spec.md section 4 defines symmetric draw contempt.",
    "Neither spec fixes annealing boundary arithmetic. This report uses start at iteration 1, (iteration-1)/span before span, end at/after span; schedule checks are conditional on that convention.",
    "Archives do not persist original planes, masks or z. Their reconstruction cannot establish what the learner consumed; only exact replay matches independently check persisted training rows.",
    "Exact duplicate planes/mask/policy bytes can belong to multiple histories/outcomes. Ambiguous replay matches are reported, not treated as unique provenance.",
    "Search mode and value_target_mix come from resolved run settings. Archive root values are float16; mixed replay checks propagate half an adjacent float16 ULP plus 1e-6, while in-memory checks retain 1e-6.",
]


def wilson_interval(successes: int, n: int) -> list[float]:
    """Binomial Wilson 95% interval; empty populations have no measured rate."""
    if type(n) is not int or n < 0 or not math.isfinite(successes) or not 0 <= successes <= n:
        raise ValueError("invalid binomial counts")
    if not n:
        return [0.0, 1.0]
    z2 = 1.96 ** 2
    p = successes / n
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = 1.96 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return [max(0.0, center - half), min(1.0, center + half)]


def rate(successes: int, n: int) -> dict:
    return {"n": n, "successes": successes, "rate": successes / n if n else None,
            "wilson95": wilson_interval(successes, n)}


def random_reference(counts) -> dict:
    """Exact mean conditional success probability, not sampled Bernoulli trials."""
    probabilities = [good / legal for good, legal in counts]
    return {"n": len(probabilities), "rate": float(np.mean(probabilities)) if probabilities else None,
            "wilson95": None,
            "reason": "Exact conditional expectation over legal actions; a binomial interval is inapplicable."}


def distribution(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {"n": len(a), "mean": float(a.mean()) if len(a) else None,
            "quantiles": dict(zip(("q05", "q25", "q50", "q75", "q95"),
                                  np.quantile(a, [.05, .25, .5, .75, .95]).tolist())) if len(a) else None}


@dataclass
class Mean:
    n: int = 0
    total: float = 0.0

    def add(self, value):
        self.n += 1
        self.total += float(value)

    def report(self):
        return {"n": self.n, "mean": self.total / self.n if self.n else None}


@dataclass
class Audit:
    checks: dict = field(default_factory=lambda: {
        name: {"n": 0, "disagreements": 0, "examples": []} for name in CHECKS})
    games: int = 0
    decisions: int = 0
    unavailable: list = field(default_factory=list)

    def check(self, name, expected, found, game_index, ply=None, *, equal=None):
        row = self.checks.setdefault(name, {"n": 0, "disagreements": 0, "examples": []})
        row["n"] += 1
        if equal is None:
            equal = expected == found
        if not equal:
            row["disagreements"] += 1
            if len(row["examples"]) < 5:
                row["examples"].append({"game_index": str(game_index), "ply": ply,
                                        "expected": expected, "found": found})

    def array(self, name, expected, found, game_index, ply):
        a, b = np.asarray(expected), np.asarray(found)
        equal = np.array_equal(a, b)
        detail = {"shape": list(b.shape)}
        want = {"shape": list(a.shape)}
        if a.shape == b.shape and not equal:
            indices = np.argwhere(a != b)[:5]
            want["cells"] = [{"index": ix.tolist(), "value": float(a[tuple(ix)])} for ix in indices]
            detail["cells"] = [{"index": ix.tolist(), "value": float(b[tuple(ix)])} for ix in indices]
        self.check(name, want, detail, game_index, ply, equal=equal)

    def number(self, name, expected, found, game_index, ply=None):
        self.check(name, float(expected), float(found), game_index, ply,
                   equal=bool(np.isfinite(found) and abs(expected - found) <= 1e-6))

    def report(self):
        return {"status": "measured", "games": {"n": self.games}, "decisions": {"n": self.decisions},
                "disagreements": {"n": sum(r["n"] for r in self.checks.values()),
                                  "count": sum(r["disagreements"] for r in self.checks.values())},
                "checks": self.checks, "not_available": self.unavailable,
                "spec_silences": SPEC_SILENCES}


def draw_target(settings: dict, iteration: int) -> float:
    """Explicit analysis convention where the normative spec leaves endpoints open."""
    start = float(settings.get("draw_value_start", -0.2))
    end = float(settings.get("draw_value_end", 0.0))
    span = int(settings.get("draw_value_iterations", 100))
    if span <= 0 or iteration >= span:
        return end
    return start + (end - start) * max(0, iteration - 1) / span


def expected_target(outcome: dict, mover: int, settings: dict, iteration: int, spec) -> float:
    """Derive z from engine outcome and run settings, independent of stored players/rewards."""
    if outcome["truncated"]:
        return float(settings.get("truncation_value", 0.0))
    if outcome["winner"] >= 0:
        return 1.0 if outcome["winner"] == mover else -1.0
    reasons = settings.get("draw_reasons")
    if reasons is None:
        reasons = getattr(spec, "draw_reasons", ())
    if outcome["reason"] in (reasons or ()) and outcome["reason"] in getattr(spec, "draw_reasons", ()):
        return draw_target(settings, iteration)
    return 0.0


def capture_counter(spec, game, mover):
    """Optional generic hook, then the engine's public cumulative capture counter."""
    hook = getattr(spec, "diagnostic_capture_count", None)
    if hook is not None:
        return hook(game, mover)
    counter = getattr(game.position, "captures", None)
    return counter(mover) if callable(counter) else None


@contextmanager
def after_move(spec, game, mover, action):
    if getattr(spec, "supports_undo", False):
        result = spec.step(game, {mover: int(action)})
        try:
            yield game, result
        finally:
            game.undo()
    else:
        child = game.clone()
        try:
            result = spec.step(child, {mover: int(action)})
            yield child, result
        finally:
            child.close()


def label_position(spec, game) -> dict:
    """Exhaustive depth-two outcome labels; only engine transitions decide wins."""
    from .games.base import require_search_game

    require_search_game(spec)
    if game.outcome.terminal:
        raise ValueError("tactical labels require a nonterminal position")
    mover, = spec.actors(game)
    legal = sorted(spec.legal_actions(game, mover))
    wins, safe, captures = [], [], []
    before = capture_counter(spec, game, mover)
    work = game.clone()
    try:
        for action in legal:
            with after_move(spec, work, mover, action) as (child, result):
                if before is not None and capture_counter(spec, child, mover) > before:
                    captures.append(action)
                if result.truncated:
                    # A resource stop is neither an engine win nor an invitation to reply.
                    loses = False
                elif result.terminal or child.outcome.terminal:
                    if child.outcome.winner == mover:
                        wins.append(action)
                    loses = child.outcome.winner == 1 - mover
                else:
                    opponent, = spec.actors(child)
                    loses = False
                    for reply in spec.legal_actions(child, opponent):
                        with after_move(spec, child, opponent, reply) as (leaf, reply_result):
                            if (not reply_result.truncated and leaf.outcome.terminal
                                    and leaf.outcome.winner == 1 - mover):
                                loses = True
                        if loses:
                            break
                if not loses:
                    safe.append(action)
    finally:
        work.close()
    return {"legal": legal, "win_in_one": wins,
            "forced_defence": safe if not wins and 0 < len(safe) < len(legal) else [],
            "forced_loss": bool(legal) and not safe,
            "capture": captures if before is not None else None}


def choose_baseline(spec, game, name, seed=0):
    from hasami.agents import SearchLimits

    agent, _ = spec.baseline_agent(name, seed)
    work = game.clone()
    try:
        decision = agent.choose(work, SearchLimits(1, 3600000, 2**63 - 1))
    finally:
        work.close()
    if not decision.completed or decision.action not in spec.legal_actions(game, spec.actors(game)[0]):
        raise ValueError(f"{name} did not complete a legal depth-one decision")
    return decision.action


def generate_positions(spec, count, seed, backend, rules=None):
    """Reservoir sample seeded random/greedy/mixed full histories, owning every clone."""
    rng = random.Random(seed)
    sample, seen = [], 0
    try:
        for episode in range(max(3, math.ceil(count / 20)) if count else 0):
            game = spec.create_game(backend, rules=rules, seed=seed + episode)
            try:
                for ply in range(min(spec.default_max_plies or 160, 160)):
                    if game.outcome.terminal:
                        break
                    seen += 1
                    slot = seen - 1 if len(sample) < count else rng.randrange(seen)
                    if slot < count:
                        item = (game.clone(), {"generator": ("random", "greedy", "mixed")[episode % 3],
                                               "episode": episode, "ply": ply})
                        if slot < len(sample):
                            sample[slot][0].close()
                            sample[slot] = item
                        else:
                            sample.append(item)
                    mover, = spec.actors(game)
                    use_greedy = episode % 3 == 1 or (episode % 3 == 2 and rng.random() < .5)
                    action = (choose_baseline(spec, game, "greedy", seed + episode) if use_greedy
                              else rng.choice(sorted(spec.legal_actions(game, mover))))
                    result = spec.step(game, {mover: int(action)})
                    if result.terminal or result.truncated:
                        break
            finally:
                game.close()
    except BaseException:
        for game, _ in sample:
            game.close()
        raise
    return sample, seen


def _read_archive(path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key != "metadata"}
        metadata = json.loads(str(data["metadata"].item()))
    return arrays, metadata


def _policy(arrays, index, size):
    if "policy" in arrays:
        return np.asarray(arrays["policy"][index])
    offsets = arrays["policy_offsets"]
    begin, end = map(int, offsets[index:index + 2])
    indices = np.asarray(arrays["policy_index"][begin:end], dtype=np.int64)
    if (indices < 0).any() or (indices >= size).any() or len(set(indices.tolist())) != len(indices):
        raise ValueError("invalid sparse policy indices")
    policy = np.zeros(size, dtype=np.float16)
    policy[indices] = arrays["policy_prob"][begin:end]
    return policy


def _stored_value(meta, player):
    """Interpret archived evidence; this is not the expected-target calculation."""
    if meta["truncated"]:
        return float(meta["truncation_value"])
    if meta["winner"] == -1 and meta["reason"] in meta.get("draw_reasons", ()):
        return float(meta["draw_value"])
    return float(meta["rewards"][player])


def mixed_target(outcome_value, root_value, mix):
    """Independently compute the mixed target and its archive rounding bound."""
    if not math.isfinite(mix) or not 0 <= mix <= 1:
        raise ValueError("value_target_mix must be finite and within [0, 1]")
    if not mix:
        return outcome_value, 1e-6
    if not np.isfinite(root_value) or abs(root_value) > 1:
        raise ValueError("mixed targets require a finite stored root value in [-1, 1]")
    tolerance = 1e-6
    if np.asarray(root_value).dtype == np.dtype("float16"):
        below = float(np.nextafter(root_value, np.float16(-np.inf)))
        above = float(np.nextafter(root_value, np.float16(np.inf)))
        tolerance += mix * max(float(root_value) - below, above - float(root_value)) / 2
    return (1 - mix) * outcome_value + mix * float(root_value), tolerance


def audit_game(spec, encoder, arrays, meta, settings, audit, game_index, backend="python",
               *, initial=None, on_decision=None):
    """Replay twice: establish final truth, then reconstruct every recorded decision."""
    from .games.base import encode_player

    audit.games += 1
    moves = [int(a) for a in arrays["moves"]]
    iteration = int(meta["iteration"])
    mix = float(settings.get("value_target_mix", 0.0))
    cap = settings.get("max_plies", spec.default_max_plies)
    cap = spec.default_max_plies if cap is None else int(cap)
    game = spec.create_game(backend, rules=meta.get("rules"), position=initial, seed=meta.get("seed"))
    try:
        last = None
        for ply, action in enumerate(moves):
            if (game.outcome.terminal or (last and (last.terminal or last.truncated))
                    or (cap and ply >= cap)):
                raise ValueError(f"history continues after termination/truncation/cap at ply {ply}")
            mover, = spec.actors(game)
            joints = meta.get("joint_actions")
            if joints:
                joint = {int(p): int(a) for p, a in joints[ply].items()}
                audit.check("history", {str(mover): action}, {str(p): a for p, a in joint.items()}, game_index, ply)
            last = spec.step(game, {mover: action})
        terminal = game.outcome.terminal or bool(last and last.terminal)
        adapter_truncated = bool(last and last.truncated)
        truncated = adapter_truncated or (not terminal and bool(cap and len(moves) >= cap))
        if not terminal and not truncated:
            raise ValueError("record ends without an engine outcome, adapter stop or configured cap")
        reason = last.info.get("reason", game.outcome.reason) if last else game.outcome.reason
        if truncated and not adapter_truncated:
            reason = "max_plies"
        outcome = {"winner": int(game.outcome.winner), "reason": reason,
                   "truncated": truncated, "plies": len(moves)}
        for key, expected in outcome.items():
            audit.check("outcome", {key: expected}, {key: meta.get(key)}, game_index)
            if key in arrays and np.asarray(arrays[key]).ndim == 0:
                audit.check("outcome", {key: expected}, {key: arrays[key].item()}, game_index)
        rewards = [0.0, 0.0] if truncated or outcome["winner"] == -1 else [
            1.0 if p == outcome["winner"] else -1.0 for p in range(2)]
        audit.array("outcome", rewards, meta["rewards"], game_index, None)
    finally:
        game.close()
    audit.number("draw_value", draw_target(settings, iteration), meta.get("draw_value", 0), game_index)
    configured_reasons = settings.get("draw_reasons")
    reasons = sorted(set(getattr(spec, "draw_reasons", ()) if configured_reasons is None else configured_reasons))
    reasons = [r for r in reasons if r in getattr(spec, "draw_reasons", ())]
    audit.check("draw_reasons", reasons, sorted(set(meta.get("draw_reasons", ()))), game_index)
    audit.number("truncation_value", settings.get("truncation_value", 0), meta.get("truncation_value", 0), game_index)
    players = arrays["players"]
    plies = [int(p) for p in arrays.get("decision_plies", range(len(players)))]
    if len(players) != len(plies) or plies != sorted(set(plies)) or any(p < 0 or p >= len(moves) for p in plies):
        raise ValueError("invalid decision plies")
    decisions = dict(zip(plies, range(len(players))))
    game = spec.create_game(backend, rules=meta.get("rules"), position=initial, seed=meta.get("seed"))
    try:
        for ply, action in enumerate(moves):
            mover, = spec.actors(game)
            if ply in decisions:
                i = decisions[ply]
                audit.decisions += 1
                recorded_mover = int(players[i])
                audit.check("mover", mover, recorded_mover, game_index, ply)
                mask = np.array(encoder.legal_mask(game, mover), dtype=np.float32, copy=True)
                planes = np.array(encode_player(encoder, game, mover), dtype=np.float32, copy=True)
                if "planes" in arrays:
                    audit.array("planes", planes, arrays["planes"][i], game_index, ply)
                if "mask" in arrays:
                    audit.array("mask", mask, arrays["mask"][i], game_index, ply)
                policy = _policy(arrays, i, spec.action_size)
                valid = (policy.shape == mask.shape and np.isfinite(policy).all() and (policy >= 0).all()
                         and abs(float(policy.astype(np.float64).sum()) - 1) <= .001
                         and not np.any(policy[mask == 0]))
                audit.check("policy", "normalized finite legal mass (float16 tolerance 0.001)",
                            "valid" if valid else "invalid", game_index, ply, equal=bool(valid))
                expected = expected_target(outcome, mover, settings, iteration, spec)
                if mix and "outcome_value" in arrays:
                    found = float(arrays["outcome_value"][i])
                elif "value" in arrays and not mix:
                    found = float(arrays["value"][i])
                elif 0 <= recorded_mover < 2:
                    found = _stored_value(meta, recorded_mover)
                else:
                    found = float("nan")
                audit.number("value_target", expected, found, game_index, ply)
                tolerance = 1e-6
                if "value_target_mix" in arrays:
                    audit.number("value_target_mix", mix, arrays["value_target_mix"][i], game_index, ply)
                if mix:
                    root = arrays.get("root_value", [float("nan")] * len(players))[i]
                    valid_root = bool(np.isfinite(root) and abs(root) <= 1)
                    audit.check("root_value", "finite stored root value in [-1, 1]", float(root), game_index, ply,
                                equal=valid_root)
                    if not valid_root:
                        # Do not fabricate a mixed target or skip the history step.
                        raise ValueError("mixed target requires a finite stored root value in [-1, 1]")
                    expected, tolerance = mixed_target(expected, root, mix)
                    stored_mix, _ = mixed_target(found, root, mix)
                    found_mix = float(arrays["value"][i]) if "value" in arrays else stored_mix
                    audit.check("mixed_value_target", expected, found_mix, game_index, ply,
                                equal=bool(np.isfinite(found_mix) and abs(expected - found_mix) <= tolerance))
                if on_decision is not None and valid:
                    on_decision(planes, mask, policy, expected, mover, ply, outcome, tolerance)
            spec.step(game, {mover: action})
    finally:
        game.close()
    return outcome


def audit_trajectory(trajectory, spec, encoder, settings, iteration=1, backend="python", initial=None, values=None):
    """Audit an in-memory trajectory, treating its emitted values strictly as evidence."""
    joints = trajectory.joint_actions
    moves = [next(iter(j.values())) for j in joints] if joints else trajectory.actions
    arrays = {"moves": moves, "players": trajectory.players, "planes": trajectory.planes,
              "mask": trajectory.masks, "policy": trajectory.policies,
              "decision_plies": [e.get("decision_ply", i) for i, e in enumerate(trajectory.extras)],
              "value": trajectory.values(settings.get("truncation_value", 0)) if values is None else values}
    if settings.get("value_target_mix", 0) or any(e.get("value_target_mix", 0) for e in trajectory.extras):
        arrays.update(root_value=[e.get("root_value", float("nan")) for e in trajectory.extras],
                      value_target_mix=[e.get("value_target_mix", 0) for e in trajectory.extras],
                      outcome_value=[e.get("outcome_value", trajectory.outcome_value_for(
                          i, settings.get("truncation_value", 0))) for i, e in enumerate(trajectory.extras)])
    meta = {key: getattr(trajectory, key) for key in
            ("winner", "reason", "truncated", "plies", "rewards", "draw_value", "draw_reasons", "seed")}
    meta.update(iteration=iteration, joint_actions=joints, truncation_value=settings.get("truncation_value", 0))
    audit = Audit()
    audit_game(spec, encoder, arrays, meta, settings, audit, "trajectory", backend, initial=initial)
    return audit.report()


def _bytes(array, dtype):
    return np.ascontiguousarray(array, dtype=dtype).tobytes()


def _digest(*parts):
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
    return h.digest()


def _read_exact(source, count):
    data = source.read(count)
    if len(data) != count:
        raise ValueError("truncated NPY payload")
    return data


def _sample_member(archive, name, indices, size, expected_shape):
    """Read selected rows of a compressed NPY member with at most 1 MiB discard buffers."""
    with archive.open(name + ".npy", "r") as source:
        version = np.lib.format.read_magic(source)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(source)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(source)
        else:
            raise ValueError(f"unsupported NPY header {version}")
        if dtype.hasobject or fortran or tuple(shape[1:]) != tuple(expected_shape) or not 0 <= size <= shape[0]:
            raise ValueError(f"invalid replay {name} layout: {shape}, {dtype}")
        row_bytes = math.prod(shape[1:]) * dtype.itemsize
        if row_bytes > 16 * 2**20:
            raise ValueError("replay row exceeds diagnostic memory bound")
        rows, cursor = [], 0
        for index in indices:
            discard = (int(index) - cursor) * row_bytes
            while discard:
                amount = min(discard, 2**20)
                _read_exact(source, amount)
                discard -= amount
            data = _read_exact(source, row_bytes)
            rows.append(np.frombuffer(data, dtype=dtype).reshape(shape[1:]).copy())
            cursor = int(index) + 1
        return rows


class ReplaySample:
    """Exact matching of a bounded uniform replay sample, including spec symmetries."""

    def __init__(self, path, spec, encoder, count, seed):
        self.path = Path(path)
        self.rows = []
        self.by_policy = defaultdict(list)
        self.store_seen = 0
        self.store_matched = 0
        self.status = "not_available"
        self.reason = "replay.npz is absent"
        self.size = 0
        if not self.path.is_file():
            return
        try:
            with np.load(self.path, allow_pickle=False) as archive:
                self.size = int(archive["__size"].item())
            if self.size < 0:
                raise ValueError("negative replay size")
            indices = sorted(random.Random(seed).sample(range(self.size), min(count, self.size)))
            shapes = {"planes": (encoder.num_planes, *spec.board_shape),
                      "mask": (spec.action_size,), "policy": (spec.action_size,), "value": ()}
            with ZipFile(self.path, "r") as archive:
                arrays = {name: _sample_member(archive, name, indices, self.size, shape)
                          for name, shape in shapes.items()}
            for i, index in enumerate(indices):
                parts = [_bytes(arrays[name][i], dtype) for name, dtype in
                         (("planes", "<f4"), ("mask", "u1"), ("policy", "<f2"))]
                row = {"index": index, "parts": parts, "value": float(arrays["value"][i]),
                       "expected": set(), "tolerances": {}, "mixed": False,
                       "matches": 0, "sources": [], "symmetries": set()}
                self.by_policy[_digest(parts[2])].append(i)
                self.rows.append(row)
            self.status, self.reason = "measured", None
        except (OSError, ValueError, KeyError, BadZipFile, EOFError) as exc:
            self.rows.clear()
            self.by_policy.clear()
            self.reason = f"unreadable replay: {type(exc).__name__}: {exc}"

    def match(self, spec, planes, mask, policy, expected, game_index, ply, tolerance=1e-6, mixed=False):
        self.store_seen += 1
        matched = False
        variants = [("identity", None), *[(s.name, s) for s in spec.symmetries()]]
        for name, symmetry in variants:
            candidate_policy = policy if symmetry is None else symmetry.policy(policy)
            policy_bytes = _bytes(candidate_policy, "<f2")
            candidates = self.by_policy.get(_digest(policy_bytes), ())
            if not candidates:
                continue
            parts = [_bytes(planes if symmetry is None else symmetry.planes(planes), "<f4"),
                     _bytes(mask if symmetry is None else symmetry.policy(mask), "u1"), policy_bytes]
            for index in candidates:
                row = self.rows[index]
                if row["parts"] != parts:
                    continue
                matched = True
                row["expected"].add(float(np.float32(expected)))
                key = float(np.float32(expected))
                row["tolerances"][key] = max(tolerance, row["tolerances"].get(key, 0))
                row["mixed"] |= mixed
                row["matches"] += 1
                row["symmetries"].add(name)
                if len(row["sources"]) < 5:
                    row["sources"].append({"game_index": str(game_index), "ply": ply, "transform": name})
        self.store_matched += int(matched)

    def report(self, audit):
        matched = ambiguous = symmetry = 0
        unmatched_examples, ambiguous_examples = [], []
        for row in self.rows:
            expected = sorted(row["expected"])
            if not expected:
                if len(unmatched_examples) < 5:
                    unmatched_examples.append(row["index"])
                continue
            matched += 1
            symmetry += int(any(name != "identity" for name in row["symmetries"]))
            if len(expected) > 1:
                ambiguous += 1
                if len(ambiguous_examples) < 5:
                    ambiguous_examples.append({"row": row["index"], "expected_candidates": expected,
                                               "found": row["value"], "sources": row["sources"]})
                # A value outside every possible target is still an unambiguous failure.
                if any(abs(v - row["value"]) <= row["tolerances"][v] for v in expected):
                    continue
            equal = any(abs(v - row["value"]) <= row["tolerances"][v] for v in expected)
            audit.check("replay_value", expected, row["value"], row["sources"][0]["game_index"],
                        row["sources"][0]["ply"], equal=equal)
            if row["mixed"]:
                audit.check("mixed_value_target", expected, row["value"], row["sources"][0]["game_index"],
                            row["sources"][0]["ply"], equal=equal)
        return {"status": self.status, "reason": self.reason, "path": str(self.path),
                "population": {"n": self.size}, "sample": {"n": len(self.rows)},
                "matched": {"n": len(self.rows), "count": matched},
                "unmatched": {"n": len(self.rows), "count": len(self.rows) - matched, "examples": unmatched_examples},
                "ambiguous": {"n": matched, "count": ambiguous, "examples": ambiguous_examples},
                "symmetry_matches": {"n": matched, "count": symmetry},
                "store_decisions_matching_sample": {"n": self.store_seen, "count": self.store_matched},
                "matching": "Exact float32 planes / uint8 mask / float16 policy bytes; no policy renormalization. Unmatched rows have unknown provenance, not confirmed target errors."}


@dataclass
class IterationStats:
    games: int = 0
    plies: Mean = field(default_factory=Mean)
    entropy: Mean = field(default_factory=Mean)
    top1: Mean = field(default_factory=Mean)
    zero: int = 0
    legal: int = 0
    outcomes: Counter = field(default_factory=Counter)
    reasons: Counter = field(default_factory=Counter)
    modes: dict = field(default_factory=dict)

    def decision(self, mask, policy, mode="puct"):
        p = policy.astype(np.float64)
        p /= p.sum()
        positive = p[p > 0]
        self.entropy.add(-np.sum(positive * np.log(positive)))
        self.top1.add(p.max())
        entropy, top1 = self.modes.setdefault(mode, (Mean(), Mean()))
        entropy.add(-np.sum(positive * np.log(positive)))
        top1.add(p.max())
        self.zero += int(np.sum((mask > 0) & (p == 0)))
        self.legal += int(np.sum(mask > 0))

    def game(self, outcome):
        self.games += 1
        self.plies.add(outcome["plies"])
        kind = ("truncated" if outcome["truncated"] else "draw" if outcome["winner"] == -1
                else f"player_{outcome['winner']}_win")
        self.outcomes[kind] += 1
        self.reasons[outcome["reason"]] += 1

    def report(self):
        decisive = self.outcomes["player_0_win"] + self.outcomes["player_1_win"]
        return {"n": self.games, "mean_plies": self.plies.report(), "policy_entropy_nats": self.entropy.report(),
                "policy_top1_mass": self.top1.report(),
                "by_search_mode": {mode: {"policy_entropy_nats": entropy.report(),
                                          "policy_top1_mass": top1.report()}
                                   for mode, (entropy, top1) in sorted(self.modes.items())},
                "zero_visit_legal_share": {"n": self.legal, "count": self.zero,
                                          "rate": self.zero / self.legal if self.legal else None},
                "first_player_win_share_all_games": rate(self.outcomes["player_0_win"], self.games),
                "first_player_win_share_decisive": rate(self.outcomes["player_0_win"], decisive),
                "outcomes": {k: rate(v, self.games) for k, v in sorted(self.outcomes.items())},
                "end_reasons": {k: rate(v, self.games) for k, v in sorted(self.reasons.items())}}


def _predict(agent, planes):
    import torch

    with torch.inference_mode():
        _, values = agent.network(torch.as_tensor(np.stack(planes), dtype=torch.float32, device=agent.device))
    if values is None:
        raise ValueError("diagnostics require a value head")
    result = values.float().cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("value head emitted nonfinite predictions")
    return result


def calibration_report(agent, sample, seen):
    groups = defaultdict(list)
    all_rows = []
    for start in range(0, len(sample), 16):
        batch = sample[start:start + 16]
        predictions = _predict(agent, [row["planes"] for row in batch])
        for saved, value in zip(batch, predictions):
            row = {**saved, "prediction": float(value)}
            row.pop("planes")
            groups[(row["kind"], row["ply_range"])].append(row)
            all_rows.append(row)

    def metrics(rows):
        observed = [r for r in rows if r["outcome"] is not None]
        return {"n": len(rows), "prediction": distribution([r["prediction"] for r in rows]),
                "observed_outcome": distribution([r["outcome"] for r in observed]),
                "brier_score": distribution([((r["prediction"] - r["outcome"]) / 2)**2 for r in observed]),
                "training_target_mse": distribution([(r["prediction"] - r["target"])**2 for r in rows])}

    def reliability(rows):
        table = []
        buckets = 10
        for index in range(buckets):
            bucket = [r for r in rows if min(buckets - 1, max(0, int((r["prediction"] + 1) * 5))) == index]
            table.append({"bucket": index, "prediction_range": [-1 + index * .2, -1 + (index + 1) * .2],
                          **metrics(bucket)})
        return table

    return {"status": "measured" if sample else "not_available", "population": {"n": seen},
            "sampling": "Uniform reservoir over recorded decisions; at most --positions, seeded.",
            "definition": "Brier is ((v+1)/2 - (outcome+1)/2)^2, a fractional score for draws (0.5), not a three-class Brier. Truncations have no observed outcome and are excluded.",
            "overall": metrics(all_rows), "reliability": reliability(all_rows),
            "by_outcome_and_ply": [{"outcome_kind": kind, "ply_range": ply, **metrics(rows),
                                    "reliability": reliability(rows)}
                                   for (kind, ply), rows in sorted(groups.items())]}


def analyze_store(run, agent, positions, seed, backend, progress=None):
    """Stream every committed archive; retain at most N calibration and N replay rows."""
    from .games import default_rules

    run = Path(run)
    config_path = run / "config.json"
    if not config_path.is_file():
        return {"status": "not_available", "reason": "run config.json is absent; targets cannot be derived independently"}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    spec, encoder = agent.spec, agent.encoder
    configured_game = config.get("game", {"name": "hasami", "params": {}})
    if configured_game.get("name") != spec.game_id or configured_game.get("params", {}) != agent.manifest.game_params:
        raise ValueError("run game variant differs from checkpoint")
    algorithm = config.get("algorithm", {}).get("params", {})
    settings = dict(config.get("selfplay", {}), value_target_mix=algorithm.get("value_target_mix", 0))
    search_mode = algorithm.get("root_selection", "puct")
    store = Path(config.get("datastore", {}).get("path") or run / "selfplay")
    files = sorted((store / "games").glob("*.npz"))
    if not files:
        return {"status": "not_available", "reason": "selfplay datastore has no committed games"}
    replay_path = run / "replay.npz"
    if not replay_path.is_file():
        replay_path = Path(agent.checkpoint_dir) / "replay.npz"
    replay = ReplaySample(replay_path, spec, encoder, positions, seed + 1)
    audit, iterations = Audit(), defaultdict(IterationStats)
    rng, calibration, seen = random.Random(seed + 2), [], 0
    snapshot = []
    for path in files:
        try:
            stat = path.stat()
        except OSError as exc:
            audit.unavailable.append(f"Archive {path.name} unavailable during snapshot: {type(exc).__name__}: {exc}")
        else:
            snapshot.append((path, (stat.st_size, stat.st_mtime_ns)))
    rules = default_rules(spec)
    for number, (path, signature) in enumerate(snapshot):
        try:
            arrays, meta = _read_archive(path)
            identity = {"game_id": spec.game_id, "game_params": dict(agent.manifest.game_params),
                        "num_players": 2, "ruleset_id": spec.ruleset_id, "rules": rules}
            actual = {key: meta.get(key, expected) for key, expected in identity.items()}
            audit.check("variant", identity, actual, path.name)
            if actual != identity:
                raise ValueError("archive variant differs from run/checkpoint")
            stats = iterations[int(meta["iteration"])]

            def decision(planes, mask, policy, expected, mover, ply, outcome, tolerance):
                nonlocal seen
                stats.decision(mask, policy, search_mode)
                replay.match(spec, planes, mask, policy, expected, path.name, ply, tolerance,
                             mixed=bool(settings["value_target_mix"]))
                seen += 1
                slot = seen - 1 if len(calibration) < positions else rng.randrange(seen)
                if slot >= positions:
                    return
                kind = ("truncated" if meta["truncated"] else "draw:" + meta["reason"] if meta["winner"] == -1
                        else "win" if meta["winner"] == mover else "loss")
                target_outcome = None if meta["truncated"] else (0.0 if meta["winner"] == -1
                                                               else 1.0 if meta["winner"] == mover else -1.0)
                item = {"planes": planes.copy(), "target": expected, "outcome": target_outcome, "kind": kind,
                        "ply_range": "0-39" if ply < 40 else "40-99" if ply < 100 else "100+"}
                if slot < len(calibration):
                    calibration[slot] = item
                else:
                    calibration.append(item)

            outcome = audit_game(spec, encoder, arrays, meta, settings, audit, path.name, backend,
                                 on_decision=decision)
            stats.game(outcome)
            stat = path.stat()
            unchanged = signature == (stat.st_size, stat.st_mtime_ns)
            audit.check("archive", "unchanged during read", "unchanged" if unchanged else "changed",
                        path.name, equal=unchanged)
        except (OSError, ValueError, KeyError, IndexError, TypeError, BadZipFile, EOFError) as exc:
            audit.check("archive", "readable, replayable archive", f"{type(exc).__name__}: {exc}", path.name)
        if progress and (number + 1) % 250 == 0:
            progress(f"監査: {number + 1}/{len(snapshot)} 局、{audit.decisions} 判断")
    replay_report = replay.report(audit)
    for name in ("planes", "mask"):
        if not audit.checks[name]["n"]:
            audit.unavailable.append(f"Original trajectory {name} are not stored; see sampled exact replay matches.")
    if replay_report["status"] != "measured":
        audit.unavailable.append(replay_report["reason"])
    return {"status": "measured", "audit": audit.report(), "replay": replay_report,
            "target_statistics": {"status": "measured", "population": {"n": len(snapshot)},
                                  "iterations": {str(i): s.report() for i, s in sorted(iterations.items())}},
            "calibration": calibration_report(agent, calibration, seen),
            "scope": "Retained committed archives at enumeration time, including iterations newer than checkpoint. Concurrent changes are reported; this is not a locked training snapshot."}


def probe_report(agent, positions, simulations, seed, backend, progress=None):
    from .inference import MctsAgent, _mcts_params_from_manifest

    spec = agent.spec
    samples, seen = generate_positions(spec, positions, seed, backend, agent.manifest.rules)
    labels, values, actions = [], [], {"policy": [], "alphabeta:1": []}
    roots = {}
    try:
        for index, (game, _) in enumerate(samples):
            labels.append(label_position(spec, game))
            logits, value, mask = agent.evaluate(game)
            if value is None or not np.isfinite(value) or not np.isfinite(logits[mask > 0]).all():
                raise ValueError("nonfinite policy/value head output")
            values.append(value)
            actions["policy"].append(int(np.argmax(np.where(mask > 0, logits, -np.inf))))
            actions["alphabeta:1"].append(choose_baseline(spec, game, "alphabeta:1", seed))
            if progress and (index + 1) % 50 == 0:
                progress(f"戦術ラベル: {index + 1}/{len(samples)} 局面")
        params = _mcts_params_from_manifest(agent.manifest.algorithm_params)
        params.update((agent.manifest.inference_hint or {}).get("mcts_params", {}))
        params.update(add_noise=False, batch_size=16)
        for budget in sorted(set(simulations)):
            mode = f"mcts_{budget}"
            actions[mode], roots[mode] = [], []
            search_agent = MctsAgent(agent.network, agent.encoder, agent.device, simulations=budget,
                                     mcts_params=params, rng=np.random.default_rng(seed), temperature=0,
                                     amp="off", spec=spec)
            for begin in range(0, len(samples), 16):
                games = [g for g, _ in samples[begin:begin + 16]]
                results = search_agent.search.run_many(games, simulations=budget)
                for game, result in zip(games, results):
                    if result.simulations != budget:
                        raise ValueError("incomplete diagnostic search budget")
                    mask = spec.legal_mask(game, spec.actors(game)[0])
                    policy = result.policy_target if params.get("root_selection") == "gumbel" else result.visits
                    action = int(np.argmax(np.where(mask > 0, policy, -1)))
                    actions[mode].append(action)
                    roots[mode].append(float(result.root_value))
            if progress:
                progress(f"探索測定: {budget} simulations、{len(samples)} 局面")
        tactical = {}
        for name in CLASSES:
            selected = [i for i, label in enumerate(labels) if label[name]]
            supported = name != "capture" or any(label["capture"] is not None for label in labels)
            tactical[name] = {
                "status": "measured" if selected else "not_available",
                "reason": None if selected else "no sampled examples" if supported else "no capture-counter hook",
                "n": len(selected),
                "modes": {mode: rate(sum(chosen[i] in labels[i][name] for i in selected), len(selected))
                          for mode, chosen in actions.items()},
                "uniform_random": random_reference([(len(labels[i][name]), len(labels[i]["legal"])) for i in selected])}
        value_report = {}
        for name, predicate, sign in (("win_in_one", lambda l: bool(l["win_in_one"]), 1),
                                      ("forced_loss", lambda l: l["forced_loss"], -1)):
            selected = [i for i, label in enumerate(labels) if predicate(label)]
            value_report[name] = {
                "expected_sign": sign, "head": distribution([values[i] for i in selected]),
                "head_wrong_sign": rate(sum(values[i] * sign <= 0 for i in selected), len(selected)),
                "search_roots": {mode: {"value": distribution([v[i] for i in selected]),
                                        "wrong_sign": rate(sum(v[i] * sign <= 0 for i in selected), len(selected))}
                                 for mode, v in roots.items()}}
        hook = getattr(agent.encoder, "null_perspective", None)
        if hook is None:
            value_report["null_perspective"] = {
                "status": "not_available", "n": 0,
                "reason": "This encoder only accepts the current actor and promises no null-perspective antisymmetry. Changing turn changes the game, not merely its value perspective."}
        else:
            residuals = []
            for i, (game, _) in enumerate(samples):
                planes = agent.encoder.encode(game)
                residuals.append(values[i] + float(_predict(agent, [hook(planes.copy())])[0]))
            value_report["null_perspective"] = {"status": "measured", "sum_of_opposite_values": distribution(residuals)}
        return {"status": "measured" if samples else "not_available", "n": len(samples),
                "generated_population": {"n": seen}, "classes": tactical,
                "root_selection": params.get("root_selection", "puct"),
                "sampling": "Seeded reservoir over capped random, greedy and mixed trajectories; positions within a game are correlated. Wilson intervals are descriptive, not independent-game strength estimates.",
                "search": "Python MCTS via inference API, engine backend reported separately; fixed simulations, no noise or deadlines, temperature zero. mcts_0 uses its root prior.",
                "sample_generators": dict(Counter(meta["generator"] for _, meta in samples))}, value_report
    finally:
        for game, _ in samples:
            game.close()


def collect_anomalies(tactical, values, store):
    anomalies = []
    for name, row in tactical.get("classes", {}).items():
        reference = row["uniform_random"]["rate"]
        if reference is None:
            continue
        for mode, result in row["modes"].items():
            if mode == "alphabeta:1":
                continue
            if result["rate"] < reference:
                anomalies.append({"check": "tactical_below_random", "class": name, "mode": mode,
                                  "criterion": "measured success share < exact uniform legal reference",
                                  "n": result["n"], "observed": result["rate"], "reference": reference})
    for name in ("win_in_one", "forced_loss"):
        row = values[name]
        for mode, result in [("head", row["head_wrong_sign"]),
                             *[(m, r["wrong_sign"]) for m, r in row["search_roots"].items()]]:
            if result["n"] and result["rate"] > .05:
                anomalies.append({"check": "value_sign", "class": name, "mode": mode,
                                  "criterion": "wrong or zero sign share > 0.05 on exact terminal-adjacent labels",
                                  "n": result["n"], "observed": result["rate"], "threshold": .05})
    for name, row in store.get("audit", {}).get("checks", {}).items():
        if row["disagreements"]:
            anomalies.append({"check": "target_disagreement", "class": name,
                              "criterion": "any independently checked disagreement",
                              "n": row["n"], "count": row["disagreements"], "examples": row["examples"]})
    return anomalies


def diagnose(checkpoint, run=None, device="cpu", positions=300, simulations=(0, 64, 256), seed=0, progress=None):
    """Compute a deterministic report without opening any input for writing."""
    from .games.base import require_search_game
    from .inference import create_agent

    if type(positions) is not int or positions < 0 or any(type(s) is not int or s < 0 for s in simulations):
        raise ValueError("positions and simulations must be nonnegative integers")
    agent = create_agent(str(checkpoint), device=device, mode="policy", temperature=0, seed=seed,
                         amp="off", aux_outputs=False)
    require_search_game(agent.spec)
    backend = agent.spec.resolve_backend("auto")
    if progress:
        progress(f"チェックポイント: iteration {agent.manifest.iteration}、{device}、engine={backend}")
    tactical, values = probe_report(agent, positions, simulations, seed, backend, progress)
    store = (analyze_store(run, agent, positions, seed, backend, progress) if run is not None else
             {"status": "not_available", "reason": "--run was not supplied"})
    missing = {"status": "not_available", "reason": store.get("reason"), "n": 0}
    return {"schema_version": 1, "checkpoint": {"path": agent.checkpoint_dir,
                                                "iteration": agent.manifest.iteration, "game_id": agent.spec.game_id},
            "settings": {"device": str(agent.device), "engine_backend": backend, "seed": seed,
                         "positions_per_population": positions, "simulations": sorted(set(simulations)),
                         "memory": "At most N generated histories, N calibration positions and N replay rows; one archive at a time; compressed replay discards <=1 MiB at a time."},
            "tactical_probes": tactical, "value_probes": values,
            "calibration": store.get("calibration", missing), "target_statistics": store.get("target_statistics", missing),
            "target_audit": store.get("audit", missing), "replay_comparison": store.get("replay", missing),
            "store_scope": store.get("scope"), "anomalies": collect_anomalies(tactical, values, store)}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def validate_output_path(out, checkpoint, run=None):
    """Refuse output inside inputs, through symlinks, or over an existing file."""
    from .manifest import resolve_checkpoint_dir

    destination = Path(out).resolve()
    resolved = resolve_checkpoint_dir(checkpoint)
    roots = [Path(checkpoint).resolve(), resolved]
    if resolved.parent.name == "checkpoints":
        roots.append(resolved.parent.parent)
    if run is not None:
        root = Path(run).resolve()
        roots.append(root)
        config = root / "config.json"
        if config.is_file():
            store = json.loads(config.read_text()).get("datastore", {}).get("path")
            if store:
                roots.append(Path(store).resolve())
    if any(destination == root or root in destination.parents for root in roots):
        raise ValueError("--out must be outside the run, datastore and checkpoint")
    if destination.exists() or Path(out).is_symlink():
        raise ValueError("--out must be a new file (existing files are never overwritten)")
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--positions", type=int, default=300)
    parser.add_argument("--simulations", nargs="+", type=int, default=[0, 64, 256])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.positions < 0 or args.seed < 0 or any(s < 0 for s in args.simulations):
        parser.error("positions, simulations and seed must be nonnegative")
    import torch

    # Limit CPU contention with the independent long-running training process.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out = validate_output_path(args.out, args.checkpoint, args.run)
    report = diagnose(args.checkpoint, args.run, args.device, args.positions, args.simulations, args.seed,
                      progress=lambda text: print(text, file=sys.stderr, flush=True))
    payload = json.dumps(_json_safe(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with out.open("x", encoding="utf-8") as destination:
        destination.write(payload)
    print(f"測定済み: 戦術 {report['tactical_probes']['n']} 局面。異常基準への該当 {len(report['anomalies'])} 件。")
    audit = report["target_audit"]
    if audit["status"] == "measured":
        print(f"測定済み: 目標監査 {audit['games']['n']} 局 / {audit['decisions']['n']} 判断、相違 {audit['disagreements']['count']} 件。")
    unavailable = []
    for name in ("calibration", "target_statistics", "target_audit", "replay_comparison"):
        if report[name]["status"] == "not_available":
            unavailable.append(name + ": " + str(report[name].get("reason")))
    if report["value_probes"]["null_perspective"]["status"] == "not_available":
        unavailable.append("手番を変えない視点反転: エンコーダ契約に対応なし")
    for name, row in report["tactical_probes"]["classes"].items():
        if not row["n"]:
            unavailable.append(name + ": 該当標本なし／未対応")
    print("利用不可: " + ("; ".join(unavailable) if unavailable else "なし"))
    print(f"結果 JSON: {out}")
    return 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
