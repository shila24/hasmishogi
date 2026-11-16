"""Evaluation matches against classical baselines and pool members."""

from __future__ import annotations

import json
import math
import random
import secrets
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from hasami.agents import Agent, Decision, SearchLimits, check_cancel
from .games import get_spec, GameSpec
from .games.base import as_v2
from .selfplay import choose_agent, run_table_game
from .games.records import save_record


def wilson_interval(successes: float, games: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, allowing half points; no games gives [0, 1]."""
    if type(games) is not int or games < 0:
        raise ValueError("games must be a nonnegative integer")
    if not math.isfinite(successes) or not 0 <= successes <= games:
        raise ValueError("successes must be finite and between zero and games")
    if not math.isfinite(z) or z < 0:
        raise ValueError("z must be finite and nonnegative")
    if not games:
        return 0.0, 1.0
    p = successes / games
    z2 = z * z
    denominator = 1 + z2 / games
    center = (p + z2 / (2 * games)) / denominator
    margin = z * math.sqrt(p * (1 - p) / games + z2 / (4 * games * games)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


@dataclass
class MatchResult:
    """Evaluation totals; legacy draws include half-scored experimental caps.

    ``truncated`` and ``reasons['max_plies']`` identify those caps separately.
    They do not change the engine's ongoing outcome into a terminal draw.
    """

    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    truncated: int = 0
    plies: int = 0
    seconds: float = 0.0
    reasons: Counter[str] = field(default_factory=Counter)
    num_players: int = 2
    placements: Counter[int] = field(default_factory=Counter)
    mean_points: float = 0.0
    tables: list[dict[str, Any]] = field(default_factory=list)

    distinct_games: int | None = None
    distinct_openings: int | None = None
    effective_games: int | None = None
    effective_successes: float | None = None

    @property
    def sample(self) -> tuple[float, int]:
        """Group means and group count; legacy aggregate-only results retain raw intervals."""
        if self.effective_games is None:
            return self.wins + 0.5 * self.draws, self.games
        return self.effective_successes, self.effective_games

    @property
    def win_rate(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.0

    @property
    def draw_rate(self) -> float:
        """Share receiving a draw score, including separately reported caps."""
        return self.draws / self.games if self.games else 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # asdict reconstructs Counter from pairs, so serialize its mapping explicitly.
        data["reasons"] = dict(self.reasons)
        data.pop("tables")
        data.pop("num_players")
        if self.num_players == 2:
            data.pop("placements")
            data.pop("mean_points")
        else:
            data["placements"] = dict(self.placements)
        extra = {key: data.pop(key) for key in
                 ("distinct_games", "distinct_openings", "effective_games", "effective_successes")}
        if self.effective_games is not None:
            data["extra"] = {**extra, "sample_unit": "opening_actions_and_seating"}
        data["win_rate"] = round(self.win_rate, 4)
        data["elo_delta"] = round(estimate_elo_delta(self.win_rate), 1) if self.games else None
        data["ci_low"], data["ci_high"] = wilson_interval(*self.sample)
        data["draw_rate"] = round(self.draw_rate, 4)
        return data


def estimate_elo_delta(win_rate: float, cap: float = 800.0) -> float:
    p = min(max(win_rate, 1e-4), 1 - 1e-4)
    return max(-cap, min(cap, 400.0 * math.log10(p / (1 - p))))


def baseline_agent(spec: str, seed: int = 0) -> tuple[Agent, SearchLimits | None]:
    """Legacy Hasami baseline entry point; game-aware callers use their spec."""
    return get_spec().baseline_agent(spec, seed)


class _EvaluationAgent:
    """Replay an opening prefix, then apply simulation-only limits to learned MCTS."""

    def __init__(self, agent, opening, player):
        self.agent = agent
        self.actions = deque(step[player] for step in opening if player in step)

    def choose(self, game, limits, cancel=None, player=None):
        check_cancel(cancel)
        if self.actions:
            return Decision(self.actions.popleft())
        from .inference import MctsAgent

        if isinstance(self.agent, MctsAgent):
            return self.agent.choose(game, replace(limits, node_limit=self.agent.simulations), cancel,
                                     player=player, use_time_limit=False)
        return choose_agent(self.agent, game, limits, cancel, player)


def _opening(spec, backend, rules, deal_seed, opening_seed, cycle, plies, cancel):
    """Draw at most 128 nonterminal prefixes, keeping chance on the original deal seed."""
    for stream in range(128):
        rng = random.Random(f"{opening_seed}:{cycle}:{stream}")
        game = spec.create_game(backend, rules, seed=deal_seed)
        actions = []
        try:
            if game.outcome.terminal or getattr(game, "truncated", False):
                continue
            for _ in range(plies):
                check_cancel(cancel)
                joint = {player: int(rng.choice(spec.legal_actions(game, player)))
                         for player in spec.actors(game)}
                outcome = spec.step(game, joint)
                if outcome.terminal or outcome.truncated:
                    break
                actions.append(joint)
            if len(actions) == plies:
                return actions
        finally:
            game.close()
    raise ValueError(f"could not draw a nonterminal opening of {plies} plies after 128 attempts (cycle {cycle})")


def _action_key(actions):
    return tuple(tuple(sorted(step.items())) for step in actions)


def play_matches(
    agent: Agent, opponent: Agent | list[Agent], games: int, *,
    spec: GameSpec | None = None, backend: str = "python", max_plies: int | None = None,
    limits: SearchLimits, opponent_limits=None, swap_sides: bool = True,
    rules=None, records_dir: Path | None = None, cancel=None, seed: int | None = None,
    opening_plies: int = 0, opening_seed: int | None = None,
) -> MatchResult:
    """Rotate the table assignment; games remains the total number of tables.

    A seed is shared by each complete cycle of seat rotations for duplicate
    deals. Ties use competition ranks (one plus the number of higher scores).
    Caps receive legacy half points, and no placement or terminal points.
    Opening plies are joint steps and count toward both plies and max_plies
    (zero cap means unlimited). Rotations share the entire opening and deal.
    The opening RNG uses only opening_seed (default seed, or zero), cycle and
    redraw index. Groups are (opening actions, seating), regardless of deal seed;
    their mean scores form the effective Wilson sample. Raw counts remain raw.
    Native records contain the opening in their full move history; optional
    openings/eval_NNNN.json sidecars label that prefix without changing formats.
    """
    spec = as_v2(spec or get_spec())
    if type(games) is not int or games < 0:
        raise ValueError("games must be a nonnegative integer")
    if type(opening_plies) is not int or opening_plies < 0:
        raise ValueError("opening_plies must be a nonnegative integer")
    for name, value in (("seed", seed), ("opening_seed", opening_seed)):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"{name} must be a nonnegative integer or None")
    cap = spec.default_max_plies if max_plies is None else max_plies
    if type(cap) is not int or cap < 0:
        raise ValueError("max_plies must be nonnegative")
    if cap and opening_plies > cap:
        raise ValueError("opening_plies must not exceed max_plies")
    opening_seed = (seed if seed is not None else 0) if opening_seed is None else opening_seed
    opponents = list(opponent) if isinstance(opponent, (list, tuple)) else [opponent] * (spec.num_players - 1)
    if len(opponents) != spec.num_players - 1:
        raise ValueError("one opponent is required for every remaining seat")
    assignment = [agent, *opponents]
    other_limits = (list(opponent_limits) if isinstance(opponent_limits, (list, tuple))
                    else [opponent_limits or limits] * len(opponents))
    if len(other_limits) != len(opponents):
        raise ValueError("opponent_limits must match opponents")
    budgets = [limits, *other_limits]
    result = MatchResult(num_players=spec.num_players)
    start = time.perf_counter()
    points = 0.0
    completed = 0
    groups = defaultdict(list)
    distinct = set()
    openings = set()
    opening = []
    for index in range(games):
        rotation = index % spec.num_players if swap_sides else 0
        cycle = index // spec.num_players if swap_sides else index
        seats = {(i + rotation) % spec.num_players: mover for i, mover in enumerate(assignment)}
        seat_limits = {(i + rotation) % spec.num_players: budget for i, budget in enumerate(budgets)}
        if rotation == 0:
            deal_seed = None if seed is None else seed + cycle
            if opening_plies:
                deal_seed = secrets.randbits(63) if deal_seed is None else deal_seed
                opening = _opening(spec, backend, rules, deal_seed, opening_seed, cycle, opening_plies, cancel)
        seats = {player: _EvaluationAgent(mover, opening, player) for player, mover in seats.items()}
        def save(game, trajectory):
            if records_dir is not None:
                records_dir.mkdir(parents=True, exist_ok=True)
                save_record(spec, game, records_dir / f"eval_{index:04d}.json", trajectory=trajectory)
                if opening_plies:
                    metadata = records_dir / "openings"
                    metadata.mkdir(exist_ok=True)
                    (metadata / f"eval_{index:04d}.json").write_text(json.dumps({
                        "opening_actions": opening, "opening_seed": opening_seed,
                        "seed": trajectory.seed, "rotation": rotation,
                    }) + "\n", encoding="utf-8")
        trajectory = run_table_game(seats, spec.make_encoder(""), spec=spec, backend=backend,
                                    max_plies=max_plies, limits=seat_limits, rules=rules, cancel=cancel,
                                    seed=deal_seed, on_finish=save)
        result.games += 1
        result.plies += trajectory.plies
        result.reasons[trajectory.reason] += 1
        values = [trajectory.rewards[(i + rotation) % spec.num_players] for i in range(spec.num_players)]
        result.tables.append({"seed": trajectory.seed, "rotation": rotation, "values": values,
                              "truncated": trajectory.truncated, "step_info": trajectory.step_info,
                              "opening_actions": opening, "actions": trajectory.joint_actions})
        opening_key = _action_key(opening)
        openings.add(opening_key)
        distinct.add((_action_key(trajectory.joint_actions), rotation))
        score = (0.5 if trajectory.truncated else 1.0 if all(values[0] > v for v in values[1:])
                 else 0.0 if any(v > values[0] for v in values[1:]) else 0.5)
        groups[(opening_key, rotation)].append(score)
        if trajectory.truncated:
            result.truncated += 1
            result.draws += 1
            continue
        completed += 1
        points += values[0]
        rank = 1 + sum(value > values[0] for value in values[1:])
        result.placements[rank] += 1
        if all(values[0] > value for value in values[1:]):
            result.wins += 1
        elif any(value > values[0] for value in values[1:]):
            result.losses += 1
        else:
            result.draws += 1
    result.distinct_games = len(distinct)
    result.distinct_openings = len(openings)
    result.effective_games = len(groups)
    result.effective_successes = sum(sum(scores) / len(scores) for scores in groups.values())
    result.mean_points = points / completed if completed else 0.0
    result.seconds = time.perf_counter() - start
    return result
