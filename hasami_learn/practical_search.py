"""Method A: batched, history-preserving expectimax over a frozen engine.

Depth counts continuation plies after the advised move. Both players use budget-zero
QRE over the top eight priors plus terminal moves; each chance node then retains
its six most probable moves and renormalizes them. Public proxies retain calibrated budgets.
Difficulty averages the prefiltered QRE error probability over own decision nodes,
weighted by their reach probabilities (zero if there are no own decisions).
"""

from __future__ import annotations

import math
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from hasami.agents import check_cancel
from .games.base import require_search_game, terminal_rewards
from .practical import (LevelInfo, PracticalAdvisor, ProxyAgent,
                        mover, top_distribution)


class _SearchExpired(Exception):
    """Discard an incomplete deepening pass, preserving completed advice."""


class _SearchControl:
    def __init__(self, cancel, deadline=None):
        self.cancel, self.deadline = cancel, deadline

    def is_set(self):
        # User cancellation always takes priority over the optional time budget.
        check_cancel(self.cancel)
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise _SearchExpired
        return False


@dataclass
class _Edge:
    game: object | None = None
    row: object | None = None
    rewards: tuple[float, ...] | None = None

    def value_for(self, spec, player):
        if self.rewards is not None:
            return self.rewards[player]
        value = self.row.value * (1 if mover(spec, self.game) == player else -1)
        if not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError("engine value must be finite and in [-1, 1]")
        return value


class SearchAdvisor(PracticalAdvisor):
    """Engine-only advisor; no practical/slip heads or training modules are needed."""

    method = "search"

    def __init__(self, engine, levels=None, *, checkpoint_root=None):
        require_search_game(engine.spec)
        self.last_search = None
        if levels is not None:
            super().__init__(engine, None, levels, checkpoint_root=checkpoint_root)
        else:
            self.engine, self.spec = engine, engine.spec
            self.checkpoint_root = Path(checkpoint_root or ".").resolve()
            self.checkpoint_engines = {}
            self.levels = [LevelInfo(i, None, None, ("qre",)) for i in range(1, 7)]
            # This is runtime configuration, not a fabricated calibration artifact.
            self.level_data = dict(k=6, game_id=self.spec.game_id, train_styles=["qre"], heldout_styles=[],
                                   notes="Uncalibrated default schedule; Elo has not been measured.",
                                   levels=[dict(level=i, elo_center=None, styles={"qre": dict(
                                       available=True, elo=None, ci=None, games=0,
                                       params=dict(beta=2**(i-1), alpha=0.5, epsilon=0.05, simulations=0))})
                                           for i in range(1, 7)])

    def _search_proxy(self, level):
        """Private one-ply policy; leave the public calibrated proxy unchanged."""
        params = dict(super().proxy_agent(level, style="qre").params)
        params["simulations"] = 0
        return ProxyAgent(self.engine, "qre", params)

    @staticmethod
    def error_probability(values, probabilities):
        """Probability of losing more than delta=0.2 against the best evaluated action."""
        best = max(values.values())
        return sum(p for a, p in probabilities.items() if best - values[a] > 0.2)

    def _successors(self, games, priors, stack, cancel, *, prefilter=True):
        """Inspect every move for terminals; evaluate only retained children, once per layer.

        Child rows are reused as the next frontier's priors and as leaf values.
        No state or value is shared between different histories.
        """
        outputs, pending = [], []
        for game, prior in zip(games, priors):
            player = mover(self.spec, game)
            retained = set(sorted(prior, key=lambda a: (-prior[a], a))[:8]) if prefilter else set(prior)
            edges = {}
            outputs.append(edges)
            for action in prior:
                check_cancel(cancel)
                with ExitStack() as ownership:
                    child = game.clone()
                    ownership.callback(child.close)
                    outcome = self.spec.step(child, {player: action})
                    if outcome.terminal:
                        rewards = tuple(float(v) for v in (outcome.rewards if outcome.rewards is not None
                                                          else terminal_rewards(self.spec, child)))
                        if len(rewards) != 2 or any(not math.isfinite(v) or not -1 <= v <= 1 for v in rewards):
                            raise ValueError("terminal values must be finite and in [-1, 1]")
                        edges[action] = _Edge(rewards=rewards)
                    elif outcome.truncated:
                        # Caps score half, without turning them into rule draws.
                        edges[action] = _Edge(rewards=(0.0, 0.0))
                    elif action in retained:
                        edge = edges[action] = _Edge(game=child)
                        pending.append(edge)
                        stack.enter_context(ownership.pop_all())
        if pending:
            rows = self.engine.evaluate([edge.game for edge in pending], cancel)
            for edge, row in zip(pending, rows):
                edge.row = row
        check_cancel(cancel)
        return outputs, len(pending)

    def _search(self, game, candidates, player, own_proxy, other_proxy, depth, cancel):
        scores = dict.fromkeys(candidates, 0.0)
        errors = dict.fromkeys(candidates, 0.0)
        own_visits = dict.fromkeys(candidates, 0.0)
        evaluated = 0

        def advance(candidate, mass, edge, frontier):
            if edge.rewards is not None:
                scores[candidate] += mass * (edge.value_for(self.spec, player) + 1) / 2
            else:
                frontier.append((candidate, mass, edge))

        with ExitStack() as lifetime:
            current = lifetime.enter_context(ExitStack())
            roots, count = self._successors([game], [dict.fromkeys(candidates, 1.0)], current,
                                             cancel, prefilter=False)
            evaluated += count
            frontier = []
            for action, edge in roots[0].items():
                advance(action, 1.0, edge, frontier)
            for _ in range(depth):
                if not frontier:
                    break
                following = lifetime.enter_context(ExitStack())
                children, count = self._successors([e.game for _, _, e in frontier],
                    [e.row.prior for _, _, e in frontier], following, cancel)
                evaluated += count
                next_frontier = []
                for (candidate, mass, node), edges in zip(frontier, children):
                    check_cancel(cancel)
                    seat = mover(self.spec, node.game)
                    own_turn = seat == player
                    qs = {a: edge.value_for(self.spec, seat) for a, edge in edges.items()}
                    proxy = own_proxy if own_turn else other_proxy
                    probabilities = proxy.from_prior({a: node.row.prior[a] for a in edges}, qs)
                    if own_turn:
                        errors[candidate] += mass * self.error_probability(qs, probabilities)
                        own_visits[candidate] += mass
                    for action, probability in top_distribution(probabilities, 6).items():
                        if probability > 0:
                            advance(candidate, mass * probability, edges[action], next_frontier)
                current.close()
                current, frontier = following, next_frontier
            for candidate, mass, edge in frontier:
                scores[candidate] += mass * (edge.value_for(self.spec, player) + 1) / 2
        check_cancel(cancel)
        difficulty = {a: min(1.0, errors[a] / own_visits[a]) if own_visits[a] else 0.0 for a in candidates}
        return scores, difficulty, evaluated

    def advise(self, game, level, opponent_level=None, *, top_k=3, guard=0.3, depth=2,
               time_ms=None, cancel=None):
        """With time_ms, complete depth 2 then attempt 4 within a cooperative budget.

        The budget profile overrides depth; its mandatory depth-2 result may exceed
        the budget. last_search reports the deepest complete result, not partial work.
        """
        started = time.perf_counter()
        self.last_search = None
        check_cancel(cancel)
        opponent_level = level if opponent_level is None else opponent_level
        self._level(level)
        self._level(opponent_level)
        if (type(top_k) is not int or top_k < 1 or type(depth) is not int or depth not in (1, 2, 4, 6)
                or not math.isfinite(guard) or guard < 0):
            raise ValueError("invalid advice top_k, depth or guard")
        if time_ms is not None and (isinstance(time_ms, bool) or not math.isfinite(time_ms) or time_ms < 0):
            raise ValueError("time_ms must be finite and nonnegative")
        player = mover(self.spec, game)
        own_proxy = self._search_proxy(level)
        other_proxy = self._search_proxy(opponent_level)
        prior = self.engine.evaluate([game], cancel)[0].prior
        values = self.engine.one_ply(game, cancel)
        best = min(values, key=lambda a: (-values[a], a))
        candidates = set(sorted(prior, key=lambda a: (-prior[a], a))[:16]) | {a for a in prior if prior[a] >= 0.01}
        candidates = sorted(a for a in candidates if values[best] - values[a] <= guard)
        fallback = not candidates
        if fallback:
            candidates = [best]
        # Section 5 naturalness uses the full calibrated style mixture, not the pruned tree policy.
        naturalness = self.proxy_agent(level).distribution(game, cancel)
        reached = 2 if time_ms is not None else depth
        scores, difficulties, evaluated = self._search(game, candidates, player, own_proxy, other_proxy,
                                                      reached, cancel)
        completed_depths = [reached]
        if time_ms is not None and time.perf_counter() < started + time_ms / 1000:
            control = _SearchControl(cancel, started + time_ms / 1000)
            try:
                deeper = self._search(game, candidates, player, own_proxy, other_proxy, 4, control)
            except _SearchExpired:
                pass
            else:
                scores, difficulties, evaluated = deeper
                reached = 4
                completed_depths.append(reached)
        check_cancel(cancel)
        self.last_search = dict(depth=reached, completed_depths=completed_depths,
                                evaluated_positions=evaluated, elapsed_ms=(time.perf_counter() - started) * 1000)
        ranked = sorted(candidates, key=lambda a: (-scores[a], -values[a], a))
        return [self._advice(game, a, min(1.0, max(0.0, scores[a])), values[a], values[best] - values[a],
                            difficulties[a],
                            naturalness.get(a, 0.0), fallback) for a in ranked[:top_k]]
