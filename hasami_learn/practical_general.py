"""Sampled expectimax for GameSpec v2, without training-time dependencies.

Depth counts arrivals at the advised player's further decisions: D=1 stops at
its next decision, D=2 samples that decision and stops at the following one.
Chance is resolved exclusively by spec.step in a freshly sampled world per rollout.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from hasami.agents import Decision, check_cancel
from .games.base import terminal_rewards
from .practical import (LevelInfo, PracticalAdvisor, ProxyAgent, normalize,
                        sample_distribution, top_distribution)


def active_player(spec, game, player=None):
    """Require an explicit seat for a simultaneous decision."""
    actors = spec.actors(game)
    if player is None:
        if len(actors) != 1:
            raise ValueError("specify player for a joint decision; chance/terminal states have no advice")
        player = actors[0]
    if type(player) is not int or player not in actors:
        raise ValueError("advised player must be an active actor")
    return player


def outcome_score(spec, values, player):
    """Map fixed-seat terminal values through the adapter's optional score hook."""
    if len(values) != spec.num_players or any(not math.isfinite(v) or not -1 <= v <= 1 for v in values):
        raise ValueError("terminal values must be finite, bounded and contain every seat")
    hook = getattr(spec, "outcome_score", None)
    score = float(hook(values, player) if hook else (values[player] + 1) / 2)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("outcome_score must be finite and in [0, 1]")
    return score


class GeneralProxy(ProxyAgent):
    """Temperature/nucleus policy plus a prior-top-eight lapse, on one's own observation."""

    def distribution(self, game, cancel=None, *, player=None):
        player = active_player(self.spec, game, player)
        check_cancel(cancel)
        legal = self.spec.legal_actions(game, player)
        if len(legal) == 1:
            return {int(legal[0]): 1.0}
        prior = self.engine.evaluate([game], cancel, players=[player])[0].prior
        if set(prior) != set(legal):
            raise ValueError("engine prior must contain exactly the actor's legal actions")
        return self.from_prior(prior)

    def from_prior(self, prior, q=None, best=None):
        base = super().from_prior(prior)
        lapse = top_distribution(prior)
        epsilon = self.params["epsilon"]
        return normalize({a: (1 - epsilon) * base.get(a, 0) + epsilon * lapse.get(a, 0) for a in prior})

    def choose(self, game, limits, cancel=None, player=None):
        started = time.perf_counter()
        distribution = self.distribution(game, cancel, player=player)
        check_cancel(cancel)
        return Decision(sample_distribution(distribution, self.rng),
                        elapsed_ms=(time.perf_counter() - started) * 1000)


class GeneralSearchAdvisor(PracticalAdvisor):
    """Method A with explicit seat identity, owned determinizations and sampled joint steps.

    Guard values are mean one-step engine values for the advised seat over K*M
    samples, on the same scale as Advice.engine_value. Difficulty estimates the
    next own decision's probability-weighted error share using one-step engine
    action values on each sampled state. No nested determinization is performed.
    Safety caps bootstrap the current observation; adapter truncations score half.
    """

    method = "search"
    general = True

    def __init__(self, engine, levels=None, *, checkpoint_root=None,
                 determinizations=4, rollouts=2, seed=0, max_steps=None):
        self.engine, self.spec = engine, engine.spec
        if not self.spec.perfect_information and not callable(getattr(self.spec, "determinize", None)):
            raise ValueError("imperfect-information advice requires spec.determinize")
        chance = getattr(self.spec, "has_chance", True)
        if self.spec.perfect_information and chance and not callable(getattr(self.spec, "resample_chance", None)):
            raise ValueError("perfect-information chance advice requires spec.resample_chance")
        for name, value in (("determinizations", determinizations), ("rollouts", rollouts)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.determinizations = 1 if self.spec.perfect_information and not chance else determinizations
        self.rollouts = rollouts
        self.seed = self._seed(seed)
        self.max_steps = getattr(self.spec, "default_max_plies", 10000) if max_steps is None else max_steps
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.checkpoint_root = Path(checkpoint_root or ".").resolve()
        self.last_search = None
        # Section 11 deliberately does not interpret a two-player Elo artifact as a strength table.
        self.levels = [LevelInfo(i, None, None, ("temperature",)) for i in range(1, 7)]
        self.level_data = dict(k=6, game_id=self.spec.game_id, train_styles=["temperature"], heldout_styles=[],
            notes="Uncalibrated temperature schedule; multiplayer strength has not been measured.",
            levels=[dict(level=i, elo_center=None, styles={"temperature": dict(
                available=True, elo=None, ci=None, games=0,
                params=dict(temperature=2 / i, epsilon=0.05))}) for i in range(1, 7)])

    @staticmethod
    def _seed(seed):
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        return seed

    def proxy_agent(self, level, *, style="mixture", seed=0):
        row = self._level(level)
        if style not in ("mixture", "temperature"):
            raise ValueError("general advice supports only the uncalibrated temperature style")
        return GeneralProxy(self.engine, "temperature", row["styles"]["temperature"]["params"], seed)

    def joint_actions(self, game, player, own, other, rng, cancel=None, action=None):
        """Sample all actors before stepping, with no actor seeing another's chosen action."""
        actions = {}
        for seat in self.spec.actors(game):
            check_cancel(cancel)
            if seat == player and action is not None:
                if action not in self.spec.legal_actions(game, seat):
                    raise ValueError("candidate became illegal in a determinization")
                actions[seat] = int(action)
            else:
                proxy = own if seat == player else other
                actions[seat] = sample_distribution(proxy.distribution(game, cancel, player=seat), rng)
        return actions

    def _score(self, game, player, outcome=None, cancel=None):
        check_cancel(cancel)
        if outcome is not None and outcome.truncated:
            return 0.5
        if (outcome is not None and outcome.terminal) or game.outcome.terminal:
            values = (outcome.rewards if outcome is not None and outcome.rewards is not None
                      else terminal_rewards(self.spec, game))
            return outcome_score(self.spec, values, player)
        value = float(self.engine.evaluate([game], cancel, players=[player])[0].value)
        if not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError("engine value must be finite and in [-1, 1]")
        return (value + 1) / 2

    def _node_values(self, game, player, own, other, rng, cancel):
        """One-step values on an already sampled state; opponents use their own observations."""
        values = {}
        for action in self.spec.legal_actions(game, player):
            check_cancel(cancel)
            child = game.clone()
            try:
                outcome = self.spec.step(child, self.joint_actions(
                    child, player, own, other, rng, cancel, action))
                values[action] = 2 * self._score(child, player, outcome, cancel) - 1
            finally:
                child.close()
        return values

    def decision_difficulty(self, game, player, level, opponent_level, rng, cancel=None):
        own, other = self.proxy_agent(level), self.proxy_agent(opponent_level)
        values = self._node_values(game, player, own, other, rng, cancel)
        best = max(values.values())
        errors = {a: best - value > 0.2 for a, value in values.items()}
        probabilities = own.distribution(game, cancel, player=player)
        return (sum(probabilities.get(a, 0) for a in values if errors[a]),
                sum(errors.values()) / len(errors), int(sum(not e for e in errors.values()) == 1))

    def _estimate(self, game, player, actions, level, opponent_level, depth, seed, cancel):
        own, other = self.proxy_agent(level), self.proxy_agent(opponent_level)
        scores, difficulties = {}, {}
        caps = truncated = 0
        for action in actions:
            samples, errors = [], []
            for k in range(self.determinizations):
                for r in range(self.rollouts):
                    check_cancel(cancel)
                    world_rng = np.random.default_rng([seed, action, k, r, 0])
                    rng = np.random.default_rng([seed, action, k, r, 1])
                    error_rng = np.random.default_rng([seed, action, k, r, 2])
                    if not self.spec.perfect_information:
                        work = self.spec.determinize(game, player, world_rng)
                    elif getattr(self.spec, "has_chance", True):
                        work = self.spec.resample_chance(game, world_rng)
                    else:
                        work = game.clone()
                    if work is game:
                        raise ValueError("world sampling must return an owned clone")
                    try:
                        outcome = self.spec.step(work, self.joint_actions(
                            work, player, own, other, rng, cancel, action))
                        visits, error, steps = 0, 0.0, 1
                        while depth and not (outcome.terminal or outcome.truncated):
                            check_cancel(cancel)
                            if player in self.spec.actors(work):
                                visits += 1
                                if visits == 1:
                                    error = self.decision_difficulty(
                                        work, player, level, opponent_level, error_rng, cancel)[0]
                                if visits >= depth:
                                    break
                            if steps >= self.max_steps:
                                caps += 1
                                if visits == 0:
                                    error = None
                                break
                            outcome = self.spec.step(work, self.joint_actions(
                                work, player, own, other, rng, cancel))
                            steps += 1
                        truncated += int(outcome.truncated)
                        samples.append(self._score(work, player, outcome, cancel))
                        errors.append(error)
                    finally:
                        work.close()
            scores[action] = float(np.mean(samples))
            difficulties[action] = None if any(e is None for e in errors) else float(np.mean(errors))
        check_cancel(cancel)
        return scores, difficulties, caps, truncated

    def action_values(self, game, *, player=None, level=3, opponent_level=None, seed=None, cancel=None):
        player = active_player(self.spec, game, player)
        other = level if opponent_level is None else opponent_level
        scores, _, _, _ = self._estimate(game, player, self.spec.legal_actions(game, player),
            level, other, 0, self.seed if seed is None else self._seed(seed), cancel)
        return {a: 2 * score - 1 for a, score in scores.items()}

    def engine_best(self, game, cancel=None, *, player=None, level=3, opponent_level=None, seed=None):
        player = active_player(self.spec, game, player)
        values = self.action_values(game, player=player, level=level, opponent_level=opponent_level,
                                   seed=seed, cancel=cancel)
        best = min(values, key=lambda a: (-values[a], a))
        prior = self.engine.evaluate([game], cancel, players=[player])[0].prior
        return self._advice(game, best, (values[best] + 1) / 2, values[best], 0, None, prior[best])

    def advise(self, game, level, opponent_level=None, *, top_k=3, guard=0.3, depth=1,
               player=None, seed=None, cancel=None):
        started = time.perf_counter()
        self.last_search = None
        check_cancel(cancel)
        player = active_player(self.spec, game, player)
        other = level if opponent_level is None else opponent_level
        self._level(level)
        self._level(other)
        if (type(top_k) is not int or top_k < 1 or type(depth) is not int or depth < 1
                or not math.isfinite(guard) or guard < 0):
            raise ValueError("invalid advice top_k, depth or guard")
        seed = self.seed if seed is None else self._seed(seed)
        prior = self.engine.evaluate([game], cancel, players=[player])[0].prior
        if set(prior) != set(self.spec.legal_actions(game, player)) or not prior:
            raise ValueError("engine prior must contain exactly the actor's legal actions")
        prior = normalize(prior)
        values = self.action_values(game, player=player, level=level, opponent_level=other, seed=seed, cancel=cancel)
        best = min(values, key=lambda a: (-values[a], a))
        candidates = set(sorted(prior, key=lambda a: (-prior[a], a))[:16]) | {a for a in prior if prior[a] >= 0.01}
        candidates = sorted(a for a in candidates if values[best] - values[a] <= guard)
        fallback = not candidates
        if fallback:
            candidates = [best]
        scores, errors, caps, truncated = self._estimate(game, player, candidates, level, other, depth, seed, cancel)
        naturalness = self.proxy_agent(level).from_prior(prior)
        self.last_search = dict(depth=depth, determinizations=self.determinizations, rollouts=self.rollouts,
            seed=seed, player=player, capped_rollouts=caps, truncated_rollouts=truncated,
            elapsed_ms=(time.perf_counter() - started) * 1000)
        ranked = sorted(candidates, key=lambda a: (-scores[a], -values[a], a))
        return [self._advice(game, a, scores[a], values[a], values[best] - values[a], errors[a],
                            naturalness.get(a, 0), fallback) for a in ranked[:top_k]]
