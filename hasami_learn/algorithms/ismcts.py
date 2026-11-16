"""Single-observer information-set UCT with fresh belief samples per simulation.

Riichi's default sampler uses visible tile conservation and optional predicted
hand distributions. It is an approximate prior: it does not condition on every
past choice, riichi tenpai, or opponents' unobserved furiten. Hidden flags are
reset and future chance is reseeded. A custom sampler can supply a richer
posterior. No perfect-information MCTS or engine seed replay is used.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from hasami.agents import Decision, SearchLimits, check_cancel
from ..games import get_spec
from ..games.base import as_v2
from ..games.riichi import sample_riichi_belief  # runtime determinization lives in the adapter


@dataclass
class _Node:
    visits: dict[int, int] = field(default_factory=dict)
    totals: dict[int, float] = field(default_factory=dict)
    available: dict[int, int] = field(default_factory=dict)


class IsmctsAgent:
    """UCT for one observer, stochastic legal opponents, bounded rollout depth.

    A sampler must return an independently owned game. Other games require a
    sampler(game, player, rng, belief) callback. ``belief_agent`` may be a loaded
    PolicyAgent; only its activated auxiliary outputs seed the sampler.
    """

    def __init__(self, *, spec=None, simulations=64, rollout_steps=64, exploration=1.4,
                 truncation_value=0.0, seed=0, sampler=None, belief_agent=None):
        self.spec = as_v2(spec or get_spec("riichi"))
        if any(type(n) is not int or n < 1 for n in (simulations, rollout_steps)):
            raise ValueError("simulations and rollout_steps must be positive integers")
        if not np.isfinite(exploration) or exploration < 0 or not np.isfinite(truncation_value):
            raise ValueError("invalid exploration or truncation value")
        self.simulations, self.rollout_steps = simulations, rollout_steps
        self.exploration, self.truncation_value = exploration, truncation_value
        self.rng = np.random.default_rng(seed)
        self.sampler = sampler
        if self.sampler is None:
            if self.spec.game_id != "riichi":
                raise ValueError("this game requires an explicit information-set sampler")
            self.sampler = sample_riichi_belief
        self.belief_agent = belief_agent

    def _key(self, game, player, mask):
        observation = np.array(self.spec.observe(game, player), dtype=np.float32, copy=True)
        # History distinguishes information sets with equal current feature planes.
        history = tuple(tuple(sorted(step.items())) for step in getattr(game, "moves", [])
                        if isinstance(step, dict))
        return observation.tobytes(), mask.tobytes(), history

    def _rollout_action(self, legal):
        if self.spec.game_id == "riichi":
            for win in (108, 107):
                if win in legal:
                    return win
            if 109 in legal:
                return 109
            discards = [action for action in legal if action < 68]
            if discards:
                return int(self.rng.choice(discards))
        return int(self.rng.choice(legal))

    def _leaf(self, game, player):
        if self.spec.game_id != "riichi":
            return float(self.truncation_value)
        obs = game.observation(player)
        # A bounded heuristic uses the observer's own shanten and public points.
        return float(np.tanh((obs["scores"][0] - np.mean(obs["scores"])) / 30000
                             - 0.08 * obs["shanten"]))

    def choose(self, game, limits=None, cancel=None, *, player=None):
        check_cancel(cancel)
        start = time.perf_counter()
        limits = limits or SearchLimits()
        actors = self.spec.actors(game)
        if player is None:
            if len(actors) != 1:
                raise ValueError("specify player for simultaneous decisions")
            player = actors[0]
        mask = np.array(self.spec.legal_mask(game, player), dtype=np.float32, copy=True)
        legal = [int(a) for a in np.flatnonzero(mask > 0)]
        if not legal:
            raise ValueError("no legal action is available")
        root_key = self._key(game, player, mask)
        tree = {root_key: _Node()}
        budget = max(0, min(self.simulations, int(limits.node_limit)))
        deadline = start + max(0, limits.time_ms) / 1000
        belief = None
        if self.belief_agent is not None and budget and time.perf_counter() < deadline:
            belief = self.belief_agent.evaluate_aux(game, player)[3]
        completed = 0
        for _ in range(budget):
            check_cancel(cancel)
            if time.perf_counter() >= deadline:
                break
            work = self.sampler(game, player, self.rng, belief)
            if work is game:
                raise ValueError("belief sampler must return an owned game, not the root")
            path = []
            expanding = True
            value = None
            interrupted = False
            try:
                for depth in range(self.rollout_steps):
                    check_cancel(cancel)
                    if time.perf_counter() >= deadline:
                        interrupted = True
                        break
                    actions = {}
                    for actor in self.spec.actors(work):
                        support = np.array(self.spec.legal_mask(work, actor), dtype=np.float32, copy=True)
                        choices = [int(a) for a in np.flatnonzero(support > 0)]
                        if actor == player and expanding:
                            key = root_key if depth == 0 else self._key(work, player, support)
                            node = tree.setdefault(key, _Node())
                            for action in choices:
                                node.available[action] = node.available.get(action, 0) + 1
                            unseen = [action for action in choices if not node.visits.get(action, 0)]
                            if unseen:
                                action = int(self.rng.choice(unseen))
                                expanding = False
                            else:
                                action = max(choices, key=lambda a: node.totals[a] / node.visits[a]
                                             + self.exploration * math.sqrt(
                                                 math.log(node.available[a]) / node.visits[a]))
                            path.append((node, action))
                        else:
                            action = self._rollout_action(choices)
                        actions[actor] = action
                    outcome = self.spec.step(work, actions)
                    if outcome.truncated:
                        value = float(self.truncation_value)
                        break
                    if outcome.terminal:
                        value = float(self.spec.terminal_values(work)[player])
                        break
                if not interrupted:
                    if value is None:
                        value = self._leaf(work, player)
                    for node, action in path:
                        node.visits[action] = node.visits.get(action, 0) + 1
                        node.totals[action] = node.totals.get(action, 0.0) + value
                    completed += 1
            finally:
                work.close()
            if interrupted:
                break
        check_cancel(cancel)
        root = tree[root_key]
        action = max(legal, key=lambda a: (root.visits.get(a, 0),
                     root.totals.get(a, 0.0) / max(1, root.visits.get(a, 0))))
        value = root.totals.get(action, 0.0) / max(1, root.visits.get(action, 0))
        return Decision(int(action), int(round(value * 1000)), 0, completed,
                        (time.perf_counter() - start) * 1000, completed >= budget,
                        extra={"ismcts": {"visits": {str(a): root.visits.get(a, 0) for a in legal},
                                          "belief": "approximate"}})
