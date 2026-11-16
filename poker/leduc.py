"""Exact Leduc best responses constrained to the responder's information sets.

Policies are callables from an observation dictionary to action probabilities.
Exploitability is (BR_0 + BR_1) / 2 in chips/hand, not normalized rewards.
All 120 equally likely ordered physical-card deals are enumerated. The button
defaults to seat zero; specifying button=1 evaluates the other seating as well.
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
import math

from .game import create_game


def information_key(obs):
    return (obs['player'], tuple(obs['hole_cards']), tuple(obs['board']), obs['button'],
            tuple(tuple((e['player'], e['action']) for e in street) for street in obs['history']))


def probabilities(policy, obs):
    result = policy(obs)
    legal = obs['legal_actions']
    if (not isinstance(result, dict) or set(result) - set(legal)
            or any(type(a) is not int for a in result)):
        raise ValueError('policy must map legal action ids to probabilities')
    try:
        row = {a: float(result.get(a, 0)) for a in legal}
    except (TypeError, ValueError) as exc:
        raise ValueError('invalid policy probabilities') from exc
    if (any(not math.isfinite(v) or v < 0 for v in row.values())
            or not math.isclose(sum(row.values()), 1.0, abs_tol=1e-9, rel_tol=0)):
        raise ValueError('policy probabilities must be finite, nonnegative and sum to one')
    return row


@dataclass
class _Node:
    player: int
    key: tuple | None
    children: dict
    depth: int
    payoff: float = 0.0


@lru_cache(maxsize=2)
def _tree(button):
    nodes, roots, observations = [], [], {}

    def visit(game, depth):
        index = len(nodes)
        terminal = game.outcome.terminal
        obs = None if terminal else game.observation(game.actors()[0])
        key = None if terminal else information_key(obs)
        node = _Node(-1 if terminal else obs['player'], key, {}, depth,
                     game.outcome.chips[0] if terminal else 0.0)
        nodes.append(node)
        if not terminal:
            observations.setdefault(key, obs)
            for action in game.legal_actions():
                game.play(action)
                node.children[action] = visit(game, depth + 1)
                game.undo()
        return index

    for deal in permutations(range(6), 3):
        game = create_game({'variant': 'leduc'}, seed=0, button=button)
        game._initialize(deal + tuple(c for c in range(6) if c not in deal))
        roots.append(visit(game, 0))
    levels = []
    for depth in range(max(n.depth for n in nodes) + 1):
        groups = {}
        for i, node in enumerate(nodes):
            if node.depth == depth:
                groups.setdefault(node.key, []).append(i)
        levels.append(groups)
    return nodes, roots, observations, levels


def leduc_exploitability(policy, *, button=0):
    """Return the exact full-tree exploitability of an observation-only policy.

    No simulation, sampled determinization, or per-deal maximization is used.
    One action is chosen for every node in a responder information set, weighted
    by chance and opponent reach only (including off-policy responder branches).
    """
    if type(button) is not int or button not in (0, 1):
        raise ValueError('button must be 0 or 1')
    nodes, roots, observations, levels = _tree(button)
    # Give user policy code detached observations, never the cached tree's dictionaries.
    import copy
    strategy = {key: probabilities(policy, copy.deepcopy(obs)) for key, obs in observations.items()}
    responses = []
    for player in (0, 1):
        reach = [0.0] * len(nodes)
        for root in roots:
            reach[root] = 1 / len(roots)
        for i, node in enumerate(nodes):
            for action, child in node.children.items():
                reach[child] = reach[i] * (1 if node.player == player else strategy[node.key][action])
        values = [n.payoff * (1 if player == 0 else -1) for n in nodes]
        for groups in reversed(levels):
            for key, indices in groups.items():
                if key is None:
                    continue
                if nodes[indices[0]].player == player:
                    legal = nodes[indices[0]].children
                    action = max(legal, key=lambda a: sum(reach[i] * values[nodes[i].children[a]]
                                                         for i in indices))
                    for i in indices:
                        values[i] = values[nodes[i].children[action]]
                else:
                    for i in indices:
                        values[i] = sum(strategy[key][a] * values[child]
                                        for a, child in nodes[i].children.items())
        responses.append(sum(values[i] for i in roots) / len(roots))
    return sum(responses) / 2
