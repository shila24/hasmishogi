"""Observation-only baselines and deterministic equity estimates."""

from functools import lru_cache
from itertools import permutations
import math
import random

from .evaluator import rank7


@lru_cache(maxsize=8192)
def equity(variant, hole, board):
    """Equity against a uniform random hand: exact Leduc, 256 fixed Hold'em samples.

    Only visible cards seed the estimator; no engine seed or opponent state is
    consulted. This is a baseline heuristic, not an expert strategy.
    """
    unseen = [c for c in range(52 if variant == 'hulhe' else 6) if c not in hole + board]
    if variant == 'leduc':
        cases = ((c, board[0]) for c in unseen) if board else permutations(unseen, 2)
        scores = []
        for opponent, public in cases:
            own = (hole[0] // 2 == public // 2, hole[0] // 2)
            other = (opponent // 2 == public // 2, opponent // 2)
            scores.append(1.0 if own > other else 0.5 if own == other else 0.0)
        return sum(scores) / len(scores)
    seed = 0
    for card in hole + (-1,) + board:
        seed = seed * 54 + card + 2
    rng = random.Random(seed)
    total = 0.0
    for _ in range(256):
        draw = rng.sample(unseen, 7 - len(board))
        public = board + tuple(draw[2:])
        own, other = rank7(hole + public), rank7(tuple(draw[:2]) + public)
        total += 1 if own > other else 0.5 if own == other else 0
    return total / 256


class BaselinePolicy:
    """Callable observation -> legal action probabilities, also usable by exact evaluation."""

    def __init__(self, name='random'):
        self.name = name
        self.threshold = None
        if name.startswith('tight:'):
            try:
                self.threshold = float(name.split(':', 1)[1])
            except ValueError as exc:
                raise ValueError('tight threshold must be a number in [0, 1]') from exc
            if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
                raise ValueError('tight threshold must be in [0, 1]')
        elif name not in ('random', 'call', 'raise'):
            raise ValueError(f'unknown poker baseline: {name}')

    def __call__(self, observation):
        legal = observation['legal_actions']
        if not legal:
            raise ValueError('baseline requires an active player')
        if self.name == 'random':
            return {a: 1 / len(legal) for a in legal}
        action = 1
        if self.name == 'raise':
            action = 2 if 2 in legal else 1
        elif self.threshold is not None:
            # The specified tight baseline uses pre-flop equity on every street.
            strength = equity(observation['variant'], tuple(sorted(observation['hole_cards'])),
                              ())
            if strength > self.threshold:
                action = 2 if 2 in legal else 1
            elif 0 in legal and observation['bets'] > (1 if observation['variant'] == 'hulhe'
                                                      and observation['street'] == 0 else 0):
                action = 0
        return {a: float(a == action) for a in legal}
