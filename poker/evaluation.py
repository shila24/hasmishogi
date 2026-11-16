"""Paired duplicate-deal head-to-head evaluation in milli-big-blinds per hand."""

import math
import random
import statistics

from .game import create_game
from .leduc import probabilities


def duplicate_games(seed, rules=None):
    """Two hands with the same seat-owned cards and board, opposite buttons.

    Policies exchange seats in the second hand. Keeping physical hands at their
    absolute seats is essential: merely flipping the button with the same shuffle
    would also exchange the holes and give each policy its original cards again.
    The second world's explicit permutation has no seed-only replay identity.
    """
    first = create_game(rules, seed)
    second = create_game(rules, seed, button=1 - first.button)
    deck = list(first._deck)
    for i in range(0, 2 * first.rules.hole_count, 2):
        deck[i], deck[i + 1] = deck[i + 1], deck[i]
    second._initialize(deck)
    second.seed = None
    return first, second


def summarize_duplicates(pairs):
    """Use seed pairs as the independent units of an asymptotic 95% interval.

    Each row contains policy A's chip winnings in the two hands; one big blind
    is two chips. With one pair, an interval is unavailable (None).
    """
    pairs = list(pairs)
    if not pairs or any(len(row) != 2 or not all(math.isfinite(x) for x in row) for row in pairs):
        raise ValueError('need finite two-hand chip results')
    samples = [sum(row) / 2 * 500 for row in pairs]
    mean = statistics.mean(samples)
    margin = 1.96 * statistics.stdev(samples) / math.sqrt(len(samples)) if len(samples) > 1 else None
    return dict(hands=2 * len(pairs), pairs=len(pairs), mbb_per_hand=mean,
                ci95=None if margin is None else [mean - margin, mean + margin],
                interval_method='normal approximation over independent duplicate pairs',
                pair_chips=[list(row) for row in pairs])


def evaluate_duplicate(policy_a, policy_b, seeds, rules=None):
    """Play observation-only policies on duplicate pairs; never truncate a hand."""
    pairs = []
    seeds = list(seeds)
    for seed in seeds:
        games = duplicate_games(seed, rules)
        chips = []
        try:
            for rotation, game in enumerate(games):
                rng = random.Random(seed * 2 + rotation)
                policies = (policy_a, policy_b) if rotation == 0 else (policy_b, policy_a)
                while not game.outcome.terminal:
                    player = game.actors()[0]
                    row = probabilities(policies[player], game.observation(player))
                    action = rng.choices(list(row), weights=list(row.values()))[0]
                    game.play(action)
                chips.append(game.outcome.chips[rotation])
        finally:
            for game in games:
                game.close()
        pairs.append(chips)
    report = summarize_duplicates(pairs)
    report.update(seeds=seeds, rules=games[0].rules.to_dict())
    return report
