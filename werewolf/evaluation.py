"""Duplicate seeded tables with the evaluated agent rotated through all roles."""

import math

from .baselines import BaselineAgent
from .game import create_game, team
from .actions import ROLES


def wilson_interval(wins, games, z=1.96):
    if games == 0:
        return (0.0, 1.0)
    rate, z2 = wins / games, z * z
    center = (rate + z2 / (2 * games)) / (1 + z2 / games)
    half = z * math.sqrt(rate * (1 - rate) / games + z2 / (4 * games * games)) / (1 + z2 / games)
    return (max(0.0, center - half), min(1.0, center + half))


def summarize(values):
    games = len(values)
    wins, losses = values.count(1), values.count(-1)
    draws = games - wins - losses
    low, high = wilson_interval(wins, games)
    score_low, score_high = wilson_interval(wins + 0.5 * draws, games)
    return {'games': games, 'wins': wins, 'losses': losses, 'draws': draws,
            'win_rate': wins / games if games else None,
            'mean_outcome_score': (wins + 0.5 * draws) / games if games else None,
            'ci_low': low, 'ci_high': high,
            'score_ci_low': score_low, 'score_ci_high': score_high}


def evaluate(agent, anchor='simple', games=100, seed=0, rules=None, limits=None):
    """Count the focal seat's team outcome once per table, split by its role/team.

    Five tables per deal rotate the focal agent through every seat (and hence
    every dealt role). Intervals are descriptive Wilson intervals; duplicate
    deals share chance and must not be treated as independent strength evidence.
    """
    if type(games) is not int or games < 5 or games % 5:
        raise ValueError('evaluation requires a positive multiple of five tables')
    if anchor not in ('random', 'simple'):
        raise ValueError('anchor must be random or simple')
    values, rows = [], []
    by_role = {role: [] for role in ROLES}
    by_team = {name: [] for name in ('village', 'werewolf')}
    by_seat = {seat: [] for seat in range(5)}
    for table in range(games):
        seat, deal_seed = table % 5, seed + table // 5
        game = create_game(rules, deal_seed)
        opponents = {p: BaselineAgent(anchor, deal_seed * 5 + p) for p in range(5)}
        role = game.roles[seat]
        try:
            while not game.outcome.terminal:
                actions = {}
                for p in game.actors():
                    chooser = agent if p == seat else opponents[p]
                    clone = game.clone()
                    try:
                        actions[p] = chooser.choose(clone, limits, player=p).action
                    finally:
                        clone.close()
                game.play_joint(actions)
            value = 0 if game.outcome.winner_team is None else (1 if game.outcome.winner_team == team(role) else -1)
            values.append(value)
            by_role[role].append(value)
            by_team[team(role)].append(value)
            by_seat[seat].append(value)
            rows.append({'seed': deal_seed, 'seat': seat, 'role': role,
                         'winner_team': game.outcome.winner_team, 'value': value})
        finally:
            game.close()
    return {'anchor': anchor, **summarize(values),
            'by_role': {key: summarize(v) for key, v in by_role.items()},
            'by_team': {key: summarize(v) for key, v in by_team.items()},
            'by_seat': {str(key): summarize(v) for key, v in by_seat.items()}, 'tables': rows,
            'interval_note': 'Descriptive Wilson intervals; duplicate tables are correlated.'}
