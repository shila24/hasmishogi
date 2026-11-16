"""Leduc rule vectors, exact exploitability and duplicate-pair arithmetic."""

from fractions import Fraction
import math
import unittest

from poker import create_game, leduc_exploitability
from poker.baselines import BaselinePolicy, equity
from poker.evaluation import duplicate_games, evaluate_duplicate, summarize_duplicates
from tests.poker_suite.leduc_oracle import best_response


class LeducTests(unittest.TestCase):
    def test_antes_action_order_pair_high_card_and_tie(self):
        for deal, winner in (((0, 5, 1), 0), ((0, 5, 3), 1), ((2, 3, 0), -1)):
            game = create_game({'variant': 'leduc'}, seed=0, button=0)
            game._initialize(deal + tuple(c for c in range(6) if c not in deal))
            self.assertEqual(game.contributions, [1, 1])
            self.assertEqual(game.legal_actions(), [1, 2])
            game.play(1)
            game.play(1)
            self.assertEqual(game.actors(), [0])
            self.assertEqual(game.observation(0)['board'], [deal[2]])
            game.play(1)
            game.play(1)
            self.assertEqual(game.outcome.winner, winner)
            self.assertEqual(game.outcome.chips, (0, 0) if winner == -1 else
                             (1, -1) if winner == 0 else (-1, 1))

    def test_exact_responses_against_independent_fraction_oracle(self):
        for name, uniform, expected in (('call', False, Fraction(22, 15)),
                                         ('random', True, Fraction(1709, 720))):
            with self.subTest(policy=name):
                reference = (best_response(0, uniform) + best_response(1, uniform)) / 2
                self.assertEqual(reference, expected)
                self.assertAlmostEqual(leduc_exploitability(BaselinePolicy(name)), float(reference), places=12)
                self.assertAlmostEqual(leduc_exploitability(BaselinePolicy(name), button=1), float(reference), places=12)

    def test_policy_observes_only_its_information_set(self):
        seen = {}

        def policy(obs):
            self.assertEqual(len(obs['hole_cards']), 1)
            self.assertNotIn('seed', obs)
            self.assertNotIn('deck', obs)
            key = (obs['player'], tuple(obs['hole_cards']), tuple(obs['board']), str(obs['history']))
            if key in seen:
                self.assertEqual(seen[key], obs)
            seen[key] = obs.copy()
            return {1: 1.0}

        self.assertAlmostEqual(leduc_exploitability(policy), 22 / 15)
        self.assertGreater(len(seen), 100)
        for row in ({1: 0.5}, {1: float('nan')}, {3: 1}, {1: -1, 2: 2}, {True: 1}):
            with self.assertRaises(ValueError):
                leduc_exploitability(lambda obs: row)

    def test_equity_and_baseline_legality(self):
        self.assertEqual(equity('leduc', (0,), (1,)), 1)
        self.assertEqual(equity('leduc', (0,), (4,)), 0.125)
        for name in ('random', 'call', 'raise', 'tight:0', 'tight:0.6', 'tight:1'):
            policy = BaselinePolicy(name)
            for variant in ('leduc', 'hulhe'):
                game = create_game({'variant': variant}, seed=42)
                while not game.outcome.terminal:
                    obs = game.observation(game.actors()[0])
                    row = policy(obs)
                    self.assertEqual(set(row), set(game.legal_actions()))
                    self.assertAlmostEqual(sum(row.values()), 1)
                    game.play(max(row, key=row.get))
        for bad in ('tight:nan', 'tight:-0.1', 'tight:1.1', 'tight:x', 'expert'):
            with self.assertRaises(ValueError):
                BaselinePolicy(bad)


class DuplicateTests(unittest.TestCase):
    def test_pairs_swap_button_preserve_absolute_cards_and_board(self):
        for seed in range(20):
            a, b = duplicate_games(seed)
            self.assertNotEqual(a.button, b.button)
            for player in (0, 1):
                self.assertEqual(a.observation(player)['hole_cards'], b.observation(player)['hole_cards'])
            for game in (a, b):
                while not game.outcome.terminal:
                    game.play(1)
            self.assertEqual(a.observation(0)['board'], b.observation(0)['board'])

    def test_arithmetic_uses_independent_pairs_and_two_chip_big_blind(self):
        report = summarize_duplicates([(2, 2), (-2, -2)])
        self.assertEqual(report['hands'], 4)
        self.assertEqual(report['mbb_per_hand'], 0)
        self.assertAlmostEqual(report['ci95'][1], 1960)
        self.assertEqual(summarize_duplicates([(2, -2), (4, 0)])['mbb_per_hand'], 500)
        self.assertIsNone(summarize_duplicates([(2, -2)])['ci95'])
        call = BaselinePolicy('call')
        report = evaluate_duplicate(call, call, range(20))
        self.assertEqual(report['mbb_per_hand'], 0)
        self.assertEqual(report['ci95'], [0, 0])
        for pairs in ([], [(math.inf, 0)], [(0,)]):
            with self.assertRaises(ValueError):
                summarize_duplicates(pairs)


if __name__ == '__main__':
    unittest.main()
