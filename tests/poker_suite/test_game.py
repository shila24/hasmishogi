"""Betting, chance, hidden information and history contracts."""

import random
import unittest

from poker import create_game, replay


class GameTests(unittest.TestCase):
    def test_blinds_and_street_order(self):
        for button in (0, 1):
            game = create_game(seed=17, button=button)
            self.assertEqual(game.pot, 3)
            self.assertEqual(game.contributions[button], 1)
            self.assertEqual(game.actors(), [button])
            self.assertEqual(game.legal_actions(), [0, 1, 2])
            game.play(1)
            self.assertEqual(game.street, 0)
            self.assertEqual(game.legal_actions(), [1, 2])
            game.play(1)
            for street, board_size in ((1, 3), (2, 4), (3, 5)):
                self.assertEqual(game.street, street)
                self.assertEqual(game.actors(), [1 - button])
                self.assertEqual(len(game.observation(button)['board']), board_size)
                game.play(1)
                game.play(1)
            self.assertTrue(game.outcome.terminal)
            self.assertEqual(game.outcome.reason, 'showdown')
            self.assertEqual(sum(game.outcome.chips), 0)

    def test_caps_and_maximum(self):
        for variant, maximum in (('hulhe', 48), ('leduc', 13)):
            game = create_game({'variant': variant}, seed=3)
            while not game.outcome.terminal:
                if 2 in game.legal_actions():
                    game.play(2)
                else:
                    self.assertEqual(game.legal_actions(), [0, 1])
                    game.play(1)
            self.assertEqual(game.contributions, [maximum, maximum])

    def test_fold_and_atomic_rejection(self):
        game = create_game(seed=0)
        self.assertFalse(game.undo())
        for invalid in (True, -1, 3, 1.0, 'call', None):
            before = game.history_keys()
            with self.assertRaises(ValueError):
                game.play(invalid)
            self.assertEqual(game.history_keys(), before)
        game.play(0)
        self.assertEqual(game.outcome.chips, (-1, 1))
        self.assertEqual(game.outcome.reason, 'fold')
        self.assertEqual(game.actors(), [])
        with self.assertRaises(ValueError):
            game.play(1)
        self.assertTrue(game.undo())
        self.assertEqual(game.pot, 3)
        game.play(1)
        with self.assertRaises(ValueError):
            game.play(0)

    def test_blind_option_raises_bet_sizes_and_fold_payments(self):
        game = create_game(seed=0)
        game.play(1)
        game.play(2)
        self.assertEqual(game.contributions, [2, 4])
        self.assertEqual(game.actors(), [0])
        game.play(2)
        self.assertEqual(game.contributions, [6, 4])
        game.play(2)
        self.assertEqual(game.legal_actions(), [0, 1])
        game.play(1)
        self.assertEqual(game.contributions, [8, 8])
        self.assertEqual(game.actors(), [1])
        game.play(1)
        game.play(2)
        self.assertEqual(game.contributions, [10, 8])
        game.play(1)
        self.assertEqual(game.street, 2)
        game.play(2)
        self.assertEqual(game.contributions, [10, 14])
        game.play(0)
        self.assertEqual(game.outcome.chips, (-10, 10))
        self.assertEqual(game.outcome.winner, 1)

    def test_shared_board_split_and_hole_card_showdown(self):
        from tests.poker_suite.test_evaluator import cards
        for text, winner in (
                ('2c 3c 4d 5d Ac Kc Qc Jc Tc', -1),
                ('Ac Kc Ad Kd 2h 4s 7d 9c Jh', 0)):
            fixed = cards(text)
            game = create_game(seed=0, button=0)
            game._initialize(fixed + [c for c in range(52) if c not in fixed])
            while not game.outcome.terminal:
                game.play(2 if 2 in game.legal_actions() else 1)
            self.assertEqual(game.outcome.winner, winner)
            self.assertEqual(game.outcome.chips, (0, 0) if winner == -1 else (48, -48))
            before = game.history_keys()
            self.assertTrue(game.undo())
            self.assertFalse(game.outcome.terminal)
            game.play(1)
            self.assertEqual(game.history_keys(), before)

    def test_reproducibility_clone_undo_replay(self):
        rng = random.Random(72)
        for variant in ('hulhe', 'leduc'):
            for seed in range(100):
                game = create_game({'variant': variant}, seed=seed)
                initial = game.clone()
                while not game.outcome.terminal:
                    before = game.clone()
                    action = rng.choice(game.legal_actions())
                    game.play(action)
                    after = game.clone()
                    game.undo()
                    self.assertEqual(game.history_keys(), before.history_keys())
                    self.assertEqual(game.observation(0), before.observation(0))
                    game.play(action)
                    self.assertEqual(game.history_keys(), after.history_keys())
                rebuilt = replay(game.rules, seed, game.button, game.moves)
                self.assertEqual(game.history_keys(), rebuilt.history_keys())
                self.assertEqual(game.outcome, rebuilt.outcome)
                self.assertFalse(initial.outcome.terminal)
                self.assertEqual(initial.moves, [])


if __name__ == '__main__':
    unittest.main()
