"""Hand-end revelations (the supervision for belief heads) and the current-hand move index."""

import random
import unittest

from riichi.actions import PASS, RON, TSUMO
from riichi.game import RiichiGame
from riichi.tiles import to_counts
from tests.riichi.test_game import passes, rig

REVEAL_KEYS = ('reason', 'winners', 'scores', 'tenpai', 'waits', 'hand_counts', 'hands', 'melds',
               'riichi', 'placement', 'winning_tile', 'discarder', 'hand_number', 'dora_indicators')


def check_hand_summary(test, summary):
    """Structural invariants every hand summary must satisfy, whatever ended the hand."""
    for key in REVEAL_KEYS:
        test.assertIn(key, summary)
    for p in range(4):
        test.assertEqual(len(summary['waits'][p]), 34)
        test.assertEqual(len(summary['hand_counts'][p]), 34)
        test.assertTrue(all(w in (0, 1) for w in summary['waits'][p]))
        test.assertEqual(summary['tenpai'][p], any(summary['waits'][p]))
        test.assertEqual(list(to_counts(summary['hands'][p])), summary['hand_counts'][p])
        test.assertIn(len(summary['hands'][p]) + 3 * len(summary['melds'][p]), (13, 14))
        test.assertEqual(summary['hands'][p], sorted(summary['hands'][p]))
    order = sorted(range(4), key=lambda p: (-summary['scores'][p], p))
    test.assertEqual(summary['placement'], [order.index(p) + 1 for p in range(4)])
    test.assertEqual(sorted(summary['placement']), [1, 2, 3, 4])
    reason = summary['reason']
    if reason in ('ron', 'chankan'):
        test.assertIsInstance(summary['discarder'], int)
        test.assertNotIn(summary['discarder'], summary['winners'])
    else:
        test.assertIsNone(summary['discarder'])
    if reason in ('ron', 'chankan', 'tsumo'):
        tile = summary['winning_tile']
        test.assertIsInstance(tile, int)
        for winner in summary['winners']:
            test.assertTrue(summary['tenpai'][winner])
            test.assertEqual(summary['waits'][winner][tile // 4], 1)
    else:
        test.assertIsNone(summary['winning_tile'])


class HandEndInfoTests(unittest.TestCase):
    def test_tsumo_reveals_winner_wait_placement_and_hand_start(self):
        game = rig(['123m123p123s789s55z', None, None, None])
        drawn = game.drawn_tiles[0]
        game.play(TSUMO)
        summary = game.last_hand
        check_hand_summary(self, summary)
        self.assertEqual((summary['reason'], summary['winners']), ('tsumo', [0]))
        self.assertEqual(summary['winning_tile'], drawn)
        self.assertIsNone(summary['discarder'])
        self.assertEqual(summary['hand_number'], 1)
        self.assertEqual(sum(summary['hand_counts'][0]), 14)
        self.assertEqual(summary['placement'][0], 1)
        self.assertEqual(summary['scores'], game.scores)
        self.assertEqual(summary['riichi'], [False] * 4)
        # The next hand starts after the finishing joint step.
        self.assertFalse(game.outcome.terminal)
        self.assertEqual(game.hand_start, len(game.moves))
        self.assertEqual(game.hand_start, 1)

    def test_ron_reveals_discarder_and_every_hidden_hand(self):
        game = rig(['3m123p456p789s1122z', '12m123p123s789s55z', '33m456p789s12344z', None])
        hidden_before = [sorted(hand) for hand in game.hands]
        game.play(2)
        actions = passes(game)
        actions[1] = RON
        game.play_joint(actions)
        summary = game.last_hand
        check_hand_summary(self, summary)
        self.assertEqual((summary['reason'], summary['winners'], summary['discarder']), ('ron', [1], 0))
        self.assertEqual(summary['winning_tile'] // 4, 2)
        self.assertEqual(summary['waits'][1][2], 1)
        self.assertTrue(summary['tenpai'][1])
        self.assertFalse(summary['tenpai'][2])
        self.assertEqual(summary['hands'][1], hidden_before[1])
        self.assertEqual(len(summary['hands'][0]), 13)
        self.assertEqual(sum(summary['hand_counts'][0]), 13)

    def test_exhaustive_draw_tenpai_matches_payments(self):
        game = rig([None, '123m123p123s789s5z', None, None])
        game.play(game.drawn_tiles[0] // 4)
        game.live_wall.clear()
        game._legal_cache.clear()
        game.play_joint(passes(game))
        summary = game.last_hand
        check_hand_summary(self, summary)
        self.assertEqual(summary['reason'], 'exhaustive_draw')
        self.assertEqual(summary['tenpai'], [False, True, False, False])
        self.assertEqual(summary['waits'][1][31], 1)
        self.assertEqual(game.scores, [24000, 28000, 24000, 24000])
        self.assertEqual(game.hand_start, len(game.moves))

    def test_terminal_summary_uses_final_standings(self):
        game = rig(['123m123p123s789s55z', None, None, None])
        game.scores = [49900, 100, 25000, 25000]
        game.riichi_sticks = 2
        game.play(TSUMO)
        self.assertTrue(game.outcome.terminal)
        summary = game.last_hand
        check_hand_summary(self, summary)
        self.assertEqual(summary['scores'], game.scores)
        self.assertEqual(summary['placement'], [game.ranking().index(p) + 1 for p in range(4)])
        self.assertEqual(summary['placement'][0], 1)
        self.assertEqual(game.riichi_sticks, 0)

    def test_random_hanchan_summaries_are_consistent_and_hand_start_tracks_moves(self):
        for seed in range(3):
            game = RiichiGame(seed=seed)
            rng = random.Random(seed + 500)
            summaries = 0
            while not game.outcome.terminal:
                hand = game.hand_number
                self.assertLessEqual(game.hand_start, len(game.moves))
                game.play_joint({p: rng.choice(game.legal_actions(p)) for p in game.actors()})
                if game.hand_number != hand or game.outcome.terminal:
                    summaries += 1
                    check_hand_summary(self, game.last_hand)
                    self.assertEqual(game.last_hand['hand_number'], hand)
                    if not game.outcome.terminal:
                        self.assertEqual(game.hand_start, len(game.moves))
            self.assertGreater(summaries, 0)
            replayed = RiichiGame.replay(game.seed, game.history, game.rules)
            self.assertEqual(replayed.hand_start, game.hand_start)
            self.assertEqual(replayed.last_hand, game.last_hand)
            self.assertEqual(game.clone().hand_start, game.hand_start)


if __name__ == '__main__':
    unittest.main()
