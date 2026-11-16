"""Deterministic generative checks and an optional independent scoring reference."""
import importlib.util
import random
import unittest

from riichi.agari import decompose
from riichi.scoring import score_hand
from riichi.shanten import shanten, waits
from riichi.yaku import WinContext

MAHJONG_AVAILABLE = importlib.util.find_spec('mahjong') is not None


def complete_hands(seed=184, count=300):
    """Sample constructed complete hands, rejecting physical multiplicity overflow."""
    rng = random.Random(seed)
    produced = 0
    while produced < count:
        hand = [0] * 34
        pair = rng.randrange(34)
        hand[pair] = 2
        for _ in range(4):
            if rng.random() < 0.6:
                low = 9 * rng.randrange(3) + rng.randrange(7)
                for tile in (low, low + 1, low + 2):
                    hand[tile] += 1
            else:
                hand[rng.randrange(34)] += 3
        if max(hand) > 4:
            continue
        winning = rng.choice([t for t, n in enumerate(hand) if n])
        produced += 1
        yield tuple(hand), winning


class GenerativeTests(unittest.TestCase):
    def test_three_hundred_complete_hands_and_waits(self):
        for hand, win in complete_hands():
            with self.subTest(hand=hand, win=win):
                self.assertTrue(decompose(hand))
                self.assertEqual(shanten(hand), -1)
                before = list(hand)
                before[win] -= 1
                self.assertEqual(shanten(before), 0)
                self.assertIn(win, waits(before))
                score = score_hand(hand, win, context=WinContext(tsumo=True))
                self.assertGreater(score.payment.total, 0)

    @unittest.skipUnless(MAHJONG_AVAILABLE, 'optional mahjong library is not installed')
    def test_three_hundred_scores_against_mahjong(self):
        from mahjong.hand_calculating.hand import HandCalculator
        from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules

        calculator = HandCalculator()
        options = OptionalRules(has_open_tanyao=True, has_aka_dora=False, has_double_yakuman=False)
        for index, (hand, win) in enumerate(complete_hands()):
            with self.subTest(index=index, hand=hand, win=win):
                dealer = index % 2 == 0
                tsumo = index % 3 != 0
                seat = 27 if dealer else 28
                context = WinContext(tsumo=tsumo, riichi=True, seat_wind=seat)
                ours = score_hand(hand, win, context=context)
                ids = [4 * tile + copy for tile, n in enumerate(hand) for copy in range(n)]
                config = HandConfig(is_tsumo=tsumo, is_riichi=True, player_wind=seat, round_wind=27, options=options)
                other = calculator.estimate_hand_value(ids, win * 4, config=config)
                self.assertIsNone(other.error)
                self.assertEqual(ours.han if not ours.yakuman else ours.yakuman * 13, other.han)
                self.assertEqual(ours.fu, other.fu)
                if tsumo:
                    expected = other.cost['main'] * 3 if dealer else other.cost['main'] + 2 * other.cost['additional']
                else:
                    expected = other.cost['main']
                self.assertEqual(ours.payment.total, expected)
