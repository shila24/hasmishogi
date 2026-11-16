"""Independent five-card reference and category boundary vectors."""

from collections import Counter
from itertools import combinations
import random
import unittest

from poker.evaluator import rank7


def cards(text):
    return ['23456789TJQKA'.index(c[0]) * 4 + 'cdhs'.index(c[1]) for c in text.split()]


def five_card_reference(hand):
    ranks = [c // 4 + 2 for c in hand]
    counts = Counter(ranks)
    groups = sorted(((n, r) for r, n in counts.items()), reverse=True)
    ordered = sorted(ranks, reverse=True)
    straight = (5 if sorted(ranks) == [2, 3, 4, 5, 14] else
                max(ranks) if len(counts) == 5 and max(ranks) - min(ranks) == 4 else 0)
    flush = len({c % 4 for c in hand}) == 1
    if straight and flush:
        result = (8, straight)
    elif groups[0][0] == 4:
        result = (7, groups[0][1], groups[1][1])
    elif [n for n, _ in groups] == [3, 2]:
        result = (6, groups[0][1], groups[1][1])
    elif flush:
        result = (5, *ordered)
    elif straight:
        result = (4, straight)
    elif groups[0][0] == 3:
        result = (3, *(r for _, r in groups))
    elif [n for n, _ in groups] == [2, 2, 1]:
        result = (2, *(r for _, r in groups))
    elif groups[0][0] == 2:
        result = (1, *(r for _, r in groups))
    else:
        result = (0, *ordered)
    padded = result + (0,) * (6 - len(result))
    return sum(value * 15 ** (5 - i) for i, value in enumerate(padded))


class EvaluatorTests(unittest.TestCase):
    def test_categories_and_wheel(self):
        examples = [
            'Ac Kd Jh 9s 7c 5d 2h', 'Ac Ad Kh Js 9c 5d 2h',
            'Ac Ad Kh Ks 9c 5d 2h', 'Ac Ad Ah Ks 9c 5d 2h',
            'Ac 2d 3h 4s 5c 9d Jh', 'Ac Jc 9c 7c 3c Kd 2h',
            'Ac Ad Ah Ks Kc 5d 2h', 'Ac Ad Ah As Kc 5d 2h',
            'Ac 2c 3c 4c 5c Kd 2h',
        ]
        results = [rank7(cards(text)) for text in examples]
        self.assertEqual([r // 15**5 for r in results], list(range(9)))
        self.assertEqual(results, sorted(results))
        self.assertLess(rank7(cards(examples[4])), rank7(cards('2c 3d 4h 5s 6c Ad Jh')))
        self.assertGreater(rank7(cards('Ac Kc Qc Jc Tc 2d 3h')), results[-1])

    def test_ties_kickers_two_triples_three_pairs_seven_flush_cards(self):
        fixtures = [
            'Ac Ad Ah Ks Kc Kh 2h', 'Ac Ad Kh Ks Qc Qd 2h',
            'Ac Kc Qc 9c 7c 5c 2c', 'Ac Ad Ah As Kc Kd Kh',
            'Ac 2c 3c 4c 5c 6c 7c', '2c 3d 4h 5s 6c 7d 8h',
        ]
        for text in fixtures:
            hand = cards(text)
            self.assertEqual(rank7(hand), max(map(five_card_reference, combinations(hand, 5))))
        self.assertEqual(rank7(cards('Ac Kd Qh Js Tc 3d 2h')),
                         rank7(cards('Ah Ks Qc Jd Th 4d 2s')))
        self.assertGreater(rank7(cards('Ac Ad Kh Qs Jc 5d 2h')),
                           rank7(cards('Ac Ad Kh Qs Tc 5d 2h')))

    def test_twenty_thousand_random_hands_against_best_of_twenty_one(self):
        rng = random.Random(360036)
        for _ in range(20000):
            hand = rng.sample(range(52), 7)
            self.assertEqual(rank7(hand), max(map(five_card_reference, combinations(hand, 5))), hand)

    def test_invalid_cards(self):
        for hand in ([0] * 7, list(range(6)), list(range(8)), [True, 1, 2, 3, 4, 5, 6],
                     [0, 1, 2, 3, 4, 5, 52], [0, 1, 2, 3, 4, 5, -1]):
            with self.assertRaises(ValueError):
                rank7(hand)


if __name__ == '__main__':
    unittest.main()
