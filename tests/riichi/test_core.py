"""Tile, distance, decomposition, and rule contract tests."""
import random
import time
import unittest

from riichi.tiles import parse_tiles, format_tiles, parse_counts, to_counts, build_wall
from riichi.rules import RiichiRules
from riichi.shanten import shanten, kokushi_shanten, chiitoitsu_shanten, waits, is_tenpai, warm_tables, _standard, _bounded_standard
from riichi.agari import decompose, Meld


class CoreTests(unittest.TestCase):
    def test_tile_roundtrips(self):
        for code in ('123m4p0s7z', '0555m123p777z', '19m19p19s1234567z', ''):
            tiles = parse_tiles(code)
            self.assertEqual(to_counts(tiles), parse_counts(code))
            self.assertEqual(set(parse_tiles(format_tiles(tiles))), set(tiles))
        self.assertEqual(parse_tiles('0m0p0s'), (16, 52, 88))
        for code in ('0z', '8z', '10z', '123', 'm', '11111m', '00p', '5555s', '1x'):
            with self.subTest(code=code), self.assertRaises(ValueError):
                parse_tiles(code)

    def test_wall(self):
        a = build_wall(seed=71)
        self.assertEqual(a, build_wall(seed=71))
        self.assertNotEqual(a, build_wall(seed=72))
        self.assertEqual(len(a.live), 122)
        self.assertEqual(len(a.dead), 14)
        self.assertEqual(set(a.live + a.dead), set(range(136)))
        self.assertEqual(a.rinshan, a.dead[:4])
        self.assertEqual(a.dora_indicators, a.dead[4::2])
        self.assertEqual(a.ura_indicators, a.dead[5::2])
        self.assertEqual(a, build_wall(rng=random.Random(71)))

    def test_rules_roundtrip(self):
        rules = RiichiRules()
        self.assertEqual(rules.ruleset_id, 'riichi-4p-v1')
        self.assertEqual(RiichiRules.from_json(rules.to_json()), rules)
        self.assertFalse(rules.double_yakuman)
        self.assertTrue(rules.atozuke)
        with self.assertRaises(ValueError):
            RiichiRules.from_json('{"kuitan": "false"}')

    def test_shanten_anchors(self):
        self.assertEqual(kokushi_shanten((0,) * 34), 13)
        self.assertEqual(kokushi_shanten(parse_counts('19m19p19s1234567z')), 0)
        self.assertEqual(shanten(parse_counts('19m19p19s11234567z')), -1)
        self.assertEqual(chiitoitsu_shanten(parse_counts('1122m3344p5566s7z')), 0)
        self.assertEqual(chiitoitsu_shanten(parse_counts('111122m3344p55s6z')), 2)
        self.assertEqual(shanten(parse_counts('123m456p789s1122z')), 0)
        self.assertEqual(waits(parse_counts('123m456p789s1122z')), (27, 28))
        self.assertTrue(is_tenpai(parse_counts('123m456p789s1122z')))
        meld = Meld((0, 1, 2), 'chi')
        self.assertEqual(shanten(parse_counts('456p789s1122z'), (meld,)), 0)
        self.assertEqual(waits(parse_counts('456p789s1122z'), (meld,)), (27, 28))

    def test_decompositions(self):
        self.assertEqual(len(decompose(parse_counts('11223344556677m'))), 4)
        self.assertEqual(len(decompose(parse_counts('19m19p19s11234567z'))), 1)
        self.assertEqual(decompose(parse_counts('123m456p789s11223z')), ())
        meld = Meld((0, 0, 0, 0), 'kan', False)
        self.assertEqual(len(decompose(parse_counts('456p789s11222z'), (meld,))), 1)
        self.assertEqual(decompose(parse_counts('123m456p789s1122z')), ())

    def test_warm_performance(self):
        hands = [to_counts(random.Random(i).sample(range(136), 14)) for i in range(300)]
        warm_tables(hands)
        _standard.cache_clear()
        _bounded_standard.cache_clear()
        start = time.perf_counter()
        for hand in hands:
            shanten(hand)
        elapsed = (time.perf_counter() - start) / len(hands)
        self.assertLess(elapsed, 0.0002)

    def test_quad_cannot_wait_on_fifth_copy(self):
        for code in ('1111m222p333s444z', '1111z222m333p444s'):
            hand = parse_counts(code)
            self.assertEqual(waits(hand), ())
            self.assertEqual(shanten(hand), 1)
        called = (Meld((6, 6, 6, 6), 'kan'),)
        hand = parse_counts('89m123p456s11z')
        self.assertEqual(waits(hand, called), ())
        self.assertEqual(shanten(hand, called), 1)

    def test_invalid_inputs_and_serialization(self):
        from riichi.tiles import validate_counts, validate_ids, format_counts, dora_type
        for value in ((0,) * 33, (True,) * 34, (-1,) * 34, (5,) * 34):
            with self.assertRaises(ValueError):
                validate_counts(value)
        for value in ((136,), (-1,), (True,), (0, 0)):
            with self.assertRaises(ValueError):
                validate_ids(value)
        for code in ('1111m', '123405m678p77z', ''):
            self.assertEqual(parse_counts(format_counts(parse_counts(code))), parse_counts(code))
        for indicator, expected in ((8,0), (17,9), (26,18), (30,27), (33,31)):
            self.assertEqual(dora_type(indicator), expected)
        self.assertEqual(len(parse_tiles('5555m', red_fives=False)), 4)
        with self.assertRaises(ValueError):
            parse_tiles('0m', red_fives=False)
        for kwargs in ({'uma': (20,10,0,-20)}, {'honba':100}, {'renhou':'invalid'}, {'double_wind_pair_fu':3},
                       {'start_points':0}, {'return_points':20000}, {'riichi_stick': True}, {'uma':(10,20,-10,-20)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RiichiRules(**kwargs)
        for value in ('[]', '{"unknown": 1}'):
            with self.assertRaises(ValueError):
                RiichiRules.from_json(value)
        for args in (((0,1,3),'chi'), ((0,0,1),'pon'), ((0,0,0),'kan'), ((33,34,35),'chi'), ((0,),'invalid')):
            with self.subTest(args=args), self.assertRaises(ValueError):
                Meld(*args)
        with self.assertRaises(ValueError):
            build_wall(1, rng=random.Random(2))
        with self.assertRaises(ValueError):
            decompose(parse_counts('1111m222p333s444z'), (Meld((0,0,0),'pon'),))
        with self.assertRaises(ValueError):
            decompose(parse_counts('123m456p789s11222z'), winning_tile=33)
        with self.assertRaises(ValueError):
            waits(parse_counts('123m'))


class ReviewRegressionTests(unittest.TestCase):
    """Regressions for the riichi rules review findings."""

    def test_decompositions_are_unique_partitions(self):
        from riichi.agari import decompose
        from riichi.tiles import parse_counts

        for code in ("111123m456p789s22z", "111123m123p123s55z"):
            plain = decompose(parse_counts(code))
            partitions = {tuple(sorted(g.tiles for g in d.groups)) for d in plain if d.kind == 'standard'}
            self.assertEqual(len([d for d in plain if d.kind == 'standard']), len(partitions), code)
            with_win = [d for d in decompose(parse_counts(code), winning_tile=0) if d.kind == 'standard']
            self.assertEqual(len(with_win), 2, code)  # triplet or sequence completed by the winning tile

    def test_renhou_mangan_floor_for_bonus_only_hands(self):
        from riichi.rules import RiichiRules
        from riichi.scoring import score_hand
        from riichi.tiles import parse_tiles, to_counts
        from riichi.yaku import WinContext

        ids = parse_tiles('123055m406p406s11z')
        context = WinContext(renhou=True, seat_wind=29, dora_indicators=(3,))
        score = score_hand(to_counts(ids), 2, context=context, rules=RiichiRules(renhou='mangan'), tile_ids=ids)
        names = [y.name for y in score.yaku]
        self.assertEqual(names, ['renhou'])
        self.assertEqual(sum(y.han for y in score.yaku), 5)

    def test_meld_capacities_apply_without_a_four_count(self):
        from riichi.agari import Meld
        from riichi.shanten import shanten, standard_shanten
        from riichi.tiles import parse_counts

        hand = parse_counts('6889m112z')
        melds = (Meld((6, 6, 6), 'pon'), Meld((9, 9, 9), 'pon'))
        self.assertEqual(standard_shanten(hand, melds), 2)
        self.assertEqual(shanten(hand, melds), 2)
