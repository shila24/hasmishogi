"""Named yaku table with open/closed variants and negative boundary cases."""
from dataclasses import replace
import unittest

from riichi.agari import Meld, decompose, hand_context
from riichi.rules import RiichiRules
from riichi.scoring import NoYakuError, score_hand
from riichi.tiles import parse_counts, parse_tiles, to_counts
from riichi.yaku import WinContext, detect_yaku, nagashi_eligible


def meld(code, kind='pon', opened=True):
    counts = parse_counts(code)
    return Meld(tuple(t for t, n in enumerate(counts) for _ in range(n)), kind, opened)


# Expected han refers to the named yaku, not the total hand value.
CASES = [
    ('riichi', '123456m23455p678s', '1m', {}, (), 1),
    ('double_riichi', '123456m23455p678s', '1m', {'double_riichi': True}, (), 2),
    ('ippatsu', '123456m23455p678s', '1m', {'ippatsu': True}, (), 1),
    ('menzen_tsumo', '123456m23455p678s', '1m', {'tsumo': True}, (), 1),
    ('pinfu', '123456m23455p678s', '1m', {}, (), 1),
    ('tanyao', '234456m23455p678s', '2m', {}, (), 1),
    ('tanyao', '456m23455p678s', '4m', {'riichi': False}, (meld('234m', 'chi'),), 1),
    ('iipeikou', '112233m456p678s55z', '1m', {}, (), 1),
    ('ryanpeikou', '112233m445566p77s', '1m', {}, (), 3),
    ('sanshoku_doujun', '123m123p123789s55z', '1m', {}, (), 2),
    ('sanshoku_doujun', '123p123789s55z', '1p', {'riichi': False}, (meld('123m', 'chi'),), 1),
    ('ittsuu', '123456789m234p55z', '1m', {}, (), 2),
    ('ittsuu', '456789m234p55z', '4m', {'riichi': False}, (meld('123m', 'chi'),), 1),
    ('chanta', '123789m111p789s55z', '1m', {}, (), 2),
    ('chanta', '789m111p789s55z', '7m', {'riichi': False}, (meld('123m', 'chi'),), 1),
    ('junchan', '123789m111p789s99p', '1m', {}, (), 3),
    ('junchan', '789m111p789s99p', '7m', {'riichi': False}, (meld('123m', 'chi'),), 2),
    ('honroutou', '111999m111p999s55z', '1m', {}, (), 2),
    ('honroutou', '1199m1199p1199s55z', '1m', {}, (), 2),
    ('toitoi', '111m222p333444s55z', '4s', {}, (), 2),
    ('sanankou', '111m222p333456s55z', '6s', {}, (), 2),
    ('sankantsu', '678s55z', '6s', {'riichi': False}, (meld('1111m','kan'), meld('2222p','kan'), meld('3333s','kan')), 2),
    ('shousangen', '123m456p55566677z', '1m', {}, (), 2),
    ('honitsu', '123456789m11122z', '1m', {}, (), 3),
    ('honitsu', '456789m11122z', '4m', {'riichi': False}, (meld('123m','chi'),), 2),
    ('chinitsu', '12334556778922m', '1m', {}, (), 6),
    ('chinitsu', '34556778922m', '3m', {'riichi': False}, (meld('123m','chi'),), 5),
    ('chiitoitsu', '11m22p334455s66z77z', '1m', {}, (), 2),
    ('haku', '123456m789p55566z', '1m', {}, (), 1),
    ('hatsu', '123456m789p66677z', '1m', {}, (), 1),
    ('chun', '123456m789p77755z', '1m', {}, (), 1),
    ('seat_wind', '123456m789p22255z', '1m', {}, (), 1),
    ('round_wind', '123456m789p11155z', '1m', {}, (), 1),
    ('haitei', '123456m23455p678s', '1m', {'tsumo': True, 'haitei': True}, (), 1),
    ('houtei', '123456m23455p678s', '1m', {'houtei': True}, (), 1),
    ('rinshan', '123m456p789s55z', '1m', {'tsumo': True}, (meld('2222z','kan',False),), 1),
    ('chankan', '123456m23455p678s', '1m', {'chankan': True}, (), 1),
    ('sanshoku_doukou', '111m111p111789s55z', '1m', {}, (), 2),
    ('tenhou', '123456m23455p678s', '1m', {'riichi': False, 'tsumo': True, 'tenhou': True, 'seat_wind': 27}, (), 0),
    ('chiihou', '123456m23455p678s', '1m', {'riichi': False, 'tsumo': True, 'chiihou': True}, (), 0),
    ('kokushi', '19m19p19s11234567z', '9m', {}, (), 0),
    ('kokushi_13', '19m19p19s11234567z', '1z', {}, (), 0),
    ('suuankou', '111m222p333444s55z', '4s', {'tsumo': True}, (), 0),
    ('suuankou_tanki', '111m222p333444s55z', '5z', {}, (), 0),
    ('daisangen', '123m55p555666777z', '1m', {}, (), 0),
    ('shousuushii', '123m11122233344z', '1m', {}, (), 0),
    ('daisuushii', '55m111222333444z', '1z', {}, (), 0),
    ('tsuuiisou', '11122233355566z', '1z', {}, (), 0),
    ('tsuuiisou', '11223344556677z', '1z', {}, (), 0),
    ('chinroutou', '111999m111p999s99p', '1m', {}, (), 0),
    ('ryuuiisou', '22233344466688s', '2s', {}, (), 0),
    ('suukantsu', '55z', '5z', {'riichi': False}, (meld('1111m','kan'), meld('2222p','kan'), meld('3333s','kan'), meld('4444s','kan')), 0),
    ('chuuren', '11123455678999m', '1m', {}, (), 0),
    ('junsei_chuuren', '11123455678999m', '5m', {}, (), 0),
]


class YakuTests(unittest.TestCase):
    def test_all_named_yaku(self):
        self.assertGreaterEqual(len(CASES), 40)
        for name, code, win, options, melds, han in CASES:
            with self.subTest(name=name, hand=code, open=bool(melds)):
                options = {'riichi': True, **options}
                if name == 'rinshan':
                    options['rinshan'] = True
                context = WinContext(**options)
                counts, melds, total = hand_context(parse_counts(code), melds)
                tile = next(t for t, n in enumerate(parse_counts(win)) if n)
                choices = [detect_yaku(d, total, context) for d in decompose(counts, melds, tile)]
                found = [y for ys in choices for y in ys if y.name == name]
                self.assertTrue(found, choices)
                self.assertEqual(found[0].han, han)
                if han == 0:
                    self.assertEqual(found[0].yakuman, 1)

    def test_triplet_ron_and_best_decomposition(self):
        for win, present in ((0, False), (23, True)):
            result = score_hand(parse_counts('111m222p333456s55z'), win, context=WinContext(riichi=True))
            self.assertEqual('sanankou' in {y.name for y in result.yaku}, present)
        result = score_hand(parse_counts('112233m445566p77s'), 0, context=WinContext(riichi=True))
        names = {y.name for y in result.yaku}
        self.assertIn('ryanpeikou', names)
        self.assertNotIn('iipeikou', names)
        self.assertNotIn('chiitoitsu', names)

    def test_double_and_multiple_yakuman(self):
        for code, win in (('19m19p19s11234567z',27), ('11123455678999m',4), ('111m222p333444s55z',31)):
            self.assertEqual(score_hand(parse_counts(code), win).yakuman, 1)
            self.assertEqual(score_hand(parse_counts(code), win, rules=RiichiRules(double_yakuman=True)).yakuman, 2)
        hand = parse_counts('55m111222333444z')
        self.assertEqual(score_hand(hand, 4).yakuman, 2)
        self.assertEqual(score_hand(hand, 4, rules=RiichiRules(multiple_yakuman=False)).yakuman, 1)

    def test_bonus_counts(self):
        ids = parse_tiles('123406m23455p678s')
        context = WinContext(riichi=True, dora_indicators=(3, 12), ura_indicators=(12,))
        result = score_hand(to_counts(ids), 0, context=context, tile_ids=ids)
        values = {y.name: y.han for y in result.yaku}
        self.assertEqual((values['dora'], values['aka_dora'], values['ura_dora']), (3, 1, 2))
        result = score_hand(to_counts(ids), 0, context=replace(context, riichi=False), tile_ids=ids)
        self.assertNotIn('ura_dora', {y.name for y in result.yaku})
        result = score_hand(to_counts(ids), 0, context=context, tile_ids=ids, rules=RiichiRules(ura_dora=False, red_fives=False, kan_dora=False))
        self.assertEqual({y.name: y.han for y in result.yaku}, {'riichi': 1, 'pinfu': 1, 'dora': 1})
        with self.assertRaises(NoYakuError):
            score_hand(parse_counts('123m456p789s11222z'), 0, context=WinContext(seat_wind=29, dora_indicators=(27,)))

    def test_disabled_open_tanyao_and_atozuke(self):
        hand, called = parse_counts('456m23455p678s'), (meld('234m','chi'),)
        with self.assertRaises(NoYakuError):
            score_hand(hand, 3, called, rules=RiichiRules(kuitan=False))
        with self.assertRaises(ValueError):
            score_hand(hand, 3, called, rules=RiichiRules(atozuke=False))
        with self.assertRaises(NoYakuError):
            score_hand(hand, 3, called, rules=RiichiRules(atozuke=False), context=WinContext(yaku_established=False))
        self.assertEqual(score_hand(hand, 3, called, rules=RiichiRules(atozuke=False), context=WinContext(yaku_established=True)).han, 1)
        with self.assertRaises(ValueError):
            score_hand(hand, 3, called, context=WinContext(riichi=True))

    def test_renhou_and_nagashi(self):
        hand = parse_counts('123m456p789s11222z')
        ctx = WinContext(renhou=True, seat_wind=29)
        with self.assertRaises(NoYakuError):
            score_hand(hand, 0, context=ctx)
        self.assertEqual(score_hand(hand, 0, context=ctx, rules=RiichiRules(renhou='mangan')).base_points, 2000)
        self.assertEqual(score_hand(hand, 0, context=ctx, rules=RiichiRules(renhou='yakuman')).yakuman, 1)
        self.assertTrue(nagashi_eligible((0, 8, 27, 33)))
        self.assertFalse(nagashi_eligible((0, 1, 27)))
        self.assertFalse(nagashi_eligible((0,), any_discard_called=True))
        self.assertFalse(nagashi_eligible((0,), has_open_meld=True))
        self.assertFalse(nagashi_eligible(()))
        self.assertFalse(nagashi_eligible((0,), rules=RiichiRules(nagashi_mangan=False)))

    def test_context_validation(self):
        for kwargs in ({'ippatsu': True}, {'riichi': 1}, {'haitei': True}, {'chankan': True, 'tsumo': True},
                       {'tenhou': True, 'tsumo': True}, {'renhou': True, 'seat_wind': 27},
                       {'haitei': True, 'rinshan': True, 'tsumo': True},
                       {'riichi': True, 'ippatsu': True, 'rinshan': True, 'tsumo': True}, {'houtei': True, 'chankan': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                WinContext(**kwargs)
