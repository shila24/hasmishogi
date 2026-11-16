"""Deterministic native/reference parity, without optional dependencies."""
import ctypes as C
from dataclasses import asdict, fields
from concurrent.futures import ThreadPoolExecutor
import os
import random
import time
import unittest
from unittest.mock import patch

from riichi import native
from riichi.agari import Meld, is_agari
from riichi.rules import RiichiRules
from riichi.scoring import NoYakuError, placement_points, score_hand
from riichi.shanten import shanten, standard_shanten, chiitoitsu_shanten, kokushi_shanten, waits
from riichi.tiles import build_wall, parse_counts, parse_tiles, to_counts
from riichi.yaku import WinContext
from tests.riichi.test_yaku import CASES

SCALE = int(os.environ.get('RIICHI_PARITY_SCALE', '1'))
if SCALE < 1:
    raise ValueError('RIICHI_PARITY_SCALE must be positive')


def setUpModule():
    try:
        native.load_library()
    except (RuntimeError, OSError) as error:
        if os.environ.get('HASAMI_REQUIRE_NATIVE') == '1':
            raise RuntimeError(f'Required native riichi core unavailable: {error}') from error
        raise unittest.SkipTest(f'Native riichi core unavailable: {error}') from error


def score_dict(score):
    return dict(han=score.han, fu=score.fu, yakuman=score.yakuman,
                base_points=score.base_points, payment=asdict(score.payment),
                yaku=[[y.name, y.han, y.yakuman] for y in score.yaku])


def complete_hand(rng):
    """Generate complete structures with physical capacities and random declared sets."""
    while True:
        counts = [0] * 34
        groups = []
        for _ in range(4):
            if rng.randrange(2):
                low = rng.randrange(3) * 9 + rng.randrange(7)
                group = (low, low + 1, low + 2)
            else:
                group = (rng.randrange(34),) * 3
            groups.append(group)
            for t in group:
                counts[t] += 1
        pair = rng.randrange(34)
        counts[pair] += 2
        if max(counts) > 4:
            continue
        melds = []
        # Include open sets and both kinds of kan, while retaining a concealed winning tile.
        for group in groups[:rng.randrange(5)]:
            for t in group:
                counts[t] -= 1
            if group[0] != group[1]:
                melds.append(Meld(group, 'chi'))
            elif rng.randrange(3) == 0 and sum(g.count(group[0]) for g in groups) + 2 * (pair == group[0]) < 4:
                melds.append(Meld(group + (group[0],), 'kan', bool(rng.randrange(2))))
            else:
                melds.append(Meld(group, 'pon'))
        return tuple(counts), tuple(melds)


class NativeCoreTests(unittest.TestCase):
    def assert_score(self, counts, win, melds=(), **kwargs):
        try:
            expected = score_hand(counts, win, melds, **kwargs)
        except ValueError as error:
            with self.assertRaises(type(error)):
                native.score_hand(counts, win, melds, **kwargs)
            return set()
        actual = native.score_hand(counts, win, melds, **kwargs)
        self.assertEqual(actual, score_dict(expected))
        return {y.name for y in expected.yaku}

    def test_abi_and_discovery(self):
        self.assertTrue(native.core_available())
        # available() reports a complete game ABI; it must track the installed library, not a task stage.
        installed = native.load_library()
        self.assertEqual(native.available(), all(hasattr(installed, n) for n in native._GAME_SYMBOLS))

        class Library:
            def __init__(self, names):
                for name in names:
                    setattr(self, name, object())
        for names, expected in ((native._GAME_SYMBOLS, True), (native._GAME_SYMBOLS[:-1], False), ((), False)):
            with self.subTest(symbols=len(names)), patch.object(native, 'load_library', return_value=Library(names)):
                self.assertIs(native.available(), expected)
        with patch.object(native, 'load_library', side_effect=RuntimeError('missing')):
            self.assertFalse(native.available())
        lib = native.load_library()
        out = native.rq_rules()
        self.assertEqual(lib.rq_default_rules(C.byref(out)), 0)
        expected = native.pack_rules(RiichiRules())
        self.assertEqual(bytes(out), bytes(expected))
        with patch.dict(os.environ, {'RIICHI_LIB': '/nonexistent/riichi-library'}):
            self.assertFalse(native.core_available())
            with self.assertRaises(RuntimeError):
                native.load_library()
        class WrongABI:
            class Function:
                def __call__(self):
                    return 999
            def __getattr__(self, name):
                return self.Function()
        with patch.object(C, 'CDLL', return_value=WrongABI()):
            with self.assertRaisesRegex(RuntimeError, 'ABI version'):
                native._load_path('fake-version-library')
        class WrongLayout(WrongABI):
            def __getattr__(self, name):
                if name == 'rq_abi_version':
                    class Version(self.Function):
                        def __call__(self):
                            return 1
                    return Version()
                return self.Function()
        with patch.object(C, 'CDLL', return_value=WrongLayout()):
            with self.assertRaisesRegex(RuntimeError, 'ABI layout'):
                native._load_path('fake-layout-library')

    def test_walls(self):
        for seed in (0, 1, 2**32, 2**63 - 1, 2**64 - 1):
            rng = random.Random(seed)
            self.assertEqual(native.walls(seed, 3), tuple(build_wall(rng=rng) for _ in range(3)))
        self.assertEqual(native.walls(0, 0), ())
        for seed in (-1, 2**64, True, 1.0, '1'):
            with self.assertRaises(ValueError):
                native.walls(seed)
        for count in (-1, True, 2**31, 2**31 // 136 + 1):
            with self.assertRaises(ValueError):
                native.walls(0, count)

    def test_shanten_waits_agari_parity(self):
        rng = random.Random(23001)
        start = time.perf_counter()
        cases = 10000 * SCALE
        for i in range(cases):
            if i % 5 == 0:
                counts, melds = complete_hand(rng)
            else:
                counts, melds = to_counts(rng.sample(range(136), 14)), ()
            win = rng.choice([t for t, n in enumerate(counts) if n])
            before = list(counts)
            before[win] -= 1
            expected = (shanten(before, melds), standard_shanten(before, melds),
                        chiitoitsu_shanten(before, melds), kokushi_shanten(before, melds))
            self.assertEqual(native.shanten_all(before, melds), expected, (i, before, melds))
            self.assertEqual(native.waits(before, melds), waits(before, melds), (i, before, melds))
            self.assertEqual(native.is_agari(counts, melds), is_agari(counts, melds), (i, counts, melds))
        print(f'Native parity: {cases} shanten/waits/agari cases in {time.perf_counter() - start:.3f}s', flush=True)

    def test_capacity_and_decomposition_regressions(self):
        cases = [('1111m222p333s444z', ()), ('1111z222m333p444s', ()),
                 ('89m123p456s11z', (Meld((6,)*4, 'kan'),)),
                 ('6889m112z', (Meld((6,)*3, 'pon'), Meld((9,)*3, 'pon')))]
        for code, melds in cases:
            c = parse_counts(code)
            self.assertEqual(native.shanten(c, melds), shanten(c, melds))
            self.assertEqual(native.waits(c, melds), waits(c, melds))
        for code in ('11223344556677m', '111123m456p789s22z'):
            c = parse_counts(code)
            for win, n in enumerate(c):
                if n:
                    self.assert_score(c, win, context=WinContext(riichi=True))

    def test_every_yaku(self):
        seen = set()
        expected = {case[0] for case in CASES} | {'renhou', 'dora', 'aka_dora', 'ura_dora'}
        for name, code, win, options, melds, _ in CASES:
            options = {'riichi': True, **options}
            if name == 'rinshan':
                options['rinshan'] = True
            winning = next(t for t, n in enumerate(parse_counts(win)) if n)
            for double in (False, True):
                for multiple in (False, True):
                    seen |= self.assert_score(parse_counts(code), winning, melds,
                                              context=WinContext(**options),
                                              rules=RiichiRules(double_yakuman=double, multiple_yakuman=multiple))
        ids = parse_tiles('123406m23455p678s')
        seen |= self.assert_score(to_counts(ids), 0, tile_ids=ids,
                                  context=WinContext(riichi=True, dora_indicators=(3, 12), ura_indicators=(12,)))
        for renhou in ('none', 'mangan', 'yakuman'):
            seen |= self.assert_score(parse_counts('123m456p789s11222z'), 0,
                                      context=WinContext(renhou=True, seat_wind=29),
                                      rules=RiichiRules(renhou=renhou))
        self.assertEqual(seen, expected)

    def test_random_scores(self):
        rng = random.Random(23002)
        start = time.perf_counter()
        cases = 2000 * SCALE
        for _ in range(cases):
            counts, melds = complete_hand(rng)
            win = rng.choice([t for t, n in enumerate(counts) if n])
            closed = not any(m.open for m in melds)
            tsumo = bool(rng.randrange(2))
            riichi = closed and bool(rng.randrange(2))
            context = WinContext(tsumo=tsumo, seat_wind=rng.randrange(27, 31),
                                 round_wind=rng.randrange(27, 31), riichi=riichi,
                                 double_riichi=riichi and bool(rng.randrange(2)),
                                 ippatsu=riichi and bool(rng.randrange(2)),
                                 haitei=tsumo and rng.randrange(4) == 0,
                                 houtei=not tsumo and rng.randrange(4) == 0,
                                 dora_indicators=tuple(rng.randrange(34) for _ in range(rng.randrange(6))),
                                 ura_indicators=tuple(rng.randrange(34) for _ in range(rng.randrange(6))),
                                 yaku_established=rng.choice((None, True, False)))
            rules = RiichiRules(**{name: bool(rng.randrange(2)) for name in (
                'red_fives', 'kuitan', 'atozuke', 'ippatsu', 'ura_dora', 'kan_dora',
                'kazoe_yakuman', 'double_yakuman', 'multiple_yakuman', 'kiriage_mangan')},
                double_wind_pair_fu=rng.choice((2, 4)), honba=rng.choice((0, 300, 600)),
                riichi_stick=rng.choice((0, 1000, 2000)))
            total = list(counts)
            for m in melds:
                for t in m.tiles:
                    total[t] += 1
            ids = tuple(t*4 + copy for t, n in enumerate(total) for copy in range(n))
            self.assert_score(counts, win, melds, context=context, rules=rules,
                              honba=rng.randrange(6), riichi_sticks=rng.randrange(6),
                              tile_ids=ids if rng.randrange(2) else None)
        print(f'Native parity: {cases} scoring cases in {time.perf_counter() - start:.3f}s', flush=True)

    def test_rejected_inputs(self):
        for counts in ((0,)*33, (True,)*34, (-1,)*34, (5,)*34, (4,)*34):
            for reference, compiled in ((shanten, native.shanten), (waits, native.waits), (is_agari, native.is_agari)):
                with self.assertRaises(ValueError):
                    reference(counts)
                with self.assertRaises(ValueError):
                    compiled(counts)
        c = parse_counts('123m456p789s11222z')
        self.assert_score(c, 0, context=WinContext(seat_wind=29, dora_indicators=(27,)))
        for kwargs in ({'tile_ids': ()}, {'tile_ids': (0, 0)}, {'honba': -1}, {'riichi_sticks': -1},
                       {'context': WinContext(red_dora=3)}, {'context': WinContext(tsumo=True, rinshan=True)}):
            self.assert_score(c, 0, **kwargs)
        for win in (-1, 34, True, 1.0):
            self.assert_score(c, win)
        self.assert_score((0,)*34, 0)
        self.assert_score(parse_counts('456m23455p678s'), 3, (Meld((1, 2, 3), 'chi'),),
                          context=WinContext(riichi=True))
        for established in (None, False, True):
            self.assert_score(parse_counts('456m23455p678s'), 3, (Meld((1, 2, 3), 'chi'),),
                              rules=RiichiRules(atozuke=False), context=WinContext(yaku_established=established))

    def test_raw_abi_errors_and_buffers(self):
        lib = native.load_library()
        counts, melds = native._pack_hand(parse_counts('123456m23455p678s'), ())
        out = (C.c_int32 * 34)(*([77]*34))
        for args in ((None, None, 0, out), (counts, None, 1, out), (counts, None, -1, out), (counts, None, 5, out), (counts, None, 0, None)):
            self.assertEqual(lib.rq_shanten(*args), -1)
            self.assertTrue(lib.rq_last_error())
        self.assertEqual(tuple(out), (77,)*34)
        raw_meld = native.rq_meld(0, 2, (C.c_int32*4)(0,1,2,0))
        self.assertEqual(lib.rq_is_agari(counts, C.byref(raw_meld), 1), -1)
        for bad in (-1, 136):
            ids = (C.c_int32*1)(bad)
            self.assertEqual(lib.rq_score_hand_json(counts, 0, None, 0, None, None, 0, 0, ids, 1, None, 0), -1)
        ctx = native.pack_context(WinContext(riichi=True))
        rules = native.pack_rules(RiichiRules())
        args = (counts, 0, melds, 0, C.byref(ctx), C.byref(rules), 0, 0, None, 0)
        length = lib.rq_score_hand_json(*args, None, 0)
        self.assertGreater(length, 0)
        buffer = C.create_string_buffer(b'X' * length)
        before = bytes(buffer)
        self.assertEqual(lib.rq_score_hand_json(*args, buffer, length), -1)
        self.assertEqual(bytes(buffer), before)
        self.assertTrue(lib.rq_last_error())
        self.assertEqual(lib.rq_score_hand_json(*args, buffer, length+1), length)
        self.assertFalse(lib.rq_last_error())
        for field, value in (('tsumo', 2), ('n_dora', 6), ('n_ura', -1), ('yaku_established', 2),
                             ('seat_wind', 0), ('red_dora', 4), ('ippatsu', 2)):
            bad = native.pack_context(WinContext(riichi=True))
            setattr(bad, field, value)
            self.assertEqual(lib.rq_score_hand_json(counts,0,None,0,C.byref(bad),None,0,0,None,0,None,0), -1)
            self.assertTrue(lib.rq_last_error())
        scores, points = (C.c_int32*4)(25000,25000,25000,25000), (C.c_double*4)()
        for field, value in (('kuitan', 2), ('rounds', 2), ('renhou', 3), ('start_points', 0),
                             ('return_points', 0), ('honba', 100), ('tenpai_payment', 100),
                             ('double_wind_pair_fu', 3)):
            bad = native.pack_rules(RiichiRules())
            setattr(bad, field, value)
            self.assertEqual(lib.rq_placement_points(scores, C.byref(bad), points), -1)
            self.assertTrue(lib.rq_last_error())
        self.assertEqual(lib.rq_walls(0, 1, out, 34), -1)
        self.assertEqual(tuple(out), (77,)*34)
        self.assertEqual(lib.rq_default_rules(None), -1)

    def test_all_rule_and_context_validation_fields(self):
        lib = native.load_library()
        scores, out = (C.c_int32 * 4)(25000,25000,25000,25000), (C.c_double * 4)()
        for field in fields(RiichiRules):
            if type(field.default) is bool:
                with self.subTest(rule=field.name):
                    with self.assertRaises(ValueError):
                        RiichiRules(**{field.name: 2})
                    packed = native.pack_rules(RiichiRules())
                    setattr(packed, field.name, 2)
                    self.assertEqual(lib.rq_placement_points(scores, C.byref(packed), out), -1)
                    self.assertTrue(lib.rq_last_error())
                    valid = RiichiRules(**{field.name: not field.default})
                    self.assertEqual(getattr(native.pack_rules(valid), field.name), int(not field.default))
        counts, _ = native._pack_hand(parse_counts('123456m23455p678s'), ())
        invalid_contexts = [
            {'ippatsu': 1}, {'ippatsu': True}, {'haitei': True},
            {'chankan': True, 'tsumo': True}, {'tenhou': True, 'tsumo': True},
            {'renhou': True, 'seat_wind': 27},
            {'haitei': True, 'rinshan': True, 'tsumo': True},
            {'riichi': True, 'ippatsu': True, 'rinshan': True, 'tsumo': True},
            {'houtei': True, 'chankan': True},
            {'tsumo': True, 'chiihou': True, 'riichi': True},
            {'tsumo': True, 'tenhou': True, 'chiihou': True},
            {'seat_wind': 26}, {'round_wind': 31}, {'red_dora': -1},
        ]
        for options in invalid_contexts:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    WinContext(**options)
                packed = native.pack_context(WinContext())
                for name, value in options.items():
                    # The ABI cannot distinguish Python integer 1 from boolean True.
                    setattr(packed, name, 2 if name == 'ippatsu' and type(value) is int else int(value))
                self.assertEqual(lib.rq_score_hand_json(counts,0,None,0,C.byref(packed),None,0,0,None,0,None,0), -1)
                self.assertTrue(lib.rq_last_error())

    def test_thread_local_errors(self):
        from threading import Barrier
        barrier = Barrier(2)
        def worker(walls):
            lib = native.load_library()
            if walls:
                lib.rq_walls(0, -1, None, 0)
            else:
                lib.rq_default_rules(None)
            expected = lib.rq_last_error()
            barrier.wait(timeout=10)
            return expected, lib.rq_last_error()
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(worker, (False, True)))
        self.assertNotEqual(results[0][0], results[1][0])
        for before, after in results:
            self.assertEqual(before, after)

    def test_placement_points(self):
        rng = random.Random(23003)
        for scores in ((25000,)*4, (40000,40000,10000,10000), (0,-1000,50000,51000)):
            self.assertEqual(native.placement_points(scores), placement_points(scores))
        for _ in range(1000):
            scores = tuple(rng.randrange(-20000, 80000) for _ in range(4))
            rules = RiichiRules(start_points=30000, return_points=40000, uma=(30,10,-10,-30))
            self.assertEqual(native.placement_points(scores, rules), placement_points(scores, rules))
        for scores in ((0,)*3, (True,0,0,0), (1.0,0,0,0)):
            with self.assertRaises(ValueError):
                placement_points(scores)
            with self.assertRaises(ValueError):
                native.placement_points(scores)
