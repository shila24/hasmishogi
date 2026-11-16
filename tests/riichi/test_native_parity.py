"""Bit-exact game, observation, encoding, replay, and rejection parity."""
from array import array
import copy
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
import ctypes as C
from dataclasses import replace
import os
import random
import unittest

from riichi import native
from riichi.actions import PASS
from riichi.baselines import BaselineAgent
from riichi.encoding import encode_observation
from riichi.game import RiichiGame
from riichi.rules import RiichiRules

SCALE = max(1, int(os.environ.get('RIICHI_PARITY_SCALE', '1')))


try:
    import numpy as np
except ImportError:
    np = None


def planes_bytes(obs):
    planes = encode_observation(obs)
    if np is not None:
        return np.asarray(planes, dtype=np.float32).tobytes()
    return array('f', chain.from_iterable(chain.from_iterable(planes))).tobytes()


class NativeParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not native.available():
            if os.environ.get('HASAMI_REQUIRE_NATIVE') == '1':
                raise AssertionError('native riichi game is required')
            raise unittest.SkipTest('native riichi game is unavailable')

    def compare(self, py, cpp, *, moves=True):
        for name in ('seed', 'rules', 'outcome', 'scores', 'final_points', 'hand_number',
                     'hand_start', 'round_index', 'dealer', 'honba', 'riichi_sticks',
                     'phase', 'turn', 'remaining_wall', 'last_hand'):
            self.assertEqual(getattr(py, name), getattr(cpp, name), name)
        self.assertEqual(py.actors(), cpp.actors())
        if moves:
            self.assertEqual(py.moves, cpp.moves)
            size = len(py.moves) * 4
            out = (C.c_int32 * size)()
            self.assertEqual(cpp._lib.rq_moves(cpp._handle, out, size), size)
            self.assertEqual(list(out), [step.get(p, -1) for step in py.moves for p in range(4)])
        for p in range(4):
            self.assertEqual(py.legal_actions(p), cpp.legal_actions(p), f'legal seat {p}')
            obs = py.observation(p)
            self.assertEqual(obs, cpp.observation(p), f'observation seat {p}')
            planes = array('f', [0]) * 16320
            mask = array('f', [0]) * 111
            cpp.encode_into(p, planes)
            cpp.legal_mask_into(p, mask)
            self.assertEqual(planes_bytes(obs), planes.tobytes(), f'planes seat {p}')
            expected = array('f', [float(a in obs['legal_actions']) for a in range(111)])
            self.assertEqual(mask.tobytes(), expected.tobytes(), f'mask seat {p}')

    def test_initial_and_wraparound_claim_order(self):
        for seed in (0, 1, 2 ** 32, 2 ** 63 - 1, 2 ** 64 - 1):
            py = RiichiGame(seed)
            with native.NativeRiichiGame(seed) as cpp:
                for seat in range(4):
                    self.compare(py, cpp)
                    action = py.drawn_tiles[seat] // 4
                    self.assertEqual(py.play(action), cpp.play(action))
                    self.assertEqual(cpp.actors(), [(seat + offset) % 4 for offset in (1, 2, 3)])
                    self.compare(py, cpp)
                    actions = {p: PASS for p in py.actors()}
                    self.assertEqual(py.play_joint(actions), cpp.play_joint(actions))

    def test_lockstep_matches(self):
        variants = [dict(), dict(red_fives=False), dict(kuitan=False), dict(double_ron=False),
                    dict(nagashi_mangan=True), dict(tobi=False), dict(atama_hane=True),
                    dict(kan_dora=False, ura_dora=False, ippatsu=False),
                    dict(open_kan_dora_after_discard=False, closed_kan_dora_immediate=False),
                    dict(track_tsumogiri=False, discard_furiten=False, temporary_furiten=False,
                         riichi_furiten=False)]
        jobs = [(rounds, index, variants[index % len(variants)])
                for rounds in ('hanchan', 'tonpuu') for index in range(20 * SCALE)]
        # Independent seeded matches use separate processes, never shared handles.
        workers = max(1, min(4, int(os.environ.get('RIICHI_PARITY_WORKERS', '4'))))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(run_match, jobs))
        self.assertEqual(sum(r[0] == 'hanchan' for r in results), 20 * SCALE)
        self.assertEqual(sum(r[0] == 'tonpuu' for r in results), 20 * SCALE)
        print(f'parity: {20 * SCALE} hanchan + {20 * SCALE} tonpuu; '
              f'{sum(r[1] for r in results)} joint steps', flush=True)

    def test_invalid_actions_clone_replay_and_close(self):
        py = RiichiGame(42)
        with native.NativeRiichiGame(42) as cpp:
            for actions in ({}, {1: 0}, {0: True}, {0: 111}, {0: -1}, {0: 2 ** 64},
                            {False: 0}, {0: 1.5}, None, []):
                before = [cpp.observation(p) for p in range(4)]
                for game in (py, cpp):
                    with self.assertRaises(ValueError):
                        game.play_joint(actions)
                self.assertEqual(before, [cpp.observation(p) for p in range(4)])
                self.compare(py, cpp)
            rng = random.Random(2)
            for _ in range(20):
                actions = {p: rng.choice(py.legal_actions(p)) for p in py.actors()}
                py.play_joint(actions)
                cpp.play_joint(actions)
            with cpp.clone() as clone:
                self.compare(py, clone)
                clone.play_joint({p: clone.legal_actions(p)[0] for p in clone.actors()})
                self.compare(py, cpp)
            with native.NativeRiichiGame.replay(py.seed, py.moves, py.rules) as replay:
                self.compare(py, replay)
        cpp.close()
        for function in (cpp.actors, cpp.clone, cpp.legal_actions, lambda: cpp.observation(0),
                         lambda: cpp.play_joint({}), lambda: cpp.encode_into(0, array('f', [0]) * 16320)):
            with self.assertRaises(ValueError):
                function()

    def test_redeal_acceptance_rejection_and_rng(self):
        from tests.riichi.test_redeal import shuffled_hidden
        py = RiichiGame(11)
        with native.NativeRiichiGame(11) as cpp:
            rng = random.Random(11)
            for _ in range(9):
                actions = {p: rng.choice(py.legal_actions(p)) for p in py.actors()}
                py.play_joint(actions)
                cpp.play_joint(actions)
            hands, live, dead = shuffled_hidden(py, random.Random(1), 0)
            valid = [hands, live, dead, list(py.drawn_tiles), [False] * 4, [False] * 4, 2 ** 64 - 1]
            cases = []
            for seed in (-1, 2 ** 64, True, 1.0):
                case = copy.deepcopy(valid); case[-1] = seed; cases.append(case)
            case = copy.deepcopy(valid); case[0][0][0] = 136; cases.append(case)
            case = copy.deepcopy(valid); case[0][0][0] = True; cases.append(case)
            case = copy.deepcopy(valid); case[0][1].append(case[0][0].pop()); cases.append(case)
            case = copy.deepcopy(valid); case[2][4], case[2][5] = case[2][5], case[2][4]; cases.append(case)
            case = copy.deepcopy(valid); case[2].append(case[1].pop()); cases.append(case)
            case = copy.deepcopy(valid); case[4] = [0] * 4; cases.append(case)
            case = copy.deepcopy(valid); case[3] = [hands[p][0] if t is None else None
                                                   for p, t in enumerate(valid[3])]; cases.append(case)
            for args in cases:
                for game in (py, cpp):
                    with self.assertRaises(ValueError):
                        game.redeal(*args)
                self.compare(py, cpp)
            py.redeal(*valid); cpp.redeal(*valid)
            self.compare(py, cpp)
            for _ in range(300):
                if py.outcome.terminal:
                    break
                actions = {p: rng.choice(py.legal_actions(p)) for p in py.actors()}
                self.assertEqual(py.play_joint(actions), cpp.play_joint(actions))
                self.compare(py, cpp)

    def test_constructed_calls_kans_wins_and_aborts(self):
        from riichi.actions import (CHI_LOW, CHI_MID, CHI_HIGH, KAN, KAN_BASE,
                                    NINE_TERMINALS, PON, RIICHI, RON, TSUMO)
        cases = [
            (['123m123p123s789s55z', None, None, None], [TSUMO], 'tsumo'),
            (['19m19p19s11234567z', None, None, None], [NINE_TERMINALS], 'kyuushu_kyuuhai'),
            (['1111m123p456s789s5z', None, None, None], [KAN, {}], None),
            (['11112222m123p55s1z', None, None, None], [KAN_BASE + 1, {}], None),
            (['1111m123p456s789s5z', '9m19p19s11234567z', None, None],
             [KAN, {1: RON}], 'chankan'),
            (['1m123p456p789s1122z', '111m456s789p1234z', None, None],
             [0, {1: KAN}], None),
            (['3m123p456p789s1122z', '1245m789p123s555z', None, None],
             [2, {1: CHI_LOW}], None),
            (['3m123p456p789s1122z', '1245m789p123s555z', None, None],
             [2, {1: CHI_MID}], None),
            (['3m123p456p789s1122z', '1245m789p123s555z', None, None],
             [2, {1: CHI_HIGH}], None),
            (['3m123p456p789s1122z', '12456789m123p11s', '33m456p789s12344z', None],
             [2, {1: CHI_HIGH, 2: PON}], None),
            (['5z123m456p789s1122z', '123m123p123s789s5z', '456m456p456s789m5z', None],
             [31, {1: RON, 2: RON}], 'ron'),
            (['5z456m456p678s1122z', '123m123p123s789s5z',
              '456m456p456s789m5z', '789p111m222p333s5z'],
             [31, {1: RON, 2: RON, 3: RON}], 'triple_ron'),
            (['123m123p123s789s55z', None, None, None], [RIICHI + 31, {}], None),
        ]
        for codes, script, reason in cases:
            for overrides in ({}, {'red_fives': False, 'open_kan_dora_after_discard': False,
                                   'closed_kan_dora_immediate': False}):
                with self.subTest(codes=codes, script=script, rules=overrides):
                    py, cpp = constructed(codes, overrides)
                    try:
                        self.compare(py, cpp)
                        for action in script:
                            actions = ({**{p: PASS for p in py.actors()}, **action}
                                       if isinstance(action, dict) else {py.turn: action})
                            for p, a in actions.items():
                                self.assertIn(a, py.legal_actions(p))
                            self.assertEqual(py.play_joint(actions), cpp.play_joint(actions))
                            self.compare(py, cpp)
                        if reason:
                            self.assertEqual(cpp.last_hand['reason'], reason)
                        # Continue through further draws and the next seeded hand.
                        rng = random.Random(81)
                        for _ in range(160):
                            if py.outcome.terminal:
                                break
                            actions = {p: rng.choice(py.legal_actions(p)) for p in py.actors()}
                            self.assertEqual(py.play_joint(actions), cpp.play_joint(actions))
                            self.compare(py, cpp)
                    finally:
                        cpp.close()

    def test_four_winds_four_riichi_and_four_kans(self):
        from riichi.actions import KAN, RIICHI
        scenarios = [
            ('suufon_renda', ['1z123m456p789s2233z', '1z456m123p123s445z',
                              '1z789m789p456s667z', '1z234m234p234s557z']),
            ('suucha_riichi', ['123m123p123s789s55z', '456m456p456s789m6z',
                               '789m789p789s123m7z', '456m456p456s789p1z']),
            ('suukaikan', ['1111m123p456s789s5z', '2222m456p123s667z',
                           '3333m789p456s177z', '4444m123p789s223z']),
        ]
        for reason, codes in scenarios:
            py, cpp = constructed(codes, {})
            try:
                for seat in range(4):
                    self.assertEqual(py.actors(), [seat])
                    if reason == 'suukaikan':
                        self.assertIn(KAN, py.legal_actions(seat))
                        py.play(KAN); cpp.play(KAN)
                        self.compare(py, cpp)
                        actions = {p: PASS for p in py.actors()}
                        py.play_joint(actions); cpp.play_joint(actions)
                        self.compare(py, cpp)
                    action = (27 if reason == 'suufon_renda' else
                              RIICHI + (31 if seat == 0 else py.drawn_tiles[seat] // 4)
                              if reason == 'suucha_riichi' else
                              next(a for a in py.legal_actions(seat) if a < 34))
                    self.assertIn(action, py.legal_actions(seat))
                    py.play(action); cpp.play(action)
                    self.compare(py, cpp)
                    actions = {p: PASS for p in py.actors()}
                    py.play_joint(actions); cpp.play_joint(actions)
                    self.compare(py, cpp)
                self.assertEqual(cpp.last_hand['reason'], reason)
                self.assertEqual((cpp.dealer, cpp.honba), (0, 1))
            finally:
                cpp.close()

    def test_added_kan_and_chankan(self):
        from riichi.actions import KAN, PON, RON
        for rob in (False, True):
            py, cpp = constructed(['5m123p456p789s1122z', '55m123p456s789s12z', None, None], {})
            try:
                for actions in ({0: 4}, {1: PON, 2: PASS, 3: PASS}):
                    py.play_joint(actions); cpp.play_joint(actions)
                self.compare(py, cpp)
                # Continue with passes and discards until the caller's next draw.
                while not (py.phase == 'draw' and py.turn == 1):
                    actions = ({p: PASS for p in py.actors()} if py.phase == 'claim' else
                               {py.turn: next(a for a in py.legal_actions(py.turn) if a < 34)})
                    py.play_joint(actions); cpp.play_joint(actions)
                # Redistribute the remaining hidden tiles while preserving all public state.
                codes = [None, '5m123p456789s1z', '46m123p123s789s55z', None]
                from riichi.tiles import parse_counts
                pool = set(sum(py.hands, []) + py.live_wall + py.dead_wall) - set(py.dora_indicators)
                hands = [[] for _ in range(4)]
                possible = True
                for p, code in enumerate(codes):
                    if code is None:
                        continue
                    for t, n in enumerate(parse_counts(code)):
                        ids = sorted(v for v in pool if v // 4 == t)
                        if len(ids) < n:
                            possible = False
                            break
                        hands[p].extend(ids[:n]); pool.difference_update(ids[:n])
                # Already discarded types can make a fixture impossible: this seed is fixed.
                self.assertTrue(possible)
                for p, code in enumerate(codes):
                    if code is None:
                        hands[p] = sorted(pool)[:len(py.hands[p])]
                        pool.difference_update(hands[p])
                rest = sorted(pool)
                dead = rest[:13]; dead.insert(4, py.dora_indicators[0])
                args = (hands, rest[13:], dead, [None, hands[1][0], None, None],
                        [False] * 4, [False] * 4, 121)
                py.redeal(*args); cpp.redeal(*args)
                self.compare(py, cpp)
                self.assertIn(KAN, py.legal_actions(1))
                py.play(KAN); cpp.play(KAN)
                self.compare(py, cpp)
                self.assertIn(RON, py.legal_actions(2))
                actions = {p: RON if p == 2 and rob else PASS for p in py.actors()}
                py.play_joint(actions); cpp.play_joint(actions)
                self.compare(py, cpp)
                if rob:
                    self.assertEqual(cpp.last_hand['reason'], 'chankan')
                else:
                    self.assertEqual(cpp.observation(1)['melds'][0][0]['kind'], 'kan')
            finally:
                cpp.close()

    def test_buffers_and_abi_errors(self):
        with native.NativeRiichiGame(1) as game:
            before = game.observation(0)
            for buffer in (bytes(16320 * 4), array('d', [0]) * 16320,
                           array('f', [0]) * 16319, memoryview(array('f', [0]) * 32640)[::2]):
                with self.assertRaises(ValueError):
                    game.encode_into(0, buffer)
            for player in (-1, 4, True, 0.0):
                with self.assertRaises(ValueError):
                    game.observation(player)
            unaligned = memoryview(bytearray(1 + 16320 * 4))[1:].cast('f')
            game.encode_into(0, unaligned)
            self.assertEqual(unaligned.tobytes(), planes_bytes(before))
            lib, handle = game._lib, game._handle
            self.assertEqual(lib.rq_play_joint(handle, None), -1)
            self.assertEqual(lib.rq_get_state(None, C.byref(native.rq_state())), -1)
            self.assertEqual(lib.rq_redeal(handle, None), -1)
            self.assertEqual(lib.rq_encode(handle, 0, None, 16320), -1)
            self.assertEqual(lib.rq_legal_actions(handle, 0, (C.c_int32 * 1)(), 1), -1)
            self.assertEqual(game.observation(0), before)


def constructed(codes, overrides):
    """Reach crafted hands through the public redeal contract, preserving the indicator."""
    from riichi.tiles import parse_counts
    counts = [None if code is None else parse_counts(code) for code in codes]
    totals = [sum(row[t] for row in counts if row is not None) for t in range(34)]
    seed = next(seed for seed in range(1000)
                if totals[RiichiGame(seed).dora_indicators[0] // 4] < 4)
    rules = replace(RiichiRules(nagashi_mangan=False), **overrides)
    py = RiichiGame(seed, rules)
    cpp = native.NativeRiichiGame(seed, rules)
    indicator = py.dora_indicators[0]
    pool = set(range(136)) - {indicator}
    hands = [[] for _ in range(4)]
    for p, row in enumerate(counts):
        if row is None:
            continue
        for t, n in enumerate(row):
            for _ in range(n):
                physical = min(v for v in pool if v // 4 == t)
                pool.remove(physical)
                hands[p].append(physical)
    rng = random.Random(987)
    for p, row in enumerate(counts):
        if row is None:
            hands[p] = rng.sample(sorted(pool), 14 if p == 0 else 13)
            pool.difference_update(hands[p])
    rest = sorted(pool)
    rng.shuffle(rest)
    dead = rest[:13]
    dead.insert(4, indicator)
    args = (hands, rest[13:], dead, [hands[0][-1], None, None, None],
            [False] * 4, [False] * 4, seed)
    try:
        py.redeal(*args)
        cpp.redeal(*args)
        return py, cpp
    except BaseException:
        cpp.close()
        raise


def run_match(job):
    rounds, index, options = job
    check = NativeParityTests()
    seed = 24000 + index
    rules = replace(RiichiRules(nagashi_mangan=False), rounds=rounds, **options)
    agents = [BaselineAgent('shanten-greedy', seed + p) for p in range(4)]
    rng = random.Random(seed + 1000)
    py = RiichiGame(seed, rules)
    with native.NativeRiichiGame(seed, rules) as cpp:
        check.compare(py, cpp)
        while not py.outcome.terminal:
            check.assertLess(len(py.moves), 20000)
            actions = {p: (agents[p].choose(py, player=p).action if index % 2 else
                           rng.choice(py.legal_actions(p))) for p in py.actors()}
            check.assertEqual(py.play_joint(actions), cpp.play_joint(actions))
            try:
                check.compare(py, cpp)
            except AssertionError as exc:
                raise AssertionError(f'{rounds} seed={seed} step={len(py.moves)}: {exc}') from exc
        for game in (py, cpp):
            with check.assertRaises(ValueError):
                game.redeal(py.hands, py.live_wall, py.dead_wall, py.drawn_tiles,
                            [False] * 4, [False] * 4, 0)
        check.compare(py, cpp)
        with native.NativeRiichiGame.replay(seed, py.moves, rules) as replay:
            check.compare(py, replay)
    return rounds, len(py.moves)


if __name__ == '__main__':
    unittest.main()
