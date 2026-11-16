"""State-machine regressions for the four-player game."""

import unittest
import random

from riichi.game import RiichiGame


class GameTests(unittest.TestCase):
    def test_seeded_deal_and_atomic_invalid_action(self):
        game = RiichiGame(seed=19)
        self.assertEqual([len(h) for h in game.hands], [14, 13, 13, 13])
        self.assertEqual(game.remaining_wall, 69)
        self.assertEqual(game.hands, RiichiGame(seed=19).hands)
        self.assertEqual(game.actors(), [0])
        before = game.clone()
        for actions in ({}, {1: 0}, {0: True}, {0: 111}):
            with self.assertRaises(ValueError):
                game.play_joint(actions)
            self.assertEqual(game.hands, before.hands)
            self.assertEqual(game.moves, [])


# Scripted states allocate physical IDs once across all hands, melds and wall.
def rig(codes, *, turn=0, declared=None, rules=None):
    from riichi.game import DeclaredMeld
    from riichi.tiles import parse_counts
    game = RiichiGame(seed=123, rules=rules)
    available = set(range(136))

    def allocate(code):
        tiles = []
        for tile, count in enumerate(parse_counts(code)):
            for _ in range(count):
                physical = min(t for t in available if t // 4 == tile)
                available.remove(physical)
                tiles.append(physical)
        return tiles

    game.melds = [[] for _ in range(4)]
    for p, melds in (declared or {}).items():
        for code, kind, opened in melds:
            tiles = tuple(allocate(code))
            game.melds[p].append(DeclaredMeld(tiles, kind, opened, (p + 1) % 4 if opened else p,
                                             tiles[0] if opened else None))
    game.hands = [allocate(code) if code is not None else [] for code in codes]
    rng = random.Random(987)
    for p, code in enumerate(codes):
        if code is None:
            count = (14 if p == turn else 13) - 3 * len(game.melds[p])
            game.hands[p] = rng.sample(sorted(available), count)
            available.difference_update(game.hands[p])
        expected = (14 if p == turn else 13) - 3 * len(game.melds[p])
        if len(game.hands[p]) != expected:
            raise ValueError(f'scripted player {p} needs {expected} concealed tiles')
    remaining = sorted(available)
    rng.shuffle(remaining)
    game.dead_wall = remaining[:14]
    game.live_wall = remaining[14:]
    game.turn = turn
    game.drawn_tiles = [None] * 4
    game.drawn_tiles[turn] = game.hands[turn][-1]
    game.draw_counts = [2] * 4
    game.uninterrupted = False
    game.kan_count = sum(m.kind == 'kan' for melds in game.melds for m in melds)
    game.dora_count = 1 + game.kan_count
    game._legal_cache.clear()
    return game


def passes(game):
    from riichi.actions import PASS
    return {p: PASS for p in game.actors()}


def river_tile(game, player, tile):
    from riichi.game import Discard
    physical = next(t for t in game.live_wall if t // 4 == tile)
    game.live_wall.remove(physical)
    game.discards[player].append(Discard(physical, False))
    game._legal_cache.clear()


class HandFlowTests(unittest.TestCase):
    def test_discard_physical_policy_and_tsumogiri(self):
        game = rig(['055m123p456s789s11z', None, None, None])
        drawn = next(t for t in game.hands[0] if t // 4 == 4 and t != 16)
        game.drawn_tiles[0] = drawn
        game.play(4)
        self.assertEqual(game.discards[0][-1].tile, drawn)
        self.assertTrue(game.discards[0][-1].tsumogiri)
        self.assertIn(16, game.hands[0])

    def test_riichi_deposit_ippatsu_and_locked_discard(self):
        from riichi.actions import RIICHI
        game = rig(['123m123p123s789s55z', None, None, None])
        game.uninterrupted = True
        self.assertIn(RIICHI + 31, game.legal_actions(0))
        game.play(RIICHI + 31)
        self.assertEqual(game.scores[0], 25000)
        game.play_joint(passes(game))
        self.assertEqual(game.scores[0], 24000)
        self.assertEqual(game.riichi_sticks, 1)
        self.assertTrue(game.double_riichi[0])
        self.assertTrue(game.ippatsu[0])
        while game.turn != 0 or game.phase != 'draw':
            if game.phase == 'claim':
                game.play_joint(passes(game))
            else:
                game.play(game.drawn_tiles[game.turn] // 4)
        self.assertTrue(game.ippatsu[0])
        legal_discards = [a for a in game.legal_actions(0) if a < 68]
        self.assertEqual(legal_discards, [game.drawn_tiles[0] // 4])
        game.play(legal_discards[0])
        self.assertFalse(game.ippatsu[0])

    def test_chi_and_pon_priority_and_kuikae(self):
        from riichi.actions import CHI_HIGH, PON
        game = rig([None, '12456789m123p11s', '33m456p789s12344z', None])
        # Reserve a 3m from the wall as the discarder draw.
        tile = next(t for t in game.live_wall if t // 4 == 2)
        old = game.hands[0][-1]
        game.hands[0][-1] = tile
        game.live_wall[game.live_wall.index(tile)] = old
        game.drawn_tiles[0] = tile
        game.play(2)
        self.assertIn(CHI_HIGH, game.legal_actions(1))
        self.assertIn(PON, game.legal_actions(2))
        actions = passes(game)
        actions.update({1: CHI_HIGH, 2: PON})
        game.play_joint(actions)
        self.assertEqual(game.actors(), [2])
        self.assertEqual(game.melds[2][0].kind, 'pon')
        self.assertNotIn(2, game.legal_actions(2))
        self.assertTrue(game.discards[0][-1].called)
        self.assertFalse(game.uninterrupted)

    def test_chi_sequences_all_positions_and_next_seat_only(self):
        from riichi.actions import CHI_LOW, CHI_MID, CHI_HIGH
        game = rig(['3m123p456p789s1122z', '1245m789p123s555z', None, None])
        game.play(2)
        self.assertTrue({CHI_LOW, CHI_MID, CHI_HIGH} <= set(game.legal_actions(1)))
        for p in (2, 3):
            self.assertFalse({CHI_LOW, CHI_MID, CHI_HIGH} & set(game.legal_actions(p)))
        actions = passes(game)
        actions[1] = CHI_LOW
        game.play_joint(actions)
        self.assertEqual(game.phase, 'discard')
        self.assertEqual(sorted(t // 4 for t in game.melds[1][0].tiles), [2, 3, 4])
        self.assertFalse(any(a >= 34 for a in game.legal_actions(1)))

    def test_double_ron_and_table_sticks_once(self):
        from riichi.actions import RON
        game = rig(['5z123m456p789s1122z', '123m123p123s789s5z', '456m456p456s789m5z', None])
        game.scores[3] -= 2000
        game.riichi_sticks = 2
        game.play(31)
        actions = passes(game)
        actions.update({1: RON, 2: RON})
        game.play_joint(actions)
        self.assertEqual(game.last_hand['winners'], [1, 2])
        self.assertEqual(game.last_hand['reason'], 'ron')
        self.assertEqual(sum(game.scores), 100000)
        self.assertEqual(game.riichi_sticks, 0)
        self.assertEqual(game.dealer, 1)

    def test_triple_ron_aborts_and_atama_hane(self):
        from dataclasses import replace
        from riichi.actions import RON
        from riichi.rules import RiichiRules
        codes = ['5z456m456p678s1122z', '123m123p123s789s5z',
                 '456m456p456s789m5z', '789p111m222p333s5z']
        for atama in (False, True):
            rules = replace(RiichiRules(), nagashi_mangan=False,
                            triple_ron_abort=not atama, atama_hane=atama)
            game = rig(codes, rules=rules)
            game.play(31)
            game.play_joint({p: RON for p in game.actors()})
            self.assertEqual(game.last_hand['reason'], 'ron' if atama else 'triple_ron')
            self.assertEqual(game.last_hand['winners'], [1] if atama else [])
            if not atama:
                self.assertEqual(game.scores, [25000] * 4)
                self.assertEqual((game.dealer, game.honba), (0, 1))

    def test_double_ron_honba_and_sticks_go_to_nearest_winner(self):
        from riichi.actions import PASS, RON
        codes = ['5z123m456p789s1122z', '123m123p123s789s5z',
                 '456m456p456s789m5z', None]
        for source in range(4):
            with self.subTest(source=source):
                game = rig(codes)
                game.hands = [game.hands[(p - source) % 4] for p in range(4)]
                game.drawn_tiles = [game.drawn_tiles[(p - source) % 4] for p in range(4)]
                game.turn = source
                game.dealer = source
                nearest, farther, other = [(source + n) % 4 for n in (1, 2, 3)]
                game.honba = 1
                game.scores[other] -= 2000
                game.riichi_sticks = 2
                before = list(game.scores)
                game.play(31)
                # Submission order must not determine who receives the bonuses.
                game.play_joint({farther: RON, other: PASS, nearest: RON})
                expected = [0] * 4
                expected[source] = -16300
                expected[nearest] = 10300
                expected[farther] = 8000
                self.assertEqual([s - b for s, b in zip(game.scores, before)], expected)
                self.assertEqual(game.last_hand['winners'], [nearest, farther])
                self.assertEqual(game.riichi_sticks, 0)
                self.assertEqual(sum(game.scores), 100000)

    def test_riichi_discard_ron_does_not_deposit(self):
        from riichi.actions import RIICHI, RON
        game = rig(['123m123p123s789s55z', '456m456p456s789m5z', None, None])
        game.play(RIICHI + 31)
        actions = passes(game)
        actions[1] = RON
        game.play_joint(actions)
        self.assertEqual(game.riichi_sticks, 0)
        self.assertEqual(sum(game.scores), 100000)

    def test_tsumo_and_honba(self):
        from riichi.actions import TSUMO
        game = rig(['123m123p123s789s55z', None, None, None])
        game.honba = 2
        self.assertIn(TSUMO, game.legal_actions(0))
        game.play(TSUMO)
        self.assertEqual(game.last_hand['winners'], [0])
        self.assertEqual(game.dealer, 0)
        self.assertEqual(game.honba, 3)
        self.assertEqual(sum(game.scores), 100000)
        self.assertGreater(game.scores[0], 25000)

    def test_closed_kan_rinshan_and_immediate_dora(self):
        from riichi.actions import KAN
        game = rig(['1111m123p456s789s5z', None, None, None])
        expected = game.dead_wall[0]
        tail = game.live_wall[-1]
        before = game.remaining_wall
        game.play(KAN)
        self.assertEqual(game.phase, 'kan_claim')
        self.assertEqual(game.dora_count, 1)
        game.play_joint(passes(game))
        self.assertEqual(game.dora_count, 2)
        self.assertEqual(game.kan_count, 1)
        self.assertEqual(game.drawn_tiles[0], expected)
        self.assertEqual(game.dead_wall[0], tail)
        self.assertEqual(game.remaining_wall, before - 1)
        self.assertTrue(game.closed(0))
        self.assertFalse(game.ippatsu[0])
        self.assertEqual(game.melds[0][0].kind, 'kan')

    def test_open_kan_dora_after_discard(self):
        from riichi.actions import KAN
        game = rig(['1m123p456p789s1122z', '111m456s789p1234z', None, None])
        game.play(0)
        actions = passes(game)
        actions[1] = KAN
        game.play_joint(actions)
        self.assertEqual((game.dora_count, game.pending_dora), (1, 1))
        self.assertEqual(game.actors(), [1])
        self.assertFalse(game.closed(1))
        game.play(next(a for a in game.legal_actions(1) if a < 34))
        self.assertEqual((game.dora_count, game.pending_dora), (2, 0))

    def test_added_kan_and_chankan(self):
        from riichi.actions import KAN, RON
        for rob in (False, True):
            game = rig(['5m123p456789s1z', '46m123p123s789s55z', None, None],
                       declared={0: [('555m', 'pon', True)]})
            game.play(KAN)
            self.assertIn(RON, game.legal_actions(1))
            actions = passes(game)
            if rob:
                actions[1] = RON
            game.play_joint(actions)
            if rob:
                self.assertEqual(game.last_hand['reason'], 'chankan')
                self.assertEqual(game.last_hand['winners'], [1])
            else:
                self.assertEqual(game.melds[0][0].kind, 'kan')
                self.assertEqual((game.dora_count, game.pending_dora), (1, 1))

    def test_closed_kan_can_only_be_robbed_by_kokushi(self):
        from riichi.actions import KAN, RON
        game = rig(['1111m123p456s789s5z', '9m19p19s11234567z', None, None])
        game.play(KAN)
        self.assertIn(RON, game.legal_actions(1))
        actions = passes(game)
        actions[1] = RON
        game.play_joint(actions)
        self.assertEqual(game.last_hand['reason'], 'chankan')

    def test_added_kan_robbery_keeps_ippatsu_payment(self):
        from riichi.actions import KAN, RON
        for hand, amount in (('46m123p678s789s22z', 5200),
                             ('46m123p789s11222z', 8000)):
            with self.subTest(hand=hand):
                game = rig(['5m123p456789s1z', hand, None, None],
                           declared={0: [('555m', 'pon', True)]})
                game.riichi[1] = game.ippatsu[1] = True
                game.play(KAN)
                self.assertIn(RON, game.legal_actions(1))
                actions = passes(game)
                actions[1] = RON
                game.play_joint(actions)
                self.assertEqual(game.last_hand['reason'], 'chankan')
                self.assertEqual(game.scores, [25000 - amount, 25000 + amount, 25000, 25000])

    def test_unrobbed_kan_cancels_ippatsu_before_rinshan(self):
        from riichi.actions import KAN, TSUMO
        for hand, declared, riichi_player, drawn_type, win_type in (
            ('5m123p456789s1z', {0: [('555m', 'pon', True)]}, 1, 4, 27),
            ('1111m123p456s789s5z', None, 0, 0, 31),
        ):
            with self.subTest(hand=hand):
                game = rig([hand, None, None, None], declared=declared)
                game.drawn_tiles[0] = max(t for t in game.hands[0] if t // 4 == drawn_type)
                game.riichi[riichi_player] = game.ippatsu[riichi_player] = True
                # Put the winning tile in the first replacement slot.
                pool, index = next((pool, i)
                                   for pool in [game.live_wall, game.dead_wall, *game.hands[1:]]
                                   for i, t in enumerate(pool) if t // 4 == win_type)
                game.dead_wall[0], pool[index] = pool[index], game.dead_wall[0]
                game.play(KAN)
                game.play_joint(passes(game))
                self.assertEqual(game.ippatsu, [False] * 4)
                self.assertEqual(game.draw_kind, 'rinshan')
                self.assertIn(TSUMO, game.legal_actions(0))
                game.play(TSUMO)
                self.assertEqual(game.last_hand['reason'], 'tsumo')

    def test_multiple_kan_action_ids(self):
        from riichi.actions import KAN, KAN_BASE
        game = rig(['11112222m123p55s1z', None, None, None])
        self.assertNotIn(KAN, game.legal_actions(0))
        self.assertTrue({KAN_BASE, KAN_BASE + 1} <= set(game.legal_actions(0)))
        game.play(KAN_BASE + 1)
        game.play_joint(passes(game))
        self.assertEqual(game.melds[0][0].tiles[0] // 4, 1)

    def test_riichi_kan_waits_must_stay_unchanged(self):
        from riichi.actions import KAN
        game = rig(['1111m123p456s789s5z', None, None, None])
        game.drawn_tiles[0] = max(t for t in game.hands[0] if t // 4 == 0)
        game.riichi[0] = True
        self.assertIn(KAN, game.legal_actions(0))
        game._legal_cache.clear()
        game.drawn_tiles[0] = game.hands[0][-1]
        self.assertNotIn(KAN, game.legal_actions(0))

    def test_riichi_closed_kan_allows_alternate_decompositions(self):
        from riichi.actions import KAN
        game = rig(['1111222333m456p7s', None, None, None])
        game.drawn_tiles[0] = max(t for t in game.hands[0] if t // 4 == 0)
        game.riichi[0] = True
        self.assertEqual(game.player_waits(0), (24,))
        self.assertIn(KAN, game.legal_actions(0))
        game.play(KAN)
        game.play_joint(passes(game))
        self.assertEqual(game.player_waits(0), (24,))
        self.assertTrue(game.closed(0))
        self.assertEqual(game.melds[0][0].kind, 'kan')

    def test_riichi_closed_kan_rejects_changed_waits(self):
        from riichi.actions import KAN
        game = rig(['111123m456p789s55z', None, None, None])
        game.drawn_tiles[0] = max(t for t in game.hands[0] if t // 4 == 0)
        game.riichi[0] = True
        self.assertEqual(game.player_waits(0), (0, 3, 31))
        self.assertNotIn(KAN, game.legal_actions(0))

    def test_nine_terminals_and_four_winds_abort(self):
        from riichi.actions import NINE_TERMINALS
        game = rig(['19m19p19s11234567z', None, None, None])
        game.uninterrupted = True
        self.assertIn(NINE_TERMINALS, game.legal_actions(0))
        game.play(NINE_TERMINALS)
        self.assertEqual(game.last_hand['reason'], 'kyuushu_kyuuhai')
        self.assertEqual((game.dealer, game.honba), (0, 1))
        game = rig(['1z123m456p789s2233z', '1z456m123p123s445z',
                    '1z789m789p456s667z', '1z234m234p234s557z'])
        game.uninterrupted = True
        for p in range(4):
            game.play(27)
            game.play_joint(passes(game))
        self.assertEqual(game.last_hand['reason'], 'suufon_renda')
        self.assertEqual((game.dealer, game.honba), (0, 1))

    def test_four_riichi_and_four_kans_abort_after_ron_window(self):
        from riichi.actions import RIICHI
        game = rig([None, None, None, '123m123p123s789s55z'], turn=3)
        game.riichi[:3] = [True] * 3
        game.play(RIICHI + 31)
        game.play_joint(passes(game))
        self.assertEqual(game.last_hand['reason'], 'suucha_riichi')
        self.assertEqual(game.riichi_sticks, 1)
        game = rig([None, None, None, None])
        game.pending_four_kans = True
        game.play(game.drawn_tiles[0] // 4)
        game.play_joint(passes(game))
        self.assertEqual(game.last_hand['reason'], 'suukaikan')

    def test_exhaustive_draw_payments_and_nagashi_off(self):
        game = rig([None, '123m123p123s789s5z', None, None])
        for p in (0, 2, 3):
            self.assertFalse(game.player_waits(p))
        game.play(game.drawn_tiles[0] // 4)
        game.live_wall.clear()
        game._legal_cache.clear()
        game.play_joint(passes(game))
        self.assertEqual(game.last_hand['reason'], 'exhaustive_draw')
        self.assertEqual(game.scores, [24000, 28000, 24000, 24000])
        self.assertEqual((game.dealer, game.honba), (1, 1))
        self.assertFalse(game.rules.nagashi_mangan)

    def test_last_dealer_agari_yame_tobi_and_final_placement(self):
        from riichi.actions import TSUMO
        for kind in ('agari_yame', 'tobi'):
            game = rig(['123m123p123s789s55z', None, None, None])
            if kind == 'agari_yame':
                game.round_index = 7
                game.dealer = 0
                game.scores = [40000, 20000, 20000, 20000]
            else:
                game.scores = [49900, 100, 25000, 25000]
            game.play(TSUMO)
            self.assertTrue(game.outcome.terminal)
            self.assertEqual(game.outcome.reason, kind)
            self.assertEqual(game.actors(), [])
            self.assertEqual(len(game.final_points), 4)
            self.assertAlmostEqual(sum(game.final_points), 0)
            self.assertEqual(sum(game.scores), 100000)

    def test_clone_and_seed_replay(self):
        import random
        game = RiichiGame(seed=24)
        rng = random.Random(42)
        for _ in range(80):
            game.play_joint({p: rng.choice(game.legal_actions(p)) for p in game.actors()})
        replayed = RiichiGame.replay(game.seed, game.history, game.rules)
        clone = game.clone()
        for other in (replayed, clone):
            self.assertEqual(other.hands, game.hands)
            self.assertEqual(other.dead_wall, game.dead_wall)
            self.assertEqual(other.live_wall, game.live_wall)
            self.assertEqual(other.history, game.history)
            self.assertEqual(other.scores, game.scores)
            self.assertEqual(other.rng.getstate(), game.rng.getstate())
        clone.moves[0].clear()
        clone.hands[0].clear()
        self.assertTrue(game.moves[0])
        self.assertTrue(game.hands[0])


class BoundaryTests(unittest.TestCase):
    def test_kan_that_changes_riichi_wait_is_illegal(self):
        from riichi.actions import KAN
        game = rig(['111123m789p555s11z', None, None, None])
        game.drawn_tiles[0] = max(t for t in game.hands[0] if t // 4 == 0)
        game.riichi[0] = True
        self.assertEqual(game.player_waits(0), (0, 3, 27))
        self.assertNotIn(KAN, game.legal_actions(0))

    def test_ordinary_hand_cannot_rob_closed_kan(self):
        from riichi.actions import KAN, RON
        game = rig(['1111m456p456s789s5z', '23m123p123s789s55z', None, None])
        game.play(KAN)
        self.assertNotIn(RON, game.legal_actions(1))
        self.assertEqual(game.legal_actions(1), [109])

    def test_fourth_kan_abort_is_set_only_for_multiple_owners(self):
        from riichi.actions import KAN
        game = rig(['4444m5z', None, None, None],
                   declared={0: [('1111m', 'kan', False), ('2222m', 'kan', False),
                                 ('3333m', 'kan', False)]})
        game.play(KAN)
        game.play_joint(passes(game))
        self.assertEqual(game.kan_count, 4)
        self.assertFalse(game.pending_four_kans)
        self.assertEqual(game.dora_count, 5)
        self.assertFalse(game._kan_options(0))
        game = rig(['4444m123p5z', '123p456s789s7z', None, None],
                   declared={0: [('1111m', 'kan', False), ('2222m', 'kan', False)],
                             1: [('3333m', 'kan', False)]})
        game.play(KAN)
        game.play_joint(passes(game))
        self.assertTrue(game.pending_four_kans)
        self.assertEqual(game.kan_count, 4)
        game.play(next(a for a in game.legal_actions(0) if a < 34))
        game.play_joint(passes(game))
        self.assertEqual(game.last_hand['reason'], 'suukaikan')

    def test_last_wall_yaku_and_no_calls_or_kans(self):
        from riichi.actions import KAN, RON, TSUMO
        # This open hand has no ordinary yaku; last-wall and rinshan qualify it.
        game = rig(['456p789s55z', None, None, None],
                   declared={0: [('123m', 'chi', True), ('123s', 'chi', True)]})
        self.assertNotIn(TSUMO, game.legal_actions(0))
        game.live_wall.clear()
        game._legal_cache.clear()
        self.assertIn(TSUMO, game.legal_actions(0))
        score = game._score(0, game.drawn_tiles[0], tsumo=True)
        self.assertIn('haitei', {y.name for y in score.yaku})
        game = rig(['5z123m456p789s1122z', '456p789s5z', None, None],
                   declared={1: [('123m', 'chi', True), ('123s', 'chi', True)]})
        game.play(31)
        self.assertNotIn(RON, game.legal_actions(1))
        game.live_wall.clear()
        game._legal_cache.clear()
        self.assertIn(RON, game.legal_actions(1))
        self.assertFalse(set(game.legal_actions(1)) & {68, 69, 70, 71, KAN})
        score = game._score(1, game.last_discard[1])
        self.assertIn('houtei', {y.name for y in score.yaku})

    def test_rinshan_yaku_and_indicator_flags(self):
        from dataclasses import replace
        from riichi.actions import KAN, TSUMO
        from riichi.rules import RiichiRules
        for kan_dora, immediate in ((True, True), (True, False), (False, True)):
            rules = replace(RiichiRules(), nagashi_mangan=False,
                            kan_dora=kan_dora, closed_kan_dora_immediate=immediate)
            game = rig(['1111m456p789s5z', None, None, None],
                       declared={0: [('123s', 'chi', True)]}, rules=rules)
            desired = next(t for t in game.live_wall if t // 4 == 31)
            index = game.live_wall.index(desired)
            game.live_wall[index], game.dead_wall[0] = game.dead_wall[0], desired
            game.play(KAN)
            game.play_joint(passes(game))
            self.assertEqual(game.dora_count, 2 if kan_dora and immediate else 1)
            self.assertIn(TSUMO, game.legal_actions(0))
            score = game._score(0, game.drawn_tiles[0], tsumo=True)
            self.assertIn('rinshan', {y.name for y in score.yaku})
            self.assertNotIn('haitei', {y.name for y in score.yaku})

    def test_exhaustive_dealer_repeat_and_last_round_end(self):
        game = rig(['123m123p123s789s55z', None, None, None])
        game.play(31)
        game.live_wall.clear()
        game._legal_cache.clear()
        game.play_joint(passes(game))
        self.assertEqual((game.dealer, game.honba), (0, 1))
        self.assertTrue(game.last_hand['tenpai'][0])
        game = rig([None, None, None, None], turn=3)
        game.round_index = 7
        game.dealer = 3
        game.scores = [30000, 30000, 19000, 20000]
        game.riichi_sticks = 1
        self.assertFalse(game.player_waits(3))
        game.play(game.drawn_tiles[3] // 4)
        game.live_wall.clear()
        game._legal_cache.clear()
        game.play_joint(passes(game))
        self.assertTrue(game.outcome.terminal)
        self.assertEqual(game.outcome.reason, 'rounds_complete')
        self.assertEqual(game.outcome.winner, 0)
        self.assertEqual(game.scores, [31000, 30000, 19000, 20000])
        self.assertEqual(game.riichi_sticks, 0)
        with self.assertRaises(ValueError):
            game.play_joint({})

    def test_first_turn_yaku(self):
        for player, expected in ((0, 'tenhou'), (1, 'chiihou')):
            codes = [None] * 4
            codes[player] = '123m123p123s789s55z'
            game = rig(codes, turn=player)
            game.uninterrupted = True
            score = game._score(player, game.drawn_tiles[player], tsumo=True)
            self.assertIn(expected, {y.name for y in score.yaku})

    def test_nagashi_explicit_opt_in(self):
        from riichi.rules import RiichiRules
        game = rig([None, None, None, None], rules=RiichiRules(nagashi_mangan=True))
        game.play(game.drawn_tiles[0] // 4)
        for p in range(4):
            game.discards[p].clear()
        river_tile(game, 1, 27)
        game.live_wall.clear()
        game._legal_cache.clear()
        game.play_joint(passes(game))
        self.assertEqual(game.last_hand['reason'], 'nagashi_mangan')
        self.assertEqual(game.scores, [21000, 33000, 23000, 23000])


if __name__ == '__main__':
    unittest.main()
