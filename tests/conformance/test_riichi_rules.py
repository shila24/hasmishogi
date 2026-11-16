"""Seeded state-machine properties and GameSpec v2 information boundaries."""

import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

from riichi.actions import (ACTION_SIZE, CHI_HIGH, CHI_LOW, CHI_MID, KAN,
                            NINE_TERMINALS, PASS, PON, RIICHI, RON, TSUMO,
                            action_name)
from riichi.baselines import BaselineAgent
from riichi.encoding import encode
from riichi.game import RiichiGame
from riichi.tiles import to_counts
from tests.riichi.test_game import passes, rig, river_tile

NUMPY_AVAILABLE = importlib.util.find_spec('numpy') is not None


class RiichiRulesConformance(unittest.TestCase):
    def test_action_table_is_exactly_111(self):
        names = [action_name(a) for a in range(ACTION_SIZE)]
        self.assertEqual(len(names), 111)
        self.assertEqual(len(set(names)), 111)
        for action in (-1, 111, True, 1.0):
            with self.assertRaises(ValueError):
                action_name(action)

    def test_200_random_full_hanchan(self):
        """Every random hanchan terminates; all physical tiles and points persist."""
        total_steps = 0
        for seed in range(200):
            game = RiichiGame(seed=seed)
            rng = random.Random(seed + 10000)
            while not game.outcome.terminal:
                self.assertLess(len(game.moves), 10000, f'nonterminating seed {seed}')
                actors = game.actors()
                self.assertIn(len(actors), (1, 3))
                actions = {}
                for player in actors:
                    legal = game.legal_actions(player)
                    self.assertTrue(legal)
                    self.assertEqual(len(legal), len(set(legal)))
                    self.assertTrue(all(type(a) is int and 0 <= a < 111 for a in legal))
                    counts = to_counts(game.hands[player])
                    for action in legal:
                        if action < 68:
                            self.assertGreater(counts[action % 34], 0)
                        elif action in (CHI_LOW, CHI_MID, CHI_HIGH):
                            self.assertEqual(player, (game.turn + 1) % 4)
                            tile = game.last_discard[1] // 4
                            low = tile - (action - CHI_LOW)
                            self.assertEqual(low // 9, (low + 2) // 9)
                            self.assertTrue(all(counts[t] for t in range(low, low + 3) if t != tile))
                        elif action in (PON, KAN) and game.phase == 'claim':
                            count = 2 if action == PON else 3
                            self.assertGreaterEqual(counts[game.last_discard[1] // 4], count)
                    actions[player] = rng.choice(legal)
                game.play_joint(actions)
                self.assertEqual(sum(game.scores) + game.riichi_sticks * game.rules.riichi_stick, 100000)
                physical = (game.live_wall + game.dead_wall
                            + [t for h in game.hands for t in h]
                            + [t for ms in game.melds for m in ms for t in m.tiles]
                            + [d.tile for ds in game.discards for d in ds if not d.called])
                self.assertEqual(len(physical), 136, (seed, len(game.moves)))
                self.assertEqual(len(set(physical)), 136, (seed, len(game.moves)))
                for p in range(4):
                    expected = 13 - 3 * len(game.melds[p])
                    if p == game.turn and game.phase in ('draw', 'discard', 'kan_claim'):
                        expected += 1
                    if not game.outcome.terminal:
                        self.assertEqual(len(game.hands[p]), expected)
            self.assertEqual(game.actors(), [])
            self.assertEqual(game.riichi_sticks, 0)
            self.assertEqual(sum(game.scores), 100000)
            self.assertEqual(len(game.final_points), 4)
            total_steps += len(game.moves)
        self.assertGreater(total_steps, 100000)

    def test_riichi_requires_closed_tenpai_points_and_wall(self):
        closed = rig(['123m123p123s789s55z', None, None, None])
        self.assertTrue(any(34 <= a < 68 for a in closed.legal_actions(0)))
        for score, permitted in ((999, False), (1000, True)):
            game = closed.clone()
            game.scores[0] = score
            game._legal_cache.clear()
            self.assertEqual(any(34 <= a < 68 for a in game.legal_actions(0)), permitted)
        short = closed.clone()
        short.live_wall = short.live_wall[:3]
        short._legal_cache.clear()
        self.assertFalse(any(34 <= a < 68 for a in short.legal_actions(0)))
        opened = rig(['123p123s789s55z', None, None, None],
                     declared={0: [('123m', 'chi', True)]})
        self.assertFalse(any(34 <= a < 68 for a in opened.legal_actions(0)))
        far = rig(['147m147p147s12345z', None, None, None])
        self.assertFalse(any(34 <= a < 68 for a in far.legal_actions(0)))

    def test_discard_furiten_covers_every_wait(self):
        game = rig(['9s123m456p789m1122z', '123m123p123s78s55z', None, None])
        river_tile(game, 1, 23)
        game.play(26)
        self.assertEqual(game.player_waits(1), (23, 26))
        self.assertTrue(game.furiten(1)[0])
        self.assertNotIn(RON, game.legal_actions(1))
        self.assertIn(PASS, game.legal_actions(1))

    def test_temporary_and_riichi_furiten(self):
        for riichi in (False, True):
            game = rig(['9s123m456p789m1122z', None, '123m123p123s78s55z', None])
            game.riichi[2] = riichi
            game.play(26)
            self.assertIn(RON, game.legal_actions(2))
            game.play_joint(passes(game))
            self.assertTrue(game.temporary_furiten[2])
            self.assertEqual(game.riichi_furiten[2], riichi)
            game.play(game.drawn_tiles[1] // 4)
            self.assertNotIn(RON, game.legal_actions(2))
            game.play_joint(passes(game))
            self.assertEqual(game.turn, 2)
            self.assertFalse(game.temporary_furiten[2])
            self.assertEqual(game.riichi_furiten[2], riichi)

    def test_furiten_allows_tsumo(self):
        game = rig(['123m123p123s789s55z', None, None, None])
        game.temporary_furiten[0] = True
        game.riichi_furiten[0] = True
        game.riichi[0] = True
        self.assertIn(TSUMO, game.legal_actions(0))

    def test_passing_no_yaku_completion_blocks_same_turn_yakuhai_ron(self):
        game = rig(['4z123m123p789s1122z', '5z456m789p123s666z',
                    '456p789s4455z', None],
                   declared={2: [('123m', 'chi', True)]})
        self.assertEqual(game.player_waits(2), (30, 31))
        game.play(30)
        self.assertNotIn(RON, game.legal_actions(2))
        self.assertIsNone(game._score(2, game.last_discard[1]))
        game.play_joint(passes(game))
        game.play(31)
        # Without the passed North, white dragon would provide the required yaku.
        unfuriten = game.clone()
        unfuriten.temporary_furiten[2] = False
        unfuriten._legal_cache.clear()
        self.assertIn(RON, unfuriten.legal_actions(2))
        self.assertNotIn(RON, game.legal_actions(2))
        self.assertTrue(game.temporary_furiten[2])
        game.play_joint(passes(game))
        self.assertEqual((game.turn, game.phase), (2, 'draw'))
        self.assertFalse(game.temporary_furiten[2])

    def test_ron_priority_over_pon(self):
        game = rig(['3m123p456p789s1122z', '12m123p123s789s55z', '33m456p789s12344z', None])
        game.play(2)
        self.assertIn(RON, game.legal_actions(1))
        self.assertIn(PON, game.legal_actions(2))
        actions = passes(game)
        actions.update({1: RON, 2: PON})
        game.play_joint(actions)
        self.assertEqual(game.last_hand['winners'], [1])
        self.assertEqual(game.last_hand['reason'], 'ron')

    def test_claim_is_rejected_atomically_when_ineligible(self):
        game = RiichiGame(seed=46)
        game.play(game.legal_actions(0)[0])
        before = game.clone()
        actions = passes(game)
        actions[2] = CHI_LOW
        with self.assertRaises(ValueError):
            game.play_joint(actions)
        self.assertEqual(game.history, before.history)
        self.assertEqual(game.hands, before.hands)
        self.assertEqual(game.scores, before.scores)
        self.assertEqual(game.rng.getstate(), before.rng.getstate())

    def test_observations_ignore_all_hidden_state(self):
        for seed in range(12):
            game = RiichiGame(seed=seed)
            for step in range(6):
                for observer in range(4):
                    obs = game.observation(observer)
                    other = game.clone()
                    # Permute the entire hidden pool while preserving physical inventory.
                    locations = []
                    for p in range(4):
                        if p != observer:
                            locations.extend((other.hands[p], i) for i in range(len(other.hands[p])))
                    locations.extend((other.live_wall, i) for i in range(len(other.live_wall)))
                    visible = set(range(4, 4 + 2 * other.dora_count, 2))
                    locations.extend((other.dead_wall, i) for i in range(14) if i not in visible)
                    hidden = [container[i] for container, i in locations]
                    random.Random(seed * 100 + step).shuffle(hidden)
                    for (container, i), tile in zip(locations, hidden):
                        container[i] = tile
                    for p in range(4):
                        if p != observer and other.drawn_tiles[p] is not None:
                            other.drawn_tiles[p] = other.hands[p][-1]
                    other.seed += 100000
                    other._legal_cache.clear()
                    self.assertEqual(other.observation(observer), obs)
                    self.assertEqual(encode(other, observer), encode(game, observer))
                    self.assertNotIn('seed', obs)
                    self.assertNotIn('ura_indicators', obs)
                    self.assertNotIn('hands', obs)
                actions = {p: game.legal_actions(p)[0] for p in game.actors()}
                game.play_joint(actions)

    def test_encoding_public_events_and_private_planes(self):
        game = rig(['3m123p456p789s1122z', '1245m789p123s555z', None, None])
        game.play(2)
        actions = passes(game)
        actions[1] = CHI_LOW
        game.play_joint(actions)
        planes = encode(game, 1)
        self.assertEqual(planes[72][0][2:5], [0.25, 0.25, 0.25])
        self.assertEqual(planes[76][0][2:5], [1.0, 1.0, 1.0])
        self.assertEqual(planes[7][3][2], 0.25)
        self.assertEqual(planes[6][3][2], 0.25)
        self.assertEqual(planes[39][3][2], 1.0)
        self.assertFalse(any(planes[109][0]))
        for plane in (0, 1, 2, 3, 4, 5, 105, 106, 107, 108, 109):
            self.assertFalse(any(value for row in planes[plane][1:] for value in row))
        game = rig(['1111m123p456s789s5z', None, None, None])
        game.play(KAN)
        planes = encode(game, 2)
        self.assertEqual(planes[118][2][0], 1.0)
        game.play_joint(passes(game))
        planes = encode(game, 2)
        self.assertEqual(planes[80][2][0], 1.0)
        self.assertEqual(planes[72][2][0], 1.0)
        self.assertEqual(planes[76][2][0], 0.0)
        game = rig(['123m123p123s789s55z', None, None, None])
        game.play(RIICHI + 31)
        self.assertEqual(encode(game, 2)[91][2][31], 1.0)
        game.play_joint(passes(game))
        self.assertTrue(all(value == 1 for value in encode(game, 2)[88][2]))

    def test_baseline_wins_rejections_and_cancel(self):
        import threading
        game = rig(['123m123p123s789s55z', None, None, None])
        for name in ('tsumogiri', 'shanten-greedy'):
            self.assertEqual(BaselineAgent(name).choose(game).action, TSUMO)
        with self.assertRaises(ValueError):
            BaselineAgent('missing')
        with self.assertRaises(ValueError):
            BaselineAgent().choose(game, player=1)
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(InterruptedError):
            BaselineAgent().choose(game, cancel=cancel)
        game.play(31)
        with self.assertRaises(ValueError):
            BaselineAgent().choose(game)
        for p in (-1, 4, True, 1.0):
            with self.assertRaises(ValueError):
                game.observation(p)
        for seed in (True, 1.0, '1'):
            with self.assertRaises(ValueError):
                RiichiGame(seed=seed)

    def test_baselines_only_return_legal_actions_and_leave_game_unchanged(self):
        for name in ('random', 'tsumogiri', 'shanten-greedy'):
            agent = BaselineAgent(name, seed=2)
            game = RiichiGame(seed=3)
            for _ in range(20):
                before = game.clone()
                actions = {p: agent.choose(game, player=p).action for p in game.actors()}
                for p, action in actions.items():
                    self.assertIs(type(action), int)
                    self.assertIn(action, game.legal_actions(p))
                self.assertEqual(before.hands, game.hands)
                self.assertEqual(before.moves, game.moves)
                game.play_joint(actions)


@unittest.skipUnless(NUMPY_AVAILABLE, 'numpy is optional')
class RiichiAdapterConformance(unittest.TestCase):
    def test_spec_and_fresh_buffers(self):
        import numpy as np
        from hasami_learn.games import get_spec
        from hasami_learn.games.base import require_search_game
        spec = get_spec('riichi')
        self.assertEqual(spec.num_players, 4)
        self.assertFalse(spec.perfect_information)
        self.assertTrue(spec.has_chance)
        self.assertFalse(spec.zero_sum)
        self.assertFalse(spec.supports_undo)
        with self.assertRaises(ValueError):
            require_search_game(spec)
        game = spec.create_game(seed=8)
        planes = spec.observe(game, 0)
        mask = spec.legal_mask(game, 0)
        self.assertEqual(planes.shape, (120, 4, 34))
        self.assertEqual(planes.dtype, np.float32)
        self.assertEqual(mask.shape, (111,))
        self.assertEqual(mask.dtype, np.float32)
        np.testing.assert_array_equal(np.flatnonzero(mask), game.legal_actions(0))
        original = planes.copy()
        planes[:] = -99
        mask[:] = 0
        np.testing.assert_array_equal(spec.observe(game, 0), original)
        self.assertGreater(spec.legal_mask(game, 0).sum(), 0)
        for p in range(1, 4):
            self.assertFalse(spec.legal_mask(game, p).any())
        with self.assertRaises(ValueError):
            spec.terminal_values(game)
        result = spec.step(game, {0: game.legal_actions(0)[0]})
        self.assertFalse(result.terminal)
        self.assertFalse(result.truncated)
        self.assertIsNone(result.rewards)
        self.assertIn('reason', result.info)

    def test_adapter_options_and_baseline_registration(self):
        import json
        import threading
        import numpy as np
        from hasami.agents import AgentCancelled, Decision
        from hasami_learn.games import get_spec
        spec = get_spec('riichi')
        game = spec.create_game(backend='auto', rules={}, seed=12)
        self.assertEqual(json.loads(json.dumps(spec.rules_dict(game))), spec.rules_dict(game))
        self.assertFalse(game.rules.nagashi_mangan)
        # 'auto' prefers the native engine when riichi.native is built (riichi/native.py).
        from hasami_learn.games.riichi import _native_available
        native = _native_available()
        self.assertEqual(spec.resolve_backend('auto'), 'cpp' if native else 'python')
        self.assertEqual(spec.resolve_backend('python'), 'python')
        if not native:
            with self.assertRaises(RuntimeError):
                spec.create_game(backend='cpp')
        self.assertEqual(spec.legal_actions(game, 0), game.legal_actions(0))
        self.assertEqual(spec.action_name(108), 'tsumo')
        self.assertEqual(spec.symmetries(), [])
        encoder = spec.make_encoder()
        np.testing.assert_array_equal(encoder.encode(game), spec.observe(game, 0))
        for name in spec.default_baselines:
            agent, limits = spec.baseline_agent(name, seed=4)
            decision = agent.choose(game, limits)
            self.assertIsInstance(decision, Decision)
            self.assertIn(decision.action, game.legal_actions(0))
            cancel = threading.Event()
            cancel.set()
            with self.assertRaises(AgentCancelled):
                agent.choose(game, limits, cancel)
        for operation in (lambda: spec.create_game(backend='java'),
                          lambda: spec.create_game(position={'turn': 1}),
                          lambda: spec.make_encoder('hasami-enc-v1'),
                          lambda: spec.make_encoder(unexpected=True),
                          lambda: get_spec('riichi', {'unexpected': True})):
            with self.assertRaises(ValueError):
                operation()
        game.play(game.legal_actions(0)[0])
        with self.assertRaises(ValueError):
            encoder.encode(game)

    def test_full_record_roundtrip_and_terminal_rewards(self):
        from hasami_learn.games import get_spec
        from hasami_learn.games.records import load_record, save_record
        spec = get_spec('riichi')
        game = spec.create_game(seed=19)
        rng = random.Random(43)
        while not game.outcome.terminal:
            result = spec.step(game, {p: rng.choice(game.legal_actions(p)) for p in game.actors()})
            self.assertIn('reason', result.info)
        self.assertTrue(result.terminal)
        self.assertFalse(result.truncated)
        self.assertEqual(result.rewards, spec.terminal_values(game))
        self.assertTrue(all(-1 <= v <= 1 for v in result.rewards))
        self.assertEqual(len(result.rewards), 4)
        with tempfile.TemporaryDirectory(prefix='riichi-record-') as directory:
            path = Path(directory) / 'game.json'
            save_record(spec, game, path)
            replay = load_record(spec, path)
            self.assertEqual(replay.moves, game.moves)
            self.assertEqual(replay.scores, game.scores)
            self.assertEqual(replay.outcome, game.outcome)
            self.assertEqual(replay.dead_wall, game.dead_wall)
            self.assertEqual(replay.live_wall, game.live_wall)
            self.assertEqual(replay.hands, game.hands)
            self.assertEqual(spec.terminal_values(replay), result.rewards)
            self.assertEqual(replay.rng.getstate(), game.rng.getstate())
            for p in range(4):
                self.assertFalse(spec.legal_mask(replay, p).any())


if __name__ == '__main__':
    unittest.main()
