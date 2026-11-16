"""Poker GameSpec, observer privacy, PPO league checkpoints and method-A advice."""

import json
from pathlib import Path
import random
import tempfile
import threading
import unittest

from tests.learn._common import requires_torch

ROOT = Path(__file__).resolve().parents[2]


@requires_torch
class PokerAdapterTests(unittest.TestCase):
    def setUp(self):
        from hasami_learn.games import get_spec
        self.spec = get_spec('poker')

    def test_contract_and_encoder_semantics(self):
        import numpy as np
        from hasami_learn.games.base import GameSpec, require_search_game
        self.assertIsInstance(self.spec, GameSpec)
        self.assertEqual((self.spec.num_players, self.spec.action_size, self.spec.board_shape), (2, 3, (4, 13)))
        self.assertTrue(self.spec.has_chance and self.spec.supports_undo and self.spec.zero_sum)
        self.assertFalse(self.spec.perfect_information)
        with self.assertRaises(ValueError):
            require_search_game(self.spec)
        game = self.spec.create_game(seed=0)
        planes = self.spec.observe(game, 0)
        self.assertEqual(planes.shape, (141, 4, 13))
        self.assertEqual(planes.dtype, np.float32)
        self.assertEqual(planes[0].sum(), 2)
        self.assertEqual(planes[4].sum(), 2)
        self.assertTrue(np.all(planes[5:7] == 1))
        self.assertTrue(np.all(planes[7:10] == 0))
        for index, value in ((10, 0.125), (11, 0.25), (12, 0.25)):
            self.assertTrue(np.all(planes[index] == value))
        self.spec.step(game, {0: 1})
        self.spec.step(game, {1: 1})
        planes = self.spec.observe(game, 0)
        self.assertEqual(planes[1].sum(), 3)
        self.assertEqual(planes[4].sum(), 5)
        self.assertTrue(np.all(planes[7] == 1))
        self.assertTrue(np.all(planes[13] == 1))
        self.assertTrue(np.all(planes[19] == 1))
        opposite = self.spec.observe(game, 1)
        self.assertTrue(np.all(opposite[15] == 1))
        self.assertTrue(np.all(opposite[17] == 1))
        np.testing.assert_array_equal(self.spec.legal_mask(game, 1), [0, 1, 1])
        self.assertEqual(self.spec.legal_mask(game, 0).sum(), 0)
        self.assertEqual(self.spec.symmetries(), [])
        with self.assertRaises(ValueError):
            self.spec.create_game('cpp')
        with self.assertRaises(ValueError):
            self.spec.create_game(rules={'variant': 'leduc'})
        with self.assertRaises(ValueError):
            self.spec.make_encoder('hasami-enc-v1')

    def test_step_joint_validation_undo_and_fixed_seat_values(self):
        game = self.spec.create_game(seed=0)
        before = game.history_keys()
        for actions in ({}, {1: 1}, {0: 1, 1: 1}, {False: 1}, {0: True}):
            with self.assertRaises(ValueError):
                self.spec.step(game, actions)
            self.assertEqual(game.history_keys(), before)
        with self.assertRaises(ValueError):
            self.spec.terminal_values(game)
        for action in game.legal_actions():
            outcome = self.spec.step(game, {0: action})
            if action == 0:
                self.assertTrue(outcome.terminal)
                self.assertFalse(outcome.truncated)
                self.assertEqual(outcome.rewards, [-1 / 48, 1 / 48])
            game.undo()
            self.assertEqual(game.history_keys(), before)
        for value in (-1, 0, 1):
            self.assertEqual(self.spec.outcome_score([value, -value], 0), (value + 1) / 2)
        while not game.outcome.terminal:
            game.play(2 if 2 in game.legal_actions() else 1)
        self.assertTrue(all(-1 <= v <= 1 for v in self.spec.terminal_values(game)))
        self.assertEqual(sum(self.spec.terminal_values(game)), 0)

    def test_street_card_planes_history_and_truncation(self):
        import numpy as np
        from hasami_learn.selfplay import run_table_game
        game = self.spec.create_game(seed=42)
        while not game.outcome.terminal:
            player = game.actors()[0]
            obs = game.observation(player)
            planes = self.spec.observe(game, player)
            for plane, cards in ((0, obs['hole_cards']), (1, obs['board'][:3]),
                                 (2, obs['board'][3:4]), (3, obs['board'][4:5]),
                                 (4, obs['hole_cards'] + obs['board'])):
                expected = {(card % 4, card // 4) for card in cards}
                self.assertEqual(set(map(tuple, np.argwhere(planes[plane] == 1))), expected)
            self.assertEqual(list(planes[6:10, 0, 0]), [float(s == game.street) for s in range(4)])
            for street, events in enumerate(obs['history']):
                for slot, event in enumerate(events):
                    expected = [0.0] * 4
                    offset = (0 if event['player'] == player else 2) + event['action'] - 1
                    expected[offset] = 1.0
                    self.assertEqual(list(planes[13 + street * 32 + slot * 4:
                                                 17 + street * 32 + slot * 4, 0, 0]), expected)
            game.play(2 if 2 in game.legal_actions() else 1)
        agents = {p: self.spec.baseline_agent('call', seed=p)[0] for p in (0, 1)}
        result = run_table_game(agents, self.spec.make_encoder(), spec=self.spec, seed=0, max_plies=1)
        self.assertTrue(result.truncated)
        self.assertEqual(result.winner, -2)
        self.assertEqual(result.reason, 'max_plies')

    def test_determinize_every_street_both_seats_many_seeds(self):
        import numpy as np
        from hasami_learn.games import get_spec
        for variant in ('hulhe', 'leduc'):
            spec = get_spec('poker', {'variant': variant})
            game = spec.create_game(seed=27)
            while not game.outcome.terminal:
                for player in (0, 1):
                    before = game.history_keys()
                    worlds = set()
                    for seed in range(100):
                        work = spec.determinize(game, player, np.random.default_rng(seed))
                        self.assertIsNot(work, game)
                        self.assertEqual(game.observation(player), work.observation(player))
                        np.testing.assert_array_equal(spec.observe(game, player), spec.observe(work, player))
                        np.testing.assert_array_equal(spec.legal_mask(game, player), spec.legal_mask(work, player))
                        other = spec.determinize(work, player, np.random.default_rng(seed))
                        self.assertEqual(work.history_keys(), other.history_keys())
                        worlds.add(tuple(work.observation(1 - player)['hole_cards']))
                        other.close()
                        work.close()
                    self.assertGreater(len(worlds), 1)
                    self.assertEqual(game.history_keys(), before)
                game.play(1)

    def test_generic_records_variants_buttons_and_sampled_world_rejection(self):
        import numpy as np
        from hasami_learn.games import get_spec
        from hasami_learn.games.records import load_record, save_record
        with tempfile.TemporaryDirectory(prefix='poker-records-') as directory:
            path = Path(directory) / 'record.json'
            for variant in ('hulhe', 'leduc'):
                spec = get_spec('poker', {'variant': variant})
                for seed in range(10):
                    game = spec.create_game(seed=seed, position={'button': 1 - seed % 2})
                    while True:
                        save_record(spec, game, path)
                        restored = load_record(spec, path)
                        self.assertEqual(restored.history_keys(), game.history_keys())
                        self.assertEqual(restored.outcome, game.outcome)
                        if game.outcome.terminal:
                            break
                        sampled = spec.determinize(restored, game.actors()[0], np.random.default_rng(7))
                        with self.assertRaisesRegex(ValueError, 'seed'):
                            save_record(spec, sampled, path)
                        sampled.close()
                        restored.close()
                        game.play(1)
                    restored.close()
                with self.assertRaises(ValueError):
                    get_spec('poker', {'variant': 'hulhe' if variant == 'leduc' else 'leduc'}).create_game(
                        rules=spec.rules_dict(game))

    def test_baseline_legality_nonmutation_and_cancellation(self):
        from hasami.agents import AgentCancelled, SearchLimits
        for name in self.spec.default_baselines:
            agent, _ = self.spec.baseline_agent(name, seed=3)
            game = self.spec.create_game(seed=3)
            while not game.outcome.terminal:
                before = game.history_keys()
                decision = agent.choose(game, SearchLimits())
                self.assertIs(type(decision.action), int)
                self.assertIn(decision.action, game.legal_actions())
                self.assertEqual(game.history_keys(), before)
                game.play(decision.action)
            with self.assertRaises(ValueError):
                agent.choose(game, SearchLimits())
            event = threading.Event()
            event.set()
            with self.assertRaises(AgentCancelled):
                agent.choose(self.spec.create_game(seed=0), SearchLimits(), event)

    def test_duplicate_evaluation_arithmetic(self):
        from poker.evaluation import summarize_duplicates, evaluate_duplicate
        from poker.baselines import BaselinePolicy
        report = summarize_duplicates([(4, -2), (8, 2)])
        self.assertEqual(report['mbb_per_hand'], 1500)
        self.assertEqual(report['hands'], 4)
        self.assertEqual(report['ci95'], [-460, 3460])
        call = BaselinePolicy('call')
        self.assertEqual(evaluate_duplicate(call, call, range(10))['mbb_per_hand'], 0)


@requires_torch
class PokerLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        from hasami_learn.config import TrainingConfig
        from hasami_learn.trainer import train_from_config
        cls.addClassCleanup(random.setstate, random.getstate())
        cls.addClassCleanup(np.random.set_state, np.random.get_state())
        cls.addClassCleanup(torch.set_rng_state, torch.get_rng_state())
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        temp = tempfile.TemporaryDirectory(prefix='poker-learning-')
        cls.addClassCleanup(temp.cleanup)
        cls.root = Path(temp.name)
        cls.checkpoints = {}
        for variant, name in (('hulhe', 'poker'), ('leduc', 'leduc')):
            raw = json.loads((ROOT / f'configs/tiny/{name}-ppo-league.json').read_text())
            raw['run']['output_dir'] = str(cls.root)
            cls.checkpoints[variant] = train_from_config(TrainingConfig.from_dict(raw), echo=False)

    def test_checkpoint_plays_complete_legal_hands_and_variants_do_not_mix(self):
        from hasami.agents import SearchLimits
        from hasami_learn.inference import create_agent
        from hasami_learn.games import get_spec
        from hasami_learn.manifest import check_manifest_matches_spec
        for variant, checkpoint in self.checkpoints.items():
            agent = create_agent(str(checkpoint), device='cpu')
            for seed in range(8):
                game = agent.spec.create_game(seed=seed)
                while not game.outcome.terminal:
                    before = game.history_keys()
                    decision = agent.choose(game, SearchLimits(), player=game.actors()[0])
                    self.assertIs(type(decision.action), int)
                    self.assertIn(decision.action, game.legal_actions())
                    self.assertEqual(before, game.history_keys())
                    game.play(decision.action)
                self.assertLessEqual(len(game.moves), agent.spec.default_max_plies)
            other = get_spec('poker', {'variant': 'leduc' if variant == 'hulhe' else 'hulhe'})
            with self.assertRaises(ValueError):
                check_manifest_matches_spec(agent.manifest, other)
            metrics = [json.loads(row) for row in (checkpoint.parents[1] / 'metrics.jsonl').read_text().splitlines()]
            collected = [row for row in metrics if row['kind'] == 'collect']
            self.assertTrue(collected)
            self.assertTrue(all(row['truncated'] == 0 for row in collected))
            self.assertTrue(all({t['learner_seat'] for t in row['extra']['tables']} == {0, 1}
                                for row in collected))

    def test_leduc_selfplay_reduces_exact_exploitability_below_uniform(self):
        from hasami_learn.inference import create_agent
        from hasami_learn.games.poker import observation_policy
        from poker import leduc_exploitability
        from poker.baselines import BaselinePolicy
        agent = create_agent(str(self.checkpoints['leduc']), device='cpu')
        uniform = leduc_exploitability(BaselinePolicy('random'))
        measured = leduc_exploitability(observation_policy(agent))
        self.assertGreaterEqual(measured, -1e-12)
        self.assertLess(measured, uniform)
        print(f'LEDUC_SMOKE exploitability_chips={measured:.9f} uniform={uniform:.9f}', flush=True)

    def test_method_a_constructed_decision_all_advice_fields(self):
        from hasami_learn.practical import Advice, create_advisor
        from hasami_learn.practical_general import GeneralSearchAdvisor
        advisor = create_advisor(str(self.checkpoints['hulhe']), 'cpu')
        self.assertIsInstance(advisor, GeneralSearchAdvisor)
        game = advisor.spec.create_game(seed=15)
        game.play(1)
        game.play(1)
        game.play(2)
        player = game.actors()[0]
        before = game.history_keys()
        rows = advisor.advise(game, 3, player=player, seed=36, guard=2, top_k=3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows, advisor.advise(game, 3, player=player, seed=36, guard=2, top_k=3))
        self.assertEqual(game.history_keys(), before)
        for row in rows:
            self.assertIsInstance(row, Advice)
            self.assertIs(type(row.action), int)
            self.assertIn(row.action, game.legal_actions())
            self.assertEqual(row.move, row.action)
            self.assertTrue(row.text)
            self.assertTrue(0 <= row.practical_score <= 1)
            self.assertTrue(-1 <= row.engine_value <= 1)
            self.assertTrue(0 <= row.value_loss <= 2)
            self.assertTrue(0 <= row.difficulty <= 1)
            self.assertTrue(0 <= row.naturalness <= 1)
            self.assertIs(type(row.guard_fallback), bool)


if __name__ == '__main__':
    unittest.main()
