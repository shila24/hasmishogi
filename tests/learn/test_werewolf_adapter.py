"""Five-seat adapter, hidden-state constraints, training, and method-A integration."""

import copy
import json
from pathlib import Path
import random
import tempfile
import threading
import unittest

from tests.learn._common import requires_torch
from tests.werewolf_suite.test_game import fixture, finish_talk, execute
from werewolf.actions import target_action

ROOT = Path(__file__).resolve().parents[2]


@requires_torch
class WerewolfAdapterTests(unittest.TestCase):
    def setUp(self):
        from hasami_learn.games import get_spec
        self.spec = get_spec('werewolf')

    def test_contract_masks_and_validation(self):
        import numpy as np
        from hasami_learn.games.base import GameSpec, require_search_game
        self.assertIsInstance(self.spec, GameSpec)
        self.assertEqual((self.spec.num_players, self.spec.action_size, self.spec.board_shape), (5, 45, (5, 8)))
        self.assertFalse(self.spec.perfect_information or self.spec.zero_sum)
        self.assertTrue(self.spec.supports_undo and self.spec.has_chance)
        with self.assertRaises(ValueError):
            require_search_game(self.spec)
        game = self.spec.create_game('auto', seed=1)
        for p in range(5):
            self.assertEqual(list(np.flatnonzero(self.spec.legal_mask(game, p))), game.legal_actions(p))
            planes = self.spec.observe(game, p)
            self.assertEqual(planes.shape, (32, 5, 8))
            self.assertEqual(planes.dtype, np.float32)
            self.assertTrue(np.isfinite(planes).all())
            self.assertTrue(((planes >= 0) & (planes <= 1)).all())
        with self.assertRaises(ValueError):
            self.spec.make_encoder().encode(game)
        with self.assertRaises(ValueError):
            self.spec.create_game('cpp')
        with self.assertRaises(ValueError):
            self.spec.create_game(position={'day': 2})
        with self.assertRaises(ValueError):
            self.spec.make_encoder('other')
        with self.assertRaises(ValueError):
            self.spec.make_encoder(extra=1)
        with self.assertRaises(ValueError):
            self.spec.terminal_values(game)
        before = copy.deepcopy(game.__dict__)
        result = self.spec.step(game, {p: 0 for p in range(5)})
        self.assertFalse(result.terminal or result.truncated)
        self.assertIsNone(result.rewards)
        game.undo()
        self.assertEqual(before, game.__dict__)

    def test_encoder_public_claims_relative_rows_and_no_hidden_leaks(self):
        import numpy as np
        game = fixture()
        game.play_joint({0: 2, 1: target_action('DIVINED', 1, 3, 'werewolf'),
                         2: target_action('ESTIMATE', 2, 3, 'possessed'),
                         3: target_action('VOTE_INTENT', 3, 1), 4: 1})
        planes = self.spec.observe(game, 2)
        self.assertTrue(np.all(planes[1] == 1))
        self.assertTrue(np.all(planes[13, 3] == 1))
        self.assertEqual(planes[19, 4, 1], 1)
        self.assertTrue(np.all(planes[27, 1] == 0.25))
        self.assertEqual(planes[28, 1, 4], 1)
        self.assertTrue(np.all(planes[31] == 1))
        execute(game, 0)
        planes = self.spec.observe(game, 2)
        self.assertTrue(np.all(planes[4, 3] == 0))
        self.assertTrue(np.all(planes[30, 3] == 1))
        self.assertTrue(np.all(planes[10] == 1))
        self.assertEqual(planes[29, 4, 3], 1)
        for observer in range(5):
            before = self.spec.observe(game, observer)
            for seed in range(20):
                world = self.spec.determinize(game, observer, np.random.default_rng(seed))
                np.testing.assert_array_equal(before, self.spec.observe(world, observer))

    def test_determinize_many_seeds_every_phase_and_private_facts(self):
        import numpy as np
        from werewolf.game import role_class
        count = 0
        for seed in range(12):
            game = self.spec.create_game(seed=seed)
            rng = random.Random(seed)
            while True:
                before = copy.deepcopy(game.__dict__)
                for player in range(5):
                    for sample in range(5):
                        world = self.spec.determinize(game, player, np.random.default_rng([seed, sample]))
                        self.assertIsNot(world, game)
                        self.assertEqual(world.observation(player), game.observation(player))
                        self.assertEqual(world.public_log, game.public_log)
                        self.assertEqual(world.roles[player], game.roles[player])
                        self.assertEqual(world.outcome, game.outcome)
                        for result in game.observation(player)['results']:
                            self.assertEqual(role_class(world.roles[result['target']]), result['result'])
                        if not world.outcome.terminal:
                            initial = copy.deepcopy(world.__dict__)
                            self.spec.step(world, {p: world.legal_actions(p)[0] for p in world.actors()})
                            world.undo()
                            self.assertEqual(world.__dict__, initial)
                        count += 1
                self.assertEqual(before, game.__dict__)
                if game.outcome.terminal:
                    break
                self.spec.step(game, {p: rng.choice(game.legal_actions(p)) for p in game.actors()})
        self.assertGreater(count, 1500)

    def test_uniform_assignments_and_hidden_input_independence(self):
        import numpy as np
        from collections import Counter
        game = fixture()
        player = 0
        assignments = game.role_assignments(player)
        counts = Counter()
        for seed in range(1200):
            world = self.spec.determinize(game, player, np.random.default_rng(seed))
            counts[tuple(world.roles)] += 1
        self.assertEqual(set(counts), set(assignments))
        expected = 1200 / len(assignments)
        self.assertTrue(all(abs(n - expected) < expected * 0.6 for n in counts.values()))
        alternate = game.clone()
        alternate.redeal(list(assignments[0]), 123, player=player)
        self.assertEqual(alternate.observation(player), game.observation(player))
        for seed in range(20):
            a = self.spec.determinize(game, player, np.random.default_rng(seed))
            b = self.spec.determinize(alternate, player, np.random.default_rng(seed))
            self.assertEqual(a.__dict__, b.__dict__)

    def test_future_chance_is_redrawn_and_past_ties_preserved(self):
        import numpy as np
        game = self.spec.create_game(seed=0)
        finish_talk(game)
        joint = {p: target_action('vote', p, (p + 1) % 5) for p in range(5)}
        game.play_joint(joint)
        seen = set()
        for seed in range(40):
            world = self.spec.determinize(game, 0, np.random.default_rng(seed))
            world.play_joint(joint)
            seen.add(world.public_log[-1]['target'])
        self.assertEqual(seen, set(range(5)))
        game.play_joint(joint)
        for player in range(5):
            for seed in range(10):
                world = self.spec.determinize(game, player, np.random.default_rng(seed))
                self.assertEqual(world.observation(player), game.observation(player))

    def test_team_rewards_draws_and_training_caps_are_distinct(self):
        from hasami_learn.selfplay import run_table_game
        for target, winner in ((3, 'village'), (1, 'werewolf')):
            game = fixture()
            if target != 3:
                execute(game, 0)
                game.play_joint({2: target_action('divine', 2, 3), 3: target_action('attack', 3, 2)})
            finish_talk(game)
            result = self.spec.step(game, {p: target_action('vote', p, target if p != target else next(
                t for t in game.actors() if t != p))
                                          for p in game.actors()})
            self.assertTrue(result.terminal)
            self.assertFalse(result.truncated)
            self.assertEqual(game.outcome.winner_team, winner)
            expected = [1, 1, 1, -1, -1] if winner == 'village' else [-1, -1, -1, 1, 1]
            self.assertEqual(result.rewards, expected)
            for p in range(5):
                self.assertEqual(self.spec.outcome_score(result.rewards, p), (expected[p] + 1) / 2)
        game = fixture(max_days=1)
        execute(game, 0)
        result = self.spec.step(game, {2: target_action('divine', 2, 3), 3: target_action('attack', 3, 1)})
        self.assertEqual(result.rewards, [0] * 5)
        self.assertTrue(result.terminal)
        self.assertFalse(result.truncated)
        agents = {p: self.spec.baseline_agent('random', seed=p)[0] for p in range(5)}
        trajectory = run_table_game(agents, self.spec.make_encoder(), spec=self.spec, max_plies=1, seed=0)
        self.assertTrue(trajectory.truncated)
        self.assertNotEqual(trajectory.reason, 'max_days')

    def test_records_roundtrip_terminal_and_partial_and_reject_tampering(self):
        from hasami_learn.games.records import load_record, save_record
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'game.json'
            for seed in range(5):
                game = self.spec.create_game(seed=seed)
                rng = random.Random(seed)
                while True:
                    save_record(self.spec, game, path)
                    loaded = load_record(self.spec, path)
                    self.assertEqual(game.history_keys(), loaded.history_keys())
                    self.assertEqual(game.moves, loaded.moves)
                    self.assertEqual(game.outcome, loaded.outcome)
                    if game.outcome.terminal:
                        break
                    game.play_joint({p: rng.choice(game.legal_actions(p)) for p in game.actors()})
                data = json.loads(path.read_text())
                data['joint_actions'][0]['0'] = 45
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    load_record(self.spec, path)
            world = self.spec.determinize(game, 0, np.random.default_rng(1))
            with self.assertRaisesRegex(ValueError, 'seed'):
                save_record(self.spec, world, path)

    def test_baseline_agent_contract_and_evaluation_arithmetic(self):
        from hasami.agents import AgentCancelled
        from werewolf.evaluation import summarize, wilson_interval
        agent, _ = self.spec.baseline_agent('simple', 2)
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(AgentCancelled):
            agent.choose(fixture(), cancel=cancelled, player=0)
        self.assertEqual(summarize([1, 1, -1, 0])['mean_outcome_score'], 0.625)
        self.assertAlmostEqual(wilson_interval(5, 10)[0], 0.2365896, places=6)
        result = self.spec.strength_table({1: agent}, games=10, seed=10)
        for anchor in ('random', 'simple'):
            table = result['levels']['1'][anchor]
            self.assertEqual(table['games'], 10)
            self.assertEqual(table['by_role']['villager']['games'], 4)
            for role in ('seer', 'werewolf', 'possessed'):
                self.assertEqual(table['by_role'][role]['games'], 2)
            self.assertEqual(table['by_team']['village']['games'], 6)
            self.assertEqual(table['by_team']['werewolf']['games'], 4)
            self.assertEqual([row['games'] for row in table['by_seat'].values()], [2] * 5)
            self.assertEqual(table['wins'], sum(row['value'] == 1 for row in table['tables']))
            self.assertTrue(0 <= table['ci_low'] <= table['win_rate'] <= table['ci_high'] <= 1)
        with self.assertRaises(ValueError):
            self.spec.strength_table({1: agent}, games=6)


@requires_torch
class WerewolfTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from hasami_learn.config import TrainingConfig
        from hasami_learn.trainer import train_from_config
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        tmp = tempfile.TemporaryDirectory(prefix='werewolf-training-')
        cls.addClassCleanup(tmp.cleanup)
        cls.root = Path(tmp.name)
        cfg = TrainingConfig.load(ROOT / 'configs/tiny/werewolf-ppo-league.json')
        cfg.run.output_dir = str(cls.root)
        cls.checkpoint = train_from_config(cfg, echo=False)

    def test_tiny_ppo_checkpoint_plays_every_phase_legally(self):
        from hasami.agents import SearchLimits
        from hasami_learn.games import get_spec
        from hasami_learn.inference import create_agent
        spec = get_spec('werewolf')
        agent = create_agent(str(self.checkpoint), device='cpu')
        phases = set()
        for seed in range(10):
            game = spec.create_game(seed=seed)
            while not game.outcome.terminal:
                phases.add(game.phase)
                before = copy.deepcopy(game.__dict__)
                actions = {p: agent.choose(game, SearchLimits(), player=p).action for p in game.actors()}
                self.assertEqual(before, game.__dict__)
                self.assertTrue(all(type(a) is int and a in game.legal_actions(p) for p, a in actions.items()))
                spec.step(game, actions)
        self.assertTrue({'talk', 'vote', 'night'} <= phases)
        metrics = [json.loads(line) for line in (self.root / 'tiny-werewolf-ppo-league/metrics.jsonl').read_text().splitlines()]
        collect = next(row for row in metrics if row['kind'] == 'collect')
        self.assertEqual(collect['games'], 5)
        self.assertEqual(collect['truncated'], 0)
        self.assertEqual([row['learner_seat'] for row in collect['extra']['tables']], list(range(5)))
        update = next(row for row in metrics if row['kind'] == 'update')
        self.assertGreater(update['steps'], 0)

    def test_method_a_hint_on_constructed_day(self):
        from hasami_learn.practical import Advice, create_advisor
        from hasami_learn.practical_general import GeneralSearchAdvisor
        advisor = create_advisor(str(self.checkpoint), 'cpu')
        self.assertIsInstance(advisor, GeneralSearchAdvisor)
        game = fixture()
        game.play_joint({p: 2 if p == 2 else 0 for p in range(5)})
        # A vote decision on the constructed day keeps the integration test small.
        finish_talk(game)
        before = copy.deepcopy(game.__dict__)
        rows = advisor.advise(game, 3, player=2, seed=42)
        self.assertTrue(rows)
        self.assertEqual(game.__dict__, before)
        self.assertEqual(rows, advisor.advise(game, 3, player=2, seed=42))
        for row in rows:
            self.assertIsInstance(row, Advice)
            self.assertIn(row.action, game.legal_actions(2))
            self.assertTrue(0 <= row.practical_score <= 1)
            self.assertTrue(-1 <= row.engine_value <= 1)
            self.assertTrue(0 <= row.value_loss <= 2)
            self.assertTrue(row.difficulty is None or 0 <= row.difficulty <= 1)
            self.assertTrue(0 <= row.naturalness <= 1)
        self.assertEqual(advisor.last_search['determinizations'], 4)
        advisor.determinizations = advisor.rollouts = 1
        talk = fixture()
        night = fixture()
        execute(night, 0)
        for state, player in ((talk, 0), (night, 2), (night, 3)):
            before = copy.deepcopy(state.__dict__)
            hints = advisor.advise(state, 3, player=player, seed=11)
            self.assertTrue(hints)
            self.assertEqual(state.__dict__, before)
            for hint in hints:
                self.assertIn(hint.action, state.legal_actions(player))
                self.assertEqual(hint.move, hint.action)
                self.assertTrue(hint.text)
                self.assertIs(type(hint.guard_fallback), bool)
                self.assertTrue(0 <= hint.practical_score <= 1)
                self.assertTrue(-1 <= hint.engine_value <= 1)
                self.assertTrue(0 <= hint.value_loss <= 2)
                self.assertTrue(hint.difficulty is None or 0 <= hint.difficulty <= 1)
                self.assertTrue(0 <= hint.naturalness <= 1)
