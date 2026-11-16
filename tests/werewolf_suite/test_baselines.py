"""Baseline legality, truthful seer reports, deception, and information isolation."""

import copy
import threading
import unittest

from werewolf.baselines import BaselineAgent
from werewolf.actions import decode, target_action
from tests.werewolf_suite.test_game import fixture, execute


class BaselineTests(unittest.TestCase):
    def test_seer_comes_out_then_reports_truthfully(self):
        game = fixture()
        agent = BaselineAgent('simple', 1)
        self.assertEqual(agent.choose(game, player=2).action, 2)
        game.play_joint({p: 2 if p == 2 else 0 for p in game.actors()})
        result = game.observation(2)['results'][0]
        action = agent.choose(game, player=2).action
        self.assertEqual(decode(action, 2), {'kind': 'DIVINED', 'target': result['target'], 'result': result['result']})

    def test_wolf_deception_half_and_possessed_fake_report(self):
        game = fixture()
        claims = sum(BaselineAgent('simple', seed).choose(game, player=3).action == 2 for seed in range(1000))
        self.assertTrue(450 < claims < 550)
        agent = BaselineAgent('simple', 0)
        self.assertEqual(agent.choose(game, player=4).action, 2)
        game.play_joint({p: 2 if p == 4 else 0 for p in game.actors()})
        statement = decode(agent.choose(game, player=4).action, 4)
        self.assertEqual((statement['kind'], statement['result']), ('DIVINED', 'werewolf'))

    def test_votes_reported_wolf_and_attacks_claimed_seer(self):
        game = fixture(talk_turns=1)
        game.play_joint({p: target_action('DIVINED', p, 3, 'werewolf') if p in (0, 2) else 0
                         for p in game.actors()})
        for p in (0, 1, 2, 4):
            self.assertEqual(decode(BaselineAgent('simple').choose(game, player=p).action, p)['target'], 3)
        game = fixture(talk_turns=1)
        game.play_joint({p: 2 if p == 2 else 0 for p in game.actors()})
        execute(game, 0)
        action = BaselineAgent('simple').choose(game, player=3).action
        self.assertEqual(decode(action, 3)['target'], 2)

    def test_observation_only_same_behavior_and_legal_full_games(self):
        class View:
            def __init__(self, observation):
                self.obs = observation

            def observation(self, player):
                return copy.deepcopy(self.obs)

        for name in ('random', 'simple'):
            for seed in range(20):
                from werewolf import create_game
                game = create_game(seed=seed)
                agents = [BaselineAgent(name, seed * 5 + p) for p in range(5)]
                while not game.outcome.terminal:
                    actions = {}
                    before = game.clone().__dict__
                    for p in game.actors():
                        actions[p] = agents[p].choose(View(game.observation(p)), player=p).action
                        self.assertIn(actions[p], game.legal_actions(p))
                    self.assertEqual(game.__dict__, before)
                    game.play_joint(actions)
                with self.assertRaises(ValueError):
                    agents[0].choose(game, player=0)
        event = threading.Event()
        event.set()
        with self.assertRaises(InterruptedError):
            BaselineAgent().choose(fixture(), cancel=event, player=0)
