"""Rule and information-boundary regression tests."""

import random
import unittest

from werewolf import create_game, replay
from werewolf.actions import target_action, decode


class GameTests(unittest.TestCase):
    def test_seed_and_night_zero(self):
        for seed in range(100):
            rng = random.Random(seed)
            expected = ['villager', 'villager', 'seer', 'werewolf', 'possessed']
            rng.shuffle(expected)
            game = create_game(seed=seed)
            self.assertEqual(game.roles, expected)
            seer = expected.index('seer')
            target = rng.choice([p for p in range(5) if p != seer])
            for player in range(5):
                obs = game.observation(player)
                self.assertEqual(obs['role'], expected[player])
                self.assertNotIn('roles', obs)
                if player == seer:
                    self.assertEqual(obs['results'], [{'day': 0, 'target': target,
                        'result': 'werewolf' if expected[target] == 'werewolf' else 'human'}])
                else:
                    self.assertEqual(obs['results'], [])

    def test_joint_talk_and_undo(self):
        game = create_game(seed=3)
        before = [game.observation(p) for p in range(5)]
        with self.assertRaises(ValueError):
            game.play_joint({0: 0})
        self.assertEqual(before, [game.observation(p) for p in range(5)])
        game.play_joint({p: p + 1 if p < 4 else 0 for p in reversed(range(5))})
        self.assertEqual([e['player'] for e in game.public_log], list(range(5)))
        self.assertEqual(game.talk_turn, 1)
        game.undo()
        self.assertEqual(before, [game.observation(p) for p in range(5)])
        for _ in range(3):
            game.play_joint({p: 0 for p in game.actors()})
        self.assertEqual(game.phase, 'vote')

    def test_relative_targets(self):
        for player in range(5):
            for target in range(5):
                if target == player:
                    continue
                for kind in ('VOTE_INTENT', 'vote', 'attack', 'divine'):
                    action = target_action(kind, player, target)
                    self.assertEqual(decode(action, player)['target'], target)

    def test_random_replay_clone_and_exact_undo(self):
        for seed in range(30):
            game = create_game(seed=seed)
            rng = random.Random(seed)
            while not game.outcome.terminal:
                before = game.clone()
                joint = {p: rng.choice(game.legal_actions(p)) for p in game.actors()}
                game.play_joint(joint)
                after = game.clone()
                game.undo()
                self.assertEqual(game.history_keys(), before.history_keys())
                self.assertEqual(game.__dict__, before.__dict__)
                game.play_joint(joint)
                self.assertEqual(game.history_keys(), after.history_keys())
            restored = replay(game.rules, game.seed, game.moves)
            self.assertEqual(game.history_keys(), restored.history_keys())
            self.assertEqual(game.outcome, restored.outcome)
            self.assertEqual(game.public_log, restored.public_log)
            self.assertEqual(game.actors(), [])
            for p in range(5):
                self.assertEqual(game.legal_actions(p), [])


FIXED_ROLES = ['villager', 'villager', 'seer', 'werewolf', 'possessed']


def fixture(**rules):
    game = create_game(rules, seed=0)
    game.redeal(FIXED_ROLES, 7)
    return game


def finish_talk(game):
    while game.phase == 'talk':
        game.play_joint({p: 0 for p in game.actors()})


def execute(game, target):
    finish_talk(game)
    game.play_joint({p: target_action('vote', p, target if p != target else next(
        t for t in game.actors() if t != p)) for p in game.actors()})


class RuleBoundaryTests(unittest.TestCase):
    def test_execution_village_win_and_no_role_revelation(self):
        game = fixture()
        execute(game, 3)
        self.assertEqual((game.phase, game.outcome.winner_team, game.outcome.reason),
                         ('terminal', 'village', 'werewolf_executed'))
        event = game.public_log[-1]
        self.assertEqual(event, {'kind': 'execution', 'day': 1, 'target': 3})
        with self.assertRaises(ValueError):
            game.play_joint({})

    def test_night_is_simultaneous_and_attacked_seer_gets_result(self):
        game = fixture()
        execute(game, 0)
        self.assertEqual(game.actors(), [2, 3])
        game.play_joint({3: target_action('attack', 3, 2), 2: target_action('divine', 2, 3)})
        self.assertEqual(game.observation(2)['results'][-1], {'day': 1, 'target': 3, 'result': 'werewolf'})
        self.assertFalse(game.alive[2])
        self.assertEqual(game.day, 2)
        self.assertEqual(game.legal_actions(2), [])
        self.assertEqual(game.public_log[-1], {'kind': 'attack', 'day': 1, 'target': 2})
        self.assertNotIn('divine', [e['kind'] for e in game.public_log])
        execute(game, 1)
        self.assertEqual(game.outcome.reason, 'humans_outnumbered')
        self.assertEqual(game.outcome.winner_team, 'werewolf')
        self.assertTrue(game.alive[4])  # The possessed counts as a living human.

    def test_attack_possessed_is_legal_and_dead_seer_never_acts(self):
        game = fixture()
        execute(game, 2)
        self.assertEqual(game.actors(), [3])
        self.assertIn(target_action('attack', 3, 4), game.legal_actions(3))
        game.play_joint({3: target_action('attack', 3, 4)})
        self.assertEqual(game.actors(), [0, 1, 3])

    def test_day_cap_and_victory_precedence(self):
        game = fixture(max_days=1)
        execute(game, 0)
        game.play_joint({2: target_action('divine', 2, 3), 3: target_action('attack', 3, 1)})
        self.assertEqual((game.outcome.reason, game.outcome.winner_team), ('max_days', None))
        self.assertEqual(game.outcome.winner, -1)
        game = fixture(max_days=1)
        execute(game, 3)
        self.assertEqual(game.outcome.reason, 'werewolf_executed')

    def test_revote_then_random_tie_is_seeded_and_undo_restores_chance(self):
        executed = set()
        for seed in range(60):
            game = create_game({'talk_turns': 1}, seed)
            finish_talk(game)
            joint = {p: target_action('vote', p, (p + 1) % 5) for p in range(5)}
            game.play_joint(joint)
            self.assertEqual(game.phase, 'revote')
            self.assertEqual(game.tied, list(range(5)))
            before = game.clone()
            game.play_joint(joint)
            target = game.public_log[-1]['target']
            executed.add(target)
            history = game.history_keys()
            game.undo()
            self.assertEqual(game.__dict__, before.__dict__)
            game.play_joint(joint)
            self.assertEqual(game.public_log[-1]['target'], target)
            self.assertEqual(game.history_keys(), history)
        self.assertEqual(executed, set(range(5)))

    def test_revote_candidates_and_self_vote_exclusion(self):
        game = fixture()
        finish_talk(game)
        targets = [1, 0, 0, 1, 2]
        game.play_joint({p: target_action('vote', p, t) for p, t in enumerate(targets)})
        self.assertEqual(game.tied, [0, 1])
        for p in game.actors():
            self.assertEqual({decode(a, p)['target'] for a in game.legal_actions(p)}, {0, 1} - {p})
        game.play_joint({p: target_action('vote', p, 1 if p == 0 else 0) for p in game.actors()})
        self.assertFalse(game.alive[0])
        self.assertEqual(game.phase, 'night')

    def test_all_phases_reject_illegal_joint_actions_atomically(self):
        game = fixture()
        states = [game.clone()]
        execute(game, 0)
        states.append(game.clone())
        game.play_joint({2: target_action('divine', 2, 3), 3: target_action('attack', 3, 1)})
        states.append(game.clone())
        finish_talk(game)
        states.append(game.clone())
        for game in states:
            legal = {p: game.legal_actions(p)[0] for p in game.actors()}
            p = game.actors()[0]
            for bad in (-1, 45, True, 1.5, '0'):
                before = game.__dict__.copy()
                with self.assertRaises(ValueError):
                    game.play_joint({**legal, p: bad})
                self.assertEqual(game.__dict__, before)
            for bad in ({}, {**legal, 8: 0}, {str(k): v for k, v in legal.items()}):
                with self.assertRaises(ValueError):
                    game.play_joint(bad)
            for player in range(5):
                if not game.alive[player]:
                    self.assertEqual(game.legal_actions(player), [])
            if game.phase == 'talk':
                for player in game.actors():
                    for a in range(5, 29):
                        self.assertIn(a, game.legal_actions(player))
                    for a in range(29, 33):
                        self.assertEqual(a in game.legal_actions(player), game.alive[decode(a, player)['target']])

    def test_validation_and_observation_ownership(self):
        for rules in ({'talk_turns': 0}, {'max_days': True}, {'ruleset_id': 'wrong'}):
            with self.assertRaises(ValueError):
                create_game(rules)
        for seed in (True, '0', 1.0):
            with self.assertRaises(ValueError):
                create_game(seed=seed)
        game = fixture()
        for player in (-1, 5, True, '0'):
            with self.assertRaises(ValueError):
                game.observation(player)
        with self.assertRaises(ValueError):
            game.undo()
        obs = game.observation(2)
        obs['results'].clear()
        obs['alive'].clear()
        self.assertEqual(len(game.observation(2)['results']), 1)
        self.assertEqual(sum(game.alive), 5)

    def test_redeal_private_public_facts_and_claims_are_not_truth(self):
        game = fixture()
        game.play_joint({0: 4, 1: 3, 2: 0, 3: 2, 4: 2})
        observer = 0
        assignments = game.role_assignments(observer)
        self.assertGreater(len(assignments), 1)
        for roles in assignments:
            clone = game.clone()
            clone.redeal(list(roles), 123, player=observer)
            self.assertEqual(clone.observation(observer), game.observation(observer))
        execute(game, 0)
        game.play_joint({2: target_action('divine', 2, 3), 3: target_action('attack', 3, 1)})
        for observer in range(5):
            before = game.observation(observer)
            for roles in game.role_assignments(observer):
                clone = game.clone()
                clone.redeal(list(roles), 7, player=observer)
                self.assertEqual(before, clone.observation(observer))
                self.assertNotEqual(roles[0], 'werewolf')
                self.assertNotEqual(roles[1], 'werewolf')
        before = game.clone().__dict__
        bad = ['werewolf', 'villager', 'seer', 'villager', 'possessed']
        with self.assertRaises(ValueError):
            game.redeal(bad, 1, player=4)
        self.assertEqual(game.__dict__, before)
        with self.assertRaises(ValueError):
            game.redeal(['villager'] * 5, 1)
        bad = ['villager', 'villager', 'seer', 'possessed', 'werewolf']
        with self.assertRaises(ValueError):
            game.redeal(bad, 1, player=2)
        with self.assertRaises(ValueError):
            game.redeal(FIXED_ROLES, 1, player=True)

    def test_dependency_free_imports(self):
        import ast
        from pathlib import Path
        import sys
        for path in (Path(__file__).resolve().parents[2] / 'werewolf').glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    names = [a.name.split('.')[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    names = [node.module.split('.')[0]]
                else:
                    continue
                self.assertTrue(set(names) <= sys.stdlib_module_names | {'werewolf'}, (path, names))
