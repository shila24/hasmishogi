"""Information-set consistency, seeded records and engine dependency boundaries."""

import ast
import copy
from pathlib import Path
import random
import sys
import tempfile
import unittest

from poker import create_game
from poker.encoding import encode
from poker.records import from_record, load_record, record_dict, save_record


class HiddenAndRecordTests(unittest.TestCase):
    def test_seed_dealing_order_and_copy_isolation(self):
        for seed in range(100):
            game = create_game(seed=seed)
            deck = list(range(52))
            random.Random(seed).shuffle(deck)
            self.assertEqual(game.observation(game.button)['hole_cards'], deck[:4:2])
            self.assertEqual(game.observation(1 - game.button)['hole_cards'], deck[1:4:2])
            while not game.outcome.terminal:
                game.play(1)
            self.assertEqual(game.observation(0)['board'], deck[4:9])
            obs = game.observation(0)
            obs['board'].clear()
            obs['history'][0].clear()
            self.assertEqual(game.observation(0)['board'], deck[4:9])

    def test_redeal_all_streets_and_seeds_and_undo(self):
        for variant in ('hulhe', 'leduc'):
            game = create_game({'variant': variant}, seed=12)
            for street in range(game.rules.streets):
                for player in (0, 1):
                    obs, planes = game.observation(player), encode(game, player)
                    changed = set()
                    for seed in range(100):
                        work = game.clone()
                        work.redeal({'player': player}, seed)
                        self.assertEqual(work.observation(player), obs)
                        self.assertEqual(encode(work, player), planes)
                        changed.add(tuple(work.observation(1 - player)['hole_cards']))
                        self.assertEqual(sorted(work._deck), list(range(game.rules.deck_size)))
                        if work.moves:
                            action = work.moves[-1]
                            before = work.history_keys()
                            work.undo()
                            work.play(action)
                            self.assertEqual(work.history_keys(), before)
                    self.assertGreater(len(changed), 1)
                    self.assertEqual(game.observation(player), obs)
                game.play(1)
                game.play(1)

    def test_explicit_redeal_rejections_are_atomic(self):
        game = create_game(seed=1)
        game.play(1)
        game.play(1)
        holes = [game.observation(p)['hole_cards'] for p in (0, 1)]
        obs = game.observation(0)
        valid = dict(player=0, hole_cards=holes, board=obs['board'], deck=list(game._deck[7:]))
        work = game.clone()
        work.redeal(valid, 5)
        self.assertEqual(work.history_keys(), game.history_keys())
        for edit in ({'player': True}, {'player': 2}, {'board': []}, {'board': [0, 1, 2]},
                     {'hole_cards': [holes[1], holes[0]]}, {'hole_cards': [holes[0], holes[0]]},
                     {'hole_cards': [holes[0], [None, None]]}, {'deck': [None] * 45},
                     {'deck': [0] * 45}, {'deck': list(range(44))}, {'pot': 0}):
            with self.subTest(edit=edit):
                before = game.history_keys()
                with self.assertRaises(ValueError):
                    game.redeal({**valid, **edit}, 5)
                self.assertEqual(game.history_keys(), before)
        for seed in (-1, True, 1.0):
            with self.assertRaises(ValueError):
                game.redeal({'player': 0}, seed)
        while not game.outcome.terminal:
            game.play(1)
        with self.assertRaises(ValueError):
            game.redeal({'player': 0}, 1)

    def test_records_roundtrip_and_tampering(self):
        rng = random.Random(81)
        for variant in ('hulhe', 'leduc'):
            for seed in range(30):
                game = create_game({'variant': variant}, seed, button=1 - seed % 2)
                while True:
                    rebuilt = from_record(record_dict(game))
                    self.assertEqual(rebuilt.history_keys(), game.history_keys())
                    if game.outcome.terminal:
                        break
                    game.play(rng.choice(game.legal_actions()))
                record = record_dict(game)
                bad = copy.deepcopy(record)
                bad['outcome']['chips'][0] += 1
                with self.assertRaises(ValueError):
                    from_record(bad)
        work = create_game(seed=0)
        work.redeal({'player': 0}, 2)
        with self.assertRaises(ValueError):
            record_dict(work)

    def test_record_file_roundtrip_and_malformed_metadata(self):
        game = create_game(seed=8)
        game.play(0)
        with tempfile.TemporaryDirectory(prefix='poker-json-') as directory:
            path = Path(directory) / 'hand.json'
            self.assertEqual(save_record(game, path), path)
            restored = load_record(path)
            self.assertEqual(restored.history_keys(), game.history_keys())
            self.assertEqual(restored.outcome, game.outcome)
        valid = record_dict(game)
        invalid = [None, [], {}, {**valid, 'extra': 1}]
        invalid.extend({**valid, **edit} for edit in (
            {'version': True}, {'version': 2}, {'seed': None}, {'actions': 'fold'},
            {'actions': [True]}, {'ruleset_id': 'other'}, {'seed': -1}))
        for data in invalid:
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    from_record(data)

    def test_no_nonstandard_engine_imports(self):
        root = Path(__file__).resolve().parents[2] / 'poker'
        for path in root.glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    names = [alias.name.split('.')[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    names = [node.module.split('.')[0]]
                else:
                    continue
                self.assertTrue(all(n in sys.stdlib_module_names or n == 'poker' for n in names), (path, names))


if __name__ == '__main__':
    unittest.main()
