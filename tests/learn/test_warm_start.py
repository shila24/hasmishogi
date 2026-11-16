"""Weights-only initialization and checkpoint lineage contracts."""
from __future__ import annotations

import json
from unittest.mock import patch
from tests.learn._common import LearnTestCase, requires_torch


@requires_torch
class WarmStartTests(LearnTestCase):
    def test_fresh_weights_only_initialization(self):
        import torch
        from hasami_learn.config import tiny_config
        from hasami_learn.trainer import Trainer
        config = tiny_config('alphazero', simulations=2).with_overrides(run__output_dir=str(self.tmp))
        source = Trainer(config, echo=False)
        self.addCleanup(source.close)
        with torch.no_grad():
            for parameter in source.network.parameters():
                parameter.fill_(0.125)
        checkpoint = source.checkpoint(7)
        target = Trainer(config.with_overrides(run__run_id='warm', run__init_checkpoint=str(checkpoint)), echo=False)
        self.addCleanup(target.close)
        self.assertEqual(target.iteration, 0)
        self.assertEqual(len(target.replay), 0)
        self.assertEqual(target.learner.optimizer.state_dict()['state'], {})
        for name, tensor in source.network.state_dict().items():
            self.assertTrue(torch.equal(tensor, target.network.state_dict()[name]), name)
        self.assertEqual(target.manifest(0).init_checkpoint['iteration'], 7)

    def source(self, algorithm='alphazero', game='hasami'):
        from hasami_learn.trainer import Trainer
        params = {'simulations': 2} if algorithm == 'alphazero' else {}
        config = self.tiny(algorithm, **params).with_overrides(
            game__name=game, run__run_id=f'source-{algorithm}-{game}',
            network__params={'hidden': 8, 'layers': 1}, selfplay__games_per_iteration=1,
            selfplay__max_plies=6, replay__min_size=1, replay__batch_size=2,
            optimizer__updates_per_iteration=1)
        trainer = Trainer(config, echo=False)
        self.addCleanup(trainer.close)
        trainer.iteration = 3
        return trainer, trainer.checkpoint(3)

    def test_all_manifest_mismatch_classes_name_the_field(self):
        from hasami_learn.trainer import Trainer
        from hasami_learn.manifest import ManifestError
        source, checkpoint = self.source()
        original = json.loads((checkpoint / 'manifest.json').read_text())
        mutations = {'game_id': 'shogi', 'game_params': {'different': True}, 'ruleset_id': 'other-v1',
                     'rules': {**original['rules'], 'edge_capture': False}, 'board_shape': [4, 4],
                     'action_size': 42, 'action_encoding': 'other-actions', 'encoding_id': 'other-encoder',
                     'encoder_params': {'k': 2}, 'num_planes': 10, 'num_players': 4,
                     'network.name': 'resnet', 'network.params': {'hidden': 9, 'layers': 1},
                     'network.head': 'q', 'value_perspective': 'black'}
        for field, value in mutations.items():
            with self.subTest(field=field):
                manifest = json.loads(json.dumps(original))
                location = manifest
                parts = field.split('.')
                for key in parts[:-1]:
                    location = location[key]
                location[parts[-1]] = value
                (checkpoint / 'manifest.json').write_text(json.dumps(manifest))
                config = source.config.with_overrides(run__run_id='mismatch', run__init_checkpoint=str(checkpoint))
                with self.assertRaisesRegex(ManifestError, field):
                    Trainer(config, echo=False)
        (checkpoint / 'manifest.json').write_text(json.dumps(original))

    def test_value_perspective_rejected_before_weights_are_read(self):
        import torch
        from hasami_learn.manifest import ManifestError
        source, checkpoint = self.source()
        manifest = json.loads((checkpoint / 'manifest.json').read_text())
        manifest['value_perspective'] = 'black'
        (checkpoint / 'manifest.json').write_text(json.dumps(manifest))
        original = {name: tensor.clone() for name, tensor in source.network.state_dict().items()}
        source.config.run.init_checkpoint = str(checkpoint)
        with patch('hasami_learn.manifest.torch.load', wraps=torch.load) as load:
            with self.assertRaisesRegex(ManifestError, 'value_perspective'):
                source._warm_start(source.learner.required_head)
            load.assert_not_called()
        self.assertIsNone(source.init_checkpoint)
        for name, tensor in source.network.state_dict().items():
            self.assertTrue(torch.equal(tensor, original[name]), name)

    def test_practical_warm_start_preserves_complete_donor_and_separate_engine(self):
        import torch
        from hasami_learn.practical_calibrate import synthetic_levels
        from hasami_learn.trainer import Trainer

        engine, checkpoint = self.source()
        with torch.no_grad():
            for tensor in engine.network.state_dict().values():
                tensor.fill_(.75)
        checkpoint = engine.checkpoint(3)
        levels = self.tmp / 'levels.json'
        levels.write_text(json.dumps(synthetic_levels(checkpoint, checkpoint, k=1)))
        config = engine.config.with_overrides(
            run__run_id='practical-donor', algorithm__name='practical',
            algorithm__params={'engine_checkpoint': str(checkpoint), 'levels_file': str(levels),
                               'default_level': 1},
            network__params={**engine.config.network.params, 'aux_heads': {
                'practical': {'shape': [1, 1], 'activation': 'sigmoid'},
                'slip': {'shape': [1], 'activation': 'sigmoid'}}})
        donor = Trainer(config, echo=False)
        self.addCleanup(donor.close)
        # Without warm start, the original engine initialization still applies.
        for name, tensor in engine.network.state_dict().items():
            self.assertTrue(torch.equal(tensor, donor.network.state_dict()[name]), name)
        with torch.no_grad():
            for tensor in donor.network.state_dict().values():
                tensor.fill_(.25)
        donor_path = donor.checkpoint(7)
        for bootstrap in (False, True):
            with self.subTest(bootstrap=bootstrap):
                params = ({'bootstrap_tiny': True, 'default_level': 1} if bootstrap
                          else config.algorithm.params)
                target = Trainer(config.with_overrides(
                    run__run_id=f'practical-warm-{bootstrap}', run__init_checkpoint=str(donor_path),
                    algorithm__params=params), echo=False)
                self.addCleanup(target.close)
                self.assertEqual(set(target.network.state_dict()), set(donor.network.state_dict()))
                for name, tensor in donor.network.state_dict().items():
                    self.assertTrue(torch.equal(tensor, target.network.state_dict()[name]), name)
                if not bootstrap:
                    for name, tensor in engine.network.state_dict().items():
                        self.assertTrue(torch.equal(tensor, target.learner.engine.network.state_dict()[name]), name)
                self.assertEqual(target.iteration, 0)
                self.assertEqual(target.learner.samples, [])
                self.assertEqual(target.learner.optimizer.state_dict()['state'], {})
                self.assertEqual(target.manifest(0).init_checkpoint['path'], str(donor_path))

    def test_corrupt_hash_and_strict_tensor_mismatch_do_not_modify_target(self):
        import torch
        from hasami_learn.manifest import ManifestError, sha256_file
        source, checkpoint = self.source()
        original = {name: tensor.clone() for name, tensor in source.network.state_dict().items()}
        source.config.run.init_checkpoint = str(checkpoint)
        (checkpoint / 'weights.pt').write_bytes(b'bad')
        with self.assertRaisesRegex(ManifestError, 'sha256'):
            source._warm_start(source.learner.required_head)
        torch.save({**original, 'unexpected.tensor': torch.zeros(1)}, checkpoint / 'weights.pt')
        manifest = json.loads((checkpoint / 'manifest.json').read_text())
        manifest['weights_sha256'] = sha256_file(checkpoint / 'weights.pt')
        (checkpoint / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(RuntimeError, 'Unexpected key'):
            source._warm_start(source.learner.required_head)
        for name, tensor in source.network.state_dict().items():
            self.assertTrue(torch.equal(tensor, original[name]))

    def test_lineage_info_all_checkpoints_and_no_training_state_read(self):
        from hasami_learn.trainer import Trainer
        source, checkpoint = self.source()
        donor_manifest = json.loads((checkpoint / 'manifest.json').read_text())
        # A weights-only donor need not have readable trainer/learner state.
        (checkpoint / 'learner_state.pt').write_bytes(b'not an optimizer')
        (checkpoint / 'trainer_state.json').write_text('not json')
        config = source.config.with_overrides(run__run_id='lineage', run__init_checkpoint=str(checkpoint))
        target = Trainer(config, echo=False)
        self.addCleanup(target.close)
        expected = {'path': str(checkpoint), 'iteration': 3, 'algorithm': 'alphazero',
                    'weights_sha256': donor_manifest['weights_sha256']}
        rows = [json.loads(line) for line in (target.run_dir / 'metrics.jsonl').read_text().splitlines()]
        self.assertEqual(rows[0]['extra']['init_checkpoint'], expected)
        for iteration in (0, 1):
            path = target.checkpoint(iteration)
            self.assertEqual(json.loads((path / 'manifest.json').read_text())['init_checkpoint'], expected)
        self.assertEqual(target.iteration, 0)
        self.assertEqual(target.elapsed, 0.)
        self.assertIsNone(target.best_eval)
        self.assertEqual(target.opponents.state_dict(), source.opponents.state_dict())

    def test_fresh_rng_and_dqn_derived_network(self):
        import torch
        from hasami_learn.trainer import Trainer
        source, checkpoint = self.source('dqn')
        with torch.no_grad():
            for parameter in source.network.parameters():
                parameter.fill_(.25)
        checkpoint = source.checkpoint(3)
        config = source.config.with_overrides(run__run_id='fresh', run__seed=99)
        fresh = Trainer(config, echo=False)
        self.addCleanup(fresh.close)
        rng = torch.get_rng_state().clone()
        numpy_rng = fresh.rng.bit_generator.state
        warm = Trainer(config.with_overrides(run__run_id='warm-dqn', run__init_checkpoint=str(checkpoint)), echo=False)
        self.addCleanup(warm.close)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(warm.rng.bit_generator.state, numpy_rng)
        for name, tensor in warm.network.state_dict().items():
            self.assertTrue(torch.equal(tensor, warm.learner.target_network.state_dict()[name]))
            self.assertTrue(torch.equal(tensor, source.network.state_dict()[name]))
        self.assertEqual(warm.learner.gradient_steps, 0)

    def test_resume_uses_own_weights_and_preserves_lineage_without_donor(self):
        import torch
        from hasami_learn.config import ConfigError, USE_DEFAULT
        from hasami_learn.trainer import Trainer
        source, donor = self.source()
        config = source.config.with_overrides(run__run_id='resumable', run__init_checkpoint=str(donor))
        warm = Trainer(config, echo=False)
        self.addCleanup(warm.close)
        with torch.no_grad():
            for parameter in warm.network.parameters():
                parameter.fill_(.75)
        warm.iteration = 4
        checkpoint = warm.checkpoint(4)
        expected = warm.manifest(4).init_checkpoint
        donor.rename(donor.with_name('moved-donor'))
        for init in (str(donor), USE_DEFAULT):
            resumed = Trainer(config.with_overrides(run__init_checkpoint=init), resume=checkpoint, echo=False)
            self.addCleanup(resumed.close)
            self.assertEqual(resumed.iteration, 4)
            self.assertEqual(resumed.manifest(4).init_checkpoint, expected)
            for name, tensor in warm.network.state_dict().items():
                self.assertTrue(torch.equal(tensor, resumed.network.state_dict()[name]))
            path = resumed.checkpoint(5)
            self.assertEqual(json.loads((path / 'manifest.json').read_text())['init_checkpoint'], expected)
        with self.assertRaisesRegex(ConfigError, 'init_checkpoint'):
            Trainer(config.with_overrides(run__init_checkpoint='different'), resume=checkpoint, echo=False)
        legacy, old_checkpoint = self.source('reinforce')
        self.assertNotIn('init_checkpoint', json.loads((old_checkpoint / 'manifest.json').read_text()))
        with self.assertRaisesRegex(ConfigError, 'init_checkpoint'):
            Trainer(legacy.config.with_overrides(run__init_checkpoint='different'), resume=old_checkpoint, echo=False)

    def test_cli_initialization_and_resume_omission(self):
        from hasami_learn.cli import build_parser, resolve_train_request
        from hasami_learn.config import ConfigError
        from hasami_learn.trainer import Trainer
        source, checkpoint = self.source()
        path = self.tmp / 'config.json'
        source.config.with_overrides(run__run_id='cli-warm').save(path)
        parser = build_parser()
        args = parser.parse_args(['train', '--config', str(path), '--init-checkpoint', str(checkpoint)])
        config, resume = resolve_train_request(args)
        self.assertIsNone(resume)
        self.assertEqual(config.run.init_checkpoint, str(checkpoint))
        target = Trainer(config, echo=False)
        self.addCleanup(target.close)
        own = target.checkpoint(0)
        for suffix in ([], ['--init-checkpoint', str(checkpoint)]):
            args = parser.parse_args(['train', '--config', str(path), '--resume', str(own), *suffix])
            resumed, location = resolve_train_request(args)
            self.assertEqual(resumed.run.init_checkpoint, str(checkpoint))
            self.assertEqual(location, own)
        args = parser.parse_args(['train', '--resume', str(own), '--init-checkpoint', 'other'])
        with self.assertRaisesRegex(ConfigError, 'init_checkpoint'):
            resolve_train_request(args)

    def test_supervised_to_alphazero_cpu_smokes_for_hasami_and_shogi(self):
        import torch
        from hasami_learn.trainer import Trainer
        for game in ('hasami', 'shogi'):
            with self.subTest(game=game):
                config = self.tiny('supervised', generate_games=1, generator='random', augment=False).with_overrides(
                    game__name=game, run__run_id=f'teacher-{game}', network__params={'hidden': 8, 'layers': 1},
                    selfplay__games_per_iteration=1, selfplay__max_plies=6, replay__min_size=1,
                    replay__batch_size=2, optimizer__updates_per_iteration=1)
                teacher = Trainer(config, echo=False)
                self.addCleanup(teacher.close)
                checkpoint = teacher.run()
                config = config.with_overrides(run__run_id=f'student-{game}', algorithm__name='alphazero',
                                               algorithm__params={'simulations': 2}, run__init_checkpoint=str(checkpoint))
                student = Trainer(config, echo=False)
                self.addCleanup(student.close)
                self.assertEqual(student.iteration, 0)
                self.assertEqual(len(student.replay), 0)
                self.assertEqual(student.learner.optimizer.state_dict()['state'], {})
                for name, tensor in teacher.network.state_dict().items():
                    self.assertTrue(torch.equal(tensor, student.network.state_dict()[name]))
                final = student.run()
                manifest = json.loads((final / 'manifest.json').read_text())
                self.assertEqual(manifest['init_checkpoint']['algorithm'], 'supervised')
                self.assertEqual(manifest['iteration'], 1)
