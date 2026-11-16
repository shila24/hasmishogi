"""Seeded evaluation openings, reproducible search and effective sample regression tests."""
from __future__ import annotations

import json
import os
from unittest.mock import patch

from hasami.agents import Decision, SearchLimits
from tests.learn._common import LearnTestCase, requires_torch


class FirstAgent:
    def __init__(self, spec):
        self.spec = spec

    def choose(self, game, limits, cancel=None, player=None):
        player = self.spec.actors(game)[0] if player is None else player
        return Decision(self.spec.legal_actions(game, player)[0])


@requires_torch
class OpeningTests(LearnTestCase):
    def matches(self, **kwargs):
        from hasami_learn.evaluation import play_matches
        from hasami_learn.games import get_spec
        spec = kwargs.pop('spec', get_spec())
        return play_matches(FirstAgent(spec), FirstAgent(spec), kwargs.pop('games', 20),
                            spec=spec, max_plies=kwargs.pop('max_plies', 6),
                            limits=SearchLimits(1, 1000, 100), seed=17, **kwargs)

    def test_duplicate_games_are_only_two_effective_samples(self):
        from hasami_learn.evaluation import wilson_interval
        result = self.matches()
        self.assertEqual(result.distinct_games, 2)
        self.assertEqual(result.distinct_openings, 1)
        self.assertEqual(result.effective_games, 2)
        self.assertEqual(result.to_dict()['extra']['effective_games'], 2)
        self.assertEqual((result.to_dict()['ci_low'], result.to_dict()['ci_high']), wilson_interval(1, 2))

    def test_seeded_openings_rotate_and_diversify(self):
        first = self.matches(opening_plies=3, opening_seed=52)
        second = self.matches(opening_plies=3, opening_seed=52)
        self.assertEqual(first.tables, second.tables)
        self.assertEqual(first.distinct_games, first.games)
        self.assertEqual(first.effective_games, first.games)
        self.assertEqual(first.distinct_openings, 10)
        for index in range(0, 20, 2):
            self.assertEqual(first.tables[index]['opening_actions'], first.tables[index + 1]['opening_actions'])

    def test_zero_opening_stored_reference_results_and_records(self):
        from hasami_learn.games import get_spec
        expected_moves = {'hasami': [5841, 91, 747, 1, 1458, 173],
                          'shogi': ['9g9f', '9a9b', '9f9e', '7a7b', '9e9d', '6a7a']}
        for name, moves in expected_moves.items():
            with self.subTest(game=name), patch('hasami_learn.evaluation.time.perf_counter', side_effect=[10., 10.25]):
                result = self.matches(spec=get_spec(name), games=2, opening_plies=0,
                                      records_dir=self.tmp / name)
            data = result.to_dict()
            data.pop('extra')
            self.assertEqual(data, {'games': 2, 'wins': 0, 'losses': 0, 'draws': 2,
                                    'truncated': 2, 'plies': 12, 'seconds': .25, 'reasons': {'max_plies': 2},
                                    'win_rate': .5, 'elo_delta': 0., 'ci_low': 0.09452865480086614,
                                    'ci_high': 0.9054713451991339, 'draw_rate': 1.})
            for path in (self.tmp / name).glob('eval_*.json'):
                self.assertEqual(json.loads(path.read_text())['moves'], moves)
            self.assertEqual(sorted(p.name for p in (self.tmp / name).iterdir()),
                             ['eval_0000.json', 'eval_0001.json'])

    def test_opening_records_replay_and_consume_cap(self):
        from hasami_learn.games import get_spec
        from hasami_learn.games.records import load_record, save_record
        for name in ('hasami', 'shogi', 'riichi'):
            spec = get_spec(name)
            result = self.matches(spec=spec, games=spec.num_players, opening_plies=3,
                                  max_plies=3, records_dir=self.tmp / name)
            self.assertEqual(result.plies, 3 * spec.num_players)
            self.assertEqual(result.truncated, spec.num_players)
            for index, table in enumerate(result.tables):
                self.assertEqual(table['actions'], table['opening_actions'])
                path = self.tmp / name / f'eval_{index:04d}.json'
                metadata = json.loads((path.parent / 'openings' / path.name).read_text())
                self.assertEqual(metadata['opening_actions'],
                                 json.loads(json.dumps(table['opening_actions'])))
                game = load_record(spec, path)
                replay = spec.create_game(seed=table['seed'])
                try:
                    for step in table['actions']:
                        spec.step(replay, step)
                    self.assertEqual(game.moves, replay.moves)
                    self.assertEqual(game.outcome, replay.outcome)
                    save_record(spec, game, path.with_name('roundtrip.json'))
                    self.assertEqual(json.loads(path.read_text()),
                                     json.loads(path.with_name('roundtrip.json').read_text()))
                finally:
                    game.close()
                    replay.close()

    def test_riichi_chance_and_simultaneous_rotations(self):
        from hasami_learn.games import get_spec
        from .test_multiplayer import HiddenSpec
        for spec in (get_spec('riichi'), HiddenSpec()):
            for plies in (0, 1):
                with self.subTest(game=spec.game_id, opening=plies):
                    first = self.matches(spec=spec, games=8, opening_plies=plies, opening_seed=7)
                    second = self.matches(spec=spec, games=8, opening_plies=plies, opening_seed=7)
                    self.assertEqual(first.tables, second.tables)
                    for cycle in range(0, 8, spec.num_players):
                        tables = first.tables[cycle:cycle + spec.num_players]
                        self.assertEqual(len({t['seed'] for t in tables}), 1)
                        self.assertTrue(all(t['opening_actions'] == tables[0]['opening_actions'] for t in tables))

    def test_opening_stream_does_not_depend_on_deal_seed_or_agent(self):
        from hasami_learn.evaluation import play_matches
        from hasami_learn.games import get_spec
        spec = get_spec()
        agent = FirstAgent(spec)
        openings = []
        for seed in (1, 999):
            result = play_matches(agent, agent, 6, spec=spec, limits=SearchLimits(),
                                 seed=seed, opening_seed=8, opening_plies=3, max_plies=4)
            openings.append([t['opening_actions'] for t in result.tables])
        self.assertEqual(*openings)
        no_swap = self.matches(games=5, swap_sides=False, opening_plies=3)
        self.assertEqual(no_swap.distinct_openings, 5)
        unseeded = play_matches(agent, agent, 2, spec=spec, limits=SearchLimits(),
                                opening_seed=8, opening_plies=3, max_plies=4)
        self.assertEqual(unseeded.tables[0]['seed'], unseeded.tables[1]['seed'])

    def test_terminal_and_adapter_truncated_redraw_and_exhaustion(self):
        from types import SimpleNamespace
        from hasami_learn.games.base import StepOutcome
        from .test_multiplayer import HiddenSpec, ToyGame

        class EarlyGame(ToyGame):
            @property
            def outcome(self):
                if self.moves and self.moves[0].get(0) == 1:
                    return SimpleNamespace(terminal=True, winner=0, reason='early')
                return super().outcome

        class EarlySpec(HiddenSpec):
            def __init__(self, truncated=False, always=False):
                self.truncated, self.always, self.created, self.closed = truncated, always, 0, 0

            def create_game(self, backend='python', rules=None, position=None, seed=None):
                self.created += 1
                game = ToyGame(2, seed) if self.truncated else EarlyGame(2, seed)
                game.close = lambda: setattr(self, 'closed', self.closed + 1)
                return game

            def step(self, game, actions):
                outcome = super().step(game, actions)
                reject = self.always or (len(game.moves) == 1 and actions[0] == 1)
                return StepOutcome(outcome.terminal or (reject and not self.truncated),
                                   self.truncated and reject, outcome.rewards, outcome.info)

        for truncated in (False, True):
            spec = EarlySpec(truncated)
            # The first stream selects action 1, so this seed requires a redraw.
            import random
            seed = next(n for n in range(100) if random.Random(f'{n}:0:0').choice([0, 1]) == 1)
            result = self.matches(spec=spec, games=2, opening_plies=1, opening_seed=seed)
            self.assertEqual(result.tables[0]['opening_actions'][0][0], 0)
            self.assertGreater(spec.created, 3)
            failed = EarlySpec(truncated, always=True)
            with self.assertRaisesRegex(ValueError, '128 attempts'):
                self.matches(spec=failed, games=1, opening_plies=1)
            self.assertEqual(failed.created, 128)
            self.assertEqual(failed.closed, 128)

    def test_opening_inputs_and_cancel(self):
        import threading
        from hasami.agents import AgentCancelled
        for kwargs in ({'opening_plies': -1}, {'opening_plies': True}, {'opening_plies': 7},
                       {'opening_seed': -1}, {'opening_seed': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.matches(**kwargs)
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(AgentCancelled):
            self.matches(opening_plies=1, cancel=cancel)
        empty = self.matches(games=0, opening_plies=2)
        self.assertEqual((empty.effective_games, empty.distinct_games, empty.distinct_openings), (0, 0, 0))
        self.assertEqual((empty.to_dict()['ci_low'], empty.to_dict()['ci_high']), (0., 1.))

    def test_effective_group_means_have_equal_weight(self):
        from hasami_learn.selfplay import Trajectory
        from hasami_learn.evaluation import wilson_interval
        trajectories = [Trajectory(rewards=[1., -1.], joint_actions=[{0: 1}]),
                        Trajectory(rewards=[-1., 1.], joint_actions=[{0: 1}]),
                        Trajectory(rewards=[-1., 1.], joint_actions=[{0: 2}])]
        with patch('hasami_learn.evaluation.run_table_game', side_effect=trajectories):
            result = self.matches(games=3)
        self.assertEqual((result.wins, result.losses), (2, 1))
        self.assertEqual((result.effective_successes, result.effective_games), (1.5, 2))
        self.assertEqual((result.to_dict()['ci_low'], result.to_dict()['ci_high']), wilson_interval(1.5, 2))
        self.assertAlmostEqual(result.to_dict()['ci_low'], 0.19786250921045623)

    def test_mcts_simulations_reproduce_under_slow_evaluator_and_keep_play_deadline(self):
        import time
        import torch
        from hasami_learn.evaluation import play_matches
        from hasami_learn.games import get_spec
        from hasami_learn.inference import MctsAgent
        from hasami_learn.registry import NETWORKS
        from hasami_learn import networks  # Register network factories.
        spec = get_spec()
        encoder = spec.make_encoder('')
        torch.manual_seed(6)
        network = NETWORKS.get('mlp')(head='policy_value', num_planes=encoder.num_planes,
                                      board_shape=spec.board_shape, action_size=spec.action_size,
                                      hidden=8, layers=1)
        results = []
        for slow in (False, True):
            agent = MctsAgent(network, encoder, 'cpu', simulations=6, spec=spec)
            original = agent.search.evaluate_fn
            def evaluate(*args):
                if slow:
                    time.sleep(.003)
                return original(*args)
            agent.search.evaluate_fn = evaluate
            with patch.object(agent.search, 'run', wraps=agent.search.run) as search:
                result = play_matches(agent, FirstAgent(spec), 2, spec=spec, max_plies=4,
                                      limits=SearchLimits(1, 1, 1), opening_plies=1, seed=22)
                self.assertTrue(search.call_args_list)
                for call in search.call_args_list:
                    self.assertIsNone(call.kwargs['deadline'])
                    self.assertEqual(call.kwargs['simulations'], 6)
            results.append(result.tables)
        self.assertEqual(*results)
        game = spec.create_game()
        try:
            with patch.object(agent.search, 'run', wraps=agent.search.run) as search:
                agent.choose(game, SearchLimits(1, 1, 1))
                self.assertIsNotNone(search.call_args.kwargs['deadline'])
                self.assertEqual(search.call_args.kwargs['simulations'], 1)
        finally:
            game.close()

    def test_trainer_metrics_cli_and_ladder_plumbing(self):
        import contextlib
        import io
        from hasami_learn.trainer import Trainer
        from hasami_learn.cli import main
        from hasami_learn.ladder import run_ladder
        config = self.tiny('alphazero', simulations=2).with_overrides(
            evaluation__opening_plies=2, evaluation__games=4, evaluation__max_plies=4,
            evaluation__simulations=3, evaluation__limits={'time_ms': 1})
        trainer = Trainer(config, echo=False)
        self.addCleanup(trainer.close)
        agent = trainer.learner.make_agent('eval')
        with patch.object(trainer.learner, 'make_agent', return_value=agent), \
                patch.object(agent.search, 'run', wraps=agent.search.run) as search:
            result = trainer.evaluate(1)['random']
            self.assertTrue(search.call_args_list)
            self.assertTrue(all(call.kwargs['simulations'] == 3 for call in search.call_args_list))
        rows = [json.loads(line) for line in (trainer.run_dir / 'metrics.jsonl').read_text().splitlines()]
        extra = next(row['extra'] for row in rows if row['kind'] == 'evaluation')
        for key, value in result['extra'].items():
            self.assertEqual(extra[key], value)
        self.assertEqual(extra['ci_low'], result['ci_low'])
        checkpoint = trainer.checkpoint(0)
        out = self.tmp / 'eval.json'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['eval', '--checkpoint', str(checkpoint), '--opponent', 'random',
                                   '--games', '4', '--max-plies', '4', '--opening-plies', '2',
                                   '--opening-seed', '17', '--simulations', '2', '--time-ms', '1',
                                   '--device', 'cpu', '--out', str(out)]), 0)
        self.assertEqual(json.loads(out.read_text())['random']['extra']['distinct_openings'], 2)
        ladder = run_ladder([f'model={checkpoint}', 'random'], games=4, backend='python', max_plies=4,
                            opening_plies=2, opening_seed=17, simulations=2, time_ms=1,
                            device='cpu', out=self.tmp / 'ladder')
        resumed = run_ladder([f'model={checkpoint}', 'random'], games=4, backend='python', max_plies=4,
                             opening_plies=2, opening_seed=17, simulations=2, time_ms=1,
                             device='cpu', out=self.tmp / 'ladder', resume=True)
        self.assertEqual(ladder, resumed)
        self.assertEqual(ladder['pairs'][0]['result']['extra']['effective_games'], 4)

    def test_openings_preserve_python_native_parity(self):
        from hasami_learn.evaluation import play_matches
        from hasami_learn.games import get_spec
        for name in ('hasami', 'shogi', 'riichi'):
            with self.subTest(game=name):
                spec = get_spec(name)
                try:
                    native = spec.create_game('cpp', seed=21)
                except (OSError, RuntimeError):
                    if os.environ.get('HASAMI_REQUIRE_NATIVE') == '1':
                        raise
                    self.skipTest(f'{name} native backend is unavailable')
                else:
                    native.close()
                results = []
                for backend in ('python', 'cpp'):
                    # Explicit cpp must fail rather than silently use Python.
                    result = play_matches(FirstAgent(spec), FirstAgent(spec), spec.num_players,
                                          spec=spec, backend=backend, max_plies=6,
                                          limits=SearchLimits(), seed=21, opening_seed=77, opening_plies=3)
                    results.append(result.tables)
                self.assertEqual(*results)

    def test_partially_resumed_legacy_ladder_keeps_both_interval_populations(self):
        from collections import Counter
        from hasami_learn.evaluation import MatchResult, wilson_interval
        from hasami_learn.ladder import run_ladder

        legacy = MatchResult(games=100, losses=100, plies=100,
                             reasons=Counter({'five_captures': 100}))
        recent = MatchResult(games=100, wins=100, plies=100,
                             reasons=Counter({'five_captures': 100}), distinct_games=2,
                             distinct_openings=1, effective_games=2, effective_successes=2.)
        options = dict(entries=['candidate=random', 'old=greedy', 'new=alphabeta:1'],
                       games=100, backend='python', device='cpu', out=self.tmp / 'ladder')
        with patch('hasami_learn.ladder.play_matches', side_effect=[legacy, RuntimeError('interrupted')]):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                run_ladder(**options)
        path = self.tmp / 'ladder/ladder.json'
        partial = json.loads(path.read_text())
        self.assertEqual(len(partial['pairs']), 1)
        # Old ladders predate both opening config fields and diversity metadata.
        partial['config'].pop('opening_plies')
        partial['config'].pop('opening_seed')
        self.assertNotIn('extra', partial['pairs'][0]['result'])
        path.write_text(json.dumps(partial))
        with patch('hasami_learn.ladder.play_matches', return_value=recent) as play:
            report = run_ladder(**options, resume=True)
        self.assertEqual(play.call_count, 2)
        row = next(row for row in report['standings'] if row['name'] == 'candidate')
        self.assertEqual((row['games'], row['wins'], row['losses'], row['score_rate']), (200, 100, 100, .5))
        self.assertEqual((row['extra']['effective_successes'], row['extra']['effective_games']), (2., 4))
        self.assertEqual((row['ci_low'], row['ci_high']), wilson_interval(2., 4))
        self.assertAlmostEqual(row['ci_low'], .15003570882017145)
        self.assertEqual(row['extra']['legacy_pairs'], 1)
        self.assertEqual(row['extra']['legacy_games'], 100)
        self.assertEqual(row['extra']['legacy_effective_games'], 2)
        self.assertIsNone(row['extra']['distinct_games'])
        self.assertIsNone(row['extra']['distinct_openings'])
        self.assertIn('legacy', report['scoring'])
        # Pair history keeps its original raw interval; standings use the fallback.
        self.assertEqual(report['pairs'][0]['result']['ci_high'], wilson_interval(0, 100)[1])
        inverse = next(row for row in report['standings'] if row['name'] == 'old')
        self.assertEqual((inverse['extra']['effective_successes'], inverse['extra']['effective_games']), (4., 4))
        with patch('hasami_learn.ladder.play_matches', side_effect=AssertionError('replayed')):
            self.assertEqual(run_ladder(**options, resume=True), report)
