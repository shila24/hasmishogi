"""USI subprocess integration and deterministic protocol boundary tests."""
import io
import json
import os
from pathlib import Path
import tempfile
import queue
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import call, patch

from hasami.agents import AgentCancelled, Decision
from shogi import create_game
from shogi.native import available
from shogi.notation import move_to_usi, usi_to_move
from shogi.usi import USIEngine, go_limits, value_to_cp


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.process = subprocess.Popen([sys.executable, '-m', 'shogi.usi'], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self.lines = queue.Queue()
        self.reader = threading.Thread(target=lambda: [self.lines.put(line.rstrip()) for line in self.process.stdout], daemon=True)
        self.reader.start()

    def tearDown(self):
        if self.process.poll() is None:
            self.send('quit')
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.reader.join(timeout=1)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            stream.close()

    def send(self, command):
        self.process.stdin.write(command + '\n')
        self.process.stdin.flush()

    def until(self, prefix, timeout=5):
        deadline = time.monotonic() + timeout
        output = []
        while time.monotonic() < deadline:
            try:
                line = self.lines.get(timeout=max(.001, deadline-time.monotonic()))
            except queue.Empty:
                self.fail(f'Timed out waiting for {prefix}: {output}')
            output.append(line)
            if line.startswith(prefix):
                return line, output
        self.fail(f'Missing {prefix}')

    def test_handshake_options_position_and_info(self):
        self.send('usi')
        _, output = self.until('usiok')
        for name in ('Model', 'Mode', 'Method', 'Level', 'MaxNodes', 'Depth'):
            self.assertTrue(any(line.startswith('option name ' + name + ' ') for line in output))
        self.assertTrue(any('Method type combo default search' in line for line in output))
        self.send('isready')
        self.until('readyok')
        self.send('setoption name Depth value 1')
        self.send('usinewgame')
        self.send('position startpos moves 7g7f 3c3d')
        self.send('go depth 1 nodes 1000 movetime 1000')
        best, output = self.until('bestmove ')
        game = create_game()
        for move in ('7g7f', '3c3d'):
            game.play(usi_to_move(game, move))
        usi_to_move(game, best.split()[1])
        self.assertTrue(any(line.startswith('info depth ') and ' score cp ' in line for line in output))
        self.send('ponderhit')
        self.until('info string ponder')
        self.send('unknown-command')
        self.until('info string ignored')
        self.send('gameover win')

    def test_stop_and_deadline_and_no_duplicate_bestmove(self):
        self.send('position startpos')
        self.send('go infinite')
        self.send('stop')
        best, _ = self.until('bestmove ', timeout=2)
        usi_to_move(create_game(), best.split()[1])
        self.send('isready')
        _, extra = self.until('readyok')
        self.assertFalse(any(line.startswith('bestmove ') for line in extra))
        start = time.monotonic()
        self.send('go movetime 80 depth 64')
        best, _ = self.until('bestmove ', timeout=2)
        usi_to_move(create_game(), best.split()[1])
        self.assertLess(time.monotonic() - start, 1.0)

    def test_search_after_stop_or_timeout_returns_legal_move(self):
        game = create_game()
        self.addCleanup(game.close)
        for move in ('7g7f', '3c3d'):
            game.play(usi_to_move(game, move))
        for ending in ('stop', 'timeout'):
            with self.subTest(ending=ending):
                self.send('position startpos')
                self.send('go infinite' if ending == 'stop' else 'go movetime 30 depth 64')
                if ending == 'stop':
                    self.send('stop')
                self.until('bestmove ')
                self.send('position startpos moves 7g7f 3c3d')
                self.send('go movetime 100 depth 1')
                best, _ = self.until('bestmove ')
                usi_to_move(game, best.split()[1])

    def test_infinite_waits_for_stop_even_after_node_budget(self):
        self.send('go infinite nodes 1')
        self.until('info depth ')
        self.send('isready')
        _, output = self.until('readyok')
        self.assertFalse(any(line.startswith('bestmove ') for line in output))
        self.send('stop')
        best, _ = self.until('bestmove ')
        usi_to_move(create_game(), best.split()[1])

    def test_illegal_position_is_atomic(self):
        self.send('position startpos moves 7g7f')
        self.send('position startpos moves 7g7e')
        self.until('info string error')
        self.send('go depth 1 movetime 100')
        best, _ = self.until('bestmove ')
        game = create_game()
        game.play(usi_to_move(game, '7g7f'))
        usi_to_move(game, best.split()[1])

    def test_declaration_and_terminal_resign(self):
        self.send('position sfen 4K4/PPPP1PPPP/4P+P3/9/9/9/9/9/4k4 b 2R2B 1')
        self.send('go movetime 100')
        self.assertEqual(self.until('bestmove ')[0], 'bestmove win')
        self.send('position sfen k8/1G7/1K7/9/9/9/9/9/9 w - 1')
        self.send('go movetime 100')
        self.assertEqual(self.until('bestmove ')[0], 'bestmove resign')


class BoundaryTests(unittest.TestCase):
    def test_time_allocation_and_value_conversion(self):
        for args, turn, expected in [(['byoyomi', '1000'], 0, 800),
                                     (['btime', '3000', 'wtime', '9000'], 1, 300),
                                     (['btime', '3000', 'binc', '200'], 0, 300),
                                     (['movetime', '70'], 0, 70), (['byoyomi', '100'], 0, 1)]:
            limits, deadline = go_limits(args, turn, 3, 200000)
            self.assertEqual(deadline, expected)
            self.assertEqual(limits.time_ms, expected)
        self.assertIsNone(go_limits(['infinite'], 0, 3, 200000)[1])
        for args in (['nodes', '0'], ['depth', '0'], ['movetime'], ['btime', '-1']):
            with self.assertRaises(ValueError):
                go_limits(args, 0, 3, 200000)
        self.assertEqual(value_to_cp(0), 0)
        self.assertEqual(value_to_cp(-.5), -value_to_cp(.5))
        self.assertEqual(value_to_cp(10), value_to_cp(.999))

    def test_illegal_agent_rejected_and_late_result_suppressed(self):
        output = io.StringIO()
        engine = USIEngine(output=output)
        with patch.object(engine, '_agent', return_value=SimpleNamespace(choose=lambda *a: Decision(True))):
            engine.command('go movetime 500')
            engine.worker.join(2)
        self.assertIn('info string error', output.getvalue())
        self.assertIn('bestmove resign', output.getvalue())
        entered, release = threading.Event(), threading.Event()
        def choose(game, limits, cancel):
            entered.set()
            release.wait(2)
            return Decision(game.legal_actions()[0])
        with patch.object(engine, '_agent', return_value=SimpleNamespace(choose=choose)):
            engine.command('go infinite')
            self.assertTrue(entered.wait(2))
            engine.command('position startpos moves 7g7f')
            release.set()
            engine.worker.join(2)
        self.assertEqual(output.getvalue().count('bestmove '), 1)
        engine.close()

    def test_practical_default_and_policy_label_not_move(self):
        output = io.StringIO()
        engine = USIEngine(output=output)
        game = create_game()
        move = game.legal_actions()[0]
        advisor = SimpleNamespace(advise=lambda *a, **k: [SimpleNamespace(move=move, action=0,
            engine_value=.5, practical_score=.75, difficulty=None)])
        with patch('shogi.usi.load_advisor', return_value=advisor) as factory:
            engine.command('setoption name Model value fake')
            engine.command('setoption name Mode value practical')
            engine.command('go movetime 1000')
            engine.worker.join(2)
            factory.assert_called_once_with('fake', method='search')
        self.assertIn('practical_score 75.0%', output.getvalue())
        self.assertNotIn('bestmove resign', output.getvalue())
        engine.close()

    def test_config_factory_and_position_sfen_moves(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'agents.json'
            config.write_text(json.dumps({'agents': [{'id': 'configured', 'type': 'python',
                'factory': 'hasami_learn.inference:create_agent',
                'options': {'checkpoint': 'path with spaces', 'mode': 'policy', 'device': 'cpu'}}]}))
            engine = USIEngine(output=io.StringIO(), agents_config=config)
            with patch('shogi.usi.load_strong_agent', return_value=object()) as factory:
                engine._agent(engine.options)
                factory.assert_called_once_with('path with spaces', mode='policy', device='cpu')
            game = create_game()
            engine.command('position sfen ' + game.sfen() + ' moves 7g7f 3c3d')
            self.assertEqual(engine.game.ply, 2)
            engine.close()
            config.write_text(json.dumps({'agents': [{'id': 'bad', 'type': 'python',
                'factory': 'untrusted.module:factory', 'options': {}}]}))
            with self.assertRaises(ValueError):
                USIEngine(output=io.StringIO(), agents_config=config)

    def test_isready_warms_models_and_reuses_them_across_moves(self):
        engine = USIEngine(output=io.StringIO())
        engine.command('setoption name Model value fake')
        agent = object()
        with patch('shogi.usi.load_strong_agent', return_value=agent) as factory:
            engine.command('isready')
            self.assertIs(engine._agent(engine.options), agent)
            self.assertIs(engine._agent(engine.options), agent)
            factory.assert_called_once()
        self.assertIn('readyok', engine.output.getvalue())
        engine.close()


class WorkerRetirementTests(unittest.TestCase):
    def setUp(self):
        self.engine = USIEngine(output=io.StringIO())
        self.bestmoves = queue.Queue()
        emit = self.engine.emit

        def record(line):
            emit(line)
            if line.startswith('bestmove '):
                self.bestmoves.put(line.split()[1])

        self.engine.emit = record
        self.entered = threading.Event()
        self.cancelled = threading.Event()
        self.release = threading.Event()
        self.calls = []

        def choose(game, limits, cancel):
            self.calls.append((game.sfen(), limits))
            if len(self.calls) == 1:
                self.entered.set()
                if cancel.wait(5):
                    self.cancelled.set()
                self.release.wait(5)
                raise AgentCancelled('Previous search finished unwinding')
            return Decision(game.legal_actions()[-1])

        self.agent_patch = patch.object(self.engine, '_agent', return_value=SimpleNamespace(choose=choose))
        self.agent_patch.start()

    def tearDown(self):
        self.engine.close()
        self.release.set()
        if self.engine.worker:
            self.engine.worker.join(5)
        self.agent_patch.stop()

    def next_bestmove(self):
        try:
            return self.bestmoves.get(timeout=3)
        except queue.Empty:
            self.fail('No bestmove within the protocol deadline')

    def retire_first_search(self, ending):
        self.engine.command('go infinite' if ending == 'stop' else 'go movetime 100 depth 64')
        self.assertTrue(self.entered.wait(2))
        if ending == 'stop':
            self.engine.command('stop')
        usi_to_move(self.engine.game, self.next_bestmove())
        self.assertTrue(self.cancelled.wait(2))
        self.assertTrue(self.engine.worker.is_alive())
        self.engine.command('position startpos moves 7g7f 3c3d')

    def check_next_search(self, ending):
        self.retire_first_search(ending)
        self.engine.command('go movetime 2000 depth 1')
        self.engine.command('isready')
        self.assertIn('readyok', self.engine.output.getvalue())
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(self.bestmoves.empty(), 'Queued search must not resign')
        self.release.set()
        self.engine.worker.join(3)
        self.assertFalse(self.engine.worker.is_alive())
        self.assertEqual(self.next_bestmove(), move_to_usi(self.engine.game.legal_actions()[-1]))
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.calls[1][0], self.engine.game.sfen())
        self.assertTrue(self.bestmoves.empty(), 'A retired worker must not emit another bestmove')

    def test_next_search_waits_for_stopped_worker(self):
        self.check_next_search('stop')

    def test_next_search_waits_for_timed_out_worker(self):
        self.check_next_search('timeout')

    def test_main_waits_for_cancelled_worker_before_exiting(self):
        from shogi import usi

        def commands():
            yield 'go infinite\n'
            self.entered.wait(2)
            yield 'quit\n'

        finished = threading.Event()

        def run():
            try:
                usi.main([])
            finally:
                finished.set()

        with patch.object(usi, 'USIEngine', return_value=self.engine), patch('sys.stdin', commands()):
            runner = threading.Thread(target=run)
            runner.start()
            try:
                self.assertTrue(self.cancelled.wait(2))
                self.assertFalse(finished.wait(.05), 'Exit must wait for cancelled worker cleanup')
            finally:
                self.release.set()
                runner.join(3)
            self.assertFalse(runner.is_alive())
            self.assertFalse(self.engine.worker.is_alive())

    def test_queued_search_keeps_deadline_while_previous_worker_is_alive(self):
        self.retire_first_search('stop')
        self.engine.command('go movetime 50 depth 1')
        usi_to_move(self.engine.game, self.next_bestmove())
        self.assertFalse(self.release.is_set())
        self.assertEqual(len(self.calls), 1)
        self.release.set()
        self.engine.worker.join(3)
        self.assertFalse(self.engine.worker.is_alive())
        self.assertEqual(len(self.calls), 1, 'Expired requests must never invoke the model')
        self.assertTrue(self.bestmoves.empty())

    def test_invalidated_queued_search_never_invokes_agent(self):
        self.retire_first_search('stop')
        self.engine.command('go infinite')
        self.engine.command('position startpos moves 2g2f')
        self.engine.command('go movetime 2000 depth 1')
        self.release.set()
        self.engine.worker.join(3)
        self.assertEqual(self.next_bestmove(), move_to_usi(self.engine.game.legal_actions()[-1]))
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.calls[1][0], self.engine.game.sfen())
        self.assertTrue(self.bestmoves.empty())


class CompletedDepthTests(unittest.TestCase):
    def check_fixed_nodes(self, backend):
        for depth, nodes, completed_depth in ((4, 32, 1), (3, 335, 2)):
            with self.subTest(backend=backend, nodes=nodes):
                output = io.StringIO()
                engine = USIEngine(output=output, backend=backend)
                try:
                    engine.command(f'go depth {depth} nodes {nodes} movetime 30000')
                    engine.worker.join(5)
                    self.assertFalse(engine.worker.is_alive())
                    lines = output.getvalue().splitlines()
                    info = [line.split() for line in lines if line.startswith('info depth ')]
                    depths = [int(row[2]) for row in info]
                    self.assertEqual(max(depths), completed_depth)
                    self.assertEqual(depths, sorted(depths), 'Completed depth must never regress')
                    self.assertEqual(int(info[-1][4]), nodes, 'Nodes must include all choose calls')
                    self.assertEqual(info[-1][-1], '5i6h')
                    self.assertEqual(lines[-1], 'bestmove 5i6h')
                finally:
                    engine.close()

    def test_deepest_completed_result_survives_node_exhaustion_python(self):
        self.check_fixed_nodes('python')

    def test_deepest_completed_result_survives_node_exhaustion_native(self):
        if not available():
            if os.environ.get('HASAMI_REQUIRE_NATIVE') == '1':
                self.fail('Native shogi library is required')
            self.skipTest('Native shogi library is unavailable')
        self.check_fixed_nodes('cpp')


class NativeSearchOptionTests(unittest.TestCase):
    def native_engine(self):
        from shogi.native_search import available as search_available
        if not search_available():
            if os.environ.get('HASAMI_REQUIRE_NATIVE') == '1':
                self.fail('Native shogi library is required')
            self.skipTest('Native shogi library is unavailable')
        engine = USIEngine(output=io.StringIO())
        def close():
            engine.close()
            if engine.worker:
                engine.worker.join(5)
                self.assertFalse(engine.worker.is_alive())
        self.addCleanup(close)
        engine.command('setoption name Engine value alphabeta-cpp')
        engine.command('setoption name MaxNodes value 18446744073709551615')
        return engine

    def assert_completed_bestmove(self, engine):
        lines = engine.output.getvalue().splitlines()
        info = [line.split() for line in lines if line.startswith('info depth ')]
        self.assertTrue(info, lines)
        depths = [int(row[2]) for row in info]
        self.assertGreater(max(depths), 0)
        self.assertEqual(depths, sorted(depths))
        best = [line for line in lines if line.startswith('bestmove ')]
        self.assertEqual(best, ['bestmove ' + info[-1][-1]])
        self.assertNotEqual(best[0], 'bestmove 9g9f')
        self.assertTrue(engine.game.is_legal(usi_to_move(engine.game, info[-1][-1])))

    def test_native_deadline_preserves_completed_iteration(self):
        engine = self.native_engine()
        for milliseconds in (100, 200, 500, 1000):
            with self.subTest(milliseconds=milliseconds):
                engine.output.seek(0)
                engine.output.truncate()
                started = time.monotonic()
                engine.command(f'go depth 64 movetime {milliseconds}')
                engine.worker.join(3)
                self.assertFalse(engine.worker.is_alive())
                self.assertLess(time.monotonic() - started, milliseconds / 1000 + 1)
                self.assert_completed_bestmove(engine)

    def test_native_mid_search_stop_preserves_completed_iteration(self):
        engine = self.native_engine()
        published = threading.Event()
        emit = engine.emit
        def observe(line):
            emit(line)
            if line.startswith('info depth ') and int(line.split()[2]) >= 3:
                published.set()
        with patch.object(engine, 'emit', side_effect=observe):
            engine.command('go infinite')
            self.assertTrue(published.wait(3), 'Native iterations must publish while searching')
            self.assertTrue(engine.worker.is_alive())
            engine.command('stop')
            engine.worker.join(3)
            self.assertFalse(engine.worker.is_alive())
        self.assert_completed_bestmove(engine)

    def test_default_unchanged_and_native_option_advertised(self):
        from shogi.agents import AlphaBetaAgent
        engine = USIEngine(output=io.StringIO())
        self.addCleanup(engine.close)
        engine.command('usi')
        self.assertIn('option name Engine type combo default python var python var alphabeta-cpp', engine.output.getvalue())
        self.assertIsInstance(engine._agent(engine.options), AlphaBetaAgent)
        engine.command('setoption name Engine value invalid')
        self.assertEqual(engine.options['Engine'], 'python')

    def test_missing_native_search_reports_error(self):
        engine = USIEngine(output=io.StringIO())
        self.addCleanup(engine.close)
        with patch('shogi.native_search.find_library', side_effect=RuntimeError('native search missing')):
            engine.command('setoption name Engine value alphabeta-cpp')
            engine.command('isready')
            self.assertIn('info string error native search missing', engine.output.getvalue())
            engine.command('go depth 1 movetime 1000')
            engine.worker.join(3)
            self.assertIn('bestmove resign', engine.output.getvalue())

    def test_native_search_go(self):
        from shogi.native_search import available as search_available
        if not search_available():
            if os.environ.get('HASAMI_REQUIRE_NATIVE') == '1':
                self.fail('Native shogi library is required')
            self.skipTest('Native shogi library is unavailable')
        engine = USIEngine(output=io.StringIO())
        self.addCleanup(engine.close)
        engine.command('setoption name Engine value alphabeta-cpp')
        engine.command('go depth 3 nodes 10000 movetime 30000')
        engine.worker.join(5)
        self.assertFalse(engine.worker.is_alive())
        lines = engine.output.getvalue().splitlines()
        self.assertTrue(any(line.startswith('info depth 3 ') for line in lines))
        self.assertTrue(engine.game.is_legal(usi_to_move(engine.game, lines[-1].split()[1])))


class InstalledAssetSmokeTests(unittest.TestCase):
    def setUp(self):
        from scripts import usi_smoke
        self.smoke = usi_smoke
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.checkpoint = self.root / 'models/checkpoint'
        self.nnue = self.root / 'models/nnue'
        self.checkpoint.mkdir(parents=True)
        self.nnue.mkdir()
        for name in ('manifest.json', 'weights.pt'):
            (self.checkpoint / name).write_text('fixture')
        for name in ('weights.hlnnw1', 'weights.hlnnw1.json'):
            (self.nnue / name).write_text('fixture')
        self.index = self.root / 'models/index.json'
        self.rows = [
            {'name': 'checkpoint', 'game': 'shogi', 'asset_kind': 'checkpoint'},
            {'name': 'nnue', 'game': 'shogi', 'asset_kind': 'nnue'},
            {'name': 'uninstalled', 'game': 'shogi', 'asset_kind': 'nnue'},
            {'name': 'other', 'game': 'hasami', 'asset_kind': 'checkpoint'},
        ]
        self.index.write_text(json.dumps({'models': self.rows}))

    def test_discovery_accepts_checkpoint_and_nnue_layouts(self):
        self.assertEqual(self.smoke.installed_models(self.root),
                         [self.checkpoint, self.nnue / 'weights.hlnnw1'])

    def test_default_command_dispatches_each_installed_asset_kind(self):
        output = io.StringIO()
        with patch.object(self.smoke, 'ROOT', self.root), \
                patch.object(self.smoke, 'exercise', return_value={'ok': True}) as exercise, \
                patch('sys.stdout', output):
            status = self.smoke.main([])
        self.assertEqual(status, 0, output.getvalue())
        self.assertEqual(exercise.call_args_list, [
            call(backend='python', timeout=15),
            call(model=self.checkpoint, backend='python', timeout=15),
            call(eval_file=self.nnue / 'weights.hlnnw1', backend='python', timeout=15),
        ])
        self.assertEqual(len(json.loads(output.getvalue())['checks']), 3)

    def test_incomplete_nnue_is_rejected(self):
        (self.nnue / 'weights.hlnnw1.json').unlink()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.smoke.installed_models(self.root)

    def test_unknown_asset_kind_is_rejected(self):
        self.index.write_text(json.dumps({'models': [dict(self.rows[1], asset_kind='unknown')]}))
        with self.assertRaisesRegex(ValueError, 'asset kind'):
            self.smoke.installed_models(self.root)

    def test_explicit_eval_file_and_classical_only_selection(self):
        for args, expected in (
                (['--eval-file', str(self.nnue / 'weights.hlnnw1')],
                 [call(eval_file=self.nnue / 'weights.hlnnw1', backend='python', timeout=15)]),
                (['--classical-only'], []),
                (['--model', str(self.checkpoint)], [call(model=self.checkpoint, backend='python', timeout=15)])):
            with self.subTest(args=args), patch.object(self.smoke, 'ROOT', self.root), \
                    patch.object(self.smoke, 'exercise', return_value={'ok': True}) as exercise, \
                    patch('sys.stdout', io.StringIO()):
                self.assertEqual(self.smoke.main(args), 0)
                self.assertEqual(exercise.call_args_list, [call(backend='python', timeout=15), *expected])


class NNUEUSITests(unittest.TestCase):
    def setUp(self):
        from shogi.native_search import available as search_available
        if not search_available():
            if os.environ.get('HASAMI_REQUIRE_NATIVE') == '1':
                self.fail('Native shogi library is required')
            self.skipTest('Native shogi library is unavailable')

    def test_evalfile_selection_clear_and_errors(self):
        from tests.shogi_suite.test_native_search import write_test_nnue
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval weights.hlnnw1"
            write_test_nnue(path)
            output = io.StringIO()
            engine = USIEngine(output=output)
            try:
                engine.command("usi")
                self.assertIn("option name EvalFile type string default <empty>", output.getvalue())
                engine.command(f"setoption name EvalFile value {path}")
                engine.command("isready")
                self.assertIn("EvalFile requires Engine alphabeta-cpp", output.getvalue())
                engine.command("setoption name Engine value alphabeta-cpp")
                engine.command("isready")
                agent = engine._agent(engine.options)
                self.assertEqual(agent._search.evaluator,
                                 "hlnn-kp-v1:w1:1:1:f32:cp600")
                engine.command("position startpos")
                engine.command("go depth 1 nodes 100 movetime 2000")
                engine.worker.join(timeout=5)
                self.assertFalse(engine.worker.is_alive())
                best = [line for line in output.getvalue().splitlines() if line.startswith("bestmove ")][-1]
                usi_to_move(engine.game, best.split()[1])
                self.assertIn("score cp -75", output.getvalue())
                engine.command(f"setoption name EvalFile value {path}.missing")
                engine.command("isready")
                self.assertIn("Cannot open NNUE weights", output.getvalue())
                engine.command("setoption name EvalFile value <empty>")
                self.assertEqual(engine.options["EvalFile"], "")
                self.assertIsNone(engine._agent(engine.options)._search)
            finally:
                engine.close()

    def test_smoke_loads_evalfile_and_checks_search_and_stop(self):
        from scripts import usi_smoke
        from tests.shogi_suite.test_native_search import write_test_nnue
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'eval weights.hlnnw1'
            write_test_nnue(path)
            with patch.object(usi_smoke.Session, 'send', autospec=True,
                              side_effect=usi_smoke.Session.send) as send:
                report = usi_smoke.exercise(eval_file=path)
            self.assertTrue(report['ok'])
            self.assertEqual(report['mode'], 'nnue')
            commands = [args.args[1] for args in send.call_args_list]
            for command in ('setoption name Engine value alphabeta-cpp',
                            'setoption name Model value <empty>',
                            'setoption name EvalFile value ' + str(path.resolve())):
                self.assertIn(command, commands)
            path.write_bytes(b'invalid weights')
            with self.assertRaisesRegex(usi_smoke.ProtocolError, 'model error'):
                usi_smoke.exercise(eval_file=path)


class NNUEUSIAvailabilityTests(unittest.TestCase):
    def test_optional_and_required_search_library(self):
        with tempfile.TemporaryDirectory() as directory:
            for required in (False, True):
                with self.subTest(required=required):
                    env = {**os.environ, "SHOGI_SEARCH_LIB": str(Path(directory) / "missing-library"),
                           "HASAMI_REQUIRE_NATIVE": "1" if required else "0"}
                    result = subprocess.run(
                        [sys.executable, "-m", "unittest", "tests.shogi_suite.test_usi.NNUEUSITests", "-v"],
                        env=env, capture_output=True, text=True, timeout=30)
                    if required:
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn("Native shogi library is required", result.stderr)
                        self.assertNotIn("skipped", result.stderr)
                    else:
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertIn("skipped 'Native shogi library is unavailable'", result.stderr)
