"""Opt-in real Tk regression tests; run with HASAMI_GUI_TESTS=1."""
import os
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hasami.agents import Decision
from shogi import create_game
from shogi.encoding import move_label
from shogi.records import record_dict

GUI_ENABLED = os.environ.get('HASAMI_GUI_TESTS') == '1'


class FakeAdvisor:
    """Implement the public practical advisor protocol without learning imports."""
    levels = [SimpleNamespace(level=1, elo=None), SimpleNamespace(level=2, elo=123.0)]

    def advise(self, game, level, opponent_level=None, *, top_k=3, guard=.3, depth=1, cancel=None):
        return [SimpleNamespace(action=0, move=code, text='', practical_score=.6, engine_value=.1,
                                value_loss=.05, difficulty=.2, naturalness=.1, guard_fallback=False)
                for code in game.legal_actions()[:top_k]]

    def engine_best(self, game, cancel=None):
        return self.advise(game, 1, top_k=1)[0]

    def proxy_agent(self, level, *, style='mixture', seed=0):
        return SimpleNamespace(choose=lambda game, limits, cancel: Decision(move_label(game, game.legal_actions()[0])))


@unittest.skipUnless(GUI_ENABLED, 'Set HASAMI_GUI_TESTS=1 for real Tk events')
class TkTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        from shogi.gui import App, HUMAN
        from scripts.gui_smoke_shogi import wait_for
        self.wait_for = wait_for
        self.root = tk.Tk()
        self.app = App(self.root, advisor=FakeAdvisor())
        for var in self.app.player_vars:
            var.set(HUMAN)
        self.app.change_players()
        self.root.update()

    def tearDown(self):
        self.app.close()

    def test_spec_scenario(self):
        from scripts.gui_smoke_shogi import scenario
        scenario(self.app)

    def test_fake_advice_no_mutation_and_level_opponent(self):
        app = self.app
        before = record_dict(app.game)
        app.analysis_var.set(True)
        app.change_analysis()
        self.wait_for(self.root, lambda: len(app.advice_table.get_children()) == 4)
        self.assertEqual(record_dict(app.game), before)
        self.assertEqual(app.method_var.get(), '評価値ベース（探索）')
        app.analysis_var.set(False)
        app.player_vars[0].set('レベル 1（Elo 未計測）')
        app.change_players()
        self.wait_for(self.root, lambda: app.game.ply == 1)

    def test_stale_ai_and_advice_results(self):
        app = self.app
        for kind in ('ai', 'advice'):
            entered, release = threading.Event(), threading.Event()
            cancelled = []
            def operation(snapshot, cancel):
                entered.set()
                release.wait(3)
                cancelled.append(cancel.is_set())
                if kind == 'ai':
                    return 0, 'fake', None, Decision(snapshot.legal_actions()[0])
                hints = FakeAdvisor().advise(snapshot, 1)
                return FakeAdvisor.levels, hints, hints[0]
            app._launch(kind, operation)
            self.assertTrue(entered.wait(2))
            app.new_game()
            release.set()
            self.wait_for(self.root, lambda: app._worker is None)
            self.assertEqual(cancelled, [True])
            self.assertEqual(app.game.ply, 0)
            self.assertEqual(app.advice_table.get_children(), ())

    def test_illegal_ai_and_exception_preserve_game(self):
        app = self.app
        before = record_dict(app.game)
        with patch('shogi.gui.messagebox.showerror') as error:
            app._launch('ai', lambda snapshot, cancel: (0, 'bad', None, Decision(True)))
            self.wait_for(self.root, lambda: app._worker is None)
            self.assertTrue(error.called)
            self.assertTrue(app.paused)
        self.assertEqual(record_dict(app.game), before)
        with patch('shogi.gui.messagebox.showerror') as error:
            def fail(snapshot, cancel):
                raise RuntimeError('broken agent')
            app._launch('ai', fail)
            self.wait_for(self.root, lambda: app._worker is None)
            self.assertTrue(error.called)
        self.assertEqual(record_dict(app.game), before)

    def test_branch_confirm_cancel_and_accept(self):
        from scripts.gui_smoke_shogi import play
        from shogi.notation import usi_to_move
        app = self.app
        for text in ('7g7f', '3c3d', '2g2f'):
            play(app, text)
        full = record_dict(app.history.record)
        app.seek(1)
        code = usi_to_move(app.game, '8c8d')
        with patch('shogi.gui.messagebox.askyesno', return_value=False):
            app.apply(code)
        self.assertEqual(record_dict(app.history.record), full)
        with patch('shogi.gui.messagebox.askyesno', return_value=True):
            app.apply(code)
        self.assertEqual(app.game.ply, 2)
        self.assertEqual(app.history.length, 2)

    def test_no_optional_package_and_model_failure(self):
        app = self.app
        with patch.dict('sys.modules', {'hasami_learn.practical': None}):
            available, reason = app._probe_optional()
        self.assertFalse(available)
        self.assertIn('torch', reason)
        app._enable_learned(False)
        self.assertIn('disabled', app.analysis_check.state())
        self.assertIn('αβ', app.player_combos[0]['values'])
        app._enable_learned(True)
        before = record_dict(app.game)
        with patch('shogi.gui.filedialog.askdirectory', return_value='/missing/model'), \
                patch('shogi.gui.load_strong_agent', side_effect=ValueError('missing')), \
                patch('shogi.gui.messagebox.showerror') as error:
            app.load_model()
            self.wait_for(self.root, lambda: app._worker is None)
            self.assertTrue(error.called)
        self.assertEqual(record_dict(app.game), before)
