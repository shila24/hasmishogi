"""Worker ownership tests and opt-in, process-isolated real Tk lifecycles."""
import gc
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
from types import SimpleNamespace
import unittest
import weakref

from hasami.agents import AgentCancelled, Decision
from shogi.gui_support import run_worker


class WorkerTests(unittest.TestCase):
    def test_result_and_snapshot_cleanup(self):
        closed = []
        snapshot = SimpleNamespace(close=lambda: closed.append(True))
        results = queue.Queue()
        run_worker(results, 4, 'ai', lambda game, cancel: 123, snapshot, threading.Event())
        self.assertEqual(closed, [True])
        self.assertEqual(results.get_nowait(), (4, 'ai', 123, None))

    def test_errors_do_not_retain_operation_or_traceback(self):
        for exception in (ValueError, AgentCancelled):
            with self.subTest(exception=exception):
                class Operation:
                    def __call__(self, snapshot, cancel):
                        raise exception('worker failed')

                closed = []
                snapshot = SimpleNamespace(close=lambda: closed.append(True))
                operation = Operation()
                reference = weakref.ref(operation)
                results = queue.Queue()
                run_worker(results, 7, 'advice', operation, snapshot, threading.Event())
                del operation
                gc.collect()
                self.assertIsNone(reference(), 'The queued error retained the worker frame')
                self.assertEqual(closed, [True])
                self.assertEqual(results.get_nowait(),
                                 (7, 'advice', None, (exception is AgentCancelled, 'worker failed')))

    def test_close_failure_is_published(self):
        def fail_close():
            raise RuntimeError('close failed')

        results = queue.Queue()
        run_worker(results, 2, 'levels', lambda game, cancel: (),
                   SimpleNamespace(close=fail_close), threading.Event())
        self.assertEqual(results.get_nowait(), (2, 'levels', (), (False, 'close failed')))


def _wait_for_release(entered, release, snapshot, limits, cancel):
    """Model an agent that outlives the bounded join, using no GUI references."""
    entered.set()
    if not release.wait(10):
        raise RuntimeError('Test did not release the worker')
    gc.collect()
    return Decision(snapshot.legal_actions()[0])


def _exercise_tcl_owner():
    """Reproduce worker ownership without a display, using a real Tcl interpreter."""
    from functools import partial
    import tkinter as tk

    from shogi import create_game
    from shogi.gui import App
    from shogi.gui_support import History

    entered, release = threading.Event(), threading.Event()
    finalized = []
    app = App.__new__(App)
    app.root = tk.Tcl()
    weakref.finalize(app.root, lambda: finalized.append(threading.current_thread().name))
    app.history = History(create_game())
    app._worker, app._closed, app._generation = None, False, 0
    app._results = queue.Queue()
    app.thinking = SimpleNamespace(configure=lambda **kwargs: None)
    operation = partial(_wait_for_release, entered, release, limits=None)
    # Adapt the agent signature without capturing the GUI owner.
    def work(snapshot, cancel):
        return operation(snapshot, cancel=cancel)

    app._launch('ai', work)
    worker = app._worker
    try:
        assert entered.wait(5), 'Worker did not start'
        app._cancel.set()
        app.history.close()
        # The original Thread target's self cell owns App -> root -> tkapp here.
        del app
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive(), 'Released worker did not finish'
    assert finalized == ['MainThread'], finalized
    print('Tcl owner finalized on MainThread', flush=True)


def _exercise_windows(close_method):
    """Run in a child so Tcl aborts cannot terminate unittest discovery."""
    from functools import partial
    import time
    import tkinter as tk
    from unittest.mock import patch

    from scripts.gui_smoke_shogi import wait_for
    from shogi.gui import App, HUMAN
    from tests.shogi_suite.test_gui_smoke import FakeAdvisor

    main_thread = threading.get_ident()
    finalizer_threads = []
    callback_errors = []
    variable_del = tk.Variable.__del__

    def finalize_variable(variable):
        finalizer_threads.append(threading.get_ident())
        variable_del(variable)

    def callback_error(exc_type, value, traceback):
        callback_errors.append(str(value))

    # Collect explicitly in workers to expose surviving Tk cycles deterministically.
    gc.disable()
    with patch.object(tk.Variable, '__del__', finalize_variable), \
            patch('shogi.gui.messagebox.showerror') as error:
        for _ in range(2):
            root = tk.Tk()
            root.report_callback_exception = callback_error
            app = App(root, advisor=FakeAdvisor())
            release = threading.Event()
            worker = None
            try:
                app.paused = True
                app.player_vars[0].set('ランダム')
                app.player_vars[1].set(HUMAN)
                app.change_players()
                app._request_ai()
                wait_for(root, lambda: app.game.ply == 1 and app._worker is None)
                assert len(app.game.moves) == 1

                # Exercise a worker still running when either close path executes.
                entered = threading.Event()
                app.player_vars[1].set('ランダム')
                agent = SimpleNamespace(choose=partial(_wait_for_release, entered, release))
                with patch('shogi.gui.RandomAgent', new=lambda: agent):
                    app._request_ai()
                    assert entered.wait(5), 'Worker did not start'
                worker, cancel, poll_id = app._worker, app._cancel, app._poll_id
                app_ref, root_ref = weakref.ref(app), weakref.ref(root)
                start = time.monotonic()
                if close_method == 'close':
                    app.close()
                else:
                    root.destroy()
                assert time.monotonic() - start < 2, 'Close blocked on an uncooperative worker'
                assert app._closed and cancel.is_set()
                assert app._poll_id is None
                assert poll_id not in root.tk.call('after', 'info')
                app.close()  # Both shutdown paths must be idempotent.
            finally:
                app.close()
                del app, root
                release.set()
                if worker is not None:
                    worker.join(5)
                    assert not worker.is_alive(), 'Released worker did not finish'
            assert app_ref() is None, 'Worker retained the closed App'
            assert root_ref() is None, 'Closed Tk root survived its lifecycle'
        assert not error.called, error.call_args
    assert finalizer_threads and all(t == main_thread for t in finalizer_threads), finalizer_threads
    assert not callback_errors, callback_errors
    print('Two Tk lifecycles passed:', close_method, flush=True)


def _discard_dialog_mock(root):
    """Match the smoke's cyclic Mock -> call kwargs -> parent root ownership."""
    from unittest.mock import patch

    with patch('shogi.gui.filedialog.askopenfilename', return_value='') as dialog:
        dialog(parent=root)


def _collect_in_worker():
    worker = threading.Thread(target=gc.collect, name='forced-gc')
    worker.start()
    worker.join(5)
    assert not worker.is_alive(), 'Worker GC did not finish'


def _exercise_closed_cycles(real_tk=True):
    """Collect the smoke's discarded dialog cycle on a different thread."""
    import tkinter as tk

    from shogi import create_game
    from shogi.gui import App, HUMAN
    from shogi.gui_support import History

    gc.disable()
    finalized = []
    for _ in range(2):
        if real_tk:
            from scripts.gui_smoke_shogi import scenario
            root = tk.Tk()
            # Include the real optional-module probe, promotion and record dialogs.
            app = App(root)
            scenario(app)
        else:
            class TclRoot(tk.Tk):
                def __init__(self):
                    super().__init__(useTk=False)

                def destroy(self):
                    # A real Tcl interpreter with no Tk window commands to destroy.
                    tk.Misc.destroy(self)

            root = TclRoot()
            app = App.__new__(App)
            app.root = root
            app.history = History(create_game())
            app._closed, app._generation = False, 0
            app._worker = app._cancel = app._poll_id = None
            app.player_vars = [tk.StringVar(root, value=HUMAN)]
            app.player_combos, app.promotion_buttons, app.hand_buttons = [], [], {}
        root_ref, app_ref = weakref.ref(root), weakref.ref(app)
        variable_refs = [weakref.ref(var) for var in app.player_vars]
        weakref.finalize(root, lambda: finalized.append(threading.current_thread().name))
        _discard_dialog_mock(root)
        app.close()
        variables_released = all(ref() is None for ref in variable_refs)
        del app, root
        # Record before GC, but collect before asserting: the unfixed code must
        # exercise the abort instead of stopping at a weakref assertion first.
        released_without_gc = root_ref() is None and app_ref() is None
        _collect_in_worker()
        assert released_without_gc, 'Closed owners required later cyclic collection'
        assert variables_released, 'Close retained Tk variables'
    assert finalized == ['MainThread', 'MainThread'], finalized
    print('Closed dialog cycles finalized on MainThread', flush=True)


def _exercise_live_dialogs():
    """Destroyed transient widgets must not strand Tcl owners after close."""
    import tkinter as tk
    from tkinter import simpledialog

    from shogi import create_game
    from shogi.gui import App
    from scripts.gui_smoke_shogi import click_square, wait_for
    from tests.shogi_suite.test_gui_smoke import FakeAdvisor

    gc.disable()
    root = tk.Tk()
    app = App(root, advisor=FakeAdvisor())
    app.paused = True
    try:
        app.replace_game(create_game(sfen='k8/9/4P4/9/9/9/9/9/4K4 b - 1'))
        click_square(app, 22)
        click_square(app, 13)
        dialog_ref = weakref.ref(app._promotion)
        button_refs = [weakref.ref(button) for button in app.promotion_buttons]
        app.finish_promotion(None)
        assert dialog_ref() is None and all(ref() is None for ref in button_refs)

        class AutoDialog(simpledialog.Dialog):
            def body(self, master):
                self.entry = tk.Entry(master)
                self.entry.pack()
                return self.entry

            def wait_window(self, window=None):
                # Schedule only after Dialog has waited for visibility and grabbed.
                self.after_idle(self.cancel)
                return super().wait_window(window)

        dialog = AutoDialog(root)
        dialog_ref = weakref.ref(dialog)
        del dialog
        # simpledialog retains Entry.master -> Dialog.entry after destroy.
        # With the App alive this cycle cannot be the interpreter's last owner.
        _collect_in_worker()
        assert dialog_ref() is None
        root.update()
        app.player_vars[0].set('ランダム')
        app._request_ai()
        wait_for(root, lambda: app.game.ply == 1 and app._worker is None)
    finally:
        app.close()
    root_ref = weakref.ref(root)
    del app, root
    assert root_ref() is None
    _collect_in_worker()
    print('Live transient dialogs passed', flush=True)


@unittest.skipUnless(os.environ.get('HASAMI_GUI_TESTS') == '1',
                     'Set HASAMI_GUI_TESTS=1 for real Tk events')
class TkLifecycleTests(unittest.TestCase):
    def test_worker_never_owns_tcl_interpreter(self):
        self._assert_child('_exercise_tcl_owner()', 'Tcl owner finalized on MainThread')

    def test_two_windows_with_ai_and_active_worker_close(self):
        self._assert_child("_exercise_windows('close')", 'Two Tk lifecycles passed:')

    def test_two_windows_with_ai_and_direct_root_destroy(self):
        self._assert_child("_exercise_windows('destroy')", 'Two Tk lifecycles passed:')

    def test_closed_dialog_cycles_before_worker_gc(self):
        self._assert_child('_exercise_closed_cycles()', 'Closed dialog cycles finalized on MainThread')

    def test_closed_tcl_dialog_cycles_without_display(self):
        self._assert_child('_exercise_closed_cycles(False)', 'Closed dialog cycles finalized on MainThread')

    def test_transient_dialogs_while_window_is_alive(self):
        self._assert_child('_exercise_live_dialogs()', 'Live transient dialogs passed')

    def test_unchanged_smoke_both_backends(self):
        self._assert_process([sys.executable, '-X', 'faulthandler',
                              'scripts/gui_smoke_shogi.py', '--backend', 'both'],
                             'Shogi GUI smoke passed: cpp')

    def _assert_child(self, expression, expected):
        code = ('from tests.shogi_suite.test_gui_lifecycle import '
                '_exercise_windows, _exercise_tcl_owner, '
                f'_exercise_closed_cycles, _exercise_live_dialogs; {expression}')
        self._assert_process([sys.executable, '-X', 'faulthandler', '-c', code], expected)

    def _assert_process(self, command, expected):
        result = subprocess.run(command,
                                cwd=Path(__file__).resolve().parents[2],
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(expected, result.stdout)
        self.assertNotIn('Exception ignored in:', result.stderr)


if __name__ == '__main__':
    unittest.main()
