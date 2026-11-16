"""Display-free tests for GUI budgets and model registration."""
import queue
import subprocess
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

try:
    import tkinter
except ImportError:
    tkinter = None

if tkinter is not None:
    from hasami import gui
    from hasami.agents import AgentRegistry, Decision
    from hasami.records import record_dict


@unittest.skipIf(tkinter is None, "Tk is unavailable")
class GuiSpecTests(unittest.TestCase):
    def test_import_does_not_load_optional_packages(self):
        code = """
import sys
class RejectLearning:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'hasami_learn'}:
            raise AssertionError('Optional import: ' + fullname)
sys.meta_path.insert(0, RejectLearning())
import hasami.gui
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_classical_limits_keep_node_budget(self):
        limits = gui.compute_limits(3, 700, 200, "classical")
        self.assertEqual((limits.max_depth, limits.time_ms, limits.node_limit),
                         (3, 700, 200000))

    def test_non_classical_limits_use_sims(self):
        for family in ("alphazero", "supervised", "dqn", "ppo", "custom"):
            for sims in (1, 200, 4000):
                with self.subTest(family=family, sims=sims):
                    limits = gui.compute_limits(2, 10000, sims, family)
                    self.assertEqual((limits.max_depth, limits.time_ms, limits.node_limit),
                                     (2, 10000, sims))

    def test_invalid_budgets_are_rejected(self):
        for depth, time_ms, sims in ((0, 700, 200), (3, 0, 200),
                                     (3, 10001, 200), (3, 700, 0),
                                     (3, 700, -1), (3, 700, 4001)):
            for family in ("classical", "alphazero"):
                with self.subTest(depth=depth, time_ms=time_ms, sims=sims, family=family):
                    with self.assertRaises(ValueError):
                        gui.compute_limits(depth, time_ms, sims, family)

    def test_unique_agent_id(self):
        base = "model:checkpoint"
        for names, expected in (([], base), ([base], base + "-2"),
                                ([base, base + "-2"], base + "-3"),
                                ([base, base + "-3"], base + "-2"),
                                ([base + "-2"], base)):
            with self.subTest(names=names):
                original = names.copy()
                self.assertEqual(gui.unique_agent_id(base, iter(names)), expected)
                self.assertEqual(names, original)


@unittest.skipIf(tkinter is None, "Tk is unavailable")
class GuiWorkerTests(unittest.TestCase):
    def setUp(self):
        self.main_thread = threading.get_ident()
        app = self.app = gui.App.__new__(gui.App)
        app.registry = AgentRegistry()
        app.game = gui.create_game("python")
        self.addCleanup(lambda: app.game.close())
        app.root = self.widget()
        app.render = Mock()
        app.draw_board = Mock()
        app.model_button = self.widget()
        app.model_label = self.widget()
        app.thinking_label = self.widget()
        app.footer = self.widget()
        app.player_combos = [self.widget(), self.widget()]
        app.player_vars = [self.widget("human"), self.widget("human")]
        app.depth_var = self.widget("1")
        app.time_var = self.widget("50")
        app.sims_var = self.widget("200")
        app.backend_var = self.widget("python")
        app._closed = False
        app._generation = 0
        app._poll_id = None
        app._worker = app._model_worker = app._cancel = None
        app._results = queue.Queue()
        app._model_results = queue.Queue()
        app._agents = {}
        app._agent_names = {}
        app._assist_agent = app.registry.create("alphabeta")
        app.paused = True
        app.selected = app.hint_action = None

    def widget(self, value=None):
        def check_thread(*args, **kwargs):
            self.assertEqual(threading.get_ident(), self.main_thread)
        widget = Mock()
        widget.get.side_effect = lambda: (check_thread(), value)[1]
        widget.config.side_effect = check_thread
        return widget

    def finish_worker(self, worker):
        self.assertTrue(worker.daemon)
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive(), "Worker timed out")
        self.app._poll()

    def test_load_registers_after_new_game_and_factory_reloads(self):
        app = self.app
        inference = ModuleType("hasami_learn.inference")
        loaded = SimpleNamespace(manifest=SimpleNamespace(algorithm="alphazero"), choose=Mock())
        later = SimpleNamespace(choose=Mock())
        load_threads = []
        agents = iter((loaded, later))

        def create_agent(**kwargs):
            load_threads.append(threading.get_ident())
            return next(agents)

        inference.create_agent = Mock(side_effect=create_agent)
        inference.describe_agent = Mock(return_value="alphazero iter 3 / mode mcts")
        names = app.registry.names()
        with patch.dict(sys.modules, {"hasami_learn.inference": inference}), \
                patch.object(gui.filedialog, "askdirectory", return_value="/tmp/checkpoint"), \
                patch.object(gui.messagebox, "showerror") as error:
            app.load_model()
            worker = app._model_worker
            app.model_label.config.assert_called_with(text="読込中…")
            app.model_button.config.assert_called_with(state="disabled")
            self.assertEqual(app.registry.names(), names)
            app.new_game()
            app.paused = True
            original = record_dict(app.game)
            self.finish_worker(worker)
            self.assertNotEqual(load_threads[0], self.main_thread)
            error.assert_not_called()
            self.assertEqual(record_dict(app.game), original)
            self.assertEqual(app.registry.family("model:checkpoint"), "alphazero")
            self.assertEqual(app.registry.describe("model:checkpoint"), inference.describe_agent.return_value)
            app.model_label.config.assert_called_with(text=inference.describe_agent.return_value)
            for combo in app.player_combos:
                self.assertIn("model:checkpoint", combo.config.call_args.kwargs["values"])
            self.assertIs(app.registry.create("model:checkpoint"), loaded)
            self.assertIs(app.registry.create("model:checkpoint"), later)
            self.assertEqual(inference.create_agent.call_count, 2)
            inference.create_agent.assert_called_with(checkpoint="/tmp/checkpoint", device="auto",
                                                      mode="auto", simulations=200)

    def test_missing_learning_package_reports_venv_and_preserves_record(self):
        app = self.app
        original = record_dict(app.game)
        with patch.dict(sys.modules, {"hasami_learn.inference": None}), \
                patch.object(gui.filedialog, "askdirectory", return_value="/tmp/missing"), \
                patch.object(gui.messagebox, "showerror") as error:
            app.load_model()
            self.finish_worker(app._model_worker)
            error.assert_called_once()
            self.assertEqual(error.call_args.args[0], "モデル読込エラー")
            self.assertIn(".venv", error.call_args.args[1])
        self.assertEqual(record_dict(app.game), original)
        app.model_button.config.assert_called_with(state="normal")

    def test_loading_failure_preserves_record(self):
        app = self.app
        original = record_dict(app.game)
        inference = ModuleType("hasami_learn.inference")
        inference.create_agent = Mock(side_effect=ValueError("weights sha256 mismatch"))
        inference.describe_agent = Mock()
        with patch.dict(sys.modules, {"hasami_learn.inference": inference}), \
                patch.object(gui.filedialog, "askdirectory", return_value="/tmp/checkpoint"), \
                patch.object(gui.messagebox, "showerror") as error:
            app.load_model()
            self.finish_worker(app._model_worker)
            self.assertIn("weights sha256 mismatch", error.call_args.args[1])
        self.assertEqual(record_dict(app.game), original)
        self.assertFalse(any(name.startswith("model:") for name in app.registry.names()))

    def test_close_does_not_wait_for_model_or_apply_its_result(self):
        app = self.app
        release = threading.Event()
        inference = ModuleType("hasami_learn.inference")

        def create_agent(**kwargs):
            if not release.wait(timeout=5):
                raise RuntimeError("Test did not release the loader")
            return SimpleNamespace(manifest=SimpleNamespace(algorithm="ppo"), choose=Mock())

        inference.create_agent = create_agent
        inference.describe_agent = Mock(return_value="ppo iter 1 / mode policy")
        with patch.dict(sys.modules, {"hasami_learn.inference": inference}), \
                patch.object(gui.filedialog, "askdirectory", return_value="/tmp/checkpoint"):
            app.load_model()
            worker = app._model_worker
            try:
                app.close()
                self.assertTrue(worker.is_alive())
                self.assertTrue(app._closed)
            finally:
                release.set()
                worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
            app._poll()
        self.assertNotIn("model:checkpoint", app.registry.names())
        for combo in app.player_combos:
            combo.config.assert_not_called()

    def test_learned_factory_and_choose_run_off_main_thread(self):
        app = self.app
        original = record_dict(app.game)
        threads = []

        def choose(snapshot, limits, cancel):
            threads.append(threading.get_ident())
            self.assertIsNot(snapshot, app.game)
            self.assertEqual(record_dict(snapshot), original)
            self.assertEqual(limits.node_limit, 200)
            action = snapshot.legal_actions()[0]
            snapshot.play(action)
            return Decision(action, score=250, nodes=7, elapsed_ms=12)

        agent = SimpleNamespace(choose=choose)

        def factory():
            threads.append(threading.get_ident())
            return agent

        app.registry.register("model:test", factory, family="alphazero")
        app.player_vars[0] = self.widget("model:test")
        app.step_ai()
        self.finish_worker(app._worker)
        self.assertEqual(len(app.game.moves), 1)
        self.assertEqual(len(threads), 2)
        self.assertTrue(all(thread != self.main_thread for thread in threads))
        self.assertIs(app._agents[0], agent)
        app.thinking_label.config.assert_called_with(text="sims 7 / 12 ms / value +0.25")

    def test_stale_learned_result_is_discarded(self):
        app = self.app
        agent = SimpleNamespace(choose=lambda game, limits, cancel: Decision(game.legal_actions()[0]))
        app.registry.register("model:test", lambda: agent, family="alphazero")
        app.player_vars[0] = self.widget("model:test")
        app.step_ai()
        worker = app._worker
        app.new_game()
        app.paused = True
        self.finish_worker(worker)
        self.assertEqual(app.game.moves, [])
        self.assertEqual(app._agents, {})
        app.thinking_label.config.assert_called_with(text="以前の検討結果を破棄しました。")

    def test_hint_and_human_assist_use_classical_limits(self):
        app = self.app
        self.assertEqual(app._limits().node_limit, 200000)
        app.registry.register("model:test", lambda: None, family="alphazero")
        app.player_vars[0] = self.widget("model:test")
        self.assertEqual(app._limits().node_limit, 200)
        app._assist_agent = Mock()
        app._assist_agent.choose.return_value = Decision(app.game.legal_actions()[0])
        original = record_dict(app.game)
        app.hint()
        self.finish_worker(app._worker)
        self.assertEqual(app._assist_agent.choose.call_args.args[1].node_limit, 200000)
        self.assertEqual(record_dict(app.game), original)
        self.assertIsNotNone(app.hint_action)


if __name__ == "__main__":
    unittest.main()
