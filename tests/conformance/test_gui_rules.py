"""G1-G11 conformance through real Tk events; G9 needs no display."""

import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from hasami import create_game
from hasami.gui_support import compute_limits, unique_agent_id  # display-free: G9 always runs
from hasami.model import move_name
from hasami.records import record_dict

try:  # Tk is optional at import time: discovery must not fail on interpreters without _tkinter.
    import tkinter as tk
    from hasami.gui import App
    from scripts.gui_smoke import click_square, wait_for

    TK_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - depends on the interpreter build
    tk = None  # type: ignore[assignment]
    App = click_square = wait_for = None  # type: ignore[assignment]
    TK_AVAILABLE = False
    TK_IMPORT_ERROR = str(exc)


class GuiLimitsTests(unittest.TestCase):
    def test_G9_family_budgets(self):
        """G9: Classical nodes stay fixed; every learned family uses Sims."""
        for family in ("classical", "alphazero", "supervised", "dqn", "ppo",
                       "reinforce", "adversarial", "custom"):
            for sims in (1, 200, 4000):
                with self.subTest(family=family, sims=sims):
                    limits = compute_limits(2, 10, sims, family)
                    self.assertEqual((limits.max_depth, limits.time_ms), (2, 10))
                    self.assertEqual(limits.node_limit,
                                     200000 if family == "classical" else sims)

    def test_G9_invalid_budgets(self):
        """G9: Reject invalid GUI budgets instead of silently clamping them."""
        for family in ("classical", "alphazero"):
            for depth, ms, sims in ((0, 10, 200), (2, 0, 200), (2, 10001, 200),
                                    (2, 10, 0), (2, 10, 4001), (2, 10, True)):
                with self.subTest(family=family, depth=depth, ms=ms, sims=sims):
                    with self.assertRaises(ValueError):
                        compute_limits(depth, ms, sims, family)

    def test_G9_unique_agent_ids(self):
        """G9: Registration uses the first free suffix without changing names."""
        base = "model:checkpoint"
        for names, expected in (([], base), ([base], base + "-2"),
                                ([base, base + "-2"], base + "-3"),
                                ([base, base + "-3"], base + "-2"),
                                ([base + "-2"], base)):
            with self.subTest(names=names):
                original = names.copy()
                self.assertEqual(unique_agent_id(base, iter(names)), expected)
                self.assertEqual(names, original)


class GuiRulesCases:
    """Share the event scenarios between independent game backends."""

    def setUp(self):
        root = self.root = tk.Tk()
        try:
            self.app = App(root, self.backend)
        except Exception:
            root.destroy()
            raise
        self.addCleanup(self.app.close)
        error_patch = patch("hasami.gui.messagebox.showerror")
        self.error = error_patch.start()
        self.addCleanup(error_patch.stop)
        self.app.player_vars[0].set("human")
        self.app.player_vars[1].set("human")
        self.app.change_players()
        self.app.depth_var.set("1")
        self.app.time_var.set("10")
        self.root.update()

    def tearDown(self):
        self.error.assert_not_called()

    def play_clicks(self, *moves):
        for source, target in moves:
            before = self.app.game.moves.copy()
            action = source * 81 + target
            self.assertIn(action, self.app.game.legal_actions())
            click_square(self.app, source)
            click_square(self.app, target)
            self.assertEqual(self.app.game.moves, before + [action])

    def select_player(self, side, name):
        combo = self.app.player_combos[side]
        combo.set(name)
        combo.event_generate("<<ComboboxSelected>>")
        self.root.update()

    def assert_record_stays(self, original, duration=0.15):
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            self.root.update()
            self.assertEqual(record_dict(self.app.game), original)
            time.sleep(0.01)

    def piece_glyphs(self, square):
        app = self.app
        row, col = divmod(square, 9)
        x = app.origin_x + (col + 0.5) * app.cell
        y = app.origin_y + (row + 0.5) * app.cell
        items = app.canvas.find_enclosed(x - app.cell / 2, y - app.cell / 2,
                                         x + app.cell / 2, y + app.cell / 2)
        return [app.canvas.itemcget(item, "text") for item in items
                if app.canvas.type(item) == "text"
                and app.canvas.itemcget(item, "text") in ("歩", "と")]

    def test_G1_selection(self):
        """G1: Only a friendly first click selects a piece."""
        original = record_dict(self.app.game)
        for square in (0, 40):
            click_square(self.app, square)
            self.assertIsNone(self.app.selected)
            self.assertEqual(record_dict(self.app.game), original)
        click_square(self.app, 72)
        self.assertEqual(self.app.selected, 72)
        self.assertEqual(record_dict(self.app.game), original)
        self.assertEqual(sum(self.app.canvas.type(item) == "oval"
                             for item in self.app.canvas.find_all()), 7)

    def test_G2_illegal_destinations(self):
        """G2: A friendly-blocker jump, enemy occupancy and diagonal are rejected."""
        self.play_clicks((73, 55), (8, 17), (55, 54), (17, 26))
        click_square(self.app, 72)
        original = record_dict(self.app.game)
        history = self.app.game.history_keys()
        self.assertTrue(self.app.game.position.black & (1 << 54))
        self.assertTrue(self.app.game.position.white & 1)
        for kind, target in (("jump over 54", 45), ("occupied enemy", 0),
                             ("diagonal", 64)):
            with self.subTest(kind=kind):
                self.assertNotIn(72 * 81 + target, self.app.game.legal_actions())
                # Clear the previous hint so each click must produce its own feedback.
                self.app.footer.config(text="")
                click_square(self.app, target)
                self.assertEqual(record_dict(self.app.game), original)
                self.assertEqual(self.app.game.history_keys(), history)
                self.assertEqual(self.app.selected, 72)
                self.assertEqual(self.app.footer["text"],
                                 "そこには動けません。緑の点から移動先を選んでください。")

    def test_G3_legal_move_updates_turn_and_log(self):
        """G3: Legal clicks switch the displayed side and append exactly one row."""
        self.assertEqual(self.app.status["text"], "先手（歩）の手番")
        for index, (source, target) in enumerate(((72, 63), (0, 9))):
            self.play_clicks((source, target))
            self.assertIsNone(self.app.selected)
            self.assertEqual(self.app.status["text"],
                             ("後手（と）の手番", "先手（歩）の手番")[index])
            self.assertEqual(self.app.log.size(), index + 1)
            side = ("歩", "と")[index]
            self.assertEqual(self.app.log.get(index),
                             f"{index + 1:03d}   {side}   {move_name(source * 81 + target)}")

    def test_G4_capture_updates_score_and_canvas(self):
        """G4: A click capture removes the enemy glyph and increments the score."""
        self.play_clicks((72, 36), (1, 37))
        self.assertEqual(self.piece_glyphs(37), ["と"])
        self.assertEqual(self.app.score_label["text"], "獲得   歩  0  :  0  と")
        self.play_clicks((74, 38))
        self.assertEqual(self.app.score_label["text"], "獲得   歩  1  :  0  と")
        self.assertFalse(self.app.game.position.white & (1 << 37))
        self.assertEqual(self.piece_glyphs(37), [])
        self.assertEqual(self.piece_glyphs(38), ["歩"])
        self.assertEqual(self.piece_glyphs(74), [])
        self.assertIn("1 枚獲得", self.app.footer["text"])

    def test_G5_undo_restores_capture_and_history(self):
        """G5: Undo restores the full record, status, score, glyph and move log."""
        self.play_clicks((72, 36), (1, 37))
        original = record_dict(self.app.game)
        history = self.app.game.history_keys()
        status, score = self.app.status["text"], self.app.score_label["text"]
        log = self.app.log.get(0, "end")
        self.play_clicks((74, 38))
        self.app.undo()
        self.root.update()
        self.assertEqual(record_dict(self.app.game), original)
        self.assertEqual(self.app.game.history_keys(), history)
        self.assertEqual(self.app.status["text"], status)
        self.assertEqual(self.app.score_label["text"], score)
        self.assertEqual(self.app.log.get(0, "end"), log)
        self.assertEqual(self.piece_glyphs(37), ["と"])
        self.assertTrue(self.app.paused)

    def test_G6_ai_alternates_legal_moves_and_pauses(self):
        """G6: Real random agents alternate legal moves and stop when paused."""
        self.select_player(0, "random")
        self.select_player(1, "random")
        self.app.pause_button.invoke()
        wait_for(self.root, lambda: self.app.game.outcome.terminal
                 or len(self.app.game.moves) >= 12)
        if self.app.game.outcome.terminal:
            self.assertTrue(self.app.paused)
        else:
            self.app.pause_button.invoke()
        original = record_dict(self.app.game)
        wait_for(self.root, lambda: self.app._worker is None)
        self.assert_record_stays(original)
        self.assertTrue(self.app.paused)
        replay = create_game(self.backend, self.app.game.rules, self.app.game.initial)
        self.addCleanup(replay.close)
        for index, action in enumerate(self.app.game.moves):
            self.assertEqual(replay.position.turn, (replay.initial.turn + index) % 2)
            self.assertIn(action, replay.legal_actions())
            mover = "歩" if replay.position.turn == 0 else "と"
            self.assertEqual(self.app.log.get(index),
                             f"{index + 1:03d}   {mover}   {move_name(action)}")
            replay.play(action)
        self.assertEqual(record_dict(replay), original)

    def test_G6_terminal_stops_clicks_and_agents(self):
        """G6: A real repetition terminal cannot acquire further human or AI moves."""
        cycle = ((72, 63), (0, 9), (63, 72), (9, 0))
        self.play_clicks(*(cycle * 3))
        self.assertTrue(self.app.game.outcome.terminal)
        self.assertEqual(self.app.game.outcome.reason, "repetition")
        self.assertTrue(self.app.paused)
        self.assertIn("引き分け", self.app.status["text"])
        original = record_dict(self.app.game)
        click_square(self.app, 72)
        click_square(self.app, 63)
        self.select_player(0, "random")
        self.select_player(1, "random")
        self.app.pause_button.invoke()
        self.assert_record_stays(original)
        self.assertIsNone(self.app._worker)
        self.app.step_ai()
        self.assert_record_stays(original)
        self.assertIsNone(self.app._worker)

    def test_G7_hint_preserves_record(self):
        """G7: A real hint displays an arrow without changing the record."""
        self.play_clicks((72, 36), (1, 37))
        original = record_dict(self.app.game)
        history = self.app.game.history_keys()
        self.app.hint()
        wait_for(self.root, lambda: self.app._worker is None)
        self.assertEqual(record_dict(self.app.game), original)
        self.assertEqual(self.app.game.history_keys(), history)
        self.assertIn(self.app.hint_action, self.app.game.legal_actions())
        self.assertIn("候補：", self.app.footer["text"])
        self.assertTrue(any(self.app.canvas.itemcget(item, "arrow") == "last"
                            for item in self.app.canvas.find_all()
                            if self.app.canvas.type(item) == "line"))

    def test_G7_failed_model_load_preserves_record(self):
        """G7: Real loading of a missing checkpoint reports failure atomically."""
        self.play_clicks((72, 36), (1, 37))
        original = record_dict(self.app.game)
        names = self.app.registry.names()
        with tempfile.TemporaryDirectory() as directory:
            with patch("hasami.gui.filedialog.askdirectory",
                       return_value=str(Path(directory) / "missing")):
                self.app.model_button.invoke()
            self.assertEqual(str(self.app.model_button["state"]), "disabled")
            wait_for(self.root, lambda: self.app._model_worker is None, timeout=60)
        self.error.assert_called_once()
        self.assertEqual(self.error.call_args.args[0], "モデル読込エラー")
        self.error.reset_mock()
        self.assertEqual(record_dict(self.app.game), original)
        self.assertEqual(self.app.registry.names(), names)
        self.assertEqual(str(self.app.model_button["state"]), "normal")

    def assert_stale_discarded(self, operation):
        self.play_clicks((72, 63), (0, 9))
        self.app.step_ai()
        worker = self.app._worker
        self.assertIsNotNone(worker)
        # Complete a real decision without letting Tk consume the queued result.
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive(), "Agent did not finish")
        self.assertFalse(self.app._results.empty())
        operation()
        original = record_dict(self.app.game)
        wait_for(self.root, lambda: self.app._worker is None)
        self.assertEqual(record_dict(self.app.game), original)
        self.assertEqual(self.app.thinking_label["text"], "以前の検討結果を破棄しました。")
        self.assert_record_stays(original)

    def test_G8_new_game_discards_stale_result(self):
        """G8: New game discards an already completed old-generation decision."""
        self.assert_stale_discarded(self.app.new_game)
        self.assertEqual(self.app.game.moves, [])

    def test_G8_undo_discards_stale_result(self):
        """G8: Undo discards an already completed old-generation decision."""
        self.assert_stale_discarded(self.app.undo)
        self.assertEqual(self.app.game.moves, [72 * 81 + 63])

    def test_G8_player_change_discards_stale_result(self):
        """G8: Player selection discards a result that is still a legal move."""
        self.assert_stale_discarded(lambda: self.select_player(0, "greedy"))
        self.assertEqual(self.app.game.moves, [72 * 81 + 63, 9])

    def test_G10_save_load_roundtrip(self):
        """G10: Real file save/load restores the record and repetition history."""
        self.play_clicks((72, 63), (0, 9), (63, 72), (9, 0), (72, 54))
        original = record_dict(self.app.game)
        history = self.app.game.history_keys()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            with patch("hasami.gui.filedialog.asksaveasfilename", return_value=str(path)):
                self.app.save()
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), original)
            self.app.new_game()
            with patch("hasami.gui.filedialog.askopenfilename", return_value=str(path)):
                self.app.load()
        self.root.update()
        self.assertEqual(record_dict(self.app.game), original)
        self.assertEqual(self.app.game.history_keys(), history)
        self.assertEqual(self.app.log.size(), len(original["moves"]))

    def test_G10_backend_switch_preserves_record(self):
        """G10: Both backend switch directions preserve the full game history."""
        from tests.test_rules import native_available

        if not native_available():
            self.skipTest("Native backend unavailable")
        self.play_clicks((72, 63), (0, 9), (63, 72), (9, 0), (72, 54))
        original = record_dict(self.app.game)
        history = self.app.game.history_keys()
        for backend in ("cpp" if self.backend == "python" else "python", self.backend):
            self.app.backend_combo.set(backend)
            self.app.backend_combo.event_generate("<<ComboboxSelected>>")
            self.root.update()
            self.assertEqual(self.app.game.backend, backend)
            self.assertEqual(record_dict(self.app.game), original)
            self.assertEqual(self.app.game.history_keys(), history)

    def assert_minimum_layout(self):
        self.root.geometry("930x790")
        self.root.update()
        for widget in (self.app.footer, self.app.rules_label):
            with self.subTest(widget=widget):
                self.assertTrue(widget.winfo_ismapped())
                self.assertTrue(widget["text"])
                x = widget.winfo_rootx() - self.root.winfo_rootx()
                y = widget.winfo_rooty() - self.root.winfo_rooty()
                self.assertGreaterEqual(x, 0)
                self.assertGreaterEqual(y, 0)
                self.assertGreaterEqual(widget.winfo_height(), widget.winfo_reqheight())
                self.assertLessEqual(x + widget.winfo_width(), self.root.winfo_width())
                self.assertLessEqual(y + widget.winfo_height(), self.root.winfo_height())

    def test_G11_minimum_layout(self):
        """G11: Footer and rules remain visible at the minimum window size."""
        self.assert_minimum_layout()

    @unittest.skipUnless(os.environ.get("HASAMI_GUI_SMOKE_CHECKPOINT"),
                         "HASAMI_GUI_SMOKE_CHECKPOINT is not set")
    def test_G11_G7_real_model_layout_and_record(self):
        """G11/G7: Real model loading preserves the record and minimum layout."""
        self.play_clicks((72, 36), (1, 37))
        self.assert_minimum_layout()
        original = record_dict(self.app.game)
        names = set(self.app.registry.names())
        self.app.sims_var.set("1")
        with patch("hasami.gui.filedialog.askdirectory",
                   return_value=os.environ["HASAMI_GUI_SMOKE_CHECKPOINT"]):
            self.app.model_button.invoke()
        wait_for(self.root, lambda: self.app._model_worker is None, timeout=60)
        self.error.assert_not_called()
        added = set(self.app.registry.names()) - names
        self.assertEqual(len(added), 1)
        name = added.pop()
        self.assertTrue(name.startswith("model:"))
        for combo in self.app.player_combos:
            self.assertIn(name, combo["values"])
        self.assertTrue(self.app.model_label["text"])
        self.assertNotIn("\n", self.app.model_label["text"])
        self.assertEqual(record_dict(self.app.game), original)
        self.assert_minimum_layout()
        self.select_player(self.app.game.position.turn, name)
        legal = self.app.game.legal_actions()
        self.app.step_ai()
        wait_for(self.root, lambda: self.app._worker is None, timeout=60)
        self.assertEqual(len(self.app.game.moves), len(original["moves"]) + 1)
        self.assertIn(self.app.game.moves[-1], legal)
        self.app.undo()
        self.assertEqual(record_dict(self.app.game), original)


@unittest.skipUnless(os.environ.get("HASAMI_GUI_TESTS") == "1",
                     "Set HASAMI_GUI_TESTS=1 to run real Tk events")
class PythonGuiRulesTests(GuiRulesCases, unittest.TestCase):
    backend = "python"


@unittest.skipUnless(os.environ.get("HASAMI_GUI_TESTS") == "1",
                     "Set HASAMI_GUI_TESTS=1 to run real Tk events")
class NativeGuiRulesTests(GuiRulesCases, unittest.TestCase):
    backend = "cpp"

    @classmethod
    def setUpClass(cls):
        from tests.test_rules import native_available

        if not native_available():
            raise unittest.SkipTest("Native backend unavailable")


if __name__ == "__main__":
    unittest.main()
