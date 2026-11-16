"""Exercise real Tk events. On headless Linux, run this under xvfb-run."""
import argparse
import os
from pathlib import Path
import sys
import time
import tempfile
from unittest.mock import patch
import tkinter as tk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hasami.gui import App, compute_limits
from hasami.records import record_dict


def wait_for(root, predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        root.update()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("GUI condition timed out")


def click_square(app, square):
    r, c = divmod(square, 9)
    app.canvas.event_generate("<Button-1>", x=int(app.origin_x + (c + .5) * app.cell),
                               y=int(app.origin_y + (r + .5) * app.cell))
    app.root.update()


def check_illegal_clicks(app):
    """Reach a friendly blocker through clicks and reject jump/occupied targets."""
    for source, target in ((73, 55), (8, 17), (55, 54), (17, 26)):
        previous = app.game.moves.copy()
        click_square(app, source)
        click_square(app, target)
        if app.game.moves != previous + [source * 81 + target]:
            raise AssertionError("Illegal-click setup move failed")
    click_square(app, 72)
    original = record_dict(app.game)
    for target in (45, 0):
        # Square 54 blocks the file; square 0 is occupied by an enemy.
        app.footer.config(text="")
        click_square(app, target)
        if record_dict(app.game) != original or app.selected != 72:
            raise AssertionError("Illegal click changed the record or selection")
        if app.footer["text"] != "そこには動けません。緑の点から移動先を選んでください。":
            raise AssertionError("Illegal click did not display the destination hint")
    app.new_game()
    app.root.update()


def check_models(app):
    """Exercise model controls without requiring the optional learning package."""
    root = app.root
    original = record_dict(app.game)
    turn = app.game.position.turn
    previous = app.player_vars[turn].get()
    if not app.sims_spinbox.winfo_exists() or app.sims_var.get() != "200":
        raise AssertionError("Sims control or its default is missing")
    app.registry.register("smoke:learned", lambda: None, family="alphazero")
    try:
        app.player_vars[turn].set("smoke:learned")
        if app._limits().node_limit != 200:
            raise AssertionError("Sims was not passed to the learned agent limits")
        if compute_limits(2, 200, 200, "alphazero").node_limit != 200:
            raise AssertionError("compute_limits did not preserve Sims")
    finally:
        app.player_vars[turn].set(previous)
        app.registry.unregister("smoke:learned")
    with tempfile.TemporaryDirectory() as directory:
        missing = str(Path(directory) / "missing-checkpoint")
        with patch("hasami.gui.filedialog.askdirectory", return_value=missing), \
                patch("hasami.gui.messagebox.showerror") as error:
            app.load_model()
            if str(app.model_button["state"]) != "disabled":
                raise AssertionError("Model button stayed enabled during loading")
            wait_for(root, lambda: app._model_worker is None, timeout=60)
            error.assert_called_once()
            if error.call_args.args[0] != "モデル読込エラー":
                raise AssertionError("Wrong model loading error dialog")
    if record_dict(app.game) != original:
        raise AssertionError("Failed model loading changed the record")
    if str(app.model_button["state"]) != "normal":
        raise AssertionError("Model button was not restored after failure")

    checkpoint = os.environ.get("HASAMI_GUI_SMOKE_CHECKPOINT")
    if not checkpoint or not Path(checkpoint).is_dir():
        print("GUI model load skipped: HASAMI_GUI_SMOKE_CHECKPOINT directory unavailable")
        return
    try:
        import torch
    except ImportError as exc:
        print(f"GUI model load skipped: torch unavailable ({exc})")
        return
    names = set(app.registry.names())
    with patch("hasami.gui.filedialog.askdirectory", return_value=checkpoint), \
            patch("hasami.gui.messagebox.showerror") as error:
        app.load_model()
        wait_for(root, lambda: app._model_worker is None, timeout=60)
        if error.called:
            raise AssertionError(f"Real model loading failed: {error.call_args}")
        added = set(app.registry.names()) - names
        if len(added) != 1 or not next(iter(added)).startswith("model:"):
            raise AssertionError("Loaded model was not registered")
        name = added.pop()
        if any(name not in combo["values"] for combo in app.player_combos):
            raise AssertionError("Loaded model is missing from a player combobox")
        if record_dict(app.game) != original:
            raise AssertionError("Successful model loading changed the record")
        legal = app.game.legal_actions()
        app.player_vars[turn].set(name)
        app.change_players()
        app.step_ai()
        wait_for(root, lambda: app._worker is None, timeout=60)
        if error.called:
            raise AssertionError(f"Loaded agent failed: {error.call_args}")
        if len(app.game.moves) != len(original["moves"]) + 1 or app.game.moves[-1] not in legal:
            raise AssertionError("Loaded model did not play one legal move")
    app.undo()
    app.player_vars[turn].set(previous)
    app.change_players()
    if record_dict(app.game) != original:
        raise AssertionError("Undo after the model move lost history")
    print("GUI real model load passed")


def run(backend, screenshot=None):
    root = tk.Tk()
    app = App(root, backend)
    try:
        app.player_vars[0].set("human")
        app.player_vars[1].set("human")
        root.update()
        check_illegal_clicks(app)
        click_square(app, 72)
        if app.selected != 72:
            raise AssertionError("Piece selection failed")
        click_square(app, 63)
        if app.game.moves != [72*81+63]:
            raise AssertionError("Mouse move failed")
        app.undo()
        root.update()
        if app.game.moves:
            raise AssertionError("GUI undo failed")
        app.time_var.set("200")
        app.depth_var.set("2")
        app.step_ai()
        wait_for(root, lambda: len(app.game.moves) == 1)
        if app._worker is not None:
            wait_for(root, lambda: app._worker is None)
        # A fresh game invalidates any result from the previous generation.
        app.step_ai()
        app.new_game()
        wait_for(root, lambda: app._worker is None)
        if app.game.moves:
            raise AssertionError("A stale worker result modified the new game")
        app.game.play(72*81+36)
        app.game.play(8*81+44)
        app.game.play(76*81+40)
        app.paused = True
        app.selected = 4
        app.render()
        root.update()
        original = record_dict(app.game)
        app.hint()
        wait_for(root, lambda: app._worker is None)
        if record_dict(app.game) != original:
            raise AssertionError("Hint mutated the game")
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "gui_record.json")
            with patch("hasami.gui.filedialog.asksaveasfilename", return_value=path):
                app.save()
            app.new_game()
            with patch("hasami.gui.filedialog.askopenfilename", return_value=path):
                app.load()
            if record_dict(app.game) != original:
                raise AssertionError("GUI save/load lost history")
        app.backend_var.set("cpp" if backend == "python" else "python")
        app.switch_backend()
        if record_dict(app.game) != original:
            raise AssertionError("Backend switching changed the record")
        app.backend_var.set(backend)
        app.switch_backend()
        check_models(app)
        root.geometry("930x790")
        root.update()
        for widget in (app.footer, app.rules_label):
            bottom = widget.winfo_rooty() - root.winfo_rooty() + widget.winfo_height()
            if bottom > root.winfo_height():
                raise AssertionError("A required GUI label is clipped at minimum size")
        root.geometry("1080x870")
        app.selected = 4
        app.hint()
        wait_for(root, lambda: app._worker is None)
        if screenshot:
            from PIL import ImageGrab
            root.update()
            x, y = root.winfo_rootx(), root.winfo_rooty()
            ImageGrab.grab(bbox=(x, y, x + root.winfo_width(), y + root.winfo_height())).save(screenshot)
        print(f"GUI smoke passed: {backend} (click, illegal click, move, undo, AI, stale result, hint, save/load, backend switch, resize, Sims, model error)")
    finally:
        app.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("python", "cpp", "both"), default="both")
    parser.add_argument("--screenshot")
    args = parser.parse_args()
    for backend in (("python", "cpp") if args.backend == "both" else (args.backend,)):
        run(backend, args.screenshot if backend == "cpp" or args.backend == "python" else None)


if __name__ == "__main__":
    main()
