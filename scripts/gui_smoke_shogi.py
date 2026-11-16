#!/usr/bin/env python3
"""Exercise real Tk events on macOS or under xvfb-run on Linux."""
import argparse
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tkinter as tk
from shogi import Rules, create_game
from shogi.gui import App, HUMAN
from shogi.model import decode_move
from shogi.notation import usi_to_move
from shogi.records import record_dict


def wait_for(root, predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        root.update()
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError('GUI operation timed out')


def click_square(app, sq):
    app.root.update()
    x, y = app.center(sq)
    app.canvas.event_generate('<Button-1>', x=int(x), y=int(y))
    app.root.update()


def click_button(root, button):
    root.update()
    x, y = button.winfo_width()//2, button.winfo_height()//2
    button.event_generate('<Enter>', x=x, y=y)
    button.event_generate('<ButtonPress-1>', x=x, y=y)
    button.event_generate('<ButtonRelease-1>', x=x, y=y)
    root.update()


def play(app, text, promote=None):
    code = usi_to_move(app.game, text)
    move = decode_move(code)
    before = app.game.ply
    if move.kind == 'drop':
        click_button(app.root, app.hand_buttons[app.game.position.turn, move.piece-1])
    else:
        click_square(app, move.from_sq)
    click_square(app, move.to_sq)
    if app._promotion is not None:
        if promote is None:
            promote = move.promote
        click_button(app.root, app.promotion_buttons[0 if promote else 1])
    if app.game.ply != before + 1 or app.game.moves[-1] != code:
        raise AssertionError(f'Tk move failed: {text}')


def scenario(app):
    root = app.root
    for var in app.player_vars:
        var.set(HUMAN)
    app.change_players()
    wait_for(root, lambda: app._worker is None)
    with patch('shogi.gui.messagebox.showerror') as error:
        for text in ('7g7f', '3c3d', '2g2f', '8c8d', '2f2e'):
            play(app, text)
        if app.game.ply != 5:
            raise AssertionError('Five-move scenario failed')
        original = record_dict(app.game)
        app.flip()
        root.update()
        if record_dict(app.game) != original:
            raise AssertionError('Flip changed the game')
        app.undo()
        root.update()
        play(app, '2f2e')
        if record_dict(app.game) != original:
            raise AssertionError('Undo and replay changed the record')
        with tempfile.TemporaryDirectory() as directory:
            for extension in ('.kifu', '.csa'):
                path = str(Path(directory) / ('smoke' + extension))
                with patch('shogi.gui.filedialog.asksaveasfilename', return_value=path):
                    app.save()
                app.new_game()
                with patch('shogi.gui.filedialog.askopenfilename', return_value=path):
                    app.load()
                if record_dict(app.game) != original:
                    raise AssertionError('Record reload mismatch')
        app.replace_game(create_game(app.backend, sfen='k8/9/4P4/9/9/9/9/9/4K4 b S 1'))
        play(app, '5c5b+', promote=True)
        play(app, '9a8a')
        play(app, 'S*4e')
        promoted = record_dict(app.game)
        app.undo()
        play(app, 'S*4e')
        if record_dict(app.game) != promoted:
            raise AssertionError('Drop undo mismatch')
        app.replace_game(create_game(app.backend, Rules(max_moves=8)))
        app.player_vars[0].set('ランダム')
        app.player_vars[1].set('αβ')
        app.depth_var.set('1')
        app.time_var.set('50')
        app.change_players()
        app.paused = False
        wait_for(root, lambda: app.game.outcome.terminal)
        if app.game.ply != 8 or app.game.outcome.reason != 'max_moves':
            raise AssertionError('Short classical game did not finish under engine rules')
        if error.called:
            raise AssertionError(f'Unexpected GUI error: {error.call_args}')


def run(backend='python'):
    root = tk.Tk()
    app = App(root, backend)
    try:
        scenario(app)
        print(f'Shogi GUI smoke passed: {backend}; five Tk moves, promotion, drop, undo, flip, KIF/CSA reload, classical game')
    finally:
        app.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=('python', 'cpp', 'both'), default='python')
    args = parser.parse_args()
    for backend in (('python', 'cpp') if args.backend == 'both' else (args.backend,)):
        run(backend)


if __name__ == '__main__':
    main()
