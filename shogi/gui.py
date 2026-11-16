"""Japanese shogi workbench; Tk stays on the main thread and agents receive clones."""
from __future__ import annotations

import argparse
from functools import partial
import gc
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, simpledialog, ttk

from hasami.agents import AgentCancelled, SearchLimits
from . import create_game
from .agents import AlphaBetaAgent, GreedyAgent, RandomAgent
from .gui_support import (History, InputState, MoveCodeAgent, advice_rows, advisor_methods, analyze, click_hand, click_square,
                          choose_promotion, declaration, discover_models, display_to_square,
                          get_advisor, highlights, level_text, load_strong_agent,
                          outcome_text, piece_name, run_worker, square_to_display)
from .model import decode_move
from .records import load_record, save_record

HUMAN = '人間'
METHODS = {'評価値ベース（探索）': 'search', '学習ベース': 'learned'}
CLASSICAL = ('ランダム', '貪欲', 'αβ')
WOOD, INK, BG = '#e7cca1', '#263243', '#f2f0e9'
WORKER_JOIN_TIMEOUT = .2


class App:
    def __init__(self, root, backend='python', *, advisor=None, models_directory=None):
        self.root, self.backend = root, backend
        self.history = History(create_game(backend))
        self.input = InputState()
        self.flipped = False
        self.paused = False
        self._closed = False
        self._generation = 0
        self._worker = None
        self._cancel = None
        self._results = queue.Queue()
        self._poll_id = None
        self._promotion = None
        self.promotion_buttons = []
        self._analysis_generation = -1
        self._hints = []
        self._best = None
        self._agents = {}
        self._advisors = {}
        self._injected_advisor = advisor
        self._available = advisor is not None
        self._labels = []
        self.models = {}
        self._levels = []
        self._level_model = None
        self.cell, self.origin_x, self.origin_y = 56, 35, 35
        root.title('将棋 Lab')
        root.geometry('1160x900')
        root.minsize(1040, 820)
        root.configure(bg=BG)
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.bind('<Destroy>', self._on_destroy, add='+')
        families = set(tkfont.families(root))
        self.jp_font = next((name for name in ('Hiragino Sans', 'Yu Gothic', 'Noto Sans CJK JP', 'Meiryo')
                             if name in families), 'TkDefaultFont')
        ttk.Style(root).configure('TButton', font=(self.jp_font, 10))
        self._build()
        try:
            self.models = {'モデル：' + name: path for name, path in discover_models(models_directory).items()}
        except (OSError, ValueError) as exc:
            self.model_status.set(str(exc))
        self._refresh_choices()
        if advisor is not None:
            self._set_levels(advisor.levels)
            self.model_status.set('テスト助言器を使用中')
            self._enable_learned(True)
        else:
            self._enable_learned(False)
            self._launch('availability', self._probe_optional)
        self._refresh_labels()
        self.render()
        self._poll_id = root.after(30, self._poll)

    @property
    def game(self):
        return self.history.game

    def _build(self):
        menu = tk.Menu(self.root)
        file_menu = tk.Menu(menu, tearoff=False)
        for label, command in [('新規対局（平手）', self.new_game), ('新規対局（SFEN）', self.new_sfen),
                               ('棋譜を開く', self.load), ('棋譜を保存', self.save),
                               ('SFEN をコピー', self.copy_sfen), ('SFEN を貼り付け', self.paste_sfen),
                               ('モデル読込', self.load_model), ('終了', self.close)]:
            file_menu.add_command(label=label, command=command)
        menu.add_cascade(label='ファイル', menu=file_menu)
        menu.add_command(label='ヘルプ', command=self.help)
        self.root.config(menu=menu)
        self.status = ttk.Label(self.root, font=(self.jp_font, 14))
        self.status.pack(anchor='w', padx=18, pady=(12, 4))
        body = ttk.Frame(self.root)
        body.pack(fill='both', expand=True, padx=16, pady=8)
        board = ttk.Frame(body)
        board.pack(side='left', fill='both', expand=True, padx=(0, 12))
        self.top_hand = ttk.Frame(board)
        self.top_hand.pack(fill='x')
        self.canvas = tk.Canvas(board, width=590, height=590, bg=BG, highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Button-1>', self.on_click)
        self.canvas.bind('<Configure>', lambda event: self.draw_board())
        self.bottom_hand = ttk.Frame(board)
        self.bottom_hand.pack(fill='x')
        self.hand_buttons = {}
        controls = ttk.Frame(board)
        controls.pack(fill='x', pady=8)
        for label, command in [('新規対局', self.new_game), ('待った', self.undo), ('投了', self.resign),
                               ('盤面反転', self.flip)]:
            ttk.Button(controls, text=label, command=command).pack(side='left', padx=2)
        self.declare_button = ttk.Button(controls, text='宣言', command=self.declare)
        self.declare_button.pack(side='left', padx=2)
        self.footer = ttk.Label(board, text='駒または持駒を選択 → 移動先をクリック', wraplength=580)
        self.footer.pack(fill='x', pady=(0, 8))
        side = ttk.Frame(body, width=430)
        side.pack(side='right', fill='y')
        side.pack_propagate(False)
        self.player_vars, self.player_combos = [], []
        for player, label in enumerate(('先手', '後手')):
            row = ttk.Frame(side)
            row.pack(fill='x', pady=3)
            ttk.Label(row, text=label).pack(side='left')
            var = tk.StringVar(self.root, value=HUMAN if player == 0 else 'αβ')
            combo = ttk.Combobox(row, textvariable=var, state='readonly', width=32)
            combo.pack(side='right')
            combo.bind('<<ComboboxSelected>>', self.change_players)
            self.player_vars.append(var)
            self.player_combos.append(combo)
        limits = ttk.Frame(side)
        limits.pack(fill='x', pady=4)
        self.depth_var, self.time_var, self.nodes_var = (tk.StringVar(self.root, value=x) for x in ('2', '700', '200000'))
        for label, var, start, stop, width in [('深さ', self.depth_var, 1, 6, 2),
                ('時間 ms', self.time_var, 1, 10000, 5), ('ノード', self.nodes_var, 1, 2000000, 7)]:
            ttk.Label(limits, text=label).pack(side='left')
            ttk.Spinbox(limits, textvariable=var, from_=start, to=stop, width=width).pack(side='left')
        self.pause_button = ttk.Button(side, text='AI を停止', command=self.toggle_pause)
        self.pause_button.pack(fill='x', pady=4)
        self.thinking = ttk.Label(side, text='', wraplength=420)
        self.thinking.pack(fill='x')
        model_row = ttk.Frame(side)
        model_row.pack(fill='x', pady=4)
        self.model_var = tk.StringVar(self.root)
        self.model_combo = ttk.Combobox(model_row, textvariable=self.model_var, state='readonly', width=27)
        self.model_combo.pack(side='left', fill='x', expand=True)
        self.model_combo.bind('<<ComboboxSelected>>', self.change_analysis)
        self.model_button = ttk.Button(model_row, text='モデル読込', command=self.load_model)
        self.model_button.pack(side='right')
        self.model_status = tk.StringVar(self.root, value='学習機能を確認中…')
        ttk.Label(side, textvariable=self.model_status, wraplength=420).pack(fill='x')
        self.analysis_var = tk.BooleanVar(self.root, value=False)
        self.analysis_check = ttk.Checkbutton(side, text='検討モード（間違いにくい手）',
                                              variable=self.analysis_var, command=self.change_analysis)
        self.analysis_check.pack(anchor='w', pady=(7, 3))
        self.method_var = tk.StringVar(self.root, value='評価値ベース（探索）')
        self.method_combo = ttk.Combobox(side, textvariable=self.method_var, values=tuple(METHODS), state='readonly')
        self.method_combo.pack(fill='x')
        self.method_combo.bind('<<ComboboxSelected>>', self.change_analysis)
        level_row = ttk.Frame(side)
        level_row.pack(fill='x', pady=3)
        self.level_var = tk.StringVar(self.root)
        self.level_combo = ttk.Combobox(level_row, textvariable=self.level_var, state='readonly')
        self.level_combo.pack(side='left', fill='x', expand=True)
        self.level_combo.bind('<<ComboboxSelected>>', self.change_analysis)
        self.level_button = ttk.Button(level_row, text='レベル別 AI 設定', command=self.configure_levels)
        self.level_button.pack(side='right')
        self.advice_table = ttk.Treeview(side, columns=('move', 'score', 'loss', 'difficulty'), show='headings', height=4)
        for key, title, width in [('move', '指し手', 120), ('score', '実戦勝率', 85),
                                  ('loss', '最善手との差', 100), ('difficulty', '次の難しさ', 100)]:
            self.advice_table.heading(key, text=title)
            self.advice_table.column(key, width=width, minwidth=55)
        self.advice_table.pack(fill='x', pady=4)
        ttk.Label(side, text='人間代理モデルによる推定であり、人間での検証はしていない。\n'
                  'レベルはプログラム間の Elo で、段級位ではありません。', wraplength=420).pack(fill='x')
        nav = ttk.Frame(side)
        nav.pack(fill='x', pady=(12, 4))
        for text, target in [('先頭', lambda: 0), ('１手戻る', lambda: max(0, self.game.ply-1)),
                             ('１手進む', lambda: min(self.history.length, self.game.ply+1)),
                             ('末尾', lambda: self.history.length)]:
            ttk.Button(nav, text=text, command=lambda get=target: self.seek(get())).pack(side='left')
        self.log = tk.Listbox(side, font=(self.jp_font, 11), exportselection=False, height=8)
        self.log.pack(fill='both', expand=True)
        self.log.bind('<<ListboxSelect>>', self.on_history)
        record_buttons = ttk.Frame(side)
        record_buttons.pack(fill='x', pady=6)
        ttk.Button(record_buttons, text='KIF / CSA を開く', command=self.load).pack(side='left', expand=True, fill='x')
        ttk.Button(record_buttons, text='KIF / CSA を保存', command=self.save).pack(side='left', expand=True, fill='x')

    @staticmethod
    def _probe_optional(snapshot=None, cancel=None):
        try:
            from hasami_learn.practical import create_advisor
            return callable(create_advisor), 'モデルを選択または読み込んでください。'
        except Exception as exc:
            return False, '学習機能は無効です（torch / hasami_learn）。.venv の Python を使用してください。原因: ' + str(exc)

    def _enable_learned(self, available):
        self._available = available
        for widget in (self.model_button, self.analysis_check, self.level_button):
            widget.configure(state='normal' if available else 'disabled')
        for widget in (self.model_combo, self.method_combo, self.level_combo):
            widget.configure(state='readonly' if available else 'disabled')
        self._refresh_choices()

    def _refresh_choices(self):
        choices = [HUMAN, *CLASSICAL]
        if self._available:
            choices += list(self.models) + [level_text(level) for level in self._levels]
        for combo in self.player_combos:
            combo.configure(values=choices)
        self.model_combo.configure(values=list(self.models))
        if self.models and not self.model_var.get():
            self.model_var.set(next(iter(self.models)))
        methods = ('search', 'learned') if self._injected_advisor else ('search',)
        path = self.models.get(self.model_var.get())
        if path:
            try:
                methods = advisor_methods(path)
            except (OSError, ValueError, TypeError, AttributeError):
                pass
        names = [name for name, method in METHODS.items() if method in methods]
        self.method_combo.configure(values=names)
        if self.method_var.get() not in names:
            self.method_var.set(names[0])

    def _set_levels(self, levels):
        self._levels = list(levels)
        self._level_model = self.model_var.get()
        names = [level_text(level) for level in self._levels]
        self.level_combo.configure(values=names)
        if names and self.level_var.get() not in names:
            self.level_var.set(names[0])
        self._refresh_choices()

    def _invalidate(self):
        self._generation += 1
        if self._cancel:
            self._cancel.set()
        self.input = InputState()
        self._hints, self._best = [], None
        self._close_promotion()
        self.advice_table.delete(*self.advice_table.get_children())

    def _refresh_labels(self):
        self._labels = self.history.labels()
        self.log.delete(0, 'end')
        self.log.insert('end', '開始局面')
        for index, label in enumerate(self._labels, 1):
            self.log.insert('end', f'{index:03d}  {label}')
        if self.history.record.outcome.reason == 'resign':
            self.log.insert('end', '投了')

    def render(self):
        if self._closed:
            return
        turn = '先手' if self.game.position.turn == 0 else '後手'
        status = outcome_text(self.game.outcome) if self.game.outcome.terminal else turn + 'の手番'
        self.status.configure(text=f'{status}  ·  {self.game.ply} 手目 / {self.history.length} 手'
                              + ('  王手' if self.game.in_check() else ''))
        self.declare_button.configure(state='normal' if self._human_turn() and self.game.can_declare() else 'disabled')
        self.pause_button.configure(text='AI を再開' if self.paused else 'AI を停止')
        self.log.selection_clear(0, 'end')
        self.log.selection_set(self.game.ply)
        self.log.see(self.game.ply)
        self._draw_hands()
        self.draw_board()

    def _draw_hands(self):
        for frame in (self.top_hand, self.bottom_hand):
            for child in frame.winfo_children():
                child.destroy()
        self.hand_buttons.clear()
        for side in (0, 1):
            frame = self.top_hand if side != int(self.flipped) else self.bottom_hand
            align = 'left' if frame is self.top_hand else 'right'
            holder = ttk.Frame(frame)
            holder.pack(side=align)
            ttk.Label(holder, text=('先手' if side == 0 else '後手') + ' 持駒').pack(side='left')
            for hand, count in enumerate(self.game.position.hands[side]):
                button = ttk.Button(holder, text=f'{piece_name(hand+1)}×{count}', width=5,
                                    command=lambda p=side, h=hand: self.on_hand(p, h))
                button.pack(side='left')
                button.configure(state='normal' if count and side == self.game.position.turn and self._human_turn() else 'disabled')
                self.hand_buttons[side, hand] = button

    def center(self, sq):
        row, col = square_to_display(sq, self.flipped)
        return self.origin_x + (col+.5)*self.cell, self.origin_y + (row+.5)*self.cell

    def draw_board(self):
        if self._closed:
            return
        canvas = self.canvas
        width, height = max(200, canvas.winfo_width()), max(200, canvas.winfo_height())
        self.cell = min((width-65)/9, (height-65)/9)
        size = self.cell*9
        self.origin_x, self.origin_y = (width-size)/2-5, (height-size)/2+5
        ox, oy, cell = self.origin_x, self.origin_y, self.cell
        canvas.delete('all')
        canvas.create_rectangle(ox, oy, ox+size, oy+size, fill=WOOD, outline=INK)
        squares = []
        if self.game.moves:
            last = decode_move(self.game.moves[-1])
            squares += [(sq, '#f8e9b8') for sq in (last.from_sq, last.to_sq) if sq >= 0]
        if self.input.square is not None:
            squares.append((self.input.square, '#91c9b5'))
        if self.game.in_check():
            squares.append((self.game.position.board.index(8 + self.game.position.turn*16), '#ed9b8a'))
        for sq, color in squares:
            x, y = self.center(sq)
            canvas.create_rectangle(x-cell/2, y-cell/2, x+cell/2, y+cell/2, fill=color, outline='')
        for index in range(10):
            canvas.create_line(ox+index*cell, oy, ox+index*cell, oy+size, fill='#8c7456')
            canvas.create_line(ox, oy+index*cell, ox+size, oy+index*cell, fill='#8c7456')
        for index in range(9):
            canvas.create_text(ox+(index+.5)*cell, oy-19,
                               text=('１２３４５６７８９' if self.flipped else '９８７６５４３２１')[index], font=(self.jp_font, 11))
            canvas.create_text(ox+size+20, oy+(index+.5)*cell,
                               text=('九八七六五四三二一' if self.flipped else '一二三四五六七八九')[index], font=(self.jp_font, 11))
        for sq, piece in enumerate(self.game.position.board):
            if not piece:
                continue
            x, y = self.center(sq)
            sign = 1 if piece >> 4 == int(self.flipped) else -1
            w, h = cell*.34, cell*.40
            canvas.create_polygon(x, y-sign*h, x+w*.8, y-sign*h*.6, x+w, y+sign*h,
                                  x-w, y+sign*h, x-w*.8, y-sign*h*.6, fill='#fff0cf', outline='#9a7347')
            canvas.create_text(x, y+sign*cell*.04, text=piece_name(piece), angle=0 if sign == 1 else 180,
                               fill='#aa342e' if (piece & 15) > 8 else INK,
                               font=(self.jp_font, max(12, int(cell*(.29 if len(piece_name(piece)) > 1 else .40))), 'bold'))
        for sq in highlights(self.input, self.game):
            x, y = self.center(sq)
            r = cell*.09
            canvas.create_oval(x-r, y-r, x+r, y+r, fill='#25776a', outline='')
        for index, item in enumerate(self._hints):
            move = decode_move(item.move)
            if move.to_sq < 0:
                continue
            x, y = self.center(move.to_sq)
            hand_y = oy+size+20 if self.game.position.turn == int(self.flipped) else oy-20
            start = self.center(move.from_sq) if move.from_sq >= 0 else (x, hand_y)
            canvas.create_line(*start, x, y, arrow='last', fill=('#147c78', '#5865bb', '#b86b27')[index], width=3)

    def _human_turn(self):
        return not self.game.outcome.terminal and self.player_vars[self.game.position.turn].get() == HUMAN

    def on_click(self, event):
        if not self._human_turn():
            return
        row, col = int((event.y-self.origin_y)//self.cell), int((event.x-self.origin_x)//self.cell)
        sq = display_to_square(row, col, self.flipped)
        self.input, action, _ = click_square(self.input, self.game, sq)
        if action is not None:
            self.apply(action)
        elif self.input.mode == 'promotion':
            self._ask_promotion()
        self.draw_board()

    def on_hand(self, side, hand):
        if self._human_turn():
            self.input, _, _ = click_hand(self.input, self.game, side, hand)
            self.draw_board()

    def _ask_promotion(self):
        if self._promotion:
            return
        dialog = tk.Toplevel(self.root)
        self._promotion = dialog
        dialog.title('成りますか？')
        dialog.transient(self.root)
        ttk.Label(dialog, text='成りますか？', padding=15).pack()
        self.promotion_buttons = []
        for label, choice in [('成る', True), ('成らない', False), ('取消', None)]:
            button = ttk.Button(dialog, text=label, command=lambda p=choice: self.finish_promotion(p))
            button.pack(side='left', padx=8, pady=12)
            self.promotion_buttons.append(button)
        dialog.protocol('WM_DELETE_WINDOW', lambda: self.finish_promotion(None))
        dialog.grab_set()

    def finish_promotion(self, promote):
        self.input, action, _ = choose_promotion(self.input, promote)
        self._close_promotion()
        if action is not None:
            self.apply(action)
        self.render()

    def _close_promotion(self):
        if self._promotion:
            self._promotion.destroy()
            self._promotion = None
        self.promotion_buttons.clear()

    def apply(self, code):
        if type(code) is not int or not self.game.is_legal(code):
            raise ValueError('不正な着手を拒否しました。')
        branch = self.history.browsing
        if branch and not messagebox.askyesno('手順の分岐', '以後の手順を破棄して指しますか？', parent=self.root):
            self.input = InputState()
            return
        self._invalidate()
        self.history.play(code, branch=branch)
        self._refresh_labels()
        self.render()

    def replace_game(self, game):
        self._invalidate()
        self.history.close()
        self.history = History(game)
        self._agents.clear()
        self._refresh_labels()
        self.render()

    def new_game(self):
        self.replace_game(create_game(self.backend))
        self.paused = False
        self.render()

    def new_sfen(self):
        sfen = simpledialog.askstring('新規対局', 'SFEN を入力してください。', parent=self.root)
        if sfen is not None:
            self._open_sfen(sfen)

    def _open_sfen(self, sfen):
        try:
            game = create_game(self.backend, sfen=sfen)
        except Exception as exc:
            messagebox.showerror('SFEN エラー', str(exc), parent=self.root)
            return
        self.paused = True
        self.replace_game(game)

    def copy_sfen(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.game.sfen())

    def paste_sfen(self):
        try:
            text = self.root.clipboard_get()
        except tk.TclError as exc:
            messagebox.showerror('SFEN エラー', str(exc), parent=self.root)
            return
        self._open_sfen(text)

    def undo(self):
        self._invalidate()
        self.paused = True
        self.history.undo(tuple(var.get() == HUMAN for var in self.player_vars))
        self._refresh_labels()
        self.render()

    def seek(self, ply):
        self._invalidate()
        self.paused = True
        self.history.seek(ply)
        self.render()

    def on_history(self, event):
        selected = self.log.curselection()
        if selected:
            ply = min(selected[0], self.history.length)
            if ply != self.game.ply:
                self.seek(ply)

    def resign(self):
        if self._human_turn() and not self.history.browsing:
            if messagebox.askyesno('投了', '投了しますか？', parent=self.root):
                self._invalidate()
                self.history.resign()
                self._refresh_labels()
                self.render()

    def declare(self):
        if self._human_turn():
            _, code, _ = declaration(self.game)
            if code is not None:
                self.apply(code)

    def flip(self):
        self.flipped = not self.flipped
        self.render()

    def change_players(self, event=None):
        self._invalidate()
        self._agents.clear()
        self.render()

    def toggle_pause(self):
        self._invalidate()
        self.paused = not self.paused
        self.render()

    def change_analysis(self, event=None):
        self._invalidate()
        self._agents.clear()
        model = self.model_var.get()
        if self._level_model != model and self._injected_advisor is None:
            self._levels = []
            self.level_var.set('')
            self.level_combo.configure(values=())
            for var in self.player_vars:
                if var.get().startswith('レベル '):
                    var.set(HUMAN)
        self._level_model = model
        self._refresh_choices()
        self.render()

    def save(self):
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension='.kifu',
                 filetypes=[('KIF UTF-8', '*.kifu'), ('KIF', '*.kif'), ('CSA', '*.csa')])
        if path:
            try:
                save_record(self.history.record, path)
            except Exception as exc:
                messagebox.showerror('保存エラー', str(exc), parent=self.root)

    def load(self):
        path = filedialog.askopenfilename(parent=self.root,
                 filetypes=[('将棋棋譜', '*.kif *.kifu *.csa *.json')])
        if not path:
            return
        loaded = replacement = None
        try:
            loaded = load_record(path)
            replacement = create_game(self.backend, loaded.rules, loaded.initial.sfen())
            for code in loaded.moves:
                replacement.play(code)
            if loaded.outcome.reason == 'resign':
                replacement.resign()
            replacement.comments = loaded.comments.copy()
        except Exception as exc:
            if replacement:
                replacement.close()
            messagebox.showerror('読込エラー', str(exc), parent=self.root)
            return
        finally:
            if loaded:
                loaded.close()
        self.paused = True
        self.replace_game(replacement)

    def _limits(self):
        depth, time_ms, nodes = int(self.depth_var.get()), int(self.time_var.get()), int(self.nodes_var.get())
        if not 1 <= depth <= 6 or not 1 <= time_ms <= 10000:
            raise ValueError('深さは 1〜6、時間は 1〜10000 ms で指定してください。')
        return SearchLimits(depth, time_ms, nodes)

    def _launch(self, kind, operation):
        """Start an operation whose captures contain no App, widgets, or Tk variables."""
        if self._worker is not None or self._closed:
            return False
        token, cancel, snapshot = self._generation, threading.Event(), self.game.clone()
        self._cancel = cancel
        self.thinking.configure(text='読込中…' if kind in ('model', 'availability', 'levels') else '思考中…')
        # A target capturing self can finalize App -> root -> tkapp on this thread.
        self._worker = threading.Thread(target=run_worker,
            args=(self._results, token, kind, operation, snapshot, cancel),
            daemon=True, name='shogi-' + kind)
        self._worker.start()
        return True

    def _advisor_request(self, path, method):
        return partial(get_advisor, path, method, self._injected_advisor, self._advisors)

    def configure_levels(self):
        path = self.models.get(self.model_var.get())
        method = METHODS[self.method_var.get()]
        advisor = self._advisor_request(path, method)
        self._launch('levels', lambda snapshot, cancel: advisor().levels)

    def load_model(self):
        if self._worker is not None:
            return
        path = filedialog.askdirectory(parent=self.root, title='チェックポイントを選択')
        if not path:
            return
        rules = self.game.rules.to_dict()
        def operation(snapshot, cancel):
            agent = load_strong_agent(path, rules=rules)
            return path, agent
        self._launch('model', operation)

    def _request_ai(self):
        try:
            limits = self._limits()
            side = self.game.position.turn
            name = self.player_vars[side].get()
            path = self.models.get(name)
            advisor_path = self.models.get(self.model_var.get())
            method = METHODS[self.method_var.get()]
            level = next((level.level for level in self._levels if level_text(level) == name), None)
            cached = self._agents.get((side, name))
            advisor = self._advisor_request(advisor_path, method)
            def operation(snapshot, cancel):
                agent = cached
                if agent is None:
                    if name in CLASSICAL:
                        agent = {'ランダム': RandomAgent, '貪欲': GreedyAgent, 'αβ': AlphaBetaAgent}[name]()
                    elif path:
                        agent = load_strong_agent(path, rules=snapshot.rules.to_dict())
                    elif level is not None:
                        agent = MoveCodeAgent(advisor().proxy_agent(level))
                    else:
                        raise ValueError('対局者を選択してください。')
                decision = agent.choose(snapshot, limits, cancel)
                return side, name, agent, decision
            self._launch('ai', operation)
        except Exception as exc:
            self._error(exc)

    def _request_advice(self):
        path = self.models.get(self.model_var.get())
        method = METHODS[self.method_var.get()]
        level = next((item.level for item in self._levels if level_text(item) == self.level_var.get()), None)
        resolve_advisor = self._advisor_request(path, method)
        def operation(snapshot, cancel):
            advisor = resolve_advisor()
            chosen = level if level is not None else advisor.levels[0].level
            hints, best = analyze(advisor, snapshot, chosen, cancel)
            return advisor.levels, hints, best
        self._analysis_generation = self._generation
        self._launch('advice', operation)

    def _error(self, error):
        self.paused = True
        self.analysis_var.set(False)
        self.thinking.configure(text='処理を停止しました。')
        if not isinstance(error, AgentCancelled):
            messagebox.showerror('エラー', str(error), parent=self.root)

    def _poll(self):
        if self._closed:
            return
        # A queued result can precede Thread.run releasing its arguments.
        if self._worker is not None and self._worker.is_alive():
            self._poll_id = self.root.after(30, self._poll)
            return
        try:
            token, kind, result, error = self._results.get_nowait()
        except queue.Empty:
            pass
        else:
            self._worker, self._cancel = None, None
            self.thinking.configure(text='')
            if kind == 'availability' and error is None:
                available, reason = result
                self._enable_learned(available)
                self.model_status.set(reason)
            elif token == self._generation or kind == 'model':
                try:
                    if error is not None:
                        cancelled, message = error
                        self._error(AgentCancelled(message) if cancelled else RuntimeError(message))
                    elif kind == 'model':
                        path, agent = result
                        name, suffix = 'モデル：' + Path(path).name, 2
                        while name in self.models:
                            name = 'モデル：' + Path(path).name + f' ({suffix})'
                            suffix += 1
                        self.models[name] = path
                        self.model_var.set(name)
                        self.change_analysis()
                        self.model_status.set('読込済み：' + Path(path).name)
                        # Keep each seat's agent independent; load on its first turn.
                    elif kind == 'levels':
                        self._set_levels(result)
                    elif kind == 'advice':
                        levels, hints, best = result
                        self._set_levels(levels)
                        self._hints, self._best = hints, best
                        self.advice_table.delete(*self.advice_table.get_children())
                        for row in advice_rows(self.game, hints):
                            self.advice_table.insert('', 'end', values=row)
                        best_row = advice_rows(self.game, [best])[0]
                        self.advice_table.insert('', 'end', values=('最強：' + best_row[0], *best_row[1:]))
                    elif kind == 'ai':
                        side, name, agent, decision = result
                        if type(decision.action) is not int or not self.game.is_legal(decision.action):
                            raise ValueError('AI の不正な着手を拒否しました。')
                        self._agents[side, name] = agent
                        self.apply(decision.action)
                        self.thinking.configure(text=f'深さ {decision.depth} / {decision.nodes:,} ノード / {decision.elapsed_ms:.0f} ms')
                except Exception as exc:
                    self._error(exc)
            self.render()
        if self._worker is None and not self.game.outcome.terminal:
            if not self.paused and not self.history.browsing and not self._human_turn():
                self._request_ai()
            elif self.analysis_var.get() and self._analysis_generation != self._generation:
                self._request_advice()
        self._poll_id = self.root.after(30, self._poll)

    def help(self):
        messagebox.showinfo('操作説明', '自分の駒 → 合法な移動先をクリック。持駒は駒台から選びます。\n'
            '成・不成の両方が合法なときだけ選択画面が開きます。\n'
            '棋譜の先頭・前後・末尾へ移動し、過去から指すと確認後に分岐します。\n'
            '待ったは直前の人間の手まで戻します。検討の既定は評価値ベース（探索）です。\n'
            'モデルはローカルから読み込みます。ネットワーク取得や学習は行いません。', parent=self.root)

    def _shutdown(self):
        if self._closed:
            return
        self._closed = True
        self._generation += 1
        if self._cancel is not None:
            self._cancel.set()
        if self._poll_id is not None:
            self.root.after_cancel(self._poll_id)
            self._poll_id = None
        if self._worker is not None:
            self._worker.join(WORKER_JOIN_TIMEOUT)
            # Slow imports or agents may outlive the window, but own no Tk references.
            self._worker = None
        self._cancel = None
        self.history.close()

    def _on_destroy(self, event):
        if event.widget is self.root and not self._closed:
            self._shutdown()
            self._release_tk_references()
            # event.widget keeps the interpreter alive until collection completes.
            # Also cover callers that destroy the root directly instead of close().
            gc.collect()

    def _release_tk_references(self):
        """Detach the App's Tk references on the UI thread during teardown."""
        self.player_vars.clear()
        self.player_combos.clear()
        self.hand_buttons.clear()
        self.promotion_buttons.clear()
        for name, value in tuple(vars(self).items()):
            if isinstance(value, (tk.Misc, tk.Variable)):
                setattr(self, name, None)

    def close(self):
        if self._closed:
            return
        root = self.root
        self._shutdown()
        root.destroy()
        self._release_tk_references()
        # destroy() deletes Tcl commands, not all Python cycles. Dialog objects
        # (and mock dialog call histories in the smoke) can still retain root.
        # Collect while this main-thread frame pins the interpreter; a later
        # worker allocation must never be its last owner. Do not disable GC
        # process-wide: a slow worker may outlive the bounded join above.
        gc.collect()


def main(argv=None):
    parser = argparse.ArgumentParser(description='将棋 GUI')
    parser.add_argument('--backend', choices=('python', 'cpp'), default='python')
    args = parser.parse_args(argv)
    root = tk.Tk()
    App(root, args.backend)
    root.mainloop()


if __name__ == '__main__':
    main()
