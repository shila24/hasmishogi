"""Playable Tk GUI. Only the main thread may access Tk objects."""
from __future__ import annotations
import argparse
from pathlib import Path
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, ttk
from . import create_game
from .agents import AgentRegistry, SearchLimits, AgentCancelled
from .model import bits, move_name, decode_action
from .records import record_dict, from_record, load_record, save_record
# Tk-free helpers live in gui_support so tests can exercise them without a display or _tkinter.
from .gui_support import compute_limits, unique_agent_id  # noqa: F401  (re-exported for callers)

BG = "#101B2D"
PANEL = "#1A2940"
TEXT = "#EDF2F7"
MUTED = "#9EB0C7"
ACCENT = "#50D4BE"
WOOD = "#E7CCA1"
INK = "#263243"


class App:
    def __init__(self, root: tk.Tk, backend: str, registry: AgentRegistry | None = None):
        self.root = root
        self.registry = registry or AgentRegistry()
        self.game = create_game(backend)
        self.selected = None
        self.hint_action = None
        self.paused = False
        self._closed = False
        self._poll_id = None
        self._generation = 0
        self._worker = None
        self._cancel = None
        self._results = queue.Queue()
        self._model_results = queue.Queue()
        self._model_worker = None
        self._assist_agent = self.registry.create("alphabeta")
        self._agents = {}
        self._agent_names = {}
        self.origin_x = self.origin_y = 32
        self.cell = 62
        self.root.title("Hasami Lab | はさみ将棋")
        self.root.geometry("1080x870")
        self.root.minsize(930, 790)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        families = set(tkfont.families(root))
        self.jp_font = next((x for x in ("Hiragino Sans", "Yu Gothic", "Noto Sans CJK JP",
                                          "Meiryo") if x in families), "TkDefaultFont")
        self.ui_font = (self.jp_font, 10)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TButton", font=self.ui_font, padding=(8, 7))
        style.configure("TCombobox", font=self.ui_font, padding=4)
        self._build_widgets(backend)
        self.render()
        self._poll_id = self.root.after(40, self._poll)

    def label(self, parent, text, size=10, color=TEXT, **kwargs):
        return tk.Label(parent, text=text, bg=parent["bg"], fg=color,
                        font=(self.jp_font, size), **kwargs)

    def _build_widgets(self, backend):
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=25, pady=(18, 8))
        self.label(header, "HASAMI  /  LAB", 22, ACCENT).pack(side="left")
        self.label(header, "はさみ将棋  ·  Game & Agent Workbench", 11, MUTED).pack(side="right")
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=20)
        board_area = tk.Frame(body, bg=BG)
        board_area.pack(side="left", fill="both", expand=True, padx=(0, 18))
        board_area.columnconfigure(0, weight=1)
        board_area.rowconfigure(2, weight=1)
        self.status = self.label(board_area, "", 15, anchor="w")
        self.status.grid(row=0, column=0, sticky="ew", pady=(8, 5))
        self.detail = self.label(board_area, "", 10, MUTED, anchor="w", wraplength=620)
        self.detail.grid(row=1, column=0, sticky="ew", pady=(0, 4))
        self.canvas = tk.Canvas(board_area, bg=BG, highlightthickness=0, width=650, height=635)
        self.canvas.grid(row=2, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda event: self.draw_board())
        self.canvas.bind("<Button-1>", self.on_click)
        self.footer = self.label(board_area, "駒を選択 → 移動先をクリック", 10, MUTED, anchor="w")
        self.footer.grid(row=3, column=0, sticky="ew", pady=(3, 14))
        side = tk.Frame(body, bg=PANEL, width=310)
        side.pack(side="right", fill="y", pady=(8, 15))
        side.pack_propagate(False)
        # The rules label is packed first with side="bottom" so it keeps its full height at the
        # minimum window size; the move log is the only widget allowed to shrink.
        self.rules_label = self.label(side, "", 9, MUTED, wraplength=272, justify="left", anchor="w")
        self.rules_label.pack(side="bottom", fill="x", padx=18, pady=(0, 14))
        self.label(side, "CONTROL ROOM", 12, ACCENT).pack(anchor="w", padx=18, pady=(15, 10))
        self.backend_var = tk.StringVar(value=backend)
        engine_row = tk.Frame(side, bg=PANEL)
        engine_row.pack(fill="x", padx=18)
        self.label(engine_row, "Engine", 10, MUTED).pack(side="left")
        self.backend_combo = ttk.Combobox(engine_row, textvariable=self.backend_var,
                                          values=("cpp", "python"), state="readonly", width=12)
        self.backend_combo.pack(side="right")
        self.backend_combo.bind("<<ComboboxSelected>>", self.switch_backend)
        self.engine_label = self.label(side, "", 9, MUTED, anchor="w")
        self.engine_label.pack(fill="x", padx=18, pady=(6, 10))
        self.player_vars = []
        self.player_combos = []
        choices = ["human"] + self.registry.names()
        for side_id, title in enumerate(("先手 / 歩", "後手 / と")):
            row = tk.Frame(side, bg=PANEL)
            row.pack(fill="x", padx=18, pady=4)
            self.label(row, title, 10).pack(side="left")
            var = tk.StringVar(value="human" if side_id == 0 else "alphabeta")
            self.player_vars.append(var)
            combo = ttk.Combobox(row, textvariable=var, values=choices, state="readonly", width=15)
            combo.pack(side="right")
            combo.bind("<<ComboboxSelected>>", self.change_players)
            self.player_combos.append(combo)
        limit_row = tk.Frame(side, bg=PANEL)
        limit_row.pack(fill="x", padx=18, pady=(9, 8))
        self.depth_var = tk.StringVar(value="3")
        self.time_var = tk.StringVar(value="700")
        self.sims_var = tk.StringVar(value="200")
        self.label(limit_row, "Depth", 9, MUTED).pack(side="left")
        ttk.Spinbox(limit_row, from_=1, to=12, width=2, textvariable=self.depth_var).pack(side="left", padx=2)
        self.label(limit_row, "ms", 9, MUTED).pack(side="left")
        ttk.Spinbox(limit_row, from_=10, to=10000, increment=100, width=5,
                    textvariable=self.time_var).pack(side="left", padx=2)
        self.label(limit_row, "Sims", 9, MUTED).pack(side="left")
        self.sims_spinbox = ttk.Spinbox(limit_row, from_=1, to=4000, width=4,
                                        textvariable=self.sims_var)
        self.sims_spinbox.pack(side="left", padx=2)
        self.score_label = self.label(side, "", 13, anchor="w")
        self.score_label.pack(fill="x", padx=18, pady=(7, 5))
        self.pending_label = self.label(side, "", 10, ACCENT, justify="left", wraplength=273, anchor="w")
        self.pending_label.pack(fill="x", padx=18, pady=(0, 8))
        button_grid = tk.Frame(side, bg=PANEL)
        button_grid.pack(fill="x", padx=15)
        for col in (0, 1):
            button_grid.columnconfigure(col, weight=1)
        buttons = [("新しい対局", self.new_game), ("待った（1手）", self.undo),
                   ("AI 1手", self.step_ai), ("ヒント", self.hint),
                   ("棋譜を保存", self.save), ("棋譜を開く", self.load)]
        for index, (label, command) in enumerate(buttons):
            ttk.Button(button_grid, text=label, command=command).grid(
                row=index // 2, column=index % 2, sticky="ew", padx=3, pady=3)
        self.model_button = ttk.Button(side, text="モデル読込", command=self.load_model)
        self.model_button.pack(fill="x", padx=18, pady=(4, 3))
        self.model_label = self.label(side, "", 9, MUTED, wraplength=273,
                                      anchor="w", justify="left")
        self.model_label.pack(fill="x", padx=18)
        self.pause_button = ttk.Button(side, text="AI を一時停止", command=self.toggle_pause)
        self.pause_button.pack(fill="x", padx=18, pady=(7, 7))
        self.thinking_label = self.label(side, "", 9, MUTED, wraplength=273, anchor="w", justify="left")
        self.thinking_label.pack(fill="x", padx=18, pady=(0, 7))
        self.label(side, "MOVE HISTORY", 10, ACCENT).pack(anchor="w", padx=18, pady=(2, 5))
        log_frame = tk.Frame(side, bg=PANEL)
        log_frame.pack(fill="both", expand=True, padx=18, pady=(0, 8))
        scrollbar = ttk.Scrollbar(log_frame)
        scrollbar.pack(side="right", fill="y")
        self.log = tk.Listbox(log_frame, bg=BG, fg=TEXT, borderwidth=0, highlightthickness=0,
                              selectbackground="#314B69", font=(self.jp_font, 10), height=3,
                              yscrollcommand=scrollbar.set)
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.log.yview)

    def _invalidate(self):
        self._generation += 1
        if self._cancel:
            self._cancel.set()
        self.hint_action = None
        self.selected = None

    def render(self):
        p, out = self.game.position, self.game.outcome
        names = ["先手（歩）", "後手（と）"]
        if out.terminal:
            result = "引き分け" if out.winner == -1 else names[out.winner] + "の勝ち"
            reasons = {"five_captures": "規定枚数を獲得", "lead_confirmed": "駒差を確定",
                       "repetition": "千日手", "no_moves": "合法手なし"}
            self.status.config(text=f"{result}  /  {reasons[out.reason]}")
        else:
            self.status.config(text=names[p.turn] + "の手番")
        self.detail.config(text=f"{len(self.game.moves)} 手目まで  ·  合法手 {len(self.game.legal_actions())}  ·  "
                           + ("AI 停止中" if self.paused else "AI 自動応手 ON"))
        self.engine_label.config(text=("C++20 native  /  2 × uint64 per side" if self.backend_var.get() == "cpp"
                                       else "Python reference  /  81-bit integers"))
        self.score_label.config(text=f"獲得   歩  {p.captures(0)}  :  {p.captures(1)}  と")
        if p.pending != -1:
            self.pending_label.config(text="駒差の応手待ち：この手で差を\nしきい値未満に戻す必要があります。")
        else:
            self.pending_label.config(text="歩と と金の動きは同じです。")
        self.pause_button.config(text="AI を再開" if self.paused else "AI を一時停止")
        r = self.game.rules
        self.rules_label.config(text=f"獲得 {r.capture_target or 'OFF'} 枚 / 駒差 {r.lead_target or 'OFF'} 枚 + 応手\n"
                                f"千日手 {r.repetition_limit or 'OFF'} 回 / 端の封鎖 {'ON' if r.edge_capture else 'OFF'}")
        self.log.delete(0, "end")
        first = self.game.initial.turn
        for index, action in enumerate(self.game.moves):
            side = "歩" if (first + index) % 2 == 0 else "と"
            self.log.insert("end", f"{index + 1:03d}   {side}   {move_name(action)}")
        self.log.yview_moveto(1.0)
        self.draw_board()

    def draw_board(self):
        if self._closed:
            return
        canvas = self.canvas
        width, height = max(200, canvas.winfo_width()), max(200, canvas.winfo_height())
        self.cell = min((width - 56) / 9, (height - 46) / 9)
        size = self.cell * 9
        self.origin_x = (width - size) / 2 + 9
        self.origin_y = (height - size) / 2 + 6
        ox, oy, cell = self.origin_x, self.origin_y, self.cell
        canvas.delete("all")
        canvas.create_rectangle(ox - 8, oy - 8, ox + size + 8, oy + size + 8,
                                fill="#C19A67", outline="")
        canvas.create_rectangle(ox, oy, ox + size, oy + size, fill=WOOD, outline=INK, width=2)
        if self.game.moves:
            for sq in decode_action(self.game.moves[-1]):
                r, c = divmod(sq, 9)
                canvas.create_rectangle(ox + c * cell + 1, oy + r * cell + 1,
                                        ox + (c + 1) * cell - 1, oy + (r + 1) * cell - 1,
                                        fill="#F6E5C6", outline="")
        if self.selected is not None:
            r, c = divmod(self.selected, 9)
            canvas.create_rectangle(ox + c * cell + 1, oy + r * cell + 1,
                                    ox + (c + 1) * cell - 1, oy + (r + 1) * cell - 1,
                                    fill="#8BCAB5", outline="")
        for index in range(10):
            canvas.create_line(ox + index * cell, oy, ox + index * cell, oy + size, fill="#8C7456")
            canvas.create_line(ox, oy + index * cell, ox + size, oy + index * cell, fill="#8C7456")
        for index in range(9):
            canvas.create_text(ox + (index + .5) * cell, oy - 21, text=chr(65 + index),
                               fill=MUTED, font=(self.jp_font, 10))
            canvas.create_text(ox - 22, oy + (index + .5) * cell, text=str(index + 1),
                               fill=MUTED, font=(self.jp_font, 10))
        for side in (0, 1):
            for sq in bits(self.game.position.pieces(side)):
                r, c = divmod(sq, 9)
                x, y = ox + (c + .5) * cell, oy + (r + .5) * cell
                sign = 1 if side == 0 else -1
                w, h = cell * .32, cell * .37
                points = [(x, y - sign * h), (x + w * .8, y - sign * h * .63),
                          (x + w, y + sign * h), (x - w, y + sign * h),
                          (x - w * .8, y - sign * h * .63)]
                canvas.create_polygon(points, fill="#FFF0CF", outline="#9A7347", width=1.5)
                canvas.create_text(x, y + sign * .06 * cell, text="歩" if side == 0 else "と",
                                   fill="#222A32" if side == 0 else "#AB4339",
                                   font=(self.jp_font, max(14, int(cell * .34)), "bold"))
        if self.selected is not None:
            for action in self.game.legal_actions():
                source, target = divmod(action, 81)
                if source == self.selected:
                    r, c = divmod(target, 9)
                    x, y = ox + (c + .5) * cell, oy + (r + .5) * cell
                    radius = cell * .08
                    canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                       fill="#27786C", outline="")
        if self.hint_action is not None:
            source, target = divmod(self.hint_action, 81)
            points = []
            for sq in (source, target):
                r, c = divmod(sq, 9)
                points.extend((ox + (c + .5) * cell, oy + (r + .5) * cell))
            canvas.create_line(*points, fill="#267A94", width=4, arrow="last",
                               arrowshape=(14, 16, 6))

    def on_click(self, event):
        if self.game.outcome.terminal or self._worker is not None:
            return
        if self.player_vars[self.game.position.turn].get() != "human":
            self.footer.config(text="この手番はエージェントです。human に変更すると操作できます。")
            return
        c = int((event.x - self.origin_x) // self.cell)
        r = int((event.y - self.origin_y) // self.cell)
        if not (0 <= r < 9 and 0 <= c < 9):
            return
        sq = r * 9 + c
        if self.game.position.pieces(self.game.position.turn) & (1 << sq):
            self.selected = sq
            self.draw_board()
        elif self.selected is not None:
            action = self.selected * 81 + sq
            if action in self.game.legal_actions():
                self.apply(action)
            else:
                self.footer.config(text="そこには動けません。緑の点から移動先を選んでください。")

    def apply(self, action):
        try:
            result = self.game.play(action)
        except Exception as exc:
            self.paused = True
            messagebox.showerror("着手エラー", str(exc), parent=self.root)
            return
        self._invalidate()
        self.footer.config(text=move_name(action) + f"  /  {result.captured.bit_count()} 枚獲得")
        if result.outcome.terminal:
            self.paused = True
        self.render()

    def new_game(self):
        self._invalidate()
        old = self.game
        self.game = create_game(self.backend_var.get(), old.rules)
        old.close()
        self.paused = False
        self._agents.clear()
        self._agent_names.clear()
        self.thinking_label.config(text="")
        self.footer.config(text="新しい対局を開始しました。")
        self.render()

    def undo(self):
        self._invalidate()
        self.paused = True
        self.game.undo()
        self.thinking_label.config(text="")
        self.footer.config(text="1手戻しました。AI は停止中です。")
        self.render()

    def toggle_pause(self):
        self._invalidate()
        self.paused = not self.paused
        self.render()

    def change_players(self, _event=None):
        self._invalidate()
        self.paused = True
        self._agents.clear()
        self._agent_names.clear()
        self.footer.config(text="エージェント変更後は「AI を再開」または「AI 1手」を押してください。")
        self.render()

    def switch_backend(self, _event=None):
        requested = self.backend_var.get()
        try:
            new_game = from_record(record_dict(self.game), requested)
        except Exception as exc:
            self.backend_var.set(self.game.backend)
            messagebox.showerror("バックエンドを変更できません", str(exc), parent=self.root)
            return
        self._invalidate()
        old = self.game
        self.game = new_game
        old.close()
        self.paused = True
        self.render()

    def save(self):
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension=".json",
                                           filetypes=[("Hasami record", "*.json")])
        if path:
            try:
                save_record(self.game, path)
                self.footer.config(text=f"保存しました：{Path(path).name}")
            except Exception as exc:
                messagebox.showerror("保存エラー", str(exc), parent=self.root)

    def load(self):
        path = filedialog.askopenfilename(parent=self.root, filetypes=[("Hasami record", "*.json")])
        if path:
            try:
                new_game = load_record(path, self.backend_var.get())
            except Exception as exc:
                messagebox.showerror("読み込みエラー", str(exc), parent=self.root)
                return
            self._invalidate()
            old = self.game
            self.game = new_game
            old.close()
            self.paused = True
            self._agents.clear()
            self._agent_names.clear()
            self.render()

    def load_model(self):
        if self._closed or self._model_worker is not None:
            return
        path = filedialog.askdirectory(parent=self.root, title="モデルのチェックポイントを選択")
        if not path:
            return
        try:
            simulations = compute_limits(1, 1, int(self.sims_var.get()), "model").node_limit
        except ValueError as exc:
            messagebox.showerror("モデル読込エラー", str(exc), parent=self.root)
            return
        results = self._model_results
        self.model_button.config(state="disabled")
        self.model_label.config(text="読込中…")

        def work():
            try:
                # The optional learning package is imported only on explicit loading.
                from hasami_learn.inference import create_agent, describe_agent

                agent = create_agent(checkpoint=path, device="auto", mode="auto",
                                     simulations=simulations)
                family = agent.manifest.algorithm
                # One short line: a multi-line description would push the bottom labels off screen
                # at the 930x790 minimum window size.
                description = describe_agent(agent, compact=True)
                first_agent = agent

                def factory():
                    nonlocal first_agent
                    if first_agent is not None:
                        loaded, first_agent = first_agent, None
                        return loaded
                    return create_agent(checkpoint=path, device="auto", mode="auto",
                                        simulations=simulations)

                results.put((path, factory, family, description, None))
            except Exception as exc:
                error = str(exc)
                if isinstance(exc, ImportError):
                    error += ("\n学習用の依存関係（torch / hasami_learn）を確認し、"
                              ".venv/bin/python で GUI を起動してください。")
                results.put((path, None, None, None, error))

        self._model_worker = threading.Thread(target=work, daemon=True, name="hasami-model")
        self._model_worker.start()

    def _poll_model(self):
        try:
            path, factory, family, description, error = self._model_results.get_nowait()
        except queue.Empty:
            return
        self._model_worker = None
        self.model_button.config(state="normal")
        if error is None:
            try:
                name = unique_agent_id(f"model:{Path(path).name}", self.registry.names())
                self.registry.register(name, factory, family=family, description=description)
            except Exception as exc:
                error = str(exc)
        if error is not None:
            self.model_label.config(text="モデルを読み込めませんでした。")
            messagebox.showerror("モデル読込エラー", error, parent=self.root)
            return
        choices = ["human"] + self.registry.names()
        for combo in self.player_combos:
            combo.config(values=choices)
        self.model_label.config(text=description)

    def _limits(self, name=None):
        if name is None:
            name = self.player_vars[self.game.position.turn].get()
        if name == "human":
            name = "alphabeta"
        return compute_limits(int(self.depth_var.get()), int(self.time_var.get()),
                              int(self.sims_var.get()), self.registry.family(name))

    def _request(self, hint=False):
        if self._worker is not None or self.game.outcome.terminal:
            return
        turn = self.game.position.turn
        name = "alphabeta" if hint else self.player_vars[turn].get()
        assist = hint or name == "human"
        if assist:
            name = "alphabeta"
        try:
            limits = self._limits(name)
            family = self.registry.family(name)
            if assist:
                agent = self._assist_agent
            else:
                if self._agent_names.get(turn) != name and family == "classical":
                    self._agents[turn] = self.registry.create(name)
                    self._agent_names[turn] = name
                agent = self._agents.get(turn) if self._agent_names.get(turn) == name else None
            snapshot = self.game.clone()
        except Exception as exc:
            self.paused = True
            messagebox.showerror("エージェント設定エラー", str(exc), parent=self.root)
            return
        token = self._generation
        cancel = threading.Event()
        self._cancel = cancel
        self.thinking_label.config(text=f"{name} が検討中…")

        def work():
            decision, error = None, None
            chosen = agent
            try:
                if chosen is None:
                    chosen = self.registry.create(name)
                decision = chosen.choose(snapshot, limits, cancel)
            except Exception as exc:
                error = exc
            finally:
                snapshot.close()
                self._results.put((token, hint, decision, error, turn, name, chosen, family))

        self._worker = threading.Thread(target=work, daemon=True, name="hasami-agent")
        self._worker.start()

    def step_ai(self):
        self.paused = True
        self._request(False)
        self.render()

    def hint(self):
        self.paused = True
        self._request(True)
        self.render()

    def _poll(self):
        if self._closed:
            return
        self._poll_model()
        try:
            token, hint, decision, error, turn, name, agent, family = self._results.get_nowait()
        except queue.Empty:
            pass
        else:
            self._worker = None
            self._cancel = None
            if token == self._generation:
                if error is not None:
                    self.paused = True
                    if not isinstance(error, AgentCancelled):
                        messagebox.showerror("エージェントエラー", str(error), parent=self.root)
                    self.thinking_label.config(text="エージェントを停止しました。")
                elif decision.action not in self.game.legal_actions():
                    self.paused = True
                    messagebox.showerror("エージェントエラー", "不正な着手を拒否しました。", parent=self.root)
                else:
                    if not hint and self.player_vars[turn].get() != "human":
                        self._agents[turn] = agent
                        self._agent_names[turn] = name
                    if family == "classical":
                        self.thinking_label.config(text=f"depth {decision.depth}  /  {decision.nodes:,} nodes\n"
                                                   f"{decision.elapsed_ms:.0f} ms  /  score {decision.score}")
                    else:
                        self.thinking_label.config(text=f"sims {decision.nodes} / {decision.elapsed_ms:.0f} ms / "
                                                   f"value {decision.score / 1000:+.2f}")
                    if hint:
                        self.hint_action = decision.action
                        self.footer.config(text=f"候補：{move_name(decision.action)}")
                        self.draw_board()
                    else:
                        self.apply(decision.action)
                self.render()
            else:
                self.thinking_label.config(text="以前の検討結果を破棄しました。")
        if (not self.paused and self._worker is None and not self.game.outcome.terminal
                and self.player_vars[self.game.position.turn].get() != "human"):
            self._request()
        self._poll_id = self.root.after(40, self._poll)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._poll_id is not None:
            self.root.after_cancel(self._poll_id)
            self._poll_id = None
        self._invalidate()
        self.game.close()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("auto", "cpp", "python"), default="auto")
    parser.add_argument("--agents-config")
    args = parser.parse_args()
    registry = AgentRegistry()
    if args.agents_config:
        registry.load_config(args.agents_config)
    backend = args.backend
    if backend == "auto":
        try:
            test = create_game("cpp")
            test.close()
            backend = "cpp"
        except (RuntimeError, ValueError, OSError) as exc:
            print(f"Native backend unavailable: {exc}\nUsing the Python backend.", file=sys.stderr)
            backend = "python"
    try:
        root = tk.Tk()
        App(root, backend, registry)
        root.mainloop()
    except (tk.TclError, RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(f"GUI startup failed: {exc}\nCheck Tk support, DISPLAY, and the native build.") from exc


if __name__ == "__main__":
    main()
