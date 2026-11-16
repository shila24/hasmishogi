"""Tk-free input transitions and presentation; legality belongs to the engine."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

from hasami.agents import AgentCancelled

from . import create_game
from .model import DECLARE, decode_move
from .notation import move_to_japanese


def run_worker(results, token, kind, operation, snapshot, cancel):
    """Run Tk-free work and publish data, never exception tracebacks or GUI owners."""
    result, error = None, None
    try:
        try:
            result = operation(snapshot, cancel)
        finally:
            snapshot.close()
    except Exception as exc:
        # Tracebacks retain worker frames, including operation closures and their owners.
        error = (isinstance(exc, AgentCancelled), str(exc))
    results.put((token, kind, result, error))


def get_advisor(path, method, injected, cache):
    """Resolve an advisor using only Tk-free state owned by the serial worker."""
    if injected is not None:
        return injected
    if not path:
        raise ValueError('検討するモデルを選択してください。')
    key = path, method
    if key not in cache:
        cache[key] = load_advisor(path, method=method)
    return cache[key]


@dataclass(frozen=True)
class InputState:
    mode: str = 'idle'
    square: int | None = None
    hand: int | None = None
    choices: tuple[int, ...] = ()


def _moves(state, game):
    result = []
    for code in game.legal_actions():
        move = decode_move(code)
        if ((state.mode == 'board' and move.kind == 'move' and move.from_sq == state.square)
                or (state.mode == 'hand' and move.kind == 'drop' and move.piece == state.hand + 1)):
            result.append(code)
    return result


def highlights(state, game):
    return tuple(sorted({decode_move(code).to_sq for code in _moves(state, game)}))


def click_square(state, game, sq):
    """Return a selection or engine move, without modifying the supplied game."""
    if game.outcome.terminal:
        return InputState(), None, ()
    if state.mode == 'promotion' or type(sq) is not int or not 0 <= sq < 81:
        return state, None, highlights(state, game)
    if state.mode == 'board' and state.square == sq:
        return InputState(), None, ()
    choices = tuple(code for code in _moves(state, game) if decode_move(code).to_sq == sq)
    if len(choices) == 1:
        return InputState(), choices[0], ()
    if len(choices) == 2:
        return InputState('promotion', state.square, choices=choices), None, (sq,)
    cell = game.position.board[sq]
    if cell and cell >> 4 == game.position.turn:
        state = InputState('board', sq)
    return state, None, highlights(state, game)


def click_hand(state, game, player, hand_index):
    if game.outcome.terminal:
        return InputState(), None, ()
    if (state.mode == 'promotion' or type(player) is not int or player != game.position.turn
            or type(hand_index) is not int or not 0 <= hand_index < 7
            or not game.position.hands[player][hand_index]):
        return state, None, highlights(state, game)
    if state.mode == 'hand' and state.hand == hand_index:
        return InputState(), None, ()
    state = InputState('hand', hand=hand_index)
    return state, None, highlights(state, game)


def choose_promotion(state, promote):
    if state.mode != 'promotion' or promote is None:
        return InputState(), None, ()
    if type(promote) is not bool:
        raise ValueError('Promotion must be boolean or None')
    return InputState(), next(code for code in state.choices if decode_move(code).promote == promote), ()


def declaration(game):
    return InputState(), DECLARE if game.is_legal(DECLARE) else None, ()


def square_to_display(sq, flipped=False):
    if type(sq) is not int or not 0 <= sq < 81:
        raise ValueError('Invalid square')
    return divmod(80 - sq if flipped else sq, 9)


def display_to_square(row, col, flipped=False):
    if not 0 <= row < 9 or not 0 <= col < 9:
        return None
    sq = row * 9 + col
    return 80 - sq if flipped else sq


def piece_name(cell):
    names = ('', '歩', '香', '桂', '銀', '金', '角', '飛', '玉', 'と', '成香', '成桂', '成銀', '馬', '竜')
    return '王' if cell == 24 else names[cell & 15]


def move_text(game, code):
    return move_to_japanese(game, code)


def outcome_text(outcome):
    reasons = {'ongoing': '対局中', 'checkmate': '詰み', 'no_legal_moves': '合法手なし',
               'declaration': '入玉宣言', 'perpetual_check': '連続王手の千日手',
               'repetition': '千日手', 'max_moves': '最大手数', 'resign': '投了'}
    if not outcome.terminal:
        return reasons[outcome.reason]
    result = '引き分け' if outcome.winner == -1 else ('先手' if outcome.winner == 0 else '後手') + 'の勝ち'
    return result + ' / ' + reasons[outcome.reason]


def percentage(value):
    return '未計測' if value is None else f'{value * 100:.1f}%'


def level_text(level):
    elo = '未計測' if level.elo is None else f'{level.elo:.0f}'
    return f'レベル {level.level}（Elo {elo}）'


def advice_rows(game, advice):
    return [(move_text(game, item.move), percentage(item.practical_score), f'{item.value_loss:.3f}',
             percentage(item.difficulty)) for item in advice]


def analyze(advisor, game, level, cancel=None):
    """Call only the public advisor protocol and validate absolute engine moves."""
    snapshot = game.clone()
    try:
        hints = advisor.advise(snapshot, level, top_k=3, cancel=cancel)
        best = advisor.engine_best(snapshot, cancel=cancel)
        if any(type(item.move) is not int or not game.is_legal(item.move) for item in [*hints, best]):
            raise ValueError('助言器の不正な着手を拒否しました。')
        return hints[:3], best
    finally:
        snapshot.close()


def discover_models(directory=None):
    """List installed shogi checkpoints from the local release index, without importing them."""
    root = Path(directory) if directory is not None else Path(__file__).resolve().parents[1] / 'models'
    path = root / 'index.json'
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or not isinstance(data.get('models'), list):
        raise ValueError('models/index.json の形式が不正です。')
    result = {}
    for row in data['models']:
        if not isinstance(row, dict) or row.get('game') != 'shogi':
            continue
        name = row.get('name')
        if not isinstance(name, str) or not name or name in ('.', '..') or '/' in name or '\\' in name:
            continue
        checkpoint = (root / name).resolve()
        if checkpoint.parent != root.resolve():
            continue
        files = ('weights.hlnnw1', 'weights.hlnnw1.json') if row.get('asset_kind') == 'nnue' else ('manifest.json', 'weights.pt')
        if all((checkpoint / file).is_file() for file in files):
            result[name] = str(checkpoint)
    return result


class MoveCodeAgent:
    """Translate optional inference/proxy policy labels into absolute engine moves."""
    def __init__(self, agent):
        self.agent = agent

    def choose(self, game, limits, cancel=None):
        from .encoding import label_to_move

        snapshot = game.clone()
        try:
            decision = self.agent.choose(snapshot, limits, cancel)
        finally:
            snapshot.close()
        if type(decision.action) is not int:
            raise ValueError('Invalid policy label')
        return replace(decision, action=label_to_move(game, decision.action))


def load_strong_agent(checkpoint, **options):
    """Import optional inference only on an explicit model request."""
    path = Path(checkpoint)
    weights = path / 'weights.hlnnw1' if path.is_dir() else path
    if path.suffix == '.hlnnw1' or (path.is_dir() and weights.is_file()):
        from .native_search import AlphaBetaCppAgent
        return AlphaBetaCppAgent(nnue=str(weights))
    from hasami_learn.inference import create_agent

    options = dict(options)
    options.update(checkpoint=checkpoint, game_id='shogi')
    options.setdefault('device', 'auto')
    # Explicit MCTS selects the strongest policy/value mode even for practical checkpoints.
    if 'mode' not in options:
        manifest = json.loads((Path(checkpoint) / 'manifest.json').read_text(encoding='utf-8'))
        options['mode'] = 'q' if manifest.get('network', {}).get('head') == 'q' else 'mcts'
    options.setdefault('simulations', 800)
    return MoveCodeAgent(create_agent(**options))


def advisor_methods(checkpoint):
    data = json.loads((Path(checkpoint) / 'manifest.json').read_text(encoding='utf-8'))
    if not isinstance(data, dict) or data.get('game_id') != 'shogi':
        raise ValueError('将棋用のチェックポイントが必要です。')
    return ('search', 'learned') if (data.get('inference_hint') or {}).get('mode') == 'practical' else ('search',)


def load_advisor(checkpoint, method='search'):
    methods = advisor_methods(checkpoint)
    if method != 'auto' and method not in methods:
        raise ValueError('このモデルでは指定された助言方式を使用できません。')
    from hasami_learn.practical import create_advisor

    return create_advisor(checkpoint=checkpoint, method=method)


class History:
    """Keep complete engine snapshots for record navigation, never separate rule state."""
    def __init__(self, game):
        self.game = game
        self.record = game.clone()

    @property
    def length(self):
        return self.record.ply

    @property
    def browsing(self):
        return self.game.ply < self.record.ply or self.game.outcome != self.record.outcome

    def _commit(self):
        self.record.close()
        self.record = self.game.clone()

    def seek(self, ply):
        if type(ply) is not int or not 0 <= ply <= self.length:
            raise ValueError('Invalid history position')
        view = self.record.clone()
        if ply < view.ply and view.outcome.reason == 'resign':
            view.undo()
        while view.ply > ply:
            view.undo()
        self.game.close()
        self.game = view

    def play(self, code, *, branch=False):
        if self.browsing and not branch:
            raise ValueError('Branch confirmation required')
        self.game.play(code)
        self._commit()

    def resign(self):
        if self.browsing:
            raise ValueError('Cannot resign while browsing')
        self.game.resign()
        self._commit()

    def undo(self, humans):
        if self.game.outcome.reason == 'resign':
            self.game.undo()
        elif self.game.undo():
            while self.game.ply and any(humans) and not humans[self.game.position.turn]:
                self.game.undo()
        self._commit()

    def labels(self):
        replay = create_game(getattr(self.game, 'backend', 'python'), self.record.rules, self.record.initial.sfen())
        try:
            labels = []
            for code in self.record.moves:
                labels.append(move_text(replay, code))
                replay.play(code)
            return labels
        finally:
            replay.close()

    def close(self):
        self.game.close()
        self.record.close()
