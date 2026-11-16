"""Line-flushed USI adapter with cancellable snapshot workers and deadline replies."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import threading
import time

from hasami.agents import AgentCancelled, Decision, SearchLimits
from . import create_game
from .agents import AlphaBetaAgent
from .gui_support import MoveCodeAgent, discover_models, load_advisor, load_strong_agent, percentage
from .model import DECLARE
from .notation import move_to_usi, usi_to_move


def value_to_cp(value):
    return round(600 * math.atanh(max(-.999, min(.999, value))))


def go_limits(tokens, turn, depth, nodes):
    """Allocate a move budget; None is an infinite-search protocol deadline."""
    values, infinite = {}, False
    index = 0
    fields = {'btime', 'wtime', 'byoyomi', 'binc', 'winc', 'movetime', 'nodes', 'depth'}
    while index < len(tokens):
        name = tokens[index]
        if name == 'infinite':
            infinite = True
            index += 1
            continue
        if name not in fields or index + 1 == len(tokens):
            raise ValueError('Unsupported go parameter: ' + name)
        value = int(tokens[index + 1])
        if value < 0 or (name in ('nodes', 'depth', 'movetime') and value == 0):
            raise ValueError('Invalid go budget')
        values[name] = value
        index += 2
    if 'movetime' in values:
        budget = values['movetime']
    elif infinite:
        budget = 3600000
    elif values.get('byoyomi', 0):
        budget = max(1, values['byoyomi'] - 200)
    else:
        remaining = values.get('wtime' if turn else 'btime', 30000)
        increment = values.get('winc' if turn else 'binc', 0)
        budget = max(1, min(remaining + increment, remaining // 30 + increment))
    budget = min(budget, 3600000)
    limits = SearchLimits(values.get('depth', 64 if infinite else depth), budget, values.get('nodes', nodes))
    return limits, None if infinite and 'movetime' not in values else budget


@dataclass
class _Request:
    token: int
    cancel: threading.Event
    best: int | None
    started: float
    timer: threading.Timer | None = None


class USIEngine:
    def __init__(self, *, output=None, backend='python', agents_config=None):
        self.output = output if output is not None else sys.stdout
        self.backend = backend
        self.game = create_game(backend)
        self.options = {'Model': '', 'Mode': 'strong', 'Method': 'search', 'Level': 1,
                        'MaxNodes': 200000, 'Depth': 3, 'Engine': 'python', 'EvalFile': ''}
        self.lock = threading.RLock()
        self.request = None
        self.worker = None
        self.generation = 0
        self.configured = {}
        self._agent_cache = {}
        self._advisor_cache = {}
        if agents_config:
            data = json.loads(Path(agents_config).read_text(encoding='utf-8'))
            if not isinstance(data, dict) or not isinstance(data.get('agents'), list):
                raise ValueError('Invalid agents-config')
            for entry in data['agents']:
                if (entry.get('type') != 'python'
                        or entry.get('factory') != 'hasami_learn.inference:create_agent'):
                    raise ValueError('USI supports only hasami_learn.inference:create_agent')
                options = dict(entry.get('options', {}))
                checkpoint = options.pop('checkpoint')
                self.configured[entry['id']] = (checkpoint, options)
            if self.configured:
                self.options['Model'] = next(iter(self.configured))

    def emit(self, line):
        with self.lock:
            self.output.write(str(line).replace('\n', ' ').replace('\r', ' ') + '\n')
            self.output.flush()

    def _finish(self, request, *, error=None):
        with self.lock:
            if request is None or self.request is not request:
                return
            self.request = None
            request.cancel.set()
            if request.timer:
                request.timer.cancel()
            if error:
                self.emit('info string error ' + str(error))
                request.best = None
            self.emit('bestmove ' + ('resign' if request.best is None else move_to_usi(request.best)))

    def _invalidate(self):
        with self.lock:
            self.generation += 1
            request, self.request = self.request, None
            if request:
                request.cancel.set()
                if request.timer:
                    request.timer.cancel()

    def _checkpoint(self, model):
        if model in self.configured:
            return self.configured[model]
        return discover_models().get(model, model), {}

    def _advisor(self, checkpoint, method):
        key = checkpoint, method
        if key not in self._advisor_cache:
            self._advisor_cache[key] = load_advisor(checkpoint, method=method)
        return self._advisor_cache[key]

    @staticmethod
    def _validate_evaluation(options):
        if options['EvalFile'] and options['Engine'] != 'alphabeta-cpp':
            raise ValueError('EvalFile requires Engine alphabeta-cpp')
        if options['Engine'] == 'alphabeta-cpp' and (options['Model'] or options['Mode'] != 'strong'):
            raise ValueError('Native search requires strong mode and an empty Model option')

    def _agent(self, options):
        self._validate_evaluation(options)
        if not options['Model']:
            if options['Mode'] != 'strong':
                raise ValueError('Model is required for practical/level mode')
            if options['Engine'] == 'alphabeta-cpp':
                from .native_search import AlphaBetaCppAgent
                evaluation = discover_models().get(options['EvalFile'], options['EvalFile'])
                if evaluation and Path(evaluation).is_dir():
                    evaluation = str(Path(evaluation) / 'weights.hlnnw1')
                key = ('native', evaluation)
                if key not in self._agent_cache:
                    agent = AlphaBetaCppAgent(nnue=evaluation or None)
                    # Active workers retain their own reference until they finish.
                    for old in list(self._agent_cache):
                        if old[0] == 'native':
                            del self._agent_cache[old]
                    self._agent_cache[key] = agent
                return self._agent_cache[key]
            return AlphaBetaAgent()
        key = tuple(options[name] for name in ('Model', 'Mode', 'Method', 'Level'))
        if key not in self._agent_cache:
            checkpoint, settings = self._checkpoint(options['Model'])
            if Path(checkpoint).suffix == '.hlnnw1' or (Path(checkpoint) / 'weights.hlnnw1').is_file():
                raise ValueError('Select NNUE with EvalFile and Engine alphabeta-cpp')
            if options['Mode'] == 'level':
                agent = MoveCodeAgent(self._advisor(checkpoint, options['Method']).proxy_agent(options['Level']))
            else:
                agent = load_strong_agent(checkpoint, **settings)
            self._agent_cache[key] = agent
        return self._agent_cache[key]

    def _publish(self, request, snapshot, decision, *, learned=False):
        if type(decision.action) is not int or not snapshot.is_legal(decision.action):
            raise ValueError('Illegal agent move rejected')
        with self.lock:
            if self.request is not request:
                return
            request.best = decision.action
            score = value_to_cp(decision.score / 1000) if learned else decision.score
            elapsed = int((time.monotonic() - request.started) * 1000)
            mate = decision.extra.get('mate') if not learned else None
            score_text = f'mate {mate}' if mate is not None else f'cp {score}'
            self.emit(f'info depth {decision.depth} nodes {decision.nodes} time {elapsed} '
                      f'score {score_text} pv {move_to_usi(decision.action)}')

    def go(self, tokens):
        self._invalidate()
        if 'ponder' in tokens:
            self.emit('info string ponder is disabled')
            return
        options = self.options.copy()
        limits, budget = go_limits(tokens, self.game.position.turn, options['Depth'], options['MaxNodes'])
        legal = self.game.legal_actions()
        if not legal:
            self.emit('bestmove resign')
            return
        if DECLARE in legal:
            self.emit('bestmove win')
            return
        previous_worker = self.worker
        snapshot = self.game.clone()
        request = _Request(self.generation, threading.Event(), legal[0], time.monotonic())
        with self.lock:
            self.request = request
        if budget is not None:
            request.timer = threading.Timer(budget / 1000, self._finish, args=(request,))
            request.timer.daemon = True
            request.timer.start()

        def work():
            try:
                # Wait off the protocol thread, including the previous snapshot's cleanup.
                # The new request's timer runs during this wait and can return its legal fallback.
                if previous_worker is not None:
                    previous_worker.join()
                if request.cancel.is_set():
                    return
                self._validate_evaluation(options)
                if options['Mode'] == 'practical':
                    if not options['Model']:
                        raise ValueError('Model is required for practical mode')
                    checkpoint, _ = self._checkpoint(options['Model'])
                    advisor = self._advisor(checkpoint, options['Method'])
                    if request.cancel.is_set():
                        return
                    advice = advisor.advise(snapshot, options['Level'], top_k=1, cancel=request.cancel)
                    if not advice:
                        raise ValueError('Advisor returned no moves')
                    item = advice[0]
                    self._publish(request, snapshot, Decision(item.move, round(item.engine_value * 1000)), learned=True)
                    with self.lock:
                        if self.request is request:
                            self.emit('info string practical_score ' + percentage(item.practical_score)
                                      + ' difficulty ' + percentage(item.difficulty)
                                      + ' proxy estimate; not validated with humans')
                else:
                    agent = self._agent(options)
                    if isinstance(agent, AlphaBetaAgent):
                        used_nodes = 0
                        best_decision = None
                        for depth in range(1, limits.max_depth + 1):
                            remaining = limits.time_ms - int((time.monotonic() - request.started) * 1000)
                            if request.cancel.is_set() or remaining <= 0 or used_nodes >= limits.node_limit:
                                break
                            decision = agent.choose(snapshot, SearchLimits(depth, remaining, limits.node_limit-used_nodes), request.cancel)
                            used_nodes += decision.nodes
                            # choose() restarts deepening; a later partial call may finish less work.
                            if best_decision is None or decision.depth >= best_decision.depth:
                                best_decision = decision
                            self._publish(request, snapshot, Decision(best_decision.action, best_decision.score, best_decision.depth,
                                used_nodes, decision.elapsed_ms, decision.completed))
                            if not decision.completed:
                                break
                    else:
                        remaining = limits.time_ms - int((time.monotonic() - request.started) * 1000)
                        if request.cancel.is_set() or remaining <= 0:
                            self._finish(request)
                            return
                        search_limits = SearchLimits(limits.max_depth, remaining, limits.node_limit)
                        if options['Engine'] == 'alphabeta-cpp':
                            # Publish native iterations before a timer/stop retires the request.
                            # Cancellation may discard the final return, but keeps this best move.
                            decision = agent.choose(snapshot, search_limits, request.cancel,
                                on_iteration=lambda result: self._publish(request, snapshot, result))
                        else:
                            decision = agent.choose(snapshot, search_limits, request.cancel)
                        self._publish(request, snapshot, decision, learned=bool(options['Model']))
                if budget is not None:
                    self._finish(request)
            except AgentCancelled:
                self._finish(request)
            except Exception as exc:
                self._finish(request, error=exc)
            finally:
                snapshot.close()

        self.worker = threading.Thread(target=work, daemon=True, name='shogi-usi')
        self.worker.start()

    def position(self, tokens):
        self._invalidate()
        if not tokens:
            raise ValueError('Missing position')
        if tokens[0] == 'startpos':
            game, rest = create_game(self.backend), tokens[1:]
        elif tokens[0] == 'sfen' and len(tokens) >= 5:
            game, rest = create_game(self.backend, sfen=' '.join(tokens[1:5])), tokens[5:]
        else:
            raise ValueError('Invalid position')
        try:
            if rest:
                if rest[0] != 'moves':
                    raise ValueError('Expected moves')
                for text in rest[1:]:
                    game.play(usi_to_move(game, text))
        except Exception:
            game.close()
            raise
        self.game.close()
        self.game = game

    def setoption(self, tokens):
        if not tokens or tokens[0] != 'name':
            raise ValueError('Expected option name')
        split = tokens.index('value') if 'value' in tokens else len(tokens)
        name, value = ' '.join(tokens[1:split]), ' '.join(tokens[split+1:])
        if name not in self.options:
            self.emit('info string ignored option ' + name)
            return
        if name in ('Depth', 'Level', 'MaxNodes'):
            value = int(value)
            maximum = {'Depth': 64, 'Level': 1000000, 'MaxNodes': (1 << 64)-1}[name]
            if not 1 <= value <= maximum:
                raise ValueError('Invalid option value')
        elif name == 'Engine' and value not in ('python', 'alphabeta-cpp'):
            raise ValueError('Invalid Engine')
        elif name == 'Mode' and value not in ('strong', 'practical', 'level'):
            raise ValueError('Invalid Mode')
        elif name == 'Method' and value not in ('auto', 'search', 'learned'):
            raise ValueError('Invalid Method')
        if name in ('Model', 'EvalFile') and value == '<empty>':
            value = ''
        self._invalidate()
        self.options[name] = value
        if name == 'EvalFile':
            for key in list(self._agent_cache):
                if key[0] == 'native':
                    del self._agent_cache[key]

    def command(self, line):
        tokens = line.split()
        if not tokens:
            return True
        name, args = tokens[0], tokens[1:]
        try:
            if name == 'usi':
                for text in ('id name Hasami Lab Shogi', 'id author Hasami Lab',
                             'option name Model type string default <empty>',
                             'option name EvalFile type string default <empty>',
                             'option name Engine type combo default python var python var alphabeta-cpp',
                             'option name Mode type combo default strong var strong var practical var level',
                             'option name Method type combo default search var auto var search var learned',
                             'option name Level type spin default 1 min 1 max 1000000',
                             'option name MaxNodes type spin default 200000 min 1 max 2147483647',
                             'option name Depth type spin default 3 min 1 max 64', 'usiok'):
                    self.emit(text)
            elif name == 'isready':
                # USI readiness is the explicit initialization boundary, outside move clocks.
                try:
                    self._validate_evaluation(self.options)
                    if (self.options['Model'] or self.options['EvalFile'] or self.options['Engine'] == 'alphabeta-cpp') and (self.worker is None or not self.worker.is_alive()):
                        if self.options['Mode'] == 'practical':
                            checkpoint, _ = self._checkpoint(self.options['Model'])
                            self._advisor(checkpoint, self.options['Method'])
                        else:
                            self._agent(self.options)
                except Exception as exc:
                    self.emit('info string error ' + str(exc))
                self.emit('readyok')
            elif name == 'setoption':
                self.setoption(args)
            elif name == 'position':
                self.position(args)
            elif name == 'usinewgame':
                self.position(['startpos'])
            elif name == 'go':
                self.go(args)
            elif name == 'stop':
                if self.request:
                    self._finish(self.request)
            elif name == 'ponderhit':
                self.emit('info string ponder is disabled')
            elif name == 'gameover':
                self._invalidate()
            elif name == 'quit':
                self.close()
                return False
            else:
                self.emit('info string ignored command ' + name)
        except Exception as exc:
            self.emit('info string error ' + str(exc))
        return True

    def close(self):
        self._invalidate()
        self.game.close()
        self._agent_cache.clear()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Shogi USI engine')
    parser.add_argument('--backend', choices=('python', 'cpp'), default='python')
    parser.add_argument('--agents-config')
    args = parser.parse_args(argv)
    engine = USIEngine(backend=args.backend, agents_config=args.agents_config)
    try:
        for line in sys.stdin:
            if not engine.command(line):
                break
    finally:
        engine.close()
        # Native callbacks and handles must finish unwinding before interpreter shutdown.
        if engine.worker is not None:
            engine.worker.join()


if __name__ == '__main__':
    main()
