"""Exercise a real USI subprocess and check replies against the Python shogi engine."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def declared_options(root: Path) -> tuple[str, ...]:
    """Read literal declarations without importing or executing the engine source."""
    source = root / 'shogi/usi.py'
    tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
    declarations = sorted(
        (node for node in ast.walk(tree)
         if isinstance(node, ast.Constant) and isinstance(node.value, str)
         and node.value.startswith('option name ')),
        key=lambda node: (node.lineno, node.col_offset))
    options = tuple(node.value for node in declarations)
    if not options or len({option.split(' type ', 1)[0] for option in options}) != len(options):
        raise ValueError('Missing or duplicate literal USI option declarations')
    return options


OPTIONS = declared_options(ROOT)


class ProtocolError(RuntimeError):
    """The subprocess did not meet its protocol contract."""


class Session:
    def __init__(self, command, root: Path, timeout: float):
        self.timeout = timeout
        self.lines = queue.Queue()
        self.process = subprocess.Popen(command, cwd=root, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                        text=True, encoding='utf-8', bufsize=1)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                self.lines.put(line.rstrip('\r\n'))
        finally:
            self.lines.put(None)

    def send(self, line: str):
        try:
            self.process.stdin.write(line + '\n')
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise ProtocolError('Engine closed its input') from exc

    def until(self, prefix: str) -> list[str]:
        deadline = time.monotonic() + self.timeout
        received = []
        while True:
            try:
                line = self.lines.get(timeout=max(0, deadline - time.monotonic()))
            except queue.Empty as exc:
                raise ProtocolError('Timed out waiting for ' + prefix) from exc
            if line is None:
                raise ProtocolError('Engine exited before ' + prefix)
            if line.startswith('info string error'):
                raise ProtocolError('Engine reported a protocol or model error')
            received.append(line)
            if line == prefix or line.startswith(prefix + ' '):
                return received
            if time.monotonic() >= deadline:
                raise ProtocolError('Timed out waiting for ' + prefix)

    def quit(self):
        self.send('quit')
        try:
            code = self.process.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise ProtocolError('Engine did not exit after quit') from exc
        if code:
            raise ProtocolError('Engine exited with a failure status')

    def close(self):
        # This bound applies to the direct child, not descendant processes or memory.
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        self.reader.join(timeout=1)
        self.process.stdin.close()
        self.process.stdout.close()


def exercise(*, root: Path = ROOT, model: Path | None = None, timeout: float = 15,
             eval_file: Path | None = None, backend: str = 'python', command=None) -> dict:
    if model is not None and eval_file is not None:
        raise ValueError('Select either a checkpoint or NNUE weights')
    options = declared_options(root)
    sys.path.insert(0, str(root))
    from shogi import create_game
    from shogi.notation import move_to_usi, usi_to_move

    def legal_reply(line: str) -> str:
        tokens = line.split()
        if len(tokens) != 2 or tokens[1] not in {move_to_usi(action) for action in game.legal_actions()}:
            raise ProtocolError('Engine returned an illegal or malformed bestmove')
        return tokens[1]

    session = Session(command or [sys.executable, '-m', 'shogi.usi', '--backend', backend], root, timeout)
    game = create_game('python')
    try:
        session.send('usi')
        handshake = session.until('usiok')
        if {line for line in handshake if line.startswith('option ')} != set(options):
            raise ProtocolError('Declared USI options differ from the documented contract')
        session.send('setoption name Mode value strong')
        session.send('setoption name Depth value 1')
        session.send('setoption name MaxNodes value 2' if model else 'setoption name MaxNodes value 1000')
        if model:
            session.send('setoption name Model value ' + str(model.resolve()))
        if eval_file:
            session.send('setoption name Engine value alphabeta-cpp')
            session.send('setoption name Model value <empty>')
            session.send('setoption name EvalFile value ' + str(eval_file.resolve()))
        session.send('isready')
        session.until('readyok')
        session.send('usinewgame')
        moves = ['7g7f', '3c3d']
        for move in moves:
            game.play(usi_to_move(game, move))
        session.send('position startpos moves ' + ' '.join(moves))
        session.send('go btime 0 wtime 0 byoyomi 2000')
        replies = session.until('bestmove')
        move = legal_reply(replies[-1])
        if (model or eval_file) and not any(line.startswith('info depth ') for line in replies):
            raise ProtocolError('Model search did not complete before the deadline')
        moves.append(move)
        game.play(usi_to_move(game, move))
        session.send('position startpos moves ' + ' '.join(moves))
        session.send('go infinite')
        session.until('info depth')
        session.send('stop')
        stopped = legal_reply(session.until('bestmove')[-1])
        session.quit()
        return {'mode': 'nnue' if eval_file else 'model' if model else 'classical', 'ok': True,
                'bestmove': move, 'stopped_bestmove': stopped, 'options': list(options)}
    finally:
        game.close()
        session.close()


def installed_models(root: Path) -> list[Path]:
    """Return checkpoint directories and NNUE weight files declared by the local index."""
    index = root / 'models/index.json'
    if not index.exists():
        return []
    models = []
    for row in json.loads(index.read_text(encoding='utf-8'))['models']:
        if row['game'] != 'shogi':
            continue
        name = row['name']
        if not name or Path(name).name != name or name in ('.', '..') or '\\' in name:
            raise ValueError('Invalid model name')
        path = root / 'models' / name
        if path.exists():
            kind = row.get('asset_kind', 'checkpoint')
            if kind not in ('checkpoint', 'nnue'):
                raise ValueError('Unknown model asset kind')
            files = ('weights.hlnnw1', 'weights.hlnnw1.json') if kind == 'nnue' else ('manifest.json', 'weights.pt')
            if not all((path / file).is_file() for file in files):
                raise ValueError('Installed model is incomplete')
            models.append(path / 'weights.hlnnw1' if kind == 'nnue' else path)
    return models


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('python', 'cpp'), default='python')
    parser.add_argument('--timeout', type=float, default=15)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument('--model', type=Path)
    choice.add_argument('--eval-file', type=Path, help='NNUE weights loaded through the native USI EvalFile option')
    choice.add_argument('--classical-only', action='store_true')
    args = parser.parse_args(argv)
    if not 0 < args.timeout <= 300:
        parser.error('--timeout must be greater than zero and at most 300 seconds')
    try:
        checks = [exercise(backend=args.backend, timeout=args.timeout)]
        if args.classical_only:
            selections = []
        elif args.model:
            selections = [{'model': args.model}]
        elif args.eval_file:
            selections = [{'eval_file': args.eval_file}]
        else:
            selections = [{'eval_file' if model.is_file() else 'model': model}
                          for model in installed_models(ROOT)]
        checks.extend(exercise(**selection, backend=args.backend, timeout=args.timeout) for selection in selections)
        print(json.dumps({'ok': True, 'checks': checks,
                          'model_status': 'tested' if selections else 'not run: no model selected or installed'}))
        return 0
    except (ProtocolError, ValueError, OSError) as exc:
        print(json.dumps({'ok': False, 'error': str(exc)}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
