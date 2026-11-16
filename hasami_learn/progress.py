"""Read-only, standard-library training progress reports.

The trainer writes collect.wins as seat wins followed by draws/caps: for two
players, [seat 0 wins, seat 1 wins, draws/caps]. There is no learner index in
self-play. MetricsLogger moves longer arrays to extra.seat_wins. In evaluation,
Trainer.evaluate writes [learner wins, learner losses, draws], even when seats
rotate. Evaluation draws include half-scored experimental truncations; they are
not necessarily terminal game draws. Multiplayer wins mean strictly outranking
all opponents, losses mean being outranked, and draws include ties for first.
These meanings follow trainer.py, evaluation.py, selfplay.py, algorithms/base.py
and hasami_learn/metrics.py, not the ambiguous name 'wins'.

Repeated (kind, iteration, opponent) records use the last occurrence in file
order. Totals describe this retained history, not all physical work on retries.
All resume occurrences are counted. Info metadata fields retain their latest
recorded values independently of lifecycle-only info rows, including rows at the
same iteration. Completed progress follows the last update, checkpoint or resume
in file order. Timing estimates use collect + update seconds
for up to ten latest completed updates; evaluation/checkpoint overhead is absent.
The wall span includes downtime and overwritten records. Liveness is only a
heuristic (record timestamp age <= 120 seconds), never a process health check.
Only newline-terminated records are committed; an unfinished tail is held until
the writer finishes it. Watch retains offsets and aggregates, resetting on inode
replacement or shrinkage. Retained unique records still take memory and chart
preparation scans them; metrics JSON is not reparsed on every refresh.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import html
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any

from .runlock import PROGRESS_LOCK, cleanup_progress_files, run_lock, writer_path

Row = dict[str, Any]
KINDS = {'collect', 'update', 'evaluation', 'checkpoint', 'resume', 'info', 'opponent_pool'}
LIVE_SECONDS = 120
MAX_POINTS = 2000


def finite(value: Any) -> bool:
    """Reject booleans and non-finite or non-numeric values."""
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def timestamp(value: Any) -> float | None:
    if finite(value):
        return float(value)
    if isinstance(value, str):
        try:
            date = datetime.fromisoformat(value.replace('Z', '+00:00'))
            if date.tzinfo is None:
                return None
            return date.timestamp()
        except (ValueError, OverflowError, OSError):
            pass
    return None


def age(value: Any, now: float) -> float | None:
    stamp = timestamp(value)
    return max(0.0, now - stamp) if stamp is not None and stamp <= now else None


def mapping(value: Any) -> Row:
    return value if isinstance(value, dict) else {}


def read_object(path: Path) -> Row:
    try:
        return mapping(json.loads(path.read_text(encoding='utf-8')))
    except (OSError, ValueError):
        return {}


def clean_json(value: Any) -> Any:
    """Represent non-finite numbers as strings in interoperable JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return 'nan'
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    return value


def json_text(value: Any) -> str:
    return json.dumps(clean_json(value), ensure_ascii=False, allow_nan=False, indent=2)


class MetricsReader:
    """Incrementally retain deduplicated records without locking the writer."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.bytes_read = 0
        self._reset()

    def _reset(self) -> None:
        self.identity = None
        self.offset = 0
        self.pending = b''
        self.records: dict[tuple, Row] = {}
        self.info_metadata: Row = {}
        self.malformed_lines = 0
        self.unknown_kinds = 0
        self.resume_count = 0
        self.first_time = None
        self.last_time = None
        self.latest: Row = {}

    def refresh(self) -> None:
        try:
            with self.path.open('rb') as handle:
                stat = os.fstat(handle.fileno())
                identity = (stat.st_dev, stat.st_ino)
                if self.identity != identity or stat.st_size < self.offset:
                    self._reset()
                    self.identity = identity
                handle.seek(self.offset)
                # Limit this snapshot to the observed size, even for a busy writer.
                remaining = stat.st_size - self.offset
                while remaining:
                    block = handle.read(min(1024 * 1024, remaining))
                    if not block:
                        break
                    self.offset += len(block)
                    self.bytes_read += len(block)
                    remaining -= len(block)
                    lines = (self.pending + block).split(b'\n')
                    self.pending = lines.pop()
                    for line in lines:
                        self._consume(line)
        except FileNotFoundError:
            self._reset()

    def _consume(self, line: bytes) -> None:
        if not line.strip():
            return
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('record must be an object')
        except (ValueError, UnicodeError):
            self.malformed_lines += 1
            return
        self.latest = row
        stamp = timestamp(row.get('timestamp'))
        if stamp is not None:
            self.first_time = stamp if self.first_time is None else min(stamp, self.first_time)
            self.last_time = stamp if self.last_time is None else max(stamp, self.last_time)
        kind = row.get('kind')
        if not isinstance(kind, str) or kind not in KINDS:
            self.unknown_kinds += 1
            return
        if kind == 'resume':
            self.resume_count += 1
        iteration = row.get('iteration')
        if type(iteration) is not int or iteration < 0:
            self.malformed_lines += 1
            return
        opponent = row.get('opponent') if kind == 'evaluation' else None
        if opponent is not None and not isinstance(opponent, str):
            self.malformed_lines += 1
            return
        if kind == 'info':
            # Preserve metadata even when a lifecycle row replaces this iteration.
            extra = mapping(row.get('extra'))
            for field in ('backend', 'algorithm', 'network'):
                if field in extra:
                    self.info_metadata[field] = extra[field]
        key = (kind, iteration, opponent)
        self.records.pop(key, None)
        self.records[key] = row


def wilson_interval(successes: float, games: int) -> tuple[float, float]:
    """Wilson 95% interval with half points, matching autoloop evidence."""
    if type(games) is not int or games < 0 or not finite(successes) or not 0 <= successes <= games:
        raise ValueError('invalid counts')
    if not games:
        return 0.0, 1.0
    p = successes / games
    z2 = 1.96**2
    denominator = 1 + z2 / games
    center = (p + z2 / (2 * games)) / denominator
    margin = 1.96 * math.sqrt(p * (1 - p) / games + z2 / (4 * games**2)) / denominator
    return max(0, center - margin), min(1, center + margin)


def ratio(numerator: Any, denominator: Any) -> float | None:
    if finite(numerator) and finite(denominator) and denominator > 0:
        return numerator / denominator
    return None


def evaluation(row: Row) -> Row:
    counts = row.get('wins')
    counts = counts if isinstance(counts, list) and len(counts) == 3 else [None] * 3
    wins, losses, draws = counts
    games = row.get('games')
    valid = all(type(n) is int and n >= 0 for n in [games, *counts])
    valid = valid and sum(counts) == games
    score = ratio(wins + .5 * draws, games) if valid else None
    interval = wilson_interval(wins + .5 * draws, games) if valid else None
    extra = mapping(row.get('extra'))
    if 'effective_games' in extra:
        count, successes = extra.get('effective_games'), extra.get('effective_successes')
        interval = (wilson_interval(successes, count) if valid and type(count) is int and
                    0 <= count <= games and finite(successes) and 0 <= successes <= count else None)
    return dict(distinct_games=extra.get('distinct_games'), effective_games=extra.get('effective_games'),
                iteration=row['iteration'], games=games, wins=wins, losses=losses,
                draws=draws, truncated=row.get('truncated'), score=score, wilson95=interval)


def _total(rows: list[Row], key: str) -> int | float | None:
    values = [row.get(key) for row in rows]
    return sum(values) if values and all(finite(v) and v >= 0 for v in values) else None


def _checkpoint(directory: Path, now: float, records: list[Row]) -> Row | None:
    candidates = {}
    for path in (directory / 'checkpoints').glob('iter_*'):
        if not re.fullmatch(r'iter_\d+', path.name):
            continue
        try:
            if path.is_dir():
                candidates[int(path.name[5:])] = (path.stat().st_mtime, path)
        except FileNotFoundError:
            continue
    if not candidates:
        return None
    # Copying a run changes directory mtimes; use saved chronology when available.
    saved = [row['iteration'] for row in records
             if row['kind'] == 'checkpoint' and row['iteration'] in candidates]
    modified, path = candidates[saved[-1] if saved else max(candidates)]
    return {'path': str(path), 'age_seconds': age(modified, now)}


def summarize_run(directory: Path | str, *, reader: MetricsReader | None = None,
                  now: float | None = None) -> Row:
    directory = Path(directory)
    now = time.time() if now is None else now
    reader = reader if reader is not None else MetricsReader(directory / 'metrics.jsonl')
    reader.refresh()
    config = read_object(directory / 'config.json')
    run = mapping(config.get('run'))
    records = list(reader.records.values())
    collects = [row for row in records if row['kind'] == 'collect']
    updates = [row for row in records if row['kind'] == 'update']
    info = reader.info_metadata
    progress = [row for row in records if row['kind'] in ('update', 'checkpoint', 'resume')]
    done = progress[-1]['iteration'] if progress else 0
    planned = run.get('iterations')
    planned = planned if type(planned) is int and planned >= 0 else None
    by_iteration = {row['iteration']: row for row in collects}
    file_order = {id(row): index for index, row in enumerate(records)}
    durations = []
    recent = [row for row in updates if row['iteration'] <= done][-10:]
    for row in recent:
        collect_row = by_iteration.get(row['iteration'], {})
        # A replacement collect may belong to an unfinished retry of an old update.
        if file_order.get(id(collect_row), -1) > file_order[id(row)]:
            continue
        collect_seconds = collect_row.get('seconds')
        update_seconds = row.get('seconds')
        if all(finite(v) and v >= 0 for v in (collect_seconds, update_seconds)):
            durations.append(collect_seconds + update_seconds)
    mean = sum(durations) / len(durations) if durations else None
    eta = max(0, planned - done) * mean if planned is not None and mean is not None else None
    latest_update = updates[-1] if updates else {}
    earlier = next((row for row in updates if row['iteration'] == latest_update.get('iteration', -1) - 10), {})
    losses = {}
    for name in ('loss', 'policy_loss', 'value_loss'):
        current, previous = latest_update.get(name), earlier.get(name)
        change = current - previous if finite(current) and finite(previous) else None
        losses[name] = {'value': current, 'change_10': change}
    latest_collect = collects[-1] if collects else {}
    collect = {key: latest_collect.get(key) for key in
               ('iteration', 'games', 'plies', 'wins', 'truncated', 'terminal_reasons')}
    collect['seat_wins'] = mapping(latest_collect.get('extra')).get('seat_wins')
    collect['mean_plies'] = ratio(collect['plies'], collect['games'])
    evaluations: Row = {}
    for row in records:
        if row['kind'] == 'evaluation':
            name = row.get('opponent', '対戦相手不明')
            result = evaluation(row)
            prior = evaluations.get(name, {}).get('best')
            best = prior
            if result['score'] is not None and (prior is None or result['score'] >= prior['score']):
                best = result
            evaluations[name] = {'latest': result, 'best': best}
    devices = [row.get('device') for row in records if isinstance(row.get('device'), str) and row['device']]
    last_age = age(reader.latest.get('timestamp'), now)
    return {
        'type': 'run', 'path': str(directory),
        'run_id': run.get('run_id', reader.latest.get('run_id', directory.name)),
        'game': mapping(config.get('game')).get('name'),
        'algorithm': info.get('algorithm', mapping(config.get('algorithm')).get('name')),
        'network': mapping(config.get('network')).get('name', mapping(info.get('network')).get('name')),
        'device': devices[-1] if devices else None,
        'configured_device': mapping(config.get('device')).get('device'),
        'backend': info.get('backend'), 'configured_backend': run.get('backend'),
        'done': done, 'planned': planned,
        'wall_seconds': reader.last_time - reader.first_time if reader.first_time is not None else None,
        'mean_iteration_seconds': mean, 'timing_samples': len(durations), 'eta_seconds': eta,
        'total_games': _total(collects, 'games'), 'total_plies': _total(collects, 'plies'),
        'losses': losses, 'update_iteration': latest_update.get('iteration'), 'collect': collect,
        'evaluations': evaluations, 'checkpoint': _checkpoint(directory, now, records),
        'resume_count': reader.resume_count, 'last_record_age_seconds': last_age,
        'looks_live': last_age <= LIVE_SECONDS if last_age is not None else None,
        'malformed_lines': reader.malformed_lines, 'unknown_kinds': reader.unknown_kinds,
        'partial_line': bool(reader.pending.strip()), 'empty': not (collects or updates or evaluations),
        'series': records,
    }


def _loop_run_path(directory: Path, loop_id: str, stage: str, cycle: int) -> Path:
    """Reuse only the read-only layout helpers; never instantiate a controller."""
    try:
        from .autoloop.plan import Plan
        from .autoloop.stages import Stages
    except ImportError:
        return directory.parent / 'runs' / f'{loop_id}-{stage}-c{cycle}'
    layout = SimpleNamespace(output=directory.parent, data={'loop_id': loop_id})
    layout.run_id = lambda spec, number: Plan.run_id(layout, spec, number)
    return Stages(layout).run_dir({'id': stage}, cycle)


def _recent_events(path: Path, count: int = 20) -> tuple[list[Row], int]:
    """Read a bounded tail rather than rereading a potentially huge event log."""
    try:
        with path.open('rb') as handle:
            handle.seek(0, os.SEEK_END)
            start = max(0, handle.tell() - 256 * 1024)
            handle.seek(start)
            if start:
                handle.readline()
            lines = handle.read().split(b'\n')[:-1]
    except FileNotFoundError:
        return [], 0
    recent: deque = deque(maxlen=count)
    malformed = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError('event must be an object')
            recent.append(value)
        except (ValueError, UnicodeError):
            malformed += 1
    return list(recent), malformed


def summarize_loop(directory: Path | str, *, readers: dict | None = None,
                   now: float | None = None) -> Row:
    directory = Path(directory)
    now = time.time() if now is None else now
    readers = {} if readers is None else readers
    state = read_object(directory / 'loop_state.json')
    status = read_object(directory / 'status.json')
    loop_id = state.get('loop_id', directory.name)
    loop_id = loop_id if isinstance(loop_id, str) else directory.name
    candidates: set[Path] = set()
    for key in mapping(state.get('stages')):
        match = re.fullmatch(r'(\d+):(.+)', key)
        if match:
            candidates.add(_loop_run_path(directory, loop_id, match[2], int(match[1])))
    # The persisted stage log layout also locates runs before state completion.
    for cycle_dir in directory.glob('c[0-9]*'):
        if cycle_dir.is_dir() and cycle_dir.name[1:].isdigit():
            for stage_dir in cycle_dir.iterdir():
                if stage_dir.is_dir():
                    candidates.add(_loop_run_path(directory, loop_id, stage_dir.name, int(cycle_dir.name[1:])))
                    candidates.update(p.parent for p in stage_dir.rglob('metrics.jsonl'))
                    candidates.update(p.parent for p in stage_dir.rglob('config.json')
                                      if 'run' in read_object(p))
    runs_root = directory.parent / 'runs'
    if runs_root.is_dir():
        candidates.update(p for p in runs_root.iterdir()
                          if p.is_dir() and p.name.startswith(loop_id + '-')
                          and re.search(r'-c\d+$', p.name))
    current_id = status.get('run_id')
    current = runs_root / current_id if isinstance(current_id, str) else None
    if current:
        candidates.add(current)
    available = {p.resolve() for p in candidates if p.is_dir() and
                 ((p / 'config.json').is_file() or (p / 'metrics.jsonl').is_file())}
    def order(path: Path) -> tuple:
        match = re.search(r'-c(\d+)$', path.name)
        return (path != current.resolve() if current else True,
                -int(match[1]) if match else 0, str(path))
    runs = []
    for path in sorted(available, key=order):
        if path not in readers:
            readers[path] = MetricsReader(path / 'metrics.jsonl')
        runs.append(summarize_run(path, reader=readers[path], now=now))
    events, malformed = _recent_events(directory / 'events.jsonl')
    cycle = status.get('cycle', state.get('cycle'))
    if not status and state.get('status') == 'complete' and type(cycle) is int:
        cycle -= 1
    return {'type': 'loop', 'path': str(directory), 'loop_id': loop_id,
            'status': status.get('status', state.get('status')), 'cycle': cycle,
            'stage': status.get('stage'), 'free_gb': status.get('free_gb'),
            'heartbeat_age_seconds': age(status.get('last_heartbeat', state.get('last_heartbeat')), now),
            'gpu_probe': status.get('gpu_probe'), 'events': events,
            'malformed_event_tail_lines': malformed, 'runs': runs}


def display(value: Any) -> str:
    if value is None:
        return 'unknown（不明）'
    if isinstance(value, bool):
        return 'はい' if value else 'いいえ'
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(clean_json(value), ensure_ascii=False, allow_nan=False)
    if isinstance(value, float):
        return f'{value:.4g}' if finite(value) else 'nan'
    return str(value)


def summary_facts(data: Row) -> list[tuple[str, Any]]:
    """Share the same facts between text output and HTML tables."""
    if data['type'] == 'loop':
        return [('ループ', data['loop_id']), ('状態', data['status']), ('サイクル', data['cycle']),
                ('ステージ', data['stage']), ('記録された空き容量 (GiB)', data['free_gb']),
                ('最終ハートビート経過 (秒)', data['heartbeat_age_seconds']),
                ('記録された GPU 検査', data['gpu_probe']),
                ('イベント末尾の不正行数', data['malformed_event_tail_lines'])]
    facts = [('実行 ID', data['run_id']), ('ゲーム', data['game']), ('アルゴリズム', data['algorithm']),
             ('ネットワーク', data['network']), ('実測デバイス', data['device']),
             ('設定デバイス', data['configured_device']), ('実測バックエンド', data['backend']),
             ('設定バックエンド', data['configured_backend']),
             ('反復（更新完了 / 計画）', f"{data['done']} / {display(data['planned'])}"),
             ('記録の時間幅 (秒・休止を含む)', data['wall_seconds']),
             ('直近最大10反復の平均 (秒・収集＋更新)', data['mean_iteration_seconds']),
             ('時間推定に使えた反復数', data['timing_samples']),
             ('残り時間推定 (秒・評価と保存を除く)', data['eta_seconds']),
             ('自己対戦・収集合計の局数', data['total_games']), ('収集合計の手数', data['total_plies']),
             ('最新更新の反復', data['update_iteration'])]
    for key, label in [('loss', '総損失'), ('policy_loss', '方策損失'), ('value_loss', '価値損失')]:
        facts.extend([(label, data['losses'][key]['value']),
                      (label + '（10反復前との差）', data['losses'][key]['change_10'])])
    collect = data['collect']
    facts.extend([('最新収集の反復', collect['iteration']), ('最新収集の局数', collect['games']),
                  ('最新収集の平均手数', collect['mean_plies']),
                  ('収集勝敗 [席0勝, 席1勝, 引分・打切]（原記録）', collect['wins']),
                  ('多人数の席別勝数＋引分・打切（原記録）', collect['seat_wins']),
                  ('最新収集の打切数', collect['truncated']),
                  ('最新収集の終局理由別局数（max_plies は打切）', collect['terminal_reasons'])])
    for opponent, results in data['evaluations'].items():
        for key, label in [('latest', '最新'), ('best', '最高スコア')]:
            result = results[key]
            prefix = f'評価 {opponent} / {label}'
            if result is None:
                facts.append((prefix, None))
                continue
            games_label = result['games']
            if result.get('distinct_games') is not None:
                games_label = f"{games_label} (distinct_games={result['distinct_games']})"
            facts.extend([(prefix + ' 反復', result['iteration']), (prefix + ' 局数', games_label),
                          (prefix + ' [学習者勝, 引分・打切, 学習者負]（原記録）',
                           [result['wins'], result['draws'], result['losses']]),
                          (prefix + ' 打切数', result['truncated']), (prefix + ' スコア', result['score']),
                          (prefix + ' Wilson 95% 区間', result['wilson95'])])
    checkpoint = data['checkpoint'] or {}
    facts.extend([('最新チェックポイント（保存記録順・なければ最大反復・検証なし）', checkpoint.get('path')),
                  ('チェックポイントのディレクトリ更新から (秒)', checkpoint.get('age_seconds')),
                  ('再開記録の総数', data['resume_count']),
                  ('最終記録からの経過 (秒)', data['last_record_age_seconds']),
                  ('記録が新しい（120秒以内・稼働保証なし）', data['looks_live']),
                  ('不正行数', data['malformed_lines']), ('未知の記録種別数', data['unknown_kinds']),
                  ('改行待ちの末尾あり', data['partial_line'])])
    return facts


def render_text(data: Row) -> str:
    parts = [f'{label}: {display(value)}' for label, value in summary_facts(data)]
    if data['type'] == 'loop':
        parts.append('最近のイベント（最大20件・末尾256 KiB内）:')
        parts.extend(display(event) for event in data['events'])
        parts.extend('\n' + render_text(run) for run in data['runs'])
        if not data['runs']:
            parts.append('まだ記録がありません')
    elif data['empty']:
        parts.append('まだ記録がありません')
    return '\n'.join(parts)


def _escape(value: Any) -> str:
    return html.escape(display(value), quote=True)


def _table(data: Row) -> str:
    return '<table><tbody>' + ''.join(
        f'<tr><th scope="row">{_escape(label)}</th><td>{_escape(value)}</td></tr>'
        for label, value in summary_facts(data)) + '</tbody></table>'


def chart(title: str, ylabel: str, series: list[tuple[str, list]], *, score: bool = False) -> str:
    """Render finite points and optional Wilson error bars in standalone SVG."""
    colors = ['#1769aa', '#bd3c32', '#19855d', '#8456aa', '#946400', '#b12b78']
    prepared = []
    downsampled = False
    for label, points in series:
        valid = [point for point in points if finite(point[0]) and finite(point[1])]
        valid.sort(key=lambda point: point[0])
        if len(valid) > MAX_POINTS:
            downsampled = True
            valid = [valid[round(i * (len(valid) - 1) / (MAX_POINTS - 1))] for i in range(MAX_POINTS)]
        prepared.append((label, valid))
    points = [point for _, values in prepared for point in values]
    xmin = min((p[0] for p in points), default=0)
    xmax = max((p[0] for p in points), default=1)
    if xmin == xmax:
        xmin, xmax = xmin - .5, xmax + .5
    ys = [v for p in points for v in p[1:] if finite(v)]
    ymin, ymax = (0, 1) if score else (min(ys, default=0), max(ys, default=1))
    if ymin == ymax:
        padding = max(abs(ymin) * .1, .5)
        ymin, ymax = ymin - padding, ymax + padding
    def x(value):
        return 86 + (value - xmin) / (xmax - xmin) * 620
    def y(value):
        return 270 - (value - ymin) / (ymax - ymin) * 230
    svg = [f'<svg viewBox="0 0 760 335" role="img" aria-label="{_escape(title)}">',
           f'<title>{_escape(title)}</title>',
           '<rect x="86" y="40" width="620" height="230" fill="white" stroke="#aaa"/>']
    for tick in range(5):
        xv = xmin + (xmax - xmin) * tick / 4
        yv = ymin + (ymax - ymin) * tick / 4
        svg.extend([f'<line x1="86" x2="706" y1="{y(yv):.2f}" y2="{y(yv):.2f}" stroke="#e3e8ee"/>',
                    f'<text x="{x(xv):.2f}" y="291" text-anchor="middle">{xv:.4g}</text>',
                    f'<text x="77" y="{y(yv)+4:.2f}" text-anchor="end">{yv:.4g}</text>'])
    svg.extend(['<text x="396" y="321" text-anchor="middle">反復</text>',
                f'<text transform="translate(18 155) rotate(-90)" text-anchor="middle">{_escape(ylabel)}</text>'])
    if score:
        svg.append(f'<line x1="86" x2="706" y1="{y(.5):.2f}" y2="{y(.5):.2f}" stroke="#555" stroke-dasharray="6 4"/>')
    legend = []
    for index, (label, values) in enumerate(prepared):
        color = colors[index % len(colors)]
        legend.append(f'<span style="color:{color}">● {_escape(label)}</span>')
        if values:
            coordinates = ' '.join(f'{x(p[0]):.2f},{y(p[1]):.2f}' for p in values)
            svg.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>')
        for point in values:
            if len(point) == 4 and finite(point[2]) and finite(point[3]):
                px, low, high = x(point[0]), y(point[2]), y(point[3])
                svg.append(f'<path d="M {px:.2f} {low:.2f} V {high:.2f} M {px-3:.2f} {low:.2f} H {px+3:.2f} M {px-3:.2f} {high:.2f} H {px+3:.2f}" stroke="{color}" opacity="0.25"/>')
            svg.append(f'<circle cx="{x(point[0]):.2f}" cy="{y(point[1]):.2f}" r="2" fill="{color}"/>')
    if not points:
        svg.append('<text x="396" y="155" text-anchor="middle">まだ記録がありません</text>')
    svg.append('</svg>')
    if score:
        legend.append('<span>破線: 0.5 / 縦棒: Wilson 95% 区間</span>')
    caption = '横軸: 反復。非有限値・欠損値は描画対象外。'
    if downsampled:
        caption += ' 描画は各系列最大2,000点へ間引き（両端を保持、要約は全件）。'
    return (f'<figure><h3>{_escape(title)}</h3>' + ''.join(svg) +
            '<div class="legend">' + ' '.join(legend) + '</div>' +
            f'<figcaption>{caption}</figcaption></figure>')


def _charts(data: Row) -> str:
    rows = data['series']
    def points(kind, field):
        return [(r['iteration'], r.get(field)) for r in rows if r['kind'] == kind]
    losses = [(label, points('update', field)) for label, field in
              [('総損失', 'loss'), ('方策損失', 'policy_loss'), ('価値損失', 'value_loss')]]
    scores: dict[str, list] = {}
    for row in rows:
        if row['kind'] == 'evaluation':
            value = evaluation(row)
            scores.setdefault(row.get('opponent', '対戦相手不明'), []).append(
                (row['iteration'], value['score'], *(value['wilson95'] or (None, None))))
    collects = [r for r in rows if r['kind'] == 'collect']
    plies = [(r['iteration'], ratio(r.get('plies'), r.get('games'))) for r in collects]
    truncated = [(r['iteration'], ratio(r.get('truncated'), r.get('games'))) for r in collects]
    reasons = sorted({key for row in collects for key in mapping(row.get('terminal_reasons'))})
    reason_series = [(reason, [(row['iteration'], mapping(row.get('terminal_reasons')).get(reason, 0)
                               if isinstance(row.get('terminal_reasons'), dict) else None)
                              for row in collects]) for reason in reasons]
    return ''.join([
        chart('学習損失', '損失', losses),
        chart('対戦相手別の評価', '学習者スコア', list(scores.items()) or [('評価', [])], score=True),
        chart('自己対戦・収集の長さ', '平均手数 / 局', [('平均手数', plies)]),
        chart('自己対戦・収集の打切率', '打切局数 / 局数', [('打切率', truncated)]),
        chart('自己対戦・収集の終局理由（max_plies は打切）', '局数', reason_series or [('未記録', [])]),
        chart('反復ごとの処理時間', '秒', [('収集', points('collect', 'seconds')), ('更新', points('update', 'seconds'))]),
        chart('リプレイの大きさ', '要素数', [('収集時', points('collect', 'replay_size')), ('更新時', points('update', 'replay_size'))]),
    ])


def render_html(data: Row, refresh: float | None = None) -> str:
    if refresh is not None and (not finite(refresh) or refresh <= 0):
        raise ValueError('refresh must be positive and finite')
    meta = f'<meta http-equiv="refresh" content="{refresh:g}">' if refresh else ''
    parts = ['<!doctype html><html lang="ja"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">', meta,
             '<title>学習進捗</title><style>',
             'body{font-family:system-ui,sans-serif;margin:2rem auto;padding:0 1rem;max-width:1100px;color:#203044;background:#f4f7fa}'
             'table{border-collapse:collapse;width:100%;background:white}th,td{padding:.45rem .7rem;border-bottom:1px solid #ddd;text-align:left;overflow-wrap:anywhere}'
             'th{width:45%}figure{margin:1.5rem 0;padding:1rem;background:white;border-radius:8px}svg{width:100%;height:auto}svg text{font-size:12px;fill:#203044}'
             '.legend{display:flex;gap:1rem;flex-wrap:wrap}figcaption,p{line-height:1.65}pre{white-space:pre-wrap;overflow-wrap:anywhere}',
             '</style></head><body><h1>学習進捗</h1>',
             '<p>記録を読み取った時点の報告です。損失の低下だけで強さは判断できません。'
             '評価スコアは（学習者勝数＋0.5×記録上の引分数）/ 局数。'
             '引分数には打切の半点が含まれます。ベースライン相手の結果は対人棋力の証明ではありません。'
             '再開後の重複は同じ種別・反復・相手の最後の記録を使用します。</p>']
    if data['type'] == 'loop':
        parts.extend([_table(data), '<h2>最近のイベント（最大20件・末尾256 KiB内）</h2>',
                      '<pre>' + _escape(data['events']) + '</pre>'])
        runs = data['runs']
        if not runs:
            parts.append('<p>まだ記録がありません</p>')
    else:
        runs = [data]
    for run in runs:
        parts.extend([f'<section><h2>{_escape(run["run_id"])}</h2>', _table(run)])
        if run['empty']:
            parts.append('<p>まだ記録がありません</p>')
        parts.extend([_charts(run), '</section>'])
    parts.append('</body></html>')
    return ''.join(parts)


def write_html(path: Path | str, document: str) -> None:
    """Recover dead reporters and publish without acquiring the training lock."""
    path = writer_path(path)
    with run_lock(path.parent, PROGRESS_LOCK, blocking=True) as directory:
        path = directory / path.name
        cleanup_progress_files(directory)
        fd, temporary = tempfile.mkstemp(prefix='.progress-', suffix='.tmp', dir=directory)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                handle.write(document)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)


def _positive(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('正の秒数を指定してください') from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('正の有限の秒数を指定してください')
    return number


def _one_line(data: Row) -> str:
    if data['type'] == 'loop':
        value = f"ループ {display(data['loop_id'])}: {display(data['status'])} / サイクル {display(data['cycle'])} / ステージ {display(data['stage'])} / 実行数 {len(data['runs'])}"
    else:
        value = f"{display(data['run_id'])}: 反復 {data['done']}/{display(data['planned'])} / 残り推定 {display(data['eta_seconds'])} 秒 / 最終記録 {display(data['last_record_age_seconds'])} 秒前"
    return value.replace('\r', '\\r').replace('\n', '\\n')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='標準ライブラリだけで学習進捗を表示します')
    commands = parser.add_subparsers(dest='command', required=True)
    for command, help_text in [('summary', '要約を表示'), ('html', 'HTMLを保存'), ('watch', 'HTMLを定期更新')]:
        sub = commands.add_parser(command, help=help_text)
        source = sub.add_mutually_exclusive_group(required=True)
        source.add_argument('--run', type=Path, help='学習実行ディレクトリ')
        source.add_argument('--loop', type=Path, help='自律ループディレクトリ')
        if command == 'summary':
            sub.add_argument('--json', action='store_true', help='JSON形式で表示')
        else:
            sub.add_argument('--out', type=Path, help='出力先（既定: 入力先/progress.html）')
            sub.add_argument('--refresh', type=_positive, help='ブラウザーの再読込間隔（秒）')
        if command == 'watch':
            sub.add_argument('--interval', type=_positive, default=30, help='生成間隔（秒、既定30）')
            sub.add_argument('--once', action='store_true', help='1回だけ生成')
    args = parser.parse_args(argv)
    directory = args.run or args.loop
    readers: dict = {}
    reader = MetricsReader(directory / 'metrics.jsonl') if args.run else None
    try:
        while True:
            data = (summarize_run(directory, reader=reader) if args.run else
                    summarize_loop(directory, readers=readers))
            if args.command == 'summary':
                print(json_text(data) if args.json else render_text(data))
                return 0
            output = args.out or directory / 'progress.html'
            write_html(output, render_html(data, refresh=args.refresh))
            print(_one_line(data), flush=True)
            if args.command == 'html' or args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError) as exc:
        print(f'進捗の読取り・保存に失敗しました: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
