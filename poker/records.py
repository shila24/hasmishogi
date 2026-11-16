"""Versioned, seed-replayed JSON records; sampled worlds are not seed-replayable."""

from dataclasses import asdict
import json
from pathlib import Path

from .game import replay


def record_dict(game):
    if game.seed is None:
        raise ValueError('a redealt game has no replayable creation seed')
    return dict(version=1, ruleset_id=game.rules.ruleset_id, rules=game.rules.to_dict(), seed=game.seed,
                button=game.button, actions=game.moves,
                outcome=json.loads(json.dumps(asdict(game.outcome))))


def from_record(data):
    if not isinstance(data, dict) or set(data) != {'version', 'ruleset_id', 'rules', 'seed', 'button', 'actions', 'outcome'}:
        raise ValueError('invalid poker record fields')
    if type(data['version']) is not int or data['version'] != 1 or data['seed'] is None:
        raise ValueError('unsupported record version or missing seed')
    if not isinstance(data['actions'], list):
        raise ValueError('actions must be a list')
    game = replay(data['rules'], data['seed'], data['button'], data['actions'])
    if record_dict(game) != data:
        raise ValueError('record does not match replayed rules and outcome')
    return game


def save_record(game, path):
    Path(path).write_text(json.dumps(record_dict(game), allow_nan=False) + '\n', encoding='utf-8')
    return Path(path)


def load_record(path):
    return from_record(json.loads(Path(path).read_text(encoding='utf-8')))
