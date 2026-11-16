"""Dependency-free ctypes bridge to the independent native riichi engine."""
from __future__ import annotations

import ctypes as C
from dataclasses import fields
from functools import lru_cache
import json
import os
from pathlib import Path
import secrets
import sys
from typing import Iterable

from .agari import Meld
from .rules import RiichiRules
from .scoring import NoYakuError
from .tiles import Wall, validate_counts, validate_ids
from .yaku import WinContext


class rq_rules(C.Structure):
    _fields_ = (
        [(n, C.c_int32) for n in ('rounds', 'start_points', 'return_points')]
        + [('uma', C.c_int32 * 4)]
        + [(n, C.c_int32) for n in (
            'red_fives', 'kuitan', 'atozuke', 'ippatsu', 'ura_dora', 'kan_dora',
            'open_kan_dora_after_discard', 'closed_kan_dora_immediate',
            'double_ron', 'atama_hane', 'triple_ron_abort', 'kyuushu_kyuuhai',
            'suufon_renda', 'suucha_riichi', 'suukaikan', 'tenpai_payment',
            'riichi_stick', 'honba', 'tobi', 'agari_yame', 'kazoe_yakuman',
            'double_yakuman', 'multiple_yakuman', 'pao', 'renhou',
            'kiriage_mangan', 'double_wind_pair_fu', 'nagashi_mangan',
            'discard_furiten', 'temporary_furiten', 'riichi_furiten', 'track_tsumogiri')]
    )


class rq_meld(C.Structure):
    _fields_ = [('kind', C.c_int32), ('open', C.c_int32), ('tiles', C.c_int32 * 4)]


class rq_win_context(C.Structure):
    _fields_ = (
        [(n, C.c_int32) for n in (
            'tsumo', 'seat_wind', 'round_wind', 'riichi', 'double_riichi',
            'ippatsu', 'rinshan', 'chankan', 'haitei', 'houtei', 'tenhou', 'chiihou', 'renhou')]
        + [('n_dora', C.c_int32), ('dora_indicators', C.c_int32 * 5),
           ('n_ura', C.c_int32), ('ura_indicators', C.c_int32 * 5),
           ('red_dora', C.c_int32), ('yaku_established', C.c_int32)]
    )


class rq_state(C.Structure):
    _fields_ = (
        [(n, C.c_int32) for n in (
            'terminal', 'winner', 'reason', 'phase', 'turn', 'dealer', 'round_index',
            'honba', 'riichi_sticks', 'hand_number', 'hand_start', 'num_moves', 'remaining_wall')]
        + [('scores', C.c_int32 * 4), ('final_points', C.c_double * 4), ('seed', C.c_uint64)]
    )


class rq_hidden(C.Structure):
    _fields_ = [
        ('hand_sizes', C.c_int32 * 4), ('hands', (C.c_int32 * 14) * 4),
        ('live_count', C.c_int32), ('live_wall', C.c_int32 * 122),
        ('dead_wall', C.c_int32 * 14), ('drawn_tiles', C.c_int32 * 4),
        ('temporary_furiten', C.c_int32 * 4), ('riichi_furiten', C.c_int32 * 4),
        ('seed', C.c_uint64),
    ]


_RULE_FIELDS = {name for name, _ in rq_rules._fields_}
assert _RULE_FIELDS == {f.name for f in fields(RiichiRules)} - {'ruleset_id'}


def _int32(value: int, name: str) -> int:
    if type(value) is not int or not -(1 << 31) <= value < (1 << 31):
        raise ValueError(f'{name} must be an integer in the int32 range')
    return value


def pack_rules(rules: RiichiRules) -> rq_rules:
    if not isinstance(rules, RiichiRules):
        raise ValueError('expected RiichiRules')
    result = rq_rules()
    for field in fields(rules):
        name, value = field.name, getattr(rules, field.name)
        if name == 'ruleset_id':
            continue
        if name == 'rounds':
            value = {'hanchan': 0, 'tonpuu': 1}[value]
        elif name == 'renhou':
            value = {'none': 0, 'mangan': 1, 'yakuman': 2}[value]
        elif name == 'uma':
            value = (C.c_int32 * 4)(*(_int32(n, name) for n in value))
        elif type(value) is bool:
            value = int(value)
        else:
            value = _int32(value, name)
        setattr(result, name, value)
    return result


def pack_context(context: WinContext) -> rq_win_context:
    if not isinstance(context, WinContext):
        raise ValueError('expected WinContext')
    result = rq_win_context()
    for field in fields(context):
        name, value = field.name, getattr(context, field.name)
        if name in ('dora_indicators', 'ura_indicators'):
            setattr(result, 'n_dora' if name == 'dora_indicators' else 'n_ura', len(value))
            value = (C.c_int32 * 5)(*value)
        elif name == 'yaku_established':
            value = -1 if value is None else int(value)
        elif type(value) is bool:
            value = int(value)
        setattr(result, name, value)
    return result


def _pack_hand(counts: Iterable[int], melds: Iterable[Meld]):
    values = validate_counts(counts)
    called = tuple(melds)
    if len(called) > 4 or any(not isinstance(m, Meld) for m in called):
        raise ValueError('expected at most four Meld objects')
    packed = (rq_meld * len(called))()
    for i, meld in enumerate(called):
        packed[i].kind = {'chi': 0, 'pon': 1, 'kan': 2}[meld.kind]
        packed[i].open = int(meld.open)
        packed[i].tiles = (C.c_int32 * 4)(*meld.tiles)
    return (C.c_int32 * 34)(*values), packed


def find_library() -> Path:
    explicit = os.environ.get('RIICHI_LIB')
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f'RIICHI_LIB does not exist: {path}')
        return path
    root = Path(__file__).resolve().parent.parent
    filename = ('riichi_native.dll' if sys.platform == 'win32' else
                'libriichi_native.dylib' if sys.platform == 'darwin' else 'libriichi_native.so')
    for directory in (root / 'build', root / 'build' / 'Release', root / 'build' / 'Debug'):
        path = directory / filename
        if path.is_file():
            return path
    raise RuntimeError('Native riichi library not found. Run: python3 scripts/build_native.py')


@lru_cache(maxsize=4)
def _load_path(path: str):
    lib = C.CDLL(path)
    integer = C.c_int32
    ints = C.POINTER(integer)
    melds = C.POINTER(rq_meld)
    rules = C.POINTER(rq_rules)
    signatures = {
        'rq_abi_version': ([], integer),
        'rq_last_error': ([], C.c_char_p),
        'rq_default_rules': ([rules], integer),
        'rq_walls': ([C.c_uint64, integer, ints, integer], integer),
        'rq_shanten': ([ints, melds, integer, ints], integer),
        'rq_waits': ([ints, melds, integer, ints], integer),
        'rq_is_agari': ([ints, melds, integer], integer),
        'rq_score_hand_json': ([ints, integer, melds, integer, C.POINTER(rq_win_context),
                               rules, integer, integer, ints, integer, C.POINTER(C.c_char), integer], integer),
        'rq_placement_points': ([ints, rules, C.POINTER(C.c_double)], integer),
    }
    try:
        for name, (args, result) in signatures.items():
            function = getattr(lib, name)
            function.argtypes, function.restype = args, result
        if lib.rq_abi_version() != 1:
            raise RuntimeError('Native riichi ABI version mismatch; rebuild the library')
        for name, structure in (('rules', rq_rules), ('meld', rq_meld),
                                ('win_context', rq_win_context), ('state', rq_state), ('hidden', rq_hidden)):
            function = getattr(lib, 'rq_sizeof_' + name)
            function.argtypes, function.restype = [], integer
            if function() != C.sizeof(structure):
                raise RuntimeError(f'Native riichi ABI layout mismatch: {name}')
    except AttributeError as error:
        raise RuntimeError('Native riichi core ABI symbols missing; rebuild the library') from error
    return lib


def load_library():
    return _load_path(str(find_library()))


def core_available() -> bool:
    try:
        load_library()
    except (RuntimeError, OSError):
        return False
    return True


_GAME_SYMBOLS = (
    'rq_create', 'rq_destroy', 'rq_clone', 'rq_get_state', 'rq_get_rules',
    'rq_actors', 'rq_legal_actions', 'rq_play_joint', 'rq_encode', 'rq_legal_mask',
    'rq_moves', 'rq_observation_json', 'rq_last_hand_json', 'rq_redeal',
)


def available() -> bool:
    try:
        lib = load_library()
        return all(hasattr(lib, name) for name in _GAME_SYMBOLS)
    except (RuntimeError, OSError):
        return False


def _check(lib, status: int) -> int:
    if status < 0:
        message = lib.rq_last_error()
        error = NoYakuError if status == -2 else ValueError
        raise error(message.decode('utf-8', 'replace') if message else 'Unknown native error')
    return status


def shanten_all(counts: Iterable[int], melds: Iterable[Meld] = ()) -> tuple[int, ...]:
    hand, called = _pack_hand(counts, melds)
    out = (C.c_int32 * 4)()
    lib = load_library()
    _check(lib, lib.rq_shanten(hand, called, len(called), out))
    return tuple(out)


def shanten(counts: Iterable[int], melds: Iterable[Meld] = ()) -> int:
    return shanten_all(counts, melds)[0]


def waits(counts: Iterable[int], melds: Iterable[Meld] = ()) -> tuple[int, ...]:
    hand, called = _pack_hand(counts, melds)
    out = (C.c_int32 * 34)()
    lib = load_library()
    _check(lib, lib.rq_waits(hand, called, len(called), out))
    return tuple(t for t, present in enumerate(out) if present)


def is_agari(counts: Iterable[int], melds: Iterable[Meld] = ()) -> bool:
    hand, called = _pack_hand(counts, melds)
    lib = load_library()
    return bool(_check(lib, lib.rq_is_agari(hand, called, len(called))))


def score_hand(hand: Iterable[int], winning_tile: int, melds: Iterable[Meld] = (), *,
               context: WinContext = WinContext(), rules: RiichiRules = RiichiRules(),
               honba: int = 0, riichi_sticks: int = 0,
               tile_ids: Iterable[int] | None = None) -> dict:
    counts, called = _pack_hand(hand, melds)
    win = _int32(winning_tile, 'winning_tile')
    ctx, rule = pack_context(context), pack_rules(rules)
    ids = None if tile_ids is None else validate_ids(tile_ids)
    packed_ids = None if ids is None else (C.c_int32 * len(ids))(*ids)
    args = (counts, win, called, len(called), C.byref(ctx), C.byref(rule),
            _int32(honba, 'honba'), _int32(riichi_sticks, 'riichi_sticks'),
            packed_ids, 0 if ids is None else len(ids))
    lib = load_library()
    size = _check(lib, lib.rq_score_hand_json(*args, None, 0))
    out = C.create_string_buffer(size + 1)
    _check(lib, lib.rq_score_hand_json(*args, out, len(out)))
    return json.loads(out.value)


def walls(seed: int | None, count: int = 1) -> tuple[Wall, ...]:
    if seed is None:
        seed = secrets.randbits(63)
    if type(seed) is not int or not 0 <= seed < (1 << 64):
        raise ValueError('seed must be an integer in [0, 2^64)')
    count = _int32(count, 'count')
    lib = load_library()
    size = _check(lib, lib.rq_walls(seed, count, None, 0))
    out = (C.c_int32 * size)()
    _check(lib, lib.rq_walls(seed, count, out, size))
    return tuple(Wall(tuple(out[i:i + 122]), tuple(out[i + 122:i + 136]))
                 for i in range(0, size, 136))


def placement_points(scores: Iterable[int], rules: RiichiRules = RiichiRules()) -> tuple[float, ...]:
    scores = tuple(scores)
    if len(scores) != 4:
        raise ValueError('expected four integer scores')
    packed = (C.c_int32 * 4)(*(_int32(n, 'score') for n in scores))
    rule, out = pack_rules(rules), (C.c_double * 4)()
    lib = load_library()
    _check(lib, lib.rq_placement_points(packed, C.byref(rule), out))
    return tuple(out)


def _configure_game(lib):
    integer, handle = C.c_int32, C.c_void_p
    ints, chars = C.POINTER(integer), C.POINTER(C.c_char)
    signatures = {
        'rq_create': ([C.c_uint64, C.POINTER(rq_rules)], handle),
        'rq_destroy': ([handle], None), 'rq_clone': ([handle], handle),
        'rq_get_state': ([handle, C.POINTER(rq_state)], integer),
        'rq_get_rules': ([handle, C.POINTER(rq_rules)], integer),
        'rq_actors': ([handle, ints], integer),
        'rq_legal_actions': ([handle, integer, ints, integer], integer),
        'rq_play_joint': ([handle, ints], integer),
        'rq_encode': ([handle, integer, C.POINTER(C.c_float), integer], integer),
        'rq_legal_mask': ([handle, integer, C.POINTER(C.c_float), integer], integer),
        'rq_moves': ([handle, ints, integer], integer),
        'rq_observation_json': ([handle, integer, chars, integer], integer),
        'rq_last_hand_json': ([handle, chars, integer], integer),
        'rq_redeal': ([handle, C.POINTER(rq_hidden)], integer),
    }
    try:
        for name, (args, result) in signatures.items():
            function = getattr(lib, name)
            function.argtypes, function.restype = args, result
    except AttributeError as exc:
        raise RuntimeError('Native riichi game ABI symbols missing; rebuild the library') from exc


def _seed(seed):
    if seed is None:
        seed = secrets.randbits(63)
    if type(seed) is not int or not 0 <= seed < 2 ** 64:
        raise ValueError('seed must be an integer in [0, 2**64)')
    return seed


def _player(player):
    if type(player) is not int or not 0 <= player < 4:
        raise ValueError('player must be an integer in 0..3')
    return player


class NativeRiichiGame:
    """Owned native match handle with dependency-free observation and buffer APIs."""

    def __init__(self, seed: int | None = None, rules: RiichiRules | None = None):
        self._handle = None
        seed = _seed(seed)
        if rules is None:
            rules = RiichiRules(nagashi_mangan=False)
        packed = pack_rules(rules)
        self._lib = load_library()
        _configure_game(self._lib)
        self._handle = self._lib.rq_create(seed, C.byref(packed))
        if not self._handle:
            _check(self._lib, -1)
        self.rules = rules
        self.moves: list[dict[int, int]] = []
        self.initial = {}
        self._last_hand = None
        self._refresh()

    def _require_open(self):
        if not self._handle:
            raise ValueError('game is closed')
        return self._handle

    def _refresh(self):
        self._state = rq_state()
        _check(self._lib, self._lib.rq_get_state(self._require_open(), C.byref(self._state)))
        from .game import Outcome
        self._outcome = Outcome(bool(self._state.terminal), self._state.winner,
                                ('ongoing', 'tobi', 'agari_yame', 'rounds_complete')[self._state.reason])

    def _scalar(self, name):
        self._require_open()
        return getattr(self._state, name)

    seed = property(lambda self: self._scalar('seed'))
    hand_number = property(lambda self: self._scalar('hand_number'))
    hand_start = property(lambda self: self._scalar('hand_start'))
    round_index = property(lambda self: self._scalar('round_index'))
    dealer = property(lambda self: self._scalar('dealer'))
    honba = property(lambda self: self._scalar('honba'))
    riichi_sticks = property(lambda self: self._scalar('riichi_sticks'))
    turn = property(lambda self: self._scalar('turn'))
    remaining_wall = property(lambda self: self._scalar('remaining_wall'))
    scores = property(lambda self: list(self._scalar('scores')))
    phase = property(lambda self: ('draw', 'discard', 'claim', 'kan_claim', 'terminal')[self._scalar('phase')])

    @property
    def outcome(self):
        self._require_open()
        return self._outcome

    @property
    def final_points(self):
        return tuple(self._scalar('final_points')) if self.outcome.terminal else None

    @property
    def position(self):
        from .game import Position
        return Position(self.turn)

    @property
    def history(self):
        self._require_open()
        return [dict(step) for step in self.moves]

    def _json(self, function, *args):
        handle = self._require_open()
        size = _check(self._lib, function(handle, *args, None, 0))
        out = C.create_string_buffer(size + 1)
        _check(self._lib, function(handle, *args, out, len(out)))
        return json.loads(out.value)

    @property
    def last_hand(self):
        import copy
        self._require_open()
        if self._last_hand is None:
            self._last_hand = self._json(self._lib.rq_last_hand_json)
        return copy.deepcopy(self._last_hand)

    def actors(self) -> list[int]:
        out = (C.c_int32 * 4)()
        count = _check(self._lib, self._lib.rq_actors(self._require_open(), out))
        return list(out[:count])

    def legal_actions(self, player: int | None = None) -> list[int]:
        self._require_open()
        if player is None:
            actors = self.actors()
            if len(actors) != 1:
                raise ValueError('specify player for simultaneous or terminal decisions')
            player = actors[0]
        player = _player(player)
        out = (C.c_int32 * 111)()
        count = _check(self._lib, self._lib.rq_legal_actions(self._handle, player, out, 111))
        return list(out[:count])

    def play_joint(self, actions: dict[int, int]):
        self._require_open()
        if (not isinstance(actions, dict) or self.outcome.terminal
                or any(type(p) is not int for p in actions) or set(actions) != set(self.actors())):
            raise ValueError('joint actions must match the current actors')
        packed = (C.c_int32 * 4)(-1, -1, -1, -1)
        for player, action in actions.items():
            if type(action) is not int or not 0 <= action < 111:
                raise ValueError(f'illegal action for player {player}: {action!r}')
            packed[player] = action
        joint = dict(actions)
        _check(self._lib, self._lib.rq_play_joint(self._handle, packed))
        self.moves.append(joint)
        self._refresh()
        self._last_hand = None
        return self.outcome

    def play(self, action: int):
        actors = self.actors()
        if len(actors) != 1:
            raise ValueError('play_joint is required for simultaneous decisions')
        return self.play_joint({actors[0]: action})

    def observation(self, player: int) -> dict:
        obs = self._json(self._lib.rq_observation_json, _player(player))
        for name in ('last_discard', 'pending_kan'):
            if obs[name] is not None:
                obs[name] = tuple(obs[name])
        return obs

    def _into(self, function, player, buffer, size):
        self._require_open()
        player = _player(player)
        try:
            view = memoryview(buffer)
        except TypeError as exc:
            raise ValueError('expected a writable C-contiguous float32 buffer') from exc
        if (view.readonly or not view.c_contiguous or view.itemsize != 4
                or view.format not in ('f', '@f', '=f', '<f' if sys.byteorder == 'little' else '>f')
                or view.nbytes != size * 4):
            raise ValueError('expected a writable C-contiguous float32 buffer of the required size')
        target = C.c_float.from_buffer(view)
        if C.addressof(target) % C.alignment(C.c_float):
            # Buffer exporters can expose unaligned float32 views. C++ requires alignment.
            aligned = (C.c_float * size)()
            _check(self._lib, function(self._handle, player, aligned, size))
            C.memmove(C.addressof(target), aligned, view.nbytes)
        else:
            _check(self._lib, function(self._handle, player, C.byref(target), size))

    def encode_into(self, player: int, buffer) -> None:
        self._into(self._lib.rq_encode, player, buffer, 16320)

    def legal_mask_into(self, player: int, buffer) -> None:
        self._into(self._lib.rq_legal_mask, player, buffer, 111)

    def redeal(self, hands, live_wall, dead_wall, drawn_tiles,
               temporary_furiten, riichi_furiten, seed: int) -> None:
        self._require_open()
        if self.outcome.terminal:
            raise ValueError('cannot redeal a terminal game')
        if type(seed) is not int or not 0 <= seed < 2 ** 64:
            raise ValueError('redeal seed must be an integer in [0, 2**64)')
        hands = [list(hand) for hand in hands]
        live, dead, drawn = list(live_wall), list(dead_wall), list(drawn_tiles)
        flags = [list(temporary_furiten), list(riichi_furiten)]
        if len(hands) != 4 or len(drawn) != 4 or any(len(f) != 4 for f in flags):
            raise ValueError('redeal needs four hands, drawn tiles and furiten flags')
        if any(type(v) is not bool for f in flags for v in f):
            raise ValueError('furiten flags must be booleans')
        tiles = [t for hand in hands for t in hand] + live + dead
        if any(type(t) is not int for t in tiles):
            raise ValueError('tile ids must be integers')
        if any(not 0 <= t < 136 for t in tiles):
            raise ValueError('redeal must permute exactly the hidden tiles')
        if any(len(hand) > 14 for hand in hands) or len(live) > 122 or len(dead) != 14:
            raise ValueError('hand sizes and wall lengths are public and must not change')
        hidden = rq_hidden()
        for p, hand in enumerate(hands):
            hidden.hand_sizes[p] = len(hand)
            hidden.hands[p] = (C.c_int32 * 14)(*hand)
            if drawn[p] is not None and (type(drawn[p]) is not int or drawn[p] not in hand):
                raise ValueError('a drawn tile must be in its seat hand')
            hidden.drawn_tiles[p] = -1 if drawn[p] is None else drawn[p]
            hidden.temporary_furiten[p], hidden.riichi_furiten[p] = flags[0][p], flags[1][p]
        hidden.live_count = len(live)
        hidden.live_wall = (C.c_int32 * 122)(*live)
        hidden.dead_wall = (C.c_int32 * 14)(*dead)
        hidden.seed = seed
        _check(self._lib, self._lib.rq_redeal(self._handle, C.byref(hidden)))
        self._refresh()

    def clone(self):
        source = self._require_open()
        work = object.__new__(type(self))
        work._handle, work._lib = None, self._lib
        work._handle = self._lib.rq_clone(source)
        if not work._handle:
            _check(self._lib, -1)
        try:
            work.rules = self.rules
            work.moves = self.history
            work.initial = {}
            work._last_hand = None
            work._refresh()
            return work
        except BaseException:
            work.close()
            raise

    def close(self) -> None:
        handle = getattr(self, '_handle', None)
        if handle:
            self._handle = None
            self._lib.rq_destroy(handle)

    def __del__(self):
        self.close()

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *exc):
        self.close()

    @classmethod
    def replay(cls, seed: int, actions: Iterable[dict[int, int]], rules: RiichiRules | None = None):
        game = cls(seed=seed, rules=rules)
        try:
            for joint in actions:
                game.play_joint(joint)
            return game
        except BaseException:
            game.close()
            raise
