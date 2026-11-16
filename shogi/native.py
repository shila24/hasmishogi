"""Dependency-free ctypes bridge to the independent C++ shogi engine."""
from __future__ import annotations

import ctypes as C
from functools import lru_cache
import os
from pathlib import Path
import sys
import weakref

from .model import (DECLARE, K, REASONS, START_SFEN, MoveResult, Outcome,
                    Position, Rules, require_int)
from .notation import parse_sfen


class sg_rules(C.Structure):
    _fields_ = [(name, C.c_int32) for name in ("max_moves", "declaration")]


class sg_state(C.Structure):
    _fields_ = [(name, C.c_int32) for name in (
        "turn", "ply", "move_number", "winner", "reason", "in_check", "can_declare")]


def find_library() -> Path:
    explicit = os.environ.get("SHOGI_LIB")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"SHOGI_LIB does not exist: {path}")
        return path
    root = Path(__file__).resolve().parent.parent
    filename = ("shogi_native.dll" if sys.platform == "win32" else
                "libshogi_native.dylib" if sys.platform == "darwin" else "libshogi_native.so")
    for directory in (root / "build", root / "build/Release", root / "build/Debug"):
        path = directory / filename
        if path.is_file():
            return path
    raise RuntimeError("Native shogi library not found. Run: python3 scripts/build_native.py")


@lru_cache(maxsize=4)
def _load_path(path: str):
    try:
        lib = C.CDLL(path)
        integer, handle = C.c_int32, C.c_void_p
        ints, chars, floats = C.POINTER(integer), C.POINTER(C.c_char), C.POINTER(C.c_float)
        signatures = {
            "sg_abi_version": ([], integer),
            "sg_sizeof_rules": ([], C.c_size_t),
            "sg_sizeof_state": ([], C.c_size_t),
            "sg_last_error": ([], C.c_char_p),
            "sg_default_rules": ([C.POINTER(sg_rules)], integer),
            "sg_create": ([C.POINTER(sg_rules), C.c_char_p], handle),
            "sg_clone": ([handle], handle),
            "sg_destroy": ([handle], None),
            "sg_get_state": ([handle, C.POINTER(sg_state)], integer),
            "sg_sfen": ([handle, chars, integer], integer),
            "sg_history_key": ([handle, integer, chars, integer], integer),
            "sg_is_legal": ([handle, integer], integer),
            "sg_move_label": ([handle, integer], integer),
            "sg_label_to_move": ([handle, integer], integer),
            "sg_play": ([handle, integer, ints, ints], integer),
            "sg_undo": ([handle], integer),
            "sg_resign": ([handle], integer),
            "sg_encode": ([handle, integer, floats, integer], integer),
            "sg_legal_mask": ([handle, integer, floats, integer], integer),
            "sg_perft": ([handle, integer], C.c_int64),
        }
        for name in ("sg_board", "sg_hands", "sg_moves", "sg_legal_moves", "sg_legal_labels"):
            signatures[name] = ([handle, ints, integer], integer)
        for name, (args, result) in signatures.items():
            function = getattr(lib, name)
            function.argtypes, function.restype = args, result
        if lib.sg_abi_version() != 1:
            raise RuntimeError("Native shogi ABI version mismatch")
        for structure in (sg_rules, sg_state):
            if getattr(lib, "sg_sizeof_" + structure.__name__[3:])() != C.sizeof(structure):
                raise RuntimeError(f"Native shogi {structure.__name__} size mismatch")
        return lib
    except (OSError, AttributeError) as exc:
        raise RuntimeError("Unable to load the native shogi ABI") from exc


def available() -> bool:
    try:
        _load_path(str(find_library()))
        return True
    except RuntimeError:
        return False


class NativeShogiGame:
    """Owned native handle; cached immutable snapshots never implement game transitions."""
    backend = "cpp"

    def __init__(self, rules: Rules | None = None, sfen: str | None = None):
        self._handle = None
        self._rules = rules if rules is not None else Rules()
        if not isinstance(self._rules, Rules):
            raise ValueError("Expected Rules")
        # Python text validation also prevents embedded NUL truncation in the C ABI.
        self._initial = parse_sfen(START_SFEN if sfen is None else sfen)
        self._lib = _load_path(str(find_library()))
        packed = sg_rules(self._rules.max_moves, int(self._rules.declaration == "csa27"))
        handle = self._lib.sg_create(C.byref(packed), self._initial.sfen().encode("ascii"))
        if not handle:
            self._error()
        self._adopt(handle)
        self.comments: list[str] = []

    def _adopt(self, handle):
        self._handle = handle
        try:
            self._finalizer = weakref.finalize(self, self._lib.sg_destroy, handle)
        except BaseException:
            self._lib.sg_destroy(handle)
            self._handle = None
            raise
        self._invalidate()

    def _invalidate(self):
        self._state_cache = None
        self._position_cache = None
        self._legal_cache = None
        self._labels_cache = None
        self._keys_cache = None
        self._moves_cache = None

    def _ensure_open(self):
        if self._handle is None:
            raise ValueError("Game is closed")

    def _error(self):
        message = self._lib.sg_last_error()
        raise ValueError(message.decode("utf-8") if message else "Native shogi error")

    def _check(self, status):
        if status < 0:
            self._error()
        return status

    def _state(self):
        self._ensure_open()
        if self._state_cache is None:
            value = sg_state()
            self._check(self._lib.sg_get_state(self._handle, C.byref(value)))
            self._state_cache = value
        return self._state_cache

    def _integers(self, function, size=None):
        self._ensure_open()
        if size is None:
            size = self._check(function(self._handle, None, 0))
        buffer = (C.c_int32 * size)()
        count = self._check(function(self._handle, buffer, size))
        return tuple(buffer[:count])

    def _string(self, function, *args):
        self._ensure_open()
        size = self._check(function(self._handle, *args, None, 0))
        buffer = C.create_string_buffer(size + 1)
        self._check(function(self._handle, *args, buffer, len(buffer)))
        return buffer.value.decode("ascii")

    @property
    def rules(self) -> Rules:
        self._ensure_open()
        return self._rules

    @property
    def initial(self) -> Position:
        self._ensure_open()
        return self._initial

    @property
    def position(self) -> Position:
        state = self._state()
        if self._position_cache is None:
            board = self._integers(self._lib.sg_board, 81)
            hands = self._integers(self._lib.sg_hands, 14)
            self._position_cache = Position(board, (hands[:7], hands[7:]), state.turn, state.move_number)
        return self._position_cache

    @property
    def outcome(self) -> Outcome:
        state = self._state()
        return Outcome(state.winner, REASONS[state.reason])

    @property
    def ply(self) -> int:
        return self._state().ply

    @property
    def moves(self) -> list[int]:
        self._ensure_open()
        if self._moves_cache is None:
            self._moves_cache = self._integers(self._lib.sg_moves)
        return list(self._moves_cache)

    def legal_actions(self) -> list[int]:
        self._ensure_open()
        if self._legal_cache is None:
            self._legal_cache = self._integers(self._lib.sg_legal_moves)
        return list(self._legal_cache)

    def legal_labels(self) -> list[int]:
        self._ensure_open()
        if self._labels_cache is None:
            self._labels_cache = self._integers(self._lib.sg_legal_labels)
        return list(self._labels_cache)

    def is_legal(self, code: int) -> bool:
        self._ensure_open()
        if type(code) is not int or not 0 <= code <= DECLARE:
            return False
        return bool(self._check(self._lib.sg_is_legal(self._handle, code)))

    def play(self, code: int) -> MoveResult:
        self._ensure_open()
        # Preserve error precedence, and never allow ctypes integer wrapping.
        if type(code) is not int or not 0 <= code <= DECLARE:
            raise ValueError("Game is over" if self._state().winner != -2 else "Illegal move")
        captured, check = C.c_int32(), C.c_int32()
        self._check(self._lib.sg_play(self._handle, code, C.byref(captured), C.byref(check)))
        keys, moves = self._keys_cache, self._moves_cache
        self._invalidate()
        if keys is not None:
            self._keys_cache = keys + (self._string(self._lib.sg_history_key, len(keys)),)
        if moves is not None:
            self._moves_cache = moves + (code,)
        return MoveResult(code, captured.value, bool(check.value))

    def undo(self) -> bool:
        self._ensure_open()
        result = bool(self._check(self._lib.sg_undo(self._handle)))
        self._invalidate()
        return result

    def resign(self) -> None:
        self._ensure_open()
        self._check(self._lib.sg_resign(self._handle))
        self._invalidate()

    def clone(self) -> NativeShogiGame:
        self._ensure_open()
        other = object.__new__(type(self))
        other._handle = None
        other._rules, other._initial, other._lib = self._rules, self._initial, self._lib
        handle = self._lib.sg_clone(self._handle)
        if not handle:
            self._error()
        other._adopt(handle)
        other.comments = self.comments.copy()
        return other

    def history_keys(self) -> list[str]:
        self._ensure_open()
        if self._keys_cache is None:
            self._keys_cache = tuple(self._string(self._lib.sg_history_key, i) for i in range(self.ply + 1))
        return list(self._keys_cache)

    def in_check(self, player=None) -> bool:
        state = self._state()
        side = state.turn if player is None else require_int(player, "player", 0, 1)
        if side == state.turn:
            return bool(state.in_check)
        # The ABI exposes only the current side. This read-only query uses the
        # public board helper, never a second game or a second transition path.
        from .mechanics import attacked
        board = self.position.board
        return attacked(board, board.index(K + 16 * side), 1 - side)

    def checkers(self) -> list[int]:
        from .mechanics import attackers
        position = self.position
        return sorted(attackers(position.board, position.board.index(K + 16 * position.turn), 1 - position.turn))

    def can_declare(self) -> bool:
        return bool(self._state().can_declare)

    def sfen(self) -> str:
        return self._string(self._lib.sg_sfen)

    def _into(self, function, buffer, size):
        self._ensure_open()
        try:
            view = memoryview(buffer)
        except TypeError as exc:
            raise ValueError("Expected a writable C-contiguous float32 buffer") from exc
        if (view.readonly or not view.c_contiguous or view.itemsize != 4
                or view.format not in ("f", "@f", "=f", "<f" if sys.byteorder == "little" else ">f")
                or view.nbytes != size * 4):
            raise ValueError("Expected a writable C-contiguous float32 buffer of the required size")
        target = C.c_float.from_buffer(view)
        if C.addressof(target) % C.alignment(C.c_float):
            raise ValueError("Expected an aligned float32 buffer")
        self._check(function(self._handle, self._state().turn, C.byref(target), size))

    def encode_into(self, buffer) -> None:
        self._into(self._lib.sg_encode, buffer, 49 * 81)

    def legal_mask_into(self, buffer) -> None:
        self._into(self._lib.sg_legal_mask, buffer, 2188)

    def perft(self, depth: int) -> int:
        self._ensure_open()
        require_int(depth, "depth", 0, 2**31 - 1)
        return self._check(self._lib.sg_perft(self._handle, depth))

    def close(self) -> None:
        if self._handle is not None:
            self._finalizer()
            self._handle = None
            self._invalidate()

    @classmethod
    def replay(cls, rules, sfen, codes):
        game = cls(rules, sfen)
        try:
            for code in codes:
                game.play(code)
        except BaseException:
            game.close()
            raise
        return game
