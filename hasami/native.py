"""ctypes bridge to the C++ engine. No native rules are reimplemented here."""
from __future__ import annotations
import ctypes as C
import os
import sys
from pathlib import Path
from .model import Rules, Position, Outcome, MoveResult, REASONS, decode_action


class CPosition(C.Structure):
    _fields_ = [("black_lo", C.c_uint64), ("black_hi", C.c_uint64),
                ("white_lo", C.c_uint64), ("white_hi", C.c_uint64),
                ("turn", C.c_int32), ("pending", C.c_int32)]


class CRules(C.Structure):
    _fields_ = [(x, C.c_int32) for x in ("edge_capture", "capture_target", "lead_target",
                                        "repetition_limit", "no_moves_draw")]


class CState(C.Structure):
    _fields_ = [("position", CPosition), ("winner", C.c_int32), ("reason", C.c_int32),
                ("ply", C.c_int32), ("captured_lo", C.c_uint64), ("captured_hi", C.c_uint64)]


class CSearch(C.Structure):
    _fields_ = [("action", C.c_int32), ("score", C.c_int32), ("depth", C.c_int32),
                ("completed", C.c_int32), ("nodes", C.c_uint64), ("elapsed_ms", C.c_double)]


def pack_position(p: Position) -> CPosition:
    mask = (1 << 64) - 1
    return CPosition(p.black & mask, p.black >> 64, p.white & mask, p.white >> 64,
                     p.turn, p.pending)


def unpack_position(p: CPosition) -> Position:
    return Position(p.black_lo | (p.black_hi << 64), p.white_lo | (p.white_hi << 64),
                    p.turn, p.pending)


def find_library() -> Path:
    explicit = os.environ.get("HASAMI_LIB")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"HASAMI_LIB does not exist: {path}")
        return path
    root = Path(__file__).resolve().parent.parent
    filename = ("hasami_native.dll" if sys.platform == "win32" else
                "libhasami_native.dylib" if sys.platform == "darwin" else "libhasami_native.so")
    for directory in (root / "build", root / "build" / "Release", root / "build" / "Debug"):
        path = directory / filename
        if path.is_file():
            return path
    raise RuntimeError("Native library not found. Run: python scripts/build_native.py")


def load_library():
    lib = C.CDLL(str(find_library()))
    signatures = {
        "hs_abi_version": ([], C.c_int32),
        "hs_last_error": ([], C.c_char_p),
        "hs_create": ([C.POINTER(CRules), C.POINTER(CPosition)], C.c_void_p),
        "hs_destroy": ([C.c_void_p], None),
        "hs_clone": ([C.c_void_p], C.c_void_p),
        "hs_get_state": ([C.c_void_p, C.POINTER(CState)], C.c_int32),
        "hs_legal_actions": ([C.c_void_p, C.POINTER(C.c_int32), C.c_int32], C.c_int32),
        "hs_history": ([C.c_void_p, C.POINTER(CPosition), C.c_int32], C.c_int32),
        "hs_play": ([C.c_void_p, C.c_int32], C.c_int32),
        "hs_undo": ([C.c_void_p], C.c_int32),
        "hs_search": ([C.c_void_p, C.c_int32, C.c_int32, C.c_uint64,
                       C.POINTER(CSearch)], C.c_int32),
        "hs_perft": ([C.c_void_p, C.c_int32, C.POINTER(C.c_uint64)], C.c_int32),
    }
    for name, (args, result) in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, result
    if lib.hs_abi_version() != 1:
        raise RuntimeError("Native ABI version mismatch; rebuild the library")
    for name, struct in (("position", CPosition), ("rules", CRules),
                         ("state", CState), ("search_result", CSearch)):
        fn = getattr(lib, "hs_sizeof_" + name)
        fn.argtypes, fn.restype = [], C.c_int32
        if fn() != C.sizeof(struct):
            raise RuntimeError(f"Native ABI layout mismatch: {name}")
    return lib


class NativeGame:
    backend = "cpp"

    def __init__(self, rules: Rules | None = None, position: Position | None = None):
        self.rules = rules or Rules()
        self.initial = position or Position()
        self.moves: list[int] = []
        self._handle = None
        self._lib = load_library()
        r = CRules(int(self.rules.edge_capture), self.rules.capture_target,
                   self.rules.lead_target, self.rules.repetition_limit, int(self.rules.no_moves_draw))
        p = pack_position(self.initial)
        self._handle = self._lib.hs_create(C.byref(r), C.byref(p))
        if not self._handle:
            self._error()

    def _error(self):
        msg = self._lib.hs_last_error()
        raise ValueError(msg.decode("utf-8", "replace") if msg else "Unknown native error")

    def _check(self, status):
        if status < 0:
            self._error()
        return status

    def _open(self):
        if not self._handle:
            raise RuntimeError("Native game is closed")
        return self._handle

    def _state(self) -> CState:
        out = CState()
        self._check(self._lib.hs_get_state(self._open(), C.byref(out)))
        return out

    @property
    def position(self) -> Position:
        return unpack_position(self._state().position)

    @property
    def outcome(self) -> Outcome:
        s = self._state()
        return Outcome(s.winner, REASONS[s.reason])

    @property
    def last_capture(self) -> int:
        s = self._state()
        return s.captured_lo | (s.captured_hi << 64)

    def legal_actions(self) -> list[int]:
        buffer = (C.c_int32 * 144)()
        count = self._check(self._lib.hs_legal_actions(self._open(), buffer, 144))
        return list(buffer[:count])

    def play(self, action: int) -> MoveResult:
        decode_action(action)
        self._check(self._lib.hs_play(self._open(), action))
        self.moves.append(action)
        s = self._state()
        return MoveResult(action, s.captured_lo | (s.captured_hi << 64),
                          Outcome(s.winner, REASONS[s.reason]))

    def undo(self) -> bool:
        done = self._check(self._lib.hs_undo(self._open()))
        if done:
            self.moves.pop()
        return bool(done)

    def clone(self) -> NativeGame:
        other = object.__new__(NativeGame)
        other._handle = None
        other._lib = self._lib
        other.rules, other.initial = self.rules, self.initial
        other.moves = self.moves.copy()
        other._handle = self._lib.hs_clone(self._open())
        if not other._handle:
            self._error()
        return other

    def history_keys(self) -> list[tuple[int, int, int, int]]:
        size = self._check(self._lib.hs_history(self._open(), None, 0))
        out = (CPosition * size)()
        self._check(self._lib.hs_history(self._open(), out, size))
        return [unpack_position(p).key() for p in out]

    def search(self, depth=3, time_ms=1000, node_limit=200000) -> dict:
        from .agents import SearchLimits
        SearchLimits(depth, time_ms, node_limit)
        out = CSearch()
        self._check(self._lib.hs_search(self._open(), depth, time_ms, node_limit, C.byref(out)))
        return dict(action=out.action, score=out.score, depth=out.depth, nodes=out.nodes,
                    elapsed_ms=out.elapsed_ms, completed=bool(out.completed))

    def perft(self, depth: int) -> int:
        from .model import require_int
        require_int(depth, "depth")
        if not 0 <= depth <= 6:
            raise ValueError("Native perft depth must be in [0, 6]")
        out = C.c_uint64()
        self._check(self._lib.hs_perft(self._open(), depth, C.byref(out)))
        return out.value

    def close(self):
        if self._handle:
            self._lib.hs_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if getattr(self, "_handle", None):
            self.close()
