"""Standard-library bridge to the project's MIT C++ alpha-beta searcher.

The input is an initial SFEN plus the complete legal move history. Search owns
its position and never changes the supplied game. Independent instances may run
on different threads; calls on one instance are serialized except atomic stop.
"""
from __future__ import annotations

import ctypes as C
from functools import lru_cache
import os
from pathlib import Path
import sys
import threading
import weakref

from hasami.agents import Decision, SearchLimits, check_cancel
from .model import DECLARE, REASONS, require_int

FEATURES = {name: 1 << index for index, name in enumerate((
    "pvs", "aspiration", "transposition", "quiescence", "null_move", "lmr",
    "futility", "razoring", "check_extension", "see_pruning", "ordering"))}
ALL_FEATURES = sum(FEATURES.values())


class ss_options(C.Structure):
    _fields_ = [("features", C.c_uint32), ("hash_mb", C.c_int32), ("qdepth", C.c_int32)]


class ss_limits(C.Structure):
    _fields_ = [("nodes", C.c_uint64)] + [(n, C.c_int32) for n in
                ("depth", "time_ms", "soft_time_ms", "multipv")]


class ss_result(C.Structure):
    _fields_ = [("nodes", C.c_uint64), ("elapsed_ms", C.c_double)] + [(n, C.c_int32) for n in
                ("move", "score", "depth", "completed", "stopped", "root_count", "mate_plies")]


ss_progress = C.CFUNCTYPE(C.c_int32, C.POINTER(ss_result), C.POINTER(C.c_int32), C.c_int32, C.c_void_p)


class ss_root_score(C.Structure):
    _fields_ = [(n, C.c_int32) for n in ("move", "score", "depth")]


class ss_state(C.Structure):
    _fields_ = [("key", C.c_uint64)] + [(n, C.c_int32) for n in ("turn", "ply", "winner", "reason")]


def find_library() -> Path:
    explicit = os.environ.get("SHOGI_SEARCH_LIB")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"SHOGI_SEARCH_LIB does not exist: {path}")
        return path
    root = Path(__file__).resolve().parent.parent
    filename = ("shogi_search_native.dll" if sys.platform == "win32" else
                "libshogi_search_native.dylib" if sys.platform == "darwin" else "libshogi_search_native.so")
    for directory in (root / "build", root / "build/Release", root / "build/Debug"):
        path = directory / filename
        if path.is_file():
            return path
    raise RuntimeError("Native shogi search library not found. Run: python3 scripts/build_native.py")


@lru_cache(maxsize=4)
def _load_path(path):
    try:
        lib = C.CDLL(path)
        integer, handle = C.c_int32, C.c_void_p
        ints = C.POINTER(integer)
        signatures = {
            "ss_abi_version": ([], integer), "ss_last_error": ([], C.c_char_p),
            "ss_create": ([C.POINTER(ss_options)], handle), "ss_destroy": ([handle], None),
            "ss_set_position": ([handle, C.c_char_p, integer, integer, ints, integer, integer], integer),
            "ss_get_state": ([handle, C.POINTER(ss_state)], integer),
            "ss_legal_moves": ([handle, ints, integer], integer),
            "ss_perft": ([handle, integer], C.c_int64),
            "ss_evaluate": ([handle, ints], integer), "ss_see": ([handle, integer, ints], integer),
            "ss_go": ([handle, C.POINTER(ss_limits), ints, integer, C.POINTER(ss_result)], integer),
            "ss_go_with_progress": ([handle, C.POINTER(ss_limits), ints, integer,
                                     C.POINTER(ss_result), ss_progress, handle], integer),
            "ss_root_scores": ([handle, C.POINTER(ss_root_score), integer], integer),
            "ss_pv": ([handle, integer, ints, integer], integer),
            "ss_load_nnue": ([handle, C.c_char_p], integer),
            "ss_evaluator": ([handle], C.c_char_p),
            "ss_evaluate_float": ([handle, C.POINTER(C.c_float)], integer),
            "ss_nnue_features": ([handle, integer, ints, integer], integer),
            "ss_save_nnue": ([handle, C.c_char_p], integer),
            "ss_data_roundtrip": ([C.c_char_p, C.c_char_p], integer),
            "ss_stop": ([handle], integer), "ss_clear_stop": ([handle], integer),
        }
        structures = (ss_options, ss_limits, ss_result, ss_root_score, ss_state)
        for structure in structures:
            signatures["ss_sizeof_" + structure.__name__[3:]] = ([], C.c_size_t)
        for name, (args, result) in signatures.items():
            function = getattr(lib, name)
            function.argtypes, function.restype = args, result
        if lib.ss_abi_version() != 1:
            raise RuntimeError("Native shogi search ABI version mismatch")
        for structure in structures:
            if getattr(lib, "ss_sizeof_" + structure.__name__[3:])() != C.sizeof(structure):
                raise RuntimeError(f"Native shogi search {structure.__name__} size mismatch")
        return lib
    except (OSError, AttributeError) as exc:
        raise RuntimeError("Unable to load the native shogi search ABI") from exc


def available():
    try:
        _load_path(str(find_library()))
        return True
    except RuntimeError:
        return False


def _path_bytes(path):
    if path is None:
        return None
    value = os.fsencode(Path(path).expanduser())
    if b"\0" in value:
        raise ValueError("NUL in NNUE path")
    return value


def _codes(values):
    values = list(values)
    for code in values:
        require_int(code, "move", 0, DECLARE)
    return (C.c_int32 * len(values))(*values)


def _result_data(value, pv):
    return {"move": value.move, "score": value.score, "depth": value.depth,
            "nodes": value.nodes, "elapsed_ms": value.elapsed_ms,
            "completed": bool(value.completed), "stopped": bool(value.stopped),
            "mate": value.mate_plies or None, "mate_plies": value.mate_plies or None, "pv": pv}


class NativeSearch:
    def __init__(self, game=None, *, features=ALL_FEATURES, hash_mb=16, qdepth=12, nnue=None):
        require_int(features, "features", 0, ALL_FEATURES)
        require_int(hash_mb, "hash_mb", 1, 512)
        require_int(qdepth, "qdepth", 0, 32)
        self._lock = threading.RLock()
        self._life_lock = threading.RLock()
        self._handle = None
        self._lib = _load_path(str(find_library()))
        options = ss_options(features, hash_mb, qdepth)
        handle = self._lib.ss_create(C.byref(options))
        if not handle:
            self._error()
        self._handle = handle
        try:
            self._finalizer = weakref.finalize(self, self._lib.ss_destroy, handle)
        except BaseException:
            self._lib.ss_destroy(handle)
            self._handle = None
            raise
        try:
            if nnue is not None:
                self.load_nnue(nnue)
            if game is not None:
                self.set_position(game)
        except BaseException:
            self.close()
            raise

    def _error(self):
        message = self._lib.ss_last_error()
        raise ValueError(message.decode("utf-8") if message else "Native shogi search error")

    def _check(self, status):
        if status < 0:
            self._error()
        return status

    def _open(self):
        if self._handle is None:
            raise ValueError("Search is closed")
        return self._handle

    def set_position(self, game):
        """Replay the complete history, including declaration/resignation outcomes."""
        with self._lock:
            handle = self._open()
            moves = _codes(game.moves)
            self._check(self._lib.ss_set_position(handle, game.initial.sfen().encode("ascii"),
                game.rules.max_moves, int(game.rules.declaration == "csa27"), moves, len(moves),
                int(game.outcome.reason == "resign")))

    def state(self):
        with self._lock:
            value = ss_state()
            self._check(self._lib.ss_get_state(self._open(), C.byref(value)))
            return {"key": value.key, "turn": value.turn, "ply": value.ply,
                    "winner": value.winner, "reason": REASONS[value.reason]}

    def legal_actions(self):
        with self._lock:
            handle = self._open()
            count = self._check(self._lib.ss_legal_moves(handle, None, 0))
            values = (C.c_int32 * count)()
            self._check(self._lib.ss_legal_moves(handle, values, count))
            return list(values)

    def perft(self, depth):
        require_int(depth, "depth", 0, 8)
        with self._lock:
            return self._check(self._lib.ss_perft(self._open(), depth))

    def evaluate(self):
        with self._lock:
            score = C.c_int32()
            self._check(self._lib.ss_evaluate(self._open(), C.byref(score)))
            return score.value

    def load_nnue(self, path=None):
        """Load verified hlnnw1 weights atomically; None restores bootstrap evaluation."""
        with self._lock:
            self._check(self._lib.ss_load_nnue(self._open(), _path_bytes(path)))

    @property
    def evaluator(self):
        with self._lock:
            name = self._lib.ss_evaluator(self._open())
            if name is None:
                self._error()
            return name.decode("ascii")

    def evaluate_float(self):
        with self._lock:
            score = C.c_float()
            self._check(self._lib.ss_evaluate_float(self._open(), C.byref(score)))
            return score.value

    def nnue_features(self, perspective):
        require_int(perspective, "perspective", 0, 1)
        with self._lock:
            values = (C.c_int32 * 39)()
            count = self._check(self._lib.ss_nnue_features(self._open(), perspective, values, 39))
            return list(values[:count])

    def save_nnue(self, path):
        with self._lock:
            self._check(self._lib.ss_save_nnue(self._open(), _path_bytes(path)))

    def see(self, move):
        require_int(move, "move", 0, DECLARE)
        with self._lock:
            score = C.c_int32()
            self._check(self._lib.ss_see(self._open(), move, C.byref(score)))
            return score.value

    def stop(self):
        with self._life_lock:
            self._check(self._lib.ss_stop(self._open()))

    def _pv(self, index):
        count = self._check(self._lib.ss_pv(self._handle, index, None, 0))
        values = (C.c_int32 * count)()
        self._check(self._lib.ss_pv(self._handle, index, values, count))
        return list(values)

    def go(self, *, depth=None, nodes=None, time_ms=None, soft_time_ms=None,
           multipv=1, root_moves=(), cancel=None, on_iteration=None):
        """Return plain data; multipv='all' gives full-window scores at one depth.

        Missing node/time budgets are unlimited. Nodes/time alone use depth 64.
        An incomplete iteration never replaces a completed result. Cancellation
        raises AgentCancelled; explicit stop returns the last completed result.
        on_iteration receives copied summary/PV data (no root_scores) after each
        completed iteration, on the calling thread. Only stop() may be called on
        this instance from the callback. Callback errors stop search and propagate.
        MultiPV and restricted roots use fresh heuristics/TT and a full window
        for each root at each depth, making completed per-root scores comparable.
        """
        depth = (64 if nodes is not None or time_ms is not None else 6) if depth is None else depth
        require_int(depth, "depth", 1, 64)
        for name, value, maximum in (("nodes", nodes, 2**64-1), ("time_ms", time_ms, 3600000),
                                     ("soft_time_ms", soft_time_ms, 3600000)):
            if value is not None:
                require_int(value, name, 1, maximum)
        if multipv == "all":
            multipv = 0
        require_int(multipv, "multipv", 0, 2048)
        moves = _codes(root_moves)
        limits = ss_limits(nodes or 0, depth, time_ms or 0, soft_time_ms or 0, multipv)
        with self._lock:
            handle = self._open()
            check_cancel(cancel)
            self._check(self._lib.ss_clear_stop(handle))
            done = threading.Event()
            watcher = None
            if cancel is not None:
                def watch():
                    while not done.wait(.002):
                        if cancel.is_set():
                            self.stop()
                            return
                watcher = threading.Thread(target=watch, name="shogi-search-cancel", daemon=True)
                watcher.start()
            try:
                value = ss_result()
                callback_errors = []
                if on_iteration is None:
                    status = self._lib.ss_go(handle, C.byref(limits), moves, len(moves), C.byref(value))
                else:
                    @ss_progress
                    def progress(summary, pv, count, _):
                        try:
                            on_iteration(_result_data(summary.contents, list(pv[:count])))
                            return 0
                        except BaseException as exc:
                            # Exceptions must never escape a ctypes callback into C++.
                            callback_errors.append(exc)
                            return 1
                    status = self._lib.ss_go_with_progress(
                        handle, C.byref(limits), moves, len(moves), C.byref(value), progress, None)
                self._check(status)
                if callback_errors:
                    raise callback_errors[0]
            finally:
                done.set()
                if watcher is not None:
                    watcher.join()
            check_cancel(cancel)
            rows = (ss_root_score * value.root_count)()
            self._check(self._lib.ss_root_scores(handle, rows, len(rows)))
            result = _result_data(value, self._pv(-1))
            result["root_scores"] = [{"move": row.move, "score": row.score, "depth": row.depth,
                                     "pv": self._pv(i)} for i, row in enumerate(rows)]
            return result

    def close(self):
        with self._lock, self._life_lock:
            if self._handle is not None:
                self._finalizer()
                self._handle = None

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *args):
        self.close()


class AlphaBetaCppAgent:
    """Legal, cancellable native search with no implicit Python fallback."""
    def __init__(self, **options):
        # Resolve eagerly so registry/USI selection fails explicitly if unbuilt.
        _load_path(str(find_library()))
        self.options = options
        self._search = NativeSearch(**options) if options.get("nnue") is not None else None
        self._lock = threading.RLock()

    def close(self):
        with self._lock:
            if self._search is not None:
                self._search.close()
                self._search = None

    def choose(self, game, limits: SearchLimits, cancel=None, *, on_iteration=None) -> Decision:
        check_cancel(cancel)
        work = game.clone()
        try:
            legal = work.legal_actions()
            if not legal:
                raise ValueError("No legal action is available")
            def decision(result):
                if type(result["move"]) is not int or result["move"] not in legal:
                    raise ValueError("Native search returned an illegal move")
                return Decision(result["move"], result["score"], result["depth"], result["nodes"],
                                result["elapsed_ms"], result["completed"],
                                {"pv": result["pv"], "mate": result["mate"]})
            progress = None if on_iteration is None else lambda result: on_iteration(decision(result))
            with self._lock:
                if self._search is None:
                    with NativeSearch(work, **self.options) as search:
                        result = search.go(depth=limits.max_depth, nodes=limits.node_limit,
                                           time_ms=limits.time_ms, cancel=cancel, on_iteration=progress)
                else:
                    self._search.set_position(work)
                    result = self._search.go(depth=limits.max_depth, nodes=limits.node_limit,
                                            time_ms=limits.time_ms, cancel=cancel, on_iteration=progress)
            check_cancel(cancel)
            return decision(result)
        finally:
            work.close()
