"""C++ PUCT with one Python evaluator and NumPy-owned Dirichlet draws.

Only the unmodified Hasami/Shogi adapters and v1 encoders are supported. Full
initial state, rules, moves and resignation are replayed into owned native
engines; no opaque handle layout or cross-library C++ object ABI is assumed.
Every search owns fresh trees and joins its workers even on evaluator failure.

Cancellation/deadlines use the reference's checks before each simulation path.
When either is supplied, descent runs one tree at a time; completed leaves are
still evaluated in the same batches before backup. Root evaluation always
finishes. Expiring real-time deadlines cannot produce identical simulation
counts across implementations with different runtimes; pre-expired deadlines,
pre-set cancellation, and cancellation at matching check boundaries do agree.

The standard adapters never report truncation: shogi max_moves is a terminal
rule draw and selfplay.max_plies is a runner cap outside both searches. Native
search does not invent a different cap or convert rule draws to truncations.
"""

from __future__ import annotations

import ctypes as C
from contextlib import contextmanager
from functools import lru_cache
import os
from pathlib import Path
import sys

import numpy as np

from ..games import get_spec
from ..games.hasami import HasamiSpec, PlaneEncoder as HasamiEncoder
from ..games.shogi import ShogiSpec, PlaneEncoder as ShogiEncoder
from .mcts import MCTS, MctsResult, _integer_budget
from .draw import draw_reasons_for


def find_library() -> Path:
    explicit = os.environ.get("MCTS_LIB")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"MCTS_LIB does not exist: {path}")
        return path
    root = Path(__file__).resolve().parents[2]
    name = ("mcts_native.dll" if sys.platform == "win32" else
            "libmcts_native.dylib" if sys.platform == "darwin" else "libmcts_native.so")
    for directory in (root / "build", root / "build/Release", root / "build/Debug"):
        path = directory / name
        if path.is_file():
            return path
    raise RuntimeError("Native MCTS library not found; run scripts/build_native.py")


@lru_cache(maxsize=4)
def _load_path(path: str):
    try:
        lib = C.CDLL(path)
        p, n, i, d = C.c_void_p, C.c_size_t, C.c_int32, C.c_double
        signatures = {
            "mz_abi_version": ([], i),
            "mz_last_error": ([p, n], i),
            "mz_create": ([i, i, d, d], p),
            "mz_draw_policy": ([p, d, C.c_uint32], i),
            "mz_destroy": ([p], None),
            "mz_add_hasami": ([p, p, n, p, n, p, n], i),
            "mz_add_shogi": ([p, C.c_char_p, n, p, n, p, n, i], i),
            "mz_prepare": ([p, p, n, i, p, n, p, n, p, n], i),
            "mz_expand": ([p, p, n, p, n, p, n], i),
            "mz_noise": ([p, i, p, n, d], i),
            "mz_gumbel": ([p, i, p, n, C.c_int64, C.c_int64, d, d], i),
            "mz_policy": ([p, p, n, p, n], i),
            "mz_backup": ([p, p, n], i),
            "mz_observe": ([p, p, n, p, n], i),
            "mz_results": ([p, p, n, p, n, p, n, p, n], i),
        }
        for name, (args, result) in signatures.items():
            function = getattr(lib, name)
            function.argtypes, function.restype = args, result
        if lib.mz_abi_version() != 3:
            raise RuntimeError("Native MCTS ABI version mismatch; rebuild the library")
        return lib
    except (OSError, AttributeError) as exc:
        raise RuntimeError("Unable to load the native MCTS ABI; rebuild the library") from exc


def _ptr(array):
    return array.ctypes.data_as(C.c_void_p)


def _error(lib):
    size = lib.mz_last_error(None, 0)
    message = C.create_string_buffer(size + 1)
    lib.mz_last_error(message, len(message))
    raise ValueError(message.value.decode("utf-8", "replace") or "Native MCTS error")


class NativeMCTS:
    def __init__(self, evaluate_fn, encoder, config, rng, spec=None, threads=None):
        self.spec = spec if spec is not None else get_spec()
        if not self.supports(self.spec, encoder, None):
            raise ValueError("Native MCTS requires an unmodified Hasami/Shogi spec and its v1 encoder")
        if threads is not None:
            threads = _integer_budget(threads, "search_threads", minimum=1)
            if threads > 2**31 - 1:
                raise ValueError("search_threads exceeds the native ABI limit")
        self.evaluate_fn, self.encoder, self.config, self.rng = evaluate_fn, encoder, config, rng
        self.threads = threads
        self._lib = _load_path(str(find_library()))

    @classmethod
    def supports(cls, spec, encoder, game):
        """Refuse custom adapter/encoder behavior and non-native/closed games."""
        from hasami.native import NativeGame
        from shogi.native import NativeShogiGame

        expected = {HasamiSpec: (HasamiEncoder, NativeGame), ShogiSpec: (ShogiEncoder, NativeShogiGame)}
        classes = expected.get(type(spec))
        if classes is None or type(encoder) is not classes[0] or vars(encoder):
            return False
        allowed_fields = {"rules"} if type(spec) is ShogiSpec else set()
        if set(vars(spec)) - allowed_fields:
            return False
        return game is None or (type(game) is classes[1] and bool(game._handle))

    def _check(self, result):
        if result < 0:
            _error(self._lib)
        return result

    @contextmanager
    def _session(self, games):
        for game in games:
            if not self.supports(self.spec, self.encoder, game):
                raise ValueError("Native MCTS requires open backend='cpp' games matching its supported spec/encoder")
        threads = min(len(games) or 1, self.threads or min(8, os.cpu_count() or 1))
        handle = self._lib.mz_create(int(type(self.spec) is ShogiSpec), threads,
                                     self.config.c_puct, self.config.truncation_value)
        if not handle:
            _error(self._lib)
        try:
            reasons = draw_reasons_for(self.spec, self.config.draw_reasons)
            reason_bits = int("repetition" in reasons) | (int("max_moves" in reasons) << 1)
            self._check(self._lib.mz_draw_policy(handle, self.config.draw_value, reason_bits))
            for game in games:
                moves = np.ascontiguousarray(game.moves, dtype=np.int32)
                rules, initial = game.rules, game.initial
                if type(self.spec) is HasamiSpec:
                    low_mask = (1 << 64) - 1
                    boards = np.array([initial.black & low_mask, initial.black >> 64,
                                       initial.white & low_mask, initial.white >> 64], dtype=np.uint64)
                    settings = np.array([initial.turn, initial.pending, rules.edge_capture, rules.capture_target,
                                         rules.lead_target, rules.repetition_limit, rules.no_moves_draw], dtype=np.int32)
                    self._check(self._lib.mz_add_hasami(handle, _ptr(boards), boards.size,
                                                       _ptr(settings), settings.size, _ptr(moves), moves.size))
                else:
                    sfen = initial.sfen().encode("ascii")
                    settings = np.array([rules.max_moves, rules.declaration == "csa27"], dtype=np.int32)
                    self._check(self._lib.mz_add_shogi(handle, sfen, len(sfen), _ptr(settings), settings.size,
                                                      _ptr(moves), moves.size, int(game.outcome.reason == "resign")))
            yield handle
        finally:
            self._lib.mz_destroy(handle)

    def _buffers(self, count):
        return (np.empty((count, self.encoder.num_planes, *self.spec.board_shape), dtype=np.float32),
                np.empty((count, self.spec.action_size), dtype=np.float32), np.empty(count, dtype=np.int32))

    def encode_many(self, games):
        """Expose native observation/mask snapshots for parity tests and diagnostics."""
        with self._session(games) as handle:
            planes, masks, _ = self._buffers(len(games))
            self._check(self._lib.mz_observe(handle, _ptr(planes), planes.size, _ptr(masks), masks.size))
            return planes, masks

    def _prepare(self, handle, ids, descend, planes, masks, leaves):
        return self._check(self._lib.mz_prepare(handle, _ptr(ids), len(ids), int(descend),
                                               _ptr(planes), planes.size, _ptr(masks), masks.size,
                                               _ptr(leaves), leaves.size))

    def _expand_many(self, handle, ids, planes, masks):
        for begin in range(0, len(ids), self.config.batch_size):
            batch = slice(begin, begin + self.config.batch_size)
            # A fresh batch matches np.stack in the reference, including evaluators
            # that retain or mutate their arguments. Native masks remain immutable.
            probs, values = self.evaluate_fn(planes[batch].copy(), masks[batch].copy())
            probs, values = np.asarray(probs), np.asarray(values)
            if probs.shape != masks[batch].shape or values.shape != (len(ids[batch]),):
                raise ValueError("Evaluator output must have shapes (B,action_size) and (B,)")
            probs = np.ascontiguousarray(probs, dtype=np.float64)
            values = np.ascontiguousarray(values, dtype=np.float64)
            self._check(self._lib.mz_expand(handle, _ptr(ids[batch]), len(ids[batch]),
                                           _ptr(probs), probs.size, _ptr(values), values.size))

    def run(self, game, simulations=None, cancel=None, deadline=None):
        return self.run_many([game], simulations, cancel, deadline)[0]

    def run_many(self, games, simulations=None, cancel=None, deadline=None):
        budget = _integer_budget(self.config.simulations if simulations is None else simulations, "simulations")
        if budget > 2**63 - 1 or self.config.gumbel_m > 2**63 - 1:
            raise ValueError("Gumbel/sample budget exceeds the native ABI limit")
        with self._session(games) as handle:
            planes, masks, leaves = self._buffers(len(games))
            all_ids = np.arange(len(games), dtype=np.int32)
            count = self._prepare(handle, all_ids, False, planes, masks, leaves)
            active = leaves[:count].copy()
            self._expand_many(handle, active, planes[:count], masks[:count])
            if self.config.root_selection == "gumbel":
                for row, index in enumerate(active):
                    size = np.count_nonzero(masks[row] > 0)
                    noise = self.rng.gumbel(size=size) if self.config.add_noise else np.zeros(size, np.float64)
                    self._check(self._lib.mz_gumbel(handle, int(index), _ptr(noise), size, budget,
                                                   self.config.gumbel_m, self.config.gumbel_c_visit,
                                                   self.config.gumbel_c_scale))
            elif self.config.add_noise:
                for row, index in enumerate(active):
                    noise = self.rng.dirichlet(np.full(np.count_nonzero(masks[row] > 0), self.config.dirichlet_alpha))
                    self._check(self._lib.mz_noise(handle, int(index), _ptr(noise), len(noise),
                                                  self.config.dirichlet_epsilon))
            for _ in range(budget if len(active) else 0):
                if cancel is None and deadline is None:
                    pending = active
                    count = self._prepare(handle, pending, True, planes, masks, leaves)
                else:
                    count = 0
                    completed = 0
                    for offset in range(len(active)):
                        if MCTS._stopped(cancel, deadline):
                            break
                        count += self._prepare(handle, active[offset:offset + 1], True,
                                               planes[count:count + 1], masks[count:count + 1], leaves[count:count + 1])
                        completed += 1
                    pending = active[:completed]
                if not len(pending):
                    break
                self._expand_many(handle, leaves[:count], planes[:count], masks[:count])
                self._check(self._lib.mz_backup(handle, _ptr(pending), len(pending)))
            visits = np.empty((len(games), self.spec.action_size), np.float32)
            q_values = np.empty_like(visits)
            values = np.empty(len(games), np.float64)
            completed = np.empty(len(games), np.int64)
            self._check(self._lib.mz_results(handle, _ptr(visits), visits.size, _ptr(q_values), q_values.size,
                                            _ptr(values), values.size, _ptr(completed), completed.size))
            selected = np.full(len(games), -1, np.int32)
            if self.config.root_selection == "gumbel":
                policies = np.empty(visits.shape, np.float64)
                self._check(self._lib.mz_policy(handle, _ptr(policies), policies.size,
                                               _ptr(selected), selected.size))
            else:
                policies = [v / v.sum() if v.sum() else v.copy() for v in visits]
            return [MctsResult(visits[i], float(values[i]), int(completed[i]), q_values[i], policies[i],
                               int(selected[i]) if selected[i] >= 0 else None) for i in range(len(games))]
