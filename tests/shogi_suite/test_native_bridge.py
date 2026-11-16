"""Native failure atomicity, ABI validation, buffers, and owned-handle lifecycle."""
from array import array
import ctypes as C
import gc
import os
import unittest
from unittest import mock
import weakref

from shogi import ShogiGame
from shogi.encoding import encode, legal_labels
from shogi.model import DECLARE, Rules, START_SFEN, encode_drop, encode_move
import shogi.native as bridge

NATIVE = bridge.available()
if not NATIVE and os.environ.get("HASAMI_REQUIRE_NATIVE") == "1":
    raise RuntimeError("Native shogi library is required")


def fresh_snapshot(game):
    """Read all state from a new native clone, never the source's Python caches."""
    clone = game.clone()
    try:
        planes = array("f", [0]) * 3969
        mask = array("f", [0]) * 2188
        clone.encode_into(planes)
        clone.legal_mask_into(mask)
        return (clone.position, clone.outcome, clone.ply, clone.moves,
                clone.history_keys(), clone.sfen(), clone.legal_actions(),
                clone.legal_labels(), clone.in_check(), clone.can_declare(),
                planes.tobytes(), mask.tobytes())
    finally:
        clone.close()


@unittest.skipUnless(NATIVE, "Native shogi library is unavailable")
class NativeBridgeTests(unittest.TestCase):
    def setUp(self):
        self.game = bridge.NativeShogiGame()
        self.addCleanup(self.game.close)
        self.lib = self.game._lib

    def assert_native_failure(self, function, args, message, unchanged_outputs=()):
        before = fresh_snapshot(self.game)
        outputs = [bytes(output) for output in unchanged_outputs]
        self.assertEqual(function(*args), -1)
        self.assertEqual(self.lib.sg_last_error().decode(), message)
        self.assertEqual([bytes(output) for output in unchanged_outputs], outputs)
        self.assertEqual(fresh_snapshot(self.game), before)

    def test_in_range_illegal_moves_reach_native_and_preserve_state(self):
        python = ShogiGame()
        # Same square, wrong side, jumping pawn, occupied target, unavailable drop/declaration.
        codes = [encode_move(54, 54), encode_move(0, 9), encode_move(54, 36),
                 encode_move(72, 54), encode_drop(0, 40), DECLARE]
        for code in codes:
            with self.subTest(code=code):
                before = fresh_snapshot(self.game)
                with mock.patch.object(self.lib, "sg_play", wraps=self.lib.sg_play) as play:
                    for game in (python, self.game):
                        with self.assertRaisesRegex(ValueError, "^Illegal move$"):
                            game.play(code)
                    play.assert_called_once()
                self.assertEqual(fresh_snapshot(self.game), before)
        self.game.play(self.game.legal_actions()[0])
        before = fresh_snapshot(self.game)
        with self.assertRaisesRegex(ValueError, "^Illegal move$"):
            self.game.play(0)
        self.assertEqual(fresh_snapshot(self.game), before)

    def test_terminal_calls_reach_native_and_preserve_state(self):
        for ending in ("resign", "max_moves", "checkmate", "declaration"):
            with self.subTest(ending=ending):
                sfen = {"checkmate": "k8/9/1GN6/1R7/9/9/9/9/4K4 b - 1",
                        "declaration": "GGGG1SSSS/RB7/4K4/9/9/9/9/9/4k4 b RB 1"}.get(ending)
                python = ShogiGame(Rules(1), sfen)
                game = bridge.NativeShogiGame(python.rules, sfen)
                try:
                    if ending == "resign":
                        python.resign()
                        game.resign()
                    else:
                        code = (DECLARE if ending == "declaration" else
                                encode_move(28, 27) if ending == "checkmate" else python.legal_actions()[0])
                        python.play(code)
                        game.play(code)
                    self.assertEqual(game.outcome.reason, ending)
                    before = fresh_snapshot(game)
                    with mock.patch.object(self.lib, "sg_play", wraps=self.lib.sg_play) as play:
                        for engine in (python, game):
                            with self.assertRaisesRegex(ValueError, "^Game is over$"):
                                engine.play(0)
                        play.assert_called_once()
                    for engine in (python, game):
                        with self.assertRaisesRegex(ValueError, "^Game is over$"):
                            engine.resign()
                    self.assertEqual(fresh_snapshot(game), before)
                finally:
                    game.close()

    def test_direct_abi_move_failures_and_null_outputs(self):
        captured, check = C.c_int32(91), C.c_int32(92)
        handle = self.game._handle
        for code in (-1, 0, DECLARE, DECLARE + 1, 2**31 - 1, -(2**31)):
            self.assert_native_failure(self.lib.sg_play,
                                       (handle, code, C.byref(captured), C.byref(check)),
                                       "Illegal move", (captured, check))
        code = self.game.legal_actions()[0]
        for outputs in ((None, C.byref(check)), (C.byref(captured), None), (None, None)):
            self.assert_native_failure(self.lib.sg_play, (handle, code, *outputs),
                                       "Null output pointer", (captured, check))
        self.assert_native_failure(self.lib.sg_get_state, (handle, None), "Null output pointer")
        self.assert_native_failure(self.lib.sg_history_key, (handle, -1, None, 0), "Invalid history index")
        self.assert_native_failure(self.lib.sg_history_key, (handle, 1, None, 0), "Invalid history index")
        self.assert_native_failure(self.lib.sg_perft, (handle, -1), "Invalid depth")
        self.assert_native_failure(self.lib.sg_move_label, (handle, 0), "Illegal move")
        for label in (-1, 2188):
            self.assert_native_failure(self.lib.sg_label_to_move, (handle, label), "Invalid policy label")
        self.assert_native_failure(self.lib.sg_label_to_move, (handle, 2187),
                                   "No legal move has this policy label")
        for code, label in zip(self.game.legal_actions(), self.game.legal_labels()):
            self.assertEqual(self.lib.sg_move_label(handle, code), label)
            self.assertEqual(self.lib.sg_label_to_move(handle, label), code)

    def test_direct_abi_buffers_queries_and_sentinels(self):
        self.game.play(self.game.legal_actions()[0])
        handle = self.game._handle
        functions = [(getattr(self.lib, name), (), C.c_int32) for name in
                     ("sg_board", "sg_hands", "sg_moves", "sg_legal_moves", "sg_legal_labels")]
        functions += [(self.lib.sg_sfen, (), C.c_char), (self.lib.sg_history_key, (1,), C.c_char),
                      (self.lib.sg_encode, (1,), C.c_float), (self.lib.sg_legal_mask, (1,), C.c_float)]
        for function, prefix, element in functions:
            with self.subTest(function=function.__name__):
                size = function(handle, *prefix, None, 0)
                self.assertGreater(size, 0)
                required = size + int(element is C.c_char)
                buffer = (element * (required + 1))()
                C.memset(buffer, 0x5A, C.sizeof(buffer))
                for capacity in (-1, 0, required - 1):
                    self.assert_native_failure(function, (handle, *prefix, buffer, capacity),
                                               "Output buffer too small", (buffer,))
                self.assert_native_failure(function, (handle, *prefix, None, required), "Null output pointer")
                tail = bytes(buffer)[required * C.sizeof(element):]
                self.assertEqual(function(handle, *prefix, buffer, required), size)
                self.assertEqual(bytes(buffer)[required * C.sizeof(element):], tail)
                if element is C.c_char:
                    self.assertEqual(buffer[size], b"\0")
        for function in (self.lib.sg_encode, self.lib.sg_legal_mask):
            for side, message in ((-1, "Invalid player"), (2, "Invalid player"),
                                  (0, "The encoder player must be the side to move")):
                self.assert_native_failure(function, (handle, side, None, 0), message)

    def test_null_handles_and_invalid_construction(self):
        state, captured = bridge.sg_state(), C.c_int32()
        calls = [(self.lib.sg_get_state, (None, C.byref(state))),
                 (self.lib.sg_board, (None, None, 0)), (self.lib.sg_hands, (None, None, 0)),
                 (self.lib.sg_moves, (None, None, 0)), (self.lib.sg_sfen, (None, None, 0)),
                 (self.lib.sg_history_key, (None, 0, None, 0)),
                 (self.lib.sg_legal_moves, (None, None, 0)), (self.lib.sg_legal_labels, (None, None, 0)),
                 (self.lib.sg_is_legal, (None, 0)), (self.lib.sg_move_label, (None, 0)),
                 (self.lib.sg_label_to_move, (None, 0)),
                 (self.lib.sg_play, (None, 0, C.byref(captured), C.byref(captured))),
                 (self.lib.sg_undo, (None,)), (self.lib.sg_resign, (None,)),
                 (self.lib.sg_encode, (None, 0, None, 0)),
                 (self.lib.sg_legal_mask, (None, 0, None, 0)), (self.lib.sg_perft, (None, 0))]
        for function, args in calls:
            self.assert_native_failure(function, args, "Null game handle")
        self.assertIsNone(self.lib.sg_clone(None))
        self.assertEqual(self.lib.sg_last_error(), b"Null game handle")
        self.lib.sg_destroy(None)
        self.assert_native_failure(self.lib.sg_default_rules, (None,), "Null output pointer")
        rules = bridge.sg_rules()
        self.assertEqual(self.lib.sg_default_rules(C.byref(rules)), 0)
        self.assertEqual((rules.max_moves, rules.declaration), (320, 1))
        for maximum, declaration, message in ((-1, 1, "Invalid max_moves"),
                                              (1000001, 1, "Invalid max_moves"),
                                              (320, 2, "Invalid declaration rule")):
            rules = bridge.sg_rules(maximum, declaration)
            self.assertIsNone(self.lib.sg_create(C.byref(rules), None))
            self.assertEqual(self.lib.sg_last_error().decode(), message)
        for sfen in ("", START_SFEN[:-1] + "1000001", "9/9/9/9/9/9/9/9/9 b - 1"):
            with self.assertRaises(ValueError) as reference:
                ShogiGame(sfen=sfen)
            self.assertIsNone(self.lib.sg_create(None, sfen.encode()))
            self.assertEqual(self.lib.sg_last_error().decode(), str(reference.exception))

    def test_python_buffer_validation_and_valid_outputs(self):
        for method, size in ((self.game.encode_into, 3969), (self.game.legal_mask_into, 2188)):
            invalid = [None, [], bytes(size * 4), bytearray(size * 4), array("d", [0]) * size,
                       array("f", [0]) * (size - 1), array("f", [0]) * (size + 1),
                       memoryview(array("f", [0]) * size).toreadonly(),
                       memoryview(array("f", [0]) * (size * 2))[::2],
                       memoryview(bytearray(size * 4 + 1))[1:].cast("f")]
            before = fresh_snapshot(self.game)
            for buffer in invalid:
                with self.subTest(size=size, buffer_type=type(buffer)), self.assertRaises(ValueError):
                    method(buffer)
                self.assertEqual(fresh_snapshot(self.game), before)
            output = array("f", [42]) * size
            method(output)
            expected = array("f", encode(ShogiGame(), 0)) if size == 3969 else array("f", [0]) * size
            if size == 2188:
                for label in legal_labels(ShogiGame()):
                    expected[label] = 1
            self.assertEqual(output.tobytes(), expected.tobytes())

    def test_close_finalizer_clone_and_use_after_close(self):
        self.game.play(self.game.legal_actions()[0])
        clone = self.game.clone()
        self.addCleanup(clone.close)
        finalizer = self.game._finalizer
        with mock.patch.object(self.lib, "sg_destroy", wraps=self.lib.sg_destroy) as destroy:
            ephemeral = bridge.NativeShogiGame()
            reference = weakref.ref(ephemeral)
            ephemeral_handle = ephemeral._handle
            del ephemeral
            gc.collect()
            self.assertIsNone(reference())
            destroy.assert_called_once_with(ephemeral_handle)
        before = fresh_snapshot(clone)
        self.game.close()
        self.game.close()
        finalizer()
        self.assertFalse(finalizer.alive)
        self.assertEqual(fresh_snapshot(clone), before)
        self.assertTrue(clone.undo())
        self.assertEqual(clone.sfen(), START_SFEN)
        operations = [lambda name=name: getattr(self.game, name) for name in
                      ("rules", "initial", "position", "outcome", "ply", "moves")]
        operations += [self.game.legal_actions, self.game.legal_labels, self.game.clone,
                       self.game.undo, self.game.history_keys, self.game.in_check, self.game.checkers,
                       self.game.can_declare, self.game.resign, self.game.sfen,
                       lambda: self.game.is_legal(0), lambda: self.game.play(0),
                       lambda: self.game.perft(0), lambda: self.game.encode_into(array("f", [0]) * 3969),
                       lambda: self.game.legal_mask_into(array("f", [0]) * 2188)]
        for operation in operations:
            with self.assertRaisesRegex(ValueError, "^Game is closed$"):
                operation()

    def test_replay_failure_closes_handle(self):
        with mock.patch.object(self.lib, "sg_destroy", wraps=self.lib.sg_destroy) as destroy:
            with self.assertRaisesRegex(ValueError, "^Illegal move$"):
                bridge.NativeShogiGame.replay(Rules(), START_SFEN, [self.game.legal_actions()[0], 0])
            destroy.assert_called_once()


class NativeLoaderTests(unittest.TestCase):
    def test_version_and_each_struct_size_mismatch(self):
        for symbol, message in (("sg_abi_version", "ABI version mismatch"),
                                ("sg_sizeof_rules", "sg_rules size mismatch"),
                                ("sg_sizeof_state", "sg_state size mismatch")):
            library = mock.Mock()
            library.sg_abi_version.return_value = 1
            library.sg_sizeof_rules.return_value = C.sizeof(bridge.sg_rules)
            library.sg_sizeof_state.return_value = C.sizeof(bridge.sg_state)
            getattr(library, symbol).return_value += 1
            with self.subTest(symbol=symbol), mock.patch.object(bridge.C, "CDLL", return_value=library):
                with self.assertRaisesRegex(RuntimeError, message):
                    bridge._load_path.__wrapped__("mock-library")

    def test_missing_symbols_load_failure_and_absent_library(self):
        for failure in (OSError("unavailable"), AttributeError("missing symbol")):
            with mock.patch.object(bridge.C, "CDLL", side_effect=failure):
                with self.assertRaisesRegex(RuntimeError, "Unable to load the native shogi ABI"):
                    bridge._load_path.__wrapped__("mock-library")
        with mock.patch.object(bridge, "find_library", side_effect=RuntimeError("missing")):
            self.assertFalse(bridge.available())
            with self.assertRaisesRegex(RuntimeError, "missing"):
                bridge.NativeShogiGame()
        with mock.patch.dict(os.environ, {"SHOGI_LIB": "missing-shogi-library"}):
            with self.assertRaisesRegex(RuntimeError, "SHOGI_LIB does not exist"):
                bridge.find_library()
