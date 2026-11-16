"""Independent native search bridge and shared agent contract regressions."""
import unittest


class LoaderTests(unittest.TestCase):
    def test_module_is_dependency_free(self):
        from shogi.native_search import NativeSearch, AlphaBetaCppAgent
        self.assertTrue(callable(NativeSearch))
        self.assertTrue(callable(AlphaBetaCppAgent))

import ctypes as C
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import random
import subprocess
import sys
import threading
from unittest import mock

from hasami.agents import AgentCancelled, SearchLimits
from shogi import create_game
from shogi.model import Rules, REASONS, DECLARE
from shogi.notation import usi_to_move
from shogi import native_search as bridge
from tests.shogi_suite.support import snapshot, play_text
from tests.shogi_suite.test_perft import CASES, ROWS

NATIVE = bridge.available()
if not NATIVE and os.environ.get("HASAMI_REQUIRE_NATIVE") == "1":
    raise RuntimeError("Native shogi library is required")


class NativeLoaderTests(unittest.TestCase):
    def test_missing_library_is_explicit_and_never_falls_back(self):
        with mock.patch.object(bridge, "find_library", side_effect=RuntimeError("missing")):
            self.assertFalse(bridge.available())
            with self.assertRaisesRegex(RuntimeError, "missing"):
                bridge.NativeSearch()
            with self.assertRaisesRegex(RuntimeError, "missing"):
                bridge.AlphaBetaCppAgent()
            from shogi.agents import native_baseline
            with self.assertRaisesRegex(RuntimeError, "missing"):
                native_baseline("alphabeta-cpp:2")
        with mock.patch.dict(os.environ, {"SHOGI_SEARCH_LIB": "/nonexistent/hl0049.so"}):
            with self.assertRaisesRegex(RuntimeError, "SHOGI_SEARCH_LIB"):
                bridge.find_library()

    def test_abi_version_layout_and_missing_symbols(self):
        structures = (bridge.ss_options, bridge.ss_limits, bridge.ss_result,
                      bridge.ss_root_score, bridge.ss_state)
        symbols = ["ss_abi_version"] + ["ss_sizeof_" + s.__name__[3:] for s in structures]
        for symbol in symbols:
            lib = mock.Mock()
            lib.ss_abi_version.return_value = 1
            for structure in structures:
                getattr(lib, "ss_sizeof_" + structure.__name__[3:]).return_value = C.sizeof(structure)
            getattr(lib, symbol).return_value += 1
            with self.subTest(symbol=symbol), mock.patch.object(bridge.C, "CDLL", return_value=lib):
                with self.assertRaisesRegex(RuntimeError, "mismatch"):
                    bridge._load_path.__wrapped__("fake")
        for error in (OSError("missing"), AttributeError("missing")):
            with mock.patch.object(bridge.C, "CDLL", side_effect=error):
                with self.assertRaisesRegex(RuntimeError, "Unable to load"):
                    bridge._load_path.__wrapped__("fake")

    def test_required_library_failure_and_allowlisted_skip(self):
        command = [sys.executable, "-m", "unittest", "tests.shogi_suite.test_native_search.SearchTests.test_start_perft", "-v"]
        env = {**os.environ, "SHOGI_SEARCH_LIB": "/nonexistent/hl0049.so", "HASAMI_REQUIRE_NATIVE": "1"}
        required = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertNotEqual(required.returncode, 0)
        self.assertIn("Native shogi library is required", required.stderr)
        env["HASAMI_REQUIRE_NATIVE"] = "0"
        optional = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertEqual(optional.returncode, 0)
        self.assertIn("Native shogi library is unavailable", optional.stderr)


@unittest.skipUnless(NATIVE, "Native shogi library is unavailable")
class SearchTests(unittest.TestCase):
    def test_start_perft(self):
        with bridge.NativeSearch(create_game()) as search:
            self.assertEqual(search.perft(3), 25470)
            self.assertEqual(search.perft(0), 1)

    def test_all_vector_legal_moves_and_outcomes(self):
        with bridge.NativeSearch(hash_mb=1) as search:
            for row in ROWS:
                game = create_game(sfen=row["sfen"])
                search.set_position(game)
                self.assertEqual(search.legal_actions(), game.legal_actions(), row["sfen"])
                state = search.state()
                self.assertEqual((state["winner"], state["reason"]), (game.outcome.winner, game.outcome.reason))
            for row in CASES["declaration_cases"]:
                for declaration in ("none", "csa27"):
                    game = create_game(rules=Rules(declaration=declaration), sfen=row["sfen"])
                    search.set_position(game)
                    self.assertEqual(search.legal_actions(), game.legal_actions())

    def test_seeded_games_and_agent_contract_on_both_backends(self):
        rng = random.Random(490049)
        agent = bridge.AlphaBetaCppAgent(hash_mb=1)
        for backend in ("python", "cpp"):
            with bridge.NativeSearch(hash_mb=1) as search:
                for _ in range(4):
                    game = create_game(backend, Rules(max_moves=40))
                    self.addCleanup(game.close)
                    while True:
                        search.set_position(game)
                        state = search.state()
                        self.assertEqual(search.legal_actions(), game.legal_actions())
                        self.assertEqual((state["ply"], state["turn"], state["winner"], state["reason"]),
                            (game.ply, game.position.turn, game.outcome.winner, game.outcome.reason))
                        if game.outcome.terminal:
                            with self.assertRaisesRegex(ValueError, "No legal action"):
                                search.go(depth=1)
                            break
                        if game.ply % 5 == 0:
                            before = snapshot(game)
                            decision = agent.choose(game, SearchLimits(2, 10000, 500))
                            self.assertIs(type(decision.action), int)
                            self.assertIn(decision.action, game.legal_actions())
                            self.assertEqual(snapshot(game), before)
                        game.play(rng.choice(game.legal_actions()))

    def test_history_repetition_perpetual_check_and_caps(self):
        for sfen, cycle, score in (
            ("4k4/9/9/9/9/9/9/9/4K4 b - 1", "5i6i 5a6a 6i5i 6a5a", 0),
            ("4k4/9/5R3/9/9/9/9/9/4K4 b - 1", "4c5c 5a4a 5c4c 4a5a", 999999),
        ):
            game = create_game(rules=Rules(12), sfen=sfen)
            steps = cycle.split() * 3
            play_text(game, steps[:-1])
            last = usi_to_move(game, steps[-1])
            with bridge.NativeSearch(game, features=0, hash_mb=1) as search:
                result = search.go(depth=1, root_moves=[last])
                self.assertEqual(result["score"], score)
                game.play(last)
                search.set_position(game)
                self.assertEqual(search.state()["reason"], game.outcome.reason)
                self.assertEqual(search.legal_actions(), [])
        with bridge.NativeSearch(create_game(rules=Rules(1)), hash_mb=1) as search:
            self.assertEqual(search.go(depth=2)["score"], 0)

    def test_declaration_and_resignation(self):
        for case in CASES["declaration_cases"]:
            if not case["declarable"]:
                continue
            game = create_game(sfen=case["sfen"])
            with bridge.NativeSearch(game, features=0, hash_mb=1) as search:
                result = search.go(depth=1)
                self.assertEqual(result["move"], DECLARE)
                game.play(DECLARE)
                search.set_position(game)
                self.assertEqual(search.state()["reason"], "declaration")
                self.assertEqual(search.legal_actions(), [])
        game = create_game()
        game.resign()
        with bridge.NativeSearch(game, hash_mb=1) as search:
            self.assertEqual(search.state()["reason"], "resign")
            with self.assertRaises(ValueError):
                search.go(depth=1)

    def test_fixed_depth_nodes_multipv_and_legal_pv(self):
        game = create_game()
        with bridge.NativeSearch(game, hash_mb=1) as search:
            for limits in ({"depth": 3}, {"depth": 64, "nodes": 777}):
                a, b = search.go(**limits), search.go(**limits)
                for name in ("move", "score", "depth", "nodes", "pv"):
                    self.assertEqual(a[name], b[name])
                work = game.clone()
                for code in a["pv"]:
                    self.assertTrue(work.is_legal(code))
                    work.play(code)
                if "nodes" in limits:
                    self.assertEqual(a["nodes"], 777)
            all_moves = search.go(depth=2, multipv="all")
            self.assertEqual({row["move"] for row in all_moves["root_scores"]}, set(game.legal_actions()))
            for row in all_moves["root_scores"]:
                single = search.go(depth=2, root_moves=[row["move"]])
                self.assertEqual((row["score"], row["depth"]), (single["score"], 2))

    def test_mate_claim_requires_checkmate_proof(self):
        vectors = Path(__file__).resolve().parents[2] / "cpp/shogi_search/tests/data/mates-v1.tsv"
        for line in vectors.read_text().splitlines():
            if line.startswith("#"):
                continue
            _, depth, sfen, _ = line.split("\t")
            with bridge.NativeSearch(create_game(sfen=sfen), features=0, hash_mb=1) as search:
                result = search.go(depth=int(depth))
                self.assertEqual((result["mate"], result["mate_plies"]), (int(depth), int(depth)))
        mate = create_game(sfen="k8/2G6/9/9/1R7/9/9/9/8K b - 1")
        with bridge.NativeSearch(mate, features=0, hash_mb=1) as search:
            result = search.go(depth=1)
            self.assertEqual((result["mate"], result["mate_plies"]), (1, 1))
        no_reply = create_game(sfen="k8/9/9/GR7/3N5/9/9/9/8K b - 1")
        with bridge.NativeSearch(no_reply, features=0, hash_mb=1) as search:
            result = search.go(depth=1)
            no_reply.play(result["move"])
            self.assertEqual(no_reply.outcome.reason, "no_legal_moves")
            self.assertIsNone(result["mate"])

    def test_multipv_matches_isolated_roots_with_reductions(self):
        cases = ((None, 5, 12448),
                 ("lnsgkgsn1/2r4bl/p1pp2ppp/1p2pp3/9/4P4/PPPP1PPPP/1B1KR1S1L/LNSG1G1N1 b - 11",
                  4, 10822))
        for sfen, depth, reported_move in cases:
            with self.subTest(sfen=sfen, depth=depth):
                game = create_game(sfen=sfen)
                self.addCleanup(game.close)
                with bridge.NativeSearch(game, hash_mb=1) as search:
                    result = search.go(depth=depth, multipv="all")
                    self.assertEqual(result["depth"], depth)
                    rows = {row["move"]: row for row in result["root_scores"]}
                    # Check the reported counterexample first, then every legal root.
                    moves = [reported_move] + [m for m in game.legal_actions() if m != reported_move]
                    for move in moves:
                        with self.subTest(move=move):
                            single = search.go(depth=depth, root_moves=[move])
                            self.assertEqual(single["depth"], depth)
                            self.assertEqual(rows[move]["score"], single["score"])

    def test_invalid_inputs_buffers_and_failure_atomicity(self):
        game = create_game()
        with bridge.NativeSearch(game, hash_mb=1) as search:
            before = search.state()
            for kwargs in ({"depth": 0}, {"depth": True}, {"nodes": 2**64}, {"time_ms": -1},
                           {"multipv": -1}, {"soft_time_ms": 20, "time_ms": 1},
                           {"root_moves": [-1]}, {"root_moves": [0]},
                           {"root_moves": [game.legal_actions()[0]] * 2}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    search.go(**kwargs)
                self.assertEqual(search.state(), before)
            lib, handle = search._lib, search._handle
            sentinel = C.c_int32(123)
            self.assertEqual(lib.ss_legal_moves(handle, C.byref(sentinel), 1), -1)
            self.assertEqual(sentinel.value, 123)
            self.assertEqual(lib.ss_go(None, None, None, 0, None), -1)
            self.assertEqual(lib.ss_set_position(handle, b"bad", 320, 1, None, 0, 0), -1)
            self.assertEqual(search.state(), before)
            with self.assertRaises(ValueError):
                search.see(0)
        search.close()
        for call in (search.state, search.legal_actions, lambda: search.go(depth=1), search.stop):
            with self.assertRaisesRegex(ValueError, "closed"):
                call()

    def test_cancellation_time_and_external_stop(self):
        game = create_game()
        before = snapshot(game)
        event = threading.Event()
        event.set()
        agent = bridge.AlphaBetaCppAgent(hash_mb=1)
        with self.assertRaises(AgentCancelled):
            agent.choose(game, SearchLimits(), event)
        event.clear()
        with bridge.NativeSearch(game, hash_mb=1) as search:
            original = search._lib.ss_go
            entered = threading.Event()
            def signal(*args):
                entered.set()
                return original(*args)
            with mock.patch.object(search._lib, "ss_go", side_effect=signal):
                with ThreadPoolExecutor(1) as pool:
                    future = pool.submit(search.go, depth=64, cancel=event)
                    self.assertTrue(entered.wait(5))
                    event.set()
                    with self.assertRaises(AgentCancelled):
                        future.result(timeout=5)
                entered.clear()
                with ThreadPoolExecutor(1) as pool:
                    future = pool.submit(search.go, depth=64)
                    self.assertTrue(entered.wait(5))
                    search.stop()
                    result = future.result(timeout=5)
                    self.assertTrue(result["stopped"])
            result = search.go(depth=64, time_ms=1)
            self.assertFalse(result["completed"])
            self.assertIn(result["move"], game.legal_actions())
        self.assertEqual(snapshot(game), before)

    def test_baseline_budget_parser(self):
        from shogi.agents import native_baseline
        for spec, depth, nodes in (("alphabeta-cpp:2", 2, 2**64-1), ("alphabeta-cpp:nodes=500", 64, 500)):
            agent, limits = native_baseline(spec)
            self.assertIsInstance(agent, bridge.AlphaBetaCppAgent)
            self.assertEqual((limits.max_depth, limits.node_limit), (depth, nodes))
        for spec in ("alphabeta-cpp", "alphabeta-cpp:", "alphabeta-cpp:0", "alphabeta-cpp:nodes=0"):
            with self.assertRaises(ValueError):
                native_baseline(spec)

    def test_progress_copies_completed_results_and_preserves_node_budget(self):
        game = create_game()
        self.addCleanup(game.close)
        with bridge.NativeSearch(game, hash_mb=1) as search:
            for limits in ({"depth": 4}, {"depth": 64, "nodes": 777}):
                with self.subTest(limits=limits):
                    iterations = []
                    expected = search.go(**limits)
                    actual = search.go(**limits, on_iteration=iterations.append)
                    self.assertEqual([row["depth"] for row in iterations], list(range(1, actual["depth"] + 1)))
                    for key in ("move", "score", "depth", "nodes", "pv", "completed"):
                        self.assertEqual(expected[key], actual[key])
                    for key in ("move", "score", "depth", "pv"):
                        self.assertEqual(iterations[-1][key], actual[key])
                    for row in iterations:
                        work = game.clone()
                        try:
                            for move in row["pv"]:
                                self.assertTrue(work.is_legal(move))
                                work.play(move)
                        finally:
                            work.close()
                        self.assertIsNone(row["mate"])

    def test_progress_stop_cancel_and_callback_error_cleanup(self):
        game = create_game()
        self.addCleanup(game.close)
        before = snapshot(game)
        with bridge.NativeSearch(game, hash_mb=1) as search:
            iterations = []
            def stop(row):
                iterations.append(row)
                if row["depth"] == 3:
                    search.stop()
            result = search.go(depth=64, on_iteration=stop)
            self.assertTrue(result["stopped"])
            self.assertEqual(result["depth"], 3)
            self.assertEqual(result["move"], iterations[-1]["move"])
            self.assertEqual(result["nodes"], iterations[-1]["nodes"])
            def fail(row):
                raise RuntimeError("observer failed")
            with self.assertRaisesRegex(RuntimeError, "observer failed"):
                search.go(depth=64, on_iteration=fail)
            self.assertTrue(search.go(depth=1)["completed"])
        cancel = threading.Event()
        decisions = []
        def cancel_after_iteration(decision):
            decisions.append(decision)
            if decision.depth == 3:
                cancel.set()
        with self.assertRaises(AgentCancelled):
            bridge.AlphaBetaCppAgent(hash_mb=1).choose(
                game, SearchLimits(64, 30000, 2**64-1), cancel,
                on_iteration=cancel_after_iteration)
        self.assertGreaterEqual(decisions[-1].depth, 3)
        self.assertTrue(all(game.is_legal(decision.action) for decision in decisions))
        self.assertEqual(snapshot(game), before)


def write_test_nnue(path):
    """Write a one-wide constant 75 cp hlnnw1 fixture using only the standard library."""
    import hashlib
    import struct
    count = 183303 + 1 + 2 + 1 + 1 + 1 + 1 + 1
    header = struct.pack("<8sI16s48sIIIIfQ", b"hlnnw1\0\0", 1, b"hlnn-kp-v1",
                         b"hlnn-kp-v1:w1:1:1:f32:cp600", 183303, 1, 1, 1, 600., count)
    data = header + b"\0" * (4*(count-1)) + struct.pack("<f", .125)
    path.write_bytes(data + hashlib.sha256(data).digest())


@unittest.skipUnless(NATIVE, "Native shogi library is unavailable")
class NNUESearchTests(unittest.TestCase):
    def test_weights_search_agent_and_baseline(self):
        import tempfile
        from shogi.agents import native_baseline
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights with spaces.hlnnw1"
            write_test_nnue(path)
            game = create_game()
            play_text(game, ["7g7f", "3c3d"])
            before = snapshot(game)
            with bridge.NativeSearch(game, nnue=path, hash_mb=1) as search:
                self.assertEqual(search.evaluate_float(), 75.)
                self.assertEqual(search.evaluate(), 75)
                a, b = search.go(nodes=150), search.go(nodes=150)
                for field in ("move", "score", "depth", "nodes", "pv", "root_scores"):
                    self.assertEqual(a[field], b[field])
                self.assertIn(a["move"], game.legal_actions())
                with self.assertRaises(ValueError):
                    search.load_nnue(path.with_name("missing"))
                self.assertEqual(search.evaluate(), 75)
                search.load_nnue()
                self.assertEqual(search.evaluator, "bootstrap-v1")
            agent, limits = native_baseline(f"alphabeta-cpp:nodes=100,nnue={path}")
            try:
                for _ in range(2):
                    self.assertIn(agent.choose(game, limits).action, game.legal_actions())
                cancel = threading.Event(); cancel.set()
                with self.assertRaises(AgentCancelled):
                    agent.choose(game, limits, cancel)
            finally:
                agent.close()
            self.assertEqual(snapshot(game), before)
            for spec in ("alphabeta-cpp:nodes=100,nnue=", "alphabeta-cpp:nodes=100,unknown=x"):
                with self.assertRaises(ValueError):
                    native_baseline(spec)
