"""Native PUCT parity, ownership, and training integration regressions."""

import os
import unittest

from tests.learn._common import LearnTestCase, requires_torch


@requires_torch
class NativeMctsTests(LearnTestCase):
    def test_initial_search_parity(self):
        import numpy as np
        from hasami_learn.games import get_spec
        from hasami_learn.search.mcts import MCTS, MctsConfig
        from hasami_learn.search.native_mcts import NativeMCTS

        spec = get_spec()
        game = spec.create_game("cpp")
        self.addCleanup(game.close)
        self.assertEqual(len(game.legal_actions()), 63)
        self.assertEqual(game.perft(2), 3717)
        encoder = spec.make_encoder()
        def evaluate(planes, masks):
            return masks / masks.sum(1, keepdims=True), np.zeros(len(planes), np.float32)
        config = MctsConfig(simulations=32)
        reference = MCTS(evaluate, encoder, config, np.random.default_rng(7), spec)
        native = NativeMCTS(evaluate, encoder, config, np.random.default_rng(7), spec)
        expected, actual = reference.run(game), native.run(game)
        np.testing.assert_array_equal(actual.visits, expected.visits)
        np.testing.assert_array_equal(actual.q_values, expected.q_values)
        self.assertEqual(actual.root_value, expected.root_value)
        self.assertEqual(actual.simulations, expected.simulations)

    @classmethod
    def setUpClass(cls):
        from hasami_learn.games import get_spec
        from hasami_learn.search.native_mcts import find_library, _load_path

        try:
            _load_path(str(find_library()))
            for name in ("hasami", "shogi"):
                game = get_spec(name).create_game("cpp")
                game.close()
        except (RuntimeError, OSError, ValueError) as exc:
            if os.environ.get("HASAMI_REQUIRE_NATIVE") == "1":
                raise
            raise unittest.SkipTest(f"Native library is not built: {exc}") from exc

    @staticmethod
    def snapshot(game):
        return (game.initial, game.position, game.rules, game.outcome, tuple(game.moves),
                tuple(game.history_keys()), getattr(game, "last_capture", None))

    @staticmethod
    def hash_evaluator(planes, masks):
        import hashlib
        import numpy as np

        probabilities = np.zeros_like(masks)
        values = np.empty(len(planes), np.float32)
        for i, plane in enumerate(planes):
            seed = int.from_bytes(hashlib.sha256(plane.tobytes()).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            probabilities[i] = rng.random(len(masks[i]), dtype=np.float32) * masks[i]
            values[i] = rng.uniform(-1, 1)
        return probabilities, values

    def sampled_positions(self, spec):
        import random
        from hasami.agents import SearchLimits
        from hasami.model import Rules
        from tests.test_rules import setup

        positions = []
        def retain(game):
            snapshot = game.clone()
            self.addCleanup(snapshot.close)
            positions.append(snapshot)
        rng = random.Random(3914)
        for mode in ("random", "greedy"):
            collected = 0
            episode = 0
            while collected < 100:
                params = ({"declaration": "none" if episode % 2 else "csa27", "max_moves": 320}
                          if spec.game_id == "shogi" else None)
                game = spec.create_game("cpp", rules=params)
                agent, limits = spec.baseline_agent(mode, episode + 39)
                limits = limits or SearchLimits(max_depth=1, time_ms=100000, node_limit=1000000)
                try:
                    # Random lead-ins make successive greedy episodes distinct.
                    for _ in range(episode % 7):
                        if not game.outcome.terminal:
                            game.play(rng.choice(game.legal_actions()))
                    for _ in range(50):
                        retain(game)
                        collected += 1
                        if game.outcome.terminal or collected >= 100:
                            break
                        clone = game.clone()
                        try:
                            action = agent.choose(clone, limits).action
                        finally:
                            clone.close()
                        spec.step(game, {game.position.turn: action})
                finally:
                    game.close()
                episode += 1
        if spec.game_id == "hasami":
            game = spec.create_game("cpp", position=setup(
                [38, 67, 72, 73, 74, 75, 76, 77, 78], [39, 37, 3, 0, 1, 2, 4]))
            try:
                retain(game)
                game.play(67 * 81 + 40)
                self.assertEqual(game.position.pending, 0)
                retain(game)
                game.play(9)
                self.assertEqual(game.outcome.reason, "lead_confirmed")
                retain(game)
                game.undo()
                game.play(3 * 81 + 39)
                self.assertEqual(game.position.pending, -1)
                retain(game)
            finally:
                game.close()
            game = spec.create_game("cpp")
            try:
                for action in [72*81+63, 9, 63*81+72, 9*81] * 3:
                    game.play(action)
                    retain(game)
                self.assertEqual(game.outcome.reason, "repetition")
            finally:
                game.close()
            for rules in (Rules(edge_capture=False), Rules(repetition_limit=2),
                          Rules(repetition_limit=100), Rules(lead_target=0),
                          Rules(capture_target=0, lead_target=0, repetition_limit=0, no_moves_draw=True)):
                game = spec.create_game("cpp", rules=rules)
                try:
                    for _ in range(5):
                        retain(game)
                        game.play(rng.choice(game.legal_actions()))
                finally:
                    game.close()
        else:
            import json
            from pathlib import Path
            from shogi.notation import usi_to_move
            from shogi.model import DECLARE

            cases = json.loads((Path(__file__).parents[1] / "shogi_suite/vectors/rule_cases.json").read_text())
            for declaration in ("none", "csa27"):
                for case in cases["move_cases"] + cases["declaration_cases"]:
                    game = spec.create_game("cpp", rules={"declaration": declaration}, position=case["sfen"])
                    try:
                        retain(game)
                        if "move" in case and case["legal"]:
                            game.play(usi_to_move(game, case["move"]))
                            retain(game)
                        if declaration == "csa27" and case.get("declarable"):
                            game.play(DECLARE)
                            retain(game)
                    finally:
                        game.close()
                for cap in (0, 1, 2, 7):
                    game = spec.create_game("cpp", rules={"max_moves": cap, "declaration": declaration})
                    try:
                        for _ in range((cap or 3) + 1):
                            retain(game)
                            if game.outcome.terminal:
                                break
                            game.play(rng.choice(game.legal_actions()))
                    finally:
                        game.close()
            for sfen, cycle in (
                ("4k4/9/9/9/9/9/9/9/4K4 b - 1", ["5i6i", "5a6a", "6i5i", "6a5a"]),
                ("4k4/9/5R3/9/9/9/9/9/4K4 b - 1", ["4c5c", "5a4a", "5c4c", "4a5a"]),
            ):
                game = spec.create_game("cpp", position=sfen)
                try:
                    for move in cycle * 3:
                        retain(game)
                        game.play(usi_to_move(game, move))
                    retain(game)
                    self.assertTrue(game.outcome.terminal)
                finally:
                    game.close()
            game = spec.create_game("cpp")
            game.play(game.legal_actions()[0])
            game.resign()
            retain(game)
            game.close()
        self.assertGreaterEqual(len(positions), 200)
        self.assertGreaterEqual(len({(g.position, tuple(g.history_keys()), g.rules) for g in positions}), 200)
        return positions

    def assert_results_equal(self, expected, actual):
        import numpy as np
        self.assertEqual(len(expected), len(actual))
        for index, (a, b) in enumerate(zip(expected, actual)):
            with self.subTest(position=index):
                np.testing.assert_array_equal(a.visits, b.visits)
                np.testing.assert_allclose(a.q_values, b.q_values, rtol=0, atol=1e-12, equal_nan=True)
                self.assertAlmostEqual(a.root_value, b.root_value, delta=1e-12)
                self.assertEqual(a.simulations, b.simulations)
                np.testing.assert_allclose(a.policy_target, b.policy_target, rtol=0, atol=1e-12)
                self.assertEqual(a.selected_action, b.selected_action)

    def corpus_parity(self, name, mode="puct", noise=True):
        import hashlib
        import numpy as np
        import torch
        from hasami_learn.games import get_spec
        from hasami_learn.networks.mlp import MlpNet
        from hasami_learn.search.mcts import MCTS, MctsConfig, batched_evaluator
        from hasami_learn.search.native_mcts import NativeMCTS

        spec = get_spec(name)
        encoder = spec.make_encoder()
        games = self.sampled_positions(spec)
        snapshots = [self.snapshot(game) for game in games]
        config = MctsConfig(simulations=64 if mode == "gumbel" else 16, batch_size=17,
                            truncation_value=0.4, root_selection=mode, add_noise=noise)
        torch.manual_seed(39)
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        network = MlpNet(num_planes=encoder.num_planes, board_shape=spec.board_shape,
                         action_size=spec.action_size, hidden=32, layers=1)
        for evaluator in (self.hash_evaluator, batched_evaluator(network, "cpu", "off")):
            traces = []
            def tracked(planes, masks):
                traces.append((len(planes), hashlib.sha256(planes.tobytes() + masks.tobytes()).digest()))
                return evaluator(planes, masks)
            rng = np.random.default_rng(39)
            expected = MCTS(tracked, encoder, config, rng, spec).run_many(games)
            expected_trace, expected_rng = traces.copy(), rng.bit_generator.state
            for threads in (1, 4, 8):
                with self.subTest(game=name, evaluator=str(evaluator), threads=threads):
                    traces.clear()
                    rng = np.random.default_rng(39)
                    search = NativeMCTS(tracked, encoder, config, rng, spec, threads)
                    planes, masks = search.encode_many(games)
                    np.testing.assert_array_equal(planes, np.stack([encoder.encode(game) for game in games]))
                    np.testing.assert_array_equal(masks, np.stack([spec.legal_mask(game, game.position.turn) for game in games]))
                    actual = search.run_many(games)
                    self.assert_results_equal(expected, actual)
                    self.assertEqual(traces, expected_trace)
                    self.assertEqual(rng.bit_generator.state, expected_rng)
                    self.assertEqual([self.snapshot(game) for game in games], snapshots)

    def test_hasami_200_positions_hash_network_encoding_history_and_threads(self):
        self.corpus_parity("hasami")

    def test_shogi_200_positions_hash_network_encoding_history_and_threads(self):
        self.corpus_parity("shogi")

    def test_gumbel_hasami_corpus_noisy_and_noiseless(self):
        for noise in (True, False):
            self.corpus_parity("hasami", "gumbel", noise)

    def test_gumbel_shogi_corpus_noisy_and_noiseless(self):
        for noise in (True, False):
            self.corpus_parity("shogi", "gumbel", noise)

    def test_gumbel_small_budgets_odd_m_and_cancellation(self):
        import numpy as np
        from hasami_learn.games import get_spec
        from hasami_learn.search.mcts import MCTS, MctsConfig
        from hasami_learn.search.native_mcts import NativeMCTS

        class Cancel:
            def __init__(self):
                self.calls = 0

            def is_set(self):
                self.calls += 1
                return self.calls >= 6

        for name in ("hasami", "shogi"):
            spec = get_spec(name)
            games = [spec.create_game("cpp") for _ in range(3)]
            for game in games:
                self.addCleanup(game.close)
            for budget in (0, 1, 2, 7, 17, 64):
                for m in (1, 3, 5, 100):
                    for cancelled in (False, True):
                        config = MctsConfig(root_selection="gumbel", simulations=budget, gumbel_m=m)
                        results = [cls(self.hash_evaluator, spec.make_encoder(), config,
                                       np.random.default_rng(48), spec).run_many(
                                           games, cancel=Cancel() if cancelled else None)
                                   for cls in (MCTS, NativeMCTS)]
                        self.assert_results_equal(*results)

    def test_zero_terminal_cancel_deadline_and_tie_parity(self):
        import threading
        import numpy as np
        from hasami_learn.games import get_spec
        from hasami_learn.search.mcts import MCTS, MctsConfig
        from hasami_learn.search.native_mcts import NativeMCTS
        from unittest.mock import patch

        class BoundaryCancel:
            def __init__(self):
                self.calls = 0
            def is_set(self):
                self.calls += 1
                return self.calls >= 6
        for name in ("hasami", "shogi"):
            spec = get_spec(name)
            encoder = spec.make_encoder()
            games = [spec.create_game("cpp") for _ in range(4)]
            for game in games:
                self.addCleanup(game.close)
            def uniform(planes, masks):
                return masks.copy(), np.zeros(len(planes), np.float32)
            for noise in (True, False):
                for scenario in ("zero", "expired", "cancelled", "boundaries", "ties", "eval_cancel"):
                    results, rngs = [], []
                    for cls in (MCTS, NativeMCTS):
                        rng = np.random.default_rng(40)
                        config = MctsConfig(simulations=80, batch_size=3, add_noise=noise)
                        cancel = BoundaryCancel() if scenario == "boundaries" else threading.Event()
                        if scenario == "cancelled":
                            cancel.set()
                        evaluations = []
                        def evaluator(planes, masks):
                            evaluations.append(len(planes))
                            if scenario == "eval_cancel":
                                cancel.set()
                            return uniform(planes, masks)
                        kwargs = {"cancel": cancel} if scenario not in ("ties", "zero") else {}
                        if scenario == "expired":
                            kwargs["deadline"] = 0.0
                        if scenario == "zero":
                            kwargs["simulations"] = 0
                        results.append(cls(evaluator, encoder, config, rng, spec).run_many(games, **kwargs))
                        rngs.append(rng.bit_generator.state)
                        self.assertEqual(evaluations[:2], [3, 1])
                    self.assert_results_equal(*results)
                    self.assertEqual(*rngs)
                    if scenario == "boundaries":
                        self.assertEqual([r.simulations for r in results[0]], [2, 1, 1, 1])
            search = NativeMCTS(uniform, encoder, MctsConfig(), np.random.default_rng(0), spec)
            self.assertEqual(search.run_many([]), [])
            for budget in (-1, True, 1.5):
                with self.assertRaises(ValueError):
                    search.run(games[0], budget)
            # A controlled deadline crosses the same check boundary despite different runtimes.
            results = []
            for cls in (MCTS, NativeMCTS):
                with patch("hasami_learn.search.mcts.time.perf_counter", side_effect=[0, 0, 0, 0, 1, 2]):
                    results.append(cls(uniform, encoder, MctsConfig(simulations=3),
                                       np.random.default_rng(0), spec).run_many(games, deadline=0.5))
            self.assert_results_equal(*results)
            self.assertEqual([r.simulations for r in results[0]], [1] * 4)

    def test_evaluator_errors_and_invalid_outputs(self):
        import numpy as np
        from hasami_learn.games import get_spec
        from hasami_learn.search.mcts import MCTS, MctsConfig
        from hasami_learn.search.native_mcts import NativeMCTS

        for name in ("hasami", "shogi"):
            spec = get_spec(name)
            game = spec.create_game("cpp")
            self.addCleanup(game.close)
            before = self.snapshot(game)
            encoder = spec.make_encoder()
            for variant in ("nan", "inf", "negative", "zero", "value_nan", "policy_shape", "value_shape", "exception"):
                for fail_at in (1, 3):
                    for cls in (MCTS, NativeMCTS):
                        calls = []
                        def evaluator(planes, masks):
                            calls.append(1)
                            probs, values = self.hash_evaluator(planes, masks)
                            if len(calls) != fail_at:
                                return probs, values
                            if variant == "exception":
                                raise LookupError("Evaluator failed")
                            if variant == "policy_shape":
                                return probs[:, :-1], values
                            if variant == "value_shape":
                                return probs, values[:, None]
                            if variant == "value_nan":
                                values[0] = np.nan
                            elif variant == "zero":
                                probs[:] = 0
                            else:
                                probs[0, np.flatnonzero(masks[0])[0]] = {
                                    "nan": np.nan, "inf": np.inf, "negative": -1}[variant]
                            return probs, values
                        search = cls(evaluator, encoder, MctsConfig(simulations=4), np.random.default_rng(0), spec)
                        with self.subTest(game=name, variant=variant, fail_at=fail_at, cls=cls):
                            with self.assertRaises(LookupError if variant == "exception" else ValueError):
                                search.run(game)
                            self.assertEqual(self.snapshot(game), before)
                            search.evaluate_fn = self.hash_evaluator
                            self.assertEqual(search.run(game).simulations, 4)
            # Illegal prior entries are ignored exactly as in _Node.expand.
            def ignored(planes, masks):
                return np.where(masks > 0, 1.0, np.nan), np.zeros(len(planes))
            self.assert_results_equal(
                [MCTS(ignored, encoder, MctsConfig(), np.random.default_rng(0), spec).run(game)],
                [NativeMCTS(ignored, encoder, MctsConfig(), np.random.default_rng(0), spec).run(game)])

    def test_supports_library_abi_and_configuration_errors(self):
        import numpy as np
        from unittest.mock import patch
        from hasami_learn.games import get_spec
        from hasami_learn.search.mcts import MctsConfig
        from hasami_learn.search.native_mcts import NativeMCTS, _load_path
        from hasami_learn.algorithms.alphazero import AlphaZeroParams

        spec = get_spec()
        encoder = spec.make_encoder()
        game = spec.create_game("cpp")
        self.addCleanup(game.close)
        self.assertTrue(NativeMCTS.supports(spec, encoder, game))
        python_game = spec.create_game("python")
        self.addCleanup(python_game.close)
        self.assertFalse(NativeMCTS.supports(spec, encoder, python_game))
        args = (self.hash_evaluator, encoder, MctsConfig(), np.random.default_rng(0), spec)
        search = NativeMCTS(*args)
        with self.assertRaisesRegex(ValueError, "backend='cpp'"):
            search.run(python_game)
        other_spec = get_spec("shogi")
        self.assertFalse(NativeMCTS.supports(other_spec, encoder, game))
        class CustomSpec(type(spec)):
            pass
        with self.assertRaisesRegex(ValueError, "unmodified"):
            NativeMCTS(*args[:4], CustomSpec())
        for value in (False, 0, -1, 1.5, 2**32):
            with self.assertRaises(ValueError):
                NativeMCTS(*args, threads=value)
        with patch.dict(os.environ, MCTS_LIB=str(self.tmp / "missing.so")):
            with self.assertRaisesRegex(RuntimeError, "MCTS_LIB"):
                NativeMCTS(*args)
        with patch("hasami_learn.search.native_mcts.C.CDLL") as load:
            load.return_value.mz_abi_version.return_value = 999
            with self.assertRaisesRegex(RuntimeError, "ABI version mismatch"):
                _load_path("test-abi-mismatch")
        game.close()
        self.assertFalse(NativeMCTS.supports(spec, encoder, game))
        self.assertEqual(AlphaZeroParams().search_backend, "python")
        for params in ({"search_backend": "auto"}, {"search_threads": 0}, {"search_threads": True}):
            with self.assertRaises(ValueError):
                AlphaZeroParams(**params)

    def test_alphazero_tiny_replay_training_and_resume_both_games_workers(self):
        import numpy as np
        import torch
        from hasami_learn.trainer import Trainer
        from hasami_learn.config import NamedConfig, validate_config_dict
        from hasami_learn.search.native_mcts import NativeMCTS
        from hasami_learn.search.mcts import MCTS

        for name in ("hasami", "shogi"):
            for workers in (1, 2):
                replay, weights = [], []
                checkpoints = []
                for backend in ("python", "cpp"):
                    config = self.tiny("alphazero", simulations=4, batch_games=2,
                                       search_backend=backend, search_threads=4)
                    config.game = NamedConfig(name=name, params={})
                    config.encoder.name = ""
                    config.run.run_id = f"native-mcts-{name}-{workers}-{backend}"
                    config.run.backend = "cpp"
                    config.device.deterministic = True
                    config.device.threads = 1
                    config.selfplay.workers = workers
                    config.selfplay.games_per_iteration = 3
                    config.selfplay.max_plies = 3
                    config.selfplay.truncation_value = 0.4
                    config.replay.min_size = 1
                    config.replay.persist = True
                    config.run.eval_every = 0
                    self.assertEqual(validate_config_dict(config.to_dict()), [])
                    trainer = Trainer(config, echo=False)
                    try:
                        self.assertIsInstance(trainer.learner.search, NativeMCTS if backend == "cpp" else MCTS)
                        checkpoints.append(trainer.run())
                        replay.append({key: array.copy() for key, array in trainer.replay.all().items()})
                        weights.append({key: tensor.clone() for key, tensor in trainer.network.state_dict().items()})
                    finally:
                        trainer.close()
                self.assertEqual(replay[0].keys(), replay[1].keys())
                for key in replay[0]:
                    np.testing.assert_array_equal(replay[0][key], replay[1][key])
                for key in weights[0]:
                    self.assertTrue(torch.equal(weights[0][key], weights[1][key]), key)
                np.testing.assert_allclose(replay[1]["value"], 0.4)
                # Resume in either direction without converting persisted state.
                for index, target in enumerate(("cpp", "python")):
                    config.algorithm.params["search_backend"] = target
                    config.run.run_id = f"resume-{name}-{workers}-{target}"
                    resumed = Trainer(config, resume=checkpoints[index], echo=False)
                    try:
                        self.assertIsInstance(resumed.learner.search, NativeMCTS if target == "cpp" else MCTS)
                        self.assertEqual(resumed.iteration, 1)
                        for key, value in replay[index].items():
                            np.testing.assert_array_equal(value, resumed.replay.all()[key])
                    finally:
                        resumed.close()

    def test_gumbel_mixed_tiny_replay_workers_backends_and_audit(self):
        import numpy as np
        import torch
        from hasami_learn.trainer import Trainer
        from hasami_learn.diagnostics import analyze_store
        from hasami_learn.inference import create_agent

        for name in ("hasami", "shogi"):
            for workers in (1, 2):
                replay, weights = [], []
                for backend in ("python", "cpp"):
                    config = self.tiny("alphazero", simulations=8, batch_games=2, root_selection="gumbel",
                                       value_target_mix=.5, search_backend=backend, search_threads=4)
                    config.game.name = name
                    config.encoder.name = ""
                    config.run.run_id = f"gumbel-{name}-{workers}-{backend}"
                    config.run.backend = "cpp"
                    config.device.deterministic = True
                    config.device.threads = 1
                    config.selfplay.workers = workers
                    config.selfplay.games_per_iteration = 3
                    config.selfplay.max_plies = 3
                    config.selfplay.truncation_value = .4
                    config.selfplay.symmetry_augmentation = True
                    config.datastore.enabled = True
                    config.replay.min_size = 1
                    config.replay.persist = True
                    config.run.eval_every = 0
                    trainer = Trainer(config, echo=False)
                    try:
                        checkpoint = trainer.run()
                        replay.append({k: v.copy() for k, v in trainer.replay.all().items()})
                        weights.append({k: v.clone() for k, v in trainer.network.state_dict().items()})
                    finally:
                        trainer.close()
                    agent = create_agent(str(checkpoint), device="cpu")
                    self.assertEqual(agent.config.root_selection, "gumbel")
                    self.assertFalse(agent.config.add_noise)
                    report = analyze_store(checkpoint.parent.parent, agent, 100, 48, "cpp")
                    self.assertEqual(report["audit"]["disagreements"]["count"], 0, report["audit"])
                    self.assertEqual(report["replay"]["unmatched"]["count"], 0)
                    self.assertGreater(report["audit"]["checks"]["mixed_value_target"]["n"], 0)
                    self.assertIn("gumbel", report["target_statistics"]["iterations"]["1"]["by_search_mode"])
                for key in replay[0]:
                    np.testing.assert_array_equal(replay[0][key], replay[1][key])
                for key in weights[0]:
                    self.assertTrue(torch.equal(weights[0][key], weights[1][key]), key)
