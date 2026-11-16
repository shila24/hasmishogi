"""Independent diagnostics, corruption detection, and read-only CPU integration."""

import copy
import hashlib
import json
import os
from pathlib import Path
import unittest

from tests.learn._common import LearnTestCase, requires_torch


@requires_torch
class ArithmeticTests(unittest.TestCase):
    def test_wilson_and_exact_random(self):
        from hasami_learn.diagnostics import random_reference, wilson_interval

        self.assertEqual(wilson_interval(0, 0), [0.0, 1.0])
        lo, hi = wilson_interval(5, 10)
        self.assertAlmostEqual(lo, 0.2365895936154873)
        self.assertAlmostEqual(hi, 0.7634104063845127)
        result = random_reference([(1, 4), (2, 3)])
        self.assertEqual(result["n"], 2)
        self.assertAlmostEqual(result["rate"], 11 / 24)
        self.assertIsNone(result["wilson95"])
        with self.assertRaises(ValueError):
            wilson_interval(3, 2)

    def test_calibration_arithmetic_excludes_truncations(self):
        import numpy as np
        import torch
        from types import SimpleNamespace
        from hasami_learn.diagnostics import calibration_report

        agent = SimpleNamespace(device="cpu", network=lambda x: (None, torch.full((len(x),), .5)))
        sample = [{"planes": np.zeros((1, 1, 1)), "kind": kind, "ply_range": "0-39",
                   "outcome": outcome, "target": 0. if outcome is None else outcome}
                  for kind, outcome in (("win", 1.), ("loss", -1.), ("draw:repetition", 0.), ("truncated", None))]
        report = calibration_report(agent, sample, 4)
        self.assertEqual(report["overall"]["brier_score"]["n"], 3)
        self.assertAlmostEqual(report["overall"]["brier_score"]["mean"], 11 / 48)
        self.assertEqual(sum(row["n"] for row in report["reliability"]), 4)
        capped = next(r for r in report["by_outcome_and_ply"] if r["outcome_kind"] == "truncated")
        self.assertEqual(capped["n"], 1)
        self.assertIsNone(capped["brier_score"]["mean"])


@requires_torch
class TacticalTests(LearnTestCase):
    def test_hasami_win_capture_defence_and_loss(self):
        from hasami_learn.diagnostics import label_position
        from hasami_learn.games import get_spec
        from tests.test_rules import setup

        spec = get_spec("hasami")
        positions = [
            setup([38, 67, 72, 73, 74, 75], [39, 0, 1, 2, 3]),
            setup([38, 67, 72, 73, 74, 75, 76, 77, 78], [39, 37, 3, 0, 1, 2, 4]),
            setup([38, 68, 72, 73, 74, 75, 76, 77, 78], [39, 40, 37, 3, 0, 1, 2]),
        ]
        for i, position in enumerate(positions):
            game = spec.create_game("python", position=position)
            self.addCleanup(game.close)
            if i:
                game.play((67 * 81 + 40) if i == 1 else (68 * 81 + 41))
            before = (game.position, game.outcome, game.history_keys())
            labels = label_position(spec, game)
            self.assertEqual((game.position, game.outcome, game.history_keys()), before)
            if i == 0:
                self.assertIn(67 * 81 + 40, labels["win_in_one"])
                self.assertIn(67 * 81 + 40, labels["capture"])
            elif i == 1:
                self.assertIn(3 * 81 + 39, labels["forced_defence"])
                self.assertNotIn(0 * 81 + 9, labels["forced_defence"])
            else:
                self.assertTrue(labels["forced_loss"])

    def test_shogi_mate_labels_use_adapter_actions(self):
        from hasami_learn.diagnostics import label_position
        from hasami_learn.games import get_spec
        from shogi.encoding import move_label
        from shogi.notation import usi_to_move

        spec = get_spec("shogi")
        game = spec.create_game("python", position="k8/9/1GN6/1R7/9/9/9/9/4K4 b - 1")
        self.addCleanup(game.close)
        winning = move_label(game, usi_to_move(game, "8d9d"))
        labels = label_position(spec, game)
        self.assertIn(winning, labels["win_in_one"])
        self.assertIsNone(labels["capture"])

    def test_shogi_forced_defence_and_forced_loss(self):
        from hasami_learn.diagnostics import label_position
        from hasami_learn.games import get_spec
        from shogi.encoding import move_label
        from shogi.notation import usi_to_move

        spec = get_spec("shogi")
        defence = spec.create_game("python", position="kg7/9/1GN6/1R7/9/9/9/9/4K4 w - 1")
        lost = spec.create_game("python", position="k7p/9/1GN6/1R7/9/9/9/9/4K4 w - 1")
        self.addCleanup(defence.close)
        self.addCleanup(lost.close)
        labels = label_position(spec, defence)
        self.assertEqual(labels["win_in_one"], [])
        self.assertEqual(labels["forced_defence"], [move_label(defence, usi_to_move(defence, "8a9b"))])
        labels = label_position(spec, lost)
        self.assertTrue(labels["forced_loss"])
        self.assertEqual(labels["forced_defence"], [])

    def test_terminal_adjacent_root_sign_is_in_mover_frame(self):
        import numpy as np
        from hasami_learn.games import get_spec
        from hasami_learn.search.mcts import MCTS, MctsConfig
        from tests.test_rules import setup

        spec = get_spec("hasami")
        game = spec.create_game("python", position=setup([38, 67, 72, 73, 74, 75], [39, 0, 1, 2, 3]))
        self.addCleanup(game.close)
        winning_action = 67 * 81 + 40

        def evaluator(planes, masks):
            probabilities = np.zeros_like(masks)
            probabilities[:, winning_action] = 1
            return probabilities, np.full(len(planes), -1., dtype=np.float32)

        search = MCTS(evaluator, spec.make_encoder(""), MctsConfig(add_noise=False),
                      np.random.default_rng(0), spec=spec)
        self.assertEqual(search.run(game, simulations=0).root_value, -1)
        self.assertEqual(search.run(game, simulations=2).root_value, 1)
        game.play(winning_action)
        self.assertEqual(search.run(game, simulations=2).root_value, -1)

    def test_native_label_parity_and_opening_anchors(self):
        from hasami_learn.diagnostics import label_position
        from hasami_learn.games import get_spec
        from tests.test_rules import native_available, setup

        if not native_available():
            self.skipTest("Native library is not built")
        spec = get_spec("hasami")
        native = spec.create_game("cpp")
        self.addCleanup(native.close)
        self.assertEqual(len(native.legal_actions()), 63)
        count = 0
        for action in native.legal_actions():
            native.play(action)
            count += len(native.legal_actions())
            native.undo()
        self.assertEqual(count, 3717)
        fixtures = [
            ("hasami", setup([38, 67, 72, 73, 74, 75], [39, 0, 1, 2, 3])),
            ("shogi", "k8/9/1GN6/1R7/9/9/9/9/4K4 b - 1"),
        ]
        for name, initial in fixtures:
            spec = get_spec(name)
            if spec.resolve_backend("auto") != "cpp":
                if os.environ.get("HASAMI_REQUIRE_NATIVE") == "1":
                    self.fail(f"{name} native engine is required")
                self.skipTest(f"{name} native engine unavailable")
            games = [spec.create_game(backend, position=initial) for backend in ("python", "cpp")]
            for game in games:
                self.addCleanup(game.close)
            self.assertEqual(label_position(spec, games[0]), label_position(spec, games[1]))


def fingerprint(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(root).rglob("*")) if path.is_file()}


def capped_spec(supports_undo=True, cap=1):
    from hasami_learn.games.base import StepOutcome
    from hasami_learn.games.hasami import HasamiSpec

    class Capped(HasamiSpec):
        def step(self, game, actions):
            if len(game.moves) >= cap:
                raise RuntimeError("step called after adapter truncation")
            outcome = super().step(game, actions)
            if len(game.moves) >= cap and not outcome.terminal:
                return StepOutcome(False, True, outcome.rewards, {"reason": "resource_limit"})
            return outcome

    spec = Capped()
    spec.supports_undo = supports_undo
    return spec


@requires_torch
class AdapterStopTests(LearnTestCase):
    def trajectory(self, spec):
        import numpy as np
        from unittest.mock import patch
        from hasami_learn.selfplay import run_selfplay_game

        def actor(game, planes, mask, ply):
            action = int(np.flatnonzero(mask)[0])
            policy = np.zeros_like(mask)
            policy[action] = 1
            return action, policy, {}

        encoder = spec.make_encoder("")
        with patch("hasami_learn.selfplay._DRAW_STATE", {"value": -.2, "reasons": ("repetition",)}):
            trajectory = run_selfplay_game(actor, encoder, spec=spec, backend="python", max_plies=300, seed=7)
        self.assertTrue(trajectory.truncated)
        self.assertEqual(trajectory.reason, "resource_limit")
        self.assertEqual(trajectory.winner, -2)
        return trajectory, encoder

    def test_adapter_truncation_audit_and_store(self):
        from types import SimpleNamespace
        from hasami_learn.datastore import SelfplayStore
        from hasami_learn.diagnostics import analyze_store, audit_trajectory

        settings = {"max_plies": 300, "truncation_value": .375}
        for undo in (True, False):
            for cap in (1, 2):
                with self.subTest(undo=undo, cap=cap):
                    spec = capped_spec(undo, cap)
                    trajectory, encoder = self.trajectory(spec)
                    report = audit_trajectory(trajectory, spec, encoder, settings)
                    self.assertEqual(report["disagreements"]["count"], 0, report)
                    self.assertEqual(report["checks"]["value_target"]["n"], cap)
                    bad = audit_trajectory(trajectory, spec, encoder, settings, values=[-.375] * cap)
                    self.assertEqual(bad["checks"]["value_target"]["disagreements"], cap)
                    run = self.tmp / f"run-{undo}-{cap}"
                    store = SelfplayStore(run / "selfplay", spec=spec)
                    store.truncation_value = .375
                    store.add(trajectory, 1)
                    (run / "config.json").write_text(json.dumps({"selfplay": settings}))
                    agent = SimpleNamespace(spec=spec, encoder=encoder, checkpoint_dir=run,
                                            manifest=SimpleNamespace(game_params={}))
                    before = fingerprint(run)
                    result = analyze_store(run, agent, 0, 7, "python")
                    self.assertEqual(result["audit"]["disagreements"]["count"], 0, result)
                    self.assertEqual(result["audit"]["games"]["n"], 1)
                    self.assertEqual(fingerprint(run), before)

    def test_history_after_adapter_truncation_is_rejected_before_stepping(self):
        from hasami_learn.diagnostics import audit_trajectory

        spec = capped_spec()
        trajectory, encoder = self.trajectory(spec)
        trajectory.joint_actions.append({1: 0 * 81 + 9})
        trajectory.plies += 1
        with self.assertRaisesRegex(ValueError, "history continues after"):
            audit_trajectory(trajectory, spec, encoder, {"max_plies": 300})

    def test_labels_honor_adapter_stop_with_undo_and_clone(self):
        from hasami_learn.diagnostics import label_position

        for undo in (True, False):
            with self.subTest(undo=undo):
                spec = capped_spec(undo)
                game = spec.create_game("python")
                self.addCleanup(game.close)
                before = (game.position, game.outcome, game.history_keys(), list(game.moves))
                labels = label_position(spec, game)
                self.assertEqual(len(labels["legal"]), 63)
                self.assertEqual(labels["win_in_one"], [])
                self.assertEqual(labels["forced_defence"], [])
                self.assertFalse(labels["forced_loss"])
                self.assertEqual((game.position, game.outcome, game.history_keys(), list(game.moves)), before)

    def test_generation_never_samples_or_steps_past_adapter_stop(self):
        from hasami_learn.diagnostics import generate_positions

        for undo in (True, False):
            with self.subTest(undo=undo):
                sample, seen = generate_positions(capped_spec(undo), 10, 7, "python")
                try:
                    self.assertEqual(seen, 3)
                    self.assertEqual(len(sample), 3)
                    self.assertEqual({meta["generator"] for _, meta in sample}, {"random", "greedy", "mixed"})
                    for game, meta in sample:
                        self.assertEqual(list(game.moves), [])
                        self.assertEqual(meta["ply"], 0)
                finally:
                    for game, _ in sample:
                        game.close()


@requires_torch
class AuditTests(LearnTestCase):
    def test_mixed_targets_independent_audit_and_corruption_class(self):
        from hasami_learn.diagnostics import audit_trajectory

        trajectory, spec, encoder = self.trajectory([72 * 81 + 63, 9])
        settings = {"max_plies": 2, "truncation_value": .4, "value_target_mix": .5}
        for extra, root in zip(trajectory.extras, (.8, -.6)):
            extra.update(root_value=root, value_target_mix=.5, outcome_value=.4)
        report = audit_trajectory(trajectory, spec, encoder, settings)
        self.assertEqual(report["disagreements"]["count"], 0, report)
        self.assertEqual(report["checks"]["mixed_value_target"]["n"], 2)
        report = audit_trajectory(trajectory, spec, encoder, settings, values=[.4, .4])
        self.assertEqual(report["checks"]["mixed_value_target"]["disagreements"], 2)
        self.assertEqual(report["checks"]["value_target"]["disagreements"], 0)
        saved = trajectory.values(.4)
        trajectory.extras[1]["root_value"] *= -1
        report = audit_trajectory(trajectory, spec, encoder, settings, values=saved)
        self.assertEqual(report["checks"]["mixed_value_target"]["disagreements"], 1)
        trajectory.extras[1]["value_target_mix"] = 0
        report = audit_trajectory(trajectory, spec, encoder, settings, values=saved)
        self.assertEqual(report["checks"]["value_target_mix"]["disagreements"], 1)

    def test_archive_root_rounding_bound_and_mode_statistics(self):
        import numpy as np
        from hasami_learn.diagnostics import mixed_target, IterationStats, Audit, ReplaySample
        from hasami_learn.games import get_spec
        from hasami_learn.replay import ReplayBuffer, fields_for

        root = .123456789
        expected, tolerance = mixed_target(.4, np.float16(root), .5)
        self.assertLess(abs(expected - (.2 + .5 * root)), tolerance)
        self.assertLess(tolerance, .00004)
        self.assertEqual(mixed_target(.4, root, .5)[1], 1e-6)
        for bad in (np.float16(65504), float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                mixed_target(.4, bad, .5)
        spec = get_spec()
        encoder = spec.make_encoder()
        game = spec.create_game("python")
        self.addCleanup(game.close)
        planes, mask = encoder.encode(game), encoder.legal_mask(game)
        policy = mask / mask.sum()
        replay = ReplayBuffer(2, fields_for(spec, encoder.num_planes, "pv"), np.random.default_rng(48))
        replay.add(planes=planes, mask=mask, policy=policy, value=.2 + .5 * root)
        replay.add(planes=planes, mask=mask, policy=policy, value=.2 + .5 * root + .01)
        path = self.tmp / "mixed.npz"
        replay.save(path)
        sample = ReplaySample(path, spec, encoder, 2, 48)
        sample.match(spec, planes, mask, policy, expected, "mixed", 0, tolerance, mixed=True)
        audit = Audit()
        sample.report(audit)
        self.assertEqual(audit.checks["mixed_value_target"]["n"], 2)
        self.assertEqual(audit.checks["mixed_value_target"]["disagreements"], 1)
        stats = IterationStats()
        stats.decision(np.ones(2), np.array([.5, .5]), "puct")
        stats.decision(np.ones(2), np.array([1., 0.]), "gumbel")
        modes = stats.report()["by_search_mode"]
        self.assertAlmostEqual(modes["puct"]["policy_entropy_nats"]["mean"], np.log(2))
        self.assertEqual(modes["gumbel"]["policy_top1_mass"]["mean"], 1.)

    def trajectory(self, moves, initial=None):
        import numpy as np
        from hasami_learn.games import get_spec
        from hasami_learn.selfplay import Trajectory

        spec = get_spec("hasami")
        encoder = spec.make_encoder("")
        game = spec.create_game("python", position=initial)
        self.addCleanup(game.close)
        result = Trajectory(draw_value=-.2, draw_reasons=("repetition",))
        for ply, action in enumerate(moves):
            mover = game.position.turn
            mask = encoder.legal_mask(game)
            policy = np.zeros(spec.action_size, dtype=np.float32)
            policy[action] = 1
            result.append(encoder.encode(game), mask, action, mover, policy, {"decision_ply": ply})
            result.joint_actions.append({mover: action})
            spec.step(game, {mover: action})
        result.plies = len(moves)
        result.winner = game.outcome.winner
        result.truncated = not game.outcome.terminal
        result.reason = "max_plies" if result.truncated else game.outcome.reason
        result.rewards = [0., 0.] if result.truncated else spec.terminal_values(game)
        return result, spec, encoder

    def test_corrupt_targets_mover_mask_planes_and_draw_are_classified(self):
        from hasami_learn.diagnostics import audit_trajectory
        from tests.test_rules import setup

        initial = setup([38, 67, 72, 73, 74, 75], [39, 0, 1, 2, 3])
        trajectory, spec, encoder = self.trajectory([67 * 81 + 40], initial)
        clean = audit_trajectory(trajectory, spec, encoder, {"max_plies": 300}, initial=initial)
        self.assertEqual(clean["disagreements"]["count"], 0, clean)
        for kind in ("value_target", "mover", "mask", "planes"):
            bad = copy.deepcopy(trajectory)
            values = None
            if kind == "value_target":
                values = -bad.values()
            elif kind == "mover":
                bad.players[0] = 1
            elif kind == "mask":
                bad.masks[0][bad.actions[0]] = 0
            else:
                bad.planes[0][0, 0, 0] = 1 - bad.planes[0][0, 0, 0]
            report = audit_trajectory(bad, spec, encoder, {"max_plies": 300}, initial=initial, values=values)
            self.assertEqual(report["checks"][kind]["disagreements"], 1, report)
            self.assertEqual(report["checks"][kind]["examples"][0]["ply"], 0)
        cycle = [72 * 81 + 63, 0 * 81 + 9, 63 * 81 + 72, 9 * 81 + 0]
        drawn, spec, encoder = self.trajectory(cycle * 3)
        self.assertEqual(audit_trajectory(drawn, spec, encoder, {})["disagreements"]["count"], 0)
        drawn.draw_value = .2
        report = audit_trajectory(drawn, spec, encoder, {})
        self.assertEqual(report["checks"]["draw_value"]["disagreements"], 1)
        self.assertEqual(report["checks"]["value_target"]["disagreements"], 12)

    def test_truncation_and_unlisted_draw_reason(self):
        from hasami_learn.diagnostics import audit_trajectory, expected_target

        trajectory, spec, encoder = self.trajectory([72 * 81 + 63, 0 * 81 + 9])
        settings = {"max_plies": 2, "truncation_value": .375}
        self.assertEqual(audit_trajectory(trajectory, spec, encoder, settings)["disagreements"]["count"], 0)
        wrong = audit_trajectory(trajectory, spec, encoder, settings, values=[.375, -.375])
        self.assertEqual(wrong["checks"]["value_target"]["disagreements"], 1)
        self.assertEqual(expected_target({"winner": -1, "reason": "no_moves", "truncated": False},
                                         0, settings, 1, spec), 0)

    def test_concurrent_eviction_during_archive_snapshot(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from hasami_learn.datastore import SelfplayStore
        from hasami_learn.diagnostics import analyze_store

        trajectory, spec, encoder = self.trajectory([72 * 81 + 63])
        store = SelfplayStore(self.tmp / "selfplay", spec=spec)
        paths = [store.add(trajectory, 1), store.add(trajectory, 1)]
        (self.tmp / "config.json").write_text(json.dumps({"selfplay": {"max_plies": 1}}))
        agent = SimpleNamespace(spec=spec, encoder=encoder, checkpoint_dir=self.tmp,
                                manifest=SimpleNamespace(game_params={}))
        before = fingerprint(self.tmp)
        original_stat = Path.stat
        for missing in (paths[:1], paths):
            with self.subTest(missing=len(missing)):
                def evicted_stat(path, *args, **kwargs):
                    if path in missing:
                        raise FileNotFoundError(f"simulated concurrent eviction: {path.name}")
                    return original_stat(path, *args, **kwargs)

                with patch.object(Path, "stat", evicted_stat):
                    report = analyze_store(self.tmp, agent, 0, 7, "python")
                self.assertEqual(report["status"], "measured")
                self.assertEqual(report["audit"]["games"]["n"], len(paths) - len(missing))
                self.assertEqual(report["audit"]["disagreements"]["count"], 0)
                for path in missing:
                    self.assertTrue(any(path.name in reason and "FileNotFoundError" in reason
                                        for reason in report["audit"]["not_available"]))
                json.dumps(report, allow_nan=False)
                self.assertEqual(fingerprint(self.tmp), before)

    def test_replay_exact_bytes_symmetry_corruption_and_unused_capacity(self):
        import numpy as np
        from hasami_learn.diagnostics import Audit, ReplaySample

        trajectory, spec, encoder = self.trajectory([72 * 81 + 63])
        p, m, policy = trajectory.planes[0], trajectory.masks[0], trajectory.policies[0]
        symmetry, = spec.symmetries()
        arrays = {
            "planes": np.stack([p, symmetry.planes(p), p, p]),
            "mask": np.stack([m, symmetry.policy(m), m, m]).astype(np.uint8),
            "policy": np.stack([policy, symmetry.policy(policy), policy, policy]).astype(np.float16),
            "value": np.array([.25, .25, -.25, 999], dtype=np.float32),
        }
        path = self.tmp / "replay.npz"
        np.savez_compressed(path, __size=3, **arrays)
        before = path.read_bytes()
        sample = ReplaySample(path, spec, encoder, 10, 0)
        sample.match(spec, p, m, policy, .25, "game", 0)
        audit = Audit()
        report = sample.report(audit)
        self.assertEqual(report["sample"]["n"], 3)
        self.assertEqual(report["matched"]["count"], 3)
        self.assertGreaterEqual(report["symmetry_matches"]["count"], 1)
        self.assertEqual(audit.checks["replay_value"]["disagreements"], 1)
        self.assertEqual(path.read_bytes(), before)
        arrays["mask"][0][0] = 1
        np.savez_compressed(path, __size=1, **arrays)
        sample = ReplaySample(path, spec, encoder, 10, 0)
        sample.match(spec, p, m, policy, .25, "game", 0)
        self.assertEqual(sample.report(Audit())["unmatched"]["count"], 1)
        path.write_bytes(b"broken archive")
        self.assertEqual(ReplaySample(path, spec, encoder, 10, 0).status, "not_available")

    def test_ambiguous_replay_provenance_is_not_assumed_unique(self):
        import numpy as np
        from hasami_learn.diagnostics import Audit, ReplaySample

        trajectory, spec, encoder = self.trajectory([72 * 81 + 63])
        p, m, policy = trajectory.planes[0], trajectory.masks[0], trajectory.policies[0]
        path = self.tmp / "replay.npz"
        np.savez_compressed(path, __size=1, planes=[p], mask=np.asarray([m], dtype=np.uint8),
                            policy=np.asarray([policy], dtype=np.float16), value=[1.])
        sample = ReplaySample(path, spec, encoder, 1, 0)
        for z in (-1, 1):
            sample.match(spec, p, m, policy, z, "game", 0)
        audit = Audit()
        self.assertEqual(sample.report(audit)["ambiguous"]["count"], 1)
        self.assertEqual(audit.checks["replay_value"]["n"], 0)


@requires_torch
class IntegrationTests(LearnTestCase):
    def test_clean_tiny_runs_determinism_and_read_only(self):
        import torch
        from hasami_learn.diagnostics import analyze_store, diagnose, validate_output_path
        from hasami_learn.games import get_spec
        from hasami_learn.inference import create_agent
        from hasami_learn.trainer import train_from_config

        torch.set_num_threads(1)
        for game_id in ("hasami", "shogi"):
            cfg = self.tiny("alphazero", simulations=2, batch_games=2)
            cfg.run.run_id = "diagnostics-" + game_id
            cfg.game.name = game_id
            cfg.game.params = {"max_moves": 4} if game_id == "shogi" else {}
            cfg.encoder.name = get_spec(game_id).default_encoder
            cfg.run.backend = get_spec(game_id).resolve_backend("auto")
            if os.environ.get("HASAMI_REQUIRE_NATIVE") == "1":
                self.assertEqual(cfg.run.backend, "cpp", f"native {game_id} required")
            cfg.run.iterations = 1
            cfg.run.eval_every = 0
            cfg.selfplay.max_plies = 4
            cfg.selfplay.truncation_value = .25
            cfg.selfplay.draw_reasons = list(get_spec(game_id).draw_reasons)
            cfg.selfplay.symmetry_augmentation = True
            cfg.datastore.enabled = True
            cfg.replay.persist = True
            cfg.replay.capacity = 32
            cfg.replay.min_size = 1
            cfg.replay.batch_size = 4
            cfg.optimizer.updates_per_iteration = 1
            checkpoint = train_from_config(cfg, echo=False)
            run = checkpoint.parent.parent
            # A torn journal and a staging file must not trigger reader-side recovery.
            with (run / "selfplay/index.jsonl").open("ab") as journal:
                journal.write(b'{"torn":')
            (run / "selfplay/games/.selfplay-test.tmp").write_bytes(b"do not remove")
            before = fingerprint(run)
            agent = create_agent(str(checkpoint), device="cpu", mode="policy")
            report = analyze_store(run, agent, 32, 3, cfg.run.backend)
            # STOP on a clean pipeline disagreement: never weaken the audit to pass this check.
            self.assertEqual(report["audit"]["disagreements"]["count"], 0,
                             "STOP: clean pipeline disagreement: " + json.dumps(report["audit"]))
            self.assertEqual(report["replay"]["unmatched"]["count"], 0, report["replay"])
            self.assertEqual(report["audit"]["decisions"]["n"], 8)
            self.assertEqual(report["calibration"]["overall"]["n"], 8)
            self.assertEqual(report["calibration"]["overall"]["brier_score"]["n"],
                             8 if game_id == "shogi" else 0)
            first = diagnose(checkpoint, run, "cpu", 3, [0, 2], 17)
            second = diagnose(checkpoint, run, "cpu", 3, [0, 2], 17)
            self.assertEqual(first, second)
            self.assertEqual(fingerprint(run), before)
            for out in (run / "new.json", checkpoint / "manifest.json", run / "replay.npz"):
                with self.assertRaises(ValueError):
                    validate_output_path(out, checkpoint, run)
            alias = self.tmp / (game_id + "-alias")
            alias.symlink_to(run, target_is_directory=True)
            with self.assertRaises(ValueError):
                validate_output_path(alias / "new.json", checkpoint, run)


if __name__ == "__main__":
    unittest.main()
