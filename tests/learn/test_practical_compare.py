"""Joint practical learning and matched E8 comparison regression tests."""

from __future__ import annotations

import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.learn._common import requires_torch

ROOT = Path(__file__).resolve().parents[2]


@requires_torch
class JointLossTests(unittest.TestCase):
    def test_head_matching_includes_100_elo_boundary_and_requires_coverage(self):
        from hasami_learn.practical_eval import match_head_levels
        reference = dict(levels=[dict(level=1, elo_center=0)])
        for centre, expected in ((-100, 1), (100, 1), (-100.001, None), (100.001, None)):
            with self.subTest(centre=centre):
                training = dict(train_styles=["qre", "temperature"], levels=[dict(
                    level=1, elo_center=centre,
                    styles={"qre": dict(available=True), "temperature": dict(available=True)})])
                self.assertEqual(match_head_levels(reference, training)[0]["head_level"], expected)
        training["levels"] = [dict(level=index, elo_center=centre,
            styles={"qre": dict(available=True), "temperature": dict(available=index != 2)})
            for index, centre in enumerate((-50, 0, 50), 1)]
        # The exact centre lacks coverage; tied covered centres pick the lower index.
        self.assertEqual(match_head_levels(reference, training)[0]["head_level"], 1)
        del training["levels"][0]["styles"]["temperature"]
        self.assertEqual(match_head_levels(reference, training)[0]["head_level"], 3)

    def test_joint_loss_targets_and_masked_cells_receive_gradient(self):
        import torch
        from hasami_learn.algorithms.practical import joint_loss

        policy = torch.zeros((1, 3), requires_grad=True)
        value = torch.zeros(1, requires_grad=True)
        practical = torch.zeros((1, 2, 2), requires_grad=True)
        slip = torch.zeros((1, 2), requires_grad=True)
        az = dict(mask=torch.tensor([[1., 1., 0.]]), policy=torch.tensor([[0.25, 0.75, 0.]]),
                  value=torch.tensor([1.]))
        proxy = dict(own=torch.tensor([0]), other=torch.tensor([1]),
                     target=torch.tensor([1.]), slip=torch.tensor([1.]))
        loss, terms = joint_loss(policy, value, az, practical, slip, proxy, slip_weight=0.5)
        self.assertAlmostEqual(float(loss.detach()), float((terms["policy"] + terms["value"] +
            terms["practical"] + 0.5 * terms["slip"]).detach()))
        loss.backward()
        self.assertGreater(float(policy.grad[0, 0]), 0)
        self.assertLess(float(policy.grad[0, 1]), 0)
        self.assertEqual(float(policy.grad[0, 2]), 0)
        self.assertLess(float(value.grad[0]), 0)
        self.assertEqual(torch.nonzero(practical.grad).tolist(), [[0, 0, 1]])
        self.assertEqual(torch.nonzero(slip.grad).tolist(), [[0, 0]])

    def test_default_remains_engine_and_joint_rejects_pretrained_inputs(self):
        from hasami_learn.algorithms.practical import PracticalParams
        self.assertEqual(PracticalParams().init, "engine")
        for kwargs in ({"engine_checkpoint": "engine"}, {"levels_file": "levels.json"},
                       {"bootstrap_tiny": True}, {"recalibrate_every": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PracticalParams(init="joint", **kwargs)

    def test_compute_uses_metrics_device_and_checkpoint_iteration(self):
        from hasami_learn.practical_eval import training_compute
        with tempfile.TemporaryDirectory(prefix="compare-compute-") as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoints" / "iter_000002"
            checkpoint.mkdir(parents=True)
            (checkpoint / "manifest.json").write_text(json.dumps(dict(iteration=2)))
            rows = [dict(iteration=1, kind="collect", seconds=20, device="cpu"),
                    dict(iteration=2, kind="update", seconds=3600, device="cuda:0"),
                    dict(iteration=2, kind="checkpoint", seconds=100, device="cpu"),
                    dict(iteration=3, kind="update", seconds=1000, device="cuda:0")]
            (root / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            result = training_compute(checkpoint)
            self.assertEqual(result["wall_clock_seconds"], 3620)
            self.assertEqual(result["gpu_hours"], 1)
            self.assertEqual(result["device_seconds"], {"cpu": 20, "cuda:0": 3600})


@requires_torch
class PracticalComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        from hasami_learn.algorithms.practical import PracticalLearner
        from hasami_learn.config import TrainingConfig
        from hasami_learn.practical_calibrate import calibrate
        from hasami_learn.trainer import Trainer, train_from_config

        cls.addClassCleanup(random.setstate, random.getstate())
        cls.addClassCleanup(np.random.set_state, np.random.get_state())
        cls.addClassCleanup(torch.set_rng_state, torch.get_rng_state())
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        temporary = tempfile.TemporaryDirectory(prefix="practical-compare-")
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        cls.raw = json.loads((ROOT / "configs/tiny/practical-joint-hasami.json").read_text())
        cls.raw["run"]["output_dir"] = str(cls.root)
        cls.events = []
        recalibrate, collect_az, proxy = PracticalLearner._recalibrate, PracticalLearner._collect_alphazero, PracticalLearner._proxy
        def calibration(learner, iteration):
            cls.events.append(("calibration", iteration))
            return recalibrate(learner, iteration)
        def az(learner, iteration):
            cls.events.append(("alphazero", iteration))
            return collect_az(learner, iteration)
        def current_proxy(learner, level, seed):
            cls.events.append(("proxy", learner.calibration_iteration))
            result = proxy(learner, level, seed)
            if result.policy_engine.network is not learner.network:
                raise AssertionError("joint proxy must use the current network")
            return result
        trainer = Trainer(TrainingConfig.from_dict(copy.deepcopy(cls.raw)), echo=False)
        cls.initial = {key: value.clone() for key, value in trainer.network.state_dict().items()}
        cls.learner = trainer.learner
        try:
            with patch.object(PracticalLearner, "_recalibrate", calibration), patch.object(
                    PracticalLearner, "_collect_alphazero", az), patch.object(PracticalLearner, "_proxy", current_proxy):
                cls.joint = trainer.run()
        finally:
            trainer.close()
        engine_raw = copy.deepcopy(cls.raw)
        engine_raw["run"].update(run_id="reference-engine", iterations=1, eval_every=0)
        engine_raw["algorithm"] = dict(name="alphazero", params=dict(simulations=2, batch_games=2))
        engine_raw["network"]["params"].pop("aux_heads")
        engine_raw["evaluation"]["games"] = 0
        trainer = Trainer(TrainingConfig.from_dict(engine_raw), echo=False)
        try:
            # Keep this smoke fixture away from E1's decided-position exclusion.
            with torch.no_grad():
                for parameter in trainer.network.parameters():
                    parameter.mul_(0.02)
            cls.engine = trainer.run()
        finally:
            trainer.close()
        calibrate(cls.engine, output=cls.engine / "levels.json", k=1, tiny_budget=True, smoke=True,
                  train_styles=("qre", "temperature"), heldout_styles=("alphabeta",))
        b_raw = copy.deepcopy(cls.raw)
        b_raw["run"].update(run_id="method-b", iterations=1, eval_every=0)
        b_raw["algorithm"]["params"] = dict(engine_checkpoint=str(cls.engine),
            levels_file=str(cls.engine / "levels.json"), default_level=1, games_per_iteration=2,
            batch_games=2, on_policy_rate=0)
        b_raw["evaluation"]["games"] = 0
        cls.method_b = train_from_config(TrainingConfig.from_dict(b_raw), echo=False)

    def test_joint_alternation_recalibration_and_live_network(self):
        import torch
        from hasami.agents import SearchLimits
        from hasami_learn.inference import MctsAgent

        self.assertEqual([e for e in self.events if e[0] == "calibration"], [("calibration", 0), ("calibration", 2)])
        self.assertEqual([e for e in self.events if e[0] == "alphazero"], [("alphazero", 1), ("alphazero", 2)])
        self.assertEqual(self.events[0], ("calibration", 0))
        self.assertEqual(self.events[1], ("alphazero", 1))
        self.assertEqual(self.events[-1], ("calibration", 2))
        az2 = self.events.index(("alphazero", 2))
        self.assertTrue(all(e[0] == "proxy" for e in self.events[2:az2]))
        self.assertTrue(all(e[0] == "proxy" for e in self.events[az2 + 1:-1]))
        self.assertIs(self.learner.engine.network, self.learner.network)
        self.assertIsInstance(self.learner.make_agent("eval"), MctsAgent)
        self.assertTrue(any(not torch.equal(self.initial[key], value)
                            for key, value in self.learner.network.state_dict().items()))
        game = self.learner.spec.create_game("cpp")
        try:
            self.assertEqual(len(game.legal_actions()), 63)
            self.assertEqual(game.perft(2), 3717)
            action = self.learner.make_agent("engine").choose(game, SearchLimits()).action
            self.assertIs(type(action), int)
            self.assertIn(action, game.legal_actions())
        finally:
            game.close()
        metrics = [json.loads(line) for line in (self.joint.parent.parent / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual(len([row for row in metrics if row["kind"] == "evaluation"]), 2)
        for row in (row for row in metrics if row["kind"] == "collect"):
            self.assertEqual(row["extra"]["alphazero_games"], 2)
            self.assertEqual(row["extra"]["proxy_games"], 2)
            self.assertEqual(row["truncated"], 4)
            self.assertEqual(row["wins"], [0, 0, 0])
            self.assertTrue(row["extra"]["smoke"])

    def test_checkpoint_bundle_roundtrip_and_resume(self):
        import numpy as np
        import torch
        from hasami_learn.checkpoint_bundle import bundle_checkpoint, sha256_file
        from hasami_learn.config import TrainingConfig
        from hasami_learn.manifest import load_checkpoint
        from hasami_learn.practical import create_advisor
        from hasami_learn.trainer import Trainer

        bundle = self.root / "joint-bundle"
        bundle_checkpoint(self.joint, bundle)
        advisor = create_advisor(str(bundle), "cpu", method="learned")
        self.assertEqual(sha256_file(bundle / "weights.pt"), sha256_file(bundle / "engine/weights.pt"))
        self.assertEqual(advisor.level_data["engine_sha256"], sha256_file(bundle / "weights.pt"))
        self.assertIn("Calibration iteration=2", advisor.level_data["notes"])
        game = advisor.spec.create_game("cpp")
        try:
            before = (game.position, game.history_keys())
            self.assertIn(advisor.advise(game, 1)[0].action, game.legal_actions())
            self.assertEqual(before, (game.position, game.history_keys()))
        finally:
            game.close()
        first = self.joint.parent / "iter_000001"
        loaded = load_checkpoint(first, "cpu")
        samples = loaded.learner_state["extra"]["az_samples"]
        self.assertTrue(samples)
        for row in samples:
            self.assertAlmostEqual(float(row["policy"].sum()), 1)
            self.assertTrue(torch.all(row["policy"][row["mask"] == 0] == 0))
            self.assertEqual(row["value"], 0.0)
        raw = copy.deepcopy(self.raw)
        raw["run"].update(run_id="resumed-joint")
        trainer = Trainer(TrainingConfig.from_dict(raw), resume=first, echo=False)
        try:
            self.assertEqual(trainer.learner.calibration_iteration, 0)
            self.assertEqual(len(trainer.learner.samples), 8)
            self.assertEqual(len(trainer.learner.az_samples), len(samples))
            for actual, expected in zip(trainer.learner.az_samples, samples):
                np.testing.assert_array_equal(actual["policy"], expected["policy"].numpy())
            resumed = trainer.run()
        finally:
            trainer.close()
        resumed_model = load_checkpoint(resumed, "cpu")
        full = load_checkpoint(self.joint, "cpu")
        for name, value in full.network.state_dict().items():
            self.assertTrue(torch.equal(value, resumed_model.network.state_dict()[name]), name)

    def test_e8_matched_positions_proxies_seeds_and_omitted_arms(self):
        from hasami_learn.practical import validate_artifact
        from hasami_learn import practical_eval as evaluation

        calls = []
        paired = evaluation.paired_intervention
        def observed(advisor, game, levels, actions, seeds, **kwargs):
            traces = {}
            result = paired(advisor, game, levels, actions, seeds, traces=traces, **kwargs)
            calls.append(dict(advisor=advisor, history=game.history_keys(), levels=levels,
                              actions=dict(actions), seeds=list(seeds), traces=traces, style=kwargs["style"]))
            return result
        kwargs = dict(checkpoint=str(self.engine), output=self.root / "compare.json", level=1,
                      positions=1, rollouts=1, max_plies=1, reference_simulations=2, guard=0, smoke=True)
        with patch.object(evaluation, "paired_intervention", side_effect=observed):
            report = evaluation.run_comparison(b_checkpoint=str(self.method_b), c_checkpoint=str(self.joint), **kwargs)
        validate_artifact(report, "eval")
        expected = ["own", "engine", "A_depth2", "A_depth4", "B", "C"]
        self.assertEqual(report["config"]["arms"], expected)
        self.assertTrue(calls)
        self.assertEqual({row["style"] for row in calls}, {"mixture", "qre", "temperature", "alphabeta"})
        for call in calls:
            self.assertEqual(list(call["actions"]), expected)
            self.assertEqual(call["history"], calls[0]["history"])
            self.assertEqual(call["seeds"], calls[0]["seeds"])
            self.assertEqual(call["levels"], (1, 1))
            self.assertIs(call["advisor"], calls[0]["advisor"])
            for seed in call["seeds"]:
                traces = [call["traces"][(arm, seed)] for arm in expected]
                self.assertTrue(all(trace == traces[0] for trace in traces))
        self.assertTrue(report["smoke"])
        self.assertTrue((self.root / "compare.done.json").is_file())
        metrics = {(row.get("arm"), row["metric"]) for row in report["results"]}
        for arm in expected:
            for metric in ("score_difference", "tag_team_score_rho=0.1", "error_share", "value_loss",
                           "advice_latency_p95", "training_wall_clock_seconds"):
                self.assertIn((arm, metric), metrics)
        self.assertIn(("C", "own_engine_score"), metrics)
        self.assertEqual(report["config"]["arm_compute_sources"]["B"], ["engine", "B"])
        kwargs["output"] = self.root / "compare-omitted.json"
        omitted = evaluation.run_experiment("E8", **kwargs)
        self.assertEqual(omitted["config"]["arms"], expected[:4])
        self.assertFalse(any(row.get("arm") in ("B", "C") for row in omitted["results"]))
        self.assertEqual(len(omitted["checkpoints"]), 1)

    def test_compute_missing_metrics_and_smoke_and_arm_validation(self):
        from hasami_learn.practical_eval import training_compute, run_comparison
        from hasami_learn.checkpoint_bundle import bundle_checkpoint
        from hasami_learn.practical_calibrate import calibrate

        source = training_compute(self.joint)
        self.assertGreater(source["wall_clock_seconds"], 0)
        self.assertEqual(set(source["device_seconds"]), {"cpu"})
        isolated = self.root / "isolated-engine"
        bundle_checkpoint(self.engine, isolated)
        self.assertIsNone(training_compute(isolated)["wall_clock_seconds"])
        with self.assertRaisesRegex(ValueError, "smoke"):
            run_comparison(str(self.engine), output=self.root / "not-written.json")
        with self.assertRaisesRegex(ValueError, "init=joint"):
            run_comparison(str(self.engine), c_checkpoint=str(self.method_b), smoke=True,
                           output=self.root / "not-written.json")
        with self.assertRaisesRegex(ValueError, "share the common engine"):
            run_comparison(str(self.joint), b_checkpoint=str(self.method_b), smoke=True,
                           output=self.root / "not-written.json")
        with self.assertRaisesRegex(ValueError, "k=1"):
            calibrate(self.engine, output=self.root / "not-written.json", tiny_budget=True, smoke=True, k=2)

    def comparison_fixture(self, centres, unavailable=()):
        import numpy as np
        from hasami_learn.practical import PracticalAdvisor, Evaluation, create_advisor, validate_levels
        from hasami_learn.practical_search import SearchAdvisor

        reference = create_advisor(str(self.engine), "cpu", method="search")
        learned = create_advisor(str(self.joint), "cpu", method="learned")

        def ladder(source, values):
            data = copy.deepcopy(source.level_data)
            data["k"] = len(values)
            data["levels"] = [copy.deepcopy(data["levels"][0]) for _ in values]
            for index, (row, centre) in enumerate(zip(data["levels"], values), 1):
                row.update(level=index, elo_center=centre)
                for style, entry in row["styles"].items():
                    entry.update(available=True, elo=centre, ci=[centre - 50, centre + 50])
                    if (index, style) in unavailable and source is learned:
                        entry.update(available=False, elo=None, ci=None)
            return validate_levels(data)

        common = SearchAdvisor(reference.engine, ladder(reference, [0, 300]),
                               checkpoint_root=reference.checkpoint_root)
        matrix = np.arange(len(centres)**2).reshape(len(centres), len(centres)) / 10
        slips = np.arange(1, len(centres) + 1) / 10
        heads = SimpleNamespace(evaluate=lambda games, cancel=None: [
            Evaluation({}, 0, matrix.copy(), slips.copy()) for _ in games])
        model = PracticalAdvisor(learned.engine, heads, ladder(learned, centres),
                                 checkpoint_root=learned.checkpoint_root)
        for target, source in ((common, reference), (model, learned)):
            for name in ("manifest", "checkpoint_dir", "engine_checkpoint", "levels_path"):
                setattr(target, name, getattr(source, name))
        return common, model

    def compare_fixture(self, reference, model, **kwargs):
        from hasami_learn import practical_eval as evaluation
        with patch.object(evaluation, "create_advisor", side_effect=[reference, model]):
            return evaluation.run_comparison(str(self.engine), c_checkpoint=str(self.joint),
                output=self.root / "compare-levels.json", level=1, opponent_level=2,
                positions=1, rollouts=1, max_plies=1, reference_simulations=2, guard=0,
                smoke=True, **kwargs)

    def assert_missing_learned_arm(self, report):
        from hasami_learn.practical import validate_artifact
        validate_artifact(report, "eval")
        self.assertTrue(report["config"]["matched_cases"])
        self.assertTrue(all("C" not in case["actions"] for case in report["config"]["matched_cases"]))
        missing = [row for row in report["results"] if row.get("arm") == "C" and row["metric"] == "missing_level"]
        self.assertTrue(missing)
        self.assertTrue(all(row["value"] is None and row["n"] == 0 for row in missing))
        self.assertTrue(report["config"]["missing_arms"])
        self.assertIn("missing", report["notes"].lower())
        self.assertFalse(any(row.get("arm") == "C" and row.get("level") is not None
                             and row["value"] is not None for row in report["results"]))

    def test_e8_excludes_centres_shifted_by_1000_elo(self):
        reference, model = self.comparison_fixture([1000, 1300])
        with patch.object(model.practical, "evaluate", side_effect=AssertionError("unmatched head used")):
            self.assert_missing_learned_arm(self.compare_fixture(reference, model))

    def test_e8_excludes_unavailable_training_style_on_either_axis(self):
        for index in (1, 2):
            with self.subTest(index=index):
                reference, model = self.comparison_fixture([0, 300], unavailable=[(index, "temperature")])
                with patch.object(model.practical, "evaluate", side_effect=AssertionError("untrained cell used")):
                    self.assert_missing_learned_arm(self.compare_fixture(reference, model))

    def test_e8_remaps_both_head_axes_and_slip_by_elo(self):
        from hasami_learn.practical import PracticalAdvisor, validate_artifact
        reference, model = self.comparison_fixture([-300, 50, 350])
        observed = []
        advise = PracticalAdvisor.advise
        def capture(advisor, *args, **kwargs):
            rows = advise(advisor, *args, **kwargs)
            observed.extend(rows)
            return rows
        with patch.object(PracticalAdvisor, "advise", capture):
            report = self.compare_fixture(reference, model)
        validate_artifact(report, "eval")
        mapping = report["config"]["head_level_matches"]["C"]
        self.assertEqual([row["head_level"] for row in mapping], [2, 3])
        self.assertTrue(observed)
        # After our move, the opponent is the mover: head [2, 1], not [1, 0].
        for row in observed:
            self.assertAlmostEqual(row.practical_score, 0.3)
            self.assertAlmostEqual(row.difficulty, 0.2)
        slips = [row for row in report["results"] if row.get("arm") == "C" and row["metric"] == "predicted_slip"]
        self.assertTrue(slips)
        self.assertTrue(all(abs(row["value"] - 0.2) < 1e-6 for row in slips))
        self.assertEqual(model.level_data["levels"][0]["elo_center"], -300)

    def test_e8_keeps_original_sampling_seed_after_omission(self):
        from hasami_learn import practical_eval as evaluation
        from hasami_learn.games.base import StepOutcome
        sampled_positions, decision_rng = evaluation.sampled_positions, evaluation.decision_rng
        observed = []

        def filtered(advisor, levels, seed, **kwargs):
            step = advisor.spec.step
            calls = []
            def stop_first(game, actions):
                result = step(game, actions)
                calls.append(game)
                return StepOutcome(False, True) if len(calls) == 1 else result
            def fixed_ply(seed, player, ply, stream=0):
                return SimpleNamespace(integers=lambda *args: 1) if stream == 5 else decision_rng(seed, player, ply, stream)
            proxy = SimpleNamespace(distribution=lambda game: {game.legal_actions()[0]: 1.0})
            with patch.object(advisor.spec, "step", side_effect=stop_first), patch.object(
                    evaluation, "decision_rng", side_effect=fixed_ply), patch.object(
                    evaluation, "_agents", return_value=[proxy, proxy]):
                retained = sampled_positions(advisor, levels, seed, **kwargs)
            observed.append(seed)
            self.assertEqual(len(retained), 1)
            return retained

        with patch.object(evaluation, "sampled_positions", side_effect=filtered):
            report = evaluation.run_comparison(str(self.engine), output=self.root / "filtered-seeds.json",
                seed=700, level=1, positions=2, rollouts=1, max_plies=2, reference_simulations=2, guard=0, smoke=True)
        case, = report["config"]["matched_cases"]
        self.assertEqual(case["position_seed"], observed[0] + 1)
        self.assertEqual(case["sample_index"], 1)
        self.assertEqual(case["rollout_seeds"], [observed[0] + 1001])
        advisor = evaluation.create_advisor(str(self.engine), "cpu", method="search")
        retained = filtered(advisor, (1, 1), 700, count=2, max_plies=2, backend="cpp")
        try:
            self.assertEqual((retained[0].index, retained[0].seed), (1, 701))
        finally:
            for sample in retained:
                sample.game.close()


if __name__ == "__main__":
    unittest.main()
