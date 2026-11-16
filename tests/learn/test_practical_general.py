"""General practical search: seat perspective, hidden states, chance and ownership."""

from __future__ import annotations

import copy
import json
import random
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.learn._common import requires_torch


class Table:
    def __init__(self, stage=0, hidden=0, clones=None):
        self.stage, self.hidden = stage, hidden
        self.clones = [] if clones is None else clones
        self.closed = False
        self.outcome = SimpleNamespace(terminal=False)
        self.action = 0

    def clone(self):
        child = copy.copy(self)
        child.outcome = copy.copy(self.outcome)
        self.clones.append(child)
        return child

    def close(self):
        assert not self.closed, "double close"
        self.closed = True


class TableSpec:
    game_id = "table"
    num_players = 3
    perfect_information = False
    zero_sum = False
    default_max_plies = 12

    def __init__(self):
        self.draws = []
        self.steps = []

    def actors(self, game):
        return {0: [2, 0], 1: [], 2: [1], 3: [2, 1], 4: []}[game.stage]

    def legal_actions(self, game, player):
        return [0, 1] if player in self.actors(game) else []

    def determinize(self, game, player, rng):
        child = game.clone()
        child.hidden = int(rng.integers(2))
        self.draws.append(child.hidden)
        return child

    def step(self, game, actions):
        from hasami_learn.games.base import StepOutcome
        assert set(actions) == set(self.actors(game))
        self.steps.append((game.stage, dict(actions)))
        if game.stage == 0:
            game.action = actions[2]
        game.stage += 1
        # The StepOutcome is authoritative even when the object's outcome lags.
        return StepOutcome(game.stage == 4, rewards=[-0.4, 0.1, 0.6] if game.stage == 4 else None)

    def outcome_score(self, values, player):
        return [0.1, 0.2, 0.9][player]

    def action_name(self, action):
        return str(action)


class TableEvaluator:
    device = "cpu"

    def __init__(self, spec=None):
        self.spec = spec or TableSpec()
        self.seen = []

    def evaluate(self, games, cancel=None, *, players=None):
        from hasami_learn.practical import Evaluation
        rows = []
        for game, player in zip(games, players):
            self.seen.append((game.stage, player))
            prior = {0: 0.75, 1: 0.25} if player in self.spec.actors(game) else {}
            value = (0.2 + 0.4 * game.action + 0.1 * game.hidden) if player == 2 else -0.8
            rows.append(Evaluation(prior, value))
        return rows


class CoinTable(Table):
    def __init__(self, seed):
        super().__init__()
        self.rng = random.Random(seed)

    def clone(self):
        child = super().clone()
        child.rng = copy.deepcopy(self.rng)
        return child


class CoinSpec(TableSpec):
    """An actual engine coin toss, either on the action or at a later chance step."""

    has_chance = True
    outcome_score = None

    def __init__(self, perfect_information, delayed=False):
        super().__init__()
        self.perfect_information = perfect_information
        self.delayed = delayed
        self.world_seeds = []

    def actors(self, game):
        return [2] if game.stage == 0 else []

    def resample_chance(self, game, rng):
        child = game.clone()
        seed = int(rng.integers(2**63))
        child.rng.seed(seed)
        self.world_seeds.append(seed)
        return child

    def determinize(self, game, player, rng):
        child = self.resample_chance(game, rng)
        child.hidden = int(rng.integers(2))
        return child

    def step(self, game, actions):
        from hasami_learn.games.base import StepOutcome
        assert set(actions) == set(self.actors(game))
        if game.stage == 0:
            game.action = actions[2]
            if self.delayed:
                game.stage = 1
                return StepOutcome(False)
        score = float(game.rng.randrange(2)) if game.action == 0 else 0.6
        self.draws.append((game.action, score))
        game.stage = 2
        return StepOutcome(True, rewards=[0, 0, 2 * score - 1])


@requires_torch
class ChanceRegressionTests(unittest.TestCase):
    def test_fresh_chance_converges_for_both_information_types(self):
        from hasami_learn.practical_general import GeneralSearchAdvisor
        for perfect in (True, False):
            for delayed in (False, True):
                for root_seed in (0, 1):
                    with self.subTest(perfect=perfect, delayed=delayed, root_seed=root_seed):
                        spec = CoinSpec(perfect, delayed)
                        root = CoinTable(root_seed)
                        before = root.rng.getstate()
                        estimates = []
                        for budget in (1, 2, 512):
                            advisor = GeneralSearchAdvisor(TableEvaluator(spec), determinizations=1,
                                                           rollouts=budget)
                            rows = advisor.advise(root, 3, player=2, guard=2, seed=19)
                            coin = next(row for row in rows if row.action == 0)
                            estimates.append(coin.practical_score)
                        self.assertAlmostEqual(estimates[-1], 0.5, delta=0.05)
                        self.assertLessEqual(abs(estimates[-1] - 0.5), abs(estimates[0] - 0.5))
                        self.assertEqual(rows[0].action, 1)
                        self.assertAlmostEqual(rows[0].practical_score, 0.6)
                        self.assertEqual(rows, advisor.advise(root, 3, player=2, guard=2, seed=19))
                        changed = advisor.advise(root, 3, player=2, guard=2, seed=20)
                        self.assertNotEqual(rows, changed)
                        if not delayed:
                            self.assertAlmostEqual(coin.engine_value, 2 * coin.practical_score - 1)
                            self.assertEqual(advisor.advise(root, 3, player=2, guard=0, seed=19)[0].action, 1)
                        self.assertEqual(before, root.rng.getstate())
                        self.assertEqual(root.stage, 0)
                        self.assertFalse(root.closed)
                        self.assertTrue(all(child.closed for child in root.clones))

    def test_each_rollout_has_a_distinct_world_and_cancellation_closes_it(self):
        from hasami.agents import AgentCancelled
        from hasami_learn.practical_general import GeneralSearchAdvisor
        for perfect in (True, False):
            with self.subTest(perfect=perfect):
                spec = CoinSpec(perfect)
                root = CoinTable(0)
                advisor = GeneralSearchAdvisor(TableEvaluator(spec), determinizations=3, rollouts=4)
                advisor.action_values(root, player=2)
                self.assertEqual(len(spec.world_seeds), 2 * 3 * 4)
                self.assertEqual(len(set(spec.world_seeds)), len(spec.world_seeds))
                event = threading.Event()
                original_step = spec.step
                def step(game, actions):
                    outcome = original_step(game, actions)
                    event.set()
                    return outcome
                spec.step = step
                with self.assertRaises(AgentCancelled):
                    advisor.advise(root, 3, player=2, cancel=event)
                self.assertFalse(root.closed)
                self.assertTrue(all(child.closed for child in root.clones))

    def test_chance_hook_required_and_owned_world_enforced(self):
        from hasami_learn.practical_general import GeneralSearchAdvisor
        spec = CoinSpec(True)
        spec.resample_chance = None
        with self.assertRaisesRegex(ValueError, "resample_chance"):
            GeneralSearchAdvisor(TableEvaluator(spec))
        for perfect in (True, False):
            spec = CoinSpec(perfect)
            if perfect:
                spec.resample_chance = lambda game, rng: game
            else:
                spec.determinize = lambda game, player, rng: game
            root = CoinTable(0)
            advisor = GeneralSearchAdvisor(TableEvaluator(spec))
            with self.assertRaisesRegex(ValueError, "owned clone"):
                advisor.advise(root, 3, player=2)
            self.assertFalse(root.closed)
            self.assertEqual(root.stage, 0)


@requires_torch
class GeneralArithmeticTests(unittest.TestCase):
    def advisor(self, **kwargs):
        from hasami_learn.practical_general import GeneralSearchAdvisor
        return GeneralSearchAdvisor(TableEvaluator(), **kwargs)

    def assert_clean(self, root):
        self.assertEqual(root.stage, 0)
        self.assertFalse(root.closed)
        self.assertTrue(root.clones)
        self.assertTrue(all(child.closed for child in root.clones))

    def test_hidden_joint_chance_perspective_and_seed(self):
        advisor = self.advisor(determinizations=4, rollouts=2)
        root = Table()
        rows = advisor.advise(root, 2, player=2, guard=2, top_k=2, seed=23)
        self.assertEqual([row.action for row in rows], [1, 0])
        for row in rows:
            self.assertGreaterEqual(row.practical_score, 0.6 + 0.2 * row.action)
            self.assertLessEqual(row.practical_score, 0.65 + 0.2 * row.action)
            self.assertAlmostEqual(row.engine_value, 2 * row.practical_score - 1)
            self.assertEqual(row.difficulty, 0)
        self.assertEqual(rows, advisor.advise(root, 2, player=2, guard=2, top_k=2, seed=23))
        self.assertTrue(any(stage == 1 and actions == {} for stage, actions in advisor.spec.steps))
        self.assertTrue(any(stage == 0 and set(actions) == {0, 2} for stage, actions in advisor.spec.steps))
        self.assertIn((2, 1), advisor.engine.seen)
        self.assertIn((3, 2), advisor.engine.seen)
        self.assert_clean(root)

    def test_terminal_hook_and_default_mapping(self):
        advisor = self.advisor(determinizations=1, rollouts=1)
        root = Table()
        rows = advisor.advise(root, 1, player=2, guard=2, depth=2)
        self.assertTrue(all(row.practical_score == 0.9 for row in rows))
        advisor.spec.outcome_score = None
        rows = advisor.advise(root, 1, player=2, guard=2, depth=2)
        self.assertTrue(all(row.practical_score == 0.8 for row in rows))
        self.assert_clean(root)

    def test_explicit_joint_player_and_cancellation(self):
        from hasami.agents import AgentCancelled
        advisor = self.advisor()
        root = Table()
        with self.assertRaises(ValueError):
            advisor.advise(root, 1)
        with self.assertRaises(ValueError):
            advisor.advise(root, 1, player=1)
        event = threading.Event()
        original = advisor.spec.step
        def step(game, actions):
            outcome = original(game, actions)
            event.set()
            return outcome
        advisor.spec.step = step
        with self.assertRaises(AgentCancelled):
            advisor.advise(root, 1, player=2, cancel=event)
        self.assert_clean(root)


    def test_exact_hidden_average_and_probability_weighted_difficulty(self):
        import numpy as np
        from hasami_learn.games.base import StepOutcome
        advisor = self.advisor()
        original_step = advisor.spec.step
        def step(game, actions):
            if game.stage == 3:
                game.stage = 4
                return StepOutcome(True, rewards=[0, 0, 1 if actions[2] == 0 else -1])
            return original_step(game, actions)
        advisor.spec.step = step
        advisor.spec.outcome_score = None
        root = Table()
        rows = advisor.advise(root, 2, player=2, guard=2, top_k=2, seed=23)
        for row in rows:
            hidden = [int(np.random.default_rng([23, row.action, k, r, 0]).integers(2))
                      for k in range(4) for r in range(2)]
            expected = (1 + 0.2 + 0.4 * row.action + 0.1 * np.mean(hidden)) / 2
            self.assertAlmostEqual(row.practical_score, expected)
            self.assertAlmostEqual(row.difficulty, 0.25)
        self.assert_clean(root)

    def test_perfect_information_identity_and_caps(self):
        advisor = self.advisor(max_steps=1)
        advisor.spec.perfect_information = True
        advisor.spec.has_chance = False
        from hasami_learn.practical_general import GeneralSearchAdvisor
        advisor = GeneralSearchAdvisor(advisor.engine, max_steps=1)
        root = Table()
        advice = advisor.advise(root, 1, player=2, guard=2)
        self.assertEqual(advisor.determinizations, 1)
        self.assertFalse(advisor.spec.draws)
        self.assertEqual(advisor.last_search["capped_rollouts"], 4)
        self.assertTrue(all(a.practical_score > 0.5 for a in advice))
        self.assertTrue(all(a.difficulty is None for a in advice))
        self.assert_clean(root)

    def test_candidate_cap_guard_fallback_and_hidden_input_independence(self):
        from hasami_learn.practical import Evaluation
        advisor = self.advisor(determinizations=1, rollouts=1)
        legal = advisor.spec.legal_actions
        advisor.spec.legal_actions = lambda game, player: (list(range(18)) if game.stage == 0 and player == 2
                                                          else legal(game, player))
        original = advisor.engine.evaluate
        prior = {a: 0.98 / 16 if a < 16 else 0.015 if a == 16 else 0.005 for a in range(18)}
        def evaluate(games, cancel=None, *, players=None):
            rows = original(games, cancel, players=players)
            return [Evaluation(prior if game.stage == 0 and player == 2 else row.prior,
                               0.8 if game.action == 17 else 0.0)
                    for game, player, row in zip(games, players, rows)]
        advisor.engine.evaluate = evaluate
        root = Table()
        rows = advisor.advise(root, 1, player=2, guard=2, top_k=30)
        self.assertEqual([row.action for row in rows], list(range(17)))
        fallback = advisor.advise(root, 1, player=2, guard=0)[0]
        self.assertEqual(fallback.action, 17)
        self.assertTrue(fallback.guard_fallback)
        root.hidden = 1
        self.assertEqual(rows, advisor.advise(root, 1, player=2, guard=2, top_k=30))
        self.assert_clean(root)

    def test_missing_sampler_and_invalid_outcomes_fail_closed(self):
        from hasami_learn.practical_general import GeneralSearchAdvisor, outcome_score
        engine = TableEvaluator()
        engine.spec.determinize = None
        with self.assertRaisesRegex(ValueError, "determinize"):
            GeneralSearchAdvisor(engine)
        for values in ([0, 0], [0, 0, float("nan")], [0, 0, 2]):
            with self.assertRaises(ValueError):
                outcome_score(engine.spec, values, 2)
        engine.spec.outcome_score = lambda values, player: 1.1
        with self.assertRaises(ValueError):
            outcome_score(engine.spec, [0, 0, 0], 2)

    def test_adapter_truncation_and_error_cleanup(self):
        from hasami_learn.games.base import StepOutcome
        advisor = self.advisor()
        advisor.spec.step = lambda game, actions: StepOutcome(False, True)
        root = Table()
        rows = advisor.advise(root, 1, player=2)
        self.assertTrue(all(a.practical_score == 0.5 and a.engine_value == 0 for a in rows))
        self.assertGreater(advisor.last_search["truncated_rollouts"], 0)
        self.assert_clean(root)
        advisor = self.advisor()
        original = advisor.engine.evaluate
        def fail(games, cancel=None, *, players=None):
            if any(game.stage for game in games):
                raise RuntimeError("evaluation failure")
            return original(games, cancel, players=players)
        advisor.engine.evaluate = fail
        root = Table()
        with self.assertRaisesRegex(RuntimeError, "evaluation failure"):
            advisor.advise(root, 1, player=2)
        self.assert_clean(root)

    def test_guard_proxy_and_invalid_arguments(self):
        from hasami.agents import SearchLimits
        advisor = self.advisor()
        root = Table()
        rows = advisor.advise(root, 2, player=2, guard=0)
        self.assertEqual([row.action for row in rows], [1])
        a, b = advisor.proxy_agent(3, seed=5), advisor.proxy_agent(3, seed=5)
        self.assertEqual([a.choose(root, SearchLimits(), player=2).action for _ in range(20)],
                         [b.choose(root, SearchLimits(), player=2).action for _ in range(20)])
        self.assertAlmostEqual(sum(a.distribution(root, player=2).values()), 1)
        for options in ({"depth": 0}, {"guard": float("nan")}, {"seed": -1}, {"top_k": 0}):
            with self.assertRaises(ValueError):
                advisor.advise(root, 1, player=2, **options)
        for options in ({"determinizations": 0}, {"rollouts": True}, {"max_steps": 0}):
            with self.assertRaises(ValueError):
                self.advisor(**options)
        with self.assertRaises(ValueError):
            advisor.proxy_agent(1, style="qre")
        self.assert_clean(root)

    def test_general_evaluation_seats_outcomes_pairing_and_caps(self):
        from hasami_learn.practical_eval import continuation, paired_intervention
        advisor = self.advisor()
        root = Table()
        scores, cap = continuation(advisor, root, [1, 2, 3], 8, max_plies=8)
        self.assertEqual(scores, [0.1, 0.2, 0.9])
        self.assertFalse(cap)
        scores, cap = continuation(advisor, root, [1, 2, 3], 8, max_plies=1)
        self.assertEqual(scores, [0.5] * 3)
        self.assertTrue(cap)
        traces = {}
        samples, caps = paired_intervention(advisor, root, [1, 2, 3], {"own": 0, "search": 1},
                                            [7, 8], max_plies=8, traces=traces, player=2)
        self.assertEqual(samples, {"own": [0.9, 0.9], "search": [0.9, 0.9]})
        self.assertEqual(caps, {"own": [0.0, 0.0], "search": [0.0, 0.0]})
        for seed in (7, 8):
            self.assertEqual(traces[("own", seed)], traces[("search", seed)])
        self.assert_clean(root)

    def test_runtime_import_closure(self):
        code = """import sys
import hasami_learn.practical_general
for name in ('trainer', 'selfplay', 'opponents', 'replay', 'algorithms'):
    assert 'hasami_learn.' + name not in sys.modules, name
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_agent_defaults_and_explicit_depth_for_both_advisors(self):
        from hasami.agents import SearchLimits
        from hasami_learn.practical import PracticalAgent, create_practical_agent
        from hasami_learn.practical_search import SearchAdvisor
        from tests.learn.test_practical_search import SearchEvaluator, SearchTree
        nodes = {(): dict(prior={0: 1.0}, q={0: 0.0}), (0,): dict(terminal=[1, -1])}
        for advisor, root, player, default in (
                (self.advisor(), Table(), 2, 1),
                (SearchAdvisor(SearchEvaluator(nodes)), SearchTree(nodes), 0, 2)):
            advisor.manifest, advisor.checkpoint_dir = None, "fixture"
            for factory in (False, True):
                for depth in (None, 1, 2):
                    with self.subTest(general=getattr(advisor, "general", False), factory=factory, depth=depth):
                        options = {} if depth is None else {"depth": depth}
                        if factory:
                            with patch("hasami_learn.practical.create_advisor", return_value=advisor):
                                agent = create_practical_agent("fixture", **options)
                        else:
                            agent = PracticalAgent(advisor, **options)
                        decision = agent.choose(root, SearchLimits(), player=player)
                        self.assertEqual(decision.depth, default if depth is None else depth)
                        self.assertEqual(advisor.last_search["depth"], decision.depth)
                        self.assertTrue(all(child.closed for child in root.clones))


@requires_torch
class RiichiGeneralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import random
        import numpy as np
        import torch
        from hasami_learn.config import TrainingConfig
        from hasami_learn.trainer import train_from_config
        cls.addClassCleanup(random.setstate, random.getstate())
        cls.addClassCleanup(np.random.set_state, np.random.get_state())
        cls.addClassCleanup(torch.set_rng_state, torch.get_rng_state())
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        temp = tempfile.TemporaryDirectory(prefix="practical-general-")
        cls.addClassCleanup(temp.cleanup)
        cls.root = Path(temp.name)
        raw = json.loads((Path(__file__).resolve().parents[2] / "configs/tiny/riichi-ppo-league.json").read_text())
        raw["run"].update(output_dir=str(cls.root), backend="cpp", seed=1)
        raw["selfplay"].update(max_plies=8)
        cls.checkpoint = train_from_config(TrainingConfig.from_dict(raw), echo=False)

    def advisor(self):
        from hasami_learn.practical import create_advisor
        from hasami_learn.practical_general import GeneralSearchAdvisor
        advisor = create_advisor(str(self.checkpoint), "cpu")
        self.assertIsInstance(advisor, GeneralSearchAdvisor)
        self.assertTrue(all(row.elo is None for row in advisor.levels))
        return advisor

    def call_position(self, advisor, backend):
        from riichi.actions import PASS
        game = advisor.spec.create_game(backend, seed=0)
        try:
            for _ in range(100):
                actors = advisor.spec.actors(game)
                callers = [p for p in actors if PASS in advisor.spec.legal_actions(game, p)
                           and len(advisor.spec.legal_actions(game, p)) > 1]
                if callers:
                    return game, callers[0]
                advisor.spec.step(game, {p: min(advisor.spec.legal_actions(game, p)) for p in actors})
            self.fail("seeded fixture did not reach a nontrivial call")
        except BaseException:
            game.close()
            raise

    def tracked_advice(self, advisor, game, player):
        cls = type(game)
        original_clone, original_close = cls.clone, cls.close
        created, closed = [], set()
        def clone(source):
            work = original_clone(source)
            created.append(work)
            return work
        def close(work):
            self.assertNotIn(id(work), closed)
            closed.add(id(work))
            return original_close(work)
        before = copy.deepcopy(game.observation(player)), copy.deepcopy(game.moves)
        with patch.object(cls, "clone", clone), patch.object(cls, "close", close):
            rows = advisor.advise(game, 3, player=player, seed=42)
        self.assertEqual(closed, {id(work) for work in created})
        self.assertEqual(before, (game.observation(player), game.moves))
        self.assertEqual(rows, advisor.advise(game, 3, player=player, seed=42))
        for row in rows:
            self.assertIs(type(row.action), int)
            self.assertIn(row.action, advisor.spec.legal_actions(game, player))
            self.assertTrue(0 <= row.practical_score <= 1)
            self.assertTrue(-1 <= row.engine_value <= 1)
            self.assertTrue(0 <= row.value_loss <= 2)
            self.assertTrue(0 <= row.difficulty <= 1)
            self.assertTrue(0 <= row.naturalness <= 1)
        return rows

    def test_discard_call_both_backends_and_agent_factory(self):
        from hasami.agents import SearchLimits
        from hasami_learn.practical import create_practical_agent
        advisor = self.advisor()
        results = {}
        for backend in ("python", "cpp"):
            game = advisor.spec.create_game(backend, seed=0)
            try:
                player = advisor.spec.actors(game)[0]
                results[backend, "discard"] = self.tracked_advice(advisor, game, player)
            finally:
                game.close()
            game, player = self.call_position(advisor, backend)
            try:
                results[backend, "call"] = self.tracked_advice(advisor, game, player)
                agent = create_practical_agent(str(self.checkpoint), "cpu")
                decision = agent.choose(game, SearchLimits(), player=player)
                self.assertEqual(decision.depth, 1)
                self.assertIn(decision.action,
                              advisor.spec.legal_actions(game, player))
            finally:
                game.close()
        for kind in ("discard", "call"):
            self.assertEqual(results["python", kind], results["cpp", kind])

    def test_reports_and_default_native_cpu_latency(self):
        import numpy as np
        from hasami_learn.practical import validate_artifact
        from hasami_learn.practical_eval import run_experiment
        advisor = self.advisor()
        for experiment in ("E1", "E4", "E7"):
            output = self.root / f"{experiment}.json"
            report = run_experiment(experiment, str(self.checkpoint), output=output, backend="cpp", device="cpu",
                level=3, positions=2, rollouts=2, max_plies=8, smoke=True)
            validate_artifact(report, "eval")
            self.assertTrue(report["smoke"])
            self.assertEqual([report["config"][key] for key in ("K", "M", "D")], [4, 2, 1])
            self.assertTrue(output.with_suffix(".done.json").exists())
            self.assertTrue(any(row["n"] > 0 for row in report["results"]))
            self.assertTrue(any(row["metric"] == "advice_latency_median" and row["n"] > 0
                                for row in report["results"]))
        timings = {}
        for kind in ("discard", "call"):
            if kind == "discard":
                game = advisor.spec.create_game("cpp", seed=0)
                player = advisor.spec.actors(game)[0]
            else:
                game, player = self.call_position(advisor, "cpp")
            try:
                elapsed = []
                for seed in range(5):
                    start = time.perf_counter()
                    advisor.advise(game, 3, player=player, seed=seed)
                    elapsed.append((time.perf_counter() - start) * 1000)
                timings[kind] = {"median_ms": float(np.median(elapsed)),
                                 "p95_ms": float(np.quantile(elapsed, 0.95)), "samples": len(elapsed)}
            finally:
                game.close()
        print("GENERAL_RIICHI_LATENCY " + json.dumps(dict(backend="cpp", device="cpu", K=4, M=2, D=1,
                                                          smoke=True, timings=timings)), flush=True)


if __name__ == "__main__":
    unittest.main()
