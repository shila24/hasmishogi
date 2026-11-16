"""Method A arithmetic, ownership, runtime imports and checkpoint smoke coverage."""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.learn._common import requires_torch


class SearchTree:
    def __init__(self, nodes, path=(), clones=None):
        self.nodes, self.path = nodes, path
        self.clones = [] if clones is None else clones
        self.closed = False
        self.outcome = SimpleNamespace(terminal=False)

    def clone(self):
        child = SearchTree(self.nodes, self.path, self.clones)
        self.clones.append(child)
        return child

    def close(self):
        assert not self.closed, "clone closed twice"
        self.closed = True


class SearchSpec:
    game_id = "tree"
    num_players = 2
    perfect_information = True
    zero_sum = True

    def actors(self, game):
        return [game.nodes[game.path].get("player", len(game.path) % 2)]

    def legal_actions(self, game, player):
        return list(game.nodes[game.path]["prior"])

    def step(self, game, actions):
        from hasami_learn.games.base import StepOutcome
        game.path += (actions[self.actors(game)[0]],)
        node = game.nodes[game.path]
        return StepOutcome("terminal" in node, node.get("truncated", False), node.get("terminal"))

    def terminal_values(self, game):
        return game.nodes[game.path]["terminal"]

    def action_name(self, action):
        return str(action)


class SearchEvaluator:
    batch_size = 64

    def __init__(self, nodes):
        self.spec, self.nodes = SearchSpec(), nodes
        self.batches, self.q_batches = [], []

    def evaluate(self, games, cancel=None):
        from hasami_learn.practical import Evaluation
        self.batches.append([g.path for g in games])
        return [Evaluation(self.nodes[g.path].get("prior", {0: 1.0}),
                           self.nodes[g.path].get("value", 0.0)) for g in games]

    def one_ply_many(self, games, cancel=None):
        self.q_batches.append([g.path for g in games])
        return [self.nodes[g.path]["q"] for g in games]

    def one_ply(self, game, cancel=None):
        return self.one_ply_many([game], cancel)[0]


@requires_torch
class SearchArithmeticTests(unittest.TestCase):
    def advisor(self, nodes):
        from hasami_learn.practical_search import SearchAdvisor
        return SearchAdvisor(SearchEvaluator(nodes))

    def assert_clean(self, root):
        self.assertEqual(root.path, ())
        self.assertFalse(root.closed)
        self.assertTrue(root.clones)
        self.assertTrue(all(g.closed for g in root.clones))

    def test_expectimax_own_opponent_top_six_terminal_and_difficulty(self):
        nodes = {(): dict(prior={0: 1.0}, q={0: 0.0}),
                 (0,): dict(prior={0: 0.8, 1: 0.2}, q={0: -0.3, 1: 0.4})}
        for reply in range(2):
            nodes[(0, reply)] = dict(prior={a: (7-a)/28 for a in range(7)},
                                     value=-nodes[(0,)]["q"][reply],
                                     q={a: 1.0 if a == 0 else 0.1*a - 0.2*reply for a in range(7)})
            for action in range(7):
                nodes[(0, reply, action)] = (dict(terminal=[1.0, -1.0]) if action == 0
                    else dict(value=0.2 * reply - 0.1 * action))
        advisor = self.advisor(nodes)
        root = SearchTree(nodes)

        def probabilities(node, beta):
            prior, q = node["prior"], node["q"]
            weights = {a: math.sqrt(p) * math.exp(beta*q[a]) for a, p in prior.items()}
            total = sum(weights.values())
            return {a: 0.95*w/total + 0.05*prior[a] for a, w in weights.items()}

        opponent = probabilities(nodes[(0,)], 2)
        expected, difficulty = 0.0, 0.0
        for b in opponent:
            own = probabilities(nodes[(0, b)], 1)
            retained = sorted(own, key=lambda a: (-own[a], a))[:6]
            mass = sum(own[a] for a in retained)
            expected += opponent[b] * sum(own[a]/mass * (1.0 if a == 0 else
                       (1 - nodes[(0, b, a)]["value"])/2) for a in retained)
            difficulty += opponent[b] * (1 - own[0])
        row = advisor.advise(root, 1, 2, depth=2)[0]
        self.assertAlmostEqual(row.practical_score, expected)
        self.assertAlmostEqual(row.difficulty, difficulty)
        self.assertEqual(row.naturalness, 1)
        layers = [batch for batch in advisor.engine.batches if batch and batch[0]]
        self.assertEqual([len(batch) for batch in layers], [1, 2, 12])
        self.assertTrue(all(len({len(path) for path in batch}) == 1 for batch in layers))
        self.assertEqual(row, advisor.advise(root, 1, 2, depth=2)[0])
        self.assert_clean(root)

    def test_candidate_threshold_guard_fallback_and_ties(self):
        prior = {a: 0.98/16 if a < 16 else 0.015 if a == 16 else 0.005 for a in range(18)}
        nodes = {(): dict(prior=prior, q={a: 0.0 for a in prior})}
        nodes.update({(a,): dict(terminal=[0.0, 0.0]) for a in prior})
        advisor = self.advisor(nodes)
        root = SearchTree(nodes)
        self.assertEqual([a.action for a in advisor.advise(root, 1, top_k=30)], list(range(17)))
        nodes[()]["q"][1] = 0.4
        self.assertEqual([a.action for a in advisor.advise(root, 1, guard=0.1)], [1])
        nodes[()]["q"][17] = 1.0
        row = advisor.advise(root, 1, guard=0.1)[0]
        self.assertEqual(row.action, 17)
        self.assertEqual(row.value_loss, 0)
        self.assertTrue(row.guard_fallback)
        self.assert_clean(root)

    def test_prior_prefilter_keeps_terminal_moves_then_renormalizes_top_six(self):
        nodes = {(): dict(prior={0: 1.0}, q={0: 0.0}),
                 (0,): dict(player=0, prior={a: (10-a)/55 for a in range(10)})}
        for a in range(9):
            nodes[(0, a)] = dict(player=1, value=-a/10)
        nodes[(0, 9)] = dict(terminal=[1.0, -1.0])
        advisor, root = self.advisor(nodes), SearchTree(nodes)
        retained = [*range(8), 9]
        # Independent scalar QRE calculation, including the top-eight lapse.
        prior = nodes[(0,)]["prior"]
        q = {a: a/10 if a < 9 else 1.0 for a in retained}
        weights = {a: math.sqrt(prior[a]) * math.exp(4*q[a]) for a in retained}
        total, lapse = sum(weights.values()), sum(prior[a] for a in range(8))
        probabilities = {a: 0.95*w/total + (0.05*prior[a]/lapse if a < 8 else 0)
                         for a, w in weights.items()}
        top = sorted(probabilities, key=lambda a: (-probabilities[a], a))[:6]
        mass = sum(probabilities[a] for a in top)
        expected = sum(probabilities[a]/mass * (q[a]+1)/2 for a in top)
        row = advisor.advise(root, 3, depth=1)[0]
        self.assertIn(9, top)
        self.assertAlmostEqual(row.practical_score, expected)
        self.assertAlmostEqual(row.difficulty, sum(p for a, p in probabilities.items() if 1-q[a] > 0.2))
        evaluated = [path for batch in advisor.engine.batches for path in batch]
        self.assertNotIn((0, 8), evaluated)
        self.assertNotIn((0, 9), evaluated)
        self.assertEqual(len([p for p in evaluated if len(p) == 2]), 8)
        self.assert_clean(root)

    def test_frontier_is_one_batch_per_layer_even_above_evaluator_batch_size(self):
        nodes = {(): dict(prior={a: 1/17 for a in range(17)}, q={a: 0.0 for a in range(17)})}
        for a in range(17):
            nodes[(a,)] = dict(prior={b: 1/9 for b in range(9)})
            for b in range(9):
                nodes[(a, b)] = dict(value=b/10)
        advisor, root = self.advisor(nodes), SearchTree(nodes)
        advisor.advise(root, 1, depth=1)
        layers = [batch for batch in advisor.engine.batches if batch and batch[0]]
        self.assertEqual([len(batch) for batch in layers], [17, 136])
        self.assertEqual(advisor.last_search["evaluated_positions"], 153)
        self.assert_clean(root)

    def test_time_budget_keeps_deepest_complete_result_and_closes_partial_tree(self):
        nodes = {(): dict(prior={0: 1.0}, q={0: 0.0})}
        for n in range(1, 6):
            nodes[(0,)*n] = dict(prior={0: 1.0}, value=n/10)
        for expire_at in (None, 2, 4):
            with self.subTest(expire_at=expire_at):
                advisor, root = self.advisor(nodes), SearchTree(nodes)
                shallow = advisor.advise(root, 1, depth=2)
                deep = advisor.advise(root, 1, depth=4)
                clock = [0.0]
                original = advisor.engine.evaluate
                def evaluate(games, cancel=None):
                    rows = original(games, cancel)
                    if expire_at is not None and any(len(g.path) == expire_at for g in games):
                        clock[0] = 2.0
                    return rows
                with patch.object(advisor.engine, "evaluate", side_effect=evaluate), patch(
                        "hasami_learn.practical_search.time.perf_counter", side_effect=lambda: clock[0]):
                    result = advisor.advise(root, 1, time_ms=1000)
                self.assertEqual(result, deep if expire_at is None else shallow)
                self.assertEqual(advisor.last_search["completed_depths"], [2, 4] if expire_at is None else [2])
                self.assertEqual(advisor.last_search["depth"], 4 if expire_at is None else 2)
                self.assertEqual(advisor.advise(root, 1, depth=6, time_ms=0), shallow)
                self.assert_clean(root)

    def test_cancellation_during_deepening_is_not_a_timeout_fallback(self):
        from hasami.agents import AgentCancelled
        nodes = {(): dict(prior={0: 1.0}, q={0: 0.0})}
        for n in range(1, 6):
            nodes[(0,)*n] = dict(prior={0: 1.0}, value=0.0)
        advisor, root = self.advisor(nodes), SearchTree(nodes)
        event = threading.Event()
        original = advisor.engine.evaluate
        def evaluate(games, cancel=None):
            if any(len(g.path) == 4 for g in games):
                event.set()
            return original(games, cancel)
        with patch.object(advisor.engine, "evaluate", side_effect=evaluate), self.assertRaises(AgentCancelled):
            advisor.advise(root, 1, time_ms=10000, cancel=event)
        self.assertIsNone(advisor.last_search)
        self.assert_clean(root)

    def test_identity_extra_turns_terminal_and_truncation(self):
        nodes = {(): dict(player=1, prior={0: 0.5, 1: 0.5}, q={0: 0.0, 1: 0.0}),
                 (0,): dict(player=1, prior={0: 1.0}, q={0: 0.0}),
                 (1,): dict(truncated=True), (0, 0): dict(terminal=[-1.0, 1.0])}
        root = SearchTree(nodes)
        rows = self.advisor(nodes).advise(root, 1, depth=1)
        self.assertEqual([r.practical_score for r in rows], [1.0, 0.5])
        self.assertEqual([r.difficulty for r in rows], [0.0, 0.0])
        self.assert_clean(root)

    def test_default_two_and_explicit_four_continuation_plies_and_leaf_identity(self):
        nodes = {(): dict(player=1, prior={0: 1.0}, q={0: 0.0})}
        for n in range(1, 6):
            nodes[(0,)*n] = dict(player=1, prior={0: 1.0}, q={0: 0.0}, value=n/10)
        root = SearchTree(nodes)
        advisor = self.advisor(nodes)
        row = advisor.advise(root, 1)[0]
        self.assertAlmostEqual(row.practical_score, 0.65)
        self.assertEqual(advisor.last_search["depth"], 2)
        row = advisor.advise(root, 1, depth=4)[0]
        self.assertAlmostEqual(row.practical_score, 0.75)
        self.assert_clean(root)

    def test_difficulty_weights_reach_across_own_decisions(self):
        nodes = {(): dict(prior={0: 1.0}, q={0: 0.0}),
                 (0,): dict(player=0, prior={0: 0.8, 1: 0.2}, q={0: 0.0, 1: 0.0}),
                 (0, 1): dict(terminal=[0.0, 0.0]),
                 (0, 0): dict(player=0, prior={0: 0.5, 1: 0.5}, q={0: 1.0, 1: -1.0}),
                 (0, 0, 0): dict(terminal=[1.0, -1.0]),
                 (0, 0, 1): dict(terminal=[-1.0, 1.0])}
        advisor, root = self.advisor(nodes), SearchTree(nodes)
        proxy = advisor.proxy_agent(1, style="qre")
        reach = proxy.from_prior(nodes[(0,)]["prior"], nodes[(0,)]["q"])[0]
        error = proxy.from_prior(nodes[(0, 0)]["prior"], nodes[(0, 0)]["q"])[1]
        row = advisor.advise(root, 1, depth=2)[0]
        self.assertAlmostEqual(row.difficulty, reach * error / (1 + reach))
        self.assert_clean(root)

    def test_cancellation_and_exception_close_clones(self):
        from hasami.agents import AgentCancelled
        nodes = {(): dict(prior={0: 1.0}, q={0: 0.0}),
                 (0,): dict(prior={0: 1.0}, q={0: 0.0})}
        for failure in ("cancel", "evaluate", "step"):
            with self.subTest(failure=failure):
                root, advisor = SearchTree(nodes), self.advisor(nodes)
                event = threading.Event()
                original = advisor.engine.evaluate
                def evaluate(games, cancel=None):
                    if any(g.path for g in games):
                        if failure == "cancel":
                            event.set()
                        else:
                            raise RuntimeError("evaluation failed")
                    return original(games, cancel)
                advisor.engine.evaluate = evaluate
                if failure == "step":
                    advisor.spec.step = lambda *args: (_ for _ in ()).throw(RuntimeError("step failed"))
                with self.assertRaises(AgentCancelled if failure == "cancel" else RuntimeError):
                    advisor.advise(root, 1, cancel=event)
                self.assert_clean(root)
        event.set()
        with self.assertRaises(AgentCancelled):
            self.advisor(nodes).advise(SearchTree(nodes), 1, cancel=event)

    def test_invalid_arguments_and_default_levels(self):
        advisor = self.advisor({(): dict(prior={0: 1.0}, q={0: 0.0})})
        self.assertEqual([r.elo for r in advisor.levels], [None]*6)
        self.assertEqual([advisor.proxy_agent(i, style="qre").params["beta"] for i in range(1, 7)],
                         [1, 2, 4, 8, 16, 32])
        for kwargs in ({"level": 0}, {"level": True}, {"level": 7}, {"opponent_level": 7},
                       {"depth": 0}, {"depth": True}, {"depth": 2.5}, {"depth": 3}, {"depth": 7}, {"top_k": 0},
                       {"guard": -1}, {"guard": float("nan")}, {"time_ms": -1},
                       {"time_ms": True}, {"time_ms": float("nan")}, {"time_ms": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                advisor.advise(SearchTree(advisor.engine.nodes), **dict({"level": 1}, **kwargs))

    def test_runtime_import_has_no_training_modules(self):
        code = """
import sys
import hasami_learn.practical_search
for name in ('trainer', 'cli', 'replay', 'datastore', 'parallel', 'datasets', 'selfplay',
             'batched_selfplay', 'opponents', 'evaluation', 'ladder', 'algorithms'):
    prefix = 'hasami_learn.' + name
    assert not any(m == prefix or m.startswith(prefix + '.') for m in sys.modules), prefix
"""
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, timeout=30)


@requires_torch
class SearchCheckpointTests(unittest.TestCase):
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
        temp = tempfile.TemporaryDirectory(prefix="practical-search-")
        cls.addClassCleanup(temp.cleanup)
        cls.root = Path(temp.name)
        cls.checkpoints = {}
        root = Path(__file__).resolve().parents[2]
        for game, filename in (("hasami", "alphazero.json"), ("shogi", "alphazero-shogi.json")):
            raw = json.loads((root / "configs/tiny" / filename).read_text())
            raw["run"].update(output_dir=str(cls.root), backend="cpp" if game == "hasami" else "python")
            raw["network"]["params"]["hidden"] = 16
            raw["algorithm"]["params"].update(simulations=1, batch_games=1)
            raw["selfplay"].update(games_per_iteration=1, max_plies=4)
            raw["replay"].update(batch_size=4, min_size=1)
            raw["optimizer"]["updates_per_iteration"] = 1
            raw["datastore"]["enabled"] = False
            cls.checkpoints[game] = train_from_config(TrainingConfig.from_dict(raw), echo=False)

    def calibration(self, checkpoint):
        from hasami_learn.practical_calibrate import synthetic_levels
        levels = synthetic_levels(checkpoint, checkpoint, k=3)
        levels["train_styles"], levels["heldout_styles"] = ["qre"], []
        for row in levels["levels"]:
            row["styles"] = {"qre": row["styles"]["qre"]}
        return levels

    def test_trained_hasami_shogi_with_and_without_calibration(self):
        from hasami.agents import AgentCancelled, SearchLimits
        from hasami_learn.practical import create_advisor, create_practical_agent, create_proxy_agent
        from hasami_learn.practical_search import SearchAdvisor
        for game_id, checkpoint in self.checkpoints.items():
            for calibrated in (False, True):
                with self.subTest(game=game_id, calibrated=calibrated):
                    path = checkpoint / "levels.json"
                    if calibrated:
                        path.write_text(json.dumps(self.calibration(checkpoint)))
                        self.addCleanup(path.unlink, missing_ok=True)
                    advisor = create_advisor(str(checkpoint), "cpu")
                    self.assertIsInstance(advisor, SearchAdvisor)
                    self.assertIsInstance(create_advisor(str(checkpoint), "cpu", method="auto"), SearchAdvisor)
                    self.assertTrue(all((r.elo is not None) == calibrated for r in advisor.levels))
                    self.assertFalse(getattr(advisor.engine.network, "aux_heads", {}))
                    game = advisor.spec.create_game("cpp" if game_id == "hasami" else "python")
                    try:
                        if game_id == "hasami":
                            self.assertEqual(len(game.legal_actions()), 63)
                            self.assertEqual(game.perft(2), 3717)
                        # Exercise history and the opposite side's shogi label mapping.
                        advisor.spec.step(game, {0: advisor.spec.legal_actions(game, 0)[0]})
                        snapshot = (game.position, list(game.moves), list(game.history_keys()))
                        legal = advisor.spec.legal_actions(game, 1)
                        proxies = [advisor.proxy_agent(1, seed=19) for _ in range(2)]
                        self.assertEqual([proxies[0].choose(game, SearchLimits()).action for _ in range(3)],
                                         [proxies[1].choose(game, SearchLimits()).action for _ in range(3)])
                        rows = advisor.advise(game, 1, 2, depth=1)
                        self.assertEqual(rows, advisor.advise(game, 1, 2, depth=1))
                        for row in rows:
                            self.assertIs(type(row.action), int)
                            self.assertIn(row.action, legal)
                            self.assertIn(row.move, game.legal_actions())
                            self.assertTrue(row.text)
                            self.assertTrue(0 <= row.practical_score <= 1)
                            self.assertTrue(-1 <= row.engine_value <= 1)
                            self.assertTrue(0 <= row.value_loss <= 0.3)
                            self.assertTrue(0 <= row.difficulty <= 1)
                            self.assertTrue(0 <= row.naturalness <= 1)
                        self.assertIn(advisor.engine_best(game).action, legal)
                        for factory in (create_practical_agent, create_proxy_agent):
                            agent = factory(str(checkpoint), "cpu", level=1)
                            decision = agent.choose(game, SearchLimits())
                            self.assertIn(decision.action, legal)
                            if factory == create_practical_agent:
                                self.assertEqual(agent.depth, 2)
                                self.assertEqual(decision.depth, 2)
                            cancel = threading.Event()
                            cancel.set()
                            with self.assertRaises(AgentCancelled):
                                agent.choose(game, SearchLimits(), cancel)
                        timed = create_practical_agent(str(checkpoint), "cpu", level=1, time_ms=0)
                        self.assertEqual(timed.choose(game, SearchLimits()).depth, 2)
                        self.assertEqual(snapshot, (game.position, list(game.moves), list(game.history_keys())))
                    finally:
                        game.close()
                        if calibrated:
                            path.unlink()

    def test_calibration_hash_game_mismatch_unavailable_and_fixed_budget(self):
        from hasami_learn.practical import create_advisor
        checkpoint = self.checkpoints["hasami"]
        levels = self.calibration(checkpoint)
        path = checkpoint / "levels.json"
        self.addCleanup(path.unlink, missing_ok=True)
        for key, value in (("engine_sha256", "0"*64), ("game_id", "other"), ("game_params", {"wrong": 1})):
            bad = copy.deepcopy(levels)
            bad[key] = value
            path.write_text(json.dumps(bad))
            with self.subTest(key=key), self.assertRaises(ValueError):
                create_advisor(str(checkpoint), "cpu")
        levels["levels"][0]["styles"]["qre"]["params"]["simulations"] = 64
        levels["levels"][1]["styles"]["qre"]["available"] = False
        path.write_text(json.dumps(levels))
        advisor = create_advisor(str(checkpoint), "cpu")
        self.assertEqual(advisor.proxy_agent(1, style="qre").params["simulations"], 64)
        self.assertEqual(advisor._search_proxy(1).params["simulations"], 0)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            advisor.proxy_agent(2, style="qre")
        with self.assertRaisesRegex(ValueError, "practical checkpoint"):
            create_advisor(str(checkpoint), "cpu", method="learned")
        with self.assertRaisesRegex(ValueError, "method"):
            create_advisor(str(checkpoint), "cpu", method="typo")

    def test_e1_e3_e4_e7_search_reports_and_cli(self):
        from hasami_learn.practical import validate_artifact
        from hasami_learn.practical_eval import main, run_experiment
        checkpoint = self.checkpoints["hasami"]
        for experiment in ("E1", "E3", "E4", "E7"):
            with self.subTest(experiment=experiment):
                output = self.root / f"search-{experiment}.json"
                report = run_experiment(experiment, str(checkpoint), output=output, method="search", seed=0,
                                        depth=1, smoke=True, level=1, positions=1, rollouts=1,
                                        max_plies=2, reference_simulations=2)
                validate_artifact(report, "eval")
                self.assertEqual(json.loads(output.with_suffix(".done.json").read_text()), report)
                self.assertTrue(report["smoke"])
                self.assertEqual(report["config"]["method"], "search")
                self.assertNotIn("levels_sha256", report)
                self.assertTrue(all(v is None for v in report["config"]["level_elo"].values()))
                self.assertTrue(any(r["n"] > 0 and r["metric"] == "advice_latency_p95" for r in report["results"]))
                self.assertTrue(any(r.get("arm", "").startswith("search") for r in report["results"]))
        output = self.root / "search-cli.json"
        self.assertEqual(main(["E7", "--checkpoint", str(checkpoint), "--method", "search",
                               "--output", str(output), "--depth", "1", "--positions", "1",
                               "--rollouts", "1", "--max-plies", "2", "--level", "1", "--smoke"]), 0)
        validate_artifact(json.loads(output.read_text()), "eval")
        calibration_path = checkpoint / "levels.json"
        calibration_path.write_text(json.dumps(self.calibration(checkpoint)))
        self.addCleanup(calibration_path.unlink, missing_ok=True)
        report = run_experiment("E7", str(checkpoint), output=output, method="search", depth=1,
                                smoke=True, level=1, positions=1, rollouts=1, max_plies=2)
        self.assertEqual(len(report["levels_sha256"]), 64)
        self.assertIsNotNone(report["config"]["level_elo"]["1"])
