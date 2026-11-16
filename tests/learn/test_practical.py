"""Practical play contract, arithmetic, training and E1-E7 CPU smoke tests."""

from __future__ import annotations

import copy
import json
import math
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

ROOT = Path(__file__).resolve().parents[2]


class TreeGame:
    def __init__(self, path=(), terminal=False):
        self.path = path
        self.outcome = SimpleNamespace(terminal=terminal)
        self.closed = False

    def clone(self):
        return TreeGame(self.path, self.outcome.terminal)

    def close(self):
        self.closed = True


class TreeSpec:
    game_id = "tree"
    num_players = 2
    perfect_information = True
    zero_sum = True

    def actors(self, game):
        return [] if game.outcome.terminal else [len(game.path) % 2]

    def legal_actions(self, game, player):
        return list(range(18)) if not game.path else [0, 1]

    def step(self, game, actions):
        from hasami_learn.games.base import StepOutcome
        game.path += (next(iter(actions.values())),)
        return StepOutcome(False)

    def action_name(self, action):
        return str(action)

    def terminal_values(self, game):
        return [1.0, -1.0]


class TreeEvaluator:
    def __init__(self, values=None):
        self.spec = TreeSpec()
        self.values = values or {a: 0.0 for a in range(18)}
        self.batch_lengths = []

    def one_ply(self, game, cancel=None):
        return self.values

    def evaluate(self, games, cancel=None):
        import numpy as np
        from hasami_learn.practical import Evaluation
        self.batch_lengths.append(len(games))
        rows = []
        for game in games:
            prior = {a: 0.998 / 16 if a < 16 else 0.001 for a in range(18)}
            if len(game.path) == 1:
                ps = 0.2 if game.path[0] == 0 else 0.4
            elif len(game.path) >= 2:
                ps = 0.8 if game.path[1] == 0 else 0.2
            else:
                ps = 0.5
            rows.append(Evaluation(prior, 0, np.full((3, 3), ps), np.full(3, 0.3)))
        return rows


@requires_torch
class PracticalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from hasami_learn.config import TrainingConfig
        from hasami_learn.trainer import train_from_config
        from hasami_learn.practical import create_advisor
        import random
        import numpy as np
        import torch
        cls.addClassCleanup(random.setstate, random.getstate())
        cls.addClassCleanup(np.random.set_state, np.random.get_state())
        cls.addClassCleanup(torch.set_rng_state, torch.get_rng_state())
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        cls.started = time.perf_counter()
        cls.temp = tempfile.TemporaryDirectory(prefix="practical-tests-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.raw = json.loads((ROOT / "configs/tiny/practical-hasami.json").read_text())
        cls.raw["run"]["output_dir"] = str(cls.root)
        cls.checkpoint = train_from_config(TrainingConfig.from_dict(copy.deepcopy(cls.raw)), echo=False)
        cls.advisor = create_advisor(str(cls.checkpoint), "cpu", method="learned")
        cls.levels = cls.advisor.level_data
        cls.engine = cls.advisor.engine_checkpoint
        cls.loso = {}
        for style in cls.levels["train_styles"]:
            data = copy.deepcopy(cls.levels)
            data["train_styles"].remove(style)
            data["heldout_styles"].append(style)
            for row in data["levels"]:
                params = row["styles"]["checkpoint"]["params"]
                params["checkpoint"] = str(cls.checkpoint / params["checkpoint"])
            levels_file = cls.root / f"levels-without-{style}.json"
            levels_file.write_text(json.dumps(data))
            raw = copy.deepcopy(cls.raw)
            raw["run"]["run_id"] = f"without-{style}"
            raw["algorithm"]["params"].update(bootstrap_tiny=False, engine_checkpoint=str(cls.engine),
                                                levels_file=str(levels_file), on_policy_rate=0)
            cls.loso[style] = str(train_from_config(TrainingConfig.from_dict(raw), echo=False))

    def game(self, backend="cpp"):
        game = self.advisor.spec.create_game(backend)
        self.addCleanup(game.close)
        return game

    def test_proxy_legality_distributions_seed_and_no_mutation(self):
        from hasami.agents import SearchLimits
        game = self.game()
        snapshot = (game.position, list(game.moves), list(game.history_keys()))
        self.assertEqual(len(game.legal_actions()), 63)
        self.assertEqual(game.perft(2), 3717)
        for style in ("qre", "temperature", "checkpoint", "alphabeta", "mixture"):
            with self.subTest(style=style):
                first = self.advisor.proxy_agent(2, style=style, seed=7)
                second = self.advisor.proxy_agent(2, style=style, seed=7)
                probabilities = first.distribution(game)
                self.assertAlmostEqual(sum(probabilities.values()), 1)
                self.assertTrue(set(probabilities) <= set(game.legal_actions()))
                self.assertTrue(all(p >= 0 for p in probabilities.values()))
                a = [first.choose(game, SearchLimits()).action for _ in range(3)]
                b = [second.choose(game, SearchLimits()).action for _ in range(3)]
                self.assertEqual(a, b)
                self.assertTrue(all(type(action) is int and action in game.legal_actions() for action in a))
        self.assertEqual(snapshot, (game.position, list(game.moves), list(game.history_keys())))

    def test_qre_one_ply_and_visited_mcts_values(self):
        import numpy as np
        from hasami_learn.practical import ProxyAgent, normalize
        from hasami_learn.search.mcts import MctsResult
        game = self.game()
        engine = self.advisor.engine
        prior = engine.evaluate([game])[0].prior
        q = engine.one_ply(game)
        agent = ProxyAgent(engine, "qre", {"beta": 2, "simulations": 0})
        expected = normalize({a: p**0.5 * math.exp(2*q[a]) for a, p in prior.items()})
        actual = agent.distribution(game)
        np.testing.assert_allclose(list(actual.values()), list(expected.values()))
        values = np.full(engine.spec.action_size, np.nan)
        action = min(q)
        values[action] = 1
        found = MctsResult(np.zeros_like(values), 0, 2, values)
        agent = ProxyAgent(engine, "qre", {"beta": 2, "simulations": 2})
        with patch("hasami_learn.practical.MCTS.run_many", return_value=[found]):
            actual = agent.distribution(game)
        q[action] = 1
        expected = normalize({a: p**0.5 * math.exp(2*q[a]) for a, p in prior.items()})
        np.testing.assert_allclose(list(actual.values()), list(expected.values()))

    def test_search_preserves_calibrated_proxies_and_naturalness(self):
        from hasami_learn.practical import PracticalAdvisor
        from hasami_learn.practical_search import SearchAdvisor
        from hasami_learn.practical_eval import _agents
        levels = copy.deepcopy(self.levels)
        for row in levels["levels"]:
            row["styles"]["qre"]["params"].update(simulations=64, beta=8)
        learned = PracticalAdvisor(self.advisor.engine, self.advisor.practical, levels,
                                   checkpoint_root=self.checkpoint)
        search = SearchAdvisor(self.advisor.engine, levels, checkpoint_root=self.checkpoint)
        game = self.game()
        self.assertEqual(search.levels, learned.levels)
        for style in ("qre", "mixture"):
            with self.subTest(style=style):
                expected = learned.proxy_agent(1, style=style).distribution(game)
                actual = search.proxy_agent(1, style=style).distribution(game)
                self.assertEqual(actual, expected)
        for seed in range(6):
            for actual, expected in zip(_agents(search, (1, 2), seed, "mixture"),
                                        _agents(learned, (1, 2), seed, "mixture")):
                self.assertEqual(actual.params, expected.params)
                self.assertEqual(actual.distribution(game), expected.distribution(game))
        expected = learned.proxy_agent(1).distribution(game)
        for row in search.advise(game, 1, depth=1):
            self.assertAlmostEqual(row.naturalness, expected[row.action])
        # Only root naturalness may use calibrated MCTS; expectimax must use one-ply values.
        original = search.proxy_agent
        def proxy(level, **kwargs):
            result = original(level, **kwargs)
            if kwargs.get("style", "mixture") == "mixture":
                result.distribution = lambda game, cancel=None: expected
            return result
        with patch.object(search, "proxy_agent", side_effect=proxy), patch(
                "hasami_learn.practical.MCTS.run_many", side_effect=AssertionError("search used MCTS")):
            advice = search.advise(game, 1, depth=2)
        for row in advice:
            self.assertAlmostEqual(row.naturalness, expected[row.action])
        self.assertEqual(search.level_data, levels)

    def test_search_evaluation_skips_unavailable_qre_levels(self):
        from hasami_learn.practical import validate_levels, validate_artifact
        from hasami_learn.practical_search import SearchAdvisor
        from hasami_learn.practical_eval import run_experiment
        levels = copy.deepcopy(self.levels)
        levels["k"], levels["levels"] = 2, levels["levels"][:2]
        levels["train_styles"], levels["heldout_styles"] = ["qre", "temperature"], []
        levels["levels"][1]["styles"]["qre"].update(available=False, elo=None, ci=None)
        validate_levels(levels)
        search = SearchAdvisor(self.advisor.engine, levels, checkpoint_root=self.checkpoint)
        for name in ("checkpoint_dir", "engine_checkpoint", "levels_path"):
            setattr(search, name, getattr(self.advisor, name))
        with patch("hasami_learn.practical_eval.create_advisor", return_value=search):
            for experiment in ("E1", "E3", "E4", "E7"):
                with self.subTest(experiment=experiment):
                    report = run_experiment(experiment, str(self.checkpoint),
                        output=self.root / f"partial-{experiment}.json", smoke=True,
                        depth=1, positions=1, rollouts=1, max_plies=2, reference_simulations=2)
                    validate_artifact(report, "eval")
                    self.assertEqual(report["config"]["levels"], [1])
                    self.assertEqual(report["config"]["search_qre_params"]["1"]["simulations"], 0)
            for kwargs in ({"level": 2}, {"opponent_level": 2}):
                with self.assertRaisesRegex(ValueError, "qre is unavailable at level 2"):
                    run_experiment("E7", str(self.checkpoint), output=self.root / "bad-level.json",
                                   smoke=True, **kwargs)

    def test_calibration_bisection_unreachable_and_monotonicity_failure(self):
        from hasami_learn.practical_calibrate import bisect_strength
        from hasami_learn.practical import validate_levels
        measure = lambda x: dict(elo=x*1000, ci=[x*1000-10, x*1000+10], games=200)
        row = bisect_strength(measure, 350, tolerance=1)
        self.assertTrue(row["available"])
        self.assertLessEqual(abs(row["elo"] - 350), 1)
        self.assertFalse(bisect_strength(measure, 1200)["available"])
        with self.assertRaisesRegex(ValueError, "monotone"):
            bisect_strength(lambda x: dict(elo=-x, ci=[-x-1, -x+1], games=2), 0)
        data = copy.deepcopy(self.levels)
        data["levels"][1]["styles"]["qre"] = copy.deepcopy(data["levels"][0]["styles"]["qre"])
        with self.assertRaisesRegex(ValueError, "non-increasing"):
            validate_levels(data)
        data = copy.deepcopy(self.levels)
        data["extra"] = True
        with self.assertRaises(Exception):
            validate_levels(data)

    def test_masked_practical_cell_and_slip_gradient(self):
        import torch
        from hasami_learn.algorithms.practical import masked_practical_loss
        logits = torch.zeros((2, 3, 3), requires_grad=True)
        loss = masked_practical_loss(logits, torch.tensor([0, 2]), torch.tensor([1, 0]), torch.tensor([1., 0.]))
        loss.backward()
        nonzero = torch.nonzero(logits.grad).tolist()
        self.assertEqual(nonzero, [[0, 0, 1], [1, 2, 0]])
        self.assertLess(logits.grad[0, 0, 1], 0)
        self.assertGreater(logits.grad[1, 2, 0], 0)
        slip = torch.zeros((2, 3), requires_grad=True)
        torch.nn.functional.binary_cross_entropy_with_logits(
            slip[torch.arange(2), torch.tensor([0, 2])], torch.tensor([1., 0.])).backward()
        self.assertEqual(torch.nonzero(slip.grad).tolist(), [[0, 0], [1, 2]])

    def tree_advisor(self, values=None):
        from hasami_learn.practical import PracticalAdvisor
        levels = copy.deepcopy(self.levels)
        levels["game_id"] = "tree"
        engine, practical = TreeEvaluator(values), TreeEvaluator()
        advisor = PracticalAdvisor(engine, practical, levels)
        proxy = SimpleNamespace(distribution=lambda game, cancel=None: ({0: 0.75, 1: 0.25} if game.path else
                                  {a: 1/18 for a in range(18)}))
        advisor.proxy_agent = lambda *args, **kwargs: proxy
        return advisor

    def test_advisor_ranking_guard_fallback_and_top16(self):
        game = TreeGame()
        advisor = self.tree_advisor()
        advice = advisor.advise(game, 1)
        self.assertEqual([a.action for a in advice], [0, 1, 2])
        self.assertAlmostEqual(advice[0].practical_score, 0.8)
        self.assertAlmostEqual(advice[0].difficulty, 0.3)
        guarded = self.tree_advisor({a: 0.2 if a == 1 else 0.0 for a in range(18)})
        self.assertEqual(guarded.advise(game, 1, guard=0.1)[0].action, 1)
        loose = guarded.advise(game, 1, guard=0.3)[0]
        self.assertEqual(loose.action, 0)
        self.assertAlmostEqual(loose.value_loss, 0.2)
        advisor = self.tree_advisor({a: 1.0 if a == 17 else 0.0 for a in range(18)})
        advice = advisor.advise(game, 1, guard=0.3)
        self.assertEqual(len(advice), 1)
        self.assertEqual(advice[0].action, 17)
        self.assertTrue(advice[0].guard_fallback)
        self.assertEqual(advice[0].value_loss, 0)
        self.assertEqual(game.path, ())
        with self.assertRaises(ValueError):
            advisor.advise(game, 0)
        with self.assertRaises(ValueError):
            advisor.advise(game, 1, guard=-1)

    def test_depth_two_expectimax_batched_arithmetic(self):
        advisor = self.tree_advisor()
        original = advisor.practical.evaluate
        def asymmetric(games, cancel=None):
            rows = original(games, cancel)
            for game, row in zip(games, rows):
                if len(game.path) == 1:
                    row.practical[1, 0] = 0.9
                    row.practical[0, 1] = 0.1
                else:
                    row.practical[1, 0] = 0.95
                row.slip[1:] = 0.8
            return rows
        advisor.practical.evaluate = asymmetric
        shallow = advisor.advise(TreeGame(), 1, opponent_level=2, depth=1)
        self.assertAlmostEqual(shallow[0].practical_score, 0.1)
        advice = advisor.advise(TreeGame(), 1, opponent_level=2, depth=2)
        self.assertAlmostEqual(advice[0].practical_score, 0.75*0.8 + 0.25*0.2)
        self.assertAlmostEqual(advice[0].difficulty, 0.3)
        self.assertIn(16, advisor.practical.batch_lengths)
        self.assertIn(32, advisor.practical.batch_lengths)

    def test_terminal_child_scores_and_adapter_hooks(self):
        from hasami_learn.games.base import StepOutcome
        advisor = self.tree_advisor()
        def step(game, actions):
            game.path += (next(iter(actions.values())),)
            game.outcome.terminal = True
            return StepOutcome(True)
        advisor.spec.step = step
        advisor.spec.action_to_move = lambda game, action: action + 100
        advisor.spec.move_text = lambda game, action: f"move:{action}"
        advice = advisor.advise(TreeGame(), 1)[0]
        self.assertEqual(advice.practical_score, 1)
        self.assertEqual(advice.difficulty, 0)
        self.assertEqual(advice.move, advice.action + 100)
        self.assertEqual(advice.text, f"move:{advice.action}")

    def test_difficulty_uses_next_own_decision_with_extra_turns(self):
        for player in (0, 1):
            for depth in (1, 2):
                with self.subTest(player=player, depth=depth):
                    advisor = self.tree_advisor()
                    advisor.spec.actors = lambda game: [player if len(game.path) < 2 else 1 - player]
                    original = advisor.practical.evaluate

                    def asymmetric(games, cancel=None):
                        rows = original(games, cancel)
                        for game, row in zip(games, rows):
                            row.slip[:] = [0.1 if len(game.path) == 1 else 0.9, 0.7, 0.8]
                        return rows

                    advisor.practical.evaluate = asymmetric
                    game = TreeGame()
                    advice = advisor.advise(game, 1, opponent_level=2, depth=depth)
                    self.assertTrue(advice)
                    for row in advice:
                        self.assertAlmostEqual(row.difficulty, 0.1)
                    self.assertEqual(game.path, ())

    def test_difficulty_unknown_when_opponent_keeps_turn_beyond_horizon(self):
        advisor = self.tree_advisor()
        advisor.spec.actors = lambda game: [0 if not game.path else 1]
        for depth in (1, 2):
            with self.subTest(depth=depth):
                for row in advisor.advise(TreeGame(), 1, opponent_level=2, depth=depth):
                    self.assertIsNone(row.difficulty)

    def test_checkpoint_factories_legal_ranges_and_cancel(self):
        from hasami.agents import AgentCancelled, SearchLimits
        from hasami_learn.inference import create_agent
        from hasami_learn.practical import create_practical_agent, create_proxy_agent
        from hasami_learn.practical_search import SearchAdvisor
        game = self.game()
        snapshot = (game.position, list(game.history_keys()))
        learned = create_practical_agent(str(self.checkpoint), device="cpu", method="learned")
        self.assertNotIsInstance(learned.advisor, SearchAdvisor)
        self.assertIn(learned.choose(game, SearchLimits()).action, game.legal_actions())
        proxy = create_proxy_agent(str(self.checkpoint), device="cpu", method="learned")
        self.assertIn(proxy.choose(game, SearchLimits()).action, game.legal_actions())
        for factory in (create_agent, create_practical_agent, create_proxy_agent):
            agent = factory(str(self.checkpoint), device="cpu")
            self.assertIs(type(agent.choose(game, SearchLimits()).action), int)
            self.assertIn(agent.choose(game, SearchLimits()).action, game.legal_actions())
            cancel = threading.Event()
            cancel.set()
            with self.assertRaises(AgentCancelled):
                agent.choose(game, SearchLimits(), cancel)
        for advice in self.advisor.advise(game, 3, depth=2):
            self.assertIn(advice.action, game.legal_actions())
            self.assertEqual(advice.move, advice.action)
            self.assertTrue(advice.text)
            self.assertTrue(0 <= advice.practical_score <= 1)
            self.assertTrue(-1 <= advice.engine_value <= 1)
            self.assertTrue(0 <= advice.value_loss <= 0.3)
            self.assertTrue(0 <= advice.difficulty <= 1)
            self.assertTrue(0 <= advice.naturalness <= 1)
        self.assertEqual(snapshot, (game.position, list(game.history_keys())))

    def test_agents_config_factories(self):
        from hasami.agents import AgentRegistry, SearchLimits
        entries = [dict(id=name, type="python", factory=f"hasami_learn.practical:{factory}",
                        options=dict(checkpoint=str(self.checkpoint), device="cpu", level=2))
                   for name, factory in (("practical", "create_practical_agent"), ("proxy", "create_proxy_agent"))]
        path = self.root / "agents.json"
        path.write_text(json.dumps({"agents": entries}))
        registry = AgentRegistry()
        registry.load_config(path)
        game = self.game()
        for name in ("practical", "proxy"):
            self.assertIn(registry.create(name).choose(game, SearchLimits()).action, game.legal_actions())

    def test_bundle_and_engine_hash_refusal(self):
        from hasami_learn.checkpoint_bundle import bundle_checkpoint
        from hasami_learn.practical import create_advisor
        from hasami_learn.practical_search import SearchAdvisor
        target = self.root / "bundle"
        bundle_checkpoint(self.checkpoint, target)
        advisor = create_advisor(str(target), "cpu", method="learned")
        search = create_advisor(str(target), "cpu")
        self.assertIsInstance(search, SearchAdvisor)
        self.assertEqual(search.engine_checkpoint, (target / "engine").resolve())
        self.assertEqual(search.level_data, advisor.level_data)
        self.assertIn(search.advise(self.game(), 3, depth=1)[0].action, self.game().legal_actions())
        advisor.proxy_agent(3, style="checkpoint").distribution(self.game())
        self.assertTrue(all(Path(path).is_relative_to(target.resolve()) for path in advisor.checkpoint_engines))
        self.assertIn(advisor.advise(self.game(), 3, top_k=1)[0].action, self.game().legal_actions())
        weights = target / "engine" / "weights.pt"
        with weights.open("ab") as handle:
            handle.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "sha256"):
            create_advisor(str(target), "cpu", method="learned")
        with self.assertRaisesRegex(ValueError, "sha256"):
            create_advisor(str(target), "cpu")

    def test_collect_update_resume_and_frozen_teacher(self):
        import torch
        from hasami_learn.config import TrainingConfig
        from hasami_learn.trainer import Trainer
        raw = copy.deepcopy(self.raw)
        raw["run"]["run_id"] = "state-test"
        raw["algorithm"]["params"].update(bootstrap_tiny=False, engine_checkpoint=str(self.engine),
            levels_file=str(self.checkpoint / "levels.json"), on_policy_rate=0, engine_store=str(self.engine.parent.parent / "selfplay"))
        invalid = copy.deepcopy(raw)
        invalid["opponents"]["snapshot_every"] = 1
        with self.assertRaisesRegex(ValueError, "pool snapshots"):
            Trainer(TrainingConfig.from_dict(invalid), echo=False)
        trainer = Trainer(TrainingConfig.from_dict(raw), echo=False)
        self.addCleanup(trainer.close)
        learner = trainer.learner
        self.assertGreater(len(learner.starts), 0)
        import numpy as np
        incompatible = self.root / "incompatible-store"
        (incompatible / "games").mkdir(parents=True)
        metadata = dict(game_id=learner.spec.game_id, game_params={}, ruleset_id=learner.spec.ruleset_id, rules={})
        np.savez(incompatible / "games" / "bad.npz", metadata=json.dumps(metadata), moves=np.array([], dtype=int))
        with patch.object(learner.params, "engine_store", str(incompatible)):
            with self.assertRaisesRegex(ValueError, "incompatible"):
                learner._store_starts()
        before = {k: v.clone() for k, v in learner.engine.network.state_dict().items()}
        stats = learner.collect(1)
        self.assertEqual(stats.games, 2)
        self.assertEqual(stats.truncated, 2)
        self.assertTrue(all(row["target"] == 0.5 for row in learner.samples))
        self.assertTrue(all(0 <= row["own"] < 3 and 0 <= row["other"] < 3 for row in learner.samples))
        result = learner.update(1)
        self.assertTrue(math.isfinite(result.loss))
        self.assertEqual(result.steps, 2)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in learner.engine.network.state_dict().items()))
        state = learner.state_dict()
        count = len(learner.samples)
        learner.samples = []
        learner.load_state_dict(state)
        self.assertEqual(len(learner.samples), count)
        self.assertTrue(all(p.grad is None for p in learner.engine.network.parameters()))
        from hasami_learn.games.base import StepOutcome
        original_step = learner.spec.step
        learner.starts = []
        learner.samples = []
        def finish_after_two(game, actions):
            original_step(game, actions)
            return StepOutcome(len(game.moves) == 2)
        with patch.object(learner.spec, "step", side_effect=finish_after_two), \
                patch.object(learner.spec, "terminal_values", return_value=[1.0, -1.0]):
            terminal_stats = learner.collect(2)
        self.assertEqual(terminal_stats.truncated, 0)
        self.assertEqual(terminal_stats.wins, (2, 0, 0))
        for row in learner.samples:
            player = int(row["planes"][4, 0, 0])
            self.assertEqual(row["target"], 1.0 if player == 0 else 0.0)

    def test_e1_common_random_numbers(self):
        from hasami_learn.practical_eval import paired_intervention, decision_rng
        game = self.game()
        action = game.legal_actions()[0]
        traces = {}
        scores, caps = paired_intervention(self.advisor, game, (1, 2), {"engine": action, "practical": action},
                                          [901, 902], max_plies=3, traces=traces)
        self.assertEqual(scores["engine"], scores["practical"])
        self.assertEqual(caps["engine"], caps["practical"])
        for seed in (901, 902):
            self.assertEqual(traces[("engine", seed)], traces[("practical", seed)])
            self.assertEqual(decision_rng(seed, 1, 2).random(), decision_rng(seed, 1, 2).random())

    def test_sampled_positions_exclude_adapter_stopped_states(self):
        from hasami_learn.games.base import StepOutcome
        from hasami_learn.practical_eval import sampled_positions
        for terminal, truncated, ply in ((False, True, 1), (True, False, 1),
                                          (False, False, 1), (False, True, 0)):
            with self.subTest(terminal=terminal, truncated=truncated, ply=ply):
                advisor = self.tree_advisor()
                created = []

                def create_game(backend, seed=None):
                    game = TreeGame()
                    created.append(game)
                    return game

                def step(game, actions):
                    if getattr(game, "stopped", False):
                        raise RuntimeError("step after adapter stop")
                    game.path += (next(iter(actions.values())),)
                    game.stopped = terminal or truncated
                    return StepOutcome(terminal, truncated)

                advisor.spec.create_game = create_game
                advisor.spec.step = step
                rng = SimpleNamespace(integers=lambda *args: ply, choice=lambda actions, p=None: actions[0])
                with patch("hasami_learn.practical_eval.decision_rng", return_value=rng):
                    positions = sampled_positions(advisor, (1, 2), 17, count=1, max_plies=4, backend="python")
                for sample in positions:
                    self.addCleanup(sample.game.close)
                self.assertEqual(len(positions), int(ply == 0 or not (terminal or truncated)))
                self.assertTrue(all(game.closed for game in created))
                self.assertTrue(all(not game.outcome.terminal for game in created))
                if positions:
                    sample = positions[0]
                    game, phase = sample.game, sample.phase
                    self.assertEqual((sample.index, sample.seed), (0, 17))
                    self.assertEqual(len(game.path), ply)
                    self.assertEqual(phase, "early")
                    self.assertFalse(game.closed)

    def test_continuation_keeps_adapter_reported_terminal_results(self):
        # Review finding: an adapter may report terminal=True with rewards while the engine outcome stays ongoing;
        # the continuation must score that result (like run_table_game), not turn it into a truncated half point.
        from hasami_learn.games.base import StepOutcome
        from hasami_learn.practical_eval import continuation

        for rewards, expected in (([1.0, -1.0], [1.0, 0.0]), ([-1.0, 1.0], [0.0, 1.0]), ([0.0, 0.0], [0.5, 0.5])):
            with self.subTest(rewards=rewards):
                advisor = self.tree_advisor()

                def step(game, actions, rewards=rewards):
                    game.path += (next(iter(actions.values())),)
                    return StepOutcome(True, False, list(rewards), {"reason": "adapter"})

                advisor.spec.step = step
                game = TreeGame()
                self.addCleanup(game.close)
                scores, truncated = continuation(advisor, game, (1, 1), 5, max_plies=4)
                self.assertFalse(truncated)
                self.assertEqual(scores, expected)
        # A real cap still counts as truncation with half points.
        advisor = self.tree_advisor()
        advisor.spec.step = lambda game, actions: (setattr(game, "path", game.path + (next(iter(actions.values())),)),
                                                    StepOutcome(False))[1]
        game = TreeGame()
        self.addCleanup(game.close)
        self.assertEqual(continuation(advisor, game, (1, 1), 5, max_plies=1), ([0.5, 0.5], True))

    def test_omitted_style_defaults_train_and_evaluate(self):
        from hasami_learn.config import TrainingConfig
        from hasami_learn.practical import create_advisor, validate_artifact, validate_levels
        from hasami_learn.practical_eval import run_experiment
        from hasami_learn.trainer import train_from_config
        data = copy.deepcopy(self.levels)
        del data["train_styles"], data["heldout_styles"]
        for row in data["levels"]:
            params = row["styles"]["checkpoint"]["params"]
            params["checkpoint"] = str(self.checkpoint / params["checkpoint"])
        validate_levels(data)
        levels_file = self.root / "implicit-styles.json"
        levels_file.write_text(json.dumps(data))
        raw = copy.deepcopy(self.raw)
        raw["run"]["run_id"] = "implicit-styles"
        raw["algorithm"]["params"].update(bootstrap_tiny=False, engine_checkpoint=str(self.engine),
                                            levels_file=str(levels_file), on_policy_rate=0)
        checkpoint = train_from_config(TrainingConfig.from_dict(raw), echo=False)
        saved_path = checkpoint / "levels.json"
        saved = json.loads(saved_path.read_text())
        self.assertEqual(saved["train_styles"], ["qre", "temperature", "checkpoint"])
        self.assertEqual(saved["heldout_styles"], ["alphabeta"])
        del saved["train_styles"], saved["heldout_styles"]
        saved_path.write_text(json.dumps(saved))
        advisor = create_advisor(str(checkpoint), "cpu", method="learned")
        self.assertEqual([agent.style for agent in advisor.proxy_agent(1).agents],
                         ["qre", "temperature", "checkpoint"])
        output = self.root / "implicit-styles-e2.json"
        result = run_experiment("E2", str(checkpoint), output=output, smoke=True, method="learned",
                                positions=1, rollouts=1, max_plies=2, reference_simulations=2,
                                leave_one_out=self.loso)
        validate_artifact(result, "eval")
        self.assertEqual(result["config"]["levels"], [1, 2, 3])
        self.assertEqual(result["config"]["train_styles"], ["qre", "temperature", "checkpoint"])
        self.assertEqual(result["config"]["heldout_styles"], ["alphabeta"])
        self.assertTrue(any(row.get("style") == "alphabeta" and row["n"] > 0 for row in result["results"]))
        self.assertEqual(json.loads(output.with_suffix(".done.json").read_text()), result)

    def test_e1_to_e7_reports_intervals_and_done_files(self):
        from hasami_learn.practical_eval import EXPERIMENTS, run_experiment
        from hasami_learn.practical import validate_artifact
        for experiment in EXPERIMENTS:
            with self.subTest(experiment=experiment):
                output = self.root / f"{experiment}.json"
                result = run_experiment(experiment, str(self.checkpoint), output=output, method="learned",
                    smoke=True, positions=1, rollouts=1, max_plies=2, reference_simulations=2,
                    leave_one_out=self.loso if experiment == "robustness" else None,
                    level=None if experiment == "levels" else 1)
                validate_artifact(result, "eval")
                self.assertTrue(result["smoke"])
                self.assertEqual(result["status"], "complete")
                self.assertTrue(result["seeds"])
                self.assertTrue(result["config"])
                self.assertTrue(result["created_at"])
                self.assertTrue(result["results"])
                if experiment in ("single_intervention", "robustness"):
                    self.assertTrue(any(row["metric"] == "score_difference" and row["n"] > 0
                                        for row in result["results"]))
                self.assertEqual(json.loads(output.with_suffix('.done.json').read_text()), result)
                for row in result["results"]:
                    if row["value"] is not None:
                        self.assertLessEqual(row["ci_low"], row["value"])
                        self.assertLessEqual(row["value"], row["ci_high"])
        self.assertLess(time.perf_counter() - self.started, 180)

    def test_one_ply_identity_terminal_truncation_and_clone_cleanup(self):
        from hasami_learn.practical import NetworkEvaluator, Evaluation
        from hasami_learn.games.base import StepOutcome
        spec = TreeSpec()
        spec.legal_actions = lambda game, player: [0, 1, 2, 3]
        spec.actors = lambda game: [] if game.outcome.terminal else ([0] if game.path == (0,) else [len(game.path) % 2])
        def step(game, actions):
            game.path += (next(iter(actions.values())),)
            game.outcome.terminal = game.path == (2,)
            return StepOutcome(game.outcome.terminal, game.path == (3,))
        spec.step = step
        evaluator = object.__new__(NetworkEvaluator)
        evaluator.spec = spec
        evaluator.evaluate = lambda games, cancel=None: [Evaluation({}, 0.4) for g in games]
        root = TreeGame()
        children = []
        def clone():
            child = TreeGame()
            children.append(child)
            return child
        root.clone = clone
        values = evaluator.one_ply(root)
        self.assertEqual(values, {0: 0.4, 1: -0.4, 2: 1.0, 3: 0.0})
        self.assertTrue(all(child.closed for child in children))
        self.assertEqual(root.path, ())

    def test_unavailable_style_and_invalid_parameters_fail(self):
        from hasami_learn.practical import PracticalAdvisor, ProxyAgent
        from hasami_learn.algorithms.practical import PracticalParams
        for params in ({"simulations": -1}, {"simulations": True}, {"epsilon": 2}, {"beta": float("nan")}):
            with self.assertRaises(ValueError):
                ProxyAgent(self.advisor.engine, "qre", params)
        with self.assertRaises(ValueError):
            PracticalParams(on_policy_rate=2)
        with self.assertRaises(ValueError):
            PracticalParams(games_per_iteration=0)
        data = copy.deepcopy(self.levels)
        data["levels"][0]["styles"]["qre"]["available"] = False
        advisor = PracticalAdvisor(self.advisor.engine, self.advisor.practical, data, checkpoint_root=self.checkpoint)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            advisor.proxy_agent(1, style="qre")
        self.assertNotIn("qre", [agent.style for agent in advisor.proxy_agent(1).agents])

    def test_runtime_import_has_no_training_modules(self):
        code = """
import sys
import hasami_learn.practical
for name in ('trainer', 'cli', 'replay', 'datastore', 'parallel', 'datasets', 'selfplay',
             'batched_selfplay', 'opponents', 'evaluation', 'ladder', 'algorithms'):
    prefix = 'hasami_learn.' + name
    assert not any(m == prefix or m.startswith(prefix + '.') for m in sys.modules), prefix
"""
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
