"""A1-A8 conformance for classical, configured external, and learned agents."""

from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import random
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from hasami import create_game
from hasami.agents import AgentCancelled, AgentRegistry, RandomAgent, SearchLimits
from hasami.agents import SubprocessAgent
from hasami.env import TurnBasedEnv
from hasami.model import Rules
from hasami.records import from_record
from tests.learn._common import TORCH_AVAILABLE, requires_torch
from tests.test_rules import native_available

from ._agents_support import CYCLE, classical_cases, position, random_records, root_state


LEARNED_RUNS = {}
RANDOM_RECORDS = []
NATIVE_AVAILABLE = native_available()


def setUpModule():
    """Build seeded positions and train each optional algorithm exactly once."""
    RANDOM_RECORDS[:] = random_records()
    LEARNED_RUNS.clear()
    unittest.addModuleCleanup(LEARNED_RUNS.clear)
    if not TORCH_AVAILABLE:
        return

    import numpy as np
    import torch
    import hasami_learn.algorithms  # Register algorithms before enumerating them.
    from hasami_learn.config import tiny_config
    from hasami_learn.registry import ALGORITHMS
    from hasami_learn.trainer import train_from_config

    unittest.addModuleCleanup(random.setstate, random.getstate())
    unittest.addModuleCleanup(np.random.set_state, np.random.get_state())
    unittest.addModuleCleanup(torch.set_rng_state, torch.get_rng_state())
    unittest.addModuleCleanup(torch.set_num_threads, torch.get_num_threads())
    temporary = tempfile.TemporaryDirectory(prefix="hasami-agent-conformance-")
    unittest.addModuleCleanup(temporary.cleanup)
    params = {
        "alphazero": {"simulations": 2, "batch_games": 2, "eval_simulations": 32},
        "supervised": {"generate_games": 2, "generator": "random"},
    }
    for name in ALGORITHMS.names():
        if not getattr(ALGORITHMS.get(name), "conformance_tiny", True):
            continue  # needs prepared artifacts (e.g. practical: an engine checkpoint); covered by its own tests
        config = tiny_config(name, run_id=f"conformance-{name}", **params.get(name, {}))
        config.run.output_dir = temporary.name
        train_from_config(config, echo=False)
        LEARNED_RUNS[name] = Path(temporary.name) / config.run.run_id


class AgentProperties:
    backend = "python"

    def setUp(self):
        self.cases = classical_cases()

    def assert_legal_decision(self, decision, legal):
        self.assertIs(type(decision.action), int)
        self.assertIn(decision.action, legal)

    def test_A1_A2_random_legality_and_root_immutability(self):
        """A1/A2: Every agent preserves 30 random roots and returns legal plain ints."""
        self.assertGreaterEqual(len(RANDOM_RECORDS), 30)
        turns = [record["final"]["position"]["turn"] for record in RANDOM_RECORDS]
        self.assertGreaterEqual(turns.count(0), 15)
        self.assertGreaterEqual(turns.count(1), 15)
        for name, agent, limits in self.cases:
            for index, record in enumerate(RANDOM_RECORDS):
                with self.subTest(agent=name, position=index, backend=self.backend):
                    with closing(from_record(record, self.backend)) as game:
                        self.assertFalse(game.outcome.terminal)
                        legal = game.legal_actions()
                        before = root_state(game)
                        decision = agent.choose(game, limits)
                        self.assertEqual(root_state(game), before)
                        self.assert_legal_decision(decision, legal)

    def test_A2_pending_and_repeated_history_survive_search(self):
        """A2: Search preserves pending replies, capture state, and repeated history."""
        for name, agent, limits in self.cases:
            for leader in (0, 1):
                with self.subTest(agent=name, pending_leader=leader):
                    black = [38, 67, 72, 73, 74, 75, 76, 77, 78]
                    white = [39, 37, 3, 0, 1, 2, 4]
                    initial = (position(black, white) if leader == 0 else
                               position(white, black, turn=1))
                    with closing(create_game(self.backend, position=initial)) as game:
                        game.play(67 * 81 + 40)
                        self.assertEqual(game.position.pending, leader)
                        self.assertNotEqual(game.last_capture, 0)
                        before = root_state(game)
                        decision = agent.choose(game, limits)
                        self.assertEqual(root_state(game), before)
                        self.assert_legal_decision(decision, game.legal_actions())
            with self.subTest(agent=name, history="third occurrence"):
                with closing(create_game(self.backend)) as game:
                    for action in CYCLE * 2:
                        game.play(action)
                    before = root_state(game)
                    self.assertEqual(game.history_keys().count(game.position.key()), 3)
                    decision = agent.choose(game, limits)
                    self.assertEqual(root_state(game), before)
                    self.assert_legal_decision(decision, game.legal_actions())
                    for action in CYCLE:
                        game.play(action)
                    self.assertEqual(game.outcome.reason, "repetition")
                    self.assertTrue(game.undo())
                    self.assertFalse(game.outcome.terminal)

    def test_A3_terminal_and_no_legal_roots_raise(self):
        """A3: Won, repetition-drawn, and geometrically blocked roots raise ValueError."""
        for name, agent, limits in self.cases:
            for kind in ("five_captures", "repetition", "no_moves"):
                with self.subTest(agent=name, terminal=kind):
                    if kind == "five_captures":
                        game = create_game(self.backend, position=position(range(72, 81), range(4)))
                    elif kind == "no_moves":
                        game = create_game(self.backend, Rules(capture_target=0, lead_target=0),
                                           position([0], [1, 9]))
                    else:
                        game = create_game(self.backend)
                    with closing(game):
                        if kind == "repetition":
                            for action in CYCLE * 3:
                                game.play(action)
                        self.assertEqual(game.outcome.reason, kind)
                        self.assertEqual(game.legal_actions(), [])
                        before = root_state(game)
                        with self.assertRaises(ValueError):
                            agent.choose(game, limits)
                        self.assertEqual(root_state(game), before)

    def test_A4_preset_cancel_is_honoured(self):
        """A4: A pre-set cancel event raises AgentCancelled without changing the root."""
        cancel = threading.Event()
        cancel.set()
        for name, agent, limits in self.cases:
            with self.subTest(agent=name), closing(create_game(self.backend)) as game:
                before = root_state(game)
                with self.assertRaises(AgentCancelled):
                    agent.choose(game, limits, cancel)
                self.assertEqual(root_state(game), before)

    def test_A4_tiny_time_budget_returns_legal_action(self):
        """A4: A one-millisecond soft search budget still produces a legal action."""
        for name, agent, limits in self.cases:
            with self.subTest(agent=name), closing(create_game(self.backend)) as game:
                before = root_state(game)
                decision = agent.choose(game, SearchLimits(limits.max_depth, 1, 200000))
                self.assert_legal_decision(decision, game.legal_actions())
                self.assertEqual(root_state(game), before)

    def test_A5_environment_games_alternate_until_done(self):
        """A5: Every agent plays both colours legally until termination or an explicit cap."""
        cap = 40
        for name, agent, limits in self.cases:
            for agent_side in (0, 1):
                with self.subTest(agent=name, side=agent_side):
                    with closing(TurnBasedEnv(self.backend, max_plies=cap)) as env:
                        players = [RandomAgent(55), RandomAgent(56)]
                        players[agent_side] = agent
                        observation = env.reset()
                        for ply in range(cap):
                            actor = env.game.position.turn
                            self.assertEqual(actor, ply % 2)
                            self.assertEqual(observation["to_play"], actor)
                            legal = env.game.legal_actions()
                            before = root_state(env.game)
                            with closing(env.game.clone()) as snapshot:
                                decision = players[actor].choose(snapshot, limits)
                                self.assertEqual(root_state(snapshot), before)
                            self.assertEqual(root_state(env.game), before)
                            self.assert_legal_decision(decision, legal)
                            result = env.step(decision.action)
                            self.assertEqual(result.info["actor"], actor)
                            self.assertEqual(env.game.position.turn, 1 - actor)
                            self.assertEqual(len(env.game.moves), ply + 1)
                            observation = result.observation
                            if result.terminated or result.truncated:
                                break
                        self.assertTrue(result.terminated or result.truncated)
                        self.assertEqual(result.terminated, env.game.outcome.terminal)
                        if result.truncated:
                            self.assertFalse(result.terminated)
                            self.assertEqual(len(env.game.moves), cap)
                            self.assertEqual(env.game.outcome.winner, -2)
                            self.assertEqual(result.rewards, (0.0, 0.0))
                        else:
                            self.assertEqual(env.game.legal_actions(), [])
                        with self.assertRaises(ValueError):
                            env.step(decision.action)


class PythonAgentConformance(AgentProperties, unittest.TestCase):
    pass


@unittest.skipUnless(NATIVE_AVAILABLE, "Native library is not built")
class NativeAgentConformance(AgentProperties, unittest.TestCase):
    backend = "cpp"


@requires_torch
class LearnedAgentConformance(AgentProperties, unittest.TestCase):
    def setUp(self):
        from hasami_learn.inference import create_agent
        from hasami_learn.registry import ALGORITHMS

        self.assertTrue(LEARNED_RUNS)
        self.assertEqual(set(LEARNED_RUNS), {name for name in ALGORITHMS.names()
                                             if getattr(ALGORITHMS.get(name), "conformance_tiny", True)})
        self.cases = [(name, create_agent(str(run_dir), device="cpu"),
                       SearchLimits(1, 20, 16))
                      for name, run_dir in LEARNED_RUNS.items()]

    def test_A6_mcts_visits_match_budget_and_legality(self):
        """A6: MctsAgent caps a larger configured search at 16 legal root visits."""
        from hasami_learn.inference import MctsAgent

        cases = [(name, agent) for name, agent, _ in self.cases if isinstance(agent, MctsAgent)]
        self.assertTrue(cases, "The learned matrix must include MctsAgent")
        for name, agent in cases:
            self.assertGreater(agent.simulations, 16)
            for record in RANDOM_RECORDS[:2]:
                with self.subTest(agent=name, turn=record["final"]["position"]["turn"]):
                    with closing(from_record(record, self.backend)) as game:
                        results = []
                        run = agent.search.run

                        def record_search(*args, **kwargs):
                            result = run(*args, **kwargs)
                            results.append(result)
                            return result

                        before = root_state(game)
                        with patch.object(agent.search, "run", side_effect=record_search):
                            decision = agent.choose(game, SearchLimits(node_limit=16))
                        self.assertEqual(len(results), 1)
                        result = results[0]
                        self.assertEqual(result.simulations, 16)
                        self.assertEqual(decision.nodes, 16)
                        self.assertTrue(decision.completed)
                        self.assertEqual(len(result.visits), 6561)
                        self.assertEqual(float(result.visits.sum()), 16.0)
                        legal = set(game.legal_actions())
                        for action, visits in enumerate(result.visits):
                            self.assertGreaterEqual(visits, 0)
                            self.assertEqual(float(visits), int(visits))
                            if action not in legal:
                                self.assertEqual(visits, 0)
                        self.assert_legal_decision(decision, legal)
                        self.assertEqual(root_state(game), before)

    def test_A1_initial_rule_anchors(self):
        """A1: Learner fixtures retain the 63 legal-action and 3717 perft(2) anchors."""
        with closing(create_game(self.backend)) as game:
            actions = game.legal_actions()
            self.assertEqual(len(actions), 63)
            before = root_state(game)
            leaves = 0
            for action in actions:
                game.play(action)
                leaves += len(game.legal_actions())
                self.assertTrue(game.undo())
            self.assertEqual(leaves, 3717)
            self.assertEqual(root_state(game), before)


@unittest.skipUnless(NATIVE_AVAILABLE, "Native library is not built")
class NativeLearnedAgentConformance(LearnedAgentConformance):
    backend = "cpp"


class RegistryAndExternalConformance(unittest.TestCase):
    def test_A7_registry_metadata_and_lifecycle(self):
        """A7: Registry metadata, protected built-ins, duplicates, and removal are explicit."""
        registry = AgentRegistry()
        self.assertEqual(registry.family("human"), "human")
        for name in ("random", "greedy", "alphabeta"):
            with self.subTest(builtin=name):
                self.assertEqual(registry.family(name), "classical")
                self.assertTrue(registry.describe(name))
                with self.assertRaises(ValueError):
                    registry.unregister(name)
                with self.assertRaises(ValueError):
                    registry.register(name, RandomAgent)
                self.assertTrue(callable(registry.create(name).choose))
        registry.register("test_policy", RandomAgent, family="rl", description="Test policy")
        self.assertEqual(registry.family("test_policy"), "rl")
        self.assertEqual(registry.describe("test_policy"), "Test policy")
        with self.assertRaises(ValueError):
            registry.register("test_policy", RandomAgent, family="custom")
        self.assertEqual(registry.family("test_policy"), "rl")
        registry.unregister("test_policy")
        self.assertNotIn("test_policy", registry.names())
        for method in (registry.create, registry.family, registry.describe, registry.unregister):
            with self.subTest(method=method.__name__), self.assertRaises(ValueError):
                method("test_policy")
        registry.register("test_policy", RandomAgent)
        self.assertEqual(registry.family("test_policy"), "custom")

    def test_A8_invalid_external_responses_raise(self):
        """A8: Protocol mismatch, illegal moves, and invalid action types are rejected."""
        responses = [
            ({"protocol": "wrong-version", "action": 5895}, "protocol mismatch"),
            ({"protocol": "hasami-agent-v1", "action": 0}, "illegal action"),
            ({"protocol": "hasami-agent-v1", "action": True}, None),
            ({"protocol": "hasami-agent-v1", "action": 5895.0}, None),
            ({"protocol": "hasami-agent-v1", "action": 6561}, None),
            ({"protocol": "hasami-agent-v1"}, None),
        ]
        for response, message in responses:
            with self.subTest(response=response), closing(create_game("python")) as game:
                code = f"print({json.dumps(response)!r})"
                agent = SubprocessAgent([sys.executable, "-c", code])
                before = root_state(game)
                with self.assertRaisesRegex(ValueError, message or ".+"):
                    agent.choose(game, SearchLimits(time_ms=1000))
                self.assertEqual(root_state(game), before)

    def test_A8_external_timeout_is_explicit(self):
        """A8: A sleeping direct child raises TimeoutError and leaves the root unchanged."""
        agent = SubprocessAgent([sys.executable, "-c", "import time; time.sleep(5)"],
                                startup_ms=0)
        with closing(create_game("python")) as game:
            before = root_state(game)
            with self.assertRaisesRegex(TimeoutError, "process budget"):
                agent.choose(game, SearchLimits(time_ms=30))
            self.assertEqual(root_state(game), before)


if __name__ == "__main__":
    unittest.main()
