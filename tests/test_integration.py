import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import threading
import unittest
from hasami import create_game
from hasami.agents import (AgentRegistry, RandomAgent, GreedyAgent, AlphaBetaAgent,
                          SearchLimits, SubprocessAgent, AgentCancelled, evaluate)
from hasami.env import TurnBasedEnv
from hasami.model import Position, Rules
from hasami.observation import observe
from hasami.records import record_dict, from_record, save_record, load_record
from tests.test_rules import native_available, setup, mask


class IntegrationTests:
    backend = "python"

    def game(self, position=None, rules=None):
        game = create_game(self.backend, rules, position)
        self.addCleanup(game.close)
        return game

    def test_record_roundtrip_and_pending(self):
        game = self.game(setup([38,67,72,73,74,75,76,77,78], [39,37,3,0,1,2,4]))
        game.play(67*81+40)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "match.json"
            save_record(game, path)
            loaded = load_record(path, self.backend)
            self.addCleanup(loaded.close)
            self.assertEqual(record_dict(loaded), record_dict(game))
            self.assertEqual(loaded.history_keys(), game.history_keys())
            loaded.play(3*81+39)
            self.assertFalse(loaded.outcome.terminal)

    def test_record_roundtrip_with_repetition(self):
        game = self.game()
        cycle = [72*81+63, 0*81+9, 63*81+72, 9*81+0]
        for index in range(12):
            game.play(cycle[index % 4])
        loaded = from_record(record_dict(game), self.backend)
        self.addCleanup(loaded.close)
        self.assertEqual(loaded.outcome, game.outcome)
        self.assertEqual(loaded.history_keys(), game.history_keys())
        loaded.undo()
        self.assertFalse(loaded.outcome.terminal)

    def test_record_rejects_tampering(self):
        data = record_dict(self.game())
        cases = []
        for key, value in (("ruleset_id", "unknown"), ("moves", [1]), ("moves", [True]),
                           ("initial", {"black": 123})):
            case = copy.deepcopy(data)
            case[key] = value
            cases.append(case)
        case = copy.deepcopy(data)
        case["final"]["winner"] = 1
        cases.append(case)
        case = copy.deepcopy(data)
        case["rules"]["lead_target"] = True
        cases.append(case)
        for case in cases:
            with self.assertRaises(ValueError):
                from_record(case, self.backend)

    def test_observation_and_mask(self):
        game = self.game()
        observation = observe(game)
        self.assertEqual(sum(observation["action_mask"]), 63)
        self.assertEqual(len(observation["action_mask"]), 6561)
        self.assertEqual(len(observation["planes"]), 5)
        self.assertEqual(observation["planes"][0][8], [1]*9)
        self.assertEqual(observation["history_counts"][0]["count"], 1)
        game.play(72*81+63)
        observation = observe(game)
        self.assertEqual(observation["to_play"], 1)
        self.assertEqual(observation["planes"][0][0], [1]*9)
        self.assertEqual(len(observation["history"]), 2)
        self.assertEqual(observation["planes"][4][0], [1]*9)

    def test_env_truncation_is_not_a_draw(self):
        env = TurnBasedEnv(self.backend, max_plies=1)
        self.addCleanup(env.close)
        state = record_dict(env.game)
        with self.assertRaises(ValueError):
            env.step(0)
        self.assertEqual(record_dict(env.game), state)
        result = env.step(72*81+63)
        self.assertTrue(result.truncated)
        self.assertFalse(result.terminated)
        self.assertEqual(result.rewards, (0.0, 0.0))
        self.assertEqual(env.game.outcome.winner, -2)
        with self.assertRaises(ValueError):
            env.step(env.game.legal_actions()[0])
        env.reset()
        self.assertFalse(env.truncated)
        self.assertEqual(len(env.game.moves), 0)

    def test_terminal_reward_fixed_player_perspective(self):
        env = TurnBasedEnv(self.backend, max_plies=1)
        self.addCleanup(env.close)
        env.game.close()
        env.game = create_game(self.backend, position=setup([38,67,72,73,74,75], [39,0,1,2,3]))
        result = env.step(67*81+40)
        self.assertEqual(result.rewards, (1.0, -1.0))
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)
        self.assertEqual(result.observation["to_play"], 1)
        self.assertEqual(sum(result.observation["action_mask"]), 0)

    def test_baselines_return_legal_actions_without_mutation(self):
        game = self.game()
        original = record_dict(game)
        for agent in (RandomAgent(1), GreedyAgent(), AlphaBetaAgent()):
            result = agent.choose(game, SearchLimits(2, 10000, 1000000))
            self.assertIn(result.action, game.legal_actions())
            self.assertEqual(record_dict(game), original)
            self.assertEqual(len(game.history_keys()), 1)

    def test_search_finds_immediate_capture_victory(self):
        game = self.game(setup([38,67,72,73,74,75], [39,0,1,2,3]))
        result = AlphaBetaAgent().choose(game, SearchLimits(1, 10000, 1000000))
        game.play(result.action)
        self.assertEqual(game.outcome.winner, 0)

    def test_search_resolves_pending_reply_at_horizon(self):
        game = self.game(setup([38,67,72,73,74,75,76,77,78], [39,37,3,0,1,2,4]))
        game.play(67*81+40)
        result = AlphaBetaAgent().choose(game, SearchLimits(1, 10000, 1000000))
        game.play(result.action)
        self.assertNotEqual(game.outcome.winner, 0)

    def test_search_one_node_returns_legal_fallback(self):
        game = self.game()
        result = AlphaBetaAgent().choose(game, SearchLimits(4, 10000, 1))
        self.assertIn(result.action, game.legal_actions())
        self.assertEqual(result.depth, 0)
        self.assertFalse(result.completed)
        self.assertLessEqual(result.nodes, 1)

    def test_agent_cancellation(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(AgentCancelled):
            AlphaBetaAgent().choose(self.game(), SearchLimits(), cancel)


class PythonIntegrationTests(IntegrationTests, unittest.TestCase):
    backend = "python"


@unittest.skipUnless(native_available(), "Native library is not built")
class NativeIntegrationTests(IntegrationTests, unittest.TestCase):
    backend = "cpp"


class PluginTests(unittest.TestCase):
    def setUp(self):
        self.game = create_game("python")
        self.addCleanup(self.game.close)

    def test_registered_plugin_and_external_example(self):
        registry = AgentRegistry()
        root = Path(__file__).resolve().parents[1]
        registry.load_config(root / "examples/agents.example.json")
        for name in ("plugin_first", "external_random"):
            result = registry.create(name).choose(self.game, SearchLimits(time_ms=2000))
            self.assertIn(result.action, self.game.legal_actions())
        with self.assertRaises(ValueError):
            registry.register("random", RandomAgent)

    def test_illegal_external_action_is_rejected(self):
        code = "import json; print(json.dumps({'protocol':'hasami-agent-v1','action':0}))"
        agent = SubprocessAgent([sys.executable, "-c", code])
        with self.assertRaises(ValueError):
            agent.choose(self.game, SearchLimits(time_ms=2000))

    def test_external_agent_timeout(self):
        agent = SubprocessAgent([sys.executable, "-c", "import time; time.sleep(2)"], startup_ms=0)
        with self.assertRaises(TimeoutError):
            agent.choose(self.game, SearchLimits(time_ms=30))

    def test_external_failure_is_explicit(self):
        agent = SubprocessAgent([sys.executable, "-c", "raise RuntimeError('test')"])
        with self.assertRaises(RuntimeError):
            agent.choose(self.game, SearchLimits(time_ms=2000))


@unittest.skipUnless(native_available(), "Native library is not built")
class ParityTests(unittest.TestCase):
    def test_random_trajectories(self):
        rng = random.Random(1701)
        compared = 0
        for index in range(60):
            python = create_game("python")
            native = create_game("cpp")
            try:
                for ply in range(240):
                    self.assertEqual(python.position, native.position)
                    self.assertEqual(python.outcome, native.outcome)
                    self.assertEqual(python.last_capture, native.last_capture)
                    actions = python.legal_actions()
                    self.assertEqual(actions, native.legal_actions())
                    if python.outcome.terminal:
                        break
                    action = rng.choice(actions)
                    a, b = python.play(action), native.play(action)
                    self.assertEqual(a, b)
                    compared += 1
                    if ply % 17 == 0:
                        self.assertEqual(python.history_keys(), native.history_keys())
                        self.assertTrue(python.undo())
                        self.assertTrue(native.undo())
                        self.assertEqual(python.position, native.position)
                        self.assertEqual(python.play(action), native.play(action))
                self.assertEqual(record_dict(python), record_dict(native))
            finally:
                python.close()
                native.close()
        self.assertGreater(compared, 1000)
        print(f"\nRandom trajectory parity: {compared} transitions across 60 games")

    def test_exact_shallow_search_parity(self):
        rng = random.Random(117)
        for index in range(5):
            squares = rng.sample(range(81), 18)
            position = setup(squares[:9], squares[9:], index % 2)
            python = create_game("python", position=position)
            native = create_game("cpp", position=position)
            try:
                limits = SearchLimits(2, 30000, 10000000)
                a = AlphaBetaAgent().choose(python, limits)
                b = AlphaBetaAgent().choose(native, limits)
                self.assertEqual(a.depth, 2)
                self.assertEqual(b.depth, 2)
                self.assertEqual((a.action, a.score, a.nodes), (b.action, b.score, b.nodes))
            finally:
                python.close()
                native.close()

    def test_cross_backend_record_import(self):
        python = create_game("python")
        self.addCleanup(python.close)
        rng = random.Random(1)
        for _ in range(30):
            if python.outcome.terminal:
                break
            python.play(rng.choice(python.legal_actions()))
        native = from_record(record_dict(python), "cpp")
        self.addCleanup(native.close)
        self.assertEqual(record_dict(python), record_dict(native))
        self.assertEqual(python.history_keys(), native.history_keys())

    def test_closed_native_game_fails_safely(self):
        game = create_game("cpp")
        game.close()
        game.close()
        with self.assertRaises(RuntimeError):
            game.legal_actions()
