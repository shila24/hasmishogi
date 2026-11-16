"""Legal decisions, deterministic completed search, budgets and clone ownership."""
import random
import threading
import time
import unittest
from unittest.mock import patch

from hasami.agents import AgentCancelled, SearchLimits
from shogi import create_game
from shogi.agents import RandomAgent, GreedyAgent, AlphaBetaAgent, evaluate, WIN
from shogi.model import K, DECLARE, Rules
from shogi.notation import usi_to_move
from tests.shogi_suite.support import setup, snapshot, play_text
from tests.shogi_suite.test_perft import CASES


class AgentTests(unittest.TestCase):
    def test_legal_actions_and_root_unchanged(self):
        rng = random.Random(26)
        game = create_game()
        agents = (RandomAgent(26), GreedyAgent(), AlphaBetaAgent())
        for _ in range(30):
            if game.outcome.terminal:
                break
            before = snapshot(game)
            for agent in agents:
                decision = agent.choose(game, SearchLimits(max_depth=1, time_ms=1000, node_limit=2000))
                self.assertIs(type(decision.action), int)
                self.assertIn(decision.action, game.legal_actions())
                self.assertEqual(snapshot(game), before)
            game.play(rng.choice(game.legal_actions()))

    def test_seeded_random_and_greedy_capture(self):
        game = create_game()
        left, right = RandomAgent(123), RandomAgent(123)
        limits = SearchLimits()
        self.assertEqual([left.choose(game, limits).action for _ in range(20)],
                         [right.choose(game, limits).action for _ in range(20)])
        game = setup({"5i": K, "9a": K + 16, "5e": 7, "5c": 22, "3e": 17})
        self.assertEqual(GreedyAgent().choose(game, limits).action, usi_to_move(game, "5e5c+"))
        game = create_game()
        self.assertEqual(GreedyAgent().choose(game, limits).action, min(game.legal_actions()))

    def test_terminal_and_cancel_all_agents(self):
        game = create_game()
        event = threading.Event()
        event.set()
        for agent in (RandomAgent(), GreedyAgent(), AlphaBetaAgent()):
            with self.assertRaises(AgentCancelled):
                agent.choose(game, SearchLimits(), event)
        game.resign()
        for agent in (RandomAgent(), GreedyAgent(), AlphaBetaAgent()):
            with self.assertRaises(ValueError):
                agent.choose(game, SearchLimits())

    def test_completed_fixed_depth_and_nodes(self):
        game = setup({"5i": K, "5a": K + 16, "4g": 5, "6c": 21})
        limits = SearchLimits(max_depth=3, time_ms=10000, node_limit=100000)
        before = snapshot(game)
        left, right = AlphaBetaAgent().choose(game, limits), AlphaBetaAgent().choose(game, limits)
        self.assertTrue(left.completed)
        self.assertEqual((left.action, left.score, left.nodes, left.depth),
                         (right.action, right.score, right.nodes, 3))
        self.assertEqual(snapshot(game), before)
        fixed = SearchLimits(max_depth=20, time_ms=10000, node_limit=100)
        left, right = AlphaBetaAgent().choose(game, fixed), AlphaBetaAgent().choose(game, fixed)
        self.assertEqual((left.action, left.score, left.nodes, left.depth),
                         (right.action, right.score, 100, right.depth))
        self.assertFalse(left.completed)

    def test_search_score_matches_exhaustive_minimax(self):
        game = setup({"5i": K, "5a": K + 16, "4g": 5, "6c": 21})
        def minimax(work, depth, ply):
            if work.outcome.terminal:
                winner = work.outcome.winner
                return 0 if winner == -1 else WIN - ply if winner == work.position.turn else -WIN + ply
            if depth == 0:
                return evaluate(work)
            scores = []
            for action in work.legal_actions():
                actor = work.position.turn
                work.play(action)
                score = minimax(work, depth - 1, ply + 1)
                scores.append(score if work.position.turn == actor else -score)
                work.undo()
            return max(scores)
        expected = minimax(game.clone(), 2, 0)
        decision = AlphaBetaAgent().choose(game, SearchLimits(max_depth=2, time_ms=10000))
        self.assertTrue(decision.completed)
        self.assertEqual(decision.score, expected)

    def test_time_node_and_mid_search_cancel(self):
        game = create_game()
        before = snapshot(game)
        decision = AlphaBetaAgent().choose(game, SearchLimits(max_depth=8, time_ms=10000, node_limit=1))
        self.assertEqual(decision.nodes, 1)
        self.assertEqual(decision.depth, 0)
        self.assertFalse(decision.completed)
        self.assertIn(decision.action, game.legal_actions())
        start = time.monotonic()
        decision = AlphaBetaAgent().choose(game, SearchLimits(max_depth=64, time_ms=1))
        self.assertLess(time.monotonic() - start, 0.5)
        self.assertIn(decision.action, game.legal_actions())
        self.assertFalse(decision.completed)
        event = threading.Event()
        original = type(game).play
        def cancelling_play(work, code):
            result = original(work, code)
            event.set()
            return result
        with patch.object(type(game), "play", cancelling_play):
            with self.assertRaises(AgentCancelled):
                AlphaBetaAgent().choose(game, SearchLimits(), event)
        self.assertEqual(snapshot(game), before)

    def test_search_declaration_and_mate(self):
        game = create_game(sfen=CASES["declaration_cases"][0]["sfen"])
        for agent in (GreedyAgent(), AlphaBetaAgent()):
            decision = agent.choose(game, SearchLimits(max_depth=1, time_ms=10000))
            self.assertEqual(decision.action, DECLARE)
            self.assertGreater(decision.score, 900000)
        game = setup({"9a": K + 16, "5i": K, "8c": 5, "7c": 3, "8d": 7})
        decision = AlphaBetaAgent().choose(game, SearchLimits(max_depth=1, time_ms=10000))
        game.play(decision.action)
        self.assertEqual(game.outcome.winner, 0)
        self.assertIn(game.outcome.reason, ("checkmate", "no_legal_moves"))

    def test_agents_complete_a_rule_limited_game(self):
        game = create_game(rules=Rules(40))
        agents = (RandomAgent(3), GreedyAgent())
        while not game.outcome.terminal:
            actor = game.position.turn
            decision = agents[actor].choose(game.clone(), SearchLimits())
            self.assertIn(decision.action, game.legal_actions())
            game.play(decision.action)
            self.assertEqual(game.position.turn, 1 - actor)
        self.assertLessEqual(game.ply, 40)
