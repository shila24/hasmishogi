"""Evaluation statistics, engine end reasons, and reproducible ladder tests."""

from __future__ import annotations

import json
import math
from unittest.mock import patch

from hasami import create_game
from hasami.agents import Decision, SearchLimits
from tests.learn._common import LearnTestCase, requires_torch


class CyclingAgent:
    """Repeat a legal four-ply cycle using the full game history."""

    def choose(self, game, limits, cancel=None):
        cycle = (72 * 81 + 63, 0 * 81 + 9, 63 * 81 + 72, 9 * 81)
        action = cycle[len(game.moves) % 4]
        assert type(action) is int and action in game.legal_actions()
        return Decision(action)


@requires_torch
class EvaluationTests(LearnTestCase):
    def test_wilson_known_values_and_boundaries(self):
        from hasami_learn.evaluation import wilson_interval

        low, high = wilson_interval(50, 100)
        self.assertAlmostEqual(low, 0.4038298, places=6)
        self.assertAlmostEqual(high, 0.5961702, places=6)
        self.assertEqual(wilson_interval(0, 0), (0.0, 1.0))
        self.assertAlmostEqual(wilson_interval(0, 100)[0], 0.0)
        self.assertAlmostEqual(wilson_interval(0, 100)[1], 0.0369948, places=6)
        self.assertAlmostEqual(wilson_interval(100, 100)[1], 1.0)
        self.assertAlmostEqual(sum(wilson_interval(0.5, 1)), 1.0)
        self.assertEqual(wilson_interval(25, 100, z=0), (0.25, 0.25))

    def test_wilson_rejects_invalid_inputs(self):
        from hasami_learn.evaluation import wilson_interval

        for successes, games, z in ((-1, 10, 1.96), (11, 10, 1.96),
                                    (0, -1, 1.96), (0, 1.5, 1.96),
                                    (float("nan"), 10, 1.96), (1, 10, -1),
                                    (1, 10, float("inf")), (1, 0, 1.96)):
            with self.subTest(successes=successes, games=games, z=z):
                with self.assertRaises(ValueError):
                    wilson_interval(successes, games, z)

    def test_match_result_serializes_counter_and_preserves_fields(self):
        from hasami_learn.evaluation import MatchResult, wilson_interval

        result = MatchResult(10, 4, 2, 4, 1, 120, 0.25)
        result.reasons.update({"five_captures": 6, "repetition": 3, "max_plies": 1})
        data = result.to_dict()
        self.assertEqual(set(data), {"games", "wins", "losses", "draws", "truncated",
                                     "plies", "seconds", "win_rate", "elo_delta",
                                     "ci_low", "ci_high", "draw_rate", "reasons"})
        self.assertEqual(data["win_rate"], 0.6)
        self.assertEqual(result.draw_rate, 0.4)
        self.assertEqual((data["ci_low"], data["ci_high"]), wilson_interval(6, 10))
        self.assertEqual(json.loads(json.dumps(data))["reasons"], dict(result.reasons))
        data["reasons"]["repetition"] = 99
        self.assertEqual(result.reasons["repetition"], 3)
        self.assertEqual(MatchResult().reasons, {})
        self.assertEqual(MatchResult().draw_rate, 0.0)
        self.assertEqual(MatchResult().to_dict()["ci_high"], 1.0)

    def test_repetition_and_cap_are_distinct_end_reasons(self):
        from hasami_learn.evaluation import play_matches

        limits = SearchLimits(1, 1000, 1000)
        for cap, reason, plies, truncated in ((12, "repetition", 12, 0),
                                              (3, "max_plies", 3, 2)):
            with self.subTest(reason=reason):
                result = play_matches(CyclingAgent(), CyclingAgent(), 2,
                                      max_plies=cap, limits=limits,
                                      records_dir=self.tmp / reason)
                self.assertEqual(result.reasons, {reason: 2})
                self.assertEqual(result.truncated, truncated)
                self.assertEqual(result.plies, 2 * plies)
                self.assertEqual(result.draws, 2)
                self.assertEqual(result.win_rate, 0.5)
                record = json.loads((self.tmp / reason / "eval_0000.json").read_text())
                self.assertEqual(record["final"]["reason"],
                                 "ongoing" if truncated else "repetition")
                from hasami.records import from_record

                replayed = from_record(record, "python")
                try:
                    self.assertEqual(replayed.outcome.terminal, not bool(truncated))
                    self.assertEqual(len(replayed.history_keys()), plies + 1)
                finally:
                    replayed.close()

    def test_engine_anchors(self):
        game = create_game("python")
        try:
            actions = game.legal_actions()
            self.assertEqual(len(actions), 63)
            leaves = 0
            for action in actions:
                game.play(action)
                leaves += len(game.legal_actions())
                game.undo()
            self.assertEqual(leaves, 3717)
        finally:
            game.close()


@requires_torch
class LadderTests(LearnTestCase):
    def test_bradley_terry_order_anchor_and_separation(self):
        from hasami_learn.evaluation import MatchResult
        from hasami_learn.ladder import fit_elo

        pairs = [("random", "greedy", MatchResult(games=10, losses=10)),
                 ("random", "strong", MatchResult(games=10, losses=10)),
                 ("greedy", "strong", MatchResult(games=10, wins=2, losses=8))]
        ratings = fit_elo(["random", "greedy", "strong"], pairs, anchor="random")
        self.assertEqual(ratings["random"], 0.0)
        self.assertGreater(ratings["strong"], ratings["greedy"])
        self.assertGreater(ratings["greedy"], ratings["random"])
        self.assertTrue(all(math.isfinite(rating) for rating in ratings.values()))
        tied = fit_elo(["a", "b"], [("a", "b", MatchResult(games=4, draws=4))])
        self.assertEqual(tied, {"a": 0.0, "b": 0.0})
        known = fit_elo(["a", "b"], [("a", "b", MatchResult(games=10, wins=8, losses=2))])
        self.assertEqual(known["a"], 0.0)
        self.assertAlmostEqual(known["b"], -400 * math.log10(8.5 / 2.5), places=6)

    def test_three_baselines_reproduce_and_resume_finished_pairs(self):
        from hasami_learn.ladder import run_ladder

        kwargs = dict(entries=["random", "greedy", "alphabeta:1"], games=4,
                      backend="python", max_plies=60, seed=0)
        first = run_ladder(out=self.tmp / "first", **kwargs)
        second = run_ladder(out=self.tmp / "second", **kwargs)
        self.assertEqual(len(first["pairs"]), 3)
        self.assertEqual(first["standings"], second["standings"])
        for a, b in zip(first["pairs"], second["pairs"]):
            self.assertEqual(a["result"]["games"], 4)
            self.assertEqual(sum(a["result"]["reasons"].values()), 4)
            self.assertEqual({k: v for k, v in a["result"].items() if k != "seconds"},
                             {k: v for k, v in b["result"].items() if k != "seconds"})
        ratings = {row["name"]: row["elo"] for row in first["standings"]}
        self.assertEqual(ratings["random"], 0.0)
        self.assertGreater(ratings["alphabeta:1"], ratings["random"])
        self.assertGreaterEqual(ratings["alphabeta:1"], ratings["greedy"])
        self.assertTrue((self.tmp / "first" / "ladder.md").is_file())
        with patch("hasami_learn.ladder.play_matches", side_effect=AssertionError("replayed")):
            resumed = run_ladder(out=self.tmp / "first", resume=True, **kwargs)
        self.assertEqual(resumed, first)
        path = self.tmp / "first" / "ladder.json"
        partial = json.loads(path.read_text())
        partial["pairs"].pop()
        path.write_text(json.dumps(partial))
        from hasami_learn.evaluation import play_matches

        with patch("hasami_learn.ladder.play_matches", wraps=play_matches) as play:
            resumed = run_ladder(out=self.tmp / "first", resume=True, **kwargs)
        self.assertEqual(play.call_count, 1)
        self.assertEqual(resumed["standings"], first["standings"])

    def test_resume_rejects_changed_protocol_and_corrupt_counts(self):
        from hasami_learn.ladder import run_ladder

        kwargs = dict(entries=["r=random", "g=greedy"], games=2,
                      backend="python", max_plies=1, out=self.tmp)
        report = run_ladder(**kwargs)
        self.assertEqual(report["anchor"], "r")
        with self.assertRaisesRegex(ValueError, "configuration"):
            run_ladder(**{**kwargs, "seed": 123}, resume=True)
        report["pairs"][0]["result"]["wins"] = 100
        (self.tmp / "ladder.json").write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "result"):
            run_ladder(**kwargs, resume=True)

    def test_invalid_entries_and_odd_games(self):
        from hasami_learn.ladder import run_ladder

        for entries, games in ((["random"], 2), (["random", "random"], 2),
                                (["random", "greedy"], 3), (["=random", "greedy"], 2),
                                (["random", "alphabeta:"], 2)):
            with self.subTest(entries=entries, games=games):
                with self.assertRaises(ValueError):
                    run_ladder(entries, games=games, out=self.tmp)

    def test_checkpoint_entry_cpu_and_resume_fingerprint(self):
        from hasami_learn.ladder import run_ladder
        from hasami_learn.manifest import ModelManifest, save_checkpoint
        from hasami_learn.registry import NETWORKS
        import hasami_learn.networks  # noqa: F401

        network = NETWORKS.get("mlp")(head="policy_value", num_planes=9, hidden=8, layers=1)
        manifest = ModelManifest(algorithm="reinforce", encoding_id="hasami-enc-v1", num_planes=9,
                                 network={"name": "mlp", "head": "policy_value",
                                          "params": {"hidden": 8, "layers": 1}})
        checkpoint = save_checkpoint(self.tmp / "checkpoint", network, manifest)
        kwargs = dict(entries=[f"model={checkpoint}", "baseline=random"], games=2,
                      backend="python", max_plies=2, simulations=4, device="cpu", out=self.tmp / "ladder")
        report = run_ladder(**kwargs)
        self.assertEqual(report["anchor"], "baseline")
        self.assertEqual(report["pairs"][0]["result"]["reasons"], {"max_plies": 2})
        with patch("hasami_learn.ladder.play_matches", side_effect=AssertionError("replayed")):
            self.assertEqual(run_ladder(**kwargs, resume=True), report)
        path = checkpoint / "manifest.json"
        data = json.loads(path.read_text())
        data["notes"] = "Changed checkpoint identity"
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "configuration"):
            run_ladder(**kwargs, resume=True)

    def test_limits_reach_both_agents_and_classical_budget_is_independent(self):
        from hasami_learn.evaluation import MatchResult
        from hasami_learn.ladder import run_ladder

        result = MatchResult(games=2, draws=2, truncated=2, plies=2)
        result.reasons["max_plies"] = 2
        with patch("hasami_learn.ladder.play_matches", return_value=result) as play:
            report = run_ladder(["depth1=alphabeta:1", "depth3=alphabeta:3"], games=2,
                                simulations=7, time_ms=321, max_plies=1, out=self.tmp)
        self.assertEqual(report["anchor"], "depth1")
        self.assertEqual(play.call_args.kwargs["limits"], SearchLimits(1, 321, 200000))
        self.assertEqual(play.call_args.kwargs["opponent_limits"], SearchLimits(3, 321, 200000))
        self.assertIs(play.call_args.kwargs["swap_sides"], True)

    def test_interrupted_round_robin_keeps_finished_pairs(self):
        from hasami_learn.evaluation import play_matches
        from hasami_learn.ladder import run_ladder

        calls = 0

        def interrupt_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("interrupted")
            return play_matches(*args, **kwargs)

        kwargs = dict(entries=["random", "greedy", "alphabeta:1"], games=2,
                      backend="python", max_plies=1, out=self.tmp)
        with patch("hasami_learn.ladder.play_matches", side_effect=interrupt_second):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                run_ladder(**kwargs)
        partial = json.loads((self.tmp / "ladder.json").read_text())
        self.assertEqual(len(partial["pairs"]), 1)
        self.assertFalse(partial["complete"])
        with patch("hasami_learn.ladder.play_matches", wraps=play_matches) as play:
            report = run_ladder(**kwargs, resume=True)
        self.assertEqual(play.call_count, 2)
        self.assertTrue(report["complete"])
        self.assertEqual(report["pairs"][0], partial["pairs"][0])
