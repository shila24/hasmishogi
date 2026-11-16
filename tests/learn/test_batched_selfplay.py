"""batched collection: batched multi-table runner, fixed-shape evaluator and seats (docs/METHOD_ja.md#methods)."""

from __future__ import annotations

import unittest

from tests.learn._common import requires_torch


def require_native(test):
    """Repository convention (test_riichi_native_backend): fail when native is required, otherwise skip."""
    import os
    from riichi.native import available

    if not available():
        if os.environ.get("HASAMI_REQUIRE_NATIVE") == "1":
            test.fail("native riichi game is required")
        test.skipTest("native riichi game is unavailable")


def assert_trajectories_equal(test, a, b, context=""):
    """Exact, field-by-field equality of two trajectories (planes bitwise, floats exact)."""
    import numpy as np

    test.assertEqual(len(a.planes), len(b.planes), context)
    test.assertEqual(len(a.masks), len(b.masks), context)
    for i, (x, y) in enumerate(zip(a.planes, b.planes)):
        test.assertEqual((x.dtype, x.shape), (y.dtype, y.shape), f"{context} planes[{i}]")
        test.assertTrue(np.array_equal(x, y), f"{context} planes[{i}]")
    for i, (x, y) in enumerate(zip(a.masks, b.masks)):
        test.assertTrue(np.array_equal(x, y) and x.dtype == y.dtype, f"{context} masks[{i}]")
    for name in ("actions", "players", "extras", "rewards", "num_players", "seed", "joint_actions", "step_info",
                 "truncated", "winner", "reason", "plies", "draw_value", "draw_reasons", "game_id"):
        test.assertEqual(getattr(a, name), getattr(b, name), f"{context} {name}")
    test.assertEqual(len(a.policies), len(b.policies), context)
    for x, y in zip(a.policies, b.policies):
        test.assertTrue((x is None and y is None) or np.array_equal(x, y), context)


@requires_torch
class BatchedSelfplayTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch
        from hasami_learn.games import get_spec
        from hasami_learn.registry import NETWORKS
        import hasami_learn.networks  # noqa: F401

        require_native(self)
        torch.manual_seed(0)
        self.spec = get_spec("riichi", {"rounds": "tonpuu"})
        self.encoder = self.spec.make_encoder("riichi-history", k=2)
        self.network = NETWORKS.get("mlp")(head="policy_value", num_planes=self.encoder.num_planes,
                                           board_shape=(4, 34), action_size=111, hidden=16, layers=1)
        self.np = np

    def evaluator(self, batch_size=4, fixed_shape=True):
        from hasami_learn.batched_selfplay import BatchEvaluator

        return BatchEvaluator(self.network, "cpu", "off", batch_size=batch_size, fixed_shape=fixed_shape)

    def seats(self, evaluator, prefix, temperature=0.0, skip_forced=True, recording_seat=0):
        from hasami_learn.batched_selfplay import PolicySeat, decision_rng

        rng_for = lambda table, seat, ply: decision_rng(prefix, table, seat, ply)
        seats = {p: PolicySeat(evaluator, self.encoder, temperature, rng_for, skip_forced) for p in range(4)}
        if recording_seat is not None:
            seats[recording_seat] = RecordingSeat(evaluator, self.encoder, rng_for)
        return seats

    def test_fixed_shape_rows_do_not_depend_on_companions_or_chunking(self):
        np = self.np
        rng = np.random.default_rng(3)
        planes = (rng.random((13, self.encoder.num_planes, 4, 34)) < 0.1).astype(np.float32)
        evaluator = self.evaluator(batch_size=4)
        logits, values = evaluator.forward(planes)
        for i in range(13):
            single_logits, single_values = evaluator.forward(planes[i:i + 1])
            self.assertTrue(np.array_equal(single_logits[0], logits[i]))
            self.assertEqual(single_values[0], values[i])
        order = rng.permutation(13)
        shuffled, _ = evaluator.forward(planes[order])
        self.assertTrue(np.array_equal(shuffled, logits[order]))
        self.assertEqual(evaluator.timing.padded_rows, 3 + 13 * 3 + 3)
        variable = self.evaluator(batch_size=4, fixed_shape=False)
        variable.forward(planes[:3])
        self.assertEqual(variable.timing.padded_rows, 0)

    def test_evaluator_restores_mode_and_validates(self):
        from hasami_learn.batched_selfplay import BatchEvaluator

        self.network.train()
        self.evaluator().forward(self.np.zeros((2, self.encoder.num_planes, 4, 34), self.np.float32))
        self.assertTrue(self.network.training)
        for bad in (0, -1, 2.0):
            with self.assertRaises(ValueError):
                BatchEvaluator(self.network, "cpu", batch_size=bad)
        with self.assertRaises(ValueError):
            self.evaluator().forward(self.np.zeros((0, self.encoder.num_planes, 4, 34), self.np.float32))

    def test_agents_only_tables_equal_run_table_game(self):
        from riichi.baselines import BaselineAgent
        from hasami_learn.batched_selfplay import TableRequest, run_table_games
        from hasami_learn.selfplay import run_table_game

        def seats(table):
            return {p: BaselineAgent("shanten-greedy" if p % 2 else "tsumogiri", seed=100 * table + p) for p in range(4)}

        # The spec's own cap: matches end naturally, in different rounds; one table is capped early.
        for cap in (None, 150):
            requests = [TableRequest(seats(t), 40 + t, t) for t in range(3)]  # fresh agents: their RNGs advance
            batched = run_table_games(requests, self.encoder, spec=self.spec, backend="cpp", max_plies=cap)
            for t, trajectory in enumerate(batched):
                reference = run_table_game(seats(t), self.encoder, spec=self.spec, backend="cpp", max_plies=cap,
                                           seed=40 + t)
                assert_trajectories_equal(self, trajectory, reference, f"cap {cap} table {t}")
            if cap is None:
                self.assertTrue(all(not t.truncated and t.reason != "ongoing" for t in batched))
                self.assertEqual(len({t.plies for t in batched}), 3)
            else:
                self.assertTrue(all(t.truncated and t.plies == cap for t in batched))

    def test_batched_equals_reference_adapters_and_is_schedule_independent(self):
        from hasami_learn.batched_selfplay import SeatActor, SeatAgent, TableRequest, run_table_games
        from hasami_learn.selfplay import run_table_game

        prefix = (5, 0, 2)
        for temperature, cap, count in ((0.0, 300, 4), (1.0, 300, 4), (1.0, None, 2)):
            evaluator = self.evaluator(batch_size=4)
            requests = [TableRequest(self.seats(evaluator, prefix, temperature), 70 + t, t) for t in range(count)]
            batched = run_table_games(requests, self.encoder, spec=self.spec, backend="cpp", max_plies=cap)
            # Reference: the same seats one table after another, one real row per forward, no skipping.
            for t, trajectory in enumerate(batched):
                seats = self.seats(self.evaluator(batch_size=4), prefix, temperature)
                adapted = {p: (SeatActor(seat, t) if seat.records else SeatAgent(seat, t, self.spec))
                           for p, seat in seats.items()}
                reference = run_table_game(adapted, self.encoder, spec=self.spec, backend="cpp", max_plies=cap,
                                           seed=70 + t)
                assert_trajectories_equal(self, trajectory, reference, f"T={temperature} cap={cap} table {t}")
                self.assertGreater(len(trajectory), 0)
            if cap is None:  # natural ends, tables finishing in different rounds
                self.assertTrue(all(not t.truncated and t.reason != "ongoing" for t in batched))
                self.assertNotEqual(batched[0].plies, batched[1].plies)
                continue
            # Scheduling: a different subset and order of tables in flight changes nothing.
            evaluator = self.evaluator(batch_size=4)
            again = run_table_games([TableRequest(self.seats(evaluator, prefix, temperature), 70 + t, t)
                                     for t in (3, 1)], self.encoder, spec=self.spec, backend="cpp", max_plies=300)
            assert_trajectories_equal(self, again[0], batched[3], "reordered table 3")
            assert_trajectories_equal(self, again[1], batched[1], "reordered table 1")

    def test_forced_skipping_changes_nothing_and_is_counted(self):
        from hasami_learn.batched_selfplay import CollectionTiming, TableRequest, run_table_games

        prefix = (9, 0, 1)
        results = {}
        for skip in (True, False):
            timing = CollectionTiming()
            evaluator = self.evaluator(batch_size=8)
            evaluator.timing = timing
            requests = [TableRequest(self.seats(evaluator, prefix, 1.0, skip_forced=skip), 90 + t, t)
                        for t in range(2)]
            results[skip] = (run_table_games(requests, self.encoder, spec=self.spec, backend="cpp", max_plies=300,
                                             timing=timing), timing)
        for a, b in zip(results[True][0], results[False][0]):
            assert_trajectories_equal(self, a, b, "skip on/off")
        self.assertGreater(results[True][1].forced_skipped, 0)
        self.assertEqual(results[False][1].forced_skipped, 0)
        self.assertEqual(results[True][1].evaluated + results[True][1].forced_skipped, results[True][1].decisions)

    def test_policy_seat_matches_policy_agent_rule(self):
        np = self.np
        from hasami_learn.batched_selfplay import DecisionRequest, PolicySeat, decision_rng
        from hasami_learn.inference import PolicyAgent, policy_action

        agent = PolicyAgent(self.network, self.encoder, "cpu", temperature=0.7, spec=self.spec)
        evaluator = self.evaluator()
        seat = PolicySeat.from_agent(agent, evaluator, lambda t, s, p: decision_rng((1, 0, 0), t, s, p))
        rng = np.random.default_rng(2)
        for ply in range(50):
            logits = rng.normal(size=111).astype(np.float32)
            mask = (rng.random(111) < 0.3).astype(np.float32)
            mask[rng.integers(111)] = 1
            request = DecisionRequest(1, 2, ply, np.zeros(1, np.float32), mask)
            expected = policy_action(logits, mask, decision_rng((1, 0, 0), 1, 2, ply), 0.7)
            self.assertEqual(seat.decide(request, logits, 0.0)[0], expected)
        with self.assertRaises(TypeError):
            PolicySeat.from_agent(object(), evaluator, None)
        other = self.evaluator()
        other.network = object()
        with self.assertRaises(ValueError):
            PolicySeat.from_agent(agent, other, None)

    def test_illegal_actions_and_cancellation_close_every_game(self):
        import threading
        from hasami.agents import AgentCancelled, Decision
        from riichi.baselines import BaselineAgent
        from hasami_learn.batched_selfplay import TableRequest, run_table_games

        closed = []
        spec = self.spec
        original = spec.create_game

        def tracking(*args, **kwargs):
            game = original(*args, **kwargs)
            close = game.close
            game.close = lambda: (closed.append(1), close())
            return game

        class Illegal:
            def choose(self, game, limits, cancel=None, *, player=None):
                return Decision(110 if 110 not in game.legal_actions(player) else 0)

        spec.create_game = tracking
        try:
            seats = {p: BaselineAgent("tsumogiri", seed=p) for p in range(4)}
            with self.assertRaises(ValueError):
                run_table_games([TableRequest(dict(seats), 1, 0), TableRequest({**seats, 0: Illegal()}, 2, 1)],
                                self.encoder, spec=spec, backend="cpp")
            self.assertEqual(len(closed), 2)
            cancel = threading.Event()
            cancel.set()
            with self.assertRaises(AgentCancelled):
                run_table_games([TableRequest(dict(seats), 3, 0)], self.encoder, spec=spec, backend="cpp",
                                cancel=cancel)
            self.assertEqual(len(closed), 3)
        finally:
            spec.create_game = original

    def test_streams_follow_the_spec_derivations(self):
        np = self.np
        from hasami_learn.batched_selfplay import agent_seed, decision_rng, opponent_rng, stream_prefix

        # docs/METHOD_ja.md#methods section 3.1, written out independently of the helpers.
        prefix = stream_prefix(12, 3, 7)
        self.assertEqual(prefix, (12, 3, 7))
        self.assertEqual(opponent_rng(prefix).random(4).tolist(), np.random.default_rng([12, 3, 7, 1]).random(4).tolist())
        self.assertEqual(decision_rng(prefix, 5, 2, 41).random(4).tolist(),
                         np.random.default_rng([12, 3, 7, 2, 5, 2, 41]).random(4).tolist())
        expected = int(np.random.SeedSequence([12, 3, 7, 3, 5, 2]).generate_state(1, np.uint64)[0]) % 2 ** 63
        self.assertEqual(agent_seed(prefix, 5, 2), expected)
        streams = {tuple(decision_rng(prefix, t, s_, p).random(2)) for t in range(3) for s_ in range(4) for p in range(3)}
        self.assertEqual(len(streams), 36)  # every decision has its own stream
        with self.assertRaises(ValueError):
            stream_prefix(1, -1, 0)

    def test_seats_use_their_own_evaluator_and_encoder(self):
        np = self.np
        import torch
        from hasami_learn.batched_selfplay import BatchEvaluator, SeatAgent, TableRequest, decision_rng, run_table_games
        from hasami_learn.registry import NETWORKS
        from hasami_learn.selfplay import run_table_game

        torch.manual_seed(1)
        other = NETWORKS.get("mlp")(head="policy_value", num_planes=self.encoder.num_planes, board_shape=(4, 34),
                                    action_size=111, hidden=16, layers=1)
        flipped = FlippedEncoder(self.encoder)
        rng_for = lambda t, s_, p: decision_rng((11, 0, 1), t, s_, p)

        def seats(log):
            first, second = self.evaluator(batch_size=4), BatchEvaluator(other, "cpu", "off", batch_size=4)
            return {0: SpySeat(first, self.encoder, 1.0, rng_for, log), 1: SpySeat(first, self.encoder, 0.0, rng_for, log),
                    2: SpySeat(second, flipped, 1.0, rng_for, log), 3: SpySeat(second, flipped, 0.0, rng_for, log)}

        batched_log, reference_log = {}, {}
        # Only batch seats: the runner's own encoder must never be used.
        batched = run_table_games([TableRequest(seats(batched_log), 30 + t, t) for t in range(3)], ForbiddenEncoder(),
                                  spec=self.spec, backend="cpp", max_plies=120)
        for t, trajectory in enumerate(batched):
            adapted = {p: SeatAgent(seat, t, self.spec) for p, seat in seats(reference_log).items()}
            reference = run_table_game(adapted, ForbiddenEncoder(), spec=self.spec, backend="cpp", max_plies=120,
                                       seed=30 + t)
            assert_trajectories_equal(self, trajectory, reference, f"table {t}")
        self.assertEqual(set(batched_log), set(reference_log))
        crossed = 0
        for key, (seat, planes, logits) in batched_log.items():
            self.assertTrue(np.array_equal(planes, reference_log[key][1]), f"planes {key}")
            if logits is None:
                continue
            # Each row went through the seat's own network (fixed shape: a lone row gives the same bits).
            self.assertTrue(np.array_equal(logits, seat.evaluator.forward(planes[None])[0][0]), f"network {key}")
            wrong = seats({})[0 if key[1] >= 2 else 2].evaluator
            crossed += not np.array_equal(logits, wrong.forward(planes[None])[0][0])
        self.assertGreater(crossed, 0)  # the two networks disagree, so a misrouted row would be detected
        players = {key[1] for key in batched_log}
        self.assertEqual(players, {0, 1, 2, 3})

    def test_cancellation_between_rounds_stops_batch_only_tables(self):
        import threading
        from hasami.agents import AgentCancelled
        from hasami_learn.batched_selfplay import PolicySeat, TableRequest, decision_rng, run_table_games

        cancel = threading.Event()
        decided = []

        class Cancelling(PolicySeat):
            def decide(self, request, logits, value):
                decided.append(request.ply)
                if request.ply >= 5:
                    cancel.set()
                return super().decide(request, logits, value)

        evaluator = self.evaluator()
        rng_for = lambda t, s_, p: decision_rng((2, 0, 0), t, s_, p)
        closed = self.track_closes()
        requests = [TableRequest({p: Cancelling(evaluator, self.encoder, 0.0, rng_for) for p in range(4)}, 5 + t, t)
                    for t in range(2)]
        with self.assertRaises(AgentCancelled):
            run_table_games(requests, ForbiddenEncoder(), spec=self.spec, backend="cpp", cancel=cancel)
        self.assertEqual(len(closed), 2)
        self.assertEqual(max(decided), 5)  # the round in which the event was set finishes, the next one does not start

    def test_illegal_batch_and_actor_actions_are_rejected(self):
        np = self.np
        from riichi.baselines import BaselineAgent
        from hasami_learn.batched_selfplay import PolicySeat, TableRequest, decision_rng, run_table_games

        rng_for = lambda t, s_, p: decision_rng((3, 0, 0), t, s_, p)

        class Illegal(PolicySeat):
            def decide(self, request, logits, value):
                return int(np.flatnonzero(request.mask == 0)[0]), None, {}

        class NotInt(PolicySeat):
            def decide(self, request, logits, value):
                action, policy, extra = super().decide(request, logits, value)
                return np.int64(action), policy, extra

        def illegal_actor(game, planes, mask, ply, player):
            return int(np.flatnonzero(mask == 0)[0]), None, {}

        closed = self.track_closes()
        base = {p: BaselineAgent("tsumogiri", seed=p) for p in range(1, 4)}
        for seat in (Illegal(self.evaluator(), self.encoder, 0.0, rng_for),
                     NotInt(self.evaluator(), self.encoder, 0.0, rng_for), illegal_actor):
            with self.assertRaisesRegex(ValueError, "^Actor returned an illegal action$"):  # the runner, not the engine
                run_table_games([TableRequest({0: seat, **base}, 4, 0)], self.encoder, spec=self.spec, backend="cpp")
        self.assertEqual(len(closed), 3)

    def test_every_game_is_closed_even_if_a_close_fails(self):
        from riichi.baselines import BaselineAgent
        from hasami_learn.batched_selfplay import TableRequest, run_table_games

        closed = self.track_closes(fail_first=True)
        seats = lambda: {p: BaselineAgent("tsumogiri", seed=p) for p in range(4)}
        # A completed run surfaces the close error after closing every game.
        with self.assertRaisesRegex(RuntimeError, "close failed"):
            run_table_games([TableRequest(seats(), 1 + t, t) for t in range(3)], self.encoder, spec=self.spec,
                            backend="cpp", max_plies=20)
        self.assertEqual(len(closed), 3)
        # A failing run keeps its own exception; the close error does not hide it.
        closed.clear()
        with self.assertRaisesRegex(ValueError, "seed must be"):
            run_table_games([TableRequest(seats(), 1, 0), TableRequest(seats(), 2, 1), TableRequest(seats(), -1, 2)],
                            self.encoder, spec=self.spec, backend="cpp", max_plies=20)
        self.assertEqual(len(closed), 2)

    def track_closes(self, fail_first=False):
        """Wrap the spec's create_game so every close is recorded (the first one optionally raises afterwards)."""
        closed = []
        original = self.spec.create_game

        def tracking(*args, **kwargs):
            game = original(*args, **kwargs)
            close = game.close

            def recorded():
                close()
                closed.append(1)
                if fail_first and len(closed) == 1:
                    raise RuntimeError("close failed")

            game.close = recorded
            return game

        self.spec.create_game = tracking
        self.addCleanup(setattr, self.spec, "create_game", original)
        return closed


class RecordingSeat:
    """A minimal recording batch seat for runner tests (sampling at temperature 1 from the decision stream)."""

    records = True

    def __init__(self, evaluator, encoder, rng_for):
        self.evaluator, self.encoder, self.rng_for = evaluator, encoder, rng_for

    def needs_evaluation(self, request):
        return True

    def decide(self, request, logits, value):
        from hasami_learn.inference import policy_action

        action = policy_action(logits, request.mask, self.rng_for(request.table, request.player, request.ply), 1.0)
        return int(action), None, {"value": float(value), "logit": float(logits[action])}


class SpySeat:
    """A ``PolicySeat`` that logs (seat, planes, logits) per (table, player, ply) before deciding."""

    def __new__(cls, evaluator, encoder, temperature, rng_for, log):
        from hasami_learn.batched_selfplay import PolicySeat

        class _Spy(PolicySeat):
            def decide(self, request, logits, value):
                log[(request.table, request.player, request.ply)] = (
                    self, request.planes.copy(), None if logits is None else logits.copy())
                return super().decide(request, logits, value)

        return _Spy(evaluator, encoder, temperature, rng_for)


class FlippedEncoder:
    """Same shape as the wrapped encoder, different content (plane order reversed)."""

    def __init__(self, base):
        self.base, self.num_planes = base, base.num_planes
        self.encoding_id = base.encoding_id + "-flipped"

    def encode(self, game, player=None):
        return self.base.encode(game, player)[::-1].copy()


class ForbiddenEncoder:
    """The runner encoder in tests whose seats all bring their own; any use is a defect."""

    num_planes = 0
    encoding_id = "forbidden"

    def encode(self, game, player=None):
        raise AssertionError("the runner encoder was used for a seat that has its own")


if __name__ == "__main__":
    unittest.main()
