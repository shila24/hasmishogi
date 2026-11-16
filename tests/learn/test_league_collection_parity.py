"""batched collection acceptance 5.1: reference and batched league collection produce identical data (exact, on CPU).

Everything is compared field by field: opponent assignments, episode order, every seat's joint actions (forced
ones included), recorded planes/masks/actions/logp/values, step info, terminal states, rewards, pool statistics,
rollout buffers after ingestion and, after two full iterations, the network weights. No comparison goes through
training losses.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.learn._common import requires_torch
from tests.learn.test_batched_selfplay import assert_trajectories_equal

ROOT = Path(__file__).resolve().parents[2]
BUFFERS = ("_planes", "_masks", "_actions", "_logp_old", "_value_old", "_temperatures", "_advantages", "_returns")


def league_config(output_dir, rounds="tonpuu", max_plies=2000, iterations=2, games=4, device="cpu", selfplay=None,
                  opponent_params=None, **params):
    from hasami_learn.config import TrainingConfig

    data = json.loads((ROOT / "configs/tiny/riichi-ppo-league.json").read_text())
    data["run"].update(output_dir=str(output_dir), run_id="parity", iterations=iterations, backend="cpp",
                       checkpoint_every=1, keep_checkpoints=iterations + 1)
    data["device"].update(device=device, amp="off", inference_batch=8, fixed_batch_shape=True)
    data["game"]["params"] = {"rounds": rounds}
    data["selfplay"].update(games_per_iteration=games, max_plies=max_plies, **(selfplay or {}))
    data["opponents"].update(strategy="pfsp", classical=["random", "tsumogiri", "shanten-greedy"],
                             classical_weight=0.5, snapshot_every=1, params=opponent_params or {})
    data["algorithm"]["params"].update(params)
    path = Path(output_dir) / "config.json"
    path.write_text(json.dumps(data))
    return TrainingConfig.load(path)


def without_run_dir(value, config):
    """Pool checkpoint references contain each run's own output directory; compare everything else exactly."""
    return json.loads(json.dumps(value).replace(json.dumps(str(config.run.output_dir))[1:-1], "<out>"))


def run_and_capture(config, perturb_learner_rng=False):
    """Run the trainer; capture every collection's trajectories, statistics, pool state and rollout buffers."""
    import copy
    from hasami_learn.algorithms import ppo_league
    from hasami_learn.trainer import Trainer

    events = []
    originals = (ppo_league.run_table_games, ppo_league.run_table_game)

    def games_batched(*args, **kwargs):
        events.append("game")
        return originals[0](*args, **kwargs)

    def games_reference(*args, **kwargs):
        events.append("game")
        return originals[1](*args, **kwargs)

    ppo_league.run_table_games, ppo_league.run_table_game = games_batched, games_reference
    trainer = Trainer(config, echo=False)
    try:
        learner = trainer.learner
        pool = trainer.opponents
        sample, record = pool.sample, pool.record_result
        pool.sample = lambda rng=None: (events.append("sample"), sample(rng))[1]
        records = []
        pool.record_result = lambda entry, score: (events.append("record"), records.append((entry.entry_id, score)),
                                                   record(entry, score))[2]
        captured, current = [], []
        ingest, collect = learner.ingest_trajectories, learner.collect

        def capture_ingest(trajectories):
            current.extend(trajectories)
            return ingest(trajectories)

        def capture_collect(iteration):
            if perturb_learner_rng:
                learner.rng.random(17)
            events.append("collect")
            current.clear()
            records.clear()
            stats = collect(iteration)
            extra = {k: v for k, v in stats.extra.items() if k not in ("timing", "collection")}
            captured.append({
                "trajectories": list(current),
                "stats": (stats.games, stats.plies, stats.truncated, stats.wins, without_run_dir(extra, config)),
                "pool": without_run_dir(pool.stats(), config),
                "buffers": {name: copy.deepcopy(getattr(learner, name)) for name in BUFFERS},
                "aux": copy.deepcopy(learner._aux_targets),
                "timing": stats.extra["timing"],
                "records": without_run_dir(list(records), config),
            })
            return stats

        learner.ingest_trajectories, learner.collect = capture_ingest, capture_collect
        trainer.run()
        weights = {k: v.detach().clone() for k, v in learner.network.state_dict().items()}
        return captured, weights, events
    finally:
        trainer.close()
        ppo_league.run_table_games, ppo_league.run_table_game = originals


MPS_SKIP = "Apple MPS device is not available"


def mps_available():
    try:
        from hasami_learn.devices import mps_available as available

        return available()
    except ImportError:
        return False


@requires_torch
class LeagueCollectionParityTests(unittest.TestCase):
    def setUp(self):
        from tests.learn.test_batched_selfplay import require_native

        require_native(self)

    def assert_same_collections(self, a, b, label):
        import numpy as np

        self.assertEqual(len(a), len(b), label)
        for iteration, (x, y) in enumerate(zip(a, b), start=1):
            context = f"{label} iteration {iteration}"
            self.assertEqual(x["stats"], y["stats"], f"{context}: assignments, order, wins, sample counts")
            self.assertEqual(x["pool"], y["pool"], f"{context}: pool statistics")
            self.assertEqual(len(x["trajectories"]), len(y["trajectories"]), context)
            for t, (tx, ty) in enumerate(zip(x["trajectories"], y["trajectories"])):
                assert_trajectories_equal(self, tx, ty, f"{context} table {t}")
            for name in BUFFERS:
                bx, by = x["buffers"][name], y["buffers"][name]
                self.assertEqual(len(bx), len(by), f"{context} {name}")
                for i, (u, v) in enumerate(zip(bx, by)):
                    self.assertTrue(np.array_equal(np.asarray(u), np.asarray(v)), f"{context} {name}[{i}]")
            self.assertEqual(len(x["aux"]), len(y["aux"]), context)
            for i, (u, v) in enumerate(zip(x["aux"], y["aux"])):
                self.assertEqual(set(u), set(v), f"{context} aux[{i}]")
                for key in u:
                    self.assertTrue(np.array_equal(u[key], v[key]), f"{context} aux[{i}][{key}]")

    def assert_same_weights(self, a, b, label):
        import torch

        self.assertEqual(set(a), set(b))
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]), f"{label}: weight {key}")

    def run_pair(self, label, reference_params, batched_params, **config):
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            ref = run_and_capture(league_config(left, **config, **reference_params))
            bat = run_and_capture(league_config(right, **config, **batched_params))
        self.assert_same_collections(ref[0], bat[0], label)
        self.assert_same_weights(ref[1], bat[1], label)
        return ref, bat

    def test_tonpuu_complete_matches_reference_equals_batched(self):
        ref, bat = self.run_pair("tonpuu", {"collection": "reference"}, {"collection": "batched"},
                                 rounds="tonpuu", max_plies=4000)
        trajectories = [t for c in bat[0] for t in c["trajectories"]]
        self.assertTrue(all(not t.truncated and t.reason != "ongoing" for t in trajectories))
        kinds = {seat_entry.split(":")[0] for c in bat[0] for table in c["stats"][4]["tables"]
                 for seat_entry in table["opponents"]}
        self.assertIn("classical", kinds)
        self.assertTrue(kinds & {"current", "pool"}, kinds)  # policy opponents share or own evaluators
        self.assertGreater(bat[0][0]["timing"]["forced_skipped"], 0)
        # The reference runner times only inference and the total; its other phases are unmeasured, not zero.
        phases = ("encode", "decide", "env", "agents")
        for timing in (c["timing"] for c in ref[0]):
            self.assertEqual({name: timing[name] for name in phases}, dict.fromkeys(phases))
            self.assertGreater(timing["inference"], 0.0)
            self.assertGreater(timing["total"], 0.0)
        for timing in (c["timing"] for c in bat[0]):
            for name in (*phases, "inference", "total"):
                self.assertGreater(timing[name], 0.0, name)
        # Forced actions of every seat are part of the compared joint actions.
        forced = sum(1 for c in bat[0] for t in c["trajectories"] for step in t.joint_actions
                     for action in step.values() if action == 109)
        self.assertGreater(forced, 0)

    def test_hanchan_truncated_chunked_and_unskipped_equal_reference(self):
        # Learner temperature 0.5 for the first 40 plies, then 1.0; checkpoint opponents sample at 0.7.
        ref, bat = self.run_pair("hanchan chunked", {"collection": "reference"},
                                 {"collection": "batched", "parallel_tables": 3, "skip_forced_decisions": False},
                                 rounds="hanchan", max_plies=300, games=8,
                                 selfplay={"temperature": 0.5, "temperature_plies": 40},
                                 opponent_params={"temperature": 0.7})
        temperatures = set()
        for collection in bat[0]:
            for trajectory in collection["trajectories"]:
                for extra in trajectory.extras:
                    expected = 0.5 if extra["decision_ply"] < 40 else 1.0
                    self.assertEqual(extra["temperature"], expected)
                    temperatures.add(extra["temperature"])
        self.assertEqual(temperatures, {0.5, 1.0})
        pool_seats = [entry for c in bat[0] for table in c["stats"][4]["tables"] for entry in table["opponents"]
                      if entry.startswith("pool:")]
        self.assertTrue(pool_seats)  # stochastic checkpoint opponents took part in the compared games

    def test_hanchan_linear_kernel_tail_matches_reference(self):
        # On x86, row 6 of an eight-row Linear forward can differ from row 0
        # even with identical inputs. Table 2, seat 2 first records a different
        # value at ply 15; keep this short regression independent of later games.
        self.run_pair("hanchan linear tail", {"collection": "reference"},
                      {"collection": "batched", "parallel_tables": 3, "skip_forced_decisions": False},
                      rounds="hanchan", max_plies=20, games=8, iterations=1,
                      selfplay={"temperature": 0.5, "temperature_plies": 40},
                      opponent_params={"temperature": 0.7})

    def test_canonical_cpu_shape_keeps_gpu_and_variable_batching(self):
        import torch
        from unittest.mock import patch
        from hasami_learn.algorithms import ppo_league
        from hasami_learn.trainer import Trainer

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        trainer = Trainer(league_config(tmp.name, iterations=1, max_plies=20), echo=False)
        self.addCleanup(trainer.close)
        learner = trainer.learner
        # No tables: inspect device dispatch without pretending to execute GPU kernels.
        with patch.object(learner, "_plan", return_value=[]):
            for device, fixed, configured, expected in (
                    ("cpu", True, 8, 1), ("cpu", True, None, 1),
                    ("cpu", False, 8, 8), ("cuda", True, 8, 8),
                    ("cuda", True, None, 64), ("mps", True, None, 32)):
                with self.subTest(device=device, fixed=fixed, configured=configured):
                    learner.device = torch.device(device)
                    learner.config.device.fixed_batch_shape = fixed
                    learner.config.device.inference_batch = configured
                    with patch.object(ppo_league, "BatchEvaluator", wraps=ppo_league.BatchEvaluator) as factory:
                        stats = learner.collect(1)
                    self.assertEqual(factory.call_count, 1)
                    self.assertEqual(factory.call_args.args[3:5], (expected, fixed))
                    self.assertEqual(stats.extra["inference_batch"], expected)

    def test_assignments_precede_games_and_results_follow_them(self):
        # 700 plies: iteration 1 has complete and truncated tables, iteration 2 only complete ones.
        with tempfile.TemporaryDirectory() as tmp:
            captured, _, events = run_and_capture(league_config(tmp, iterations=2, max_plies=700))
        collections = " ".join(events).split("collect")[1:]
        self.assertEqual(len(collections), 2)
        flags = [t.truncated for c in captured for t in c["trajectories"]]
        self.assertIn(True, flags)
        self.assertIn(False, flags)
        for chunk, collection in zip(collections, captured):
            words = chunk.split()
            samples = [i for i, w in enumerate(words) if w == "sample"]
            games = [i for i, w in enumerate(words) if w == "game"]
            records = [i for i, w in enumerate(words) if w == "record"]
            self.assertLess(max(samples), min(games))
            self.assertTrue(records)
            self.assertGreater(min(records), max(games))
            # Canonical table order, one result per distinct entry per table, none for truncated tables.
            expected = []
            for table, trajectory in zip(collection["stats"][4]["tables"], collection["trajectories"]):
                if trajectory.truncated:
                    continue
                score = (trajectory.rewards[table["learner_seat"]] + 1.0) / 2.0
                expected.extend((entry, score) for entry in dict.fromkeys(table["opponents"]))
            self.assertEqual([tuple(r) for r in collection["records"]], expected)

    def test_plan_follows_the_spec_streams(self):
        import numpy as np
        from hasami_learn.batched_selfplay import stream_prefix
        from hasami_learn.trainer import Trainer

        with tempfile.TemporaryDirectory() as tmp:
            for duplicate in (True, False):
                (Path(tmp) / str(duplicate)).mkdir()
                config = league_config(Path(tmp) / str(duplicate), iterations=1, max_plies=50, games=8,
                                       duplicate_deals=duplicate)
                trainer = Trainer(config, echo=False)
                try:
                    plans = trainer.learner._plan(3, stream_prefix(config.run.seed, 0, 3))
                    # Spec 3.1, written out: one opponent stream per collection, groups in order, entries first.
                    stream = np.random.default_rng([config.run.seed, 0, 3, 1])
                    rotations, expected = (4 if duplicate else 1), []
                    for group in range(8 // rotations):
                        entries = tuple(trainer.opponents.sample(stream).entry_id for _ in range(3))
                        learner_seats = range(4) if duplicate else [int(stream.integers(4))]
                        for rotation, seat in enumerate(learner_seats):
                            expected.append((group * rotations + rotation, group, rotation,
                                             config.run.seed + 3 * 8 + group, seat, entries))
                    self.assertEqual([(p.table, p.group, p.rotation, p.seed, p.learner_seat,
                                       tuple(e.entry_id for e in p.entries)) for p in plans], expected)
                finally:
                    trainer.close()

    def test_learner_and_current_model_share_one_evaluator(self):
        from hasami_learn.algorithms import ppo_league
        from hasami_learn.batched_selfplay import BatchSeat, PolicySeat
        from hasami_learn.trainer import Trainer

        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(league_config(tmp, iterations=1, max_plies=60), echo=False)
            try:
                trainer.run()  # one iteration: the pool now holds a checkpoint snapshot
                learner, pool = trainer.learner, trainer.opponents
                cycle = [pool.current, pool.entries[-1], pool.classical[0]]
                draws = iter(cycle * 10)
                pool.sample = lambda rng=None: next(draws)
                requests, original = [], ppo_league.run_table_games
                ppo_league.run_table_games = lambda tables, *a, **k: (requests.extend(tables), original(tables, *a, **k))[1]
                try:
                    stats = learner.collect(2)
                finally:
                    ppo_league.run_table_games = original
            finally:
                trainer.close()
        shared = {id(seat.evaluator) for request in requests for seat in request.seats.values() if getattr(seat, "records", False)}
        self.assertEqual(len(shared), 1)  # one learner evaluator for every table of the collection
        classical = []
        for request, table in zip(requests, stats.extra["tables"]):
            learner_seat = request.seats[table["learner_seat"]]
            self.assertIs(learner_seat.evaluator.network, learner.network)
            for seat_id, entry in table["seats"].items():
                seat = request.seats[int(seat_id)]
                if entry == "current":
                    self.assertIsInstance(seat, PolicySeat)
                    self.assertIs(seat.evaluator, learner_seat.evaluator)
                elif entry.startswith("pool:"):
                    self.assertIsInstance(seat, PolicySeat)
                    self.assertIsNot(seat.evaluator, learner_seat.evaluator)
                    self.assertIsNot(seat.evaluator.network, learner.network)
                else:
                    self.assertNotIsInstance(seat, BatchSeat)
                    classical.append(seat)
        self.assertEqual(len({id(seat) for seat in classical}), len(classical))  # fresh per (table, seat)
        self.assertTrue(classical)

    @unittest.skipUnless(mps_available(), MPS_SKIP)
    def test_tonpuu_reference_equals_batched_on_mps(self):
        self.run_pair("tonpuu mps", {"collection": "reference"}, {"collection": "batched"},
                      rounds="tonpuu", max_plies=4000, device="mps")

    def test_collection_does_not_use_the_learner_rng(self):
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            plain = run_and_capture(league_config(left, iterations=1, max_plies=300))
            perturbed = run_and_capture(league_config(right, iterations=1, max_plies=300), perturb_learner_rng=True)
        self.assert_same_collections(plain[0], perturbed[0], "learner rng perturbed")

    def test_unknown_stochastic_opponents_are_rejected(self):
        from hasami.agents import Decision
        from hasami_learn.config import ConfigError
        from hasami_learn.trainer import Trainer

        class Stochastic:
            rng = None

            def choose(self, game, limits, cancel=None, *, player=None):
                return Decision(game.legal_actions(player)[0])

        with tempfile.TemporaryDirectory() as tmp:
            config = league_config(tmp, iterations=1, max_plies=50)
            config.opponents.classical_weight = 0.0
            trainer = Trainer(config, echo=False)
            try:
                trainer.opponents.set_current(lambda: Stochastic())
                with self.assertRaises(ConfigError):
                    trainer.learner.collect(1)
            finally:
                trainer.close()


if __name__ == "__main__":
    unittest.main()
