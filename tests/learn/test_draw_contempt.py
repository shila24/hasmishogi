"""Search/target agreement for symmetric draw contempt and collection schedules."""

import json
import os
import unittest
from contextlib import closing
from unittest.mock import patch

from tests.learn._common import LearnTestCase, requires_torch


@requires_torch
class DrawContemptTests(LearnTestCase):
    def test_rule_draw_backup_matches_both_players_targets(self):
        import numpy as np
        from hasami_learn.games import get_spec
        from hasami_learn.search.mcts import MCTS, MctsConfig
        from hasami_learn.selfplay import Trajectory

        spec = get_spec("shogi", {"max_moves": 1})
        game = spec.create_game("python")
        self.addCleanup(game.close)
        trajectory = Trajectory(winner=-1, reason="max_moves", players=[0, 1],
                                draw_value=-0.2, draw_reasons=("max_moves",))
        for mode in ("puct", "gumbel"):
            for reasons, expected in ((("max_moves",), -0.2), (("repetition",), 0.0), ((), 0.0)):
                with self.subTest(mode=mode, reasons=reasons):
                    config = MctsConfig(simulations=8, add_noise=False, root_selection=mode,
                                        draw_value=-0.2, draw_reasons=reasons)
                    def evaluate(planes, masks):
                        return masks, np.full(len(planes), -0.2)
                    result = MCTS(evaluate, spec.make_encoder(), config,
                                  np.random.default_rng(1), spec).run(game)
                    self.assertAlmostEqual(result.root_value, expected)
                    np.testing.assert_allclose(result.q_values[np.isfinite(result.q_values)], expected)
                    if reasons == ("max_moves",):
                        for player in (0, 1):
                            self.assertEqual(result.root_value, trajectory.outcome_value_for(player))

    def shogi_config(self, **params):
        from hasami_learn.config import NamedConfig

        config = self.tiny("alphazero", simulations=4, batch_games=2, **params)
        config.game = NamedConfig("shogi", {"max_moves": 1})
        config.encoder = NamedConfig("")
        config.selfplay.draw_reasons = ["repetition", "max_moves"]
        config.selfplay.games_per_iteration = 4
        config.selfplay.draw_value_start = -0.2
        config.selfplay.draw_value_end = 0.0
        config.selfplay.draw_value_iterations = 8
        return config.resolve_defaults()

    def test_collection_propagates_anneal_to_batched_search_and_spawned_workers(self):
        import numpy as np
        from hasami_learn.selfplay import draw_value_for_iteration, set_draw_value
        from hasami_learn.trainer import Trainer

        self.addCleanup(set_draw_value, 0.0, ("repetition",))
        for backend in ("python", "cpp"):
            for workers in (1, 2):
                with self.subTest(backend=backend, workers=workers):
                    config = self.shogi_config(search_backend=backend)
                    config.run.backend = backend
                    config.run.run_id = f"{backend}-{workers}"
                    config.selfplay.workers = workers
                    with closing(Trainer(config, echo=False)) as trainer:
                        learner = trainer.learner
                        # Deliberately stale globals must not override collect(iteration).
                        set_draw_value(0.7, [])
                        for iteration in (1, 4, 8):
                            value = draw_value_for_iteration(config.selfplay, iteration)
                            with patch.object(learner, "ingest_trajectories", wraps=learner.ingest_trajectories) as ingest:
                                stats = learner.collect(iteration)
                            self.assertEqual(stats.extra["terminal_reasons"], {"max_moves": 4})
                            self.assertEqual(stats.truncated, 0)
                            self.assertEqual(learner.search.config.draw_value, value)
                            job = learner._selfplay_job(iteration)
                            self.assertEqual(job["mcts_params"]["draw_value"], value)
                            for call in ingest.call_args_list:
                                for trajectory in call.args[0]:
                                    self.assertEqual(trajectory.draw_value, value)
                                    for i, extra in enumerate(trajectory.extras):
                                        self.assertAlmostEqual(extra["root_value"], value)
                                        self.assertAlmostEqual(trajectory.value_for(i), value)
                                        self.assertGreater(trajectory.masks[i][trajectory.actions[i]], 0)
                                        np.testing.assert_allclose(trajectory.policies[i].sum(), 1)

    def test_resume_mid_anneal_matches_uninterrupted_collection(self):
        import numpy as np
        from hasami_learn.selfplay import set_draw_value
        from hasami_learn.trainer import Trainer

        self.addCleanup(set_draw_value, 0.0, ("repetition",))
        def execute(config, resume=None, stop=False):
            with closing(Trainer(config, resume=resume, echo=False)) as trainer:
                if stop:
                    update = trainer.learner.update
                    def stop_after_second(iteration):
                        result = update(iteration)
                        if iteration == 2:
                            trainer._stop_reason = "test interruption"
                        return result
                    with patch.object(trainer.learner, "update", side_effect=stop_after_second):
                        trainer.run()
                else:
                    trainer.run()
                return trainer.run_dir, trainer.replay.all(), trainer.network.state_dict()

        config = self.shogi_config()
        config.run.iterations = 4
        config.replay.persist = True
        config.run.run_id = "uninterrupted"
        full_dir, full_replay, full_weights = execute(config)
        config.run.run_id = "resumed"
        run_dir, _, _ = execute(config, stop=True)
        set_draw_value(0.9, [])
        _, replay, weights = execute(config, resume=run_dir)
        for key in replay:
            np.testing.assert_array_equal(replay[key], full_replay[key])
        for key in weights:
            np.testing.assert_array_equal(weights[key].numpy(), full_weights[key].numpy())
        def collects(path):
            return [(row["iteration"], row["extra"]["draw_value"], row["terminal_reasons"])
                    for row in map(json.loads, (path / "metrics.jsonl").read_text().splitlines())
                    if row["kind"] == "collect"]
        self.assertEqual(collects(run_dir), collects(full_dir))
        np.testing.assert_allclose([value for _, value, _ in collects(run_dir)], [-0.2, -0.175, -0.15, -0.125])

    def test_serial_and_batched_table_actor_share_target_policy(self):
        from hasami_learn.batched_selfplay import TableRequest, run_table_games
        from hasami_learn.selfplay import run_selfplay_game, set_draw_value
        from hasami_learn.trainer import Trainer

        self.addCleanup(set_draw_value, 0.0, ("repetition",))
        with closing(Trainer(self.shogi_config(), echo=False)) as trainer:
            actor = trainer.learner.actor_fn()
            for reasons, expected in ((["max_moves"], -0.3), (["repetition"], 0.0)):
                set_draw_value(-0.3, reasons)
                serial = run_selfplay_game(actor, trainer.encoder, spec=trainer.spec, seed=1)
                batched = run_table_games([TableRequest({0: actor, 1: actor}, 1, 0)],
                                          trainer.encoder, spec=trainer.spec)[0]
                for trajectory in (serial, batched):
                    self.assertEqual(trajectory.reason, "max_moves")
                    self.assertAlmostEqual(trajectory.value_for(0), expected)
                    self.assertAlmostEqual(trajectory.extras[0]["root_value"], expected)

    def test_old_checkpoint_resume_rejected_but_inference_and_warm_start_allowed(self):
        from hasami_learn.config import ConfigError
        from hasami_learn.inference import create_agent
        from hasami_learn.manifest import save_checkpoint
        from hasami_learn.trainer import Trainer

        config = self.shogi_config()
        with closing(Trainer(config, echo=False)) as trainer:
            manifest = trainer.manifest(0)
            self.assertEqual(manifest.algorithm_params.pop("search_semantics_version"), 2)
            state = trainer.learner.state_dict()
            state["extra"] = {}
            old = self.tmp / "old"
            save_checkpoint(old, trainer.network, manifest, state, trainer.trainer_state())
            with self.assertRaisesRegex(ConfigError, "old search semantics"):
                trainer.learner.load_state_dict(state)
        with self.assertRaisesRegex(ConfigError, "old search semantics"):
            Trainer(config, resume=old, echo=False)
        self.assertEqual(create_agent(str(old), device="cpu").config.draw_value, 0.0)
        config.run.run_id = "warm-start"
        config.run.init_checkpoint = str(old)
        with closing(Trainer(config, echo=False)) as trainer:
            self.assertEqual(trainer.iteration, 0)
            self.assertEqual(len(trainer.replay), 0)
            self.assertEqual(trainer.manifest(0).algorithm_params["search_semantics_version"], 2)

    def test_config_validation_and_default_inference_ignores_training_globals(self):
        from hasami_learn.search.mcts import MctsConfig
        from hasami_learn.selfplay import set_draw_value
        from hasami_learn.inference import MctsAgent, create_agent
        from hasami_learn.trainer import Trainer

        for value in (float("nan"), float("inf"), -1.01, 1.01):
            with self.assertRaisesRegex(ValueError, "draw_value"):
                MctsConfig(draw_value=value)
        for reasons in (None, "repetition", (1,), ("",)):
            with self.assertRaisesRegex(ValueError, "draw_reasons"):
                MctsConfig(draw_reasons=reasons)
        self.addCleanup(set_draw_value, 0.0, ("repetition",))
        set_draw_value(-0.7, ["max_moves"])
        with closing(Trainer(self.shogi_config(), echo=False)) as trainer:
            learner = trainer.learner
            checkpoint = trainer.checkpoint(0)
            for agent in (learner.make_agent(), learner.make_agent("eval"),
                          MctsAgent(trainer.network, trainer.encoder, "cpu", spec=trainer.spec),
                          create_agent(str(checkpoint), device="cpu")):
                self.assertEqual(agent.config.draw_value, 0.0)
                self.assertEqual(agent.config.draw_reasons, ())
            self.assertEqual(learner.make_agent("selfplay").config.draw_value, -0.7)
            override = create_agent(str(checkpoint), device="cpu",
                                    mcts_params={"draw_value": -0.3, "draw_reasons": ["max_moves"]})
            self.assertEqual(override.config.draw_value, -0.3)
            self.assertEqual(override.config.draw_reasons, ("max_moves",))

    def test_gui_usi_ladder_and_practical_defaults(self):
        from contextlib import ExitStack
        from shogi.gui_support import load_strong_agent
        from shogi.usi import load_strong_agent as usi_load
        from hasami_learn.ladder import _make_agent
        from hasami_learn.practical import NetworkEvaluator, ProxyAgent, proxy_distributions
        from hasami_learn.practical_general import GeneralSearchAdvisor
        from hasami_learn.practical_search import SearchAdvisor
        from hasami_learn.search.mcts import MCTS
        from hasami_learn.selfplay import set_draw_value
        from hasami_learn.trainer import Trainer

        self.addCleanup(set_draw_value, 0.0, ("repetition",))
        set_draw_value(-0.8, ["max_moves"])
        with closing(Trainer(self.shogi_config(), echo=False)) as trainer:
            checkpoint = str(trainer.checkpoint(0))
            for factory in (load_strong_agent, usi_load):
                agent = factory(checkpoint, device="cpu", simulations=4).agent
                self.assertEqual(agent.config.draw_value, 0.0)
                self.assertEqual(agent.config.draw_reasons, ())
            ladder, _ = _make_agent({"spec": checkpoint, "manifest_sha256": "test"},
                                     1, 1000, 4, "cpu", trainer.spec, trainer.config.game.params)
            self.assertEqual(ladder.config.draw_value, 0.0)
            engine = NetworkEvaluator(trainer.network, trainer.encoder, trainer.spec)
            game = trainer.spec.create_game("python")
            self.addCleanup(game.close)
            self.assertEqual(set(engine.one_ply(game).values()), {0.0})
            with patch("hasami_learn.practical.MCTS", wraps=MCTS) as factory:
                proxy_distributions([ProxyAgent(engine, "qre", {"simulations": 4})], [game])
            self.assertEqual(factory.call_args.args[2].draw_value, 0.0)
            with ExitStack() as stack:
                edges, _ = SearchAdvisor(engine)._successors(
                    [game], [dict.fromkeys(trainer.spec.legal_actions(game, 0), 1)], stack, None)
                for edge in edges[0].values():
                    self.assertEqual(edge.rewards, (0.0, 0.0))
            trainer.spec.step(game, {0: trainer.spec.legal_actions(game, 0)[0]})
            general = GeneralSearchAdvisor(engine)
            self.assertEqual(general._score(game, 0), 0.5)
            self.assertEqual(general._score(game, 1), 0.5)

    def test_metrics_schema_accepts_legacy_and_rejects_invalid_counts(self):
        from hasami_learn.schema_validation import load_schema, validate
        from hasami_learn.metrics import MetricsLogger

        with MetricsLogger(self.tmp / "metrics.jsonl", "test", echo=False) as logger:
            row = logger.log("collect", 1, games=3, terminal_reasons={"repetition": 2, "max_plies": 1})
        schema = load_schema("training-metrics.schema.json")
        self.assertFalse(validate(schema, row))
        legacy = {key: value for key, value in row.items() if key != "terminal_reasons"}
        self.assertFalse(validate(schema, legacy))
        for bad in (-1, 1.5, True, "2"):
            self.assertTrue(validate(schema, dict(row, terminal_reasons={"repetition": bad})))


@requires_torch
class NativeDrawContemptTests(LearnTestCase):
    @classmethod
    def setUpClass(cls):
        from hasami_learn.search.native_mcts import find_library, _load_path
        try:
            _load_path(str(find_library()))
        except (RuntimeError, OSError, ValueError) as exc:
            if os.environ.get("HASAMI_REQUIRE_NATIVE") == "1":
                raise
            raise unittest.SkipTest(str(exc)) from exc

    def compare(self, spec, game, mode, forced, expected, reasons):
        import numpy as np
        from hasami_learn.search.mcts import MCTS, MctsConfig
        from hasami_learn.search.native_mcts import NativeMCTS

        def evaluate(planes, masks):
            priors = masks.astype(np.float64)
            for i, plane in enumerate(planes):
                action = forced.get(plane.tobytes())
                if action is not None:
                    priors[i] = 0
                    priors[i, action] = 1
            return priors, np.zeros(len(planes))
        config = MctsConfig(simulations=4, add_noise=False, root_selection=mode, gumbel_m=1,
                            draw_value=-0.2, draw_reasons=reasons)
        before = (game.position, game.outcome, tuple(game.history_keys()), tuple(game.moves))
        reference = MCTS(evaluate, spec.make_encoder(), config, np.random.default_rng(42), spec).run(game)
        native = NativeMCTS(evaluate, spec.make_encoder(), config, np.random.default_rng(42), spec).run(game)
        for field in ("visits", "q_values", "policy_target"):
            np.testing.assert_allclose(getattr(native, field), getattr(reference, field), rtol=0, atol=1e-12)
        self.assertEqual(native.root_value, reference.root_value)
        self.assertAlmostEqual(native.root_value, expected)
        self.assertEqual(native.simulations, reference.simulations)
        self.assertEqual(native.selected_action, reference.selected_action)
        self.assertEqual(before, (game.position, game.outcome, tuple(game.history_keys()), tuple(game.moves)))

    def test_repetition_both_games_both_players_and_multi_ply_backups(self):
        from hasami_learn.games import get_spec
        from shogi.notation import usi_to_move
        from shogi.encoding import move_label

        for name in ("hasami", "shogi"):
            spec = get_spec(name)
            options = {"position": "4k4/9/9/9/9/9/9/9/4K4 b - 1"} if name == "shogi" else {}
            for mode in ("puct", "gumbel"):
                with self.subTest(game=name, mode=mode):
                    game = spec.create_game("cpp", **options)
                    self.addCleanup(game.close)
                    cycle = ([72*81+63, 9, 63*81+72, 9*81] if name == "hasami" else
                             ["5i6i", "5a6a", "6i5i", "6a5a"])
                    forced = {}
                    for move in cycle * 3:
                        code = move if name == "hasami" else usi_to_move(game, move)
                        label = code if name == "hasami" else move_label(game, code)
                        forced[spec.make_encoder().encode(game).tobytes()] = label
                        game.play(code)
                    self.assertEqual(game.outcome.reason, "repetition")
                    self.compare(spec, game, mode, forced, -0.2, ("repetition",))
                    self.compare(spec, game, mode, forced, 0.0, ())
                    game.undo()
                    self.assertEqual(game.position.turn, 1)
                    self.compare(spec, game, mode, forced, -0.2, ("repetition",))
                    game.undo()
                    self.assertEqual(game.position.turn, 0)
                    # First simulation evaluates the intermediate node; the next three end in a draw.
                    self.compare(spec, game, mode, forced, -0.15, ("repetition",))

    def test_max_moves_and_declaration_with_nonzero_contempt(self):
        from pathlib import Path
        from hasami_learn.games import get_spec
        from shogi.model import DECLARE

        cases = json.loads((Path(__file__).parents[1] / "shogi_suite/vectors/rule_cases.json").read_text())
        declaration = next(case for case in cases["declaration_cases"] if case.get("declarable"))
        for mode in ("puct", "gumbel"):
            for turn in (0, 1):
                spec = get_spec("shogi", {"max_moves": turn + 1})
                game = spec.create_game("cpp")
                self.addCleanup(game.close)
                if turn:
                    game.play(game.legal_actions()[0])
                with self.subTest(mode=mode, turn=turn):
                    self.compare(spec, game, mode, {}, -0.2, ("max_moves",))
                    self.compare(spec, game, mode, {}, 0.0, ("repetition",))
            spec = get_spec("shogi")
            game = spec.create_game("cpp", position=declaration["sfen"])
            self.addCleanup(game.close)
            forced = {spec.make_encoder().encode(game).tobytes(): 2187}
            self.compare(spec, game, mode, forced, 1.0, ("repetition", "max_moves", "declaration"))
            game.play(DECLARE)
            self.assertEqual(game.outcome.reason, "declaration")
            expected = 1.0 if game.outcome.winner == game.position.turn else -1.0
            self.compare(spec, game, mode, {}, expected, ("repetition", "max_moves", "declaration"))
