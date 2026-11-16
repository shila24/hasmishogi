"""Executable GameSpec v2 contracts, including simultaneous hidden decisions."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ._common import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import numpy as np
    from hasami.agents import Decision, SearchLimits
    from hasami_learn.games.base import StepOutcome
    from hasami_learn.registry import GAMES

    class ToyGame:
        def __init__(self, players, seed=None):
            self.seed = seed
            self.rng = np.random.default_rng(seed)
            self.cards = self.rng.integers(0, 10, players).tolist()
            self.scores = [0] * players
            self.moves = []
            self.draws = []
            self.initial = {}

        @property
        def outcome(self):
            terminal = len(self.moves) == 3
            best = max(self.scores)
            winners = [p for p, score in enumerate(self.scores) if score == best]
            return SimpleNamespace(terminal=terminal, winner=winners[0] if terminal and len(winners) == 1 else -1 if terminal else -2,
                                   reason="finished" if terminal else "ongoing")

        def clone(self):
            return copy.deepcopy(self)

        def close(self):
            pass

    class ToyEncoder:
        encoding_id = "multiplayer-toy-enc"
        num_planes = 2

        def __init__(self, spec):
            self.spec = spec

        def encode(self, game, player=None):
            if player is None:
                player = self.spec.actors(game)[0]
            return self.spec.observe(game, player)

        def legal_mask(self, game, player=None):
            return self.spec.legal_mask(game, self.spec.actors(game)[0] if player is None else player)

    class FirstAgent:
        def __init__(self):
            self.seats = []

        def choose(self, game, limits, cancel=None, player=None):
            self.seats.append(player)
            return Decision(0, 0, 0, 0, 0, True)

    @GAMES.register("multiplayer-race-test")
    class RaceSpec:
        game_id = "multiplayer-race-test"
        ruleset_id = "race-test-v2"
        num_players = 3
        perfect_information = True
        board_shape = (1, 3)
        action_size = 2
        default_max_plies = 6

        def create_game(self, backend="python", rules=None, position=None, seed=None):
            return ToyGame(self.num_players, seed)

        def resolve_backend(self, backend):
            return "python"

        def rules_dict(self, game):
            return {}

        def actors(self, game):
            return [] if game.outcome.terminal else [len(game.moves) % self.num_players]

        def legal_actions(self, game, player):
            return [0, 1] if player in self.actors(game) else []

        def legal_mask(self, game, player):
            mask = np.zeros(2, dtype=np.float32)
            mask[self.legal_actions(game, player)] = 1
            return mask

        def observe(self, game, player):
            return np.array([[game.scores], [[player] * 3]], dtype=np.float32)

        def step(self, game, actions):
            if set(actions) != set(self.actors(game)):
                raise ValueError("wrong actors")
            for player, action in actions.items():
                if type(action) is not int or action not in self.legal_actions(game, player):
                    raise ValueError("illegal action")
                game.scores[player] += action + 1
            game.moves.append(dict(actions))
            return StepOutcome(game.outcome.terminal, False,
                               self.terminal_values(game) if game.outcome.terminal else None,
                               {"reason": game.outcome.reason})

        def terminal_values(self, game):
            if not game.outcome.terminal:
                raise ValueError("requires terminal game")
            return [score / 10 for score in game.scores]

        def make_encoder(self, name="", **params):
            return ToyEncoder(self)

        def baseline_agent(self, spec, seed=0):
            return FirstAgent(), None

        def symmetries(self):
            return []

        def action_name(self, action):
            return str(action)

    @GAMES.register("multiplayer-hidden-test")
    class HiddenSpec(RaceSpec):
        game_id = "multiplayer-hidden-test"
        ruleset_id = "hidden-test-v2"
        num_players = 2
        perfect_information = False

        def actors(self, game):
            return [] if game.outcome.terminal else [0, 1]

        def observe(self, game, player):
            return np.array([[[game.cards[player], len(game.moves), player]],
                             [[game.scores[player], 0, 0]]], dtype=np.float32)

        def step(self, game, actions):
            result = super().step(game, actions)
            draw = int(game.rng.integers(0, 2))
            game.draws.append(draw)
            game.scores[draw] += 1
            return StepOutcome(result.terminal, False, self.terminal_values(game) if result.terminal else None,
                               dict(result.info, draw=draw))

    class ScalarHiddenSpec(RaceSpec):
        """Two players alternate; the engine stores scalar moves and hides its seed (no game.seed)."""
        game_id = "scalar-hidden-test"
        ruleset_id = "scalar-hidden-v1"
        num_players = 2

        def create_game(self, backend="python", rules=None, position=None, seed=None):
            game = ToyGame(self.num_players, seed)
            if hasattr(game, "seed"):
                del game.seed  # the seed is consumed by the RNG and never exposed
            return game

        def actors(self, game):
            return [] if game.outcome.terminal else [len(game.moves) % 2]

        def observe(self, game, player):
            return np.array([[[game.cards[player], len(game.moves), player]], [[game.scores[player], 0, 0]]], dtype=np.float32)

        def step(self, game, actions):
            if set(actions) != set(self.actors(game)):
                raise ValueError("wrong actors")
            (player, action), = actions.items()
            if type(action) is not int or action not in self.legal_actions(game, player):
                raise ValueError("illegal action")
            game.scores[player] += action + 1
            game.moves.append(int(action))
            return StepOutcome(game.outcome.terminal, False,
                               self.terminal_values(game) if game.outcome.terminal else None,
                               {"reason": game.outcome.reason})


@unittest.skipUnless(TORCH_AVAILABLE, "torch is optional")
class MultiplayerTests(unittest.TestCase):
    def test_joint_observations_rewards_and_duplicate_chance(self):
        from hasami_learn.selfplay import run_selfplay_game, run_table_game
        for spec in (RaceSpec(), HiddenSpec()):
            observed = []
            def actor(game, planes, mask, ply, player):
                np.testing.assert_array_equal(planes, spec.observe(game, player))
                observed.append((ply, player))
                return 1, mask / mask.sum(), {}
            first = run_selfplay_game(actor, spec.make_encoder(), spec=spec, seed=31)
            second = run_selfplay_game(actor, spec.make_encoder(), spec=spec, seed=31)
            self.assertEqual(first.rewards, second.rewards)
            self.assertEqual(first.step_info, second.step_info)
            self.assertEqual(len(first.rewards), spec.num_players)
            self.assertEqual(first.values().tolist(), np.array([first.rewards[p] for p in first.players], np.float32).tolist())
            self.assertEqual(set(first.players), set(range(spec.num_players)))
            self.assertEqual(first.plies, 3)
            mixed = run_table_game({p: actor if p == 0 else FirstAgent() for p in range(spec.num_players)},
                                   spec.make_encoder(), spec=spec, seed=31)
            self.assertEqual(set(mixed.players), {0})
            self.assertEqual(len(mixed.joint_actions), 3)

    def test_table_rotations_and_legacy_keys(self):
        from hasami_learn.evaluation import MatchResult, play_matches
        for spec in (RaceSpec(), HiddenSpec()):
            agents = [FirstAgent() for _ in range(spec.num_players)]
            result = play_matches(agents[0], agents[1:], spec.num_players, spec=spec,
                                  limits=SearchLimits(), seed=42)
            self.assertEqual(result.games, spec.num_players)
            self.assertEqual(set(agents[0].seats), set(range(spec.num_players)))
            self.assertEqual(sum(result.placements.values()), spec.num_players)
            self.assertTrue(np.isfinite(result.mean_points))
            self.assertEqual(len({g["seed"] for g in result.tables}), 1)
            if not spec.perfect_information:
                self.assertEqual(result.tables[0]["step_info"], result.tables[1]["step_info"])
            if spec.num_players == 2:
                self.assertNotIn("placements", result.to_dict())
        self.assertNotIn("placements", MatchResult().to_dict())

    def test_train_and_checkpoint_each_game(self):
        from hasami_learn.config import NamedConfig, tiny_config
        from hasami_learn.trainer import Trainer
        from hasami_learn.inference import create_agent
        for spec in (RaceSpec(), HiddenSpec()):
            for algorithm in ("reinforce", "ppo", "dqn", "supervised", "adversarial"):
                with self.subTest(game=spec.game_id, algorithm=algorithm), tempfile.TemporaryDirectory() as directory:
                    cfg = tiny_config(algorithm)
                    cfg.game = NamedConfig(spec.game_id)
                    cfg.run.output_dir = directory
                    cfg.selfplay.games_per_iteration = 3
                    cfg.selfplay.max_plies = 6
                    cfg.evaluation.games = spec.num_players
                    cfg.evaluation.opponents = ["random"]
                    cfg.replay.min_size = 1
                    if algorithm == "supervised":
                        cfg.algorithm.params = {"generate_games": 2}
                    if algorithm == "adversarial":
                        cfg.algorithm.params = {"snapshot_every": 1, "exploiter_every": 1,
                                                "exploiter_games": 1, "exploiter_iterations": 1}
                    trainer = Trainer(cfg, echo=False)
                    try:
                        checkpoint = trainer.run()
                        agent = create_agent(str(checkpoint), device="cpu")
                        game = spec.create_game(seed=1)
                        try:
                            for player in spec.actors(game):
                                self.assertIn(agent.choose(game, SearchLimits(), player=player).action,
                                              spec.legal_actions(game, player))
                        finally:
                            game.close()
                    finally:
                        trainer.close()

    def test_mcts_refuses_unsupported_games(self):
        from hasami_learn.algorithms.mcts import MCTS, MctsConfig
        from hasami_learn.config import NamedConfig, tiny_config
        from hasami_learn.trainer import Trainer
        for spec in (RaceSpec(), HiddenSpec()):
            with self.assertRaisesRegex(ValueError, "two.player.*perfect.information"):
                MCTS(lambda x: None, spec.make_encoder(), MctsConfig(), np.random.default_rng(0), spec=spec)
            with tempfile.TemporaryDirectory() as directory:
                cfg = tiny_config("alphazero")
                cfg.game = NamedConfig(spec.game_id)
                cfg.run.output_dir = directory
                with self.assertRaisesRegex(ValueError, "two.player.*perfect.information"):
                    Trainer(cfg, echo=False)

    def test_archive_and_dataset_joint_replay(self):
        from hasami_learn.datastore import SelfplayStore
        from hasami_learn.selfplay import run_selfplay_game
        for spec in (RaceSpec(), HiddenSpec()):
            def actor(game, planes, mask, ply):
                return 0, mask / mask.sum(), {}
            with tempfile.TemporaryDirectory() as directory:
                trajectory = run_selfplay_game(actor, spec.make_encoder(), spec=spec, seed=21)
                store = SelfplayStore(Path(directory), spec=spec)
                store.add(trajectory, 1)
                saved = next(store.iter_games())
                self.assertEqual(saved["num_players"], spec.num_players)
                data = store.to_dataset(spec.make_encoder())
                np.testing.assert_array_equal(data["planes"], np.stack(trajectory.planes))
                np.testing.assert_array_equal(data["value"], trajectory.values())

    def test_records_and_supervised_labels_use_each_seat(self):
        from hasami_learn.datasets import generate_games, build_dataset, validate_dataset
        for spec in (RaceSpec(), HiddenSpec()):
            with tempfile.TemporaryDirectory() as directory:
                paths = generate_games(directory, 2, generator="random", spec=spec, seed=41)
                data = build_dataset(paths, spec.make_encoder(), augment=False, spec=spec)
                validate_dataset(data, spec=spec)
                self.assertEqual(data["meta"]["num_players"], spec.num_players)
                self.assertEqual(data["planes"].shape[0], 6 if spec.num_players == 3 else 12)
                self.assertTrue(np.all(data["value"] > 0))

    def test_ladder_ratings_rotations_and_resume(self):
        from hasami_learn.ladder import fit_plackett_luce, run_ladder
        names = ["a", "b", "c"]
        ratings = fit_plackett_luce(names, [{"entries": names, "values": [1, 0, -1]}] * 10)
        self.assertGreater(ratings["a"], ratings["b"])
        self.assertGreater(ratings["b"], ratings["c"])
        tied = fit_plackett_luce(names, [{"entries": names, "values": [1, 1, 1]}])
        self.assertEqual(tied, dict.fromkeys(names, 0.0))
        tied_leaders = fit_plackett_luce(names, [{"entries": names, "values": [1, 1, -1]}] * 10)
        self.assertAlmostEqual(tied_leaders["a"], tied_leaders["b"])
        self.assertGreater(tied_leaders["b"], tied_leaders["c"])
        with tempfile.TemporaryDirectory() as directory:
            kwargs = dict(games=3, backend="python", game_id=RaceSpec.game_id, out=directory)
            report = run_ladder(["a=random", "b=random", "c=random"], **kwargs)
            self.assertTrue(report["complete"])
            self.assertEqual(report["elo_method"]["name"], "Plackett-Luce MM")
            games = report["tables"][0]["games"]
            self.assertEqual([g["rotation"] for g in games], [0, 1, 2])
            self.assertEqual(len({g["seed"] for g in games}), 1)
            self.assertEqual(report, run_ladder(["a=random", "b=random", "c=random"], resume=True, **kwargs))

    def test_chance_only_step_and_actor_isolation(self):
        from hasami_learn.selfplay import run_selfplay_game
        class ChanceSpec(HiddenSpec):
            def actors(self, game):
                return [] if len(game.moves) == 0 else super().actors(game)

            def step(self, game, actions):
                return super().step(game, actions)
        spec = ChanceSpec()
        def actor(game, planes, mask, ply, player):
            game.scores[player] = 999
            return 0, None, {}
        result = run_selfplay_game(actor, spec.make_encoder(), spec=spec, seed=4)
        self.assertEqual(result.joint_actions[0], {})
        self.assertTrue(all(v <= 1 for v in result.rewards))
        self.assertEqual(len(result), 4)
        def illegal(game, planes, mask, ply):
            return True, None, {}
        with self.assertRaisesRegex(ValueError, "illegal action"):
            run_selfplay_game(illegal, spec.make_encoder(), spec=spec, seed=4)
        def broken(game, planes, mask, ply, player):
            raise TypeError("callback failed internally")
        with self.assertRaisesRegex(TypeError, "callback failed internally"):
            run_selfplay_game(broken, spec.make_encoder(), spec=spec, seed=4)

    def test_dqn_transitions_never_bootstrap_another_seat(self):
        from hasami_learn.config import NamedConfig, tiny_config
        from hasami_learn.selfplay import run_selfplay_game
        from hasami_learn.trainer import Trainer
        spec = HiddenSpec()
        with tempfile.TemporaryDirectory() as directory:
            cfg = tiny_config("dqn")
            cfg.game = NamedConfig(spec.game_id)
            cfg.run.output_dir = directory
            cfg.selfplay.symmetry_augmentation = False
            trainer = Trainer(cfg, echo=False)
            try:
                trajectory = run_selfplay_game(lambda g, p, m, t: (0, None, {}), spec.make_encoder(), spec=spec, seed=15)
                trainer.learner.ingest_trajectories([trajectory])
                data = trainer.replay.all()
                for i in range(4):
                    np.testing.assert_array_equal(data["next_planes"][i], trajectory.planes[i + 2])
                np.testing.assert_array_equal(data["done"], [0, 0, 0, 0, 1, 1])
                np.testing.assert_allclose(data["reward"][-2:], trajectory.rewards)
            finally:
                trainer.close()

    def test_game_defaults_and_metadata_mismatch(self):
        from hasami_learn.config import NamedConfig, TrainingConfig
        from hasami_learn.games import GameSpec, get_spec
        from hasami_learn.manifest import ModelManifest, ManifestError, check_manifest_matches_spec
        cfg = TrainingConfig(game=NamedConfig(RaceSpec.game_id))
        self.assertIsInstance(get_spec(RaceSpec.game_id), GameSpec)
        self.assertEqual(cfg.selfplay.draw_reasons, [])
        self.assertEqual(cfg.opponents.classical, ["random"])
        changed = cfg.with_overrides(game__name="hasami")
        self.assertEqual(changed.selfplay.draw_reasons, ["repetition"])
        self.assertEqual(changed.evaluation.opponents, ["random", "greedy"])
        cfg.evaluation.opponents.append("custom")
        cfg.resolve_defaults()
        self.assertEqual(cfg.to_dict()["evaluation"]["opponents"], ["random", "custom"])
        manifest = ModelManifest(algorithm="reinforce", network={"name": "mlp", "params": {}, "head": "policy_value"},
                                 encoding_id="hasami-enc-v1", num_planes=9, num_players=3)
        with self.assertRaisesRegex(ManifestError, "num_players"):
            check_manifest_matches_spec(manifest, get_spec())

    def test_parallel_batch_records_joint_actor_seats(self):
        from hasami_learn.parallel import selfplay_batches
        spec = HiddenSpec()
        class Search:
            config = SimpleNamespace(simulations=1)
            def run_many(self, games, simulations, players):
                return [spec.legal_mask(game, p) for game, p in zip(games, players)]
        def decision(result, mask, ply):
            return 0, mask / mask.sum(), {}
        trajectories = next(selfplay_batches(Search(), spec.make_encoder(), decision, games=2,
                                             batch_games=2, backend="python", max_plies=6,
                                             draw_value=-0.9, draw_reasons=["finished"], spec=spec, seed=15))
        for trajectory in trajectories:
            self.assertEqual(trajectory.num_players, 2)
            self.assertEqual(trajectory.players, [0, 1] * 3)
            self.assertEqual(trajectory.draw_reasons, ())
            np.testing.assert_allclose(trajectory.values(), [trajectory.rewards[p] for p in trajectory.players])


@unittest.skipUnless(TORCH_AVAILABLE, "torch is optional")
class ReviewRegressionTests(unittest.TestCase):
    """Regressions for the multiplayer search review findings."""

    def _spec(self, name):
        from hasami_learn.games import get_spec
        from hasami_learn.registry import GAMES

        if name not in GAMES:
            self.skipTest(f"{name} is registered by the multiplayer test setup")
        return get_spec(name, {})

    def test_resume_accepts_legacy_config_with_spelled_out_defaults(self):
        import argparse
        import json
        import tempfile
        from pathlib import Path

        from hasami_learn.cli import resolve_train_request
        from hasami_learn.config import tiny_config
        from hasami_learn.trainer import train_from_config

        with tempfile.TemporaryDirectory() as directory:
            config = tiny_config("reinforce", run_id="legacy-lists")
            config.run.output_dir = str(Path(directory) / "runs")
            train_from_config(config, echo=False)
            saved_path = Path(directory) / "runs" / "legacy-lists" / "config.json"
            legacy = json.loads(saved_path.read_text())
            resolved = config.to_dict(preserve_defaults=False)  # a pre-change config spelled every default out
            legacy["selfplay"]["draw_reasons"] = resolved["selfplay"]["draw_reasons"]
            legacy["opponents"]["classical"] = resolved["opponents"]["classical"]
            legacy["evaluation"]["opponents"] = resolved["evaluation"]["opponents"]
            legacy["selfplay"]["max_plies"] = resolved["selfplay"]["max_plies"]
            saved_path.write_text(json.dumps(legacy))
            args = argparse.Namespace(config=None, algorithm="reinforce", tiny=True, resume="auto", set=[], device=None, iterations=2,
                                      run_id="legacy-lists", output_dir=str(Path(directory) / "runs"), seed=None, backend=None, games=None, quiet=True)
            resolved_config, resume = resolve_train_request(args)
            self.assertIsNotNone(resume)
            self.assertEqual(resolved_config.run.iterations, 2)

    def test_loader_rejects_wrong_player_count(self):
        import json
        from pathlib import Path

        from hasami_learn.manifest import MANIFEST_FILE, ManifestError, load_checkpoint, resolve_checkpoint_dir

        directory = resolve_checkpoint_dir(Path("runs/alphazero-mps-demo")) if Path("runs/alphazero-mps-demo").exists() else None
        if directory is None:
            self.skipTest("demo checkpoint not present")
        manifest = json.loads((directory / MANIFEST_FILE).read_text())
        manifest["num_players"] = 3
        from unittest.mock import patch
        original = Path.read_text

        def fake_read_text(self, *args, **kwargs):
            if self.name == MANIFEST_FILE and self.parent == directory:
                return json.dumps(manifest)
            return original(self, *args, **kwargs)

        with patch.object(Path, "read_text", fake_read_text):
            with self.assertRaises(ManifestError):
                load_checkpoint(directory, "cpu", build_network=False)

    def test_v2_record_survives_load_save_load(self):
        import tempfile
        from pathlib import Path

        from hasami_learn.games.records import RecordedGame, load_record, save_record
        from hasami_learn.selfplay import run_selfplay_game

        for name in ("multiplayer-race-test", "multiplayer-hidden-test"):
            spec = self._spec(name)
            encoder = spec.make_encoder("")

            def actor(game, planes, mask, ply, player=None):
                import numpy as np
                legal = np.flatnonzero(mask > 0)
                return int(legal[0]), None, {}

            trajectory = run_selfplay_game(actor, encoder, spec=spec, backend="python", max_plies=spec.default_max_plies, seed=11)
            with tempfile.TemporaryDirectory() as directory:
                first = Path(directory) / "a.json"
                game = spec.create_game("python", seed=11)
                try:
                    for joint in trajectory.joint_actions:
                        spec.step(game, {int(k): v for k, v in joint.items()})
                    save_record(spec, game, first, trajectory=trajectory)
                finally:
                    game.close()
                loaded = load_record(spec, first)
                self.assertIsInstance(loaded, RecordedGame)
                second = Path(directory) / "b.json"
                save_record(spec, loaded, second)  # no trajectory: metadata must be preserved
                again = load_record(spec, second)
                self.assertIsInstance(again, RecordedGame)
                self.assertEqual(again.record["joint_actions"], loaded.record["joint_actions"])
                self.assertEqual(again.record["num_players"], spec.num_players)

    def test_exploiter_games_are_seeded_deterministically(self):
        import inspect

        from hasami_learn.algorithms import adversarial

        source = inspect.getsource(adversarial.AdversarialLearner._train_exploiter)
        self.assertIn("seed=exploiter_base", source)

    def test_two_player_ladder_forwards_the_pair_seed(self):
        import inspect

        from hasami_learn import ladder

        source = inspect.getsource(ladder.run_ladder)
        self.assertIn("seed=pair_seed", source)

    def test_direct_save_of_v2_games_and_continued_records(self):
        import tempfile
        from pathlib import Path

        from hasami_learn.games.records import RecordedGame, load_record, save_record

        def finish(spec, game):
            while not game.outcome.terminal:
                actors = spec.actors(game)
                if not actors:
                    break
                spec.step(game, {p: spec.legal_actions(game, p)[0] for p in actors})

        for name in ("multiplayer-race-test", "multiplayer-hidden-test"):
            spec = self._spec(name)
            with tempfile.TemporaryDirectory() as directory:
                game = spec.create_game("python", seed=5)
                try:
                    finish(spec, game)
                    path = Path(directory) / "direct.json"
                    save_record(spec, game, path)  # no trajectory, plain engine object
                    loaded = load_record(spec, path)
                    self.assertEqual(list(loaded.moves), list(game.moves))
                finally:
                    game.close()
                # A loaded record that is continued and saved again must reflect the extra steps.
                partial = spec.create_game("python", seed=6)
                try:
                    actors = spec.actors(partial)
                    spec.step(partial, {p: spec.legal_actions(partial, p)[0] for p in actors})
                    first = Path(directory) / "partial.json"
                    save_record(spec, partial, first)
                finally:
                    partial.close()
                reloaded = load_record(spec, first)
                finish(spec, reloaded)
                second = Path(directory) / "continued.json"
                save_record(spec, reloaded, second)
                final = load_record(spec, second)
                self.assertEqual(list(final.moves), list(reloaded.moves))
                self.assertTrue(final.outcome.terminal)

    def test_ladder_accepts_adapter_defined_truncation_reasons(self):
        from hasami_learn.ladder import _restore_result

        data = {"games": 4, "wins": 1, "losses": 1, "draws": 2, "truncated": 2, "plies": 40, "seconds": 0.5,
                "reasons": {"resource_limit": 2, "finished": 2}}
        result = _restore_result(data, 4, spec=self._spec("multiplayer-hidden-test"))  # adapter without a reason whitelist
        self.assertEqual(result.truncated, 2)
        self.assertEqual(result.reasons["resource_limit"], 2)

    def test_capped_record_keeps_truncation_and_seed_without_public_seed(self):
        import json
        import tempfile
        from pathlib import Path

        from hasami_learn.games.records import load_record, save_record
        from hasami_learn.selfplay import run_selfplay_game

        spec = self._spec("multiplayer-hidden-test")
        encoder = spec.make_encoder("")

        def actor(game, planes, mask, ply, player=None):
            import numpy as np
            return int(np.flatnonzero(mask > 0)[0]), None, {}

        trajectory = run_selfplay_game(actor, encoder, spec=spec, backend="python", max_plies=1, seed=31)
        self.assertTrue(trajectory.truncated)
        with tempfile.TemporaryDirectory() as directory:
            game = spec.create_game("python", seed=31)
            try:
                for joint in trajectory.joint_actions:
                    spec.step(game, {int(k): v for k, v in joint.items()})
                first = Path(directory) / "capped.json"
                save_record(spec, game, first, trajectory=trajectory)
            finally:
                game.close()
            loaded = load_record(spec, first)
            delattr(loaded.game, "seed")  # engine that accepts a seed but does not expose it
            second = Path(directory) / "resaved.json"
            save_record(spec, loaded, second)
            data = json.loads(second.read_text())
            self.assertEqual(data["version"], 2)
            self.assertTrue(data["truncated"])
            self.assertEqual(data["seed"], 31)
            self.assertIsInstance(load_record(spec, second).record, dict)

    def test_batched_runner_keeps_adapter_truncation_reason(self):
        from types import SimpleNamespace

        import numpy as np
        from hasami_learn.games.base import StepOutcome
        from hasami_learn.parallel import selfplay_batches

        hidden = self._spec("multiplayer-hidden-test")

        class Capped(type(hidden)):
            def step(self, game, actions):
                super().step(game, actions)
                return StepOutcome(False, True, None, {"reason": "resource_limit"})

        class FakeSearch:
            config = SimpleNamespace(simulations=1)

            def run_many(self, snapshots, players=None, **kwargs):
                return [None] * len(snapshots)

        def decision(result, mask, plies):
            return int(np.flatnonzero(mask > 0)[0]), None, {}

        spec = Capped()
        batches = list(selfplay_batches(FakeSearch(), spec.make_encoder(""), decision, games=1, batch_games=1, backend="python",
                                        max_plies=0, draw_value=0.0, draw_reasons=[], spec=spec, seed=3))
        trajectory = batches[0][0]
        self.assertTrue(trajectory.truncated)
        self.assertEqual(trajectory.reason, "resource_limit")

    def test_dataset_regenerates_observations_from_a_seeded_live_game(self):
        import numpy as np
        from hasami_learn.datasets import iter_record_samples

        spec = self._spec("multiplayer-hidden-test")
        encoder = spec.make_encoder("")
        game = spec.create_game("python", seed=5)
        seen = []
        try:
            while not game.outcome.terminal:
                actors = spec.actors(game)
                if not actors:
                    break
                for p in actors:
                    seen.append(spec.observe(game, p).tobytes())
                spec.step(game, {p: spec.legal_actions(game, p)[0] for p in actors})
            regenerated = [planes.tobytes() for planes, _, _, _ in iter_record_samples(game, encoder, spec=spec)]
        finally:
            game.close()
        self.assertEqual(sorted(regenerated), sorted(seen))

    def test_dataset_uses_the_continued_history_and_clones_keep_metadata(self):
        import json
        import tempfile
        from pathlib import Path

        from hasami_learn.datasets import _decision_count, iter_record_samples
        from hasami_learn.games.records import RecordedGame, load_record, save_record

        spec = self._spec("multiplayer-hidden-test")
        encoder = spec.make_encoder("")
        with tempfile.TemporaryDirectory() as directory:
            partial = spec.create_game("python", seed=31)
            try:
                actors = spec.actors(partial)
                spec.step(partial, {p: spec.legal_actions(partial, p)[0] for p in actors})
                path = Path(directory) / "one-step.json"
                save_record(spec, partial, path)
            finally:
                partial.close()
            loaded = load_record(spec, path)
            self.assertIsInstance(loaded, RecordedGame)
            clone = loaded.clone()
            self.assertIsInstance(clone, RecordedGame)
            delattr(clone.game, "seed")
            clone_path = Path(directory) / "clone.json"
            save_record(spec, clone, clone_path)
            self.assertEqual(json.loads(clone_path.read_text())["seed"], 31)
            while not loaded.outcome.terminal:
                actors = spec.actors(loaded)
                spec.step(loaded, {p: spec.legal_actions(loaded, p)[0] for p in actors})
            samples = list(iter_record_samples(loaded, encoder, spec=spec))
            self.assertEqual(len(samples), 6)  # 2 actors x 3 joint steps, including the continuation
            self.assertEqual(_decision_count(loaded), 6)

    def test_edited_scalar_history_is_not_masked_by_stale_joint_actions(self):
        import json
        import tempfile
        from pathlib import Path

        from hasami_learn.datasets import _history_steps, iter_record_samples
        from hasami_learn.games.records import RecordedGame, history_matches, load_record, save_record

        self.assertTrue(history_matches([{"0": 0}, {"1": 1}], [0, 1]))
        self.assertFalse(history_matches([{"0": 0}, {"1": 1}], [0, 2]))
        self.assertFalse(history_matches([{"0": 0, "1": 1}], [0]))
        spec = ScalarHiddenSpec()
        encoder = spec.make_encoder("")
        with tempfile.TemporaryDirectory() as directory:
            # Build a scalar-move game whose history [0, 0, 1] differs from a stale joint record of the same
            # length ([0, 0, 0]); the stale record must not mask the edited history anywhere.
            other = spec.create_game("python", seed=2)
            try:
                for step, action in enumerate((0, 0, 1)):
                    spec.step(other, {spec.actors(other)[0]: action})
                self.assertTrue(other.outcome.terminal)
                stale = {"version": 2, "seed": 2, "joint_actions": [{"0": 0}, {"1": 0}, {"0": 0}], "rewards": None, "truncated": False}
                edited = RecordedGame(other, stale)
                self.assertEqual(_history_steps(edited), [0, 0, 1])
                second = Path(directory) / "edited.json"
                save_record(spec, edited, second)
                written = json.loads(second.read_text())
                self.assertEqual(written["version"], 1)
                self.assertNotIn("joint_actions", written)
                self.assertEqual(written["moves"], [0, 0, 1])
                self.assertEqual(written["seed"], 2)
                reloaded = load_record(spec, second)
                self.assertEqual(list(reloaded.moves), [0, 0, 1])
                self.assertEqual(reloaded.record.get("seed"), 2)
                samples = list(iter_record_samples(reloaded, encoder, spec=spec))
                self.assertEqual([sample[2] for sample in samples], [0, 0, 1])
                self.assertEqual(list(reloaded.cards), list(other.cards))
            finally:
                other.close()

    def test_recorded_game_forwards_assignments_to_the_engine(self):
        from hasami_learn.games.records import RecordedGame

        spec = self._spec("multiplayer-race-test")
        game = spec.create_game("python", seed=1)
        try:
            wrapped = RecordedGame(game, {"version": 2, "joint_actions": [], "seed": 1})
            wrapped.scores = [7, 7, 7]
            self.assertEqual(game.scores, [7, 7, 7])
            self.assertEqual(wrapped.clone().scores, [7, 7, 7])
        finally:
            game.close()

    def test_eval_default_opponents_come_from_the_game_spec(self):
        import inspect

        from hasami_learn import cli

        source = inspect.getsource(cli.cmd_eval)
        self.assertIn("default_baselines", source)
        self.assertNotIn('["random", "greedy"]', inspect.getsource(cli.main))

    def test_scalar_history_record_keeps_hidden_seed_through_continue_reload_extract_resave(self):
        """A continued v2 record of a scalar-move engine is resaved as version 1; its seed must survive
        reload (RecordedGame), dataset extraction (same deal), and another save."""
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        from hasami_learn.datasets import iter_record_samples
        from hasami_learn.games.records import RecordedGame, load_record, save_record

        spec = ScalarHiddenSpec()
        encoder = spec.make_encoder("")
        seed = 11
        with tempfile.TemporaryDirectory() as directory:
            game = spec.create_game("python", seed=seed)
            try:
                self.assertFalse(hasattr(game, "seed"))
                spec.step(game, {0: 1})
                first = Path(directory) / "v2.json"
                save_record(spec, game, first, trajectory=SimpleNamespace(joint_actions=[{0: 1}], rewards=None, truncated=False, seed=seed))
            finally:
                game.close()
            self.assertEqual(json.loads(first.read_text())["version"], 2)
            loaded = load_record(spec, first)
            spec.step(loaded, {1: 0})  # continue: the scalar history no longer matches the recorded joint actions
            second = Path(directory) / "continued.json"
            save_record(spec, loaded, second)
            written = json.loads(second.read_text())
            self.assertEqual((written["version"], written["seed"], written["moves"]), (1, seed, [1, 0]))

            reloaded = load_record(spec, second)
            self.assertIsInstance(reloaded, RecordedGame)
            self.assertEqual(reloaded.record.get("seed"), seed)
            expected_cards = spec.create_game("python", seed=seed).cards
            self.assertEqual(list(reloaded.cards), list(expected_cards))
            samples = list(iter_record_samples(reloaded, encoder, spec=spec))
            self.assertEqual([sample[2] for sample in samples], [1, 0])
            self.assertEqual(float(samples[0][0][0, 0, 0]), float(expected_cards[0]))
            self.assertEqual(float(samples[1][0][0, 0, 0]), float(expected_cards[1]))

            third = Path(directory) / "resaved.json"
            save_record(spec, reloaded, third)
            self.assertEqual(json.loads(third.read_text())["seed"], seed)
            self.assertEqual(reloaded.clone().record.get("seed"), seed)

    def test_batch_construction_failure_closes_created_games(self):
        """selfplay_batches registers each game's close right after creation, so a later create_game
        failure never leaks the games created earlier in the batch."""
        from hasami_learn.parallel import selfplay_batches

        closed = []

        class Flaky(ScalarHiddenSpec):
            calls = 0

            def create_game(self, backend="python", rules=None, position=None, seed=None):
                Flaky.calls += 1
                if Flaky.calls == 2:
                    raise RuntimeError("second creation fails")
                game = super().create_game(backend, rules, position, seed)
                game.close = lambda: closed.append(seed)
                return game

        spec = Flaky()

        class Search:
            config = SimpleNamespace(simulations=1)

            def run_many(self, games, simulations, players):
                return [spec.legal_mask(game, p) for game, p in zip(games, players)]

        def decision(result, mask, ply):
            return 0, mask / mask.sum(), {}

        with self.assertRaisesRegex(RuntimeError, "second creation fails"):
            next(selfplay_batches(Search(), spec.make_encoder(), decision, games=2, batch_games=2, backend="python",
                                  max_plies=6, draw_value=0.0, draw_reasons=["finished"], spec=spec, seed=15))
        self.assertEqual(Flaky.calls, 2)
        self.assertEqual(len(closed), 1)

    def _translating_spec(self, with_hook):
        """Scalar-move engine whose adapter action a plays raw move 1 - a."""

        class Translating(ScalarHiddenSpec):
            game_id = "translating-hook-test" if with_hook else "translating-nohook-test"
            ruleset_id = "translating-v1"

            def create_game(self, backend="python", rules=None, position=None, seed=None):
                return ToyGame(self.num_players, seed)  # exposes game.seed, so live games save directly

            def step(self, game, actions):
                if set(actions) != set(self.actors(game)):
                    raise ValueError("wrong actors")
                (player, action), = actions.items()
                if type(action) is not int or action not in self.legal_actions(game, player):
                    raise ValueError("illegal action")
                raw = 1 - action
                game.scores[player] += raw + 1
                game.moves.append(raw)
                return StepOutcome(game.outcome.terminal, False,
                                   self.terminal_values(game) if game.outcome.terminal else None,
                                   {"reason": game.outcome.reason})

        if with_hook:
            def history_actions(self, game):
                return [{len(game.moves[:i]) % 2: 1 - raw} for i, raw in enumerate(game.moves)]
            Translating.history_actions = history_actions
        return Translating()

    def test_translating_adapter_records_and_datasets_stay_in_adapter_action_space(self):
        import json
        import tempfile
        from pathlib import Path

        from hasami_learn.datasets import _history_steps, iter_record_samples
        from hasami_learn.games.records import RecordedGame, load_record, save_record

        spec = self._translating_spec(with_hook=True)
        encoder = spec.make_encoder("")
        with tempfile.TemporaryDirectory() as directory:
            game = spec.create_game("python", seed=3)
            try:
                for action in (0, 0):
                    spec.step(game, {spec.actors(game)[0]: action})
                self.assertEqual(list(game.moves), [1, 1])  # raw engine moves differ from the actions
                self.assertEqual(_history_steps(game, spec), [{0: 0}, {1: 0}])
                first = Path(directory) / "translated.json"
                save_record(spec, game, first)
                scores = list(game.scores)
            finally:
                game.close()
            written = json.loads(first.read_text())
            self.assertEqual((written["version"], written["joint_actions"]), (2, [{"0": 0}, {"1": 0}]))
            loaded = load_record(spec, first)
            self.assertIsInstance(loaded, RecordedGame)
            self.assertEqual(list(loaded.moves), [1, 1])
            self.assertEqual(list(loaded.scores), scores)
            # Unchanged history: resaving keeps the joint actions instead of downgrading to raw moves.
            second = Path(directory) / "resaved.json"
            save_record(spec, loaded, second)
            self.assertEqual(json.loads(second.read_text())["joint_actions"], [{"0": 0}, {"1": 0}])
            samples = list(iter_record_samples(loaded, encoder, spec=spec))
            self.assertEqual([sample[2] for sample in samples], [0, 0])
            # Continuing the loaded record keeps the adapter action space through the hook.
            spec.step(loaded, {0: 1})
            self.assertTrue(loaded.outcome.terminal)
            third = Path(directory) / "continued.json"
            save_record(spec, loaded, third)
            continued = json.loads(third.read_text())
            self.assertEqual((continued["version"], continued["joint_actions"]), (2, [{"0": 0}, {"1": 0}, {"0": 1}]))
            self.assertEqual(continued["rewards"], list(spec.terminal_values(loaded)))
            final = load_record(spec, third)
            self.assertEqual(list(final.scores), list(loaded.scores))
            self.assertEqual([sample[2] for sample in iter_record_samples(final, encoder, spec=spec)], [0, 0, 1])
            with self.assertRaisesRegex(ValueError, "version-1"):
                stale = Path(directory) / "v1.json"
                stale.write_text(json.dumps(dict(written, version=1, moves=[1, 1])))
                load_record(spec, stale)

    def test_hookless_translating_adapter_is_rejected_instead_of_mislabelled(self):
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        from hasami_learn.datasets import iter_record_samples
        from hasami_learn.games.records import load_record, save_record

        spec = self._translating_spec(with_hook=False)
        encoder = spec.make_encoder("")
        with tempfile.TemporaryDirectory() as directory:
            game = spec.create_game("python", seed=3)
            try:
                for action in (0, 0, 0):
                    spec.step(game, {spec.actors(game)[0]: action})
                with self.assertRaisesRegex(ValueError, "history_actions"):
                    list(iter_record_samples(game, encoder, spec=spec))
                path = Path(directory) / "translated.json"
                save_record(spec, game, path, trajectory=SimpleNamespace(joint_actions=[{0: 0}, {1: 0}, {0: 0}], rewards=None,
                                                                        truncated=False, seed=3))
            finally:
                game.close()
            with self.assertRaisesRegex(ValueError, "history_actions"):
                load_record(spec, path)
            # A version-1 record of an ongoing game replays through step() to different engine moves
            # (and scores) while both outcomes stay ongoing; the replay check must refuse it.
            ongoing = spec.create_game("python", seed=3)
            try:
                for action in (0, 0):
                    spec.step(ongoing, {spec.actors(ongoing)[0]: action})
                self.assertFalse(ongoing.outcome.terminal)
                self.assertEqual(list(ongoing.moves), [1, 1])
                v1 = Path(directory) / "ongoing.json"
                save_record(spec, ongoing, v1)
            finally:
                ongoing.close()
            self.assertEqual(json.loads(v1.read_text())["version"], 1)
            with self.assertRaisesRegex(ValueError, "history_actions"):
                load_record(spec, v1)

    def test_batch_runner_copies_observations_from_a_reused_encoder_buffer(self):
        """The batch runner keeps one observation per (game, player); an encoder that fills and returns
        the same buffer must not turn every stored sample into the last player's observation."""
        from hasami_learn.parallel import selfplay_batches

        spec = HiddenSpec()

        class ReusingEncoder(ToyEncoder):
            def __init__(self, spec):
                super().__init__(spec)
                self.buffer = np.zeros((2, 1, 3), dtype=np.float32)

            def encode(self, game, player=None):
                self.buffer[...] = super().encode(game, player)
                return self.buffer

        class Search:
            config = SimpleNamespace(simulations=1)

            def run_many(self, games, simulations, players):
                return [spec.legal_mask(game, p) for game, p in zip(games, players)]

        def decision(result, mask, ply):
            return 0, mask / mask.sum(), {}

        trajectories = next(selfplay_batches(Search(), ReusingEncoder(spec), decision, games=2, batch_games=2, backend="python",
                                             max_plies=6, draw_value=0.0, draw_reasons=["finished"], spec=spec, seed=21))
        for trajectory in trajectories:
            self.assertEqual(trajectory.players, [0, 1] * 3)
            self.assertEqual([int(planes[0, 0, 2]) for planes in trajectory.planes], trajectory.players)
            self.assertEqual([int(planes[0, 0, 1]) for planes in trajectory.planes], [0, 0, 1, 1, 2, 2])

    def _reusing_encoder(self, spec):
        class ReusingEncoder(ToyEncoder):
            def __init__(self, spec):
                super().__init__(spec)
                self.buffer = np.zeros((2, 1, 3), dtype=np.float32)
                self.mask_buffer = np.zeros(2, dtype=np.float32)

            def encode(self, game, player=None):
                self.buffer[...] = super().encode(game, player)
                return self.buffer

            def legal_mask(self, game, player=None):
                self.mask_buffer[...] = super().legal_mask(game, player)
                return self.mask_buffer

        return ReusingEncoder(spec)

    def test_serial_runner_snapshots_observations_before_a_searching_actor(self):
        """A searching actor may advance its clone and re-encode it through the shared encoder; the
        trajectory must keep the observation of the decision, not of the actor's search leaf."""
        from hasami_learn.selfplay import run_selfplay_game

        spec = HiddenSpec()
        encoder = self._reusing_encoder(spec)

        def actor(game, planes, mask, ply, player=None):
            spec.step(game, {p: 0 for p in spec.actors(game)})  # search: advance the supplied clone
            encoder.encode(game, player)  # ...and re-encode it, overwriting a reusable buffer
            return 0, mask / mask.sum(), {}

        trajectory = run_selfplay_game(actor, encoder, spec=spec, seed=21)
        self.assertEqual(trajectory.players, [0, 1] * 3)
        self.assertEqual([int(planes[0, 0, 1]) for planes in trajectory.planes], [0, 0, 1, 1, 2, 2])
        self.assertEqual([int(planes[0, 0, 2]) for planes in trajectory.planes], trajectory.players)

    def test_datastore_replay_copies_observations_from_a_reused_encoder_buffer(self):
        from hasami_learn.datastore import SelfplayStore
        from hasami_learn.selfplay import run_selfplay_game

        spec = HiddenSpec()

        def actor(game, planes, mask, ply):
            return 0, mask / mask.sum(), {}

        with tempfile.TemporaryDirectory() as directory:
            trajectory = run_selfplay_game(actor, spec.make_encoder(), spec=spec, seed=21)
            store = SelfplayStore(Path(directory), spec=spec)
            store.add(trajectory, 1)
            data = store.to_dataset(self._reusing_encoder(spec))
            np.testing.assert_array_equal(data["planes"], np.stack(trajectory.planes))
            np.testing.assert_array_equal(data["mask"], np.stack(trajectory.masks))
            self.assertEqual([int(p) for p in data["planes"][:, 0, 0, 2]], [0, 1, 0, 1, 0, 1])

    def test_record_sample_extraction_yields_independent_arrays(self):
        from hasami_learn.datasets import iter_record_samples
        from hasami_learn.selfplay import run_selfplay_game

        spec = HiddenSpec()

        def actor(game, planes, mask, ply):
            return 0, mask / mask.sum(), {}

        trajectory = run_selfplay_game(actor, spec.make_encoder(), spec=spec, seed=21)
        game = spec.create_game("python", seed=21)
        try:
            for joint in trajectory.joint_actions:
                spec.step(game, {int(k): v for k, v in joint.items()})
            samples = list(iter_record_samples(game, self._reusing_encoder(spec), spec=spec))
        finally:
            game.close()
        np.testing.assert_array_equal(np.stack([s[0] for s in samples]), np.stack(trajectory.planes))
        np.testing.assert_array_equal(np.stack([s[1] for s in samples]), np.stack(trajectory.masks))

    def test_chance_games_without_a_recoverable_seed_are_refused(self):
        """A hidden-seed live game cannot be replayed from a record without its seed: saving and
        extraction refuse instead of silently regenerating different private observations."""
        import json
        from types import SimpleNamespace

        from hasami_learn.datasets import iter_record_samples
        from hasami_learn.games.records import load_record, save_record

        spec = ScalarHiddenSpec()  # hides its seed; has_chance defaults to True
        encoder = spec.make_encoder("")
        with tempfile.TemporaryDirectory() as directory:
            game = spec.create_game("python", seed=3)
            try:
                spec.step(game, {0: 1})
                with self.assertRaisesRegex(ValueError, "seed"):
                    save_record(spec, game, Path(directory) / "unsaveable.json")
                with self.assertRaisesRegex(ValueError, "seed"):
                    list(iter_record_samples(game, encoder, spec=spec))
                path = Path(directory) / "seeded.json"
                save_record(spec, game, path, trajectory=SimpleNamespace(joint_actions=[{0: 1}], rewards=None, truncated=False, seed=3))
                loaded = load_record(spec, path)
                self.assertEqual(list(loaded.cards), list(game.cards))
                self.assertEqual([s[2] for s in iter_record_samples(loaded, encoder, spec=spec)], [1])
                stripped = Path(directory) / "stripped.json"
                stripped.write_text(json.dumps(dict(json.loads(path.read_text()), seed=None)))
                with self.assertRaisesRegex(ValueError, "seed"):
                    load_record(spec, stripped)
            finally:
                game.close()

            class Deterministic(ScalarHiddenSpec):
                has_chance = False

            spec = Deterministic()
            game = spec.create_game("python", seed=None)
            try:
                spec.step(game, {0: 0})
                path = Path(directory) / "deterministic.json"
                save_record(spec, game, path)
                self.assertIsNone(json.loads(path.read_text())["seed"])
                self.assertEqual(list(load_record(spec, path).moves), [0])
            finally:
                game.close()

    def test_masks_are_copied_before_encoding_queries_other_seats(self):
        """An adapter may reuse its legal-mask buffer and read every seat's mask while encoding public
        features; the mask handed to the actor and stored must be the one fetched for that seat."""
        from hasami_learn.datastore import SelfplayStore
        from hasami_learn.selfplay import run_selfplay_game

        class MaskReusingSpec(HiddenSpec):
            game_id = "mask-reusing-test"

            def __init__(self):
                self.mask_buffer = np.zeros(2, dtype=np.float32)

            def legal_actions(self, game, player):
                return [player] if player in self.actors(game) else []

            def legal_mask(self, game, player):
                self.mask_buffer[...] = 0
                self.mask_buffer[self.legal_actions(game, player)] = 1
                return self.mask_buffer

            def observe(self, game, player):
                counts = [float(self.legal_mask(game, p).sum()) for p in range(self.num_players)]  # public feature
                return np.array([[[game.cards[player], len(game.moves), player]], [[counts[0], counts[1], 0]]], dtype=np.float32)

        spec = MaskReusingSpec()

        def actor(game, planes, mask, ply, player=None):
            return int(np.flatnonzero(mask)[0]), mask / mask.sum(), {}

        trajectory = run_selfplay_game(actor, spec.make_encoder(), spec=spec, seed=21)
        self.assertEqual(trajectory.players, [0, 1] * 3)
        self.assertEqual(trajectory.actions, [0, 1] * 3)
        for mask, player in zip(trajectory.masks, trajectory.players):
            self.assertEqual(list(mask), [1.0, 0.0] if player == 0 else [0.0, 1.0])
        with tempfile.TemporaryDirectory() as directory:
            store = SelfplayStore(Path(directory), spec=spec)
            store.add(trajectory, 1)
            data = store.to_dataset(spec.make_encoder())
            np.testing.assert_array_equal(data["mask"], np.stack(trajectory.masks))
            self.assertFalse(np.any(data["policy"][data["mask"] == 0]))

    def test_inference_agents_copy_the_observation_before_the_mask_is_computed(self):
        """An adapter may reuse its observation buffer and read every seat's observation while building
        the legal mask; the network must still receive the requested player's observation."""
        import torch
        from hasami.agents import SearchLimits
        from hasami_learn.inference import PolicyAgent, QAgent

        class ObserveReusingSpec(HiddenSpec):
            game_id = "observe-reusing-test"

            def __init__(self):
                self.buffer = np.zeros((2, 1, 3), dtype=np.float32)

            def observe(self, game, player):
                self.buffer[...] = super().observe(game, player)
                return self.buffer

            def legal_mask(self, game, player):
                for p in range(self.num_players):  # public feature derived from every seat's observation
                    self.observe(game, p)
                return super().legal_mask(game, player)

        class SeatNetwork(torch.nn.Module):
            """Logits prefer action == the player index encoded in the observation it receives."""

            def forward(self, x):
                seat = x[:, 0, 0, 2]
                logits = torch.stack([1 - seat, seat], dim=1)
                return logits, torch.zeros(x.shape[0])

        spec = ObserveReusingSpec()
        game = spec.create_game("python", seed=4)
        try:
            limits = SearchLimits(1, 100, 10)
            for agent in (PolicyAgent(SeatNetwork(), spec.make_encoder(), "cpu", spec=spec),
                          QAgent(SeatNetwork(), spec.make_encoder(), "cpu", spec=spec)):
                for player in (0, 1):
                    self.assertEqual(agent.choose(game, limits, player=player).action, player)
                    out, _, mask = agent.evaluate(game, player)
                    self.assertEqual(int(np.argmax(out)), player)
                    self.assertEqual(list(mask), [1.0, 1.0])
        finally:
            game.close()
