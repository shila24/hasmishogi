"""Game adapters preserve Hasami contracts and support independent toy engines."""

import tempfile
import unittest
from types import SimpleNamespace
from ._common import TORCH_AVAILABLE


@unittest.skipUnless(TORCH_AVAILABLE, "torch is optional")
class GameSpecTests(unittest.TestCase):
    def test_registry_and_hasami_symmetry(self):
        import numpy as np
        from hasami_learn.games import GameSpec, get_spec
        from hasami_learn.registry import GAMES
        spec = get_spec()
        self.assertIn("hasami", GAMES)
        self.assertIsInstance(spec, GameSpec)
        encoder = spec.make_encoder("")
        game = spec.create_game()
        try:
            self.assertEqual(len(game.legal_actions()), 63)
            planes = encoder.encode(game)
            self.assertEqual(planes.shape, (encoder.num_planes, *spec.board_shape))
            policy = np.arange(spec.action_size)
            for sym in spec.symmetries():
                np.testing.assert_array_equal(sym.planes(sym.planes(planes)), planes)
                np.testing.assert_array_equal(sym.policy(sym.policy(policy)), policy)
                for action in range(spec.action_size):
                    self.assertEqual(sym.action(sym.action(action)), action)
        finally:
            game.close()

    def test_toy_training_and_checkpoint(self):
        import copy
        import numpy as np
        from hasami_learn.games import get_spec
        from hasami_learn.registry import GAMES
        from hasami_learn.config import tiny_config, NamedConfig, TrainingConfig
        from hasami_learn.trainer import Trainer
        from hasami_learn.manifest import load_checkpoint, ManifestError
        from hasami_learn.replay import fields_for
        from hasami_learn.inference import create_agent
        from hasami.agents import Decision, SearchLimits

        class ToyGame:
            def __init__(self):
                self.moves = []
                self.initial = {"turn": 0}
                self.rules = {}

            @property
            def position(self):
                return SimpleNamespace(turn=len(self.moves) % 2)

            @property
            def outcome(self):
                done = len(self.moves) == 3
                return SimpleNamespace(terminal=done, winner=0 if done else -2,
                                       reason="win" if done else "ongoing")

            def legal_actions(self):
                return [] if self.outcome.terminal else [i for i in range(16) if i not in self.moves]

            def play(self, action):
                if type(action) is not int or action not in self.legal_actions():
                    raise ValueError("illegal toy action")
                self.moves.append(action)

            def undo(self):
                self.moves.pop()

            def clone(self):
                return copy.deepcopy(self)

            def history_keys(self):
                return [tuple(self.moves[:i]) for i in range(len(self.moves) + 1)]

            def close(self):
                pass

        class ToyEncoder:
            encoding_id = "toy-enc"
            num_planes = 2

            def encode(self, game):
                planes = np.zeros((2, 4, 4), dtype=np.float32)
                planes[0].flat[game.moves] = 1
                planes[1] = game.position.turn
                return planes

            def legal_mask(self, game):
                mask = np.zeros(16, dtype=np.float32)
                mask[game.legal_actions()] = 1
                return mask

        class ToyEnv:
            def __init__(self, max_plies):
                self.game = ToyGame()
                self.max_plies = max_plies
                self.truncated = False

            def step(self, action):
                self.game.play(action)
                done = self.game.outcome.terminal
                self.truncated = not done and len(self.game.moves) >= self.max_plies
                return SimpleNamespace(rewards=(1., -1.) if done else (0., 0.),
                                       terminated=done, truncated=self.truncated,
                                       info={"reason": "max_plies" if self.truncated else self.game.outcome.reason})

            def close(self):
                self.game.close()

        class ToySpec:
            game_id = "toy-test"
            ruleset_id = "toy-v1"
            board_shape = (4, 4)
            action_size = 16
            default_max_plies = 5

            def create_game(self, backend="python", rules=None, position=None):
                return ToyGame()

            def resolve_backend(self, spec):
                return "python"

            def rules_dict(self, game):
                return {}

            def make_encoder(self, name="", **params):
                if name not in ("", "toy-enc"):
                    raise ValueError(name)
                return ToyEncoder()

            def terminal_value_for_side_to_move(self, game):
                return 1. if game.position.turn == game.outcome.winner else -1.

            def symmetries(self):
                return []

            def baseline_agent(self, spec, seed=0):
                class FirstAgent:
                    def choose(self, game, limits, cancel=None):
                        return Decision(game.legal_actions()[0], 0, 0, 0, 0, True)
                return FirstAgent(), None

            def action_name(self, action):
                return str(action)

            def env(self, backend, max_plies, rules=None):
                return ToyEnv(max_plies)

        GAMES.register("toy-test", ToySpec)
        self.addCleanup(GAMES._items.pop, "toy-test")
        spec = get_spec("toy-test")
        defaults = TrainingConfig(game=NamedConfig("toy-test"))
        self.assertEqual(defaults.selfplay.max_plies, spec.default_max_plies)
        self.assertEqual(defaults.evaluation.max_plies, spec.default_max_plies)
        self.assertEqual(fields_for(spec, 2)["planes"][0], (2, 4, 4))
        self.assertEqual(fields_for(spec, 2, "q")["next_mask"][0], (16,))
        with tempfile.TemporaryDirectory() as directory:
            cfg = tiny_config("reinforce")
            cfg.game = NamedConfig("toy-test")
            cfg.encoder = NamedConfig("")
            cfg.run.output_dir = directory
            cfg.selfplay.symmetry_augmentation = True
            trainer = Trainer(cfg, echo=False)
            try:
                self.assertEqual(trainer.network.board_shape, (4, 4))
                checkpoint = trainer.run()
                self.assertEqual(trainer.iteration, 1)
                self.assertIs(trainer.learner.spec, trainer.spec)
                loaded = load_checkpoint(checkpoint)
                self.assertEqual(loaded.manifest.game_id, "toy-test")
                self.assertEqual(tuple(loaded.manifest.board_shape), (4, 4))
                with self.assertRaises(ManifestError):
                    load_checkpoint(checkpoint, expect_game_id="hasami")
                agent = create_agent(str(checkpoint), device="cpu")
                game = spec.create_game()
                self.assertIn(agent.choose(game, SearchLimits()).action, game.legal_actions())
            finally:
                trainer.close()

            from hasami_learn.evaluation import play_matches
            from hasami_learn.ladder import run_ladder
            from hasami_learn.parallel import _run_worker
            from hasami_learn.datastore import SelfplayStore
            for algorithm in ("alphazero", "dqn", "ppo", "supervised", "adversarial"):
                with self.subTest(algorithm=algorithm):
                    cfg = tiny_config(algorithm)
                    cfg.game = NamedConfig("toy-test")
                    cfg.encoder = NamedConfig("")
                    cfg.run.output_dir = directory
                    cfg.selfplay.max_plies = 4
                    cfg.replay.min_size = 1
                    if algorithm == "alphazero":
                        cfg.algorithm.params = {"simulations": 2, "batch_games": 2}
                        cfg.datastore.enabled = True
                    if algorithm == "adversarial":
                        cfg.algorithm.params = {"snapshot_every": 1, "exploiter_every": 1,
                                                "exploiter_games": 1, "exploiter_iterations": 1}
                    if algorithm == "supervised":
                        cfg.algorithm.params = {"generate_games": 2, "generator": "first"}
                    trainer = Trainer(cfg, echo=False)
                    try:
                        checkpoint = trainer.run()
                        agent = create_agent(str(checkpoint), device="cpu")
                        opponent, _ = spec.baseline_agent("first")
                        result = play_matches(agent, opponent, 2, spec=spec,
                                              limits=SearchLimits(time_ms=1000, node_limit=2))
                        self.assertEqual(result.games, 2)
                        self.assertEqual(result.plies, 6)
                        if algorithm == "alphazero":
                            job = trainer.learner._selfplay_job(1)
                            job.update(seed=1, games=1, device_spec="cpu")
                            trajectories = _run_worker(job)
                            self.assertEqual(trajectories[0].game_id, "toy-test")
                            store = trainer.learner.store
                            self.assertEqual(store.to_dataset(trainer.encoder)["planes"].shape[1:], (2, 4, 4))
                            self.assertTrue(all(row["game_id"] == "toy-test" for row in store._entries))
                            with self.assertRaises(ValueError):
                                SelfplayStore(store.root)
                    finally:
                        trainer.close()
            report = run_ladder(["a=first", "b=first"], games=2, backend="python",
                                game_id="toy-test", out=directory + "/ladder")
            self.assertEqual(report["config"]["game_id"], "toy-test")
            self.assertTrue(report["complete"])

    def test_network_shapes_and_legacy_checkpoint(self):
        import json
        import torch
        from pathlib import Path
        from hasami_learn.games import get_spec
        from hasami_learn.registry import NETWORKS
        from hasami_learn import networks
        from hasami_learn.manifest import ModelManifest, save_checkpoint, load_checkpoint, ManifestError
        for name in ("mlp", "resnet"):
            for head in ("policy_value", "q"):
                with self.subTest(network=name, head=head):
                    params = {"hidden": 8}
                    if name == "resnet":
                        params.update(channels=4, blocks=1, head_channels=2)
                    network = NETWORKS.get(name)(head=head, num_planes=2, board_shape=(3, 4),
                                                action_size=17, **params)
                    logits, value = network(torch.zeros(2, 2, 3, 4))
                    self.assertEqual(tuple(logits.shape), (2, 17))
                    if head == "q":
                        self.assertIsNone(value)
                    else:
                        self.assertEqual(tuple(value.shape), (2,))
        spec = get_spec()
        encoder = spec.make_encoder("")
        network = NETWORKS.get("mlp")(hidden=8, layers=1)
        manifest = ModelManifest("reinforce", {"name": "mlp", "params": {"hidden": 8, "layers": 1},
                                              "head": "policy_value"}, encoder.encoding_id, encoder.num_planes)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_checkpoint(Path(directory) / "model", network, manifest)
            path = checkpoint / "manifest.json"
            original = json.loads(path.read_text())
            legacy = dict(original)
            legacy.pop("game_id")
            legacy.pop("board_shape")
            legacy.pop("game_params")
            path.write_text(json.dumps(legacy))
            self.assertEqual(load_checkpoint(checkpoint).manifest.game_id, "hasami")
            for key, bad_value in (("game_id", "unknown-game"), ("board_shape", [4, 4]),
                                   ("action_size", 16)):
                with self.subTest(field=key):
                    invalid = dict(original)
                    invalid[key] = bad_value
                    path.write_text(json.dumps(invalid))
                    with self.assertRaises(ManifestError):
                        load_checkpoint(checkpoint)

    def test_no_hasami_dimensions_outside_adapter(self):
        import ast
        from pathlib import Path
        root = Path(__file__).resolve().parents[2] / "hasami_learn"
        violations = []
        for path in root.rglob("*.py"):
            if path == root / "games" / "hasami.py":
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Constant) and type(node.value) is int and node.value in (9, 81, 6561):
                    violations.append(f"{path.relative_to(root)}:{node.lineno}")
        self.assertEqual(violations, [])


@unittest.skipUnless(TORCH_AVAILABLE, "torch is optional")
class ReviewRegressionTests(unittest.TestCase):
    """Regressions for the game adapters review findings."""

    def test_manifest_persists_encoder_params_and_loader_uses_them(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from hasami_learn.config import tiny_config
        from hasami_learn.games import get_spec
        from hasami_learn.manifest import load_checkpoint
        from hasami_learn.trainer import Trainer, train_from_config

        config = tiny_config("reinforce", run_id="encoder-params")
        with tempfile.TemporaryDirectory() as directory:
            config.run.output_dir = str(Path(directory) / "runs")
            trainer = Trainer(config, echo=False)
            try:
                manifest = trainer.manifest(1)
            finally:
                trainer.close()
            self.assertEqual(manifest.encoder_params, dict(config.encoder.params))
            self.assertIn("encoder_params", manifest.to_dict())
            final = train_from_config(config, echo=False)
            real = get_spec("hasami", {})
            calls = []

            class Recording:
                def __getattr__(self, name):
                    return getattr(real, name)

                def make_encoder(self, name="", **params):
                    calls.append(dict(params))
                    return real.make_encoder(name, **params)

            with patch("hasami_learn.manifest.get_spec", return_value=Recording()):
                loaded = load_checkpoint(final, "cpu")
            self.assertEqual(calls, [dict(config.encoder.params)])
            self.assertEqual(loaded.manifest.encoder_params, dict(config.encoder.params))

    def test_dataset_meta_persists_game_params(self):
        import tempfile
        from pathlib import Path

        from hasami_learn.datasets import build_dataset, generate_games, load_dataset, save_dataset
        from hasami_learn.games import get_spec

        spec = get_spec("hasami", {})
        with tempfile.TemporaryDirectory() as directory:
            paths = generate_games(Path(directory) / "games", 1, generator="random", backend="python", spec=spec, max_plies=4, seed=2, random_opening_plies=1)
            data = build_dataset(paths, spec.make_encoder(""), "python", False, spec=spec, game_params={})
            self.assertEqual(data["meta"]["game_params"], {})
            saved = save_dataset(Path(directory) / "d.npz", data, spec=spec)
            self.assertEqual(load_dataset(saved)["meta"]["game_params"], {})

    def test_split_groups_of_any_size_never_leak_variants(self):
        import numpy as np
        from hasami_learn.algorithms.supervised import split_augmentation_groups

        rng = np.random.default_rng(3)
        for group_size, groups in ((1, 7), (3, 5), (8, 4)):
            count = group_size * groups
            train, val = split_augmentation_groups(count, group_size, 0.25, rng)
            self.assertEqual(sorted(train.tolist() + val.tolist()), list(range(count)))
            self.assertTrue(len(val) % group_size == 0 and len(train) % group_size == 0)
            self.assertEqual({int(i) // group_size for i in val}.isdisjoint({int(i) // group_size for i in train}), True)
        with self.assertRaises(ValueError):
            split_augmentation_groups(7, 3, 0.2, rng)
        train, val = split_augmentation_groups(1, 1, 0.5, rng)
        self.assertEqual((len(train), len(val)), (1, 0))

    def test_dataset_meta_records_group_size(self):
        import tempfile
        from pathlib import Path

        from hasami_learn.datasets import build_dataset, generate_games
        from hasami_learn.games import get_spec

        spec = get_spec("hasami", {})
        with tempfile.TemporaryDirectory() as directory:
            paths = generate_games(Path(directory) / "games", 1, generator="random", backend="python", spec=spec, max_plies=6, seed=1, random_opening_plies=2)
            data = build_dataset(paths, spec.make_encoder(""), "python", True, spec=spec)
            self.assertEqual(data["meta"]["group_size"], 1 + len(spec.symmetries()))
            self.assertEqual(len(data["value"]) % data["meta"]["group_size"], 0)
            plain = build_dataset(paths, spec.make_encoder(""), "python", False, spec=spec)
            self.assertEqual(plain["meta"]["group_size"], 1)

    def test_json_config_may_omit_max_plies(self):
        from hasami_learn.config import TrainingConfig, tiny_config

        data = tiny_config("reinforce").to_dict()
        del data["selfplay"]["max_plies"]
        data["evaluation"]["max_plies"] = None
        config = TrainingConfig.from_dict(data)
        self.assertEqual(config.selfplay.max_plies, 300)
        self.assertEqual(config.evaluation.max_plies, 300)
        data["selfplay"]["max_plies"] = 0
        self.assertEqual(TrainingConfig.from_dict(data).selfplay.max_plies, 0)

    def test_ladder_resume_accepts_legacy_config_without_game_identity(self):
        import json
        import tempfile
        from pathlib import Path

        from hasami_learn.ladder import run_ladder

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "ladder"
            run_ladder(["random", "greedy"], games=2, backend="python", max_plies=10, time_ms=50, simulations=None, device="cpu", seed=0, out=out)
            saved = json.loads((out / "ladder.json").read_text(encoding="utf-8"))
            saved["config"].pop("game_id", None)
            saved["config"].pop("game_params", None)
            (out / "ladder.json").write_text(json.dumps(saved), encoding="utf-8")
            resumed = run_ladder(["random", "greedy"], games=2, backend="python", max_plies=10, time_ms=50, simulations=None, device="cpu", seed=0, out=out, resume=True)
            self.assertIsNotNone(resumed)

    def test_adversarial_snapshot_manifest_keeps_encoder_params(self):
        import tempfile
        from pathlib import Path

        from hasami_learn.config import tiny_config
        from hasami_learn.trainer import Trainer

        config = tiny_config("adversarial", run_id="adv-encoder-params", inner_algorithm="reinforce", snapshot_every=1, exploiter_every=0)
        with tempfile.TemporaryDirectory() as directory:
            config.run.output_dir = str(Path(directory) / "runs")
            trainer = Trainer(config, echo=False)
            try:
                manifest = trainer.learner._manifest(trainer.learner.inner, 1)
            finally:
                trainer.close()
            self.assertEqual(manifest.encoder_params, dict(config.encoder.params))
            self.assertEqual(manifest.encoding_id, trainer.encoder.encoding_id)

    def test_dataset_with_other_game_params_is_rejected(self):
        import tempfile
        from pathlib import Path

        from hasami_learn.datasets import build_dataset, generate_games, load_dataset, save_dataset, validate_dataset
        from hasami_learn.games import get_spec

        spec = get_spec("hasami", {})
        with tempfile.TemporaryDirectory() as directory:
            paths = generate_games(Path(directory) / "games", 1, generator="random", backend="python", spec=spec, max_plies=4, seed=5, random_opening_plies=1)
            data = build_dataset(paths, spec.make_encoder(""), "python", False, spec=spec, game_params={"variant": "other"})
            validate_dataset(data, spec=spec)  # no configured params: accepted
            validate_dataset(data, spec=spec, game_params={"variant": "other"})
            with self.assertRaises(ValueError):
                validate_dataset(data, spec=spec, game_params={})
            saved = save_dataset(Path(directory) / "d.npz", data, spec=spec)
            with self.assertRaises(ValueError):
                load_dataset(saved, spec=spec, game_params={"variant": "mine"})

    def test_datastore_rejects_archives_from_another_variant(self):
        import tempfile
        from pathlib import Path

        from hasami_learn.datastore import SelfplayStore
        from hasami_learn.games import get_spec
        from hasami_learn.selfplay import run_selfplay_game

        spec = get_spec("hasami", {})
        encoder = spec.make_encoder("")

        def actor(game, planes, mask, ply):
            import numpy as np
            legal = np.flatnonzero(mask > 0)
            policy = np.zeros(mask.shape[0], dtype=np.float32)
            policy[legal] = 1.0 / len(legal)
            return int(legal[0]), policy, {"decision_ply": ply}

        with tempfile.TemporaryDirectory() as directory:
            store = SelfplayStore(Path(directory), max_games=10, max_bytes=10 ** 7, spec=spec, game_params={"variant": "a"})
            trajectory = run_selfplay_game(actor, encoder, backend="python", max_plies=6)
            written = store.add(trajectory, 1)
            self.assertIsNotNone(written)
            self.assertEqual(store.stats()["games"], 1)
            same = SelfplayStore(Path(directory), max_games=10, max_bytes=10 ** 7, spec=spec, game_params={"variant": "a"})
            self.assertEqual(same.stats()["games"], 1)
            with self.assertRaises(ValueError):
                SelfplayStore(Path(directory), max_games=10, max_bytes=10 ** 7, spec=spec, game_params={})

    def test_dataset_encoder_params_are_persisted_and_checked(self):
        import tempfile
        from pathlib import Path

        from hasami_learn.datasets import build_dataset, generate_games, load_dataset, save_dataset, validate_dataset
        from hasami_learn.games import get_spec

        spec = get_spec("hasami", {})
        with tempfile.TemporaryDirectory() as directory:
            paths = generate_games(Path(directory) / "games", 1, generator="random", backend="python", spec=spec, max_plies=4, seed=7, random_opening_plies=1)
            data = build_dataset(paths, spec.make_encoder(""), "python", False, spec=spec, encoder_params={"scale": 1})
            self.assertEqual(data["meta"]["encoder_params"], {"scale": 1})
            validate_dataset(data, spec=spec, encoder_params={"scale": 1})
            with self.assertRaises(ValueError):
                validate_dataset(data, spec=spec, encoder_params={"scale": 2})
            saved = save_dataset(Path(directory) / "d.npz", data, spec=spec)
            with self.assertRaises(ValueError):
                load_dataset(saved, spec=spec, encoder_params={})
            self.assertEqual(load_dataset(saved, spec=spec, encoder_params={"scale": 1})["meta"]["encoder_params"], {"scale": 1})

    def test_resume_accepts_spelled_out_default_encoder(self):
        import argparse
        import tempfile
        from pathlib import Path

        from hasami_learn.cli import resolve_train_request
        from hasami_learn.config import tiny_config
        from hasami_learn.trainer import train_from_config

        with tempfile.TemporaryDirectory() as directory:
            config = tiny_config("reinforce", run_id="tiny-reinforce")
            config.run.output_dir = str(Path(directory) / "runs")
            config.encoder.name = "hasami-enc-v1"  # older saved configs spelled the default encoder out
            train_from_config(config, echo=False)
            args = argparse.Namespace(config=None, algorithm="reinforce", tiny=True, resume=str(Path(directory) / "runs" / "tiny-reinforce"),
                                      set=[], device=None, iterations=2, run_id=None, output_dir=str(Path(directory) / "runs"), seed=None, backend=None, games=None, quiet=True)
            resolved, resume = resolve_train_request(args)
            self.assertIsNotNone(resume)
            self.assertEqual(resolved.encoder.name, "hasami-enc-v1")
            self.assertEqual(resolved.run.iterations, 2)

    def test_eval_refuses_checkpoints_from_different_variants(self):
        from hasami_learn.cli import check_compatible_manifests
        from hasami_learn.config import ConfigError
        from hasami_learn.manifest import ModelManifest

        base = dict(algorithm="reinforce", network={"name": "mlp", "params": {}, "head": "policy_value"}, encoding_id="hasami-enc-v1", num_planes=9)
        primary = ModelManifest(**base)
        same = ModelManifest(**base)
        check_compatible_manifests(primary, same)
        other = ModelManifest(**base, game_params={"variant": "b"})
        with self.assertRaises(ConfigError):
            check_compatible_manifests(primary, other)
        rules = ModelManifest(**base)
        rules.rules = dict(rules.rules, edge_capture=False)
        with self.assertRaises(ConfigError):
            check_compatible_manifests(primary, rules)

    def test_legacy_dataset_metadata_means_default_parameters(self):
        import numpy as np
        import tempfile
        from pathlib import Path

        from hasami_learn.datasets import build_dataset, generate_games, validate_dataset
        from hasami_learn.games import get_spec

        spec = get_spec("hasami", {})
        with tempfile.TemporaryDirectory() as directory:
            paths = generate_games(Path(directory) / "games", 1, generator="random", backend="python", spec=spec, max_plies=4, seed=9, random_opening_plies=1)
            data = build_dataset(paths, spec.make_encoder(""), "python", False, spec=spec)
            del data["meta"]["game_params"]
            del data["meta"]["encoder_params"]
            validate_dataset(data, spec=spec, game_params={}, encoder_params={})
            with self.assertRaises(ValueError):
                validate_dataset(data, spec=spec, game_params={"variant": "other"})
            with self.assertRaises(ValueError):
                validate_dataset(data, spec=spec, encoder_params={"scale": 2})
            data["meta"]["game_params"] = "malformed"
            with self.assertRaises(ValueError):
                validate_dataset(data, spec=spec, game_params={})

    def test_manifest_must_match_selected_game_variant(self):
        from hasami_learn.games import get_spec
        from hasami_learn.manifest import ManifestError, ModelManifest, check_manifest_matches_spec

        spec = get_spec("hasami", {})
        base = dict(algorithm="reinforce", network={"name": "mlp", "params": {}, "head": "policy_value"}, encoding_id="hasami-enc-v1", num_planes=9)
        check_manifest_matches_spec(ModelManifest(**base), spec, {})
        with self.assertRaises(ManifestError):
            check_manifest_matches_spec(ModelManifest(**base, game_params={"size": 5}), spec, {})
        wrong_shape = ModelManifest(**base)
        wrong_shape.board_shape = [5, 5]
        with self.assertRaises(ManifestError):
            check_manifest_matches_spec(wrong_shape, spec, {})
        wrong_actions = ModelManifest(**base)
        wrong_actions.action_size = 625
        with self.assertRaises(ManifestError):
            check_manifest_matches_spec(wrong_actions, spec, {})

    def test_config_accepts_game_specific_draw_reasons(self):
        from hasami_learn.config import TrainingConfig, tiny_config

        data = tiny_config("reinforce").to_dict()
        data["selfplay"]["draw_reasons"] = ["board_full", "stalemate"]
        config = TrainingConfig.from_dict(data)
        self.assertEqual(config.selfplay.draw_reasons, ["board_full", "stalemate"])
        data["selfplay"]["draw_reasons"] = [""]
        with self.assertRaises(Exception):
            TrainingConfig.from_dict(data)

    def test_opponent_pool_rejects_checkpoints_from_other_variants(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        import numpy as np
        from hasami_learn.config import OpponentsConfig
        from hasami_learn.games import get_spec
        from hasami_learn.manifest import ManifestError, ModelManifest
        from hasami_learn.opponents import OpponentPool, PoolEntry

        spec = get_spec("hasami", {})
        base = dict(algorithm="reinforce", network={"name": "mlp", "params": {}, "head": "policy_value"}, encoding_id="hasami-enc-v1", num_planes=9)

        class Fake:
            def __init__(self, manifest):
                self.manifest = manifest

            def choose(self, game, limits, cancel=None):
                raise AssertionError("not used")

        with tempfile.TemporaryDirectory() as directory:
            pool = OpponentPool(OpponentsConfig(strategy="uniform"), "cpu", spec.make_encoder(""), np.random.default_rng(0), Path(directory), spec=spec, game_params={})
            entry = PoolEntry("pool:x", "checkpoint", str(Path(directory) / "missing"))
            with patch("hasami_learn.inference.create_agent", return_value=Fake(ModelManifest(**base, game_params={"size": 5}))):
                with self.assertRaises(ManifestError):
                    pool.agent_for(entry)
            self.assertNotIn(entry.entry_id, pool._agent_cache)
            with patch("hasami_learn.inference.create_agent", return_value=Fake(ModelManifest(**base))):
                self.assertIsNotNone(pool.agent_for(entry))

    def test_game_override_resolves_move_limits_from_the_new_game(self):
        from hasami_learn.config import TrainingConfig, tiny_config
        from hasami_learn.games.hasami import HasamiSpec
        from hasami_learn.registry import GAMES

        name = "review-short-game"
        if name not in GAMES:
            class ShortHasami(HasamiSpec):
                game_id = name
                default_max_plies = 5

            GAMES.register(name, ShortHasami)
        omitted = TrainingConfig()  # caps omitted: resolved from hasami's default
        self.assertEqual((omitted.selfplay.max_plies, omitted.evaluation.max_plies), (300, 300))
        self.assertEqual(omitted.defaulted_max_plies(), {"selfplay", "evaluation"})
        switched = omitted.with_overrides(game__name=name)
        self.assertEqual(switched.game.name, name)
        self.assertEqual((switched.selfplay.max_plies, switched.evaluation.max_plies), (5, 5))
        explicit_override = omitted.with_overrides(game__name=name, selfplay__max_plies=42)
        self.assertEqual((explicit_override.selfplay.max_plies, explicit_override.evaluation.max_plies), (42, 5))
        tiny = tiny_config("reinforce")  # explicit 40-ply caps survive a game switch
        self.assertEqual(tiny.defaulted_max_plies(), set())
        self.assertEqual(tiny.with_overrides(game__name=name).selfplay.max_plies, 40)
        equal_to_default = TrainingConfig.from_dict(TrainingConfig().to_dict(preserve_defaults=False))  # JSON with explicit 300 caps
        self.assertEqual(equal_to_default.defaulted_max_plies(), set())
        self.assertEqual((equal_to_default.with_overrides(game__name=name).selfplay.max_plies,
                          equal_to_default.with_overrides(game__name=name).evaluation.max_plies), (300, 300))

    def test_omitted_move_caps_survive_round_trips_and_the_cli_path(self):
        import argparse
        import tempfile
        from pathlib import Path

        from hasami_learn.cli import resolve_train_request
        from hasami_learn.config import TrainingConfig, tiny_config
        from hasami_learn.games.hasami import HasamiSpec
        from hasami_learn.registry import GAMES

        name = "review-short-game"
        if name not in GAMES:
            class ShortHasami(HasamiSpec):
                game_id = name
                default_max_plies = 5

            GAMES.register(name, ShortHasami)
        chained = TrainingConfig().with_overrides(run__iterations=2).with_overrides(device__device="cpu").with_overrides(game__name=name)
        self.assertEqual((chained.selfplay.max_plies, chained.evaluation.max_plies), (5, 5))
        self.assertIsNone(TrainingConfig().to_dict()["selfplay"]["max_plies"])
        self.assertEqual(TrainingConfig().to_dict(preserve_defaults=False)["selfplay"]["max_plies"], 300)
        self.assertEqual(tiny_config("reinforce").to_dict()["selfplay"]["max_plies"], 40)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c.json"
            TrainingConfig().save(path)
            reloaded = TrainingConfig.load(path)
            self.assertEqual(reloaded.defaulted_max_plies(), {"selfplay", "evaluation"})
            self.assertEqual(reloaded.with_overrides(game__name=name).selfplay.max_plies, 5)
        args = argparse.Namespace(config=None, algorithm="reinforce", tiny=False, resume=None, set=[f"game.name={name}"], device="cpu",
                                  iterations=1, run_id="cli-short", output_dir=str(Path(tempfile.gettempdir()) / "hasami-cli-short"), seed=None, backend="python", games=None, quiet=True)
        config, resume = resolve_train_request(args)
        self.assertIsNone(resume)
        self.assertEqual(config.game.name, name)
        self.assertEqual((config.selfplay.max_plies, config.evaluation.max_plies), (5, 5))

    def test_caps_assigned_after_construction_and_null_overrides(self):
        import argparse
        import tempfile
        from pathlib import Path

        from hasami_learn.cli import resolve_train_request
        from hasami_learn.config import USE_DEFAULT, TrainingConfig, tiny_config

        cfg = TrainingConfig()
        cfg.selfplay.max_plies = 12
        cfg.evaluation.max_plies = 24
        self.assertEqual(cfg.defaulted_max_plies(), set())
        self.assertEqual(cfg.to_dict()["selfplay"]["max_plies"], 12)
        rebuilt = TrainingConfig.from_dict(cfg.to_dict())
        self.assertEqual((rebuilt.selfplay.max_plies, rebuilt.evaluation.max_plies), (12, 24))
        unrelated = cfg.with_overrides(run__iterations=2)
        self.assertEqual((unrelated.selfplay.max_plies, unrelated.evaluation.max_plies), (12, 24))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c.json"
            cfg.save(path)
            self.assertEqual(TrainingConfig.load(path).evaluation.max_plies, 24)
        reset = tiny_config("reinforce").with_overrides(selfplay__max_plies=USE_DEFAULT)
        self.assertEqual(reset.selfplay.max_plies, 300)
        self.assertEqual(reset.evaluation.max_plies, 40)
        args = argparse.Namespace(config=None, algorithm="reinforce", tiny=True, resume=None, set=["selfplay.max_plies=null"], device="cpu",
                                  iterations=1, run_id="null-cap", output_dir=str(Path(tempfile.gettempdir()) / "hasami-null-cap"), seed=None, backend="python", games=None, quiet=True)
        config, _ = resolve_train_request(args)
        self.assertEqual(config.selfplay.max_plies, 300)
        self.assertEqual(config.evaluation.max_plies, 40)

    def test_explicit_assignment_equal_to_default_stays_explicit(self):
        from hasami_learn.config import TrainingConfig
        from hasami_learn.games.hasami import HasamiSpec
        from hasami_learn.registry import GAMES

        name = "review-short-game"
        if name not in GAMES:
            class ShortHasami(HasamiSpec):
                game_id = name
                default_max_plies = 5

            GAMES.register(name, ShortHasami)
        cfg = TrainingConfig()
        cfg.selfplay.max_plies = 300  # same number as the default, but explicitly assigned
        self.assertEqual(cfg.defaulted_max_plies(), {"evaluation"})
        self.assertEqual(cfg.to_dict()["selfplay"]["max_plies"], 300)
        self.assertIsNone(cfg.to_dict()["evaluation"]["max_plies"])
        switched = cfg.with_overrides(game__name=name)
        self.assertEqual((switched.selfplay.max_plies, switched.evaluation.max_plies), (300, 5))
        rebuilt = TrainingConfig.from_dict(cfg.to_dict())
        self.assertEqual(rebuilt.defaulted_max_plies(), {"evaluation"})
        cfg.evaluation.max_plies = None  # assigning None makes it game-derived again
        self.assertEqual(cfg.defaulted_max_plies(), {"evaluation"})

    def test_trainer_resolves_defaulted_caps_against_a_reassigned_game(self):
        import tempfile
        from pathlib import Path

        from hasami_learn.config import NamedConfig, TrainingConfig, tiny_config
        from hasami_learn.games.hasami import HasamiSpec
        from hasami_learn.registry import GAMES
        from hasami_learn.trainer import Trainer

        name = "review-short-game"
        if name not in GAMES:
            class ShortHasami(HasamiSpec):
                game_id = name
                default_max_plies = 5

            GAMES.register(name, ShortHasami)
        cfg = TrainingConfig()
        cfg.game = NamedConfig(name)  # reassigned after construction
        self.assertEqual(cfg.selfplay.max_plies, 300)  # stale until resolved
        self.assertEqual(cfg.resolve_defaults().selfplay.max_plies, 5)
        small = tiny_config("reinforce", run_id="reassigned-game")
        small.selfplay.max_plies = None
        small.evaluation.max_plies = None
        small.game = NamedConfig(name)
        with tempfile.TemporaryDirectory() as directory:
            small.run.output_dir = str(Path(directory) / "runs")
            trainer = Trainer(small, echo=False)
            try:
                self.assertEqual((trainer.config.selfplay.max_plies, trainer.config.evaluation.max_plies), (5, 5))
                saved = TrainingConfig.load(Path(directory) / "runs" / "reassigned-game" / "config.json")
                self.assertEqual((saved.selfplay.max_plies, saved.evaluation.max_plies), (5, 5))
            finally:
                trainer.close()

    def test_repeated_null_override_is_accepted_on_resume(self):
        import argparse
        import tempfile
        from pathlib import Path

        from hasami_learn.cli import resolve_train_request
        from hasami_learn.config import tiny_config
        from hasami_learn.trainer import train_from_config

        with tempfile.TemporaryDirectory() as directory:
            config = tiny_config("reinforce", run_id="null-resume").with_overrides(selfplay__max_plies=None)
            config.selfplay.max_plies = None  # game-derived cap, saved as null
            config.run.output_dir = str(Path(directory) / "runs")
            train_from_config(config, echo=False)
            args = argparse.Namespace(config=None, algorithm="reinforce", tiny=True, resume="auto", set=["selfplay.max_plies=null"], device=None,
                                      iterations=2, run_id="null-resume", output_dir=str(Path(directory) / "runs"), seed=None, backend=None, games=None, quiet=True)
            resolved, resume = resolve_train_request(args)
            self.assertIsNotNone(resume)
            self.assertEqual(resolved.selfplay.max_plies, 300)

