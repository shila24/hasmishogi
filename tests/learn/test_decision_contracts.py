"""Contracts prepared for batched collection: seat-signature caching, the shared decision rule, self-play agents."""

from __future__ import annotations

import functools
import unittest

from tests.learn._common import LearnTestCase, requires_torch


@requires_torch
class SeatSignatureCacheTests(unittest.TestCase):
    def test_cached_dispatch_matches_signatures(self):
        from hasami_learn import selfplay

        class Seated:
            def __call__(self, game, planes, mask, ply, player):
                return ("seated", player)

            def act(self, game, planes, mask, ply, player):
                return ("seated", player)

            def choose(self, game, limits, cancel=None, *, player=None):
                return ("seated", player)

        class Legacy:
            def act(self, game, planes, mask, ply):
                return ("legacy", None)

            def choose(self, game, limits, cancel=None):
                return ("legacy", None)

        def plain(game, planes, mask, ply):
            return ("plain", None)

        partial = functools.partial(lambda game, planes, mask, ply, player, tag: (tag, player), tag="partial")
        for _ in range(3):  # the second and third rounds are served from the cache
            self.assertEqual(selfplay._call_actor(Seated().act, None, None, None, 0, 2), ("seated", 2))
            self.assertEqual(selfplay._call_actor(Legacy().act, None, None, None, 0, 2), ("legacy", None))
            self.assertEqual(selfplay._call_actor(plain, None, None, None, 0, 2), ("plain", None))
            self.assertEqual(selfplay._call_actor(Seated(), None, None, None, 0, 3), ("seated", 3))
            self.assertEqual(selfplay._call_actor(partial, None, None, None, 0, 1), ("partial", 1))
            self.assertEqual(selfplay.choose_agent(Seated(), None, None, None, 1), ("seated", 1))
            self.assertEqual(selfplay.choose_agent(Legacy(), None, None, None, 1), ("legacy", None))

    def test_cache_hits_skip_inspection(self):
        from unittest import mock
        from hasami_learn import selfplay, signature_cache

        class Seated:
            def act(self, game, planes, mask, ply, player):
                return player

        def plain(game, planes, mask, ply):
            return "plain"

        calls = []
        real = signature_cache.inspect.signature
        with mock.patch.object(signature_cache.inspect, "signature",
                               side_effect=lambda target: (calls.append(target), real(target))[1]):
            for _ in range(3):
                self.assertEqual(selfplay._call_actor(Seated().act, None, None, None, 0, 2), 2)
                self.assertEqual(selfplay._call_actor(plain, None, None, None, 0, 2), "plain")
        self.assertEqual(len(calls), 2)  # one inspection per (function, owner class), then cache hits

    def test_closures_are_not_kept_alive(self):
        import gc
        import weakref
        from hasami_learn import selfplay

        class Payload:
            pass

        refs = []
        for _ in range(20):  # e.g. AlphaZero/supervised actor_fn() returns a new closure per game
            payload = Payload()

            def actor(game, planes, mask, ply, player, payload=payload):
                return player

            self.assertEqual(selfplay._call_actor(actor, None, None, None, 0, 1), 1)
            refs.append((weakref.ref(actor), weakref.ref(payload)))
            del actor, payload
        gc.collect()
        self.assertFalse(any(ref() is not None for pair in refs for ref in pair))

    def test_call_patterns_are_cached_separately(self):
        from hasami_learn import selfplay

        def shared(game, limits, cancel=None, *, player=None):
            return player

        class Agent:
            choose = staticmethod(shared)

        # As an actor (five positional arguments) it cannot take the seat; as an agent (seat by keyword) it can.
        # One cached answer must not be reused for the other call pattern.
        with self.assertRaises(TypeError):
            selfplay._call_actor(shared, None, None, None, 0, 2)
        self.assertEqual(selfplay.choose_agent(Agent(), None, None, None, 3), 3)

    def test_subclass_overrides_are_keyed_by_owner_class(self):
        from hasami_learn import selfplay

        class Base:
            def choose(self, game, limits, cancel=None):
                return "base"

        class Child(Base):
            def choose(self, game, limits, cancel=None, *, player=None):
                return ("child", player)

        self.assertEqual(selfplay.choose_agent(Base(), None, None, None, 2), "base")
        self.assertEqual(selfplay.choose_agent(Child(), None, None, None, 2), ("child", 2))


@requires_torch
class DecisionRuleTests(LearnTestCase):
    def test_policy_action_is_the_policy_agent_rule(self):
        import numpy as np
        from hasami_learn.encoding import sample_from_policy
        from hasami_learn.inference import _softmax_masked, policy_action

        rng = np.random.default_rng(3)
        for _ in range(200):
            logits = rng.normal(size=111).astype(np.float32)
            mask = (rng.random(111) < 0.2).astype(np.float32)
            mask[rng.integers(111)] = 1.0
            for temperature in (0.0, 0.5, 1.0):
                a, b = np.random.default_rng(9), np.random.default_rng(9)
                self.assertEqual(policy_action(logits, mask, a, temperature),
                                 sample_from_policy(_softmax_masked(logits, mask), mask, b, temperature))
                self.assertEqual(a.bit_generator.state, b.bit_generator.state)

    def test_selfplay_agents_skip_auxiliary_outputs(self):
        from hasami_learn.config import NamedConfig
        from hasami_learn.trainer import Trainer

        config = self.tiny("ppo", epochs=1, minibatches=1)
        config.game = NamedConfig("riichi", {"rounds": "tonpuu"})
        config.network = NamedConfig("mlp", {"hidden": 16, "layers": 1, "aux_heads": {"tenpai": [3]}})
        trainer = Trainer(config, echo=False)
        try:
            self.assertFalse(trainer.learner.make_agent("selfplay").aux_outputs)
            self.assertTrue(trainer.learner.make_agent("eval").aux_outputs)
            self.assertFalse(trainer.opponents._current_fn().aux_outputs)
        finally:
            trainer.close()


@requires_torch
class DeviceBatchingTests(unittest.TestCase):
    def test_per_device_defaults_and_overrides(self):
        import torch
        from hasami_learn.devices import DEFAULT_INFERENCE_BATCH, resolve_inference_batch

        for kind in ("cpu", "mps", "cuda"):
            self.assertEqual(resolve_inference_batch(None, torch.device(kind)), DEFAULT_INFERENCE_BATCH[kind])
            self.assertEqual(resolve_inference_batch(5, torch.device(kind)), 5)
        # The M2 CPU limit must never leak into GPU defaults.
        self.assertGreater(DEFAULT_INFERENCE_BATCH["cuda"], DEFAULT_INFERENCE_BATCH["cpu"])
        self.assertGreater(DEFAULT_INFERENCE_BATCH["mps"], DEFAULT_INFERENCE_BATCH["cpu"])
        for bad in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                resolve_inference_batch(bad, torch.device("cpu"))

    def test_schema_accepts_null_and_fixed_shape(self):
        import json
        from pathlib import Path
        from hasami_learn.config import validate_config_dict

        data = json.loads((Path(__file__).resolve().parents[2] / "configs/tiny/riichi-ppo-league.json").read_text())
        for value, ok in ((None, True), (8, True), (0, False), ("8", False)):
            data["device"]["inference_batch"] = value
            self.assertEqual(not validate_config_dict(data), ok, value)
        data["device"]["inference_batch"] = None
        data["device"]["fixed_batch_shape"] = "yes"
        self.assertTrue(validate_config_dict(data))

    def test_integral_floats_load_as_int_and_others_fail_at_load(self):
        import json
        from pathlib import Path
        from hasami_learn.config import ConfigError, TrainingConfig

        data = json.loads((Path(__file__).resolve().parents[2] / "configs/tiny/riichi-ppo-league.json").read_text())
        data["device"]["inference_batch"] = 8.0  # JSON schema "integer" accepts 8.0
        config = TrainingConfig.from_dict(data)
        self.assertIs(type(config.device.inference_batch), int)
        self.assertEqual(config.device.inference_batch, 8)
        for bad in (8.5, 0.0, True):
            data["device"]["inference_batch"] = bad
            with self.assertRaises(ConfigError):
                TrainingConfig.from_dict(data, validate=False)


@requires_torch
class PoolSamplingStreamTests(unittest.TestCase):
    def test_sample_uses_the_supplied_stream_only(self):
        import numpy as np
        from hasami_learn.config import OpponentsConfig
        from hasami_learn.games import get_spec
        from hasami_learn.opponents import OpponentPool

        spec = get_spec("riichi")
        encoder = spec.make_encoder()
        cfg = OpponentsConfig(strategy="pfsp", classical=["random", "tsumogiri"], classical_weight=0.5)
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        pool = OpponentPool(cfg, "cpu", encoder, np.random.default_rng(1), tmp.name, current_agent_fn=lambda: None,
                            spec=spec)
        before = pool.rng.bit_generator.state
        picks = [pool.sample(np.random.default_rng([7, i])).entry_id for i in range(30)]
        self.assertEqual(pool.rng.bit_generator.state, before)
        self.assertEqual(picks, [pool.sample(np.random.default_rng([7, i])).entry_id for i in range(30)])


class CollectionBenchmarkReportTests(unittest.TestCase):
    def test_speedups_compare_the_reference_at_the_same_batch(self):
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_collection.py"
        spec = importlib.util.spec_from_file_location("benchmark_collection", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        row = lambda mode, batch, rate: {"device": "cpu", "tables": 8, "mode": mode, "inference_batch": batch,
                                         "decisions_per_s": rate, "samples_per_s": rate, "joint_steps_per_s": rate}
        legacy = {"device": "cpu", "tables": 8, "mode": "legacy@abc", "decisions_per_s": 10.0,
                  "samples_per_s": 10.0, "joint_steps_per_s": 10.0}
        results = [row("batched", 4, 40.0), row("reference", 4, 20.0), row("batched", 8, 80.0),
                   row("reference", 8, 10.0), legacy, {"device": "cpu", "tables": 8, "inference_batch": 4,
                                                       "modes_identical": True}]
        got = {(s["batch"], s["versus"], s["versus_batch"]): s["decisions_per_s_ratio"]
               for s in module.compute_speedups(results)}
        self.assertEqual(got, {(4, "reference", 4): 2.0, (4, "legacy@abc", None): 4.0,
                               (8, "reference", 8): 8.0, (8, "legacy@abc", None): 8.0})


if __name__ == "__main__":
    unittest.main()
