"""Contracts prepared for riichi learning: riichi-history encoder, auxiliary heads, Decision.extra, hand-end step info."""

from __future__ import annotations

import json
import random
import unittest

from tests.learn._common import LearnTestCase, requires_torch

AUX_HEADS = {
    "tenpai": {"shape": [3], "activation": "sigmoid"},
    "waits": {"shape": [3, 34], "activation": "sigmoid"},
    "placement": {"shape": [4], "activation": "softmax"},
    "raw": [2],
}


def train_from_config_quiet(config):
    from hasami_learn.trainer import train_from_config

    return train_from_config(config, echo=False)


@requires_torch
class RiichiHistoryEncoderTests(unittest.TestCase):
    def test_registration_defaults_and_validation(self):
        from hasami_learn.games import get_spec
        from hasami_learn.registry import ENCODERS

        spec = get_spec("riichi")
        self.assertIn("riichi-history", ENCODERS.names())
        encoder = spec.make_encoder("riichi-history")
        self.assertEqual((encoder.encoding_id, encoder.k, encoder.num_planes, encoder.action_size),
                         ("riichi-history", 16, 120 + 32, 111))
        self.assertEqual(ENCODERS.get("riichi-history")(k=3).num_planes, 126)
        for bad in ({"k": 0}, {"k": True}, {"k": "4"}, {"k": 2.0}, {"k": 2, "extra": 1}, {"steps": 2}):
            with self.subTest(params=bad), self.assertRaises(ValueError):
                spec.make_encoder("riichi-history", **bad)
        with self.assertRaises(ValueError):
            spec.make_encoder("riichi-enc-v1", k=2)

    def test_history_planes_track_the_last_joint_steps_relative_to_the_observer(self):
        import numpy as np
        from hasami_learn.games import get_spec
        from hasami_learn.games.riichi import RiichiActionLayout

        spec = get_spec("riichi")
        encoder = spec.make_encoder("riichi-history", k=3)
        game = spec.create_game(seed=5)
        planes = encoder.encode(game, 0)
        self.assertEqual((planes.shape, planes.dtype), ((126, 4, 34), np.float32))
        np.testing.assert_array_equal(planes[:120], spec.observe(game, 0))
        self.assertFalse(planes[120:].any())
        tile = game.legal_actions(0)[0]
        self.assertLess(tile, 34)
        spec.step(game, {0: tile})
        # Seat 0 discarded; observer 1 sees seat 0 in relative row 3, in the newest block.
        planes = encoder.encode(game, 1)
        newest = planes[124:126]
        self.assertEqual(newest[0, 3, tile], 1.0)
        self.assertEqual(newest[0].sum(), 1.0)
        self.assertEqual(newest[1, 3, RiichiActionLayout.KINDS.index("discard")], 1.0)
        self.assertEqual(newest[1].sum(), 1.0)
        self.assertFalse(planes[120:124].any())
        self.assertEqual(encoder.encode(game, 0)[124, 0, tile], 1.0)
        # Everyone passes: the pass step becomes the newest block, the discard moves one block back.
        spec.step(game, {p: 109 for p in (1, 2, 3)})
        planes = encoder.encode(game, 1)
        newest, previous = planes[124:126], planes[122:124]
        self.assertFalse(newest[0].any())
        for seat, row in ((1, 0), (2, 1), (3, 2)):
            self.assertEqual(newest[1, row, RiichiActionLayout.KINDS.index("pass")], 1.0)
        self.assertEqual(newest[1].sum(), 3.0)
        self.assertEqual(previous[0, 3, tile], 1.0)
        self.assertFalse(planes[120:122].any())
        np.testing.assert_array_equal(encoder.encode(game), planes)
        np.testing.assert_array_equal(encoder.legal_mask(game, 1), spec.legal_mask(game, 1))
        with self.assertRaises(ValueError):
            RiichiActionLayout().render(np.zeros((2, 4, 34), np.float32), 0, 111)

    def test_history_is_stateless_and_resets_at_hand_boundaries(self):
        import numpy as np
        from riichi.actions import TSUMO
        from riichi.game import RiichiGame
        from hasami_learn.games import get_spec
        from tests.riichi.test_game import rig

        spec = get_spec("riichi")
        encoder = spec.make_encoder("riichi-history", k=4)
        game = spec.create_game(seed=9)
        rng = random.Random(1)
        for _ in range(20):
            spec.step(game, {p: rng.choice(game.legal_actions(p)) for p in game.actors()})
        player = game.actors()[0]
        first = encoder.encode(game, player)
        self.assertTrue(first[120:].any())
        np.testing.assert_array_equal(encoder.encode(game.clone(), player), first)
        np.testing.assert_array_equal(encoder.encode(RiichiGame.replay(game.seed, game.history), player), first)
        again = encoder.encode(game, player)
        first[:] = -1
        np.testing.assert_array_equal(encoder.encode(game, player), again)
        rigged = rig(["123m123p123s789s55z", None, None, None])
        rigged.play(TSUMO)
        self.assertFalse(rigged.outcome.terminal)
        planes = encoder.encode(rigged, rigged.actors()[0])
        self.assertFalse(planes[120:].any())

    def test_generic_scalar_layout_and_joint_history_helper(self):
        import numpy as np
        from hasami_learn.history import HistoryEncoder, ScalarActionLayout, joint_history
        from hasami_learn.games import get_spec

        layout = ScalarActionLayout((2, 3), 8)
        self.assertEqual(layout.planes_per_step, 3)
        block = np.zeros((3, 2, 3), np.float32)
        layout.render(block, 1, 7)
        self.assertEqual(block[1, 0, 1], 1.0)
        self.assertEqual(block[:2].sum(), 1.0)
        np.testing.assert_array_equal(block[2, 1], np.ones(3))
        with self.assertRaises(ValueError):
            layout.render(block, 0, 8)
        spec = get_spec("riichi")
        game = spec.create_game(seed=1)
        spec.step(game, {0: game.legal_actions(0)[0]})
        self.assertEqual(joint_history(spec, game), game.history)
        base = spec.make_encoder("")
        for bad in ({"k": 0}, {"num_players": 0}):
            with self.assertRaises(ValueError):
                HistoryEncoder(base, layout=layout, encoding_id="x", history=lambda g: [],
                               **{"k": 1, "num_players": 4, **bad})


@requires_torch
class AuxiliaryHeadTests(LearnTestCase):
    def test_networks_keep_the_head_contract_and_expose_forward_aux(self):
        import torch
        from hasami_learn.networks.mlp import MlpNet
        from hasami_learn.networks.resnet import ResNet

        for cls, params in ((MlpNet, dict(hidden=16, layers=1)), (ResNet, dict(channels=4, blocks=1, hidden=8))):
            with self.subTest(network=cls.__name__):
                net = cls(head="policy_value", num_planes=5, board_shape=(4, 34), action_size=111,
                          aux_heads=AUX_HEADS, **params).eval()
                x = torch.randn(2, 5, 4, 34)
                head, value = net(x)
                self.assertEqual((tuple(head.shape), tuple(value.shape)), ((2, 111), (2,)))
                head2, value2, raw = net.forward_aux(x)
                torch.testing.assert_close(head, head2)
                torch.testing.assert_close(value, value2)
                self.assertEqual(set(raw), set(AUX_HEADS))
                self.assertEqual({k: tuple(v.shape) for k, v in raw.items()},
                                 {"tenpai": (2, 3), "waits": (2, 3, 34), "placement": (2, 4), "raw": (2, 2)})
                self.assertEqual(net.aux_heads["raw"], {"shape": [2], "activation": "none"})
                activated = net.aux.activate(raw)
                self.assertTrue(((activated["tenpai"] >= 0) & (activated["tenpai"] <= 1)).all())
                torch.testing.assert_close(activated["placement"].sum(dim=-1), torch.ones(2))
                torch.testing.assert_close(activated["raw"], raw["raw"])
                plain = cls(head="policy_value", num_planes=5, board_shape=(4, 34), action_size=111, **params)
                self.assertEqual(plain.aux_heads, {})
                self.assertIsNone(plain.aux)
                self.assertEqual(plain.forward_aux(x)[2], {})
                with self.assertRaises(ValueError):
                    cls(head="policy_value", num_planes=5, board_shape=(4, 34), action_size=111,
                        aux_heads={"bad name": [1]}, **params)

    def test_declarations_are_validated(self):
        from hasami_learn.networks.base import normalize_aux_heads

        self.assertEqual(normalize_aux_heads(None), {})
        self.assertEqual(normalize_aux_heads({"a": 3, "b": (2, 5)}),
                         {"a": {"shape": [3], "activation": "none"}, "b": {"shape": [2, 5], "activation": "none"}})
        for bad in ({"bad name": [1]}, {"a": [0]}, {"a": []}, {"a": [True]}, {"a": "x"}, {1: [1]}, ["a"],
                    {"a": {"shape": [1], "activation": "relu"}}, {"a": {"shape": [1], "extra": 1}}, {"a": {}}):
            with self.subTest(declaration=bad), self.assertRaises(ValueError):
                normalize_aux_heads(bad)

    def test_policy_and_q_agents_publish_activated_outputs_in_decision_extra(self):
        from hasami.agents import SearchLimits
        from hasami_learn.games import get_spec
        from hasami_learn.inference import PolicyAgent, QAgent
        from hasami_learn.networks.mlp import MlpNet

        spec = get_spec("riichi")
        encoder = spec.make_encoder("riichi-history", k=2)
        game = spec.create_game(seed=2)
        limits = SearchLimits(1, 100, 10)
        for head, cls in (("policy_value", PolicyAgent), ("q", QAgent)):
            with self.subTest(agent=cls.__name__):
                net = MlpNet(head=head, num_planes=encoder.num_planes, board_shape=(4, 34), action_size=111,
                             hidden=16, layers=1, aux_heads=AUX_HEADS)
                agent = cls(net, encoder, "cpu", spec=spec)
                self.assertEqual(set(agent.aux_heads), set(AUX_HEADS))
                decision = agent.choose(game, limits, player=0)
                self.assertIn(decision.action, game.legal_actions(0))
                aux = decision.extra["aux"]
                self.assertEqual(set(aux), set(AUX_HEADS))
                self.assertEqual((len(aux["waits"]), len(aux["waits"][0]), len(aux["tenpai"])), (3, 34, 3))
                self.assertTrue(all(0 <= v <= 1 for row in aux["waits"] for v in row))
                self.assertAlmostEqual(sum(aux["placement"]), 1.0, places=5)
                self.assertEqual(json.loads(json.dumps(decision.extra)), decision.extra)
                out, value, mask = agent.evaluate(game, 0)
                self.assertEqual((out.shape, mask.shape), ((111,), (111,)))
                self.assertEqual(agent.evaluate_aux(game, 0, aux=False)[3], {})
                self.assertEqual(cls(net, encoder, "cpu", spec=spec, aux_outputs=False).choose(game, limits, player=0).extra, {})
                bare = MlpNet(head=head, num_planes=encoder.num_planes, board_shape=(4, 34), action_size=111, hidden=16, layers=1)
                self.assertEqual(cls(bare, encoder, "cpu", spec=spec).choose(game, limits, player=0).extra, {})

    def test_checkpoint_roundtrip_restores_history_encoder_and_aux_heads(self):
        from hasami.agents import SearchLimits
        from hasami_learn.config import NamedConfig
        from hasami_learn.games import get_spec
        from hasami_learn.inference import create_agent, describe_agent
        from hasami_learn.trainer import train_from_config

        config = self.tiny("ppo", epochs=1, minibatches=1)
        config.game = NamedConfig("riichi", {})
        config.encoder = NamedConfig("riichi-history", {"k": 2})
        config.network = NamedConfig("mlp", {"hidden": 16, "layers": 1, "aux_heads": AUX_HEADS})
        config.selfplay.games_per_iteration = 1
        config.selfplay.max_plies = 12
        final = train_from_config(config, echo=False)
        run_dir = final.parent.parent
        agent = create_agent(str(run_dir), device="cpu")
        manifest = agent.manifest
        self.assertEqual((manifest.encoding_id, manifest.encoder_params, manifest.num_planes),
                         ("riichi-history", {"k": 2}, 124))
        self.assertEqual(manifest.network["head"], "policy_value")
        self.assertEqual(manifest.network["params"]["aux_heads"], AUX_HEADS)
        self.assertEqual((agent.encoder.encoding_id, agent.encoder.num_planes), ("riichi-history", 124))
        self.assertEqual(set(agent.aux_heads), set(AUX_HEADS))
        spec = get_spec("riichi")
        game = spec.create_game(seed=11)
        decision = agent.choose(game, SearchLimits(1, 100, 10))
        self.assertIn(decision.action, game.legal_actions(0))
        self.assertEqual(set(decision.extra["aux"]), set(AUX_HEADS))
        self.assertIn("aux placement,raw,tenpai,waits", describe_agent(agent))
        self.assertEqual(create_agent(str(run_dir), device="cpu", aux_outputs=False)
                         .choose(game, SearchLimits(1, 100, 10)).extra, {})

    def test_rule_params_select_tonpuu_and_survive_the_checkpoint(self):
        from hasami.agents import SearchLimits
        from hasami_learn.config import NamedConfig
        from hasami_learn.games import get_spec
        from hasami_learn.inference import create_agent
        from hasami_learn.manifest import check_manifest_matches_spec

        spec = get_spec("riichi", {"rounds": "tonpuu"})
        self.assertEqual((spec.rules.rounds, spec.rules.nagashi_mangan), ("tonpuu", False))
        self.assertEqual(spec.create_game(rules={"red_fives": False}).rules.rounds, "tonpuu")
        with self.assertRaises(ValueError):
            get_spec("riichi", {"ruleset_id": "riichi-4p-v1"})
        game, rng = spec.create_game(seed=5), random.Random(5)
        while not spec.step(game, {p: rng.choice(spec.legal_actions(game, p))
                                   for p in spec.actors(game)}).terminal:
            pass
        self.assertLessEqual(game.round_index, 4)

        config = self.tiny("ppo", epochs=1, minibatches=1)
        config.game = NamedConfig("riichi", {"rounds": "tonpuu"})
        config.network = NamedConfig("mlp", {"hidden": 16, "layers": 1})
        config.selfplay.games_per_iteration = 1
        config.selfplay.max_plies = 12
        final = train_from_config_quiet(config)
        agent = create_agent(str(final.parent.parent), device="cpu")
        self.assertEqual(agent.manifest.game_params, {"rounds": "tonpuu"})
        self.assertEqual(agent.manifest.rules["rounds"], "tonpuu")
        self.assertEqual(agent.spec.rules.rounds, "tonpuu")
        check_manifest_matches_spec(agent.manifest, spec, {"rounds": "tonpuu"})
        with self.assertRaises(ValueError):
            check_manifest_matches_spec(agent.manifest, get_spec("riichi"), {})
        played = agent.spec.create_game(seed=3)
        self.assertIn(agent.choose(played, SearchLimits(1, 100, 10)).action, played.legal_actions(0))

    def test_step_info_reveals_hidden_state_at_hand_end_and_is_detached(self):
        from hasami_learn.games import get_spec
        from hasami_learn.selfplay import run_table_game
        from tests.riichi.test_hand_end import REVEAL_KEYS, check_hand_summary

        spec = get_spec("riichi")
        game = spec.create_game(seed=21)
        rng = random.Random(3)
        outcome = spec.step(game, {p: rng.choice(game.legal_actions(p)) for p in game.actors()})
        while "hand" not in outcome.info:
            outcome = spec.step(game, {p: rng.choice(game.legal_actions(p)) for p in game.actors()})
        summary = outcome.info["hand"]
        check_hand_summary(self, summary)
        self.assertEqual(len(outcome.info["score_delta"]), 4)
        summary["waits"][0][0] = 99
        summary["hands"][0].append(-1)
        self.assertNotEqual(game.last_hand["waits"][0][0], 99)
        self.assertNotIn(-1, game.last_hand["hands"][0])
        seats = {p: spec.baseline_agent("random", seed=p)[0] for p in range(4)}
        trajectory = run_table_game(seats, spec.make_encoder(""), spec=spec, seed=21, max_plies=400)
        summaries = [info["hand"] for info in trajectory.step_info if "hand" in info]
        self.assertTrue(summaries, "no hand ended within the ply cap")
        for summary in summaries:
            for key in REVEAL_KEYS:
                self.assertIn(key, summary)
        self.assertEqual(len(trajectory.step_info), trajectory.plies)


if __name__ == "__main__":
    unittest.main()
