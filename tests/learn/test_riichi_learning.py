"""League, belief supervision and information-set search regression tests."""

import json
from pathlib import Path
import unittest

from tests.learn._common import LearnTestCase, requires_torch


@requires_torch
class RiichiLearningTests(LearnTestCase):
    def test_tilenet_shapes_and_aux_gradients(self):
        import torch
        from hasami_learn.networks.tilenet import TileNet
        for shape in ((4, 34), (2, 7)):
            net = TileNet(num_planes=6, board_shape=shape, action_size=11,
                          channels=8, hidden=16, layers=1,
                          aux_heads={"waits": {"shape": [3, 34], "activation": "sigmoid"}})
            x = torch.randn(2, 6, *shape)
            policy, value, aux = net.forward_aux(x)
            self.assertEqual(policy.shape, (2, 11))
            self.assertEqual(value.shape, (2,))
            self.assertEqual(aux["waits"].shape, (2, 3, 34))
            torch.testing.assert_close(net(x)[0], policy)
            aux["waits"].square().mean().backward()
            self.assertTrue(any(p.grad is not None for p in net.body.parameters()))

    def test_auxiliary_alignment_relative_seats_and_final_placement(self):
        import numpy as np
        from hasami_learn.auxiliary import trajectory_targets
        from hasami_learn.selfplay import Trajectory
        waits = np.eye(4, 34).tolist()
        first = dict(tenpai=[1, 0, 1, 0], waits=waits,
                     hand_counts=(np.eye(4, 34) * 4).tolist(), placement=[4, 3, 2, 1])
        final = dict(first, placement=[1, 2, 3, 4])
        trajectory = Trajectory(num_players=4, players=[2, 0, 1], actions=[0, 0, 0],
                                extras=[{"decision_ply": i} for i in (0, 2, 4)],
                                step_info=[{}, {"hand": first}, {}, {"hand": final}, {}],
                                truncated=True)
        targets = trajectory_targets(trajectory)
        np.testing.assert_array_equal(targets[0]["tenpai"], [0, 1, 0])
        self.assertEqual(targets[0]["waits"][0, 3], 1)
        self.assertEqual(targets[0]["hand"][1, 0], 1)
        self.assertEqual(targets[0]["danger"].sum(), 3)
        self.assertNotIn("placement", targets[0])
        self.assertEqual(targets[2], {})
        trajectory.truncated = False
        trajectory.step_info.pop()
        targets = trajectory_targets(trajectory)
        self.assertEqual(targets[0]["placement"].argmax(), 2)

    def test_ismcts_legal_and_private_state_independent(self):
        import numpy as np
        from hasami.agents import SearchLimits
        from hasami_learn.algorithms.ismcts import IsmctsAgent, sample_riichi_belief
        from hasami_learn.games import get_spec
        spec = get_spec("riichi")
        game = spec.create_game(seed=5)
        before = game.clone()
        other = game.clone()
        other.hands[1], other.hands[2] = other.hands[2], other.hands[1]
        other.live_wall.reverse()
        a = sample_riichi_belief(game, 0, np.random.default_rng(2))
        b = sample_riichi_belief(other, 0, np.random.default_rng(2))
        self.assertEqual(a.hands, b.hands)
        self.assertEqual(a.live_wall, b.live_wall)
        self.assertEqual(a.hands[0], game.hands[0])
        tiles = sum(a.hands, []) + a.live_wall + a.dead_wall
        self.assertEqual(sorted(tiles), list(range(136)))
        limits = SearchLimits(time_ms=10000, node_limit=3)
        agent = IsmctsAgent(spec=spec, simulations=3, rollout_steps=4, seed=2)
        decision = agent.choose(game, limits)
        self.assertIn(decision.action, spec.legal_actions(game, 0))
        self.assertEqual(decision.nodes, 3)
        self.assertEqual(game.hands, before.hands)
        self.assertEqual(game.live_wall, before.live_wall)
        spec.step(game, {0: decision.action})
        for player in spec.actors(game):
            self.assertIn(agent.choose(game, limits, player=player).action,
                          spec.legal_actions(game, player))

    def test_tiny_league_checkpoint_with_beliefs(self):
        from hasami.agents import SearchLimits
        from hasami_learn.config import TrainingConfig
        from hasami_learn.games import get_spec
        from hasami_learn.inference import create_agent
        from hasami_learn.trainer import Trainer
        cfg = TrainingConfig.from_dict(json.loads(Path(
            "configs/tiny/riichi-ppo-league.json").read_text()))
        cfg.run.output_dir = str(self.tmp)
        trainer = Trainer(cfg, echo=False)
        try:
            checkpoint = trainer.run()
            agent = create_agent(str(checkpoint), device="cpu")
            self.assertEqual(agent.encoder.encoding_id, "riichi-history")
            game = get_spec("riichi").create_game(seed=11)
            result = agent.choose(game, SearchLimits())
            self.assertIn(result.action, game.legal_actions(0))
            self.assertEqual(set(result.extra["aux"]), {"tenpai", "waits", "hand", "placement", "danger"})
            self.assertAlmostEqual(sum(result.extra["aux"]["hand"][0]), 1, places=5)
            json.dumps(result.extra, allow_nan=False)
        finally:
            trainer.close()

    def test_collection_rotates_duplicate_deals_masks_missing_labels_and_copies(self):
        import numpy as np
        import torch
        from hasami_learn.config import TrainingConfig
        from hasami_learn.trainer import Trainer
        cfg = TrainingConfig.from_dict(json.loads(Path("configs/tiny/riichi-ppo-league.json").read_text()))
        cfg.run.output_dir = str(self.tmp)
        trainer = Trainer(cfg, echo=False)
        try:
            learner = trainer.learner
            stats = learner.collect(1)
            self.assertEqual(stats.games, 4)
            self.assertEqual([row["learner_seat"] for row in stats.extra["tables"]], [0, 1, 2, 3])
            self.assertEqual(len({row["seed"] for row in stats.extra["tables"]}), 1)
            self.assertEqual(stats.truncated, 4)
            self.assertEqual(trainer.opponents.current.games, 0)
            self.assertGreater(stats.extra["aux_labelled"], 0)
            self.assertTrue(all("placement" not in t for t in learner._aux_targets))
            self.assertTrue(all(mask[action] for mask, action in zip(learner._masks, learner._actions)))
            before = {name: parameter.detach().clone() for name, parameter in trainer.network.aux.named_parameters()}
            updated = learner.update(1)
            self.assertTrue(np.isfinite(updated.loss))
            self.assertEqual(set(updated.extra["aux_loss"]), {"tenpai", "waits", "hand", "danger"})
            self.assertTrue(any(not torch.equal(before[name], parameter)
                                for name, parameter in trainer.network.aux.named_parameters()))
            self.assertEqual(len(learner._aux_targets), 0)
        finally:
            trainer.close()

    def test_auxiliary_loss_masks_unrevealed_examples(self):
        import numpy as np
        import torch
        from hasami_learn.auxiliary import auxiliary_loss
        logits = torch.zeros((2, 3), requires_grad=True)
        loss, metrics = auxiliary_loss({"tenpai": logits}, [{"tenpai": np.ones(3)}, {}])
        loss.backward()
        self.assertAlmostEqual(metrics["tenpai"], np.log(2), places=6)
        self.assertTrue(torch.all(logits.grad[0] < 0))
        self.assertTrue(torch.all(logits.grad[1] == 0))
        missing, metrics = auxiliary_loss({"tenpai": logits}, [{}, {}])
        self.assertEqual(float(missing.detach()), 0)
        self.assertEqual(metrics, {})

    def test_per_decision_rewards_use_fixed_seats_and_hand_shaping(self):
        import numpy as np
        from hasami_learn.algorithms.ppo_league import PpoLeagueParams, league_gae
        from hasami_learn.selfplay import Trajectory
        params = PpoLeagueParams(gamma=1, gae_lambda=1, hand_reward_coef=0.1, hand_reward_scale=100)
        trajectory = Trajectory(num_players=3, players=[0, 2, 0, 2], actions=[0] * 4,
                                extras=[{"decision_ply": p} for p in [0, 0, 2, 2]],
                                rewards=[0.8, 0.1, -0.4], winner=0,
                                step_info=[{}, {"hand": {"reason": "test"}, "score_delta": [100, 0, -100]}, {}])
        _, returns = league_gae(trajectory, [0] * 4, params, 0)
        np.testing.assert_allclose(returns, [0.9, -0.5, 0.8, -0.4])
        trajectory.truncated = True
        _, returns = league_gae(trajectory, [0] * 4, params, 0.25)
        np.testing.assert_allclose(returns, [0.35, 0.15, 0.25, 0.25])

    def test_exploiters_train_and_checkpoint_without_mutating_main(self):
        import copy
        import torch
        from hasami.agents import SearchLimits
        from hasami_learn.config import NamedConfig
        from hasami_learn.inference import create_agent
        from hasami_learn.trainer import Trainer
        from .test_multiplayer import HiddenSpec
        cfg = self.tiny("ppo-league", epochs=1, minibatches=1, exploiter_every=1,
                        exploiter_iterations=1, exploiter_games=2)
        cfg.game = NamedConfig(HiddenSpec.game_id)
        cfg.selfplay.max_plies = 6
        cfg.opponents.pool_size = 8
        trainer = Trainer(cfg, echo=False)
        try:
            learner = trainer.learner
            main = {name: p.detach().clone() for name, p in trainer.network.named_parameters()}
            rng_state = copy.deepcopy(learner.rng.bit_generator.state)
            results = learner._train_exploiters(1)
            self.assertEqual([r["archetype"] for r in results], ["defensive", "aggressive", "fast"])
            self.assertTrue(all(r["steps"] > 0 and r["games"] == 2 for r in results))
            self.assertEqual(learner.rng.bit_generator.state, rng_state)
            for name, p in trainer.network.named_parameters():
                torch.testing.assert_close(main[name], p)
            for entry in trainer.opponents.entries:
                agent = create_agent(entry.ref, device="cpu")
                self.assertTrue(any(not torch.equal(main[name], p)
                                    for name, p in agent.network.named_parameters()))
                game = trainer.spec.create_game(seed=3)
                for seat in trainer.spec.actors(game):
                    self.assertIn(agent.choose(game, SearchLimits(), player=seat).action, [0, 1])
            state = learner.state_dict()
            learner.load_state_dict(state)
            self.assertEqual(learner._exploiters, 3)
            learner.collect(2)
            updated = learner.update(2)
            self.assertEqual(len(updated.extra["exploiters"]), 3)
        finally:
            trainer.close()

    def test_ingestion_copies_arrays_and_rejects_off_policy_samples(self):
        import numpy as np
        from hasami_learn.config import NamedConfig
        from hasami_learn.selfplay import Trajectory
        from hasami_learn.trainer import Trainer
        from .test_multiplayer import HiddenSpec
        cfg = self.tiny("ppo-league", epochs=1)
        cfg.game = NamedConfig(HiddenSpec.game_id)
        trainer = Trainer(cfg, echo=False)
        try:
            planes = np.ones((2, 1, 3), dtype=np.float32)
            mask = np.ones(2, dtype=np.float32)
            trajectory = Trajectory(players=[0], actions=[0], planes=[planes], masks=[mask],
                                    extras=[{"decision_ply": 0, "logp": -0.69, "value": 0, "temperature": 1}])
            trainer.learner.ingest_trajectories([trajectory])
            planes.fill(99)
            mask.fill(0)
            self.assertTrue(np.all(trainer.learner._planes[0] == 1))
            self.assertTrue(np.all(trainer.learner._masks[0] == 1))
            trajectory.extras[0].pop("logp")
            with self.assertRaisesRegex(ValueError, "on-policy"):
                trainer.learner.ingest_trajectories([trajectory])
        finally:
            trainer.close()

    def test_ismcts_cancellation_deadline_and_adapter_truncation(self):
        import threading
        from hasami.agents import AgentCancelled, SearchLimits
        from hasami_learn.algorithms.ismcts import IsmctsAgent
        from hasami_learn.games import get_spec
        from hasami_learn.games.base import StepOutcome
        from .test_multiplayer import HiddenSpec
        class TruncatingSpec(HiddenSpec):
            def step(self, game, actions):
                if game.moves:
                    raise AssertionError("must not step after truncation")
                super().step(game, actions)
                return StepOutcome(False, True, None, {})
        spec = TruncatingSpec()
        def sample(game, player, rng, belief):
            work = game.clone()
            work.cards[1 - player] = int(rng.integers(10))
            return work
        agent = IsmctsAgent(spec=spec, sampler=sample, simulations=4, truncation_value=0.75)
        game = spec.create_game(seed=0)
        decision = agent.choose(game, SearchLimits(time_ms=10000, node_limit=4), player=0)
        self.assertEqual((decision.nodes, decision.score), (4, 750))
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(AgentCancelled):
            agent.choose(game, SearchLimits(), cancel, player=0)
        from unittest.mock import patch
        with patch("hasami_learn.algorithms.ismcts.time.perf_counter", side_effect=[0, 1, 1]):
            decision = agent.choose(game, SearchLimits(time_ms=1), player=0)
        self.assertIn(decision.action, [0, 1])
        self.assertEqual(decision.nodes, 0)
        self.assertFalse(decision.completed)
        riichi = get_spec("riichi")
        agent = IsmctsAgent(spec=riichi)
        game = riichi.create_game(seed=1)
        riichi.step(game, {0: game.legal_actions(0)[0]})
        with self.assertRaisesRegex(ValueError, "simultaneous"):
            agent.choose(game)


if __name__ == "__main__":
    unittest.main()
