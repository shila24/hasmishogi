"""Native adapter fast paths, cross-backend records, search, and PPO league smoke."""
import json
import os
from pathlib import Path
import random
import unittest

from tests.learn._common import LearnTestCase, requires_torch


@requires_torch
class NativeBackendTests(LearnTestCase):
    def setUp(self):
        super().setUp()
        from riichi.native import available
        if not available():
            if os.environ.get('HASAMI_REQUIRE_NATIVE') == '1':
                self.fail('native riichi game is required')
            self.skipTest('native riichi game is unavailable')

    def test_adapter_fast_paths_and_cross_backend_records(self):
        import numpy as np
        from hasami_learn.games import get_spec
        from hasami_learn.games.records import save_record, load_record
        spec = get_spec('riichi', {'rounds': 'tonpuu'})
        self.assertEqual(spec.resolve_backend('auto'), 'cpp')
        py = spec.create_game('python', seed=34)
        cpp = spec.create_game('cpp', seed=34)
        rng = random.Random(34)
        try:
            while not py.outcome.terminal:
                self.assertLess(len(py.moves), 20000)
                for p in range(4):
                    self.assertEqual(spec.observe(py, p).tobytes(), spec.observe(cpp, p).tobytes())
                    self.assertEqual(spec.legal_mask(py, p).tobytes(), spec.legal_mask(cpp, p).tobytes())
                    buf = np.empty((120, 4, 34), dtype=np.float32)
                    cpp.encode_into(p, buf)
                    self.assertEqual(buf.tobytes(), spec.observe(py, p).tobytes())
                actions = {p: rng.choice(py.legal_actions(p)) for p in py.actors()}
                self.assertEqual(spec.step(py, actions), spec.step(cpp, actions))
            for game, backend in ((py, 'cpp'), (cpp, 'python')):
                path = save_record(spec, game, self.tmp / f'{backend}.json')
                replay = load_record(spec, path, backend)
                try:
                    self.assertEqual(replay.moves, game.moves)
                    self.assertEqual(replay.last_hand, game.last_hand)
                    self.assertEqual(replay.final_points, game.final_points)
                    self.assertEqual(replay.outcome, game.outcome)
                    for p in range(4):
                        self.assertEqual(replay.observation(p), game.observation(p))
                finally:
                    replay.close()
        finally:
            py.close(); cpp.close()

    def test_ismcts_native_actions_are_legal_and_root_unchanged(self):
        from hasami.agents import SearchLimits
        from hasami_learn.algorithms.ismcts import IsmctsAgent
        from hasami_learn.games import get_spec
        spec = get_spec('riichi')
        with spec.create_game('cpp', seed=5) as game:
            agent = IsmctsAgent(spec=spec, simulations=3, rollout_steps=4, seed=2)
            limits = SearchLimits(time_ms=10000, node_limit=3)
            for _ in range(3):
                before = [game.observation(p) for p in range(4)]
                actions = {}
                for p in game.actors():
                    decision = agent.choose(game, limits, player=p)
                    self.assertIn(decision.action, game.legal_actions(p))
                    self.assertEqual(decision.nodes, 3)
                    actions[p] = decision.action
                self.assertEqual(before, [game.observation(p) for p in range(4)])
                spec.step(game, actions)

    def test_tiny_ppo_league_native_checkpoint(self):
        from hasami.agents import SearchLimits
        from hasami_learn.config import TrainingConfig
        from hasami_learn.inference import create_agent
        from hasami_learn.trainer import Trainer
        cfg = TrainingConfig.from_dict(json.loads(Path('configs/tiny/riichi-ppo-league.json').read_text()))
        cfg.run.output_dir = str(self.tmp)
        cfg.run.backend = 'cpp'
        cfg.device.device = 'cpu'
        trainer = Trainer(cfg, echo=False)
        try:
            checkpoint = trainer.run()
            agent = create_agent(str(checkpoint), device='cpu')
            with agent.spec.create_game('cpp', seed=11) as game:
                result = agent.choose(game, SearchLimits())
                self.assertIn(result.action, game.legal_actions(0))
                self.assertEqual(agent.encoder.encoding_id, 'riichi-history')
        finally:
            trainer.close()
