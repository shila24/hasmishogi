"""Inference bundles remain legal and independent of optimizer and replay state."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(TORCH_AVAILABLE, 'torch not installed')
class InferenceBundleTests(unittest.TestCase):
    def test_legacy_and_version_two_inference_ignore_training_state(self):
        from hasami import create_game
        from hasami.agents import SearchLimits
        from hasami_learn.checkpoint_bundle import bundle_checkpoint
        from hasami_learn.inference import create_agent
        from hasami_learn.manifest import ModelManifest, save_checkpoint
        from hasami_learn.networks.mlp import MlpNet

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for version in (None, 2):
                directory = root / f'source-{version}'
                params = {} if version is None else {'search_semantics_version': version}
                network = MlpNet(head='policy_value', num_planes=9, hidden=8, layers=1)
                manifest = ModelManifest(algorithm='alphazero', network=dict(name='mlp', head='policy_value', params=dict(hidden=8, layers=1)),
                                         encoding_id='hasami-enc-v1', num_planes=9, algorithm_params=params,
                                         inference_hint=dict(mode='mcts', simulations=2))
                save_checkpoint(directory, network, manifest)
                (directory / 'learner_state.pt').write_bytes(b'not an inference input')
                (directory / 'trainer_state.json').write_text('invalid training state')
                (directory / 'replay_snapshot.npz').write_bytes(b'replay')
                bundle = root / f'bundle-{version}'
                bundle_checkpoint(directory, bundle, inference_only=True)
                for candidate in (directory, bundle):
                    agent = create_agent(str(candidate), device='cpu', simulations=2)
                    game = create_game('python')
                    try:
                        before = game.position
                        decision = agent.choose(game, SearchLimits(max_depth=1, time_ms=10000, node_limit=2))
                        self.assertIs(type(decision.action), int)
                        self.assertIn(decision.action, game.legal_actions())
                        self.assertEqual(game.position, before)
                        self.assertEqual(agent.manifest.algorithm_params, params)
                    finally:
                        game.close()
                self.assertEqual({p.name for p in bundle.iterdir()}, {'weights.pt', 'manifest.json'})
