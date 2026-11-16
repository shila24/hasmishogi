"""The riichi adapter's determinize hook (practical-play spec section 11)."""

from __future__ import annotations

import random
import unittest

from tests.learn._common import requires_torch


@requires_torch
class RiichiDeterminizeTests(unittest.TestCase):
    def play(self, backend, plies=40, seed=7):
        from hasami_learn.games import get_spec

        spec = get_spec("riichi")
        game = spec.create_game(backend, seed=seed)
        self.addCleanup(game.close)
        rng = random.Random(seed)
        for _ in range(plies):
            actors = spec.actors(game)
            if not actors:
                break
            spec.step(game, {player: rng.choice(spec.legal_actions(game, player)) for player in actors})
        return spec, game

    def check(self, backend):
        import numpy as np

        spec, game = self.play(backend)
        player = spec.actors(game)[0]
        before = {p: game.observation(p) for p in range(4)}
        hidden = set()
        for seed in range(6):
            world = spec.determinize(game, player, np.random.default_rng(seed))
            try:
                self.assertIsNot(world, game)
                self.assertEqual(world.observation(player), before[player])  # the observer sees the same game
                self.assertEqual(spec.actors(world), spec.actors(game))
                hidden.add(tuple(tuple(sorted(world.observation(p)["hand"])) for p in range(4) if p != player))
            finally:
                world.close()
        self.assertEqual({p: game.observation(p) for p in range(4)}, before)  # the original is untouched
        self.assertGreater(len(hidden), 1)  # opponents' hands are actually resampled

    def test_python_backend(self):
        self.check("python")

    def test_native_backend(self):
        from hasami_learn.games.riichi import _native_available

        if not _native_available():
            self.skipTest("native riichi engine not built")
        self.check("cpp")


if __name__ == "__main__":
    unittest.main()
