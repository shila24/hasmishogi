"""RiichiGame.redeal: the hidden-state replacement contract shared by both engine backends."""
import copy
import random
import unittest

from riichi.baselines import BaselineAgent
from riichi.game import RiichiGame


def advance(game, plies, seed):
    agents = [BaselineAgent('random', seed + p) for p in range(4)]
    for _ in range(plies):
        actors = game.actors()
        if not actors:
            break
        game.play_joint({p: agents[p].choose(game, player=p).action for p in actors})


def snapshot(game):
    return copy.deepcopy({k: v for k, v in vars(game).items() if k not in ('rng', '_legal_cache')})


def shuffled_hidden(game, rng, player):
    """Swap concealed tiles between two opponents and permute the unrevealed walls."""
    hands = [list(h) for h in game.hands]
    a, b = [(player + 1) % 4, (player + 2) % 4]
    pool = hands[a] + hands[b]
    keep = {t for t in (game.drawn_tiles[a], game.drawn_tiles[b]) if t is not None}
    free = [t for t in pool if t not in keep]
    rng.shuffle(free)
    for seat in (a, b):
        fixed = [t for t in hands[seat] if t in keep]
        hands[seat] = fixed + [free.pop() for _ in range(len(hands[seat]) - len(fixed))]
    live = list(game.live_wall)
    rng.shuffle(live)
    return hands, live, list(game.dead_wall)


class RedealTests(unittest.TestCase):
    def test_permuting_hidden_tiles_is_accepted_and_reseeds(self):
        game = RiichiGame(seed=11)
        advance(game, 9, 3)
        hands, live, dead = shuffled_hidden(game, random.Random(1), 0)
        game.redeal(hands, live, dead, list(game.drawn_tiles), [False] * 4, [False] * 4, 2 ** 64 - 1)
        self.assertEqual(game.hands, [sorted(h) for h in hands])
        self.assertEqual((game.live_wall, game.seed), (live, 2 ** 64 - 1))
        self.assertEqual(game.rng.random(), random.Random(2 ** 64 - 1).random())
        advance(game, 200, 5)  # the redealt state stays playable

    def test_inconsistent_hidden_state_is_rejected_without_change(self):
        game = RiichiGame(seed=4)
        advance(game, 6, 1)
        before = snapshot(game)
        hands, live, dead = [list(h) for h in game.hands], list(game.live_wall), list(game.dead_wall)
        drawn, flags = list(game.drawn_tiles), [False] * 4
        cases = {
            'foreign tile': ([[*hands[0][:-1], game.discards[0][0].tile if game.discards[0] else 135],
                              *hands[1:]], live, dead, drawn),
            'hand size': ([hands[0][:-1], hands[1] + hands[0][-1:], hands[2], hands[3]], live, dead, drawn),
            'indicator': (hands, live, [*dead[:4], dead[5], dead[4], *dead[6:]], drawn),
            'wall length': (hands, live[:-1], dead + live[-1:], drawn),
            'drawn seat': (hands, live, dead, [hands[s][0] if d is None else None for s, d in enumerate(drawn)]),
        }
        for name, (h, l, d, dr) in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                game.redeal(h, l, d, dr, flags, flags, 7)
            self.assertEqual(snapshot(game), before)
        for bad_seed in (-1, 2 ** 64, 1.0):
            with self.assertRaises(ValueError):
                game.redeal(hands, live, dead, drawn, flags, flags, bad_seed)
        with self.assertRaises(ValueError):
            game.redeal(hands, live, dead, drawn, [0] * 4, flags, 7)

    def test_terminal_game_cannot_be_redealt(self):
        game = RiichiGame(seed=2)
        advance(game, 100000, 2)
        self.assertTrue(game.outcome.terminal)
        with self.assertRaises(ValueError):
            game.redeal(game.hands, game.live_wall, game.dead_wall, game.drawn_tiles, [False] * 4,
                        [False] * 4, 1)


if __name__ == '__main__':
    unittest.main()
