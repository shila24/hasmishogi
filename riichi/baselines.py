"""Information-set-only random, tsumogiri, and shanten-greedy baselines."""

from dataclasses import dataclass
import random

from .actions import NINE_TERMINALS, PASS, RIICHI, RON, TSUMO
from .agari import Meld
from .shanten import shanten


@dataclass(frozen=True)
class Decision:
    action: int
    score: int = 0
    depth: int = 0
    nodes: int = 1
    elapsed_ms: float = 0.0
    completed: bool = True


class BaselineAgent:
    def __init__(self, name: str = 'random', seed: int = 0):
        if name not in ('random', 'tsumogiri', 'shanten-greedy'):
            raise ValueError(f'unknown riichi baseline: {name}')
        self.name = name
        self.rng = random.Random(seed)

    def choose(self, game, limits=None, cancel=None, *, player: int | None = None) -> Decision:
        if cancel is not None and cancel.is_set():
            raise InterruptedError('agent request was cancelled')
        if player is None:
            actors = game.actors()
            if len(actors) != 1:
                raise ValueError('specify player for simultaneous decisions')
            player = actors[0]
        obs = game.observation(player)
        legal = obs['legal_actions']
        if not legal:
            raise ValueError('no legal actions for player')
        if self.name == 'random':
            return Decision(self.rng.choice(legal))
        for action in (TSUMO, RON):
            if action in legal:
                return Decision(action)
        if PASS in legal:
            return Decision(PASS)
        if self.name == 'tsumogiri':
            drawn = obs['drawn_tile']
            if drawn is not None and drawn // 4 in legal:
                return Decision(drawn // 4)
            return Decision(next(action for action in legal if action < 34))
        melds = tuple(Meld(tuple(t // 4 for t in m['tiles']), m['kind'], m['open'])
                      for m in obs['melds'][0])
        candidates = []
        for action in legal:
            if action >= 68:
                continue
            counts = list(obs['hand_counts'])
            counts[action % 34] -= 1
            candidates.append((shanten(counts, melds), action < RIICHI, action))
        if candidates:
            best = min(candidates)[:2]
            actions = [a for distance, plain, a in candidates if (distance, plain) == best]
            return Decision(self.rng.choice(actions))
        return Decision(NINE_TERMINALS if NINE_TERMINALS in legal else legal[0])
