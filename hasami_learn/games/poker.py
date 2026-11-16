"""GameSpec v2 for heads-up limit Hold'em and Leduc verification."""

import math
import random

import numpy as np

from hasami.agents import Decision, check_cancel
from poker import PokerRules, create_game
from poker.baselines import BaselinePolicy
from poker.encoding import BOARD_SHAPE, ENCODING_ID, NUM_PLANES, encode, encode_observation
from ..registry import ENCODERS, GAMES
from .base import StepOutcome
from .records import RecordedGame


class PlaneEncoder:
    encoding_id = ENCODING_ID
    num_planes = NUM_PLANES
    action_size = 3

    def __init__(self, **params):
        if params:
            raise ValueError('poker-enc-v1 takes no parameters')

    @staticmethod
    def _player(game, player):
        if player is None:
            actors = game.actors()
            if len(actors) != 1:
                raise ValueError('terminal observations require an explicit player')
            return actors[0]
        return player

    def encode(self, game, player=None):
        return np.asarray(encode(game, self._player(game, player)), dtype=np.float32)

    def legal_mask(self, game, player=None):
        mask = np.zeros(3, dtype=np.float32)
        mask[game.legal_actions(self._player(game, player))] = 1
        return mask


ENCODERS.register(ENCODING_ID, PlaneEncoder)


class PokerBaselineAgent:
    def __init__(self, name, seed):
        self.policy = BaselinePolicy(name)
        self.rng = random.Random(seed)

    def choose(self, game, limits=None, cancel=None, *, player=None):
        check_cancel(cancel)
        player = PlaneEncoder._player(game, player)
        row = self.policy(game.observation(player))
        action = self.rng.choices(list(row), weights=list(row.values()))[0]
        check_cancel(cancel)
        return Decision(action)


@GAMES.register('poker')
class PokerSpec:
    game_id = 'poker'
    board_shape = BOARD_SHAPE
    action_size = 3
    action_encoding = 'poker-fold-call-raise-v1'
    default_max_plies = 32
    default_encoder = ENCODING_ID
    num_players = 2
    perfect_information = False
    has_chance = True
    zero_sum = True
    supports_undo = True
    draw_reasons = ()
    default_baselines = ('random', 'call', 'raise', 'tight:0.6')
    default_teacher = 'tight:0.6'

    def __init__(self, variant='hulhe'):
        self.rules = PokerRules(variant)
        self.ruleset_id = self.rules.ruleset_id

    def create_game(self, backend='python', rules=None, position=None, seed=None):
        self.resolve_backend(backend)
        if position is not None and (not isinstance(position, dict) or set(position) - {'button'}):
            raise ValueError('poker initial position may specify only the button')
        if rules is None:
            rules = self.rules
        elif isinstance(rules, dict):
            rules = {**self.rules.to_dict(), **rules}
        game = create_game(rules, seed, button=(position or {}).get('button'))
        if game.rules != self.rules:
            raise ValueError('poker rules must match the adapter variant')
        return game

    def resolve_backend(self, backend):
        if backend not in ('python', 'auto'):
            raise ValueError('poker supports only the Python backend')
        return 'python'

    def rules_dict(self, game):
        return game.rules.to_dict()

    def actors(self, game):
        return game.actors()

    def legal_actions(self, game, player):
        return game.legal_actions(player)

    def observe(self, game, player):
        return PlaneEncoder().encode(game, player)

    def legal_mask(self, game, player):
        return PlaneEncoder().legal_mask(game, player)

    def step(self, game, actions):
        game.play_joint(actions)
        return StepOutcome(game.outcome.terminal, False,
                           self.terminal_values(game) if game.outcome.terminal else None,
                           {'reason': game.outcome.reason})

    def terminal_values(self, game):
        if not game.outcome.terminal:
            raise ValueError('terminal_values requires a terminal hand')
        return [float(c / game.rules.max_contribution) for c in game.outcome.chips]

    def make_encoder(self, name='', **params):
        if name not in ('', ENCODING_ID):
            raise ValueError(f'unknown poker encoder: {name}')
        return PlaneEncoder(**params)

    def symmetries(self):
        return []

    def baseline_agent(self, spec, seed=0):
        return PokerBaselineAgent(spec, seed), None

    @staticmethod
    def action_name(action):
        if type(action) is not int or not 0 <= action < 3:
            raise ValueError('unknown poker action')
        return ('fold', 'call/check', 'raise/bet')[action]

    def determinize(self, game, player, rng):
        # A sampled world must not inherit a loaded record's original creation seed.
        source = game.game if isinstance(game, RecordedGame) else game
        work = source.clone()
        try:
            work.redeal({'player': player}, int(rng.integers(2**63)))
            return work
        except BaseException:
            work.close()
            raise

    @staticmethod
    def outcome_score(values, player):
        if (type(player) is not int or player not in (0, 1) or len(values) != 2
                or any(not math.isfinite(v) or not -1 <= v <= 1 for v in values)):
            raise ValueError('invalid seat or terminal values')
        return 0.5 + 0.5 * values[player]


def observation_policy(agent, temperature=1.0):
    """Expose a loaded poker policy's probabilities to exact/duplicate evaluation.

    Evaluation receives only an information-set observation, never a full game.
    The supplied agent must be a poker policy/value agent (e.g. create_agent).
    """
    import torch
    if (agent.spec.game_id != 'poker' or agent.encoder.encoding_id != ENCODING_ID
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('requires a poker policy and a positive temperature')

    def policy(obs):
        if obs['variant'] != agent.spec.rules.variant:
            raise ValueError('policy variant mismatch')
        planes = np.asarray(encode_observation(obs), dtype=np.float32)
        with torch.inference_mode():
            logits, _ = agent.network(torch.as_tensor(planes[None], device=agent.device))
            legal = obs['legal_actions']
            probs = torch.softmax(logits[0, legal].float() / temperature, dim=0).cpu().tolist()
        # Float32 softmax has rounding error; the exact evaluator requires a probability simplex.
        total = sum(probs)
        return {a: p / total for a, p in zip(legal, probs)}

    return policy
