"""GameSpec v2 adapter and observation-only werewolf-enc-v1 planes.

DIVINED uses four pairs of planes: speaker-relative target offset 1..4,
then human/werewolf. Cells are observer-relative claimer row/target column.
All eight planes are used without discarding contradictory public claims.
"""

import math

import numpy as np

from hasami.agents import AgentCancelled, Decision
from werewolf import Rules, create_game
from werewolf.actions import ACTION_SIZE, ROLES, action_name
from werewolf.baselines import BaselineAgent
from werewolf.evaluation import evaluate
from werewolf.game import RULESET_ID, team
from ..registry import ENCODERS, GAMES
from .base import StepOutcome

ENCODING_ID = 'werewolf-enc-v1'
BOARD_SHAPE = (5, 8)
NUM_PLANES = 32


class PlaneEncoder:
    encoding_id, num_planes, action_size = ENCODING_ID, NUM_PLANES, ACTION_SIZE

    def __init__(self, **params):
        if params:
            raise ValueError('werewolf-enc-v1 takes no parameters')

    @staticmethod
    def _player(game, player):
        if player is None:
            actors = game.actors()
            if len(actors) != 1:
                raise ValueError('specify player for simultaneous or terminal observations')
            return actors[0]
        return player

    def encode(self, game, player=None):
        player = self._player(game, player)
        obs = game.observation(player)
        planes = np.zeros((NUM_PLANES, *BOARD_SHAPE), dtype=np.float32)
        row = lambda p: (p - player) % 5
        planes[ROLES.index(obs['role'])] = 1
        for p in obs['alive']:
            planes[4, row(p)] = 1
        for result in obs['results']:
            planes[5 + (result['result'] == 'werewolf'), row(result['target'])] = 1
        if obs['phase'] != 'terminal':
            planes[7 + ('talk', 'vote', 'revote', 'night').index(obs['phase'])] = 1
        planes[11] = obs['day'] / game.rules.max_days
        claims, estimates, intents, votes = {}, {}, {}, {}
        for event in obs['public_log']:
            kind = event['kind']
            if kind == 'COMINGOUT':
                claims[event['player']] = event['role']
            elif kind == 'DIVINED':
                offset = (event['target'] - event['player']) % 5 - 1
                plane = 16 + 2 * offset + (event['result'] == 'werewolf')
                planes[plane, row(event['player']), row(event['target'])] = 1
            elif kind == 'ESTIMATE':
                estimates[event['player'], event['target']] = event['role']
            elif kind == 'VOTE_INTENT':
                intents[event['player']] = event['target']
            elif kind in ('vote', 'revote'):
                votes = event['votes']
            elif kind in ('execution', 'attack'):
                planes[30, row(event['target'])] = 1
        for p, role in claims.items():
            planes[12 + ROLES.index(role), row(p)] = 1
        for (_, target), role in estimates.items():
            planes[24 + ROLES.index(role), row(target)] += 0.25
        for index, mapping in ((28, intents), (29, votes)):
            for p, target in mapping.items():
                planes[index, row(p), row(target)] = 1
        planes[31] = 1
        return planes

    def legal_mask(self, game, player=None):
        player = self._player(game, player)
        mask = np.zeros(ACTION_SIZE, dtype=np.float32)
        mask[game.legal_actions(player)] = 1
        return mask


ENCODERS.register(ENCODING_ID, PlaneEncoder)


class WerewolfBaselineAgent(BaselineAgent):
    def choose(self, game, limits=None, cancel=None, *, player=None):
        try:
            decision = super().choose(game, limits, cancel, player=player)
        except InterruptedError as exc:
            raise AgentCancelled(str(exc)) from exc
        return Decision(decision.action)


@GAMES.register('werewolf')
class WerewolfSpec:
    game_id, ruleset_id = 'werewolf', RULESET_ID
    board_shape, action_size = BOARD_SHAPE, ACTION_SIZE
    action_encoding = 'werewolf-relative-45-v1'
    default_encoder = ENCODING_ID
    default_max_plies = 64
    num_players = 5
    perfect_information, zero_sum = False, False
    has_chance, supports_undo = True, True
    default_baselines = ('random', 'simple')
    default_teacher = 'simple'
    draw_reasons = ()

    def __init__(self, **params):
        self.rules = Rules(**params)
        self.default_max_plies = self.rules.max_days * (self.rules.talk_turns + 3)

    def create_game(self, backend='python', rules=None, position=None, seed=None):
        self.resolve_backend(backend)
        if position not in (None, {}):
            raise ValueError('werewolf starts from a seed; replay joint actions for a later state')
        if isinstance(rules, dict):
            rules = Rules(**{**self.rules.to_dict(), **rules})
        return create_game(self.rules if rules is None else rules, seed)

    def resolve_backend(self, backend):
        if backend not in ('python', 'auto'):
            raise ValueError('werewolf supports only the Python backend')
        return 'python'

    def rules_dict(self, game):
        return game.rules.to_dict()

    def actors(self, game):
        return game.actors()

    def legal_actions(self, game, player):
        return game.legal_actions(player)

    def legal_mask(self, game, player):
        return PlaneEncoder().legal_mask(game, player)

    def observe(self, game, player):
        return PlaneEncoder().encode(game, player)

    def step(self, game, actions):
        game.play_joint(actions)
        return StepOutcome(game.outcome.terminal, False,
                           self.terminal_values(game) if game.outcome.terminal else None,
                           {'reason': game.outcome.reason})

    def terminal_values(self, game):
        if not game.outcome.terminal:
            raise ValueError('terminal_values requires a terminal game')
        winner = game.outcome.winner_team
        return [0.0 if winner is None else 1.0 if team(role) == winner else -1.0 for role in game.roles]

    def make_encoder(self, name='', **params):
        if name not in ('', ENCODING_ID):
            raise ValueError('unknown werewolf encoder')
        return PlaneEncoder(**params)

    def symmetries(self):
        return []

    def baseline_agent(self, spec, seed=0):
        return WerewolfBaselineAgent(spec, seed), None

    def action_name(self, action):
        return action_name(action)

    def determinize(self, game, player, rng):
        assignments = game.role_assignments(player)
        if not assignments:
            raise ValueError('no roles consistent with the observer')
        roles = assignments[int(rng.integers(len(assignments)))]
        work = game.clone()
        try:
            work.redeal(list(roles), int(rng.integers(2**63)), player=player)
            return work
        except BaseException:
            work.close()
            raise

    @staticmethod
    def outcome_score(values, player):
        if (type(player) is not int or not 0 <= player < 5 or len(values) != 5
                or any(not math.isfinite(v) or not -1 <= v <= 1 for v in values)):
            raise ValueError('expected five finite bounded seat values and a valid player')
        return (values[player] + 1) / 2

    def strength_table(self, level_agents, anchors=('random', 'simple'), games=100, seed=0):
        """Measure focal-seat team scores; calibration is left to the caller."""
        return {'game_id': self.game_id, 'games_per_anchor': games, 'seed': seed,
                'levels': {str(level): {anchor: evaluate(agent, anchor, games, seed, self.rules)
                                       for anchor in anchors} for level, agent in level_agents.items()}}
