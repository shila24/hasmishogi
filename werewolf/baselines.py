"""Observation-only baselines; claims are evidence, never privileged knowledge."""

from collections import Counter
from dataclasses import dataclass
import random

from .actions import target_action


@dataclass(frozen=True)
class Decision:
    action: int


class BaselineAgent:
    def __init__(self, name='simple', seed=0):
        if name not in ('random', 'simple'):
            raise ValueError('unknown werewolf baseline')
        self.name, self.rng = name, random.Random(seed)

    def choose(self, game, limits=None, cancel=None, *, player=None):
        if cancel is not None and cancel.is_set():
            raise InterruptedError('cancelled')
        if player is None:
            actors = game.actors()
            if len(actors) != 1:
                raise ValueError('specify player for a simultaneous decision')
            player = actors[0]
        obs = game.observation(player)
        legal = obs['legal_actions']
        if not legal:
            raise ValueError('no legal actions')
        if self.name == 'random':
            return Decision(self.rng.choice(legal))
        action = self._simple(obs)
        if action not in legal:
            raise ValueError('baseline proposed an illegal action')
        return Decision(action)

    def _suspect(self, obs, candidates):
        reports = Counter(e['target'] for e in obs['public_log']
                          if e['kind'] == 'DIVINED' and e['result'] == 'werewolf')
        intents = {}
        for event in obs['public_log']:
            if event['kind'] == 'VOTE_INTENT' and event['day'] == obs['day']:
                intents[event['player']] = event['target']
        counts = reports if any(reports[p] for p in candidates) else Counter(intents.values())
        best = max(counts[p] for p in candidates)
        return self.rng.choice([p for p in candidates if counts[p] == best])

    def _simple(self, obs):
        player, role = obs['player'], obs['role']
        others = [p for p in obs['alive'] if p != player]
        own = [e for e in obs['public_log'] if e.get('player') == player]
        if obs['phase'] in ('vote', 'revote'):
            candidates = [p for p in others if obs['phase'] != 'revote' or p in obs['tied']]
            return target_action('vote', player, self._suspect(obs, candidates))
        if obs['phase'] == 'night':
            if role == 'werewolf':
                claims = {}
                for event in obs['public_log']:
                    if event['kind'] == 'COMINGOUT':
                        claims[event['player']] = event['role']
                seers = [p for p in others if claims.get(p) == 'seer']
                return target_action('attack', player, self.rng.choice(seers or others))
            known = {r['target'] for r in obs['results']}
            return target_action('divine', player, self.rng.choice([p for p in others if p not in known] or others))
        claims = [e for e in own if e['kind'] == 'COMINGOUT']
        if role == 'seer':
            if not claims:
                return 2
            for result in obs['results']:
                if not any(e['kind'] == 'DIVINED' and e['target'] == result['target']
                           and e['result'] == result['result'] for e in own):
                    return target_action('DIVINED', player, result['target'], result['result'])
        if role in ('werewolf', 'possessed') and obs['day'] == 1:
            if obs['talk_turn'] == 0 and (role == 'possessed' or self.rng.random() < 0.5):
                return 2
            if (claims and claims[-1]['role'] == 'seer'
                    and not any(e['kind'] == 'DIVINED' for e in own)):
                # The possessed cannot know which other seat is the wolf.
                return target_action('DIVINED', player, self.rng.choice(others), 'werewolf')
        return target_action('VOTE_INTENT', player, self._suspect(obs, others))
