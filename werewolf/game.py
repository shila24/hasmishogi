"""Seeded simultaneous five-seat village with reversible full history.

Public statements are untrusted claims. Night decisions and role assignments
remain private, including after death. A redealt simulation has no replay seed;
its synthetic hidden history must never masquerade as an original deal record.
"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import random

from .actions import ROLE_DECK, decode, target_action, validate_player

RULESET_ID = 'werewolf-5p-v1'


@dataclass(frozen=True)
class Rules:
    talk_turns: int = 3
    max_days: int = 5
    ruleset_id: str = RULESET_ID

    def __post_init__(self):
        if self.ruleset_id != RULESET_ID:
            raise ValueError('unsupported werewolf ruleset')
        for name in ('talk_turns', 'max_days'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Outcome:
    winner_team: str | None = None
    reason: str = 'ongoing'

    @property
    def terminal(self):
        return self.reason != 'ongoing'

    @property
    def winner(self):
        """No unique winning seat: -2, or -1 for an actual terminal draw.

        Consumers must use winner_team or fixed-seat rewards for team games.
        """
        return -1 if self.terminal and self.winner_team is None else -2


def team(role: str) -> str:
    return 'werewolf' if role in ('werewolf', 'possessed') else 'village'


def role_class(role: str) -> str:
    return 'werewolf' if role == 'werewolf' else 'human'


def death_outcome(roles, alive):
    if not alive[roles.index('werewolf')]:
        return Outcome('village', 'werewolf_executed')
    if sum(alive) <= 2:
        return Outcome('werewolf', 'humans_outnumbered')
    return Outcome()


class WerewolfGame:
    def __init__(self, rules=None, seed=None):
        if rules is not None and not isinstance(rules, (Rules, dict)):
            raise ValueError('rules must be Rules or a dictionary')
        self.rules = Rules(**rules) if isinstance(rules, dict) else rules or Rules()
        if seed is not None and type(seed) is not int:
            raise ValueError('seed must be an integer or None')
        self.seed = random.SystemRandom().randrange(2**63) if seed is None else seed
        rng = random.Random(self.seed)
        roles = list(ROLE_DECK)
        rng.shuffle(roles)
        self._initialize(roles, rng)

    def _initialize(self, roles, rng, initial_target=None):
        self.roles = list(roles)
        self.alive = [True] * 5
        self.day, self.phase, self.talk_turn = 1, 'talk', 0
        self.tied = []
        self.outcome = Outcome()
        self.initial = {}
        self.public_log = []
        self._results = [[] for _ in range(5)]
        seer = self.roles.index('seer')
        target = rng.choice([p for p in range(5) if p != seer]) if initial_target is None else initial_target
        self._divine(seer, target, 0)
        self._rng_state = rng.getstate()
        self._moves, self._steps, self._undo = [], [], []
        self._keys = [self._key()]

    @property
    def moves(self):
        return copy.deepcopy(self._moves)

    def actors(self):
        if self.outcome.terminal:
            return []
        return [p for p in range(5) if self.alive[p] and
                (self.phase != 'night' or self.roles[p] in ('seer', 'werewolf'))]

    def legal_actions(self, player):
        validate_player(player)
        if player not in self.actors():
            return []
        if self.phase == 'talk':
            return [a for a in range(33) if a < 29 or self.alive[decode(a, player)['target']]]
        kind = ('attack' if self.roles[player] == 'werewolf' else 'divine') if self.phase == 'night' else 'vote'
        targets = self.tied if self.phase == 'revote' else range(5)
        return sorted(target_action(kind, player, t) for t in targets if self.alive[t] and t != player)

    def observation(self, player):
        validate_player(player)
        return {'player': player, 'role': self.roles[player], 'results': copy.deepcopy(self._results[player]),
                'day': self.day, 'phase': self.phase, 'talk_turn': self.talk_turn,
                'alive': [p for p in range(5) if self.alive[p]], 'tied': list(self.tied),
                'public_log': copy.deepcopy(self.public_log), 'legal_actions': self.legal_actions(player)}

    def _divine(self, player, target, day):
        self._results[player].append({'day': day, 'target': target, 'result': role_class(self.roles[target])})

    def _snapshot(self):
        return copy.deepcopy({k: v for k, v in self.__dict__.items() if k != '_undo'})

    def play_joint(self, actions):
        self._play(actions)
        return self.outcome

    def _play(self, actions, tie_target=None):
        if (self.outcome.terminal or not isinstance(actions, dict)
                or any(type(p) is not int for p in actions) or set(actions) != set(self.actors())
                or any(type(a) is not int or a not in self.legal_actions(p) for p, a in actions.items())):
            raise ValueError('joint actions must provide exactly one legal action per actor')
        snapshot = self._snapshot()
        phase, start = self.phase, len(self.public_log)
        actions = dict(sorted(actions.items()))
        if phase == 'talk':
            self.public_log.extend({'day': self.day, 'turn': self.talk_turn, 'player': p,
                                    'action': a, **decode(a, p)} for p, a in actions.items())
            self.talk_turn += 1
            if self.talk_turn == self.rules.talk_turns:
                self.phase = 'vote'
        elif phase in ('vote', 'revote'):
            votes = {p: decode(a, p)['target'] for p, a in actions.items()}
            self.public_log.append({'kind': phase, 'day': self.day, 'votes': votes})
            counts = Counter(votes.values())
            tied = sorted(p for p, n in counts.items() if n == max(counts.values()))
            if len(tied) > 1 and phase == 'vote':
                self.tied, self.phase = tied, 'revote'
            else:
                rng = random.Random()
                rng.setstate(self._rng_state)
                target = rng.choice(tied) if len(tied) > 1 else tied[0]
                if tie_target is not None:
                    if tie_target not in tied:
                        raise ValueError('inconsistent public execution')
                    target = tie_target
                self._rng_state = rng.getstate()
                self.alive[target] = False
                self.public_log.append({'kind': 'execution', 'day': self.day, 'target': target})
                self.tied = []
                self.outcome = death_outcome(self.roles, self.alive)
                self.phase = 'terminal' if self.outcome.terminal else 'night'
        else:
            for p, a in actions.items():
                if self.roles[p] == 'seer':
                    self._divine(p, decode(a, p)['target'], self.day)
            wolf = self.roles.index('werewolf')
            target = decode(actions[wolf], wolf)['target']
            self.alive[target] = False
            self.public_log.append({'kind': 'attack', 'day': self.day, 'target': target})
            self.outcome = death_outcome(self.roles, self.alive)
            if not self.outcome.terminal and self.day >= self.rules.max_days:
                self.outcome = Outcome(None, 'max_days')
            if self.outcome.terminal:
                self.phase = 'terminal'
            else:
                self.day += 1
                self.phase, self.talk_turn = 'talk', 0
        self._moves.append(actions)
        self._steps.append({'phase': phase, 'events': copy.deepcopy(self.public_log[start:])})
        self._keys.append(self._key())
        self._undo.append(snapshot)

    def _key(self):
        data = (self.roles, self.alive, self.day, self.phase, self.talk_turn, self.tied,
                asdict(self.outcome), self._results, self.public_log, self._rng_state)
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def history_keys(self):
        return tuple(self._keys)

    def undo(self):
        if not self._undo:
            raise ValueError('no joint action to undo')
        snapshot = self._undo.pop()
        history = self._undo
        self.__dict__.clear()
        self.__dict__.update(snapshot)
        self._undo = history

    def clone(self):
        return copy.deepcopy(self)

    def close(self):
        """The pure Python engine owns no external resources."""

    def consistent_roles(self, roles, player=None):
        """Check hard public facts, plus an explicitly named observer's private facts.

        Claims never constrain roles. Public deaths and continued play constrain
        the wolf even though executed roles are never revealed.
        """
        if (not isinstance(roles, (list, tuple)) or any(type(role) is not str for role in roles)
                or Counter(roles) != Counter(ROLE_DECK)):
            return False
        if player is not None:
            validate_player(player)
            if roles[player] != self.roles[player]:
                return False
            if any(role_class(roles[r['target']]) != r['result'] for r in self._results[player]):
                return False
        alive = [True] * 5
        deaths = [e for e in self.public_log if e['kind'] in ('execution', 'attack')]
        for i, event in enumerate(deaths):
            target = event['target']
            if event['kind'] == 'attack' and roles[target] == 'werewolf':
                return False
            alive[target] = False
            result = death_outcome(roles, alive)
            if result.terminal and i != len(deaths) - 1:
                return False
            if i == len(deaths) - 1:
                if self.outcome.reason == 'max_days':
                    if result.terminal:
                        return False
                elif result != self.outcome:
                    return False
        return True

    def role_assignments(self, player):
        """Enumerate the 60 distinct deals using only public and observer facts."""
        validate_player(player)
        return [roles for roles in sorted(set(itertools.permutations(ROLE_DECK)))
                if self.consistent_roles(roles, player)]

    def redeal(self, roles, seed, *, player=None):
        """Atomically replace hidden state and resample all future chance.

        Without ``player`` only public facts are fixed. Advice always supplies
        its observer. Hidden historical night actions are regenerated, never
        used to infer the original wolf or seer. Public random executions stay
        fixed, while subsequent random ties use the supplied seed.
        """
        if type(seed) is not int or not self.consistent_roles(roles, player):
            raise ValueError('inconsistent role assignment or invalid seed')
        observed = self.observation(player) if player is not None else None
        rng = random.Random(seed)
        work = WerewolfGame(self.rules, seed)
        work.seed = None
        target = self._results[player][0]['target'] if player is not None and roles[player] == 'seer' else None
        work._initialize(roles, rng, target)
        for step, old_actions in zip(self._steps, self._moves):
            events = step['events']
            if step['phase'] == 'night':
                attacked = next(e['target'] for e in events if e['kind'] == 'attack')
                actions = {}
                for seat in work.actors():
                    if roles[seat] == 'werewolf':
                        actions[seat] = target_action('attack', seat, attacked)
                    elif seat == player:
                        result = next(r for r in self._results[player] if r['day'] == work.day)
                        actions[seat] = target_action('divine', seat, result['target'])
                    else:
                        actions[seat] = rng.choice(work.legal_actions(seat))
            else:
                actions = old_actions
            executed = next((e['target'] for e in events if e['kind'] == 'execution'), None)
            work._play(actions, executed)
        if (work.public_log != self.public_log or work.outcome != self.outcome
                or (player is not None and work.observation(player) != observed)):
            raise ValueError('assignment cannot reproduce the observed history')
        work._rng_state = rng.getstate()
        work._keys[-1] = work._key()
        self.__dict__.clear()
        self.__dict__.update(work.__dict__)


def create_game(rules=None, seed=None):
    return WerewolfGame(rules, seed)


def replay(rules, seed, joint_actions):
    game = create_game(rules, seed)
    for actions in joint_actions:
        game.play_joint(actions)
    return game
