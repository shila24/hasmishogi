"""Seeded heads-up limit Hold'em and Leduc with immutable undo snapshots."""

from dataclasses import asdict, dataclass, replace
import random

from .evaluator import rank7


def _seat(player):
    if type(player) is not int or player not in (0, 1):
        raise ValueError('player must be 0 or 1')
    return player


def _seed(seed):
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    return seed


@dataclass(frozen=True)
class PokerRules:
    variant: str = 'hulhe'

    def __post_init__(self):
        if self.variant not in ('hulhe', 'leduc'):
            raise ValueError('variant must be hulhe or leduc')

    @property
    def ruleset_id(self):
        return 'hulhe-v1' if self.variant == 'hulhe' else 'leduc-v1'

    @property
    def deck_size(self):
        return 52 if self.variant == 'hulhe' else 6

    @property
    def hole_count(self):
        return 2 if self.variant == 'hulhe' else 1

    @property
    def max_contribution(self):
        return 48 if self.variant == 'hulhe' else 13

    @property
    def bet_cap(self):
        return 4 if self.variant == 'hulhe' else 2

    @property
    def streets(self):
        return 4 if self.variant == 'hulhe' else 2

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Outcome:
    winner: int = -2
    reason: str = 'ongoing'
    chips: tuple[float, float] = (0, 0)

    @property
    def terminal(self):
        return self.winner != -2


@dataclass(frozen=True)
class Position:
    street: int
    turn: int
    contributions: tuple[int, int]
    street_contributions: tuple[int, int]
    acted: tuple[bool, bool]
    bets: int
    board: tuple[int, ...] = ()
    outcome: Outcome = Outcome()


class PokerGame:
    """One hand; chips have no stack cap and no carry-over between hands.

    Leduc uses physical card ids 0..5 (rank = card // 2, J/Q/K),
    antes of one, and button-first action on both rounds. There are no burns.
    Engine state and snapshots are immutable values; public lists are copies.
    """

    def __init__(self, rules=None, seed=None, button=None):
        if rules is None:
            rules = PokerRules()
        elif isinstance(rules, dict):
            if set(rules) - {'variant'}:
                raise ValueError('unknown poker rule')
            rules = PokerRules(**rules)
        if not isinstance(rules, PokerRules):
            raise ValueError('rules must be PokerRules or a rule dictionary')
        self.rules = rules
        self.seed = random.SystemRandom().getrandbits(63) if seed is None else _seed(seed)
        self.button = self.seed % 2 if button is None else _seat(button)
        deck = list(range(rules.deck_size))
        random.Random(self.seed).shuffle(deck)
        self._initialize(tuple(deck))

    def _initialize(self, deck):
        self._deck = tuple(deck)
        self._holes = tuple(tuple(self._deck[i] for i in range(2 * self.rules.hole_count)
                                  if (self.button + i) % 2 == p) for p in (0, 1))
        contributions = [1, 1]
        street_contributions = [0, 0]
        bets = 0
        if self.rules.variant == 'hulhe':
            contributions[1 - self.button] = 2
            street_contributions = contributions.copy()
            bets = 1
        self._states = [Position(0, self.button, tuple(contributions), tuple(street_contributions),
                                 (False, False), bets)]
        self._moves = []
        self._events = []

    @property
    def initial(self):
        return {'button': self.button}

    @property
    def position(self):
        return self._states[-1]

    @property
    def street(self):
        return self.position.street

    @property
    def contributions(self):
        return list(self.position.contributions)

    @property
    def pot(self):
        return sum(self.position.contributions)

    @property
    def outcome(self):
        return self.position.outcome

    @property
    def moves(self):
        return list(self._moves)

    @property
    def bet_size(self):
        return (2 if self.street < 2 else 4) if self.rules.variant == 'hulhe' else (2, 4)[self.street]

    def actors(self):
        return [] if self.outcome.terminal else [self.position.turn]

    def to_call(self, player):
        _seat(player)
        return max(self.position.street_contributions) - self.position.street_contributions[player]

    def legal_actions(self, player=None):
        player = self.position.turn if player is None else _seat(player)
        if self.outcome.terminal or player != self.position.turn:
            return []
        actions = [0, 1] if self.to_call(player) else [1]
        if self.position.bets < self.rules.bet_cap:
            actions.append(2)
        return actions

    def _finish(self, state, winner, reason):
        if winner == -1:
            chips = tuple(sum(state.contributions) / 2 - c for c in state.contributions)
        else:
            chips = tuple(sum(state.contributions) - c if p == winner else -c
                          for p, c in enumerate(state.contributions))
        return replace(state, outcome=Outcome(winner, reason, chips))

    def _showdown(self, state):
        if self.rules.variant == 'hulhe':
            ranks = [rank7(hole + state.board) for hole in self._holes]
        else:
            board = state.board[0] // 2
            ranks = [(hole[0] // 2 == board, hole[0] // 2) for hole in self._holes]
        winner = -1 if ranks[0] == ranks[1] else int(ranks[1] > ranks[0])
        return self._finish(state, winner, 'showdown')

    def play(self, action):
        if type(action) is not int or action not in self.legal_actions():
            raise ValueError('illegal poker action')
        old = self.position
        player = old.turn
        state = replace(old, turn=1 - player)
        if action == 0:
            state = self._finish(state, 1 - player, 'fold')
        else:
            payment = self.to_call(player) + (self.bet_size if action == 2 else 0)
            total, street = list(old.contributions), list(old.street_contributions)
            total[player] += payment
            street[player] += payment
            acted = list(old.acted)
            acted[player] = True
            state = replace(state, contributions=tuple(total), street_contributions=tuple(street),
                            acted=tuple(acted), bets=old.bets + int(action == 2))
            if action == 1 and all(acted) and street[0] == street[1]:
                if old.street == self.rules.streets - 1:
                    state = self._showdown(state)
                else:
                    new_street = old.street + 1
                    count = (3, 4, 5)[new_street - 1] if self.rules.variant == 'hulhe' else 1
                    start = 2 * self.rules.hole_count
                    first = 1 - self.button if self.rules.variant == 'hulhe' else self.button
                    state = replace(state, street=new_street, turn=first, street_contributions=(0, 0),
                                    acted=(False, False), bets=0, board=self._deck[start:start + count])
        self._states.append(state)
        self._moves.append(action)
        self._events.append((old.street, player, action))
        return self.outcome

    def play_joint(self, actions):
        if (not isinstance(actions, dict) or len(actions) != 1
                or any(type(p) is not int for p in actions) or set(actions) != set(self.actors())):
            raise ValueError('joint actions must match the current actor')
        return self.play(next(iter(actions.values())))

    def undo(self):
        if not self._moves:
            return False
        self._states.pop()
        self._events.pop()
        self._moves.pop()
        return True

    def clone(self):
        work = object.__new__(type(self))
        work.__dict__ = self.__dict__.copy()
        work._states = self._states.copy()
        work._moves = self._moves.copy()
        work._events = self._events.copy()
        return work

    def history_keys(self):
        """Full-state keys, including the deck; never an agent observation."""
        return [(self.rules.variant, self.button, self._deck, state) for state in self._states]

    def observation(self, player):
        _seat(player)
        history = [[] for _ in range(self.rules.streets)]
        for street, actor, action in self._events:
            history[street].append({'player': actor, 'action': action})
        return dict(player=player, variant=self.rules.variant, hole_cards=list(self._holes[player]),
                    board=list(self.position.board), history=history, pot=self.pot,
                    contributions=self.contributions, button=self.button, street=self.street,
                    to_act=None if self.outcome.terminal else self.position.turn,
                    street_contributions=list(self.position.street_contributions), bets=self.position.bets,
                    bet_size=self.bet_size, to_call=self.to_call(player), legal_actions=self.legal_actions(player))

    def redeal(self, hidden: dict, seed: int):
        """Atomically replace hidden cards, rebuilding every undo snapshot.

        Required key: ``player`` (observer). Optional ``hole_cards`` is a pair
        of hands, ``board`` the already dealt board, and ``deck`` the entire
        undealt tail in draw order. Omitted hidden slots are sampled uniformly.
        Specified visible cards must exactly match the observation. Sampled
        games have no replay seed: a seed-only record must not misrepresent them.
        """
        _seed(seed)
        if (not isinstance(hidden, dict) or 'player' not in hidden
                or set(hidden) - {'player', 'hole_cards', 'board', 'deck'} or self.outcome.terminal):
            raise ValueError('redeal requires a nonterminal observer assignment')
        player = _seat(hidden['player'])
        before = self.observation(player)
        deck = [None] * self.rules.deck_size
        for i in range(2 * self.rules.hole_count):
            if (self.button + i) % 2 == player:
                deck[i] = self._deck[i]
        start = 2 * self.rules.hole_count
        end = start + len(self.position.board)
        deck[start:end] = self.position.board
        if 'board' in hidden:
            board = hidden['board']
            if (not isinstance(board, (list, tuple)) or any(type(c) is not int for c in board)
                    or tuple(board) != self.position.board):
                raise ValueError('cannot change public board')
        if 'hole_cards' in hidden:
            holes = hidden['hole_cards']
            if (not isinstance(holes, (list, tuple)) or len(holes) != 2
                    or any(not isinstance(h, (list, tuple)) or len(h) != self.rules.hole_count for h in holes)
                    or any(type(c) is not int for h in holes for c in h)
                    or tuple(holes[player]) != self._holes[player]):
                raise ValueError('cannot change observer cards or hand sizes')
            for i in range(start):
                deck[i] = holes[(self.button + i) % 2][i // 2]
        if 'deck' in hidden:
            tail = hidden['deck']
            if (not isinstance(tail, (list, tuple)) or len(tail) != len(deck) - end
                    or any(type(c) is not int for c in tail)):
                raise ValueError('invalid undealt deck length')
            deck[end:] = tail
        fixed = [c for c in deck if c is not None]
        if (any(type(c) is not int or not 0 <= c < len(deck) for c in fixed)
                or len(set(fixed)) != len(fixed)):
            raise ValueError('invalid or repeated cards')
        unseen = [c for c in range(len(deck)) if c not in fixed]
        random.Random(seed).shuffle(unseen)
        for i, card in enumerate(deck):
            if card is None:
                deck[i] = unseen.pop()
        work = self.clone()
        work._initialize(deck)
        for action in self._moves:
            work.play(action)
        if work.observation(player) != before:
            raise ValueError('redeal changed public or observer state')
        work.seed = None
        self.__dict__.update(work.__dict__)

    def close(self):
        """No external resources are owned by the Python engine."""


def create_game(rules=None, seed=None, button=None):
    return PokerGame(rules, seed, button)


def replay(rules, seed, button, actions):
    game = create_game(rules, seed, button)
    for action in actions:
        game.play(action)
    return game
