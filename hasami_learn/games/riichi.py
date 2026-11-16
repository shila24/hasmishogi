"""GameSpec v2 adapter for seeded four-player Riichi Mahjong."""

import copy
from dataclasses import asdict, fields
import importlib
import json
import numpy as np

from hasami.agents import AgentCancelled, Decision
from riichi.actions import (ACTION_SIZE, CHI_HIGH, CHI_LOW, CHI_MID, KAN, KAN_BASE, NINE_TERMINALS,
                            PASS, PON, RIICHI, RON, TSUMO, action_name)
from riichi.baselines import BaselineAgent
from riichi.encoding import BOARD_SHAPE, ENCODING_ID, NUM_PLANES, encode
from riichi.game import RiichiGame
from riichi.rules import RiichiRules
from ..history import HistoryEncoder
from ..registry import ENCODERS, GAMES
from .base import StepOutcome

HISTORY_ENCODING_ID = 'riichi-history'
DEFAULT_HISTORY_STEPS = 16  # two full turns of the table (four discards and four claim phases each)


class PlaneEncoder:
    encoding_id = ENCODING_ID
    num_planes = NUM_PLANES
    action_size = ACTION_SIZE

    def __init__(self, **params):
        if params:
            raise ValueError(f'{ENCODING_ID} takes no parameters')

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
        if hasattr(game, 'encode_into'):
            # Native backend: the engine writes riichi-enc-v1 planes straight into the buffer.
            planes = np.empty((NUM_PLANES, *BOARD_SHAPE), dtype=np.float32)
            game.encode_into(player, planes)
            return planes
        return np.array(encode(game, player), dtype=np.float32)

    def legal_mask(self, game, player=None):
        player = self._player(game, player)
        mask = np.zeros(ACTION_SIZE, dtype=np.float32)
        if hasattr(game, 'legal_mask_into'):
            game.legal_mask_into(player, mask)
            return mask
        mask[game.legal_actions(player)] = 1.0
        return mask


ENCODERS.register(ENCODING_ID, PlaneEncoder)


class RiichiActionLayout:
    """Two planes per joint step: the named tile at ``[actor row, tile]`` for discards, riichi
    discards and kans of a tile, and the action kind one-hot at ``[actor row, kind]``."""

    planes_per_step = 2
    KINDS = ('discard', 'riichi', 'chi_low', 'chi_mid', 'chi_high', 'pon', 'kan', 'ron', 'tsumo',
             'pass', 'nine_terminals')
    _KIND_OF = {CHI_LOW: KINDS.index('chi_low'), CHI_MID: KINDS.index('chi_mid'),
                CHI_HIGH: KINDS.index('chi_high'), PON: KINDS.index('pon'), KAN: KINDS.index('kan'),
                RON: KINDS.index('ron'), TSUMO: KINDS.index('tsumo'), PASS: KINDS.index('pass'),
                NINE_TERMINALS: KINDS.index('nine_terminals')}

    def render(self, planes, row, action):
        if not 0 <= action < ACTION_SIZE:
            raise ValueError(f'action {action} outside the riichi action table')
        if action < CHI_LOW:  # 0..67: discard, or discard with riichi
            planes[0, row, action % 34] = 1.0
            kind = self.KINDS.index('riichi' if action >= RIICHI else 'discard')
        elif KAN_BASE <= action < RON:
            planes[0, row, action - KAN_BASE] = 1.0
            kind = self.KINDS.index('kan')
        else:
            kind = self._KIND_OF[action]
        planes[1, row, kind] = 1.0


def current_hand_history(game):
    """Joint steps of the hand in progress; earlier hands are public but stale for the stack."""
    return game.moves[game.hand_start:]


def make_history_encoder(k=DEFAULT_HISTORY_STEPS, **params):
    """``riichi-history``: the 120 base planes plus ``2 * k`` planes for the last ``k`` joint steps."""
    if params:
        raise ValueError(f'{HISTORY_ENCODING_ID} accepts only k, got {sorted(params)}')
    return HistoryEncoder(PlaneEncoder(), k=k, layout=RiichiActionLayout(), num_players=4,
                          encoding_id=HISTORY_ENCODING_ID, history=current_hand_history)


ENCODERS.register(HISTORY_ENCODING_ID, make_history_encoder)


class RiichiBaselineAgent(BaselineAgent):
    """Translate the dependency-free baseline to the learner's Agent contract."""

    def choose(self, game, limits=None, cancel=None, *, player=None):
        try:
            decision = super().choose(game, limits, cancel, player=player)
        except InterruptedError as exc:
            raise AgentCancelled(str(exc)) from exc
        return Decision(**asdict(decision))


def _native():
    """The ctypes bridge riichi.native (riichi/native.py), imported lazily."""
    return importlib.import_module('riichi.native')


def _native_available() -> bool:
    try:
        return bool(_native().available())
    except (ImportError, OSError, RuntimeError, AttributeError):
        return False


def sample_riichi_belief(game, player, rng, belief=None):
    """Replace all private opponent state on an owned clone using only observation.

    ``belief['hand']`` is a nonnegative (3, 34) tile distribution, relative to the
    observer. Physical tile sampling is without replacement. Public pending kan
    tiles are reserved before filling hands; exposed indicators remain fixed.
    """
    obs = game.observation(player)
    weights = np.ones((3, 34), dtype=np.float64)
    if belief and "hand" in belief:
        weights = np.array(belief["hand"], dtype=np.float64, copy=True)
        if weights.shape != (3, 34) or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("hand belief must be a finite nonnegative (3, 34) array")
    known = set(obs["hand"]) | set(obs["dora_indicators"])
    for row in range(4):
        known.update(d["tile"] for d in obs["discards"][row])
        known.update(tile for meld in obs["melds"][row] for tile in meld["tiles"])
    unseen = [tile for tile in range(136) if tile not in known]
    hands = [None] * 4
    hands[player] = list(obs["hand"])
    pending = obs["pending_kan"]
    for row in range(1, 4):
        seat = (player + row) % 4
        hands[seat] = []
        if pending is not None and pending[0] == row:
            needed = 4 if pending[1] == "closed" else 1
            choices = [tile for tile in unseen if tile // 4 == pending[2]]
            if len(choices) < needed:
                raise ValueError("public pending kan contradicts visible tiles")
            for tile in choices[:needed]:
                hands[seat].append(tile)
                unseen.remove(tile)
    # Randomize opponent allocation order to avoid always favouring the next seat.
    for row in rng.permutation([1, 2, 3]):
        seat = (player + int(row)) % 4
        size = 13 - 3 * len(obs["melds"][row])
        if row == obs["turn"] and obs["phase"] in ("draw", "discard", "kan_claim"):
            size += 1
        while len(hands[seat]) < size:
            mass = np.array([weights[row - 1, tile // 4] for tile in unseen]) + 1e-8
            index = int(rng.choice(len(unseen), p=mass / mass.sum()))
            hands[seat].append(unseen.pop(index))
        hands[seat].sort()
    rng.shuffle(unseen)
    dead = [None] * 14
    for index, tile in enumerate(obs["dora_indicators"]):
        dead[4 + 2 * index] = tile
    for index in range(14):
        if dead[index] is None:
            dead[index] = unseen.pop()
    if len(unseen) != obs["remaining_wall"]:
        raise ValueError("sampled tile counts do not match the public remaining wall")
    temporary = [False] * 4
    riichi_flags = [False] * 4
    temporary[player] = bool(obs["furiten"][1])
    riichi_flags[player] = bool(obs["furiten"][2])
    drawn = [None] * 4
    drawn[player] = obs["drawn_tile"]
    turn = (player + obs["turn"]) % 4
    if turn != player and obs["phase"] in ("draw", "kan_claim"):
        choices = hands[turn]
        if pending is not None:
            choices = [tile for tile in choices if tile // 4 == pending[2]]
        drawn[turn] = int(rng.choice(choices))
    work = game.clone()
    try:
        # The engine validates that only hidden tiles moved (shared contract of both backends).
        work.redeal(hands, unseen, dead, drawn, temporary, riichi_flags, int(rng.integers(2**63)))
        return work
    except BaseException:
        work.close()
        raise


@GAMES.register('riichi')
class RiichiSpec:
    game_id = 'riichi'
    ruleset_id = 'riichi-4p-v1'
    board_shape = BOARD_SHAPE
    action_size = ACTION_SIZE
    action_encoding = 'riichi-111-v1'
    default_max_plies = 10000
    default_encoder = ENCODING_ID
    num_players = 4
    perfect_information = False
    has_chance = True
    zero_sum = False
    supports_undo = False
    draw_reasons = ()
    default_baselines = ('random', 'tsumogiri', 'shanten-greedy')
    default_teacher = 'shanten-greedy'

    # Game params are RiichiRules overrides (e.g. {"rounds": "tonpuu"}); the ruleset id is fixed.
    RULE_PARAMS = frozenset(f.name for f in fields(RiichiRules)) - {'ruleset_id'}

    def __init__(self, **params):
        unknown = set(params) - self.RULE_PARAMS
        if unknown:
            raise ValueError(f'riichi game params must be rule fields, got {sorted(unknown)}')
        # The match engine's default disables nagashi mangan; params override it like any rule.
        self.rules = RiichiRules(**{'nagashi_mangan': False, **params})

    def create_game(self, backend='python', rules=None, position=None, seed=None):
        backend = self.resolve_backend(backend)
        if position not in (None, {}):
            raise ValueError('riichi starts from a seed; replay joint actions for a later state')
        if rules is None:
            rules = self.rules
        elif isinstance(rules, dict):
            rules = RiichiRules(**{**json.loads(self.rules.to_json()), **rules})
        if backend == 'cpp':
            return _native().NativeRiichiGame(seed=seed, rules=rules)
        return RiichiGame(seed=seed, rules=rules)

    def resolve_backend(self, backend):
        if backend == 'python':
            return 'python'
        if backend == 'cpp':
            if not _native_available():
                raise RuntimeError('native riichi engine unavailable; run scripts/build_native.py')
            return 'cpp'
        if backend == 'auto':
            return 'cpp' if _native_available() else 'python'
        raise ValueError(f'unknown riichi backend {backend!r}')

    def rules_dict(self, game):
        return json.loads(game.rules.to_json())

    def actors(self, game):
        return game.actors()

    def legal_actions(self, game, player):
        return game.legal_actions(player)

    def legal_mask(self, game, player):
        return PlaneEncoder().legal_mask(game, player)

    def observe(self, game, player):
        return PlaneEncoder().encode(game, player)

    def step(self, game, actions):
        previous_hand = game.hand_number
        previous_scores = list(game.scores)
        game.play_joint(actions)
        info = {'reason': game.outcome.reason}
        if game.hand_number != previous_hand or game.outcome.terminal:
            # The hand summary reveals every seat's hidden state (tenpai, waits, concealed tiles,
            # placement, winning tile, discarder); copy it so retained step info never aliases the engine.
            info.update(hand=copy.deepcopy(game.last_hand),
                        score_delta=[s - old for s, old in zip(game.scores, previous_scores)])
            if not game.outcome.terminal:
                info['reason'] = game.last_hand['reason']
        return StepOutcome(game.outcome.terminal, False,
                           self.terminal_values(game) if game.outcome.terminal else None, info)

    def terminal_values(self, game):
        """Placement points including uma/oka, divided by 100 and clipped to [-1, 1]."""
        if not game.outcome.terminal:
            raise ValueError('terminal_values requires a terminal game')
        return [max(-1.0, min(1.0, value / 100.0)) for value in game.final_points]

    def make_encoder(self, name='', **params):
        if name in ('', ENCODING_ID):
            return PlaneEncoder(**params)
        if name == HISTORY_ENCODING_ID:
            return make_history_encoder(**params)
        raise ValueError(f'unknown riichi encoder: {name}')

    def symmetries(self):
        # Seat-relative observations already rotate seats. Suit swaps must also
        # preserve red identity and the ordered action table, so none are assumed.
        return []

    def baseline_agent(self, spec, seed=0):
        return RiichiBaselineAgent(spec, seed), None

    def determinize(self, game, player, rng):
        """A full hidden state consistent with ``player``'s observation (an owned clone; practical-play spec section 11)."""
        return sample_riichi_belief(game, player, rng)

    action_name = staticmethod(action_name)
