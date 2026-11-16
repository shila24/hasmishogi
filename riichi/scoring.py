"""Fu, limit points, four-player payments, and placement settlement."""
from dataclasses import dataclass, replace
from typing import Iterable

from .agari import Decomposition, Meld, decompose, hand_context
from .rules import RiichiRules
from .tiles import ORPHANS, RED_IDS, to_counts, validate_ids
from .yaku import WinContext, Yaku, concealed_triplet, detect_yaku, pair_fu


class NoYakuError(ValueError):
    """The structural hand has no legal yaku under this context."""


def calculate_fu(d: Decomposition, context: WinContext, rules: RiichiRules = RiichiRules()) -> int:
    """Evaluate fu for one winning assignment, rounding once at the end."""
    if d.kind == 'chiitoitsu':
        return 25
    if d.kind == 'kokushi':
        return 0
    if d.winning_tile is None:
        raise ValueError('fu requires a winning-tile assignment')
    pinfu = d.menzen and all(g.sequence for g in d.groups) and not pair_fu(d.pair, context, rules) and d.wait == 'ryanmen'
    if pinfu:
        return 20 if context.tsumo else 30
    fu = 20 + pair_fu(d.pair, context, rules)
    if context.tsumo:
        fu += 2
    elif d.menzen:
        fu += 10
    if d.wait in ('tanki', 'kanchan', 'penchan'):
        fu += 2
    for index, group in enumerate(d.groups):
        if not group.sequence:
            value = 8 if group.kan else 2
            if concealed_triplet(group, index, d, context):
                value *= 2
            if group.tiles[0] in ORPHANS:
                value *= 2
            fu += value
    return max(30, ((fu + 9) // 10) * 10)


def _nonnegative(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')


def base_points(han: int, fu: int, *, yakuman: int = 0, rules: RiichiRules = RiichiRules()) -> int:
    """Return base points before dealer multipliers and hundred-point rounding."""
    for name, value in (('han', han), ('fu', fu), ('yakuman', yakuman)):
        _nonnegative(value, name)
    if yakuman:
        return 8000 * yakuman
    if han < 1 or not (fu in (20, 25) or 30 <= fu <= 160 and fu % 10 == 0):
        raise ValueError('positive han and standard fu required')
    if han >= 13 and rules.kazoe_yakuman:
        return 8000
    if han >= 11:
        return 6000
    if han >= 8:
        return 4000
    if han >= 6:
        return 3000
    if han >= 5 or rules.kiriage_mangan and (han, fu) in ((4, 30), (3, 60)):
        return 2000
    return min(2000, fu * (2 ** (han + 2)))


@dataclass(frozen=True)
class Payment:
    """Each payer amount includes honba; total additionally includes table sticks."""
    ron: int = 0
    tsumo_dealer: int = 0
    tsumo_non_dealer: int = 0
    riichi_bonus: int = 0
    total: int = 0


def payments(base: int, *, dealer: bool = False, tsumo: bool = False, honba: int = 0,
             riichi_sticks: int = 0, rules: RiichiRules = RiichiRules()) -> Payment:
    for name, value in (('base', base), ('honba', honba), ('riichi_sticks', riichi_sticks)):
        _nonnegative(value, name)
    if base == 0 or type(dealer) is not bool or type(tsumo) is not bool:
        raise ValueError('positive base and boolean dealer/tsumo required')
    bonus = rules.riichi_stick * riichi_sticks

    def rounded(value: int) -> int:
        return ((value + 99) // 100) * 100

    if not tsumo:
        ron = rounded(base * (6 if dealer else 4)) + honba * rules.honba
        return Payment(ron=ron, riichi_bonus=bonus, total=ron + bonus)
    extra = honba * rules.honba // 3
    low = rounded(base * (2 if dealer else 1)) + extra
    high = 0 if dealer else rounded(base * 2) + extra
    return Payment(tsumo_dealer=high, tsumo_non_dealer=low, riichi_bonus=bonus,
                   total=(3 * low if dealer else 2 * low + high) + bonus)


@dataclass(frozen=True)
class Score:
    han: int
    fu: int
    yakuman: int
    base_points: int
    yaku: tuple[Yaku, ...]
    payment: Payment
    decomposition: Decomposition


def score_hand(hand: Iterable[int], winning_tile: int, melds: Iterable[Meld] = (), *,
               context: WinContext = WinContext(), rules: RiichiRules = RiichiRules(),
               honba: int = 0, riichi_sticks: int = 0,
               tile_ids: Iterable[int] | None = None) -> Score:
    """Maximize actual payment over all decompositions and winning assignments.

    Hand is concealed type counts INCLUDING the winning tile. Optional tile_ids
    contains the whole hand including declared melds and preserves red identity.
    """
    counts, melds, total = hand_context(hand, melds)
    if tile_ids is not None:
        ids = validate_ids(tile_ids)
        if to_counts(ids) != total:
            raise ValueError('tile_ids do not match the hand and melds')
        context = replace(context, red_dora=sum(t in RED_IDS for t in ids) if rules.red_fives else 0)
    if context.red_dora > sum(total[t] > 0 for t in (4, 13, 22)):
        raise ValueError('red_dora exceeds available five types')
    decompositions = decompose(counts, melds, winning_tile)
    if not decompositions:
        raise ValueError('hand is not complete')
    scores = []
    for d in decompositions:
        yaku = detect_yaku(d, total, context, rules)
        if not yaku:
            continue
        units = sum(y.yakuman for y in yaku)
        if units and not rules.multiple_yakuman:
            units = max(y.yakuman for y in yaku)
        han = sum(y.han for y in yaku)
        fu = calculate_fu(d, context, rules)
        base = base_points(han, fu, yakuman=units, rules=rules)
        pay = payments(base, dealer=context.dealer, tsumo=context.tsumo, honba=honba, riichi_sticks=riichi_sticks, rules=rules)
        scores.append(Score(han, fu, units, base, yaku, pay, d))
    if not scores:
        raise NoYakuError('no yaku; dora alone cannot qualify a hand')
    return max(scores, key=lambda score: (score.payment.total, score.yakuman, score.han, score.fu))


def tenpai_payments(tenpai: Iterable[bool], rules: RiichiRules = RiichiRules()) -> tuple[int, ...]:
    flags = tuple(tenpai)
    if len(flags) != 4 or any(type(t) is not bool for t in flags):
        raise ValueError('expected four tenpai booleans')
    n = sum(flags)
    if n in (0, 4):
        return (0,) * 4
    return tuple(rules.tenpai_payment // n if t else -rules.tenpai_payment // (4 - n) for t in flags)


def placement_points(scores: Iterable[int], rules: RiichiRules = RiichiRules(), *,
                     tie_order: Iterable[int] = (0, 1, 2, 3)) -> tuple[float, ...]:
    """Return thousands of points; ties follow initial seat priority, oka goes first."""
    scores, order = tuple(scores), tuple(tie_order)
    if len(scores) != 4 or any(type(s) is not int for s in scores):
        raise ValueError('expected four integer scores')
    if len(order) != 4 or set(order) != set(range(4)):
        raise ValueError('tie_order must be a seat permutation')
    ranking = sorted(order, key=lambda seat: -scores[seat])
    result = [0.0] * 4
    for rank, seat in enumerate(ranking):
        result[seat] = (scores[seat] - rules.return_points) / 1000 + rules.uma[rank]
    result[ranking[0]] += 4 * (rules.return_points - rules.start_points) / 1000
    return tuple(result)


def settle_payment(payment: Payment, *, winner: int, dealer: int, discarder: int | None = None) -> tuple[int, ...]:
    """Seat deltas for a single win; sticks are external to the four seat accounts."""
    if any(type(s) is not int or not 0 <= s < 4 for s in (winner, dealer)):
        raise ValueError('invalid seat')
    deltas = [0] * 4
    if payment.ron:
        if type(discarder) is not int or not 0 <= discarder < 4 or discarder == winner:
            raise ValueError('ron requires another discarder seat')
        deltas[discarder] = -payment.ron
    else:
        if discarder is not None:
            raise ValueError('tsumo has no discarder')
        if (winner == dealer) != (payment.tsumo_dealer == 0):
            raise ValueError('payment dealer status does not match seats')
        for seat in range(4):
            if seat != winner:
                deltas[seat] = -(payment.tsumo_dealer if seat == dealer else payment.tsumo_non_dealer)
    deltas[winner] = payment.total
    return tuple(deltas)
