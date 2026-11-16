"""Pattern yaku, contextual yaku, yakuman variants, and bonus-tile counting."""
from collections import Counter
from dataclasses import dataclass, fields
from typing import Iterable

from .agari import Decomposition, Group
from .rules import RiichiRules
from .tiles import HONORS, ORPHANS, TERMINALS, dora_type


@dataclass(frozen=True)
class WinContext:
    """Historical facts supplied by the caller; winds and indicators are tile TYPES."""
    tsumo: bool = False
    seat_wind: int = 28
    round_wind: int = 27
    riichi: bool = False
    double_riichi: bool = False
    ippatsu: bool = False
    rinshan: bool = False
    chankan: bool = False
    haitei: bool = False
    houtei: bool = False
    tenhou: bool = False
    chiihou: bool = False
    renhou: bool = False
    dora_indicators: tuple[int, ...] = ()
    ura_indicators: tuple[int, ...] = ()
    red_dora: int = 0
    yaku_established: bool | None = None

    def __post_init__(self) -> None:
        for field in fields(self):
            if type(field.default) is bool and type(getattr(self, field.name)) is not bool:
                raise ValueError(f'{field.name} must be boolean')
        if any(type(w) is not int or not 27 <= w <= 30 for w in (self.seat_wind, self.round_wind)):
            raise ValueError('winds must be tile types 27..30')
        for name in ('dora_indicators', 'ura_indicators'):
            values = tuple(getattr(self, name))
            if len(values) > 5:
                raise ValueError('at most five indicators')
            for value in values:
                dora_type(value)
            object.__setattr__(self, name, values)
        if type(self.red_dora) is not int or not 0 <= self.red_dora <= 3:
            raise ValueError('red_dora must be an integer in 0..3')
        if self.yaku_established is not None and type(self.yaku_established) is not bool:
            raise ValueError('yaku_established must be a boolean or None')
        if self.ippatsu and not (self.riichi or self.double_riichi):
            raise ValueError('ippatsu requires riichi')
        if (self.rinshan or self.haitei or self.tenhou or self.chiihou) and not self.tsumo:
            raise ValueError('tsumo-only context on ron')
        if (self.chankan or self.houtei or self.renhou) and self.tsumo:
            raise ValueError('ron-only context on tsumo')
        if self.rinshan and self.ippatsu:
            raise ValueError('kan cancels ippatsu before a rinshan draw')
        if self.rinshan and self.haitei or self.chankan and self.houtei:
            raise ValueError('last-wall and kan events are mutually exclusive')
        if sum((self.tenhou, self.chiihou, self.renhou)) > 1:
            raise ValueError('first-turn wins are mutually exclusive')
        if self.tenhou and self.seat_wind != 27 or (self.chiihou or self.renhou) and self.seat_wind == 27:
            raise ValueError('first-turn win has wrong seat wind')
        if (self.tenhou or self.chiihou or self.renhou) and any((self.riichi, self.double_riichi, self.ippatsu, self.rinshan, self.chankan, self.haitei, self.houtei)):
            raise ValueError('inconsistent first-turn context')

    @property
    def dealer(self) -> bool:
        return self.seat_wind == 27


@dataclass(frozen=True)
class Yaku:
    name: str
    han: int = 0
    yakuman: int = 0


def pair_fu(pair: int | None, context: WinContext, rules: RiichiRules) -> int:
    if pair is None:
        return 0
    if pair >= 31:
        return 2
    values = int(pair == context.seat_wind) + int(pair == context.round_wind)
    return rules.double_wind_pair_fu if values == 2 else values * 2


def concealed_triplet(group: Group, index: int, decomposition: Decomposition, context: WinContext) -> bool:
    return not group.sequence and not group.open and (context.tsumo or index != decomposition.winning_group)


def validate_context(decomposition: Decomposition, context: WinContext, rules: RiichiRules) -> None:
    if not decomposition.menzen and (context.riichi or context.double_riichi or context.ippatsu):
        raise ValueError('riichi requires a closed hand')
    if (context.tenhou or context.chiihou or context.renhou) and any(g.declared for g in decomposition.groups):
        raise ValueError('first-turn win cannot contain declared melds')
    if context.rinshan and not any(g.kan for g in decomposition.groups):
        raise ValueError('rinshan requires a kan')
    if not rules.atozuke and not decomposition.menzen and context.yaku_established is None:
        raise ValueError('atozuke-disabled open scoring requires yaku_established history')


def detect_yaku(decomposition: Decomposition, counts: tuple[int, ...], context: WinContext,
                rules: RiichiRules = RiichiRules()) -> tuple[Yaku, ...]:
    """Evaluate one assignment; counts include meld tiles, including kan fourth tiles."""
    d = decomposition
    validate_context(d, context, rules)
    if d.winning_tile is None:
        raise ValueError('yaku evaluation requires a winning-tile assignment')
    tiles = {i for i, n in enumerate(counts) if n}
    groups = d.groups
    sequences = [g.tiles[0] for g in groups if g.sequence]
    triplets = [g.tiles[0] for g in groups if not g.sequence]
    concealed = sum(concealed_triplet(g, i, d, context) for i, g in enumerate(groups))
    kans = sum(g.kan for g in groups)
    yakuman: list[Yaku] = []

    def yak(name: str, condition: bool, double: bool = False) -> None:
        if condition:
            yakuman.append(Yaku(name, yakuman=2 if double and rules.double_yakuman else 1))

    yak('tenhou', context.tenhou)
    yak('chiihou', context.chiihou)
    yak('renhou', context.renhou and rules.renhou == 'yakuman')
    if d.kind == 'kokushi':
        thirteen = counts[d.winning_tile] == 2
        yak('kokushi_13' if thirteen else 'kokushi', True, thirteen)
    yak('daisangen', all(t in triplets for t in (31, 32, 33)))
    winds = sum(t in triplets for t in range(27, 31))
    yak('daisuushii', winds == 4, True)
    yak('shousuushii', winds == 3 and d.pair in range(27, 31))
    yak('tsuuiisou', bool(tiles) and tiles <= HONORS)
    yak('chinroutou', bool(tiles) and tiles <= TERMINALS)
    yak('ryuuiisou', bool(tiles) and tiles <= {19, 20, 21, 23, 25, 32})
    yak('suukantsu', kans == 4)
    if concealed == 4:
        tanki = d.wait == 'tanki'
        yak('suuankou_tanki' if tanki else 'suuankou', True, tanki)
    suits = {t // 9 for t in tiles if t < 27}
    if d.menzen and not any(g.declared for g in groups) and len(suits) == 1 and not tiles & HONORS:
        suit = next(iter(suits)) * 9
        shape = counts[suit:suit + 9]
        base = (3, 1, 1, 1, 1, 1, 1, 1, 3)
        if all(a >= b for a, b in zip(shape, base)):
            before = list(shape)
            before[d.winning_tile % 9] -= 1
            pure = tuple(before) == base
            yak('junsei_chuuren' if pure else 'chuuren', True, pure)
    if yakuman:
        return tuple(yakuman)
    if not rules.atozuke and not d.menzen and context.yaku_established is False:
        return ()
    yaku: list[Yaku] = []

    def add(name: str, condition: bool, closed: int, opened: int | None = None) -> None:
        han = closed if d.menzen else (closed if opened is None else opened)
        if condition and han:
            yaku.append(Yaku(name, han))

    add('double_riichi' if context.double_riichi else 'riichi', context.riichi or context.double_riichi, 2 if context.double_riichi else 1, 0)
    add('ippatsu', context.ippatsu and rules.ippatsu, 1, 0)
    add('menzen_tsumo', context.tsumo, 1, 0)
    add('rinshan', context.rinshan, 1)
    add('chankan', context.chankan, 1)
    add('haitei', context.haitei, 1)
    add('houtei', context.houtei, 1)
    add('tanyao', not tiles & ORPHANS and (d.menzen or rules.kuitan), 1)
    add('chiitoitsu', d.kind == 'chiitoitsu', 2, 0)
    add('honroutou', tiles <= ORPHANS, 2)
    if len(suits) == 1:
        add('honitsu' if tiles & HONORS else 'chinitsu', True, 3 if tiles & HONORS else 6, 2 if tiles & HONORS else 5)
    if d.kind == 'standard':
        add('pinfu', len(sequences) == 4 and pair_fu(d.pair, context, rules) == 0 and d.wait == 'ryanmen', 1, 0)
        repeated = sum(n // 2 for n in Counter(sequences).values())
        add('ryanpeikou' if repeated == 2 else 'iipeikou', repeated > 0, 3 if repeated == 2 else 1, 0)
        add('toitoi', not sequences, 2)
        add('sanankou', concealed >= 3, 2)
        add('sankantsu', kans == 3, 2)
        add('shousangen', sum(t in triplets for t in (31, 32, 33)) == 2 and d.pair in (31, 32, 33), 2)
        for t, name in ((31, 'haku'), (32, 'hatsu'), (33, 'chun'), (context.seat_wind, 'seat_wind'), (context.round_wind, 'round_wind')):
            add(name, t in triplets, 1)
        add('sanshoku_doujun', any(all(n + s * 9 in sequences for s in range(3)) for n in range(7)), 2, 1)
        add('sanshoku_doukou', any(all(n + s * 9 in triplets for s in range(3)) for n in range(9)), 2)
        add('ittsuu', any(all(s * 9 + n in sequences for n in (0, 3, 6)) for s in range(3)), 2, 1)
        outside = d.pair in ORPHANS and all(any(t in ORPHANS for t in g.tiles) for g in groups)
        add('chanta' if tiles & HONORS else 'junchan', outside and bool(sequences), 2 if tiles & HONORS else 3, 1 if tiles & HONORS else 2)
    if not yaku and not (context.renhou and rules.renhou == 'mangan'):
        return ()
    dora_indicators = context.dora_indicators if rules.kan_dora else context.dora_indicators[:1]
    ura_indicators = context.ura_indicators if rules.kan_dora else context.ura_indicators[:1]
    add('dora', True, sum(counts[dora_type(t)] for t in dora_indicators))
    add('aka_dora', rules.red_fives, context.red_dora)
    add('ura_dora', rules.ura_dora and (context.riichi or context.double_riichi), sum(counts[dora_type(t)] for t in ura_indicators))
    if context.renhou and rules.renhou == 'mangan':
        # Renhou is a mangan floor: a hand keeps its own yaku only when an ordinary (non-bonus)
        # yaku exists and the hand as a whole reaches five han; dora alone never replaces renhou.
        bonus = {'dora', 'aka_dora', 'ura_dora'}
        has_ordinary = any(y.name not in bonus for y in yaku)
        if not has_ordinary or sum(y.han for y in yaku) < 5:
            return (Yaku('renhou', 5),)
    return tuple(yaku)


def nagashi_eligible(discards: Iterable[int], *, any_discard_called: bool = False,
                     has_open_meld: bool = False, rules: RiichiRules = RiichiRules()) -> bool:
    """Check exhaustive-draw nagashi facts; payment uses mangan tsumo settlement."""
    discards = tuple(discards)
    if any(type(t) is not int or not 0 <= t < 34 for t in discards):
        raise ValueError('discards must be tile types')
    return rules.nagashi_mangan and bool(discards) and not any_discard_called and not has_open_meld and set(discards) <= ORPHANS
