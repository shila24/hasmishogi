"""Enumerate all structural decompositions and winning-tile assignments."""
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from .tiles import ORPHANS, validate_counts


@dataclass(frozen=True)
class Meld:
    """A declared set of tile TYPES, including all four tiles for a kan."""
    tiles: tuple[int, ...]
    kind: str
    open: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, 'tiles', tuple(sorted(self.tiles)))
        if any(type(t) is not int or not 0 <= t < 34 for t in self.tiles) or type(self.open) is not bool:
            raise ValueError('invalid meld tiles or open flag')
        t = self.tiles
        if self.kind == 'chi':
            valid = len(t) == 3 and t[0] < 27 and t[0] // 9 == t[-1] // 9 and t == tuple(range(t[0], t[0] + 3)) and self.open
        elif self.kind in ('pon', 'kan'):
            valid = len(t) == (4 if self.kind == 'kan' else 3) and len(set(t)) == 1 and (self.open or self.kind == 'kan')
        else:
            valid = False
        if not valid:
            raise ValueError('invalid declared meld')


@dataclass(frozen=True)
class Group:
    tiles: tuple[int, ...]
    open: bool = False
    declared: bool = False

    @property
    def sequence(self) -> bool:
        return self.tiles[0] != self.tiles[1]

    @property
    def kan(self) -> bool:
        return len(self.tiles) == 4


@dataclass(frozen=True)
class Decomposition:
    """Winning group -1 means pair; None means no assignment was requested."""
    kind: str
    groups: tuple[Group, ...] = ()
    pair: int | None = None
    winning_group: int | None = None
    winning_tile: int | None = None
    special_pairs: tuple[int, ...] = ()

    @property
    def menzen(self) -> bool:
        return not any(group.open for group in self.groups)

    @property
    def wait(self) -> str:
        if self.kind == 'kokushi':
            return 'kokushi'
        if self.kind == 'chiitoitsu' or self.winning_group == -1:
            return 'tanki'
        if self.winning_group is None:
            return 'unknown'
        group = self.groups[self.winning_group]
        if not group.sequence:
            return 'shanpon'
        low = group.tiles[0]
        if self.winning_tile == low + 1:
            return 'kanchan'
        if (low % 9 == 0 and self.winning_tile == low + 2) or (low % 9 == 6 and self.winning_tile == low):
            return 'penchan'
        return 'ryanmen'


def hand_context(hand: Iterable[int], melds: Iterable[Meld] = ()) -> tuple[tuple[int, ...], tuple[Meld, ...], tuple[int, ...]]:
    """Validate concealed counts and aggregate physical availability across melds."""
    counts = validate_counts(hand)
    melds = tuple(melds)
    if len(melds) > 4 or any(not isinstance(m, Meld) for m in melds):
        raise ValueError('expected at most four Meld objects')
    total = list(counts)
    for meld in melds:
        for tile in meld.tiles:
            total[tile] += 1
    validate_counts(total)
    if sum(counts) > 14 - 3 * len(melds):
        raise ValueError('too many concealed tiles for meld context')
    return counts, melds, tuple(total)


@lru_cache(maxsize=65536)
def _sets(counts: tuple[int, ...]) -> tuple[tuple[tuple[int, ...], ...], ...]:
    first = next((i for i, n in enumerate(counts) if n), None)
    if first is None:
        return ((),)
    result = []
    work = list(counts)
    if work[first] >= 3:
        work[first] -= 3
        result.extend((((first,) * 3,) + tail) for tail in _sets(tuple(work)))
        work[first] += 3
    if first < 27 and first % 9 <= 6 and work[first + 1] and work[first + 2]:
        for tile in range(first, first + 3):
            work[tile] -= 1
        result.extend(((tuple(range(first, first + 3)),) + tail) for tail in _sets(tuple(work)))
    return tuple(result)


def decompose(hand: Iterable[int], melds: Iterable[Meld] = (), winning_tile: int | None = None) -> tuple[Decomposition, ...]:
    """Concealed counts INCLUDE the winning tile and EXCLUDE declared melds."""
    counts, melds, _ = hand_context(hand, melds)
    if winning_tile is not None and (type(winning_tile) is not int or not 0 <= winning_tile < 34 or not counts[winning_tile]):
        raise ValueError('winning tile must occur in the concealed hand')
    if sum(counts) != 14 - 3 * len(melds):
        return ()
    result = []
    if not melds:
        if sum(n == 2 for n in counts) == 7:
            result.append(Decomposition('chiitoitsu', winning_tile=winning_tile,
                                        special_pairs=tuple(i for i, n in enumerate(counts) if n == 2)))
        if all(counts[t] for t in ORPHANS) and sum(counts[t] for t in ORPHANS) == 14:
            result.append(Decomposition('kokushi', pair=next(i for i, n in enumerate(counts) if n == 2), winning_tile=winning_tile))
    declared = tuple(Group(m.tiles, m.open, True) for m in melds)
    for pair, count in enumerate(counts):
        if count < 2:
            continue
        work = list(counts)
        work[pair] -= 2
        # Triplet-first and sequence-first branches can emit the same partition in different group
        # orders; canonicalise (sorted groups) and keep the first occurrence.
        for sets in dict.fromkeys(tuple(sorted(partition)) for partition in _sets(tuple(work))):
            groups = tuple(Group(tiles) for tiles in sets) + declared
            assignments: list[int | None] = [None]
            if winning_tile is not None:
                assignments = [-1] if pair == winning_tile else []
                seen = set()
                for index, group in enumerate(groups):
                    if not group.declared and winning_tile in group.tiles and group.tiles not in seen:
                        assignments.append(index)
                        seen.add(group.tiles)
            for index in assignments:
                result.append(Decomposition('standard', groups, pair, index, winning_tile))
    return tuple(result)


def is_agari(hand: Iterable[int], melds: Iterable[Meld] = ()) -> bool:
    return bool(decompose(hand, melds))
