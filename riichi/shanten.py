"""Suit-table dynamic programming for standard, seven-pairs, and orphan shanten."""
from functools import lru_cache
from itertools import combinations_with_replacement, product
from typing import Iterable

from .agari import Meld, hand_context, is_agari
from .tiles import ORPHANS, validate_counts


@lru_cache(maxsize=200000)
def suit_table(counts: tuple[int, ...], honors: bool = False) -> tuple[tuple[int, int, int], ...]:
    """Memoized precomputed suit frontier: (sets, partial sets, head) possibilities."""
    first = next((i for i, n in enumerate(counts) if n), None)
    if first is None:
        return ((0, 0, 0),)
    work = list(counts)
    work[first] -= 1
    options = set(suit_table(tuple(work), honors))

    def take(tiles: tuple[int, ...], sets: int, partial: int, head: int) -> None:
        for tile in tiles:
            work[tile] -= 1
        for m, t, p in suit_table(tuple(work), honors):
            if m + sets <= 4 and p + head <= 1:
                options.add((m + sets, min(4, t + partial), p + head))
        for tile in tiles:
            work[tile] += 1

    if work[first] >= 2:
        take((first, first), 1, 0, 0)
    if work[first]:
        take((first,), 0, 1, 0)
        take((first,), 0, 0, 1)
    if not honors:
        if first <= 6 and work[first + 1] and work[first + 2]:
            take((first + 1, first + 2), 1, 0, 0)
        for delta in (1, 2):
            if first + delta < 9 and work[first + delta]:
                take((first + delta,), 0, 1, 0)
    best: dict[tuple[int, int], int] = {}
    for m, t, p in options:
        best[m, p] = max(t, best.get((m, p), -1))
    return tuple((m, t, p) for (m, p), t in sorted(best.items()))


def warm_tables(hands: Iterable[Iterable[int]] = ()) -> None:
    """Precompute elementary suits and requested workload suits without disk assets."""
    for suit in product(range(2), repeat=9):
        suit_table(suit)
    for hand in hands:
        counts = validate_counts(hand)
        for start in (0, 9, 18):
            suit_table(counts[start:start + 9])
        suit_table(counts[27:], True)
        if 4 in counts:
            _bounded_standard(counts, (4,) * 34, 0)


def kokushi_shanten(hand: Iterable[int], melds: Iterable[Meld] = ()) -> int:
    counts, melds, _ = hand_context(hand, melds)
    if melds:
        return 99
    return 13 - sum(counts[t] > 0 for t in ORPHANS) - int(any(counts[t] >= 2 for t in ORPHANS))


def chiitoitsu_shanten(hand: Iterable[int], melds: Iterable[Meld] = ()) -> int:
    counts, melds, _ = hand_context(hand, melds)
    if melds:
        return 99
    return 6 - sum(n >= 2 for n in counts) + max(0, 7 - sum(n > 0 for n in counts))


@lru_cache(maxsize=65536)
def _standard(counts: tuple[int, ...], meld_count: int) -> int:
    frontier = {(meld_count, 0): 0}
    tables = [suit_table(counts[i:i + 9]) for i in (0, 9, 18)] + [suit_table(counts[27:], True)]
    for table in tables:
        following: dict[tuple[int, int], int] = {}
        for (m, p), t in frontier.items():
            for sm, st, sp in table:
                nm, np = m + sm, p + sp
                if nm <= 4 and np <= 1:
                    key = nm, np
                    following[key] = max(following.get(key, -1), min(4 - nm, t + st))
        frontier = following
    return min(8 - 2 * m - t - p for (m, p), t in frontier.items())



@lru_cache(maxsize=20)
def _targets(honors: bool, sets: int, pair: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    """All bounded complete suit patterns, represented sparsely for distance lookup."""
    size = 7 if honors else 9
    groups = [(t, t, t) for t in range(size)]
    if not honors:
        groups += [(t, t + 1, t + 2) for t in range(7)]
    patterns = set()
    for selected in combinations_with_replacement(groups, sets):
        counts = [0] * size
        for group in selected:
            for tile in group:
                counts[tile] += 1
        if max(counts) > 4:
            continue
        for head in range(size) if pair else (None,):
            if head is not None:
                counts[head] += 2
            if max(counts) <= 4:
                patterns.add(tuple((t, n) for t, n in enumerate(counts) if n))
            if head is not None:
                counts[head] -= 2
    return tuple(sorted(patterns))


@lru_cache(maxsize=16384)
def _bounded_costs(counts: tuple[int, ...], capacities: tuple[int, ...],
                   honors: bool, maximum: int) -> tuple[tuple[int, int, int], ...]:
    result = []
    for sets in range(maximum + 1):
        for pair in (0, 1):
            best = 99
            for target in _targets(honors, sets, pair):
                cost = 0
                for tile, n in target:
                    if n > capacities[tile]:
                        cost = 99
                        break
                    cost += max(0, n - counts[tile])
                    if cost >= best:
                        break
                best = min(best, cost)
                if best == max(0, 3 * sets + 2 * pair - sum(counts)):
                    break
            result.append((sets, pair, best))
    return tuple(result)


@lru_cache(maxsize=16384)
def _bounded_standard(counts: tuple[int, ...], capacities: tuple[int, ...], meld_count: int) -> int:
    """Exact replacement distance when a fourth copy makes a structural wait dead."""
    maximum = 4 - meld_count
    frontier = {(0, 0): 0}
    for start, end in ((0, 9), (9, 18), (18, 27), (27, 34)):
        costs = _bounded_costs(counts[start:end], capacities[start:end], start == 27, maximum)
        following = {}
        for (m, p), cost in frontier.items():
            for sm, sp, extra in costs:
                key = (m + sm, p + sp)
                if key[0] <= maximum and key[1] <= 1:
                    following[key] = min(following.get(key, 99), cost + extra)
        frontier = following
    return frontier[maximum, 1] - 1


def _standard_with_capacity(counts: tuple[int, ...], melds: tuple[Meld, ...], total: tuple[int, ...]) -> int:
    # Declared melds consume copies too: a pon of 6m leaves one 6m for the concealed part, so
    # capacities must be applied whenever melds exist, not only when some tile already reaches four.
    if (melds or 4 in total) and sum(counts) in (13 - 3 * len(melds), 14 - 3 * len(melds)):
        capacities = tuple(4 - (n - c) for c, n in zip(counts, total))
        return _bounded_standard(counts, capacities, len(melds))
    return _standard(counts, len(melds))


def standard_shanten(hand: Iterable[int], melds: Iterable[Meld] = ()) -> int:
    counts, melds, total = hand_context(hand, melds)
    return _standard_with_capacity(counts, melds, total)


def shanten(hand: Iterable[int], melds: Iterable[Meld] = ()) -> int:
    """Return -1 for complete, 0 for tenpai; meld tiles are excluded from hand."""
    counts, melds, total = hand_context(hand, melds)
    result = _standard_with_capacity(counts, melds, total)
    if not melds:
        pairs = sum(n >= 2 for n in counts)
        unique = sum(n > 0 for n in counts)
        chiitoi = 6 - pairs + max(0, 7 - unique)
        kokushi = 13 - sum(counts[t] > 0 for t in ORPHANS) - int(any(counts[t] >= 2 for t in ORPHANS))
        result = min(result, chiitoi, kokushi)
    return result


def waits(hand: Iterable[int], melds: Iterable[Meld] = ()) -> tuple[int, ...]:
    """Return structural winning types with fewer than four copies in hand+melds."""
    counts, melds, total = hand_context(hand, melds)
    if sum(counts) != 13 - 3 * len(melds):
        raise ValueError('waits requires a hand before the winning draw')
    result = []
    for tile, n in enumerate(total):
        if n == 4:
            continue
        work = list(counts)
        work[tile] += 1
        if is_agari(work, melds):
            result.append(tile)
    return tuple(result)


def is_tenpai(hand: Iterable[int], melds: Iterable[Meld] = ()) -> bool:
    return bool(waits(hand, melds))
