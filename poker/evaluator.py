"""Seven-card evaluation without tables or third-party dependencies.

Ranks are base-15 integers: category followed by five padded tiebreak digits.
Cards use rank * 4 + suit, with rank zero representing a deuce.
"""

from collections import Counter


def _score(category: int, ranks: list[int]) -> int:
    result = category
    for rank in ranks + [0] * (5 - len(ranks)):
        result = result * 15 + rank
    return result


def _straight(ranks) -> int:
    present = set(ranks)
    if 14 in present:
        present.add(1)
    for high in range(14, 4, -1):
        if all(rank in present for rank in range(high - 4, high + 1)):
            return high
    return 0


def rank7(cards) -> int:
    """Return the rank of the best five of exactly seven distinct cards."""
    cards = tuple(cards)
    if (len(cards) != 7 or any(type(c) is not int or not 0 <= c < 52 for c in cards)
            or len(set(cards)) != 7):
        raise ValueError('rank7 requires seven distinct card ids in 0..51')
    ranks = sorted((c // 4 + 2 for c in cards), reverse=True)
    counts = Counter(ranks)
    suits = [[c // 4 + 2 for c in cards if c % 4 == suit] for suit in range(4)]
    flush = next((sorted(row, reverse=True) for row in suits if len(row) >= 5), None)
    if flush and (high := _straight(flush)):
        return _score(8, [high])
    quads = [r for r in counts if counts[r] == 4]
    if quads:
        return _score(7, [quads[0], max(r for r in ranks if r != quads[0])])
    triples = sorted((r for r in counts if counts[r] >= 3), reverse=True)
    if triples:
        pairs = [r for r in counts if counts[r] >= 2 and r != triples[0]]
        if pairs:
            return _score(6, [triples[0], max(pairs)])
    if flush:
        return _score(5, flush[:5])
    if high := _straight(ranks):
        return _score(4, [high])
    if triples:
        return _score(3, [triples[0]] + [r for r in ranks if r != triples[0]][:2])
    pairs = sorted((r for r in counts if counts[r] == 2), reverse=True)
    if len(pairs) >= 2:
        return _score(2, pairs[:2] + [max(r for r in ranks if r not in pairs[:2])])
    if pairs:
        return _score(1, pairs + [r for r in ranks if r != pairs[0]][:3])
    return _score(0, ranks[:5])
