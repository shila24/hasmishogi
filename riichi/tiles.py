"""Physical tile IDs, type counts, notation, and reproducible wall layout."""
from dataclasses import dataclass
import random
import re
from typing import Iterable

SUITS = 'mpsz'
TERMINALS = frozenset((0, 8, 9, 17, 18, 26))
HONORS = frozenset(range(27, 34))
ORPHANS = TERMINALS | HONORS
RED_IDS = frozenset((16, 52, 88))


def validate_counts(counts: Iterable[int]) -> tuple[int, ...]:
    """Validate a 34-entry type-count vector, without guessing its representation."""
    counts = tuple(counts)
    if len(counts) != 34 or any(type(n) is not int or not 0 <= n <= 4 for n in counts):
        raise ValueError('expected 34 integer counts in 0..4')
    return counts


def validate_ids(tiles: Iterable[int]) -> tuple[int, ...]:
    tiles = tuple(tiles)
    if any(type(t) is not int or not 0 <= t < 136 for t in tiles):
        raise ValueError('physical tile IDs must be integers in 0..135')
    if len(set(tiles)) != len(tiles):
        raise ValueError('duplicate physical tile ID')
    return tiles


def to_counts(tiles: Iterable[int]) -> tuple[int, ...]:
    counts = [0] * 34
    for tile in validate_ids(tiles):
        counts[tile // 4] += 1
    return tuple(counts)


def _codes(code: str) -> list[tuple[int, bool]]:
    if not isinstance(code, str):
        raise ValueError('tile notation must be a string')
    result = []
    pos = 0
    for match in re.finditer(r'([0-9]+)([mpsz])', code):
        if match.start() != pos:
            raise ValueError('invalid tile notation')
        digits, suit = match.groups()
        offset = SUITS.index(suit) * 9
        for digit in digits:
            n = int(digit)
            if suit == 'z' and not 1 <= n <= 7:
                raise ValueError('honors use 1z..7z')
            result.append((offset + (5 if n == 0 else n) - 1, n == 0))
        pos = match.end()
    if pos != len(code):
        raise ValueError('invalid tile notation')
    return result


def parse_tiles(code: str, *, red_fives: bool = True) -> tuple[int, ...]:
    """Parse physical IDs; 0 is red, 5 is non-red, first five copy is reserved red."""
    used: set[int] = set()
    result = []
    for tile, red in _codes(code):
        if red and not red_fives:
            raise ValueError('red fives are disabled')
        candidates = (tile * 4,) if red else range(tile * 4, tile * 4 + 4)
        available = [t for t in candidates if t not in used and
                     (red or not red_fives or t not in RED_IDS)]
        if not available:
            raise ValueError('too many copies of a tile')
        result.append(available[0])
        used.add(available[0])
    return tuple(result)


def parse_counts(code: str) -> tuple[int, ...]:
    """Parse type counts; red and normal five notation have the same type."""
    result = [0] * 34
    for tile, _ in _codes(code):
        result[tile] += 1
    return validate_counts(result)


def format_tiles(tiles: Iterable[int], *, red_fives: bool = True) -> str:
    """Format a multiset; physical copy identity is intentionally not serialized."""
    groups: list[list[str]] = [[], [], [], []]
    for tile in sorted(validate_ids(tiles)):
        kind = tile // 4
        groups[kind // 9].append('0' if red_fives and tile in RED_IDS else str(kind % 9 + 1))
    return ''.join(''.join(digits) + suit for digits, suit in zip(groups, SUITS) if digits)


def format_counts(counts: Iterable[int]) -> str:
    counts = validate_counts(counts)
    return ''.join(''.join(str(t % 9 + 1) * counts[t] for t in range(s * 9, min(s * 9 + 9, 34)))
                   + suit for s, suit in enumerate(SUITS) if any(counts[s * 9:s * 9 + 9]))


def dora_type(indicator: int) -> int:
    """Convert an indicator TYPE (not physical ID) to its dora TYPE."""
    if type(indicator) is not int or not 0 <= indicator < 34:
        raise ValueError('indicator type must be in 0..33')
    if indicator < 27:
        return indicator // 9 * 9 + (indicator + 1) % 9
    start, length = (27, 4) if indicator < 31 else (31, 3)
    return start + (indicator - start + 1) % length


@dataclass(frozen=True)
class Wall:
    """Draw left-to-right; dead wall is four rinshan then five indicator pairs."""
    live: tuple[int, ...]
    dead: tuple[int, ...]

    @property
    def rinshan(self) -> tuple[int, ...]:
        return self.dead[:4]

    @property
    def dora_indicators(self) -> tuple[int, ...]:
        return self.dead[4::2]

    @property
    def ura_indicators(self) -> tuple[int, ...]:
        return self.dead[5::2]


def build_wall(seed: int | None = None, *, rng: random.Random | None = None) -> Wall:
    """Shuffle all 136 IDs using an isolated seeded RNG or a supplied RNG."""
    if seed is not None and rng is not None:
        raise ValueError('supply seed or rng, not both')
    tiles = list(range(136))
    (rng if rng is not None else random.Random(seed)).shuffle(tiles)
    return Wall(tuple(tiles[:122]), tuple(tiles[122:]))
