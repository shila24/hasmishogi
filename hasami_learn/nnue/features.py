"""hlnn-kp-v1; layout.py is the shared source for the generated C++ table."""
from __future__ import annotations

from shogi.model import SQUARES
from .layout import LAYOUT

FEATURE_SET = LAYOUT["feature_set"]
BOARD_TYPES = tuple(LAYOUT["board_types"])
HAND_LIMITS = tuple(LAYOUT["hand_limits"])
MAX_ACTIVE = LAYOUT["max_active"]
BOARD_FEATURES = 2 * len(BOARD_TYPES) * SQUARES
HAND_OFFSETS = tuple(sum(HAND_LIMITS[:i]) for i in range(7))
HAND_FEATURES = 2 * sum(HAND_LIMITS)
KING_STRIDE = BOARD_FEATURES + HAND_FEATURES + SQUARES
FEATURE_COUNT = SQUARES * KING_STRIDE


def validate_position(position):
    """Check structural/material bounds without asserting history or legality."""
    totals = [0] * 7
    kings = [[], []]
    for sq, cell in enumerate(position.board):
        if not cell:
            continue
        side, kind = cell >> 4, cell & 15
        if kind == 8:
            kings[side].append(sq)
        else:
            base = kind - (8 if kind <= 12 else 7) if kind > 8 else kind
            totals[base - 1] += 1
    for hand in position.hands:
        for i, count in enumerate(hand):
            totals[i] += count
    if any(len(k) != 1 for k in kings) or any(n > cap for n, cap in zip(totals, HAND_LIMITS)):
        raise ValueError("NNUE position requires two kings and bounded material")
    return [k[0] for k in kings]


def active_features(position):
    """Two ascending feature lists, ordered [side to move, other side]."""
    kings = validate_position(position)
    output = []
    for side in (position.turn, 1 - position.turn):
        relative = lambda square: 80 - square if side else square
        offset = relative(kings[side]) * KING_STRIDE
        indices = []
        for square, cell in enumerate(position.board):
            if cell and cell & 15 != 8:
                owner = (cell >> 4) ^ side
                kind = BOARD_TYPES.index(cell & 15)
                indices.append(offset + (owner * 13 + kind) * SQUARES + relative(square))
        for owner in range(2):
            for hand, count in enumerate(position.hands[owner ^ side]):
                base = offset + BOARD_FEATURES + owner * 38 + HAND_OFFSETS[hand]
                indices.extend(base + index for index in range(count))
        indices.append(offset + BOARD_FEATURES + HAND_FEATURES + relative(kings[1-side]))
        output.append(sorted(indices))
    return output


def batch_features(positions, *, device=None):
    """Return padded int64 (batch, 2, 39); -1 denotes inactive slots."""
    import torch

    result = torch.full((len(positions), 2, MAX_ACTIVE), -1, dtype=torch.long, device=device)
    for row, position in enumerate(positions):
        for side, indices in enumerate(active_features(position)):
            result[row, side, :len(indices)] = torch.tensor(indices, device=device)
    return result


def generated_header():
    """Deterministic C++ table; verified in tests rather than generated at build time."""
    kinds = ", ".join(map(str, BOARD_TYPES))
    limits = ", ".join(map(str, HAND_LIMITS))
    return f'''// SPDX-License-Identifier: MIT
// Generated from hasami_learn/nnue/layout.py by features.generated_header().
#ifndef SHOGI_SEARCH_NNUE_LAYOUT_HPP
#define SHOGI_SEARCH_NNUE_LAYOUT_HPP
#include <array>
namespace shogi_search::nnue {{
inline constexpr char feature_set[] = "{FEATURE_SET}";
inline constexpr std::array<int, 13> board_types{{{kinds}}};
inline constexpr std::array<int, 7> hand_limits{{{limits}}};
inline constexpr int max_active = {MAX_ACTIVE};
inline constexpr int king_stride = {KING_STRIDE};
inline constexpr int feature_count = {FEATURE_COUNT};
inline constexpr float output_scale = {LAYOUT["output_scale"]}.0F;
}}
#endif
'''
