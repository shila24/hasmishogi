"""Canonical hlnn-kp-v1 description shared with the generated C++ table."""

from shogi.model import P, L, N, S, G, B, R, PP, PL, PN, PS, PB, PR, SQUARES

LAYOUT = {
    "feature_set": "hlnn-kp-v1",
    "board_types": [P, L, N, S, G, B, R, PP, PL, PN, PS, PB, PR],
    "hand_limits": [18, 4, 4, 4, 4, 2, 2],
    "squares": SQUARES,
    "max_active": 39,
    "output_scale": 600,
}
