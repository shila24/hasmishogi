"""The fixed 111-action table for riichi-4p-v1.

Discards name a type, not a physical copy: discard the drawn copy when its
kind matches, otherwise the lowest non-red copy (preserving a red five).
Calls likewise consume non-red copies first. Chi low/mid/high describes the
called tile's position in the sequence. Use 72 for exactly one kan option,
and 73 + type only when several options exist; aliases are never both legal.
"""

ACTION_SIZE = 111
RIICHI = 34
CHI_LOW, CHI_MID, CHI_HIGH = 68, 69, 70
PON, KAN, KAN_BASE = 71, 72, 73
RON, TSUMO, PASS, NINE_TERMINALS = 107, 108, 109, 110


def action_name(action: int) -> str:
    if type(action) is not int or not 0 <= action < ACTION_SIZE:
        raise ValueError('action must be an integer in 0..110')
    if action < 68 or KAN_BASE <= action < RON:
        tile = action % 34 if action < 68 else action - KAN_BASE
        code = f'{tile % 9 + 1}{"mpsz"[tile // 9]}'
        verb = 'discard' if action < 34 else 'riichi' if action < 68 else 'kan'
        return f'{verb} {code}'
    return {68: 'chi low', 69: 'chi mid', 70: 'chi high', 71: 'pon',
            72: 'kan', 107: 'ron', 108: 'tsumo', 109: 'pass',
            110: 'nine terminals'}[action]
