"""Information-set encoding riichi-enc-v1: 120 planes of shape (4, 34).

Rows are observer, next seat, opposite, previous seat. Columns are tile types.
The layout is fixed, including normalization; unmentioned cells are zero:

  0:4    own hand count thresholds >= 1,2,3,4 (row 0)
  4      own red fives (row 0)
  5      own drawn tile (row 0)
  6      discard counts / 4 (including called discards)
  7      called discard counts / 4
  8:40   last 32 discards, oldest to newest, right aligned, one plane/event
  40:72  same windows, one-hot only when tsumogiri
  72:76  each of four meld slots: type counts / 4
  76:80  each meld's open flag at its tile types
  80:84  each meld's kan flag at its tile types
  84:88  each meld's red fives
  88:91  riichi, double riichi, ippatsu (constant per row)
  91     riichi declaration tile
  92:97  five visible dora indicators (one-hot, broadcast to all rows)
  97     discarded red fives, including called discards
  98     remaining live wall / 70 (global constant)
  99     score / 100000 (constant per row, no clipping)
  100    placement / 4 (constant per row; initial-seat tie order)
  101    round wind (one-hot, broadcast to all rows)
  102    seat wind (one-hot per row)
  103    honba / 10 (global constant, no clipping)
  104    table sticks / 10 (global constant, no clipping)
  105:108 own discard/temporary/riichi furiten (constant in row 0)
  108    own shanten / 8 (constant in row 0; complete = -1/8)
  109    own structural waits before the draw (row 0)
  110    dealer row (ones)
  111    turn/discarder/kan declarer row (ones)
  112:117 phase flags: draw, discard after call, claim, kan claim, terminal
  117    last discarded tile (one-hot in discarder row)
  118    pending kan type (one-hot in declarer row)
  119    bias (ones)

A post-call hand awaiting discard has no single pre-draw wait set: plane 109
is zero. Windows omit earlier order, but aggregate discards include it. No seed,
wall identities, ura indicators, other hands, other waits, other furiten, or
other players' legal actions are read. All input comes from observation(player).
The core returns ordinary Python lists; the optional adapter supplies float32.
"""

from .tiles import RED_IDS

ENCODING_ID = 'riichi-enc-v1'
NUM_PLANES = 120
BOARD_SHAPE = (4, 34)


def encode_observation(obs: dict) -> list[list[list[float]]]:
    planes = [[[0.0] * 34 for _ in range(4)] for _ in range(NUM_PLANES)]

    def row_constant(index: int, row: int, value: float) -> None:
        planes[index][row] = [float(value)] * 34

    def constant(index: int, value: float) -> None:
        for row in range(4):
            row_constant(index, row, value)

    for tile, count in enumerate(obs['hand_counts']):
        for level in range(4):
            planes[level][0][tile] = float(count > level)
    for physical in obs['hand']:
        if obs['red_fives'] and physical in RED_IDS:
            planes[4][0][physical // 4] = 1.0
    if obs['drawn_tile'] is not None:
        planes[5][0][obs['drawn_tile'] // 4] = 1.0
    for row in range(4):
        discards = obs['discards'][row]
        for discard in discards:
            tile = discard['tile'] // 4
            planes[6][row][tile] += 0.25
            planes[7][row][tile] += 0.25 * discard['called']
            if discard['riichi']:
                planes[91][row][tile] = 1.0
            if obs['red_fives'] and discard['tile'] in RED_IDS:
                planes[97][row][tile] = 1.0
        window = discards[-32:]
        for index, discard in enumerate(window, 32 - len(window)):
            tile = discard['tile'] // 4
            planes[8 + index][row][tile] = 1.0
            planes[40 + index][row][tile] = float(discard['tsumogiri'])
        for index, meld in enumerate(obs['melds'][row]):
            for physical in meld['tiles']:
                tile = physical // 4
                planes[72 + index][row][tile] += 0.25
                planes[76 + index][row][tile] = float(meld['open'])
                planes[80 + index][row][tile] = float(meld['kind'] == 'kan')
                if obs['red_fives'] and physical in RED_IDS:
                    planes[84 + index][row][tile] = 1.0
        for index, key in ((88, 'riichi'), (89, 'double_riichi'), (90, 'ippatsu'),
                           (99, 'scores'), (100, 'placement')):
            scale = 100000 if index == 99 else 4 if index == 100 else 1
            row_constant(index, row, obs[key][row] / scale)
        for index, physical in enumerate(obs['dora_indicators']):
            planes[92 + index][row][physical // 4] = 1.0
        planes[101][row][obs['round_wind']] = 1.0
        planes[102][row][obs['seat_winds'][row]] = 1.0
    constant(98, obs['remaining_wall'] / 70)
    constant(103, obs['honba'] / 10)
    constant(104, obs['riichi_sticks'] / 10)
    for index, flag in enumerate(obs['furiten']):
        row_constant(105 + index, 0, flag)
    row_constant(108, 0, obs['shanten'] / 8)
    for tile in obs['waits']:
        planes[109][0][tile] = 1.0
    row_constant(110, obs['dealer'], 1)
    row_constant(111, obs['turn'], 1)
    phases = ('draw', 'discard', 'claim', 'kan_claim', 'terminal')
    constant(112 + phases.index(obs['phase']), 1)
    if obs['last_discard'] is not None:
        row, physical = obs['last_discard']
        planes[117][row][physical // 4] = 1.0
    if obs['pending_kan'] is not None:
        row, _, tile = obs['pending_kan']
        planes[118][row][tile] = 1.0
    constant(119, 1)
    return planes


def encode(game, player: int) -> list[list[list[float]]]:
    return encode_observation(game.observation(player))
