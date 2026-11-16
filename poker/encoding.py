"""Information-set planes as plain Python floats (no array dependency)."""

ENCODING_ID = 'poker-enc-v1'
NUM_PLANES = 141
BOARD_SHAPE = (4, 13)


def encode_observation(obs):
    planes = [[[0.0] * 13 for _ in range(4)] for _ in range(NUM_PLANES)]

    def cards(plane, values):
        for card in values:
            suit, rank = (card % 4, card // 4) if obs['variant'] == 'hulhe' else (card % 2, 9 + card // 2)
            planes[plane][suit][rank] = 1.0

    def constant(plane, value):
        planes[plane] = [[float(value)] * 13 for _ in range(4)]

    player = obs['player']
    cards(0, obs['hole_cards'])
    cards(1, obs['board'][:3])
    cards(2, obs['board'][3:4])
    cards(3, obs['board'][4:5])
    cards(4, obs['hole_cards'] + obs['board'])
    constant(5, player == obs['button'])
    constant(6 + obs['street'], 1)
    constant(10, obs['street_contributions'][player] / obs['bet_size'] / 4)
    constant(11, obs['street_contributions'][1 - player] / obs['bet_size'] / 4)
    constant(12, obs['to_call'] / 4)
    for street, history in enumerate(obs['history']):
        if len(history) > 8:
            raise ValueError('betting history exceeds the encoder contract')
        for slot, event in enumerate(history):
            if event['action'] == 0:
                continue
            offset = 0 if event['player'] == player else 2
            constant(13 + street * 32 + slot * 4 + offset + event['action'] - 1, 1)
    return planes


def encode(game, player):
    return encode_observation(game.observation(player))
