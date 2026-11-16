"""The fixed 45-action protocol; all targets are clockwise relative seats."""

ACTION_SIZE = 45
ROLES = ('villager', 'seer', 'werewolf', 'possessed')
ROLE_DECK = ('villager', 'villager', 'seer', 'werewolf', 'possessed')
RESULTS = ('human', 'werewolf')


def validate_player(player: int) -> None:
    if type(player) is not int or not 0 <= player < 5:
        raise ValueError('player must be an integer seat in 0..4')


def decode(action: int, player: int) -> dict:
    validate_player(player)
    if type(action) is not int or not 0 <= action < ACTION_SIZE:
        raise ValueError('action must be an integer in 0..44')
    if action == 0:
        return {'kind': 'SKIP'}
    if action <= 4:
        return {'kind': 'COMINGOUT', 'role': ROLES[action - 1]}
    if action <= 12:
        offset, result = divmod(action - 5, 2)
        return {'kind': 'DIVINED', 'target': (player + offset + 1) % 5, 'result': RESULTS[result]}
    if action <= 28:
        offset, role = divmod(action - 13, 4)
        return {'kind': 'ESTIMATE', 'target': (player + offset + 1) % 5, 'role': ROLES[role]}
    for kind, start in (('VOTE_INTENT', 29), ('vote', 33), ('attack', 37), ('divine', 41)):
        if action < start + 4:
            return {'kind': kind, 'target': (player + action - start + 1) % 5}
    raise AssertionError('unreachable action')


def target_action(kind: str, player: int, target: int, value: str | None = None) -> int:
    validate_player(player)
    validate_player(target)
    if target == player:
        raise ValueError('self targets are forbidden')
    offset = (target - player) % 5 - 1
    if kind == 'DIVINED':
        return 5 + 2 * offset + RESULTS.index(value)
    if kind == 'ESTIMATE':
        return 13 + 4 * offset + ROLES.index(value)
    starts = {'VOTE_INTENT': 29, 'vote': 33, 'attack': 37, 'divine': 41}
    if kind not in starts:
        raise ValueError('unknown targeted action')
    return starts[kind] + offset


def action_name(action: int) -> str:
    statement = decode(action, 0)
    return ' '.join(str(value) for value in statement.values())
