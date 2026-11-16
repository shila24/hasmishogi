"""Independent rational-arithmetic Leduc best response for call/uniform opponents.

This reference recursively partitions weighted deals by the responder's visible
cards. It has its own betting transitions and never imports the poker engine.
"""

from fractions import Fraction
from itertools import permutations


def best_response(responder, uniform):
    worlds = [(deal, Fraction(1, 120)) for deal in permutations(range(6), 3)]

    def solve(worlds, turn, street, paid, round_paid, count, acted):
        if turn == responder:
            groups = {}
            for deal, weight in worlds:
                key = (deal[responder], deal[2] if street else None)
                groups.setdefault(key, []).append((deal, weight))
            return sum(max(branch(group, action, turn, street, paid, round_paid, count, acted)
                           for action in legal(turn, round_paid, count)) for group in groups.values())
        actions = legal(turn, round_paid, count) if uniform else [1]
        scaled = [(d, w / len(actions)) for d, w in worlds]
        return sum(branch(scaled, a, turn, street, paid, round_paid, count, acted) for a in actions)

    def legal(turn, paid, count):
        return ([0] if paid[turn] < paid[1 - turn] else []) + [1] + ([2] if count < 2 else [])

    def branch(worlds, action, turn, street, paid, round_paid, count, acted):
        if action == 0:
            value = -paid[responder] if turn == responder else paid[turn]
            return sum(w for _, w in worlds) * value
        amount = max(round_paid) - round_paid[turn] + (2 * (street + 1) if action == 2 else 0)
        paid = tuple(p + (amount if i == turn else 0) for i, p in enumerate(paid))
        round_paid = tuple(p + (amount if i == turn else 0) for i, p in enumerate(round_paid))
        acted = acted | {turn}
        if action == 1 and len(acted) == 2 and round_paid[0] == round_paid[1]:
            if street == 0:
                return solve(worlds, 0, 1, paid, (0, 0), 0, set())
            result = Fraction(0)
            for deal, weight in worlds:
                strengths = [(deal[p] // 2 == deal[2] // 2, deal[p] // 2) for p in (0, 1)]
                if strengths[responder] != strengths[1 - responder]:
                    result += weight * (paid[1 - responder] if strengths[responder] > strengths[1 - responder]
                                        else -paid[responder])
            return result
        return solve(worlds, 1 - turn, street, paid, round_paid, count + (action == 2), acted)

    return solve(worlds, 0, 0, (1, 1), (0, 0), 0, set())
