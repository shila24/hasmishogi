"""Public observations for agents; no learning framework is imported."""
from collections import Counter
from .model import ACTION_SIZE, GameLike, Position, RULESET_ID


def observe(game: GameLike) -> dict:
    p = game.position
    legal = game.legal_actions()
    mask = [0] * ACTION_SIZE
    for action in legal:
        mask[action] = 1
    mine, enemy = p.pieces(p.turn), p.pieces(1 - p.turn)
    # Coordinates remain absolute even though piece channels are side-relative.
    flat = [[int(bool(board & (1 << s))) for s in range(81)] for board in (mine, enemy)]
    flat.extend([[int(p.pending == p.turn)] * 81,
                 [int(p.pending == 1 - p.turn)] * 81,
                 [p.turn] * 81])
    planes = [[plane[r * 9:(r + 1) * 9] for r in range(9)] for plane in flat]
    history = game.history_keys()
    return {
        "ruleset_id": RULESET_ID,
        "rules": game.rules.to_dict(),
        "state": p.to_dict(),
        "planes": planes,
        "action_mask": mask,
        "legal_actions": legal,
        "to_play": p.turn,
        "ply": len(game.moves),
        "captures": [p.captures(0), p.captures(1)],
        "history": [Position(*key).to_dict() for key in history],
        "history_counts": [dict(position=Position(*key).to_dict(), count=count)
                           for key, count in sorted(Counter(history).items())],
        "current_repetitions": history.count(p.key()),
        "outcome": {"winner": game.outcome.winner, "reason": game.outcome.reason},
    }
