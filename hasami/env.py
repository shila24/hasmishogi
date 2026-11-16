"""A dependency-free turn-based environment, not a Gymnasium API claim."""
from __future__ import annotations
from dataclasses import dataclass
from . import create_game
from .model import Rules, require_int
from .observation import observe


@dataclass(frozen=True)
class StepResult:
    observation: dict
    rewards: tuple[float, float]
    terminated: bool
    truncated: bool
    info: dict


class TurnBasedEnv:
    def __init__(self, backend="python", rules=None, max_plies=1000):
        require_int(max_plies, "max_plies")
        if max_plies < 0:
            raise ValueError("max_plies must be nonnegative; zero disables")
        self.backend = backend
        self.rules = rules or Rules()
        self.max_plies = max_plies
        self.game = create_game(backend, self.rules)
        self.truncated = False

    def reset(self):
        self.game.close()
        self.game = create_game(self.backend, self.rules)
        self.truncated = False
        return observe(self.game)

    def step(self, action: int) -> StepResult:
        if self.truncated or self.game.outcome.terminal:
            raise ValueError("Reset is required after termination or truncation")
        actor = self.game.position.turn
        result = self.game.play(action)
        terminal = self.game.outcome.terminal
        self.truncated = bool(not terminal and self.max_plies and len(self.game.moves) >= self.max_plies)
        rewards = [0.0, 0.0]
        if terminal and result.outcome.winner in (0, 1):
            winner = result.outcome.winner
            rewards[winner], rewards[1 - winner] = 1.0, -1.0
        return StepResult(observe(self.game), tuple(rewards), terminal, self.truncated,
                          {"actor": actor, "action": action, "captured": hex(result.captured),
                           "winner": result.outcome.winner,
                           "reason": "max_plies" if self.truncated else result.outcome.reason})

    def close(self):
        self.game.close()
