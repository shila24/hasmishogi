"""Stateless history planes stacked on a base encoder (for example ``riichi-history``).

The wrapper keeps no per-game state. It renders the last ``k`` joint steps of the engine's own
joint-action log through a game-specific ``ActionLayout``, so clones, parallel workers and a
searching agent's re-encodes all produce identical planes, and a checkpoint manifest restores the
encoder from ``encoding_id`` plus ``encoder_params`` alone. Observations are copied on return.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from .games.base import encode_player


class ActionLayout(Protocol):
    """Render one actor's action into the ``planes_per_step`` planes reserved for a joint step."""

    planes_per_step: int

    def render(self, planes: np.ndarray, row: int, action: int) -> None: ...


class ScalarActionLayout:
    """Generic layout: the action id one-hot folded over ``(planes, H, W)`` plus an actor-row plane."""

    def __init__(self, board_shape: tuple[int, int], action_size: int) -> None:
        height, width = (int(n) for n in board_shape)
        if height <= 0 or width <= 0 or type(action_size) is not int or action_size <= 0:
            raise ValueError("board_shape and action_size must be positive")
        self.board_shape = (height, width)
        self.action_size = action_size
        self.planes_per_step = -(-action_size // (height * width)) + 1

    def render(self, planes: np.ndarray, row: int, action: int) -> None:
        if not 0 <= action < self.action_size:
            raise ValueError(f"action {action} outside the action table")
        height, width = self.board_shape
        plane, cell = divmod(int(action), height * width)
        planes[plane, cell // width, cell % width] = 1.0
        planes[-1, row, :] = 1.0


def joint_history(spec, game) -> list[dict[int, int]]:
    """The complete decision history in adapter action space.

    Uses ``spec.history_actions`` when the adapter translates action ids, else the engine's own
    per-step ``{player: action}`` moves. Engines that record scalar moves without a hook cannot be
    stacked, because the actor of each move is not recoverable here.
    """
    hook = getattr(spec, "history_actions", None)
    if hook is not None:
        return [dict(step) for step in hook(game)]
    moves = list(getattr(game, "moves", None) or [])
    if moves and not all(isinstance(step, dict) for step in moves):
        raise ValueError("history stacking needs joint-action steps or a history_actions hook")
    return [{int(player): int(action) for player, action in step.items()} for step in moves]


class HistoryEncoder:
    """``base`` planes followed by ``k`` blocks of ``layout.planes_per_step`` planes, oldest first.

    Rows of the history planes are seats relative to the observer (``(actor - player) % num_players``,
    folded modulo the board height), matching seat-relative base encodings. Fewer than ``k`` recorded
    steps leave the oldest blocks zero; ``history`` selects which steps count (for riichi, the current
    hand only).
    """

    def __init__(self, base, *, k: int, layout: ActionLayout, num_players: int, encoding_id: str,
                 history: Callable[[object], list[dict[int, int]]]) -> None:
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer number of joint steps")
        if type(num_players) is not int or num_players < 1:
            raise ValueError("num_players must be a positive integer")
        self.base = base
        self.k = k
        self.layout = layout
        self.num_players = num_players
        self.encoding_id = encoding_id
        self.history = history
        self.base_planes = int(base.num_planes)
        self.num_planes = self.base_planes + k * int(layout.planes_per_step)
        self.action_size = getattr(base, "action_size", None)

    def resolve_player(self, game, player):
        if player is None:
            actors = game.actors()
            if len(actors) != 1:
                raise ValueError("specify player for simultaneous or terminal observations")
            return actors[0]
        return player

    def encode(self, game, player=None) -> np.ndarray:
        player = int(self.resolve_player(game, player))
        base = np.asarray(encode_player(self.base, game, player), dtype=np.float32)
        if base.ndim != 3 or base.shape[0] != self.base_planes:
            raise ValueError("base encoder returned planes of an unexpected shape")
        height = base.shape[1]
        planes = np.zeros((self.num_planes,) + base.shape[1:], dtype=np.float32)
        planes[: self.base_planes] = base
        steps = list(self.history(game))[-self.k:]
        per_step = int(self.layout.planes_per_step)
        offset = self.base_planes + (self.k - len(steps)) * per_step
        for index, step in enumerate(steps):
            block = planes[offset + index * per_step: offset + (index + 1) * per_step]
            for actor, action in step.items():
                row = ((int(actor) - player) % self.num_players) % height
                self.layout.render(block, row, int(action))
        return planes

    def legal_mask(self, game, player=None) -> np.ndarray:
        return self.base.legal_mask(game, player)
