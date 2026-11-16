"""Episode runners shared by all learners. Rules stay inside the selected game adapter."""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from hasami.agents import Agent, SearchLimits, check_cancel
from .games import get_spec, GameSpec
from hasami.model import GameLike
from .games import Encoder
from .games.base import as_v2, encode_player, terminal_rewards
from .signature_cache import accepts
from .encoding import sample_from_policy
from .search.draw import draw_reasons_for, draw_utility

ActorFn = Callable[..., tuple[int, np.ndarray | None, dict[str, Any]]]

# Draw contempt state. The trainer sets it once per iteration; runners stamp it onto every trajectory so
# learners never have to thread the schedule through their own code.
_DRAW_STATE: dict[str, Any] = {"value": 0.0, "reasons": tuple(getattr(get_spec(), "draw_reasons", ()))}


def draw_value_for_iteration(cfg: Any, iteration: int) -> float:
    """Linear anneal from draw_value_start to draw_value_end over draw_value_iterations."""
    start = float(getattr(cfg, "draw_value_start", 0.0))
    end = float(getattr(cfg, "draw_value_end", 0.0))
    span = int(getattr(cfg, "draw_value_iterations", 0) or 0)
    if span <= 0 or iteration >= span:
        return end
    fraction = max(0, iteration - 1) / span
    return start + (end - start) * fraction


def set_draw_value(value: float, reasons: list[str] | tuple[str, ...] | None = None) -> None:
    _DRAW_STATE["value"] = float(value)
    if reasons is not None:
        _DRAW_STATE["reasons"] = tuple(reasons)


def current_draw_value() -> float:
    return float(_DRAW_STATE["value"])


def current_draw_reasons(spec) -> tuple[str, ...]:
    return draw_reasons_for(spec, _DRAW_STATE["reasons"])


@dataclass
class Trajectory:
    """Recorded learner decisions of one game plus the fixed-player outcome."""

    game_id: str = "hasami"
    planes: list[np.ndarray] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    players: list[int] = field(default_factory=list)
    policies: list[np.ndarray | None] = field(default_factory=list)
    extras: list[dict[str, Any]] = field(default_factory=list)
    rewards: list[float] = field(default_factory=lambda: [0.0, 0.0])
    num_players: int = 2
    seed: int | None = None
    joint_actions: list[dict[int, int]] = field(default_factory=list)
    step_info: list[dict[str, Any]] = field(default_factory=list)
    truncated: bool = False
    winner: int = -2
    reason: str = "ongoing"
    plies: int = 0
    draw_value: float = 0.0  # contempt applied to draws whose reason is in draw_reasons (both sides)
    draw_reasons: tuple[str, ...] = field(default_factory=lambda: tuple(getattr(get_spec(), "draw_reasons", ())))

    def __len__(self) -> int:
        return len(self.actions)

    def append(self, planes: np.ndarray, mask: np.ndarray, action: int, player: int, policy: np.ndarray | None, extra: dict[str, Any]) -> None:
        self.planes.append(planes)
        self.masks.append(mask)
        self.actions.append(int(action))
        self.players.append(int(player))
        self.policies.append(policy)
        self.extras.append(extra)

    def outcome_value_for(self, index: int, truncation_value: float = 0.0) -> float:
        """Observing-player outcome component for recorded step ``index``.

        Capped games use ``truncation_value``; draws listed in ``draw_reasons`` use the
        annealed contempt ``draw_value`` (symmetric for both players); other draws are 0.
        """
        if self.truncated:
            return float(truncation_value)
        draw = draw_utility(self.winner, self.reason, self.draw_value, self.draw_reasons)
        if draw is not None:
            return draw
        return float(self.rewards[self.players[index]])

    def value_for(self, index: int, truncation_value: float = 0.0) -> float:
        """Outcome target optionally mixed with this mover's root search value."""
        outcome = self.outcome_value_for(index, truncation_value)
        extra = self.extras[index] if index < len(self.extras) else {}
        mix = float(extra.get("value_target_mix", 0.0))
        if not np.isfinite(mix) or not 0 <= mix <= 1:
            raise ValueError("value_target_mix must be finite and within [0, 1]")
        if not mix:
            return outcome
        root_value = float(extra["root_value"])
        if not np.isfinite(root_value):
            raise ValueError("root_value must be finite")
        return (1 - mix) * outcome + mix * root_value

    def values(self, truncation_value: float = 0.0) -> np.ndarray:
        return np.array([self.value_for(i, truncation_value) for i in range(len(self))], dtype=np.float32)

    def result_index(self) -> int:
        """Winner seat, or num_players for a draw/cap (CollectStats.wins)."""
        if 0 <= self.winner < self.num_players:
            return self.winner
        return self.num_players


def search_decision(result, mask, ply, rng, *, root_selection, temperature,
                    temperature_plies, value_target_mix=0.0):
    """Share the serial/spawned move rule and retain the actual training policy.

    Gumbel plays the halving winner before temperature_plies, then the legal
    improved-policy argmax (ascending ties). PUCT retains tempered sampling.
    """
    policy = result.policy_target
    if policy is None:
        policy = result.visits / result.visits.sum()
    if root_selection == "gumbel":
        action = (result.selected_action if ply < temperature_plies else
                  int(np.argmax(np.where(mask > 0, policy, -np.inf))))
        if type(action) is not int or not 0 <= action < len(mask) or mask[action] <= 0:
            raise ValueError("Gumbel search must select a legal action")
    else:
        action = sample_from_policy(policy, mask, rng, temperature if ply < temperature_plies else 0.0)
    return action, policy.astype(np.float32), {"root_value": result.root_value,
                                               "simulations": result.simulations,
                                               "root_selection": root_selection,
                                               "value_target_mix": value_target_mix}


def _call_actor(actor, game, planes, mask, ply, player):
    args = (game, planes, mask, ply)
    if accepts(actor, "actor", args + (player,), {}):
        return actor(*args, player)
    return actor(*args)


def choose_agent(agent, game, limits, cancel, player):
    """Pass seat identity when supported, preserving legacy Agent signatures."""
    if accepts(agent.choose, "choose", (game, limits, cancel), {"player": player}):
        return agent.choose(game, limits, cancel, player=player)
    return agent.choose(game, limits, cancel)


def run_table_game(
    seats: dict[int, ActorFn | Agent], encoder: Encoder, *, spec: GameSpec | None = None,
    backend: str = "python", max_plies: int | None = None, limits=None,
    rules=None, cancel: threading.Event | None = None, seed: int | None = None,
    on_finish=None,
) -> Trajectory:
    """Resolve joint steps; record callbacks and give each seat an isolated clone.

    A supplied seed reproduces both initial deals and chance inside steps. The
    generated seed and all joint steps are retained for exact archive replay.
    """
    spec = as_v2(spec or get_spec())
    if set(seats) != set(range(spec.num_players)):
        raise ValueError("seats must contain every player exactly once")
    cap = spec.default_max_plies if max_plies is None else max_plies
    if type(cap) is not int or cap < 0:
        raise ValueError("max_plies must be nonnegative")
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")
    seed = secrets.randbits(63) if seed is None else seed
    game = spec.create_game(backend, rules, seed=seed)
    trajectory = Trajectory(game_id=spec.game_id, num_players=spec.num_players,
                            rewards=[0.0] * spec.num_players, seed=seed)
    default_limits = SearchLimits(max_depth=1, time_ms=200, node_limit=20000)
    last = None
    try:
        terminal = game.outcome.terminal
        while not terminal and not trajectory.truncated:
            check_cancel(cancel)
            actors = spec.actors(game)
            if len(set(actors)) != len(actors) or any(type(p) is not int or p not in seats for p in actors):
                raise ValueError("Invalid actor seats")
            actions = {}
            for player in actors:
                mover = seats[player]
                # Copy on return: an adapter may reuse its mask buffer, and encoding the observation
                # may query legal masks of other seats before this one is consumed.
                mask = np.array(spec.legal_mask(game, player), copy=True)
                snapshot = game.clone()
                try:
                    if hasattr(mover, "choose"):
                        seat_limits = limits.get(player, default_limits) if isinstance(limits, dict) else limits or default_limits
                        action = choose_agent(mover, snapshot, seat_limits, cancel, player).action
                    else:
                        # Snapshot before the actor runs: a searching actor may re-encode its clone through
                        # the same encoder, and encoders may hand out a reused buffer.
                        planes = np.array(encode_player(encoder, snapshot, player), copy=True)
                        action, policy, extra = _call_actor(mover, snapshot, planes.copy(), mask.copy(), trajectory.plies, player)
                        extra = dict(extra, decision_ply=trajectory.plies)
                        trajectory.append(planes, mask, action, player, policy, extra)
                finally:
                    snapshot.close()
                if type(action) is not int or action not in spec.legal_actions(game, player):
                    raise ValueError("Actor returned an illegal action")
                actions[player] = action
            last = spec.step(game, actions)
            trajectory.joint_actions.append(dict(actions))
            trajectory.step_info.append(dict(last.info))
            trajectory.plies += 1
            terminal = bool(last.terminal)
            trajectory.truncated = bool(last.truncated or (not terminal and cap and trajectory.plies >= cap))
        if terminal and not trajectory.truncated:
            trajectory.rewards = terminal_rewards(spec, game)
        trajectory.winner = getattr(game.outcome, "winner", -2)
        trajectory.reason = ("max_plies" if trajectory.truncated and not (last and last.truncated)
                             else (last.info.get("reason", game.outcome.reason) if last else game.outcome.reason))
        trajectory.draw_value = current_draw_value()
        trajectory.draw_reasons = current_draw_reasons(spec)
        if on_finish is not None:
            on_finish(game, trajectory)
        return trajectory
    finally:
        game.close()


def run_selfplay_game(actor: ActorFn, encoder: Encoder, *, spec=None, **kwargs) -> Trajectory:
    """Play and record every seat with the shared actor callback."""
    spec = as_v2(spec or get_spec())
    return run_table_game(dict.fromkeys(range(spec.num_players), actor), encoder, spec=spec, **kwargs)


def run_vs_opponent_game(actor: ActorFn, opponent: Agent, learner_player: int,
                         encoder: Encoder, *, spec=None, **kwargs) -> Trajectory:
    """Record one learner seat and use the opponent for all remaining seats."""
    spec = as_v2(spec or get_spec())
    if type(learner_player) is not int or not 0 <= learner_player < spec.num_players:
        raise ValueError("Invalid learner player")
    seats = dict.fromkeys(range(spec.num_players), opponent)
    seats[learner_player] = actor
    return run_table_game(seats, encoder, spec=spec, **kwargs)


# Legacy callers may still import the Hasami mirror helper here.
from .games.hasami import augment_horizontal


def augment(spec, planes, mask, policy=None, action=None):
    """Yield each configured rule symmetry; the caller keeps the original sample."""
    for symmetry in spec.symmetries():
        yield (symmetry.planes(planes), symmetry.policy(mask),
               None if policy is None else symmetry.policy(policy),
               None if action is None else symmetry.action(action))
