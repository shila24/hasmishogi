"""Multi-table self-play with shared, fixed-shape batched evaluation (docs/METHOD_ja.md#methods).

``run_table_games`` plays several tables round by round. Seats that implement the ``BatchSeat`` protocol never
receive a game: the runner encodes their decisions, evaluates every pending request of one network in a single
batched forward, and lets each seat decide from its own row. Actor functions and ``Agent``s are served exactly as
in ``run_table_game``. Randomness comes from per-decision streams (``decision_rng``), so results do not depend on
scheduling, chunking or skipped decisions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

import numpy as np
import torch

from hasami.agents import Decision, SearchLimits, check_cancel
from .devices import autocast_context
from .games import get_spec
from .games.base import as_v2, encode_player, terminal_rewards
from .inference import policy_action
from .selfplay import Trajectory, _call_actor, choose_agent, current_draw_value, current_draw_reasons

STREAM_OPPONENTS, STREAM_POLICY, STREAM_AGENT = 1, 2, 3


def stream_prefix(run_seed: int, namespace: int, iteration: int) -> tuple[int, int, int]:
    """Entropy prefix shared by every stream of one collection (spec section 3.1)."""
    values = (int(run_seed), int(namespace), int(iteration))
    if any(v < 0 for v in values):
        raise ValueError("stream entropy must be nonnegative")
    return values


def opponent_rng(prefix: tuple[int, int, int]) -> np.random.Generator:
    return np.random.default_rng([*prefix, STREAM_OPPONENTS])


def decision_rng(prefix: tuple[int, int, int], table: int, seat: int, ply: int) -> np.random.Generator:
    """The generator of one stochastic decision; independent of every other decision."""
    return np.random.default_rng([*prefix, STREAM_POLICY, table, seat, ply])


def agent_seed(prefix: tuple[int, int, int], table: int, seat: int) -> int:
    """Seed for a per-(table, seat) instance of an agent with private RNG state."""
    state = np.random.SeedSequence([*prefix, STREAM_AGENT, table, seat]).generate_state(1, np.uint64)[0]
    return int(state) % (2 ** 63)


@dataclass
class CollectionTiming:
    """Counters of one collection (spec section 3.3); seconds are wall-clock.

    Phases named in ``unmeasured`` were not timed by the runner that played the collection and are reported as
    ``None``, so an unavailable measurement is never read as zero seconds.
    """

    decisions: int = 0
    evaluated: int = 0
    forced_skipped: int = 0
    forward_calls: int = 0
    padded_rows: int = 0
    max_batch: int = 0
    encode: float = 0.0
    inference: float = 0.0
    decide: float = 0.0
    env: float = 0.0
    agents: float = 0.0
    total: float = 0.0
    unmeasured: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        rows = self.evaluated
        return {"decisions": self.decisions, "evaluated": rows, "forced_skipped": self.forced_skipped,
                "forward_calls": self.forward_calls, "padded_rows": self.padded_rows,
                "mean_batch": round(rows / self.forward_calls, 3) if self.forward_calls else 0.0,
                "max_batch": self.max_batch,
                **{name: None if name in self.unmeasured else round(getattr(self, name), 4)
                   for name in ("encode", "inference", "decide", "env", "agents", "total")}}


class BatchEvaluator:
    """Batched policy/value forwards of one network; with ``fixed_shape`` every chunk has ``batch_size`` rows.

    Padding removes shape changes, but does not guarantee identical arithmetic at different row positions:
    CPU linear kernels can round their tail rows differently. Exact CPU league collection therefore uses
    ``batch_size=1`` for both runners and every policy seat. GPU callers retain device-specific batching.
    No auxiliary heads are requested.
    """

    def __init__(self, network: torch.nn.Module, device: torch.device | str, amp: str = "auto",
                 batch_size: int = 8, fixed_shape: bool = True, timing: CollectionTiming | None = None) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.network = network
        self.device = torch.device(device)
        self.amp = amp
        self.batch_size = batch_size
        self.fixed_shape = bool(fixed_shape)
        self.timing = timing if timing is not None else CollectionTiming()

    def forward(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        planes = np.asarray(planes, dtype=np.float32)
        if planes.ndim < 2 or not len(planes):
            raise ValueError("forward expects a non-empty batch of planes")
        start = time.perf_counter()
        was_training = self.network.training
        self.network.eval()
        logits, values = [], []
        try:
            with torch.inference_mode(), autocast_context(self.device, self.amp):
                for offset in range(0, len(planes), self.batch_size):
                    chunk = planes[offset: offset + self.batch_size]
                    rows = len(chunk)
                    if self.fixed_shape and rows < self.batch_size:
                        padded = np.zeros((self.batch_size,) + chunk.shape[1:], dtype=np.float32)
                        padded[:rows] = chunk
                        self.timing.padded_rows += self.batch_size - rows
                        chunk = padded
                    x = torch.from_numpy(np.ascontiguousarray(chunk)).to(self.device)
                    head, value = self.network(x)
                    if value is None:
                        raise ValueError("BatchEvaluator needs a policy_value network")
                    logits.append(head[:rows].float().cpu().numpy())
                    values.append(value[:rows].float().reshape(rows).cpu().numpy())
                    self.timing.forward_calls += 1
                    self.timing.max_batch = max(self.timing.max_batch, rows)
        finally:
            if was_training:
                self.network.train()
        self.timing.evaluated += len(planes)
        self.timing.inference += time.perf_counter() - start
        return np.concatenate(logits), np.concatenate(values)


@dataclass(frozen=True)
class DecisionRequest:
    table: int
    player: int
    ply: int
    planes: np.ndarray
    mask: np.ndarray


@runtime_checkable
class BatchSeat(Protocol):
    """A seat decided from batched network outputs; it never receives the game."""

    evaluator: BatchEvaluator
    encoder: Any
    records: bool

    def needs_evaluation(self, request: DecisionRequest) -> bool: ...

    def decide(self, request: DecisionRequest, logits: np.ndarray | None,
               value: float | None) -> tuple[int, Any, dict[str, Any]]: ...


class PolicySeat:
    """The ``PolicyAgent`` decision rule (``policy_action``) served from a shared evaluator.

    A decision with exactly one legal action is that action whatever the network says, so non-recording seats
    skip its evaluation; with per-decision random streams this cannot affect any other decision.
    """

    records = False

    def __init__(self, evaluator: BatchEvaluator, encoder, temperature: float,
                 rng_for: Callable[[int, int, int], np.random.Generator], skip_forced: bool = True) -> None:
        self.evaluator = evaluator
        self.encoder = encoder
        self.temperature = float(temperature)
        self.rng_for = rng_for
        self.skip_forced = bool(skip_forced)

    @classmethod
    def from_agent(cls, agent, evaluator: BatchEvaluator, rng_for, skip_forced: bool = True) -> "PolicySeat":
        from .inference import PolicyAgent

        if type(agent) is not PolicyAgent:
            raise TypeError("PolicySeat.from_agent requires exactly a PolicyAgent")
        if evaluator.network is not agent.network:
            raise ValueError("the evaluator must serve the agent's own network")
        return cls(evaluator, agent.encoder, agent.temperature, rng_for, skip_forced)

    def needs_evaluation(self, request: DecisionRequest) -> bool:
        return not (self.skip_forced and int(np.count_nonzero(request.mask > 0)) == 1)

    def decide(self, request, logits, value):
        legal = np.flatnonzero(request.mask > 0)
        if not legal.size:
            raise ValueError("No legal action is available")
        if logits is None:
            return int(legal[0]), None, {}
        rng = self.rng_for(request.table, request.player, request.ply) if self.temperature > 0 else None
        return int(policy_action(logits, request.mask, rng, self.temperature)), None, {}


def _single(seat, table: int, player: int, ply: int, planes: np.ndarray, mask: np.ndarray):
    request = DecisionRequest(table, player, ply, planes, mask)
    logits, values = seat.evaluator.forward(planes[None])
    return seat.decide(request, logits[0], float(values[0]))


class SeatActor:
    """Reference adapter: a recording seat as a ``run_table_game`` actor function (one row per forward)."""

    def __init__(self, seat, table: int) -> None:
        self.seat, self.table = seat, table

    def __call__(self, game, planes, mask, ply, player):
        return _single(self.seat, self.table, player, ply, np.array(planes, dtype=np.float32, copy=True),
                       np.array(mask, copy=True))


class SeatAgent:
    """Reference adapter: a non-recording seat as an ``Agent`` evaluating every decision (no skipping)."""

    def __init__(self, seat, table: int, spec) -> None:
        self.seat, self.table, self.spec = seat, table, spec

    def choose(self, game, limits, cancel=None, *, player=None):
        check_cancel(cancel)
        planes = np.array(encode_player(self.seat.encoder, game, player), dtype=np.float32, copy=True)
        mask = np.array(self.spec.legal_mask(game, player), copy=True)
        # ``run_table_game`` does not pass the ply to agents; the decision stream needs the joint-step ply, which
        # the engines record as one ``moves`` entry per joint step (riichi and hasami; checked by the parity tests).
        moves = getattr(game, "moves", None)
        if moves is None:
            raise TypeError("SeatAgent needs game.moves (one entry per joint step) to derive the decision ply")
        ply = len(moves)
        action, _, _ = _single(self.seat, self.table, player, ply, planes, mask)
        return Decision(int(action))


@dataclass
class TableRequest:
    seats: dict[int, Any]
    seed: int
    table: int


@dataclass
class _Table:
    request: TableRequest
    game: Any
    trajectory: Trajectory
    last: Any = None
    done: bool = False
    pending: dict[int, Any] = field(default_factory=dict)


def _finish(spec, table: _Table) -> None:
    game, trajectory, last = table.game, table.trajectory, table.last
    terminal = bool(last.terminal) if last is not None else bool(game.outcome.terminal)
    if terminal and not trajectory.truncated:
        trajectory.rewards = terminal_rewards(spec, game)
    trajectory.winner = getattr(game.outcome, "winner", -2)
    trajectory.reason = ("max_plies" if trajectory.truncated and not (last and last.truncated)
                         else (last.info.get("reason", game.outcome.reason) if last else game.outcome.reason))
    trajectory.draw_value = current_draw_value()
    trajectory.draw_reasons = current_draw_reasons(spec)
    table.done = True


def run_table_games(tables: Sequence[TableRequest], encoder, *, spec=None, backend: str = "python",
                    max_plies: int | None = None, limits=None, cancel=None,
                    timing: CollectionTiming | None = None) -> list[Trajectory]:
    """Play every table to the end; per table the result equals ``run_table_game`` with the same seats."""
    spec = as_v2(spec or get_spec())
    timing = timing if timing is not None else CollectionTiming()
    cap = spec.default_max_plies if max_plies is None else max_plies
    if type(cap) is not int or cap < 0:
        raise ValueError("max_plies must be nonnegative")
    default_limits = SearchLimits(max_depth=1, time_ms=200, node_limit=20000)
    started = time.perf_counter()
    states: list[_Table] = []
    completed = False
    try:
        for request in tables:
            if set(request.seats) != set(range(spec.num_players)):
                raise ValueError("seats must contain every player exactly once")
            if type(request.seed) is not int or request.seed < 0:
                raise ValueError("seed must be a nonnegative integer")
            game = spec.create_game(backend, None, seed=request.seed)
            trajectory = Trajectory(game_id=spec.game_id, num_players=spec.num_players,
                                    rewards=[0.0] * spec.num_players, seed=request.seed)
            states.append(_Table(request, game, trajectory, done=bool(game.outcome.terminal)))
            if states[-1].done:
                _finish(spec, states[-1])
        while not all(state.done for state in states):
            check_cancel(cancel)
            pending: list[tuple[_Table, Any, DecisionRequest]] = []
            for state in states:
                if state.done:
                    continue
                state.pending = {}
                tick = time.perf_counter()
                actors = spec.actors(state.game)
                if len(set(actors)) != len(actors) or any(type(p) is not int or p not in state.request.seats
                                                         for p in actors):
                    raise ValueError("Invalid actor seats")
                timing.env += time.perf_counter() - tick
                for player in actors:
                    seat = state.request.seats[player]
                    timing.decisions += 1
                    tick = time.perf_counter()
                    mask = np.array(spec.legal_mask(state.game, player), copy=True)
                    if isinstance(seat, BatchSeat):
                        planes = np.array(encode_player(seat.encoder, state.game, player), dtype=np.float32, copy=True)
                        timing.encode += time.perf_counter() - tick
                        request = DecisionRequest(state.request.table, player, state.trajectory.plies, planes, mask)
                        pending.append((state, seat, request))
                        state.pending[player] = None
                        continue
                    snapshot = state.game.clone()
                    try:
                        if hasattr(seat, "choose"):
                            seat_limits = (limits.get(player, default_limits) if isinstance(limits, dict)
                                           else limits or default_limits)
                            action = choose_agent(seat, snapshot, seat_limits, cancel, player).action
                            state.pending[player] = (action, None)
                        else:
                            planes = np.array(encode_player(encoder, snapshot, player), copy=True)
                            action, policy, extra = _call_actor(seat, snapshot, planes.copy(), mask.copy(),
                                                                state.trajectory.plies, player)
                            extra = dict(extra, decision_ply=state.trajectory.plies)
                            state.pending[player] = (action, (planes, mask, policy, extra))
                    finally:
                        snapshot.close()
                    timing.agents += time.perf_counter() - tick
            # One batched forward per evaluator over every table's pending requests (first-seen order).
            groups: dict[int, tuple[BatchEvaluator, list[int]]] = {}
            for index, (_, seat, request) in enumerate(pending):
                if seat.needs_evaluation(request):
                    groups.setdefault(id(seat.evaluator), (seat.evaluator, []))[1].append(index)
                else:
                    timing.forced_skipped += 1
            outputs: dict[int, tuple[np.ndarray, float]] = {}
            for evaluator, indices in groups.values():
                logits, values = evaluator.forward(np.stack([pending[i][2].planes for i in indices]))
                for row, index in enumerate(indices):
                    outputs[index] = (logits[row], float(values[row]))
            tick = time.perf_counter()
            for index, (state, seat, request) in enumerate(pending):
                logits, value = outputs.get(index, (None, None))
                action, policy, extra = seat.decide(request, logits, value)
                record = None
                if seat.records:
                    record = (request.planes, request.mask, policy, dict(extra, decision_ply=request.ply))
                state.pending[request.player] = (action, record)
            timing.decide += time.perf_counter() - tick
            tick = time.perf_counter()
            for state in states:
                if state.done:
                    continue
                actions = {}
                for player, (action, record) in state.pending.items():
                    if type(action) is not int or action not in spec.legal_actions(state.game, player):
                        raise ValueError("Actor returned an illegal action")
                    if record is not None:
                        planes, mask, policy, extra = record
                        state.trajectory.append(planes, mask, action, player, policy, extra)
                    actions[player] = action
                state.last = spec.step(state.game, actions)
                state.trajectory.joint_actions.append(dict(actions))
                state.trajectory.step_info.append(dict(state.last.info))
                state.trajectory.plies += 1
                terminal = bool(state.last.terminal)
                state.trajectory.truncated = bool(state.last.truncated or (
                    not terminal and cap and state.trajectory.plies >= cap))
                if terminal or state.trajectory.truncated:
                    _finish(spec, state)
            timing.env += time.perf_counter() - tick
        completed = True
        return [state.trajectory for state in states]
    finally:
        # Close every game even if one close fails; a close error surfaces only when nothing else is propagating.
        close_error = None
        for state in states:
            try:
                state.game.close()
            except Exception as exc:  # noqa: BLE001
                close_error = close_error or exc
        timing.total += time.perf_counter() - started
        if close_error is not None and completed:
            raise close_error
