"""History-preserving PUCT/Gumbel search with batched inference across games.

Each tree has at most one outstanding leaf, so virtual loss is unnecessary;
the config field is reserved for callers using the shared MCTS interface.
No scores or nodes are shared between histories or successive searches.
"""

from __future__ import annotations

from ..games.base import as_v2, require_search_game

import inspect
import math
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from hasami.agents import AgentCancelled, check_cancel
from hasami.model import GameLike
from ..devices import autocast_context, resolve_device, to_device
from ..games import get_spec
from ..games import Encoder
from ..encoding import masked_softmax
from .draw import draw_reasons_for, draw_utility

Evaluator = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass
class MctsConfig:
    simulations: int = 64
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    add_noise: bool = True
    batch_size: int = 8
    virtual_loss: float = 1.0
    # Value of a state the adapter truncated (StepOutcome.truncated); mirrors selfplay.truncation_value,
    # which the training targets use for every player of a capped game (no zero-sum negation).
    truncation_value: float = 0.0
    # Explicit search utility; inference defaults never inherit trainer globals.
    draw_value: float = 0.0
    draw_reasons: tuple[str, ...] = ()
    root_selection: str = "puct"
    gumbel_m: int = 16
    gumbel_c_visit: float = 50.0
    gumbel_c_scale: float = 1.0

    def __post_init__(self) -> None:
        _integer_budget(self.simulations, "simulations")
        _integer_budget(self.batch_size, "batch_size", minimum=1)
        _integer_budget(self.gumbel_m, "gumbel_m", minimum=1)
        if self.root_selection not in ("puct", "gumbel"):
            raise ValueError("root_selection must be 'puct' or 'gumbel'")
        for name in ("c_puct", "virtual_loss", "dirichlet_alpha", "dirichlet_epsilon",
                     "gumbel_c_visit", "gumbel_c_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.truncation_value) or abs(self.truncation_value) > 1:
            raise ValueError("truncation_value must be finite and within [-1, 1]")
        if not math.isfinite(self.draw_value) or abs(self.draw_value) > 1:
            raise ValueError("draw_value must be finite and within [-1, 1]")
        if not isinstance(self.draw_reasons, (list, tuple)) or any(
                not isinstance(reason, str) or not reason for reason in self.draw_reasons):
            raise ValueError("draw_reasons must be a list or tuple of nonempty strings")
        self.draw_reasons = tuple(self.draw_reasons)
        if self.dirichlet_alpha == 0:
            raise ValueError("dirichlet_alpha must be positive")
        if self.dirichlet_epsilon > 1:
            raise ValueError("dirichlet_epsilon must be at most one")


def _integer_budget(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass
class MctsResult:
    """Root edge counts, mean backed-up value, and completed simulations.

    With zero completed simulations, visits contains the normalized root prior
    instead of counts. Terminal roots have no actions and return all zeros.
    ``q_values`` holds, per action id, the mean backed-up value of each visited root
    edge from the root mover's perspective, and NaN for unvisited or illegal actions.
    ``policy_target`` is normalized visits (PUCT) or the completed-Q improved
    policy (Gumbel). ``selected_action`` is the Gumbel halving winner, else None.
    """

    visits: np.ndarray
    root_value: float
    simulations: int
    q_values: np.ndarray | None = None
    policy_target: np.ndarray | None = None
    selected_action: int | None = None


@dataclass
class _Node:
    actions: np.ndarray | None = None
    prior: np.ndarray | None = None
    visits: np.ndarray | None = None
    value_sum: np.ndarray | None = None
    children: dict[int, _Node] = field(default_factory=dict)
    total: int = 0
    value: float = 0.0
    player: int = -1  # perspective of ``value``: the player to move here (side reported by the engine at a terminal)
    stopped: bool = False  # adapter-truncated state: a leaf holding truncation_value for every player, never stepped
    symmetric: bool = False  # terminal draw utility is identical for both players

    def expand(self, mask: np.ndarray, probs: np.ndarray, value: float) -> None:
        self.actions = np.flatnonzero(mask > 0)
        if not len(self.actions):
            raise ValueError("A nonterminal MCTS leaf must have legal actions")
        prior = np.asarray(probs, dtype=np.float64)[self.actions].copy()
        if not np.isfinite(prior).all() or (prior < 0).any() or prior.sum() <= 0:
            raise ValueError("Evaluator must return a finite nonnegative legal policy")
        if not math.isfinite(value):
            raise ValueError("Evaluator must return finite values")
        self.prior = prior / prior.sum()
        self.visits = np.zeros(len(prior), dtype=np.int64)
        self.value_sum = np.zeros(len(prior), dtype=np.float64)
        self.value = float(value)

    def select(self, c_puct: float) -> int:
        q = self.value_sum / np.maximum(self.visits, 1)
        u = c_puct * self.prior * math.sqrt(max(1, self.total)) / (1 + self.visits)
        return int(np.argmax(q + u))

    def completed_q(self) -> np.ndarray:
        """Unvisited edges use the paper's network/visited-prior value mix."""
        seen = self.visits > 0
        q = self.value_sum / np.maximum(self.visits, 1)
        mass = self.prior[seen].sum()
        mixed = self.value
        if self.total and mass > 0:
            weighted = (self.prior[seen] * q[seen]).sum() / mass
            mixed = (self.value + self.total * weighted) / (1 + self.total)
        return np.where(seen, q, mixed)

    def improved_logits(self, config: MctsConfig) -> np.ndarray:
        # Scalar libm operations match the native implementation. The evaluator
        # supplies probabilities; log(prior) differs from logits by a constant.
        logits = np.array([math.log(p) if p > 0 else -math.inf for p in self.prior])
        scale = (config.gumbel_c_visit + int(self.visits.max())) * config.gumbel_c_scale
        return logits + scale * self.completed_q()

    def improved_policy(self, config: MctsConfig) -> np.ndarray:
        logits = self.improved_logits(config)
        peak = float(logits.max())
        probabilities = np.array([math.exp(float(x) - peak) for x in logits])
        return probabilities / probabilities.sum()

    def select_improved(self, config: MctsConfig) -> int:
        """Equation 14: match visits to the completed-Q improved policy."""
        return int(np.argmax(self.improved_policy(config) - self.visits / (1 + self.total)))


class _SequentialHalving:
    """Budget-exact rounds, ascending action ties, ceiling half for odd sets.

    Sample min(m, legal, max(1, budget)) candidates. A phase receives complete
    rounds from remaining_budget / remaining_phases, at least one round when
    affordable. The last phase uses all remaining simulations; a partial round
    visits ascending actions first. A single candidate receives the remainder.
    Cancellation returns the best currently surviving candidate without backup.
    """

    def __init__(self, root, config, noise, budget):
        self.config, self.noise, self.budget = config, noise, budget
        # Sampling uses only log prior + Gumbel; completed Q is for elimination.
        scores = np.array([math.log(p) if p > 0 else -math.inf for p in root.prior]) + noise
        count = min(config.gumbel_m, len(scores), max(1, budget))
        self.candidates = sorted(sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:count])
        self.begin = 0
        self.end = self._phase_end(0)

    def _phase_end(self, total):
        count = len(self.candidates)
        remaining = self.budget - total
        phases = max(1, (count - 1).bit_length())
        allocation = remaining if phases == 1 else max(count, remaining // (phases * count) * count)
        return total + min(remaining, allocation)

    def _rank(self, root):
        scores = root.improved_logits(self.config) + self.noise
        return sorted(self.candidates, key=lambda i: (-scores[i], i))

    def select(self, root):
        if root.total >= self.budget:
            raise ValueError("Sequential halving simulation budget exhausted")
        if root.total >= self.end and len(self.candidates) > 1:
            self.candidates = sorted(self._rank(root)[:(len(self.candidates) + 1) // 2])
            self.begin = root.total
            self.end = self._phase_end(root.total)
        return self.candidates[(root.total - self.begin) % len(self.candidates)]

    def winner(self, root):
        return self._rank(root)[0]


def batched_evaluator(network, device, amp: str = "auto") -> Evaluator:
    """Evaluate (B,C,H,W) planes and (B,action_size) masks without gradients."""
    device = resolve_device(device)

    @torch.no_grad()
    def evaluate(planes: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        was_training = network.training
        network.eval()
        try:
            batch = to_device({"planes": np.ascontiguousarray(planes, dtype=np.float32),
                               "mask": np.ascontiguousarray(masks, dtype=np.float32)}, device)
            with autocast_context(device, amp):
                logits, values = network(batch["planes"])
            if values is None:
                raise ValueError("MCTS requires a policy_value network")
            probs = masked_softmax(logits.float(), batch["mask"])
            return probs.cpu().numpy(), values.float().cpu().numpy()
        finally:
            network.train(was_training)

    return evaluate


class MCTS:
    def __init__(self, evaluate_fn: Evaluator, encoder: Encoder,
                 config: MctsConfig, rng: np.random.Generator, spec=None) -> None:
        self.spec = as_v2(spec or get_spec())
        require_search_game(self.spec)
        # Simulations advance through the adapter's step(); the state is restored by undo() only
        # when the adapter declares that undo reverses step exactly, else by cloning per simulation.
        self.undoable = bool(getattr(self.spec, "supports_undo", False))
        self.evaluate_fn = evaluate_fn
        self.encoder = encoder
        # Resolve the encoder signature once (legacy encoders take no player) instead of per leaf.
        try:
            inspect.signature(encoder.encode).bind(None, 0)
            self._encode = encoder.encode
        except TypeError:
            self._encode = lambda work, player: encoder.encode(work)
        self.config = config
        self.rng = rng

    @staticmethod
    def _stopped(cancel, deadline) -> bool:
        try:
            check_cancel(cancel)
        except AgentCancelled:
            return True
        return deadline is not None and time.perf_counter() >= deadline

    def _expand_many(self, leaves: list) -> None:
        for begin in range(0, len(leaves), self.config.batch_size):
            batch = leaves[begin:begin + self.config.batch_size]
            planes = np.stack([item[1] for item in batch])
            masks = np.stack([item[2] for item in batch])
            probs, values = self.evaluate_fn(planes, masks)
            probs, values = np.asarray(probs), np.asarray(values)
            if probs.shape != masks.shape or values.shape != (len(batch),):
                raise ValueError("Evaluator output must have shapes (B,action_size) and (B,)")
            for index, (node, _, mask) in enumerate(batch):
                node.expand(mask, probs[index], float(values[index]))

    def _mover(self, work: GameLike) -> int:
        """The single player to move at a nonterminal node; simultaneous decisions are unsupported."""
        actors = self.spec.actors(work)
        if len(actors) != 1:
            raise ValueError("MCTS requires exactly one player to move per decision; "
                             f"game {self.spec.game_id} reports actors {list(actors)}")
        return int(actors[0])

    def _set_terminal(self, node: _Node, work: GameLike, rewards=None) -> None:
        """Validate rule rewards before applying the optional symmetric draw utility."""
        values = list(self.spec.terminal_values(work) if rewards is None else rewards)
        if len(values) != 2 or abs(sum(values)) > 1e-6:
            raise ValueError("MCTS supports zero-sum two-player games only; terminal values "
                             f"{values} of game {self.spec.game_id} do not sum to zero")
        node.player = int(work.position.turn)
        reasons = draw_reasons_for(self.spec, self.config.draw_reasons)
        # Adapters without a draw-reason contract keep their original reward-only interface.
        draw = (draw_utility(work.outcome.winner, work.outcome.reason, self.config.draw_value, reasons)
                if reasons else None)
        node.symmetric = draw is not None
        node.value = float(values[node.player]) if draw is None else draw

    def _evaluation_inputs(self, work: GameLike, player: int) -> tuple[np.ndarray, np.ndarray]:
        # Inputs are queued across trees before one batched evaluation, and encoders may return a
        # reused buffer, so every queued observation and mask is an independent copy.
        planes = np.array(self._encode(work, player), copy=True)
        mask = np.array(self.spec.legal_mask(work, player), copy=True)
        return planes, mask

    def _leaf(self, work: GameLike, root: _Node, halving=None) -> tuple[_Node, list, tuple | None]:
        """Descend to a leaf through spec.step(); ``work`` is left unchanged on return."""
        node = root
        path = []
        sim = work if self.undoable else work.clone()
        try:
            rewards = None
            while node.actions is not None and not sim.outcome.terminal:
                edge = (halving.select(node) if node is root and halving is not None else
                        node.select_improved(self.config) if self.config.root_selection == "gumbel" else
                        node.select(self.config.c_puct))
                outcome = self.spec.step(sim, {node.player: int(node.actions[edge])})
                rewards = outcome.rewards if outcome.terminal else None
                path.append((node, edge))
                if edge not in node.children:
                    node.children[edge] = _Node()
                parent, node = node, node.children[edge]
                if outcome.truncated and not outcome.terminal:
                    # The adapter stopped the game here: value it like a capped self-play game and never
                    # evaluate, expand, or step beyond it (re-visits take the same edge and stop again).
                    node.stopped = True
                    node.player = parent.player
                    node.value = self.config.truncation_value
                    return node, path, None
            if sim.outcome.terminal:
                self._set_terminal(node, sim, rewards)
                return node, path, None
            node.player = self._mover(sim)
            planes, mask = self._evaluation_inputs(sim, node.player)
            return node, path, (node, planes, mask)
        finally:
            if self.undoable:
                for _ in path:
                    sim.undo()
            else:
                sim.close()

    @staticmethod
    def _backup(node: _Node, path: list) -> None:
        # Values are backed up by player identity (zero-sum: the other side sees the negation), so
        # a game whose turn order is not strictly alternating (extra turns, forced replies) is
        # credited to the right side instead of assuming one negation per ply.
        for parent, edge in reversed(path):
            value = node.value if node.stopped or node.symmetric or parent.player == node.player else -node.value
            parent.visits[edge] += 1
            parent.value_sum[edge] += value
            parent.total += 1

    def run(self, game: GameLike, simulations: int | None = None,
            cancel: threading.Event | None = None, deadline: float | None = None) -> MctsResult:
        return self.run_many([game], simulations, cancel, deadline)[0]

    def run_many(self, games: list[GameLike], simulations: int | None = None,
                 cancel: threading.Event | None = None,
                 deadline: float | None = None) -> list[MctsResult]:
        """Search independent trees, evaluating one leaf per game per step.

        Deadlines use perf_counter. Root evaluation is always completed to
        provide a legal prior, even with an expired deadline or cancellation.
        Once initialized, budgets are checked between simulation paths.
        """
        budget = _integer_budget(self.config.simulations if simulations is None else simulations,
                                 "simulations")
        roots = [_Node() for _ in games]
        with ExitStack() as stack:
            works = []
            leaves = []
            active = []
            for index, game in enumerate(games):
                work = game.clone()
                stack.callback(work.close)
                works.append(work)
                if work.outcome.terminal:
                    self._set_terminal(roots[index], work)
                else:
                    roots[index].player = self._mover(work)
                    active.append(index)
                    planes, mask = self._evaluation_inputs(work, roots[index].player)
                    leaves.append((roots[index], planes, mask))
            self._expand_many(leaves)
            halving = {}
            if self.config.root_selection == "gumbel":
                for index in active:
                    root = roots[index]
                    # One draw per nonterminal tree, after root evaluation, in
                    # input-tree order and ascending legal-action order.
                    noise = (self.rng.gumbel(size=len(root.actions)) if self.config.add_noise
                             else np.zeros(len(root.actions), dtype=np.float64))
                    halving[index] = _SequentialHalving(root, self.config, noise, budget)
            elif self.config.add_noise:
                for index in active:
                    root = roots[index]
                    noise = self.rng.dirichlet(np.full(len(root.actions), self.config.dirichlet_alpha))
                    epsilon = self.config.dirichlet_epsilon
                    root.prior = (1 - epsilon) * root.prior + epsilon * noise
            for _ in range(budget if active else 0):
                pending = []
                leaves = []
                for index in active:
                    if self._stopped(cancel, deadline):
                        break
                    node, path, leaf = self._leaf(works[index], roots[index], halving.get(index))
                    pending.append((node, path))
                    if leaf is not None:
                        leaves.append(leaf)
                if not pending:
                    break
                self._expand_many(leaves)
                for node, path in pending:
                    self._backup(node, path)
            results = []
            for index, root in enumerate(roots):
                visits = np.zeros(self.spec.action_size, dtype=np.float32)
                q_values = np.full(self.spec.action_size, np.nan, dtype=np.float32)
                value = root.value
                if root.actions is not None:
                    visits[root.actions] = root.visits if root.total else root.prior
                    seen = root.visits > 0
                    q_values[root.actions[seen]] = root.value_sum[seen] / root.visits[seen]
                    if root.total:
                        value = float(root.value_sum.sum() / root.total)
                selected = None
                if self.config.root_selection == "gumbel":
                    policy = np.zeros(self.spec.action_size, dtype=np.float64)
                    if index in halving:
                        policy[root.actions] = root.improved_policy(self.config)
                        selected = int(root.actions[halving[index].winner(root)])
                else:
                    policy = visits / visits.sum() if visits.sum() else visits.copy()
                results.append(MctsResult(visits, value, root.total, q_values, policy, selected))
            return results
