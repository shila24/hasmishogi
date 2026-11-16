"""Classical baselines, trusted plugins, and a language-neutral process adapter."""
from __future__ import annotations
from dataclasses import dataclass, field
import importlib
import json
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
from typing import Callable, Protocol
from .model import GameLike, Position, require_int, decode_action
from .engine import geometric_actions, captured_after_move
from .observation import observe


@dataclass(frozen=True)
class SearchLimits:
    max_depth: int = 3
    time_ms: int = 1000
    node_limit: int = 200000

    def __post_init__(self):
        for key in ("max_depth", "time_ms", "node_limit"):
            require_int(getattr(self, key), key)
        if not 1 <= self.max_depth <= 64 or not 1 <= self.time_ms <= 3600000:
            raise ValueError("Invalid depth or time budget")
        if not 1 <= self.node_limit < (1 << 64):
            raise ValueError("Invalid node limit")


@dataclass(frozen=True)
class Decision:
    action: int
    score: int = 0
    depth: int = 0
    nodes: int = 0
    elapsed_ms: float = 0.0
    completed: bool = True
    # Optional, JSON-serialisable diagnostics (e.g. belief outputs for the dashboard); never required.
    extra: dict = field(default_factory=dict)


class Agent(Protocol):
    def choose(self, game: GameLike, limits: SearchLimits,
               cancel: threading.Event | None = None) -> Decision: ...


class AgentCancelled(RuntimeError):
    pass


def check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise AgentCancelled("Agent request was cancelled")


class RandomAgent:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def choose(self, game, limits, cancel=None):
        check_cancel(cancel)
        actions = game.legal_actions()
        if not actions:
            raise ValueError("No legal action is available")
        return Decision(self.rng.choice(actions))


class GreedyAgent:
    def choose(self, game, limits, cancel=None):
        start = time.monotonic()
        actions = game.legal_actions()
        if not actions:
            raise ValueError("No legal action is available")
        best, best_score, tested = actions[0], -2000000, 0
        work = game.clone()
        actor = game.position.turn
        try:
            for action in actions:
                check_cancel(cancel)
                if tested >= limits.node_limit or (time.monotonic() - start) * 1000 >= limits.time_ms:
                    break
                result = work.play(action)
                try:
                    score = result.captured.bit_count() * 1000
                    if result.outcome.terminal:
                        score = (0 if result.outcome.winner == -1 else
                                 1000000 if result.outcome.winner == actor else -1000000)
                finally:
                    work.undo()
                tested += 1
                if score > best_score:
                    best, best_score = action, score
        finally:
            work.close()
        return Decision(best, best_score if tested else 0, 1 if tested == len(actions) else 0,
                        tested, (time.monotonic() - start) * 1000, tested == len(actions))


def evaluate(p: Position) -> int:
    score = 1000 * (p.black.bit_count() - p.white.bit_count())
    black = Position(p.black, p.white, 0)
    white = Position(p.black, p.white, 1)
    score += 3 * (len(geometric_actions(black)) - len(geometric_actions(white)))
    from .model import bits
    for side in (0, 1):
        for sq in bits(p.pieces(side)):
            center = 8 - abs(sq // 9 - 4) - abs(sq % 9 - 4)
            score += center if side == 0 else -center
    return score if p.turn == 0 else -score


class _Stop(Exception):
    pass


class _PythonSearch:
    def __init__(self, limits, cancel):
        self.limits, self.cancel = limits, cancel
        self.start = time.monotonic()
        self.nodes = 0
        # Hints influence ordering only, never value reuse or pruning correctness.
        self.hints: dict[tuple, int] = {}

    def elapsed(self):
        return (time.monotonic() - self.start) * 1000

    def check(self):
        if ((self.cancel is not None and self.cancel.is_set())
                or self.nodes >= self.limits.node_limit or self.elapsed() >= self.limits.time_ms):
            raise _Stop

    def ordered(self, game):
        p = game.position
        hint = self.hints.get(p.key(), -1)
        ranked = []
        for action in game.legal_actions():
            self.check()
            source, target = divmod(action, 81)
            me = (p.pieces(p.turn) ^ (1 << source)) | (1 << target)
            captured = captured_after_move(me, p.pieces(1 - p.turn), target, game.rules.edge_capture)
            priority = captured.bit_count() * 100 + (100000 if action == hint else 0)
            ranked.append((-priority, action))
        return [action for _, action in sorted(ranked)]

    def negamax(self, game, depth, alpha, beta, ply):
        self.check()
        self.nodes += 1
        if game.outcome.terminal:
            if game.outcome.winner == -1:
                return 0, -1
            return (1000000 - ply if game.outcome.winner == game.position.turn else -1000000 + ply), -1
        if depth <= 0 and game.position.pending == -1:
            return evaluate(game.position), -1
        if depth <= 0:
            depth = 1
        best, best_action = -2000000, -1
        for action in self.ordered(game):
            game.play(action)
            try:
                score = -self.negamax(game, depth - 1, -beta, -alpha, ply + 1)[0]
            finally:
                game.undo()
            if score > best:
                best, best_action = score, action
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        if len(self.hints) > 50000:
            self.hints.clear()
        self.hints[game.position.key()] = best_action
        return best, best_action


class AlphaBetaAgent:
    """Use native search for a native game and independent Python search otherwise."""
    def choose(self, game, limits, cancel=None):
        check_cancel(cancel)
        if getattr(game, "backend", None) == "cpp":
            decision = Decision(**game.search(limits.max_depth, limits.time_ms, limits.node_limit))
            check_cancel(cancel)
            return decision
        search = _PythonSearch(limits, cancel)
        actions = game.legal_actions()
        if not actions:
            raise ValueError("No legal action is available")
        action, score, completed_depth = actions[0], evaluate(game.position), 0
        work = game.clone()
        try:
            for depth in range(1, limits.max_depth + 1):
                try:
                    candidate_score, candidate = search.negamax(work, depth, -2000000, 2000000, 0)
                    action, score, completed_depth = candidate, candidate_score, depth
                except _Stop:
                    break
        finally:
            work.close()
        check_cancel(cancel)
        return Decision(action, score, completed_depth, search.nodes, search.elapsed(),
                        completed_depth == limits.max_depth)


class SubprocessAgent:
    """Run one trusted child process per decision using JSON on stdin/stdout."""
    def __init__(self, argv: list[str], cwd: str | None = None, startup_ms: int = 500):
        if not argv or not all(isinstance(x, str) and x for x in argv):
            raise ValueError("Agent argv must be a nonempty list of strings")
        if type(startup_ms) is not int or not 0 <= startup_ms <= 60000:
            raise ValueError("Invalid process startup allowance")
        self.argv, self.cwd, self.startup_ms = argv, cwd, startup_ms

    def choose(self, game, limits, cancel=None):
        check_cancel(cancel)
        start = time.monotonic()
        observation = observe(game)
        if not observation["legal_actions"]:
            raise ValueError("No legal action is available")
        request = {"protocol": "hasami-agent-v1", "observation": observation,
                   "limits": {"max_depth": limits.max_depth, "time_ms": limits.time_ms,
                              "node_limit": limits.node_limit}}
        payload = (json.dumps(request, separators=(",", ":")) + "\n").encode("utf-8")
        deadline = start + (limits.time_ms + self.startup_ms) / 1000
        process = subprocess.Popen(self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, cwd=self.cwd, shell=False)
        first = True
        try:
            while True:
                check_cancel(cancel)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("External agent exceeded its process budget")
                try:
                    stdout, stderr = process.communicate(input=payload if first else None,
                                                         timeout=min(0.05, remaining))
                    break
                except subprocess.TimeoutExpired:
                    first = False
            if process.returncode != 0:
                detail = stderr[:2048].decode("utf-8", "replace")
                raise RuntimeError(f"External agent failed ({process.returncode}): {detail}")
            if len(stdout) > 1048576:
                raise ValueError("External agent response is too large")
            response = json.loads(stdout.decode("utf-8"))
            if not isinstance(response, dict) or response.get("protocol") != "hasami-agent-v1":
                raise ValueError("External agent protocol mismatch")
            action = response.get("action")
            decode_action(action)
            if action not in observation["legal_actions"]:
                raise ValueError("External agent returned an illegal action")
            return Decision(action, elapsed_ms=(time.monotonic() - start) * 1000)
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate()


@dataclass(frozen=True)
class AgentSpec:
    factory: Callable[[], Agent]
    family: str
    description: str


class AgentRegistry:
    def __init__(self, seed=0):
        self._specs: dict[str, AgentSpec] = {}
        self.register("random", lambda: RandomAgent(seed), "classical", "Seeded random legal play")
        self.register("greedy", GreedyAgent, "classical", "Immediate captures")
        self.register("alphabeta", AlphaBetaAgent, "classical", "Iterative-deepening alpha-beta")

    def register(self, name, factory, family="custom", description=""):
        if (not isinstance(name, str) or not name or name == "human" or name in self._specs
                or not callable(factory)):
            raise ValueError("Invalid or duplicate agent registration")
        self._specs[name] = AgentSpec(factory, family, description)

    def names(self):
        return list(self._specs)

    def family(self, name: str) -> str:
        """Return the registered family label; "human" is not an agent."""
        if name == "human":
            return "human"
        try:
            return self._specs[name].family
        except KeyError as exc:
            raise ValueError(f"Unknown agent: {name}") from exc

    def describe(self, name: str) -> str:
        try:
            return self._specs[name].description
        except KeyError as exc:
            raise ValueError(f"Unknown agent: {name}") from exc

    def unregister(self, name: str) -> None:
        """Remove a dynamically registered agent; built-in classical agents stay."""
        if name not in self._specs:
            raise ValueError(f"Unknown agent: {name}")
        if self._specs[name].family == "classical" and name in ("random", "greedy", "alphabeta"):
            raise ValueError("Built-in agents cannot be removed")
        del self._specs[name]

    def create(self, name):
        try:
            agent = self._specs[name].factory()
        except KeyError as exc:
            raise ValueError(f"Unknown agent: {name}") from exc
        if not callable(getattr(agent, "choose", None)):
            raise TypeError("An agent must implement choose(game, limits, cancel)")
        return agent

    def load_config(self, path: str | Path):
        """Load explicitly selected trusted configuration, not downloaded defaults."""
        root = Path(__file__).resolve().parents[1]
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("agents"), list):
            raise ValueError("Expected an agents configuration object")
        for item in data["agents"]:
            if item["type"] == "subprocess":
                argv = [x.replace("{python}", sys.executable).replace("{root}", str(root))
                        for x in item["argv"]]
                factory = lambda argv=argv: SubprocessAgent(argv, str(root))
            elif item["type"] == "python":
                module_name, separator, symbol = item["factory"].partition(":")
                if not separator or not module_name or not symbol:
                    raise ValueError("Python factory must use module:callable syntax")
                function = getattr(importlib.import_module(module_name), symbol)
                options = {key: (value.replace("{root}", str(root)) if isinstance(value, str) else value)
                           for key, value in item.get("options", {}).items()}
                factory = lambda function=function, options=options: function(**options)
            else:
                raise ValueError("Unknown agent adapter type")
            self.register(item["id"], factory, item.get("family", "custom"), item.get("description", ""))
