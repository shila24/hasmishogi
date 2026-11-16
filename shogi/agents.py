"""Classical legal baselines using the common Agent decision/limit protocol."""
import random
import time

from hasami.agents import Decision, SearchLimits, check_cancel

from .mechanics import MOVEMENT
from .model import K, UNPROMOTED, DECLARE, decode_move

MATERIAL = (0, 100, 300, 320, 450, 550, 800, 1000, 0)
PROMOTION_BONUS = 200
WIN = 1_000_000


def _piece_value(piece):
    return MATERIAL[UNPROMOTED.get(piece, piece)] + (PROMOTION_BONUS if piece in UNPROMOTED else 0)


def material_gain(game, code):
    if code == DECLARE:
        return WIN
    move = decode_move(code)
    captured = game.position.board[move.to_sq] & 15
    gain = _piece_value(captured) + MATERIAL[UNPROMOTED.get(captured, captured)] if captured else 0
    return gain + (PROMOTION_BONUS if move.promote else 0)


def evaluate(game):
    """Return material, promotion and adjacent king defenders from the mover's view."""
    p = game.position
    scores = [0, 0]
    for cell in p.board:
        if cell:
            scores[cell >> 4] += _piece_value(cell & 15)
    for side in (0, 1):
        scores[side] += sum(count * MATERIAL[hand + 1] for hand, count in enumerate(p.hands[side]))
        king = p.board.index(K + 16 * side)
        steps, _ = MOVEMENT[side][K][king]
        scores[side] += 8 * sum(bool(p.board[sq]) and p.board[sq] >> 4 == side for sq in steps)
    return scores[p.turn] - scores[1 - p.turn]


class RandomAgent:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def choose(self, game, limits: SearchLimits, cancel=None) -> Decision:
        check_cancel(cancel)
        actions = game.legal_actions()
        if not actions:
            raise ValueError("No legal action is available")
        return Decision(self.rng.choice(actions))


class GreedyAgent:
    """Maximize immediate material gain; resolve ties by the lowest move code."""
    def choose(self, game, limits: SearchLimits, cancel=None) -> Decision:
        check_cancel(cancel)
        actions = game.legal_actions()
        if not actions:
            raise ValueError("No legal action is available")
        ranked = []
        for code in actions:
            check_cancel(cancel)
            ranked.append((-material_gain(game, code), code))
        score, action = min(ranked)
        return Decision(action, -score, depth=1, nodes=len(actions))


class _BudgetExpired(Exception):
    pass


class _Search:
    def __init__(self, limits, cancel, start):
        self.limits, self.cancel, self.start = limits, cancel, start
        self.nodes = 0

    def elapsed(self):
        return (time.monotonic() - self.start) * 1000

    def check(self):
        check_cancel(self.cancel)
        if self.nodes >= self.limits.node_limit or self.elapsed() >= self.limits.time_ms:
            raise _BudgetExpired

    def search(self, game, depth, alpha, beta, ply):
        self.check()
        self.nodes += 1
        if game.outcome.terminal:
            winner = game.outcome.winner
            return (0 if winner == -1 else WIN - ply if winner == game.position.turn else -WIN + ply), -1
        if depth == 0:
            return evaluate(game), -1
        actions = sorted(game.legal_actions(), key=lambda code: (-material_gain(game, code), code))
        best, action = -2 * WIN, actions[0]
        actor = game.position.turn
        for code in actions:
            self.check()
            game.play(code)
            try:
                if game.position.turn == actor:
                    score, _ = self.search(game, depth - 1, alpha, beta, ply + 1)
                else:
                    score, _ = self.search(game, depth - 1, -beta, -alpha, ply + 1)
                    score = -score
            finally:
                game.undo()
            if score > best:
                best, action = score, code
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best, action


class AlphaBetaAgent:
    """Deterministic iterative deepening with full history and no score cache.

    Material values are P=100, L=300, N=320, S=450, G=550, B=800,
    R=1000, K=0; every promoted piece gains 200. Each friendly piece
    adjacent to its king adds 8 for king safety. Terminal wins score
    1,000,000 minus distance. No evaluation term was learned or tuned.
    Time is a cooperative budget checked between nodes and moves; an
    unfinished iteration keeps the last completed result (or a legal fallback).
    """
    def choose(self, game, limits: SearchLimits, cancel=None) -> Decision:
        start = time.monotonic()
        check_cancel(cancel)
        actions = game.legal_actions()
        if not actions:
            raise ValueError("No legal action is available")
        search = _Search(limits, cancel, start)
        action, score, completed_depth = actions[0], evaluate(game), 0
        work = game.clone()
        try:
            for depth in range(1, limits.max_depth + 1):
                try:
                    candidate_score, candidate = search.search(work, depth, -2 * WIN, 2 * WIN, 0)
                except _BudgetExpired:
                    break
                action, score, completed_depth = candidate, candidate_score, depth
        finally:
            work.close()
        check_cancel(cancel)
        return Decision(action, score, completed_depth, search.nodes, search.elapsed(),
                        completed_depth == limits.max_depth)


def native_baseline(spec):
    """Resolve a native baseline and its explicit depth or node budget."""
    from .native_search import AlphaBetaCppAgent

    name, separator, budget = spec.partition(":")
    if name != "alphabeta-cpp" or not separator or not budget:
        raise ValueError("Use alphabeta-cpp:<depth> or alphabeta-cpp:nodes=<n>")
    budget, separator, extra = budget.partition(",nnue=")
    if separator and not extra:
        raise ValueError("nnue requires a weights path")
    if budget.startswith("nodes="):
        nodes = int(budget[6:])
        limits = SearchLimits(max_depth=64, time_ms=3600000, node_limit=nodes)
    else:
        limits = SearchLimits(max_depth=int(budget), time_ms=3600000, node_limit=2**64-1)
    return AlphaBetaCppAgent(**({"nnue": extra} if separator else {})), limits
