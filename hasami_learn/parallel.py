"""Spawned, deterministic MCTS self-play with numpy-only trajectory results."""

from __future__ import annotations

import io
import inspect
import multiprocessing
from contextlib import ExitStack
from dataclasses import replace
from typing import Callable, Iterator

import numpy as np

from .games import get_spec
from .games.base import as_v2, encode_player, terminal_rewards
from .devices import configure_threads, resolve_device, seed_everything
from .selfplay import Trajectory, set_draw_value, search_decision
from .search.draw import draw_reasons_for


def selfplay_batches(search, encoder, decision: Callable, *, games: int,
                     batch_games: int, backend: str, max_plies: int,
                     draw_value: float, draw_reasons: list[str], spec=None, seed: int | None = None) -> Iterator[list[Trajectory]]:
    """Use the same environment lifecycle and search batches in either process path."""
    spec = as_v2(spec or get_spec())
    from .search.mcts import MctsConfig

    # The collection's explicit settings are authoritative in both local and spawned jobs.
    # Other search families have their own configuration and retain their semantics.
    reasons = draw_reasons_for(spec, draw_reasons)
    if isinstance(search.config, MctsConfig):
        search.config = replace(search.config, draw_value=draw_value, draw_reasons=reasons)
    rng = np.random.default_rng(seed)
    for begin in range(0, games, batch_games):
        count = min(batch_games, games - begin)
        with ExitStack() as stack:
            trajectories = [Trajectory(game_id=spec.game_id, num_players=spec.num_players,
                                       rewards=[0.0] * spec.num_players, seed=int(rng.integers(2**63)),
                                       draw_value=draw_value, draw_reasons=reasons)
                            for _ in range(count)]
            states = []
            for trajectory in trajectories:
                # Register cleanup right after each creation so a failure mid-batch closes the games
                # created so far instead of leaking adapter-owned resources.
                game = spec.create_game(backend, seed=trajectory.seed)
                stack.callback(game.close)
                states.append(game)
            active = []
            for i, game in enumerate(states):
                if game.outcome.terminal:
                    trajectories[i].rewards = terminal_rewards(spec, game)
                    trajectories[i].winner = game.outcome.winner
                    trajectories[i].reason = game.outcome.reason
                else:
                    active.append(i)
            while active:
                decisions = [(i, p) for i in active for p in spec.actors(states[i])]
                snapshots = []
                with ExitStack() as clones:
                    for i, p in decisions:
                        snapshot = states[i].clone()
                        clones.callback(snapshot.close)
                        snapshots.append(snapshot)
                    kwargs = {"simulations": search.config.simulations}
                    if "players" in inspect.signature(search.run_many).parameters:
                        results = search.run_many(snapshots, players=[p for _, p in decisions], **kwargs) if decisions else []
                    else:
                        results = search.run_many(snapshots, **kwargs) if decisions else []
                if len(results) != len(decisions):
                    raise ValueError("Search must return one result per actor")
                joint = {i: {} for i in active}
                for (i, player), result in zip(decisions, results):
                    game, trajectory = states[i], trajectories[i]
                    # Copy what is retained: encoders may hand out a reused buffer, and the batch keeps
                    # one observation per (game, player) until the trajectory is ingested.
                    planes = np.array(encode_player(encoder, game, player), copy=True)
                    mask = np.array(spec.legal_mask(game, player), copy=True)
                    action, policy, extra = decision(result, mask.copy(), trajectory.plies)
                    if type(action) is not int or action not in spec.legal_actions(game, player):
                        raise ValueError("Search returned an illegal action")
                    trajectory.append(planes, mask, action, player, policy, dict(extra, decision_ply=trajectory.plies))
                    joint[i][player] = action
                next_active = []
                for i in active:
                    game, trajectory = states[i], trajectories[i]
                    step = spec.step(game, joint[i])
                    trajectory.joint_actions.append(joint[i])
                    trajectory.step_info.append(dict(step.info))
                    trajectory.plies += 1
                    trajectory.truncated = bool(step.truncated or (not step.terminal and max_plies and trajectory.plies >= max_plies))
                    if step.terminal or trajectory.truncated:
                        trajectory.rewards = terminal_rewards(spec, game) if step.terminal else [0.0] * spec.num_players
                        trajectory.winner = game.outcome.winner
                        # Mirror the serial runner: a truncation reported by the adapter keeps its own reason;
                        # only the runner's ply cap is labelled max_plies.
                        trajectory.reason = ("max_plies" if trajectory.truncated and not step.truncated
                                             else step.info.get("reason", game.outcome.reason))
                    else:
                        next_active.append(i)
                active = next_active
        yield trajectories


def _run_worker(job: dict) -> list[Trajectory]:
    """Build one network for this job; no tensors or engine handles leave the worker."""
    from .search.mcts import MCTS, MctsConfig, batched_evaluator
    from . import networks  # noqa: F401
    from .registry import NETWORKS
    import torch

    configure_threads(job.get("threads") or 1)
    rng = seed_everything(job["seed"], job.get("deterministic", False))
    device = resolve_device(job["device_spec"])
    spec = get_spec(job.get("game_id", "hasami"), job.get("game_params"))
    if job.get("num_players", 2) != spec.num_players:
        raise ValueError("Worker player count does not match game spec")
    encoder = spec.make_encoder(job["encoder_name"], **job.get("encoder_params", {}))
    network = NETWORKS.get(job["network_name"])(head=job["head"], num_planes=encoder.num_planes, board_shape=spec.board_shape, action_size=spec.action_size,
                                                 **job["network_params"])
    state = job["state_dict"]
    if isinstance(state, bytes):
        state = torch.load(io.BytesIO(state), map_location="cpu", weights_only=True)
    network.load_state_dict(state)
    network.to(device).eval()
    set_draw_value(job["draw_value"], job["draw_reasons"])
    search_class, search_options = MCTS, {}
    search_backend = job.get("search_backend", "python")
    if search_backend == "cpp":
        from .search.native_mcts import NativeMCTS

        if job["backend"] != "cpp":
            raise ValueError("Native MCTS workers require backend='cpp'")
        search_class = NativeMCTS
        search_options["threads"] = job.get("search_threads")
    elif search_backend != "python":
        raise ValueError("search_backend must be 'python' or 'cpp'")
    search = search_class(batched_evaluator(network, device, job.get("amp", "auto")), encoder,
                          MctsConfig(**job["mcts_params"]), rng, spec=spec, **search_options)

    def decision(result, mask, ply):
        return search_decision(result, mask, ply, rng, root_selection=search.config.root_selection,
                               temperature=job["temperature"], temperature_plies=job["temperature_plies"],
                               value_target_mix=job.get("value_target_mix", 0.0))

    return [trajectory for batch in selfplay_batches(
        search, encoder, decision, games=job["games"], batch_games=job["batch_games"],
        backend=job["backend"], max_plies=job["max_plies"], draw_value=job["draw_value"],
        draw_reasons=job["draw_reasons"], spec=spec, seed=job["seed"]) for trajectory in batch]


class SelfplayWorkers:
    """Own a spawn pool; run jobs in seed order and reap it on any failure.

    Jobs supply one seed and game count per logical worker in ``seeds`` and
    ``games_per_worker``. The latter may also be a scalar count for every worker.
    Use this object as a context manager, or call ``close`` after successful runs.
    """

    def __init__(self, num_workers: int, device_spec, start_method: str = "spawn") -> None:
        from .search.mcts import _integer_budget

        self.num_workers = _integer_budget(num_workers, "num_workers", minimum=1)
        if start_method != "spawn":
            raise ValueError("SelfplayWorkers requires the spawn start method")
        self.device_spec = str(device_spec)
        self._pool = multiprocessing.get_context("spawn").Pool(self.num_workers)

    def run(self, job: dict) -> list[Trajectory]:
        from .search.mcts import _integer_budget
        import torch

        if self._pool is None:
            raise RuntimeError("SelfplayWorkers is closed")
        try:
            seeds = job["seeds"]
            counts = job["games_per_worker"]
            if isinstance(counts, int):
                counts = [counts] * self.num_workers
            if len(seeds) != self.num_workers or len(counts) != self.num_workers:
                raise ValueError("seeds and games_per_worker must match num_workers")
            # Inline CPU weights avoid torch's shared-memory manager and its IPC daemon.
            weights = io.BytesIO()
            torch.save(job["state_dict"], weights)
            payload = dict(job, state_dict=weights.getvalue())
            jobs = [dict(payload, seed=int(seed), games=_integer_budget(count, "games_per_worker"),
                         device_spec=self.device_spec) for seed, count in zip(seeds, counts)]
            # Pool replaces crashed children, but their lost AsyncResult never completes.
            # Retain the original process objects to detect this and propagate the failure.
            processes = list(self._pool._pool)
            pending = [self._pool.apply_async(_run_worker, (item,)) for item in jobs]
            batches = [None] * len(pending)
            remaining = set(range(len(pending)))
            while remaining:
                for i in list(remaining):
                    if pending[i].ready():
                        batches[i] = pending[i].get()
                        remaining.remove(i)
                if remaining:
                    if any(process.exitcode is not None for process in processes):
                        raise RuntimeError("A self-play worker exited before returning its result")
                    pending[min(remaining)].wait(timeout=0.1)
            return [trajectory for batch in batches for trajectory in batch]
        except BaseException:
            self._shutdown(terminate=True)
            raise

    def _shutdown(self, terminate: bool = False) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            if terminate:
                pool.terminate()
            else:
                pool.close()
            pool.join()

    def close(self) -> None:
        self._shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._shutdown(terminate=exc_type is not None)
