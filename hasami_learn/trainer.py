"""Training loop: config -> learner -> iterations -> checkpoints -> evaluation -> resume."""

from __future__ import annotations

from contextlib import contextmanager
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .games import get_spec, default_rules, action_encoding
from hasami.agents import SearchLimits
from .algorithms import IMPORT_ERRORS, LearnerContext
from .config import ConfigError, TrainingConfig
from .devices import configure_threads, device_report, resolve_device, seed_everything
from .evaluation import play_matches
from .manifest import (ModelManifest, atomic_write_json, cleanup_checkpoint_temps,
                       find_resume_checkpoint, fsync_directory, resume_run_dir,
                       load_checkpoint, save_checkpoint, sha256_file, write_latest_pointer)
from . import runlock
from .metrics import MetricsLogger
from .opponents import OpponentPool
from .registry import ALGORITHMS, NETWORKS
from .replay import ReplayBuffer, fields_for, memory_warning
from .selfplay import draw_value_for_iteration, set_draw_value
from . import networks  # noqa: F401  (register factories)


def resolve_backend(spec: str) -> str:
    return get_spec().resolve_backend(spec)


class Trainer:
    def __init__(self, config: TrainingConfig, resume: str | Path | None = None, echo: bool = True) -> None:
        self.run_dir = runlock.writer_path(Path(config.run.output_dir) / config.run.run_id)
        if resume is not None:
            resume = runlock.writer_path(Path(resume).expanduser())
        with runlock.run_lock(self.run_dir) as self.run_dir:
            skipped = cleanup_checkpoint_temps(self.run_dir)
            self._initialize(config, resume, echo, skipped)
            self._revision = self._disk_revision()
            self._writing_depth = 0

    def _initialize(self, config, resume, echo, skipped):
        self.config = config.resolve_defaults()  # game-derived caps follow the selected game
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = resolve_device(config.device.device)
        configure_threads(config.device.threads)
        self.rng = seed_everything(config.run.seed, config.device.deterministic)
        self.spec = get_spec(config.game.name, config.game.params)
        self.backend = self.spec.resolve_backend(config.run.backend)
        self.encoder = self.spec.make_encoder(config.encoder.name, **config.encoder.params)
        if config.algorithm.name not in ALGORITHMS:
            hint = IMPORT_ERRORS.get(config.algorithm.name)
            raise ConfigError(f"algorithm '{config.algorithm.name}' is not registered" + (f" (import error: {hint})" if hint else f"; known: {ALGORITHMS.names()}"))
        learner_cls = ALGORITHMS.get(config.algorithm.name)
        self.network = NETWORKS.get(config.network.name)(head=learner_cls.required_head, num_planes=self.encoder.num_planes, board_shape=self.spec.board_shape, action_size=self.spec.action_size, **config.network.params).to(self.device)
        self.init_checkpoint: dict[str, Any] | None = None
        if resume is None and config.run.init_checkpoint is not None:
            self._warm_start(learner_cls.required_head)
        self.replay = ReplayBuffer(config.replay.capacity, fields_for(self.spec, self.encoder.num_planes, learner_cls.replay_kind), self.rng) if learner_cls.replay_kind else None
        self.metrics = MetricsLogger(runlock.writer_path(self.run_dir / "metrics.jsonl"), config.run.run_id, str(self.device), echo=echo)
        self.opponents = OpponentPool(config.opponents, self.device, self.encoder, self.rng, self.run_dir, seed=config.run.seed, spec=self.spec,
                                      game_params=dict(config.game.params))
        ctx = LearnerContext(config, self.network, self.encoder, self.device, self.rng, self.replay, self.opponents, self.metrics, self.backend, self.run_dir, self.spec)
        try:
            self.learner = learner_cls(ctx)
        except BaseException:
            self.metrics.close()
            raise
        self.opponents.set_current(lambda: self.learner.make_agent("selfplay"))
        self.iteration = 0
        self.elapsed = 0.0
        self.best_eval: float | None = None
        self.last_checkpoint: Path | None = None
        self._stop_reason: str | None = None
        try:
            if resume is not None:
                self._resume(Path(resume), skipped)
            else:
                if skipped:
                    self.metrics.log("resume", 0, extra={"skipped": skipped, "fresh_start": True})
            atomic_write_json(self.run_dir / "config.json", config.to_dict())
        except BaseException:
            self.metrics.close()
            raise
        extra = {"device_report": device_report(), "backend": self.backend, "algorithm": self.learner.name,
                 "network": config.network.to_dict() if hasattr(config.network, "to_dict") else config.network.__dict__,
                 "parameters": sum(p.numel() for p in self.network.parameters())}
        if self.init_checkpoint is not None:
            extra["init_checkpoint"] = dict(self.init_checkpoint)
        if self.replay is not None:
            extra.update(replay_capacity=self.replay.capacity,
                         replay_sample_bytes=self.replay.sample_nbytes,
                         replay_capacity_bytes=self.replay.capacity_nbytes)
            warning = memory_warning(self.replay.capacity_nbytes)
            if warning is not None:
                print(warning, file=sys.stderr, flush=True)
        self.metrics.log("info", self.iteration, extra=extra)

    def _disk_revision(self):
        revision = []
        for name in ("config.json", "trainer_state.json", "checkpoints", "pool"):
            path = self.run_dir / name
            stat = path.stat() if path.exists() else None
            revision.append(None if stat is None else (stat.st_ino, stat.st_size, stat.st_mtime_ns))
        return tuple(revision)

    @contextmanager
    def _writing(self):
        # Initialization and run() are separate public operations. Refuse to
        # publish stale in-memory state if another writer used that gap.
        with runlock.run_lock(self.run_dir) as self.run_dir:
            if self._writing_depth == 0 and self._disk_revision() != self._revision:
                raise RuntimeError("run changed since this trainer loaded it; create a new trainer to resume")
            self._writing_depth += 1
            try:
                yield
            finally:
                self._writing_depth -= 1
                self._revision = self._disk_revision()

    # ------------------------------------------------------------ checkpoints
    def manifest(self, iteration: int, metrics: dict[str, Any] | None = None) -> ModelManifest:
        cfg = self.config
        return ModelManifest(
            game_id=self.spec.game_id, num_players=self.spec.num_players, board_shape=self.spec.board_shape,
            game_params=dict(cfg.game.params), ruleset_id=self.spec.ruleset_id, rules=default_rules(self.spec),
            action_size=self.spec.action_size, action_encoding=action_encoding(self.spec),
            algorithm=self.learner.name,
            network={"name": cfg.network.name, "params": dict(cfg.network.params), "head": self.learner.required_head},
            encoding_id=self.encoder.encoding_id,
            encoder_params=dict(cfg.encoder.params),
            num_planes=self.encoder.num_planes,
            algorithm_params={**cfg.algorithm.params,
                              **({"search_semantics_version": cfg.search_semantics_version}
                                 if self.learner.name == "alphazero" else {})},
            run_id=cfg.run.run_id,
            iteration=int(iteration),
            train_device=str(self.device),
            inference_hint=self.learner.inference_hint(),
            metrics=metrics or {},
            notes=cfg.run.notes,
            init_checkpoint=self.init_checkpoint,
        )

    def _warm_start(self, head: str) -> None:
        """Initialize weights before constructing learners with derived target networks."""
        cfg = self.config
        expected = {
            "game_id": self.spec.game_id, "game_params": dict(cfg.game.params),
            "num_players": self.spec.num_players, "ruleset_id": self.spec.ruleset_id,
            "rules": default_rules(self.spec), "board_shape": list(self.spec.board_shape),
            "action_size": self.spec.action_size, "action_encoding": action_encoding(self.spec),
            "encoding_id": self.encoder.encoding_id, "encoder_params": dict(cfg.encoder.params),
            "value_perspective": "side_to_move",
            "num_planes": self.encoder.num_planes, "network.name": cfg.network.name,
            "network.params": dict(cfg.network.params), "network.head": head,
        }
        # Building the donor network must not advance this fresh run's RNG.
        # No optimizer, replay, opponent, iteration or RNG state is even read.
        with torch.random.fork_rng(devices=[]):
            loaded = load_checkpoint(cfg.run.init_checkpoint, "cpu", expect_compatibility=expected,
                                     load_training_state=False)
        self.network.load_state_dict(loaded.network.state_dict(), strict=True)
        self.init_checkpoint = {"path": cfg.run.init_checkpoint, "iteration": loaded.manifest.iteration,
                                "algorithm": loaded.manifest.algorithm,
                                "weights_sha256": loaded.manifest.weights_sha256}

    def trainer_state(self) -> dict[str, Any]:
        return {
            "iteration": int(self.iteration),
            "elapsed_seconds": round(self.elapsed, 3),
            "best_eval": self.best_eval,
            "rng_state": self.rng.bit_generator.state,
            "torch_rng": torch.get_rng_state().tolist(),
            "backend": self.backend,
        }

    def checkpoint(self, iteration: int, metrics: dict[str, Any] | None = None) -> Path:
        with self._writing():
            return self._checkpoint(iteration, metrics)

    def _checkpoint(self, iteration: int, metrics: dict[str, Any] | None = None) -> Path:
        directory = runlock.writer_path(self.run_dir / "checkpoints" / f"iter_{iteration:06d}")
        learner_state = dict(self.learner.state_dict())
        learner_state["opponents"] = self.opponents.state_dict()
        state = self.trainer_state()

        def prepare(temporary):
            self.learner.on_checkpoint(temporary)
            if self.replay is not None and self.config.replay.persist:
                # This snapshot commits with weights, optimizer and RNG. The
                # run-root replay remains a compatibility copy, never authority.
                replay = temporary / "replay_snapshot.npz"
                self.replay.save(replay)
                state["replay_snapshot_sha256"] = sha256_file(replay)
                atomic_write_json(temporary / "trainer_state.json", state)

        save_checkpoint(directory, self.network, self.manifest(iteration, metrics), learner_state, state,
                        on_checkpoint=prepare)
        runlock.checkpoint_boundary("after_checkpoint_publish", iteration)
        if self.replay is not None and self.config.replay.persist:
            replay_target = runlock.writer_path(self.run_dir / "replay.npz")
            fd, name = tempfile.mkstemp(prefix=".tmp_replay_", suffix=".npz", dir=self.run_dir)
            os.close(fd)
            temporary = Path(name)
            try:
                shutil.copyfile(directory / "replay_snapshot.npz", temporary)
                runlock.checkpoint_boundary("during_replay_write", iteration)
                with temporary.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary, replay_target)
                fsync_directory(self.run_dir)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            # Non-replay learners cross the same sequence boundary without I/O.
            runlock.checkpoint_boundary("during_replay_write", iteration)
        write_latest_pointer(self.run_dir, directory, iteration)
        atomic_write_json(self.run_dir / "trainer_state.json", state, boundary="during_trainer_state_write")
        self._prune_checkpoints(iteration)
        self.last_checkpoint = directory
        runlock.checkpoint_boundary("before_metrics_append", iteration)
        self.metrics.log("checkpoint", iteration, checkpoint=str(directory))
        return directory

    def _prune_checkpoints(self, iteration: int) -> None:
        keep = max(1, self.config.run.keep_checkpoints)
        # Future generations may be corrupt after fallback; they must not evict
        # the valid checkpoint that was just published.
        dirs = sorted(directory for directory in runlock.writer_path(self.run_dir / "checkpoints").glob("iter_*")
                      if runlock.plain_directory(directory) and directory.name[5:].isdigit()
                      and int(directory.name[5:]) <= iteration)
        for old in dirs[:-keep]:
            shutil.rmtree(old, ignore_errors=True)

    def _resume(self, path: Path, skipped=None) -> None:
        with runlock.run_lock(resume_run_dir(path)):
            self._resume_locked(path, skipped)

    def _resume_locked(self, path: Path, skipped) -> None:
        loaded, recovered = find_resume_checkpoint(path)
        skipped = list(skipped or []) + recovered
        target = loaded.directory
        lineage = loaded.manifest.init_checkpoint
        requested = self.config.run.init_checkpoint
        if requested is not None and requested != (lineage or {}).get("path"):
            raise ConfigError("run.init_checkpoint differs from the resumed checkpoint lineage")
        self.init_checkpoint = dict(lineage) if lineage is not None else None
        self.config.run.init_checkpoint = (lineage or {}).get("path")
        if loaded.manifest.game_id != self.spec.game_id or loaded.manifest.game_params != self.config.game.params:
            raise ConfigError("checkpoint game does not match configured game")
        if loaded.manifest.algorithm != self.learner.name:
            raise ConfigError(f"checkpoint algorithm {loaded.manifest.algorithm} != configured {self.learner.name}")
        if self.learner.name == "alphazero":
            from .search.draw import SEARCH_SEMANTICS_VERSION

            if (loaded.manifest.algorithm_params.get("search_semantics_version") != SEARCH_SEMANTICS_VERSION
                    or not loaded.learner_state):
                raise ConfigError("Cannot resume AlphaZero checkpoint with old search semantics: "
                                  "version 2 applies symmetric draw contempt inside search. Start a new run "
                                  "(optionally run.init_checkpoint for weights only); old replay is incompatible.")
        self.network.load_state_dict(loaded.network.state_dict(), strict=True)
        self.network.to(self.device)
        if loaded.learner_state:
            learner_state = dict(loaded.learner_state)
            pool_state = learner_state.pop("opponents", None)
            if pool_state:
                self.opponents.load_state_dict(pool_state)
            self.learner.load_state_dict(learner_state)
        if loaded.trainer_state:
            self.iteration = int(loaded.trainer_state.get("iteration", 0))
            self.elapsed = float(loaded.trainer_state.get("elapsed_seconds", 0.0))
            self.best_eval = loaded.trainer_state.get("best_eval")
            rng_state = loaded.trainer_state.get("rng_state")
            if rng_state:
                self.rng.bit_generator.state = rng_state
            if "torch_rng" in loaded.trainer_state:
                torch.set_rng_state(torch.tensor(loaded.trainer_state["torch_rng"], dtype=torch.uint8))
        if "replay_snapshot_sha256" in (loaded.trainer_state or {}):
            replay_file = target / "replay_snapshot.npz"
        else:
            replay_file = resume_run_dir(path) / "replay.npz"
        if not replay_file.is_file() and (target / "replay.npz").is_file():
            # Runs written before the run-root layout kept replay.npz inside each checkpoint directory.
            replay_file = target / "replay.npz"
        if self.replay is not None and self.config.replay.persist and replay_file.is_file():
            self.replay.load(replay_file)
        self.last_checkpoint = target
        if target.is_relative_to(self.run_dir):
            write_latest_pointer(self.run_dir, target, self.iteration)
        atomic_write_json(self.run_dir / "trainer_state.json", self.trainer_state())
        self.metrics.log("resume", self.iteration, checkpoint=str(target), replay_size=len(self.replay) if self.replay else 0,
                         extra={"skipped": skipped})

    # ------------------------------------------------------------ evaluation
    def evaluate(self, iteration: int) -> dict[str, Any]:
        cfg = self.config.evaluation
        if cfg.games <= 0 or not cfg.opponents:
            return {}
        agent = self.learner.make_agent("eval")
        limits = SearchLimits(**{**{"max_depth": 1, "time_ms": 200, "node_limit": 20000}, **cfg.limits})
        if cfg.simulations:
            from .inference import MctsAgent

            if isinstance(agent, MctsAgent):
                agent.simulations = cfg.simulations
        results: dict[str, Any] = {}
        for spec in cfg.opponents:
            if spec.startswith("pool:"):
                entry = self.opponents.find(spec) if spec != "pool:latest" else (self.opponents.entries[-1] if self.opponents.entries else None)
                if entry is None:
                    continue
                opponent, opp_limits = self.opponents.agent_for(entry), limits
            else:
                opponent, opp_limits = self.spec.baseline_agent(spec, seed=self.config.run.seed + iteration)
            records_dir = (self.run_dir / "games" / f"iter_{iteration:06d}_{spec.replace(':', '_')}") if cfg.save_records else None
            if records_dir is not None:
                records_dir = runlock.writer_path(records_dir)
            match = play_matches(agent, opponent, cfg.games, backend=self.backend, max_plies=cfg.max_plies, limits=limits,
                                 opponent_limits=opp_limits or limits, swap_sides=cfg.swap_sides, records_dir=records_dir, spec=self.spec, seed=self.config.run.seed + iteration * cfg.games,
                                 opening_plies=cfg.opening_plies)
            results[spec] = match.to_dict()
            self.metrics.log("evaluation", iteration, opponent=spec, games=match.games, wins=[match.wins, match.losses, match.draws],
                             truncated=match.truncated, plies=match.plies, seconds=match.seconds, win_rate=match.win_rate,
                             elo_delta=results[spec]["elo_delta"],
                             extra={**results[spec].get("extra", {}),
                                    "ci_low": results[spec]["ci_low"], "ci_high": results[spec]["ci_high"],
                                    "placements": dict(match.placements), "mean_points": match.mean_points,
                                    "num_players": self.spec.num_players})
        if results:
            mean_rate = float(np.mean([r["win_rate"] for r in results.values()]))
            if self.best_eval is None or mean_rate > self.best_eval:
                self.best_eval = mean_rate
        return results

    # ------------------------------------------------------------ main loop
    def _request_stop(self, signum: int, _frame: Any) -> None:
        self._stop_reason = signal.Signals(signum).name

    def run(self) -> Path:
        with self._writing():
            return self._run_locked()

    def _run_locked(self) -> Path:
        previous = {}
        self._stop_reason = None
        try:
            if threading.current_thread() is threading.main_thread():
                for signum in (signal.SIGTERM, signal.SIGINT):
                    previous[signum] = signal.signal(signum, self._request_stop)
            return self._run_iterations()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)

    def _run_iterations(self) -> Path:
        cfg = self.config
        total = cfg.run.iterations
        start_iteration = self.iteration + 1
        last_eval: dict[str, Any] = {}
        last_checkpoint_time = time.monotonic()
        for iteration in range(start_iteration, total + 1):
            began = time.perf_counter()
            draw_value = draw_value_for_iteration(cfg.selfplay, iteration)
            set_draw_value(draw_value, cfg.selfplay.draw_reasons)
            collect = self.learner.collect(iteration)
            self.metrics.log("collect", iteration, games=collect.games, plies=collect.plies, wins=list(collect.wins), truncated=collect.truncated,
                             seconds=collect.seconds, replay_size=len(self.replay) if self.replay else 0,
                             **({"terminal_reasons": collect.extra["terminal_reasons"]}
                                if "terminal_reasons" in collect.extra else {}),
                             extra={**{k: v for k, v in collect.extra.items() if k != "terminal_reasons"},
                                    "draw_value": draw_value})
            update = self.learner.update(iteration)
            self.metrics.log("update", iteration, steps=update.steps, loss=update.loss, policy_loss=update.policy_loss, value_loss=update.value_loss,
                             seconds=update.seconds, replay_size=len(self.replay) if self.replay else 0, extra=update.extra)
            if not self._stop_reason and cfg.opponents.snapshot_every and iteration % cfg.opponents.snapshot_every == 0:
                entry = self.opponents.add_snapshot(self.network, self.manifest(iteration), iteration)
                self.metrics.log("opponent_pool", iteration, opponent=entry.entry_id, extra={"pool": self.opponents.stats()})
            if not self._stop_reason and cfg.run.eval_every and iteration % cfg.run.eval_every == 0:
                last_eval = self.evaluate(iteration)
            self.iteration = iteration
            self.elapsed += time.perf_counter() - began
            seconds = cfg.run.checkpoint_every_seconds
            timed_checkpoint = seconds > 0 and time.monotonic() - last_checkpoint_time >= seconds
            if (iteration % max(1, cfg.run.checkpoint_every) == 0 or iteration == total
                    or timed_checkpoint or self._stop_reason):
                self.checkpoint(iteration, {"eval": last_eval, "last_loss": update.loss})
                last_checkpoint_time = time.monotonic()
            if self._stop_reason:
                if self.last_checkpoint is None or self.last_checkpoint.name != f"iter_{iteration:06d}":
                    self.checkpoint(iteration, {"eval": last_eval, "last_loss": update.loss})
                self.metrics.log("info", iteration, extra={"stop_reason": self._stop_reason})
                break
        if self.last_checkpoint is None:
            self.checkpoint(self.iteration, {"eval": last_eval})
        return self.last_checkpoint

    def close(self) -> None:
        self.metrics.close()


def train_from_config(config: TrainingConfig, resume: str | Path | None = None, echo: bool = True) -> Path:
    trainer = Trainer(config, resume=resume, echo=echo)
    try:
        return trainer.run()
    finally:
        trainer.close()
