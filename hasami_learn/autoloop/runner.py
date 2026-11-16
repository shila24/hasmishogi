"""Persistent training plan controller with injectable external effects."""
from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

from .health import Clock, SuspendDetector, free_gb, probe
from .plan import ROOT, git, validation_gate
from .stages import Stages, checkpoint
from .state import LoopError, StateStore, atomic_json, cleanup_json_temps, read_json
from ..runlock import writer_path


class Controller:
    def __init__(self, plan, *, clock=None, health_probe=probe, disk=free_gb,
                 launcher=subprocess.Popen, gate=validation_gate):
        self.plan = plan
        self.clock = clock or Clock()
        self.probe, self.disk, self.launcher, self.gate = health_probe, disk, launcher, gate
        self.directory = writer_path(plan.loop_dir)
        self.control = writer_path(self.directory / "control")
        self.store = StateStore(self.directory, self.event)
        self.stages = Stages(plan)
        self.state = None
        self.child = None
        self.stopping = False
        self.last_probe = None
        self.log = None
        self.started = self.clock.monotonic()
        self.detector = SuspendDetector(self.clock, plan.limits["suspend_gap_s"])

    def event(self, kind, **fields):
        path = writer_path(self.directory / "events.jsonl")
        self.directory.mkdir(parents=True, exist_ok=True)
        row = {"event": kind, "time": self.clock.wall(), **fields}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if self.log:
            self.log.info(json.dumps(row))

    def transition(self, status):
        self.state["status"] = status
        self.state["last_heartbeat"] = self.clock.wall()
        self.store.save(self.state)

    def stop_requested(self):
        marker = self.control / "stop"
        if self.stopping or marker.exists():
            return True
        cap = self.plan.data.get("stop_after_hours")
        if cap is not None and self.clock.monotonic() - self.started >= cap * 3600:
            # Persist before stopping the stage: ExecCondition must also reject
            # a restart if this controller dies while waiting for checkpointing.
            atomic_json(marker, {"reason": "stop_after_hours", "time": self.clock.wall()})
            return True
        return False

    def pause(self, seconds):
        end = self.clock.monotonic() + seconds
        while self.clock.monotonic() < end and not self.stop_requested():
            self.heartbeat()
            self.clock.sleep(max(0, min(self.plan.limits["heartbeat_s"], end - self.clock.monotonic())))

    def heartbeat(self):
        if not self.state:
            return
        index, cycle = self.state["stage_index"], self.state["cycle"]
        if self.state["status"] == "complete":
            # The persisted cursor points past the last cycle for resumption.
            cycle -= 1
            stage = None
        else:
            stage = self.plan.stage(index, cycle) if index < len(self.plan.data["stages"]) else None
        iteration = age = None
        if stage and stage["kind"] == "train":
            try:
                cp, _ = checkpoint(self.stages.run_dir(stage, cycle))
                iteration = read_json(cp / "trainer_state.json")["iteration"]
                age = max(0, self.clock.wall() - (cp / "manifest.json").stat().st_mtime)
            except (OSError, ValueError, KeyError, TypeError):
                pass
        self.state["last_heartbeat"] = self.clock.wall()
        atomic_json(self.directory / "status.json", {"status": self.state["status"], "cycle": cycle,
                    "stage": stage["id"] if stage else None, "run_id": self.plan.run_id(stage, cycle) if stage else None,
                    "latest_iteration": iteration, "last_checkpoint_age": age,
                    "gpu_probe": self.last_probe, "free_gb": self.disk(self.plan.output),
                    "last_heartbeat": self.state["last_heartbeat"]})

    def stop_child(self):
        child = self.child
        if child is None:
            return
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            # The guardian owns the stage group and enforces stop_grace_s.
            # Killing it here could release the lock before descendants stop.
        child.wait()
        if child.stdin:
            child.stdin.close()
        self.child = None

    def backoff(self, count):
        low, high = self.plan.limits["backoff_seconds"]
        return min(high, low * 2**min(max(0, count - 1), 30))

    def health(self, stage):
        self.transition("paused_suspend")
        device = stage.get("device", self.plan.data.get("device", "cpu"))
        for attempt in range(1, self.plan.limits["probe_failures"] + 1):
            if self.stop_requested():
                return False
            self.last_probe = self.probe(device, self.plan.output, self.plan.limits["probe_timeout_s"])
            self.event("probe", attempt=attempt, result=self.last_probe)
            if self.last_probe.get("ok") is True:
                writer_path(self.control / "needs_wsl_restart").unlink(missing_ok=True)
                self.transition("running")
                # Probe/backoff time is not a second suspend.
                self.detector.previous = self.detector.sample()
                return True
            self.stop_child()
            if attempt < self.plan.limits["probe_failures"]:
                self.pause(self.backoff(attempt))
        self.transition("needs_wsl_restart")
        atomic_json(self.control / "needs_wsl_restart", {"time": self.clock.wall(), "probe": self.last_probe})
        return False

    def disk_guard(self):
        while self.disk(self.plan.output) < self.plan.limits["min_free_gb"]:
            if self.stop_requested():
                return False
            if self.state["status"] != "paused_disk":
                self.event("paused", reason="disk")
                self.transition("paused_disk")
            self.pause(60)
        return not self.stop_requested()

    def execute(self, stage, cycle):
        """Return (exit code, transient failure); stage artifacts decide completion."""
        command = self.stages.command(stage, cycle)
        directory = self.plan.stage_dir(stage, cycle)
        with writer_path(directory / "stage.log").open("a", encoding="utf-8") as log:
            self.child = self.launcher([sys.executable, "-m", "hasami_learn.autoloop.worker",
                                       str(self.plan.limits["stop_grace_s"]), *command],
                                      cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.PIPE,
                                      start_new_session=True, pass_fds=(self.store.lock.fileno(),),
                                      env={"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", **os.environ,
                                           "HASAMI_AUTOLOOP_STAGE_DIR": str(directory),
                                           "HASAMI_AUTOLOOP_DONE_FILE": str(writer_path(self.stages.done_file(stage, cycle))) if stage["kind"] == "module" else ""})
            progress = self.stages.progress(stage, cycle)
            last_progress = self.clock.monotonic()
            short_deadline = None
            transient = False
            self.detector.previous = self.detector.sample()
            while self.child is not None and self.child.poll() is None:
                if self.stop_requested():
                    self.stop_child()
                    return 0, False
                self.heartbeat()
                gap = self.detector.check(self.control / "resume-check")
                if gap:
                    self.event("suspend_detected", **gap)
                    transient = True
                    if not self.health(stage):
                        return 75, True
                    if self.child is None:
                        return 75, True
                    short_deadline = self.clock.monotonic() + self.plan.limits["post_resume_progress_s"]
                    last_progress = self.clock.monotonic()
                    progress = self.stages.progress(stage, cycle)
                current = self.stages.progress(stage, cycle)
                if current != progress:
                    progress, last_progress = current, self.clock.monotonic()
                    short_deadline = None
                    transient = False
                stalled = self.clock.monotonic() - last_progress >= self.plan.limits["stall_timeout_s"]
                if stalled or (short_deadline is not None and self.clock.monotonic() >= short_deadline):
                    self.event("stall", stage=stage["id"], post_resume=short_deadline is not None)
                    self.stop_child()
                    return 1, False  # Watchdog restarts always consume the failure budget.
                self.clock.sleep(self.plan.limits["heartbeat_s"])
            code = self.child.wait() if self.child else 75
            if self.child and self.child.stdin:
                self.child.stdin.close()
            self.child = None
            gap = self.detector.check(self.control / "resume-check")
            if gap:
                self.event("suspend_detected", **gap)
                transient = True
                if not self.health(stage):
                    return 75, True
        if code == 0:
            self.stages.finalize(stage, cycle)
        return code, transient or code == 75

    def run(self, *, allow_unvalidated=False, accept_config_change=False):
        if not self.store.acquire():
            print("loop already running (or its child is checkpointing)", file=sys.stderr)
            return 3
        handlers = {}
        handler = None
        try:
            self.control = writer_path(self.control)
            self.control.mkdir(exist_ok=True)
            cleanup_json_temps(self.directory, ("loop_state.json", "loop_state.json.bak", "status.json"))
            cleanup_json_temps(self.control, ("stop", "resume-check", "needs_wsl_restart"))
            self.stages.cleanup_existing()
            log_path = writer_path(self.directory / "loop.log")
            for index in range(1, 6):
                writer_path(self.directory / f"loop.log.{index}")
            handler = RotatingFileHandler(log_path, maxBytes=10 * 1024**2, backupCount=5)
            self.log = logging.getLogger(f"autoloop.{id(self)}")
            self.log.setLevel(logging.INFO)
            self.log.propagate = False
            self.log.addHandler(handler)
            if threading.current_thread() is threading.main_thread():
                for sig in (signal.SIGTERM, signal.SIGINT):
                    handlers[sig] = signal.signal(sig, lambda *_: setattr(self, "stopping", True))
            state = self.store.load()
            if state and state["config_sha256"] != self.plan.digest and not accept_config_change:
                raise LoopError("config changed; use --accept-config-change after reviewing the plan")
            try:
                evidence = self.gate(self.plan.data, allow_unvalidated)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                evidence = {"required": self.plan.data["require_validation"], "gate": "FAIL",
                            "commit": git("rev-parse", "HEAD"), "error": str(exc)}
            if state is None:
                state = {"schema_version": 1, "loop_id": self.plan.data["loop_id"], "config_sha256": self.plan.digest,
                         "code_commit": evidence["commit"], "status": "idle", "cycle": 1, "stage_index": 0,
                         "stages": {}, "best": None, "validation": evidence, "last_heartbeat": self.clock.wall(),
                         "config_changes": []}
            if state["loop_id"] != self.plan.data["loop_id"]:
                raise LoopError("state belongs to another loop")
            if state["config_sha256"] != self.plan.digest:
                state["config_changes"].append({"from": state["config_sha256"], "to": self.plan.digest, "time": self.clock.wall()})
                state["config_sha256"] = self.plan.digest
            validation_blocked = state["status"] == "blocked" and state["validation"].get("gate") == "FAIL"
            state["validation"] = evidence
            state["code_commit"] = evidence["commit"]
            self.state = state
            if evidence.get("gate") == "FAIL":
                self.event("blocked", reason=evidence["error"], validation=True)
                self.transition("blocked")
                print(evidence["error"], file=sys.stderr)
                return 1
            if state["status"] == "blocked" and not validation_blocked:
                return 1
            writer_path(self.control / "stop").unlink(missing_ok=True)
            self.transition("idle")
            return self._run()
        finally:
            self.stop_child()
            if self.state:
                self.heartbeat()
            for sig, previous in handlers.items():
                signal.signal(sig, previous)
            if handler:
                self.log.removeHandler(handler)
                handler.close()
            self.store.close()

    def _run(self):
        temporary_attempts = 0
        while self.state["cycle"] <= self.plan.data["cycles"]:
            if self.stop_requested():
                return self.stopped()
            cycle, index = self.state["cycle"], self.state["stage_index"]
            if index == len(self.plan.data["stages"]):
                self.state["cycle"] += 1
                self.state["stage_index"] = 0
                self.transition("idle")
                continue
            if index > len(self.plan.data["stages"]):
                raise LoopError("saved stage index exceeds changed plan")
            stage = self.plan.stage(index, cycle)
            self.stages.cleanup(stage, cycle)
            key = f"{cycle}:{stage['id']}"
            record = self.state["stages"].setdefault(key, {"status": "idle", "attempts": 0, "started_at": None,
                    "finished_at": None, "last_error": None, "outputs": {}})
            if record["status"] == "complete":
                self.state["stage_index"] += 1
                self.transition("idle")
                continue
            outputs = self.stages.complete(stage, cycle)
            if outputs:
                record.update(status="complete", finished_at=self.clock.wall(), outputs=outputs, last_error=None)
                best = self.plan.output / "models" / "best"
                if best.exists():
                    _, digest = checkpoint(best)
                    self.state["best"] = {"path": str(best), "sha256": digest}
                self.event("stage_done", stage=stage["id"], cycle=cycle, outputs=outputs)
                self.state["stage_index"] += 1
                self.transition("idle")
                temporary_attempts = 0
                continue
            if record["attempts"] >= self.plan.limits["max_attempts"]:
                record["status"] = "blocked"
                self.event("blocked", stage=stage["id"], reason=record["last_error"])
                self.transition("blocked")
                return 1
            if not self.disk_guard():
                return self.stopped()
            if (self.control / "needs_wsl_restart").exists():
                if not self.health(stage):
                    return self.stopped() if self.stop_requested() else 75
            # Recheck evidence before every stage, including retries.
            try:
                self.state["validation"] = self.gate(self.plan.data, self.state["validation"].get("allow_unvalidated", False))
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                self.state["validation"].update(gate="FAIL", error=str(exc))
                self.event("blocked", reason=str(exc), validation=True)
                self.transition("blocked")
                return 1
            self.plan.stage_dir(stage, cycle).mkdir(parents=True, exist_ok=True)
            record.update(status="running", started_at=self.clock.wall())
            self.transition("running")
            self.event("stage_start", stage=stage["id"], cycle=cycle)
            try:
                code, transient = self.execute(stage, cycle)
                error = f"stage exit {code}; completion not verified"
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.stop_child()
                code, transient, error = 1, False, str(exc)
            if self.stop_requested():
                return self.stopped()
            if self.state["status"] == "needs_wsl_restart":
                return 75
            if code == 0 and self.stages.complete(stage, cycle):
                continue
            # Even a nonzero exit may have committed a valid idempotent artifact.
            if self.stages.complete(stage, cycle):
                continue
            if not transient:
                record["attempts"] += 1
            temporary_attempts += 1
            record.update(status="retry", last_error=error)
            delay = self.backoff(temporary_attempts if transient else record["attempts"])
            self.event("stage_failed", stage=stage["id"], code=code, transient=transient, delay=delay)
            self.transition("idle")
            if record["attempts"] < self.plan.limits["max_attempts"]:
                self.event("restart", stage=stage["id"], delay=delay)
                self.pause(delay)
        self.transition("complete")
        return 0

    def stopped(self):
        self.stop_child()
        self.event("stop")
        self.transition("stopped")
        return 0
