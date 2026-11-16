"""Process-kill resume test, and the verification step of a manual power-loss test.

What this checks and what it does not:
- Default mode SIMULATES an abrupt process death: it SIGKILLs a training run right after a checkpoint, then resumes
  with ``--resume auto``. The kernel keeps running, so writes still in the page cache survive; this proves the
  checkpoint/rename/resume logic, NOT that data reached the disk.
- ``--verify-run DIR`` is the second half of a MANUAL power-loss test (docs/TRAINING_ja.md section 6): after a
  real power cut or ``wsl --shutdown`` interrupted a run, it resumes that run and checks the same postconditions.

Prints a JSON summary; exit code 0 only when every postcondition holds. Repository root:
    .venv/bin/python scripts/kill_resume_smoke.py [--backend cpp] [--iterations 6] [--kill-after 2]
    .venv/bin/python scripts/kill_resume_smoke.py --verify-run ~/hasami-power-test/power-loss --iterations 200
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONFIG = ROOT / "configs" / "tiny" / "riichi-ppo-league.json"
BOUNDARIES = ("after_checkpoint_publish", "during_trainer_state_write",
              "during_replay_write", "before_metrics_append")


def boundary_command(argv: list[str], boundary: str, iteration: int) -> list[str]:
    """Stop at the exact seam; the parent observes WUNTRACED and sends SIGKILL."""
    script = """import os, signal, sys
from hasami_learn import runlock
def stop(name, iteration):
    if name == sys.argv[1] and iteration == int(sys.argv[2]):
        os.kill(os.getpid(), signal.SIGSTOP)
runlock.checkpoint_boundary = stop
from hasami_learn.cli import main
raise SystemExit(main(sys.argv[3:]))
"""
    return [sys.executable, "-c", script, boundary, str(iteration), *argv[3:]]


def boundary_reached(child: subprocess.Popen) -> bool:
    """Poll a kernel stop notification, independent of filesystem timing."""
    pid, status = os.waitpid(child.pid, os.WNOHANG | os.WUNTRACED)
    if not pid:
        return False
    if os.WIFSTOPPED(status):
        return True
    child.returncode = os.waitstatus_to_exitcode(status)
    return False


def command(output: Path, run_id: str, iterations: int, backend: str) -> list[str]:
    return [sys.executable, "-m", "hasami_learn.cli", "train", "--config", str(CONFIG), "--output-dir", str(output),
            "--run-id", run_id, "--resume", "auto", "--quiet",
            "--set", f"run.iterations={iterations}", "--set", "run.checkpoint_every=1",
            "--set", f"run.backend={backend}", "--set", "device.device=cpu"]


def checkpoints(run: Path) -> list[str]:
    directory = run / "checkpoints"
    return sorted(p.name for p in directory.glob("iter_*")) if directory.is_dir() else []


def postconditions(run: Path, iterations: int, resume: subprocess.CompletedProcess, result: dict) -> bool:
    """Resume exited cleanly, reached the target, left no temporary files, and the final checkpoint loads."""
    result["resume_exit_code"] = resume.returncode
    final = checkpoints(run)
    result["checkpoints_final"] = final
    result["temporary_leftovers"] = [str(p.relative_to(run)) for p in run.rglob(".tmp_*")]
    kinds, saved = [], []
    metrics = run / "metrics.jsonl"
    for line in (metrics.read_text().splitlines() if metrics.is_file() else []):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            result["torn_metrics_lines"] = result.get("torn_metrics_lines", 0) + 1  # a cut can tear the last line
            continue
        kinds.append(row.get("kind"))
        if row.get("kind") == "checkpoint":
            saved.append(row.get("iteration"))
    result["resume_lines"] = kinds.count("resume")
    result["checkpoint_iterations"] = saved
    loadable = False
    if final:
        from hasami_learn.inference import create_agent
        loadable = create_agent(str(run / "checkpoints" / final[-1]), device="cpu").manifest.iteration == iterations
    result["final_checkpoint_loadable"] = loadable
    ok = (resume.returncode == 0 and f"iter_{iterations:06d}" in final and not result["temporary_leftovers"]
          and result["resume_lines"] >= 1 and loadable)
    if not ok:
        result["stderr"] = (resume.stderr or "")[-2000:]
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="cpp", choices=("cpp", "python"))
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--kill-after", type=int, default=2, help="kill once this iteration's checkpoint exists")
    parser.add_argument("--kill-boundary", choices=BOUNDARIES,
                        help="deterministically stop at this checkpoint boundary before SIGKILL (POSIX)")
    parser.add_argument("--verify-run", default=None, help="resume and check a run interrupted by real power loss")
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()
    if args.kill_boundary and (args.verify_run or os.name != "posix"):
        parser.error("--kill-boundary requires POSIX simulated-kill mode")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if args.verify_run:
        from hasami_learn.runlock import writer_path

        run = writer_path(Path(args.verify_run).expanduser())
        result: dict[str, object] = {"test": "manual power loss (verification after restart)",
                                     "checkpoints_before_resume": checkpoints(run)}
        resume = subprocess.run(command(run.parent, run.name, args.iterations, args.backend), cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=args.timeout)
        result["ok"] = postconditions(run, args.iterations, resume, result)
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1
    if not 1 <= args.kill_after < args.iterations:
        parser.error("--kill-after must be in [1, iterations)")
    result = {"test": "simulated process kill (SIGKILL; not a power-loss test)", "backend": args.backend,
              "iterations": args.iterations, "kill_after": args.kill_after}
    if args.kill_boundary:
        result["kill_boundary"] = args.kill_boundary
    with tempfile.TemporaryDirectory(prefix="hasami-kill-resume-") as tmp:
        output = Path(tmp)
        run = output / "kill-resume"
        target = f"iter_{args.kill_after:06d}"
        argv = command(output, run.name, args.iterations, args.backend)
        if args.kill_boundary:
            argv = boundary_command(argv, args.kill_boundary, args.kill_after)
        first = subprocess.Popen(argv, cwd=ROOT, env=env,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + args.timeout
        while not (boundary_reached(first) if args.kill_boundary else target in checkpoints(run)):
            if first.poll() is not None or time.monotonic() > deadline:
                first.kill()
                result.update(ok=False, error="the first run ended or timed out before the kill point",
                              stderr=(first.stderr.read() or "")[-2000:])
                print(json.dumps(result, indent=2))
                return 1
            time.sleep(0.02)
        first.send_signal(signal.SIGKILL)
        first.wait()
        first.stderr.close()
        result["checkpoints_after_kill"] = checkpoints(run)
        resume = subprocess.run(command(output, run.name, args.iterations, args.backend), cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=args.timeout)
        result["ok"] = postconditions(run, args.iterations, resume, result)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
