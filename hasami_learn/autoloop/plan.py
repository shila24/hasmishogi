"""Plan validation and production evidence gate, without learning imports."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import re
import subprocess

from .state import LoopError, read_json, validate
from ..runlock import writer_path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIMITS = dict(max_attempts=5, backoff_seconds=[10, 1800], stall_timeout_s=7200,
                      probe_timeout_s=120, probe_failures=3, min_free_gb=20,
                      heartbeat_s=30, suspend_gap_s=120, stop_grace_s=900,
                      post_resume_progress_s=900)


def git(*args, root=ROOT):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.PIPE).strip()


def validation_gate(config, allow=False, root=ROOT):
    required = config["require_validation"]
    if required and allow:
        raise LoopError("--allow-unvalidated cannot bypass production validation")
    commit = git("rev-parse", "HEAD", root=root)
    if not required:
        return {"required": False, "commit": commit, "allow_unvalidated": allow}
    directory = Path(config["validation_results"])
    try:
        summary = read_json(directory / "summary.json")
        env = read_json(directory / "env.json")
        if not isinstance(summary, dict) or not isinstance(env, dict):
            raise LoopError("validation reports must be objects")
        checkout = env["checkout"]
        reported = checkout["commit"]
        if not isinstance(reported, str) or not re.fullmatch(r"[0-9a-f]{12,40}", reported):
            raise LoopError("invalid validation commit")
        # The existing recorder emits 12 hex digits. Resolve them, rather than
        # accepting an arbitrary prefix, and compare full object identities.
        validated_commit = git("rev-parse", "--verify", reported + "^{commit}", root=root)
        clean = not git("status", "--porcelain", "--untracked-files=no", root=root)
        if summary.get("gate") != "PASS" or checkout.get("dirty") is not False or not clean or validated_commit != commit:
            raise LoopError("validation requires PASS, matching commit, and clean validation/current tracked trees")
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        raise LoopError(f"validation gate refused: {exc}") from exc
    return {"required": True, "gate": "PASS", "commit": commit, "results": str(directory)}


class Plan:
    def __init__(self, path, *, running=False):
        self.path = Path(path).resolve()
        self.data = read_json(self.path)
        validate(self.data, "config")
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.data = copy.deepcopy(self.data)
        self.output = writer_path(ROOT / Path(self.data["output_dir"]).expanduser())
        if self.output == Path("/mnt") or Path("/mnt") in self.output.parents:
            raise LoopError("output_dir must be on the Linux filesystem, not /mnt")
        self.data["output_dir"] = str(self.output)
        if self.data.get("validation_results"):
            self.data["validation_results"] = str(self.resolve(self.data["validation_results"]))
        if self.data["require_validation"] and not self.data.get("validation_results"):
            raise LoopError("production plans require validation_results")
        self.limits = {**DEFAULT_LIMITS, **self.data.get("limits", {})}
        if self.limits["backoff_seconds"][0] > self.limits["backoff_seconds"][1]:
            raise LoopError("backoff minimum exceeds maximum")
        self.loop_dir = writer_path(self.output / self.data["loop_id"])
        self.warnings = []
        ids = set()
        kinds = {}
        for stage in self.data["stages"]:
            if stage["id"] in ids:
                raise LoopError("duplicate stage id")
            ids.add(stage["id"])
            required = {"train": ["config"], "evaluate": ["source", "opponents", "games", "seed"],
                        "gate": ["source", "min_score", "min_lower"], "module": ["module", "done_file"], "export": []}
            for key in required[stage["kind"]]:
                if key not in stage:
                    raise LoopError(f"stage {stage['id']} requires {key}")
            if stage.get("source") and stage["source"] not in ids - {stage["id"]}:
                raise LoopError("stage source must precede its consumer")
            expected = {"evaluate": "train", "gate": "evaluate"}.get(stage["kind"])
            if expected and kinds.get(stage["source"]) != expected:
                raise LoopError(f"{stage['kind']} requires a {expected} source")
            kinds[stage["id"]] = stage["kind"]
            if stage["kind"] == "train":
                source = self.resolve(stage["config"])
                if not source.is_file():
                    self.warnings.append(f"missing training config: {source}")
                else:
                    training = read_json(source)
                    if training.get("datastore", {}).get("max_games", 10000) > 10000 or training.get("datastore", {}).get("max_bytes", 30 * 2**30) > 30 * 2**30:
                        raise LoopError("self-play data caps exceed 10000 games / 30 GiB")
            if stage["kind"] == "module":
                module = stage["module"]
                if not re.fullmatch(r"hasami_learn\.[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module):
                    raise LoopError("module must be under hasami_learn")
                module_path = ROOT.joinpath(*module.split("."))
                if not module_path.with_suffix(".py").is_file() and not (module_path / "__main__.py").is_file():
                    self.warnings.append(f"missing module: {module}")
            for overrides in [stage.get("set", {}), *stage.get("cycle_overrides", {}).values()]:
                for key, value in overrides.items():
                    if key in ("run.run_id", "run.output_dir", "datastore.path"):
                        raise LoopError(f"controller owns {key}")
                    if key == "datastore.max_games" and (type(value) is not int or not 1 <= value <= 10000):
                        raise LoopError("invalid datastore.max_games")
                    if key == "datastore.max_bytes" and (type(value) is not int or not 1 <= value <= 30 * 2**30):
                        raise LoopError("invalid datastore.max_bytes")
        if running and self.warnings:
            raise LoopError("; ".join(self.warnings))

    def resolve(self, path):
        # Repository-relative paths are consistent with the existing CLI configs.
        return (ROOT / Path(path).expanduser()).resolve()

    def stage(self, index, cycle):
        stage = copy.deepcopy(self.data["stages"][index])
        stage["set"] = {**stage.get("set", {}), **stage.get("cycle_overrides", {}).get(str(cycle), {})}
        def substitute(value):
            if isinstance(value, str):
                return self.substitute(value, cycle)
            if isinstance(value, dict):
                return {key: substitute(item) for key, item in value.items()}
            if isinstance(value, list):
                return [substitute(item) for item in value]
            return value
        stage["set"] = substitute(stage["set"])
        return stage

    def run_id(self, stage, cycle):
        return f"{self.data['loop_id']}-{stage['id']}-c{cycle}"

    def stage_dir(self, stage, cycle):
        return writer_path(self.loop_dir / f"c{cycle}" / stage["id"])

    def substitute(self, value, cycle):
        return value.replace("{cycle}", str(cycle)).replace("{best}", str(self.output / "models" / "best"))
