"""Stage commands and verified, idempotent local artifacts."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import uuid

from .state import LoopError, atomic_json, cleanup_json_temps, fsync_dir, read_json, sha256
from .evidence import model_name, utc_now, validate_document, validate_evaluation, wilson_interval, effective_sample
from ..runlock import RANDOM_SUFFIX, cleanup_temps, plain_directory, run_lock, writer_path


def _output_arguments(arguments):
    """Yield output value positions, flag prefixes and original destination paths."""
    from .plan import ROOT

    tokens = iter(enumerate(arguments))
    for index, argument in tokens:
        if argument in ("--out", "--output"):
            value = next(tokens, None)
            if value is None:
                continue
            index, argument = value
            prefix = ""
        elif argument.startswith(("--out=", "--output=")):
            flag, argument = argument.split("=", 1)
            prefix = flag + "="
        else:
            continue
        yield index, prefix, ROOT / Path(argument).expanduser()


def checkpoint(path):
    path = Path(path).resolve()
    if not (path / "manifest.json").is_file():
        pointer = read_json(path / "checkpoints" / "latest.json")
        target = (path / pointer["path"]).resolve()
        if not target.is_relative_to(path):
            raise LoopError("checkpoint pointer escapes run")
        path = target
    manifest = read_json(path / "manifest.json")
    if not isinstance(manifest, dict):
        raise LoopError("checkpoint manifest must be an object")
    weights = path / manifest.get("weights_file", "weights.pt")
    if not weights.resolve().is_relative_to(path) or sha256(weights) != manifest["weights_sha256"]:
        raise LoopError("checkpoint sha256 mismatch")
    return path, manifest["weights_sha256"]


def verify_release(path):
    path = Path(path)
    try:
        record = read_json(path / "release.json")
        validate_document(record, "release-candidate")
        evaluation = read_json(path / "eval.json")
        validate_evaluation(evaluation)
        _, digest = checkpoint(path)
        manifest = read_json(path / "manifest.json")
        if record["weights_sha256"] != digest or evaluation["checkpoint_sha256"] != digest:
            return False
        if record["game_id"] != manifest.get("game_id", "hasami"):
            return False
        if evaluation.get("game_id", record["game_id"]) != record["game_id"]:
            return False
        files = record["files"]
        if record.get("status") != "complete" or not isinstance(files, dict) or not {"weights.pt", "manifest.json", "eval.json", "config.json"} <= files.keys():
            return False
        actual = {file.relative_to(path).as_posix() for file in path.rglob("*")
                  if file.is_file() and file != path / "release.json"}
        if actual != files.keys():
            return False
        return all((path / name).resolve().is_relative_to(path.resolve()) and sha256(path / name) == digest
                   for name, digest in files.items())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def publish_checkpoint(source, target, extra=None, *, release=None):
    """Bundle a checkpoint and its metadata before publishing its directory."""
    target = writer_path(target)
    with run_lock(target.parent, ".publish.lock") as directory:
        cleanup_temps(directory, directories=(rf"\.tmp_{RANDOM_SUFFIX}",))
        return _publish_checkpoint(source, directory / target.name, extra, release=release)


def _publish_checkpoint(source, target, extra=None, *, release=None):
    # Called by local_stage subprocesses, never by the controller's monitor loop.
    from hasami_learn.checkpoint_bundle import bundle_checkpoint, file_hashes

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".tmp_", dir=target.parent))
    temporary = scratch / "checkpoint"
    try:
        bundle_checkpoint(source, temporary)
        for name, value in (extra or {}).items():
            atomic_json(temporary / name, value)
        if release is not None:
            validate_evaluation(read_json(temporary / "eval.json"))
            hashes = file_hashes(temporary)
            hashes.pop("release.json", None)
            record = {**release, "files": hashes}
            validate_document(record, "release-candidate")
            atomic_json(temporary / "release.json", record)
            if not verify_release(temporary):
                raise LoopError("release candidate failed verification")
        fsync_dir(temporary)
        os.replace(temporary, target)
        fsync_dir(target.parent)
    finally:
        shutil.rmtree(scratch)  # Only this operation's unpublished scratch.


class Stages:
    def __init__(self, plan):
        self.plan = plan

    def cleanup(self, stage, cycle):
        """Run only while the controller owns loop.lock and its stage is stopped."""
        directory = self.plan.stage_dir(stage, cycle)
        outputs = [directory / name for name in ("results-device.json", "eval.json", "gate.json")]
        if stage["kind"] == "module":
            outputs.append(self.done_file(stage, cycle))
            arguments = [self.plan.substitute(a, cycle) for a in stage.get("args", [])]
            for _, _, output in _output_arguments(arguments):
                outputs.extend((output, output.with_suffix(".done.json")))
        parents = {}
        for output in outputs:
            try:
                output = writer_path(output)
            except RuntimeError:
                continue
            parent = output.parent
            if parent.is_relative_to(directory) and plain_directory(parent):
                parents.setdefault(parent, []).append(output.name)
        for parent, names in parents.items():
            cleanup_json_temps(parent, names)
        if stage["kind"] in ("gate", "export"):
            self.cleanup_artifacts()

    def cleanup_existing(self):
        """Visit only this loop's known cycle/stage directories, never links."""
        for cycle_dir in writer_path(self.plan.loop_dir).iterdir():
            if not re.fullmatch(r"c[1-9][0-9]*", cycle_dir.name) or not plain_directory(cycle_dir):
                continue
            cycle = int(cycle_dir.name[1:])
            for index in range(len(self.plan.data["stages"])):
                self.cleanup(self.plan.stage(index, cycle), cycle)
        self.cleanup_artifacts()

    def cleanup_artifacts(self):
        # Artifact namespaces are shared across loops. Their own lock also
        # excludes a directly invoked local promotion/export process.
        with run_lock(self.plan.output, ".artifacts.lock") as output:
            for name in ("models", "release"):
                directory = output / name
                if plain_directory(directory):
                    with run_lock(directory, ".publish.lock") as directory:
                        cleanup_temps(directory, directories=(rf"\.tmp_{RANDOM_SUFFIX}",),
                                      links=(r"\.best-[0-9a-f]{32}",) if name == "models" else ())

    def source(self, stage, cycle):
        return self.plan.stage_dir({"id": stage["source"]}, cycle)

    def run_dir(self, stage, cycle):
        return writer_path(self.plan.output / "runs" / self.plan.run_id(stage, cycle))

    def candidate(self, stage, cycle):
        source = next(s for s in self.plan.data["stages"] if s["id"] == stage["source"])
        return checkpoint(self.run_dir(source, cycle))

    def target_iterations(self, stage):
        return stage["set"].get("run.iterations", read_json(self.plan.resolve(stage["config"]))["run"]["iterations"])

    def done_file(self, stage, cycle):
        value = self.plan.substitute(stage["done_file"], cycle)
        path = Path(value)
        return path if path.is_absolute() else self.plan.stage_dir(stage, cycle) / path

    def complete(self, stage, cycle):
        directory = self.plan.stage_dir(stage, cycle)
        try:
            kind = stage["kind"]
            if kind == "train":
                run = self.run_dir(stage, cycle)
                cp, digest = checkpoint(run)
                iteration = read_json(cp / "trainer_state.json")["iteration"]
                run_iteration = read_json(run / "trainer_state.json")["iteration"]
                if min(iteration, run_iteration) >= self.target_iterations(stage):
                    return {"checkpoint": str(cp), "sha256": digest, "iteration": iteration}
            elif kind == "evaluate":
                data = read_json(directory / "eval.json")
                validate_evaluation(data)
                cp, digest = self.candidate(stage, cycle)
                if data["checkpoint_sha256"] == digest and data["checkpoint"] == str(cp):
                    self.score(data)
                    return {"eval": str(directory / "eval.json"), "sha256": digest}
            elif kind == "gate":
                data = read_json(directory / "gate.json")
                evaluation = read_json(self.source(stage, cycle) / "eval.json")
                validate_evaluation(evaluation)
                if type(data["promoted"]) is bool and data["sha256"] == evaluation["checkpoint_sha256"]:
                    if data["promoted"]:
                        _, digest = checkpoint(data["best"])
                        if digest != data["sha256"]:
                            return None
                    return data
            elif kind == "module":
                data = read_json(self.done_file(stage, cycle))
                if isinstance(data, dict) and data.get("status") == "complete":
                    return {"done_file": str(self.done_file(stage, cycle))}
            elif kind == "export":
                target = self.release_dir(cycle)
                if verify_release(target):
                    return {"release": str(target)}
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def evaluation_opening_plies(self, stage, cycle):
        if "opening_plies" in stage:
            return stage["opening_plies"]
        source = next(s for s in self.plan.data["stages"] if s["id"] == stage["source"])
        saved = self.run_dir(source, cycle) / "config.json"
        if saved.is_file():
            return read_json(saved).get("evaluation", {}).get("opening_plies", 0)
        source_index = self.plan.data["stages"].index(source)
        resolved = self.plan.stage(source_index, cycle)
        return resolved["set"].get("evaluation.opening_plies",
                                   read_json(self.plan.resolve(source["config"])).get("evaluation", {}).get("opening_plies", 0))

    def command(self, stage, cycle):
        directory = self.plan.stage_dir(stage, cycle)
        kind = stage["kind"]
        cli = [sys.executable, "-m", "hasami_learn.cli"]
        if kind == "train":
            command = [*cli, "train", "--config", str(self.plan.resolve(stage["config"])),
                       "--run-id", self.plan.run_id(stage, cycle), "--output-dir", str(self.plan.output / "runs"),
                       "--resume", "auto", "--device", stage.get("device", self.plan.data.get("device", "cpu"))]
            import json
            for key, value in stage["set"].items():
                command.extend(["--set", f"{key}={json.dumps(value)}"])
            return command
        if kind == "evaluate":
            cp, _ = self.candidate(stage, cycle)
            command = [sys.executable, "-m", "hasami_learn.autoloop.evaluate",
                       "--checkpoint", str(cp), "--games", str(stage["games"]),
                       "--seed", str(stage["seed"]), "--device", stage.get("device", self.plan.data.get("device", "cpu")),
                       "--out", str(directory / "results.json"),
                       "--opening-plies", str(self.evaluation_opening_plies(stage, cycle)),
                       "--opening-seed", str(stage["seed"])]
            for opponent in stage["opponents"]:
                command.extend(["--opponent", self.plan.substitute(opponent, cycle)])
            for key in ("simulations", "max_plies", "mode", "backend"):
                if key in stage:
                    command.extend(["--" + key.replace("_", "-"), str(stage[key])])
            return command
        if kind == "module":
            # The controller passes the completion destination separately via
            # the environment. Validate it before launching any module work.
            writer_path(self.done_file(stage, cycle))
            arguments = [self.plan.substitute(a, cycle) for a in stage.get("args", [])]
            for index, prefix, output in _output_arguments(arguments):
                arguments[index] = prefix + str(writer_path(output))
            return [sys.executable, "-m", stage["module"], *arguments]
        return [sys.executable, "-m", "hasami_learn.autoloop.local_stage", str(self.plan.path),
                self.plan.digest, stage["id"], str(cycle)]

    def finalize(self, stage, cycle):
        if stage["kind"] == "evaluate":
            directory = self.plan.stage_dir(stage, cycle)
            cp, digest = self.candidate(stage, cycle)
            metadata = read_json(directory / "results-device.json")
            evaluation = {"schema_version": 1, "checkpoint": str(cp), "checkpoint_sha256": digest,
                          "created_at": metadata["created_at"], "device": metadata["device"],
                          "game_id": read_json(cp / "manifest.json").get("game_id", "hasami"),
                          "seed": stage["seed"], "results": read_json(directory / "results.json"),
                          "notes": "Rows with extra.sample_unit=opening_actions_and_seating carry Wilson 95% "
                                   "intervals on extra.effective_successes (sum of group mean scores) over "
                                   "extra.effective_games (group count). Legacy rows without sample metadata retain their raw interval; "
                                   "the gate credits at most two legacy seatings at the raw score rate. "
                                   "Raw game counts and score_rate remain reported. "
                                   "Learner MCTS uses simulations only; classical limits are unchanged.",
                          "protocol": {"games": stage["games"], "swap_sides": True,
                                       "opening_plies": self.evaluation_opening_plies(stage, cycle),
                                       "opening_seed": stage["seed"],
                                       "sample_unit": "opening_actions_and_seating",
                                       **{k: stage[k] for k in ("simulations", "max_plies", "mode", "backend") if k in stage}}}
            self.score(evaluation)
            for row in evaluation["results"].values():
                if row["games"] <= 0:
                    raise LoopError("evaluation opponent contains no games")
                successes = row["wins"] + 0.5 * row["draws"]
                row["score_rate"] = successes / row["games"]
                row["ci_low"], row["ci_high"] = wilson_interval(*effective_sample(row, legacy_raw=True))
            validate_evaluation(evaluation)
            atomic_json(directory / "eval.json", evaluation)

    @staticmethod
    def score(evaluation):
        if not isinstance(evaluation, dict) or not isinstance(evaluation.get("results"), dict):
            raise LoopError("evaluation results must be an object")
        successes = 0.0
        games = 0
        for row in evaluation["results"].values():
            counts = [row[k] for k in ("wins", "losses", "draws", "games", "truncated")]
            if any(type(v) is not int or v < 0 for v in counts) or sum(counts[:3]) != counts[3] or counts[4] > counts[2]:
                raise LoopError("invalid evaluation counts")
            if not row["games"]:
                raise LoopError("evaluation opponent contains no games")
            points, count = effective_sample(row)
            successes += points
            games += count
        if not games:
            raise LoopError("evaluation contains no games")
        return successes / games, wilson_interval(successes, games)[0]

    def release_dir(self, cycle):
        return writer_path(self.plan.output / "release" / f"{self.plan.data['loop_id']}-c{cycle}")

    def local(self, stage, cycle):
        with run_lock(self.plan.output, ".artifacts.lock"):
            self.cleanup_artifacts()
            return self._local(stage, cycle)

    def _local(self, stage, cycle):
        directory = self.plan.stage_dir(stage, cycle)
        models = writer_path(self.plan.output / "models")
        best = models / "best"
        if stage["kind"] == "gate":
            evaluation = read_json(self.source(stage, cycle) / "eval.json")
            validate_evaluation(evaluation)
            cp, digest = checkpoint(evaluation["checkpoint"])
            if digest != evaluation["checkpoint_sha256"]:
                raise LoopError("evaluated checkpoint changed")
            score, lower = self.score(evaluation)
            previous = None
            if best.exists():
                _, best_digest = checkpoint(best)
                previous = read_json(best / "eval.json")
                validate_evaluation(previous)
                if previous["checkpoint_sha256"] != best_digest:
                    raise LoopError("best evaluation does not match checkpoint")
                if previous["results"].keys() != evaluation["results"].keys() or previous["seed"] != evaluation["seed"]:
                    raise LoopError("candidate and best evaluation protocols differ")
                for key, default in (("opening_plies", 0), ("opening_seed", evaluation["seed"])):
                    if previous.get("protocol", {}).get(key, default) != evaluation.get("protocol", {}).get(key, default):
                        raise LoopError("candidate and best evaluation opening protocols differ")
            promote = score >= stage["min_score"] and lower >= stage["min_lower"] and (previous is None or score >= self.score(previous)[0])
            if promote:
                evaluation_digest = sha256(self.source(stage, cycle) / "eval.json")
                # Rebundle on every promotion attempt: weights/evaluation hashes
                # alone cannot identify changed extra files or engine hints.
                immutable = models / f"candidate-{digest}-{evaluation_digest[:16]}-{uuid.uuid4().hex}"
                publish_checkpoint(cp, immutable,
                                   {"eval.json": evaluation, "config.json": read_json(cp.parent.parent / "config.json")})
                checkpoint(immutable)
                temporary = writer_path(models / (".best-" + uuid.uuid4().hex))
                try:
                    temporary.symlink_to(immutable.name, target_is_directory=True)
                    os.replace(temporary, best)
                    fsync_dir(models)
                finally:
                    temporary.unlink(missing_ok=True)
            atomic_json(directory / "gate.json", {"promoted": promote, "sha256": digest, "score": score,
                                                   "lower": lower, "best": str(best) if best.exists() else None})
        elif stage["kind"] == "export":
            from .plan import git

            _, digest = checkpoint(best)
            target = self.release_dir(cycle)
            if target.exists():
                raise LoopError("existing release failed verification; refusing overwrite")
            release = {"schema_version": 1, "status": "complete",
                       "name": model_name(self.plan.data["loop_id"], cycle),
                       "loop_id": self.plan.data["loop_id"], "cycle": cycle,
                       "game_id": read_json(best / "manifest.json").get("game_id", "hasami"),
                       "created_at": utc_now(), "source_commit": git("rev-parse", "HEAD"),
                       "weights_sha256": digest, "evaluation": "eval.json", "config": "config.json",
                       "training_data": {"kind": "self-play", "human_game_records": False}}
            publish_checkpoint(best, target, {"loop-config.json": read_json(self.plan.path)}, release=release)

    def progress(self, stage, cycle):
        directory = self.run_dir(stage, cycle) if stage["kind"] == "train" else self.plan.stage_dir(stage, cycle)
        if stage["kind"] == "train":
            paths = [directory / "metrics.jsonl", directory / "checkpoints" / "latest.json"]
        else:
            paths = list(directory.rglob("*"))
            if stage["kind"] == "module":
                paths.append(self.done_file(stage, cycle))
        stats = []
        for path in paths:
            try:
                if path.is_file():
                    stat = path.stat()
                    stats.append((str(path), stat.st_size, stat.st_mtime_ns))
            except FileNotFoundError:
                pass
        return tuple(sorted(stats))
