"""One resumable family-N generation, run as an existing autoloop module stage."""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import shutil
import sys

from .control import Stop, atomic_json, identity, load_config, lock, read_json, sha256, validate_record
from .data import sync_directory
from .generate import generate, validate_manifest
from .match import verify_evidence
from hasami_learn.autoloop.evidence import wilson_interval


def verify_best(path):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    validate_record(manifest, "best_manifest")
    if manifest["format"] != "hlnn-best-v1" or manifest["status"] != "complete":
        raise ValueError("Incomplete NNUE best bundle")
    required = {"weights.hlnnw1", "training.json", "config.json", "data-manifest.json", "eval.json"}
    if set(manifest["files"]) != required:
        raise ValueError("NNUE best bundle missing evidence")
    for name, digest in manifest["files"].items():
        if sha256(path / name) != digest:
            raise ValueError("NNUE best bundle hash mismatch")
    evidence = read_json(path / "eval.json")
    successes, count = verify_evidence(evidence)
    if evidence["checkpoint_sha256"] != manifest["files"]["weights.hlnnw1"]:
        raise ValueError("Best weights/evidence mismatch")
    gate = read_json(path / "config.json")["gate"]
    if successes/count < gate["min_score"] or wilson_interval(successes, count)[0] < gate["min_lower"]:
        raise ValueError("Best bundle did not pass its gate")
    return path.resolve() / "weights.hlnnw1"


def gate_decision(candidate, evidence_path, config, *, opponent_sha256):
    """Compute a decision only from verified candidate and match evidence."""
    candidate, evidence_path = Path(candidate), Path(evidence_path)
    evidence = read_json(evidence_path)
    successes, count = verify_evidence(evidence)
    digest = identity(candidate / "weights.hlnnw1")
    if (evidence["checkpoint_sha256"] != digest
            or evidence["protocol"]["candidate_sha256"] != digest
            or evidence["protocol"]["opponent_sha256"] != opponent_sha256
            or evidence["protocol"]["settings"] != config["match"]):
        raise ValueError("Promotion weights/protocol mismatch")
    score, lower = successes/count, wilson_interval(successes, count)[0]
    accepted = score >= config["gate"]["min_score"] and lower >= config["gate"]["min_lower"]
    return dict(promoted=accepted, score=score, lower=lower, effective_games=count, sha256=digest,
                evidence_sha256=sha256(evidence_path))


def bundle_path(best, result):
    return Path(best).parent / f"candidate-{result['sha256']}-{result['evidence_sha256']}"


def promote(candidate, best, evidence_path, config, data_manifest, *, opponent_sha256):
    """Publish under the root lock; stale evidence cannot replace the current best."""
    candidate, best, evidence_path = Path(candidate), Path(best), Path(evidence_path)
    result = gate_decision(candidate, evidence_path, config, opponent_sha256=opponent_sha256)
    if not result["promoted"]:
        return result
    target = bundle_path(best, result)
    current = verify_best(best) if best.exists() else None
    if identity(current) != opponent_sha256:
        # A crash after replacing the link but before recording the decision is resumable.
        if current is not None and current.parent == target.resolve() and read_json(target / "config.json") == config:
            result["best"] = str(best)
            return result
        raise ValueError("Stale opponent evidence: current best changed")
    best.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = best.parent / ("." + target.name + ".tmp")
        temporary.mkdir(exist_ok=True)
        for source, name in ((candidate / "weights.hlnnw1", "weights.hlnnw1"),
                             (candidate / "training.json", "training.json"), (Path(data_manifest), "data-manifest.json"),
                             (evidence_path, "eval.json")):
            shutil.copyfile(source, temporary / name)
            with (temporary / name).open("rb") as stream:
                os.fsync(stream.fileno())
        atomic_json(temporary / "config.json", config)
        files = {name: sha256(temporary / name) for name in
                 ("weights.hlnnw1", "training.json", "config.json", "data-manifest.json", "eval.json")}
        atomic_json(temporary / "manifest.json", dict(format="hlnn-best-v1", status="complete", files=files))
        verify_best(temporary)
        os.replace(temporary, target)
        sync_directory(best.parent)
    verify_best(target)
    if read_json(target / "config.json") != config:
        raise ValueError("Promotion bundle configuration mismatch")
    pointer = best.parent / ".best.tmp"
    pointer.unlink(missing_ok=True)
    pointer.symlink_to(target.name, target_is_directory=True)
    os.replace(pointer, best)
    sync_directory(best.parent)
    result["best"] = str(best)
    return result


def run_generation(config_path, root, cycle, *, directory=None, done_file=None):
    if type(cycle) is not int or cycle < 1:
        raise ValueError("cycle must be a positive integer")
    config = load_config(config_path)
    if "replay" in config and cycle > config["replay"]["max_cycles"]:
        raise ValueError("Replay cycle budget exhausted")
    root = Path(root).resolve()
    directory = Path(directory or root / "generations" / f"c{cycle}").resolve()
    done_file = Path(done_file or directory / "done.json")
    best = root / "best"
    with lock(root), Stop() as stop:
        directory.mkdir(parents=True, exist_ok=True)
        state_path = directory / "generation.json"
        if state_path.exists():
            state = read_json(state_path)
            validate_record(state, "generation_state")
            if state["config"] != config or state["cycle"] != cycle:
                raise ValueError("NNUE generation resume configuration mismatch")
        else:
            if done_file.exists() or (directory / "gate.json").exists():
                raise ValueError("Completed generation is missing its state")
            previous = verify_best(best) if best.exists() else None
            state = dict(cycle=cycle, config=config, previous=str(previous) if previous else None,
                         previous_sha256=identity(previous))
            validate_record(state, "generation_state")
            atomic_json(state_path, state)
        previous = state["previous"]
        if identity(previous) != state["previous_sha256"]:
            raise ValueError("Previous weights changed during generation")
        gate_path = directory / "gate.json"
        if done_file.exists() or gate_path.exists():
            recorded = read_json(gate_path)
            expected = gate_decision(directory / "candidate", directory / "match/eval.json", config,
                                     opponent_sha256=state["previous_sha256"])
            expected.update(status="complete", cycle=cycle, previous_sha256=state["previous_sha256"])
            if expected["promoted"]:
                expected["best"] = str(best)
                bundle = bundle_path(best, expected)
                if (identity(verify_best(bundle)) != expected["sha256"]
                        or sha256(bundle / "eval.json") != expected["evidence_sha256"]
                        or read_json(bundle / "config.json") != config):
                    raise ValueError("Completed generation bundle mismatch")
            if recorded != expected or (done_file.exists() and read_json(done_file) != expected):
                raise ValueError("Completed generation decision mismatch")
            if not done_file.exists():
                atomic_json(done_file, recorded)
            return recorded
        data = directory / "data"
        generation = {**config["generation"], "seed": config["generation"]["seed"] + cycle - 1}
        if "replay" in config:
            generation["first_id"] += (cycle-1)*generation["games"]
        if (data / "manifest.json").exists() and read_json(data / "manifest.json")["status"] == "complete":
            manifest = validate_manifest(data)
        else:
            manifest = generate(data, generation, weights=previous, stop=stop)
        if manifest["settings"]["evaluator_sha256"] != state["previous_sha256"]:
            raise ValueError("Generated data evaluator differs from previous best")
        training_data = data
        if "distillation" in config:
            training_data = directory / "distilled"
            options = config["distillation"]
            command = [sys.executable, "-m", "hasami_learn.nnue.distill", "--data", str(data),
                       "--out", str(training_data), "--checkpoint", options["checkpoint"],
                       "--device", options.get("device", config["training"]["device"]),
                       "--batch-size", str(options.get("batch_size", 256))]
            stop.run(command)
        candidate = directory / "candidate"
        data_manifest = training_data / "manifest.json"
        if "replay" in config:
            from .window import select_window
            data_manifest = select_window(root, directory, cycle, config)
            data_args = ["--data-list", str(data_manifest)]
        else:
            data_args = ["--data", str(Path(glob.escape(str(training_data))) / "*.hlnn1")]
        command = [sys.executable, "-m", "hasami_learn.nnue.train", *data_args, "--out", str(candidate)]
        for key, value in config["training"].items():
            command += ["--"+key.replace("_", "-"), str(value).lower() if type(value) is bool else str(value)]
        if (candidate / "checkpoint.pt").exists():
            command += ["--resume", str(candidate / "checkpoint.pt")]
        elif previous:
            command += ["--init", previous]
        stop.run(command)
        match_dir = directory / "match"
        command = [sys.executable, "-m", "hasami_learn.nnue.match", "--candidate", str(candidate / "weights.hlnnw1"),
                   "--out", str(match_dir), "--settings", json.dumps(config["match"])]
        if previous:
            command += ["--opponent", previous]
        stop.run(command)
        result = promote(candidate, best, match_dir / "eval.json", config, data_manifest,
                         opponent_sha256=state["previous_sha256"])
        result.update(status="complete", cycle=cycle, previous_sha256=state["previous_sha256"])
        atomic_json(gate_path, result)
        atomic_json(done_file, result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--directory", default=os.environ.get("HASAMI_AUTOLOOP_STAGE_DIR"))
    parser.add_argument("--done-file", default=os.environ.get("HASAMI_AUTOLOOP_DONE_FILE") or None)
    args = vars(parser.parse_args(argv))
    args["config_path"] = args.pop("config")
    try:
        run_generation(**args)
    except InterruptedError:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
