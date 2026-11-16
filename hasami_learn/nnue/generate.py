"""Run and verify complete-game native self-play without importing torch."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .control import GENERATION, ROOT, Stop, atomic_json, read_json, sha256, validate_section, validate_record
from .data import HEADER, RECORD, read_shard


def tool_path():
    for path in (ROOT / "build/shogi_search_selfplay", ROOT / "build/Release/shogi_search_selfplay.exe"):
        if path.is_file():
            return path
    raise RuntimeError("Build shogi_search_selfplay with scripts/build_native.py")


def validate_manifest(directory, *, complete=True, deep=False):
    directory = Path(directory)
    data = read_json(directory / "manifest.json")
    validate_record(data, "generation_manifest")
    if data["format"] != "hlnn-run-v1" or (complete and data["status"] != "complete"):
        raise ValueError("Incomplete generation manifest")
    settings = data["settings"]
    if settings != read_json(directory / "settings.json"):
        raise ValueError("Generation settings mismatch")
    rows = data["games"]
    expected = list(range(settings["first_id"], settings["first_id"] + len(rows)))
    if ([r["id"] for r in rows] != expected or len(rows) != data["completed_games"]
            or ((complete or data["status"] == "complete") and len(rows) != settings["games"])):
        raise ValueError("Missing or duplicate completed games")
    resident = 0
    for row in rows:
        name = f'g{row["id"]:012d}.hlnn1'
        if row["shard"] != name or row["plies"] != len(row["moves"]) or row["reason"] not in range(1, 7):
            raise ValueError("Invalid game metadata")
        path = directory / name
        side = Path(str(path) + ".json")
        if row["evicted"]:
            if path.exists() or side.exists():
                raise ValueError("Evicted shard remains")
            continue
        metadata = read_json(side)
        if (sha256(path) != row["sha256"] or sha256(side) != row["sidecar_sha256"]
                or metadata != dict(format="hlnn1", version=1, record_size=RECORD.size,
                                    records=row["records"], games=int(row["records"] > 0),
                                    bytes=HEADER.size + RECORD.size * row["records"], sha256=row["sha256"])
                or path.stat().st_size != metadata["bytes"]
                or path.stat().st_size + side.stat().st_size != row["bytes"]):
            raise ValueError("Generation shard/sidecar hash or count mismatch")
        resident += row["bytes"]
        if deep:
            for record in read_shard(path):
                if record.game_id != row["id"] or record.ply >= row["plies"]:
                    raise ValueError("Record game identity mismatch")
    if (resident != data["resident_bytes"] or resident > settings["byte_budget"]
            or sum(r["records"] for r in rows) != data["positions"]
            or sum(r["nodes"] for r in rows) != data["nodes"]
            or dict(Counter(str(r["reason"]) for r in rows)) != data["outcomes_by_reason"]):
        raise ValueError("Generation aggregate mismatch")
    return data


def generate(out, settings=None, *, weights=None, stop_file=None, done_file=None, stop=None):
    options = {**GENERATION, **(settings or {})}
    validate_section(options, "generation")
    command = [str(tool_path()), "--out", str(out)]
    for key, value in options.items():
        command += ["--" + key.replace("_", "-"), str(int(value) if type(value) is bool else value)]
    if weights:
        command += ["--weights", str(weights)]
    if stop_file:
        command += ["--stop-file", str(stop_file)]
    if stop is None:
        with Stop() as control:
            control.run(command)
    else:
        stop.run(command)
    result = validate_manifest(out, complete=False)
    if result["status"] != "complete":
        raise InterruptedError("Generation stopped after completed games")
    if done_file:
        atomic_json(done_file, {"status": "complete", "manifest_sha256": sha256(Path(out) / "manifest.json")})
    return result


def main(argv=None):
    from .control import load_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--stop-file")
    parser.add_argument("--done-file")
    args = parser.parse_args(argv)
    try:
        generate(args.out, load_config(args.config)["generation"], weights=args.weights,
                 stop_file=args.stop_file, done_file=args.done_file)
    except InterruptedError:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
