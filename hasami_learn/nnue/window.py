"""Immutable cumulative training selections; source shards are never deleted here."""
from __future__ import annotations

from pathlib import Path

from .control import atomic_json, read_json, sha256, validate_record
from .generate import validate_manifest


def select_window(root, directory, cycle, config):
    """Register completed data and freeze a whole-game window for this candidate."""
    root, directory = Path(root), Path(directory)
    policy = config["replay"]
    if cycle > policy["max_cycles"]:
        raise ValueError("Replay cycle budget exhausted")
    index_dir = root / "replay"
    index_dir.mkdir(exist_ok=True)
    data = directory / "data"
    current = dict(cycle=cycle, directory=str(data), manifest_sha256=sha256(data / "manifest.json"))
    validate_record(current, "replay_index")
    index_path = index_dir / f"c{cycle}.json"
    if index_path.exists() and read_json(index_path) != current:
        raise ValueError("Replay cycle registration changed")
    atomic_json(index_path, current)
    snapshot = directory / "training-data.json"
    if snapshot.exists():
        value = read_json(snapshot)
        validate_record(value, "training_window")
        if value["cycle"] != cycle or value["policy"] != policy:
            raise ValueError("Replay window resume mismatch")
        return snapshot
    entries, positions = [], 0
    for previous in range(cycle, 0, -1):
        registered = read_json(index_dir / f"c{previous}.json")
        validate_record(registered, "replay_index")
        source = Path(registered["directory"])
        if registered["cycle"] != previous or sha256(source / "manifest.json") != registered["manifest_sha256"]:
            raise ValueError("Replay source manifest changed")
        manifest = validate_manifest(source)
        generation = config["generation"]
        expected_id = generation["first_id"] + (previous-1)*generation["games"]
        if (manifest["settings"]["first_id"] != expected_id
                or manifest["settings"]["games"] != generation["games"]
                or manifest["resident_bytes"] > generation["byte_budget"]):
            raise ValueError("Replay source ID range or data budget mismatch")
        # Old games are left on disk for audit/resume; only this bounded view is trained.
        for game in reversed(manifest["games"]):
            if game["evicted"] or not game["records"]:
                continue
            if positions + game["records"] > policy["window_positions"]:
                break
            entries.append(dict(path=str(source / game["shard"]), sha256=game["sha256"],
                                sidecar_sha256=game["sidecar_sha256"], records=game["records"]))
            positions += game["records"]
        else:
            continue
        break
    if not entries:
        raise ValueError("Replay position window cannot hold a complete game")
    value = dict(format="hlnn-window-v1", cycle=cycle, policy=policy,
                 positions=positions, shards=list(reversed(entries)))
    validate_record(value, "training_window")
    atomic_json(snapshot, value)
    return snapshot


def window_paths(path):
    value = read_json(path)
    validate_record(value, "training_window")
    if sum(row["records"] for row in value["shards"]) != value["positions"]:
        raise ValueError("Replay window position count mismatch")
    paths = []
    for row in value["shards"]:
        shard = Path(row["path"])
        if sha256(shard) != row["sha256"] or sha256(str(shard)+".json") != row["sidecar_sha256"]:
            raise ValueError("Replay window shard hash mismatch")
        paths.append(str(shard))
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate replay shard")
    return paths
