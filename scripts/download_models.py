"""Explicit model download command using only the standard library."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
import urllib.request

BASE_URL = "https://github.com/shila24/hasmishogi/releases/download/"
NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
MAX_UNPACKED_SIZE = 4 * 1024 ** 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unpack_verified(archive: Path, expected: str, destination: Path, model: str, expected_members: dict, asset_kind="checkpoint"):
    """Verify before parsing; reject every unsafe member before creating model files."""
    if not NAME.fullmatch(model) or not re.fullmatch(r"[a-f0-9]{64}", expected):
        raise ValueError("Invalid model name or checksum")
    if sha256(archive) != expected:
        raise ValueError("SHA256 mismatch")
    if asset_kind not in ("checkpoint", "nnue"):
        raise ValueError("Unknown model asset kind")
    names = ("weights.hlnnw1", "weights.hlnnw1.json", "config.json", "eval.json") if asset_kind == "nnue" else ("manifest.json", "weights.pt")
    required = {model + "/" + name for name in names}
    if not isinstance(expected_members, dict) or not required <= expected_members.keys():
        raise ValueError("Incomplete checkpoint member inventory")
    for name, checksum in expected_members.items():
        if (not isinstance(name, str) or not name.startswith(model + "/")
                or any(p in ("", ".", "..") for p in name.split("/"))
                or "\\" in name or ":" in name or any(ord(c) < 32 for c in name)
                or not isinstance(checksum, str) or not re.fullmatch(r"[a-f0-9]{64}", checksum)):
            raise ValueError("Unsafe checkpoint member inventory")
    if destination.is_symlink():
        raise ValueError("Symlinked destination refused")
    # Resolve OS aliases such as macOS /tmp before creating our private staging directory.
    destination = destination.resolve()
    target = destination / model
    if target.exists() or target.is_symlink():
        raise ValueError("Model destination already exists")
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        seen = set()
        total = 0
        for member in members:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or any(p in ("", ".", "..") for p in member.name.split("/"))
                    or "\\" in member.name or ":" in member.name
                    or any(ord(c) < 32 for c in member.name)
                    or not member.isfile() or member.name in seen):
                raise ValueError("Unsafe or duplicate archive member")
            seen.add(member.name)
            if member.name not in expected_members:
                raise ValueError("Unexpected archive member")
            total += member.size
            if total > MAX_UNPACKED_SIZE or member.size < 0:
                raise ValueError("Archive exceeds unpacked size limit")
        if seen != expected_members.keys():
            raise ValueError("Incomplete checkpoint archive member list")
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".model-", dir=destination) as tmp:
            stage = Path(tmp) / model
            stage.mkdir()
            for member in members:
                target_file = stage.joinpath(*PurePosixPath(member.name).parts[1:])
                target_file.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target_file.open("xb") as output:
                    shutil.copyfileobj(source, output)
                if sha256(target_file) != expected_members[member.name]:
                    raise ValueError("Checkpoint member SHA256 mismatch")
            if asset_kind == "checkpoint":
                manifest = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))
                if manifest.get("weights_sha256") != sha256(stage / "weights.pt"):
                    raise ValueError("Checkpoint weights checksum mismatch")
            else:
                sidecar = json.loads((stage / "weights.hlnnw1.json").read_text(encoding="utf-8"))
                if sidecar.get("format") != "hlnnw1" or sidecar.get("bytes") != (stage / "weights.hlnnw1").stat().st_size:
                    raise ValueError("NNUE sidecar mismatch")
                evidence = json.loads((stage / "eval.json").read_text(encoding="utf-8"))
                if evidence.get("checkpoint_sha256") != sha256(stage / "weights.hlnnw1"):
                    raise ValueError("NNUE evaluation checksum mismatch")
            os.rename(stage, target)


def download(url: str, path: Path, limit: int):
    """Fetch only on explicit command execution, with a bounded response size."""
    with urllib.request.urlopen(url, timeout=60) as response, path.open("wb") as output:
        count = 0
        while chunk := response.read(1024 * 1024):
            count += len(chunk)
            if count > limit:
                raise ValueError("Download exceeds declared size")
            output.write(chunk)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    index = json.loads((root / "models/index.json").read_text(encoding="utf-8"))
    models = [row for row in index["models"] if args.model is None or row["name"] == args.model]
    if args.model and not models:
        parser.error("Unknown model")
    for row in models:
        name, tag = row["name"], row["release_tag"]
        if not NAME.fullmatch(name) or not re.fullmatch(r"v\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?", tag):
            raise ValueError("Unsafe model or release tag")
        if tag != index["version"] or row["asset"] != name + ".tar.gz":
            raise ValueError("Model index version or asset mismatch")
        if type(row["size"]) is not int or not 0 < row["size"] <= MAX_UNPACKED_SIZE:
            raise ValueError("Invalid archive size")
        with tempfile.TemporaryDirectory(prefix="hasami-download-") as tmp:
            sums = Path(tmp) / "SHA256SUMS"
            download(BASE_URL + tag + "/SHA256SUMS", sums, 1024 * 1024)
            entries = {}
            for line in sums.read_text(encoding="utf-8").splitlines():
                checksum, filename = line.split()
                if filename in entries:
                    raise ValueError("Duplicate SHA256SUMS entry")
                entries[filename] = checksum
            if entries.get(row["asset"]) != row["sha256"]:
                raise ValueError("SHA256SUMS differs from installed model index")
            archive = Path(tmp) / row["asset"]
            download(BASE_URL + tag + "/" + row["asset"], archive, row["size"])
            if archive.stat().st_size != row["size"]:
                raise ValueError("Archive size mismatch")
            unpack_verified(archive, row["sha256"], root / "models", name, row["members"], row.get("asset_kind", "checkpoint"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
