"""Self-contained copies of checkpoint directories for promotion, export and release assets.

A checkpoint directory holds ``manifest.json``, ``weights.pt`` and any extra files its learner writes (a practical-play
checkpoint adds ``levels.json``). A manifest whose ``inference_hint`` names an ``engine_checkpoint`` depends on another
checkpoint (hasami_learn/checkpoint_bundle.py); a bundle copies that dependency into ``engine/`` and points
the hint at it, so the copy works on its own. Standard library only: the autonomous-loop controller and the release
builder import this module without torch.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

ENGINE_DIR = "engine"
MANIFEST = "manifest.json"
WEIGHTS = "weights.pt"


class BundleError(ValueError):
    """A checkpoint cannot be bundled (missing files, links, hash mismatch, unsafe paths)."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(directory: Path) -> dict[str, Any]:
    path = directory / MANIFEST
    if not path.is_file() or not (directory / WEIGHTS).is_file():
        raise BundleError(f"{directory} is not a checkpoint directory (needs {MANIFEST} and {WEIGHTS})")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BundleError(f"{path}: {exc}") from exc


def _verify_weights(directory: Path, manifest: dict[str, Any]) -> str:
    actual = sha256_file(directory / WEIGHTS)
    expected = manifest.get("weights_sha256")
    if expected != actual:
        raise BundleError(f"{directory / WEIGHTS}: sha256 {actual} does not match the manifest ({expected})")
    return actual


TRAINING_FILES = {"learner_state.pt", "trainer_state.json", "replay.npz", "replay_snapshot.npz",
                  "run-config.json", "config.json", "metrics.jsonl"}


def _copy_files(source: Path, target: Path, skip: str | None = None, *, inference_only=False) -> None:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if skip is not None and relative.parts[0] == skip:
            continue
        if path.is_symlink():
            raise BundleError(f"{path}: checkpoints must not contain links")
        if inference_only and path.name in TRAINING_FILES:
            continue
        if path.is_dir():
            (target / relative).mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            (target / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target / relative)
        else:
            raise BundleError(f"{path}: not a regular file")


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    for directory in [*sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True), root]:
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def file_hashes(root: str | Path) -> dict[str, str]:
    """``{posix relative path: sha256}`` of every regular file under ``root``; links are refused."""
    root = Path(root)
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise BundleError(f"{path}: checkpoints must not contain links")
        if path.is_file():
            hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    return hashes


def engine_dependency(directory: str | Path) -> Path | None:
    """The engine checkpoint a checkpoint depends on (resolved, verified), or None."""
    directory = Path(directory)
    hint = _manifest(directory).get("inference_hint") or {}
    reference = hint.get("engine_checkpoint")
    if not reference:
        return None
    if Path(reference).is_absolute():
        raise BundleError(f"{directory}: engine_checkpoint must be relative to the checkpoint, got {reference}")
    engine = (directory / reference).resolve()
    _verify_engine(engine, hint.get("engine_sha256"))
    return engine


def _verify_engine(engine: Path, expected: str | None) -> None:
    manifest = _manifest(engine)
    actual = _verify_weights(engine, manifest)
    if expected != actual:
        raise BundleError(f"engine checkpoint {engine}: weights sha256 {actual} does not match engine_sha256 {expected}")
    if (manifest.get("inference_hint") or {}).get("engine_checkpoint"):
        raise BundleError(f"engine checkpoint {engine} must not depend on another engine")


def bundle_checkpoint(source: str | Path, target: str | Path, *, inference_only=False) -> dict[str, str]:
    """Copy checkpoint ``source`` into the new directory ``target`` as a self-contained bundle.

    Every regular file is copied (links are refused) and the weights hash is verified. When the manifest names an
    ``engine_checkpoint``, that checkpoint is verified against ``engine_sha256``, copied into ``target/engine`` and the
    copied manifest's hint is rewritten to ``"engine"``. The bundle is written into a temporary sibling, fsynced and
    renamed into place, so ``target`` either does not exist or is complete. Returns ``file_hashes(target)``.
    ``inference_only`` omits optimizer, trainer, replay and run metadata, including in dependencies.
    Promotion and training copies retain their existing complete-state behavior by default.
    """
    # Release inventory tooling loads the read-only helpers as a standalone
    # module. Only publication needs the package's writer lock.
    from .runlock import run_lock, writer_path

    # Autoloop scratch cleanup shares this namespace (for example .tmp_baseline).
    target = writer_path(target)
    with run_lock(target.parent, ".publish.lock") as directory:
        return _bundle_checkpoint(Path(source), directory / target.name, inference_only=inference_only)


def _bundle_checkpoint(source: Path, target: Path, *, inference_only=False) -> dict[str, str]:
    if target.exists():
        raise BundleError(f"{target} already exists")
    manifest = _manifest(source)
    _verify_weights(source, manifest)
    engine = engine_dependency(source)
    temporary = target.with_name(f".tmp_{target.name}")
    if temporary.is_symlink():
        raise BundleError(f"{temporary}: staging directory must not be a link")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        _copy_files(source, temporary, skip=ENGINE_DIR if engine is not None else None,
                    inference_only=inference_only)
        if engine is not None:
            _copy_files(engine, temporary / ENGINE_DIR, inference_only=inference_only)
            manifest["inference_hint"]["engine_checkpoint"] = ENGINE_DIR
            (temporary / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            _verify_engine(temporary / ENGINE_DIR, manifest["inference_hint"].get("engine_sha256"))
        _verify_weights(temporary, _manifest(temporary))
        _fsync_tree(temporary)
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return file_hashes(target)
