"""Model manifest and checkpoint layout (see hasami_learn/schemas/model-manifest.schema.json)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

from .games import get_spec, default_rules, action_encoding
from .registry import NETWORKS
from .schema_validation import SCHEMA_DIR, load_schema, validate as _validate
from . import runlock

MANIFEST_ID = "hasami-model-manifest-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_NAME = "model-manifest.schema.json"
SCHEMA_PATH = SCHEMA_DIR / SCHEMA_NAME  # packaged copy of hasami_learn/schemas/model-manifest.schema.json
WEIGHTS_FILE = "weights.pt"
LEARNER_STATE_FILE = "learner_state.pt"
TRAINER_STATE_FILE = "trainer_state.json"
MANIFEST_FILE = "manifest.json"
LATEST_FILE = "latest.json"


class ManifestError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ModelManifest:
    algorithm: str
    network: dict[str, Any]
    encoding_id: str
    num_planes: int
    schema_version: int = 1
    manifest_id: str = MANIFEST_ID
    game_id: str = "hasami"
    num_players: int = 2
    game_params: dict[str, Any] = field(default_factory=dict)
    encoder_params: dict[str, Any] = field(default_factory=dict)  # constructor kwargs for spec.make_encoder
    board_shape: tuple[int, int] = field(default_factory=lambda: get_spec().board_shape)
    ruleset_id: str = field(default_factory=lambda: get_spec().ruleset_id)
    rules: dict[str, Any] = field(default_factory=lambda: default_rules(get_spec()))
    action_size: int = field(default_factory=lambda: get_spec().action_size)
    action_encoding: str = field(default_factory=lambda: action_encoding(get_spec()))
    value_perspective: str = "side_to_move"
    algorithm_params: dict[str, Any] = field(default_factory=dict)
    framework: dict[str, str] = field(default_factory=lambda: {"name": "torch", "version": torch.__version__})
    weights_file: str = WEIGHTS_FILE
    weights_sha256: str = "0" * 64
    created_at: str = ""
    run_id: str = ""
    iteration: int = 0
    train_device: str = ""
    inference_hint: dict[str, Any] = field(default_factory=lambda: {"mode": "policy", "simulations": 0})
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    init_checkpoint: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["board_shape"] = list(self.board_shape)
        if self.init_checkpoint is None:
            data.pop("init_checkpoint")
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelManifest":
        data = dict(data)
        if "game_id" not in data:
            data["game_id"] = "hasami"
            data.setdefault("board_shape", list(get_spec().board_shape))
        errors = validate_manifest_dict(data)
        if errors:
            raise ManifestError("invalid manifest:\n" + "\n".join(errors))
        return cls(**data)

    def validate(self) -> list[str]:
        return validate_manifest_dict(self.to_dict())


def validate_manifest_dict(data: dict[str, Any]) -> list[str]:
    """Validate against the packaged model-manifest schema (see hasami_learn.schema_validation)."""
    return _validate(load_schema(SCHEMA_NAME), data)


def fsync_directory(directory: str | Path) -> None:
    """Persist directory entries on POSIX filesystems."""
    directory = runlock.writer_path(directory)
    if os.name == "nt":
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_json(path: str | Path, data: Any, *, boundary: str | None = None) -> Path:
    """Publish a complete JSON file and persist the rename."""
    target = runlock.writer_path(path)
    # Standalone evaluation/calibration writers do not own the trainer or loop
    # lock. Exclude their cleanup from before mkstemp through final publication.
    with runlock.run_lock(target.parent, runlock.FILE_LOCK, blocking=True):
        return _atomic_write_json(target, data, boundary=boundary)


def _atomic_write_json(target: Path, data: Any, *, boundary: str | None) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".tmp_{target.name}_", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if boundary is not None:
            runlock.checkpoint_boundary(boundary, data["iteration"])
        os.replace(temporary, target)
        fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def save_checkpoint(
    directory: str | Path,
    network: torch.nn.Module,
    manifest: ModelManifest,
    learner_state: dict[str, Any] | None = None,
    trainer_state: dict[str, Any] | None = None,
    *,
    on_checkpoint: Callable[[Path], None] | None = None,
) -> Path:
    target = runlock.writer_path(directory)
    # Both the trainer and independent pool writers share the run lock.
    root = next((parent.parent for parent in target.parents if parent.name in ("checkpoints", "pool")), target.parent)
    with runlock.run_lock(root):
        return _save_checkpoint(target, network, manifest, learner_state, trainer_state, on_checkpoint=on_checkpoint)


def _save_checkpoint(target, network, manifest, learner_state, trainer_state, *, on_checkpoint):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = runlock.writer_path(target.with_name(f".tmp_{target.name}"))
    temporary.mkdir()
    backup = None
    published = False
    try:
        weights_path = temporary / WEIGHTS_FILE
        state = {key: value.detach().cpu() for key, value in network.state_dict().items()}
        torch.save(state, weights_path)
        manifest.weights_file = WEIGHTS_FILE
        manifest.weights_sha256 = sha256_file(weights_path)
        if not manifest.created_at:
            manifest.created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        errors = manifest.validate()
        if errors:
            raise ManifestError("refusing to write invalid manifest:\n" + "\n".join(errors))
        atomic_write_json(temporary / MANIFEST_FILE, manifest.to_dict())
        if learner_state is not None:
            torch.save(_cpu(learner_state), temporary / LEARNER_STATE_FILE)
        if trainer_state is not None:
            atomic_write_json(temporary / TRAINER_STATE_FILE, trainer_state)
        if on_checkpoint is not None:
            on_checkpoint(temporary)
        for path in temporary.rglob("*"):
            if path.is_file():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
        for path in sorted(temporary.rglob("*"), reverse=True):
            if path.is_dir():
                fsync_directory(path)
        fsync_directory(temporary)
        if target.exists():
            # POSIX cannot replace a nonempty directory. Retire the old generation under a
            # deterministic name (no random suffix, so recovery can parse it unambiguously);
            # a stale backup from an earlier crash is discarded because ``target`` is valid here.
            backup = runlock.writer_path(target.with_name(f"{REPLACED_PREFIX}{target.name}"))
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(target, backup)
        try:
            os.replace(temporary, target)
            published = True
        except OSError:
            if backup is not None:
                os.replace(backup, target)
                backup = None
            raise
        fsync_directory(target.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        if published and backup is not None and backup.exists():
            shutil.rmtree(backup)
    return target


def _cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu(v) for v in value)
    return value


def check_manifest_matches_spec(manifest: "ModelManifest", spec, game_params: dict | None = None) -> None:
    """Refuse a checkpoint that does not belong to the selected game variant.

    ``load_checkpoint`` validates a checkpoint against the spec reconstructed from its own manifest;
    callers that already selected a game (ladder, evaluation) must additionally compare identity and
    dimensions against that selection.
    """
    expected = {
        "game_id": spec.game_id,
        "num_players": spec.num_players,
        "ruleset_id": getattr(spec, "ruleset_id", manifest.ruleset_id),
        "board_shape": list(spec.board_shape),
        "action_size": spec.action_size,
    }
    actual = {
        "game_id": manifest.game_id,
        "num_players": manifest.num_players,
        "ruleset_id": manifest.ruleset_id,
        "board_shape": list(manifest.board_shape),
        "action_size": manifest.action_size,
    }
    if game_params is not None:
        expected["game_params"] = dict(game_params)
        actual["game_params"] = dict(manifest.game_params or {})
    encoding = getattr(spec, "action_encoding", None)
    if encoding is not None:
        expected["action_encoding"] = encoding
        actual["action_encoding"] = manifest.action_encoding
    for key, want in expected.items():
        if actual[key] != want:
            raise ManifestError(f"checkpoint {key} {actual[key]!r} does not match the selected game ({want!r})")


def write_latest_pointer(run_dir: str | Path, checkpoint_dir: str | Path, iteration: int) -> Path:
    run = runlock.writer_path(run_dir)
    checkpoint = runlock.writer_path(checkpoint_dir)
    pointer = runlock.writer_path(run / "checkpoints" / LATEST_FILE)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    relative = checkpoint.relative_to(run).as_posix()
    atomic_write_json(pointer, {"path": relative, "iteration": int(iteration)})
    return pointer


def resolve_checkpoint_dir(path: str | Path) -> Path:
    """Accept a checkpoint dir, a run dir, or a checkpoints/ dir and return the checkpoint dir."""
    base = Path(path).expanduser().resolve()
    if (base / MANIFEST_FILE).is_file():
        return base
    for pointer in (base / LATEST_FILE, base / "checkpoints" / LATEST_FILE):
        if pointer.is_file():
            data = json.loads(pointer.read_text(encoding="utf-8"))
            run_root = pointer.parent.parent if pointer.parent.name == "checkpoints" else pointer.parent
            candidate = (run_root / data["path"]).resolve()
            if (candidate / MANIFEST_FILE).is_file():
                return candidate
            raise ManifestError(f"latest pointer {pointer} references missing checkpoint {candidate}")
    raise ManifestError(f"no checkpoint found at {base} (expected manifest.json or latest.json)")


@dataclass
class LoadedCheckpoint:
    directory: Path
    manifest: ModelManifest
    network: torch.nn.Module | None
    learner_state: dict[str, Any] | None
    trainer_state: dict[str, Any] | None


def load_checkpoint(
    directory: str | Path,
    device: torch.device | str = "cpu",
    expect_rules: Any | None = None,
    build_network: bool = True,
    verify_hash: bool = True,
    expect_game_id: str | None = None,
    expect_compatibility: dict[str, Any] | None = None,
    load_training_state: bool = True,
) -> LoadedCheckpoint:
    from .devices import resolve_device
    from . import networks  # noqa: F401  (registers network factories)

    target = resolve_checkpoint_dir(directory)
    manifest = ModelManifest.from_dict(json.loads((target / MANIFEST_FILE).read_text(encoding="utf-8")))
    if expect_compatibility is not None:
        actual = manifest.to_dict()
        for key, expected in expect_compatibility.items():
            value = actual
            for part in key.split("."):
                value = value.get(part, {})
            if value != expected:
                raise ManifestError(f"checkpoint {key} {value!r} != configured {expected!r}")
    if expect_game_id is not None and manifest.game_id != expect_game_id:
        raise ManifestError(f"checkpoint game {manifest.game_id} != {expect_game_id}")
    try:
        spec = get_spec(manifest.game_id, manifest.game_params)
    except (KeyError, ValueError) as exc:
        raise ManifestError(f"unknown or invalid checkpoint game: {exc}") from exc
    if manifest.ruleset_id != spec.ruleset_id:
        raise ManifestError(f"checkpoint ruleset {manifest.ruleset_id} != {spec.ruleset_id}")
    if int(manifest.num_players) != int(getattr(spec, "num_players", 2)):
        raise ManifestError(f"checkpoint num_players {manifest.num_players} != {getattr(spec, 'num_players', 2)} of game {spec.game_id}")
    expected_rules = default_rules(spec) if expect_rules is None else (
        expect_rules.to_dict() if hasattr(expect_rules, "to_dict") else expect_rules)
    if manifest.rules != expected_rules:
        raise ManifestError(f"checkpoint rules {manifest.rules} do not match {expected_rules}")
    if (manifest.action_size != spec.action_size or tuple(manifest.board_shape) != tuple(spec.board_shape)
            or manifest.action_encoding != action_encoding(spec)):
        raise ManifestError("checkpoint board shape or action encoding is incompatible")
    encoder = spec.make_encoder(manifest.encoding_id, **(manifest.encoder_params or {}))
    if encoder.num_planes != manifest.num_planes:
        raise ManifestError("checkpoint encoder plane count is incompatible")
    weights_path = target / manifest.weights_file
    if not weights_path.is_file():
        raise ManifestError(f"weights file missing: {weights_path}")
    if verify_hash and sha256_file(weights_path) != manifest.weights_sha256:
        raise ManifestError(f"weights sha256 mismatch for {weights_path}")
    resolved = resolve_device(device)
    network = None
    if build_network:
        factory = NETWORKS.get(manifest.network["name"])
        network = factory(head=manifest.network["head"], num_planes=manifest.num_planes, board_shape=spec.board_shape, action_size=spec.action_size, **manifest.network.get("params", {}))
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        network.load_state_dict(state, strict=True)
        network.to(resolved).eval()
    learner_state = None
    if load_training_state and (target / LEARNER_STATE_FILE).is_file():
        learner_state = torch.load(target / LEARNER_STATE_FILE, map_location="cpu", weights_only=True)
    trainer_state = None
    if load_training_state and (target / TRAINER_STATE_FILE).is_file():
        trainer_state = json.loads((target / TRAINER_STATE_FILE).read_text(encoding="utf-8"))
    return LoadedCheckpoint(target, manifest, network, learner_state, trainer_state)


def resume_run_dir(path: str | Path) -> Path:
    """Locate the run root for a run, checkpoints directory, or iteration."""
    base = runlock.writer_directory(Path(path).expanduser())
    if base.parent.name == "checkpoints":
        return base.parent.parent
    if base.name == "checkpoints":
        return base.parent
    return base


REPLACED_PREFIX = ".tmp_replaced_"


def _restore_retired_generation(backup: Path) -> Path | None:
    """Put a ``.tmp_replaced_<name>`` backup back if its committed directory vanished mid-replace."""
    if not backup.name.startswith(REPLACED_PREFIX):
        return None
    committed_name = backup.name[len(REPLACED_PREFIX):]
    committed = backup.parent / committed_name
    if committed.is_symlink():
        return None
    if _valid_generation(committed, require_training=(backup / TRAINER_STATE_FILE).is_file()):
        return None
    if committed.exists():
        shutil.rmtree(committed)
    os.replace(backup, committed)
    return committed


def _valid_generation(directory: Path, *, require_training: bool = False) -> bool:
    if not runlock.plain_directory(directory) or any(p.is_symlink() for p in directory.rglob("*")):
        return False
    try:
        loaded = load_checkpoint(directory, build_network=False)
        if require_training:
            _check_training_state(loaded)
    except Exception:
        return False
    return True


def recover_retired_checkpoints(run_dir: str | Path) -> list[dict[str, str]]:
    """Recover interrupted replacements while excluding every run writer."""
    with runlock.run_lock(run_dir) as directory:
        return _recover_retired_checkpoints(directory)


def _recover_retired_checkpoints(run_dir: Path) -> list[dict[str, str]]:
    restored = []
    for parent in (Path(run_dir) / "checkpoints", Path(run_dir) / "pool"):
        if not runlock.plain_directory(parent):
            continue
        for backup in sorted(parent.glob(f"{REPLACED_PREFIX}*")):
            if re.fullmatch(REPLACED_PREFIX + runlock.GENERATION, backup.name) and _valid_generation(backup):
                target = _restore_retired_generation(backup)
                if target is not None:
                    restored.append({"path": str(target), "reason": "restored retired checkpoint after interrupted replacement"})
    return restored


def cleanup_checkpoint_temps(run_dir: str | Path) -> list[dict[str, str]]:
    """Remove unfinished checkpoint generations before restarting a sole writer.

    A backup created while replacing an existing iteration (``.tmp_replaced_<name>_*``) is restored
    first when the committed directory is missing or incomplete, so a crash between the two renames
    never loses the last valid checkpoint.
    """
    with runlock.run_lock(run_dir) as run_dir:
        skipped = recover_retired_checkpoints(run_dir)
        for parent in (Path(run_dir) / "checkpoints", Path(run_dir) / "pool"):
            skipped += runlock.cleanup_temps(parent, directories=(r"\.tmp_" + runlock.GENERATION,))
            if not runlock.plain_directory(parent):
                continue
            for backup in sorted(parent.glob(f"{REPLACED_PREFIX}*")):
                if (re.fullmatch(REPLACED_PREFIX + runlock.GENERATION, backup.name)
                        and runlock.plain_directory(backup)
                        and _valid_generation(parent / backup.name[len(REPLACED_PREFIX):],
                                              require_training=(backup / TRAINER_STATE_FILE).is_file())):
                    shutil.rmtree(backup)
                    skipped.append({"path": str(backup), "reason": "removed recovered checkpoint backup"})
        skipped += runlock.cleanup_training_files(run_dir)
        # Practical smoke engines are retained dependencies, not disposable
        # mkdtemp scratch. Clean their writers under their own run locks only.
        for child in Path(run_dir).iterdir():
            if re.fullmatch(r"smoke-engine-" + runlock.RANDOM_SUFFIX, child.name) and runlock.plain_directory(child):
                skipped += cleanup_checkpoint_temps(child)
        return skipped


def _check_training_state(loaded: LoadedCheckpoint) -> None:
    state = loaded.trainer_state
    if not isinstance(state, dict) or not isinstance(loaded.learner_state, dict):
        raise ManifestError("checkpoint lacks trainer or learner state")
    if type(state.get("iteration")) is not int or state["iteration"] != loaded.manifest.iteration:
        raise ManifestError("checkpoint iteration does not match manifest")
    if "replay_snapshot_sha256" in state:
        replay = loaded.directory / "replay_snapshot.npz"
        if replay.is_symlink() or sha256_file(replay) != state["replay_snapshot_sha256"]:
            raise ManifestError("checkpoint replay sha256 mismatch")


def find_resume_checkpoint(
    path: str | Path, *, cleanup: bool = True,
) -> tuple[LoadedCheckpoint, list[dict[str, str]]]:
    """Try the selected checkpoint followed by older complete iterations."""
    if cleanup:
        with runlock.run_lock(resume_run_dir(path)):
            skipped = cleanup_checkpoint_temps(resume_run_dir(path))
            loaded, rejected = find_resume_checkpoint(path, cleanup=False)
            return loaded, skipped + rejected
    base = runlock.writer_directory(Path(path).expanduser())
    root = resume_run_dir(base)
    checkpoints = root / "checkpoints"
    skipped = []
    candidates: list[Path] = []
    try:
        candidates.append(resolve_checkpoint_dir(base))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        skipped.append({"path": str(base), "reason": str(exc)})
    # An explicit iteration limits recovery to that iteration and its predecessors.
    selected = base if base.parent == checkpoints else None
    ceiling = None
    if selected is not None and selected.name.startswith("iter_"):
        try:
            ceiling = int(selected.name[5:])
        except ValueError:
            pass
    older = []
    for directory in checkpoints.glob("iter_*"):
        if runlock.plain_directory(directory) and directory.name[5:].isdigit():
            number = int(directory.name[5:])
            if ceiling is None or number <= ceiling:
                older.append((number, directory))
    # A published directory is the commit point. The latest pointer may still
    # refer to its predecessor if the writer died before updating metadata.
    ordered = [directory for _, directory in sorted(older, reverse=True)]
    candidates = ([base] if selected is not None else []) + ordered + candidates
    candidates = list(dict.fromkeys(candidates))
    for candidate in candidates:
        try:
            loaded = load_checkpoint(candidate, "cpu")
            _check_training_state(loaded)
            return loaded, skipped
        except Exception as exc:
            # Corrupt JSON, torch archives, and incompatible tensors have different
            # exception types. Recovery intentionally handles all load failures.
            skipped.append({"path": str(candidate), "reason": f"{type(exc).__name__}: {exc}"})
    raise ManifestError(f"no valid training checkpoint at {base}; skipped: {skipped}")
