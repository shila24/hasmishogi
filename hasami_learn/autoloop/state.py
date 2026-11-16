"""Durable JSON, packaged schema checks, and kernel-owned instance locks."""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from ..runlock import FILE_LOCK, cleanup_json_files, plain_directory, run_lock, writer_path


class LoopError(ValueError):
    """A configuration or persisted contract is invalid."""


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(LoopError(value)))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fsync_dir(path):
    path = writer_path(path)
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, data):
    # Directory-inode locks leave checkpoint inventories unchanged and allow
    # independent short JSON writes to queue instead of failing under load.
    path = writer_path(path, allow_best_alias=True)
    with run_lock(path.parent, FILE_LOCK, blocking=True):
        return _atomic_json(path, data)


def _atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".tmp_{path.name}_", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        fsync_dir(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


def cleanup_json_temps(directory, names):
    """Clean legacy and named JSON staging under the JSON writer's lock."""
    return cleanup_json_files(directory, names, legacy=True)


def validate_schema(data, schema, location="$"):
    """Evaluate the small JSON Schema vocabulary used by our packaged schemas.

    Keeping this evaluator local avoids importing the torch-dependent package
    validators. The schemas are also checked with jsonschema in tests.
    """
    types = {"object": dict, "array": list, "string": str, "integer": int,
             "number": (int, float), "boolean": bool, "null": type(None)}
    kind = schema.get("type")
    if kind:
        kinds = kind if isinstance(kind, list) else [kind]
        if not any(isinstance(data, types[k]) and not (k in ("number", "integer") and isinstance(data, bool)) for k in kinds):
            raise LoopError(f"{location}: expected {kind}")
    if "enum" in schema and data not in schema["enum"]:
        raise LoopError(f"{location}: invalid value {data!r}")
    if "const" in schema and (data != schema["const"] or
                              isinstance(data, bool) != isinstance(schema["const"], bool)):
        raise LoopError(f"{location}: expected {schema['const']!r}")
    if "not" in schema:
        try:
            validate_schema(data, schema["not"], location)
        except LoopError:
            pass
        else:
            raise LoopError(f"{location}: disallowed value")
    if isinstance(data, dict):
        if len(data) < schema.get("minProperties", 0):
            raise LoopError(f"{location}: too few properties")
        for key in schema.get("required", []):
            if key not in data:
                raise LoopError(f"{location}: missing {key}")
        props = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for key, value in data.items():
            if "propertyNames" in schema:
                validate_schema(key, schema["propertyNames"], location)
            if key in props:
                validate_schema(value, props[key], f"{location}.{key}")
            elif extra is False:
                raise LoopError(f"{location}: unknown field {key}")
            elif isinstance(extra, dict):
                validate_schema(value, extra, f"{location}.{key}")
    if isinstance(data, list):
        if len(data) < schema.get("minItems", 0) or len(data) > schema.get("maxItems", math.inf):
            raise LoopError(f"{location}: invalid array length")
        for i, value in enumerate(data):
            validate_schema(value, schema.get("items", {}), f"{location}[{i}]")
    if isinstance(data, str):
        if len(data) < schema.get("minLength", 0) or ("pattern" in schema and re.search(schema["pattern"], data) is None):
            raise LoopError(f"{location}: invalid string")
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        if not math.isfinite(data) or data < schema.get("minimum", -math.inf) or data > schema.get("maximum", math.inf):
            raise LoopError(f"{location}: number out of range")
        if "exclusiveMinimum" in schema and data <= schema["exclusiveMinimum"]:
            raise LoopError(f"{location}: number must be positive")


def validate(data, name):
    schema = read_json(Path(__file__).resolve().parents[1] / "schemas" / f"autoloop-{name}.schema.json")
    validate_schema(data, schema)


class StateStore:
    def __init__(self, directory, event=lambda *args, **kwargs: None):
        self.directory = writer_path(directory)
        self.path = self.directory / "loop_state.json"
        self.backup = self.path.with_suffix(".json.bak")
        self.event = event
        self.lock = None

    def acquire(self):
        self.directory = writer_path(self.directory)
        path = writer_path(self.directory / "loop.lock")
        self.directory.mkdir(parents=True, exist_ok=True)
        if not plain_directory(self.directory):
            raise LoopError("loop directory must not contain symlinks")
        if path.is_symlink():
            raise LoopError("loop lock must not be a symlink")
        fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        self.lock = os.fdopen(fd, "a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            self.lock = None
            return False
        return True

    def close(self):
        if self.lock:
            # Do not unlock explicitly: a supervising child may still hold this
            # open description after its controller dies.
            self.lock.close()
            self.lock = None

    def load(self):
        if not self.path.exists() and not self.backup.exists():
            return None
        for path in (self.path, self.backup):
            try:
                state = read_json(path)
                validate(state, "state")
                if path == self.backup:
                    self.event("state_recovered", source=str(path))
                return state
            except (OSError, ValueError, TypeError) as exc:
                self.event("state_invalid", path=str(path), reason=str(exc))
        raise LoopError("state and backup are both invalid; refusing a fresh start")

    def save(self, state):
        self.path = writer_path(self.path)
        self.backup = writer_path(self.backup)
        validate(state, "state")
        if self.path.exists():
            try:
                previous = read_json(self.path)
                validate(previous, "state")
            except (OSError, ValueError, TypeError):
                pass  # Preserve the last valid backup when recovering a corrupt primary.
            else:
                atomic_json(self.backup, previous)
        atomic_json(self.path, state)
