"""Validate packaged JSON contracts, with a dependency-free subset fallback.

The fallback supports exactly the vocabulary used by the packaged schemas and
rejects unknown keywords. It does not resolve remote references or access files.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any

SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"
ANNOTATIONS = {"$schema", "$id", "title", "description", "default"}
KEYWORDS = ANNOTATIONS | {
    "$defs", "type", "const", "enum", "not", "properties", "required",
    "additionalProperties", "propertyNames", "minProperties", "items",
    "minItems", "maxItems", "minimum", "maximum", "exclusiveMinimum",
    "minLength", "pattern",
}


@lru_cache(maxsize=None)
def _load(name: str) -> str:
    return (SCHEMA_DIR / name).read_text(encoding="utf-8")


def load_schema(name: str) -> dict[str, Any]:
    """Return a fresh copy of a packaged schema."""
    return json.loads(_load(name))


def check_subset(schema: Any) -> None:
    """Reject unsupported validation even inside currently inactive branches."""
    if isinstance(schema, bool):
        return
    if not isinstance(schema, dict):
        raise ValueError("Schema must be an object or boolean")
    unknown = schema.keys() - KEYWORDS
    if unknown:
        raise ValueError("Unsupported schema keywords: " + ", ".join(sorted(unknown)))
    for key in ("properties", "$defs"):
        for child in schema.get(key, {}).values():
            check_subset(child)
    for key in ("items", "additionalProperties", "propertyNames", "not"):
        if key in schema:
            check_subset(schema[key])
    kinds = schema.get("type", [])
    if isinstance(kinds, str):
        kinds = [kinds]
    if set(kinds) - {"object", "array", "string", "integer", "number", "boolean", "null"}:
        raise ValueError("Unsupported schema type")


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equal(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def validate_subset(schema: Any, data: Any) -> list[str]:
    """Validate the packaged vocabulary; error wording is not a stable API."""
    check_subset(schema)

    def visit(spec: Any, value: Any, path: str) -> list[str]:
        if isinstance(spec, bool):
            return [] if spec else [f"{path}: forbidden value"]
        errors = []

        def error(message):
            errors.append(f"{path}: {message}")

        types = {
            "object": isinstance(value, dict), "array": isinstance(value, list),
            "string": isinstance(value, str), "boolean": isinstance(value, bool),
            "null": value is None,
            "integer": type(value) is int or (type(value) is float and value.is_integer()),
            "number": type(value) in (int, float),
        }
        kinds = spec.get("type", [])
        if isinstance(kinds, str):
            kinds = [kinds]
        if kinds and not any(types[kind] for kind in kinds):
            error("type mismatch")
            return errors
        if "const" in spec and not _equal(value, spec["const"]):
            error("constant mismatch")
        if "enum" in spec and not any(_equal(value, item) for item in spec["enum"]):
            error("not in enum")
        if "not" in spec and not visit(spec["not"], value, path):
            error("matches forbidden schema")
        if isinstance(value, dict):
            for key in spec.get("required", []):
                if key not in value:
                    error("missing required property " + key)
            if len(value) < spec.get("minProperties", 0):
                error("too few properties")
            properties = spec.get("properties", {})
            for key, item in value.items():
                if "propertyNames" in spec:
                    errors.extend(visit(spec["propertyNames"], key, path))
                child = properties.get(key, spec.get("additionalProperties", True))
                errors.extend(visit(child, item, path + "/" + key))
        if isinstance(value, list):
            if len(value) < spec.get("minItems", 0) or len(value) > spec.get("maxItems", float("inf")):
                error("array length out of range")
            for index, item in enumerate(value):
                errors.extend(visit(spec.get("items", True), item, f"{path}/{index}"))
        if isinstance(value, str):
            if len(value) < spec.get("minLength", 0):
                error("string too short")
            if "pattern" in spec and re.search(spec["pattern"], value) is None:
                error("pattern mismatch")
        if type(value) in (int, float):
            if "minimum" in spec and value < spec["minimum"]:
                error("below minimum")
            if "maximum" in spec and value > spec["maximum"]:
                error("above maximum")
            if "exclusiveMinimum" in spec and value <= spec["exclusiveMinimum"]:
                error("not above exclusive minimum")
        return errors

    return visit(schema, data, "$")


def validate(schema: dict[str, Any], data: Any) -> list[str]:
    """Return errors, using Draft 2020-12 when available, otherwise the local subset."""
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        return validate_subset(schema, data)
    return [f"{'/'.join(str(p) for p in e.absolute_path) or '$'}: {e.message}"
            for e in Draft202012Validator(schema).iter_errors(data)]
