"""Compare the dependency-free validator with Draft 2020-12 on packaged contracts."""
import copy
import json
import unittest
from unittest import mock
from hasami_learn import schema_validation as validation
from tests.learn._common import requires_torch


def example(schema):
    if schema is True:
        return None
    if "const" in schema:
        return copy.deepcopy(schema["const"])
    if "enum" in schema:
        return copy.deepcopy(schema["enum"][0])
    kind = schema.get("type", "object")
    if isinstance(kind, list):
        kind = kind[0]
    if kind == "object":
        result = {key: example(schema.get("properties", {}).get(key, {})) for key in schema.get("required", [])}
        while len(result) < schema.get("minProperties", 0):
            result["item" + str(len(result))] = example(schema.get("additionalProperties", {}))
        return result
    if kind == "array":
        return [example(schema.get("items", {})) for _ in range(schema.get("minItems", 0))]
    if kind == "string":
        pattern = schema.get("pattern", "")
        if "{64}" in pattern:
            return "a" * 64
        if "cpu|cuda" in pattern:
            return "cpu"
        return "a" * max(1, schema.get("minLength", 0))
    if kind in ("integer", "number"):
        return max(schema.get("minimum", 0), schema.get("exclusiveMinimum", -1) + 1)
    return {"boolean": False, "null": None}[kind]


class SchemaSubsetTests(unittest.TestCase):
    def test_all_packaged_schemas_accept_constructed_examples_without_dependencies(self):
        for path in validation.SCHEMA_DIR.glob("*.json"):
            with self.subTest(schema=path.name):
                schema = json.loads(path.read_text())
                self.assertEqual(validation.validate_subset(schema, example(schema)), [])
                self.assertTrue(validation.validate_subset(schema, None))

    def test_missing_and_old_jsonschema_use_fallback(self):
        for module in (None, object()):
            with mock.patch.dict("sys.modules", {"jsonschema": module}):
                self.assertEqual(validation.validate({"type": "integer"}, 2), [])
                self.assertTrue(validation.validate({"type": "integer"}, True))

    def test_unsupported_vocabulary_fails_even_in_unvisited_branch(self):
        with self.assertRaisesRegex(ValueError, "Unsupported schema"):
            validation.validate_subset({"properties": {"absent": {"$ref": "remote"}}}, {})
        self.assertTrue(validation.validate_subset({"enum": [1]}, True))
        self.assertEqual(validation.validate_subset({"type": "integer"}, 1.0), [])
        self.assertTrue(validation.validate_subset({"exclusiveMinimum": 0}, 0))

    @requires_torch
    def test_draft202012_parity_every_schema_and_invalid_examples(self):
        from jsonschema import Draft202012Validator
        comparisons = 0

        def compare(schema):
            nonlocal comparisons
            validator = Draft202012Validator(schema)
            sample = example(schema)
            values = [sample, None, True, False, -1, 0, 1, 1.0, 1.5, "", "bad!", "a" * 64, [], {}, [None], {"extra": True}]
            if isinstance(sample, dict):
                values += [{key: val for key, val in sample.items() if key != missing} for missing in sample]
            for value in values:
                self.assertEqual(not validation.validate_subset(schema, value), validator.is_valid(value), (schema, value))
                comparisons += 1
            if isinstance(schema, dict):
                for key in ("properties", "$defs"):
                    for child in schema.get(key, {}).values():
                        compare(child)
                for key in ("items", "additionalProperties", "not", "propertyNames"):
                    if isinstance(schema.get(key), dict):
                        compare(schema[key])

        for path in validation.SCHEMA_DIR.glob("*.json"):
            with self.subTest(schema=path.name):
                compare(json.loads(path.read_text()))
        self.assertGreater(comparisons, 5000)
