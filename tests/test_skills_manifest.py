"""Tests for nomorals/skills/manifest.py — validation with reasons."""

from __future__ import annotations

import unittest

from nomorals.skills.manifest import (
    ManifestError,
    SkillManifest,
    WiringError,
    resolve_expression,
    validate_schema,
)


def _good_manifest(**overrides):
    data = {
        "name": "shout_wrap",
        "version": "1.0.0",
        "tools": ["shout", "wrap"],
        "input_schema": {"text": "str", "tag": "str?"},
        "output_schema": {"wrapped": "str"},
        "wiring": [{"text": "$input.text"},
                   {"shouted": "$0.shouted", "tag": "$input.tag"}],
        "owner": "devon",
        "description": "shout then wrap",
    }
    data.update(overrides)
    return data


class ManifestValidationTests(unittest.TestCase):
    def test_good_manifest_validates_clean(self):
        manifest = SkillManifest.from_dict(_good_manifest())
        self.assertEqual(manifest.validate(), [])
        self.assertEqual(manifest.name, "shout_wrap")
        self.assertEqual(manifest.version, "1.0.0")
        self.assertEqual(manifest.tools, ["shout", "wrap"])

    def test_bad_name_rejected_with_reason(self):
        for bad in ("", "has space", "-leading-dash", "x" * 65, 42):
            manifest = SkillManifest(name=bad, tools=["a"])  # type: ignore[arg-type]
            errors = manifest.validate()
            self.assertTrue(any("name" in e for e in errors),
                            f"name {bad!r} should be rejected: {errors}")

    def test_bad_version_rejected_with_reason(self):
        for bad in ("", "v1", "1", "one.two.three", "1.0.0.0.0"):
            manifest = SkillManifest(name="ok", version=bad, tools=["a"])
            errors = manifest.validate()
            self.assertTrue(any("version" in e for e in errors),
                            f"version {bad!r} should be rejected: {errors}")

    def test_semver_ish_variants_accepted(self):
        for good in ("1.0", "1.0.0", "2.1.3-rc1", "10.20.30"):
            manifest = SkillManifest(name="ok", version=good, tools=["a"])
            self.assertEqual(manifest.validate(), [],
                             f"version {good!r} should be accepted")

    def test_empty_tools_rejected(self):
        for bad_tools in ([], "", "shout"):
            manifest = SkillManifest(name="ok", tools=bad_tools)  # type: ignore[arg-type]
            errors = manifest.validate()
            self.assertTrue(any("tools" in e for e in errors),
                            f"tools {bad_tools!r} should be rejected")

    def test_blank_tool_name_rejected(self):
        manifest = SkillManifest(name="ok", tools=["shout", "  "])
        errors = manifest.validate()
        self.assertTrue(any("tools[1]" in e for e in errors), errors)

    def test_unknown_schema_type_rejected(self):
        manifest = SkillManifest(name="ok", tools=["a"],
                                 input_schema={"x": "strnig"})
        errors = manifest.validate()
        self.assertTrue(any("unknown type" in e for e in errors), errors)

    def test_bad_wiring_reference_rejected(self):
        manifest = SkillManifest(
            name="ok", tools=["a", "b"],
            wiring=[{}, {"x": "$5.y"}])
        errors = manifest.validate()
        self.assertTrue(any("has not run yet" in e for e in errors), errors)

    def test_last_at_step_zero_rejected(self):
        manifest = SkillManifest(
            name="ok", tools=["a"], wiring=[{"x": "$last.y"}])
        errors = manifest.validate()
        self.assertTrue(any("$last" in e for e in errors), errors)

    def test_wiring_longer_than_tools_rejected(self):
        manifest = SkillManifest(name="ok", tools=["a"],
                                 wiring=[{}, {}, {}])
        errors = manifest.validate()
        self.assertTrue(any("wiring" in e for e in errors), errors)

    def test_from_dict_raises_manifest_error_with_reasons(self):
        with self.assertRaises(ManifestError) as ctx:
            SkillManifest.from_dict({"name": "", "tools": []})
        self.assertGreater(len(ctx.exception.errors), 1)
        self.assertIn("name", " ".join(ctx.exception.errors))

    def test_from_dict_rejects_non_dict(self):
        with self.assertRaises(ManifestError):
            SkillManifest.from_dict(["not", "a", "dict"])  # type: ignore[arg-type]

    def test_round_trip(self):
        manifest = SkillManifest.from_dict(_good_manifest())
        clone = SkillManifest.from_dict(manifest.to_dict())
        self.assertEqual(clone.to_dict(), manifest.to_dict())


class SchemaValidationTests(unittest.TestCase):
    def test_required_and_optional_keys(self):
        schema = {"text": "str", "n": "int?", "tags": {"type": "list",
                                                       "required": False}}
        self.assertEqual(validate_schema(schema, {"text": "a"}), [])
        problems = validate_schema(schema, {"n": 3})
        self.assertTrue(any("missing required" in p and "'text'" in p
                            for p in problems), problems)

    def test_type_mismatches_named(self):
        schema = {"n": "int", "flag": "bool", "payload": "dict"}
        problems = validate_schema(schema, {"n": "3", "flag": 1,
                                            "payload": []})
        self.assertEqual(len(problems), 3)
        self.assertTrue(any("'n'" in p and "int" in p for p in problems))
        # bool is not an int here
        self.assertTrue(any("'flag'" in p for p in problems))

    def test_bool_not_accepted_as_int(self):
        problems = validate_schema({"n": "int"}, {"n": True})
        self.assertTrue(any("expected int, got bool" in p for p in problems),
                        problems)

    def test_non_dict_data_is_one_problem(self):
        problems = validate_schema({"a": "str"}, ["not", "a", "dict"])
        self.assertEqual(len(problems), 1)
        self.assertIn("expected an object", problems[0])

    def test_dict_style_spec_with_default_is_optional(self):
        schema = {"mode": {"type": "str", "default": "fast"}}
        self.assertEqual(validate_schema(schema, {}), [])
        problems = validate_schema(schema, {"mode": 5})
        self.assertTrue(any("'mode'" in p for p in problems))


class WiringResolutionTests(unittest.TestCase):
    def test_input_reference(self):
        value = resolve_expression("$input.user.name",
                                   skill_input={"user": {"name": "ada"}},
                                   step_outputs=[], step_index=0)
        self.assertEqual(value, "ada")

    def test_step_index_reference(self):
        value = resolve_expression(
            "$0.summary", skill_input={},
            step_outputs=[{"summary": "done"}], step_index=1)
        self.assertEqual(value, "done")

    def test_last_reference(self):
        value = resolve_expression(
            "$last.items[0]", skill_input={},
            step_outputs=[{"items": ["a", "b"]}], step_index=1)
        self.assertEqual(value, "a")

    def test_forward_reference_rejected_at_resolve_time(self):
        with self.assertRaises(WiringError) as ctx:
            resolve_expression("$2.x", skill_input={},
                               step_outputs=[{}, {}], step_index=1)
        self.assertIn("has not run yet", str(ctx.exception))

    def test_missing_key_rejected_with_concrete_message(self):
        with self.assertRaises(WiringError) as ctx:
            resolve_expression("$0.nope", skill_input={},
                               step_outputs=[{"yes": 1}], step_index=1)
        self.assertIn("'nope'", str(ctx.exception))

    def test_malformed_expression_rejected(self):
        with self.assertRaises(WiringError):
            resolve_expression("$bogus!!!", skill_input={},
                               step_outputs=[], step_index=0)

    def test_literals_pass_through(self):
        for literal in (42, "plain", None, True, [1], {"a": 1}):
            self.assertEqual(
                resolve_expression(literal, skill_input={},
                                   step_outputs=[], step_index=0),
                literal)


if __name__ == "__main__":
    unittest.main()
