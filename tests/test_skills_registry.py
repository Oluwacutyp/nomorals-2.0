"""Tests for nomorals/skills/registry.py — persisted, versioned registry."""

from __future__ import annotations

import unittest

from nomorals.core.errors import NotFound
from nomorals.skills.manifest import ManifestError, SkillManifest
from nomorals.skills.registry import SkillRegistry
from nomorals.storage.db import Database


def _manifest(name="demo", version="1.0.0", **overrides):
    data = {
        "name": name,
        "version": version,
        "tools": ["tool_a", "tool_b"],
        "input_schema": {"text": "str"},
        "output_schema": {"done": "bool"},
        "description": f"{name} skill",
    }
    data.update(overrides)
    return SkillManifest.from_dict(data)


class RegistryInstallTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.registry = SkillRegistry(self.db)

    def test_install_validates_and_returns_installed(self):
        installed = self.registry.install(_manifest())
        self.assertEqual(installed.name, "demo")
        self.assertEqual(installed.version, "1.0.0")
        self.assertTrue(installed.enabled)
        self.assertTrue(installed.active)

    def test_install_rejects_invalid_manifest_with_reasons(self):
        with self.assertRaises(ManifestError) as ctx:
            self.registry.install({"name": "", "tools": []})
        self.assertTrue(ctx.exception.errors)

    def test_install_accepts_raw_dict(self):
        installed = self.registry.install({
            "name": "raw", "version": "0.1", "tools": ["t"]})
        self.assertEqual(installed.name, "raw")

    def test_first_version_becomes_active_pin(self):
        self.registry.install(_manifest(version="1.0.0"))
        resolved = self.registry.get("demo")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.version, "1.0.0")  # type: ignore[union-attr]

    def test_second_version_does_not_steal_pin(self):
        self.registry.install(_manifest(version="1.0.0"))
        self.registry.install(_manifest(version="2.0.0"))
        resolved = self.registry.get("demo")
        self.assertEqual(resolved.version, "1.0.0")  # type: ignore[union-attr]

    def test_pin_promotes_and_rollback_restores(self):
        self.registry.install(_manifest(version="1.0.0"))
        self.registry.install(_manifest(version="2.0.0"))
        pinned = self.registry.pin("demo", "2.0.0")
        self.assertTrue(pinned.active)
        self.assertEqual(self.registry.get("demo").version, "2.0.0")  # type: ignore[union-attr]
        # rollback to the older version
        rolled = self.registry.pin("demo", "1.0.0")
        self.assertTrue(rolled.active)
        self.assertEqual(self.registry.get("demo").version, "1.0.0")  # type: ignore[union-attr]

    def test_pin_unknown_version_raises_not_found(self):
        self.registry.install(_manifest(version="1.0.0"))
        with self.assertRaises(NotFound):
            self.registry.pin("demo", "9.9.9")

    def test_get_specific_version(self):
        self.registry.install(_manifest(version="1.0.0"))
        self.registry.install(_manifest(version="2.0.0"))
        got = self.registry.get("demo", "2.0.0")
        self.assertIsNotNone(got)
        self.assertEqual(got.version, "2.0.0")  # type: ignore[union-attr]

    def test_get_unknown_returns_none(self):
        self.assertIsNone(self.registry.get("nope"))

    def test_reinstall_keeps_pin_state(self):
        self.registry.install(_manifest(version="1.0.0"))
        self.registry.install(_manifest(version="2.0.0"))
        self.registry.pin("demo", "2.0.0")
        # re-installing the pinned version must not drop the pin
        self.registry.install(_manifest(version="2.0.0"))
        self.assertEqual(self.registry.get("demo").version, "2.0.0")  # type: ignore[union-attr]
        # re-installing the unpinned version must not steal it either
        self.registry.install(_manifest(version="1.0.0"))
        self.assertEqual(self.registry.get("demo").version, "2.0.0")  # type: ignore[union-attr]

    def test_versions_lists_all(self):
        self.registry.install(_manifest(version="1.0.0"))
        self.registry.install(_manifest(version="2.0.0"))
        self.assertEqual(self.registry.versions("demo"), ["1.0.0", "2.0.0"])


class RegistryEnableDisableTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.registry = SkillRegistry(self.db)
        self.registry.install(_manifest())

    def test_disable_and_enable(self):
        self.assertTrue(self.registry.disable("demo"))
        self.assertFalse(self.registry.is_enabled("demo"))
        self.assertTrue(self.registry.enable("demo"))
        self.assertTrue(self.registry.is_enabled("demo"))

    def test_enable_unknown_returns_false(self):
        self.assertFalse(self.registry.enable("ghost"))
        self.assertFalse(self.registry.disable("ghost"))

    def test_list_reports_state(self):
        self.registry.install(_manifest(name="other"))
        items = {i["name"]: i for i in self.registry.list()}
        self.assertEqual(set(items), {"demo", "other"})
        self.assertEqual(items["demo"]["active_version"], "1.0.0")
        self.assertEqual(items["demo"]["versions"], ["1.0.0"])
        self.assertTrue(items["demo"]["enabled"])
        self.assertEqual(items["demo"]["tools"], ["tool_a", "tool_b"])
        self.registry.disable("demo")
        items = {i["name"]: i for i in self.registry.list()}
        self.assertFalse(items["demo"]["enabled"])


class RegistryPersistenceTests(unittest.TestCase):
    def test_registry_survives_reconstruction(self):
        db = Database(":memory:")
        SkillRegistry(db).install(_manifest(version="3.2.1"))
        # a fresh registry over the same db sees the install
        fresh = SkillRegistry(db)
        resolved = fresh.get("demo")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.version, "3.2.1")  # type: ignore[union-attr]
        self.assertEqual(resolved.manifest.tools,  # type: ignore[union-attr]
                         ["tool_a", "tool_b"])


if __name__ == "__main__":
    unittest.main()
