"""Tests for the plugin package model (nomorals/plugins)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nomorals.plugins import (
    AlreadyInstalled,
    KNOWN_PERMISSIONS,
    LoadedPlugin,
    ManifestError,
    NotInstalled,
    PermissionDenied,
    PluginCapabilities,
    PluginManifest,
    PluginRegistry,
    load_manifest,
    load_manifest_file,
    load_plugin,
)
from nomorals.plugins.errors import LoadError, PluginError


def _plugin_dir(base: Path, name: str = "demo", version: str = "1.0.0",
               permissions: list[str] | None = None,
               code: str = "") -> Path:
    d = base / f"{name}-{version}-src"
    d.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": name,
        "version": version,
        "description": "demo plugin",
        "author": "test",
        "entry_points": {"main": "demo:run"},
        "permissions": permissions or [],
    }
    (d / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    (d / "demo.py").write_text(
        code or "def run(caps):\n    return {'ok': True, 'perms': sorted(caps.granted)}\n",
        encoding="utf-8")
    return d


class ManifestTests(unittest.TestCase):
    def test_valid(self):
        m = load_manifest({
            "name": "demo", "version": "1.0.0",
            "entry_points": {"main": "demo:run"},
            "permissions": ["artifacts.read"],
        })
        self.assertEqual(m.name, "demo")
        self.assertIn("artifacts.read", m.permissions)

    def test_bad_name(self):
        with self.assertRaises(ManifestError):
            load_manifest({"name": "Bad Name!", "version": "1.0.0",
                           "entry_points": {"main": "m:f"}})

    def test_bad_version(self):
        with self.assertRaises(ManifestError):
            load_manifest({"name": "demo", "version": "v1",
                           "entry_points": {"main": "m:f"}})

    def test_missing_entry_points(self):
        with self.assertRaises(ManifestError):
            load_manifest({"name": "demo", "version": "1.0.0"})

    def test_bad_entry_point_shape(self):
        with self.assertRaises(ManifestError):
            load_manifest({"name": "demo", "version": "1.0.0",
                           "entry_points": {"main": "no-colon"}})

    def test_unknown_permission_rejected(self):
        with self.assertRaises(ManifestError):
            load_manifest({"name": "demo", "version": "1.0.0",
                           "entry_points": {"main": "m:f"},
                           "permissions": ["root.access"]})

    def test_missing_manifest_file(self):
        with self.assertRaises(ManifestError):
            load_manifest_file("/nonexistent-xyz")


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        from nomorals.storage.db import Database
        self.db = Database(str(Path(self.tmp.name) / "t.db"))
        self.reg = PluginRegistry(self.db, Path(self.tmp.name) / "plugins")

    def tearDown(self):
        self.tmp.cleanup()

    def test_install_list_get(self):
        src = _plugin_dir(Path(self.tmp.name))
        p = self.reg.install(src)
        self.assertEqual(p.name, "demo")
        self.assertTrue(p.enabled)
        self.assertEqual(len(self.reg.list()), 1)
        self.assertEqual(self.reg.get("demo").version, "1.0.0")

    def test_install_duplicate_raises(self):
        src = _plugin_dir(Path(self.tmp.name))
        self.reg.install(src)
        with self.assertRaises(AlreadyInstalled):
            self.reg.install(src)

    def test_install_bad_manifest_raises(self):
        bad = Path(self.tmp.name) / "bad"
        bad.mkdir()
        (bad / "plugin.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ManifestError):
            self.reg.install(bad)

    def test_enable_disable(self):
        src = _plugin_dir(Path(self.tmp.name))
        self.reg.install(src)
        self.reg.disable("demo")
        self.assertFalse(self.reg.get("demo").enabled)
        self.assertEqual(self.reg.list(enabled_only=True), [])
        self.reg.enable("demo")
        self.assertTrue(self.reg.get("demo").enabled)

    def test_remove(self):
        src = _plugin_dir(Path(self.tmp.name))
        self.reg.install(src)
        self.reg.remove("demo")
        self.assertEqual(self.reg.list(), [])
        with self.assertRaises(NotInstalled):
            self.reg.get("demo")

    def test_remove_missing_raises(self):
        with self.assertRaises(NotInstalled):
            self.reg.remove("nope")

    def test_get_missing_raises(self):
        with self.assertRaises(NotInstalled):
            self.reg.get("nope")


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_and_run(self):
        src = _plugin_dir(Path(self.tmp.name),
                          permissions=["artifacts.read"])
        manifest = load_manifest_file(src)
        loaded = load_plugin(manifest, src)
        self.assertIsInstance(loaded, LoadedPlugin)
        caps = PluginCapabilities(granted=frozenset(["artifacts.read"]))
        result = loaded.entry("main", caps)
        self.assertEqual(result["ok"], True)
        self.assertIn("artifacts.read", result["perms"])

    def test_missing_entry_point_raises(self):
        src = _plugin_dir(Path(self.tmp.name))
        manifest = load_manifest_file(src)
        loaded = load_plugin(manifest, src)
        caps = PluginCapabilities(granted=frozenset())
        with self.assertRaises(LoadError):
            loaded.entry("nope", caps)

    def test_broken_module_raises(self):
        src = _plugin_dir(Path(self.tmp.name), code="raise RuntimeError('boom')\n")
        # fix the entry code: module-level raise
        (src / "demo.py").write_text("raise RuntimeError('boom')\n",
                                     encoding="utf-8")
        manifest = load_manifest_file(src)
        with self.assertRaises(LoadError):
            load_plugin(manifest, src)

    def test_no_modules_raises(self):
        empty = Path(self.tmp.name) / "empty"
        empty.mkdir()
        (empty / "plugin.json").write_text(json.dumps({
            "name": "empty", "version": "1.0.0",
            "entry_points": {"main": "m:f"}}), encoding="utf-8")
        manifest = load_manifest_file(empty)
        with self.assertRaises(LoadError):
            load_plugin(manifest, empty)


class CapabilitiesTests(unittest.TestCase):
    def test_granted_ok(self):
        caps = PluginCapabilities(granted=frozenset(["network.fetch"]),
                                  fetcher=lambda url, timeout=30: b"data")
        self.assertEqual(caps.fetch("http://x"), b"data")

    def test_denied_raises(self):
        caps = PluginCapabilities(granted=frozenset())
        with self.assertRaises(PermissionDenied):
            caps.fetch("http://x")

    def test_unwired_capability_raises(self):
        caps = PluginCapabilities(granted=frozenset(["network.fetch"]))
        with self.assertRaises(PermissionDenied):
            caps.fetch("http://x")

    def test_artifacts_write_needs_permission(self):
        caps = PluginCapabilities(granted=frozenset(["artifacts.read"]))
        with self.assertRaises(PermissionDenied):
            caps.artifacts().put(b"x")

    def test_kv_needs_permission(self):
        caps = PluginCapabilities(granted=frozenset())
        with self.assertRaises(PermissionDenied):
            caps.kv()


if __name__ == "__main__":
    unittest.main()
