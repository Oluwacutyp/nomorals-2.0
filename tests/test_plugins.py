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


class ZipInstallTests(unittest.TestCase):
    """Zip installs were broken (R21): the manifest was validated against
    the zip *path* before unpacking, so every zip install failed."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        from nomorals.storage.db import Database
        self.db = Database(str(Path(self.tmp.name) / "t.db"))
        self.reg = PluginRegistry(self.db, Path(self.tmp.name) / "plugins")

    def tearDown(self):
        self.tmp.cleanup()

    def _zip_of(self, src: Path, name: str = "p.zip") -> Path:
        import zipfile
        zp = Path(self.tmp.name) / name
        with zipfile.ZipFile(zp, "w") as zf:
            for p in src.rglob("*"):
                zf.write(p, p.relative_to(src))
        return zp

    def test_install_from_zip(self):
        src = _plugin_dir(Path(self.tmp.name))
        zp = self._zip_of(src)
        p = self.reg.install(zp)
        self.assertEqual(p.name, "demo")
        self.assertTrue((Path(p.path) / "plugin.json").is_file())

    def test_install_from_zip_with_wrapper_dir(self):
        import zipfile
        src = _plugin_dir(Path(self.tmp.name))
        zp = Path(self.tmp.name) / "wrapped.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            for p in src.rglob("*"):
                zf.write(p, Path("wrapper") / p.relative_to(src))
        p = self.reg.install(zp)
        self.assertEqual(p.name, "demo")
        self.assertTrue((Path(p.path) / "plugin.json").is_file())

    def test_zip_slip_rejected(self):
        import zipfile
        src = _plugin_dir(Path(self.tmp.name))
        zp = Path(self.tmp.name) / "evil.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            for p in src.rglob("*"):
                zf.write(p, p.relative_to(src))
            zf.writestr("../evil.txt", "pwned")
        with self.assertRaises(ManifestError):
            self.reg.install(zp)
        self.assertFalse((Path(self.tmp.name) / "evil.txt").exists())

    def test_non_zip_non_dir_rejected(self):
        p = Path(self.tmp.name) / "notes.txt"
        p.write_text("hello", encoding="utf-8")
        with self.assertRaises(ManifestError):
            self.reg.install(p)


class VersionOrderingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        from nomorals.storage.db import Database
        self.db = Database(str(Path(self.tmp.name) / "t.db"))
        self.reg = PluginRegistry(self.db, Path(self.tmp.name) / "plugins")

    def tearDown(self):
        self.tmp.cleanup()

    def test_latest_version_is_numeric(self):
        # R21: ORDER BY version DESC is lexicographic ("1.9.0" > "1.10.0")
        self.reg.install(_plugin_dir(Path(self.tmp.name), version="1.9.0"))
        self.reg.install(_plugin_dir(Path(self.tmp.name), version="1.10.0"))
        self.reg.install(_plugin_dir(Path(self.tmp.name), version="1.2.0"))
        self.assertEqual(self.reg.get("demo").version, "1.10.0")

    def test_prerelease_version_accepted(self):
        m = load_manifest({
            "name": "pre", "version": "1.0.0-beta",
            "entry_points": {"main": "m:f"}})
        self.assertEqual(m.version, "1.0.0-beta")
        self.reg.install(_plugin_dir(Path(self.tmp.name), name="pre",
                                     version="1.0.0-beta"))
        self.assertEqual(self.reg.get("pre").version, "1.0.0-beta")


class UnloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_unload_purges_modules(self):
        import sys
        from nomorals.plugins import unload_plugin
        src = _plugin_dir(Path(self.tmp.name))
        manifest = load_manifest_file(src)
        loaded = load_plugin(manifest, src)
        ns = "nomorals.plugins._loaded.demo"
        self.assertIn(ns + ".demo", sys.modules)
        unload_plugin(loaded)
        self.assertNotIn(ns + ".demo", sys.modules)
        self.assertNotIn(ns, sys.modules)

    def test_reload_after_unload_runs_new_code(self):
        from nomorals.plugins import unload_plugin
        src = _plugin_dir(Path(self.tmp.name))
        manifest = load_manifest_file(src)
        loaded = load_plugin(manifest, src)
        caps = PluginCapabilities(granted=frozenset())
        self.assertEqual(loaded.entry("main", caps)["ok"], True)
        unload_plugin(loaded)
        # new code under the same plugin name loads fresh
        (src / "demo.py").write_text(
            "def run(caps):\n    return {'ok': False, 'v2': True}\n",
            encoding="utf-8")
        loaded2 = load_plugin(manifest, src)
        self.assertEqual(loaded2.entry("main", caps)["v2"], True)


class WiringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        from nomorals.storage.db import Database
        self.db = Database(str(Path(self.tmp.name) / "t.db"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_kv_roundtrip(self):
        from nomorals.plugins import PluginKV
        kv = PluginKV(self.db, "demo")
        self.assertIsNone(kv.get("missing"))
        kv.set("count", 3)
        kv.set("cfg", {"a": [1, 2]})
        self.assertEqual(kv.get("count"), 3)
        self.assertEqual(kv.get("cfg"), {"a": [1, 2]})
        self.assertEqual(sorted(kv.keys()), ["cfg", "count"])
        self.assertTrue(kv.delete("count"))
        self.assertFalse(kv.delete("count"))
        # namespaced per plugin
        kv2 = PluginKV(self.db, "other")
        self.assertIsNone(kv2.get("cfg"))

    def test_kv_rejects_non_json(self):
        from nomorals.plugins import PluginKV
        kv = PluginKV(self.db, "demo")
        with self.assertRaises(PluginError):
            kv.set("bad", object())

    def test_fetcher_rejects_non_http(self):
        from nomorals.plugins import make_fetcher
        fetch = make_fetcher()
        with self.assertRaises(PluginError):
            fetch("file:///etc/passwd")

    def _wired_caps(self, permissions=("llm.chat",), router=None):
        """wire_capabilities with fake context/plugin for chat tests."""
        from types import SimpleNamespace
        from nomorals.plugins.wiring import wire_capabilities
        ctx = SimpleNamespace(
            db=self.db,
            settings=SimpleNamespace(
                workspace_dir=str(Path(self.tmp.name) / "ws")))
        if router is not None:
            ctx.router = router
        plugin = SimpleNamespace(
            name="demo",
            manifest=SimpleNamespace(permissions=list(permissions)))
        return wire_capabilities(ctx, plugin)

    @staticmethod
    def _mock_router(**kwargs):
        from nomorals.llm.providers.mock import MockProvider
        from nomorals.llm.router import LLMRouter
        router = LLMRouter()
        router.add(MockProvider(**kwargs), primary=True, name="mock")
        return router

    def test_chat_serves_via_mock_chain(self):
        from unittest.mock import patch
        from nomorals.plugins import wiring
        router = self._mock_router()
        with patch.object(wiring, "build_chain", return_value=router):
            caps = self._wired_caps()
            text = caps.chat([{"role": "user", "content": "hello"}])
        self.assertIsInstance(text, str)
        self.assertTrue(text)
        # the broker is attached: plugins get capability-routed chat
        self.assertIsNotNone(router.broker)
        # chain is built once, even across calls
        with patch.object(wiring, "build_chain",
                          side_effect=AssertionError("rebuilt")) as bc:
            caps.chat([{"role": "user", "content": "again"}])
            bc.assert_not_called()

    def test_chat_prefers_context_router(self):
        # When the context carries its own router (the agent's configured
        # chain), plugins use it — no parallel env chain is built.
        from unittest.mock import patch
        from nomorals.plugins import wiring
        router = self._mock_router()
        with patch.object(wiring, "build_chain",
                          side_effect=AssertionError("should not build")):
            caps = self._wired_caps(router=router)
            text = caps.chat([{"role": "user", "content": "hello"}])
        self.assertIsInstance(text, str)
        self.assertTrue(text)
        # the context router is reused as-is (its own broker untouched)
        self.assertIsNone(router.broker)

    def test_chat_rejects_malformed_messages(self):
        caps = self._wired_caps()
        for bad in ("nope", None, [], [{"role": "bogus", "content": "x"}],
                    [{"content": "x"}], ["str-msg"]):
            with self.assertRaises(PluginError, msg=f"bad={bad!r}"):
                caps.chat(bad)

    def test_chat_failure_surfaces_plugin_error(self):
        from unittest.mock import patch
        from nomorals.plugins import wiring
        router = self._mock_router(failure_rate=1.0)
        with patch.object(wiring, "build_chain", return_value=router):
            caps = self._wired_caps()
            with self.assertRaises(PluginError) as ctx:
                caps.chat([{"role": "user", "content": "hi"}])
        self.assertIn("llm.chat failed", str(ctx.exception))

    def test_chat_unwired_without_specs(self):
        from unittest.mock import patch
        from nomorals.plugins import wiring
        from nomorals.plugins.errors import PermissionDenied
        with patch.object(wiring, "specs_from_env", return_value=[]):
            caps = self._wired_caps()
            with self.assertRaises(PermissionDenied) as ctx:
                caps.chat([{"role": "user", "content": "hi"}])
        self.assertIn("no model wired", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
