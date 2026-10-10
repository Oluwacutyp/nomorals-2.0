"""Sweep tests for the upgraded plugin system (nomorals/plugins).

Covers the mined-then-built features: extended manifests, version
specs, the hook bus, entry timeouts, capability audit + budgets,
notify, schema-validated config, dependency/engine checks, lifecycle
hooks, upgrade, search, health, purge-on-remove, rendering themes, and
the llm.chat NameError regression fix.
"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

from nomorals.plugins import (
    AlreadyInstalled,
    DependencyError,
    HookResults,
    IncompatibleEngine,
    InstalledPlugin,
    LoadError,
    LoadedPlugin,
    ManifestError,
    NotInstalled,
    PermissionDenied,
    PluginCapabilities,
    PluginConfig,
    PluginError,
    PluginHookBus,
    PluginKV,
    PluginManifest,
    PluginRegistry,
    PluginTimeout,
    call_entry,
    load_manifest,
    load_manifest_file,
    load_plugin,
    parse_version_spec,
    render_health,
    render_plugin_card,
    render_plugin_table,
    satisfies_version,
    unload_plugin,
    validate_against_schema,
    version_key,
)
from nomorals.plugins.wiring import _make_chatter, make_audit, make_notifier


def _plugin_dir(base: Path, name: str = "demo", version: str = "1.0.0",
                permissions: list[str] | None = None,
                code: str = "",
                manifest_extra: dict | None = None) -> Path:
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
    manifest.update(manifest_extra or {})
    (d / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    (d / "demo.py").write_text(
        code or "def run(caps):\n    return {'ok': True}\n",
        encoding="utf-8")
    return d


def _db(tmp: Path):
    from nomorals.storage.db import Database
    return Database(str(tmp / "t.db"))


# ── version specs ───────────────────────────────────────────────────

class VersionSpecTests(unittest.TestCase):
    def test_ordering(self):
        self.assertTrue(version_key("1.10.0") > version_key("1.9.0"))
        self.assertTrue(version_key("1.0.0") > version_key("1.0.0-beta"))

    def test_satisfies(self):
        self.assertTrue(satisfies_version("1.5.0", ">=1.2, <2.0"))
        self.assertFalse(satisfies_version("2.0.0", ">=1.2, <2.0"))
        self.assertTrue(satisfies_version("1.0.0", ""))
        self.assertTrue(satisfies_version("1.4.7", "~=1.4.2"))
        self.assertFalse(satisfies_version("1.5.0", "~=1.4.2"))
        self.assertTrue(satisfies_version("1.4.2", "==1.4.2"))
        self.assertFalse(satisfies_version("1.4.3", "==1.4.2"))
        self.assertTrue(satisfies_version("1.4.3", "!=1.4.2"))

    def test_bad_spec_raises(self):
        with self.assertRaises(ManifestError):
            parse_version_spec("banana")
        with self.assertRaises(ManifestError):
            parse_version_spec(">=nope")


# ── manifest extensions ─────────────────────────────────────────────

class ManifestSweepTests(unittest.TestCase):
    def _base(self, **kw):
        data = {"name": "demo", "version": "1.0.0",
                "entry_points": {"main": "m:f"}}
        data.update(kw)
        return data

    def test_extended_fields(self):
        m = load_manifest(self._base(
            display_name="Demo!", tags=["tools", "web"], license="MIT",
            homepage="https://example.com", icon="icon.png",
            requires_devon=">=0.1.0",
            dependencies=["helper>=1.2", {"name": "other", "version": "<3"}],
            requirements=["requests>=2"],
            permissions=["notify.send"],
            purge_data_on_remove=True,
            custom_future_key="kept",
        ))
        self.assertEqual(m.display_name, "Demo!")
        self.assertEqual(m.tags, ["tools", "web"])
        self.assertEqual(m.requires_devon, ">=0.1.0")
        self.assertEqual(m.dependencies[0].name, "helper")
        self.assertTrue(m.dependencies[0].satisfied_by("1.5.0"))
        self.assertFalse(m.dependencies[0].satisfied_by("1.0.0"))
        self.assertEqual(m.dependencies[1].name, "other")
        self.assertIn("notify.send", m.permissions)
        self.assertTrue(m.purge_data_on_remove)
        self.assertEqual(m.extra["custom_future_key"], "kept")
        # round-trip
        m2 = load_manifest(m.to_dict())
        self.assertEqual(m2.display_name, "Demo!")
        self.assertEqual(m2.extra["custom_future_key"], "kept")

    def test_display_name_defaults_to_name(self):
        m = load_manifest(self._base())
        self.assertEqual(m.display_name, "demo")

    def test_bad_requires_devon(self):
        with self.assertRaises(ManifestError):
            load_manifest(self._base(requires_devon="soon"))

    def test_bad_dependency(self):
        with self.assertRaises(ManifestError):
            load_manifest(self._base(dependencies=["Bad Name!"]))
        with self.assertRaises(ManifestError):
            load_manifest(self._base(dependencies=["ok>=banana"]))

    def test_config_schema_validates_defaults(self):
        schema = {"type": "object",
                  "properties": {"loud": {"type": "boolean"}},
                  "required": ["loud"]}
        m = load_manifest(self._base(config_schema=schema,
                                     default_config={"loud": True}))
        self.assertEqual(m.default_config, {"loud": True})
        with self.assertRaises(ManifestError):
            load_manifest(self._base(config_schema=schema,
                                     default_config={"loud": "yes"}))
        with self.assertRaises(ManifestError):
            load_manifest(self._base(config_schema=schema,
                                     default_config={}))

    def test_schema_defaults_applied(self):
        schema = {"type": "object",
                  "properties": {"loud": {"type": "boolean",
                                          "default": False}}}
        m = load_manifest(self._base(config_schema=schema,
                                     default_config={}))
        self.assertEqual(m.default_config, {"loud": False})

    def test_hooks_normalized(self):
        m = load_manifest(self._base(hooks={
            "on_message": "m:fn",
            "on_boot": {"entry": "m:boot", "priority": 10}}))
        by_name = {h.name: h for h in m.hooks}
        self.assertEqual(by_name["on_message"].entry, "m:fn")
        self.assertEqual(by_name["on_message"].priority, 0)
        self.assertEqual(by_name["on_boot"].priority, 10)
        with self.assertRaises(ManifestError):
            load_manifest(self._base(hooks={"Bad Hook": "m:fn"}))

    def test_lifecycle_from_entry_points(self):
        m = load_manifest(self._base(
            entry_points={"main": "m:f", "on_install": "m:setup"}))
        self.assertEqual(m.lifecycle["on_install"], "m:setup")
        with self.assertRaises(ManifestError):
            load_manifest(self._base(lifecycle={"on_bogus": "m:f"}))

    def test_contributes(self):
        m = load_manifest(self._base(contributes={
            "commands": [{"name": "shout", "title": "Shout"}],
            "tools": [{"name": "shout", "description": "shouts",
                       "schema": {"type": "object"}, "entry": "m:shout"}]}))
        summary = m.contribution_summary()
        self.assertEqual(summary["commands"], ["shout"])
        self.assertEqual(summary["tools"], ["shout"])
        with self.assertRaises(ManifestError):
            load_manifest(self._base(
                contributes={"tools": [{"description": "no name"}]}))

    def test_limits(self):
        m = load_manifest(self._base(limits={"chat_calls": 5}))
        self.assertEqual(m.limits["chat_calls"], 5.0)
        # defaults fill in
        self.assertIn("fetch_calls", m.limits)
        with self.assertRaises(ManifestError):
            load_manifest(self._base(limits={"nope": 1}))
        with self.assertRaises(ManifestError):
            load_manifest(self._base(limits={"chat_calls": -2}))

    def test_schema_validator_subset(self):
        validate_against_schema("x", {"type": "string", "minLength": 1})
        with self.assertRaises(ManifestError):
            validate_against_schema(5, {"type": "string"})
        with self.assertRaises(ManifestError):
            validate_against_schema(True, {"type": "integer"})
        with self.assertRaises(ManifestError):
            validate_against_schema("b", {"enum": ["a"]})
        with self.assertRaises(ManifestError):
            validate_against_schema({}, {"type": "object",
                                          "required": ["x"]})


# ── hook bus ────────────────────────────────────────────────────────

def _hook_plugin_dir(base: Path, name: str, hooks: dict,
                     impl_code: str) -> Path:
    d = base / f"{name}-src"
    d.mkdir(parents=True, exist_ok=True)
    (d / "plugin.json").write_text(json.dumps({
        "name": name, "version": "1.0.0",
        "entry_points": {"main": "impl:run"},
        "hooks": hooks}), encoding="utf-8")
    (d / "impl.py").write_text(impl_code, encoding="utf-8")
    return d


class HookBusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.bus = PluginHookBus()

    def tearDown(self):
        self.tmp.cleanup()

    def _attach(self, name: str, hooks: dict, code: str):
        src = _hook_plugin_dir(self.base, name, hooks, code)
        manifest = load_manifest_file(src)
        loaded = load_plugin(manifest, src)
        caps = PluginCapabilities(granted=frozenset(),
                                  plugin_name=name)
        self.bus.attach(loaded, caps)
        return loaded

    def test_emit_priority_order(self):
        order = []
        self._attach("aaa", {"ev": {"entry": "impl:h", "priority": 1}},
                     "def h(caps):\n    return 'aaa'\n"
                     "def run(caps):\n    return None\n")
        self._attach("zzz", {"ev": {"entry": "impl:h", "priority": 50}},
                     "def h(caps):\n    return 'zzz'\n"
                     "def run(caps):\n    return None\n")
        res = self.bus.emit("ev")
        self.assertIsInstance(res, HookResults)
        self.assertEqual([p for p, _ in res.results], ["zzz", "aaa"])
        self.assertEqual(res.errors, [])

    def test_error_isolation(self):
        self._attach("bad", {"ev": "impl:h"},
                     "def h(caps):\n    raise RuntimeError('boom')\n"
                     "def run(caps):\n    return None\n")
        self._attach("good", {"ev": "impl:h"},
                     "def h(caps):\n    return 42\n"
                     "def run(caps):\n    return None\n")
        res = self.bus.emit("ev")
        self.assertEqual(res.results, [("good", 42)])
        self.assertEqual(len(res.errors), 1)
        self.assertEqual(res.errors[0][0], "bad")
        self.assertIn("boom", res.errors[0][1])

    def test_emit_first(self):
        self._attach("nully", {"ev": {"entry": "impl:h", "priority": 99}},
                     "def h(caps):\n    return None\n"
                     "def run(caps):\n    return None\n")
        self._attach("winner", {"ev": "impl:h"},
                     "def h(caps):\n    return 'win'\n"
                     "def run(caps):\n    return None\n")
        self.assertEqual(self.bus.emit_first("ev"), "win")

    def test_attach_validates_impl(self):
        # hook impl with no parameters: must be rejected at attach time,
        # not when the hook fires.
        src = _hook_plugin_dir(
            self.base, "badsig", {"ev": "impl:h"},
            "def h():\n    return 1\n"
            "def run(caps):\n    return None\n")
        manifest = load_manifest_file(src)
        loaded = load_plugin(manifest, src)
        caps = PluginCapabilities(granted=frozenset(), plugin_name="badsig")
        with self.assertRaises(LoadError):
            self.bus.attach(loaded, caps)

    def test_detach(self):
        self._attach("solo", {"ev": "impl:h"},
                     "def h(caps):\n    return 1\n"
                     "def run(caps):\n    return None\n")
        self.assertEqual(self.bus.hooks(), {"ev": ["solo"]})
        self.assertEqual(self.bus.detach("solo"), 1)
        self.assertEqual(self.bus.hooks(), {})
        self.assertEqual(self.bus.emit("ev").results, [])


# ── call_entry timeout ──────────────────────────────────────────────

class CallEntryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _loaded(self, code: str):
        src = _plugin_dir(self.base, code=code)
        return load_plugin(load_manifest_file(src), src)

    def test_no_timeout_passthrough(self):
        loaded = self._loaded("def run(caps):\n    return 'fast'\n")
        caps = PluginCapabilities(granted=frozenset())
        self.assertEqual(call_entry(loaded, "main", caps), "fast")

    def test_timeout_raises(self):
        loaded = self._loaded(
            "import time\ndef run(caps):\n    time.sleep(30)\n")
        caps = PluginCapabilities(granted=frozenset())
        with self.assertRaises(PluginTimeout):
            call_entry(loaded, "main", caps, timeout=0.2)

    def test_plugin_exception_propagates(self):
        loaded = self._loaded(
            "def run(caps):\n    raise ValueError('nope')\n")
        caps = PluginCapabilities(granted=frozenset())
        with self.assertRaises(ValueError):
            call_entry(loaded, "main", caps, timeout=5)


# ── capabilities: audit, budgets, notify ────────────────────────────

class CapabilitiesSweepTests(unittest.TestCase):
    def test_audit_sink_captures(self):
        seen = []
        caps = PluginCapabilities(
            granted=frozenset(["network.fetch"]),
            plugin_name="demo",
            fetcher=lambda url, timeout=30: b"data",
            audit=lambda a, p, d: seen.append((a, p, d)))
        caps.fetch("http://example.com/x")
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][:2], ("fetch", "network.fetch"))
        self.assertIn("http://example.com/x", seen[0][2])

    def test_audit_failure_is_fail_open(self):
        def bad(a, p, d):
            raise RuntimeError("bus down")
        caps = PluginCapabilities(
            granted=frozenset(["network.fetch"]),
            fetcher=lambda url, timeout=30: b"data",
            audit=bad)
        self.assertEqual(caps.fetch("http://x"), b"data")  # no raise

    def test_chat_budget_enforced(self):
        caps = PluginCapabilities(
            granted=frozenset(["llm.chat"]),
            chatter=lambda messages: "hi",
            limits={"chat_calls": 2})
        caps.chat([{"role": "user", "content": "a"}])
        caps.chat([{"role": "user", "content": "b"}])
        with self.assertRaises(PermissionDenied):
            caps.chat([{"role": "user", "content": "c"}])
        self.assertEqual(caps.usage["chat_calls"], 2)

    def test_fetch_budget_enforced(self):
        caps = PluginCapabilities(
            granted=frozenset(["network.fetch"]),
            fetcher=lambda url, timeout=30: b"d",
            limits={"fetch_calls": 1})
        caps.fetch("http://x")
        with self.assertRaises(PermissionDenied):
            caps.fetch("http://x")

    def test_notify(self):
        notes = []
        caps = PluginCapabilities(
            granted=frozenset(["notify.send"]),
            notifier=lambda text, title="": notes.append((title, text)))
        caps.notify("hello owner", title="ping")
        self.assertEqual(notes, [("ping", "hello owner")])

    def test_notify_denied_without_permission(self):
        caps = PluginCapabilities(granted=frozenset())
        with self.assertRaises(PermissionDenied):
            caps.notify("x")

    def test_notify_unwired(self):
        caps = PluginCapabilities(granted=frozenset(["notify.send"]))
        with self.assertRaises(PermissionDenied):
            caps.notify("x")

    def test_make_notifier_and_audit_publish(self):
        from nomorals.core.events import global_bus
        events = []
        sub_id = global_bus.subscribe(
            "plugin.notify", lambda e: events.append(e), sync=True)
        try:
            make_notifier("demo")("hi", title="t")
        finally:
            global_bus.unsubscribe(sub_id)
        self.assertTrue(any(e.topic == "plugin.notify" and
                            e.data["plugin"] == "demo" for e in events))


# ── plugin config ───────────────────────────────────────────────────

class PluginConfigTests(unittest.TestCase):
    def _manifest(self):
        return load_manifest({
            "name": "demo", "version": "1.0.0",
            "entry_points": {"main": "m:f"},
            "config_schema": {
                "type": "object",
                "properties": {
                    "loud": {"type": "boolean", "default": False},
                    "level": {"type": "integer", "minimum": 1,
                              "maximum": 10},
                },
            },
            "default_config": {"level": 3},
        })

    def test_defaults(self):
        cfg = PluginConfig(self._manifest())
        self.assertEqual(cfg.get("loud"), False)
        self.assertEqual(cfg.get("level"), 3)
        self.assertEqual(cfg.get("missing", "dflt"), "dflt")

    def test_set_and_persist(self):
        kv = PluginKV(_db(Path(tempfile.mkdtemp())), "demo")
        cfg = PluginConfig(self._manifest(), kv)
        cfg.set("loud", True)
        self.assertTrue(cfg.get("loud"))
        # new instance reads the persisted override
        cfg2 = PluginConfig(self._manifest(), kv)
        self.assertTrue(cfg2.get("loud"))
        self.assertEqual(cfg2.get("level"), 3)

    def test_set_invalid_raises(self):
        kv = PluginKV(_db(Path(tempfile.mkdtemp())), "demo")
        cfg = PluginConfig(self._manifest(), kv)
        with self.assertRaises(ManifestError):
            cfg.set("level", 99)
        with self.assertRaises(ManifestError):
            cfg.set("loud", "yes")

    def test_set_without_kv_raises(self):
        cfg = PluginConfig(self._manifest())
        with self.assertRaises(PermissionDenied):
            cfg.set("loud", True)

    def test_reset(self):
        kv = PluginKV(_db(Path(tempfile.mkdtemp())), "demo")
        cfg = PluginConfig(self._manifest(), kv)
        cfg.set("loud", True)
        cfg.reset()
        self.assertEqual(cfg.get("loud"), False)

    def test_caps_config(self):
        kv = PluginKV(_db(Path(tempfile.mkdtemp())), "demo")
        caps = PluginCapabilities(
            granted=frozenset(["storage.kv"]), plugin_name="demo",
            manifest=self._manifest(), kv=kv)
        cfg = caps.config()
        self.assertIsInstance(cfg, PluginConfig)
        self.assertEqual(cfg.get("level"), 3)


# ── registry sweep ──────────────────────────────────────────────────

class RegistrySweepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.reg = PluginRegistry(_db(self.base), self.base / "plugins")

    def tearDown(self):
        self.tmp.cleanup()

    def test_incompatible_engine(self):
        src = _plugin_dir(self.base, manifest_extra={
            "requires_devon": ">=999.0.0"})
        with self.assertRaises(IncompatibleEngine):
            self.reg.install(src)

    def test_missing_dependency(self):
        src = _plugin_dir(self.base, manifest_extra={
            "dependencies": ["ghost-helper>=1.0"]})
        with self.assertRaises(DependencyError) as ctx:
            self.reg.install(src)
        self.assertIn("ghost-helper", str(ctx.exception))

    def test_satisfied_dependency(self):
        dep = _plugin_dir(self.base, name="helper", version="1.2.0")
        self.reg.install(dep)
        src = _plugin_dir(self.base, manifest_extra={
            "dependencies": ["helper>=1.0"]})
        p = self.reg.install(src)
        self.assertEqual(p.name, "demo")

    def test_unsatisfied_version_dependency(self):
        dep = _plugin_dir(self.base, name="helper", version="1.0.0")
        self.reg.install(dep)
        src = _plugin_dir(self.base, manifest_extra={
            "dependencies": ["helper>=2.0"]})
        with self.assertRaises(DependencyError):
            self.reg.install(src)

    def test_on_install_lifecycle_runs(self):
        src = _plugin_dir(
            self.base, permissions=["storage.kv"],
            manifest_extra={"entry_points": {"main": "demo:run",
                                              "on_install": "demo:setup"}},
            code=("def run(caps):\n    return None\n"
                  "def setup(caps):\n"
                  "    caps.kv().set('seeded', True)\n"
                  "    return 'seeded'\n"))
        p = self.reg.install(src)
        self.assertEqual(p.name, "demo")
        kv = PluginKV(self.reg.db, "demo")
        self.assertTrue(kv.get("seeded"))

    def test_failing_on_install_rolls_back(self):
        src = _plugin_dir(
            self.base, manifest_extra={"entry_points": {
                "main": "demo:run", "on_install": "demo:setup"}},
            code=("def run(caps):\n    return None\n"
                  "def setup(caps):\n    raise RuntimeError('nope')\n"))
        with self.assertRaises(PluginError):
            self.reg.install(src)
        with self.assertRaises(NotInstalled):
            self.reg.get("demo")
        self.assertFalse((self.base / "plugins" / "demo-1.0.0").exists())

    def test_on_disable_fail_open(self):
        src = _plugin_dir(
            self.base, manifest_extra={"entry_points": {
                "main": "demo:run", "on_disable": "demo:bye"}},
            code=("def run(caps):\n    return None\n"
                  "def bye(caps):\n    raise RuntimeError('broken')\n"))
        self.reg.install(src)
        p = self.reg.disable("demo")  # must not raise
        self.assertFalse(p.enabled)

    def test_upgrade(self):
        v1 = _plugin_dir(self.base, version="1.0.0", permissions=["storage.kv"],
                         code="def run(caps):\n    caps.kv().set('k', 'v1')\n")
        p1 = self.reg.install(v1)
        # simulate the plugin having stored data
        PluginKV(self.reg.db, "demo").set("keep", 1)
        v2 = _plugin_dir(self.base, version="1.1.0", permissions=["storage.kv"],
                         manifest_extra={"entry_points": {
                             "main": "demo:run",
                             "on_upgrade": "demo:migrate"}},
                         code=("def run(caps):\n    return None\n"
                               "def migrate(caps):\n"
                               "    caps.kv().set('migrated', True)\n"))
        p2 = self.reg.upgrade(v2)
        self.assertEqual(p2.version, "1.1.0")
        self.assertEqual(self.reg.get("demo").version, "1.1.0")
        # kv carried over (per-name namespace) + migration ran
        kv = PluginKV(self.reg.db, "demo")
        self.assertEqual(kv.get("keep"), 1)
        self.assertTrue(kv.get("migrated"))
        # old version still installed
        self.assertEqual(self.reg.get("demo", "1.0.0").version, "1.0.0")

    def test_upgrade_rejects_older(self):
        v1 = _plugin_dir(self.base, version="2.0.0")
        self.reg.install(v1)
        v0 = _plugin_dir(self.base, version="1.0.0")
        with self.assertRaises(PluginError):
            self.reg.upgrade(v0)

    def test_search(self):
        _plugin_dir(self.base, name="shouter", manifest_extra={
            "description": "makes noise loudly", "tags": ["audio"]})
        _plugin_dir(self.base, name="quiet-one", manifest_extra={
            "description": "says nothing"})
        self.reg.install(self.base / "shouter-1.0.0-src")
        self.reg.install(self.base / "quiet-one-1.0.0-src")
        hits = self.reg.search("loudly")
        self.assertEqual([p.name for p in hits], ["shouter"])
        hits = self.reg.search("AUDIO")
        self.assertEqual([p.name for p in hits], ["shouter"])
        self.assertEqual(self.reg.search(""), [])

    def test_health(self):
        helper = _plugin_dir(self.base, name="helper", version="1.0.0")
        self.reg.install(helper)
        src = _plugin_dir(self.base, manifest_extra={
            "dependencies": ["helper>=1.0"],
            "entry_points": {"main": "demo:run",
                             "broken": "demo:missing"}})
        self.reg.install(src)
        # dependency satisfied while helper is present…
        self.assertTrue(self.reg.health("demo")["dependencies"][0]
                        ["satisfied"])
        # …then break it by removing the helper
        self.reg.remove("helper")
        report = self.reg.health("demo")
        self.assertTrue(report["manifest_ok"])
        self.assertTrue(report["loadable"])
        self.assertTrue(report["entry_points"]["main"]["ok"])
        self.assertFalse(report["entry_points"]["broken"]["ok"])
        self.assertFalse(report["dependencies"][0]["satisfied"])
        self.assertTrue(report["engine"]["ok"])
        # health has no side effects: entry point was not called
        self.assertEqual(self.reg.get("demo").enabled, True)

    def test_purge_data_on_remove(self):
        src = _plugin_dir(self.base, permissions=["storage.kv"],
                          manifest_extra={"purge_data_on_remove": True})
        self.reg.install(src)
        PluginKV(self.reg.db, "demo").set("k", "v")
        self.reg.remove("demo")
        self.assertEqual(PluginKV(self.reg.db, "demo").keys(), [])

    def test_no_purge_by_default(self):
        src = _plugin_dir(self.base, permissions=["storage.kv"])
        self.reg.install(src)
        PluginKV(self.reg.db, "demo").set("k", "v")
        self.reg.remove("demo")
        self.assertEqual(PluginKV(self.reg.db, "demo").get("k"), "v")

    def test_to_dict_extended(self):
        src = _plugin_dir(self.base, manifest_extra={
            "display_name": "Demo!", "tags": ["x"],
            "contributes": {"commands": [{"name": "shout"}]},
            "hooks": {"ev": "demo:run"}})
        p = self.reg.install(src)
        d = p.to_dict()
        self.assertEqual(d["display_name"], "Demo!")
        self.assertEqual(d["contributes"]["commands"], ["shout"])
        self.assertEqual(d["contributes"]["hooks"], ["ev"])


# ── llm.chat regression (the self.context NameError) ────────────────

class ChatterRegressionTests(unittest.TestCase):
    def test_chatter_uses_context_router(self):
        from nomorals.llm.base import LLMResponse

        class FakeRouter:
            def chat(self, messages, params=None, **kw):
                return LLMResponse(
                    text="hello from fake", model="fake", provider="fake")

        class FakeContext:
            router = FakeRouter()

        chatter = _make_chatter([], FakeContext())
        # Before the fix this raised NameError: name 'self' is not defined.
        result = chatter([{"role": "user", "content": "hi"}])
        self.assertEqual(result, "hello from fake")

    def test_chatter_validates_input_first(self):
        class FakeContext:
            router = None
        chatter = _make_chatter([], FakeContext())
        with self.assertRaises(PluginError):
            chatter("not a list")
        with self.assertRaises(PluginError):
            chatter([])


# ── rendering ───────────────────────────────────────────────────────

class RenderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.reg = PluginRegistry(_db(self.base), self.base / "plugins")
        src = _plugin_dir(self.base, manifest_extra={
            "display_name": "Demo!", "tags": ["tools"],
            "permissions": ["network.fetch"],
            "contributes": {"commands": [{"name": "shout"}]},
            "hooks": {"ev": "demo:run"}})
        self.plugin = self.reg.install(src)

    def tearDown(self):
        self.tmp.cleanup()

    def test_all_themes(self):
        for theme in ("unicode", "plain", "compact", "markdown"):
            card = render_plugin_card(self.plugin, theme)
            self.assertIn("demo", card)
            table = render_plugin_table([self.plugin], theme)
            self.assertIn("demo", table)
            health = render_health(self.reg.health("demo"), theme)
            self.assertIn("demo", health)

    def test_bad_theme(self):
        with self.assertRaises(ValueError):
            render_plugin_card(self.plugin, "neon")

    def test_empty_table(self):
        self.assertIn("no plugins", render_plugin_table([]))

    def test_compact_is_single_line(self):
        card = render_plugin_card(self.plugin, "compact")
        self.assertNotIn("\n", card)
        self.assertIn("1.0.0", card)

    def test_card_shows_contributions(self):
        card = render_plugin_card(self.plugin, "unicode")
        self.assertIn("shout", card)
        self.assertIn("network.fetch", card)


if __name__ == "__main__":
    unittest.main()
