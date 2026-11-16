"""Sweep tests for nomorals/os/ — new behavior added in the os sweep.

Covers: health probe roles/thresholds/manual checks/background loop,
service scopes/lifecycle/overrides, kernel shutdown hooks, session
presence/expiry, bridge handoff, project tags/search/summary, timeline
FTS/prune/export/import, update journal/recovery/quarantine/history,
new verifiers, mission guards/callbacks/diagrams, replay compare/HTML,
snapshot tags/prune/diff/tar/pre-post, resource render/alerts, generic
session adapter.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.core.events import EventBus
from nomorals.os import adapters as os_adapters
from nomorals.os.health import (
    HealthMonitor,
    HealthStatus,
    as_health_check,
)
from nomorals.os.kernel import OSKernel
from nomorals.os.mission_state import (
    COMPLETED,
    FAILED,
    PLANNED,
    RUNNING,
    VERIFYING,
    GuardFailed,
    InvalidTransition,
    clear_callbacks,
    clear_guards,
    current_state,
    describe as describe_machine,
    guard,
    on_enter,
    on_exit,
    on_transition,
    to_mermaid,
    transition,
    transition_log,
)
from nomorals.os.project import ProjectStore
from nomorals.os.replay import (
    MissionReplay,
    compare_replays,
    replay_mission,
)
from nomorals.os.resources import ResourceManager
from nomorals.os.services import FACTORY, SINGLETON, ServiceRegistry
from nomorals.os.session import SessionStore
from nomorals.os.session_bridge import SessionBridge
from nomorals.os.snapshots import SnapshotManager
from nomorals.os.timeline import Timeline
from nomorals.os.update import UpdateManager, UpdateReport
from nomorals.os.verifiers import (
    CompositeVerifier,
    DiskSpaceVerifier,
    GitCleanVerifier,
    LintVerifier,
    Verdict,
    default_registry,
    gate_registry,
)


def _mission_store():
    from nomorals.missions.mission import MissionStore
    from nomorals.storage.db import Database

    db = Database(":memory:")
    db.migrate()
    return MissionStore(db), db


def _event(topic, data, event_id="e1", ts=1700000000.0):
    return SimpleNamespace(topic=topic, data=data, event_id=event_id,
                           ts=ts, source="test")


class HealthSweepTests(unittest.TestCase):
    def test_roles_recorded(self) -> None:
        m = HealthMonitor()
        m.register_check(lambda: True, name="live", role="liveness")
        m.register_check(lambda: True, name="ready", role="readiness")
        m.register_check(lambda: True, name="boot", role="startup")
        self.assertEqual(m.check_names(role="liveness"), ["live"])
        results = m.check_all()
        self.assertEqual(results["live"].role, "liveness")

    def test_flap_suppression(self) -> None:
        fired: list[HealthStatus] = []
        m = HealthMonitor()
        m.register_check(lambda: False, name="flaky", failure_threshold=3,
                         success_threshold=2)
        m.on_failure("flaky", fired.append)
        m.check_all()
        m.check_all()
        self.assertEqual(fired, [])  # not yet declared down
        self.assertFalse(m.is_down("flaky"))
        m.check_all()
        self.assertTrue(m.is_down("flaky"))
        self.assertEqual(len(fired), 1)  # fired once, on declaration

    def test_recovery_needs_consecutive_success(self) -> None:
        state = {"ok": False}
        m = HealthMonitor()
        m.register_check(lambda: state["ok"], name="svc",
                         failure_threshold=1, success_threshold=2)
        m.check_all()
        self.assertTrue(m.is_down("svc"))
        state["ok"] = True
        m.check_all()
        self.assertTrue(m.is_down("svc"))  # one success is not enough
        m.check_all()
        self.assertFalse(m.is_down("svc"))

    def test_manual_check(self) -> None:
        m = HealthMonitor()
        manual = m.register_manual("cache-warm")
        self.assertFalse(m.check_one("cache-warm").ok)
        manual.set(True, "warmed 1024 entries")
        status = m.check_one("cache-warm")
        self.assertTrue(status.ok)
        self.assertEqual(status.role, "startup")

    def test_check_timeout(self) -> None:
        m = HealthMonitor()
        m.register_check(lambda: (time.sleep(5), True)[1], name="slow",
                         timeout_s=0.2)
        status = m.check_one("slow")
        self.assertFalse(status.ok)
        self.assertIn("timed out", status.detail)

    def test_background_loop_and_summary(self) -> None:
        m = HealthMonitor()
        m.register_check(lambda: True, name="ok1", role="readiness",
                         interval_s=0.5)
        m.register_check(lambda: False, name="bad1", role="liveness",
                         interval_s=0.5, failure_threshold=1)
        m.start_background()
        try:
            deadline = time.time() + 5
            while time.time() < deadline:
                if m.last_status("ok1") and m.last_status("bad1"):
                    break
                time.sleep(0.1)
            summary = m.summary()
            self.assertFalse(summary["alive"])
            self.assertTrue(summary["ready"])
            self.assertIn("liveness:bad1", summary["down"])
            rendered = m.render()
            self.assertIn("✗ [liveness] bad1", rendered)
            self.assertIn("✓ [readiness] ok1", rendered)
        finally:
            m.stop_background()

    def test_history(self) -> None:
        m = HealthMonitor()
        m.register_check(lambda: True, name="h")
        m.check_all()
        m.check_all()
        self.assertEqual(len(m.history("h", limit=10)), 2)


class ServicesSweepTests(unittest.TestCase):
    def test_factory_scope_builds_fresh(self) -> None:
        reg = ServiceRegistry()
        reg.register("fresh", lambda: object(), scope=FACTORY)
        self.assertIsNot(reg.lookup("fresh"), reg.lookup("fresh"))

    def test_singleton_scope_caches(self) -> None:
        reg = ServiceRegistry()
        reg.register("one", lambda: object(), scope=SINGLETON)
        self.assertIs(reg.lookup("one"), reg.lookup("one"))

    def test_bad_scope_rejected(self) -> None:
        reg = ServiceRegistry()
        with self.assertRaises(ValueError):
            reg.register("x", lambda: 1, scope="bogus")

    def test_override_context(self) -> None:
        reg = ServiceRegistry()
        reg.register("svc", lambda: "real")
        self.assertEqual(reg.lookup("svc"), "real")
        with reg.override("svc", "mock"):
            self.assertEqual(reg.lookup("svc"), "mock")
        self.assertEqual(reg.lookup("svc"), "real")

    def test_override_unknown_raises(self) -> None:
        reg = ServiceRegistry()
        with self.assertRaises(Exception):
            with reg.override("nope", 1):
                pass

    def test_lifecycle_start_stop_order(self) -> None:
        reg = ServiceRegistry()
        order: list[str] = []
        reg.register("base", lambda: "b",
                     on_start=lambda i: order.append("start:base"),
                     on_stop=lambda i: order.append("stop:base"))
        reg.register("top", lambda: "t", depends_on=("base",),
                     on_start=lambda i: order.append("start:top"),
                     on_stop=lambda i: order.append("stop:top"))
        reg.start_service("top")
        self.assertEqual(order, ["start:base", "start:top"])
        self.assertTrue(reg.is_started("top"))
        reg.stop_service("base")  # stops dependent first
        self.assertEqual(order[-2:], ["stop:top", "stop:base"])
        self.assertFalse(reg.is_started("base"))

    def test_start_all_respects_autostart(self) -> None:
        reg = ServiceRegistry()
        reg.register("lazy", lambda: object())
        reg.register("eager", lambda: object(), autostart=True)
        reg.start_all()
        self.assertFalse(reg.is_started("lazy"))
        self.assertTrue(reg.is_started("eager"))

    def test_restart_recycles(self) -> None:
        reg = ServiceRegistry()
        reg.register("r", lambda: object())
        first = reg.start_service("r")
        second = reg.restart("r")
        self.assertIsNot(first, second)
        self.assertTrue(reg.is_started("r"))

    def test_circular_dependency_rejected(self) -> None:
        reg = ServiceRegistry()
        reg.register("a", lambda: 1, depends_on=("b",))
        reg.register("b", lambda: 2, depends_on=("a",))
        with self.assertRaises(ValueError):
            reg.start_all(only_autostart=False)

    def test_tags_and_render(self) -> None:
        reg = ServiceRegistry()
        reg.register("web", lambda: 1, tags=("net", "public"))
        self.assertEqual(reg.find_by_tag("net"), ["web"])
        rendered = reg.render()
        self.assertIn("web", rendered)
        self.assertIn("[net,public]", rendered)

    def test_health_check_for_still_works(self) -> None:
        reg = ServiceRegistry()
        self.assertIsNotNone(reg.health_check_for("browser"))


class KernelSweepTests(unittest.TestCase):
    def test_shutdown_hooks_run_lifo(self) -> None:
        kernel = OSKernel(bus=EventBus())
        calls: list[str] = []
        kernel.on_shutdown(lambda: calls.append("first"), name="first")
        kernel.on_shutdown(lambda: calls.append("second"), name="second")
        kernel.start()
        self.assertTrue(kernel.ready)
        kernel.stop()
        self.assertEqual(calls, ["second", "first"])
        self.assertFalse(kernel.ready)

    def test_failing_hook_does_not_break_stop(self) -> None:
        kernel = OSKernel(bus=EventBus())

        def bad() -> None:
            raise RuntimeError("boom")

        kernel.on_shutdown(bad)
        kernel.start()
        kernel.stop()  # must not raise
        self.assertFalse(kernel.started)

    def test_render_and_status(self) -> None:
        kernel = OSKernel(bus=EventBus())
        kernel.start()
        try:
            st = kernel.status()
            self.assertTrue(st["ready"])
            self.assertIsNotNone(st["uptime_s"])
            self.assertIn("browser", st["services"])
            self.assertIn("health_summary", st)
            rendered = kernel.render()
            self.assertIn("READY", rendered)
            self.assertIn("services", rendered)
        finally:
            kernel.stop()

    def test_restart_service(self) -> None:
        kernel = OSKernel(bus=EventBus())
        kernel.services.register("tmp", lambda: object())
        first = kernel.services.start_service("tmp")
        second = kernel.restart_service("tmp")
        self.assertIsNot(first, second)
        kernel.stop()


class SessionSweepTests(unittest.TestCase):
    def test_presence(self) -> None:
        store = SessionStore()
        s = store.create(frontend="cli")
        self.assertFalse(s.attached)
        self.assertEqual(store.attach(s.id), 1)
        self.assertEqual(store.attach(s.id), 2)
        self.assertEqual(store.detach(s.id), 1)
        self.assertEqual(store.detach(s.id), 0)
        self.assertEqual(store.detach(s.id), 0)  # floors at 0
        self.assertIsNone(store.attach("nope"))

    def test_idle_and_purge(self) -> None:
        store = SessionStore()
        old = store.create(frontend="cli")
        old.last_activity_at = time.time() - 3600
        store.update(old)
        fresh = store.create(frontend="cli")
        store.activity(fresh.id)  # keeps it fresh
        attached = store.create(frontend="cli")
        attached.last_activity_at = time.time() - 3600
        store.update(attached)
        store.attach(attached.id)  # attached → spared
        ended = store.purge_expired(600)
        self.assertIn(old.id, ended)
        self.assertNotIn(fresh.id, ended)
        self.assertNotIn(attached.id, ended)
        self.assertIsNotNone(store.get(fresh.id))

    def test_rename_and_render(self) -> None:
        store = SessionStore()
        s = store.create(frontend="telegram", name="work chat")
        self.assertTrue(store.rename(s.id, "home chat"))
        self.assertEqual(store.get(s.id).display_name, "home chat")
        self.assertFalse(store.rename("nope", "x"))
        rendered = store.render()
        self.assertIn("home chat", rendered)
        self.assertIn("telegram", rendered)

    def test_counts_by_frontend(self) -> None:
        store = SessionStore()
        store.create(frontend="cli")
        store.create(frontend="cli")
        store.create(frontend="telegram")
        self.assertEqual(store.counts_by_frontend(),
                         {"cli": 2, "telegram": 1})


class BridgeSweepTests(unittest.TestCase):
    def _bridge(self) -> SessionBridge:
        return SessionBridge(db=":memory:")

    def test_handoff_moves_conversation(self) -> None:
        bridge = self._bridge()
        s = bridge.store.create(frontend="telegram", principal="owner",
                                conversation_id="telegram:123")
        moved = bridge.handoff_session(s.id, "cli", "console")
        self.assertIsNotNone(moved)
        assert moved is not None
        self.assertEqual(moved.conversation_id, "cli:console")
        self.assertEqual(moved.frontend, "cli")
        self.assertEqual(len(moved.state["handoff_history"]), 1)
        self.assertEqual(moved.state["handoff_history"][0]["from"],
                         "telegram:123")

    def test_handoff_unknown_returns_none(self) -> None:
        bridge = self._bridge()
        self.assertIsNone(bridge.handoff_session("nope", "cli", "x"))

    def test_principal_and_counts(self) -> None:
        bridge = self._bridge()
        bridge.store.create(frontend="cli", principal="owner")
        bridge.store.create(frontend="telegram", principal="guest")
        self.assertEqual(len(bridge.sessions_for_principal("owner")), 1)
        counts = bridge.session_counts()
        self.assertEqual(counts["total"], 2)
        self.assertEqual(counts["by_principal"], {"owner": 1, "guest": 1})
        self.assertIn("cli", bridge.render())


class ProjectSweepTests(unittest.TestCase):
    def test_tags(self) -> None:
        store = ProjectStore()
        p = store.create("alpha", tags=["ml"])
        self.assertTrue(store.tag(p.id, "urgent"))
        self.assertTrue(store.tag(p.id, "urgent"))  # idempotent
        self.assertEqual(store.get(p.id).tags, ["ml", "urgent"])
        self.assertTrue(store.untag(p.id, "ml"))
        self.assertEqual(store.get(p.id).tags, ["urgent"])
        self.assertEqual([x.id for x in store.with_tag("urgent")], [p.id])

    def test_search_and_archive_and_rename(self) -> None:
        store = ProjectStore()
        p = store.create("Website Redesign")
        self.assertEqual([x.id for x in store.search("website")], [p.id])
        self.assertEqual(store.search("website", state="active"), [store.get(p.id)])
        self.assertTrue(store.rename(p.id, "Site Redesign"))
        self.assertEqual(store.get(p.id).name, "Site Redesign")
        self.assertTrue(store.archive(p.id))
        self.assertTrue(store.get(p.id).archived)
        self.assertEqual(store.search("site", state="active"), [])

    def test_summary(self) -> None:
        store = ProjectStore()
        p = store.create("beta")
        store.add_mission(p.id, "m1")

        class FakeMissionStore:
            def get(self, mid: str):
                return SimpleNamespace(status="running")

        summary = store.summary(p.id, mission_store=FakeMissionStore())
        assert summary is not None
        self.assertEqual(summary["missions"]["total"], 1)
        self.assertEqual(summary["missions"]["by_status"], {"running": 1})
        self.assertEqual(summary["sessions"], -1)  # not provided
        self.assertIsNone(store.summary("nope"))
        self.assertIn("beta", store.render())


class TimelineSweepTests(unittest.TestCase):
    def test_fts_search(self) -> None:
        t = Timeline()
        t.record(_event("mission.transition",
                        {"note": "deployed the quantum flux capacitor"}))
        t.record(_event("mission.transition", {"note": "unrelated"},
                        event_id="e2"))
        if not t.fts_available:
            self.skipTest("FTS5 unavailable")
        hits = t.search("quantum")
        self.assertEqual([h["event_id"] for h in hits], ["e1"])
        hits = t.search('"flux capacitor"')
        self.assertEqual(len(hits), 1)

    def test_envelope_and_trace(self) -> None:
        t = Timeline()
        t.record(_event("task.started",
                        {"correlation_id": "corr-9", "actor": "runner"}))
        t.record(_event("task.done",
                        {"correlation_id": "corr-9", "actor": "runner"},
                        event_id="e2", ts=1700000001.0))
        rows = t.trace("corr-9")
        self.assertEqual([r["event_id"] for r in rows], ["e1", "e2"])
        self.assertEqual(rows[0]["actor"], "runner")
        self.assertEqual(rows[0]["correlation_id"], "corr-9")

    def test_prune(self) -> None:
        t = Timeline()
        t.record(_event("a", {}, ts=100.0))
        t.record(_event("b", {}, event_id="e2", ts=200.0))
        t.record(_event("c", {}, event_id="e3", ts=300.0))
        self.assertEqual(t.prune_older_than(150.0), 1)
        self.assertEqual(t.stats()["events"], 2)
        self.assertEqual(t.prune_keep_latest(0), 2)
        self.assertEqual(t.stats()["events"], 0)

    def test_export_import_roundtrip(self) -> None:
        t = Timeline()
        t.record(_event("mission.transition",
                        {"mission_id": "m1", "correlation_id": "c1",
                         "actor": "runner"}))
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "dump.jsonl")
            self.assertEqual(t.export_jsonl(path), 1)
            t2 = Timeline()
            self.assertEqual(t2.import_jsonl(path), 1)
            rows = t2.query()
            self.assertEqual(rows[0]["topic"], "mission.transition")
            self.assertEqual(rows[0]["correlation_id"], "c1")
            self.assertEqual(rows[0]["actor"], "runner")

    def test_topic_counts_and_render(self) -> None:
        t = Timeline()
        t.record(_event("mission.transition", {}))
        t.record(_event("mission.transition", {}, event_id="e2"))
        t.record(_event("task.started", {}, event_id="e3"))
        counts = dict(t.topic_counts())
        self.assertEqual(counts["mission.transition"], 2)
        self.assertIn("mission.transition", t.render())


class UpdateSweepTests(unittest.TestCase):
    def _manager(self, tmp: str) -> UpdateManager:
        home = Path(tmp) / "home"
        snaps = SnapshotManager(home, Path(tmp) / "db.sqlite")
        return UpdateManager(Path(tmp) / "repo",
                             snaps,
                             health_checks=[lambda: (True, "ok")])

    def test_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._manager(tmp)
            report = mgr.dry_run()
            self.assertTrue(report.ok)
            self.assertTrue(any(s["step"].startswith("health:")
                                for s in report.steps))

    def test_dry_run_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            snaps = SnapshotManager(home, Path(tmp) / "db.sqlite")
            mgr = UpdateManager(Path(tmp) / "repo", snaps,
                                health_checks=[lambda: (False, "kaput")])
            report = mgr.dry_run()
            self.assertFalse(report.ok)
            self.assertIn("kaput", report.error)

    def test_run_no_pull_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._manager(tmp)
            report = mgr.run(pull=False)
            self.assertTrue(report.ok, report.error)
            self.assertTrue(report.pre_update_snapshot)
            # journal + history recorded
            self.assertTrue(mgr.journal_path.exists())
            self.assertEqual(len(mgr.history()), 1)
            self.assertTrue(mgr.history()[0]["ok"])
            self.assertIn("✓ OK", report.render())

    def test_run_health_failure_rolls_back_and_quarantines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            snaps = SnapshotManager(home, Path(tmp) / "db.sqlite")
            repo = Path(tmp) / "repo"
            repo.mkdir()
            mgr = UpdateManager(repo, snaps,
                                health_checks=[lambda: (False, "broken")],
                                git_bin="git")
            report = mgr.run(pull=False)
            self.assertFalse(report.ok)
            self.assertTrue(report.rolled_back)
            self.assertIn("broken", report.error)
            self.assertIn("↩ rolled back", report.render())
            # no git repo → no sha → nothing quarantined
            self.assertEqual(mgr.quarantine_list(), [])

    def test_quarantine_add_list_clear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._manager(tmp)
            mgr._quarantine_add("abc123", "health failed")
            mgr._quarantine_add("abc123", "health failed")  # idempotent
            listed = mgr.quarantine_list()
            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0]["sha"], "abc123")
            self.assertEqual(mgr.quarantine_clear("abc123"), 1)
            self.assertEqual(mgr.quarantine_list(), [])

    def test_recover_interrupted_rolls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            snaps = SnapshotManager(home, Path(tmp) / "db.sqlite")
            snap = snaps.create(label="pre")
            mgr = UpdateManager(Path(tmp) / "repo", snaps,
                                health_checks=[lambda: (True, "ok")])
            # fake an interrupted run that reached the dangerous "migrate" step
            mgr._journal({"run_id": "deadbeef", "event": "start",
                          "ts": time.time()})
            mgr._journal({"run_id": "deadbeef", "event": "step",
                          "step": "snapshot", "snapshot": snap.id,
                          "ts": time.time()})
            mgr._journal({"run_id": "deadbeef", "event": "step",
                          "step": "migrate", "ok": True, "ts": time.time()})
            receipt = mgr.recover_interrupted()
            self.assertTrue(receipt["recovered"])
            self.assertEqual(receipt["action"], "rolled_back")
            self.assertEqual(receipt["restored_snapshot"], snap.id)
            # second call: already recovered
            receipt2 = mgr.recover_interrupted()
            self.assertFalse(receipt2["recovered"])

    def test_recover_interrupted_early_death_aborts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._manager(tmp)
            mgr._journal({"run_id": "r2", "event": "start", "ts": time.time()})
            mgr._journal({"run_id": "r2", "event": "step", "step": "snapshot",
                          "snapshot": "snap-x", "ts": time.time()})
            receipt = mgr.recover_interrupted()
            self.assertTrue(receipt["recovered"])
            self.assertEqual(receipt["action"], "aborted")

    def test_report_render_and_dict(self) -> None:
        report = UpdateReport(ok=True, seconds=1.5)
        report.record("git_pull", True, "already up to date")
        d = report.to_dict()
        self.assertIn("quarantined", d)
        self.assertIn("✓ git_pull", report.render())


class VerifierSweepTests(unittest.TestCase):
    def test_composite_all(self) -> None:
        v = CompositeVerifier("suite", [
            SimpleNamespace(name="a",
                            verify=lambda t: Verdict(True, "fine")),
            SimpleNamespace(name="b",
                            verify=lambda t: Verdict(False, "bad")),
        ], mode="all")
        verdict = v.verify({})
        self.assertFalse(verdict.passed)
        self.assertIn("FAIL b", verdict.details)
        self.assertIn("1/2 passed", verdict.details)

    def test_composite_any(self) -> None:
        v = CompositeVerifier("suite", [
            SimpleNamespace(name="a",
                            verify=lambda t: Verdict(False, "bad")),
            SimpleNamespace(name="b",
                            verify=lambda t: Verdict(True, "fine")),
        ], mode="any")
        self.assertTrue(v.verify({}).passed)

    def test_composite_bad_mode(self) -> None:
        with self.assertRaises(ValueError):
            CompositeVerifier("x", [], mode="most")

    def test_lint_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.py"
            good.write_text("x = 1\n")
            bad = Path(tmp) / "bad.py"
            bad.write_text("def broken(:\n")
            verdict = LintVerifier().verify({"paths": [tmp]})
            self.assertFalse(verdict.passed)
            self.assertIn("bad.py", verdict.details)
            verdict = LintVerifier().verify({"paths": [str(good)]})
            self.assertTrue(verdict.passed)

    def test_git_clean_verifier(self) -> None:
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["git", "init", "-q", tmp], check=True,
                           timeout=30)
            verdict = GitCleanVerifier().verify({"repo_dir": tmp})
            self.assertTrue(verdict.passed, verdict.details)
            # untracked files are ignored by default
            Path(tmp, "new.txt").write_text("hello")
            verdict = GitCleanVerifier().verify({"repo_dir": tmp})
            self.assertTrue(verdict.passed)
            # staged changes count as dirty
            subprocess.run(["git", "-C", tmp, "add", "new.txt"], check=True,
                           timeout=30)
            verdict = GitCleanVerifier().verify({"repo_dir": tmp})
            self.assertFalse(verdict.passed)
            self.assertIn("new.txt", verdict.details)

    def test_disk_space_verifier(self) -> None:
        verdict = DiskSpaceVerifier().verify({"path": "/", "min_free_mb": 1})
        self.assertTrue(verdict.passed)
        self.assertIn("free", verdict.details)
        verdict = DiskSpaceVerifier().verify({"path": "/",
                                              "min_free_mb": 10**12})
        self.assertFalse(verdict.passed)

    def test_gate_registry(self) -> None:
        reg = gate_registry()
        self.assertIn("pre_update_gates", reg.list())
        self.assertIn("lint", reg.list())
        # default registry contract unchanged
        self.assertEqual(default_registry().list(),
                         ["code_tests", "docs_render"])

    def test_verdict_render(self) -> None:
        rendered = Verdict(False, "line1\nline2",
                           ["art://x"]).render()
        self.assertIn("✗ FAIL", rendered)
        self.assertIn("line1", rendered)
        self.assertIn("artifact: art://x", rendered)


class MissionStateSweepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store, self.db = _mission_store()
        self.addCleanup(self.db.close)
        self.addCleanup(clear_guards)
        self.addCleanup(clear_callbacks)

    def test_guard_veto(self) -> None:
        @guard("RUNNING", "COMPLETED")
        def _needs_evidence(mission, from_s, to_s, note):
            return bool((mission.state or {}).get("evidence"))

        m = self.store.create_new("g")
        transition(self.store, m.id, PLANNED)
        transition(self.store, m.id, RUNNING)
        with self.assertRaises(GuardFailed):
            transition(self.store, m.id, COMPLETED)
        m2 = self.store.get(m.id)
        m2.state["evidence"] = {"x": 1}
        self.store.save(m2)
        transition(self.store, m.id, COMPLETED)
        self.assertEqual(current_state(self.store.get(m.id)), COMPLETED)

    def test_guard_exception_vetoes(self) -> None:
        @guard()
        def _boom(mission, from_s, to_s, note):
            raise RuntimeError("nope")

        m = self.store.create_new("g2")
        with self.assertRaises(GuardFailed):
            transition(self.store, m.id, PLANNED)

    def test_callbacks_fire_in_order(self) -> None:
        calls: list[str] = []

        @on_exit("RUNNING")
        def _ex(mission, from_s, to_s, note):
            calls.append(f"exit:{from_s}")

        @on_enter("VERIFYING")
        def _en(mission, from_s, to_s, note):
            calls.append(f"enter:{to_s}")

        @on_transition
        def _tr(mission, from_s, to_s, note):
            calls.append(f"move:{from_s}->{to_s}")

        m = self.store.create_new("c")
        transition(self.store, m.id, PLANNED)
        transition(self.store, m.id, RUNNING)
        transition(self.store, m.id, VERIFYING)
        self.assertIn("exit:RUNNING", calls)
        self.assertIn("enter:VERIFYING", calls)
        self.assertIn("move:RUNNING->VERIFYING", calls)

    def test_raising_callback_does_not_break_transition(self) -> None:
        @on_enter()
        def _bad(mission, from_s, to_s, note):
            raise RuntimeError("callback boom")

        m = self.store.create_new("cb")
        transition(self.store, m.id, PLANNED)  # must not raise
        self.assertEqual(current_state(self.store.get(m.id)), PLANNED)

    def test_actor_in_log(self) -> None:
        m = self.store.create_new("a")
        transition(self.store, m.id, PLANNED, note="kickoff", actor="runner")
        log = transition_log(self.store.get(m.id))
        self.assertEqual(log[-1]["actor"], "runner")
        self.assertEqual(log[-1]["to"], PLANNED)

    def test_mermaid_and_describe(self) -> None:
        diagram = to_mermaid()
        self.assertIn("stateDiagram-v2", diagram)
        self.assertIn("RUNNING --> VERIFYING", diagram)
        self.assertIn("[*] --> CREATED", diagram)
        text = describe_machine()
        self.assertIn("RUNNING", text)
        self.assertIn("(terminal)", text)

    def test_illegal_still_raises(self) -> None:
        m = self.store.create_new("i")
        with self.assertRaises(InvalidTransition):
            transition(self.store, m.id, COMPLETED)


class ReplaySweepTests(unittest.TestCase):
    def _replay(self, mission_id: str, path: list[str]) -> MissionReplay:
        t = Timeline()
        ts = 1700000000.0
        prev = path[0]
        for i, state in enumerate(path[1:], 1):
            t.record(_event("mission.transition",
                            {"mission_id": mission_id,
                             "from_state": prev, "to_state": state},
                            event_id=f"{mission_id}-{i}", ts=ts + i * 10))
            prev = state
        t.record(_event("mission.verify",
                        {"mission_id": mission_id, "verdict": "pass"},
                        event_id=f"{mission_id}-v", ts=ts + 999))
        return replay_mission(t, mission_id)

    def test_durations(self) -> None:
        r = self._replay("m1", ["CREATED", "PLANNED", "RUNNING", "COMPLETED"])
        dur = r.durations()
        self.assertEqual(dur["transition_count"], 3)
        self.assertAlmostEqual(dur["wall_s"], 20.0)
        self.assertAlmostEqual(dur["per_state_s"]["RUNNING"], 10.0)
        self.assertIsNotNone(dur["longest_leg"])

    def test_compare(self) -> None:
        a = self._replay("a", ["CREATED", "PLANNED", "RUNNING", "COMPLETED"])
        b = self._replay("b", ["CREATED", "PLANNED", "RUNNING", "FAILED"])
        cmp = compare_replays(a, b)
        self.assertEqual(cmp.common_prefix,
                         ["CREATED", "PLANNED", "RUNNING"])
        self.assertEqual(cmp.diverged_at, "RUNNING")
        self.assertEqual(cmp.transition_path_b[-1], "FAILED")
        narrative = cmp.narrative()
        self.assertIn("diverged at: RUNNING", narrative)
        d = cmp.to_dict()
        self.assertEqual(d["replay_a"], "a")

    def test_compare_identical(self) -> None:
        a = self._replay("a", ["CREATED", "PLANNED"])
        b = self._replay("b", ["CREATED", "PLANNED"])
        cmp = compare_replays(a, b)
        self.assertIsNone(cmp.diverged_at)

    def test_to_html(self) -> None:
        r = self._replay("m1", ["CREATED", "PLANNED", "RUNNING"])
        html = r.to_html()
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("m1", html)
        self.assertIn("state transitions", html)


class SnapshotSweepTests(unittest.TestCase):
    def _manager(self, tmp: str) -> SnapshotManager:
        db = Path(tmp) / "live.sqlite"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.commit()
        conn.close()
        return SnapshotManager(Path(tmp) / "home", db)

    def test_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._manager(tmp)
            snap = mgr.create_tagged("before risky thing", "pre-update")
            self.assertIn("pre-update", snap.manifest["tags"])
            mgr.tag(snap.id, "daily")
            self.assertIn("daily", mgr.get(snap.id).manifest["tags"])
            mgr.untag(snap.id, "daily")
            self.assertNotIn("daily", mgr.get(snap.id).manifest["tags"])
            self.assertEqual([s.id for s in mgr.with_tag("pre-update")],
                             [snap.id])
            self.assertIn("pre-update", mgr.render_list())

    def test_prune_keeps_tagged_and_recent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._manager(tmp)
            ids = [mgr.create(f"snap{i}").id for i in range(4)]
            mgr.tag(ids[0], "pre-update")
            victims = mgr.prune(keep_last=2, keep_tags=("pre-update",),
                                dry_run=True)
            self.assertEqual(victims, [ids[1]])
            victims = mgr.prune(keep_last=2, keep_tags=("pre-update",))
            self.assertEqual(victims, [ids[1]])
            remaining = {s.id for s in mgr.list()}
            self.assertEqual(remaining, {ids[0], ids[2], ids[3]})

    def test_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._manager(tmp)
            a = mgr.create("a")
            db = Path(tmp) / "live.sqlite"
            conn = sqlite3.connect(str(db))
            conn.executemany("INSERT INTO t VALUES (?)",
                             [(i,) for i in range(3000)])
            conn.commit()
            conn.close()
            b = mgr.create("b")
            diff = mgr.diff(a.id, b.id)
            self.assertIn("db.sqlite", diff["changed"])
            self.assertGreater(diff["size_delta_bytes"], 0)

    def test_pre_post(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._manager(tmp)
            with mgr.pre_post("deploy", tag="auto") as holder:
                pre = holder["pre"]
                self.assertIn("pre", pre.manifest["tags"])
            self.assertIsNotNone(holder["post"])
            assert holder["post"] is not None
            self.assertIn("post", holder["post"].manifest["tags"])
            # failure path still takes a post snapshot tagged failed
            with self.assertRaises(RuntimeError):
                with mgr.pre_post("oops", tag="auto") as holder2:
                    raise RuntimeError("boom")
            assert holder2["post"] is not None
            self.assertIn("failed", holder2["post"].manifest["tags"])

    def test_export_import_tar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._manager(tmp)
            snap = mgr.create_tagged("ship it", "manual")
            tarball = Path(tmp) / "snap.tar.gz"
            mgr.export_tar(snap.id, tarball)
            self.assertTrue(tarball.exists())
            mgr.delete(snap.id)
            imported = mgr.import_tar(tarball)
            self.assertEqual(imported.id, snap.id)
            self.assertIn("manual", imported.manifest["tags"])


class ResourcesSweepTests(unittest.TestCase):
    def test_render(self) -> None:
        mgr = ResourceManager()
        rendered = mgr.render()
        self.assertIn("resources", rendered)
        self.assertIn("cpu", rendered)
        self.assertIn("mem", rendered)
        self.assertIn("overall", rendered)
        # gauge characters present
        self.assertTrue("█" in rendered or "░" in rendered)

    def test_summary_line(self) -> None:
        mgr = ResourceManager()
        line = mgr.summary_line()
        self.assertIn("cpu", line)
        self.assertIn("mem", line)

    def test_alert_edge_triggered(self) -> None:
        mgr = ResourceManager()
        events: list[dict] = []
        mgr.on_alert("overall", 0.0, events.append)
        sample = mgr.sample()
        # fires on crossing above 0.0 (0.0 -> firing)
        self.assertTrue(any(e["firing"] for e in events))
        n = len(events)
        mgr._eval_alerts(sample, mgr.pressure(sample))
        self.assertEqual(len(events), n)  # no re-fire without a crossing
        self.assertTrue(mgr.clear_alert("overall:0:above"))

    def test_alert_below_threshold(self) -> None:
        mgr = ResourceManager()
        events: list[dict] = []
        # disk can never be >= 100% used, so "below 100" always fires
        mgr.on_alert("disk_percent", 100.0, events.append, above=False)
        sample = mgr.sample()
        self.assertIsNotNone(sample.disk_percent)
        mgr._eval_alerts(sample, {})
        self.assertTrue(events and events[0]["firing"])

    def test_consult_still_works(self) -> None:
        mgr = ResourceManager()
        advice = mgr.consult()
        self.assertIn("ok", advice)


class AdaptersSweepTests(unittest.TestCase):
    def test_generic_adapter(self) -> None:
        adapter = os_adapters.GenericSessionAdapter(kind="miniapp",
                                                    frontend="telegram")
        session = adapter.to_os_session({"session_id": "abc", "user_id": 7})
        self.assertEqual(session.frontend, "telegram")
        self.assertEqual(session.conversation_id, "abc")
        self.assertEqual(session.state["kind"], "miniapp")
        self.assertEqual(session.state["user_id"], 7)

    def test_generic_adapter_never_raises(self) -> None:
        adapter = os_adapters.GenericSessionAdapter()
        session = adapter.to_os_session(None)
        self.assertTrue(session.state.get("degraded"))

    def test_adapter_kinds(self) -> None:
        kinds = os_adapters.adapter_kinds()
        self.assertIn("account", kinds)
        self.assertIn("voice", kinds)
        self.assertIn("generic", kinds)
        self.assertEqual(len(os_adapters.ADAPTERS), 3)


if __name__ == "__main__":
    unittest.main()
