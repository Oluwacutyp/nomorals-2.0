"""Wave E gap closures: bounded _send_async, real CLI commands, the
research->partner lexicon acquisition loop, and persisted router
telemetry for `nm mind`.

(a) CoreMind._send_async is semaphore-bounded (mind.max_inflight,
    default 8): concurrent sends never exceed the bound, and a send that
    cannot take a slot within the acquire timeout is shed — logged,
    counted, job marked failed with an explicit note.
(b) The six former stub CLI commands are real: book (BookForge), hub
    (MediaHub), arena (Arena), trial (TrialFlow), skill (SkillLibrary),
    simulate (SandboxSimulator). No _cmd_stub remains.
(c) The lexicon acquisition loop: scored research findings ->
    pull_scored_findings -> per-category acquire (scored, versioned) ->
    LexiconFeed.reload(), driven from the research dispatch.
(d) Router telemetry persists per-route decision counts + last
    plan_error (migration 65); `nm mind` displays them.
"""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.coremind import (
    INFLIGHT_ACQUIRE_TIMEOUT_S,
    MAX_INFLIGHT_DEFAULT,
    CoreMind,
    Intent,
)
from nomorals.partner.lexicon_acquire import (
    LOOP_THRESHOLD,
    feed_partner_lexicon,
    pull_scored_findings,
)
from nomorals.partner.lexicon_feed import LEXICON_MODULE, LexiconFeed
from nomorals.storage.db import Database
from nomorals.storage.router_telemetry import (
    record_model_check,
    record_plan_error,
    record_route,
    snapshot,
)


def _db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


class _FakeSettings:
    def __init__(self, mind=None, workspace_dir=""):
        self._mind = mind
        self._workspace_dir = workspace_dir

    def resolve(self, key):
        raise RuntimeError("no settings in tests")

    def __getattr__(self, name):
        if name == "mind":
            if self._mind is None:
                raise AttributeError(name)
            return self._mind
        if name == "workspace_dir":
            return self._workspace_dir
        raise AttributeError(name)


class _FakeContext:
    def __init__(self, db=None, mind=None, workspace_dir=""):
        self.settings = _FakeSettings(mind, workspace_dir)
        self.extras = {}
        self.memory = None
        self.router = None
        self.db = db


def _mind_settings(max_inflight, timeout):
    return SimpleNamespace(max_inflight=max_inflight,
                           inflight_acquire_timeout_s=timeout)


# ── (a) bounded _send_async ──────────────────────────────────────────────


class SendAsyncBoundTest(unittest.TestCase):
    def test_concurrent_sends_never_exceed_bound(self):
        mind = CoreMind(_FakeContext(mind=_mind_settings(2, 5.0)))
        peak = 0
        live = 0
        guard = threading.Lock()
        entered = threading.Event()

        def job():
            nonlocal peak, live
            with guard:
                live += 1
                peak = max(peak, live)
            entered.set()
            time.sleep(0.3)
            with guard:
                live -= 1
            return "done"

        threads = [
            threading.Thread(target=mind._send_async,
                             args=("k:console", job, f"j{i}", "started"))
            for i in range(6)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        # let the worker threads finish so the semaphore drains
        deadline = time.time() + 10
        while mind._inflight_now and time.time() < deadline:
            time.sleep(0.05)
        self.assertLessEqual(peak, 2)
        self.assertEqual(mind._send_started, 6)
        self.assertEqual(mind._send_shed, 0)

    def test_send_shed_when_no_slot_within_timeout(self):
        mind = CoreMind(_FakeContext(mind=_mind_settings(1, 0.05)))
        release = threading.Event()

        def slow_job():
            release.wait(timeout=5)
            return "slow done"

        mind._send_async("k:console", slow_job, "slow", "started")
        # wait until the slow job holds the only slot
        deadline = time.time() + 5
        while mind._inflight_now < 1 and time.time() < deadline:
            time.sleep(0.02)
        self.assertEqual(mind._inflight_now, 1)

        with self.assertLogs("nomorals.coremind", level="WARNING") as logs:
            note = mind._send_async("k:console", lambda: "x", "shed1",
                                    "started")
        release.set()
        self.assertEqual(mind._send_shed, 1)
        self.assertIn("shed", note)
        self.assertTrue(any("shed" in m for m in logs.output))
        deadline = time.time() + 5
        while mind._inflight_now and time.time() < deadline:
            time.sleep(0.05)
        self.assertEqual(mind._inflight_now, 0)

    def test_defaults_when_no_mind_settings(self):
        mind = CoreMind(_FakeContext())
        self.assertEqual(mind._max_inflight, MAX_INFLIGHT_DEFAULT)
        self.assertEqual(mind._inflight_timeout, INFLIGHT_ACQUIRE_TIMEOUT_S)
        self.assertEqual(MAX_INFLIGHT_DEFAULT, 8)

    def test_status_reports_inflight_metrics(self):
        mind = CoreMind(_FakeContext(mind=_mind_settings(4, 5.0)))
        text = mind.status()
        self.assertIn("background jobs: 0/4 in flight", text)
        self.assertIn("shed 0", text)


# ── (b) real CLI commands ────────────────────────────────────────────────


def _ns(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


class NoStubsLeftTest(unittest.TestCase):
    def test_cmd_stub_is_gone(self):
        import nomorals.cli as cli_mod

        self.assertFalse(hasattr(cli_mod, "_cmd_stub"))
        for name in ("book", "hub", "arena", "trial", "skill", "simulate"):
            self.assertTrue(hasattr(cli_mod, f"_cmd_{name}"),
                            f"_cmd_{name} missing")


class BookCommandTest(unittest.TestCase):
    def test_list_empty(self):
        from nomorals.cli import _cmd_book

        with tempfile.TemporaryDirectory() as tmp:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _cmd_book(_ns(action="list", json=False),
                               _FakeContext(workspace_dir=tmp))
        self.assertEqual(rc, 0)
        self.assertIn("no books", buf.getvalue())

    def test_create_needs_topic(self):
        from nomorals.cli import _cmd_book

        with tempfile.TemporaryDirectory() as tmp:
            rc = _cmd_book(_ns(action="create", topic="", chapters=5, words=2000,
                                no_research=True, json=False),
                           _FakeContext(workspace_dir=tmp))
        self.assertEqual(rc, 2)


class HubCommandTest(unittest.TestCase):
    def test_status_runs_against_media_hub(self):
        from nomorals.cli import _cmd_hub

        with mock.patch("nomorals.media.MediaHub") as hub_cls:
            hub = hub_cls.return_value
            hub.status.return_value = {"queue": [], "state": "idle"}
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _cmd_hub(_ns(mode="status", json=True), _FakeContext())
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["state"], "idle")

    def test_styles_lists_song_styles(self):
        from nomorals.cli import _cmd_hub

        with mock.patch("nomorals.media.MediaHub") as hub_cls:
            hub = hub_cls.return_value
            hub.styles.return_value = {
                "lofi": {"label": "Lofi", "tempo": [70, 90],
                         "mode": "minor", "energy": "low"}}
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _cmd_hub(_ns(mode="styles", json=False), _FakeContext())
        self.assertEqual(rc, 0)
        self.assertIn("lofi", buf.getvalue())

    def test_song_mode_needs_query(self):
        from nomorals.cli import _cmd_hub

        rc = _cmd_hub(_ns(mode="song", query="", style="pop", platform="",
                           no_play=False, json=False),
                      _FakeContext())
        self.assertEqual(rc, 2)


class ArenaCommandTest(unittest.TestCase):
    def _ctx(self):
        ctx = _FakeContext(db=_db())
        ctx.settings = SimpleNamespace(home="~/.nomorals")
        return ctx

    def test_status_shows_build_counts(self):
        from nomorals.cli import _cmd_arena

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_arena(_ns(arena_command="status", json=False),
                            self._ctx())
        self.assertEqual(rc, 0)
        self.assertIn("arena status:", buf.getvalue())

    def test_approve_unknown_build_fails_cleanly(self):
        from nomorals.cli import _cmd_arena

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_arena(_ns(arena_command="approve", build_id="nope",
                                json=False),
                            self._ctx())
        self.assertEqual(rc, 1)
        self.assertIn("no arena build", buf.getvalue())


class TrialCommandTest(unittest.TestCase):
    def _ctx(self):
        ctx = _FakeContext()
        ctx.settings = SimpleNamespace(home="~/.nomorals")
        return ctx

    def test_list_empty(self):
        from nomorals.cli import _cmd_trial

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_trial(_ns(trial_command="list", json=False), self._ctx())
        self.assertEqual(rc, 0)
        self.assertIn("no stored trial accounts", buf.getvalue())

    def test_save_needs_all_fields(self):
        from nomorals.cli import _cmd_trial

        rc = _cmd_trial(_ns(trial_command="save", platform="x", login="",
                            password="", note="", json=False),
                       self._ctx())
        self.assertEqual(rc, 2)


class SkillCommandTest(unittest.TestCase):
    def _ctx(self):
        return _FakeContext(db=_db())

    def _ns_skill(self, action, **kw):
        base = dict(action=action, name="", description="", body="",
                    kind="strategy", tags="", pruned=False, json=False)
        base.update(kw)
        return _ns(**base)

    def test_list_create_show_delete_roundtrip(self):
        from nomorals.cli import _cmd_skill

        ctx = self._ctx()
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_skill(self._ns_skill(
                "create", name="wave_e_probe", description="probe skill",
                body="when X, do Y", kind="strategy", tags="probe,wave-e"), ctx)
        self.assertEqual(rc, 0)
        self.assertIn("saved skill wave_e_probe", buf.getvalue())

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_skill(self._ns_skill("list"), ctx)
        self.assertEqual(rc, 0)
        self.assertIn("wave_e_probe", buf.getvalue())

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_skill(self._ns_skill("show", name="wave_e_probe"), ctx)
        self.assertEqual(rc, 0)
        self.assertIn("when X, do Y", buf.getvalue())

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_skill(self._ns_skill("delete", name="wave_e_probe"), ctx)
        self.assertEqual(rc, 0)
        self.assertIn("deleted skill wave_e_probe", buf.getvalue())

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_skill(self._ns_skill("list"), ctx)
        self.assertEqual(rc, 0)
        self.assertNotIn("wave_e_probe", buf.getvalue())

    def test_show_unknown_skill_fails(self):
        from nomorals.cli import _cmd_skill

        rc = _cmd_skill(self._ns_skill("show", name="nope"), self._ctx())
        self.assertEqual(rc, 1)

    def test_stats_and_prune(self):
        from nomorals.cli import _cmd_skill

        ctx = self._ctx()
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_skill(self._ns_skill("stats", json=True), ctx)
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertIn("total", payload)

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_skill(self._ns_skill("prune", json=True), ctx)
        self.assertEqual(rc, 0)

    def test_create_needs_name(self):
        from nomorals.cli import _cmd_skill

        rc = _cmd_skill(self._ns_skill("create"), self._ctx())
        self.assertEqual(rc, 2)


class SimulateCommandTest(unittest.TestCase):
    def test_risk_classifies_without_running(self):
        from nomorals.cli import _cmd_simulate

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_simulate(_ns(simulate_action="risk",
                                   cmd="rm -rf /tmp/x", json=False),
                               _FakeContext())
        self.assertEqual(rc, 0)
        self.assertIn("high", buf.getvalue())

    def test_dry_run_executes_nothing(self):
        from nomorals.cli import _cmd_simulate

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_simulate(_ns(simulate_action="dry-run",
                                   cmd="echo hello", json=True),
                               _FakeContext())
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertTrue(payload["dry_run"])
        self.assertEqual(payload["risk"]["level"], "low")

    def test_run_executes_in_sandbox(self):
        from nomorals.cli import _cmd_simulate

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_simulate(_ns(simulate_action="run",
                                   cmd="echo sim-ok", confirm=False,
                                   timeout=30.0, json=True),
                               _FakeContext())
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertTrue(payload["ok"])
        self.assertIn("sim-ok", payload["stdout"])

    def test_high_risk_run_blocked_without_confirm(self):
        from nomorals.cli import _cmd_simulate

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_simulate(_ns(simulate_action="run",
                                   cmd="rm -rf /tmp/x", confirm=False,
                                   timeout=30.0, json=True),
                               _FakeContext())
        self.assertEqual(rc, 1)
        payload = json.loads(buf.getvalue())
        self.assertFalse(payload["ok"])


# ── (c) lexicon acquisition loop ─────────────────────────────────────────


def _finding(claim, confidence=0.9):
    return SimpleNamespace(claim=claim, angle="a", sources=[],
                           confidence=confidence)


class LexiconLoopTest(unittest.TestCase):
    def test_pull_keeps_only_scored_findings(self):
        findings = [
            _finding("good claim", 0.9),
            _finding("weak claim", 0.1),
            _finding("", 0.9),
            {"claim": "dict claim", "confidence": 0.8},
            {"claim": "low dict", "confidence": 0.2},
        ]
        kept = pull_scored_findings(findings, min_confidence=0.4)
        self.assertEqual(len(kept), 2)

    def test_loop_feeds_partner_categories_scored_and_versioned(self):
        db = _db()
        findings = [
            _finding('She always says "tell me everything" when listening', 0.9),
            _finding('Her opener is "okay so" before big thoughts', 0.85),
            _finding("low confidence noise here", 0.1),
        ]
        report = feed_partner_lexicon(db, findings, source="test")
        self.assertEqual(report["findings"], 2)  # the weak one was pulled out
        self.assertGreater(report["candidates"], 0)
        self.assertGreater(len(report["added"]), 0)
        self.assertGreater(report["version"], 0)
        # versioned + visible through the partner feed (the reload step)
        feed = LexiconFeed(db)
        reloaded = feed.reload()
        self.assertEqual(reloaded["version"], report["version"])
        self.assertEqual(reloaded["module"], LEXICON_MODULE)
        total = sum(reloaded["categories"].values())
        self.assertEqual(total, len(report["added"]))

    def test_loop_is_idempotent_on_rerun(self):
        db = _db()
        findings = [_finding('She says "tell me everything" a lot', 0.9)]
        first = feed_partner_lexicon(db, findings, source="test")
        second = feed_partner_lexicon(db, findings, source="test")
        self.assertGreater(len(first["added"]), 0)
        self.assertEqual(second["added"], [])  # duplicates rejected

    def test_loop_never_raises_without_db(self):
        report = feed_partner_lexicon(None, [_finding("x", 0.9)])
        self.assertEqual(report["reason"], "no db")

    def test_loop_threshold_documented(self):
        self.assertEqual(LOOP_THRESHOLD, 0.35)


# ── (d) router telemetry + nm mind ───────────────────────────────────────


class RouterTelemetryTest(unittest.TestCase):
    def test_migration_65_creates_table(self):
        db = _db()
        tables = {t.lower() for t in db.tables()}
        self.assertIn("coremind_telemetry", tables)

    def test_route_counts_persist_per_route(self):
        db = _db()
        record_route(db, "research_swarm")
        record_route(db, "research_swarm")
        record_route(db, "coding")
        snap = snapshot(db)
        self.assertEqual(snap["routes"]["research_swarm"], 2)
        self.assertEqual(snap["routes"]["coding"], 1)

    def test_model_check_counts(self):
        db = _db()
        record_model_check(db)
        record_model_check(db, timed_out=True)
        snap = snapshot(db)
        self.assertEqual(snap["model_consults"], 2)
        self.assertEqual(snap["model_timeouts"], 1)

    def test_last_plan_error_with_timestamp(self):
        db = _db()
        before = time.time()
        record_plan_error(db, "no LLM router configured — used template plan",
                          route="devon")
        snap = snapshot(db)
        err = snap["last_plan_error"]
        self.assertIsNotNone(err)
        self.assertIn("template plan", err["error"])
        self.assertEqual(err["route"], "devon")
        self.assertGreaterEqual(err["at"], before)
        # a second error supersedes the first
        record_plan_error(db, "second failure", route="coding")
        err2 = snapshot(db)["last_plan_error"]
        self.assertIn("second failure", err2["error"])

    def test_snapshot_empty_without_db(self):
        snap = snapshot(None)
        self.assertEqual(snap["routes"], {})
        self.assertIsNone(snap["last_plan_error"])

    def test_telemetry_never_raises_on_broken_db(self):
        broken = object()
        record_route(broken, "x")
        record_plan_error(broken, "x")
        self.assertEqual(snapshot(broken)["routes"], {})

    def test_decide_records_route_decision(self):
        db = _db()
        mind = CoreMind(_FakeContext(db=db))
        mind._record_route(Intent("research", 0.9, route="research_swarm",
                                 why="test"))
        mind._record_route(Intent("build", 0.9, route="coding", why="test"))
        snap = snapshot(db)
        self.assertEqual(snap["routes"]["research_swarm"], 1)
        self.assertEqual(snap["routes"]["coding"], 1)

    def test_record_plan_error_helper(self):
        db = _db()
        mind = CoreMind(_FakeContext(db=db))
        mind.record_plan_error("heuristic fallback used", route="devon")
        err = snapshot(db)["last_plan_error"]
        self.assertIn("heuristic fallback", err["error"])


class MindCommandTest(unittest.TestCase):
    def test_mind_json_shows_persisted_telemetry(self):
        from nomorals.cli import _cmd_mind

        db = _db()
        record_route(db, "research_swarm")
        record_route(db, "research_swarm")
        record_plan_error(db, "template plan used", route="devon")
        ctx = _FakeContext(db=db)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_mind(_ns(json=True), ctx)
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["router_calls"]["per_route"]["research_swarm"], 2)
        self.assertEqual(payload["router_calls"]["total"], 2)
        self.assertIn("template plan", payload["last_plan_error"]["error"])

    def test_mind_text_shows_telemetry(self):
        from nomorals.cli import _cmd_mind

        db = _db()
        record_route(db, "coding")
        ctx = _FakeContext(db=db)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_mind(_ns(json=False), ctx)
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("router calls: 1 total", out)
        self.assertIn("coding: 1×", out)
        self.assertIn("last plan_error: none recorded", out)


# ── research-loop CLI (follow-up item) ───────────────────────────────────


class ResearchLoopCommandTest(unittest.TestCase):
    def _ctx(self):
        return _FakeContext(db=_db())

    def test_status_shows_job_and_gates(self):
        from nomorals.cli import _cmd_research_loop

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_research_loop(
                _ns(action="status", topic="", topics="", max_topics=0,
                    json=False),
                self._ctx())
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("research loop:", out)
        self.assertIn("feature=off", out)  # flag defaults off

    def test_topics_set_topics_roundtrip(self):
        from nomorals.cli import _cmd_research_loop

        ctx = self._ctx()
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_research_loop(
                _ns(action="set_topics", topic="", topics="fusion reactors, tidal lagoons",
                    max_topics=0, json=False),
                ctx)
        self.assertEqual(rc, 0)
        self.assertIn("2", buf.getvalue())

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_research_loop(
                _ns(action="topics", topic="", topics="", max_topics=0,
                    json=False),
                ctx)
        self.assertEqual(rc, 0)
        self.assertIn("fusion reactors", buf.getvalue())

    def test_ensure_enable_disable(self):
        from nomorals.cli import _cmd_research_loop

        ctx = self._ctx()
        ns = lambda a: _ns(action=a, topic="", topics="", max_topics=0,
                           json=True)
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(_cmd_research_loop(ns("ensure"), ctx), 0)
            ensure_out = json.loads(buf.getvalue())
        self.assertTrue(ensure_out.get("scheduled") or
                        ensure_out.get("already_scheduled"))

        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(_cmd_research_loop(ns("disable"), ctx), 0)
            self.assertFalse(json.loads(buf.getvalue())["enabled"])
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(_cmd_research_loop(ns("enable"), ctx), 0)
            self.assertTrue(json.loads(buf.getvalue())["enabled"])

    def test_run_defers_when_feature_off(self):
        from nomorals.cli import _cmd_research_loop

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_research_loop(
                _ns(action="run", topic="tidal lagoons", topics="",
                    max_topics=0, json=True),
                self._ctx())
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertTrue(payload["deferred"])
        self.assertIn("feature 'research' off", payload["skipped_reason"])

    def test_run_needs_topic(self):
        from nomorals.cli import _cmd_research_loop

        rc = _cmd_research_loop(
            _ns(action="run", topic="", topics="", max_topics=0, json=False),
            self._ctx())
        self.assertEqual(rc, 2)

    def test_status_section_in_nm_status(self):
        from nomorals.cli import _cmd_status

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_status(_ns(json=True), self._ctx())
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertIn("research_loop", payload["sections"])
        self.assertIn("feature_research", payload["sections"]["research_loop"])


class ResearchFeatureFlagTest(unittest.TestCase):
    def test_flag_defaults_off_without_db(self):
        from nomorals.agents.features import feature_enabled

        self.assertFalse(feature_enabled(_FakeContext(), "research"))

    def test_flag_roundtrip_through_kv(self):
        from nomorals.agents.features import FeatureRegistry, feature_enabled

        db = _db()
        reg = FeatureRegistry(db)
        self.assertFalse(reg.get("research"))
        self.assertTrue(reg.set("research", True))
        self.assertTrue(reg.get("research"))
        self.assertTrue(feature_enabled(_FakeContext(db=db), "research"))

    def test_boot_block_uses_feature_flag_not_settings(self):
        import inspect

        # Wave H3: PartnerRuntime.start() lives in the partner subpackage now;
        # the facade (nomorals.agents.partner_runtime) only re-exports it.
        from nomorals.agents.partner import runtime as partner_runtime_mod

        src = inspect.getsource(partner_runtime_mod)
        self.assertNotIn('getattr(self.settings, "research", None)', src)
        self.assertIn('feature_enabled', src)


if __name__ == "__main__":
    unittest.main()
