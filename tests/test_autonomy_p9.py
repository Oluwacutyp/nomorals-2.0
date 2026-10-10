"""Phase 9 Slice B tests — god-tier autonomy core.

Real tests for breakable behavior:
* idle monitor reads the SAME database the activity writers use
* coordinator ticks each organ exactly once per cycle (no event loss),
  journals the cycle, and emits a completion event
* weakness detection → research handoff uses a directive kind the research
  organ actually drains; recovery auto-resolves; approval queues a build
* presence enforces the serendipity cooldown and dedups research nudges
* ledger failure-rate telemetry is correct
* partner autonomy learns from owner replies
"""

import sqlite3
import time
from types import SimpleNamespace

import pytest

from nomorals.autonomy import idle as _idle
from nomorals.autonomy import weakness as _weakness
from nomorals.autonomy import patterns as _patterns
from nomorals.autonomy import presence as _presence
from nomorals.autonomy.coordinator import IdleCoordinator
from nomorals.agents.autonomy_ledger import AutonomyLedger, record_ledger
from nomorals.core.events import Event, global_bus
from nomorals import organs as _organs


@pytest.fixture
def db():
    # A real Database — the organs store and the ledger need .query() and
    # .transaction(), which raw sqlite3 connections don't have.
    from nomorals.storage.db import Database
    conn = Database(":memory:")
    yield conn
    conn.close()


# ── idle: the DB-unification fix ──────────────────────────────────────────

class TestIdleDbUnification:
    def test_workspace_db_resolves_directory(self, tmp_path):
        from nomorals.storage.db import Database
        db = _idle.workspace_db(tmp_path)
        assert isinstance(db, Database)
        assert (tmp_path / "nomorals.db").exists()
        db.close()

    def test_workspace_db_accepts_file_path(self, tmp_path):
        from nomorals.storage.db import Database
        target = tmp_path / "custom.sqlite"
        db = _idle.workspace_db(target)
        assert isinstance(db, Database)
        assert target.exists()
        db.close()

    def test_monitor_sees_writer_activity(self, tmp_path):
        """The core regression: writers and monitor share one store."""
        db = _idle.workspace_db(tmp_path)
        try:
            _idle.note_activity(db)  # what the runtime's message path does
            mon = _idle.IdleMonitor(tmp_path, idle_seconds=3600)
            assert mon.check_once() == "unchanged"  # active, not idle
            state = _idle.idle_state(db)
            assert state["idle"] is False
            assert state["last_activity_ts"] > 0
        finally:
            db.close()

    def test_monitor_detects_idle_end_to_end(self, tmp_path):
        db = _idle.workspace_db(tmp_path)
        try:
            fired = []
            _idle.note_activity(db, ts=time.time() - 7200)
            mon = _idle.IdleMonitor(
                tmp_path, idle_seconds=60, on_idle=lambda s: fired.append(s))
            assert mon.check_once() == "idle"
            assert len(fired) == 1
            assert fired[0] > 7000
            # Activity resumes → active again.
            _idle.note_activity(db)
            assert mon.check_once() == "active"
            state = _idle.idle_state(db)
            assert state["idle"] is False
        finally:
            db.close()


# ── coordinator: orchestration without event loss ─────────────────────────

class TestCoordinator:
    def test_cycle_journals_and_emits(self, tmp_path):
        """Full cycle: each tick runs once, ledger gets start/end, and a
        system.idle_cycle event fires."""
        events = []
        sub_id = global_bus.subscribe(
            "system.idle_cycle", lambda e: events.append(e), sync=True)

        coord = IdleCoordinator(tmp_path)
        coord._idle = True
        # Stub the heavy organ ticks — we test orchestration here, not
        # the research pipeline itself.
        calls = []

        def _fake(name):
            def _tick(db, started):
                calls.append(name)
                return {"ok": True, "seconds": 0.0}
            return _tick

        coord._tick_research = _fake("research")
        coord._tick_wisdom = _fake("wisdom")
        coord._tick_memory = _fake("memory")
        coord._tick_presence = _fake("presence")
        coord._tick_weakness = _fake("weakness")
        coord._tick_patterns = _fake("patterns")

        try:
            coord._maintenance_cycle()
        finally:
            global_bus.unsubscribe(sub_id)

        assert calls == ["research", "wisdom", "memory", "presence",
                         "weakness", "patterns"]
        assert len(events) == 1
        assert events[0].topic == "system.idle_cycle"
        assert set(events[0].data["steps"]) == set(calls)

        # Ledger: cycle_start + cycle_end, under system "idle".
        db = _idle.workspace_db(tmp_path)
        try:
            ledger = AutonomyLedger(db)
            idle_rows = ledger.recent(system="idle", limit=10)
            kinds = {r["kind"] for r in idle_rows}
            assert "cycle_start" in kinds
            assert "cycle_end" in kinds
        finally:
            db.close()

    def test_second_cycle_skipped_when_running(self, tmp_path):
        coord = IdleCoordinator(tmp_path)
        coord._idle = True
        coord._tick_research = lambda db, s: {"ok": True}
        coord._tick_wisdom = lambda db, s: {"ok": True}
        coord._tick_memory = lambda db, s: {"ok": True}
        coord._tick_presence = lambda db, s: {"ok": True}
        coord._tick_weakness = lambda db, s: {"ok": True}
        coord._tick_patterns = lambda db, s: {"ok": True}
        # Hold the lock to simulate a cycle already in flight.
        assert coord._cycle_lock.acquire(blocking=False)
        try:
            coord._maintenance_cycle()  # must skip, not run twice
        finally:
            coord._cycle_lock.release()
        assert coord._cycles == 0

    def test_patterns_tick_runs_for_real(self, tmp_path):
        """The cheap hygiene tick runs without stubs."""
        db = _idle.workspace_db(tmp_path)
        try:
            _patterns.record_interests(
                db, "ancient pottery", ts=time.time() - 60 * 86400)
            coord = IdleCoordinator(tmp_path)
            coord._idle = True
            out = coord._tick_patterns(db, time.time())
            assert out["ok"] is True
            assert out["interests_dropped"] >= 1
        finally:
            db.close()

    def test_memory_tick_consolidates(self, tmp_path):
        """Memory consolidation runs through the coordinator path."""
        db = _idle.workspace_db(tmp_path)
        try:
            db.migrate()  # production runs migrations at boot
            coord = IdleCoordinator(tmp_path)
            coord._idle = True
            out = coord._tick_memory(db, time.time())
            assert out["ok"] is True
            assert "episodes" in out
        finally:
            db.close()


# ── weakness: the self-healing chain ──────────────────────────────────────

class TestWeaknessChain:
    def test_detect_routes_to_research_directive_gap(self, db):
        """Three sightings → researching + a directive.gap organ event the
        research organ's tick actually drains (not a dead custom kind)."""
        for i in range(3):
            wid = _weakness.report_weakness(
                db, "tool_failure", "flaky_tool", {"error": f"boom {i}"})
        row = db.execute(
            "SELECT status FROM weaknesses WHERE id = ?", (wid,)).fetchone()
        assert row[0] == "researching"

        events = _organs.drain(db, dst="research")
        gaps = [e for e in events if e["kind"] == "directive.gap"]
        assert len(gaps) == 1
        assert "flaky_tool" in gaps[0]["payload"]["question"]
        assert gaps[0]["payload"]["origin"] == "weakness.investigation"

    def test_investigation_question_framing(self, db):
        wid = _weakness.report_weakness(
            db, "capability_gap", "quantum teleportation")
        assert _weakness.investigate_weakness(db, wid) is True
        events = _organs.drain(db, dst="research")
        q = [e for e in events if e["kind"] == "directive.gap"][0][
            "payload"]["question"]
        assert "quantum teleportation" in q

    def test_tool_result_recovery_auto_resolves(self, db):
        for _ in range(3):
            _weakness.record_tool_result(db, "wobbly", False, "timeout")
        items = _weakness.open_weaknesses(db)
        assert any(i["subject"] == "wobbly" for i in items)

        out = {}
        for _ in range(5):
            out = _weakness.record_tool_result(db, "wobbly", True)
        assert out["ok_streak"] == 5
        assert "auto_resolved" in out
        # The case is gone from the open list.
        items = _weakness.open_weaknesses(db)
        assert not any(i["subject"] == "wobbly" for i in items)

    def test_failure_resets_ok_streak(self, db):
        _weakness.record_tool_result(db, "wobbly", True)
        _weakness.record_tool_result(db, "wobbly", True)
        _weakness.record_tool_result(db, "wobbly", False, "boom")
        health = _weakness.tool_health(db, "wobbly")
        assert health["ok_streak"] == 0
        assert health["fail_streak"] == 1

    def test_approve_queues_sandbox_build(self, db):
        events = []
        sub_id = global_bus.subscribe(
            "weakness.approved", lambda e: events.append(e), sync=True)
        try:
            wid = _weakness.report_weakness(db, "tool_failure", "x")
            _weakness.record_proposal(db, wid, "fix: add retry with backoff")
            assert _weakness.approve_weakness(db, wid) is True

            assert len(events) == 1
            assert events[0].data["weakness_id"] == wid

            sched_events = _organs.drain(db, dst="scheduler")
            builds = [e for e in sched_events
                      if e["kind"] == "weakness.sandbox_build"]
            assert len(builds) == 1
            assert builds[0]["payload"]["sandbox"] is True
            assert "retry" in builds[0]["payload"]["proposal"]
        finally:
            global_bus.unsubscribe(sub_id)

    def test_approve_rejects_non_proposed(self, db):
        wid = _weakness.report_weakness(db, "tool_failure", "y")
        assert _weakness.approve_weakness(db, wid) is False

    def test_resolve_and_dismiss(self, db):
        wid = _weakness.report_weakness(db, "tool_failure", "z")
        assert _weakness.resolve_weakness(db, wid, "fixed upstream") is True
        assert _weakness.resolve_weakness(db, wid) is False  # already gone
        wid2 = _weakness.report_weakness(db, "tool_failure", "z2")
        assert _weakness.dismiss_weakness(db, wid2) is True
        assert _weakness.open_weaknesses(db) == []

    def test_scan_tool_failures_feeds_recovery(self, db):
        now = time.time()
        audit = [{"tool": "t1", "ok": False, "error": "e", "ts": now - i}
                 for i in range(4)]
        opened = _weakness.scan_tool_failures(db, audit)
        assert opened == 1
        # Successes in the same audit feed the recovery streak.
        audit_ok = [{"tool": "t1", "ok": True, "ts": now} for _ in range(5)]
        _weakness.scan_tool_failures(db, audit_ok)
        assert not any(i["subject"] == "t1"
                       for i in _weakness.open_weaknesses(db))

    def test_scan_ledger_failures_opens_case(self, db):
        from nomorals.storage.db import Database
        tmp = Database(":memory:")
        try:
            for _ in range(10):
                record_ledger(tmp, "scheduler", "run", "job1", "ok")
            for _ in range(12):
                record_ledger(tmp, "scheduler", "run", "job1", "boom",
                              ok=False)
            opened = _weakness.scan_ledger_failures(
                tmp, window_hours=24, min_runs=10, rate_threshold=0.5)
            assert len(opened) == 1
            items = _weakness.open_weaknesses(tmp)
            assert items[0]["subject"] == "ledger:scheduler"
        finally:
            tmp.close()


# ── presence: honest, non-spammy ──────────────────────────────────────────

class TestPresence:
    def test_surface_cooldown_enforced(self, db):
        first = _presence.surface(db, "rising_interest", "topic one")
        assert first["ok"] is True
        second = _presence.surface(db, "rising_interest", "topic two")
        assert second["ok"] is False
        assert second["held"] is True
        assert second["reason"] == "cooldown"
        # Force bypass exists for owner-triggered surfaces.
        third = _presence.surface(db, "rising_interest", "topic three",
                                  force=True)
        assert third["ok"] is True

    def test_surface_audit_and_delivery_flow(self, db):
        res = _presence.surface(db, "rising_interest", "afrobeats deep dive")
        pending = _presence.pending_surfaces(db)
        assert len(pending) == 1
        assert pending[0]["title"] == "afrobeats deep dive"
        assert _presence.mark_surface(db, pending[0]["id"], "delivered")
        assert _presence.pending_surfaces(db) == []

    def test_heartbeat_dedups_rising_nudges(self, db):
        for _ in range(4):
            _patterns.record_interests(db, "kwame afrobeats rhythm")
        did1 = _presence.heartbeat(db)
        assert any("afrobeats" in n for n in did1["noticed"])
        # Second heartbeat within the nudge cooldown: no re-emit.
        did2 = _presence.heartbeat(db)
        assert not any("afrobeats" in n for n in did2["noticed"])

    def test_heartbeat_surfaces_at_most_one(self, db):
        for _ in range(4):
            _patterns.record_interests(db, "kwame afrobeats rhythm")
        did = _presence.heartbeat(db)
        assert len(did["surfaced"]) <= 1

    def test_sense_includes_anticipation(self, db):
        for _ in range(4):
            _patterns.record_sequence(db, "morning-pulse", "news-check")
        _presence.note_trigger(db, "morning-pulse")
        snap = _presence.sense(db)
        assert snap["last_trigger"]["trigger_sig"] == "morning-pulse"
        actions = [a["action"] for a in snap["anticipation"]]
        assert "news-check" in actions

    def test_nudge_claim_is_immediate(self, db):
        assert _presence._nudge_due(db, "k") is True
        assert _presence._nudge_due(db, "k") is False  # already claimed
        # A different key is independent.
        assert _presence._nudge_due(db, "k2") is True


# ── patterns: hygiene ─────────────────────────────────────────────────────

class TestPatternHygiene:
    def test_prune_drops_decayed_interests(self, db):
        _patterns.record_interests(db, "fresh topic here")
        _patterns.record_interests(
            db, "ancient pottery", ts=time.time() - 60 * 86400)
        report = _patterns.prune(db)
        assert report["interests_dropped"] >= 1
        topics = [i["topic"] for i in _patterns.current_interests(db)]
        assert "fresh topic here" in " ".join(topics)

    def test_prune_drops_stale_patterns(self, db):
        old = time.time() - 200 * 86400
        _patterns.record_sequence(db, "old-trigger", "old-action", ts=old)
        for _ in range(3):  # likely_next needs count >= 3 by default
            _patterns.record_sequence(db, "new-trigger", "new-action")
        report = _patterns.prune(db)
        assert report["patterns_dropped"] == 1
        assert _patterns.likely_next(db, "old-trigger") == []
        assert _patterns.likely_next(db, "new-trigger")


# ── ledger: failure-rate telemetry ────────────────────────────────────────

class TestLedgerFailureRate:
    def test_failure_rate_math(self, db):
        ledger = AutonomyLedger(db)
        for _ in range(8):
            ledger.record("scheduler", "run", "j", "ok")
        for _ in range(2):
            ledger.record("scheduler", "run", "j", "boom", ok=False)
        fr = ledger.failure_rate("scheduler", window_hours=24)
        assert fr["runs"] == 10
        assert fr["failures"] == 2
        assert fr["failure_rate"] == pytest.approx(0.2)

    def test_failure_rate_zero_runs(self, db):
        ledger = AutonomyLedger(db)
        fr = ledger.failure_rate("idle", window_hours=24)
        assert fr["runs"] == 0
        assert fr["failure_rate"] == 0.0

    def test_new_systems_accepted(self, db):
        ledger = AutonomyLedger(db)
        for system in ("idle", "presence", "weakness", "improvement"):
            entry_id = ledger.record(system, "tick", "", "test")
            assert entry_id  # not collapsed to "other"
        rows = ledger.recent(system="idle", limit=5)
        assert len(rows) == 1
        assert rows[0]["system"] == "idle"


# ── partner autonomy: learning from outcomes ──────────────────────────────

def _fake_autonomy_agent(db):
    """Build an AutonomyAgent with faked brain/gateway (no LLM calls)."""
    from nomorals.agents.autonomy import AutonomyAgent

    class _FakeChat:
        key = "tg:123"
        platform = "tg"
        chat_id = "123"

    mood = SimpleNamespace(
        current=lambda: SimpleNamespace(label="calm",
                                        values={"energy": 80.0}),
        describe=lambda: "calm",
    )
    brain = SimpleNamespace(
        mood=mood,
        persona=SimpleNamespace(name="Devon", interests=["music"]),
        relationship=SimpleNamespace(is_romantic=lambda: False),
        background=SimpleNamespace(ambient_lines=lambda **k: []),
    )
    gateway = SimpleNamespace(
        send=lambda platform, chat, content: SimpleNamespace(ok=True,
                                                             error=None))
    context = SimpleNamespace(db=db, settings=SimpleNamespace())
    agent = AutonomyAgent(context, brain, gateway, mode="suggest",
                          owner_chats={"tg:123"})
    return agent


class TestAutonomyOutcomes:
    def test_note_outcome_replied_boldens(self, tmp_path):
        from nomorals.storage.db import Database
        db = Database(str(tmp_path / "t.db"))
        try:
            agent = _fake_autonomy_agent(db)
            before = agent._threshold
            agent._strategy_by_proposal["p1"] = "silence_checkin"
            res = agent.note_outcome("p1", "replied")
            assert res["ok"] is True
            assert res["strategy"] == "silence_checkin"
            assert agent._threshold < before
            assert agent.strategy_stats["silence_checkin"]["wins"] == 1
        finally:
            db.close()

    def test_note_outcome_ignored_is_half_step(self, tmp_path):
        from nomorals.storage.db import Database
        db = Database(str(tmp_path / "t.db"))
        try:
            agent = _fake_autonomy_agent(db)
            agent._strategy_by_proposal["p2"] = "ambient_share"
            t0 = agent._threshold
            agent.note_outcome("p2", "ignored")
            t1 = agent._threshold
            agent.note_outcome("p2", "denied")
            t2 = agent._threshold
            assert t1 > t0
            # ignored is gentler than a full denial step (0.05)
            assert (t1 - t0) < (t2 - t1) + 1e-9
            assert agent.strategy_stats["ambient_share"]["denied"] == 1
        finally:
            db.close()

    def test_note_outcome_rejects_garbage(self, tmp_path):
        from nomorals.storage.db import Database
        db = Database(str(tmp_path / "t.db"))
        try:
            agent = _fake_autonomy_agent(db)
            res = agent.note_outcome("p1", "exploded")
            assert res["ok"] is False
        finally:
            db.close()

    def test_on_owner_reply_credits_proposal(self, tmp_path):
        from nomorals.storage.db import Database
        db = Database(str(tmp_path / "t.db"))
        try:
            db.execute(
                """CREATE TABLE IF NOT EXISTS proactive_log (
                    id TEXT PRIMARY KEY, kind TEXT, platform TEXT,
                    chat_id TEXT, content TEXT, status TEXT, reason TEXT,
                    decided_at REAL, acted_at REAL)""")
            db.execute(
                "INSERT INTO proactive_log (id, kind, platform, chat_id, "
                "content, status, acted_at) VALUES "
                "('prop-9', 'dm', 'tg', '123', 'hey', 'sent', ?)",
                (time.time(),))

            agent = _fake_autonomy_agent(db)
            agent._strategy_by_proposal["prop-9"] = "silence_checkin"
            res = agent.on_owner_reply("tg:123")
            assert res["ok"] is True
            assert res["noted"] is True
            assert res["outcome"] == "replied"
        finally:
            db.close()

    def test_on_owner_reply_no_proposal(self, tmp_path):
        from nomorals.storage.db import Database
        db = Database(str(tmp_path / "t.db"))
        try:
            db.execute(
                """CREATE TABLE IF NOT EXISTS proactive_log (
                    id TEXT PRIMARY KEY, kind TEXT, platform TEXT,
                    chat_id TEXT, content TEXT, status TEXT, reason TEXT,
                    decided_at REAL, acted_at REAL)""")

            agent = _fake_autonomy_agent(db)
            res = agent.on_owner_reply("tg:999")
            assert res["ok"] is True
            assert res["noted"] is False
        finally:
            db.close()

    def test_custom_strategy_does_not_crash_tick(self, tmp_path):
        """A plugged-in strategy unknown to strategy_stats must not
        KeyError the tick (regression: cross-test pollution caught this)."""
        from types import SimpleNamespace
        from nomorals.agents.autonomy import (
            AutonomyAgent, Proposal, Strategy)
        from nomorals.storage.db import Database
        db = Database(str(tmp_path / "t.db"))
        try:
            agent = _fake_autonomy_agent(db)

            class Weak(Strategy):
                name = "weak"

                def evaluate(self, agent, ctx):
                    chat = SimpleNamespace(key="tg:123", platform="tg",
                                           chat_id="123", kind="dm")
                    return [Proposal(kind="dm", chat=chat, reason="x",
                                     score=0.99, strategy=self.name,
                                     draft_instruction="say hi")]

            agent._strategies = [Weak()]
            # Threshold above the proposal score: the tick reaches the
            # threshold gate (past the strategy-stats bookkeeping that
            # used to KeyError) without drafting or sending.
            agent._threshold = 1.0
            out = agent.tick(now=time.time())
            assert out.get("decision") == "below threshold"
            assert agent.strategy_stats["weak"]["proposed"] == 1
        finally:
            db.close()

    def test_status_includes_strategy_stats(self, tmp_path):
        from nomorals.storage.db import Database
        db = Database(str(tmp_path / "t.db"))
        try:
            agent = _fake_autonomy_agent(db)
            st = agent.status()
            assert "strategy_stats" in st
            assert set(st["strategy_stats"]) == {
                "silence_checkin", "ambient_share", "group_post"}
        finally:
            db.close()
