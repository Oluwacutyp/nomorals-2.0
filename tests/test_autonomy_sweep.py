"""Sweep tests for nomorals.autonomy — real behavior, sqlite-backed.

Covers the 2026-10-10 sweep additions: idle stages/inhibitors/adaptive
threshold, weakness signatures/severity/memory/expiry, YAKE topics,
routine decay, transition confidence, presence interruptibility /
serendipity / feedback / digest, and coordinator budget/backoff/report.
"""

import sqlite3
import time

import pytest

from nomorals.autonomy import idle as idle_mod
from nomorals.autonomy import weakness as weak_mod
from nomorals.autonomy import patterns as pat_mod
from nomorals.autonomy import presence as pres_mod
from nomorals.autonomy.coordinator import IdleCoordinator


def make_db():
    db = sqlite3.connect(":memory:")
    db.row_factory = None
    return db


# ── idle: stages ─────────────────────────────────────────────────────

def test_stage_for_graduated():
    assert idle_mod.stage_for(0, 900) == "active"
    assert idle_mod.stage_for(899, 900) == "active"
    assert idle_mod.stage_for(900, 900) == "shallow"
    assert idle_mod.stage_for(3600, 900) == "deep"      # 4x
    assert idle_mod.stage_for(10800, 900) == "night"    # 12x
    assert idle_mod.stage_for(10**7, 900) == "night"


def test_stage_residency():
    now = time.time()
    assert idle_mod.stage_residency_met("shallow", now - 1, ts=now)
    assert not idle_mod.stage_residency_met("deep", now - 10, ts=now)
    assert idle_mod.stage_residency_met("deep", now - 130, ts=now)
    assert not idle_mod.stage_residency_met("night", now - 100, ts=now)
    assert idle_mod.stage_residency_met("night", now - 700, ts=now)


def test_inhibitors_block_and_release():
    db = make_db()
    assert not idle_mod.idle_inhibited(db)
    idle_mod.inhibit_idle(db, "trainer", "long training run")
    assert idle_mod.idle_inhibited(db)
    held = idle_mod.inhibitors(db)
    assert held[0]["name"] == "trainer"
    assert idle_mod.release_idle(db, "trainer")
    assert not idle_mod.idle_inhibited(db)
    assert not idle_mod.release_idle(db, "trainer")  # already gone


def test_inhibitor_context_manager():
    db = make_db()
    with idle_mod.idle_inhibited_scope(db, "job", "testing"):
        assert idle_mod.idle_inhibited(db)
    assert not idle_mod.idle_inhibited(db)


def test_prune_stale_inhibitors():
    db = make_db()
    idle_mod.inhibit_idle(db, "crashed", "old")
    db.execute("UPDATE autonomy_inhibitors SET ts = ?",
               (time.time() - 99999,))
    db.commit()
    assert idle_mod.prune_stale_inhibitors(db, max_age=60) == 1
    assert not idle_mod.idle_inhibited(db)


def test_adaptive_threshold_learns():
    db = make_db()
    # No history → default.
    assert idle_mod.learned_idle_threshold(db, 900) == 900
    base = time.time() - 10000
    # Simulate ~5-minute gaps between activities.
    for i in range(8):
        idle_mod.note_activity(db, ts=base + i * 300)
    learned = idle_mod.learned_idle_threshold(db, 900)
    assert 300 <= learned <= 3600
    assert learned < 900  # 5-min gaps → shorter fuse than 15-min default


def test_idle_session_history_recorded():
    db = make_db()
    now = time.time()
    idle_mod._record_idle_session(db, now - 5000, now - 1000, "deep")
    hist = idle_mod.idle_history(db)
    assert len(hist) == 1
    assert hist[0]["max_stage"] == "deep"
    assert hist[0]["duration_s"] == pytest.approx(4000, abs=1)
    stats = idle_mod.idle_stats(db)
    assert stats["sessions"] == 1
    assert stats["median_s"] == pytest.approx(4000, abs=1)


def test_monitor_stage_transitions(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    db = idle_mod.workspace_db(ws)
    mon = idle_mod.IdleMonitor(ws, idle_seconds=100)
    # Fake: last activity 500s ago → deep stage.
    idle_mod.note_activity(db, ts=time.time() - 500)
    assert mon.check_once() == "idle"
    assert mon._stage == "deep"
    # Fresh activity → active, session recorded.
    idle_mod.note_activity(db)
    assert mon.check_once() == "active"
    assert idle_mod.idle_history(db)[0]["max_stage"] == "deep"
    db.close()


# ── weakness: signatures ─────────────────────────────────────────────

def test_error_signature_normalizes():
    s1 = weak_mod.error_signature(
        "ValueError: invalid literal for int(): 'abc123' at /tmp/x.py:88")
    s2 = weak_mod.error_signature(
        "ValueError: invalid literal for int(): 'xyz999' at /tmp/y.py:91")
    s3 = weak_mod.error_signature("TimeoutError: connection timed out")
    assert s1 == s2  # same structure → same signature
    assert s1 != s3
    assert "valueerror" in s1


def test_report_weakness_fuzzy_clusters():
    db = make_db()
    w1 = weak_mod.report_weakness(
        db, "error_pattern",
        "ValueError: bad id 12345 in /app/tools.py")
    w2 = weak_mod.report_weakness(
        db, "error_pattern",
        "ValueError: bad id 67890 in /app/tools.py")
    assert w1 == w2  # clustered, not two cases
    row = db.execute(
        "SELECT occurrences FROM weaknesses WHERE id = ?", (w1,)).fetchone()
    assert row[0] == 2


def test_burn_rate_severity():
    db = make_db()
    weak_mod.ensure_schema(db)
    wid = weak_mod._new_id()
    now = time.time()
    db.execute(
        "INSERT INTO weaknesses (id, kind, subject, first_seen, last_seen,"
        " occurrences) VALUES (?, 'tool_failure', 't', ?, ?, 1)",
        (wid, now - 86400, now))
    # 5 sightings in the last hour, none older → acute spike.
    for i in range(5):
        db.execute(
            "INSERT INTO weakness_sightings (weakness_id, ts) "
            "VALUES (?, ?)", (wid, now - i * 600))
    db.commit()
    sev = weak_mod.severity(db, wid)
    assert sev["level"] == "critical"
    assert sev["fast_1h"] == 5


def test_fixed_weakness_memory_one_step_resolve():
    db = make_db()
    wid = weak_mod.report_weakness(db, "tool_failure", "flaky_tool")
    for _ in range(weak_mod.FAILURE_THRESHOLD):
        weak_mod.report_weakness(db, "tool_failure", "flaky_tool")
    # Threshold crossed → researching.
    row = db.execute(
        "SELECT status FROM weaknesses WHERE id = ?", (wid,)).fetchone()
    assert row[0] == "researching"
    # Resolve WITH a verified fix → memorized.
    assert weak_mod.resolve_weakness(db, wid, "fixed", fix="add retry")
    fix = weak_mod.recall_fix(db, "tool_failure", "flaky_tool")
    assert fix and "add retry" in fix["fix"]
    # Recurrence → one-step resolve from memory, no re-diagnosis.
    wid2 = weak_mod.report_weakness(db, "tool_failure", "flaky_tool")
    for _ in range(weak_mod.FAILURE_THRESHOLD):
        weak_mod.report_weakness(db, "tool_failure", "flaky_tool")
    row = db.execute(
        "SELECT status FROM weaknesses WHERE id = ?", (wid2,)).fetchone()
    assert row[0] == "resolved"


def test_retry_loop_and_cascade():
    db = make_db()
    assert weak_mod.report_retry_loop(db, "search", 2) is None
    wid = weak_mod.report_retry_loop(db, "search", 6, "boom")
    assert wid
    for _ in range(weak_mod.FAILURE_THRESHOLD):
        weak_mod.report_retry_loop(db, "search", 6, "boom")
    row = db.execute(
        "SELECT kind, status FROM weaknesses WHERE id = ?", (wid,)).fetchone()
    assert row[0] == "retry_loop" and row[1] == "researching"

    assert weak_mod.report_cascade(db, ["a", "b"]) is None
    cid = weak_mod.report_cascade(db, ["a", "b", "c"])
    assert cid
    row = db.execute(
        "SELECT kind FROM weaknesses WHERE id = ?", (cid,)).fetchone()
    assert row[0] == "cascading_failure"


def test_silent_degradation_detection():
    db = make_db()
    for i in range(6):
        weak_mod.record_degradation_sample(db, "answerer", "score", 0.9)
    out = {}
    for i in range(6):
        out = weak_mod.record_degradation_sample(db, "answerer", "score", 0.5)
    assert "weakness_id" in out
    row = db.execute(
        "SELECT kind FROM weaknesses WHERE id = ?",
        (out["weakness_id"],)).fetchone()
    assert row[0] == "silent_degradation"


def test_expire_stale():
    db = make_db()
    wid = weak_mod.report_weakness(db, "tool_failure", "old_tool")
    db.execute("UPDATE weaknesses SET last_seen = ? WHERE id = ?",
               (time.time() - 40 * 86400, wid))
    db.commit()
    assert weak_mod.expire_stale(db, stale_days=30) == 1
    row = db.execute(
        "SELECT status FROM weaknesses WHERE id = ?", (wid,)).fetchone()
    assert row[0] == "dismissed"


def test_acceleration():
    db = make_db()
    wid = weak_mod.report_weakness(db, "tool_failure", "accel_tool")
    now = time.time()
    # Old: 1 sighting/day for 6 days. Recent: 4 in the last day.
    for d in range(2, 8):
        db.execute(
            "INSERT INTO weakness_sightings (weakness_id, ts) "
            "VALUES (?, ?)", (wid, now - d * 86400))
    for h in range(4):
        db.execute(
            "INSERT INTO weakness_sightings (weakness_id, ts) "
            "VALUES (?, ?)", (wid, now - h * 3600))
    db.commit()
    acc = weak_mod.acceleration(db, "tool_failure", "accel_tool")
    assert acc["accelerating"]


def test_weakness_digest_renders():
    db = make_db()
    for _ in range(weak_mod.FAILURE_THRESHOLD):
        weak_mod.report_weakness(db, "tool_failure", "digest_tool")
    plain = weak_mod.weakness_digest(db, theme="plain")
    assert "digest_tool" in plain and "tool_failure" in plain
    rich = weak_mod.weakness_digest(db, theme="rich")
    assert "╭" in rich and "digest_tool" in rich
    empty = weak_mod.weakness_digest(make_db(), theme="plain")
    assert "all clear" in empty


# ── patterns: yake topics ────────────────────────────────────────────

def test_yake_extracts_key_phrases():
    topics = pat_mod.extract_topics(
        "The quantum computer achieved quantum supremacy in the "
        "quantum computer lab. Quantum supremacy changes everything.")
    assert any("quantum" in t for t in topics)
    # Multi-word phrase preferred over lone tokens.
    assert any(" " in t for t in topics)


def test_yake_short_text():
    topics = pat_mod.extract_topics("fix the broken payment webhook")
    assert topics  # works on short chat-length text
    assert any("payment" in t or "webhook" in t for t in topics)


def test_yake_dedupes():
    topics = pat_mod.extract_topics(
        "machine learning models. machine learning is great. "
        "I love machine learning.", max_topics=5)
    # No near-duplicate phrases.
    for i, a in enumerate(topics):
        for b in topics[i + 1:]:
            at, bt = set(a.split()), set(b.split())
            assert len(at & bt) / max(len(at | bt), 1) <= 0.8


def test_record_interests_cooccur_and_clusters():
    db = make_db()
    pat_mod.record_interests(
        db, "training neural networks for music generation with neural nets")
    pat_mod.record_interests(
        db, "neural networks composing music generation tracks")
    clusters = pat_mod.topic_clusters(db, min_cooccur=1)
    assert clusters  # some pair co-occurred twice
    assert any(len(c) >= 2 for c in clusters)


def test_routine_decay_adapts():
    db = make_db()
    # Heavy old habit at hour 7.
    for _ in range(50):
        pat_mod.record_activity(db, ts=1700000000)  # fixed old ts
    db.execute("UPDATE routine_activity SET count = 50")
    db.execute(
        "INSERT INTO pattern_meta (key, value) VALUES "
        "('routine_last_decay', '0') "
        "ON CONFLICT (key) DO UPDATE SET value = '0'")
    db.commit()
    pat_mod._maybe_decay_routine(db, ts=time.time())
    row = db.execute(
        "SELECT count FROM routine_activity").fetchone()
    assert row[0] < 50  # decayed


def test_transition_model_confidence():
    db = make_db()
    for _ in range(10):
        pat_mod.record_sequence(db, "morning", "news")
    for _ in range(2):
        pat_mod.record_sequence(db, "morning", "music")
    model = pat_mod.transition_model(db, "morning")
    assert model[0]["action"] == "news"
    assert model[0]["probability"] > model[1]["probability"]
    assert 0 < model[0]["confidence"] <= 1
    assert model[0]["confidence"] >= model[1]["confidence"]
    nxt = pat_mod.likely_next(db, "morning", min_count=3)
    assert [n["action"] for n in nxt] == ["news"]  # music below min_count
    assert "probability" in nxt[0]


def test_next_activity_confidence():
    db = make_db()
    assert pat_mod.next_activity(db)["predicted"] is False
    import datetime as _dt
    base = _dt.datetime(2026, 10, 5, 7, 0)  # a Monday 07:00
    for d in range(30):
        ts = (base + _dt.timedelta(days=d)).timestamp()
        pat_mod.record_activity(db, ts=ts)
    nxt = pat_mod.next_activity(db)
    assert nxt["predicted"] is True
    assert nxt["hour"] == 7
    assert nxt["confidence"] in ("high", "medium", "low")


def test_quiet_hours_learned():
    db = make_db()
    import datetime as _dt
    base = _dt.datetime(2026, 10, 5, 9, 0)
    for d in range(10):
        for h in (9, 12, 18):
            ts = (base + _dt.timedelta(days=d, hours=h - 9)).timestamp()
            for _ in range(3):
                pat_mod.record_activity(db, ts=ts)
    quiet = pat_mod.quiet_hours(db)
    assert 3 in quiet  # 3am never active
    assert 9 not in quiet and 12 not in quiet


def test_export_import_roundtrip():
    db = make_db()
    pat_mod.record_activity(db, ts=1700000000)
    pat_mod.record_interests(db, "quantum computing breakthrough")
    pat_mod.record_sequence(db, "a", "b")
    dump = pat_mod.export_models(db)
    db2 = make_db()
    counts = pat_mod.import_models(db2, dump)
    assert counts["routine"] >= 1 and counts["interests"] >= 1
    assert counts["patterns"] >= 1
    assert pat_mod.models_summary(db2)["interests"] >= 1


# ── presence ─────────────────────────────────────────────────────────

def test_interruptibility_quiet_hours():
    db = make_db()
    # Night hour, no activity history → default quiet 0-6.
    score = pres_mod.interruptibility(db, tz_offset=1.0,
                                      ts=_ts_at(3, 30))
    assert score["score"] <= 0.2
    assert not score["interruptible"]
    morning = pres_mod.interruptibility(db, tz_offset=1.0,
                                        ts=_ts_at(9, 30))
    assert morning["score"] > score["score"]


def _ts_at(hour, minute):
    import datetime as _dt
    return _dt.datetime(2026, 10, 10, hour, minute,
                        tzinfo=_dt.timezone.utc).timestamp()


def test_interruptibility_recent_activity_boost():
    db = make_db()
    idle_mod.note_activity(db)  # active right now
    s = pres_mod.interruptibility(db, ts=_ts_at(3, 30))
    assert s["score"] >= 0.9  # active <15m ago overrides night


def test_candidate_queue_bounded_deferral():
    db = make_db()
    cid = pres_mod.queue_candidate(db, "rising_interest", "Cool thing",
                                   priority="normal", ttl_hours=24)
    assert cid > 0
    due = pres_mod.due_candidates(db)
    assert due and due[0]["id"] == cid
    assert not due[0]["due"]  # deadline far → waits for better moment
    # Expired candidate is always due.
    db.execute("UPDATE presence_candidates SET deadline_ts = ? "
               "WHERE id = ?", (time.time() - 10, cid))
    db.commit()
    due = pres_mod.due_candidates(db)
    assert due[0]["due"]
    assert pres_mod.drop_candidate(db, cid)
    assert pres_mod.due_candidates(db) == []


def test_serendipity_scoring():
    db = make_db()
    pat_mod.record_interests(db, "training neural networks for music")
    pat_mod.record_interests(db, "neural networks and music generation")
    s = pres_mod.serendipity_score(
        db, "rising_interest", "New paper on neural networks",
        "about music generation")
    assert s["relevance"] > 0.3  # ties to real interests
    assert 0 < s["score"] <= 1.5
    # Repeated surface → novelty drops.
    pres_mod.surface(db, "rising_interest", "New paper on neural networks",
                     force=True)
    s2 = pres_mod.serendipity_score(
        db, "rising_interest", "New paper on neural networks")
    assert s2["novelty"] < s["novelty"]


def test_feedback_loop_tunes_weights():
    db = make_db()
    r = pres_mod.surface(db, "rising_interest", "Topic X", force=True)
    sid = r["surface_id"]
    assert pres_mod.kind_weight(db, "rising_interest") == 1.0
    fb = pres_mod.record_surface_feedback(db, sid, "dismissed")
    assert fb["ok"] and fb["weight"] < 1.0
    fb = pres_mod.record_surface_feedback(db, sid, "engaged")
    assert fb["weight"] > 0.7  # recovered upward
    stats = pres_mod.kind_stats(db)
    assert stats[0]["kind"] == "rising_interest"
    assert stats[0]["dismissed"] == 1 and stats[0]["engaged"] == 1
    bad = pres_mod.record_surface_feedback(db, 99999, "engaged")
    assert not bad["ok"]
    bad2 = pres_mod.record_surface_feedback(db, sid, "meh")
    assert not bad2["ok"]


def test_surface_cooldown_and_urgent():
    db = make_db()
    r1 = pres_mod.surface(db, "rising_interest", "First", force=True)
    assert r1["ok"]
    r2 = pres_mod.surface(db, "rising_interest", "Second")
    assert not r2["ok"] and r2["held"]
    r3 = pres_mod.surface(db, "weakness_proposal", "Urgent fix",
                          priority="urgent")
    assert r3["ok"]  # urgent bypasses cooldown


def test_heartbeat_gates_on_interruptibility(monkeypatch):
    db = make_db()
    # Rising interest → auto candidate for surfacing.
    for _ in range(4):
        pat_mod.record_interests(db, "quantum computing breakthrough lab")
    # Force night: low interruptibility → candidate held, not surfaced.
    monkeypatch.setattr(
        pres_mod, "_now_in_tz",
        lambda *a, **k: {"hour": 3, "dow": 5, "dow_name": "Saturday",
                         "daypart": "night",
                         "iso": "2026-10-10T03:00"})
    did = pres_mod.heartbeat(db, tz_offset=1.0)
    assert did["interruptibility"] < pres_mod.INTERRUPTIBILITY_FLOOR
    assert did["surfaced"] == []  # held for a better moment
    # The candidate wasn't dropped — it waits in the queue.
    queued = [c for c in pres_mod.due_candidates(db)
              if c["kind"] == "rising_interest"]
    assert queued


def test_digest_compiles_batch():
    db = make_db()
    pres_mod.queue_candidate(db, "rising_interest", "Cool paper",
                             "about neural nets", priority="normal")
    text = pres_mod.digest(db, theme="plain")
    assert "Cool paper" in text
    rich = pres_mod.digest(db, theme="rich")
    assert "╭" in rich and "Cool paper" in rich
    # Digest itself is audited.
    rows = db.execute(
        "SELECT kind FROM presence_surface WHERE kind = 'digest'").fetchall()
    assert rows


def test_daypart_tone_differs():
    db = make_db()
    morning = pres_mod.digest(db, tz_offset=-4.0, theme="plain")  # EDT-ish
    assert "☀️" in morning or "morning" in morning.lower()


# ── coordinator ──────────────────────────────────────────────────────

def test_budget_plan_sums_and_minimums():
    db = make_db()
    coord = IdleCoordinator("/tmp/autonomy-test-ws")
    plan = coord._budget_plan(db, total=600)
    assert abs(sum(plan.values()) - 600) < 5
    assert all(v >= 20.0 for v in plan.values())  # nobody starves
    assert set(plan) == set(("weakness", "research", "memory", "presence",
                             "wisdom", "patterns"))


def test_backoff_skips_failing_organ():
    coord = IdleCoordinator("/tmp/autonomy-test-ws")
    for i in range(3):
        coord._record_fail("research", f"err{i}", "cycle-1")
    assert coord._backoff_skip("research") >= 1
    coord._record_ok("research", 1.0, "cycle-2")
    assert coord._backoff_skip("research") == 0  # success resets


def test_coordinator_status_shape():
    coord = IdleCoordinator("/tmp/autonomy-test-ws")
    coord._record_ok("memory", 2.5, "cycle-9")
    st = coord.coordinator_status()
    assert st["cycles"] == 0
    assert st["organs"]["memory"]["avg_seconds"] == 2.5
    assert st["organs"]["memory"]["consecutive_failures"] == 0


def test_cycle_report_before_and_after():
    coord = IdleCoordinator("/tmp/autonomy-test-ws")
    assert "no idle maintenance cycle" in coord.cycle_report()
    coord._last_cycle = {
        "cycle": "cycle-1-123", "seconds": 42.0,
        "steps": {
            "weakness": {"ok": True, "seconds": 1.0, "open": 2,
                         "routed": 1},
            "research": {"ok": False, "skipped": "research organ "
                                                "not available"},
            "memory": {"ok": True, "seconds": 3.0, "episodes": 5,
                       "merged": 1},
            "presence": {"ok": True, "seconds": 0.5, "noticed": 1,
                         "surfaced": 0},
            "wisdom": {"ok": True, "seconds": 2.0, "report": "ok"},
            "patterns": {"ok": True, "seconds": 0.2,
                         "interests_dropped": 3},
        },
    }
    plain = coord.cycle_report(theme="plain")
    assert "cycle-1-123" in plain and "✅" in plain and "⏭️" in plain
    rich = coord.cycle_report(theme="rich")
    assert "╭" in rich
    deferred = IdleCoordinator("/tmp/x")
    deferred._last_cycle = {"cycle": "c", "deferred": ["trainer"]}
    assert "inhibitors" in deferred.cycle_report()
