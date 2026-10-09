"""Tests for the autonomous nervous system."""

import sqlite3
import time

import pytest

from nomorals.autonomy import idle as _idle
from nomorals.autonomy import weakness as _weakness
from nomorals.autonomy import patterns as _patterns
from nomorals.autonomy import presence as _presence


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


class TestIdle:
    def test_note_and_read_activity(self, db):
        _idle.note_activity(db)
        ts = _idle.last_activity_ts(db)
        assert ts > 0
        assert time.time() - ts < 5

    def test_monitor_detects_idle(self, db, tmp_path):
        import pathlib
        # Simulate old activity by writing directly.
        _idle.ensure_schema(db)
        old = time.time() - 3600
        db.execute("UPDATE autonomy_activity SET last_activity_ts = ? "
                   "WHERE id = 1", (old,))
        db.commit()

        fired = []
        mon = _idle.IdleMonitor(tmp_path, idle_seconds=60,
                                on_idle=lambda s: fired.append(s))
        # Point the monitor at our in-memory db by monkeypatching _db.
        mon._db = lambda: db  # noqa: SLF001
        result = mon.check_once()
        assert result == "idle"
        assert len(fired) == 1
        assert fired[0] > 3000  # ~3600s idle

    def test_monitor_detects_active_again(self, db, tmp_path):
        _idle.note_activity(db)
        mon = _idle.IdleMonitor(tmp_path, idle_seconds=3600)
        mon._db = lambda: db  # noqa: SLF001
        mon._was_idle = True  # simulate previously idle
        result = mon.check_once()
        assert result == "active"


class TestWeakness:
    def test_report_clusters(self, db):
        wid1 = _weakness.report_weakness(db, "tool_failure", "foo",
                                         {"e": "x"})
        wid2 = _weakness.report_weakness(db, "tool_failure", "foo",
                                         {"e": "y"})
        assert wid1 == wid2  # clustered

    def test_threshold_triggers_research(self, db):
        wid = None
        for _ in range(3):
            wid = _weakness.report_weakness(db, "tool_failure", "bar")
        row = db.execute("SELECT status FROM weaknesses WHERE id = ?",
                         (wid,)).fetchone()
        assert row[0] == "researching"

    def test_proposal_and_approve(self, db):
        wid = _weakness.report_weakness(db, "capability_gap", "flying")
        for _ in range(2):
            _weakness.report_weakness(db, "capability_gap", "flying")
        _weakness.record_proposal(db, wid, "build wings")
        assert _weakness.approve_weakness(db, wid) is True
        row = db.execute("SELECT status FROM weaknesses WHERE id = ?",
                         (wid,)).fetchone()
        assert row[0] == "approved"

    def test_open_weaknesses(self, db):
        _weakness.report_weakness(db, "tool_failure", "baz")
        items = _weakness.open_weaknesses(db)
        assert len(items) == 1
        assert items[0]["subject"] == "baz"


class TestPatterns:
    def test_routine_recording(self, db):
        for _ in range(25):
            _patterns.record_activity(db)
        hist = _patterns.routine_histogram(db)
        assert hist["total"] == 25

    def test_predict_needs_minimum(self, db):
        _patterns.record_activity(db)
        assert _patterns.predict_active_windows(db) == []

    def test_interests_with_decay(self, db):
        _patterns.record_interests(db, "I love afrobeats music and drums")
        interests = _patterns.current_interests(db)
        assert len(interests) > 0
        # Old interest decays.
        old_ts = time.time() - 60 * 86400  # 60 days ago
        _patterns.record_interests(db, "ancient pottery", ts=old_ts)
        interests = _patterns.current_interests(db, limit=50)
        pottery = [i for i in interests if i["topic"] == "pottery"]
        assert not pottery or pottery[0]["score"] < 0.5

    def test_rising_interests(self, db):
        for _ in range(4):
            _patterns.record_interests(db, "kwame afrobeats rhythm")
        rising = _patterns.rising_interests(db)
        assert any("afrobeats" in r["topic"] for r in rising)

    def test_sequences(self, db):
        for _ in range(4):
            _patterns.record_sequence(db, "morning", "news-check")
        nxt = _patterns.likely_next(db, "morning")
        assert len(nxt) == 1
        assert nxt[0]["action"] == "news-check"


class TestPresence:
    def test_sense_snapshot(self, db):
        snap = _presence.sense(db)
        assert "time" in snap
        assert "daypart" in snap["time"]
        assert "interests" in snap
        assert "weaknesses" in snap

    def test_heartbeat_runs(self, db):
        did = _presence.heartbeat(db)
        assert "noticed" in did
        assert "prepared" in did
        assert "surfaced" in did
