"""Failure pins for nomorals/autonomy/patterns.py (tail-67).

All six sites are the same adjudicated EXCLUSION shape: a
``db.commit()`` flush guard AFTER the real writes. The writes
themselves are never guarded — an execute() failure propagates to
the caller's own loud handler (partner runtime's record_failure,
the coordinator's error report). The production store
(nomorals.storage.db.Database) is autocommit
(``isolation_level=None``) and has no ``commit()`` method at all, so
on the production path the guarded call raises AttributeError every
time, after the writes have already landed durably. Each pin proves
the write was attempted, the commit was attempted, and the designed
return value / landed state is unchanged.
"""
import sqlite3

from nomorals.autonomy import patterns as pat
from nomorals.storage.db import Database


class Cursor:
    def __init__(self, rows=None, one=None, rowcount=0):
        self._rows = rows or []
        self._one = one
        self.rowcount = rowcount

    def fetchone(self):
        return self._one

    def fetchall(self):
        return list(self._rows)


class CommitRaisesDB:
    """execute() lands on a real sqlite connection; commit() is a
    spy that always raises."""

    def __init__(self, one=None):
        self.conn = sqlite3.connect(":memory:")
        self.calls = []
        self.commit_calls = 0
        self._one = one

    def execute(self, sql, params=()):
        self.calls.append(sql)
        if self._one is not None and "pattern_meta" in sql \
                and "SELECT" in sql:
            return Cursor(one=self._one)
        return self.conn.execute(sql, params)

    def commit(self):
        self.commit_calls += 1
        raise RuntimeError("commit boom")


# 121 — _meta_set: the meta write lands via execute; the commit
# guard absorbs only the flush.
def test_121_meta_set_commit_failure_write_lands():
    db = CommitRaisesDB()
    assert pat._meta_set(db, "k1", "v1") is None
    assert any("INSERT INTO pattern_meta" in s for s in db.calls)
    assert db.commit_calls >= 1
    row = db.conn.execute(
        "SELECT value FROM pattern_meta WHERE key='k1'").fetchone()
    assert row == ("v1",)


# 151 — _maybe_decay_routine: decay executes run; only the final
# flush is guarded.
def test_151_decay_routine_commit_failure():
    db = CommitRaisesDB(one=("0",))  # last decay = 0 -> decay is due
    assert pat._maybe_decay_routine(db, ts=200 * 86400.0) is None
    assert any("UPDATE routine_activity" in s for s in db.calls)
    assert db.commit_calls >= 1


# 171 — record_activity: INSERT lands; on the production Database
# (autocommit, no commit method) the call is a no-raise and the
# activity cell is really recorded.
def test_171_record_activity_commit_failure():
    db = CommitRaisesDB()
    assert pat.record_activity(db, ts=1700000000.0) is None
    assert any("INSERT INTO routine_activity" in s for s in db.calls)
    assert db.commit_calls >= 1
    assert not hasattr(Database, "commit")  # production shape
    real_db = Database(":memory:")
    pat.record_activity(real_db, ts=1700000000.0)
    assert pat.routine_histogram(real_db)["total"] == 1


# 443 — record_interests: the returned topics are the extraction
# result (computed before the flush); INSERTs are attempted and
# only the commit is guarded.
def test_443_record_interests_commit_failure():
    db = CommitRaisesDB()
    topics = pat.record_interests(
        db, "Gold prices surged in Lagos markets. Gold traders "
            "watched gold prices closely as markets rallied.",
        ts=1700000000.0)
    assert isinstance(topics, list) and len(topics) >= 1
    assert any("INSERT INTO interests" in s for s in db.calls)
    assert db.commit_calls >= 1


# 574 — prune's INNER commit guard: the report is already computed
# from executed DELETEs; a commit failure leaves the clean report
# untouched (contrast: an execute failure reaches prune's outer
# handler and DOES land in report["error"]).
def test_574_prune_inner_commit_failure_clean_report():
    class PruneDB(CommitRaisesDB):
        def execute(self, sql, params=()):
            self.calls.append(sql)
            if "FROM interests" in sql:
                return Cursor(rows=[])
            if "COUNT(*)" in sql:
                return Cursor(one=(0,))
            return Cursor(rowcount=0)

    db = PruneDB()
    report = pat.prune(db, ts=1700000000.0)
    assert report == {"interests_dropped": 0, "cells_dropped": 0,
                      "patterns_dropped": 0, "weak_pairs_dropped": 0}
    assert "error" not in report
    assert db.commit_calls >= 1


# 598 — record_sequence: INSERT attempted; only the flush is guarded.
def test_598_record_sequence_commit_failure():
    db = CommitRaisesDB()
    assert pat.record_sequence(
        db, "morning-pulse", "news-check", ts=1700000000.0) is None
    assert any("INSERT INTO patterns" in s for s in db.calls)
    assert db.commit_calls >= 1
    row = db.conn.execute(
        "SELECT count FROM patterns WHERE trigger_sig='morning-pulse'"
    ).fetchone()
    assert row == (1,)
