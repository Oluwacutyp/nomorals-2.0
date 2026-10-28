"""Tail-27 pins: autonomy/idle.py commit sites, all exclusions.

Verdict: pins-only, 0 converts / 9 exclusions. All 9 sites are
``try: db.commit() except: pass`` blocks, and the decisive fact is
about the store, not the handlers: the canonical store
(``workspace_db`` -> ``Database``) has NO ``commit`` method and
autocommits every statement (``isolation_level=None``). In production
the commit call ALWAYS raises ``AttributeError``, yet every
post-commit claim still holds — the writes already persisted. There
is no lie to fix, and no safe minimal fix exists: signaling commit
failure would report failure for every successful Database write
(``release_idle`` always False, prune always 0, ``check_once`` never
emitting) — the cure would create the lie it purports to fix. The
correct change is a persistence cleanup (drop the vestigial commit
calls for autocommit stores, or commit-if-in-transaction), which is a
design change outside error-handling scope; it is documented in
SURVEY.md as a follow-up, not converted here. The failure mode IS real
for a non-autocommit store whose commit raises (a probe in the verify
dir shows the claims going false after rollback there), but that path
is not live: the only non-autocommit callers are test doubles, where
commit succeeds, and a commit failure there requires an already-sick
DB whose preceding (uncaptured, propagating) executes would have
failed first.

Each pin drives the handler through its except path with the REAL
Database — whose commit call genuinely raises on every call — and pins
the exact outcome plus the fact that the agents failures ledger is
unchanged (snapshot before/after, tail-8 method). idle.py calls no
record function anywhere (grep-proven), so the ledger must not move.

The pins file itself contains no except handlers, so it adds zero
census sites.
"""

import tempfile
import time
from unittest import mock

from nomorals.agents import failures as ag_fail
from nomorals.autonomy import idle as I
from nomorals.storage.db import Database


def _ledger():
    return list(ag_fail.recent_failures())


def test_set_config_commit_failure_still_persists():
    """167: _set_config round-trips even though commit() raises."""
    before = _ledger()
    with tempfile.TemporaryDirectory() as d:
        db = I.workspace_db(d)
        I._set_config(db, "gap_ema", "42.5")
        assert I._get_config(db, "gap_ema") == "42.5"
        assert I._get_config(db, "unset_key", "dflt") == "dflt"
    assert _ledger() == before


def test_note_activity_commit_failure_still_resets_clock():
    """202: note_activity updates last_activity_ts despite commit raising."""
    before = _ledger()
    with tempfile.TemporaryDirectory() as d:
        db = I.workspace_db(d)
        assert I.last_activity_ts(db) == 0.0
        I.note_activity(db, ts=1000.0)
        assert I.last_activity_ts(db) == 1000.0
    assert _ledger() == before


def test_inhibit_idle_commit_failure_still_holds():
    """302: inhibit_idle holds the inhibitor despite commit raising."""
    before = _ledger()
    with tempfile.TemporaryDirectory() as d:
        db = I.workspace_db(d)
        assert I.inhibitors(db) == []
        I.inhibit_idle(db, "trainer", "long run")
        assert [x["name"] for x in I.inhibitors(db)] == ["trainer"]
        assert I.idle_inhibited(db) is True
    assert _ledger() == before


def test_release_idle_commit_failure_still_releases():
    """315: release_idle True then False despite commit raising."""
    before = _ledger()
    with tempfile.TemporaryDirectory() as d:
        db = I.workspace_db(d)
        assert I.release_idle(db, "never-held") is False
        I.inhibit_idle(db, "trainer", "run")
        assert I.release_idle(db, "trainer") is True
        assert I.release_idle(db, "trainer") is False
        assert I.idle_inhibited(db) is False
    assert _ledger() == before


def test_prune_commit_failure_still_prunes():
    """360: prune returns the count and rows are gone despite commit raising."""
    before = _ledger()
    with tempfile.TemporaryDirectory() as d:
        db = I.workspace_db(d)
        I.inhibit_idle(db, "crashed", "old")
        db.execute(
            "UPDATE autonomy_inhibitors SET ts = ? WHERE name = 'crashed'",
            (time.time() - 99999,))
        assert I.prune_stale_inhibitors(db, max_age=60) == 1
        assert I.inhibitors(db) == []
        I.inhibit_idle(db, "fresh", "x")
        assert I.prune_stale_inhibitors(db, max_age=99999) == 0
    assert _ledger() == before


def test_record_idle_session_commit_failure_still_records():
    """384: _record_idle_session appears in history despite commit raising."""
    before = _ledger()
    with tempfile.TemporaryDirectory() as d:
        db = I.workspace_db(d)
        assert I.idle_history(db) == []
        I._record_idle_session(db, 100.0, 700.0, "deep")
        hist = I.idle_history(db)
        assert len(hist) == 1
        assert hist[0]["duration_s"] == 600.0
        assert hist[0]["max_stage"] == "deep"
        assert I.idle_stats(db)["sessions"] == 1
    assert _ledger() == before


def test_check_once_idle_branch_commit_failure_still_idles():
    """504: check_once returns 'idle' and the DB row says idle=1."""
    before = _ledger()
    with tempfile.TemporaryDirectory() as d:
        mon = I.IdleMonitor(d, idle_seconds=60)
        assert mon.check_once() == "unchanged"
        I.note_activity(mon._db(), ts=time.time() - 1000)
        assert mon.check_once() == "idle"
        assert I.idle_state(mon._db())["idle"] is True
        assert mon._was_idle is True
    assert _ledger() == before


def test_check_once_active_branch_commit_failure_still_activates():
    """541: check_once returns 'active', row says idle=0, session recorded."""
    before = _ledger()
    with tempfile.TemporaryDirectory() as d:
        mon = I.IdleMonitor(d, idle_seconds=60)
        I.note_activity(mon._db(), ts=time.time() - 1000)
        assert mon.check_once() == "idle"
        I.note_activity(mon._db())
        assert mon.check_once() == "active"
        assert I.idle_state(mon._db())["idle"] is False
        assert len(I.idle_history(mon._db())) == 1
    assert _ledger() == before


def test_check_once_close_failure_still_returns():
    """569: a close() failure in the finally block never breaks check_once."""
    before = _ledger()
    with tempfile.TemporaryDirectory() as d:
        mon = I.IdleMonitor(d, idle_seconds=60)
        with mock.patch.object(Database, "close",
                               side_effect=RuntimeError("close boom")):
            assert mon.check_once() == "unchanged"
        assert mon.check_once() == "unchanged"
    assert _ledger() == before
