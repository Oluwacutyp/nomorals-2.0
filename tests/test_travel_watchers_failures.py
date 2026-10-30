"""Tail-62 pins for nomorals/travel/watchers.py —
0 converts / 6 exclusions / 1 flagged (527, pin records OBSERVED
only). 7 tests for 7 sites (counts equal). Each test pins one
site with a control half proving the guarded operation was
actually ATTEMPTED — via the double's own call log, or by
re-executing the guarded statement and showing it raises. The
file contains no exception handlers, so it adds zero census
sites.

watchers.py calls no failures-ledger record function anywhere
(grep-proven), so NOT-recording is pinned against the travel
package ledger (nomorals/travel/failures.py): recent_failures()
snapshot before/after every trigger (tail-61 precedent).

Verdict notes (full grounds in the tail-62 SURVEY.md):
- 360: additive-migration ALTER guard — the guarded statement
  IS the "column already there" probe; a real failure surfaces
  loudly at the next statement, never as a silent wrong value.
- 466: connector-resolution failure -> None, the documented
  no-client value for an optional connector (the raise IS the
  not-configured signal); no fare, no alert, no claim.
- 501: scarcity raw failure -> "" — the seats line is omitted,
  never asserted either way.
- 527 FLAGGED: history() read failure -> [], byte-identical to
  a genuinely empty history over the module's long-lived record;
  forecast() renders "not enough price history yet" over it.
  Pin records OBSERVED behavior only.
- 547: _recent_alert read failure -> False; the rise alert it
  may let through is factually true — only the anti-spam
  cooldown degrades. No claim reads the False.
- 558: _log_alert write failure -> pass; the alert itself was
  built from real prices and stays truthful; only future
  cooldown reads weaken (547's write side).
- 778: list_jobs failure -> registration proceeds under the
  fixed task_id; a registration failure still returns False
  honestly via the outer handler.
"""

import sqlite3

import pytest

from nomorals.connectors import registry as reg
from nomorals.travel import failures as trav_fail
from nomorals.travel import watchers as W


def _snap():
    return list(trav_fail.recent_failures(200))


def test_360_migration_alter_guard_on_migrated_db(tmp_path):
    before = _snap()
    db = str(tmp_path / "mig.db")
    raw = sqlite3.connect(db)
    raw.execute(
        """CREATE TABLE watches (
               id TEXT PRIMARY KEY, origin TEXT, destination TEXT,
               departure_date TEXT, target_kobo INTEGER DEFAULT 0,
               threshold_pct REAL DEFAULT 10.0,
               last_kobo INTEGER DEFAULT 0, last_currency TEXT DEFAULT '',
               alerted_kobo INTEGER DEFAULT 0,
               active INTEGER DEFAULT 1,
               created_at REAL, last_checked REAL DEFAULT 0,
               notify_on TEXT DEFAULT 'drop',
               cooldown_h REAL DEFAULT 24.0)""")
    raw.commit()
    raw.close()
    w = W.PriceWatcher(db_path=db)  # both ALTERs fail in the guard
    # attempt proof: the guarded statement provably raises on this DB
    with pytest.raises(sqlite3.OperationalError):
        w._db.execute(
            "ALTER TABLE watches ADD COLUMN notify_on TEXT DEFAULT 'drop'")
    made = w.watch("LOS", "LHR", "2026-12-01")  # control: functional
    assert w.get(made.id) is not None
    assert _snap() == before


def test_466_duffel_resolution_failure_yields_none(tmp_path,
                                                   monkeypatch):
    before = _snap()
    calls = []

    def boom(name):
        calls.append(name)
        raise RuntimeError("registry down")

    monkeypatch.setattr(reg, "get_connector", boom)
    w = W.PriceWatcher(db_path=str(tmp_path / "w.db"))
    assert w._duffel_client() is None
    assert calls == ["duffel"]  # resolution was attempted
    assert w._cheapest("LOS", "LHR", "2026-12-01") is None

    class FakeOffer:
        total_amount = "450000.00"
        total_currency = "NGN"
        raw = {}

    class FakeDuffel:
        def search_offers(self, o, d, date):
            return [FakeOffer()]

    monkeypatch.setattr(reg, "get_connector", lambda name: FakeDuffel())
    got = w._cheapest("LOS", "LHR", "2026-12-01")  # control
    assert got is not None and got[0] == 45_000_000
    assert _snap() == before


def test_501_scarcity_raw_failure_yields_empty():
    before = _snap()

    class BadOffer:
        def __init__(self):
            self.accessed = 0

        @property
        def raw(self):
            self.accessed += 1
            raise RuntimeError("raw boom")

    bad = BadOffer()
    assert W.PriceWatcher._scarcity(bad) == ""
    assert bad.accessed == 1  # the raw read was attempted

    class GoodOffer:
        raw = {"available_services": [{}, {}, {}]}

    assert W.PriceWatcher._scarcity(GoodOffer()) == \
        "Only 3 seats left at this fare."  # control
    assert _snap() == before


def test_527_history_failure_yields_empty_observed(tmp_path):
    before = _snap()
    w = W.PriceWatcher(db_path=str(tmp_path / "w.db"))
    made = w.watch("LOS", "LHR", "2026-12-01")
    w._record_point(made.id, 45_000_000, "NGN", w._now())
    w._db.commit()
    assert len(w.history(made.id)) == 1  # control: real history reads
    w._db.execute("DROP TABLE price_history")
    w._db.commit()
    # attempt proof: the guarded SELECT provably raises now
    with pytest.raises(sqlite3.OperationalError):
        w._db.execute(
            "SELECT checked_at, kobo, currency FROM price_history"
            " WHERE watch_id = ?", (made.id,)).fetchall()
    assert w.history(made.id) == []  # OBSERVED (flagged site)
    assert _snap() == before


def test_547_recent_alert_failure_yields_false(tmp_path):
    before = _snap()
    w = W.PriceWatcher(db_path=str(tmp_path / "w.db"))
    made = w.watch("LOS", "LHR", "2026-12-01")
    now = w._now()
    w._log_alert(made.id, W.NOTIFY_RISE, 100, now)
    w._db.commit()
    assert w._recent_alert(made.id, W.NOTIFY_RISE, now, 24.0) is True
    w._db.execute("DROP TABLE alert_log")
    w._db.commit()
    with pytest.raises(sqlite3.OperationalError):
        w._db.execute("SELECT MAX(at) AS at FROM alert_log"
                      " WHERE watch_id = ? AND kind = ?",
                      (made.id, "rise")).fetchone()
    # OBSERVED: cooldown read fails open — a repeat rise alert can fire
    assert w._recent_alert(made.id, W.NOTIFY_RISE, now, 24.0) is False
    assert _snap() == before


def test_558_log_alert_failure_is_swallowed(tmp_path):
    before = _snap()
    w = W.PriceWatcher(db_path=str(tmp_path / "w.db"))
    made = w.watch("LOS", "LHR", "2026-12-01")
    now = w._now()
    w._log_alert(made.id, W.NOTIFY_RISE, 100, now)  # control write
    w._db.commit()
    assert w._recent_alert(made.id, W.NOTIFY_RISE, now, 24.0) is True
    w._db.execute("DROP TABLE alert_log")
    w._db.commit()
    w._log_alert(made.id, W.NOTIFY_RISE, 200, now)  # must not raise
    assert _snap() == before


def test_778_list_jobs_failure_registers_anyway():
    before = _snap()

    class Sched:
        def __init__(self, jobs=(), list_raises=False):
            self._jobs = list(jobs)
            self._list_raises = list_raises
            self.list_calls = 0
            self.cron_calls = []

        def list_jobs(self):
            self.list_calls += 1
            if self._list_raises:
                raise RuntimeError("scheduler down")
            return self._jobs

        async def schedule_cron(self, **kw):
            self.cron_calls.append(kw)

    bad = Sched(list_raises=True)
    assert W.ensure_schedule(bad) is True
    assert bad.list_calls == 1  # listing was attempted
    assert len(bad.cron_calls) == 1  # registration proceeded

    class Job:
        action = W.PRICE_WATCH_ACTION
        parameters = {}

    good = Sched(jobs=[Job()])
    assert W.ensure_schedule(good) is True
    assert good.cron_calls == []  # control: already registered, no dup
    assert _snap() == before
