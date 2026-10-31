"""Tail-73 pins for nomorals/health/drift.py —
0 converts / 6 exclusions (102, 252, 295, 348, 584, 598).

6 tests for 6 sites (counts equal). Each test pins one site
with a control half proving the guarded operation was actually
ATTEMPTED, asserting via the operation's own double / the
healthy-path result. The file contains no exception handlers,
so it adds zero census sites.

drift.py calls no failures-ledger record function anywhere
(grep-proven; its recording sibling handlers log via _log), so
NOT-recording is pinned against the health package ledger
(nomorals/health/failures.py): recent_failures() snapshot
before/after every trigger, plus caplog silence on the module
logger for the sites whose contract is silence.

Verdict notes (full grounds in the tail-73 SURVEY.md):
- 102: _sleep_hours failure -> None, the documented
  no-data value; sole consumer _series skips None days.
- 252: illness_watch failure -> IllnessWatch(active=False);
  silence, never a "you are well" claim; the only production
  caller adds its own logged guard. Fail-closed.
- 295: projected_recovery failure -> "couldn't project
  recovery right now.", an honest string distinct from every
  success output; expose.forecast_burnout surfaces it verbatim.
- 348: close() teardown swallow; no in-tree caller, nothing
  decides on it.
- 584: consecutive_act_days failure -> 0 = genuine no-streak;
  internal-only chain, failure can only suppress an escalation
  addendum (fail-closed). Weakest exclusion of the batch.
- 598: escalation_note failure -> "" = genuine streak<3;
  appended only when non-empty; also the guard_coaching
  escape hatch.
"""

import logging
import time
from types import SimpleNamespace
from unittest import mock

import nomorals.health.drift as D
from nomorals.health.failures import recent_failures


def _snap():
    return list(recent_failures(200))


def _monitor(tmp_path, name="d.db"):
    tl = SimpleNamespace(timeline=lambda **k: [])
    return D.DriftMonitor(tl, db_path=tmp_path / name)


def test_102_sleep_hours_failure_is_none_not_a_verdict(tmp_path, caplog):
    before = _snap()
    assert D._sleep_hours("slept 7.5h last night") == 7.5
    assert D._sleep_hours("no numbers here") is None
    with caplog.at_level(logging.DEBUG, logger="nomorals.health.drift"):
        caplog.clear()
        # non-str text: re.search raises TypeError inside the guard
        assert D._sleep_hours(12345) is None  # type: ignore[arg-type]
    assert [r for r in caplog.records
            if r.name == "nomorals.health.drift"] == []
    assert _snap() == before


def test_252_illness_watch_failure_is_silence_not_clearance(tmp_path):
    before = _snap()
    healthy = D.illness_watch(hrv_ratio=0.80, rhr_delta_bpm=7,
                              sleep_hours=7.0, baseline_sleep=7.0)
    assert healthy.active is True and len(healthy.signals) >= 2
    broken = D.illness_watch(hrv_ratio="bad-type",  # type: ignore[arg-type]
                             rhr_delta_bpm=7)
    assert broken.active is False
    assert broken.detail == "" and broken.signals == []
    assert _snap() == before


def test_295_projected_recovery_failure_string_is_honest(tmp_path):
    before = _snap()
    healthy = D.projected_recovery([5.0, 5.5, 6.0, 6.5, 7.0])
    assert healthy != "couldn't project recovery right now."
    broken = D.projected_recovery([object(), 1, 2, 3, 4])  # type: ignore[list-item]
    assert broken == "couldn't project recovery right now."
    assert _snap() == before


def test_348_close_teardown_swallows_db_failure(tmp_path):
    before = _snap()
    mon = _monitor(tmp_path)
    healthy_close = mock.Mock()
    mon._db.close = healthy_close
    mon.close()
    assert healthy_close.call_count == 1
    mon2 = _monitor(tmp_path, "d2.db")
    mon2._db.close = mock.Mock(side_effect=RuntimeError("db gone"))
    mon2.close()  # must not raise
    assert mon2._db.close.call_count == 1
    assert _snap() == before


def test_584_consecutive_act_days_failure_reads_as_no_streak(
        tmp_path, caplog):
    before = _snap()
    mon = _monitor(tmp_path)
    now = time.time()
    for back in (0, 1, 2):  # control: 3 real act days -> streak 3
        mon._db.execute(
            "INSERT OR REPLACE INTO drift_reports "
            "(day, severity, kinds, ts) VALUES (?,?,?,?)",
            (D._day(now - back * 86400), "act", "sleep_debt", now))
    assert mon.consecutive_act_days(now=now) == 3
    broken = _monitor(tmp_path, "d2.db")
    broken._db.query = mock.Mock(side_effect=RuntimeError("db gone"))
    with caplog.at_level(logging.DEBUG, logger="nomorals.health.drift"):
        caplog.clear()
        assert broken.consecutive_act_days() == 0
    assert broken._db.query.call_count >= 1  # the read was attempted
    assert [r for r in caplog.records
            if r.name == "nomorals.health.drift"] == []
    assert _snap() == before


def test_598_escalation_note_failure_is_empty_not_a_claim(tmp_path):
    before = _snap()
    mon = _monitor(tmp_path)
    now = time.time()
    assert mon.escalation_note(now=now) == ""  # no streak yet
    for back in (0, 1, 2):  # control: streak 3 -> real note text
        mon._db.execute(
            "INSERT OR REPLACE INTO drift_reports "
            "(day, severity, kinds, ts) VALUES (?,?,?,?)",
            (D._day(now - back * 86400), "act", "sleep_debt", now))
    note = mon.escalation_note(now=now)
    assert "day 3" in note
    broken = _monitor(tmp_path, "d2.db")
    broken._db.query = mock.Mock(side_effect=RuntimeError("db gone"))
    assert broken.escalation_note() == ""
    assert _snap() == before
