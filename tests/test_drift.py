"""Drift detection + recovery actions (build-map #56).

All offline: the timeline is a real SQLite HealthTimeline on a tmp path,
HRV/activity come from injected callables.
"""
import re
import time

import pytest

from nomorals.health import (
    BANNED_PHRASES,
    CRISIS_RESOURCES,
    DriftMonitor,
    DriftReport,
    HealthTimeline,
    recovery_plan,
)
from nomorals.health import drift as drift_module
from nomorals.health.coach import COACH_BANNED_PHRASES

DAY = 86400


@pytest.fixture()
def tl(tmp_path):
    t = HealthTimeline(db_path=str(tmp_path / "health.db"))
    yield t
    t.close()


@pytest.fixture()
def mon(tl, tmp_path):
    m = DriftMonitor(tl, db_path=str(tmp_path / "drift.db"))
    yield m
    m.close()


def _log_3days(tl, now, sleeps, moods):
    """Log one sleep + one mood per day for the last 3 days (oldest first)."""
    for i, (h, mood) in enumerate(zip(sleeps, moods)):
        ts = now - (2 - i) * DAY - 3600  # morning-ish, 3/2/1 days ago
        tl.log("sleep", f"slept {h} hours", ts=ts)
        tl.log("mood", "day note", severity=mood, ts=ts + 3600)


def _no_banned(text):
    lowered = text.lower()
    for phrase in list(BANNED_PHRASES) + list(COACH_BANNED_PHRASES):
        assert phrase.lower() not in lowered, f"banned phrase {phrase!r}"


# ── detection ────────────────────────────────────────────────────────

def test_stable_returns_none(mon, tl):
    now = time.time()
    _log_3days(tl, now, [7.5, 8.0, 7.0], [4, 4, 5])
    assert mon.check(now=now) is None


def test_sleep_debt_detected(mon, tl):
    now = time.time()
    _log_3days(tl, now, [5.5, 5.0, 4.5], [3, 3, 3])
    report = mon.check(now=now)
    assert report is not None
    assert any(s.kind == "sleep_debt" for s in report.signals)
    assert "4.5h" not in report.forecast  # forecast uses the average
    assert "tonight will likely be rough" in report.forecast.lower() or \
        "tonight" in report.forecast.lower()


def test_severe_sleep_debt_is_act(mon, tl):
    now = time.time()
    _log_3days(tl, now, [4.5, 4.0, 4.5], [3, 3, 3])
    report = mon.check(now=now)
    assert report is not None
    assert report.severity == "act"


def test_moderate_sleep_debt_is_watch(mon, tl):
    now = time.time()
    _log_3days(tl, now, [6.0, 5.8, 5.9], [4, 4, 4])
    report = mon.check(now=now)
    assert report is not None
    assert report.severity == "watch"


def test_sleep_decline_detected(mon, tl):
    now = time.time()
    _log_3days(tl, now, [8.0, 7.0, 6.0], [4, 4, 4])  # 2h drop, avg 7.0
    report = mon.check(now=now)
    assert report is not None
    assert any(s.kind == "sleep_decline" for s in report.signals)


def test_mood_drop_detected(mon, tl):
    now = time.time()
    _log_3days(tl, now, [8.0, 7.5, 8.0], [4, 3, 2])  # slipping each day
    report = mon.check(now=now)
    assert report is not None
    assert any(s.kind == "mood_drop" for s in report.signals)


def test_low_avg_mood_detected(mon, tl):
    now = time.time()
    _log_3days(tl, now, [8.0, 7.5, 8.0], [2, 3, 2])  # avg 2.33
    report = mon.check(now=now)
    assert report is not None
    assert any(s.kind == "mood_drop" for s in report.signals)


def test_minimum_data_two_days_returns_none(mon, tl):
    now = time.time()
    for i, h in enumerate([4.0, 3.5]):
        ts = now - (1 - i) * DAY
        tl.log("sleep", f"slept {h} hours", ts=ts)
        tl.log("mood", "bad", severity=1, ts=ts + 3600)
    assert mon.check(now=now) is None


def test_no_data_returns_none(mon, tl):
    assert mon.check() is None


def test_hrv_decline_plus_poor_sleep_is_act(tl, tmp_path):
    now = time.time()
    _log_3days(tl, now, [5.5, 5.0, 5.5], [3, 3, 3])
    baseline = [(now - d * DAY, 60.0) for d in range(4, 11)]
    recent = [(now - d * DAY, 45.0) for d in range(1, 4)]  # 25% below
    mon = DriftMonitor(
        tl, db_path=str(tmp_path / "drift.db"),
        hrv_series=lambda: baseline + recent)
    try:
        report = mon.check(now=now)
        assert report is not None
        assert any(s.kind == "hrv_decline" for s in report.signals)
        assert report.severity == "act"  # burnout combo
    finally:
        mon.close()


def test_hrv_stable_no_signal(tl, tmp_path):
    now = time.time()
    _log_3days(tl, now, [5.5, 5.0, 5.5], [3, 3, 3])
    pts = [(now - d * DAY, 60.0) for d in range(1, 11)]
    mon = DriftMonitor(
        tl, db_path=str(tmp_path / "drift.db"),
        hrv_series=lambda: pts)
    try:
        report = mon.check(now=now)
        assert report is not None
        assert not any(s.kind == "hrv_decline" for s in report.signals)
    finally:
        mon.close()


def test_no_biometrics_still_works(mon, tl):
    now = time.time()
    _log_3days(tl, now, [5.0, 5.0, 5.0], [3, 3, 3])
    report = mon.check(now=now)
    assert report is not None  # timeline-only detection, no fabrication


# ── report formatting ────────────────────────────────────────────────

def test_forecast_text_mentions_cause(mon, tl):
    now = time.time()
    _log_3days(tl, now, [5.5, 5.0, 4.5], [3, 3, 3])
    report = mon.check(now=now)
    assert report is not None
    assert "5.0h" in report.forecast  # the 3-day average
    assert "tonight" in report.forecast.lower()


def test_format_has_no_banned_phrases(mon, tl):
    now = time.time()
    _log_3days(tl, now, [4.0, 4.0, 4.0], [2, 2, 1])
    report = mon.check(now=now)
    assert report is not None
    _no_banned(report.format())


def test_crisis_resources_on_very_low_mood(mon, tl):
    now = time.time()
    _log_3days(tl, now, [7.5, 7.0, 7.5], [3, 2, 1])  # mood 1 → crisis
    report = mon.check(now=now)
    assert report is not None
    assert report.crisis is True
    assert report.severity == "act"
    text = report.format()
    assert any("112" in line for line in text.splitlines())
    assert CRISIS_RESOURCES[0].split(":")[0] in text or "112" in text


def test_no_crisis_when_mood_ok(mon, tl):
    now = time.time()
    _log_3days(tl, now, [5.0, 5.0, 5.0], [3, 3, 3])
    report = mon.check(now=now)
    assert report is not None
    assert report.crisis is False
    assert "112" not in report.format()


# ── recovery plan ────────────────────────────────────────────────────

def test_recovery_plan_sleep(mon, tl):
    now = time.time()
    _log_3days(tl, now, [5.0, 5.0, 5.0], [3, 3, 3])
    plan = recovery_plan(mon.check(now=now))
    kinds = {a.kind for a in plan}
    assert "wind_down" in kinds
    assert "bedtime_nudge" in kinds
    assert all(a.schedules for a in plan if a.kind in
               ("wind_down", "bedtime_nudge"))


def test_recovery_plan_mood(mon, tl):
    now = time.time()
    _log_3days(tl, now, [8.0, 8.0, 8.0], [2, 2, 2])
    plan = recovery_plan(mon.check(now=now))
    assert any(a.kind == "light_day" for a in plan)


def test_recovery_plan_empty_for_stable(mon, tl):
    now = time.time()
    _log_3days(tl, now, [8.0, 8.0, 8.0], [4, 4, 4])
    assert mon.check(now=now) is None
    assert recovery_plan(DriftReport(severity="watch")) == []


# ── approval gate ────────────────────────────────────────────────────

def test_schedule_recovery_needs_approval(mon, tl):
    now = time.time()
    _log_3days(tl, now, [5.0, 5.0, 5.0], [3, 3, 3])
    plan = recovery_plan(mon.check(now=now))
    calls = []
    # no approval → nothing scheduled
    assert mon.schedule_recovery(plan, approved_kinds=None,
                                 schedule_fn=lambda **k: calls.append(k)) == []
    assert mon.schedule_recovery(plan, approved_kinds=set(),
                                 schedule_fn=lambda **k: calls.append(k)) == []
    assert calls == []


def test_schedule_recovery_only_approved(mon, tl):
    now = time.time()
    _log_3days(tl, now, [5.0, 5.0, 5.0], [3, 3, 3])
    plan = recovery_plan(mon.check(now=now))
    calls = []
    scheduled = mon.schedule_recovery(
        plan, approved_kinds={"wind_down"},
        schedule_fn=lambda **k: calls.append(k), now=now)
    assert scheduled == ["wind_down"]
    assert len(calls) == 1
    assert calls[0]["action"] == "drift.nudge"
    assert "parameters" in calls[0]


def test_nudge_time_same_evening(mon):
    now = time.mktime((2026, 10, 8, 12, 0, 0, 0, 0, -1))  # noon
    t = mon._nudge_time("wind_down", now)
    lt = time.localtime(t)
    assert (lt.tm_hour, lt.tm_min) == (22, 0)
    assert t > now


def test_nudge_time_rolls_to_tomorrow(mon):
    now = time.mktime((2026, 10, 8, 23, 0, 0, 0, 0, -1))  # 11pm
    t = mon._nudge_time("wind_down", now)
    assert time.localtime(t).tm_mday == 9


# ── proactive delivery ───────────────────────────────────────────────

def test_maybe_notify_act_level(mon, tl):
    now = time.time()
    _log_3days(tl, now, [4.0, 4.0, 4.0], [3, 3, 3])
    report = mon.check(now=now)
    assert report.severity == "act"
    sent = []
    assert mon.maybe_notify(report, sent.append, now=now) is True
    assert len(sent) == 1
    _no_banned(sent[0])


def test_maybe_notify_watch_stays_quiet(mon, tl):
    now = time.time()
    _log_3days(tl, now, [6.0, 5.8, 5.9], [4, 4, 4])
    report = mon.check(now=now)
    assert report.severity == "watch"
    sent = []
    assert mon.maybe_notify(report, sent.append, now=now) is False
    assert sent == []


def test_maybe_notify_none_report(mon):
    sent = []
    assert mon.maybe_notify(None, sent.append) is False
    assert sent == []


def test_once_per_day_cap(mon, tl):
    now = time.time()
    _log_3days(tl, now, [4.0, 4.0, 4.0], [3, 3, 3])
    report = mon.check(now=now)
    sent = []
    assert mon.maybe_notify(report, sent.append, now=now) is True
    # second attempt same day → blocked
    assert mon.maybe_notify(report, sent.append, now=now + 3600) is False
    assert len(sent) == 1
    assert mon.pings_today(now=now) == 1


def test_ping_resets_next_day(mon, tl):
    now = time.time()
    _log_3days(tl, now, [4.0, 4.0, 4.0], [3, 3, 3])
    report = mon.check(now=now)
    sent = []
    assert mon.maybe_notify(report, sent.append, now=now) is True
    assert mon.maybe_notify(report, sent.append, now=now + DAY) is True
    assert len(sent) == 2


def test_notify_includes_recovery_offer(mon, tl):
    now = time.time()
    _log_3days(tl, now, [4.0, 4.0, 4.0], [3, 3, 3])
    report = mon.check(now=now)
    sent = []
    mon.maybe_notify(report, sent.append, now=now)
    assert "Want me to set any of these up?" in sent[0]


# ── scoping ──────────────────────────────────────────────────────────

def test_community_raises(tl, tmp_path):
    with pytest.raises(PermissionError, match="owner-scoped"):
        DriftMonitor(tl, db_path=str(tmp_path / "d.db"), community=True)


def test_thresholds_are_documented_constants():
    assert drift_module.MIN_DAYS == 3
    assert drift_module.MAX_PINGS_PER_DAY == 1
    assert drift_module.SLEEP_DEBT_H == 6.0
