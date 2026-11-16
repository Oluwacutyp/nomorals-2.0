"""Gapfix R2-28 — the health drift monitor runs on a schedule.

Failing-first: DriftMonitor itself is proven (tests/test_drift.py); what
never existed is the trigger — a durable daily tick that builds the monitor
over the timeline store and surfaces an "act" report, plus a morning
briefing provider for the "watch" half. Every test here fails until the
wiring lands.
"""

import time
from types import SimpleNamespace

import pytest

from nomorals.health import DriftMonitor, HealthTimeline

DAY = 86400


# ── fixtures ─────────────────────────────────────────────────────────


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
    """One sleep + one mood per day for the last 3 days (oldest first)."""
    for i, (h, mood) in enumerate(zip(sleeps, moods, strict=True)):
        ts = now - (2 - i) * DAY - 3600
        tl.log("sleep", f"slept {h} hours", ts=ts)
        tl.log("mood", "day note", severity=mood, ts=ts + 3600)


class _Sched:
    """Minimal stand-in for agents.scheduler.Scheduler (digest pattern)."""

    def __init__(self):
        self.jobs = []

    def list_jobs(self):
        return list(self.jobs)

    def add(self, name, spec, kind, payload, routine=False):
        job = {"name": name, "spec": spec, "kind": kind,
               "payload": payload, "routine": routine}
        self.jobs.append(job)
        return job

    def remove(self, job_id):
        self.jobs = [j for j in self.jobs if j.get("id") != job_id]


class _Notifier:
    """Records publish() calls; never delivers."""

    def __init__(self):
        self.published = []

    def publish(self, kind, title="", body="", **kw):
        self.published.append({"kind": kind, "title": title, "body": body})
        return {"ok": True}


# ── job registration ─────────────────────────────────────────────────


def test_ensure_registers_durable_routine_tool_job():
    from nomorals.health.drift import (
        DRIFT_JOB,
        DRIFT_TOOL,
        ensure_health_drift_job,
    )

    sched = _Sched()
    job = ensure_health_drift_job(sched)
    assert job["name"] == DRIFT_JOB
    assert job["spec"] == "daily 07:00"
    assert job["kind"] == "tool"
    assert job["payload"] == {"tool": DRIFT_TOOL, "args": {}}
    assert job["routine"] is True
    assert len(sched.jobs) == 1


def test_ensure_is_idempotent():
    from nomorals.health.drift import ensure_health_drift_job

    sched = _Sched()
    first = ensure_health_drift_job(sched)
    second = ensure_health_drift_job(sched)
    assert second == first
    assert len(sched.jobs) == 1


def test_ensure_replaces_stale_spec():
    from nomorals.health.drift import DRIFT_JOB, ensure_health_drift_job

    sched = _Sched()
    sched.jobs.append({"id": "old", "name": DRIFT_JOB, "spec": "every 6h"})
    job = ensure_health_drift_job(sched)
    assert job["spec"] == "daily 07:00"
    assert len(sched.jobs) == 1
    assert sched.jobs[0].get("id") != "old"


def test_ensure_none_scheduler_returns_none():
    from nomorals.health.drift import ensure_health_drift_job

    assert ensure_health_drift_job(None) is None


def test_ensure_broken_scheduler_returns_none():
    from nomorals.health.drift import ensure_health_drift_job

    class _Boom:
        def list_jobs(self):
            raise RuntimeError("scheduler down")

    assert ensure_health_drift_job(_Boom()) is None


# ── the tick ───────────────────────────────────────────────────────


def test_tick_stable_timeline_stays_silent(tl, mon):
    from nomorals.health.drift import run_health_drift_tick_once

    now = time.time()
    _log_3days(tl, now, [7.5, 8.0, 7.0], [4, 4, 5])
    notifier = _Notifier()
    out = run_health_drift_tick_once(
        timeline=tl, monitor=mon, notifier=notifier, now=now)
    assert out["ok"] is True
    assert out["drift"] is False
    assert notifier.published == []


def test_tick_act_report_pings_once_then_caps(tl, mon):
    from nomorals.health.drift import run_health_drift_tick_once

    now = time.time()
    _log_3days(tl, now, [4.5, 4.5, 4.5], [4, 4, 4])
    notifier = _Notifier()
    out = run_health_drift_tick_once(
        timeline=tl, monitor=mon, notifier=notifier, now=now)
    assert out["ok"] is True
    assert out["drift"] is True
    assert out["severity"] == "act"
    assert out["notified"] is True
    assert len(notifier.published) == 1
    assert notifier.published[0]["kind"] == "health-drift"
    assert "4.5h" in notifier.published[0]["body"]
    # Second tick the same day: the 1/day cap holds — nothing re-sent.
    out2 = run_health_drift_tick_once(
        timeline=tl, monitor=mon, notifier=notifier, now=now)
    assert out2["notified"] is False
    assert len(notifier.published) == 1


def test_tick_watch_report_never_pings(tl, mon):
    from nomorals.health.drift import run_health_drift_tick_once

    now = time.time()
    _log_3days(tl, now, [6.0, 5.8, 5.6], [4, 4, 4])
    notifier = _Notifier()
    out = run_health_drift_tick_once(
        timeline=tl, monitor=mon, notifier=notifier, now=now)
    assert out["ok"] is True
    assert out["drift"] is True
    assert out["severity"] == "watch"
    assert out["notified"] is False
    assert notifier.published == []
    # ... but the report still reaches the caller (the briefing's half).
    assert "sleep" in out["message"].lower()
    assert out["plan"], "a watch report still carries suggestions"


def test_tick_without_notifier_returns_payload(tl, mon):
    from nomorals.health.drift import run_health_drift_tick_once

    now = time.time()
    _log_3days(tl, now, [4.5, 4.5, 4.5], [4, 4, 4])
    out = run_health_drift_tick_once(timeline=tl, monitor=mon, now=now)
    assert out["ok"] is True
    assert out["drift"] is True
    assert out["severity"] == "act"
    assert out["notified"] is False
    assert "4.5h" in out["message"]


def test_tick_records_report_row(tl, mon, tmp_path):
    from nomorals.health.drift import run_health_drift_tick_once

    now = time.time()
    _log_3days(tl, now, [4.5, 4.5, 4.5], [4, 4, 4])
    out = run_health_drift_tick_once(
        timeline=tl, monitor=mon, notifier=_Notifier(), now=now)
    assert out["drift"] is True
    rows = mon._db.query("SELECT day, severity, kinds FROM drift_reports")
    assert len(rows) == 1
    assert rows[0]["severity"] == "act"
    assert "sleep_debt" in rows[0]["kinds"]


def test_tick_illness_watch_rides_in_payload(tl, mon):
    from nomorals.health.drift import run_health_drift_tick_once

    now = time.time()
    _log_3days(tl, now, [7.5, 8.0, 7.0], [4, 4, 5])
    out = run_health_drift_tick_once(
        timeline=tl, monitor=mon, notifier=_Notifier(), now=now)
    assert out["ok"] is True
    assert "illness_watch" in out
    assert out["illness_watch"]["active"] is False


# ── briefing provider ──────────────────────────────────────────────


def test_provider_stable_timeline_returns_none(tl, mon):
    from nomorals.health.drift import DriftBriefingProvider

    now = time.time()
    _log_3days(tl, now, [7.5, 8.0, 7.0], [4, 4, 5])
    provider = DriftBriefingProvider(timeline=tl, monitor=mon)
    assert provider.collect(SimpleNamespace(), now) is None


def test_provider_drift_returns_section_with_suggestions(tl, mon):
    from nomorals.health.drift import DriftBriefingProvider

    now = time.time()
    _log_3days(tl, now, [4.5, 4.5, 4.5], [4, 4, 4])
    provider = DriftBriefingProvider(timeline=tl, monitor=mon)
    section = provider.collect(SimpleNamespace(), now)
    assert section is not None
    assert section.name == "health-drift"
    text = "\n".join(section.lines)
    assert "4.5h" in text
    assert "wind-down" in text or "bedtime" in text


def test_provider_never_raises(monkeypatch):
    from nomorals.health import timeline as timeline_module
    from nomorals.health.drift import DriftBriefingProvider

    def _boom(*a, **k):
        raise RuntimeError("timeline down")

    monkeypatch.setattr(timeline_module, "HealthTimeline", _boom)
    provider = DriftBriefingProvider()
    assert provider.collect(SimpleNamespace(), time.time()) is None


# ── scheduler tool ─────────────────────────────────────────────────


def _registry(ctx):
    captured = {}

    class _Registry:
        def __init__(self):
            self.context = ctx

        def register(self, name, fn=None, **kw):
            if fn is None:
                def _deco(f):
                    captured[name] = f
                    return f
                return _deco
            captured[name] = fn
            return fn

    return _Registry(), captured


def test_tool_registered_and_runs_one_tick(monkeypatch, tl, mon):
    from nomorals.health import drift as drift_module
    from nomorals.health import timeline as timeline_module
    from nomorals.tools import autonomy as autonomy_tools

    monkeypatch.setattr(timeline_module, "HealthTimeline", lambda *a, **k: tl)
    monkeypatch.setattr(
        drift_module, "DriftMonitor", lambda *a, **k: mon)
    reg, captured = _registry(SimpleNamespace(workspace=None))
    autonomy_tools.register(reg)
    assert "__health_drift_tick__" in captured
    now = time.time()
    _log_3days(tl, now, [7.5, 8.0, 7.0], [4, 4, 5])
    out = captured["__health_drift_tick__"]()
    assert out.get("ok") is True
    assert out["drift"] is False


def test_tool_never_raises(monkeypatch):
    from nomorals.health import timeline as timeline_module
    from nomorals.tools import autonomy as autonomy_tools

    def _boom(*a, **k):
        raise RuntimeError("drift down")

    monkeypatch.setattr(timeline_module, "HealthTimeline", _boom)
    reg, captured = _registry(SimpleNamespace(workspace=None))
    autonomy_tools.register(reg)
    out = captured["__health_drift_tick__"]()
    assert out.get("ok") is False
    assert "error" in out
