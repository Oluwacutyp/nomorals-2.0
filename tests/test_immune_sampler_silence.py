"""Pass 3 — the immune sampler's reporting contract.

Before this pass, the durable sampler job ("immune baseline
sampler", every 1m) alerted the owner on EVERY run through the
scheduler's generic per-run success alert: the job name carried none
of the scheduler's routine markers, so a tick that sampled nothing
reached chat as "sampled: no, learned: no … reason: no live process
sample available" — a non-event reported as news, forever. Real
events (gate enforcement, incident close) had the opposite problem:
their announcements accumulated in ``monitor.announcements``, which
nothing consumed.

The contract pinned here:

* a no-sample / routine / already-announced-incident tick produces
  NO notification of any kind, while still being recorded (job
  last_result, persisted store timestamps);
* a real event — enforcement, an announcement (fever start,
  incident close, baseline ready), a tick error — produces exactly
  ONE notification, published by the immune tool itself, carrying
  the specifics;
* the scheduler's shared alert path is untouched: other jobs still
  alert on success exactly as before;
* a legacy-named sampler job from before the rename is retired, not
  left running alongside the new one.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from nomorals.immune.baseline import MIN_SAMPLES
from nomorals.immune.wiring import (
    IMMUNE_SAMPLER_JOB,
    IMMUNE_SAMPLER_TOOL,
    LEGACY_IMMUNE_SAMPLER_JOB,
    ImmuneMonitor,
    default_store_path,
    ensure_immune_sampler_job,
    reset_shared_monitors,
    shared_monitor,
    summarize_tick_event,
)

NORMAL = {"processes_spawned": 100.0, "process_growth": 1.0}
RUNAWAY = {"processes_spawned": 5000.0, "process_growth": 50.0}


@pytest.fixture(autouse=True)
def _clean_monitors():
    reset_shared_monitors()
    yield
    reset_shared_monitors()


def _ctx(tmp_path):
    from nomorals.storage.db import Database

    # File-backed, not ":memory:": Database connections are
    # per-thread, and the scheduler runs tool payloads on a worker
    # thread — a :memory: db would give that thread its own empty
    # database (production is file-backed and unaffected).
    db = Database(tmp_path / "test.db")
    db.migrate()
    settings = SimpleNamespace(
        partner=SimpleNamespace(owner_chats="telegram:111"),
        resolve=lambda rel: tmp_path / rel,
        workspace_dir=str(tmp_path),
    )
    return SimpleNamespace(db=db, settings=settings, tools=None, extras={})


def _capture_tools(ctx):
    """Register the real autonomy tools into a capture registry."""
    from nomorals.tools import autonomy as autonomy_tools

    captured: dict = {}

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

    autonomy_tools.register(_Registry())

    def _call(tool, capabilities=None, **kw):
        return SimpleNamespace(ok=True, value=captured[tool](), error=None)

    ctx.tools = SimpleNamespace(call=_call)
    return captured


def _notifications(ctx):
    return ctx.db.query(
        "SELECT kind, title, body FROM notifications ORDER BY created_at")


def _run_due(sched, ctx, job_id):
    ctx.db.execute(
        "UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
        (time.time() - 1, job_id))
    return sched.tick()


def _world(tmp_path, collect):
    """ctx + scheduler + the shared monitor the tool will use.

    ``collect`` is a zero-arg callable returning a sample dict (or
    None, or raising) — wrapped to the monitor's collect signature.
    Returns (ctx, sched, captured, monitor, collect_box): flip
    ``collect_box["fn"]`` to change what later ticks collect.
    """
    from nomorals.agents.scheduler import Scheduler

    ctx = _ctx(tmp_path)
    captured = _capture_tools(ctx)
    box = {"fn": collect}
    monitor = shared_monitor(
        default_store_path(ctx), collect=lambda baseline: box["fn"]())
    sched = Scheduler(ctx)
    return ctx, sched, captured, monitor, box


def _sampler_job_id(sched):
    jobs = [j for j in sched.list_jobs()
            if j.get("name") == IMMUNE_SAMPLER_JOB]
    assert len(jobs) == 1
    return jobs[0]["id"]


# ── the silence: non-events never notify ──────────────────────────

def test_no_sample_ticks_are_silent_but_recorded(tmp_path):
    ctx, sched, captured, monitor, box = _world(tmp_path, lambda: None)
    ensure_immune_sampler_job(sched)
    job_id = _sampler_job_id(sched)
    for _ in range(3):
        _run_due(sched, ctx, job_id)
    # not one notification — no "schedule" alert, no immune event
    assert _notifications(ctx) == []
    # …but the ticks are on record: the job's last_result renders the
    # honest no-sample report, and the store persisted the tick time
    row = ctx.db.query_one(
        "SELECT last_result FROM schedule_jobs WHERE id = ?", (job_id,))
    assert "sampled: no" in row["last_result"]
    assert "no live process sample" in row["last_result"]
    store = json.loads(
        (tmp_path / "data" / "immune-baseline.json").read_text())
    assert store["last_tick_at"] > 0
    assert store["samples"] == []
    assert monitor.status()["sampler_running"] is True


def test_routine_learned_ticks_are_silent(tmp_path):
    ctx, sched, captured, monitor, box = _world(
        tmp_path, lambda: dict(NORMAL))
    tool = captured[IMMUNE_SAMPLER_TOOL]
    # samples land and are learned — routine telemetry, not news
    for _ in range(MIN_SAMPLES - 1):
        out = tool()
        assert out["learned"] is True
    assert _notifications(ctx) == []


def test_quarantine_ticks_under_announced_incident_are_silent(tmp_path):
    ctx, sched, captured, monitor, box = _world(
        tmp_path, lambda: dict(NORMAL))
    tool = captured[IMMUNE_SAMPLER_TOOL]
    for _ in range(40):
        monitor.tick(sample=dict(NORMAL))
    monitor.drain_announcements()  # baseline-ready milestone, delivered
    box["fn"] = lambda: dict(RUNAWAY)
    out = tool()
    assert out["enforced"] is True
    assert len(_notifications(ctx)) == 1  # the enforcement (next test)
    # the incident is open and announced; further runaway ticks just
    # quarantine — no per-tick paging
    for _ in range(3):
        out = tool()
        assert out["incident"] is True and out["learned"] is False
    assert len(_notifications(ctx)) == 1


# ── the events: exactly one notification, with specifics ──────────

def test_baseline_ready_milestone_notifies_once(tmp_path):
    ctx, sched, captured, monitor, box = _world(
        tmp_path, lambda: dict(NORMAL))
    tool = captured[IMMUNE_SAMPLER_TOOL]
    for _ in range(MIN_SAMPLES - 1):
        tool()
    assert _notifications(ctx) == []
    out = tool()  # the sample that makes the baseline ready
    assert out["learned"] is True
    notes = _notifications(ctx)
    assert len(notes) == 1
    assert notes[0]["kind"] == "immune"
    assert "baseline ready" in notes[0]["title"]
    assert str(MIN_SAMPLES) in notes[0]["title"]
    # and it does not repeat on the next routine tick
    tool()
    assert len(_notifications(ctx)) == 1


def test_enforcement_notifies_once_with_specifics(tmp_path):
    ctx, sched, captured, monitor, box = _world(
        tmp_path, lambda: dict(NORMAL))
    for _ in range(40):
        monitor.tick(sample=dict(NORMAL))
    monitor.drain_announcements()
    ensure_immune_sampler_job(sched)
    job_id = _sampler_job_id(sched)
    box["fn"] = lambda: dict(RUNAWAY)
    _run_due(sched, ctx, job_id)
    notes = _notifications(ctx)
    assert len(notes) == 1
    note = notes[0]
    assert note["kind"] == "immune"  # never the generic scheduled alert
    assert "ENFORCED" in note["title"]
    assert "5000" in note["body"]  # the offending sample, named
    # Formatting F5: the stored body is the alert Doc's canonical
    # body — the sample's facts ride as labeled rows ("Quarantined")
    # instead of the old composed prose line; the specifics named
    # here are unchanged.
    assert "quarantined" in note["body"].lower()
    # the runaway stops: the closing tick is itself an event, once
    box["fn"] = lambda: dict(NORMAL)
    _run_due(sched, ctx, job_id)
    notes = _notifications(ctx)
    assert len(notes) == 2
    assert "incident closed" in notes[1]["title"]


def test_error_tick_notifies_as_error_once(tmp_path):
    def _boom():
        raise RuntimeError("collect boom")

    ctx, sched, captured, monitor, box = _world(tmp_path, _boom)
    ensure_immune_sampler_job(sched)
    job_id = _sampler_job_id(sched)
    _run_due(sched, ctx, job_id)
    notes = _notifications(ctx)
    assert len(notes) == 1
    assert notes[0]["kind"] == "immune"
    assert "error" in notes[0]["title"]
    assert "collect boom" in notes[0]["body"]
    # a persisting error does not become per-minute spam either:
    # the notifier's dedupe collapses the identical alert
    _run_due(sched, ctx, job_id)
    assert len(_notifications(ctx)) == 1


# ── the shared path and the rename ────────────────────────────────

def test_other_jobs_still_alert_on_success(tmp_path):
    """The scheduler's generic alert is untouched for everyone else."""
    from nomorals.agents.scheduler import Scheduler

    ctx = _ctx(tmp_path)
    captured = _capture_tools(ctx)
    captured["owner_report"] = lambda: {"ok": True, "note": "hello"}
    sched = Scheduler(ctx)
    job = sched.add("owner report", "every 1h", "tool",
                    {"tool": "owner_report", "args": {}})
    _run_due(sched, ctx, job["id"])
    notes = _notifications(ctx)
    assert len(notes) == 1
    assert notes[0]["kind"] == "schedule"
    assert "owner report" in notes[0]["title"]


def test_legacy_sampler_job_is_retired_not_duplicated(tmp_path):
    from nomorals.agents.scheduler import Scheduler

    sched = Scheduler(_ctx(tmp_path))
    sched.add(LEGACY_IMMUNE_SAMPLER_JOB, "every 1m", "tool",
              {"tool": IMMUNE_SAMPLER_TOOL, "args": {}})
    ensure_immune_sampler_job(sched)
    names = [j.get("name") for j in sched.list_jobs()]
    assert LEGACY_IMMUNE_SAMPLER_JOB not in names
    assert names.count(IMMUNE_SAMPLER_JOB) == 1
    kept_id = [j for j in sched.list_jobs()
               if j.get("name") == IMMUNE_SAMPLER_JOB][0]["id"]
    ensure_immune_sampler_job(sched)  # idempotent thereafter
    jobs = [j for j in sched.list_jobs()
            if j.get("name") == IMMUNE_SAMPLER_JOB]
    assert len(jobs) == 1 and jobs[0]["id"] == kept_id


def test_sampler_job_registers_as_routine(tmp_path):
    # Pass #71: routine classification is a first-class flag on the
    # job definition, not a substring of its name — the sampler's
    # registrar sets it explicitly, so the scheduler silences the
    # job's successful runs whatever the job is called.  (Before #71
    # this pinned the name rule itself: "tick" in the job name.)
    from nomorals.agents.scheduler import Scheduler

    sched = Scheduler(_ctx(tmp_path))
    ensure_immune_sampler_job(sched)
    jobs = [j for j in sched.list_jobs()
            if j.get("name") == IMMUNE_SAMPLER_JOB]
    assert len(jobs) == 1
    assert jobs[0]["routine"] is True


# ── the classifier + drain, unit level ────────────────────────────

def test_summarize_tick_event_classifies():
    no_sample = {"ok": True, "sampled": False, "learned": False,
                 "enforced": False, "quarantined": 0, "samples": 0,
                 "reason": "no live process sample available on this "
                           "platform — nothing recorded"}
    assert summarize_tick_event(no_sample) is None
    learned = {"ok": True, "sampled": True, "learned": True,
               "enforced": False, "quarantined": 0, "samples": 41,
               "sample": dict(NORMAL), "reason": "below thresholds"}
    assert summarize_tick_event(learned) is None
    quarantined = dict(learned, learned=False, incident=True,
                       quarantined=3)
    assert summarize_tick_event(quarantined) is None
    enforced = dict(quarantined, enforced=True,
                    sample=dict(RUNAWAY), quarantined=1,
                    reason="gate enforced: spawned over p95")
    event = summarize_tick_event(enforced, ["fever started"])
    assert event is not None and event["critical"] is True
    assert "ENFORCED" in event["title"]
    assert "5000" in event["body"] and "fever started" in event["body"]
    announced = summarize_tick_event(learned, ["immune incident closed — x"])
    assert announced is not None and announced["critical"] is False
    assert "incident closed" in announced["title"]
    error = summarize_tick_event({"ok": False, "error": "tick failed: x"})
    assert error is not None and "error" in error["title"]
    assert "tick failed: x" in error["body"]


def test_drain_announcements_hands_over_once(tmp_path):
    mon = ImmuneMonitor(tmp_path / "b.json")
    for _ in range(40):
        mon.tick(sample=dict(NORMAL))
    first = mon.drain_announcements()
    assert any("baseline ready" in a for a in first)
    assert mon.drain_announcements() == []
    # history is kept — draining decides notification, not evidence
    assert any("baseline ready" in a for a in mon.announcements)
