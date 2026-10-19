"""PASS #11: scheduler — durable job registration for the autonomy core.

Two verified bugs, failing-first:

1. ``ensure_agency_shift_job`` was not idempotent against the REAL
   scheduler: ``list_jobs()`` formats cron specs with a ``"cron "``
   prefix (``"cron */30 * * * *"``), but the ensure compared verbatim
   against ``"*/30 * * * *"`` — so every boot removed and re-added the
   job (ID churn, orphaned run history, lost retry/last_result state).
   The old test only passed because its FakeScheduler stored raw specs.

2. The agency shift and the agent-modes tick inherited the scheduler's
   300s ``wall_seconds`` cap: a shift legitimately running longer than
   5 minutes was marked ``"job failed: timed out"`` (worker thread
   abandoned but still running), the owner got a false failure alert,
   and the work's result was lost — the blunt-timeout pattern the
   heartbeat design replaced. Both jobs now register an explicit
   generous ``run_timeout_s`` (1800s = the shift's own interval; a job
   that cannot finish inside its own slot is genuinely wedged), and
   existing registrations are upgraded on the next ensure.
"""

from types import SimpleNamespace

from nomorals.agency.wiring import (
    AGENCY_SHIFT_JOB,
    ensure_agency_shift_job,
)
from nomorals.agents import modes as M
from nomorals.agents.scheduler import Scheduler
from nomorals.storage.db import Database


def _ctx(**kw):
    db = Database(":memory:")
    db.migrate()
    tools = SimpleNamespace(
        call=lambda tool, capabilities=None, **kw2: SimpleNamespace(
            ok=True, value="ok", error=None)
    )
    ctx = SimpleNamespace(db=db, tools=tools, settings=SimpleNamespace(),
                          extras={})
    for k, v in kw.items():
        setattr(ctx, k, v)
    return ctx


def _shift_job(sched):
    jobs = [j for j in sched.list_jobs() if j.get("name") == AGENCY_SHIFT_JOB]
    assert len(jobs) == 1
    return jobs[0]


def _tick_job(sched):
    jobs = [j for j in sched.list_jobs()
            if j.get("name") == M.AGENT_MODES_TICK_JOB]
    assert len(jobs) == 1
    return jobs[0]


# ── Bug 1: shift ensure must be idempotent against the real scheduler ──

def test_shift_ensure_idempotent_real_scheduler():
    sched = Scheduler(_ctx())
    first = ensure_agency_shift_job(sched)
    second = ensure_agency_shift_job(sched)
    jobs = [j for j in sched.list_jobs() if j.get("name") == AGENCY_SHIFT_JOB]
    assert len(jobs) == 1, "ensure re-registered the shift job"
    assert jobs[0]["id"] == first["id"] == second["id"], \
        "job id churned across ensures (stored spec has 'cron ' prefix)"


def test_shift_ensure_replaces_stale_spec():
    sched = Scheduler(_ctx())
    first = ensure_agency_shift_job(sched)
    # a genuinely different cadence still replaces the job
    third = ensure_agency_shift_job(sched, spec="0 * * * *")
    jobs = [j for j in sched.list_jobs() if j.get("name") == AGENCY_SHIFT_JOB]
    assert len(jobs) == 1
    assert jobs[0]["id"] != first["id"]
    assert third["id"] == jobs[0]["id"]


# ── Bug 2: generous run cap on the autonomy core jobs ──

def test_shift_registers_generous_run_timeout():
    sched = Scheduler(_ctx())
    ensure_agency_shift_job(sched)
    assert _shift_job(sched)["run_timeout_s"] == 1800.0


def test_shift_upgrades_legacy_run_timeout():
    # an existing deployment registered the job before the cap existed
    # (run_timeout_s=0 → scheduler wall_seconds=300s fallback)
    sched = Scheduler(_ctx())
    sched.add(AGENCY_SHIFT_JOB, "*/30 * * * *", "tool",
              {"tool": "__agency_shift__"})
    assert _shift_job(sched)["run_timeout_s"] == 0.0
    ensure_agency_shift_job(sched)
    assert _shift_job(sched)["run_timeout_s"] == 1800.0


def test_modes_tick_registers_generous_run_timeout():
    sched = Scheduler(_ctx())
    M.ensure_agent_modes_tick_job(sched)
    assert _tick_job(sched)["run_timeout_s"] == 1800.0


def test_modes_tick_upgrades_legacy_run_timeout():
    sched = Scheduler(_ctx())
    sched.add(M.AGENT_MODES_TICK_JOB, M.AGENT_MODES_TICK_SPEC, "tool",
              {"tool": M.AGENT_MODES_TICK_TOOL, "args": {}})
    assert _tick_job(sched)["run_timeout_s"] == 0.0
    M.ensure_agent_modes_tick_job(sched)
    assert _tick_job(sched)["run_timeout_s"] == 1800.0


def test_modes_tick_still_idempotent():
    sched = Scheduler(_ctx())
    first = M.ensure_agent_modes_tick_job(sched)
    second = M.ensure_agent_modes_tick_job(sched)
    assert first["id"] == second["id"]
    assert len([j for j in sched.list_jobs()
                if j.get("name") == M.AGENT_MODES_TICK_JOB]) == 1
