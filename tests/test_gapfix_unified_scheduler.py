"""Gapfix lane 12 — the autonomy unified scheduler (audit gap R1-11)
actually runs.

Before the fix: ``build_stack`` is called in production only by
``nm why`` and ``run_due()`` only from ``nm why run`` — the unified
scheduler has no live driver, so a mission scheduled on it would only
fire if someone happened to run ``nm why``. These tests drive the
PRODUCTION entry points (the scheduler job + the
``__unified_scheduler_tick__`` tool + the tick core), never the
dormant organs in isolation.
"""

from __future__ import annotations

import time
from types import SimpleNamespace


def _scheduler_ctx(tmp_path, *, home=None):
    from nomorals.storage.db import Database

    db = Database(":memory:")
    db.migrate()
    settings = SimpleNamespace(
        partner=SimpleNamespace(owner_chats="telegram:111"),
        resolve=lambda rel: tmp_path / rel,
        workspace_dir=str(tmp_path),
        home=str(home if home is not None else tmp_path),
    )
    return SimpleNamespace(db=db, settings=settings, tools=None, extras={})


def _capture_tools(ctx):
    """Register the autonomy tools into a capturing registry."""
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
    return captured


class _FakeScheduler:
    """Minimal live-Scheduler shape for ensure_*_job tests."""

    def __init__(self):
        self.jobs: list[dict] = []
        self.add_calls = 0

    def list_jobs(self):
        return list(self.jobs)

    def add(self, name, spec, kind, payload, *, routine=None):
        job = {"id": f"job-{len(self.jobs)}", "name": name,
               "spec": spec, "kind": kind, "payload": payload,
               "routine": routine}
        self.jobs.append(job)
        self.add_calls += 1
        return job

    def remove(self, ref):
        self.jobs = [j for j in self.jobs if j["id"] != ref]
        return True


def _echo(mission, step):
    return "echo-ok"


def _due_mission(*, goal="deliver the thing"):
    return {"goal": goal,
            "plan": [{"name": "echo", "handler": "echo"}],
            "due_at": time.time() - 60}


# ── ensure ────────────────────────────────────────────────────────────


def test_ensure_registers_durable_job():
    """The boot job is a durable tool job on the live scheduler:
    name, 5m cadence, the tick tool, routine (silent at the
    scheduler — firings explain themselves into the why store)."""
    from nomorals.autonomy.unified import (
        UNIFIED_SCHEDULER_JOB,
        UNIFIED_SCHEDULER_SPEC,
        UNIFIED_SCHEDULER_TOOL,
        ensure_unified_scheduler_job,
    )

    fake = _FakeScheduler()
    job = ensure_unified_scheduler_job(fake)
    assert job["name"] == UNIFIED_SCHEDULER_JOB
    assert job["spec"] == UNIFIED_SCHEDULER_SPEC == "every 5m"
    assert job["payload"] == {"tool": UNIFIED_SCHEDULER_TOOL, "args": {}}
    assert job["routine"] is True
    assert fake.add_calls == 1


def test_ensure_idempotent_same_spec():
    """A second boot at the same cadence returns the existing job —
    no duplicate driver."""
    from nomorals.autonomy.unified import ensure_unified_scheduler_job

    fake = _FakeScheduler()
    first = ensure_unified_scheduler_job(fake)
    second = ensure_unified_scheduler_job(fake)
    assert second is first
    assert fake.add_calls == 1


def test_ensure_replaces_stale_spec():
    """A job at a stale cadence is replaced, not left running."""
    from nomorals.autonomy.unified import (
        UNIFIED_SCHEDULER_SPEC,
        ensure_unified_scheduler_job,
    )

    fake = _FakeScheduler()
    fake.add("unified scheduler tick", "every 1h", "tool",
             {"tool": "__unified_scheduler_tick__", "args": {}},
             routine=True)
    job = ensure_unified_scheduler_job(fake)
    assert job["spec"] == UNIFIED_SCHEDULER_SPEC
    assert len(fake.jobs) == 1
    assert fake.add_calls == 2


# ── the tick fires real work ──────────────────────────────────────────


def test_tick_fires_due_mission_on_shared_stack(tmp_path):
    """A mission scheduled on the boot-built stack fires on the tick
    — not only when someone runs ``nm why`` — and the firing is
    explained into the why store a fresh stack (what ``nm why``
    builds) can read."""
    from nomorals.autonomy.wiring import build_stack

    stack = build_stack(tmp_path)
    stack.scheduler.register_handler("echo", _echo)
    mission_id = stack.schedule_mission(_due_mission())

    ctx = _scheduler_ctx(tmp_path)
    ctx.extras["autonomy_stack"] = stack
    tools = _capture_tools(ctx)
    result = tools["__unified_scheduler_tick__"]()
    assert result["ok"] is True
    assert result["fired"] == 1
    assert result["deferred"] == 0
    assert result["failed"] == 0
    assert stack.scheduler.get_mission(mission_id)["status"] == "done"

    fresh = build_stack(tmp_path)
    last = fresh.why_store.why_last()
    assert last["found"] is True
    assert mission_id in last["what_fired"]


def test_tick_does_not_refire_done_mission(tmp_path):
    """A done mission stays done: the next tick fires nothing for
    it and does not run its steps a second time."""
    from nomorals.autonomy.wiring import build_stack

    stack = build_stack(tmp_path)
    calls: list = []
    stack.scheduler.register_handler(
        "echo", lambda m, s: calls.append(s["name"]) or "ok")
    stack.schedule_mission(_due_mission())

    ctx = _scheduler_ctx(tmp_path)
    ctx.extras["autonomy_stack"] = stack
    tools = _capture_tools(ctx)
    assert tools["__unified_scheduler_tick__"]()["fired"] == 1
    assert tools["__unified_scheduler_tick__"]()["fired"] == 0
    assert calls == ["echo"]


def test_rhythm_deferral_applies(tmp_path):
    """A firing the rhythm refuses is deferred, not executed — the
    mission stays pending for a later tick."""
    from nomorals.autonomy.unified import (
        UnifiedScheduler,
        run_unified_scheduler_tick_once,
    )

    class _NeverFire:
        def should_fire(self, now, *, priority="ambient",
                        kind="", force=False):
            return False

    sched = UnifiedScheduler(tmp_path / "q.json",
                             rhythm=_NeverFire())
    sched.register_handler("echo", _echo)
    mission_id = sched.schedule_mission(_due_mission())
    result = run_unified_scheduler_tick_once(sched)
    assert result["ok"] is True
    assert result["fired"] == 0
    assert result["deferred"] == 1
    assert sched.get_mission(mission_id)["status"] == "pending"


def test_budget_deferral_applies(tmp_path):
    """A firing the budget refuses is deferred, not executed."""
    from nomorals.autonomy.unified import (
        UnifiedScheduler,
        run_unified_scheduler_tick_once,
    )

    class _NoBudget:
        def check(self, organ):
            return {"allowed": False,
                    "reason": "daily tokens ceiling reached"}

    sched = UnifiedScheduler(tmp_path / "q.json",
                             budget=_NoBudget())
    sched.register_handler("echo", _echo)
    mission_id = sched.schedule_mission(_due_mission())
    result = run_unified_scheduler_tick_once(sched)
    assert result["ok"] is True
    assert result["deferred"] == 1
    assert sched.get_mission(mission_id)["status"] == "pending"


def test_failing_tick_never_kills_run_due(tmp_path):
    """One bad organ tick is recorded as failed; the due mission
    still runs and the tick tool still returns ok."""
    from nomorals.autonomy.wiring import build_stack

    stack = build_stack(tmp_path)
    stack.scheduler.register_handler("echo", _echo)
    stack.schedule_mission(_due_mission())

    def _boom():
        raise RuntimeError("tick boom")

    stack.scheduler.register_organ_tick(
        "flaky", _boom, interval=60, start_now=time.time() - 61)

    ctx = _scheduler_ctx(tmp_path)
    ctx.extras["autonomy_stack"] = stack
    tools = _capture_tools(ctx)
    result = tools["__unified_scheduler_tick__"]()
    assert result["ok"] is True
    assert result["fired"] == 1
    assert result["failed"] == 1
    assert any(r.get("name") == "flaky" and r["status"] == "failed"
               for r in result["results"])


def test_tick_tool_builds_stack_from_home_when_boot_never_ran(tmp_path):
    """The fallback path: no boot-built stack on the runtime, the
    tool builds a fresh stack from the same home — the mission
    queue is a file, so the fresh stack sees the due mission and
    fires it."""
    from nomorals.autonomy.wiring import build_stack

    producer = build_stack(tmp_path)
    producer.scheduler.register_handler("echo", _echo)
    mission_id = producer.schedule_mission(_due_mission())

    ctx = _scheduler_ctx(tmp_path, home=tmp_path)
    tools = _capture_tools(ctx)
    result = tools["__unified_scheduler_tick__"]()
    assert result["ok"] is True
    assert result["fired"] == 1
    assert build_stack(tmp_path).scheduler.get_mission(
        mission_id)["status"] == "done"


def test_tick_tool_never_raises_without_stack_or_home(tmp_path):
    """No stack on the runtime and no home to build one from: the
    tick degrades to an honest error, it never raises into the
    scheduler."""
    ctx = _scheduler_ctx(tmp_path)
    ctx.settings = SimpleNamespace(
        partner=SimpleNamespace(owner_chats="telegram:111"),
        resolve=lambda rel: tmp_path / rel,
        workspace_dir=str(tmp_path),
    )
    tools = _capture_tools(ctx)
    result = tools["__unified_scheduler_tick__"]()
    assert result["ok"] is False
    assert "error" in result
