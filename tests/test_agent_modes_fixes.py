"""Regression tests for the four modes-layer weaknesses fixed on top of
``cd87917b`` (three-agent-modes build):

1. Autonomous initiatives had no delivery/approval path — a proactive
   suggestion became an initiative whose action returned its payload,
   but it never flowed into the ``/proposals`` approve/deny queue.
2. No scheduled autonomous tick was wired — ``tick()`` ran only on
   ``/agent tick`` or programmatic calls, not cron/heartbeat.
3. ``PlanningRunner`` had no step timeout or cancellation — a hung
   ``execute_step`` blocked the run.
4. ``ModeStore`` path was cwd-relative (``data/agent-modes.json``) — chat
   and CLI could read different stores depending on working directory.

Each test pins the fixed behaviour end-to-end, including the real
scheduler firing the real tool and the real ``/proposals`` round-trip.
"""

from __future__ import annotations

import asyncio
import os
import time
from types import SimpleNamespace

import pytest

from nomorals.agents import modes as M


# ── 1. initiative → /proposals → approve executes / deny discards ────────

def _held_engine(calls, *, require=True):
    """Engine whose initiatives need approval and file into ``calls``."""
    filed: list[dict] = []

    def _sink(initiative, spec, ctx):
        filed.append({"id": initiative.id, "label": initiative.label,
                      "spec": spec, "ctx": ctx})
        return initiative.id

    engine = M.AutonomousEngine(
        proposal_sink=_sink,
        requires_approval=(lambda spec, ctx: require),
    )
    engine._test_filed = filed  # type: ignore[attr-defined]
    engine._test_calls = calls  # type: ignore[attr-defined]
    return engine


def test_held_initiative_is_proposed_not_executed():
    calls: list[str] = []
    engine = _held_engine(calls)
    engine.add_source("watcher", lambda ctx: {
        "label": "ping owner", "action": lambda c: calls.append("ran")})
    done = engine.tick({})
    assert len(done) == 1
    assert done[0].status == "proposed"
    assert calls == []  # held: the action must NOT run before approval
    assert engine.pending_proposals()[0].id == done[0].id
    assert engine._test_filed[0]["label"] == "ping owner"


def test_approve_proposal_executes_action():
    calls: list[str] = []
    engine = _held_engine(calls)
    engine.add_source("watcher", lambda ctx: {
        "label": "ping owner", "action": lambda c: calls.append("ran") or "ok"})
    (init,) = engine.tick({})
    approved = engine.approve_proposal(init.id)
    assert approved.status == "done"
    assert approved.result == "ok"
    assert calls == ["ran"]
    assert engine.pending_proposals() == []
    # second approval is honest: nothing left to approve
    with pytest.raises(KeyError):
        engine.approve_proposal(init.id)


def test_deny_proposal_discards_action():
    calls: list[str] = []
    engine = _held_engine(calls)
    engine.add_source("watcher", lambda ctx: {
        "label": "ping owner", "action": lambda c: calls.append("ran")})
    (init,) = engine.tick({})
    denied = engine.deny_proposal(init.id)
    assert denied.status == "denied"
    assert calls == []  # denied: the action never runs
    assert engine.pending_proposals() == []


def test_approval_required_without_sink_fails_closed():
    calls: list[str] = []
    engine = M.AutonomousEngine(
        requires_approval=lambda spec, ctx: True)  # no sink configured
    engine.add_source("watcher", lambda ctx: {
        "label": "ping owner", "action": lambda c: calls.append("ran")})
    (init,) = engine.tick({})
    assert init.status == "failed"
    assert "sink" in (init.error or "").lower()
    assert calls == []  # fail closed: never execute past a missing gate


def test_auto_policy_executes_immediately_without_proposal():
    calls: list[str] = []
    engine = M.AutonomousEngine()  # default: no approval gate
    engine.add_source("watcher", lambda ctx: {
        "label": "ping owner", "action": lambda c: calls.append("ran") or "ok"})
    done = engine.tick({})
    assert done[0].status == "done"
    assert calls == ["ran"]
    assert engine.pending_proposals() == []


def test_spec_level_requires_approval_overrides_callable():
    calls: list[str] = []
    filed: list[str] = []
    engine = M.AutonomousEngine(
        proposal_sink=lambda i, s, c: filed.append(i.id) or i.id,
        requires_approval=lambda spec, ctx: False,  # global policy: auto
    )
    engine.add_source("watcher", lambda ctx: [
        {"label": "held one", "requires_approval": True,
         "action": lambda c: calls.append("held")},
        {"label": "auto one", "action": lambda c: calls.append("auto")},
    ])
    done = engine.tick({})
    by_label = {i.label: i for i in done}
    assert by_label["held one"].status == "proposed"
    assert by_label["auto one"].status == "done"
    assert calls == ["auto"]  # only the auto one ran
    assert filed == [by_label["held one"].id]


def test_proactive_bridge_specs_require_approval():
    async def _suggestions(user_id, limit=5, min_confidence=0.5, context=None):
        item = SimpleNamespace(title="check the inbox",
                               confidence=0.9)
        return [item]

    proactive = SimpleNamespace(get_suggestions=_suggestions)
    source = M.proactive_source(proactive, "owner")
    specs = source({})
    assert len(specs) == 1
    assert specs[0].get("requires_approval") is True


def test_runtime_proposals_roundtrip(tmp_path, monkeypatch):
    """End-to-end: suggest-mode tick files into /proposals; approve runs
    the action (it does NOT send the proposal text as a chat message);
    deny discards it. Auto mode executes immediately, files nothing."""
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import reset_settings
    from nomorals.storage.db import Database

    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    reset_settings()
    try:
        db = Database(":memory:")
        db.migrate()
        settings = SimpleNamespace(
            partner=SimpleNamespace(autonomy_mode="suggest"))
        ctx = SimpleNamespace(db=db, settings=settings, extras={})
        sent: list[str] = []

        def _send(platform, chat, text, **kw):
            sent.append(text)
            return SimpleNamespace(ok=True, error=None)

        rt = PartnerRuntime.__new__(PartnerRuntime)
        rt.context = ctx
        rt.settings = settings
        rt.gateway = SimpleNamespace(send=_send)
        rt._autonomy = None

        calls: list[str] = []
        modes = rt.agent_modes
        modes.set_modes("telegram:123", {"autonomous"})
        modes.autonomous.add_source("watcher", lambda c: {
            "label": "ping owner",
            "action": lambda cc: calls.append("ran") or "done"})
        (init,) = modes.tick("telegram:123", {})
        assert init.status == "proposed"
        assert calls == []

        rows = rt.proposals()
        mine = [r for r in rows if r["id"] == init.id]
        assert len(mine) == 1
        assert mine[0]["queue"] == "proactive"
        assert mine[0]["kind"] == "initiative"

        res = rt.approve(init.id)
        assert res["ok"] is True
        assert calls == ["ran"]  # approve EXECUTED the action
        assert sent == []  # ...and did not send proposal text as a message
        assert [r for r in rt.proposals() if r["id"] == init.id] == []

        # deny path: a second initiative is proposed, then denied
        (init2,) = modes.tick("telegram:123", {})
        res = rt.deny(init2.id)
        assert res["ok"] is True
        assert res["status"] == "denied"
        assert calls == ["ran"]  # deny never ran the second action
        assert [r for r in rt.proposals() if r["id"] == init2.id] == []

        # auto policy: the same source executes immediately, files nothing
        settings.partner.autonomy_mode = "auto"
        (init3,) = modes.tick("telegram:123", {})
        assert init3.status == "done"
        assert calls == ["ran", "ran"]
        assert [r for r in rt.proposals()
                if r["id"] == init3.id] == []
    finally:
        reset_settings()


# ── 2. scheduled autonomous tick ──────────────────────────────────────────

def _scheduler_ctx(modes):
    from nomorals.storage.db import Database

    db = Database(":memory:")
    db.migrate()
    partner = SimpleNamespace(owner_chats="telegram:111")
    settings = SimpleNamespace(partner=partner)
    ctx = SimpleNamespace(db=db, settings=settings, tools=None,
                          extras={"agent_modes": modes})
    return ctx


def test_ensure_agent_modes_tick_job_is_idempotent(tmp_path):
    from nomorals.agents.scheduler import Scheduler

    modes = M.AgentModes(store=M.ModeStore(tmp_path / "modes.json"))
    ctx = _scheduler_ctx(modes)
    sched = Scheduler(ctx)
    first = M.ensure_agent_modes_tick_job(sched)
    second = M.ensure_agent_modes_tick_job(sched)
    assert first is not None and second is not None
    jobs = [j for j in sched.list_jobs()
            if j.get("name") == M.AGENT_MODES_TICK_JOB]
    assert len(jobs) == 1
    assert jobs[0].get("payload_kind") == "tool"
    import json as _json

    row = ctx.db.query_one(
        "SELECT payload FROM schedule_jobs WHERE id = ?", (jobs[0]["id"],))
    payload = _json.loads(row["payload"])
    assert payload.get("tool") == M.AGENT_MODES_TICK_TOOL


def test_scheduled_tick_fires_and_runs_sources(tmp_path):
    """The real scheduler fires the real ``__agent_modes_tick__`` tool,
    which ticks the modes engine — sources actually run."""
    import time as _time

    from nomorals.agents.scheduler import Scheduler
    from nomorals.tools import autonomy as autonomy_tools

    store = M.ModeStore(tmp_path / "modes.json")
    modes = M.AgentModes(store=store)
    modes.set_modes("telegram:123", {"autonomous"})
    ran: list[str] = []
    modes.autonomous.add_source(
        "watcher",
        lambda ctx: {"label": "scheduled work",
                     "action": lambda c: ran.append("ran") or "ok"})

    ctx = _scheduler_ctx(modes)
    captured: dict[str, object] = {}

    class _Registry:
        def __init__(self):
            self.context = ctx

        def register(self, name, fn=None, **kw):
            if fn is None:
                # decorator use: @register("name", description=...)
                def _deco(f):
                    captured[name] = f
                    return f
                return _deco
            captured[name] = fn
            return fn

    autonomy_tools.register(_Registry())

    def _call(tool, capabilities=None, **kw):
        fn = captured[tool]
        return SimpleNamespace(ok=True, value=fn(), error=None)

    ctx.tools = SimpleNamespace(call=_call)
    sched = Scheduler(ctx)
    M.ensure_agent_modes_tick_job(sched)
    # force the job due now (same trick as the scheduler's own tests)
    job = [j for j in sched.list_jobs()
           if j.get("name") == M.AGENT_MODES_TICK_JOB][0]
    ctx.db.execute(
        "UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
        (_time.time() - 1, job["id"]))
    outcomes = sched.tick()
    assert ran == ["ran"]
    assert any(o.get("id") == job["id"] for o in outcomes)
    assert modes.autonomous.history()[-1].label == "scheduled work"


def test_tick_all_only_ticks_autonomous_chats(tmp_path):
    store = M.ModeStore(tmp_path / "modes.json")
    modes = M.AgentModes(store=store)
    modes.set_modes("telegram:1", {"autonomous"})
    modes.set_modes("telegram:2", {"planning"})
    ran: list[str] = []
    modes.autonomous.add_source(
        "watcher",
        lambda ctx: {"label": f"work for {ctx.get('chat_key')}",
                     "action": lambda c: ran.append(ctx.get("chat_key"))})
    out = modes.tick_all({})
    assert list(out) == ["telegram:1"]
    assert ran == ["telegram:1"]


# ── 3. PlanningRunner step timeout + cancellation ─────────────────────────

def _one_step_decompose(goal, evidence=None):
    return [{"name": "only", "description": "only step", "action": "wait",
             "parameters": {"seconds": 0}, "depends_on": []}]


def test_hung_async_step_times_out_and_run_reports_cancelled():
    runner = M.PlanningRunner(step_timeout_s=0.2)

    async def _hang(step, ctx):
        await asyncio.sleep(30)
        return "never"

    started = time.time()
    result = asyncio.run(runner.run(
        "goal", decompose=_one_step_decompose, execute_step=_hang))
    elapsed = time.time() - started
    assert elapsed < 10  # aborted, not stuck for 30s
    assert result.success is False
    assert result.cancelled is True
    assert result.steps_completed == 0
    assert result.results == {}  # half-executed step never claimed done
    assert any("timed out" in e for e in result.errors)


def test_hung_sync_step_times_out():
    runner = M.PlanningRunner(step_timeout_s=0.2)

    def _hang(step, ctx):
        time.sleep(30)
        return "never"

    started = time.time()
    result = asyncio.run(runner.run(
        "goal", decompose=_one_step_decompose, execute_step=_hang))
    assert time.time() - started < 10
    assert result.success is False
    assert result.cancelled is True
    assert result.results == {}


def test_timeout_does_not_replan():
    runner = M.PlanningRunner(step_timeout_s=0.2)
    calls = {"n": 0}

    def _decompose(goal, evidence=None):
        calls["n"] += 1
        return _one_step_decompose(goal, evidence)

    async def _hang(step, ctx):
        await asyncio.sleep(30)
        return "never"

    result = asyncio.run(runner.run(
        "goal", decompose=_decompose, execute_step=_hang, max_replans=2))
    assert calls["n"] == 1  # timeout is terminal, not a replan trigger
    assert result.cancelled is True


def test_cancel_event_stops_run_cleanly():
    runner = M.PlanningRunner()
    import threading

    stop = threading.Event()
    executed: list[str] = []

    def _decompose(goal, evidence=None):
        return [
            {"name": "first", "description": "1", "action": "wait",
             "parameters": {"seconds": 0}, "depends_on": []},
            {"name": "second", "description": "2", "action": "wait",
             "parameters": {"seconds": 0}, "depends_on": ["step_1"]},
            {"name": "third", "description": "3", "action": "wait",
             "parameters": {"seconds": 0}, "depends_on": ["step_2"]},
        ]

    def _exec(step, ctx):
        executed.append(step.name)
        if step.name == "first":
            stop.set()  # cancel requested after the first step
        return f"{step.name}-done"

    result = asyncio.run(runner.run(
        "goal", decompose=_decompose, execute_step=_exec, cancel=stop))
    assert executed == ["first"]  # later steps never ran
    assert result.cancelled is True
    assert result.success is False
    assert result.steps_completed == 1
    assert result.results == {"step_1": "first-done"}  # finished work kept


def test_cancel_before_start_runs_nothing():
    import threading

    runner = M.PlanningRunner()
    stop = threading.Event()
    stop.set()
    executed: list[str] = []

    result = asyncio.run(runner.run(
        "goal", decompose=_one_step_decompose,
        execute_step=lambda s, c: executed.append(s.name),
        cancel=stop))
    assert executed == []
    assert result.cancelled is True
    assert result.steps_completed == 0


def test_run_sync_honours_timeout():
    runner = M.PlanningRunner(step_timeout_s=0.2)

    async def _hang(step, ctx):
        await asyncio.sleep(30)
        return "never"

    started = time.time()
    result = runner.run_sync(
        "goal", decompose=_one_step_decompose, execute_step=_hang)
    assert time.time() - started < 10
    assert result.cancelled is True


# ── 4. ModeStore resolves against the settings data dir ───────────────────

def test_default_store_same_from_two_cwds(tmp_path, monkeypatch):
    from nomorals.core.config import reset_settings

    home = tmp_path / "home"
    cwd_a = tmp_path / "a"
    cwd_b = tmp_path / "b"
    cwd_a.mkdir()
    cwd_b.mkdir()
    monkeypatch.setenv("NM_HOME", str(home))
    monkeypatch.delenv("NM_AGENT_MODES_STORE", raising=False)
    reset_settings()
    try:
        monkeypatch.chdir(cwd_a)
        store_a = M.ModeStore()
        store_a.set("telegram:1", {"planning"})
        monkeypatch.chdir(cwd_b)
        store_b = M.ModeStore()
        assert store_b.path == store_a.path  # same file, not cwd-relative
        assert store_b.get("telegram:1") == frozenset({"planning"})
        assert store_a.path == home / "data" / "agent-modes.json"
    finally:
        reset_settings()


def test_explicit_absolute_store_env_wins(tmp_path, monkeypatch):
    from nomorals.core.config import reset_settings

    target = tmp_path / "custom" / "modes.json"
    monkeypatch.setenv("NM_AGENT_MODES_STORE", str(target))
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    reset_settings()
    try:
        assert M.ModeStore().path == target
    finally:
        reset_settings()


def test_relative_store_env_resolves_against_home(tmp_path, monkeypatch):
    from nomorals.core.config import reset_settings

    home = tmp_path / "home"
    monkeypatch.setenv("NM_AGENT_MODES_STORE", "data/custom-modes.json")
    monkeypatch.setenv("NM_HOME", str(home))
    reset_settings()
    try:
        assert M.ModeStore().path == home / "data" / "custom-modes.json"
    finally:
        reset_settings()


def test_agent_modes_shares_store_across_cwds(tmp_path, monkeypatch):
    from nomorals.core.config import reset_settings

    home = tmp_path / "home"
    cwd_a = tmp_path / "a"
    cwd_b = tmp_path / "b"
    cwd_a.mkdir()
    cwd_b.mkdir()
    monkeypatch.setenv("NM_HOME", str(home))
    monkeypatch.delenv("NM_AGENT_MODES_STORE", raising=False)
    reset_settings()
    try:
        monkeypatch.chdir(cwd_a)
        agent_a = M.AgentModes()
        agent_a.set_modes("telegram:9", {"autonomous", "planning"})
        monkeypatch.chdir(cwd_b)
        agent_b = M.AgentModes()
        assert agent_b.modes_for("telegram:9") == frozenset(
            {"autonomous", "planning"})
    finally:
        reset_settings()
