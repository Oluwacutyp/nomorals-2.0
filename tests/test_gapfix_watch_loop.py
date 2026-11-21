"""Gapfix R2-22 — the resident WatchLoop starts at operator boot.

Failing-first: the job registrar, the scheduler tool, and the
``nm watch loop`` verb do not exist yet, so the Job/Tool/CLI tests
fail before wiring. The Tick tests pin the EXISTING organ
(``WatchLoop.tick``) — they pass before wiring and guard the wiring
against re-implementing the tick instead of running it.
"""
from types import SimpleNamespace

import nomorals.agents.notifier as notifier_mod
import nomorals.agents.ops_alerts as ops_mod
import nomorals.agents.reasoning as reasoning_mod

_FINDING = {"kind": "stuck_mission", "key": "stuck:abc123",
            "title": "Mission stuck: abc123 — dig the well",
            "body": "0 iteration(s), 0 step(s) done", "data": {}}
_REPORT = {"sent": [_FINDING], "self_healed": [], "cooldown": [],
           "findings": [_FINDING], "pending": 2}


def _patch_surfaces(monkeypatch, *, redelivered=2, mined=("p1",)):
    monkeypatch.setattr(
        ops_mod, "ops_alerts",
        lambda ctx, heal_background=True: dict(_REPORT))
    monkeypatch.setattr(notifier_mod, "Notifier",
                        _Notifier(result=redelivered))
    monkeypatch.setattr(reasoning_mod, "ReasoningAgent",
                        _Reasoner(result=list(mined)))


class _Notifier:
    def __init__(self, result=0):
        self._result = result

    def __call__(self, ctx):
        outer = self

        class _N:
            def redeliver(self):
                return outer._result

        return _N()


class _Reasoner:
    def __init__(self, result=()):
        self._result = list(result)

    def __call__(self, ctx):
        outer = self

        class _R:
            def mine_preventions(self):
                return list(outer._result)

        return _R()


class _Sched:
    """Minimal stand-in for agents.scheduler.Scheduler (digest pattern).

    Mirrors the real Scheduler.add signature, including
    ``overlap_policy`` (scheduler.py) — the watch-loop ensure
    registers overlap skip and reconciles persisted rows via
    ``set_overlap_policy``, so a fake without those would fail the
    registration instead of testing it.
    """

    def __init__(self):
        self.jobs = []

    def list_jobs(self):
        return list(self.jobs)

    def add(self, name, spec, kind, payload, routine=False,
            overlap_policy="concurrent"):
        job = {"id": f"job-{len(self.jobs)}", "name": name, "spec": spec,
               "kind": kind, "payload": payload, "routine": routine,
               "overlap_policy": overlap_policy}
        self.jobs.append(job)
        return job

    def set_overlap_policy(self, job_id, policy):
        for job in self.jobs:
            if job.get("id") == job_id:
                job["overlap_policy"] = policy

    def remove(self, job_id):
        self.jobs = [j for j in self.jobs if j.get("id") != job_id]


# ── job registration ──────────────────────────────────────────────

def test_ensure_registers_durable_routine_tool_job():
    from nomorals.agents.watcher import (
        WATCH_LOOP_JOB,
        WATCH_LOOP_TOOL,
        ensure_watch_loop_job,
    )

    sched = _Sched()
    job = ensure_watch_loop_job(sched)
    assert job["name"] == WATCH_LOOP_JOB
    assert job["spec"] == "every 5m"
    assert job["kind"] == "tool"
    assert job["payload"] == {"tool": WATCH_LOOP_TOOL, "args": {}}
    assert job["routine"] is True
    assert len(sched.jobs) == 1


def test_ensure_is_idempotent():
    from nomorals.agents.watcher import ensure_watch_loop_job

    sched = _Sched()
    first = ensure_watch_loop_job(sched)
    second = ensure_watch_loop_job(sched)
    assert second == first
    assert len(sched.jobs) == 1


def test_ensure_replaces_stale_spec():
    from nomorals.agents.watcher import (
        WATCH_LOOP_JOB,
        ensure_watch_loop_job,
    )

    sched = _Sched()
    sched.jobs.append({"id": "old", "name": WATCH_LOOP_JOB,
                       "spec": "every 1h"})
    job = ensure_watch_loop_job(sched)
    assert job["spec"] == "every 5m"
    assert len(sched.jobs) == 1
    assert sched.jobs[0].get("id") != "old"


def test_ensure_none_scheduler_returns_none():
    from nomorals.agents.watcher import ensure_watch_loop_job

    assert ensure_watch_loop_job(None) is None


# ── the existing tick (pins the organ; passes before wiring) ──────

def test_tick_pages_finding_and_redelivers(monkeypatch):
    from nomorals.agents.watcher import WatchLoop

    _patch_surfaces(monkeypatch)
    out = WatchLoop(SimpleNamespace(workspace=None)).tick()
    assert out["tick"] == 1
    assert out["paged"] == ["Mission stuck: abc123 — dig the well"]
    assert out["redelivered"] == 2
    assert out["mined"] == 1
    assert out["pending"] == 2
    assert out["findings"] == 1


def test_tick_quiet_world_reports_all_quiet(monkeypatch):
    from nomorals.agents.watcher import WatchLoop

    monkeypatch.setattr(
        ops_mod, "ops_alerts",
        lambda ctx, heal_background=True: {
            "sent": [], "self_healed": [], "cooldown": [],
            "findings": [], "pending": 0})
    monkeypatch.setattr(notifier_mod, "Notifier", _Notifier(result=0))
    monkeypatch.setattr(reasoning_mod, "ReasoningAgent", _Reasoner())
    out = WatchLoop(SimpleNamespace(workspace=None)).tick()
    assert out["paged"] == []
    assert out["redelivered"] == 0
    assert "all quiet" in WatchLoop.format(out)


# ── scheduler tool ────────────────────────────────────────────────

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


def test_tool_registered_and_runs_one_tick(monkeypatch):
    from nomorals.tools import autonomy as autonomy_tools

    _patch_surfaces(monkeypatch)
    reg, captured = _registry(SimpleNamespace(workspace=None))
    autonomy_tools.register(reg)
    assert "__watch_loop_tick__" in captured
    out = captured["__watch_loop_tick__"]()
    assert out.get("ok") is True
    assert out["paged"] == ["Mission stuck: abc123 — dig the well"]
    assert out["redelivered"] == 2
    assert out["mined"] == 1


def test_tool_never_raises(monkeypatch):
    from nomorals.tools import autonomy as autonomy_tools

    def _boom(ctx, heal_background=True):
        raise RuntimeError("scan boom")

    monkeypatch.setattr(ops_mod, "ops_alerts", _boom)
    reg, captured = _registry(SimpleNamespace(workspace=None))
    autonomy_tools.register(reg)
    out = captured["__watch_loop_tick__"]()
    assert out.get("ok") is False
    assert "error" in out


# ── CLI verb ──────────────────────────────────────────────────────

def test_watch_loop_runs_one_tick_and_prints(monkeypatch, capsys):
    from nomorals.cmdline.commands.security import _cmd_watch

    _patch_surfaces(monkeypatch)
    ctx = SimpleNamespace(workspace=None)
    args = SimpleNamespace(action="loop", ref="", severity="",
                           interval=None, quiet="", channels="",
                           limit=50, json=False)
    rc = _cmd_watch(args, ctx)
    assert rc == 0
    text = capsys.readouterr().out
    # format() prints counts, not titles — the titles ride the report
    # dict (asserted in the tool test); the printed line is the count.
    assert "tick 1" in text
    assert "1 paged" in text
    assert "2 alert(s) redelivered" in text


def test_parser_accepts_watch_loop():
    from nomorals.cmdline.parser import _parser

    args = _parser().parse_args(["watch", "loop"])
    assert args.action == "loop"
