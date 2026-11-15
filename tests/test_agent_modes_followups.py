"""Follow-on weaknesses for the agent-modes layer (2026-10-19 day log).

Failing-first: each test pins one of the four remainders recorded
2026-11-15 after the four §8+§4 remainders were closed no-commit
(d03c921c/aa8def19/9bc786c3):

1. held initiatives are in-memory only — a restart orphans them;
2. ``nm agent tick`` ticks a fresh sourceless engine;
3. the scheduled-tick proactive bridge uses a no-arg ProactiveEngine
   (and its time suggestions run on a hardcoded noon default);
4. ``AgentModes.handle_task`` has no production caller.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from types import SimpleNamespace

import pytest

from nomorals.agents import modes as M


def _proactive_spec(payload: dict) -> dict:
    """A proactive-source initiative spec, as the bridge builds it."""
    return {"label": f"proactive: {payload['title']}",
            "proposal": payload["title"],
            "requires_approval": True,
            "payload": dict(payload),
            "action": lambda c, p=dict(payload): dict(p)}


def _modes(tmp_path, **kw):
    store = M.ModeStore(path=tmp_path / "agent-modes.json")
    return M.AgentModes(store=store, **kw)


# ── 1. held initiatives survive a restart ────────────────────────────────

def test_held_proactive_initiative_survives_restart(tmp_path):
    payload = {"title": "file the report", "confidence": 0.9}
    filed: list = []

    def _sink(init, spec, ctx):
        filed.append(init.id)
        return init.id

    first = _modes(tmp_path, proposal_sink=_sink)
    first.set_modes("telegram:1", {"autonomous"})
    first.autonomous.add_source("proactive-test",
                                lambda ctx: _proactive_spec(payload))
    done = first.tick("telegram:1", {})
    assert len(done) == 1
    assert done[0].status == "proposed"
    assert filed == [done[0].proposal_id]
    # The hold is durable: a fresh engine on the same store sees it.
    second = _modes(tmp_path, proposal_sink=_sink)
    assert [i.proposal_id for i in second.pending_proposals()] == filed
    init = second.approve_proposal(filed[0])
    assert init.status == "done"
    assert init.result == payload


def test_deny_clears_persisted_held_record(tmp_path):
    payload = {"title": "file the report", "confidence": 0.9}
    first = _modes(tmp_path, proposal_sink=lambda i, s, c: i.id)
    first.set_modes("telegram:1", {"autonomous"})
    first.autonomous.add_source("proactive-test",
                                lambda ctx: _proactive_spec(payload))
    done = first.tick("telegram:1", {})
    pid = done[0].proposal_id
    assert M.ModeStore(path=tmp_path / "agent-modes.json"
                       ).pending_records()
    second = _modes(tmp_path, proposal_sink=lambda i, s, c: i.id)
    denied = second.deny_proposal(pid)
    assert denied.status == "denied"
    assert M.ModeStore(path=tmp_path / "agent-modes.json"
                       ).pending_records() == []
    third = _modes(tmp_path, proposal_sink=lambda i, s, c: i.id)
    assert third.pending_proposals() == []


def test_non_rebuildable_held_record_stays_orphaned_honestly(tmp_path):
    # A spec whose action is a bare closure (no payload data) cannot be
    # rebuilt after a restart: approve still fails honestly, deny clears.
    first = _modes(tmp_path, proposal_sink=lambda i, s, c: i.id)
    first.set_modes("telegram:1", {"autonomous"})
    first.autonomous.add_source(
        "closure-test",
        lambda ctx: {"label": "closure work", "proposal": "closure work",
                     "requires_approval": True,
                     "action": lambda c: "ran"})
    done = first.tick("telegram:1", {})
    pid = done[0].proposal_id
    assert len(M.ModeStore(path=tmp_path / "agent-modes.json"
                           ).pending_records()) == 1
    second = _modes(tmp_path, proposal_sink=lambda i, s, c: i.id)
    assert second.pending_proposals() == []
    with pytest.raises(KeyError):
        second.approve_proposal(pid)
    second.deny_proposal(pid)
    assert M.ModeStore(path=tmp_path / "agent-modes.json"
                       ).pending_records() == []


# ── 2. ``nm agent tick`` ticks a real engine ─────────────────────────────

class _FakeDB:
    """Records the SQL the proposal sink files."""

    def __init__(self) -> None:
        self.executed: list = []

    def execute(self, sql, params=()):
        self.executed.append((sql, params))
        return None


class _StubEngine:
    """A ProactiveEngine that always suggests one thing."""

    async def get_suggestions(self, user_id, *, limit=5,
                              min_confidence=0.5, context=None):
        return [SimpleNamespace(title="file the report", confidence=0.9)]


def test_cli_agent_tick_runs_proactive_source(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NM_AGENT_MODES_STORE",
                       str(tmp_path / "agent-modes.json"))
    M.AgentModes().set_modes("cli", {"autonomous"})
    monkeypatch.setattr("nomorals.agents.proactive.ProactiveEngine",
                        _StubEngine)
    from nomorals.cmdline.commands.agent import _cmd_agent

    db = _FakeDB()
    args = argparse.Namespace(chat="cli", mode_args=["tick"])
    assert _cmd_agent(args, SimpleNamespace(db=db)) == 0
    out = capsys.readouterr().out
    assert "file the report" in out
    assert "nothing to do" not in out
    assert any("proactive_log" in sql for sql, _ in db.executed)


def test_with_production_sources_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr("nomorals.agents.proactive.ProactiveEngine",
                        _StubEngine)
    agent = _modes(tmp_path)
    M.with_production_sources(agent)
    M.with_production_sources(agent)
    assert agent.autonomous.source_names() == ["proactive"]


# ── 3. the scheduled-tick bridge runs on real wall-clock time ────────────

def test_tick_ctx_carries_wall_clock(tmp_path):
    agent = _modes(tmp_path)
    agent.set_modes("telegram:1", {"autonomous"})
    seen: dict = {}
    agent.autonomous.add_source(
        "recorder", lambda ctx: (seen.update(ctx), None)[1])
    before = datetime.now()
    agent.tick("telegram:1", {})
    after = datetime.now()
    assert seen["hour"] in {before.hour, after.hour}
    assert seen["day_of_week"] in {before.strftime("%A").lower(),
                                   after.strftime("%A").lower()}
    assert seen["month"] in {before.month, after.month}


def test_scheduled_tick_fallback_ticks_with_sources(tmp_path, monkeypatch):
    # The tick tool's fallback (no runtime engine in extras) must tick
    # with the production sources, not a bare engine — drive it the way
    # the scheduler does, through register() + the captured tool fn.
    monkeypatch.setenv("NM_AGENT_MODES_STORE",
                       str(tmp_path / "agent-modes.json"))
    M.AgentModes().set_modes("telegram:1", {"autonomous"})
    monkeypatch.setattr("nomorals.agents.proactive.ProactiveEngine",
                        _StubEngine)
    from nomorals.tools import autonomy as auto

    db = _FakeDB()
    monkeypatch.setattr("nomorals.autonomy.idle.workspace_db",
                        lambda ws: db)
    captured: dict = {}
    ctx = SimpleNamespace(
        extras={},
        settings=SimpleNamespace(workspace_dir=str(tmp_path)))

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

    auto.register(_Registry())
    result = captured["__agent_modes_tick__"]()
    assert result["ok"] is True
    assert result["chats"] == ["telegram:1"]
    assert result["proposed"] == 1
    assert any("proactive_log" in sql for sql, _ in db.executed)


# ── 4. planning-mode ticks run through handle_task ───────────────────────

def test_planning_tick_routes_through_handle_task(tmp_path):
    agent = _modes(tmp_path)
    agent.set_modes("telegram:1", {"autonomous", "planning"})
    events: list[str] = []

    def _decompose(goal, evidence=None):
        return [{"name": "step-one", "description": "one", "action": "wait",
                 "parameters": {"seconds": 0}, "depends_on": []}]

    agent.autonomous.add_source(
        "worker",
        lambda ctx: {"label": "initiative work",
                     "goal": "do initiative work",
                     "decompose": _decompose,
                     "execute_step": lambda s, c: events.append(
                         f"exec:{s.name}"),
                     "on_plan": lambda p: events.append("plan")})
    calls: list[str] = []
    orig = agent.handle_task

    async def _spy(goal, chat_key, **kw):
        calls.append(goal)
        return await orig(goal, chat_key, **kw)

    agent.handle_task = _spy  # type: ignore[method-assign]
    done = agent.tick("telegram:1", {})
    assert calls == ["do initiative work"]
    assert len(done) == 1
    assert done[0].status == "done"
    assert events[0] == "plan"  # plan-first even for self-initiated work
    assert events[1] == "exec:step-one"
