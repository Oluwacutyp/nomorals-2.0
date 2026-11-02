"""Tail-97 pins: nomorals/agents/orchestrator.py census sites.

All four sites are held as exclusions. Two are prompt-enrichment
bonuses in ``plan()`` whose absence cannot change a plan's validity
(real degradations on that path ARE surfaced by name — the no-router
case sets ``plan_error`` explicitly); two are telemetry emits whose
contracts are stated in their own docstrings ("Never raises —
observability must not break the run") and whose durable records
live elsewhere: a reevaluation decision is appended to
``reevaluation_log`` (and summarized into run results) before the
bus emit is attempted, and ``_record_role`` updates the in-memory
``_role_stats`` entry BEFORE emitting, so a dead bus loses only the
live notification, never the record.

263 — repo-orientation block injection crash -> prompt simply
lacks the bonus block; the plan is still built and returned.
274 — lesson-memory injection crash -> prompt lacks the lessons
block ("lessons are a bonus, never fatal").
1038 — ``_emit_reevaluation`` context.emit crash -> swallowed;
the decision remains in reevaluation_log / run results.
1094 — ``_record_role`` context.emit crash -> swallowed; the
stats entry was already updated and is asserted verbatim.

NOT-recording is pinned with the snapshot method: orchestrator.py
calls no failures-ledger record function anywhere, so both the
agency and agents ledgers must be unchanged across each trigger.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
from nomorals.agents import orchestrator as OM


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def _raise(*args, **kwargs):
    raise RuntimeError("boom")


class _FakeBrain:
    def __init__(self):
        self.prompts: list[str] = []

    def chat(self, messages, params=None, **kwargs):
        self.prompts.append(messages[0].content)
        return SimpleNamespace(
            ok=True, model="fake",
            text=json.dumps({"steps": [
                {"name": "s1", "goal": "g", "role": "research"}]}))


def _orch(context=None):
    if context is None:
        context = SimpleNamespace(router=object(), executor=None,
                                  blackboard=None)
    return OM.MasterOrchestrator(context)


# ── 263: orientation crash -> plan built without the block ──────

def test_263_orientation_failure_plans_without_block():
    before = _snap()
    brain = _FakeBrain()
    with mock.patch.object(OM, "brain_for", lambda ctx: brain), \
            mock.patch("nomorals.agents.orientation.repo_orientation_block",
                       _raise):
        plan = _orch().plan("link Spotify to devon")
    assert [s.name for s in plan.steps] == ["s1"]
    assert "ORIENTATION_SENTINEL" not in brain.prompts[0]
    # control: when the builder works, the block reaches the prompt.
    brain2 = _FakeBrain()
    with mock.patch.object(OM, "brain_for", lambda ctx: brain2), \
            mock.patch("nomorals.agents.orientation.repo_orientation_block",
                       lambda: "ORIENTATION_SENTINEL"):
        plan2 = _orch().plan("link Spotify to devon")
    assert "ORIENTATION_SENTINEL" in brain2.prompts[0]
    assert [s.name for s in plan2.steps] == ["s1"]
    assert _snap() == before


# ── 274: lessons crash -> plan built without the block ──────────

def test_274_lessons_failure_plans_without_block():
    before = _snap()
    brain = _FakeBrain()
    with mock.patch.object(OM, "brain_for", lambda ctx: brain), \
            mock.patch("nomorals.agents.failure.enrich_with_lessons",
                       _raise):
        plan = _orch().plan("plan a quiet week")
    assert [s.name for s in plan.steps] == ["s1"]
    assert "LESSONS_SENTINEL" not in brain.prompts[0]
    # control: returned lessons reach the prompt verbatim.
    brain2 = _FakeBrain()
    with mock.patch.object(OM, "brain_for", lambda ctx: brain2), \
            mock.patch("nomorals.agents.failure.enrich_with_lessons",
                       lambda ctx, goal, limit=3: "LESSONS_SENTINEL"):
        plan2 = _orch().plan("plan a quiet week")
    assert "LESSONS_SENTINEL" in brain2.prompts[0]
    assert [s.name for s in plan2.steps] == ["s1"]
    assert _snap() == before


# ── 1038: reevaluation emit crash never breaks the caller ───────

class _BoomCtx:
    router = None
    executor = None
    blackboard = None
    db = None

    def emit(self, *args, **kwargs):
        raise RuntimeError("bus down")


class _OkCtx:
    router = None
    executor = None
    blackboard = None
    db = None

    def __init__(self):
        self.events: list = []

    def emit(self, event, **kwargs):
        self.events.append((event, kwargs))


def _no_telemetry(*args, **kwargs):
    return None


def test_1038_emit_failure_never_breaks_reevaluation_emit():
    before = _snap()
    ev = OM.Reevaluation(action="trim", reason="r",
                         affected=["s1"], at_task="t1")
    orch = _orch(_BoomCtx())
    orch._run_goal = "goal-x"
    with mock.patch("nomorals.storage.router_telemetry.record_reevaluation",
                    _no_telemetry):
        assert orch._emit_reevaluation(ev) is None
    # control: a healthy bus receives the event with its fields.
    ctx = _OkCtx()
    orch2 = _orch(ctx)
    orch2._run_goal = "goal-x"
    with mock.patch("nomorals.storage.router_telemetry.record_reevaluation",
                    _no_telemetry):
        orch2._emit_reevaluation(ev)
    assert ctx.events and ctx.events[0][0] == "plan.reevaluated"
    assert ctx.events[0][1]["action"] == "trim"
    assert ctx.events[0][1]["at_task"] == "t1"
    assert _snap() == before


# ── 1094: role emit crash -> stats entry already durable ────────

def test_1094_emit_failure_still_records_role_stats():
    before = _snap()
    orch = _orch(_BoomCtx())
    orch._record_role("coding", 1.25, ok=True, denials=2)
    assert orch._role_stats["coding"] == {
        "tasks": 1, "ok": 1, "failed": 0, "seconds": 1.25, "denials": 2}
    # control: a healthy bus receives swarm.task_done for the role.
    ctx = _OkCtx()
    orch2 = _orch(ctx)
    orch2._record_role("research", 0.5, ok=False, denials=0)
    assert ctx.events and ctx.events[0][0] == "swarm.task_done"
    assert ctx.events[0][1]["role"] == "research"
    assert ctx.events[0][1]["ok"] is False
    assert _snap() == before
