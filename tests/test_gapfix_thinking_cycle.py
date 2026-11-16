"""Gapfix lane 11 (audit gap R1-10): the cognition thinking cycle.

The thinking cycle (plan -> act -> reflect -> remember) exists as
``nomorals.cognition.thinking.ThinkingStore`` but nothing in production
ever called it: autonomous goal advances produced no structured
reflections and no trajectory rows.  This file pins the wiring from the
live heartbeat — ``CognitiveLoop._tick_goals`` — so every goal step the
tick actually executes becomes one thinking cycle: the changed steps
are the acts, the reflection is returned in the tick summary, and every
act lands as a row in the shared ``TrajectoryStore`` (same home DB the
LLM-router learning hook writes).

FAILING-FIRST: before the fix ``_tick_goals`` returns no ``thinking``
entry and writes no trajectory rows, so every test here fails.
"""

from __future__ import annotations

from unittest import mock

import pytest

from nomorals.agents.cognition import CognitiveLoop
from nomorals.agents.context import build_context
from nomorals.agents.goals import GoalSystem
from nomorals.cognition.trajectories import TrajectoryStore
from nomorals.storage.db import Database

TASK_KIND = "goal-step"
CAPABILITY = "cognitive-loop"


def _ctx():
    return build_context(db=Database(":memory:"), with_tools=False,
                         with_router=False, with_memory=False)


def _goal(ctx, title="Ship the report", steps=("Draft it", "Send it")):
    return GoalSystem(ctx).create(title, plan=list(steps))


def _traj_rows(ctx, task_kind=TASK_KIND):
    # The table only exists once a TrajectoryStore has been constructed
    # (its DDL runs in __init__); a tick with no acts never touches the
    # store, so "no table yet" honestly means "no rows".
    exists = ctx.db.query(
        "SELECT name FROM sqlite_master WHERE type='table'"
        " AND name='cog_trajectories'")
    if not exists:
        return []
    return ctx.db.query(
        "SELECT task_kind, capability, model_id, success, error"
        " FROM cog_trajectories WHERE task_kind = ?",
        (task_kind,))


def _failing_executor(description):
    raise RuntimeError("executor blew up: gateway down")


class TestThinkingCycle:
    def test_tick_goals_produces_row_and_reflection(self):
        ctx = _ctx()
        goal = _goal(ctx)
        loop = CognitiveLoop(ctx)
        result = loop._tick_goals()
        thinking = result.get("thinking") or {}
        assert thinking.get("cycles") == 1, thinking
        entry = (thinking.get("goals") or [{}])[0]
        assert entry.get("goal_id") == goal.id
        assert entry.get("trajectories_written") == 1
        assert entry.get("success_rate") == pytest.approx(1.0)
        assert entry.get("worked") == ["Draft it"]
        assert entry.get("failed") == []
        assert entry.get("lessons"), "a successful cycle states its lesson"
        # The act landed in the SHARED trajectory store (same home DB
        # the LLM-router learning hook writes), in the shape the
        # calibration loop reads.
        rows = _traj_rows(ctx)
        assert len(rows) == 1
        assert rows[0]["capability"] == CAPABILITY
        assert rows[0]["success"] == 1
        assert rows[0]["model_id"] == ""
        # ... and the broker can already score the slice from it.
        store = TrajectoryStore(ctx.db)
        assert store.success_rate(TASK_KIND, CAPABILITY) == pytest.approx(1.0)

    def test_no_phantom_reflection_without_acts(self):
        ctx = _ctx()
        goal = _goal(ctx)
        # Every step already done: the tick advances nothing for this goal.
        ctx.db.execute(
            "UPDATE agent_goal_steps SET status='done',"
            " result='done earlier' WHERE goal_id=?", (goal.id,))
        loop = CognitiveLoop(ctx)
        before = len(_traj_rows(ctx))
        result = loop._tick_goals()
        thinking = result.get("thinking") or {}
        assert thinking.get("cycles") == 0, thinking
        assert thinking.get("goals") == []
        assert len(_traj_rows(ctx)) == before

    def test_second_tick_does_not_re_record_completed_step(self):
        ctx = _ctx()
        _goal(ctx)
        loop = CognitiveLoop(ctx)
        first = loop._tick_goals()
        assert (first.get("thinking") or {}).get("cycles") == 1
        assert len(_traj_rows(ctx)) == 1
        # Tick two advances the SECOND step only.  The first step's act
        # must not be written again: the diff is by state change, so a
        # step that already finished contributes nothing.
        second = loop._tick_goals()
        thinking = second.get("thinking") or {}
        assert thinking.get("cycles") == 1, thinking
        entry = (thinking.get("goals") or [{}])[0]
        assert entry.get("worked") == ["Send it"]
        assert entry.get("trajectories_written") == 1
        assert len(_traj_rows(ctx)) == 2

    def test_blocked_step_records_failure_row_and_lesson(self):
        ctx = _ctx()
        _goal(ctx)
        loop = CognitiveLoop(ctx)
        result = loop._tick_goals(executor=_failing_executor)
        thinking = result.get("thinking") or {}
        assert thinking.get("cycles") == 1, thinking
        entry = (thinking.get("goals") or [{}])[0]
        assert entry.get("success_rate") == pytest.approx(0.0)
        assert entry.get("failed") == ["Draft it"]
        assert entry.get("lessons"), "a failed step must teach something"
        assert any("Draft it" in lesson for lesson in entry["lessons"])
        rows = _traj_rows(ctx)
        assert len(rows) == 1
        assert rows[0]["success"] == 0
        assert "gateway down" in (rows[0]["error"] or "")

    def test_reflection_failure_degrades_tick(self):
        ctx = _ctx()
        _goal(ctx)
        loop = CognitiveLoop(ctx)
        with mock.patch(
                "nomorals.cognition.thinking.ThinkingStore.remember",
                side_effect=RuntimeError("trajectory disk wedged")):
            result = loop._tick_goals()
        # The goal still advanced (the tick did its real work) and the
        # failure is recorded on the thinking entry instead of killing
        # the tick.
        assert result.get("advanced"), (
            f"goal must still advance: {result!r}")
        thinking = result.get("thinking") or {}
        assert thinking.get("cycles") == 0, thinking
        entry = (thinking.get("goals") or [{}])[0]
        assert "trajectory disk wedged" in (entry.get("error") or ""), entry
        assert _traj_rows(ctx) == []
