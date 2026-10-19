"""Pass #15: PlanningRunner replan must not reuse stale results on id collision.

When a failed step triggers a replan, the new plan's steps are keyed by
step_id and completed results are carried over. If decompose() regenerates
positional ids (the PlanningRunner default is ``step_{i+1}``), a replanned
step with the SAME id but DIFFERENT content was silently skipped as
"already done" — its stale result stood in for work that never ran.
The skip must apply only when the replanned step is the SAME work.
"""
import asyncio

from nomorals.agents.modes import PlanningRunner


def _run(coro):
    return asyncio.run(coro)


def test_replan_reruns_step_with_same_id_but_different_content():
    calls = []

    def decompose(goal, evidence):
        if evidence is None:
            return [
                {"step_id": "step_1", "name": "first",
                 "action": "a", "parameters": {"x": 1}},
                {"step_id": "step_2", "name": "second",
                 "action": "b", "parameters": {"x": 2},
                 "depends_on": ["step_1"]},
            ]
        # Replan: same positional ids, but step_1 is genuinely different work.
        return [
            {"step_id": "step_1", "name": "first-redone",
             "action": "a", "parameters": {"x": 99}},
            {"step_id": "step_2", "name": "second-redone",
             "action": "b", "parameters": {"x": 2},
             "depends_on": ["step_1"]},
        ]

    def execute_step(step, ctx):
        calls.append((step.step_id, step.name,
                      dict(step.parameters)))
        if step.name == "second":
            raise RuntimeError("boom")
        return f"result-{step.name}"

    runner = PlanningRunner()
    # PlanningRunner reuses AgentPlanner.validate_plan: register the
    # test actions so the plans are valid. The runner itself never
    # calls these handlers (execute_step is injected).
    runner.planner._action_handlers.update(
        {"a": lambda **kw: None, "b": lambda **kw: None})
    result = _run(runner.run("goal", decompose=decompose,
                             execute_step=execute_step,
                             verify_step=None, max_replans=2))

    assert result.success, result.errors
    step_1_calls = [c for c in calls if c[0] == "step_1"]
    # The replanned step_1 is different work (x=99) — it must have run.
    assert len(step_1_calls) == 2, (
        f"step_1 ran {len(step_1_calls)}x; the replanned step_1 "
        f"(different parameters) was skipped on a stale result")
    assert step_1_calls[1][2] == {"x": 99}
    assert result.results["step_1"] == "result-first-redone"


def test_replan_skips_identical_completed_step():
    """The skip optimization still applies when the work is identical."""
    calls = []

    def decompose(goal, evidence):
        if evidence is None:
            return [
                {"step_id": "step_1", "name": "first",
                 "action": "a", "parameters": {"x": 1}},
                {"step_id": "step_2", "name": "second",
                 "action": "b", "parameters": {"x": 2},
                 "depends_on": ["step_1"]},
            ]
        # Replan re-emits step_1 identically — must NOT re-run.
        return [
            {"step_id": "step_1", "name": "first",
             "action": "a", "parameters": {"x": 1}},
            {"step_id": "step_2", "name": "second-redone",
             "action": "b", "parameters": {"x": 3},
             "depends_on": ["step_1"]},
        ]

    def execute_step(step, ctx):
        calls.append(step.step_id)
        if step.name == "second":
            raise RuntimeError("boom")
        return f"result-{step.name}"

    runner = PlanningRunner()
    runner.planner._action_handlers.update(
        {"a": lambda **kw: None, "b": lambda **kw: None})
    result = _run(runner.run("goal", decompose=decompose,
                             execute_step=execute_step,
                             verify_step=None, max_replans=2))

    assert result.success, result.errors
    assert calls.count("step_1") == 1, (
        f"identical step_1 re-ran {calls.count('step_1')}x")
    assert result.results["step_1"] == "result-first"


def test_replan_cap_still_holds():
    """max_replans bounds the replan loop even with repeated failures."""
    plans = []

    def decompose(goal, evidence):
        plans.append(evidence)
        return [{"step_id": "s1", "name": "only",
                 "action": "a", "parameters": {}}]

    def execute_step(step, ctx):
        raise RuntimeError("always fails")

    runner = PlanningRunner()
    runner.planner._action_handlers.update(
        {"a": lambda **kw: None})
    result = _run(runner.run("goal", decompose=decompose,
                             execute_step=execute_step,
                             verify_step=None, max_replans=2))

    assert not result.success
    # 1 initial plan + 2 replans = 3 decompose calls
    assert len(plans) == 3, f"decompose called {len(plans)}x"
    # failure evidence reached the replan
    assert plans[1]["failed_step"] == "only"
    assert "always fails" in plans[1]["error"]
