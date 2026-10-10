"""Phase 9 Slice C — Planning & Goals & Morning Pulse.

Real tests for breakable behavior: PERT math, topological scheduling,
window feasibility, congestion routing, goal deadlines/estimates/graph
projection, plan validation + parallel batches + retry, pulse catch-up,
briefing goal/disruption providers.
"""
from __future__ import annotations

import asyncio
import math
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch


# ── fixtures ─────────────────────────────────────────────────────────────

def _goal_ctx():
    from nomorals.storage.db import Database
    db = Database(":memory:")
    db.migrate()
    return SimpleNamespace(
        db=db,
        settings=SimpleNamespace(rooms_auto_create=False, autonomy=None,
                                 workspace_dir=tempfile.mkdtemp()),
        router=None,
    )


# ═══════════════════════════════════════════════════════════════════════
# planning.estimates — PERT + aggregation
# ═══════════════════════════════════════════════════════════════════════

class PertTests(unittest.TestCase):
    def test_three_point_math(self):
        from nomorals.planning import pert
        te, sigma = pert(10, 20, 60)
        self.assertAlmostEqual(te, 25.0)
        self.assertAlmostEqual(sigma, 8.33, places=2)

    def test_swaps_when_pessimistic_lt_optimistic(self):
        from nomorals.planning import pert
        te, sigma = pert(60, 20, 10)
        self.assertAlmostEqual(te, 25.0)
        self.assertAlmostEqual(sigma, 8.33, places=2)

    def test_bad_input_never_raises(self):
        from nomorals.planning import pert
        self.assertEqual(pert("x", None, []), (0.0, 0.0))

    def test_pert_estimate_band(self):
        from nomorals.planning import pert_estimate
        est = pert_estimate("deploy", 10, 20, 60)
        self.assertEqual(est.source, "pert")
        self.assertAlmostEqual(est.point, 25.0)
        self.assertAlmostEqual(est.low, 25.0 - 2 * 8.33, places=1)
        self.assertAlmostEqual(est.high, 25.0 + 2 * 8.33, places=1)
        # Meituan: quoted deadline is the conservative end
        self.assertEqual(est.deadline, est.high)

    def test_aggregate_sums_expectations_quadrature_variance(self):
        from nomorals.planning import aggregate, Estimate
        a = Estimate("a", 10.0, 8.0, 12.0, 0.5, [], 12.0, "heuristic", 0,
                     std_minutes=2.0)
        b = Estimate("b", 20.0, 17.0, 23.0, 0.5, [], 23.0, "heuristic", 0,
                     std_minutes=3.0)
        total = aggregate("proj", [a, b])
        self.assertAlmostEqual(total.point, 30.0)
        self.assertAlmostEqual(total.std_minutes, math.sqrt(4 + 9), places=2)
        self.assertLess(total.low, total.point)
        self.assertGreater(total.high, total.point)

    def test_aggregate_empty_is_honest(self):
        from nomorals.planning import aggregate
        total = aggregate("proj", [])
        self.assertGreater(total.high, total.point)

    def test_store_estimate_carries_std(self):
        from nomorals.planning.estimates import EstimateStore
        store = EstimateStore(":memory:")
        est = store.estimate("task", {"prep": 10, "work": 20})
        self.assertGreaterEqual(est.std_minutes, 0.0)
        self.assertGreater(est.high - est.low, 0)


# ═══════════════════════════════════════════════════════════════════════
# planning.graph — scheduling order, PERT critical path, goal projection
# ═══════════════════════════════════════════════════════════════════════

class GraphSchedulingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _graph(self):
        from nomorals.planning.graph import WorldGraph
        return WorldGraph(
            db_path=os.path.join(self._tmp.name,
                                 f"wg-{time.time_ns()}.db"))

    def test_schedule_order_respects_depends_on(self):
        g = self._graph()
        b = g.add_node("commitment", "pack bags")
        a = g.add_node("commitment", "catch flight")
        g.add_edge(a.node_id, b.node_id, "depends_on")
        order = [n.label for n in g.schedule_order()]
        self.assertLess(order.index("pack bags"), order.index("catch flight"))

    def test_schedule_order_due_date_breaks_ties(self):
        g = self._graph()
        now = time.time()
        late = g.add_node("commitment", "later job",
                          {"due_ts": now + 7200})
        soon = g.add_node("commitment", "urgent job",
                          {"due_ts": now + 600})
        order = [n.label for n in g.schedule_order()]
        self.assertLess(order.index("urgent job"), order.index("later job"))

    def test_schedule_order_cycle_safe(self):
        g = self._graph()
        a = g.add_node("commitment", "A")
        b = g.add_node("commitment", "B")
        g.add_edge(a.node_id, b.node_id, "depends_on")
        g.add_edge(b.node_id, a.node_id, "depends_on")
        order = g.schedule_order()
        self.assertEqual(len(order), 2)  # no hang, nothing dropped

    def test_critical_path_pert_sums_te(self):
        g = self._graph()
        z = g.add_node("commitment", "finish",
                       {"optimistic": 5, "most_likely": 10, "pessimistic": 15})
        y = g.add_node("commitment", "middle",
                       {"optimistic": 10, "most_likely": 20, "pessimistic": 30})
        x = g.add_node("commitment", "start",
                       {"optimistic": 5, "most_likely": 5, "pessimistic": 5})
        g.add_edge(x.node_id, y.node_id, "depends_on")
        g.add_edge(y.node_id, z.node_id, "depends_on")
        r = g.critical_path_pert()
        self.assertTrue(r["ok"])
        # TE: 10 + 20 + 5 = 35
        self.assertAlmostEqual(r["expected_minutes"], 35.0, places=1)
        # variance: (10/6)^2 + (20/6)^2 + 0
        self.assertAlmostEqual(r["std_minutes"],
                               math.sqrt((10 / 6) ** 2 + (20 / 6) ** 2),
                               places=1)
        self.assertEqual(len(r["path"]), 3)

    def test_project_goal_idempotent(self):
        g = self._graph()
        goal = SimpleNamespace(
            id="goal_abc", title="Ship it", status="active", progress=0.25,
            deadline=0.0,
            steps=[SimpleNamespace(id="s1", description="write code",
                                   status="done"),
                   SimpleNamespace(id="s2", description="write tests",
                                   status="pending")])
        n1 = g.project_goal(goal)
        self.assertEqual(n1, 3)  # goal + 2 steps
        before = len(g.list_nodes())
        n2 = g.project_goal(goal)
        self.assertEqual(len(g.list_nodes()), before)  # idempotent
        # sequential depends_on chain exists
        step_nodes = [n for n in g.list_nodes()
                      if n.attrs.get("source") == "goals"
                      and n.type == "commitment"]
        self.assertEqual(len(step_nodes), 2)
        second = next(n for n in step_nodes if "tests" in n.label)
        deps = g.dependencies(second.node_id)
        self.assertTrue(any("write code" in d.label for d in deps))


# ═══════════════════════════════════════════════════════════════════════
# planning.route — window feasibility
# ═══════════════════════════════════════════════════════════════════════

class RouteWindowTests(unittest.TestCase):
    def test_missed_window_is_reported(self):
        from nomorals.planning.route import RouteSolver, CostModel, Stop
        cost = CostModel(db_path=":memory:")
        now = time.time()
        home = Stop("h", "home", lat=6.52, lng=3.37)
        bank = Stop("b", "bank", lat=6.53, lng=3.38,
                    window=(now - 7200, now - 3600))  # already closed
        res = RouteSolver("greedy").solve([home, bank], cost,
                                          start_time=now)
        self.assertTrue(any("bank" in v for v in res.window_violations))

    def test_open_window_has_no_violation(self):
        from nomorals.planning.route import RouteSolver, CostModel, Stop
        cost = CostModel(db_path=":memory:")
        now = time.time()
        home = Stop("h", "home", lat=6.52, lng=3.37)
        bank = Stop("b", "bank", lat=6.53, lng=3.38,
                    window=(now - 3600, now + 7200))
        res = RouteSolver("greedy").solve([home, bank], cost,
                                          start_time=now)
        self.assertEqual(res.window_violations, [])

    def test_format_route_surfaces_violations(self):
        from nomorals.planning.route import (
            RouteSolver, CostModel, Stop, format_route)
        cost = CostModel(db_path=":memory:")
        now = time.time()
        home = Stop("h", "home", lat=6.52, lng=3.37)
        bank = Stop("b", "bank", lat=6.53, lng=3.38,
                    window=(now - 7200, now - 3600))
        res = RouteSolver("greedy").solve([home, bank], cost,
                                          start_time=now)
        text = format_route(res)
        self.assertIn("window misses", text)


# ═══════════════════════════════════════════════════════════════════════
# planning.congestion — hold() + route_around()
# ═══════════════════════════════════════════════════════════════════════

class CongestionTests(unittest.TestCase):
    def test_hold_releases_on_exit(self):
        from nomorals.planning.congestion import (
            ContentionMonitor, hold)
        mon = ContentionMonitor(in_memory=True)
        mon.register("api", kind="api_key", capacity=2)
        with hold(mon, "api", "agent-1") as h:
            self.assertIsNotNone(h)
            self.assertEqual(mon.queue_depth("api"), 1)
        self.assertEqual(mon.queue_depth("api"), 0)

    def test_hold_releases_on_exception(self):
        from nomorals.planning.congestion import (
            ContentionMonitor, hold)
        mon = ContentionMonitor(in_memory=True)
        mon.register("api", kind="api_key", capacity=2)
        with self.assertRaises(RuntimeError):
            with hold(mon, "api", "agent-1"):
                raise RuntimeError("boom")
        self.assertEqual(mon.queue_depth("api"), 0)

    def test_hold_none_monitor_is_fail_open(self):
        from nomorals.planning.congestion import hold
        with hold(None, "api", "agent-1") as h:
            self.assertIsNone(h)

    def test_route_around_switches_to_alternate(self):
        from nomorals.planning.congestion import (
            ContentionMonitor, route_around)
        mon = ContentionMonitor(in_memory=True)
        mon.register("primary", kind="api_key", capacity=1)
        mon.register("backup", kind="api_key", capacity=4)
        mon.register_alternate("primary", "backup")
        mon.acquire("primary", "agent-9")  # saturate: 1/1
        routed = route_around(mon, ["primary"], agents_waiting=3)
        self.assertEqual(routed["primary"]["target"], "backup")
        self.assertEqual(routed["primary"]["action"], "switch")

    def test_route_around_clear_resource_stays(self):
        from nomorals.planning.congestion import (
            ContentionMonitor, route_around)
        mon = ContentionMonitor(in_memory=True)
        mon.register("calm", kind="endpoint", capacity=8)
        routed = route_around(mon, ["calm"])
        self.assertEqual(routed["calm"]["target"], "calm")
        self.assertEqual(routed["calm"]["action"], "proceed")

    def test_route_around_none_monitor(self):
        from nomorals.planning.congestion import route_around
        self.assertEqual(route_around(None, ["x"]), {})


# ═══════════════════════════════════════════════════════════════════════
# agents.goals — deadlines, estimates, graph projection
# ═══════════════════════════════════════════════════════════════════════

class GoalDeadlineTests(unittest.TestCase):
    def _gs(self):
        from nomorals.agents.goals import GoalSystem
        return GoalSystem(_goal_ctx())

    def test_set_deadline_relative(self):
        gs = self._gs()
        g = gs.create("D", plan=["a"])
        before = time.time()
        g = gs.set_deadline(g.id, "in 2d")
        assert g is not None
        self.assertAlmostEqual(g.deadline, before + 172800, delta=60)

    def test_set_deadline_iso(self):
        gs = self._gs()
        g = gs.create("D", plan=["a"])
        g = gs.set_deadline(g.id, "2026-12-01")
        assert g is not None
        self.assertGreater(g.deadline, time.time())

    def test_set_deadline_bad_raises(self):
        gs = self._gs()
        g = gs.create("D", plan=["a"])
        with self.assertRaises(ValueError):
            gs.set_deadline(g.id, "someday-ish")

    def test_due_soon_and_overdue(self):
        gs = self._gs()
        soon = gs.create("Soon", plan=["a"])
        late = gs.create("Late", plan=["a"])
        gs.set_deadline(soon.id, "in 1h")
        gs.set_deadline(late.id, time.time() - 60)
        self.assertTrue(any(g.id == soon.id
                            for g in gs.due_soon(within_hours=24)))
        self.assertTrue(any(g.id == late.id for g in gs.overdue()))
        self.assertFalse(any(g.id == late.id
                             for g in gs.due_soon(within_hours=24)))

    def test_deadline_text(self):
        gs = self._gs()
        g = gs.create("D", plan=["a"])
        self.assertEqual(gs.deadline_text(g), "no deadline")
        g = gs.set_deadline(g.id, "in 3h")
        assert g is not None
        self.assertIn("due in", gs.deadline_text(g))
        g = gs.set_deadline(g.id, time.time() - 3600)
        assert g is not None
        self.assertIn("overdue by", gs.deadline_text(g))

    def test_clear_deadline(self):
        gs = self._gs()
        g = gs.create("D", plan=["a"])
        gs.set_deadline(g.id, "in 1d")
        g = gs.clear_deadline(g.id)
        assert g is not None
        self.assertEqual(g.deadline, 0.0)

    def test_ensure_schema_idempotent(self):
        # initializing GoalSystem twice on the same db must not blow up
        from nomorals.storage.db import Database
        from nomorals.agents.goals import GoalSystem
        db = Database(":memory:")
        db.migrate()
        ctx = SimpleNamespace(
            db=db,
            settings=SimpleNamespace(rooms_auto_create=False, autonomy=None),
            router=None)
        GoalSystem(ctx)
        GoalSystem(ctx)
        cols = {r["name"] for r in db.query("PRAGMA table_info(agent_goals)")}
        self.assertIn("deadline", cols)
        scols = {r["name"] for r in
                 db.query("PRAGMA table_info(agent_goal_steps)")}
        self.assertIn("est_minutes", scols)

    def test_estimate_goal(self):
        gs = self._gs()
        g = gs.create("Build thing", plan=["design", "build", "test"])
        for s in g.steps:
            gs.set_step_estimate(g.id, s.id, 30.0)
        tmp = tempfile.mktemp(suffix=".db")
        try:
            res = gs.estimate_goal(g.id, db_path=tmp)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        self.assertTrue(res["ok"])
        self.assertEqual(res["steps"], 3)
        # 3 × 30min ≈ 90min point (banded, so allow tolerance)
        self.assertAlmostEqual(res["point_minutes"], 90.0, delta=5.0)
        self.assertIn("⏱️", res["text"])

    def test_estimate_goal_no_work_left(self):
        gs = self._gs()
        g = gs.create("Done", plan=["a"])
        gs.complete(g.id)
        tmp = tempfile.mktemp(suffix=".db")
        try:
            res = gs.estimate_goal(g.id, db_path=tmp)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        self.assertFalse(res["ok"])

    def test_record_step_outcome(self):
        gs = self._gs()
        g = gs.create("T", plan=["a"])
        tmp = tempfile.mktemp(suffix=".db")
        try:
            ok = gs.record_step_outcome(g.id, g.steps[0].id, 42.0,
                                        db_path=tmp)
            from nomorals.planning.estimates import EstimateStore
            misses = EstimateStore(tmp).miss_count("goal_step")
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        self.assertTrue(ok)
        self.assertGreaterEqual(misses, 0)

    def test_project_to_graph(self):
        from nomorals.planning.graph import WorldGraph
        gs = self._gs()
        g = gs.create("Deploy", plan=["build", "push"])
        with tempfile.TemporaryDirectory() as td:
            graph = WorldGraph(db_path=os.path.join(td, "wg.db"))
            res = gs.project_to_graph(g.id, graph=graph)
            self.assertTrue(res["ok"])
            self.assertEqual(res["nodes"], 3)
            projects = graph.list_nodes("project")
            self.assertTrue(any("Deploy" in n.label for n in projects))

    def test_tool_deadline_and_due_actions(self):
        from nomorals.agents.goals import register
        calls = {}

        class FakeRegistry:
            def __init__(self):
                self.context = _goal_ctx()
            def register(self, name, **kw):
                def deco(fn):
                    calls[name] = fn
                    return fn
                return deco

        reg = FakeRegistry()
        register(reg)
        tool = calls["goal"]
        created = tool("create", title="Tool goal", description="")
        gid = created["goal"]["id"]
        dl = tool("deadline", goal_id=gid, deadline="in 5h")
        self.assertTrue(dl["ok"])
        self.assertGreater(dl["goal"]["deadline"], time.time())
        due = tool("due", within_hours="24")
        self.assertTrue(any(g["id"] == gid for g in due["due_soon"]))
        est = tool("estimate", goal_id=gid)
        self.assertTrue(est["ok"])


# ═══════════════════════════════════════════════════════════════════════
# agents.planner — validation, batches, retry, parallel
# ═══════════════════════════════════════════════════════════════════════

class PlannerValidationTests(unittest.TestCase):
    def _planner(self):
        from nomorals.agents.planner import AgentPlanner
        return AgentPlanner()

    def _plan(self, steps):
        from nomorals.agents.planner import Plan, PlanStep
        return Plan(plan_id="p1", goal="g",
                    steps=[PlanStep(**s) for s in steps])

    def test_dangling_dependency(self):
        p = self._planner()
        plan = self._plan([{"step_id": "s1", "name": "a", "description": "d",
                            "action": "wait", "depends_on": ["nope"]}])
        problems = p.validate_plan(plan)
        self.assertTrue(any("unknown step id" in x for x in problems))

    def test_cycle_detected(self):
        p = self._planner()
        plan = self._plan([
            {"step_id": "s1", "name": "a", "description": "d",
             "action": "wait", "depends_on": ["s2"]},
            {"step_id": "s2", "name": "b", "description": "d",
             "action": "wait", "depends_on": ["s1"]},
        ])
        problems = p.validate_plan(plan)
        self.assertTrue(any("cycle" in x for x in problems))

    def test_unknown_action(self):
        p = self._planner()
        plan = self._plan([{"step_id": "s1", "name": "a", "description": "d",
                            "action": "teleport"}])
        problems = p.validate_plan(plan)
        self.assertTrue(any("unknown action" in x for x in problems))

    def test_valid_plan_passes(self):
        p = self._planner()
        plan = self._plan([
            {"step_id": "s1", "name": "a", "description": "d",
             "action": "wait"},
            {"step_id": "s2", "name": "b", "description": "d",
             "action": "wait", "depends_on": ["s1"]},
        ])
        self.assertEqual(p.validate_plan(plan), [])

    def test_batches_topological(self):
        p = self._planner()
        plan = self._plan([
            {"step_id": "s1", "name": "a", "description": "d",
             "action": "wait"},
            {"step_id": "s2", "name": "b", "description": "d",
             "action": "wait"},
            {"step_id": "s3", "name": "c", "description": "d",
             "action": "wait", "depends_on": ["s1", "s2"]},
        ])
        batches = p.execution_batches(plan)
        self.assertEqual(len(batches), 2)
        self.assertEqual({s.step_id for s in batches[0]}, {"s1", "s2"})
        self.assertEqual([s.step_id for s in batches[1]], ["s3"])

    def test_invalid_plan_not_executed(self):
        async def _run():
            p = self._planner()
            plan = self._plan([{"step_id": "s1", "name": "a",
                                "description": "d", "action": "wait",
                                "depends_on": ["ghost"]}])
            # bypass LLM generation: execute() generates its own plan,
            # so call the internals directly via a stubbed _generate_plan
            async def fake_gen(goal, account):
                return plan
            p._generate_plan = fake_gen
            calls = []
            async def fake_step(step, account):
                calls.append(step.step_id)
                return {"ok": True}
            p._execute_step = fake_step
            result = await p.execute(goal="g", account="a")
            return result, calls
        result, calls = asyncio.run(_run())
        self.assertFalse(result.success)
        self.assertEqual(calls, [])  # nothing executed
        self.assertTrue(any("invalid plan" in e for e in result.errors))

    def test_independent_steps_run_in_parallel(self):
        async def _run():
            from nomorals.agents.planner import AgentPlanner, Plan, PlanStep
            p = AgentPlanner()
            in_flight = 0
            peak = 0

            async def slow(step, account):
                nonlocal in_flight, peak
                in_flight += 1
                peak = max(peak, in_flight)
                await asyncio.sleep(0.05)
                in_flight -= 1
                return {"step": step.step_id}

            async def fake_gen(goal, account):
                return Plan(plan_id="p", goal=goal, steps=[
                    PlanStep("s1", "a", "d", "wait"),
                    PlanStep("s2", "b", "d", "wait"),
                ])
            p._generate_plan = fake_gen
            p._execute_step = slow
            result = await p.execute(goal="g", account="a")
            return result, peak
        result, peak = asyncio.run(_run())
        self.assertTrue(result.success)
        self.assertEqual(peak, 2)  # actually concurrent

    def test_non_retryable_failure_skips_dependents(self):
        async def _run():
            from nomorals.agents.planner import AgentPlanner, Plan, PlanStep
            p = AgentPlanner()

            async def fake_gen(goal, account):
                return Plan(plan_id="p", goal=goal, steps=[
                    PlanStep("s1", "ok step", "d", "wait"),
                    PlanStep("s2", "bad step", "d", "wait"),
                    PlanStep("s3", "downstream", "d", "wait",
                             depends_on=["s2"]),
                ])

            async def fake_step(step, account):
                if step.step_id == "s2":
                    raise ValueError("hard failure")
                return {"ok": True}

            p._generate_plan = fake_gen
            p._execute_step = fake_step
            return await p.execute(goal="g", account="a", max_retries=0)
        result = asyncio.run(_run())
        self.assertFalse(result.success)
        by_id = {s: s for s in []}  # results keyed by step_id below
        self.assertIn("s1", result.results)
        self.assertNotIn("s2", result.results)
        self.assertNotIn("s3", result.results)  # skipped: dep failed

    def test_retryable_step_retries_then_succeeds(self):
        async def _run():
            from nomorals.agents.planner import AgentPlanner, Plan, PlanStep
            p = AgentPlanner()
            attempts = []

            async def fake_gen(goal, account):
                return Plan(plan_id="p", goal=goal, steps=[
                    PlanStep("s1", "flaky", "d", "wait"),
                ])

            class Flaky(Exception):
                pass

            async def fake_step(step, account):
                attempts.append(1)
                if len(attempts) < 2:
                    raise ConnectionError("transient")
                return {"ok": True}

            # make ConnectionError look retryable to ErrorIntelligence
            orig = p.error_intel.analyze
            def patched(exc, context=None, **kw):
                a = orig(exc, context=context, **kw)
                if isinstance(exc, ConnectionError):
                    a.retryable = True
                    a.explanation = "transient network blip"
                return a
            p.error_intel.analyze = patched
            p._generate_plan = fake_gen
            p._execute_step = fake_step
            result = await p.execute(goal="g", account="a", max_retries=2)
            return result, len(attempts)
        result, n = asyncio.run(_run())
        self.assertTrue(result.success)
        self.assertEqual(n, 2)


# ═══════════════════════════════════════════════════════════════════════
# agents.morning_pulse — catch-up + marker
# ═══════════════════════════════════════════════════════════════════════

class PulseCatchupTests(unittest.TestCase):
    def _ctx(self, **kw):
        ctx = SimpleNamespace(
            settings=SimpleNamespace(
                workspace_dir=tempfile.mkdtemp(),
                pulse=SimpleNamespace(enabled=True, time="23:00",
                                      timezone="America/Denver")),
            router=None)
        for k, v in kw.items():
            setattr(ctx, k, v)
        return ctx

    def test_disabled_pulse_no_catchup(self):
        from nomorals.agents.morning_pulse import check_pulse_catchup
        ctx = self._ctx()
        ctx.settings.pulse.enabled = False
        res = check_pulse_catchup(ctx)
        self.assertFalse(res["catchup"])
        self.assertEqual(res["reason"], "pulse disabled")

    def test_marker_roundtrip(self):
        from nomorals.agents.morning_pulse import (
            _write_marker, last_pulse_run)
        ctx = self._ctx()
        _write_marker(ctx, {"stages": ["news", "compose"],
                            "delivered_text": True,
                            "delivered_audio": False, "elapsed_s": 12.5})
        m = last_pulse_run(ctx)
        self.assertTrue(m["delivered_text"])
        self.assertEqual(m["stages"], ["news", "compose"])
        self.assertGreater(m["ts"], 0)

    def test_catchup_fires_when_nothing_delivered(self):
        import nomorals.agents.morning_pulse as mp
        ctx = self._ctx()
        ran = []

        def fake_pulse(c):
            ran.append(1)
            return {"ok": True, "delivered_text": True}

        with patch.object(mp, "_pulse_due_today",
                          return_value=time.time() - 3600), \
             patch.object(mp, "run_pulse", side_effect=fake_pulse):
            res = mp.check_pulse_catchup(ctx)
        self.assertTrue(res["catchup"])
        self.assertEqual(len(ran), 1)

    def test_catchup_skips_when_already_delivered(self):
        import nomorals.agents.morning_pulse as mp
        from nomorals.agents.morning_pulse import _write_marker
        ctx = self._ctx()
        _write_marker(ctx, {"stages": ["news", "compose", "deliver"],
                            "delivered_text": True,
                            "delivered_audio": True, "elapsed_s": 5.0})
        with patch.object(mp, "_pulse_due_today",
                          return_value=time.time() - 3600), \
             patch.object(mp, "run_pulse",
                          side_effect=AssertionError("must not run")):
            res = mp.check_pulse_catchup(ctx)
        self.assertFalse(res["catchup"])
        self.assertEqual(res["reason"], "already delivered")

    def test_pulse_tool_status_shows_last_run(self):
        import nomorals.agents.morning_pulse as mp
        from nomorals.agents.morning_pulse import _write_marker
        ctx = self._ctx()

        captured = {}
        class FakeRegistry:
            def __init__(self):
                self.context = ctx
            def register(self, name, **kw):
                def deco(fn):
                    captured[name] = fn
                    return fn
                return deco

        mp.register(FakeRegistry())
        _write_marker(ctx, {"stages": ["deliver"], "delivered_text": True,
                            "delivered_audio": False, "elapsed_s": 3.0})
        with patch("nomorals.agents.scheduler.Scheduler") as sched_cls:
            sched_cls.return_value.list_jobs.return_value = []
            res = captured["pulse"](action="status")
        self.assertTrue(res["last_run"]["delivered_text"])
        self.assertEqual(res["time"], "23:00")


# ═══════════════════════════════════════════════════════════════════════
# agents.morning_briefing — goals + disruption providers
# ═══════════════════════════════════════════════════════════════════════

class BriefingProvidersTests(unittest.TestCase):
    def test_goals_provider_surfaces_due_goal(self):
        from nomorals.agents.morning_briefing import GoalsProvider
        from nomorals.agents.goals import GoalSystem
        ctx = _goal_ctx()
        gs = GoalSystem(ctx)
        g = gs.create("Finish report", plan=["draft", "send"])
        gs.set_deadline(g.id, "in 6h")
        sec = GoalsProvider().collect(ctx, 0.0)
        self.assertIsNotNone(sec)
        assert sec is not None
        self.assertTrue(any("Finish report" in ln for ln in sec.lines))

    def test_goals_provider_quiet_when_no_goals(self):
        from nomorals.agents.morning_briefing import GoalsProvider
        sec = GoalsProvider().collect(_goal_ctx(), 0.0)
        self.assertIsNone(sec)

    def test_disruption_provider_surfaces_disrupted_node(self):
        from nomorals.agents.morning_briefing import DisruptionProvider
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "wg.db")
            with patch("nomorals.planning.graph._default_db_path",
                       return_value=db_path):
                from nomorals.planning.graph import WorldGraph
                g = WorldGraph()
                node = g.add_node("schedule", "Lagos flight LOS123")
                assert node is not None
                g.mark_disrupted(node.node_id, note="delayed 3h")
                ctx = SimpleNamespace(db=None,
                                      settings=SimpleNamespace())
                sec = DisruptionProvider().collect(ctx, 0.0)
        self.assertIsNotNone(sec)
        assert sec is not None
        self.assertTrue(any("Lagos flight" in ln for ln in sec.lines))

    def test_disruption_provider_quiet_when_clean(self):
        from nomorals.agents.morning_briefing import DisruptionProvider
        with tempfile.TemporaryDirectory() as td:
            with patch("nomorals.planning.graph._default_db_path",
                       return_value=os.path.join(td, "wg.db")):
                sec = DisruptionProvider().collect(
                    SimpleNamespace(db=None,
                                    settings=SimpleNamespace()), 0.0)
        self.assertIsNone(sec)

    def test_composer_includes_new_providers(self):
        from nomorals.agents.morning_briefing import BriefingComposer
        names = {p.name for p in BriefingComposer().providers}
        self.assertIn("goals", names)
        self.assertIn("disruptions", names)


if __name__ == "__main__":
    unittest.main()
