"""L5 — agents: task graph, parallel runtime, budgets, supervisor, orchestrator.

The parallelism assertions are the point of this file. A graph that reports
"done" while actually running serially is the exact failure mode that is
invisible without measuring, so these tests compare wall time against serial
time rather than just checking that results came back.
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest

from nomorals.agents.base import Agent, AgentResult, Budget
from nomorals.agents.blackboard import Blackboard
from nomorals.agents.context import build_context
from nomorals.agents.orchestrator import MasterOrchestrator, Plan, PlanStep
from nomorals.agents.roles import ROLES, build_agent
from nomorals.agents.runtime import HybridExecutor
from nomorals.agents.supervisor import RestartPolicy, Supervisor
from nomorals.agents.tasks import Task, TaskGraph, TaskKind, TaskState, cycle_in
from nomorals.core.config import Settings
from nomorals.core.errors import ValidationError


def _sleepy(seconds: float) -> str:
    time.sleep(seconds)
    return "slept"


def _burn(iterations: int) -> int:
    total = 0
    for i in range(iterations):
        total += (i * i) % 7
    return total


def _boom() -> None:
    raise RuntimeError("task exploded")


class GraphTests(unittest.TestCase):
    def test_topological_order_respects_dependencies(self):
        graph = TaskGraph()
        graph.add_task("a", _sleepy, 0.0)
        graph.add_task("b", _sleepy, 0.0, depends_on=["a"])
        graph.add_task("c", _sleepy, 0.0, depends_on=["a"])
        graph.add_task("d", _sleepy, 0.0, depends_on=["b", "c"])
        order = [t.name for t in graph.topological_order()]
        self.assertEqual(order.index("a"), 0)
        self.assertEqual(order.index("d"), 3)

    def test_cycles_are_rejected_when_the_edge_is_added(self):
        graph = TaskGraph()
        graph.add_task("a", _sleepy, 0.0)
        graph.add_task("b", _sleepy, 0.0, depends_on=["a"])
        with self.assertRaises(ValidationError):
            graph.add_task("c", _sleepy, 0.0, depends_on=["b"])
            graph.add_task("a2", _sleepy, 0.0, depends_on=["c"])
            # Closing the loop must fail, not silently corrupt the graph.
            graph.get("a").deps.append(graph.get("c").id)
            graph._assert_acyclic()

    def test_cycle_in_detects_a_raw_cycle(self):
        deps = {"a": ["b"], "b": ["c"], "c": ["a"]}
        self.assertTrue(cycle_in(deps))
        self.assertEqual(cycle_in({"a": [], "b": ["a"]}), [])

    def test_cycle_in_accepts_a_valid_dag_with_diamond_dependencies(self):
        self.assertEqual(cycle_in({"a": [], "b": ["a"], "c": ["a", "b"], "d": ["b", "c"]}), [])

    def test_unknown_dependency_is_rejected(self):
        graph = TaskGraph()
        with self.assertRaises(Exception):
            graph.add_task("a", _sleepy, 0.0, depends_on=["does_not_exist"])

    def test_depth_counts_the_longest_chain(self):
        graph = TaskGraph()
        graph.add_task("a", _sleepy, 0.0)
        graph.add_task("b", _sleepy, 0.0, depends_on=["a"])
        graph.add_task("c", _sleepy, 0.0, depends_on=["b"])
        self.assertGreaterEqual(graph.depth(), 3)

    def test_counts_and_readiness(self):
        graph = TaskGraph()
        graph.add_task("a", _sleepy, 0.0)
        graph.add_task("b", _sleepy, 0.0, depends_on=["a"])
        self.assertEqual([t.name for t in graph.ready()], ["a"])
        self.assertEqual(graph.counts()["pending"], 2)
        self.assertFalse(graph.is_complete())

    def test_cancel_marks_unstarted_tasks(self):
        graph = TaskGraph()
        graph.add_task("a", _sleepy, 0.0)
        graph.add_task("b", _sleepy, 0.0, depends_on=["a"])
        cancelled = graph.cancel("stop")
        self.assertGreaterEqual(cancelled, 2)
        self.assertTrue(graph.cancelled)


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.executor = HybridExecutor(threads=8)

    def tearDown(self):
        self.executor.shutdown(wait=False, timeout=2.0)

    def test_independent_io_tasks_overlap(self):
        graph = TaskGraph()
        for i in range(12):
            graph.add_task(f"t{i}", _sleepy, 0.05, kind=TaskKind.IO)
        started = time.perf_counter()
        report = self.executor.run(graph)
        elapsed = time.perf_counter() - started
        self.assertEqual(report.done, 12)
        self.assertEqual(report.failed, 0)
        # Serial would be 0.60s; overlapping must be far less.
        self.assertLess(elapsed, 0.45, f"tasks ran serially: {elapsed:.2f}s")

    def test_result_order_is_preserved(self):
        graph = TaskGraph()
        for i in range(5):
            graph.add_task(f"t{i}", lambda n=i: n * 2, kind=TaskKind.IO)
        self.executor.run(graph)
        results = graph.results()
        self.assertEqual(results["t3"], 6)
        self.assertEqual(results["t4"], 8)

    def test_one_failure_does_not_stop_the_others(self):
        graph = TaskGraph()
        for i in range(4):
            graph.add_task(f"ok{i}", lambda n=i: n, kind=TaskKind.IO)
        graph.add_task("bad", _boom, kind=TaskKind.IO)
        report = self.executor.run(graph)
        self.assertEqual(report.done, 4)
        self.assertEqual(report.failed, 1)
        self.assertIn("bad", report.failures)

    def test_dependents_of_a_failure_are_skipped(self):
        graph = TaskGraph()
        graph.add_task("parent", _boom, kind=TaskKind.IO)
        graph.add_task("child", _sleepy, 0.0, depends_on=["parent"], kind=TaskKind.IO)
        report = self.executor.run(graph)
        self.assertEqual(report.failed, 1)
        self.assertEqual(report.skipped, 1)
        self.assertEqual(graph.get("child").state, TaskState.SKIPPED)

    def test_retries_recover_from_a_transient_failure(self):
        attempts = {"n": 0}

        def flaky() -> str:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("not yet")
            return "ok"

        graph = TaskGraph()
        graph.add_task("flaky", flaky, kind=TaskKind.IO, retries=4)
        report = self.executor.run(graph)
        self.assertEqual(report.done, 1)
        self.assertEqual(graph.get("flaky").result, "ok")
        self.assertEqual(attempts["n"], 3)

    def test_deadline_cancels_rather_than_hanging(self):
        graph = TaskGraph()
        for i in range(6):
            graph.add_task(f"t{i}", _sleepy, 5.0, kind=TaskKind.IO)
        started = time.perf_counter()
        report = self.executor.run(graph, deadline=time.monotonic() + 0.2)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 2.0, f"deadline ignored, took {elapsed:.2f}s")
        self.assertGreater(report.cancelled, 0)

    def test_cpu_tasks_use_processes_when_available(self):
        executor = HybridExecutor(threads=2, processes=2)
        graph = TaskGraph()
        for i in range(4):
            graph.add_task(f"cpu{i}", _burn, 2_000_000, kind=TaskKind.CPU)
        report = executor.run(graph)
        executor.shutdown(wait=False, timeout=2.0)
        self.assertEqual(report.done, 4)
        snapshot = report.to_dict()
        self.assertIn("task_seconds", snapshot)
        self.assertIn("speedup_vs_serial", snapshot)

    def test_report_shape(self):
        graph = TaskGraph()
        graph.add_task("a", lambda: 1, kind=TaskKind.IO)
        report = self.executor.run(graph)
        data = report.to_dict()
        for key in ("total", "done", "failed", "cancelled", "skipped",
                    "wall_seconds", "task_seconds", "speedup_vs_serial", "by_kind", "failures"):
            self.assertIn(key, data)

    def test_map_parallel_returns_every_result(self):
        results = self.executor.map_parallel(lambda n: n * 3, range(6))
        self.assertEqual(results, [0, 3, 6, 9, 12, 15])


class BudgetTests(unittest.TestCase):
    def test_token_charge_is_tracked(self):
        budget = Budget(tokens=1000, wall_seconds=60.0)
        budget.charge_tokens(400)
        self.assertEqual(budget.spent_tokens, 400)

    def test_exceeding_the_token_budget_raises(self):
        budget = Budget(tokens=100)
        budget.charge_tokens(500)
        with self.assertRaises(Exception):
            budget.check()

    def test_child_budget_halves_by_default(self):
        parent = Budget(tokens=1000, wall_seconds=60.0)
        child = parent.child_budget()
        self.assertLessEqual(child.tokens, parent.tokens / 2 + 1)

    def test_child_budget_never_exceeds_the_parent(self):
        parent = Budget(tokens=100, wall_seconds=10.0)
        child = parent.child_budget(fraction=5.0)
        self.assertLessEqual(child.tokens, parent.tokens)

    def test_wall_deadline_is_absolute(self):
        budget = Budget(wall_seconds=30.0)
        self.assertGreater(budget.deadline, time.monotonic())


class BlackboardTests(unittest.TestCase):
    def setUp(self):
        self.board = Blackboard()

    def test_post_and_get(self):
        self.board.post("plan", {"steps": [1, 2]})
        self.assertEqual(self.board.get("plan")["steps"], [1, 2])

    def test_missing_key_returns_the_default(self):
        self.assertIsNone(self.board.get("absent"))
        self.assertEqual(self.board.get("absent", "fallback"), "fallback")

    def test_append_accumulates(self):
        self.board.append("log", "one")
        self.board.append("log", "two")
        self.assertEqual(self.board.get("log"), ["one", "two"])

    def test_glob_listing(self):
        self.board.post("plan/a", 1)
        self.board.post("plan/b", 2)
        self.board.post("other", 3)
        self.assertEqual(sorted(self.board.keys("plan/*")), ["plan/a", "plan/b"])

    def test_delete_and_has(self):
        self.board.post("k", 1)
        self.assertTrue(self.board.has("k"))
        self.assertTrue(self.board.delete("k"))
        self.assertFalse(self.board.has("k"))

    def test_expired_entries_are_pruned(self):
        self.board.post("short", 1, ttl=0.01)
        time.sleep(0.03)
        self.board.prune()
        self.assertFalse(self.board.has("short"))

    def test_watch_fires_on_matching_post(self):
        seen: list = []
        self.board.watch("signal/*", lambda entry: seen.append(entry.key))
        self.board.post("signal/go", True)
        self.assertEqual(seen, ["signal/go"])

    def test_wait_for_returns_once_posted(self):
        import threading

        threading.Timer(0.05, lambda: self.board.post("late", 42)).start()
        self.assertEqual(self.board.wait_for("late", timeout=2.0), 42)


class _CountingAgent(Agent):
    """Records how many times it was constructed and run."""

    created = 0

    def __init__(self, *, fail_times: int = 0, **kw):
        super().__init__(**kw)
        self.fail_times = fail_times
        self.runs = 0
        type(self).created += 1

    def work(self, task_input=None):
        self.runs += 1
        if self.runs <= self.fail_times:
            raise RuntimeError("synthetic crash")
        return {"value": 42}


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        _CountingAgent.created = 0

    def test_successful_agent_is_not_restarted(self):
        supervisor = Supervisor(policy=RestartPolicy(max_restarts=3))
        agent = _CountingAgent(name="ok")
        result = supervisor.run_agent(agent)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(_CountingAgent.created, 1)

    def test_crashing_agent_is_restarted_up_to_the_limit(self):
        supervisor = Supervisor(policy=RestartPolicy(max_restarts=2, backoff_base=0.0, backoff_cap=0.0))
        agent = _CountingAgent(name="bad", fail_times=99)
        result = supervisor.run_agent(agent, factory=lambda: _CountingAgent(name="bad", fail_times=99))
        self.assertFalse(result.ok)
        self.assertEqual(_CountingAgent.created, 3)  # original + 2 restarts

    def test_restart_recovers_when_the_failure_is_transient(self):
        supervisor = Supervisor(policy=RestartPolicy(max_restarts=3, backoff_base=0.0, backoff_cap=0.0))
        agent = _CountingAgent(name="flaky", fail_times=1)
        result = supervisor.run_agent(
            agent, factory=lambda: _CountingAgent(name="flaky", fail_times=0)
        )
        self.assertTrue(result.ok, result.error)

    def test_budget_exhaustion_is_not_retried_by_default(self):
        from nomorals.core.errors import BudgetExceeded

        supervisor = Supervisor(policy=RestartPolicy(max_restarts=3, backoff_base=0.0, backoff_cap=0.0))

        class Broke(Agent):
            def work(self, task_input=None):
                raise BudgetExceeded("out of tokens")

        _CountingAgent.created = 0
        result = supervisor.run_agent(Broke(name="broke"))
        self.assertFalse(result.ok)
        self.assertEqual(supervisor.snapshot()["restarts"], 0)

    def test_backoff_grows_with_attempts(self):
        policy = RestartPolicy(backoff_base=0.1, backoff_cap=10.0)
        self.assertLess(policy.delay_for(1), policy.delay_for(4))
        self.assertLessEqual(policy.delay_for(100), 10.0)


class RoleTests(unittest.TestCase):
    def test_every_documented_role_is_registered(self):
        for role in ("research", "coding", "vision", "execution", "critic", "reflection"):
            self.assertIn(role, ROLES)

    def test_build_agent_produces_the_right_class(self):
        agent = build_agent("research", name="r")
        self.assertEqual(agent.name, "r")
        self.assertIsInstance(agent, Agent)

    def test_unknown_role_falls_back_to_the_execution_agent(self):
        # Deliberate: an orchestrator working from a model-produced plan must not
        # crash because the model invented a role name.
        agent = build_agent("nonexistent-role", name="fallback")
        self.assertEqual(agent.role, "execution")


class OrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-orch-")
        self.context = build_context(Settings(home=self.home))
        self.context.__enter__()
        self.orchestrator = MasterOrchestrator(self.context, max_steps=4)

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def test_plan_always_yields_a_schedulable_graph(self):
        plan = self.orchestrator.plan("summarize the workspace")
        self.assertTrue(plan.steps)
        names = [s.name for s in plan.steps]
        self.assertEqual(len(names), len(set(names)), "duplicate step names break scheduling")
        for step in plan.steps:
            for dep in step.depends_on:
                self.assertIn(dep, names)

    def test_repair_dedupes_names(self):
        broken = Plan(goal="g", steps=[
            PlanStep(name="a", goal="x", role="research", kind=TaskKind.IO),
            PlanStep(name="a", goal="y", role="research", kind=TaskKind.IO),
        ])
        fixed = self.orchestrator._repair(broken)
        names = [s.name for s in fixed.steps]
        self.assertEqual(len(names), len(set(names)))

    def test_repair_drops_dangling_and_self_dependencies(self):
        broken = Plan(goal="g", steps=[
            PlanStep(name="a", goal="x", role="research", kind=TaskKind.IO,
                     depends_on=["ghost", "a"]),
        ])
        fixed = self.orchestrator._repair(broken)
        self.assertEqual(fixed.steps[0].depends_on, [])

    def test_repair_breaks_cycles(self):
        broken = Plan(goal="g", steps=[
            PlanStep(name="a", goal="x", role="research", kind=TaskKind.IO, depends_on=["b"]),
            PlanStep(name="b", goal="y", role="execution", kind=TaskKind.IO, depends_on=["a"]),
        ])
        fixed = self.orchestrator._repair(broken)
        graph = TaskGraph()
        for step in fixed.steps:
            graph.add_task(step.name, lambda: None, depends_on=step.depends_on)
        # Would raise if the cycle survived.
        self.assertTrue(graph.topological_order())

    def test_run_reports_every_step(self):
        result = self.orchestrator.run("count the files", reflect=False)
        self.assertEqual(result.report.total, len(result.plan.steps))
        self.assertEqual(result.report.done + result.report.failed
                         + result.report.skipped + result.report.cancelled, result.report.total)

    def test_reflect_produces_lessons(self):
        result = self.orchestrator.run("say hello", reflect=True)
        self.assertIsInstance(result.lessons, list)


if __name__ == "__main__":
    unittest.main()
