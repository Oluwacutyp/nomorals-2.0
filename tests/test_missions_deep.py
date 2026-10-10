"""Mission deep upgrades: dependency-aware execution, replanning,
degraded mode, lock keys, ledger, bus events."""

from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from nomorals.agents.orchestrator import Plan, PlanStep
from nomorals.core.errors import ValidationError
from nomorals.core.events import EventBus
from nomorals.missions import MissionStatus, MissionStore
from nomorals.missions.runner import (
    MissionRunner,
    StepOutcome,
    _rehydrate_plan,
    _serialize_plan,
    _step_policy,
    _topo_levels,
)
from nomorals.storage.db import Database


def _ctx():
    db = Database(":memory:")
    db.migrate()
    return SimpleNamespace(db=db, tools=SimpleNamespace(), memory=None,
                          settings=SimpleNamespace(), extras={})


def _step(name, deps=(), **payload):
    from nomorals.core.tasks import TaskKind

    return PlanStep(name=name, goal=f"do {name}", role="execution",
                    kind=TaskKind.IO, depends_on=list(deps), payload=payload)


def _mission_with_plan(store, steps, name="m"):
    mission = store.create_new("test goal", name=name)
    mission.state["plan"] = _serialize_plan(SimpleNamespace(steps=steps))
    store.save(mission)
    return mission


def _ok_runner(ctx, store, fail_on=(), **kw):
    """Runner whose steps succeed except the named ones."""
    runner = MissionRunner(ctx, store=store, milestones=False,
                           orchestrator_factory=lambda: MagicMock(), **kw)

    def _exec(mission, step):
        if step.name in fail_on:
            return StepOutcome(step=step.name, ok=False,
                               detail=f"{step.name} exploded", seconds=0.5)
        return StepOutcome(step=step.name, ok=True, seconds=0.5,
                           payload={"done": step.name})

    runner._execute_step = _exec
    return runner


class TopoLevelsTests(unittest.TestCase):
    def test_linear_chain(self):
        levels = _topo_levels([_step("a"), _step("b", ["a"]),
                               _step("c", ["b"])])
        self.assertEqual([[s.name for s in lvl] for lvl in levels],
                         [["a"], ["b"], ["c"]])

    def test_fan_out(self):
        levels = _topo_levels([_step("a"), _step("b"),
                               _step("c", ["a", "b"])])
        self.assertEqual([[s.name for s in lvl] for lvl in levels],
                         [["a", "b"], ["c"]])

    def test_cycle_raises(self):
        with self.assertRaises(ValidationError):
            _topo_levels([_step("a", ["b"]), _step("b", ["a"])])

    def test_unknown_dependency_raises(self):
        with self.assertRaises(ValidationError):
            _topo_levels([_step("a", ["ghost"])])

    def test_empty(self):
        self.assertEqual(_topo_levels([]), [])


class StepPolicyTests(unittest.TestCase):
    def test_defaults(self):
        p = _step_policy(_step("a"))
        self.assertEqual(p, {"optional": False, "on_failure": "fail_fast",
                             "retries": 0, "retry_backoff_s": 0.0,
                             "retry_on": "transient", "timeout_s": 0.0,
                             "schedule_timeout_s": 0.0,
                             "needs_approval": False, "compensate": ""})

    def test_optional_and_continue(self):
        p = _step_policy(_step("a", optional=True, on_failure="continue"))
        self.assertTrue(p["optional"])
        self.assertEqual(p["on_failure"], "continue")

    def test_bad_on_failure_degrades_to_fail_fast(self):
        p = _step_policy(_step("a", on_failure="explode"))
        self.assertEqual(p["on_failure"], "fail_fast")

    def test_policy_round_trips_through_serialization(self):
        steps = [_step("a", optional=True, on_failure="continue")]
        raw = _serialize_plan(SimpleNamespace(steps=steps))
        back = _rehydrate_plan("g", raw)
        self.assertEqual(_step_policy(back[0])["on_failure"], "continue")
        self.assertTrue(_step_policy(back[0])["optional"])


class DependencyExecutionTests(unittest.TestCase):
    def setUp(self):
        self.ctx = _ctx()
        self.store = MissionStore(self.ctx.db)

    def test_failed_step_blocks_dependents(self):
        mission = _mission_with_plan(
            self.store,
            [_step("a"), _step("b", ["a"]), _step("c", ["b"]),
             _step("solo")])
        mission.metadata["replan_policy"] = "off"  # isolate the gate logic
        self.store.save(mission)
        runner = _ok_runner(self.ctx, self.store, fail_on={"b"})
        result = runner.run(mission, reflect=False)
        self.assertEqual(result.status, MissionStatus.FAILED)
        by_name = {s.step: s for s in result.steps}
        # c never executed — blocked by b
        self.assertIn("blocked", by_name["c"].detail)
        self.assertFalse(by_name["c"].ok)
        # solo (independent) still ran
        self.assertTrue(by_name["solo"].ok)

    def test_independent_levels_all_run(self):
        mission = _mission_with_plan(
            self.store,
            [_step("a"), _step("b"), _step("c", ["a", "b"])])
        runner = _ok_runner(self.ctx, self.store)
        result = runner.run(mission, reflect=False)
        self.assertTrue(result.ok)
        self.assertEqual(
            sorted(s.step for s in result.steps if s.ok), ["a", "b", "c"])

    def test_resume_skips_completed(self):
        mission = _mission_with_plan(
            self.store, [_step("a"), _step("b", ["a"])])
        mission.state["completed_steps"] = ["a"]
        self.store.save(mission)
        seen = []
        runner = _ok_runner(self.ctx, self.store)
        orig = runner._execute_step

        def _exec(m, step):
            seen.append(step.name)
            return orig(m, step)

        runner._execute_step = _exec
        result = runner.run(mission, reflect=False)
        self.assertTrue(result.ok)
        self.assertEqual(seen, ["b"])


class DegradedModeTests(unittest.TestCase):
    def setUp(self):
        self.ctx = _ctx()
        self.store = MissionStore(self.ctx.db)

    def test_optional_failure_does_not_fail_mission(self):
        mission = _mission_with_plan(
            self.store,
            [_step("a"), _step("b", ["a"], optional=True),
             _step("c", ["a"])])
        runner = _ok_runner(self.ctx, self.store, fail_on={"b"})
        result = runner.run(mission, reflect=False)
        self.assertTrue(result.ok)
        self.assertEqual(result.status, MissionStatus.DONE)

    def test_on_failure_continue_runs_later_steps(self):
        mission = _mission_with_plan(
            self.store,
            [_step("a", on_failure="continue"), _step("b")])
        runner = _ok_runner(self.ctx, self.store, fail_on={"a"})
        result = runner.run(mission, reflect=False)
        by_name = {s.step: s for s in result.steps}
        self.assertTrue(by_name["b"].ok)  # b still ran
        self.assertEqual(result.status, MissionStatus.FAILED)  # but not DONE

    def test_fail_fast_stops(self):
        mission = _mission_with_plan(
            self.store, [_step("a"), _step("b")])
        mission.metadata["replan_policy"] = "off"
        self.store.save(mission)
        runner = _ok_runner(self.ctx, self.store, fail_on={"a"})
        result = runner.run(mission, reflect=False)
        by_name = {s.step: s for s in result.steps}
        # b never executed — recorded as not reached, not as a run
        self.assertTrue(by_name["b"].detail.startswith("not reached"))
        self.assertEqual(result.status, MissionStatus.FAILED)


class ReplanTests(unittest.TestCase):
    def setUp(self):
        self.ctx = _ctx()
        self.store = MissionStore(self.ctx.db)

    def _replan_runner(self, new_steps):
        """Runner whose orchestrator returns new_steps on replan."""
        plan = Plan(goal="recovery", steps=new_steps)

        def factory():
            orch = MagicMock()
            orch.plan.return_value = plan
            return orch

        runner = MissionRunner(self.ctx, store=self.store, milestones=False,
                               orchestrator_factory=factory)

        def _exec(mission, step):
            if step.name == "bad":
                return StepOutcome(step=step.name, ok=False,
                                   detail="bad exploded", seconds=0.2)
            return StepOutcome(step=step.name, ok=True, seconds=0.2)

        runner._execute_step = _exec
        return runner

    def test_replan_recovers(self):
        mission = _mission_with_plan(
            self.store, [_step("good"), _step("bad", ["good"])])
        runner = self._replan_runner(
            [_step("fix1"), _step("fix2", ["fix1"])])
        result = runner.run(mission, reflect=False)
        self.assertTrue(result.ok, result.error)
        ran = [s.step for s in result.steps]
        self.assertIn("fix1", ran)
        self.assertIn("fix2", ran)
        m = self.store.get(mission.id)
        self.assertEqual(m.state.get("replans"), 1)

    def test_replan_policy_off(self):
        mission = _mission_with_plan(
            self.store, [_step("bad")])
        mission.metadata["replan_policy"] = "off"
        self.store.save(mission)
        runner = self._replan_runner([_step("fix1")])
        result = runner.run(mission, reflect=False)
        self.assertEqual(result.status, MissionStatus.FAILED)
        self.assertNotIn("fix1", [s.step for s in result.steps])

    def test_replan_budget_spent(self):
        mission = _mission_with_plan(
            self.store, [_step("bad")])
        mission.state["replans"] = 1  # auto policy already used its one
        self.store.save(mission)
        runner = self._replan_runner([_step("fix1")])
        result = runner.run(mission, reflect=False)
        self.assertEqual(result.status, MissionStatus.FAILED)


class LockKeyTests(unittest.TestCase):
    def setUp(self):
        self.ctx = _ctx()
        self.store = MissionStore(self.ctx.db)

    def test_second_mission_with_same_lock_refused(self):
        runner = MissionRunner(self.ctx, store=self.store, milestones=False,
                               orchestrator_factory=lambda: MagicMock())
        runner._execute_step = lambda m, s: StepOutcome(
            step=s.name, ok=True, seconds=0.01)
        m1 = self.store.create_new("goal one",
                                   metadata={"lock_key": "nightly-backup"})
        m1.status = MissionStatus.RUNNING
        self.store.save(m1)
        with self.assertRaises(ValidationError):
            runner.start("goal two", metadata={"lock_key": "nightly-backup"},
                         reflect=False, max_iterations=1)

    def test_terminal_mission_releases_lock(self):
        runner = MissionRunner(self.ctx, store=self.store, milestones=False,
                               orchestrator_factory=lambda: MagicMock())
        runner._execute_step = lambda m, s: StepOutcome(
            step=s.name, ok=True, seconds=0.01)
        m1 = self.store.create_new("goal one",
                                   metadata={"lock_key": "k"})
        m1.status = MissionStatus.DONE
        self.store.save(m1)
        # no plan needed: start() plans via the factory stub (no steps)
        factory_orch = MagicMock()
        factory_orch.plan.return_value = Plan(goal="g", steps=[])
        runner2 = MissionRunner(
            self.ctx, store=self.store, milestones=False,
            orchestrator_factory=lambda: factory_orch)
        result = runner2.start("goal two", metadata={"lock_key": "k"},
                               reflect=False, max_iterations=1)
        self.assertTrue(result.ok)


class LedgerAndBusTests(unittest.TestCase):
    def setUp(self):
        self.ctx = _ctx()
        self.store = MissionStore(self.ctx.db)

    def test_steps_and_terminal_land_in_ledger(self):
        from nomorals.agents.autonomy_ledger import AutonomyLedger

        mission = _mission_with_plan(self.store, [_step("a"), _step("b")])
        runner = _ok_runner(self.ctx, self.store)
        runner.run(mission, reflect=False)
        entries = AutonomyLedger(self.ctx.db).recent(
            limit=20, system="mission")
        kinds = [e["kind"] for e in entries]
        self.assertIn("step", kinds)
        self.assertIn("terminal", kinds)
        terminal = next(e for e in entries if e["kind"] == "terminal")
        self.assertTrue(terminal["ok"])

    def test_bus_events(self):
        import nomorals.missions.runner as runner_mod

        bus = EventBus()
        seen = []
        bus.subscribe("mission.*", lambda e: seen.append(e.topic), sync=True)
        orig = runner_mod.global_bus
        runner_mod.global_bus = bus
        try:
            mission = _mission_with_plan(self.store, [_step("a")])
            runner = _ok_runner(self.ctx, self.store)
            runner.run(mission, reflect=False)
        finally:
            runner_mod.global_bus = orig
        self.assertIn("mission.step.finished", seen)
        self.assertIn("mission.terminal", seen)


class EscalationTests(unittest.TestCase):
    def test_failed_mission_escalates(self):
        ctx = _ctx()
        store = MissionStore(ctx.db)
        published = []

        class FakeNotifier:
            def __init__(self, context):
                pass

            def publish(self, kind, title, body=""):
                published.append((kind, title, body))
                return {"delivered": True}

        import nomorals.missions.runner as runner_mod
        import nomorals.agents.notifier as notifier_mod

        orig = notifier_mod.Notifier
        notifier_mod.Notifier = FakeNotifier
        try:
            mission = _mission_with_plan(store, [_step("bad")])
            runner = MissionRunner(ctx, store=store, milestones=False,
                                   orchestrator_factory=lambda: MagicMock())
            runner._execute_step = lambda m, s: StepOutcome(
                step=s.name, ok=False, detail="boom", seconds=0.1)
            result = runner.run(mission, reflect=False)
        finally:
            notifier_mod.Notifier = orig
        self.assertEqual(result.status, MissionStatus.FAILED)
        self.assertTrue(published)
        self.assertIn("mission failed", published[0][1])


if __name__ == "__main__":
    unittest.main()
