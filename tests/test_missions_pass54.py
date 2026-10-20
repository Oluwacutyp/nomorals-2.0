"""Pass #54 — missions/projects execution engine: crash truth, step
integrity, concurrency, completion delivery.

Every test drives the real store + runner (or the real ProjectManager)
with the documented seams stubbed (``orchestrator_factory`` /
``runner._execute_step`` / project's ``executor=``). Crashes are fault
injections at step boundaries, never mocks of the store.
"""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from nomorals.agents.orchestrator import PlanStep
from nomorals.agents.projects import ProjectManager
from nomorals.core.errors import ValidationError
from nomorals.core.tasks import TaskKind
from nomorals.missions import MissionStatus, MissionStore
from nomorals.missions.runner import MissionRunner, StepOutcome, _serialize_plan
from nomorals.storage.db import Database


def _ctx(db=None):
    if db is None:
        db = Database(":memory:")
        db.migrate()
    return SimpleNamespace(db=db, tools=SimpleNamespace(), memory=None,
                           settings=SimpleNamespace(), extras={})


def _step(name, deps=()):
    return PlanStep(name=name, goal=f"do {name}", role="execution",
                    kind=TaskKind.IO, depends_on=list(deps), payload={})


def _mission_with_plan(store, steps, name="m"):
    mission = store.create_new("test goal", name=name)
    mission.state["plan"] = _serialize_plan(SimpleNamespace(steps=steps))
    store.save(mission)
    return mission


def _runner(ctx, store, executed, fail_on=(), **kw):
    runner = MissionRunner(ctx, store=store, milestones=False,
                           orchestrator_factory=lambda: MagicMock(), **kw)

    def _exec(mission, step):
        executed.append(step.name)
        if step.name in fail_on:
            return StepOutcome(step=step.name, ok=False,
                               detail=f"{step.name} exploded", seconds=0.1)
        return StepOutcome(step=step.name, ok=True, seconds=0.1,
                           payload={"done": step.name})

    runner._execute_step = _exec
    return runner


class IterationCapTests(unittest.TestCase):
    """The cap counts THIS run's steps; <= 0 means unlimited."""

    def setUp(self):
        self.ctx = _ctx()
        self.store = MissionStore(self.ctx.db)

    def test_zero_max_iterations_is_unlimited(self):
        # agents/devon.py spawns chat missions with max_iterations=0
        # documented as "run unlimited (the budget stops it)" — the run
        # must execute its steps, not die at "iteration limit 0".
        mission = _mission_with_plan(self.store, [_step("a"), _step("b")])
        executed: list[str] = []
        runner = _runner(self.ctx, self.store, executed)
        result = runner.run(self.store.get(mission.id),
                            max_iterations=0, reflect=False)
        self.assertEqual(result.status, MissionStatus.DONE)
        self.assertEqual(executed, ["a", "b"])

    def test_resume_cap_counts_this_runs_steps_not_lifetime(self):
        # A mission that already spent 8 lifetime iterations (e.g. under
        # a larger CLI cap) must still make progress when resumed with
        # the chat default cap of 8: the cap buys 8 MORE steps.
        steps = [_step(f"s{i}") for i in range(10)]
        mission = _mission_with_plan(self.store, steps)
        row = self.store.get(mission.id)
        row.iterations = 8
        row.state["completed_steps"] = [f"s{i}" for i in range(8)]
        row.status = MissionStatus.PAUSED
        self.store.save(row)
        executed: list[str] = []
        runner = _runner(self.ctx, self.store, executed)
        result = runner.resume(mission.id, reflect=False)  # default cap 8
        self.assertEqual(result.status, MissionStatus.DONE)
        self.assertEqual(executed, ["s8", "s9"])

    def test_cap_still_stops_a_fresh_run(self):
        steps = [_step(f"s{i}") for i in range(6)]
        mission = _mission_with_plan(self.store, steps)
        executed: list[str] = []
        runner = _runner(self.ctx, self.store, executed)
        result = runner.run(self.store.get(mission.id),
                            max_iterations=2, reflect=False)
        self.assertEqual(result.status, MissionStatus.FAILED)
        self.assertIn("iteration limit 2", result.error)
        self.assertEqual(len(executed), 2)


class ExternalStatusTests(unittest.TestCase):
    """A status written by another process (chat pause/cancel) must be
    honored by an in-flight run — never overwritten by it."""

    def _file_ctx(self):
        tmp = Path(tempfile.mkdtemp(prefix="nm-p54-"))
        db = Database(str(tmp / "t.db"))
        db.migrate()
        return _ctx(db), MissionStore(db)

    def _run_with_blocked_first_step(self, store, ctx, mission_id):
        executed: list[str] = []
        runner = MissionRunner(ctx, store=store, milestones=False,
                               orchestrator_factory=lambda: MagicMock())
        release = threading.Event()

        def _exec(mission, step):
            executed.append(step.name)
            if step.name == "a":
                release.wait(timeout=30)
            return StepOutcome(step=step.name, ok=True, seconds=0.1,
                               payload={"done": step.name})

        runner._execute_step = _exec
        holder: dict = {}
        thread = threading.Thread(
            target=lambda: holder.setdefault(
                "res", runner.run(store.get(mission_id), reflect=False)),
            daemon=True)
        thread.start()
        for _ in range(100):
            if executed:
                break
            time.sleep(0.05)
        self.assertEqual(executed, ["a"])
        return runner, release, holder, thread, executed

    def test_external_cancel_stops_the_run_and_sticks(self):
        ctx, store = self._file_ctx()
        mission = _mission_with_plan(
            store, [_step("a"), _step("b"), _step("c")])
        _, release, holder, thread, executed = \
            self._run_with_blocked_first_step(store, ctx, mission.id)
        # exactly what /mission cancel does: store flip + cancel() on a
        # freshly built runner (a different instance than the driver).
        store.set_status(mission.id, MissionStatus.CANCELLED,
                         note="cancelled by owner")
        MissionRunner(ctx, store=store, milestones=False).cancel("owner")
        release.set()
        thread.join(timeout=30)
        result = holder["res"]
        self.assertEqual(result.status, MissionStatus.CANCELLED)
        self.assertEqual(executed, ["a"])  # b and c never ran
        final = store.get(mission.id)
        self.assertEqual(final.status, MissionStatus.CANCELLED)

    def test_external_pause_parks_the_run_resumably(self):
        ctx, store = self._file_ctx()
        mission = _mission_with_plan(
            store, [_step("a"), _step("b"), _step("c")])
        _, release, holder, thread, executed = \
            self._run_with_blocked_first_step(store, ctx, mission.id)
        store.set_status(mission.id, MissionStatus.PAUSED,
                         note="paused by owner")
        release.set()
        thread.join(timeout=30)
        result = holder["res"]
        self.assertEqual(result.status, MissionStatus.PAUSED)
        self.assertEqual(executed, ["a"])
        final = store.get(mission.id)
        self.assertEqual(final.status, MissionStatus.PAUSED)
        self.assertEqual(final.state.get("completed_steps"), ["a"])
        # and the parked mission resumes to completion afterwards
        executed2: list[str] = []
        runner2 = _runner(ctx, store, executed2)
        result2 = runner2.resume(mission.id, reflect=False)
        self.assertEqual(result2.status, MissionStatus.DONE)
        self.assertEqual(executed2, ["b", "c"])


class RunClaimTests(unittest.TestCase):
    """Two drivers must never execute the same mission concurrently."""

    def setUp(self):
        tmp = Path(tempfile.mkdtemp(prefix="nm-p54-claim-"))
        db = Database(str(tmp / "t.db"))
        db.migrate()
        self.ctx = _ctx(db)
        self.store = MissionStore(db)

    def test_second_concurrent_driver_is_refused(self):
        mission = _mission_with_plan(
            self.store, [_step("a"), _step("b"), _step("c")])
        row = self.store.get(mission.id)
        row.status = MissionStatus.PAUSED
        self.store.save(row)
        executed: list[str] = []
        lock = threading.Lock()
        release = threading.Event()

        def _exec(mission_, step):
            with lock:
                executed.append(step.name)
            if step.name == "a":
                release.wait(timeout=30)
            return StepOutcome(step=step.name, ok=True, seconds=0.1,
                               payload={"done": step.name})

        def _drive(results, key):
            runner = MissionRunner(self.ctx, store=self.store,
                                   milestones=False,
                                   orchestrator_factory=lambda: MagicMock())
            runner._execute_step = _exec
            try:
                results[key] = runner.resume(mission.id, reflect=False)
            except Exception as exc:  # noqa: BLE001 - recorded for assert
                results[key] = exc

        results: dict = {}
        t1 = threading.Thread(target=_drive, args=(results, "one"),
                              daemon=True)
        t1.start()
        for _ in range(100):
            if executed:
                break
            time.sleep(0.05)
        self.assertEqual(executed, ["a"])  # first driver is mid-step
        t2 = threading.Thread(target=_drive, args=(results, "two"),
                              daemon=True)
        t2.start()
        t2.join(timeout=30)
        self.assertIsInstance(results.get("two"), ValidationError)
        release.set()
        t1.join(timeout=30)
        self.assertEqual(results["one"].status, MissionStatus.DONE)
        # every step executed exactly once across both drivers
        self.assertEqual(sorted(executed), ["a", "b", "c"])
        # the claim is released: the mission can be driven again
        row = self.store.get(mission.id)
        self.assertFalse((row.state.get("run_claim") or {}))

    def test_dead_drivers_claim_is_stealable(self):
        mission = _mission_with_plan(self.store, [_step("a")])
        row = self.store.get(mission.id)
        row.status = MissionStatus.RUNNING
        # a claim from a process that no longer exists, heartbeat stale
        row.state["run_claim"] = {"token": "ghost", "pid": 99999999,
                                  "at": time.time() - 100000}
        row.state["heartbeat"] = {"pid": 99999999,
                                   "at": time.time() - 100000}
        self.store.save(row)
        executed: list[str] = []
        runner = _runner(self.ctx, self.store, executed)
        result = runner.resume(mission.id, reflect=False)
        self.assertEqual(result.status, MissionStatus.DONE)
        self.assertEqual(executed, ["a"])


class _RecordingReporter:
    """Duck-type of MissionMilestones: marks live in mission.state
    (the real reporter's persisted marker log), pushes are recorded."""

    def __init__(self):
        self.events: list[str] = []

    def _marked(self, mission, key):
        log = mission.state.get("milestones") or []
        return any(isinstance(e, dict) and e.get("event") == key
                   for e in log)

    def on_started(self, mission):
        self.events.append("started")

    def on_step(self, mission, outcome):
        self.events.append("step")

    def on_stalled(self, mission):
        self.events.append("stalled")

    def on_terminal(self, mission, status, error=""):
        if self._marked(mission, f"terminal:{status}"):
            return
        self.events.append(f"terminal:{status}")
        log = list(mission.state.get("milestones") or [])
        log.append({"event": f"terminal:{status}", "title": "",
                    "at": time.time()})
        mission.state["milestones"] = log


class TerminalCatchUpTests(unittest.TestCase):
    """A terminal push lost to a crash between the terminal save and the
    notification is re-delivered — exactly once — by the next resume."""

    def setUp(self):
        self.ctx = _ctx()
        self.store = MissionStore(self.ctx.db)

    def test_unmarked_terminal_mission_redelivers_once(self):
        mission = _mission_with_plan(self.store, [_step("a")])
        row = self.store.get(mission.id)
        row.status = MissionStatus.DONE  # saved terminal…
        row.state["completed_steps"] = ["a"]
        # …but the terminal milestone was never marked (process died
        # between _finish's save and the push landing).
        self.store.save(row)
        reporter = _RecordingReporter()
        runner = MissionRunner(self.ctx, store=self.store, milestones=False,
                               milestone_reporter=reporter,
                               orchestrator_factory=lambda: MagicMock())
        result = runner.resume(mission.id, reflect=False)
        self.assertEqual(result.status, MissionStatus.DONE)
        self.assertEqual(reporter.events, ["terminal:done"])
        # the catch-up persisted the marker: a fresh runner + reporter
        # looking at the same mission does not push a second time
        reporter2 = _RecordingReporter()
        runner2 = MissionRunner(self.ctx, store=self.store, milestones=False,
                                milestone_reporter=reporter2,
                                orchestrator_factory=lambda: MagicMock())
        runner2.resume(mission.id, reflect=False)
        self.assertEqual(reporter2.events, [])


class ProjectCrashTests(unittest.TestCase):
    """ProjectManager.advance: a step's attempt must be persisted BEFORE
    it executes, and an interrupted step must be recovered — counted,
    never silently re-run as if nothing happened."""

    def _manager(self):
        db = Database(":memory:")
        db.migrate()
        ctx = SimpleNamespace(
            db=db, tools=SimpleNamespace(), memory=None,
            settings=SimpleNamespace(workspace_dir="/tmp",
                                     rooms_auto_create=False),
            extras={})
        return ProjectManager(ctx), ctx

    def test_crash_mid_step_is_persisted_and_counted(self):
        mgr, _ = self._manager()
        project = mgr.create("crash test", objective="do two things",
                             steps=["first step", "second step"])
        effects: list[str] = []

        def crashing(desc: str) -> str:
            effects.append(desc)
            if desc == "second step":
                raise KeyboardInterrupt()  # process death mid-step
            return "ok"

        with self.assertRaises(KeyboardInterrupt):
            mgr.advance(project.id, executor=crashing)
            mgr.advance(project.id, executor=crashing)
        row = mgr._load(project.id)
        second = [s for s in row.steps if s.description == "second step"][0]
        # the interrupted attempt is on the record: in_progress, counted
        self.assertEqual(second.status, "in_progress")
        self.assertEqual(second.attempts, 1)
        # recovery: the next advance resolves the interruption honestly
        mgr2 = ProjectManager(mgr.context)  # same store, fresh manager
        out = mgr2.run(project.id, executor=lambda d: effects.append(d) or "ok")
        self.assertEqual(out["status"], "done")
        row = mgr2._load(project.id)
        second = [s for s in row.steps if s.description == "second step"][0]
        self.assertEqual(second.status, "done")
        self.assertEqual(second.attempts, 2)  # crash counted, not erased

    def test_step_that_keeps_killing_the_process_fails_bounded(self):
        mgr, _ = self._manager()
        project = mgr.create("killer step", objective="one thing",
                             steps=["only step"])

        def crashing(desc: str) -> str:
            raise KeyboardInterrupt()

        # each "process" gets one advance; the step dies every time.
        for _ in range(3):
            with self.assertRaises(KeyboardInterrupt):
                mgr.advance(project.id, executor=crashing)
        # attempts are exhausted (max_attempts=3): recovery fails the
        # project honestly instead of scheduling a 4th execution.
        out = mgr.run(project.id, executor=lambda d: "ok")
        self.assertEqual(out["status"], "failed")
        row = mgr._load(project.id)
        self.assertEqual(row.steps[0].status, "failed")
        self.assertIn("interrupted", row.steps[0].result)


if __name__ == "__main__":
    unittest.main()
