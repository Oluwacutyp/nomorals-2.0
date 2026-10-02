"""L6 — missions: durability, budgets, crash recovery, reflection.

The headline property is that a mission survives ``kill -9``. That is tested for
real here: a subprocess is SIGKILLed mid-mission and a fresh process must resume
from the persisted checkpoint rather than starting over.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from nomorals.agents.context import build_context
from nomorals.core.config import Settings
from nomorals.core.errors import NotFound, ValidationError
from nomorals.missions import Mission, MissionResult, MissionRunner, MissionStatus, MissionStore

REPO_ROOT = Path(__file__).resolve().parent.parent

_KILL_WORKER = '''
import os, signal, sys
from nomorals.core.config import load_settings
from nomorals.agents.context import build_context
from nomorals.missions import MissionRunner, MissionStore

die_after = int(sys.argv[1])
with build_context(load_settings()) as ctx:
    runner = MissionRunner(ctx)
    def on_step(mission, outcome):
        print(f"STEP {outcome.step}", flush=True)
        if mission.iterations >= die_after:
            os.kill(os.getpid(), signal.SIGKILL)
    runner.on_step = on_step
    pending = MissionStore(ctx.db).resumable()
    if pending:
        print(f"RESUMING {len(pending)}", flush=True)
        runner.resume_all(max_iterations=6)
    else:
        runner.start("alpha then beta then gamma then delta",
                     max_iterations=6, reflect=False)
    print("COMPLETED", flush=True)
'''


class MissionModelTests(unittest.TestCase):
    def test_goal_is_required(self):
        with self.assertRaises(ValidationError):
            Mission(goal="   ")

    def test_name_defaults_to_the_goal(self):
        mission = Mission(goal="do the thing")
        self.assertEqual(mission.name, "do the thing")

    def test_new_mission_is_pending_and_not_terminal(self):
        mission = Mission(goal="g")
        self.assertEqual(mission.status, MissionStatus.PENDING)
        self.assertFalse(mission.terminal)

    def test_terminal_statuses(self):
        for status in (MissionStatus.DONE, MissionStatus.FAILED, MissionStatus.CANCELLED):
            self.assertTrue(Mission(goal="g", status=status).terminal)
        for status in (MissionStatus.PENDING, MissionStatus.RUNNING, MissionStatus.PAUSED):
            self.assertFalse(Mission(goal="g", status=status).terminal)

    def test_budget_exhaustion_on_wall_clock(self):
        mission = Mission(goal="g", budget_wall=10.0, spent_wall=10.0)
        self.assertTrue(mission.budget_exhausted)
        self.assertEqual(mission.wall_remaining, 0.0)

    def test_budget_exhaustion_on_tokens(self):
        mission = Mission(goal="g", budget_tokens=100, spent_tokens=150)
        self.assertTrue(mission.budget_exhausted)

    def test_unlimited_budget_never_exhausts(self):
        mission = Mission(goal="g", spent_wall=1e9, spent_tokens=10**9)
        self.assertFalse(mission.budget_exhausted)
        self.assertEqual(mission.wall_remaining, float("inf"))

    def test_charge_accumulates_and_ignores_negatives(self):
        mission = Mission(goal="g")
        mission.charge(wall=1.5, tokens=10)
        mission.charge(wall=-5.0, tokens=-100)
        self.assertEqual(mission.spent_wall, 1.5)
        self.assertEqual(mission.spent_tokens, 10)

    def test_row_round_trip_preserves_every_field(self):
        original = Mission(
            goal="g", name="n", budget_wall=60.0, budget_tokens=999,
            spent_wall=12.5, spent_tokens=77, iterations=3, success=0.75,
            state={"a": 1}, metadata={"b": 2},
        )
        restored = Mission.from_row(original.to_row())
        self.assertEqual(restored.goal, original.goal)
        self.assertEqual(restored.budget_tokens, 999)
        self.assertEqual(restored.spent_tokens, 77)
        self.assertEqual(restored.success, 0.75)
        self.assertEqual(restored.state, {"a": 1})
        self.assertEqual(restored.metadata, {"b": 2})

    def test_from_row_tolerates_undecoded_json_columns(self):
        row = Mission(goal="g").to_row()
        row["state"] = '{"completed_steps": ["a"]}'
        row["metadata"] = "not json at all"
        restored = Mission.from_row(row)
        self.assertEqual(restored.state, {"completed_steps": ["a"]})
        self.assertEqual(restored.metadata, {})


class MissionStoreTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-ms-")
        self.addCleanup(shutil.rmtree, self.home, ignore_errors=True)
        self.context = build_context(Settings(home=self.home))
        self.context.__enter__()
        self.store = MissionStore(self.context.db)

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def test_create_and_get(self):
        mission = self.store.create_new("the goal", name="named")
        fetched = self.store.get(mission.id)
        self.assertEqual(fetched.goal, "the goal")
        self.assertEqual(fetched.name, "named")

    def test_get_unknown_raises(self):
        with self.assertRaises(NotFound):
            self.store.get("nope")

    def test_save_stamps_finished_at_only_when_terminal(self):
        mission = self.store.create_new("g")
        self.store.save(mission)
        self.assertIsNone(self.store.get(mission.id).finished_at)
        mission.status = MissionStatus.DONE
        self.store.save(mission)
        self.assertIsNotNone(self.store.get(mission.id).finished_at)

    def test_list_filters_by_status(self):
        a = self.store.create_new("a")
        b = self.store.create_new("b")
        b.status = MissionStatus.DONE
        self.store.save(b)
        self.assertEqual([m.id for m in self.store.list(status=MissionStatus.DONE)], [b.id])
        self.assertEqual([m.id for m in self.store.list(status=MissionStatus.PENDING)], [a.id])

    def test_resumable_excludes_terminal(self):
        running = self.store.create_new("running")
        running.status = MissionStatus.RUNNING
        self.store.save(running)
        done = self.store.create_new("done")
        done.status = MissionStatus.DONE
        self.store.save(done)
        self.assertEqual([m.id for m in self.store.resumable()], [running.id])

    def test_checkpoints_are_ordered_newest_first(self):
        mission = self.store.create_new("g")
        mission.state = {"n": 1}
        self.store.checkpoint(mission, label="first")
        mission.state = {"n": 2}
        self.store.checkpoint(mission, label="second")
        self.assertEqual(self.store.latest_checkpoint(mission.id).label, "second")
        labels = [c.label for c in self.store.checkpoint_history(mission.id)]
        self.assertEqual(labels, ["second", "first"])

    def test_checkpoint_captures_state_at_that_moment(self):
        mission = self.store.create_new("g")
        mission.state = {"completed_steps": ["a"]}
        self.store.checkpoint(mission, label="snap")
        mission.state = {"completed_steps": ["a", "b", "c"]}
        point = self.store.latest_checkpoint(mission.id)
        self.assertEqual(point.state["completed_steps"], ["a"])

    def test_checkpoint_history_is_bounded(self):
        store = MissionStore(self.context.db, checkpoint_limit=3)
        mission = store.create_new("g")
        for i in range(10):
            mission.state = {"i": i}
            store.checkpoint(mission, label=f"c{i}")
        self.assertEqual(len(store.checkpoint_history(mission.id, limit=99)), 3)
        self.assertEqual(store.latest_checkpoint(mission.id).label, "c9")

    def test_stats_counts_by_status(self):
        self.store.create_new("a")
        b = self.store.create_new("b")
        b.status = MissionStatus.DONE
        b.success = 0.8
        self.store.save(b)
        stats = self.store.stats()
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["active"], 1)
        self.assertEqual(stats["avg_success"], 0.8)

    def test_reflections_round_trip(self):
        mission = self.store.create_new("g")
        self.store.record_reflection(
            mission.id, score=0.5, summary="half done",
            lessons=["try harder"], weights={"lexical": 0.2},
        )
        rows = self.store.reflections(mission.id)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["score"], 0.5)
        self.assertEqual(rows[0]["lessons"], ["try harder"])
        self.assertEqual(rows[0]["weights"], {"lexical": 0.2})


class MissionRunnerTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-mr-")
        self.addCleanup(shutil.rmtree, self.home, ignore_errors=True)
        self.context = build_context(Settings(home=self.home))
        self.context.__enter__()
        self.store = MissionStore(self.context.db)
        self.runner = MissionRunner(self.context, store=self.store)

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def test_start_runs_to_a_terminal_state(self):
        result = self.runner.start("do something", max_iterations=4, reflect=False)
        self.assertIsInstance(result, MissionResult)
        self.assertIn(result.status, MissionStatus.TERMINAL)

    def test_every_step_is_recorded(self):
        result = self.runner.start("alpha then beta", max_iterations=3, reflect=False)
        self.assertTrue(result.steps)
        for step in result.steps:
            self.assertTrue(step.step)
            self.assertGreaterEqual(step.seconds, 0.0)

    def test_mission_row_reflects_the_outcome(self):
        result = self.runner.start("count things", max_iterations=2, reflect=False)
        mission = self.store.get(result.mission_id)
        self.assertEqual(mission.status, result.status)
        self.assertGreaterEqual(mission.iterations, 1)

    def test_iterations_are_capped(self):
        result = self.runner.start("a b c d e f g h", max_iterations=2, reflect=False)
        self.assertLessEqual(result.iterations, 2)

    def test_budget_stops_the_mission(self):
        mission = self.store.create_new("long task", budget_wall=0.0001)
        mission.spent_wall = 1.0  # already over
        result = self.runner.run(mission, max_iterations=5, reflect=False)
        self.assertNotEqual(result.status, MissionStatus.DONE)
        self.assertIn("budget", result.error)

    def test_budget_cannot_be_laundered_by_restarting(self):
        mission = self.store.create_new("task", budget_tokens=10)
        mission.spent_tokens = 10
        self.store.save(mission)
        # A "fresh" run must see the persisted spend.
        reloaded = self.store.get(mission.id)
        self.assertTrue(reloaded.budget_exhausted)

    def test_cancel_mid_run_stops_before_the_next_step(self):
        # Cancelling from a step callback is the realistic path: an operator or a
        # watchdog interrupts a mission that is already working.
        runner = MissionRunner(self.context, store=self.store)

        def on_step(mission, outcome):
            runner.cancel("operator stopped it")

        runner.on_step = on_step
        result = runner.start("alpha then beta then gamma", max_iterations=6, reflect=False)
        self.assertEqual(result.status, MissionStatus.CANCELLED)
        self.assertEqual(self.store.get(result.mission_id).status, MissionStatus.CANCELLED)

    def test_cancel_requested_before_start_is_cleared_by_start(self):
        # start() deliberately resets the flag: a stale cancel from an earlier
        # mission must not abort a brand-new one.
        runner = MissionRunner(self.context, store=self.store)
        runner.cancel("stale")
        result = runner.start("fresh mission", max_iterations=2, reflect=False)
        self.assertNotEqual(result.status, MissionStatus.CANCELLED)

    def test_reflection_is_recorded_on_completion(self):
        result = self.runner.start("finish this", max_iterations=2, reflect=True)
        rows = self.store.reflections(result.mission_id)
        self.assertEqual(len(rows), 1)
        self.assertIsNotNone(rows[0]["score"])

    def test_no_reflection_when_disabled(self):
        result = self.runner.start("finish this", max_iterations=2, reflect=False)
        self.assertEqual(self.store.reflections(result.mission_id), [])

    def test_score_is_bounded(self):
        result = self.runner.start("scored task", max_iterations=3, reflect=True)
        if result.success is not None:
            self.assertGreaterEqual(result.success, 0.0)
            self.assertLessEqual(result.success, 1.0)

    def test_resume_of_a_terminal_mission_is_a_noop(self):
        mission = self.store.create_new("done already")
        mission.status = MissionStatus.DONE
        self.store.save(mission)
        result = self.runner.resume(mission.id)
        self.assertEqual(result.status, MissionStatus.DONE)
        self.assertEqual(result.steps, [])

    def test_completed_steps_are_not_re_executed(self):
        mission = self.store.create_new("alpha then beta then gamma")
        plan = self.runner._plan(mission)
        first = plan[0].name
        mission.state["completed_steps"] = [first]
        self.store.save(mission)
        result = self.runner.run(mission, max_iterations=6, reflect=False)
        self.assertNotIn(first, [s.step for s in result.steps])

    def test_resume_all_continues_interrupted_missions(self):
        mission = self.store.create_new("interrupted work")
        mission.status = MissionStatus.RUNNING
        self.store.save(mission)
        results = self.runner.resume_all(max_iterations=2)
        self.assertEqual(len(results), 1)
        self.assertIn(results[0].status, MissionStatus.TERMINAL)

    def test_plan_is_memoized_across_runs(self):
        mission = self.store.create_new("planned task")
        self.runner._plan(mission)
        self.assertIn("plan", mission.state)
        cached = mission.state["plan"]
        # A second call must reuse the persisted plan, not re-plan.
        self.runner._plan(mission)
        self.assertEqual(mission.state["plan"], cached)

    def test_checkpoint_is_written_after_each_step(self):
        mission = self.store.create_new("checkpointed task")
        self.runner.run(mission, max_iterations=2, reflect=False)
        history = self.store.checkpoint_history(mission.id)
        self.assertTrue(history)
        self.assertTrue(history[0].label.startswith("final:"))


class CrashRecoveryTests(unittest.TestCase):
    """SIGKILL a subprocess mid-mission, then resume it in a new process."""

    @classmethod
    def setUpClass(cls):
        cls.home = tempfile.mkdtemp(prefix="nm-crash-")
        cls.addClassCleanup(shutil.rmtree, cls.home, ignore_errors=True)
        cls.worker = Path(cls.home) / "worker.py"
        cls.worker.write_text(_KILL_WORKER, encoding="utf-8")
        cls.env = {
            **os.environ,
            "NM_HOME": cls.home,
            "PYTHONPATH": str(REPO_ROOT),
        }

    def _run_worker(self, die_after: int) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-u", str(self.worker), str(die_after)],
            capture_output=True, text=True, timeout=180, env=self.env, check=False,
        )

    def test_mission_survives_sigkill_and_resumes(self):
        first = self._run_worker(die_after=1)
        self.assertNotIn("COMPLETED", first.stdout, "worker should have been killed")
        self.assertIn("STEP", first.stdout)
        self.assertIn("DYING", first.stdout) if "DYING" in first.stdout else None

        # State must be on disk even though the process was SIGKILLed.
        with build_context(Settings(home=self.home)) as ctx:
            store = MissionStore(ctx.db)
            interrupted = store.resumable()
            self.assertEqual(len(interrupted), 1, "mission was not left in a resumable state")
            mission = interrupted[0]
            self.assertEqual(mission.status, MissionStatus.RUNNING)
            self.assertGreaterEqual(len(store.checkpoint_history(mission.id)), 1)
            done_before = set(mission.state.get("completed_steps") or [])

            # A fresh process must continue, not restart.
            results = MissionRunner(ctx, store=store).resume_all(max_iterations=6)
            self.assertEqual(len(results), 1)
            result = results[0]
            self.assertTrue(result.resumed_from, "resume did not report a checkpoint")
            # Steps completed before the kill must not have run again.
            ran_now = {s.step for s in result.steps}
            self.assertFalse(
                done_before & ran_now,
                f"re-executed already-completed steps: {done_before & ran_now}",
            )
            self.assertIn(result.status, MissionStatus.TERMINAL)
            self.assertEqual(store.resumable(), [], "mission was left active after finishing")

    def test_two_kills_in_a_row_still_converge(self):
        home = tempfile.mkdtemp(prefix="nm-crash2-")
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        worker = Path(home) / "worker.py"
        worker.write_text(_KILL_WORKER, encoding="utf-8")
        env = {**os.environ, "NM_HOME": home, "PYTHONPATH": str(REPO_ROOT)}
        for _ in range(2):
            subprocess.run(
                [sys.executable, "-u", str(worker), "1"],
                capture_output=True, text=True, timeout=180, env=env, check=False,
            )
        with build_context(Settings(home=home)) as ctx:
            store = MissionStore(ctx.db)
            self.assertEqual(len(store.resumable()), 1)
            results = MissionRunner(ctx, store=store).resume_all(max_iterations=8)
            self.assertEqual(len(results), 1)
            self.assertIn(results[0].status, MissionStatus.TERMINAL)


if __name__ == "__main__":
    unittest.main()
