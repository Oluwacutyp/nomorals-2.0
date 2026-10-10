"""Phase 9 Slice A — god-tier missions.

Covers the new breakable behavior only (existing suites cover the rest):
per-step retries with jittered backoff, per-step timeouts, bounded
intra-level parallelism, idempotency-by-default in wired_runner,
idempotency purge hygiene, trailing-window ETA, attempt surfacing in
detail(), resume_all robustness, and golden attempt counts.

Unit tier: fully offline. Real MissionStore on an in-memory database; the
runner is driven with stubbed agents so no LLM is ever constructed.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from nomorals.agents.orchestrator import PlanStep
from nomorals.core.tasks import TaskKind
from nomorals.missions import (
    IdempotencyStore,
    MissionRunner,
    MissionStatus,
    MissionStore,
    wired_runner,
)
from nomorals.missions.golden import GoldenContext, GoldenRunner, GoldenStep
from nomorals.missions.idempotency import COMPLETED, FAILED, RUNNING
from nomorals.missions.progress import estimate_eta
from nomorals.missions.runner import (
    StepOutcome,
    _backoff_delay,
    _retryable,
    _serialize_plan,
    _step_policy,
)
from nomorals.storage.db import Database


def make_db():
    db = Database(":memory:")
    db.migrate()
    return db


def make_ctx(db):
    return SimpleNamespace(db=db, memory=None)


def _step(name, deps=(), **payload):
    return PlanStep(name=name, goal=f"do {name}", role="execution",
                    kind=TaskKind.IO, depends_on=list(deps), payload=payload)


def _mission_with_plan(store, steps, name="m", **meta):
    mission = store.create_new("test goal", name=name,
                               metadata=dict(meta) if meta else {})
    mission.state["plan"] = _serialize_plan(SimpleNamespace(steps=steps))
    store.save(mission)
    return mission


def _runner(ctx, store, **kw):
    kw.setdefault("milestones", False)
    kw.setdefault("orchestrator_factory", lambda: MagicMock())
    return MissionRunner(ctx, store=store, **kw)


# ── retry policy ─────────────────────────────────────────────────────────────

class RetryPolicyTests(unittest.TestCase):
    def setUp(self):
        self.db = make_db()
        self.addCleanup(self.db.close)
        self.ctx = make_ctx(self.db)
        self.store = MissionStore(self.db)

    def _retry_runner(self, script, **kw):
        """Stub _run_step_agent with a per-call script of outcomes."""
        runner = _runner(self.ctx, self.store, **kw)
        calls = []

        def fake_run(mission, step, started, policy=None):
            calls.append(step.name)
            action = script[min(len(calls) - 1, len(script) - 1)]
            if action == "transient":
                return StepOutcome(step=step.name, ok=False,
                                   detail="connection reset by peer",
                                   seconds=0.1)
            if action == "fatal":
                return StepOutcome(step=step.name, ok=False,
                                   detail="ValidationError: bad args",
                                   seconds=0.1)
            if action == "plain":
                return StepOutcome(step=step.name, ok=False,
                                   detail="boom", seconds=0.1)
            return StepOutcome(step=step.name, ok=True, seconds=0.1,
                               payload={"done": step.name})

        runner._run_step_agent = fake_run
        runner.calls = calls
        return runner

    def test_transient_failure_retries_then_succeeds(self):
        runner = self._retry_runner(["transient", "transient", "ok"])
        mission = self.store.create_new("g")
        step = _step("a", retries=2, retry_backoff_s=0.01)
        outcome = runner._execute_step(mission, step)
        self.assertTrue(outcome.ok)
        self.assertEqual(runner.calls, ["a", "a", "a"])
        self.assertEqual(mission.state["step_attempts"]["a"], 3)
        self.assertEqual(outcome.payload["_attempts"], 3)

    def test_non_transient_failure_does_not_retry(self):
        runner = self._retry_runner(["fatal", "ok"])
        mission = self.store.create_new("g")
        step = _step("a", retries=3, retry_backoff_s=0.01)
        outcome = runner._execute_step(mission, step)
        self.assertFalse(outcome.ok)
        self.assertEqual(runner.calls, ["a"])  # ValidationError: never retried
        self.assertEqual(mission.state["step_attempts"]["a"], 1)

    def test_no_retries_by_default(self):
        runner = self._retry_runner(["transient", "ok"])
        mission = self.store.create_new("g")
        outcome = runner._execute_step(mission, _step("a"))
        self.assertFalse(outcome.ok)
        self.assertEqual(runner.calls, ["a"])

    def test_retry_on_none_disables(self):
        runner = self._retry_runner(["transient", "ok"])
        mission = self.store.create_new("g")
        step = _step("a", retries=5, retry_on="none", retry_backoff_s=0.01)
        outcome = runner._execute_step(mission, step)
        self.assertFalse(outcome.ok)
        self.assertEqual(runner.calls, ["a"])

    def test_retry_on_any_retries_plain_failures(self):
        # "any" retries plain failures — but ValidationError stays
        # non-retryable (bad arguments fail identically every time).
        runner = self._retry_runner(["plain", "ok"])
        mission = self.store.create_new("g")
        step = _step("a", retries=1, retry_on="any", retry_backoff_s=0.01)
        outcome = runner._execute_step(mission, step)
        self.assertTrue(outcome.ok)
        self.assertEqual(runner.calls, ["a", "a"])

    def test_retry_on_any_still_skips_validation_errors(self):
        runner = self._retry_runner(["fatal", "ok"])
        mission = self.store.create_new("g")
        step = _step("a", retries=3, retry_on="any", retry_backoff_s=0.01)
        outcome = runner._execute_step(mission, step)
        self.assertFalse(outcome.ok)
        self.assertEqual(runner.calls, ["a"])

    def test_retry_budget_survives_in_state(self):
        runner = self._retry_runner(["transient", "transient", "transient"])
        mission = self.store.create_new("g")
        step = _step("a", retries=2, retry_backoff_s=0.01)
        outcome = runner._execute_step(mission, step)
        self.assertFalse(outcome.ok)
        reloaded = self.store.get(mission.id)
        # attempts were counted even though the row wasn't re-saved mid-run;
        # the settle phase persists them.
        self.assertEqual(mission.state["step_attempts"]["a"], 3)
        self.assertIsNotNone(reloaded)

    def test_retryable_classification(self):
        policy = {"retry_on": "transient"}
        self.assertTrue(_retryable("connection reset by peer", policy))
        self.assertTrue(_retryable("HTTP 429 too many requests", policy))
        self.assertTrue(_retryable("upstream 503", policy))
        self.assertTrue(_retryable("timed out waiting", policy))
        self.assertFalse(_retryable("ValidationError: bad args", policy))
        self.assertFalse(_retryable("plain boom", policy))
        self.assertTrue(_retryable("plain boom", {"retry_on": "any"}))
        self.assertFalse(_retryable("connection reset", {"retry_on": "none"}))

    def test_backoff_delay_bounds(self):
        policy = {"retry_backoff_s": 2.0}
        for attempt in (1, 2, 3):
            for _ in range(50):
                delay = _backoff_delay(policy, attempt)
                lo = 2.0 * (2.0 ** (attempt - 1)) * 0.5
                hi = 2.0 * (2.0 ** (attempt - 1)) * 1.5
                self.assertGreaterEqual(delay, lo)
                self.assertLessEqual(delay, hi)
        self.assertLessEqual(_backoff_delay({"retry_backoff_s": 1000.0}, 10),
                             300.0)

    def test_cancel_during_backoff(self):
        runner = self._retry_runner(["transient"] * 10)
        mission = self.store.create_new("g")
        step = _step("a", retries=9, retry_backoff_s=30.0)
        box = {}

        def _drive():
            box["outcome"] = runner._execute_step(mission, step)

        thread = threading.Thread(target=_drive, daemon=True)
        thread.start()
        time.sleep(0.4)  # first attempt fails, now sleeping in backoff
        runner.cancel("test cancel")
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertFalse(box["outcome"].ok)
        self.assertIn("cancel", box["outcome"].detail)
        self.assertEqual(mission.state["step_attempts"]["a"], 1)

    def test_policy_parsing_edge_cases(self):
        p = _step_policy(_step("a", retries="not-a-number"))
        self.assertEqual(p["retries"], 0)
        p = _step_policy(_step("a", retries=999))
        self.assertEqual(p["retries"], 10)  # capped
        p = _step_policy(_step("a", retry_on="bogus"))
        self.assertEqual(p["retry_on"], "transient")


# ── per-step timeout ─────────────────────────────────────────────────────────

class StepTimeoutTests(unittest.TestCase):
    """The timeout lives in the real ``_run_step_agent`` path, so these
    tests patch ``build_agent`` (not ``_run_step_agent``) to hang — the
    full policy plumbing is exercised."""

    def setUp(self):
        self.db = make_db()
        self.addCleanup(self.db.close)
        self.ctx = make_ctx(self.db)
        self.store = MissionStore(self.db)
        import nomorals.agents.roles as roles
        self._orig_build_agent = roles.build_agent
        self.addCleanup(setattr, roles, "build_agent", self._orig_build_agent)

    def _hang_then(self, script):
        """script: list of "hang" | "ok" per agent construction."""
        import nomorals.agents.roles as roles
        calls = []

        class _Agent:
            def __init__(self, mode):
                self.mode = mode

            def run(self, prompt):
                calls.append(self.mode)
                if self.mode == "hang":
                    time.sleep(30)  # abandoned daemon thread, not joined
                return SimpleNamespace(ok=True, output={"done": 1},
                                       error="", tokens=0)

        def fake_build_agent(role, name="", context=None):
            mode = script[min(len(calls), len(script) - 1)]
            return _Agent(mode)

        roles.build_agent = fake_build_agent
        return calls

    def test_hanging_step_times_out(self):
        self._hang_then(["hang"])
        runner = _runner(self.ctx, self.store)
        mission = self.store.create_new("g")
        outcome = runner._execute_step(mission, _step("a", timeout_s=0.2))
        self.assertFalse(outcome.ok)
        self.assertIn("timed out", outcome.detail)
        self.assertIn("daemon", outcome.detail)

    def test_timed_out_step_can_retry(self):
        calls = self._hang_then(["hang", "ok"])
        runner = _runner(self.ctx, self.store)
        mission = self.store.create_new("g")
        step = _step("a", timeout_s=0.2, retries=1, retry_backoff_s=0.01)
        outcome = runner._execute_step(mission, step)
        self.assertTrue(outcome.ok)
        self.assertEqual(calls, ["hang", "ok"])
        self.assertEqual(mission.state["step_attempts"]["a"], 2)

    def test_metadata_step_timeout_default(self):
        self._hang_then(["hang"])
        runner = _runner(self.ctx, self.store)
        mission = self.store.create_new("g",
                                        metadata={"step_timeout_s": 0.2})
        outcome = runner._execute_step(mission, _step("a"))
        self.assertFalse(outcome.ok)
        self.assertIn("timed out", outcome.detail)

    def test_fast_step_under_timeout_succeeds(self):
        self._hang_then(["ok"])
        runner = _runner(self.ctx, self.store)
        mission = self.store.create_new("g")
        outcome = runner._execute_step(mission, _step("a", timeout_s=5.0))
        self.assertTrue(outcome.ok)


# ── bounded intra-level parallelism ──────────────────────────────────────────

class ParallelLevelTests(unittest.TestCase):
    def setUp(self):
        self.db = make_db()
        self.addCleanup(self.db.close)
        self.ctx = make_ctx(self.db)
        self.store = MissionStore(self.db)

    def _overlap_runner(self, sleep_s=0.3):
        runner = _runner(self.ctx, self.store)
        state = {"active": 0, "max": 0}
        lock = threading.Lock()

        def fake_run(mission, step, started, policy=None):
            with lock:
                state["active"] += 1
                state["max"] = max(state["max"], state["active"])
            try:
                time.sleep(sleep_s)
            finally:
                with lock:
                    state["active"] -= 1
            return StepOutcome(step=step.name, ok=True, seconds=sleep_s)

        runner._run_step_agent = fake_run
        runner.overlap = state
        return runner

    def test_independent_steps_run_concurrently(self):
        runner = self._overlap_runner()
        mission = _mission_with_plan(self.store, [_step("a"), _step("b")],
                                     max_parallel=2)
        result = runner.run(mission, max_iterations=4, reflect=False)
        self.assertTrue(result.ok)
        self.assertEqual(runner.overlap["max"], 2)
        self.assertEqual(
            sorted(s.step for s in result.steps if s.ok), ["a", "b"])

    def test_default_is_sequential(self):
        runner = self._overlap_runner()
        mission = _mission_with_plan(self.store, [_step("a"), _step("b")])
        result = runner.run(mission, max_iterations=4, reflect=False)
        self.assertTrue(result.ok)
        self.assertEqual(runner.overlap["max"], 1)

    def test_parallel_settles_in_plan_order(self):
        runner = _runner(self.ctx, self.store)
        settled = []

        def fake_run(mission, step, started, policy=None):
            if step.name == "slow":
                time.sleep(0.4)  # finishes last...
            return StepOutcome(step=step.name, ok=True, seconds=0.1)

        runner._run_step_agent = fake_run
        runner.on_step = lambda mission, outcome: settled.append(outcome.step)
        mission = _mission_with_plan(self.store, [_step("slow"), _step("fast")],
                                     max_parallel=2)
        result = runner.run(mission, max_iterations=4, reflect=False)
        self.assertTrue(result.ok)
        self.assertEqual(settled, ["slow", "fast"])  # ...but settles first

    def test_parallel_respects_dependencies(self):
        runner = _runner(self.ctx, self.store)
        calls = []
        runner._run_step_agent = (
            lambda m, s, t, policy=None: calls.append(s.name)
            or StepOutcome(step=s.name, ok=True, seconds=0.1))
        mission = _mission_with_plan(
            self.store, [_step("a"), _step("b", deps=("a",))], max_parallel=2)
        result = runner.run(mission, max_iterations=4, reflect=False)
        self.assertTrue(result.ok)
        self.assertLess(calls.index("a"), calls.index("b"))

    def test_parallel_blocks_dependents_of_failed_step(self):
        runner = _runner(self.ctx, self.store)
        calls = []

        def fake_run(mission, step, started, policy=None):
            calls.append(step.name)
            if step.name == "a":
                return StepOutcome(step=step.name, ok=False,
                                   detail="a exploded", seconds=0.1)
            return StepOutcome(step=step.name, ok=True, seconds=0.1)

        runner._run_step_agent = fake_run
        mission = _mission_with_plan(
            self.store,
            [_step("a", on_failure="continue"), _step("b", deps=("a",))],
            max_parallel=2)
        result = runner.run(mission, max_iterations=4, reflect=False)
        self.assertFalse(result.ok)  # on_failure=continue still ends FAILED
        self.assertNotIn("b", calls)  # b never executed
        blocked = [s for s in result.steps if s.step == "b"][0]
        self.assertIn("blocked", blocked.detail)

    def test_parallel_fail_fast_stops_run(self):
        runner = _runner(self.ctx, self.store)
        calls = []

        def fake_run(mission, step, started, policy=None):
            calls.append(step.name)
            ok = step.name != "a"
            return StepOutcome(step=step.name, ok=ok,
                               detail="" if ok else "a exploded",
                               seconds=0.1)

        runner._run_step_agent = fake_run
        mission = _mission_with_plan(
            self.store,
            [_step("a"), _step("b"), _step("c", deps=("a", "b"))],
            max_parallel=2)
        # fail_fast with replanning disabled: the mission dies and the
        # dependent of the failed step never runs.
        mission.metadata["replan_policy"] = "off"
        self.store.save(mission)
        result = runner.run(mission, max_iterations=6, reflect=False)
        self.assertFalse(result.ok)
        self.assertNotIn("c", calls)  # dependent of failed "a" never ran

    def test_max_parallel_clamped(self):
        runner = _runner(self.ctx, self.store)
        mission = self.store.create_new("g", metadata={"max_parallel": 99})
        self.assertEqual(runner._max_parallel(mission), 8)
        mission2 = self.store.create_new("g2", metadata={"max_parallel": 0})
        self.assertEqual(runner._max_parallel(mission2), 1)


# ── idempotency by default + purge ───────────────────────────────────────────

class WiredIdempotencyTests(unittest.TestCase):
    def setUp(self):
        self.db = make_db()
        self.addCleanup(self.db.close)
        self.ctx = make_ctx(self.db)

    def test_wired_runner_attaches_idempotency_by_default(self):
        runner = wired_runner(self.ctx, milestones=False)
        self.assertIsInstance(runner.idempotency, IdempotencyStore)

    def test_wired_runner_explicit_none_opts_out(self):
        runner = wired_runner(self.ctx, milestones=False, idempotency=None)
        self.assertIsNone(runner.idempotency)

    def test_wired_runner_respects_explicit_store(self):
        store = IdempotencyStore(self.db)
        runner = wired_runner(self.ctx, milestones=False, idempotency=store)
        self.assertIs(runner.idempotency, store)


class IdempotencyPurgeTests(unittest.TestCase):
    def setUp(self):
        self.db = make_db()
        self.addCleanup(self.db.close)
        self.store = IdempotencyStore(self.db)

    def _seed(self, key, status, age_s):
        now = time.time()
        self.db.execute(
            "INSERT INTO idempotency_keys (key, status, result_json, error,"
            " owner, created_at, updated_at) VALUES (?, ?, '{}', '', '', ?, ?)",
            (key, status, now - age_s, now - age_s))

    def test_purge_deletes_old_settled_keeps_running(self):
        self._seed("old-done", COMPLETED, age_s=30 * 86400)
        self._seed("old-failed", FAILED, age_s=30 * 86400)
        self._seed("new-done", COMPLETED, age_s=60)
        self._seed("old-running", RUNNING, age_s=30 * 86400)
        deleted = self.store.purge(older_than_seconds=7 * 86400)
        self.assertEqual(deleted, 2)
        self.assertIsNone(self.store.get("old-done"))
        self.assertIsNone(self.store.get("old-failed"))
        self.assertIsNotNone(self.store.get("new-done"))
        # running keys are never purged — dropping one could duplicate work
        self.assertIsNotNone(self.store.get("old-running"))
        self.assertEqual(self.store.status("old-running"), RUNNING)


# ── trailing-window ETA ──────────────────────────────────────────────────────

class TrailingEtaTests(unittest.TestCase):
    def setUp(self):
        self.db = make_db()
        self.addCleanup(self.db.close)
        self.store = MissionStore(self.db)

    def _mission(self, durations, spent_wall=0.0):
        names = ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]
        mission = self.store.create_new("g")
        mission.state["plan"] = [{"name": n} for n in names]
        mission.state["completed_steps"] = names[:6]
        mission.state["step_durations"] = dict(durations)
        mission.spent_wall = spent_wall
        return mission

    def test_eta_uses_trailing_window_not_whole_run(self):
        # s1 was a 1000s outlier; the last five steps took 10s each.
        durations = [("s1", 1000.0), ("s2", 10.0), ("s3", 10.0),
                     ("s4", 10.0), ("s5", 10.0), ("s6", 10.0)]
        eta, note = estimate_eta(self._mission(durations))
        self.assertAlmostEqual(eta, 10.0)
        self.assertIn("last 5", note)

    def test_eta_falls_back_to_whole_run_average(self):
        mission = self._mission([], spent_wall=60.0)
        eta, _note = estimate_eta(mission)
        self.assertAlmostEqual(eta, 10.0)  # 60s / 6 done * 1 remaining

    def test_eta_none_without_any_timing(self):
        mission = self._mission([])
        eta, note = estimate_eta(mission)
        self.assertIsNone(eta)
        self.assertIn("no step timing", note)

    def test_runner_records_step_durations(self):
        db = self.db
        ctx = make_ctx(db)
        store = MissionStore(db)
        runner = _runner(ctx, store)
        runner._run_step_agent = (
            lambda m, s, t, policy=None: StepOutcome(step=s.name, ok=True,
                                                     seconds=0.25))
        mission = _mission_with_plan(store, [_step("a"), _step("b")])
        result = runner.run(mission, max_iterations=4, reflect=False)
        self.assertTrue(result.ok)
        reloaded = store.get(mission.id)
        durations = reloaded.state.get("step_durations") or {}
        self.assertEqual(set(durations), {"a", "b"})
        self.assertGreater(durations["a"], 0)


# ── detail() surfaces attempts ───────────────────────────────────────────────

class DetailAttemptsTests(unittest.TestCase):
    def test_detail_includes_step_attempts(self):
        db = make_db()
        self.addCleanup(db.close)
        ctx = make_ctx(db)
        store = MissionStore(db)
        runner = _runner(ctx, store)
        calls = []

        def fake_run(mission, step, started, policy=None):
            calls.append(1)
            if len(calls) == 1:
                return StepOutcome(step=step.name, ok=False,
                                   detail="connection reset", seconds=0.1)
            return StepOutcome(step=step.name, ok=True, seconds=0.1)

        runner._run_step_agent = fake_run
        mission = _mission_with_plan(store, [_step("a", retries=1,
                                                   retry_backoff_s=0.01)])
        result = runner.run(mission, max_iterations=4, reflect=False)
        self.assertTrue(result.ok)
        detail = store.detail(mission.id)
        self.assertEqual(detail["attempts"], {"a": 2})


# ── resume_all robustness ────────────────────────────────────────────────────

class ResumeAllRobustnessTests(unittest.TestCase):
    def test_one_bad_row_does_not_abort_the_batch(self):
        db = make_db()
        self.addCleanup(db.close)
        ctx = make_ctx(db)
        store = MissionStore(db)
        runner = _runner(ctx, store)
        runner._run_step_agent = (
            lambda m, s, t, policy=None: StepOutcome(step=s.name, ok=True,
                                                     seconds=0.1))
        m1 = _mission_with_plan(store, [_step("a")], name="m1")
        m2 = _mission_with_plan(store, [_step("b")], name="m2")
        store.set_status(m1.id, MissionStatus.RUNNING)
        store.set_status(m2.id, MissionStatus.RUNNING)

        calls = []
        orig_run = runner.run

        def flaky_run(mission, **kw):
            calls.append(mission.id)
            if mission.id == m1.id:
                raise RuntimeError("unexpected storage corruption")
            return orig_run(mission, **kw)

        runner.run = flaky_run
        results = runner.resume_all(max_iterations=4)
        self.assertEqual(len(calls), 2)  # both attempted
        self.assertEqual(len(results), 1)  # the good one completed
        self.assertEqual(results[0].mission_id, m2.id)


# ── golden attempt counts ────────────────────────────────────────────────────

class GoldenAttemptsTests(unittest.TestCase):
    def test_golden_step_report_counts_repair_as_attempt(self):
        db = make_db()
        self.addCleanup(db.close)
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        runner = GoldenRunner(db, workdir_root=tmp)
        mission = runner.store.create_new("g")

        def verify(output):
            if output.get("fixed"):
                return True, "ok"
            return False, "not fixed yet"

        step = GoldenStep("s", run=lambda ctx: {"x": 1}, verify=verify,
                          repair=lambda o, ctx: {"fixed": True})
        outputs = {}
        report = runner._execute_step(
            mission, step, GoldenContext(workdir=Path(tmp)), outputs)
        self.assertTrue(report["ok"])
        self.assertTrue(report["repaired"])
        self.assertEqual(report["attempts"], 2)

    def test_golden_clean_step_reports_single_attempt(self):
        db = make_db()
        self.addCleanup(db.close)
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        runner = GoldenRunner(db, workdir_root=tmp)
        mission = runner.store.create_new("g")
        step = GoldenStep("s", run=lambda ctx: {"x": 1},
                          verify=lambda o: (True, "ok"))
        report = runner._execute_step(
            mission, step, GoldenContext(workdir=Path(tmp)), {})
        self.assertTrue(report["ok"])
        self.assertEqual(report["attempts"], 1)


if __name__ == "__main__":
    unittest.main()
