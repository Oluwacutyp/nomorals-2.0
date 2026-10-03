"""Wave G3 — missions reliability.

Covers the three reliability gaps fixed in this wave:

1. **Accurate status** — ``status`` / ``list`` / ``watch`` reconcile on
   read: a mission whose runner died (stale heartbeat + gone process)
   flips from "running" to the true terminal state with a reason instead
   of reporting "running" forever.
2. **Idempotent verbs** — pause / resume / cancel are clean no-ops when
   repeated (double-pause, resume-of-running, cancel-of-finished); retry
   of a finished mission explains that it creates a NEW attempt linked
   via ``metadata.retry_of`` and never rewrites the original's record.
3. **Stall reasons** — every stall code renders with code + human reason
   + what would unblock it; no stall path can produce a bare "stalled".
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.errors import ValidationError
from nomorals.missions import (
    MissionStatus,
    MissionStore,
    StallCode,
    StepOutcome,
    render_status_text,
)
from nomorals.missions.progress import STALL_AFTER_FAILURES
from nomorals.missions.runner import MissionRunner as RealRunner
from nomorals.storage.db import Database


# ── fixtures ─────────────────────────────────────────────────────────────────

def make_ctx(test=None):
    tmp = tempfile.mkdtemp(prefix="missions-g3-")
    if test is not None:
        test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()
    return SimpleNamespace(db=db)


def make_mission(store, name="m1", goal="test goal", plan_names=("a", "b", "c")):
    mission = store.create_new(goal, name=name)
    mission.state["plan"] = [
        {"name": n, "goal": f"do {n}", "role": "execution",
         "kind": "io", "depends_on": []}
        for n in plan_names
    ]
    store.save(mission)
    return mission


def make_dead_running(store, name="dead", age_s=3600.0, pid=999_999_999):
    """A mission whose runner is gone: status running, heartbeat stale,
    pid that cannot exist."""
    mission = make_mission(store, name=name)
    mission.status = MissionStatus.RUNNING
    mission.state["heartbeat"] = {"pid": pid, "at": time.time() - age_s}
    return store.save(mission)


def make_live_running(store, name="live"):
    """A mission with a genuinely live runner (this process)."""
    mission = make_mission(store, name=name)
    mission.status = MissionStatus.RUNNING
    mission.state["heartbeat"] = {"pid": os.getpid(), "at": time.time()}
    return store.save(mission)


def control(ctx, tail, chat_key="local:console"):
    """Drive the real ``/mission`` chat handler without a full runtime.

    ``_control_mission`` only touches ``self.context.db``,
    ``self._find_mission`` and ``self._mission_template_spec``, so a
    lightweight stand-in is faithful for these verbs.
    """
    ctrl = SimpleNamespace(
        context=ctx,
        _find_mission=PartnerRuntime._find_mission,
        _mission_template_spec=PartnerRuntime._mission_template_spec,
    )
    return PartnerRuntime._control_mission(ctrl, tail, chat_key)


class RunnerStub(RealRunner):
    """Records run/resume/cancel instead of executing (the real ones would
    plan and run agents — slow and model-dependent)."""

    instances: list = []

    def __init__(self, context, **kw):
        kw.pop("milestones", None)
        super().__init__(context, store=kw.get("store"), milestones=False)
        self.calls: list = []
        RunnerStub.instances.append(self)

    def run(self, mission, **kw):
        self.calls.append(("run", getattr(mission, "id", mission)))
        return None

    def resume(self, mission_id, **kw):
        self.calls.append(("resume", mission_id))
        return None

    def cancel(self, reason="cancelled"):
        self.calls.append(("cancel", reason))


def _wait(predicate, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _stubbed():
    return patch("nomorals.missions.wired_runner", RunnerStub)


# ── 1. reconcile ─────────────────────────────────────────────────────────────

class ReconcileTests(unittest.TestCase):
    def setUp(self):
        self.ctx = make_ctx(self)
        self.store = MissionStore(self.ctx.db)

    def test_dead_runner_flips_to_failed_with_reason(self):
        mission = make_dead_running(self.store)
        rec = self.store.reconcile(mission.id)
        self.assertTrue(rec["changed"])
        self.assertEqual(rec["status"], MissionStatus.FAILED)
        self.assertIn("runner died", rec["reason"])
        again = self.store.get(mission.id)
        self.assertEqual(again.status, MissionStatus.FAILED)
        self.assertTrue(again.terminal)
        self.assertIsNotNone(again.finished_at)
        self.assertIn("runner died", again.state.get("last_error", ""))

    def test_missing_heartbeat_ever_also_flips(self):
        # a "running" row from before heartbeats existed, or a crashed
        # writer: no heartbeat key at all -> still detected as dead
        mission = make_mission(self.store, name="ancient")
        mission.status = MissionStatus.RUNNING
        self.store.save(mission)
        rec = self.store.reconcile(mission.id)
        self.assertTrue(rec["changed"])
        self.assertEqual(rec["status"], MissionStatus.FAILED)
        self.assertIn("no heartbeat ever recorded", rec["reason"])

    def test_live_process_stays_running_despite_stale_heartbeat(self):
        # a stale heartbeat with a LIVE process is a long step, not death
        mission = make_mission(self.store, name="longstep")
        mission.status = MissionStatus.RUNNING
        mission.state["heartbeat"] = {"pid": os.getpid(),
                                      "at": time.time() - 7200.0}
        self.store.save(mission)
        rec = self.store.reconcile(mission.id)
        self.assertFalse(rec["changed"])
        self.assertEqual(self.store.get(mission.id).status, MissionStatus.RUNNING)

    def test_fresh_heartbeat_stays_running(self):
        mission = make_live_running(self.store)
        rec = self.store.reconcile(mission.id)
        self.assertFalse(rec["changed"])
        self.assertTrue(rec["process_alive"])
        self.assertEqual(self.store.get(mission.id).status, MissionStatus.RUNNING)

    def test_non_running_missions_are_never_touched(self):
        for status in (MissionStatus.PENDING, MissionStatus.PAUSED,
                       MissionStatus.DONE, MissionStatus.FAILED,
                       MissionStatus.CANCELLED):
            mission = make_mission(self.store, name=f"s-{status}")
            self.store.set_status(mission.id, status)
            rec = self.store.reconcile(mission.id)
            self.assertFalse(rec["changed"], status)
            self.assertEqual(rec["status"], status)
            self.assertEqual(self.store.get(mission.id).status, status)

    def test_reconcile_honors_explicit_stale_after(self):
        mission = make_mission(self.store, name="threshold")
        mission.status = MissionStatus.RUNNING
        mission.state["heartbeat"] = {"pid": 999_999_999,
                                      "at": time.time() - 100.0}
        self.store.save(mission)
        # 100s old beats a 60s limit -> dead
        rec = self.store.reconcile(mission.id, stale_after=60.0)
        self.assertTrue(rec["changed"])
        # restore and check the default 15-minute limit keeps it running
        mission = self.store.get(mission.id)
        mission.status = MissionStatus.RUNNING
        mission.state["heartbeat"] = {"pid": 999_999_999,
                                      "at": time.time() - 100.0}
        self.store.save(mission)
        rec = self.store.reconcile(mission.id)
        self.assertFalse(rec["changed"])

    def test_runner_writes_heartbeat_on_start_and_steps(self):
        mission = make_mission(self.store, name="hb", plan_names=("a", "b"))
        runner = RealRunner(self.ctx, store=self.store, milestones=False)
        runner._execute_step = lambda m, step: StepOutcome(
            step=step.name, ok=True, seconds=1.0)  # noqa: E731 - test double
        result = runner.run(mission)
        self.assertTrue(result.ok)
        hb = self.store.get(mission.id).state.get("heartbeat") or {}
        self.assertEqual(hb.get("pid"), os.getpid())
        self.assertLess(time.time() - hb.get("at", 0), 60.0)

    def test_status_command_reconciles_dead_running(self):
        mission = make_dead_running(self.store, name="statushb")
        reply = control(self.ctx, f"status {mission.id}")
        self.assertIn("reconciled", reply)
        self.assertIn("[failed]", reply)
        self.assertNotIn("[running]", reply)
        self.assertEqual(self.store.get(mission.id).status, MissionStatus.FAILED)

    def test_list_never_reports_dead_as_running(self):
        make_dead_running(self.store, name="listhb")
        reply = control(self.ctx, "list")
        self.assertNotIn("[running]", reply)
        self.assertIn("[failed]", reply)

    def test_watch_reports_reconciled_state(self):
        mission = make_dead_running(self.store, name="watchhb")
        reply = control(self.ctx, f"watch {mission.id}")
        self.assertIn("currently failed", reply)


# ── 2. idempotency ────────────────────────────────────────────────────────────

class IdempotencyTests(unittest.TestCase):
    def setUp(self):
        self.ctx = make_ctx(self)
        self.store = MissionStore(self.ctx.db)
        RunnerStub.instances.clear()

    def test_double_pause_is_noop_success(self):
        mission = make_mission(self.store, name="pause1")
        first = control(self.ctx, f"pause {mission.id}")
        self.assertIn("paused", first)
        second = control(self.ctx, f"pause {mission.id}")
        self.assertIn("already paused", second)
        self.assertEqual(self.store.get(mission.id).status, MissionStatus.PAUSED)

    def test_pause_of_finished_is_informative_not_error(self):
        mission = make_mission(self.store, name="pause2")
        self.store.set_status(mission.id, MissionStatus.DONE)
        reply = control(self.ctx, f"pause {mission.id}")
        self.assertIn("already done", reply)
        self.assertEqual(self.store.get(mission.id).status, MissionStatus.DONE)

    def test_resume_of_running_is_noop(self):
        mission = make_live_running(self.store, name="resume1")
        with _stubbed():
            reply = control(self.ctx, f"resume {mission.id}")
        self.assertIn("already running", reply)
        # no runner was even constructed: nothing to restart
        self.assertEqual(RunnerStub.instances, [])
        self.assertEqual(self.store.get(mission.id).status, MissionStatus.RUNNING)

    def test_resume_of_dead_running_restarts_instead_of_noop(self):
        mission = make_dead_running(self.store, name="resume2")
        with _stubbed():
            reply = control(self.ctx, f"resume {mission.id}")
        # reconcile flipped it to failed -> terminal message, suggests retry
        self.assertIn("fresh attempt", reply)
        self.assertEqual(self.store.get(mission.id).status, MissionStatus.FAILED)

    def test_resume_of_paused_starts_background_job(self):
        mission = make_mission(self.store, name="resume3")
        self.store.set_status(mission.id, MissionStatus.PAUSED)
        with _stubbed():
            reply = control(self.ctx, f"resume {mission.id}")
        self.assertIn("resumed in the background", reply)
        self.assertTrue(_wait(lambda: any(
            c == ("resume", mission.id)
            for inst in RunnerStub.instances for c in inst.calls)))

    def test_double_cancel_is_noop_success(self):
        mission = make_live_running(self.store, name="cancel1")
        with _stubbed():
            first = control(self.ctx, f"cancel {mission.id}")
            self.assertIn("cancelled", first)
            self.assertEqual(self.store.get(mission.id).status,
                             MissionStatus.CANCELLED)
            second = control(self.ctx, f"cancel {mission.id}")
        self.assertIn("already cancelled", second)
        self.assertEqual(self.store.get(mission.id).status,
                         MissionStatus.CANCELLED)

    def test_cancel_of_done_keeps_done(self):
        # the bug this guards: cancel used to rewrite done -> cancelled,
        # destroying the finished record
        mission = make_mission(self.store, name="cancel2")
        self.store.set_status(mission.id, MissionStatus.DONE)
        with _stubbed():
            reply = control(self.ctx, f"cancel {mission.id}")
        self.assertIn("already done", reply)
        self.assertEqual(self.store.get(mission.id).status, MissionStatus.DONE)

    def test_cancel_of_failed_keeps_failed(self):
        mission = make_mission(self.store, name="cancel3")
        self.store.set_status(mission.id, MissionStatus.FAILED)
        with _stubbed():
            reply = control(self.ctx, f"cancel {mission.id} oops")
        self.assertIn("already failed", reply)
        self.assertEqual(self.store.get(mission.id).status, MissionStatus.FAILED)

    def test_retry_of_finished_creates_linked_new_attempt(self):
        mission = make_mission(self.store, name="retry1")
        self.store.set_status(mission.id, MissionStatus.FAILED)
        with _stubbed():
            reply = control(self.ctx, f"retry {mission.id}")
        self.assertIn("NEW attempt", reply)
        self.assertIn("retry_of", reply)
        news = [m for m in self.store.list(limit=50)
                if m.metadata.get("retry_of") == mission.id]
        self.assertEqual(len(news), 1)
        new = news[0]
        self.assertNotEqual(new.id, mission.id)
        self.assertEqual(new.goal, mission.goal)
        self.assertEqual(new.status, MissionStatus.PENDING)
        # the original's terminal record is preserved, not rewritten
        self.assertEqual(self.store.get(mission.id).status, MissionStatus.FAILED)
        # and the new attempt actually started in the background
        self.assertTrue(_wait(lambda: any(
            c == ("run", new.id)
            for inst in RunnerStub.instances for c in inst.calls)))

    def test_retry_of_finished_twice_is_two_clear_attempts(self):
        mission = make_mission(self.store, name="retry2")
        self.store.set_status(mission.id, MissionStatus.DONE)
        with _stubbed():
            first = control(self.ctx, f"retry {mission.id}")
            second = control(self.ctx, f"retry {mission.id}")
        for reply in (first, second):
            self.assertIn("NEW attempt", reply)
            self.assertIn("retry_of", reply)
        news = [m for m in self.store.list(limit=50)
                if m.metadata.get("retry_of") == mission.id]
        self.assertEqual(len(news), 2)
        # original untouched by either retry
        self.assertEqual(self.store.get(mission.id).status, MissionStatus.DONE)

    def test_retry_of_running_refuses(self):
        mission = make_live_running(self.store, name="retry3")
        before = len(self.store.list(limit=50))
        with _stubbed():
            reply = control(self.ctx, f"retry {mission.id}")
        self.assertIn("still running", reply)
        self.assertEqual(len(self.store.list(limit=50)), before)

    def test_retry_of_paused_supersedes_old_attempt(self):
        mission = make_mission(self.store, name="retry4")
        self.store.set_status(mission.id, MissionStatus.PAUSED)
        with _stubbed():
            reply = control(self.ctx, f"retry {mission.id}")
        self.assertIn("NEW attempt", reply)
        self.assertIn("previous attempt was cancelled", reply)
        self.assertEqual(self.store.get(mission.id).status,
                         MissionStatus.CANCELLED)


# ── 3. stall reasons ─────────────────────────────────────────────────────────

class StallReasonTests(unittest.TestCase):
    def setUp(self):
        self.ctx = make_ctx(self)
        self.store = MissionStore(self.ctx.db)
        RunnerStub.instances.clear()

    def test_every_code_renders_code_reason_and_unblock(self):
        for code in sorted(StallCode.ALL):
            mission = make_mission(self.store, name=f"stall-{code}")
            self.store.mark_stalled(mission.id, code,
                                    f"test blocker for {code}", step="a")
            text = render_status_text(self.store.detail(mission.id))
            self.assertIn(code, text, code)
            self.assertIn(StallCode.LABELS[code], text, code)
            self.assertIn(f"test blocker for {code}", text, code)
            self.assertIn(StallCode.UNBLOCK_HINTS[code], text, code)

    def test_stall_command_reply_has_code_reason_unblock(self):
        mission = make_mission(self.store, name="stallcmd")
        with _stubbed():
            reply = control(
                self.ctx,
                f"stall {mission.id} blocked_on_approval "
                "need your go-ahead on the deploy")
        self.assertIn("blocked_on_approval", reply)
        self.assertIn("blocked on approval", reply)
        self.assertIn("need your go-ahead on the deploy", reply)
        self.assertIn("unblocks", reply)

    def test_status_command_shows_full_stall_reason(self):
        mission = make_mission(self.store, name="stallstatus")
        self.store.mark_stalled(mission.id, StallCode.DEPENDENCY_MISSING,
                                "ffmpeg is not installed", step="render")
        reply = control(self.ctx, f"status {mission.id}")
        self.assertIn("dependency_missing", reply)
        self.assertIn("dependency missing", reply)
        self.assertIn("ffmpeg is not installed", reply)
        self.assertIn("unblocks", reply)

    def test_stall_rejects_unknown_code_and_empty_message(self):
        mission = make_mission(self.store, name="stallbad")
        with self.assertRaises(ValidationError):
            self.store.mark_stalled(mission.id, "just_waiting", "soon")
        with self.assertRaises(ValidationError):
            self.store.mark_stalled(mission.id,
                                    StallCode.WAITING_ON_PROVIDER, "   ")
        # a stall record can never exist without code + message
        self.assertIsNone(self.store.stall(mission.id))

    def test_runner_consecutive_failure_stall_has_reason(self):
        mission = make_mission(self.store, name="stallauto")
        runner = RealRunner(self.ctx, store=self.store, milestones=False)
        for _ in range(STALL_AFTER_FAILURES):
            runner._apply_step_result(
                mission, StepOutcome(step="a", ok=False, detail="boom"), "a")
        self.store.save(mission)
        stall = mission.state.get("stall")
        self.assertEqual(stall["code"], StallCode.RETRY_BUDGET_EXHAUSTED)
        text = render_status_text(self.store.detail(mission.id))
        self.assertIn("retry_budget_exhausted", text)
        self.assertIn("retry budget exhausted", text)
        self.assertIn(stall["message"], text)
        self.assertIn(StallCode.UNBLOCK_HINTS[StallCode.RETRY_BUDGET_EXHAUSTED],
                      text)

    def test_runner_budget_stall_has_reason(self):
        mission = make_mission(self.store, name="stallbudget")
        mission.budget_wall = 60.0
        mission.spent_wall = 60.0
        runner = RealRunner(self.ctx, store=self.store, milestones=False)
        runner._record_budget_stall(mission)
        stall = self.store.get(mission.id).state.get("stall")
        self.assertEqual(stall["code"], StallCode.BUDGET_EXHAUSTED)
        text = render_status_text(self.store.detail(mission.id))
        self.assertIn("budget_exhausted", text)
        self.assertIn("budget exhausted", text)
        self.assertIn(StallCode.UNBLOCK_HINTS[StallCode.BUDGET_EXHAUSTED], text)

    def test_list_shows_stall_code_and_human_label(self):
        mission = make_live_running(self.store, name="stalllist")
        self.store.mark_stalled(mission.id, StallCode.WAITING_ON_PROVIDER,
                                "groq is rate-limiting us")
        reply = control(self.ctx, "list")
        self.assertIn("waiting_on_provider", reply)
        self.assertIn("waiting on provider", reply)


if __name__ == "__main__":
    unittest.main()
