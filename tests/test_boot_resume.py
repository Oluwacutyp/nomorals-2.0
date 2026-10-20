"""Boot-time mission resumption (post-pass-#54 fix).

The defect: ``PartnerRuntime.start`` never resumed interrupted
missions, so a mission whose driver died (restart, crash, self-update)
sat RUNNING forever until some surface touched it — and the first
``/mission status`` read then reconcile-flipped it to FAILED.

These tests simulate the real sequence: process A starts a mission and
dies mid-step (fault injection at the runner's ``_execute_step`` seam —
the pass #54 pattern — leaving the exact row a kill -9 leaves:
RUNNING, stale heartbeat from a dead pid, a stranded run claim), then
process B boots through the REAL ``PartnerRuntime.start`` path (or the
exact boot function it calls) and the mission completes with completed
steps NOT re-executed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from nomorals.agents.context import build_context
from nomorals.agents.orchestrator import PlanStep
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.core.tasks import TaskKind
from nomorals.llm.base import LLMResponse, Message, SamplingParams
from nomorals.missions import MissionStatus, MissionStore
from nomorals.missions.runner import MissionRunner, StepOutcome, _serialize_plan
from nomorals.social.chat.base import ChatAdapter, ChatRef, SendResult
from nomorals.social.chat.gateway import ChatGateway

DEAD_PID = 99999999  # certainly not a live process (pass #54 pattern)


# ── harness (mirrors tests/test_partner_runtime.py) ──────────────────────


class FakeRouter:
    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    def chat(self, messages: list[Message], params: SamplingParams | None = None,
             **kw: Any) -> LLMResponse:
        self.calls.append(list(messages))
        return LLMResponse(text="mhm.", model="fake-7b")


class FakeAdapter(ChatAdapter):
    def __init__(self, name: str = "local") -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self.sent: list[str] = []
        self._handler = None
        self.started_flag = False

    def run(self, handler) -> None:
        self._handler = handler
        self.started_flag = True
        while not getattr(self, "stopped", False):
            time.sleep(0.01)

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             buttons: list[list[tuple[str, str]]] | None = None,
             parse_mode: str = "") -> SendResult:
        self.sent.append(text)
        return SendResult(ok=True, platform=self.name,
                          message_id=f"m{len(self.sent)}")

    def wait_started(self, timeout: float = 5.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if self.started_flag:
                return True
            time.sleep(0.01)
        return False


def _make_context() -> tuple[Any, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory(prefix="nm-boot-")
    settings = load_settings(
        overrides={
            "home": tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
            # Autonomy OFF: start() must not fire catch-up briefings
            # into the adapter; these tests measure the boot path.
            "partner.enabled": "false",
        }
    )
    context = build_context(settings, with_executor=False, with_tools=False)
    context.router = FakeRouter()
    ws = Path(context.settings.workspace_dir)
    for marker in ("pulse_last_run.json", "pulse_last_run_afternoon.json",
                   "pulse_last_run_evening.json",
                   "pulse_last_run_midnight.json"):
        (ws / marker).write_text(
            json.dumps({"ts": time.time(), "delivered_text": True}),
            encoding="utf-8")
    from nomorals.agents.morning_briefing import _today_str
    context.db.execute(
        "INSERT OR REPLACE INTO briefings (id, date, sections_json, generated_at)"
        " VALUES (?, ?, '[]', ?)",
        ("test-seed", _today_str(context), time.time()),
    )
    return context, tmp


def _wait(predicate, timeout: float = 30.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return False


# ── mission fixtures ─────────────────────────────────────────────────────


def _step(name: str, deps: tuple[str, ...] = ()) -> PlanStep:
    return PlanStep(name=name, goal=f"do {name}", role="execution",
                    kind=TaskKind.IO, depends_on=list(deps), payload={})


def _mission_with_plan(store: MissionStore, steps, name: str = "m",
                       goal: str = "test goal"):
    mission = store.create_new(goal, name=name)
    mission.state["plan"] = _serialize_plan(SimpleNamespace(steps=steps))
    store.save(mission)
    return mission


def _exec_patch(executed: list[str], crash_on: str | None = None):
    """The pass #54 seam: replace step execution with a recorder.

    ``crash_on`` raises KeyboardInterrupt (a BaseException the runner
    cannot settle into an outcome) — process A dying mid-step.
    """
    def _exec(self, mission, step):
        if crash_on is not None and step.name == crash_on:
            raise KeyboardInterrupt("simulated kill mid-step")
        executed.append(step.name)
        return StepOutcome(step=step.name, ok=True, seconds=0.01,
                           payload={"done": step.name})
    return patch.object(MissionRunner, "_execute_step", _exec)


def _crashed_mission(ctx, store: MissionStore, name: str = "crashed",
                     goal: str = "finish the thing"):
    """Process A: start a 3-step mission, die during step b.

    Returns (mission_id, executed_by_a). The persisted row is then
    rewritten to the exact shape a kill -9 leaves behind: RUNNING,
    heartbeat + run claim held by a dead pid, long stale.
    """
    steps = [_step("a"), _step("b", ("a",)), _step("c", ("b",))]
    mission = _mission_with_plan(store, steps, name=name, goal=goal)
    executed: list[str] = []
    runner = MissionRunner(ctx, store=store, milestones=False,
                           orchestrator_factory=lambda: MagicMock())
    with _exec_patch(executed, crash_on="b"):
        try:
            runner.run(store.get(mission.id), reflect=False)
        except KeyboardInterrupt:
            pass
    assert executed == ["a"], executed
    row = store.get(mission.id)
    assert row.status == MissionStatus.RUNNING, row.status
    assert row.state.get("completed_steps") == ["a"], row.state
    stale = time.time() - 100000
    row.state["heartbeat"] = {"pid": DEAD_PID, "at": stale}
    row.state["run_claim"] = {"token": "ghost", "pid": DEAD_PID, "at": stale}
    store.save(row)
    return mission.id, executed


def _missions_notices(context) -> list[dict[str, Any]]:
    from nomorals.agents.notifier import Notifier
    return Notifier(context).recent(20, kind="missions")


class _BootBase(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()
        self.store = MissionStore(self.context.db)
        self.runtime: PartnerRuntime | None = None

    def tearDown(self) -> None:
        if self.runtime is not None:
            try:
                self.runtime.stop()
            except Exception:  # noqa: BLE001 - teardown must not fail
                pass
        try:
            self.context.close()
        except Exception:  # noqa: BLE001
            pass
        self.tmp.cleanup()

    def _runtime(self, **kw) -> PartnerRuntime:
        adapter = FakeAdapter("local")
        gateway = ChatGateway({"local": adapter}, db=self.context.db)
        self.adapter = adapter
        self.runtime = PartnerRuntime(self.context, gateway=gateway, **kw)
        return self.runtime


# ── the real start() path ────────────────────────────────────────────────


class BootStartResumeTest(_BootBase):
    def test_start_resumes_crashed_mission_and_catches_up_terminal(self):
        crashed_id, _a = _crashed_mission(self.context, self.store,
                                          name="crashed", goal="ship it")
        # A PAUSED mission (explicit owner hold) with work left.
        paused = _mission_with_plan(self.store, [_step("x")], name="held")
        row = self.store.get(paused.id)
        row.status = MissionStatus.PAUSED
        self.store.save(row)
        # A mission that finished while down, terminal push never marked
        # as delivered → boot must re-deliver it exactly once.
        lost = _mission_with_plan(self.store, [_step("z")], name="lostpush")
        row = self.store.get(lost.id)
        row.status = MissionStatus.DONE
        row.state["completed_steps"] = ["z"]
        self.store.save(row)
        # A finished mission whose push WAS delivered → stays silent.
        sent = _mission_with_plan(self.store, [_step("z")], name="sentpush")
        row = self.store.get(sent.id)
        row.status = MissionStatus.DONE
        row.state["completed_steps"] = ["z"]
        row.state["milestones"] = [
            {"event": "terminal:done", "title": "t", "at": time.time()}]
        self.store.save(row)

        executed: list[str] = []
        runtime = self._runtime()
        with _exec_patch(executed):
            runtime.start()
            self.assertIsNotNone(
                getattr(runtime, "_boot_resume_thread", None),
                "start() did not kick off boot mission resumption")
            self.assertTrue(
                _wait(lambda: self.store.get(crashed_id).status
                      == MissionStatus.DONE),
                "crashed mission was not resumed to completion at boot")
            self.assertTrue(
                _wait(lambda: any(
                    e.get("event") == "terminal:done"
                    for e in (self.store.get(lost.id).state.get("milestones")
                              or []))),
                "lost terminal push was not caught up at boot")
        # Completed step NOT re-executed; the rest ran exactly once.
        self.assertEqual(executed, ["b", "c"])
        # PAUSED stayed PAUSED and was never driven.
        self.assertEqual(self.store.get(paused.id).status,
                         MissionStatus.PAUSED)
        # Exactly one owner summary for the boot resume.
        notices = _missions_notices(self.context)
        self.assertEqual(len(notices), 1, notices)
        self.assertIn("crashed", notices[0]["title"] + notices[0]["body"])

        # A second sweep is a no-op: nothing re-driven, nothing re-sent.
        executed.clear()
        summary = runtime._resume_interrupted_missions()
        self.assertEqual(summary["resumed"], [])
        self.assertEqual(summary["failed"], [])
        self.assertEqual(summary["terminal_caught_up"], [])
        self.assertEqual(executed, [])
        self.assertEqual(len(_missions_notices(self.context)), 1)

    def test_dry_run_start_never_resumes(self):
        crashed_id, _a = _crashed_mission(self.context, self.store)
        executed: list[str] = []
        runtime = self._runtime(dry_run=True)
        with _exec_patch(executed):
            runtime.start()
            time.sleep(0.5)
        self.assertIsNone(getattr(runtime, "_boot_resume_thread", None))
        self.assertEqual(self.store.get(crashed_id).status,
                         MissionStatus.RUNNING)
        self.assertEqual(executed, [])
        self.assertEqual(_missions_notices(self.context), [])


# ── the boot sweep itself (the exact function start() calls) ─────────────


class BootSweepTest(_BootBase):
    def test_live_drivers_are_refused_by_the_claim(self):
        # Same-process live driver: fresh claim, our pid.
        mine = _mission_with_plan(self.store, [_step("a")], name="mine")
        row = self.store.get(mine.id)
        row.status = MissionStatus.RUNNING
        row.state["run_claim"] = {"token": "live", "pid": os.getpid(),
                                  "at": time.time()}
        row.state["heartbeat"] = {"pid": os.getpid(), "at": time.time()}
        self.store.save(row)
        # Foreign live driver: a real sleeping process holds the claim.
        sleeper = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            foreign = _mission_with_plan(self.store, [_step("a")],
                                         name="foreign")
            row = self.store.get(foreign.id)
            row.status = MissionStatus.RUNNING
            row.state["run_claim"] = {"token": "live", "pid": sleeper.pid,
                                      "at": time.time()}
            row.state["heartbeat"] = {"pid": sleeper.pid,
                                      "at": time.time()}
            self.store.save(row)
            runtime = self._runtime()
            executed: list[str] = []
            with _exec_patch(executed):
                summary = runtime._resume_interrupted_missions()
            self.assertEqual(executed, [])
            self.assertEqual(summary["resumed"], [])
            self.assertEqual(summary["failed"], [])
            self.assertEqual(
                sorted(summary["skipped_live"]),
                sorted([mine.id, foreign.id]))
            self.assertEqual(self.store.get(mine.id).status,
                             MissionStatus.RUNNING)
            self.assertEqual(self.store.get(foreign.id).status,
                             MissionStatus.RUNNING)
            self.assertEqual(_missions_notices(self.context), [])
        finally:
            sleeper.kill()
            sleeper.wait()

    def test_paused_stays_paused_and_reconciled_failure_not_revived(self):
        paused = _mission_with_plan(self.store, [_step("x")], name="held")
        row = self.store.get(paused.id)
        row.status = MissionStatus.PAUSED
        self.store.save(row)
        # The owner's plausibly-already-flipped mission: a /mission
        # status read reconcile-flipped it to FAILED with the crash
        # marker. Boot resume must NOT guess-heal it — /mission replan
        # is the deliberate revival path.
        flipped = _mission_with_plan(self.store, [_step("y")],
                                     name="flipped")
        row = self.store.get(flipped.id)
        row.status = MissionStatus.FAILED
        row.state["status_note"] = (
            "reconciled: runner died — no heartbeat for 3600s "
            f"(pid {DEAD_PID} is gone)")
        row.state["last_error"] = row.state["status_note"].split(": ", 1)[1]
        self.store.save(row)

        runtime = self._runtime()
        executed: list[str] = []
        with _exec_patch(executed):
            summary = runtime._resume_interrupted_missions()
        self.assertEqual(executed, [])
        self.assertEqual(summary["resumed"], [])
        self.assertEqual(summary["held_paused"], 1)
        self.assertEqual(self.store.get(paused.id).status,
                         MissionStatus.PAUSED)
        self.assertEqual(self.store.get(flipped.id).status,
                         MissionStatus.FAILED)
        self.assertEqual(_missions_notices(self.context), [])

    def test_per_mission_failure_isolated_and_reported(self):
        bad_id, _ = _crashed_mission(self.context, self.store, name="bad",
                                     goal="bad goal")
        good_id, _ = _crashed_mission(self.context, self.store,
                                      name="good", goal="good goal")
        runtime = self._runtime()
        real_run = MissionRunner.run

        def flaky_run(self_runner, mission, **kw):
            if mission.id == bad_id:
                raise RuntimeError("drive exploded")
            return real_run(self_runner, mission, **kw)

        executed: list[str] = []
        with _exec_patch(executed):
            with patch.object(MissionRunner, "run", flaky_run):
                summary = runtime._resume_interrupted_missions()
        # The good mission still completed; the bad one is reported,
        # not fatal to the batch or the boot.
        self.assertEqual(self.store.get(good_id).status, MissionStatus.DONE)
        self.assertEqual([r["id"] for r in summary["resumed"]], [good_id])
        self.assertEqual(len(summary["failed"]), 1)
        self.assertEqual(summary["failed"][0]["id"], bad_id)
        self.assertIn("drive exploded", summary["failed"][0]["error"])
        notices = _missions_notices(self.context)
        self.assertEqual(len(notices), 1, notices)
        self.assertIn("bad", notices[0]["body"])

    def test_pending_interrupted_resumes_pristine_pending_untouched(self):
        # Died between create and the first heartbeat flip: PENDING but
        # with drive evidence (a completed step + a stale heartbeat).
        steps = [_step("a"), _step("b", ("a",))]
        interrupted = _mission_with_plan(self.store, steps, name="half")
        row = self.store.get(interrupted.id)
        row.state["completed_steps"] = ["a"]
        row.state["heartbeat"] = {"pid": DEAD_PID,
                                  "at": time.time() - 100000}
        self.store.save(row)
        # Never driven: created, planned, deliberately not started.
        pristine = _mission_with_plan(self.store, [_step("a")],
                                      name="pristine")
        runtime = self._runtime()
        executed: list[str] = []
        with _exec_patch(executed):
            summary = runtime._resume_interrupted_missions()
        self.assertEqual([r["id"] for r in summary["resumed"]],
                         [interrupted.id])
        self.assertEqual(executed, ["b"])
        self.assertEqual(self.store.get(interrupted.id).status,
                         MissionStatus.DONE)
        self.assertEqual(self.store.get(pristine.id).status,
                         MissionStatus.PENDING)

    def test_silence_when_nothing_to_resume(self):
        runtime = self._runtime()
        summary = runtime._resume_interrupted_missions()
        self.assertEqual(summary["resumed"], [])
        self.assertEqual(summary["failed"], [])
        self.assertEqual(summary["terminal_caught_up"], [])
        self.assertEqual(_missions_notices(self.context), [])


if __name__ == "__main__":
    unittest.main()
