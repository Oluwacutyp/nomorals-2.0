"""Wave F2 — missions: chat-visible progress, milestone pushes, stall reasons.

Covers the new progress layer (nomorals/missions/progress.py) and its
wiring into the runner:

- status queries: ``MissionStore.detail`` (% complete, current step,
  honest ETA, stall record) and ``render_status_text``
- proactive milestone delivery through the existing Notifier
  (started / step / stalled / done) with Wave D discipline:
  transition-gating, per-mission cooldowns, quiet-hours holds, and the
  Notifier's own dedupe choke point
- stalled-reason tracking: concrete blockers only, auto-recorded by the
  runner (consecutive failures, budget exhaustion) or explicit via
  ``mark_stalled`` / ``clear_stalled``
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nomorals.agents.notifier import Notifier
from nomorals.core.errors import NotFound, ValidationError
from nomorals.missions import (
    MissionMilestones,
    MissionRunner,
    MissionStatus,
    MissionStore,
    StallCode,
    StepOutcome,
    render_status_text,
)
from nomorals.missions import progress as progress_mod
from nomorals.storage.db import Database


# ── fakes ────────────────────────────────────────────────────────────────────

class _SendResult:
    def __init__(self, ok=True):
        self.ok = ok


class FakeGateway:
    """Records every send; owner channels report running."""

    def __init__(self, live_platforms=("telegram",)):
        self.live = set(live_platforms)
        self.sent = []  # (platform, chat_id, text)

    def status(self):
        return {p: {"running_in_session": True} for p in self.live}

    def send(self, platform, chat_ref, text):
        self.sent.append((platform, chat_ref.chat_id, text))
        return _SendResult(ok=True)


def make_partner(**kw):
    base = dict(
        owner_chats="telegram:111",
        proactive_enabled=True,
        proactive_briefing=True,
        proactive_watchers=True,
        quiet_start=0,   # 0 == 0 -> quiet hours disabled (deterministic tests)
        quiet_end=0,
        timezone="UTC",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def make_ctx(test=None, partner=None, gateway=None):
    tmp = tempfile.mkdtemp(prefix="missions-f2-")
    if test is not None:
        test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()
    settings = SimpleNamespace(workspace_dir=tmp,
                               partner=partner or make_partner())
    ctx = SimpleNamespace(db=db, settings=settings,
                          extras={"gateway": gateway} if gateway else {})
    return ctx, tmp


def make_clock(start=1_000_000.0):
    now = [start]
    return now, lambda: now[0]


def make_mission(store, name="m1", goal="test goal", plan_names=("a", "b", "c")):
    mission = store.create_new(goal, name=name)
    mission.state["plan"] = [
        {"name": n, "goal": f"do {n}", "role": "execution",
         "kind": "io", "depends_on": []}
        for n in plan_names
    ]
    store.save(mission)
    return mission


def make_reporter(ctx, store, start=1_000_000.0, cooldown=600.0):
    now, clock = make_clock(start)
    reporter = MissionMilestones(ctx, store=store, clock=clock,
                                 step_cooldown_seconds=cooldown)
    return reporter, now


OK_STEP = StepOutcome(step="a", ok=True, detail="", seconds=30.0, tokens=100)


# ── status queries ───────────────────────────────────────────────────────────

class StatusQueryTests(unittest.TestCase):
    def setUp(self):
        self.ctx, _ = make_ctx(self)
        self.store = MissionStore(self.ctx.db)

    def test_detail_reports_percent_current_step_and_eta(self):
        mission = make_mission(self.store)
        mission.state["completed_steps"] = ["a"]
        mission.spent_wall = 30.0
        self.store.save(mission)

        d = self.store.detail(mission.id)
        p = d["progress"]
        self.assertEqual(p["steps_done"], 1)
        self.assertEqual(p["total_steps"], 3)
        self.assertAlmostEqual(p["percent"], 100.0 / 3)
        self.assertEqual(p["current_step"], "b")
        # 30s per step measured, 2 steps left -> ~60s ETA
        self.assertAlmostEqual(d["eta_seconds"], 60.0)
        self.assertIsNone(d["stall"])

    def test_detail_eta_unknown_without_step_timing(self):
        mission = make_mission(self.store)
        d = self.store.detail(mission.id)
        self.assertIsNone(d["eta_seconds"])
        self.assertIn("no step timing", d["eta_note"])

    def test_detail_eta_unknown_without_plan(self):
        mission = self.store.create_new("no plan")
        d = self.store.detail(mission.id)
        self.assertIsNone(d["eta_seconds"])
        self.assertIn("no plan", d["eta_note"])

    def test_detail_eta_warns_when_wall_budget_runs_out_first(self):
        mission = make_mission(self.store, plan_names=("a", "b"))
        mission.state["completed_steps"] = ["a"]
        mission.spent_wall = 100.0
        mission.budget_wall = 120.0
        self.store.save(mission)
        d = self.store.detail(mission.id)
        self.assertAlmostEqual(d["eta_seconds"], 100.0)
        self.assertIn("budget runs out first", d["eta_note"])

    def test_detail_surfaces_stall_record(self):
        mission = make_mission(self.store)
        self.store.mark_stalled(mission.id, StallCode.WAITING_ON_PROVIDER,
                                "Groq is rate-limiting us", step="b")
        d = self.store.detail(mission.id)
        self.assertEqual(d["stall"]["code"], "waiting_on_provider")
        self.assertIn("Groq", d["stall"]["message"])
        self.assertEqual(d["stall"]["step"], "b")

    def test_detail_progress_excludes_plan_error_marker(self):
        mission = make_mission(self.store, plan_names=("a", "b"))
        mission.state["plan"].append({"__plan_error__": "degraded template"})
        self.store.save(mission)
        p = self.store.progress(mission.id)
        self.assertEqual(p["total_steps"], 2)

    def test_detail_feeds_devon_progress_report_shape(self):
        """devon.py _mission_progress_report reads store.detail(m.id) and
        touches d['progress'][...] + d['recent_checkpoints'] — this is the
        access pattern that used to AttributeError (no detail() existed)."""
        mission = make_mission(self.store)
        mission.state["completed_steps"] = ["a"]
        self.store.save(mission)
        self.store.checkpoint(mission, label="after:a")
        d = self.store.detail(mission.id)
        p = d["progress"]
        line = (
            f"{p['steps_done']}/{p['total_steps']} steps ({p['percent']:.0f}%), "
            f"wall {p['spent_wall_seconds']:.0f}s, tokens {p['spent_tokens']}"
            + (f", current: {p['current_step']}" if p.get("current_step") else "")
            + (f", last error: {p['last_error'][:120]}" if p.get("last_error") else "")
        )
        cps = d["recent_checkpoints"]
        self.assertIn("1/3 steps (33%)", line)
        self.assertEqual(cps[0]["label"], "after:a")

    def test_detail_unknown_mission_raises(self):
        with self.assertRaises(NotFound):
            self.store.detail("nope")


# ── milestone delivery ───────────────────────────────────────────────────────

class MilestoneDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.gateway = FakeGateway()
        self.ctx, _ = make_ctx(self, gateway=self.gateway)
        self.store = MissionStore(self.ctx.db)

    def test_started_pushes_once_across_resume(self):
        mission = make_mission(self.store)
        reporter, _ = make_reporter(self.ctx, self.store)
        r1 = reporter.on_started(mission)
        r2 = reporter.on_started(mission)  # resume must not re-send
        self.assertTrue(r1.get("delivered"))
        self.assertEqual(r2.get("suppressed"), "already-notified")
        self.assertEqual(len(self.gateway.sent), 1)
        self.assertIn("mission started", self.gateway.sent[0][2])

    def test_step_push_cooldown_holds(self):
        """Two steps 10s apart -> one chat push. No per-sub-step spam."""
        mission = make_mission(self.store)
        reporter, now = make_reporter(self.ctx, self.store, cooldown=600.0)
        mission.state["completed_steps"] = ["a"]
        self.store.save(mission)

        first = reporter.on_step(mission, OK_STEP)
        now[0] += 10.0
        second = reporter.on_step(mission, OK_STEP)

        self.assertTrue(first.get("delivered"))
        self.assertEqual(second.get("suppressed"), "cooldown")
        self.assertEqual(len(self.gateway.sent), 1)

    def test_step_push_allowed_after_cooldown(self):
        mission = make_mission(self.store)
        reporter, now = make_reporter(self.ctx, self.store, cooldown=600.0)
        mission.state["completed_steps"] = ["a"]
        self.store.save(mission)

        reporter.on_step(mission, OK_STEP)
        # real progress happened -> a new title, outside the fake cooldown
        mission.state["completed_steps"] = ["a", "b"]
        self.store.save(mission)
        now[0] += 601.0
        second = reporter.on_step(mission, OK_STEP)

        self.assertTrue(second.get("delivered"))
        self.assertEqual(len(self.gateway.sent), 2)
        self.assertIn("67%", self.gateway.sent[1][2])

    def test_terminal_push_always_sent_even_in_cooldown(self):
        mission = make_mission(self.store)
        reporter, now = make_reporter(self.ctx, self.store, cooldown=600.0)
        mission.state["completed_steps"] = ["a"]
        self.store.save(mission)

        reporter.on_step(mission, OK_STEP)
        now[0] += 5.0  # inside the step cooldown
        terminal = reporter.on_terminal(mission, MissionStatus.DONE)

        self.assertTrue(terminal.get("delivered"))
        self.assertEqual(len(self.gateway.sent), 2)
        self.assertIn("mission done", self.gateway.sent[1][2])

    def test_terminal_push_fires_once(self):
        mission = make_mission(self.store)
        reporter, _ = make_reporter(self.ctx, self.store)
        reporter.on_terminal(mission, MissionStatus.FAILED, error="boom")
        again = reporter.on_terminal(mission, MissionStatus.FAILED, error="boom")
        self.assertEqual(again.get("suppressed"), "already-notified")
        self.assertEqual(len(self.gateway.sent), 1)

    def test_stalled_push_on_transition_only(self):
        mission = make_mission(self.store)
        reporter, _ = make_reporter(self.ctx, self.store)

        self.store.mark_stalled(mission.id, StallCode.WAITING_ON_PROVIDER,
                                "Groq rate-limited")
        mission = self.store.get(mission.id)
        first = reporter.on_stalled(mission)
        same_again = reporter.on_stalled(mission)

        self.assertTrue(first.get("delivered"))
        self.assertEqual(same_again.get("suppressed"), "already-notified")
        self.assertEqual(len(self.gateway.sent), 1)
        self.assertIn("mission stalled", self.gateway.sent[0][2])
        self.assertIn("Groq", self.gateway.sent[0][2])

        # a *different* stall reason is a new transition -> pushed
        self.store.mark_stalled(mission.id, StallCode.BLOCKED_ON_APPROVAL,
                                "needs your go-ahead")
        mission = self.store.get(mission.id)
        second = reporter.on_stalled(mission)
        self.assertTrue(second.get("delivered"))
        self.assertEqual(len(self.gateway.sent), 2)

    def test_quiet_hours_holds_step_push(self):
        mission = make_mission(self.store)
        reporter, _ = make_reporter(self.ctx, self.store)
        mission.state["completed_steps"] = ["a"]
        self.store.save(mission)

        with patch.object(progress_mod, "_in_quiet_hours_now", return_value=True):
            res = reporter.on_step(mission, OK_STEP)

        self.assertEqual(res.get("held"), "quiet-hours")
        self.assertEqual(self.gateway.sent, [])  # nothing woke the owner
        rows = Notifier(self.ctx).recent(5, kind="mission")
        self.assertEqual(rows[0]["delivery_state"], "held-quiet-hours")

    def test_milestone_never_raises_without_db(self):
        ctx = SimpleNamespace(db=None, settings=SimpleNamespace(
            partner=make_partner()), extras={})
        reporter = MissionMilestones(ctx, store=None)
        mission = SimpleNamespace(
            id="x", name="m", goal="g", status="running",
            budget_wall=0.0, budget_tokens=0,
            state={"plan": [], "completed_steps": []},
        )
        # no db at all -> every event degrades to a suppression dict,
        # never raises
        results = [
            reporter.on_started(mission),
            reporter.on_step(mission, OK_STEP),
            reporter.on_stalled(mission),
            reporter.on_terminal(mission, MissionStatus.DONE),
        ]
        for res in results:
            self.assertIsInstance(res, dict)


# ── Wave D discipline on mission pushes ──────────────────────────────────────

class NotifierDisciplineTests(unittest.TestCase):
    def setUp(self):
        self.gateway = FakeGateway()
        self.ctx, _ = make_ctx(self, gateway=self.gateway)

    def test_mission_kind_dedupe_choke_point(self):
        """The same (kind, title) inside the dedupe window collapses —
        mission pushes get the Wave D dedupe for free."""
        n = Notifier(self.ctx)
        first = n.publish("mission", "mission update: m1 — 50%", "body")
        second = n.publish("mission", "mission update: m1 — 50%", "body")
        self.assertTrue(first.get("delivered"))
        self.assertTrue(second.get("deduped"))
        self.assertEqual(second.get("delivery_state"), "deduped")
        self.assertEqual(len(self.gateway.sent), 1)

    def test_milestone_rows_carry_delivery_state(self):
        store = MissionStore(self.ctx.db)
        mission = make_mission(store)
        reporter, _ = make_reporter(self.ctx, store)
        reporter.on_started(mission)
        rows = Notifier(self.ctx).recent(5, kind="mission")
        self.assertEqual(rows[0]["kind"], "mission")
        self.assertEqual(rows[0]["delivery_state"], "sent")
        self.assertTrue(rows[0]["delivered"])


# ── stalled-reason tracking ──────────────────────────────────────────────────

class StallTrackingTests(unittest.TestCase):
    def setUp(self):
        self.gateway = FakeGateway()
        self.ctx, _ = make_ctx(self, gateway=self.gateway)
        self.store = MissionStore(self.ctx.db)

    def test_mark_stalled_round_trip(self):
        mission = make_mission(self.store)
        out = self.store.mark_stalled(
            mission.id, StallCode.WAITING_ON_PROVIDER,
            "OpenAI is timing out", step="fetch")
        self.assertTrue(out["changed"])
        stall = self.store.stall(mission.id)
        self.assertEqual(stall["code"], "waiting_on_provider")
        self.assertEqual(stall["message"], "OpenAI is timing out")
        self.assertEqual(stall["step"], "fetch")
        self.assertGreater(stall["since"], 0)

    def test_identical_remark_is_not_a_new_event(self):
        mission = make_mission(self.store)
        self.store.mark_stalled(mission.id, StallCode.DEPENDENCY_MISSING,
                                "ffmpeg not installed")
        again = self.store.mark_stalled(mission.id, StallCode.DEPENDENCY_MISSING,
                                        "ffmpeg not installed")
        self.assertFalse(again["changed"])

    def test_mark_stalled_rejects_unknown_code(self):
        mission = make_mission(self.store)
        with self.assertRaises(ValidationError):
            self.store.mark_stalled(mission.id, "vibes", "bad vibes")

    def test_mark_stalled_rejects_empty_message(self):
        mission = make_mission(self.store)
        with self.assertRaises(ValidationError):
            self.store.mark_stalled(mission.id, StallCode.BUDGET_EXHAUSTED, "  ")

    def test_mark_stalled_refuses_terminal_mission(self):
        mission = make_mission(self.store)
        mission.status = MissionStatus.DONE
        self.store.save(mission)
        with self.assertRaises(ValidationError):
            self.store.mark_stalled(mission.id, StallCode.BUDGET_EXHAUSTED, "x")

    def test_clear_stall(self):
        mission = make_mission(self.store)
        self.assertFalse(self.store.clear_stall(mission.id))
        self.store.mark_stalled(mission.id, StallCode.BLOCKED_ON_APPROVAL,
                                "needs a yes")
        self.assertTrue(self.store.clear_stall(mission.id))
        self.assertIsNone(self.store.stall(mission.id))

    def test_runner_auto_stalls_after_consecutive_failures(self):
        """Three failed steps in a row -> concrete stall + one chat push."""
        reporter, _ = make_reporter(self.ctx, self.store)
        runner = MissionRunner(self.ctx, store=self.store,
                               milestone_reporter=reporter)
        mission = make_mission(self.store)
        outcome = StepOutcome(step="s1", ok=False, detail="provider exploded")

        changed = [runner._apply_step_result(mission, outcome, "s1")
                   for _ in range(3)]
        self.assertEqual(changed, [False, False, True])
        # _apply_step_result mutates in memory; persist like run() does
        self.store.save(mission)
        stall = self.store.stall(mission.id)
        self.assertEqual(stall["code"], "retry_budget_exhausted")
        self.assertIn("3 consecutive step failures", stall["message"])
        self.assertIn("provider exploded", stall["message"])

        runner._report("on_stalled", mission)
        self.assertEqual(len(self.gateway.sent), 1)
        self.assertIn("retry budget exhausted", self.gateway.sent[0][2])

        # a 4th identical failure is not a new transition -> no re-push
        mission = self.store.get(mission.id)
        self.assertFalse(runner._apply_step_result(mission, outcome, "s1"))

    def test_runner_stall_clears_on_progress(self):
        reporter, _ = make_reporter(self.ctx, self.store)
        runner = MissionRunner(self.ctx, store=self.store,
                               milestone_reporter=reporter)
        mission = make_mission(self.store)
        bad = StepOutcome(step="s1", ok=False, detail="boom")
        for _ in range(3):
            runner._apply_step_result(mission, bad, "s1")
        self.store.save(mission)
        self.assertIsNotNone(self.store.stall(mission.id))

        mission = self.store.get(mission.id)
        runner._apply_step_result(mission, OK_STEP, "s1")
        self.store.save(mission)
        self.assertIsNone(self.store.stall(mission.id))
        self.assertEqual(mission.state.get("consec_failures"), 0)

    def test_runner_budget_exhaustion_records_concrete_stall(self):
        reporter, _ = make_reporter(self.ctx, self.store)
        runner = MissionRunner(self.ctx, store=self.store,
                               milestone_reporter=reporter)
        mission = make_mission(self.store)
        mission.budget_wall = 60.0
        mission.spent_wall = 60.0
        mission.budget_tokens = 1000
        mission.spent_tokens = 1000
        self.store.save(mission)

        mission = self.store.get(mission.id)
        self.assertTrue(runner._record_budget_stall(mission))
        stall = self.store.stall(mission.id)
        self.assertEqual(stall["code"], "budget_exhausted")
        self.assertIn("60s/60s", stall["message"])
        self.assertIn("1000/1000", stall["message"])
        self.assertEqual(len(self.gateway.sent), 1)
        self.assertIn("mission stalled", self.gateway.sent[0][2])

        # not a new event on repeat -> no duplicate push
        mission = self.store.get(mission.id)
        self.assertFalse(runner._record_budget_stall(mission))
        self.assertEqual(len(self.gateway.sent), 1)

    def test_runner_mark_stalled_public_api_pushes(self):
        reporter, _ = make_reporter(self.ctx, self.store)
        runner = MissionRunner(self.ctx, store=self.store,
                               milestone_reporter=reporter)
        mission = make_mission(self.store)
        out = runner.mark_stalled(
            mission.id, StallCode.BLOCKED_ON_APPROVAL,
            "needs your go-ahead on the deploy")
        self.assertTrue(out["changed"])
        self.assertEqual(len(self.gateway.sent), 1)
        self.assertIn("blocked on approval", self.gateway.sent[0][2])
        self.assertIn("go-ahead", self.gateway.sent[0][2])
        # repeat of the identical stall: no re-push
        runner.mark_stalled(mission.id, StallCode.BLOCKED_ON_APPROVAL,
                            "needs your go-ahead on the deploy")
        self.assertEqual(len(self.gateway.sent), 1)


# ── rendering ────────────────────────────────────────────────────────────────

class RenderTests(unittest.TestCase):
    def setUp(self):
        self.ctx, _ = make_ctx(self)
        self.store = MissionStore(self.ctx.db)

    def test_render_includes_all_sections(self):
        mission = make_mission(self.store, name="deploy-bot",
                               goal="deploy the bot")
        mission.state["completed_steps"] = ["a"]
        mission.spent_wall = 30.0
        mission.spent_tokens = 500
        self.store.save(mission)
        self.store.mark_stalled(mission.id, StallCode.WAITING_ON_PROVIDER,
                                "Serv00 signup still pending review", step="b")
        text = render_status_text(self.store.detail(mission.id))
        self.assertIn("deploy-bot", text)
        self.assertIn("33%", text)
        self.assertIn("current: b", text)
        self.assertIn("eta:", text)
        self.assertIn("500 tokens", text)
        self.assertIn("stalled: waiting on provider", text)
        self.assertIn("Serv00 signup still pending review", text)

    def test_render_without_stall_omits_stall_line(self):
        mission = make_mission(self.store)
        text = render_status_text(self.store.detail(mission.id))
        self.assertNotIn("stalled", text)
        self.assertIn("eta: unknown", text)  # honest, not invented


if __name__ == "__main__":
    unittest.main()
