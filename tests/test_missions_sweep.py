"""Sweep tests for nomorals/missions: the 2026-10-10 system-wide upgrade.

Covers the new behavior only — pre-existing behavior keeps its own test
files (test_missions.py, test_golden_missions.py, ...):

- mission.py: priority/tags/parent_id, templates, sub-missions, list
  filters, bulk ops, archive, summary cards
- progress.py: EMA ETA + historical priors, renderers (bar, sparkline,
  status styles, card, table, result card), milestone notify levels +
  digest
- idempotency.py: fingerprint conflicts, TTL expiry, failed_since/redrive,
  peek, labeled stats
- runner.py: error classification, Retry-After backoff, approval gates,
  saga compensation, dry-run, heartbeat progress, schedule timeout
- golden.py: crash_no_dup + saga_undo drills, normalize_output,
  check_regression, render_golden_report
"""

from __future__ import annotations

import shutil
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.agents.orchestrator import PlanStep
from nomorals.core.config import Settings
from nomorals.core.errors import NotFound, ValidationError
from nomorals.core.tasks import TaskKind
from nomorals.missions import (
    MISSION_TEMPLATES,
    GoldenResult,
    GoldenRunner,
    IdempotencyConflict,
    IdempotencyStore,
    Mission,
    MissionResult,
    MissionRunner,
    MissionStatus,
    MissionStore,
    check_regression,
    dedupe,
    estimate_eta,
    eta_breakdown,
    list_golden_missions,
    list_mission_templates,
    normalize_output,
    render_golden_report,
    render_mission_table,
    render_progress_bar,
    render_result_card,
    render_sparkline,
    render_status_text,
)
from nomorals.missions.golden import GOLDEN_MISSIONS
from nomorals.missions.progress import MissionMilestones
from nomorals.missions.runner import (
    ERROR_PERMANENT,
    ERROR_RATE_LIMIT,
    ERROR_SERVER,
    ERROR_TRANSIENT,
    StepOutcome,
    _backoff_delay,
    _classify_error,
    _retry_after_seconds,
    render_plan_text,
)
from nomorals.storage.db import Database


def make_context(test):
    home = tempfile.mkdtemp(prefix="nm-sweep-")
    test.addCleanup(shutil.rmtree, home, ignore_errors=True)
    context = build_context(Settings(home=home))
    context.__enter__()
    test.addCleanup(context.__exit__, None, None, None)
    return context


def make_db(test):
    tmp = tempfile.mkdtemp(prefix="nm-sweep-db-")
    test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    db = Database(str(Path(tmp) / "sweep.db"))
    db.migrate()
    test.addCleanup(db.close)
    return db


def stub_orchestrator(steps):
    """An orchestrator factory returning a fixed plan (no model)."""
    plan = SimpleNamespace(steps=list(steps), plan_error="")

    def factory():
        return SimpleNamespace(plan=lambda goal: plan, max_steps=8)

    return factory


def ok_step_runner(runner):
    """Monkeypatch the agent invocation: every step succeeds instantly."""

    def _run(mission, step, started, policy=None):
        return StepOutcome(step=step.name, ok=True, detail="stub ok",
                           seconds=0.01, tokens=3,
                           payload={"note": "stub"})

    runner._run_step_agent = _run


class MissionModelSweepTests(unittest.TestCase):
    def setUp(self):
        self.context = make_context(self)
        self.store = MissionStore(self.context.db)

    def test_priority_tags_parent_round_trip(self):
        mission = self.store.create_new(
            "sweep goal", priority=5, tags=["nightly", "research"],
            parent_id="parent-123")
        fetched = self.store.get(mission.id)
        self.assertEqual(fetched.priority, 5)
        self.assertEqual(fetched.tags, ["nightly", "research"])
        self.assertEqual(fetched.parent_id, "parent-123")
        self.assertTrue(fetched.is_child)
        # metadata keeps only the caller's own keys
        self.assertNotIn("priority", fetched.metadata)

    def test_priority_defaults(self):
        mission = self.store.create_new("plain goal")
        self.assertEqual(mission.priority, 0)
        self.assertEqual(mission.tags, [])
        self.assertFalse(mission.is_child)

    def test_summary_card(self):
        mission = self.store.create_new("card goal", name="card mission")
        card = mission.summary_card()
        self.assertIn("card mission", card)
        self.assertIn("┌", card)
        self.assertIn("progress", card)

    def test_templates_catalog(self):
        catalog = list_mission_templates()
        self.assertEqual({t["key"] for t in catalog}, set(MISSION_TEMPLATES))
        self.assertTrue(all(t["steps"] for t in catalog))

    def test_create_from_template(self):
        mission = self.store.create_from_template(
            "research_write_verify", "research the sweep")
        plan = mission.state.get("plan") or []
        self.assertEqual([s["name"] for s in plan],
                         ["collect", "draft", "verify"])
        self.assertEqual(mission.state.get("template"), "research_write_verify")
        self.assertIn("research", mission.tags)
        # policies ride along in the persisted plan
        collect = plan[0]
        self.assertEqual(collect["policy"]["retries"], 2)

    def test_create_from_template_unknown(self):
        with self.assertRaises(ValidationError):
            self.store.create_from_template("nope", "goal")

    def test_spawn_child_tree(self):
        parent = self.store.create_new("parent goal", tags=["ops"])
        child = self.store.spawn_child(parent.id, "child goal")
        grandchild = self.store.spawn_child(child.id, "grandchild goal")
        self.assertEqual(child.parent_id, parent.id)
        self.assertIn("child", child.tags)
        self.assertIn("ops", child.tags)
        self.assertEqual([c.id for c in self.store.children(parent.id)],
                         [child.id])
        descendants = self.store.descendants(parent.id)
        self.assertEqual({d.id for d in descendants},
                         {child.id, grandchild.id})
        tree = self.store.tree(parent.id)
        self.assertEqual(tree["mission"]["id"], parent.id)
        self.assertEqual(tree["children"][0]["children"][0]["mission"]["id"],
                         grandchild.id)

    def test_spawn_child_terminal_parent(self):
        parent = self.store.create_new("parent goal")
        self.store.set_status(parent.id, MissionStatus.DONE)
        with self.assertRaises(ValidationError):
            self.store.spawn_child(parent.id, "child goal")

    def test_list_filters(self):
        self.store.create_new("alpha research", priority=1, tags=["research"])
        self.store.create_new("beta build", priority=9, tags=["build"])
        self.store.create_new("gamma research notes", priority=5,
                              tags=["research"])
        by_tag = self.store.list(tag="research")
        self.assertEqual(len(by_tag), 2)
        by_prio = self.store.list(min_priority=5)
        self.assertEqual(len(by_prio), 2)
        by_search = self.store.search("gamma")
        self.assertEqual(len(by_search), 1)
        ordered = self.store.list(order="priority")
        self.assertEqual([m.priority for m in ordered], [9, 5, 1])
        with self.assertRaises(ValidationError):
            self.store.search("   ")

    def test_bulk_set_status(self):
        a = self.store.create_new("a")
        b = self.store.create_new("b")
        result = self.store.bulk_set_status(
            [a.id, b.id, "missing-id"], MissionStatus.CANCELLED, note="sweep")
        self.assertEqual(result["updated_count"], 2)
        self.assertIn("missing-id", result["skipped"])
        self.assertEqual(self.store.get(a.id).status, MissionStatus.CANCELLED)

    def test_bulk_cancel_keeps_terminal(self):
        live = self.store.create_new("live one")
        done = self.store.create_new("done one")
        self.store.set_status(done.id, MissionStatus.DONE)
        result = self.store.bulk_cancel()
        self.assertEqual(result["updated"], [live.id])
        self.assertEqual(self.store.get(done.id).status, MissionStatus.DONE)

    def test_archive(self):
        old = self.store.create_new("old mission")
        self.store.checkpoint(old, label="c1")
        self.store.set_status(old.id, MissionStatus.DONE)
        # backdate the finish past the cutoff
        row_old = self.store.get(old.id)
        row_old.finished_at = time.time() - 40 * 86400
        self.store.save(row_old)
        fresh = self.store.create_new("fresh mission")
        self.store.set_status(fresh.id, MissionStatus.DONE)
        deleted = self.store.archive(older_than_seconds=30 * 86400)
        self.assertEqual(deleted["missions"], 1)
        self.assertEqual(deleted["checkpoints"], 1)
        with self.assertRaises(Exception):
            self.store.get(old.id)
        # the fresh terminal mission survives
        self.assertEqual(self.store.get(fresh.id).status, MissionStatus.DONE)


class ProgressSweepTests(unittest.TestCase):
    def setUp(self):
        self.context = make_context(self)
        self.store = MissionStore(self.context.db)

    def _mission_with_timing(self):
        mission = self.store.create_new("timed goal")
        mission.state["plan"] = [
            {"name": "a", "goal": "", "role": "", "kind": "io",
             "depends_on": []},
            {"name": "b", "goal": "", "role": "", "kind": "io",
             "depends_on": []},
            {"name": "c", "goal": "", "role": "", "kind": "io",
             "depends_on": []},
            {"name": "d", "goal": "", "role": "", "kind": "io",
             "depends_on": []},
        ]
        mission.state["completed_steps"] = ["a", "b"]
        # insertion-ordered: a took 10s, b took 30s.
        # EMA(0.3): 10 -> 0.3*30 + 0.7*10 = 16s per step.
        mission.state["step_durations"] = {"a": 10.0, "b": 30.0}
        return self.store.save(mission)

    def test_eta_uses_ema(self):
        mission = self._mission_with_timing()
        info = eta_breakdown(mission)
        self.assertEqual(info["method"], "ema")
        self.assertAlmostEqual(info["per_step_s"], 16.0, places=1)
        self.assertAlmostEqual(info["eta_seconds"], 32.0, places=1)
        self.assertEqual(info["remaining"], 2)
        eta, note = estimate_eta(mission)
        self.assertAlmostEqual(eta, 32.0, places=1)
        self.assertIn("EMA", note)

    def test_eta_historical_prior(self):
        # seed history: one finished mission, 2 iterations, 40s wall
        hist = self.store.create_new("history goal")
        hist.status = MissionStatus.DONE
        hist.iterations = 2
        hist.spent_wall = 40.0
        hist.finished_at = time.time()
        self.store.save(hist)
        mission = self.store.create_new("fresh goal")
        mission.state["plan"] = [
            {"name": "x", "goal": "", "role": "", "kind": "io",
             "depends_on": []},
            {"name": "y", "goal": "", "role": "", "kind": "io",
             "depends_on": []},
        ]
        mission = self.store.save(mission)
        info = eta_breakdown(mission, self.context.db)
        self.assertEqual(info["method"], "history")
        self.assertAlmostEqual(info["per_step_s"], 20.0, places=1)
        self.assertAlmostEqual(info["eta_seconds"], 40.0, places=1)

    def test_eta_no_data(self):
        mission = self.store.create_new("no plan goal")
        info = eta_breakdown(mission, self.context.db)
        self.assertIsNone(info["eta_seconds"])
        self.assertEqual(info["method"], "none")

    def test_render_progress_bar(self):
        bar = render_progress_bar(50.0, width=10)
        self.assertEqual(bar, "█████░░░░░ 50%")
        self.assertEqual(render_progress_bar(100, width=4), "████ 100%")
        self.assertEqual(render_progress_bar(0, width=4), "░░░░ 0%")
        ascii_bar = render_progress_bar(50.0, width=10, ascii_only=True)
        self.assertEqual(ascii_bar, "#####----- 50%")
        # clamps, never raises
        self.assertIn("100%", render_progress_bar(999, width=4))

    def test_render_sparkline(self):
        self.assertEqual(render_sparkline([1, 2, 3, 4]), "▁▃▅█")
        self.assertEqual(render_sparkline([]), "")
        self.assertEqual(render_sparkline([5, 5, 5]), "▄▄▄")

    def test_status_text_styles(self):
        mission = self._mission_with_timing()
        detail = self.store.detail(mission.id)
        full = render_status_text(detail)
        self.assertIn("progress:", full)
        compact = render_status_text(detail, style="compact")
        self.assertNotIn("\n", compact)
        self.assertIn("steps", compact)
        plain = render_status_text(detail, style="plain")
        self.assertNotIn("🎯", plain)
        self.assertNotIn("⚙️", plain)
        card = render_status_text(detail, style="card")
        self.assertIn("┌", card)
        self.assertIn("progress", card)
        with self.assertRaises(ValueError):
            render_status_text(detail, style="bogus")

    def test_detail_carries_eta_breakdown(self):
        mission = self._mission_with_timing()
        detail = self.store.detail(mission.id)
        self.assertIn("eta_breakdown", detail)
        self.assertEqual(detail["eta_breakdown"]["method"], "ema")

    def test_render_mission_table(self):
        self.store.create_new("table alpha", name="alpha")
        self.store.create_new("table beta", name="beta")
        table = render_mission_table(self.store.list(limit=10))
        self.assertIn("alpha", table)
        self.assertIn("beta", table)
        self.assertIn("ID", table)
        self.assertEqual(render_mission_table([]), "(no missions)")

    def test_render_result_card(self):
        result = MissionResult(
            mission_id="abc123", status=MissionStatus.DONE, success=1.0,
            iterations=3,
            steps=[StepOutcome(step="a", ok=True, seconds=1.0),
                   StepOutcome(step="b", ok=False, detail="boom")],
            seconds=12.0, lessons=["lesson one"])
        card = result.summary_card()
        self.assertIn("┌", card)
        self.assertIn("1/2 ok", card)
        self.assertIn("lesson one", card)
        direct = render_result_card(result.to_dict())
        self.assertEqual(card, direct)

    def test_milestone_notify_levels(self):
        mission = self.store.create_new("notify goal")
        reporter = MissionMilestones(
            self.context, store=self.store, step_cooldown_seconds=0,
            notify_level="milestones")
        outcome = StepOutcome(step="s", ok=True)
        suppressed = reporter.on_step(mission, outcome)
        self.assertEqual(suppressed["suppressed"], "notify-level")
        reporter.set_notify_level("all")
        # with cooldown 0 the step push goes through the notifier path
        # (telemetry-safe: returns a dict either way)
        res = reporter.on_step(mission, outcome)
        self.assertEqual(res.get("event", "step"), "step")
        with self.assertRaises(ValueError):
            reporter.set_notify_level("bogus")

    def test_milestone_digest(self):
        mission = self.store.create_new("digest goal")
        reporter = MissionMilestones(self.context, store=self.store)
        reporter._mark(mission, "started", "mission started: digest goal")
        reporter._mark(mission, "terminal:done", "mission done")
        digest = reporter.digest(mission)
        self.assertIn("digest", digest)
        self.assertIn("started", digest)


class IdempotencySweepTests(unittest.TestCase):
    def setUp(self):
        self.db = make_db(self)
        self.store = IdempotencyStore(self.db)

    def test_fingerprint_conflict(self):
        first = dedupe(self.store, "fp-key", lambda: {"v": 1},
                       fingerprint={"op": "create"})
        self.assertTrue(first.executed)
        replay = dedupe(self.store, "fp-key", lambda: {"v": 2},
                        fingerprint={"op": "create"})
        self.assertFalse(replay.executed)
        self.assertEqual(replay.value, {"v": 1})
        with self.assertRaises(IdempotencyConflict):
            dedupe(self.store, "fp-key", lambda: {"v": 3},
                   fingerprint={"op": "delete"})

    def test_fingerprint_opt_in_only(self):
        # historical keys without fingerprints keep replaying silently
        dedupe(self.store, "legacy-key", lambda: {"v": 1})
        replay = dedupe(self.store, "legacy-key", lambda: {"v": 2},
                        fingerprint={"op": "anything"})
        self.assertFalse(replay.executed)

    def test_ttl_expiry(self):
        dedupe(self.store, "ttl-key", lambda: {"v": 1}, ttl_seconds=3600)
        self.assertIsNotNone(self.store.get("ttl-key"))
        # expire it by hand (simulating the clock passing the TTL)
        self.db.execute(
            "UPDATE idempotency_keys SET expires_at = ? WHERE key = ?",
            (time.time() - 1, "ttl-key"))
        self.assertIsNone(self.store.get("ttl-key"))
        # ...and the key is a new operation again
        second = dedupe(self.store, "ttl-key", lambda: {"v": 2},
                        ttl_seconds=3600)
        self.assertTrue(second.executed)
        self.assertEqual(second.value, {"v": 2})

    def test_failed_since_and_redrive(self):
        def boom():
            raise RuntimeError("downstream exploded")

        with self.assertRaises(RuntimeError):
            dedupe(self.store, "fail-key", boom)
        failed = self.store.failed_since(time.time() - 60)
        self.assertEqual([f["key"] for f in failed], ["fail-key"])
        self.assertTrue(self.store.redrive("fail-key"))
        # completed keys refuse redrive
        dedupe(self.store, "ok-key", lambda: 1)
        self.assertFalse(self.store.redrive("ok-key"))
        self.assertFalse(self.store.redrive("missing-key"))

    def test_peek_does_not_claim(self):
        self.assertIsNone(self.store.peek("peek-key"))
        dedupe(self.store, "peek-key", lambda: {"v": 1}, label="sweep")
        peek = self.store.peek("peek-key")
        self.assertEqual(peek["status"], "completed")
        self.assertEqual(peek["label"], "sweep")
        stats = self.store.stats()
        self.assertEqual(stats["by_label"], {"sweep": 1})


class RunnerSweepTests(unittest.TestCase):
    def setUp(self):
        self.context = make_context(self)
        self.store = MissionStore(self.context.db)

    def _runner(self, steps, **kwargs):
        runner = MissionRunner(
            self.context, store=self.store,
            orchestrator_factory=stub_orchestrator(steps),
            milestones=False, **kwargs)
        ok_step_runner(runner)
        return runner

    # ── error classification / backoff ──────────────────────────────

    def test_classify_error(self):
        self.assertEqual(_classify_error("429 rate limit exceeded"),
                         ERROR_RATE_LIMIT)
        self.assertEqual(_classify_error("Too Many Requests"), ERROR_RATE_LIMIT)
        self.assertEqual(_classify_error("connection reset by peer"),
                         ERROR_TRANSIENT)
        self.assertEqual(_classify_error("502 Bad Gateway"), ERROR_SERVER)
        self.assertEqual(_classify_error("ValidationError: bad arg"),
                         ERROR_PERMANENT)
        self.assertEqual(_classify_error("some business rule failed"),
                         ERROR_PERMANENT)

    def test_retry_after_parsing(self):
        self.assertEqual(_retry_after_seconds("429: retry after 30 seconds"),
                         30.0)
        self.assertEqual(_retry_after_seconds("Retry-After: 5"), 5.0)
        self.assertEqual(_retry_after_seconds("retry-after: 2m"), 120.0)
        self.assertEqual(_retry_after_seconds("plain failure"), 0.0)

    def test_backoff_full_jitter_bounds(self):
        policy = {"retry_backoff_s": 2.0}
        for _ in range(50):
            delay = _backoff_delay(policy, 1, ERROR_TRANSIENT)
            self.assertGreaterEqual(delay, 0.0)
            self.assertLessEqual(delay, 2.0)
            limited = _backoff_delay(policy, 1, ERROR_RATE_LIMIT)
            self.assertLessEqual(limited, 6.0)  # 3x multiplier
        # server-asked wait wins
        policy = {"retry_backoff_s": 2.0, "retry_after_s": 42.0}
        self.assertEqual(_backoff_delay(policy, 3, ERROR_TRANSIENT), 42.0)
        # cap respected at high attempts
        self.assertLessEqual(_backoff_delay({"retry_backoff_s": 300.0}, 10),
                             300.0)

    # ── approval gates ──────────────────────────────────────────────

    def _approval_steps(self):
        return [
            PlanStep(name="first", goal="do first", role="execution",
                     kind=TaskKind.IO),
            PlanStep(name="gate", goal="needs a human", role="execution",
                     kind=TaskKind.IO,
                     payload={"needs_approval": True}),
            PlanStep(name="last", goal="do last", role="execution",
                     kind=TaskKind.IO, depends_on=["gate"]),
        ]

    def test_approval_gate_pauses_and_resumes(self):
        runner = self._runner(self._approval_steps())
        result = runner.run(self.store.create_new("approval mission"),
                            reflect=False)
        self.assertEqual(result.status, MissionStatus.PAUSED)
        mission_id = result.mission_id
        pending = runner.pending_approval(mission_id)
        self.assertEqual(pending["step"], "gate")
        mission = self.store.get(mission_id)
        self.assertIsNotNone(mission.state.get("stall"))
        self.assertEqual(mission.state["stall"]["code"], "blocked_on_approval")
        # approving an unknown mission is NotFound, not a silent no-op
        with self.assertRaises(NotFound):
            runner.approve("missing-id")
        # grant: the mission resumes and completes
        verdict = runner.approve(mission_id, approved=True, note="looks good")
        self.assertTrue(verdict["approved"])
        self.assertEqual(verdict["status"], MissionStatus.DONE)
        resumed = self.store.get(mission_id)
        self.assertIsNone(resumed.state.get("pending_approval"))
        self.assertTrue(resumed.state["approvals"]["gate"]["approved"])

    def test_approval_deny_fails(self):
        runner = self._runner(self._approval_steps())
        result = runner.run(self.store.create_new("deny mission"),
                            reflect=False)
        verdict = runner.deny(result.mission_id, note="not yet")
        self.assertFalse(verdict["approved"])
        self.assertEqual(verdict["status"], MissionStatus.FAILED)
        mission = self.store.get(result.mission_id)
        self.assertIn("approval denied", mission.state["last_error"])

    def test_approve_without_pending(self):
        runner = self._runner(self._approval_steps())
        mission = self.store.create_new("no gate mission")
        with self.assertRaises(ValidationError):
            runner.approve(mission.id)

    # ── saga compensation ───────────────────────────────────────────

    def test_compensate_skips_without_policy(self):
        runner = self._runner([PlanStep(name="a", goal="a",
                                        role="execution",
                                        kind=TaskKind.IO)])
        mission = self.store.create_new("no saga mission")
        log = runner._compensate(
            mission, [StepOutcome(step="a", ok=True)])
        self.assertEqual(log, [])

    def test_compensate_runs_reverse_idempotent(self):
        steps = [
            PlanStep(name="a", goal="make a", role="execution",
                     kind=TaskKind.IO, payload={"compensate": "undo a"}),
            PlanStep(name="b", goal="make b", role="execution",
                     kind=TaskKind.IO, payload={"compensate": "undo b"}),
        ]
        runner = self._runner(steps)
        mission = self.store.create_new("saga mission")
        mission.state["plan"] = [
            {"name": "a", "goal": "make a", "role": "execution",
             "kind": "io", "depends_on": [],
             "policy": {"compensate": "undo a"}},
            {"name": "b", "goal": "make b", "role": "execution",
             "kind": "io", "depends_on": [],
             "policy": {"compensate": "undo b"}},
        ]
        mission.state["completed_steps"] = ["a", "b"]
        self.store.save(mission)
        calls = []

        class FakeAgent:
            def run(self, prompt):
                calls.append(prompt)
                return SimpleNamespace(ok=True, error="",
                                       output={"undone": True})

        with mock.patch("nomorals.agents.roles.build_agent",
                        return_value=FakeAgent()):
            log = runner._compensate(
                mission, [StepOutcome(step="a", ok=True),
                          StepOutcome(step="b", ok=True)])
        self.assertEqual([e["step"] for e in log], ["b", "a"])
        self.assertTrue(all(e["ok"] for e in log))
        self.assertIn("undo b", calls[0])
        # second call skips already-compensated steps
        with mock.patch("nomorals.agents.roles.build_agent",
                        return_value=FakeAgent()):
            again = runner._compensate(
                mission, [StepOutcome(step="a", ok=True),
                          StepOutcome(step="b", ok=True)])
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(again), 2)

    # ── dry run / plan rendering ────────────────────────────────────

    def test_dry_run_plans_without_executing(self):
        steps = [
            PlanStep(name="one", goal="first", role="research",
                     kind=TaskKind.IO),
            PlanStep(name="two", goal="second", role="execution",
                     kind=TaskKind.IO, depends_on=["one"],
                     payload={"needs_approval": True}),
        ]
        runner = self._runner(steps)
        preview = runner.dry_run("preview goal", name="preview")
        self.assertEqual(preview["step_count"], 2)
        self.assertEqual(preview["levels"], [["one"], ["two"]])
        self.assertIn("level 0", preview["text"])
        self.assertIn("needs-approval", preview["text"])
        self.assertTrue(any("approval" in w for w in preview["warnings"]))
        # nothing persisted
        self.assertEqual(self.store.list(), [])

    def test_render_plan_text(self):
        steps = [
            PlanStep(name="a", goal="ga", role="research", kind=TaskKind.IO),
            PlanStep(name="b", goal="gb", role="execution", kind=TaskKind.IO,
                     depends_on=["a"],
                     payload={"retries": 2, "compensate": "undo b"}),
        ]
        from nomorals.missions.runner import _topo_levels

        text = render_plan_text(_topo_levels(steps))
        self.assertIn("level 0", text)
        self.assertIn("• a [research]", text)
        self.assertIn("← a", text)
        self.assertIn("retries=2", text)
        self.assertIn("compensate", text)
        self.assertEqual(render_plan_text([]), "(empty plan)")

    # ── heartbeat ───────────────────────────────────────────────────

    def test_heartbeat_progress(self):
        runner = self._runner([])
        mission = self.store.create_new("hb mission")
        self.assertTrue(runner.heartbeat_progress(mission.id, "4/10 files"))
        hb = self.store.get(mission.id).state["heartbeat"]
        self.assertEqual(hb["progress"], "4/10 files")
        self.assertIn("pid", hb)
        self.assertFalse(runner.heartbeat_progress("missing-id", "x"))

    def test_schedule_timeout_policy_parsed(self):
        from nomorals.missions.runner import _step_policy

        step = PlanStep(name="q", goal="g", role="execution",
                        kind=TaskKind.IO,
                        payload={"schedule_timeout_s": 30})
        self.assertEqual(_step_policy(step)["schedule_timeout_s"], 30.0)


class GoldenSweepTests(unittest.TestCase):
    def setUp(self):
        self.db = make_db(self)
        tmp = tempfile.mkdtemp(prefix="golden-sweep-")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        self.workdir = tmp

    def test_registry_has_five_missions(self):
        missions = list_golden_missions()
        self.assertEqual(
            {m["key"] for m in missions},
            {"research_write_verify", "build_test_fix", "audit_remediate_rescan",
             "crash_no_dup", "saga_undo"},
        )

    def test_crash_no_dup_kill_and_resume(self):
        runner = GoldenRunner(self.db, workdir_root=self.workdir)
        original = runner._execute_step
        calls = []

        def killing_execute(mission, step, ctx, outputs):
            report = original(mission, step, ctx, outputs)
            calls.append(step.name)
            if len(calls) == 1:
                runner.kill("simulated SIGKILL after first step")
            return report

        runner._execute_step = killing_execute
        result = runner.run("crash_no_dup")
        self.assertEqual(result.status, MissionStatus.PAUSED)
        # resume in a fresh runner (a fresh process in production)
        runner2 = GoldenRunner(self.db, workdir_root=self.workdir)
        resumed = runner2.resume(result.mission_id)
        self.assertTrue(resumed.ok, resumed.error)
        self.assertEqual(resumed.status, MissionStatus.DONE)
        log = Path(self.workdir) / result.mission_id / "side_effects.log"
        lines = log.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1, "completed step re-executed!")

    def test_saga_undo_compensates_reverse(self):
        runner = GoldenRunner(self.db, workdir_root=self.workdir)
        result = runner.run("saga_undo")
        self.assertFalse(result.ok)
        self.assertEqual(result.status, MissionStatus.FAILED)
        self.assertEqual([c["step"] for c in result.compensations],
                         ["create_b", "create_a"])
        self.assertTrue(all(c["ok"] for c in result.compensations))
        workdir = Path(self.workdir) / result.mission_id
        self.assertFalse((workdir / "a.txt").exists())
        self.assertFalse((workdir / "b.txt").exists())

    def test_normalize_output(self):
        raw = {"facts": ["a"], "seconds": 1.2, "pid": 123,
               "nested": {"created_at": 1.0, "value": 7},
               "items": [{"finished_at": 2.0, "n": 1}]}
        clean = normalize_output(raw)
        self.assertEqual(clean, {"facts": ["a"], "nested": {"value": 7},
                                 "items": [{"n": 1}]})
        self.assertEqual(normalize_output("plain"), "plain")

    def test_check_regression(self):
        ok = GoldenResult(mission_id="1", key="saga_undo", ok=True,
                          status="done")
        bad = GoldenResult(mission_id="2", key="saga_undo", ok=False,
                           status="failed", error="boom")
        report = check_regression([ok, bad])
        self.assertEqual(report["saga_undo"]["pass_rate"], 0.5)
        self.assertTrue(report["saga_undo"]["regression"])
        clean = check_regression([ok])
        self.assertFalse(clean["saga_undo"]["regression"])

    def test_render_golden_report(self):
        result = GoldenResult(
            mission_id="abc123", key="saga_undo", ok=False,
            status="failed", seconds=3.0,
            steps=[{"step": "create_a", "ok": True, "detail": "created",
                    "seconds": 0.1, "repaired": False}],
            compensations=[{"step": "create_a", "ok": True,
                             "detail": "removed"}],
            error="boom always fails")
        report = render_golden_report(result)
        self.assertIn("golden:saga_undo", report)
        self.assertIn("┌", report)
        self.assertIn("↩ create_a: undone", report)


if __name__ == "__main__":
    unittest.main()
