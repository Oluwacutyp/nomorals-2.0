"""Wave 81 systems: mission ops report, orchestrator what-if simulate,
reasoning audit timeline.  All hermetic — no model, no network."""

from __future__ import annotations

import contextlib
import io
import tempfile
import time
import types
import unittest
from unittest import mock

from nomorals.agents.base import AgentResult
from nomorals.agents.context import build_context
from nomorals.agents.orchestrator import MasterOrchestrator, Plan, PlanStep
from nomorals.core.config import Settings
from nomorals.missions import MissionRunner, MissionStatus, MissionStore


class _Base(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-w81-")
        self.context = build_context(Settings(home=self.home,
                                              reasoning_mode="off"))
        self.context.__enter__()

    def tearDown(self):
        self.context.__exit__(None, None, None)


# ── mission ops report ─────────────────────────────────────────────────────

class OpsReportTests(_Base):
    def _store(self):
        return MissionStore(self.context.db)

    def test_ops_report_shows_pivots_and_failures(self):
        class _Fake:
            def run(self, prompt):
                if "[PIVOT]" in str(prompt):
                    return AgentResult(agent_id="f", role="research",
                                       output={"text": "ok"}, ok=True)
                return AgentResult(agent_id="f", role="research", output="",
                                   ok=False, error="ValueError: wrong approach")

        store = self._store()
        runner = MissionRunner(self.context, store=store)
        with mock.patch("nomorals.agents.roles.build_agent",
                        lambda role, **kw: _Fake()):
            mission = store.create_new("recover the card")
            mission.state["plan"] = [{
                "name": "gather", "goal": "find the card", "role": "research",
                "kind": "io", "depends_on": []}]
            store.save(mission)
            result = runner.run(mission, max_iterations=2, reflect=False)
        self.assertEqual(result.status, MissionStatus.DONE)

        report = runner.ops_report(limit=10)
        row = next(m for m in report["missions"]
                   if m["id"] == mission.id)
        self.assertEqual(row["status"], MissionStatus.DONE)
        self.assertEqual(row["pivots"], 1)
        self.assertEqual(row["fail_counts"], {"gather": 1})
        self.assertEqual(row["completed"], 1)
        self.assertEqual(row["plan_steps"], 1)
        self.assertIn("totals", report)
        self.assertIn("stuck", report)
        self.assertIn("resumable", report)

    def test_stuck_running_mission_is_flagged(self):
        store = self._store()
        runner = MissionRunner(self.context, store=store)
        mission = store.create_new("stuck worker")
        mission.status = MissionStatus.RUNNING
        store.save(mission)
        # age the row past the stuck threshold (save() stamps now)
        self.context.db.execute(
            "UPDATE missions SET updated_at = ? WHERE id = ?",
            (time.time() - MissionRunner.STUCK_AFTER_SECONDS - 60,
             mission.id))

        report = runner.ops_report(limit=10)
        self.assertIn(mission.id, report["stuck"])
        row = next(m for m in report["missions"] if m["id"] == mission.id)
        self.assertEqual(row["status"], MissionStatus.RUNNING)

    def test_empty_store_is_graceful(self):
        runner = MissionRunner(self.context, store=self._store())
        report = runner.ops_report(limit=5)
        self.assertEqual(report["missions"], [])
        self.assertEqual(report["stuck"], [])


# ── orchestrator what-if ───────────────────────────────────────────────────

class SimulateTests(_Base):
    def setUp(self):
        super().setUp()
        self.orch = MasterOrchestrator(self.context, max_steps=4)

    def test_simulate_plans_routes_and_checks(self):
        out = self.orch.simulate("build a backup script")
        self.assertEqual(len(out["steps"]), 3)
        self.assertEqual(out["parallelism"]["steps"], 3)
        for step in out["steps"]:
            self.assertTrue(step["agent"].endswith("Agent"))
            self.assertTrue(step["proceed"])
            self.assertIn("level", step)
        self.assertEqual(out["would_abort"], [])

    def test_simulate_flags_hard_risks_before_running(self):
        # the plan carries a step whose payload declares a hard risk
        plan = Plan(goal="g", steps=[
            PlanStep(name="danger", goal="wipe the database",
                     role="execution",
                     payload={"hard_risks": ["hard risk: untested code"]}),
        ])
        with mock.patch.object(self.orch, "plan", return_value=plan):
            out = self.orch.simulate("wipe the database")
        self.assertIn("danger", out["would_abort"])
        danger = next(s for s in out["steps"] if s["name"] == "danger")
        self.assertFalse(danger["proceed"])
        self.assertTrue(any(r.startswith("hard risk:") for r in danger["risks"]))

    def test_simulate_executes_nothing(self):
        self.orch.simulate("count the files")
        self.assertEqual(MasterOrchestrator.runs(self.context), [])
        rows = self.context.db.query(
            "SELECT COUNT(*) AS n FROM failures")
        self.assertEqual(rows[0]["n"], 0)
        # no blackboard task.* entries — nothing ran
        posts = self.orch.blackboard.keys()
        self.assertFalse(any(k.startswith("task.") for k in posts))


# ── reasoning audit ────────────────────────────────────────────────────────

class ReasonAuditTests(_Base):
    def _args(self, **kw):
        base = dict(audit=True, audit_limit=30, json=False, goal=[],
                    context="", traces=0, challenge="", strategy="auto",
                    depth=1, tools=False, eval=False, limit=0)
        base.update(kw)
        return types.SimpleNamespace(**base)

    def test_audit_merges_all_three_journals_newest_first(self):
        from nomorals.cli import _cmd_reason  # noqa — real handler under test
        from nomorals.agents.reasoning import ReasoningAgent

        agent = ReasoningAgent(self.context)
        # journal an old think trace, then a mid-task check, then a
        # self-challenge — the audit must come back newest first
        class _R:
            strategy = "cot"
            answer = "42"
            confidence = 0.9
            trace = [1, 2]
        agent.record_think_trace("why is it slow?", _R())
        time.sleep(0.01)
        agent.mid_task_check("run a big migration", log=True)
        time.sleep(0.01)
        agent.self_challenge("This is definitely the cause")

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = _cmd_reason(self._args(), self.context)
        self.assertEqual(rc, 0)
        text = out.getvalue()
        self.assertIn("REASONING AUDIT", text)
        i_challenge = text.find("challenge")
        i_midtask = text.find("mid-task")
        i_think = text.find("think")
        self.assertGreater(i_challenge, -1)
        self.assertGreater(i_midtask, -1)
        self.assertGreater(i_think, -1)
        # newest first: the challenge (recorded last) appears first
        self.assertLess(i_challenge, i_midtask)
        self.assertLess(i_midtask, i_think)

    def test_audit_json_shape(self):
        from nomorals.cli import _cmd_reason
        from nomorals.agents.reasoning import ReasoningAgent

        agent = ReasoningAgent(self.context)
        agent.mid_task_check("install a package", log=True)
        agent.self_challenge("always works")

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = _cmd_reason(self._args(json=True), self.context)
        self.assertEqual(rc, 0)
        import json as _json
        data = _json.loads(out.getvalue())
        kinds = {e["kind"] for e in data["events"]}
        self.assertIn("mid-task", kinds)
        self.assertIn("self_challenge", kinds)
        self.assertGreaterEqual(len(data["events"]), 2)

    def test_audit_empty_is_graceful(self):
        from nomorals.cli import _cmd_reason
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = _cmd_reason(self._args(), self.context)
        self.assertEqual(rc, 0)
        self.assertIn("no reasoning activity", out.getvalue())


if __name__ == "__main__":
    unittest.main()
