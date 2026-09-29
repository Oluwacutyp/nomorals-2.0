"""Wave 82 systems: ops alerting, simulate --go pinned execution,
audit → prevention-skill loop.  All hermetic — no model, no network."""

from __future__ import annotations

import json
import tempfile
import time
import types
import unittest
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.agents.orchestrator import MasterOrchestrator, Plan, PlanStep
from nomorals.agents.skills import SkillLibrary
from nomorals.core.config import Settings
from nomorals.missions import MissionRunner, MissionStatus, MissionStore


class _Base(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-w82-")
        self.context = build_context(Settings(home=self.home,
                                              reasoning_mode="off"))
        self.context.__enter__()

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def _notifications(self, kind=""):
        sql = "SELECT * FROM notifications"
        params: tuple = ()
        if kind:
            sql += " WHERE kind = ?"
            params = (kind,)
        return self.context.db.query(sql + " LIMIT 50", params)


# ── ops alerts ─────────────────────────────────────────────────────────────

class OpsAlertTests(_Base):
    def test_empty_system_is_quiet(self):
        from nomorals.agents.ops_alerts import scan_ops
        self.assertEqual(scan_ops(self.context), [])

    def test_stuck_mission_self_heals_then_pages_once(self):
        """Wave 83 escalation: first scan self-heals (no page), the
        still-stuck mission pages once, then goes quiet on cooldown."""
        from nomorals.agents.ops_alerts import _kv_set, ops_alerts

        store = MissionStore(self.context.db)
        mission = store.create_new("stuck worker")
        mission.status = MissionStatus.RUNNING
        store.save(mission)
        self.context.db.execute(
            "UPDATE missions SET updated_at = ? WHERE id = ?",
            (time.time() - MissionRunner.STUCK_AFTER_SECONDS - 60,
             mission.id))

        # wave 83: a fresh stuck mission gets one self-heal, not a page
        # (mocked: the attempt runs but the mission stays stuck)
        with mock.patch.object(MissionRunner, "self_heal",
                               return_value={"attempted": True,
                                             "status": "still_stuck"}):
            first = ops_alerts(self.context, heal_background=False)
        self.assertEqual(first["sent"], [])
        self.assertEqual(len(first["self_healed"]), 1)

        # pretend the heal did not save it and it is stuck AGAIN
        self.context.db.execute(
            "UPDATE missions SET updated_at = ? WHERE id = ?",
            (time.time() - MissionRunner.STUCK_AFTER_SECONDS - 60,
             mission.id))
        second = ops_alerts(self.context, heal_background=False)
        self.assertEqual(len(second["sent"]), 1)
        self.assertEqual(second["sent"][0]["kind"], "stuck_mission")
        rows = self._notifications("ops")
        self.assertEqual(len(rows), 1)
        self.assertIn("stuck", rows[0]["title"])

        # the same stuck mission does NOT page again within the cooldown
        third = ops_alerts(self.context, heal_background=False)
        self.assertEqual(third["sent"], [])
        self.assertEqual(len(third["cooldown"]), 1)
        self.assertEqual(len(self._notifications("ops")), 1)

    def test_chronic_role_is_found(self):
        from nomorals.agents.ops_alerts import scan_ops
        lib = SkillLibrary(self.context.db)
        skill = lib.save("route:vision", kind="routing",
                         description="Routing experience for the vision role")
        for _ in range(3):
            lib.record_use(skill.id, success=False, task="t", outcome="bad")
        findings = scan_ops(self.context)
        kinds = [f["kind"] for f in findings]
        self.assertIn("chronic_role", kinds)

    def test_unrecovered_pivot_is_found(self):
        from nomorals.agents.ops_alerts import scan_ops
        orch = MasterOrchestrator(self.context, max_steps=4)
        orch._journal_run(
            None,
            types.SimpleNamespace(
                report=types.SimpleNamespace(total=1, done=0, failed=1),
                ok=False, score=0.0, seconds=0.1),
            "fragile goal",
            [{"task": "doit", "from_role": "execution",
              "to_role": "execution", "pivot": "change approach",
              "recovered": False}],
            [], [])
        findings = scan_ops(self.context)
        kinds = [f["kind"] for f in findings]
        self.assertIn("unrecovered_pivot", kinds)


# ── simulate --go ──────────────────────────────────────────────────────────

class SimulateGoTests(_Base):
    def test_simulate_carries_the_executable_plan(self):
        orch = MasterOrchestrator(self.context, max_steps=4)
        out = orch.simulate("count the files")
        self.assertIsInstance(out["plan"], Plan)
        self.assertEqual(len(out["plan"].steps), len(out["steps"]))

    def test_go_refuses_aborting_plan_without_force(self):
        plan = Plan(goal="g", steps=[
            PlanStep(name="danger", goal="wipe the database",
                     role="execution",
                     payload={"hard_risks": ["hard risk: untested code"]}),
        ])
        orch = MasterOrchestrator(self.context, max_steps=4)
        with mock.patch.object(type(orch), "plan",
                               lambda self, goal, context_hint="": plan):
            out = orch.simulate("wipe the database")
        self.assertIn("danger", out["would_abort"])
        self.assertIs(out["plan"], plan)

    def test_go_executes_exactly_the_simulated_plan(self):
        # the plan simulate produced is the one that runs — the steps
        # execute with the plan's goals, and the result is a real run
        orch = MasterOrchestrator(self.context, max_steps=4)
        out = orch.simulate("count the files")
        seen_goals = []

        def spy(task):
            seen_goals.append(str(task.payload.get("goal", "")))
            return {"ok": True}

        result = orch.run(out["goal"], plan=out["plan"], reflect=False,
                          handlers={"research": spy, "execution": spy,
                                    "critic": spy})
        self.assertEqual(result.report.done, len(out["steps"]))
        self.assertEqual(len(seen_goals), len(out["steps"]))


# ── audit → prevention loop ────────────────────────────────────────────────

class PreventionLoopTests(_Base):
    def test_repeated_abort_becomes_prevention_skill(self):
        from nomorals.agents.reasoning import ReasoningAgent
        agent = ReasoningAgent(self.context)

        # two aborts on the same action family, slightly reworded
        # (journaled; the second one auto-mines)
        for text in ("drop the users table now",
                     "drop the users table now, please"):
            out = agent.mid_task_check(
                text, hard_risks=["hard risk: untested code"], log=True)
            self.assertFalse(out["proceed"])

        lib = SkillLibrary(self.context.db)
        skills = lib.list(kind="prevention", limit=20)
        self.assertTrue(any(s.name.startswith("prevent:drop") for s in skills))

        # the third check on a similar action now sees the memory
        third = agent.mid_task_check("drop the users table now again",
                                     log=False)
        self.assertTrue(any(r.startswith("repeated abort:")
                            for r in third["risks"]))

    def test_mine_is_idempotent(self):
        from nomorals.agents.reasoning import ReasoningAgent
        agent = ReasoningAgent(self.context)
        for _ in range(2):
            agent.mid_task_check("format the system disk",
                                 hard_risks=["hard risk: destructive"],
                                 log=True)
        first = agent.mine_preventions()
        second = agent.mine_preventions()
        names = [s["name"] for s in first] + [s["name"] for s in second]
        # same family, same skill — upserted, not duplicated
        self.assertEqual(len(set(names)), 1)
        lib = SkillLibrary(self.context.db)
        self.assertEqual(len(lib.list(kind="prevention", limit=20)), 1)

    def test_single_abort_does_not_learn(self):
        from nomorals.agents.reasoning import ReasoningAgent
        agent = ReasoningAgent(self.context)
        agent.mid_task_check("risky one-shot action",
                             hard_risks=["hard risk: untested"],
                             log=True)
        written = agent.mine_preventions()
        self.assertEqual(written, [])
        lib = SkillLibrary(self.context.db)
        self.assertEqual(lib.list(kind="prevention", limit=20), [])


if __name__ == "__main__":
    unittest.main()
