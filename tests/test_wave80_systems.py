"""Wave 80 systems: the orchestrator run journal + telemetry surface
(nm orchestrator report).  All hermetic — no model, no network."""

from __future__ import annotations

import json
import tempfile
import time
import types
import unittest

from nomorals.agents.context import build_context
from nomorals.agents.orchestrator import (MasterOrchestrator, OrchestrationResult,
                                          Plan, PlanStep)
from nomorals.agents.skills import SkillLibrary
from nomorals.core.config import Settings


class _JournalBase(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-w80-")
        self.context = build_context(Settings(home=self.home,
                                              reasoning_mode="off"))
        self.context.__enter__()
        self.orch = MasterOrchestrator(self.context, max_steps=4)

    def tearDown(self):
        self.context.__exit__(None, None, None)


class RunJournalTests(_JournalBase):
    def test_run_is_journaled(self):
        self.orch.run("count things", reflect=False, handlers={
            "research": lambda t: {"found": "3"},
            "execution": lambda t: {"done": True},
            "critic": lambda t: {"verdict": "ok"},
        })
        runs = MasterOrchestrator.runs(self.context, limit=5)
        self.assertEqual(len(runs), 1)
        entry = runs[0]
        self.assertEqual(entry["total"], 3)
        self.assertEqual(entry["done"], 3)
        self.assertTrue(entry["ok"])
        self.assertEqual(len(entry["routing"]), 3)
        self.assertTrue(all(r["chosen"] for r in entry["routing"]))

    def test_journal_is_newest_first_and_bounded(self):
        def _fake_result(i: int):
            return types.SimpleNamespace(
                report=types.SimpleNamespace(total=1, done=1, failed=0),
                ok=True, score=0.9, seconds=0.1)

        graph = None
        for i in range(105):
            self.orch._journal_run(graph, _fake_result(i), f"goal {i}",
                                   [], [], [])
        runs = MasterOrchestrator.runs(self.context, limit=10)
        rows = self.context.db.query_one(
            "SELECT value FROM kv_store WHERE key='orchestrator.runs'")
        journal = json.loads(rows["value"])
        self.assertLessEqual(len(journal), MasterOrchestrator._RUNS_BOUND)
        self.assertEqual(runs[0]["goal"], "goal 104")
        self.assertEqual(runs[1]["goal"], "goal 103")

    def test_pivot_recovery_is_journaled(self):
        # ledger makes course_correct pivot; the retry succeeds
        db = self.context.db
        for _ in range(2):
            db.execute(
                "INSERT INTO failures (id, source, summary, error, family, "
                "lesson, ts) VALUES (?,?,?,?,?,?,?)",
                (f"j-{time.time_ns()}", "orchestrator",
                 "dothing failed unhandled runtimeerror boom",
                 "boom", "dothing", "", time.time()))
        state = {"n": 0}

        def flaky(task):
            state["n"] += 1
            if state["n"] == 1:
                raise RuntimeError("boom")
            return {"ok": True}

        plan = Plan(goal="g", steps=[PlanStep(name="dothing",
                                              goal="do it")])
        self.orch.run("g", plan=plan, reflect=False,
                      handlers={"execution": flaky})
        runs = MasterOrchestrator.runs(self.context, limit=1)
        self.assertEqual(len(runs), 1)
        pivots = runs[0]["pivots"]
        self.assertEqual(len(pivots), 1)
        self.assertEqual(pivots[0]["task"], "dothing")
        self.assertTrue(pivots[0]["recovered"])
        self.assertTrue(pivots[0]["pivot"])


class TelemetryTests(_JournalBase):
    def _seed_skills(self):
        lib = SkillLibrary(self.context.db)
        proven = lib.save("route:research", kind="routing",
                          description="Routing experience for the research role")
        for _ in range(4):
            lib.record_use(proven.id, success=True, task="t", outcome="ok")
        chronic = lib.save("route:vision", kind="routing",
                           description="Routing experience for the vision role")
        for _ in range(4):
            lib.record_use(chronic.id, success=False, task="t", outcome="bad")
        winner = lib.save("verdict:critic", kind="routing",
                          description="Conflict-arbitration record for the critic role")
        lib.record_use(winner.id, success=True, task="arbitration:verdict",
                       outcome="verdict")
        lib.record_use(winner.id, success=False, task="arbitration:answer",
                       outcome="answer")

    def test_empty_context_is_graceful(self):
        data = self.orch.telemetry()
        self.assertEqual(data["summary"]["runs"], 0)
        self.assertIsNone(data["summary"]["ok_rate"])
        self.assertEqual(data["routing"], [])
        self.assertEqual(data["pivots"], [])

    def test_telemmetry_aggregates_every_surface(self):
        self._seed_skills()
        self.orch.run("find the answer", reflect=False, handlers={
            "research": lambda t: {"found": "42"},
            "execution": lambda t: {"done": True},
            "critic": lambda t: {"verdict": "ok"},
        })
        data = self.orch.telemetry()

        self.assertEqual(data["summary"]["runs"], 1)
        self.assertEqual(data["summary"]["ok_rate"], 1.0)

        routing = {r["name"]: r for r in data["routing"]}
        self.assertIn("route:research", routing)
        self.assertEqual(routing["route:research"]["prior"], "proven (+0.2)")
        self.assertEqual(routing["route:vision"]["prior"], "chronic (-0.15)")

        verdicts = {v["name"]: v for v in data["verdicts"]}
        self.assertEqual(verdicts["verdict:critic"]["wins"], 1)
        self.assertEqual(verdicts["verdict:critic"]["losses"], 1)

    def test_pivot_and_conflict_history(self):
        # a conflicting pair: worker vs critic
        plan = Plan(goal="g", steps=[
            PlanStep(name="worker", goal="claim", role="execution"),
            PlanStep(name="critic", goal="check", role="critic"),
        ])
        self.orch.run("g", plan=plan, reflect=False, handlers={
            "execution": lambda t: {"verdict": "yes"},
            "critic": lambda t: {"verdict": "no"},
        })
        data = self.orch.telemetry()
        self.assertEqual(data["summary"]["conflicts"], 1)
        conflict = data["conflicts"][0]
        self.assertEqual(conflict["winner"], "critic")
        self.assertEqual(conflict["field"], "verdict")
        self.assertIn("run_ts", conflict)


if __name__ == "__main__":
    unittest.main()
