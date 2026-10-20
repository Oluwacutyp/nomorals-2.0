"""Pass #52 — cognition + failure-knowledge + improvement measurement.

Pins the fixes for the defects found in the failure-learning chain:

1. Tool-registry failure-ledger rows were written stripped (no
   ``fingerprint``, no ``skill``), which made them invisible to the
   lesson-usefulness recurrence check and to skill-evolution detection.
2. ``FailureAnalyzer.collect()`` never read the ``failures`` ledger, so
   ``learn_from_all`` distilled lessons from coding_log/devon_memory
   only — the failures the system actually hits (tool errors) were
   recorded but never learned from. Its "already learned" seen-key
   could never match, so every call re-analyzed the full window.
3. Lesson usefulness credited "prevented" when the same failure had
   recurred through the registry path (a consequence of 1).
4. ``EvolutionAgent``'s benchmark regression gate recorded its
   first-ever baseline from the already-edited tree, so a regressed
   first edit passed and lowered the bar. The baseline is now primed
   pre-edit in ``apply()``.
5. ``failure_loop.apply_proposal`` trusted a ``sandbox_passed`` flag
   on the proposal dict without re-checking the fix text, so a
   forged/edited proposal with a destructive fix could be applied.
6. ``ReasoningAgent.record_outcome`` had no production caller; the
   orchestrator now feeds failed task attempts into the ledger at the
   course-correct consult point.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.storage.db import Database


def _ctx(**settings_kw):
    db = Database(":memory:")
    db.migrate()
    return SimpleNamespace(db=db, settings=SimpleNamespace(**settings_kw),
                           router=None)


class KeyBoom(Exception):
    """Exception whose str() carries the KeyError text, like a wrapped
    provider error surfacing through a tool."""


def _boom():
    raise KeyBoom("KeyError: 'api_key'")


def _registry_with_boom(context):
    from nomorals.tools.registry import ToolRegistry

    reg = ToolRegistry(context=context)
    reg.register("boom_tool", description="always fails", capability="")(_boom)
    return reg


class RegistryLedgerAttributionTests(unittest.TestCase):
    def test_tool_failure_row_carries_fingerprint_and_skill(self):
        from nomorals.agents.failure import failure_fingerprint

        ctx = _ctx()
        reg = _registry_with_boom(ctx)
        result = reg.call("boom_tool", actor="chat:test")
        self.assertFalse(result.ok)
        rows = ctx.db.query("SELECT * FROM failures")
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["skill"], "boom_tool")
        self.assertEqual(
            row["fingerprint"],
            failure_fingerprint("tool", "KeyError: 'api_key'", "", "boom_tool"))
        self.assertNotEqual(row["fingerprint"], "")

    def test_skill_evolution_detect_attributes_tool_failures(self):
        from nomorals.agents.skill_evolution import SkillEvolutionLoop

        ctx = _ctx(improvement=SimpleNamespace(mode="approval"))
        reg = _registry_with_boom(ctx)
        for _ in range(3):
            reg.call("boom_tool", actor="chat:test")
        candidates = SkillEvolutionLoop(ctx).detect()
        by_skill = {c["skill"]: c["count"] for c in candidates}
        self.assertEqual(by_skill.get("boom_tool"), 3)


class LearnFromLedgerTests(unittest.TestCase):
    def test_collect_includes_failures_ledger(self):
        from nomorals.agents.failure import FailureAnalyzer

        ctx = _ctx()
        reg = _registry_with_boom(ctx)
        reg.call("boom_tool", actor="chat:test")
        cases = FailureAnalyzer(ctx).collect(limit=10)
        tool_cases = [c for c in cases if c.source == "tool"]
        self.assertEqual(len(tool_cases), 1)
        self.assertEqual(tool_cases[0].skill, "boom_tool")
        self.assertIn("KeyError", tool_cases[0].error)

    def test_learn_from_all_distills_registry_failures_once(self):
        from nomorals.agents.failure import FailureAnalyzer

        ctx = _ctx()
        reg = _registry_with_boom(ctx)
        reg.call("boom_tool", actor="chat:test")
        analyzer = FailureAnalyzer(ctx)
        self.assertEqual(analyzer.learn_from_all(limit=10), 1)
        lessons = ctx.db.query("SELECT * FROM lessons")
        self.assertEqual(len(lessons), 1)
        # second run: the lesson fingerprint is known -> nothing re-learned
        self.assertEqual(analyzer.learn_from_all(limit=10), 0)
        self.assertEqual(
            ctx.db.query("SELECT COUNT(*) AS n FROM lessons")[0]["n"], 1)

    def test_collect_excludes_denial_telemetry(self):
        from nomorals.agents.failure import FailureAnalyzer
        from nomorals.agents.role_specs import record_tool_denial

        ctx = _ctx()
        record_tool_denial(ctx, "worker", "shell", "not in allowlist")
        cases = FailureAnalyzer(ctx).collect(limit=10)
        self.assertEqual(
            [c for c in cases if c.source == "role"], [])


class UsefulnessHonestyTests(unittest.TestCase):
    def _lesson_and_surfacing(self, ctx):
        from nomorals.agents.failure import FailureAnalyzer

        analyzer = FailureAnalyzer(ctx)
        out = analyzer.record("tool", "boom task", "KeyError: 'api_key'",
                              skill="boom_tool")
        self.assertTrue(out["learned"])
        self.assertTrue(analyzer.prevention_context("keyerror api_key"))
        return analyzer

    def test_recurrence_via_registry_is_not_credited_as_prevented(self):
        ctx = _ctx()
        analyzer = self._lesson_and_surfacing(ctx)
        reg = _registry_with_boom(ctx)
        reg.call("boom_tool", actor="chat:test")  # the failure recurs
        stats = analyzer.evaluate_usefulness(window_days=0.0)
        self.assertEqual(stats["evaluated"], 1)
        lesson = analyzer.recent(limit=1)[0]
        self.assertEqual(lesson.times_prevented, 0)
        self.assertEqual(lesson.usefulness, 0.0)

    def test_no_recurrence_still_credits_prevented(self):
        ctx = _ctx()
        analyzer = self._lesson_and_surfacing(ctx)
        stats = analyzer.evaluate_usefulness(window_days=0.0)
        self.assertEqual(stats["evaluated"], 1)
        lesson = analyzer.recent(limit=1)[0]
        self.assertEqual(lesson.times_prevented, 1)
        self.assertEqual(lesson.usefulness, 1.0)


class BenchmarkBaselinePrimingTests(unittest.TestCase):
    def _agent(self, ctx):
        from nomorals.agents.evolution import EvolutionAgent

        return EvolutionAgent(ctx)

    def _report(self, overall):
        import nomorals.agents.benchmark as bench

        class DS:
            def __init__(self, name, score):
                self.name, self.score = name, score

        class Rep:
            measurable = True

            def __init__(self, overall):
                self.overall = overall
                self.scores = {"reasoning": DS("reasoning", overall)}

        return Rep(overall)

    def _evolution_ctx(self):
        return _ctx(
            improvement=SimpleNamespace(mode="approval"),
            evolution=SimpleNamespace(benchmark="on"))

    def test_prime_records_pre_edit_baseline_and_gate_blocks_regression(self):
        import nomorals.agents.benchmark as bench

        ctx = self._evolution_ctx()
        agent = self._agent(ctx)
        with mock.patch.object(bench, "run_benchmark",
                               return_value=self._report(0.90)):
            agent._prime_benchmark_baseline()
        self.assertAlmostEqual(agent._load_benchmark_baseline(), 0.90)
        # the edited tree now scores 0.30: the gate must block, and the
        # baseline must remain the pre-edit 0.90 (not be re-recorded).
        with mock.patch.object(bench, "run_benchmark",
                               return_value=self._report(0.30)):
            gate = agent._benchmark_gate()
        self.assertIsNotNone(gate)
        self.assertFalse(gate["ok"])
        self.assertAlmostEqual(agent._load_benchmark_baseline(), 0.90)

    def test_prime_is_noop_when_baseline_exists_or_gate_off(self):
        import nomorals.agents.benchmark as bench

        ctx = self._evolution_ctx()
        agent = self._agent(ctx)
        agent._save_benchmark_baseline(0.5)
        with mock.patch.object(bench, "run_benchmark",
                               return_value=self._report(0.99)) as rb:
            agent._prime_benchmark_baseline()
        self.assertAlmostEqual(agent._load_benchmark_baseline(), 0.5)

        ctx2 = _ctx(improvement=SimpleNamespace(mode="approval"),
                    evolution=SimpleNamespace(benchmark="off"))
        agent2 = self._agent(ctx2)
        with mock.patch.object(bench, "run_benchmark",
                               return_value=self._report(0.99)):
            agent2._prime_benchmark_baseline()
        self.assertIsNone(agent2._load_benchmark_baseline())


class FailureLoopApplyGateTests(unittest.TestCase):
    CLUSTER = {"task_kind": "chat", "error_signature": "Boom: x",
               "failure_class": "network", "count": 5}

    def test_forged_flag_with_dangerous_fix_is_blocked(self):
        from nomorals.cognition.failure_loop import (
            apply_proposal, build_proposal)

        proposal = build_proposal(self.CLUSTER)
        proposal["suggested_fix"] = "run rm -rf / to clear the cache"
        proposal["sandbox_passed"] = True  # forged: no sandbox ran on this
        out = apply_proposal(proposal, approve=True)
        self.assertFalse(out["applied"])
        self.assertEqual(out["status"], "sandbox_required")
        self.assertIn("re-check", out["apply_blocked_reason"])

    def test_fix_edited_after_real_pass_is_blocked(self):
        from nomorals.cognition.failure_loop import (
            apply_proposal, build_proposal, sandbox_check)

        proposal = build_proposal(self.CLUSTER)
        result = sandbox_check(proposal)
        self.assertTrue(result["passed"])
        proposal["sandbox_passed"] = True
        proposal["suggested_fix"] = "harmless-looking; also DROP TABLE x"
        out = apply_proposal(proposal, approve=True)
        self.assertFalse(out["applied"])

    def test_legit_pass_still_applies(self):
        from nomorals.cognition.failure_loop import (
            apply_proposal, build_proposal, sandbox_check)

        proposal = build_proposal(self.CLUSTER,
                                  sandbox_result=sandbox_check(
                                      build_proposal(self.CLUSTER)))
        out = apply_proposal(proposal, approve=True)
        self.assertTrue(out["applied"])
        self.assertEqual(out["status"], "applied")


class OrchestratorOutcomeWiringTests(unittest.TestCase):
    def test_course_correct_pivot_records_failed_task(self):
        from nomorals.agents.orchestrator import MasterOrchestrator
        from nomorals.core.tasks import Task, TaskState

        db = Database(":memory:")
        db.migrate()
        ctx = SimpleNamespace(
            db=db, settings=SimpleNamespace(), router=None, executor=None,
            blackboard=None, emit=lambda *a, **k: None)
        orch = MasterOrchestrator(ctx)
        task = Task(name="step_one")
        task.state = TaskState.FAILED
        task.error = "KeyError: 'api_key'"
        task.attempts = 1
        orch._course_correct_pivot(task)
        rows = ctx.db.query(
            "SELECT * FROM failures WHERE source='orchestrator'")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["summary"], "step_one")
        self.assertIn("KeyError", rows[0]["error"])


if __name__ == "__main__":
    unittest.main()
