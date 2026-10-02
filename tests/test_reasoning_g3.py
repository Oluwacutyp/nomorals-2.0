"""Wave G3 — reasoning reliability.

1. MID-FLIGHT RE-EVALUATION TRIGGERS: scripted sideways plans (hard
   failures, error-carrying results, premise-contradicting directives)
   must actually call ``reevaluate()`` and revise/abort — never blindly
   continue.  Calibration is pinned on both sides: dominant failures
   abort, minority failures do not.
2. SINGLE PRIMARY PATH UNDER CONCURRENT LOAD: N concurrent identical
   requests at the dispatch path each wake exactly one orchestrator run
   — no duplicate agents per request, no cross-talk, no swallowed
   requests.  Dedup/coalescing is deliberately NOT implemented (see the
   docstring on TestSinglePrimaryPathConcurrent for why).
3. VISIBLE DEGRADATION: provider failures name the provider(s); tool
   failures name the step, role, and handler; both reach the chat-facing
   summary — never a bare "something went wrong".

Hermetic: no model, no network, no real DB.  Run with TMPDIR=/var/tmp.
"""
from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace

from nomorals.agents import coremind as cm
from nomorals.agents.blackboard import Blackboard
from nomorals.agents.orchestrator import (
    MasterOrchestrator,
    Plan,
    PlanStep,
)
from nomorals.core.tasks import TaskKind
from nomorals.llm.base import LLMProvider, LLMResponse
from nomorals.llm.router import LLMRouter


# ── fakes ──────────────────────────────────────────────────────────────────

class FakeSettings:
    def resolve(self, key):
        raise RuntimeError("no settings in tests")


class FakeContext:
    def __init__(self, router=None):
        self.settings = FakeSettings()
        self.router = router
        self.memory = None
        self.db = None
        self.tools = None
        self.blackboard = None
        self.emitted = []

    def emit(self, event, **kw):
        self.emitted.append((event, kw))


def _make_orch(ctx, **kw):
    from nomorals.agents.runtime import HybridExecutor
    from nomorals.agents.supervisor import RestartPolicy, Supervisor

    executor = HybridExecutor(threads=4, use_processes=False, max_in_flight=1)
    supervisor = Supervisor(
        policy=RestartPolicy(backoff_base=0.0, max_restarts=1))
    return MasterOrchestrator(ctx, executor=executor,
                              supervisor=supervisor, **kw)


def _plan(*specs):
    steps = []
    for spec in specs:
        deps = spec[2] if len(spec) > 2 else []
        steps.append(PlanStep(name=spec[0], goal=spec[1],
                              role="execution", depends_on=deps))
    return Plan(goal="g3 test goal", steps=steps)


def _ok_handler(task):
    return {"ok": True, "task": task.name}


class _FailProvider(LLMProvider):
    """Returns an error response instead of raising."""

    def __init__(self, name, error):
        super().__init__()
        self.name = name
        self._error = error

    @property
    def model_id(self):
        return f"{self.name}-model"

    def chat(self, messages, params=None, **kw):
        return LLMResponse(text="", model=self.model_id,
                           provider=self.name, error=self._error)


class _RaiseProvider(LLMProvider):
    """Raises instead of answering."""

    def __init__(self, name):
        super().__init__()
        self.name = name

    @property
    def model_id(self):
        return f"{self.name}-model"

    def chat(self, messages, params=None, **kw):
        raise RuntimeError("connection refused")


def _failing_router():
    router = LLMRouter()
    router.add(_FailProvider("hf_serverless", "ModelError: 503"), primary=True)
    router.add(_RaiseProvider("groq"))
    return router


# ── 1. mid-flight re-evaluation triggers ───────────────────────────────────

class TestReevaluateTriggers(unittest.TestCase):
    def _spy_reevaluate(self, orch):
        calls = []
        orig = MasterOrchestrator.reevaluate

        def spy(inner_self, goal, graph, **kw):
            trigger = kw.get("trigger")
            calls.append(trigger.name if trigger is not None else None)
            return orig(inner_self, goal, graph, **kw)

        orch.reevaluate = spy.__get__(orch, MasterOrchestrator)
        return calls

    def test_reevaluate_called_and_aborts_sideways_plan(self):
        """A plan going sideways must trip reevaluate() mid-flight and the
        run must abort — remaining steps never run (no blind continue)."""
        ctx = FakeContext()
        orch = _make_orch(ctx)
        calls = self._spy_reevaluate(orch)
        ran = []

        def handler(task):
            ran.append(task.name)
            if task.name in ("s1", "s2"):
                raise RuntimeError(f"{task.name} blew up")
            return {"ok": True}

        plan = _plan(("s1", "one"), ("s2", "two"),
                     ("s3", "three"), ("s4", "four"))
        result = orch.run("sideways", plan=plan,
                          handlers={"execution": handler}, reflect=False)

        # reevaluate() was genuinely invoked mid-flight, grounded in the
        # failing trigger tasks — not just recorded after the fact.
        self.assertEqual(calls, ["s1", "s2"])
        aborts = [e for e in result.reevaluations if e["action"] == "abort"]
        self.assertEqual(len(aborts), 1)
        self.assertIn("2 of 2", aborts[0]["reason"])
        # the rest of the plan never ran: aborted, not blindly continued
        self.assertNotIn("s3", ran)
        self.assertNotIn("s4", ran)
        self.assertEqual(result.report.cancelled, 2)
        self.assertFalse(result.ok)

    def test_error_carrying_results_trigger_cascade(self):
        """A step whose result carries {"error": ...} failed — two of them
        dominating the settled set must abort the plan, and the mission
        must not report ok."""
        ctx = FakeContext()
        orch = _make_orch(ctx)
        calls = self._spy_reevaluate(orch)

        def handler(task):
            if task.name in ("s1", "s2"):
                return {"error": f"tool exploded on {task.name}"}
            return {"ok": True}

        plan = _plan(("s1", "one"), ("s2", "two"))
        result = orch.run("softfail", plan=plan,
                          handlers={"execution": handler}, reflect=False)

        self.assertEqual(calls, ["s1", "s2"])
        aborts = [e for e in result.reevaluations if e["action"] == "abort"]
        self.assertEqual(len(aborts), 1)
        self.assertIn("2 of 2", aborts[0]["reason"])
        # soft failures are real failures: counted, retried, not ok
        self.assertEqual(result.report.failed, 2)
        self.assertFalse(result.ok)
        self.assertIn("s1", result.report.failures)
        self.assertIn("s2", result.report.failures)

    def test_premise_contradiction_revises_plan(self):
        """A step result contradicting the plan premise (explicit
        revise_plan directive) must call reevaluate() with that trigger
        and drop the contradicted steps."""
        ctx = FakeContext()
        orch = _make_orch(ctx)
        calls = self._spy_reevaluate(orch)

        def handler(task):
            if task.name == "a":
                return {"revise_plan": {"drop": ["b"],
                                        "note": "b contradicts the findings"}}
            return {"ok": True, "task": task.name}

        plan = _plan(("a", "first"),
                     ("b", "second", ["a"]),
                     ("c", "third", ["a"]))
        result = orch.run("contradiction", plan=plan,
                          handlers={"execution": handler}, reflect=False)

        self.assertIn("a", calls)  # reevaluate() saw the trigger task
        revises = [e for e in result.reevaluations
                   if e["action"] == "revise"]
        self.assertEqual(len(revises), 1)
        self.assertEqual(revises[0]["affected"], ["b"])
        self.assertIn("contradicts", revises[0]["reason"])
        self.assertEqual(result.report.skipped, 1)
        self.assertEqual(result.report.done, 2)

    def test_healthy_plan_never_triggers_action(self):
        """Checkpoints still run on a healthy plan, but every decision is
        'continue' — nothing is revised, trimmed, or aborted, and nothing
        is broadcast."""
        ctx = FakeContext()
        orch = _make_orch(ctx)
        calls = self._spy_reevaluate(orch)
        plan = _plan(("a", "one"), ("b", "two", ["a"]))
        result = orch.run("healthy", plan=plan,
                          handlers={"execution": _ok_handler}, reflect=False)

        self.assertTrue(calls)  # checkpoints ran
        self.assertTrue(result.reevaluations)
        self.assertTrue(all(e["action"] == "continue"
                            for e in result.reevaluations))
        bus = [e for e in ctx.emitted if e[0] == "plan.reevaluated"]
        self.assertEqual(bus, [])
        self.assertTrue(result.ok)


# ── calibration: both sides of the cascade rule ────────────────────────────

class TestCascadeCalibration(unittest.TestCase):
    def test_dominant_failures_abort(self):
        """2 failures dominating the settled set (2/2) abort the plan."""
        ctx = FakeContext()
        orch = _make_orch(ctx)
        ran = []

        def handler(task):
            ran.append(task.name)
            if task.name in ("s1", "s2"):
                raise RuntimeError("boom")
            return {"ok": True}

        plan = _plan(("s1", "one"), ("s2", "two"),
                     ("s3", "three"), ("s4", "four"))
        result = orch.run("dominant", plan=plan,
                          handlers={"execution": handler}, reflect=False)

        aborts = [e for e in result.reevaluations if e["action"] == "abort"]
        self.assertEqual(len(aborts), 1)
        self.assertNotIn("s3", ran)
        self.assertNotIn("s4", ran)

    def test_minority_failures_do_not_abort(self):
        """2 failures that are a minority of settled work (2/5) must NOT
        abort — killing a mostly-healthy plan would be firing pointlessly."""
        ctx = FakeContext()
        orch = _make_orch(ctx)
        ran = []

        def handler(task):
            ran.append(task.name)
            if task.name in ("s1", "s5"):
                raise RuntimeError("boom")
            return {"ok": True, "task": task.name}

        plan = _plan(("s1", "one"), ("s2", "two"), ("s3", "three"),
                     ("s4", "four"), ("s5", "five"))
        result = orch.run("minority", plan=plan,
                          handlers={"execution": handler}, reflect=False)

        aborts = [e for e in result.reevaluations if e["action"] == "abort"]
        self.assertEqual(aborts, [])
        # every step got its chance — the plan continued, not the abort
        for name in ("s1", "s2", "s3", "s4", "s5"):
            self.assertIn(name, ran)
        self.assertEqual(result.report.done, 3)
        self.assertEqual(result.report.failed, 2)
        self.assertEqual(result.report.cancelled, 0)


# ── 2. single primary path under concurrent load ───────────────────────────

class TestSinglePrimaryPathConcurrent(unittest.TestCase):
    """Why no dedup/coalescing: every inbound message is a distinct job —
    its own job_id, its own objective-memory entry, its own orchestrator
    run.  Two identical texts can be a legitimate repeat ("do it again"),
    and silently merging them would drop user intent with no audit trail.

    The invariant the design DOES guarantee — and these tests prove — is
    one-request-one-path: N concurrent requests wake exactly N primary
    paths.  Never 2N (no duplicate agents spawned for the same request),
    never fewer (no swallowed requests), and no shared-state cross-talk
    between them.
    """

    def _sync_mind(self):
        ctx = FakeContext()
        mind = cm.CoreMind(ctx)
        # run dispatch jobs on the calling thread: deterministic, and it
        # keeps the test about the dispatch path, not thread scheduling.
        mind._send_async = (
            lambda chat_key, job, job_id, note, kind="job": job())
        return mind

    def test_concurrent_identical_requests_one_path_each(self):
        import nomorals.agents.orchestrator as orch_mod

        lock = threading.Lock()
        instantiations = 0
        run_goals = []

        class FakeOrch:
            def __init__(self, context):
                nonlocal instantiations
                with lock:
                    instantiations += 1

            def run(self, goal, reflect=False):
                with lock:
                    run_goals.append(goal)
                return SimpleNamespace(
                    ok=True,
                    plan=SimpleNamespace(
                        steps=[SimpleNamespace(name="s1")], plan_error=""),
                    report=SimpleNamespace(results={"s1": "x"},
                                           failures={}),
                    reevaluations=[])

        orig = orch_mod.MasterOrchestrator
        orch_mod.MasterOrchestrator = FakeOrch
        try:
            mind = self._sync_mind()
            intent = cm.Intent("multi", 0.8, target="do x and y",
                               route="orchestrator", why="test")
            errors = []
            n = 12

            def one(i):
                try:
                    mind._dispatch(intent, f"chat:{i}", None)
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [threading.Thread(target=one, args=(i,))
                       for i in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            orch_mod.MasterOrchestrator = orig

        self.assertEqual(errors, [])
        # exactly one primary path per request: no duplicates, no drops
        self.assertEqual(instantiations, n)
        self.assertEqual(len(run_goals), n)
        self.assertEqual(run_goals, ["do x and y"] * n)
        # every request became its own job — no cross-talk, no merging
        job_ids = [j["id"] for j in mind._jobs]
        self.assertEqual(len(job_ids), n)
        self.assertEqual(len(set(job_ids)), n)

    def test_concurrent_runs_share_no_state(self):
        """Separate orchestrator instances sharing one context, one
        blackboard, and one executor: concurrent runs must not corrupt
        each other's per-run state (goal, checkpoints, reevaluations)."""
        from nomorals.agents.runtime import HybridExecutor

        ctx = FakeContext()
        ctx.blackboard = Blackboard()
        executor = HybridExecutor(threads=8, use_processes=False,
                                  max_in_flight=2)
        n = 8
        goals = [f"concurrent goal {i}" for i in range(n)]
        results = {}
        errors = []
        barrier = threading.Barrier(n)

        def one(i):
            try:
                goal = goals[i]
                orch = MasterOrchestrator(ctx, executor=executor)
                barrier.wait(timeout=30)  # maximize overlap

                def handler(task, _goal=goal):
                    return {"ok": True, "goal": _goal, "task": task.name}

                plan = _plan(("a", f"step a for {goal}"),
                             ("b", f"step b for {goal}", ["a"]))
                results[i] = orch.run(goal, plan=plan,
                                      handlers={"execution": handler},
                                      reflect=False)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=one, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(results), n)
        for i, goal in enumerate(goals):
            result = results[i]
            self.assertTrue(result.ok)
            # each run's outcome is its own: goal, steps, and every
            # checkpoint decision reference only this run's work
            self.assertEqual(result.goal, goal)
            self.assertEqual(sorted(result.report.results), ["a", "b"])
            for ev in result.reevaluations:
                self.assertIn(ev["at_task"], ("a", "b", ""))


# ── 3. visible degradation ─────────────────────────────────────────────────

class TestVisibleDegradation(unittest.TestCase):
    def test_plan_error_names_failed_providers(self):
        """A dead model chain must name every provider that failed — the
        plan_error is the caller's only evidence of what went wrong."""
        ctx = FakeContext(router=_failing_router())
        orch = _make_orch(ctx)
        plan = orch.plan("do the thing")

        self.assertTrue(plan.steps, "fallback must still produce steps")
        self.assertTrue(plan.plan_error)
        self.assertIn("model call failed", plan.plan_error)
        self.assertIn("hf_serverless", plan.plan_error)
        self.assertIn("groq", plan.plan_error)

    def test_provider_failure_surfaces_to_chat_summary(self):
        """End to end: provider failure -> plan_error -> the chat-facing
        summary names the providers and flags the degraded plan."""
        ctx = FakeContext(router=_failing_router())
        orch = _make_orch(ctx)
        result = orch.run("degraded run", reflect=False)
        summary = cm._summarize_orchestration(result)

        self.assertIn("hf_serverless", summary)
        self.assertIn("groq", summary)
        self.assertIn("degraded plan", summary)

    def test_tool_failure_names_step_role_and_handler(self):
        """A tool blowing up mid-plan must name the step, the role, and
        the handler — and the reason must reach the chat summary."""
        ctx = FakeContext()
        orch = _make_orch(ctx)

        def boom(task):
            if task.name == "s2":
                raise RuntimeError("disk on fire")
            return {"ok": True}

        plan = _plan(("s1", "one"), ("s2", "two"))
        result = orch.run("toolboom", plan=plan,
                          handlers={"execution": boom}, reflect=False)

        self.assertFalse(result.ok)
        msg = result.report.failures["s2"]
        for needle in ("s2", "execution", "boom", "RuntimeError",
                       "disk on fire"):
            self.assertIn(needle, msg,
                          f"surfaced error must name {needle!r}: {msg!r}")
        summary = cm._summarize_orchestration(result)
        self.assertIn("[✗] s2", summary)
        self.assertIn("disk on fire", summary)

    def test_error_dict_result_fails_loudly(self):
        """A handler returning {"error": ...} fails the step — loudly and
        by name — instead of masquerading as a success."""
        ctx = FakeContext()
        orch = _make_orch(ctx)

        def flaky(task):
            if task.name == "s1":
                return {"error": "tool exploded"}
            return {"ok": True}

        plan = _plan(("s1", "one"), ("s2", "two"))
        result = orch.run("flaky", plan=plan,
                          handlers={"execution": flaky}, reflect=False)

        self.assertFalse(result.ok)
        self.assertEqual(result.report.failed, 1)
        self.assertEqual(result.report.done, 1)
        msg = result.report.failures["s1"]
        for needle in ("s1", "execution", "flaky", "tool exploded"):
            self.assertIn(needle, msg,
                          f"surfaced error must name {needle!r}: {msg!r}")
        # the error result never pollutes the aggregated answer
        self.assertNotIn("tool exploded", result.answer)


if __name__ == "__main__":
    unittest.main()
