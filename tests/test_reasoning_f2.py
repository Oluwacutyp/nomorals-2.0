"""Wave F2 — reasoning / orchestrator.

- Mid-task re-evaluation: long runs checkpoint the plan mid-flight and
  can revise / trim / abort it with a recorded reason (never blindly
  continue).
- One primary path per request: a single request wakes exactly one
  execution path; specialists are only invoked when the primary path
  explicitly delegates.
- Observability: re-evaluations, path choices, and delegations are
  logged, emitted, persisted to the router telemetry, and queryable via
  ``nm mind`` / ``stats()``.

Hermetic: no model, no network, no real DB.  Run with TMPDIR=/var/tmp.
"""
from __future__ import annotations

import io
import sqlite3
import types
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

from nomorals.agents import coremind as cm
from nomorals.agents.orchestrator import (
    MasterOrchestrator,
    Plan,
    PlanStep,
    Reevaluation,
)
from nomorals.core.tasks import Task, TaskKind, TaskState
from nomorals.storage import router_telemetry as rt


# ── fakes ──────────────────────────────────────────────────────────────────

class FakeSettings:
    def resolve(self, key):
        raise RuntimeError("no settings in tests")


class FakeContext:
    def __init__(self):
        self.settings = FakeSettings()
        self.router = None
        self.memory = None
        self.db = None
        self.tools = None
        self.blackboard = None
        self.emitted = []

    def emit(self, event, **kw):
        self.emitted.append((event, kw))


class FakeDB:
    """Minimal sqlite stand-in: execute() + query() like the real wrapper."""

    def __init__(self):
        self.cx = sqlite3.connect(":memory:")
        self.cx.execute(
            "CREATE TABLE coremind_telemetry "
            "(key TEXT PRIMARY KEY, value TEXT, updated_at REAL)")

    def execute(self, sql, params=()):
        self.cx.execute(sql, params)
        self.cx.commit()

    def query(self, sql, params=()):
        cur = self.cx.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


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
    return Plan(goal="f2 test goal", steps=steps)


def _ok_handler(task):
    return {"ok": True, "task": task.name}


# ── 1. mid-flight re-evaluation ────────────────────────────────────────────

class TestReevaluation(unittest.TestCase):
    def test_abort_on_failure_cascade(self):
        ctx = FakeContext()
        ctx.db = FakeDB()
        orch = _make_orch(ctx)
        plan = _plan(("s1", "do one"), ("s2", "do two"),
                     ("s3", "do three"), ("s4", "do four"))

        def handler(task):
            if task.name in ("s1", "s2"):
                raise RuntimeError(f"{task.name} blew up")
            return {"ok": True, "task": task.name}

        result = orch.run("cascade", plan=plan,
                          handlers={"execution": handler}, reflect=False)

        self.assertFalse(result.ok)
        self.assertEqual(result.report.failed, 2)
        # the cascade abort cancelled the rest instead of running them
        self.assertEqual(result.report.cancelled, 2)
        aborts = [e for e in result.reevaluations
                  if e["action"] == "abort"]
        self.assertEqual(len(aborts), 1)
        abort = aborts[0]
        self.assertIn("2 of 2", abort["reason"])
        self.assertEqual(sorted(abort["affected"]), ["s3", "s4"])
        self.assertTrue(abort["at_task"])  # grounded in the triggering task
        # visible: event bus + persisted telemetry + stats()
        bus = [e for e in ctx.emitted
               if e[0] == "plan.reevaluated" and e[1]["action"] == "abort"]
        self.assertEqual(len(bus), 1)
        snap = rt.snapshot(ctx.db)
        self.assertEqual(snap["last_reevaluation"]["action"], "abort")
        self.assertIn("2 of 2", snap["last_reevaluation"]["reason"])
        self.assertGreaterEqual(orch.stats()["reevaluations"]["aborts"], 1)

    def test_no_abort_below_cascade_threshold(self):
        ctx = FakeContext()
        orch = _make_orch(ctx)
        plan = _plan(("s1", "do one"), ("s2", "do two"),
                     ("s3", "do three"))

        def handler(task):
            if task.name == "s1":
                raise RuntimeError("s1 blew up")
            return {"ok": True, "task": task.name}

        result = orch.run("one failure", plan=plan,
                          handlers={"execution": handler}, reflect=False)
        aborts = [e for e in result.reevaluations
                  if e["action"] == "abort"]
        self.assertEqual(aborts, [])
        self.assertEqual(result.report.done, 2)

    def test_trim_superseded_step(self):
        ctx = FakeContext()
        orch = _make_orch(ctx)
        plan = _plan(("a", "Fetch the weather"),
                     ("b", "fetch the  WEATHER!", ["a"]))
        result = orch.run("dupes", plan=plan,
                          handlers={"execution": _ok_handler}, reflect=False)

        trims = [e for e in result.reevaluations if e["action"] == "trim"]
        self.assertEqual(len(trims), 1)
        self.assertEqual(trims[0]["affected"], ["b"])
        self.assertIn("superseded", trims[0]["reason"])
        self.assertEqual(result.report.skipped, 1)
        self.assertTrue(result.ok)

    def test_revise_via_handler_directive(self):
        ctx = FakeContext()
        orch = _make_orch(ctx)
        plan = _plan(("a", "first"),
                     ("b", "second", ["a"]),
                     ("c", "third", ["a"]))

        def handler(task):
            if task.name == "a":
                return {"revise_plan":
                        {"drop": ["b"], "note": "b is obsolete"}}
            return {"ok": True, "task": task.name}

        result = orch.run("directive", plan=plan,
                          handlers={"execution": handler}, reflect=False)

        revises = [e for e in result.reevaluations
                   if e["action"] == "revise"]
        self.assertEqual(len(revises), 1)
        self.assertEqual(revises[0]["affected"], ["b"])
        self.assertIn("obsolete", revises[0]["reason"])
        self.assertEqual(result.report.skipped, 1)
        self.assertEqual(result.report.done, 2)

    def test_abort_via_handler_directive(self):
        ctx = FakeContext()
        orch = _make_orch(ctx)
        plan = _plan(("a", "first"), ("b", "second", ["a"]))

        def handler(task):
            if task.name == "a":
                return {"revise_plan": {"abort": "pointless now"}}
            return {"ok": True, "task": task.name}

        result = orch.run("stop", plan=plan,
                          handlers={"execution": handler}, reflect=False)

        aborts = [e for e in result.reevaluations
                  if e["action"] == "abort"]
        self.assertEqual(len(aborts), 1)
        self.assertEqual(aborts[0]["affected"], ["b"])
        self.assertIn("pointless now", aborts[0]["reason"])
        self.assertEqual(result.report.cancelled, 1)

    def test_continue_when_plan_valid(self):
        ctx = FakeContext()
        orch = _make_orch(ctx)
        plan = _plan(("a", "one"), ("b", "two", ["a"]))
        result = orch.run("fine", plan=plan,
                          handlers={"execution": _ok_handler}, reflect=False)

        self.assertTrue(result.ok)
        self.assertTrue(result.reevaluations)
        self.assertTrue(all(e["action"] == "continue"
                            for e in result.reevaluations))
        # continue decisions are recorded, not broadcast
        bus = [e for e in ctx.emitted if e[0] == "plan.reevaluated"]
        self.assertEqual(bus, [])

    def test_checkpoint_never_breaks_the_run(self):
        ctx = FakeContext()
        orch = _make_orch(ctx)

        def boom(*args, **kwargs):
            raise RuntimeError("checkpoint exploded")

        orch.reevaluate = boom
        plan = _plan(("a", "one"))
        result = orch.run("resilient", plan=plan,
                          handlers={"execution": _ok_handler}, reflect=False)
        self.assertTrue(result.ok)
        self.assertEqual(result.report.done, 1)

    def test_reevaluation_log_is_bounded(self):
        orch = _make_orch(FakeContext())
        for i in range(70):
            orch.reevaluation_log.append(Reevaluation("continue", f"r{i}"))
        self.assertEqual(len(orch.reevaluation_log), 64)

    def test_stats_includes_reevaluations(self):
        ctx = FakeContext()
        orch = _make_orch(ctx)
        plan = _plan(("a", "one"))
        orch.run("stats", plan=plan,
                 handlers={"execution": _ok_handler}, reflect=False)
        summary = orch.stats()["reevaluations"]
        self.assertGreaterEqual(summary["total"], 1)
        self.assertEqual(summary["aborts"], 0)
        self.assertEqual(summary["last"]["action"], "continue")

    def test_reasoning_modules_still_present(self):
        import importlib

        for dotted in ("nomorals.agents.orchestrator",
                       "nomorals.agents.reasoning",
                       "nomorals.agents.coremind",
                       "nomorals.agents.supervisor",
                       "nomorals.agents.runtime",
                       "nomorals.missions.runner"):
            importlib.import_module(dotted)
        from nomorals.agents.orchestrator import (  # noqa: F401
            MasterOrchestrator, Reevaluation)


# ── 2. one primary path per request ────────────────────────────────────────

class TestSinglePrimaryPath(unittest.TestCase):
    def test_one_handler_call_per_task(self):
        """A run wakes each task's handler exactly once — no pileup."""
        ctx = FakeContext()
        orch = _make_orch(ctx)
        calls: dict[str, int] = {}

        def handler(task):
            calls[task.name] = calls.get(task.name, 0) + 1
            return {"ok": True}

        plan = _plan(("a", "one"), ("b", "two", ["a"]),
                     ("c", "three", ["b"]))
        result = orch.run("single", plan=plan,
                          handlers={"execution": handler}, reflect=False)
        self.assertTrue(result.ok)
        self.assertEqual(calls, {"a": 1, "b": 1, "c": 1})

    def test_no_specialist_without_delegation(self):
        """_default_handler consults agent_for once; a None answer wakes
        nothing — the stub result carries the honest status."""
        ctx = FakeContext()
        consulted = []
        ctx.tools = SimpleNamespace(
            agent_for=lambda role: consulted.append(role) or None)
        orch = _make_orch(ctx)
        task = Task(name="t1", kind=TaskKind.IO, role="research",
                    payload={"goal": "g"})
        out = orch._default_handler(task)
        self.assertEqual(out["status"], "no handler")
        self.assertEqual(consulted, ["research"])

    def test_specialist_invoked_exactly_once_when_delegated(self):
        """When the primary path explicitly delegates, the specialist runs
        exactly once for the task — never twice, never alongside another."""
        ctx = FakeContext()
        runs = []

        class SpyAgent:
            def run(self, payload):
                runs.append(payload)
                return SimpleNamespace(output={"done": 1})

        spy = SpyAgent()
        ctx.tools = SimpleNamespace(agent_for=lambda role: spy)
        orch = _make_orch(ctx)
        task = Task(name="t1", kind=TaskKind.IO, role="coding",
                    payload={"goal": "g"})
        out = orch._default_handler(task)
        self.assertEqual(out, {"done": 1})
        self.assertEqual(runs, [{"goal": "g"}])

    def test_multi_intent_runs_single_orchestrator(self):
        """The multi-intent dispatch is one MasterOrchestrator run — one
        instantiation, one run() call.  (Previously this path raised
        ImportError on a stale ``Orchestrator`` name.)"""
        import nomorals.agents.orchestrator as orch_mod

        calls = []

        class FakeOrch:
            def __init__(self, context):
                calls.append(("new", context))

            def run(self, goal, reflect=False):
                calls.append(("run", goal, reflect))
                return SimpleNamespace(
                    ok=True,
                    plan=SimpleNamespace(
                        steps=[SimpleNamespace(name="s1")]),
                    report=SimpleNamespace(results={"s1": "x"},
                                           failures={}))

        orig = orch_mod.MasterOrchestrator
        orch_mod.MasterOrchestrator = FakeOrch
        try:
            ctx = FakeContext()
            mind = cm.CoreMind(ctx)
            captured = {}

            def fake_send_async(chat_key, job, job_id, note, kind="job"):
                captured["job"] = job
                return note

            mind._send_async = fake_send_async
            intent = cm.Intent("multi", 0.8, target="do x and y",
                              route="orchestrator", why="test")
            mind._dispatch_multi(intent, "job1", "chatkey", None)
            out = captured["job"]()
        finally:
            orch_mod.MasterOrchestrator = orig

        self.assertEqual(len([c for c in calls if c[0] == "new"]), 1)
        runs = [c for c in calls if c[0] == "run"]
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0][1], "do x and y")
        self.assertIn("orchestrator: done (1 steps)", out)
        self.assertIn("[✓] s1", out)

    def test_no_bare_orchestrator_import_in_coremind(self):
        """Regression: the stale ``from .orchestrator import Orchestrator``
        name broke every multi-intent dispatch with ImportError."""
        import pathlib

        src = pathlib.Path(cm.__file__).read_text()
        self.assertNotIn("from .orchestrator import Orchestrator", src)

    def test_summarize_orchestration_shapes(self):
        def stub(ok, steps, results, failures):
            return SimpleNamespace(
                ok=ok,
                plan=SimpleNamespace(
                    steps=[SimpleNamespace(name=n) for n in steps]),
                report=SimpleNamespace(results=results,
                                       failures=failures))

        text = cm._summarize_orchestration(
            stub(True, ["a", "b"], {"a": 1, "b": 2}, {}))
        self.assertIn("orchestrator: done (2 steps)", text)
        self.assertIn("[✓] a", text)
        text = cm._summarize_orchestration(
            stub(False, ["a", "b"], {"a": 1}, {"b": "boom"}))
        self.assertIn("1 failed", text)
        self.assertIn("[✗] b", text)


# ── 3. observability ─────────────────────────────────────────────────────

class TestObservability(unittest.TestCase):
    def test_reevaluation_telemetry_roundtrip(self):
        db = FakeDB()
        rt.record_reevaluation(db, "trim", "dropped a duplicate",
                               goal="some goal")
        rt.record_route(db, "orchestrator")
        snap = rt.snapshot(db)
        last = snap["last_reevaluation"]
        self.assertEqual(last["action"], "trim")
        self.assertEqual(last["reason"], "dropped a duplicate")
        self.assertEqual(last["goal"], "some goal")
        self.assertGreater(last["at"], 0)
        self.assertEqual(snap["routes"]["orchestrator"], 1)

    def test_record_reevaluation_never_raises(self):
        rt.record_reevaluation(None, "abort", "no db here")
        self.assertEqual(rt.snapshot(None)["last_reevaluation"], None)

    def test_nm_mind_shows_last_reevaluation(self):
        from nomorals import cli

        db = FakeDB()
        rt.record_reevaluation(db, "abort", "cascade stop",
                               goal="big goal")
        ctx = FakeContext()
        ctx.db = db
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli._cmd_mind(SimpleNamespace(json=True), ctx)
        self.assertEqual(rc, 0)
        import json

        payload = json.loads(buf.getvalue())
        last = payload["last_reevaluation"]
        self.assertEqual(last["action"], "abort")
        self.assertIn("cascade stop", last["reason"])


if __name__ == "__main__":
    unittest.main()
