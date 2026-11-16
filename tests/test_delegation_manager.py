"""Delegation manager: async spawn / track / steer / resume / collect,
restart recovery, and the plan step — over the untouched governance
substrate (agents/delegation.py) and the roster workers
(agents/subagents.py).

The manager never runs work synchronously: spawn returns while the
worker runs on a background thread, every state change lands on a
durable row, and a second manager over the same database (a restart)
can recover and resume what the first one left running.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.delegation import (
    DelegationError,
    DelegationStore,
    StaleGenerationError,
)
from nomorals.agents.delegation_manager import (
    PLAN_ROLE_TO_ROSTER,
    DelegationManager,
)
from nomorals.storage.db import Database


def _db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


class _FakeAgent:
    """A scripted roster worker: one behavior per invocation.

    Behaviors: ("ok", payload) returns {"ok": True, "result": payload};
    ("fail", error) returns {"ok": False, "error": error};
    ("raise", error) raises; ("block", event) waits on the event
    (up to 30s) then returns ok.  Every call is recorded with the
    exact input dict it received, so steering can be proven to reach
    the worker.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def __call__(self, role, input_dict):
        with self._lock:
            self.calls.append({"role": role,
                               "input": dict(input_dict or {})})
            index = len(self.calls) - 1
        behavior = self.script[min(index, len(self.script) - 1)]
        kind = behavior[0]
        if kind == "ok":
            return {"ok": True, "result": behavior[1]}
        if kind == "fail":
            return {"ok": False, "error": behavior[1]}
        if kind == "raise":
            raise RuntimeError(behavior[1])
        if kind == "block":
            behavior[1].wait(timeout=30)
            return {"ok": True, "result": {"unblocked": True}}
        raise AssertionError(f"unknown behavior {behavior!r}")


class _FakeNotifier:
    """Records every publish; delivery always succeeds."""

    def __init__(self):
        self.published: list[dict] = []

    def __call__(self, kind, title, body, **_kw):
        self.published.append({"kind": kind, "title": title,
                               "body": body})
        return {"delivered": True, "delivery_state": "sent"}


def _manager(db, fake, notifier, **kw):
    store = DelegationStore(db)
    return DelegationManager(store, db=db, run_agent=fake,
                             notifier=notifier, **kw)


def _spawn(manager, **kw):
    base = dict(role="tester", goal="cover the new retry path")
    base.update(kw)
    return manager.spawn(**base)


class SpawnTrackTests(unittest.TestCase):
    def test_spawn_runs_async_and_tracks(self):
        db = _db()
        event = threading.Event()
        fake = _FakeAgent([("block", event)])
        notifier = _FakeNotifier()
        manager = _manager(db, fake, notifier)
        run = _spawn(manager)
        try:
            # Spawn returns while the worker is still running.
            self.assertIn(manager.get_run(run.id).status,
                          ("running", "steering"))
            event.set()
            done = manager.wait(run.id, timeout=10)
            self.assertEqual(done.status, "completed")
            self.assertEqual(json.loads(done.result_json),
                             {"ok": True, "result": {"unblocked": True}})
            # The contract settled through the shipped gate.
            store = DelegationStore(db)
            contract = store.get_contract(run.contract_id)
            self.assertIsNotNone(contract)
            self.assertEqual(contract.status, "completed")
            claims = store.claims(contract.id)
            self.assertTrue(any(c.kind == "evidence" for c in claims))
            self.assertTrue(any(c.kind == "signoff" for c in claims))
        finally:
            event.set()
            manager.shutdown(wait=False)

    def test_spawn_unknown_role_refused(self):
        db = _db()
        manager = _manager(db, _FakeAgent([]), _FakeNotifier())
        try:
            with self.assertRaises(DelegationError) as caught:
                _spawn(manager, role="no-such-role")
            self.assertIn("no-such-role", str(caught.exception))
        finally:
            manager.shutdown(wait=False)

    def test_spawn_missing_input_field_refused(self):
        db = _db()
        manager = _manager(db, _FakeAgent([]), _FakeNotifier())
        try:
            with self.assertRaises(DelegationError) as caught:
                _spawn(manager, role="implementer",
                       goal="fix the retry loop")
            self.assertIn("path", str(caught.exception))
        finally:
            manager.shutdown(wait=False)

    def test_spawn_worker_failure_records_failure(self):
        db = _db()
        fake = _FakeAgent([("fail", "the worker gave up")])
        manager = _manager(db, fake, _FakeNotifier())
        run = _spawn(manager)
        done = manager.wait(run.id, timeout=10)
        self.assertEqual(done.status, "failed")
        self.assertIn("the worker gave up", done.error or "")
        store = DelegationStore(db)
        contract = store.get_contract(run.contract_id)
        self.assertEqual(contract.status, "failed")
        self.assertTrue(any(c.kind == "failure"
                            for c in store.claims(contract.id)))
        manager.shutdown(wait=False)

    def test_spawn_exception_in_worker_is_a_record_not_a_raise(self):
        db = _db()
        fake = _FakeAgent([("raise", "boom")])
        manager = _manager(db, fake, _FakeNotifier())
        run = _spawn(manager)
        done = manager.wait(run.id, timeout=10)
        self.assertEqual(done.status, "failed")
        self.assertIn("boom", done.error or "")
        manager.shutdown(wait=False)


class SteerResumeTests(unittest.TestCase):
    def test_steer_records_and_applies_on_resume(self):
        db = _db()
        fake = _FakeAgent([("fail", "stuck on the first try"),
                           ("ok", {"fixed": True})])
        manager = _manager(db, fake, _FakeNotifier())
        run = _spawn(manager)
        failed = manager.wait(run.id, timeout=10)
        self.assertEqual(failed.status, "failed")
        steered = manager.steer(run.id, "try the other approach",
                                by="owner")
        self.assertIn("try the other approach", steered.notes)
        store = DelegationStore(db)
        self.assertTrue(any(c.kind == "steer"
                            for c in store.claims(run.contract_id)))
        resumed = manager.resume(run.id, by="owner")
        self.assertEqual(resumed.generation, 2)
        done = manager.wait(run.id, timeout=10)
        self.assertEqual(done.status, "completed")
        # The steering reached the worker's input on the re-run.
        self.assertEqual(len(fake.calls), 2)
        focus = str(fake.calls[1]["input"].get("focus") or "")
        self.assertIn("try the other approach", focus)
        # The old generation is fenced: its late evidence is refused.
        with self.assertRaises(StaleGenerationError):
            store.submit_evidence(run.contract_id, kind="test run",
                                  claim="late from generation 1",
                                  producer="worker", generation=1)
        manager.shutdown(wait=False)

    def test_steer_terminal_run_refused(self):
        db = _db()
        fake = _FakeAgent([("ok", {"done": True})])
        manager = _manager(db, fake, _FakeNotifier())
        run = _spawn(manager)
        manager.wait(run.id, timeout=10)
        manager.collect(run.id)
        with self.assertRaises(DelegationError):
            manager.steer(run.id, "too late")
        manager.shutdown(wait=False)

    def test_resume_live_run_refused(self):
        db = _db()
        event = threading.Event()
        fake = _FakeAgent([("block", event)])
        manager = _manager(db, fake, _FakeNotifier())
        run = _spawn(manager)
        try:
            with self.assertRaises(DelegationError):
                manager.resume(run.id, by="owner")
        finally:
            event.set()
            manager.shutdown(wait=False)

    def test_resume_unknown_run_refused(self):
        db = _db()
        manager = _manager(db, _FakeAgent([]), _FakeNotifier())
        with self.assertRaises(DelegationError):
            manager.resume("dr-no-such-run", by="owner")
        manager.shutdown(wait=False)


class CollectTests(unittest.TestCase):
    def test_collect_delivers_and_marks(self):
        db = _db()
        fake = _FakeAgent([("ok", {"tests": 12, "passed": 12})])
        notifier = _FakeNotifier()
        manager = _manager(db, fake, notifier)
        run = _spawn(manager)
        manager.wait(run.id, timeout=10)
        outcome = manager.collect(run.id)
        self.assertTrue(outcome["ok"])
        self.assertTrue(outcome["delivered"])
        self.assertEqual(len(notifier.published), 1)
        published = notifier.published[0]
        self.assertEqual(published["kind"], "delegation")
        self.assertIn(run.contract_id, published["body"])
        self.assertEqual(manager.get_run(run.id).status, "collected")
        # Collecting again replays without a second delivery.
        again = manager.collect(run.id)
        self.assertTrue(again["ok"])
        self.assertTrue(again["replayed"])
        self.assertEqual(len(notifier.published), 1)
        manager.shutdown(wait=False)

    def test_collect_refuses_when_gate_blocked(self):
        db = _db()
        # An implementer run with declared writes whose result is
        # empty: the settle gate finds the defect, contradicts the
        # evidence, and refuses completion — collect must say so.
        # A second, fixed result then resumes through the gate.
        fake = _FakeAgent([("ok", {}),
                           ("ok", {"writes": ["nomorals/retry.py"],
                                   "edits": 1})])
        manager = _manager(db, fake, _FakeNotifier())
        run = manager.spawn(
            role="implementer", goal="rewrite the retry loop",
            input={"path": "nomorals/retry.py",
                   "instruction": "rewrite the retry loop",
                   "file_content": ""},
            scope=("nomorals/retry.py",))
        done = manager.wait(run.id, timeout=10)
        self.assertEqual(done.status, "completed")
        store = DelegationStore(db)
        contract = store.get_contract(run.contract_id)
        self.assertEqual(contract.status, "open")
        outcome = manager.collect(run.id)
        self.assertFalse(outcome["ok"])
        self.assertTrue(outcome["reasons"])
        self.assertEqual(manager.get_run(run.id).status, "completed")
        # The gate-refused run resumes: steer, re-run, settle, collect.
        manager.steer(run.id, "actually write the file this time")
        resumed = manager.resume(run.id, by="owner")
        self.assertEqual(resumed.generation, 2)
        redone = manager.wait(run.id, timeout=10)
        self.assertEqual(redone.status, "completed")
        contract = store.get_contract(run.contract_id)
        self.assertEqual(contract.status, "completed")
        outcome = manager.collect(run.id)
        self.assertTrue(outcome["ok"])
        manager.shutdown(wait=False)

    def test_collect_running_run_refused(self):
        db = _db()
        event = threading.Event()
        fake = _FakeAgent([("block", event)])
        manager = _manager(db, fake, _FakeNotifier())
        run = _spawn(manager)
        try:
            outcome = manager.collect(run.id)
            self.assertFalse(outcome["ok"])
            self.assertTrue(any("running" in r for r in outcome["reasons"]))
        finally:
            event.set()
            manager.shutdown(wait=False)

    def test_collect_delivery_failure_still_collects(self):
        db = _db()
        fake = _FakeAgent([("ok", {"done": True})])

        def dead_channel(kind, title, body, **kw):
            return {"delivered": False, "reason": "no channel"}

        manager = _manager(db, fake, dead_channel)
        run = _spawn(manager)
        manager.wait(run.id, timeout=10)
        outcome = manager.collect(run.id)
        # The contract is completed and the run is collected even
        # when no chat channel is live — the delivery failure is
        # named, never silent, and the mission loop is not blocked
        # on a dead channel.
        self.assertTrue(outcome["ok"])
        self.assertFalse(outcome["delivered"])
        self.assertIn("no channel", " ".join(outcome["reasons"]))
        self.assertEqual(manager.get_run(run.id).status, "collected")
        manager.shutdown(wait=False)


class RestartRecoveryTests(unittest.TestCase):
    def test_restart_recovery_and_resume(self):
        db = _db()
        event = threading.Event()
        fake = _FakeAgent([("block", event), ("ok", {"done": True})])
        notifier = _FakeNotifier()
        first = _manager(db, fake, notifier)
        run = _spawn(first)
        # A second manager over the same database is a restart: no
        # live futures, only durable rows.
        second = _manager(db, fake, notifier)
        interrupted = second.recover()
        self.assertEqual([r.id for r in interrupted], [run.id])
        self.assertEqual(second.get_run(run.id).status, "interrupted")
        store = DelegationStore(db)
        self.assertTrue(any(c.kind == "failure"
                            for c in store.claims(run.contract_id)))
        resumed = second.resume(run.id, by="owner")
        self.assertEqual(resumed.generation, 2)
        done = second.wait(run.id, timeout=10)
        self.assertEqual(done.status, "completed")
        self.assertEqual(json.loads(done.result_json)["result"],
                         {"done": True})
        # The first worker finally unblocks: its late finalize must
        # not resurrect or overwrite the resumed run.
        event.set()
        import time as _time
        _time.sleep(0.3)
        self.assertEqual(second.get_run(run.id).status, "completed")
        claims = store.claims(run.contract_id)
        evidence = [c for c in claims if c.kind == "evidence"]
        self.assertEqual(len(evidence), 2)
        first.shutdown(wait=False)
        second.shutdown(wait=False)

    def test_recover_is_a_noop_when_nothing_is_running(self):
        db = _db()
        manager = _manager(db, _FakeAgent([]), _FakeNotifier())
        self.assertEqual(manager.recover(), [])
        manager.shutdown(wait=False)


class PollTests(unittest.TestCase):
    def test_poll_collects_completed_runs(self):
        db = _db()
        fake = _FakeAgent([("ok", {"done": True})])
        notifier = _FakeNotifier()
        manager = _manager(db, fake, notifier)
        run = _spawn(manager)
        manager.wait(run.id, timeout=10)
        outcome = manager.poll()
        self.assertEqual(outcome["collected"], [run.id])
        self.assertEqual(len(notifier.published), 1)
        self.assertEqual(manager.get_run(run.id).status, "collected")
        # A second poll changes nothing and delivers nothing twice.
        again = manager.poll()
        self.assertEqual(again["collected"], [])
        self.assertEqual(len(notifier.published), 1)
        manager.shutdown(wait=False)

    def test_poll_heals_interrupted_runs(self):
        db = _db()
        event = threading.Event()
        fake = _FakeAgent([("block", event), ("ok", {"healed": True})])
        notifier = _FakeNotifier()
        first = _manager(db, fake, notifier)
        run = _spawn(first)
        second = _manager(db, fake, notifier)
        # The tick heals what the restart interrupted, then collects
        # the resumed run once it finishes.
        healed = second.poll()
        self.assertEqual(healed["resumed"], [run.id])
        done = second.wait(run.id, timeout=10)
        # The tick may already have collected the resumed run
        # inside the same poll that healed it (finalize + collect
        # share the tick), so either landing state is correct —
        # what matters is exactly one delivery at the end.
        self.assertIn(done.status, ("completed", "collected"))
        event.set()
        second.poll()
        self.assertEqual(second.get_run(run.id).status, "collected")
        self.assertEqual(len(notifier.published), 1)
        first.shutdown(wait=False)
        second.shutdown(wait=False)


class PlanTests(unittest.TestCase):
    def test_plan_and_spawn(self):
        db = _db()
        fake = _FakeAgent([("ok", {"items": []})])
        manager = _manager(db, fake, _FakeNotifier())
        result = manager.plan_and_spawn("test the retry path")
        self.assertIn("plan_error", result)
        # No model is configured in tests, so the template plan runs
        # and the degradation is named, never silent.
        self.assertTrue(result["plan_error"])
        self.assertEqual([r.role for r in result["runs"]], ["planner"])
        self.assertEqual(
            sorted(u["step"] for u in result["unassigned"]),
            ["execute", "verify"])
        reasons = " ".join(u["reason"] for u in result["unassigned"])
        self.assertIn("no roster worker", reasons)
        done = manager.wait(result["runs"][0].id, timeout=10)
        self.assertEqual(done.status, "completed")
        manager.shutdown(wait=False)

    def test_plan_role_mapping(self):
        self.assertEqual(PLAN_ROLE_TO_ROSTER["research"], "planner")
        self.assertEqual(PLAN_ROLE_TO_ROSTER["coding"], "implementer")
        self.assertEqual(PLAN_ROLE_TO_ROSTER["critic"], "reviewer")
        self.assertEqual(PLAN_ROLE_TO_ROSTER["reflection"], "planner")
        self.assertNotIn("execution", PLAN_ROLE_TO_ROSTER)
        self.assertNotIn("vision", PLAN_ROLE_TO_ROSTER)


class DefaultRunnerTests(unittest.TestCase):
    def test_default_runner_executes_a_real_roster_worker(self):
        # No run_agent injected: the manager runs the real Planner
        # from subagents.py. With no model configured the Planner
        # falls back to its explicit single-file plan — a real
        # roster execution, offline.
        db = _db()
        store = DelegationStore(db)
        manager = DelegationManager(store, db=db,
                                    notifier=_FakeNotifier())
        run = manager.spawn(role="planner", goal="plan the fix",
                            input={"task": "plan the fix"})
        done = manager.wait(run.id, timeout=30)
        self.assertEqual(done.status, "completed")
        result = json.loads(done.result_json)["result"]
        self.assertIn("items", result)
        manager.shutdown(wait=False)


class JobTests(unittest.TestCase):
    def test_ensure_job_registers_and_is_idempotent(self):
        from nomorals.agents.delegation_manager import (
            DELEGATION_JOB,
            DELEGATION_SPEC,
            DELEGATION_TOOL,
            ensure_delegation_manager_job,
        )

        class _Scheduler:
            def __init__(self):
                self.jobs = []

            def list_jobs(self):
                return list(self.jobs)

            def add(self, name, spec, kind, payload, routine=False):
                job = {"id": f"job-{len(self.jobs)}", "name": name,
                       "spec": spec, "kind": kind, "payload": payload,
                       "routine": routine}
                self.jobs.append(job)
                return job

            def remove(self, job_id):
                self.jobs = [j for j in self.jobs if j["id"] != job_id]

        scheduler = _Scheduler()
        job = ensure_delegation_manager_job(scheduler)
        self.assertEqual(job["name"], DELEGATION_JOB)
        self.assertEqual(job["spec"], DELEGATION_SPEC)
        self.assertEqual(job["payload"]["tool"], DELEGATION_TOOL)
        again = ensure_delegation_manager_job(scheduler)
        self.assertEqual(again["id"], job["id"])
        self.assertEqual(len(scheduler.jobs), 1)

    def test_tick_once_never_raises(self):
        from nomorals.agents.delegation_manager import (
            run_delegation_manager_tick_once,
        )

        db = _db()
        fake = _FakeAgent([("ok", {"done": True})])
        context = SimpleNamespace(db=db)
        manager = _manager(db, fake, _FakeNotifier())
        run = _spawn(manager)
        manager.wait(run.id, timeout=10)
        manager.shutdown(wait=False)
        # The tick builds its own manager from the context (a fresh
        # process shape) and collects what the first one left done.
        outcome = run_delegation_manager_tick_once(context)
        self.assertIn("collected", outcome)
        self.assertIn(run.id, outcome["collected"])

    def test_tick_without_db_is_honest(self):
        from nomorals.agents.delegation_manager import (
            run_delegation_manager_tick_once,
        )

        outcome = run_delegation_manager_tick_once(SimpleNamespace(db=None))
        self.assertFalse(outcome["ok"])
        self.assertIn("database", outcome["error"])

    def test_autonomy_registers_the_tick_tool(self):
        from nomorals.agents.delegation_manager import DELEGATION_TOOL
        from nomorals.tools import autonomy as autonomy_mod

        class _FakeRegistry:
            def __init__(self, context):
                self.context = context
                self.fns = {}

            def register(self, name, fn=None, **kw):
                if fn is None:
                    def deco(f):
                        self.fns[name] = f
                        return f
                    return deco
                self.fns[name] = fn
                return fn

        db = _db()
        fake = _FakeAgent([("ok", {"done": True})])
        manager = _manager(db, fake, _FakeNotifier())
        run = _spawn(manager)
        manager.wait(run.id, timeout=10)
        manager.shutdown(wait=False)
        registry = _FakeRegistry(SimpleNamespace(db=db))
        autonomy_mod.register(registry)
        self.assertIn(DELEGATION_TOOL, registry.fns)
        # The registered tool runs the tick: the completed run is
        # collected (delivery parks pending with no live gateway —
        # the run still collects, and the tick reports it).
        outcome = registry.fns[DELEGATION_TOOL]()
        self.assertTrue(outcome["ok"])
        self.assertIn(run.id, outcome["collected"])
        self.assertEqual(manager.get_run(run.id).status, "collected")


def _handler(test: unittest.TestCase, db=None, manager=None):
    from nomorals.agents.partner.runtime_expression import (
        RuntimeExpressionMixin,
    )

    handler = RuntimeExpressionMixin()
    handler.context = SimpleNamespace(db=db)
    tmp = tempfile.mkdtemp(dir=os.environ.get("NM_TEST_TMPDIR") or None)
    test.addCleanup(lambda: shutil.rmtree(tmp, True))
    env = mock.patch.dict(os.environ, {"NM_HOME": tmp})
    env.start()
    test.addCleanup(env.stop)
    if manager is not None:
        handler._delegation_manager_cache = manager
    return handler


class ChatSurfaceTests(unittest.TestCase):
    def test_spawn_track_steer_resume_collect_through_delegate(self):
        db = _db()
        fake = _FakeAgent([("fail", "stuck"),
                           ("ok", {"tests": 3, "passed": 3})])
        notifier = _FakeNotifier()
        manager = _manager(db, fake, notifier)
        handler = _handler(self, db=db, manager=manager)
        try:
            spawned = handler._control_delegate(
                "spawn tester cover the new retry path", None)
            self.assertIn("spawned", spawned)
            run_id = spawned.split("run ")[1].split()[0]
            tracked = handler._control_delegate(f"track {run_id}", None)
            self.assertIn(run_id, tracked)
            manager.wait(run_id, timeout=10)
            self.assertEqual(manager.get_run(run_id).status, "failed")
            steered = handler._control_delegate(
                f"steer {run_id} try the other approach", None)
            self.assertIn("steered", steered)
            resumed = handler._control_delegate(f"resume {run_id}", None)
            self.assertIn("resumed", resumed)
            manager.wait(run_id, timeout=10)
            collected = handler._control_delegate(
                f"collect {run_id}", None)
            self.assertIn("collected", collected)
            self.assertEqual(len(notifier.published), 1)
            listed = handler._control_delegate("runs", None)
            self.assertIn(run_id, listed)
        finally:
            manager.shutdown(wait=False)

    def test_delegate_verbs_fail_plainly_without_db(self):
        handler = _handler(self, db=None)
        for tail in ("spawn tester x", "runs", "track dr-x",
                     "steer dr-x note", "resume dr-x", "collect dr-x",
                     "recover", "plan test the thing"):
            out = handler._control_delegate(tail, None)
            self.assertIn("unavailable", out, tail)

    def test_delegate_unknown_verb_names_the_verbs(self):
        db = _db()
        handler = _handler(self, db=db)
        out = handler._control_delegate("frobnicate", None)
        self.assertIn("spawn", out)
        self.assertIn("collect", out)


class RegistryTests(unittest.TestCase):
    def test_delegate_help_names_the_manager_verbs(self):
        from nomorals.social.chat import control

        usage = control.COMMAND_DETAILS["delegate"]["usage"]
        for verb in ("spawn", "plan", "runs", "track", "steer",
                     "resume", "collect"):
            self.assertIn(verb, usage)

    def test_delegate_stays_owner_only(self):
        from nomorals.social.chat import control

        self.assertNotIn("delegate", control.PUBLIC_COMMANDS)
        self.assertFalse(control.caller_may_run("delegate", False))
        self.assertTrue(control.caller_may_run("delegate", True))


if __name__ == "__main__":
    unittest.main()
