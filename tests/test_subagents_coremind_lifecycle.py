"""Lifecycle completion tests for the agents-imp-core wave.

subagents.py
- Subagent is abstract: bare instantiation fails fast with TypeError
  (the old raise NotImplementedError at line 180 only failed at call time).
- ApiDesigner removes its mkdtemp scratch dir after import validation
  (no temp-tree litter per run).

coremind.py
- _dispatch leaves an async-started job "active" until the worker thread
  finalizes it (no premature "done" recorded at dispatch time).
- a shed job stays "failed" through _dispatch (the old code flipped it
  back to "done" because the shed note didn't start with ❌).
- _send_async survives Thread.start() failure without leaking the
  semaphore slot or the in-flight gauge.
- _dispatch retries a crashed dispatch fn exactly once, then reports the
  failure with the attempt count (never silently dropped).
- _new_job tolerates target=None and is lock-guarded.
- _browse_url is one normalization shared by the chat and console paths.
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.agents import coremind as C
from nomorals.agents import subagents as S
from nomorals.agents.coremind import CoreMind, Intent
from nomorals.agents.subagents import ApiDesigner, ApiDesignInput
from nomorals.llm.providers.mock import MockProvider


# ── fakes ────────────────────────────────────────────────────────────────


class _FakeSettings:
    def __init__(self, mind=None):
        self._mind = mind

    def resolve(self, key):
        raise RuntimeError("no settings in tests")

    def __getattr__(self, name):
        if name == "mind":
            if self._mind is None:
                raise AttributeError(name)
            return self._mind
        raise AttributeError(name)


class _FakeContext:
    def __init__(self, mind=None):
        self.settings = _FakeSettings(mind)
        self.extras = {}
        self.memory = None
        self.router = None
        self.db = None


def _mind_settings(max_inflight, timeout):
    return SimpleNamespace(max_inflight=max_inflight,
                           inflight_acquire_timeout_s=timeout)


_STUB = '''"""Widget API stub."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Widget", "WidgetResult", "make_widget"]


@dataclass
class WidgetResult:
    ok: bool
    name: str = ""


@dataclass
class Widget:
    name: str


def make_widget(name: str) -> WidgetResult:
    ...
'''


# ── subagents: the abstract base ──────────────────────────────────────────


class AbstractBaseTest(unittest.TestCase):
    def test_bare_base_cannot_instantiate(self):
        with self.assertRaises(TypeError):
            S.Subagent()

    def test_subclass_without_run_cannot_instantiate(self):
        class Broken(S.Subagent):
            roster_type = "broken"

        with self.assertRaises(TypeError):
            Broken()

    def test_all_roster_classes_implement_run(self):
        for name, cls in S.ROSTER.items():
            agent = cls(router=None, project_root=".")
            self.assertTrue(callable(agent.run), name)


# ── subagents: ApiDesigner temp hygiene ───────────────────────────────────


class ApiDesignerTempTest(unittest.TestCase):
    def test_scratch_dir_removed_after_validation(self):
        created: list[str] = []
        real_mkdtemp = tempfile.mkdtemp

        def spy_mkdtemp(*a, **kw):
            d = real_mkdtemp(*a, **kw)
            created.append(d)
            return d

        router = MockProvider(scripted={
            "Design the public Python API": json.dumps({"stub": _STUB})})
        designer = ApiDesigner(router=router, project_root=".")
        with mock.patch("tempfile.mkdtemp", spy_mkdtemp):
            result = designer.run(ApiDesignInput(
                feature="a widget factory", sample_modules=[]))
        self.assertTrue(result.imports_ok, result.note)
        self.assertTrue(created, "expected mkdtemp to be used")
        for d in created:
            self.assertFalse(
                __import__("os").path.exists(d),
                f"scratch dir leaked: {d}")


# ── coremind: async job lifecycle ─────────────────────────────────────────


class AsyncJobStateTest(unittest.TestCase):
    def _mind(self):
        return CoreMind(_FakeContext(mind=_mind_settings(4, 5.0)))

    def test_async_job_stays_active_until_worker_finishes(self):
        mind = self._mind()
        release = threading.Event()

        def job():
            release.wait(timeout=10)
            return "work finished"

        intent = Intent("build", 0.9, target="x", route="coding")
        job_id_holder: dict[str, str] = {}
        real_new_job = mind._new_job

        def spy_new_job(i):
            jid = real_new_job(i)
            job_id_holder["id"] = jid
            return jid

        with mock.patch.object(mind, "_new_job", spy_new_job), \
             mock.patch.object(mind, "_dispatch_build",
                               lambda i, j, c, m: mind._send_async(
                                   c, job, j, "started", kind="build")):
            reply = mind._dispatch(intent, "k:console", None)
        self.assertIsInstance(reply, str)
        status_now = next(j["status"] for j in mind._jobs
                          if j["id"] == job_id_holder["id"])
        self.assertEqual(status_now, "active",
                         "job must not be 'done' before the worker finishes")
        release.set()
        deadline = time.time() + 10
        while True:
            final = next(j["status"] for j in mind._jobs
                         if j["id"] == job_id_holder["id"])
            if final != "active" or time.time() > deadline:
                break
            time.sleep(0.05)
        self.assertEqual(final, "done")

    def test_shed_job_stays_failed_through_dispatch(self):
        mind = CoreMind(_FakeContext(mind=_mind_settings(1, 0.05)))
        hold = threading.Event()

        def slow():
            hold.wait(timeout=10)
            return "slow"

        mind._send_async("k:console", slow, "slow", "started", kind="build")
        deadline = time.time() + 5
        while mind._inflight_now < 1 and time.time() < deadline:
            time.sleep(0.02)

        intent = Intent("research", 0.9, target="x", route="research_swarm")
        with mock.patch.object(
                mind, "_dispatch_research",
                lambda i, j, c, m: mind._send_async(
                    c, lambda: "x", j, "started", kind="research")):
            reply = mind._dispatch(intent, "k:console", None)
        hold.set()
        self.assertIn("shed", reply)
        self.assertTrue(reply.startswith("❌") or "\n❌" in reply)
        job = mind._jobs[-1]
        self.assertEqual(job["status"], "failed",
                         "a shed job must stay failed, not flip to done")
        self.assertIn("shed", job["note"])
        deadline = time.time() + 10
        while mind._inflight_now and time.time() < deadline:
            time.sleep(0.05)

    def test_thread_start_failure_does_not_leak_slot(self):
        mind = self._mind()

        class BoomThread(threading.Thread):
            def start(self):  # noqa: D102
                raise RuntimeError("cannot spawn")

        with mock.patch.object(C.threading, "Thread", BoomThread):
            jid = mind._new_job(Intent("build", 0.9, target="x", route="coding"))
            reply = mind._send_async("k:console", lambda: "x", jid,
                                     "started", kind="build")
        self.assertIn("❌", reply)
        self.assertEqual(mind._inflight_now, 0)
        # the slot is usable again
        self.assertTrue(mind._inflight.acquire(blocking=False))
        mind._inflight.release()
        job = mind._jobs[-1]
        self.assertEqual(job["status"], "failed")
        self.assertIn("background thread", job["note"])


# ── coremind: dispatch retry ──────────────────────────────────────────────


class DispatchRetryTest(unittest.TestCase):
    def _mind(self):
        return CoreMind(_FakeContext(mind=_mind_settings(4, 5.0)))

    def test_transient_crash_retried_once_then_succeeds(self):
        mind = self._mind()
        calls = {"n": 0}

        def flaky(intent, job_id, chat_key, message):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient boom")
            return "recovered"

        with mock.patch.object(mind, "_dispatch_status", flaky):
            reply = mind._dispatch(
                Intent("status", 0.9, route="mind"), "k:console", None)
        self.assertEqual(calls["n"], 2)
        self.assertEqual(reply, "recovered")
        self.assertEqual(mind._jobs[-1]["status"], "done")

    def test_persistent_crash_fails_loud_with_attempt_count(self):
        mind = self._mind()

        def always_boom(intent, job_id, chat_key, message):
            raise RuntimeError("still broken")

        with mock.patch.object(mind, "_dispatch_status", always_boom):
            reply = mind._dispatch(
                Intent("status", 0.9, route="mind"), "k:console", None)
        self.assertIn("that route just failed", reply)
        job = mind._jobs[-1]
        self.assertEqual(job["status"], "failed")
        self.assertIn("2 attempts", job["note"])


# ── coremind: job registry + browse normalization ─────────────────────────


class JobRegistryTest(unittest.TestCase):
    def test_new_job_tolerates_none_target(self):
        mind = CoreMind(_FakeContext())
        jid = mind._new_job(Intent("research", 0.9, target=None))  # type: ignore[arg-type]
        job = next(j for j in mind._jobs if j["id"] == jid)
        self.assertEqual(job["status"], "active")
        self.assertEqual(job["target"], "")

    def test_new_job_is_lock_guarded_under_concurrency(self):
        mind = CoreMind(_FakeContext())
        errors: list[BaseException] = []

        def make(i):
            try:
                mind._new_job(Intent("chat", 1.0, target=f"t{i}"))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=make, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        self.assertFalse(errors)
        self.assertEqual(len(mind._jobs), 20)
        self.assertEqual(len({j["id"] for j in mind._jobs}), 20)


class BrowseUrlTest(unittest.TestCase):
    def test_shared_normalization(self):
        self.assertEqual(CoreMind._browse_url("https://x.io/a"),
                         "https://x.io/a")
        self.assertEqual(CoreMind._browse_url("http://x.io"),
                         "http://x.io")
        self.assertEqual(CoreMind._browse_url("hn"),
                         "https://news.ycombinator.com")
        self.assertEqual(CoreMind._browse_url("Hacker News"),
                         "https://news.ycombinator.com")
        self.assertEqual(CoreMind._browse_url("example.com"),
                         "https://example.com")
        self.assertEqual(CoreMind._browse_url("example"),
                         "https://example.com")


if __name__ == "__main__":
    unittest.main()
