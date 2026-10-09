"""Async research jobs: immediate job id, background execution,
status/result/cancel, notifier delivery, never-raises.

No database here on purpose — the in-memory fallback is exercised, which
is exactly what the tests need for determinism.
"""

from __future__ import annotations

import threading
import time
import unittest
from typing import Any
from unittest import mock

from nomorals.agents.search import jobs
from nomorals.agents.search import engine as engine_mod


class _Ctx:
    db = None  # force the in-memory fallback


def _report(query: str) -> dict[str, Any]:
    return {
        "id": "search-test",
        "query": query,
        "mode": "quick",
        "summary": f"summary for {query}",
        "pages_read": ["https://a.com/x"],
        "sources": [{"n": 1, "title": "A", "url": "https://a.com/x"}],
        "seconds": 0.5,
    }


def _wait_state(ctx: Any, job_id: str, states: tuple[str, ...],
                timeout: float = 10.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = jobs.status(ctx, job_id)
        if st["state"] in states:
            return st
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never reached {states}: {st}")


class JobLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = _Ctx()
        self.delivered: list[tuple[str, str]] = []
        self._notify_patch = mock.patch(
            "nomorals.agents.notifier.notify", self._fake_notify)
        self._notify_patch.start()
        self._run_patch = mock.patch.object(
            engine_mod.SearchEngine, "run",
            lambda self, query, **kw: _report(query))
        self._run_patch.start()

    def tearDown(self) -> None:
        self._run_patch.stop()
        self._notify_patch.stop()

    def _fake_notify(self, context: Any, kind: str, title: str,
                     body: str = "", **kw: Any) -> dict[str, Any]:
        self.delivered.append((kind, title))
        return {"ok": True}

    def test_start_returns_immediately(self) -> None:
        started = time.time()
        job_id = jobs.start(self.ctx, "bitcoin price")
        self.assertLess(time.time() - started, 5.0)
        self.assertTrue(job_id)

    def test_status_while_running_then_done(self) -> None:
        job_id = jobs.start(self.ctx, "bitcoin price")
        first = jobs.status(self.ctx, job_id)
        self.assertIn(first["state"], {"queued", "running", "done"})
        done = _wait_state(self.ctx, job_id, ("done",))
        self.assertEqual(done["query"], "bitcoin price")
        self.assertGreaterEqual(done["elapsed_seconds"], 0)

    def test_result_returns_full_report(self) -> None:
        job_id = jobs.start(self.ctx, "bitcoin price")
        _wait_state(self.ctx, job_id, ("done",))
        report = jobs.result(self.ctx, job_id)
        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(report["query"], "bitcoin price")
        self.assertEqual(report["summary"], "summary for bitcoin price")

    def test_result_none_while_running(self) -> None:
        gate = threading.Event()

        def _slow_run(self: Any, query: str, **kw: Any) -> dict[str, Any]:
            gate.wait(5.0)
            return _report(query)

        with mock.patch.object(engine_mod.SearchEngine, "run", _slow_run):
            job_id = jobs.start(self.ctx, "slow query")
            try:
                self.assertIsNone(jobs.result(self.ctx, job_id))
            finally:
                gate.set()
            _wait_state(self.ctx, job_id, ("done",))

    def test_completion_delivers_via_notifier(self) -> None:
        job_id = jobs.start(self.ctx, "bitcoin price")
        _wait_state(self.ctx, job_id, ("done",))
        kinds = [k for k, _t in self.delivered]
        self.assertIn("research", kinds)
        titles = [t for _k, t in self.delivered]
        self.assertTrue(any("bitcoin price" in t for t in titles))

    def test_failure_is_recorded_not_raised(self) -> None:
        def _boom(self: Any, query: str, **kw: Any) -> dict[str, Any]:
            raise RuntimeError("engine exploded")

        with mock.patch.object(engine_mod.SearchEngine, "run", _boom):
            job_id = jobs.start(self.ctx, "doomed query")
            failed = _wait_state(self.ctx, job_id, ("failed",))
        self.assertIn("engine exploded", failed["error"])
        self.assertIsNone(jobs.result(self.ctx, job_id))
        # the failure itself is delivered, never swallowed
        self.assertTrue(any("doomed query" in t for _k, t in self.delivered))

    def test_empty_query_fails_fast_with_clear_error(self) -> None:
        job_id = jobs.start(self.ctx, "   ")
        st = jobs.status(self.ctx, job_id)
        self.assertEqual(st["state"], "failed")
        self.assertIn("query", st["error"].lower())

    def test_unknown_job_status(self) -> None:
        st = jobs.status(self.ctx, "rj-does-not-exist")
        self.assertEqual(st["state"], "unknown")

    def test_cancel(self) -> None:
        gate = threading.Event()

        def _slow_run(self: Any, query: str, **kw: Any) -> dict[str, Any]:
            gate.wait(5.0)
            return _report(query)

        with mock.patch.object(engine_mod.SearchEngine, "run", _slow_run):
            job_id = jobs.start(self.ctx, "cancellable query")
            _wait_state(self.ctx, job_id, ("running", "queued"))
            self.assertTrue(jobs.cancel(self.ctx, job_id))
            gate.set()
            cancelled = _wait_state(self.ctx, job_id, ("cancelled",))
            self.assertEqual(cancelled["state"], "cancelled")
            # a cancelled job is never delivered
            self.assertFalse(any("cancellable query" in t
                                 for _k, t in self.delivered))

    def test_cancel_unknown_job_is_false(self) -> None:
        self.assertFalse(jobs.cancel(self.ctx, "rj-nope"))

    def test_deep_without_power_fails_fast_with_clear_error(self) -> None:
        power = mock.Mock()
        power.active = False
        with mock.patch("nomorals.agents.power.power_mode_for",
                        return_value=power):
            job_id = jobs.start(self.ctx, "deep question", mode="deep")
        st = jobs.status(self.ctx, job_id)
        self.assertEqual(st["state"], "failed")
        self.assertIn("power", st["error"].lower())

    def test_list_jobs_newest_first(self) -> None:
        ids = [jobs.start(self.ctx, f"q{i}") for i in range(3)]
        for job_id in ids:
            _wait_state(self.ctx, job_id, ("done", "failed"))
        listed = jobs.list_jobs(self.ctx, limit=10)
        listed_ids = [j["id"] for j in listed]
        for job_id in ids:
            self.assertIn(job_id, listed_ids)
        # newest first: q2 (created last) must sort ahead of q1, ahead of q0
        first_seen = [listed_ids.index(i) for i in ids]
        self.assertEqual(first_seen, sorted(first_seen, reverse=True))


class QuickScopeSpreadTests(unittest.TestCase):
    """The quick engine's scope-spread picker: both regions represented."""

    def test_scope_spread_covers_both_scopes(self) -> None:
        results = [
            {"url": f"https://us{i}.com/x", "score": 0.9 - i * 0.1, "_scope": "us"}
            for i in range(4)
        ] + [
            {"url": f"https://ng{i}.com/x", "score": 0.5 - i * 0.1, "_scope": "ng"}
            for i in range(4)
        ]
        picked = engine_mod.SearchEngine._scope_spread(results, 3)
        scopes = {r["_scope"] for r in picked}
        self.assertIn("us", scopes)
        self.assertIn("ng", scopes)
        self.assertEqual(len(picked), 3)

    def test_scope_spread_single_scope_degrades_gracefully(self) -> None:
        results = [{"url": f"https://g{i}.com/x", "score": 0.9 - i * 0.1,
                    "_scope": "global"} for i in range(4)]
        picked = engine_mod.SearchEngine._scope_spread(results, 2)
        self.assertEqual(len(picked), 2)
        # best-first by score
        self.assertEqual(picked[0]["url"], "https://g0.com/x")

    def test_scope_spread_empty(self) -> None:
        self.assertEqual(engine_mod.SearchEngine._scope_spread([], 3), [])


if __name__ == "__main__":
    unittest.main()
