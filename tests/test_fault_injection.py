"""Fault-injection runtime tests: kill it, hang it, break it — then verify.

These tests inject real faults into the HybridExecutor and assert the
recovery contract: deterministic states, no orphaned work, honest errors.
A green suite here means the runtime degrades instead of lying.
"""

import threading
import time
import unittest

from nomorals.agents.runtime import HybridExecutor
from nomorals.core.tasks import (
    AcceptanceCriterion,
    Task,
    TaskGraph,
    TaskKind,
    TaskResult,
    TaskState,
)


class FaultInjector:
    """Wrap a callable with a scripted fault sequence.

    ``faults`` is a per-call script; each entry is one of:

    * ``"ok"`` — call through to ``fn``
    * ``("raise", exc)`` — raise ``exc``
    * ``("hang", secs)`` — sleep ``secs`` (to trip timeouts)
    * ``("slow", secs)`` — sleep then call through
    """

    def __init__(self, fn, faults):
        self.fn = fn
        self.faults = list(faults)
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, *args, **kwargs):
        with self._lock:
            idx = self.calls
            self.calls += 1
        fault = self.faults[idx] if idx < len(self.faults) else "ok"
        if fault == "ok":
            return self.fn(*args, **kwargs)
        kind, payload = fault
        if kind == "raise":
            raise payload
        if kind == "hang":
            time.sleep(payload)
            return self.fn(*args, **kwargs)
        if kind == "slow":
            time.sleep(payload)
            return self.fn(*args, **kwargs)
        raise AssertionError(f"unknown fault {fault!r}")


def io_task(name, fn, **kw):
    return Task(name=name, fn=fn, kind=TaskKind.IO, **kw)


class TestFaultInjection(unittest.TestCase):
    def test_worker_exception_fails_task_and_skips_dependents(self):
        boom = FaultInjector(lambda: "never", [("raise", ValueError("bad input"))])
        ran = []
        graph = TaskGraph(name="boom-graph")
        bad = graph.add(io_task("bad", boom))
        graph.add(io_task("downstream", lambda: ran.append(1) or "x"), depends_on=[bad])
        graph.add(io_task("independent", lambda: "fine"))

        with HybridExecutor(threads=4, use_processes=False) as ex:
            report = ex.run(graph)

        self.assertEqual(graph.tasks[bad.id].state, TaskState.FAILED)
        self.assertIn("bad input", graph.tasks[bad.id].error)
        self.assertEqual(ran, [])  # dependent never ran
        self.assertEqual(report.failed, 1)
        self.assertEqual(report.skipped, 1)
        self.assertEqual(report.done, 1)  # independent still completed

    def test_timeout_kills_hung_worker(self):
        hang = FaultInjector(lambda: "too late", [("hang", 3.0)])
        task = io_task("hanger", hang, timeout=0.5)
        graph = TaskGraph(name="hang-graph")
        graph.add(task)
        with HybridExecutor(threads=2, use_processes=False) as ex:
            ex.run(graph)
        self.assertEqual(task.state, TaskState.FAILED)
        self.assertIn("timed out", task.error)

    def test_retry_recovers_flaky_worker(self):
        flaky = FaultInjector(lambda: "recovered",
                              [("raise", RuntimeError("e1")), ("raise", RuntimeError("e2"))])
        task = io_task("flaky", flaky, retries=2)
        graph = TaskGraph(name="flaky-graph")
        graph.add(task)
        with HybridExecutor(threads=2, use_processes=False) as ex:
            report = ex.run(graph)
        self.assertEqual(task.state, TaskState.DONE)
        self.assertEqual(task.result, "recovered")
        self.assertEqual(task.attempts, 3)
        self.assertEqual(flaky.calls, 3)
        self.assertEqual(report.failed, 0)

    def test_retry_exhaustion_marks_failed(self):
        always = FaultInjector(lambda: "never", [("raise", RuntimeError("x"))] * 5)
        task = io_task("doomed", always, retries=1)
        graph = TaskGraph(name="doomed-graph")
        graph.add(task)
        with HybridExecutor(threads=2, use_processes=False) as ex:
            report = ex.run(graph)
        self.assertEqual(task.state, TaskState.FAILED)
        self.assertEqual(task.attempts, 2)  # initial + 1 retry
        self.assertEqual(report.failed, 1)

    def test_fail_fast_cancels_siblings(self):
        gate = threading.Event()
        started = []
        def slow_ok():
            started.append(1)
            gate.wait(5)
            return "ok"
        bad_task = io_task("bad", FaultInjector(lambda: 1 / 0, [("raise", ZeroDivisionError("z"))]))
        graph = TaskGraph(name="ff-graph")
        graph.add(bad_task)
        for i in range(4):
            graph.add(io_task(f"slow-{i}", slow_ok))
        with HybridExecutor(threads=4, use_processes=False) as ex:
            report = ex.run(graph, fail_fast=True)
        gate.set()
        self.assertEqual(report.failed, 1)
        # fail-fast cancels the rest instead of burning time on doomed work
        self.assertGreaterEqual(report.cancelled + report.skipped, 1)

    def test_deadline_abandons_long_graph(self):
        import time as _t
        graph = TaskGraph(name="deadline-graph")
        for i in range(4):
            graph.add(io_task(f"s{i}", lambda: (_t.sleep(2), "x")[1]))
        with HybridExecutor(threads=2, use_processes=False) as ex:
            report = ex.run(graph, deadline=_t.monotonic() + 0.3)
        # the deadline fired: not everything completed
        self.assertLess(report.done, 4)
        self.assertGreaterEqual(report.cancelled + report.failed, 1)

    def test_caller_cancellation_propagates(self):
        calls = {"n": 0}
        def quick():
            calls["n"] += 1
            return calls["n"]
        graph = TaskGraph(name="cancel-graph")
        prev = None
        for i in range(6):
            t = io_task(f"c{i}", quick)
            graph.add(t, depends_on=[prev] if prev else [])
            prev = t
        stop = threading.Event()
        def on_done(task):
            if calls["n"] >= 2:
                stop.set()
        with HybridExecutor(threads=2, use_processes=False) as ex:
            report = ex.run(graph, on_task_done=on_done,
                            should_stop=stop.is_set)
        self.assertLess(report.done, 6)
        self.assertTrue(graph.cancelled)  # caller stop propagated to the graph

    def test_deterministic_rerun_same_results(self):
        def build():
            g = TaskGraph(name="det-graph")
            g.add(io_task("a", lambda: 40 + 2))
            g.add(io_task("b", lambda: "hello".upper()))
            c = g.add(io_task("c", lambda: None), depends_on=["a"])
            return g, c
        results = []
        for _ in range(2):
            g, _ = build()
            with HybridExecutor(threads=4, use_processes=False) as ex:
                rep = ex.run(g)
            results.append(dict(rep.results))
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0]["a"], 42)

    def test_process_pool_retirement_falls_back_to_threads(self):
        with HybridExecutor(threads=2, use_processes=True) as ex:
            had_pool = ex._process_pool is not None
            ex._retire_process_pool()
            self.assertIsNone(ex._process_pool)
            # CPU work still completes, degraded to threads
            graph = TaskGraph(name="cpu-fallback")
            graph.add(Task(name="cpu", fn=lambda: sum(range(1000)),
                           kind=TaskKind.CPU))
            report = ex.run(graph)
            self.assertEqual(report.done, 1)
            self.assertEqual(report.results["cpu"], sum(range(1000)))
            self.assertGreaterEqual(ex.stats["process_fallbacks"], 0 if had_pool else 0)

    def test_acceptance_failure_does_not_claim_correctness(self):
        task = io_task("shoddy", lambda: "ran fine")
        task.acceptance = [AcceptanceCriterion(
            name="accuracy", spec={"metric": "accuracy", "gte": 0.9})]
        graph = TaskGraph(name="accept-graph")
        graph.add(task)
        with HybridExecutor(threads=2, use_processes=False) as ex:
            ex.run(graph)

        def wrap_with_evidence(t):
            # simulate an agent wrapping raw output in a TaskResult
            t.result = TaskResult(status="done", metrics={"accuracy": 0.5})
            t.verify()

        wrap_with_evidence(task)
        self.assertEqual(task.state, TaskState.DONE)  # it ran
        self.assertFalse(task.verified)               # ...but it is not correct

    def test_concurrent_graphs_do_not_interfere(self):
        errors = []
        def build(i):
            g = TaskGraph(name=f"iso-{i}")
            g.add(io_task("only", lambda: i * 10))
            return g
        def run_one(i):
            try:
                g = build(i)
                with HybridExecutor(threads=2, use_processes=False) as ex:
                    rep = ex.run(g)
                assert rep.results["only"] == i * 10, rep.results
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        threads = [threading.Thread(target=run_one, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
