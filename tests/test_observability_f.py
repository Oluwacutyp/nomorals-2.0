"""Section F observability upgrades: logging, metrics, tracing, health, tasks, clock."""

import json
import logging
import os
import random
import tempfile
import time
import unittest

from nomorals.core.clock import Backoff, Deadline, FrozenClock, SystemClock
from nomorals.core.logging_setup import (
    LogCapture,
    RedactionFilter,
    SampleFilter,
    audit_log,
    bind_log_context,
    clear_log_context,
    json_formatter,
    log_context,
    redact,
    setup_logging,
    shutdown_logging,
)
from nomorals.core.observability import (
    HealthChecker,
    Metrics,
    MetricsExporter,
    Tracer,
    trace,
)
from nomorals.core.tasks import (
    RetryBudget,
    RetryPolicy,
    Task,
    TaskGraph,
    TaskKind,
    TaskState,
)
from nomorals.core.errors import ValidationError


class TestLogContext(unittest.TestCase):
    def setUp(self):
        clear_log_context()

    def tearDown(self):
        clear_log_context()

    def test_bind_injects_into_json(self):
        fmt = json_formatter()
        bind_log_context(trace_id="trc_abc", mission_id="m1", agent="devon")
        record = logging.LogRecord("x", logging.INFO, __file__, 1, "hello", (), None)
        line = fmt.format(record)
        payload = json.loads(line)
        self.assertEqual(payload["context"]["trace_id"], "trc_abc")
        self.assertEqual(payload["context"]["mission_id"], "m1")
        self.assertEqual(payload["context"]["agent"], "devon")

    def test_log_context_manager_clears(self):
        with log_context(trace_id="t1"):
            self.assertEqual(__import__("nomorals.core.logging_setup", fromlist=["_log_context"])._log_context.get()["trace_id"], "t1")
        self.assertEqual(__import__("nomorals.core.logging_setup", fromlist=["_log_context"])._log_context.get(), {})

    def test_nested_context_merges(self):
        with log_context(trace_id="outer", agent="a"):
            with log_context(task_id="tk1"):
                ctx = __import__("nomorals.core.logging_setup", fromlist=["_log_context"])._log_context.get()
                self.assertEqual(ctx["trace_id"], "outer")
                self.assertEqual(ctx["task_id"], "tk1")

    def test_extras_survive_json(self):
        fmt = json_formatter()
        record = logging.LogRecord("x", logging.INFO, __file__, 1, "hello", (), None)
        record.custom_field = "yes"
        payload = json.loads(fmt.format(record))
        self.assertEqual(payload["custom_field"], "yes")


class TestRedaction(unittest.TestCase):
    def test_free_text(self):
        self.assertIn("[REDACTED_KEY]", redact("key is sk_abcdefghijklmnop here"))
        # Header-shaped secrets are scrubbed to end-of-line first (documented order).
        self.assertIn("Authorization=[REDACTED]", redact("Authorization: Bearer deadbeefcafe1234"))
        self.assertIn("[REDACTED_AUTH]", redact("retry with Bearer deadbeefcafe1234 inline"))
        self.assertIn("[REDACTED_AWS]", redact("id AKIAIOSFODNN7EXAMPLE here"))
        self.assertIn("[REDACTED_JWT]", redact("jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.SflKxwRJABCdefg"))

    def test_dict_payload_scrubbed_by_key(self):
        record = logging.LogRecord("x", logging.INFO, __file__, 1, "login", (), None)
        record.payload = {"user": "bob", "password": "hunter2", "nested": {"token": "abc"}}
        RedactionFilter().filter(record)
        self.assertEqual(record.payload["password"], "[REDACTED]")
        self.assertEqual(record.payload["nested"]["token"], "[REDACTED]")
        self.assertEqual(record.payload["user"], "bob")

    def test_fingerprint_keys(self):
        record = logging.LogRecord("x", logging.INFO, __file__, 1, "x", (), None)
        record.payload = {"email": "a@b.com"}
        RedactionFilter().filter(record)
        self.assertTrue(record.payload["email"].startswith("sha256:"))
        self.assertNotIn("a@b.com", record.payload["email"])

    def test_filter_never_raises(self):
        record = logging.LogRecord("x", logging.INFO, __file__, 1, object(), (), None)
        self.assertTrue(RedactionFilter().filter(record))


class TestSampling(unittest.TestCase):
    def test_errors_always_kept(self):
        f = SampleFilter({"default": 0.0})
        rec = logging.LogRecord("x", logging.ERROR, __file__, 1, "boom", (), None)
        self.assertTrue(f.filter(rec))

    def test_deterministic_sampling(self):
        f = SampleFilter({"default": 0.5})
        rec1 = logging.LogRecord("x", logging.INFO, __file__, 1, "same", (), None)
        rec1.trace_id = "trc_same"
        rec2 = logging.LogRecord("x", logging.INFO, __file__, 1, "same", (), None)
        rec2.trace_id = "trc_same"
        self.assertEqual(f.filter(rec1), f.filter(rec2))

    def test_zero_rate_drops(self):
        f = SampleFilter({"INFO": 0.0})
        rec = logging.LogRecord("x", logging.INFO, __file__, 1, "quiet", (), None)
        self.assertFalse(f.filter(rec))


class TestAuditLog(unittest.TestCase):
    def test_audit_marker(self):
        logging.getLogger("nomorals.test.audit").setLevel(logging.DEBUG)
        logger = audit_log("test.audit")
        with LogCapture("nomorals.test.audit") as cap:
            logger.info("auth decision", extra={"user": "x"})
        self.assertTrue(cap.records)
        self.assertTrue(any(getattr(r, "audit", False) for r in cap.records))


class TestSetupLogging(unittest.TestCase):
    def test_queue_mode_delivers(self):
        with tempfile.TemporaryDirectory() as tmp:
            logfile = os.path.join(tmp, "app.log")
            setup_logging("INFO", file=logfile, queue=True, force=True)
            log = logging.getLogger("nomorals.qtest")
            log.info("queued message")
            shutdown_logging()
            with open(logfile) as fh:
                content = fh.read()
            self.assertIn("queued message", content)
            setup_logging("INFO", force=True)  # restore console-only

    def test_sample_rates_param(self):
        with tempfile.TemporaryDirectory() as tmp:
            logfile = os.path.join(tmp, "s.log")
            setup_logging("DEBUG", file=logfile, sample_rates={"DEBUG": 0.0}, force=True)
            log = logging.getLogger("nomorals.stest")
            log.debug("should vanish")
            log.error("should stay")
            with open(logfile) as fh:
                content = fh.read()
            self.assertNotIn("should vanish", content)
            self.assertIn("should stay", content)
            setup_logging("INFO", force=True)


class TestMetricsLabels(unittest.TestCase):
    def test_prometheus_includes_labels(self):
        m = Metrics()
        m.incr("tool_calls", 3, tool="search", status="ok")
        m.incr("tool_calls", 1, tool="exec", status="error")
        m.set_gauge("queue_depth", 7, kind="io")
        out = m.as_prometheus()
        self.assertIn('nomorals_tool_calls{status="ok",tool="search"} 3', out)
        self.assertIn('nomorals_tool_calls{status="error",tool="exec"} 1', out)
        self.assertIn('nomorals_queue_depth{kind="io"} 7', out)
        # unlabeled totals still present
        self.assertIn("nomorals_tool_calls 4", out)

    def test_exporter_flush(self):
        m = Metrics()
        m.incr("jobs", 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "metrics.json")
            ex = MetricsExporter(m, path, interval=60)
            ex.flush()
            with open(path) as fh:
                snap = json.load(fh)
            self.assertEqual(snap["counters"]["jobs"], 2)


class TestTracer(unittest.TestCase):
    def test_nesting_and_error_retention(self):
        tr = Tracer(sample_rate=0.0)  # sample nothing...
        with self.assertRaises(RuntimeError):
            with tr.span("root") as root:
                with tr.span("child") as child:
                    self.assertEqual(child.parent_id, root.span_id)
                    self.assertEqual(child.trace_id, root.trace_id)
                    raise RuntimeError("kablam")
        # ...but the error trace is still retained (tail rule)
        traces = tr.traces()
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]["status"], "error")
        self.assertIn("exception", [e["name"] for e in traces[0]["children"][0]["events"]])
        self.assertEqual(tr.dropped_spans, 0)

    def test_ok_span_dropped_when_unsampled(self):
        tr = Tracer(sample_rate=0.0)
        with tr.span("quiet"):
            pass
        self.assertEqual(tr.traces(), [])
        self.assertEqual(tr.dropped_spans, 1)

    def test_record_exception_and_redaction(self):
        tr = Tracer()
        try:
            with tr.span("s", password="hunter2", safe="yes") as span:
                raise ValueError("token=sk_abcdefghijklmnop")
        except ValueError:
            pass
        kept = tr.traces()[0]
        self.assertNotIn("password", kept["attributes"])
        self.assertEqual(kept["attributes"]["safe"], "yes")
        self.assertNotIn("sk_abcdefghijklmnop", kept["error"])

    def test_traceparent_roundtrip(self):
        tr = Tracer()
        with tr.span("a") as a:
            header = a.traceparent()
            with tr.span("b", traceparent=header) as b:
                pass
        self.assertEqual(b.trace_id, a.trace_id)
        self.assertEqual(b.parent_id, a.span_id)

    def test_drain_to_jsonl(self):
        tr = Tracer()
        with tr.span("x"):
            pass
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "traces.jsonl")
            n = tr.drain_to_jsonl(path)
            self.assertEqual(n, 1)
            with open(path) as fh:
                line = json.loads(fh.readline())
            self.assertEqual(line["name"], "x")
            self.assertEqual(tr.traces(), [])  # drained

    def test_otlp_shape(self):
        tr = Tracer()
        with tr.span("op") as span:
            otlp = span.to_otlp()
        self.assertEqual(otlp["name"], "op")
        self.assertIn("traceId", otlp)


class TestHealthChecker(unittest.TestCase):
    def test_run_and_split(self):
        hc = HealthChecker(version="t")
        hc.register("db", lambda: {"ok": True, "detail": "connected"}, kind="readiness", ttl=60)
        hc.register("proc", lambda: {"ok": True, "detail": "alive"}, kind="liveness")
        full = hc.run()
        self.assertTrue(full.healthy)
        self.assertIn("db", full.checks)
        self.assertIn("proc", full.checks)
        self.assertEqual(len(hc.liveness().checks), 1)
        self.assertEqual(len(hc.readiness().checks), 1)

    def test_failing_critical_degrades(self):
        hc = HealthChecker(version="t")
        hc.register("broken", lambda: {"ok": False, "detail": "down"}, critical=True)
        rep = hc.run()
        self.assertFalse(rep.healthy)
        self.assertIn("DEGRADED", rep.as_text())

    def test_noncritical_does_not_degrade(self):
        hc = HealthChecker(version="t")
        hc.register("flaky", lambda: {"ok": False, "detail": "meh"}, critical=False)
        self.assertTrue(hc.run().healthy)

    def test_check_never_raises_and_timeout(self):
        hc = HealthChecker(version="t")
        hc.register("raises", lambda: 1 / 0)
        hc.register("slow", lambda: (time.sleep(2), {"ok": True})[1], timeout=0.2)
        rep = hc.run()
        self.assertFalse(rep.checks["raises"]["ok"])
        self.assertIn("ZeroDivisionError", rep.checks["raises"]["detail"])
        self.assertFalse(rep.checks["slow"]["ok"])
        self.assertIn("timed out", rep.checks["slow"]["detail"])

    def test_transitions_tracked(self):
        state = {"ok": True}
        hc = HealthChecker(version="t")
        hc.register("flip", lambda: {"ok": state["ok"], "detail": ""})
        hc.run()
        state["ok"] = False
        rep = hc.run()
        self.assertEqual(rep.checks["flip"]["transitions"], 1)
        self.assertEqual(rep.checks["flip"]["consecutive_failures"], 1)

    def test_default_checks(self):
        hc = HealthChecker(version="t")
        hc.register_default_checks()
        rep = hc.liveness()
        self.assertTrue(rep.healthy)
        self.assertIn("memory", rep.checks)


class TestRetryPolicy(unittest.TestCase):
    def test_backoff_shape(self):
        rng = random.Random(0)
        p = RetryPolicy(max_retries=3, base_delay=1.0, factor=2.0, jitter=False)
        self.assertEqual(p.delay_for(1), 1.0)
        self.assertEqual(p.delay_for(2), 2.0)
        self.assertEqual(p.delay_for(3), 4.0)
        self.assertTrue(p.should_retry(IOError("x"), 3))
        self.assertFalse(p.should_retry(IOError("x"), 4))

    def test_retryable_classes(self):
        p = RetryPolicy(retryable=(IOError,))
        self.assertTrue(p.should_retry(IOError("x"), 1))
        self.assertFalse(p.should_retry(ValueError("permanent"), 1))

    def test_jitter_bounds(self):
        p = RetryPolicy(base_delay=10.0, jitter=True, rng=random.Random(1))
        for _ in range(50):
            d = p.delay_for(1)
            self.assertGreaterEqual(d, 0.0)
            self.assertLessEqual(d, 10.0)

    def test_retry_after_hint_wins(self):
        p = RetryPolicy(base_delay=60.0, jitter=False)
        exc = IOError("rate limited")
        exc.retry_after = 5
        self.assertEqual(p.delay_for(1, exc), 5.0)

    def test_max_delay_cap(self):
        p = RetryPolicy(base_delay=100.0, max_delay=30.0, jitter=False)
        self.assertEqual(p.delay_for(5), 30.0)


class TestRetryBudget(unittest.TestCase):
    def test_exhaustion(self):
        b = RetryBudget(max_retries=2)
        self.assertTrue(b.spend())
        self.assertTrue(b.spend())
        self.assertFalse(b.spend())
        self.assertEqual(b.remaining(), 0)
        b.reset()
        self.assertEqual(b.remaining(), 2)


class TestTaskGraphUpgrades(unittest.TestCase):
    def test_duplicate_idempotency_key_rejected(self):
        g = TaskGraph()
        g.add(Task(name="a", idempotency_key="k1"))
        with self.assertRaises(ValidationError):
            g.add(Task(name="b", idempotency_key="k1"))

    def test_retry_later_sets_not_before(self):
        g = TaskGraph()
        t = g.add(Task(name="a"))
        t.mark_failed("boom")
        ok = g.retry_later(t, 60.0, "retrying")
        self.assertTrue(ok)
        self.assertEqual(t.state, TaskState.PENDING)
        self.assertGreater(t.not_before, time.monotonic())
        self.assertEqual(g.ready(), [])  # backoff window not elapsed
        t.not_before = 0.0
        self.assertEqual([x.name for x in g.ready()], ["a"])

    def test_retry_budget_exhaustion_fails_fast(self):
        g = TaskGraph(retry_budget=RetryBudget(max_retries=0))
        t = g.add(Task(name="a"))
        t.mark_failed("boom")
        ok = g.retry_later(t, 1.0)
        self.assertFalse(ok)
        self.assertEqual(t.state, TaskState.FAILED)
        self.assertIn("budget exhausted", t.error)

    def test_cascade_skip_marks_dependents(self):
        g = TaskGraph()
        a = g.add(Task(name="a"))
        b = g.add(Task(name="b"), depends_on=[a])
        c = g.add(Task(name="c"), depends_on=[b])
        a.mark_failed("root cause")
        n = g.cascade_skip()
        self.assertEqual(n, 2)
        self.assertEqual(b.state, TaskState.SKIPPED)
        self.assertEqual(c.state, TaskState.SKIPPED)
        self.assertIn("a", b.error)  # names the failed ancestor

    def test_dead_letters_and_replay(self):
        g = TaskGraph()
        t = g.add(Task(name="a"))
        t.mark_failed("disk full")
        dead = g.dead_letters()
        self.assertEqual(len(dead), 1)
        self.assertEqual(dead[0]["error"], "disk full")
        self.assertTrue(dead[0]["replayable"])
        replayed = g.requeue_dead("a")
        self.assertEqual(replayed.state, TaskState.PENDING)
        with self.assertRaises(ValidationError):
            g.requeue_dead("a")  # not failed anymore

    def test_dry_run(self):
        g = TaskGraph()
        calls = []
        a = g.add(Task(name="a", fn=lambda: calls.append("a") or 1))
        b = g.add(Task(name="b", fn=lambda: calls.append("b") or 2), depends_on=[a])
        c = g.add(Task(name="c", fn=lambda: 1 / 0), depends_on=[a])
        d = g.add(Task(name="d", fn=lambda: calls.append("d")), depends_on=[c])
        summary = g.dry_run()
        self.assertEqual(summary["ran"], 2)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["skipped"], 1)
        self.assertEqual(calls, ["a", "b"])
        self.assertIn("c", summary["failures"])

    def test_hooks_notified(self):
        g = TaskGraph()
        seen = []
        g.add_hook(lambda task, old, new: seen.append((task.name, old, new)))
        t = g.add(Task(name="a"))
        t.mark_failed("x")
        g.cascade_skip()
        g.retry_later(t, 0.0)
        self.assertTrue(any(s[2] is TaskState.PENDING for s in seen))

    def test_hook_exception_does_not_break_graph(self):
        g = TaskGraph()
        g.add_hook(lambda *a: 1 / 0)
        t = g.add(Task(name="a"))
        g.retry_later(t, 0.0)  # must not raise

    def test_add_task_accepts_new_kwargs(self):
        g = TaskGraph()
        t = g.add_task("a", lambda: 1, retry_policy=RetryPolicy(max_retries=5),
                       idempotency_key="k9")
        self.assertEqual(t.max_attempts, 6)
        self.assertEqual(t.idempotency_key, "k9")


class TestClockUpgrades(unittest.TestCase):
    def test_deadline(self):
        clock = FrozenClock(100.0)
        dl = Deadline.after(30, clock)
        self.assertAlmostEqual(dl.remaining(), 30.0)
        self.assertFalse(dl.expired())
        clock.advance(31)
        self.assertTrue(dl.expired())
        self.assertEqual(dl.remaining(), 0.0)
        with self.assertRaises(TimeoutError):
            dl.raise_if_expired()

    def test_deadline_child_respects_parent(self):
        clock = FrozenClock()
        parent = Deadline.after(10, clock)
        child = parent.child(60)
        self.assertAlmostEqual(child.remaining(), 10.0)

    def test_timeout_contextmanager(self):
        clock = FrozenClock()
        with clock.timeout(5) as dl:
            self.assertIsInstance(dl, Deadline)
            self.assertAlmostEqual(dl.remaining(), 5.0)

    def test_backoff_advances_frozen_clock(self):
        clock = FrozenClock()
        b = Backoff(base=2.0, jitter=False, clock=clock)
        waited = b.wait()
        self.assertEqual(waited, 2.0)
        self.assertEqual(clock.monotonic(), 2.0)
        self.assertEqual(b.attempt, 1)
        b.reset()
        self.assertEqual(b.attempt, 0)

    def test_backoff_real_clock_fast(self):
        b = Backoff(base=0.001, jitter=False, clock=SystemClock())
        b.wait()
        self.assertEqual(b.attempt, 1)

    def test_sleep_until(self):
        clock = FrozenClock()
        clock.sleep_until(10.0)
        self.assertEqual(clock.monotonic(), 10.0)
        clock.sleep_until(5.0)  # past: no-op
        self.assertEqual(clock.monotonic(), 10.0)


class TestGlobalShorthands(unittest.TestCase):
    def test_trace_shorthand(self):
        with trace("smoke") as span:
            self.assertIsNotNone(span.trace_id)
        from nomorals.core.observability import global_tracer
        self.assertEqual(len(global_tracer.traces(limit=1)), 1)


if __name__ == "__main__":
    unittest.main()
