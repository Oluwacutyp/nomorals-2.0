"""Tests for section E upgrades: events reliability, error intelligence
learning, error doctor analyzers, retry/resilience pipeline, rate limiters,
self-heal learning loop.

All offline and fast.
"""

import os
import tempfile
import textwrap
import time
import unittest

from nomorals.core.error_doctor import diagnose
from nomorals.core.error_intelligence import (
    ErrorFingerprint,
    ErrorIntelligence,
    ErrorKnowledgeBase,
)
from nomorals.core.errors import (
    NoMoralsError,
    ProviderUnavailable,
    RateLimited,
    error_codes,
    resolve_error,
    retry_after_of,
)
from nomorals.core.events import BusOverloaded, Event, EventBus
from nomorals.core.ratelimit import (
    GCRALimiter,
    RateLimitExceeded,
    SlidingWindowCounter,
    TokenBucket,
)
from nomorals.core.retry import (
    BackoffPolicy,
    CircuitBreaker,
    FallbackPolicy,
    HedgingPolicy,
    JitterStyle,
    ResiliencePipeline,
    RetryBudget,
    TimeoutPolicy,
    retry_call,
)
from nomorals.core.self_heal import (
    FIX_STRATEGIES,
    FixStrategy,
    heal_all,
    heal_one,
    heal_probe,
    register_fix_strategy,
)


def _kb():
    tmp = tempfile.mkdtemp()
    return os.path.join(tmp, "kb.json")


class TestErrors(unittest.TestCase):
    def test_registry_and_round_trip(self):
        self.assertIs(resolve_error("model.provider.rate_limited"), RateLimited)
        self.assertGreater(len(error_codes()), 27)
        e = RateLimited("slow", retry_after=12.5, details={"x": 1})
        e2 = NoMoralsError.from_dict(e.to_dict())
        self.assertIsInstance(e2, RateLimited)
        self.assertEqual(e2.retry_after, 12.5)

    def test_retry_after_of(self):
        self.assertEqual(retry_after_of(RateLimited("x", retry_after=3)), 3.0)
        self.assertIsNone(retry_after_of(ValueError("v")))


class TestEventBusReliability(unittest.TestCase):
    def test_seq_ordering(self):
        bus = EventBus().start()
        try:
            got = []
            bus.subscribe("o.*", lambda e: got.append(e.seq))
            for _ in range(20):
                bus.emit("o.x")
            time.sleep(0.8)
            self.assertEqual(len(got), 20)
            self.assertEqual(got, sorted(got))
        finally:
            bus.stop()

    def test_handler_retry_then_dlq(self):
        bus = EventBus().start()
        try:
            n = {"c": 0}

            def bad(e):
                n["c"] += 1
                raise RuntimeError("kaput")

            bus.subscribe("d", bad, retries=2, retry_backoff=0.02)
            bus.emit("d")
            time.sleep(0.8)
            self.assertEqual(n["c"], 3)
            dlq = bus.dlq()
            self.assertEqual(len(dlq), 1)
            self.assertEqual(dlq[0]["attempts"], 3)
            # surfaced on eventbus.error
            topics = [e.topic for e in bus.history(100)]
            self.assertIn("eventbus.error", topics)
        finally:
            bus.stop()

    def test_dlq_replay_and_redirect(self):
        bus = EventBus().start()
        try:
            state = {"broken": True}
            seen = []

            def h(e):
                seen.append(e.topic)
                if state["broken"]:
                    raise RuntimeError("x")

            sid = bus.subscribe("r.*", h)
            bus.emit("r.a")
            time.sleep(0.5)
            self.assertEqual(bus.stats()["dlq_depth"], 1)
            r = bus.replay_dlq()
            self.assertEqual(r["failed"], 1)
            state["broken"] = False
            r = bus.replay_dlq()
            self.assertEqual(r["succeeded"], 1)
            self.assertEqual(bus.stats()["dlq_depth"], 0)
            # redirect: original sub gone, new matching handler picks it up
            bus.unsubscribe(sid)

            def bad2(e):
                raise RuntimeError("y")

            sid2 = bus.subscribe("r.*", bad2)
            bus.emit("r.b")
            time.sleep(0.5)
            bus.unsubscribe(sid2)
            got = []
            bus.subscribe("r.*", lambda e: got.append(e.topic))
            r = bus.replay_dlq()
            self.assertEqual(r["redirected"], 1)
            self.assertEqual(got, ["r.b"])
        finally:
            bus.stop()

    def test_dedupe_window(self):
        bus = EventBus(dedupe_window=50)
        hits = []
        bus.subscribe("dup", lambda e: hits.append(1), sync=True)
        ev = Event(topic="dup")
        bus.publish(ev)
        bus.publish(ev)
        self.assertEqual(hits, [1])
        self.assertEqual(bus.stats()["dedup_dropped"], 1)

    def test_backpressure_raise(self):
        import threading

        bus = EventBus(max_queue=1, on_full="raise")
        gate = threading.Event()
        bus.subscribe("x", lambda e: gate.wait(3))
        bus.start()
        try:
            bus.emit("x")
            time.sleep(0.2)
            with self.assertRaises(BusOverloaded):
                for _ in range(30):
                    bus.emit("x")
        finally:
            gate.set()
            bus.stop()

    def test_journal_replay(self):
        tmp = tempfile.mkdtemp()
        jp = os.path.join(tmp, "j.jsonl")
        w = EventBus(journal_path=jp)
        for i in range(4):
            w.emit("jt", i=i)
        w.stop()
        back = EventBus(journal_path=jp).journal_replay(since_seq=2, topic="jt")
        self.assertEqual([e.data["i"] for e in back], [2, 3])
        r = EventBus()
        got = []
        r.subscribe("jt", lambda e: got.append(e.data["i"]), sync=True)
        self.assertEqual(r.replay(back), 2)
        self.assertEqual(got, [2, 3])


class TestErrorIntelligence(unittest.TestCase):
    def test_fingerprint_groups_dynamic_values(self):
        a = ConnectionError("Connection refused to 127.0.0.1:6379")
        b = ConnectionError("Connection refused to 10.0.0.1:6379")
        self.assertEqual(ErrorFingerprint.of(a), ErrorFingerprint.of(b))
        self.assertNotEqual(ErrorFingerprint.of(a),
                            ErrorFingerprint.of(TimeoutError("timed out")))

    def test_learn_and_demote(self):
        ei = ErrorIntelligence(store_path=_kb())
        try:
            raise ConnectionError("Connection refused to db:5432")
        except Exception as e:
            a1 = ei.analyze(e)
        ei.learn_fix(a1.fingerprint, "restart postgres")
        try:
            raise ConnectionError("Connection refused to db:5432")
        except Exception as e:
            a2 = ei.analyze(e)
        self.assertEqual(a2.suggested_fix, "restart postgres")
        self.assertEqual(a2.fix_source, "learned")
        for _ in range(3):
            ei.record_outcome(a2, fixed=False)
        try:
            raise ConnectionError("Connection refused to db:5432")
        except Exception as e:
            a3 = ei.analyze(e)
        self.assertNotEqual(a3.fix_source, "learned")

    def test_persistence_and_regression(self):
        path = _kb()
        ei = ErrorIntelligence(store_path=path)
        try:
            raise RuntimeError("persist me 12345")
        except Exception as e:
            a1 = ei.analyze(e)
        ei.flush()
        ei2 = ErrorIntelligence(store_path=path)
        groups = {g["fingerprint"]: g for g in ei2.groups()}
        self.assertIn(a1.fingerprint, groups)
        self.assertTrue(ei2.resolve(a1.fingerprint))
        g = ei2.learning.get_group(a1.fingerprint)
        g.resolved_at = "2020-01-01T00:00:00+00:00"
        try:
            raise RuntimeError("persist me 12345")
        except Exception as e:
            a2 = ei2.analyze(e)
        self.assertTrue(a2.regression)

    def test_spike_and_escalation(self):
        ei = ErrorIntelligence(store_path=_kb())
        for _ in range(6):
            try:
                raise RuntimeError("spike unique abcdef")
            except Exception as e:
                a = ei.analyze(e)
        self.assertTrue(a.spike)

        class Weird(Exception):
            pass

        try:
            raise Weird("quux")
        except Exception as e:
            b = ei.analyze(e)
        self.assertTrue(b.needs_escalation)
        self.assertIn("ESCALATION", b.escalation_message())

    def test_retryable_from_hierarchy(self):
        ei = ErrorIntelligence(store_path=_kb())
        try:
            raise RateLimited("s", retry_after=5)
        except Exception as e:
            a = ei.analyze(e)
        self.assertTrue(a.retryable)
        self.assertEqual(a.retry_after, 5.0)
        try:
            raise ValueError("bad")
        except Exception as e:
            self.assertFalse(ei.analyze(e).retryable)

    def test_kb_match_compat(self):
        m = ErrorKnowledgeBase.match("connection refused")
        self.assertEqual(m["pattern_name"], "connection_refused")


class TestErrorDoctorNew(unittest.TestCase):
    def _mod(self, src):
        tmp = tempfile.mkdtemp()
        p = os.path.join(tmp, "m.py")
        with open(p, "w") as f:
            f.write(textwrap.dedent(src))
        import importlib.util

        spec = importlib.util.spec_from_file_location("doctest_mod", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _cap(self, fn, *a):
        try:
            fn(*a)
        except Exception as e:
            return e
        self.fail("did not raise")

    def test_index_error(self):
        mod = self._mod("def f():\n    xs=[1,2]\n    i=9\n    return xs[i]\n")
        d = diagnose(self._cap(mod.f))
        self.assertEqual(d["error_type"], "IndexError")
        self.assertIn("i=9", d["root_cause"])

    def test_zero_division(self):
        mod = self._mod("def f():\n    n=0\n    return 10/n\n")
        d = diagnose(self._cap(mod.f))
        self.assertIn("n=0", d["root_cause"])

    def test_assertion(self):
        mod = self._mod("def f():\n    x=1\n    assert x>5\n")
        d = diagnose(self._cap(mod.f))
        self.assertIn("x=1", d["root_cause"])

    def test_value_error_literal(self):
        mod = self._mod('def f():\n    return int("12px")\n')
        d = diagnose(self._cap(mod.f))
        self.assertIn("12px", d["root_cause"])

    def test_chained(self):
        mod = self._mod(
            "def f():\n"
            "    try:\n"
            "        open('/nope_xyz/q')\n"
            "    except OSError as e:\n"
            "        raise RuntimeError('boom') from e\n")
        d = diagnose(self._cap(mod.f))
        self.assertEqual(d["error_type"], "RuntimeError")
        self.assertEqual(d["cause"]["error_type"], "FileNotFoundError")


class TestRetryPipeline(unittest.TestCase):
    def test_jitter_styles_bounded(self):
        p = BackoffPolicy(base=1.0, factor=2.0, cap=8.0,
                          jitter_style=JitterStyle.NONE)
        self.assertEqual(p.delay(3), 4.0)
        p = BackoffPolicy(base=1.0, jitter_style=JitterStyle.EQUAL)
        for _ in range(30):
            self.assertTrue(1.0 <= p.delay(2) <= 2.0)
        p = BackoffPolicy(base=0.5, cap=10.0,
                          jitter_style=JitterStyle.DECORRELATED)
        prev = None
        for i in range(1, 10):
            d = p.delay(i, prev_delay=prev)
            self.assertTrue(0.5 <= d <= 10.0)
            prev = d

    def test_retry_honors_retry_after(self):
        n = {"c": 0}

        def rl():
            n["c"] += 1
            raise RateLimited("s", retry_after=0.02)

        t0 = time.monotonic()
        with self.assertRaises(RateLimited):
            retry_call(rl, policy=BackoffPolicy(max_attempts=2, base=30.0),
                       sleep=time.sleep)
        self.assertEqual(n["c"], 2)
        self.assertLess(time.monotonic() - t0, 10.0)

    def test_budget_denies_retries(self):
        n = {"c": 0}

        def fail():
            n["c"] += 1
            raise ProviderUnavailable("down")

        b = RetryBudget(retry_ratio=0.5, min_tokens=0.0)
        with self.assertRaises(ProviderUnavailable):
            retry_call(fail, policy=BackoffPolicy(max_attempts=5, base=0.0),
                       sleep=lambda s: None, budget=b)
        self.assertEqual(n["c"], 1)

    def test_timeout_and_fallback(self):
        tp = TimeoutPolicy(0.1)
        from nomorals.core.errors import TimeoutError_
        with self.assertRaises(TimeoutError_):
            tp.execute(time.sleep, 5)
        self.assertEqual(tp.execute(lambda: 1), 1)
        fp = FallbackPolicy.with_value("deg")
        self.assertEqual(fp.execute(lambda: 1 / 0), "deg")

    def test_hedging_first_success_wins(self):
        n = {"c": 0}

        def slow():
            n["c"] += 1
            time.sleep(0.3 if n["c"] == 1 else 0.01)
            return n["c"]

        self.assertEqual(HedgingPolicy(delay=0.05, max_hedges=1).execute(slow), 2)

    def test_pipeline_breaker_and_fallback(self):
        opened = {"n": 0}
        pipe = (ResiliencePipeline("t")
                .with_retry(BackoffPolicy(max_attempts=2, base=0.0))
                .with_breaker(failure_threshold=2, reset_timeout=60.0,
                              on_open=lambda nm: opened.update(n=1))
                .with_fallback(FallbackPolicy.with_value("fb")))
        n = {"c": 0}

        def flaky():
            n["c"] += 1
            raise ProviderUnavailable("down")

        self.assertEqual(pipe.execute(flaky), "fb")
        self.assertEqual(pipe.execute(flaky), "fb")
        self.assertEqual(opened["n"], 1)
        c0 = n["c"]
        self.assertEqual(pipe.execute(flaky), "fb")
        self.assertEqual(n["c"], c0)  # open breaker: fn not called


class TestRateLimiters(unittest.TestCase):
    def test_gcra_burst_exact(self):
        g = GCRALimiter(rate=2.0, period=1.0, burst=2)
        self.assertTrue(g.decide().allowed)
        self.assertTrue(g.decide().allowed)
        d = g.decide()
        self.assertFalse(d.allowed)
        self.assertGreater(d.retry_after, 0)

    def test_gcra_sustained(self):
        g = GCRALimiter(rate=20.0, period=1.0, burst=1)
        for _ in range(4):
            self.assertTrue(g.decide().allowed)
            time.sleep(0.06)

    def test_sliding_counter_exact_wait(self):
        s = SlidingWindowCounter(limit=2, window=0.4)
        self.assertTrue(s.try_acquire())
        self.assertTrue(s.try_acquire())
        self.assertFalse(s.try_acquire())
        w = s.wait_time()
        self.assertGreater(w, 0)
        time.sleep(w + 0.03)
        self.assertTrue(s.decide().allowed)

    def test_rate_limit_exceeded_unified(self):
        e = RateLimitExceeded(2.0)
        self.assertIsInstance(e, RateLimited)
        self.assertTrue(e.retryable)
        self.assertEqual(e.retry_after, 2.0)
        b = TokenBucket(rate=1.0, capacity=1.0)
        b.decide()
        with self.assertRaises(RateLimitExceeded):
            b.acquire_or_raise()
        d = b.decide()
        self.assertIn("Retry-After", d.to_headers())


class TestSelfHeal(unittest.TestCase):
    def test_probe_tuple_compat(self):
        def evil(text, chat_key="", message=None):
            raise RuntimeError("boom")

        broken, exc, err, loc = heal_probe(evil, "x")
        self.assertTrue(broken and exc is not None)

    def test_timeout_is_a_finding(self):
        import nomorals.core.self_heal as sh

        def hang(text, chat_key="", message=None):
            time.sleep(30)

        old = sh._PROBE_TIMEOUT
        sh._PROBE_TIMEOUT = 0.5
        try:
            res = heal_one(hang, "slow")
        finally:
            sh._PROBE_TIMEOUT = old
        self.assertTrue(res.broken and res.timed_out)
        self.assertFalse(res.fixable)

    def test_registry_has_builtin_strategies(self):
        kinds = [s.kind for s in FIX_STRATEGIES]
        self.assertIn("unbound_local", kinds)
        self.assertIn("dead_method", kinds)
        self.assertIn("missing_package", kinds)

    def test_custom_strategy(self):
        register_fix_strategy(FixStrategy(
            "test_report", lambda e: isinstance(e, KeyError), None,
            report_reason="test says no"))
        try:
            def kb(text, chat_key="", message=None):
                raise KeyError("k")

            res = heal_one(kb, "k")
            self.assertEqual(res.skip_reason, "test says no")
        finally:
            FIX_STRATEGIES.pop()

    def test_learning_loop(self):
        tmp = tempfile.mkdtemp()
        p = os.path.join(tmp, "hm.py")
        # downstream tolerates None (f-string), so the auto-fix verifies
        with open(p, "w") as f:
            f.write(textwrap.dedent(
                "def handler(kind):\n"
                "    if kind == 'a':\n"
                "        chat = 'x'\n"
                "    if kind == 'b':\n"
                "        return f'b:{chat}'\n"))
        ei = ErrorIntelligence(store_path=_kb())

        def fake(text, chat_key="", message=None):
            import importlib.util

            spec = importlib.util.spec_from_file_location("healmod_t", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.handler("b")

        res = heal_one(fake, "b", dry_run=False, intelligence=ei)
        self.assertTrue(res.fixed)
        learned = ei.learning.suggest_fix(res.fingerprint)
        self.assertIsNotNone(learned)


if __name__ == "__main__":
    unittest.main()
