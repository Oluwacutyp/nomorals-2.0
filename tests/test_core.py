"""Tests for the L1 kernel."""

from __future__ import annotations

import logging
import threading
import time
import unittest

from nomorals.core.clock import FrozenClock, ScaledClock
from nomorals.core.config import Settings, load_settings
from nomorals.core.errors import (
    CapabilityDenied,
    ConfigError,
    NoMoralsError,
    ProviderUnavailable,
    ValidationError,
    classify,
)
from nomorals.core.events import Event, EventBus
from nomorals.core.ids import ULID, decode_time, ulid_batch, ulid_now, ulid_range
from nomorals.core.logging_setup import LogCapture, redact, setup_logging
from nomorals.core.observability import Metrics, Tracer
from nomorals.core.policy import Capability, CapabilitySet, Policy
from nomorals.core.ratelimit import SemaphorePool, SlidingWindowLimiter, TokenBucket
from nomorals.core.result import Err, Ok, partition, unwrap_all
from nomorals.core.retry import (
    BackoffPolicy,
    CircuitBreaker,
    CircuitOpen,
    retry,
    retry_call,
)
from nomorals.core import text as textmod


# ── result ─────────────────────────────────────────────────────────────────────


class TestResult(unittest.TestCase):
    def test_ok_maps_and_chains(self) -> None:
        self.assertEqual(Ok(3).map(lambda x: x * 2).value, 6)
        self.assertEqual(Ok(3).and_then(lambda x: Ok(x + 1)).value, 4)
        self.assertTrue(Ok(0))

    def test_err_short_circuits(self) -> None:
        result = Err("bad").map(lambda x: x * 2)
        self.assertFalse(result.ok)
        self.assertIsNone(result.value)
        self.assertEqual(result.error.message, "bad")

    def test_err_from_exception_is_classified(self) -> None:
        self.assertEqual(Err(ValueError("x")).error.code, ValidationError.code)
        self.assertEqual(Err(ProviderUnavailable("y")).error.code, "model.provider.unavailable")

    def test_or_else_recovers(self) -> None:
        recovered = Err("x").or_else(lambda e: Ok(42))
        self.assertEqual(recovered.value, 42)
        self.assertEqual(Ok(1).or_else(lambda e: Ok(9)).value, 1)

    def test_unwrap_semantics(self) -> None:
        self.assertEqual(Ok(7).unwrap(), 7)
        self.assertEqual(Err("e").unwrap_or(3), 3)
        self.assertEqual(Err("e").unwrap_or_else(lambda e: len(e.message)), 1)
        with self.assertRaises(NoMoralsError):
            Err("boom").unwrap()

    def test_map_captures_handler_exceptions(self) -> None:
        result = Ok(1).map(lambda x: 1 / 0)
        self.assertFalse(result.ok)
        self.assertIsInstance(result.error, NoMoralsError)

    def test_unwrap_all_and_partition(self) -> None:
        self.assertEqual(unwrap_all([Ok(1), Ok(2)]).value, [1, 2])
        self.assertFalse(unwrap_all([Ok(1), Err("z")]).ok)
        oks, errs = partition([Ok(1), Err("e"), Ok(3)])
        self.assertEqual(oks, [1, 3])
        self.assertEqual(len(errs), 1)

    def test_classify_is_idempotent(self) -> None:
        original = ProviderUnavailable("x")
        self.assertIs(classify(original), original)
        self.assertTrue(classify(TimeoutError("t")).retryable)
        self.assertEqual(classify(FileNotFoundError("f")).code, "storage.not_found")


# ── ids ────────────────────────────────────────────────────────────────────────


class TestIds(unittest.TestCase):
    def test_ulids_are_monotonic_and_unique(self) -> None:
        ids = list(ulid_batch(5000))
        self.assertEqual(ids, sorted(ids), "ULIDs must sort in generation order")
        self.assertEqual(len(set(ids)), 5000, "ULIDs must be unique")

    def test_ulid_length_and_alphabet(self) -> None:
        value = ulid_now()
        self.assertEqual(len(value), 26)
        self.assertTrue(all(c in "0123456789ABCDEFGHJKMNPQRSTVWXYZ" for c in value))

    def test_ulid_encodes_time(self) -> None:
        value = ulid_now()
        self.assertLess(abs(decode_time(value) - time.time()), 2.0)

    def test_ulid_range_bounds(self) -> None:
        low, high = ulid_range(0, 10**12)
        self.assertLess(low, high)
        self.assertTrue(ulid_now() >= low)

    def test_ulid_object(self) -> None:
        raw = ulid_now()
        obj = ULID(raw)
        self.assertEqual(str(obj), raw)
        self.assertGreater(obj.timestamp, 0)
        self.assertEqual(ULID(raw), ULID(raw))
        with self.assertRaises(ValueError):
            ULID("short")

    def test_concurrent_generation_stays_unique(self) -> None:
        seen: list[list[str]] = [[] for _ in range(8)]

        def worker(index: int) -> None:
            seen[index] = list(ulid_batch(250))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        flat = [v for chunk in seen for v in chunk]
        self.assertEqual(len(flat), 2000)
        self.assertEqual(len(set(flat)), 2000)


# ── clock ──────────────────────────────────────────────────────────────────────


class TestClock(unittest.TestCase):
    def test_frozen_clock_advances_explicitly(self) -> None:
        clock = FrozenClock(1000.0)
        self.assertEqual(clock.now(), 1000.0)
        self.assertEqual(clock.advance(60), 1060.0)
        self.assertEqual(clock.now_ms(), 1_060_000)

    def test_scaled_clock_runs_fast(self) -> None:
        clock = ScaledClock(factor=1000.0)
        start = clock.monotonic()
        time.sleep(0.01)
        self.assertGreater(clock.monotonic() - start, 1.0)

    def test_iso_format(self) -> None:
        stamp = FrozenClock(0.0).iso()
        self.assertEqual(stamp, "1970-01-01T00:00:00Z")


# ── config ─────────────────────────────────────────────────────────────────────


class TestConfig(unittest.TestCase):
    def test_defaults(self) -> None:
        settings = load_settings(env={}, use_env_file=False)
        self.assertEqual(settings.profile, "workstation")
        self.assertEqual(settings.concurrency.threads, 16)
        self.assertIsInstance(settings.storage, type(settings.storage))

    def test_profile_preset_applies(self) -> None:
        settings = load_settings(env={"NM_PROFILE": "termux"}, use_env_file=False)
        self.assertFalse(settings.concurrency.use_processes)
        self.assertEqual(settings.concurrency.threads, 4)
        self.assertEqual(settings.training.backend, "native")

    def test_env_overrides_with_coercion(self) -> None:
        settings = load_settings(
            env={"NM_THREADS": "32", "NM_API_PORT": "9000", "NM_DB_WAL": "false"},
            use_env_file=False,
        )
        self.assertEqual(settings.concurrency.threads, 32)
        self.assertEqual(settings.api.port, 9000)
        self.assertIs(settings.storage.wal, False)

    def test_nested_section_is_a_dataclass_not_a_dict(self) -> None:
        settings = load_settings(env={}, use_env_file=False)
        self.assertEqual(settings.storage.synchronous, "NORMAL")
        self.assertAlmostEqual(settings.budget.wall_seconds, 3600.0)

    def test_bare_field_name_is_located(self) -> None:
        settings = load_settings(env={"NM_LEARNING_RATE": "1e-4"}, use_env_file=False)
        self.assertAlmostEqual(settings.training.learning_rate, 1e-4)

    def test_secrets_are_redacted(self) -> None:
        settings = load_settings(env={"HF_TOKEN": "hf_supersecretvalue123"}, use_env_file=False)
        self.assertEqual(settings.llm.hf_token, "hf_supersecretvalue123")
        self.assertNotEqual(settings.to_dict()["llm"]["hf_token"], "hf_supersecretvalue123")
        self.assertEqual(
            settings.to_dict(redact=False)["llm"]["hf_token"], "hf_supersecretvalue123"
        )

    def test_dotted_get(self) -> None:
        settings = load_settings(env={}, use_env_file=False)
        self.assertEqual(settings.get("llm.provider"), "mock")
        self.assertEqual(settings.get("does.not.exist", "fallback"), "fallback")

    def test_unknown_profile_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            load_settings(env={"NM_PROFILE": "bogus"}, use_env_file=False)

    def test_unknown_key_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            load_settings(overrides={"storage.bogus": 1}, use_env_file=False)

    def test_invalid_value_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            load_settings(overrides={"storage.synchronous": "SOMETIMES"}, use_env_file=False)

    def test_env_file_parsing(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                '# comment\nexport NM_THREADS="6"\nNM_PROFILE=laptop  # trailing\n\nBADLINE\n',
                encoding="utf-8",
            )
            import os

            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                settings = load_settings(env={}, use_env_file=True)
            finally:
                os.chdir(cwd)
        self.assertEqual(settings.concurrency.threads, 6)
        self.assertEqual(settings.profile, "laptop")

    def test_toml_file_loading(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.toml"
            cfg.write_text('[memory]\nrecall_limit = 99\n', encoding="utf-8")
            settings = load_settings(config_file=cfg, env={}, use_env_file=False)
        self.assertEqual(settings.memory.recall_limit, 99)
        self.assertEqual(settings.config_file, str(cfg))

    def test_settings_replace_is_pure(self) -> None:
        settings = load_settings(env={}, use_env_file=False)
        other = settings.replace(profile="laptop")
        self.assertEqual(settings.profile, "workstation")
        self.assertEqual(other.profile, "laptop")

    def test_defaults_construct_directly(self) -> None:
        self.assertEqual(Settings().embedding.dimensions, 512)


# ── events ─────────────────────────────────────────────────────────────────────


class TestEvents(unittest.TestCase):
    def test_glob_and_wildcard_matching(self) -> None:
        bus = EventBus()
        seen: list[str] = []
        bus.subscribe("task.*", lambda e: seen.append(e.topic), sync=True)
        bus.publish(Event(topic="task.started"))
        bus.publish("task.finished", {"x": 1})
        bus.publish("other.thing")
        self.assertEqual(seen, ["task.started", "task.finished"])

    def test_exact_and_star_patterns(self) -> None:
        bus = EventBus()
        exact: list[str] = []
        everything: list[str] = []
        bus.subscribe("a.b", lambda e: exact.append(e.topic), sync=True)
        bus.subscribe("*", lambda e: everything.append(e.topic), sync=True)
        bus.publish("a.b")
        bus.publish("a.c")
        self.assertEqual(exact, ["a.b"])
        self.assertEqual(everything, ["a.b", "a.c"])

    def test_async_dispatch_and_wait_for(self) -> None:
        bus = EventBus().start()
        try:
            threading.Timer(0.05, lambda: bus.emit("ping", v=1)).start()
            event = bus.wait_for("ping", timeout=3.0)
            self.assertIsNotNone(event)
            self.assertEqual(event.data["v"], 1)
        finally:
            bus.stop()

    def test_once_subscription_fires_once(self) -> None:
        bus = EventBus()
        hits: list[int] = []
        bus.subscribe("x", lambda e: hits.append(1), sync=True, once=True)
        bus.publish("x")
        bus.publish("x")
        self.assertEqual(len(hits), 1)

    def test_unsubscribe(self) -> None:
        bus = EventBus()
        hits: list[int] = []
        sub_id = bus.subscribe("x", lambda e: hits.append(1), sync=True)
        bus.publish("x")
        self.assertTrue(bus.unsubscribe(sub_id))
        bus.publish("x")
        self.assertEqual(len(hits), 1)

    def test_handler_errors_do_not_propagate(self) -> None:
        bus = EventBus()
        bus.subscribe("boom", lambda e: 1 / 0, sync=True)
        with LogCapture("nomorals.core.events"):
            event = bus.publish("boom")
        self.assertTrue(event.errors)

    def test_history_and_stats(self) -> None:
        bus = EventBus(history=4)
        for i in range(10):
            bus.emit("tick", i=i)
        self.assertEqual(len(bus.history()), 4)
        self.assertEqual(bus.stats()["published"], 10)

    def test_priority_ordering(self) -> None:
        bus = EventBus()
        order: list[str] = []
        bus.subscribe("p", lambda e: order.append("low"), sync=True, priority=0)
        bus.subscribe("p", lambda e: order.append("high"), sync=True, priority=10)
        bus.publish("p")
        self.assertEqual(order, ["high", "low"])

    def test_event_payload_helpers(self) -> None:
        event = Event(topic="t", data={"a": 1}, source="test")
        self.assertEqual(event.get("a"), 1)
        self.assertIsNone(event.get("missing"))
        self.assertEqual(event.to_dict()["topic"], "t")
        self.assertTrue(event.event_id.startswith("evt_"))


# ── retry ──────────────────────────────────────────────────────────────────────


class TestRetry(unittest.TestCase):
    def test_retries_until_success(self) -> None:
        calls = {"n": 0}

        def flaky() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise ProviderUnavailable("x")
            return "ok"

        self.assertEqual(
            retry_call(flaky, policy=BackoffPolicy(base=0.0001, max_attempts=5)), "ok"
        )
        self.assertEqual(calls["n"], 3)

    def test_non_retryable_raises_immediately(self) -> None:
        calls = {"n": 0}

        def hard() -> None:
            calls["n"] += 1
            raise NoMoralsError("nope")

        with self.assertRaises(NoMoralsError):
            retry_call(hard, policy=BackoffPolicy(base=0.0001, max_attempts=5))
        self.assertEqual(calls["n"], 1)

    def test_exhausted_attempts_raise_last_error(self) -> None:
        with self.assertRaises(ProviderUnavailable):
            retry_call(
                lambda: (_ for _ in ()).throw(ProviderUnavailable("still down")),
                policy=BackoffPolicy(base=0.0001, max_attempts=3),
            )

    def test_backoff_schedule_is_exponential_and_capped(self) -> None:
        policy = BackoffPolicy(base=1.0, factor=2.0, jitter=False, cap=8.0)
        self.assertEqual(policy.delay(1), 1.0)
        self.assertEqual(policy.delay(2), 2.0)
        self.assertEqual(policy.delay(3), 4.0)
        self.assertEqual(policy.delay(10), 8.0)

    def test_retry_after_is_respected(self) -> None:
        policy = BackoffPolicy(base=1.0, jitter=False)
        self.assertEqual(policy.delay(1, retry_after=5.0), 5.0)
        self.assertEqual(policy.delay(1, retry_after=999.0), 60.0)

    def test_decorator_records_stats(self) -> None:
        @retry(BackoffPolicy(base=0.0001, max_attempts=3))
        def flaky() -> str:
            raise ProviderUnavailable("x")

        with self.assertRaises(ProviderUnavailable):
            flaky()
        stats = flaky.stats.as_dict()  # type: ignore[attr-defined]
        self.assertEqual(stats["calls"], 1, "one logical call regardless of attempts")
        self.assertEqual(stats["attempts"], 3)
        self.assertEqual(stats["failures"], 3)
        self.assertEqual(stats["successes"], 0)
        self.assertEqual(stats["avg_attempts"], 3.0)
        self.assertEqual(stats["success_rate"], 0.0)

    def test_on_retry_callback(self) -> None:
        seen: list[int] = []
        calls = {"n": 0}

        def flaky() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise ProviderUnavailable("x")
            return "ok"

        retry_call(
            flaky,
            policy=BackoffPolicy(base=0.0001, max_attempts=5),
            on_retry=lambda attempt, exc, delay: seen.append(attempt),
        )
        self.assertEqual(seen, [1, 2])

    def test_circuit_breaker_opens_and_recovers(self) -> None:
        clock = {"t": 0.0}
        breaker = CircuitBreaker(
            "x", failure_threshold=2, reset_timeout=10.0, clock=lambda: clock["t"]
        )
        breaker.record_failure()
        self.assertEqual(breaker.state, CircuitBreaker.CLOSED)
        breaker.record_failure()
        with self.assertRaises(CircuitOpen):
            breaker.before_call()
        clock["t"] = 11.0
        self.assertEqual(breaker.state, CircuitBreaker.HALF_OPEN)
        breaker.before_call()
        breaker.record_success()
        self.assertEqual(breaker.state, CircuitBreaker.CLOSED)

    def test_half_open_failure_reopens(self) -> None:
        clock = {"t": 0.0}
        breaker = CircuitBreaker(
            "y", failure_threshold=1, reset_timeout=5.0, clock=lambda: clock["t"]
        )
        breaker.record_failure()
        clock["t"] = 6.0
        self.assertEqual(breaker.state, CircuitBreaker.HALF_OPEN)
        breaker.record_failure()
        self.assertEqual(breaker.state, CircuitBreaker.OPEN)


# ── rate limiting ──────────────────────────────────────────────────────────────


class TestRateLimit(unittest.TestCase):
    def test_token_bucket_burst_then_refill(self) -> None:
        bucket = TokenBucket(rate=1000.0, capacity=3)
        self.assertEqual([bucket.try_acquire() for _ in range(4)], [True, True, True, False])
        self.assertTrue(bucket.acquire(timeout=1.0))

    def test_token_bucket_throttle_returns_wait(self) -> None:
        bucket = TokenBucket(rate=50.0, capacity=1)
        bucket.try_acquire()
        self.assertGreater(bucket.throttle(), 0.0)

    def test_wait_time_reports_zero_when_available(self) -> None:
        bucket = TokenBucket(rate=1.0, capacity=5)
        self.assertEqual(bucket.wait_time(), 0.0)

    def test_invalid_bucket_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TokenBucket(rate=0.0, capacity=1)
        with self.assertRaises(ValueError):
            TokenBucket(rate=1.0, capacity=0)

    def test_sliding_window_forbids_burst(self) -> None:
        window = SlidingWindowLimiter(2, 60.0)
        self.assertTrue(window.try_acquire())
        self.assertTrue(window.try_acquire())
        self.assertFalse(window.try_acquire())
        self.assertEqual(window.remaining, 0)
        self.assertGreater(window.wait_time(), 0.0)

    def test_semaphore_pool_bounds_concurrency(self) -> None:
        pool = SemaphorePool({"net": 2})
        peak = {"now": 0, "max": 0}
        completed = {"n": 0}
        lock = threading.Lock()

        def worker() -> None:
            with pool.slot("net"):
                with lock:
                    peak["now"] += 1
                    peak["max"] = max(peak["max"], peak["now"])
                time.sleep(0.02)
                with lock:
                    peak["now"] -= 1
                    completed["n"] += 1

        threads = [threading.Thread(target=worker) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)
        self.assertEqual(completed["n"], 12, "every worker must eventually run")
        self.assertLessEqual(peak["max"], 2, f"permit bound exceeded: {peak}")
        self.assertEqual(pool.snapshot()["net"]["in_use"], 0)

    def test_slot_raises_rather_than_running_unguarded(self) -> None:
        """The body must never execute when no permit was obtained."""
        pool = SemaphorePool({"one": 1})
        ran = {"body": False}
        pool.acquire("one")
        try:
            with self.assertRaises(TimeoutError):
                with pool.slot("one", timeout=0.05):
                    ran["body"] = True
        finally:
            pool.release("one")
        self.assertFalse(ran["body"], "body ran without holding a permit")

    def test_try_slot_reports_availability(self) -> None:
        pool = SemaphorePool({"one": 1})
        with pool.try_slot("one") as first:
            self.assertTrue(first)
            with pool.try_slot("one") as second:
                self.assertFalse(second)
        self.assertEqual(pool.snapshot()["one"]["in_use"], 0)

    def test_semaphore_acquire_timeout(self) -> None:
        pool = SemaphorePool({"one": 1})
        self.assertTrue(pool.acquire("one"))
        self.assertFalse(pool.acquire("one", timeout=0.05))
        pool.release("one")
        self.assertTrue(pool.acquire("one", timeout=0.05))
        pool.release("one")

    def test_unknown_resource_is_auto_registered(self) -> None:
        pool = SemaphorePool()
        with pool.slot("mystery"):
            self.assertEqual(pool.snapshot()["mystery"]["in_use"], 1)
        self.assertIn("mystery", pool.snapshot())
        self.assertEqual(pool.snapshot()["mystery"]["in_use"], 0)


# ── policy ─────────────────────────────────────────────────────────────────────


class TestPolicy(unittest.TestCase):
    def test_role_preset_grants(self) -> None:
        grant = CapabilitySet.role("coding")
        self.assertTrue(grant.grants(Capability.FS_WRITE))
        self.assertFalse(grant.grants(Capability.SOCIAL_POST))

    def test_wildcard_grants(self) -> None:
        self.assertTrue(CapabilitySet.all().grants(Capability.DB_ADMIN))
        self.assertTrue(CapabilitySet.of("fs.*").grants(Capability.FS_DELETE))
        self.assertFalse(CapabilitySet.none().grants(Capability.FS_READ))

    def test_intersection_narrows_privilege(self) -> None:
        narrowed = CapabilitySet.role("orchestrator").intersect(CapabilitySet.role("readonly"))
        self.assertTrue(narrowed.grants(Capability.FS_READ))
        self.assertFalse(narrowed.grants(Capability.FS_WRITE))

    def test_intersection_with_wildcard(self) -> None:
        readonly = CapabilitySet.role("readonly")
        self.assertEqual(
            CapabilitySet.all().intersect(readonly).as_list(), readonly.as_list()
        )

    def test_denied_without_grant(self) -> None:
        policy = Policy(default_grant=CapabilitySet.role("coding"))
        decision = policy.check(Capability.SOCIAL_POST, actor="a")
        self.assertFalse(decision.allowed)
        self.assertIn("lacks capability", decision.reason)

    def test_deny_rule_beats_grant(self) -> None:
        policy = Policy(default_grant=CapabilitySet.all())
        policy.deny("exec.shell", note="no shell here")
        decision = policy.check(Capability.EXEC_SHELL, actor="a")
        self.assertFalse(decision.allowed)
        self.assertIn("no shell here", decision.reason)

    def test_require_raises(self) -> None:
        policy = Policy(default_grant=CapabilitySet.none())
        with self.assertRaises(CapabilityDenied):
            policy.require(Capability.FS_WRITE, actor="a")

    def test_confirmable_needs_single_use_bound_token(self) -> None:
        policy = Policy(default_grant=CapabilitySet.of(Capability.FS_DELETE))
        self.assertFalse(policy.check(Capability.FS_DELETE, actor="a").allowed)

        token = policy.issue_confirmation(Capability.FS_DELETE)
        self.assertTrue(
            policy.check(Capability.FS_DELETE, actor="a", confirmation=token).allowed
        )
        self.assertFalse(
            policy.check(Capability.FS_DELETE, actor="a", confirmation=token).allowed,
            "confirmation tokens must be single-use",
        )

    def test_confirmation_is_capability_bound(self) -> None:
        policy = Policy(default_grant=CapabilitySet.of(Capability.FS_DELETE, Capability.DB_ADMIN))
        token = policy.issue_confirmation(Capability.FS_DELETE)
        decision = policy.check(Capability.DB_ADMIN, actor="a", confirmation=token)
        self.assertFalse(decision.allowed, "token minted for fs.delete must not unlock db.admin")

    def test_confirmation_expires(self) -> None:
        clock = FrozenClock(1000.0)
        policy = Policy(
            default_grant=CapabilitySet.of(Capability.FS_DELETE),
            confirmation_ttl=10.0,
            clock=clock,
        )
        token = policy.issue_confirmation(Capability.FS_DELETE)
        clock.advance(20.0)
        self.assertFalse(
            policy.check(Capability.FS_DELETE, actor="a", confirmation=token).allowed
        )

    def test_enforce_disabled_allows_all(self) -> None:
        policy = Policy(default_grant=CapabilitySet.none(), enforce=False)
        self.assertTrue(policy.check(Capability.SYS_SHUTDOWN, actor="a").allowed)

    def test_audit_log_records_decisions(self) -> None:
        policy = Policy(default_grant=CapabilitySet.of(Capability.FS_READ))
        policy.check(Capability.FS_READ, actor="a")
        policy.check(Capability.FS_WRITE, actor="a")
        log = policy.audit_log()
        self.assertEqual(len(log), 2)
        self.assertEqual({entry["kind"] for entry in log}, {"allow", "deny"})
        self.assertTrue(all(entry["audit_id"] for entry in log))

    def test_unknown_role_rejected(self) -> None:
        with self.assertRaises(KeyError):
            CapabilitySet.role("wizard")

    def test_minus_removes_capabilities(self) -> None:
        grant = CapabilitySet.of("fs.*").minus(Capability.FS_DELETE)
        self.assertTrue(grant.grants(Capability.FS_READ))
        self.assertFalse(grant.grants(Capability.FS_DELETE))


# ── text ───────────────────────────────────────────────────────────────────────


class TestText(unittest.TestCase):
    def test_normalize(self) -> None:
        self.assertEqual(textmod.normalize_text("a   b\r\nc"), "a b\nc")
        self.assertEqual(textmod.normalize_text("  x  "), "x")

    def test_approx_token_count(self) -> None:
        self.assertEqual(textmod.approx_token_count(""), 0)
        self.assertGreater(textmod.approx_token_count("hello world"), 0)
        self.assertGreater(
            textmod.approx_token_count("antidisestablishmentarianism " * 5),
            textmod.approx_token_count("hi " * 5),
        )

    def test_truncate_to_tokens(self) -> None:
        long_text = "word " * 500
        trimmed = textmod.truncate_to_tokens(long_text, 10)
        self.assertLessEqual(textmod.approx_token_count(trimmed), 11)
        self.assertTrue(trimmed.endswith("…"))
        self.assertEqual(textmod.truncate_to_tokens("short", 100), "short")
        self.assertEqual(textmod.truncate_to_tokens("anything", 0), "")

    def test_chunking_respects_boundaries(self) -> None:
        text = "This is a sentence about a thing. " * 200
        chunks = textmod.chunk_text(text, max_tokens=64, overlap_tokens=8, source="doc")
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(c.tokens <= 96 for c in chunks), [c.tokens for c in chunks])
        self.assertEqual(chunks[0].source, "doc")
        self.assertEqual([c.index for c in chunks], list(range(len(chunks))))

    def test_chunking_empty_and_overlap_validation(self) -> None:
        self.assertEqual(textmod.chunk_text("   "), [])
        with self.assertRaises(ValueError):
            textmod.chunk_text("text", max_tokens=10, overlap_tokens=10)

    def test_chunking_oversized_sentence_hard_splits(self) -> None:
        text = " ".join(f"word{i}" for i in range(600))
        chunks = textmod.chunk_text(text, max_tokens=50, overlap_tokens=5)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(c.meta.get("hard_split") for c in chunks))

    def test_sentences_handle_abbreviations(self) -> None:
        self.assertEqual(
            textmod.sentences("Dr. Smith went home. He slept."),
            ["Dr. Smith went home.", "He slept."],
        )

    def test_simhash_distance_tracks_fraction_of_changed_features(self) -> None:
        # SimHash Hamming distance is proportional to the *fraction* of features
        # that differ. One word changed in a 200-word document is ~1.5% of trigram
        # features, so the distance must be small; a wholly different document
        # shares no features, so it lands near 32 of 64 bits.
        words = [f"token{i}" for i in range(200)]
        base = " ".join(words)
        variant = " ".join(
            "changed" if i == 100 else w for i, w in enumerate(words)
        )
        unrelated = " ".join(f"other{i}" for i in range(200))

        base_hash = textmod.SimHash.from_text(base)
        self.assertLessEqual(base_hash.hamming(textmod.SimHash.from_text(variant)), 6)
        self.assertGreater(base_hash.hamming(textmod.SimHash.from_text(unrelated)), 15)

    def test_simhash_similarity_bounds(self) -> None:
        a = textmod.SimHash.from_text("alpha beta gamma delta epsilon zeta")
        self.assertEqual(a.similarity(a), 1.0)
        self.assertEqual(textmod.SimHash(0).hamming(textmod.SimHash(0)), 0)

    def test_simhash_empty(self) -> None:
        self.assertEqual(int(textmod.SimHash.from_text("")), 0)

    def test_dedupe_drops_copies_keeps_distinct(self) -> None:
        docs = [
            "the quick brown fox jumps over the lazy dog " * 8,
            "the quick brown fox jumps over the lazy dog " * 8,
            "quantum chromodynamics explains the strong nuclear interaction " * 8,
        ]
        result = textmod.dedupe_by_simhash(docs)
        self.assertEqual(result.dropped, [1])
        self.assertEqual(result.kept, [0, 2])
        self.assertGreater(result.dropped_ratio, 0)

    def test_dedupe_min_length(self) -> None:
        result = textmod.dedupe_by_simhash(["tiny", "another reasonably sized document here"], min_length=20)
        self.assertEqual(result.kept, [1])

    def test_levenshtein_and_similarity(self) -> None:
        self.assertEqual(textmod.levenshtein("kitten", "sitting"), 3)
        self.assertEqual(textmod.levenshtein("", "abc"), 3)
        self.assertEqual(textmod.levenshtein("same", "same"), 0)
        self.assertEqual(textmod.similarity("abc", "abc"), 1.0)
        self.assertLess(textmod.similarity("abc", "xyz"), 0.5)

    def test_shingle_jaccard(self) -> None:
        a = textmod.shingle("one two three four five six seven")
        b = textmod.shingle("one two three four five six seven")
        self.assertEqual(textmod.jaccard(a, b), 1.0)
        self.assertEqual(textmod.jaccard(set(), set()), 1.0)
        self.assertEqual(textmod.jaccard({"a"}, set()), 0.0)

    def test_word_frequencies_drops_stopwords(self) -> None:
        counts = textmod.word_frequencies("the cat and the dog and the bird")
        self.assertNotIn("the", counts)
        self.assertIn("cat", counts)

    def test_summarize_is_extractive(self) -> None:
        text = (
            "The reactor reached criticality at noon. "
            "Coolant flow remained nominal throughout the test. "
            "Engineers recorded a three percent efficiency gain. "
            "The next trial is scheduled for March. "
            "Weather delayed the original timeline by a week."
        )
        summary = textmod.summarize(text, max_sentences=2)
        self.assertEqual(len(textmod.sentences(summary)), 2)

    def test_cosine_counts(self) -> None:
        from collections import Counter

        self.assertAlmostEqual(
            textmod.cosine_counts(Counter("a b".split()), Counter("a b".split())), 1.0
        )
        self.assertEqual(textmod.cosine_counts(Counter(), Counter("a")), 0.0)

    def test_byte_tokenizer_roundtrips_multibyte_text(self) -> None:
        for text in ("héllo wörld", "日本語のテキスト", "plain ascii", "emoji 🎉 inside"):
            with self.subTest(text=text):
                pieces = textmod.tokenize_bpe_bytes(text, chunk=3)
                self.assertTrue(all(isinstance(p, bytes) for p in pieces))
                self.assertEqual(textmod.detokenize_bytes(pieces), text)

    def test_byte_tokenizer_validates_chunk(self) -> None:
        with self.assertRaises(ValueError):
            textmod.tokenize_bpe_bytes("x", chunk=0)

    def test_ngrams_validation(self) -> None:
        self.assertEqual(list(textmod.ngrams(["a", "b", "c"], 2)), [("a", "b"), ("b", "c")])
        with self.assertRaises(ValueError):
            list(textmod.ngrams(["a"], 0))


# ── observability ──────────────────────────────────────────────────────────────


class TestObservability(unittest.TestCase):
    def test_counters_gauges_histograms(self) -> None:
        metrics = Metrics()
        metrics.incr("requests")
        metrics.incr("requests", 2)
        metrics.set_gauge("agents", 3)
        for value in (0.01, 0.2, 0.9):
            metrics.observe("latency", value)
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["counters"]["requests"], 3.0)
        self.assertEqual(snapshot["gauges"]["agents"], 3)
        self.assertEqual(snapshot["histograms"]["latency"]["count"], 3)
        self.assertGreater(snapshot["histograms"]["latency"]["p95"], 0)

    def test_timer_records_observation(self) -> None:
        metrics = Metrics()
        with metrics.timer("op"):
            time.sleep(0.005)
        self.assertEqual(metrics.get("op.count"), 1.0)
        self.assertGreater(metrics.snapshot()["histograms"]["op"]["avg"], 0)

    def test_prometheus_exposition(self) -> None:
        metrics = Metrics()
        metrics.incr("requests")
        metrics.set_gauge("up", 1)
        metrics.observe("latency", 0.5)
        body = metrics.as_prometheus()
        self.assertIn("# TYPE nomorals_requests counter", body)
        self.assertIn("nomorals_requests 1.0", body)
        self.assertIn('# TYPE nomorals_up gauge', body)
        self.assertIn('nomorals_latency_seconds_bucket{le="+Inf"} 1', body)

    def test_labelled_counters(self) -> None:
        metrics = Metrics()
        metrics.incr("calls", 1, provider="mock")
        metrics.incr("calls", 2, provider="hf")
        self.assertEqual(metrics.snapshot()["labels"]["calls"]["provider=mock"], 1.0)

    def test_nested_spans(self) -> None:
        tracer = Tracer()
        with tracer.span("outer", k=1):
            with tracer.span("inner"):
                pass
        traces = tracer.traces()
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]["name"], "outer")
        self.assertEqual(traces[0]["attributes"], {"k": 1})
        self.assertEqual(traces[0]["children"][0]["name"], "inner")
        self.assertEqual(traces[0]["children"][0]["parent_id"], traces[0]["span_id"])

    def test_span_records_error_and_reraises(self) -> None:
        tracer = Tracer()
        with self.assertRaises(ValueError):
            with tracer.span("failing"):
                raise ValueError("bad")
        self.assertEqual(tracer.traces()[0]["status"], "error")
        self.assertIn("ValueError", tracer.traces()[0]["error"])

    def test_reset_clears(self) -> None:
        metrics = Metrics()
        metrics.incr("x")
        metrics.reset()
        self.assertEqual(metrics.snapshot()["counters"], {})


# ── logging ────────────────────────────────────────────────────────────────────


class TestLogging(unittest.TestCase):
    def test_redaction_patterns(self) -> None:
        cases = {
            "key=hf_abcdefghijkl123456": "[REDACTED_KEY]",
            "token ghp_abcdefghijklmnop1234": "[REDACTED_KEY]",
            "api_key: supersecret123": "api_key=[REDACTED]",
            "password=hunter22secret": "password=[REDACTED]",
            "app_password: abc-defg-hijk": "app_password=[REDACTED]",
            "AKIAIOSFODNN7EXAMPLE": "[REDACTED_AWS]",
            "Authorization: Bearer abcdefghijklmnop": "[REDACTED]",
            "used bearer abcdefghijklmnop in call": "[REDACTED_AUTH]",
        }
        for source, expected_fragment in cases.items():
            with self.subTest(source=source):
                self.assertIn(expected_fragment, redact(source))

    def test_redaction_never_leaks_the_secret_it_matched(self) -> None:
        """Regression: the key=value rule used to eat the word "Bearer" and leave
        the token itself in the log."""
        secrets = [
            "abcdefghijklmnop",
            "supersecret123",
            "hunter22secret",
            "hf_abcdefghijkl123456",
        ]
        sources = [
            "Authorization: Bearer abcdefghijklmnop",
            "authorization=abcdefghijklmnop",
            "api_key: supersecret123",
            "password=hunter22secret",
            "Cookie: session=abcdefghijklmnop",
            "using hf_abcdefghijkl123456 for hub access",
        ]
        for source in sources:
            scrubbed = redact(source)
            for secret in secrets:
                if secret in source:
                    self.assertNotIn(secret, scrubbed, f"leaked in {scrubbed!r}")

    def test_redaction_leaves_clean_text_alone(self) -> None:
        self.assertEqual(redact("nothing secret here"), "nothing secret here")
        self.assertEqual(redact(""), "")

    def test_log_filter_redacts_records(self) -> None:
        setup_logging("DEBUG", file=None, force=True)
        logger = logging.getLogger("nomorals.test.redaction")
        with LogCapture("nomorals.test.redaction") as capture:
            logger.info("using token hf_abcdefghij1234567890 now")
        self.assertIn("[REDACTED_KEY]", capture.messages[0])
        self.assertNotIn("hf_abcdefghij1234567890", capture.messages[0])

    def test_log_capture_find(self) -> None:
        logger = logging.getLogger("nomorals.test.find")
        with LogCapture("nomorals.test.find") as capture:
            logger.info("alpha message")
            logger.warning("beta message")
        self.assertEqual(capture.find("beta"), ["beta message"])


if __name__ == "__main__":
    unittest.main()
