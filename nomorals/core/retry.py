"""Retry with backoff, jitter, circuit breaking, and composed resilience pipelines.

Pieces:

* :class:`BackoffPolicy` — how long to wait between attempts (jitter styles
  include AWS-style full/equal/decorrelated jitter).
* :func:`retry` / :func:`retry_call` — apply the policy, honoring each
  error's ``retryable`` flag and ``retry_after`` hint (one error dialect —
  see :mod:`nomorals.core.errors` and :mod:`nomorals.core.ratelimit`).
* :class:`CircuitBreaker` — stops hammering a dependency that is clearly down.
* :class:`TimeoutPolicy` — every remote call gets a deadline; fail fast.
* :class:`FallbackPolicy` — a deliberate degraded answer beats an error.
* :class:`HedgingPolicy` — race a slow call against a delayed duplicate.
* :class:`RetryBudget` — cap retries as a fraction of total traffic so one
  caller's retries can't become a fleet-wide retry storm.
* :class:`ResiliencePipeline` — compose the above Polly-v8-style:
  ``fallback → breaker → retry → hedging → timeout → call``.

Retryable-ness is decided by the *error*, not by guesswork: every framework
error carries a ``retryable`` flag, and the default predicate consults it.
Transport-level transients (429/5xx/timeout) are retried *inside* the wrapper,
invisible to the caller; semantic failures are not retries — they are new
attempts that must differ.
"""

from __future__ import annotations

import functools
import logging
import queue
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Sequence, TypeVar

from .errors import NoMoralsError, ProviderUnavailable, RateLimited, TimeoutError_, classify

__all__ = [
    "BackoffPolicy",
    "CircuitBreaker",
    "CircuitOpen",
    "FallbackPolicy",
    "HedgingPolicy",
    "JitterStyle",
    "ResiliencePipeline",
    "RetryBudget",
    "RetryStats",
    "TimeoutPolicy",
    "retry",
    "retry_call",
]

_log = logging.getLogger(__name__)
T = TypeVar("T")


class CircuitOpen(NoMoralsError):
    """The breaker is open; calls are being rejected without hitting the dependency."""

    code = "circuit.open"
    retryable = True


class JitterStyle(Enum):
    """How randomness is applied to the backoff delay.

    * ``NONE`` — deterministic ``base * factor**(n-1)`` (tests only; never use
      in production — synchronized retries become a thundering herd).
    * ``FULL`` — ``uniform(0, capped)``. The AWS-recommended default: it
      decorrelates retries when many agents hit the same limit at once.
    * ``EQUAL`` — ``capped/2 + uniform(0, capped/2)``. Keeps half the backoff
      deterministic while still spreading the herd.
    * ``DECORRELATED`` — ``uniform(base, min(cap, prev*3))`` (Marc Brooker's
      algorithm, Polly's ``DecorrelatedJitterBackoffV2``). The production
      choice for fleet-wide retries: each delay depends on the previous one,
      so retries spread smoothly instead of clustering.
    """

    NONE = "none"
    FULL = "full"
    EQUAL = "equal"
    DECORRELATED = "decorrelated"


@dataclass
class BackoffPolicy:
    """Exponential backoff with configurable jitter.

    Delay for attempt ``n`` is derived from ``min(cap, base * factor**n)``
    and then jittered per :attr:`jitter_style`. ``retry_after`` hints from
    the error (HTTP ``Retry-After``, ``RateLimited.retry_after``) are honored
    first — the server told us when to come back; we listen.
    """

    base: float = 0.5
    factor: float = 2.0
    cap: float = 60.0
    max_attempts: int = 4
    jitter: bool = True
    jitter_style: JitterStyle = JitterStyle.FULL
    respect_retry_after: bool = True

    def delay(
        self,
        attempt: int,
        retry_after: float | None = None,
        prev_delay: float | None = None,
    ) -> float:
        if retry_after is not None and self.respect_retry_after:
            return max(0.0, min(float(retry_after), self.cap))
        raw = min(self.cap, self.base * (self.factor ** max(0, attempt - 1)))
        style = self.jitter_style if self.jitter else JitterStyle.NONE
        if style is JitterStyle.NONE:
            return raw
        if style is JitterStyle.EQUAL:
            return raw / 2.0 + random.uniform(0.0, raw / 2.0)
        if style is JitterStyle.DECORRELATED:
            prev = prev_delay if prev_delay is not None else self.base
            return random.uniform(self.base, min(self.cap, max(self.base, prev * 3.0)))
        return random.uniform(0.0, raw)  # FULL

    def schedule(self, attempts: int | None = None) -> list[float]:
        count = attempts or self.max_attempts
        prev: float | None = None
        out = []
        for i in range(1, count):
            d = self.delay(i, prev_delay=prev)
            out.append(d)
            prev = d
        return out


def default_retryable(exc: BaseException) -> bool:
    """Decide whether ``exc`` is worth retrying."""
    if isinstance(exc, NoMoralsError):
        return exc.retryable
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    return False


@dataclass
class RetryStats:
    """Aggregate outcome counters, attached to the wrapped function as ``.stats``."""

    calls: int = 0
    successes: int = 0
    failures: int = 0
    attempts: int = 0
    last_error: str = ""
    total_delay: float = 0.0
    by_error: dict[str, int] = field(default_factory=dict)

    def record_call(self) -> None:
        """One logical call, however many attempts it takes."""
        self.calls += 1

    def record_attempt(self) -> None:
        self.attempts += 1

    def record_success(self) -> None:
        self.successes += 1

    def record_failure(self, exc: BaseException) -> None:
        self.failures += 1
        code = getattr(exc, "code", type(exc).__name__)
        self.last_error = f"{code}: {exc}"
        self.by_error[code] = self.by_error.get(code, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "successes": self.successes,
            "failures": self.failures,
            "attempts": self.attempts,
            "avg_attempts": round(self.attempts / self.calls, 3) if self.calls else 0.0,
            "success_rate": round(self.successes / self.calls, 4) if self.calls else 0.0,
            "total_delay": round(self.total_delay, 3),
            "last_error": self.last_error,
            "by_error": dict(self.by_error),
        }


class RetryBudget:
    """Cap retries as a fraction of total calls (token-bucket enforced).

    Per-call retry policies multiply load 3–5x fleet-wide under partial
    outage even with jitter. The budget converts "every call retries
    independently" into "the system retries sustainably": amortized,
    ``retries <= retry_ratio * calls`` (default 10%).

    Token mechanics: each logical call deposits 1 token; each retry costs
    ``1/retry_ratio`` tokens. When the bucket is empty the retry is denied
    and the original error propagates immediately.
    """

    def __init__(self, retry_ratio: float = 0.1, min_tokens: float = 20.0) -> None:
        if not 0 < retry_ratio <= 1:
            raise ValueError("retry_ratio must be in (0, 1]")
        self.retry_ratio = retry_ratio
        self._cost = 1.0 / retry_ratio
        self._tokens = float(min_tokens)
        self._capacity = float(min_tokens)
        self._allowed = 0
        self._denied = 0
        self._lock = threading.Lock()

    def before_call(self) -> None:
        """Deposit one token for a new logical call."""
        with self._lock:
            self._tokens = min(self._capacity, self._tokens + 1.0)

    def allow_retry(self) -> bool:
        """Consume the retry cost; ``False`` means the budget is spent."""
        with self._lock:
            if self._tokens >= self._cost:
                self._tokens -= self._cost
                self._allowed += 1
                return True
            self._denied += 1
            return False

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "retry_ratio": self.retry_ratio,
                "tokens": round(self._tokens, 2),
                "capacity": self._capacity,
                "retries_allowed": self._allowed,
                "retries_denied": self._denied,
            }


@dataclass
class TimeoutPolicy:
    """Every remote call gets a deadline. Fail fast, then retry.

    Runs ``fn`` on a daemon thread and raises :class:`TimeoutError_` when the
    deadline expires. Honest limitation: a thread cannot be killed, so the
    timed-out call keeps running detached in the background — it just stops
    blocking *you*. Do not use for calls whose late side effects are harmful.
    """

    timeout: float

    def execute(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        box: dict[str, Any] = {}

        def target() -> None:
            try:
                box["result"] = fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                box["error"] = exc

        thread = threading.Thread(target=target, daemon=True,
                                  name=f"nm-timeout-{self.timeout:g}s")
        thread.start()
        thread.join(self.timeout)
        if thread.is_alive():
            raise TimeoutError_(
                f"timed out after {self.timeout:g}s",
                retryable=True,
                details={"timeout": self.timeout},
            )
        if "error" in box:
            raise box["error"]
        return box["result"]


@dataclass
class FallbackPolicy:
    """When the dependency is unavailable, return something useful instead.

    The fallback is a *product* decision — a cached value, a default, a
    "degraded" notice — made deliberately per call site, never a silent
    default. ``predicate`` selects which errors trigger the fallback;
    anything else propagates.
    """

    fallback: Callable[..., T]
    predicate: Callable[[BaseException], bool] | None = None

    @classmethod
    def with_value(cls, value: T,
                   predicate: Callable[[BaseException], bool] | None = None) -> "FallbackPolicy":
        """Fallback to a constant value."""
        return cls(lambda *a, **k: value, predicate=predicate)

    def execute(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - predicate decides
            if self.predicate is not None and not self.predicate(exc):
                raise
            _log.debug("fallback engaged after %s: %s", type(exc).__name__, exc)
            return self.fallback(*args, **kwargs)


@dataclass
class HedgingPolicy:
    """Race a slow call against a delayed duplicate; first success wins.

    After ``delay`` seconds without a *successful* result, a second attempt
    is launched in parallel (up to ``max_hedges`` hedges). The first success
    wins; if every attempt fails, the first error is raised.

    Hedged calls MUST be idempotent (or safely repeatable): two copies may
    genuinely execute. Never hedge a payment, a post, or any other
    non-idempotent write. ``total_timeout`` bounds the whole race so a
    standalone hedge cannot block forever.
    """

    delay: float = 2.0
    max_hedges: int = 1
    total_timeout: float | None = None

    def execute(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        total = 1 + max(0, self.max_hedges)
        results: queue.Queue[tuple[bool, Any]] = queue.Queue()
        stop = threading.Event()

        def run() -> None:
            if stop.is_set():
                return
            try:
                out = fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - collected below
                results.put((False, exc))
            else:
                if not stop.is_set():
                    results.put((True, out))

        def launch() -> None:
            threading.Thread(target=run, daemon=True,
                             name="nm-hedge").start()

        launched = 0
        launch()
        launched += 1
        errors: list[BaseException] = []
        deadline = None if self.total_timeout is None else time.monotonic() + self.total_timeout
        try:
            while True:
                remaining = total - launched
                wait = self.delay if remaining > 0 else None
                if deadline is not None:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise TimeoutError_(
                            f"hedged call exceeded total timeout of {self.total_timeout:g}s",
                            retryable=True)
                    wait = left if wait is None else min(wait, left)
                try:
                    ok, val = results.get(timeout=wait)
                except queue.Empty:
                    launch()  # delay elapsed with no success — hedge
                    launched += 1
                    continue
                if ok:
                    stop.set()
                    return val
                errors.append(val)
                if launched >= total and len(errors) >= launched:
                    break
                # an attempt failed while others may still be in flight —
                # keep waiting for them (or for the next hedge delay)
        finally:
            stop.set()
        if errors:
            raise errors[0]
        raise TimeoutError_("hedged call produced no result", retryable=True)


def retry_call(
    fn: Callable[..., T],
    *args: Any,
    policy: BackoffPolicy | None = None,
    retryable: Callable[[BaseException], bool] | None = None,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    breaker: "CircuitBreaker | None" = None,
    stats: RetryStats | None = None,
    budget: "RetryBudget | None" = None,
    **kwargs: Any,
) -> T:
    """Execute ``fn`` with retries. Raises the last error when attempts are exhausted.

    ``budget`` caps retries as a fraction of total traffic (see
    :class:`RetryBudget`); when it is spent the original error propagates
    immediately instead of retrying.
    """
    policy = policy or BackoffPolicy()
    predicate = retryable or default_retryable
    stats = stats or RetryStats()
    stats.record_call()
    if budget is not None:
        budget.before_call()
    last: BaseException | None = None
    prev_delay: float | None = None

    for attempt in range(1, policy.max_attempts + 1):
        if breaker is not None:
            breaker.before_call()
        stats.record_attempt()
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - classified below
            last = exc
            stats.record_failure(exc)
            if breaker is not None:
                breaker.record_failure(exc)
            if not predicate(exc) or attempt >= policy.max_attempts:
                raise
            if budget is not None and not budget.allow_retry():
                _log.debug("retry budget exhausted; not retrying %s", type(exc).__name__)
                raise
            retry_after = getattr(exc, "retry_after", None)
            delay = policy.delay(attempt, retry_after, prev_delay)
            prev_delay = delay
            stats.total_delay += delay
            _log.debug(
                "retry %d/%d after %.2fs (%s: %s)",
                attempt,
                policy.max_attempts,
                delay,
                type(exc).__name__,
                exc,
            )
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            if delay > 0:
                sleep(delay)
        else:
            if breaker is not None:
                breaker.record_success()
            stats.record_success()
            return result

    assert last is not None  # noqa: S101 - loop always sets it before raising
    raise last


def retry(
    policy: BackoffPolicy | None = None,
    *,
    retryable: Callable[[BaseException], bool] | None = None,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    breaker: "CircuitBreaker | None" = None,
    budget: "RetryBudget | None" = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator form of :func:`retry_call`.

        >>> @retry(BackoffPolicy(max_attempts=3, base=0.0))
        ... def flaky():
        ...     raise ProviderUnavailable("boom")
    """
    effective = policy or BackoffPolicy()

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        stats = RetryStats()

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return retry_call(
                fn,
                *args,
                policy=effective,
                retryable=retryable,
                on_retry=on_retry,
                breaker=breaker,
                stats=stats,
                budget=budget,
                **kwargs,
            )

        wrapper.stats = stats  # type: ignore[attr-defined]
        wrapper.policy = effective  # type: ignore[attr-defined]
        return wrapper

    return decorator


class CircuitBreaker:
    """Three-state breaker: closed → open → half-open.

    After ``failure_threshold`` consecutive failures the breaker opens and every
    call is rejected with :class:`CircuitOpen` for ``reset_timeout`` seconds. The
    next call is allowed through as a probe; success closes it, failure re-opens.

    This is what stops 64 sub-agents from turning one dead endpoint into a
    self-inflicted denial of service.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        name: str = "default",
        *,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
        half_open_max: int = 1,
        clock: Callable[[], float] = time.monotonic,
        on_open: Callable[[str], None] | None = None,
        on_close: Callable[[str], None] | None = None,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max = half_open_max
        self._clock = clock
        self._on_open = on_open
        self._on_close = on_close
        self._state = self.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._half_open_calls = 0
        self._lock = threading.RLock()
        self.stats = RetryStats()

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == self.OPEN and self._clock() - self._opened_at >= self.reset_timeout:
                self._state = self.HALF_OPEN
                self._half_open_calls = 0
            return self._state

    def before_call(self) -> None:
        with self._lock:
            current = self.state
            if current == self.OPEN:
                wait = self.reset_timeout - (self._clock() - self._opened_at)
                raise CircuitOpen(
                    f"circuit {self.name!r} open for another {wait:.1f}s",
                    details={"breaker": self.name, "retry_in": max(0.0, wait)},
                )
            if current == self.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max:
                    raise CircuitOpen(
                        f"circuit {self.name!r} half-open probe limit reached",
                        details={"breaker": self.name},
                    )
                self._half_open_calls += 1

    def record_success(self) -> None:
        with self._lock:
            was_open = self._state in (self.OPEN, self.HALF_OPEN)
            self._failures = 0
            self._state = self.CLOSED
            self.stats.record_success()
        if was_open and self._on_close is not None:
            try:
                self._on_close(self.name)
            except Exception:  # noqa: BLE001 - observability must not break the breaker
                _log.exception("breaker on_close hook failed")

    def _open(self) -> None:
        self._state = self.OPEN
        self._opened_at = self._clock()
        _log.warning("circuit %r opened after %d failures", self.name, self._failures)
        if self._on_open is not None:
            try:
                self._on_open(self.name)
            except Exception:  # noqa: BLE001 - observability must not break the breaker
                _log.exception("breaker on_open hook failed")

    def record_failure(self, exc: BaseException | None = None) -> None:
        with self._lock:
            if exc is not None:
                self.stats.record_failure(exc)
            self._failures += 1
            if self._state == self.HALF_OPEN or self._failures >= self.failure_threshold:
                if self._state != self.OPEN:
                    self._open()
                else:
                    # already open and still failing: keep the window pushed out
                    self._opened_at = self._clock()

    def reset(self) -> None:
        with self._lock:
            self._state = self.CLOSED
            self._failures = 0
            self._half_open_calls = 0

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "state": self.state,
                "failures": self._failures,
                "failure_threshold": self.failure_threshold,
                "reset_timeout": self.reset_timeout,
                "stats": self.stats.as_dict(),
            }


class BreakerRegistry:
    """One breaker per named dependency, shared across threads and agents."""

    def __init__(self, **defaults: Any) -> None:
        self._defaults = defaults
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> CircuitBreaker:
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, **self._defaults)
            return self._breakers[name]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {name: b.as_dict() for name, b in self._breakers.items()}

    def reset_all(self) -> None:
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()


class ResiliencePipeline:
    """Compose resilience policies into one pipeline, Polly-v8-style.

    Execution order (innermost first): ``timeout → hedging → retry →
    circuit breaker → fallback``. The timeout bounds each attempt; hedging
    races slow attempts; retry handles transient failures inside the
    wrapper; the breaker sees the *logical* call outcome once (so retries
    don't trip it N times faster); fallback is the deliberate last resort.

        >>> pipeline = (ResiliencePipeline("llm")
        ...     .with_timeout(20.0)
        ...     .with_retry(BackoffPolicy(max_attempts=3),
        ...                 budget=RetryBudget(0.1))
        ...     .with_breaker(failure_threshold=5, reset_timeout=30.0)
        ...     .with_fallback(FallbackPolicy.with_value("degraded")))
        >>> pipeline.execute(call_llm, prompt)
        'degraded'

    Reusable across threads; per-call stats accumulate on the shared
    :class:`RetryStats`.
    """

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._timeout: TimeoutPolicy | None = None
        self._hedging: HedgingPolicy | None = None
        self._retry_policy: BackoffPolicy | None = None
        self._retryable: Callable[[BaseException], bool] | None = None
        self._on_retry: Callable[[int, BaseException, float], None] | None = None
        self._budget: RetryBudget | None = None
        self._breaker: CircuitBreaker | None = None
        self._fallback: FallbackPolicy | None = None
        self.stats = RetryStats()

    # -- builder -----------------------------------------------------------
    def with_timeout(self, seconds: float) -> "ResiliencePipeline":
        self._timeout = TimeoutPolicy(seconds)
        return self

    def with_hedging(self, delay: float = 2.0, max_hedges: int = 1,
                     total_timeout: float | None = None) -> "ResiliencePipeline":
        self._hedging = HedgingPolicy(delay=delay, max_hedges=max_hedges,
                                      total_timeout=total_timeout)
        return self

    def with_retry(
        self,
        policy: BackoffPolicy | None = None,
        *,
        retryable: Callable[[BaseException], bool] | None = None,
        on_retry: Callable[[int, BaseException, float], None] | None = None,
        budget: RetryBudget | None = None,
    ) -> "ResiliencePipeline":
        self._retry_policy = policy or BackoffPolicy()
        self._retryable = retryable
        self._on_retry = on_retry
        self._budget = budget
        return self

    def with_breaker(self, name: str | None = None, **kwargs: Any) -> "ResiliencePipeline":
        self._breaker = CircuitBreaker(name or self.name, **kwargs)
        return self

    def with_fallback(self, fallback: FallbackPolicy | Callable[..., Any],
                      predicate: Callable[[BaseException], bool] | None = None,
                      ) -> "ResiliencePipeline":
        if isinstance(fallback, FallbackPolicy):
            self._fallback = fallback
        else:
            self._fallback = FallbackPolicy(fallback, predicate=predicate)
        return self

    # -- execution ---------------------------------------------------------
    def execute(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run ``fn`` through the composed policies."""
        def core() -> T:
            op: Callable[..., T] = fn
            timeout = self._timeout
            if timeout is not None:
                inner = op
                def op(*a: Any, _inner=inner, _t=timeout, **k: Any) -> T:  # noqa: E306
                    return _t.execute(_inner, *a, **k)
            hedging = self._hedging
            if hedging is not None:
                inner = op
                def op(*a: Any, _inner=inner, _h=hedging, **k: Any) -> T:  # noqa: E306
                    return _h.execute(_inner, *a, **k)
            if self._retry_policy is not None:
                return retry_call(
                    op, *args,
                    policy=self._retry_policy,
                    retryable=self._retryable,
                    on_retry=self._on_retry,
                    stats=self.stats,
                    budget=self._budget,
                    **kwargs,
                )
            self.stats.record_call()
            try:
                result = op(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - stats only
                self.stats.record_failure(exc)
                raise
            self.stats.record_success()
            return result

        def guarded() -> T:
            breaker = self._breaker
            if breaker is None:
                return core()
            try:
                breaker.before_call()
            except Exception as exc:  # noqa: BLE001 - breaker rejected the call
                # The retry loop never ran; record the rejection so pipeline
                # stats stay truthful about every logical call.
                self.stats.record_call()
                self.stats.record_failure(exc)
                raise
            try:
                result = core()
            except Exception as exc:  # noqa: BLE001 - breaker accounting
                breaker.record_failure(exc)
                raise
            breaker.record_success()
            return result

        if self._fallback is not None:
            return self._fallback.execute(guarded)
        return guarded()

    def describe(self) -> dict[str, Any]:
        """Configured strategies, for tests and dashboards."""
        return {
            "name": self.name,
            "timeout": None if self._timeout is None else self._timeout.timeout,
            "hedging": None if self._hedging is None else {
                "delay": self._hedging.delay, "max_hedges": self._hedging.max_hedges},
            "retry": None if self._retry_policy is None else {
                "max_attempts": self._retry_policy.max_attempts,
                "jitter_style": self._retry_policy.jitter_style.value,
                "budget": None if self._budget is None else self._budget.as_dict()},
            "breaker": None if self._breaker is None else self._breaker.as_dict(),
            "fallback": self._fallback is not None,
            "stats": self.stats.as_dict(),
        }
