"""Retry with backoff, jitter, and circuit breaking.

Three pieces:

* :class:`BackoffPolicy` — how long to wait between attempts.
* :func:`retry` — decorator/wrapper that applies the policy.
* :class:`CircuitBreaker` — stops hammering a dependency that is clearly down.

Retryable-ness is decided by the *error*, not by guesswork: every framework error
carries a ``retryable`` flag, and the default predicate consults it.
"""

from __future__ import annotations

import functools
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence, TypeVar

from .errors import NoMoralsError, ProviderUnavailable, RateLimited, classify

__all__ = [
    "BackoffPolicy",
    "CircuitBreaker",
    "CircuitOpen",
    "RetryStats",
    "retry",
    "retry_call",
]

_log = logging.getLogger(__name__)
T = TypeVar("T")


class CircuitOpen(NoMoralsError):
    """The breaker is open; calls are being rejected without hitting the dependency."""

    code = "circuit.open"
    retryable = True


@dataclass
class BackoffPolicy:
    """Exponential backoff with full jitter.

    Delay for attempt ``n`` is ``uniform(0, min(cap, base * factor**n))`` unless
    ``jitter`` is disabled, in which case the deterministic value is used. Full
    jitter is the AWS-recommended choice: it decorrelates thundering herds when
    many sub-agents hit the same rate limit simultaneously.
    """

    base: float = 0.5
    factor: float = 2.0
    cap: float = 60.0
    max_attempts: int = 4
    jitter: bool = True
    respect_retry_after: bool = True

    def delay(self, attempt: int, retry_after: float | None = None) -> float:
        if retry_after is not None and self.respect_retry_after:
            return max(0.0, min(float(retry_after), self.cap))
        raw = min(self.cap, self.base * (self.factor ** max(0, attempt - 1)))
        if self.jitter:
            return random.uniform(0.0, raw)
        return raw

    def schedule(self, attempts: int | None = None) -> list[float]:
        count = attempts or self.max_attempts
        return [self.delay(i) for i in range(1, count)]


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


def retry_call(
    fn: Callable[..., T],
    *args: Any,
    policy: BackoffPolicy | None = None,
    retryable: Callable[[BaseException], bool] | None = None,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    breaker: "CircuitBreaker | None" = None,
    stats: RetryStats | None = None,
    **kwargs: Any,
) -> T:
    """Execute ``fn`` with retries. Raises the last error when attempts are exhausted."""
    policy = policy or BackoffPolicy()
    predicate = retryable or default_retryable
    stats = stats or RetryStats()
    stats.record_call()
    last: BaseException | None = None

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
            retry_after = getattr(exc, "retry_after", None)
            delay = policy.delay(attempt, retry_after)
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
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max = half_open_max
        self._clock = clock
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
            self._failures = 0
            self._state = self.CLOSED
            self.stats.record_success()

    def record_failure(self, exc: BaseException | None = None) -> None:
        with self._lock:
            if exc is not None:
                self.stats.record_failure(exc)
            self._failures += 1
            if self._state == self.HALF_OPEN or self._failures >= self.failure_threshold:
                self._state = self.OPEN
                self._opened_at = self._clock()
                _log.warning("circuit %r opened after %d failures", self.name, self._failures)

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
