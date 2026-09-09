"""Rate limiting: token buckets, sliding windows, and per-key registries.

Used for three distinct purposes, all needing the same primitive:

* outbound HTTP to a provider (respect their limits),
* social platform posting (respect platform limits, and keep accounts alive),
* internal resource access (do not open 500 sockets at once).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque

__all__ = [
    "RateLimiter",
    "RateLimitExceeded",
    "SemaphorePool",
    "SlidingWindowLimiter",
    "TokenBucket",
    "limiter_registry",
]


class RateLimitExceeded(Exception):
    """Raised by non-blocking acquisition when the bucket is empty."""

    def __init__(self, retry_after: float) -> None:
        super().__init__(f"rate limited; retry after {retry_after:.3f}s")
        self.retry_after = retry_after


@dataclass
class TokenBucket:
    """Classic token bucket.

    Tokens refill continuously at ``rate`` per second up to ``capacity``. A burst
    of ``capacity`` is therefore allowed after an idle period, which matches how
    nearly every real API limit behaves.
    """

    rate: float
    capacity: float
    _tokens: float = field(init=False)
    _updated: float = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.rate <= 0:
            raise ValueError("rate must be positive")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self._tokens = float(self.capacity)
        self._updated = time.monotonic()

    def _refill(self, now: float) -> None:
        elapsed = now - self._updated
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._updated = now

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Take tokens if available; never blocks."""
        with self._lock:
            now = time.monotonic()
            self._refill(now)
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def wait_time(self, tokens: float = 1.0) -> float:
        """Seconds until ``tokens`` would be available (0 if now)."""
        with self._lock:
            self._refill(time.monotonic())
            if self._tokens >= tokens:
                return 0.0
            return (tokens - self._tokens) / self.rate

    def acquire(self, tokens: float = 1.0, timeout: float | None = None) -> bool:
        """Block until tokens are available. Returns False on timeout."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            wait = self.wait_time(tokens)
            if wait <= 0:
                if self.try_acquire(tokens):
                    return True
                continue
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                wait = min(wait, remaining)
            time.sleep(min(wait, 0.25))

    def throttle(self, tokens: float = 1.0) -> float:
        """Block until permitted; returns the number of seconds waited."""
        waited = 0.0
        while True:
            wait = self.wait_time(tokens)
            if wait <= 0:
                if self.try_acquire(tokens):
                    return waited
                continue
            time.sleep(wait)
            waited += wait

    @property
    def available(self) -> float:
        with self._lock:
            self._refill(time.monotonic())
            return self._tokens

    def reset(self) -> None:
        with self._lock:
            self._tokens = self.capacity
            self._updated = time.monotonic()

    def as_dict(self) -> dict[str, float]:
        return {
            "rate": self.rate,
            "capacity": self.capacity,
            "available": round(self.available, 4),
        }


class RateLimiter(TokenBucket):
    """Alias with a name that reads better at call sites."""


class SlidingWindowLimiter:
    """At most ``limit`` events per ``window`` seconds, counted precisely.

    More conservative than a token bucket: it forbids the double-burst that a
    bucket allows at a window boundary. Used for social posting budgets where a
    burst is exactly what gets an account flagged.
    """

    def __init__(self, limit: int, window: float) -> None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        self.limit = limit
        self.window = window
        self._events: Deque[float] = deque()
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        cutoff = now - self.window
        while self._events and self._events[0] <= cutoff:
            self._events.popleft()

    def try_acquire(self) -> bool:
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            if len(self._events) >= self.limit:
                return False
            self._events.append(now)
            return True

    def wait_time(self) -> float:
        with self._lock:
            self._prune(time.monotonic())
            if len(self._events) < self.limit:
                return 0.0
            return max(0.0, self._events[0] + self.window - time.monotonic())

    def acquire(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            wait = self.wait_time()
            if wait <= 0:
                if self.try_acquire():
                    return True
                continue
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                wait = min(wait, remaining)
            time.sleep(min(wait, 0.25))

    @property
    def remaining(self) -> int:
        with self._lock:
            self._prune(time.monotonic())
            return max(0, self.limit - len(self._events))

    def reset(self) -> None:
        with self._lock:
            self._events.clear()

    def as_dict(self) -> dict[str, float]:
        return {"limit": self.limit, "window": self.window, "remaining": self.remaining}


class SemaphorePool:
    """Named semaphores bounding concurrent use of a scarce resource.

    Keeps 64 sub-agents from opening 64 sockets or 8 GPU contexts.
    """

    def __init__(self, permits: dict[str, int] | None = None) -> None:
        self._semaphores: dict[str, threading.BoundedSemaphore] = {}
        self._limits: dict[str, int] = dict(permits or {})
        self._in_use: dict[str, int] = {}
        # RLock: _get() holds the lock while calling register(), which takes it too.
        self._lock = threading.RLock()
        for name, count in self._limits.items():
            self._semaphores[name] = threading.BoundedSemaphore(count)
            self._in_use[name] = 0

    def register(self, name: str, permits: int) -> None:
        """Declare (or redeclare) the permit count for a resource."""
        with self._lock:
            self._limits[name] = permits
            self._semaphores[name] = threading.BoundedSemaphore(permits)
            self._in_use.setdefault(name, 0)

    def _get(self, name: str) -> threading.BoundedSemaphore:
        with self._lock:
            if name not in self._semaphores:
                # Default to unbounded-ish but tracked, so unknown resources still work.
                self.register(name, self._limits.get(name, 1024))
            return self._semaphores[name]

    def acquire(self, name: str, timeout: float | None = None) -> bool:
        sem = self._get(name)
        acquired = sem.acquire(timeout=timeout) if timeout is not None else sem.acquire()
        if acquired:
            with self._lock:
                self._in_use[name] = self._in_use.get(name, 0) + 1
        return acquired

    def release(self, name: str) -> None:
        with self._lock:
            sem = self._semaphores.get(name)
            if sem is None:
                return
            self._in_use[name] = max(0, self._in_use.get(name, 1) - 1)
        sem.release()

    class _Slot:
        """Guarantees the body runs only while a permit is actually held.

        ``__enter__`` raising rather than returning ``False`` is deliberate: a
        context manager that reports failure through its return value still
        executes the body, which silently defeats the concurrency bound. That is
        precisely the failure mode this class exists to prevent.
        """

        def __init__(self, pool: "SemaphorePool", name: str, timeout: float | None) -> None:
            self._pool = pool
            self._name = name
            self._timeout = timeout

        def __enter__(self) -> "SemaphorePool._Slot":
            if not self._pool.acquire(self._name, self._timeout):
                raise TimeoutError(
                    f"timed out after {self._timeout}s waiting for a {self._name!r} permit"
                )
            return self

        def __exit__(self, *exc: object) -> None:
            self._pool.release(self._name)

    class _TrySlot:
        """Non-blocking variant: the body runs only if a permit was free *now*."""

        def __init__(self, pool: "SemaphorePool", name: str) -> None:
            self._pool = pool
            self._name = name
            self.acquired = False

        def __enter__(self) -> bool:
            self.acquired = self._pool.acquire(self._name, timeout=0.0)
            return self.acquired

        def __exit__(self, *exc: object) -> None:
            if self.acquired:
                self._pool.release(self._name)

    def slot(self, name: str, timeout: float | None = None) -> "_Slot":
        """Blocking context manager. Raises :class:`TimeoutError` if it cannot acquire.

            with pool.slot("network"):
                ...  # always runs with a permit held
        """
        return SemaphorePool._Slot(self, name, timeout)

    def try_slot(self, name: str) -> "_TrySlot":
        """Non-blocking context manager; ``__enter__`` returns whether it got a permit.

            with pool.try_slot("gpu") as got:
                if not got:
                    return defer()
        """
        return SemaphorePool._TrySlot(self, name)

    def snapshot(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {
                name: {"limit": self._limits.get(name, 0), "in_use": self._in_use.get(name, 0)}
                for name in self._semaphores
            }


class LimiterRegistry:
    """Per-key rate limiters, e.g. one bucket per social platform or provider."""

    def __init__(self, factory: Callable[[str], TokenBucket] | None = None) -> None:
        self._factory = factory or (lambda key: TokenBucket(rate=1.0, capacity=4.0))
        self._limiters: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> TokenBucket:
        with self._lock:
            if key not in self._limiters:
                self._limiters[key] = self._factory(key)
            return self._limiters[key]

    def set(self, key: str, rate: float, capacity: float) -> TokenBucket:
        with self._lock:
            bucket = TokenBucket(rate=rate, capacity=capacity)
            self._limiters[key] = bucket
            return bucket

    def snapshot(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return {key: bucket.as_dict() for key, bucket in self._limiters.items()}


limiter_registry = LimiterRegistry()
