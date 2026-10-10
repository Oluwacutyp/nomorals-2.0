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
from typing import Any, Callable, Deque, Protocol

from .errors import RateLimited

# ── human presets & store protocol ─────────────────────────────────────────

#: "100/minute", "10/second", "1000/hour", "5000/day" → (limit, window_s).
_RATE_PRESETS = {
    "second": 1.0, "sec": 1.0, "s": 1.0,
    "minute": 60.0, "min": 60.0, "m": 60.0,
    "hour": 3600.0, "h": 3600.0,
    "day": 86400.0, "d": 86400.0,
}


def parse_rate(spec: str) -> tuple[float, float]:
    """Parse ``"100/minute"`` → ``(100.0, 60.0)`` (limit, window seconds).

    Accepts ``"<n>/<unit>"`` and ``"<n> per <unit>"``; units: second(s),
    minute(s), hour(s), day(s) and their abbreviations.
    """
    import re as _re

    m = _re.match(r"^\s*(\d+(?:\.\d+)?)\s*(?:/|per\s+)\s*([a-z]+)\s*$",
                  spec.strip().lower())
    if not m:
        raise ValueError(f"bad rate spec {spec!r}; want like '100/minute'")
    limit = float(m.group(1))
    raw_unit = m.group(2)
    # normalize plurals: "seconds"→"second", "mins"→"min"→"minute"…
    unit = raw_unit
    if unit.endswith("s") and unit not in _RATE_PRESETS:
        unit = unit[:-1]
    window = _RATE_PRESETS.get(unit)
    if window is None:
        raise ValueError(f"unknown rate unit {raw_unit!r} in {spec!r}")
    if limit <= 0:
        raise ValueError("rate limit must be positive")
    return limit, window


def limiter_from_preset(spec: str, *,
                        algorithm: str = "token_bucket",
                        burst: float | None = None) -> TokenBucket:
    """Build a limiter from ``"100/minute"``.

    ``algorithm``: ``token_bucket`` (burst-tolerant, the default),
    ``sliding_window`` (no boundary burst — for social posting),
    ``gcra`` (cheapest per-key). ``burst`` overrides the token bucket
    capacity (defaults to the per-second rate, i.e. a 1-second burst).
    """
    limit, window = parse_rate(spec)
    rate = limit / window
    if algorithm == "token_bucket":
        return TokenBucket(rate=rate, capacity=burst or max(1.0, rate))
    if algorithm == "sliding_window":
        return SlidingWindowLimiter(limit=max(1, int(limit)), window=window)
    if algorithm == "gcra":
        return GCRALimiter(rate=rate, period=1.0,
                           burst=max(1, int(burst or rate)))
    raise ValueError(f"unknown algorithm {algorithm!r}")


class LimiterStore(Protocol):
    """Pluggable per-key state backend for distributed limiting.

    The in-process limiters stay allocation-free; a Redis (or SQLite)
    implementation of this protocol lets the *same* algorithms run
    fleet-wide later (ratelink's decoupled-backend design). Methods are
    synchronous; a distributed backend should do one round-trip per call.
    """

    def load(self, key: str) -> dict[str, float] | None:
        """Stored state for ``key`` (``None`` = fresh)."""
        ...
    def save(self, key: str, state: dict[str, float], ttl_s: float) -> None:
        """Persist state with a TTL so dead keys evaporate."""
        ...


class MemoryLimiterStore:
    """In-process :class:`LimiterStore` — the default; also the test double
    for distributed backends."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[dict[str, float], float]] = {}
        self._lock = threading.Lock()

    def load(self, key: str) -> dict[str, float] | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            state, expires = entry
            if expires < time.monotonic():
                del self._data[key]
                return None
            return dict(state)

    def save(self, key: str, state: dict[str, float], ttl_s: float) -> None:
        with self._lock:
            self._data[key] = (dict(state), time.monotonic() + ttl_s)

__all__ = [
    "GCRALimiter",
    "LeakyBucket",
    "LimiterStore",
    "MemoryLimiterStore",
    "RateLimitDecision",
    "RateLimiter",
    "RateLimitExceeded",
    "SemaphorePool",
    "SlidingWindowCounter",
    "SlidingWindowLimiter",
    "TokenBucket",
    "limiter_from_preset",
    "limiter_registry",
    "parse_rate",
]


class RateLimitExceeded(RateLimited):
    """Non-blocking acquisition failed.

    Unifies with :class:`nomorals.core.errors.RateLimited` (one error
    dialect): it is ``retryable`` and carries ``retry_after``, so the retry
    loop in :mod:`nomorals.core.retry` honors it automatically.
    """

    code = "rate_limited"

    def __init__(self, retry_after: float) -> None:
        super().__init__(
            f"rate limited; retry after {max(0.0, retry_after):.3f}s",
            retry_after=max(0.0, float(retry_after)),
        )


@dataclass
class RateLimitDecision:
    """Outcome of one admission check, with client-facing signals.

    A bare 429 forces every integration to guess when it is safe to retry;
    these headers let well-behaved clients back off correctly instead of
    tight-looping.
    """

    allowed: bool
    limit: float
    remaining: float
    reset_after: float  # seconds until the allowance meaningfully refills
    retry_after: float = 0.0  # >0 only when not allowed

    def to_headers(self) -> dict[str, str]:
        """``Retry-After`` + ``X-RateLimit-*`` headers for a 429 response."""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": f"{max(0.0, self.remaining):.4f}",
            "X-RateLimit-Reset": str(int(time.time() + max(0.0, self.reset_after))),
        }
        if not self.allowed:
            headers["Retry-After"] = f"{max(0.0, self.retry_after):.3f}"
        return headers

    def or_raise(self) -> "RateLimitDecision":
        """Raise :class:`RateLimitExceeded` when not allowed; else return self."""
        if not self.allowed:
            raise RateLimitExceeded(self.retry_after)
        return self


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

    def decide(self, tokens: float = 1.0) -> RateLimitDecision:
        """Single atomic admission check that consumes on allow.

        ``tokens`` is the *cost* of the operation — an expensive search can
        cost more budget than a cheap lookup, so a handful of heavy calls
        cannot do more damage than a thousand light ones.
        """
        with self._lock:
            now = time.monotonic()
            self._refill(now)
            if self._tokens >= tokens:
                self._tokens -= tokens
                return RateLimitDecision(
                    allowed=True,
                    limit=self.capacity,
                    remaining=self._tokens,
                    reset_after=0.0,
                )
            deficit = tokens - self._tokens
            wait = deficit / self.rate
            return RateLimitDecision(
                allowed=False,
                limit=self.capacity,
                remaining=self._tokens,
                reset_after=wait,
                retry_after=wait,
            )

    def acquire_or_raise(self, tokens: float = 1.0) -> RateLimitDecision:
        """Like :meth:`decide`, but raises :class:`RateLimitExceeded`.

        The raised error is ``retryable`` with ``retry_after`` set, so the
        retry loop honors it — one error dialect across the codebase.
        """
        return self.decide(tokens).or_raise()


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

    def decide(self) -> RateLimitDecision:
        """Atomic admission check that consumes on allow."""
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            if len(self._events) < self.limit:
                self._events.append(now)
                return RateLimitDecision(
                    allowed=True,
                    limit=float(self.limit),
                    remaining=float(self.limit - len(self._events)),
                    reset_after=0.0,
                )
            wait = max(0.0, self._events[0] + self.window - now)
            return RateLimitDecision(
                allowed=False,
                limit=float(self.limit),
                remaining=0.0,
                reset_after=wait,
                retry_after=wait,
            )

    def acquire_or_raise(self) -> RateLimitDecision:
        return self.decide().or_raise()

    def as_dict(self) -> dict[str, float]:
        return {"limit": self.limit, "window": self.window, "remaining": self.remaining}


class GCRALimiter:
    """Generic Cell Rate Algorithm — token-bucket-equivalent rate meter.

    State is a single timestamp (TAT, the theoretical arrival time): O(1)
    memory per key, exact, and cheaper than a token bucket at high key
    counts. ``rate`` events per ``period`` seconds sustained, with bursts up
    to ``burst`` absorbed.

    (ATM Forum TM 4.0; the same math as token bucket / leaky bucket as a
    meter — pick this for high-throughput per-key limiting, token bucket for
    app code.)
    """

    def __init__(self, rate: float, period: float = 1.0, burst: int = 1) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        if period <= 0:
            raise ValueError("period must be positive")
        if burst < 1:
            raise ValueError("burst must be >= 1")
        self.rate = rate
        self.period = period
        self.burst = burst
        self._emission_interval = period / rate
        # Tolerance L: exactly `burst` back-to-back cells conform, the
        # (burst+1)-th does not. Proof: after k cells TAT = t0 + k*T and the
        # k-th conforms iff t0 >= t0 + (k-1)*T - L, i.e. L >= (k-1)*T;
        # the (burst+1)-th needs L >= burst*T. So (burst-1)*T <= L < burst*T.
        self._tolerance = (burst - 1) * self._emission_interval
        self._tat: float | None = None
        self._lock = threading.Lock()

    def _simulate(self, tat: float, at: float, cost: int) -> tuple[bool, float]:
        """Would ``cost`` cells arriving at ``at`` conform? Returns
        (conforms, resulting TAT) — or (False, TAT-at-first-failure)."""
        t = tat
        for _ in range(cost):
            if at < t - self._tolerance:
                return False, t
            t = max(at, t) + self._emission_interval
        return True, t

    def _wait_for(self, tat: float, now: float, cost: int) -> float:
        """Seconds until ``cost`` cells arriving together would conform."""
        at = now
        for _ in range(cost + 1):  # converges in <= cost steps
            ok, t = self._simulate(tat, at, cost)
            if ok:
                return max(0.0, at - now)
            at = max(at, t - self._tolerance)
        return max(0.0, at - now)

    def _remaining(self, tat: float, now: float) -> int:
        """How many more single cells could arrive right now."""
        t = tat
        n = 0
        while n <= self.burst and now >= t - self._tolerance:
            n += 1
            t = max(now, t) + self._emission_interval
        return n

    def decide(self, cost: int = 1) -> RateLimitDecision:
        """Atomic admission check that consumes ``cost`` cells on allow."""
        if cost < 1:
            raise ValueError("cost must be >= 1")
        with self._lock:
            now = time.monotonic()
            tat = self._tat if self._tat is not None else now
            ok, new_tat = self._simulate(tat, now, cost)
            if ok:
                self._tat = new_tat
                return RateLimitDecision(
                    allowed=True, limit=float(self.burst),
                    remaining=float(self._remaining(new_tat, now)),
                    reset_after=0.0)
            wait = self._wait_for(tat, now, cost)
            return RateLimitDecision(
                allowed=False, limit=float(self.burst), remaining=0.0,
                reset_after=wait, retry_after=wait)

    def try_acquire(self, cost: int = 1) -> bool:
        return self.decide(cost).allowed

    def wait_time(self, cost: int = 1) -> float:
        with self._lock:
            now = time.monotonic()
            tat = self._tat if self._tat is not None else now
            ok, _ = self._simulate(tat, now, cost)
            if ok:
                return 0.0
            return self._wait_for(tat, now, cost)

    def acquire_or_raise(self, cost: int = 1) -> RateLimitDecision:
        return self.decide(cost).or_raise()

    def reset(self) -> None:
        with self._lock:
            self._tat = None

    def as_dict(self) -> dict[str, float]:
        return {
            "rate": self.rate, "period": self.period, "burst": float(self.burst),
            "wait": round(self.wait_time(), 4),
        }


class LeakyBucket:
    """Leaky bucket *shaper*: smooths bursty arrivals into a steady outflow.

    The meter half (token bucket / GCRA) decides *whether* a request may go;
    the leaky bucket decides *when*: water pours in per request and leaks at
    ``rate``/s. Unlike the bucket-as-meter, this is used to *pace* outbound
    work — ``throttle()`` blocks until the request's turn, so a burst of 50
    queued posts leaves as 50 evenly-spaced posts instead of one spike that
    gets the account flagged.

    ``capacity`` bounds how much burst can queue before callers start
    waiting; ``decide()`` never blocks and reports the wait instead.
    """

    def __init__(self, rate: float, capacity: float) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.rate = rate
        self.capacity = capacity
        self._level = 0.0
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def _leak(self, now: float) -> None:
        elapsed = now - self._updated
        if elapsed > 0:
            self._level = max(0.0, self._level - elapsed * self.rate)
            self._updated = now

    def decide(self, cost: float = 1.0) -> RateLimitDecision:
        """Admit if the pour fits; otherwise report when it would."""
        if cost <= 0:
            raise ValueError("cost must be positive")
        with self._lock:
            now = time.monotonic()
            self._leak(now)
            if self._level + cost <= self.capacity:
                self._level += cost
                return RateLimitDecision(
                    allowed=True, limit=self.capacity,
                    remaining=self.capacity - self._level, reset_after=0.0)
            # wait until enough has leaked for this pour to fit
            wait = (self._level + cost - self.capacity) / self.rate
            return RateLimitDecision(
                allowed=False, limit=self.capacity,
                remaining=max(0.0, self.capacity - self._level),
                reset_after=wait, retry_after=wait)

    def wait_time(self, cost: float = 1.0) -> float:
        """Seconds until ``cost`` would be admitted (0 if now). Non-consuming."""
        with self._lock:
            now = time.monotonic()
            self._leak(now)
            if self._level + cost <= self.capacity:
                return 0.0
            return (self._level + cost - self.capacity) / self.rate

    def try_acquire(self, cost: float = 1.0) -> bool:
        return self.decide(cost).allowed

    def acquire_or_raise(self, cost: float = 1.0) -> RateLimitDecision:
        return self.decide(cost).or_raise()

    def throttle(self, cost: float = 1.0) -> float:
        """Block until the pour is admitted; returns seconds waited."""
        waited = 0.0
        while True:
            decision = self.decide(cost)
            if decision.allowed:
                return waited
            time.sleep(min(decision.retry_after, 0.25))
            waited += min(decision.retry_after, 0.25)

    def as_dict(self) -> dict[str, float]:
        with self._lock:
            self._leak(time.monotonic())
            return {"rate": self.rate, "capacity": self.capacity,
                    "level": round(self._level, 4)}


class SlidingWindowCounter:
    """Approximate sliding window: ~99% accurate, O(1) memory.

    Keeps two fixed-window counters (previous + current) and weights the
    previous window by its overlap with the sliding window. No 2x boundary
    burst like a naive fixed window, no O(limit) timestamp log. The best
    default for distributed per-key limiting.
    """

    def __init__(self, limit: int, window: float) -> None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        if window <= 0:
            raise ValueError("window must be positive")
        self.limit = limit
        self.window = window
        self._prev_count = 0
        self._cur_count = 0
        self._window_start = time.monotonic()
        self._lock = threading.Lock()

    def _roll(self, now: float) -> None:
        elapsed = now - self._window_start
        if elapsed >= self.window:
            windows = int(elapsed // self.window)
            if windows >= 2:
                self._prev_count = 0
            else:
                self._prev_count = self._cur_count
            self._cur_count = 0
            self._window_start += windows * self.window

    def _estimate(self, now: float) -> float:
        self._roll(now)
        overlap = max(0.0, 1.0 - (now - self._window_start) / self.window)
        return self._prev_count * overlap + self._cur_count

    def decide(self, cost: int = 1) -> RateLimitDecision:
        if cost < 1:
            raise ValueError("cost must be >= 1")
        with self._lock:
            now = time.monotonic()
            estimate = self._estimate(now)
            if estimate + cost <= self.limit:
                self._cur_count += cost
                return RateLimitDecision(
                    allowed=True, limit=float(self.limit),
                    remaining=max(0.0, self.limit - estimate - cost),
                    reset_after=0.0)
            # wait until enough of the previous window slides out
            wait = self._window_start + self.window - now
            return RateLimitDecision(
                allowed=False, limit=float(self.limit), remaining=0.0,
                reset_after=max(0.0, wait), retry_after=max(0.0, wait))

    def try_acquire(self, cost: int = 1) -> bool:
        return self.decide(cost).allowed

    def wait_time(self, cost: int = 1) -> float:
        """Exact seconds until ``cost`` would be admitted (``inf`` if never).

        Accounts for window rolls during the wait: events slide into the
        previous window with decaying weight, so the answer is not simply
        "until the window ends".
        """
        if cost < 1:
            raise ValueError("cost must be >= 1")
        if cost > self.limit:
            return float("inf")  # can never be admitted
        with self._lock:
            now = time.monotonic()
            ws, prev, cur = self._window_start, self._prev_count, self._cur_count
            # roll the *snapshot* forward to now (no mutation)
            while now - ws >= self.window:
                prev, cur = cur, 0
                ws += self.window
            t = now
            for _ in range(4):  # at most ~2 rolls needed; bounded regardless
                if prev <= 0:
                    if cur + cost <= self.limit:
                        return max(0.0, t - now)
                    # constant within this segment — advance past the roll
                    t = ws + self.window
                    prev, cur = cur, 0
                    ws = t
                    continue
                # solve prev*(1-(s-ws)/w) + cur + cost <= limit for s
                target = (self.limit - cur - cost) / prev
                s_needed = ws + self.window * (1.0 - target)
                # s_needed is the exact crossing point; never report a time
                # before the already-simulated t.
                if s_needed < ws + self.window:
                    return max(0.0, max(s_needed, t) - now)
                t = ws + self.window
                prev, cur = cur, 0
                ws = t
            return max(0.0, t - now)

    def acquire_or_raise(self, cost: int = 1) -> RateLimitDecision:
        return self.decide(cost).or_raise()

    def reset(self) -> None:
        with self._lock:
            self._prev_count = 0
            self._cur_count = 0
            self._window_start = time.monotonic()

    def as_dict(self) -> dict[str, float]:
        with self._lock:
            now = time.monotonic()
            return {"limit": float(self.limit), "window": self.window,
                    "estimated": round(self._estimate(now), 3)}


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

    def describe(self, theme: Any = None) -> str:
        """Human-readable limiter table for dashboards and chat."""
        from .style import styled_table, header

        snap = self.snapshot()
        if not snap:
            return "no limiters registered"
        rows = []
        for key in sorted(snap):
            s = snap[key]
            rows.append([
                key,
                f"{s.get('rate', s.get('limit', '?'))}",
                f"{s.get('available', s.get('remaining', '?'))}",
            ])
        return (header("rate limiters", theme) + "\n"
                + styled_table(["limiter", "rate", "available"], rows, theme))


limiter_registry = LimiterRegistry()
