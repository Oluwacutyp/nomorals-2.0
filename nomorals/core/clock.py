"""Injectable time.

No logic path in this framework calls ``time.time()`` directly. Decay functions,
backoff schedules, lease expiries and mission deadlines all go through a
:class:`Clock`, which means a test can simulate three months of memory decay in
microseconds and a mission can be replayed deterministically.

:class:`Deadline` is the absolute counterpart of a timeout: compute it once
from the clock, then hand the *remaining* budget down the call chain (outer
deadline ≤ sum of inner + margin). :class:`Backoff` is the retry-delay policy
made testable — it computes the same exponential-backoff-with-jitter delays
as :class:`nomorals.core.tasks.RetryPolicy` but owns the *waiting* part, and it
advances a :class:`FrozenClock` instead of sleeping when one is in play.
"""

from __future__ import annotations

import abc
import random
import threading
import time
from contextlib import contextmanager
from typing import Iterator

__all__ = [
    "Backoff",
    "Clock",
    "Deadline",
    "FrozenClock",
    "MonotonicClock",
    "ScaledClock",
    "SystemClock",
    "Timeout",
    "default_clock",
    "set_default_clock",
]


class Clock(abc.ABC):
    """Time source abstraction."""

    @abc.abstractmethod
    def now(self) -> float:
        """Wall-clock time in unix seconds."""

    @abc.abstractmethod
    def monotonic(self) -> float:
        """Monotonic seconds; never goes backwards, undefined epoch."""

    def now_ms(self) -> int:
        """Wall-clock time in integer milliseconds."""
        return int(self.now() * 1000)

    def iso(self, ts: float | None = None) -> str:
        """UTC ISO-8601 timestamp with second precision."""
        import datetime as _dt

        moment = _dt.datetime.fromtimestamp(
            self.now() if ts is None else ts, _dt.timezone.utc
        )
        return moment.strftime("%Y-%m-%dT%H:%M:%SZ")

    def elapsed_since(self, mark: float) -> float:
        """Seconds elapsed on the monotonic clock since ``mark``."""
        return self.monotonic() - mark

    def sleep(self, seconds: float) -> None:
        """Block for ``seconds`` of *this* clock's time. The base
        implementation sleeps real time; :class:`FrozenClock` advances
        virtual time instead."""
        if seconds > 0:
            time.sleep(seconds)

    def sleep_until(self, ts: float) -> None:
        """Sleep until the monotonic clock reaches ``ts`` (no-op if past)."""
        self.sleep(max(0.0, ts - self.monotonic()))

    def deadline_after(self, seconds: float) -> "Deadline":
        """An absolute deadline ``seconds`` from now on this clock."""
        return Deadline(self.monotonic() + seconds, self)

    def timeout(self, seconds: float) -> "Timeout":
        """Context manager yielding a :class:`Deadline` for the block.

        >>> with clock.timeout(30) as deadline:
        ...     while not deadline.expired():
        ...         work(deadline.remaining())
        """
        return Timeout(self.deadline_after(seconds))

    @contextmanager
    def span(self) -> Iterator["_Span"]:
        """Measure a block; the span exposes ``.elapsed`` on exit."""
        mark = _Span(self)
        mark.start()
        try:
            yield mark
        finally:
            mark.stop()


class _Span:
    __slots__ = ("_clock", "_start", "_end")

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._start = 0.0
        self._end = 0.0

    def start(self) -> None:
        self._start = self._clock.monotonic()

    def stop(self) -> None:
        self._end = self._clock.monotonic()

    @property
    def elapsed(self) -> float:
        end = self._end or self._clock.monotonic()
        return end - self._start


class Deadline:
    """An absolute point on a clock's monotonic timeline.

    Timeouts *relative* to "now" drift when a function calls three nested
    helpers each applying their own timeout. Deadlines don't: compute once
    at the edge, pass ``deadline.remaining()`` down, and every layer shares
    one budget.
    """

    __slots__ = ("at", "clock")

    def __init__(self, at: float, clock: Clock | None = None) -> None:
        self.at = float(at)
        self.clock = clock or SystemClock()

    @classmethod
    def after(cls, seconds: float, clock: Clock | None = None) -> "Deadline":
        clock = clock or SystemClock()
        return cls(clock.monotonic() + seconds, clock)

    @classmethod
    def never(cls, clock: Clock | None = None) -> "Deadline":
        """A deadline that never expires — for code paths where the budget
        is optional."""
        return cls(float("inf"), clock)

    def remaining(self) -> float:
        """Seconds left; 0.0 when expired."""
        return max(0.0, self.at - self.clock.monotonic())

    def expired(self) -> bool:
        return self.clock.monotonic() >= self.at

    def child(self, seconds: float) -> "Deadline":
        """A sub-deadline that also respects this one: the earlier of the
        two wins, so an inner layer can never outlive its parent."""
        return Deadline(min(self.at, self.clock.monotonic() + seconds), self.clock)

    def raise_if_expired(self, message: str = "deadline exceeded") -> None:
        if self.expired():
            raise TimeoutError(message)

    def __bool__(self) -> bool:
        return not self.expired()

    def __repr__(self) -> str:
        return f"Deadline(remaining={self.remaining():.3f}s)"


class Timeout:
    """Context manager wrapper so ``with clock.timeout(30) as dl:`` reads
    naturally. Yields the :class:`Deadline`."""

    __slots__ = ("deadline",)

    def __init__(self, deadline: Deadline) -> None:
        self.deadline = deadline

    def __enter__(self) -> Deadline:
        return self.deadline

    def __exit__(self, *exc: object) -> None:
        return None


class Backoff:
    """Exponential backoff with full jitter, on an injectable clock.

    >>> clock = FrozenClock()
    >>> backoff = Backoff(base=1.0, clock=clock, rng=random.Random(42))
    >>> backoff.wait()          # advances the frozen clock instead of sleeping
    >>> backoff.attempt
    1

    ``wait()`` on a real clock sleeps; on a :class:`FrozenClock` it advances
    virtual time, so retry loops are unit-testable at full speed.
    """

    __slots__ = ("base", "factor", "max_delay", "jitter", "clock", "rng", "attempt", "_lock")

    def __init__(
        self,
        base: float = 1.0,
        factor: float = 2.0,
        max_delay: float = 300.0,
        jitter: bool = True,
        clock: Clock | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.base = float(base)
        self.factor = float(factor)
        self.max_delay = float(max_delay)
        self.jitter = jitter
        self.clock = clock or SystemClock()
        self.rng = rng or random.Random()
        self.attempt = 0
        self._lock = threading.Lock()

    def delay_for(self, attempt: int | None = None) -> float:
        """The delay the *next* wait would use (does not consume it)."""
        n = self.attempt if attempt is None else attempt
        delay = min(self.base * (self.factor ** max(0, n)), self.max_delay)
        if self.jitter:
            delay = self.rng.uniform(0, delay)
        return delay

    def wait(self) -> float:
        """Sleep the backoff delay for the current attempt, then advance the
        attempt counter. Returns the delay that was waited."""
        with self._lock:
            delay = self.delay_for()
            self.attempt += 1
        self.clock.sleep(delay)
        return delay

    def reset(self) -> None:
        with self._lock:
            self.attempt = 0


class SystemClock(Clock):
    """Real time. The default in production."""

    __slots__ = ()

    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class MonotonicClock(SystemClock):
    """Alias kept for readability at call sites that only need monotonic time."""

    __slots__ = ()


class FrozenClock(Clock):
    """A clock the test controls explicitly.

        >>> clock = FrozenClock(1000.0)
        >>> clock.now()
        1000.0
        >>> clock.advance(60)
        1060.0
    """

    __slots__ = ("_now", "_mono", "_lock")

    def __init__(self, start: float = 0.0) -> None:
        self._now = float(start)
        self._mono = float(start)
        self._lock = threading.Lock()

    def now(self) -> float:
        with self._lock:
            return self._now

    def monotonic(self) -> float:
        with self._lock:
            return self._mono

    def set(self, value: float) -> float:
        with self._lock:
            self._now = float(value)
            self._mono = float(value)
            return self._now

    def advance(self, seconds: float) -> float:
        with self._lock:
            self._now += seconds
            self._mono += seconds
            return self._now

    def sleep(self, seconds: float) -> float:
        """Advances virtual time instead of blocking."""
        return self.advance(seconds)


class ScaledClock(Clock):
    """Real clock with a multiplier, for load-testing decay/schedule behaviour.

    ``ScaledClock(factor=60)`` makes one real second look like one minute.
    """

    __slots__ = ("_factor", "_epoch_real", "_epoch_virtual", "_lock", "_base")

    def __init__(self, factor: float = 1.0, base: Clock | None = None) -> None:
        self._factor = float(factor)
        self._base = base or SystemClock()
        self._epoch_real = self._base.monotonic()
        self._epoch_virtual = self._base.now()
        self._lock = threading.Lock()

    def now(self) -> float:
        with self._lock:
            real = self._base.monotonic() - self._epoch_real
            return self._epoch_virtual + real * self._factor

    def monotonic(self) -> float:
        with self._lock:
            return (self._base.monotonic() - self._epoch_real) * self._factor

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self._base.sleep(seconds / self._factor)


#: Module-level default. Swap it in tests with :func:`set_default_clock`.
_default_clock: Clock = SystemClock()


def default_clock() -> Clock:
    return _default_clock


def set_default_clock(clock: Clock) -> None:
    global _default_clock
    _default_clock = clock
