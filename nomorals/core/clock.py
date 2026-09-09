"""Injectable time.

No logic path in this framework calls ``time.time()`` directly. Decay functions,
backoff schedules, lease expiries and mission deadlines all go through a
:class:`Clock`, which means a test can simulate three months of memory decay in
microseconds and a mission can be replayed deterministically.
"""

from __future__ import annotations

import abc
import threading
import time
from contextlib import contextmanager
from typing import Iterator

__all__ = ["Clock", "FrozenClock", "MonotonicClock", "ScaledClock", "SystemClock"]


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

    __slots__ = ("_factor", "_epoch_real", "_epoch_virtual", "_lock")

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
