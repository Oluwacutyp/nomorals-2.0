"""Frozen clock for tests — deterministic "now", no wall-clock reads.

Tests must never read the wall clock directly (``time.time()``,
``datetime.now()`` / ``datetime.today()``): a test that passes at noon
fails at 3am (see the money-guard "unusual time" signal, which fires
23:00–05:00 local). Instead, tests take their "now" from here and pass
it explicitly (``now=`` / ``ts=``) to the code under test.

The frozen instant is noon local on a FIXED date — a ``datetime()``
literal, not a clock read — so the hour is always safely outside the
guard's unusual-hours window in every timezone, and the suite is
repeatable on any date. No function in this module reads the clock.

Two escape hatches, both grandfathered and documented where used:

* Production code that reads the real clock internally (idle monitor,
  weakness scan) cannot take a frozen "now" — those tests keep a real
  ``time.time()`` and are recorded in the ``test_no_direct_clock``
  ratchet baseline with a comment saying why.
* ``FrozenClock.advance()`` lets a single test move time forward
  (cooldowns, expiries) without touching the wall clock.
"""

from __future__ import annotations

import datetime

import pytest

__all__ = [
    "FrozenClock",
    "FrozenClockMixin",
    "MIDDAY",
    "MIDDAY_TS",
    "frozen_clock",
]

#: Noon local on a fixed date — the frozen "now" for every test.
MIDDAY = datetime.datetime(2026, 10, 14, 12, 0, 0)

#: ``MIDDAY`` as a timestamp (local-time conversion; not a clock read).
MIDDAY_TS = MIDDAY.timestamp()


class FrozenClock:
    """A clock pinned to one instant, movable by tests."""

    def __init__(self, ts: float = MIDDAY_TS) -> None:
        self._ts = float(ts)

    def now(self) -> float:
        """The frozen instant as a timestamp."""
        return self._ts

    def now_dt(self) -> datetime.datetime:
        """The frozen instant as a local datetime."""
        return datetime.datetime.fromtimestamp(self._ts)

    def advance(self, seconds: float) -> float:
        """Move the frozen instant forward; returns the new ``now()``."""
        self._ts += float(seconds)
        return self._ts

    def set(self, ts: float) -> float:
        """Pin the clock to a new instant; returns the new ``now()``."""
        self._ts = float(ts)
        return self._ts


@pytest.fixture
def frozen_clock() -> FrozenClock:
    """A fresh frozen clock (noon local, fixed date) per test."""
    return FrozenClock()


class FrozenClockMixin:
    """Unittest mixin: ``self.clock`` / ``self.now`` in every test.

    Mix it into a ``unittest.TestCase`` and call ``super().setUp()`` —
    the mixin cooperates through the MRO.
    """

    def setUp(self) -> None:
        super().setUp()  # type: ignore[misc] — MRO cooperative
        self.clock = FrozenClock()
        self.now = self.clock.now()
