"""Fever — a bounded, announced self-throttle.

When the immune system suspects something is wrong with itself, it
runs a fever: it throttles its own actions for a bounded time while
the suspicion is looked at. A fever is:

* **Never silent.** ``start`` without an ``announce_fn`` raises
  :class:`ValueError`. The announce message is fixed in form:
  ``"I've throttled myself while I look at something: {reason}, max {max}s"``.
* **Bounded.** ``max_seconds`` is capped at :data:`FEVER_MAX_SECONDS`
  (3600s). A fever always ends.
* **Honest.** :meth:`Fever.end` returns a :class:`FeverReport` with an
  honest note about what the fever did and did NOT resolve — a fever
  throttles; it does not diagnose or fix.

``throttle()`` increments the counter only while the fever is active;
after it ends it is a no-op.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

__all__ = ["FEVER_MAX_SECONDS", "FeverReport", "Fever"]

FEVER_MAX_SECONDS = 3600


@dataclass
class FeverReport:
    started: float
    ended: float
    reason: str
    resolved: bool
    actions_throttled: int
    note: str = ""


class Fever:
    def __init__(self, announce_fn: Callable[[str], None] | None = None) -> None:
        self.announce_fn = announce_fn
        self.active: bool = False
        self.started_ts: Optional[float] = None
        self.max_seconds: float = 0.0
        self.reason: str = ""
        self.actions_throttled: int = 0

    def start(
        self,
        reason: str,
        max_seconds: float,
        announce_fn: Callable[[str], None] | None = None,
        *,
        now: float | None = None,
    ) -> str:
        """Start a fever. Announces immediately; raises if no announcer."""
        fn = announce_fn if announce_fn is not None else self.announce_fn
        if fn is None:
            raise ValueError(
                "fever must never be silent — start requires an announce_fn"
            )
        if max_seconds is None or max_seconds <= 0:
            raise ValueError("fever max_seconds must be positive")
        capped = min(float(max_seconds), float(FEVER_MAX_SECONDS))
        self.started_ts = now if now is not None else time.time()
        self.max_seconds = capped
        self.reason = str(reason)
        self.actions_throttled = 0
        self.active = True
        message = (
            f"I've throttled myself while I look at something: {reason}, "
            f"max {int(capped) if float(capped).is_integer() else capped}s"
        )
        fn(message)
        return message

    def is_active(self, now: float | None = None) -> bool:
        if not self.active or self.started_ts is None:
            return False
        current = now if now is not None else time.time()
        return current < self.started_ts + self.max_seconds

    def remaining(self, now: float | None = None) -> float:
        if not self.is_active(now):
            return 0.0
        current = now if now is not None else time.time()
        assert self.started_ts is not None
        return max(0.0, (self.started_ts + self.max_seconds) - current)

    def throttle(self, now: float | None = None) -> bool:
        """Count one throttled action if the fever is active, else no-op."""
        if self.is_active(now):
            self.actions_throttled += 1
            return True
        return False

    def end(self, *, now: float | None = None, resolved: bool = False) -> FeverReport:
        ended = now if now is not None else time.time()
        started = self.started_ts if self.started_ts is not None else ended
        elapsed = max(0.0, ended - started)
        throttled = self.actions_throttled
        reason = self.reason
        self.active = False
        note = (
            f"Fever throttled {throttled} action(s) over {elapsed:.1f}s while "
            f"looking at: {reason}. It did NOT resolve the underlying issue by "
            f"itself — throttling only bought time to look. "
            f"resolved={resolved}."
        )
        return FeverReport(
            started=started, ended=ended, reason=reason,
            resolved=bool(resolved), actions_throttled=throttled, note=note,
        )
