"""Owner rhythm — when the owner wants to be left alone, and how hard
to push when they don't.

Three knobs, all enforced in code (never in a prompt):

* **Quiet hours** — a daily window (default 22:00–07:00) during which
  ambient / background / serendipity firings are *held*, not sent.
  Safety and owner-direct firings always pass — a security alert at
  3am is not serendipity. Scheduled work also passes: an overnight
  backup is not a ping.
* **Intensity** — ``low`` / ``normal`` / ``high``. ``low`` means only
  scheduled-and-above fires (ambient/background defer); ``normal`` is
  the everyday setting; ``high`` lets ambient through even while busy
  (the owner asked for full power), but quiet hours still hold
  serendipity.
* **Busy flag** — set while the owner is in a meeting / focused / on a
  call. Ambient/background defer; scheduled, owner-direct, and safety
  still fire.

Serendipity held during quiet hours is not dropped: :meth:`hold_serendipity`
queues it, and :meth:`release_at_morning` returns the held items once
quiet hours end, so the morning digest can carry them ("held till
morning", per the spec — the presence digest is the delivery vehicle).

``now`` may be a ``datetime``, an epoch timestamp, or ``None`` (now).
Naive datetimes are read as owner-local time.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Callable

__all__ = ["OwnerRhythm", "INTENSITIES"]

INTENSITIES = ("low", "normal", "high")

#: Priorities that quiet hours hold (everything else passes).
_QUIET_HELD = frozenset({"ambient", "background", "serendipity"})
#: Priorities that the busy flag holds.
_BUSY_HELD = frozenset({"ambient", "background", "serendipity"})
#: Priorities that low intensity holds.
_LOW_HELD = frozenset({"ambient", "background", "serendipity"})
#: Priorities that always fire, whatever the rhythm says.
_ALWAYS = frozenset({"safety", "owner-direct", "owner_direct", "urgent",
                     "critical"})


def _hour_of(now: Any, clock: Callable[[], float]) -> int:
    if now is None:
        return datetime.fromtimestamp(clock()).hour
    if isinstance(now, datetime):
        return now.hour
    if isinstance(now, (int, float)):
        return datetime.fromtimestamp(float(now)).hour
    ts = getattr(now, "timestamp", None)
    if callable(ts):
        return datetime.fromtimestamp(float(ts())).hour
    return datetime.fromtimestamp(float(now)).hour


class OwnerRhythm:
    """Quiet hours + intensity + busy flag → should this fire now?"""

    def __init__(self, *, quiet_start: int = 22, quiet_end: int = 7,
                 intensity: str = "normal", busy: bool = False,
                 clock: Callable[[], float] | None = None) -> None:
        self.set_quiet_hours(quiet_start, quiet_end)
        self.set_intensity(intensity)
        self.busy = bool(busy)
        self._clock = clock or time.time
        #: Serendipity items held during quiet hours, oldest first.
        self._held: list[dict[str, Any]] = []

    # ── configuration ────────────────────────────────────────────

    def set_quiet_hours(self, start: int, end: int) -> None:
        if not (0 <= int(start) <= 23 and 0 <= int(end) <= 23):
            raise ValueError("quiet hours must be hours 0–23")
        self.quiet_start = int(start)
        self.quiet_end = int(end)

    def set_intensity(self, intensity: str) -> str:
        level = str(intensity or "normal").lower()
        if level not in INTENSITIES:
            raise ValueError(
                f"unknown intensity {intensity!r} — one of: {INTENSITIES}")
        self.intensity = level
        return level

    def set_busy(self, busy: bool) -> bool:
        self.busy = bool(busy)
        return self.busy

    # ── queries ──────────────────────────────────────────────────

    def is_quiet(self, now: Any = None) -> bool:
        """True when ``now`` falls inside the quiet-hours window."""
        if self.quiet_start == self.quiet_end:
            return False  # start == end means "no quiet hours"
        hour = _hour_of(now, self._clock)
        if self.quiet_start < self.quiet_end:
            return self.quiet_start <= hour < self.quiet_end
        return hour >= self.quiet_start or hour < self.quiet_end

    def should_fire(self, now: Any = None, *,
                    priority: str = "ambient", kind: str = "",
                    force: bool = False) -> bool:
        """Should a firing with this priority go out now?

        ``priority``: safety / owner-direct / scheduled / ambient /
        background (``kind="serendipity"`` counts as ambient). ``force``
        bypasses rhythm (owner-triggered surfaces only).
        """
        if force:
            return True
        prio = str(priority or "ambient").lower()
        if prio in _ALWAYS:
            return True
        if str(kind or "").lower() == "serendipity":
            prio = "serendipity"
        if prio in _QUIET_HELD and self.is_quiet(now):
            return False
        if prio in _BUSY_HELD and self.busy and self.intensity != "high":
            return False
        if prio in _LOW_HELD and self.intensity == "low":
            return False
        return True

    def status(self, now: Any = None) -> dict[str, Any]:
        return {
            "quiet": self.is_quiet(now),
            "quiet_hours": [self.quiet_start, self.quiet_end],
            "intensity": self.intensity,
            "busy": self.busy,
            "held_serendipity": len(self._held),
        }

    # ── held serendipity ─────────────────────────────────────────

    def hold_serendipity(self, item: Any) -> dict[str, Any]:
        """Queue a serendipity item held for the morning. Returns the
        queued entry."""
        entry = {"item": item, "held_at": self._clock()}
        self._held.append(entry)
        return dict(entry)

    def held_serendipity(self) -> list[dict[str, Any]]:
        return [dict(e) for e in self._held]

    def release_at_morning(self, now: Any = None) -> list[dict[str, Any]]:
        """If quiet hours have ended, release everything held and clear
        the queue. During quiet hours returns [] and keeps holding."""
        if self.is_quiet(now):
            return []
        held, self._held = self._held, []
        return [dict(e) for e in held]
