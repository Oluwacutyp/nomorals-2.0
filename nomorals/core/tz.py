"""Timezone resolution that never crashes.

On minimal installs (notably Termux without the ``tzdata`` package) the
``zoneinfo`` database can be entirely absent — even ``ZoneInfo("UTC")``
raises ``ZoneInfoNotFoundError("No time zone found with key UTC")``.
Every caller that touches a timezone must go through :func:`safe_zoneinfo`
so a missing tz database degrades to a fixed UTC offset instead of
crashing the scheduler, the briefing, or quiet-hours checks.
"""

from __future__ import annotations

from datetime import tzinfo

__all__ = ["safe_zoneinfo", "utc_fallback"]


def utc_fallback() -> tzinfo:
    """A tzinfo that is always available: fixed UTC, no tzdata needed."""
    from datetime import timezone

    return timezone.utc


def safe_zoneinfo(name: str | None) -> tzinfo:
    """Return a tzinfo for ``name``; never raises.

    Order: ``ZoneInfo(name)`` → ``ZoneInfo("UTC")`` → ``timezone.utc``.
    An empty/blank name goes straight to the UTC fallback chain.
    """
    if name:
        try:
            from zoneinfo import ZoneInfo

            return ZoneInfo(str(name))
        except Exception:  # noqa: BLE001 - bad name or missing tzdata
            pass
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo("UTC")
    except Exception:  # noqa: BLE001 - tzdata entirely absent (Termux)
        pass
    return utc_fallback()
