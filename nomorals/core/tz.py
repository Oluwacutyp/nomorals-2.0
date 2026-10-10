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

__all__ = ["safe_zoneinfo", "utc_fallback", "now_in", "format_ts",
           "parse_ts", "to_utc"]


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


def now_in(name: str | None) -> "datetime":
    """Current time in ``name``'s zone (never raises — falls back to UTC)."""
    from datetime import datetime

    return datetime.now(safe_zoneinfo(name))


def to_utc(dt: "datetime") -> "datetime":
    """Convert an aware datetime to UTC; naive is assumed to be UTC."""
    from datetime import datetime, timezone

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def format_ts(ts: float | None = None, *, tz: str | None = None,
              fmt: str = "%Y-%m-%d %H:%M %Z") -> str:
    """Format a unix timestamp (default: now) in ``tz`` (default: UTC).

    ``fmt`` accepts strftime directives; ``%Z`` shows the zone abbreviation.
    """
    from datetime import datetime, timezone

    dt = datetime.fromtimestamp(ts if ts is not None else
                                __import__("time").time(),
                                tz=timezone.utc)
    if tz:
        dt = dt.astimezone(safe_zoneinfo(tz))
    return dt.strftime(fmt)


def parse_ts(text: str, *, tz: str | None = None) -> float:
    """Parse common timestamp spellings to unix seconds.

    Accepts ISO 8601 (with or without offset), ``YYYY-MM-DD HH:MM``,
    and plain epoch numbers. Naive inputs are interpreted in ``tz``
    (default UTC). Raises ValueError on garbage.
    """
    from datetime import datetime, timezone

    s = (text or "").strip()
    if not s:
        raise ValueError("empty timestamp")
    if s.lstrip("+-").replace(".", "", 1).isdigit():
        return float(s)
    iso = s.replace("Z", "+00:00") if s.endswith(("Z", "z")) else s
    dt = None
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
                    "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M",
                    "%d %b %Y %H:%M", "%d %b %Y"):
            try:
                dt = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
    if dt is None:
        raise ValueError(f"unparseable timestamp {text!r}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=safe_zoneinfo(tz))
    return dt.timestamp()
