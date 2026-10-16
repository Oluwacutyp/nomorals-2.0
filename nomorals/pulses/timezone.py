"""Pulse timezone anchor (GROUPS Tier 0).

The legacy pulses were anchored to ``America/Denver`` (the phone's
timezone): morning 23:00, afternoon 13:00, evening 21:00 Denver.
The owner actually lives in Lagos (``Africa/Lagos``, WAT = UTC+1, no
DST). Read in owner time, the legacy schedule is:

* morning  23:00 Denver  ->  06:00 next day Lagos
* afternoon 13:00 Denver ->  20:00 same day Lagos
* evening  21:00 Denver  ->  04:00 next day Lagos  <-- flagged

An "evening wrap" firing at 04:00 owner-local is plainly unintentional,
and it *drifts*: when Denver leaves DST (MDT UTC-6 -> MST UTC-7, first
Sunday in November) the same 21:00 Denver wall time fires 1h later in
UTC, so it lands at 05:00 Lagos instead of 04:00. A fixed UTC offset (``tz_offset``) cannot fix
this — it bakes in one side of the DST transition and is wrong for
half the year.

This module anchors every pulse to the owner's real timezone:

* :class:`PulseAnchor` — owner tz (configurable, default
  ``Africa/Lagos``), IANA resolution with a fixed-offset fallback for
  tzdata-less installs (Termux without the ``tzdata`` package).
* :func:`fire_at_local` — wall-clock time in a zone -> aware UTC
  datetime. DST gaps (spring-forward nonexistent times) shift forward
  to the first real instant; DST folds (fall-back ambiguous times)
  take the first occurrence (``fold=0``). Never raises.
* :func:`confirm_routine` — the current pulse schedule rendered in
  owner-local times, with every suspicious slot flagged for the owner
  to confirm. The evening 04:00 slot is flagged ``needs_confirmation``
  until the owner explicitly approves it (or the anchor switches to
  the proposed owner-local schedule).
* :func:`drift_report` — proves the DST drift across a transition:
  same Denver wall time, different owner-local time.

No pulse fires from this module — it only computes *when*. The
scheduler / pulse runners own firing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, time as _time, timedelta, timezone, tzinfo
from typing import Any

__all__ = [
    "OWNER_TZ_DEFAULT",
    "LAGOS_FIXED_OFFSET_MINUTES",
    "LEGACY_DENVER_SCHEDULE",
    "PROPOSED_OWNER_SCHEDULE",
    "NIGHT_HOURS",
    "FixedOffset",
    "PulseAnchor",
    "resolve_owner_tz",
    "fire_at_local",
    "confirm_routine",
    "drift_report",
]

#: Default owner timezone — the owner lives in Lagos, Nigeria.
OWNER_TZ_DEFAULT = "Africa/Lagos"

#: WAT is UTC+1 year-round (Nigeria has no DST). Fixed-offset fallback.
LAGOS_FIXED_OFFSET_MINUTES = 60

#: Hours that are "night" for a routine — a briefing here is suspect.
NIGHT_HOURS = set(range(0, 5))  # 00:00–04:59

#: Legacy schedule: (wall time, anchor zone) as shipped.
LEGACY_DENVER_SCHEDULE: dict[str, tuple[str, str]] = {
    "morning": ("23:00", "America/Denver"),
    "afternoon": ("13:00", "America/Denver"),
    "evening": ("21:00", "America/Denver"),
}

#: Proposed owner-anchored schedule (needs owner confirmation to adopt).
PROPOSED_OWNER_SCHEDULE: dict[str, str] = {
    "morning": "06:00",
    "afternoon": "13:00",
    "evening": "19:00",
}


class FixedOffset(tzinfo):
    """Fixed UTC-offset tzinfo — the tzdata-less fallback.

    Used when ``zoneinfo`` cannot resolve a name (Termux without the
    ``tzdata`` package). Carries no DST, so it is exact for zones like
    Africa/Lagos and a documented approximation for DST zones.
    """

    def __init__(self, offset_minutes: int, name: str = "fixed") -> None:
        self._offset = timedelta(minutes=int(offset_minutes))
        self._name = name

    def utcoffset(self, dt: datetime | None) -> timedelta:
        return self._offset

    def tzname(self, dt: datetime | None) -> str:
        return self._name

    def dst(self, dt: datetime | None) -> timedelta:
        return timedelta(0)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"FixedOffset({self._offset}, {self._name!r})"


def _try_zoneinfo(name: str) -> tzinfo | None:
    """ZoneInfo for ``name``, or None when unavailable/invalid."""
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:  # noqa: BLE001 - missing tzdata or bad name
        return None


def resolve_owner_tz(config: dict[str, Any] | Any | None = None,
                     fallback_offset_minutes: int = LAGOS_FIXED_OFFSET_MINUTES
                     ) -> tuple[str, tzinfo, str]:
    """Resolve the owner's timezone -> (name, tzinfo, source).

    Precedence: explicit ``config`` (dict keys ``owner_timezone`` /
    ``timezone`` / ``pulse_timezone``, or an object with those attrs)
    > ``NM_OWNER_TZ`` env > ``NM_PULSE_TIMEZONE`` env >
    ``Africa/Lagos`` default. An unresolvable name falls back to a
    fixed offset (Lagos +1:00) and reports source ``fixed-offset`` so
    callers know DST is not modelled.
    """
    name = ""
    source = ""
    if isinstance(config, dict):
        for key in ("owner_timezone", "timezone", "pulse_timezone"):
            if config.get(key):
                name, source = str(config[key]), f"config:{key}"
                break
    elif config is not None:
        for key in ("owner_timezone", "timezone", "pulse_timezone"):
            val = getattr(config, key, None)
            if val:
                name, source = str(val), f"config:{key}"
                break
    if not name:
        env = (os.environ.get("NM_OWNER_TZ") or "").strip()
        if env:
            name, source = env, "env:NM_OWNER_TZ"
    if not name:
        env = (os.environ.get("NM_PULSE_TIMEZONE") or "").strip()
        if env:
            name, source = env, "env:NM_PULSE_TIMEZONE"
    if not name:
        name, source = OWNER_TZ_DEFAULT, "default:Africa/Lagos"
    tz = _try_zoneinfo(name)
    if tz is not None:
        return name, tz, source
    # tzdata missing or bad name: fixed-offset fallback, never crash.
    return name, FixedOffset(fallback_offset_minutes,
                             f"{name} (fixed)"), f"{source}+fixed-offset"


def _parse_hhmm(hhmm: str) -> tuple[int, int]:
    parts = str(hhmm or "").strip().split(":")
    try:
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        hour, minute = 0, 0
    return max(0, min(23, hour)), max(0, min(59, minute))


def _coerce_date(on_date: date | str | None,
                 tz: tzinfo) -> date:
    if isinstance(on_date, date):
        return on_date
    if isinstance(on_date, str) and on_date.strip():
        try:
            return date.fromisoformat(on_date.strip()[:10])
        except ValueError:
            pass
    return datetime.now(tz).date()


def fire_at_local(hour: int, tz: str | tzinfo,
                  minute: int = 0,
                  on_date: date | str | None = None,
                  now: datetime | None = None) -> datetime:
    """Wall-clock ``hour:minute`` in ``tz`` on ``on_date`` -> UTC instant.

    ``tz`` is an IANA name or a tzinfo. ``on_date`` defaults to "today
    in ``tz``" (derived from ``now`` when given, else the real clock).
    DST gaps shift forward to the first real instant after the gap;
    DST folds take the first occurrence (``fold=0``). Never raises —
    on garbage input returns midnight UTC of the requested date.
    """
    try:
        tzinfo_obj: tzinfo
        if isinstance(tz, tzinfo):
            tzinfo_obj = tz
        elif isinstance(tz, str) and tz.strip():
            resolved = _try_zoneinfo(tz.strip())
            tzinfo_obj = resolved if resolved is not None else FixedOffset(
                LAGOS_FIXED_OFFSET_MINUTES, f"{tz} (fixed)")
        else:
            tzinfo_obj = _try_zoneinfo(OWNER_TZ_DEFAULT) or FixedOffset(
                LAGOS_FIXED_OFFSET_MINUTES, f"{OWNER_TZ_DEFAULT} (fixed)")
        day = _coerce_date(on_date, tzinfo_obj)
        if now is not None and on_date is None:
            day = now.astimezone(tzinfo_obj).date()
        hour = max(0, min(23, int(hour)))
        minute = max(0, min(59, int(minute)))
        naive = datetime.combine(day, _time(hour, minute))
        # First try: fold=0 (first occurrence for ambiguous times).
        candidate = naive.replace(tzinfo=tzinfo_obj, fold=0)
        if candidate.astimezone(timezone.utc).astimezone(
                tzinfo_obj).replace(tzinfo=None) == naive:
            return candidate.astimezone(timezone.utc)
        # fold=1 (second occurrence) — also round-trips for ambiguous.
        candidate = naive.replace(tzinfo=tzinfo_obj, fold=1)
        if candidate.astimezone(timezone.utc).astimezone(
                tzinfo_obj).replace(tzinfo=None) == naive:
            return candidate.astimezone(timezone.utc)
        # Nonexistent (spring-forward gap): walk forward minute by minute
        # to the first real wall time — at most ~3h for any real zone.
        probe = naive
        for _ in range(240):
            probe += timedelta(minutes=1)
            cand = probe.replace(tzinfo=tzinfo_obj, fold=0)
            if cand.astimezone(timezone.utc).astimezone(
                    tzinfo_obj).replace(tzinfo=None) == probe:
                return cand.astimezone(timezone.utc)
        return naive.replace(tzinfo=timezone.utc)
    except Exception:  # noqa: BLE001 - never raise from time math
        try:
            day = _coerce_date(on_date, timezone.utc)
        except Exception:  # noqa: BLE001
            day = datetime.now(timezone.utc).date()
        return datetime.combine(day, _time(0, 0),
                                tzinfo=timezone.utc)


@dataclass
class PulseAnchor:
    """The owner's timezone anchor for all pulses.

    ``owner_tz`` is configurable (default Africa/Lagos). ``fire()``
    computes UTC instants from owner-local wall times; ``confirm()``
    renders the live schedule for the owner to approve.
    """

    owner_tz: str = OWNER_TZ_DEFAULT
    fallback_offset_minutes: int = LAGOS_FIXED_OFFSET_MINUTES
    _tzinfo: tzinfo | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        name, tz, _src = resolve_owner_tz(
            {"owner_timezone": self.owner_tz},
            fallback_offset_minutes=self.fallback_offset_minutes)
        self.owner_tz = name
        self._tzinfo = tz

    @property
    def tzinfo(self) -> tzinfo:
        assert self._tzinfo is not None
        return self._tzinfo

    def fire(self, hour: int, minute: int = 0,
             on_date: date | str | None = None,
             now: datetime | None = None) -> datetime:
        """Owner-local ``hour:minute`` -> UTC instant (DST-safe)."""
        return fire_at_local(hour, self.tzinfo, minute=minute,
                             on_date=on_date, now=now)

    def fire_slot(self, slot: str,
                  on_date: date | str | None = None,
                  schedule: dict[str, str] | None = None) -> datetime:
        """UTC instant for a named slot under ``schedule`` (owner tz)."""
        sched = schedule or PROPOSED_OWNER_SCHEDULE
        hhmm = sched.get((slot or "").strip().lower(), "06:00")
        hour, minute = _parse_hhmm(hhmm)
        return self.fire(hour, minute, on_date=on_date)

    def schedule_for(self, on_date: date | str | None = None,
                     schedule: dict[str, str] | None = None
                     ) -> dict[str, dict[str, Any]]:
        """Every slot's fire time for one date, in UTC + owner-local."""
        sched = schedule or PROPOSED_OWNER_SCHEDULE
        out: dict[str, dict[str, Any]] = {}
        for slot, hhmm in sched.items():
            hour, minute = _parse_hhmm(hhmm)
            utc_dt = self.fire(hour, minute, on_date=on_date)
            local = utc_dt.astimezone(self.tzinfo)
            out[slot] = {
                "wall": hhmm,
                "utc": utc_dt.isoformat(),
                "owner_local": local.strftime("%Y-%m-%d %H:%M %Z"),
                "owner_hour": local.hour,
                "day_shift": (local.date()
                              - _coerce_date(on_date, self.tzinfo)).days,
            }
        return out

    def confirm(self, on_date: date | str | None = None) -> dict[str, Any]:
        """Alias for :func:`confirm_routine` bound to this anchor."""
        return confirm_routine(self, on_date=on_date)


def _legacy_in_owner_tz(anchor: PulseAnchor,
                        on_date: date | str | None = None
                        ) -> dict[str, dict[str, Any]]:
    """Legacy Denver-anchored slots rendered in owner-local time."""
    out: dict[str, dict[str, Any]] = {}
    for slot, (hhmm, zone) in LEGACY_DENVER_SCHEDULE.items():
        hour, minute = _parse_hhmm(hhmm)
        utc_dt = fire_at_local(hour, zone, minute=minute, on_date=on_date)
        # NOTE: on_date is interpreted in the *anchor* zone for the
        # legacy rows too — the Denver fire that lands on the owner's
        # date. The day_shift field makes the mapping explicit.
        local = utc_dt.astimezone(anchor.tzinfo)
        out[slot] = {
            "legacy_wall": f"{hhmm} {zone}",
            "utc": utc_dt.isoformat(),
            "owner_local": local.strftime("%Y-%m-%d %H:%M %Z"),
            "owner_hour": local.hour,
            "owner_date": local.date().isoformat(),
        }
    return out


def confirm_routine(anchor: PulseAnchor | None = None,
                    on_date: date | str | None = None) -> dict[str, Any]:
    """Current pulse schedule in owner-local times, for owner approval.

    Returns the legacy (Denver-anchored) schedule as it actually lands
    in the owner's timezone, the proposed owner-anchored schedule, and
    a ``flags`` list. The evening slot landing at 04:00 owner-local is
    flagged ``needs_confirmation`` — it is treated as unintentional
    until the owner explicitly confirms it. Any routine in night hours
    (00:00–04:59 owner-local) gets the same treatment.

    The caller (chat/CLI) presents this to the owner; nothing here
    flips a schedule — adoption needs a separate explicit confirmation.
    """
    anchor = anchor or PulseAnchor()
    legacy = _legacy_in_owner_tz(anchor, on_date=on_date)
    proposed = anchor.schedule_for(on_date=on_date)
    flags: list[dict[str, Any]] = []
    for slot, row in legacy.items():
        owner_hour = int(row["owner_hour"])
        if slot == "evening" and owner_hour == 4:
            flags.append({
                "slot": slot,
                "severity": "needs_confirmation",
                "status": "unconfirmed",
                "reason": (
                    "evening wrap lands at 04:00 owner-local "
                    f"({row['legacy_wall']} -> {row['owner_local']}). "
                    "04:00 is a night hour, not evening — this looks "
                    "unintentional (Denver anchor, not owner anchor). "
                    "Confirm explicitly to keep it, or adopt the "
                    "proposed 19:00 owner-local evening slot."
                ),
                "legacy": row,
                "proposed_wall": PROPOSED_OWNER_SCHEDULE.get(slot),
            })
        elif owner_hour in NIGHT_HOURS:
            flags.append({
                "slot": slot,
                "severity": "needs_confirmation",
                "status": "unconfirmed",
                "reason": (
                    f"{slot} lands at {owner_hour:02d}:00 owner-local "
                    f"({row['legacy_wall']} -> {row['owner_local']}), "
                    "inside night hours (00:00–04:59). Confirm "
                    "explicitly to keep it."
                ),
                "legacy": row,
                "proposed_wall": PROPOSED_OWNER_SCHEDULE.get(slot),
            })
    return {
        "owner_tz": anchor.owner_tz,
        "as_of": (on_date if isinstance(on_date, str)
                  else datetime.now(anchor.tzinfo).date().isoformat()),
        "legacy_denver_anchored": legacy,
        "proposed_owner_anchored": proposed,
        "flags": flags,
        "needs_confirmation": bool(flags),
        "confirm_prompt": (
            "Reply CONFIRM to adopt the proposed owner-anchored "
            "schedule (morning 06:00, afternoon 13:00, evening 19:00 "
            f"{anchor.owner_tz}), or name a slot + time to override."
            if flags else
            "Schedule looks sane in owner-local time — no confirmation "
            "needed."
        ),
    }


def drift_report(anchor: PulseAnchor | None = None,
                 before: str = "2026-10-20",
                 after: str = "2026-11-10") -> dict[str, Any]:
    """Show the DST drift: same Denver wall time, different owner time.

    ``before`` is in MDT (UTC-6), ``after`` in MST (UTC-7) — Denver
    falls back 2026-11-01. Lagos (no DST) is the fixed reference, so
    every legacy slot shifts +1h owner-local across the transition.
    """
    anchor = anchor or PulseAnchor()
    rows: dict[str, dict[str, Any]] = {}
    for slot, (hhmm, zone) in LEGACY_DENVER_SCHEDULE.items():
        hour, minute = _parse_hhmm(hhmm)
        utc_before = fire_at_local(hour, zone, minute=minute,
                                   on_date=before)
        utc_after = fire_at_local(hour, zone, minute=minute, on_date=after)
        loc_before = utc_before.astimezone(anchor.tzinfo)
        loc_after = utc_after.astimezone(anchor.tzinfo)
        rows[slot] = {
            "legacy_wall": f"{hhmm} {zone}",
            "before": loc_before.strftime("%Y-%m-%d %H:%M %Z"),
            "after": loc_after.strftime("%Y-%m-%d %H:%M %Z"),
            "shift_minutes": (
                (loc_after.hour * 60 + loc_after.minute)
                - (loc_before.hour * 60 + loc_before.minute)
            ),
            "utc_shift_minutes": int(
                (utc_after - utc_before).total_seconds() // 60) % (24 * 60),
        }
    return {
        "owner_tz": anchor.owner_tz,
        "before_date": before,
        "after_date": after,
        "note": ("Denver falls back 2026-11-01 (MDT->MST). A Denver-"
                 "anchored pulse fires 1h later in UTC after the "
                 "transition, so its owner-local time shifts. "
                 "Owner-anchored pulses never drift."),
        "slots": rows,
    }
