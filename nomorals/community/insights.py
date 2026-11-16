"""Community insights: engagement, quiet members, turnout.

Pure functions over caller-supplied data — no I/O, no store, no owner
data. The caller passes group activity (messages, event RSVPs) that it
already holds; this module only aggregates it.

* Engagement — messages per member, ranked; totals and active counts.
* Quiet members — members with no message in the last N days (or never).
* Turnout — per-event RSVP vs attended, no-shows, and overall rate.

Inputs accept dicts (``{"member_id": ..., "ts": ...}``) or objects with
the same attributes, so callers can pass their own group/event shapes
without this module importing them.

Isolation: stdlib only. No owner-data imports.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any, Iterable

__all__ = [
    "engagement",
    "engagement_from_counts",
    "quiet_members",
    "turnout",
    "summary",
]


def _field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _member_id(item: Any) -> str:
    value = _field(item, "member_id", "") or _field(item, "user_id", "")
    return str(value)


def _display(item: Any) -> str:
    return str(_field(item, "display_name", "") or _field(item, "name", ""))


def _rank_counts(counts: Counter[str]) -> dict[str, Any]:
    """Shared ranking core: a member→count map → the engagement
    dict. Zero counts never rank (a member with no messages is
    not an active member)."""
    total = sum(counts.values())
    per_member = [
        {
            "member_id": member,
            "messages": count,
            "share": (count / total) if total else 0.0,
        }
        for member, count in counts.most_common()
    ]
    return {
        "total": total,
        "active_members": len(counts),
        "per_member": per_member,
    }


def engagement(
    messages: Iterable[Any],
) -> dict[str, Any]:
    """Message counts per member, ranked most → least.

    Returns ``{"total": int, "active_members": int,
    "per_member": [{member_id, messages, share}]}``. ``share`` is the
    member's fraction of all messages. Empty input → zeros, no crash.
    """
    counts: Counter[str] = Counter()
    for message in messages:
        member = _member_id(message)
        if member:
            counts[member] += 1
    return _rank_counts(counts)


def engagement_from_counts(
    counts: dict[str, int] | Any,
) -> dict[str, Any]:
    """Engagement from per-member message counts, same shape as
    :func:`engagement`. For callers that hold a ledger of
    cumulative counts rather than a message stream — building a
    synthetic stream of one row per message would be wasteful
    and dishonest as an input shape. Zero counts never rank;
    a negative count raises ``ValueError``."""
    ranked: Counter[str] = Counter()
    for member, count in dict(counts).items():
        count = int(count)
        if count < 0:
            raise ValueError("message counts cannot be negative")
        if count:
            ranked[str(member)] += count
    return _rank_counts(ranked)


def quiet_members(
    members: Iterable[Any],
    messages: Iterable[Any],
    days: int = 14,
    now: float | None = None,
    *,
    min_age_days: float | None = None,
) -> list[dict[str, Any]]:
    """Members with no message in the last ``days`` days (or ever).

    Returns ``[{member_id, display_name, last_message_ts,
    days_since_last}]`` sorted longest-silent first. ``last_message_ts``
    is None for members who never posted; ``days_since_last`` is None
    for them too — "never" is reported, never faked as a number.

    ``min_age_days`` is the newcomer guard: a member first seen
    (``first_seen`` or ``joined_ts``) more recently than
    ``min_age_days`` ago is never quiet — a newcomer who has not
    spoken yet is new, not silent. ``None`` (the default) applies
    no guard. A member with no first-seen info at all is treated
    as old enough — absence of join info never silently exempts.
    """
    if days < 0:
        raise ValueError("days cannot be negative")
    if min_age_days is not None and min_age_days < 0:
        raise ValueError("min_age_days cannot be negative")
    now = now if now is not None else time.time()
    cutoff = now - days * 86400
    min_age_cutoff = (
        now - min_age_days * 86400 if min_age_days is not None else None
    )
    last_seen: dict[str, float] = {}
    for message in messages:
        member = _member_id(message)
        ts = _field(message, "ts", 0) or 0
        try:
            ts = float(ts)
        except (TypeError, ValueError):
            continue
        if member and ts > last_seen.get(member, 0):
            last_seen[member] = ts
    quiet = []
    for member in members:
        member_id = _member_id(member)
        if not member_id:
            continue
        if min_age_cutoff is not None:
            first = _field(member, "first_seen", None)
            if first is None:
                first = _field(member, "joined_ts", None)
            if first is not None:
                try:
                    first = float(first)
                except (TypeError, ValueError):
                    first = None
                if first is not None and first >= min_age_cutoff:
                    continue
        last = last_seen.get(member_id)
        if last is None or last < cutoff:
            quiet.append(
                {
                    "member_id": member_id,
                    "display_name": _display(member),
                    "last_message_ts": last,
                    "days_since_last": (
                        (now - last) / 86400 if last is not None else None
                    ),
                }
            )
    quiet.sort(
        key=lambda q: (
            q["days_since_last"] is not None,
            -(q["days_since_last"] or 0),
        )
    )
    return quiet


def turnout(events: Iterable[Any]) -> dict[str, Any]:
    """RSVP vs attended per event, plus the overall rate.

    Each event supplies ``rsvp`` (list of member ids who said yes) and
    ``attended`` (list who showed). Walk-ins (attended but not RSVP'd)
    count as attended, not as no-shows. Overall rate = total attended /
    total RSVP'd; no RSVPs at all → rate None (not zero — "no data"
    differs from "nobody came").
    """
    per_event = []
    total_rsvp = 0
    total_attended = 0
    for event in events:
        rsvp = list(_field(event, "rsvp", []) or [])
        attended = list(_field(event, "attended", []) or [])
        rsvp_set = set(map(str, rsvp))
        attended_set = set(map(str, attended))
        no_shows = sorted(rsvp_set - attended_set)
        walk_ins = sorted(attended_set - rsvp_set)
        per_event.append(
            {
                "event_id": str(_field(event, "event_id", "") or _field(event, "id", "")),
                "title": str(_field(event, "title", "")),
                "rsvp": len(rsvp_set),
                "attended": len(attended_set),
                "no_shows": no_shows,
                "walk_ins": walk_ins,
                "rate": (len(attended_set) / len(rsvp_set)) if rsvp_set else None,
            }
        )
        total_rsvp += len(rsvp_set)
        total_attended += len(attended_set)
    return {
        "events": per_event,
        "total_rsvp": total_rsvp,
        "total_attended": total_attended,
        "overall_rate": (total_attended / total_rsvp) if total_rsvp else None,
    }


def summary(
    members: Iterable[Any],
    messages: Iterable[Any],
    events: Iterable[Any],
    quiet_days: int = 14,
    now: float | None = None,
) -> dict[str, Any]:
    """One combined snapshot: engagement + quiet members + turnout."""
    member_list = list(members)
    message_list = list(messages)
    event_list = list(events)
    return {
        "engagement": engagement(message_list),
        "quiet_members": quiet_members(
            member_list, message_list, days=quiet_days, now=now
        ),
        "turnout": turnout(event_list),
    }
