"""WhatsApp group health analytics (census §6 #6).

Telegram has rhythm layers; WhatsApp has none — yet the vault
already holds everything a health report needs, per group, in
the GroupStore:

- the member ledger (``message_count`` / ``last_seen`` /
  ``joined_ts`` / ``left_ts`` / ``is_bot`` / ``deleted``),
- the 500-row audit ring (``join`` / ``leave`` rows with ``ts``),
- the 10-day rhythm counters (``extras.rhythm_days``:
  messages/joins/leaves per group-local day),
- the roster-diff state (cadence, last run, last counts).

This module is the pure half: it reads that store and turns it
into health — churn, rhythm shape, silent members, newcomers and
leavers, and a plain-words verdict.  No new polling, no network,
no invented data:

- a count is only claimed for the window the records actually
  cover (the audit ring is bounded — when its oldest row is
  newer than the window start, the report says the window is
  incomplete instead of presenting a partial count as whole);
- a member counts as silent only when they are still in the
  room, are not a bot or a deleted account, and were first seen
  more than ``silent_days`` ago (a newcomer who has not spoken
  yet is new, not silent);
- renders name members by display name only — never a raw JID
  or user id (a WhatsApp JID carries the phone number).

The live roster is the one thing the store cannot give on its
own: the scheduled roster diff could never fetch a WhatsApp
roster (``fetch_live_roster`` only knows the Telegram-shaped
``admin_members`` verbs, while the bridge already answers
``group_participants``).  :func:`refresh_live_roster` runs one
live diff on demand through the roster-diff organ, so churn
stops being traffic-only the moment an admin asks.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Any

__all__ = [
    "SILENT_DAYS", "WINDOW_DAYS", "SILENT_LIST_CAP",
    "TOPIC_READ_LIMIT",
    "build_group_health", "build_overview",
    "render_group_health", "render_overview", "render_topics",
    "one_line", "read_group_messages",
    "refresh_live_roster",
]

#: A member who has not been seen for this long counts as silent.
SILENT_DAYS = 14
#: The churn/activity window, in days.
WINDOW_DAYS = 7
#: Names shown on the silent list before "and N more".
SILENT_LIST_CAP = 10
#: Recorded messages read for the topic clusters — the same bound
#: the digest read uses, so one /ghealth never scans the whole log.
TOPIC_READ_LIMIT = 500

#: Platforms this surface covers.  The organ math is platform
#: agnostic (it reads the store), but the command is scoped to
#: WhatsApp — that is the census item, and Telegram already has
#: its rhythm layers.
HEALTH_PLATFORMS = ("whatsapp", "wa")


def _display_name(member: dict[str, Any]) -> str:
    name = str(member.get("name") or "").strip()
    if name:
        return name
    username = str(member.get("username") or "").strip()
    if username:
        return f"@{username}"
    return "unnamed member"


def _window_days(surface: Any, group_key: str, now: float,
                 window_days: int) -> list[str]:
    """The last ``window_days`` group-local dates, ending today."""
    try:
        today = str(surface.rhythm_local(group_key, now)["day"])
        year, month, dom = (int(p) for p in today.split("-"))
        base = date(year, month, dom)
    except (TypeError, ValueError, AttributeError, KeyError):
        base = date.fromtimestamp(now)
    return [(base - timedelta(days=i)).isoformat()
            for i in range(window_days - 1, -1, -1)]


def _fmt_day(ts: float) -> str:
    return time.strftime("%d %b", time.localtime(ts))


def read_group_messages(log: Any, group_key: str, *,
                        since: float, limit: int = TOPIC_READ_LIMIT
                        ) -> list[dict[str, Any]] | None:
    """The group's recorded messages since ``since`` from the
    brain's persisted log — the same rows the digest reads
    (``messages_after``), mapped to the organ's plain dicts.
    None when the log is unavailable; a read failure is never
    dressed as an empty room."""
    if log is None:
        return None
    try:
        rows = log.messages_after(group_key, since, limit=limit) or []
    except Exception:  # noqa: BLE001 - no log, no topics
        return None
    return [{
        "sender": str(row.get("name") or "?"),
        "text": str(row.get("content") or ""),
        "ts": float(row.get("created_at") or 0.0),
    } for row in rows]


def _fmt_rate(rate: float | None) -> str:
    return f"{rate * 100:.1f}%" if rate is not None else "—"


def render_topics(topics: dict[str, Any] | None) -> str:
    """The topics block: what the group talked about, or the
    honest line when there is no message record to cluster."""
    if topics is None:
        return ("Topics: no message record for this group yet — "
                "nothing to cluster.")
    clusters = topics.get("clusters") or []
    if not clusters:
        return (f"Topics: {topics.get('messages_scanned', 0)} messages "
                "in the last 7 days fit no topic.")
    lines = [f"Talking about (last 7 days, "
             f"{topics.get('messages_scanned', 0)} messages):"]
    for cluster in clusters:
        label = " · ".join(cluster.get("label") or [])
        senders = cluster.get("senders") or []
        top = ", ".join(f"{name} ({count})" for name, count in senders[:2])
        example = str(cluster.get("example") or "")[:120]
        lines.append(
            f"· {label} — {cluster.get('messages', 0)} messages "
            f"({cluster.get('share', 0.0) * 100:.0f}%)"
            + (f" · top: {top}" if top else "")
            + (f' — e.g. "{example}"' if example else ""))
    if topics.get("unclustered"):
        lines.append(f"· {topics['unclustered']} message(s) fit no topic.")
    return "\n".join(lines)


def build_group_health(surface: Any, group_key: str,
                       now: float | None = None, *,
                       silent_days: int = SILENT_DAYS,
                       window_days: int = WINDOW_DAYS,
                       messages: list[dict[str, Any]] | None = None
                       ) -> dict[str, Any] | None:
    """One WhatsApp group's health from the store, or None when
    the store has never seen the group (a pure read — it creates
    no state)."""
    now = now if now is not None else time.time()
    entry = (getattr(surface.store, "data", None) or {}).get(group_key)
    if entry is None:
        return None
    platform = str(group_key).split(":", 1)[0].lower()
    try:
        claim = surface.claim_state(group_key)
    except Exception:  # noqa: BLE001 - claim facts are advisory here
        claim = {"claimed": False, "source": ""}
    members = entry.get("members") or {}
    window_after = now - window_days * 86400.0

    # Churn / growth / topic clusters (the R2-27 remainder): the
    # new organ reads the same ledger the report already reads —
    # joins and leaves are exact over any window, since a member's
    # joined_ts / left_ts pins them — and the supplied messages.
    # No messages supplied (the board, a db-less rig) → topics is
    # None and the render says so, never a faked cluster.
    from ...community import growth as _growth

    growth = _growth.churn_and_growth(
        [m for m in members.values() if isinstance(m, dict)],
        window_days=window_days, now=now)
    topics = (_growth.topic_clusters(messages)
              if messages is not None else None)

    current: list[dict[str, Any]] = []
    bots = 0
    deleted = 0
    left = 0
    active_window = 0
    never_spoke = 0
    for member in members.values():
        if not isinstance(member, dict):
            continue
        if member.get("left_ts"):
            left += 1
            continue
        if member.get("is_bot"):
            bots += 1
            continue
        if member.get("deleted"):
            deleted += 1
            continue
        current.append(member)
        last_seen = float(member.get("last_seen") or 0.0)
        if last_seen >= window_after:
            active_window += 1
        if not int(member.get("message_count") or 0):
            never_spoke += 1

    # Quiet members + engagement come from the community insights
    # organ (audit gap R2-27) — the one source for "who went
    # quiet" and for the per-member ranking. The ledger holds
    # per-member aggregates, not a message stream, so the organ
    # is fed one message per member at its last_seen (members
    # who never spoke get no message — "never" is reported, never
    # faked), and the newcomer guard rides the organ's
    # min_age_days, not a second inline copy of the rule.
    from ...community import insights as _insights

    quiet = _insights.quiet_members(
        current,
        [{"member_id": str(m.get("user_id") or ""),
          "ts": float(m.get("last_seen") or 0.0)}
         for m in current
         if float(m.get("last_seen") or 0.0) > 0.0],
        days=silent_days, now=now, min_age_days=silent_days)
    by_id = {str(m.get("user_id") or ""): m for m in current}
    silent: list[dict[str, Any]] = []
    for row in quiet:
        member = by_id.get(row["member_id"]) or {}
        days_quiet = row["days_since_last"]
        silent.append({
            "user_id": row["member_id"],
            "name": _display_name(member),
            "username": str(member.get("username") or ""),
            "last_seen": float(member.get("last_seen") or 0.0),
            "first_seen": float(member.get("first_seen") or 0.0),
            "message_count": int(member.get("message_count") or 0),
            "days_quiet": (
                int(days_quiet) if days_quiet is not None else None),
        })
    engagement = _insights.engagement_from_counts(
        {str(m.get("user_id") or ""): int(m.get("message_count") or 0)
         for m in current})
    # Names for the top-talkers line are resolved here, where the
    # ledger lives — the render only ever sees display names.
    top_talkers = [
        {"name": _display_name(by_id.get(row["member_id"]) or {}),
         "messages": row["messages"]}
        for row in (engagement.get("per_member") or [])[:5]]

    extras = surface._extras(group_key)
    rhythm_days = extras.get("rhythm_days") or {}
    days = _window_days(surface, group_key, now, window_days)
    msgs_window = 0
    joins_window = 0
    leaves_window = 0
    for day in days:
        row = rhythm_days.get(day) or {}
        msgs_window += int(row.get("messages") or 0)
        joins_window += int(row.get("joins") or 0)
        leaves_window += int(row.get("leaves") or 0)
    kept = sorted(rhythm_days.keys())
    busiest: tuple[str, int] | None = None
    quiet_days: list[str] = []
    for day in kept:
        msgs = int((rhythm_days.get(day) or {}).get("messages") or 0)
        if busiest is None or msgs > busiest[1]:
            busiest = (day, msgs)
        if msgs == 0:
            quiet_days.append(day)

    audit = [row for row in (entry.get("audit") or [])
             if isinstance(row, dict)]
    # Rows stamped after `now` are clock-skew, not observations
    # inside the window ending at `now` — ignore them.
    audit_ts = [float(row.get("ts") or 0.0) for row in audit
                if 0.0 < float(row.get("ts") or 0.0) <= now]
    audit_covers_since = min(audit_ts) if audit_ts else None
    audit_window_complete = bool(
        audit_covers_since is not None
        and audit_covers_since <= window_after)
    newcomers: list[dict[str, Any]] = []
    leavers: list[dict[str, Any]] = []
    for row in audit:
        ts = float(row.get("ts") or 0.0)
        if ts < window_after or ts > now:
            continue
        kind = str(row.get("kind") or "")
        ledger = members.get(str(row.get("user_id") or "")) or {}
        name = str(row.get("name") or "")
        if not name and ledger:
            name = _display_name(ledger)
        if kind == "join" and name:
            newcomers.append({"name": name, "ts": ts})
        elif kind == "leave" and name:
            leavers.append({"name": name, "ts": ts})
    newcomers.sort(key=lambda item: item["ts"])
    leavers.sort(key=lambda item: item["ts"])

    from . import group_roster_diff as _rd

    roster = _rd.roster_diff_settings(surface, group_key)

    total = len(current)
    if leaves_window > joins_window and leaves_window >= 2:
        verdict = (f"churning — {leaves_window} left, "
                   f"{joins_window} joined in the last "
                   f"{window_days} days.")
    elif msgs_window == 0:
        verdict = (f"quiet — no messages in the last "
                   f"{window_days} days.")
    elif total >= 5 and len(silent) * 2 >= total:
        verdict = (f"most of the room has gone quiet — "
                   f"{len(silent)} of {total} silent "
                   f"{silent_days} days or more.")
    else:
        verdict = (f"healthy — {active_window} of {total} spoke "
                   f"in the last {window_days} days.")
    return {
        "group_key": group_key,
        # The stored title, or "" — the render says "this group"
        # rather than printing the raw key (a WhatsApp key is a
        # JID; the overview falls back to the key, like the
        # /groups board already does).
        "title": str(entry.get("title") or ""),
        "platform": platform,
        "claimed": bool(claim.get("claimed")),
        "claim_source": str(claim.get("source") or ""),
        "members_total": total,
        "members_left": left,
        "bots": bots,
        "deleted": deleted,
        "active_7d": active_window,
        "silent": silent,
        "silent_count": len(silent),
        "never_spoke": never_spoke,
        "engagement": engagement,
        "top_talkers": top_talkers,
        "msgs_7d": msgs_window,
        "msgs_per_day": (msgs_window / window_days
                         if window_days else 0.0),
        "joins_7d": joins_window,
        "leaves_7d": leaves_window,
        "busiest_day": busiest,
        "quiet_days": quiet_days,
        "rhythm_days_kept": len(kept),
        "newcomers_7d": newcomers,
        "leavers_7d": leavers,
        "growth": growth,
        "topics": topics,
        "audit_covers_since": audit_covers_since,
        "audit_window_complete": audit_window_complete,
        "roster": roster,
        "verdict": verdict,
    }


def build_overview(surface: Any, now: float | None = None, *,
                   platform: str = "whatsapp",
                   silent_days: int = SILENT_DAYS,
                   window_days: int = WINDOW_DAYS
                   ) -> dict[str, Any]:
    """Every group of one platform in the store, healthiest
    first by recent messages.  Channels are not groups."""
    now = now if now is not None else time.time()
    wanted = str(platform or "").lower()
    groups: list[dict[str, Any]] = []
    for key in sorted(getattr(surface.store, "data", None) or {}):
        if str(key).split(":", 1)[0].lower() != wanted:
            continue
        try:
            if surface.is_channel(key):
                continue
        except Exception:  # noqa: BLE001 - unreadable key is skipped
            continue
        health = build_group_health(
            surface, key, now=now, silent_days=silent_days,
            window_days=window_days)
        if health is not None:
            groups.append(health)
    groups.sort(key=lambda item: (-item["msgs_7d"],
                                  item["group_key"]))
    return {
        "platform": wanted,
        "groups": groups,
        "totals": {
            "groups": len(groups),
            "members": sum(g["members_total"] for g in groups),
            "active_7d": sum(g["active_7d"] for g in groups),
            "silent": sum(g["silent_count"] for g in groups),
            "msgs_7d": sum(g["msgs_7d"] for g in groups),
            "joins_7d": sum(g["joins_7d"] for g in groups),
            "leaves_7d": sum(g["leaves_7d"] for g in groups),
            "net_7d": sum(g["growth"]["current"]["net"] for g in groups),
        },
    }


def one_line(health: dict[str, Any]) -> str:
    """The board's one-liner for a group."""
    return (f"{health['members_total']} members · "
            f"{health['active_7d']} spoke in 7d · "
            f"{health['silent_count']} silent · "
            f"{health['msgs_7d']} msgs in 7d")


def render_group_health(health: dict[str, Any]) -> str:
    """The full report, in plain words.  Names only — never a
    raw JID or user id."""
    lines = [f"Health — {health['title'] or 'this group'}",
             health["verdict"]]
    lines.append(
        f"{health['members_total']} members · "
        f"{health['active_7d']} spoke in the last 7 days · "
        f"{health['silent_count']} silent 14 days or more · "
        f"{health['never_spoke']} never spoke")
    if health["members_left"]:
        lines.append(f"{health['members_left']} have left "
                     f"(still in the ledger)")
    if health["bots"]:
        lines.append(f"{health['bots']} bot(s) in the room")
    if health["deleted"]:
        lines.append(f"{health['deleted']} deleted account(s) "
                     f"still in the ledger")
    busy = health["busiest_day"]
    lines.append(
        f"Messages: {health['msgs_7d']} in the last 7 days "
        f"({health['msgs_per_day']:.1f} a day)"
        + (f" · busiest {busy[0]} ({busy[1]})" if busy else "")
        + (f" · quiet days: {', '.join(health['quiet_days'])}"
           if health["quiet_days"] else ""))
    top_talkers = health.get("top_talkers") or []
    if top_talkers:
        lines.append("Top talkers (all time): " + " · ".join(
            f"{row['name']} ({row['messages']})"
            for row in top_talkers[:5]))
    lines.append(
        f"Churn: +{health['joins_7d']} joined, "
        f"−{health['leaves_7d']} left in the last 7 days "
        f"(from the daily counts)")
    growth = health.get("growth") or {}
    current = growth.get("current") or {}
    direction = growth.get("direction") or "no data"
    lines.append(
        f"Growth: {current.get('net', 0):+d} net in 7 days "
        f"· churn {_fmt_rate(current.get('churn_rate'))} "
        f"· growth {_fmt_rate(current.get('growth_rate'))}"
        + (f" · {direction} vs the prior 7 days"
           if direction != "no data" else
           " · no prior window to compare"))
    lines.append(render_topics(health.get("topics")))
    if health["newcomers_7d"]:
        lines.append("New: " + ", ".join(
            f"{item['name']} ({_fmt_day(item['ts'])})"
            for item in health["newcomers_7d"]))
    if health["leavers_7d"]:
        lines.append("Left: " + ", ".join(
            f"{item['name']} ({_fmt_day(item['ts'])})"
            for item in health["leavers_7d"]))
    if not health["audit_window_complete"]:
        if health["audit_covers_since"] is None:
            lines.append("No join/leave history recorded yet — "
                         "newcomers and leavers above come only "
                         "from the daily counts.")
        else:
            lines.append(
                "Join/leave history only reaches back to "
                f"{_fmt_day(health['audit_covers_since'])} — "
                "the names above may be partial.")
    if health["silent"]:
        shown = health["silent"][:SILENT_LIST_CAP]
        rest = len(health["silent"]) - len(shown)
        lines.append("Silent 14d+: " + ", ".join(
            f"{item['name']} ({item['days_quiet']}d)"
            if item["days_quiet"] is not None
            else f"{item['name']} (never spoke)"
            for item in shown)
            + (f" · and {rest} more" if rest else ""))
    roster = health["roster"] or {}
    last = float(roster.get("last_run_ts") or 0.0)
    from . import group_roster_diff as _rd

    lines.append(
        "Roster: "
        + (_rd.describe_roster_diff(roster)
           + (f" · last checked {_fmt_day(last)}" if last else
              " · never checked live"))
        + (" — counts above come from traffic I saw; run "
           "/ghealth refresh to check the live roster now."
           if not last else ""))
    if not health["claimed"]:
        lines.append("Not claimed — system output is off here; "
                     "the counts above are from traffic I saw.")
    return "\n".join(lines)


def render_overview(overview: dict[str, Any]) -> str:
    """One line per group, healthiest first."""
    totals = overview["totals"]
    lines = [
        f"{totals['groups']} WhatsApp group"
        f"{'s' if totals['groups'] != 1 else ''} · "
        f"{totals['members']} members · "
        f"{totals['active_7d']} spoke in the last 7 days · "
        f"{totals['silent']} silent · "
        f"{totals['msgs_7d']} msgs in 7 days · "
        f"net {totals.get('net_7d', 0):+d} in 7 days"]
    for health in overview["groups"]:
        label = health["title"] or health["group_key"]
        lines.append(f"— {label}: {one_line(health)} · "
                     f"{health['verdict']}")
    if not overview["groups"]:
        lines.append("No WhatsApp groups in the store yet.")
    return "\n".join(lines)


def refresh_live_roster(surface: Any, group_key: str,
                        adapters: list, *,
                        now: float | None = None
                        ) -> dict[str, Any]:
    """One live roster diff, right now: fetch the live member
    list (the bridge's group roster on WhatsApp) and diff it
    against the snapshot.  Returns the diff summary, or
    ``{"ok": False, "error": ...}`` when no member-list route
    exists — absence is never dressed as an empty roster."""
    from . import group_roster_diff as _rd

    fetched = _rd.fetch_live_roster(adapters or [], group_key,
                                    surface=surface)
    if not isinstance(fetched, dict) or not fetched.get("ok"):
        return {"ok": False,
                "error": str((fetched or {}).get("error")
                             or "no member-list route")}
    result = _rd.diff_group_roster(
        surface, group_key, fetched.get("rows") or [],
        now=now, admin_ids=fetched.get("admin_ids"),
        truncated=bool(fetched.get("truncated")),
        source=str(fetched.get("source") or ""))
    result["ok"] = True
    result["source"] = str(fetched.get("source") or "")
    return result
