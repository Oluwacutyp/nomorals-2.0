"""Per-contact total recall briefs (census §6 candidate #5).

"Everything between me and X": messages, media, statuses, calls,
across platforms — assembled only from what Devon actually
recorded: the personal vault's journal, the DM cache rings, the
recovery file, live per-chat history verbs, and the group ledger.
Every section names its source; an empty source says so instead of
inventing a record.

Pure leaf organ (stdlib only): the handler gathers the pieces, this
module matches, assembles and renders.  It imports nothing from the
runtime or the vault — the handler passes plain rows in.
"""
from __future__ import annotations

import datetime
from typing import Any

#: Journal kinds whose rows carry a counterpart (a person, not a
#: channel and not the owner).  Channel kinds (channel_post,
#: tg_channel_post, channel_post_missed, channel_posted) belong to
#: /channeltexts; status_posted / profile_changed / presence are
#: the owner's own acts, not a contact's.
CONTACT_JOURNAL_KINDS = (
    "status_arrival", "story", "status", "view_once", "call_offer",
    "status_view_trace", "status_react", "story_react",
)

#: Kinds that ARE a post by the contact (an arrival and its saved
#: copy are the same post — deduped by source id).
POST_KINDS = ("status_arrival", "story", "status")

#: Nigeria time — fixed UTC+1, Africa/Lagos has no DST, never tzdata
#: (same shape as the status radar).
WAT = datetime.timezone(datetime.timedelta(hours=1))

#: How many recent items each section shows; the counts stay exact.
SHOW_RECENT = 5


def _digits(value: Any) -> str:
    return "".join(ch for ch in str(value or "") if ch.isdigit())


def _norm_name(value: Any) -> str:
    return str(value or "").strip().casefold()


def _fmt_ts(ts: Any) -> str:
    try:
        moment = datetime.datetime.fromtimestamp(float(ts), tz=WAT)
    except (TypeError, ValueError, OverflowError):
        return "?"
    return moment.strftime("%d %b %Y %H:%M")


def matches_contact(query: Any, user_id: Any, username: Any = "",
                    name: Any = "") -> bool:
    """Does an observed identity match a query string?

    The house discipline, shared with presence watch: digits match
    an id by digit tail (a bare phone number matches
    ``<digits>@s.whatsapp.net`` and ``<digits>@lid``); otherwise the
    username (leading @ stripped, case-insensitive) or the display
    name (case-insensitive exact) must match.  An exact id string
    matches itself.  A name never matches by substring — a brief
    for "Ada" must not pull in "Adaeze".
    """
    query = str(query or "").strip()
    if not query:
        return False
    if str(user_id or "") == query:
        return True
    digits = _digits(query)
    if len(digits) >= 7:
        observed = _digits(user_id)
        return bool(observed) and (
            observed == digits
            or observed.endswith(digits)
            or digits.endswith(observed))
    if _norm_name(username).lstrip("@") == _norm_name(query).lstrip("@"):
        return True
    return _norm_name(name) == _norm_name(query)


def build_contact_index(rows: list[dict[str, Any]]
                        ) -> list[dict[str, Any]]:
    """One entry per (platform, counterpart) the journal knows:
    names seen, per-kind counts, first/last seen.  Newest first."""
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if str(row.get("kind") or "") not in CONTACT_JOURNAL_KINDS:
            continue
        platform = str(row.get("platform") or "")
        cid = str(row.get("counterpart_id") or "")
        if not platform or not cid:
            continue
        slot = merged.setdefault((platform, cid), {
            "platform": platform, "counterpart_id": cid,
            "names": set(), "kinds": {}, "total": 0,
            "first_ts": 0.0, "last_ts": 0.0})
        name = str(row.get("poster_name") or "")
        if name:
            slot["names"].add(name)
        kind = str(row.get("kind") or "")
        slot["kinds"][kind] = slot["kinds"].get(kind, 0) + 1
        slot["total"] += 1
        ts = float(row.get("ts") or 0)
        if ts:
            if not slot["first_ts"] or ts < slot["first_ts"]:
                slot["first_ts"] = ts
            if ts > slot["last_ts"]:
                slot["last_ts"] = ts
    out = []
    for slot in merged.values():
        slot["names"] = sorted(slot["names"])
        out.append(slot)
    out.sort(key=lambda e: e["last_ts"], reverse=True)
    return out


def resolve_contact(query: Any, index: list[dict[str, Any]], *,
                    platform: str = "") -> list[dict[str, Any]]:
    """Index entries matching the query (optionally one platform).
    More than one hit is ambiguity, never a guess — the caller
    lists them."""
    query = str(query or "").strip()
    if not query:
        return []
    out = []
    for entry in index:
        if platform and entry["platform"] != platform:
            continue
        cid = str(entry["counterpart_id"] or "")
        if matches_contact(query, cid):
            out.append(entry)
            continue
        if any(matches_contact(query, cid, name=name)
               for name in entry["names"]):
            out.append(entry)
    return out


def dedupe_posts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per post: a status arrival and its saved copy share
    a source id.  Rows without a source id key on (kind, ts)."""
    seen: set[tuple[str, str]] = set()
    out = []
    for row in rows:
        if str(row.get("kind") or "") not in POST_KINDS:
            continue
        source = str(row.get("source_id") or "")
        key = (str(row.get("platform") or ""),
               source or f"{row.get('kind')}:{row.get('ts')}")
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    out.sort(key=lambda r: float(r.get("ts") or 0))
    return out


def assemble_brief(contact: dict[str, Any], *,
                   journal_rows: list[dict[str, Any]],
                   recoveries: list[dict[str, Any]],
                   cached_messages: list[dict[str, Any]],
                   live_history: list[dict[str, Any]],
                   groups: list[dict[str, Any]],
                   live_notes: list[str]) -> dict[str, Any]:
    """One brief from the gathered pieces.  Every list is plain
    rows the handler fetched; nothing here touches a store."""
    posts = dedupe_posts(journal_rows)
    viewed = [r for r in journal_rows
              if str(r.get("kind") or "") == "status_view_trace"]
    reacted = [r for r in journal_rows
               if str(r.get("kind") or "")
               in ("status_react", "story_react")]
    calls = [r for r in journal_rows
             if str(r.get("kind") or "") == "call_offer"]
    media_saved = [
        r for r in journal_rows
        if str(r.get("kind") or "") == "view_once"
        or (str(r.get("kind") or "") in ("story", "status")
            and str(r.get("path") or ""))]
    cached = sorted(cached_messages,
                    key=lambda m: float(m.get("ts") or 0))
    live = sorted(live_history,
                  key=lambda m: float(m.get("ts") or 0))
    stamps = [float(r.get("ts") or 0) for r in journal_rows]
    stamps += [float(m.get("ts") or 0) for m in cached]
    stamps += [float(m.get("ts") or 0) for m in live]
    stamps = [ts for ts in stamps if ts]
    return {
        "platform": str(contact.get("platform") or ""),
        "counterpart_id": str(contact.get("counterpart_id") or ""),
        "names": list(contact.get("names") or []),
        "first_ts": min(stamps) if stamps else 0.0,
        "last_ts": max(stamps) if stamps else 0.0,
        "messages": {"cached": cached, "live": live},
        "statuses": {"posts": posts, "viewed": viewed,
                     "reacted": reacted},
        "calls": sorted(calls, key=lambda r: float(r.get("ts") or 0)),
        "media_saved": sorted(
            media_saved, key=lambda r: float(r.get("ts") or 0)),
        "recoveries": sorted(
            recoveries, key=lambda r: float(r.get("ts") or 0)),
        "groups": list(groups),
        "live_notes": list(live_notes),
    }


def _clip(text: Any, limit: int = 120) -> str:
    text = " ".join(str(text or "").split())
    if len(text) > limit:
        return text[:limit - 1] + "…"
    return text


def _msg_line(row: dict[str, Any]) -> str:
    arrow = "←" if row.get("incoming") else "→"
    return (f"{arrow} {_fmt_ts(row.get('ts'))} "
            f"{_clip(row.get('text') or '(media)')}")


def render_brief(brief: dict[str, Any]) -> str:
    """Plain-text brief.  Sections that have nothing say so;
    a brief with nothing anywhere says so once, never inventing."""
    names = brief.get("names") or []
    title = names[0] if names else brief.get("counterpart_id") or "?"
    platform = str(brief.get("platform") or "")
    label = {"whatsapp": "WhatsApp", "telegram": "Telegram"}.get(
        platform, platform or "?")
    lines = [f"{title} — {label} · {brief.get('counterpart_id') or ''}"]
    if names[1:]:
        lines.append(f"also known as: {', '.join(names[1:])}")
    if brief.get("first_ts"):
        lines.append(
            f"first seen {_fmt_ts(brief['first_ts'])} WAT · "
            f"last seen {_fmt_ts(brief['last_ts'])} WAT")
    messages = brief["messages"]
    cached = messages["cached"]
    live = messages["live"]
    if cached or live:
        lines.append("")
        lines.append(
            f"Messages — {len(cached)} cached in the vault "
            f"(last 100 kept per chat) · {len(live)} live from the "
            f"account's history")
        for row in sorted(cached + live,
                        key=lambda m: float(m.get("ts") or 0))[-SHOW_RECENT:]:
            lines.append(_msg_line(row))
    else:
        lines.append("")
        lines.append("Messages — none in the vault cache, no live "
                     "history read")
    posts = brief["statuses"]["posts"]
    viewed = brief["statuses"]["viewed"]
    reacted = brief["statuses"]["reacted"]
    if posts or viewed or reacted:
        lines.append("")
        lines.append(
            f"Statuses & stories — {len(posts)} posts "
            f"(vault journal)")
        for row in posts[-SHOW_RECENT:]:
            text = _clip(row.get("text") or "")
            kind = str(row.get("kind") or "")
            stamp = _fmt_ts(row.get("ts"))
            if text:
                lines.append(f"· {stamp} [{kind}] {text}")
            else:
                lines.append(f"· {stamp} [{kind}] (media, no text)")
        if viewed or reacted:
            lines.append(
                f"Devon viewed {len(viewed)} of their statuses, "
                f"reacted to {len(reacted)}")
    calls = brief["calls"]
    if calls:
        lines.append("")
        lines.append(f"Calls — {len(calls)} (vault journal)")
        for row in calls[-SHOW_RECENT:]:
            lines.append(
                f"· {_fmt_ts(row.get('ts'))} — "
                f"{_clip(row.get('note') or 'call offer')}")
    media = brief["media_saved"]
    if media:
        vo = sum(1 for r in media
                 if str(r.get("kind") or "") == "view_once")
        lines.append("")
        lines.append(
            f"Media Devon kept — {len(media)} items "
            f"(view-once {vo}, statuses/stories {len(media) - vo})")
    recoveries = brief["recoveries"]
    if recoveries:
        lines.append("")
        lines.append(f"Recovered — {len(recoveries)} deleted/edited "
                     f"messages Devon kept the originals of")
        for row in recoveries[-3:]:
            kind = str(row.get("kind") or "")
            lines.append(
                f"· {_fmt_ts(row.get('ts'))} [{kind}] "
                f"{_clip(row.get('original_text') or '')}")
    groups = brief["groups"]
    if groups:
        lines.append("")
        lines.append(f"Groups — in {len(groups)} of Devon's groups")
        for entry in groups:
            lines.append(
                f"· {entry.get('title') or entry.get('key')} — "
                f"{int(entry.get('message_count') or 0)} messages"
                + (f" · last seen {_fmt_ts(entry.get('last_seen'))}"
                   if entry.get("last_seen") else ""))
            history = entry.get("identity_history") or []
            if history:
                last = history[-1]
                lines.append(
                    f"  last name change: "
                    f"{last.get('old') or '—'} → "
                    f"{last.get('new') or '—'} "
                    f"({_fmt_ts(last.get('ts'))})")
    else:
        lines.append("")
        lines.append("Groups — no record in Devon's groups")
    for note in brief["live_notes"]:
        lines.append("")
        lines.append(str(note))
    has_anything = bool(
        cached or live or posts or viewed or reacted or calls
        or media or recoveries or groups)
    if not has_anything:
        lines.append("")
        lines.append(
            f"nothing recorded for {title} — no vault rows, no "
            f"cached messages, no live history, no group record.")
    return "\n".join(lines)
