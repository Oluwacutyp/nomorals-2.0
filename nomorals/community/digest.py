"""Group digest — "while you were away" — plus scheduled rituals.

**Digest.** Given the group's *supplied* messages (plain dicts:
``{"id", "sender", "text", "ts", "reply_to"}``), the digest picks:

* the **top 20 decisions** — messages that ask for a decision or
  pose a question, ranked by a deterministic score (decision
  keywords + mentions + replies + recency), never random;
* **unanswered questions** — decision/question messages with no
  reply referencing them and no resolved marker in-thread;
* mentions of ``me``, and resolved threads (for context, not action).

Ranking is a pure function of the messages, so the same input
always yields the same digest. A question answered later in the
thread is *resolved*, not unanswered — the digest never nags about
something the group already settled.

**Rituals.** Scheduled group moments — daily verse, market open,
good morning, Friday vibes, weekly poll — expressed as
:class:`RecurrenceDescriptor` values parsed from RFC 5545-style
``RRULE`` strings (``FREQ=DAILY;BYHOUR=7;BYMINUTE=0`` …). The
recurrence engine is stdlib-only (``datetime`` + ``calendar``):
DAILY / WEEKLY / MONTHLY / YEARLY, INTERVAL, COUNT, UNTIL, BYDAY,
BYMONTHDAY, BYHOUR, BYMINUTE, BYMONTH. No ``dateutil`` dependency,
so it runs on Termux exactly as on a workstation.
:class:`RitualScheduler` tracks ``last_delivered`` per ritual and
reports which rituals are due at a given ``now``.

Isolation: stdlib + ``nomorals.core`` only. No memory, accounts,
vaults, or connectors.
"""

from __future__ import annotations

import calendar
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "GroupDigest",
    "RecurrenceDescriptor",
    "Ritual",
    "RitualScheduler",
    "DEFAULT_RITUALS",
    "DIGEST_LIMIT",
    "parse_rrule",
    "next_occurrences",
    "build_digest",
    "render_digest",
    "classify_message",
    "render_ritual",
]

DIGEST_LIMIT = 20

_DECISION_RES = (
    re.compile(r"\?+\s*$"),
    re.compile(
        r"\b(should we|shall we|decide|decision|vote|confirm|approve|"
        r"need you|your call|what do you think|any objection|"
        r"thoughts\?|pick one|choose|deadline|by when)\b",
        re.I,
    ),
)
_RESOLVED_RES = (
    re.compile(r"\b(resolved|sorted|settled|agreed|done|closed|fixed|confirmed|approved|decided)\b", re.I),
    re.compile(r"✅|✔️|👍|🎉"),
)
_MENTION_RE = re.compile(r"@(\w+)")


def classify_message(text: str, *, me: str = "") -> dict[str, bool]:
    """Flags for one message: mentioned / decision / resolved."""
    t = text or ""
    mentioned = False
    if me:
        name = me.strip().lstrip("@").lower()
        if name and (f"@{name}" in t.lower() or re.search(
            rf"(?<![a-z0-9_]){re.escape(name)}(?![a-z0-9_])", t.lower()
        )):
            mentioned = True
    decision = any(rx.search(t) for rx in _DECISION_RES)
    resolved = any(rx.search(t) for rx in _RESOLVED_RES)
    return {"mentioned": mentioned, "decision": decision, "resolved": resolved}


def _score(msg: dict[str, Any], reply_counts: dict[str, int], me: str) -> float:
    """Deterministic decision score: keywords + mentions + replies + recency."""
    text = str(msg.get("text", ""))
    flags = classify_message(text, me=me)
    score = 0.0
    if flags["decision"]:
        score += 10.0
    if text.rstrip().endswith("?"):
        score += 4.0
    if flags["mentioned"]:
        score += 6.0
    if "@" in text:
        score += 2.0
    for kw in ("vote", "deadline", "decide", "approve", "confirm"):
        if kw in text.lower():
            score += 3.0
    score += min(5, reply_counts.get(str(msg.get("id", "")), 0)) * 1.5
    # Recency: newer scores slightly higher, but never overtakes content.
    ts = float(msg.get("ts") or 0.0)
    score += min(2.0, max(0.0, (ts - 1_700_000_000) / 86400 / 365))
    return round(score, 3)


@dataclass
class GroupDigest:
    group_id: str = ""
    group_name: str = ""
    since: float = 0.0
    generated_at: float = 0.0
    decisions: list[dict[str, Any]] = field(default_factory=list)
    unanswered: list[dict[str, Any]] = field(default_factory=list)
    mentions: list[dict[str, Any]] = field(default_factory=list)
    resolved: list[dict[str, Any]] = field(default_factory=list)
    total_messages: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def needs_attention(self) -> bool:
        return bool(self.unanswered or self.mentions)


def build_digest(
    messages: list[dict[str, Any]],
    *,
    group_id: str = "",
    group_name: str = "",
    me: str = "",
    since: float | None = None,
    limit: int = DIGEST_LIMIT,
    now: float | None = None,
) -> GroupDigest:
    """Build the "while you were away" digest from supplied messages.

    Decisions and unanswered questions are picked by deterministic
    scoring over the messages — never random, never sampled. A
    question with a reply in-thread (or a resolved marker) counts as
    resolved, not unanswered.
    """
    now = now if now is not None else time.time()
    msgs = [m for m in messages if since is None or float(m.get("ts") or 0) >= since]
    msgs = sorted(msgs, key=lambda m: (float(m.get("ts") or 0), str(m.get("id", ""))))
    reply_counts: dict[str, int] = {}
    replied_ids: set[str] = set()
    for m in msgs:
        parent = str(m.get("reply_to") or "")
        if parent:
            reply_counts[parent] = reply_counts.get(parent, 0) + 1
            replied_ids.add(parent)
    # A message is resolved if it carries a resolved marker, or a
    # reply to it carries one, or it has any reply that is not itself
    # a question (an answer, not an echo).
    by_id = {str(m.get("id", "")): m for m in msgs if m.get("id") is not None}
    resolved_ids: set[str] = set()
    for m in msgs:
        mid = str(m.get("id", ""))
        if classify_message(str(m.get("text", "")))["resolved"]:
            resolved_ids.add(mid)
    for m in msgs:
        parent = str(m.get("reply_to") or "")
        if parent in by_id:
            parent_text = str(by_id[parent].get("text", ""))
            if classify_message(parent_text)["decision"]:
                reply_text = str(m.get("text", ""))
                reply_flags = classify_message(reply_text)
                if reply_flags["resolved"] or not reply_flags["decision"]:
                    resolved_ids.add(parent)
    decisions: list[dict[str, Any]] = []
    unanswered: list[dict[str, Any]] = []
    mentions: list[dict[str, Any]] = []
    resolved: list[dict[str, Any]] = []
    for m in msgs:
        flags = classify_message(str(m.get("text", "")), me=me)
        mid = str(m.get("id", ""))
        entry = {
            "id": mid,
            "sender": str(m.get("sender", "")),
            "text": str(m.get("text", "")),
            "ts": float(m.get("ts") or 0),
            "score": _score(m, reply_counts, me),
            "replies": reply_counts.get(mid, 0),
        }
        if flags["mentioned"]:
            mentions.append(entry)
        if flags["decision"]:
            if mid in resolved_ids:
                resolved.append(entry)
            else:
                decisions.append(entry)
                if mid not in replied_ids:
                    unanswered.append(entry)
    # Deterministic ranking: score desc, then ts desc, then id asc.
    key = lambda e: (-e["score"], -e["ts"], e["id"])
    decisions.sort(key=key)
    unanswered.sort(key=key)
    mentions.sort(key=key)
    resolved.sort(key=key)
    return GroupDigest(
        group_id=group_id,
        group_name=group_name,
        since=since if since is not None else 0.0,
        generated_at=now,
        decisions=decisions[:limit],
        unanswered=unanswered[:limit],
        mentions=mentions[:limit],
        resolved=resolved[:limit],
        total_messages=len(msgs),
    )


def render_digest(digest: GroupDigest, *, me: str = "") -> str:
    lines = [f"📬 While you were away — {digest.group_name or digest.group_id or 'group'}"]
    lines.append(f"{digest.total_messages} messages scanned · {len(digest.unanswered)} unanswered")
    if digest.mentions:
        lines.append("\n👋 Mentioned you:")
        for m in digest.mentions[:10]:
            lines.append(f"  • {m['sender']}: {m['text'][:200]}")
    if digest.decisions:
        lines.append(f"\n🗳️ Top {len(digest.decisions)} decisions:")
        for i, m in enumerate(digest.decisions, 1):
            mark = "❓" if m in digest.unanswered else "✓"
            lines.append(f"  {i}. {mark} {m['sender']}: {m['text'][:200]} ({m['replies']} replies)")
    if digest.unanswered:
        lines.append(f"\n❓ Unanswered ({len(digest.unanswered)}):")
        for m in digest.unanswered:
            lines.append(f"  • {m['sender']}: {m['text'][:200]}")
    if not digest.decisions and not digest.mentions:
        lines.append("\nNothing needs you — all quiet. 🌙")
    return "\n".join(lines)


# ── recurrence (RFC 5545-style RRULE, stdlib only) ─────────────────────

_WEEKDAY = {"MO": 0, "TU": 1, "WE": 2, "TH": 3, "FR": 4, "SA": 5, "SU": 6}
_FREQ = ("DAILY", "WEEKLY", "MONTHLY", "YEARLY")


@dataclass
class RecurrenceDescriptor:
    freq: str = "DAILY"
    interval: int = 1
    count: int | None = None
    until: float | None = None  # epoch seconds, inclusive
    byday: tuple[int, ...] = ()  # weekday numbers 0=Mon
    bymonthday: tuple[int, ...] = ()
    bymonth: tuple[int, ...] = ()
    byhour: tuple[int, ...] = ()
    byminute: tuple[int, ...] = (0,)

    def to_rrule(self) -> str:
        parts = [f"FREQ={self.freq}", f"INTERVAL={self.interval}"]
        if self.count is not None:
            parts.append(f"COUNT={self.count}")
        if self.until is not None:
            parts.append(
                "UNTIL=" + datetime.fromtimestamp(self.until, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            )
        if self.byday:
            inv = {v: k for k, v in _WEEKDAY.items()}
            parts.append("BYDAY=" + ",".join(inv[d] for d in self.byday))
        if self.bymonthday:
            parts.append("BYMONTHDAY=" + ",".join(str(d) for d in self.bymonthday))
        if self.bymonth:
            parts.append("BYMONTH=" + ",".join(str(m) for m in self.bymonth))
        if self.byhour:
            parts.append("BYHOUR=" + ",".join(str(h) for h in self.byhour))
        if self.byminute:
            parts.append("BYMINUTE=" + ",".join(str(m) for m in self.byminute))
        return ";".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "rrule": self.to_rrule()}


def parse_rrule(rrule: str) -> RecurrenceDescriptor:
    """Parse ``FREQ=DAILY;INTERVAL=1;BYHOUR=7;BYMINUTE=0`` into a descriptor."""
    kv: dict[str, str] = {}
    for part in (rrule or "").split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        kv[k.strip().upper()] = v.strip()
    freq = kv.get("FREQ", "DAILY").upper()
    if freq not in _FREQ:
        raise ValueError(f"unsupported FREQ {freq!r} (DAILY|WEEKLY|MONTHLY|YEARLY)")

    def _ints(key: str) -> tuple[int, ...]:
        raw = kv.get(key, "")
        if not raw:
            return ()
        return tuple(int(x) for x in raw.split(",") if x.strip() != "")

    byday: tuple[int, ...] = ()
    if kv.get("BYDAY"):
        days = []
        for token in kv["BYDAY"].split(","):
            token = token.strip().upper()
            # Ordinal forms (1MO, -1FR) collapse to the weekday — the
            # scheduler expands monthly ordinals via BYMONTHDAY instead.
            day = token[-2:]
            if day in _WEEKDAY:
                days.append(_WEEKDAY[day])
        byday = tuple(sorted(set(days)))
    until = None
    if kv.get("UNTIL"):
        raw = kv["UNTIL"]
        try:
            if raw.endswith("Z"):
                dt = datetime.strptime(raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            elif "T" in raw:
                dt = datetime.strptime(raw, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            else:
                dt = datetime.strptime(raw, "%Y%m%d").replace(tzinfo=timezone.utc)
            until = dt.timestamp()
        except ValueError as exc:
            raise ValueError(f"bad UNTIL {raw!r}") from exc
    return RecurrenceDescriptor(
        freq=freq,
        interval=max(1, int(kv.get("INTERVAL", "1") or 1)),
        count=int(kv["COUNT"]) if kv.get("COUNT") else None,
        until=until,
        byday=byday,
        bymonthday=_ints("BYMONTHDAY"),
        bymonth=_ints("BYMONTH"),
        byhour=_ints("BYHOUR"),
        byminute=_ints("BYMINUTE") if kv.get("BYMINUTE") else (0,),
    )


def _iter_days(start, end):
    """Yield dates from ``start`` to ``end`` inclusive (both are dates)."""
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def next_occurrences(
    descriptor: RecurrenceDescriptor,
    after: datetime,
    count: int = 5,
    *,
    start: datetime | None = None,
) -> list[datetime]:
    """Next ``count`` occurrences strictly after ``after``.

    ``start`` anchors INTERVAL counting (defaults to ``after``).
    Naive datetimes are treated as UTC; aware ones keep their tz.
    """
    if count <= 0:
        return []
    anchor = start or after
    tz = after.tzinfo
    # Search window: generous but bounded — a daily ritual finds its
    # next occurrence within days; a yearly one within ~2 years.
    step_days = {"DAILY": 1, "WEEKLY": 7, "MONTHLY": 31, "YEARLY": 366}[descriptor.freq]
    window_days = max(370, step_days * (count + 2) * descriptor.interval + 370)
    horizon = after + timedelta(days=window_days)
    hours = descriptor.byhour or (anchor.hour,)
    minutes = descriptor.byminute or (0,)
    out: list[datetime] = []
    emitted = 0
    for day in _iter_days(anchor.date(), horizon.date()):
        # INTERVAL gating on the anchor.
        if descriptor.freq == "DAILY":
            delta = (day - anchor.date()).days
            if delta % descriptor.interval:
                continue
        elif descriptor.freq == "WEEKLY":
            delta_weeks = (day - anchor.date()).days // 7
            if delta_weeks % descriptor.interval:
                continue
            if descriptor.byday and day.weekday() not in descriptor.byday:
                continue
        elif descriptor.freq == "MONTHLY":
            months = (day.year - anchor.year) * 12 + (day.month - anchor.month)
            if months % descriptor.interval:
                continue
            if descriptor.bymonthday and day.day not in descriptor.bymonthday:
                continue
            if descriptor.byday and day.weekday() not in descriptor.byday:
                continue
        elif descriptor.freq == "YEARLY":
            if (day.year - anchor.year) % descriptor.interval:
                continue
            if descriptor.bymonth and day.month not in descriptor.bymonth:
                continue
            if descriptor.bymonthday and day.day not in descriptor.bymonthday:
                continue
            if descriptor.byday and day.weekday() not in descriptor.byday:
                continue
        # WEEKLY without BYDAY defaults to the anchor weekday; MONTHLY
        # without BYMONTHDAY/BYDAY defaults to the anchor day-of-month.
        if descriptor.freq == "WEEKLY" and not descriptor.byday and day.weekday() != anchor.weekday():
            continue
        if descriptor.freq == "MONTHLY" and not descriptor.bymonthday and not descriptor.byday and day.day != anchor.day:
            continue
        if descriptor.freq == "YEARLY" and not descriptor.bymonth and not descriptor.bymonthday and (day.month, day.day) != (anchor.month, anchor.day):
            continue
        for h in sorted(set(hours)):
            for mi in sorted(set(minutes)):
                try:
                    cand = datetime(day.year, day.month, day.day, h, mi, tzinfo=tz)
                except ValueError:
                    continue  # e.g. Feb 30 in BYMONTHDAY
                if cand <= after:
                    continue
                if descriptor.until is not None and cand.timestamp() > descriptor.until:
                    return out
                if descriptor.count is not None and emitted >= descriptor.count:
                    return out
                out.append(cand)
                emitted += 1
                if len(out) >= count:
                    return out
    return out


# ── rituals ────────────────────────────────────────────────────────────

@dataclass
class Ritual:
    name: str = ""
    kind: str = "custom"  # daily_verse | market_open | good_morning | friday_vibes | weekly_poll | custom
    rrule: str = "FREQ=DAILY;BYHOUR=7;BYMINUTE=0"
    template: str = ""
    group_id: str = ""
    enabled: bool = True
    last_delivered: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Ritual":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def recurrence(self) -> RecurrenceDescriptor:
        return parse_rrule(self.rrule)

    def next_due(self, after: datetime) -> datetime | None:
        occ = next_occurrences(self.recurrence, after, 1, start=after)
        return occ[0] if occ else None


def render_ritual(ritual: Ritual, when: datetime | None = None) -> str:
    when = when or datetime.now(tz=timezone.utc)
    text = (ritual.template or ritual.name).format(
        date=when.strftime("%A, %d %B %Y"),
        time=when.strftime("%H:%M"),
        group=ritual.group_id,
        name=ritual.name,
    )
    return text


DEFAULT_RITUALS: tuple[Ritual, ...] = (
    Ritual(
        name="Daily verse",
        kind="daily_verse",
        rrule="FREQ=DAILY;BYHOUR=6;BYMINUTE=30",
        template="📖 Daily verse — {date}\nToday's verse arrives with the morning. Drop your reflection below.",
    ),
    Ritual(
        name="Good morning",
        kind="good_morning",
        rrule="FREQ=DAILY;BYHOUR=7;BYMINUTE=0",
        template="☀️ Good morning! {date}\nWhat's everyone working on today? Drop your top 1.",
    ),
    Ritual(
        name="Market open",
        kind="market_open",
        rrule="FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=9;BYMINUTE=30",
        template="📈 Market open — {date}\nLevels, bias, and watchlist: what are you trading today?",
    ),
    Ritual(
        name="Friday vibes",
        kind="friday_vibes",
        rrule="FREQ=WEEKLY;BYDAY=FR;BYHOUR=16;BYMINUTE=0",
        template="🎉 Friday vibes — {date}\nWeek's done. Wins, lessons, and weekend plans — go.",
    ),
    Ritual(
        name="Weekly poll",
        kind="weekly_poll",
        rrule="FREQ=WEEKLY;BYDAY=MO;BYHOUR=9;BYMINUTE=0",
        template="🗳️ Weekly poll — {date}\nWhat should the group focus on this week? Vote below.",
    ),
)


class RitualScheduler:
    """Which rituals are due at ``now``, with delivery bookkeeping."""

    def __init__(self, rituals: list[Ritual] | None = None) -> None:
        self.rituals: list[Ritual] = list(rituals) if rituals is not None else [
            Ritual(**{**r.to_dict(), "last_delivered": None}) for r in DEFAULT_RITUALS
        ]

    def add(self, ritual: Ritual) -> None:
        self.rituals.append(ritual)

    def due(self, now: datetime, *, window_s: float = 3600) -> list[Ritual]:
        """Rituals whose next occurrence is within ``window_s`` after
        ``now`` and that have not already been delivered for that
        occurrence. A ritual delivered for an earlier occurrence is
        due again at its next one."""
        out: list[Ritual] = []
        now_ts = now.timestamp()
        for ritual in self.rituals:
            if not ritual.enabled:
                continue
            anchor = (
                datetime.fromtimestamp(ritual.last_delivered, tz=now.tzinfo)
                if ritual.last_delivered
                else now - timedelta(days=2)
            )
            # Next occurrence after the last delivery (or recent past).
            nxt = next_occurrences(
                ritual.recurrence, anchor, 1,
                start=anchor,
            )
            if not nxt:
                continue
            occ = nxt[0]
            if occ.timestamp() <= now_ts + window_s and (
                ritual.last_delivered is None
                or occ.timestamp() > ritual.last_delivered
            ):
                out.append(ritual)
        return out

    def mark_delivered(self, ritual: Ritual, when: datetime | None = None) -> None:
        ritual.last_delivered = (when or datetime.now(tz=timezone.utc)).timestamp()
