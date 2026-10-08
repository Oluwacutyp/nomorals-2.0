"""Community events: create → RSVP → calendar sync → reminders → summary.

#84 — events with in-app calendar, reminders, and post-event
summaries. Scheduled rituals ("meetup every Wednesday") are
recurrence rules on the same lifecycle.

Isolation: stdlib + ``nomorals.core`` only. Calendar sync goes
through an injectable ``calendar_provider`` callable — the community
package never imports ``nomorals.connectors.*``; the chat layer
outside this package injects a real provider (or none, honestly).

Reminders: ``ensure_schedule(scheduler)`` registers the daily cron;
``check_reminders(now)`` is the host entry point returning reminder
texts (the host delivers them — delivery needs platform handles the
store doesn't keep).
"""

from __future__ import annotations

import calendar
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger

_log = get_logger("nomorals.community.events")

_DATA_DIR = Path.home() / ".devon" / "community" / "events"

#: Scheduler action handled by the host for event reminders.
REMINDERS_ACTION = "community_event_reminders"
REMINDERS_CRON = "0 8 * * *"  # daily 08:00 — mirrors the travel watcher seam

RSVP_YES = "yes"
RSVP_NO = "no"
RSVP_MAYBE = "maybe"
RSVPS = (RSVP_YES, RSVP_NO, RSVP_MAYBE)

_REMIND_WINDOWS = (24 * 3600.0, 3600.0)  # 1 day before, 1 hour before


@dataclass
class CommunityEvent:
    """One event in a themed group."""

    id: str
    group_id: str
    title: str
    starts_at: float  # epoch
    where: str = ""
    created_by: str = ""
    created_at: float = 0.0
    recurrence: str = ""  # "" | "weekly"
    weekday: int = -1  # 0=Mon..6=Sun when recurrence == "weekly"
    rsvps: dict[str, str] = field(default_factory=dict)  # member → yes/no/maybe
    calendar_ref: str = ""  # provider's event id/link, "" when not synced
    reminders_sent: list[float] = field(default_factory=list)  # windows already sent
    summary: str = ""  # post-event summary text

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CommunityEvent":
        return cls(
            id=str(data.get("id", "")),
            group_id=str(data.get("group_id", "")),
            title=str(data.get("title", "")),
            starts_at=float(data.get("starts_at") or 0.0),
            where=str(data.get("where", "")),
            created_by=str(data.get("created_by", "")),
            created_at=float(data.get("created_at") or 0.0),
            recurrence=str(data.get("recurrence", "")),
            weekday=int(data.get("weekday") if data.get("weekday") is not None else -1),
            rsvps=dict(data.get("rsvps") or {}),
            calendar_ref=str(data.get("calendar_ref", "")),
            reminders_sent=list(data.get("reminders_sent") or []),
            summary=str(data.get("summary", "")),
        )

    def attendees(self) -> list[str]:
        return [m for m, r in self.rsvps.items() if r == RSVP_YES]

    def is_past(self, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        return self.starts_at < now


class EventStore:
    """JSON persistence for community events. Never raises.

    ``calendar_provider``: ``Callable[[dict], str | None]`` — receives
    an event dict, returns a calendar ref (link/id), or None when the
    provider can't / isn't connected. Injected by the chat layer.
    """

    def __init__(self, data_dir: Path | str | None = None,
                 calendar_provider: Callable[[dict[str, Any]], str | None] | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else _DATA_DIR
        self._calendar = calendar_provider

    def _path(self) -> Path:
        return self.data_dir / "events.json"

    def _load(self) -> list[CommunityEvent]:
        try:
            raw = json.loads(self._path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        out: list[CommunityEvent] = []
        for item in raw if isinstance(raw, list) else []:
            try:
                e = CommunityEvent.from_dict(item)
                if e.id and e.title:
                    out.append(e)
            except (TypeError, ValueError, AttributeError):
                continue
        return out

    def _save(self, events: list[CommunityEvent]) -> None:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            tmp = self._path().with_suffix(".tmp")
            tmp.write_text(
                json.dumps([e.to_dict() for e in events],
                           ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
            tmp.replace(self._path())
        except OSError as exc:
            _log.warning("event save failed: %s", exc)

    # ── lifecycle ────────────────────────────────────────────────────

    def create_event(self, group_id: str, title: str, starts_at: float,
                     where: str = "", created_by: str = "",
                     recurrence: str = "", weekday: int = -1) -> CommunityEvent | None:
        """Create an event. Syncs to the calendar provider when wired."""
        title = (title or "").strip()
        if not title or starts_at <= 0:
            return None
        now = time.time()
        ev = CommunityEvent(
            id="evt_" + new_short_id(),
            group_id=group_id,
            title=title[:120],
            starts_at=starts_at,
            where=(where or "").strip()[:200],
            created_by=created_by,
            created_at=now,
            recurrence=recurrence if recurrence == "weekly" else "",
            weekday=weekday if 0 <= weekday <= 6 else -1,
        )
        ref = self._sync_calendar(ev)
        if ref:
            ev.calendar_ref = ref
        events = self._load()
        events.append(ev)
        self._save(events)
        return ev

    def _sync_calendar(self, ev: CommunityEvent) -> str:
        """Best-effort calendar sync via the injected provider. Honest."""
        if self._calendar is None:
            return ""
        try:
            ref = self._calendar(ev.to_dict())
            return str(ref or "")
        except Exception as exc:  # noqa: BLE001 — calendar must never break events
            _log.warning("calendar sync failed for %s: %s", ev.id, exc)
            return ""

    def sync_event(self, event_id: str) -> str:
        """Retry calendar sync for an existing event. Returns the ref or ''."""
        events = self._load()
        for ev in events:
            if ev.id == event_id:
                ref = self._sync_calendar(ev)
                if ref:
                    ev.calendar_ref = ref
                    self._save(events)
                return ref
        return ""

    def get(self, event_id: str) -> CommunityEvent | None:
        for e in self._load():
            if e.id == event_id:
                return e
        return None

    def list_events(self, group_id: str = "",
                    upcoming_only: bool = True,
                    now: float | None = None) -> list[CommunityEvent]:
        now = now if now is not None else time.time()
        out = [e for e in self._load()
               if (not group_id or e.group_id == group_id)]
        if upcoming_only:
            out = [e for e in out if not e.is_past(now)]
        return sorted(out, key=lambda e: e.starts_at)

    def rsvp(self, event_id: str, member: str, choice: str) -> bool:
        """Record an RSVP. True when the event exists and choice is valid."""
        choice = (choice or "").lower()
        if choice not in RSVPS or not member:
            return False
        events = self._load()
        for ev in events:
            if ev.id == event_id:
                ev.rsvps[member] = choice
                self._save(events)
                return True
        return False

    def rsvp_counts(self, event_id: str) -> dict[str, int]:
        ev = self.get(event_id)
        counts = {"yes": 0, "no": 0, "maybe": 0}
        if ev:
            for r in ev.rsvps.values():
                counts[r] = counts.get(r, 0) + 1
        return counts

    def missing_rsvps(self, event_id: str,
                      members: list[str]) -> list[str]:
        """Members who haven't RSVP'd — the anti-ghosting list."""
        ev = self.get(event_id)
        if ev is None:
            return []
        return [m for m in members if m not in ev.rsvps]

    def post_event_summary(self, event_id: str, text: str,
                           now: float | None = None) -> CommunityEvent | None:
        """Attach a post-event summary (who came, what happened)."""
        now = now if now is not None else time.time()
        events = self._load()
        for ev in events:
            if ev.id == event_id and ev.is_past(now):
                ev.summary = (text or "").strip()[:2000]
                self._save(events)
                return ev
        return None

    def cancel_event(self, event_id: str) -> bool:
        events = self._load()
        kept = [e for e in events if e.id != event_id]
        if len(kept) == len(events):
            return False
        self._save(kept)
        return True

    # ── rituals: recurrence ──────────────────────────────────────────

    def next_occurrences(self, event_id: str, n: int = 4,
                         now: float | None = None) -> list[float]:
        """Future start times for a weekly ritual event."""
        now = now if now is not None else time.time()
        ev = self.get(event_id)
        if ev is None or ev.recurrence != "weekly" or ev.weekday < 0:
            return []
        out: list[float] = []
        day = time.localtime(ev.starts_at)
        base = time.mktime((day.tm_year, day.tm_mon, day.tm_mday,
                            day.tm_hour, day.tm_min, 0,
                            0, 0, day.tm_isdst))
        # advance to the next matching weekday after `now`
        cursor = base
        while cursor <= now:
            cursor += 7 * 86400.0
        # align: find the first future weekday match
        while time.localtime(cursor).tm_wday != ev.weekday:
            cursor += 86400.0
        for _ in range(max(0, n)):
            out.append(cursor)
            cursor += 7 * 86400.0
        return out

    def expand_recurrence(self, event_id: str, n: int = 4,
                          now: float | None = None) -> list[CommunityEvent]:
        """Materialize the next ``n`` instances of a weekly ritual."""
        now = now if now is not None else time.time()
        ev = self.get(event_id)
        if ev is None or ev.recurrence != "weekly":
            return []
        instances: list[CommunityEvent] = []
        events = self._load()
        for starts in self.next_occurrences(event_id, n, now):
            inst = CommunityEvent(
                id="evt_" + new_short_id(),
                group_id=ev.group_id,
                title=ev.title,
                starts_at=starts,
                where=ev.where,
                created_by=ev.created_by,
                created_at=now,
            )
            instances.append(inst)
        events.extend(instances)
        self._save(events)
        return instances

    # ── reminders ────────────────────────────────────────────────────

    def check_reminders(self, now: float | None = None) -> list[dict[str, Any]]:
        """Reminder payloads for events entering a remind window.

        Returns ``[{event_id, window, text, members}]``. The host
        delivers; the store never sends anything itself.
        """
        now = now if now is not None else time.time()
        due: list[dict[str, Any]] = []
        events = self._load()
        changed = False
        for ev in events:
            if ev.is_past(now):
                continue
            delta = ev.starts_at - now
            for window in _REMIND_WINDOWS:
                if delta <= window and window not in ev.reminders_sent:
                    label = "tomorrow" if window > 7200 else "in an hour"
                    members = ev.attendees() or list(ev.rsvps)
                    due.append({
                        "event_id": ev.id,
                        "window": window,
                        "text": (f"⏰ Reminder: **{ev.title}** starts {label} "
                                 f"({ev.where or 'location TBD'}). "
                                 f"{len(ev.attendees())} going."),
                        "members": members,
                    })
                    ev.reminders_sent.append(window)
                    changed = True
        if changed:
            self._save(events)
        return due

    # ── formatting ───────────────────────────────────────────────────

    @staticmethod
    def format_event(ev: CommunityEvent) -> str:
        when = time.strftime("%a %Y-%m-%d %H:%M", time.localtime(ev.starts_at))
        yes = len(ev.attendees())
        cal = f" 📅 {ev.calendar_ref}" if ev.calendar_ref else ""
        ritual = " 🔁 weekly" if ev.recurrence == "weekly" else ""
        return (f"📍 **{ev.title}** `{ev.id}`{ritual}{cal}\n"
                f"   {when} · {ev.where or 'location TBD'} · {yes} going")


# ── scheduler seam ─────────────────────────────────────────────────────────


def ensure_schedule(scheduler: Any) -> bool:
    """Register the daily event-reminder cron. Mirrors the travel seam."""
    try:
        import asyncio

        async def _ensure() -> bool:
            try:
                jobs = scheduler.list_jobs()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                jobs = []
            for j in jobs or []:
                if getattr(j, "action", "") == REMINDERS_ACTION:
                    return True
            await scheduler.schedule_cron(
                task_id="community-event-reminders",
                cron_expr=REMINDERS_CRON,
                action=REMINDERS_ACTION,
                parameters={},
            )
            return True

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            # called from async context — schedule as a task and assume ok
            loop.create_task(_ensure())
            return True
        return bool(asyncio.run(_ensure()))
    except Exception as exc:  # noqa: BLE001 — scheduling must never raise
        _log.warning("ensure_schedule failed: %s", exc)
        return False


def _parse_when(text: str) -> float:
    """Parse 'YYYY-MM-DD HH:MM' (local) → epoch. 0.0 when unparseable."""
    m = __import__("re").fullmatch(
        r"(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})", (text or "").strip())
    if not m:
        return 0.0
    try:
        return time.mktime((
            int(m.group(1)), int(m.group(2)), int(m.group(3)),
            int(m.group(4)), int(m.group(5)), 0, 0, 0, -1))
    except (ValueError, OverflowError):
        return 0.0


def control(tail: str, context=None, chat=None,
            sender: str = "", sender_id: str = "") -> str:
    """``/cgroup event …`` subcommand handler. Never raises."""
    try:
        return _control(tail, context, chat, sender, sender_id)
    except Exception as exc:  # noqa: BLE001
        _log.warning("event control failed: %s", exc)
        return "Event handling hit a snag — try again."


def _control(tail: str, context, chat, sender: str, sender_id: str) -> str:
    provider = None
    if context is not None:
        provider = getattr(context, "community_calendar", None)
    store = EventStore(calendar_provider=provider)
    who = (sender or sender_id or "anon").strip() or "anon"
    rest = (tail or "").strip()
    # strip the leading "event" the parent dispatcher matched on
    if rest.lower().startswith("event"):
        rest = rest[5:].strip()
    parts = rest.split(None, 1)
    if not parts:
        return ("/cgroup event new <group_id> | <title> | <YYYY-MM-DD HH:MM> | [where] | "
                "[weekly]\n/cgroup event rsvp <event_id> <yes|no|maybe> · "
                "list [group_id] · show <event_id> · summary <event_id> <text> · "
                "ritual <event_id> <n>")
    cmd, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else "")

    if cmd == "new":
        segs = [s.strip() for s in arg.split("|")]
        if len(segs) < 3:
            return "Usage: /cgroup event new <group_id> | <title> | <YYYY-MM-DD HH:MM> | [where] | [weekly]"
        starts = _parse_when(segs[2])
        if not starts:
            return "Date format: YYYY-MM-DD HH:MM (e.g. 2026-10-10 18:00)"
        recurrence = "weekly" if any("weekly" in s.lower() for s in segs[3:]) else ""
        weekday = time.localtime(starts).tm_wday if recurrence else -1
        where = next((s for s in segs[3:] if "weekly" not in s.lower()), "")
        ev = store.create_event(segs[0], segs[1], starts,
                                where=where, created_by=who,
                                recurrence=recurrence, weekday=weekday)
        if ev is None:
            return "Couldn't create the event."
        note = "" if ev.calendar_ref else " (calendar not connected — sync skipped)"
        return f"📍 Event created{note}:\n{EventStore.format_event(ev)}"

    if cmd == "rsvp":
        segs = arg.split()
        if len(segs) < 2:
            return "Usage: /cgroup event rsvp <event_id> <yes|no|maybe>"
        if store.rsvp(segs[0], who, segs[1]):
            counts = store.rsvp_counts(segs[0])
            return (f"RSVP recorded: {segs[1].lower()}. "
                    f"✅ {counts['yes']} · ❌ {counts['no']} · ❔ {counts['maybe']}")
        return "RSVP failed — check the event id and choice (yes/no/maybe)."

    if cmd == "list":
        events = store.list_events(arg.strip())
        if not events:
            return "No upcoming events."
        lines = ["📍 **Upcoming events:**"]
        lines.extend(EventStore.format_event(e) for e in events[:10])
        return "\n".join(lines)

    if cmd == "show":
        ev = store.get(arg.strip())
        if ev is None:
            return "No such event."
        lines = [EventStore.format_event(ev), "RSVPs:"]
        for m, r in ev.rsvps.items():
            lines.append(f"  {m}: {r}")
        if ev.summary:
            lines.append(f"Summary: {ev.summary[:300]}")
        return "\n".join(lines)

    if cmd == "summary":
        eid, _, text = arg.partition(" ")
        ev = store.post_event_summary(eid.strip(), text)
        if ev is None:
            return "Couldn't save the summary (event must exist and be past)."
        return f"📝 Summary saved for **{ev.title}**."

    if cmd == "ritual":
        segs = arg.split()
        n = int(segs[1]) if len(segs) > 1 and segs[1].isdigit() else 4
        instances = store.expand_recurrence(segs[0] if segs else "", n)
        if not instances:
            return "Couldn't expand — needs a weekly ritual event id."
        return f"🔁 Created {len(instances)} upcoming instances."

    if cmd == "cancel":
        return "Event cancelled." if store.cancel_event(arg.strip()) else "No such event."

    if cmd == "nudge":
        # Anti-ghosting: members of an event who haven't RSVP'd.
        from .groups import GroupStore
        ev = store.get(arg.strip())
        if ev is None:
            return "Usage: /cgroup event nudge <event_id>"
        gstore = GroupStore()
        group = gstore.get(ev.group_id)
        members = group.members if group else list(ev.rsvps)
        missing = store.missing_rsvps(ev.id, members)
        if not missing:
            return "Everyone has RSVP'd. No nudges needed. 🎉"
        lines = [f"🔔 **Your turn — RSVP to {ev.title}:**"]
        lines.extend(f"  {m}" for m in missing[:20])
        return "\n".join(lines)

    return "Unknown event command. Try: new · rsvp · list · show · summary · ritual · cancel · nudge"
