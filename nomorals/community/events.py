"""Community events: create → RSVP → calendar sync → reminders → summary.

#84 — events with in-app calendar, reminders, and post-event
summaries. Scheduled rituals are recurrence rules on the same lifecycle.

Luma/Meetup-grade upgrades (mined 2026-10-10):
* **Real recurrence engine** (RFC 5545-inspired, stdlib-only — zero deps by
  the standing build order): ``daily[:N]``, ``weekly[:MO,WE,…][:N]``,
  ``monthly[:day][:N]``, plus ``until=``/``count=`` terminators and
  per-date exceptions (``exdates``). Legacy ``recurrence="weekly"`` +
  ``weekday`` keeps working byte-for-byte.
* **Capacity + waitlist** (Luma/Meetup): ``capacity`` caps seats;
  overflow RSVPs queue FIFO in ``waitlist``; a cancellation auto-promotes
  the next in line and yields a notify payload for the host.
* **+N guests** (Paperless Post/Punchbowl): RSVP carries a guest count;
  guests count against capacity.
* **RSVP window** (meetup-parity): ``rsvp_opens_at`` / ``rsvp_closes_at``.
* **Description, end time, link**: events finally have a body.
* **Proxy RSVPs** (Partiful manual-add): hosts add off-platform replies;
  a later self-RSVP attaches to the same record instead of duplicating.
* **Richer reminders**: 7-day, 1-day, 1-hour windows with waitlist +
  guest counts; ``my_events`` = the member's own attendance history
  (wordcamp.org's "Events I attended" pattern).

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
import re
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
RSVP_WAITLIST = "waitlist"
RSVPS = (RSVP_YES, RSVP_NO, RSVP_MAYBE, RSVP_WAITLIST)

_WEEKDAYS = {"MO": 0, "TU": 1, "WE": 2, "TH": 3, "FR": 4, "SA": 5, "SU": 6}
_WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


# ── recurrence engine (stdlib, RFC 5545-flavoured) ─────────────────────────


def parse_recurrence(rule: str) -> dict[str, Any] | None:
    """Parse a recurrence rule string → spec dict, or None.

    Grammar (case-insensitive):
        "daily" / "daily:3"                       — every N days
        "weekly" / "weekly:MO,WE,FR" / "weekly:MO:2"
        "monthly" / "monthly:15" / "monthly:-1"    — day of month (-1 = last)
        "monthly:3:15"                            — every 3rd month, 15th
      Terminators: ";until=2026-12-31 23:59" / ";count=12"
    """
    if not rule:
        return None
    head, _, term = str(rule).strip().lower().partition(";")
    parts = head.split(":")
    freq = parts[0]
    if freq not in ("daily", "weekly", "monthly"):
        return None
    spec: dict[str, Any] = {"freq": freq, "interval": 1,
                            "byweekday": [], "bymonthday": 0}
    try:
        if freq == "daily":
            if len(parts) > 1 and parts[1]:
                spec["interval"] = max(1, min(365, int(parts[1])))
        elif freq == "weekly":
            days: list[int] = []
            for tok in (parts[1].split(",") if len(parts) > 1 and parts[1] else []):
                tok = tok.strip().upper()
                if tok in _WEEKDAYS:
                    days.append(_WEEKDAYS[tok])
                elif tok.isdigit():
                    spec["interval"] = max(1, min(52, int(tok)))
            spec["byweekday"] = sorted(set(days))
        else:  # monthly
            rest = parts[1:]
            nums = []
            for tok in rest:
                tok = tok.strip()
                if re.fullmatch(r"-?\d+", tok):
                    nums.append(int(tok))
            if len(nums) == 1:
                # monthly:15 → the 15th; monthly:-1 → last day
                spec["bymonthday"] = max(-31, min(31, nums[0]))
            elif len(nums) >= 2:
                # monthly:3:15 → every 3rd month on the 15th
                spec["interval"] = max(1, min(12, nums[0]))
                spec["bymonthday"] = max(-31, min(31, nums[1]))
        if term:
            m = re.fullmatch(r"until=(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}):(\d{2}))?",
                             term.strip())
            if m:
                spec["until"] = time.mktime((
                    int(m.group(1)), int(m.group(2)), int(m.group(3)),
                    int(m.group(4) or 23), int(m.group(5) or 59), 0, 0, 0, -1))
            m2 = re.fullmatch(r"count=(\d{1,4})", term.strip())
            if m2:
                spec["count"] = max(1, min(500, int(m2.group(1))))
    except (ValueError, OverflowError):
        return None
    return spec


def describe_recurrence(rule: str) -> str:
    """Human text for a recurrence rule: 'every Mon, Wed'."""
    spec = parse_recurrence(rule)
    if not spec:
        return "weekly" if rule == "weekly" else ""
    iv = spec["interval"]
    every = "" if iv == 1 else f" {iv}"
    if spec["freq"] == "daily":
        return f"every{every} day" if iv == 1 else f"every{every} days"
    if spec["freq"] == "weekly":
        days = ", ".join(_WEEKDAY_NAMES[d] for d in spec["byweekday"]) or "week"
        return f"every{every} week ({days})" if spec["byweekday"] else f"every{every} week"
    md = spec["bymonthday"]
    day = "last day" if md == -1 else f"the {abs(md)}th"
    return f"every{every} month ({day})" if md else f"every{every} month"


def _month_len(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def _add_months(year: int, month: int, n: int) -> tuple[int, int]:
    total = (year * 12 + (month - 1)) + n
    return total // 12, total % 12 + 1


def occurrences_after(starts_at: float, spec: dict[str, Any],
                      exdates: list[float] | None = None,
                      after: float | None = None, n: int = 4) -> list[float]:
    """Next ``n`` occurrences strictly after ``after``. Never raises.

    ``count``/``until`` bound the series from its origin (RFC 5545-style):
    a series exhausted by its terminator yields nothing, even when the
    caller asks for dates far in the future.
    """
    out: list[float] = []
    try:
        if not spec or n <= 0:
            return out
        now = after if after is not None else time.time()
        skip: set[tuple[int, int, int]] = set()
        for x in (exdates or []):
            t = time.localtime(x)
            skip.add((t.tm_year, t.tm_mon, t.tm_mday))
        until = spec.get("until", 0.0) or 0.0
        count = spec.get("count", 0) or 0
        day = time.localtime(starts_at)
        hour, minute = day.tm_hour, day.tm_min

        def emit(y: int, mo: int, d: int) -> float:
            return time.mktime((y, mo, d, hour, minute, 0, 0, 0, -1))

        freq, iv = spec["freq"], spec["interval"]
        if until and starts_at > until:
            return out  # series ends before it begins
        total = 1  # DTSTART is occurrence #1 (RFC 5545)
        if count and total > count:
            return out
        done = False

        def consider(occ: float) -> bool:
            """Feed one candidate. Returns True when collection is finished."""
            nonlocal total, done
            if occ <= starts_at or done:
                return done
            t = time.localtime(occ)
            if (t.tm_year, t.tm_mon, t.tm_mday) in skip:
                return False  # excluded: doesn't count against the series
            total += 1
            if count and total > count:
                done = True
                return True
            if until and occ > until:
                done = True
                return True
            if occ > now:
                out.append(occ)
            return len(out) >= n

        if freq == "daily":
            cur = starts_at
            guard = 0
            while not done and len(out) < n and guard < 4000:
                guard += 1
                cur += iv * 86400.0
                if consider(cur):
                    break
        elif freq == "weekly":
            wdays = spec["byweekday"] or [time.localtime(starts_at).tm_wday]
            base_date = time.mktime((day.tm_year, day.tm_mon, day.tm_mday,
                                     hour, minute, 0, 0, 0, -1))
            base_wd = time.localtime(base_date).tm_wday
            seen: set[float] = set()
            week = 0
            outer_done = False
            while not outer_done and len(out) < n and week < 520:
                for wd in wdays:
                    delta = (wd - base_wd) % 7
                    occ = base_date + delta * 86400.0 + week * iv * 7 * 86400.0
                    if occ in seen:
                        continue
                    seen.add(occ)
                    if consider(occ):
                        outer_done = True
                        break
                week += 1
        else:  # monthly
            md = spec["bymonthday"] or day.tm_mday
            y, mo = day.tm_year, day.tm_mon
            k = 0
            while not done and len(out) < n and k < 600:
                yy, mm = _add_months(y, mo, k * iv)
                last = _month_len(yy, mm)
                d = last if md == -1 else min(abs(md), last)
                if consider(emit(yy, mm, d)):
                    break
                k += 1
        out.sort()
        return out[:n]
    except Exception:  # noqa: BLE001 — engine never raises
        return out


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
    recurrence: str = ""  # "" | "weekly" (legacy) | "daily[:N]" | "weekly:MO,WE" | "monthly…"
    weekday: int = -1  # 0=Mon..6=Sun when recurrence == "weekly" (legacy)
    rsvps: dict[str, str] = field(default_factory=dict)  # member → yes/no/maybe/waitlist
    calendar_ref: str = ""  # provider's event id/link, "" when not synced
    reminders_sent: list[float] = field(default_factory=list)  # windows already sent
    summary: str = ""  # post-event summary text
    # Luma/Meetup-grade fields — defaulted, old JSON loads fine.
    description: str = ""
    ends_at: float = 0.0
    link: str = ""
    capacity: int = 0  # 0 = unlimited
    rsvp_opens_at: float = 0.0
    rsvp_closes_at: float = 0.0
    waitlisted: list[str] = field(default_factory=list)  # FIFO queue
    guests: dict[str, int] = field(default_factory=dict)  # member → +N guests
    proxy: dict[str, str] = field(default_factory=dict)  # member → host who added them
    exdates: list[float] = field(default_factory=list)  # cancelled dates (recurring)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CommunityEvent":
        def _f(key: str) -> float:
            try:
                return float(data.get(key) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        wd = data.get("weekday")
        return cls(
            id=str(data.get("id", "")),
            group_id=str(data.get("group_id", "")),
            title=str(data.get("title", "")),
            starts_at=_f("starts_at"),
            where=str(data.get("where", "")),
            created_by=str(data.get("created_by", "")),
            created_at=_f("created_at"),
            recurrence=str(data.get("recurrence", "")),
            weekday=int(wd) if wd is not None else -1,
            rsvps=dict(data.get("rsvps") or {}),
            calendar_ref=str(data.get("calendar_ref", "")),
            reminders_sent=list(data.get("reminders_sent") or []),
            summary=str(data.get("summary", "")),
            description=str(data.get("description", "")),
            ends_at=_f("ends_at"),
            link=str(data.get("link", "")),
            capacity=max(0, int(data.get("capacity") or 0)),
            rsvp_opens_at=_f("rsvp_opens_at"),
            rsvp_closes_at=_f("rsvp_closes_at"),
            waitlisted=[str(m) for m in (data.get("waitlisted") or [])],
            guests={str(k): max(0, int(v)) for k, v in (data.get("guests") or {}).items()},
            proxy={str(k): str(v) for k, v in (data.get("proxy") or {}).items()},
            exdates=[float(x) for x in (data.get("exdates") or [])],
        )

    def attendees(self) -> list[str]:
        return [m for m, r in self.rsvps.items() if r == RSVP_YES]

    def seats_used(self) -> int:
        """Seats taken, counting +N guests."""
        return sum(1 + self.guests.get(m, 0)
                   for m, r in self.rsvps.items() if r == RSVP_YES)

    def seats_left(self) -> int | None:
        if self.capacity <= 0:
            return None
        return max(0, self.capacity - self.seats_used())

    def rsvp_open(self, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        if self.rsvp_opens_at and now < self.rsvp_opens_at:
            return False
        if self.rsvp_closes_at and now > self.rsvp_closes_at:
            return False
        return True

    def is_recurring(self) -> bool:
        return bool(self.recurrence) and (
            self.recurrence == "weekly" or parse_recurrence(self.recurrence) is not None)

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

    def _mutate(self, event_id: str, fn) -> CommunityEvent | None:
        events = self._load()
        for ev in events:
            if ev.id == event_id:
                try:
                    fn(ev)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("event mutation failed: %s", exc)
                    return None
                self._save(events)
                return ev
        return None

    # ── lifecycle ────────────────────────────────────────────────────

    def create_event(self, group_id: str, title: str, starts_at: float,
                     where: str = "", created_by: str = "",
                     recurrence: str = "", weekday: int = -1,
                     **kwargs: Any) -> CommunityEvent | None:
        """Create an event. Syncs to the calendar provider when wired.

        Extra kwargs: ``description``, ``ends_at``, ``link``,
        ``capacity`` (int), ``rsvp_opens_at``, ``rsvp_closes_at``.
        ``recurrence``: "" | "weekly" (legacy) | engine rules
        ("daily[:N]", "weekly:MO,WE[:N]", "monthly[:day][:N]",
        ";until=…"/";count=…" suffixes).
        """
        title = (title or "").strip()
        if not title or starts_at <= 0:
            return None
        recurrence = (recurrence or "").strip().lower()
        if recurrence == "weekly" or (recurrence and parse_recurrence(recurrence)):
            pass  # valid
        elif recurrence:
            return None  # unknown rule — fail loud, not silent
        now = time.time()
        ev = CommunityEvent(
            id="evt_" + new_short_id(),
            group_id=group_id,
            title=title[:120],
            starts_at=starts_at,
            where=(where or "").strip()[:200],
            created_by=created_by,
            created_at=now,
            recurrence=recurrence if recurrence == "weekly" or parse_recurrence(recurrence or "") else "",
            weekday=weekday if 0 <= weekday <= 6 else -1,
            description=str(kwargs.get("description", ""))[:2000],
            ends_at=float(kwargs.get("ends_at") or 0.0),
            link=str(kwargs.get("link", ""))[:300],
            capacity=max(0, int(kwargs.get("capacity") or 0)),
            rsvp_opens_at=float(kwargs.get("rsvp_opens_at") or 0.0),
            rsvp_closes_at=float(kwargs.get("rsvp_closes_at") or 0.0),
        )
        ref = self._sync_calendar(ev)
        if ref:
            ev.calendar_ref = ref
        events = self._load()
        events.append(ev)
        self._save(events)
        return ev

    def edit_event(self, event_id: str, **fields: Any) -> CommunityEvent | None:
        """Edit title/where/description/ends_at/link/capacity/windows."""
        allowed = {"title", "where", "description", "ends_at", "link",
                   "capacity", "rsvp_opens_at", "rsvp_closes_at"}

        def _edit(ev: CommunityEvent) -> None:
            for k, v in fields.items():
                if k in allowed and v is not None:
                    if k in ("title",):
                        setattr(ev, k, str(v).strip()[:120] or ev.title)
                    elif k in ("where",):
                        setattr(ev, k, str(v).strip()[:200])
                    elif k in ("description",):
                        setattr(ev, k, str(v).strip()[:2000])
                    elif k in ("link",):
                        setattr(ev, k, str(v).strip()[:300])
                    elif k == "capacity":
                        ev.capacity = max(0, int(v))
                    else:
                        setattr(ev, k, float(v))

        return self._mutate(event_id, _edit)

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

    def my_events(self, member: str, upcoming_only: bool = True,
                  now: float | None = None) -> list[CommunityEvent]:
        """Events a member RSVP'd yes to — 'Events I attended' (wordcamp)."""
        now = now if now is not None else time.time()
        out = [e for e in self._load() if e.rsvps.get(member) == RSVP_YES]
        if upcoming_only:
            out = [e for e in out if not e.is_past(now)]
        return sorted(out, key=lambda e: e.starts_at)

    # ── RSVP ─────────────────────────────────────────────────────────

    def rsvp(self, event_id: str, member: str, choice: str,
             guests: int = 0, now: float | None = None) -> str:
        """Record an RSVP. Returns "yes"|"no"|"maybe"|"waitlist"|"".

        Full house + "yes" → waitlist (FIFO), honest about it.
        ``guests`` = +N guests, counts against capacity.
        """
        choice = (choice or "").lower()
        if choice not in (RSVP_YES, RSVP_NO, RSVP_MAYBE) or not member:
            return ""
        now = now if now is not None else time.time()
        outcome: dict[str, Any] = {"choice": ""}

        def _rsvp(ev: CommunityEvent) -> None:
            if not ev.rsvp_open(now):
                outcome["choice"] = ""
                return
            guests_n = max(0, min(20, int(guests or 0)))
            prev = ev.rsvps.get(member)
            if prev == RSVP_YES and choice == RSVP_YES:
                # repeat yes → just update the guest count
                ev.guests[member] = guests_n
                outcome["choice"] = RSVP_YES
                return
            if prev == RSVP_WAITLIST and choice != RSVP_YES:
                # leaving the waitlist
                if member in ev.waitlisted:
                    ev.waitlisted.remove(member)
            if choice == RSVP_YES:
                left = ev.seats_left()
                need = 1 + guests_n - (1 + ev.guests.get(member, 0) if prev == RSVP_YES else 0)
                if left is not None and need > 0 and ev.seats_used() + need > ev.capacity:
                    ev.rsvps[member] = RSVP_WAITLIST
                    if member not in ev.waitlisted:
                        ev.waitlisted.append(member)
                    outcome["choice"] = RSVP_WAITLIST
                    return
                if member in ev.waitlisted:
                    ev.waitlisted.remove(member)
            ev.rsvps[member] = choice
            ev.guests[member] = guests_n
            # self-RSVP attaches to a proxy record (Partiful manual-add pattern)
            ev.proxy.pop(member, None)
            outcome["choice"] = choice

        ev = self._mutate(event_id, _rsvp)
        return outcome["choice"] if ev else ""

    def rsvp_for(self, event_id: str, member: str, choice: str,
                 by: str = "") -> bool:
        """Host proxy RSVP — for off-platform replies. Attaches on self-RSVP."""
        choice = (choice or "").lower()
        if choice not in (RSVP_YES, RSVP_NO, RSVP_MAYBE) or not member:
            return False

        def _proxy(ev: CommunityEvent) -> None:
            ev.rsvps[member] = choice
            ev.proxy[member] = (by or "").strip()[:60] or "host"

        return self._mutate(event_id, _proxy) is not None

    def cancel_rsvp(self, event_id: str, member: str) -> str:
        """Withdraw an RSVP. Auto-promotes the waitlist head → returns them."""
        promoted = ""

        def _cancel(ev: CommunityEvent) -> None:
            nonlocal promoted
            if ev.rsvps.get(member) == RSVP_WAITLIST and member in ev.waitlisted:
                ev.waitlisted.remove(member)
            ev.rsvps.pop(member, None)
            ev.guests.pop(member, None)
            ev.proxy.pop(member, None)
            if ev.waitlisted:
                nxt = ev.waitlisted.pop(0)
                ev.rsvps[nxt] = RSVP_YES
                promoted = nxt

        ev = self._mutate(event_id, _cancel)
        return promoted if ev else ""

    def waitlist(self, event_id: str) -> list[str]:
        ev = self.get(event_id)
        return list(ev.waitlisted) if ev else []

    def rsvp_counts(self, event_id: str) -> dict[str, int]:
        ev = self.get(event_id)
        counts = {"yes": 0, "no": 0, "maybe": 0}
        if ev:
            for r in ev.rsvps.values():
                if r == RSVP_WAITLIST:
                    counts["waitlist"] = counts.get("waitlist", 0) + 1
                else:
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

        def _sum(ev: CommunityEvent) -> None:
            if ev.is_past(now):
                ev.summary = (text or "").strip()[:2000]

        ev = self._mutate(event_id, _sum)
        return ev if ev and ev.summary else None

    def cancel_event(self, event_id: str) -> bool:
        events = self._load()
        kept = [e for e in events if e.id != event_id]
        if len(kept) == len(events):
            return False
        self._save(kept)
        return True

    def skip_occurrence(self, event_id: str, date_ts: float) -> bool:
        """Add a date exception (holiday skip) to a recurring event."""

        def _skip(ev: CommunityEvent) -> None:
            if ev.is_recurring() and date_ts not in ev.exdates:
                ev.exdates.append(float(date_ts))

        return self._mutate(event_id, _skip) is not None

    # ── rituals: recurrence ──────────────────────────────────────────

    def next_occurrences(self, event_id: str, n: int = 4,
                         now: float | None = None) -> list[float]:
        """Future start times for a recurring ritual event."""
        now = now if now is not None else time.time()
        ev = self.get(event_id)
        if ev is None or not ev.is_recurring():
            return []
        if ev.recurrence == "weekly":
            return self._weekly_occurrences(ev, n, now)
        spec = parse_recurrence(ev.recurrence)
        if not spec:
            return []
        if not spec["byweekday"] and ev.weekday >= 0:
            spec = dict(spec, byweekday=[ev.weekday])
        return occurrences_after(ev.starts_at, spec,
                                 exdates=ev.exdates, after=now, n=n)

    def _weekly_occurrences(self, ev: CommunityEvent, n: int,
                            now: float) -> list[float]:
        """Legacy weekly engine — unchanged behavior."""
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
        """Materialize the next ``n`` instances of a ritual."""
        now = now if now is not None else time.time()
        ev = self.get(event_id)
        if ev is None or not ev.is_recurring():
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
                description=ev.description,
                link=ev.link,
                capacity=ev.capacity,
            )
            instances.append(inst)
        events.extend(instances)
        self._save(events)
        return instances

    # ── reminders ────────────────────────────────────────────────────

    def check_reminders(self, now: float | None = None) -> list[dict[str, Any]]:
        """Reminder payloads for events entering a remind window.

        Returns ``[{event_id, window, text, members, promoted}]``. The host
        delivers; the store never sends anything itself.
        """
        now = now if now is not None else time.time()
        due: list[dict[str, Any]] = []
        events = self._load()
        changed = False
        # (window, fires-only-when-delta-above): the legacy 24h/1h windows
        # are cumulative (a 30-min-out event fires both); the 7-day window
        # is a genuine week-ahead heads-up and fires only then.
        windows = ((7 * 86400.0, 86400.0), (86400.0, 0.0), (3600.0, 0.0))
        for ev in events:
            if ev.is_past(now):
                continue
            delta = ev.starts_at - now
            for window, above in windows:
                if window not in ev.reminders_sent and above < delta <= window:
                    if window > 2 * 86400:
                        label = "in a week"
                    elif window > 7200:
                        label = "tomorrow"
                    else:
                        label = "in an hour"
                    members = ev.attendees() or list(ev.rsvps)
                    guest_n = sum(ev.guests.get(m, 0) for m in ev.attendees())
                    wl = f" · {len(ev.waitlisted)} waitlisted" if ev.waitlisted else ""
                    cap = ""
                    if ev.capacity:
                        cap = f" ({ev.seats_used()}/{ev.capacity} seats)"
                    due.append({
                        "event_id": ev.id,
                        "window": window,
                        "text": (f"⏰ Reminder: **{ev.title}** starts {label} "
                                 f"({ev.where or 'location TBD'})."
                                 f"{cap} {len(ev.attendees())} going"
                                 f"{f' +{guest_n} guests' if guest_n else ''}{wl}."),
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
        ritual = ""
        if ev.is_recurring():
            ritual = f" 🔁 {describe_recurrence(ev.recurrence) or 'recurring'}"
        cap = ""
        if ev.capacity:
            left = ev.seats_left()
            bar = "█" * round(8 * ev.seats_used() / ev.capacity) + \
                "░" * (8 - round(8 * ev.seats_used() / ev.capacity))
            cap = f"\n   🎟️ {bar} {ev.seats_used()}/{ev.capacity} seats" + \
                  (f" · {left} left" if left else " · FULL")
            if ev.waitlisted:
                cap += f" · ⏳ {len(ev.waitlisted)} waitlisted"
        end = ""
        if ev.ends_at and ev.ends_at > ev.starts_at:
            end = f" → {time.strftime('%H:%M', time.localtime(ev.ends_at))}"
        desc = f"\n   {ev.description[:160]}" if ev.description else ""
        link = f"\n   🔗 {ev.link}" if ev.link else ""
        return (f"📍 **{ev.title}** `{ev.id}`{ritual}{cal}\n"
                f"   {when}{end} · {ev.where or 'location TBD'} · {yes} going"
                f"{desc}{link}{cap}")


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
    m = re.fullmatch(
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
                "[desc:…] | [cap:20] | [link:…] | [recur:daily[:2]] | [weekly]\n"
                "/cgroup event rsvp <event_id> <yes|no|maybe> [+N guests] · "
                "list [group_id] · show <event_id> · summary <event_id> <text> · "
                "ritual <event_id> <n> · cancel <event_id> · nudge <event_id> · "
                "my · waitlist <event_id> · edit <event_id> <field:value> …")
    cmd, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else "")

    if cmd == "new":
        segs = [s.strip() for s in arg.split("|")]
        if len(segs) < 3:
            return "Usage: /cgroup event new <group_id> | <title> | <YYYY-MM-DD HH:MM> | [where] | [extras…]"
        starts = _parse_when(segs[2])
        if not starts:
            return "Date format: YYYY-MM-DD HH:MM (e.g. 2026-10-10 18:00)"
        extras = [s for s in segs[3:] if s]
        recurrence = ""
        rrule = next((s[6:] for s in extras if s.lower().startswith("recur:")), "")
        if rrule:
            spec = parse_recurrence(rrule)
            if spec is None:
                return (f"Couldn't parse recurrence {rrule!r} — try daily[:N], "
                        "weekly[:MO,WE][:N], monthly[:day][:N].")
            recurrence = rrule.lower()
        elif any("weekly" in s.lower() for s in extras):
            recurrence = "weekly"
        weekday = time.localtime(starts).tm_wday if recurrence else -1
        where = next((s for s in extras
                      if not s.lower().startswith(("weekly", "recur:", "cap:",
                                                   "desc:", "link:"))), "")
        desc = next((s[5:] for s in extras if s.lower().startswith("desc:")), "")
        link = next((s[5:] for s in extras if s.lower().startswith("link:")), "")
        cap_raw = next((s[4:] for s in extras if s.lower().startswith("cap:")), "")
        ev = store.create_event(segs[0], segs[1], starts,
                                where=where, created_by=who,
                                recurrence=recurrence, weekday=weekday,
                                description=desc, link=link,
                                capacity=int(cap_raw) if cap_raw.isdigit() else 0)
        if ev is None:
            return "Couldn't create the event."
        note = "" if ev.calendar_ref else " (calendar not connected — sync skipped)"
        return f"📍 Event created{note}:\n{EventStore.format_event(ev)}"

    if cmd == "rsvp":
        segs = arg.split()
        if len(segs) < 2:
            return "Usage: /cgroup event rsvp <event_id> <yes|no|maybe> [+N]"
        guests = 0
        for tok in segs[2:]:
            if tok.startswith("+") and tok[1:].isdigit():
                guests = int(tok[1:])
        result = store.rsvp(segs[0], who, segs[1], guests=guests)
        if not result:
            return "RSVP failed — check the event id and choice (yes/no/maybe)."
        if result == RSVP_WAITLIST:
            pos = len(store.waitlist(segs[0]))
            return (f"⏳ Event is full — you're #{pos} on the waitlist. "
                    "We'll ping you if a seat opens.")
        counts = store.rsvp_counts(segs[0])
        wl = f" · ⏳ {counts['waitlist']} waitlisted" if counts["waitlist"] else ""
        return (f"RSVP recorded: {segs[1].lower()}{' +' + str(guests) if guests else ''}. "
                f"✅ {counts['yes']} · ❌ {counts['no']} · ❔ {counts['maybe']}{wl}")

    if cmd == "unrsvp":
        promoted = store.cancel_rsvp(arg.strip(), who)
        msg = "RSVP withdrawn."
        if promoted:
            msg += f" 🎉 {promoted} was promoted from the waitlist!"
        return msg

    if cmd == "add":
        # host proxy RSVP for off-platform replies
        segs = arg.split()
        if len(segs) < 3:
            return "Usage: /cgroup event add <event_id> <member> <yes|no|maybe>"
        ok = store.rsvp_for(segs[0], segs[1], segs[2], by=who)
        return "Added for them. 📝" if ok else "Couldn't add that RSVP."

    if cmd == "list":
        events = store.list_events(arg.strip())
        if not events:
            return "No upcoming events."
        lines = ["📍 **Upcoming events:**"]
        lines.extend(EventStore.format_event(e) for e in events[:10])
        return "\n".join(lines)

    if cmd == "my":
        events = store.my_events(who)
        if not events:
            return "You haven't RSVP'd to anything upcoming."
        lines = ["📍 **Your events:**"]
        lines.extend(EventStore.format_event(e) for e in events[:10])
        return "\n".join(lines)

    if cmd == "show":
        ev = store.get(arg.strip())
        if ev is None:
            return "No such event."
        lines = [EventStore.format_event(ev), "RSVPs:"]
        for m, r in ev.rsvps.items():
            g = f" +{ev.guests[m]}" if ev.guests.get(m) else ""
            px = f" (via {ev.proxy[m]})" if m in ev.proxy else ""
            lines.append(f"  {m}{g}: {r}{px}")
        if ev.waitlisted:
            lines.append(f"⏳ Waitlist: {', '.join(ev.waitlisted)}")
        if ev.summary:
            lines.append(f"Summary: {ev.summary[:300]}")
        if ev.recurrence:
            lines.append(f"🔁 Repeats: {describe_recurrence(ev.recurrence) or ev.recurrence}")
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
            return "Couldn't expand — needs a recurring event id."
        return f"🔁 Created {len(instances)} upcoming instances."

    if cmd == "skip":
        segs = arg.split(None, 1)
        if len(segs) < 2:
            return "Usage: /cgroup event skip <event_id> <YYYY-MM-DD>"
        day = _parse_when(segs[1].strip() + " 00:00")
        if not day or not store.skip_occurrence(segs[0], day):
            return "Couldn't skip — needs a recurring event id and a valid date."
        return "⏭️ That date is skipped for this series."

    if cmd == "edit":
        eid, _, rest_args = arg.partition(" ")
        fields: dict[str, Any] = {}
        for chunk in rest_args.split("|"):
            k, _, v = chunk.partition(":")
            k, v = k.strip().lower(), v.strip()
            if k in ("title", "where", "description", "desc", "link"):
                fields["description" if k == "desc" else k] = v
            elif k in ("capacity", "cap"):
                fields["capacity"] = int(v) if v.isdigit() else 0
            elif k == "ends":
                fields["ends_at"] = _parse_when(v)
        if not eid or not fields:
            return "Usage: /cgroup event edit <event_id> <title:…> | <where:…> | <desc:…> | <cap:20> | <ends:YYYY-MM-DD HH:MM>"
        ev = store.edit_event(eid.strip(), **fields)
        return "Event updated. ✏️" if ev else "Couldn't edit that event."

    if cmd == "cancel":
        return "Event cancelled." if store.cancel_event(arg.strip()) else "No such event."

    if cmd == "waitlist":
        wl = store.waitlist(arg.strip())
        return f"⏳ Waitlist ({len(wl)}): {', '.join(wl)}" if wl else "Waitlist is empty."

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

    return ("Unknown event command. Try: new · rsvp · unrsvp · add · list · my · "
            "show · summary · ritual · skip · edit · cancel · waitlist · nudge")
