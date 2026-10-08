"""IRL meetup coordination: venue polls → attendees → day-of → check-in.

#84 — the meetup layer on top of ``events.py``. Venue polls pick
where, the attendee list comes from yes-RSVPs, day-of reminders go
out the morning of, and check-in records who actually showed up
(feeds the post-event summary and future nudges).

Isolation: stdlib + ``nomorals.core`` only. Never raises.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger
from .events import EventStore

_log = get_logger("nomorals.community.meetups")

_DATA_DIR = Path.home() / ".devon" / "community" / "meetups"
_DAY_OF_WINDOW = 6 * 3600.0  # remind within 6h before start


@dataclass
class VenuePoll:
    """Where should we meet? Options + per-member votes."""

    id: str
    event_id: str
    options: list[str] = field(default_factory=list)
    votes: dict[str, str] = field(default_factory=dict)  # member → option
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VenuePoll":
        return cls(
            id=str(data.get("id", "")),
            event_id=str(data.get("event_id", "")),
            options=[str(o) for o in (data.get("options") or [])],
            votes=dict(data.get("votes") or {}),
            created_at=float(data.get("created_at") or 0.0),
        )

    def winner(self) -> str:
        counts: dict[str, int] = {}
        for opt in self.votes.values():
            counts[opt] = counts.get(opt, 0) + 1
        if not counts:
            return ""
        return max(counts.items(), key=lambda kv: kv[1])[0]


@dataclass
class CheckIn:
    """Who actually showed up."""

    id: str
    event_id: str
    member: str
    checked_in_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckIn":
        return cls(
            id=str(data.get("id", "")),
            event_id=str(data.get("event_id", "")),
            member=str(data.get("member", "")),
            checked_in_at=float(data.get("checked_in_at") or 0.0),
        )


class MeetupStore:
    """JSON persistence for venue polls and check-ins. Never raises."""

    def __init__(self, data_dir: Path | str | None = None,
                 events: EventStore | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else _DATA_DIR
        self.events = events or EventStore()

    def _polls_path(self) -> Path:
        return self.data_dir / "venue_polls.json"

    def _checkins_path(self) -> Path:
        return self.data_dir / "checkins.json"

    def _load_polls(self) -> list[VenuePoll]:
        try:
            raw = json.loads(self._polls_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        out: list[VenuePoll] = []
        for item in raw if isinstance(raw, list) else []:
            try:
                p = VenuePoll.from_dict(item)
                if p.id and p.event_id:
                    out.append(p)
            except (TypeError, ValueError, AttributeError):
                continue
        return out

    def _save_polls(self, polls: list[VenuePoll]) -> None:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            tmp = self._polls_path().with_suffix(".tmp")
            tmp.write_text(
                json.dumps([p.to_dict() for p in polls],
                           ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
            tmp.replace(self._polls_path())
        except OSError as exc:
            _log.warning("venue-poll save failed: %s", exc)

    def _load_checkins(self) -> list[CheckIn]:
        try:
            raw = json.loads(self._checkins_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        out: list[CheckIn] = []
        for item in raw if isinstance(raw, list) else []:
            try:
                c = CheckIn.from_dict(item)
                if c.id and c.event_id:
                    out.append(c)
            except (TypeError, ValueError, AttributeError):
                continue
        return out

    def _save_checkins(self, checkins: list[CheckIn]) -> None:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            tmp = self._checkins_path().with_suffix(".tmp")
            tmp.write_text(
                json.dumps([c.to_dict() for c in checkins],
                           ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
            tmp.replace(self._checkins_path())
        except OSError as exc:
            _log.warning("checkin save failed: %s", exc)

    # ── venue polls ──────────────────────────────────────────────────

    def venue_poll(self, event_id: str, options: list[str]) -> VenuePoll | None:
        """Open a venue poll for an event. Needs ≥2 options."""
        options = [o.strip()[:120] for o in options if o and o.strip()]
        if self.events.get(event_id) is None or len(options) < 2:
            return None
        poll = VenuePoll(
            id="vpoll_" + new_short_id(),
            event_id=event_id,
            options=options,
            created_at=time.time(),
        )
        polls = self._load_polls()
        polls.append(poll)
        self._save_polls(polls)
        return poll

    def get_poll(self, poll_id: str) -> VenuePoll | None:
        for p in self._load_polls():
            if p.id == poll_id:
                return p
        return None

    def polls_for(self, event_id: str) -> list[VenuePoll]:
        return [p for p in self._load_polls() if p.event_id == event_id]

    def vote_poll(self, poll_id: str, member: str, option: str) -> bool:
        """Vote by option index or exact text. One vote per member."""
        polls = self._load_polls()
        for p in polls:
            if p.id == poll_id:
                choice = ""
                if option.isdigit():
                    idx = int(option)
                    if 0 <= idx < len(p.options):
                        choice = p.options[idx]
                elif option in p.options:
                    choice = option
                if not choice or not member:
                    return False
                p.votes[member] = choice
                self._save_polls(polls)
                return True
        return False

    def poll_results(self, poll_id: str) -> dict[str, int]:
        p = self.get_poll(poll_id)
        counts = {o: 0 for o in (p.options if p else [])}
        if p:
            for opt in p.votes.values():
                counts[opt] = counts.get(opt, 0) + 1
        return counts

    # ── attendees + day-of + check-in ────────────────────────────────

    def attendees(self, event_id: str) -> list[str]:
        """Everyone with a yes-RSVP — the attendee list."""
        ev = self.events.get(event_id)
        return ev.attendees() if ev else []

    def day_of_reminders(self, now: float | None = None) -> list[dict[str, Any]]:
        """Events starting within 6h → reminder payloads for attendees."""
        now = now if now is not None else time.time()
        due: list[dict[str, Any]] = []
        for ev in self.events.list_events(upcoming_only=True, now=now):
            delta = ev.starts_at - now
            if 0 < delta <= _DAY_OF_WINDOW:
                due.append({
                    "event_id": ev.id,
                    "text": (f"📍 **Today:** {ev.title} at "
                             f"{time.strftime('%H:%M', time.localtime(ev.starts_at))}, "
                             f"{ev.where or 'location TBD'}. See you there!"),
                    "members": ev.attendees(),
                })
        return due

    def check_in(self, event_id: str, member: str) -> bool:
        """Record that a member showed up. Idempotent per member."""
        if self.events.get(event_id) is None or not member:
            return False
        checkins = self._load_checkins()
        if any(c.event_id == event_id and c.member == member for c in checkins):
            return True
        checkins.append(CheckIn(
            id="ckin_" + new_short_id(),
            event_id=event_id,
            member=member,
            checked_in_at=time.time(),
        ))
        self._save_checkins(checkins)
        return True

    def checked_in(self, event_id: str) -> list[str]:
        return [c.member for c in self._load_checkins()
                if c.event_id == event_id]

    def turnout(self, event_id: str) -> dict[str, Any]:
        """Attendees vs check-ins — feeds the post-event summary."""
        attend = self.attendees(event_id)
        arrived = self.checked_in(event_id)
        return {
            "rsvp_yes": len(attend),
            "checked_in": len(arrived),
            "showed": sorted(set(arrived)),
            "no_shows": sorted(set(attend) - set(arrived)),
        }


def control(tail: str, context=None, chat=None,
            sender: str = "", sender_id: str = "") -> str:
    """``/cgroup meetup …`` subcommand handler. Never raises."""
    try:
        return _control(tail, context, chat, sender, sender_id)
    except Exception as exc:  # noqa: BLE001
        _log.warning("meetup control failed: %s", exc)
        return "Meetup handling hit a snag — try again."


def _control(tail: str, context, chat, sender: str, sender_id: str) -> str:
    store = MeetupStore()
    who = (sender or sender_id or "anon").strip() or "anon"
    rest = (tail or "").strip()
    if rest.lower().startswith("meetup"):
        rest = rest[6:].strip()
    parts = rest.split(None, 1)
    if not parts:
        return ("/cgroup meetup poll <event_id> | <venue 1> | <venue 2> […]\n"
                "/cgroup meetup vote <poll_id> <option#> · results <poll_id> · "
                "attendees <event_id> · checkin <event_id> · turnout <event_id>")
    cmd, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else "")

    if cmd == "poll":
        segs = [s.strip() for s in arg.split("|")]
        if len(segs) < 3:
            return "Usage: /cgroup meetup poll <event_id> | <venue 1> | <venue 2> […]"
        poll = store.venue_poll(segs[0], segs[1:])
        if poll is None:
            return "Poll failed — need a valid event id and at least 2 venues."
        opts = "\n".join(f"  {i}. {o}" for i, o in enumerate(poll.options))
        return (f"🗳️ Venue poll opened `{poll.id}`:\n{opts}\n"
                f"Vote: /cgroup meetup vote {poll.id} <option#>")

    if cmd == "vote":
        segs = arg.split()
        if len(segs) < 2:
            return "Usage: /cgroup meetup vote <poll_id> <option#>"
        return ("Vote recorded." if store.vote_poll(segs[0], who, segs[1])
                else "Vote failed — check the poll id and option number.")

    if cmd == "results":
        poll = store.get_poll(arg.strip())
        if poll is None:
            return "No such poll."
        counts = store.poll_results(poll.id)
        lines = [f"🗳️ **Venue poll** `{poll.id}`:"]
        for i, o in enumerate(poll.options):
            lines.append(f"  {i}. {o} — {counts.get(o, 0)} votes")
        if poll.winner():
            lines.append(f"🏆 Leading: {poll.winner()}")
        return "\n".join(lines)

    if cmd == "attendees":
        attend = store.attendees(arg.strip())
        return (f"👥 **Attendees** ({len(attend)}): {', '.join(attend) or 'none yet'}"
                if arg.strip() else "Usage: /cgroup meetup attendees <event_id>")

    if cmd == "checkin":
        return ("✅ Checked in. See you there!"
                if store.check_in(arg.strip(), who)
                else "Check-in failed — check the event id.")

    if cmd == "turnout":
        t = store.turnout(arg.strip())
        lines = [f"📊 **Turnout:** {t['checked_in']}/{t['rsvp_yes']} showed up"]
        if t["showed"]:
            lines.append("Showed: " + ", ".join(t["showed"]))
        if t["no_shows"]:
            lines.append("No-shows: " + ", ".join(t["no_shows"]))
        return "\n".join(lines)

    return "Unknown meetup command. Try: poll · vote · results · attendees · checkin · turnout"
