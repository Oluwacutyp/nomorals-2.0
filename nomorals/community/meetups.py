"""IRL meetup coordination: venue polls → attendees → day-of → check-in.

#84 — the meetup layer on top of ``events.py``. Venue polls pick
where, the attendee list comes from yes-RSVPs, day-of reminders go
out the morning of, and check-in records who actually showed up
(feeds the post-event summary and future nudges).

Doodle/Meetup-grade upgrades (mined 2026-10-10):
* **Approval voting** (Doodle): ``multi`` polls let each member vote for
  every venue they'd accept; winner = most approvals. Single-choice
  stays the default.
* **Open vs blind polls** (Doodle research: open polls get higher
  response rates): ``blind=True`` hides running tallies until closed.
* **Poll deadlines + auto-close**: ``closes_at``; ``due_polls()`` returns
  expired polls so the host can close and announce.
* **Rich venues**: options carry name + address + link (strings still work).
* **Winner adoption**: ``adopt_winner`` writes the winning venue into the
  event's ``where`` and returns a notify payload (Doodle's "choose the
  final date and notify" pattern).
* **Reliability score** (real Meetup attendance-guideline pattern):
  ``reliability(member)`` = check-ins − no-shows across all meetups;
  chronic no-shows are flagged in turnout.
* **Host roll-call**: ``host_check_in`` lets the host tick attendance
  (Meetup's "organiser marks them attended" pattern).
* **Instant meetups** (the HN "spontaneous happy hour" idea):
  ``quick_meetup`` creates event + venue poll + notify in one call.

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


# ── venue options ──────────────────────────────────────────────────────────


def _norm_option(opt: Any) -> dict[str, str]:
    """Option → {name, address, link}. Plain strings keep working."""
    if isinstance(opt, dict):
        return {
            "name": str(opt.get("name", "")).strip()[:120],
            "address": str(opt.get("address", "")).strip()[:200],
            "link": str(opt.get("link", "")).strip()[:300],
        }
    return {"name": str(opt).strip()[:120], "address": "", "link": ""}


def _option_name(opt: Any) -> str:
    if isinstance(opt, dict):
        return str(opt.get("name", "")).strip()
    return str(opt).strip()


@dataclass
class VenuePoll:
    """Where should we meet? Options + per-member votes."""

    id: str
    event_id: str
    options: list[Any] = field(default_factory=list)  # str or {name,address,link}
    votes: dict[str, str] = field(default_factory=dict)  # member → option (single)
    ballots: dict[str, list[str]] = field(default_factory=dict)  # member → options (approval)
    multi: bool = False  # approval voting: vote for every acceptable venue
    blind: bool = False  # hide tallies until closed
    closed: bool = False
    closes_at: float = 0.0  # deadline epoch; 0 = none
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VenuePoll":
        return cls(
            id=str(data.get("id", "")),
            event_id=str(data.get("event_id", "")),
            options=list(data.get("options") or []),
            votes=dict(data.get("votes") or {}),
            ballots={str(k): [str(x) for x in (v or [])]
                      for k, v in (data.get("ballots") or {}).items()},
            multi=bool(data.get("multi", False)),
            blind=bool(data.get("blind", False)),
            closed=bool(data.get("closed", False)),
            created_at=float(data.get("created_at") or 0.0),
            closes_at=float(data.get("closes_at") or 0.0),
        )

    def names(self) -> list[str]:
        return [_option_name(o) for o in self.options]

    def counts(self) -> dict[str, int]:
        """Tally by venue name. Single: one vote each. Multi: approvals."""
        counts = {n: 0 for n in self.names()}
        if self.multi:
            for ballot in self.ballots.values():
                for n in set(ballot):
                    if n in counts:
                        counts[n] += 1
        else:
            for n in self.votes.values():
                if n in counts:
                    counts[n] += 1
        return counts

    def winner(self) -> str:
        counts = self.counts()
        if not any(counts.values()):
            return ""
        # approval/single winner = most votes; ties → earliest option
        best = max(counts.values())
        for n in self.names():
            if counts[n] == best:
                return n
        return ""

    def expired(self, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        return bool(self.closes_at) and not self.closed and now >= self.closes_at


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

    def _mutate_poll(self, poll_id: str, fn) -> VenuePoll | None:
        polls = self._load_polls()
        for p in polls:
            if p.id == poll_id:
                try:
                    fn(p)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("poll mutation failed: %s", exc)
                    return None
                self._save_polls(polls)
                return p
        return None

    # ── venue polls ──────────────────────────────────────────────────

    def venue_poll(self, event_id: str, options: list[Any],
                   multi: bool = False, blind: bool = False,
                   closes_in_hours: float = 0.0) -> VenuePoll | None:
        """Open a venue poll for an event. Needs ≥2 options.

        ``multi`` = approval voting (Doodle pattern). ``blind`` hides
        tallies until closed. ``closes_in_hours`` sets a deadline.
        """
        normed = [_norm_option(o) for o in options or []]
        normed = [o for o in normed if o["name"]]
        if self.events.get(event_id) is None or len(normed) < 2:
            return None
        poll = VenuePoll(
            id="vpoll_" + new_short_id(),
            event_id=event_id,
            options=normed,
            multi=bool(multi),
            blind=bool(blind),
            closes_at=(time.time() + closes_in_hours * 3600.0
                       if closes_in_hours > 0 else 0.0),
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

    def due_polls(self, now: float | None = None) -> list[VenuePoll]:
        """Polls past their deadline but not closed — host should close them."""
        return [p for p in self._load_polls() if p.expired(now)]

    def _resolve(self, poll: VenuePoll, option: str) -> str:
        """Option index or exact name → venue name, or ''."""
        option = (option or "").strip()
        names = poll.names()
        if option.isdigit():
            idx = int(option)
            if 0 <= idx < len(names):
                return names[idx]
            return ""
        return option if option in names else ""

    def vote_poll(self, poll_id: str, member: str, option: str) -> bool:
        """Vote by option index or exact name. One vote per member."""
        if not member:
            return False

        def _vote(p: VenuePoll) -> None:
            if p.closed:
                raise PermissionError("closed")
            name = self._resolve(p, option)
            if not name:
                raise ValueError("bad option")
            if p.multi:
                # single vote on a multi poll = ballot of one
                p.ballots[member] = [name]
            else:
                p.votes[member] = name

        try:
            return self._mutate_poll(poll_id, _vote) is not None
        except (PermissionError, ValueError):
            return False

    def vote_approval(self, poll_id: str, member: str,
                      options: list[str]) -> bool:
        """Approval ballot: vote for every venue you'd accept (Doodle)."""
        if not member or not options:
            return False

        def _ballot(p: VenuePoll) -> None:
            if p.closed:
                raise PermissionError("closed")
            names = [self._resolve(p, o) for o in options]
            names = [n for n in names if n]
            if not names:
                raise ValueError("no valid options")
            p.ballots[member] = sorted(set(names))

        try:
            return self._mutate_poll(poll_id, _ballot) is not None
        except (PermissionError, ValueError):
            return False

    def close_poll(self, poll_id: str) -> VenuePoll | None:
        def _close(p: VenuePoll) -> None:
            p.closed = True

        return self._mutate_poll(poll_id, _close)

    def adopt_winner(self, poll_id: str) -> dict[str, Any]:
        """Write the winning venue into the event's ``where``.

        Returns {"ok", "venue"|"reason", "notify"}. The host announces
        the notify payload (Doodle's "choose the final date" pattern).
        """
        poll = self.get_poll(poll_id)
        if poll is None:
            return {"ok": False, "reason": "no such poll"}
        winner = poll.winner()
        if not winner:
            return {"ok": False, "reason": "no votes yet — nothing to adopt"}
        ev = self.events.get(poll.event_id)
        if ev is None:
            return {"ok": False, "reason": "event is gone"}
        detail = next((o for o in poll.options
                       if _option_name(o) == winner), {})
        addr = detail.get("address", "") if isinstance(detail, dict) else ""
        where = f"{winner} ({addr})" if addr else winner
        self.events.edit_event(ev.id, where=where)
        self.close_poll(poll_id)
        return {"ok": True, "venue": winner,
                "notify": (f"📍 Venue decided: **{winner}**"
                           + (f" — {addr}" if addr else "")
                           + f" for *{ev.title}*. See you there!")}

    def poll_results(self, poll_id: str) -> dict[str, int]:
        p = self.get_poll(poll_id)
        return p.counts() if p else {}

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

    def host_check_in(self, event_id: str, members: list[str],
                      by: str = "") -> int:
        """Host roll-call: tick attendance for several members at once."""
        n = 0
        for m in members or []:
            if m and self.check_in(event_id, m):
                n += 1
        return n

    def checked_in(self, event_id: str) -> list[str]:
        return [c.member for c in self._load_checkins()
                if c.event_id == event_id]

    def turnout(self, event_id: str) -> dict[str, Any]:
        """Attendees vs check-ins — feeds the post-event summary."""
        attend = self.attendees(event_id)
        arrived = self.checked_in(event_id)
        showed = sorted(set(arrived))
        no_shows = sorted(set(attend) - set(arrived))
        flags = {m: self.reliability(m) for m in no_shows}
        return {
            "rsvp_yes": len(attend),
            "checked_in": len(arrived),
            "showed": showed,
            "no_shows": no_shows,
            "reliability": flags,
        }

    def reliability(self, member: str) -> dict[str, Any]:
        """+1 per check-in, −1 per yes-RSVP no-show (real Meetup pattern).

        Computed from this module's own stores — never owner data.
        """
        shows = 0
        no_shows = 0
        try:
            for ev in self.events._load():
                if ev.rsvps.get(member) != "yes":
                    continue
                if member in set(self.checked_in(ev.id)):
                    shows += 1
                elif ev.is_past():
                    no_shows += 1
        except Exception:  # noqa: BLE001
            pass
        score = shows - no_shows
        label = ("🌟 reliable" if score >= 3 else "✅ solid" if score >= 1
                 else "⚠️ flaky" if score <= -1 else "🆕 new")
        return {"member": member, "score": score, "shows": shows,
                "no_shows": no_shows, "label": label}

    # ── instant meetups ──────────────────────────────────────────────

    def quick_meetup(self, group_id: str, title: str, starts_at: float,
                     venues: list[Any], created_by: str = "",
                     where_hint: str = "") -> dict[str, Any]:
        """One call: create event + open venue poll + notify payload.

        The spontaneous-happy-hour pattern — plans that used to die in
        chat threads become a real event with a vote attached.
        """
        ev = self.events.create_event(group_id, title, starts_at,
                                      where=where_hint, created_by=created_by)
        if ev is None:
            return {"ok": False, "reason": "couldn't create the event"}
        poll = self.venue_poll(ev.id, venues)
        when = time.strftime("%a %H:%M", time.localtime(starts_at))
        return {
            "ok": True,
            "event": ev,
            "poll": poll,
            "notify": (f"⚡ **Spontaneous meetup:** {title} — {when}, "
                       f"{where_hint or 'venue TBD'}.\n"
                       f"Vote where: /cgroup meetup vote {poll.id if poll else ''} <option#>\n"
                       f"RSVP: /cgroup event rsvp {ev.id} yes"),
        }


# ── rendering ──────────────────────────────────────────────────────────────


def render_poll(poll: VenuePoll, show_votes: bool = True) -> str:
    counts = poll.counts()
    total = sum(counts.values())
    mode = "approval" if poll.multi else "single-choice"
    vis = "🙈 blind" if poll.blind and not poll.closed else "👀 open"
    status = "🔒 closed" if poll.closed else f"🟢 open · {vis} · {mode}"
    if poll.closes_at and not poll.closed:
        left = poll.closes_at - time.time()
        status += f" · closes in {int(left // 3600)}h{int(left % 3600 // 60)}m" if left > 0 else " · overdue"
    lines = [f"🗳️ **Venue poll** `{poll.id}` [{status}]", ""]
    names = poll.names()
    for i, opt in enumerate(poll.options):
        name = _option_name(opt)
        addr = opt.get("address", "") if isinstance(opt, dict) else ""
        sub = f" — _{addr}_" if addr else ""
        if show_votes and not (poll.blind and not poll.closed):
            c = counts.get(name, 0)
            bar = "█" * round(10 * c / total) if total else ""
            lines.append(f"  {i}. **{name}**{sub} — {c} vote{'s' if c != 1 else ''} {bar}")
        else:
            lines.append(f"  {i}. **{name}**{sub}")
    if poll.winner():
        lines.append(f"\n🏆 Leading: **{poll.winner()}**")
    lines.append(f"\nVote: /cgroup meetup vote {poll.id} <option#>")
    if poll.multi:
        lines.append(f"Approval: /cgroup meetup approve {poll.id} <#> <#> …")
    return "\n".join(lines)


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
        return ("/cgroup meetup poll <event_id> | <venue 1> | <venue 2> […] [multi] [blind] [hours:N]\n"
                "/cgroup meetup vote <poll_id> <option#> · approve <poll_id> <#>…\n"
                "/cgroup meetup results <poll_id> · close <poll_id> · adopt <poll_id>\n"
                "/cgroup meetup attendees <event_id> · checkin <event_id> · rollcall <event_id> <m1,m2> · turnout <event_id>\n"
                "/cgroup meetup rep <member> · quick <group_id> | <title> | <YYYY-MM-DD HH:MM> | <venue1> | <venue2>…")
    cmd, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else "")

    if cmd == "poll":
        segs = [s.strip() for s in arg.split("|")]
        if len(segs) < 3:
            return "Usage: /cgroup meetup poll <event_id> | <venue 1> | <venue 2> […]"
        flags = {s.lower() for s in segs if s.lower() in ("multi", "blind")}
        hours = 0.0
        opts = []
        for s in segs[1:]:
            low = s.lower()
            if low in ("multi", "blind"):
                continue
            if low.startswith("hours:"):
                try:
                    hours = float(low[6:])
                except ValueError:
                    pass
                continue
            opts.append(s)
        poll = store.venue_poll(segs[0], opts, multi="multi" in flags,
                                blind="blind" in flags, closes_in_hours=hours)
        if poll is None:
            return "Poll failed — need a valid event id and at least 2 venues."
        return f"🗳️ Venue poll opened:\n\n{render_poll(poll)}"

    if cmd == "vote":
        segs = arg.split()
        if len(segs) < 2:
            return "Usage: /cgroup meetup vote <poll_id> <option#>"
        return ("Vote recorded." if store.vote_poll(segs[0], who, segs[1])
                else "Vote failed — check the poll id and option number.")

    if cmd == "approve":
        segs = arg.split()
        if len(segs) < 2:
            return "Usage: /cgroup meetup approve <poll_id> <option#> [<option#> …]"
        return ("Approval ballot recorded." if store.vote_approval(segs[0], who, segs[1:])
                else "Ballot failed — check the poll id and option numbers.")

    if cmd == "results":
        poll = store.get_poll(arg.strip())
        if poll is None:
            return "No such poll."
        return render_poll(poll)

    if cmd == "close":
        poll = store.close_poll(arg.strip())
        if poll is None:
            return "No such poll."
        return f"🔒 Poll closed.\n\n{render_poll(poll)}"

    if cmd == "adopt":
        res = store.adopt_winner(arg.strip())
        if not res["ok"]:
            return f"Couldn't adopt: {res['reason']}"
        return f"✅ {res['notify']}"

    if cmd == "attendees":
        attend = store.attendees(arg.strip())
        return (f"👥 **Attendees** ({len(attend)}): {', '.join(attend) or 'none yet'}"
                if arg.strip() else "Usage: /cgroup meetup attendees <event_id>")

    if cmd == "checkin":
        return ("✅ Checked in. See you there!"
                if store.check_in(arg.strip(), who)
                else "Check-in failed — check the event id.")

    if cmd == "rollcall":
        eid, _, names = arg.partition(" ")
        members = [m.strip() for m in names.split(",") if m.strip()]
        if not eid or not members:
            return "Usage: /cgroup meetup rollcall <event_id> <member1,member2,…>"
        n = store.host_check_in(eid.strip(), members, by=who)
        return f"📋 Roll-call: {n} checked in."

    if cmd == "rep":
        member = arg.strip() or who
        r = store.reliability(member)
        return (f"{r['label']} **{member}** — reliability {r['score']:+d} "
                f"({r['shows']} showed · {r['no_shows']} no-showed)")

    if cmd == "turnout":
        t = store.turnout(arg.strip())
        lines = [f"📊 **Turnout:** {t['checked_in']}/{t['rsvp_yes']} showed up"]
        if t["showed"]:
            lines.append("Showed: " + ", ".join(t["showed"]))
        if t["no_shows"]:
            flagged = []
            for m in t["no_shows"]:
                r = t["reliability"].get(m, {})
                flag = f" ({r.get('label', '')})" if r.get("score", 0) <= -2 else ""
                flagged.append(m + flag)
            lines.append("No-shows: " + ", ".join(flagged))
        return "\n".join(lines)

    if cmd == "quick":
        from .events import _parse_when
        segs = [s.strip() for s in arg.split("|")]
        if len(segs) < 5:
            return ("Usage: /cgroup meetup quick <group_id> | <title> | "
                    "<YYYY-MM-DD HH:MM> | <venue 1> | <venue 2> […]")
        starts = _parse_when(segs[2])
        if not starts:
            return "Date format: YYYY-MM-DD HH:MM"
        res = store.quick_meetup(segs[0], segs[1], starts, segs[3:], created_by=who)
        return res["notify"] if res["ok"] else f"Couldn't: {res['reason']}"

    return ("Unknown meetup command. Try: poll · vote · approve · results · "
            "close · adopt · attendees · checkin · rollcall · rep · turnout · quick")
