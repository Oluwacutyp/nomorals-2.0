"""Group travel stack: one chat that does it all — polls, shared expenses,
per-person budgets, per-traveler legs, proposals vs agreed items, and a
merged itinerary.

Build-map #74. The meta-finding: no single app owns group travel (the
stack is five apps). Devon IS the stack.

ISOLATION CONTRACT (mirrors ``nomorals.community`` #12):
* Group state lives under ``~/.devon/community/group_trips/`` — NEVER in
  the owner's database tables.
* Participants are identified only by the platform user id / display name
  that arrives with the chat message itself. No owner memory, no accounts,
  no vaults, no private connectors. This module imports stdlib +
  ``nomorals.core`` + ``nomorals.travel.itinerary`` (pure data classes and
  the confirmation parser — no owner state) only.
* The ``/gtrip`` command is group-chats only and is NOT owner-gated:
  every group member drives it. (``/trip`` stays owner-only for the
  owner's private itineraries — #72.)

Features:
* Polls (Troupe pattern): decision objects with deadlines, one vote per
  member, winner on demand.
* Expenses (Splitwise pattern): who paid / for whom, minimized settlement
  ("who owes whom" with the fewest transfers).
* Per-person budgets + spending tracking.
* Per-traveler legs (G8Trip steal): each person's flights planned
  separately, merged into one view.
* Event/idea duality (Plan Harmony): proposals and agreed items are
  different objects.
* Merged itinerary: all legs + agreed items in one readable view.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger
from .itinerary import Flight, parse_confirmation

_log = get_logger("nomorals.travel.groups")

__all__ = [
    "GroupTrip",
    "GroupPoll",
    "GroupExpense",
    "TripLeg",
    "TripProposal",
    "GroupTripStore",
    "settle_balances",
    "parse_naira_kobo",
    "merged_itinerary",
    "control_gtrip",
    "register",
]

#: Group-scoped state dir — the community namespace, never owner tables.
_DEFAULT_DIR = Path.home() / ".devon" / "community" / "group_trips"

_PROPOSAL_PROPOSED = "proposed"
_PROPOSAL_AGREED = "agreed"


def _sanitize_group_key(group_key: str) -> str:
    """Make a group key safe as a filename (platform:chat_id → platform_chat_id)."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", group_key).strip("_") or "group"


def _now() -> float:
    return time.time()


# ── data model ─────────────────────────────────────────────────────────────


@dataclass
class GroupTrip:
    """One group trip living in one group chat."""
    id: str
    name: str
    members: list[str] = field(default_factory=list)
    created_by: str = ""
    created_at: float = 0.0


@dataclass
class GroupPoll:
    """A decision object with a deadline (Troupe pattern)."""
    id: str
    trip_id: str
    question: str
    options: list[str] = field(default_factory=list)
    deadline: float = 0.0          # 0 = no deadline
    created_by: str = ""
    created_at: float = 0.0

    def is_closed(self, now: float | None = None) -> bool:
        now = now if now is not None else _now()
        return bool(self.deadline) and now >= self.deadline


@dataclass
class GroupExpense:
    """One shared expense (Splitwise pattern)."""
    id: str
    trip_id: str
    who_paid: str
    amount_kobo: int
    for_whom: list[str] = field(default_factory=list)  # empty = all members
    description: str = ""
    created_at: float = 0.0


@dataclass
class TripLeg:
    """One traveler's flight leg (G8Trip pattern)."""
    id: str
    trip_id: str
    member: str
    flight: dict = field(default_factory=dict)  # Flight.asdict()
    note: str = ""
    created_at: float = 0.0


@dataclass
class TripProposal:
    """A proposal (idea) vs an agreed item — different objects (Plan Harmony)."""
    id: str
    trip_id: str
    member: str
    idea: str
    status: str = _PROPOSAL_PROPOSED   # proposed | agreed
    created_at: float = 0.0


# ── store ──────────────────────────────────────────────────────────────────


class GroupTripStore:
    """SQLite-backed group trip store. Group-scoped, never raises on reads.

    One database file per group chat (``<sanitized group key>.db``) under
    ``~/.devon/community/group_trips/``.
    """

    def __init__(self, group_key: str, db_path: str = "") -> None:
        gkey = _sanitize_group_key(group_key or "group")
        path = db_path or str(_DEFAULT_DIR / f"{gkey}.db")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.group_key = gkey
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        self._init_schema()

    # -- schema ---------------------------------------------------------

    def _init_schema(self) -> None:
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS group_trips (
                id TEXT PRIMARY KEY, name TEXT NOT NULL,
                members_json TEXT NOT NULL DEFAULT '[]',
                created_by TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS polls (
                id TEXT PRIMARY KEY, trip_id TEXT NOT NULL,
                question TEXT NOT NULL, options_json TEXT NOT NULL DEFAULT '[]',
                deadline REAL NOT NULL DEFAULT 0,
                created_by TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS votes (
                poll_id TEXT NOT NULL, member TEXT NOT NULL,
                option_idx INTEGER NOT NULL,
                PRIMARY KEY (poll_id, member));
            CREATE TABLE IF NOT EXISTS expenses (
                id TEXT PRIMARY KEY, trip_id TEXT NOT NULL,
                who_paid TEXT NOT NULL, amount_kobo INTEGER NOT NULL,
                for_whom_json TEXT NOT NULL DEFAULT '[]',
                description TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS budgets (
                trip_id TEXT NOT NULL, member TEXT NOT NULL,
                amount_kobo INTEGER NOT NULL,
                PRIMARY KEY (trip_id, member));
            CREATE TABLE IF NOT EXISTS legs (
                id TEXT PRIMARY KEY, trip_id TEXT NOT NULL,
                member TEXT NOT NULL, flight_json TEXT NOT NULL DEFAULT '{}',
                note TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS proposals (
                id TEXT PRIMARY KEY, trip_id TEXT NOT NULL,
                member TEXT NOT NULL, idea TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'proposed',
                created_at REAL NOT NULL DEFAULT 0);
        """)
        self._db.commit()

    # -- trips ----------------------------------------------------------

    def create_trip(self, name: str, members: list[str],
                    created_by: str = "") -> GroupTrip:
        name = (name or "").strip() or "Group trip"
        seen: list[str] = []
        for m in members or []:
            m = (m or "").strip()
            if m and m not in seen:
                seen.append(m)
        if created_by and created_by not in seen:
            seen.append(created_by)
        trip = GroupTrip(id="gtrip_" + new_short_id(length=8), name=name,
                         members=seen, created_by=created_by,
                         created_at=_now())
        self._db.execute(
            "INSERT INTO group_trips (id, name, members_json, created_by,"
            " created_at) VALUES (?, ?, ?, ?, ?)",
            (trip.id, trip.name, json.dumps(trip.members),
             trip.created_by, trip.created_at))
        self._db.commit()
        return trip

    def get_trip(self, trip_id: str) -> GroupTrip | None:
        try:
            row = self._db.execute(
                "SELECT * FROM group_trips WHERE id = ?", (trip_id,)
            ).fetchone()
        except Exception:  # noqa: BLE001
            _log.debug("get_trip failed", exc_info=True)
            return None
        if row is None:
            return None
        return GroupTrip(id=row["id"], name=row["name"],
                         members=json.loads(row["members_json"] or "[]"),
                         created_by=row["created_by"],
                         created_at=row["created_at"])

    def list_trips(self) -> list[GroupTrip]:
        try:
            rows = self._db.execute(
                "SELECT * FROM group_trips ORDER BY created_at DESC").fetchall()
        except Exception:  # noqa: BLE001
            return []
        return [GroupTrip(id=r["id"], name=r["name"],
                          members=json.loads(r["members_json"] or "[]"),
                          created_by=r["created_by"],
                          created_at=r["created_at"]) for r in rows]

    def add_member(self, trip_id: str, member: str) -> GroupTrip | None:
        trip = self.get_trip(trip_id)
        if trip is None:
            return None
        member = (member or "").strip()
        if member and member not in trip.members:
            trip.members.append(member)
            self._db.execute(
                "UPDATE group_trips SET members_json = ? WHERE id = ?",
                (json.dumps(trip.members), trip.id))
            self._db.commit()
        return trip

    # -- polls ----------------------------------------------------------

    def create_poll(self, trip_id: str, question: str, options: list[str],
                    deadline: float = 0.0,
                    created_by: str = "") -> GroupPoll | None:
        if self.get_trip(trip_id) is None:
            return None
        opts = [o.strip() for o in (options or []) if o and o.strip()]
        if len(opts) < 2:
            return None
        poll = GroupPoll(id="poll_" + new_short_id(length=8), trip_id=trip_id,
                         question=(question or "").strip() or "Decision",
                         options=opts, deadline=float(deadline or 0.0),
                         created_by=created_by, created_at=_now())
        self._db.execute(
            "INSERT INTO polls (id, trip_id, question, options_json, deadline,"
            " created_by, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (poll.id, poll.trip_id, poll.question,
             json.dumps(poll.options), poll.deadline,
             poll.created_by, poll.created_at))
        self._db.commit()
        return poll

    def get_poll(self, poll_id: str) -> GroupPoll | None:
        try:
            row = self._db.execute(
                "SELECT * FROM polls WHERE id = ?", (poll_id,)).fetchone()
        except Exception:  # noqa: BLE001
            return None
        if row is None:
            return None
        return GroupPoll(id=row["id"], trip_id=row["trip_id"],
                         question=row["question"],
                         options=json.loads(row["options_json"] or "[]"),
                         deadline=row["deadline"],
                         created_by=row["created_by"],
                         created_at=row["created_at"])

    def vote(self, poll_id: str, member: str, option: str | int) -> bool:
        """Record one member's vote. Returns False when invalid/closed."""
        poll = self.get_poll(poll_id)
        if poll is None or poll.is_closed():
            return False
        idx = self._option_index(poll, option)
        if idx is None:
            return False
        member = (member or "").strip() or "anon"
        self._db.execute(
            "INSERT OR REPLACE INTO votes (poll_id, member, option_idx)"
            " VALUES (?, ?, ?)", (poll_id, member, idx))
        self._db.commit()
        return True

    @staticmethod
    def _option_index(poll: GroupPoll, option: str | int) -> int | None:
        if isinstance(option, int):
            return option if 0 <= option < len(poll.options) else None
        text = str(option or "").strip().lower()
        for i, opt in enumerate(poll.options):
            if opt.lower() == text or str(i + 1) == text:
                return i
        return None

    def poll_result(self, poll_id: str) -> dict[str, Any] | None:
        poll = self.get_poll(poll_id)
        if poll is None:
            return None
        try:
            rows = self._db.execute(
                "SELECT member, option_idx FROM votes WHERE poll_id = ?",
                (poll_id,)).fetchall()
        except Exception:  # noqa: BLE001
            rows = []
        counts = [0] * len(poll.options)
        voters: dict[str, int] = {}
        for r in rows:
            idx = r["option_idx"]
            if 0 <= idx < len(counts):
                counts[idx] += 1
                voters[r["member"]] = idx
        total = sum(counts)
        winner = max(range(len(counts)), key=lambda i: counts[i]) if total else None
        return {"poll": poll, "counts": counts, "total": total,
                "winner": winner, "closed": poll.is_closed(),
                "voters": voters}


    # -- expenses -------------------------------------------------------

    def add_expense(self, trip_id: str, who_paid: str, amount_kobo: int,
                    for_whom: list[str] | None, description: str = "",
                    ) -> GroupExpense | None:
        trip = self.get_trip(trip_id)
        if trip is None:
            return None
        who_paid = (who_paid or "").strip() or "anon"
        amount_kobo = int(amount_kobo or 0)
        if amount_kobo <= 0:
            return None
        # empty for_whom = split among all trip members
        whom = [m for m in (for_whom or []) if m and m.strip()]
        if not whom:
            whom = list(trip.members) or [who_paid]
        if who_paid not in whom:
            whom = whom + [who_paid]
        exp = GroupExpense(id="exp_" + new_short_id(length=8),
                           trip_id=trip_id, who_paid=who_paid,
                           amount_kobo=amount_kobo, for_whom=whom,
                           description=(description or "").strip(),
                           created_at=_now())
        self._db.execute(
            "INSERT INTO expenses (id, trip_id, who_paid, amount_kobo,"
            " for_whom_json, description, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (exp.id, exp.trip_id, exp.who_paid, exp.amount_kobo,
             json.dumps(exp.for_whom), exp.description, exp.created_at))
        self._db.commit()
        return exp

    def list_expenses(self, trip_id: str) -> list[GroupExpense]:
        try:
            rows = self._db.execute(
                "SELECT * FROM expenses WHERE trip_id = ?"
                " ORDER BY created_at", (trip_id,)).fetchall()
        except Exception:  # noqa: BLE001
            return []
        return [GroupExpense(
            id=r["id"], trip_id=r["trip_id"], who_paid=r["who_paid"],
            amount_kobo=r["amount_kobo"],
            for_whom=json.loads(r["for_whom_json"] or "[]"),
            description=r["description"], created_at=r["created_at"])
            for r in rows]

    def balances(self, trip_id: str) -> dict[str, int]:
        """Net balance per person in kobo. Positive = owed money."""
        bal: dict[str, int] = {}
        for exp in self.list_expenses(trip_id):
            n = len(exp.for_whom) or 1
            share, rem = divmod(exp.amount_kobo, n)
            bal[exp.who_paid] = bal.get(exp.who_paid, 0) + exp.amount_kobo
            for i, person in enumerate(exp.for_whom):
                # first `rem` people absorb one extra kobo — no dust lost
                owed = share + (1 if i < rem else 0)
                bal[person] = bal.get(person, 0) - owed
        return bal

    def settlement(self, trip_id: str) -> list[tuple[str, str, int]]:
        """Minimized (debtor, creditor, amount_kobo) transfers to settle up."""
        return settle_balances(self.balances(trip_id))

    # -- budgets ----------------------------------------------------------

    def set_budget(self, trip_id: str, member: str,
                   amount_kobo: int) -> bool:
        if self.get_trip(trip_id) is None:
            return False
        member = (member or "").strip()
        if not member or int(amount_kobo or 0) <= 0:
            return False
        self._db.execute(
            "INSERT OR REPLACE INTO budgets (trip_id, member, amount_kobo)"
            " VALUES (?, ?, ?)", (trip_id, member, int(amount_kobo)))
        self._db.commit()
        return True

    def get_budget(self, trip_id: str, member: str) -> int:
        try:
            row = self._db.execute(
                "SELECT amount_kobo FROM budgets WHERE trip_id = ?"
                " AND member = ?", (trip_id, member)).fetchone()
        except Exception:  # noqa: BLE001
            return 0
        return int(row["amount_kobo"]) if row else 0

    def spending(self, trip_id: str, member: str) -> int:
        """Total kobo this member owes across expenses (their share)."""
        total = 0
        for exp in self.list_expenses(trip_id):
            if member in exp.for_whom:
                n = len(exp.for_whom) or 1
                share, rem = divmod(exp.amount_kobo, n)
                idx = exp.for_whom.index(member)
                total += share + (1 if idx < rem else 0)
        return total


    # -- legs (per-traveler) ----------------------------------------------

    def add_leg(self, trip_id: str, member: str,
                flight: Flight | dict, note: str = "") -> TripLeg | None:
        trip = self.get_trip(trip_id)
        if trip is None:
            return None
        member = (member or "").strip() or "anon"
        fdict = asdict(flight) if isinstance(flight, Flight) else dict(flight or {})
        leg = TripLeg(id="leg_" + new_short_id(length=8), trip_id=trip_id,
                      member=member, flight=fdict,
                      note=(note or "").strip(), created_at=_now())
        self._db.execute(
            "INSERT INTO legs (id, trip_id, member, flight_json, note,"
            " created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (leg.id, leg.trip_id, leg.member, json.dumps(leg.flight),
             leg.note, leg.created_at))
        self._db.commit()
        return leg

    def list_legs(self, trip_id: str) -> list[TripLeg]:
        try:
            rows = self._db.execute(
                "SELECT * FROM legs WHERE trip_id = ? ORDER BY created_at",
                (trip_id,)).fetchall()
        except Exception:  # noqa: BLE001
            return []
        return [TripLeg(id=r["id"], trip_id=r["trip_id"], member=r["member"],
                        flight=json.loads(r["flight_json"] or "{}"),
                        note=r["note"], created_at=r["created_at"])
                for r in rows]

    # -- proposals vs agreed (Plan Harmony duality) -------------------------

    def propose(self, trip_id: str, member: str, idea: str) -> TripProposal | None:
        if self.get_trip(trip_id) is None:
            return None
        idea = (idea or "").strip()
        if not idea:
            return None
        prop = TripProposal(id="prop_" + new_short_id(length=8),
                            trip_id=trip_id,
                            member=(member or "").strip() or "anon",
                            idea=idea, status=_PROPOSAL_PROPOSED,
                            created_at=_now())
        self._db.execute(
            "INSERT INTO proposals (id, trip_id, member, idea, status,"
            " created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (prop.id, prop.trip_id, prop.member, prop.idea,
             prop.status, prop.created_at))
        self._db.commit()
        return prop

    def agree(self, trip_id: str, proposal_id: str) -> TripProposal | None:
        try:
            row = self._db.execute(
                "SELECT * FROM proposals WHERE id = ? AND trip_id = ?",
                (proposal_id, trip_id)).fetchone()
        except Exception:  # noqa: BLE001
            return None
        if row is None:
            return None
        self._db.execute(
            "UPDATE proposals SET status = ? WHERE id = ?",
            (_PROPOSAL_AGREED, proposal_id))
        self._db.commit()
        return TripProposal(id=row["id"], trip_id=row["trip_id"],
                            member=row["member"], idea=row["idea"],
                            status=_PROPOSAL_AGREED,
                            created_at=row["created_at"])

    def list_proposals(self, trip_id: str,
                       status: str = "") -> list[TripProposal]:
        try:
            if status:
                rows = self._db.execute(
                    "SELECT * FROM proposals WHERE trip_id = ? AND status = ?"
                    " ORDER BY created_at", (trip_id, status)).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT * FROM proposals WHERE trip_id = ?"
                    " ORDER BY created_at", (trip_id,)).fetchall()
        except Exception:  # noqa: BLE001
            return []
        return [TripProposal(id=r["id"], trip_id=r["trip_id"],
                             member=r["member"], idea=r["idea"],
                             status=r["status"], created_at=r["created_at"])
                for r in rows]

    # -- merged itinerary ---------------------------------------------------

    def merged_itinerary(self, trip_id: str) -> str:
        trip = self.get_trip(trip_id)
        if trip is None:
            return "no such group trip."
        return merged_itinerary(trip, self.list_legs(trip_id),
                                self.list_proposals(trip_id, _PROPOSAL_AGREED))



# ── settlement ─────────────────────────────────────────────────────────────


def settle_balances(balances: dict[str, int]) -> list[tuple[str, str, int]]:
    """Greedy minimized settlement: fewest transfers to zero everyone out.

    Debtors (negative) pay creditors (positive), largest amounts first.
    Pure function — the math is testable without a database.
    """
    debtors = sorted(((p, -b) for p, b in balances.items() if b < 0),
                     key=lambda x: -x[1])
    creditors = sorted(((p, b) for p, b in balances.items() if b > 0),
                       key=lambda x: -x[1])
    transfers: list[tuple[str, str, int]] = []
    i = j = 0
    debtors = [list(d) for d in debtors]
    creditors = [list(c) for c in creditors]
    while i < len(debtors) and j < len(creditors):
        d_person, d_amt = debtors[i]
        c_person, c_amt = creditors[j]
        pay = min(d_amt, c_amt)
        if pay > 0:
            transfers.append((d_person, c_person, pay))
        debtors[i][1] -= pay
        creditors[j][1] -= pay
        if debtors[i][1] <= 0:
            i += 1
        if creditors[j][1] <= 0:
            j += 1
    return transfers


def _naira(kobo: int) -> str:
    return f"₦{kobo / 100:,.0f}"


def parse_naira_kobo(text: str) -> int | None:
    """Naira-first amount parsing: '5000', '5k', '₦5,000', '2.5k' → kobo."""
    raw = (text or "").strip().lower().replace("₦", "").replace(",", "")
    raw = re.sub(r"^(ngn|naira)\s*", "", raw)
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*k?", raw)
    if not m:
        return None
    try:
        naira = float(m.group(1)) * (1000 if raw.endswith("k") else 1)
    except ValueError:
        return None
    return int(round(naira * 100))


def merged_itinerary(trip: GroupTrip, legs: list[TripLeg],
                     agreed: list[TripProposal]) -> str:
    """One readable view: every traveler's legs + the agreed items."""
    lines = [f"🧳 {trip.name} — group itinerary"]
    if legs:
        lines.append("\n✈️ Flights (per traveler):")
        for leg in legs:
            f = leg.flight or {}
            fn = f.get("flight_number") or "flight"
            route = ""
            if f.get("origin") and f.get("destination"):
                route = f"{f['origin']}→{f['destination']} "
            dep = f.get("departs", "")
            dep_s = f", departs {dep[11:16]}" if len(dep) > 12 else ""
            pnr = f", PNR {f['pnr']}" if f.get("pnr") else ""
            note = f" — {leg.note}" if leg.note else ""
            lines.append(f"  • {leg.member}: {route}{fn}{dep_s}{pnr}{note}")
    if agreed:
        lines.append("\n✅ Agreed:")
        for p in agreed:
            lines.append(f"  • {p.idea} ({p.member})")
    if not legs and not agreed:
        lines.append("\n(nothing planned yet — add legs and proposals!)")
    return "\n".join(lines)


# ── chat: /gtrip (group chats only) ─────────────────────────────────────────


_DEADLINE_RE = re.compile(r"\bdeadline\s+(\d+)\s*([mhd])\s*$", re.IGNORECASE)


def _parse_deadline(tail: str) -> tuple[str, float]:
    """Pull a trailing 'deadline 2h' off the tail → (rest, epoch)."""
    m = _DEADLINE_RE.search(tail or "")
    if not m:
        return tail, 0.0
    qty, unit = int(m.group(1)), m.group(2).lower()
    secs = {"m": 60, "h": 3600, "d": 86400}[unit] * qty
    return _DEADLINE_RE.sub("", tail).strip(), _now() + secs


def _group_key(chat: Any) -> str:
    platform = getattr(chat, "platform", "") or "chat"
    cid = getattr(chat, "id", "") or getattr(chat, "chat_id", "") or "main"
    return f"{platform}:{cid}"


def _default_trip(store: GroupTripStore) -> GroupTrip | None:
    trips = store.list_trips()
    return trips[0] if len(trips) == 1 else None


def _resolve_trip(store: GroupTripStore, trip_id: str) -> GroupTrip | None:
    trip_id = (trip_id or "").strip()
    if trip_id:
        return store.get_trip(trip_id)
    return _default_trip(store)


def _fmt_deadline(poll: GroupPoll) -> str:
    if not poll.deadline:
        return "no deadline"
    left = poll.deadline - _now()
    if left <= 0:
        return "closed"
    if left >= 3600:
        return f"closes in {left / 3600:.1f}h"
    return f"closes in {left / 60:.0f}m"


def control_gtrip(tail: str, context: Any = None, chat: Any = None,
                  sender_id: str = "", sender: str = "") -> str:
    """``/gtrip`` — the group-travel stack. Group chats only; never raises.

    Not owner-gated: every group member drives it. State is group-scoped
    under ``~/.devon/community/group_trips/``.
    """
    try:
        if chat is None or getattr(chat, "kind", "dm") != "group":
            return ("Group trips live in group chats — polls, shared expenses"
                    " and legs need the whole crew. Try this in a group. 👥")
        data_dir = getattr(getattr(context, "settings", None),
                           "community_dir", None)
        if data_dir:
            db_path = str(Path(data_dir) / "group_trips" /
                          f"{_sanitize_group_key(_group_key(chat))}.db")
        else:
            db_path = ""
        store = GroupTripStore(_group_key(chat), db_path=db_path)
        user_id = (sender_id or "").strip() or (sender or "").strip() or "anon"
        name = (sender or "").strip() or user_id

        words = (tail or "").split(None, 1)
        sub = words[0].lower() if words else "help"
        rest = words[1] if len(words) > 1 else ""

        if sub == "new":
            trip = store.create_trip(rest or "Group trip", [name],
                                     created_by=name)
            return (f"🧳 Group trip created: **{trip.name}**\n"
                    f"ID: `{trip.id}`\nMembers: {', '.join(trip.members)}\n"
                    f"Now: `/gtrip poll`, `/gtrip expense`, `/gtrip leg`,"
                    f" `/gtrip propose`.")

        if sub == "list":
            trips = store.list_trips()
            if not trips:
                return "No group trips here yet — `/gtrip new <name>`."
            return "🧳 Group trips:\n" + "\n".join(
                f"  • {t.name} `{t.id}` ({len(t.members)} members)"
                for t in trips)

        if sub == "join":
            trip = _resolve_trip(store, rest)
            if trip is None:
                return "Which trip? `/gtrip list`, then `/gtrip join <id>`."
            store.add_member(trip.id, name)
            return f"You're in **{trip.name}**. ✈️"

        if sub == "poll":
            trip = _resolve_trip(store, "")
            if trip is None:
                return "Create a trip first: `/gtrip new <name>`."
            body, deadline = _parse_deadline(rest)
            parts = [p.strip() for p in body.split("|") if p.strip()]
            if len(parts) < 3:
                return ("Usage: `/gtrip poll <question> | <opt1> | <opt2>"
                        " [| opt3…] [deadline 2h]`")
            poll = store.create_poll(trip.id, parts[0], parts[1:],
                                     deadline=deadline, created_by=name)
            if poll is None:
                return "Couldn't create that poll."
            opts = "\n".join(f"  {i + 1}. {o}"
                             for i, o in enumerate(poll.options))
            return (f"📊 **{poll.question}**\n{opts}\n"
                    f"Vote: `/gtrip vote {poll.id} <option>`"
                    f" ({_fmt_deadline(poll)})")

        if sub == "vote":
            parts = rest.split(None, 1)
            if len(parts) < 2:
                return "Usage: `/gtrip vote <poll_id> <option>`."
            ok = store.vote(parts[0], name, parts[1])
            if not ok:
                return ("Vote didn't count — check the poll id/option,"
                        " or the poll is closed.")
            return f"Vote counted for **{name}**. 🗳️"

        if sub == "result":
            res = store.poll_result(rest.strip())
            if res is None:
                return "No such poll."
            poll = res["poll"]
            lines = [f"📊 **{poll.question}**"
                     f" ({'closed' if res['closed'] else 'open'})"]
            for i, opt in enumerate(poll.options):
                lines.append(f"  {i + 1}. {opt} — {res['counts'][i]} vote(s)")
            if res["winner"] is not None:
                lines.append(f"🏆 Leading: **{poll.options[res['winner']]}**")
            return "\n".join(lines)

        if sub == "expense":
            trip = _resolve_trip(store, "")
            if trip is None:
                return "Create a trip first: `/gtrip new <name>`."
            store.add_member(trip.id, name)
            parts = rest.split(None, 1)
            if not parts:
                return ("Usage: `/gtrip expense <amount> <what>"
                        " [for <member,member>]` — e.g."
                        " `/gtrip expense 5k suya for Ada,Mama`.")
            amount = parse_naira_kobo(parts[0])
            if amount is None:
                return f"Couldn't read the amount '{parts[0]}'."
            desc, for_whom = parts[1] if len(parts) > 1 else "", None
            m = re.search(r"\bfor\s+(.+)$", desc, re.IGNORECASE)
            if m:
                for_whom = [x.strip() for x in m.group(1).split(",")
                            if x.strip()]
                desc = desc[:m.start()].strip()
            exp = store.add_expense(trip.id, name, amount, for_whom,
                                    desc or "expense")
            if exp is None:
                return "Couldn't log that expense."
            whom_s = ", ".join(exp.for_whom)
            return (f"💸 Logged: **{name}** paid {_naira(amount)}"
                    f" ({exp.description}) — split: {whom_s}.")

        if sub == "settle":
            trip = _resolve_trip(store, rest)
            if trip is None:
                return "Which trip? `/gtrip list`, then `/gtrip settle <id>`."
            transfers = store.settlement(trip.id)
            if not transfers:
                return "Everyone's settled up. ✅"
            lines = ["🧾 Settle up:"]
            for debtor, creditor, amt in transfers:
                lines.append(f"  • {debtor} → {creditor}: {_naira(amt)}")
            return "\n".join(lines)

        if sub == "budget":
            trip = _resolve_trip(store, "")
            if trip is None:
                return "Create a trip first: `/gtrip new <name>`."
            parts = rest.split(None, 1)
            if len(parts) < 2:
                return "Usage: `/gtrip budget <member> <amount>`."
            amount = parse_naira_kobo(parts[1])
            if amount is None:
                return f"Couldn't read the amount '{parts[1]}'."
            if store.set_budget(trip.id, parts[0], amount):
                return (f"Budget set: **{parts[0]}** — {_naira(amount)}"
                        f" for {trip.name}.")
            return "Couldn't set that budget."

        if sub == "spending":
            trip = _resolve_trip(store, "")
            if trip is None:
                return "Create a trip first: `/gtrip new <name>`."
            member = rest.strip() or name
            spent = store.spending(trip.id, member)
            budget = store.get_budget(trip.id, member)
            line = f"💰 {member} has spent {_naira(spent)}"
            if budget:
                line += f" of {_naira(budget)} budget"
                if spent > budget:
                    line += " ⚠️ over budget!"
            return line + "."

        if sub == "leg":
            trip = _resolve_trip(store, "")
            if trip is None:
                return "Create a trip first: `/gtrip new <name>`."
            store.add_member(trip.id, name)
            if not rest.strip():
                return ("Usage: `/gtrip leg <flight details>` — e.g."
                        " `/gtrip leg BA075 LOS to LHR departs 22:45`.")
            try:
                parsed = parse_confirmation(rest)
            except Exception:  # noqa: BLE001
                parsed = {}
            flights = parsed.get("flights") or []
            if flights:
                f = flights[0]
                flight = Flight(
                    airline=f.get("airline", ""),
                    flight_number=f.get("flight_number", ""),
                    origin=f.get("origin", ""), destination=f.get("destination", ""),
                    departs=f.get("departs", ""), arrives=f.get("arrives", ""),
                    pnr=f.get("pnr", ""))
            else:
                flight = Flight(flight_number=rest.strip()[:40])
            leg = store.add_leg(trip.id, name, flight)
            if leg is None:
                return "Couldn't add that leg."
            return f"✈️ Leg added for **{name}**: {flight.one_line()}."

        if sub == "propose":
            trip = _resolve_trip(store, "")
            if trip is None:
                return "Create a trip first: `/gtrip new <name>`."
            prop = store.propose(trip.id, name, rest)
            if prop is None:
                return "Usage: `/gtrip propose <idea>`."
            return (f"💡 Proposed: *{prop.idea}*\n"
                    f"Agree with `/gtrip agree {prop.id}`.")

        if sub == "agree":
            trip = _resolve_trip(store, "")
            if trip is None:
                return "Create a trip first: `/gtrip new <name>`."
            prop = store.agree(trip.id, rest.strip())
            if prop is None:
                return "No such proposal."
            return f"✅ Agreed: *{prop.idea}*"

        if sub == "proposals":
            trip = _resolve_trip(store, rest)
            if trip is None:
                return "Which trip? `/gtrip list`."
            props = store.list_proposals(trip.id, _PROPOSAL_PROPOSED)
            if not props:
                return "No open proposals."
            return "💡 Proposals:\n" + "\n".join(
                f"  • {p.idea} ({p.member}) — `/gtrip agree {p.id}`"
                for p in props)

        if sub == "itinerary":
            trip = _resolve_trip(store, rest)
            if trip is None:
                return "Which trip? `/gtrip list`, then `/gtrip itinerary <id>`."
            return store.merged_itinerary(trip.id)

        return ("🧳 **/gtrip** — the group-travel stack:\n"
                "`/gtrip new <name>` · `/gtrip list` · `/gtrip join <id>`\n"
                "`/gtrip poll <q> | <opt1> | <opt2> [deadline 2h]` ·"
                " `/gtrip vote <id> <opt>` · `/gtrip result <id>`\n"
                "`/gtrip expense <amount> <what> [for <m1,m2>]` ·"
                " `/gtrip settle [id]`\n"
                "`/gtrip budget <member> <amount>` · `/gtrip spending [member]`\n"
                "`/gtrip leg <flight details>` · `/gtrip itinerary [id]`\n"
                "`/gtrip propose <idea>` · `/gtrip agree <id>` ·"
                " `/gtrip proposals`")
    except Exception as exc:  # noqa: BLE001 — chat surface never raises
        _log.debug("control_gtrip failed", exc_info=True)
        return f"Group trip hiccup: {exc}. Try `/gtrip` for usage."


def register(registry: Any) -> None:
    """Tool hook: ``gtrip`` — group travel stack (group chats only)."""
    from ..core.policy import Capability

    @registry.register(
        "gtrip",
        description=("Group-travel stack: polls with deadlines, shared"
                     " expenses with minimized settle-up, per-person budgets,"
                     " per-traveler flight legs, proposals vs agreed items,"
                     " merged itinerary. Group chats only; group-scoped state."),
        capability=Capability("community.miniapp"),
    )
    def _gtrip_tool(ctx: Any, action: str = "help",
                    **kwargs: Any) -> str:
        _ = ctx, kwargs
        return control_gtrip(action if action != "help" else "")
