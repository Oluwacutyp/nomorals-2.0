"""Travel <-> community group implementation.

The bridge between :mod:`nomorals.travel.groups` (the group-trip stack:
polls, expenses, per-person budgets, merged itinerary) and a community
group surface: a trip is *shared* to a community group as one post
(destination, members, dates, cost split), and members RSVP
(going / maybe / declined) against it.

Isolation: this module deliberately does NOT import
``nomorals.community``.  Participants are identified only by the
display names passed in by the caller, and the share goes out through
an injected ``post`` callable (``post(payload) -> ref``) supplied by
whatever community surface owns the group — the bridge never touches
owner memory, accounts, vaults, or private connectors.  State lives in
the group-scoped :class:`GroupTripStore` (``~/.devon/community/``)
plus an in-memory RSVP table keyed by trip id.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .groups import GroupTrip, GroupTripStore, compute_shares

__all__ = [
    "GroupTripShare",
    "RSVP",
    "CommunityGroupTripBridge",
    "RSVP_GOING",
    "RSVP_MAYBE",
    "RSVP_DECLINED",
]

RSVP_GOING = "going"
RSVP_MAYBE = "maybe"
RSVP_DECLINED = "declined"
_RSVP_STATUSES = (RSVP_GOING, RSVP_MAYBE, RSVP_DECLINED)


@dataclass
class GroupTripShare:
    """One trip shared to a community group: the post payload + receipt."""

    trip_id: str = ""
    trip_name: str = ""
    destination: str = ""
    members: list[str] = field(default_factory=list)
    start_date: str = ""
    end_date: str = ""
    itinerary: str = ""
    cost_split: dict[str, int] = field(default_factory=dict)  # member -> kobo owed
    total_kobo: int = 0
    community_ref: str = ""   # group id / channel the post went to
    post_ref: str = ""        # receipt returned by the post callable
    shared_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trip_id": self.trip_id,
            "trip_name": self.trip_name,
            "destination": self.destination,
            "members": list(self.members),
            "start_date": self.start_date,
            "end_date": self.end_date,
            "itinerary": self.itinerary,
            "cost_split": dict(self.cost_split),
            "total_kobo": self.total_kobo,
            "community_ref": self.community_ref,
            "post_ref": self.post_ref,
            "shared_at": self.shared_at,
        }

    def format_post(self) -> str:
        """The share text: destination, dates, who's in, what each owes."""
        lines = [f"🧳 {self.trip_name or 'Group trip'}"]
        if self.destination:
            dates = ""
            if self.start_date:
                dates = f" · {self.start_date}" + (
                    f" → {self.end_date}" if self.end_date else "")
            lines.append(f"📍 {self.destination}{dates}")
        if self.members:
            lines.append(f"👥 {', '.join(self.members)}")
        if self.itinerary:
            lines.append("")
            lines.append(self.itinerary)
        if self.cost_split:
            lines.append("")
            lines.append("💰 cost split (kobo owed):")
            for member in sorted(self.cost_split):
                lines.append(f"  • {member}: ₦{self.cost_split[member] / 100:,.0f}")
        lines.append("")
        lines.append("RSVP: going / maybe / declined.")
        return "\n".join(lines)


@dataclass
class RSVP:
    """One member's RSVP on one trip."""

    trip_id: str = ""
    member: str = ""
    status: str = RSVP_MAYBE
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"trip_id": self.trip_id, "member": self.member,
                "status": self.status, "updated_at": self.updated_at}


class CommunityGroupTripBridge:
    """Group trips shared to community groups, with RSVPs.

    ``store`` is the group-scoped :class:`GroupTripStore` holding the
    trip (polls, expenses, legs).  ``post`` is the community surface's
    publish hook: ``post(payload: dict) -> str`` returning a receipt
    (message id / post id).  With no ``post`` hook, shares are built
    and recorded locally and ``post_ref`` stays empty — the share is
    never silently pretended to have gone out.
    """

    def __init__(self, store: GroupTripStore | None = None, *,
                 post: Callable[[dict[str, Any]], str] | None = None,
                 db_path: str = "") -> None:
        if store is not None:
            self.store = store
        else:
            path = db_path or tempfile.mktemp(
                prefix="community-trip-bridge-", suffix=".db")
            self.store = GroupTripStore("bridge", db_path=path)
        self.post = post
        self._rsvps: dict[str, dict[str, RSVP]] = {}
        self._shares: dict[str, GroupTripShare] = {}

    # ── trips ───────────────────────────────────────────────────────────

    def create_group_trip(self, name: str, members: list[str], *,
                          destination: str = "", start_date: str = "",
                          end_date: str = "", created_by: str = "") -> GroupTrip:
        """Create the trip (destination, members, dates) in group state."""
        return self.store.create_trip(
            name, members, created_by=created_by or (members[0] if members else ""),
            destination=destination, start_date=start_date, end_date=end_date)

    def add_cost(self, trip_id: str, who_paid: str, amount_kobo: int,
                 description: str = "", *, category: str = "") -> Any:
        """Log a shared cost on the trip (split equally by default)."""
        return self.store.add_expense(
            trip_id, who_paid, amount_kobo, None,
            description or "shared cost", category=category)

    def cost_split(self, trip_id: str, total_kobo: int) -> dict[str, int]:
        """Equal kobo split of ``total_kobo`` across trip members —
        integer math via :func:`compute_shares`, sums exactly."""
        trip = self.store.get_trip(trip_id)
        if trip is None:
            raise ValueError(f"no such trip {trip_id!r}")
        shares = compute_shares(int(total_kobo), list(trip.members))
        return dict(zip(trip.members, shares))

    def settle(self, trip_id: str) -> list[tuple[str, str, int]]:
        """Minimized settle-up transfers from the trip's real expenses."""
        return self.store.settlement(trip_id)

    # ── sharing ─────────────────────────────────────────────────────────

    def share_itinerary(self, trip_id: str, community_ref: str) -> GroupTripShare:
        """Build the share for ``trip_id`` and publish it to the
        community group ``community_ref`` via the post hook.

        The itinerary is the trip's merged view (legs + agreed items);
        the cost split is the live balance per member from recorded
        expenses (what each member currently owes/ is owed, floored at
        0 for the share card — the full settle-up lives in
        :meth:`settle`).
        """
        trip = self.store.get_trip(trip_id)
        if trip is None:
            raise ValueError(f"no such trip {trip_id!r}")
        itinerary = self.store.merged_itinerary(trip_id)
        balances = self.store.balances(trip_id)
        cost_split = {m: max(0, -balances.get(m, 0))
                      for m in trip.members}
        total = sum(e.amount_kobo for e in self.store.list_expenses(trip_id))
        share = GroupTripShare(
            trip_id=trip.id, trip_name=trip.name,
            destination=trip.destination, members=list(trip.members),
            start_date=trip.start_date, end_date=trip.end_date,
            itinerary=itinerary, cost_split=cost_split, total_kobo=total,
            community_ref=community_ref, shared_at=time.time())
        payload = {"kind": "group-trip-share",
                   "community_ref": community_ref,
                   "text": share.format_post(),
                   **share.to_dict()}
        if self.post is not None:
            share.post_ref = str(self.post(payload) or "")
        self._shares[trip_id] = share
        return share

    def last_share(self, trip_id: str) -> GroupTripShare | None:
        return self._shares.get(trip_id)

    # ── RSVP ────────────────────────────────────────────────────────────

    def rsvp(self, trip_id: str, member: str, status: str) -> RSVP:
        """Record (or update) one member's RSVP. Unknown trip → error;
        unknown status → error (no silent coercion to 'maybe')."""
        if self.store.get_trip(trip_id) is None:
            raise ValueError(f"no such trip {trip_id!r}")
        status = (status or "").strip().lower()
        if status not in _RSVP_STATUSES:
            raise ValueError(
                f"unknown RSVP status {status!r}; expected {_RSVP_STATUSES}")
        member = (member or "").strip()
        if not member:
            raise ValueError("RSVP needs a member name")
        rec = RSVP(trip_id=trip_id, member=member, status=status,
                   updated_at=time.time())
        self._rsvps.setdefault(trip_id, {})[member] = rec
        return rec

    def rsvp_summary(self, trip_id: str) -> dict[str, Any]:
        """Counts + member lists per RSVP status. Members with no RSVP
        are listed under ``no_reply`` — silence is visible, not assumed."""
        trip = self.store.get_trip(trip_id)
        if trip is None:
            raise ValueError(f"no such trip {trip_id!r}")
        table = self._rsvps.get(trip_id, {})
        buckets: dict[str, list[str]] = {s: [] for s in _RSVP_STATUSES}
        buckets["no_reply"] = []
        for member in trip.members:
            rec = table.get(member)
            if rec is None:
                buckets["no_reply"].append(member)
            else:
                buckets[rec.status].append(member)
        return {
            "trip_id": trip_id,
            "counts": {k: len(v) for k, v in buckets.items()},
            "going": buckets[RSVP_GOING],
            "maybe": buckets[RSVP_MAYBE],
            "declined": buckets[RSVP_DECLINED],
            "no_reply": buckets["no_reply"],
        }
