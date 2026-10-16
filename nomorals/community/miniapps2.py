"""Community mini-apps, second wave: carpool, rotas, marketplace/barter,
league/fantasy, receipts + split.

These sit beside ``miniapps.py`` (polls / expenses / quizzes / RSVPs)
rather than extending it, so the original mini-app surface stays
untouched. Same rules: group-scoped state only, JSON under
``~/.devon/community/``, members identified by platform user id /
display name from the chat message itself.

* Carpool — a ride with a route, a seat count, and riders.
* Rotas — a schedule of shifts with assignees, plus swaps.
* Marketplace / barter — listings with offers; accepting an offer
  closes the listing. Barter offers carry free text ("2 bags of rice")
  instead of a price — no money is ever moved by this module.
* League / fantasy — teams, results, points, standings (win 3 / draw 1,
  then goal difference, then goals for).
* Receipts / split — a receipt split by shares with exact-cent
  accounting, and minimal settlements ("who owes whom").

Isolation: stdlib + ``nomorals.core`` only. No owner-data imports.
Money here is bookkeeping in cents — this module never charges anyone.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.ids import new_short_id

__all__ = [
    "CarpoolRide",
    "Rota",
    "Shift",
    "Listing",
    "Offer",
    "League",
    "Team",
    "Receipt",
    "SplitBill",
    "split_by_shares",
    "settle_balances",
    "MiniApp2Store",
    "default_miniapps2_dir",
]


def default_miniapps2_dir() -> Path:
    """Default on-disk home: inside the community dir."""
    return Path.home() / ".devon" / "community" / "miniapps2"


def _now() -> float:
    return time.time()


# ── shared money helper ──────────────────────────────────────────────────
def split_by_shares(total_cents: int, shares: dict[str, float]) -> dict[str, int]:
    """Split ``total_cents`` by weights, in whole cents, summing EXACTLY.

    Largest-remainder method: every member gets ``floor(exact share)``,
    then leftover cents go one each to the members with the largest
    fractional remainders (stable order: remainder desc, then member id).
    Zero/negative weights are rejected; an empty ``shares`` map or a
    zero total returns an empty / all-zero split respectively.
    """
    if total_cents < 0:
        raise ValueError("total cannot be negative")
    if not shares:
        return {}
    if any(w <= 0 for w in shares.values()):
        raise ValueError("every share weight must be positive")
    weight_total = sum(shares.values())
    exact = {m: total_cents * w / weight_total for m, w in shares.items()}
    floors = {m: int(v // 1) for m, v in exact.items()}
    leftover = total_cents - sum(floors.values())
    remainders = sorted(
        shares, key=lambda m: (-(exact[m] - floors[m]), m)
    )
    result = dict(floors)
    for member in remainders[:leftover]:
        result[member] += 1
    return result


def settle_balances(balances: dict[str, int]) -> list[dict[str, Any]]:
    """Minimal settlements for a set of net balances (cents).

    Positive balance = is owed money; negative = owes money. Greedy
    largest-debtor ↔ largest-creditor pairing yields at most n−1
    transfers. The input is not mutated; zero balances produce no
    transfers. Dust (< 1 cent) is impossible here — everything is ints.
    """
    debtors = sorted(
        ((m, -b) for m, b in balances.items() if b < 0),
        key=lambda x: (-x[1], x[0]),
    )
    creditors = sorted(
        ((m, b) for m, b in balances.items() if b > 0),
        key=lambda x: (-x[1], x[0]),
    )
    transfers: list[dict[str, Any]] = []
    i = j = 0
    debt = list(debtors)
    cred = list(creditors)
    while i < len(debt) and j < len(cred):
        debtor, owe = debt[i]
        creditor, owed = cred[j]
        amount = min(owe, owed)
        transfers.append(
            {"from_id": debtor, "to_id": creditor, "amount_cents": amount}
        )
        owe -= amount
        owed -= amount
        if owe == 0:
            i += 1
        else:
            debt[i] = (debtor, owe)
        if owed == 0:
            j += 1
        else:
            cred[j] = (creditor, owed)
    return transfers


# ── carpool ─────────────────────────────────────────────────────────────
@dataclass
class CarpoolRide:
    id: str
    group_key: str
    route_from: str
    route_to: str
    depart_at: str  # free-form ("Sat 8:00") — the group writes it
    seats_total: int
    driver_id: str
    driver_name: str
    riders: list[dict[str, str]] = field(default_factory=list)
    created_ts: float = 0.0

    @property
    def seats_left(self) -> int:
        """Seats excluding the driver; never below zero."""
        return max(0, self.seats_total - len(self.riders))

    @property
    def is_full(self) -> bool:
        return self.seats_left == 0

    def add_rider(self, user_id: str, display_name: str) -> None:
        if any(r["user_id"] == user_id for r in self.riders):
            raise ValueError(f"{display_name} is already riding")
        if user_id == self.driver_id:
            raise ValueError("the driver already has a seat")
        if self.is_full:
            raise ValueError("ride is full")
        self.riders.append({"user_id": user_id, "display_name": display_name})

    def remove_rider(self, user_id: str) -> bool:
        before = len(self.riders)
        self.riders = [r for r in self.riders if r["user_id"] != user_id]
        return len(self.riders) < before

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CarpoolRide":
        return cls(
            id=str(data["id"]),
            group_key=str(data["group_key"]),
            route_from=str(data["route_from"]),
            route_to=str(data["route_to"]),
            depart_at=str(data.get("depart_at", "")),
            seats_total=int(data["seats_total"]),
            driver_id=str(data["driver_id"]),
            driver_name=str(data.get("driver_name", "")),
            riders=list(data.get("riders", [])),
            created_ts=float(data.get("created_ts", 0.0)),
        )


# ── rotas ───────────────────────────────────────────────────────────────
@dataclass
class Shift:
    date: str  # "2026-10-19" or any group-agreed label
    slot: str  # "morning" / "cleaning" / free-form
    assignee_id: str = ""
    assignee_name: str = ""

    @property
    def is_assigned(self) -> bool:
        return bool(self.assignee_id)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Shift":
        return cls(
            date=str(data["date"]),
            slot=str(data["slot"]),
            assignee_id=str(data.get("assignee_id", "")),
            assignee_name=str(data.get("assignee_name", "")),
        )


@dataclass
class Rota:
    id: str
    group_key: str
    title: str
    shifts: list[Shift] = field(default_factory=list)
    created_ts: float = 0.0

    def add_shift(self, date: str, slot: str) -> Shift:
        if any(s.date == date and s.slot == slot for s in self.shifts):
            raise ValueError(f"shift {date} {slot} already exists")
        shift = Shift(date=date, slot=slot)
        self.shifts.append(shift)
        return shift

    def _find(self, date: str, slot: str) -> Shift:
        for s in self.shifts:
            if s.date == date and s.slot == slot:
                return s
        raise ValueError(f"no shift {date} {slot}")

    def assign(self, date: str, slot: str, user_id: str, display_name: str) -> None:
        shift = self._find(date, slot)
        shift.assignee_id = user_id
        shift.assignee_name = display_name

    def unassign(self, date: str, slot: str) -> None:
        shift = self._find(date, slot)
        shift.assignee_id = ""
        shift.assignee_name = ""

    def swap(self, first: tuple[str, str], second: tuple[str, str]) -> None:
        """Swap the assignees of two shifts (one may be unassigned)."""
        a = self._find(*first)
        b = self._find(*second)
        (a.assignee_id, b.assignee_id) = (b.assignee_id, a.assignee_id)
        (a.assignee_name, b.assignee_name) = (b.assignee_name, a.assignee_name)

    def unassigned(self) -> list[Shift]:
        return [s for s in self.shifts if not s.is_assigned]

    def shifts_for(self, user_id: str) -> list[Shift]:
        return [s for s in self.shifts if s.assignee_id == user_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "group_key": self.group_key,
            "title": self.title,
            "shifts": [s.to_dict() for s in self.shifts],
            "created_ts": self.created_ts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Rota":
        return cls(
            id=str(data["id"]),
            group_key=str(data["group_key"]),
            title=str(data["title"]),
            shifts=[Shift.from_dict(s) for s in data.get("shifts", [])],
            created_ts=float(data.get("created_ts", 0.0)),
        )


# ── marketplace / barter ────────────────────────────────────────────────
LISTING_OPEN = "open"
LISTING_CLOSED = "closed"
OFFER_PENDING = "pending"
OFFER_ACCEPTED = "accepted"
OFFER_DECLINED = "declined"


@dataclass
class Offer:
    id: str
    listing_id: str
    buyer_id: str
    buyer_name: str
    amount_cents: int | None = None  # None for a pure barter offer
    barter_text: str = ""  # e.g. "2 bags of rice" — free-form, never money
    status: str = OFFER_PENDING
    created_ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Offer":
        amount = data.get("amount_cents")
        return cls(
            id=str(data["id"]),
            listing_id=str(data["listing_id"]),
            buyer_id=str(data["buyer_id"]),
            buyer_name=str(data.get("buyer_name", "")),
            amount_cents=int(amount) if amount is not None else None,
            barter_text=str(data.get("barter_text", "")),
            status=str(data.get("status", OFFER_PENDING)),
            created_ts=float(data.get("created_ts", 0.0)),
        )


@dataclass
class Listing:
    id: str
    group_key: str
    seller_id: str
    seller_name: str
    title: str
    kind: str = "sell"  # "sell" | "barter" | "wanted"
    price_cents: int | None = None
    description: str = ""
    status: str = LISTING_OPEN
    offers: list[Offer] = field(default_factory=list)
    created_ts: float = 0.0

    def make_offer(
        self,
        buyer_id: str,
        buyer_name: str,
        amount_cents: int | None = None,
        barter_text: str = "",
    ) -> Offer:
        if self.status != LISTING_OPEN:
            raise ValueError("listing is closed")
        if buyer_id == self.seller_id:
            raise ValueError("seller cannot offer on their own listing")
        if amount_cents is not None and amount_cents < 0:
            raise ValueError("offer amount cannot be negative")
        if amount_cents is None and not barter_text.strip():
            raise ValueError("barter offers need barter_text")
        offer = Offer(
            id=new_short_id("offer"),
            listing_id=self.id,
            buyer_id=buyer_id,
            buyer_name=buyer_name,
            amount_cents=amount_cents,
            barter_text=barter_text,
            created_ts=_now(),
        )
        self.offers.append(offer)
        return offer

    def _find_offer(self, offer_id: str) -> Offer:
        for o in self.offers:
            if o.id == offer_id:
                return o
        raise ValueError(f"no offer {offer_id}")

    def accept_offer(self, offer_id: str) -> Offer:
        """Accept one offer: it becomes accepted, the rest decline, and
        the listing closes. The deal itself happens off-app."""
        if self.status != LISTING_OPEN:
            raise ValueError("listing is closed")
        chosen = self._find_offer(offer_id)
        if chosen.status != OFFER_PENDING:
            raise ValueError("offer is no longer pending")
        for o in self.offers:
            o.status = OFFER_ACCEPTED if o.id == offer_id else (
                OFFER_DECLINED if o.status == OFFER_PENDING else o.status
            )
        self.status = LISTING_CLOSED
        return chosen

    def decline_offer(self, offer_id: str) -> Offer:
        offer = self._find_offer(offer_id)
        if offer.status != OFFER_PENDING:
            raise ValueError("offer is no longer pending")
        offer.status = OFFER_DECLINED
        return offer

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "group_key": self.group_key,
            "seller_id": self.seller_id,
            "seller_name": self.seller_name,
            "title": self.title,
            "kind": self.kind,
            "price_cents": self.price_cents,
            "description": self.description,
            "status": self.status,
            "offers": [o.to_dict() for o in self.offers],
            "created_ts": self.created_ts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Listing":
        price = data.get("price_cents")
        return cls(
            id=str(data["id"]),
            group_key=str(data["group_key"]),
            seller_id=str(data["seller_id"]),
            seller_name=str(data.get("seller_name", "")),
            title=str(data["title"]),
            kind=str(data.get("kind", "sell")),
            price_cents=int(price) if price is not None else None,
            description=str(data.get("description", "")),
            status=str(data.get("status", LISTING_OPEN)),
            offers=[Offer.from_dict(o) for o in data.get("offers", [])],
            created_ts=float(data.get("created_ts", 0.0)),
        )


# ── league / fantasy ────────────────────────────────────────────────────
@dataclass
class Team:
    id: str
    name: str
    member_ids: list[str] = field(default_factory=list)
    played: int = 0
    won: int = 0
    drawn: int = 0
    lost: int = 0
    goals_for: int = 0
    goals_against: int = 0

    @property
    def points(self) -> int:
        return self.won * 3 + self.drawn

    @property
    def goal_diff(self) -> int:
        return self.goals_for - self.goals_against

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Team":
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            member_ids=list(data.get("member_ids", [])),
            played=int(data.get("played", 0)),
            won=int(data.get("won", 0)),
            drawn=int(data.get("drawn", 0)),
            lost=int(data.get("lost", 0)),
            goals_for=int(data.get("goals_for", 0)),
            goals_against=int(data.get("goals_against", 0)),
        )


@dataclass
class League:
    id: str
    group_key: str
    name: str
    teams: dict[str, Team] = field(default_factory=dict)
    created_ts: float = 0.0

    def add_team(self, name: str, member_ids: list[str] | None = None) -> Team:
        team = Team(
            id=new_short_id("team"), name=name, member_ids=list(member_ids or [])
        )
        self.teams[team.id] = team
        return team

    def record_result(
        self, team_a_id: str, score_a: int, team_b_id: str, score_b: int
    ) -> None:
        """Book a result. Scores must be non-negative; a team can't play itself."""
        if team_a_id == team_b_id:
            raise ValueError("a team cannot play itself")
        if score_a < 0 or score_b < 0:
            raise ValueError("scores cannot be negative")
        try:
            a = self.teams[team_a_id]
            b = self.teams[team_b_id]
        except KeyError as exc:
            raise ValueError(f"unknown team {exc.args[0]!r}") from exc
        a.played += 1
        b.played += 1
        a.goals_for += score_a
        a.goals_against += score_b
        b.goals_for += score_b
        b.goals_against += score_a
        if score_a > score_b:
            a.won += 1
            b.lost += 1
        elif score_b > score_a:
            b.won += 1
            a.lost += 1
        else:
            a.drawn += 1
            b.drawn += 1

    def standings(self) -> list[Team]:
        """Points desc, then goal difference, then goals for, then name."""
        return sorted(
            self.teams.values(),
            key=lambda t: (-t.points, -t.goal_diff, -t.goals_for, t.name),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "group_key": self.group_key,
            "name": self.name,
            "teams": {tid: t.to_dict() for tid, t in self.teams.items()},
            "created_ts": self.created_ts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "League":
        return cls(
            id=str(data["id"]),
            group_key=str(data["group_key"]),
            name=str(data["name"]),
            teams={
                tid: Team.from_dict(t)
                for tid, t in data.get("teams", {}).items()
            },
            created_ts=float(data.get("created_ts", 0.0)),
        )


# ── receipts / split ────────────────────────────────────────────────────
@dataclass
class Receipt:
    id: str
    group_key: str
    payer_id: str
    payer_name: str
    total_cents: int
    description: str = ""
    shares: dict[str, float] = field(default_factory=dict)
    created_ts: float = 0.0

    def split(self) -> dict[str, int]:
        """Each member's share of this receipt in cents; sums exactly."""
        if not self.shares:
            raise ValueError("receipt has no shares to split by")
        return split_by_shares(self.total_cents, self.shares)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Receipt":
        return cls(
            id=str(data["id"]),
            group_key=str(data["group_key"]),
            payer_id=str(data["payer_id"]),
            payer_name=str(data.get("payer_name", "")),
            total_cents=int(data["total_cents"]),
            description=str(data.get("description", "")),
            shares={str(k): float(v) for k, v in data.get("shares", {}).items()},
            created_ts=float(data.get("created_ts", 0.0)),
        )


@dataclass
class SplitBill:
    """A running tab: many receipts, one balance sheet, minimal settlements."""

    id: str
    group_key: str
    title: str
    receipts: list[Receipt] = field(default_factory=list)
    created_ts: float = 0.0

    def add_receipt(
        self,
        payer_id: str,
        payer_name: str,
        total_cents: int,
        shares: dict[str, float],
        description: str = "",
    ) -> Receipt:
        if total_cents < 0:
            raise ValueError("receipt total cannot be negative")
        receipt = Receipt(
            id=new_short_id("rcpt"),
            group_key=self.group_key,
            payer_id=payer_id,
            payer_name=payer_name,
            total_cents=total_cents,
            description=description,
            shares=dict(shares),
            created_ts=_now(),
        )
        receipt.split()  # validate shares now, not at settlement time
        self.receipts.append(receipt)
        return receipt

    def balances(self) -> dict[str, int]:
        """Net per member in cents. Positive = is owed; negative = owes.
        Sums to exactly zero across the whole bill."""
        balances: dict[str, int] = {}
        for receipt in self.receipts:
            for member, share in receipt.split().items():
                balances[member] = balances.get(member, 0) - share
            balances[receipt.payer_id] = (
                balances.get(receipt.payer_id, 0) + receipt.total_cents
            )
        return {m: b for m, b in balances.items() if b != 0}

    def who_owes_whom(self) -> list[dict[str, Any]]:
        """Minimal settlement transfers that zero every balance."""
        return settle_balances(self.balances())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "group_key": self.group_key,
            "title": self.title,
            "receipts": [r.to_dict() for r in self.receipts],
            "created_ts": self.created_ts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SplitBill":
        return cls(
            id=str(data["id"]),
            group_key=str(data["group_key"]),
            title=str(data["title"]),
            receipts=[Receipt.from_dict(r) for r in data.get("receipts", [])],
            created_ts=float(data.get("created_ts", 0.0)),
        )


# ── persistence ─────────────────────────────────────────────────────────
class MiniApp2Store:
    """JSON persistence for the second-wave mini-apps.

    One file per collection per group: ``<group>/<kind>.json``.
    """

    _KINDS = ("carpools", "rotas", "listings", "leagues", "bills")

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else default_miniapps2_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, group_key: str, kind: str) -> Path:
        safe = "".join(c for c in group_key if c.isalnum() or c in "-_")
        folder = self.data_dir / (safe or "default")
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{kind}.json"

    def _load_raw(self, group_key: str, kind: str) -> list[dict[str, Any]]:
        path = self._path(group_key, kind)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        return data if isinstance(data, list) else []

    def _save_raw(
        self, group_key: str, kind: str, items: list[dict[str, Any]]
    ) -> None:
        self._path(group_key, kind).write_text(
            json.dumps(items, indent=2), encoding="utf-8"
        )

    # Generic put/get per kind, keyed on the object's ``id``.
    def put(self, group_key: str, kind: str, obj: Any) -> None:
        if kind not in self._KINDS:
            raise ValueError(f"unknown mini-app kind {kind!r}")
        items = self._load_raw(group_key, kind)
        items = [i for i in items if i.get("id") != obj.id]
        items.append(obj.to_dict())
        self._save_raw(group_key, kind, items)

    def get(self, group_key: str, kind: str, obj_id: str) -> dict[str, Any] | None:
        for item in self._load_raw(group_key, kind):
            if item.get("id") == obj_id:
                return item
        return None

    def list(self, group_key: str, kind: str) -> list[dict[str, Any]]:
        if kind not in self._KINDS:
            raise ValueError(f"unknown mini-app kind {kind!r}")
        return self._load_raw(group_key, kind)

    def remove(self, group_key: str, kind: str, obj_id: str) -> bool:
        items = self._load_raw(group_key, kind)
        kept = [i for i in items if i.get("id") != obj_id]
        if len(kept) == len(items):
            return False
        self._save_raw(group_key, kind, kept)
        return True
