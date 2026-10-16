"""Ajo / esusu rotating savings: pure group finance logic.

An ajo (Yoruba) / esusu circle is a rotating savings group: every member
contributes a fixed amount each round, and one member — in turn order —
takes the whole pot that round. After a full cycle every member has taken
the pot exactly once, so the ledger nets to zero: money is conserved, it
only rotates.

What this module is
-------------------
* Roster + turn order (first-come order by default, reorderable by the
  group admin before the first round starts).
* Contributions per round, with per-round "who has paid" tracking.
* Payouts: ``next_payout()`` names the current recipient and the pot;
  ``record_payout()`` books it and advances the turn.
* Reminders: members who have not yet contributed this round.
* Ledger: every contribution / payout is an append-only entry;
  ``member_balance()`` = contributed − received, so a full cycle nets
  every member to exactly zero.
* JSON persistence under ``~/.devon/community/ajo/`` (or a ``data_dir``
  passed in — tests pass a tmp dir). No owner database, no owner data:
  members are identified only by the platform user id / display name
  that arrives with the chat message itself.

Isolation: stdlib + ``nomorals.core`` only. No owner-data imports.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.ids import new_short_id

__all__ = [
    "AjoMember",
    "LedgerEntry",
    "AjoGroup",
    "AjoStore",
    "create_ajo",
    "default_ajo_dir",
]


def default_ajo_dir() -> Path:
    """Default on-disk home for ajo state: inside the community dir."""
    return Path.home() / ".devon" / "community" / "ajo"


@dataclass
class AjoMember:
    """One circle member. Identity is group-scoped only."""

    user_id: str
    display_name: str
    joined_ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AjoMember":
        return cls(
            user_id=str(data["user_id"]),
            display_name=str(data.get("display_name", "")),
            joined_ts=float(data.get("joined_ts", 0.0)),
        )


@dataclass
class LedgerEntry:
    """One append-only ledger line: a contribution or a payout."""

    id: str
    kind: str  # "contribution" | "payout"
    member_id: str
    amount_cents: int
    round_no: int
    ts: float
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LedgerEntry":
        return cls(
            id=str(data["id"]),
            kind=str(data["kind"]),
            member_id=str(data["member_id"]),
            amount_cents=int(data["amount_cents"]),
            round_no=int(data["round_no"]),
            ts=float(data.get("ts", 0.0)),
            note=str(data.get("note", "")),
        )


@dataclass
class AjoGroup:
    """One rotating-savings circle. Pure logic; persistence lives in AjoStore."""

    id: str
    group_key: str
    name: str
    contribution_cents: int
    currency: str = "NGN"
    members: list[AjoMember] = field(default_factory=list)
    turn_order: list[str] = field(default_factory=list)
    current_turn: int = 0  # index into turn_order: who takes the next pot
    current_round: int = 1
    started: bool = False
    ledger: list[LedgerEntry] = field(default_factory=list)
    created_ts: float = 0.0

    # ── roster ──────────────────────────────────────────────────────
    def add_member(self, user_id: str, display_name: str) -> AjoMember:
        """Add a member. New joiners append to the end of the turn order."""
        if any(m.user_id == user_id for m in self.members):
            raise ValueError(f"member {user_id!r} is already in this circle")
        member = AjoMember(
            user_id=user_id, display_name=display_name, joined_ts=time.time()
        )
        self.members.append(member)
        self.turn_order.append(user_id)
        return member

    def remove_member(self, user_id: str) -> None:
        """Remove a member. Refused mid-cycle: it would break the pot math."""
        if self.started and self.ledger:
            raise ValueError("cannot remove a member mid-cycle: ledger is live")
        self.members = [m for m in self.members if m.user_id != user_id]
        self.turn_order = [u for u in self.turn_order if u != user_id]
        if self.current_turn >= len(self.turn_order):
            self.current_turn = 0

    def set_turn_order(self, order: list[str]) -> None:
        """Reorder turns. Only before the first contribution lands."""
        if self.ledger:
            raise ValueError("cannot reorder turns once the ledger is live")
        if sorted(order) != sorted(m.user_id for m in self.members):
            raise ValueError("turn order must list every member exactly once")
        self.turn_order = list(order)
        self.current_turn = 0

    def member(self, user_id: str) -> AjoMember | None:
        for m in self.members:
            if m.user_id == user_id:
                return m
        return None

    # ── rounds / contributions ──────────────────────────────────────
    @property
    def pot_cents(self) -> int:
        """Expected pot for one round: contribution × member count."""
        return self.contribution_cents * len(self.members)

    def contributed_this_round(self, user_id: str) -> int:
        """Cents this member has contributed in the current round."""
        return sum(
            e.amount_cents
            for e in self.ledger
            if e.kind == "contribution"
            and e.member_id == user_id
            and e.round_no == self.current_round
        )

    def round_complete(self) -> bool:
        """True when every member has contributed the full amount this round."""
        if not self.members:
            return False
        return all(
            self.contributed_this_round(m.user_id) >= self.contribution_cents
            for m in self.members
        )

    def record_contribution(
        self, user_id: str, amount_cents: int, note: str = ""
    ) -> LedgerEntry:
        """Book a contribution. Positive amounts only; overpay is allowed
        (it shows up in the member's balance) but never silently dropped."""
        if self.member(user_id) is None:
            raise ValueError(f"unknown member {user_id!r}")
        if amount_cents <= 0:
            raise ValueError("contribution must be a positive amount")
        entry = LedgerEntry(
            id=new_short_id("ajo"),
            kind="contribution",
            member_id=user_id,
            amount_cents=int(amount_cents),
            round_no=self.current_round,
            ts=time.time(),
            note=note,
        )
        self.ledger.append(entry)
        self.started = True
        return entry

    def due_reminders(self) -> list[dict[str, Any]]:
        """Members who have not yet fully contributed this round.

        Returns one reminder dict per owing member: who, how much is
        outstanding, and which round — the caller formats/sends it.
        """
        reminders = []
        for m in self.members:
            paid = self.contributed_this_round(m.user_id)
            if paid < self.contribution_cents:
                reminders.append(
                    {
                        "user_id": m.user_id,
                        "display_name": m.display_name,
                        "round_no": self.current_round,
                        "paid_cents": paid,
                        "due_cents": self.contribution_cents - paid,
                    }
                )
        return reminders

    # ── payouts ─────────────────────────────────────────────────────
    def next_payout(self) -> dict[str, Any] | None:
        """The next payout: recipient, pot, round. None with no members."""
        if not self.turn_order:
            return None
        recipient_id = self.turn_order[self.current_turn % len(self.turn_order)]
        recipient = self.member(recipient_id)
        return {
            "recipient_id": recipient_id,
            "recipient_name": recipient.display_name if recipient else recipient_id,
            "amount_cents": self.pot_cents,
            "round_no": self.current_round,
            "round_complete": self.round_complete(),
        }

    def record_payout(self, note: str = "") -> LedgerEntry:
        """Book the current payout and advance to the next round.

        Requires the round to be fully contributed first — a payout may
        never go out on an incomplete pot. Each round funds exactly one
        payout: after it, the turn moves to the next member AND the
        round advances, so the next payout is funded by fresh
        contributions. When the turn wraps, the full cycle is complete
        and every member nets to exactly zero.
        """
        if not self.turn_order:
            raise ValueError("no members: nobody to pay out to")
        if not self.round_complete():
            missing = [r["display_name"] for r in self.due_reminders()]
            raise ValueError(
                "round incomplete: still waiting on " + ", ".join(missing)
            )
        recipient_id = self.turn_order[self.current_turn % len(self.turn_order)]
        entry = LedgerEntry(
            id=new_short_id("ajo"),
            kind="payout",
            member_id=recipient_id,
            amount_cents=self.pot_cents,
            round_no=self.current_round,
            ts=time.time(),
            note=note,
        )
        self.ledger.append(entry)
        self.current_turn += 1
        self.current_round += 1
        if self.current_turn >= len(self.turn_order):
            # Full cycle done: turn order starts over for the next cycle.
            self.current_turn = 0
        return entry

    # ── balances ────────────────────────────────────────────────────
    def member_balance(self, user_id: str) -> int:
        """Net position in cents: contributed − received.

        Negative = has taken out more than put in (ahead in the rotation);
        positive = has put in more than taken out. A completed cycle nets
        every member to exactly zero when everyone paid the fixed amount.
        """
        if self.member(user_id) is None:
            raise ValueError(f"unknown member {user_id!r}")
        contributed = sum(
            e.amount_cents
            for e in self.ledger
            if e.kind == "contribution" and e.member_id == user_id
        )
        received = sum(
            e.amount_cents
            for e in self.ledger
            if e.kind == "payout" and e.member_id == user_id
        )
        return contributed - received

    def balances(self) -> dict[str, int]:
        """Net position for every member. Sums to zero over a full cycle."""
        return {m.user_id: self.member_balance(m.user_id) for m in self.members}

    def ledger_for(self, user_id: str) -> list[LedgerEntry]:
        return [e for e in self.ledger if e.member_id == user_id]

    # ── serialisation ────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "group_key": self.group_key,
            "name": self.name,
            "contribution_cents": self.contribution_cents,
            "currency": self.currency,
            "members": [m.to_dict() for m in self.members],
            "turn_order": list(self.turn_order),
            "current_turn": self.current_turn,
            "current_round": self.current_round,
            "started": self.started,
            "ledger": [e.to_dict() for e in self.ledger],
            "created_ts": self.created_ts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AjoGroup":
        return cls(
            id=str(data["id"]),
            group_key=str(data["group_key"]),
            name=str(data["name"]),
            contribution_cents=int(data["contribution_cents"]),
            currency=str(data.get("currency", "NGN")),
            members=[AjoMember.from_dict(m) for m in data.get("members", [])],
            turn_order=list(data.get("turn_order", [])),
            current_turn=int(data.get("current_turn", 0)),
            current_round=int(data.get("current_round", 1)),
            started=bool(data.get("started", False)),
            ledger=[LedgerEntry.from_dict(e) for e in data.get("ledger", [])],
            created_ts=float(data.get("created_ts", 0.0)),
        )


def create_ajo(
    group_key: str, name: str, contribution_cents: int, currency: str = "NGN"
) -> AjoGroup:
    """Create a circle. Contribution must be positive."""
    if contribution_cents <= 0:
        raise ValueError("contribution must be a positive amount")
    return AjoGroup(
        id=new_short_id("ajo"),
        group_key=group_key,
        name=name,
        contribution_cents=int(contribution_cents),
        currency=currency,
        created_ts=time.time(),
    )


class AjoStore:
    """JSON persistence for ajo circles under the community data dir."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else default_ajo_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, ajo_id: str) -> Path:
        safe = "".join(c for c in ajo_id if c.isalnum() or c in "-_")
        return self.data_dir / f"{safe}.json"

    def put(self, group: AjoGroup) -> None:
        self._path(group.id).write_text(
            json.dumps(group.to_dict(), indent=2), encoding="utf-8"
        )

    def get(self, ajo_id: str) -> AjoGroup | None:
        path = self._path(ajo_id)
        if not path.exists():
            return None
        return AjoGroup.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list(self, group_key: str = "") -> list[AjoGroup]:
        groups = []
        for path in sorted(self.data_dir.glob("*.json")):
            try:
                group = AjoGroup.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            if not group_key or group.group_key == group_key:
                groups.append(group)
        return groups

    def remove(self, ajo_id: str) -> bool:
        path = self._path(ajo_id)
        if path.exists():
            path.unlink()
            return True
        return False
