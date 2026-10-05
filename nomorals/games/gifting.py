"""Player-to-player gifting: coins, gear, and shop items.

A gift is a two-step flow so nobody fat-fingers their myth katana away:

    /gift @name 100            → preview: "send 100 coins to Ada?"
    /gift confirm              → coins move, both sides get receipts
    /gift cancel               → drop the pending gift

    /gift @name katana_rare    → gear (by slug; must own it, unequips first)
    /gift @name potion         → shop item (must own at least one)

Pending gifts expire after 5 minutes. Every completed gift is logged to
``game_wallet`` (coins) or the gear ledger so balances stay auditable.
Transfers are atomic-ish: the giver's debit and the recipient's credit
happen in one DB transaction where the store supports it; on any
failure the giver is refunded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..core.ids import new_id
from ..core.logging_setup import get_logger

__all__ = ["Gift", "GiftStore", "GIFT_EXPIRY_S", "resolve_recipient"]

_log = get_logger(__name__)

#: Pending gifts die after 5 minutes — long enough to read the preview,
#: short enough that a stale confirm can't surprise anyone.
GIFT_EXPIRY_S = 5 * 60


@dataclass
class Gift:
    """One pending (or completed) gift."""

    id: str
    giver_key: str
    giver_name: str
    recipient_key: str
    recipient_name: str
    kind: str            # "coins" | "gear" | "item"
    ref: str             # amount for coins, slug for gear/item
    amount: int = 0      # coins only
    created_at: float = field(default_factory=time.time)
    status: str = "pending"   # pending | done | cancelled | expired

    def is_expired(self) -> bool:
        return time.time() - self.created_at > GIFT_EXPIRY_S

    def describe(self) -> str:
        if self.kind == "coins":
            what = f"{self.amount} coins"
        elif self.kind == "gear":
            what = f"gear {self.ref}"
        else:
            what = f"{self.ref} ×1"
        return f"{what} → {self.recipient_name}"


class GiftStore:
    """Pending-gift ledger. One pending gift per giver at a time —
    starting a new one replaces the old."""

    def __init__(self, db: Any) -> None:
        self.db = db
        self._ensure()

    def _ensure(self) -> None:
        if self.db is None:
            return
        try:
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS game_gifts ("
                "id TEXT PRIMARY KEY, giver_key TEXT NOT NULL, "
                "giver_name TEXT NOT NULL DEFAULT '', "
                "recipient_key TEXT NOT NULL, "
                "recipient_name TEXT NOT NULL DEFAULT '', "
                "kind TEXT NOT NULL, ref TEXT NOT NULL DEFAULT '', "
                "amount INTEGER NOT NULL DEFAULT 0, "
                "created_at REAL NOT NULL, status TEXT NOT NULL DEFAULT "
                "'pending')")
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_game_gifts_giver "
                "ON game_gifts (giver_key, status)")
        except Exception:  # noqa: BLE001
            _log.debug("game_gifts ensure failed", exc_info=True)

    def pending_for(self, giver_key: str) -> Gift | None:
        """The giver's live pending gift, if any (expired ones are swept)."""
        if self.db is None:
            return None
        try:
            row = self.db.query_one(
                "SELECT * FROM game_gifts WHERE giver_key = ? "
                "AND status = 'pending' ORDER BY created_at DESC LIMIT 1",
                (giver_key,))
        except Exception:  # noqa: BLE001
            return None
        if row is None:
            return None
        gift = _row_to_gift(row)
        if gift.is_expired():
            self._set_status(gift.id, "expired")
            return None
        return gift

    def create(self, giver_key: str, giver_name: str,
               recipient_key: str, recipient_name: str,
               kind: str, ref: str, amount: int = 0) -> Gift:
        """Stage a new pending gift, replacing any older pending one."""
        self._ensure()
        gift = Gift(
            id=new_id("gift"), giver_key=giver_key, giver_name=giver_name,
            recipient_key=recipient_key, recipient_name=recipient_name,
            kind=kind, ref=ref, amount=int(amount))
        if self.db is None:
            return gift
        try:
            with self.db.transaction():
                self.db.execute(
                    "UPDATE game_gifts SET status = 'cancelled' "
                    "WHERE giver_key = ? AND status = 'pending'",
                    (giver_key,))
                self.db.execute(
                    "INSERT INTO game_gifts (id, giver_key, giver_name, "
                    "recipient_key, recipient_name, kind, ref, amount, "
                    "created_at, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, "
                    "'pending')",
                    (gift.id, gift.giver_key, gift.giver_name,
                     gift.recipient_key, gift.recipient_name, gift.kind,
                     gift.ref, gift.amount, gift.created_at))
        except Exception:  # noqa: BLE001
            _log.debug("game_gifts create failed", exc_info=True)
        return gift

    def _set_status(self, gift_id: str, status: str) -> None:
        if self.db is None:
            return
        try:
            self.db.execute(
                "UPDATE game_gifts SET status = ? WHERE id = ?",
                (status, gift_id))
        except Exception:  # noqa: BLE001
            pass

    def mark_done(self, gift_id: str) -> None:
        self._set_status(gift_id, "done")

    def cancel(self, giver_key: str) -> Gift | None:
        gift = self.pending_for(giver_key)
        if gift is None:
            return None
        self._set_status(gift.id, "cancelled")
        gift.status = "cancelled"
        return gift

    def history(self, player_key: str, limit: int = 10) -> list[Gift]:
        """Recent completed gifts involving this player."""
        if self.db is None:
            return []
        try:
            rows = self.db.query(
                "SELECT * FROM game_gifts WHERE status = 'done' AND "
                "(giver_key = ? OR recipient_key = ?) "
                "ORDER BY created_at DESC LIMIT ?",
                (player_key, player_key, limit)) or []
        except Exception:  # noqa: BLE001
            return []
        return [_row_to_gift(r) for r in rows]


def _row_to_gift(row: Any) -> Gift:
    get = row.get if hasattr(row, "get") else lambda k, d="": row[k]
    return Gift(
        id=get("id"), giver_key=get("giver_key"),
        giver_name=get("giver_name") or "",
        recipient_key=get("recipient_key"),
        recipient_name=get("recipient_name") or "",
        kind=get("kind"), ref=get("ref") or "",
        amount=int(get("amount") or 0),
        created_at=float(get("created_at") or 0),
        status=get("status") or "pending")


def resolve_recipient(store: Any, name: str) -> tuple[Any | None, str]:
    """Find a player profile by display-name mention.

    Returns (profile, message). Strips a leading @. Exact
    case-insensitive match wins; a single partial match is accepted;
    zero or many matches return (None, helpful message).
    """
    label = (name or "").strip().lstrip("@")
    if not label:
        return None, "gift to whom? /gift @name <coins|gear|item>"
    try:
        profiles = store.all(limit=500)
    except Exception:  # noqa: BLE001
        profiles = []
    exact = [p for p in profiles
             if (p.name or "").lower() == label.lower()]
    if len(exact) == 1:
        return exact[0], ""
    partial = [p for p in profiles
               if label.lower() in (p.name or "").lower()]
    if len(exact) > 1:
        names = ", ".join(p.name for p in exact[:5])
        return None, (f"several players match {label!r}: {names} — "
                      f"be more specific.")
    if len(partial) == 1:
        return partial[0], ""
    if len(partial) > 1:
        names = ", ".join(p.name for p in partial[:5])
        return None, (f"several players match {label!r}: {names} — "
                      f"be more specific.")
    return None, (f"no player found matching {label!r} — they need to "
                  f"have played at least once.")
