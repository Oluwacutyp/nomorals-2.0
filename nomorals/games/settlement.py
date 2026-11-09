"""The settlement ledger: one durable record per finished room.

A game's finish moves real value — coins, points, wager payouts.
Before this ledger, the settle was a loop of independent writes:
a crash (or one player's credit raising) halfway through left the
early players paid and the rest unpaid, with the room already
marked finished — a silent half-pay that no retry could repair,
because nothing recorded what the settle was supposed to do.

The contract now:

* **Plan first.** When a finish begins, a row is written with
  status ``planned`` (its own committed write, before any credit).
* **Atomic apply.** Every games-wallet write of the settle runs
  inside ONE database transaction together with the flip to
  ``applied`` and the receipts. A failure anywhere rolls the
  whole settle back — nobody is paid — and the ``planned`` row
  remains as the durable retry signal.
* **Receipts forever.** The ``applied`` row carries the per-player
  receipts (coins, reason, points, outcome, score, balance). A
  finish that runs again after the commit — the crash window
  where the room row never got its ``finished`` write — resolves
  to those receipts and moves nothing twice.

The table is runtime-created (the living-cards pattern):
``storage/migrations.py`` is never touched. The group-purse mirror
(the coin sink) is a second store and cannot join the SQLite
transaction; it fires after the commit, per player, and each
fired obligation is marked in the receipts (``sink_done``) so a
recovery pass fires only what is still owed.
"""
from __future__ import annotations

import json
import time
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["SettlementStore", "SETTLEMENT_PLANNED", "SETTLEMENT_APPLIED"]

SETTLEMENT_PLANNED = "planned"
SETTLEMENT_APPLIED = "applied"

_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS game_settlements ("
    "room_id TEXT PRIMARY KEY, game TEXT NOT NULL DEFAULT '', "
    "chat_key TEXT NOT NULL DEFAULT '', "
    "status TEXT NOT NULL DEFAULT 'planned', "
    "receipts TEXT NOT NULL DEFAULT '{}', "
    "created_at REAL NOT NULL DEFAULT 0, "
    "applied_at REAL NOT NULL DEFAULT 0)")


class SettlementStore:
    """Durable settlement records over the context db.

    Every method is fail-soft in the read direction (a missing
    table or dead db reads as "no record" — the engine then runs
    its legacy settle) and honest in the write direction: a
    failed ``begin`` returns False so the caller knows the
    atomicity guarantee is not in force for this settle.
    """

    def __init__(self, db: Any) -> None:
        self._db = db
        self._ready = False
        if db is not None:
            try:
                db.execute(_TABLE_SQL)
                self._ready = True
            except Exception as exc:  # noqa: BLE001 - degrade honestly
                _log.debug("game_settlements table unavailable: %s", exc)

    @property
    def available(self) -> bool:
        return self._ready and self._db is not None

    def get(self, room_id: str) -> dict[str, Any] | None:
        """The settlement record for one room, or None."""
        if not self.available:
            return None
        try:
            row = self._db.query_one(
                "SELECT * FROM game_settlements WHERE room_id = ?",
                (str(room_id),))
        except Exception as exc:  # noqa: BLE001
            _log.debug("settlement read failed: %s", exc)
            return None
        if row is None:
            return None
        try:
            receipts = json.loads(row["receipts"] or "{}")
        except (TypeError, ValueError):
            receipts = {}
        return {"room_id": row["room_id"], "game": row["game"],
                "chat_key": row["chat_key"], "status": row["status"],
                "receipts": receipts,
                "created_at": float(row["created_at"] or 0),
                "applied_at": float(row["applied_at"] or 0)}

    def begin(self, room_id: str, game: str, chat_key: str) -> bool:
        """Write the ``planned`` row (idempotent). Returns whether
        the atomic settle is in force for this room."""
        if not self.available:
            return False
        try:
            self._db.execute(
                "INSERT OR IGNORE INTO game_settlements "
                "(room_id, game, chat_key, status, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (str(room_id), str(game), str(chat_key),
                 SETTLEMENT_PLANNED, time.time()))
            return True
        except Exception as exc:  # noqa: BLE001
            _log.debug("settlement begin failed: %s", exc)
            return False

    def mark_applied(self, room_id: str,
                     receipts: dict[str, Any]) -> None:
        """Flip the row to ``applied`` with its receipts. The
        engine calls this INSIDE the settle transaction, so the
        flip commits — or rolls back — with the credits."""
        if not self.available:
            return
        self._db.execute(
            "UPDATE game_settlements SET status = ?, receipts = ?, "
            "applied_at = ? WHERE room_id = ?",
            (SETTLEMENT_APPLIED, json.dumps(receipts), time.time(),
             str(room_id)))

    def save_receipts(self, room_id: str,
                      receipts: dict[str, Any]) -> None:
        """Persist updated receipts on an applied row (the sink
        bookkeeping that lands after the settle commit)."""
        if not self.available:
            return
        try:
            self._db.execute(
                "UPDATE game_settlements SET receipts = ? "
                "WHERE room_id = ?",
                (json.dumps(receipts), str(room_id)))
        except Exception as exc:  # noqa: BLE001
            _log.debug("settlement receipts save failed: %s", exc)

    def pending_for(self, room_id: str) -> bool:
        """True when a settlement row exists in ``planned`` state —
        a settle that was begun but never landed (crash/abort)."""
        rec = self.get(room_id)
        return bool(rec and rec["status"] == SETTLEMENT_PLANNED)
