"""Versioned sync store: the durable unit of replication.

Every record carries ``updated_at`` (unix seconds, set by the writer) and
``device_id``. Deletes are tombstones, not removals, so deletes replicate.

Replication uses a monotonic per-store sequence number (``seq``), assigned
on every local write — including writes applied from a peer. The engine's
push/pull cursors are seqs, not timestamps, so a backdated write (explicit
old ``updated_at``, phone/cloud clock skew) can never slip past a cursor
unseen. ``updated_at``/``device_id`` remain the conflict-resolution key
(last-write-wins); ``seq`` is only the "what changed since X" cursor.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from ..core.logging_setup import get_logger
from ..storage.db import Database

__all__ = ["SyncRecord", "SyncStore"]

_log = get_logger(__name__)

TABLE = "sync_records"


def ensure_column(db: Database, table: str, column: str, ddl: str) -> None:
    """Add ``column`` to ``table`` if missing.

    Tolerates losing a creation race with another thread/process building
    the same schema: if the ALTER fails, the column is re-checked and the
    error only propagates when the column is still genuinely absent.
    """
    cols = {r["name"] for r in db.table_info(table)}
    if column in cols:
        return
    try:
        db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
    except Exception:  # noqa: BLE001 - re-checked below; real errors re-raise
        cols = {r["name"] for r in db.table_info(table)}
        if column not in cols:
            raise
        _log.debug("column %s.%s appeared via a concurrent migration",
                   table, column)


@dataclass
class SyncRecord:
    key: str
    value: dict[str, Any]
    updated_at: float
    device_id: str
    deleted: bool = False
    # Local replication cursor, assigned by SyncStore on write. Not part of
    # the LWW identity — two records are "the same" when (updated_at,
    # device_id, value, deleted) match, regardless of seq.
    seq: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": dict(self.value),
            "updated_at": self.updated_at,
            "device_id": self.device_id,
            "deleted": self.deleted,
            "seq": self.seq,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SyncRecord":
        """Rebuild from :meth:`to_dict` output (hub HTTP wire format).

        ``seq`` is informational here: the receiver's store assigns its own
        seq on write, and LWW identity ignores it.
        """
        return cls(
            key=str(data.get("key") or ""),
            value=dict(data.get("value") or {}),
            updated_at=float(data.get("updated_at") or 0.0),
            device_id=str(data.get("device_id") or ""),
            deleted=bool(data.get("deleted")),
            seq=int(data.get("seq") or 0),
        )

    @staticmethod
    def wins(a: "SyncRecord", b: "SyncRecord") -> "SyncRecord":
        """Last-write-wins; ties break deterministically by device_id."""
        if (a.updated_at, a.device_id) >= (b.updated_at, b.device_id):
            return a
        return b


class SyncStore:
    """SQLite-backed versioned KV store."""

    def __init__(self, db: Database, device_id: str) -> None:
        if not device_id:
            raise ValueError("device_id is required")
        self.db = db
        self.device_id = device_id
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.db.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL DEFAULT '{{}}',
                updated_at REAL NOT NULL DEFAULT 0,
                device_id TEXT NOT NULL DEFAULT '',
                deleted INTEGER NOT NULL DEFAULT 0,
                seq INTEGER NOT NULL DEFAULT 0
            )"""
        )
        self.db.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_updated ON {TABLE}(updated_at)"
        )
        # Migration first: the seq index below requires the column to exist
        # on pre-seq databases.
        self._ensure_seq_column()
        self.db.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_seq ON {TABLE}(seq)"
        )

    def _ensure_seq_column(self) -> None:
        """Migration for pre-seq databases: add the column, then backfill.

        Backfill order is (updated_at, rowid) continuing from the current
        max seq, so rows that predate the seq era keep a sensible order and
        no two rows share a seq.
        """
        ensure_column(self.db, TABLE, "seq", "INTEGER NOT NULL DEFAULT 0")
        with self.db.transaction():
            pending = self.db.query(
                f"SELECT rowid AS rid FROM {TABLE} "
                "WHERE seq = 0 ORDER BY updated_at, rowid"
            )
            if not pending:
                return
            start = int(self.db.scalar(
                f"SELECT COALESCE(MAX(seq), 0) FROM {TABLE}", default=0))
            for i, row in enumerate(pending, start=1):
                self.db.execute(
                    f"UPDATE {TABLE} SET seq = ? WHERE rowid = ?",
                    (start + i, row["rid"]),
                )
            _log.info("sync store: backfilled seq for %d rows", len(pending))

    def put(
        self,
        key: str,
        value: dict[str, Any],
        *,
        updated_at: float | None = None,
    ) -> SyncRecord:
        if not key:
            raise ValueError("key is required")
        rec = SyncRecord(
            key=key,
            value=value,
            updated_at=updated_at if updated_at is not None else time.time(),
            device_id=self.device_id,
            deleted=False,
        )
        self._upsert(rec)
        return rec

    def delete(self, key: str) -> SyncRecord:
        """Write a tombstone. The key disappears from reads but replicates."""
        rec = SyncRecord(
            key=key, value={}, updated_at=time.time(),
            device_id=self.device_id, deleted=True,
        )
        self._upsert(rec)
        return rec

    def get(self, key: str) -> SyncRecord | None:
        row = self.db.query_one(f"SELECT * FROM {TABLE} WHERE key=?", (key,))
        if not row or row["deleted"]:
            return None
        return self._row_to_record(row)

    def get_record(self, key: str) -> SyncRecord | None:
        """Like get(), but returns tombstones too (for replication)."""
        row = self.db.query_one(f"SELECT * FROM {TABLE} WHERE key=?", (key,))
        return self._row_to_record(row) if row else None

    def list_changed_since(self, since: float) -> list[SyncRecord]:
        rows = self.db.query(
            f"SELECT * FROM {TABLE} WHERE updated_at > ? ORDER BY updated_at",
            (since,),
        )
        return [self._row_to_record(r) for r in rows]

    def list_since_seq(self, seq: int) -> list[SyncRecord]:
        """Records written after replication cursor ``seq``, in seq order.

        This is the cursor the sync engine uses. Unlike
        :meth:`list_changed_since` it cannot miss a backdated write
        (explicit old ``updated_at``, phone/cloud clock skew): the cursor
        is a monotonic local sequence, not a timestamp.
        """
        rows = self.db.query(
            f"SELECT * FROM {TABLE} WHERE seq > ? ORDER BY seq",
            (seq,),
        )
        return [self._row_to_record(r) for r in rows]

    def max_seq(self) -> int:
        return int(self.db.scalar(
            f"SELECT COALESCE(MAX(seq), 0) FROM {TABLE}", default=0))

    def apply(self, rec: SyncRecord) -> bool:
        """Merge a remote record. Returns True if it changed local state."""
        if not rec.key:
            raise ValueError("record key is required")
        current = self.get_record(rec.key)
        if current is None:
            self._upsert(rec)
            return True
        # Identical record (e.g., our own push echoed back): no-op.
        if (current.updated_at == rec.updated_at
                and current.device_id == rec.device_id
                and current.value == rec.value
                and current.deleted == rec.deleted):
            return False
        winner = SyncRecord.wins(rec, current)
        if winner is rec:
            self._upsert(rec)
            return True
        return False

    def keys(self) -> list[str]:
        rows = self.db.query(
            f"SELECT key FROM {TABLE} WHERE deleted=0 ORDER BY key"
        )
        return [r["key"] for r in rows]

    def count(self) -> int:
        rows = self.db.query(
            f"SELECT COUNT(*) AS n FROM {TABLE} WHERE deleted=0"
        )
        return int(rows[0]["n"]) if rows else 0

    def _upsert(self, rec: SyncRecord) -> None:
        # Every local write — own put/delete or a peer's record applied here
        # — advances the replication cursor. Pulled records MUST bump the
        # local seq too: that is what lets a hub fan records out to a third
        # device. The write and the seq assignment are one transaction, so
        # concurrent writers can never share or skip a seq.
        with self.db.transaction():
            nxt = int(self.db.scalar(
                f"SELECT COALESCE(MAX(seq), 0) FROM {TABLE}", default=0)) + 1
            self.db.execute(
                f"""INSERT INTO {TABLE} (key, value, updated_at, device_id, deleted, seq)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value=excluded.value,
                        updated_at=excluded.updated_at,
                        device_id=excluded.device_id,
                        deleted=excluded.deleted,
                        seq=excluded.seq""",
                (
                    rec.key,
                    json.dumps(rec.value, default=str),
                    rec.updated_at,
                    rec.device_id,
                    1 if rec.deleted else 0,
                    nxt,
                ),
            )
        rec.seq = nxt

    @staticmethod
    def _row_to_record(row: Any) -> SyncRecord:
        return SyncRecord(
            key=row["key"],
            value=json.loads(row["value"] or "{}"),
            updated_at=row["updated_at"],
            device_id=row["device_id"],
            deleted=bool(row["deleted"]),
            seq=int(row.get("seq") or 0),
        )
