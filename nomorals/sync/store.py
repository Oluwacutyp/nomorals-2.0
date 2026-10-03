"""Versioned sync store: the durable unit of replication.

Every record carries ``updated_at`` (unix seconds, set by the writer) and
``device_id``. Deletes are tombstones, not removals, so deletes replicate.
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


@dataclass
class SyncRecord:
    key: str
    value: dict[str, Any]
    updated_at: float
    device_id: str
    deleted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": dict(self.value),
            "updated_at": self.updated_at,
            "device_id": self.device_id,
            "deleted": self.deleted,
        }

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
                deleted INTEGER NOT NULL DEFAULT 0
            )"""
        )
        self.db.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_updated ON {TABLE}(updated_at)"
        )

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
        self.db.execute(
            f"""INSERT INTO {TABLE} (key, value, updated_at, device_id, deleted)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value,
                    updated_at=excluded.updated_at,
                    device_id=excluded.device_id,
                    deleted=excluded.deleted""",
            (
                rec.key,
                json.dumps(rec.value, default=str),
                rec.updated_at,
                rec.device_id,
                1 if rec.deleted else 0,
            ),
        )

    @staticmethod
    def _row_to_record(row: Any) -> SyncRecord:
        return SyncRecord(
            key=row["key"],
            value=json.loads(row["value"] or "{}"),
            updated_at=row["updated_at"],
            device_id=row["device_id"],
            deleted=bool(row["deleted"]),
        )
