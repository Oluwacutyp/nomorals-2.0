"""Life archive — the owner's private, append-only life log.

Events from any source (journal, health, finance, messages, …) land
here with their provenance intact: *where* each entry came from and
*when* it was recorded. The archive is private by design:

* Owner-only: every public method requires ``owner=True`` (keyword),
  otherwise :class:`PermissionError`. The default is ``False`` so a
  caller that forgets the flag is denied, not silently admitted.
* No public by default: there is deliberately NO publish method on
  this class. Publishing life entries — if ever wanted — goes through
  an explicit, separate publicface flow, never this module.

Storage: SQLite, ``:memory:`` by default. Pass a file path for a
durable archive. ``search`` is substring matching over ``text`` +
``kind`` (case-insensitive); ``timeline`` returns newest-first.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from typing import Any

__all__ = ["LifeArchive"]


def _require_owner(owner: bool) -> None:
    if owner is not True:
        raise PermissionError("life archive is owner-only")


class LifeArchive:
    """SQLite life-event archive. Private; no publish method exists."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._db = sqlite3.connect(db_path)
        self._db.row_factory = sqlite3.Row
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS life_events (
                   id TEXT PRIMARY KEY, source TEXT NOT NULL,
                   kind TEXT NOT NULL, text TEXT NOT NULL,
                   ts REAL NOT NULL, provenance TEXT NOT NULL,
                   private INTEGER NOT NULL, created_ts REAL NOT NULL)""")
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_life_ts ON life_events(ts)")
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_life_kind ON life_events(kind)")
        self._db.commit()

    # ── write ────────────────────────────────────────────────────
    def record_event(self, source: str, kind: str, text: str,
                     ts: float | None = None, provenance: str = "",
                     private: bool = True, *, owner: bool = False) -> str:
        """Record one event; return its id.

        ``provenance`` is stored verbatim (e.g. ``"journal:2026-10-16"``
        or ``"import:health-app"``) so every entry stays attributable
        to its source. ``private`` defaults to True.
        """
        _require_owner(owner)
        if not (source or "").strip():
            raise ValueError("source is required")
        if not (kind or "").strip():
            raise ValueError("kind is required")
        if not (text or "").strip():
            raise ValueError("text is required")
        event_id = uuid.uuid4().hex[:12]
        now = time.time()
        self._db.execute(
            "INSERT INTO life_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (event_id, source, kind, text,
             float(ts if ts is not None else now), provenance,
             1 if private else 0, now))
        self._db.commit()
        return event_id

    # ── reads ────────────────────────────────────────────────────
    def get(self, event_id: str, *, owner: bool = False) -> dict[str, Any] | None:
        """Fetch one event by id, or ``None`` when absent."""
        _require_owner(owner)
        row = self._db.execute(
            "SELECT * FROM life_events WHERE id = ?", (event_id,)).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def timeline(self, limit: int = 50, kind: str | None = None, *,
                 owner: bool = False) -> list[dict[str, Any]]:
        """Newest-first events, optionally filtered by ``kind``."""
        _require_owner(owner)
        if limit < 0:
            raise ValueError("limit must be >= 0")
        if kind is None:
            rows = self._db.execute(
                "SELECT * FROM life_events ORDER BY ts DESC, created_ts DESC "
                "LIMIT ?", (limit,)).fetchall()
        else:
            rows = self._db.execute(
                "SELECT * FROM life_events WHERE kind = ? "
                "ORDER BY ts DESC, created_ts DESC LIMIT ?", (kind, limit)).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def search(self, query: str, *, owner: bool = False) -> list[dict[str, Any]]:
        """Substring search (case-insensitive) over ``text`` + ``kind``."""
        _require_owner(owner)
        needle = f"%{(query or '').lower()}%"
        rows = self._db.execute(
            "SELECT * FROM life_events "
            "WHERE LOWER(text) LIKE ? OR LOWER(kind) LIKE ? "
            "ORDER BY ts DESC, created_ts DESC",
            (needle, needle)).fetchall()
        return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"], "source": row["source"], "kind": row["kind"],
            "text": row["text"], "ts": row["ts"],
            "provenance": row["provenance"],
            "private": bool(row["private"]), "created_ts": row["created_ts"],
        }

    def close(self) -> None:
        self._db.close()
