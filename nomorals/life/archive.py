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
The archive also runs over the document-store seam (Mongo program
P5 — ``life-archive__life_events``) when the composition layer
injects a port; this module is layer L0 and never imports storage.
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


#: The archive's own DDL — the native path executes it and the seam
#: mapping is derived from it, so the two can never drift. (The ts /
#: kind indexes stay native-path concerns; the seam reads by find.)
_DDL = """CREATE TABLE IF NOT EXISTS life_events (
       id TEXT PRIMARY KEY, source TEXT NOT NULL,
       kind TEXT NOT NULL, text TEXT NOT NULL,
       ts REAL NOT NULL, provenance TEXT NOT NULL,
       private INTEGER NOT NULL, created_ts REAL NOT NULL)"""


class LifeArchive:
    """Life-event archive. Private; no publish method exists.

    Native path: a SQLite file (or ``:memory:``). When the
    composition layer injects a ``port`` (Mongo program P5 — a
    SidecarPort built by ``storage.sidecars.sidecar_port``, from
    this module's own ``_DDL``), the same owner-gated API runs over
    the seam in the ``life-archive__life_events`` collection
    instead, and a file-backed archive's existing rows are imported
    once at construction. This module is layer L0 (stdlib only):
    the port arrives injected and is used duck-typed; the module
    never imports storage.
    """

    def __init__(self, db_path: str = ":memory:", *,
                 port: Any = None) -> None:
        self.db_path = db_path
        self._db: sqlite3.Connection | None = None
        self._port: Any = None
        if port is not None:
            table = port.tables["life_events"]

            def write_events(rows: list[dict[str, Any]]) -> int:
                for row in rows:
                    table.put(row)
                return len(rows)

            port.ensure_imported({"life_events": write_events})
            self._port = port
            return
        self._db = sqlite3.connect(db_path)
        self._db.row_factory = sqlite3.Row
        self._db.execute(_DDL)
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
        if self._port is not None:
            self._port.tables["life_events"].put({
                "id": event_id, "source": source, "kind": kind,
                "text": text,
                "ts": float(ts if ts is not None else now),
                "provenance": provenance,
                "private": 1 if private else 0, "created_ts": now})
            return event_id
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
        if self._port is not None:
            row = self._port.tables["life_events"].get(event_id)
            return self._row_to_dict(row) if row is not None else None
        row = self._db.execute(
            "SELECT * FROM life_events WHERE id = ?", (event_id,)).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def timeline(self, limit: int = 50, kind: str | None = None, *,
                 owner: bool = False) -> list[dict[str, Any]]:
        """Newest-first events, optionally filtered by ``kind``."""
        _require_owner(owner)
        if limit < 0:
            raise ValueError("limit must be >= 0")
        if self._port is not None:
            rows = self._port.tables["life_events"].all()
            if kind is not None:
                rows = [r for r in rows if r.get("kind") == kind]
            # ts DESC, created_ts DESC — the native ORDER BY.
            rows.sort(key=lambda r: (r["ts"], r["created_ts"]),
                      reverse=True)
            return [self._row_to_dict(r) for r in rows[:limit]]
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
        if self._port is not None:
            needle = (query or "").lower()
            rows = [r for r in self._port.tables["life_events"].all()
                    if needle in str(r.get("text") or "").lower()
                    or needle in str(r.get("kind") or "").lower()]
            rows.sort(key=lambda r: (r["ts"], r["created_ts"]),
                      reverse=True)
            return [self._row_to_dict(r) for r in rows]
        needle = f"%{(query or '').lower()}%"
        rows = self._db.execute(
            "SELECT * FROM life_events "
            "WHERE LOWER(text) LIKE ? OR LOWER(kind) LIKE ? "
            "ORDER BY ts DESC, created_ts DESC",
            (needle, needle)).fetchall()
        return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        return {
            "id": row["id"], "source": row["source"], "kind": row["kind"],
            "text": row["text"], "ts": row["ts"],
            "provenance": row["provenance"],
            "private": bool(row["private"]), "created_ts": row["created_ts"],
        }

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
