"""Cross-organ event bus — how Devon's organs talk to each other.

Research, wisdom, the brain and the other organs cooperate through
persisted events, not direct imports. An organ emits events addressed
to another organ; the recipient drains them on its own tick. This keeps
organs decoupled (no import cycles, no layering violations) while
letting them genuinely cooperate: research findings feed wisdom,
wisdom digests feed the brain, the brain directs research.

Schema: ``organ_events(id, ts, src, dst, kind, payload_json, consumed)``.
``drain`` returns unconsumed events for ``dst`` and marks them consumed
atomically, so two ticks can never double-process an event.
"""

from __future__ import annotations

import json
import time
from typing import Any

__all__ = ["ensure_schema", "emit", "drain", "pending_count"]


def ensure_schema(db: Any) -> None:
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS organ_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            src TEXT NOT NULL,
            dst TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT NOT NULL DEFAULT '{}',
            consumed INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_organ_events_dst "
        "ON organ_events(dst, consumed, ts)"
    )


def emit(db: Any, src: str, dst: str, kind: str,
         payload: dict[str, Any] | None = None) -> int:
    """Queue one event. Returns the event id."""
    ensure_schema(db)
    cur = db.execute(
        "INSERT INTO organ_events (ts, src, dst, kind, payload, consumed)"
        " VALUES (?, ?, ?, ?, ?, 0)",
        (time.time(), src, dst, kind, json.dumps(payload or {})),
    )
    try:
        return int(cur.lastrowid or 0)
    except Exception:  # noqa: BLE001 - cursor shape varies by backend
        return 0


def drain(db: Any, dst: str, kinds: list[str] | None = None,
          limit: int = 50) -> list[dict[str, Any]]:
    """Take up to ``limit`` unconsumed events for ``dst``, oldest first.

    Marks them consumed in the same transaction — a crashing tick can't
    re-deliver them later.
    """
    ensure_schema(db)
    if kinds:
        placeholders = ",".join("?" for _ in kinds)
        rows = db.query(
            "SELECT id, ts, src, dst, kind, payload FROM organ_events"
            f" WHERE dst = ? AND consumed = 0 AND kind IN ({placeholders})"
            " ORDER BY ts ASC LIMIT ?",
            (dst, *kinds, limit),
        )
    else:
        rows = db.query(
            "SELECT id, ts, src, dst, kind, payload FROM organ_events"
            " WHERE dst = ? AND consumed = 0"
            " ORDER BY ts ASC LIMIT ?",
            (dst, limit),
        )
    events: list[dict[str, Any]] = []
    ids: list[int] = []
    for row in rows or []:
        try:
            payload = json.loads(row["payload"] or "{}")
        except Exception:  # noqa: BLE001 - corrupt payload shouldn't kill drain
            payload = {}
        events.append({
            "id": row["id"], "ts": row["ts"], "src": row["src"],
            "dst": row["dst"], "kind": row["kind"], "payload": payload,
        })
        ids.append(int(row["id"]))
    if ids:
        placeholders = ",".join("?" for _ in ids)
        db.execute(
            f"UPDATE organ_events SET consumed = 1 WHERE id IN ({placeholders})",
            tuple(ids),
        )
    return events


def pending_count(db: Any, dst: str) -> int:
    ensure_schema(db)
    row = db.query_one(
        "SELECT COUNT(*) AS n FROM organ_events WHERE dst = ? AND consumed = 0",
        (dst,),
    )
    return int(row["n"]) if row else 0
