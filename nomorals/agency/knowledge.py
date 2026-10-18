"""Always-growing knowledge base (owner refinement).

The knowledge accumulation itself is the point: every system deposits
what it reads, explores, and learns — claims, findings, lessons —
into ONE durable store that is always handy for every other system.
Deduped by normalized content so ten sightings of one fact are one
fact, not ten. Entries carry source/system/provenance + freshness so
a disappeared URL is tracked as retrieval availability, not truth.
"""
from __future__ import annotations

import re
import time
import uuid
from typing import Any

__all__ = ["KnowledgeStore"]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


class KnowledgeStore:
    """Append-only, deduped knowledge over the injected DB handle."""

    def __init__(self, db: Any) -> None:
        self.db = db
        db.execute("""
            CREATE TABLE IF NOT EXISTS agency_knowledge (
                id TEXT PRIMARY KEY, system TEXT NOT NULL DEFAULT '',
                kind TEXT NOT NULL DEFAULT 'fact',
                text TEXT NOT NULL, source TEXT NOT NULL DEFAULT '',
                entities TEXT NOT NULL DEFAULT '',
                ts REAL NOT NULL, expires_at REAL NOT NULL DEFAULT 0,
                dedupe_key TEXT NOT NULL DEFAULT ''
            )""")
        db.execute("CREATE INDEX IF NOT EXISTS idx_agency_knowledge_key"
                   " ON agency_knowledge(dedupe_key)")

    def deposit(self, system: str, text: str, *, source: str = "",
                kind: str = "fact", entities: str = "",
                ttl_s: float = 0.0, now: float | None = None) -> bool:
        """Store one knowledge item. False = duplicate, not stored twice."""
        now = time.time() if now is None else now
        key = _normalize(text)
        if not key:
            return False
        row = self.db.query_one(
            "SELECT id FROM agency_knowledge WHERE dedupe_key = ?", (key,))
        if row:
            return False
        self.db.execute(
            "INSERT INTO agency_knowledge VALUES (?,?,?,?,?,?,?,?,?)",
            (f"know-{uuid.uuid4().hex[:12]}", system, kind, text, source,
             entities, now, (now + ttl_s) if ttl_s else 0.0, key))
        return True

    def search(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        terms = [t for t in _normalize(query).split() if t]
        if not terms:
            return []
        cond = " AND ".join("LOWER(text) LIKE ?" for _ in terms)
        params = [f"%{t}%" for t in terms]
        return self.db.query(
            f"SELECT * FROM agency_knowledge WHERE {cond}"
            f" ORDER BY ts DESC LIMIT ?", (*params, limit))

    def count(self) -> int:
        row = self.db.query_one("SELECT COUNT(*) AS n FROM agency_knowledge")
        return int(row["n"]) if row else 0

    def prune_expired(self, *, now: float | None = None) -> int:
        now = time.time() if now is None else now
        cur = self.db.execute(
            "DELETE FROM agency_knowledge WHERE expires_at > 0 AND expires_at <= ?",
            (now,))
        return int(getattr(cur, "rowcount", 0) or 0)
