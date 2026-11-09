"""Always-growing knowledge base (owner refinement).

The knowledge accumulation itself is the point: every system deposits
what it reads, explores, and learns — claims, findings, lessons —
into ONE durable store that is always handy for every other system.
Deduped by normalized content so ten sightings of one fact are one
fact, not ten — but the sightings are kept as corroboration
(``sightings`` count, ``last_seen``, every ``sources`` witness), so a
fact seen by five systems carries more weight than one seen by one.
Entries carry source/system/provenance + freshness so a disappeared
URL is tracked as retrieval availability, not truth. Expired entries
are never served by :meth:`KnowledgeStore.search`; :meth:`KnowledgeStore.prune_expired`
reclaims them.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

__all__ = ["KnowledgeStore"]


def _normalize(text: str) -> str:
    """Unicode-aware normalization for dedupe keys and search terms.

    Letters/digits from EVERY script are word characters (Arabic,
    Chinese, Devanagari, Cyrillic, accented Latin, …): a fact's
    identity must never depend on the script it was written in (F11 —
    the old ``[^a-z0-9]`` pattern deleted non-Latin content outright,
    so entire deposits normalized to "" and were silently refused).
    Everything else — punctuation, symbols, emoji, whitespace — is a
    word separator. The ORIGINAL text is always what gets stored;
    this only derives keys/terms from it.
    """
    lowered = (text or "").lower()
    chars = [ch if ch.isalnum() else " " for ch in lowered]
    return " ".join("".join(chars).split())


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
                dedupe_key TEXT NOT NULL DEFAULT '',
                sightings INTEGER NOT NULL DEFAULT 1,
                last_seen REAL NOT NULL DEFAULT 0,
                sources TEXT NOT NULL DEFAULT ''
            )""")
        db.execute("CREATE INDEX IF NOT EXISTS idx_agency_knowledge_key"
                   " ON agency_knowledge(dedupe_key)")
        # Belt and braces for DBs created before the corroboration
        # columns existed. Never raises.
        try:
            cols = {c["name"] for c in db.table_info("agency_knowledge")}
            for name, ddl in (("sightings", "INTEGER NOT NULL DEFAULT 1"),
                              ("last_seen", "REAL NOT NULL DEFAULT 0"),
                              ("sources", "TEXT NOT NULL DEFAULT ''")):
                if name not in cols:
                    db.execute(f"ALTER TABLE agency_knowledge ADD COLUMN {name} {ddl}")
        except Exception:  # noqa: BLE001
            pass

    def deposit(self, system: str, text: str, *, source: str = "",
                kind: str = "fact", entities: str = "",
                ttl_s: float = 0.0, now: float | None = None) -> bool:
        """Store one knowledge item. False = duplicate, not stored twice.

        A duplicate sighting is NOT lost: the corroboration is recorded
        (sightings count, fresh last_seen, the new source appended) so ten
        sightings of one fact stay one fact *with ten witnesses*.
        """
        now = time.time() if now is None else now
        key = _normalize(text)
        if not key:
            return False
        row = self.db.query_one(
            "SELECT id, sightings, sources, ts, expires_at"
            " FROM agency_knowledge WHERE dedupe_key = ?", (key,))
        if row:
            # Corroboration, not a new fact: record the witness.
            seen = [s for s in str(row.get("sources") or "").split(",") if s]
            if source and source not in seen:
                seen.append(source)
            # Freshness renewal (F12): a re-observation is a FRESH
            # sighting, so it renews the entry's freshness — with the
            # TTL supplied alongside this sighting when given, else by
            # re-applying the entry's own original lifetime from now.
            # Before this, a duplicate sighting updated last_seen but
            # left an already-passed expires_at untouched, so a fact
            # that was being seen again stayed invisible forever.
            expires = float(row.get("expires_at") or 0)
            if ttl_s:
                expires = now + float(ttl_s)
            elif expires and expires <= now:
                lifetime = expires - float(row.get("ts") or now)
                expires = now + lifetime if lifetime > 0 else 0.0
            self.db.execute(
                "UPDATE agency_knowledge SET sightings = COALESCE(sightings, 0) + 1,"
                " last_seen = ?, sources = ?, expires_at = ? WHERE id = ?",
                (now, ",".join(seen), expires, row["id"]))
            return False
        self.db.execute(
            "INSERT INTO agency_knowledge (id, system, kind, text, source,"
            " entities, ts, expires_at, dedupe_key, sightings, last_seen, sources)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"know-{uuid.uuid4().hex[:12]}", system, kind, text, source,
             entities, now, (now + ttl_s) if ttl_s else 0.0, key,
             1, now, source))
        return True

    def search(self, query: str, *, limit: int = 20,
               now: float | None = None) -> list[dict[str, Any]]:
        """Find knowledge matching all query terms, newest first.

        Expired entries (expires_at passed) are never served as fresh —
        use prune_expired() to reclaim the space.
        """
        terms = [t for t in _normalize(query).split() if t]
        if not terms:
            return []
        now = time.time() if now is None else now
        # Matching is done on the Python-normalized text, NOT SQL
        # LIKE: SQLite's LOWER()/LIKE fold ASCII only, so a Cyrillic
        # (or otherwise non-ASCII-cased) term could never match its
        # own stored text (F11). SQL pre-filters expiry only; the
        # term check is the same all-terms-substring semantics the
        # LIKE form had, applied script-correctly.
        rows = self.db.query(
            "SELECT * FROM agency_knowledge"
            " WHERE (expires_at <= 0 OR expires_at > ?)"
            " ORDER BY ts DESC", (now,))
        out: list[dict[str, Any]] = []
        for row in rows:
            hay = _normalize(str(row.get("text") or ""))
            if all(t in hay for t in terms):
                out.append(row)
                if len(out) >= limit:
                    break
        return out

    def count(self) -> int:
        row = self.db.query_one("SELECT COUNT(*) AS n FROM agency_knowledge")
        return int(row["n"]) if row else 0

    def prune_expired(self, *, now: float | None = None) -> int:
        now = time.time() if now is None else now
        cur = self.db.execute(
            "DELETE FROM agency_knowledge WHERE expires_at > 0 AND expires_at <= ?",
            (now,))
        return int(getattr(cur, "rowcount", 0) or 0)
