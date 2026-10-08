"""Two-tier memory architecture (Khoj pattern) — build-map #36.

ADD-ON-TOP, NEVER REPLACE: the existing ``MemoryManager`` (remember/recall)
is completely untouched by this module. This is a parallel, opt-in memory
layer the agent can consult *additionally*. The user's trusted recall stays
exactly as it is.

Tier 1 — EventStore: conversation events split into ~256-token recursive
chunks (never mid-sentence), embedded and vector-searched for temporal /
contextual recall ("what did we discuss last Tuesday?").

Tier 2 — FactStore: atomic first-person facts maintained by a dedicated
extraction agent (the "Muninn" pattern: an LLM returns
``MemoryUpdates(create, supersede)`` per exchange). Facts rank first for
direct questions ("what's my girlfriend's name?"). Facts version through
auditable supersede chains — never silently overwritten.

This is semantic fact distillation, NOT the entity/keyword extraction in
``nomorals/memory/extract.py`` — different job, different module.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..storage.db import Database
from ..storage.vectors import VectorStore
from .embeddings import Embedder
from .vector_backends import VectorBackend, select_vector_backend

_log = get_logger(__name__)

__all__ = [
    "EventStore",
    "FactStore",
    "Fact",
    "MemoryUpdates",
    "TwoTierMemory",
    "TwoTierRecall",
    "distill_facts",
    "two_tier_db_path",
]

# ~256 tokens ≈ ~1000 characters. Chunks never split mid-sentence.
_CHUNK_MAX_CHARS = 1000

_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+(?=[A-Z0-9\"'“(\[])")
_CLAUSE_RE = re.compile(r"(?<=[,;:])\s+")

# Direct questions — facts rank first for these.
_DIRECT_Q_RE = re.compile(
    r"\b(what'?s|what is|who'?s|who is|when'?s|when is|where'?s|where is|"
    r"how old is|do i|am i|is my|are my|my (name|birthday|age|favorite|"
    r"favourite|wife|husband|girlfriend|boyfriend|job|address|phone))\b",
    re.IGNORECASE,
)

# Temporal/contextual questions — events carry these.
_TEMPORAL_RE = re.compile(
    r"\b(last|yesterday|tuesday|wednesday|monday|discuss|talked about|"
    r"conversation|we (said|discussed|talked))\b",
    re.IGNORECASE,
)


def two_tier_db_path(settings: Any = None) -> Path:
    """Home-dir path for the two-tier store, honoring runtime settings."""
    if settings is not None:
        home = getattr(settings, "home_path", None)
        if home:
            return Path(home) / "memory" / "two_tier.db"
    return Path.home() / ".nomorals" / "memory" / "two_tier.db"


def _init_vector_table(db: Database) -> None:
    """The embeddings table the vector backends need. Self-contained so the
    two-tier DB never depends on the main DB's migrations."""
    db.execute(
        """CREATE TABLE IF NOT EXISTS embeddings (
               id         TEXT PRIMARY KEY,
               owner_type TEXT NOT NULL,
               owner_id   TEXT NOT NULL,
               model      TEXT NOT NULL,
               dim        INTEGER NOT NULL,
               norm       REAL NOT NULL DEFAULT 0,
               vector     BLOB NOT NULL,
               created_at REAL NOT NULL
           )""")
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_embeddings_owner "
        "ON embeddings(owner_type, owner_id)")


def _split_sentences(text: str) -> list[str]:
    """Split into sentences; fall back to clause and hard splits."""
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENTENCE_RE.split(text) if p.strip()]
    out: list[str] = []
    for part in parts:
        if len(part) <= _CHUNK_MAX_CHARS:
            out.append(part)
            continue
        # Long sentence: try clause boundaries, then hard split.
        clauses = [c.strip() for c in _CLAUSE_RE.split(part) if c.strip()]
        if len(clauses) > 1 and all(len(c) <= _CHUNK_MAX_CHARS for c in clauses):
            out.extend(clauses)
        else:
            for i in range(0, len(part), _CHUNK_MAX_CHARS):
                piece = part[i:i + _CHUNK_MAX_CHARS].strip()
                if piece:
                    out.append(piece)
    return out


def chunk_text(text: str, max_chars: int = _CHUNK_MAX_CHARS) -> list[str]:
    """Pack sentences into chunks of at most ``max_chars``. Never splits
    mid-sentence (mid-clause only when a single sentence overflows)."""
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sent in sentences:
        extra = len(sent) + (1 if current else 0)
        if current and current_len + extra > max_chars:
            chunks.append(" ".join(current))
            current, current_len = [sent], len(sent)
        else:
            current.append(sent)
            current_len += extra
    if current:
        chunks.append(" ".join(current))
    return chunks


# ── Tier 1: events ───────────────────────────────────────────────────────────


@dataclass
class EventHit:
    chunk_id: str
    text: str
    ts: float
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"chunk_id": self.chunk_id, "text": self.text,
                "ts": self.ts, "score": round(self.score, 4)}


class EventStore:
    """Tier 1: chunked conversation events with vector search."""

    def __init__(self, db: Database, *, embedder: Embedder | None = None,
                 vector_preference: str | None = None) -> None:
        self.db = db
        self.embedder = embedder or Embedder(provider="hashing")
        self._init_schema()
        # VectorStore first: it owns the embeddings table. The backend
        # wraps the shared instance (same pattern as MemoryManager).
        self._vectors = VectorStore(db)
        self.vectors: VectorBackend = select_vector_backend(
            db, owner_type="tier_event", preference=vector_preference,
            vectors=self._vectors)

    def _init_schema(self) -> None:
        _init_vector_table(self.db)
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS tier_events (
                   id TEXT PRIMARY KEY,
                   chunk TEXT NOT NULL,
                   ts REAL NOT NULL,
                   tags TEXT NOT NULL DEFAULT ''
               )""")
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_tier_events_ts ON tier_events(ts)")

    def record_event(self, text: str, *, ts: float | None = None,
                     tags: str = "") -> list[str]:
        """Chunk + embed + store one exchange. Returns chunk ids."""
        text = (text or "").strip()
        if not text:
            return []
        now = ts if ts is not None else time.time()
        chunks = chunk_text(text)
        if not chunks:
            return []
        vectors = self.embedder.embed_many(chunks)
        ids = [new_id() for _ in chunks]
        with self.db.transaction():
            for cid, chunk, vec in zip(ids, chunks, vectors):
                self.db.execute(
                    "INSERT INTO tier_events (id, chunk, ts, tags) VALUES (?,?,?,?)",
                    (cid, chunk, now, tags or ""))
                self.vectors.put(vec, cid)
        return ids

    def search_events(self, query: str, *, since: float | None = None,
                      limit: int = 5) -> list[EventHit]:
        """Vector search over event chunks, newest-aware."""
        query = (query or "").strip()
        if not query:
            return []
        vector = self.embedder.embed(query)
        hits = self.vectors.search(vector, limit=limit * 4)
        if not hits:
            return []
        ids = [h.owner_id for h in hits]
        placeholders = ",".join("?" for _ in ids)
        rows = self.db.query(
            f"SELECT id, chunk, ts FROM tier_events WHERE id IN ({placeholders})",
            ids)
        by_id = {r["id"]: r for r in rows}
        out: list[EventHit] = []
        for h in hits:
            row = by_id.get(h.owner_id)
            if row is None:
                continue
            if since is not None and float(row["ts"]) < since:
                continue
            out.append(EventHit(chunk_id=h.owner_id, text=str(row["chunk"]),
                                ts=float(row["ts"]), score=h.score))
            if len(out) >= limit:
                break
        return out

    def count(self) -> int:
        return int(self.db.scalar("SELECT COUNT(*) FROM tier_events",
                                  default=0) or 0)


# ── Tier 2: facts ────────────────────────────────────────────────────────────


@dataclass
class Fact:
    """One atomic first-person fact. Versions via ``supersedes`` chain."""
    id: str
    text: str
    confidence: float = 0.7
    source_ts: float = field(default_factory=time.time)
    supersedes: str | None = None
    active: bool = True
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text,
                "confidence": round(self.confidence, 3),
                "source_ts": self.source_ts, "supersedes": self.supersedes,
                "active": self.active, "created_at": self.created_at}


@dataclass
class FactHit:
    fact: Fact
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = self.fact.to_dict()
        d["score"] = round(self.score, 4)
        return d


class FactStore:
    """Tier 2: atomic facts with auditable supersede chains."""

    def __init__(self, db: Database, *, embedder: Embedder | None = None,
                 vector_preference: str | None = None) -> None:
        self.db = db
        self.embedder = embedder or Embedder(provider="hashing")
        self._init_schema()
        self._vectors = VectorStore(db)
        self.vectors: VectorBackend = select_vector_backend(
            db, owner_type="tier_fact", preference=vector_preference,
            vectors=self._vectors)

    def _init_schema(self) -> None:
        _init_vector_table(self.db)
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS tier_facts (
                   id TEXT PRIMARY KEY,
                   text TEXT NOT NULL,
                   confidence REAL NOT NULL DEFAULT 0.7,
                   source_ts REAL NOT NULL,
                   supersedes TEXT,
                   active INTEGER NOT NULL DEFAULT 1,
                   created_at REAL NOT NULL
               )""")
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_tier_facts_active ON tier_facts(active)")

    def add_fact(self, text: str, *, confidence: float = 0.7,
                 source_ts: float | None = None) -> Fact:
        text = (text or "").strip()
        if not text:
            raise ValueError("fact text must not be empty")
        now = time.time()
        fact = Fact(id=new_id(), text=text,
                    confidence=max(0.0, min(1.0, confidence)),
                    source_ts=source_ts if source_ts is not None else now,
                    created_at=now)
        with self.db.transaction():
            self.db.execute(
                """INSERT INTO tier_facts
                   (id, text, confidence, source_ts, supersedes, active, created_at)
                   VALUES (?,?,?,?,?,1,?)""",
                (fact.id, fact.text, fact.confidence, fact.source_ts,
                 None, fact.created_at))
            self.vectors.put(self.embedder.embed(text), fact.id)
        return fact

    def get_fact(self, fact_id: str) -> Fact | None:
        row = self.db.query_one(
            "SELECT * FROM tier_facts WHERE id = ?", (fact_id,))
        return self._row_to_fact(row) if row else None

    @staticmethod
    def _row_to_fact(row: dict[str, Any]) -> Fact:
        return Fact(
            id=str(row["id"]), text=str(row["text"]),
            confidence=float(row["confidence"]),
            source_ts=float(row["source_ts"]),
            supersedes=str(row["supersedes"]) if row["supersedes"] else None,
            active=bool(row["active"]), created_at=float(row["created_at"]))

    def supersede_fact(self, old_id: str, new_text: str, *,
                       confidence: float = 0.7) -> Fact:
        """Version a fact: old goes inactive, new links back. The chain is
        auditable via :meth:`history`. Never silently overwrites."""
        old = self.get_fact(old_id)
        if old is None:
            raise ValueError(f"unknown fact {old_id!r}")
        if not old.active:
            raise ValueError(
                f"fact {old_id!r} is already superseded — refusing to fork "
                "the chain; supersede the current head instead")
        new = self.add_fact(new_text, confidence=confidence)
        with self.db.transaction():
            self.db.execute(
                "UPDATE tier_facts SET active = 0 WHERE id = ?", (old_id,))
            self.db.execute(
                "UPDATE tier_facts SET supersedes = ? WHERE id = ?",
                (old_id, new.id))
        new.supersedes = old_id
        old.active = False
        return new

    def history(self, fact_id: str) -> list[Fact]:
        """The full version chain for one fact, oldest first."""
        chain: list[Fact] = []
        seen: set[str] = set()
        current = self.get_fact(fact_id)
        while current is not None and current.id not in seen:
            seen.add(current.id)
            chain.append(current)
            current = (self.get_fact(current.supersedes)
                       if current.supersedes else None)
        chain.reverse()
        return chain

    def search_facts(self, query: str, *, limit: int = 5,
                     active_only: bool = True) -> list[FactHit]:
        query = (query or "").strip()
        if not query:
            return []
        vector = self.embedder.embed(query)
        hits = self.vectors.search(vector, limit=limit * 4)
        if not hits:
            return []
        ids = [h.owner_id for h in hits]
        placeholders = ",".join("?" for _ in ids)
        rows = self.db.query(
            f"SELECT * FROM tier_facts WHERE id IN ({placeholders})", ids)
        by_id = {str(r["id"]): r for r in rows}
        out: list[FactHit] = []
        for h in hits:
            row = by_id.get(h.owner_id)
            if row is None:
                continue
            fact = self._row_to_fact(row)
            if active_only and not fact.active:
                continue
            out.append(FactHit(fact=fact, score=h.score))
            if len(out) >= limit:
                break
        # Confidence-weighted re-rank: a confident fact beats a vague near-match.
        out.sort(key=lambda fh: (fh.score * (0.5 + 0.5 * fh.fact.confidence)),
                 reverse=True)
        return out

    def active_count(self) -> int:
        return int(self.db.scalar(
            "SELECT COUNT(*) FROM tier_facts WHERE active = 1", default=0) or 0)


# ── Muninn: the extraction agent contract ────────────────────────────────────

MUNINN_SYSTEM = """You are Muninn, a memory-extraction service for a personal AI.\
 You read ONE conversation exchange (user message + assistant reply) and\
 distill durable first-person facts about the user.

RULES:
1. Output ONLY a JSON object: {"create": [{"text": "<fact>", "confidence": 0.0-1.0}],\
 "supersede": [{"old_id": "<id>", "text": "<replacement fact>", "confidence": 0.0-1.0}]}
2. Facts are ATOMIC and FIRST-PERSON: "my girlfriend's name is Ada",\
 "I prefer morning briefings", "my dog died in 2024". One claim per fact.
3. Only DURABLE facts — things that will still be true next month.\
 Skip greetings, chit-chat, one-off tasks, and anything said tentatively.
4. No duplicates: if EXISTING FACTS already cover it, create nothing.
5. SUPERSEDE when the exchange contradicts an existing fact: the old fact\
 stays in history, the new one becomes current. Use the exact old_id.
6. Confidence: 0.9+ for explicit statements ("my name is X"), 0.6-0.8 for\
 strong implications, below 0.6 don't create the fact at all.
7. Empty result is fine: {"create": [], "supersede": []} when nothing durable\
 was said. Never invent facts to fill the lists."""

MUNINN_USER_TEMPLATE = """EXISTING FACTS (id: text):
{facts}

EXCHANGE:
{exchange}

Distill durable first-person facts. JSON only."""


@dataclass
class MemoryUpdates:
    """What the extraction agent wants changed this turn."""
    create: list[dict[str, Any]] = field(default_factory=list)
    supersede: list[dict[str, Any]] = field(default_factory=list)

    def empty(self) -> bool:
        return not self.create and not self.supersede

    def to_dict(self) -> dict[str, Any]:
        return {"create": self.create, "supersede": self.supersede}


def distill_facts(exchange_text: str, *,
                  llm_fn: Callable[[str], str] | None = None,
                  existing_facts: Sequence[Fact] | None = None) -> MemoryUpdates:
    """Run the Muninn extraction pass over one exchange.

    ``llm_fn`` takes the full prompt and returns the model's raw text.
    ``None`` → empty updates (never blocks, never guesses). Parse failures
    → empty updates, logged. Never raises.
    """
    if llm_fn is None:
        return MemoryUpdates()
    exchange = (exchange_text or "").strip()
    if not exchange:
        return MemoryUpdates()
    facts_block = "\n".join(
        f"- {f.id}: {f.text}" for f in (existing_facts or [])) or "(none)"
    prompt = (MUNINN_SYSTEM + "\n\n"
              + MUNINN_USER_TEMPLATE.format(facts=facts_block,
                                            exchange=exchange[:6000]))
    try:
        raw = llm_fn(prompt)
    except Exception as exc:  # noqa: BLE001 — extraction never blocks chat
        _log.warning("muninn llm_fn failed: %s", exc)
        return MemoryUpdates()
    return _parse_updates(raw)


def _parse_updates(raw: str) -> MemoryUpdates:
    """Defensive parse of the model's JSON. Garbage → empty updates."""
    try:
        text = (raw or "").strip()
        # Tolerate code fences.
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence:
            text = fence.group(1)
        else:
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end <= start:
                raise ValueError("no JSON object found")
            text = text[start:end + 1]
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("top-level JSON is not an object")
    except Exception as exc:  # noqa: BLE001
        _log.warning("muninn parse failed: %s", exc)
        return MemoryUpdates()
    updates = MemoryUpdates()
    for item in (data.get("create") or []):
        if not isinstance(item, dict):
            continue
        fact_text = str(item.get("text") or "").strip()
        if not fact_text:
            continue
        try:
            conf = float(item.get("confidence", 0.7))
        except (TypeError, ValueError):
            conf = 0.7
        if conf < 0.6:  # below the creation bar
            continue
        updates.create.append({"text": fact_text,
                               "confidence": max(0.0, min(1.0, conf))})
    for item in (data.get("supersede") or []):
        if not isinstance(item, dict):
            continue
        old_id = str(item.get("old_id") or "").strip()
        fact_text = str(item.get("text") or "").strip()
        if not old_id or not fact_text:
            continue
        try:
            conf = float(item.get("confidence", 0.7))
        except (TypeError, ValueError):
            conf = 0.7
        updates.supersede.append({"old_id": old_id, "text": fact_text,
                                  "confidence": max(0.0, min(1.0, conf))})
    return updates


# ── facade ───────────────────────────────────────────────────────────────────


@dataclass
class TwoTierRecall:
    query: str
    facts: list[FactHit] = field(default_factory=list)
    events: list[EventHit] = field(default_factory=list)
    facts_first: bool = True
    elapsed_ms: float = 0.0

    @property
    def texts(self) -> list[str]:
        """Merged answer order: facts first for direct questions."""
        ordered = (self.facts + self.events) if self.facts_first \
            else (self.events + self.facts)
        return [h.fact.text if isinstance(h, FactHit) else h.text
                for h in ordered]

    def to_dict(self) -> dict[str, Any]:
        return {"query": self.query, "facts_first": self.facts_first,
                "elapsed_ms": round(self.elapsed_ms, 3),
                "facts": [h.to_dict() for h in self.facts],
                "events": [h.to_dict() for h in self.events]}


class TwoTierMemory:
    """The two-tier facade. Parallel to MemoryManager — never a replacement."""

    def __init__(self, db: Database | str | Path | None = None, *,
                 embedder: Embedder | None = None,
                 llm_fn: Callable[[str], str] | None = None,
                 settings: Any = None) -> None:
        if db is None:
            path = two_tier_db_path(settings)
            path.parent.mkdir(parents=True, exist_ok=True)
            db = Database(path)
        elif isinstance(db, (str, Path)):
            db = Database(db)
        self.db = db
        self.embedder = embedder or Embedder(provider="hashing")
        self.llm_fn = llm_fn
        self.events = EventStore(db, embedder=self.embedder)
        self.facts = FactStore(db, embedder=self.embedder)
        self._lock = threading.RLock()
        self.stats = {"observed": 0, "distilled": 0, "distill_errors": 0}

    # ── write path ───────────────────────────────────────────────────────
    def observe(self, exchange_text: str, *, tags: str = "",
                distill: bool = True) -> list[str]:
        """Record the exchange as events (sync, fast) and queue fact
        distillation on a daemon thread (never blocks chat). Never raises."""
        try:
            chunk_ids = self.events.record_event(exchange_text, tags=tags)
        except Exception as exc:  # noqa: BLE001
            _log.warning("two-tier observe failed: %s", exc)
            return []
        self.stats["observed"] += 1
        if distill and self.llm_fn is not None and (exchange_text or "").strip():
            threading.Thread(
                target=self._distill_async,
                args=((exchange_text or "").strip(),),
                name="two-tier-distill", daemon=True,
            ).start()
        return chunk_ids

    def _distill_async(self, exchange_text: str) -> None:
        try:
            with self._lock:
                existing = [h.fact for h in
                            self.facts.search_facts(exchange_text, limit=10)]
            updates = distill_facts(exchange_text, llm_fn=self.llm_fn,
                                    existing_facts=existing)
            if not updates.empty():
                self.apply_updates(updates)
                self.stats["distilled"] += 1
        except Exception as exc:  # noqa: BLE001
            self.stats["distill_errors"] += 1
            _log.warning("two-tier distill failed: %s", exc)

    def apply_updates(self, updates: MemoryUpdates) -> dict[str, int]:
        """Apply a MemoryUpdates batch to the FactStore. Never raises."""
        done = {"created": 0, "superseded": 0, "skipped": 0}
        with self._lock:
            for item in updates.create:
                try:
                    self.facts.add_fact(item["text"],
                                        confidence=float(item.get("confidence", 0.7)))
                    done["created"] += 1
                except Exception as exc:  # noqa: BLE001
                    _log.warning("two-tier create failed: %s", exc)
                    done["skipped"] += 1
            for item in updates.supersede:
                try:
                    self.facts.supersede_fact(
                        item["old_id"], item["text"],
                        confidence=float(item.get("confidence", 0.7)))
                    done["superseded"] += 1
                except Exception as exc:  # noqa: BLE001
                    _log.warning("two-tier supersede failed: %s", exc)
                    done["skipped"] += 1
        return done

    # ── read path ────────────────────────────────────────────────────────
    def recall(self, query: str, *, limit: int = 8) -> TwoTierRecall:
        """Query both tiers. Facts rank first for direct questions
        ("what's my girlfriend's name?"); events lead for temporal context
        ("what did we discuss last Tuesday?")."""
        started = time.perf_counter()
        query = (query or "").strip()
        facts_first = True
        if query and _TEMPORAL_RE.search(query) and not _DIRECT_Q_RE.search(query):
            facts_first = False
        fact_hits = self.facts.search_facts(query, limit=limit)
        event_hits = self.events.search_events(query, limit=limit)
        return TwoTierRecall(
            query=query, facts=fact_hits, events=event_hits,
            facts_first=facts_first,
            elapsed_ms=(time.perf_counter() - started) * 1000.0)

    def close(self) -> None:
        try:
            self.db.close()
        except Exception:  # noqa: BLE001
            pass
