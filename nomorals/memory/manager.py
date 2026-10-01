"""Memory manager: the unified façade.

Implements the four cooperating stores described in ARCHITECTURE.md §6 —
episodic, semantic, working, and the consolidation loop — as classes in this
module rather than four files, because they share the same record type, the same
scoring function, and the same index, and splitting them added import churn
without adding clarity.

Public surface used by agents:

    memory.remember(text, kind="fact", importance=0.8)
    memory.recall("what did we decide about X", limit=8)
    memory.build_context(goal, budget_tokens=4000)
    memory.consolidate()
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..core.errors import classify
from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..core.text import approx_token_count, chunk_text, dedupe_by_simhash, summarize
from ..storage.db import Database
from ..storage.fts import FTSIndex
from ..storage.repository import Repository
from ..storage.vectors import VectorStore
from .base import DEFAULT_WEIGHTS, MemoryKind, MemoryRecord, normalize_scores, score_memory
from .embeddings import Embedder

__all__ = ["MemoryManager"]


def _is_private(record: MemoryRecord) -> bool:
    """Prompt 11: the owner-marked privacy flag lives in metadata."""
    try:
        return bool((record.metadata or {}).get("private"))
    except Exception:  # noqa: BLE001
        return False


def _is_sensitive(content: str, kind: str) -> str:
    """Prompt 11: refuse to persist sensitive attributes as FACT/PREFERENCE/
    RELATIONSHIP.  Lazy import keeps manager ↔ persona dependency one-way."""
    if kind not in (MemoryKind.FACT, MemoryKind.PREFERENCE,
                    MemoryKind.RELATIONSHIP):
        return ""
    try:
        from .persona import is_sensitive_text
        return is_sensitive_text(content)
    except Exception:  # noqa: BLE001
        return ""

_log = get_logger(__name__)


@dataclass
class RecallResult:
    records: list[MemoryRecord]
    query: str
    elapsed_ms: float = 0.0

    def __iter__(self):
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def texts(self) -> list[str]:
        return [r.content for r in self.records]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "count": len(self.records),
            "elapsed_ms": round(self.elapsed_ms, 3),
            "records": [r.to_dict() for r in self.records],
        }


class MemoryManager:
    """Episodic + semantic + working memory with consolidation."""

    def __init__(
        self,
        context: Any,
        *,
        embedder: Embedder | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.context = context
        self.db: Database = context.db
        settings = getattr(context, "settings", None)
        memory_settings = getattr(settings, "memory", None) if settings else None

        self.limit = getattr(memory_settings, "recall_limit", 12) if memory_settings else 12
        self.context_budget = (
            getattr(memory_settings, "context_budget_tokens", 6000) if memory_settings else 6000
        )
        self.half_life = (
            float(getattr(memory_settings, "decay_half_life_hours", 72.0)) * 3600.0
            if memory_settings
            else 259200.0
        )
        self.forget_threshold = (
            float(getattr(memory_settings, "forget_threshold", 0.02)) if memory_settings else 0.02
        )
        self.consolidate_every = (
            int(getattr(memory_settings, "consolidate_on_pressure", 5000)) if memory_settings else 5000
        )
        self.weights = dict(weights or (memory_settings.weights if memory_settings else DEFAULT_WEIGHTS))

        self.repo = Repository(self.db, "memories", json_columns=("metadata",))
        self.vectors = VectorStore(self.db)
        self.fts = FTSIndex(self.db, "memories_fts", columns=["content"])
        self.embedder = embedder or Embedder(
            provider=getattr(getattr(settings, "embedding", None), "provider", "hashing") if settings else "hashing",
            model=getattr(getattr(settings, "embedding", None), "model", "") if settings else "",
            dimensions=getattr(getattr(settings, "embedding", None), "dimensions", 512) if settings else 512,
            router=getattr(context, "router", None),
        )
        self.stats = {"remembered": 0, "recalls": 0, "consolidations": 0, "forgotten": 0}
        self._last_consolidation = 0.0

    # ── write path ───────────────────────────────────────────────────────────
    def remember(
        self,
        content: str,
        *,
        kind: str = MemoryKind.EPISODE,
        importance: float = 0.5,
        source: str = "",
        agent: str = "",
        metadata: dict[str, Any] | None = None,
        ttl_seconds: float = 0.0,
        index: bool = True,
        tags: Any = "",
        origin: str = "",
    ) -> str:
        """Store a memory and index it for both vector and lexical recall.

        Prompt 11: FACT/PREFERENCE/RELATIONSHIP records touching sensitive
        attributes (health, politics, religion, race, sexuality) are refused
        — the owner never gets a shadow profile.  Returns "" on refusal.
        """
        content = (content or "").strip()
        if not content:
            return ""
        if _is_sensitive(content, kind):
            _log.info("remember refused: sensitive attribute (%s)", kind)
            return ""
        now = time.time()
        record_id = new_id()
        row = {
            "id": record_id,
            "kind": kind,
            "content": content,
            "importance": max(0.0, min(1.0, importance)),
            "salience": max(0.0, min(1.0, importance)),
            "decay": 1.0,
            "access_count": 0,
            "last_access": 0.0,
            "source": source,
            "agent": agent,
            "tags": join_tags(tags),
            "origin": (origin or "").strip(),
            "created_at": now,
            "updated_at": now,
            "expires_at": now + ttl_seconds if ttl_seconds else None,
            "metadata": metadata or {},
        }
        with self.db.transaction():
            # Must go through the Repository: it owns the JSON encoding for the
            # metadata column. A raw db.insert() hands sqlite a dict and blows up.
            self.repo.create(row, commit=False)
            if index:
                vector = self.embedder.embed(content)
                embedding_id = self.vectors.put(vector, owner_type="memory", owner_id=record_id)
                self.db.execute(
                    "UPDATE memories SET embedding_id = ? WHERE id = ?", (embedding_id, record_id)
                )
                self._index_text(record_id, content)
        self.stats["remembered"] += 1
        if self.repo.count() >= self.consolidate_every and (now - self._last_consolidation) > 60:
            self.consolidate()
        return record_id

    def remember_many(
        self, items: Iterable[tuple[str, str]], *, source: str = "", **kw: Any
    ) -> list[str]:
        """Bulk write as ``(content, kind)`` pairs, indexed in one pass."""
        materialized = [(c.strip(), k) for c, k in items if c and c.strip()]
        if not materialized:
            return []
        now = time.time()
        ids = [new_id() for _ in materialized]
        rows = [
            {
                "id": ids[i], "kind": kind, "content": content,
                "importance": kw.get("importance", 0.5), "salience": kw.get("importance", 0.5),
                "decay": 1.0, "access_count": 0, "last_access": 0.0,
                "source": source, "agent": kw.get("agent", ""),
                "created_at": now, "updated_at": now, "metadata": kw.get("metadata") or {},
            }
            for i, (content, kind) in enumerate(materialized)
        ]
        vectors = self.embedder.embed_many([c for c, _ in materialized])
        with self.db.transaction():
            self.repo.create_many(rows)
            self.vectors.put_many(
                [(vectors[i], "memory", ids[i]) for i in range(len(ids))]
            )
            self.fts.put_many(
                (self._rowid(ids[i]), [materialized[i][0]]) for i in range(len(ids))
            )
        self.stats["remembered"] += len(ids)
        return ids

    def _index_text(self, record_id: str, content: str) -> None:
        """Mirror the text into the FTS index, keyed by the table rowid."""
        rowid = self._rowid(record_id)
        if rowid:
            self.fts.put(rowid, [content])

    def _rowid(self, record_id: str) -> int:
        return int(self.db.scalar("SELECT rowid FROM memories WHERE id = ?", (record_id,), default=0) or 0)

    def _id_for_rowid(self, rowid: int) -> str:
        return str(self.db.scalar("SELECT id FROM memories WHERE rowid = ?", (rowid,), default="") or "")

    # ── read path ────────────────────────────────────────────────────────────
    def recall(
        self,
        query: str,
        *,
        limit: int | None = None,
        kind: str = "",
        min_score: float = 0.0,
        source: str = "",
        include_expired: bool = False,
        include_private: bool = False,
        tags: str = "",
    ) -> RecallResult:
        """Merged semantic + lexical + recency recall.

        ``tags`` (comma-separated) keeps only records carrying ALL of the
        requested tags — the tag lane the /remember command populates.

        Private-marked records are excluded unless ``include_private`` is
        set — proactive recall and training pipelines never see them.
        """
        wanted_tags = {t.strip() for t in (tags or "").split(",") if t.strip()}
        started = time.perf_counter()
        limit = limit or self.limit
        if not query.strip():
            recents = self._recent(limit, kind=kind)
            if not include_private:
                recents = [r for r in recents if not _is_private(r)]
            return RecallResult(records=recents, query=query)
        self.stats["recalls"] += 1

        candidates: dict[str, MemoryRecord] = {}
        semantic_scores: dict[str, float] = {}
        lexical_scores: dict[str, float] = {}

        # Semantic pass: over-fetch, because the merge re-ranks.
        vector = self.embedder.embed(query)
        for hit in self.vectors.search(vector, limit=limit * 6, owner_type="memory"):
            semantic_scores[hit.owner_id] = hit.score
            candidates[hit.owner_id] = hit.owner_id  # placeholder, resolved below

        # Lexical pass: catches identifiers and rare terms vectors blur.
        for hit in self.fts.search(query, limit=limit * 4):
            record_id = self._id_for_rowid(hit.rowid)
            if record_id:
                lexical_scores[record_id] = max(0.0, min(1.0, (hit.score + 20.0) / 25.0))
                candidates.setdefault(record_id, record_id)

        ids = list(candidates)
        if not ids:
            recents = self._recent(limit, kind=kind)
            if not include_private:
                recents = [r for r in recents if not _is_private(r)]
            return RecallResult(records=recents, query=query,
                                elapsed_ms=(time.perf_counter() - started) * 1000)

        placeholders = ", ".join("?" for _ in ids)
        rows = self.db.query(f"SELECT * FROM memories WHERE id IN ({placeholders})", ids)
        now = time.time()
        scored: list[MemoryRecord] = []
        for row in rows:
            record = MemoryRecord.from_row(row)
            if record.expired and not include_expired:
                continue
            if not include_private and _is_private(record):
                continue
            if kind and record.kind != kind:
                continue
            if source and record.source != source:
                continue
            if wanted_tags:
                have = {t.strip() for t in (record.tags or "").split(",")}
                if not wanted_tags <= have:
                    continue
            record.semantic = semantic_scores.get(record.id, 0.0)
            record.lexical = lexical_scores.get(record.id, 0.0)
            record.score = score_memory(
                record,
                semantic=record.semantic,
                lexical=record.lexical,
                weights=self.weights,
                half_life_seconds=self.half_life,
                now=now,
            )
            if record.score >= min_score:
                scored.append(record)

        scored.sort(key=lambda r: -r.score)
        top = scored[:limit]
        normalize_scores(top)
        if top:
            self._touch([r.id for r in top])
        return RecallResult(records=top, query=query, elapsed_ms=(time.perf_counter() - started) * 1000)

    def _recent(self, limit: int, *, kind: str = "") -> list[MemoryRecord]:
        query = self.repo.query().order_by("created_at DESC").limit(limit)
        if kind:
            query.where("kind = ?", kind)
        sql, params = query.build()
        return [MemoryRecord.from_row(r) for r in self.db.query(sql, params)]

    def _touch(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        placeholders = ", ".join("?" for _ in ids)
        self.db.execute(
            f"UPDATE memories SET access_count = access_count + 1, last_access = ?, "
            f"decay = MIN(1.0, decay + 0.02) WHERE id IN ({placeholders})",
            [time.time(), *ids],
        )

    def find_one(self, query: str, *, kind: str = "") -> MemoryRecord | None:
        """Resolve free text ("the server password") to the best record.

        Recall first (semantic+lexical ranking), then a LIKE fallback; a
        hit must share at least one word with the query, so a miss is a
        clean ``None`` rather than whatever happened to rank top.
        """
        query = (query or "").strip()
        if not query:
            return None
        result = self.recall(query, limit=4, kind=kind)
        import re as _re
        qwords = set(_re.findall(r"[a-z0-9']+", query.lower()))
        for record in result.records:
            if qwords & set(_re.findall(r"[a-z0-9']+", record.content.lower())):
                return record
        if kind:
            row = self.db.query_one(
                "SELECT * FROM memories WHERE kind = ? AND content LIKE ? "
                "ORDER BY updated_at DESC LIMIT 1", (kind, f"%{query}%"))
        else:
            row = self.db.query_one(
                "SELECT * FROM memories WHERE content LIKE ? "
                "ORDER BY updated_at DESC LIMIT 1", (f"%{query}%",))
        if row is None:
            return None
        record = MemoryRecord.from_row(row)
        if qwords and not (qwords & set(record.content.lower().split())):
            return None
        return record

    def get(self, record_id: str) -> MemoryRecord | None:
        row = self.repo.get(record_id)
        return MemoryRecord.from_row(row) if row else None

    def update(self, record_id: str, **changes: Any) -> int:
        allowed = {"content", "importance", "salience", "decay", "kind", "metadata", "expires_at"}
        payload = {k: v for k, v in changes.items() if k in allowed}
        if not payload:
            return 0
        count = self.repo.update(record_id, payload)
        if "content" in payload:
            self.vectors.delete_owner("memory", record_id)
            self.vectors.put(self.embedder.embed(payload["content"]), owner_type="memory", owner_id=record_id)
            self._index_text(record_id, payload["content"])
        return count

    def forget(self, record_id: str) -> int:
        self.vectors.delete_owner("memory", record_id)
        rowid = self._rowid(record_id)
        if rowid:
            self.fts.delete(rowid)
        removed = self.repo.delete(record_id)
        self.stats["forgotten"] += removed
        return removed

    def forget_below(self, threshold: float) -> int:
        """Drop low-value memories: the forgetting half of consolidation."""
        now = time.time()
        rows = self.db.query("SELECT id, importance, decay, created_at, kind FROM memories")
        doomed: list[str] = []
        for row in rows:
            record = MemoryRecord.from_row({**row, "content": ""})
            if record.kind in {MemoryKind.FACT, MemoryKind.PREFERENCE}:
                continue  # never forget stated facts
            if record.recency(self.half_life, now) * record.importance < threshold:
                doomed.append(row["id"])
        for record_id in doomed:
            self.forget(record_id)
        return len(doomed)

    # ── Prompt 11: privacy controls ──────────────────────────────────────
    def mark_private(self, record_id: str) -> int:
        """Mark a record private: excluded from proactive recall, context
        blocks, and training pipelines.  Returns rows updated."""
        record = self.get(record_id)
        if record is None:
            return 0
        metadata = dict(record.metadata or {})
        metadata["private"] = True
        return self.update(record_id, metadata=metadata)

    def mark_public(self, record_id: str) -> int:
        """Clear the private flag."""
        record = self.get(record_id)
        if record is None:
            return 0
        metadata = dict(record.metadata or {})
        metadata.pop("private", None)
        return self.update(record_id, metadata=metadata)

    def for_training(self, *, kind: str = "",
                     limit: int = 5000) -> list[MemoryRecord]:
        """Records safe for model-training pipelines (Prompt 01's
        self-improvement and any future training loops).

        Contract: NEVER yields private-marked records.  Any pipeline that
        ingests memory records must go through this method.
        """
        try:
            result = self.recall("", limit=limit, kind=kind or "",
                                 include_private=False)
        except Exception:  # noqa: BLE001
            return []
        # belt and braces: recall already filters, but the contract is
        # explicit here so a future recall change can't leak
        return [r for r in result.records if not _is_private(r)]

    # ── working memory ───────────────────────────────────────────────────────
    def build_context(
        self,
        goal: str,
        *,
        budget_tokens: int | None = None,
        limit: int | None = None,
        kinds: Sequence[str] = (),
        include_recent: int = 3,
        include_private: bool = False,
    ) -> str:
        """Assemble a token-budgeted context block for a model call.

        Ranking is by relevance, but assembly walks the ranked list and stops when
        the budget is spent — so a huge memory never crowds out the actual task.

        Private-marked records are excluded unless ``include_private`` is set.
        """
        budget = budget_tokens or self.context_budget
        parts: list[str] = []
        used = 0

        recalled = self.recall(goal, limit=limit or self.limit,
                               include_private=include_private)
        pool = [r for r in recalled.records if not kinds or r.kind in kinds]
        if include_recent:
            recent_ids = {r.id for r in pool}
            for record in self._recent(include_recent):
                if record.id not in recent_ids and (
                        include_private or not _is_private(record)):
                    pool.append(record)

        for record in pool:
            line = f"- [{record.kind}] {record.content}"
            cost = approx_token_count(line)
            if used + cost > budget:
                continue
            parts.append(line)
            used += cost

        if not parts:
            return ""
        header = "Relevant memory:"
        if used + approx_token_count(header) > budget:
            return "\n".join(parts)
        return f"{header}\n" + "\n".join(parts)

    # ── consolidation ────────────────────────────────────────────────────────
    def consolidate(self, *, min_episodes: int = 5, forget: bool = True) -> dict[str, Any]:
        """The sleep cycle: dedupe, distill episodes into facts, then forget.

        Runs on a schedule or when memory pressure crosses a threshold. Without
        the forgetting step a long-running system's recall degrades monotonically
        as the index fills with stale near-duplicates.
        """
        started = time.perf_counter()
        self._last_consolidation = time.time()
        self.stats["consolidations"] += 1
        rows = self.db.query(
            "SELECT id, content, importance, created_at FROM memories WHERE kind = ? ORDER BY created_at",
            (MemoryKind.EPISODE,),
        )
        report: dict[str, Any] = {"episodes": len(rows), "merged": 0, "summaries": 0, "forgotten": 0}
        if len(rows) < min_episodes:
            return report

        contents = [r["content"] for r in rows]
        dedup = dedupe_by_simhash(contents, threshold=4)
        report["merged"] = len(dedup.dropped)
        for index in dedup.dropped:
            self.forget(rows[index]["id"])
            report["forgotten"] += 1

        kept = [rows[i] for i in dedup.kept]
        if len(kept) >= min_episodes:
            summary = self._summarize_episodes([r["content"] for r in kept])
            if summary:
                self.remember(
                    summary,
                    kind=MemoryKind.FACT,
                    importance=0.7,
                    source="consolidation",
                    metadata={"distilled_from": len(kept)},
                )
                report["summaries"] = 1

        if forget:
            report["forgotten"] += self.forget_below(self.forget_threshold)
        report["seconds"] = round(time.perf_counter() - started, 3)
        report["remaining"] = self.repo.count()
        _log.debug("consolidation: %s", report)
        return report

    def _summarize_episodes(self, contents: Sequence[str]) -> str:
        router = getattr(self.context, "router", None) if self.context else None
        corpus = "\n".join(f"- {c}" for c in contents)
        if router is not None:
            from ..llm.base import Message, SamplingParams

            response = router.chat(
                [
                    Message.system("Distil these notes into at most 5 durable facts. One per line."),
                    Message.user(corpus[:8000]),
                ],
                SamplingParams(temperature=0.2, max_tokens=512),
            )
            if response.ok and response.text.strip():
                return response.text.strip()
        return summarize(corpus, max_sentences=5)

    # ── retrieval over documents ─────────────────────────────────────────────
    def ingest_document(self, text: str, *, source: str = "", chunk_tokens: int = 512) -> int:
        """Chunk a document into memory so its contents become recallable."""
        chunks = chunk_text(text, max_tokens=chunk_tokens, source=source)
        if not chunks:
            return 0
        return len(
            self.remember_many(
                [(c.text, MemoryKind.EPISODE) for c in chunks],
                source=source or "document",
                importance=0.45,
            )
        )

    # ── introspection ────────────────────────────────────────────────────────
    def counts_by_kind(self) -> dict[str, int]:
        rows = self.db.query("SELECT kind, COUNT(*) AS n FROM memories GROUP BY kind")
        return {r["kind"]: int(r["n"]) for r in rows}

    def tune_weights(self, updates: dict[str, float]) -> dict[str, float]:
        """Adjust recall weights (called by the reflector after a mission)."""
        for key, value in updates.items():
            if key in self.weights:
                self.weights[key] = max(0.0, float(value))
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: round(v / total, 4) for k, v in self.weights.items()}
        return dict(self.weights)

    def stats_snapshot(self) -> dict[str, Any]:
        return {
            **self.stats,
            "records": self.repo.count(),
            "vectors": self.vectors.count(),
            "fts_documents": self.fts.count(),
            "by_kind": self.counts_by_kind(),
            "weights": dict(self.weights),
            "embedder": self.embedder.stats_snapshot(),
        }


def join_tags(tags: Any) -> str:
    """Normalize tags into a comma-joined canonical string.

    Accepts a list or an already-joined string; lowercases, trims, drops
    empties and duplicates, preserves first-seen order, caps at 8 tags so
    one sloppy input can't bloat the row.
    """
    if not tags:
        return ""
    if isinstance(tags, str):
        parts: list[str] = tags.split(",")
    else:
        try:
            parts = [str(t) for t in tags]
        except TypeError:
            return ""
    seen: list[str] = []
    for part in parts:
        tag = part.strip().lower()
        if tag and tag not in seen:
            seen.append(tag)
        if len(seen) >= 8:
            break
    return ",".join(seen)
