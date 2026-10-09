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

from ..llm.brain import brain_for
from ..core.errors import classify
from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..core.text import approx_token_count, chunk_text, dedupe_by_simhash, summarize
from ..storage.db import Database
from ..storage.fts import FTSIndex
from ..storage.repository import Repository
from ..storage.vectors import VectorStore
from .base import (
    DEFAULT_WEIGHTS,
    TRUSTED,
    UNTRUSTED,
    MemoryKind,
    MemoryRecord,
    infer_trust,
    normalize_scores,
)
from .cadence import status as _cadence_status
from .embeddings import Embedder
from .scopes import normalize_scope, record_matches_scope, scope_tag
from .vector_backends import VectorBackend, select_vector_backend

__all__ = ["MemoryManager"]


def _is_private(record: MemoryRecord) -> bool:
    """Prompt 11: the owner-marked privacy flag lives in metadata."""
    try:
        return bool((record.metadata or {}).get("private"))
    except Exception:  # noqa: BLE001
        return False


_log = get_logger(__name__)


def _explain_score(
    record: MemoryRecord,
    *,
    semantic: float = 0.0,
    lexical: float = 0.0,
    weights: dict[str, float] | None = None,
    half_life_seconds: float = 259200.0,
    now: float | None = None,
) -> tuple[float, dict[str, float]]:
    """``score_memory`` with its work shown.

    Returns ``(score, contributions)`` where the contributions are the four
    weighted signal terms — they sum to ``score`` (before recall-time
    adjustments like the session boost). The scoring math is identical to
    :func:`nomorals.memory.base.score_memory`; this exists so
    ``recall(..., explain=True)`` can trace the path without duplicating
    (and drifting from) the formula.
    """
    w = weights or DEFAULT_WEIGHTS
    total = sum(w.values()) or 1.0
    recency = record.recency(half_life_seconds, now)
    importance = max(0.0, min(1.0, record.importance + record.reinforcement()))
    semantic = max(0.0, min(1.0, max(-1.0, min(1.0, semantic))))
    lexical = max(0.0, min(1.0, lexical))
    contributions = {
        "recency": w.get("recency", 0.0) * recency / total,
        "importance": w.get("importance", 0.0) * importance / total,
        "semantic": w.get("semantic", 0.0) * semantic / total,
        "lexical": w.get("lexical", 0.0) * lexical / total,
    }
    score = max(0.0, min(1.0, sum(contributions.values())))
    # Full precision: the contributions must sum to the score exactly, so
    # rounding happens at display time, not here.
    return score, contributions


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
        vector_backend: str | None = None,
    ) -> None:
        """``vector_backend`` selects the semantic-recall vector store:

        ``"auto"`` (default) takes the first available backend in quality
        order — sqlite-vec → usearch → legacy — so a box with ``sqlite-vec``
        installed gets exact in-database KNN and every other box keeps the
        zero-dependency legacy store. A named backend that is not installed
        fails fast with the ``pip install`` command. ``"legacy"`` pins the
        historical behaviour exactly.
        """
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
        # Semantic-recall substrate. The legacy store instance is shared with
        # the backend when "legacy" is selected, so introspection stays coherent.
        self.semantic: VectorBackend = select_vector_backend(
            self.db,
            preference=vector_backend or "auto",
            owner_type="memory",
            vectors=self.vectors,
        )
        self.embedder = embedder or Embedder(
            provider=getattr(getattr(settings, "embedding", None), "provider", "auto") if settings else "auto",
            model=getattr(getattr(settings, "embedding", None), "model", "") if settings else "",
            dimensions=getattr(getattr(settings, "embedding", None), "dimensions", 512) if settings else 512,
            router=getattr(context, "router", None),
        )
        self.stats = {"remembered": 0, "recalls": 0, "consolidations": 0, "forgotten": 0}
        self._last_consolidation = 0.0
        #: Last consolidation report (from consolidate() or
        #: consolidate_additive()); surfaced by health().
        self._last_consolidation_report: dict[str, Any] = {}
        #: Degraded-recall events: (timestamp, lane, reason). Bounded so a
        #: long-running process can't grow it without limit.
        self._health_events: list[dict[str, Any]] = []
        # Trust provenance columns (migration 82) — belt and braces for DBs
        # that were created without running migrations. Never raises.
        self._ensure_trust_columns()

    def _note_health_event(self, lane: str, reason: str) -> None:
        """Record a degraded-path event for health(). Never raises."""
        try:
            self._health_events.append(
                {"at": time.time(), "lane": lane, "reason": str(reason)[:300]}
            )
            del self._health_events[:-50]
        except Exception:  # noqa: BLE001
            pass

    def _ensure_trust_columns(self) -> None:
        """Make sure ``memories`` has the trust/session_id columns.

        Migration 82 covers migrated DBs; this covers hand-built ones
        (tests, older snapshots). Never raises.
        """
        try:
            columns = {c["name"] for c in self.db.table_info("memories")}
            for name in ("trust", "session_id"):
                if name not in columns:
                    self.db.execute(
                        f"ALTER TABLE memories ADD COLUMN {name} TEXT NOT NULL DEFAULT ''"
                    )
        except Exception:  # noqa: BLE001
            pass

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
        trust: str = "",
        session_id: str = "",
        scope: str = "",
    ) -> str:
        """Store a memory and index it for both vector and lexical recall.

        The owner decides what is remembered — no content-based refusals.
        Use ``mark_private`` / the ``private`` metadata flag when a record
        should stay out of model context.

        ``scope`` (e.g. ``"project:devon-arena"``) puts the record in a
        named memory space: ``recall(scope=...)`` sees that space plus
        global records, and never records scoped to another space.  Empty
        (default) = global, visible everywhere.

        Trust provenance (mem-false-fact hardening): ``trust`` is
        ``"trusted"`` or ``"untrusted"``; when empty it is derived from
        ``source``/``origin`` — tool output and other external sources
        default to untrusted, direct user input defaults to trusted.
        Untrusted records are downranked at recall time and untrusted
        preferences are flagged, never applied silently.
        """
        content = (content or "").strip()
        if not content:
            return ""
        now = time.time()
        record_id = new_id()
        origin = (origin or "").strip()
        tag_str = join_tags(tags)
        scope_name = normalize_scope(scope)
        if scope_name:
            # the scope tag leads so the 8-tag cap never drops the space
            # marker — a scoped record that loses its scope tag leaks.
            tag_str = (join_tags([scope_tag(scope_name), tag_str])
                       if tag_str else scope_tag(scope_name))
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
            "tags": tag_str,
            "origin": origin,
            "trust": infer_trust(source, origin, explicit=trust),
            "session_id": (session_id or origin).strip(),
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
                embedding_id = self.semantic.put(vector, record_id)
                self.db.execute(
                    "UPDATE memories SET embedding_id = ? WHERE id = ?", (embedding_id, record_id)
                )
                self._index_text(record_id, content)
        self.stats["remembered"] += 1
        if self.repo.count() >= self.consolidate_every and (now - self._last_consolidation) > 60:
            self.consolidate()
        return record_id

    def remember_many(
        self, items: Iterable[tuple[str, str]], *, source: str = "",
        scope: str = "", **kw: Any
    ) -> list[str]:
        """Bulk write as ``(content, kind)`` pairs, indexed in one pass.

        ``scope`` puts every record in the batch into one named memory
        space (see :meth:`remember`).
        """
        materialized = [(c.strip(), k) for c, k in items if c and c.strip()]
        if not materialized:
            return []
        now = time.time()
        ids = [new_id() for _ in materialized]
        scope_name = normalize_scope(scope)
        batch_tags = join_tags(kw.get("tags", ""))
        if scope_name:
            batch_tags = (join_tags([scope_tag(scope_name), batch_tags])
                          if batch_tags else scope_tag(scope_name))
        rows = [
            {
                "id": ids[i], "kind": kind, "content": content,
                "importance": kw.get("importance", 0.5), "salience": kw.get("importance", 0.5),
                "decay": 1.0, "access_count": 0, "last_access": 0.0,
                "source": source, "agent": kw.get("agent", ""),
                "trust": infer_trust(source, kw.get("origin", ""), explicit=kw.get("trust", "")),
                "session_id": (kw.get("session_id") or kw.get("origin") or "").strip(),
                "tags": batch_tags,
                "created_at": now, "updated_at": now, "metadata": kw.get("metadata") or {},
            }
            for i, (content, kind) in enumerate(materialized)
        ]
        vectors = self.embedder.embed_many([c for c, _ in materialized])
        with self.db.transaction():
            self.repo.create_many(rows)
            self.semantic.put_many(
                [(vectors[i], ids[i]) for i in range(len(ids))]
            )
            # NB: embedding_id is intentionally not written back here. The
            # VectorBackend.put_many contract returns one id per input but
            # does NOT guarantee input order (usearch groups by dimension),
            # so positional mapping would corrupt the pointer on some
            # backends. The column is informational only — nothing reads it
            # functionally — and repair_embeddings() sets it correctly via
            # the single-put path, which is per-owner by contract.
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
        origin: str = "",
        include_superseded: bool = False,
        trust_filter: str = "",
        explain: bool = False,
        scope: str = "",
    ) -> RecallResult:
        """Merged semantic + lexical + recency recall.

        ``tags`` (comma-separated) keeps only records carrying ALL of the
        requested tags — the tag lane the /remember command populates.

        ``scope`` (e.g. ``"project:devon-arena"``) restricts recall to one
        memory space: records scoped to a *different* space never leak in.
        Global (unscoped) records are always visible.  Empty (default) =
        no restriction — historical behaviour.

        ``origin`` (e.g. "chat:tg:123") boosts memories from the same
        origin — session-relevant memories rank higher, but global
        knowledge is still accessible (boost, not filter).

        ``trust_filter`` keeps only ``"trusted"`` or only ``"untrusted"``
        records; anything else disables the filter. Untrusted records are
        always *downranked* (mem-false-fact), and untrusted records from a
        different session are downranked *further* (mem-cross-session) —
        tool-planted content never outranks the owner's own memories.

        Private-marked records are excluded unless ``include_private`` is
        set — proactive recall and training pipelines never see them.

        Superseded records (replaced via ``supersede()``) are excluded
        unless ``include_superseded`` is set — recall surfaces the
        current fact; the audit trail stays reachable via ``get()`` and
        ``supersession_chain()``.

        ``explain=True`` attaches a per-record ``explanation`` dict tracing
        the recall path: each signal's weighted contribution, which lane(s)
        surfaced the record and at what rank, and every adjustment applied
        (session boost, trust downrank). The ranking itself is unchanged.

        The semantic lane degrades, never dies: if the vector index fails
        (e.g. embedding dimension drift after a provider switch), recall
        continues on the lexical lane and the incident is recorded for
        ``health()``. Recall never raises.
        """
        wanted_tags = {t.strip() for t in (tags or "").split(",") if t.strip()}
        trust_wanted = (trust_filter or "").strip().lower()
        if trust_wanted not in (TRUSTED, UNTRUSTED):
            trust_wanted = ""
        scope_wanted = normalize_scope(scope)
        started = time.perf_counter()
        limit = limit or self.limit
        if not query.strip():
            recents = self._recent(limit, kind=kind, trust_filter=trust_wanted,
                                   scope=scope_wanted)
            if not include_private:
                recents = [r for r in recents if not _is_private(r)]
            if not include_superseded:
                recents = [r for r in recents
                           if not (r.metadata or {}).get("superseded_by")]
            if explain:
                for record in recents:
                    record.explanation = {
                        "lane": "recent-fallback",
                        "signals": {},
                        "weights": dict(self.weights),
                        "score_before_adjustments": 0.0,
                        "adjustments": [],
                        "trust": record.trust,
                        "note": "empty query: recency order, no scoring",
                    }
            return RecallResult(records=recents, query=query,
                                elapsed_ms=(time.perf_counter() - started) * 1000)
        self.stats["recalls"] += 1

        candidates: dict[str, MemoryRecord] = {}
        semantic_scores: dict[str, float] = {}
        lexical_scores: dict[str, float] = {}
        semantic_ranks: dict[str, int] = {}
        lexical_ranks: dict[str, int] = {}

        # Semantic pass: over-fetch, because the merge re-ranks.
        # Degraded, never dead: a broken vector index (dimension drift,
        # corrupt store) falls back to lexical-only recall and is logged
        # for health(). A recall that raises is a recall that loses the
        # user's memories.
        vector = self.embedder.embed(query)
        try:
            for rank, hit in enumerate(self.semantic.search(vector, limit=limit * 6)):
                semantic_scores[hit.owner_id] = hit.score
                semantic_ranks.setdefault(hit.owner_id, rank)
                candidates[hit.owner_id] = hit.owner_id  # placeholder, resolved below
        except Exception as exc:  # noqa: BLE001 — degrade, don't die
            _log.warning("memory recall: semantic lane failed (%s); lexical-only",
                         classify(exc).message)
            self._note_health_event("semantic", classify(exc).message)

        # Lexical pass: catches identifiers and rare terms vectors blur.
        # rowid→id is resolved in ONE query, not one per hit (N+1): with a
        # busy FTS index this pass can return limit*4 hits per recall.
        fts_hits = self.fts.search(query, limit=limit * 4)
        if fts_hits:
            id_placeholders = ", ".join("?" for _ in fts_hits)
            id_rows = self.db.query(
                f"SELECT rowid, id FROM memories WHERE rowid IN ({id_placeholders})",
                [hit.rowid for hit in fts_hits],
            )
            rowid_to_id = {row["rowid"]: row["id"] for row in id_rows}
            for rank, hit in enumerate(fts_hits):
                record_id = rowid_to_id.get(hit.rowid)
                if record_id:
                    lexical_scores[record_id] = max(0.0, min(1.0, (hit.score + 20.0) / 25.0))
                    lexical_ranks.setdefault(record_id, rank)
                    candidates.setdefault(record_id, record_id)

        ids = list(candidates)
        if not ids:
            recents = self._recent(limit, kind=kind, trust_filter=trust_wanted,
                                   scope=scope_wanted)
            if not include_private:
                recents = [r for r in recents if not _is_private(r)]
            if not include_superseded:
                recents = [r for r in recents
                           if not (r.metadata or {}).get("superseded_by")]
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
            if not include_superseded and (record.metadata or {}).get("superseded_by"):
                continue
            if kind and record.kind != kind:
                continue
            if source and record.source != source:
                continue
            if wanted_tags:
                have = {t.strip() for t in (record.tags or "").split(",")}
                if not wanted_tags <= have:
                    continue
            # Scope gate: a record scoped to another space never leaks in.
            # Global records (no scope tag) stay visible everywhere.
            if scope_wanted and not record_matches_scope(record, scope_wanted):
                continue
            record.semantic = semantic_scores.get(record.id, 0.0)
            record.lexical = lexical_scores.get(record.id, 0.0)
            base_score, contributions = _explain_score(
                record,
                semantic=record.semantic,
                lexical=record.lexical,
                weights=self.weights,
                half_life_seconds=self.half_life,
                now=now,
            )
            record.score = base_score
            adjustments: list[str] = []
            # Session boost: memories from the same origin (chat) rank higher.
            # This is a boost, not a filter — global knowledge stays accessible.
            if origin and record.origin == origin:
                record.score = min(1.0, record.score + 0.15)
                adjustments.append("session_boost:+0.15")
            # Trust downrank (mem-false-fact): tool output / external content
            # never outranks the owner's own memories, no matter how well it
            # matches. Cross-session untrusted content is downranked further
            # (mem-cross-session). Never raises.
            if trust_wanted and record.trust != trust_wanted:
                continue
            if record.is_untrusted:
                try:
                    record.score *= 0.5
                    adjustments.append("untrusted_downrank:x0.5")
                    rec_session = (record.session_id or record.origin or "").strip()
                    if origin and rec_session and rec_session != origin.strip():
                        record.score *= 0.5
                        adjustments.append("cross_session_downrank:x0.5")
                except Exception:  # noqa: BLE001
                    pass
            if explain:
                lanes = []
                if record.id in semantic_ranks:
                    lanes.append("semantic")
                if record.id in lexical_ranks:
                    lanes.append("lexical")
                record.explanation = {
                    "lane": "+".join(lanes) if lanes else "unscored",
                    "signals": contributions,
                    "weights": dict(self.weights),
                    "score_before_adjustments": round(base_score, 5),
                    "adjustments": adjustments,
                    "semantic_rank": semantic_ranks.get(record.id),
                    "lexical_rank": lexical_ranks.get(record.id),
                    "trust": record.trust,
                    "note": "contributions sum to score_before_adjustments",
                }
            if record.score >= min_score:
                scored.append(record)

        scored.sort(key=lambda r: -r.score)
        top = scored[:limit]
        if explain and top:
            # normalize_scores rescales so the top hit is 1.0 — record the
            # factor so the explanation traces the final score exactly.
            peak = max(r.score for r in top)
            factor = (1.0 / peak) if peak > 0 else 1.0
            for record in top:
                if record.explanation:
                    record.explanation["adjustments"].append(
                        f"normalize:x{round(factor, 4)}")
        normalize_scores(top)
        if top:
            self._touch([r.id for r in top])
        return RecallResult(records=top, query=query, elapsed_ms=(time.perf_counter() - started) * 1000)

    def _recent(self, limit: int, *, kind: str = "",
                trust_filter: str = "", scope: str = "") -> list[MemoryRecord]:
        # Scoped recent: over-fetch then gate, so the limit still holds
        # after the anti-leak filter runs.
        fetch = limit * 4 if scope else limit
        query = self.repo.query().order_by("created_at DESC").limit(fetch)
        if kind:
            query.where("kind = ?", kind)
        sql, params = query.build()
        records = [MemoryRecord.from_row(r) for r in self.db.query(sql, params)]
        if scope:
            records = [r for r in records
                       if record_matches_scope(r, scope)][:limit]
        if trust_filter in (TRUSTED, UNTRUSTED):
            records = [r for r in records if r.trust == trust_filter]
        return records

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
            self.semantic.delete_owner(record_id)
            self.semantic.put(self.embedder.embed(payload["content"]), record_id)
            self._index_text(record_id, payload["content"])
        return count

    def supersede(self, old_record_id: str, new_content: str, **kwargs: Any) -> str:
        """Replace a fact without deleting it — the audit trail is preserved.

        Personal-AI-army pattern: facts are never deleted, only superseded.
        The old record keeps its content but is marked
        ``metadata.superseded_by``; the new record links back via
        ``metadata.supersedes``. Superseded records are excluded from
        default recall (they're stale) but stay queryable through
        ``get()`` and ``supersession_chain()``. Never raises.
        """
        try:
            old = self.get(old_record_id)
            if old is None:
                return ""
            md = dict(kwargs.pop("metadata", None) or {})
            md["supersedes"] = old_record_id
            new_id = self.remember(new_content, metadata=md, **kwargs)
            if not new_id:
                return ""
            old_md = dict(old.metadata or {})
            old_md["superseded_by"] = new_id
            old_md["superseded_at"] = time.time()
            self.update(old_record_id, metadata=old_md)
            self.stats["superseded"] = self.stats.get("superseded", 0) + 1
            return new_id
        except Exception:
            return ""

    def supersession_chain(self, record_id: str) -> list[MemoryRecord]:
        """Oldest → newest chain of supersessions. Never raises."""
        try:
            rec = self.get(record_id)
            if rec is None:
                return []
            # Walk back to the oldest ancestor.
            seen = {rec.id}
            while (rec.metadata or {}).get("supersedes"):
                parent_id = rec.metadata["supersedes"]
                if parent_id in seen:
                    break
                parent = self.get(parent_id)
                if parent is None:
                    break
                seen.add(parent.id)
                rec = parent
            # Walk forward to the newest descendant.
            chain = []
            while rec is not None and rec.id not in {r.id for r in chain}:
                chain.append(rec)
                nxt = (rec.metadata or {}).get("superseded_by")
                rec = self.get(nxt) if nxt else None
            return chain
        except Exception:
            return []

    def forget(self, record_id: str) -> int:
        self.semantic.delete_owner(record_id)
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

        Untrusted preference records are FLAGGED inline (mem-pref-override):
        they are never applied silently — the flag tells the caller to get
        explicit user confirmation first.
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
            flag = ""
            if record.requires_confirmation:
                flag = " [⚠ UNVERIFIED PREFERENCE — needs user confirmation]"
            elif record.is_untrusted:
                flag = " [untrusted]"
            line = f"- [{record.kind}]{flag} {record.content}"
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
        self._last_consolidation_report = dict(report)
        _log.debug("consolidation: %s", report)
        return report

    @staticmethod
    def _metadata_dict(row: dict[str, Any]) -> dict[str, Any]:
        """Parse a row's metadata into a dict. Never raises.

        Raw ``db.query`` rows carry metadata as a JSON string (only the
        Repository encodes/decodes it), so consolidation and health checks
        that read rows directly need this normalization.
        """
        try:
            md = row.get("metadata")
            if isinstance(md, str):
                md = json.loads(md or "{}")
            return dict(md or {})
        except Exception:  # noqa: BLE001
            return {}

    def consolidate_additive(
        self, *, min_episodes: int = 5, batch: int = 50
    ) -> dict[str, Any]:
        """Distil episodes into durable facts WITHOUT deleting anything.

        The additive sibling of :meth:`consolidate`: raw episodes stay fully
        recallable (the owner's "forget only on explicit command" rule), and
        each distilled episode is marked ``metadata.distilled_into`` so a
        re-run never distils the same episode twice. The distilled fact links
        back via ``metadata.distilled_ids``. Never raises.
        """
        started = time.perf_counter()
        report: dict[str, Any] = {
            "mode": "additive", "episodes": 0, "batches": 0,
            "summaries": 0, "forgotten": 0, "skipped_distilled": 0,
        }
        try:
            rows = self.db.query(
                "SELECT id, content, metadata, created_at FROM memories "
                "WHERE kind = ? ORDER BY created_at",
                (MemoryKind.EPISODE,),
            )
            pending: list[dict[str, Any]] = []
            for row in rows:
                if self._metadata_dict(row).get("distilled_into"):
                    report["skipped_distilled"] += 1
                    continue
                pending.append(row)
            report["episodes"] = len(pending)
            if len(pending) < min_episodes:
                # Not enough undistilled episodes to be worth a summary run.
                report["seconds"] = round(time.perf_counter() - started, 3)
                report["remaining"] = self.repo.count()
                self._last_consolidation_report = dict(report)
                return report
            for start in range(0, len(pending), batch):
                chunk = pending[start:start + batch]
                if len(chunk) < min_episodes:
                    # Small tail chunk: leave it to accumulate for the next
                    # run rather than forcing a thin summary.
                    report["deferred"] = report.get("deferred", 0) + len(chunk)
                    continue
                summary = self._summarize_episodes([r["content"] for r in chunk])
                if not summary:
                    continue
                fact_id = self.remember(
                    summary,
                    kind=MemoryKind.FACT,
                    importance=0.7,
                    source="consolidation",
                    metadata={
                        "distilled_from": len(chunk),
                        "distilled_ids": [r["id"] for r in chunk],
                        "additive": True,
                    },
                )
                if not fact_id:
                    continue
                report["batches"] += 1
                report["summaries"] += 1
                for row in chunk:
                    md = self._metadata_dict(row)
                    md["distilled_into"] = fact_id
                    self.update(row["id"], metadata=md)
        except Exception as exc:  # noqa: BLE001 — consolidation never breaks chat
            _log.warning("additive consolidation failed: %s", classify(exc).message)
            report["error"] = classify(exc).message
        report["seconds"] = round(time.perf_counter() - started, 3)
        report["remaining"] = self.repo.count()
        self._last_consolidation = time.time()
        self._last_consolidation_report = dict(report)
        self.stats["consolidations"] += 1
        return report

    # ── health & repair ──────────────────────────────────────────────────
    def health(self) -> dict[str, Any]:
        """Memory-system diagnostics. Never raises.

        Reports record counts, index parity (records ↔ vectors ↔ FTS),
        embedding dimension drift, the embedder's state, the last
        consolidation, and any degraded-recall incidents. The parity
        checks are what catch silent index corruption before it eats
        recall quality.
        """
        report: dict[str, Any] = {"ok": True, "problems": []}
        try:
            report["records"] = self.repo.count()
        except Exception:  # noqa: BLE001
            report["records"] = -1
            report["problems"].append("record count unreadable")
        try:
            report["by_kind"] = self.counts_by_kind()
        except Exception:  # noqa: BLE001
            report["by_kind"] = {}
        try:
            report["vectors"] = self.semantic.count()
            report["vector_backend"] = self.semantic.name
        except Exception as exc:  # noqa: BLE001
            report["vectors"] = -1
            report["problems"].append(f"vector backend unreadable: {exc}")
        try:
            report["fts_documents"] = self.fts.count()
        except Exception:  # noqa: BLE001
            report["fts_documents"] = -1
        try:
            report["embedder"] = self.embedder.stats_snapshot()
            report["embedder_dims"] = int(self.embedder.dimensions)
        except Exception:  # noqa: BLE001
            report["embedder"] = {}
            report["embedder_dims"] = -1

        # Index parity: every record should have exactly one vector and one
        # FTS document; every vector/FTS doc should point at a live record.
        emb_table = VectorStore.TABLE
        try:
            missing_vectors = self.db.query(
                f"SELECT m.id FROM memories m LEFT JOIN \"{emb_table}\" e "
                "ON e.owner_id = m.id AND e.owner_type = 'memory' "
                "WHERE e.id IS NULL LIMIT 20"
            )
            orphan_vectors = self.db.query(
                f"SELECT e.owner_id FROM \"{emb_table}\" e LEFT JOIN memories m "
                "ON m.id = e.owner_id WHERE e.owner_type = 'memory' "
                "AND m.id IS NULL LIMIT 20"
            )
            fts_orphans = self.db.scalar(
                "SELECT COUNT(*) FROM memories_fts WHERE rowid NOT IN "
                "(SELECT rowid FROM memories)",
                default=0,
            )
            records_missing_fts = self.db.query(
                "SELECT m.id FROM memories m LEFT JOIN memories_fts f "
                "ON f.rowid = m.rowid WHERE f.rowid IS NULL LIMIT 20"
            )
            report["parity"] = {
                "records_missing_vectors": [r["id"] for r in missing_vectors],
                "orphan_vectors": [r["owner_id"] for r in orphan_vectors],
                "orphan_fts_documents": int(fts_orphans or 0),
                "records_missing_fts": [r["id"] for r in records_missing_fts],
            }
            for key, bad in (
                ("records_missing_vectors", report["parity"]["records_missing_vectors"]),
                ("orphan_vectors", report["parity"]["orphan_vectors"]),
                ("records_missing_fts", report["parity"]["records_missing_fts"]),
            ):
                if bad:
                    report["problems"].append(f"{key}: {len(bad)}+ (showing first 20)")
            if report["parity"]["orphan_fts_documents"]:
                report["problems"].append(
                    f"orphan_fts_documents: {report['parity']['orphan_fts_documents']}"
                )
        except Exception as exc:  # noqa: BLE001
            report["parity"] = {"error": str(exc)[:200]}

        # Dimension drift: vectors stored under a different embedding
        # dimension than the current embedder produces. The semantic lane
        # degrades to lexical-only until repair_embeddings() runs.
        try:
            dim_rows = self.db.query(
                f"SELECT DISTINCT dim FROM \"{emb_table}\" WHERE owner_type = 'memory'"
            )
            dims = sorted(int(r["dim"]) for r in dim_rows)
            report["stored_dims"] = dims
            report["dimension_drift"] = len(dims) > 1 or (
                len(dims) == 1 and dims[0] != report.get("embedder_dims", dims[0])
            )
            if report["dimension_drift"]:
                report["problems"].append(
                    f"dimension drift: stored dims {dims} vs embedder "
                    f"{report.get('embedder_dims')} — run repair_embeddings()"
                )
        except Exception as exc:  # noqa: BLE001
            report["stored_dims"] = []
            report["dimension_drift"] = False
            report["problems"].append(f"drift check failed: {exc}")

        report["last_consolidation"] = self._last_consolidation
        report["last_consolidation_report"] = dict(self._last_consolidation_report)
        # The consolidation cadence: is the schedule alive, when does it
        # next run, how big is the undistilled backlog.  Additive-only —
        # the cadence never deletes, so this is safe to run hands-off.
        try:
            report["consolidation_cadence"] = _cadence_status(self)
        except Exception:  # noqa: BLE001
            report["consolidation_cadence"] = {}
        report["degraded_recalls"] = len(self._health_events)
        report["recent_events"] = list(self._health_events[-10:])
        report["stats"] = dict(self.stats)
        report["weights"] = dict(self.weights)
        if report["problems"]:
            report["ok"] = False
        return report

    def repair_embeddings(self, *, dry_run: bool = False) -> dict[str, Any]:
        """Re-embed records whose vectors are missing or dimension-drifted.

        Additive repair: records are never touched, only their vectors are
        rebuilt with the current embedder. Fixes the state ``health()``
        reports as ``dimension_drift`` or ``records_missing_vectors`` —
        e.g. after switching the embedding provider. Never raises.
        """
        report: dict[str, Any] = {
            "dry_run": dry_run, "repaired": 0, "missing": 0,
            "drifted": 0, "failed": 0,
        }
        try:
            try:
                current_dims = int(self.embedder.dimensions)
            except Exception:  # noqa: BLE001
                current_dims = 0
            emb_table = VectorStore.TABLE
            rows = self.db.query(
                f"SELECT m.id, m.content, e.dim AS dim FROM memories m "
                f"LEFT JOIN \"{emb_table}\" e ON e.owner_id = m.id "
                "AND e.owner_type = 'memory'"
            )
            targets: list[tuple[str, str, str]] = []
            for row in rows:
                if not (row.get("content") or "").strip():
                    continue
                dim = row.get("dim")
                if dim is None:
                    report["missing"] += 1
                    targets.append((row["id"], row["content"], "missing"))
                elif current_dims and int(dim) != current_dims:
                    report["drifted"] += 1
                    targets.append((row["id"], row["content"], "drifted"))
            report["targets"] = len(targets)
            if dry_run:
                report["target_ids"] = [t[0] for t in targets[:20]]
                return report
            for record_id, content, _reason in targets:
                try:
                    vector = self.embedder.embed(content)
                    self.semantic.delete_owner(record_id)
                    embedding_id = self.semantic.put(vector, record_id)
                    self.db.execute(
                        "UPDATE memories SET embedding_id = ? WHERE id = ?",
                        (embedding_id, record_id),
                    )
                    report["repaired"] += 1
                except Exception as exc:  # noqa: BLE001
                    report["failed"] += 1
                    _log.warning("repair_embeddings: %s failed: %s",
                                 record_id, classify(exc).message)
        except Exception as exc:  # noqa: BLE001
            report["error"] = classify(exc).message
        return report

    def _summarize_episodes(self, contents: Sequence[str]) -> str:
        router = getattr(self.context, "router", None) if self.context else None
        corpus = "\n".join(f"- {c}" for c in contents)
        if router is not None:
            from ..llm.base import Message, SamplingParams

            response = brain_for(self.context).chat(
                [
                    Message.system("Distil these notes into at most 5 durable facts. One per line."),
                    Message.user(corpus[:8000]),
                ],
                SamplingParams(temperature=0.2, max_tokens=512),
            task_kind="judge")
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
            "vectors": self.semantic.count(),
            "vector_backend": self.semantic.name,
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
