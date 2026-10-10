"""Memory records and relevance scoring.

Recall is a *merge*, not a lookup. Four signals, each normalized to [0, 1]:

* **recency** — exponential decay with a configurable half-life
* **importance** — how much the record matters, set at write time and revisited
* **semantic** — cosine similarity to the query embedding
* **lexical** — bm25 from FTS5, which catches exact identifiers vectors miss

The weights are not constants: the reflector tunes them from mission outcomes, so
the system learns how to remember. That loop is the difference between a vector
database and a memory.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DEFAULT_WEIGHTS",
    "MemoryKind",
    "MemoryRecord",
    "TRUSTED",
    "UNTRUSTED",
    "decay_factor",
    "format_record",
    "format_recall",
    "infer_trust",
    "score_memory",
]

DEFAULT_WEIGHTS: dict[str, float] = {
    "recency": 0.25,
    "importance": 0.30,
    "semantic": 0.30,
    "lexical": 0.15,
}


class MemoryKind:
    EPISODE = "episode"      # something that happened
    FACT = "fact"            # something that is true
    PREFERENCE = "preference"  # how the user wants things done
    SKILL = "skill"          # a procedure that worked
    LESSON = "lesson"        # what a failure taught
    DECISION = "decision"    # a choice that was made
    RELATIONSHIP = "relationship"  # a connection between entities


#: All memory kinds for iteration/validation
ALL_KINDS = frozenset({
    MemoryKind.EPISODE,
    MemoryKind.FACT,
    MemoryKind.PREFERENCE,
    MemoryKind.SKILL,
    MemoryKind.LESSON,
    MemoryKind.DECISION,
    MemoryKind.RELATIONSHIP,
})


# ── trust provenance ──────────────────────────────────────────────────────
# Root-cause fix for mem-false-fact / mem-pref-override / mem-cross-session:
# every record carries *where it came from*, so recall can downrank content
# that arrived via tool output or other external paths instead of surfacing
# it as trusted truth.

#: direct user/owner input — full weight in recall
TRUSTED = "trusted"
#: tool output, external content, or anything not written by the owner —
#: recalled only with a downrank, and preferences are flagged for confirmation
UNTRUSTED = "untrusted"

_TRUSTED_SOURCE_PREFIXES = ("user:", "owner:", "cli", "tui")

#: substrings in a record's source/origin that mark external provenance:
#: tool calls, web/browser output, document ingestion, extraction passes,
#: connector results, and other agent-written (non-owner) content.
_UNTRUSTED_SOURCE_MARKERS = (
    "tool", "browser", "web", "page", "search", "scrape", "crawl",
    "external", "extract", "document", "ingest", "connector",
    "agent", "curate", "feed", "rss",
)


def infer_trust(source: str = "", origin: str = "",
                explicit: str = "") -> str:
    """Resolve a record's trust level. Never raises.

    Precedence: an explicit ``"trusted"``/``"untrusted"`` wins; a source
    that starts with a user/owner prefix is trusted; a source/origin that
    carries a tool/external marker is untrusted; anything else (including
    legacy records with an empty source) defaults to trusted so existing
    recall behaviour is unchanged.
    """
    try:
        e = (explicit or "").strip().lower()
        if e in (TRUSTED, UNTRUSTED):
            return e
        s = (source or "").strip().lower()
        if any(s.startswith(p) for p in _TRUSTED_SOURCE_PREFIXES):
            return TRUSTED
        if any(m in s for m in _UNTRUSTED_SOURCE_MARKERS):
            return UNTRUSTED
        o = (origin or "").strip().lower()
        if any(m in o for m in _UNTRUSTED_SOURCE_MARKERS):
            return UNTRUSTED
        return TRUSTED
    except Exception:  # noqa: BLE001
        return TRUSTED


#: Kinds that decay slowly because they stay relevant.
SLOW_DECAY = frozenset({MemoryKind.FACT, MemoryKind.PREFERENCE, MemoryKind.SKILL})


def decay_factor(age_seconds: float, half_life_seconds: float) -> float:
    """Exponential decay: 1.0 at age 0, 0.5 at one half-life."""
    if half_life_seconds <= 0:
        return 1.0
    if age_seconds <= 0:
        return 1.0
    return math.exp(-math.log(2) * age_seconds / half_life_seconds)


@dataclass
class MemoryRecord:
    """One memory."""

    id: str
    kind: str
    content: str
    importance: float = 0.5
    salience: float = 0.5
    decay: float = 1.0
    access_count: int = 0
    last_access: float = 0.0
    source: str = ""
    agent: str = ""
    #: trust provenance: "trusted" (direct user/owner input) or
    #: "untrusted" (tool output / external content). Resolved at write time
    #: by ``infer_trust()``; never empty after ``from_row``.
    trust: str = ""
    #: session (chat) this record was captured in; used to downrank
    #: untrusted records recalled across sessions.
    session_id: str = ""
    #: comma-joined normalized tags (see manager.join_tags) and the pointer
    #: to the chat this memory was distilled from ("chat:tg:123")
    tags: str = ""
    origin: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Transient, populated during recall — never persisted.
    score: float = 0.0
    semantic: float = 0.0
    lexical: float = 0.0
    embedding_id: str = ""
    #: Populated by ``MemoryManager.recall(..., explain=True)``: the full
    #: trace of why this record ranked where it did — per-signal weighted
    #: contributions, surfacing lane(s) and ranks, adjustments applied.
    explanation: dict[str, Any] = field(default_factory=dict)

    def age(self, now: float | None = None) -> float:
        return max(0.0, (now or time.time()) - self.created_at)

    @property
    def expired(self) -> bool:
        return self.expires_at is not None and time.time() > self.expires_at

    @property
    def is_trusted(self) -> bool:
        """True when this record came from direct user/owner input."""
        return self.trust == TRUSTED

    @property
    def is_untrusted(self) -> bool:
        """True when this record arrived via tool output or external content."""
        return self.trust == UNTRUSTED

    @property
    def requires_confirmation(self) -> bool:
        """Untrusted preferences are never applied silently: mem-pref-override.

        A ``kind="preference"`` record from an untrusted source must be
        confirmed by the user before it changes behaviour.
        """
        return self.kind == MemoryKind.PREFERENCE and self.trust == UNTRUSTED

    def recency(self, half_life_seconds: float, now: float | None = None) -> float:
        """Decay rate depends on kind: facts persist, episodes fade."""
        life = half_life_seconds * 8 if self.kind in SLOW_DECAY else half_life_seconds
        return decay_factor(self.age(now), life) * self.decay

    def reinforcement(self) -> float:
        """Accessing a memory strengthens it — a log-linear bonus, bounded."""
        if self.access_count <= 0:
            return 0.0
        return min(0.2, 0.05 * math.log1p(self.access_count))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "kind": self.kind,
            "content": self.content,
            "importance": self.importance,
            "salience": self.salience,
            "decay": self.decay,
            "access_count": self.access_count,
            "source": self.source,
            "agent": self.agent,
            "trust": self.trust,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "metadata": self.metadata,
            "score": round(self.score, 5),
            "semantic": round(self.semantic, 5),
            "lexical": round(self.lexical, 5),
        }
        if self.explanation:
            payload["explanation"] = self.explanation
        return payload

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "MemoryRecord":
        metadata = row.get("metadata")
        if isinstance(metadata, str):
            import json

            try:
                metadata = json.loads(metadata or "{}")
            except (ValueError, TypeError):
                metadata = {}
        source = row.get("source") or ""
        origin = row.get("origin") or ""
        return cls(
            id=row["id"],
            kind=row.get("kind") or MemoryKind.EPISODE,
            content=row.get("content") or "",
            # NB: explicit None checks — 0.0 is a valid importance/salience
            # (a barely-stated preference), not a missing value
            importance=float(0.5 if row.get("importance") is None
                             else row["importance"]),
            salience=float(0.5 if row.get("salience") is None
                           else row["salience"]),
            decay=float(row.get("decay") or 1.0),
            access_count=int(row.get("access_count") or 0),
            last_access=float(row.get("last_access") or 0.0),
            source=source,
            agent=row.get("agent") or "",
            # Trust is always normalized: legacy rows (no trust column yet)
            # resolve through infer_trust() from their source/origin, so
            # nothing is ever recalled with an ambiguous provenance.
            trust=infer_trust(source, origin, explicit=row.get("trust") or ""),
            session_id=row.get("session_id") or "",
            created_at=float(row.get("created_at") or time.time()),
            updated_at=float(row.get("updated_at") or time.time()),
            expires_at=row.get("expires_at"),
            metadata=metadata or {},
            embedding_id=row.get("embedding_id") or "",
            tags=row.get("tags") or "",
            origin=row.get("origin") or "",
        )


def score_memory(
    record: MemoryRecord,
    *,
    semantic: float = 0.0,
    lexical: float = 0.0,
    weights: dict[str, float] | None = None,
    half_life_seconds: float = 259200.0,
    now: float | None = None,
) -> float:
    """Weighted relevance in [0, 1]. Higher means recall it first."""
    w = weights or DEFAULT_WEIGHTS
    total = sum(w.values()) or 1.0
    recency = record.recency(half_life_seconds, now)
    importance = max(0.0, min(1.0, record.importance + record.reinforcement()))
    semantic = max(-1.0, min(1.0, semantic))
    lexical = max(0.0, min(1.0, lexical))
    raw = (
        w.get("recency", 0.0) * recency
        + w.get("importance", 0.0) * importance
        + w.get("semantic", 0.0) * max(0.0, semantic)
        + w.get("lexical", 0.0) * lexical
    )
    return max(0.0, min(1.0, raw / total))


def normalize_scores(records: list[MemoryRecord]) -> None:
    """Rescale a result set so the top hit is 1.0.

    Makes scores comparable across queries, which matters when the orchestrator
    decides whether a recall was good enough to act on.
    """
    if not records:
        return
    top = max(r.score for r in records)
    if top <= 0:
        return
    for record in records:
        record.score = record.score / top


# ── presentation ────────────────────────────────────────────────────────────
# Memory that *looks* god-tier, not functional. Four output styles for the
# different surfaces that render recall: logs/debug (compact), the model
# prompt (rich), chat/Telegram (chat), and the morning briefing (briefing).

RECORD_STYLES = ("compact", "rich", "chat", "briefing")

_KIND_GLYPH = {
    MemoryKind.EPISODE: "📝",
    MemoryKind.FACT: "📌",
    MemoryKind.PREFERENCE: "💭",
    MemoryKind.SKILL: "🛠️",
    MemoryKind.LESSON: "🎓",
    MemoryKind.DECISION: "⚖️",
    MemoryKind.RELATIONSHIP: "🤝",
}


def _age_str(record: MemoryRecord, now: float) -> str:
    age = record.age(now)
    if age < 3600:
        return f"{max(1, int(age // 60))}m ago"
    if age < 86400:
        return f"{int(age // 3600)}h ago"
    days = int(age // 86400)
    if days < 30:
        return f"{days}d ago"
    if days < 365:
        return f"{days // 30}mo ago"
    return f"{days // 365}y ago"


def format_record(record: MemoryRecord, style: str = "compact",
                  now: float | None = None) -> str:
    """Render one record for a surface.

    Styles:
    - ``"compact"`` — one line for logs/debug: ``📌 fact · 0.87 · content``.
    - ``"rich"`` — multi-line with provenance, score, age: for prompt
      context and inspection.
    - ``"chat"`` — short, human, no scores or metadata: safe to show the
      user in a Telegram/WhatsApp message.
    - ``"briefing"`` — magazine-style line with kind glyph and age: for
      the morning pulse / digest surfaces.

    Unknown styles fall back to ``"compact"``. Never raises.
    """
    try:
        style = (style or "compact").lower()
        ts = now or time.time()
        content = (record.content or "").strip()
        glyph = _KIND_GLYPH.get(record.kind, "•")
        if style == "rich":
            trust = ("trusted" if record.is_trusted else "UNTRUSTED")
            lines = [
                f"{glyph} [{record.kind}] {content}",
                f"   score={record.score:.3f} age={_age_str(record, ts)} "
                f"trust={trust} accesses={record.access_count}",
            ]
            if record.requires_confirmation:
                lines.append("   ⚠ unverified preference — confirm with the "
                             "owner before applying")
            if record.tags:
                lines.append(f"   tags: {record.tags}")
            return "\n".join(lines)
        if style == "chat":
            flag = " (unverified — please confirm)" \
                if record.requires_confirmation else ""
            return f"{content}{flag}"
        if style == "briefing":
            return f"{glyph} {content} — {_age_str(record, ts)}"
        # compact
        score = f" · {record.score:.2f}" if record.score > 0 else ""
        trust = "" if record.is_trusted else " [untrusted]"
        return f"{glyph} {record.kind}{score} · {content}{trust}"
    except Exception:  # noqa: BLE001 — formatting never breaks the surface
        try:
            return (record.content or "").strip() or record.id
        except Exception:  # noqa: BLE001
            return ""


def format_recall(records: list[MemoryRecord], style: str = "compact",
                  *, header: str = "", now: float | None = None,
                  numbered: bool = False) -> str:
    """Render a recall result set in a style. "" when empty."""
    try:
        lines = [format_record(r, style, now)
                 for r in (records or [])]
        lines = [ln for ln in lines if ln]
        if not lines:
            return ""
        if numbered:
            lines = [f"{i + 1}. {ln}" for i, ln in enumerate(lines)]
        body = "\n".join(lines)
        return f"{header}\n{body}" if header else body
    except Exception:  # noqa: BLE001
        return ""
