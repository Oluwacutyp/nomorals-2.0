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

__all__ = ["DEFAULT_WEIGHTS", "MemoryKind", "MemoryRecord", "decay_factor", "score_memory"]

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
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Transient, populated during recall — never persisted.
    score: float = 0.0
    semantic: float = 0.0
    lexical: float = 0.0
    embedding_id: str = ""

    def age(self, now: float | None = None) -> float:
        return max(0.0, (now or time.time()) - self.created_at)

    @property
    def expired(self) -> bool:
        return self.expires_at is not None and time.time() > self.expires_at

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
        return {
            "id": self.id,
            "kind": self.kind,
            "content": self.content,
            "importance": self.importance,
            "salience": self.salience,
            "decay": self.decay,
            "access_count": self.access_count,
            "source": self.source,
            "agent": self.agent,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "metadata": self.metadata,
            "score": round(self.score, 5),
            "semantic": round(self.semantic, 5),
            "lexical": round(self.lexical, 5),
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "MemoryRecord":
        metadata = row.get("metadata")
        if isinstance(metadata, str):
            import json

            try:
                metadata = json.loads(metadata or "{}")
            except (ValueError, TypeError):
                metadata = {}
        return cls(
            id=row["id"],
            kind=row.get("kind") or MemoryKind.EPISODE,
            content=row.get("content") or "",
            importance=float(row.get("importance") or 0.5),
            salience=float(row.get("salience") or 0.5),
            decay=float(row.get("decay") or 1.0),
            access_count=int(row.get("access_count") or 0),
            last_access=float(row.get("last_access") or 0.0),
            source=row.get("source") or "",
            agent=row.get("agent") or "",
            created_at=float(row.get("created_at") or time.time()),
            updated_at=float(row.get("updated_at") or time.time()),
            expires_at=row.get("expires_at"),
            metadata=metadata or {},
            embedding_id=row.get("embedding_id") or "",
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
