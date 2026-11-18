"""Duplicate-candidate search over mirrored issues (autocraft port).

Autocraft finds duplicate candidates by vector similarity (>= 0.75,
at most 3 per item, same kind) and then lets a model judge the pairs.
Devon has no embedding model wired for this, so the cheap first pass
here is TF cosine over title + body tokens — the same threshold and
cap, the same shape, an honest stdlib rung. The model verdict still
comes from the brain (see :mod:`nomorals.triage.engine`); similarity
alone never closes anything.
"""
from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any

__all__ = ["tokens", "cosine_similarity", "duplicate_candidates"]

#: Autocraft's candidate threshold (docs/pipeline.md, Duplicates).
SIMILARITY_THRESHOLD = 0.75
#: Autocraft's per-item candidate cap.
MAX_CANDIDATES_PER_ITEM = 3
#: A closed item counts as a duplicate candidate only within a year.
CLOSED_WITHIN_DAYS = 365

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+\-.]*")

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "it", "this", "that", "be", "by", "at",
    "as", "from", "when", "please", "i", "we", "you", "my", "our",
})


def tokens(text: str) -> list[str]:
    """Lowercase word tokens, stopwords dropped."""
    return [t for t in _TOKEN_RE.findall((text or "").lower())
            if t not in _STOPWORDS and len(t) > 1]


def _term_freq(text: str) -> dict[str, int]:
    freq: dict[str, int] = {}
    for tok in tokens(text):
        freq[tok] = freq.get(tok, 0) + 1
    return freq


def cosine_similarity(a: str, b: str) -> float:
    """TF cosine between two texts; 1.0 for identical, 0.0 for disjoint."""
    fa, fb = _term_freq(a), _term_freq(b)
    if not fa or not fb:
        return 0.0
    dot = sum(fa[t] * fb.get(t, 0) for t in fa)
    if dot == 0:
        return 0.0
    na = math.sqrt(sum(v * v for v in fa.values()))
    nb = math.sqrt(sum(v * v for v in fb.values()))
    return dot / (na * nb)


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        text = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        from datetime import UTC
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def duplicate_candidates(
    items: list[dict[str, Any]], item: dict[str, Any], *,
    threshold: float = SIMILARITY_THRESHOLD,
    limit: int = MAX_CANDIDATES_PER_ITEM,
    ruled_out: frozenset[tuple[str, str]] | None = None,
    now: datetime | None = None,
) -> list[tuple[dict[str, Any], float]]:
    """Candidate duplicates for ``item`` among ``items``.

    Same repo, same kind (issues never pair with PRs), open items plus
    items closed within the last year; a pair an operator already ruled
    on (``ruled_out`` holds ``(key_a, key_b)`` in either order) is
    never asked again. Best similarity first, at most ``limit``.
    """
    from .store import item_key

    ruled = ruled_out or frozenset()
    if now is None:
        from datetime import UTC
        now = datetime.now(UTC)
    base_text = f"{item.get('title', '')}\n{item.get('body', '')}"
    base_key = item_key(str(item.get("repo", "")),
                        int(item.get("number", 0)))
    scored: list[tuple[float, dict[str, Any]]] = []
    for other in items:
        if other is item:
            continue
        if str(other.get("repo", "")) != str(item.get("repo", "")):
            continue
        if str(other.get("kind", "issue")) != str(item.get("kind", "issue")):
            continue
        other_key = item_key(str(other.get("repo", "")),
                             int(other.get("number", 0)))
        if other_key == base_key:
            continue
        if ((base_key, other_key) in ruled
                or (other_key, base_key) in ruled):
            continue
        state = str(other.get("state", "open"))
        if state != "open":
            updated = _parse_time(other.get("updated_at"))
            if updated is None or (now - updated).days > CLOSED_WITHIN_DAYS:
                continue
        sim = cosine_similarity(
            base_text,
            f"{other.get('title', '')}\n{other.get('body', '')}")
        if sim >= threshold:
            scored.append((sim, other))
    scored.sort(key=lambda pair: (-pair[0], pair[1].get("number", 0)))
    return [(other, sim) for sim, other in scored[:limit]]
