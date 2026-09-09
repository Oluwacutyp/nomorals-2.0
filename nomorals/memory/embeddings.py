"""Embedding with graceful degradation.

Provider embeddings when online; deterministic feature-hashing embeddings when
not. Same interface, same store, so a phone with no network still has working
retrieval — just less semantic.

The hashing embedder is not a toy: it is a real random-projection embedding. It
captures lexical overlap (shared tokens land in the same dimensions), which is
enough for "recall what we discussed about X" and degrades only on paraphrase.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from typing import Any, Callable, Sequence

from ..core.errors import classify
from ..core.logging_setup import get_logger
from ..core.text import normalize_text, ngrams

__all__ = ["Embedder"]

_log = get_logger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")


_SUFFIXES = (
    "ational", "iveness", "fulness", "ousness", "ization", "isation",
    "ements", "ement", "ations", "ation", "ically", "ingly", "edly",
    "ings", "ing", "ness", "ment", "able", "ible", "ally", "ence",
    "ance", "ies", "ers", "est", "ed", "es", "ly", "er", "s",
)


def _stem(token: str) -> str:
    """Strip common English suffixes so morphological variants collide.

    Iterative on purpose: "preferred" -> "preferr" -> "pref" has to land on the
    same feature as "prefers" -> "pref", otherwise recall misses the obvious
    match. Deliberately crude overall — a full Porter stemmer is not worth the
    dependency, and over-stemming is harmless in a bag-of-features hash because
    the bigram features still carry word order.
    """
    if len(token) <= 3:
        return token
    for _ in range(3):
        stripped = token
        for suffix in _SUFFIXES:
            if stripped.endswith(suffix) and len(stripped) - len(suffix) >= 3:
                stripped = stripped[: -len(suffix)]
                if suffix in ("ed", "ing", "ingly", "edly"):
                    stripped = _undouble(stripped)
                break
        if stripped == token:
            break
        token = stripped
    return token


_DO_NOT_UNDOUBLE = frozenset({"ll", "ss", "zz"})


def _undouble(word: str) -> str:
    """Undo consonant doubling after dropping -ed/-ing.

    "preferred" -> "preferr" is wrong; Porter undoubles it to "prefer" so it
    lands on the same feature as "prefers". ll/ss/zz are real endings, not
    doubling, so they are left alone.
    """
    if len(word) >= 3 and word[-1] == word[-2] and word[-1].isalpha() and word[-1] not in "aeiou":
        if word[-2:] not in _DO_NOT_UNDOUBLE:
            return word[:-1]
    return word


class Embedder:
    """Produces fixed-dimension vectors for text."""

    def __init__(
        self,
        *,
        provider: str = "hashing",
        model: str = "",
        dimensions: int = 512,
        router: Any = None,
        ngram: int = 2,
        cache_size: int = 4096,
    ) -> None:
        self.provider = provider
        self.model = model
        self.dimensions = dimensions
        self.router = router
        self.ngram = max(1, ngram)
        self._cache: dict[str, list[float]] = {}
        self._cache_size = cache_size
        self.stats = {"calls": 0, "texts": 0, "cache_hits": 0, "fallbacks": 0}

    # ── public API ───────────────────────────────────────────────────────────
    @property
    def is_semantic(self) -> bool:
        return self.provider != "hashing" and self.router is not None

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        self.stats["calls"] += 1
        self.stats["texts"] += len(texts)
        if not texts:
            return []

        vectors: list[list[float] | None] = [None] * len(texts)
        missing: list[int] = []
        for index, text in enumerate(texts):
            key = self._key(text)
            cached = self._cache.get(key)
            if cached is not None:
                vectors[index] = cached
                self.stats["cache_hits"] += 1
            else:
                missing.append(index)

        if missing:
            produced = self._produce([texts[i] for i in missing])
            for position, index in enumerate(missing):
                vector = produced[position]
                vectors[index] = vector
                self._remember(self._key(texts[index]), vector)
        return [v if v is not None else [0.0] * self.dimensions for v in vectors]

    # ── backends ─────────────────────────────────────────────────────────────
    def _produce(self, texts: Sequence[str]) -> list[list[float]]:
        if self.is_semantic:
            try:
                vectors = self.router.embed(list(texts))
                if vectors and all(len(v) == len(vectors[0]) for v in vectors):
                    self.dimensions = len(vectors[0])
                    return [_l2(v) for v in vectors]
                _log.warning("embedding provider returned ragged vectors; using hashing")
            except Exception as exc:  # noqa: BLE001 - never fail recall over embeddings
                _log.debug("embedding provider failed (%s); using hashing", classify(exc).message)
            self.stats["fallbacks"] += 1
        return [self._hash(text) for text in texts]

    def _hash(self, text: str) -> list[float]:
        """Feature-hashing embedding with unigram and n-gram features."""
        vector = [0.0] * self.dimensions
        normalized = normalize_text(text, lower=True)
        tokens = _TOKEN.findall(normalized)
        if not tokens:
            vector[0] = 1.0
            return vector

        stems = [_stem(token) for token in tokens]
        features: list[tuple[str, float]] = [(stem, 1.0) for stem in stems]
        if self.ngram > 1:
            features += [
                (" ".join(gram), 1.5) for gram in ngrams(stems, self.ngram)
            ]
        for feature, weight in features:
            digest = int.from_bytes(
                hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big"
            )
            index = digest % self.dimensions
            sign = 1.0 if (digest >> 63) & 1 else -1.0
            vector[index] += sign * weight
        return _l2(vector)

    # ── cache ────────────────────────────────────────────────────────────────
    def _key(self, text: str) -> str:
        return hashlib.blake2b(
            f"{self.provider}:{self.model}:{self.dimensions}:{text}".encode(), digest_size=16
        ).hexdigest()

    def _remember(self, key: str, vector: list[float]) -> None:
        if len(self._cache) >= self._cache_size:
            # Drop an arbitrary tenth rather than paying for an LRU on every write.
            for stale in list(self._cache)[: self._cache_size // 10]:
                del self._cache[stale]
        self._cache[key] = vector

    def clear_cache(self) -> None:
        self._cache.clear()

    def stats_snapshot(self) -> dict[str, Any]:
        return {
            **self.stats,
            "provider": self.provider,
            "model": self.model,
            "dimensions": self.dimensions,
            "cached": len(self._cache),
        }


def _l2(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return list(vector)
    return [v / norm for v in vector]
