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
import json
import math
import os
import re
import time
import urllib.request
from typing import Any, Callable, Sequence

from ..core.errors import classify
from ..core.logging_setup import get_logger
from ..core.text import normalize_text, ngrams

__all__ = [
    "Embedder",
    "contextualize_chunk",
    "matryoshka_truncate",
    "select_for_profile",
]

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


class Qwen3Embedder:
    """Qwen3-Embedding-0.6B via a local llama-server ``/v1/embeddings`` endpoint.

    Apache 2.0, ~0.6B params — beats commercial embedding APIs on MTEB
    multilingual while running fully offline on the owner's own hardware.
    Serve it with a GGUFServerManager on a second port (embeddings want
    their own server: different model, different context needs)::

        mgr = GGUFServerManager(port=8081, ctx_size=512)
        mgr.start("qwen3-embedding-0.6b-q8_0.gguf")

    Config: ``EMBEDDING_URL`` (default ``http://127.0.0.1:8081``),
    ``EMBEDDING_MODEL`` (default ``qwen3-embedding-0.6b``). Lazy — no
    network traffic until the first embed call. Raises on failure; the
    :class:`Embedder` ``"qwen3"`` mode catches that and falls back to
    hashing (failures are counted in stats, never fatal).
    """

    def __init__(self, url: str = "", model: str = "",
                 timeout: float = 10.0) -> None:
        self.url = (url or os.environ.get("EMBEDDING_URL", "")
                    or "http://127.0.0.1:8081").rstrip("/")
        self.model = (model or os.environ.get("EMBEDDING_MODEL", "")
                      or "qwen3-embedding-0.6b")
        self.timeout = timeout
        self.dimensions = 1024  # Qwen3-Embedding-0.6B native dim

    def available(self) -> bool:
        """Quick probe — True when the endpoint answers. Never raises."""
        try:
            vectors = self.embed_many(["probe"])
            return bool(vectors and len(vectors[0]) > 0)
        except Exception:  # noqa: BLE001
            return False

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model, "input": list(texts)}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.url}/v1/embeddings", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"qwen3 embedding endpoint failed: {exc}") from exc
        try:
            items = sorted(payload["data"], key=lambda d: d["index"])
            return [list(map(float, d["embedding"])) for d in items]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected embeddings response shape: {exc}") from exc


class Embedder:
    """Produces fixed-dimension vectors for text.

    Provider modes:
    - ``"hashing"``: deterministic feature-hashing (default, offline, zero deps).
    - ``"auto"``: probe the router for a working embedding backend on first
      use; use it if available, otherwise fall back to hashing. The probe
      result is cached so we don't pay for a failed probe on every call.
    - ``"native"``: native-first — the local Qwen3-Embedding-0.6B server
      when it is up, hashing when it is not. NEVER touches the router or
      any API embedding endpoint: fully offline, zero data egress.
    - ``"qwen3"``: Qwen3-Embedding-0.6B via a local llama-server embeddings
      endpoint (:class:`Qwen3Embedder`); falls back to hashing when the
      server is down.
    - Any other string: use the router's embedding backend directly, falling
      back to hashing on failure (existing behavior).

    Honest quality ladder (no marketing):
    - ``hashing``: captures *lexical overlap* — shared tokens land in the
      same dimensions, so "recall what we discussed about X" works. It
      degrades on paraphrase ("automobile" vs "car") and on cross-lingual
      queries. Zero cost, zero latency, works on a phone.
    - ``native``/``qwen3``: real dense semantic embeddings from a 0.6B
      model running on the owner's own hardware — paraphrase and
      multilingual recall work at roughly API-class quality for
      personal-memory scale. Costs: a ~500MB model download, a
      llama-server process, and workstation-class RAM/CPU. Not a phone
      default.
    - router/API embeddings: the best raw quality when online, at the
      price of an external dependency and sending memory text to a third
      party. Fallback, never the priority.
    """

    #: quality ladder, weakest → strongest.  Used by
    #: :func:`select_for_profile` and surfaced in stats.
    QUALITY_LADDER = ("hashing", "native", "router")

    def __init__(
        self,
        *,
        provider: str = "auto",
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
        # Auto-mode probe state: None = not probed yet, True/False = result.
        self._auto_probed: bool | None = None
        self._auto_works: bool = False
        # qwen3-mode state: lazily-built embedder, None = not probed yet.
        self._qwen3: Qwen3Embedder | None = None
        self._qwen3_works: bool | None = None
        # Brain over the router for embeddings: the never-raises
        # ``brain.embed`` returns (vectors, error) instead of raising, so
        # the hashing fallback below stays simple and honest.
        self._brain: Any = None

    def _brain_for_embed(self) -> Any:
        """Brain wrapping ``self.router`` (or the router itself when the
        wrap is impossible).  Never raises."""
        if self._brain is None:
            try:
                from ..llm.brain import Brain

                self._brain = Brain(router=self.router)
            except Exception:  # noqa: BLE001
                return self.router
        return self._brain

    # ── public API ───────────────────────────────────────────────────────────
    @property
    def is_semantic(self) -> bool:
        if self.provider == "hashing":
            return False
        if self.provider in ("qwen3", "native"):
            # Optimistic until a probe fails — same contract as auto mode.
            return self._qwen3_works is not False
        if self.provider == "auto":
            # In auto mode, we're semantic if the probe succeeded.
            # If not probed yet, check router availability optimistically.
            if self._auto_probed is not None:
                return self._auto_works
            return self.router is not None
        return self.router is not None

    def _probe_auto(self) -> bool:
        """Probe the router for working embeddings (auto mode only).

        Called once on first use. Returns True if embeddings work.
        Skips mock providers — their embeddings are just a different hash,
        not semantic, so hashing is equally good (and faster).
        """
        if self._auto_probed is not None:
            return self._auto_works
        self._auto_probed = True
        if self.router is None:
            self._auto_works = False
            return False
        # Skip if the only embedding providers are mocks.
        try:
            providers = self.router.providers()
            # Check if there's a non-mock provider with embed capability.
            has_real = False
            for name in providers:
                if name == "mock":
                    continue
                provider = self.router.get(name)
                if provider is not None and "embed" in provider.capabilities:
                    has_real = True
                    break
            if not has_real:
                self._auto_works = False
                return False
        except Exception as exc:  # noqa: BLE001
            _log.debug("embedding auto-mode: provider capability check failed (%s)", exc)
        try:
            vectors, error = self._brain_for_embed().embed(["probe"])
            if vectors and len(vectors[0]) > 0:
                self._auto_works = True
                self.dimensions = len(vectors[0])
                _log.info("embedding auto-mode: using router backend (dim=%d)", self.dimensions)
                return True
            if error:
                _log.debug("embedding auto-mode probe failed (%s); using hashing", error)
        except Exception as exc:  # noqa: BLE001
            _log.debug("embedding auto-mode probe failed (%s); using hashing", classify(exc).message)
        self._auto_works = False
        self.stats["fallbacks"] += 1
        return False

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
        # native mode: local Qwen3 server only, never the router/API —
        # the offline-first default the Devon Studio program demands.
        # qwen3 mode: same mechanism (historical behaviour preserved:
        # local server, hashing fallback).  "native" is what profile
        # selection picks; "qwen3" is the explicit pin.
        if self.provider in ("native", "qwen3"):
            return self._produce_qwen3(texts)
        # Auto mode: probe once, then use the cached result.
        if self.provider == "auto":
            if self._probe_auto():
                vectors, error = self._brain_for_embed().embed(list(texts))
                if vectors and all(len(v) == len(vectors[0]) for v in vectors):
                    self.dimensions = len(vectors[0])
                    return [_l2(v) for v in vectors]
                if error:
                    _log.debug("embedding backend failed (%s); using hashing", error)
                self.stats["fallbacks"] += 1
            return [self._hash(text) for text in texts]
        if self.is_semantic:
            vectors, error = self._brain_for_embed().embed(list(texts))
            if vectors and all(len(v) == len(vectors[0]) for v in vectors):
                self.dimensions = len(vectors[0])
                return [_l2(v) for v in vectors]
            if error:
                _log.debug("embedding provider failed (%s); using hashing", error)
            else:
                _log.warning("embedding provider returned ragged vectors; using hashing")
            self.stats["fallbacks"] += 1
        return [self._hash(text) for text in texts]

    def _produce_qwen3(self, texts: Sequence[str]) -> list[list[float]]:
        """Local Qwen3 embedding server with hashing fallback. Never raises.

        Offline-only: the router/API is never consulted here — an embedding
        provider that phones home is not a native embedding provider.
        """
        if self._qwen3_works is not False:
            try:
                if self._qwen3 is None:
                    self._qwen3 = Qwen3Embedder(model=self.model or "")
                vectors = self._qwen3.embed_many(list(texts))
                if vectors and all(len(v) == len(vectors[0]) for v in vectors):
                    self._qwen3_works = True
                    self.dimensions = len(vectors[0])
                    return [_l2(v) for v in vectors]
                _log.warning("qwen3 embedding server returned ragged vectors; using hashing")
            except Exception as exc:  # noqa: BLE001 - never fail recall over embeddings
                _log.debug("qwen3 embedding server failed (%s); using hashing", exc)
            self._qwen3_works = False
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


def contextualize_chunk(document: str, source: str = "",
                        head_chars: int = 400) -> str:
    """Build the document-context header for contextual chunking.

    The implementable half of Anthropic's Contextual Retrieval / Jina's
    late chunking: a short, stable header ("this is from document X, which
    is about Y") prepended to each chunk's *embedding text* so chunks that
    are meaningless alone ("he doubled the dose") still carry document
    context into the vector. The stored/displayed chunk text is unchanged.

    Heuristic-first (no LLM call): document head + source. Callers with a
    brain can replace the head with an LLM summary and pass the result
    through the same ``<header>\\n<chunk>`` shape.
    """
    head = re.sub(r"\s+", " ", (document or "").strip())[:head_chars]
    src = (source or "note").strip()
    if head:
        return f"[document: {src} — {head}]"
    return f"[document: {src}]"


def matryoshka_truncate(vector: Sequence[float], dims: int) -> list[float]:
    """Truncate a vector to ``dims`` (Matryoshka embedding support).

    Matryoshka-trained models (Qwen3-Embedding, jina-v3, BGE-M3) order
    dimensions by importance, so dropping the tail costs little quality
    while halving storage/compute. A no-op when ``dims`` is 0 or already
    ≥ the vector length. The caller should L2-normalize afterwards for
    cosine backends (the vector backends do this on put).
    """
    if dims <= 0 or dims >= len(vector):
        return list(vector)
    return list(vector[:dims])


def select_for_profile(profile: str | None, *,
                       prefer_local_server: bool = True) -> str:
    """Pick an embedding provider for a resource profile — never designed
    down, profile-gated at runtime (standing rule).

    - ``"termux"`` → ``"hashing"``: the phone gets zero-cost, zero-latency
      lexical recall; a 0.6B embedding server is not a phone workload.
    - ``"laptop"`` / ``"workstation"`` → ``"native"`` when the local
      Qwen3 server answers (probed, cached), else ``"hashing"``: full
      semantic quality where the hardware can run it, honest fallback
      where it can't.
    - anything else → ``"auto"`` (historical behaviour: router, then
      hashing).

    The API embedding path is never *selected* here — it stays available
    as an explicit ``provider=`` choice or via ``"auto"``, as a fallback,
    never the priority.
    """
    p = (profile or "").strip().lower()
    if p == "termux":
        return "hashing"
    if p in ("laptop", "workstation", "server"):
        if prefer_local_server:
            try:
                if Qwen3Embedder().available():
                    return "native"
            except Exception:  # noqa: BLE001 — probe is best-effort
                pass
        return "hashing"
    return "auto"
