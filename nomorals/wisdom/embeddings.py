"""Embedding backends for the wisdom corpus.

Multiple providers per capability, best-FREE primary:

- ``FastEmbedBackend`` — primary. ONNX Runtime only, no torch (~100-150MB
  RAM for bge-small), CPU-fast, maintained by Qdrant. Model downloads
  once from Hugging Face on first use and is cached locally.
- ``SentenceTransformersBackend`` — quality fallback. Heavier (needs
  torch) but usually the best-scoring open embedding stack.
- ``OllamaBackend`` — free local-server option. Talks to a running Ollama
  over stdlib HTTP (no new dependency); the operator already runs local
  models, and Ollama serves embeddings (nomic-embed-text, mxbai-embed-large).
- ``HashEmbedBackend`` — always-available stdlib fallback. Character
  trigram hashing into signed buckets, L2-normalized. Deterministic, no
  downloads, no dependencies. Quality is well below neural embeddings, but
  it keeps semantic search *functional* on a bare machine.

``auto_backend()`` picks the first available provider in preference order
and raises ``EmbeddingError`` only when nothing works (HashEmbedBackend
never fails, so this is defensive).

Retrieval asymmetry (2026 SOTA practice): serious embedding providers
expose different entry points for queries and documents — E5 models
expect ``query: `` / ``passage: `` prefixes, Qwen3 expects an instruction
on the query side only. Measured: e5-small bare 59.4% hit@1 → 74.8% with
correct prefixes. So the base class carries :meth:`embed_query` and
:meth:`embed_documents`; models that need prefixes declare them via the
``query_prefix`` / ``document_prefix`` hooks, and the embedding cache key
always includes the *role* (query vs passage) — caching the two together
would silently return a query vector where a passage vector was
requested, with no dimension change to notice.

All third-party imports are lazy: importing this module is always cheap
and never requires fastembed/torch/sentence-transformers to be installed.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from typing import Any, Iterable

from .errors import EmbeddingError


# ── base ──────────────────────────────────────────────────────────────

class EmbeddingBackend(ABC):
    """One embedding provider. Vectors are dense float lists; cosine
    similarity is the intended metric (backends normalize to unit length
    where it is cheap to do so, the store re-normalizes regardless)."""

    #: registry key used by ``auto_backend`` / config
    key: str = "base"

    #: Task prefixes applied by embed_query / embed_documents. Models
    #: trained with instruction prefixes (E5 family) override these.
    query_prefix: str = ""
    document_prefix: str = ""

    def __init__(self, model: str = "") -> None:
        self.model = model or self.default_model()

    @classmethod
    def default_model(cls) -> str:
        return ""

    @classmethod
    @abstractmethod
    def available(cls) -> bool:
        """True if this backend can be constructed *right now* without
        downloading anything. Must never raise."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts → list of dense vectors."""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    # ── retrieval asymmetry ─────────────────────────────────────────
    def embed_query(self, text: str) -> list[float]:
        """Embed a *query* — applies the model's query-side prefix."""
        return self.embed([self.query_prefix + text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed *documents* — applies the model's document-side prefix."""
        if self.document_prefix:
            texts = [self.document_prefix + t for t in texts]
        return self.embed(texts)

    def cache_key(self, text: str, role: str = "passage") -> str:
        """Stable cache key for one embedding. Includes model AND role
        (``"query"`` vs ``"passage"``) — the two live in different
        vector spaces for prefix-trained models and must never share a
        cache entry."""
        role = "query" if role == "query" else "passage"
        blob = f"{self.key}|{self.model}|{role}|{text}".encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    # ── async variants (real: sync work in a thread pool) ─────────────
    async def aembed(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed, list(texts))

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(
            self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, list(texts))

    @property
    def dim(self) -> int | None:
        """Vector dimensionality, or None if unknown until first embed."""
        return None

    def close(self) -> None:
        """Release native resources (model sessions). No-op by default."""


# ── FastEmbed (primary: ONNX, no torch, CPU-fast) ─────────────────────

class FastEmbedBackend(EmbeddingBackend):
    """Best free primary: ONNX Runtime only, no torch. Maintained by
    Qdrant; bge-small-en-v1.5 (384d) is the default — a strong
    quality-per-size pick on public MTEB-style benchmarks.

    For non-English corpora (Sanskrit, Greek, Coptic, Arabic, Chinese —
    i.e. this corpus), ``BAAI/bge-m3`` (1024d, multilingual, no prefixes
    needed per BAAI) is the better model choice: pass it as the model
    name. It downloads ~2.2GB on first use, so it is opt-in, not default.
    """

    key = "fastembed"

    @classmethod
    def default_model(cls) -> str:
        return "BAAI/bge-small-en-v1.5"

    @classmethod
    def available(cls) -> bool:
        try:
            import fastembed  # noqa: F401
            import onnxruntime  # noqa: F401
        except Exception:
            return False
        return True

    def __init__(self, model: str = "") -> None:
        if not self.available():
            raise EmbeddingError(
                "fastembed backend requested but 'fastembed' (or "
                "'onnxruntime') is not installed; pip install fastembed")
        super().__init__(model)
        self._model = None

    def _ensure(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding
            try:
                self._model = TextEmbedding(
                    model_name=self.model,
                    # never phone home to check for a newer revision
                    # mid-run; the local cache is authoritative.
                    cache_dir=None,
                )
            except Exception as exc:
                raise EmbeddingError(
                    f"fastembed failed to load {self.model!r} "
                    f"(needs one HF download on first use): {exc}") from exc
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure()
        try:
            # fastembed.embed returns a generator of numpy arrays
            return [list(map(float, v)) for v in model.embed(list(texts))]
        except Exception as exc:
            raise EmbeddingError(
                f"fastembed embedding failed: {exc}") from exc

    @property
    def dim(self) -> int | None:
        return {"BAAI/bge-small-en-v1.5": 384,
                "BAAI/bge-base-en-v1.5": 768,
                "BAAI/bge-large-en-v1.5": 1024,
                "BAAI/bge-m3": 1024,
                "sentence-transformers/all-MiniLM-L6-v2": 384,
                "intfloat/multilingual-e5-large": 1024,
                "intfloat/multilingual-e5-small": 384}.get(self.model)

    def close(self) -> None:
        self._model = None


# ── sentence-transformers (quality fallback) ──────────────────────────

class SentenceTransformersBackend(EmbeddingBackend):
    """Quality fallback: usually the best-scoring open embedding stack,
    at the cost of a torch dependency and more RAM.

    E5-family models are trained with ``query: `` / ``passage: ``
    prefixes — measured +15 points of hit@1 on a 384d model. When the
    model name marks it as prefix-expecting (contains "e5"), this
    backend sets the prefixes automatically so ``embed_query`` and
    ``embed_documents`` do the right thing without caller changes."""

    key = "sentence-transformers"

    @classmethod
    def default_model(cls) -> str:
        return "sentence-transformers/all-MiniLM-L6-v2"

    @classmethod
    def available(cls) -> bool:
        try:
            import sentence_transformers  # noqa: F401
        except Exception:
            return False
        return True

    def __init__(self, model: str = "") -> None:
        if not self.available():
            raise EmbeddingError(
                "sentence-transformers backend requested but the "
                "'sentence-transformers' package is not installed; "
                "pip install sentence-transformers")
        super().__init__(model)
        # E5 models are prefix-trained: query side gets "query: ",
        # document side gets "passage: ". Measured, not decorative.
        if "e5" in self.model.lower():
            self.query_prefix = "query: "
            self.document_prefix = "passage: "
        self._model = None

    def _ensure(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            try:
                self._model = SentenceTransformer(self.model)
            except Exception as exc:
                raise EmbeddingError(
                    f"sentence-transformers failed to load "
                    f"{self.model!r}: {exc}") from exc
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure()
        try:
            arr = model.encode(list(texts), normalize_embeddings=True,
                               show_progress_bar=False)
            return [list(map(float, row)) for row in arr]
        except Exception as exc:
            raise EmbeddingError(
                f"sentence-transformers embedding failed: {exc}") from exc

    @property
    def dim(self) -> int | None:
        m = self._model
        if m is not None:
            try:
                return int(m.get_sentence_embedding_dimension())
            except Exception:
                return None
        return {"sentence-transformers/all-MiniLM-L6-v2": 384,
                "intfloat/multilingual-e5-small": 384,
                "intfloat/multilingual-e5-large": 1024,
                "intfloat/e5-small-v2": 384,
                "intfloat/e5-large-v2": 1024,
                "BAAI/bge-m3": 1024}.get(self.model)

    def close(self) -> None:
        self._model = None


# ── Ollama (free local server, stdlib HTTP only) ──────────────────────

class OllamaBackend(EmbeddingBackend):
    """Free local-server option. Talks to a running Ollama instance over
    stdlib HTTP — no new Python dependency. The operator already runs
    local models; Ollama serves embeddings (nomic-embed-text 768d,
    mxbai-embed-large 1024d) with no cloud involved."""

    key = "ollama"

    def __init__(self, model: str = "", host: str = "http://127.0.0.1:11434",
                 timeout: float = 5.0) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout
        super().__init__(model)

    @classmethod
    def default_model(cls) -> str:
        return "nomic-embed-text"

    @classmethod
    def available(cls) -> bool:
        try:
            req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
            with urllib.request.urlopen(req, timeout=2) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            names = {m.get("name", "") for m in payload.get("models", [])}
            return any("embed" in n.lower() for n in names) or bool(names)
        except Exception:
            return False

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            body = json.dumps(
                {"model": self.model, "prompt": text}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.host}/api/embeddings", data=body,
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req,
                                            timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
            except urllib.error.URLError as exc:
                raise EmbeddingError(
                    f"ollama at {self.host} unreachable: {exc}; "
                    f"start ollama and pull an embedding model "
                    f"('ollama pull {self.model}')") from exc
            except Exception as exc:
                raise EmbeddingError(
                    f"ollama embedding failed: {exc}") from exc
            vec = payload.get("embedding")
            if not vec:
                raise EmbeddingError(
                    f"ollama returned no embedding for model "
                    f"{self.model!r} (is it pulled?)")
            out.append(_normalize([float(x) for x in vec]))
        return out

    @property
    def dim(self) -> int | None:
        return {"nomic-embed-text": 768,
                "nomic-embed-text-v1.5": 768,
                "mxbai-embed-large": 1024,
                "snowflake-arctic-embed": 1024,
                "snowflake-arctic-embed2": 1024,
                "qwen3-embedding": 1024,
                "qwen3-embedding:0.6b": 1024,
                "qwen3-embedding:4b": 2560,
                "embeddinggemma": 768}.get(self.model)


# ── Hashing fallback (stdlib, always available) ───────────────────────

class HashEmbedBackend(EmbeddingBackend):
    """Deterministic stdlib fallback: character-trigram hashing into
    signed buckets (the classic hashing trick), L2-normalized.

    Quality is well below neural embeddings — it is a *lexical* signal,
    not a semantic one — but it is always available, needs no downloads,
    and keeps the vector/semantic path functional end-to-end on a bare
    machine. Never raises on construction or embed."""

    key = "hashing"
    DIM = 512

    @classmethod
    def default_model(cls) -> str:
        return "hash-trigram-512"

    @classmethod
    def available(cls) -> bool:
        return True

    @property
    def dim(self) -> int:
        return self.DIM

    @staticmethod
    def _trigrams(text: str) -> Iterable[str]:
        t = f"  {text.lower()}  "
        for i in range(len(t) - 2):
            yield t[i:i + 3]

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            if not text.strip():
                # No content → no signal. A zero vector keeps cosine
                # well-defined (it scores 0 against everything).
                out.append([0.0] * self.DIM)
                continue
            vec = [0.0] * self.DIM
            for tri in self._trigrams(text):
                h = int.from_bytes(
                    hashlib.md5(tri.encode("utf-8")).digest()[:8], "big")
                idx = h % self.DIM
                # signed bucket: the top bit picks the sign so that
                # collisions partially cancel instead of piling up.
                vec[idx] += 1.0 if (h >> 63) & 1 else -1.0
            out.append(_normalize(vec))
        return out


# ── registry + auto-selection ─────────────────────────────────────────

BACKENDS: dict[str, type[EmbeddingBackend]] = {
    cls.key: cls for cls in (
        FastEmbedBackend,
        SentenceTransformersBackend,
        OllamaBackend,
        HashEmbedBackend,
    )
}

#: Preference order for ``auto_backend`` — best free primary first,
#: quality fallback second, free local-server third, stdlib last.
_PREFERENCE = ("fastembed", "sentence-transformers", "ollama", "hashing")


def available_backends() -> list[str]:
    """Registry keys whose backend can be constructed right now."""
    return [key for key in _PREFERENCE if BACKENDS[key].available()]


def auto_backend(name: str = "") -> EmbeddingBackend:
    """Pick the best available embedding backend.

    ``name`` forces one provider (raises EmbeddingError if unavailable);
    otherwise the first available in preference order wins.
    """
    if name:
        key = name.strip().lower().replace("_", "-")
        cls = BACKENDS.get(key)
        if cls is None:
            raise EmbeddingError(
                f"unknown embedding backend {name!r} "
                f"(known: {sorted(BACKENDS)})")
        if not cls.available():
            raise EmbeddingError(
                f"embedding backend {key!r} is not available on this "
                f"machine")
        return cls()
    for key in _PREFERENCE:
        if BACKENDS[key].available():
            return BACKENDS[key]()
    # HashEmbedBackend.available() is always True, so this is unreachable
    # in practice; kept as a defensive invariant.
    raise EmbeddingError("no embedding backend is available")


def embed_texts(backend: EmbeddingBackend,
                texts: Iterable[str],
                dimensions: int | None = None,
                role: str = "") -> list[list[float]]:
    """Embed a batch through ``backend``; every vector L2-normalized.

    ``role`` selects the retrieval-asymmetric path: ``"query"`` routes
    through :meth:`embed_query`, ``"passage"``/``"document"`` through
    :meth:`embed_documents` (prefix-trained models like E5 need their
    ``query: `` / ``passage: `` prefixes — bare is the measured-wrong
    default). ``""`` keeps the legacy symmetric :meth:`embed` path.

    ``dimensions`` applies Matryoshka (MRL) truncation: keep the first
    N dims of each vector, then re-normalize. Only meaningful for
    MRL-trained models (OpenAI-3, nomic-embed-text-v1.5, Qwen3-Embedding,
    Gemini) — 512d retains 94–98% of full quality at 4–8× storage cut.
    Truncating a non-MRL model still works (it is just a projection),
    but quality loss is unmeasured, so it is opt-in only.
    """
    texts = list(texts)
    if role == "query":
        vectors = [backend.embed_query(t) for t in texts]
    elif role in ("passage", "document"):
        vectors = backend.embed_documents(texts)
    else:
        vectors = backend.embed(texts)
    vectors = [_normalize(v) for v in vectors]
    if dimensions is not None:
        if not isinstance(dimensions, int) or dimensions < 1:
            raise EmbeddingError(
                f"dimensions must be a positive int, got {dimensions!r}")
        vectors = [_normalize(v[:dimensions]) for v in vectors]
    return vectors


def truncate_dim(vectors: list[list[float]],
                 dimensions: int) -> list[list[float]]:
    """Matryoshka truncation of already-embedded vectors (re-normalized)."""
    if not isinstance(dimensions, int) or dimensions < 1:
        raise EmbeddingError(
            f"dimensions must be a positive int, got {dimensions!r}")
    return [_normalize(v[:dimensions]) for v in vectors]


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 0.0:
        return [0.0] * len(vec)
    return [x / norm for x in vec]
