"""Vector similarity search on SQLite.

Two backends behind one API:

* **pure Python** — ``array('f')`` storage and a scalar dot product. No
  dependencies. Fine up to ~100k vectors, which is a lot of personal memory.
* **numpy** — vectorized cosine over a matrix loaded once and cached. ~30x faster.

On top of either sits an optional **coarse quantizer** (IVF-lite): vectors are
bucketed by nearest centroid and a search only scans ``nprobe`` buckets. That is
what makes a million-vector index tractable without an external vector database.

Vectors are stored as little-endian float32 blobs, pre-normalized, so cosine
similarity reduces to a dot product.
"""

from __future__ import annotations

import array
import math
import random
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..compat import available, load_optional
from ..core.errors import StorageError, ValidationError
from ..core.ids import new_id
from ..core.logging_setup import get_logger
from .db import Database

__all__ = ["VectorRecord", "VectorSearchHit", "VectorStore", "cosine", "normalize"]

_log = get_logger(__name__)

np = load_optional("numpy")


def normalize(vector: Sequence[float]) -> list[float]:
    """L2-normalize. A zero vector is returned unchanged."""
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return list(vector)
    return [v / norm for v in vector]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors."""
    if len(a) != len(b):
        raise ValidationError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _pack(vector: Sequence[float]) -> bytes:
    return array.array("f", (float(v) for v in vector)).tobytes()


def _unpack(blob: bytes, dim: int) -> list[float]:
    data = array.array("f")
    data.frombytes(blob)
    if len(data) != dim:
        raise StorageError(f"stored vector has {len(data)} dims, expected {dim}")
    return list(data)


@dataclass
class VectorRecord:
    id: str
    owner_type: str
    owner_id: str
    model: str
    dim: int
    vector: list[float]
    created_at: float = 0.0


@dataclass
class VectorSearchHit:
    """A neighbour found by search."""

    id: str
    owner_type: str
    owner_id: str
    score: float
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "owner_type": self.owner_type,
            "owner_id": self.owner_id,
            "score": self.score,
            "model": self.model,
        }


@dataclass
class _Centroid:
    vector: list[float]
    members: list[int] = field(default_factory=list)


class VectorStore:
    """Similarity index over the ``embeddings`` table."""

    TABLE = "embeddings"

    def __init__(
        self,
        db: Database,
        *,
        table: str = TABLE,
        model: str = "default",
        use_numpy: bool = True,
    ) -> None:
        self.db = db
        self.table = table
        self.model = model
        self._numpy = np if (use_numpy and available("numpy")) else None
        self._matrix: Any = None
        self._meta: list[dict[str, Any]] = []
        self._centroids: list[_Centroid] = []
        self._index_dirty = True
        self.stats = {"puts": 0, "searches": 0, "search_seconds": 0.0}

    # ── writes ───────────────────────────────────────────────────────────────
    def put(
        self,
        vector: Sequence[float],
        *,
        owner_type: str,
        owner_id: str,
        model: str | None = None,
        normalize_vectors: bool = True,
    ) -> str:
        if not vector:
            raise ValidationError("cannot store an empty vector")
        values = normalize(vector) if normalize_vectors else list(vector)
        record_id = new_id()
        norm = math.sqrt(sum(v * v for v in values))
        self.db.insert(
            self.table,
            {
                "id": record_id,
                "owner_type": owner_type,
                "owner_id": owner_id,
                "model": model or self.model,
                "dim": len(values),
                "norm": norm,
                "vector": _pack(values),
                "created_at": time.time(),
            },
        )
        self.stats["puts"] += 1
        self._index_dirty = True
        return record_id

    def put_many(
        self,
        items: Iterable[tuple[Sequence[float], str, str]],
        *,
        model: str | None = None,
    ) -> list[str]:
        """Bulk insert as ``(vector, owner_type, owner_id)`` tuples."""
        ids: list[str] = []
        rows = []
        now = time.time()
        for vector, owner_type, owner_id in items:
            values = normalize(vector)
            record_id = new_id()
            ids.append(record_id)
            rows.append(
                (
                    record_id,
                    owner_type,
                    owner_id,
                    model or self.model,
                    len(values),
                    math.sqrt(sum(v * v for v in values)),
                    _pack(values),
                    now,
                )
            )
        if rows:
            with self.db.transaction():
                self.db.executemany(
                    f'INSERT INTO "{self.table}" (id, owner_type, owner_id, model, dim, norm, vector, created_at) '
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
            self.stats["puts"] += len(rows)
            self._index_dirty = True
        return ids

    def delete(self, record_id: str) -> int:
        removed = self.db.delete(self.table, "id = ?", (record_id,))
        self._index_dirty = True
        return removed

    def delete_owner(self, owner_type: str, owner_id: str) -> int:
        removed = self.db.delete(self.table, "owner_type = ? AND owner_id = ?", (owner_type, owner_id))
        self._index_dirty = True
        return removed

    def get(self, record_id: str) -> VectorRecord | None:
        row = self.db.query_one(f'SELECT * FROM "{self.table}" WHERE id = ?', (record_id,))
        if row is None:
            return None
        return VectorRecord(
            id=row["id"],
            owner_type=row["owner_type"],
            owner_id=row["owner_id"],
            model=row["model"],
            dim=row["dim"],
            vector=_unpack(row["vector"], row["dim"]),
            created_at=row["created_at"],
        )

    def count(self) -> int:
        return int(self.db.scalar(f'SELECT COUNT(*) FROM "{self.table}"', default=0))

    # ── index construction ───────────────────────────────────────────────────
    def _load(self) -> None:
        if not self._index_dirty and self._matrix is not None:
            return
        rows = self.db.query(f'SELECT * FROM "{self.table}"')
        self._meta = [
            {
                "id": r["id"],
                "owner_type": r["owner_type"],
                "owner_id": r["owner_id"],
                "model": r["model"],
                "dim": r["dim"],
            }
            for r in rows
        ]
        vectors = [_unpack(r["vector"], r["dim"]) for r in rows]
        if self._numpy is not None and vectors:
            self._matrix = self._numpy.asarray(vectors, dtype="float32")
        else:
            self._matrix = vectors
        self._index_dirty = False

    def build_index(self, clusters: int = 0, *, seed: int = 1234) -> int:
        """Build the coarse quantizer. Returns the number of centroids.

        ``clusters=0`` auto-selects ``sqrt(n)`` centroids, capped at 256. Below
        512 vectors no index is built — brute force is faster than the bookkeeping.
        """
        self._load()
        total = len(self._meta)
        if total < 512:
            self._centroids = []
            return 0
        k = clusters or max(1, min(256, int(math.sqrt(total))))
        k = min(k, total)
        rng = random.Random(seed)
        vectors = self._as_lists()
        self._centroids = [
            _Centroid(vector=list(vectors[rng.randrange(total)])) for _ in range(k)
        ]
        for index, vector in enumerate(vectors):
            best, best_score = 0, -math.inf
            for ci, centroid in enumerate(self._centroids):
                score = _dot(vector, centroid.vector)
                if score > best_score:
                    best, best_score = ci, score
            self._centroids[best].members.append(index)
        _log.debug("vector index: %d vectors in %d clusters", total, k)
        return k

    def _as_lists(self) -> list[list[float]]:
        if self._numpy is not None and not isinstance(self._matrix, list):
            return [list(map(float, row)) for row in self._matrix]
        return list(self._matrix or [])

    # ── search ───────────────────────────────────────────────────────────────
    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int = 10,
        owner_type: str | None = None,
        min_score: float = -1.0,
        nprobe: int = 0,
    ) -> list[VectorSearchHit]:
        """Return the ``limit`` most similar stored vectors.

        ``nprobe > 0`` restricts the scan to that many quantizer buckets. It is
        ignored when no index has been built.
        """
        if limit <= 0:
            return []
        started = time.perf_counter()
        self._load()
        total = len(self._meta)
        self.stats["searches"] += 1
        if total == 0:
            return []

        query = normalize(vector)
        candidates: Sequence[int]
        if nprobe > 0 and self._centroids:
            ranked = sorted(
                range(len(self._centroids)),
                key=lambda ci: -_dot(query, self._centroids[ci].vector),
            )[:nprobe]
            bucket: list[int] = []
            for ci in ranked:
                bucket.extend(self._centroids[ci].members)
            candidates = bucket
        else:
            candidates = range(total)

        scored: list[tuple[float, int]] = []
        if self._numpy is not None and not isinstance(self._matrix, list):
            idx = list(candidates)
            if not idx:
                return []
            sub = self._matrix[idx]
            sims = sub @ self._numpy.asarray(query, dtype="float32")
            for position, score in enumerate(sims.tolist()):
                scored.append((float(score), idx[position]))
        else:
            lists = self._as_lists()
            for i in candidates:
                scored.append((_dot(query, lists[i]), i))

        scored.sort(key=lambda pair: -pair[0])
        hits: list[VectorSearchHit] = []
        for score, index in scored:
            if score < min_score:
                continue
            meta = self._meta[index]
            if owner_type is not None and meta["owner_type"] != owner_type:
                continue
            hits.append(
                VectorSearchHit(
                    id=meta["id"],
                    owner_type=meta["owner_type"],
                    owner_id=meta["owner_id"],
                    score=score,
                    model=meta["model"],
                )
            )
            if len(hits) >= limit:
                break
        self.stats["search_seconds"] += time.perf_counter() - started
        return hits

    def search_by_owner(
        self, owner_type: str, owner_id: str, **kwargs: Any
    ) -> list[VectorSearchHit]:
        """Nearest neighbours restricted to one owner."""
        record = self.db.query_one(
            f'SELECT * FROM "{self.table}" WHERE owner_type = ? AND owner_id = ? ORDER BY created_at DESC LIMIT 1',
            (owner_type, owner_id),
        )
        if record is None:
            return []
        vector = _unpack(record["vector"], record["dim"])
        return self.search(vector, **kwargs)

    def invalidate(self) -> None:
        """Force the in-memory matrix to reload on the next search."""
        self._index_dirty = True
        self._matrix = None
        self._meta = []

    def stats_snapshot(self) -> dict[str, Any]:
        return {
            **self.stats,
            "vectors": self.count(),
            "clusters": len(self._centroids),
            "backend": "numpy" if self._numpy is not None else "pure-python",
        }


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def topk_native(query: Sequence[float], candidates: list[Sequence[float]], k: int = 10) -> list[int]:
    """Stub: native top-k search (removed, use VectorStore.search instead)."""
    # Pure Python fallback
    scores = [(i, _dot(query, c)) for i, c in enumerate(candidates)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in scores[:k]]
