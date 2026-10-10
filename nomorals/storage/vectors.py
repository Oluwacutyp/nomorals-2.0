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
from ..core.style import active_theme, header, kv_lines
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
        use_native: bool = False,
    ) -> None:
        self.db = db
        self.table = table
        self.model = model
        self._numpy = np if (use_numpy and available("numpy")) else None
        #: C++ vecsim kernel: the phone's hot path.  Explicit opt-in because
        #: it scores in float32 (agreement with the float64 paths is to ~5
        #: decimal places — every parity test this ships with checks that).
        self._native = False
        self._native_flat: Any = None
        if use_native:
            try:
                from .. import native as _native
                self._native = bool(_native.available())
            except Exception:  # noqa: BLE001 — no extension is not an error
                self._native = False
        self._matrix: Any = None
        self._dim_set: frozenset[int] = frozenset()
        self._meta: list[dict[str, Any]] = []
        self._centroids: list[_Centroid] = []
        #: Vector count the trained index covers. Writes after training make
        #: the IVF buckets stale, so search() falls back to brute force when
        #: the table drifts — an index must never silently return partial
        #: results.
        self._trained_count: int | None = None
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
        self._native_flat = None  # matrix changed — repack lazily
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
        # Fail fast on dimension drift HERE, before numpy sees the rows:
        # ``asarray`` on ragged vectors raises a raw ValueError whose message
        # says nothing about dimensions, while the documented contract is the
        # catchable StorageError below. One rule for all backends.
        self._dim_set = frozenset(r["dim"] for r in rows)
        if len(self._dim_set) > 1:
            raise StorageError(
                f"index mixes vector dimensions {sorted(self._dim_set)}; rebuild required"
            )
        if self._numpy is not None and vectors:
            self._matrix = self._numpy.asarray(vectors, dtype="float32")
        else:
            self._matrix = vectors
        self._index_dirty = False

    def build_index(self, clusters: int = 0, *, seed: int = 1234,
                    iterations: int = 10) -> int:
        """Train the coarse quantizer with k-means (FAISS ``index.train``).

        ``clusters=0`` auto-selects ``sqrt(n)`` centroids, capped at 256. Below
        512 vectors no index is built — brute force is faster than the
        bookkeeping. Seeding is k-means++ (not uniform random), then Lloyd's
        algorithm refines for ``iterations`` passes; empty clusters are
        reseeded from the worst-fit vector. The trained index is persisted to
        ``<table>_index`` so a restart doesn't retrain.
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
        centroids = self._kmeans_plus_plus(vectors, k, rng)
        members: list[list[int]] = [[] for _ in range(k)]
        for _ in range(max(1, iterations)):
            members = [[] for _ in range(k)]
            self._assign(vectors, centroids, members)
            moved = self._recompute(vectors, centroids, members, rng)
            if moved == 0:
                break
        self._centroids = [
            _Centroid(vector=centroids[i], members=members[i]) for i in range(k)
        ]
        self._trained_count = total
        self._save_index()
        _log.debug("vector index: %d vectors in %d clusters", total, k)
        return k

    # ── k-means internals ────────────────────────────────────────────────
    @staticmethod
    def _kmeans_plus_plus(
        vectors: list[list[float]], k: int, rng: random.Random
    ) -> list[list[float]]:
        """k-means++ seeding: each new centroid is sampled with probability
        proportional to its squared distance from the nearest existing one."""
        first = rng.randrange(len(vectors))
        centroids = [list(vectors[first])]
        closest_sq = [float("inf")] * len(vectors)
        for _ in range(1, k):
            newest = centroids[-1]
            total = 0.0
            for i, vector in enumerate(vectors):
                dist_sq = sum(
                    (a - b) ** 2 for a, b in zip(vector, newest)
                )
                if dist_sq < closest_sq[i]:
                    closest_sq[i] = dist_sq
                total += closest_sq[i]
            if total <= 0.0:  # all vectors identical; duplicate the first
                centroids.append(list(vectors[first]))
                continue
            pick = rng.random() * total
            chosen = 0
            for i, dist_sq in enumerate(closest_sq):
                pick -= dist_sq
                if pick <= 0:
                    chosen = i
                    break
            centroids.append(list(vectors[chosen]))
        return centroids

    def _assign(
        self,
        vectors: list[list[float]],
        centroids: list[list[float]],
        members: list[list[int]],
    ) -> None:
        """Assign every vector to its nearest centroid (cosine via dot)."""
        if self._numpy is not None:
            matrix = self._numpy.asarray(vectors, dtype="float32")
            cents = self._numpy.asarray(centroids, dtype="float32")
            sims = matrix @ cents.T
            best = sims.argmax(axis=1).tolist()
            for index, ci in enumerate(best):
                members[int(ci)].append(index)
            return
        for index, vector in enumerate(vectors):
            best, best_score = 0, -math.inf
            for ci, centroid in enumerate(centroids):
                score = _dot(vector, centroid)
                if score > best_score:
                    best, best_score = ci, score
            members[best].append(index)

    def _recompute(
        self,
        vectors: list[list[float]],
        centroids: list[list[float]],
        members: list[list[int]],
        rng: random.Random,
    ) -> int:
        """Move each centroid to its members' mean (renormalized). Returns
        the number of centroids that moved; reseeds empty ones."""
        dim = len(centroids[0])
        moved = 0
        # Worst-fit vector for empty-cluster reseeding.
        worst_index, worst_score = 0, math.inf
        for index, vector in enumerate(vectors):
            score = max(_dot(vector, c) for c in centroids)
            if score < worst_score:
                worst_index, worst_score = index, score
        for ci in range(len(centroids)):
            cluster = members[ci]
            if not cluster:
                centroids[ci] = list(vectors[worst_index])
                moved += 1
                continue
            mean = [0.0] * dim
            for index in cluster:
                vector = vectors[index]
                for d in range(dim):
                    mean[d] += vector[d]
            count = len(cluster)
            mean = [v / count for v in mean]
            norm = math.sqrt(sum(v * v for v in mean))
            if norm > 0:
                mean = [v / norm for v in mean]
            if any(abs(a - b) > 1e-9 for a, b in zip(mean, centroids[ci])):
                moved += 1
            centroids[ci] = mean
        return moved

    # ── index persistence ────────────────────────────────────────────────
    @property
    def _index_table(self) -> str:
        return f"{self.table}_index"

    def _save_index(self) -> None:
        """Persist centroids + member lists so restarts skip retraining."""
        import json

        table = self._index_table
        with self.db.transaction():
            self.db.execute(
                f'CREATE TABLE IF NOT EXISTS "{table}" ('
                "centroid_id INTEGER PRIMARY KEY, "
                "dim INTEGER NOT NULL, "
                "vector BLOB NOT NULL, "
                "members TEXT NOT NULL)"
            )
            self.db.execute(f'DELETE FROM "{table}"')
            rows = [
                (ci, len(c.vector), _pack(c.vector), json.dumps(c.members))
                for ci, c in enumerate(self._centroids)
            ]
            if rows:
                self.db.executemany(
                    f'INSERT INTO "{table}" (centroid_id, dim, vector, members) '
                    "VALUES (?, ?, ?, ?)",
                    rows,
                )

    def load_index(self) -> int:
        """Load a previously trained index. Returns centroid count (0 if none)."""
        import json

        table = self._index_table
        if not self.db.table_exists(table):
            return 0
        self._load()
        rows = self.db.query(f'SELECT * FROM "{table}" ORDER BY centroid_id')
        centroids: list[_Centroid] = []
        for row in rows:
            vector = _unpack(bytes(row["vector"]), int(row["dim"]))
            try:
                members = [int(m) for m in json.loads(row["members"])]
            except (ValueError, TypeError):
                members = []
            centroids.append(_Centroid(vector=vector, members=members))
        # A trained index is only valid for the exact vector set it was
        # trained on; row-count drift means retrain.
        if centroids and sum(len(c.members) for c in centroids) != len(self._meta):
            _log.warning(
                "vector index covers %d vectors but table has %d; ignoring stale index",
                sum(len(c.members) for c in centroids), len(self._meta),
            )
            return 0
        self._centroids = centroids
        self._trained_count = len(self._meta)
        return len(centroids)

    def drop_index(self) -> None:
        """Delete the trained index (memory and persisted)."""
        self._centroids = []
        self.db.execute(f'DROP TABLE IF EXISTS "{self._index_table}"')

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
        owner_ids: set[str] | None = None,
        min_score: float = -1.0,
        nprobe: int = 0,
        score_mode: str = "similarity",
    ) -> list[VectorSearchHit]:
        """Return the ``limit`` most similar stored vectors.

        ``nprobe > 0`` restricts the scan to that many quantizer buckets. It is
        ignored when no index has been built, and clamped to the trained
        cluster count (nprobe can never exceed nlist). ``score_mode`` is
        ``"similarity"`` (cosine, higher is better) or ``"distance"``
        (``1 - cosine``, lower is better — ``min_score`` then acts as a
        maximum-distance threshold).
        """
        if limit <= 0:
            return []
        if score_mode not in {"similarity", "distance"}:
            raise ValidationError(
                f"score_mode must be 'similarity' or 'distance', got {score_mode!r}"
            )
        started = time.perf_counter()
        self._load()
        total = len(self._meta)
        self.stats["searches"] += 1
        if total == 0:
            return []

        query = normalize(vector)
        # Dimension checks run against the cached dim set from _load() — no
        # O(n) set rebuild per search. Mixed dims already raised in _load().
        dims = self._dim_set
        if dims and len(query) != next(iter(dims)):
            raise ValidationError(
                f"query dimension {len(query)} does not match stored dimension {next(iter(dims))}"
            )
        candidates: Sequence[int]
        index_valid = (
            bool(self._centroids)
            and self._trained_count is not None
            and self._trained_count == total
        )
        if nprobe > 0 and index_valid:
            probed = min(nprobe, len(self._centroids))
            if probed < nprobe:
                _log.debug("nprobe %d clamped to %d trained clusters",
                           nprobe, len(self._centroids))
            ranked = sorted(
                range(len(self._centroids)),
                key=lambda ci: -_dot(query, self._centroids[ci].vector),
            )[:probed]
            bucket: list[int] = []
            for ci in ranked:
                bucket.extend(self._centroids[ci].members)
            candidates = bucket
        else:
            candidates = range(total)

        scored: list[tuple[float, int]] = []
        if self._native:
            idx = list(candidates)
            if idx:
                from .. import native as _native
                if len(idx) == total and idx[0] == 0:
                    if self._native_flat is None:
                        from array import array as _array
                        packed = _array("f")
                        for row in self._as_lists():
                            packed.extend(row)
                        self._native_flat = packed
                    pairs = _native.topk(self._native_flat, query, total)
                    scored = [(float(s), int(r)) for s, r in pairs]
                else:
                    lists = self._as_lists()
                    matrix = [lists[i] for i in idx]
                    pairs = _native.topk(matrix, query, len(matrix))
                    scored = [(float(s), idx[int(r)]) for s, r in pairs]
        elif self._numpy is not None and not isinstance(self._matrix, list):
            idx = list(candidates)
            if not idx:
                return []
            # The common case scans the whole index: fancy-indexing
            # ``self._matrix[idx]`` copies the entire matrix (~14MB per query
            # at 7k×512) for zero benefit. Score against it in place instead —
            # identical scores, no copy.
            if len(idx) == total and idx[0] == 0 and idx[-1] == total - 1:
                sims = self._matrix @ self._numpy.asarray(query, dtype="float32")
                for position, score in enumerate(sims.tolist()):
                    scored.append((float(score), position))
            else:
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
            meta = self._meta[index]
            if owner_type is not None and meta["owner_type"] != owner_type:
                continue
            if owner_ids is not None and meta["owner_id"] not in owner_ids:
                continue
            if score_mode == "distance":
                score = 1.0 - score
                if score > min_score and min_score >= 0:
                    continue
            elif score < min_score:
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

    def search_many(
        self,
        vectors: Sequence[Sequence[float]],
        *,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[list[VectorSearchHit]]:
        """Batch search: one ``_load()`` for many queries."""
        self._load()
        return [self.search(vector, limit=limit, **kwargs) for vector in vectors]

    def evaluate_recall(
        self,
        queries: Sequence[Sequence[float]],
        *,
        k: int = 10,
        nprobe: int = 0,
    ) -> dict[str, Any]:
        """Measure IVF recall@k: brute-force top-k vs ``nprobe`` top-k.

        The FAISS-benchmark workflow for tuning ``nprobe``: recall of 1.0
        means the index loses nothing; lower means speed is costing accuracy.
        ``queries`` are raw vectors; ground truth is computed by brute force.
        """
        if not queries:
            return {"queries": 0, "k": k, "nprobe": nprobe, "recall_at_k": 0.0}
        recalls: list[float] = []
        for vector in queries:
            truth = {h.id for h in self.search(vector, limit=k)}
            approx = {h.id for h in self.search(vector, limit=k, nprobe=nprobe)}
            recalls.append(len(truth & approx) / max(1, len(truth)))
        return {
            "queries": len(queries),
            "k": k,
            "nprobe": nprobe,
            "recall_at_k": sum(recalls) / len(recalls),
            "min_recall": min(recalls),
        }

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
            "backend": ("native-cpp" if self._native
                        else "numpy" if self._numpy is not None else "pure-python"),
        }

    def format_stats(self, theme: Any = None) -> str:
        """Human-readable index overview through the shared style layer."""
        theme = theme or active_theme()
        snap = self.stats_snapshot()
        avg_ms = (
            snap["search_seconds"] / snap["searches"] * 1000
            if snap["searches"] else 0.0
        )
        return "\n".join([
            header("vector index", theme=theme),
            *kv_lines(
                {
                    "table": self.table,
                    "model": self.model,
                    "backend": snap["backend"],
                    "vectors": snap["vectors"],
                    "clusters": snap["clusters"],
                    "searches": snap["searches"],
                    "avg search": f"{avg_ms:.2f}ms",
                    "puts": snap["puts"],
                },
                theme=theme,
            ),
        ])


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
