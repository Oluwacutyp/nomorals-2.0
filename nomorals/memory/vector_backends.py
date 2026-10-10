"""Pluggable vector-search backends for semantic recall.

Recall quality lives or dies on the vector substrate. Three backends behind
one contract (:class:`VectorBackend`), evaluated on quality and simplicity
rather than fame:

* **sqlite-vec** — the primary. ``vec0`` virtual tables live *inside* the
  existing SQLite database file: same WAL, same backup story, ACID with the
  memory rows, zero new servers and zero new file formats. Pure-C extension
  (~200-500KB), exact KNN with cosine/L2/L1 metrics, int8/binary quantization
  available. For personal-memory scale (thousands to low hundreds of thousands
  of vectors) exact search is the right index — ANN only pays past ~1M vectors
  or ~1k QPS. Needs the ``sqlite-vec`` pip package plus a Python build that
  allows extension loading.
* **usearch** — HNSW approximate index for the large-memory end. SIMD-tuned
  C++, Apache-2.0, persists as a sidecar file next to the database. Faster
  than FAISS-flat by an order of magnitude where it matters, at the cost of
  approximate (not exact) neighbours and a second file to back up.
* **legacy** — the existing :class:`nomorals.storage.vectors.VectorStore`
  (pure-Python / numpy / native-C++ paths). Always available, zero new
  dependencies. The honest fallback, not a strawman: it is also the referee —
  backend-parity tests compare every other backend against it.

Selection is fail-open by contract: ``auto`` picks the first available backend
in preference order (sqlite-vec → usearch → legacy). Asking for a named backend
that is not installed fails fast with the exact ``pip install`` command —
never a silent downgrade, because a silent downgrade is how you debug recall
quality for a week.

The keyword lane (SQLite FTS5, :class:`nomorals.storage.fts.FTSIndex`) is
untouched: semantic recall complements it, it does not replace it. Vectors
blur exact identifiers; FTS catches them.
"""

from __future__ import annotations

import math
import os
import struct
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from ..compat import load_optional
from ..core.errors import StorageError, ValidationError
from ..core.logging_setup import get_logger
from ..storage.db import Database
from ..storage.vectors import VectorStore

__all__ = [
    "BACKEND_NAMES",
    "LegacyStoreBackend",
    "SqliteVecBackend",
    "USearchBackend",
    "VectorBackend",
    "VectorHit",
    "available_backends",
    "select_vector_backend",
]

_log = get_logger(__name__)

_sqlite_vec = load_optional("sqlite_vec")
_usearch_index = load_optional("usearch.index")
_np = load_optional("numpy")


@dataclass
class VectorHit:
    """One neighbour returned by a vector backend."""

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
            "score": round(self.score, 6),
            "model": self.model,
        }


def _normalize(vector: Sequence[float]) -> list[float]:
    """L2-normalize. Empty vectors are a caller bug — fail fast."""
    values = [float(v) for v in vector]
    if not values:
        raise ValidationError("cannot store an empty vector")
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0.0:
        return values
    return [v / norm for v in values]


def _pack_f32(values: Sequence[float]) -> bytes:
    """sqlite-vec wire format for float32 vectors: raw little-endian floats."""
    return struct.pack(f"<{len(values)}f", *values)


class VectorBackend(ABC):
    """Contract every semantic-recall vector store honours.

    ``owner_id`` is the memory record id; ``owner_type`` is fixed to
    ``"memory"`` by convention so one database can host several vector users
    without their vectors colliding.
    """

    name: str = "backend"

    def __init__(self, db: Database, *, owner_type: str = "memory",
                 truncate_dims: int = 0) -> None:
        self._db = db
        self.owner_type = owner_type
        #: Matryoshka truncation (0 = off): store only the first N
        #: dimensions. Matryoshka-trained embedders (Qwen3-Embedding,
        #: jina-v3, BGE-M3) order dimensions by importance, so 1024→512
        #: halves storage at a small quality cost. Applies to writes and
        #: to query vectors, so mixed-dimension stores stay aligned.
        self.truncate_dims = max(0, int(truncate_dims))

    def _truncate(self, vector: Sequence[float]) -> list[float]:
        if self.truncate_dims > 0 and len(vector) > self.truncate_dims:
            return list(vector[: self.truncate_dims])
        return list(vector)

    # ── availability ─────────────────────────────────────────────────────
    @classmethod
    @abstractmethod
    def available(cls) -> tuple[bool, str]:
        """(is_available, human-readable reason). Never raises."""
        ...  # pragma: no cover - contract only

    @property
    def install_hint(self) -> str:
        return ""

    # ── writes ───────────────────────────────────────────────────────────
    @abstractmethod
    def put(self, vector: Sequence[float], owner_id: str) -> str:
        """Store one vector; returns the backend's record id."""
        ...  # pragma: no cover - contract only

    def put_many(self, items: Sequence[tuple[Sequence[float], str]]) -> list[str]:
        """Store several ``(vector, owner_id)`` pairs. Defaults to one-by-one."""
        return [self.put(vector, owner_id) for vector, owner_id in items]

    @abstractmethod
    def delete(self, record_id: str) -> int:
        """Delete by backend record id. Returns rows removed."""
        ...  # pragma: no cover - contract only

    @abstractmethod
    def delete_owner(self, owner_id: str) -> int:
        """Delete every vector belonging to ``owner_id``. Returns rows removed."""
        ...  # pragma: no cover - contract only

    # ── reads ────────────────────────────────────────────────────────────
    @abstractmethod
    def search(
        self, vector: Sequence[float], *, limit: int = 10, min_score: float = -1.0
    ) -> list[VectorHit]:
        """Top-``limit`` neighbours by cosine similarity, best first."""
        ...  # pragma: no cover - contract only

    @abstractmethod
    def count(self) -> int:
        """Number of vectors held for this backend's owner type."""
        ...  # pragma: no cover - contract only

    def stats_snapshot(self) -> dict[str, Any]:
        return {"backend": self.name}


# ── legacy: the existing VectorStore ──────────────────────────────────────────


class LegacyStoreBackend(VectorBackend):
    """The pre-existing :class:`VectorStore` (pure-Python / numpy / native).

    Always available, zero new dependencies. Wraps the caller's store instance
    when one is supplied so statistics stay coherent with existing introspection.
    """

    name = "legacy"

    def __init__(
        self,
        db: Database,
        *,
        owner_type: str = "memory",
        vectors: VectorStore | None = None,
        model: str = "default",
        truncate_dims: int = 0,
    ) -> None:
        super().__init__(db, owner_type=owner_type,
                         truncate_dims=truncate_dims)
        self._store = vectors if vectors is not None else VectorStore(db)
        self.model = model

    @classmethod
    def available(cls) -> tuple[bool, str]:
        return (True, "built in — no extra dependency")

    def put(self, vector: Sequence[float], owner_id: str) -> str:
        return self._store.put(
            self._truncate(vector), owner_type=self.owner_type,
            owner_id=str(owner_id), model=self.model
        )

    def put_many(self, items: Sequence[tuple[Sequence[float], str]]) -> list[str]:
        return self._store.put_many(
            [(self._truncate(vector), self.owner_type, str(owner_id))
             for vector, owner_id in items],
            model=self.model,
        )

    def delete(self, record_id: str) -> int:
        return self._store.delete(record_id)

    def delete_owner(self, owner_id: str) -> int:
        return self._store.delete_owner(self.owner_type, str(owner_id))

    def search(
        self, vector: Sequence[float], *, limit: int = 10, min_score: float = -1.0
    ) -> list[VectorHit]:
        hits = self._store.search(
            self._truncate(vector), limit=limit, owner_type=self.owner_type,
            min_score=min_score
        )
        return [
            VectorHit(
                id=h.id,
                owner_type=h.owner_type,
                owner_id=h.owner_id,
                score=h.score,
                model=h.model,
            )
            for h in hits
        ]

    def count(self) -> int:
        rows = self._db.query(
            f'SELECT COUNT(*) AS n FROM "{VectorStore.TABLE}" WHERE owner_type = ?',
            (self.owner_type,),
        )
        return int(rows[0]["n"]) if rows else 0

    def stats_snapshot(self) -> dict[str, Any]:
        return {"backend": self.name, **self._store.stats_snapshot()}


# ── sqlite-vec: the primary backend ──────────────────────────────────────────


def _probe_sqlite_vec() -> tuple[bool, str]:
    """Prove vec0 actually works on this interpreter. Never raises."""
    import sqlite3

    if _sqlite_vec is None:
        return (False, "sqlite-vec is not installed")
    conn = None
    try:
        conn = sqlite3.connect(":memory:")
        _sqlite_vec.load(conn)
        conn.execute("CREATE VIRTUAL TABLE _probe USING vec0(x float[4] distance_metric=cosine)")
        conn.execute("INSERT INTO _probe(rowid, x) VALUES (1, ?)", (_pack_f32([1.0, 0.0, 0.0, 0.0]),))
        rows = conn.execute(
            "SELECT distance FROM _probe WHERE x MATCH ? AND k = 1", (_pack_f32([1.0, 0.0, 0.0, 0.0]),)
        ).fetchall()
        if not rows:
            return (False, "vec0 probe query returned no rows")
        return (True, "sqlite-vec vec0 ready")
    except Exception as exc:  # noqa: BLE001 - probe must never raise; the reason is the product
        return (False, f"vec0 probe failed: {exc}")
    finally:
        if conn is not None:
            with suppress(Exception):  # noqa: BLE001 - teardown only
                conn.close()


class SqliteVecBackend(VectorBackend):
    """sqlite-vec ``vec0`` tables inside the existing database file.

    One virtual table per embedding dimension (``memory_vec_<dim>``), so a
    provider change that alters dimensions degrades into a fresh table rather
    than a corrupt index. Vectors are stored cosine-normalized; search returns
    ``1 - cosine_distance`` so scores agree with the legacy backend to ~1e-6.

    Extension loading is per-connection and :class:`Database` pools
    thread-local connections, so each connection is loaded once, lazily, the
    first time this backend touches it. This reaches ``db._connection()``
    because ``Database`` exposes no public raw-connection hook — the
    alternative (a second connection to the same file) cannot see ``:memory:``
    databases at all.
    """

    name = "sqlite-vec"
    _TABLE_PREFIX = "memory_vec_"

    def __init__(self, db: Database, *, owner_type: str = "memory", model: str = "default",
                 truncate_dims: int = 0) -> None:
        super().__init__(db, owner_type=owner_type, truncate_dims=truncate_dims)
        self.model = model
        self._lock = threading.RLock()
        # id()s of connections already carrying the extension. Database retains
        # every connection it opens, so ids stay stable for the db's lifetime.
        self._loaded_conns: set[int] = set()
        self.stats = {"puts": 0, "searches": 0, "search_seconds": 0.0}

    @classmethod
    def available(cls) -> tuple[bool, str]:
        return _probe_sqlite_vec()

    @property
    def install_hint(self) -> str:
        return "pip install sqlite-vec"

    # ── extension handling ───────────────────────────────────────────────
    def _ensure_loaded(self) -> None:
        ok, reason = type(self).available()
        if not ok:
            raise StorageError(f"sqlite-vec backend unavailable: {reason}. {self.install_hint}")
        conn = self._db._connection()  # noqa: SLF001 - no public raw-connection hook exists
        with self._lock:
            if id(conn) in self._loaded_conns:
                return
            try:
                _sqlite_vec.load(conn)
            except Exception as exc:  # noqa: BLE001 - surface as a storage error with context
                raise StorageError(f"could not load sqlite-vec extension: {exc}") from exc
            self._loaded_conns.add(id(conn))

    # ── table management ─────────────────────────────────────────────────
    def _table(self, dim: int) -> str:
        if dim <= 0:
            raise ValidationError(f"invalid vector dimension {dim}")
        name = f"{self._TABLE_PREFIX}{dim}"
        self._ensure_loaded()
        self._db.execute(
            f'CREATE VIRTUAL TABLE IF NOT EXISTS "{name}" USING vec0('
            f"owner_id TEXT PARTITION KEY, embedding float[{dim}] distance_metric=cosine)"
        )
        return name

    def _tables(self) -> list[tuple[str, int]]:
        """Existing (table, dim) pairs. Shadow tables (``*_chunks`` …) excluded."""
        rows = self._db.query(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE ?",
            (f"{self._TABLE_PREFIX}%",),
        )
        out: list[tuple[str, int]] = []
        for row in rows:
            suffix = row["name"][len(self._TABLE_PREFIX):]
            if suffix.isdigit():
                out.append((row["name"], int(suffix)))
        return out

    # ── writes ───────────────────────────────────────────────────────────
    def put(self, vector: Sequence[float], owner_id: str) -> str:
        values = _normalize(self._truncate(vector))
        table = self._table(len(values))
        cursor = self._db.execute(
            f'INSERT INTO "{table}" (owner_id, embedding) VALUES (?, ?)',
            (str(owner_id), _pack_f32(values)),
        )
        self.stats["puts"] += 1
        return str(cursor.lastrowid or 0)

    def put_many(self, items: Sequence[tuple[Sequence[float], str]]) -> list[str]:
        # vec0 has no UPSERT and executemany() cannot report per-row rowids,
        # so batch by dimension and insert row-by-row: correct beats clever.
        return [self.put(vector, owner_id) for vector, owner_id in items]

    def delete(self, record_id: str) -> int:
        self._ensure_loaded()
        try:
            rowid = int(record_id)
        except (TypeError, ValueError):
            return 0
        total = 0
        for table, _ in self._tables():
            cursor = self._db.execute(f'DELETE FROM "{table}" WHERE rowid = ?', (rowid,))
            total += cursor.rowcount
        return total

    def delete_owner(self, owner_id: str) -> int:
        self._ensure_loaded()
        total = 0
        for table, _ in self._tables():
            cursor = self._db.execute(
                f'DELETE FROM "{table}" WHERE owner_id = ?', (str(owner_id),)
            )
            total += cursor.rowcount
        return total

    # ── reads ────────────────────────────────────────────────────────────
    def search(
        self, vector: Sequence[float], *, limit: int = 10, min_score: float = -1.0
    ) -> list[VectorHit]:
        if limit <= 0:
            return []
        started = time.perf_counter()
        values = _normalize(self._truncate(vector))
        table = f"{self._TABLE_PREFIX}{len(values)}"
        self._ensure_loaded()
        if not self._db.table_exists(table):
            return []
        rows = self._db.query(
            f'SELECT rowid AS vid, owner_id, distance FROM "{table}" '
            "WHERE embedding MATCH ? AND k = ?",
            (_pack_f32(values), limit),
        )
        hits: list[VectorHit] = []
        for row in rows:
            score = max(-1.0, min(1.0, 1.0 - float(row["distance"])))
            if score < min_score:
                continue
            hits.append(
                VectorHit(
                    id=str(row["vid"]),
                    owner_type=self.owner_type,
                    owner_id=str(row["owner_id"]),
                    score=score,
                    model=self.model,
                )
            )
        self.stats["searches"] += 1
        self.stats["search_seconds"] += time.perf_counter() - started
        return hits

    def count(self) -> int:
        self._ensure_loaded()
        total = 0
        for table, _ in self._tables():
            total += int(self._db.scalar(f'SELECT COUNT(*) FROM "{table}"', default=0))
        return total

    def stats_snapshot(self) -> dict[str, Any]:
        return {
            **self.stats,
            "backend": self.name,
            "tables": [name for name, _ in self._tables()],
            "truncate_dims": self.truncate_dims,
        }


# ── usearch: HNSW for the large-memory end ────────────────────────────────────


def _probe_usearch() -> tuple[bool, str]:
    """Prove usearch actually works on this interpreter. Never raises."""
    if _usearch_index is None:
        return (False, "usearch is not installed")
    if _np is None:
        return (False, "usearch backend needs numpy")
    try:
        index = _usearch_index.Index(ndim=8, metric="ip", dtype="f32")
        vectors = _np.eye(8, dtype=_np.float32)[:2]
        index.add(_np.array([1, 2], dtype=_np.uint64), vectors)
        matches = index.search(vectors[:1], 2)
        keys = _np.atleast_2d(matches.keys)
        if keys.shape[1] == 0:
            return (False, "usearch probe search returned no matches")
        return (True, "usearch HNSW ready")
    except Exception as exc:  # noqa: BLE001 - probe must never raise; the reason is the product
        return (False, f"usearch probe failed: {exc}")


class USearchBackend(VectorBackend):
    """USearch HNSW index with a SQLite key→owner mapping table.

    Vectors live in the HNSW index (inner-product metric over L2-normalized
    vectors, which is cosine similarity); the ``memory_usearch_keys`` table
    maps the index's uint64 keys back to memory record ids, so the mapping
    survives restarts alongside the persisted index file. Approximate
    neighbours — exactness is what sqlite-vec is for.
    """

    name = "usearch"
    _KEYS_TABLE = "memory_usearch_keys"

    def __init__(
        self,
        db: Database,
        *,
        owner_type: str = "memory",
        model: str = "default",
        path: str | None = None,
        truncate_dims: int = 0,
    ) -> None:
        super().__init__(db, owner_type=owner_type, truncate_dims=truncate_dims)
        self.model = model
        if path is None and db.path is not None:
            path = str(db.path) + ".usearch"
        self._path = path
        self._indexes: dict[int, Any] = {}
        self._lock = threading.RLock()
        self.stats = {"puts": 0, "searches": 0, "search_seconds": 0.0}
        self._ensure_keys_table()

    @classmethod
    def available(cls) -> tuple[bool, str]:
        return _probe_usearch()

    @property
    def install_hint(self) -> str:
        return "pip install usearch"

    # ── index handling ───────────────────────────────────────────────────
    def _ensure_keys_table(self) -> None:
        self._db.execute(
            f'CREATE TABLE IF NOT EXISTS "{self._KEYS_TABLE}" ('
            "ukey INTEGER NOT NULL, dim INTEGER NOT NULL, "
            "owner_id TEXT NOT NULL, PRIMARY KEY (ukey, dim))"
        )

    def _index_file(self, dim: int) -> str | None:
        return f"{self._path}.{dim}" if self._path else None

    def _index_for(self, dim: int) -> Any:
        """Return the HNSW index for ``dim``, building/loading it on demand."""
        with self._lock:
            index = self._indexes.get(dim)
            if index is None:
                index = _usearch_index.Index(ndim=dim, metric="ip", dtype="f32")
                fpath = self._index_file(dim)
                if fpath and os.path.exists(fpath):
                    index.load(fpath)
                self._indexes[dim] = index
            return index

    def _save(self, dim: int) -> None:
        fpath = self._index_file(dim)
        if fpath:
            index = self._indexes.get(dim)
            if index is not None:
                index.save(fpath)

    def _next_keys(self, dim: int, n: int) -> list[int]:
        start = int(
            self._db.scalar(
                f'SELECT COALESCE(MAX(ukey), 0) FROM "{self._KEYS_TABLE}" WHERE dim = ?',
                (dim,),
                default=0,
            )
        )
        return list(range(start + 1, start + 1 + n))

    # ── writes ───────────────────────────────────────────────────────────
    def put(self, vector: Sequence[float], owner_id: str) -> str:
        return self.put_many([(vector, owner_id)])[0]

    def put_many(self, items: Sequence[tuple[Sequence[float], str]]) -> list[str]:
        materialized = [(v, oid) for v, oid in items]
        if not materialized:
            return []
        ids: list[str] = []
        by_dim: dict[int, list[tuple[list[float], str]]] = {}
        for vector, owner_id in materialized:
            values = _normalize(self._truncate(vector))
            by_dim.setdefault(len(values), []).append((values, str(owner_id)))
        for dim, rows in by_dim.items():
            index = self._index_for(dim)
            keys = self._next_keys(dim, len(rows))
            matrix = _np.asarray([r[0] for r in rows], dtype=_np.float32)
            index.add(_np.asarray(keys, dtype=_np.uint64), matrix)
            self._db.executemany(
                f'INSERT INTO "{self._KEYS_TABLE}" (ukey, dim, owner_id) VALUES (?, ?, ?)',
                [(key, dim, owner_id) for key, (_, owner_id) in zip(keys, rows, strict=True)],
            )
            self._save(dim)
            ids.extend(str(key) for key in keys)
            self.stats["puts"] += len(rows)
        return ids

    def delete(self, record_id: str) -> int:
        try:
            ukey = int(record_id)
        except (TypeError, ValueError):
            return 0
        rows = self._db.query(
            f'SELECT dim FROM "{self._KEYS_TABLE}" WHERE ukey = ?', (ukey,)
        )
        total = 0
        for row in rows:
            dim = int(row["dim"])
            index = self._index_for(dim)
            index.remove(_np.asarray([ukey], dtype=_np.uint64))
            self._db.execute(
                f'DELETE FROM "{self._KEYS_TABLE}" WHERE ukey = ? AND dim = ?', (ukey, dim)
            )
            self._save(dim)
            total += 1
        return total

    def delete_owner(self, owner_id: str) -> int:
        rows = self._db.query(
            f'SELECT ukey, dim FROM "{self._KEYS_TABLE}" WHERE owner_id = ?', (str(owner_id),)
        )
        by_dim: dict[int, list[int]] = {}
        for row in rows:
            by_dim.setdefault(int(row["dim"]), []).append(int(row["ukey"]))
        total = 0
        for dim, keys in by_dim.items():
            index = self._index_for(dim)
            index.remove(_np.asarray(keys, dtype=_np.uint64))
            self._db.execute(
                f'DELETE FROM "{self._KEYS_TABLE}" WHERE dim = ? AND owner_id = ?',
                (dim, str(owner_id)),
            )
            self._save(dim)
            total += len(keys)
        return total

    # ── reads ────────────────────────────────────────────────────────────
    def search(
        self, vector: Sequence[float], *, limit: int = 10, min_score: float = -1.0
    ) -> list[VectorHit]:
        if limit <= 0:
            return []
        started = time.perf_counter()
        values = _normalize(self._truncate(vector))
        dim = len(values)
        index = self._indexes.get(dim)
        if index is None:
            fpath = self._index_file(dim)
            if not (fpath and os.path.exists(fpath)):
                return []
            index = self._index_for(dim)
        size = len(index)
        if size == 0:
            return []
        query = _np.asarray([values], dtype=_np.float32)
        matches = index.search(query, min(limit, size))
        # usearch squeezes the batch dimension for a single query vector
        keys = [int(k) for k in _np.atleast_2d(matches.keys)[0].tolist()]
        distances = [float(d) for d in _np.atleast_2d(matches.distances)[0].tolist()]
        placeholders = ", ".join("?" for _ in keys)
        rows = self._db.query(
            f'SELECT ukey, owner_id FROM "{self._KEYS_TABLE}" '
            f"WHERE dim = ? AND ukey IN ({placeholders})",
            [dim, *keys],
        )
        owners = {int(r["ukey"]): str(r["owner_id"]) for r in rows}
        hits: list[VectorHit] = []
        for key, distance in zip(keys, distances, strict=True):
            # inner-product distance on normalized vectors is 1 - cosine
            score = max(-1.0, min(1.0, 1.0 - distance))
            if score < min_score:
                continue
            hits.append(
                VectorHit(
                    id=str(key),
                    owner_type=self.owner_type,
                    owner_id=owners.get(key, ""),
                    score=score,
                    model=self.model,
                )
            )
        self.stats["searches"] += 1
        self.stats["search_seconds"] += time.perf_counter() - started
        return hits

    def count(self) -> int:
        return int(self._db.scalar(f'SELECT COUNT(*) FROM "{self._KEYS_TABLE}"', default=0))

    def stats_snapshot(self) -> dict[str, Any]:
        return {
            **self.stats,
            "backend": self.name,
            "dims": sorted(self._indexes),
            "path": self._path,
        }


# ── selection ────────────────────────────────────────────────────────────────

BACKEND_NAMES = ("sqlite-vec", "usearch", "legacy")

_BUILDERS: dict[str, type[VectorBackend]] = {
    "sqlite-vec": SqliteVecBackend,
    "usearch": USearchBackend,
    "legacy": LegacyStoreBackend,
}

_INSTALL_HINTS = {
    "sqlite-vec": "pip install sqlite-vec",
    "usearch": "pip install usearch",
    "legacy": "",
}


def available_backends() -> list[dict[str, Any]]:
    """Every known backend with its availability and install hint."""
    report = []
    for name in BACKEND_NAMES:
        ok, reason = _BUILDERS[name].available()
        report.append(
            {
                "name": name,
                "available": ok,
                "reason": reason,
                "install": _INSTALL_HINTS[name],
            }
        )
    return report


def select_vector_backend(
    db: Database,
    *,
    preference: str | None = None,
    owner_type: str = "memory",
    vectors: VectorStore | None = None,
    index_path: str | None = None,
) -> VectorBackend:
    """Pick a vector backend.

    ``preference="auto"`` (the default) takes the first available backend in
    quality order: sqlite-vec → usearch → legacy. A named preference that is
    not installed raises :class:`RuntimeError` with the install command —
    fail fast, never silently downgrade.
    """
    pref = (preference or "auto").strip().lower()
    if pref == "auto":
        for name in BACKEND_NAMES:
            ok, _ = _BUILDERS[name].available()
            if ok:
                backend = _build(name, db, owner_type=owner_type, vectors=vectors, index_path=index_path)
                _log.debug("vector backend selected: %s", name)
                return backend
        raise StorageError("no vector backend available — not even the built-in legacy store")
    if pref not in _BUILDERS:
        raise ValueError(
            f"unknown vector backend {preference!r}; choose from 'auto', "
            + ", ".join(repr(n) for n in BACKEND_NAMES)
        )
    ok, reason = _BUILDERS[pref].available()
    if not ok:
        hint = _INSTALL_HINTS[pref]
        raise RuntimeError(
            f"vector backend {pref!r} is not available: {reason}. Install it with: {hint}"
        )
    return _build(pref, db, owner_type=owner_type, vectors=vectors, index_path=index_path)


def _build(
    name: str,
    db: Database,
    *,
    owner_type: str,
    vectors: VectorStore | None,
    index_path: str | None,
) -> VectorBackend:
    if name == "legacy":
        return LegacyStoreBackend(db, owner_type=owner_type, vectors=vectors)
    if name == "usearch":
        return USearchBackend(db, owner_type=owner_type, path=index_path)
    return SqliteVecBackend(db, owner_type=owner_type)
