"""Vector index for the wisdom corpus.

Two storage tiers, chosen at open time:

- **sqlite-vec** (``vec0`` virtual table) when the ``sqlite_vec`` package
  is installed *and* this Python's SQLite build allows loading extensions.
  Zero-server, in-process, KNN via the native ``MATCH`` operator.
- **Pure-Python brute force** otherwise: vectors stored as float32 BLOBs
  in a plain SQLite table, cosine similarity computed in Python with
  ``struct``/``math`` (stdlib only). Slower at scale, but works on every
  machine and every Python build — the corpus is passage-level (thousands
  of rows, not millions), so this is genuinely fine.

The index is keyed by passage key (``"<book_slug>#<chapter_number>"``)
and remembers which backend + dimension produced the vectors; a store
built by a different model refuses to serve until rebuilt (vectors from
different models live in different spaces — mixing them is silent
garbage, so this fails fast instead).
"""
from __future__ import annotations

import math
import sqlite3
import struct
from typing import Any, Iterable

from .embeddings import EmbeddingBackend, embed_texts
from .errors import VectorStoreError

_META_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)
"""
_PLAIN_SCHEMA = """
CREATE TABLE IF NOT EXISTS vectors (
    key TEXT PRIMARY KEY,
    vector BLOB NOT NULL,
    tradition TEXT NOT NULL DEFAULT '',
    work TEXT NOT NULL DEFAULT ''
)
"""

#: Embedding cache: sha256(backend|model|role|text) → vector. Rebuilds
#: consult this first, so re-indexing an unchanged corpus is free instead
#: of re-embedding everything (the vault-rag "carried over" pattern).
_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS embed_cache (
    cache_key TEXT PRIMARY KEY,
    vector BLOB NOT NULL,
    created_at REAL NOT NULL
)
"""


def _serialize_f32(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _deserialize_f32(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob[:n * 4]))


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise VectorStoreError(
            f"vector dimension mismatch in cosine: {len(a)} vs {len(b)}")
    dot = 0.0
    for x, y in zip(a, b):
        dot += x * y
    return dot  # vectors are unit-normalized, so dot == cosine


def _try_load_vec0(con: sqlite3.Connection) -> bool:
    """Load the sqlite-vec extension into ``con``. Returns True on
    success; False when the package is missing or the SQLite build
    forbids extension loading. Never raises."""
    try:
        import sqlite_vec
    except Exception:
        return False
    if not hasattr(con, "enable_load_extension"):
        return False
    try:
        con.enable_load_extension(True)
        sqlite_vec.load(con)
        con.enable_load_extension(False)
    except Exception:
        return False
    # Prove the virtual-table module actually registered.
    try:
        con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _vec0_probe "
                    "USING vec0(embedding float[2])")
        con.execute("DROP TABLE _vec0_probe")
    except Exception:
        return False
    return True


class VectorIndex:
    """One corpus-wide vector index, persisted in SQLite."""

    def __init__(self, path: str, *, backend_name: str,
                 dim: int) -> None:
        self.path = path
        self.backend_name = backend_name
        self.dim = dim
        self._con = sqlite3.connect(path)
        self._use_vec0 = _try_load_vec0(self._con)
        self._init_schema()

    # ── schema ────────────────────────────────────────────────────
    def _init_schema(self) -> None:
        cur = self._con.execute("SELECT name FROM sqlite_master "
                                "WHERE type='table' AND name='meta'")
        if cur.fetchone() is None:
            self._con.execute(_META_SCHEMA)
            self._con.execute(_PLAIN_SCHEMA)
            self._con.execute(_CACHE_SCHEMA)
            if self._use_vec0:
                self._con.execute(
                    f"CREATE VIRTUAL TABLE vec_items USING "
                    f"vec0(embedding float[{self.dim}], text_key TEXT)")
            self._set_meta("backend", self.backend_name)
            self._set_meta("dim", str(self.dim))
            self._set_meta("engine", "vec0" if self._use_vec0 else "python")
            self._con.commit()
        else:
            stored_backend = self._get_meta("backend", "")
            stored_dim = self._get_meta("dim", "")
            if stored_backend != self.backend_name or \
                    stored_dim != str(self.dim):
                raise VectorStoreError(
                    f"vector store at {self.path} was built with "
                    f"backend={stored_backend!r} dim={stored_dim!r}, "
                    f"but this index wants backend={self.backend_name!r} "
                    f"dim={self.dim}; delete or rebuild the index")
            self._con.execute(_CACHE_SCHEMA)
            self._migrate_filter_columns()
            engine = self._get_meta("engine", "")
            # The native tier may come and go between runs (package
            # installed/removed); re-detect and align the schema.
            want_vec0 = self._use_vec0
            has_vec0 = engine == "vec0"
            if want_vec0 and not has_vec0:
                self._con.execute(
                    f"CREATE VIRTUAL TABLE vec_items USING "
                    f"vec0(embedding float[{self.dim}], text_key TEXT)")
                for key, blob in self._con.execute(
                        "SELECT key, vector FROM vectors"):
                    self._vec0_insert(key, _deserialize_f32(bytes(blob)))
                self._set_meta("engine", "vec0")
                self._con.commit()
            elif has_vec0 and not want_vec0:
                # vec0 gone (or unloadable) this run: fall back to the
                # plain table, which is always kept in sync.
                self._set_meta("engine", "python")
                self._con.commit()

    def _migrate_filter_columns(self) -> None:
        """Add tradition/work filter columns to older indexes.

        New columns so old rows keep working (empty = unfiltered);
        build() backfills them.
        """
        cols = {r[1] for r in self._con.execute(
            "PRAGMA table_info(vectors)").fetchall()}
        for col in ("tradition", "work"):
            if col not in cols:
                self._con.execute(
                    f"ALTER TABLE vectors ADD COLUMN {col} TEXT DEFAULT ''")
        self._con.execute(
            "CREATE INDEX IF NOT EXISTS idx_vectors_tradition "
            "ON vectors(tradition)")
        self._con.commit()

    def _set_meta(self, key: str, value: str) -> None:
        self._con.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            (key, value))

    def _get_meta(self, key: str, default: str = "") -> str:
        row = self._con.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    # ── writes ────────────────────────────────────────────────────
    def _vec0_insert(self, key: str, vec: list[float]) -> None:
        import sqlite_vec
        self._con.execute(
            "INSERT INTO vec_items(rowid, embedding, text_key) "
            "VALUES ((SELECT COALESCE(MAX(rowid),0)+1 FROM vec_items), "
            "?, ?)",
            (sqlite_vec.serialize_float32(vec), key))

    def upsert(self, key: str, vector: list[float], *,
               tradition: str = "", work: str = "") -> None:
        if len(vector) != self.dim:
            raise VectorStoreError(
                f"vector for {key!r} has dim {len(vector)}, "
                f"index expects {self.dim}")
        norm = math.sqrt(sum(x * x for x in vector))
        vec = [x / norm for x in vector] if norm > 0 else vector
        blob = _serialize_f32(vec)
        self._con.execute(
            "INSERT OR REPLACE INTO vectors(key, vector, tradition, work)"
            " VALUES (?, ?, ?, ?)",
            (key, blob, tradition or "", work or ""))
        if self._use_vec0:
            self._con.execute("DELETE FROM vec_items WHERE text_key = ?",
                              (key,))
            self._vec0_insert(key, vec)
        self._con.commit()

    # ── embedding cache ───────────────────────────────────────────
    def cache_lookup(self, cache_key: str) -> list[float] | None:
        """A cached vector for ``cache_key`` (sha256 of
        backend|model|role|text), or None on miss."""
        row = self._con.execute(
            "SELECT vector FROM embed_cache WHERE cache_key = ?",
            (cache_key,)).fetchone()
        if not row:
            return None
        return _deserialize_f32(bytes(row[0]))

    def cache_store(self, cache_key: str, vector: list[float]) -> None:
        import time
        norm = math.sqrt(sum(x * x for x in vector))
        vec = [x / norm for x in vector] if norm > 0 else vector
        self._con.execute(
            "INSERT OR REPLACE INTO embed_cache(cache_key, vector,"
            " created_at) VALUES (?, ?, ?)",
            (cache_key, _serialize_f32(vec), time.time()))

    def cache_stats(self) -> dict[str, int]:
        row = self._con.execute(
            "SELECT COUNT(*) FROM embed_cache").fetchone()
        return {"cached_vectors": int(row[0]) if row else 0}

    def build(self, keys: list[str], texts: list[str],
              backend: EmbeddingBackend, *,
              batch: int = 64, progress=None,
              metadata: dict[str, dict[str, str]] | None = None,
              use_cache: bool = True) -> dict[str, int]:
        """Embed every (key, text) pair and (re)build the whole index.

        The embedding cache is consulted first: unchanged passages keep
        their stored vectors instead of being re-embedded, so rebuilding
        after adding a few texts costs almost nothing. ``metadata`` maps
        a key → {"tradition": ..., "work": ...} for pre-filtered search.

        Returns {"vectors": n, "embedded": e, "cached": c}.
        """
        import time
        if len(keys) != len(texts):
            raise VectorStoreError(
                f"keys/texts length mismatch: {len(keys)} vs {len(texts)}")
        metadata = metadata or {}
        self._con.execute("DELETE FROM vectors")
        if self._use_vec0:
            self._con.execute("DELETE FROM vec_items")
        self._con.commit()
        n = embedded = cached = 0
        t0 = time.time()
        for i in range(0, len(keys), batch):
            chunk_keys = keys[i:i + batch]
            chunk_texts = texts[i:i + batch]
            # Cache first: only embed what we don't already have.
            to_embed: list[int] = []
            chunk_vecs: list[list[float] | None] = [None] * len(chunk_keys)
            if use_cache:
                for j, text in enumerate(chunk_texts):
                    hit = self.cache_lookup(
                        backend.cache_key(text, role="passage"))
                    if hit is not None and len(hit) == self.dim:
                        chunk_vecs[j] = hit
                        cached += 1
                    else:
                        to_embed.append(j)
            else:
                to_embed = list(range(len(chunk_keys)))
            if to_embed:
                fresh = embed_texts(
                    backend, [chunk_texts[j] for j in to_embed],
                    role="passage")
                for j, vec in zip(to_embed, fresh):
                    chunk_vecs[j] = vec
                    self.cache_store(
                        backend.cache_key(chunk_texts[j], role="passage"),
                        vec)
                    embedded += 1
            for key, vec, text in zip(chunk_keys, chunk_vecs, chunk_texts):
                assert vec is not None
                meta = metadata.get(key, {})
                self.upsert(key, vec,
                            tradition=meta.get("tradition", ""),
                            work=meta.get("work", ""))
                n += 1
            self._con.commit()
            if progress is not None:
                progress(n, len(keys))
        self._set_meta("built_at", str(time.time()))
        self._set_meta("build_seconds", f"{time.time() - t0:.1f}")
        self._con.commit()
        return {"vectors": n, "embedded": embedded, "cached": cached}

    def drop(self) -> None:
        self._con.execute("DELETE FROM vectors")
        if self._use_vec0:
            self._con.execute("DELETE FROM vec_items")
        self._set_meta("built_at", "")
        self._con.commit()

    # ── reads ─────────────────────────────────────────────────────
    @property
    def engine(self) -> str:
        """'vec0' (native sqlite-vec KNN) or 'python' (brute force)."""
        return self._get_meta("engine", "python")

    def count(self) -> int:
        row = self._con.execute(
            "SELECT COUNT(*) FROM vectors").fetchone()
        return int(row[0]) if row else 0

    def search(self, query_vector: list[float], top: int = 8, *,
               tradition: str = "", min_score: float = 0.0
               ) -> list[tuple[str, float]]:
        """KNN over the index → [(passage_key, cosine_similarity)].

        ``tradition`` pre-filters the candidate set (metadata filtering
        BEFORE vector search, per best practice) — on the native vec0
        tier this over-fetches then filters, since vec0 has no payload
        filtering; on the python tier it is a SQL WHERE. ``min_score``
        is the evidence gate: similarities below it are dropped so the
        vector half never returns "top K of anything".
        """
        if len(query_vector) != self.dim:
            raise VectorStoreError(
                f"query dim {len(query_vector)} != index dim {self.dim}")
        norm = math.sqrt(sum(x * x for x in query_vector))
        q = [x / norm for x in query_vector] if norm > 0 else query_vector
        if self._use_vec0:
            hits = self._search_vec0(q, max(top * 4, top), tradition)
        else:
            hits = self._search_python(q, max(top * 4, top), tradition)
        if min_score > 0:
            hits = [(k, s) for k, s in hits if s >= min_score]
        return hits[:max(top, 0)]

    def _search_python(self, q: list[float], top: int,
                       tradition: str = "") -> list[tuple[str, float]]:
        scored: list[tuple[str, float]] = []
        if tradition:
            rows = self._con.execute(
                "SELECT key, vector FROM vectors WHERE tradition = ?",
                (tradition,))
        else:
            rows = self._con.execute("SELECT key, vector FROM vectors")
        for key, blob in rows:
            scored.append((key, _cosine(q, _deserialize_f32(bytes(blob)))))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:max(top, 0)]

    def _search_vec0(self, q: list[float], top: int,
                     tradition: str = "") -> list[tuple[str, float]]:
        import sqlite_vec
        rows = self._con.execute(
            "SELECT text_key, distance FROM vec_items "
            "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (sqlite_vec.serialize_float32(q), top)).fetchall()
        # vec0 MATCH returns cosine *distance* (0 = identical);
        # report similarity = 1 - distance to match the python path.
        hits = [(key, 1.0 - float(dist)) for key, dist in rows]
        if tradition:
            allowed = {r[0] for r in self._con.execute(
                "SELECT key FROM vectors WHERE tradition = ?", (tradition,))}
            hits = [(k, s) for k, s in hits if k in allowed]
        return hits

    # ── health ────────────────────────────────────────────────────
    def stats(self) -> dict[str, Any]:
        """Index health: engine, counts, cache, build timing."""
        info = {
            "path": self.path,
            "backend": self.backend_name,
            "dim": self.dim,
            "engine": self.engine,
            "vectors": self.count(),
            "built_at": self._get_meta("built_at", ""),
            "build_seconds": self._get_meta("build_seconds", ""),
        }
        info.update(self.cache_stats())
        return info

    def vacuum(self) -> None:
        """Reclaim space (cache churn, deleted rows). Cheap maintenance."""
        self._con.execute("VACUUM")
        self._con.commit()

    def close(self) -> None:
        self._con.close()

    # context-manager sugar
    def __enter__(self) -> "VectorIndex":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def open_index(path: str, backend: EmbeddingBackend) -> VectorIndex:
    """Open (creating if needed) the vector index for ``backend``."""
    dim = backend.dim
    if dim is None:
        # Some backends only know their dim after the first embed.
        probe = backend.embed_one("")
        dim = len(probe)
    return VectorIndex(path, backend_name=backend.key, dim=int(dim))
