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
    vector BLOB NOT NULL
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

    def upsert(self, key: str, vector: list[float]) -> None:
        if len(vector) != self.dim:
            raise VectorStoreError(
                f"vector for {key!r} has dim {len(vector)}, "
                f"index expects {self.dim}")
        norm = math.sqrt(sum(x * x for x in vector))
        vec = [x / norm for x in vector] if norm > 0 else vector
        blob = _serialize_f32(vec)
        self._con.execute(
            "INSERT OR REPLACE INTO vectors(key, vector) VALUES (?, ?)",
            (key, blob))
        if self._use_vec0:
            self._con.execute("DELETE FROM vec_items WHERE text_key = ?",
                              (key,))
            self._vec0_insert(key, vec)
        self._con.commit()

    def build(self, keys: list[str], texts: list[str],
              backend: EmbeddingBackend, *,
              batch: int = 64, progress=None) -> int:
        """Embed every (key, text) pair and (re)build the whole index.
        Returns the number of vectors stored."""
        if len(keys) != len(texts):
            raise VectorStoreError(
                f"keys/texts length mismatch: {len(keys)} vs {len(texts)}")
        self._con.execute("DELETE FROM vectors")
        if self._use_vec0:
            self._con.execute("DELETE FROM vec_items")
        self._con.commit()
        n = 0
        for i in range(0, len(keys), batch):
            chunk_keys = keys[i:i + batch]
            chunk_texts = texts[i:i + batch]
            vectors = embed_texts(backend, chunk_texts)
            for key, vec in zip(chunk_keys, vectors):
                self.upsert(key, vec)
                n += 1
            if progress is not None:
                progress(n, len(keys))
        return n

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

    def search(self, query_vector: list[float],
               top: int = 8) -> list[tuple[str, float]]:
        """KNN over the index → [(passage_key, cosine_similarity)]."""
        if len(query_vector) != self.dim:
            raise VectorStoreError(
                f"query dim {len(query_vector)} != index dim {self.dim}")
        norm = math.sqrt(sum(x * x for x in query_vector))
        q = [x / norm for x in query_vector] if norm > 0 else query_vector
        if self._use_vec0:
            return self._search_vec0(q, top)
        return self._search_python(q, top)

    def _search_python(self, q: list[float],
                       top: int) -> list[tuple[str, float]]:
        scored: list[tuple[str, float]] = []
        for key, blob in self._con.execute(
                "SELECT key, vector FROM vectors"):
            scored.append((key, _cosine(q, _deserialize_f32(bytes(blob)))))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:max(top, 0)]

    def _search_vec0(self, q: list[float],
                     top: int) -> list[tuple[str, float]]:
        import sqlite_vec
        rows = self._con.execute(
            "SELECT text_key, distance FROM vec_items "
            "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (sqlite_vec.serialize_float32(q), top)).fetchall()
        # vec0 MATCH returns cosine *distance* (0 = identical);
        # report similarity = 1 - distance to match the python path.
        return [(key, 1.0 - float(dist)) for key, dist in rows]

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
