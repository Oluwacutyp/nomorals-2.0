"""Full-text search over parsed documents, ranked with BM25.

The index stores documents in a small SQLite database (``:memory:`` by
default) and ranks with the existing :class:`nomorals.storage.fts.FTSIndex`
(SQLite FTS5 ``bm25()``) instead of duplicating ranking logic here.

Query language (Meilisearch/Tantivy-inspired, FTS5-native):

* plain terms — ``london bridge`` (OR by default, ``operator="AND"``)
* exact phrases — ``"london bridge"``
* field filters — ``title:report`` (fields: ``title``, ``text``)
* prefix search — ``operator``/``prefix=True`` turns ``lond`` into ``lond*``
  (the portable typo-tolerance fallback)
* weighted columns — ``weights={"title": 3.0}`` boosts title hits

Result shape: ``search()`` returns ``{doc_id, title, score, snippet}``
dicts; ``score`` is the FTS5 ``bm25()`` rank negated (higher is better).
``highlight=True`` wraps matches in ``<mark>`` tags (FTS5 ``highlight()``);
otherwise snippets keep the ~120-character window around the first hit.

Persistence: :meth:`DocumentIndex.save` writes a version-3 SQLite file
(version 2 adds the ``format`` column; version 3 is current).
:meth:`DocumentIndex.load` reads version-2/3 files and transparently
migrates legacy version-1 JSON indexes (written before the BM25 move) by
rebuilding them on the new backend.

If the SQLite build lacks FTS5 the index fails fast with
:class:`DocumentError` at construction time — it never silently returns
empty results.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from ..core.errors import StorageError
from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger
from ..storage.db import Database
from ..storage.fts import FTSIndex, build_match_query, escape_fts
from .errors import DocumentError
from .model import Document, Section, full_text

__all__ = ["DocumentIndex"]

_log = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SNIPPET_RADIUS_BEFORE = 40
_SNIPPET_RADIUS_AFTER = 80

#: Persistence format written by :meth:`DocumentIndex.save`.
_INDEX_VERSION = 3
#: Previous SQLite format (no ``format`` column); :meth:`load` still reads it.
_PREVIOUS_SQLITE_VERSION = 2
#: Legacy JSON format written before the BM25 migration; :meth:`load` rebuilds it.
_LEGACY_JSON_VERSION = 1
_SQLITE_MAGIC = b"SQLite format 3\x00"

_FTS_TABLE = "documents_fts"
_META_TABLE = "documents_meta"
_VERSION_TABLE = "index_meta"
_VOCAB_TABLE = "documents_fts_vocab"

#: Fields addressable by ``field:term`` filters.
_SEARCH_FIELDS = ("title", "text")
_FIELD_FILTER_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):(.+)$")
_QUERY_TOKEN_RE = re.compile(r'"([^"]*)"|(\S+)')


def _emit(topic: str, data: dict[str, Any]) -> None:
    """Publish a telemetry event. Best-effort: a broken bus or subscriber
    must never break indexing (fail-open telemetry, fail-closed function)."""
    try:
        global_bus.publish(Event(topic=topic, data=data, source=__name__))
    except Exception:  # noqa: BLE001 - telemetry is fail-open
        _log.debug("event %s failed", topic, exc_info=True)


def _tokenize(text: str) -> list[str]:
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if len(tok) >= 2]


_fts5_probe_result: bool | None = None


def _fts5_available() -> bool:
    """Probe whether this SQLite build ships the FTS5 module (cached)."""
    global _fts5_probe_result
    if _fts5_probe_result is None:
        try:
            conn = sqlite3.connect(":memory:")
            try:
                conn.execute('CREATE VIRTUAL TABLE "_fts5_probe" USING fts5("x")')
            finally:
                conn.close()
        except sqlite3.OperationalError:
            _fts5_probe_result = False
        else:
            _fts5_probe_result = True
    return _fts5_probe_result


def _ensure_schema(db: Database) -> None:
    db.execute(
        f'CREATE TABLE IF NOT EXISTS "{_META_TABLE}" ('
        '"doc_id" TEXT PRIMARY KEY, '
        '"title" TEXT NOT NULL, '
        '"text" TEXT NOT NULL, '
        '"format" TEXT NOT NULL DEFAULT \'\')'
    )
    # v2 → v3 migration: documents indexed before the format column
    # existed get it added in place.
    try:
        columns = {row["name"] for row in
                   db.query(f'PRAGMA table_info("{_META_TABLE}")')}
    except StorageError:
        columns = set()
    if "format" not in columns:
        try:
            db.execute(f'ALTER TABLE "{_META_TABLE}" ADD COLUMN '
                       '"format" TEXT NOT NULL DEFAULT \'\'')
        except StorageError:
            pass  # raced with another _ensure_schema; column is there now
    db.execute(
        f'CREATE TABLE IF NOT EXISTS "{_VERSION_TABLE}" ('
        '"key" TEXT PRIMARY KEY, '
        '"value" TEXT NOT NULL)'
    )
    # The CREATE VIRTUAL TABLE is what actually requires FTS5; the probe in
    # __init__ guarantees we never reach this on a build without it.
    db.execute(
        f'CREATE VIRTUAL TABLE IF NOT EXISTS "{_FTS_TABLE}" '
        'USING fts5("title", "text")'
    )
    # Vocabulary table for suggest(): fts5vocab is a query-time view over
    # the FTS index, safe to create even for pre-existing tables.
    try:
        db.execute(
            f'CREATE VIRTUAL TABLE IF NOT EXISTS "{_VOCAB_TABLE}" '
            f'USING fts5vocab("{_FTS_TABLE}", \'row\')'
        )
    except StorageError:
        _log.debug("fts5vocab unavailable; suggest() will return []")


class DocumentIndex:
    """BM25-ranked full-text index over :class:`Document` full text.

    Backed by :class:`nomorals.storage.fts.FTSIndex` (SQLite FTS5) over an
    in-memory database by default.  Pass ``db`` to share a file-backed
    :class:`~nomorals.storage.db.Database` instead.
    """

    def __init__(self, db: Database | None = None) -> None:
        if not _fts5_available():
            raise DocumentError(
                "FTS5 is not available in this SQLite build; DocumentIndex "
                "requires FTS5-backed BM25 ranking and refuses to degrade to "
                "silent empty results"
            )
        self._db = db if db is not None else Database(":memory:")
        _ensure_schema(self._db)
        self._fts = FTSIndex(self._db, _FTS_TABLE, columns=["title", "text"])
        if not self._fts.available:
            # Defensive: construction probes FTS5 above, so reaching here
            # means the table vanished under us.
            raise DocumentError(
                f"FTS table {_FTS_TABLE!r} is unavailable; refusing to serve "
                "unranked (empty) search results"
            )

    def __len__(self) -> int:
        return int(
            self._db.scalar(f'SELECT COUNT(*) FROM "{_META_TABLE}"', default=0)
        )

    def close(self) -> None:
        """Release the backing database connections."""
        self._db.close()

    # ── writes ───────────────────────────────────────────────────────────────
    def add(self, doc: Document) -> None:
        """Index ``doc``.  Re-adding an existing id replaces the entry."""
        if not doc.id:
            raise DocumentError("cannot index a document without an id")
        text = full_text(doc)
        if not text.strip():
            raise DocumentError(f"document {doc.id} has no indexable text")
        self.remove(doc.id)
        rowid = self._db.insert(
            _META_TABLE,
            {"doc_id": doc.id, "title": doc.title, "text": text,
             "format": doc.format or ""},
        )
        self._fts.put(rowid, [doc.title, text])
        _emit("document.indexed", {"doc_id": doc.id, "title": doc.title})

    def remove(self, doc_id: str) -> bool:
        """Drop ``doc_id`` from the index.  Returns True when it was present."""
        row = self._db.query_one(
            f'SELECT rowid FROM "{_META_TABLE}" WHERE "doc_id" = ?', (doc_id,)
        )
        if row is None:
            return False
        self._fts.delete(int(row["rowid"]))
        self._db.delete(_META_TABLE, '"doc_id" = ?', (doc_id,))
        return True

    # ── reads ────────────────────────────────────────────────────────────────
    def _snippet(self, text: str, terms: list[str]) -> str:
        lowered = text.lower()
        hit = -1
        for term in terms:
            pos = lowered.find(term)
            if pos >= 0 and (hit < 0 or pos < hit):
                hit = pos
        if hit < 0:
            window = text[:_SNIPPET_RADIUS_BEFORE + _SNIPPET_RADIUS_AFTER]
            return window + ("…" if len(text) > len(window) else "")
        start = max(0, hit - _SNIPPET_RADIUS_BEFORE)
        end = min(len(text), hit + _SNIPPET_RADIUS_AFTER)
        snippet = text[start:end]
        if start > 0:
            snippet = "…" + snippet.lstrip()
        if end < len(text):
            snippet = snippet.rstrip() + "…"
        return " ".join(snippet.split())

    @staticmethod
    def _build_match(query: str, operator: str, prefix: bool) -> str:
        """Compile the query language to an FTS5 MATCH expression.

        Preserves ``"exact phrases"`` and ``field:term`` filters (fields:
        title, text); plain terms are phrase-escaped; ``prefix=True``
        appends ``*`` to each term.
        """
        joiner = " AND " if operator.upper() == "AND" else " OR "
        parts: list[str] = []
        for match in _QUERY_TOKEN_RE.finditer(query):
            phrase, word = match.group(1), match.group(2)
            if phrase is not None:
                toks = _tokenize(phrase)
                if toks:
                    parts.append('"' + " ".join(toks) + '"')
                continue
            field_match = _FIELD_FILTER_RE.match(word or "")
            if field_match and field_match.group(1) in _SEARCH_FIELDS:
                field, raw = field_match.group(1), field_match.group(2)
                toks = _tokenize(raw)
                if not toks:
                    continue
                term = toks[0] if len(toks) == 1 else \
                    '"' + " ".join(toks) + '"'
                if prefix:
                    term += "*"
                parts.append("{%s} : %s" % (field, term))
                continue
            for tok in _tokenize(word or ""):
                parts.append(escape_fts(tok) + ("*" if prefix else ""))
        return joiner.join(parts)

    def _run_match(self, match: str, limit: int,
                   highlight: bool) -> list[tuple[int, float, str | None]]:
        """Execute a raw MATCH expression → [(rowid, score, snippet)]."""
        if highlight:
            snip_sql = (f'highlight("{_FTS_TABLE}", 1, '
                        f"'<mark>', '</mark>') AS snip")
        else:
            snip_sql = "NULL AS snip"
        sql = (
            f'SELECT rowid, bm25("{_FTS_TABLE}") AS rank, {snip_sql} '
            f'FROM "{_FTS_TABLE}" WHERE "{_FTS_TABLE}" MATCH ? '
            f"ORDER BY rank LIMIT ?"
        )
        try:
            rows = self._db.query(sql, (match, limit))
        except (sqlite3.OperationalError, StorageError) as exc:
            raise DocumentError(f"bad search query: {exc}") from exc
        return [(int(r["rowid"]), -float(r["rank"]), r["snip"]) for r in rows]

    def search(self, query: str, limit: int = 10, *,
               operator: str = "OR", prefix: bool = False,
               highlight: bool = False,
               weights: dict[str, float] | None = None,
               explain: bool = False) -> list[dict]:
        """Search the index; each hit is {doc_id, title, score, snippet}.

        Score is the FTS5 ``bm25()`` rank (negated so higher is better) — a
        float.  A document matches when it contains any query term (OR
        semantics, as before); ties break on doc id so ordering is
        deterministic.

        Query language: ``"exact phrases"``, ``title:term`` / ``text:term``
        field filters, ``operator="AND"`` for all-terms matching,
        ``prefix=True`` for ``term*`` prefix matching (typo-tolerance
        fallback), ``weights={"title": 3.0}`` to boost column hits,
        ``highlight=True`` for ``<mark>``-wrapped FTS5 snippets, and
        ``explain=True`` to attach the compiled MATCH expression to hits.
        """
        terms = _tokenize(query or "")
        if not terms:
            raise DocumentError("search query has no indexable terms")
        if limit <= 0:
            raise DocumentError(f"limit must be positive, got {limit}")
        if operator.upper() not in ("AND", "OR"):
            raise DocumentError(
                f"operator must be AND or OR, got {operator!r}")
        if not self._fts.available:
            raise DocumentError(
                "FTS backend became unavailable; refusing to return empty results"
            )
        match = self._build_match(query, operator, prefix)
        if not match:
            raise DocumentError("search query has no indexable terms")
        raw_hits = self._run_match(match, limit, highlight)
        if not raw_hits:
            return []
        rows = {
            int(r["rowid"]): r
            for r in self._db.query(
                f'SELECT rowid, "doc_id", "title", "text", "format" '
                f'FROM "{_META_TABLE}" '
                f'WHERE "rowid" IN ({", ".join("?" * len(raw_hits))})',
                [rowid for rowid, _, _ in raw_hits],
            )
        }
        ordered = sorted(
            (h for h in raw_hits if h[0] in rows),
            key=lambda h: (-h[1], rows[h[0]]["doc_id"]),
        )
        results = []
        for rowid, score, fts_snippet in ordered:
            record = rows[rowid]
            snippet = (fts_snippet if highlight and fts_snippet
                       else self._snippet(record["text"], terms))
            hit: dict[str, Any] = {
                "doc_id": record["doc_id"],
                "title": record["title"],
                "score": float(score),
                "snippet": snippet,
            }
            if weights:
                # Per-column bm25 weights are applied at query time; the
                # score above already reflects the default weighting, so
                # re-run weighted when requested.
                hit["score"] = self._weighted_score(
                    match, rowid, weights)
                hit["score"] = float(hit["score"])
            if explain:
                hit["explain"] = {"match": match, "operator": operator.upper(),
                                  "prefix": prefix}
            results.append(hit)
        if weights:
            # Re-sort after weight adjustment (deterministic tie-break).
            results.sort(key=lambda h: (-h["score"], h["doc_id"]))
        return results

    def _weighted_score(self, match: str, rowid: int,
                        weights: dict[str, float]) -> float:
        """Re-score one hit with per-column bm25 weights."""
        args = ", ".join(str(float(weights.get(c, 1.0)))
                         for c in ("title", "text"))
        sql = (f'SELECT bm25("{_FTS_TABLE}", {args}) AS rank '
               f'FROM "{_FTS_TABLE}" WHERE rowid = ? '
               f'AND "{_FTS_TABLE}" MATCH ?')
        try:
            row = self._db.query_one(sql, (rowid, match))
        except (sqlite3.OperationalError, StorageError):
            return 0.0
        return -float(row["rank"]) if row else 0.0

    def count(self, query: str, *, operator: str = "OR",
              prefix: bool = False) -> int:
        """Number of documents matching ``query`` (same query language as
        :meth:`search`)."""
        terms = _tokenize(query or "")
        if not terms:
            raise DocumentError("search query has no indexable terms")
        match = self._build_match(query, operator, prefix)
        if not match:
            raise DocumentError("search query has no indexable terms")
        try:
            return int(self._db.scalar(
                f'SELECT COUNT(*) FROM "{_FTS_TABLE}" '
                f'WHERE "{_FTS_TABLE}" MATCH ?',
                (match,), default=0))
        except (sqlite3.OperationalError, StorageError) as exc:
            raise DocumentError(f"bad search query: {exc}") from exc

    def suggest(self, prefix: str, limit: int = 8) -> list[str]:
        """Term completions for ``prefix`` from the FTS5 vocabulary
        (search-as-you-type hook).  Returns [] when the prefix is too
        short or the vocab table is unavailable — never raises."""
        if limit <= 0:
            raise DocumentError(f"limit must be positive, got {limit}")
        clean = (prefix or "").strip().lower()
        if len(clean) < 2:
            return []
        like = (clean.replace("\\", "\\\\").replace("%", "\\%")
                .replace("_", "\\_"))
        try:
            rows = self._db.query(
                f'SELECT DISTINCT term FROM "{_VOCAB_TABLE}" '
                f"WHERE term LIKE ? ESCAPE '\\' ORDER BY term LIMIT ?",
                (like + "%", limit))
        except (sqlite3.OperationalError, StorageError):
            return []
        return [str(r["term"]) for r in rows]

    def facet(self, field: str = "format") -> dict[str, int]:
        """Count indexed documents per value of ``field`` (faceted-search
        hook).  Currently ``field`` must be ``"format"``."""
        if field != "format":
            raise DocumentError(
                f"cannot facet on {field!r}: only 'format' is facetable")
        try:
            rows = self._db.query(
                f'SELECT "format", COUNT(*) AS n FROM "{_META_TABLE}" '
                f'GROUP BY "format" ORDER BY n DESC')
        except StorageError as exc:
            raise DocumentError(f"facet failed: {exc}") from exc
        return {str(r["format"]) or "(unknown)": int(r["n"]) for r in rows}

    def stats(self) -> dict[str, Any]:
        """Index statistics: document count, per-format facet, version."""
        return {
            "documents": len(self),
            "formats": self.facet(),
            "index_version": _INDEX_VERSION,
        }

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> Path:
        """Persist the index as a version-3 SQLite file.

        Replaces any existing file at ``path``.  Legacy version-1 JSON files
        are not written anymore — see :meth:`load` for the migration path.
        """
        file_path = Path(path)
        if file_path.is_dir():
            raise DocumentError(f"cannot write index to {file_path}: is a directory")
        if (
            self._db.path is not None
            and file_path.resolve() == self._db.path.resolve()
        ):
            # Already file-backed at this exact path: stamp the version and
            # compact; the data is already there.
            self._write_version(self._db)
            self._fts.optimize()
            return file_path
        try:
            if file_path.exists():
                file_path.unlink()
        except OSError as exc:
            raise DocumentError(
                f"cannot write index to {file_path}: {exc}") from exc
        try:
            dest = Database(file_path)
        except (OSError, sqlite3.Error) as exc:
            raise DocumentError(
                f"cannot write index to {file_path}: {exc}") from exc
        try:
            _ensure_schema(dest)
            self._write_version(dest)
            fts = FTSIndex(dest, _FTS_TABLE, columns=["title", "text"])
            for row in self._db.query(
                f'SELECT "doc_id", "title", "text", "format" FROM "{_META_TABLE}"'
            ):
                rowid = dest.insert(
                    _META_TABLE,
                    {
                        "doc_id": row["doc_id"],
                        "title": row["title"],
                        "text": row["text"],
                        "format": row.get("format", "") or "",
                    },
                )
                fts.put(rowid, [row["title"], row["text"]])
            fts.optimize()
        finally:
            dest.close()
        return file_path

    @staticmethod
    def _write_version(db: Database) -> None:
        db.execute(
            f'INSERT INTO "{_VERSION_TABLE}" ("key", "value") VALUES (?, ?) '
            'ON CONFLICT("key") DO UPDATE SET "value" = excluded."value"',
            ("version", str(_INDEX_VERSION)),
        )

    @classmethod
    def load(cls, path: str | Path) -> DocumentIndex:
        """Load an index saved with :meth:`save`.

        Accepts version-2/3 SQLite files (v2 gains empty ``format`` values)
        and transparently migrates legacy version-1 JSON files (rebuilt on
        the BM25 backend).  The returned index is always in-memory; call
        :meth:`save` to persist changes.
        """
        file_path = Path(path)
        if not file_path.is_file():
            raise DocumentError(f"cannot load index from {file_path}: no such file")
        try:
            with open(file_path, "rb") as handle:
                magic = handle.read(len(_SQLITE_MAGIC))
        except OSError as exc:
            raise DocumentError(
                f"cannot load index from {file_path}: {exc}") from exc
        if magic == _SQLITE_MAGIC:
            return cls._load_sqlite(file_path)
        return cls._load_legacy_json(file_path)

    @classmethod
    def _load_sqlite(cls, file_path: Path) -> DocumentIndex:
        try:
            source = Database(file_path)
        except (OSError, sqlite3.Error) as exc:
            raise DocumentError(
                f"cannot load index from {file_path}: {exc}") from exc
        try:
            try:
                has_schema = source.table_exists(_VERSION_TABLE) and \
                    source.table_exists(_META_TABLE)
            except StorageError as exc:
                raise DocumentError(
                    f"cannot load index from {file_path}: {exc}") from exc
            if not has_schema:
                raise DocumentError(
                    f"not a document index file: {file_path}")
            try:
                version = source.scalar(
                    f'SELECT "value" FROM "{_VERSION_TABLE}" '
                    'WHERE "key" = \'version\'',
                    default=None,
                )
            except StorageError as exc:
                raise DocumentError(
                    f"cannot load index from {file_path}: {exc}") from exc
            if str(version) not in (str(_INDEX_VERSION),
                                    str(_PREVIOUS_SQLITE_VERSION)):
                raise DocumentError(
                    f"unsupported document index version {version!r} in "
                    f"{file_path} (expected {_INDEX_VERSION})")
            try:
                rows = source.query(
                    f'SELECT "doc_id", "title", "text", "format" '
                    f'FROM "{_META_TABLE}"'
                )
            except StorageError:
                # Version-2 files have no format column.
                try:
                    rows = source.query(
                        f'SELECT "doc_id", "title", "text" FROM "{_META_TABLE}"'
                    )
                except StorageError as exc:
                    raise DocumentError(
                        f"cannot load index from {file_path}: {exc}") from exc
        finally:
            source.close()
        index = cls()
        for row in rows:
            doc = Document(id=str(row["doc_id"]), title=str(row["title"]),
                           format=str(row.get("format", "") or ""))
            # Rebuild via add() so meta + FTS rows stay consistent; the stored
            # text is injected as a single section because only text persisted.
            doc.sections = [Section(level=1, heading="", text=str(row["text"]))]
            index.add(doc)
        return index

    @classmethod
    def _load_legacy_json(cls, file_path: Path) -> DocumentIndex:
        """Migrate a version-1 JSON index (pre-BM25) onto the new backend."""
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DocumentError(
                f"cannot load index from {file_path}: {exc}") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("version") != _LEGACY_JSON_VERSION
        ):
            raise DocumentError(f"not a document index file: {file_path}")
        index = cls()
        for entry in payload.get("docs", []):
            doc = Document(id=str(entry.get("id", "")),
                           title=str(entry.get("title", "")))
            doc.sections = [Section(level=1, heading="", text=str(entry.get("text", "")))]
            index.add(doc)
        return index
