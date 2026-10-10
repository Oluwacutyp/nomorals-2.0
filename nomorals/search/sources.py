"""Source adapters for federated search.

Reuse survey (done before writing anything — this layer federates, it
does not reimplement):

- books (``nomorals/books/library.py``) — **reuse directly**. ``Library``
  already does FTS5 BM25 search with snippets and returns ``SearchHit``.
  The adapter only maps ``SearchHit`` into ``SearchResult``.
- documents (``nomorals/documents/index.py``) — **reuse, thin adapter**.
  ``DocumentIndex`` is the Wave K inverted index; it is in-memory, so the
  adapter resolves a populated index: an explicitly passed index, an
  ad-hoc build from ``--dir`` (same suffix allowlist as ``nm doc
  search``), or a persisted index at ``<workspace>/documents/index.json``
  (``DocumentIndex.save``/``load`` round-trip). No index found → the
  source is skipped with a note, not a crash.
- memory (``nomorals/memory/manager.py``) — **reuse directly** via
  ``context.memory``. ``MemoryManager.recall`` merges semantic + lexical
  + recency. The adapter drops the recency fallback records (both
  ``semantic`` and ``lexical`` are 0.0 on those — they never matched the
  query) so "no matches" stays honest, and keeps the default private
  exclusion.
- wisdom (``nomorals/wisdom/keeper.py``) — **reuse directly**.
  ``WisdomKeeper.ask`` returns ``Answer.passages`` (``ProvenanceHit``:
  work/translator/section/url/score/canon_status); the adapter passes
  that provenance through verbatim. Note: the corpus delegates to
  ``books.Library`` internally, but on its own root — it is a distinct
  searchable corpus from the books source.
- timeline (``nomorals/os/timeline.py``) — **wrap with the thinnest
  adapter**. ``Timeline.query`` filters by topic glob + time but has no
  free-text search, so the adapter pulls rows (pushing since/before into
  the query) and substring-matches query terms over
  topic/source/payload in Python. The ``Timeline`` instance is *injected*
  — never imported here — because ``nomorals/os`` is L6 and this package
  is L5.
- code (``nomorals/tools/code_indexer.py``) — **reuse via the existing
  helpers**: ``_get_indexer(context)`` builds the indexer (hashing
  embedder, no model download) and ``_run_async`` drives its async
  ``search``. No new machinery.
- connectors (``nomorals/connectors/``) — **skip**. The connectors own no
  local searchable store; their reads (e.g. ``list_repos``) are live API
  calls needing network + credentials, out of scope for this layer.
- web (``nomorals/search/web.py``) — **new in the universal upgrade**.
  Six live web-search backends behind this same ``SourceAdapter``
  interface: SearXNG (keyless metasearch, primary free), ddgs
  (keyless multi-engine via the optional ``ddgs`` package), Tavily,
  Serper, Exa, and Brave (keyed free-tier APIs; Brave's free tier was
  withdrawn 2026-02, so it is last). Each backend probes cheaply and
  reports itself unavailable when unconfigured — no key/instance/package
  means a skip note, never a crash.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from ..books.library import Library
from ..documents import DocumentIndex, parse_path
from ..memory.manager import MemoryManager
from ..tools.code_indexer import _get_indexer, _run_async
from ..wisdom import WisdomKeeper
from .base import SourceAdapter
from .errors import SearchError
from .model import SearchResult
from .osint import OSINT_SPECS
from .web import WEB_SPECS, WebSearchSource

__all__ = [
    "SourceAdapter",
    "BooksAdapter",
    "DocsAdapter",
    "MemoryAdapter",
    "WisdomAdapter",
    "CodeAdapter",
    "TimelineAdapter",
    "WebSearchSource",
    "SOURCE_SPECS",
    "list_sources",
    "build_adapters",
    "default_doc_index_path",
]

_TERM_RE = re.compile(r"[a-z0-9]+")

#: Same suffix allowlist ``nm doc search`` uses for ad-hoc indexing.
_DOC_SUFFIXES = frozenset({
    ".pdf", ".docx", ".xlsx", ".pptx", ".html", ".htm",
    ".md", ".markdown", ".csv", ".tsv", ".txt",
})


def _workspace_dir(context: Any) -> Path:
    settings = getattr(context, "settings", None)
    root = getattr(settings, "workspace_dir", None) if settings else None
    return Path(root) if root else Path.cwd() / "workspace"


def default_doc_index_path(context: Any) -> Path:
    """Where a persisted document index is looked for by default."""
    return _workspace_dir(context) / "documents" / "index.json"


def _query_terms(query: str) -> list[str]:
    return [t for t in _TERM_RE.findall(query.lower()) if len(t) >= 2]


# SourceAdapter now lives in .base (shared with web.py, avoiding an
# import cycle); re-exported here so ``from .sources import
# SourceAdapter`` keeps working (it is also in __all__ above).


# ── books ──────────────────────────────────────────────────────────────

class BooksAdapter(SourceAdapter):
    """FTS5 BM25 book search — reuses ``books.Library`` directly."""

    name = "books"
    result_type = "book"
    description = "book library full-text search (FTS5 BM25)"

    def __init__(self, context: Any, library: Library | None = None) -> None:
        self._context = context
        self._library = library

    def _lib(self) -> Library:
        if self._library is None:
            self._library = Library(self._context)
        return self._library

    def probe(self) -> str | None:
        try:
            db_path = self._lib().db_path()
        except Exception as exc:  # noqa: BLE001 - probe reports, never raises
            return f"unavailable: {exc}"
        if not db_path.exists():
            return "no library database yet (no books ingested)"
        try:
            con = sqlite3.connect(str(db_path))
            try:
                count = con.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
            finally:
                con.close()
        except Exception as exc:  # noqa: BLE001 - probe reports, never raises
            return f"unavailable: {exc}"
        if not count:
            return "library is empty (no passages ingested)"
        return None

    def search(self, query, *, limit, since=None, before=None) -> list[SearchResult]:
        hits = self._lib().search(query, top=limit)
        out = []
        for h in hits:
            out.append(SearchResult(
                query=query,
                title=h.title or h.book,
                snippet=h.passage,
                source=self.name,
                type=self.result_type,
                raw_score=h.score,
                provenance={
                    "book": h.book,
                    "title": h.title,
                    "author": getattr(h, "author", ""),
                    "chapter": h.chapter,
                    "chapter_number": getattr(h, "chapter_number", 0),
                    "book_slug": h.book_slug,
                    "match": h.source,
                    "ingested_at": getattr(h, "ingested_at", 0.0),
                },
                timestamp=getattr(h, "ingested_at", 0.0) or None,
                source_id=f"book:{h.book_slug}:{h.chapter}",
            ))
        return out


# ── documents ──────────────────────────────────────────────────────────

def _index_dir(root: Path) -> DocumentIndex:
    """Build a throwaway DocumentIndex from a directory of documents."""
    index = DocumentIndex()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _DOC_SUFFIXES:
            continue
        try:
            index.add(parse_path(str(path)))
        except Exception:  # noqa: BLE001 - skip unreadable files, keep going
            continue
    return index


class DocsAdapter(SourceAdapter):
    """Wave K inverted index — reuses ``DocumentIndex``; resolves a
    populated index from (in order) an injected index, ``--dir``, or the
    persisted default ``<workspace>/documents/index.json``."""

    name = "docs"
    result_type = "doc"
    description = "parsed document index (Wave K inverted index)"

    def __init__(
        self,
        context: Any,
        index: DocumentIndex | None = None,
        *,
        doc_dir: str | Path | None = None,
        index_path: str | Path | None = None,
    ) -> None:
        self._context = context
        self._index = index
        self._doc_dir = Path(doc_dir).expanduser() if doc_dir else None
        self._index_path = (
            Path(index_path).expanduser() if index_path
            else default_doc_index_path(context)
        )

    def _resolve(self) -> DocumentIndex | None:
        if self._index is not None:
            return self._index
        if self._doc_dir is not None:
            if not self._doc_dir.is_dir():
                raise SearchError(f"not a directory: {self._doc_dir}")
            self._index = _index_dir(self._doc_dir)
            return self._index
        if self._index_path.exists():
            try:
                self._index = DocumentIndex.load(self._index_path)
            except Exception as exc:
                raise SearchError(
                    f"cannot load document index from {self._index_path}: {exc}"
                ) from exc
            return self._index
        return None

    def probe(self) -> str | None:
        try:
            index = self._resolve()
        except SearchError as exc:
            return str(exc)
        if index is None:
            return (
                "no document index: save one at "
                f"{self._index_path} (DocumentIndex.save) or pass --dir"
            )
        if len(index) == 0:
            return "document index is empty"
        return None

    def search(self, query, *, limit, since=None, before=None) -> list[SearchResult]:
        index = self._resolve()
        if index is None:  # probe() already reported this; stay total
            return []
        hits = index.search(query, limit=limit)
        out = []
        for h in hits:
            out.append(SearchResult(
                query=query,
                title=h["title"] or h["doc_id"],
                snippet=h["snippet"],
                source=self.name,
                type=self.result_type,
                raw_score=float(h["score"]),
                provenance={"doc_id": h["doc_id"], "title": h["title"]},
                timestamp=None,  # the index persists id/title/text only
                source_id=f"doc:{h['doc_id']}",
            ))
        return out


# ── memory ─────────────────────────────────────────────────────────────

class MemoryAdapter(SourceAdapter):
    """Episodic + semantic memory — reuses ``MemoryManager.recall``."""

    name = "memory"
    result_type = "memory"
    description = "episodic + semantic memory recall"

    def __init__(self, context: Any, manager: MemoryManager | None = None) -> None:
        self._context = context
        self._manager = manager

    def _mgr(self) -> MemoryManager | None:
        if self._manager is None:
            self._manager = getattr(self._context, "memory", None)
        return self._manager

    def probe(self) -> str | None:
        mgr = self._mgr()
        if mgr is None:
            return "no memory manager on context"
        try:
            records = int(mgr.stats_snapshot().get("records", 0))
        except Exception as exc:  # noqa: BLE001 - probe reports, never raises
            return f"unavailable: {exc}"
        if records == 0:
            return "no memories stored yet"
        return None

    def search(self, query, *, limit, since=None, before=None) -> list[SearchResult]:
        mgr = self._mgr()
        if mgr is None:
            return []
        result = mgr.recall(query, limit=limit)
        out = []
        for r in result.records:
            # recall() falls back to recent memories when nothing matches;
            # those records carry zero semantic AND zero lexical scores, so
            # they are recency, not query matches — exclude them.
            if (r.semantic or 0.0) == 0.0 and (r.lexical or 0.0) == 0.0:
                continue
            text = r.content or ""
            title = f"[{r.kind}] {text[:80]}".strip()
            out.append(SearchResult(
                query=query,
                title=title,
                snippet=text[:400],
                source=self.name,
                type=self.result_type,
                raw_score=float(r.score or 0.0),
                provenance={
                    "id": r.id,
                    "kind": r.kind,
                    "tags": r.tags,
                    "origin": r.origin,
                    "record_source": r.source,
                    "importance": r.importance,
                },
                timestamp=r.created_at,
                source_id=f"memory:{r.id}",
            ))
        return out


# ── wisdom ─────────────────────────────────────────────────────────────

class WisdomAdapter(SourceAdapter):
    """WisdomKeeper corpus — reuses ``WisdomKeeper.ask``; provenance
    (work/translator/section/url/canon_status) passes through verbatim."""

    name = "wisdom"
    result_type = "passage"
    description = "WisdomKeeper esoteric corpus passages with provenance"

    def __init__(self, context: Any, keeper: WisdomKeeper | None = None) -> None:
        self._context = context
        self._keeper = keeper

    def _keeper_or_build(self) -> WisdomKeeper:
        if self._keeper is None:
            self._keeper = WisdomKeeper(self._context)
        return self._keeper

    def probe(self) -> str | None:
        try:
            ingested = self._keeper_or_build().status()["corpus"]["ingested"]
        except Exception as exc:  # noqa: BLE001 - probe reports, never raises
            return f"unavailable: {exc}"
        if not ingested:
            return "wisdom corpus has no ingested texts yet"
        return None

    def search(self, query, *, limit, since=None, before=None) -> list[SearchResult]:
        answer = self._keeper_or_build().ask(query, top=limit)
        out = []
        for p in answer.passages:
            out.append(SearchResult(
                query=query,
                title=p.work,
                snippet=p.snippet,
                source=self.name,
                type=self.result_type,
                raw_score=float(p.score),
                provenance={
                    "work": p.work,
                    "translator": p.translator,
                    "section": p.section,
                    "url": p.url,
                    "canon_status": p.canon_status,
                },
                timestamp=None,  # the corpus stores no per-passage timestamps
                source_id=f"wisdom:{p.work}:{p.section}",
            ))
        return out


# ── code ───────────────────────────────────────────────────────────────

class CodeAdapter(SourceAdapter):
    """Semantic code search — reuses ``tools/code_indexer`` via its own
    ``_get_indexer`` constructor helper and ``_run_async`` sync driver."""

    name = "code"
    result_type = "code"
    description = "semantic code-unit search over the indexed repo"

    def __init__(self, context: Any, indexer: Any = None) -> None:
        self._context = context
        self._indexer = indexer

    def _idx(self) -> Any:
        if self._indexer is None:
            self._indexer = _get_indexer(self._context)
        return self._indexer

    def _count(self) -> int:
        idx = self._idx()
        return int(idx.db.scalar("SELECT COUNT(*) FROM code_units", default=0) or 0)

    def probe(self) -> str | None:
        try:
            count = self._count()
        except Exception as exc:  # noqa: BLE001 - probe reports, never raises
            return f"unavailable: {exc}"
        if count == 0:
            return "no code indexed yet (index a repo first)"
        return None

    def search(self, query, *, limit, since=None, before=None) -> list[SearchResult]:
        results = _run_async(self._idx().search(query, limit=limit))
        out = []
        for r in results:
            u = r.unit
            snippet = u.docstring or (u.code or "")[:240]
            out.append(SearchResult(
                query=query,
                title=f"{u.name} ({u.file_path}:{u.line_start})",
                snippet=snippet,
                source=self.name,
                type=self.result_type,
                raw_score=float(r.score),
                provenance={
                    "file": u.file_path,
                    "name": u.name,
                    "unit_type": u.unit_type,
                    "line": u.line_start,
                    "line_end": u.line_end,
                    "language": u.language,
                    "parent": u.parent,
                },
                timestamp=None,  # code units carry no timestamps
                source_id=f"code:{u.unit_id}",
            ))
        return out


# ── timeline ───────────────────────────────────────────────────────────

class TimelineAdapter(SourceAdapter):
    """Event timeline — the thinnest wrap over ``os.Timeline``.

    ``Timeline.query`` has no free-text search (topic glob + time only),
    so this adapter pushes since/before into the query and substring-
    matches query terms over topic/source/payload in Python. The
    ``Timeline`` instance is injected by the caller (L7 wiring) because
    this package must not import ``nomorals.os`` (L6).
    """

    name = "timeline"
    result_type = "event"
    description = "persisted event timeline"

    def __init__(self, timeline: Any = None) -> None:
        self._timeline = timeline

    def probe(self) -> str | None:
        if self._timeline is None:
            return "no timeline store wired"
        try:
            rows = self._timeline.query(limit=1)
        except Exception as exc:  # noqa: BLE001 - probe reports, never raises
            return f"unavailable: {exc}"
        if not rows:
            return "no events recorded yet"
        return None

    def search(
        self, query, *, limit, since=None, before=None
    ) -> list[SearchResult]:
        if self._timeline is None:
            return []
        terms = _query_terms(query)
        if not terms:
            return []
        rows = self._timeline.query(
            since=since, until=before, limit=max(limit * 20, 200)
        )
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            haystack = " ".join((
                str(row.get("topic", "")),
                str(row.get("source", "")),
                str(row.get("event_id", "")),
                json.dumps(row.get("data", {}), ensure_ascii=False, default=str),
            )).lower()
            matched = sum(1 for t in terms if t in haystack)
            if matched:
                scored.append((matched / len(terms), row))
        scored.sort(key=lambda kv: (-kv[0], -(kv[1].get("ts") or 0)))
        out = []
        for frac, row in scored[:limit]:
            data = row.get("data", {})
            snippet = json.dumps(data, ensure_ascii=False, default=str)[:300]
            out.append(SearchResult(
                query=query,
                title=str(row.get("topic", "(event)")),
                snippet=snippet or str(row.get("topic", "")),
                source=self.name,
                type=self.result_type,
                raw_score=frac,
                provenance={
                    "event_id": row.get("event_id"),
                    "topic": row.get("topic"),
                    "source": row.get("source"),
                    "session_id": row.get("session_id"),
                    "project_id": row.get("project_id"),
                    "mission_id": row.get("mission_id"),
                    "artifact_id": row.get("artifact_id"),
                },
                timestamp=row.get("ts"),
                source_id=f"event:{row.get('event_id')}",
            ))
        return out


#: Canonical source order: (name, result type, description, adapter class).
#: The order doubles as the ranking tie-break. Local knowledge first
#: (memory → wisdom → books → docs → code → timeline), then the live web
#: backends in free-first priority order (see ``web.WEB_SPECS``).
SOURCE_SPECS: list[tuple[str, str, str, type[SourceAdapter]]] = [
    ("memory", "memory", MemoryAdapter.description, MemoryAdapter),
    ("wisdom", "passage", WisdomAdapter.description, WisdomAdapter),
    ("books", "book", BooksAdapter.description, BooksAdapter),
    ("docs", "doc", DocsAdapter.description, DocsAdapter),
    ("code", "code", CodeAdapter.description, CodeAdapter),
    ("timeline", "event", TimelineAdapter.description, TimelineAdapter),
    *[
        (name, WebSearchSource.result_type, cls.description, cls)
        for name, cls in WEB_SPECS
    ],
    # OSINT primitives (keyless): username sweep, email check,
    # domain recon (crt.sh + RDAP), IP intel (ipwho.is + InternetDB).
    *[
        (name, rtype, desc, cls)
        for name, rtype, desc, cls in OSINT_SPECS
    ],
]


def list_sources() -> list[dict[str, str]]:
    """Metadata for every known source (name, type, description)."""
    return [
        {"name": name, "type": rtype, "description": desc}
        for name, rtype, desc, _ in SOURCE_SPECS
    ]


def valid_source_names() -> list[str]:
    return [name for name, _, _, _ in SOURCE_SPECS]


def valid_types() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for _, rtype, _, _ in SOURCE_SPECS:
        if rtype not in seen:
            seen.add(rtype)
            out.append(rtype)
    return out


def build_adapters(
    context: Any,
    *,
    doc_dir: str | Path | None = None,
    doc_index_path: str | Path | None = None,
    timeline: Any = None,
    web_backends: list[str] | None = None,
) -> dict[str, SourceAdapter]:
    """Instantiate one adapter per source. ``timeline`` is the injected
    ``os.Timeline`` (or duck-typed equivalent); ``None`` makes the
    timeline source report itself unavailable instead of crashing.
    ``web_backends`` optionally restricts which web backends are built
    (names from ``web.WEB_SPECS``); the default builds all six — each
    web backend probes cheaply and reports itself unavailable when its
    key/instance/package is missing, so building them never costs a
    network call."""
    if context is None:
        raise SearchError("federated_search needs a context (or prebuilt adapters)")
    wanted = set(web_backends) if web_backends is not None else None
    unknown = (wanted - {name for name, _ in WEB_SPECS}) if wanted else set()
    if unknown:
        raise SearchError(
            f"unknown web backends: {sorted(unknown)}; "
            f"valid: {[n for n, _ in WEB_SPECS]}"
        )
    adapters: dict[str, SourceAdapter] = {
        "memory": MemoryAdapter(context),
        "wisdom": WisdomAdapter(context),
        "books": BooksAdapter(context),
        "docs": DocsAdapter(context, doc_dir=doc_dir, index_path=doc_index_path),
        "code": CodeAdapter(context),
        "timeline": TimelineAdapter(timeline),
    }
    for name, cls in WEB_SPECS:
        if wanted is None or name in wanted:
            adapters[name] = cls(context)
    return adapters
