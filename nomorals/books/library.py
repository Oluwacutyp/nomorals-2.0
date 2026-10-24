"""The library: ingest real books, search across all of them, read passages.

BookForge *writes* books; the library *holds* the owner's existing ones —
the PDFs on the phone, the ``.txt``/``.md`` files saved over the years.
A book in the library is the raw material every other system can draw on:

    workspace/library/
        <slug>/
            book.json          # meta + chapter table of contents
            chapters/001.md    # one file per chapter (readable, editable)
            <filename>.txt     # the original, untouched, kept as the source
    workspace/library/library.db   # SQLite FTS5 full-text index (passage-level)

``ingest`` splits a book into chapters (markdown headings, ``Chapter N``
style headers, or a sensible chunker when neither is present) and indexes
every chapter.  ``search`` runs a real FTS5 BM25 query across ALL books
and returns ranked passages with context — each hit carries full
provenance (book, author, chapter, chapter number, ingest time, match
source) so federated search (``nm search``) can cite it precisely.  When
the runtime lacks FTS5 the searcher falls back to an in-memory BM25 —
same interface, same results shape, still real search.

Formats: ``.txt``/``.md`` read directly; ``.pdf``, ``.epub``, ``.docx``,
``.html``/``.htm`` are extracted through ``nomorals.documents`` parsers.

Beyond holding books, the library is a reading companion: per-book
reading progress (remember where you left off), bookmarks, annotations,
collections/shelves, tags, and star ratings — all in the same SQLite db.
"""
from __future__ import annotations

import html as _html
import json
import math
import re
import sqlite3
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from ..core.ids import new_id
from .failures import record_failure

__all__ = ["Library", "LibraryError", "IngestResult", "SearchHit",
           "HybridHit", "Digest"]

_WORD = re.compile(r"[A-Za-z0-9']+")

# ingest paths: plain text reads straight off disk; everything else goes
# through the nomorals.documents parsers (pdf, epub, docx, html).
_TEXT_EXTS = (".txt", ".md", ".markdown", ".text")
_DOC_EXTS = (".pdf", ".epub", ".docx", ".html", ".htm")
_INGEST_EXTS = _TEXT_EXTS + _DOC_EXTS

# ── museum + library next-level constants ─────────────────────────────
# URL fetch: bounded read so a hostile or broken server cannot grow
# memory without limit; the 200-char minimum and max_chars truncation
# still apply after extraction, exactly as for files.
_FETCH_MAX_BYTES = 20_000_000
_FETCH_TIMEOUT = 30.0
_FETCH_UA = "DevonLibrary/1.0 (+library ingest)"
# Embedding text is truncated per chapter: the hashing embedder is
# length-free, but native/router endpoints have context limits.
_EMBED_CHARS = 4000
# Reciprocal rank fusion constant (standard k=60).
_RRF_K = 60.0
# Health: consecutive failed checks before a source is dead.
_HEALTH_FAIL_THRESHOLD = 3
# Feed dedupe memory: seen item links per feed source, oldest pruned.
_FEED_SEEN_CAP = 1000
# Vector lane: one owner_type in the shared embeddings table; owner_id
# encodes the kind so passages and digests share one namespace.
_VECTOR_OWNER = "library"
# The embeddings table DDL is byte-identical to
# nomorals/storage/migrations.py (the memory system's table) so the
# real VectorStore works against library.db unchanged. This creates
# the table in the LIBRARY file only; migrations.py is never touched.
_EMBEDDINGS_DDL = (
    "CREATE TABLE IF NOT EXISTS embeddings ("
    " id TEXT PRIMARY KEY, owner_type TEXT NOT NULL,"
    " owner_id TEXT NOT NULL, model TEXT NOT NULL,"
    " dim INTEGER NOT NULL, norm REAL NOT NULL DEFAULT 0,"
    " vector BLOB NOT NULL, created_at REAL NOT NULL)"
)
_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
_TAG_RE = re.compile(r"<[^>]+>")


class LibraryError(ValueError):
    """Raised for invalid library operations."""


@dataclass
class IngestResult:
    slug: str
    title: str
    author: str
    chapters: int
    words: int
    source_file: str
    seconds: float
    strategy: str
    note: str = ""
    source_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug, "title": self.title, "author": self.author,
            "chapters": self.chapters, "words": self.words,
            "source_file": self.source_file,
            "seconds": round(self.seconds, 2), "strategy": self.strategy,
            "note": self.note, "source_id": self.source_id,
        }


@dataclass
class SearchHit:
    book: str
    title: str
    chapter: str
    score: float
    passage: str
    source: str  # "fts5" | "bm25-memory"
    book_slug: str = ""
    author: str = ""
    chapter_number: int = 0
    ingested_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "book": self.title or self.book,
            "chapter": self.chapter,
            "score": round(self.score, 4),
            "passage": self.passage,
            "source": self.source,
            "book_slug": self.book_slug,
            "author": self.author,
            "chapter_number": self.chapter_number,
            "ingested_at": self.ingested_at,
        }


@dataclass
class Digest:
    """A structured, revisable knowledge record — the brain's digest of
    a source, not raw text. Revisions live in `digest_revisions`; this
    row is always the current version."""

    record_id: str
    source_id: str
    claim: str
    confidence: float
    scope: str = ""
    status: str = "active"
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id, "source_id": self.source_id,
            "claim": self.claim, "confidence": round(self.confidence, 4),
            "scope": self.scope, "status": self.status,
            "created_at": self.created_at, "updated_at": self.updated_at,
        }


@dataclass
class HybridHit:
    """One fused retrieval hit: a passage or a digest, with the rank
    breakdown that produced its score (reciprocal rank fusion over the
    keyword-passage, keyword-digest, and vector lanes)."""

    kind: str  # "passage" | "digest"
    text: str
    score: float
    title: str = ""
    slug: str = ""
    chapter: str = ""
    chapter_number: int = 0
    record_id: str = ""
    confidence: float | None = None
    health: str = "unknown"
    source_id: str = ""
    scope: str = ""
    keyword_rank: int | None = None
    vector_rank: int | None = None
    lanes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "text": self.text,
            "score": round(self.score, 4), "title": self.title,
            "slug": self.slug, "chapter": self.chapter,
            "chapter_number": self.chapter_number,
            "record_id": self.record_id, "confidence": self.confidence,
            "health": self.health, "source_id": self.source_id,
            "scope": self.scope, "keyword_rank": self.keyword_rank,
            "vector_rank": self.vector_rank, "lanes": list(self.lanes),
        }


# ── chapter splitting ───────────────────────────────────────────────────

_MD_H1 = re.compile(r"^#[ \t]+(.+?)\s*$", re.MULTILINE)
_MD_ANY = re.compile(r"^#{1,2}[ \t]+(.+?)\s*$", re.MULTILINE)
# plain-text chapter heads: "Chapter 12", "CHAPTER 12 — THE DOOR",
# "12. The Door", "Part Two", "Book I", "Prologue", "Epilogue"
_PT_CHAPTER = re.compile(
    r"^[ \t]*(?:"
    r"(?i:chapter|part|book|section|prologue|epilogue|appendix|introduction)\b[\s.:\-–]*"
    r"[A-Za-z0-9IVXLCM]*"
    r"|(?i:\d{1,3}\.)(?=\s+[A-Z])"
    r")[ \t]*(.{0,80})[ \t]*$",
    re.MULTILINE,
)


def _clean_title(raw: str) -> str:
    return raw.strip().strip("#").strip(" \t:—-–")


def split_chapters(text: str) -> tuple[list[tuple[str, str]], str]:
    """Split a book into (title, body) chapters.

    Returns (chapters, strategy) with chapters as (name, body) tuples.
    Strategy is "markdown", "plain-chapters", or "chunks" — the caller
    reports it so the owner knows how the book was cut.
    """
    text = (text or "").replace("\r\n", "\n").strip()
    if not text:
        return [], "empty"

    # 1. markdown headings — one file per H1 (fall back to any heading)
    if text.count("\n#") or text.startswith("#"):
        heads = list(_MD_H1.finditer(text))
        if len(heads) >= 2:
            chapters = []
            for i, m in enumerate(heads):
                start = m.end()
                end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
                body = text[start:end].strip()
                if body:
                    chapters.append((_clean_title(m.group(1)), body))
            if chapters:
                return chapters, "markdown"
        heads = list(_MD_ANY.finditer(text))
        if len(heads) >= 2:
            chapters = []
            for i, m in enumerate(heads):
                start = m.end()
                end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
                body = text[start:end].strip()
                if body:
                    chapters.append((_clean_title(m.group(1)), body))
            if chapters:
                return chapters, "markdown"

    # 2. plain-text chapter heads
    heads = list(_PT_CHAPTER.finditer(text))
    # keep only heads that look like a chapter start (short line near a
    # paragraph boundary) and give a useful number of cuts
    if len(heads) >= 2:
        # drop heads sitting mid-paragraph (preceded by >40 chars on line)
        good = [h for h in heads
                if len(h.group(0).strip()) <= 100]
        if len(good) >= 2:
            chapters = []
            for i, h in enumerate(good):
                start = h.end()
                end = good[i + 1].start() if i + 1 < len(good) else len(text)
                body = text[start:end].strip()
                name = _clean_title(h.group(0))
                if body:
                    chapters.append((name or f"Part {i + 1}", body))
            if chapters:
                return chapters, "plain-chapters"

    # 3. fallback: size-based chunks on paragraph boundaries
    words = len(text.split())
    chunk_words = 800
    target = max(1, min(60, round(words / chunk_words)))
    size = max(300, len(text) // target)
    chapters = []
    pos = 0
    n = 1
    while pos < len(text):
        cut = text.find("\n\n", pos + size - 200)
        if cut == -1 or cut > pos + size + 2000:
            cut = len(text) if pos + size >= len(text) else (
                text.find("\n", pos + size) if text.find("\n", pos + size)
                != -1 and text.find("\n", pos + size) < pos + size + 400
                else len(text))
            if cut == -1:
                cut = len(text)
        end = min(len(text), max(pos + 1, cut))
        body = text[pos:end].strip()
        if body:
            chapters.append((f"Section {n}", body))
            n += 1
        pos = end
    return [c for c in chapters if c[1]], "chunks"


# ── url / feed fetching + discovery helpers ─────────────────────────

def _default_fetch(url: str, timeout: float = _FETCH_TIMEOUT,
                   ) -> tuple[bytes, str, str]:
    """Fetch a URL: (body bytes, content-type, final URL after redirects).

    Stdlib urllib, the repo's established fetch shape. Bounded read:
    bodies past _FETCH_MAX_BYTES raise LibraryError instead of growing
    memory. Network errors surface as LibraryError (never raw URLError
    — the caller decides whether a source is failing or dead)."""
    req = urllib.request.Request(url, headers={"User-Agent": _FETCH_UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            body = resp.read(_FETCH_MAX_BYTES + 1)
            final = resp.geturl()
    except urllib.error.HTTPError as exc:
        raise LibraryError(f"fetch {url} failed: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise LibraryError(f"fetch {url} failed: {exc.reason}") from exc
    except (OSError, ValueError) as exc:
        raise LibraryError(f"fetch {url} failed: {exc}") from exc
    if len(body) > _FETCH_MAX_BYTES:
        raise LibraryError(f"fetch {url} exceeded {_FETCH_MAX_BYTES} bytes")
    return body, ctype, final


def _default_probe(url: str, timeout: float = 15.0) -> int:
    """Liveness probe for health checks: GET, read ≤4KB, return status.

    A HEAD would be lighter but lies on servers that reject it; a
    small GET proves the document is actually servable."""
    req = urllib.request.Request(url, headers={"User-Agent": _FETCH_UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read(4096)
            return int(resp.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise LibraryError(f"probe {url} failed: {exc}") from exc


def _strip_html_fragment(text: str) -> str:
    """Best-effort text from an HTML fragment (feed descriptions).

    Full documents go through the documents engine; fragments get this
    small stripper (tags out, entities decoded, whitespace collapsed).
    It never raises on malformed input."""
    try:
        out = _TAG_RE.sub(" ", text or "")
        out = _html.unescape(out)
        return re.sub(r"\s+", " ", out).strip()
    except Exception as exc:  # noqa: BLE001 — fragment stripping never fails ingest
        record_failure("library._strip_html_fragment", exc)
        return (text or "").strip()


def _bytes_to_text(data: bytes, ctype: str, url: str) -> tuple[str, str, str]:
    """Extract (text, title, author) from fetched bytes.

    HTML/pdf/epub/docx go through the documents engine (same parsers
    as file ingest); plain text decodes directly. Returns "" title /
    author when the document carries none — the caller guesses."""
    if ctype in ("text/plain", "text/markdown", "text/csv", "") or \
            not ctype.startswith(("text/html", "application/pdf",
                                  "application/epub", "application/"
                                  "vnd.openxmlformats")):
        # Unknown types fall through to the parsers too when they look
        # like documents; plain text is the fast path.
        if ctype.startswith("text/") or ctype == "":
            try:
                return data.decode("utf-8", errors="replace"), "", ""
            except Exception as exc:  # noqa: BLE001 — decode fallback below
                record_failure("library._bytes_to_text", exc)
                pass
    from ..documents.model import full_text
    from ..documents.parsers import parse_bytes

    try:
        doc = parse_bytes(data, filename=url, mime=ctype)
    except Exception as exc:  # noqa: BLE001 - wrap, don't leak
        raise LibraryError(f"could not extract text from {url}: {exc}") from exc
    return full_text(doc), (doc.title or ""), (doc.author or "")


def _feed_local(tag: str) -> str:
    """Element tag without its namespace: '{http://...}item' -> 'item'."""
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _feed_children(elem: Any, name: str) -> list[Any]:
    return [c for c in list(elem) if _feed_local(c.tag) == name]


def _feed_first_text(elem: Any, names: tuple[str, ...]) -> str:
    for name in names:
        for child in _feed_children(elem, name):
            text = (child.text or "").strip()
            if text:
                return text
    return ""


def _parse_feed(data: bytes) -> tuple[str, list[dict[str, str]]]:
    """Parse RSS 2.0 / Atom with stdlib ElementTree (no feedparser on
    this box, and stdlib covers both formats). Returns (feed title,
    items) with items as {title, body, link} and HTML stripped from
    bodies. Raises LibraryError on unparseable XML or zero usable
    items."""
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise LibraryError(f"feed XML did not parse: {exc}") from exc
    feed_title = ""
    if _feed_local(root.tag) == "feed":  # Atom
        feed_title = _feed_first_text(root, ("title",))
        entries = _feed_children(root, "entry")
    else:  # RSS: rss/channel/item (or a bare channel)
        channels = _feed_children(root, "channel")
        for ch in channels:
            feed_title = _feed_first_text(ch, ("title",))
            if feed_title:
                break
        entries = []
        for ch in channels or [root]:
            entries.extend(_feed_children(ch, "item"))
    items: list[dict[str, str]] = []
    for entry in entries:
        title = _feed_first_text(entry, ("title",))
        body = ""
        # content:encoded is namespaced content; localname match finds it.
        for child in list(entry):
            if _feed_local(child.tag) == "encoded" and (child.text or "").strip():
                body = child.text or ""
                break
        if not body:
            body = _feed_first_text(entry, ("description", "summary",
                                            "content"))
        link = ""
        for child in _feed_children(entry, "link"):
            href = (child.get("href") or "").strip()
            text = (child.text or "").strip()
            if href:
                link = href
                break
            if text.startswith("http"):
                link = text
                break
        items.append({
            "title": _strip_html_fragment(title),
            "body": _strip_html_fragment(body),
            "link": link,
        })
    items = [i for i in items if i["title"] or i["body"]]
    if not items:
        raise LibraryError("feed contained no usable items")
    return feed_title, items


def _extract_urls(text: str) -> list[str]:
    """Bare URLs in text, trailing punctuation stripped, fragment cut.

    Used by discovery: links inside ingested content become candidate
    sources. Order-preserving, deduplicated."""
    out: list[str] = []
    seen: set[str] = set()
    for m in _URL_RE.finditer(text or ""):
        url = m.group(0).rstrip(".,;:!?\"'])").strip()
        url = url.split("#", 1)[0]
        if url and url not in seen:
            seen.add(url)
            out.append(url)
    return out


# ── the library ─────────────────────────────────────────────────────────

class Library:
    """Owns ``workspace/library/``: ingest, search, read, list, drop."""

    def __init__(self, context: Any, embedder: Any = None) -> None:
        self.context = context
        # Embedder is lazy: None means "build the hashing default on
        # first vector use" — no import-time I/O, no network, no cost.
        # Callers may inject a native/router Embedder instead.
        self.embedder = embedder

    # ── paths ─────────────────────────────────────────────────────────
    def root(self) -> Path:
        settings = getattr(self.context, "settings", None)
        root = None
        if settings is not None:
            root = getattr(settings, "workspace_dir", None)
        if not root:
            root = Path.cwd() / "workspace"
        d = Path(root) / "library"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def db_path(self) -> Path:
        return self.root() / "library.db"

    def book_dir(self, slug: str) -> Path:
        return self.root() / slug

    # ── db ────────────────────────────────────────────────────────────
    @staticmethod
    def _fts5_available() -> bool:
        probe = sqlite3.connect(":memory:")
        try:
            probe.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
            return True
        except sqlite3.OperationalError:
            return False
        finally:
            probe.close()

    def _connect(self) -> tuple[sqlite3.Connection, bool]:
        """(connection, uses_fts5).  The passages scheme is fixed per db
        file and remembered in meta — FTS5 if the runtime has it, a plain
        table otherwise (search then uses in-memory BM25)."""
        con = sqlite3.connect(self.db_path())
        con.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        con.execute(
            "CREATE TABLE IF NOT EXISTS books ("
            " slug TEXT PRIMARY KEY, title TEXT, author TEXT, "
            " source_file TEXT, chapters INTEGER, words INTEGER, "
            " strategy TEXT, ingested_at REAL)")
        # ── reading state + curation (progress, bookmarks, notes,
        #    collections, tags, ratings) ──────────────────────────────
        con.execute(
            "CREATE TABLE IF NOT EXISTS reading_progress ("
            " slug TEXT PRIMARY KEY, chapter INTEGER, offset_chars INTEGER, "
            " updated_at REAL)")
        con.execute(
            "CREATE TABLE IF NOT EXISTS bookmarks ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT, slug TEXT, "
            " chapter INTEGER, offset_chars INTEGER, label TEXT, "
            " created_at REAL)")
        con.execute(
            "CREATE TABLE IF NOT EXISTS annotations ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT, slug TEXT, "
            " chapter INTEGER, offset_chars INTEGER, quote TEXT, "
            " note TEXT, created_at REAL)")
        con.execute(
            "CREATE TABLE IF NOT EXISTS collections ("
            " name TEXT PRIMARY KEY, created_at REAL)")
        con.execute(
            "CREATE TABLE IF NOT EXISTS collection_books ("
            " collection TEXT, slug TEXT, added_at REAL, "
            " PRIMARY KEY (collection, slug))")
        con.execute(
            "CREATE TABLE IF NOT EXISTS book_tags ("
            " slug TEXT, tag TEXT, PRIMARY KEY (slug, tag))")
        con.execute(
            "CREATE TABLE IF NOT EXISTS ratings ("
            " slug TEXT PRIMARY KEY, stars INTEGER, rated_at REAL)")
        # ── reading statistics (KOReader-style sessions + streaks) ──
        con.execute(
            "CREATE TABLE IF NOT EXISTS reading_sessions ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT, slug TEXT, "
            " started_at REAL, ended_at REAL, seconds REAL, "
            " chapters INTEGER DEFAULT 0, words INTEGER DEFAULT 0)")
        con.execute(
            "CREATE TABLE IF NOT EXISTS book_series ("
            " slug TEXT PRIMARY KEY, series TEXT, series_index REAL)")
        # ── next-level: sources, digests, vectors ─────────────────────
        # Sources are records, not files: every ingest (file included)
        # creates one, so URL/feed/file share one provenance shape.
        con.execute(
            "CREATE TABLE IF NOT EXISTS sources ("
            " source_id TEXT PRIMARY KEY, kind TEXT, origin TEXT,"
            " title TEXT, status TEXT, health TEXT, failures INTEGER,"
            " fetched_at REAL, last_checked REAL, last_error TEXT,"
            " created_at REAL, meta TEXT)")
        # Structured revisable records: the current version lives here,
        # every earlier version in digest_revisions (append-only).
        con.execute(
            "CREATE TABLE IF NOT EXISTS digests ("
            " record_id TEXT PRIMARY KEY, source_id TEXT, claim TEXT,"
            " confidence REAL, scope TEXT, status TEXT,"
            " created_at REAL, updated_at REAL)")
        con.execute(
            "CREATE TABLE IF NOT EXISTS digest_revisions ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT, record_id TEXT,"
            " claim TEXT, confidence REAL, note TEXT, revised_by TEXT,"
            " revised_at REAL)")
        # Derived state: recomputed on create/revise, never history.
        con.execute(
            "CREATE TABLE IF NOT EXISTS digest_flags ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT, record_id TEXT,"
            " other_id TEXT, strategy TEXT, confidence REAL, note TEXT,"
            " created_at REAL)")
        # The memory system's embeddings table, in the library file, so
        # the real VectorStore works here unchanged (see _EMBEDDINGS_DDL).
        con.execute(_EMBEDDINGS_DDL)
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_owner"
            " ON embeddings(owner_type, owner_id)")
        # Books gain their source link idempotently: old library.db
        # files get the column on next open, existing rows keep NULL
        # (unknown source — health factor 1.0, never excluded).
        cols = {r[1] for r in con.execute("PRAGMA table_info(books)")}
        if "source_id" not in cols:
            con.execute("ALTER TABLE books ADD COLUMN source_id TEXT")
        row = con.execute(
            "SELECT value FROM meta WHERE key = 'fts5'").fetchone()
        if row is None:
            fts5 = self._fts5_available()
            if fts5:
                con.execute(
                    "CREATE VIRTUAL TABLE passages USING fts5("
                    "book_slug UNINDEXED, book_title UNINDEXED, "
                    "chapter UNINDEXED, number UNINDEXED, text)")
            else:
                con.execute(
                    "CREATE TABLE passages (book_slug TEXT, book_title TEXT, "
                    "chapter TEXT, number INTEGER, text TEXT)")
            con.execute(
                "INSERT INTO meta(key, value) VALUES ('fts5', ?)",
                ("1" if fts5 else "0",))
            con.commit()
            row = con.execute(
                "SELECT value FROM meta WHERE key = 'fts5'").fetchone()
        if row[0] == "1":
            # Digest keyword lane mirrors the passages lane: FTS5 when
            # the runtime has it, Python token overlap otherwise.
            con.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS digest_fts USING fts5("
                "record_id UNINDEXED, claim)")
            con.commit()
        return con, row[0] == "1"

    def _reindex_all(self, con: sqlite3.Connection,
                     fts5: bool) -> None:
        """Rebuild the whole passages index from every book.json.
        Repair utility only — ingest/drop use `_index_book` now."""
        con.execute("DELETE FROM passages")
        rows: list[tuple[str, str, str, int, str]] = []
        for book_json in self.root().glob("*/book.json"):
            slug = book_json.parent.name
            try:
                meta = json.loads(book_json.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            title = meta.get("title", "")
            for ch in meta.get("chapters", []):
                num = int(ch.get("number", 0))
                body_file = book_json.parent / "chapters" / f"{num:03d}.md"
                try:
                    body = body_file.read_text(encoding="utf-8")
                except OSError:
                    continue
                rows.append((slug, title, ch.get("title", ""), num, body))
        if rows:
            con.executemany(
                "INSERT INTO passages(book_slug, book_title, chapter, "
                "number, text) VALUES (?,?,?,?,?)", rows)
        con.commit()

    def _index_book(self, con: sqlite3.Connection, fts5: bool,
                    slug: str) -> None:
        """Index ONE book's chapters: delete that slug's passages,
        insert fresh from its book.json. The ingest/drop paths use
        this now — `_reindex_all`'s whole-index rebuild was a scaling
        wall the moment sources arrived as a stream. `_reindex_all`
        stays as a repair utility."""
        con.execute("DELETE FROM passages WHERE book_slug = ?", (slug,))
        book_json = self.book_dir(slug) / "book.json"
        try:
            meta = json.loads(book_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            con.commit()
            return
        title = meta.get("title", "")
        rows: list[tuple[str, str, str, int, str]] = []
        for ch in meta.get("chapters", []):
            num = int(ch.get("number", 0))
            body_file = book_json.parent / "chapters" / f"{num:03d}.md"
            try:
                body = body_file.read_text(encoding="utf-8")
            except OSError:
                continue
            rows.append((slug, title, ch.get("title", ""), num, body))
        if rows:
            con.executemany(
                "INSERT INTO passages(book_slug, book_title, chapter, "
                "number, text) VALUES (?,?,?,?,?)", rows)
        con.commit()

    def _persist_book(self, *, slug: str, title: str, author: str,
                      chapters: list[tuple[str, str]], strategy: str,
                      original_bytes: bytes, original_name: str,
                      source: dict[str, Any], source_file: str,
                      started: float,
                      chapter_extra: list[dict[str, Any]] | None = None,
                      ) -> IngestResult:
        """Shared write tail for file/URL/feed ingest: chapter files +
        original + book.json + incremental index + books row + source
        row + chapter vectors (best-effort). One shape for all kinds."""
        dest = self.book_dir(slug)
        chapters_dir = dest / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)
        (dest / original_name).write_bytes(original_bytes)

        total_words = 0
        meta_chapters = []
        for i, (name, body) in enumerate(chapters, 1):
            (chapters_dir / f"{i:03d}.md").write_text(
                f"# {name}\n\n{body}\n", encoding="utf-8")
            words = len(body.split())
            total_words += words
            entry: dict[str, Any] = {"number": i, "title": name,
                                     "words": words}
            if chapter_extra and i - 1 < len(chapter_extra):
                entry.update({k: v for k, v in chapter_extra[i - 1].items()
                              if v})
            meta_chapters.append(entry)

        now = time.time()
        meta = {
            "slug": slug, "title": title, "author": author,
            "source_file": original_name, "strategy": strategy,
            "chapters": meta_chapters, "total_words": total_words,
            "ingested_at": now, "source_id": source["source_id"],
        }
        (dest / "book.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        con, fts5 = self._connect()
        try:
            self._index_book(con, fts5, slug)
            con.execute(
                "INSERT OR REPLACE INTO books(slug,title,author,"
                "source_file,chapters,words,strategy,ingested_at,"
                "source_id) VALUES (?,?,?,?,?,?,?,?,?)",
                (slug, title, author, original_name, len(chapters),
                 total_words, strategy, now, source["source_id"]))
            con.execute(
                "INSERT OR REPLACE INTO sources(source_id,kind,origin,"
                "title,status,health,failures,fetched_at,last_checked,"
                "last_error,created_at,meta) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (source["source_id"], source.get("kind", "file"),
                 source.get("origin", ""), source.get("title", title),
                 source.get("status", "active"),
                 source.get("health", "unknown"),
                 int(source.get("failures", 0)),
                 source.get("fetched_at", now),
                 source.get("last_checked", 0.0),
                 source.get("last_error", ""),
                 source.get("created_at", now),
                 json.dumps(source.get("meta") or {})))
            con.commit()
        finally:
            con.close()

        note = ""
        vec_error = self._embed_book_vectors(slug, chapters)
        if vec_error:
            # Vectors are the retrieval accelerator, not the ingest:
            # a failed embed never fails the ingest, but it is said
            # out loud (note) and repaired by backfill_vectors().
            note = f"vectors skipped: {vec_error}"

        return IngestResult(
            slug=slug, title=title, author=author,
            chapters=len(chapters), words=total_words,
            source_file=source_file, seconds=time.perf_counter() - started,
            strategy=strategy, note=note,
            source_id=source["source_id"],
        )

    # ── ingest ────────────────────────────────────────────────────────
    def ingest(self, path: str | Path, *, title: str = "",
               author: str = "", max_chars: int = 4_000_000) -> IngestResult:
        started = time.perf_counter()
        p = Path(path).expanduser()
        if not p.exists():
            raise LibraryError(f"file not found: {p}")
        suffix = p.suffix.lower()
        if suffix not in _INGEST_EXTS:
            raise LibraryError(
                f"unsupported type {p.suffix!r} — supported: "
                + " ".join(_INGEST_EXTS))
        if suffix in _TEXT_EXTS:
            raw = p.read_text(encoding="utf-8", errors="replace")
            doc_title, doc_author = "", ""
        else:
            # pdf / epub / docx / html — reuse the documents engine
            from ..documents.model import full_text
            from ..documents.parsers import parse_path

            try:
                doc = parse_path(p)
            except Exception as exc:  # noqa: BLE001 - wrap, don't leak
                raise LibraryError(f"could not extract text from {p.name}: {exc}") from exc
            raw = full_text(doc)
            doc_title, doc_author = doc.title, doc.author
        if len(raw) > max_chars:
            raw = raw[:max_chars]
        if len(raw.strip()) < 200:
            raise LibraryError("file is too small to be a book")

        chapters, strategy = split_chapters(raw)
        if not chapters:
            raise LibraryError("could not cut the file into chapters")

        book_title = title.strip() or doc_title.strip()
        if not book_title:
            book_title = self._guess_title(raw, p)
        book_author = author.strip() or doc_author.strip()
        if not book_author:
            book_author = self._guess_author(raw, book_title, p)
        slug = self._slug(book_title, p.stem)

        source = {
            "source_id": new_id(), "kind": "file", "origin": str(p),
            "title": book_title, "status": "active", "health": "unknown",
            "failures": 0, "fetched_at": time.time(), "last_checked": 0.0,
            "last_error": "", "created_at": time.time(), "meta": {},
        }
        return self._persist_book(
            slug=slug, title=book_title, author=book_author,
            chapters=chapters, strategy=strategy,
            original_bytes=p.read_bytes(), original_name=p.name,
            source=source, source_file=str(p), started=started,
        )

    @staticmethod
    def _slug(title: str, stem: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9]+", "-", (title or "").strip().lower())
        slug = slug.strip("-")[:60] or re.sub(r"[^A-Za-z0-9]+", "-", stem).strip("-")[:60]
        return slug or "book"

    @staticmethod
    def _guess_title(raw: str, p: Path) -> str:
        # YAML-ish front matter
        if raw.startswith("---"):
            end = raw.find("\n---", 3)
            if end != -1:
                fm = raw[3:end]
                m = re.search(r"^title\s*:\s*(.+)$", fm, re.MULTILINE)
                if m:
                    return m.group(1).strip().strip("\"'")
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        for ln in lines[:12]:
            if ln.startswith("#"):
                continue  # a markdown heading is a chapter, not the title
            if len(ln) < 80 and not ln.endswith((".", ",")) and \
                    not re.match(r"^(chapter|part|prologue|by |copyright|isbn)", ln, re.I):
                if re.search(r"[A-Za-z]{3,}", ln):
                    return ln
        return p.stem.replace("-", " ").replace("_", " ").title()

    @staticmethod
    def _guess_author(raw: str, title: str, p: Path) -> str:
        if raw.startswith("---"):
            end = raw.find("\n---", 3)
            if end != -1:
                m = re.search(r"^author\s*:\s*(.+)$", raw[3:end], re.MULTILINE)
                if m:
                    return m.group(1).strip().strip("\"'")
        head = raw[:3000]
        m = re.search(r"(?im)^by[ \t]+([A-Z][A-Za-z'.-]+(?:[ \t][A-Z][A-Za-z'.-]+){0,3})\s*$", head)
        if m and m.group(1).strip() != title.strip():
            return m.group(1).strip()
        m = re.search(r"(?i)^\s*by ([A-Z][A-Za-z'.-]+(?:[ \t][A-Z][A-Za-z'.-]+){0,3})\s*$", head, re.MULTILINE)
        if m and m.group(1).strip() != title.strip():
            return m.group(1).strip()
        return ""

    # ── search ────────────────────────────────────────────────────────
    def search(self, query: str, top: int = 8, context_chars: int = 260,
               book: str = "") -> list[SearchHit]:
        query = (query or "").strip()
        if not query:
            return []
        con, fts5 = self._connect()
        try:
            if fts5:
                hits = self._search_fts5(con, query, top, context_chars, book)
                if not hits:
                    # strict AND query found nothing → OR of the words
                    words = [w for w in _WORD.findall(query) if len(w) > 1][:8]
                    if len(words) >= 2:
                        relaxed = " OR ".join(f'"{w}"' for w in words)
                        hits = self._search_fts5(
                            con, relaxed, top, context_chars, book)
            else:
                hits = self._search_memory(con, query, top, context_chars, book)
        finally:
            con.close()
        return hits[:top]

    def _search_fts5(self, con, query, top, context_chars, book) -> list[SearchHit]:
        try:
            fts_query = self._sanitize_fts(query)
        except LibraryError:
            return []
        sql = (
            "SELECT book_slug, book_title, chapter, number, text, "
            "bm25(passages) AS score FROM passages WHERE passages MATCH ?"
        )
        params: list[Any] = [fts_query]
        if book:
            sql += " AND book_slug = ?"
            params.append(book)
        sql += " ORDER BY score LIMIT ?"
        params.append(top * 4)
        try:
            rows = con.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return []
        meta = {r[0]: (r[1] or "", r[2] or 0.0)
                for r in con.execute(
                    "SELECT slug, author, ingested_at FROM books")}
        hits = []
        for slug, title, chapter, number, text, score in rows:
            author, ingested = meta.get(slug, ("", 0.0))
            hits.append(SearchHit(
                book=title, title=title, chapter=self._chapter_label(chapter, number),
                score=score, passage=self._context_window(text, fts_query,
                                                           context_chars),
                source="fts5", book_slug=slug, author=author,
                chapter_number=int(number or 0), ingested_at=ingested))
        return hits

    @staticmethod
    def _sanitize_fts(query: str) -> str:
        # FTS5 query syntax: quotes each token, joins with AND (implicit
        # in FTS5 when tokens are space-separated).  Drop anything that
        # can carry FTS operators into the query string.
        words = [w for w in _WORD.findall(query) if len(w) >= 2]
        if not words:
            raise LibraryError("no searchable words in query")
        return " ".join(f'"{w}"' for w in words)

    @staticmethod
    def _context_window(text: str, query: str, context_chars: int = 260) -> str:
        words = [w for w in _WORD.findall(query) if len(w) >= 2]
        low = text.lower()
        best, best_len = -1, -1
        for w in words[:6]:
            i = low.find(w.lower())
            if i == -1:
                continue
            score = len(w)
            if score > best_len:
                best, best_len = i, score
        if best == -1:
            return text[:context_chars].rstrip() + ("…" if len(text) > context_chars else "")
        start = max(0, best - context_chars // 3)
        end = min(len(text), start + context_chars)
        out = text[start:end].strip()
        return ("…" if start > 0 else "") + out + ("…" if end < len(text) else "")

    @staticmethod
    def _chapter_label(chapter: str, number: int) -> str:
        return chapter or f"Section {number}"

    def _search_memory(self, con, query, top, context_chars,
                       book) -> list[SearchHit]:
        """In-memory BM25 — used only when FTS5 is unavailable."""
        rows = con.execute(
            "SELECT book_slug, book_title, chapter, number, text FROM passages"
        ).fetchall()
        if book:
            rows = [r for r in rows if r[0] == book]
        if not rows:
            return []
        q_words = {w.lower() for w in _WORD.findall(query) if len(w) >= 2}
        if not q_words:
            return []
        n = len(rows)
        docs = []
        for slug, title, chapter, number, text in rows:
            toks = [t.lower() for t in _WORD.findall(text)]
            docs.append((slug, title, chapter, number, text, toks))
        df: dict[str, int] = {}
        for _, _, _, _, _, toks in docs:
            for w in set(toks):
                df[w] = df.get(w, 0) + 1
        scored = []
        for slug, title, chapter, number, text, toks in docs:
            tf: dict[str, int] = {}
            for t in toks:
                if t in q_words:
                    tf[t] = tf.get(t, 0) + 1
            if not tf:
                continue
            s = 0.0
            for w, f in tf.items():
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * f * (len(toks) and 1.0)
            scored.append((s, slug, title, chapter, number, text))
        scored.sort(key=lambda x: -x[0])
        meta = {r[0]: (r[1] or "", r[2] or 0.0)
                for r in con.execute(
                    "SELECT slug, author, ingested_at FROM books")}
        out = []
        for s, slug, title, ch, num, text in scored[:top]:
            author, ingested = meta.get(slug, ("", 0.0))
            out.append(SearchHit(
                book=title, title=title, chapter=self._chapter_label(ch, num),
                score=s, passage=self._context_window(text, query, context_chars),
                source="bm25-memory", book_slug=slug, author=author,
                chapter_number=int(num or 0), ingested_at=ingested))
        return out

    # ── read ──────────────────────────────────────────────────────────
    def read(self, slug: str, chapter: int = 0, limit: int = 4000) -> dict[str, Any]:
        # an explicit chapter goes through the same range check as
        # bookmarks/notes: out-of-range is an honest LibraryError, never
        # a raw IndexError — and never Python's negative indexing, which
        # silently served (and recorded progress for) the wrong chapter.
        b = self._check_chapter(slug, chapter) if chapter else self.load(slug)
        ch = b["chapters"][chapter - 1] if chapter else None
        out = {
            "slug": b["slug"], "title": b["title"], "author": b["author"],
            "chapters": len(b["chapters"]), "total_words": b["total_words"],
        }
        if ch is None:
            out["outline"] = [
                {"n": c["number"], "title": c["title"], "words": c["words"]}
                for c in b["chapters"]]
        else:
            path = self.book_dir(slug) / "chapters" / f"{ch['number']:03d}.md"
            text = path.read_text(encoding="utf-8") if path.exists() else ""
            out["chapter"] = ch["number"]
            out["chapter_title"] = ch["title"]
            out["words"] = ch["words"]
            out["text"] = text[:limit]
            out["truncated"] = len(text) > limit
            # reading a chapter is reading — remember where they left off
            self.set_progress(slug, ch["number"], 0)
        return out

    def load(self, slug: str) -> dict[str, Any]:
        path = self.book_dir(slug) / "book.json"
        if not path.exists():
            raise LibraryError(f"no book {slug!r} in the library — nm book list")
        return json.loads(path.read_text(encoding="utf-8"))

    def list_books(self) -> list[dict[str, Any]]:
        out = []
        for book_json in sorted(self.root().glob("*/book.json")):
            try:
                meta = json.loads(book_json.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append({
                "slug": meta.get("slug", book_json.parent.name),
                "title": meta.get("title", ""),
                "author": meta.get("author", ""),
                "chapters": len(meta.get("chapters", [])),
                "words": meta.get("total_words", 0),
                "strategy": meta.get("strategy", ""),
                "ingested": time.strftime(
                    "%Y-%m-%d", time.localtime(meta.get("ingested_at", 0))),
            })
        # enrich with curation state (ratings, tags, progress) — one query set
        con, _ = self._connect()
        try:
            ratings = {r[0]: r[1] for r in con.execute(
                "SELECT slug, stars FROM ratings")}
            tag_rows = con.execute(
                "SELECT slug, tag FROM book_tags ORDER BY tag").fetchall()
            prog_rows = con.execute(
                "SELECT slug, chapter FROM reading_progress").fetchall()
        finally:
            con.close()
        tags: dict[str, list[str]] = {}
        for slug, tag in tag_rows:
            tags.setdefault(slug, []).append(tag)
        chapters_by_slug = {b["slug"]: b["chapters"] for b in out}
        for b in out:
            slug = b["slug"]
            b["rating"] = ratings.get(slug, 0)
            b["tags"] = tags.get(slug, [])
            prog = next((p for p in prog_rows if p[0] == slug), None)
            total = max(1, chapters_by_slug.get(slug, 0))
            b["progress_percent"] = (
                round(100.0 * min(int(prog[1]), total) / total, 1)
                if prog else 0.0)
        return out

    def drop(self, slug: str) -> dict[str, Any]:
        dest = self.book_dir(slug)
        if not dest.exists():
            raise LibraryError(f"no book {slug!r} in the library")
        # Chapter numbers first: vector cleanup needs them, and the
        # dir is gone afterwards. A vector-cleanup failure never fails
        # the drop — stale vectors resolve to nothing in hybrid search
        # and backfill never resurrects them (no chapters, no rows).
        vec_note = ""
        try:
            meta = json.loads((dest / "book.json").read_text(encoding="utf-8"))
            numbers = [int(c.get("number", 0))
                       for c in meta.get("chapters", [])]
            self._delete_book_vectors(slug, numbers)
        except Exception as exc:  # noqa: BLE001 — drop proceeds regardless
            vec_note = f"vectors stale: {exc}"
        for f in dest.iterdir():
            if f.is_dir():
                for c in f.iterdir():
                    c.unlink(missing_ok=True)
                f.rmdir()
            else:
                f.unlink()
        dest.rmdir()
        con, fts5 = self._connect()
        con.execute("DELETE FROM passages WHERE book_slug = ?", (slug,))
        con.execute("DELETE FROM books WHERE slug = ?", (slug,))
        # drop the reading state too — no orphaned progress/bookmarks/notes
        for table in ("reading_progress", "bookmarks", "annotations",
                      "collection_books", "book_tags", "ratings"):
            con.execute(f"DELETE FROM {table} WHERE slug = ?", (slug,))
        con.commit()
        con.close()
        # Digests are knowledge, not the book: dropping a book never
        # deletes digests made from it (shed_dead archives them when
        # the SOURCE dies — a different, explicit act).
        out: dict[str, Any] = {"dropped": slug}
        if vec_note:
            out["vectors"] = vec_note
        return out

    # ── url / feed ingest ─────────────────────────────────────────────
    def _source_for_origin(self, con: sqlite3.Connection, kind: str,
                           origin: str) -> dict[str, Any] | None:
        row = con.execute(
            "SELECT source_id, kind, origin, title, status, health,"
            " failures, fetched_at, last_checked, last_error,"
            " created_at, meta FROM sources WHERE kind = ? AND origin = ?",
            (kind, origin)).fetchone()
        if row is None:
            return None
        d = {
            "source_id": row[0], "kind": row[1], "origin": row[2],
            "title": row[3], "status": row[4], "health": row[5],
            "failures": row[6], "fetched_at": row[7],
            "last_checked": row[8], "last_error": row[9],
            "created_at": row[10],
        }
        try:
            d["meta"] = json.loads(row[11] or "{}")
        except ValueError:
            d["meta"] = {}
        return d

    def ingest_url(self, url: str, *, title: str = "", author: str = "",
                   fetcher: Any = None, max_chars: int = 4_000_000,
                   ) -> IngestResult:
        """Ingest a URL as a book: fetch, extract, split, index — the
        same record shape as file ingest (book dir, chapters, passages,
        source row). Fetch happens FIRST: a failed fetch leaves no
        partial book and no source row. Re-ingesting the same URL
        updates its existing source row and book (same slug when the
        title matches; a changed title lands a new slug, the old book
        stays — same as re-ingesting a renamed file)."""
        started = time.perf_counter()
        url = (url or "").strip()
        if not url.startswith(("http://", "https://")):
            raise LibraryError(f"ingest_url needs an http(s) URL, got {url!r}")
        fetch = fetcher or _default_fetch
        try:
            data, ctype = fetch(url, _FETCH_TIMEOUT)[:2]
        except LibraryError:
            raise
        except Exception as exc:  # noqa: BLE001 - wrap, don't leak
            raise LibraryError(f"fetch {url} failed: {exc}") from exc
        raw, doc_title, doc_author = _bytes_to_text(data, ctype, url)
        if len(raw) > max_chars:
            raw = raw[:max_chars]
        if len(raw.strip()) < 200:
            raise LibraryError("fetched document is too small to be a book")
        chapters, strategy = split_chapters(raw)
        if not chapters:
            raise LibraryError("could not cut the fetched document into chapters")
        book_title = title.strip() or doc_title.strip()
        stem = url.rstrip("/").rsplit("/", 1)[-1].split("?", 1)[0] or "page"
        if not book_title:
            book_title = self._guess_title(raw, Path(stem))
        book_author = author.strip() or doc_author.strip()
        if not book_author:
            book_author = self._guess_author(raw, book_title, Path(stem))
        slug = self._slug(book_title, stem)
        ctype_name = {"text/html": "source.html", "application/pdf": "source.pdf",
                      "application/epub+zip": "source.epub",
                      "application/vnd.openxmlformats-officedocument."
                      "wordprocessingml.document": "source.docx"}.get(ctype, "source.txt")
        con, _ = self._connect()
        try:
            existing = self._source_for_origin(con, "url", url)
        finally:
            con.close()
        now = time.time()
        source = {
            "source_id": existing["source_id"] if existing else new_id(),
            "kind": "url", "origin": url, "title": book_title,
            "status": "active", "health": "ok", "failures": 0,
            "fetched_at": now,
            "last_checked": existing["last_checked"] if existing else 0.0,
            "last_error": "",
            "created_at": existing["created_at"] if existing else now,
            "meta": existing["meta"] if existing else {},
        }
        return self._persist_book(
            slug=slug, title=book_title, author=book_author,
            chapters=chapters, strategy=strategy,
            original_bytes=data, original_name=ctype_name,
            source=source, source_file=url, started=started,
        )

    def _read_book_chapters(self, slug: str,
                            ) -> tuple[list[tuple[str, str]],
                                       list[dict[str, Any]]]:
        """Reconstruct (chapters, extras) from an existing book's files
        — the feed-upsert path needs the old chapters to append to."""
        book_json = self.book_dir(slug) / "book.json"
        meta = json.loads(book_json.read_text(encoding="utf-8"))
        chapters: list[tuple[str, str]] = []
        extras: list[dict[str, Any]] = []
        for ch in meta.get("chapters", []):
            num = int(ch.get("number", 0))
            path = book_json.parent / "chapters" / f"{num:03d}.md"
            text = path.read_text(encoding="utf-8") if path.exists() else ""
            # Files are "# {title}\n\n{body}\n" — strip one header line.
            lines = text.split("\n", 1)
            body = lines[1].lstrip("\n") if len(lines) > 1 and \
                lines[0].startswith("# ") else text
            chapters.append((ch.get("title", ""), body.rstrip("\n")))
            extras.append({"link": ch.get("link", "")} if ch.get("link") else {})
        return chapters, extras

    def ingest_feed(self, url: str, *, title: str = "",
                    fetcher: Any = None, max_items: int = 100,
                    ) -> IngestResult:
        """Ingest an RSS/Atom feed as one book: items become chapters
        (item title = chapter title, item link in chapter meta).
        Re-ingesting upserts: unseen items append as new chapters,
        seen items are never duplicated (dedupe by link, else title).
        The feed's source row carries the seen set in its meta."""
        started = time.perf_counter()
        url = (url or "").strip()
        if not url.startswith(("http://", "https://")):
            raise LibraryError(f"ingest_feed needs an http(s) URL, got {url!r}")
        fetch = fetcher or _default_fetch
        try:
            data = fetch(url, _FETCH_TIMEOUT)[0]
        except LibraryError:
            raise
        except Exception as exc:  # noqa: BLE001 - wrap, don't leak
            raise LibraryError(f"fetch {url} failed: {exc}") from exc
        feed_title, items = _parse_feed(data)
        items = items[:max(1, max_items)]
        con, _ = self._connect()
        try:
            existing = self._source_for_origin(con, "feed", url)
        finally:
            con.close()
        now = time.time()
        book_title = title.strip() or feed_title.strip()
        slug = ""
        old_chapters: list[tuple[str, str]] = []
        old_extras: list[dict[str, Any]] = []
        seen: list[str] = []
        if existing:
            seen = list((existing.get("meta") or {}).get("seen") or [])
            book_slug = (existing.get("meta") or {}).get("book_slug", "")
            old_meta = None
            if book_slug:
                try:
                    old_meta = json.loads(
                        (self.book_dir(book_slug) / "book.json")
                        .read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    old_meta = None
            if old_meta is not None:
                slug = old_meta.get("slug", "")
                old_chapters, old_extras = self._read_book_chapters(slug)
        if not book_title:
            bodies = "\n\n".join(i["body"] for i in items)
            book_title = self._guess_title(bodies, Path("feed"))
        if not slug:
            stem = re.sub(r"[^A-Za-z0-9]+", "-",
                          url.split("://", 1)[-1].split("?", 1)[0]
                          .split("#", 1)[0]).strip("-")[:60] or "feed"
            slug = self._slug(book_title, stem)
        seen_set = set(seen)
        new_chapters: list[tuple[str, str]] = []
        new_extras: list[dict[str, Any]] = []
        for item in items:
            key = item["link"] or item["title"]
            if key and key in seen_set:
                continue
            new_chapters.append((item["title"] or "Untitled",
                                 item["body"] or item["title"]))
            new_extras.append({"link": item["link"]} if item["link"] else {})
            if key:
                seen.append(key)
        chapters = old_chapters + new_chapters
        extras = old_extras + new_extras
        if not chapters:
            raise LibraryError("feed yielded no chapters to ingest")
        seen = seen[-_FEED_SEEN_CAP:]
        source = {
            "source_id": existing["source_id"] if existing else new_id(),
            "kind": "feed", "origin": url, "title": book_title,
            "status": "active", "health": "ok", "failures": 0,
            "fetched_at": now,
            "last_checked": existing["last_checked"] if existing else 0.0,
            "last_error": "",
            "created_at": existing["created_at"] if existing else now,
            "meta": {"seen": seen, "book_slug": slug},
        }
        res = self._persist_book(
            slug=slug, title=book_title, author="",
            chapters=chapters, strategy="feed",
            original_bytes=data, original_name="source.xml",
            source=source, source_file=url, started=started,
            chapter_extra=extras,
        )
        if old_chapters and new_chapters:
            res.note = (f"upsert: {len(new_chapters)} new items, "
                        f"{len(chapters)} chapters total")
        return res

    # ── sources as seeds ──────────────────────────────────────────────
    def list_sources(self, status: str = "", kind: str = "",
                     ) -> list[dict[str, Any]]:
        con, _ = self._connect()
        try:
            sql = ("SELECT source_id, kind, origin, title, status, health,"
                   " failures, fetched_at, last_checked, last_error,"
                   " created_at, meta FROM sources")
            conds: list[str] = []
            params: list[Any] = []
            if status:
                conds.append("status = ?")
                params.append(status)
            if kind:
                conds.append("kind = ?")
                params.append(kind)
            if conds:
                sql += " WHERE " + " AND ".join(conds)
            sql += " ORDER BY created_at DESC"
            out = []
            for row in con.execute(sql, params).fetchall():
                try:
                    meta = json.loads(row[11] or "{}")
                except ValueError:
                    meta = {}
                out.append({
                    "source_id": row[0], "kind": row[1], "origin": row[2],
                    "title": row[3], "status": row[4], "health": row[5],
                    "failures": row[6], "fetched_at": row[7],
                    "last_checked": row[8], "last_error": row[9],
                    "created_at": row[10], "meta": meta,
                })
            return out
        finally:
            con.close()

    def check_sources(self, *, fetcher: Any = None) -> dict[str, Any]:
        """Health-check every active URL/feed source: liveness probe,
        failures counted, health ok → failing → dead at 3 consecutive
        failures. One source's failure never aborts the run (recorded
        and continued). File/learn sources have no URL to check and
        are skipped, never marked failing for it."""
        probe = fetcher or _default_probe
        con, _ = self._connect()
        try:
            rows = con.execute(
                "SELECT source_id, origin, failures FROM sources"
                " WHERE status = 'active' AND kind IN ('url', 'feed')"
                " ORDER BY origin").fetchall()
        finally:
            con.close()
        ok = failing = dead = 0
        details: list[dict[str, Any]] = []
        for source_id, origin, failures in rows:
            now = time.time()
            try:
                status = probe(origin, 15.0)
                if status >= 400:
                    raise LibraryError(f"HTTP {status}")
                health, new_failures, err = "ok", 0, ""
                ok += 1
            except Exception as exc:  # noqa: BLE001 — recorded, not raised
                new_failures = int(failures or 0) + 1
                health = "dead" if new_failures >= _HEALTH_FAIL_THRESHOLD \
                    else "failing"
                err = str(exc)[:300]
                if health == "dead":
                    dead += 1
                else:
                    failing += 1
            con, _ = self._connect()
            try:
                con.execute(
                    "UPDATE sources SET health = ?, failures = ?,"
                    " last_checked = ?, last_error = ? WHERE source_id = ?",
                    (health, new_failures, now, err, source_id))
                con.commit()
            finally:
                con.close()
            details.append({"source_id": source_id, "origin": origin,
                            "health": health, "failures": new_failures,
                            "error": err})
        return {"checked": len(rows), "ok": ok, "failing": failing,
                "dead": dead, "details": details}

    def shed_dead(self) -> dict[str, Any]:
        """Archive dead sources and their digests. Archive ≠ erase:
        passages stay (hybrid search excludes archived by default and
        includes them with include_archived=True), digests stay
        readable with status archived. Shedding is explicit — a dead
        health flag alone never archives anything."""
        con, _ = self._connect()
        try:
            dead = [r[0] for r in con.execute(
                "SELECT source_id FROM sources WHERE health = 'dead'"
                " AND status = 'active'").fetchall()]
            archived_digests = 0
            for source_id in dead:
                con.execute(
                    "UPDATE sources SET status = 'archived'"
                    " WHERE source_id = ?", (source_id,))
                cur = con.execute(
                    "UPDATE digests SET status = 'archived'"
                    " WHERE source_id = ? AND status = 'active'",
                    (source_id,))
                archived_digests += cur.rowcount
            con.commit()
        finally:
            con.close()
        return {"archived_sources": dead,
                "archived_digests": archived_digests}

    def discover(self, source_id: str = "", *, limit: int = 50,
                 ) -> list[dict[str, str]]:
        """Candidate sources found inside ingested content: bare URLs
        in chapter text plus feed item links from chapter meta.
        Excludes the source's own origin and every already-known
        origin — discovery proposes, the owner (or a caller) disposes
        by ingesting the good ones. Honest limit: links lost by
        HTML→text extraction are not recoverable from chapter text."""
        con, _ = self._connect()
        try:
            if source_id:
                row = con.execute(
                    "SELECT source_id FROM sources WHERE source_id = ?",
                    (source_id,)).fetchone()
                if row is None:
                    raise LibraryError(f"unknown source {source_id!r}")
                slugs = [r[0] for r in con.execute(
                    "SELECT slug FROM books WHERE source_id = ?",
                    (source_id,)).fetchall()]
            else:
                slugs = [r[0] for r in con.execute(
                    "SELECT slug FROM books").fetchall()]
            known = {r[0] for r in con.execute(
                "SELECT origin FROM sources").fetchall()}
            out: list[dict[str, str]] = []
            seen: set[str] = set()
            for slug in slugs:
                for (text,) in con.execute(
                        "SELECT text FROM passages WHERE book_slug = ?",
                        (slug,)).fetchall():
                    for url in _extract_urls(text):
                        if url not in known and url not in seen:
                            seen.add(url)
                            out.append({"url": url, "found_in": slug})
                            if len(out) >= limit:
                                return out
                try:
                    meta = json.loads(
                        (self.book_dir(slug) / "book.json").read_text(
                            encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                for ch in meta.get("chapters", []):
                    link = (ch.get("link") or "").strip()
                    if link and link not in known and link not in seen:
                        seen.add(link)
                        out.append({"url": link, "found_in": slug})
                        if len(out) >= limit:
                            return out
            return out
        finally:
            con.close()

    # ── structured revisable records (digests) ────────────────────────
    def _contradiction_scan(self, con: sqlite3.Connection,
                            record_id: str, claim: str, scope: str,
                            ) -> list[dict[str, Any]]:
        """Flag contradictions of this digest against other ACTIVE
        digests in the same scope plus global-scope records (the memory
        scope rule: a scoped view sees its scope + global, never
        another scope). Adapter objects satisfy detect_against — but
        note the trap: `_eligible` drops records whose kind is not
        FACT/PREFERENCE, silently. Digests are FACT-kind, always.

        Flags are DERIVED state: old flags for this record are replaced
        (the revisions table is the history, flags are the current
        verdict). Rule strategies only — no llm_fn, so no model call.
        A missing flag is not proof of agreement: the strategies fire
        on negation flips (≥2 shared 4+ letter topic words), value
        changes, and preference antonym swaps — short or subtly-worded
        disagreements pass undetected, honestly."""
        from types import SimpleNamespace

        from ..memory.base import MemoryKind
        from ..memory.contradictions import detect_against

        rows = con.execute(
            "SELECT record_id, claim FROM digests WHERE status = 'active'"
            " AND record_id != ? AND (scope = ? OR scope = '')",
            (record_id, scope)).fetchall()
        new = SimpleNamespace(id=record_id, content=claim,
                              kind=MemoryKind.FACT, metadata={})
        cands = [SimpleNamespace(id=r[0], content=r[1],
                                 kind=MemoryKind.FACT, metadata={})
                 for r in rows]
        found = detect_against(new, cands)
        con.execute("DELETE FROM digest_flags WHERE record_id = ?",
                    (record_id,))
        now = time.time()
        for c in found:
            con.execute(
                "INSERT INTO digest_flags(record_id, other_id, strategy,"
                " confidence, note, created_at) VALUES (?,?,?,?,?,?)",
                (record_id, c.older_id, c.strategy, c.confidence,
                 c.note, now))
        return [c.to_dict() for c in found]

    def create_digest(self, source_id: str, claim: str, *,
                      confidence: float = 0.5, scope: str = "",
                      check_contradictions: bool = True,
                      ) -> dict[str, Any]:
        """Digest a source into a structured record: claim + confidence
        + scope, revision 1 appended, contradiction flags computed, and
        (best-effort) a vector for hybrid retrieval. Fail-fast on bad
        input — an empty claim, an out-of-range confidence, or an
        unknown source is a LibraryError, never a silent clamp or a
        dangling record."""
        claim = (claim or "").strip()
        if not claim:
            raise LibraryError("digest claim is required")
        try:
            conf = float(confidence)
        except (TypeError, ValueError):
            raise LibraryError("digest confidence must be a number") from None
        if not 0.0 <= conf <= 1.0:
            raise LibraryError("digest confidence must be in [0, 1]")
        scope = (scope or "").strip()
        con, _ = self._connect()
        try:
            row = con.execute(
                "SELECT source_id FROM sources WHERE source_id = ?",
                (source_id,)).fetchone()
            if row is None:
                raise LibraryError(f"unknown source {source_id!r}")
            record_id = new_id()
            now = time.time()
            con.execute(
                "INSERT INTO digests(record_id, source_id, claim,"
                " confidence, scope, status, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (record_id, source_id, claim, conf, scope, "active",
                 now, now))
            con.execute(
                "INSERT INTO digest_revisions(record_id, claim,"
                " confidence, note, revised_by, revised_at)"
                " VALUES (?,?,?,?,?,?)",
                (record_id, claim, conf, "created", "owner", now))
            try:
                con.execute(
                    "INSERT INTO digest_fts(record_id, claim)"
                    " VALUES (?,?)", (record_id, claim))
            except sqlite3.OperationalError as exc:  # non-FTS5 db: no digest_fts table, overlap fallback
                record_failure("library.Library.create_digest.fts_insert", exc)
            flags = self._contradiction_scan(con, record_id, claim, scope) \
                if check_contradictions else []
            con.commit()
        finally:
            con.close()
        vec_error = self._embed_items([(f"digest:{record_id}", claim)])
        out = self.get_digest(record_id)
        out["flags"] = flags
        if vec_error:
            out["vector_error"] = vec_error
        return out

    def revise_digest(self, record_id: str, *, claim: str | None = None,
                      confidence: float | None = None, note: str = "",
                      revised_by: str = "owner") -> dict[str, Any]:
        """Revise a digest: the revision is APPENDED, the current row
        updated, contradictions re-scanned, vector replaced. The old
        version stays readable via digest_revisions() — a record is
        never silently overwritten. No-arg revise is a LibraryError."""
        if claim is None and confidence is None:
            raise LibraryError("revise needs a new claim or confidence")
        con, _ = self._connect()
        try:
            row = con.execute(
                "SELECT claim, confidence, scope FROM digests"
                " WHERE record_id = ?", (record_id,)).fetchone()
            if row is None:
                raise LibraryError(f"unknown digest {record_id!r}")
            new_claim = row[0] if claim is None else claim.strip()
            if not new_claim:
                raise LibraryError("digest claim is required")
            new_conf = row[1] if confidence is None else float(confidence)
            if not 0.0 <= new_conf <= 1.0:
                raise LibraryError("digest confidence must be in [0, 1]")
            now = time.time()
            con.execute(
                "INSERT INTO digest_revisions(record_id, claim,"
                " confidence, note, revised_by, revised_at)"
                " VALUES (?,?,?,?,?,?)",
                (record_id, new_claim, new_conf, note or "",
                 revised_by or "owner", now))
            con.execute(
                "UPDATE digests SET claim = ?, confidence = ?,"
                " updated_at = ? WHERE record_id = ?",
                (new_claim, new_conf, now, record_id))
            try:
                rows = con.execute(
                    "SELECT rowid FROM digest_fts WHERE record_id = ?",
                    (record_id,)).fetchall()
                for (rid,) in rows:
                    con.execute("DELETE FROM digest_fts WHERE rowid = ?",
                                (rid,))
                con.execute(
                    "INSERT INTO digest_fts(record_id, claim)"
                    " VALUES (?,?)", (record_id, new_claim))
            except sqlite3.OperationalError as exc:  # non-FTS5 db
                record_failure("library.Library.revise_digest.fts_replace", exc)
            flags = self._contradiction_scan(con, record_id, new_claim,
                                             row[2] or "")
            con.commit()
        finally:
            con.close()
        vec_error = self._replace_digest_vector(record_id, new_claim)
        out = self.get_digest(record_id)
        out["flags"] = flags
        if vec_error:
            out["vector_error"] = vec_error
        return out

    def get_digest(self, record_id: str) -> dict[str, Any]:
        con, _ = self._connect()
        try:
            row = con.execute(
                "SELECT record_id, source_id, claim, confidence, scope,"
                " status, created_at, updated_at FROM digests"
                " WHERE record_id = ?", (record_id,)).fetchone()
        finally:
            con.close()
        if row is None:
            raise LibraryError(f"unknown digest {record_id!r}")
        return {"record_id": row[0], "source_id": row[1], "claim": row[2],
                "confidence": row[3], "scope": row[4], "status": row[5],
                "created_at": row[6], "updated_at": row[7]}

    def list_digests(self, source_id: str = "", scope: str = "",
                     status: str = "active") -> list[dict[str, Any]]:
        con, _ = self._connect()
        try:
            sql = ("SELECT record_id, source_id, claim, confidence, scope,"
                   " status, created_at, updated_at FROM digests")
            conds: list[str] = []
            params: list[Any] = []
            if source_id:
                conds.append("source_id = ?")
                params.append(source_id)
            if scope:
                conds.append("(scope = ? OR scope = '')")
                params.append(scope)
            if status:
                conds.append("status = ?")
                params.append(status)
            if conds:
                sql += " WHERE " + " AND ".join(conds)
            sql += " ORDER BY updated_at DESC"
            return [{
                "record_id": r[0], "source_id": r[1], "claim": r[2],
                "confidence": r[3], "scope": r[4], "status": r[5],
                "created_at": r[6], "updated_at": r[7],
            } for r in con.execute(sql, params).fetchall()]
        finally:
            con.close()

    def digest_revisions(self, record_id: str) -> list[dict[str, Any]]:
        self.get_digest(record_id)  # fail fast on unknown ids
        con, _ = self._connect()
        try:
            return [{
                "id": r[0], "claim": r[1], "confidence": r[2],
                "note": r[3], "revised_by": r[4], "revised_at": r[5],
            } for r in con.execute(
                "SELECT id, claim, confidence, note, revised_by,"
                " revised_at FROM digest_revisions WHERE record_id = ?"
                " ORDER BY id", (record_id,)).fetchall()]
        finally:
            con.close()

    def digest_flags(self, record_id: str) -> list[dict[str, Any]]:
        self.get_digest(record_id)  # fail fast on unknown ids
        con, _ = self._connect()
        try:
            return [{
                "id": r[0], "other_id": r[1], "strategy": r[2],
                "confidence": r[3], "note": r[4], "created_at": r[5],
            } for r in con.execute(
                "SELECT id, other_id, strategy, confidence, note,"
                " created_at FROM digest_flags WHERE record_id = ?"
                " ORDER BY confidence DESC", (record_id,)).fetchall()]
        finally:
            con.close()

    def ingest_learn_journal(self, char_id: str, *,
                             data_dir: str | Path | None = None,
                             min_grade: float = 0.0) -> dict[str, Any]:
        """Bridge: a character's graded journal becomes Library digests
        (character taste as structured records — the learn loop and the
        Library compound). Claim = the graded line, confidence =
        (grade+1)/2, scope = character:{char_id}. Dedupe by journal
        entry id in the source meta, so re-running ingests only new
        grades. learn.py is NOT modified — public interfaces only
        (LearnStore + export_training_data). min_grade is caller-set
        (default 0.0 = non-negative grades only); pass -1.0 for all."""
        from ..characters.learn import LearnStore

        store = LearnStore(data_dir)
        rows = store.export_training_data(char_id)
        origin = f"character:{char_id}"
        con, _ = self._connect()
        try:
            existing = self._source_for_origin(con, "learn", origin)
        finally:
            con.close()
        now = time.time()
        source = {
            "source_id": existing["source_id"] if existing else new_id(),
            "kind": "learn", "origin": origin, "title": f"learned: {char_id}",
            "status": "active", "health": "unknown", "failures": 0,
            "fetched_at": now,
            "last_checked": existing["last_checked"] if existing else 0.0,
            "last_error": "",
            "created_at": existing["created_at"] if existing else now,
            "meta": dict(existing["meta"]) if existing else {},
        }
        seen = set((source["meta"] or {}).get("entries") or [])
        # The source row must exist BEFORE the first digest: create_digest
        # validates its source, and the meta update at the end only
        # refreshes the seen set.
        con, _ = self._connect()
        try:
            con.execute(
                "INSERT OR REPLACE INTO sources(source_id,kind,origin,"
                "title,status,health,failures,fetched_at,last_checked,"
                "last_error,created_at,meta) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (source["source_id"], "learn", origin, source["title"],
                 "active", "unknown", 0, now,
                 source["last_checked"], "", source["created_at"],
                 json.dumps(source["meta"])))
            con.commit()
        finally:
            con.close()
        ingested = skipped = 0
        for row in rows:
            entry_id = row.get("entry_id", "")
            line = (row.get("line") or "").strip()
            try:
                label = float(row.get("label"))
            except (TypeError, ValueError):
                skipped += 1
                continue
            if entry_id in seen or not line or label < min_grade:
                skipped += 1
                continue
            self.create_digest(
                source["source_id"], line,
                confidence=(label + 1.0) / 2.0,
                scope=f"character:{char_id}")
            seen.add(entry_id)
            ingested += 1
        source["meta"]["entries"] = sorted(seen)[-5000:]
        con, _ = self._connect()
        try:
            con.execute(
                "INSERT OR REPLACE INTO sources(source_id,kind,origin,"
                "title,status,health,failures,fetched_at,last_checked,"
                "last_error,created_at,meta) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (source["source_id"], "learn", origin, source["title"],
                 "active", "unknown", 0, now,
                 source["last_checked"], "", source["created_at"],
                 json.dumps(source["meta"])))
            con.commit()
        finally:
            con.close()
        return {"ingested": ingested, "skipped": skipped,
                "source_id": source["source_id"],
                "scope": f"character:{char_id}"}

    # ── vector lane (the memory system's store, in the library file) ─
    def _get_embedder(self) -> Any:
        if self.embedder is None:
            from ..memory.embeddings import Embedder

            # Hashing default: deterministic, zero deps, zero cost,
            # never raises. Native/router embedders are caller-injected
            # and degrade to hashing inside Embedder itself.
            self.embedder = Embedder(provider="hashing", dimensions=512)
        return self.embedder

    @contextmanager
    def _vector_backend(self) -> Iterator[Any]:
        """The memory system's vector lane against library.db: the real
        VectorStore + select_vector_backend(auto) — sqlite-vec →
        usearch → legacy — with owner_type 'library'. Database(wal=
        False) leaves the file's journal mode untouched, and the
        Database is closed on exit (no held connections, no WAL
        sidecars). Never import-time: all three imports are lazy."""
        from ..memory.vector_backends import select_vector_backend
        from ..storage.db import Database
        from ..storage.vectors import VectorStore

        db = Database(self.db_path(), wal=False)
        try:
            store = VectorStore(db)
            backend = select_vector_backend(
                db, preference="auto", owner_type=_VECTOR_OWNER,
                vectors=store)
            yield backend
        finally:
            db.close()

    def _pinned_dims(self) -> int | None:
        con, _ = self._connect()
        try:
            row = con.execute(
                "SELECT value FROM meta WHERE key = 'vector_dims'"
                ).fetchone()
            return int(row[0]) if row else None
        finally:
            con.close()

    def _check_pin_dims(self, dims: int, *, for_write: bool) -> str:
        """Pin the vector dimensionality at first WRITE (later writers
        with different dims fail fast with a clear error — the store
        itself raises on mixed dims at search, this is earlier and
        clearer). Reads never pin and never raise: a mismatch on the
        read path degrades the vector lane instead."""
        con, _ = self._connect()
        try:
            row = con.execute(
                "SELECT value FROM meta WHERE key = 'vector_dims'"
                ).fetchone()
            if row is None:
                if for_write:
                    con.execute(
                        "INSERT INTO meta(key, value) VALUES"
                        " ('vector_dims', ?)", (str(dims),))
                    con.commit()
                return ""
            if int(row[0]) != dims:
                return (f"embedder dimensions {dims} do not match pinned"
                        f" vector dimensions {row[0]}")
            return ""
        finally:
            con.close()

    def _embed_items(self, items: list[tuple[str, str]]) -> str:
        """Embed (owner_id, text) pairs into the vector lane. Returns
        "" on success or an error string — embedding never fails the
        caller (ingest/digest succeed; the vector is repaired by
        backfill_vectors)."""
        if not items:
            return ""
        try:
            embedder = self._get_embedder()
            err = self._check_pin_dims(embedder.dimensions, for_write=True)
            if err:
                return err
            texts = [text[:_EMBED_CHARS] for _oid, text in items]
            vectors = embedder.embed_many(texts)
            with self._vector_backend() as backend:
                backend.put_many([(v, oid)
                                  for v, (oid, _t) in zip(vectors, items)])
            return ""
        except Exception as exc:  # noqa: BLE001 — best-effort lane
            return str(exc)[:300]

    def _replace_digest_vector(self, record_id: str, claim: str) -> str:
        """Replace a digest's vector: compute first, then delete +
        put, so a failed embed leaves the old vector alone only when
        the embed itself failed — when the embed succeeds the old
        vector is replaced, never duplicated (VectorStore.put always
        inserts a new row; delete_owner is the only update)."""
        try:
            embedder = self._get_embedder()
            err = self._check_pin_dims(embedder.dimensions, for_write=True)
            if err:
                return err
            vector = embedder.embed(claim[:_EMBED_CHARS])
            with self._vector_backend() as backend:
                backend.delete_owner(f"digest:{record_id}")
                backend.put(vector, f"digest:{record_id}")
            return ""
        except Exception as exc:  # noqa: BLE001 — best-effort lane
            return str(exc)[:300]

    def _embed_book_vectors(self, slug: str,
                            chapters: list[tuple[str, str]]) -> str:
        """Embed a book's chapters. Stale vectors for the slug are
        cleared first (re-ingest may shrink chapters — a vector for a
        chapter that no longer exists must not haunt search)."""
        if not chapters:
            return ""
        try:
            embedder = self._get_embedder()
            err = self._check_pin_dims(embedder.dimensions, for_write=True)
            if err:
                return err
            from ..storage.db import Database

            db = Database(self.db_path(), wal=False)
            try:
                stale = [r["owner_id"] for r in db.query(
                    "SELECT owner_id FROM embeddings WHERE owner_type = ?"
                    " AND owner_id LIKE ?",
                    (_VECTOR_OWNER, f"passage:{slug}:%"))]
            finally:
                db.close()
            items = [(f"passage:{slug}:{i}", body)
                     for i, (_name, body) in enumerate(chapters, 1)]
            texts = [body[:_EMBED_CHARS] for _oid, body in items]
            vectors = embedder.embed_many(texts)
            with self._vector_backend() as backend:
                for oid in stale:
                    backend.delete_owner(oid)
                backend.put_many([(v, oid)
                                  for v, (oid, _t) in zip(vectors, items)])
            return ""
        except Exception as exc:  # noqa: BLE001 — best-effort lane
            return str(exc)[:300]

    def _delete_vectors(self, owner_ids: list[str]) -> None:
        if not owner_ids:
            return
        with self._vector_backend() as backend:
            for oid in owner_ids:
                backend.delete_owner(oid)

    def _delete_book_vectors(self, slug: str, numbers: list[int]) -> None:
        self._delete_vectors([f"passage:{slug}:{n}" for n in numbers])

    def vector_status(self) -> dict[str, Any]:
        """The vector lane, reported honestly: backend, embedder,
        pinned dims, counts, fallbacks, unvectorized digests. A failed
        backend reports 'unavailable' with the reason — never a fake
        count, never a raise out of a status call."""
        embedder = self._get_embedder()
        stats = getattr(embedder, "stats", {}) or {}
        out: dict[str, Any] = {
            "embedder_provider": getattr(embedder, "provider", "?"),
            "embedder_dims": getattr(embedder, "dimensions", 0),
            "pinned_dims": self._pinned_dims(),
            "fallbacks": stats.get("fallbacks", 0),
        }
        try:
            with self._vector_backend() as backend:
                out["backend"] = getattr(backend, "name", "?")
                out["vector_count"] = backend.count()
        except Exception as exc:  # noqa: BLE001 — status never raises
            out["backend"] = "unavailable"
            out["backend_error"] = str(exc)[:300]
            out["vector_count"] = 0
        try:
            from ..storage.db import Database

            db = Database(self.db_path(), wal=False)
            try:
                vectored = {r["owner_id"] for r in db.query(
                    "SELECT owner_id FROM embeddings WHERE owner_type = ?"
                    " AND owner_id LIKE 'digest:%'", (_VECTOR_OWNER,))}
            finally:
                db.close()
        except Exception as exc:  # noqa: BLE001 — status never raises
            record_failure("library.Library.vector_status.vectored", exc)
            vectored = set()
        con, _ = self._connect()
        try:
            active = [r[0] for r in con.execute(
                "SELECT record_id FROM digests"
                " WHERE status = 'active'").fetchall()]
        finally:
            con.close()
        out["unvectorized_digests"] = len(
            [d for d in active if f"digest:{d}" not in vectored])
        return out

    def backfill_vectors(self, *, max_items: int = 0) -> dict[str, Any]:
        """Embed every passage + active digest lacking a vector (books
        ingested before the lane existed, digests whose embed failed).
        Digests first (small, high-value), then passages. max_items=0
        means all; otherwise a hard cap on items processed. Errors are
        collected per batch, never raised — backfill is a repair path."""
        try:
            from ..storage.db import Database

            db = Database(self.db_path(), wal=False)
            try:
                existing = {r["owner_id"] for r in db.query(
                    "SELECT owner_id FROM embeddings WHERE owner_type = ?",
                    (_VECTOR_OWNER,))}
            finally:
                db.close()
        except Exception as exc:  # noqa: BLE001 — no table yet means none exist
            record_failure("library.Library.backfill_vectors.existing", exc)
            existing = set()
        missing_digests: list[tuple[str, str]] = []
        for d in self.list_digests(status=""):
            if d["status"] != "active":
                continue
            if f"digest:{d['record_id']}" not in existing:
                missing_digests.append((f"digest:{d['record_id']}",
                                        d["claim"]))
        missing_passages: list[tuple[str, str]] = []
        for book_json in sorted(self.root().glob("*/book.json")):
            slug = book_json.parent.name
            try:
                meta = json.loads(book_json.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for ch in meta.get("chapters", []):
                num = int(ch.get("number", 0))
                oid = f"passage:{slug}:{num}"
                if oid in existing:
                    continue
                path = book_json.parent / "chapters" / f"{num:03d}.md"
                try:
                    text = path.read_text(encoding="utf-8")
                except OSError:
                    continue
                lines = text.split("\n", 1)
                body = lines[1].lstrip("\n") if len(lines) > 1 and \
                    lines[0].startswith("# ") else text
                missing_passages.append((oid, body))
        errors: list[str] = []
        done_d = done_p = 0
        budget = max_items if max_items and max_items > 0 else None
        for batch_start in range(0, len(missing_digests), 50):
            if budget is not None and done_d + done_p >= budget:
                break
            batch = missing_digests[batch_start:batch_start + 50]
            if budget is not None:
                batch = batch[:budget - done_d - done_p]
            err = self._embed_items(batch)
            if err:
                errors.append(err)
            else:
                done_d += len(batch)
        for batch_start in range(0, len(missing_passages), 50):
            if budget is not None and done_d + done_p >= budget:
                break
            batch = missing_passages[batch_start:batch_start + 50]
            if budget is not None:
                batch = batch[:budget - done_d - done_p]
            err = self._embed_items(batch)
            if err:
                errors.append(err)
            else:
                done_p += len(batch)
        return {"passages": done_p, "digests": done_d, "errors": errors}

    # ── hybrid retrieval ──────────────────────────────────────────────
    def _search_digests_keyword(self, query: str, *, scope: str,
                                include_archived: bool, limit: int,
                                ) -> list[dict[str, Any]]:
        """Keyword lane over digests: FTS5 MATCH when available, else
        Python token overlap. Scope rule: a scoped query sees its
        scope PLUS global records, never another scope's."""
        con, fts5 = self._connect()
        try:
            ids: list[str] = []
            if fts5:
                try:
                    fts_q = self._sanitize_fts(query)
                except LibraryError:
                    return []
                try:
                    rows = con.execute(
                        "SELECT record_id FROM digest_fts"
                        " WHERE digest_fts MATCH ? LIMIT ?",
                        (fts_q, limit * 4)).fetchall()
                except sqlite3.OperationalError:
                    rows = []  # table predates this build; overlap below
                ids = [r[0] for r in rows]
            if not ids and not fts5:
                qtok = {w.lower() for w in _WORD.findall(query)
                        if len(w) >= 2}
                scored: list[tuple[int, str]] = []
                for rid, claim in con.execute(
                        "SELECT record_id, claim FROM digests").fetchall():
                    ctok = {w.lower() for w in _WORD.findall(claim or "")}
                    ov = len(qtok & ctok)
                    if ov:
                        scored.append((ov, rid))
                scored.sort(key=lambda x: -x[0])
                ids = [rid for _ov, rid in scored[:limit * 4]]
            out = []
            for rid in ids:
                row = con.execute(
                    "SELECT record_id, source_id, claim, confidence,"
                    " scope, status FROM digests WHERE record_id = ?",
                    (rid,)).fetchone()
                if row is None:
                    continue
                if not include_archived and row[5] != "active":
                    continue
                if scope and row[4] not in (scope, ""):
                    continue
                out.append({
                    "record_id": row[0], "source_id": row[1],
                    "claim": row[2], "confidence": row[3],
                    "scope": row[4], "status": row[5],
                })
                if len(out) >= limit:
                    break
            return out
        finally:
            con.close()

    def _passage_text_for(self, slug: str, number: int, query: str,
                          ) -> str:
        """Context window for a passage the keyword lane did not
        return (vector-only hit): chapter file → window around the
        query words. Empty on any failure — a vector hit without
        resolvable text is skipped, never faked."""
        try:
            meta = json.loads(
                (self.book_dir(slug) / "book.json").read_text(
                    encoding="utf-8"))
            ch = next((c for c in meta.get("chapters", [])
                       if int(c.get("number", 0)) == number), None)
            if ch is None:
                return ""
            path = self.book_dir(slug) / "chapters" / f"{number:03d}.md"
            text = path.read_text(encoding="utf-8")
            return self._context_window(text, query)
        except (OSError, ValueError):
            return ""

    def search_hybrid(self, query: str, *, top: int = 8, scope: str = "",
                      include_archived: bool = False,
                      vector_top: int = 20) -> dict[str, Any]:
        """Hybrid retrieval: FTS5/BM25 keyword + the memory system's
        vector lane, fused by reciprocal rank fusion (k=60) over three
        lanes — keyword-passages, keyword-digests, vector — because
        FTS5 negative scores and BM25 positive scores must never be
        mixed arithmetically; ranks are scale-free. Final score = RRF
        × source-health factor × (0.5 + 0.5×confidence for digests).
        Scope narrows DIGESTS only (passages are shared deep text).
        A failed vector lane degrades (named in lanes.degraded) and
        keyword hits still serve."""
        query = (query or "").strip()
        if not query:
            return {"hits": [], "lanes": {"keyword_passages": 0,
                                          "keyword_digests": 0,
                                          "vector": 0, "degraded": []}}
        # Source health map for exclusion + weighting.
        con, _ = self._connect()
        try:
            books_map = {r[0]: r[1] for r in con.execute(
                "SELECT slug, source_id FROM books").fetchall()}
            sources_map = {r[0]: {"status": r[1], "health": r[2]}
                           for r in con.execute(
                               "SELECT source_id, status, health"
                               " FROM sources").fetchall()}
        finally:
            con.close()

        def health_of(source_id: str | None) -> tuple[str, float, bool]:
            """(health label, factor, excluded-by-default)."""
            if not source_id or source_id not in sources_map:
                return "unknown", 1.0, False
            s = sources_map[source_id]
            if s["status"] == "archived" or s["health"] == "dead":
                return s["health"], 0.0, True
            if s["health"] == "failing":
                return "failing", 0.7, False
            return s["health"] or "unknown", 1.0, False

        kw_n = max(top * 3, vector_top)
        kw_passages = self.search(query, top=kw_n)
        kw_digests = self._search_digests_keyword(
            query, scope=scope, include_archived=include_archived,
            limit=kw_n)
        kw_pass_rank: dict[tuple[str, int], int] = {}
        for rank, hit in enumerate(kw_passages):
            kw_pass_rank[(hit.book_slug, hit.chapter_number)] = rank
        kw_dig_rank: dict[str, int] = {}
        for rank, d in enumerate(kw_digests):
            kw_dig_rank[d["record_id"]] = rank

        # Vector lane: best-effort, never fatal to search.
        vec_rank: dict[str, int] = {}
        degraded: list[str] = []
        try:
            embedder = self._get_embedder()
            dim_err = self._check_pin_dims(embedder.dimensions,
                                           for_write=False)
            if dim_err:
                degraded.append(dim_err)
            else:
                vector = embedder.embed(query)
                with self._vector_backend() as backend:
                    for rank, hit in enumerate(backend.search(
                            vector, limit=vector_top)):
                        vec_rank[hit.owner_id] = rank
        except Exception as exc:  # noqa: BLE001 — degraded, not fatal
            degraded.append(f"vector lane unavailable: {str(exc)[:200]}")

        def rrf(rank: int | None) -> float:
            return 1.0 / (_RRF_K + rank) if rank is not None else 0.0

        hits: list[HybridHit] = []
        for hit in kw_passages:
            health, factor, excluded = health_of(
                books_map.get(hit.book_slug))
            if excluded and not include_archived:
                continue
            key = (hit.book_slug, hit.chapter_number)
            lanes = ["keyword"]
            if f"passage:{hit.book_slug}:{hit.chapter_number}" in vec_rank:
                lanes.append("vector")
            score = (rrf(kw_pass_rank[key])
                     + rrf(vec_rank.get(
                         f"passage:{hit.book_slug}:{hit.chapter_number}"))) \
                * factor
            hits.append(HybridHit(
                kind="passage", text=hit.passage, score=score,
                title=hit.title, slug=hit.book_slug,
                chapter=hit.chapter, chapter_number=hit.chapter_number,
                health=health, source_id=books_map.get(hit.book_slug, ""),
                keyword_rank=kw_pass_rank[key],
                vector_rank=vec_rank.get(
                    f"passage:{hit.book_slug}:{hit.chapter_number}"),
                lanes=tuple(lanes)))
        for d in kw_digests:
            health, factor, excluded = health_of(d["source_id"])
            if excluded and not include_archived:
                continue
            lanes = ["keyword"]
            if f"digest:{d['record_id']}" in vec_rank:
                lanes.append("vector")
            score = (rrf(kw_dig_rank[d["record_id"]])
                     + rrf(vec_rank.get(f"digest:{d['record_id']}"))) \
                * factor * (0.5 + 0.5 * float(d["confidence"] or 0.0))
            hits.append(HybridHit(
                kind="digest", text=d["claim"], score=score,
                record_id=d["record_id"], confidence=d["confidence"],
                health=health, source_id=d["source_id"],
                scope=d["scope"],
                keyword_rank=kw_dig_rank[d["record_id"]],
                vector_rank=vec_rank.get(f"digest:{d['record_id']}"),
                lanes=tuple(lanes)))
        for owner_id, rank in vec_rank.items():
            if owner_id.startswith("passage:"):
                parts = owner_id.split(":")
                if len(parts) != 3:
                    continue
                slug, num_s = parts[1], parts[2]
                try:
                    num = int(num_s)
                except ValueError:
                    continue
                if (slug, num) in kw_pass_rank:
                    continue  # already emitted above
                health, factor, excluded = health_of(books_map.get(slug))
                if excluded and not include_archived:
                    continue
                text = self._passage_text_for(slug, num, query)
                if not text:
                    continue  # dropped book's stale vector: skip, not fake
                meta = None
                try:
                    meta = json.loads(
                        (self.book_dir(slug) / "book.json").read_text(
                            encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                ch = next((c for c in meta.get("chapters", [])
                           if int(c.get("number", 0)) == num), None)
                hits.append(HybridHit(
                    kind="passage", text=text, score=rrf(rank) * factor,
                    title=meta.get("title", ""), slug=slug,
                    chapter=ch.get("title", "") if ch else "",
                    chapter_number=num, health=health,
                    source_id=books_map.get(slug, ""),
                    vector_rank=rank, lanes=("vector",)))
            elif owner_id.startswith("digest:"):
                record_id = owner_id.split(":", 1)[1]
                if record_id in kw_dig_rank:
                    continue
                try:
                    d = self.get_digest(record_id)
                except LibraryError:
                    continue
                if not include_archived and d["status"] != "active":
                    continue
                if scope and d["scope"] not in (scope, ""):
                    continue
                health, factor, excluded = health_of(d["source_id"])
                if excluded and not include_archived:
                    continue
                hits.append(HybridHit(
                    kind="digest", text=d["claim"],
                    score=rrf(rank) * factor
                    * (0.5 + 0.5 * float(d["confidence"] or 0.0)),
                    record_id=record_id, confidence=d["confidence"],
                    health=health, source_id=d["source_id"],
                    scope=d["scope"], vector_rank=rank,
                    lanes=("vector",)))
        hits.sort(key=lambda h: -h.score)
        return {
            "hits": [h.to_dict() for h in hits[:top]],
            "lanes": {"keyword_passages": len(kw_pass_rank),
                      "keyword_digests": len(kw_dig_rank),
                      "vector": len(vec_rank), "degraded": degraded},
        }

    # ── reading progress ──────────────────────────────────────────────
    def set_progress(self, slug: str, chapter: int, offset_chars: int = 0) -> dict[str, Any]:
        """Remember where the owner left off: chapter + char offset."""
        b = self.load(slug)  # fail fast on unknown slugs
        total = max(1, len(b["chapters"]))
        chapter = max(1, min(int(chapter), total))
        offset_chars = max(0, int(offset_chars))
        con, _ = self._connect()
        try:
            con.execute(
                "INSERT OR REPLACE INTO reading_progress"
                "(slug, chapter, offset_chars, updated_at) VALUES (?,?,?,?)",
                (slug, chapter, offset_chars, time.time()))
            con.commit()
        finally:
            con.close()
        return self.get_progress(slug)

    def get_progress(self, slug: str) -> dict[str, Any]:
        b = self.load(slug)
        total = len(b["chapters"])
        con, _ = self._connect()
        try:
            row = con.execute(
                "SELECT chapter, offset_chars, updated_at FROM reading_progress"
                " WHERE slug = ?", (slug,)).fetchone()
        finally:
            con.close()
        if row is None:
            return {"slug": slug, "title": b["title"], "started": False,
                    "chapter": 0, "chapter_title": "", "offset_chars": 0,
                    "percent": 0.0, "chapters": total, "updated_at": 0.0}
        ch_no = max(1, min(int(row[0]), max(1, total)))
        title = b["chapters"][ch_no - 1]["title"] if 1 <= ch_no <= total else ""
        return {"slug": slug, "title": b["title"], "started": True,
                "chapter": ch_no, "chapter_title": title,
                "offset_chars": int(row[1]),
                "percent": round(100.0 * ch_no / max(1, total), 1),
                "chapters": total, "updated_at": row[2]}

    def resume(self, slug: str) -> dict[str, Any]:
        """Where you left off, plus the next chapter to read."""
        prog = self.get_progress(slug)
        b = self.load(slug)
        total = len(b["chapters"])
        if not prog["started"]:
            nxt: dict[str, Any] = {"number": 1,
                                   "title": b["chapters"][0]["title"] if total else ""}
            hint = f"not started — begin at chapter 1: {nxt['title']!r}"
        elif prog["chapter"] >= total:
            nxt = {"number": total, "title": b["chapters"][-1]["title"]}
            hint = "finished — last chapter read"
        else:
            ch = b["chapters"][prog["chapter"]]
            nxt = {"number": ch["number"], "title": ch["title"],
                   "words": ch["words"]}
            hint = (f"continue at chapter {ch['number']}: {ch['title']!r} "
                    f"({prog['percent']}% through)")
        return {"progress": prog, "next": nxt, "hint": hint}

    # ── bookmarks ─────────────────────────────────────────────────────
    def _check_chapter(self, slug: str, chapter: int) -> dict[str, Any]:
        b = self.load(slug)
        total = len(b["chapters"])
        chapter = int(chapter)
        if not 1 <= chapter <= total:
            raise LibraryError(
                f"chapter {chapter} out of range for {slug!r} (1–{total})")
        return b

    def add_bookmark(self, slug: str, chapter: int, offset_chars: int = 0,
                     label: str = "") -> dict[str, Any]:
        self._check_chapter(slug, chapter)
        offset_chars = max(0, int(offset_chars))
        con, _ = self._connect()
        try:
            cur = con.execute(
                "INSERT INTO bookmarks(slug, chapter, offset_chars, label, created_at)"
                " VALUES (?,?,?,?,?)",
                (slug, int(chapter), offset_chars, label.strip()[:200],
                 time.time()))
            con.commit()
            bid = cur.lastrowid
        finally:
            con.close()
        return {"id": bid, "slug": slug, "chapter": int(chapter),
                "offset_chars": offset_chars, "label": label.strip()}

    def list_bookmarks(self, slug: str = "") -> list[dict[str, Any]]:
        con, _ = self._connect()
        try:
            if slug:
                rows = con.execute(
                    "SELECT id, slug, chapter, offset_chars, label, created_at"
                    " FROM bookmarks WHERE slug = ? ORDER BY chapter, offset_chars",
                    (slug,)).fetchall()
            else:
                rows = con.execute(
                    "SELECT id, slug, chapter, offset_chars, label, created_at"
                    " FROM bookmarks ORDER BY created_at DESC").fetchall()
        finally:
            con.close()
        return [{"id": r[0], "slug": r[1], "chapter": r[2],
                 "offset_chars": r[3], "label": r[4], "created_at": r[5]}
                for r in rows]

    def remove_bookmark(self, bookmark_id: int) -> dict[str, Any]:
        con, _ = self._connect()
        try:
            cur = con.execute("DELETE FROM bookmarks WHERE id = ?",
                              (int(bookmark_id),))
            con.commit()
            if cur.rowcount == 0:
                raise LibraryError(f"no bookmark {bookmark_id}")
        finally:
            con.close()
        return {"removed": int(bookmark_id)}

    # ── annotations / notes ───────────────────────────────────────────
    def add_note(self, slug: str, chapter: int, offset_chars: int = 0,
                 quote: str = "", note: str = "") -> dict[str, Any]:
        self._check_chapter(slug, chapter)
        note = (note or "").strip()
        if not note:
            raise LibraryError("note text is required")
        con, _ = self._connect()
        try:
            cur = con.execute(
                "INSERT INTO annotations(slug, chapter, offset_chars, quote, note, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (slug, int(chapter), max(0, int(offset_chars)),
                 (quote or "").strip()[:2000], note[:4000], time.time()))
            con.commit()
            nid = cur.lastrowid
        finally:
            con.close()
        return {"id": nid, "slug": slug, "chapter": int(chapter),
                "offset_chars": max(0, int(offset_chars)),
                "quote": (quote or "").strip()[:2000], "note": note}

    def list_notes(self, slug: str = "") -> list[dict[str, Any]]:
        con, _ = self._connect()
        try:
            if slug:
                rows = con.execute(
                    "SELECT id, slug, chapter, offset_chars, quote, note, created_at"
                    " FROM annotations WHERE slug = ? ORDER BY chapter, offset_chars",
                    (slug,)).fetchall()
            else:
                rows = con.execute(
                    "SELECT id, slug, chapter, offset_chars, quote, note, created_at"
                    " FROM annotations ORDER BY created_at DESC").fetchall()
        finally:
            con.close()
        return [{"id": r[0], "slug": r[1], "chapter": r[2],
                 "offset_chars": r[3], "quote": r[4], "note": r[5],
                 "created_at": r[6]} for r in rows]

    def remove_note(self, note_id: int) -> dict[str, Any]:
        con, _ = self._connect()
        try:
            cur = con.execute("DELETE FROM annotations WHERE id = ?",
                              (int(note_id),))
            con.commit()
            if cur.rowcount == 0:
                raise LibraryError(f"no note {note_id}")
        finally:
            con.close()
        return {"removed": int(note_id)}

    # ── collections / shelves ─────────────────────────────────────────
    @staticmethod
    def _clean_collection(name: str) -> str:
        name = (name or "").strip()[:80]
        if not name:
            raise LibraryError("collection name is required")
        return name

    def create_collection(self, name: str) -> dict[str, Any]:
        name = self._clean_collection(name)
        con, _ = self._connect()
        try:
            cur = con.execute(
                "INSERT OR IGNORE INTO collections(name, created_at) VALUES (?,?)",
                (name, time.time()))
            con.commit()
            created = cur.rowcount > 0
        finally:
            con.close()
        return {"collection": name, "created": created}

    def delete_collection(self, name: str) -> dict[str, Any]:
        name = self._clean_collection(name)
        con, _ = self._connect()
        try:
            cur = con.execute("DELETE FROM collections WHERE name = ?", (name,))
            con.execute("DELETE FROM collection_books WHERE collection = ?", (name,))
            con.commit()
            if cur.rowcount == 0:
                raise LibraryError(f"no collection {name!r}")
        finally:
            con.close()
        return {"deleted": name}

    def add_to_collection(self, name: str, slug: str) -> dict[str, Any]:
        name = self._clean_collection(name)
        self.load(slug)  # fail fast on unknown books
        con, _ = self._connect()
        try:
            row = con.execute("SELECT 1 FROM collections WHERE name = ?",
                              (name,)).fetchone()
            if row is None:
                raise LibraryError(
                    f"no collection {name!r} — create it first")
            con.execute(
                "INSERT OR IGNORE INTO collection_books(collection, slug, added_at)"
                " VALUES (?,?,?)", (name, slug, time.time()))
            con.commit()
        finally:
            con.close()
        return {"collection": name, "added": slug}

    def remove_from_collection(self, name: str, slug: str) -> dict[str, Any]:
        name = self._clean_collection(name)
        con, _ = self._connect()
        try:
            cur = con.execute(
                "DELETE FROM collection_books WHERE collection = ? AND slug = ?",
                (name, slug))
            con.commit()
            if cur.rowcount == 0:
                raise LibraryError(
                    f"{slug!r} is not in collection {name!r}")
        finally:
            con.close()
        return {"collection": name, "removed": slug}

    def list_collections(self) -> list[dict[str, Any]]:
        con, _ = self._connect()
        try:
            rows = con.execute(
                "SELECT c.name, c.created_at, COUNT(cb.slug)"
                " FROM collections c LEFT JOIN collection_books cb"
                " ON cb.collection = c.name"
                " GROUP BY c.name ORDER BY c.name").fetchall()
        finally:
            con.close()
        return [{"name": r[0], "books": r[2], "created_at": r[1]} for r in rows]

    def shelf(self, name: str) -> dict[str, Any]:
        """Every book on a collection shelf, with progress + rating."""
        name = self._clean_collection(name)
        con, _ = self._connect()
        try:
            if con.execute("SELECT 1 FROM collections WHERE name = ?",
                           (name,)).fetchone() is None:
                raise LibraryError(f"no collection {name!r}")
            slugs = [r[0] for r in con.execute(
                "SELECT slug FROM collection_books WHERE collection = ?"
                " ORDER BY added_at", (name,)).fetchall()]
        finally:
            con.close()
        books = []
        for slug in slugs:
            try:
                meta = self.load(slug)
            except LibraryError:
                continue
            prog = self.get_progress(slug)
            books.append({"slug": slug, "title": meta.get("title", ""),
                          "author": meta.get("author", ""),
                          "words": meta.get("total_words", 0),
                          "rating": self.get_rating(slug),
                          "tags": self.get_tags(slug),
                          "progress_percent": prog["percent"]})
        return {"collection": name, "books": books}

    # ── tags ──────────────────────────────────────────────────────────
    @staticmethod
    def _clean_tags(tags: list[str] | str) -> list[str]:
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]
        out = []
        for t in tags or []:
            t = str(t).strip().lower()[:40]
            if t and t not in out:
                out.append(t)
        return out

    def set_tags(self, slug: str, tags: list[str] | str) -> dict[str, Any]:
        """Replace the book's tags (empty list clears them)."""
        self.load(slug)
        tags = self._clean_tags(tags)
        con, _ = self._connect()
        try:
            con.execute("DELETE FROM book_tags WHERE slug = ?", (slug,))
            con.executemany("INSERT INTO book_tags(slug, tag) VALUES (?,?)",
                            [(slug, t) for t in tags])
            con.commit()
        finally:
            con.close()
        return {"slug": slug, "tags": tags}

    def get_tags(self, slug: str) -> list[str]:
        self.load(slug)
        con, _ = self._connect()
        try:
            rows = con.execute(
                "SELECT tag FROM book_tags WHERE slug = ? ORDER BY tag",
                (slug,)).fetchall()
        finally:
            con.close()
        return [r[0] for r in rows]

    def list_tags(self) -> list[dict[str, Any]]:
        con, _ = self._connect()
        try:
            rows = con.execute(
                "SELECT tag, COUNT(slug) FROM book_tags GROUP BY tag"
                " ORDER BY COUNT(slug) DESC, tag").fetchall()
        finally:
            con.close()
        return [{"tag": r[0], "books": r[1]} for r in rows]

    def books_with_tag(self, tag: str) -> list[dict[str, Any]]:
        tag = (tag or "").strip().lower()
        if not tag:
            raise LibraryError("tag is required")
        con, _ = self._connect()
        try:
            slugs = [r[0] for r in con.execute(
                "SELECT slug FROM book_tags WHERE tag = ?", (tag,)).fetchall()]
        finally:
            con.close()
        out = []
        for slug in slugs:
            try:
                meta = self.load(slug)
            except LibraryError:
                continue
            out.append({"slug": slug, "title": meta.get("title", ""),
                        "author": meta.get("author", "")})
        return out

    # ── ratings ───────────────────────────────────────────────────────
    def rate(self, slug: str, stars: int) -> dict[str, Any]:
        self.load(slug)
        stars = int(stars)
        if not 1 <= stars <= 5:
            raise LibraryError(f"rating must be 1–5 stars, got {stars}")
        con, _ = self._connect()
        try:
            con.execute(
                "INSERT OR REPLACE INTO ratings(slug, stars, rated_at)"
                " VALUES (?,?,?)", (slug, stars, time.time()))
            con.commit()
        finally:
            con.close()
        return {"slug": slug, "stars": stars}

    def get_rating(self, slug: str) -> int:
        self.load(slug)
        con, _ = self._connect()
        try:
            row = con.execute(
                "SELECT stars FROM ratings WHERE slug = ?", (slug,)).fetchone()
        finally:
            con.close()
        return int(row[0]) if row else 0

    # ── reading statistics ────────────────────────────────────────────
    #
    # Mined from KOReader's statistics plugin: a row per reading session
    # (not a timer you start/stop manually — the session is the unit),
    # from which WPM, totals, calendar days and streaks derive.

    def start_session(self, slug: str) -> dict[str, Any]:
        """Open a reading session. Returns the session id."""
        self.load(slug)
        con, _ = self._connect()
        try:
            cur = con.execute(
                "INSERT INTO reading_sessions(slug, started_at) "
                "VALUES (?, ?)", (slug, time.time()))
            con.commit()
            sid = cur.lastrowid
        finally:
            con.close()
        return {"session_id": sid, "slug": slug}

    def end_session(self, session_id: int, *, chapters: int = 0,
                    words: int = 0) -> dict[str, Any]:
        """Close a reading session, recording duration + throughput."""
        con, _ = self._connect()
        try:
            row = con.execute(
                "SELECT slug, started_at FROM reading_sessions WHERE id = ?",
                (session_id,)).fetchone()
            if row is None:
                raise LibraryError(f"no session {session_id}")
            now = time.time()
            seconds = max(0.0, now - float(row[1]))
            con.execute(
                "UPDATE reading_sessions SET ended_at = ?, seconds = ?, "
                "chapters = ?, words = ? WHERE id = ?",
                (now, seconds, int(chapters), int(words), session_id))
            con.commit()
        finally:
            con.close()
        wpm = round(words / (seconds / 60), 1) if seconds > 5 and words else 0
        return {"session_id": session_id, "slug": row[0],
                "seconds": round(seconds, 1), "chapters": chapters,
                "words": words, "wpm": wpm}

    def stats(self, slug: str = "") -> dict[str, Any]:
        """Reading stats: sessions, minutes, WPM, streak, per-book table."""
        con, _ = self._connect()
        try:
            filt = "WHERE slug = ?" if slug else ""
            args: tuple = (slug,) if slug else ()
            if slug:
                self.load(slug)
            row = con.execute(
                f"SELECT COUNT(*), COALESCE(SUM(seconds),0), "
                f"COALESCE(SUM(words),0), COALESCE(SUM(chapters),0) "
                f"FROM reading_sessions {filt}", args).fetchone()
            sessions, seconds, words, chapters = row
            per_book = con.execute(
                "SELECT slug, COUNT(*), COALESCE(SUM(seconds),0), "
                "COALESCE(SUM(words),0) FROM reading_sessions "
                "GROUP BY slug ORDER BY SUM(seconds) DESC LIMIT 20").fetchall()
        finally:
            con.close()
        minutes = round(seconds / 60, 1)
        wpm = round(words / minutes, 1) if minutes > 0 else 0
        return {
            "slug": slug or "all",
            "sessions": sessions,
            "minutes": minutes,
            "words_read": words,
            "chapters_read": chapters,
            "wpm": wpm,
            "streak_days": self.reading_streak(),
            "per_book": [
                {"slug": s, "sessions": n, "minutes": round(sec / 60, 1),
                 "words": w}
                for s, n, sec, w in per_book],
        }

    def reading_streak(self) -> int:
        """Consecutive days (ending today/yesterday) with a session."""
        con, _ = self._connect()
        try:
            rows = con.execute(
                "SELECT DISTINCT date(started_at, 'unixepoch', 'localtime') "
                "FROM reading_sessions ORDER BY 1 DESC").fetchall()
        finally:
            con.close()
        days = [r[0] for r in rows]
        if not days:
            return 0
        import datetime as _dt
        today = _dt.date.today()
        # streak may start yesterday (haven't read yet today)
        cursor = today
        if days[0] != today.isoformat():
            if days[0] != (today - _dt.timedelta(days=1)).isoformat():
                return 0
            cursor = today - _dt.timedelta(days=1)
        streak = 0
        for d in days:
            if d == cursor.isoformat():
                streak += 1
                cursor -= _dt.timedelta(days=1)
            else:
                break
        return streak

    def currently_reading(self, days: int = 7) -> list[dict[str, Any]]:
        """Books with a session (or progress update) in the last ``days``."""
        cutoff = time.time() - days * 86400
        con, _ = self._connect()
        try:
            rows = con.execute(
                "SELECT DISTINCT slug FROM reading_sessions "
                "WHERE started_at > ? "
                "UNION SELECT DISTINCT slug FROM reading_progress "
                "WHERE updated_at > ?", (cutoff, cutoff)).fetchall()
        finally:
            con.close()
        out = []
        for (slug,) in rows:
            try:
                meta = self.load(slug)
                prog = self.get_progress(slug)
                out.append({"slug": slug, "title": meta.get("title", slug),
                            "chapter": prog.get("chapter", 0),
                            "percent": prog.get("percent", 0)})
            except LibraryError:
                continue
        return out

    # ── series ────────────────────────────────────────────────────────
    def set_series(self, slug: str, series: str,
                   index: float = 0) -> dict[str, Any]:
        self.load(slug)
        con, _ = self._connect()
        try:
            con.execute(
                "INSERT OR REPLACE INTO book_series(slug, series, series_index)"
                " VALUES (?,?,?)", (slug, series.strip(), float(index)))
            con.commit()
        finally:
            con.close()
        return {"slug": slug, "series": series.strip(), "index": float(index)}

    def get_series(self, slug: str) -> dict[str, Any]:
        con, _ = self._connect()
        try:
            row = con.execute(
                "SELECT series, series_index FROM book_series WHERE slug = ?",
                (slug,)).fetchone()
        finally:
            con.close()
        if not row:
            return {"slug": slug, "series": "", "index": 0}
        return {"slug": slug, "series": row[0], "index": row[1]}

    def list_series(self) -> list[dict[str, Any]]:
        con, _ = self._connect()
        try:
            rows = con.execute(
                "SELECT series, slug, series_index FROM book_series "
                "ORDER BY series, series_index").fetchall()
        finally:
            con.close()
        series: dict[str, list[dict[str, Any]]] = {}
        for name, slug, idx in rows:
            series.setdefault(name, []).append({"slug": slug, "index": idx})
        return [{"series": name, "books": books}
                for name, books in series.items()]

    # ── annotation export ─────────────────────────────────────────────
    def export_annotations(self, slug: str = "",
                           format: str = "markdown") -> dict[str, Any]:
        """Export bookmarks + notes KOReader-style (md / json / text)."""
        fmt = (format or "markdown").lower()
        if fmt not in ("markdown", "json", "text"):
            raise LibraryError(f"format must be markdown|json|text, got {format!r}")
        con, _ = self._connect()
        try:
            filt = "WHERE b.slug = ?" if slug else ""
            args: tuple = (slug,) if slug else ()
            marks = con.execute(
                f"SELECT b.slug, b.chapter, b.offset_chars, b.label, b.created_at "
                f"FROM bookmarks b {filt} ORDER BY b.slug, b.chapter",
                args).fetchall()
            notes = con.execute(
                f"SELECT slug, chapter, offset_chars, quote, note, created_at "
                f"FROM annotations {('WHERE slug = ?') if slug else ''} "
                f"ORDER BY slug, chapter", args).fetchall()
        finally:
            con.close()
        if fmt == "json":
            import json as _json
            payload = {
                "bookmarks": [
                    {"slug": s, "chapter": c, "offset": o, "label": l}
                    for s, c, o, l, _ in marks],
                "notes": [
                    {"slug": s, "chapter": c, "offset": o,
                     "quote": q, "note": n}
                    for s, c, o, q, n, _ in notes],
            }
            return {"format": "json", "text": _json.dumps(payload, indent=2,
                                                         ensure_ascii=False)}
        lines: list[str] = []
        # regroup by slug for headers
        by_slug: dict[str, list[tuple]] = {}
        for m in marks:
            by_slug.setdefault(m[0], []).append(("bookmark",) + m[1:])
        for n in notes:
            by_slug.setdefault(n[0], []).append(("note",) + n[1:])
        for s in sorted(by_slug):
            try:
                title = self.load(s).get("title", s)
            except LibraryError:
                title = s
            lines.append(f"# {title}" if fmt == "markdown" else f"== {title} ==")
            for kind, chapter, offset, *rest in sorted(
                    by_slug[s], key=lambda x: (x[1], x[2])):
                if kind == "bookmark":
                    label = rest[0] or "bookmark"
                    lines.append(
                        f"- 🔖 ch.{chapter} — {label}" if fmt == "markdown"
                        else f"* [ch.{chapter}] {label}")
                else:
                    quote, note = rest[0], rest[1]
                    if fmt == "markdown":
                        lines.append(f"- 📝 ch.{chapter}: “{quote}” — {note}")
                    else:
                        lines.append(f"* [ch.{chapter}] \"{quote}\" — {note}")
            lines.append("")
        return {"format": fmt, "text": "\n".join(lines).strip()}
