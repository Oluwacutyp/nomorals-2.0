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

import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["Library", "LibraryError", "IngestResult", "SearchHit"]

_WORD = re.compile(r"[A-Za-z0-9']+")

# ingest paths: plain text reads straight off disk; everything else goes
# through the nomorals.documents parsers (pdf, epub, docx, html).
_TEXT_EXTS = (".txt", ".md", ".markdown", ".text")
_DOC_EXTS = (".pdf", ".epub", ".docx", ".html", ".htm")
_INGEST_EXTS = _TEXT_EXTS + _DOC_EXTS


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug, "title": self.title, "author": self.author,
            "chapters": self.chapters, "words": self.words,
            "source_file": self.source_file,
            "seconds": round(self.seconds, 2), "strategy": self.strategy,
            "note": self.note,
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


# ── the library ─────────────────────────────────────────────────────────

class Library:
    """Owns ``workspace/library/``: ingest, search, read, list, drop."""

    def __init__(self, context: Any) -> None:
        self.context = context

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
        return con, row[0] == "1"

    def _reindex_all(self, con: sqlite3.Connection,
                     fts5: bool) -> None:
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

        dest = self.book_dir(slug)
        chapters_dir = dest / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)
        (dest / p.name).write_bytes(p.read_bytes())  # keep the original

        total_words = 0
        meta_chapters = []
        for i, (name, body) in enumerate(chapters, 1):
            (chapters_dir / f"{i:03d}.md").write_text(
                f"# {name}\n\n{body}\n", encoding="utf-8")
            words = len(body.split())
            total_words += words
            meta_chapters.append({"number": i, "title": name, "words": words})

        meta = {
            "slug": slug, "title": book_title, "author": book_author,
            "source_file": p.name, "strategy": strategy,
            "chapters": meta_chapters, "total_words": total_words,
            "ingested_at": time.time(),
        }
        (dest / "book.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        # rebuild the whole index (books are few; chapters are the rows)
        con, fts5 = self._connect()
        self._reindex_all(con, fts5)
        con.execute(
            "INSERT OR REPLACE INTO books(slug,title,author,source_file,"
            "chapters,words,strategy,ingested_at) VALUES (?,?,?,?,?,?,?,?)",
            (slug, book_title, book_author, p.name, len(chapters),
             total_words, strategy, time.time()))
        con.commit()
        con.close()

        return IngestResult(
            slug=slug, title=book_title, author=book_author,
            chapters=len(chapters), words=total_words,
            source_file=str(p), seconds=time.perf_counter() - started,
            strategy=strategy,
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
        b = self.load(slug)
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
        for f in dest.iterdir():
            if f.is_dir():
                for c in f.iterdir():
                    c.unlink(missing_ok=True)
                f.rmdir()
            else:
                f.unlink()
        dest.rmdir()
        con, fts5 = self._connect()
        self._reindex_all(con, fts5)
        con.execute("DELETE FROM books WHERE slug = ?", (slug,))
        # drop the reading state too — no orphaned progress/bookmarks/notes
        for table in ("reading_progress", "bookmarks", "annotations",
                      "collection_books", "book_tags", "ratings"):
            con.execute(f"DELETE FROM {table} WHERE slug = ?", (slug,))
        con.commit()
        con.close()
        return {"dropped": slug}

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
