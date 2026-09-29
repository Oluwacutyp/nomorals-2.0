"""BookForge: the book pipeline.

    create  → topic + research notes + real outline (saved, resumable)
    write   → chapters one at a time (model or template floor)
    build   → manuscript.md + real PDF (TOC with true page numbers,
              chapters on fresh pages) via the pure-Python PDF writer
    send    → deliver the PDF to any live chat platform via the gateway

Every stage is idempotent and resumable: a killed run (phone off, kernel
restart, 12-hour session ending) is picked up where it stopped, because the
book and every chapter live on disk.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .model import (STATUS_BUILT, STATUS_SENT, STATUS_WRITTEN, Book, BookError,
                    count_words, slugify)
from . import outline as outline_mod
from . import write as write_mod

_log = get_logger(__name__)

__all__ = ["BookForge"]

_NOTES_CAP = 16000


class BookForge:
    def __init__(self, context: Any) -> None:
        self.context = context

    # ── paths ─────────────────────────────────────────────────────────────
    def _workspace(self) -> Path:
        settings = getattr(self.context, "settings", None)
        root = getattr(settings, "workspace_dir", None) if settings else None
        if not root:
            root = Path.cwd() / "workspace"
        return Path(root)

    def books_dir(self) -> Path:
        d = self._workspace() / "books"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def book_dir(self, slug: str) -> Path:
        return self.books_dir() / slug

    def _json_path(self, slug: str) -> Path:
        return self.book_dir(slug) / "book.json"

    # ── persistence ───────────────────────────────────────────────────────
    def save(self, book: Book) -> None:
        d = self.book_dir(book.slug)
        d.mkdir(parents=True, exist_ok=True)
        self._json_path(book.slug).write_text(
            _dumps(book.to_dict()), encoding="utf-8"
        )
        if book.notes.strip():
            (d / "notes.md").write_text(book.notes, encoding="utf-8")
        chapters_dir = d / "chapters"
        chapters_dir.mkdir(exist_ok=True)
        for c in book.chapters:
            if c.status == STATUS_WRITTEN and c.text.strip():
                (chapters_dir / f"ch{c.number:02d}.md").write_text(
                    f"# {c.title}\n\n{c.text}", encoding="utf-8"
                )

    def load(self, slug: str) -> Book:
        path = self._json_path(slug)
        if not path.exists():
            raise BookError(f"no book {slug!r} (nm book list to see what exists)")
        import json

        book = Book.from_dict(json.loads(path.read_text(encoding="utf-8")))
        book._context = self.context  # type: ignore[attr-defined]
        return book

    def list_books(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for d in sorted(self.books_dir().iterdir()):
            jf = d / "book.json"
            if not jf.exists():
                continue
            try:
                import json

                book = Book.from_dict(json.loads(jf.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001 - a corrupt book must not hide the rest
                continue
            out.append({
                "slug": book.slug,
                "title": book.display_title,
                "status": book.status,
                "chapters": len(book.chapters),
                "written": book.chapters_written,
                "words": book.total_words,
                "updated": time.strftime("%Y-%m-%d %H:%M", time.localtime(book.updated_at)),
            })
        return out

    # ── stage 1: create (research + outline) ──────────────────────────────
    def create(
        self,
        topic: str,
        *,
        title: str = "",
        subtitle: str = "",
        author: str = "",
        genre: str = "",
        description: str = "",
        chapters: int = 8,
        words_per_chapter: int = 1200,
        research: bool = True,
        notes: str = "",
    ) -> Book:
        topic = (topic or "").strip()
        if not topic:
            raise BookError("a book needs a topic")
        slug = slugify(title or topic)
        if self._json_path(slug).exists():
            raise BookError(
                f"book {slug!r} already exists — load it (nm book write {slug}) or "
                "pass a different title"
            )
        book = Book(
            topic=topic, slug=slug, title=title.strip(), subtitle=subtitle.strip(),
            author=author.strip(), genre=genre.strip(), description=description.strip(),
            target_words=max(300, int(words_per_chapter)),
        )
        book._context = self.context  # type: ignore[attr-defined]
        if notes.strip():
            book.notes = notes.strip()[:_NOTES_CAP]
        elif research:
            book.notes = self.research_topic(topic)
        outline_mod.make_outline(book, n_chapters=max(3, min(int(chapters), 16)),
                                 context=self.context)
        self.save(book)
        _log.info("book created: %s (%d chapters planned)", slug, len(book.chapters))
        return book

    def research_topic(self, topic: str, *, pages: int = 4) -> str:
        """Live research notes for the outline/writer.  Best-effort: any
        failure (no network, robots, dead endpoints) yields '' — the book
        is still written from the outline, just without external quotes."""
        try:
            from ..agents.search.engine import SearchEngine

            report = SearchEngine(self.context).run(topic, mode="quick", pages=pages)
        except Exception as exc:  # noqa: BLE001
            _log.debug("book research failed for %r: %s", topic, exc)
            return ""
        parts: list[str] = []
        summary = str(report.get("summary") or "").strip()
        if summary:
            parts.append("Research summary:\n" + summary[:3000])
        for sec in (report.get("sections") or [])[:8]:
            parts.append(f"From {sec.get('domain', sec.get('url', ''))}:\n"
                         f"{sec.get('text', '')[:600]}")
        for page in (report.get("pages") or [])[:pages]:
            text = str(page.get("text") or "")
            if len(text) > 2500:
                # grab the strongest middle chunk, not the nav-heavy top
                text = text[400:2900]
            if text.strip():
                parts.append(f"From {page.get('domain', page.get('url', ''))}:\n{text[:2500]}")
        return ("\n\n".join(parts))[:_NOTES_CAP]

    # ── stage 2: write ─────────────────────────────────────────────────────
    def plan(self, slug: str) -> Book:
        """(Re)generate the outline for an existing book (keeps written chapters
        only when the chapter count still fits; otherwise re-plans from zero)."""
        book = self.load(slug)
        outline_mod.make_outline(book, n_chapters=max(3, len(book.chapters) or 8),
                                 context=self.context)
        self.save(book)
        return book

    def write_next(self, slug: str) -> dict[str, Any]:
        """Write the next unwritten chapter (resumable unit)."""
        book = self.load(slug)
        chapter = book.next_unwritten()
        if chapter is None:
            book.status = STATUS_WRITTEN
            self.save(book)
            return {"slug": slug, "done": True, "total_words": book.total_words}
        prev_tail = ""
        for c in book.chapters:
            if c.number < chapter.number and c.status == STATUS_WRITTEN:
                prev_tail = c.text
        started = time.time()
        write_mod.write_chapter(book, chapter, context=self.context)
        self.save(book)
        if book.complete:
            book.status = STATUS_WRITTEN
            self.save(book)
        _log.info("chapter %d/%d written for %s (%d words)",
                  chapter.number, len(book.chapters), slug, chapter.words)
        return {
            "slug": slug,
            "done": book.complete,
            "chapter": chapter.number,
            "total_chapters": len(book.chapters),
            "chapter_words": chapter.words,
            "total_words": book.total_words,
            "seconds": round(time.time() - started, 2),
        }

    def write_all(self, slug: str, *, limit: int = 0) -> dict[str, Any]:
        """Write every unwritten chapter (or up to ``limit`` of them)."""
        book = self.load(slug)
        written = 0
        while not book.complete and (limit == 0 or written < limit):
            r = self.write_next(slug)
            if r.get("done"):
                break
            written += 1
        book = self.load(slug)
        return {
            "slug": slug,
            "complete": book.complete,
            "written_now": written,
            "chapters_written": book.chapters_written,
            "total_chapters": len(book.chapters),
            "total_words": book.total_words,
        }

    # ── stage 3: build ─────────────────────────────────────────────────────
    def build(self, slug: str, *, page_size: str = "A4") -> dict[str, Any]:
        """Compile manuscript.md + <slug>.pdf (real book layout)."""
        book = self.load(slug)
        unwritten = book.chapters_written
        if unwritten == 0:
            raise BookError(f"book {slug!r} has no written chapters yet")
        manuscript = book.manuscript()
        d = self.book_dir(slug)
        (d / "manuscript.md").write_text(manuscript, encoding="utf-8")

        from ..core.pdf import render_pdf

        data = render_pdf(
            manuscript,
            title=book.display_title,
            page_size=page_size,
            headings=True,
            chapter_break=True,
            toc=True,
        )
        pdf_path = d / f"{slug}.pdf"
        pdf_path.write_bytes(data)
        # page count: count page objects in the finished file
        n_pages = data.count(b"/Type /Page ") + data.count(b"/Type /Page\n")
        book.status = STATUS_BUILT
        self.save(book)
        _log.info("book built: %s (%d bytes pdf, ~%d pages)", slug, len(data), n_pages)
        return {
            "slug": slug,
            "pdf": str(pdf_path),
            "manuscript": str(d / "manuscript.md"),
            "pdf_bytes": len(data),
            "pages": n_pages,
            "words": book.total_words,
            "chapters_written": book.chapters_written,
            "total_chapters": len(book.chapters),
        }

    # ── stage 4: send ("send when done") ──────────────────────────────────
    def send(self, slug: str, platform: str, chat_id: str, *, caption: str = "") -> dict[str, Any]:
        book = self.load(slug)
        pdf_path = self.book_dir(slug) / f"{slug}.pdf"
        if not pdf_path.exists():
            self.build(slug)
            pdf_path = self.book_dir(slug) / f"{slug}.pdf"
        if not caption:
            caption = (f"📕 {book.display_title} — {book.total_words} words, "
                       f"{book.chapters_written} chapters")
        from ..tools.filesend import send_file

        result = send_file(self.context, platform, chat_id, str(pdf_path), caption=caption)
        book.status = STATUS_SENT
        self.save(book)
        result["slug"] = slug
        return result

    def deliver(self, slug: str, platform: str, chat_id: str) -> dict[str, Any]:
        """build + send in one call — the 'send when done' path."""
        built = self.build(slug)
        sent = self.send(slug, platform, chat_id)
        return {**built, **sent}

    # ── full pipeline ──────────────────────────────────────────────────────
    def run(
        self,
        topic: str,
        *,
        title: str = "",
        chapters: int = 8,
        words_per_chapter: int = 1200,
        author: str = "",
        genre: str = "",
        research: bool = True,
        send_to: tuple[str, str] | None = None,
        on_chapter: Any = None,
    ) -> dict[str, Any]:
        """create → write all → build → (optionally) send.

        ``send_to`` = (platform, chat_id); when given, the finished PDF is
        delivered to that chat ("send when done").  ``on_chapter(result)``
        fires after each chapter (progress hook for chat/CLI).
        """
        book = self.create(topic, title=title, chapters=chapters,
                           words_per_chapter=words_per_chapter, author=author,
                           genre=genre, research=research)
        slug = book.slug
        while True:
            r = self.write_next(slug)
            if on_chapter:
                on_chapter(r)
            if r.get("done"):
                break
        result = self.build(slug)
        if send_to:
            platform, chat_id = send_to
            result["sent"] = self.send(slug, platform, chat_id)
        return result


def _dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, indent=2, ensure_ascii=False)
