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

import re
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

#: words that signal the user pasted a raw request instead of a title
_REQUEST_PREFIXES = (
    "write me a ", "write me an ", "write a ", "write an ",
    "make me a ", "make me an ", "create a ", "create me a ",
    "please write ", "please make ", "can you write ", "can you make ",
    "i want a ", "i need a ", "give me a ",
)

#: genre hints → (min_chapters, max_chapters) for dynamic inference
_GENRE_CHAPTER_HINTS = (
    (("novel", "saga", "epic", "trilogy", "memoir", "autobiography"), (12, 24)),
    (("story", "tale", "fable", "novella"), (6, 14)),
    (("guide", "handbook", "manual", "course", "textbook"), (8, 16)),
    (("essay", "pamphlet", "short", "brief", "quick"), (3, 6)),
    (("poem", "poetry", "collection"), (8, 20)),
)


def clean_title(topic: str, context: Any = None) -> str:
    """Turn a raw user request into a real book title.

    "Write me a story or book to feel better my girlfriend has been acting
    strange..." → "When She's Healing: A Partner's Guide Through Recovery"

    Online-first: the model path is ALWAYS attempted when a model is usable
    (power state full/degraded), with one retry on weak output.  The heuristic
    below is the offline last resort, not a co-equal option.
    """
    raw = (topic or "").strip()
    if not raw:
        return "Untitled"
    # model path: a real title from the live model — tried first, retried once
    try:
        from ..llm.power import model_usable
        _usable = model_usable(context)
    except Exception:  # noqa: BLE001
        _usable = False
    router = getattr(context, "router", None) if context is not None else None
    if _usable and router is not None:
        title = _model_title(raw, router)
        if title:
            return title
    # offline last resort
    return _heuristic_title(raw)


def _model_title(raw: str, router: Any) -> str:
    """Ask the live model for a title.  Two attempts, strict validation."""
    from ..llm.base import Message, SamplingParams

    prompts = [
        ("You are a bestselling book editor. Reply with ONLY the title — "
         "no quotes, no explanation, no subtitle unless essential. "
         "Make it specific, evocative, and human.",
         f"Give this book a strong title (max 8 words). It is about: {raw[:400]}"),
        ("You are a book editor. Reply with ONLY a short punchy book title.",
         f"Title for a book about: {raw[:300]}"),
    ]
    for system, user in prompts:
        try:
            response = router.chat(
                [Message.system(system), Message.user(user)],
                SamplingParams(temperature=0.6, max_tokens=40),
            )
        except Exception:  # noqa: BLE001
            continue
        title = (getattr(response, "text", "") or "").strip().strip("\"'“”")
        # strip common model prefixes
        title = re.sub(r"^(title|book title)\s*:\s*", "", title, flags=re.I).strip()
        low = title.lower()
        if (getattr(response, "ok", False) and 3 <= len(title) <= 90
                and "write me" not in low and "make me" not in low
                and "as an ai" not in low and "i'm sorry" not in low
                and len(title.split()) <= 12):
            return title
    return ""


#: (relationship words, emotional themes) → title template parts
_THEME_RELATIONSHIPS = (
    "girlfriend", "boyfriend", "wife", "husband", "partner", "fiance",
    "fiancée", "spouse", "lover",
)
_THEME_EMOTIONS = {
    "trust": "Trust", "distrust": "Trust", "cheat": "Trust",
    "feel better": "Healing", "healing": "Healing", "hurt": "Healing",
    "strange": "Understanding", "distant": "Closeness", "cold": "Closeness",
    "fight": "Conflict", "argu": "Conflict", "breakup": "Letting Go",
    "break up": "Letting Go", "operation": "Recovery", "surgery": "Recovery",
    "sick": "Recovery", "grief": "Grief", "loss": "Grief", "death": "Grief",
    "anxious": "Calm", "anxiety": "Calm", "stress": "Calm",
    "love": "Love", "marriage": "Marriage", "wedding": "Marriage",
}


def _theme_title(raw: str) -> str:
    """Build a genuine title from emotional/relationship themes.

    "write me a book to feel better, my girlfriend has been acting strange
    after her operation and I can't trust her" → "Healing Together: Trust
    and Recovery in Your Relationship"
    """
    low = (raw or "").lower()
    rel = next((w for w in _THEME_RELATIONSHIPS if w in low), "")
    themes: list[str] = []
    for key, label in _THEME_EMOTIONS.items():
        if key in low and label not in themes:
            themes.append(label)
    if not rel or not themes:
        return ""
    main = themes[0]
    rest = [t for t in themes[1:] if t != main]
    if rest:
        sub = f"{rest[0]} in Your Relationship"
    else:
        sub = "A Partner's Guide"
    connector = "Together" if main in ("Healing", "Recovery", "Understanding") else "Again"
    return f"{main} {connector}: {sub}"


def _heuristic_title(raw: str) -> str:
    """Offline last-resort title cleaner (used only when no model is usable)."""
    # theme-first: emotional/support requests get a real title, not a truncation
    themed = _theme_title(raw)
    if themed:
        return themed
    # heuristic: strip request prefixes, take the core phrase, title-case it
    lowered = raw.lower()
    for prefix in _REQUEST_PREFIXES:
        if lowered.startswith(prefix):
            raw = raw[len(prefix):].strip()
            lowered = raw.lower()
            break
    # strip leading medium words ("a story or book", "a book", "an essay", ...)
    for medium in ("a story or book", "a book or story", "story or book",
                   "book or story", "a story", "a book", "story", "book",
                   "an essay", "essay", "a novel", "novel", "a guide", "guide",
                   "a poem", "poem", "a memoir", "memoir"):
        if lowered.startswith(medium):
            raw = raw[len(medium):].strip()
            lowered = raw.lower()
            break
    # drop emotional-framing clauses ("to feel better", "to cheer me up", ...)
    raw = re.sub(r"\bto feel (better|good|happier)\b,?\s*", "",
                 raw, flags=re.IGNORECASE).strip(" ,")
    lowered = raw.lower()
    # strip dangling topic introducers left behind ("about X", "on X")
    for intro in ("about ", "on ", "for "):
        if lowered.startswith(intro):
            raw = raw[len(intro):].strip()
            lowered = raw.lower()
            break
    # cut trailing rambling: keep up to the first sentence-ish boundary
    # after a reasonable length, or the whole thing if short
    core = re.split(r"[.!?]\s", raw, maxsplit=1)[0].strip()
    if len(core) > 90:
        # keep the most meaningful chunk: prefer the part after about/on/for
        m = re.search(r"\b(about|on|for)\b(.{10,80})", core, re.IGNORECASE)
        if m:
            core = m.group(2).strip(" ,:-")
        else:
            core = core[:87].rsplit(" ", 1)[0]
    core = core.strip(" ,.:-")
    if not core:
        return "Untitled"
    # cap at ~8 words for a real title feel — never end on a dangling word
    _DANGLING = {"a", "an", "the", "to", "of", "in", "on", "for", "with",
                 "and", "or", "has", "have", "had", "is", "was", "be", "been"}
    words = core.split()
    if len(words) > 9:
        cut = words[:9]
        while len(cut) > 4 and cut[-1].lower().strip(".,") in _DANGLING:
            cut = cut[:-1]
        core = " ".join(cut).strip(" ,:-")
        words = core.split()
    # title case, but keep small words lowercase mid-title
    small = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with"}
    words = core.split()
    titled = [words[0].capitalize()] + [
        w if w.lower() in small else w.capitalize() for w in words[1:]]
    return " ".join(titled)


def plan_book(topic: str, notes: str, context: Any,
              *, genre: str = "") -> dict[str, Any] | None:
    """One model call for the whole book plan: title + chapter count + outline.

    Replaces 3 sequential round-trips (title, count, outline) with a single
    call.  Returns {"title": str, "chapters": [{"title":..., "beats":[...]}]}
    or None when the model isn't usable / the response is bad.
    """
    try:
        from ..llm.power import model_usable
        if not model_usable(context):
            return None
    except Exception:  # noqa: BLE001
        return None
    router = getattr(context, "router", None)
    if router is None:
        return None
    try:
        import json as _json
        from ..llm.base import Message, SamplingParams
        prompt = (
            f"You are planning a book. Topic: {topic[:400]}\n"
            f"Genre/tone: {genre or 'practical non-fiction'}\n"
            f"Research notes (may be empty):\n{(notes or '(none)')[:4000]}\n\n"
            "Decide how many chapters this book truly needs (let the content "
            "decide — a focused guide might need 5, a memoir 15; never pad, "
            "never truncate) and outline them.\n\n"
            "Reply with ONLY a JSON object, no prose, no fences:\n"
            '{"title": "<strong specific book title, max 10 words>", '
            '"chapters": [{"title": "<chapter title>", '
            '"beats": ["<4-6 section beats>"]}]}'
        )
        response = router.chat(
            [Message.system(
                "You are a bestselling book editor. Reply with ONLY the "
                "requested JSON."),
             Message.user(prompt)],
            SamplingParams(temperature=0.5, max_tokens=3000),
        )
        text = (getattr(response, "text", "") or "").strip()
        start, end = text.find("{"), text.rfind("}")
        if not getattr(response, "ok", False) or start == -1 or end <= start:
            return None
        plan = _json.loads(text[start:end + 1])
        title = str(plan.get("title", "")).strip().strip("\"'")
        raw_chapters = plan.get("chapters") or []
        if not title or not isinstance(raw_chapters, list) or not raw_chapters:
            return None
        chapters = []
        for item in raw_chapters[:24]:
            if not isinstance(item, dict):
                continue
            ct = str(item.get("title", "")).strip()
            beats = [str(b).strip() for b in (item.get("beats") or [])
                     if str(b).strip()]
            if ct:
                chapters.append({"title": ct, "beats": beats[:8]})
        if len(chapters) < 3:
            return None
        low = title.lower()
        if "write me" in low or "as an ai" in low:
            return None
        return {"title": title, "chapters": chapters}
    except Exception:  # noqa: BLE001 - caller falls back to separate calls
        return None


def infer_chapter_count(topic: str, context: Any = None,
                        *, default: int = 8) -> int:
    """How many chapters does this book actually need?

    Model-inferred when a live model is answering; otherwise a heuristic
    based on genre hints and topic complexity.  Never blindly 8.
    """
    text = (topic or "").lower()
    router = getattr(context, "router", None) if context is not None else None
    if router is not None:
        try:
            from ..llm.base import Message, SamplingParams
            response = router.chat(
                [Message.system(
                    "You are a book editor. Reply with ONLY an integer."),
                 Message.user(
                     f"How many chapters should a book on this topic have? "
                     f"Short guide: 4-6. Standard book: 8-12. Novel/memoir: "
                     f"14-20. Reply with just the number.\nTopic: {topic[:300]}")],
                SamplingParams(temperature=0.2, max_tokens=8),
            )
            n = int(re.search(r"\d+", getattr(response, "text", "") or "").group())
            if response.ok and 3 <= n <= 24:
                return n
        except Exception:  # noqa: BLE001 - heuristic fallback below
            pass
    # heuristic fallback
    for keywords, (lo, hi) in _GENRE_CHAPTER_HINTS:
        if any(k in text for k in keywords):
            # scale within the band by topic richness
            richness = len(set(re.findall(r"[a-z]{4,}", text)))
            frac = min(1.0, richness / 25.0)
            return lo + round((hi - lo) * frac)
    # generic: 6 + 1 per 4 distinct content words, clamped to 5..12
    richness = len(set(re.findall(r"[a-z]{4,}", text)))
    return max(5, min(12, 6 + richness // 4))


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
        chapters: int = 0,
        words_per_chapter: int = 0,
        research: bool = True,
        notes: str = "",
    ) -> Book:
        topic = (topic or "").strip()
        if not topic:
            raise BookError("a book needs a topic")
        book_notes = notes.strip()[:_NOTES_CAP] if notes.strip() else ""
        if not book_notes and research:
            book_notes = self.research_topic(topic)
        explicit = int(chapters) if chapters else 0
        organic = explicit <= 0
        # Title: explicit wins; else online-first single plan_book call when
        # available (title + outline in one round-trip); else clean_title.
        # Organic mode never plans a full outline up front — it seeds and grows.
        clean = (title or "").strip()
        planned = None
        if not clean and not organic:
            try:
                from .outline import plan_book
                plan = plan_book(topic, book_notes, self.context, genre=genre)
                if plan:
                    clean = plan["title"]
                    planned = plan["chapters"]
            except Exception:
                planned = None
        if not clean:
            clean = clean_title(topic, self.context)
        slug = slugify(clean or topic)
        if self._json_path(slug).exists():
            raise BookError(
                f"book {slug!r} already exists — load it (nm book write {slug}) or "
                "pass a different title"
            )
        explicit = int(chapters) if chapters else 0
        organic = explicit <= 0
        # target_words is a soft length guide, never a quota: chapters run
        # as long as their material needs
        wpc = int(words_per_chapter) if words_per_chapter else 1000
        book = Book(
            topic=topic, slug=slug, title=clean, subtitle=subtitle.strip(),
            author=author.strip(), genre=genre.strip(), description=description.strip(),
            target_words=max(300, wpc),
        )
        book._context = self.context  # type: ignore[attr-defined]
        book.organic = organic
        book.notes = book_notes
        if organic:
            # no count is ever decided: plant the opening arc and let the
            # book grow chapter by chapter as it is written
            book.coverage = outline_mod.coverage_map(topic)
            outline_mod.seed_outline(book, context=self.context)
            _log.info("book created (organic): %s — seed of %d chapters, "
                      "grows as written", slug, len(book.chapters))
        elif planned is not None:
            # the unified plan already has title + outline — apply directly
            from .model import Chapter
            book.chapters = [
                Chapter(number=i + 1, title=c["title"], beats=c["beats"])
                for i, c in enumerate(planned)
            ]
            _log.info("book created: %s (%d chapters planned)", slug,
                      len(book.chapters))
        else:
            outline_mod.make_outline(book, n_chapters=max(3, min(explicit, 24)),
                                     context=self.context)
            _log.info("book created: %s (%d chapters planned)", slug,
                      len(book.chapters))
        self.save(book)
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
        """(Re)generate the outline for an existing book.

        Count-based books keep the old behavior (re-plan N chapters).
        Organic books get a fresh continuation assessment — the outline
        extends (or the book concludes) from what is actually written.
        """
        book = self.load(slug)
        if book.organic:
            decision = outline_mod.assess_continuation(book,
                                                       context=self.context)
            self.save(book)
            _log.info("organic plan %s: %s (added %d chapters)", slug,
                      decision["reason"], len(decision["added"]))
            return book
        n = len(book.chapters) or infer_chapter_count(book.topic, self.context)
        outline_mod.make_outline(book, n_chapters=n,
                                 context=self.context)
        self.save(book)
        return book

    def write_next(self, slug: str) -> dict[str, Any]:
        """Write the next unwritten chapter (resumable unit).

        Organic books: after each chapter the continuation assessment
        decides what comes next — the outline grows (or the book
        concludes) from the actual content, never from a preset count.
        """
        book = self.load(slug)
        chapter = book.next_unwritten()
        if chapter is None:
            if book.organic and not book.concluded:
                # seed fully written but the book hasn't been judged yet —
                # grow the outline before calling it done
                decision = outline_mod.assess_continuation(
                    book, context=self.context)
                self.save(book)
                chapter = book.next_unwritten()
                if chapter is None:
                    book.status = STATUS_WRITTEN
                    self.save(book)
                    return {"slug": slug, "done": True,
                            "total_words": book.total_words,
                            "organic": True, "concluded": book.concluded,
                            "note": decision["reason"]}
            else:
                book.status = STATUS_WRITTEN
                self.save(book)
                return {"slug": slug, "done": True,
                        "total_words": book.total_words,
                        "organic": book.organic, "concluded": book.concluded}
        prev_tail = ""
        for c in book.chapters:
            if c.number < chapter.number and c.status == STATUS_WRITTEN:
                prev_tail = c.text
        started = time.time()
        write_mod.write_chapter(book, chapter, context=self.context)
        decision: dict[str, Any] | None = None
        if book.organic and not book.concluded:
            decision = outline_mod.assess_continuation(book,
                                                       context=self.context)
        self.save(book)
        if book.complete:
            book.status = STATUS_WRITTEN
            self.save(book)
        _log.info("chapter %d/%d written for %s (%d words)",
                  chapter.number, len(book.chapters), slug, chapter.words)
        out: dict[str, Any] = {
            "slug": slug,
            "done": book.complete,
            "chapter": chapter.number,
            "total_chapters": len(book.chapters),
            "chapter_words": chapter.words,
            "total_words": book.total_words,
            "seconds": round(time.time() - started, 2),
            "organic": book.organic,
            "concluded": book.concluded,
        }
        if decision is not None:
            out["continuation"] = decision["reason"]
            out["chapters_added"] = decision["added"]
        return out

    def write_all(self, slug: str, *, limit: int = 0) -> dict[str, Any]:
        """Write every unwritten chapter (or up to ``limit`` of them).

        Organic books keep growing via continuation assessments until the
        topic is covered — the loop ends on ``book.complete``, with a
        hard iteration backstop so a pathological assessment can never
        spin forever.
        """
        book = self.load(slug)
        written = 0
        guard = 0
        while not book.complete and (limit == 0 or written < limit):
            guard += 1
            if guard > outline_mod.MAX_ORGANIC_CHAPTERS + 10:
                book.concluded = True
                self.save(book)
                break
            r = self.write_next(slug)
            if r.get("done"):
                break
            written += 1
        book = self.load(slug)
        return {
            "slug": slug,
            "complete": book.complete,
            "concluded": book.concluded,
            "organic": book.organic,
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
        chapters: int = 0,
        words_per_chapter: int = 0,
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
