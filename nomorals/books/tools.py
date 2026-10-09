"""BookForge registry tools — callable by the main AI and every sub-agent.

    book_create  topic → research notes + real outline (resumable on disk)
    book_write   write the next chapter (or all unwritten) of a book
    book_build   compile manuscript.md + real PDF (TOC, chapter breaks)
    book_send    deliver the PDF to a live chat platform
    book_run     the whole pipeline: create → write → build → send
    book_status  progress of one book
    book_list    everything on disk

    Library (the owner's existing books):
    library_ingest    add a book file (.txt/.md/.pdf/.epub/.docx/.html)
    library_search    full-text search across all books
    library_read      read a chapter (marks reading progress)
    library_progress  get/set reading progress, or resume where you left off
    library_bookmark  add/list/remove bookmarks
    library_note      add/list/remove annotations & notes
    library_shelf     collections: create/delete/add/remove/list/show
    library_tag       tag a book, list tags, find books by tag
    library_rate      1–5 star rating per book
"""

from __future__ import annotations

from typing import Any

from ..core.policy import Capability

__all__ = ["register"]


def register(registry: Any) -> None:
    context = registry.context

    def forge() -> Any:
        from .forge import BookForge

        return BookForge(context)

    @registry.register(
        "book_create",
        description=(
            "Start a book: gather live research notes on the topic and plant "
            "the opening arc. The book then grows organically — chapters "
            "emerge as it is written until the topic is genuinely covered, "
            "never a preset count. Saved to disk — resumable. Returns the plan."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "topic": "str — what the book is about",
            "title": "str (optional) — the book title (else derived from topic)",
            "chapters": "int (optional, 0) — explicit chapter count ONLY if the user asked for one; 0 = organic growth",
            "words_per_chapter": "int (optional, 1000) — soft length guide, never a quota",
            "author": "str (optional)",
            "genre": "str (optional) — tone/genre",
            "research": "bool (optional, true) — gather live research notes",
        },
    )
    def book_create(topic: str, *, title: str = "", chapters: int = 0,
                    words_per_chapter: int = 1000, author: str = "",
                    genre: str = "", research: bool = True) -> dict[str, Any]:
        book = forge().create(topic, title=title, chapters=chapters,
                              words_per_chapter=words_per_chapter, author=author,
                              genre=genre, research=research)
        return {
            "slug": book.slug,
            "title": book.display_title,
            "status": book.status,
            "chapters": [
                {"n": c.number, "title": c.title, "beats": c.beats}
                for c in book.chapters
            ],
            "research_chars": len(book.notes),
        }

    @registry.register(
        "book_write",
        description=(
            "Write book chapters: the next unwritten one by default, or all of "
            "them (all=true). Model-written when a live model is answering, "
            "template-composed otherwise — always real prose. Resumable."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "slug": "str — the book slug",
            "all": "bool (optional, false) — write every unwritten chapter",
        },
    )
    def book_write(slug: str, *, all: bool = False) -> dict[str, Any]:
        if all:
            return forge().write_all(slug)
        return forge().write_next(slug)

    @registry.register(
        "book_build",
        description=(
            "Compile a written book into manuscript.md + a real PDF (title, "
            "table of contents with true page numbers, each chapter on a fresh "
            "page). Returns the PDF path."
        ),
        capability=Capability.FS_WRITE,
        parameters={"slug": "str — the book slug",
                    "page_size": "str (optional, A4) — A4 | Letter"},
    )
    def book_build(slug: str, *, page_size: str = "A4") -> dict[str, Any]:
        return forge().build(slug, page_size=page_size)

    @registry.register(
        "book_send",
        description=(
            "Send the book's PDF to a live chat platform through the gateway "
            "(builds it first if needed). The 'send when done' step."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "slug": "str — the book slug",
            "platform": "str — telegram | discord | whatsapp | console",
            "chat_id": "str — the target chat",
            "caption": "str (optional)",
        },
    )
    def book_send(slug: str, platform: str, chat_id: str,
                  *, caption: str = "") -> dict[str, Any]:
        return forge().send(slug, platform, chat_id, caption=caption)

    @registry.register(
        "book_run",
        description=(
            "The whole book pipeline in one call: research → outline → write "
            "every chapter → build the PDF → (when send_to given) deliver it. "
            "Long-running; use book_write/book_build/book_send for stepwise."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "topic": "str — what the book is about",
            "title": "str (optional)",
            "chapters": "int (optional, 8)",
            "words_per_chapter": "int (optional, 1200)",
            "author": "str (optional)",
            "genre": "str (optional)",
            "research": "bool (optional, true) — gather live research notes first",
            "send_platform": "str (optional) — deliver the PDF here when done",
            "send_chat": "str (optional) — the target chat",
        },
    )
    def book_run(topic: str, *, title: str = "", chapters: int = 8,
                 words_per_chapter: int = 1200, author: str = "",
                 genre: str = "", research: bool = True, send_platform: str = "",
                 send_chat: str = "") -> dict[str, Any]:
        send_to = (send_platform, send_chat) if send_platform and send_chat else None
        return forge().run(topic, title=title, chapters=chapters,
                           words_per_chapter=words_per_chapter, author=author,
                           genre=genre, research=research, send_to=send_to)

    @registry.register(
        "book_status",
        description="Progress of a book: chapters written, words, state, PDF path.",
        capability=Capability.FS_READ,
        parameters={"slug": "str — the book slug"},
    )
    def book_status(slug: str) -> dict[str, Any]:
        book = forge().load(slug)
        d = forge().book_dir(slug)
        return {
            "slug": slug,
            "title": book.display_title,
            "status": book.status,
            "chapters_written": book.chapters_written,
            "total_chapters": len(book.chapters),
            "total_words": book.total_words,
            "outline": [{"n": c.number, "title": c.title} for c in book.chapters],
            "pdf": str(d / f"{slug}.pdf") if (d / f"{slug}.pdf").exists() else "",
        }

    @registry.register(
        "book_list",
        description="List every book on disk with its progress.",
        capability=Capability.FS_READ,
    )
    def book_list() -> dict[str, Any]:
        return {
            "books": forge().list_books(),
            "library": _library().list_books(),
        }

    # ── the story reader: webnovel archives, natively ───────────────────
    def _reader() -> Any:
        from .reader import StoryReader

        return StoryReader(context)

    @registry.register(
        "story_search",
        description=(
            "Search webnovel archives (freewebnovel, novelfull, royalroad) "
            "for a story by title. Returns matches with URLs — follow one "
            "with story_follow."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "query": "str — story title to search",
            "limit": "int (optional, 10) — max hits",
            "sources": "str (optional) — comma list: freewebnovel,novelfull,royalroad",
        },
    )
    def story_search(query: str, *, limit: int = 10,
                     sources: str = "") -> dict[str, Any]:
        srcs = [s.strip() for s in sources.split(",") if s.strip()] or None
        return {"query": query,
                "hits": _reader().search(query, limit=limit, sources=srcs)}

    @registry.register(
        "story_follow",
        description=(
            "Follow a webnovel: give its archive URL (or just the title — "
            "Devon searches the verified archives). Fetches metadata; "
            "chapters are pulled on demand and cached."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "url_or_title": "str — archive URL or story title",
            "source": "str (optional) — restrict title search to one source",
        },
    )
    def story_follow(url_or_title: str, *, source: str = "") -> dict[str, Any]:
        story = _reader().follow(url_or_title, source=source)
        return {"followed": story.to_dict()}

    @registry.register(
        "story_unfollow",
        description="Stop following a story (chapter cache kept by default).",
        capability=Capability.FS_WRITE,
        parameters={
            "slug": "str — the story slug",
            "keep_cache": "bool (optional, true) — keep fetched chapters",
        },
    )
    def story_unfollow(slug: str, *, keep_cache: bool = True) -> dict[str, Any]:
        return _reader().unfollow(slug, keep_cache=keep_cache)

    @registry.register(
        "story_following",
        description="Every followed story with reading progress.",
        capability=Capability.FS_READ,
    )
    def story_following() -> dict[str, Any]:
        return {"stories": _reader().following()}

    @registry.register(
        "story_read",
        description=(
            "Read a chapter of a followed story (fetches + caches when "
            "needed). Updates reading progress. Chapter 0 = resume where "
            "you left off."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "slug": "str — the story slug",
            "chapter": "int (optional, 0) — chapter number, 0 = resume",
        },
    )
    def story_read(slug: str, *, chapter: int = 0) -> dict[str, Any]:
        return _reader().read(slug, chapter=chapter)

    @registry.register(
        "story_next",
        description="Read the next chapter of a followed story.",
        capability=Capability.NET_OUT,
        parameters={"slug": "str — the story slug"},
    )
    def story_next(slug: str) -> dict[str, Any]:
        return _reader().next(slug)

    @registry.register(
        "story_prev",
        description="Read the previous chapter of a followed story.",
        capability=Capability.NET_OUT,
        parameters={"slug": "str — the story slug"},
    )
    def story_prev(slug: str) -> dict[str, Any]:
        return _reader().prev(slug)

    @registry.register(
        "story_sync",
        description=(
            "Fetch the next few unread chapters for one followed story "
            "(or all of them). Profile-gated: fewer, slower on termux."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "slug": "str (optional) — one story, else all followed",
            "chapters": "int (optional, 0) — how many; 0 = profile default",
        },
    )
    def story_sync(slug: str = "", *, chapters: int = 0) -> dict[str, Any]:
        return _reader().sync(slug, chapters=chapters)

    @registry.register(
        "story_progress",
        description=(
            "Reading progress on a followed story. get = where you left "
            "off; set = record chapter+offset; resume = progress plus the "
            "chapter to read next."
        ),
        capability=Capability.FS_READ,
        parameters={
            "action": "str — get | set | resume",
            "slug": "str — the story slug",
            "chapter": "int (optional, for set)",
            "offset": "int (optional, 0, for set)",
        },
    )
    def story_progress(action: str, slug: str, *, chapter: int = 0,
                       offset: int = 0) -> dict[str, Any]:
        reader = _reader()
        act = (action or "").strip().lower()
        if act == "set":
            return reader.set_progress(slug, chapter, offset)
        if act == "resume":
            return reader.resume(slug)
        if act == "get":
            return reader.get_progress(slug)
        raise ValueError(f"unknown progress action {action!r}")

    @registry.register(
        "story_bookmark",
        description=(
            "Bookmarks on a followed story. add saves chapter+offset+label; "
            "list shows them (one story, or all); remove deletes by id."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "action": "str — add | list | remove",
            "slug": "str (optional) — the story slug",
            "chapter": "int (optional, for add)",
            "offset": "int (optional, 0, for add)",
            "label": "str (optional, for add)",
            "id": "int (optional, for remove) — bookmark id",
        },
    )
    def story_bookmark(action: str, *, slug: str = "", chapter: int = 0,
                       offset: int = 0, label: str = "",
                       id: int = 0) -> dict[str, Any]:
        reader = _reader()
        act = (action or "").strip().lower()
        if act == "add":
            if not slug or chapter < 1:
                raise ValueError("bookmark add needs slug and chapter")
            return reader.add_bookmark(slug, chapter, offset, label)
        if act == "list":
            return {"bookmarks": reader.list_bookmarks(slug)}
        if act == "remove":
            if not slug or id < 1:
                raise ValueError("bookmark remove needs slug and id")
            return reader.remove_bookmark(slug, id)
        raise ValueError(f"unknown bookmark action {action!r}")

    # ── story bibles + continuation ─────────────────────────────────────
    def _bibles() -> Any:
        from .bible import BibleBuilder

        return BibleBuilder(context)

    @registry.register(
        "bible_build",
        description=(
            "Auto-build a story bible from a followed story's read chapters: "
            "cast, open plot threads, world rules, voice (POV/tense/tone). "
            "The bible steers continuations so long stories stay in "
            "character. Re-run to digest more chapters."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "slug": "str — the followed story slug",
            "chapters": "int (optional, 6) — how many recent chapters to digest",
        },
    )
    def bible_build(slug: str, *, chapters: int = 6) -> dict[str, Any]:
        from .continuation import StoryContinuer

        bible = StoryContinuer(context).prepare(slug, chapters=chapters)
        return {"bible": bible.to_dict(),
                "brief": bible.brief()[:2000]}

    @registry.register(
        "bible_build_text",
        description=(
            "Auto-build a story bible from pasted story text (the owner "
            "pastes what Devon should continue from)."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "title": "str — the story title",
            "text": "str — the story text so far",
        },
    )
    def bible_build_text(title: str, text: str) -> dict[str, Any]:
        from .continuation import StoryContinuer

        bible = StoryContinuer(context).prepare_from_text(title, text)
        return {"bible": bible.to_dict(),
                "brief": bible.brief()[:2000]}

    @registry.register(
        "bible_show",
        description="Show a story's bible: cast, threads, rules, voice.",
        capability=Capability.FS_READ,
        parameters={"slug": "str — the story slug"},
    )
    def bible_show(slug: str) -> dict[str, Any]:
        bible = _bibles().load(slug)
        if bible is None:
            raise ValueError(f"no bible for {slug!r} — run bible_build first")
        return {"bible": bible.to_dict()}

    @registry.register(
        "story_continue",
        description=(
            "Continue a read story in its own voice — new chapter(s) that "
            "respect the auto-built bible (cast, open threads, world "
            "rules). The owner's example: keep 'My Vampire System' going "
            "past its ending. Saved under the story's continuations/."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "slug": "str — the followed story slug",
            "n": "int (optional, 1) — how many chapters to write",
            "words": "int (optional, 1500) — target words per chapter",
            "direction": "str (optional) — a story direction for ch.1",
        },
    )
    def story_continue(slug: str, *, n: int = 1, words: int = 1500,
                       direction: str = "") -> dict[str, Any]:
        from .continuation import StoryContinuer

        return StoryContinuer(context).continue_story(
            slug, n=n, words=words, direction=direction)

    # ── FictionWriter ───────────────────────────────────────────────────
    def _fiction() -> Any:
        from .fiction import FictionWriter

        return FictionWriter(context)

    @registry.register(
        "fiction_start",
        description=(
            "Start an original story with the FictionWriter: pick a genre "
            "(mystery, thriller, horror, sci-fi, fantasy, romance — each a "
            "real engine with enforced mechanics, not a prompt prefix) and "
            "a mode: 'novel' (planned arcs, an ending that lands) or "
            "'serial' (never-ending; arcs spawn from dangling threads). "
            "WisdomKeeper themes are woven in as texture. Writes chapter 1."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "premise": "str — what the story is about",
            "genre": "str (optional, fantasy) — mystery | thriller | horror | sci-fi | fantasy | romance",
            "mode": "str (optional, novel) — novel | serial",
            "title": "str (optional)",
            "theme": "str (optional) — wisdom theme to weave in",
        },
    )
    def fiction_start(premise: str, *, genre: str = "fantasy",
                      mode: str = "novel", title: str = "",
                      theme: str = "") -> dict[str, Any]:
        result = _fiction().start(premise, genre=genre, mode=mode,
                                  title=title, theme=theme)
        ch1 = result.pop("chapter_1", {})
        result["chapter_1_words"] = ch1.get("words", 0)
        result["chapter_1_path"] = ch1.get("path", "")
        result["chapter_1_preview"] = (ch1.get("text", "") or "")[:1500]
        return result

    @registry.register(
        "fiction_write",
        description=(
            "Write the next chapter of a FictionWriter story: the genre "
            "engine briefs it (clues planted, tension curve, dread cycle, "
            "rule checks...), the chapter is validated against the "
            "mechanics, then the story state advances. Serial mode rolls "
            "arcs automatically."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "slug": "str — the story slug",
            "direction": "str (optional) — an owner steer for this chapter",
        },
    )
    def fiction_write(slug: str, *, direction: str = "") -> dict[str, Any]:
        result = _fiction().write_next(slug, direction=direction)
        preview = (result.pop("text", "") or "")[:1500]
        result["preview"] = preview
        return result

    @registry.register(
        "fiction_status",
        description=(
            "A FictionWriter story's state: chapters, arcs, open threads, "
            "cast, current arc."
        ),
        capability=Capability.FS_READ,
        parameters={"slug": "str — the story slug"},
    )
    def fiction_status(slug: str) -> dict[str, Any]:
        return _fiction().status(slug)

    @registry.register(
        "fiction_list",
        description="Every FictionWriter story on disk.",
        capability=Capability.FS_READ,
    )
    def fiction_list() -> dict[str, Any]:
        return {"stories": _fiction().list_stories()}

    # ── the library: the owner's existing books ──────────────────────
    def _library() -> Any:
        from .library import Library

        return Library(context)

    @registry.register(
        "library_ingest",
        description=(
            "Add an existing book file to the library: split into chapters, "
            "index every passage for full-text search. Formats: .txt/.md "
            "read directly; .pdf/.epub/.docx/.html extracted via the documents "
            "engine. Use when the owner points at a book they want the system "
            "to know."
        ),
        capability=Capability.FS_READ,
        parameters={
            "path": "str — path to the book file",
            "title": "str (optional) — else guessed from the file",
            "author": "str (optional)",
        },
    )
    def library_ingest(path: str, *, title: str = "",
                       author: str = "") -> dict[str, Any]:
        return _library().ingest(path, title=title, author=author).to_dict()

    @registry.register(
        "library_search",
        description=(
            "Full-text search across ALL ingested books (FTS5 BM25). "
            "Returns ranked passages with context. Use to recall what a "
            "book says about a topic, or to ground answers in the owner's "
            "own books."
        ),
        capability=Capability.FS_READ,
        parameters={
            "query": "str — the search words",
            "top": "int (optional, 8) — how many passages",
        },
    )
    def library_search(query: str, *, top: int = 8) -> dict[str, Any]:
        hits = _library().search(query, top=top)
        return {"hits": [h.to_dict() for h in hits]}

    @registry.register(
        "library_read",
        description=(
            "Read a chapter (or the chapter outline) of an ingested book. "
            "Without a chapter number returns the table of contents. "
            "Reading a chapter updates the book's reading progress."
        ),
        capability=Capability.FS_READ,
        parameters={
            "slug": "str — the book slug",
            "chapter": "int (optional) — chapter number (else outline)",
        },
    )
    def library_read(slug: str, *, chapter: int = 0) -> dict[str, Any]:
        return _library().read(slug, chapter=chapter)

    @registry.register(
        "library_progress",
        description=(
            "Reading progress for a book. action=get returns where the owner "
            "left off; set records it (chapter + char offset); resume returns "
            "the progress plus the next chapter to read."
        ),
        capability=Capability.FS_READ,
        parameters={
            "action": "str — get | set | resume",
            "slug": "str — the book slug",
            "chapter": "int (optional, for set) — chapter number",
            "offset": "int (optional, 0, for set) — char offset into the chapter",
        },
    )
    def library_progress(action: str, slug: str, *, chapter: int = 0,
                         offset: int = 0) -> dict[str, Any]:
        lib = _library()
        act = (action or "").strip().lower()
        if act == "set":
            if chapter < 1:
                raise ValueError("progress set needs chapter >= 1")
            return lib.set_progress(slug, chapter, offset)
        if act == "resume":
            return lib.resume(slug)
        if act == "get":
            return lib.get_progress(slug)
        raise ValueError(f"unknown progress action {action!r} — get | set | resume")

    @registry.register(
        "library_bookmark",
        description=(
            "Bookmarks in an ingested book. action=add saves one (chapter + "
            "offset + label); list shows them (all books, or one slug); "
            "remove deletes by id."
        ),
        capability=Capability.FS_READ,
        parameters={
            "action": "str — add | list | remove",
            "slug": "str (optional) — the book slug",
            "chapter": "int (optional, for add) — chapter number",
            "offset": "int (optional, 0, for add) — char offset",
            "label": "str (optional, for add) — short label",
            "id": "int (optional, for remove) — bookmark id",
        },
    )
    def library_bookmark(action: str, *, slug: str = "", chapter: int = 0,
                         offset: int = 0, label: str = "",
                         id: int = 0) -> dict[str, Any]:
        lib = _library()
        act = (action or "").strip().lower()
        if act == "add":
            if not slug or chapter < 1:
                raise ValueError("bookmark add needs slug and chapter")
            return lib.add_bookmark(slug, chapter, offset, label)
        if act == "list":
            return {"bookmarks": lib.list_bookmarks(slug)}
        if act == "remove":
            if id < 1:
                raise ValueError("bookmark remove needs id")
            return lib.remove_bookmark(id)
        raise ValueError(f"unknown bookmark action {action!r} — add | list | remove")

    @registry.register(
        "library_note",
        description=(
            "Annotations/notes on an ingested book. action=add saves one "
            "(chapter + offset + quoted passage + the note); list shows them; "
            "remove deletes by id."
        ),
        capability=Capability.FS_READ,
        parameters={
            "action": "str — add | list | remove",
            "slug": "str (optional) — the book slug",
            "chapter": "int (optional, for add) — chapter number",
            "offset": "int (optional, 0, for add) — char offset",
            "quote": "str (optional, for add) — the passage being annotated",
            "note": "str (for add) — the annotation text",
            "id": "int (optional, for remove) — note id",
        },
    )
    def library_note(action: str, *, slug: str = "", chapter: int = 0,
                     offset: int = 0, quote: str = "", note: str = "",
                     id: int = 0) -> dict[str, Any]:
        lib = _library()
        act = (action or "").strip().lower()
        if act == "add":
            if not slug or chapter < 1:
                raise ValueError("note add needs slug and chapter")
            return lib.add_note(slug, chapter, offset, quote=quote, note=note)
        if act == "list":
            return {"notes": lib.list_notes(slug)}
        if act == "remove":
            if id < 1:
                raise ValueError("note remove needs id")
            return lib.remove_note(id)
        raise ValueError(f"unknown note action {action!r} — add | list | remove")

    @registry.register(
        "library_shelf",
        description=(
            "Book collections/shelves. create makes one; add/remove put books "
            "on it; list shows all shelves; show lists the books on a shelf "
            "with progress and ratings; delete removes the shelf (not books)."
        ),
        capability=Capability.FS_READ,
        parameters={
            "action": "str — create | delete | add | remove | list | show",
            "name": "str (optional) — collection name",
            "slug": "str (optional, for add/remove) — the book slug",
        },
    )
    def library_shelf(action: str, *, name: str = "",
                      slug: str = "") -> dict[str, Any]:
        lib = _library()
        act = (action or "").strip().lower()
        if act == "create":
            return lib.create_collection(name)
        if act == "delete":
            return lib.delete_collection(name)
        if act == "add":
            return lib.add_to_collection(name, slug)
        if act == "remove":
            return lib.remove_from_collection(name, slug)
        if act == "list":
            return {"collections": lib.list_collections()}
        if act == "show":
            return lib.shelf(name)
        raise ValueError(
            f"unknown shelf action {action!r} — create | delete | add | remove | list | show")

    @registry.register(
        "library_tag",
        description=(
            "Tags on library books. set replaces a book's tags (comma string "
            "or list); get shows a book's tags; list shows every tag with "
            "counts; books finds every book with a tag."
        ),
        capability=Capability.FS_READ,
        parameters={
            "action": "str — set | get | list | books",
            "slug": "str (for set/get) — the book slug",
            "tags": "str | list (for set) — tags, comma-separated or a list",
            "tag": "str (for books) — the tag to search",
        },
    )
    def library_tag(action: str, *, slug: str = "", tags: Any = "",
                    tag: str = "") -> dict[str, Any]:
        lib = _library()
        act = (action or "").strip().lower()
        if act == "set":
            return lib.set_tags(slug, tags)
        if act == "get":
            return {"slug": slug, "tags": lib.get_tags(slug)}
        if act == "list":
            return {"tags": lib.list_tags()}
        if act == "books":
            return {"tag": tag, "books": lib.books_with_tag(tag)}
        raise ValueError(
            f"unknown tag action {action!r} — set | get | list | books")

    @registry.register(
        "library_rate",
        description=(
            "Star rating for a library book, 1–5. Re-rating replaces the "
            "old rating. Pass stars=0 to read the current rating."
        ),
        capability=Capability.FS_READ,
        parameters={
            "slug": "str — the book slug",
            "stars": "int — 1–5 to rate, 0 to read the current rating",
        },
    )
    def library_rate(slug: str, *, stars: int = 0) -> dict[str, Any]:
        lib = _library()
        if stars == 0:
            return {"slug": slug, "stars": lib.get_rating(slug)}
        return lib.rate(slug, stars)

    # ── collaborative writing (owner + brain + characters) ──────────────
    def _collab() -> Any:
        from .collab import CollaborativeSession
        return CollaborativeSession

    def _characters() -> Any:
        from ..characters.store import CharacterStore
        return CharacterStore()

    @registry.register(
        "collab_scene",
        description=(
            "A character agent writes a scene from their own POV in a story — "
            "their voice, biases, blind spots. The character remembers writing it."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "story_slug": "str — the story",
            "character": "str — character name (must exist in the character store)",
            "prompt": "str — the scene prompt",
        },
    )
    def collab_scene(story_slug: str, character: str, prompt: str) -> dict[str, Any]:
        from .collab import CollaborativeSession
        store = _characters()
        char = store.get(character) or store.get_by_name(character)
        if char is None:
            return {"ok": False, "reason": f"character '{character}' not found"}
        sess = CollaborativeSession(story_slug, context)
        sess.cast_character(char)
        return sess.write_scene(character, prompt)

    @registry.register(
        "collab_critique",
        description=(
            "Run the brutal-but-fair critic agent on a chapter draft: pacing, "
            "voice consistency, plot mechanics, show-don't-tell. Returns score, "
            "issues, suggestions, and a ship/revise/rewrite verdict."
        ),
        capability=Capability.FS_READ,
        parameters={
            "text": "str — the draft chapter text",
            "brief": "str — what the chapter was supposed to do",
        },
    )
    def collab_critique(text: str, brief: str = "") -> dict[str, Any]:
        from .collab import CollaborativeSession
        sess = CollaborativeSession("", context)
        return sess.critique(text, {"summary": brief})

    # ── story branches (what-if timelines) ───────────────────────────────
    @registry.register(
        "branch_fork",
        description=(
            "Fork a what-if timeline from a story chapter: test a major plot "
            "change (a death, a betrayal, an alternate path) without touching canon."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "story_slug": "str — the story",
            "branch": "str — name for this timeline",
            "from_chapter": "int — chapter to fork from",
            "premise": "str — the what-if premise",
        },
    )
    def branch_fork(story_slug: str, branch: str, from_chapter: int,
                    premise: str) -> dict[str, Any]:
        from .branches import StoryBranch
        return StoryBranch(story_slug, branch).fork(from_chapter, premise)

    @registry.register(
        "branch_write",
        description="Write a chapter in a what-if branch timeline.",
        capability=Capability.FS_WRITE,
        parameters={
            "story_slug": "str", "branch": "str",
            "text": "str — chapter text", "title": "str (optional)",
        },
    )
    def branch_write(story_slug: str, branch: str, text: str,
                     title: str = "") -> dict[str, Any]:
        from .branches import StoryBranch
        return StoryBranch(story_slug, branch).add_chapter(text, title)

    @registry.register(
        "branch_list",
        description="List all what-if timelines for a story.",
        capability=Capability.FS_READ,
        parameters={"story_slug": "str"},
    )
    def branch_list(story_slug: str) -> dict[str, Any]:
        from .branches import list_branches
        return {"branches": list_branches(story_slug)}

    @registry.register(
        "branch_merge",
        description=(
            "Merge a what-if branch back into canon (or abandon it). "
            "Merging marks the timeline adopted; abandon deletes it."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "story_slug": "str", "branch": "str",
            "action": "str — merge | abandon",
        },
    )
    def branch_merge(story_slug: str, branch: str,
                     action: str = "merge") -> dict[str, Any]:
        from .branches import StoryBranch
        b = StoryBranch(story_slug, branch)
        if action.strip().lower() == "abandon":
            return b.abandon()
        return b.merge()

    # ── serialized publishing ────────────────────────────────────────────
    @registry.register(
        "publish_start",
        description=(
            "Start serializing a story: chapters drip-publish on a cadence "
            "(daily/weekly/manual) like a webnovel. Readers can follow; "
            "reactions feed back into story direction."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "story_slug": "str",
            "cadence": "str — daily | weekly | manual",
            "title": "str (optional) — public title",
            "announce_chat": "str (optional) — chat to announce releases in",
        },
    )
    def publish_start(story_slug: str, cadence: str = "daily", title: str = "",
                      announce_chat: str = "") -> dict[str, Any]:
        from .publish import SerialPublication
        return SerialPublication(story_slug).start(cadence, title, announce_chat)

    @registry.register(
        "publish_release",
        description=(
            "Release the next due chapter of a serial publication. Returns the "
            "chapter text for delivery to followers."
        ),
        capability=Capability.FS_WRITE,
        parameters={"story_slug": "str"},
    )
    def publish_release(story_slug: str) -> dict[str, Any]:
        from .publish import SerialPublication
        return SerialPublication(story_slug).release()

    @registry.register(
        "publish_status",
        description="Status of a serial publication: released/total, followers, feedback.",
        capability=Capability.FS_READ,
        parameters={"story_slug": "str"},
    )
    def publish_status(story_slug: str) -> dict[str, Any]:
        from .publish import SerialPublication
        return SerialPublication(story_slug).status()

    @registry.register(
        "publish_follow",
        description="Follow or unfollow a serial publication as a reader.",
        capability=Capability.FS_WRITE,
        parameters={
            "story_slug": "str", "reader": "str — reader id",
            "action": "str — follow | unfollow",
        },
    )
    def publish_follow(story_slug: str, reader: str,
                       action: str = "follow") -> dict[str, Any]:
        from .publish import SerialPublication
        pub = SerialPublication(story_slug)
        if action.strip().lower() == "unfollow":
            return pub.unfollow(reader)
        return pub.follow(reader)

    @registry.register(
        "publish_feedback",
        description=(
            "A reader reacts to a released chapter (like/love/hype/meh/skip + "
            "optional comment). Aggregates into per-chapter sentiment for the author."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "story_slug": "str", "reader": "str",
            "chapter": "int", "reaction": "str",
            "comment": "str (optional)",
        },
    )
    def publish_feedback(story_slug: str, reader: str, chapter: int,
                         reaction: str, comment: str = "") -> dict[str, Any]:
        from .publish import SerialPublication
        return SerialPublication(story_slug).feedback(reader, chapter, reaction, comment)

    @registry.register(
        "publish_feedback_summary",
        description=(
            "Per-chapter reader sentiment for a serial: reaction counts and "
            "comments, so the author sees what landed and what didn't."
        ),
        capability=Capability.FS_READ,
        parameters={"story_slug": "str"},
    )
    def publish_feedback_summary(story_slug: str) -> dict[str, Any]:
        from .publish import SerialPublication
        return SerialPublication(story_slug).feedback_summary()
