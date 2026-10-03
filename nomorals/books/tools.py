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
            "Start a book: gather live research notes on the topic and write a "
            "real outline (chapters + beats). Saved to disk — resumable. Returns "
            "the plan."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "topic": "str — what the book is about",
            "title": "str (optional) — the book title (else derived from topic)",
            "chapters": "int (optional, 8) — chapters to plan (3-16)",
            "words_per_chapter": "int (optional, 1200)",
            "author": "str (optional)",
            "genre": "str (optional) — tone/genre",
            "research": "bool (optional, true) — gather live research notes",
        },
    )
    def book_create(topic: str, *, title: str = "", chapters: int = 8,
                    words_per_chapter: int = 1200, author: str = "",
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
