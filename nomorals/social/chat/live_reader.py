"""The standalone shared live reader (Doc 16 residual).

``ReaderService`` (P2) is the reading engine — sessions, paging,
contents, revision pinning, token discipline, plain-text renders —
and ``OutputRouter`` (P4) is the routing decision plus the
routed-content store.  But the *live reading* behaviour — opening a
document for live reading, actuating a reading token, reopening a
routed output, delivering its file actions — was spread across the
router's ``plan()``, the ``/read`` desk and the ``/books`` desk,
each hand-rolling the same session lookup + act + render sequence
against two separately constructed ``ReaderService`` instances (the
router's own ``router.reader`` and the runtime's
``_reader_service()``).  The doc-queue audit named that spread the
residual: the "shared chat reader" as its own module.

``LiveReader`` is that module: one shared reader over the shipped
engine.  It owns nothing the engine already owns — no second
session store, no second pager, no second token discipline.  It
owns the *live* contract the desks share:

* one construction site (``for_context``) whose service IS the
  router's service, so a session opened by a routed send and a
  session actuated by ``/read go`` are the same session, not two
  in-memory copies of one store file;
* resolver dispatch by document prefix — ``text:`` documents
  resolve through the router's routed-content store, ``story:``
  documents through an injected story resolver (the books layer
  owns that resolver; this module never imports it);
* live reads: every view is resolved at read time, so the page
  under the reader is the source's current text; a segment's
  production state is labelled from its manifest only, and a
  newer revision is flagged, never silently adopted — the
  revision switch (token verb ``u``) is explicit and honest
  about where the reader lands;
* the file actions (``/read get``) and the reopen/list desk
  (``/read open`` / ``/read``) over the same sessions.

Layer: social (L4), beside output_router and reader_service —
imports core (L1) and its own package only; resolvers for
upper-layer documents arrive injected.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...core.longdoc import LongDocumentManifest
from .output_router import (
    SEND_CHUNK_CHARS,
    OutputRouter,
    RoutePlan,
)
from .reader_service import (
    ContentsView,
    PageView,
    ReaderService,
    ReaderSession,
)

__all__ = [
    "LiveReader", "LiveReadResult", "LiveReaderFault",
]

#: What a live read can return — mirrors the engine's own
#: ``ActResult`` statuses, plus "note" for a desk-level refusal
#: (wrong surface, expired file control, empty list) that carries
#: no view at all.
READ_OK = "ok"
READ_STALE = "stale"
READ_UNKNOWN = "unknown"
READ_NOTE = "note"

#: Which chat desk a live read answers on — the expired-control and
#: wrong-document notes name the desk that can reopen the document.
SURFACE_READ = "read"
SURFACE_BOOKS = "books"

#: File-action captions, per format — the desk's own words, owned
#: here so ``/read get`` has one source for them.
_FILE_CAPTIONS = {
    "html": (
        "🌐 the branded HTML edition — complete, "
        "with its completeness stated on the cover"),
    "pdf": (
        "📕 the branded PDF edition — the designed "
        "watermark on every page, its completeness "
        "stated in the front matter"),
    "epub": (
        "📗 the branded EPUB edition — its "
        "completeness stated on the title page"),
    "txt": "⬇ the original output, byte for byte",
}


class LiveReaderFault(Exception):
    """The live reader could not do what was asked."""


@dataclass
class LiveReadResult:
    """The outcome of one live-read actuation, ready to send.

    ``status`` is "ok" (a page, contents, or file follows),
    "stale" (an older page's token — the reader's place has not
    moved), "unknown" (an expired or foreign token), or "note"
    (a desk-level refusal whose whole answer is ``text``).
    ``kind`` names what ``text`` is: "page", "contents", "file"
    or "note".  For a file, ``file`` is ``(path, name, caption)``
    and the caller sends it through the gateway."""

    status: str
    kind: str
    text: str = ""
    session: ReaderSession | None = None
    view: PageView | None = None
    contents: ContentsView | None = None
    file: tuple[Path, str, str] | None = None


@dataclass
class _Gate:
    """The surface gate for one document kind: which prefix it
    serves and what it says when a control belongs elsewhere."""

    prefix: str
    expired_note: str
    foreign_note: str


_GATES = {
    SURFACE_READ: _Gate(
        prefix="text:",
        expired_note=("that reading control has expired — "
                      "/read lists what this chat has open"),
        foreign_note=("that reading control belongs to a "
                      "different document — story controls "
                      "answer to /books go")),
    SURFACE_BOOKS: _Gate(
        prefix="story:",
        expired_note=("that reading control has expired — "
                      "/books read <slug> opens the story again"),
        foreign_note=("that reading control belongs to a "
                      "different document")),
}


class LiveReader:
    """One shared live reader over the shipped reading engine.

    Construct with an explicit root or via :meth:`for_context`;
    ``root=None`` keeps everything in memory (tests, ephemeral
    callers).  The service is the router's service unless a
    service is passed explicitly — never a second instance over
    the same store."""

    def __init__(self, root: Path | None = None, *,
                 router: OutputRouter | None = None,
                 service: ReaderService | None = None,
                 ) -> None:
        self.root = Path(root) if root else None
        self.router = router or OutputRouter(
            self.root, reader=service)
        self.service = self.router.reader

    @classmethod
    def for_context(cls, context: Any) -> LiveReader:
        """The live reader bound to a runtime context: the router
        (and its reader sessions) persist under
        ``<workspace>/longdoc/`` — the same derivation the router
        and the reader service use, so every desk shares one
        store."""
        return cls(router=OutputRouter.for_context(context))

    # ── the shared service ───────────────────────────────────────

    @property
    def reader(self) -> ReaderService:
        """The shared reading engine, for the desks that need a
        session-level call the live contract does not cover (the
        books part-continuation's fetch-then-reopen dance)."""
        return self.service

    # ── opening live reads ───────────────────────────────────────

    def open_text(self, text: str, *, reader_key: str,
                  platform: object = "", limit: int = SEND_CHUNK_CHARS,
                  intent: str = "", kind: str = "",
                  title: str = "") -> RoutePlan:
        """Route ``text`` to a chat as a live read: the router's
        decision (chunks / document / reader / reader+edition) with
        the reader session opened and the introduction built.  The
        routing decision stays in the router; this is the shared
        entry the runtime seam calls instead of building its own
        router."""
        return self.router.plan(
            text, platform=platform, reader_key=reader_key,
            limit=limit, intent=intent, kind=kind, title=title)

    def install_epub_renderer(
            self, renderer: Callable[[Any, str], bytes] | None) -> bool:
        """Install the EPUB rendition renderer on the router
        (idempotent): the EPUB export generator lives in books, a
        layer above the router, so it arrives injected — and the
        surface that owns ``/read`` does the installing.  True
        when this call installed it, False when one was already
        installed."""
        if getattr(self.router, "epub_renderer", None) is not None:
            return False
        self.router.epub_renderer = renderer
        return True

    # ── actuating a live read ────────────────────────────────────

    def _resolver_for(self, document_id: str, *,
                      story_resolver: Any = None):
        """The resolver for a document, by its prefix — the
        routed-content store for ``text:`` documents, the injected
        story resolver for ``story:`` documents, None when this
        reader has no resolver for the kind."""
        if document_id.startswith("text:"):
            return self.router.resolver()
        if document_id.startswith("story:"):
            return story_resolver
        return None

    def read(self, token: str, verb: str, *,
             surface: str = SURFACE_READ,
             story_resolver: Any = None) -> LiveReadResult:
        """Actuate one reading token on its desk: ``n`` next,
        ``p`` prev, ``c`` contents, ``j<row>`` jump to a contents
        row, ``u`` switch to the newer revision.  The token's
        session decides the document; a control for the other
        desk's document answers with the desk's own note, never
        by acting on it.  A stale token is an honest no-op — the
        reader's place has not moved.  The returned text is the
        plain-transport render (a page, or the contents with the
        desk's open hint); the view/contents ride along for
        desks with their own render."""
        gate = _GATES.get(surface) or _GATES[SURFACE_READ]
        session = self.service.session_by_token(token)
        if session is None:
            return LiveReadResult(
                status=READ_UNKNOWN, kind=READ_NOTE,
                text=gate.expired_note)
        if not session.document_id.startswith(gate.prefix):
            return LiveReadResult(
                status=READ_NOTE, kind=READ_NOTE,
                text=gate.foreign_note, session=session)
        resolver = self._resolver_for(
            session.document_id, story_resolver=story_resolver)
        if resolver is None:
            return LiveReadResult(
                status=READ_NOTE, kind=READ_NOTE,
                text=("that reading control belongs to a story "
                      "whose reader is not attached here — open "
                      "it with /books read"),
                session=session)
        result = self.service.act(token, verb, resolver)
        if result.status != READ_OK:
            return LiveReadResult(
                status=result.status, kind=READ_NOTE,
                text=result.note, session=session)
        if result.contents is not None:
            if surface == SURFACE_BOOKS:
                slug = session.document_id[len("story:"):]
                hint = (f"open one: /books read {slug} <chapter> "
                        "(an unheld chapter is fetched on read)")
            else:
                hint = (f"open one: /read go {session.token} "
                        "j<line number>")
            return LiveReadResult(
                status=READ_OK, kind="contents", session=session,
                contents=result.contents,
                text=self.service.render_contents_text(
                    result.contents, open_hint=hint))
        view = result.view
        if view is None:
            return LiveReadResult(
                status=READ_NOTE, kind=READ_NOTE,
                text="that reading control answered nothing",
                session=session)
        return LiveReadResult(
            status=READ_OK, kind="page", session=session, view=view,
            text=self.service.render_text(view))

    def contents_for(self, manifest: LongDocumentManifest, *,
                     current_segment_id: str = "",
                     open_hint: str = "") -> str:
        """The contents render of a manifest — held segments with
        their sizes, absent ones named as absent, the
        completeness truth up top.  The desk derives the manifest
        (its producer's derivation); the reader renders it."""
        view = self.service.contents_view(
            manifest, current_segment_id=current_segment_id)
        return self.service.render_contents_text(
            view, open_hint=open_hint)

    # ── reopening + listing ──────────────────────────────────────

    def recent(self, reader_key: str, limit: int = 10) -> list[dict]:
        """This reader's routed documents, newest first — journal
        rows for reader/reader+edition/document routes (the ones
        with a document to reopen)."""
        return self.router.recent(reader_key, limit=limit)

    def open_recent(self, reader_key: str, n: int) -> LiveReadResult:
        """Reopen a routed output from the list, at the reader's
        current page for it — the view resolved live, so a
        revision that arrived since is flagged on the page, never
        silently adopted."""
        rows = self.router.recent(reader_key)
        if not 1 <= int(n) <= len(rows):
            return LiveReadResult(
                status=READ_NOTE, kind=READ_NOTE,
                text=("usage: /read open <n> — n from the "
                      "/read list"))
        row = rows[int(n) - 1]
        session = self.service.session_for(
            reader_key, row.get("document_id", ""))
        if session is None:
            return LiveReadResult(
                status=READ_NOTE, kind=READ_NOTE,
                text=("that output's reading session is "
                      "gone — its text is still in the store"))
        view = self.service.view(session.session_id,
                                 self.router.resolver())
        return LiveReadResult(
            status=READ_OK, kind="page", session=session, view=view,
            text=self.service.render_text(view))

    def list_text(self, reader_key: str) -> str:
        """This chat's routed outputs, newest first, with the
        reopen pointers — the ``/read`` list."""
        rows = self.router.recent(reader_key)
        if not rows:
            return ("no routed long outputs in this chat yet — "
                    "when a report or result runs long, it "
                    "lands here with reading controls instead "
                    "of a wall of chunks")
        lines = ["📚 routed outputs in this chat (newest first):"]
        for i, row in enumerate(rows, start=1):
            title = row.get("title") or "long output"
            chars = row.get("chars", 0)
            lines.append(
                f"{i}. {title} — {chars:,} chars "
                f"· {row.get('route', '')}")
            lines.append(f"   open: /read open {i}")
        return "\n".join(lines)

    # ── file actions ─────────────────────────────────────────────

    def file_for(self, token: str, fmt: str) -> LiveReadResult:
        """The file action for a reading token: the branded HTML,
        PDF or EPUB edition, or the raw original — ``(path, name,
        caption)`` in ``file`` when the edition builds, else a
        note naming why not.  The caller sends the file through
        the gateway; this module performs no network I/O."""
        fmt = (fmt or "").lower()
        session = self.service.session_by_token(token)
        if session is None:
            return LiveReadResult(
                status=READ_UNKNOWN, kind=READ_NOTE,
                text=("that file control has expired — /read "
                      "lists what this chat has open"))
        if not session.document_id.startswith("text:"):
            return LiveReadResult(
                status=READ_NOTE, kind=READ_NOTE,
                text=("that file control belongs to a "
                      "different document"),
                session=session)
        found = self.router.file_for(session.document_id, fmt)
        if found is None:
            return LiveReadResult(
                status=READ_NOTE, kind=READ_NOTE,
                text=("that output is no longer in the store "
                      "— nothing to send"),
                session=session)
        path, name = found
        caption = _FILE_CAPTIONS.get(fmt, _FILE_CAPTIONS["txt"])
        return LiveReadResult(
            status=READ_OK, kind="file", session=session,
            file=(path, name, caption))
