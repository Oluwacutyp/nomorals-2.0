"""Generalized long-document reader service (Doc 16, Pass 2).

One reader over any :class:`~nomorals.core.longdoc.LongDocumentManifest`
— a followed story, a forge book, a documents-engine Document, a
presentation Doc, a raw long text.  Producers derive the manifest
(P1's derivations, beside each producer); the caller injects a
*resolver* that turns a manifest segment back into its canonical
readable content.  The service itself imports no producer: it owns
paging, contents, per-reader position, revision pinning and the
callback discipline, and nothing else.

Position belongs to **reader + document** (doc §5): a session is
keyed by ``(reader_key, document_id)`` and persists — segment +
char offset + the pinned manifest snapshot — in a JSON store, so a
reader's place survives restarts and is never another reader's
place.  The pinned manifest *is* the pinned revision: when a fresh
manifest for the same document arrives and the segment under the
reader is byte-identical in it, the session adopts the growth
silently (new chapters become reachable; the page under the reader
does not move).  When that segment's content changed, the session
keeps its pinned revision and flags ``update_available`` — the
reader is told, never yanked.  :meth:`ReaderService.switch_revision`
adopts the newer revision explicitly, re-anchoring by segment id;
a segment that vanished from the newer revision re-anchors to the
*nearest surviving* segment and the returned view says so — a
place is never silently relocated.

A segment's production state, where its producer records one, is
labelled from the manifest's own metadata and nothing else (Doc 16
P6): a forge chapter still ``writing`` is a draft, a ``planned``
chapter with no text is not yet written, a presentation section in
any state but ``ready`` is not final.  Producers that record no
state (stories, parsed documents, raw text) get no label — the
reader never infers a state the source did not declare.

Callbacks are short opaque tokens bound to the session and its
navigation generation: every successful navigation rotates the
token, so a duplicate tap (the same token twice) and an out-of-order
tap (an older render's token) both resolve as *stale* — an honest
no-op, never a jump.  Tokens carry no document or segment ids; the
payload is ``<prefix> <token> <verb>`` where the wiring layer owns
the prefix (``books go`` for the /books surface).

Paging rides the lossless chunker (:func:`.chunking.chunk_message`):
the pages of a segment partition its content exactly, so reading
every page end to end yields the full segment byte for byte — on
pager surfaces and plain transports alike.  Nothing here truncates:
a segment the resolver cannot serve renders as *unavailable*, a
declared-but-unheld segment renders as *absent* in contents, and
both are named, never filled in.

Layer: social (L4), beside present.py — imports core (L1) and its
own package only; resolvers are injected by the layers above.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...core.longdoc import LongDocumentManifest, ManifestSegment
from .chunking import chunk_message
from .failures import record_failure

__all__ = [
    "DEFAULT_PAGE_CHARS", "ReaderService", "ReaderSession",
    "PageView", "ContentsView", "ContentsRow", "ActResult",
    "ReaderFault", "content_resolver_from_map",
]

#: Chars of segment content per reader page — the reading pager's
#: established page budget (menu-session 3200 minus header/footer
#: headroom), used on every transport so a page is a page.
DEFAULT_PAGE_CHARS = 3000

#: A segment resolver: canonical readable content for a segment, or
#: None when the source cannot serve it (not held, deleted, gone).
Resolver = Callable[[ManifestSegment], "str | None"]

_KIND_WORDS = {
    "chapter": "chapter", "section": "section", "table": "table",
    "summary": "summary", "overflow": "overflow", "text": "document",
}


def _kind_word(kind: str) -> str:
    return _KIND_WORDS.get(kind or "", "part")


def _segment_state(seg: ManifestSegment) -> str:
    """The not-final production state a segment's manifest record
    proves, or "" when the record declares the segment final (or
    declares nothing about state at all).

    Labels come only from metadata producers actually record — a
    forge chapter's ``status`` (``writing`` = a draft with partial
    text being served; ``planned`` with no text = declared but not
    yet written) and a presentation section's ``state`` (any state
    but ``ready`` renders its state line instead of content, so it
    is not final).  A *present* segment whose status reads
    ``planned`` is contradictory source data (content exists that
    the plan never marked written) — it gets no label rather than
    an invented reconciliation."""
    md = seg.metadata or {}
    status = str(md.get("status") or "").strip().lower()
    if status == "writing":
        return "draft — still being written"
    if status == "planned" and not seg.present:
        return "planned — not yet written"
    state = str(md.get("state") or "").strip().lower()
    if state and state != "ready":
        return f"not final — {state}"
    return ""


def _segment_label(seg: ManifestSegment) -> str:
    """An owner-worded name for a segment: its title, else its kind
    + number.  Never a segment id."""
    if seg.title:
        return seg.title
    num = _chapter_number(seg.segment_id)
    word = _kind_word(seg.kind)
    return f"{word} {num}" if num is not None else word


def _session_id(reader_key: str, document_id: str) -> str:
    return hashlib.sha256(
        f"{reader_key}|{document_id}".encode()).hexdigest()[:16]


def _mint_token(session_id: str, generation: int, document_id: str) -> str:
    return hashlib.sha256(
        f"{session_id}|{generation}|{document_id}".encode()
    ).hexdigest()[:12]


class ReaderFault(Exception):
    """The reader service could not do what was asked."""


@dataclass
class ReaderSession:
    """One reader's place in one document, pinned at one revision."""

    session_id: str
    reader_key: str
    document_id: str
    source_kind: str
    manifest: dict[str, Any]  # pinned manifest snapshot (to_dict form)
    segment_id: str = ""
    offset: int = 0  # char offset into the segment — always a page start
    page_size: int = DEFAULT_PAGE_CHARS
    generation: int = 1
    token: str = ""
    token_history: list[str] = field(default_factory=list)
    command_prefix: str = "read go"
    update_available: bool = False
    latest: dict[str, Any] | None = None  # newer manifest, when flagged
    updated_at: float = 0.0

    def pinned(self) -> LongDocumentManifest:
        return LongDocumentManifest.from_dict(self.manifest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "reader_key": self.reader_key,
            "document_id": self.document_id,
            "source_kind": self.source_kind,
            "manifest": self.manifest,
            "segment_id": self.segment_id, "offset": self.offset,
            "page_size": self.page_size, "generation": self.generation,
            "token": self.token,
            "token_history": list(self.token_history),
            "command_prefix": self.command_prefix,
            "update_available": self.update_available,
            "latest": self.latest, "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReaderSession:
        return cls(
            session_id=str(data.get("session_id", "")),
            reader_key=str(data.get("reader_key", "")),
            document_id=str(data.get("document_id", "")),
            source_kind=str(data.get("source_kind", "")),
            manifest=dict(data.get("manifest", {}) or {}),
            segment_id=str(data.get("segment_id", "")),
            offset=int(data.get("offset", 0) or 0),
            page_size=int(data.get("page_size", 0) or DEFAULT_PAGE_CHARS),
            generation=int(data.get("generation", 1) or 1),
            token=str(data.get("token", "")),
            token_history=[str(t) for t in
                           data.get("token_history", []) or []],
            command_prefix=str(data.get("command_prefix", "read go")),
            update_available=bool(data.get("update_available", False)),
            latest=data.get("latest"),
            updated_at=float(data.get("updated_at", 0.0) or 0.0),
        )


@dataclass
class ContentsRow:
    """One contents line: a real segment, or a declared-but-unheld
    chapter synthesized from the manifest's missing ranges."""

    ordinal: int  # 1-based row number (the jump address)
    segment_id: str
    kind: str
    title: str
    present: bool
    char_count: int
    current: bool = False
    state: str = ""  # not-final production state, manifest-proven only


@dataclass
class ContentsView:
    document_id: str
    title: str
    rows: list[ContentsRow]
    completeness_status: str
    declared_total: int | None
    present_count: int
    note: str = ""
    revision_ref: str = ""
    update_available: bool = False


@dataclass
class PageView:
    """One rendered page of one segment, plus its control model."""

    status: str  # "ok" | "unavailable" | "empty"
    session_id: str
    document_id: str
    title: str
    revision_ref: str
    segment_id: str = ""
    segment_title: str = ""
    segment_kind: str = ""
    segment_state: str = ""  # not-final state, manifest-proven only
    segment_ordinal: int = 0  # 1-based among the manifest's segments
    segment_count: int = 0
    declared_total: int = 0  # segments the source declares (0 = undeclared)
    body: str = ""
    page_index: int = 0  # 0-based within the segment
    page_count: int = 0
    offset: int = 0
    segment_chars: int = 0
    has_prev_page: bool = False
    has_next_page: bool = False
    has_prev_segment: bool = False
    has_next_segment: bool = False
    next_segment_title: str = ""
    next_segment_kind: str = ""
    update_available: bool = False
    token: str = ""
    generation: int = 0
    note: str = ""

    def buttons(self, prefix: str) -> list[list[tuple[str, str]]]:
        """Button rows in the shipped actionable grammar: (label,
        callback payload) pairs whose payloads re-enter a handler
        as ``<prefix> <token> <verb>`` commands."""
        rows: list[list[tuple[str, str]]] = []
        nav: list[tuple[str, str]] = []
        if self.has_prev_page or self.has_prev_segment:
            nav.append(("◀ Prev", f"{prefix} {self.token} p"))
        if self.has_next_page or self.has_next_segment:
            nav.append(("Next ▶", f"{prefix} {self.token} n"))
        if nav:
            rows.append(nav)
        rows.append([("📑 Contents", f"{prefix} {self.token} c")])
        return rows


@dataclass
class ActResult:
    """The outcome of a token actuation."""

    status: str  # "ok" | "stale" | "unknown"
    view: PageView | None = None
    contents: ContentsView | None = None
    note: str = ""


def content_resolver_from_map(
        contents: dict[str, str]) -> Resolver:
    """A resolver over an in-memory ``content_ref → text`` map —
    the shape tests and ephemeral producers (raw text, present Docs
    already in hand) resolve through."""
    def resolve(segment: ManifestSegment) -> str | None:
        if not segment.present:
            return None
        return contents.get(segment.content_ref)
    return resolve


class ReaderService:
    """Sessions, paging, contents and callback discipline over
    long-document manifests.  Construct with an explicit store path
    or via :meth:`for_context`; ``store_path=None`` keeps sessions
    in memory only (ephemeral readers, tests)."""

    def __init__(self, store_path: Path | None = None) -> None:
        self._path = Path(store_path) if store_path else None
        self._lock = threading.RLock()
        self._sessions: dict[str, ReaderSession] = {}
        self._load()

    @classmethod
    def for_context(cls, context: Any) -> ReaderService:
        """The service bound to a runtime context: sessions persist
        under ``<workspace>/longdoc/reader_sessions.json`` (the
        workspace derivation mirrors StoryReader's: the context
        settings' workspace_dir, else ``./workspace``)."""
        settings = getattr(context, "settings", None)
        root = getattr(settings, "workspace_dir", None) if settings else None
        workspace = Path(root) if root else Path.cwd() / "workspace"
        return cls(workspace / "longdoc" / "reader_sessions.json")

    # ── persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - a corrupt store
            # restarts empty rather than breaking reading itself —
            # but the failure is NAMED in the package ledger, never
            # silent (a vanished reading place must be explainable)
            record_failure("reader_service.load_store", exc)
            return
        for raw in data.get("sessions", []) or []:
            try:
                session = ReaderSession.from_dict(raw)
            except Exception as exc:  # noqa: BLE001 - skip a bad row
                record_failure("reader_service.load_session_row", exc)
                continue
            if session.session_id:
                self._sessions[session.session_id] = session

    def _save(self) -> None:
        if self._path is None:
            return
        payload = {"sessions": [s.to_dict()
                                for s in self._sessions.values()]}
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False),
                           encoding="utf-8")
            os.replace(tmp, self._path)
        except Exception:  # noqa: BLE001 - persistence is best-effort;
            pass  # the in-memory session still serves this process

    # ── session lifecycle ────────────────────────────────────────────

    def session_for(self, reader_key: str,
                    document_id: str) -> ReaderSession | None:
        with self._lock:
            return self._sessions.get(_session_id(reader_key, document_id))

    def open(self, reader_key: str, manifest: LongDocumentManifest, *,
             segment_id: str = "",
             page_size: int = DEFAULT_PAGE_CHARS,
             command_prefix: str = "read go") -> ReaderSession:
        """Open (or resume) ``reader_key``'s session on ``manifest``.

        Resuming keeps the reader's place.  A fresh manifest of the
        same document is adopted silently when the segment under the
        reader is unchanged in it (growth elsewhere never moves the
        page); when that segment changed, the pinned revision stands
        and the session flags the newer one instead.  Passing
        ``segment_id`` is an explicit jump (a read action) and moves
        the cursor to that segment's first page."""
        with self._lock:
            sid = _session_id(reader_key, manifest.document_id)
            session = self._sessions.get(sid)
            if session is None:
                session = ReaderSession(
                    session_id=sid, reader_key=reader_key,
                    document_id=manifest.document_id,
                    source_kind=manifest.source_kind,
                    manifest=manifest.to_dict(),
                    page_size=max(200, int(page_size or 0)
                                  or DEFAULT_PAGE_CHARS),
                    command_prefix=command_prefix or "read go",
                )
                session.segment_id = self._landing_segment(
                    manifest, segment_id)
                session.token = _mint_token(
                    sid, session.generation, manifest.document_id)
                self._sessions[sid] = session
            else:
                session.page_size = max(
                    200, int(page_size or 0) or session.page_size)
                if command_prefix:
                    session.command_prefix = command_prefix
                pinned = session.pinned()
                if manifest.revision_ref != pinned.revision_ref:
                    cur = pinned.segment(session.segment_id)
                    fresh = manifest.segment(session.segment_id)
                    if (cur is not None and fresh is not None
                            and cur.present and fresh.present
                            and cur.sha256 == fresh.sha256):
                        # the page under the reader is byte-identical
                        # in the newer revision — adopt the growth
                        session.manifest = manifest.to_dict()
                        session.update_available = False
                        session.latest = None
                    else:
                        session.update_available = True
                        session.latest = manifest.to_dict()
                if segment_id:
                    self._jump(session, segment_id)
            session.updated_at = time.time()
            self._save()
            return session

    @staticmethod
    def _landing_segment(manifest: LongDocumentManifest,
                         wanted: str) -> str:
        if wanted and manifest.segment(wanted) is not None:
            return wanted
        for seg in manifest.segments:
            if seg.present:
                return seg.segment_id
        return manifest.segments[0].segment_id if manifest.segments else ""

    def _jump(self, session: ReaderSession, segment_id: str) -> bool:
        if segment_id == session.segment_id and session.offset == 0:
            return False
        session.segment_id = segment_id
        session.offset = 0
        return True

    def _rotate(self, session: ReaderSession) -> None:
        """A successful navigation: bump the generation and rotate
        the token, remembering the outgoing one as stale."""
        session.generation += 1
        if session.token:
            session.token_history = (
                [session.token] + session.token_history)[:4]
        session.token = _mint_token(
            session.session_id, session.generation, session.document_id)
        session.updated_at = time.time()

    # ── contents ─────────────────────────────────────────────────────

    def contents_view(self, manifest: LongDocumentManifest, *,
                      current_segment_id: str = "",
                      update_available: bool = False) -> ContentsView:
        """The contents of a manifest: every segment in order, with
        declared-but-unheld chapters (the completeness missing
        ranges) synthesized as absent rows — shown as absent, never
        invented, never silently dropped from the listing."""
        rows: list[ContentsRow] = []
        seen_numbers: set[int] = set()
        for seg in manifest.segments:
            rows.append(ContentsRow(
                ordinal=len(rows) + 1, segment_id=seg.segment_id,
                kind=seg.kind, title=seg.title, present=seg.present,
                char_count=seg.char_count,
                current=seg.segment_id == current_segment_id,
                state=_segment_state(seg)))
            num = _chapter_number(seg.segment_id)
            if num is not None:
                seen_numbers.add(num)
        comp = manifest.completeness
        for start, end in comp.missing_ranges or []:
            for n in range(int(start), int(end) + 1):
                if n in seen_numbers:
                    continue
                rows.append(ContentsRow(
                    ordinal=0, segment_id=f"ch-{n}", kind="chapter",
                    title="", present=False, char_count=0,
                    current=f"ch-{n}" == current_segment_id))
        rows.sort(key=lambda r: (_chapter_number(r.segment_id)
                                 if _chapter_number(r.segment_id)
                                 is not None else 1 << 30,
                                 r.ordinal))
        for i, row in enumerate(rows):
            row.ordinal = i + 1
        return ContentsView(
            document_id=manifest.document_id, title=manifest.title,
            rows=rows, completeness_status=comp.status,
            declared_total=comp.declared_total,
            present_count=comp.present_count, note=comp.note,
            revision_ref=manifest.revision_ref,
            update_available=update_available)

    # ── paging ───────────────────────────────────────────────────────

    @staticmethod
    def _pages(content: str, page_size: int) -> tuple[list[str], list[int]]:
        pages = chunk_message(content or "", max_text=page_size) or [""]
        starts: list[int] = []
        at = 0
        for page in pages:
            starts.append(at)
            at += len(page)
        return pages, starts

    def _segment_or_none(self, session: ReaderSession,
                         segment_id: str) -> ManifestSegment | None:
        return session.pinned().segment(segment_id)

    @staticmethod
    def _present_neighbours(manifest: LongDocumentManifest,
                            segment_id: str,
                            ) -> tuple[ManifestSegment | None,
                                       ManifestSegment | None]:
        segs = manifest.segments
        idx = next((i for i, s in enumerate(segs)
                    if s.segment_id == segment_id), -1)
        if idx < 0:
            return None, None
        prev_seg = next((segs[i] for i in range(idx - 1, -1, -1)
                         if segs[i].present), None)
        next_seg = next((segs[i] for i in range(idx + 1, len(segs))
                         if segs[i].present), None)
        return prev_seg, next_seg

    def view(self, session_id: str, resolver: Resolver) -> PageView:
        """The session's current page, resolved live.  A segment the
        resolver cannot serve renders *unavailable* with the reason
        in the note — the cursor stays where the reader left it."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ReaderFault(f"no reader session {session_id!r}")
            manifest = session.pinned()
            base = dict(
                session_id=session.session_id,
                document_id=session.document_id,
                title=manifest.title, revision_ref=manifest.revision_ref,
                segment_count=len(manifest.segments),
                declared_total=int(
                    manifest.completeness.declared_total or 0),
                update_available=session.update_available,
                token=session.token, generation=session.generation)
            if not manifest.segments:
                return PageView(
                    status="empty", **base,
                    note="the document holds no content yet")
            seg = manifest.segment(session.segment_id)
            if seg is None:
                return PageView(
                    status="unavailable", segment_id=session.segment_id,
                    note=("this part of the document is not in the "
                          "revision you opened — it may have been "
                          "removed"), **base)
            ordinal = next(
                (i + 1 for i, s in enumerate(manifest.segments)
                 if s.segment_id == seg.segment_id), 0)
            prev_seg, next_seg = self._present_neighbours(
                manifest, seg.segment_id)
            common = dict(
                segment_id=seg.segment_id, segment_title=seg.title,
                segment_kind=seg.kind,
                segment_state=_segment_state(seg),
                segment_ordinal=ordinal,
                has_prev_segment=prev_seg is not None,
                has_next_segment=next_seg is not None,
                next_segment_title=(next_seg.title if next_seg else ""),
                next_segment_kind=(next_seg.kind if next_seg else ""))
            if not seg.present:
                return PageView(
                    status="unavailable",
                    note=(f"{_kind_word(seg.kind)} "
                          f"{_chapter_number(seg.segment_id) or ordinal} "
                          "is declared by the source but not held — "
                          "nothing is invented in its place"),
                    **base, **common)
            content = resolver(seg)
            if content is None:
                return PageView(
                    status="unavailable",
                    note=("this part of the document could not be "
                          "served — its source no longer holds it"),
                    **base, **common)
            pages, starts = self._pages(content, session.page_size)
            page_index = 0
            for i, start in enumerate(starts):
                if start <= session.offset:
                    page_index = i
            session.offset = starts[page_index]
            return PageView(
                status="ok", body=pages[page_index],
                page_index=page_index, page_count=len(pages),
                offset=session.offset, segment_chars=len(content),
                has_prev_page=page_index > 0,
                has_next_page=page_index < len(pages) - 1,
                **base, **common)

    def goto(self, session_id: str, resolver: Resolver, *,
             segment_id: str = "", page_index: int = -1) -> PageView:
        """Explicit repositioning (a read action): jump to a segment
        and/or a page within the current segment.  Records the move
        as a navigation (generation bump + token rotation)."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ReaderFault(f"no reader session {session_id!r}")
            moved = False
            if segment_id and segment_id != session.segment_id:
                session.segment_id = segment_id
                session.offset = 0
                moved = True
            elif segment_id and page_index < 0 and session.offset != 0:
                # an explicit jump to the segment already open
                # re-reads it from its first page
                session.offset = 0
                moved = True
            if page_index >= 0:
                seg = session.pinned().segment(session.segment_id)
                content = resolver(seg) if seg is not None else None
                if content:
                    pages, starts = self._pages(content, session.page_size)
                    page_index = min(page_index, len(pages) - 1)
                    if starts[page_index] != session.offset:
                        session.offset = starts[page_index]
                        moved = True
            if moved:
                self._rotate(session)
                self._save()
            return self.view(session_id, resolver)

    def navigate(self, session_id: str, resolver: Resolver,
                 verb: str) -> PageView:
        """Move the cursor one step: ``next`` / ``prev`` cross
        segment boundaries at segment ends (forward lands on the
        next present segment's first page; backward on the previous
        present segment's last page).  A step that cannot move
        returns the current view with an honest note and does not
        rotate the token."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ReaderFault(f"no reader session {session_id!r}")
            view = self.view(session_id, resolver)
            manifest = session.pinned()
            if verb == "next":
                if view.has_next_page:
                    seg = manifest.segment(session.segment_id)
                    content = resolver(seg) if seg else None
                    pages, starts = self._pages(
                        content or "", session.page_size)
                    session.offset = starts[view.page_index + 1]
                elif view.has_next_segment:
                    _prev, next_seg = self._present_neighbours(
                        manifest, session.segment_id)
                    session.segment_id = next_seg.segment_id
                    session.offset = 0
                else:
                    view.note = "end of the document — nothing after this"
                    return view
            elif verb == "prev":
                if view.has_prev_page:
                    seg = manifest.segment(session.segment_id)
                    content = resolver(seg) if seg else None
                    pages, starts = self._pages(
                        content or "", session.page_size)
                    session.offset = starts[view.page_index - 1]
                elif view.has_prev_segment:
                    prev_seg, _next = self._present_neighbours(
                        manifest, session.segment_id)
                    content = resolver(prev_seg)
                    session.segment_id = prev_seg.segment_id
                    if content:
                        pages, starts = self._pages(
                            content, session.page_size)
                        session.offset = starts[-1]
                    else:
                        session.offset = 0
                else:
                    view.note = "start of the document — nothing before this"
                    return view
            else:
                raise ReaderFault(f"unknown navigation verb {verb!r}")
            self._rotate(session)
            self._save()
            return self.view(session_id, resolver)

    @staticmethod
    def _nearest_surviving(fresh: LongDocumentManifest,
                           old_seg: ManifestSegment | None,
                           ) -> ManifestSegment | None:
        """The surviving segment nearest the place a vanished
        segment occupied: the first present segment at or after its
        old ordinal (the one that followed it), else the last
        present one before it.  None when the newer revision holds
        nothing at all."""
        present = [s for s in fresh.segments if s.present]
        if not present:
            return None
        if old_seg is None:
            return present[0]
        after = [s for s in present if s.ordinal >= old_seg.ordinal]
        return after[0] if after else present[-1]

    def switch_revision(self, session_id: str,
                        resolver: Resolver) -> PageView:
        """Adopt the flagged newer revision, re-anchoring by
        segment id where the segment survives (offset clamped into
        the new content).  A vanished segment re-anchors to the
        NEAREST surviving segment — the one that followed it, else
        the one before — and the returned view says where the
        reader landed and why; a surviving segment's switch is
        noted too.  A place is never silently relocated.  Without
        a flagged revision this is a no-op returning the current
        view."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ReaderFault(f"no reader session {session_id!r}")
            if not session.latest:
                return self.view(session_id, resolver)
            old_seg = session.pinned().segment(session.segment_id)
            fresh = LongDocumentManifest.from_dict(session.latest)
            session.manifest = session.latest
            session.latest = None
            session.update_available = False
            note = ""
            fresh_seg = fresh.segment(session.segment_id)
            if fresh_seg is None or not fresh_seg.present:
                old_label = (_segment_label(old_seg) if old_seg
                             else "the part you were reading")
                landed = self._nearest_surviving(fresh, old_seg)
                if landed is not None:
                    session.segment_id = landed.segment_id
                    note = (
                        f"{old_label} is not in the newer revision "
                        f"— you're now at {_segment_label(landed)} "
                        "instead, the nearest surviving place")
                else:
                    session.segment_id = self._landing_segment(
                        fresh, "")
                    note = (
                        f"{old_label} is not in the newer revision, "
                        "and no part of the newer revision is held "
                        "yet")
                session.offset = 0
            else:
                note = ("switched to the newer revision — same "
                        "place, updated text")
            self._rotate(session)
            self._save()
            view = self.view(session_id, resolver)
            if note:
                view.note = (note if not view.note
                             else f"{note} ({view.note})")
            return view

    # ── callback discipline ──────────────────────────────────────────

    def session_by_token(self, token: str) -> ReaderSession | None:
        """The session a token belongs to — current or recently
        rotated out (so a stale tap can be told apart from an
        unknown one)."""
        with self._lock:
            for session in self._sessions.values():
                if token and (token == session.token
                              or token in session.token_history):
                    return session
            return None

    def act(self, token: str, verb: str, resolver: Resolver) -> ActResult:
        """Actuate one callback token: ``n`` next, ``p`` prev,
        ``c`` contents, ``j<row>`` jump to a contents row, ``u``
        switch to the newer revision.  A token that is not the
        session's current one is *stale*: an honest no-op — the
        cursor does not move.  An unrecognized token is *unknown*
        (expired or foreign)."""
        with self._lock:
            session = self.session_by_token(token)
            if session is None:
                return ActResult(
                    status="unknown",
                    note=("that reading control has expired — open "
                          "the document again to pick up where you "
                          "left off"))
            if token != session.token:
                return ActResult(
                    status="stale",
                    note=("that button is from an older page — your "
                          "place has not moved"))
            verb = (verb or "").strip().lower()
            if verb == "c":
                return ActResult(
                    status="ok",
                    contents=self.contents_view(
                        session.pinned(),
                        current_segment_id=session.segment_id,
                        update_available=session.update_available))
            if verb == "u":
                return ActResult(
                    status="ok",
                    view=self.switch_revision(session.session_id,
                                              resolver))
            if verb.startswith("j"):
                view = self.contents_view(
                    session.pinned(),
                    current_segment_id=session.segment_id)
                try:
                    row = view.rows[int(verb[1:]) - 1]
                except (ValueError, IndexError):
                    return ActResult(
                        status="ok",
                        view=self.view(session.session_id, resolver),
                        note="that contents entry does not exist")
                return ActResult(
                    status="ok",
                    view=self.goto(session.session_id, resolver,
                                   segment_id=row.segment_id))
            if verb in ("n", "next"):
                return ActResult(
                    status="ok",
                    view=self.navigate(session.session_id, resolver,
                                       "next"))
            if verb in ("p", "prev"):
                return ActResult(
                    status="ok",
                    view=self.navigate(session.session_id, resolver,
                                       "prev"))
            return ActResult(
                status="ok",
                view=self.view(session.session_id, resolver),
                note=f"unknown reading control {verb!r}")

    # ── plain-text renders ───────────────────────────────────────────

    def render_text(self, view: PageView, *,
                    commands: dict[str, str] | None = None) -> str:
        """The plain-transport render of a page: header, body, and
        footer pointers whose labels say what they point at — the
        next part of THIS segment reads differently from the next
        segment.  ``commands`` overrides the fallback command per
        direction (next/prev/contents/update); the default is the
        session's token payloads."""
        prefix = "read go"
        cmds = {
            "next": f"{prefix} {view.token} n",
            "prev": f"{prefix} {view.token} p",
            "contents": f"{prefix} {view.token} c",
            "update": f"{prefix} {view.token} u",
        }
        if commands:
            cmds.update({k: v for k, v in commands.items() if v})
        word = _kind_word(view.segment_kind)
        head = f"📖 {view.title}" if view.title else "📖"
        if view.segment_title and view.segment_title != view.title:
            head += f" — {view.segment_title}"
        lines = [head, ""]
        if view.status != "ok":
            lines.append(f"⚠️ {view.note}" if view.note
                         else "⚠️ this part is unavailable")
        else:
            lines.append(view.body)
            lines.append("")
            lines.append(
                f"— {word} {view.segment_ordinal} · page "
                f"{view.page_index + 1} of {view.page_count} —")
            if view.segment_state:
                lines.append(f"✎ {view.segment_state}")
        if view.has_next_page:
            lines.append(
                f"▸ next part of this {word}: {cmds['next']}")
        elif view.has_next_segment:
            nword = _kind_word(view.next_segment_kind)
            label = view.next_segment_title or nword
            lines.append(
                f"▸ end of this {word} — next {nword}: {label}: "
                f"{cmds['next']}")
        elif view.status == "ok":
            if view.declared_total and \
                    view.segment_count < view.declared_total:
                lines.append(
                    "▸ end of what is held — the source declares "
                    "more than this copy holds")
            else:
                lines.append("▸ end of the document")
        if view.has_prev_page or view.has_prev_segment:
            lines.append(f"◂ back: {cmds['prev']}")
        if view.status == "ok" or view.segment_id:
            lines.append(f"📑 contents: {cmds['contents']}")
        if view.update_available:
            lines.append(
                "✦ a newer revision of this document is available "
                f"— switch: {cmds['update']} (your place is kept)")
        if view.note and view.status == "ok":
            lines.append(view.note)
        return "\n".join(lines)

    def render_contents_text(self, view: ContentsView, *,
                             open_hint: str = "") -> str:
        """The plain-transport contents render: held segments with
        their sizes, absent ones named as absent, the completeness
        truth up top — never a listing that poses as whole."""
        if view.declared_total is not None:
            held = (f"{view.present_count} of {view.declared_total} "
                    "held")
        else:
            held = f"{view.present_count} held · total unknown"
        head = f"📑 {view.title} — contents ({held})"
        lines = [head]
        for row in view.rows:
            mark = "▸ " if row.current else "  "
            if row.present:
                line = (f"{mark}{row.ordinal}. "
                        f"{row.title or row.segment_id}"
                        f" — {row.char_count} chars")
                if row.state:
                    line += f" · {row.state}"
                lines.append(line)
            else:
                num = _chapter_number(row.segment_id)
                label = row.title or (
                    f"{_kind_word(row.kind)} {num}"
                    if num is not None else row.segment_id)
                line = f"{mark}{row.ordinal}. {label} — not held"
                if row.state:
                    line += f" · {row.state}"
                lines.append(line)
        if view.note:
            lines.append(view.note)
        if open_hint:
            lines.append(open_hint)
        return "\n".join(lines)


def _chapter_number(segment_id: str) -> int | None:
    """The chapter number of a ``ch-<n>`` segment id, else None."""
    if segment_id.startswith("ch-"):
        tail = segment_id[3:]
        if tail.isdigit():
            return int(tail)
    return None
