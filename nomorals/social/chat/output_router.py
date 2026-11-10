"""The output router facade (Doc 16, Pass 4).

``LongOutput`` (progress_throttle) routes exactly two ways — chunks
or one document — and only two shipped paths consume it (the
Telegram adapter's ``send`` and the notifier's Doc-overflow rung).
The runtime's dominant long-output path, ``_send_long_checked``,
never touched it: every long report, research result and tool
output arrived as a bare chunk dump with no reader, no edition and
no actions.  This facade is the §3 router that closes that gap.  It
sits BESIDE ``LongOutput`` rather than inside it on purpose:
``route()``'s ``("chunks", …)`` / ``("document", bytes)`` tuple is a
load-bearing legacy contract (the Telegram adapter indexes it
positionally; the notifier overflow rung branches on it), so the
legacy adapter stays byte-identical and the richer decision lives
here, where it can grow reader/edition routes without breaking a
single existing consumer.

The decision, given content + transport + explicit caller intent:

* **explicit inline intent** → ``chunks``.  A caller/user who asked
  for inline text gets inline text, chunked losslessly — never a
  forced file (the handoff's own rule, and the owner's doctrine).
* **fits one message** → ``chunks`` (the native send, unchanged).
* **declared exact content** (``kind="code"`` / ``"data"``) past one
  message → ``document``: the raw original, byte for byte, as a
  downloadable file — never re-rendered, never watermarked.
* **medium** (≤ :data:`READER_MAX_CHUNKS` chunks at the send limit)
  → ``reader``: the full text lands in a :class:`ReaderService`
  session; the chat gets one introduction carrying the first page
  and working controls.
* **long** (> :data:`READER_MAX_CHUNKS` chunks) → ``reader+edition``:
  the reader session plus two file actions — the branded HTML
  edition (:mod:`nomorals.core.edition` over the text manifest)
  and the raw original — delivered on request through ``/read get``.
* a transport that cannot hold reader pages (per-segment-billed /
  tiny message cap, e.g. SMS) degrades to ``chunks`` — lossless,
  never a dead action, never a fake button.

Every plan is journaled (``<workspace>/longdoc/routes.jsonl``:
route kind + reason + size + title + reader), so "why did this
arrive as a file / a reader / chunks?" is answerable after the
fact.  Routed texts persist content-addressed under
``<workspace>/longdoc/content/`` — that store is what lets a reader
session (which pins only the manifest) resolve its content after a
restart, and what the HTML edition and the raw download render
from.  The router performs no network I/O: it returns a
:class:`RoutePlan` and the caller (the runtime seam) executes it
through the gateway.

Layer: social (L4), beside progress_throttle and reader_service —
imports core (L1) and its own package only.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...core.edition import render_edition_from_map
from ...core.longdoc import LongDocumentManifest
from .chunking import chunk_message
from .manifest import manifest_from_text
from .platform_limits import for_platform
from .progress_throttle import DEFAULT_MAX_CHUNKS, LongOutput
from .reader_service import ReaderService

__all__ = [
    "OutputRouter", "RouteDecision", "RoutePlan", "TransportCaps",
    "transport_caps", "SEND_CHUNK_CHARS", "READER_MAX_CHUNKS",
    "ROUTE_CHUNKS", "ROUTE_READER", "ROUTE_READER_EDITION",
    "ROUTE_DOCUMENT",
]

#: The chunk size the runtime's ``_send_long`` sends at (its default
#: ``limit``) — the unit route sizes are counted in at that seam.
SEND_CHUNK_CHARS = 3800

#: Reader-route ceiling, in chunks at the send limit.  Source:
#: ``DEFAULT_MAX_CHUNKS`` in progress_throttle — LongOutput's own
#: discipline for "past this many chunks, a wall of messages is the
#: wrong shape"; here it separates the reader-only medium band from
#: the long band that also earns file actions.
READER_MAX_CHUNKS = DEFAULT_MAX_CHUNKS

#: Below this per-message cap a transport cannot hold a reader page
#: (a page is ~3000 chars); SMS (160) is the shipped case.  Such
#: transports degrade to lossless chunks.
MIN_READER_MESSAGE_CHARS = 1000

ROUTE_CHUNKS = "chunks"
ROUTE_READER = "reader"
ROUTE_READER_EDITION = "reader+edition"
ROUTE_DOCUMENT = "document"

#: Reader-session command prefix for routed outputs — the payloads
#: re-enter the runtime's ``/read`` handler, exactly as story
#: sessions re-enter ``/books go``.
READ_PREFIX = "read go"


@dataclass(frozen=True)
class TransportCaps:
    """What a transport can honestly hold, for routing."""

    buttons: bool  # pager buttons (see transport_caps for the rule)
    files: bool    # file sends are meaningful on this transport
    reader: bool   # a reader page fits this transport's message cap
    max_text: int


def transport_caps(platform: object) -> TransportCaps:
    """The routing capabilities of ``platform`` from the shipped
    limits table.  Buttons follow the reading pager's own gate
    (``_reading_pager_ok`` in the books wiring): real buttons AND
    text edit, and never WhatsApp — there the owner's controls are
    the plain-text commands, as on the /books surface.  Files are
    meaningless only where delivery is per-segment billed (SMS).
    Never raises (``for_platform`` never raises)."""
    limits = for_platform(platform)
    name = (limits.platform or "").strip().lower()
    buttons = bool(
        limits.supports_buttons and limits.supports_text_edit
        and name not in ("whatsapp", "wa"))
    files = limits.cost_model != "per_segment"
    reader = limits.max_text >= MIN_READER_MESSAGE_CHARS
    return TransportCaps(buttons=buttons, files=files, reader=reader,
                         max_text=limits.max_text)


@dataclass
class RouteDecision:
    """One routing decision: the route taken and why, in words."""

    kind: str
    reason: str
    chars: int
    chunks: int
    title: str = ""
    document_id: str = ""


@dataclass
class RoutePlan:
    """A decision plus everything the caller needs to execute it.

    ``chunks`` plans carry nothing extra — the caller runs its
    legacy chunked send.  ``document`` plans carry the raw file.
    Reader plans carry the one introduction message (first page +
    controls) and, on button transports, its button rows."""

    decision: RouteDecision
    intro_text: str = ""
    buttons: list[list[tuple[str, str]]] | None = None
    token: str = ""
    manifest: LongDocumentManifest | None = None
    file_name: str = ""
    file_bytes: bytes = b""
    caption: str = ""


def _safe_stem(title: str) -> str:
    stem = "".join(
        c if c.isalnum() or c in ("-", "_") else "_"
        for c in (title or "").strip())[:60].strip("_")
    return stem or "output"


class OutputRouter:
    """Routing decisions + the routed-content store + reader
    sessions for long outputs.  Construct with an explicit root or
    via :meth:`for_context`; ``root=None`` keeps everything in
    memory (tests, ephemeral callers)."""

    def __init__(self, root: Path | None = None, *,
                 reader: ReaderService | None = None) -> None:
        self.root = Path(root) if root else None
        self.reader = reader or ReaderService(
            self.root / "reader_sessions.json" if self.root else None)
        # Same-process content/journal for root=None routers; with a
        # real root the disk is the truth and these stay small.
        self._memory: dict[str, str] = {}
        self._memory_journal: list[dict] = []

    @classmethod
    def for_context(cls, context: Any) -> OutputRouter:
        """The router bound to a runtime context: state persists
        under ``<workspace>/longdoc/`` (the workspace derivation
        mirrors :meth:`ReaderService.for_context`: the context
        settings' workspace_dir, else ``./workspace``)."""
        settings = getattr(context, "settings", None)
        base = getattr(settings, "workspace_dir", None) if settings else None
        workspace = Path(base) if base else Path.cwd() / "workspace"
        root = workspace / "longdoc"
        return cls(root, reader=ReaderService.for_context(context))

    # ── routed-content store ─────────────────────────────────────

    def _content_path(self, fingerprint: str) -> Path | None:
        if self.root is None:
            return None
        return self.root / "content" / f"{fingerprint}.txt"

    @staticmethod
    def _fingerprint(document_id: str) -> str:
        return document_id.split(":", 1)[1] if ":" in document_id else ""

    def store_text(self, text: str) -> LongDocumentManifest:
        """Persist ``text`` content-addressed and return its
        manifest.  Idempotent: identical text is the same document
        and is written once."""
        manifest = manifest_from_text(text)
        path = self._content_path(self._fingerprint(manifest.document_id))
        if path is not None:
            try:
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    # Bytes, never text mode: the stored file IS the
                    # raw download, and text mode would rewrite its
                    # newlines on Windows hosts.
                    path.write_bytes(text.encode("utf-8"))
            except OSError:
                pass  # the in-memory map below still serves this process
        self._memory[manifest.document_id] = text
        return manifest

    def text_for(self, document_id: str) -> str | None:
        """The stored original for a routed document, or None."""
        if document_id in self._memory:
            return self._memory[document_id]
        path = self._content_path(self._fingerprint(document_id))
        if path is not None and path.exists():
            try:
                return path.read_bytes().decode("utf-8")
            except (OSError, UnicodeDecodeError):
                return None
        return None

    def resolver(self):
        """A ReaderService resolver over the routed-content store:
        a ``text:<fp>`` segment resolves to the stored original;
        anything else is not this store's to serve (None — the
        reader renders it unavailable, honestly)."""
        def resolve(segment) -> str | None:
            if not segment.present:
                return None
            ref = segment.content_ref or ""
            if ref.startswith("text:"):
                return self.text_for(f"text:{ref[len('text:'):]}")
            return None
        return resolve

    # ── editions + files ─────────────────────────────────────────

    def edition_html(self, document_id: str) -> str | None:
        """The branded HTML edition of a routed document, rendered
        from its stored original through the P3 edition renderer."""
        text = self.text_for(document_id)
        if text is None:
            return None
        manifest = manifest_from_text(text)
        contents = {seg.content_ref: text for seg in manifest.segments}
        return render_edition_from_map(manifest, contents)

    def file_for(self, document_id: str, fmt: str) -> tuple[Path, str] | None:
        """A sendable file for a routed document: ``(path, name)``.

        ``txt`` serves the stored original itself (byte for byte);
        ``html`` renders the edition once and caches it beside the
        content, content-addressed like everything else here."""
        text = self.text_for(document_id)
        if text is None or self.root is None:
            return None
        manifest = manifest_from_text(text)
        stem = _safe_stem(manifest.title)
        if fmt == "txt":
            return (self._content_path(self._fingerprint(document_id)),
                    LongOutput().document_name(stem))
        if fmt == "html":
            path = self.root / "editions" / (
                f"{self._fingerprint(document_id)}.html")
            try:
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(
                        (self.edition_html(document_id) or ""
                         ).encode("utf-8"))
            except OSError:
                return None
            return path, f"{stem}.html"
        return None

    # ── the decision journal ─────────────────────────────────────

    def _record(self, decision: RouteDecision, *, platform: str,
                reader_key: str) -> None:
        if self.root is None:
            return
        row = {
            "at": time.time(), "route": decision.kind,
            "reason": decision.reason, "chars": decision.chars,
            "chunks": decision.chunks, "title": decision.title,
            "document_id": decision.document_id,
            "platform": platform, "reader_key": reader_key,
        }
        try:
            path = self.root / "routes.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            pass  # journaling never breaks a delivery

    def recent(self, reader_key: str, limit: int = 10) -> list[dict]:
        """This reader's routed documents, newest first — journal
        rows for reader/reader+edition/document routes (the ones
        with a document to reopen)."""
        if self.root is None:
            return [dict(r) for r in self._memory_journal
                    if r.get("reader_key") == reader_key][-limit:][::-1]
        path = self.root / "routes.jsonl"
        if not path.exists():
            return []
        rows: list[dict] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("reader_key") == reader_key and \
                    row.get("route") != ROUTE_CHUNKS:
                rows.append(row)
        return rows[-limit:][::-1]

    # ── the routing decision ─────────────────────────────────────

    def plan(self, text: str, *, platform: object = "",
             reader_key: str = "", limit: int = SEND_CHUNK_CHARS,
             intent: str = "", kind: str = "",
             title: str = "") -> RoutePlan:
        """Decide how ``text`` reaches ``platform``'s chat, and
        prepare everything the caller needs to execute the route.

        ``intent="inline"`` is the explicit ask for inline text and
        always wins.  ``kind="code"``/``"data"`` declares exact
        content whose original must be preserved byte for byte.
        ``limit`` is the caller's chunk size (the unit chunk counts
        are measured in).  Reader routes open/resume the reader's
        session on the document and build the introduction."""
        text = text or ""
        caps = transport_caps(platform)
        chunks = len(chunk_message(text, max_text=max(200, int(limit)))
                     or []) if text else 0
        manifest = manifest_from_text(text) if text else None
        doc_title = (title or (manifest.title if manifest else "")
                     or "").strip()
        decision = RouteDecision(
            kind=ROUTE_CHUNKS, reason="", chars=len(text),
            chunks=chunks, title=doc_title,
            document_id=manifest.document_id if manifest else "")

        if intent == "inline":
            decision.reason = ("explicit inline intent — chunks, "
                               "losslessly; no file is forced")
        elif chunks <= 1:
            decision.reason = "fits one message — native send"
        elif kind in ("code", "data") and caps.files:
            decision.kind = ROUTE_DOCUMENT
            decision.reason = (
                f"exact content declared ({kind}) — the raw original "
                "as a document, byte for byte, never re-rendered")
        elif kind in ("code", "data"):
            decision.reason = (
                f"exact content declared ({kind}) but this transport "
                "cannot carry files — lossless chunks instead")
        elif not caps.reader:
            decision.reason = (
                "this transport's messages cannot hold a reader "
                "page — lossless chunks")
        elif not reader_key:
            decision.reason = ("no reader identity for a reading "
                               "session — lossless chunks")
        elif chunks <= READER_MAX_CHUNKS:
            decision.kind = ROUTE_READER
            decision.reason = (
                f"medium output ({chunks} chunks) — the chat reader: "
                "one introduction, the full text in a reading session")
        else:
            decision.kind = ROUTE_READER_EDITION
            decision.reason = (
                f"long output ({chunks} chunks) — the chat reader "
                "plus the branded HTML edition and the raw original "
                "as file actions")

        plan = RoutePlan(decision=decision)
        if decision.kind == ROUTE_DOCUMENT:
            self.store_text(text)
            plan.manifest = manifest
            plan.file_bytes = text.encode("utf-8")
            plan.file_name = LongOutput().document_name(
                _safe_stem(doc_title))
            plan.caption = (
                f"💾 {doc_title or 'exact output'} — the original, "
                f"byte for byte ({len(text):,} chars)")
        elif decision.kind in (ROUTE_READER, ROUTE_READER_EDITION):
            stored = self.store_text(text)
            plan.manifest = stored
            session = self.reader.open(
                reader_key, stored, command_prefix=READ_PREFIX)
            view = self.reader.view(session.session_id, self.resolver())
            plan.token = session.token
            head = (
                f"📄 {doc_title or 'long output'} — {len(text):,} "
                "chars · reading copy below, the full text preserved "
                "— nothing is cut")
            plan.intro_text = head + "\n\n" + self.reader.render_text(view)
            if decision.kind == ROUTE_READER_EDITION:
                plan.intro_text += (
                    f"\n🌐 HTML edition: /read get {session.token} html"
                    f"\n⬇ original text: /read get {session.token} txt")
            if caps.buttons:
                rows = view.buttons(READ_PREFIX)
                if decision.kind == ROUTE_READER_EDITION:
                    rows.append([
                        ("🌐 Open HTML",
                         f"read get {session.token} html"),
                        ("⬇ Download",
                         f"read get {session.token} txt"),
                    ])
                plan.buttons = rows
        self._record(decision, platform=str(
            getattr(for_platform(platform), "platform", "") or platform),
            reader_key=reader_key)
        if self.root is None:
            self._memory_journal.append({
                "route": decision.kind, "reason": decision.reason,
                "chars": decision.chars, "chunks": decision.chunks,
                "title": decision.title,
                "document_id": decision.document_id,
                "reader_key": reader_key})
        return plan
