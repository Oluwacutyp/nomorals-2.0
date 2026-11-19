"""Edit session — incremental-write editing over the object model.

Port of the *design* of storytold/pdfcraft ``crates/edit`` (``apply.rs``,
``objects.rs``, ``marks.rs``) and ``crates/cos`` ``writer.rs``
(``write_incremental`` / ``write_full``): edits accumulate as changed
and new objects; an incremental save appends only those objects plus
a fresh xref subsection and a trailer with ``/Prev`` — the original
bytes survive untouched as the file's prefix.  A full save rewrites
every object with a fresh xref table and is used after a repair, when
the source's xref cannot be trusted.  Devon's own code.

Dirty tracking is exact: only objects the session touched are written.
An incremental save with nothing changed returns the original bytes.
"""

from __future__ import annotations

from typing import Any

from ...core.pdf import escape_pdf_string, winansi_text
from ..errors import DocumentError
from .cos import (
    PdfName,
    PdfRef,
    PdfStream,
    PdfString,
    build_full,
    build_incremental,
    resolve,
)
from .model import PdfDocument

__all__ = ["PdfEditor"]

_INFO_KEYS = {
    "title": "/Title",
    "author": "/Author",
    "subject": "/Subject",
    "keywords": "/Keywords",
    "creator": "/Creator",
    "producer": "/Producer",
}


def _fail(message: str, *, kind: str = "invalid_argument") -> DocumentError:
    return DocumentError(message, kind=kind)


class PdfEditor:
    """An open PDF with pending edits.

    Open with :meth:`from_bytes` / :meth:`from_path`, apply ops
    (:meth:`set_info`, :meth:`rotate_page`, :meth:`add_text_annotation`,
    :meth:`stamp_text`), then :meth:`save_incremental` or
    :meth:`save_full`.  After a save the session re-opens the saved
    bytes, so a second save never double-appends the same edit.
    """

    def __init__(self, data: bytes) -> None:
        self._original = bytes(data)
        self.document = PdfDocument.from_bytes(self._original)
        self._objects: dict[PdfRef, Any] = {}
        self._trailer: dict[str, Any] = {}
        self._next_num = (
            max(self.document._xref.entries) + 1
            if self.document._xref.entries else 1)

    @classmethod
    def from_bytes(cls, data: bytes) -> PdfEditor:
        return cls(bytes(data))

    @classmethod
    def from_path(cls, path: str) -> PdfEditor:
        return cls(PdfDocument.from_path(path).data)

    # ── state ───────────────────────────────────────────────────────────
    @property
    def dirty(self) -> list[PdfRef]:
        """Changed/new object references, sorted by number."""
        return sorted(self._objects, key=lambda r: (r.num, r.gen))

    def summary(self) -> dict[str, Any]:
        """What this session holds: changed objects, trailer changes,
        and the save methods available."""
        return {"changed_objects": len(self._objects),
                "changed_refs": [r.num for r in self.dirty],
                "trailer_updates": sorted(self._trailer),
                "methods": ["save_incremental", "save_full"],
                "repaired_source": self.document.repaired}

    # ── internals ───────────────────────────────────────────────────────
    def _alloc(self) -> PdfRef:
        ref = PdfRef(self._next_num, 0)
        self._next_num += 1
        return ref

    def _put(self, ref: PdfRef, obj: Any) -> None:
        self._objects[ref] = obj

    def _current(self, ref: PdfRef) -> Any:
        """The live value of an object: the pending edit wins, else the
        value resolved from the file."""
        if ref in self._objects:
            return self._objects[ref]
        return resolve(self.document.data, self.document._xref, ref)

    def _changed_dict(self, ref: PdfRef) -> tuple[PdfRef, dict[str, Any]]:
        """A mutable copy of the dict at ``ref``, registered as changed.
        The original dict is never mutated in place — an incremental
        save must only ever append."""
        current = self._current(ref)
        if not isinstance(current, dict):
            raise _fail(f"object {ref.num} is not a dictionary")
        copy = dict(current)
        self._put(ref, copy)
        return ref, copy

    def _page(self, page_no: int) -> Any:
        if not 1 <= page_no <= self.document.page_count:
            raise _fail(f"page {page_no} out of range (document has "
                        f"{self.document.page_count} pages)")
        return self.document.pages[page_no - 1]

    def _after_save(self, saved: bytes) -> None:
        self._original = saved
        self.document = PdfDocument.from_bytes(saved)
        self._objects = {}
        self._trailer = {}
        self._next_num = (
            max(self.document._xref.entries) + 1
            if self.document._xref.entries else 1)

    # ── ops ─────────────────────────────────────────────────────────────
    def set_info(self, **fields: str) -> None:
        """Set Info-dictionary metadata: title, author, subject,
        keywords, creator, producer.  Unknown field names are a caller
        error — a typo'd key must not silently vanish."""
        unknown = sorted(set(fields) - set(_INFO_KEYS))
        if unknown:
            raise _fail(f"unknown info fields: {', '.join(unknown)}; "
                        f"use {', '.join(sorted(_INFO_KEYS))}")
        trailer = self.document._xref.trailer
        info_ref = trailer.get("/Info")
        if isinstance(info_ref, PdfRef):
            _ref, info = self._changed_dict(info_ref)
        else:
            info_ref = self._alloc()
            info = {}
            self._put(info_ref, info)
            self._trailer["/Info"] = info_ref
        for field, value in fields.items():
            info[_INFO_KEYS[field]] = PdfString(
                winansi_text(str(value)).encode("latin-1", "replace"))

    def rotate_page(self, page_no: int, degrees: int) -> None:
        """Set a page's /Rotate (accumulates with any existing rotation).
        Only quarter turns are real in PDF — anything else is a caller
        error, not a silent rounding."""
        if degrees % 90 != 0:
            raise _fail(f"rotate needs a multiple of 90 degrees, "
                        f"got {degrees}")
        page = self._page(page_no)
        _ref, body = self._changed_dict(page.ref)
        current = body.get("/Rotate")
        base = current if isinstance(current, int) else 0
        body["/Rotate"] = (base + degrees) % 360

    def add_text_annotation(self, page_no: int,
                            rect: tuple[float, float, float, float],
                            contents: str, *, title: str = "",
                            color: tuple[float, float, float] = (1, 1, 0),
                            open: bool = False) -> PdfRef:
        """Append a /Text annotation (a sticky note) to a page.  Returns
        the new annotation's reference.  ``rect`` is [x0, y0, x1, y1]
        in PDF points, origin bottom-left."""
        if len(rect) != 4:
            raise _fail("annotation rect needs four numbers: x0,y0,x1,y1")
        try:
            rect_list = [float(v) for v in rect]
        except (TypeError, ValueError):
            raise _fail("annotation rect needs four numbers") from None
        if rect_list[2] <= rect_list[0] or rect_list[3] <= rect_list[1]:
            raise _fail("annotation rect is empty or inverted")
        page = self._page(page_no)
        annot_ref = self._alloc()
        annot: dict[str, Any] = {
            "/Type": PdfName("Annot"),
            "/Subtype": PdfName("Text"),
            "/Rect": rect_list,
            "/Contents": PdfString(
                winansi_text(contents).encode("latin-1", "replace")),
            "/C": [float(c) for c in color],
            "/Open": bool(open),
        }
        if title:
            annot["/T"] = PdfString(
                winansi_text(title).encode("latin-1", "replace"))
        self._put(annot_ref, annot)
        _ref, body = self._changed_dict(page.ref)
        existing = body.get("/Annots")
        if isinstance(existing, list):
            body["/Annots"] = list(existing) + [annot_ref]
        else:
            body["/Annots"] = [annot_ref]
        return annot_ref

    def stamp_text(self, page_no: int, text: str, *,
                   position: str = "bottom-center",
                   font_size: int = 10) -> PdfRef:
        """Append a text stamp to a page as a NEW content stream: the
        page's /Contents becomes [old streams..., new stream], so the
        original content is never rewritten.  Returns the new stream's
        reference.  Positions: bottom/top/center × left/center/right.
        The stamp is real text (Helvetica, WinAnsi) — it extracts with
        the rest of the page."""
        page = self._page(page_no)
        x0, y0, x1, y1 = page.mediabox
        width = x1 - x0
        height = y1 - y0
        safe = winansi_text(text)
        if not safe.strip():
            raise _fail("stamp text is empty after WinAnsi encoding")
        # Helvetica advances ~0.5 em on average (core.pdf uses 0.52).
        text_width = len(safe) * font_size * 0.5
        vertical, _, horizontal = position.partition("-")
        if vertical == "bottom":
            y = y0 + 40
        elif vertical == "top":
            y = y1 - 40
        elif vertical == "center":
            y = y0 + height / 2
        else:
            raise _fail(f"unknown stamp position {position!r}; use "
                        f"bottom/top/center-left/center/right")
        if horizontal == "center":
            x = x0 + (width - text_width) / 2
        elif horizontal == "left":
            x = x0 + 40
        elif horizontal == "right":
            x = x1 - text_width - 40
        else:
            raise _fail(f"unknown stamp position {position!r}; use "
                        f"bottom/top/center-left/center/right")
        ops = (f"BT /F1 {font_size} Tf 0 g "
               f"1 0 0 1 {x:.1f} {y:.1f} Tm "
               f"({escape_pdf_string(safe)}) Tj ET").encode("latin-1")
        stream_ref = self._alloc()
        self._put(stream_ref, PdfStream({"/Length": len(ops)}, ops))
        _ref, body = self._changed_dict(page.ref)
        contents = body.get("/Contents")
        if isinstance(contents, PdfRef):
            body["/Contents"] = [contents, stream_ref]
        elif isinstance(contents, list):
            body["/Contents"] = list(contents) + [stream_ref]
        else:
            body["/Contents"] = [stream_ref]
        # The stamp draws with /F1 — guarantee the page's resources
        # name it.  Devon's own pages already do; a foreign page that
        # does not gets a real Helvetica font object allocated and
        # wired in (never a dangling reference).
        resources = body.get("/Resources")
        res: dict[str, Any] | None = None
        if isinstance(resources, PdfRef):
            _rref, res = self._changed_dict(resources)
        elif isinstance(resources, dict):
            res = resources
        if isinstance(res, dict):
            fonts = res.get("/Font")
            if isinstance(fonts, PdfRef):
                _fref, fdict = self._changed_dict(fonts)
                if "/F1" not in fdict:
                    fdict["/F1"] = self._new_helvetica()
            elif isinstance(fonts, dict) and "/F1" not in fonts:
                res["/Font"] = dict(fonts)
                res["/Font"]["/F1"] = self._new_helvetica()
        return stream_ref

    def _new_helvetica(self) -> PdfRef:
        """A fresh standard-14 Helvetica font object, for pages whose
        resources name no /F1."""
        ref = self._alloc()
        self._put(ref, {
            "/Type": PdfName("Font"),
            "/Subtype": PdfName("Type1"),
            "/BaseFont": PdfName("Helvetica"),
            "/Encoding": PdfName("WinAnsiEncoding"),
        })
        return ref

    # ── saving ──────────────────────────────────────────────────────────
    def save_incremental(self) -> bytes:
        """Append the pending edits to the original bytes (upstream
        ``write_incremental``).  The returned bytes start with the
        original file byte-for-byte; only new/changed objects, a fresh
        xref subsection and a trailer with /Prev are appended."""
        saved = build_incremental(
            self._original, dict(self._objects), dict(self._trailer),
            self.document._xref.startxref)
        self._after_save(saved)
        return saved

    def save_full(self) -> bytes:
        """Rewrite the whole file from the object graph (upstream
        ``write_full``): every object serialized fresh with a complete
        xref table.  The shape for a repaired source, where the
        original layout cannot be trusted."""
        data, xref = self.document.data, self.document._xref
        objects: dict[PdfRef, Any] = {}
        for num, entry in xref.entries.items():
            if entry.kind == "f":
                continue
            value = resolve(data, xref, PdfRef(num, entry.gen))
            if value is not None:
                objects[PdfRef(num, entry.gen)] = value
        objects.update(self._objects)
        trailer = dict(xref.trailer)
        trailer.update(self._trailer)
        root_ref = trailer.get("/Root")
        if not isinstance(root_ref, PdfRef):
            raise _fail("cannot full-save: trailer has no /Root",
                        kind="parse")
        info_ref = trailer.get("/Info")
        if not isinstance(info_ref, PdfRef):
            info_ref = None
        saved = build_full(objects, root_ref, info_ref)
        self._after_save(saved)
        return saved
