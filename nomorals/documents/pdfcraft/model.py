"""Document model — the parsed PDF as an object graph.

Port of the *design* of storytold/pdfcraft ``crates/cos/document``
(``reader.rs``, ``objects.rs``, ``state.rs``) and ``crates/model``:
open bytes, walk the catalog and page tree, and inventory what the
file holds — pages, fonts, images, annotations, metadata.  Devon's own
code; the shape (a ``Document`` holding resolved objects over an xref,
page-tree walk with inherited boxes/resources) is the adopted design.

This layer READS.  It never mutates bytes; edits live in :mod:`edit`.
Text extraction is not duplicated here — it reuses
:func:`nomorals.core.pdf.read_pdf_text` /
:func:`nomorals.core.pdf.read_pdf_page_texts`, the engine that already
knows FlateDecode, TJ/Tj operators and ToUnicode maps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ...core.pdf import read_pdf_page_texts, read_pdf_text
from ..errors import DocumentError
from .cos import PdfName, PdfRef, PdfStream, PdfString, parse_xref, resolve

__all__ = ["PdfPage", "PdfDocument"]


def _fail(message: str, *, kind: str = "parse") -> DocumentError:
    return DocumentError(message, kind=kind)


def _as_name(value: Any) -> str | None:
    if isinstance(value, PdfName):
        return "/" + str(value)
    if isinstance(value, str):
        return value
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_string(value: Any) -> str | None:
    if isinstance(value, PdfString):
        raw = bytes(value)
        if raw[:2] == b"\xfe\xff":
            try:
                return raw[2:].decode("utf-16-be")
            except UnicodeDecodeError:
                return raw.decode("latin-1", "replace")
        if raw[:2] == b"\xff\xfe":
            try:
                return raw[2:].decode("utf-16-le")
            except UnicodeDecodeError:
                return raw.decode("latin-1", "replace")
        return raw.decode("latin-1", "replace")
    if isinstance(value, str):
        return value
    return None


@dataclass
class PdfPage:
    """One page in document order: its object, geometry, resources,
    content-stream references, and annotations (raw dicts, resolved)."""

    number: int
    ref: PdfRef
    mediabox: tuple[float, float, float, float] = (0, 0, 612, 792)
    rotate: int = 0
    resources: dict[str, Any] = field(default_factory=dict)
    contents: list[PdfRef] = field(default_factory=list)
    annots: list[dict[str, Any]] = field(default_factory=list)
    #: Parallel to ``annots``: the indirect reference of each
    #: annotation (None when the annotation was inline, not indirect).
    annot_refs: list[PdfRef | None] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"number": self.number, "ref": self.ref.num,
                "mediabox": list(self.mediabox), "rotate": self.rotate,
                "contents": [c.num for c in self.contents],
                "annotations": len(self.annots)}


class PdfDocument:
    """A parsed PDF: xref + trailer + catalog + page tree + inventory.

    Open with :meth:`from_bytes` / :meth:`from_path`.  An encrypted
    file (``/Encrypt`` in the trailer) is refused with kind
    ``"encrypted"`` — the reader never guesses at ciphertext.  A file
    whose xref is broken still opens with ``repaired=True`` (the COS
    layer reconstructed the table by scanning); ``repaired`` is part
    of the inventory so callers can see the file was damaged.
    """

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.version = ""
        self.repaired = False
        self.encrypted = False
        self.catalog: dict[str, Any] = {}
        self.info: dict[str, str] = {}
        self.pages: list[PdfPage] = []
        self._xref = None
        self._parse()

    # ── opening ───────────────────────────────────────────────────────
    @classmethod
    def from_bytes(cls, data: bytes) -> PdfDocument:
        if not isinstance(data, (bytes, bytearray)):
            raise _fail("PdfDocument needs PDF bytes",
                        kind="unsupported_format")
        return cls(bytes(data))

    @classmethod
    def from_path(cls, path: str) -> PdfDocument:
        try:
            with open(path, "rb") as handle:
                data = handle.read()
        except OSError as exc:
            raise DocumentError(f"cannot read PDF {path!r}: {exc}",
                                kind="io") from exc
        return cls.from_bytes(data)

    def _parse(self) -> None:
        data = self.data
        header = re.search(rb"%PDF-(\d+\.\d+)", data[:1024])
        if not header:
            raise _fail("not a PDF file (missing %PDF- header)",
                        kind="unsupported_format")
        self.version = header.group(1).decode("ascii")
        xref = parse_xref(data)
        self._xref = xref
        self.repaired = xref.repaired
        trailer = xref.trailer
        if "/Encrypt" in trailer:
            self.encrypted = True
            raise _fail("encrypted PDF: /Encrypt present in the trailer — "
                        "the pdfcraft reader refuses encrypted files; "
                        "unlock it first (pagewise unlock)",
                        kind="encrypted")
        root_ref = trailer.get("/Root")
        if not isinstance(root_ref, PdfRef):
            raise _fail("PDF trailer has no /Root catalog reference")
        catalog = resolve(data, xref, root_ref)
        if not isinstance(catalog, dict):
            raise _fail("PDF catalog object is not a dictionary")
        self.catalog = catalog
        self._root_ref = root_ref
        # /Info lives in the trailer in most files — but Devon's own
        # writer (core/pdf.py) hangs it off the catalog instead.
        info_ref = trailer.get("/Info")
        if not isinstance(info_ref, PdfRef):
            info_ref = catalog.get("/Info")
        if isinstance(info_ref, PdfRef):
            info_obj = resolve(data, xref, info_ref)
            if isinstance(info_obj, dict):
                for key, value in info_obj.items():
                    text = _as_string(value)
                    if text is not None:
                        self.info[key.lstrip("/")] = text
        pages_ref = catalog.get("/Pages")
        if isinstance(pages_ref, PdfRef):
            node = resolve(data, xref, pages_ref)
            if isinstance(node, dict):
                self._walk_tree(node, {}, [], ref=pages_ref)
        elif isinstance(pages_ref, dict):
            self._walk_tree(pages_ref, {}, [])
        if not self.pages:
            raise _fail("no pages found in this PDF")
        self._page_texts: list[str] | None = None

    def _walk_tree(self, node: dict[str, Any], inherited: dict[str, Any],
                   seen: list[int], ref: PdfRef | None = None) -> None:
        """Recursive page-tree walk in document order, inheriting
        MediaBox / Resources / Rotate down the tree (spec §7.7.3.4).
        A node with /Kids is an intermediate node; a node with
        /Contents (or /Type /Page) is a leaf."""
        data, xref = self.data, self._xref
        box = node.get("/MediaBox")
        if isinstance(box, list) and len(box) == 4:
            nums = [_as_float(v) for v in box]
            if all(n is not None for n in nums):
                inherited = dict(inherited)
                inherited["mediabox"] = tuple(nums)  # type: ignore[assignment]
        resources = node.get("/Resources")
        if isinstance(resources, PdfRef):
            resolved = resolve(data, xref, resources)
            if isinstance(resolved, dict):
                inherited = dict(inherited)
                inherited["resources"] = resolved
        elif isinstance(resources, dict):
            inherited = dict(inherited)
            inherited["resources"] = resources
        rotate = _as_int(node.get("/Rotate"))
        if rotate is not None:
            inherited = dict(inherited)
            inherited["rotate"] = rotate % 360
        kids = node.get("/Kids")
        if isinstance(kids, list) and kids:
            for kid in kids:
                if not isinstance(kid, PdfRef):
                    continue
                if kid.num in seen:
                    continue
                seen.append(kid.num)
                child = resolve(data, xref, kid)
                if isinstance(child, dict):
                    self._walk_tree(child, inherited, seen, ref=kid)
            return
        # Leaf page.
        mediabox = inherited.get("mediabox", (0, 0, 612, 792))
        page_resources = inherited.get("resources", {})
        page_rotate = inherited.get("rotate", 0)
        contents: list[PdfRef] = []
        raw_contents = node.get("/Contents")
        if isinstance(raw_contents, PdfRef):
            contents = [raw_contents]
        elif isinstance(raw_contents, list):
            contents = [c for c in raw_contents
                        if isinstance(c, PdfRef)]
        annots: list[dict[str, Any]] = []
        annot_refs: list[PdfRef | None] = []
        raw_annots = node.get("/Annots")
        if isinstance(raw_annots, list):
            for item in raw_annots:
                if isinstance(item, PdfRef):
                    resolved = resolve(data, xref, item)
                    if isinstance(resolved, dict):
                        annots.append(resolved)
                        annot_refs.append(item)
                elif isinstance(item, dict):
                    annots.append(item)
                    annot_refs.append(None)
        self.pages.append(PdfPage(
            number=len(self.pages) + 1,
            ref=ref or PdfRef(0, 0),
            mediabox=tuple(mediabox),  # type: ignore[arg-type]
            rotate=page_rotate,
            resources=page_resources if isinstance(page_resources, dict)
            else {},
            contents=contents,
            annots=annots,
            annot_refs=annot_refs))

    # ── access ────────────────────────────────────────────────────────
    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def xref_generations(self) -> int:
        return self._xref.generations if self._xref else 0

    def get(self, ref: PdfRef) -> Any:
        """Resolve any indirect reference, or None when absent."""
        return resolve(self.data, self._xref, ref)

    def text(self, page_no: int | None = None) -> str:
        """Whole-document text, or one page's text (1-based).  The
        core.pdf engine does the extraction; this only aligns it to
        page numbers."""
        if page_no is None:
            return read_pdf_text(self.data)
        if not 1 <= page_no <= len(self.pages):
            raise _fail(f"page {page_no} out of range "
                        f"(document has {len(self.pages)})",
                        kind="invalid_argument")
        if self._page_texts is None:
            self._page_texts = read_pdf_page_texts(
                self.data, skip_empty=False)
        texts = self._page_texts
        if page_no - 1 < len(texts):
            return texts[page_no - 1]
        return ""

    def fonts(self) -> list[dict[str, Any]]:
        """Every font declared by every page: page, resource name,
        base font, subtype, encoding.  A font the file references but
        does not define is listed with ``base_font=None`` — the gap
        is visible, never papered over."""
        out: list[dict[str, Any]] = []
        for page in self.pages:
            resources = page.resources
            font_dict = resources.get("/Font")
            if isinstance(font_dict, PdfRef):
                font_dict = self.get(font_dict)
            if not isinstance(font_dict, dict):
                continue
            for name, ref in font_dict.items():
                entry: dict[str, Any] = {
                    "page": page.number, "name": name,
                    "base_font": None, "subtype": None,
                    "encoding": None}
                body = None
                if isinstance(ref, PdfRef):
                    body = self.get(ref)
                elif isinstance(ref, dict):
                    body = ref
                if isinstance(body, dict):
                    entry["base_font"] = _as_name(body.get("/BaseFont"))
                    entry["subtype"] = _as_name(body.get("/Subtype"))
                    entry["encoding"] = _as_name(body.get("/Encoding"))
                out.append(entry)
        return out

    def images(self) -> list[dict[str, Any]]:
        """Every image XObject: page, resource name, dimensions, filter,
        colour space, bits per component.  Bytes are never decoded
        here — the inventory names the image, it does not render it."""
        out: list[dict[str, Any]] = []
        for page in self.pages:
            resources = page.resources
            xobjects = resources.get("/XObject")
            if isinstance(xobjects, PdfRef):
                xobjects = self.get(xobjects)
            if not isinstance(xobjects, dict):
                continue
            for name, ref in xobjects.items():
                body = None
                if isinstance(ref, PdfRef):
                    body = self.get(ref)
                elif isinstance(ref, dict):
                    body = ref
                # An image XObject is a stream object — unwrap it.
                if isinstance(body, PdfStream):
                    body = body.dict
                if not isinstance(body, dict):
                    continue
                if _as_name(body.get("/Subtype")) != "/Image":
                    continue
                filt = body.get("/Filter")
                filt = (_as_name(filt[0])
                        if isinstance(filt, list) and filt
                        else _as_name(filt))
                out.append({
                    "page": page.number, "name": name,
                    "width": _as_int(body.get("/Width")),
                    "height": _as_int(body.get("/Height")),
                    "filter": filt,
                    "color_space": _as_name(body.get("/ColorSpace")),
                    "bits_per_component": _as_int(
                        body.get("/BitsPerComponent")),
                })
        return out

    def annotations(self) -> list[dict[str, Any]]:
        """Every annotation: page, subtype, rectangle, contents text.
        The raw object number rides along so an edit can target it."""
        out: list[dict[str, Any]] = []
        for page in self.pages:
            for annot, ref in zip(page.annots, page.annot_refs,
                                  strict=True):
                rect = annot.get("/Rect")
                rect_list = None
                if isinstance(rect, list) and len(rect) == 4:
                    nums = [_as_float(v) for v in rect]
                    if all(n is not None for n in nums):
                        rect_list = nums
                out.append({
                    "page": page.number,
                    "subtype": _as_name(annot.get("/Subtype")),
                    "rect": rect_list,
                    "contents": _as_string(annot.get("/Contents")),
                    "ref": ref.num if ref is not None else None,
                })
        return out

    def to_dict(self) -> dict[str, Any]:
        """The inventory summary the tool reply is built from."""
        return {
            "version": self.version,
            "page_count": self.page_count,
            "encrypted": self.encrypted,
            "repaired": self.repaired,
            "xref_generations": self.xref_generations,
            "size_bytes": len(self.data),
            "info": dict(self.info),
            "pages": [p.to_dict() for p in self.pages],
            "fonts": self.fonts(),
            "images": self.images(),
            "annotations": self.annotations(),
        }
