"""DeckCraft editor — mutate a :class:`DeckDocument` and save it back.

Follows the upstream ``crates/engine`` command design
(``cmd/slide.rs`` + ``cmd/shape.rs`` + ``cmd/text.rs``): edits
address slides and shapes by document order and copy only what
they touch. Every mutation records the slide's pre-edit canonical
form so :meth:`DeckDocument.fidelity_report` can tell an
intentional change from collateral damage, and marks only the
parts it dirtied — the writer copies every other part
byte-identical.

Supported edits: replace text (whole-deck or first hit), restyle
a run, add slides (from a layout, with the layout's placeholders
copied in), add text boxes / tables / pictures, remove and
reorder slides, set speaker notes, set the slide transition, set
core metadata, and create a fresh deck. Masters, layouts, themes
and charts are read-only inventory in this rung — they are
preserved byte-identical on save, never edited.
"""

from __future__ import annotations

import copy
import io
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from PIL import Image as _PILImage

from ..errors import DocumentError
from .model import DeckDocument, DeckRun, DeckSlide
from .ooxml import (
    A_NS,
    CP_NS,
    CT_NS,
    DC_NS,
    P_NS,
    R_NS,
    REL_CHART,
    REL_IMAGE,
    REL_NOTES_SLIDE,
    REL_SLIDE,
    REL_SLIDE_LAYOUT,
    children,
    first_child,
    local,
    next_rel_id,
    parse_xml_bounded,
    qname,
    read_rels,
    rels_part_for,
    serialize_xml,
    write_package,
    write_rels,
)

__all__ = ["DeckEditor"]

P = P_NS
A = A_NS
R = R_NS

#: Relationship type URIs used when the editor writes new rels.
_REL_TYPES = {
    "slide": REL_SLIDE,
    "slideLayout": REL_SLIDE_LAYOUT,
    "notesSlide": REL_NOTES_SLIDE,
    "image": REL_IMAGE,
    "chart": REL_CHART,
}

#: Content types for parts the editor creates.
_CONTENT_TYPES = {
    "slide": "application/vnd.openxmlformats-officedocument."
             "presentationml.slide+xml",
    "notesSlide": "application/vnd.openxmlformats-officedocument."
                  "presentationml.notesSlide+xml",
}

#: a:rPr child order (ECMA-376 fragment) — new formatting elements
#: are inserted at their schema position so existing siblings keep
#: order.
_RPR_CHILD_ORDER = ["ln", "noFill", "solidFill", "gradFill", "blipFill",
                    "pattFill", "grpFill", "effectLst", "highlight",
                    "uLnTx", "uLn", "uFillTx", "uFill", "latin",
                    "ea", "cs", "sym", "hlinkClick", "hlinkMouseOver",
                    "rtl", "extLst"]


def _bool_attr_value(value: bool | None) -> str | None:
    if value is None:
        return None
    return "1" if value else "0"


def _new_id(existing: list[int], start: int = 2) -> int:
    taken = set(existing)
    candidate = max([start] + list(taken))
    while candidate in taken:
        candidate += 1
    return candidate


class DeckEditor:
    """Edit a :class:`DeckDocument` in place, then :meth:`save`."""

    def __init__(self, document: DeckDocument) -> None:
        self.document = document

    # ── construction ──────────────────────────────────────────────
    @classmethod
    def from_bytes(cls, data: bytes) -> DeckEditor:
        return cls(DeckDocument.from_bytes(data))

    @classmethod
    def from_path(cls, path: str | Path) -> DeckEditor:
        return cls(DeckDocument.from_path(path))

    @classmethod
    def create(cls, title: str | None = None,
               author: str | None = None) -> DeckEditor:
        """A fresh blank deck: one master, two layouts, one slide."""
        from .model import DEFAULT_SLIDE_SIZE  # local: avoids a cycle
        parts = _blank_package(title=title, author=author)
        data = write_package(parts)
        editor = cls(DeckDocument.from_bytes(data))
        editor.document.slide_size = DEFAULT_SLIDE_SIZE
        return editor

    # ── helpers ───────────────────────────────────────────────────
    def _slide(self, index: int) -> DeckSlide:
        return self.document.slide(index)

    def _slide_root(self, slide: DeckSlide) -> ET.Element:
        root = self.document.root_for(slide.part)
        if root is None:
            raise DocumentError(f"slide part {slide.part!r} has no "
                                f"parsed root", kind="parse")
        return root

    def _sp_tree(self, slide: DeckSlide) -> ET.Element:
        root = self._slide_root(slide)
        c_sld = first_child(root, "cSld")
        if c_sld is None:
            raise DocumentError(f"slide {slide.index} has no p:cSld",
                                kind="parse")
        sp_tree = first_child(c_sld, "spTree")
        if sp_tree is None:
            raise DocumentError(f"slide {slide.index} has no p:spTree",
                                kind="parse")
        return sp_tree

    def _touch_slide(self, slide: DeckSlide) -> None:
        self.document._note_touched(slide)
        self.document.mark_dirty(slide.part)

    def _next_shape_id(self, slide: DeckSlide) -> int:
        ids = [s.shape_id for s in slide.shapes]
        for child in slide.shapes:
            ids.extend(c.shape_id for c in child.children)
        return _new_id(ids, start=2)

    def _slide_rels(self, slide: DeckSlide
                    ) -> dict[str, dict[str, str]]:
        return self.document.slide_rels(slide.part)

    def _write_slide_rels(self, slide: DeckSlide,
                          rels: dict[str, dict[str, str]]) -> None:
        part = rels_part_for(slide.part)
        self.document._package.set_part(part, write_rels(rels))
        self.document._slide_rels[slide.part] = dict(rels)

    def _set_content_type(self, part: str, content_type: str) -> None:
        """Add an Override for ``part`` to [Content_Types].xml."""
        root = self.document.root_for("[Content_Types].xml")
        if root is None:
            data = self.document._package.get("[Content_Types].xml")
            if data is None:
                raise DocumentError(
                    "package has no [Content_Types].xml", kind="parse")
            root = parse_xml_bounded(data, "[Content_Types].xml")
            self.document._roots["[Content_Types].xml"] = root
        for child in root:
            if local(child.tag) == "Override" and child.get(
                    "PartName") == "/" + part:
                return
        override = ET.Element(qname(CT_NS, "Override"))
        override.set("PartName", "/" + part)
        override.set("ContentType", content_type)
        root.append(override)
        self.document.mark_dirty("[Content_Types].xml")

    def _remove_content_type(self, part: str) -> None:
        root = self.document.root_for("[Content_Types].xml")
        if root is None:
            return
        for child in list(root):
            if local(child.tag) == "Override" and child.get(
                    "PartName") == "/" + part:
                root.remove(child)
                self.document.mark_dirty("[Content_Types].xml")
                return

    # ── text edits ────────────────────────────────────────────────
    def replace_text(self, old: str, new: str,
                     first_only: bool = False) -> int:
        """Replace ``old`` with ``new`` inside run text.

        Replacement happens inside single runs (a match split
        across two runs is not joined — the same boundary the word
        organ documents). Returns the replacement count.
        """
        if not old:
            raise DocumentError("replace_text needs non-empty text",
                                kind="invalid_argument")
        replaced = 0
        for slide in self.document.slides:
            # Note the slide's pre-edit form FIRST: the fidelity
            # report subtracts noted slides, so a note taken after
            # the mutation would not cover the change.
            has_match = any(
                old in run.text
                for shape in slide.shapes
                for para in shape.paragraphs
                for run in para.runs)
            if not has_match:
                continue
            self.document._note_touched(slide)
            changed = False
            for shape in slide.shapes:
                for para in shape.paragraphs:
                    for run in para.runs:
                        if old not in run.text:
                            continue
                        if first_only and replaced:
                            continue
                        count = run.text.count(old)
                        run.text = run.text.replace(old, new)
                        replaced += count
                        changed = True
                        if run.element is not None:
                            _set_run_text(run.element, run.text)
                        if first_only:
                            self.document.mark_dirty(slide.part)
                            return replaced
            if changed:
                self.document.mark_dirty(slide.part)
        return replaced

    def set_run_format(self, slide_index: int, shape_index: int,
                       para_index: int, run_index: int, *,
                       bold: bool | None = None,
                       italic: bool | None = None,
                       color: str | None = None,
                       size_half_points: int | None = None,
                       font: str | None = None) -> None:
        """Restyle one run (DrawingML ``a:rPr`` attributes)."""
        slide = self._slide(slide_index)
        try:
            shape = slide.shapes[shape_index]
            para = shape.paragraphs[para_index]
            run = para.runs[run_index]
        except IndexError as exc:
            raise DocumentError(
                f"run ({slide_index}, {shape_index}, {para_index}, "
                f"{run_index}) does not exist",
                kind="invalid_argument") from exc
        if run.element is None:
            raise DocumentError("run has no backing element",
                                kind="parse")
        self.document._note_touched(slide)
        r_pr = first_child(run.element, "rPr")
        if r_pr is None:
            r_pr = ET.Element(qname(A, "rPr"))
            run.element.insert(0, r_pr)
        if bold is not None:
            r_pr.set("b", _bool_attr_value(bold))
            run.bold = bold
        if italic is not None:
            r_pr.set("i", _bool_attr_value(italic))
            run.italic = italic
        if size_half_points is not None:
            # the model stores half-points; DrawingML a:sz is
            # hundredths of a point
            r_pr.set("sz", str(size_half_points * 50))
            run.size_half_points = size_half_points
        if font is not None:
            latin = first_child(r_pr, "latin")
            if latin is None:
                latin = ET.Element(qname(A, "latin"))
                _insert_rpr_child(r_pr, latin)
            latin.set("typeface", font)
            run.font = font
        if color is not None:
            solid = first_child(r_pr, "solidFill")
            if solid is None:
                solid = ET.Element(qname(A, "solidFill"))
                _insert_rpr_child(r_pr, solid)
            srgb = first_child(solid, "srgbClr")
            if srgb is None:
                srgb = ET.Element(qname(A, "srgbClr"))
                solid.append(srgb)
            srgb.set("val", color.upper().lstrip("#"))
            run.color = color.upper().lstrip("#")
        self._touch_slide(slide)

    # ── slide edits ───────────────────────────────────────────────
    def add_slide(self, layout: str = "content",
                  title: str | None = None) -> int:
        """Append a slide built from the named layout.

        The layout's placeholder shapes are deep-copied in (with
        fresh shape ids) so the new slide looks like a real slide
        from that layout, not an empty frame. Returns the new slide
        index.
        """
        layout_obj = None
        for candidate in self.document.layouts:
            if candidate.kind == layout:
                layout_obj = candidate
                break
        if layout_obj is None and self.document.layouts:
            layout_obj = self.document.layouts[-1]
        if layout_obj is None:
            raise DocumentError("deck has no layouts to build from",
                                kind="parse")
        n_slides = len(self.document.slides)
        new_part = f"ppt/slides/slide{n_slides + 1}.xml"
        while new_part in self.document._package.parts:
            n_slides += 1
            new_part = f"ppt/slides/slide{n_slides + 1}.xml"
        layout_root = self.document.root_for(layout_obj.part)
        if layout_root is None:
            raise DocumentError(
                f"layout part {layout_obj.part!r} has no parsed root",
                kind="parse")
        slide_el = _slide_from_layout(layout_root)
        self.document._roots[new_part] = slide_el
        self.document._package.set_part(
            new_part, serialize_xml(slide_el))
        # slide rels: layout only
        rels = {"rId1": {"type": REL_SLIDE_LAYOUT,
                         "target": layout_obj.part,
                         "raw": _relative_target(
                             new_part, layout_obj.part)}}
        self.document._package.set_part(
            rels_part_for(new_part), write_rels(rels))
        self.document._slide_rels[new_part] = dict(rels)
        self._set_content_type(new_part, _CONTENT_TYPES["slide"])
        # presentation: new sldId + rel
        pres_rels = read_rels(self.document._package,
                              "ppt/presentation.xml")
        rid = next_rel_id(pres_rels)
        pres_rels[rid] = {"type": REL_SLIDE,
                          "target": new_part,
                          "raw": _relative_target(
                              "ppt/presentation.xml", new_part)}
        self.document._package.set_part(
            rels_part_for("ppt/presentation.xml"),
            write_rels(pres_rels))
        sld_id = _new_id([s.slide_id for s in self.document.slides],
                         start=256)
        id_lst = first_child(self.document._pres_root, "sldIdLst")
        if id_lst is None:
            raise DocumentError("presentation has no p:sldIdLst",
                                kind="parse")
        entry = ET.Element(qname(P, "sldId"))
        entry.set("id", str(sld_id))
        entry.set(qname(R, "id"), rid)
        id_lst.append(entry)
        self.document.mark_dirty("ppt/presentation.xml")
        slide = DeckSlide(index=len(self.document.slides),
                          slide_id=sld_id, part=new_part,
                          layout_part=layout_obj.part, element=slide_el)
        self.document._parse_slide_body(
            slide, slide_el, dict(rels))
        if title is not None:
            titled = False
            for shape in slide.shapes:
                if shape.placeholder_type == "title":
                    self._set_shape_text(shape, title)
                    titled = True
                    break
            if not titled:
                # No title placeholder in this layout (a blank
                # layout has none): add a real title text box
                # rather than silently dropping the title.
                shape_id = self._next_shape_id(slide)
                sp = _build_text_shape(
                    shape_id, "Title 1", 457200, 228600, 11277600,
                    914400, title, 64)
                nv_pr = first_child(first_child(sp, "nvSpPr"), "nvPr")
                ph = ET.Element(qname(P, "ph"))
                ph.set("type", "title")
                nv_pr.append(ph)
                self._sp_tree(slide).append(sp)
                parsed = self.document._parse_shape(sp, dict(rels))
                if parsed is not None:
                    slide.shapes.insert(0, parsed)
            self.document.mark_dirty(new_part)
        self.document.slides.append(slide)
        return slide.index

    def remove_slide(self, index: int) -> None:
        """Remove a slide and its rels, notes and content-type."""
        slide = self._slide(index)
        # The removed slide is an intentional change, not collateral.
        self.document._note_touched(slide)
        id_lst = first_child(self.document._pres_root, "sldIdLst")
        if id_lst is not None:
            for entry in list(id_lst):
                if _int_attr(entry, "id") == slide.slide_id:
                    id_lst.remove(entry)
                    break
        pres_rels = read_rels(self.document._package,
                              "ppt/presentation.xml")
        for rid, rel in list(pres_rels.items()):
            if rel["target"] == slide.part:
                del pres_rels[rid]
        self.document._package.set_part(
            rels_part_for("ppt/presentation.xml"),
            write_rels(pres_rels))
        self.document.mark_dirty("ppt/presentation.xml")
        self.document._package.parts.pop(slide.part, None)
        self.document._roots.pop(slide.part, None)
        self.document._slide_rels.pop(slide.part, None)
        self.document._package.parts.pop(
            rels_part_for(slide.part), None)
        if slide.notes_part:
            self.document._package.parts.pop(slide.notes_part, None)
            self.document._roots.pop(slide.notes_part, None)
            self._remove_content_type(slide.notes_part)
        self._remove_content_type(slide.part)
        self.document.slides.pop(index)
        for i, remaining in enumerate(self.document.slides):
            remaining.index = i

    def move_slide(self, index: int, new_index: int) -> None:
        """Reorder slides: move slide ``index`` to ``new_index``."""
        slides = self.document.slides
        if not 0 <= index < len(slides):
            raise DocumentError(f"slide {index} out of range",
                                kind="invalid_argument")
        if not 0 <= new_index < len(slides):
            raise DocumentError(f"slide {new_index} out of range",
                                kind="invalid_argument")
        if index == new_index:
            return
        slide = slides.pop(index)
        slides.insert(new_index, slide)
        for i, remaining in enumerate(slides):
            remaining.index = i
        id_lst = first_child(self.document._pres_root, "sldIdLst")
        if id_lst is None:
            raise DocumentError("presentation has no p:sldIdLst",
                                kind="parse")
        entries = list(id_lst)
        by_id = {int(e.get("id", "0")): e for e in entries}
        for entry in entries:
            id_lst.remove(entry)
        for remaining in slides:
            id_lst.append(by_id[remaining.slide_id])
        self.document.mark_dirty("ppt/presentation.xml")

    # ── shape edits ───────────────────────────────────────────────
    def add_text_box(self, slide_index: int, text: str, *,
                     x: int = 457200, y: int = 1371600,
                     cx: int = 11277600, cy: int = 914400,
                     size_half_points: int = 36) -> int:
        """Append a text box shape; returns its shape id."""
        slide = self._slide(slide_index)
        shape_id = self._next_shape_id(slide)
        sp = _build_text_shape(shape_id, f"TextBox {shape_id}",
                               x, y, cx, cy, text, size_half_points)
        self.document._note_touched(slide)
        self._sp_tree(slide).append(sp)
        self._touch_slide(slide)
        parsed = self.document._parse_shape(
            sp, self._slide_rels(slide))
        if parsed is not None:
            slide.shapes.append(parsed)
        return shape_id

    def add_table(self, slide_index: int, headers: list[str],
                  rows: list[list[str]], *,
                  x: int = 457200, y: int = 1371600,
                  cx: int = 11277600, cy: int = 2743200) -> int:
        """Append a table (header row + body rows); returns the
        shape id."""
        slide = self._slide(slide_index)
        shape_id = self._next_shape_id(slide)
        all_rows = [list(headers)] + [list(r) for r in rows]
        frame = _build_table_frame(shape_id, f"Table {shape_id}",
                                   x, y, cx, cy, all_rows)
        self.document._note_touched(slide)
        self._sp_tree(slide).append(frame)
        self._touch_slide(slide)
        parsed = self.document._parse_shape(
            frame, self._slide_rels(slide))
        if parsed is not None:
            slide.shapes.append(parsed)
        return shape_id

    def add_image(self, slide_index: int, image_bytes: bytes, *,
                  x: int = 457200, y: int = 1371600,
                  cx: int = 4572000, cy: int = 2743200,
                  ext: str = "png") -> int:
        """Embed an image (new media part + rel) and place it as a
        picture shape; returns the shape id."""
        slide = self._slide(slide_index)
        try:
            with _PILImage.open(io.BytesIO(image_bytes)) as img:
                img.verify()
        except Exception as exc:
            raise DocumentError(f"not a readable image: {exc}",
                                kind="invalid_argument") from exc
        n = 1
        while f"ppt/media/image{n}.{ext}" in self.document._package.parts:
            n += 1
        part = f"ppt/media/image{n}.{ext}"
        self.document._package.set_part(part, image_bytes)
        rels = self._slide_rels(slide)
        rid = next_rel_id(rels)
        rels[rid] = {"type": REL_IMAGE,
                     "target": part,
                     "raw": _relative_target(slide.part, part)}
        self._write_slide_rels(slide, rels)
        shape_id = self._next_shape_id(slide)
        pic = _build_picture(shape_id, f"Picture {shape_id}",
                             x, y, cx, cy, rid)
        self.document._note_touched(slide)
        self._sp_tree(slide).append(pic)
        self._touch_slide(slide)
        parsed = self.document._parse_shape(pic, rels)
        if parsed is not None:
            slide.shapes.append(parsed)
        return shape_id

    # ── notes / transition / metadata ─────────────────────────────
    def set_notes(self, slide_index: int, text: str) -> None:
        """Set the speaker notes for a slide (creates the notes
        part when the slide has none)."""
        slide = self._slide(slide_index)
        if slide.notes_part:
            root = self.document.root_for(slide.notes_part)
            if root is None:
                raise DocumentError(
                    f"notes part {slide.notes_part!r} has no parsed "
                    f"root", kind="parse")
            self._set_notes_text(root, text)
            self.document.mark_dirty(slide.notes_part)
            slide.notes = text
            return
        n = 1
        while (f"ppt/notesSlides/notesSlide{n}.xml"
               in self.document._package.parts):
            n += 1
        part = f"ppt/notesSlides/notesSlide{n}.xml"
        root = _build_notes_slide(text)
        self.document._roots[part] = root
        self.document._package.set_part(part, serialize_xml(root))
        self._set_content_type(part, _CONTENT_TYPES["notesSlide"])
        rels = self._slide_rels(slide)
        rid = next_rel_id(rels)
        rels[rid] = {"type": REL_NOTES_SLIDE,
                     "target": part,
                     "raw": _relative_target(slide.part, part)}
        self._write_slide_rels(slide, rels)
        slide.notes_part = part
        slide.notes = text

    def _set_notes_text(self, root: ET.Element, text: str) -> None:
        c_sld = first_child(root, "cSld")
        sp_tree = first_child(c_sld, "spTree") \
            if c_sld is not None else None
        if sp_tree is None:
            return
        for child in sp_tree:
            if local(child.tag) != "sp":
                continue
            nv_pr = first_child(first_child(child, "nvSpPr"), "nvPr")
            ph = first_child(nv_pr, "ph") if nv_pr is not None else None
            if ph is not None and ph.get("type") == "body":
                tx_body = first_child(child, "txBody")
                if tx_body is not None:
                    _set_tx_body_text(tx_body, text)
                return

    def set_transition(self, slide_index: int, kind: str,
                       speed: str = "med") -> None:
        """Set the slide transition (``fade``/``wipe``/``push``/
        ``split``/``cover``/``dissolve``/… — the kind names the
        child element of ``p:transition``)."""
        slide = self._slide(slide_index)
        root = self._slide_root(slide)
        self.document._note_touched(slide)
        old = first_child(root, "transition")
        if old is not None:
            root.remove(old)
        transition = ET.Element(qname(P, "transition"))
        transition.set("spd", speed)
        child = ET.SubElement(transition, qname(P, kind))
        if kind == "fade":
            child.set("thruBlk", "1")
        # p:transition sits after p:cSld and before p:timing
        index = 0
        for i, child_el in enumerate(root):
            if local(child_el.tag) == "cSld":
                index = i + 1
        root.insert(index, transition)
        slide.transition = kind
        slide.transition_speed = speed
        self._touch_slide(slide)

    def set_metadata(self, **fields: str) -> None:
        """Set core properties (``title``/``author``/``description``/
        ``subject``/``keywords``)."""
        root = self.document.root_for("docProps/core.xml")
        if root is None:
            data = self.document._package.get("docProps/core.xml")
            if data is None:
                root = ET.Element(qname(CP_NS, "coreProperties"))
                self.document._roots["docProps/core.xml"] = root
                self.document._package.set_part(
                    "docProps/core.xml", serialize_xml(root))
                self._set_content_type(
                    "docProps/core.xml",
                    "application/vnd.openxmlformats-package."
                    "core-properties+xml")
            else:
                root = parse_xml_bounded(data, "docProps/core.xml")
                self.document._roots["docProps/core.xml"] = root
        mapping = {
            "title": qname(DC_NS, "title"),
            "author": qname(DC_NS, "creator"),
            "description": qname(DC_NS, "description"),
            "subject": qname(DC_NS, "subject"),
            "keywords": qname(CP_NS, "keywords"),
        }
        for key, value in fields.items():
            tag = mapping.get(key)
            if tag is None:
                raise DocumentError(
                    f"unknown metadata field {key!r}",
                    kind="invalid_argument")
            child = root.find(tag)
            if child is None:
                child = ET.Element(tag)
                root.append(child)
            child.text = value
            self.document.core_properties[key] = value
        self.document.mark_dirty("docProps/core.xml")

    def _set_shape_text(self, shape: Any, text: str) -> None:
        if shape.element is None:
            return
        tx_body = first_child(shape.element, "txBody")
        if tx_body is not None:
            _set_tx_body_text(tx_body, text)
            if shape.paragraphs:
                shape.paragraphs[0].runs = [DeckRun(text=text)]

    # ── saving ────────────────────────────────────────────────────
    def save(self) -> bytes:
        return self.document.to_bytes()


def _int_attr(element: ET.Element | None, name: str,
              default: int = 0) -> int:
    if element is None:
        return default
    try:
        return int(element.get(name, str(default)))
    except ValueError:
        return default


def _relative_target(from_part: str, to_part: str) -> str:
    """A package-relative target from ``from_part`` to ``to_part``."""
    from_dir = from_part.rpartition("/")[0].split("/") \
        if "/" in from_part else []
    to = to_part.split("/")
    common = 0
    for a, b in zip(from_dir, to, strict=False):
        if a == b:
            common += 1
        else:
            break
    ups = [".."] * (len(from_dir) - common)
    return "/".join(ups + to[common:])


def _set_run_text(run_el: ET.Element, text: str) -> None:
    for t_el in run_el.iter():
        if local(t_el.tag) == "t":
            t_el.text = text
            return
    t_el = ET.Element(qname(A, "t"))
    t_el.text = text
    run_el.append(t_el)


def _set_tx_body_text(tx_body: ET.Element, text: str) -> None:
    paras = children(tx_body, "p")
    if not paras:
        return
    first = paras[0]
    runs = children(first, "r")
    if runs:
        _set_run_text(runs[0], text)
        for extra in runs[1:]:
            first.remove(extra)
    else:
        r_el = ET.Element(qname(A, "r"))
        t_el = ET.SubElement(r_el, qname(A, "t"))
        t_el.text = text
        first.append(r_el)
    for extra in paras[1:]:
        tx_body.remove(extra)


def _insert_rpr_child(r_pr: ET.Element, child: ET.Element) -> None:
    """Insert a child of a:rPr at its schema position."""
    name = local(child.tag)
    order = _RPR_CHILD_ORDER.index(name) if name in _RPR_CHILD_ORDER \
        else len(_RPR_CHILD_ORDER)
    index = len(r_pr)
    for i, existing in enumerate(r_pr):
        existing_name = local(existing.tag)
        existing_order = _RPR_CHILD_ORDER.index(existing_name) \
            if existing_name in _RPR_CHILD_ORDER \
            else len(_RPR_CHILD_ORDER)
        if existing_order > order:
            index = i
            break
    r_pr.insert(index, child)


def _build_text_shape(shape_id: int, name: str, x: int, y: int,
                      cx: int, cy: int, text: str,
                      size_half_points: int) -> ET.Element:
    sp = ET.Element(qname(P, "sp"))
    nv = ET.SubElement(sp, qname(P, "nvSpPr"))
    c_nv = ET.SubElement(nv, qname(P, "cNvPr"))
    c_nv.set("id", str(shape_id))
    c_nv.set("name", name)
    ET.SubElement(nv, qname(P, "cNvSpPr"))
    ET.SubElement(nv, qname(P, "nvPr"))
    sp_pr = ET.SubElement(sp, qname(P, "spPr"))
    xfrm = ET.SubElement(sp_pr, qname(A, "xfrm"))
    off = ET.SubElement(xfrm, qname(A, "off"))
    off.set("x", str(x))
    off.set("y", str(y))
    ext = ET.SubElement(xfrm, qname(A, "ext"))
    ext.set("cx", str(cx))
    ext.set("cy", str(cy))
    geom = ET.SubElement(sp_pr, qname(A, "prstGeom"))
    geom.set("prst", "rect")
    ET.SubElement(geom, qname(A, "avLst"))
    ET.SubElement(sp_pr, qname(A, "noFill"))
    tx_body = ET.SubElement(sp, qname(P, "txBody"))
    ET.SubElement(tx_body, qname(A, "bodyPr"))
    ET.SubElement(tx_body, qname(A, "lstStyle"))
    p_el = ET.SubElement(tx_body, qname(A, "p"))
    r_el = ET.SubElement(p_el, qname(A, "r"))
    r_pr = ET.SubElement(r_el, qname(A, "rPr"))
    r_pr.set("lang", "en-US")
    r_pr.set("sz", str(size_half_points * 50))
    t_el = ET.SubElement(r_el, qname(A, "t"))
    t_el.text = text
    return sp


def _build_table_frame(shape_id: int, name: str, x: int, y: int,
                       cx: int, cy: int,
                       rows: list[list[str]]) -> ET.Element:
    frame = ET.Element(qname(P, "graphicFrame"))
    nv = ET.SubElement(frame, qname(P, "nvGraphicFramePr"))
    c_nv = ET.SubElement(nv, qname(P, "cNvPr"))
    c_nv.set("id", str(shape_id))
    c_nv.set("name", name)
    ET.SubElement(nv, qname(P, "cNvGraphicFramePr"))
    ET.SubElement(nv, qname(P, "nvPr"))
    xfrm = ET.SubElement(frame, qname(P, "xfrm"))
    off = ET.SubElement(xfrm, qname(A, "off"))
    off.set("x", str(x))
    off.set("y", str(y))
    ext = ET.SubElement(xfrm, qname(A, "ext"))
    ext.set("cx", str(cx))
    ext.set("cy", str(cy))
    graphic = ET.SubElement(frame, qname(A, "graphic"))
    data = ET.SubElement(graphic, qname(A, "graphicData"))
    data.set("uri", A + "/table")
    tbl = ET.SubElement(data, qname(A, "tbl"))
    tbl_pr = ET.SubElement(tbl, qname(A, "tblPr"))
    tbl_pr.set("firstRow", "1")
    tbl_pr.set("bandRow", "1")
    n_cols = max((len(r) for r in rows), default=1)
    grid = ET.SubElement(tbl, qname(A, "tblGrid"))
    for _ in range(n_cols):
        col = ET.SubElement(grid, qname(A, "gridCol"))
        col.set("w", str(cx // n_cols))
    for row in rows:
        tr = ET.SubElement(tbl, qname(A, "tr"))
        tr.set("h", "370840")
        for cell_text in row:
            tc = ET.SubElement(tr, qname(A, "tc"))
            tx_body = ET.SubElement(tc, qname(A, "txBody"))
            ET.SubElement(tx_body, qname(A, "bodyPr"))
            ET.SubElement(tx_body, qname(A, "lstStyle"))
            p_el = ET.SubElement(tx_body, qname(A, "p"))
            r_el = ET.SubElement(p_el, qname(A, "r"))
            r_pr = ET.SubElement(r_el, qname(A, "rPr"))
            r_pr.set("lang", "en-US")
            t_el = ET.SubElement(r_el, qname(A, "t"))
            t_el.text = cell_text
            ET.SubElement(tc, qname(A, "tcPr"))
    return frame


def _build_picture(shape_id: int, name: str, x: int, y: int,
                   cx: int, cy: int, rid: str) -> ET.Element:
    pic = ET.Element(qname(P, "pic"))
    nv = ET.SubElement(pic, qname(P, "nvPicPr"))
    c_nv = ET.SubElement(nv, qname(P, "cNvPr"))
    c_nv.set("id", str(shape_id))
    c_nv.set("name", name)
    ET.SubElement(nv, qname(P, "cNvPicPr"))
    ET.SubElement(nv, qname(P, "nvPr"))
    blip_fill = ET.SubElement(pic, qname(P, "blipFill"))
    blip = ET.SubElement(blip_fill, qname(A, "blip"))
    blip.set(qname(R, "embed"), rid)
    stretch = ET.SubElement(blip_fill, qname(A, "stretch"))
    ET.SubElement(stretch, qname(A, "fillRect"))
    sp_pr = ET.SubElement(pic, qname(P, "spPr"))
    xfrm = ET.SubElement(sp_pr, qname(A, "xfrm"))
    off = ET.SubElement(xfrm, qname(A, "off"))
    off.set("x", str(x))
    off.set("y", str(y))
    ext = ET.SubElement(xfrm, qname(A, "ext"))
    ext.set("cx", str(cx))
    ext.set("cy", str(cy))
    geom = ET.SubElement(sp_pr, qname(A, "prstGeom"))
    geom.set("prst", "rect")
    ET.SubElement(geom, qname(A, "avLst"))
    return pic


def _slide_from_layout(layout_root: ET.Element) -> ET.Element:
    """A new slide element: the layout's placeholder shapes copied
    in with fresh ids, plus the group scaffolding."""
    slide = ET.Element(qname(P, "sld"))
    c_sld = ET.SubElement(slide, qname(P, "cSld"))
    sp_tree = ET.SubElement(c_sld, qname(P, "spTree"))
    nv_grp = ET.SubElement(sp_tree, qname(P, "nvGrpSpPr"))
    c_nv = ET.SubElement(nv_grp, qname(P, "cNvPr"))
    c_nv.set("id", "1")
    c_nv.set("name", "")
    ET.SubElement(nv_grp, qname(P, "cNvGrpSpPr"))
    ET.SubElement(nv_grp, qname(P, "nvPr"))
    grp_pr = ET.SubElement(sp_tree, qname(P, "grpSpPr"))
    xfrm = ET.SubElement(grp_pr, qname(A, "xfrm"))
    for tag, attrs in (("off", {"x": "0", "y": "0"}),
                       ("ext", {"cx": "0", "cy": "0"}),
                       ("chOff", {"x": "0", "y": "0"}),
                       ("chExt", {"cx": "0", "cy": "0"})):
        el = ET.SubElement(xfrm, qname(A, tag))
        for key, value in attrs.items():
            el.set(key, value)
    layout_tree = first_child(first_child(layout_root, "cSld"), "spTree")
    next_id = 2
    if layout_tree is not None:
        for child in layout_tree:
            if local(child.tag) not in ("sp", "pic", "graphicFrame"):
                continue
            copied = copy.deepcopy(child)
            nv = first_child(copied, "nvSpPr")
            if nv is None:
                nv = first_child(copied, "nvPicPr")
            if nv is None:
                nv = first_child(copied, "nvGraphicFramePr")
            c_nv_pr = first_child(nv, "cNvPr") if nv is not None else None
            if c_nv_pr is not None:
                c_nv_pr.set("id", str(next_id))
                next_id += 1
            sp_tree.append(copied)
    clr_map = ET.SubElement(slide, qname(P, "clrMapOvr"))
    ET.SubElement(clr_map, qname(A, "masterClrMapping"))
    return slide


def _build_notes_slide(text: str) -> ET.Element:
    notes = ET.Element(qname(P, "notes"))
    c_sld = ET.SubElement(notes, qname(P, "cSld"))
    sp_tree = ET.SubElement(c_sld, qname(P, "spTree"))
    nv_grp = ET.SubElement(sp_tree, qname(P, "nvGrpSpPr"))
    c_nv = ET.SubElement(nv_grp, qname(P, "cNvPr"))
    c_nv.set("id", "1")
    c_nv.set("name", "")
    ET.SubElement(nv_grp, qname(P, "cNvGrpSpPr"))
    ET.SubElement(nv_grp, qname(P, "nvPr"))
    grp_pr = ET.SubElement(sp_tree, qname(P, "grpSpPr"))
    xfrm = ET.SubElement(grp_pr, qname(A, "xfrm"))
    for tag in ("off", "ext", "chOff", "chExt"):
        el = ET.SubElement(xfrm, qname(A, tag))
        el.set("x" if tag in ("off", "chOff") else "cx", "0")
        el.set("y" if tag in ("off", "chOff") else "cy", "0")
    shape = _build_text_shape(2, "Notes Placeholder 1", 457200, 2743200,
                              5943600, 3657600, text, 36)
    nv_pr = first_child(first_child(shape, "nvSpPr"), "nvPr")
    ph = ET.Element(qname(P, "ph"))
    ph.set("type", "body")
    ph.set("idx", "1")
    nv_pr.append(ph)
    sp_tree.append(shape)
    return notes


_XML_DECL = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'

_CT = "http://schemas.openxmlformats.org/package/2006/content-types"
_PR = "http://schemas.openxmlformats.org/package/2006/relationships"
_CP = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
_DC = "http://purl.org/dc/elements/1.1/"


def _blank_package(title: str | None, author: str | None
                   ) -> dict[str, bytes]:
    """The minimal package for a fresh deck: one master, two
    layouts, one blank slide, a theme, and core properties. The
    XML is composed here (the same shape the fixture generator
    composes) so ``create`` needs no file on disk."""
    content_types = (
        _XML_DECL + f'<Types xmlns="{_CT}">'
        '<Default Extension="rels" ContentType="application/'
        'vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="png" ContentType="image/png"/>'
        '<Override PartName="/ppt/presentation.xml" ContentType='
        '"application/vnd.openxmlformats-officedocument.'
        'presentationml.presentation.main+xml"/>'
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.'
        'presentationml.slideMaster+xml"/>'
        '<Override PartName="/ppt/theme/theme1.xml" ContentType='
        '"application/vnd.openxmlformats-officedocument.theme+xml"/>'
        '<Override PartName="/ppt/slides/slide1.xml" ContentType='
        '"application/vnd.openxmlformats-officedocument.'
        'presentationml.slide+xml"/>'
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.'
        'presentationml.slideLayout+xml"/>'
        '<Override PartName="/ppt/slideLayouts/slideLayout2.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.'
        'presentationml.slideLayout+xml"/>'
        '<Override PartName="/docProps/core.xml" ContentType='
        '"application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" ContentType='
        '"application/vnd.openxmlformats-officedocument.'
        'extended-properties+xml"/></Types>')
    root_rels = (
        _XML_DECL + f'<Relationships xmlns="{_PR}">'
        '<Relationship Id="rId1" Type="http://schemas.'
        'openxmlformats.org/officeDocument/2006/relationships/'
        'officeDocument" Target="ppt/presentation.xml"/>'
        '</Relationships>')
    presentation = (
        _XML_DECL + f'<p:presentation xmlns:a="{A}" xmlns:r="{R}" '
        f'xmlns:p="{P}"><p:sldMasterIdLst>'
        '<p:sldMasterId id="2147483648" r:id="rId1"/>'
        '</p:sldMasterIdLst><p:sldIdLst>'
        '<p:sldId id="256" r:id="rId2"/></p:sldIdLst>'
        '<p:sldSz cx="12192000" cy="6858000"/>'
        '<p:notesSz cx="6858000" cy="9144000"/></p:presentation>')
    pres_rels = (
        _XML_DECL + f'<Relationships xmlns="{_PR}">'
        '<Relationship Id="rId1" Type="http://schemas.'
        'openxmlformats.org/officeDocument/2006/relationships/'
        'slideMaster" Target="slideMasters/slideMaster1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.'
        'openxmlformats.org/officeDocument/2006/relationships/slide" '
        'Target="slides/slide1.xml"/>'
        '<Relationship Id="rId9" Type="http://schemas.'
        'openxmlformats.org/officeDocument/2006/relationships/theme" '
        'Target="theme/theme1.xml"/></Relationships>')
    theme = (
        _XML_DECL + f'<a:theme xmlns:a="{A}" name="Office Theme">'
        '<a:themeElements><a:clrScheme name="Office">'
        '<a:dk1><a:sysClr val="windowText" lastClr="000000"/></a:dk1>'
        '<a:lt1><a:sysClr val="window" lastClr="FFFFFF"/></a:lt1>'
        '<a:dk2><a:srgbClr val="1F497D"/></a:dk2>'
        '<a:lt2><a:srgbClr val="EEECE1"/></a:lt2>'
        '<a:accent1><a:srgbClr val="4472C4"/></a:accent1>'
        '<a:accent2><a:srgbClr val="ED7D31"/></a:accent2>'
        '<a:accent3><a:srgbClr val="A5A5A5"/></a:accent3>'
        '<a:accent4><a:srgbClr val="FFC000"/></a:accent4>'
        '<a:accent5><a:srgbClr val="5B9BD5"/></a:accent5>'
        '<a:accent6><a:srgbClr val="70AD47"/></a:accent6>'
        '<a:hlink><a:srgbClr val="0563C1"/></a:hlink>'
        '<a:folHlink><a:srgbClr val="954F72"/></a:folHlink>'
        '</a:clrScheme><a:fontScheme name="Office"><a:majorFont>'
        '<a:latin typeface="Calibri Light"/></a:majorFont>'
        '<a:minorFont><a:latin typeface="Calibri"/></a:minorFont>'
        '</a:fontScheme><a:fmtScheme name="Office"><a:fillStyleLst>'
        '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
        '</a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill>'
        '<a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst>'
        '<a:effectStyleLst/><a:bgFillStyleLst><a:solidFill>'
        '<a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst>'
        '</a:fmtScheme></a:themeElements></a:theme>')
    master = (
        _XML_DECL + f'<p:sldMaster xmlns:a="{A}" xmlns:r="{R}" '
        f'xmlns:p="{P}"><p:cSld><p:bg><p:bgPr><a:solidFill>'
        '<a:srgbClr val="FFFFFF"/></a:solidFill><a:effectLst/>'
        '</p:bgPr></p:bg><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" '
        'name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr>'
        '<a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
        '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm>'
        '</p:grpSpPr></p:spTree></p:cSld><p:clrMap bg1="lt1" tx1="dk1" '
        'bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" '
        'accent3="accent3" accent4="accent4" accent5="accent5" '
        'accent6="accent6" hlink="hlink" folHlink="folHlink"/>'
        '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/>'
        '<p:sldLayoutId id="2147483650" r:id="rId2"/></p:sldLayoutIdLst>'
        '<p:txStyles><p:titleStyle><a:lvl1pPr algn="l"><a:defRPr '
        'sz="3200"/></a:lvl1pPr></p:titleStyle><p:bodyStyle>'
        '<a:lvl1pPr algn="l"><a:defRPr sz="1800"/></a:lvl1pPr>'
        '</p:bodyStyle><p:otherStyle><a:lvl1pPr algn="l"><a:defRPr '
        'sz="1800"/></a:lvl1pPr></p:otherStyle></p:txStyles>'
        '</p:sldMaster>')
    master_rels = (
        _XML_DECL + f'<Relationships xmlns="{_PR}">'
        '<Relationship Id="rId1" Type="http://schemas.'
        'openxmlformats.org/officeDocument/2006/relationships/'
        'slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.'
        'openxmlformats.org/officeDocument/2006/relationships/'
        'slideLayout" Target="../slideLayouts/slideLayout2.xml"/>'
        '</Relationships>')
    def _make_layout(kind: str) -> str:
        return (
            _XML_DECL + f'<p:sldLayout xmlns:a="{A}" xmlns:r="{R}" '
            f'xmlns:p="{P}" type="{kind}"><p:cSld name="{kind}">'
            '<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/>'
            '<p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm>'
            '<a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" '
            'y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
            '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/>'
            '</p:clrMapOvr></p:sldLayout>')
    layout_title = _make_layout("title")
    layout_content = _make_layout("content")
    slide = (
        _XML_DECL + f'<p:sld xmlns:a="{A}" xmlns:r="{R}" xmlns:p="{P}">'
        '<p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/>'
        '<p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm>'
        '<a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" '
        'y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
        '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/>'
        '</p:clrMapOvr></p:sld>')
    slide_rels = (
        _XML_DECL + f'<Relationships xmlns="{_PR}">'
        '<Relationship Id="rId1" Type="http://schemas.'
        'openxmlformats.org/officeDocument/2006/relationships/'
        'slideLayout" Target="../slideLayouts/slideLayout2.xml"/>'
        '</Relationships>')
    core = (
        _XML_DECL + f'<cp:coreProperties xmlns:cp="{_CP}" '
        f'xmlns:dc="{_DC}">'
        + (f'<dc:title>{title}</dc:title>' if title else "")
        + (f'<dc:creator>{author}</dc:creator>' if author else "")
        + '</cp:coreProperties>')
    app = (
        _XML_DECL + '<Properties xmlns="http://schemas.'
        'openxmlformats.org/officeDocument/2006/extended-properties">'
        '<Application>DeckCraft</Application></Properties>')
    parts = {
        "[Content_Types].xml": content_types.encode(),
        "_rels/.rels": root_rels.encode(),
        "ppt/presentation.xml": presentation.encode(),
        "ppt/_rels/presentation.xml.rels": pres_rels.encode(),
        "ppt/theme/theme1.xml": theme.encode(),
        "ppt/slideMasters/slideMaster1.xml": master.encode(),
        "ppt/slideMasters/_rels/slideMaster1.xml.rels":
            master_rels.encode(),
        "ppt/slideLayouts/slideLayout1.xml": layout_title.encode(),
        "ppt/slideLayouts/slideLayout2.xml": layout_content.encode(),
        "ppt/slides/slide1.xml": slide.encode(),
        "ppt/slides/_rels/slide1.xml.rels": slide_rels.encode(),
        "docProps/core.xml": core.encode(),
        "docProps/app.xml": app.encode(),
    }
    return parts
