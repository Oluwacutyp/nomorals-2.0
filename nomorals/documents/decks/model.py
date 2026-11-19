"""DeckCraft document model — the editable presentation model.

Follows the upstream ``crates/model`` design (``lib.rs``:
``Presentation`` → ``Slide`` → ``Shape`` with ``ShapeKind``,
``TextBody``, ``Table``, ``Chart``): a :class:`DeckDocument` holds
the slides (in presentation order), the masters and layouts they
reference, the theme colour scheme, the media inventory, and the
core properties. Positions are slide indices + shape indices in
document order.

The model is element-backed: every slide/shape keeps its
``xml.etree`` element, so edits mutate the tree in place and the
writer re-serializes only the parts that were dirtied — unedited
parts leave the file byte-identical (see
:meth:`DeckDocument.fidelity_report`).

What is modelled vs preserved: text boxes, autoshapes, pictures,
tables and charts are modelled; groups are modelled recursively;
SmartArt/OLE/media shapes are kept as opaque elements (their XML
survives round trip, their fallback picture is inventoried) — the
same ``Opaque`` treatment as upstream's ``ShapeKind::Opaque``.
Animations are inventoried (timing element presence) but not
modelled; transitions are modelled as a kind + speed.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import DocumentError
from .ooxml import (
    A_NS,
    C_NS,
    CP_NS,
    DC_NS,
    DCT_NS,
    P_NS,
    R_NS,
    REL_CHART,
    REL_IMAGE,
    REL_NOTES_SLIDE,
    REL_SLIDE,
    REL_SLIDE_LAYOUT,
    REL_SLIDE_MASTER,
    REL_THEME,
    children,
    find_all,
    first_child,
    local,
    parse_part,
    parse_xml_bounded,
    qname,
    read_package,
    read_rels,
    serialize_xml,
    write_package,
)

__all__ = [
    "DeckDocument",
    "DeckLayout",
    "DeckMaster",
    "DeckParagraph",
    "DeckRun",
    "DeckShape",
    "DeckSlide",
]

P = P_NS
A = A_NS
R = R_NS
C = C_NS

#: EMU per inch — the DrawingML unit.
EMU_PER_INCH = 914400
#: Default slide size (16:9) in EMU.
DEFAULT_SLIDE_SIZE = (12192000, 6858000)


def _int_attr(element: ET.Element | None, name: str,
              default: int = 0) -> int:
    if element is None:
        return default
    try:
        return int(element.get(name, str(default)))
    except ValueError:
        return default


def _bool_attr(element: ET.Element | None, name: str) -> bool:
    """An ``a:b``-style flag: present = on (unless val=0/false)."""
    if element is None:
        return False
    val = element.get(name)
    if val is None:
        return True
    return val not in ("0", "false", "off", "none")


def _color_of(fill_el: ET.Element | None) -> str | None:
    """The hex colour of a solidFill element (srgbClr or schemeClr)."""
    if fill_el is None:
        return None
    solid = first_child(fill_el, "solidFill")
    if solid is None:
        return None
    srgb = first_child(solid, "srgbClr")
    if srgb is not None:
        return (srgb.get("val") or "").upper() or None
    scheme = first_child(solid, "schemeClr")
    if scheme is not None:
        return f"scheme:{scheme.get('val', '')}"
    return None


def _fill_of(sp_pr: ET.Element | None) -> str | None:
    """The fill of a shape properties element, or None for no fill."""
    if sp_pr is None:
        return None
    if first_child(sp_pr, "noFill") is not None:
        return None
    solid = first_child(sp_pr, "solidFill")
    if solid is not None:
        srgb = first_child(solid, "srgbClr")
        if srgb is not None:
            return (srgb.get("val") or "").upper() or None
        scheme = first_child(solid, "schemeClr")
        if scheme is not None:
            return f"scheme:{scheme.get('val', '')}"
    if first_child(sp_pr, "gradFill") is not None:
        return "gradient"
    if first_child(sp_pr, "blipFill") is not None:
        return "picture"
    return None


def _parse_xfrm(sp_pr: ET.Element | None) -> dict[str, int]:
    xfrm = first_child(sp_pr, "xfrm") if sp_pr is not None else None
    return _xfrm_dict(xfrm)


def _xfrm_dict(xfrm: ET.Element | None) -> dict[str, int]:
    """{x, y, cx, cy, rot} from an ``a:xfrm`` element."""
    if xfrm is None:
        return {"x": 0, "y": 0, "cx": 0, "cy": 0, "rot": 0}
    off = first_child(xfrm, "off")
    ext = first_child(xfrm, "ext")
    return {
        "x": _int_attr(off, "x"),
        "y": _int_attr(off, "y"),
        "cx": _int_attr(ext, "cx"),
        "cy": _int_attr(ext, "cy"),
        "rot": _int_attr(xfrm, "rot"),
    }


@dataclass
class DeckRun:
    """One run of text (``a:r``) with its formatting."""

    text: str = ""
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strike: bool = False
    color: str | None = None
    size_half_points: int | None = None
    font: str | None = None
    element: ET.Element | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text, "bold": self.bold, "italic": self.italic,
            "underline": self.underline, "strike": self.strike,
            "color": self.color,
            "size_half_points": self.size_half_points, "font": self.font,
        }


@dataclass
class DeckParagraph:
    """One paragraph (``a:p``): runs + alignment."""

    runs: list[DeckRun] = field(default_factory=list)
    alignment: str | None = None
    element: ET.Element | None = field(default=None, repr=False)

    @property
    def text(self) -> str:
        return "".join(run.text for run in self.runs)

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "alignment": self.alignment,
                "runs": [r.to_dict() for r in self.runs]}


@dataclass
class DeckShape:
    """One shape on a slide (``p:sp`` / ``p:pic`` /
    ``p:graphicFrame`` / ``p:grpSp``).

    ``kind`` is ``shape`` (autoshape/text box), ``picture``,
    ``table``, ``chart``, ``group``, or ``opaque`` (SmartArt/OLE/
    media — XML preserved, fallback picture inventoried).
    """

    kind: str = "shape"
    shape_id: int = 0
    name: str = ""
    geometry: str | None = None
    placeholder_type: str | None = None
    fill: str | None = None
    line_color: str | None = None
    line_width_emu: int | None = None
    paragraphs: list[DeckParagraph] = field(default_factory=list)
    table: list[list[str]] | None = None
    chart_title: str | None = None
    chart_part: str | None = None
    image_part: str | None = None
    xfrm: dict[str, int] = field(
        default_factory=lambda: {"x": 0, "y": 0, "cx": 0, "cy": 0,
                                 "rot": 0})
    children: list[DeckShape] = field(default_factory=list)
    element: ET.Element | None = field(default=None, repr=False)

    @property
    def text(self) -> str:
        if self.table is not None:
            return "\n".join(" | ".join(row) for row in self.table)
        parts: list[str] = []
        for para in self.paragraphs:
            if para.text:
                parts.append(para.text)
        for child in self.children:
            if child.text:
                parts.append(child.text)
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "id": self.shape_id, "name": self.name,
            "geometry": self.geometry,
            "placeholder": self.placeholder_type,
            "fill": self.fill, "line_color": self.line_color,
            "line_width_emu": self.line_width_emu,
            "text": self.text,
            "paragraphs": [p.to_dict() for p in self.paragraphs],
            "table": [list(r) for r in self.table]
                     if self.table is not None else None,
            "chart_title": self.chart_title,
            "chart_part": self.chart_part,
            "image_part": self.image_part,
            "xfrm": dict(self.xfrm),
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class DeckSlide:
    """One slide (``p:sld``): shapes in document order."""

    index: int = 0
    slide_id: int = 0
    part: str = ""
    layout_part: str = ""
    name: str = ""
    shapes: list[DeckShape] = field(default_factory=list)
    transition: str | None = None
    transition_speed: str | None = None
    background_fill: str | None = None
    hidden: bool = False
    has_animation: bool = False
    notes: str = ""
    notes_part: str = ""
    element: ET.Element | None = field(default=None, repr=False)

    @property
    def text(self) -> str:
        return "\n".join(s.text for s in self.shapes if s.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "id": self.slide_id, "part": self.part,
            "layout": self.layout_part, "name": self.name,
            "transition": self.transition,
            "transition_speed": self.transition_speed,
            "background_fill": self.background_fill,
            "hidden": self.hidden, "has_animation": self.has_animation,
            "notes": self.notes,
            "shapes": [s.to_dict() for s in self.shapes],
        }


@dataclass
class DeckLayout:
    """One slide layout (``p:sldLayout``) — inventory only."""

    part: str
    kind: str = ""
    name: str = ""
    element: ET.Element | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {"part": self.part, "kind": self.kind, "name": self.name}


@dataclass
class DeckMaster:
    """One slide master (``p:sldMaster``) — inventory only."""

    part: str
    name: str = ""
    element: ET.Element | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {"part": self.part, "name": self.name}


class DeckDocument:
    """An editable .pptx presentation.

    Open with :meth:`from_bytes` / :meth:`from_path`; inspect
    :attr:`slides`, :attr:`layouts`, :attr:`masters`,
    :attr:`theme_colors`, :attr:`images`, :attr:`core_properties`;
    mutate through :class:`nomorals.documents.decks.edit.DeckEditor`
    (or the element-backed objects directly); save with
    :meth:`to_bytes`; render with
    :func:`nomorals.documents.decks.render.render_slide_to_png`.
    """

    def __init__(self, package, pres_root: ET.Element,
                 original_bytes: bytes) -> None:
        self._package = package
        self._pres_root = pres_root
        self._original_bytes = original_bytes
        self._roots: dict[str, ET.Element] = {
            "ppt/presentation.xml": pres_root}
        self._touched_keys: set[str] = set()
        self.slides: list[DeckSlide] = []
        self.masters: list[DeckMaster] = []
        self.layouts: list[DeckLayout] = []
        self.theme_colors: dict[str, str] = {}
        self.slide_size: tuple[int, int] = DEFAULT_SLIDE_SIZE
        self.notes_size: tuple[int, int] = (6858000, 9144000)
        self.sections: list[dict[str, Any]] = []
        self.core_properties: dict[str, str] = {}
        self.images: list[dict[str, Any]] = []
        self._slide_rels: dict[str, dict[str, dict[str, str]]] = {}
        self._parse()

    # ── construction ──────────────────────────────────────────────
    @classmethod
    def from_bytes(cls, data: bytes) -> DeckDocument:
        package = read_package(data)
        root = parse_part(package, "ppt/presentation.xml")
        if root is None or local(root.tag) != "presentation":
            raise DocumentError(
                "not a .pptx file: ppt/presentation.xml has no "
                "p:presentation root", kind="parse")
        return cls(package, root, data)

    @classmethod
    def from_path(cls, path: str | Path) -> DeckDocument:
        try:
            data = Path(path).read_bytes()
        except OSError as exc:
            raise DocumentError(f"cannot read {path}: {exc}",
                                kind="io") from exc
        return cls.from_bytes(data)

    # ── parsing ───────────────────────────────────────────────────
    def _parse(self) -> None:
        self._parse_core_properties()
        self._parse_slide_size()
        pres_rels = read_rels(self._package, "ppt/presentation.xml")
        self._parse_theme(pres_rels)
        self._parse_masters(pres_rels)
        self._parse_sections()
        self._parse_slides(pres_rels)

    def _parse_core_properties(self) -> None:
        root = parse_part(self._package, "docProps/core.xml")
        if root is None:
            return
        self._roots["docProps/core.xml"] = root
        mapping = {
            qname(DC_NS, "title"): "title",
            qname(DC_NS, "creator"): "author",
            qname(DC_NS, "description"): "description",
            qname(DC_NS, "subject"): "subject",
            qname(CP_NS, "keywords"): "keywords",
            qname(CP_NS, "lastModifiedBy"): "last_modified_by",
            qname(DCT_NS, "created"): "created",
            qname(DCT_NS, "modified"): "modified",
        }
        for child in root:
            key = mapping.get(child.tag)
            if key and child.text:
                self.core_properties[key] = child.text

    def _parse_slide_size(self) -> None:
        sld_sz = first_child(self._pres_root, "sldSz")
        if sld_sz is not None:
            self.slide_size = (
                _int_attr(sld_sz, "cx", DEFAULT_SLIDE_SIZE[0]),
                _int_attr(sld_sz, "cy", DEFAULT_SLIDE_SIZE[1]))
        notes_sz = first_child(self._pres_root, "notesSz")
        if notes_sz is not None:
            self.notes_size = (
                _int_attr(notes_sz, "cx", 6858000),
                _int_attr(notes_sz, "cy", 9144000))

    def _parse_theme(self, pres_rels: dict[str, dict[str, str]]) -> None:
        for rel in pres_rels.values():
            if rel["type"] != REL_THEME:
                continue
            root = parse_part(self._package, rel["target"])
            if root is None:
                continue
            self._roots[rel["target"]] = root
            scheme = next(
                (e for e in root.iter() if local(e.tag) == "clrScheme"),
                None)
            if scheme is None:
                continue
            for child in scheme:
                name = local(child.tag)
                srgb = first_child(child, "srgbClr")
                sys = first_child(child, "sysClr")
                if srgb is not None and srgb.get("val"):
                    self.theme_colors[name] = srgb.get("val", "").upper()
                elif sys is not None and sys.get("lastClr"):
                    self.theme_colors[name] = sys.get(
                        "lastClr", "").upper()
            break

    def _parse_masters(self, pres_rels: dict[str, dict[str, str]]
                       ) -> None:
        for rel in pres_rels.values():
            if rel["type"] != REL_SLIDE_MASTER:
                continue
            root = parse_part(self._package, rel["target"])
            if root is None:
                continue
            self._roots[rel["target"]] = root
            self.masters.append(DeckMaster(
                part=rel["target"], element=root))
            master_rels = read_rels(self._package, rel["target"])
            for mrel in master_rels.values():
                if mrel["type"] != REL_SLIDE_LAYOUT:
                    continue
                lroot = parse_part(self._package, mrel["target"])
                if lroot is None:
                    continue
                self._roots[mrel["target"]] = lroot
                c_sld = first_child(lroot, "cSld")
                self.layouts.append(DeckLayout(
                    part=mrel["target"],
                    kind=lroot.get("type", ""),
                    name=c_sld.get("name", "") if c_sld is not None
                    else "",
                    element=lroot))
            break

    def _parse_sections(self) -> None:
        ext_lst = first_child(self._pres_root, "extLst")
        if ext_lst is None:
            return
        for section in find_all(ext_lst, "section"):
            name = section.get("name", "")
            slide_ids = [int(s.get("id", "0"))
                         for s in children(section, "sldId")]
            self.sections.append({"name": name, "slide_ids": slide_ids})

    def _parse_slides(self, pres_rels: dict[str, dict[str, str]]
                      ) -> None:
        id_lst = first_child(self._pres_root, "sldIdLst")
        if id_lst is None:
            return
        for index, sld_id_el in enumerate(children(id_lst, "sldId")):
            slide_id = _int_attr(sld_id_el, "id", 256 + index)
            rid = sld_id_el.get(qname(R_NS, "id"), "")
            rel = pres_rels.get(rid)
            if rel is None or rel["type"] != REL_SLIDE:
                continue
            part = rel["target"]
            root = parse_part(self._package, part)
            if root is None:
                continue
            self._roots[part] = root
            slide = DeckSlide(index=index, slide_id=slide_id, part=part,
                              element=root)
            slide_rels = read_rels(self._package, part)
            self._slide_rels[part] = slide_rels
            layout_rid = next(
                (r for r, v in slide_rels.items()
                 if v["type"] == REL_SLIDE_LAYOUT), None)
            if layout_rid:
                slide.layout_part = slide_rels[layout_rid]["target"]
            self._parse_slide_body(slide, root, slide_rels)
            self._parse_notes(slide, slide_rels)
            self.slides.append(slide)

    def _parse_slide_body(self, slide: DeckSlide, root: ET.Element,
                          rels: dict[str, dict[str, str]]) -> None:
        c_sld = first_child(root, "cSld")
        if c_sld is None:
            return
        bg = first_child(c_sld, "bg")
        if bg is not None:
            bg_pr = first_child(bg, "bgPr")
            if bg_pr is not None:
                slide.background_fill = _fill_of(bg_pr)
        sp_tree = first_child(c_sld, "spTree")
        if sp_tree is not None:
            for child in sp_tree:
                name = local(child.tag)
                if name in ("sp", "pic", "graphicFrame", "grpSp"):
                    shape = self._parse_shape(child, rels)
                    if shape is not None:
                        slide.shapes.append(shape)
        transition = first_child(root, "transition")
        if transition is not None:
            slide.transition_speed = transition.get("spd")
            for child in transition:
                slide.transition = local(child.tag)
                break
        timing = first_child(root, "timing")
        slide.has_animation = timing is not None
        slide.hidden = root.get("show", "1") == "0"

    def _parse_notes(self, slide: DeckSlide,
                     rels: dict[str, dict[str, str]]) -> None:
        for rel in rels.values():
            if rel["type"] != REL_NOTES_SLIDE:
                continue
            part = rel["target"]
            root = parse_part(self._package, part)
            if root is None:
                continue
            self._roots[part] = root
            slide.notes_part = part
            c_sld = first_child(root, "cSld")
            if c_sld is None:
                continue
            sp_tree = first_child(c_sld, "spTree")
            if sp_tree is None:
                continue
            texts: list[str] = []
            for child in sp_tree:
                if local(child.tag) != "sp":
                    continue
                shape = self._parse_shape(child, rels)
                if shape is None:
                    continue
                if shape.placeholder_type == "body" and shape.text:
                    texts.append(shape.text)
            slide.notes = "\n".join(texts)
            break

    def _parse_shape(self, element: ET.Element,
                     rels: dict[str, dict[str, str]]
                     ) -> DeckShape | None:
        name = local(element.tag)
        if name == "grpSp":
            return self._parse_group(element, rels)
        if name == "pic":
            return self._parse_picture(element, rels)
        if name == "graphicFrame":
            return self._parse_graphic_frame(element, rels)
        if name != "sp":
            return None
        nv_sp_pr = first_child(element, "nvSpPr")
        c_nv_pr = first_child(nv_sp_pr, "cNvPr") \
            if nv_sp_pr is not None else None
        nv_pr = first_child(nv_sp_pr, "nvPr") \
            if nv_sp_pr is not None else None
        ph = first_child(nv_pr, "ph") if nv_pr is not None else None
        sp_pr = first_child(element, "spPr")
        geom_el = first_child(sp_pr, "prstGeom") \
            if sp_pr is not None else None
        ln = first_child(sp_pr, "ln") if sp_pr is not None else None
        shape = DeckShape(
            kind="shape",
            shape_id=_int_attr(c_nv_pr, "id"),
            name=c_nv_pr.get("name", "") if c_nv_pr is not None else "",
            geometry=geom_el.get("prst") if geom_el is not None else None,
            placeholder_type=ph.get("type") if ph is not None else None,
            fill=_fill_of(sp_pr),
            line_color=_color_of(ln),
            line_width_emu=_int_attr(ln, "w") or None,
            xfrm=_parse_xfrm(sp_pr),
            element=element)
        tx_body = first_child(element, "txBody")
        if tx_body is not None:
            shape.paragraphs = self._parse_tx_body(tx_body)
        return shape

    def _parse_group(self, element: ET.Element,
                     rels: dict[str, dict[str, str]]) -> DeckShape:
        nv_grp = first_child(element, "nvGrpSpPr")
        c_nv_pr = first_child(nv_grp, "cNvPr") \
            if nv_grp is not None else None
        grp_sp_pr = first_child(element, "grpSpPr")
        shape = DeckShape(
            kind="group",
            shape_id=_int_attr(c_nv_pr, "id"),
            name=c_nv_pr.get("name", "") if c_nv_pr is not None else "",
            xfrm=_parse_xfrm(grp_sp_pr),
            element=element)
        for child in element:
            tag = local(child.tag)
            if tag in ("sp", "pic", "graphicFrame", "grpSp"):
                parsed = self._parse_shape(child, rels)
                if parsed is not None:
                    shape.children.append(parsed)
        return shape

    def _parse_picture(self, element: ET.Element,
                       rels: dict[str, dict[str, str]]) -> DeckShape:
        nv_pic_pr = first_child(element, "nvPicPr")
        c_nv_pr = first_child(nv_pic_pr, "cNvPr") \
            if nv_pic_pr is not None else None
        sp_pr = first_child(element, "spPr")
        shape = DeckShape(
            kind="picture",
            shape_id=_int_attr(c_nv_pr, "id"),
            name=c_nv_pr.get("name", "") if c_nv_pr is not None else "",
            xfrm=_parse_xfrm(sp_pr),
            element=element)
        blip_fill = first_child(element, "blipFill")
        blip = first_child(blip_fill, "blip") \
            if blip_fill is not None else None
        if blip is not None:
            rid = blip.get(qname(R_NS, "embed"), "")
            rel = rels.get(rid)
            if rel is not None and rel["type"] == REL_IMAGE:
                shape.image_part = rel["target"]
                data = self._package.get(rel["target"])
                self.images.append({
                    "rel_id": rid, "part": rel["target"],
                    "size": len(data) if data is not None else 0,
                })
        return shape

    def _parse_graphic_frame(
            self, element: ET.Element,
            rels: dict[str, dict[str, str]]) -> DeckShape | None:
        nv_pr = first_child(element, "nvGraphicFramePr")
        c_nv_pr = first_child(nv_pr, "cNvPr") \
            if nv_pr is not None else None
        shape = DeckShape(
            shape_id=_int_attr(c_nv_pr, "id"),
            name=c_nv_pr.get("name", "") if c_nv_pr is not None else "",
            xfrm=_xfrm_dict(first_child(element, "xfrm")),
            element=element)
        graphic = first_child(element, "graphic")
        data = first_child(graphic, "graphicData") \
            if graphic is not None else None
        if data is None:
            shape.kind = "opaque"
            return shape
        uri = data.get("uri", "")
        if uri.endswith("/table"):
            shape.kind = "table"
            shape.table = self._parse_table(data)
        elif uri.endswith("/chart"):
            shape.kind = "chart"
            chart_el = first_child(data, "chart")
            if chart_el is not None:
                rid = chart_el.get(qname(R_NS, "id"), "")
                rel = rels.get(rid)
                if rel is not None and rel["type"] == REL_CHART:
                    shape.chart_part = rel["target"]
                    shape.chart_title = self._parse_chart_title(
                        rel["target"])
        else:
            shape.kind = "opaque"
        return shape

    def _parse_table(self, data: ET.Element) -> list[list[str]]:
        tbl = first_child(data, "tbl")
        if tbl is None:
            return []
        rows: list[list[str]] = []
        for tr in children(tbl, "tr"):
            row: list[str] = []
            for tc in children(tr, "tc"):
                tx_body = first_child(tc, "txBody")
                text = ""
                if tx_body is not None:
                    text = "\n".join(
                        p.text for p in self._parse_tx_body(tx_body)
                        if p.text)
                row.append(text)
            rows.append(row)
        return rows

    def _parse_chart_title(self, part: str) -> str | None:
        root = parse_part(self._package, part)
        if root is None:
            return None
        title = next((e for e in root.iter()
                      if local(e.tag) == "title"), None)
        if title is None:
            return None
        texts = [t.text for t in title.iter()
                 if local(t.tag) == "t" and t.text]
        return " ".join(texts) if texts else None

    def _parse_tx_body(self, tx_body: ET.Element
                       ) -> list[DeckParagraph]:
        paragraphs: list[DeckParagraph] = []
        for p_el in children(tx_body, "p"):
            alignment = None
            p_pr = first_child(p_el, "pPr")
            if p_pr is not None:
                alignment = p_pr.get("algn")
            para = DeckParagraph(alignment=alignment, element=p_el)
            for child in p_el:
                tag = local(child.tag)
                if tag == "r":
                    para.runs.append(self._parse_run(child))
                elif tag == "br":
                    if para.runs:
                        para.runs[-1].text += "\n"
                    else:
                        para.runs.append(DeckRun(text="\n"))
            paragraphs.append(para)
        return paragraphs

    def _parse_run(self, element: ET.Element) -> DeckRun:
        r_pr = first_child(element, "rPr")
        text = "".join(t.text or "" for t in element.iter()
                       if local(t.tag) == "t")
        color = None
        font = None
        size = None
        if r_pr is not None:
            solid = first_child(r_pr, "solidFill")
            if solid is not None:
                srgb = first_child(solid, "srgbClr")
                if srgb is not None:
                    color = (srgb.get("val") or "").upper() or None
                else:
                    scheme = first_child(solid, "schemeClr")
                    if scheme is not None:
                        color = f"scheme:{scheme.get('val', '')}"
            latin = first_child(r_pr, "latin")
            if latin is not None:
                font = latin.get("typeface")
            if r_pr.get("sz") is not None:
                try:
                    # DrawingML a:sz is hundredths of a point; the
                    # model stores half-points like the word organ.
                    size = int(r_pr.get("sz", "0")) // 50
                except ValueError:
                    size = None
        return DeckRun(
            text=text,
            bold=_bool_attr(r_pr, "b") if r_pr is not None else False,
            italic=_bool_attr(r_pr, "i") if r_pr is not None else False,
            underline=first_child(r_pr, "u") is not None
            if r_pr is not None else False,
            strike=_bool_attr(r_pr, "strike")
            if r_pr is not None else False,
            color=color, size_half_points=size, font=font,
            element=element)

    # ── views ─────────────────────────────────────────────────────
    def slide(self, index: int) -> DeckSlide:
        try:
            return self.slides[index]
        except IndexError as exc:
            raise IndexError(
                f"slide {index} out of range for a "
                f"{len(self.slides)}-slide deck") from exc

    @property
    def part_names(self) -> list[str]:
        return self._package.part_names()

    def plain_text(self) -> str:
        return "\n".join(s.text for s in self.slides if s.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slides": [s.to_dict() for s in self.slides],
            "slide_size": list(self.slide_size),
            "masters": [m.to_dict() for m in self.masters],
            "layouts": [lay.to_dict() for lay in self.layouts],
            "theme_colors": dict(self.theme_colors),
            "sections": [dict(s) for s in self.sections],
            "images": [dict(i) for i in self.images],
            "core_properties": dict(self.core_properties),
            "part_names": self.part_names,
        }

    # ── writing ───────────────────────────────────────────────────
    def _note_touched(self, slide: DeckSlide) -> None:
        """Record a slide's pre-edit canonical form.

        The fidelity report subtracts these from the vanished set —
        an edited slide is expected to change; an untouched one is
        not.
        """
        if slide.element is not None:
            key = serialize_xml(slide.element).decode("utf-8", "replace")
            if key:
                self._touched_keys.add(key)

    def mark_dirty(self, part: str) -> None:
        self._package.mark_dirty(part)

    def set_root(self, part: str, root: ET.Element) -> None:
        self._roots[part] = root
        self.mark_dirty(part)

    def root_for(self, part: str) -> ET.Element | None:
        return self._roots.get(part)

    def slide_rels(self, part: str) -> dict[str, dict[str, str]]:
        return dict(self._slide_rels.get(part, {}))

    def to_bytes(self) -> bytes:
        """Serialize back to .pptx bytes.

        Untouched parts are written back byte-identical; only dirty
        parts are re-serialized. A document with no edits at all
        returns the original bytes verbatim.
        """
        if not self._package.dirty:
            return self._original_bytes
        parts: dict[str, bytes] = {}
        for name in self._package.part_names():
            if name in self._roots and name in self._package.dirty:
                parts[name] = serialize_xml(self._roots[name])
            else:
                raw = self._package.get(name)
                parts[name] = raw if raw is not None else b""
        return write_package(parts)

    def fidelity_report(self, original_bytes: bytes) -> dict[str, Any]:
        """Measure how much of the original survived this document.

        Part-level: every part that was not dirtied must be
        byte-identical. Slide-level: every original slide that the
        editor did not intentionally touch must still be present
        (canonical multiset comparison, so XML spelling differences
        never count).
        """
        original_pkg = read_package(original_bytes)
        orig_root = parse_xml_bounded(
            original_pkg.parts["ppt/presentation.xml"],
            "ppt/presentation.xml")
        orig_keys = Counter()
        orig_pres_rels = read_rels(original_pkg, "ppt/presentation.xml")
        id_lst = first_child(orig_root, "sldIdLst")
        if id_lst is not None:
            for sld_id_el in children(id_lst, "sldId"):
                rid = sld_id_el.get(qname(R_NS, "id"), "")
                rel = orig_pres_rels.get(rid)
                if rel is None:
                    continue
                sld_root = parse_part(original_pkg, rel["target"])
                if sld_root is not None:
                    orig_keys[serialize_xml(sld_root).decode(
                        "utf-8", "replace")] += 1
        saved_keys = Counter()
        for slide in self.slides:
            if slide.element is not None:
                saved_keys[serialize_xml(slide.element).decode(
                    "utf-8", "replace")] += 1
        removed = orig_keys - saved_keys
        touched = Counter(self._touched_keys)
        untouched_changed = sum((removed - touched).values())
        changed: list[str] = []
        identical = 0
        names = set(original_pkg.part_names()) | set(
            self._package.part_names())
        for name in sorted(names):
            orig = original_pkg.get(name)
            if name in self._package.dirty:
                changed.append(name)
            elif orig is not None and self._package.get(name) == orig:
                identical += 1
            elif orig is None and self._package.get(name) is not None:
                changed.append(name)
            else:
                changed.append(name)
        return {
            "parts_total": len(names),
            "parts_byte_identical": identical,
            "parts_changed": changed,
            "untouched_slides_changed": untouched_changed,
            "slides_original": sum(orig_keys.values()),
            "slides_saved": sum(saved_keys.values()),
        }
