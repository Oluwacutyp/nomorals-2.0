"""WordCraft document model — the editable word-processor model.

Follows the upstream ``crates/doc`` design (``document.rs``): a
:class:`WordDocument` holds the main story (body blocks) and
secondary stories (headers, footers, footnotes); a story is a list
of blocks — paragraphs and tables — and table cells hold block
lists of their own. Positions are (story, block index) paths in
document order.

The model is element-backed: every paragraph/table keeps its
``xml.etree`` element, so edits mutate the tree in place and the
writer re-serializes only the parts that were dirtied — unedited
parts leave the file byte-identical (see
:meth:`WordDocument.fidelity_report`).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import DocumentError
from .ooxml import (
    CP_NS,
    DC_NS,
    DCT_NS,
    R_NS,
    W_NS,
    children,
    find_all,
    first_child,
    local,
    parse_part,
    parse_xml_bounded,
    qname,
    read_package,
    serialize_xml,
    text_of,
    write_package,
)

__all__ = [
    "CoreProperties",
    "WordCell",
    "WordDocument",
    "WordParagraph",
    "WordRun",
    "WordStory",
    "WordStyle",
    "WordTable",
]

W = W_NS
R = R_NS


def _bool_child(rpr: ET.Element | None, name: str) -> bool:
    """A ``w:b``/``w:i``-style flag: present = on (unless val=0/false)."""
    if rpr is None:
        return False
    el = first_child(rpr, name)
    if el is None:
        return False
    val = el.get(qname(W, "val"))
    if val is None:
        return True
    return val not in ("0", "false", "off", "none")


def _val_child(element: ET.Element | None, attr: str) -> str | None:
    """The ``w:val`` (or named) attribute of an element, or None.

    Callers pass the element itself — ``_val_child(first_child(ppr,
    "pStyle"), "val")`` — never a parent plus a child name.
    """
    if element is None:
        return None
    return element.get(qname(W, attr))


def _int_child(element: ET.Element | None, attr: str) -> int | None:
    raw = _val_child(element, attr)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


@dataclass
class WordRun:
    """One run of text (``w:r``) with its formatting.

    ``kind`` is ``"text"`` or ``"drawing"`` (an inline image — the
    text is empty and ``drawing`` carries the embed id + extent).
    """

    text: str = ""
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strike: bool = False
    color: str | None = None
    size_half_points: int | None = None
    font: str | None = None
    highlight: str | None = None
    link_target: str | None = None
    kind: str = "text"
    drawing: dict[str, Any] | None = None
    element: ET.Element | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text, "bold": self.bold, "italic": self.italic,
            "underline": self.underline, "strike": self.strike,
            "color": self.color, "size_half_points": self.size_half_points,
            "font": self.font, "highlight": self.highlight,
            "link_target": self.link_target, "kind": self.kind,
        }


@dataclass
class WordParagraph:
    """One paragraph (``w:p``): runs + paragraph properties."""

    runs: list[WordRun] = field(default_factory=list)
    style_id: str = ""
    alignment: str | None = None
    numbering: dict[str, Any] | None = None
    spacing: dict[str, Any] = field(default_factory=dict)
    indent: dict[str, Any] = field(default_factory=dict)
    element: ET.Element | None = field(default=None, repr=False)
    _style_name: str = ""

    @property
    def text(self) -> str:
        return "".join(run.text for run in self.runs)

    @property
    def style_name(self) -> str:
        return self._style_name

    @property
    def heading_level(self) -> int:
        """1..6 for heading styles, 0 otherwise."""
        name = (self._style_name or "").strip().lower()
        if name.startswith("heading "):
            try:
                level = int(name.rsplit(" ", 1)[1])
            except ValueError:
                return 0
            return level if 1 <= level <= 6 else 0
        if name == "title":
            return 0
        return 0

    @property
    def is_heading(self) -> bool:
        return self.heading_level > 0

    def canonical_key(self) -> str:
        """Canonical serialization of the paragraph element.

        Both sides of a fidelity comparison go through the same
        serializer, so XML spelling differences (``<w:b/>`` vs
        ``<w:b />``) never count as a change.
        """
        if self.element is None:
            return ""
        return serialize_xml(self.element).decode("utf-8", "replace")

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text, "style_id": self.style_id,
            "style_name": self._style_name,
            "heading_level": self.heading_level,
            "alignment": self.alignment,
            "numbering": dict(self.numbering) if self.numbering else None,
            "runs": [r.to_dict() for r in self.runs],
        }


@dataclass
class WordCell:
    """One table cell: its own block list (paragraphs/tables)."""

    blocks: list[Any] = field(default_factory=list)
    grid_span: int = 1
    width: int | None = None
    element: ET.Element | None = field(default=None, repr=False)

    @property
    def text(self) -> str:
        parts = []
        for block in self.blocks:
            if isinstance(block, WordParagraph):
                parts.append(block.text)
            elif isinstance(block, WordTable):
                parts.append(block.plain_text())
        return "\n".join(p for p in parts if p)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text, "grid_span": self.grid_span,
            "blocks": [b.to_dict() for b in self.blocks],
        }


@dataclass
class WordTable:
    """One table (``w:tbl``): style, grid, rows of cells."""

    rows: list[list[WordCell]] = field(default_factory=list)
    style_id: str = ""
    grid_widths: list[int] = field(default_factory=list)
    element: ET.Element | None = field(default=None, repr=False)

    def cell_text(self, row: int, col: int) -> str:
        try:
            return self.rows[row][col].text
        except IndexError as exc:
            raise IndexError(
                f"cell ({row}, {col}) out of range for a "
                f"{len(self.rows)}x{self.n_cols} table") from exc

    @property
    def n_cols(self) -> int:
        return max((len(r) for r in self.rows), default=0)

    def rows_text(self) -> list[list[str]]:
        return [[cell.text for cell in row] for row in self.rows]

    def plain_text(self) -> str:
        return "\n".join(" | ".join(row) for row in self.rows_text())

    def to_dict(self) -> dict[str, Any]:
        return {
            "style_id": self.style_id,
            "grid_widths": list(self.grid_widths),
            "rows": [[c.to_dict() for c in row] for row in self.rows],
        }


@dataclass
class WordStyle:
    """One style definition from styles.xml."""

    style_id: str
    type: str = "paragraph"
    name: str = ""
    based_on: str = ""
    next_style: str = ""
    default: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"style_id": self.style_id, "type": self.type,
                "name": self.name, "based_on": self.based_on,
                "next": self.next_style, "default": self.default}


@dataclass
class WordStory:
    """A secondary story: a header, footer, or the footnote part."""

    name: str
    paragraphs: list[WordParagraph] = field(default_factory=list)
    part: str = ""

    def plain_text(self) -> str:
        return "\n".join(p.text for p in self.paragraphs)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "part": self.part,
                "paragraphs": [p.to_dict() for p in self.paragraphs]}


CoreProperties = dict[str, str]


class WordDocument:
    """An editable .docx document.

    Open with :meth:`from_bytes` / :meth:`from_path`; inspect
    :attr:`paragraphs`, :attr:`tables`, :attr:`headers`,
    :attr:`footers`, :attr:`styles`, :attr:`core_properties`; mutate
    through :class:`nomorals.documents.word.edit.WordEditor` (or the
    element-backed objects directly); save with :meth:`to_bytes`.
    """

    def __init__(self, package, doc_root: ET.Element,
                 original_bytes: bytes) -> None:
        self._package = package
        self._doc_root = doc_root
        self._original_bytes = original_bytes
        self._roots: dict[str, ET.Element] = {
            "word/document.xml": doc_root}
        self._touched_keys: set[str] = set()
        self.body: list[Any] = []
        self.styles: dict[str, WordStyle] = {}
        self.numbering: dict[str, Any] = {"abstracts": {}, "nums": {}}
        self.headers: list[WordStory] = []
        self.footers: list[WordStory] = []
        self.footnotes: list[str] = []
        self.sections: list[dict[str, Any]] = []
        self.core_properties: CoreProperties = {}
        self.images: list[dict[str, Any]] = []
        self._parse()

    # ── construction ──────────────────────────────────────────────
    @classmethod
    def from_bytes(cls, data: bytes) -> WordDocument:
        package = read_package(data)
        root = parse_part(package, "word/document.xml")
        if root is None or local(root.tag) != "document":
            raise DocumentError(
                "not a .docx file: word/document.xml has no "
                "w:document root", kind="parse")
        return cls(package, root, data)

    @classmethod
    def from_path(cls, path: str | Path) -> WordDocument:
        try:
            data = Path(path).read_bytes()
        except OSError as exc:
            raise DocumentError(f"cannot read {path}: {exc}",
                                kind="io") from exc
        return cls.from_bytes(data)

    # ── parsing ───────────────────────────────────────────────────
    def _parse(self) -> None:
        self._parse_styles()
        self._parse_numbering()
        self._parse_core_properties()
        self._parse_relationships()
        body = first_child(self._doc_root, "body")
        if body is None:
            raise DocumentError(
                "not a .docx file: word/document.xml has no w:body",
                kind="parse")
        for child in body:
            name = local(child.tag)
            if name == "p":
                self.body.append(self._parse_paragraph(child))
            elif name == "tbl":
                self.body.append(self._parse_table(child))
            elif name == "sectPr":
                self.sections.append(self._parse_sect_pr(child))
        self._parse_stories()
        self._parse_footnotes()

    def _parse_styles(self) -> None:
        root = parse_part(self._package, "word/styles.xml")
        if root is None:
            return
        self._roots["word/styles.xml"] = root
        for style_el in children(root, "style"):
            style_id = style_el.get(qname(W, "styleId"), "")
            if not style_id:
                continue
            name_el = first_child(style_el, "name")
            self.styles[style_id] = WordStyle(
                style_id=style_id,
                type=style_el.get(qname(W, "type"), "paragraph"),
                name=name_el.get(qname(W, "val"), "")
                if name_el is not None else "",
                based_on=_val_child(first_child(style_el, "basedOn"),
                                    "val") or "",
                next_style=_val_child(first_child(style_el, "next"),
                                      "val") or "",
                default=style_el.get(qname(W, "default")) == "1")

    def _parse_numbering(self) -> None:
        root = parse_part(self._package, "word/numbering.xml")
        if root is None:
            return
        self._roots["word/numbering.xml"] = root
        for abstract in children(root, "abstractNum"):
            aid = abstract.get(qname(W, "abstractNumId"))
            if aid is None:
                continue
            levels: dict[int, dict[str, str]] = {}
            for lvl in children(abstract, "lvl"):
                try:
                    ilvl = int(lvl.get(qname(W, "ilvl"), "0"))
                except ValueError:
                    continue
                levels[ilvl] = {
                    "numFmt": _val_child(first_child(lvl, "numFmt"),
                                         "val") or "bullet",
                    "lvlText": _val_child(first_child(lvl, "lvlText"),
                                          "val") or "",
                }
            self.numbering["abstracts"][aid] = levels
        for num in children(root, "num"):
            num_id = num.get(qname(W, "numId"))
            abstract_id = _val_child(first_child(num, "abstractNumId"),
                                     "val")
            if num_id is not None and abstract_id is not None:
                self.numbering["nums"][num_id] = abstract_id

    def numbering_kind(self, num_id: str, level: int = 0) -> str:
        """``bullet``/``decimal``/… for a numbering id + level."""
        abstract_id = self.numbering["nums"].get(str(num_id))
        if abstract_id is None:
            return "bullet"
        levels = self.numbering["abstracts"].get(abstract_id, {})
        fmt = levels.get(level, levels.get(0, {})).get("numFmt", "bullet")
        return fmt

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

    def _parse_relationships(self) -> None:
        root = parse_part(self._package, "word/_rels/document.xml.rels")
        self._doc_rels: dict[str, dict[str, str]] = {}
        if root is None:
            return
        for rel in root:
            rid = rel.get("Id", "")
            if rid:
                self._doc_rels[rid] = {
                    "type": rel.get("Type", ""),
                    "target": rel.get("Target", ""),
                }
        content_types = parse_part(self._package, "[Content_Types].xml")
        overrides: dict[str, str] = {}
        defaults: dict[str, str] = {}
        if content_types is not None:
            for child in content_types:
                name = local(child.tag)
                if name == "Override":
                    overrides[child.get("PartName", "")] = child.get(
                        "ContentType", "")
                elif name == "Default":
                    defaults[child.get("Extension", "")] = child.get(
                        "ContentType", "")
        for rid, rel in self._doc_rels.items():
            if not rel["type"].endswith("/image"):
                continue
            target = rel["target"].lstrip("/")
            part = target if target.startswith("word/") \
                else f"word/{target}"
            data = self._package.get(part)
            if data is None:
                continue
            ext = part.rsplit(".", 1)[-1].lower() if "." in part else ""
            self.images.append({
                "rel_id": rid, "part": part,
                "content_type": overrides.get("/" + part,
                                              defaults.get(ext, "")),
                "size": len(data),
            })

    def _parse_sect_pr(self, element: ET.Element) -> dict[str, Any]:
        out: dict[str, Any] = {"type": _val_child(
            first_child(element, "type"), "val") or "nextPage"}
        pg_sz = first_child(element, "pgSz")
        if pg_sz is not None:
            out["page_width"] = pg_sz.get(qname(W, "w"))
            out["page_height"] = pg_sz.get(qname(W, "h"))
        pg_mar = first_child(element, "pgMar")
        if pg_mar is not None:
            out["margins"] = {k: pg_mar.get(qname(W, k))
                              for k in ("top", "right", "bottom", "left",
                                        "header", "footer", "gutter")}
        for ref in children(element, "headerReference"):
            out.setdefault("headers", {})[ref.get(
                qname(W, "type"), "default")] = ref.get(
                    qname(R, "id"), "")
        for ref in children(element, "footerReference"):
            out.setdefault("footers", {})[ref.get(
                qname(W, "type"), "default")] = ref.get(
                    qname(R, "id"), "")
        return out

    def _parse_stories(self) -> None:
        wanted: dict[str, str] = {}
        for section in self.sections:
            for kind, refs in (("header", section.get("headers", {})),
                               ("footer", section.get("footers", {}))):
                for ref_type, rid in refs.items():
                    rel = self._doc_rels.get(rid)
                    if rel is None:
                        continue
                    target = rel["target"].lstrip("/")
                    part = target if target.startswith("word/") \
                        else f"word/{target}"
                    wanted[part] = f"{kind}:{ref_type}"
        for part, label in sorted(wanted.items()):
            root = parse_part(self._package, part)
            if root is None:
                continue
            story = WordStory(name=label, part=part)
            for child in root:
                if local(child.tag) == "p":
                    story.paragraphs.append(
                        self._parse_paragraph(child))
            if "header" in label:
                self.headers.append(story)
            else:
                self.footers.append(story)

    def _parse_footnotes(self) -> None:
        root = parse_part(self._package, "word/footnotes.xml")
        if root is None:
            return
        for note in children(root, "footnote"):
            note_id = note.get(qname(W, "id"))
            if note_id in ("0", "1"):
                continue  # separator / continuation-separator
            texts = [self._parse_paragraph(p).text
                     for p in children(note, "p")]
            joined = " ".join(t for t in texts if t)
            if joined:
                self.footnotes.append(joined)

    # ── block parsing ─────────────────────────────────────────────
    def _parse_paragraph(self, element: ET.Element) -> WordParagraph:
        ppr = first_child(element, "pPr")
        style_id = ""
        numbering: dict[str, Any] | None = None
        alignment: str | None = None
        spacing: dict[str, Any] = {}
        indent: dict[str, Any] = {}
        if ppr is not None:
            style_id = _val_child(first_child(ppr, "pStyle"), "val") or ""
            alignment = _val_child(first_child(ppr, "jc"), "val")
            num_pr = first_child(ppr, "numPr")
            if num_pr is not None:
                num_id = _val_child(first_child(num_pr, "numId"), "val")
                ilvl = _int_child(first_child(num_pr, "ilvl"), "val")
                if num_id is not None:
                    numbering = {
                        "num_id": num_id,
                        "level": ilvl if ilvl is not None else 0,
                        "kind": self.numbering_kind(
                            num_id, ilvl if ilvl is not None else 0),
                    }
            spacing_el = first_child(ppr, "spacing")
            if spacing_el is not None:
                spacing = {k: v for k, v in spacing_el.attrib.items()}
            ind_el = first_child(ppr, "ind")
            if ind_el is not None:
                indent = {local(k): v
                          for k, v in ind_el.attrib.items()}
        style = self.styles.get(style_id)
        para = WordParagraph(
            runs=[], style_id=style_id, alignment=alignment,
            numbering=numbering, spacing=spacing, indent=indent,
            element=element,
            _style_name=style.name if style else "")
        for child in element:
            name = local(child.tag)
            if name == "r":
                para.runs.append(self._parse_run(child))
            elif name == "hyperlink":
                target = self._hyperlink_target(child)
                for run_el in children(child, "r"):
                    run = self._parse_run(run_el)
                    run.link_target = target
                    para.runs.append(run)
            elif name == "smartTag":
                for run_el in find_all(child, "r"):
                    para.runs.append(self._parse_run(run_el))
        return para

    def _hyperlink_target(self, element: ET.Element) -> str | None:
        rid = element.get(qname(R, "id"))
        if rid:
            rel = self._doc_rels.get(rid)
            if rel:
                return rel["target"]
        anchor = element.get(qname(W, "anchor"))
        return f"#{anchor}" if anchor else None

    def _parse_run(self, element: ET.Element) -> WordRun:
        rpr = first_child(element, "rPr")
        drawing = first_child(element, "drawing")
        if drawing is not None:
            info: dict[str, Any] = {"kind": "drawing"}
            blip = next((e for e in drawing.iter()
                         if local(e.tag) == "blip"), None)
            if blip is not None:
                info["embed"] = blip.get(qname(R, "embed"))
            extent = next((e for e in drawing.iter()
                           if local(e.tag) == "extent"), None)
            if extent is not None:
                info["cx"] = extent.get("cx")
                info["cy"] = extent.get("cy")
            return WordRun(kind="drawing", drawing=info, element=element)
        text = text_of(element)
        return WordRun(
            text=text,
            bold=_bool_child(rpr, "b"),
            italic=_bool_child(rpr, "i"),
            underline=first_child(rpr, "u") is not None,
            strike=_bool_child(rpr, "strike"),
            color=_val_child(first_child(rpr, "color"), "val"),
            size_half_points=_int_child(first_child(rpr, "sz"), "val"),
            font=_val_child(first_child(rpr, "rFonts"), "ascii"),
            highlight=_val_child(first_child(rpr, "highlight"), "val"),
            element=element)

    def _parse_table(self, element: ET.Element) -> WordTable:
        tbl_pr = first_child(element, "tblPr")
        style_id = _val_child(first_child(tbl_pr, "tblStyle"),
                              "val") if tbl_pr is not None else ""
        grid_widths: list[int] = []
        grid = first_child(tbl_pr, "tblGrid") if tbl_pr is not None \
            else None
        if grid is not None:
            for col in children(grid, "gridCol"):
                try:
                    grid_widths.append(int(col.get(qname(W, "w"), "0")))
                except ValueError:
                    grid_widths.append(0)
        rows: list[list[WordCell]] = []
        for tr in children(element, "tr"):
            cells: list[WordCell] = []
            for tc in children(tr, "tc"):
                tc_pr = first_child(tc, "tcPr")
                span = _int_child(first_child(tc_pr, "gridSpan"),
                                  "val") if tc_pr is not None else None
                blocks: list[Any] = []
                for child in tc:
                    name = local(child.tag)
                    if name == "p":
                        blocks.append(self._parse_paragraph(child))
                    elif name == "tbl":
                        blocks.append(self._parse_table(child))
                cells.append(WordCell(
                    blocks=blocks,
                    grid_span=span if span is not None else 1,
                    element=tc))
            rows.append(cells)
        return WordTable(rows=rows, style_id=style_id or "",
                         grid_widths=grid_widths, element=element)

    # ── views ─────────────────────────────────────────────────────
    @property
    def paragraphs(self) -> list[WordParagraph]:
        return [b for b in self.body if isinstance(b, WordParagraph)]

    @property
    def tables(self) -> list[WordTable]:
        return [b for b in self.body if isinstance(b, WordTable)]

    @property
    def part_names(self) -> list[str]:
        return self._package.part_names()

    def body_blocks(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for i, block in enumerate(self.body):
            if isinstance(block, WordParagraph):
                out.append({"kind": "paragraph", "index": i,
                            "paragraph": block})
            else:
                out.append({"kind": "table", "index": i, "table": block})
        return out

    def plain_text(self) -> str:
        parts: list[str] = []
        for block in self.body:
            if isinstance(block, WordParagraph):
                if block.text:
                    parts.append(block.text)
            else:
                text = block.plain_text()
                if text:
                    parts.append(text)
        return "\n".join(parts)

    def outline(self) -> list[dict[str, Any]]:
        return [{"level": p.heading_level, "text": p.text}
                for p in self.paragraphs if p.is_heading]

    def style_name(self, style_id: str) -> str:
        style = self.styles.get(style_id)
        return style.name if style else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "paragraphs": [p.to_dict() for p in self.paragraphs],
            "tables": [t.to_dict() for t in self.tables],
            "headers": [s.to_dict() for s in self.headers],
            "footers": [s.to_dict() for s in self.footers],
            "footnotes": list(self.footnotes),
            "sections": [dict(s) for s in self.sections],
            "styles": [s.to_dict() for s in self.styles.values()],
            "images": [dict(i) for i in self.images],
            "core_properties": dict(self.core_properties),
            "part_names": self.part_names,
        }

    # ── writing ───────────────────────────────────────────────────
    def _note_touched(self, paragraph: WordParagraph) -> None:
        """Record a paragraph's pre-edit canonical form.

        The fidelity report subtracts these from the vanished set —
        an edited paragraph is expected to change; an untouched one
        is not.
        """
        key = paragraph.canonical_key()
        if key:
            self._touched_keys.add(key)

    def mark_dirty(self, part: str) -> None:
        self._package.mark_dirty(part)

    def set_root(self, part: str, root: ET.Element) -> None:
        self._roots[part] = root
        self.mark_dirty(part)

    def root_for(self, part: str) -> ET.Element | None:
        return self._roots.get(part)

    def to_bytes(self) -> bytes:
        """Serialize back to .docx bytes.

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
        byte-identical. Paragraph-level: every original paragraph
        that the editor did not intentionally touch must still be
        present (canonical multiset comparison, so XML spelling
        differences never count).
        """
        original_pkg = read_package(original_bytes)
        orig_root = parse_xml_bounded(
            original_pkg.parts["word/document.xml"], "word/document.xml")
        orig_body = first_child(orig_root, "body")
        orig_keys = Counter()
        if orig_body is not None:
            for child in orig_body:
                if local(child.tag) == "p":
                    orig_keys[serialize_xml(child).decode(
                        "utf-8", "replace")] += 1
        saved_keys = Counter()
        body = first_child(self._doc_root, "body")
        if body is not None:
            for child in body:
                if local(child.tag) == "p":
                    saved_keys[serialize_xml(child).decode(
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
            "untouched_paragraphs_changed": untouched_changed,
            "paragraphs_original": sum(orig_keys.values()),
            "paragraphs_saved": sum(saved_keys.values()),
        }
