"""WordCraft editor — mutate a :class:`WordDocument` and save it back.

Follows the upstream ``crates/doc`` edit design (``edit.rs``): edits
address blocks by document order and copy only what they touch.
Every mutation records the paragraph's pre-edit canonical form so
:meth:`WordDocument.fidelity_report` can tell an intentional change
from collateral damage, and marks only the parts it dirtied — the
writer copies every other part byte-identical.

Supported edits: replace text (whole-document or first hit),
restyle a run, restyle a paragraph, add paragraphs / headings /
list items / tables, set core metadata, remove a paragraph, and
create a fresh document. Headers, footers, and footnotes are
read-only inventory in this rung — they are preserved byte-identical
on save, never edited.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from ..errors import DocumentError
from .model import WordDocument, WordParagraph
from .ooxml import (
    CP_NS,
    CT_NS,
    DC_NS,
    DCT_NS,
    PR_NS,
    R_NS,
    W_NS,
    XML_NS,
    first_child,
    local,
    parse_xml_bounded,
    qname,
    serialize_xml,
    write_package,
)

__all__ = ["WordEditor"]

W = W_NS
R = R_NS

#: Known built-in styles the editor can define on demand (a document
#: without styles.xml still gets a real Heading 1, not an error).
_BUILTIN_STYLES: dict[str, dict[str, str]] = {
    "Normal": {"type": "paragraph", "name": "Normal", "default": "1"},
    "Title": {"type": "paragraph", "name": "Title"},
    "Heading1": {"type": "paragraph", "name": "heading 1"},
    "Heading2": {"type": "paragraph", "name": "heading 2"},
    "Heading3": {"type": "paragraph", "name": "heading 3"},
    "Heading4": {"type": "paragraph", "name": "heading 4"},
    "Heading5": {"type": "paragraph", "name": "heading 5"},
    "Heading6": {"type": "paragraph", "name": "heading 6"},
    "ListParagraph": {"type": "paragraph", "name": "List Paragraph"},
    "ListBullet": {"type": "paragraph", "name": "List Bullet"},
    "ListNumber": {"type": "paragraph", "name": "List Number"},
    "TableGrid": {"type": "table", "name": "Table Grid"},
}

#: rPr child order (ECMA-376 fragment) — new formatting elements are
#: inserted at their schema position so existing siblings keep order.
_RPR_ORDER = [
    "rStyle", "rFonts", "b", "bCs", "i", "iCs", "caps", "smallCaps",
    "strike", "dstrike", "outline", "shadow", "emboss", "imprint",
    "noProof", "snapToGrid", "vanish", "webHidden", "color", "spacing",
    "w", "kern", "position", "sz", "szCs", "highlight", "u", "effect",
    "bdr", "shd", "fitText", "vertAlign", "rtl", "cs", "em", "lang",
    "eastAsia", "specVanish", "oMath",
]

_NUMBERING_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    f'<w:numbering xmlns:w="{W}">'
    '<w:abstractNum w:abstractNumId="0">'
    '<w:multiLevelType w:val="hybridMultilevel"/>'
    '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/>'
    '<w:lvlText w:val="•"/><w:lvlJc w:val="left"/>'
    '<w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr></w:lvl>'
    '</w:abstractNum>'
    '<w:abstractNum w:abstractNumId="1">'
    '<w:multiLevelType w:val="hybridMultilevel"/>'
    '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/>'
    '<w:lvlText w:val="%1."/><w:lvlJc w:val="left"/>'
    '<w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr></w:lvl>'
    '</w:abstractNum>'
    '<w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>'
    '<w:num w:numId="2"><w:abstractNumId w:val="1"/></w:num>'
    "</w:numbering>")

_STYLES_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    f'<w:styles xmlns:w="{W}">'
    '<w:docDefaults><w:rPrDefault><w:rPr>'
    '<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/>'
    '<w:sz w:val="22"/><w:szCs w:val="22"/>'
    '<w:lang w:val="en-US"/></w:rPr></w:rPrDefault>'
    '<w:pPrDefault><w:pPr><w:spacing w:after="160" w:line="259" '
    'w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>'
    '<w:style w:type="paragraph" w:default="1" w:styleId="Normal">'
    '<w:name w:val="Normal"/><w:qFormat/></w:style>'
    '<w:style w:type="paragraph" w:styleId="Title">'
    '<w:name w:val="Title"/><w:basedOn w:val="Normal"/>'
    '<w:next w:val="Normal"/><w:qFormat/>'
    '<w:rPr><w:b/><w:sz w:val="56"/><w:szCs w:val="56"/></w:rPr></w:style>'
    '<w:style w:type="paragraph" w:styleId="Heading1">'
    '<w:name w:val="heading 1"/><w:basedOn w:val="Normal"/>'
    '<w:next w:val="Normal"/><w:qFormat/>'
    '<w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>'
    '<w:style w:type="paragraph" w:styleId="Heading2">'
    '<w:name w:val="heading 2"/><w:basedOn w:val="Normal"/>'
    '<w:next w:val="Normal"/><w:qFormat/>'
    '<w:rPr><w:b/><w:sz w:val="26"/></w:rPr></w:style>'
    '<w:style w:type="paragraph" w:styleId="Heading3">'
    '<w:name w:val="heading 3"/><w:basedOn w:val="Normal"/>'
    '<w:next w:val="Normal"/><w:qFormat/>'
    '<w:rPr><w:b/><w:sz w:val="24"/></w:rPr></w:style>'
    '<w:style w:type="paragraph" w:styleId="ListParagraph">'
    '<w:name w:val="List Paragraph"/><w:basedOn w:val="Normal"/>'
    '<w:qFormat/></w:style>'
    '<w:style w:type="table" w:default="1" w:styleId="TableGrid">'
    '<w:name w:val="Table Grid"/></w:style>'
    "</w:styles>")


def _insert_in_order(parent: ET.Element, new: ET.Element,
                     order: list[str]) -> None:
    """Insert ``new`` among ``parent``'s children at its schema slot."""
    name = local(new.tag)
    try:
        slot = order.index(name)
    except ValueError:
        parent.append(new)
        return
    for i, child in enumerate(parent):
        child_name = local(child.tag)
        if child_name == name:
            parent.remove(child)
            parent.insert(i, new)
            return
        if child_name in order and order.index(child_name) > slot:
            parent.insert(i, new)
            return
    parent.append(new)


def _ensure_child(parent: ET.Element, name: str) -> ET.Element:
    child = first_child(parent, name)
    if child is None:
        child = ET.Element(qname(W, name))
        parent.append(child)
    return child


class WordEditor:
    """Mutate a :class:`WordDocument`; :meth:`save` writes .docx bytes."""

    def __init__(self, document: WordDocument) -> None:
        self.document = document
        self.edits = 0

    # ── construction ──────────────────────────────────────────────
    @classmethod
    def from_bytes(cls, data: bytes) -> WordEditor:
        return cls(WordDocument.from_bytes(data))

    @classmethod
    def from_document(cls, document: WordDocument) -> WordEditor:
        return cls(document)

    @classmethod
    def create(cls, *, title: str = "", author: str = "") -> WordEditor:
        """A fresh blank document (minimal package, real styles)."""
        core = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            f'<cp:coreProperties xmlns:cp="{CP_NS}" '
            f'xmlns:dc="{DC_NS}" xmlns:dcterms="{DCT_NS}">'
            f"<dc:title>{title}</dc:title>"
            f"<dc:creator>{author}</dc:creator>"
            "</cp:coreProperties>")
        parts = {
            "[Content_Types].xml": (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                f'<Types xmlns="{CT_NS}">'
                '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                '<Default Extension="xml" ContentType="application/xml"/>'
                '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
                '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
                '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
                "</Types>").encode(),
            "_rels/.rels": (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                f'<Relationships xmlns="{PR_NS}">'
                '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
                "</Relationships>").encode(),
            "word/document.xml": (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                f'<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body>'
                "<w:sectPr>"
                '<w:pgSz w:w="12240" w:h="15840"/>'
                '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>'
                "</w:sectPr></w:body></w:document>").encode(),
            "word/styles.xml": _STYLES_XML.encode(),
            "word/_rels/document.xml.rels": (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                f'<Relationships xmlns="{PR_NS}">'
                f'<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
                "</Relationships>").encode(),
            "docProps/core.xml": core.encode(),
            "docProps/app.xml": (
                b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                b'<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
                b"<Application>wordcraft</Application></Properties>"),
        }
        return cls(WordDocument.from_bytes(write_package(parts)))

    # ── helpers ───────────────────────────────────────────────────
    def _body(self) -> ET.Element:
        body = first_child(self.document._doc_root, "body")
        if body is None:  # pragma: no cover - the model guarantees a body
            raise DocumentError("document has no w:body", kind="parse")
        return body

    def _refresh_paragraph(self, index: int) -> WordParagraph:
        para = self.document.paragraphs[index]
        fresh = self.document._parse_paragraph(para.element)
        for i, block in enumerate(self.document.body):
            if block is para:
                self.document.body[i] = fresh
                return fresh
        return fresh

    def _paragraph(self, index: int) -> WordParagraph:
        paras = self.document.paragraphs
        if not 0 <= index < len(paras):
            raise DocumentError(
                f"paragraph index {index} out of range "
                f"(document has {len(paras)})", kind="invalid_argument")
        return paras[index]

    def _append_body_child(self, element: ET.Element) -> None:
        body = self._body()
        sect = first_child(body, "sectPr")
        if sect is not None:
            body.insert(list(body).index(sect), element)
        else:
            body.append(element)
        self.document.mark_dirty("word/document.xml")

    def _run_elements(self, para_el: ET.Element) -> list[ET.Element]:
        out: list[ET.Element] = []
        for child in para_el:
            name = local(child.tag)
            if name == "r":
                out.append(child)
            elif name in ("hyperlink", "smartTag"):
                out.extend(c for c in child.iter()
                           if local(c.tag) == "r")
        return out

    # ── text replacement ──────────────────────────────────────────
    def replace_text(self, old: str, new: str,
                     *, first_only: bool = False) -> int:
        """Replace ``old`` with ``new`` in the body text.

        A phrase inside one run is replaced in place; a phrase
        spanning runs is merged into the first text run (its
        formatting wins) and the other text runs are emptied.
        Returns the number of replacements made.
        """
        if not old:
            raise DocumentError("replace_text needs non-empty text",
                                kind="invalid_argument")
        count = 0
        for index, para in enumerate(list(self.document.paragraphs)):
            hits = para.text.count(old)
            if not hits:
                continue
            if first_only and count >= 1:
                break
            self.document._note_touched(para)
            take = 1 if first_only else hits
            self._replace_in_paragraph(para, old, new, count=take)
            self._refresh_paragraph(index)
            count += take
            if first_only:
                break
        if count == 0:
            raise DocumentError(f"text {old!r} not found in document",
                                kind="invalid_argument")
        self.document.mark_dirty("word/document.xml")
        self.edits += 1
        return count

    def _replace_in_paragraph(self, para: WordParagraph, old: str,
                              new: str, *, count: int) -> None:
        assert para.element is not None
        runs = self._run_elements(para.element)
        text_runs = [r for r in runs
                     if first_child(r, "drawing") is None]
        for run_el in text_runs:
            parts = [t for t in run_el.iter() if local(t.tag) == "t"]
            joined = "".join(t.text or "" for t in parts)
            if old in joined:
                # single-run hit: replace inside this run's w:t nodes
                remaining = count
                for node in parts:
                    if remaining <= 0:
                        break
                    node_hits = (node.text or "").count(old)
                    if node_hits:
                        take = min(node_hits, remaining)
                        node.text = (node.text or "").replace(
                            old, new, take)
                        remaining -= take
                if remaining < count:
                    return
                # the phrase spans this run's w:t nodes — fall
                # through to the merge below
                break
        # cross-run hit: merge the paragraph text into the first run
        merged = para.text.replace(old, new, count)
        if not text_runs:
            return
        first = text_runs[0]
        t_nodes = [t for t in first.iter() if local(t.tag) == "t"]
        if t_nodes:
            t_nodes[0].text = merged
            for node in t_nodes[1:]:
                node.text = ""
        else:
            t_el = ET.Element(qname(W, "t"))
            t_el.set(qname(
                "http://www.w3.org/XML/1998/namespace", "space"),
                "preserve")
            t_el.text = merged
            first.append(t_el)
        for run_el in text_runs[1:]:
            for node in run_el.iter():
                if local(node.tag) == "t":
                    node.text = ""

    # ── formatting ────────────────────────────────────────────────
    def set_run_format(self, para_index: int, run_index: int, *,
                       bold: bool | None = None,
                       italic: bool | None = None,
                       underline: bool | None = None,
                       strike: bool | None = None,
                       color: str | None = None,
                       size_half_points: int | None = None,
                       font: str | None = None,
                       highlight: str | None = None) -> None:
        para = self._paragraph(para_index)
        runs = [r for r in para.runs if r.kind == "text"]
        if not 0 <= run_index < len(runs):
            raise DocumentError(
                f"run index {run_index} out of range "
                f"(paragraph has {len(runs)} text runs)",
                kind="invalid_argument")
        run = runs[run_index]
        element = run.element
        if element is None:  # pragma: no cover - parsed runs have one
            raise DocumentError("run has no element", kind="parse")
        self.document._note_touched(para)
        rpr = first_child(element, "rPr")
        if rpr is None:
            rpr = ET.Element(qname(W, "rPr"))
            element.insert(0, rpr)
        self._set_flag(rpr, "b", bold)
        self._set_flag(rpr, "i", italic)
        self._set_flag(rpr, "strike", strike)
        if underline is not None:
            if underline:
                u_el = first_child(rpr, "u")
                if u_el is None:
                    u_el = ET.Element(qname(W, "u"))
                    _insert_in_order(rpr, u_el, _RPR_ORDER)
                u_el.set(qname(W, "val"), "single")
            else:
                u_el = first_child(rpr, "u")
                if u_el is not None:
                    rpr.remove(u_el)
        if color is not None:
            self._set_val(rpr, "color", color)
        if size_half_points is not None:
            if size_half_points <= 0:
                raise DocumentError(
                    f"size_half_points must be positive, got "
                    f"{size_half_points}", kind="invalid_argument")
            self._set_val(rpr, "sz", str(size_half_points))
        if font is not None:
            fonts_el = first_child(rpr, "rFonts")
            if fonts_el is None:
                fonts_el = ET.Element(qname(W, "rFonts"))
                _insert_in_order(rpr, fonts_el, _RPR_ORDER)
            for attr in ("ascii", "hAnsi", "cs"):
                fonts_el.set(qname(W, attr), font)
        if highlight is not None:
            self._set_val(rpr, "highlight", highlight)
        self._refresh_paragraph(para_index)
        self.document.mark_dirty("word/document.xml")
        self.edits += 1

    @staticmethod
    def _set_flag(rpr: ET.Element, name: str,
                  value: bool | None) -> None:
        if value is None:
            return
        el = first_child(rpr, name)
        if value:
            if el is None:
                el = ET.Element(qname(W, name))
                _insert_in_order(rpr, el, _RPR_ORDER)
            el.attrib.pop(qname(W, "val"), None)
        elif el is not None:
            rpr.remove(el)

    @staticmethod
    def _set_val(rpr: ET.Element, name: str, value: str) -> None:
        el = first_child(rpr, name)
        if el is None:
            el = ET.Element(qname(W, name))
            _insert_in_order(rpr, el, _RPR_ORDER)
        el.set(qname(W, "val"), value)

    def set_paragraph_style(self, index: int, style_id: str) -> None:
        para = self._paragraph(index)
        self._ensure_style(style_id)
        self.document._note_touched(para)
        assert para.element is not None
        ppr = first_child(para.element, "pPr")
        if ppr is None:
            ppr = ET.Element(qname(W, "pPr"))
            para.element.insert(0, ppr)
        style_el = first_child(ppr, "pStyle")
        if style_el is None:
            style_el = ET.Element(qname(W, "pStyle"))
            ppr.insert(0, style_el)
        style_el.set(qname(W, "val"), style_id)
        self._refresh_paragraph(index)
        self.document.mark_dirty("word/document.xml")
        self.edits += 1

    # ── adding blocks ─────────────────────────────────────────────
    def add_paragraph(self, text: str, style: str | None = None) -> int:
        if style is not None:
            self._ensure_style(style)
        para_el = self._new_para_el(text, style_id=style or "")
        self._append_body_child(para_el)
        self.document.body.append(
            self.document._parse_paragraph(para_el))
        self.edits += 1
        return len(self.document.paragraphs) - 1

    def add_heading(self, text: str, level: int = 1) -> int:
        if not 1 <= level <= 6:
            raise DocumentError(
                f"heading level must be 1..6, got {level}",
                kind="invalid_argument")
        return self.add_paragraph(text, style=f"Heading{level}")

    def add_list_item(self, text: str, *, ordered: bool = False) -> int:
        num_id = self._ensure_numbering(ordered)
        para_el = self._new_para_el(
            text, style_id="ListParagraph",
            numbering=(num_id, 0))
        self._append_body_child(para_el)
        self.document.body.append(
            self.document._parse_paragraph(para_el))
        self.edits += 1
        return len(self.document.paragraphs) - 1

    def add_table(self, headers: list[str],
                  rows: list[list[str]]) -> int:
        if not headers:
            raise DocumentError("add_table needs a header row",
                                kind="invalid_argument")
        self._ensure_style("TableGrid")
        tbl_el = ET.Element(qname(W, "tbl"))
        tbl_pr = ET.SubElement(tbl_el, qname(W, "tblPr"))
        style_el = ET.SubElement(tbl_pr, qname(W, "tblStyle"))
        style_el.set(qname(W, "val"), "TableGrid")
        grid = ET.SubElement(tbl_pr, qname(W, "tblGrid"))
        width = max(1, 7200 // len(headers))
        for _ in headers:
            col = ET.SubElement(grid, qname(W, "gridCol"))
            col.set(qname(W, "w"), str(width))
        for row in [headers] + list(rows):
            tr = ET.SubElement(tbl_el, qname(W, "tr"))
            for cell_text in row:
                tc = ET.SubElement(tr, qname(W, "tc"))
                tc_pr = ET.SubElement(tc, qname(W, "tcPr"))
                tc_w = ET.SubElement(tc_pr, qname(W, "tcW"))
                tc_w.set(qname(W, "w"), str(width))
                tc_w.set(qname(W, "type"), "dxa")
                tc.append(self._new_para_el(cell_text))
        self._append_body_child(tbl_el)
        self.document.body.append(self.document._parse_table(tbl_el))
        self.edits += 1
        return len(self.document.tables) - 1

    def remove_paragraph(self, index: int) -> None:
        para = self._paragraph(index)
        self.document._note_touched(para)
        assert para.element is not None
        body = self._body()
        body.remove(para.element)
        self.document.body = [b for b in self.document.body
                              if b is not para]
        self.document.mark_dirty("word/document.xml")
        self.edits += 1

    # ── metadata ──────────────────────────────────────────────────
    def set_metadata(self, **fields: str) -> None:
        allowed = {"title", "author", "description", "subject",
                   "keywords", "last_modified_by"}
        unknown = sorted(set(fields) - allowed)
        if unknown:
            raise DocumentError(
                f"unknown metadata field(s): {', '.join(unknown)} "
                f"(use one of {', '.join(sorted(allowed))})",
                kind="invalid_argument")
        if not fields:
            raise DocumentError("set_metadata needs a field to set",
                                kind="invalid_argument")
        root = self.document.root_for("docProps/core.xml")
        if root is None:
            root = ET.fromstring(
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                f'<cp:coreProperties xmlns:cp="{CP_NS}" '
                f'xmlns:dc="{DC_NS}" xmlns:dcterms="{DCT_NS}"/>')
            self.document.set_root("docProps/core.xml", root)
            self._ensure_package_part(
                "docProps/core.xml",
                "application/vnd.openxmlformats-package.core-properties+xml",
                None)
        tag_for = {
            "title": qname(DC_NS, "title"),
            "author": qname(DC_NS, "creator"),
            "description": qname(DC_NS, "description"),
            "subject": qname(DC_NS, "subject"),
            "keywords": qname(CP_NS, "keywords"),
            "last_modified_by": qname(CP_NS, "lastModifiedBy"),
        }
        for key, value in fields.items():
            tag = tag_for[key]
            el = root.find(tag)
            if el is None:
                el = ET.Element(tag)
                root.append(el)
            el.text = value
            self.document.core_properties[key] = value
        self.document.set_root("docProps/core.xml", root)
        self.edits += 1

    # ── save / summary ────────────────────────────────────────────
    def save(self) -> bytes:
        return self.document.to_bytes()

    def summary(self) -> dict[str, Any]:
        return {
            "paragraphs": len(self.document.paragraphs),
            "tables": len(self.document.tables),
            "headers": len(self.document.headers),
            "footers": len(self.document.footers),
            "images": len(self.document.images),
            "edits": self.edits,
            "title": self.document.core_properties.get("title", ""),
            "author": self.document.core_properties.get("author", ""),
        }

    # ── part plumbing ─────────────────────────────────────────────
    def _ensure_package_part(self, part: str, content_type: str,
                             rel_type: str | None) -> None:
        """Register a new part in [Content_Types].xml (+ rels)."""
        types_raw = self.document._package.get("[Content_Types].xml")
        if types_raw is not None:
            root = parse_xml_bounded(types_raw, "[Content_Types].xml")
            part_attr = "/" + part
            if not any(c.get("PartName") == part_attr
                       for c in root
                       if local(c.tag) == "Override"):
                override = ET.Element(
                    qname(CT_NS, "Override"))
                override.set("PartName", part_attr)
                override.set("ContentType", content_type)
                root.append(override)
                self.document._package.set_part(
                    "[Content_Types].xml", serialize_xml(root))
        if rel_type is not None:
            rels_raw = self.document._package.get(
                "word/_rels/document.xml.rels")
            if rels_raw is None:
                rels_root = ET.Element(qname(PR_NS, "Relationships"))
                self.document._package.set_part(
                    "word/_rels/document.xml.rels",
                    serialize_xml(rels_root))
                rels_raw = self.document._package.get(
                    "word/_rels/document.xml.rels")
            assert rels_raw is not None
            rels = parse_xml_bounded(rels_raw,
                                     "word/_rels/document.xml.rels")
            target = part[len("word/"):] if part.startswith("word/") \
                else part
            if not any(c.get("Target") == target for c in rels):
                used = {c.get("Id", "") for c in rels}
                n = 1
                while f"rId{n}" in used:
                    n += 1
                rel = ET.Element(qname(PR_NS, "Relationship"))
                rel.set("Id", f"rId{n}")
                rel.set("Type", rel_type)
                rel.set("Target", target)
                rels.append(rel)
                self.document._package.set_part(
                    "word/_rels/document.xml.rels", serialize_xml(rels))

    def _ensure_numbering(self, ordered: bool) -> str:
        """Make sure numbering.xml exists; return the numId to use."""
        if "word/numbering.xml" not in self.document._package:
            root = parse_xml_bounded(_NUMBERING_XML.encode(),
                                     "word/numbering.xml")
            self.document._package.set_part(
                "word/numbering.xml", _NUMBERING_XML.encode())
            self.document._roots["word/numbering.xml"] = root
            self.document.numbering = {
                "abstracts": {
                    "0": {0: {"numFmt": "bullet", "lvlText": "•"}},
                    "1": {0: {"numFmt": "decimal", "lvlText": "%1."}},
                },
                "nums": {"1": "0", "2": "1"},
            }
            self._ensure_package_part(
                "word/numbering.xml",
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.numbering+xml",
                "http://schemas.openxmlformats.org/officeDocument/2006/"
                "relationships/numbering")
        return "2" if ordered else "1"

    def _ensure_style(self, style_id: str) -> None:
        """Define a built-in style on demand; unknown ids are errors."""
        if style_id in self.document.styles:
            return
        spec = _BUILTIN_STYLES.get(style_id)
        if spec is None:
            raise DocumentError(
                f"unknown style {style_id!r} (known: "
                f"{', '.join(sorted(self.document.styles)) or 'none'})",
                kind="invalid_argument")
        if style_id in ("ListBullet", "ListNumber"):
            self._ensure_numbering(style_id == "ListNumber")
        root = self.document.root_for("word/styles.xml")
        if root is None:
            root = parse_xml_bounded(_STYLES_XML.encode(),
                                     "word/styles.xml")
            self.document._package.set_part(
                "word/styles.xml", _STYLES_XML.encode())
            self.document._roots["word/styles.xml"] = root
            self.document._parse_styles()
            self._ensure_package_part(
                "word/styles.xml",
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.styles+xml",
                "http://schemas.openxmlformats.org/officeDocument/2006/"
                "relationships/styles")
            if style_id in self.document.styles:
                return
        style_el = ET.Element(qname(W, "style"))
        style_el.set(qname(W, "type"), spec["type"])
        style_el.set(qname(W, "styleId"), style_id)
        name_el = ET.SubElement(style_el, qname(W, "name"))
        name_el.set(qname(W, "val"), spec["name"])
        based = ET.SubElement(style_el, qname(W, "basedOn"))
        based.set(qname(W, "val"), "Normal")
        root.append(style_el)
        self.document.set_root("word/styles.xml", root)
        self.document._parse_styles()
        self._ensure_package_part(
            "word/styles.xml",
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.styles+xml",
            "http://schemas.openxmlformats.org/officeDocument/2006/"
            "relationships/styles")

    # ── element builders ──────────────────────────────────────────
    def _new_run_el(self, text: str,
                    fmt: dict[str, Any] | None = None) -> ET.Element:
        run_el = ET.Element(qname(W, "r"))
        fmt = fmt or {}
        if fmt:
            rpr = ET.SubElement(run_el, qname(W, "rPr"))
            if fmt.get("bold"):
                ET.SubElement(rpr, qname(W, "b"))
            if fmt.get("italic"):
                ET.SubElement(rpr, qname(W, "i"))
            if fmt.get("color"):
                color = ET.SubElement(rpr, qname(W, "color"))
                color.set(qname(W, "val"), fmt["color"])
            if fmt.get("size_half_points"):
                size = ET.SubElement(rpr, qname(W, "sz"))
                size.set(qname(W, "val"), str(fmt["size_half_points"]))
        t_el = ET.SubElement(run_el, qname(W, "t"))
        t_el.set(qname(XML_NS, "space"), "preserve")
        t_el.text = text
        return run_el

    def _new_para_el(self, text: str, *, style_id: str = "",
                     numbering: tuple[str, int] | None = None,
                     fmt: dict[str, Any] | None = None) -> ET.Element:
        para_el = ET.Element(qname(W, "p"))
        ppr_parts: list[ET.Element] = []
        if style_id:
            style_el = ET.Element(qname(W, "pStyle"))
            style_el.set(qname(W, "val"), style_id)
            ppr_parts.append(style_el)
        if numbering is not None:
            num_id, ilvl = numbering
            num_pr = ET.Element(qname(W, "numPr"))
            ilvl_el = ET.SubElement(num_pr, qname(W, "ilvl"))
            ilvl_el.set(qname(W, "val"), str(ilvl))
            num_el = ET.SubElement(num_pr, qname(W, "numId"))
            num_el.set(qname(W, "val"), str(num_id))
            ppr_parts.append(num_pr)
        if ppr_parts:
            ppr = ET.Element(qname(W, "pPr"))
            for part in ppr_parts:
                ppr.append(part)
            para_el.append(ppr)
        para_el.append(self._new_run_el(text, fmt))
        return para_el
