"""Writers: :class:`Document` -> XLSX / PPTX bytes (stdlib only).

Wave 2 (Echelon documents port): Excel and PowerPoint were read-only
— the plan's owner-visible choice (b) — and the owner's standing
doctrine is full capability with no artificial limitations, so the
writers exist.  They are built the way the readers and ``to_epub``
are: ``zipfile`` + hand-built XML parts, no third-party package, so
they work offline on a phone like everything else in this package.

Fidelity contract (measured, not asserted — see
:mod:`nomorals.documents.corpus` and ``documents/PARITY.md``):

* XLSX: one worksheet per table (plus a ``Document`` sheet carrying
  title/sections when prose exists); cell values are typed — integers,
  decimals, booleans and ISO dates become real spreadsheet cells,
  everything else rides the shared-strings table verbatim.
* PPTX: a title slide, one slide per section (heading as the title
  placeholder, prose lines as body paragraphs), one slide per table
  (a real DrawingML table).  Text is real text — never pictures.
"""

from __future__ import annotations

import html as html_module
import io
import re
import zipfile
from datetime import date, datetime

from .errors import DocumentError
from .model import Document, Table

__all__ = ["to_xlsx", "to_pptx"]

_esc = html_module.escape

_XML_DECL = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'

_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_CP_NS = ("http://schemas.openxmlformats.org/package/2006/"
          "metadata/core-properties")
_DC_NS = "http://purl.org/dc/elements/1.1/"
_DCTERMS_NS = "http://purl.org/dc/terms/"
_XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"

_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?$")


# ── shared cell typing ──────────────────────────────────────────────────────

def _infer_cell(raw: str) -> tuple[str, str]:
    """Classify one string cell -> (kind, payload).

    Kinds: ``blank``, ``number`` (payload is the canonical literal),
    ``bool`` (payload ``1``/``0``), ``date`` (payload is the Excel
    serial as a literal), ``string`` (payload is the raw text).
    Strings that would lose information when typed — leading zeros
    (``"007"``), thousands separators (``"1,234"``) — stay strings.
    """
    text = (raw or "").strip()
    if not text:
        return "blank", ""
    if text in ("TRUE", "FALSE"):
        return "bool", "1" if text == "TRUE" else "0"
    if _INT_RE.match(text) and not (
            len(text.lstrip("+-")) > 1 and text.lstrip("+-").startswith("0")):
        return "number", str(int(text))
    if _FLOAT_RE.match(text) and ("." in text or "e" in text or "E" in text):
        if not (text.lstrip("+-").split(".")[0].startswith("0")
                and len(text.lstrip("+-").split(".")[0]) > 1):
            return "number", repr(float(text))
    if _DATE_RE.match(text):
        try:
            parsed = datetime.fromisoformat(text.replace(" ", "T"))
        except ValueError:
            return "string", raw
        serial = (parsed - datetime(1899, 12, 30)).total_seconds() / 86400.0
        return "date", repr(serial)
    return "string", raw


def _sheet_name(raw: str, used: set[str], fallback: str) -> str:
    """A legal, unique Excel sheet name (<=31 chars, no []:*?/\\)."""
    name = re.sub(r"[\[\]:*?/\\]", "-", (raw or "").strip())[:31].strip("'")
    if not name:
        name = fallback
    base, counter = name, 2
    while name.lower() in used:
        suffix = f" ({counter})"
        name = base[:31 - len(suffix)] + suffix
        counter += 1
    used.add(name.lower())
    return name


def _col_name(index: int) -> str:
    """0-based column index -> spreadsheet letters (A, B, …, AA)."""
    letters = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def _core_props(doc: Document, *, kind: str) -> str:
    """docProps/core.xml for an OOXML package."""
    created = datetime.fromtimestamp(
        doc.created_at or 0).strftime("%Y-%m-%dT%H:%M:%SZ") \
        if doc.created_at else "1970-01-01T00:00:00Z"
    return (
        _XML_DECL
        + f'<cp:coreProperties xmlns:cp="{_CP_NS}" xmlns:dc="{_DC_NS}" '
          f'xmlns:dcterms="{_DCTERMS_NS}" xmlns:xsi="{_XSI_NS}">'
        + (f"<dc:title>{_esc(doc.title)}</dc:title>" if doc.title else "")
        + (f"<dc:creator>{_esc(doc.author)}</dc:creator>" if doc.author else "")
        + f"<dc:description>Written by nomorals documents "
          f"({kind} writer)</dc:description>"
        + f'<dcterms:created xsi:type="dcterms:W3CDTF">{created}'
          "</dcterms:created>"
        + "</cp:coreProperties>")


# ── XLSX ────────────────────────────────────────────────────────────────────

_XLSX_DATE_STYLE_INDEX = 1  # cellXfs entry 1 -> built-in numFmt 14 (mm-dd-yy)


def _xlsx_sheet_xml(grid: list[list[str]], shared: list[str],
                    shared_index: dict[str, int]) -> str:
    rows_xml: list[str] = []
    for r, row in enumerate(grid, 1):
        cells: list[str] = []
        for c, raw in enumerate(row):
            kind, payload = _infer_cell(raw)
            if kind == "blank":
                continue
            ref = f"{_col_name(c)}{r}"
            if kind == "string":
                cells.append(
                    f'<c r="{ref}" t="s"><v>{shared_index[payload]}</v></c>')
            elif kind == "bool":
                cells.append(f'<c r="{ref}" t="b"><v>{payload}</v></c>')
            elif kind == "date":
                cells.append(
                    f'<c r="{ref}" s="{_XLSX_DATE_STYLE_INDEX}">'
                    f"<v>{payload}</v></c>")
            else:
                cells.append(f'<c r="{ref}"><v>{payload}</v></c>')
        if cells:
            rows_xml.append(f'<row r="{r}">{"".join(cells)}</row>')
    return (
        _XML_DECL
        + f'<worksheet xmlns="{_MAIN_NS}" xmlns:r="{_R_NS}">'
        + f"<sheetData>{''.join(rows_xml)}</sheetData></worksheet>")


def to_xlsx(doc: Document) -> bytes:
    """Render a Document as .xlsx bytes (stdlib ``zipfile`` only).

    One worksheet per table; when the document also carries prose, a
    leading ``Document`` sheet holds the title and one row per section
    (heading, text) so no content is dropped.  Raises
    :class:`DocumentError` (``kind="empty"``) for a document with
    neither sections nor tables.
    """
    if not doc.sections and not doc.tables:
        raise DocumentError("document has no content to render as xlsx",
                            kind="empty")
    sheets: list[tuple[str, list[list[str]]]] = []
    used_names: set[str] = set()
    if doc.sections:
        grid: list[list[str]] = []
        if doc.title:
            grid.append([doc.title])
        grid.append(["Heading", "Text"])
        for section in doc.sections:
            grid.append([section.heading, section.text])
        sheets.append((_sheet_name("Document", used_names, "Document"), grid))
    for i, table in enumerate(doc.tables):
        grid = ([list(table.headers)] if table.headers else []) \
            + [list(row) for row in table.rows]
        name = _sheet_name(table.name, used_names, f"Sheet {i + 1}")
        sheets.append((name, grid))

    # Shared strings: every string cell, in first-seen order.
    shared: list[str] = []
    shared_index: dict[str, int] = {}
    string_cells = 0
    for _name, grid in sheets:
        for row in grid:
            for raw in row:
                kind, payload = _infer_cell(raw)
                if kind == "string":
                    string_cells += 1
                    if payload not in shared_index:
                        shared_index[payload] = len(shared)
                        shared.append(payload)

    workbook_sheets = "".join(
        f'<sheet name="{_esc(name)}" sheetId="{i}" r:id="rId{i}"/>'
        for i, (name, _grid) in enumerate(sheets, 1))
    workbook = (
        _XML_DECL
        + f'<workbook xmlns="{_MAIN_NS}" xmlns:r="{_R_NS}">'
        + f"<sheets>{workbook_sheets}</sheets></workbook>")
    workbook_rels = (
        _XML_DECL
        + f'<Relationships xmlns="{_PKG_REL_NS}">'
        + "".join(
            f'<Relationship Id="rId{i}" Type="{_R_NS}/worksheet" '
            f'Target="worksheets/sheet{i}.xml"/>'
            for i in range(1, len(sheets) + 1))
        + f'<Relationship Id="rId{len(sheets) + 1}" Type="{_R_NS}/styles" '
          'Target="styles.xml"/>'
        + f'<Relationship Id="rId{len(sheets) + 2}" '
          f'Type="{_R_NS}/sharedStrings" Target="sharedStrings.xml"/>'
        + "</Relationships>")
    content_types = (
        _XML_DECL
        + '<Types xmlns="http://schemas.openxmlformats.org/package/2006/'
          'content-types">'
        + '<Default Extension="rels" ContentType="application/vnd.'
          'openxmlformats-package.relationships+xml"/>'
        + '<Default Extension="xml" ContentType="application/xml"/>'
        + '<Override PartName="/xl/workbook.xml" ContentType="application/'
          'vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        + "".join(
            f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.'
            'spreadsheetml.worksheet+xml"/>'
            for i in range(1, len(sheets) + 1))
        + '<Override PartName="/xl/styles.xml" ContentType="application/'
          'vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        + '<Override PartName="/xl/sharedStrings.xml" ContentType='
          '"application/vnd.openxmlformats-officedocument.spreadsheetml.'
          'sharedStrings+xml"/>'
        + '<Override PartName="/docProps/core.xml" ContentType='
          '"application/vnd.openxmlformats-package.core-properties+xml"/>'
        + "</Types>")
    package_rels = (
        _XML_DECL
        + f'<Relationships xmlns="{_PKG_REL_NS}">'
        + f'<Relationship Id="rId1" Type="{_R_NS}/officeDocument" '
          'Target="xl/workbook.xml"/>'
        + '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/'
          'package/2006/relationships/metadata/core-properties" '
          'Target="docProps/core.xml"/>'
        + "</Relationships>")
    styles = (
        _XML_DECL
        + f'<styleSheet xmlns="{_MAIN_NS}">'
        + '<numFmts count="0"/>'
        + '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font>'
          "</fonts>"
        + '<fills count="1"><fill><patternFill patternType="none"/></fill>'
          "</fills>"
        + '<borders count="1"><border/></borders>'
        + '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" '
          'borderId="0"/></cellStyleXfs>'
        # xf 0: general; xf 1: built-in date format 14 for date cells.
        + '<cellXfs count="2">'
        + '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        + '<xf numFmtId="14" fontId="0" fillId="0" borderId="0" xfId="0" '
          'applyNumberFormat="1"/>'
        + "</cellXfs>"
        + '<cellStyles count="1"><cellStyle name="Normal" xfId="0" '
        + 'builtinId="0"/></cellStyles></styleSheet>')
    shared_xml = (
        _XML_DECL
        + f'<sst xmlns="{_MAIN_NS}" count="{string_cells}" '
          f'uniqueCount="{len(shared)}">'
        + "".join(
            f'<si><t xml:space="preserve">{_esc(s)}</t></si>' for s in shared)
        + "</sst>")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", package_rels)
        archive.writestr("docProps/core.xml", _core_props(doc, kind="xlsx"))
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/styles.xml", styles)
        archive.writestr("xl/sharedStrings.xml", shared_xml)
        for i, (_name, grid) in enumerate(sheets, 1):
            archive.writestr(f"xl/worksheets/sheet{i}.xml",
                             _xlsx_sheet_xml(grid, shared, shared_index))
    return buffer.getvalue()


# ── PPTX ────────────────────────────────────────────────────────────────────

_P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
_A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"

#: 16:9 slide size in EMU.
_SLIDE_W, _SLIDE_H = 12192000, 6858000


def _tx_box(shape_id: int, name: str, *, x: int, y: int, cx: int, cy: int,
            paragraphs: list[tuple[str, int, bool]]) -> str:
    """One text-box shape. ``paragraphs`` are (text, font_size, bold)."""
    paras = []
    for text, size, bold in paragraphs:
        bold_attr = ' b="1"' if bold else ""
        paras.append(
            f"<a:p><a:pPr><a:defRPr sz=\"{size * 100}\"{bold_attr}/></a:pPr>"
            f"<a:r><a:rPr lang=\"en-US\" sz=\"{size * 100}\"{bold_attr}/>"
            f"<a:t>{_esc(text)}</a:t></a:r></a:p>")
    if not paras:
        paras.append("<a:p/>")
    return (
        f"<p:sp><p:nvSpPr><p:cNvPr id=\"{shape_id}\" name=\"{_esc(name)}\"/>"
        "<p:cNvSpPr txBox=\"1\"/><p:nvPr/></p:nvSpPr>"
        f"<p:spPr><a:xfrm><a:off x=\"{x}\" y=\"{y}\"/>"
        f"<a:ext cx=\"{cx}\" cy=\"{cy}\"/></a:xfrm>"
        "<a:prstGeom prst=\"rect\"><a:avLst/></a:prstGeom></p:spPr>"
        f"<p:txBody><a:bodyPr wrap=\"square\"/><a:lstStyle/>"
        f"{''.join(paras)}</p:txBody></p:sp>")


def _slide_xml(shapes: list[str]) -> str:
    return (
        _XML_DECL
        + f'<p:sld xmlns:a="{_A_NS}" xmlns:p="{_P_NS}" '
          f'xmlns:r="{_R_NS}"><p:cSld><p:spTree>'
        + '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/>'
          "<p:nvPr/></p:nvGrpSpPr><p:grpSpPr/>"
        + "".join(shapes)
        + "</p:spTree></p:cSld></p:sld>")


def _table_shape(shape_id: int, table: Table) -> str:
    """A DrawingML table graphic frame (what the pptx reader renders
    back as ``a | b`` lines)."""
    grid = ([list(table.headers)] if table.headers else []) \
        + [list(row) for row in table.rows]
    width = max((len(row) for row in grid), default=0)
    if not grid or width == 0:
        return ""
    col_w = max(914400, (_SLIDE_W - 2 * 457200) // width)
    cols = "".join(f'<a:gridCol w="{col_w}"/>' for _ in range(width))
    rows = []
    for row in grid:
        cells = []
        for j in range(width):
            text = row[j] if j < len(row) else ""
            cells.append(
                f"<a:tc><a:txBody><a:bodyPr/><a:lstStyle/>"
                f"<a:p><a:r><a:rPr lang=\"en-US\"/>"
                f"<a:t>{_esc(text)}</a:t></a:r></a:p></a:txBody>"
                "<a:tcPr/></a:tc>")
        rows.append(f'<a:tr h="370000">{"".join(cells)}</a:tr>')
    return (
        "<p:graphicFrame><p:nvGraphicFramePr>"
        f"<p:cNvPr id=\"{shape_id}\" name=\"Table\"/>"
        "<p:cNvGraphicFramePr><a:graphicFrameLocks noGrp=\"1\"/>"
        "</p:cNvGraphicFramePr><p:nvPr/></p:nvGraphicFramePr>"
        "<p:xfrm><a:off x=\"457200\" y=\"1600000\"/>"
        f"<a:ext cx=\"{col_w * width}\" cy=\"{370000 * len(grid)}\"/></p:xfrm>"
        f"<a:graphic><a:graphicData uri=\"http://schemas.openxmlformats.org/"
        f"drawingml/2006/table\"><a:tbl><a:tblPr/><a:tblGrid>{cols}"
        f"</a:tblGrid>{''.join(rows)}</a:tbl></a:graphicData></a:graphic>"
        "</p:graphicFrame>")


_PPTX_MASTER = (
    _XML_DECL
    + f'<p:sldMaster xmlns:a="{_A_NS}" xmlns:p="{_P_NS}" xmlns:r="{_R_NS}">'
    + "<p:cSld><p:spTree>"
    + '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/>'
      "</p:nvGrpSpPr><p:grpSpPr/></p:spTree></p:cSld>"
    + "<p:clrMap bg1=\"lt1\" tx1=\"dk1\" bg2=\"lt2\" tx2=\"dk2\" "
      "accent1=\"accent1\" accent2=\"accent2\" accent3=\"accent3\" "
      "accent4=\"accent4\" accent5=\"accent5\" accent6=\"accent6\" "
      "hlink=\"hlink\" folHlink=\"folHlink\"/>"
    + '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/>'
      "</p:sldLayoutIdLst>"
    + "<p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>"
    + "</p:sldMaster>")

_PPTX_LAYOUT = (
    _XML_DECL
    + f'<p:sldLayout xmlns:a="{_A_NS}" xmlns:p="{_P_NS}" xmlns:r="{_R_NS}" '
      'type="blank" preserve="1"><p:cSld name="Blank"><p:spTree>'
    + '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/>'
      "</p:nvGrpSpPr><p:grpSpPr/></p:spTree></p:cSld>"
    + '<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>')

_PPTX_THEME = (
    _XML_DECL
    + f'<a:theme xmlns:a="{_A_NS}" name="Office Theme"><a:themeElements>'
    + '<a:clrScheme name="Office">'
      '<a:dk1><a:sysClr val="windowText" lastClr="000000"/></a:dk1>'
      '<a:lt1><a:sysClr val="window" lastClr="FFFFFF"/></a:lt1>'
      '<a:dk2><a:srgbClr val="44546A"/></a:dk2>'
      '<a:lt2><a:srgbClr val="E7E6E6"/></a:lt2>'
      '<a:accent1><a:srgbClr val="4472C4"/></a:accent1>'
      '<a:accent2><a:srgbClr val="ED7D31"/></a:accent2>'
      '<a:accent3><a:srgbClr val="A5A5A5"/></a:accent3>'
      '<a:accent4><a:srgbClr val="FFC000"/></a:accent4>'
      '<a:accent5><a:srgbClr val="5B9BD5"/></a:accent5>'
      '<a:accent6><a:srgbClr val="70AD47"/></a:accent6>'
      '<a:hlink><a:srgbClr val="0563C1"/></a:hlink>'
      '<a:folHlink><a:srgbClr val="954F72"/></a:folHlink></a:clrScheme>'
    + '<a:fontScheme name="Office">'
      '<a:majorFont><a:latin typeface="Calibri Light"/></a:majorFont>'
      '<a:minorFont><a:latin typeface="Calibri"/></a:minorFont>'
      "</a:fontScheme>"
    + '<a:fmtScheme name="Office">'
      '<a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
      "</a:fillStyleLst>"
      '<a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/>'
      "</a:solidFill></a:ln></a:lnStyleLst>"
      "<a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle>"
      "</a:effectStyleLst>"
      '<a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
      "</a:bgFillStyleLst></a:fmtScheme>"
    + "</a:themeElements></a:theme>")


def to_pptx(doc: Document) -> bytes:
    """Render a Document as .pptx bytes (stdlib ``zipfile`` only).

    Title slide (when a title exists), one slide per section, one
    slide per table.  Raises :class:`DocumentError`
    (``kind="empty"``) for a document with no title, sections, or
    tables.
    """
    if not doc.title and not doc.sections and not doc.tables:
        raise DocumentError("document has no content to render as pptx",
                            kind="empty")
    slides: list[str] = []
    if doc.title:
        paras = [(doc.title, 40, True)]
        if doc.author:
            paras.append((f"Author: {doc.author}", 20, False))
        slides.append(_slide_xml([_tx_box(
            2, "Title", x=457200, y=2743200, cx=_SLIDE_W - 914400,
            cy=1371600, paragraphs=paras)]))
    for section in doc.sections:
        shapes = []
        heading = section.heading or doc.title or "Section"
        shapes.append(_tx_box(
            2, "Title", x=457200, y=228600, cx=_SLIDE_W - 914400,
            cy=914400, paragraphs=[(heading, 32, True)]))
        lines = [ln.strip() for ln in section.text.split("\n") if ln.strip()]
        if lines:
            shapes.append(_tx_box(
                3, "Body", x=457200, y=1371600, cx=_SLIDE_W - 914400,
                cy=_SLIDE_H - 1828800,
                paragraphs=[(ln, 18, False) for ln in lines]))
        slides.append(_slide_xml(shapes))
    for table in doc.tables:
        shape = _table_shape(3, table)
        if not shape:
            continue
        shapes = [_tx_box(
            2, "Title", x=457200, y=228600, cx=_SLIDE_W - 914400,
            cy=914400,
            paragraphs=[(table.name or "Table", 32, True)]), shape]
        slides.append(_slide_xml(shapes))
    if not slides:
        raise DocumentError("document has no content to render as pptx",
                            kind="empty")

    n = len(slides)
    sld_ids = "".join(
        f'<p:sldId id="{256 + i}" r:id="rId{i + 2}"/>' for i in range(n))
    presentation = (
        _XML_DECL
        + f'<p:presentation xmlns:a="{_A_NS}" xmlns:p="{_P_NS}" '
          f'xmlns:r="{_R_NS}">'
        + '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/>'
          "</p:sldMasterIdLst>"
        + f"<p:sldIdLst>{sld_ids}</p:sldIdLst>"
        + f'<p:sldSz cx="{_SLIDE_W}" cy="{_SLIDE_H}" type="screen16x9"/>'
        + '<p:notesSz cx="6858000" cy="9144000"/>'
        + "</p:presentation>")
    presentation_rels = (
        _XML_DECL
        + f'<Relationships xmlns="{_PKG_REL_NS}">'
        + f'<Relationship Id="rId1" Type="{_R_NS}/slideMaster" '
          'Target="slideMasters/slideMaster1.xml"/>'
        + "".join(
            f'<Relationship Id="rId{i + 2}" Type="{_R_NS}/slide" '
            f'Target="slides/slide{i + 1}.xml"/>' for i in range(n))
        + "</Relationships>")
    slide_rels = (
        _XML_DECL
        + f'<Relationships xmlns="{_PKG_REL_NS}">'
        + f'<Relationship Id="rId1" Type="{_R_NS}/slideLayout" '
          'Target="../slideLayouts/slideLayout1.xml"/>'
        + "</Relationships>")
    master_rels = (
        _XML_DECL
        + f'<Relationships xmlns="{_PKG_REL_NS}">'
        + f'<Relationship Id="rId1" Type="{_R_NS}/slideLayout" '
          'Target="../slideLayouts/slideLayout1.xml"/>'
        + f'<Relationship Id="rId2" Type="{_R_NS}/theme" '
          'Target="../theme/theme1.xml"/>'
        + "</Relationships>")
    layout_rels = (
        _XML_DECL
        + f'<Relationships xmlns="{_PKG_REL_NS}">'
        + f'<Relationship Id="rId1" Type="{_R_NS}/slideMaster" '
          'Target="../slideMasters/slideMaster1.xml"/>'
        + "</Relationships>")
    content_types = (
        _XML_DECL
        + '<Types xmlns="http://schemas.openxmlformats.org/package/2006/'
          'content-types">'
        + '<Default Extension="rels" ContentType="application/vnd.'
          'openxmlformats-package.relationships+xml"/>'
        + '<Default Extension="xml" ContentType="application/xml"/>'
        + '<Override PartName="/ppt/presentation.xml" ContentType='
          '"application/vnd.openxmlformats-officedocument.presentationml.'
          'presentation.main+xml"/>'
        + "".join(
            f'<Override PartName="/ppt/slides/slide{i + 1}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.'
            'presentationml.slide+xml"/>' for i in range(n))
        + '<Override PartName="/ppt/slideMasters/slideMaster1.xml" '
          'ContentType="application/vnd.openxmlformats-officedocument.'
          'presentationml.slideMaster+xml"/>'
        + '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" '
          'ContentType="application/vnd.openxmlformats-officedocument.'
          'presentationml.slideLayout+xml"/>'
        + '<Override PartName="/ppt/theme/theme1.xml" ContentType='
          '"application/vnd.openxmlformats-officedocument.theme+xml"/>'
        + '<Override PartName="/docProps/core.xml" ContentType='
          '"application/vnd.openxmlformats-package.core-properties+xml"/>'
        + '<Override PartName="/docProps/app.xml" ContentType='
          '"application/vnd.openxmlformats-officedocument.extended-'
          'properties+xml"/>'
        + "</Types>")
    package_rels = (
        _XML_DECL
        + f'<Relationships xmlns="{_PKG_REL_NS}">'
        + f'<Relationship Id="rId1" Type="{_R_NS}/officeDocument" '
          'Target="ppt/presentation.xml"/>'
        + '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/'
          'package/2006/relationships/metadata/core-properties" '
          'Target="docProps/core.xml"/>'
        + '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/'
          'officeDocument/2006/relationships/extended-properties" '
          'Target="docProps/app.xml"/>'
        + "</Relationships>")
    app_props = (
        _XML_DECL
        + '<Properties xmlns="http://schemas.openxmlformats.org/'
          'officeDocument/2006/extended-properties">'
        + "<Application>nomorals</Application>"
        + f"<Slides>{n}</Slides></Properties>")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", package_rels)
        archive.writestr("docProps/core.xml", _core_props(doc, kind="pptx"))
        archive.writestr("docProps/app.xml", app_props)
        archive.writestr("ppt/presentation.xml", presentation)
        archive.writestr("ppt/_rels/presentation.xml.rels", presentation_rels)
        archive.writestr("ppt/slideMasters/slideMaster1.xml", _PPTX_MASTER)
        archive.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels",
                         master_rels)
        archive.writestr("ppt/slideLayouts/slideLayout1.xml", _PPTX_LAYOUT)
        archive.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels",
                         layout_rels)
        archive.writestr("ppt/theme/theme1.xml", _PPTX_THEME)
        for i, slide in enumerate(slides, 1):
            archive.writestr(f"ppt/slides/slide{i}.xml", slide)
            archive.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", slide_rels)
    return buffer.getvalue()
