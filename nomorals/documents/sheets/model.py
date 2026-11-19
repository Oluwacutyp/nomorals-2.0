"""Workbook model for the GridCraft port — stdlib only.

Port of the *design* of upstream ``crates/model`` (workbook, sheets,
cells, styles) + ``crates/xlsx`` (the OOXML reader/writer): a real
spreadsheet document model over real .xlsx bytes.

Preservation-first: the model keeps the original XML trees for every
part; an edit re-serializes only the parts it dirtied (and inside a
dirty sheet part, only ``sheetData``/``dimension``/``mergeCells``/
``cols`` are rebuilt — drawings, validations, conditional formatting
and every other sibling survive untouched). A workbook that was only
opened writes back with every part byte-identical.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Any

from ..errors import DocumentError
from .ooxml import (
    R_NS,
    X_NS,
    XlsxPackage,
    children,
    first_child,
    local,
    parse_part,
    qname,
    read_package,
    serialize_xml,
)

__all__ = [
    "Cell",
    "CellError",
    "CellStyle",
    "Workbook",
    "Worksheet",
    "a1_ref",
    "col_to_num",
    "date_to_serial",
    "num_to_col",
    "parse_a1",
    "parse_range",
    "serial_to_date",
]


# ── cell errors (upstream crates/core/src/value.rs) ──────────────────


class CellError(Enum):
    """Error values — the upstream taxonomy, display strings included."""

    NULL = "#NULL!"
    DIV0 = "#DIV/0!"
    VALUE = "#VALUE!"
    REF = "#REF!"
    NAME = "#NAME?"
    NUM = "#NUM!"
    NA = "#N/A"
    SPILL = "#SPILL!"
    CALC = "#CALC!"
    CIRC = "#CIRC!"
    GETTING_DATA = "#GETTING_DATA"

    @property
    def display(self) -> str:
        return self.value

    @classmethod
    def from_display(cls, text: str) -> CellError | None:
        for member in cls:
            if member.value == text:
                return member
        return None


# ── A1 references ────────────────────────────────────────────────────

_A1_RE = re.compile(r"^(\$?)([A-Za-z]{1,3})(\$?)([0-9]+)$")
_RANGE_RE = re.compile(
    r"^(\$?[A-Za-z]{1,3}\$?[0-9]+):(\$?[A-Za-z]{1,3}\$?[0-9]+)$")


def col_to_num(letters: str) -> int:
    num = 0
    for char in letters.upper():
        if not ("A" <= char <= "Z"):
            raise DocumentError(f"bad column {letters!r}",
                                kind="invalid_argument")
        num = num * 26 + (ord(char) - 64)
    if num < 1 or num > 16_384:
        raise DocumentError(f"column out of range: {letters!r}",
                            kind="invalid_argument")
    return num


def num_to_col(num: int) -> str:
    if num < 1 or num > 16_384:
        raise DocumentError(f"column out of range: {num}",
                            kind="invalid_argument")
    letters = ""
    while num:
        num, rem = divmod(num - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def parse_a1(ref: str) -> tuple[int, int]:
    """``B12`` → ``(12, 2)``; ``$B$12`` likewise. Raises DocumentError."""
    match = _A1_RE.match((ref or "").strip())
    if not match:
        raise DocumentError(f"bad cell reference {ref!r}",
                            kind="invalid_argument")
    row = int(match.group(4))
    if row < 1 or row > 1_048_576:
        raise DocumentError(f"row out of range: {ref!r}",
                            kind="invalid_argument")
    return row, col_to_num(match.group(2))


def a1_ref(row: int, col: int) -> str:
    return f"{num_to_col(col)}{row}"


def parse_range(text: str) -> tuple[int, int, int, int]:
    """``A1:B2`` → ``(1, 1, 2, 2)`` (r1, c1, r2, c2, normalized)."""
    match = _RANGE_RE.match((text or "").strip())
    if not match:
        raise DocumentError(f"bad range {text!r}",
                            kind="invalid_argument")
    r1, c1 = parse_a1(match.group(1))
    r2, c2 = parse_a1(match.group(2))
    return min(r1, r2), min(c1, c2), max(r1, r2), max(c1, c2)


# ── Excel serial dates ───────────────────────────────────────────────


def date_to_serial(year: int, month: int, day: int) -> float:
    """Excel 1900-system serial for a calendar date."""
    target = date(year, month, day)
    if target >= date(1900, 3, 1):
        return float((target - date(1899, 12, 30)).days)
    return float((target - date(1899, 12, 31)).days)


def serial_to_date(serial: float) -> date:
    """Excel 1900-system serial back to a calendar date.

    Serial 60 is Excel's fake 1900-02-29; it maps to 1900-02-28
    here (Python has no such day) — the standard compromise.
    """
    days = int(serial)
    if days < 60:
        return date(1899, 12, 31) + timedelta(days=days)
    return date(1899, 12, 30) + timedelta(days=days)


def _is_date_format(fmt: str) -> bool:
    text = fmt.lower()
    return any(tok in text for tok in ("yy", "mm", "dd", "hh"))


# ── styles ───────────────────────────────────────────────────────────

_BUILTIN_NUMFMTS = {
    0: "General", 1: "0", 2: "0.00", 3: "#,##0", 4: "#,##0.00",
    9: "0%", 10: "0.00%", 11: "0.00E+00", 12: "# ?/?", 13: "# ??/??",
    14: "mm-dd-yy", 15: "d-mmm-yy", 16: "d-mmm", 17: "mmm-yy",
    18: "h:mm AM/PM", 19: "h:mm:ss AM/PM", 20: "h:mm", 21: "h:mm:ss",
    22: "m/d/yy h:mm", 37: "#,##0 ;(#,##0)",
    38: "#,##0 ;[Red](#,##0)", 39: "#,##0.00;(#,##0.00)",
    40: "#,##0.00;[Red](#,##0.00)",
    44: '_("$"* #,##0.00_);_("$"* \\(#,##0.00\\);_("$"* "-"??_);_(@_)',
    45: "mm:ss", 46: "[h]:mm:ss", 47: "mmss.0", 48: "##0.0E+0",
    49: "@",
}


@dataclass
class CellStyle:
    """A resolved cell style (one xf entry, flattened)."""

    bold: bool = False
    italic: bool = False
    font_color: str | None = None
    font_size: float | None = None
    font_name: str | None = None
    fill: str | None = None
    number_format: str = "General"
    horizontal: str | None = None
    vertical: str | None = None
    wrap_text: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "bold": self.bold, "italic": self.italic,
            "font_color": self.font_color, "font_size": self.font_size,
            "font_name": self.font_name, "fill": self.fill,
            "number_format": self.number_format,
            "horizontal": self.horizontal, "vertical": self.vertical,
            "wrap_text": self.wrap_text,
        }


@dataclass
class Cell:
    """One cell: value or formula, cached result, style."""

    ref: str
    row: int
    col: int
    kind: str = "n"  # n | s | b | e | str | inline
    value: Any = None
    formula: str | None = None  # without the leading '='
    cached: Any = None
    style_index: int = 0
    style: CellStyle = field(default_factory=CellStyle)
    ast: Any = None  # parsed formula, filled lazily by the calc engine

    @property
    def is_formula(self) -> bool:
        return self.formula is not None

    @property
    def display_value(self) -> Any:
        """What the cell shows: the cached result for a formula."""
        if self.formula is not None:
            return self.cached
        return self.value

    def as_datetime(self) -> date | None:
        """The cell as a date, when it holds a serial under a date format."""
        if not isinstance(self.display_value, (int, float)):
            return None
        if not _is_date_format(self.style.number_format):
            return None
        try:
            return serial_to_date(float(self.display_value))
        except (ValueError, OverflowError):
            return None

    def to_dict(self) -> dict[str, Any]:
        cached = self.cached
        if isinstance(cached, CellError):
            cached = cached.display
        return {
            "ref": self.ref, "kind": self.kind,
            "value": self.value, "formula": self.formula,
            "cached": cached, "style": self.style.to_dict(),
        }


# ── worksheet ────────────────────────────────────────────────────────


class Worksheet:
    """One sheet: cells, merges, column widths, table inventory."""

    def __init__(self, name: str, sheet_id: int, part: str,
                 rid: str | None = None) -> None:
        self.name = name
        self.sheet_id = sheet_id
        self.part = part  # e.g. xl/worksheets/sheet1.xml
        # The workbook.xml.rels id for this sheet, kept so a
        # re-serialized workbook.xml still points at the right rel.
        self.rid = rid or f"rId{sheet_id}"
        self.cells: dict[tuple[int, int], Cell] = {}
        self.merged_cells: list[str] = []
        self.col_widths: dict[int, float] = {}
        self.row_heights: dict[int, float] = {}
        self.freeze: str | None = None
        self.tables: list[str] = []
        self.dimension: str = "A1"

    def cell(self, ref: str) -> Cell:
        row, col = parse_a1(ref)
        found = self.cells.get((row, col))
        if found is None:
            raise DocumentError(
                f"sheet {self.name!r} has no cell {ref!r}",
                kind="invalid_argument")
        return found

    def value(self, ref: str) -> Any:
        row, col = parse_a1(ref)
        found = self.cells.get((row, col))
        if found is None:
            return None
        return found.display_value

    @property
    def max_row(self) -> int:
        return max((row for row, _ in self.cells), default=0)

    @property
    def max_column(self) -> int:
        return max((col for _, col in self.cells), default=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "sheet_id": self.sheet_id,
            "max_row": self.max_row, "max_column": self.max_column,
            "merged_cells": list(self.merged_cells),
            "tables": list(self.tables),
            "cells": [self.cells[key].to_dict()
                      for key in sorted(self.cells)],
        }


# ── workbook ─────────────────────────────────────────────────────────


class Workbook:
    """A whole .xlsx workbook: sheets, names, styles, properties."""

    def __init__(self, package: XlsxPackage) -> None:
        self.package = package
        self.sheets: list[Worksheet] = []
        self.sheets_by_name: dict[str, Worksheet] = {}
        self.defined_names: dict[str, str] = {}
        self.core_properties: dict[str, str] = {}
        self._touched: set[tuple[str, str]] = set()
        self._dirty_sheets: set[str] = set()
        self._shared_strings: list[str] = []
        self._shared_dirty = False
        self._styles_dirty = False
        self._workbook_dirty = False
        self._core_dirty = False
        self._fonts: list[dict[str, Any]] = []
        self._fills: list[str | None] = []
        self._numfmts: dict[int, str] = {}
        self._xfs: list[dict[str, Any]] = []
        self._sheet_roots: dict[str, Any] = {}
        self._styles_root: Any = None
        self._shared_root: Any = None
        self._workbook_root: Any = None
        self._core_root: Any = None
        self._parse()

    # ── construction ──

    @classmethod
    def from_bytes(cls, data: bytes) -> Workbook:
        return cls(read_package(data))

    @classmethod
    def blank(cls) -> Workbook:
        """A minimal one-sheet workbook, built from literal parts."""
        parts = _blank_parts()
        return cls(XlsxPackage(parts))

    @property
    def sheet_names(self) -> list[str]:
        return [ws.name for ws in self.sheets]

    def sheet(self, name: str) -> Worksheet:
        found = self.sheets_by_name.get(name)
        if found is None:
            raise DocumentError(f"no sheet named {name!r}",
                                kind="invalid_argument")
        return found

    # ── editing support (the SheetEditor drives these) ──

    def mark_sheet_dirty(self, part: str) -> None:
        self._dirty_sheets.add(part)

    def touch(self, sheet: str, ref: str) -> None:
        """Record an edited cell for the fidelity report."""
        self._touched.add((sheet, ref))

    def mark_core_dirty(self) -> None:
        self._core_dirty = True

    def mark_workbook_dirty(self) -> None:
        self._workbook_dirty = True

    def add_sheet(self, name: str) -> Worksheet:
        """Append a blank sheet; the name must be new."""
        if name in self.sheets_by_name:
            raise DocumentError(f"sheet {name!r} already exists",
                                kind="invalid_argument")
        number = 1
        while f"xl/worksheets/sheet{number}.xml" in self.package.parts:
            number += 1
        part = f"xl/worksheets/sheet{number}.xml"
        sheet_id = max((ws.sheet_id for ws in self.sheets),
                       default=0) + 1
        used_rids = {ws.rid for ws in self.sheets}
        rid_number = 1
        while f"rId{rid_number}" in used_rids:
            rid_number += 1
        rid = f"rId{rid_number}"
        self.package.set_part(part, _blank_sheet_xml())
        self._register_sheet_part(part, rid)
        ws = Worksheet(name, sheet_id, part, rid=rid)
        # The serializer reads the sheet root from this map — a new
        # sheet needs an entry here or to_bytes raises KeyError.
        from xml.etree import ElementTree as ET

        self._sheet_roots[part] = ET.fromstring(_blank_sheet_xml())
        self.sheets.append(ws)
        self.sheets_by_name[name] = ws
        self._workbook_dirty = True
        return ws

    def _register_sheet_part(self, part: str, rid: str) -> None:
        """Add a new sheet to [Content_Types].xml and the workbook
        rels, so readers (including Excel) can find the part."""
        from xml.etree import ElementTree as ET

        from .ooxml import CT_NS, PR_NS

        types = parse_part(self.package, "[Content_Types].xml")
        if types is not None:
            ET.SubElement(types, qname(CT_NS, "Override"), {
                "PartName": "/" + part,
                "ContentType": "application/vnd.openxmlformats-"
                "officedocument.spreadsheetml.worksheet+xml",
            })
            self.package.set_part("[Content_Types].xml",
                                  serialize_xml(types))
        rels = parse_part(self.package, "xl/_rels/workbook.xml.rels")
        if rels is not None:
            ET.SubElement(rels, qname(PR_NS, "Relationship"), {
                "Id": rid,
                "Type": "http://schemas.openxmlformats.org/"
                "officeDocument/2006/relationships/worksheet",
                "Target": part.split("xl/", 1)[1],
            })
            self.package.set_part("xl/_rels/workbook.xml.rels",
                                  serialize_xml(rels))

    def remove_sheet(self, name: str) -> None:
        """Remove a sheet; a workbook keeps at least one sheet.

        The sheet's part stays in the package as an orphan (readers
        ignore unreferenced parts) — deleting it would also mean
        rewriting the rels and content types, and an orphan is the
        safer half of that trade.
        """
        if name not in self.sheets_by_name:
            raise DocumentError(f"no sheet named {name!r}",
                                kind="invalid_argument")
        if len(self.sheets) <= 1:
            raise DocumentError(
                "a workbook needs at least one sheet",
                kind="invalid_argument")
        ws = self.sheets_by_name.pop(name)
        self.sheets.remove(ws)
        self._workbook_dirty = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "sheets": [ws.to_dict() for ws in self.sheets],
            "defined_names": dict(self.defined_names),
            "core_properties": dict(self.core_properties),
        }

    # ── parsing ──

    def _parse(self) -> None:
        pkg = self.package
        workbook_root = parse_part(pkg, "xl/workbook.xml")
        if workbook_root is None:
            raise DocumentError("xl/workbook.xml is missing",
                                kind="parse")
        self._workbook_root = workbook_root
        rels_root = parse_part(pkg, "xl/_rels/workbook.xml.rels")
        rel_map: dict[str, str] = {}
        if rels_root is not None:
            for rel in children(rels_root, "Relationship"):
                # Targets may be relative ("worksheets/sheet1.xml")
                # or absolute ("/xl/worksheets/sheet1.xml", as
                # openpyxl writes them) — normalise both to the
                # "xl/..." part name.
                target = rel.get("Target", "").lstrip("/")
                if not target.startswith("xl/"):
                    target = "xl/" + target
                rel_map[rel.get("Id", "")] = target
        sheets_el = first_child(workbook_root, "sheets")
        if sheets_el is None:
            raise DocumentError("workbook has no <sheets>",
                                kind="parse")
        sheet_defs: list[tuple[str, int, str, str]] = []
        for sheet_el in children(sheets_el, "sheet"):
            name = sheet_el.get("name", "")
            sheet_id = int(sheet_el.get("sheetId", "1") or "1")
            rid = sheet_el.get(f"{{{R_NS}}}id", "")
            target = rel_map.get(rid, "")
            if not name or not target:
                raise DocumentError(
                    f"sheet {name!r} has no resolvable target",
                    kind="parse")
            sheet_defs.append((name, sheet_id, target, rid))
        names_el = first_child(workbook_root, "definedNames")
        if names_el is not None:
            for name_el in children(names_el, "definedName"):
                name = name_el.get("name", "")
                if name:
                    self.defined_names[name] = (name_el.text or "")
        self._parse_shared_strings()
        self._parse_styles()
        self._parse_core()
        for name, sheet_id, target, rid in sheet_defs:
            self._parse_sheet(name, sheet_id, target, rid)
        self._fill_missing_cached()

    def _fill_missing_cached(self) -> None:
        """Compute cached values for formula cells that carry none.

        openpyxl writes ``<v></v>`` for formulas and some producers
        store error displays without ``t="e"`` — both read as "no
        trustworthy cached value". The calc engine fills those in
        (recursing into dependencies, cycles surfacing as #CIRC!),
        the way Excel recomputes on open. Cells with a stored cached
        value keep it; nothing here marks any part dirty, so a
        no-edit round trip still writes back byte-identical parts.
        """
        from .formula import evaluate

        for ws in self.sheets:
            for cell in ws.cells.values():
                if cell.formula is None or cell.cached is not None:
                    continue
                try:
                    cell.cached = evaluate(self, ws.name, cell.ref,
                                           cell.formula)
                except Exception:
                    cell.cached = CellError.CALC

    def _parse_shared_strings(self) -> None:
        root = parse_part(self.package, "xl/sharedStrings.xml")
        self._shared_root = root
        if root is None:
            return
        for si in children(root, "si"):
            text = "".join(t.text or "" for t in si.iter()
                           if local(t.tag) == "t")
            self._shared_strings.append(text)

    def _parse_styles(self) -> None:
        root = parse_part(self.package, "xl/styles.xml")
        self._styles_root = root
        if root is None:
            return
        numfmts = first_child(root, "numFmts")
        if numfmts is not None:
            for fmt in children(numfmts, "numFmt"):
                try:
                    self._numfmts[int(fmt.get("numFmtId", "0"))] = \
                        fmt.get("formatCode", "General")
                except ValueError:
                    continue
        fonts = first_child(root, "fonts")
        if fonts is not None:
            for font in children(fonts, "font"):
                entry: dict[str, Any] = {}
                if first_child(font, "b") is not None:
                    entry["bold"] = True
                if first_child(font, "i") is not None:
                    entry["italic"] = True
                sz = first_child(font, "sz")
                if sz is not None:
                    with contextlib.suppress(ValueError):
                        entry["size"] = float(sz.get("val", "11"))
                color = first_child(font, "color")
                if color is not None and color.get("rgb"):
                    entry["color"] = color.get("rgb")
                name = first_child(font, "name")
                if name is not None:
                    entry["font_name"] = name.get("val")
                self._fonts.append(entry)
        fills = first_child(root, "fills")
        if fills is not None:
            for fill in children(fills, "fill"):
                pattern = first_child(fill, "patternFill")
                fg = None
                if pattern is not None:
                    fg_el = first_child(pattern, "fgColor")
                    if fg_el is not None:
                        fg = fg_el.get("rgb")
                self._fills.append(fg)
        xfs = first_child(root, "cellXfs")
        if xfs is not None:
            for xf in children(xfs, "xf"):
                try:
                    self._xfs.append({
                        "numFmtId": int(xf.get("numFmtId", "0")),
                        "fontId": int(xf.get("fontId", "0")),
                        "fillId": int(xf.get("fillId", "0")),
                        "applyNumberFormat":
                            xf.get("applyNumberFormat") == "1",
                    })
                except ValueError:
                    continue

    def _parse_core(self) -> None:
        root = parse_part(self.package, "docProps/core.xml")
        self._core_root = root
        if root is None:
            return
        for child in root:
            name = local(child.tag)
            if name in ("title", "creator", "lastModifiedBy",
                        "description", "subject", "keywords"):
                self.core_properties[name] = child.text or ""

    def _parse_sheet(self, name: str, sheet_id: int, part: str,
                     rid: str = "") -> None:
        root = parse_part(self.package, part)
        if root is None:
            raise DocumentError(f"sheet part {part!r} is missing",
                                kind="parse")
        self._sheet_roots[part] = root
        ws = Worksheet(name, sheet_id, part, rid=rid or None)
        dim = first_child(root, "dimension")
        if dim is not None and dim.get("ref"):
            ws.dimension = dim.get("ref")
        views = first_child(root, "sheetViews")
        if views is not None:
            view = first_child(views, "sheetView")
            if view is not None:
                pane = first_child(view, "pane")
                if pane is not None and pane.get("topLeftCell"):
                    ws.freeze = pane.get("topLeftCell")
        cols = first_child(root, "cols")
        if cols is not None:
            for col in children(cols, "col"):
                try:
                    lo = int(col.get("min", "1"))
                    hi = int(col.get("max", "1"))
                    width = float(col.get("width", "0"))
                except ValueError:
                    continue
                for index in range(lo, hi + 1):
                    ws.col_widths[index] = width
        merges = first_child(root, "mergeCells")
        if merges is not None:
            for cell in children(merges, "mergeCell"):
                if cell.get("ref"):
                    ws.merged_cells.append(cell.get("ref"))
        table_parts = first_child(root, "tableParts")
        if table_parts is not None:
            for table in children(table_parts, "tablePart"):
                rid = table.get(f"{{{R_NS}}}id", "")
                ws.tables.append(rid or "table")
        data = first_child(root, "sheetData")
        shared_masters: dict[str, tuple[str, int, int]] = {}
        if data is not None:
            for row_el in children(data, "row"):
                try:
                    row_num = int(row_el.get("r", "0"))
                except ValueError:
                    continue
                if row_el.get("ht"):
                    with contextlib.suppress(ValueError):
                        ws.row_heights[row_num] = float(
                            row_el.get("ht", "0"))
                for cell_el in children(row_el, "c"):
                    cell = self._parse_cell(cell_el, row_num, ws,
                                            shared_masters)
                    if cell is not None:
                        ws.cells[(cell.row, cell.col)] = cell
        self.sheets.append(ws)
        self.sheets_by_name[name] = ws

    def _parse_cell(self, cell_el: Any, row_num: int, ws: Worksheet,
                    shared_masters: dict[str, tuple[str, int, int]]
                    ) -> Cell | None:
        ref_text = cell_el.get("r", "")
        try:
            row, col = parse_a1(ref_text)
        except DocumentError:
            return None
        style_index = 0
        with contextlib.suppress(ValueError):
            style_index = int(cell_el.get("s", "0") or "0")
        kind = cell_el.get("t", "n")
        formula_el = first_child(cell_el, "f")
        value_el = first_child(cell_el, "v")
        is_el = first_child(cell_el, "is")
        formula: str | None = None
        cached: Any = None
        value: Any = None
        if formula_el is not None:
            text = formula_el.text or ""
            shared = formula_el.get("t") == "shared"
            si = formula_el.get("si")
            if shared and si is not None and not text:
                master = shared_masters.get(si)
                if master is not None:
                    master_text, mrow, mcol = master
                    formula = _translate_shared(
                        master_text, row - mrow, col - mcol)
                else:
                    formula = None
            elif shared and si is not None and text:
                shared_masters[si] = (text, row, col)
                formula = text
            else:
                formula = text
            if value_el is not None:
                text = value_el.text or ""
                if kind == "e":
                    cached = (CellError.from_display(text.strip())
                              or CellError.VALUE)
                elif kind == "str":
                    cached = text
                elif kind in ("s", "b"):
                    cached = self._parse_value(text, kind)
                elif text.strip() == "":
                    # openpyxl writes <v></v> for formulas: no cached
                    # value at all — the fill pass computes it.
                    cached = None
                else:
                    parsed = self._parse_value(text, kind)
                    # A stored error display under kind "n" is a
                    # malformed file (Excel writes t="e" for those);
                    # trust the engine over the stale text.
                    cached = (None if isinstance(parsed, CellError)
                              else parsed)
            elif is_el is not None:
                cached = "".join(t.text or "" for t in is_el.iter()
                                 if local(t.tag) == "t")
        elif is_el is not None:
            kind = "inline"
            value = "".join(t.text or "" for t in is_el.iter()
                            if local(t.tag) == "t")
        elif value_el is not None:
            value = self._parse_value(value_el.text or "", kind)
        else:
            value = None
        if formula is not None:
            kind = "str" if kind == "str" else kind
        return Cell(ref=a1_ref(row, col), row=row, col=col, kind=kind,
                    value=value, formula=formula, cached=cached,
                    style_index=style_index,
                    style=self.style_for_index(style_index))

    def _parse_value(self, text: str, kind: str) -> Any:
        if kind == "s":
            try:
                index = int(text)
            except ValueError:
                return CellError.VALUE
            if 0 <= index < len(self._shared_strings):
                return self._shared_strings[index]
            return CellError.REF
        if kind == "b":
            return text.strip() == "1"
        if kind == "e":
            return CellError.from_display(text.strip()) \
                or CellError.VALUE
        if kind == "str":
            return text
        try:
            num = float(text)
        except ValueError:
            return CellError.VALUE
        return int(num) if num.is_integer() else num

    # ── styles ──

    def style_for_index(self, index: int) -> CellStyle:
        if index < 0 or index >= len(self._xfs):
            return CellStyle()
        xf = self._xfs[index]
        style = CellStyle()
        fmt_id = xf.get("numFmtId", 0)
        style.number_format = self._numfmts.get(
            fmt_id, _BUILTIN_NUMFMTS.get(fmt_id, "General"))
        font_id = xf.get("fontId", 0)
        if 0 <= font_id < len(self._fonts):
            font = self._fonts[font_id]
            style.bold = bool(font.get("bold"))
            style.italic = bool(font.get("italic"))
            style.font_color = font.get("color")
            style.font_size = font.get("size")
            style.font_name = font.get("font_name")
        fill_id = xf.get("fillId", 0)
        if 0 <= fill_id < len(self._fills):
            style.fill = self._fills[fill_id]
        return style

    def style_index_for(self, style: CellStyle) -> int:
        """Find or create the xf entry for a resolved style."""
        for index in range(len(self._xfs)):
            if self.style_for_index(index).to_dict() \
                    == style.to_dict():
                return index
        font_id = self._font_id_for(style)
        fill_id = self._fill_id_for(style)
        fmt_id = self._numfmt_id_for(style.number_format)
        self._xfs.append({
            "numFmtId": fmt_id, "fontId": font_id, "fillId": fill_id,
            "applyNumberFormat": True,
        })
        self._styles_dirty = True
        return len(self._xfs) - 1

    def _font_id_for(self, style: CellStyle) -> int:
        entry = {"bold": style.bold, "italic": style.italic,
                 "color": style.font_color, "size": style.font_size,
                 "font_name": style.font_name}
        for index, font in enumerate(self._fonts):
            if font == entry:
                return index
        self._fonts.append(entry)
        return len(self._fonts) - 1

    def _fill_id_for(self, style: CellStyle) -> int:
        if not style.fill:
            return 0
        for index, fill in enumerate(self._fills):
            if fill == style.fill:
                return index
        self._fills.append(style.fill)
        return len(self._fills) - 1

    def _numfmt_id_for(self, fmt: str) -> int:
        for code, name in _BUILTIN_NUMFMTS.items():
            if name == fmt:
                return code
        for code, name in self._numfmts.items():
            if name == fmt:
                return code
        code = max(list(self._numfmts) + [164]) + 1
        self._numfmts[code] = fmt
        return code

    def shared_string_id(self, text: str) -> int:
        """Find or append a shared string; returns its index."""
        try:
            return self._shared_strings.index(text)
        except ValueError:
            pass
        self._shared_strings.append(text)
        self._shared_dirty = True
        return len(self._shared_strings) - 1

    # ── serialization ──

    def to_bytes(self) -> bytes:
        """Write the workbook back; clean parts stay byte-identical."""
        pkg = self.package
        for ws in self.sheets:
            if ws.part in self._dirty_sheets:
                self._serialize_sheet(ws)
        if self._shared_dirty:
            self._serialize_shared_strings()
        if self._styles_dirty:
            self._serialize_styles()
        if self._workbook_dirty:
            self._serialize_workbook()
        if self._core_dirty:
            self._serialize_core()
        return pkg.to_bytes()

    def _serialize_sheet(self, ws: Worksheet) -> None:
        from xml.etree import ElementTree as ET

        root = self._sheet_roots[ws.part]
        old_data = first_child(root, "sheetData")
        data = ET.Element(qname(X_NS, "sheetData"))
        rows: dict[int, list[Cell]] = {}
        for cell in ws.cells.values():
            rows.setdefault(cell.row, []).append(cell)
        for row_num in sorted(rows):
            row_el = ET.SubElement(data, qname(X_NS, "row"),
                                   {"r": str(row_num)})
            if row_num in ws.row_heights:
                row_el.set("ht", repr(ws.row_heights[row_num]))
            for cell in sorted(rows[row_num], key=lambda c: c.col):
                row_el.append(self._cell_element(cell))
        if old_data is not None:
            root.remove(old_data)
        # sheetData belongs after cols/sheetFormatPr in schema order;
        # appending it last is still valid for every reader we target.
        root.append(data)
        dim = first_child(root, "dimension")
        dim_ref = ws.dimension
        if ws.cells:
            dim_ref = (f"A1:{a1_ref(ws.max_row, ws.max_column)}")
        if dim is not None:
            dim.set("ref", dim_ref)
        else:
            dim_el = ET.Element(qname(X_NS, "dimension"),
                                {"ref": dim_ref})
            root.insert(0, dim_el)
        merges = first_child(root, "mergeCells")
        if merges is not None:
            root.remove(merges)
        if ws.merged_cells:
            merge_el = ET.Element(qname(X_NS, "mergeCells"),
                                  {"count": str(len(ws.merged_cells))})
            for ref in ws.merged_cells:
                ET.SubElement(merge_el, qname(X_NS, "mergeCell"),
                              {"ref": ref})
            root.append(merge_el)
        self.package.set_part(ws.part, serialize_xml(root))
        self._dirty_sheets.discard(ws.part)

    def _cell_element(self, cell: Cell) -> Any:
        from xml.etree import ElementTree as ET

        attrs: dict[str, str] = {"r": cell.ref}
        if cell.style_index:
            attrs["s"] = str(cell.style_index)
        el = ET.Element(qname(X_NS, "c"), attrs)
        if cell.formula is not None:
            if isinstance(cell.cached, CellError):
                el.set("t", "e")
            elif isinstance(cell.cached, bool):
                el.set("t", "b")
            elif isinstance(cell.cached, str):
                el.set("t", "str")
            else:
                el.set("t", "n")
            f_el = ET.SubElement(el, qname(X_NS, "f"))
            f_el.text = cell.formula
            if cell.cached is not None:
                v_el = ET.SubElement(el, qname(X_NS, "v"))
                v_el.text = _value_text(cell.cached)
            return el
        if cell.value is None:
            return el
        if isinstance(cell.value, bool):
            el.set("t", "b")
            v_el = ET.SubElement(el, qname(X_NS, "v"))
            v_el.text = "1" if cell.value else "0"
        elif isinstance(cell.value, CellError):
            el.set("t", "e")
            v_el = ET.SubElement(el, qname(X_NS, "v"))
            v_el.text = cell.value.display
        elif isinstance(cell.value, str):
            el.set("t", "s")
            v_el = ET.SubElement(el, qname(X_NS, "v"))
            v_el.text = str(self.shared_string_id(cell.value))
        else:
            v_el = ET.SubElement(el, qname(X_NS, "v"))
            v_el.text = _value_text(cell.value)
        return el

    def _serialize_shared_strings(self) -> None:
        from xml.etree import ElementTree as ET

        root = self._shared_root
        if root is None:
            root = ET.Element(qname(X_NS, "sst"),
                              {"count": "0", "uniqueCount": "0"})
            self._shared_root = root
        for si in list(root):
            root.remove(si)
        root.set("count", str(len(self._shared_strings)))
        root.set("uniqueCount", str(len(self._shared_strings)))
        for text in self._shared_strings:
            si = ET.SubElement(root, qname(X_NS, "si"))
            t_el = ET.SubElement(si, qname(X_NS, "t"))
            t_el.text = text
        self.package.set_part("xl/sharedStrings.xml",
                              serialize_xml(root))
        self._shared_dirty = False

    def _serialize_styles(self) -> None:
        from xml.etree import ElementTree as ET

        root = self._styles_root
        if root is None:
            return
        numfmts = first_child(root, "numFmts")
        if numfmts is None:
            numfmts = ET.Element(qname(X_NS, "numFmts"),
                                 {"count": "0"})
            root.insert(0, numfmts)
        for fmt in list(numfmts):
            numfmts.remove(fmt)
        custom = {k: v for k, v in self._numfmts.items() if k >= 164}
        numfmts.set("count", str(len(custom)))
        for code in sorted(custom):
            ET.SubElement(numfmts, qname(X_NS, "numFmt"),
                          {"numFmtId": str(code),
                           "formatCode": custom[code]})
        fonts = first_child(root, "fonts")
        if fonts is None:
            fonts = ET.Element(qname(X_NS, "fonts"), {"count": "0"})
            root.append(fonts)
        for font in list(fonts):
            fonts.remove(font)
        fonts.set("count", str(len(self._fonts)))
        for entry in self._fonts:
            font_el = ET.SubElement(fonts, qname(X_NS, "font"))
            if entry.get("bold"):
                ET.SubElement(font_el, qname(X_NS, "b"))
            if entry.get("italic"):
                ET.SubElement(font_el, qname(X_NS, "i"))
            if entry.get("size") is not None:
                ET.SubElement(font_el, qname(X_NS, "sz"),
                              {"val": repr(entry["size"])})
            if entry.get("color"):
                ET.SubElement(font_el, qname(X_NS, "color"),
                              {"rgb": entry["color"]})
            if entry.get("font_name"):
                ET.SubElement(font_el, qname(X_NS, "name"),
                              {"val": entry["font_name"]})
            else:
                ET.SubElement(font_el, qname(X_NS, "name"),
                              {"val": "Calibri"})
        fills = first_child(root, "fills")
        if fills is None:
            fills = ET.Element(qname(X_NS, "fills"), {"count": "0"})
            root.append(fills)
        for fill in list(fills):
            fills.remove(fill)
        fills.set("count", str(len(self._fills)))
        for fill_rgb in self._fills:
            fill_el = ET.SubElement(fills, qname(X_NS, "fill"))
            pattern = ET.SubElement(fill_el, qname(X_NS, "patternFill"),
                                    {"patternType":
                                     "solid" if fill_rgb else "none"})
            if fill_rgb:
                ET.SubElement(pattern, qname(X_NS, "fgColor"),
                              {"rgb": fill_rgb})
                ET.SubElement(pattern, qname(X_NS, "bgColor"),
                              {"rgb": fill_rgb})
        xfs = first_child(root, "cellXfs")
        if xfs is None:
            xfs = ET.Element(qname(X_NS, "cellXfs"), {"count": "0"})
            root.append(xfs)
        for xf in list(xfs):
            xfs.remove(xf)
        xfs.set("count", str(len(self._xfs)))
        for entry in self._xfs:
            attrs = {
                "numFmtId": str(entry["numFmtId"]),
                "fontId": str(entry["fontId"]),
                "fillId": str(entry["fillId"]),
                "borderId": "0", "xfId": "0",
            }
            if entry.get("applyNumberFormat"):
                attrs["applyNumberFormat"] = "1"
            ET.SubElement(xfs, qname(X_NS, "xf"), attrs)
        self.package.set_part("xl/styles.xml", serialize_xml(root))
        self._styles_dirty = False

    def _serialize_workbook(self) -> None:
        from xml.etree import ElementTree as ET

        root = self._workbook_root
        sheets_el = first_child(root, "sheets")
        if sheets_el is not None:
            for sheet_el in list(sheets_el):
                sheets_el.remove(sheet_el)
            for ws in self.sheets:
                ET.SubElement(sheets_el, qname(X_NS, "sheet"), {
                    "name": ws.name,
                    "sheetId": str(ws.sheet_id),
                    f"{{{R_NS}}}id": ws.rid,
                })
        names_el = first_child(root, "definedNames")
        if names_el is not None:
            root.remove(names_el)
        if self.defined_names:
            names_el = ET.Element(qname(X_NS, "definedNames"))
            for name, target in self.defined_names.items():
                el = ET.SubElement(names_el, qname(X_NS, "definedName"),
                                   {"name": name})
                el.text = target
            root.append(names_el)
        self.package.set_part("xl/workbook.xml", serialize_xml(root))
        self._workbook_dirty = False

    def _serialize_core(self) -> None:
        from xml.etree import ElementTree as ET

        from .ooxml import CP_NS, DC_NS

        root = self._core_root
        if root is None:
            root = ET.Element(qname(CP_NS, "coreProperties"))
            self._core_root = root
        for tag in ("title", "creator", "lastModifiedBy"):
            for child in list(root):
                if local(child.tag) == tag:
                    root.remove(child)
            if self.core_properties.get(tag):
                el = ET.SubElement(root, qname(DC_NS, tag))
                el.text = self.core_properties[tag]
        self.package.set_part("docProps/core.xml", serialize_xml(root))
        self._core_dirty = False

    # ── fidelity ──

    def fidelity_report(self, other_bytes: bytes,
                        touched: set[tuple[str, str]] | None = None
                        ) -> dict[str, Any]:
        """Compare this workbook against other bytes, part by part.

        ``touched`` names the (sheet, ref) cells an edit was allowed
        to change; every other difference is reported. Defaults to
        the touches recorded by the editor on this workbook.
        """
        if touched is None:
            touched = self._touched
        other = read_package(other_bytes)
        changed: list[str] = []
        total = 0
        identical = 0
        for name, data in self.package.parts.items():
            total += 1
            if other.get(name) == data:
                identical += 1
            else:
                changed.append(name)
        for name in other.part_names():
            if name not in self.package.parts:
                total += 1
                changed.append(name)
        other_wb = Workbook(other)
        untouched_changed = 0
        for ws in self.sheets:
            other_ws = other_wb.sheets_by_name.get(ws.name)
            refs = {(c.row, c.col) for c in ws.cells.values()}
            if other_ws is not None:
                refs |= {(c.row, c.col)
                         for c in other_ws.cells.values()}
            for row, col in refs:
                if (ws.name, a1_ref(row, col)) in touched:
                    continue
                mine = ws.cells.get((row, col))
                theirs = (other_ws.cells.get((row, col))
                          if other_ws is not None else None)
                mine_key = (mine.value, mine.formula) if mine else None
                theirs_key = (theirs.value, theirs.formula) \
                    if theirs else None
                if mine_key != theirs_key:
                    untouched_changed += 1
        return {
            "total_parts": total,
            "identical_parts": identical,
            "changed_parts": sorted(changed),
            "untouched_cells_changed": untouched_changed,
        }


def _value_text(value: Any) -> str:
    if isinstance(value, CellError):
        return value.display
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return repr(value)
    return str(value)


def _blank_sheet_xml() -> bytes:
    """A minimal blank worksheet part, for add_sheet."""
    xml = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n')
    body = (f'<worksheet xmlns="{X_NS}">'
            '<dimension ref="A1"/>'
            '<sheetViews><sheetView workbookViewId="0"/></sheetViews>'
            '<sheetFormatPr defaultRowHeight="15"/>'
            '<sheetData/>'
            '</worksheet>')
    return (xml + body).encode()


def _translate_shared(formula: str, drow: int, dcol: int) -> str:
    """Translate a shared-formula master by a row/col offset.

    Relative refs move; absolute (``$``) parts stay — the Excel
    shared-formula rule.
    """
    from .formula import format_ast, parse_formula, translate_ast

    try:
        # Stored formulas carry no '='; the parser requires it.
        node = parse_formula("=" + formula)
    except Exception:
        return formula
    return format_ast(translate_ast(node, drow, dcol))


def _blank_parts() -> dict[str, bytes]:
    """Minimal one-sheet workbook parts, as literal XML strings."""
    xml = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n')
    ct = ('<Types xmlns="http://schemas.openxmlformats.org/package/2006/'
          'content-types">'
          '<Default Extension="rels" ContentType="application/vnd.'
          'openxmlformats-package.relationships+xml"/>'
          '<Default Extension="xml" ContentType="application/xml"/>'
          '<Override PartName="/xl/workbook.xml" ContentType='
          '"application/vnd.openxmlformats-officedocument.'
          'spreadsheetml.sheet.main+xml"/>'
          '<Override PartName="/xl/worksheets/sheet1.xml" ContentType='
          '"application/vnd.openxmlformats-officedocument.'
          'spreadsheetml.worksheet+xml"/>'
          '<Override PartName="/xl/sharedStrings.xml" ContentType='
          '"application/vnd.openxmlformats-officedocument.'
          'spreadsheetml.sharedStrings+xml"/>'
          '<Override PartName="/xl/styles.xml" ContentType='
          '"application/vnd.openxmlformats-officedocument.'
          'spreadsheetml.styles+xml"/>'
          '<Override PartName="/docProps/core.xml" ContentType='
          '"application/vnd.openxmlformats-package.core-properties'
          '+xml"/>'
          '<Override PartName="/docProps/app.xml" ContentType='
          '"application/vnd.openxmlformats-officedocument.'
          'extended-properties+xml"/>'
          '</Types>')
    rels = ('<Relationships xmlns="http://schemas.openxmlformats.org/'
            'package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats'
            '.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats'
            '.org/package/2006/relationships/metadata/core-properties" '
            'Target="docProps/core.xml"/>'
            '</Relationships>')
    wb_rels = ('<Relationships xmlns="http://schemas.openxmlformats.org/'
               'package/2006/relationships">'
               '<Relationship Id="rId1" Type="http://schemas.'
               'openxmlformats.org/officeDocument/2006/relationships/'
               'worksheet" Target="worksheets/sheet1.xml"/>'
               '<Relationship Id="rId2" Type="http://schemas.'
               'openxmlformats.org/officeDocument/2006/relationships/'
               'sharedStrings" Target="sharedStrings.xml"/>'
               '<Relationship Id="rId3" Type="http://schemas.'
               'openxmlformats.org/officeDocument/2006/relationships/'
               'styles" Target="styles.xml"/>'
               '</Relationships>')
    workbook = (f'<workbook xmlns="{X_NS}" xmlns:r="{R_NS}">'
                '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/>'
                '</sheets></workbook>')
    sheet = (f'<worksheet xmlns="{X_NS}">'
             '<dimension ref="A1"/>'
             '<sheetViews><sheetView workbookViewId="0"/></sheetViews>'
             '<sheetFormatPr defaultRowHeight="15"/>'
             '<sheetData/>'
             '</worksheet>')
    shared = (f'<sst xmlns="{X_NS}" count="0" uniqueCount="0"/>')
    styles = (f'<styleSheet xmlns="{X_NS}">'
              '<fonts count="1"><font><sz val="11"/>'
              '<name val="Calibri"/></font></fonts>'
              '<fills count="2">'
              '<fill><patternFill patternType="none"/></fill>'
              '<fill><patternFill patternType="gray125"/></fill>'
              '</fills>'
              '<borders count="1"><border><left/><right/><top/>'
              '<bottom/><diagonal/></border></borders>'
              '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" '
              'fillId="0" borderId="0"/></cellStyleXfs>'
              '<cellXfs count="1"><xf numFmtId="0" fontId="0" '
              'fillId="0" borderId="0" xfId="0"/></cellXfs>'
              '<cellStyles count="1"><cellStyle name="Normal" xfId="0" '
              'builtinId="0"/></cellStyles></styleSheet>')
    core = ('<cp:coreProperties xmlns:cp="http://schemas.openxmlformats'
            '.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:dcterms="http://purl.org/dc/terms/">'
            '<dc:title></dc:title><dc:creator></dc:creator>'
            '</cp:coreProperties>')
    app = ('<Properties xmlns="http://schemas.openxmlformats.org/'
           'officeDocument/2006/extended-properties">'
           '<Application>Devon GridCraft</Application></Properties>')
    return {
        "[Content_Types].xml": (xml + ct).encode(),
        "_rels/.rels": (xml + rels).encode(),
        "xl/workbook.xml": (xml + workbook).encode(),
        "xl/_rels/workbook.xml.rels": (xml + wb_rels).encode(),
        "xl/worksheets/sheet1.xml": (xml + sheet).encode(),
        "xl/sharedStrings.xml": (xml + shared).encode(),
        "xl/styles.xml": (xml + styles).encode(),
        "docProps/core.xml": (xml + core).encode(),
        "docProps/app.xml": (xml + app).encode(),
    }
