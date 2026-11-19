"""Workbook editing for the GridCraft port — stdlib only.

``SheetEditor`` is the write path for ``nomorals/documents/sheets``:
set values and formulas, restyle cells, insert rows and columns
(formula refs shift with the insert, upstream's engine does the
same), add/rename/remove sheets, set metadata. Every edit marks
only the parts it touched dirty, so ``to_bytes`` re-serializes
only those parts and the rest of the package stays byte-identical
— the fidelity the tests measure.

Formulas are stored WITHOUT the leading ``=`` (the OOXML ``<f>``
form); ``set_formula`` accepts either. Insert shifting follows
the Excel rule: every ref at or after the insert point moves,
``$``-anchored or not — the anchor only matters when a formula is
*copied*. Defined names follow their cells the same way.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from ..errors import DocumentError
from .formula import format_ast, parse_formula, shift_ast
from .model import Cell, CellError, CellStyle, Worksheet, a1_ref, num_to_col, parse_a1, parse_range

__all__ = ["SheetEditor"]

#: A defined-name target that is a single ref or a range, with the
#: sheet qualifier optional and $ anchors preserved per coordinate.
_NAME_TARGET_RE = re.compile(
    r"^(?:(?P<sheet>'(?:[^']|'')+'|[A-Za-z_][A-Za-z0-9_.]*)!)?"
    r"(?P<ca>\$?)(?P<col>[A-Za-z]{1,3})(?P<ra>\$?)(?P<row>[0-9]+)"
    r"(?::(?P<ca2>\$?)(?P<col2>[A-Za-z]{1,3})(?P<ra2>\$?)"
    r"(?P<row2>[0-9]+))?$")

#: Core-property aliases the editor accepts.
_META_ALIASES = {"author": "creator"}


class SheetEditor:
    """Mutable view over a ``Workbook``; ``to_bytes`` saves it."""

    def __init__(self, workbook: Any) -> None:
        self.workbook = workbook

    # ── cell writes ──────────────────────────────────────────────

    def set_value(self, sheet: str, ref: str, value: Any) -> Cell:
        """Write a plain value (clears any formula on the cell)."""
        ws = self._sheet(sheet)
        row, col = parse_a1(ref)
        cell = self._cell(ws, row, col)
        cell.value = value
        cell.formula = None
        cell.ast = None
        cell.cached = None
        cell.kind = ("b" if isinstance(value, bool)
                     else "s" if isinstance(value, str)
                     else "e" if isinstance(value, CellError)
                     else "n")
        self._dirty(ws, sheet, ref)
        return cell

    def set_formula(self, sheet: str, ref: str, formula: str) -> Cell:
        """Write a formula (``=`` prefix optional); parses it now.

        A formula that does not parse raises ``ValueError`` and
        leaves the cell untouched — a bad formula never lands.
        Stored without the leading ``=`` (the OOXML ``<f>`` form).
        """
        ws = self._sheet(sheet)
        row, col = parse_a1(ref)
        ast = parse_formula(formula if formula.startswith("=")
                            else "=" + formula)
        cell = self._cell(ws, row, col)
        cell.value = None  # a formula cell holds no plain value;
        # its result lives in `cached` (a stale value here would
        # survive a round-trip as a phantom)
        cell.formula = formula[1:] if formula.startswith("=") \
            else formula
        cell.ast = ast
        cell.cached = None
        cell.kind = "n"
        self._dirty(ws, sheet, ref)
        return cell

    def restyle(self, sheet: str, ref: str, **kwargs: Any) -> Cell:
        """Restyle a cell: bold/italic/color/font_color/fill/
        number_format/font_size/font_name/horizontal/vertical/
        wrap_text as keyword arguments. ``color`` is the font color."""
        ws = self._sheet(sheet)
        row, col = parse_a1(ref)
        cell = self._cell(ws, row, col)
        changes = dict(kwargs)
        if "color" in changes:
            changes["font_color"] = changes.pop("color")
        known = set(CellStyle.__dataclass_fields__)
        unknown = sorted(set(changes) - known)
        if unknown:
            raise DocumentError(
                f"unknown style keys: {', '.join(unknown)}",
                kind="invalid_argument")
        style = replace(cell.style, **changes)
        cell.style = style
        cell.style_index = self.workbook.style_index_for(style)
        self._dirty(ws, sheet, ref)
        return cell

    # ── structural edits ─────────────────────────────────────────

    def insert_rows(self, sheet: str, at: int, count: int = 1) -> None:
        """Insert ``count`` blank rows before row ``at`` (1-based)."""
        if at < 1 or count < 1:
            raise DocumentError("insert_rows needs at >= 1, count >= 1",
                                kind="invalid_argument")
        ws = self._sheet(sheet)
        self._shift_cells(ws, at, count, rows=True)
        self._shift_formulas(sheet, at, count, rows=True)
        self._shift_merges(ws, at, count, rows=True)
        self._shift_defined_names(sheet, at, count, rows=True)
        self.workbook.mark_sheet_dirty(ws.part)

    def insert_cols(self, sheet: str, at: int | str,
                    count: int = 1) -> None:
        """Insert ``count`` blank columns before column ``at``.

        ``at`` may be a 1-based number or a column letter.
        """
        col = parse_a1(f"{at}1")[1] if isinstance(at, str) else at
        if col < 1 or count < 1:
            raise DocumentError("insert_cols needs at >= 1, count >= 1",
                                kind="invalid_argument")
        ws = self._sheet(sheet)
        self._shift_cells(ws, col, count, rows=False)
        self._shift_formulas(sheet, col, count, rows=False)
        self._shift_merges(ws, col, count, rows=False)
        self._shift_defined_names(sheet, col, count, rows=False)
        self.workbook.mark_sheet_dirty(ws.part)

    def add_sheet(self, name: str) -> Worksheet:
        """Add a blank sheet; the name must be new."""
        return self.workbook.add_sheet(name)

    def rename_sheet(self, old: str, new: str) -> Worksheet:
        """Rename a sheet and retarget every formula and defined
        name that points at it."""
        if new in self.workbook.sheet_names:
            raise DocumentError(f"sheet {new!r} already exists",
                                kind="invalid_argument")
        ws = self._sheet(old)
        for other in self.workbook.sheets:
            changed = False
            for cell in other.cells.values():
                if self._rewrite_cell(
                        cell, other.name,
                        lambda n: _rename_sheet(n, old, new)):
                    changed = True
            if changed:
                self.workbook.mark_sheet_dirty(other.part)
        for name, target in list(self.workbook.defined_names.items()):
            renamed = _rename_defined_target(target, old, new)
            if renamed != target:
                self.workbook.defined_names[name] = renamed
        del self.workbook.sheets_by_name[old]
        ws.name = new
        self.workbook.sheets_by_name[new] = ws
        self.workbook.mark_workbook_dirty()
        return ws

    def remove_sheet(self, name: str) -> None:
        """Remove a sheet; formulas elsewhere that read it go #REF!."""
        self._sheet(name)  # raises when the sheet is missing
        # Remove first: it raises on the last sheet before anything
        # else has been rewritten.
        self.workbook.remove_sheet(name)
        for other in self.workbook.sheets:
            changed = False
            for cell in other.cells.values():
                if self._rewrite_cell(cell, other.name,
                                      lambda n: _drop_sheet(n, name)):
                    changed = True
            if changed:
                self.workbook.mark_sheet_dirty(other.part)

    def set_metadata(self, **fields: Any) -> None:
        """Set core properties (title, author→creator, ...)."""
        props = self.workbook.core_properties
        for key, value in fields.items():
            target = _META_ALIASES.get(key, key)
            props[target] = str(value)
        self.workbook.mark_core_dirty()

    # ── output ───────────────────────────────────────────────────

    def to_bytes(self) -> bytes:
        """Serialize the edited workbook (dirty parts only)."""
        return self.workbook.to_bytes()

    def summary(self) -> dict[str, Any]:
        """Sheet names, sizes and formula counts, for replies."""
        return {
            "sheets": [
                {
                    "name": ws.name,
                    "rows": ws.max_row,
                    "columns": ws.max_column,
                    "cells": len(ws.cells),
                    "formulas": sum(
                        1 for c in ws.cells.values()
                        if c.formula is not None),
                    "merged": len(ws.merged_cells),
                }
                for ws in self.workbook.sheets
            ],
            "defined_names": dict(self.workbook.defined_names),
        }

    # ── internals ────────────────────────────────────────────────

    def _sheet(self, name: str) -> Worksheet:
        return self.workbook.sheet(name)

    def _cell(self, ws: Worksheet, row: int, col: int) -> Cell:
        cell = ws.cells.get((row, col))
        if cell is None:
            cell = Cell(ref=a1_ref(row, col), row=row, col=col)
            ws.cells[(row, col)] = cell
        return cell

    def _dirty(self, ws: Worksheet, sheet: str, ref: str) -> None:
        self.workbook.mark_sheet_dirty(ws.part)
        self.workbook.touch(sheet, ref)

    def _ast_of(self, cell: Cell) -> Any:
        """The cell's parsed AST, parsing lazily when needed.

        Parsed cells carry no AST until something evaluates them —
        an insert/rename must still shift their refs, so the AST is
        parsed here. A formula the engine cannot parse is left
        untouched (None) rather than rewritten wrong.
        """
        if cell.ast is not None:
            return cell.ast
        if cell.formula is None:
            return None
        try:
            cell.ast = parse_formula("=" + cell.formula)
        except Exception:
            return None
        return cell.ast

    def _rewrite_cell(self, cell: Cell, home: str, fn: Any) -> bool:
        """Apply a leaf rewrite to one formula cell, only writing
        back when the rendered formula actually changed."""
        ast = self._ast_of(cell)
        if ast is None or cell.formula is None:
            return False
        new_ast = _map_ast(ast, fn)
        new_text = format_ast(new_ast)
        if new_text != cell.formula:
            cell.ast = new_ast
            cell.formula = new_text
            cell.cached = None
            return True
        return False

    def _shift_cells(self, ws: Worksheet, at: int, count: int,
                     rows: bool) -> None:
        moved: dict[tuple[int, int], Cell] = {}
        for (row, col), cell in ws.cells.items():
            key = row if rows else col
            if key >= at:
                new_key = (row + count, col) if rows \
                    else (row, col + count)
                cell.row, cell.col = new_key
                cell.ref = a1_ref(*new_key)
                moved[new_key] = cell
                # Touch both ends: the new ref (the cell's new home)
                # and the old ref (now holding different content, or
                # nothing — the gap a missing predecessor leaves).
                self.workbook.touch(ws.name, cell.ref)
                self.workbook.touch(ws.name, a1_ref(row, col))
            else:
                moved[(row, col)] = cell
        ws.cells = moved
        # The vacated rows/columns are touched too — they were
        # emptied by the shift, so the fidelity report must not
        # count their old contents as "untouched changes".
        max_row, max_col = ws.max_row, ws.max_column
        for index in range(at, at + count):
            span = range(1, max_col + 1) if rows \
                else range(1, max_row + 1)
            for other in span:
                ref = a1_ref(index, other) if rows \
                    else a1_ref(other, index)
                self.workbook.touch(ws.name, ref)

    def _shift_formulas(self, sheet: str, at: int, count: int,
                        rows: bool) -> None:
        kwargs: dict[str, Any] = (
            {"rows_at": at, "rows_n": count} if rows
            else {"cols_at": at, "cols_n": count})
        for other in self.workbook.sheets:
            for cell in other.cells.values():
                if cell.formula is None:
                    continue
                ast = self._ast_of(cell)
                if ast is None:
                    continue
                before = cell.formula
                new_ast = _map_ast(
                    ast,
                    lambda n, home=other.name: (
                        shift_ast(n, **kwargs)
                        if _effective_sheet(n, home) == sheet else n))
                new_text = format_ast(new_ast)
                if new_text != before:
                    cell.ast = new_ast
                    cell.formula = new_text
                    cell.cached = None
                    self.workbook.touch(other.name, cell.ref)

    def _shift_merges(self, ws: Worksheet, at: int, count: int,
                      rows: bool) -> None:
        shifted: list[str] = []
        for ref in ws.merged_cells:
            r1, c1, r2, c2 = parse_range(ref)
            if rows:
                if r1 >= at:
                    r1 += count
                    r2 += count
                elif r2 >= at:
                    r2 += count
            else:
                if c1 >= at:
                    c1 += count
                    c2 += count
                elif c2 >= at:
                    c2 += count
            shifted.append(
                f"{a1_ref(r1, c1)}:{a1_ref(r2, c2)}"
                if (r1, c1) != (r2, c2) else a1_ref(r1, c1))
        ws.merged_cells = shifted

    def _shift_defined_names(self, sheet: str, at: int, count: int,
                             rows: bool) -> None:
        for name, target in list(self.workbook.defined_names.items()):
            shifted = _shift_defined_target(target, sheet, at, count,
                                            rows)
            if shifted != target:
                self.workbook.defined_names[name] = shifted
                self.workbook.mark_workbook_dirty()


# ── AST walking ────────────────────────────────────────────────────

def _effective_sheet(node: Any, home: str) -> str | None:
    """The sheet a Ref/Range node points at (its own, else home)."""
    return getattr(node, "sheet", None) or home


def _map_ast(node: Any, fn: Any) -> Any:
    """Rebuild an AST, applying ``fn`` to every Ref/Range leaf."""
    from .formula import Binary, Func, Let, Range, Ref, Unary

    if isinstance(node, (Ref, Range)):
        return fn(node)
    if isinstance(node, Func):
        return Func(node.name,
                    tuple(_map_ast(a, fn) for a in node.args))
    if isinstance(node, Let):
        return Let(
            tuple((name, _map_ast(value, fn))
                  for name, value in node.bindings),
            _map_ast(node.body, fn))
    if isinstance(node, Unary):
        return Unary(node.op, _map_ast(node.expr, fn))
    if isinstance(node, Binary):
        return Binary(node.op, _map_ast(node.left, fn),
                      _map_ast(node.right, fn))
    return node


def _rename_sheet(node: Any, old: str, new: str) -> Any:
    from .formula import Range, Ref

    if isinstance(node, Ref) and node.sheet == old:
        return Ref(new, node.row, node.col, node.row_abs, node.col_abs)
    if isinstance(node, Range) and node.sheet == old:
        return Range(new, node.row1, node.col1, node.row2, node.col2)
    return node


def _drop_sheet(node: Any, name: str) -> Any:
    """Replace refs to a removed sheet with ``#REF!``."""
    from .formula import Range, Ref

    if isinstance(node, Ref) and node.sheet == name:
        return Ref(None, 0, 0)
    if isinstance(node, Range) and node.sheet == name:
        return Ref(None, 0, 0)
    return node


# ── defined names ──────────────────────────────────────────────────

def _shift_defined_target(target: str, sheet: str, at: int,
                          count: int, rows: bool) -> str:
    """Shift a defined-name target for an insert, when it points at
    the inserted sheet. ``$`` anchors do not stop it — the named
    cell itself moved."""
    match = _NAME_TARGET_RE.match(target.strip())
    if not match:
        return target
    name = match.group("sheet")
    if name is not None:
        name = name.strip("'").replace("''", "'")
        if name != sheet:
            return target
    row = int(match.group("row"))
    col = _col_to_num(match.group("col"))
    row2 = match.group("row2")
    col2 = match.group("col2")
    if rows:
        if row >= at:
            row += count
        if row2 is not None and int(row2) >= at:
            row2 = str(int(row2) + count)
    else:
        if col >= at:
            col += count
        if col2 is not None and _col_to_num(col2) >= at:
            col2 = num_to_col(_col_to_num(col2) + count)
    prefix = f"{_quote_defined_sheet(name)}!" if name else ""
    out = (prefix + match.group("ca") + num_to_col(col)
           + match.group("ra") + str(row))
    if row2 is not None:
        out += (":" + (match.group("ca2") or "") + (col2 or "")
                + (match.group("ra2") or "") + (row2 or ""))
    return out


def _rename_defined_target(target: str, old: str, new: str) -> str:
    match = _NAME_TARGET_RE.match(target.strip())
    if not match or match.group("sheet") is None:
        return target
    name = match.group("sheet").strip("'").replace("''", "'")
    if name != old:
        return target
    rest = target[target.index("!") + 1:]
    return f"{_quote_defined_sheet(new)}!{rest}"


def _quote_defined_sheet(name: str | None) -> str:
    if name is None:
        return ""
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", name):
        return name
    return "'" + name.replace("'", "''") + "'"


def _col_to_num(letters: str) -> int:
    num = 0
    for char in letters.upper():
        num = num * 26 + (ord(char) - 64)
    return num
