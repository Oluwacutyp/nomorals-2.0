"""Formula engine for the GridCraft port — stdlib only.

Port of the *design* of upstream ``crates/formula`` (lexer, Pratt
parser, AST with anchored refs and sheet selectors) + ``crates/calc``
(evaluation) + ``crates/functions`` (the function library): a real
Excel-style formula language over Devon's workbook model.

Language: ``=1+2*3``, cell refs (``A1``, ``$A$1``), ranges
(``A1:B5``), whole columns (``A:C``) and rows (``1:3``), cross-sheet
refs (``Sheet2!A1``, ``'My Sheet'!A1``), defined names, strings,
numbers, ``%``, ``^``, ``&``, comparisons, and functions. Errors
are values, never exceptions: the taxonomy mirrors upstream
``CellError`` (``crates/core/src/value.rs``) — ``#DIV/0!``,
``#VALUE!``, ``#REF!``, ``#NAME?``, ``#NUM!``, ``#N/A``,
``#NULL!``, ``#SPILL!``, ``#CALC!`` and the upstream's own
``#CIRC!`` for a circular reference (Excel shows 0; upstream keeps
it explicit, and so do we).

Not yet: array literals (``{1,2;3,4}``), 3-D refs
(``Sheet1:Sheet3!A1``), ``LAMBDA``, ``OFFSET``/``CELL`` — the
parser refuses them with a named error rather than guessing.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from .model import CellError, date_to_serial, serial_to_date

__all__ = [
    "CellError",
    "FormulaSyntaxError",
    "RangeValue",
    "date_to_serial",
    "dependencies",
    "evaluate",
    "format_ast",
    "parse_formula",
    "serial_to_date",
    "shift_ast",
    "translate_ast",
]

# ── errors ───────────────────────────────────────────────────────────


class FormulaSyntaxError(ValueError):
    """A formula string could not be parsed."""


# ── AST ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Num:
    value: float


@dataclass(frozen=True)
class Str:
    value: str


@dataclass(frozen=True)
class Bool:
    value: bool


@dataclass(frozen=True)
class Ref:
    sheet: str | None  # None = the formula's own sheet
    row: int
    col: int
    row_abs: bool = False
    col_abs: bool = False


@dataclass(frozen=True)
class Range:
    sheet: str | None
    row1: int
    col1: int
    row2: int
    col2: int


@dataclass(frozen=True)
class Name:
    name: str


@dataclass(frozen=True)
class Func:
    name: str
    args: tuple[Any, ...]


@dataclass(frozen=True)
class Unary:
    op: str
    expr: Any


@dataclass(frozen=True)
class Binary:
    op: str
    left: Any
    right: Any


@dataclass(frozen=True)
class Let:
    bindings: tuple[tuple[str, Any], ...]
    body: Any


# ── lexer ────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"""
    (?P<space>\s+)
  | (?P<number>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<string>"(?:[^"]|"")*")
  | (?P<sheet>'(?:[^']|'')+'!)
  | (?P<op><>|<=|>=|[+\-*/^%&=(),:<>!])
  | (?P<word>[A-Za-z_\\][A-Za-z0-9_.\\]*)
  | (?P<dollar>\$)
""", re.VERBOSE)

_REF_RE = re.compile(r"^\$?([A-Za-z]{1,3})\$?([0-9]+)$")
_COL_RE = re.compile(r"^\$?([A-Za-z]{1,3})$")
_ROW_RE = re.compile(r"^\$?([0-9]+)$")


@dataclass
class Token:
    kind: str
    text: str


def _tokenize(src: str) -> list[Token]:
    tokens: list[Token] = []
    pos = 0
    while pos < len(src):
        match = _TOKEN_RE.match(src, pos)
        if not match:
            raise FormulaSyntaxError(
                f"cannot tokenize formula at {src[pos:pos + 10]!r}")
        pos = match.end()
        kind = match.lastgroup or ""
        text = match.group()
        if kind == "space":
            continue
        tokens.append(Token(kind, text))
    tokens.append(Token("eof", ""))
    return tokens


def _col_to_num(letters: str) -> int:
    num = 0
    for char in letters.upper():
        num = num * 26 + (ord(char) - 64)
    return num


def _num_to_col(num: int) -> str:
    letters = ""
    while num:
        num, rem = divmod(num - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


# ── parser (Pratt) ───────────────────────────────────────────────────

#: precedence, low to high; ``^`` is right-associative.
_PRECEDENCE = {
    "=": 1, "<>": 1, "<": 1, "<=": 1, ">": 1, ">=": 1,
    "&": 2,
    "+": 3, "-": 3,
    "*": 4, "/": 4,
    "^": 5,
}


class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def next(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, text: str) -> Token:
        tok = self.next()
        if tok.text != text:
            raise FormulaSyntaxError(
                f"expected {text!r} in formula, got {tok.text!r}")
        return tok

    def parse(self) -> Any:
        node = self.parse_expr(0)
        if self.peek().kind != "eof":
            raise FormulaSyntaxError(
                f"unexpected {self.peek().text!r} at end of formula")
        return node

    def parse_expr(self, min_prec: int) -> Any:
        left = self.parse_unary()
        while True:
            tok = self.peek()
            if tok.kind == "op" and tok.text == "%":
                self.next()
                left = Binary("/", left, Num(100.0))
                continue
            prec = _PRECEDENCE.get(tok.text, 0) if tok.kind == "op" else 0
            if prec == 0 or prec < min_prec:
                break
            self.next()
            # ``^`` is right-associative.
            rhs = self.parse_expr(prec + 1 if tok.text != "^" else prec)
            left = Binary(tok.text, left, rhs)
        return left

    def parse_unary(self) -> Any:
        tok = self.peek()
        if tok.kind == "op" and tok.text in ("-", "+"):
            self.next()
            operand = self.parse_unary()
            if tok.text == "-":
                return Unary("-", operand)
            return operand
        return self.parse_primary()

    def parse_primary(self) -> Any:
        tok = self.next()
        if tok.kind == "number":
            return Num(float(tok.text))
        if tok.kind == "string":
            return Str(tok.text[1:-1].replace('""', '"'))
        if tok.kind == "op" and tok.text == "(":
            node = self.parse_expr(0)
            self.expect(")")
            return node
        if tok.kind == "sheet":
            # 'Quoted Sheet'!A1 — the ref follows.
            sheet = tok.text[1:-2].replace("''", "'")
            return self.parse_ref(sheet)
        if tok.kind == "dollar":
            # $A1 or $A$1 — the lexer splits the $ signs out.
            word = self.next()
            if word.kind != "word":
                raise FormulaSyntaxError("expected a cell reference")
            match = _REF_RE.match(word.text)
            if match:
                # $A1: column anchored, row free.
                node: Any = Ref(None, int(match.group(2)),
                                _col_to_num(match.group(1)), False, True)
            else:
                if not _COL_RE.match(word.text):
                    raise FormulaSyntaxError(
                        f"not a cell reference: ${word.text!r}")
                row_abs = False
                if self.peek().text == "$":
                    self.next()
                    row_abs = True
                num = self.next()
                if num.kind != "number":
                    raise FormulaSyntaxError(
                        "expected a row number in a cell reference")
                node = Ref(None, int(float(num.text)),
                           _col_to_num(word.text), row_abs, True)
            return self._maybe_range(None, node)
        if tok.kind == "word":
            word = tok.text
            if self.peek().text == "!":
                self.next()
                return self.parse_ref(word)
            if self.peek().text == "(":
                return self.parse_call(word)
            upper = word.upper()
            if upper == "TRUE":
                return Bool(True)
            if upper == "FALSE":
                return Bool(False)
            ref = _REF_RE.match(word)
            if ref:
                node = Ref(None, int(ref.group(2)),
                           _col_to_num(ref.group(1)), False, False)
                return self._maybe_range(None, node)
            if _COL_RE.match(word) and self.peek().text == "$":
                # A$1 — row anchored, column free.
                self.next()
                num = self.next()
                if num.kind != "number":
                    raise FormulaSyntaxError(
                        "expected a row number in a cell reference")
                node = Ref(None, int(float(num.text)),
                           _col_to_num(word), True, False)
                return self._maybe_range(None, node)
            if _COL_RE.match(word) and self.peek().text == ":":
                # A:C — whole columns.
                self.next()
                end = self.next()
                end_col = _COL_RE.match(end.text)
                if not end_col:
                    raise FormulaSyntaxError(
                        f"bad column range end {end.text!r}")
                return Range(None, 1, _col_to_num(word),
                             1_048_576, _col_to_num(end_col.group(1)))
            return Name(word)
        raise FormulaSyntaxError(f"unexpected {tok.text!r} in formula")

    def _maybe_range(self, sheet: str | None, first: Ref) -> Any:
        if self.peek().text != ":":
            return first
        self.next()
        return self._finish_range(sheet, first)

    def _finish_range(self, sheet: str | None, first: Ref) -> Any:
        # The range end may carry $ anchors ($B$2) or be a bare
        # column (A:C) or row (1:3).
        col_abs = row_abs = False
        if self.peek().kind == "dollar":
            self.next()
            col_abs = True
        tok = self.next()
        if tok.kind == "word":
            match = _REF_RE.match(tok.text)
            if match:
                if self.peek().kind == "dollar":
                    self.next()
                    row_abs = True
                    num = self.next()
                    if num.kind != "number":
                        raise FormulaSyntaxError(
                            "expected a row number in a range end")
                    end_row = int(float(num.text))
                else:
                    end_row = int(match.group(2))
                end = Ref(sheet, end_row, _col_to_num(match.group(1)),
                          row_abs, col_abs)
                return Range(sheet, first.row, first.col,
                             end.row, end.col)
            col = _COL_RE.match(tok.text)
            if col and self.peek().kind == "dollar":
                # $B$2 — a column-anchored cell ref as the range end.
                self.next()
                num = self.next()
                if num.kind != "number":
                    raise FormulaSyntaxError(
                        "expected a row number in a range end")
                return Range(sheet, first.row, first.col,
                             int(float(num.text)),
                             _col_to_num(col.group(1)))
            if col:
                return Range(sheet, 1, first.col, 1_048_576,
                             _col_to_num(col.group(1)))
            raise FormulaSyntaxError(
                f"bad range end {tok.text!r} in formula")
        if tok.kind == "number":
            return Range(sheet, first.row, first.col, int(float(tok.text)),
                         first.col)
        raise FormulaSyntaxError(
            f"bad range end {tok.text!r} in formula")

    def parse_ref(self, sheet: str) -> Any:
        # Sheet!A1, Sheet!$A$1, Sheet!A$1, Sheet!$A1 — the $ signs
        # arrive as separate tokens.
        col_abs = row_abs = False
        if self.peek().kind == "dollar":
            self.next()
            col_abs = True
        tok = self.next()
        if tok.kind == "word":
            match = _REF_RE.match(tok.text)
            if match:
                if self.peek().kind == "dollar":
                    self.next()
                    row_abs = True
                    num = self.next()
                    if num.kind != "number":
                        raise FormulaSyntaxError(
                            "expected a row number in a cell reference")
                    row = int(float(num.text))
                else:
                    row = int(match.group(2))
                node = Ref(sheet, row, _col_to_num(match.group(1)),
                           row_abs, col_abs)
                return self._maybe_range(sheet, node)
            if _COL_RE.match(tok.text) and self.peek().kind == "dollar":
                # Sheet!A$1 — row anchored, column free.
                self.next()
                num = self.next()
                if num.kind != "number":
                    raise FormulaSyntaxError(
                        "expected a row number in a cell reference")
                node = Ref(sheet, int(float(num.text)),
                           _col_to_num(tok.text), True, col_abs)
                return self._maybe_range(sheet, node)
            col = _COL_RE.match(tok.text)
            if col and self.peek().text == ":":
                # Sheet!A:C — whole columns.
                self.next()
                end = self.next()
                end_col = _COL_RE.match(end.text)
                if not end_col:
                    raise FormulaSyntaxError(
                        f"bad column range end {end.text!r}")
                return Range(sheet, 1, _col_to_num(col.group(1)),
                             1_048_576, _col_to_num(end_col.group(1)))
            raise FormulaSyntaxError(
                f"expected a cell reference after {sheet}!, "
                f"got {tok.text!r}")
        raise FormulaSyntaxError(
            f"expected a cell reference after {sheet}!, "
            f"got {tok.text!r}")

    def parse_call(self, name: str) -> Any:
        self.expect("(")
        args: list[Any] = []
        if self.peek().text != ")":
            while True:
                args.append(self.parse_expr(0))
                if self.peek().text == ",":
                    self.next()
                    continue
                break
        self.expect(")")
        if name.upper() == "LET":
            if len(args) < 3 or len(args) % 2 == 0:
                raise FormulaSyntaxError(
                    "LET needs name/value pairs plus a body")
            bindings: list[tuple[str, Any]] = []
            for i in range(0, len(args) - 1, 2):
                if not isinstance(args[i], Name):
                    raise FormulaSyntaxError(
                        "LET binding names must be plain names")
                bindings.append((args[i].name, args[i + 1]))
            return Let(tuple(bindings), args[-1])
        return Func(name.upper(), tuple(args))


def parse_formula(text: str) -> Any:
    """Parse a formula string (``=`` prefix REQUIRED) into an AST.

    Raises :class:`FormulaSyntaxError` on bad syntax. Array
    literals, 3-D refs and ``LAMBDA`` are refused by name. Stored
    formulas (no ``=`` — the OOXML ``<f>`` form) go through
    :func:`evaluate`, which adds the prefix; the parser itself is
    strict so a bare expression never parses as a formula by
    accident.
    """
    if not isinstance(text, str) or not text.strip():
        raise FormulaSyntaxError("empty formula")
    src = text.strip()
    if not src.startswith("="):
        raise FormulaSyntaxError(
            "a formula must start with '='")
    src = src[1:]
    if not src:
        raise FormulaSyntaxError("empty formula")
    if "{" in src or "}" in src:
        raise FormulaSyntaxError(
            "array literals {…} are not supported by this engine")
    if re.search(r"[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_.]*!",
                 src):
        raise FormulaSyntaxError(
            "3-D sheet references (Sheet1:Sheet3!A1) are not supported")
    return _Parser(_tokenize(src)).parse()


# ── printing ─────────────────────────────────────────────────────────


def format_ast(node: Any) -> str:
    """Render an AST back to formula text (no leading ``=``)."""
    if isinstance(node, Num):
        value = node.value
        return str(int(value)) if float(value).is_integer() else repr(value)
    if isinstance(node, Str):
        return '"' + node.value.replace('"', '""') + '"'
    if isinstance(node, Bool):
        return "TRUE" if node.value else "FALSE"
    if isinstance(node, Ref):
        if node.row < 1 or node.col < 1:
            return "#REF!"
        col = ("$" if node.col_abs else "") + _num_to_col(node.col)
        row = ("$" if node.row_abs else "") + str(node.row)
        prefix = f"{_quote_sheet(node.sheet)}!" if node.sheet else ""
        return prefix + col + row
    if isinstance(node, Range):
        def ref(row: int, col: int) -> str:
            return _num_to_col(col) + str(row)
        prefix = f"{_quote_sheet(node.sheet)}!" if node.sheet else ""
        return (prefix + ref(node.row1, node.col1) + ":"
                + ref(node.row2, node.col2))
    if isinstance(node, Name):
        return node.name
    if isinstance(node, Func):
        return (node.name + "("
                + ",".join(format_ast(a) for a in node.args) + ")")
    if isinstance(node, Let):
        parts: list[str] = []
        for name, value in node.bindings:
            parts.append(name)
            parts.append(format_ast(value))
        parts.append(format_ast(node.body))
        return "LET(" + ",".join(parts) + ")"
    if isinstance(node, Unary):
        return node.op + format_ast(node.expr)
    if isinstance(node, Binary):
        return (format_ast(node.left) + node.op
                + format_ast(node.right))
    raise FormulaSyntaxError(f"cannot print node {node!r}")


def _quote_sheet(name: str | None) -> str:
    if name is None:
        return ""
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", name):
        return name
    return "'" + name.replace("'", "''") + "'"


def dependencies(node: Any) -> set[tuple[str | None, int, int, int, int]]:
    """Refs/ranges an AST reads: ``(sheet, r1, c1, r2, c2)`` tuples."""
    found: set[tuple[str | None, int, int, int, int]] = set()

    def walk(item: Any) -> None:
        if isinstance(item, Ref):
            found.add((item.sheet, item.row, item.col, item.row,
                       item.col))
        elif isinstance(item, Range):
            found.add((item.sheet, item.row1, item.col1, item.row2,
                       item.col2))
        elif isinstance(item, Func):
            for arg in item.args:
                walk(arg)
        elif isinstance(item, Let):
            for _, value in item.bindings:
                walk(value)
            walk(item.body)
        elif isinstance(item, (Unary,)):
            walk(item.expr)
        elif isinstance(item, Binary):
            walk(item.left)
            walk(item.right)

    walk(node)
    return found


def shift_ast(node: Any, *, rows_at: int | None = None,
              rows_n: int = 0, cols_at: int | None = None,
              cols_n: int = 0) -> Any:
    """Shift refs in an AST for an inserted row/column block.

    The Excel insert rule: EVERY ref at or after the insertion
    point moves, ``$``-anchored or not — the anchor only matters
    when a formula is *copied* (see :func:`translate_ast`), never
    when rows/columns are inserted. Ranges that straddle the line
    grow at the far end instead of moving.
    """
    def shift_ref(ref: Ref) -> Ref:
        row, col = ref.row, ref.col
        if rows_at is not None and rows_n and row >= rows_at:
            row += rows_n
        if cols_at is not None and cols_n and col >= cols_at:
            col += cols_n
        if (row, col) == (ref.row, ref.col):
            return ref
        return Ref(ref.sheet, row, col, ref.row_abs, ref.col_abs)

    def shift_range(rng: Range) -> Range:
        row1, row2, col1, col2 = rng.row1, rng.row2, rng.col1, rng.col2
        if rows_at is not None and rows_n:
            if row1 >= rows_at:
                row1 += rows_n
            if row2 >= rows_at:
                row2 += rows_n
        if cols_at is not None and cols_n:
            if col1 >= cols_at:
                col1 += cols_n
            if col2 >= cols_at:
                col2 += cols_n
        return Range(rng.sheet, row1, col1, row2, col2)

    if isinstance(node, Ref):
        return shift_ref(node)
    if isinstance(node, Range):
        return shift_range(node)
    if isinstance(node, Func):
        return Func(node.name,
                    tuple(shift_ast(a, rows_at=rows_at, rows_n=rows_n,
                                    cols_at=cols_at, cols_n=cols_n)
                          for a in node.args))
    if isinstance(node, Let):
        return Let(
            tuple((name, shift_ast(v, rows_at=rows_at, rows_n=rows_n,
                                   cols_at=cols_at, cols_n=cols_n))
                  for name, v in node.bindings),
            shift_ast(node.body, rows_at=rows_at, rows_n=rows_n,
                      cols_at=cols_at, cols_n=cols_n))
    if isinstance(node, Unary):
        return Unary(node.op, shift_ast(node.expr, rows_at=rows_at,
                                         rows_n=rows_n, cols_at=cols_at,
                                         cols_n=cols_n))
    if isinstance(node, Binary):
        return Binary(node.op,
                      shift_ast(node.left, rows_at=rows_at,
                                rows_n=rows_n, cols_at=cols_at,
                                cols_n=cols_n),
                      shift_ast(node.right, rows_at=rows_at,
                                rows_n=rows_n, cols_at=cols_at,
                                cols_n=cols_n))
    return node


def translate_ast(node: Any, drow: int, dcol: int) -> Any:
    """Translate every non-absolute ref in an AST by (drow, dcol).

    The shared-formula rule: a formula copied down/right moves its
    relative refs with it; ``$``-anchored parts stay put.
    """
    if isinstance(node, Ref):
        row = node.row if node.row_abs else node.row + drow
        col = node.col if node.col_abs else node.col + dcol
        return Ref(node.sheet, row, col, node.row_abs, node.col_abs)
    if isinstance(node, Range):
        return Range(node.sheet,
                     node.row1 + drow, node.col1 + dcol,
                     node.row2 + drow, node.col2 + dcol)
    if isinstance(node, Func):
        return Func(node.name,
                    tuple(translate_ast(a, drow, dcol)
                          for a in node.args))
    if isinstance(node, Let):
        return Let(
            tuple((name, translate_ast(v, drow, dcol))
                  for name, v in node.bindings),
            translate_ast(node.body, drow, dcol))
    if isinstance(node, Unary):
        return Unary(node.op, translate_ast(node.expr, drow, dcol))
    if isinstance(node, Binary):
        return Binary(node.op, translate_ast(node.left, drow, dcol),
                      translate_ast(node.right, drow, dcol))
    return node


# ── values ───────────────────────────────────────────────────────────


@dataclass
class RangeValue:
    """A range read as values: rows of cell values (None = blank)."""
    rows: list[list[Any]]
    sheet: str | None = None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _to_number(value: Any) -> float | CellError:
    if _is_number(value):
        return float(value)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        text = value.strip()
        try:
            return float(text)
        except ValueError:
            return CellError.VALUE
    if value is None:
        return 0.0
    return CellError.VALUE


def _to_bool(value: Any) -> bool | CellError:
    if isinstance(value, bool):
        return value
    if _is_number(value):
        return value != 0
    if isinstance(value, str):
        upper = value.strip().upper()
        if upper == "TRUE":
            return True
        if upper == "FALSE":
            return False
    return CellError.VALUE


def _to_text(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if _is_number(value):
        if float(value).is_integer():
            return str(int(value))
        return repr(value)
    if value is None:
        return ""
    if isinstance(value, CellError):
        return value.display
    return str(value)


def _first_error(values: list[Any]) -> CellError | None:
    for value in values:
        if isinstance(value, CellError):
            return value
    return None


def _flatten(values: list[Any]) -> list[Any]:
    flat: list[Any] = []
    for value in values:
        if isinstance(value, RangeValue):
            for row in value.rows:
                flat.extend(row)
        else:
            flat.append(value)
    return flat


def _numbers(values: list[Any]) -> tuple[list[float], CellError | None]:
    nums: list[float] = []
    for value in values:
        if value is None or isinstance(value, str):
            continue
        if isinstance(value, CellError):
            return [], value
        if isinstance(value, bool):
            continue
        nums.append(float(value))
    return nums, None


# ── evaluator ────────────────────────────────────────────────────────


class _Env:
    """Evaluation context: workbook, current sheet, cycle guard."""

    def __init__(self, workbook: Any, sheet: str,
                 at: str | None = None) -> None:
        self.workbook = workbook
        self.sheet = sheet
        self.visiting: set[tuple[str, int, int]] = set()
        self.names: dict[str, Any] = {}
        self.at_row = 1
        self.at_col = 1
        if at:
            match = _REF_RE.match(at.strip().upper())
            if match:
                self.at_row = int(match.group(2))
                self.at_col = _col_to_num(match.group(1))

    def cell_value(self, sheet: str, row: int, col: int) -> Any:
        """The computed value of one cell, recursing into formulas."""
        key = (sheet, row, col)
        if row < 1 or col < 1:
            # A (0, 0) ref is the AST form of #REF! (a removed sheet).
            return CellError.REF
        if key in self.visiting:
            return CellError.CIRC
        ws = self.workbook.sheets_by_name.get(sheet)
        if ws is None:
            return CellError.REF
        cell = ws.cells.get((row, col))
        if cell is None:
            return None
        if cell.formula is None:
            return cell.value
        if cell.cached is not None:
            return cell.cached
        if cell.ast is None:
            # Parsed cells carry no AST yet — parse lazily; a stored
            # formula that does not parse reads as #NAME?.
            try:
                cell.ast = parse_formula("=" + (cell.formula or ""))
            except FormulaSyntaxError:
                cell.cached = CellError.NAME
                return cell.cached
        self.visiting.add(key)
        try:
            value = self.eval_node(cell.ast)
        finally:
            self.visiting.discard(key)
        cell.cached = value
        return value

    def range_value(self, sheet: str | None, row1: int, col1: int,
                    row2: int, col2: int) -> RangeValue:
        name = sheet or self.sheet
        rows: list[list[Any]] = []
        for row in range(min(row1, row2), max(row1, row2) + 1):
            out: list[Any] = []
            for col in range(min(col1, col2), max(col1, col2) + 1):
                out.append(self.cell_value(name, row, col))
            rows.append(out)
        return RangeValue(rows, sheet=name)

    def eval_node(self, node: Any) -> Any:
        if isinstance(node, Num):
            value = node.value
            return int(value) if float(value).is_integer() else value
        if isinstance(node, Str):
            return node.value
        if isinstance(node, Bool):
            return node.value
        if isinstance(node, Ref):
            sheet = node.sheet or self.sheet
            return self.cell_value(sheet, node.row, node.col)
        if isinstance(node, Range):
            return self.range_value(node.sheet, node.row1, node.col1,
                                    node.row2, node.col2)
        if isinstance(node, Name):
            return self._resolve_name(node.name)
        if isinstance(node, Let):
            saved = dict(self.names)
            for name, value_node in node.bindings:
                self.names[name.upper()] = self.eval_node(value_node)
            try:
                return self.eval_node(node.body)
            finally:
                self.names = saved
        if isinstance(node, Unary):
            value = self.eval_node(node.expr)
            if isinstance(value, CellError):
                return value
            if node.op == "-":
                num = _to_number(value)
                return -num if not isinstance(num, CellError) else num
            return value
        if isinstance(node, Binary):
            return self._binary(node)
        if isinstance(node, Func):
            return self._call(node)
        raise FormulaSyntaxError(f"cannot evaluate node {node!r}")

    def _resolve_name(self, name: str) -> Any:
        upper = name.upper()
        if upper in self.names:
            return self.names[upper]
        target = self.workbook.defined_names.get(name)
        if target is None:
            # Case-insensitive fallback for defined names.
            for key, value in self.workbook.defined_names.items():
                if key.upper() == upper:
                    target = value
                    break
        if target is None:
            return CellError.NAME
        try:
            # Defined-name targets are stored without '='.
            node = parse_formula("=" + target)
        except FormulaSyntaxError:
            return CellError.NAME
        return self.eval_node(node)

    def _binary(self, node: Binary) -> Any:
        left = self.eval_node(node.left)
        if isinstance(left, CellError):
            return left
        right = self.eval_node(node.right)
        if isinstance(right, CellError):
            return right
        op = node.op
        if op == "&":
            return _to_text(left) + _to_text(right)
        if op in ("=", "<>", "<", "<=", ">", ">="):
            return _compare(left, right, op)
        if op == "%":
            return left
        left_num = _to_number(left)
        if isinstance(left_num, CellError):
            return left_num
        right_num = _to_number(right)
        if isinstance(right_num, CellError):
            return right_num
        if op == "+":
            return _num(left_num + right_num)
        if op == "-":
            return _num(left_num - right_num)
        if op == "*":
            return _num(left_num * right_num)
        if op == "/":
            if right_num == 0:
                return CellError.DIV0
            return _num(left_num / right_num)
        if op == "^":
            try:
                return _num(left_num ** right_num)
            except (ValueError, OverflowError, ZeroDivisionError):
                return CellError.NUM
        return CellError.VALUE

    def _call(self, node: Func) -> Any:
        name = node.name
        if name == "LAMBDA":
            return CellError.NAME
        handler = _FUNCTIONS.get(name)
        if handler is None:
            return CellError.NAME
        if name in _LAZY:
            return handler(self, node.args)
        args: list[Any] = []
        for arg in node.args:
            value = self.eval_node(arg)
            if isinstance(value, CellError) \
                    and name not in _ERROR_AWARE:
                return value
            args.append(value)
        return handler(self, args)


def _num(value: float) -> int | float:
    return int(value) if float(value).is_integer() else value


def _type_rank(value: Any) -> int:
    if _is_number(value):
        return 0
    if isinstance(value, str):
        return 1
    if isinstance(value, bool):
        return 2
    return 3


def _compare(left: Any, right: Any, op: str) -> bool | CellError:
    if isinstance(left, RangeValue):
        left = left.rows[0][0] if left.rows and left.rows[0] else None
    if isinstance(right, RangeValue):
        right = right.rows[0][0] if right.rows and right.rows[0] else None
    if _is_number(left) and _is_number(right):
        a, b = float(left), float(right)
    elif isinstance(left, str) and isinstance(right, str):
        a, b = left.lower(), right.lower()
    elif isinstance(left, bool) and isinstance(right, bool):
        a, b = left, right
    elif _type_rank(left) != _type_rank(right):
        # Excel orders types: numbers < text < booleans.
        if op == "=":
            return False
        if op == "<>":
            return True
        a, b = _type_rank(left), _type_rank(right)
    else:
        a, b = left, right
    if op == "=":
        return a == b
    if op == "<>":
        return a != b
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == ">":
        return a > b
    return a >= b


# ── function library ─────────────────────────────────────────────────


def _f_sum(env: _Env, args: list[Any]) -> Any:
    nums, err = _numbers(_flatten(args))
    if err is not None:
        return err
    return _num(sum(nums))


def _f_average(env: _Env, args: list[Any]) -> Any:
    nums, err = _numbers(_flatten(args))
    if err is not None:
        return err
    if not nums:
        return CellError.DIV0
    return _num(sum(nums) / len(nums))


def _f_count(env: _Env, args: list[Any]) -> Any:
    nums, err = _numbers(_flatten(args))
    if err is not None:
        return err
    return len(nums)


def _f_counta(env: _Env, args: list[Any]) -> Any:
    err = _first_error(_flatten(args))
    if err is not None:
        return err
    return sum(1 for v in _flatten(args) if v is not None)


def _f_countblank(env: _Env, args: list[Any]) -> Any:
    err = _first_error(_flatten(args))
    if err is not None:
        return err
    return sum(1 for v in _flatten(args) if v is None)


def _f_min(env: _Env, args: list[Any]) -> Any:
    nums, err = _numbers(_flatten(args))
    if err is not None:
        return err
    return _num(min(nums)) if nums else 0


def _f_max(env: _Env, args: list[Any]) -> Any:
    nums, err = _numbers(_flatten(args))
    if err is not None:
        return err
    return _num(max(nums)) if nums else 0


def _f_product(env: _Env, args: list[Any]) -> Any:
    nums, err = _numbers(_flatten(args))
    if err is not None:
        return err
    out = 1.0
    for num in nums:
        out *= num
    return _num(out)


def _f_sumproduct(env: _Env, args: list[Any]) -> Any:
    if not args:
        return CellError.VALUE
    flat = [_flatten([a]) for a in args]
    width = len(flat[0])
    if any(len(row) != width for row in flat):
        return CellError.VALUE
    total = 0.0
    for i in range(width):
        product = 1.0
        for row in flat:
            num = _to_number(row[i])
            if isinstance(num, CellError):
                return num
            product *= num
        total += product
    return _num(total)


def _f_median(env: _Env, args: list[Any]) -> Any:
    nums, err = _numbers(_flatten(args))
    if err is not None:
        return err
    if not nums:
        return CellError.NUM
    nums.sort()
    mid = len(nums) // 2
    if len(nums) % 2:
        return _num(nums[mid])
    return _num((nums[mid - 1] + nums[mid]) / 2)


def _f_stdev(env: _Env, args: list[Any], population: bool) -> Any:
    nums, err = _numbers(_flatten(args))
    if err is not None:
        return err
    if len(nums) < (1 if population else 2):
        return CellError.DIV0
    mean = sum(nums) / len(nums)
    var = sum((n - mean) ** 2 for n in nums)
    var /= len(nums) if population else len(nums) - 1
    return _num(math.sqrt(var))


def _f_large(env: _Env, args: list[Any]) -> Any:
    nums, err = _numbers(_flatten(args[:-1]))
    if err is not None:
        return err
    k = _to_number(args[-1])
    if isinstance(k, CellError):
        return k
    if not nums or int(k) < 1 or int(k) > len(nums):
        return CellError.NUM
    return _num(sorted(nums, reverse=True)[int(k) - 1])


def _f_small(env: _Env, args: list[Any]) -> Any:
    nums, err = _numbers(_flatten(args[:-1]))
    if err is not None:
        return err
    k = _to_number(args[-1])
    if isinstance(k, CellError):
        return k
    if not nums or int(k) < 1 or int(k) > len(nums):
        return CellError.NUM
    return _num(sorted(nums)[int(k) - 1])


def _f_rank(env: _Env, args: list[Any]) -> Any:
    num = _to_number(args[0])
    if isinstance(num, CellError):
        return num
    nums, err = _numbers(_flatten(args[1:2]))
    if err is not None:
        return err
    order = 0
    if len(args) > 2:
        order_num = _to_number(args[2])
        if isinstance(order_num, CellError):
            return order_num
        order = int(order_num)
    ordered = sorted(nums, reverse=(order == 0))
    try:
        return ordered.index(num) + 1
    except ValueError:
        return CellError.NA


def _one_num(name: str, func) -> Any:
    def handler(env: _Env, args: list[Any]) -> Any:
        if len(args) != 1:
            return CellError.VALUE
        num = _to_number(args[0])
        if isinstance(num, CellError):
            return num
        try:
            return _num(func(num))
        except (ValueError, OverflowError, ZeroDivisionError):
            return CellError.NUM
    return handler


def _f_round(env: _Env, args: list[Any]) -> Any:
    if len(args) != 2:
        return CellError.VALUE
    num = _to_number(args[0])
    digits = _to_number(args[1])
    if isinstance(num, CellError):
        return num
    if isinstance(digits, CellError):
        return digits
    return _num(round(num, int(digits)))


def _f_round_up(env: _Env, args: list[Any], up: bool) -> Any:
    if len(args) != 2:
        return CellError.VALUE
    num = _to_number(args[0])
    digits = _to_number(args[1])
    if isinstance(num, CellError):
        return num
    if isinstance(digits, CellError):
        return digits
    factor = 10.0 ** int(digits)
    scaled = num * factor
    if up:
        rounded = math.ceil(scaled) if num >= 0 else math.floor(scaled)
    else:
        rounded = math.floor(scaled) if num >= 0 else math.ceil(scaled)
    return _num(rounded / factor)


def _f_mod(env: _Env, args: list[Any]) -> Any:
    if len(args) != 2:
        return CellError.VALUE
    a = _to_number(args[0])
    b = _to_number(args[1])
    if isinstance(a, CellError):
        return a
    if isinstance(b, CellError):
        return b
    if b == 0:
        return CellError.DIV0
    return _num(a - b * math.floor(a / b))


def _f_if(env: _Env, nodes: tuple[Any, ...]) -> Any:
    if len(nodes) not in (2, 3):
        return CellError.VALUE
    cond = env.eval_node(nodes[0])
    if isinstance(cond, CellError):
        return cond
    truth = _to_bool(cond)
    if isinstance(truth, CellError):
        return truth
    if truth:
        return env.eval_node(nodes[1])
    if len(nodes) == 3:
        return env.eval_node(nodes[2])
    return False


def _f_ifs(env: _Env, nodes: tuple[Any, ...]) -> Any:
    if len(nodes) % 2 or not nodes:
        return CellError.VALUE
    for i in range(0, len(nodes), 2):
        cond = env.eval_node(nodes[i])
        if isinstance(cond, CellError):
            return cond
        truth = _to_bool(cond)
        if isinstance(truth, CellError):
            return truth
        if truth:
            return env.eval_node(nodes[i + 1])
    return CellError.NA


def _f_and(env: _Env, nodes: tuple[Any, ...]) -> Any:
    for node in nodes:
        value = env.eval_node(node)
        if isinstance(value, CellError):
            return value
        truth = _to_bool(value)
        if isinstance(truth, CellError):
            return truth
        if not truth:
            return False
    return True


def _f_or(env: _Env, nodes: tuple[Any, ...]) -> Any:
    for node in nodes:
        value = env.eval_node(node)
        if isinstance(value, CellError):
            return value
        truth = _to_bool(value)
        if isinstance(truth, CellError):
            return truth
        if truth:
            return True
    return False


def _f_not(env: _Env, nodes: tuple[Any, ...]) -> Any:
    if len(nodes) != 1:
        return CellError.VALUE
    value = env.eval_node(nodes[0])
    if isinstance(value, CellError):
        return value
    truth = _to_bool(value)
    if isinstance(truth, CellError):
        return truth
    return not truth


def _f_iferror(env: _Env, nodes: tuple[Any, ...]) -> Any:
    if len(nodes) != 2:
        return CellError.VALUE
    value = env.eval_node(nodes[0])
    if isinstance(value, CellError):
        return env.eval_node(nodes[1])
    return value


def _f_ifna(env: _Env, nodes: tuple[Any, ...]) -> Any:
    if len(nodes) != 2:
        return CellError.VALUE
    value = env.eval_node(nodes[0])
    if value is CellError.NA:
        return env.eval_node(nodes[1])
    return value


def _f_choose(env: _Env, nodes: tuple[Any, ...]) -> Any:
    if len(nodes) < 2:
        return CellError.VALUE
    index = env.eval_node(nodes[0])
    if isinstance(index, CellError):
        return index
    num = _to_number(index)
    if isinstance(num, CellError):
        return num
    pos = int(num)
    if pos < 1 or pos > len(nodes) - 1:
        return CellError.VALUE
    return env.eval_node(nodes[pos])


def _f_vlookup(env: _Env, args: list[Any], horizontal: bool) -> Any:
    if len(args) not in (3, 4):
        return CellError.VALUE
    lookup = args[0]
    table = args[1]
    col = _to_number(args[2])
    if isinstance(col, CellError):
        return col
    col = int(col)
    exact = True
    if len(args) == 4:
        flag = _to_bool(args[3])
        if isinstance(flag, CellError):
            return flag
        exact = not flag
    if not isinstance(table, RangeValue):
        return CellError.VALUE
    grid = table.rows if not horizontal else list(zip(*table.rows, strict=True)) \
        if table.rows else []
    if not grid:
        return CellError.NA
    if col < 1 or col > len(grid[0]):
        return CellError.REF
    if exact:
        for row in grid:
            if _cells_equal(lookup, row[0]):
                return row[col - 1]
        return CellError.NA
    best = None
    for row in grid:
        if _cells_less_equal(row[0], lookup):
            best = row
        else:
            break
    if best is None:
        return CellError.NA
    return best[col - 1]


def _cells_equal(a: Any, b: Any) -> bool:
    if _is_number(a) and _is_number(b):
        return float(a) == float(b)
    if isinstance(a, str) and isinstance(b, str):
        return a.lower() == b.lower()
    return a == b


def _cells_less_equal(a: Any, b: Any) -> bool:
    if _is_number(a) and _is_number(b):
        return float(a) <= float(b)
    if isinstance(a, str) and isinstance(b, str):
        return a.lower() <= b.lower()
    return False


def _f_index(env: _Env, args: list[Any]) -> Any:
    if len(args) not in (2, 3):
        return CellError.VALUE
    table = args[0]
    if not isinstance(table, RangeValue):
        return CellError.VALUE
    row = _to_number(args[1])
    if isinstance(row, CellError):
        return row
    row = int(row)
    if row < 1 or row > len(table.rows):
        return CellError.REF
    if len(args) == 2:
        if len(table.rows[row - 1]) != 1:
            return CellError.VALUE
        return table.rows[row - 1][0]
    col = _to_number(args[2])
    if isinstance(col, CellError):
        return col
    col = int(col)
    if col < 1 or col > len(table.rows[row - 1]):
        return CellError.REF
    return table.rows[row - 1][col - 1]


def _f_match(env: _Env, args: list[Any]) -> Any:
    if len(args) not in (2, 3):
        return CellError.VALUE
    lookup = args[0]
    rng = args[1]
    if not isinstance(rng, RangeValue):
        return CellError.VALUE
    flat = [v for row in rng.rows for v in row]
    mode = 0
    if len(args) == 3:
        mode_num = _to_number(args[2])
        if isinstance(mode_num, CellError):
            return mode_num
        mode = int(mode_num)
    if mode == 0:
        for i, value in enumerate(flat, start=1):
            if _cells_equal(lookup, value):
                return i
        return CellError.NA
    if mode == 1:
        best = None
        for i, value in enumerate(flat, start=1):
            if _cells_less_equal(value, lookup):
                best = i
            else:
                break
        return best if best is not None else CellError.NA
    best = None
    for i, value in enumerate(flat, start=1):
        if _cells_less_equal(lookup, value):
            best = i
            break
    return best if best is not None else CellError.NA


def _f_concat(env: _Env, args: list[Any]) -> Any:
    err = _first_error(_flatten(args))
    if err is not None:
        return err
    return "".join(_to_text(v) for v in _flatten(args))


def _f_left(env: _Env, args: list[Any]) -> Any:
    text = _to_text(args[0])
    count = 1
    if len(args) > 1:
        num = _to_number(args[1])
        if isinstance(num, CellError):
            return num
        count = int(num)
    return text[:count]


def _f_right(env: _Env, args: list[Any]) -> Any:
    text = _to_text(args[0])
    count = 1
    if len(args) > 1:
        num = _to_number(args[1])
        if isinstance(num, CellError):
            return num
        count = int(num)
    return text[len(text) - count:] if count else ""


def _f_mid(env: _Env, args: list[Any]) -> Any:
    text = _to_text(args[0])
    start = _to_number(args[1])
    count = _to_number(args[2])
    if isinstance(start, CellError):
        return start
    if isinstance(count, CellError):
        return count
    start, count = int(start), int(count)
    if start < 1 or count < 0:
        return CellError.VALUE
    return text[start - 1:start - 1 + count]


def _f_len(env: _Env, args: list[Any]) -> Any:
    return len(_to_text(args[0]))


def _f_find(env: _Env, args: list[Any]) -> Any:
    needle = _to_text(args[0])
    haystack = _to_text(args[1])
    start = 1
    if len(args) > 2:
        num = _to_number(args[2])
        if isinstance(num, CellError):
            return num
        start = int(num)
    pos = haystack.find(needle, start - 1)
    if pos < 0:
        return CellError.VALUE
    return pos + 1


def _f_search(env: _Env, args: list[Any]) -> Any:
    needle = _to_text(args[0]).lower()
    haystack = _to_text(args[1]).lower()
    start = 1
    if len(args) > 2:
        num = _to_number(args[2])
        if isinstance(num, CellError):
            return num
        start = int(num)
    pos = haystack.find(needle, start - 1)
    if pos < 0:
        return CellError.VALUE
    return pos + 1


def _f_substitute(env: _Env, args: list[Any]) -> Any:
    text = _to_text(args[0])
    old = _to_text(args[1])
    new = _to_text(args[2])
    if len(args) == 4:
        num = _to_number(args[3])
        if isinstance(num, CellError):
            return num
        parts = text.split(old)
        pos = int(num)
        if pos < 1 or pos >= len(parts):
            return text
        return old.join(parts[:pos]) + new + old.join(parts[pos:])
    return text.replace(old, new)


def _f_rept(env: _Env, args: list[Any]) -> Any:
    text = _to_text(args[0])
    count = _to_number(args[1])
    if isinstance(count, CellError):
        return count
    if int(count) < 0:
        return CellError.VALUE
    return text * int(count)


def _f_text(env: _Env, args: list[Any]) -> Any:
    value = args[0]
    fmt = _to_text(args[1])
    if isinstance(value, CellError):
        return value
    if _is_number(value):
        num = float(value)
        if fmt in ("0", "#,##0"):
            if fmt == "#,##0":
                return f"{num:,.0f}"
            return str(int(num)) if num.is_integer() else repr(num)
        if fmt == "0.00":
            return f"{num:.2f}"
        if fmt == "#,##0.00":
            return f"{num:,.2f}"
        if fmt == "0%":
            return f"{num * 100:.0f}%"
        if fmt == "0.00%":
            return f"{num * 100:.2f}%"
        if fmt == "yyyy-mm-dd":
            try:
                return serial_to_date(num).isoformat()
            except (ValueError, OverflowError):
                return CellError.VALUE
    return _to_text(value)


def _f_value(env: _Env, args: list[Any]) -> Any:
    text = _to_text(args[0]).strip().replace("%", "")
    try:
        return _num(float(text))
    except ValueError:
        return CellError.VALUE


def _f_date(env: _Env, args: list[Any]) -> Any:
    year = _to_number(args[0])
    month = _to_number(args[1])
    day = _to_number(args[2])
    for num in (year, month, day):
        if isinstance(num, CellError):
            return num
    try:
        return _num(date_to_serial(int(year), int(month), int(day)))
    except ValueError:
        return CellError.NUM


def _f_time(env: _Env, args: list[Any]) -> Any:
    hour = _to_number(args[0])
    minute = _to_number(args[1])
    second = _to_number(args[2])
    for num in (hour, minute, second):
        if isinstance(num, CellError):
            return num
    if not (0 <= hour < 24 and 0 <= minute < 60 and 0 <= second < 60):
        return CellError.NUM
    return _num((hour + minute / 60 + second / 3600) / 24)


def _f_today(env: _Env, args: list[Any]) -> Any:
    now = date.today()
    return _num(date_to_serial(now.year, now.month, now.day))


def _f_now(env: _Env, args: list[Any]) -> Any:
    moment = datetime.now()
    serial = date_to_serial(moment.year, moment.month, moment.day)
    fraction = (moment.hour * 3600 + moment.minute * 60
                + moment.second) / 86400
    return serial + fraction


def _date_part(func) -> Any:
    def handler(env: _Env, args: list[Any]) -> Any:
        num = _to_number(args[0])
        if isinstance(num, CellError):
            return num
        try:
            when = serial_to_date(num)
        except (ValueError, OverflowError):
            return CellError.NUM
        return func(when)
    return handler


def _f_weekday(env: _Env, args: list[Any]) -> Any:
    num = _to_number(args[0])
    if isinstance(num, CellError):
        return num
    try:
        when = serial_to_date(num)
    except (ValueError, OverflowError):
        return CellError.NUM
    # Excel: Sunday=1 .. Saturday=7.
    return (when.weekday() + 1) % 7 + 1


def _f_datedif(env: _Env, args: list[Any]) -> Any:
    start = _to_number(args[0])
    end = _to_number(args[1])
    unit = _to_text(args[2]).lower()
    if isinstance(start, CellError):
        return start
    if isinstance(end, CellError):
        return end
    try:
        first = serial_to_date(start)
        last = serial_to_date(end)
    except (ValueError, OverflowError):
        return CellError.NUM
    if last < first:
        return CellError.NUM
    if unit == "d":
        return (last - first).days
    if unit == "m":
        months = (last.year - first.year) * 12 + last.month - first.month
        if last.day < first.day:
            months -= 1
        return months
    if unit == "y":
        years = last.year - first.year
        if (last.month, last.day) < (first.month, first.day):
            years -= 1
        return years
    return CellError.VALUE


def _f_edate(env: _Env, args: list[Any]) -> Any:
    num = _to_number(args[0])
    months = _to_number(args[1])
    if isinstance(num, CellError):
        return num
    if isinstance(months, CellError):
        return months
    try:
        when = serial_to_date(num)
    except (ValueError, OverflowError):
        return CellError.NUM
    total = (when.year * 12 + when.month - 1) + int(months)
    year, month = divmod(total, 12)
    month += 1
    day = min(when.day, [31, 29 if year % 4 == 0
                         and (year % 100 != 0 or year % 400 == 0)
                         else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30,
                         31][month - 1])
    return _num(date_to_serial(year, month, day))


def _f_indirect(env: _Env, args: list[Any]) -> Any:
    text = _to_text(args[0])
    sheet = env.sheet
    ref_text = text
    if "!" in text:
        sheet, ref_text = text.rsplit("!", 1)
        sheet = sheet.strip("'")
    match = _REF_RE.match(ref_text.strip())
    if not match:
        return CellError.REF
    return env.cell_value(sheet, int(match.group(2)),
                         _col_to_num(match.group(1)))


def _f_row(env: _Env, args: list[Any]) -> Any:
    if not args:
        return env.at_row
    if isinstance(args[0], RangeValue):
        return 1
    return env.at_row


def _f_column(env: _Env, args: list[Any]) -> Any:
    if not args:
        return env.at_col
    if isinstance(args[0], RangeValue):
        return 1
    return env.at_col


def _f_is(env: _Env, args: list[Any], kind: str) -> Any:
    value = args[0]
    if kind == "number":
        return _is_number(value)
    if kind == "text":
        return isinstance(value, str)
    if kind == "blank":
        return value is None
    if kind == "logical":
        return isinstance(value, bool)
    if kind == "error":
        return isinstance(value, CellError)
    if kind == "na":
        return value is CellError.NA
    return False


_FUNCTIONS: dict[str, Any] = {
    "SUM": _f_sum,
    "AVERAGE": _f_average,
    "COUNT": _f_count,
    "COUNTA": _f_counta,
    "COUNTBLANK": _f_countblank,
    "MIN": _f_min,
    "MAX": _f_max,
    "PRODUCT": _f_product,
    "SUMPRODUCT": _f_sumproduct,
    "MEDIAN": _f_median,
    "STDEV": lambda e, a: _f_stdev(e, a, False),
    "STDEVP": lambda e, a: _f_stdev(e, a, True),
    "LARGE": _f_large,
    "SMALL": _f_small,
    "RANK": _f_rank,
    "ABS": _one_num("ABS", abs),
    "INT": _one_num("INT", lambda n: float(math.floor(n))),
    "SIGN": _one_num("SIGN", lambda n: 0.0 if n == 0 else (1.0 if n > 0 else -1.0)),
    "TRUNC": _one_num("TRUNC", lambda n: float(math.trunc(n))),
    "EXP": _one_num("EXP", math.exp),
    "LN": _one_num("LN", math.log),
    "LOG10": _one_num("LOG10", math.log10),
    "SQRT": _one_num("SQRT", math.sqrt),
    "ROUND": _f_round,
    "ROUNDUP": lambda e, a: _f_round_up(e, a, True),
    "ROUNDDOWN": lambda e, a: _f_round_up(e, a, False),
    "MOD": _f_mod,
    "POWER": lambda e, a: _f_mod(e, a) if False else _power(e, a),
    "IF": _f_if,
    "IFS": _f_ifs,
    "AND": _f_and,
    "OR": _f_or,
    "NOT": _f_not,
    "IFERROR": _f_iferror,
    "IFNA": _f_ifna,
    "TRUE": lambda e, a: True,
    "FALSE": lambda e, a: False,
    "CHOOSE": _f_choose,
    "VLOOKUP": lambda e, a: _f_vlookup(e, a, False),
    "HLOOKUP": lambda e, a: _f_vlookup(e, a, True),
    "INDEX": _f_index,
    "MATCH": _f_match,
    "CONCAT": _f_concat,
    "CONCATENATE": _f_concat,
    "LEFT": _f_left,
    "RIGHT": _f_right,
    "MID": _f_mid,
    "LEN": _f_len,
    "UPPER": lambda e, a: _to_text(a[0]).upper(),
    "LOWER": lambda e, a: _to_text(a[0]).lower(),
    "TRIM": lambda e, a: " ".join(_to_text(a[0]).split()),
    "TEXT": _f_text,
    "VALUE": _f_value,
    "FIND": _f_find,
    "SEARCH": _f_search,
    "SUBSTITUTE": _f_substitute,
    "REPT": _f_rept,
    "EXACT": lambda e, a: _to_text(a[0]) == _to_text(a[1]),
    "T": lambda e, a: a[0] if isinstance(a[0], str) else "",
    "DATE": _f_date,
    "TIME": _f_time,
    "TODAY": _f_today,
    "NOW": _f_now,
    "YEAR": _date_part(lambda d: d.year),
    "MONTH": _date_part(lambda d: d.month),
    "DAY": _date_part(lambda d: d.day),
    "HOUR": lambda e, a: _time_part(a[0], 0),
    "MINUTE": lambda e, a: _time_part(a[0], 1),
    "SECOND": lambda e, a: _time_part(a[0], 2),
    "WEEKDAY": _f_weekday,
    "DATEDIF": _f_datedif,
    "EDATE": _f_edate,
    "INDIRECT": _f_indirect,
    "ROW": _f_row,
    "COLUMN": _f_column,
    "ISNUMBER": lambda e, a: _f_is(e, a, "number"),
    "ISTEXT": lambda e, a: _f_is(e, a, "text"),
    "ISBLANK": lambda e, a: _f_is(e, a, "blank"),
    "ISLOGICAL": lambda e, a: _f_is(e, a, "logical"),
    "ISERROR": lambda e, a: _f_is(e, a, "error"),
    "ISNA": lambda e, a: _f_is(e, a, "na"),
    "NA": lambda e, a: CellError.NA,
    "PI": lambda e, a: math.pi,
    "RAND": lambda e, a: random.random(),
    "RANDBETWEEN": lambda e, a: _randbetween(e, a),
}

_LAZY = {"IF", "IFS", "AND", "OR", "NOT", "IFERROR", "IFNA", "CHOOSE"}
_ERROR_AWARE = {"ISNUMBER", "ISTEXT", "ISBLANK", "ISLOGICAL", "ISERROR",
                "ISNA"}


def _power(env: _Env, args: list[Any]) -> Any:
    if len(args) != 2:
        return CellError.VALUE
    base = _to_number(args[0])
    exp = _to_number(args[1])
    if isinstance(base, CellError):
        return base
    if isinstance(exp, CellError):
        return exp
    try:
        return _num(base ** exp)
    except (ValueError, OverflowError, ZeroDivisionError):
        return CellError.NUM


def _time_part(value: Any, part: int) -> Any:
    num = _to_number(value)
    if isinstance(num, CellError):
        return num
    fraction = float(num) % 1.0
    total_seconds = int(round(fraction * 86400))
    hour, rem = divmod(total_seconds, 3600)
    minute, second = divmod(rem, 60)
    return (hour, minute, second)[part]


def _randbetween(env: _Env, args: list[Any]) -> Any:
    low = _to_number(args[0])
    high = _to_number(args[1])
    if isinstance(low, CellError):
        return low
    if isinstance(high, CellError):
        return high
    if low > high:
        return CellError.NUM
    return random.randint(int(low), int(high))


def evaluate(workbook: Any, sheet: str, at: str,
             formula: Any) -> Any:
    """Evaluate a formula string (or AST) in a workbook's context.

    ``at`` is the cell the formula belongs to (for ROW()/COLUMN()
    and error messages); references in the formula are absolute
    positions, so evaluation does not depend on it. A string without
    a leading ``=`` (the stored OOXML form) is accepted here — the
    strict ``parse_formula`` still requires it. Returns a number,
    string, bool, None (blank) or a :class:`CellError` — never
    raises for a bad value.
    """
    if isinstance(formula, str) and not formula.startswith("="):
        formula = "=" + formula
    node = formula if not isinstance(formula, str) \
        else parse_formula(formula)
    env = _Env(workbook, sheet, at=at)
    return env.eval_node(node)
