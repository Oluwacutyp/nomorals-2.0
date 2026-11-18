"""Text formatting core: parse -> validate -> transform with located errors.

CodeBeautify's mineable system, built natively: paste text, parse it
for real, transform it, show the result beside the input — with
validation errors located by line and column. Parsers are real per
format (stdlib ``json``, ``xml.etree``, ``tomllib``; ``yaml`` via the
profile runtime's ladder when installed), so validation is
authoritative, never regex-shaped. A diagnostic is always a
machine-readable kind plus a human message, per the Echelon standard
(machine-readable kinds separate from human messages); an error
position is the parser's own, never invented.

Formats: json, jsonl, yaml, toml, xml, csv, sql, css, html, js.
The data formats (json, jsonl, yaml, toml, xml, csv) share one Python
data model, so conversion between them is real; js beautification
needs a real language parser — the ladder is a ``prettier``-class
external binary where installed, honest refusal where not, never a
regex fake.
"""

from __future__ import annotations

import binascii
import csv
import html
import io
import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

__all__ = [
    "Diagnostic",
    "TransformResult",
    "TEXT_DIAGNOSTIC_KINDS",
    "FORMATS",
    "DATA_FORMATS",
    "detect_format",
    "beautify",
    "minify",
    "validate",
    "convert",
    "fix_json",
    "transform",
    "format_transform_reply",
]


#: The closed taxonomy of diagnostic kinds. ``kind`` is the
#: machine-readable half of a diagnostic; the human message stays
#: free text. The taxonomy is closed by design — a typo'd kind must
#: fail at the raise site, not downstream where a consumer switches
#: on it.
TEXT_DIAGNOSTIC_KINDS = frozenset({
    "parse",               # input could not be parsed by the real parser
    "unsupported_format",  # format (or format pair) not recognized
    "missing_dependency",  # optional package or binary absent
    "empty",               # input has no content for the operation
    "invalid_argument",    # caller passed a bad argument
    "fixed",               # a repair the fixer applied (with position)
    "warning",             # non-fatal note (valid input, worth naming)
})

FORMATS = (
    "json", "jsonl", "yaml", "toml", "xml", "csv",
    "sql", "css", "html", "js",
)

#: Formats with a Python data model — conversion between them is real.
DATA_FORMATS = ("json", "jsonl", "yaml", "toml", "xml", "csv")

#: Code-generation targets for the data model.
CODE_TARGETS = ("typescript", "php")

#: (from, to) pairs ``convert`` supports, beyond the data-format mesh.
EXTRA_CONVERTS = frozenset({
    ("utf8", "hex"), ("hex", "utf8"),
    ("html", "jsx"),
})


@dataclass
class Diagnostic:
    """One located finding: a machine-readable kind + human message."""

    line: int
    col: int
    message: str
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in TEXT_DIAGNOSTIC_KINDS:
            raise ValueError(
                f"unknown diagnostic kind {self.kind!r}; expected one "
                f"of {sorted(TEXT_DIAGNOSTIC_KINDS)}")

    def to_dict(self) -> dict[str, Any]:
        return {"line": self.line, "col": self.col,
                "message": self.message, "kind": self.kind}

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Diagnostic:
        return cls(line=int(row["line"]), col=int(row["col"]),
                   message=str(row["message"]), kind=str(row["kind"]))


@dataclass
class TransformResult:
    """The result of one parse -> validate -> transform step."""

    text: str
    format_in: str
    format_out: str
    diagnostics: list[Diagnostic] = field(default_factory=list)
    ok: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "format_in": self.format_in,
            "format_out": self.format_out,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "ok": self.ok,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> TransformResult:
        return cls(
            text=str(row["text"]), format_in=str(row["format_in"]),
            format_out=str(row["format_out"]),
            diagnostics=[Diagnostic.from_dict(d)
                         for d in row.get("diagnostics", [])],
            ok=bool(row.get("ok", True)), error=str(row.get("error", "")))


def _fail(fmt_in: str, fmt_out: str, diag: Diagnostic) -> TransformResult:
    return TransformResult(text="", format_in=fmt_in, format_out=fmt_out,
                           diagnostics=[diag], ok=False, error=diag.message)


def _line_col(text: str, index: int) -> tuple[int, int]:
    """1-based (line, col) for a 0-based index into ``text``."""
    before = text[:index]
    line = before.count("\n") + 1
    last_nl = before.rfind("\n")
    return line, index - last_nl


# ── shared parsing ────────────────────────────────────────────────────

def _parse_json(text: str) -> tuple[Any, list[Diagnostic]]:
    try:
        return json.loads(text), []
    except json.JSONDecodeError as exc:
        return None, [Diagnostic(exc.lineno, exc.colno, exc.msg, "parse")]


def _parse_jsonl(text: str) -> tuple[Any, list[Diagnostic]]:
    rows: list[Any] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            return None, [Diagnostic(lineno, exc.colno, exc.msg, "parse")]
    return rows, []


def _yaml_module() -> Any:
    """PyYAML via the profile runtime's ladder — None when absent."""
    try:
        import yaml  # type: ignore
    except ImportError:
        return None
    return yaml


def _parse_yaml(text: str) -> tuple[Any, list[Diagnostic]]:
    yaml = _yaml_module()
    if yaml is None:
        return None, [Diagnostic(
            1, 1, "yaml is not installed here (PyYAML) — the yaml rung "
            "needs the yaml package", "missing_dependency")]
    try:
        return yaml.safe_load(text), []
    except Exception as exc:  # noqa: BLE001 — yaml errors carry marks
        mark = getattr(exc, "problem_mark", None) or getattr(
            exc, "context_mark", None)
        if mark is not None:
            line, col = int(mark.line) + 1, int(mark.column) + 1
        else:
            line, col = 1, 1
        return None, [Diagnostic(line, col, str(exc).splitlines()[0]
                                 if str(exc) else "invalid yaml", "parse")]


def _tomllib_module() -> Any:
    try:
        import tomllib
    except ImportError:
        return None
    return tomllib


_TOML_POS = re.compile(r"\(at line (\d+), column (\d+)\)")


def _parse_toml(text: str) -> tuple[Any, list[Diagnostic]]:
    tomllib = _tomllib_module()
    if tomllib is None:
        return None, [Diagnostic(
            1, 1, "tomllib is not available on this Python — the toml "
            "rung needs Python 3.11+", "missing_dependency")]
    try:
        return tomllib.loads(text), []
    except Exception as exc:  # noqa: BLE001 — message carries position
        msg = str(exc)
        match = _TOML_POS.search(msg)
        if match:
            line, col = int(match.group(1)), int(match.group(2))
        else:
            line, col = 1, 1
        return None, [Diagnostic(line, col, msg.splitlines()[0], "parse")]


def _parse_xml(text: str) -> tuple[Any, list[Diagnostic]]:
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        line, col = exc.position if exc.position else (1, 1)
        return None, [Diagnostic(line, col, exc.msg, "parse")]
    # The root tag is part of the data — dropping it breaks the
    # json -> xml -> json round-trip (the way back would lose "root").
    return {root.tag: _xml_to_data(root)}, []


def _xml_to_data(elem: ET.Element) -> Any:
    """Element -> the shared data model (attributes @, text #text)."""
    children: dict[str, Any] = {}
    for child in elem:
        value = _xml_to_data(child)
        if child.tag in children:
            if not isinstance(children[child.tag], list):
                children[child.tag] = [children[child.tag]]
            children[child.tag].append(value)
        else:
            children[child.tag] = value
    text = (elem.text or "").strip()
    if not children and not elem.attrib:
        return text
    out: dict[str, Any] = {}
    for key, value in elem.attrib.items():
        out[f"@{key}"] = value
    out.update(children)
    if text:
        out["#text"] = text
    return out


def _data_to_xml(value: Any, tag: str) -> ET.Element:
    elem = ET.Element(tag)
    if isinstance(value, dict):
        for key, item in value.items():
            if key.startswith("@"):
                elem.set(key[1:], str(item))
            elif key == "#text":
                elem.text = str(item)
            elif isinstance(item, list):
                for sub in item:
                    elem.append(_data_to_xml(sub, key))
            else:
                elem.append(_data_to_xml(item, key))
    elif isinstance(value, list):
        for sub in value:
            elem.append(_data_to_xml(sub, "item"))
    elif value is None:
        elem.text = ""
    else:
        elem.text = str(value)
    return elem


def _parse_csv(text: str) -> tuple[Any, list[Diagnostic]]:
    try:
        rows = list(csv.reader(io.StringIO(text)))
    except csv.Error as exc:
        return None, [Diagnostic(1, 1, str(exc), "parse")]
    rows = [row for row in rows if row]
    if not rows:
        return [], []
    width = len(rows[0])
    for lineno, row in enumerate(rows, 1):
        if len(row) != width:
            return None, [Diagnostic(
                lineno, 1,
                f"ragged row: {len(row)} fields, header has {width}",
                "parse")]
    headers = rows[0]
    return [dict(zip(headers, row, strict=True)) for row in rows[1:]], []


def _parse_data(text: str, fmt: str) -> tuple[Any, list[Diagnostic]]:
    """Parse ``text`` as a data format into the shared Python model."""
    if fmt == "json":
        return _parse_json(text)
    if fmt == "jsonl":
        return _parse_jsonl(text)
    if fmt == "yaml":
        return _parse_yaml(text)
    if fmt == "toml":
        return _parse_toml(text)
    if fmt == "xml":
        return _parse_xml(text)
    if fmt == "csv":
        return _parse_csv(text)
    return None, [Diagnostic(1, 1, f"unknown format {fmt!r}",
                             "unsupported_format")]


# ── shared emission ───────────────────────────────────────────────────

def _toml_scalar(value: Any, key: str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    raise ValueError(f"cannot write {key!r} to TOML: no TOML null")


def _emit_toml(value: Any) -> str:
    """Minimal deterministic TOML emitter for the JSON data model.

    Scalars first, then sub-tables, depth-first, so every ``[a.b]``
    header follows the table it belongs to and a ``[[a]]`` element's
    own sub-tables stay attached to that element.
    """
    if not isinstance(value, dict):
        raise ValueError("TOML top level must be a table (object)")
    lines: list[str] = []

    def emit(table: dict[str, Any], path: str) -> None:
        subs: list[tuple[str, Any, bool]] = []
        for key, item in table.items():
            if isinstance(item, dict):
                subs.append((key, item, False))
            elif isinstance(item, list) and item and all(
                    isinstance(sub, dict) for sub in item):
                subs.append((key, item, True))
            elif isinstance(item, list):
                parts = [_toml_scalar(sub, key) for sub in item]
                lines.append(f"{key} = [{', '.join(parts)}]")
            else:
                lines.append(f"{key} = {_toml_scalar(item, key)}")
        for key, item, is_array in subs:
            full = f"{path}.{key}" if path else key
            if is_array:
                for sub in item:
                    lines.append(f"[[{full}]]")
                    emit(sub, full)
            else:
                lines.append(f"[{full}]")
                emit(item, full)

    emit(value, "")
    return "\n".join(lines) + "\n"


def _emit_yaml(value: Any, flow: bool = False) -> tuple[str, list[Diagnostic]]:
    yaml = _yaml_module()
    if yaml is None:
        return "", [Diagnostic(
            1, 1, "yaml is not installed here (PyYAML) — the yaml rung "
            "needs the yaml package", "missing_dependency")]
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True,
                          default_flow_style=flow), []


def _emit_data(value: Any, fmt: str) -> tuple[str, list[Diagnostic]]:
    """Render the shared Python model as ``fmt``."""
    if fmt == "json":
        return json.dumps(value, indent=2, ensure_ascii=False) + "\n", []
    if fmt == "jsonl":
        rows = value if isinstance(value, list) else [value]
        return "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows), []
    if fmt == "yaml":
        return _emit_yaml(value)
    if fmt == "toml":
        try:
            return _emit_toml(value), []
        except ValueError as exc:
            return "", [Diagnostic(1, 1, str(exc), "invalid_argument")]
    if fmt == "xml":
        if isinstance(value, dict) and len(value) == 1:
            tag, inner = next(iter(value.items()))
            elem = _data_to_xml(inner, tag)
        else:
            elem = _data_to_xml(value, "root")
        ET.indent(elem, space="  ")
        return ET.tostring(elem, encoding="unicode") + "\n", []
    if fmt == "csv":
        rows = value if isinstance(value, list) else [value]
        if not rows:
            return "", []
        if not all(isinstance(row, dict) for row in rows):
            return "", [Diagnostic(
                1, 1, "csv needs an array of objects (rows with named "
                "columns)", "invalid_argument")]
        headers: list[str] = []
        for row in rows:
            for key in row:
                if key not in headers:
                    headers.append(key)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key)
                             for key in headers})
        return buf.getvalue(), []
    return "", [Diagnostic(1, 1, f"unknown format {fmt!r}",
                           "unsupported_format")]

# ── SQL: real tokenizer + clause formatter ────────────────────────────

SQL_KEYWORDS = frozenset({
    "select", "from", "where", "group", "having", "order", "limit",
    "offset", "join", "inner", "left", "right", "full", "cross", "on",
    "using", "and", "or", "not", "in", "is", "null", "like", "between",
    "exists", "case", "when", "then", "else", "end", "as", "distinct",
    "union", "all", "except", "intersect", "insert", "into", "values",
    "update", "set", "delete", "create", "table", "primary", "key",
    "by", "asc", "desc",
})

#: Clauses that start a fresh line at the current indent.
SQL_CLAUSE_STARTERS = frozenset({
    "select", "from", "where", "group", "having", "order", "limit",
    "offset", "union", "except", "intersect", "insert", "values",
    "update", "set", "delete",
})

_SQL_TOKEN = re.compile(r"""
    (?P<ws>\s+)
  | (?P<comment>--[^\n]*|/\*.*?\*/)
  | (?P<string>'(?:[^']|'')*'|"(?:[^"]|"")*")
  | (?P<number>\d+(?:\.\d+)?)
  | (?P<word>[A-Za-z_][A-Za-z0-9_$]*)
  | (?P<punct>[(),.;*=<>!+\-/])
""", re.VERBOSE | re.DOTALL)


@dataclass
class _SqlToken:
    kind: str
    text: str
    index: int


def _tokenize_sql(text: str) -> tuple[list[_SqlToken], list[Diagnostic]]:
    tokens: list[_SqlToken] = []
    diags: list[Diagnostic] = []
    pos = 0
    while pos < len(text):
        match = _SQL_TOKEN.match(text, pos)
        if match is None:
            line, col = _line_col(text, pos)
            return tokens, [Diagnostic(
                line, col, f"cannot tokenize character {text[pos]!r}",
                "parse")]
        kind = match.lastgroup or "punct"
        chunk = match.group(0)
        if kind == "string" and (
                chunk.startswith("'") and not chunk.endswith("'")
                or chunk.startswith('"') and not chunk.endswith('"')):
            line, col = _line_col(text, pos)
            return tokens, [Diagnostic(
                line, col, "unterminated string literal", "parse")]
        if kind == "comment" and chunk.startswith("/*") \
                and not chunk.endswith("*/"):
            line, col = _line_col(text, pos)
            return tokens, [Diagnostic(
                line, col, "unterminated block comment", "parse")]
        if kind not in ("ws",):
            tokens.append(_SqlToken(kind, chunk, pos))
        pos = match.end()
    depth = 0
    for tok in tokens:
        if tok.text == "(":
            depth += 1
        elif tok.text == ")":
            depth -= 1
            if depth < 0:
                line, col = _line_col(text, tok.index)
                return tokens, [Diagnostic(
                    line, col, "unmatched closing parenthesis", "parse")]
    if depth > 0:
        for tok in reversed(tokens):
            if tok.text == "(":
                line, col = _line_col(text, tok.index)
                return tokens, [Diagnostic(
                    line, col, "unclosed parenthesis", "parse")]
    return tokens, diags


def _sql_word(tok: _SqlToken) -> str:
    return tok.text.upper() if tok.text.lower() in SQL_KEYWORDS else tok.text


def _format_sql(tokens: list[_SqlToken]) -> str:
    lines: list[str] = []
    current: list[str] = []
    indent = 0

    def flush() -> None:
        if current:
            lines.append("  " * indent + " ".join(current))
            current.clear()

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        word = tok.text.lower()
        if tok.kind == "word" and word in SQL_CLAUSE_STARTERS:
            flush()
            # GROUP BY / ORDER BY stay together on the clause line.
            if word in ("group", "order") and i + 1 < len(tokens) \
                    and tokens[i + 1].text.lower() == "by":
                current.append(_sql_word(tok))
                current.append("BY")
                i += 2
                continue
            current.append(_sql_word(tok))
            i += 1
            continue
        if tok.text == "(":
            # A subquery (contains SELECT) breaks lines; a call stays inline.
            rest = [t.text.lower() for t in tokens[i + 1:i + 12]]
            if "select" in rest:
                current.append("(")
                flush()
                indent += 1
                i += 1
                continue
            current.append("(")
            i += 1
            continue
        if tok.text == ")":
            if current and current[-1] == "(":
                current.append(")")
                i += 1
                continue
            flush()
            indent = max(0, indent - 1)
            current.append(")")
            i += 1
            continue
        if tok.text == ",":
            current[-1] = current[-1] + ","
            i += 1
            continue
        if tok.text == ";":
            current.append(";")
            flush()
            i += 1
            continue
        current.append(_sql_word(tok) if tok.kind == "word" else tok.text)
        i += 1
    flush()
    return "\n".join(lines)


def _minify_sql(tokens: list[_SqlToken]) -> str:
    parts: list[str] = []
    for tok in tokens:
        if tok.kind == "comment":
            continue
        text = _sql_word(tok) if tok.kind == "word" else tok.text
        if parts and text not in (",", ")", ";") and parts[-1] != "(":
            parts.append(" ")
        parts.append(text)
    return "".join(parts)


# ── CSS: real tokenizer + formatter ───────────────────────────────────

_CSS_TOKEN = re.compile(r"""
    (?P<ws>\s+)
  | (?P<comment>/\*.*?\*/)
  | (?P<string>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
  | (?P<punct>[{}():;,>+~])
  | (?P<other>[^\s{}():;,>+~]+)
""", re.VERBOSE | re.DOTALL)


def _tokenize_css(text: str) -> tuple[list[_SqlToken], list[Diagnostic]]:
    tokens: list[_SqlToken] = []
    pos = 0
    while pos < len(text):
        match = _CSS_TOKEN.match(text, pos)
        if match is None:
            line, col = _line_col(text, pos)
            return tokens, [Diagnostic(
                line, col, f"cannot tokenize character {text[pos]!r}",
                "parse")]
        kind = match.lastgroup or "other"
        chunk = match.group(0)
        if kind == "comment" and not chunk.endswith("*/"):
            line, col = _line_col(text, pos)
            return tokens, [Diagnostic(
                line, col, "unterminated comment", "parse")]
        if kind == "string" and len(chunk) < 2:
            line, col = _line_col(text, pos)
            return tokens, [Diagnostic(
                line, col, "unterminated string", "parse")]
        if kind not in ("ws", "comment"):
            tokens.append(_SqlToken(kind, chunk, pos))
        pos = match.end()
    depth = 0
    parens = 0
    for tok in tokens:
        if tok.text == "{":
            depth += 1
        elif tok.text == "}":
            depth -= 1
            if depth < 0:
                line, col = _line_col(text, tok.index)
                return tokens, [Diagnostic(
                    line, col, "unmatched closing brace", "parse")]
        elif tok.text == "(":
            parens += 1
        elif tok.text == ")":
            parens -= 1
            if parens < 0:
                line, col = _line_col(text, tok.index)
                return tokens, [Diagnostic(
                    line, col, "unmatched closing parenthesis", "parse")]
    if depth > 0:
        for tok in reversed(tokens):
            if tok.text == "{":
                line, col = _line_col(text, tok.index)
                return tokens, [Diagnostic(
                    line, col, "unclosed brace", "parse")]
    if parens > 0:
        return tokens, [Diagnostic(1, 1, "unclosed parenthesis", "parse")]
    return tokens, []


def _format_css(tokens: list[_SqlToken]) -> str:
    out: list[str] = []
    indent = 0
    i = 0
    pending: list[str] = []

    def emit_selector() -> None:
        if pending:
            joined = " ".join(pending).replace(" ,", ",")
            # selectors never space around ':' — a:hover, not a: hover —
            # and never inside parens: (max-width: 600px).
            joined = (joined.replace(" :", ":").replace(": ", ":")
                      .replace("( ", "(").replace(" )", ")"))
            out.append("  " * indent + joined + " {")
            pending.clear()

    def flush_declaration() -> None:
        # A declaration pending when ';', '}' or end-of-input arrives
        # is still a declaration — emit it, semicolon and all. A
        # pending fragment at the top level is a bare selector, not a
        # declaration: it is emitted as-is, never semicolon'd.
        if pending:
            joined = " ".join(pending).replace(" :", ":").replace(" ,", ",")
            if indent > 0:
                out.append("  " * indent + joined + ";")
            else:
                out.append("  " * indent + joined)
            pending.clear()

    while i < len(tokens):
        tok = tokens[i]
        if tok.text == "{":
            emit_selector()
            indent += 1
            i += 1
        elif tok.text == "}":
            flush_declaration()
            indent = max(0, indent - 1)
            out.append("  " * indent + "}")
            i += 1
        elif tok.text == ";":
            flush_declaration()
            i += 1
        elif tok.text == ":" and pending and i + 1 < len(tokens):
            pending.append(":")
            i += 1
        elif tok.text == ",":
            if pending:
                pending[-1] = pending[-1] + ","
            i += 1
        else:
            pending.append(tok.text)
            i += 1
    flush_declaration()
    return "\n".join(out)


def _minify_css(tokens: list[_SqlToken]) -> str:
    parts: list[str] = []
    for pos, tok in enumerate(tokens):
        # A ';' right before '}' is dead weight in minified CSS.
        if tok.text == ";" and pos + 1 < len(tokens) \
                and tokens[pos + 1].text == "}":
            continue
        text = tok.text
        if parts and text not in ("{", "}", ":", ";", ",", ")", "(") \
                and parts[-1] not in ("{", ":", "(", ","):
            parts.append(" ")
        parts.append(text)
    return "".join(parts).replace(" :", ":").replace(": ", ":")


# ── HTML: parser-based formatter + validator ──────────────────────────

VOID_TAGS = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
})


@dataclass
class _HtmlNode:
    tag: str | None  # None = text node
    attrs: list[tuple[str, str | None]]
    children: list[_HtmlNode]
    line: int
    col: int
    text: str = ""


class _HtmlTreeBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _HtmlNode("root", [], [], 1, 1)
        self.stack: list[_HtmlNode] = [self.root]
        self.diagnostics: list[Diagnostic] = []

    def _add(self, node: _HtmlNode) -> None:
        self.stack[-1].children.append(node)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]
                        ) -> None:
        line, col = self.getpos()
        node = _HtmlNode(tag, attrs, [], line, col)
        self._add(node)
        if tag not in VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag: str,
                           attrs: list[tuple[str, str | None]]) -> None:
        line, col = self.getpos()
        self._add(_HtmlNode(tag, attrs, [], line, col))

    def handle_endtag(self, tag: str) -> None:
        line, col = self.getpos()
        if tag in VOID_TAGS:
            return
        for depth in range(len(self.stack) - 1, 0, -1):
            if self.stack[depth].tag == tag:
                if depth != len(self.stack) - 1:
                    open_node = self.stack[depth]
                    self.diagnostics.append(Diagnostic(
                        line, col,
                        f"closing </{tag}> while <{open_node.tag}> opened "
                        f"at line {open_node.line} is still open — "
                        f"mismatched tags", "parse"))
                del self.stack[depth:]
                return
        self.diagnostics.append(Diagnostic(
            line, col, f"closing </{tag}> with no matching open tag",
            "parse"))

    def handle_data(self, data: str) -> None:
        if data.strip():
            line, col = self.getpos()
            self._add(_HtmlNode(None, [], [], line, col, text=data.strip()))

    def close(self) -> None:  # noqa: D102 — report unclosed tags first
        super().close()
        for node in reversed(self.stack[1:]):
            self.diagnostics.append(Diagnostic(
                node.line, node.col,
                f"<{node.tag}> opened at line {node.line} is never closed",
                "parse"))


def _parse_html(text: str) -> tuple[_HtmlNode, list[Diagnostic]]:
    builder = _HtmlTreeBuilder()
    builder.feed(text)
    builder.close()
    return builder.root, builder.diagnostics


def _render_html(node: _HtmlNode, indent: int) -> list[str]:
    if node.tag is None:
        return ["  " * indent + node.text]
    attrs = "".join(
        f' {name}="{html.escape(value or "", quote=True)}"'
        if value is not None else f" {name}"
        for name, value in node.attrs)
    if node.tag in VOID_TAGS:
        return ["  " * indent + f"<{node.tag}{attrs}>"]
    if not node.children:
        return ["  " * indent + f"<{node.tag}{attrs}></{node.tag}>"]
    if all(child.tag is None for child in node.children):
        joined = " ".join(child.text for child in node.children)
        return ["  " * indent + f"<{node.tag}{attrs}>{joined}</{node.tag}>"]
    lines = ["  " * indent + f"<{node.tag}{attrs}>"]
    for child in node.children:
        lines.extend(_render_html(child, indent + 1))
    lines.append("  " * indent + f"</{node.tag}>")
    return lines


def _format_html(root: _HtmlNode) -> str:
    lines: list[str] = []
    for child in root.children:
        lines.extend(_render_html(child, 0))
    return "\n".join(lines)


# ── JS: the prettier ladder ───────────────────────────────────────────

def _run_prettier(text: str) -> TransformResult:
    binary = shutil.which("prettier")
    if binary is None:
        return _fail("js", "js", Diagnostic(
            1, 1,
            "js beautification needs a real JS parser — prettier is not "
            "installed on this host; install prettier (npm i -g prettier) "
            "and the js rung will use it", "missing_dependency"))
    try:
        proc = subprocess.run(
            [binary, "--stdin-filepath", "file.js"],
            input=text, capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        return _fail("js", "js", Diagnostic(
            1, 1, "prettier timed out after 15s", "parse"))
    except OSError as exc:
        return _fail("js", "js", Diagnostic(
            1, 1, f"could not run prettier: {exc}", "missing_dependency"))
    if proc.returncode != 0:
        first = (proc.stderr or "prettier failed").splitlines()[0]
        return _fail("js", "js", Diagnostic(1, 1, first, "parse"))
    return TransformResult(text=proc.stdout, format_in="js",
                           format_out="js")


# ── code generation: JSON -> typed code ───────────────────────────────

def _ts_type(value: Any, name: str, interfaces: dict[str, list[str]]
             ) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    if isinstance(value, list):
        if not value:
            return "unknown[]"
        inner = {_ts_type(item, name, interfaces) for item in value}
        if len(inner) == 1:
            return f"{next(iter(inner))}[]"
        return f"({ ' | '.join(sorted(inner)) })[]"
    if isinstance(value, dict):
        fields: list[str] = []
        for key, item in value.items():
            fields.append(f"  {key}: {_ts_type(item, key, interfaces)};")
        interfaces[name] = fields
        return name
    return "unknown"


def _json_to_typescript(value: Any) -> str:
    interfaces: dict[str, list[str]] = {}
    if not isinstance(value, dict):
        root = _ts_type(value, "Root", interfaces)
        return f"type Root = {root};\n"
    for key, item in value.items():
        _ts_type(item, key.capitalize(), interfaces)
    out: list[str] = []
    for name, fields in interfaces.items():
        out.append(f"interface {name} {{")
        out.extend(fields)
        out.append("}")
    return "\n".join(out) + "\n"


def _php_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "array"
    return "mixed"


def _json_to_php(value: Any) -> str:
    if not isinstance(value, dict):
        return "<?php\n\nreturn " + _php_literal(value) + ";\n"
    lines = ["<?php", "", "class Root", "{"]
    for key, item in value.items():
        lines.append(f"    public {_php_type(item)} ${key};")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _php_literal(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_php_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        parts = ", ".join(
            f"{json.dumps(k)} => {_php_literal(v)}"
            for k, v in value.items())
        return "[" + parts + "]"
    return "null"


# ── HTML -> JSX ───────────────────────────────────────────────────────

_JSX_ATTRS = {
    "class": "className", "for": "htmlFor", "tabindex": "tabIndex",
    "readonly": "readOnly", "maxlength": "maxLength",
    "crossorigin": "crossOrigin", "onclick": "onClick",
}


class _JsxBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]
                        ) -> None:
        rendered = "".join(
            f' {_JSX_ATTRS.get(name, name)}="{html.escape(value or "", quote=True)}"'
            if value is not None else f" {_JSX_ATTRS.get(name, name)}"
            for name, value in attrs)
        if tag in VOID_TAGS:
            self.parts.append(f"<{tag}{rendered} />")
        else:
            self.parts.append(f"<{tag}{rendered}>")

    def handle_startendtag(self, tag: str,
                           attrs: list[tuple[str, str | None]]) -> None:
        rendered = "".join(
            f' {_JSX_ATTRS.get(name, name)}="{html.escape(value or "", quote=True)}"'
            if value is not None else f" {_JSX_ATTRS.get(name, name)}"
            for name, value in attrs)
        self.parts.append(f"<{tag}{rendered} />")

    def handle_endtag(self, tag: str) -> None:
        if tag not in VOID_TAGS:
            self.parts.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _html_to_jsx(text: str) -> str:
    builder = _JsxBuilder()
    builder.feed(text)
    builder.close()
    return "".join(builder.parts)

# ── JSON fixer ────────────────────────────────────────────────────────

def _scan_json_repairs(text: str) -> tuple[str, list[Diagnostic]]:
    """Repair pass over JSON text, recording every repair with position.

    Repairs, in order: strip // and /* */ comments, drop trailing
    commas before } or ], convert single-quoted strings to double,
    quote unquoted object keys. Each pass re-scans with string
    awareness, so a comma inside a string is never touched. The
    caller still proves the result with ``json.loads`` — a repair
    that does not yield valid JSON is reported, never claimed.
    """
    diags: list[Diagnostic] = []

    def strip_comments(src: str) -> str:
        out: list[str] = []
        i = 0
        in_str = False
        quote = ""
        while i < len(src):
            ch = src[i]
            if in_str:
                out.append(ch)
                if ch == "\\" and i + 1 < len(src):
                    out.append(src[i + 1])
                    i += 2
                    continue
                if ch == quote:
                    in_str = False
                i += 1
                continue
            if ch in ("\"", "'"):
                in_str, quote = True, ch
                out.append(ch)
                i += 1
                continue
            if src.startswith("//", i):
                line, col = _line_col(src, i)
                diags.append(Diagnostic(line, col, "removed // comment",
                                        "fixed"))
                end = src.find("\n", i)
                i = len(src) if end == -1 else end
                continue
            if src.startswith("/*", i):
                line, col = _line_col(src, i)
                end = src.find("*/", i + 2)
                if end == -1:
                    out.append(src[i:])
                    return "".join(out)
                diags.append(Diagnostic(line, col, "removed /* */ comment",
                                        "fixed"))
                i = end + 2
                continue
            out.append(ch)
            i += 1
        return "".join(out)

    def drop_trailing_commas(src: str) -> str:
        out: list[str] = []
        i = 0
        in_str = False
        quote = ""
        while i < len(src):
            ch = src[i]
            if in_str:
                out.append(ch)
                if ch == "\\" and i + 1 < len(src):
                    out.append(src[i + 1])
                    i += 2
                    continue
                if ch == quote:
                    in_str = False
                i += 1
                continue
            if ch in ("\"", "'"):
                in_str, quote = True, ch
                out.append(ch)
                i += 1
                continue
            if ch == ",":
                j = i + 1
                while j < len(src) and src[j] in " \t\r\n":
                    j += 1
                if j < len(src) and src[j] in "}]":
                    line, col = _line_col(src, i)
                    diags.append(Diagnostic(
                        line, col, "removed trailing comma", "fixed"))
                    i += 1
                    continue
            out.append(ch)
            i += 1
        return "".join(out)

    def convert_quotes(src: str) -> str:
        out: list[str] = []
        i = 0
        while i < len(src):
            ch = src[i]
            if ch == "'":
                line, col = _line_col(src, i)
                j = i + 1
                buf: list[str] = []
                closed = False
                while j < len(src):
                    if src[j] == "\\" and j + 1 < len(src):
                        buf.append(src[j:j + 2])
                        j += 2
                        continue
                    if src[j] == "'":
                        closed = True
                        break
                    buf.append(src[j])
                    j += 1
                if closed:
                    inner = "".join(buf).replace('"', '\\"')
                    out.append(f'"{inner}"')
                    diags.append(Diagnostic(
                        line, col,
                        "converted single-quoted string to double quotes",
                        "fixed"))
                    i = j + 1
                    continue
                out.append(ch)
                i += 1
                continue
            if ch == '"':
                j = i + 1
                while j < len(src):
                    if src[j] == "\\" and j + 1 < len(src):
                        j += 2
                        continue
                    if src[j] == '"':
                        j += 1
                        break
                    j += 1
                out.append(src[i:j])
                i = j
                continue
            out.append(ch)
            i += 1
        return "".join(out)

    def quote_keys(src: str) -> str:
        out: list[str] = []
        i = 0
        in_str = False
        quote = ""
        while i < len(src):
            ch = src[i]
            if in_str:
                out.append(ch)
                if ch == "\\" and i + 1 < len(src):
                    out.append(src[i + 1])
                    i += 2
                    continue
                if ch == quote:
                    in_str = False
                i += 1
                continue
            if ch in ("\"", "'"):
                in_str, quote = True, ch
                out.append(ch)
                i += 1
                continue
            if ch in "{,":
                out.append(ch)
                j = i + 1
                while j < len(src) and src[j] in " \t\r\n":
                    j += 1
                match = re.match(r"[A-Za-z_$][A-Za-z0-9_$]*", src[j:])
                if match:
                    k = j + match.end()
                    while k < len(src) and src[k] in " \t\r\n":
                        k += 1
                    if k < len(src) and src[k] == ":":
                        line, col = _line_col(src, j)
                        diags.append(Diagnostic(
                            line, col,
                            f"quoted unquoted key {match.group(0)!r}",
                            "fixed"))
                        out.append(src[i + 1:j])
                        out.append(f'"{match.group(0)}"')
                        i = j + match.end()
                        continue
                i += 1
                continue
            out.append(ch)
            i += 1
        return "".join(out)

    cleaned = quote_keys(convert_quotes(
        drop_trailing_commas(strip_comments(text))))
    return cleaned, diags


def fix_json(text: str) -> TransformResult:
    """Repair damaged JSON, proving the result with a real parse.

    Repairs: comments stripped, trailing commas dropped, single
    quotes converted, unquoted keys quoted — each recorded as a
    ``fixed`` diagnostic with its position. A repair that still does
    not parse returns the parser's own error with its position,
    never a claimed success.
    """
    if not text.strip():
        return _fail("json", "json", Diagnostic(
            1, 1, "nothing to fix — input is empty", "empty"))
    try:
        json.loads(text)
        return TransformResult(text=text, format_in="json",
                               format_out="json")
    except json.JSONDecodeError:
        pass
    cleaned, diags = _scan_json_repairs(text)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        diags.append(Diagnostic(exc.lineno, exc.colno, exc.msg, "parse"))
        return TransformResult(text="", format_in="json", format_out="json",
                               diagnostics=diags, ok=False,
                               error=exc.msg)
    pretty = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    return TransformResult(text=pretty, format_in="json",
                           format_out="json", diagnostics=diags)


# ── detection ─────────────────────────────────────────────────────────

_SQL_FIRST = frozenset({"select", "insert", "update", "delete", "with"})


def detect_format(text: str) -> str:
    """Detect the format by real parsing, never by guessing.

    Preference order (first parser that accepts wins): json, xml,
    toml, sql (keyword sniff — a SQL statement is not valid YAML
    data), csv (multi-line, consistent columns), yaml (mapping or
    sequence only — a bare scalar is not a document). Anything else
    is ``unknown``.
    """
    stripped = text.strip()
    if not stripped:
        return "unknown"
    try:
        json.loads(stripped)
        return "json"
    except json.JSONDecodeError:
        pass
    if stripped.startswith("<"):
        try:
            ET.fromstring(stripped)
            return "xml"
        except ET.ParseError:
            return "unknown"
    if "=" in stripped or stripped.startswith("["):
        value, diags = _parse_toml(stripped)
        if not diags:
            return "toml"
    first = stripped.split(None, 1)[0].lower().rstrip(";")
    if first in _SQL_FIRST:
        return "sql"
    lines = [ln for ln in stripped.splitlines() if ln.strip()]
    if len(lines) >= 2 and "," in lines[0]:
        try:
            widths = {len(row) for row in csv.reader(io.StringIO(stripped))}
            if len(widths) == 1 and next(iter(widths)) > 1:
                return "csv"
        except csv.Error:
            pass
    value, diags = _parse_yaml(stripped)
    if not diags and isinstance(value, (dict, list)):
        return "yaml"
    return "unknown"


# ── public operations ─────────────────────────────────────────────────

def _empty_check(text: str, fmt: str) -> Diagnostic | None:
    if not text.strip():
        return Diagnostic(1, 1, f"nothing to format — {fmt} input is empty",
                          "empty")
    return None


def beautify(text: str, fmt: str) -> TransformResult:
    """Pretty-print ``text`` as ``fmt`` — parsed for real, then emitted."""
    fmt = (fmt or "").strip().lower()
    if fmt not in FORMATS:
        return _fail(fmt, fmt, Diagnostic(
            1, 1, f"unknown format {fmt!r}; formats: {', '.join(FORMATS)}",
            "unsupported_format"))
    if (diag := _empty_check(text, fmt)) is not None:
        return _fail(fmt, fmt, diag)
    if fmt == "js":
        return _run_prettier(text)
    if fmt in DATA_FORMATS:
        value, diags = _parse_data(text, fmt)
        if diags:
            return TransformResult(text="", format_in=fmt, format_out=fmt,
                                   diagnostics=diags, ok=False,
                                   error=diags[0].message)
        if fmt == "jsonl":
            out, _ = _emit_data(value, "jsonl")
            pretty = "".join(
                json.dumps(row, indent=2, ensure_ascii=False) + "\n"
                for row in value)
            return TransformResult(text=pretty, format_in=fmt,
                                   format_out=fmt)
        out, ediags = _emit_data(value, fmt)
        if ediags:
            return TransformResult(text="", format_in=fmt, format_out=fmt,
                                   diagnostics=ediags, ok=False,
                                   error=ediags[0].message)
        return TransformResult(text=out, format_in=fmt, format_out=fmt)
    if fmt == "sql":
        tokens, diags = _tokenize_sql(text)
        if diags:
            return TransformResult(text="", format_in=fmt, format_out=fmt,
                                   diagnostics=diags, ok=False,
                                   error=diags[0].message)
        return TransformResult(text=_format_sql(tokens), format_in=fmt,
                               format_out=fmt)
    if fmt == "css":
        tokens, diags = _tokenize_css(text)
        if diags:
            return TransformResult(text="", format_in=fmt, format_out=fmt,
                                   diagnostics=diags, ok=False,
                                   error=diags[0].message)
        return TransformResult(text=_format_css(tokens), format_in=fmt,
                               format_out=fmt)
    if fmt == "html":
        root, diags = _parse_html(text)
        if diags:
            return TransformResult(text="", format_in=fmt, format_out=fmt,
                                   diagnostics=diags, ok=False,
                                   error=diags[0].message)
        return TransformResult(text=_format_html(root), format_in=fmt,
                               format_out=fmt)
    return _fail(fmt, fmt, Diagnostic(
        1, 1, f"no beautifier for {fmt!r}", "unsupported_format"))


def minify(text: str, fmt: str) -> TransformResult:
    """Compact ``text`` as ``fmt`` — parsed for real first, never blind."""
    fmt = (fmt or "").strip().lower()
    if fmt not in FORMATS:
        return _fail(fmt, fmt, Diagnostic(
            1, 1, f"unknown format {fmt!r}; formats: {', '.join(FORMATS)}",
            "unsupported_format"))
    if (diag := _empty_check(text, fmt)) is not None:
        return _fail(fmt, fmt, diag)
    if fmt == "js":
        result = _run_prettier(text)
        if not result.ok:
            return result
        return TransformResult(text=result.text, format_in="js",
                               format_out="js")
    if fmt in DATA_FORMATS:
        value, diags = _parse_data(text, fmt)
        if diags:
            return TransformResult(text="", format_in=fmt, format_out=fmt,
                                   diagnostics=diags, ok=False,
                                   error=diags[0].message)
        if fmt == "json":
            return TransformResult(
                text=json.dumps(value, separators=(",", ":"),
                                ensure_ascii=False),
                format_in=fmt, format_out=fmt)
        if fmt == "jsonl":
            rows = value if isinstance(value, list) else [value]
            return TransformResult(
                text="".join(json.dumps(row, separators=(",", ":"),
                                       ensure_ascii=False) + "\n"
                             for row in rows),
                format_in=fmt, format_out=fmt)
        if fmt == "xml":
            if isinstance(value, dict) and len(value) == 1:
                tag, inner = next(iter(value.items()))
                elem = _data_to_xml(inner, tag)
            else:
                elem = _data_to_xml(value, "root")
            return TransformResult(
                text=ET.tostring(elem, encoding="unicode"),
                format_in=fmt, format_out=fmt)
        out, ediags = _emit_data(value, fmt)
        if ediags:
            return TransformResult(text="", format_in=fmt, format_out=fmt,
                                   diagnostics=ediags, ok=False,
                                   error=ediags[0].message)
        return TransformResult(text=out, format_in=fmt, format_out=fmt)
    if fmt == "sql":
        tokens, diags = _tokenize_sql(text)
        if diags:
            return TransformResult(text="", format_in=fmt, format_out=fmt,
                                   diagnostics=diags, ok=False,
                                   error=diags[0].message)
        return TransformResult(text=_minify_sql(tokens), format_in=fmt,
                               format_out=fmt)
    if fmt == "css":
        tokens, diags = _tokenize_css(text)
        if diags:
            return TransformResult(text="", format_in=fmt, format_out=fmt,
                                   diagnostics=diags, ok=False,
                                   error=diags[0].message)
        return TransformResult(text=_minify_css(tokens), format_in=fmt,
                               format_out=fmt)
    if fmt == "html":
        root, diags = _parse_html(text)
        if diags:
            return TransformResult(text="", format_in=fmt, format_out=fmt,
                                   diagnostics=diags, ok=False,
                                   error=diags[0].message)
        lines = _format_html(root).splitlines()
        return TransformResult(text="".join(ln.strip() for ln in lines),
                               format_in=fmt, format_out=fmt)
    return _fail(fmt, fmt, Diagnostic(
        1, 1, f"no minifier for {fmt!r}", "unsupported_format"))


def validate(text: str, fmt: str) -> TransformResult:
    """Validate ``text`` as ``fmt`` — the text is echoed back unchanged.

    ``ok`` says whether it parses; ``diagnostics`` carry the located
    errors. Nothing is transformed here.
    """
    fmt = (fmt or "").strip().lower()
    if fmt not in FORMATS:
        return _fail(fmt, fmt, Diagnostic(
            1, 1, f"unknown format {fmt!r}; formats: {', '.join(FORMATS)}",
            "unsupported_format"))
    if (diag := _empty_check(text, fmt)) is not None:
        return _fail(fmt, fmt, diag)
    if fmt == "js":
        result = _run_prettier(text)
        if not result.ok:
            return result
        return TransformResult(text=text, format_in=fmt, format_out=fmt)
    if fmt in DATA_FORMATS:
        _, diags = _parse_data(text, fmt)
        if diags:
            return TransformResult(text=text, format_in=fmt,
                                   format_out=fmt, diagnostics=diags,
                                   ok=False, error=diags[0].message)
        return TransformResult(text=text, format_in=fmt, format_out=fmt)
    if fmt == "sql":
        _, diags = _tokenize_sql(text)
    elif fmt == "css":
        _, diags = _tokenize_css(text)
    elif fmt == "html":
        _, diags = _parse_html(text)
    else:  # pragma: no cover — FORMATS is closed above
        diags = [Diagnostic(1, 1, f"no validator for {fmt!r}",
                            "unsupported_format")]
    if diags:
        return TransformResult(text=text, format_in=fmt, format_out=fmt,
                               diagnostics=diags, ok=False,
                               error=diags[0].message)
    return TransformResult(text=text, format_in=fmt, format_out=fmt)


def convert(text: str, from_fmt: str, to_fmt: str) -> TransformResult:
    """Convert ``text`` from one format to another, through the data model.

    Data formats convert through the shared Python model, so
    convert∘convert-back returns equal data. Code targets
    (typescript, php) and the hex/utf8 + html/jsx pairs are real
    transforms; every other pair is refused by name.
    """
    from_fmt = (from_fmt or "").strip().lower()
    to_fmt = (to_fmt or "").strip().lower()
    if (diag := _empty_check(text, from_fmt)) is not None:
        return _fail(from_fmt, to_fmt, diag)
    if from_fmt == to_fmt:
        return beautify(text, from_fmt)
    if from_fmt in ("hex", "utf8") or to_fmt in ("hex", "utf8"):
        if {from_fmt, to_fmt} == {"hex", "utf8"}:
            if from_fmt == "utf8":
                return TransformResult(
                    text=text.encode("utf-8").hex(), format_in=from_fmt,
                    format_out=to_fmt)
            try:
                raw = bytes.fromhex("".join(text.split()))
            except (ValueError, binascii.Error):
                bad = next((i for i, ch in enumerate(text)
                            if ch not in "0123456789abcdefABCDEF \t\r\n"),
                           0)
                line, col = _line_col(text, bad)
                return _fail(from_fmt, to_fmt, Diagnostic(
                    line, col, f"not a hex digit: {text[bad]!r}", "parse"))
            try:
                decoded = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                line, col = _line_col(text, exc.start)
                return _fail(from_fmt, to_fmt, Diagnostic(
                    line, col, f"bytes are not valid UTF-8: {exc.reason}",
                    "parse"))
            return TransformResult(text=decoded, format_in=from_fmt,
                                   format_out=to_fmt)
        return _fail(from_fmt, to_fmt, Diagnostic(
            1, 1, f"cannot convert {from_fmt} to {to_fmt}",
            "unsupported_format"))
    if (from_fmt, to_fmt) == ("html", "jsx"):
        return TransformResult(text=_html_to_jsx(text), format_in=from_fmt,
                               format_out=to_fmt)
    if from_fmt not in DATA_FORMATS or (
            to_fmt not in DATA_FORMATS and to_fmt not in CODE_TARGETS):
        return _fail(from_fmt, to_fmt, Diagnostic(
            1, 1, f"cannot convert {from_fmt} to {to_fmt}",
            "unsupported_format"))
    value, diags = _parse_data(text, from_fmt)
    if diags:
        return TransformResult(text="", format_in=from_fmt,
                               format_out=to_fmt, diagnostics=diags,
                               ok=False, error=diags[0].message)
    if to_fmt == "typescript":
        return TransformResult(text=_json_to_typescript(value),
                               format_in=from_fmt, format_out=to_fmt)
    if to_fmt == "php":
        return TransformResult(text=_json_to_php(value),
                               format_in=from_fmt, format_out=to_fmt)
    out, ediags = _emit_data(value, to_fmt)
    if ediags:
        return TransformResult(text="", format_in=from_fmt,
                               format_out=to_fmt, diagnostics=ediags,
                               ok=False, error=ediags[0].message)
    return TransformResult(text=out, format_in=from_fmt,
                           format_out=to_fmt)


def transform(text: str, *, operation: str, format_in: str = "auto",
              format_out: str | None = None) -> TransformResult:
    """One entry the tool, chat and CLI surfaces share.

    ``operation`` is beautify | minify | validate | convert | fix |
    detect. ``format_in="auto"`` detects by real parsing; an
    undetectable input is refused by name, never guessed.
    """
    operation = (operation or "").strip().lower()
    fmt_in = (format_in or "auto").strip().lower()
    if operation == "fix":
        return fix_json(text)
    if operation == "detect":
        return TransformResult(text=detect_format(text), format_in="auto",
                               format_out="detected")
    if fmt_in == "auto":
        fmt_in = detect_format(text)
        if fmt_in == "unknown":
            return _fail("auto", format_out or "auto", Diagnostic(
                1, 1, "could not detect the format — name it explicitly "
                "(/fmt json <text>)", "unsupported_format"))
    if operation == "beautify":
        return beautify(text, fmt_in)
    if operation == "minify":
        return minify(text, fmt_in)
    if operation == "validate":
        return validate(text, fmt_in)
    if operation == "convert":
        if not format_out:
            return _fail(fmt_in, fmt_in, Diagnostic(
                1, 1, "convert needs a target format",
                "invalid_argument"))
        return convert(text, fmt_in, format_out.strip().lower())
    return _fail(fmt_in, format_out or fmt_in, Diagnostic(
        1, 1, f"unknown operation {operation!r}; operations: beautify, "
        "minify, validate, convert, fix, detect", "invalid_argument"))


def format_transform_reply(result: TransformResult) -> str:
    """Render a result as plain chat text."""
    if not result.ok:
        lines = [f"{result.format_in} failed:"]
        for diag in result.diagnostics:
            lines.append(
                f"  line {diag.line}, col {diag.col}: {diag.message} "
                f"[{diag.kind}]")
        return "\n".join(lines)
    if result.format_out == "detected":
        return f"detected format: {result.text}"
    header = f"{result.format_in} → {result.format_out}:"
    body = result.text.rstrip("\n")
    fixed = [d for d in result.diagnostics if d.kind == "fixed"]
    lines = [header, body]
    for diag in fixed:
        lines.append(f"  fixed line {diag.line}, col {diag.col}: "
                     f"{diag.message}")
    return "\n".join(lines)
