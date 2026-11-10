"""Section-wise document structure for translation (doc 19 §4, T2).

Doc 19's translator rule: *a document keeps its shape.*  Chat text
translates as one string; a document — headings, lists, tables,
code, quotes, links — translates section by section over a parsed
structure, and the structure is never itself up for translation:

- **Code fences pass through byte-identical.**  So do HTML
  ``<pre>``/``<script>``/``<style>`` spans and comments.
- **Tables travel cell by cell** (the doc's own phrase): each cell
  is its own translation unit; the pipes, padding and separator
  row are structure, carried verbatim.
- **Links keep their destinations.**  Link text translates; the
  ``](url)`` machinery, inline code spans, footnote references,
  autolinks, bare URLs and HTML tags inside a prose run are
  replaced by placeholder tokens (``⟦P0⟧`` …) before the rung sees
  the text and restored after — with verification: a unit whose
  placeholders do not all survive exactly once is NOT rendered
  from the rung's output at all; the caller falls back to the
  unit's original text and reports the section untranslated.  A
  mangled placeholder can cost a section its translation; it can
  never corrupt the document.
- **Ticket preserve rules ride here**: ``citations`` and ``ids``
  extend the placeholder set to citation markers and ID-shaped
  tokens (UUIDs, ``ABC-123`` keys).

:func:`parse_document` returns None when the text carries no
structural block (no heading, fence, table, list item, quote or
HTML block tag) — plain prose keeps T1's single-shot path,
byte-identical.  The parser is pure stdlib and total: concatenating
every unit's ``sep`` + ``raw`` reproduces the input exactly, which
the round-trip fixtures pin.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["Unit", "Document", "parse_document",
           "protected_unit_document", "STRUCTURAL_KINDS"]

#: Unit kinds that make a text a *document* (vs plain prose).
STRUCTURAL_KINDS = frozenset(
    {"heading", "code", "table-cell", "list-item", "quote", "html"})


@dataclass
class Unit:
    """One piece of a parsed document.  ``prose is None`` marks a
    piece that never translates (code, separators, blank lines,
    tag-only HTML segments); otherwise ``prose`` is the payload the
    rung sees — with structure placeholders already substituted —
    and ``pre``/``post`` are the verbatim frame re-applied around
    the translation (``## ``, ``- ``, cell padding, …)."""

    kind: str
    raw: str
    sep: str = ""
    pre: str = ""
    post: str = ""
    prose: str | None = None
    spans: dict[str, str] = field(default_factory=dict)
    group: str = ""

    def render(self, translated_prose: str) -> str | None:
        """Frame a translation back into the document, restoring
        every placeholder.  None when any placeholder did not
        survive exactly once — the caller must fall back to
        ``raw`` rather than ship a corrupted section."""
        if self.prose is None:
            return self.raw
        out = translated_prose
        for token, original in self.spans.items():
            if out.count(token) != 1:
                return None
            out = out.replace(token, original)
        return self.pre + out + self.post


@dataclass
class Document:
    units: list[Unit]

    def reassemble(self, rendered: dict[int, str]) -> str:
        """Join the document back together; units missing from
        ``rendered`` contribute their original raw text."""
        return "".join(unit.sep + rendered.get(idx, unit.raw)
                       for idx, unit in enumerate(self.units))


# ----------------------------------------------------------------------
# inline placeholder protection
# ----------------------------------------------------------------------

_CODE_SPAN_RE = re.compile(r"(`+)(.+?)\1", re.DOTALL)
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]*)\)")
_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
_REFLINK_RE = re.compile(r"\[([^\]]*)\]\[([^\]]*)\]")
_FOOTNOTE_RE = re.compile(r"\[\^[^\]]+\]")
_AUTOLINK_RE = re.compile(r"<[A-Za-z][A-Za-z0-9+.-]*:[^>\s]*>")
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
_BARE_URL_RE = re.compile(r"https?://[^\s<>\]\)]+")
_CITATION_RE = re.compile(r"\[@[^\]]+\]|\[\d+(?:\s*,\s*\d+)*\]")
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_TICKET_ID_RE = re.compile(r"\b[A-Z]{2,}-\d+\b")
_URL_TRAILING = ".,;:!?*"


def _protect_inline(text: str, *, citations: bool = False,
                    ids: bool = False,
                    spans: dict[str, str] | None = None
                    ) -> tuple[str, dict[str, str]]:
    """Substitute structure spans with ``⟦P{n}⟧`` tokens.  Link and
    image *text* stays in the prose (it translates); only the
    destination machinery is protected.  A caller-owned ``spans``
    dict keeps token numbering continuous across pieces."""
    if spans is None:
        spans = {}
    out: list[str] = []
    pos = 0

    def token(original: str) -> str:
        mark = f"⟦P{len(spans)}⟧"
        spans[mark] = original
        return mark

    while pos < len(text):
        match = _CODE_SPAN_RE.match(text, pos)
        if match:
            out.append(token(match.group(0)))
            pos = match.end()
            continue
        match = _IMAGE_RE.match(text, pos) or _LINK_RE.match(text, pos)
        if match:
            opener = "![" if text[pos] == "!" else "["
            out.append(opener)
            out.append(match.group(1))            # the text translates
            split = match.start() + len(opener) + len(match.group(1))
            out.append(token(text[split:match.end()]))
            pos = match.end()
            continue
        match = _REFLINK_RE.match(text, pos)
        if match:
            out.append("[")
            out.append(match.group(1))
            split = match.start() + 1 + len(match.group(1))
            out.append(token(text[split:match.end()]))
            pos = match.end()
            continue
        match = (_FOOTNOTE_RE.match(text, pos)
                 or _AUTOLINK_RE.match(text, pos)
                 or _HTML_TAG_RE.match(text, pos))
        if match:
            out.append(token(match.group(0)))
            pos = match.end()
            continue
        match = _BARE_URL_RE.match(text, pos)
        if match:
            url = match.group(0).rstrip(_URL_TRAILING)
            if url:
                out.append(token(url))
                pos += len(url)
                continue
        if citations:
            match = _CITATION_RE.match(text, pos)
            if match:
                out.append(token(match.group(0)))
                pos = match.end()
                continue
        if ids:
            match = (_UUID_RE.match(text, pos)
                     or _TICKET_ID_RE.match(text, pos))
            if match:
                out.append(token(match.group(0)))
                pos = match.end()
                continue
        out.append(text[pos])
        pos += 1
    return "".join(out), spans


# ----------------------------------------------------------------------
# markdown
# ----------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})([ \t]+)(.*?)([ \t]*)$")
_ANCHOR_RE = re.compile(r"\s*\{#[^}\s]+\}\s*$")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_LIST_RE = re.compile(r"^(\s*)([-*+]|\d{1,9}[.)])([ \t]+)(.*)$")
_CHECKBOX_RE = re.compile(r"^\[[ xX]\][ \t]+")
_QUOTE_RE = re.compile(r"^(\s*>\s?)(.*)$")
_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$")


def _is_separator(line: str) -> bool:
    return bool(_SEPARATOR_RE.match(line)) and "-" in line


def _starts_block(lines: list[str], i: int) -> bool:
    line = lines[i]
    if _FENCE_RE.match(line) or _HEADING_RE.match(line) \
            or _LIST_RE.match(line) or _QUOTE_RE.match(line):
        return True
    return ("|" in line and i + 1 < len(lines)
            and _is_separator(lines[i + 1]))


def _split_row(line: str) -> list[str]:
    """Split a table row on unescaped pipes, keeping the pipes as
    their own pieces so the row partition is exact."""
    pieces: list[str] = []
    buf = ""
    i = 0
    while i < len(line):
        if line[i] == "\\" and i + 1 < len(line) and line[i + 1] == "|":
            buf += "\\|"
            i += 2
            continue
        if line[i] == "|":
            pieces.append(buf)
            pieces.append("|")
            buf = ""
            i += 1
            continue
        buf += line[i]
        i += 1
    pieces.append(buf)
    return pieces


def _prose_unit(kind: str, raw: str, sep: str, pre: str, post: str,
                core: str, *, citations: bool, ids: bool,
                group: str = "") -> Unit:
    prose, spans = _protect_inline(core, citations=citations, ids=ids)
    if not prose.strip():
        return Unit(kind="verbatim", raw=raw, sep=sep)
    return Unit(kind=kind, raw=raw, sep=sep, pre=pre, post=post,
                prose=prose, spans=spans, group=group)


def _parse_markdown(text: str, *, citations: bool, ids: bool
                    ) -> Document | None:
    lines = text.split("\n")
    units: list[Unit] = []
    table_no = 0
    i = 0

    def sep_for() -> str:
        return "" if not units else "\n"

    while i < len(lines):
        line = lines[i]
        if not line.strip():
            units.append(Unit(kind="verbatim", raw=line, sep=sep_for()))
            i += 1
            continue
        fence = _FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)
            j = i + 1
            while j < len(lines):
                closing = lines[j].strip()
                if closing and set(closing) == {marker[0]} \
                        and len(closing) >= len(marker):
                    break
                j += 1
            j = min(j, len(lines) - 1)
            units.append(Unit(kind="code", sep=sep_for(),
                              raw="\n".join(lines[i:j + 1])))
            i = j + 1
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            core = heading.group(3)
            post = heading.group(4)
            anchor = _ANCHOR_RE.search(core)
            if anchor:
                post = core[anchor.start():] + post
                core = core[:anchor.start()]
            units.append(_prose_unit(
                "heading", line, sep_for(),
                heading.group(1) + heading.group(2), post, core,
                citations=citations, ids=ids))
            i += 1
            continue
        if "|" in line and i + 1 < len(lines) \
                and _is_separator(lines[i + 1]):
            table_no += 1
            group = f"table{table_no}"
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                if _is_separator(lines[i]):
                    units.append(Unit(kind="verbatim", raw=lines[i],
                                      sep=sep_for(), group=group))
                    i += 1
                    continue
                first = True
                for piece in _split_row(lines[i]):
                    piece_sep = sep_for() if first else ""
                    first = False
                    if piece == "|":
                        units.append(Unit(kind="verbatim", raw="|",
                                          sep=piece_sep, group=group))
                        continue
                    core = piece.strip()
                    lead = piece[:len(piece) - len(piece.lstrip())]
                    trail = piece[len(piece.rstrip()):]
                    if not core:
                        units.append(Unit(kind="verbatim", raw=piece,
                                          sep=piece_sep, group=group))
                        continue
                    units.append(_prose_unit(
                        "table-cell", piece, piece_sep, lead, trail,
                        core, citations=citations, ids=ids,
                        group=group))
                i += 1
            continue
        item = _LIST_RE.match(line)
        if item:
            pre = item.group(1) + item.group(2) + item.group(3)
            core = item.group(4)
            checkbox = _CHECKBOX_RE.match(core)
            if checkbox:
                pre += checkbox.group(0)
                core = core[checkbox.end():]
            raw_lines = [line]
            j = i + 1
            while j < len(lines) and lines[j].strip() \
                    and (lines[j][0] in " \t") \
                    and not _starts_block(lines, j):
                core += "\n" + lines[j].strip()
                raw_lines.append(lines[j])
                j += 1
            units.append(_prose_unit(
                "list-item", "\n".join(raw_lines), sep_for(), pre, "",
                core, citations=citations, ids=ids))
            i = j
            continue
        if _QUOTE_RE.match(line):
            raw_lines = []
            contents = []
            pre = ""
            while i < len(lines):
                quote = _QUOTE_RE.match(lines[i])
                if not quote:
                    break
                if not raw_lines:
                    pre = quote.group(1)
                raw_lines.append(lines[i])
                contents.append(quote.group(2))
                i += 1
            units.append(_prose_unit(
                "quote", "\n".join(raw_lines), sep_for(), pre, "",
                "\n".join(contents), citations=citations, ids=ids))
            continue
        raw_lines = [line]
        j = i + 1
        while j < len(lines) and lines[j].strip() \
                and not _starts_block(lines, j):
            raw_lines.append(lines[j])
            j += 1
        units.append(_prose_unit(
            "paragraph", "\n".join(raw_lines), sep_for(), "", "",
            "\n".join(raw_lines), citations=citations, ids=ids))
        i = j
    if not any(unit.kind in STRUCTURAL_KINDS for unit in units):
        return None
    return Document(units=units)


# ----------------------------------------------------------------------
# html
# ----------------------------------------------------------------------

_HTML_TOKEN_RE = re.compile(
    r"<!--.*?-->|<(script|style|pre)\b[^>]*>.*?</\1\s*>"
    r"|<[^>]+>|[^<]+",
    re.DOTALL | re.IGNORECASE)
_TAG_NAME_RE = re.compile(r"^</?\s*([A-Za-z][A-Za-z0-9]*)")
_BLOCK_TAGS = frozenset({
    "address", "article", "aside", "blockquote", "dd", "div", "dl",
    "dt", "fieldset", "figcaption", "figure", "footer", "form",
    "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li",
    "main", "nav", "ol", "p", "section", "table", "tbody", "td",
    "tfoot", "th", "thead", "tr", "ul", "title",
})
_HTML_DOC_RE = re.compile(
    r"<\s*(p|div|h[1-6]|ul|ol|li|table|blockquote|section|article|"
    r"header|footer|main|figure|title|tr|td|th)\b", re.IGNORECASE)


def _parse_html(text: str, *, citations: bool, ids: bool
                ) -> Document | None:
    if not _HTML_DOC_RE.search(text):
        return None
    tokens: list[tuple[str, str]] = []  # (flavour, exact text)
    for match in _HTML_TOKEN_RE.finditer(text):
        piece = match.group(0)
        if piece.startswith("<!--") or match.group(1):
            tokens.append(("span", piece))      # verbatim span
        elif piece.startswith("<"):
            tokens.append(("tag", piece))
        else:
            tokens.append(("text", piece))
    segments: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    for flavour, piece in tokens:
        if flavour == "tag":
            name = _TAG_NAME_RE.match(piece)
            if name and name.group(1).lower() in _BLOCK_TAGS and current:
                segments.append(current)
                current = []
        current.append((flavour, piece))
    if current:
        segments.append(current)
    units: list[Unit] = []
    for segment in segments:
        raw = "".join(piece for _, piece in segment)
        if not any(flavour == "text" and piece.strip()
                   for flavour, piece in segment):
            units.append(Unit(kind="verbatim", raw=raw))
            continue
        prose_parts: list[str] = []
        spans: dict[str, str] = {}
        for flavour, piece in segment:
            if flavour == "text":
                protected, _ = _protect_inline(
                    piece, citations=citations, ids=ids, spans=spans)
                prose_parts.append(protected)
            else:
                mark = f"⟦P{len(spans)}⟧"
                spans[mark] = piece
                prose_parts.append(mark)
        units.append(Unit(kind="html", raw=raw,
                          prose="".join(prose_parts), spans=spans))
    if not any(unit.kind == "html" for unit in units):
        return None
    return Document(units=units)


# ----------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------

def parse_document(text: str, *, preserve: tuple[str, ...] = ()
                   ) -> Document | None:
    """Parse ``text`` into a translatable document structure, or
    None when it is plain prose (the caller keeps the single-shot
    path).  ``preserve`` carries the ticket's preserve rules;
    ``citations`` and ``ids`` widen the placeholder set."""
    if not text or not text.strip():
        return None
    citations = "citations" in preserve
    ids = "ids" in preserve
    if _HTML_DOC_RE.search(text):
        return _parse_html(text, citations=citations, ids=ids)
    return _parse_markdown(text, citations=citations, ids=ids)


def protected_unit_document(text: str, *, preserve: tuple[str, ...] = ()
                            ) -> Document | None:
    """Plain prose wrapped as a one-unit document so token-level
    preserve rules (``citations``/``ids``) still apply on the
    single-shot path.  None when the text holds nothing to
    protect."""
    prose, spans = _protect_inline(
        text, citations="citations" in preserve,
        ids="ids" in preserve)
    if not spans:
        return None
    return Document(units=[Unit(kind="paragraph", raw=text,
                                prose=prose, spans=spans)])
