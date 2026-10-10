"""Document comparison: diff two parsed documents (L4).

:func:`compare_documents` produces a structured
:class:`DocumentComparison` — section-level added/removed/changed/moved
sets, table-level changes with cell-level detail, a similarity score, and
a unified text diff.  :func:`diff_documents` is the plain unified-diff
shortcut.  :func:`word_diff` is the inline word-level diff
(diff-match-patch's semantic-cleanup idea, stdlib edition).  Renderers
(:func:`render_html`, :func:`render_terminal`, :func:`render_markdown`)
turn a comparison into presentation-grade output — delta-style, not raw
difflib dumps.  Everything is stdlib ``difflib``; no optional
dependencies.
"""

from __future__ import annotations

import difflib
import html as html_module
import re
from dataclasses import dataclass, field
from typing import Any

from .errors import DocumentError
from .model import Document, full_text

__all__ = [
    "DocumentComparison",
    "compare_documents",
    "diff_documents",
    "render_html",
    "render_markdown",
    "render_terminal",
    "similarity",
    "word_diff",
]


@dataclass
class DocumentComparison:
    """Structured result of comparing two documents."""

    summary: str
    text_changed: bool
    sections_added: list[str] = field(default_factory=list)
    sections_removed: list[str] = field(default_factory=list)
    sections_changed: list[str] = field(default_factory=list)
    tables_added: list[str] = field(default_factory=list)
    tables_removed: list[str] = field(default_factory=list)
    tables_changed: list[str] = field(default_factory=list)
    sections_moved: list[dict[str, Any]] = field(default_factory=list)
    similarity: float = 0.0
    unified_diff: str = ""
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "text_changed": self.text_changed,
            "similarity": self.similarity,
            "sections_added": list(self.sections_added),
            "sections_removed": list(self.sections_removed),
            "sections_changed": list(self.sections_changed),
            "sections_moved": [dict(m) for m in self.sections_moved],
            "tables_added": list(self.tables_added),
            "tables_removed": list(self.tables_removed),
            "tables_changed": list(self.tables_changed),
            "unified_diff": self.unified_diff,
            "stats": dict(self.stats),
        }


def _section_key(index: int, heading: str) -> str:
    return heading.strip() or f"<section {index + 1}>"


def _table_signature(table) -> list[str]:
    """Row strings (headers + rows) used for row-level diffing."""
    lines = []
    if table.headers:
        lines.append(" | ".join(table.headers))
    lines.extend(" | ".join(row) for row in table.rows)
    return lines


def _describe_table_change(before: list[str], after: list[str]) -> str:
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    added = removed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            added += j2 - j1
        elif tag == "delete":
            removed += i2 - i1
        elif tag == "replace":
            added += j2 - j1
            removed += i2 - i1
    parts = []
    if added:
        parts.append(f"{added} row(s) added")
    if removed:
        parts.append(f"{removed} row(s) removed")
    return ", ".join(parts) or "content changed"


def diff_documents(first: Document, second: Document, *,
                   context: int = 3) -> str:
    """Unified diff of the two documents' full text.

    Returns "" when the texts are identical.  Raises DocumentError when
    either document has no text to diff.
    """
    if not isinstance(first, Document) or not isinstance(second, Document):
        raise DocumentError("diff_documents needs two Document objects")
    a_text, b_text = full_text(first), full_text(second)
    if not a_text.strip() or not b_text.strip():
        raise DocumentError("cannot diff documents with no text")
    diff = difflib.unified_diff(
        a_text.splitlines(), b_text.splitlines(),
        fromfile=first.title or first.id or "a",
        tofile=second.title or second.id or "b",
        n=context, lineterm="",
    )
    return "\n".join(diff)


def word_diff(before: str, after: str) -> list[tuple[str, str]]:
    """Inline word-level diff: [(op, text)] with op in equal/insert/delete.

    diff-match-patch's semantic-cleanup idea, stdlib edition: the raw
    word alignment is post-processed so trivial equal-runs trapped between
    a deletion and an insertion (e.g. a single shared word) are absorbed
    into the change, producing human-readable word-aligned edits instead
    of character-soup fragments.
    """
    if not isinstance(before, str) or not isinstance(after, str):
        raise DocumentError("word_diff needs two strings")
    a_words = re.findall(r"\S+|\s+", before)
    b_words = re.findall(r"\S+|\s+", after)
    matcher = difflib.SequenceMatcher(a=a_words, b=b_words, autojunk=False)
    ops: list[tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            ops.append(("equal", "".join(a_words[i1:i2])))
        elif tag == "delete":
            ops.append(("delete", "".join(a_words[i1:i2])))
        elif tag == "insert":
            ops.append(("insert", "".join(b_words[j1:j2])))
        else:  # replace
            ops.append(("delete", "".join(a_words[i1:i2])))
            ops.append(("insert", "".join(b_words[j1:j2])))
    # Semantic cleanup: an "equal" run that is short, contains no newline,
    # and sits between a change on both sides is not a real commonality —
    # fold it into the surrounding change (delete+insert), diff-match-patch
    # cleanupSemantic style.
    cleaned: list[tuple[str, str]] = []
    n = len(ops)
    for idx, (op, text) in enumerate(ops):
        prev_is_change = bool(cleaned) and cleaned[-1][0] in ("delete", "insert")
        next_is_change = idx + 1 < n and ops[idx + 1][0] in ("delete", "insert")
        if (op == "equal" and prev_is_change and next_is_change
                and len(text) <= 12 and "\n" not in text):
            prev_op, prev_text = cleaned.pop()
            merged_op = "delete" if prev_op == "delete" else "insert"
            cleaned.append((merged_op, prev_text + text))
        else:
            cleaned.append((op, text))
    # Merge adjacent same-op runs.
    merged: list[tuple[str, str]] = []
    for op, text in cleaned:
        if merged and merged[-1][0] == op:
            merged[-1] = (op, merged[-1][1] + text)
        else:
            merged.append((op, text))
    return merged


def similarity(first: Document, second: Document) -> float:
    """0..1 textual similarity of two documents (SequenceMatcher ratio)."""
    if not isinstance(first, Document) or not isinstance(second, Document):
        raise DocumentError("similarity needs two Document objects")
    return round(difflib.SequenceMatcher(
        None, full_text(first), full_text(second),
        autojunk=False).ratio(), 4)


def _table_cell_changes(before, after) -> list[dict[str, Any]]:
    """Cell-level changes between two table grids (pandas
    ``DataFrame.compare`` spirit): rows are aligned with SequenceMatcher,
    then cells are compared pairwise.  Row inserts/deletes are reported
    as row-level changes."""
    b_sigs = ["\x1f".join(row) for row in before.rows]
    a_sigs = ["\x1f".join(row) for row in after.rows]
    headers = before.headers or after.headers
    changes: list[dict[str, Any]] = []

    def header_for(col: int) -> str:
        if col < len(headers):
            return headers[col]
        return f"col_{col + 1}"

    matcher = difflib.SequenceMatcher(a=b_sigs, b=a_sigs, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for bi, ai in zip(range(i1, i2), range(j1, j2)):
                brow, arow = before.rows[bi], after.rows[ai]
                width = max(len(brow), len(arow))
                for col in range(width):
                    old = brow[col] if col < len(brow) else ""
                    new = arow[col] if col < len(arow) else ""
                    if old != new:
                        changes.append({
                            "type": "cell",
                            "row": bi + 1,
                            "column": header_for(col),
                            "before": old,
                            "after": new,
                        })
        elif tag == "delete":
            for bi in range(i1, i2):
                changes.append({"type": "row_removed", "row": bi + 1,
                                "values": list(before.rows[bi])})
        elif tag == "insert":
            for ai in range(j1, j2):
                changes.append({"type": "row_added", "row": ai + 1,
                                "values": list(after.rows[ai])})
        else:  # replace: pairwise cell compare on the overlapping run
            for bi, ai in zip(range(i1, i2), range(j1, j2)):
                brow, arow = before.rows[bi], after.rows[ai]
                width = max(len(brow), len(arow))
                for col in range(width):
                    old = brow[col] if col < len(brow) else ""
                    new = arow[col] if col < len(arow) else ""
                    if old != new:
                        changes.append({
                            "type": "cell",
                            "row": bi + 1,
                            "column": header_for(col),
                            "before": old,
                            "after": new,
                        })
            for bi in range(i1 + (j2 - j1), i2):
                changes.append({"type": "row_removed", "row": bi + 1,
                                "values": list(before.rows[bi])})
            for ai in range(j1 + (i2 - i1), j2):
                changes.append({"type": "row_added", "row": ai + 1,
                                "values": list(after.rows[ai])})
    return changes


def compare_documents(first: Document, second: Document, *,
                      context: int = 3) -> DocumentComparison:
    """Compare two documents structurally and textually.

    Sections are matched by heading (untitled sections by position);
    tables by name.  Sections that vanished under one heading and
    reappeared under another with near-identical bodies are reported as
    *moved*, not added+removed.  Never raises on mismatched shapes —
    differences are reported, not errors.
    """
    if not isinstance(first, Document) or not isinstance(second, Document):
        raise DocumentError("compare_documents needs two Document objects")

    a_sections = {_section_key(i, s.heading): s
                  for i, s in enumerate(first.sections)}
    b_sections = {_section_key(i, s.heading): s
                  for i, s in enumerate(second.sections)}
    sections_added = sorted(set(b_sections) - set(a_sections))
    sections_removed = sorted(set(a_sections) - set(b_sections))
    # Moved detection: a removed section whose body near-matches an added
    # section's body was renamed/moved, not deleted+created.
    sections_moved: list[dict[str, Any]] = []
    for removed_key in list(sections_removed):
        best_key, best_ratio = "", 0.0
        removed_text = a_sections[removed_key].text
        if not removed_text.strip():
            continue
        for added_key in sections_added:
            added_text = b_sections[added_key].text
            if not added_text.strip():
                continue
            ratio = difflib.SequenceMatcher(
                None, removed_text, added_text, autojunk=False).ratio()
            if ratio > best_ratio:
                best_key, best_ratio = added_key, ratio
        if best_key and best_ratio >= 0.85:
            sections_moved.append({
                "from": removed_key,
                "to": best_key,
                "similarity": round(best_ratio, 3),
            })
            sections_removed.remove(removed_key)
            sections_added.remove(best_key)
    sections_moved.sort(key=lambda m: m["from"])
    sections_changed = sorted(
        key for key in set(a_sections) & set(b_sections)
        if a_sections[key].text.strip() != b_sections[key].text.strip()
        or a_sections[key].level != b_sections[key].level)

    a_tables = {t.name or f"<table {i + 1}>" : t
                for i, t in enumerate(first.tables)}
    b_tables = {t.name or f"<table {i + 1}>" : t
                for i, t in enumerate(second.tables)}
    tables_added = sorted(set(b_tables) - set(a_tables))
    tables_removed = sorted(set(a_tables) - set(b_tables))
    tables_changed = sorted(
        key for key in set(a_tables) & set(b_tables)
        if _table_signature(a_tables[key]) != _table_signature(b_tables[key]))

    table_details = {
        key: _describe_table_change(_table_signature(a_tables[key]),
                                    _table_signature(b_tables[key]))
        for key in tables_changed
    }
    # Cell-level detail for changed tables (DataFrame.compare spirit).
    table_cell_changes = {
        key: _table_cell_changes(a_tables[key], b_tables[key])
        for key in tables_changed
    }
    # Inline word diffs for changed sections.
    section_word_diffs = {
        key: [[op, text] for op, text in
              word_diff(a_sections[key].text, b_sections[key].text)]
        for key in sections_changed
    }

    unified = diff_documents(first, second, context=context)
    text_changed = bool(unified)
    sim = similarity(first, second)

    bits: list[str] = []
    if sections_added:
        bits.append(f"{len(sections_added)} section(s) added")
    if sections_removed:
        bits.append(f"{len(sections_removed)} section(s) removed")
    if sections_moved:
        bits.append(f"{len(sections_moved)} section(s) moved/renamed")
    if sections_changed:
        bits.append(f"{len(sections_changed)} section(s) changed")
    if tables_added:
        bits.append(f"{len(tables_added)} table(s) added")
    if tables_removed:
        bits.append(f"{len(tables_removed)} table(s) removed")
    if tables_changed:
        bits.append(f"{len(tables_changed)} table(s) changed")
    summary = "; ".join(bits) if bits else "no changes"
    if bits:
        summary += f" ({sim:.0%} similar)"

    return DocumentComparison(
        summary=summary,
        text_changed=text_changed,
        similarity=sim,
        sections_added=sections_added,
        sections_removed=sections_removed,
        sections_changed=sections_changed,
        sections_moved=sections_moved,
        tables_added=tables_added,
        tables_removed=tables_removed,
        tables_changed=tables_changed,
        unified_diff=unified,
        stats={
            "sections_added": len(sections_added),
            "sections_removed": len(sections_removed),
            "sections_moved": len(sections_moved),
            "sections_changed": len(sections_changed),
            "tables_added": len(tables_added),
            "tables_removed": len(tables_removed),
            "tables_changed": len(tables_changed),
            "diff_lines": len(unified.splitlines()) if unified else 0,
            "table_details": table_details,
            "table_cell_changes": table_cell_changes,
            "section_word_diffs": section_word_diffs,
        },
    )


# ── renderers ────────────────────────────────────────────────────────────────

_DIFF_CSS = """
body{font-family:-apple-system,'Segoe UI',Roboto,sans-serif;max-width:1100px;
margin:2rem auto;padding:0 1.5rem;color:#1f2937;background:#fff;line-height:1.6}
h1{font-size:1.5rem;border-bottom:2px solid #e5e7eb;padding-bottom:.5rem}
h2{font-size:1.1rem;margin-top:2rem;color:#374151}
.summary{background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;
padding:1rem 1.25rem;font-weight:600}
ul.changes{list-style:none;padding:0}
ul.changes li{padding:.35rem .75rem;margin:.25rem 0;border-radius:6px;font-size:.95rem}
li.add{background:#ecfdf5;border-left:4px solid #10b981}
li.del{background:#fef2f2;border-left:4px solid #ef4444}
li.chg{background:#fffbeb;border-left:4px solid #f59e0b}
li.mov{background:#eff6ff;border-left:4px solid #3b82f6}
table.cells{border-collapse:collapse;width:100%;font-size:.9rem;margin:.5rem 0}
table.cells th,table.cells td{border:1px solid #e5e7eb;padding:.4rem .6rem;text-align:left}
table.cells th{background:#f9fafb}
td.before{background:#fee2e2}td.after{background:#d1fae5}
pre.diff{background:#0f172a;color:#e2e8f0;border-radius:8px;padding:1rem;
overflow-x:auto;font-size:.85rem;line-height:1.5}
pre.diff .add{color:#4ade80}pre.diff .del{color:#f87171}
pre.diff .hunk{color:#67e8f9}pre.diff .meta{color:#94a3b8}
.word del{background:#fecaca;text-decoration:none;border-radius:3px;padding:0 2px}
.word ins{background:#bbf7d0;text-decoration:none;border-radius:3px;padding:0 2px}
"""

_ANSI = {
    "reset": "\033[0m", "bold": "\033[1m",
    "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
    "blue": "\033[34m", "cyan": "\033[36m", "gray": "\033[90m",
}


def _diff_line_class(line: str) -> str:
    if line.startswith(("+++", "---")):
        return "meta"
    if line.startswith("+"):
        return "add"
    if line.startswith("-"):
        return "del"
    if line.startswith("@@"):
        return "hunk"
    return ""


def render_html(comparison: DocumentComparison, *,
                label_a: str = "before", label_b: str = "after") -> str:
    """Render a comparison as a styled standalone HTML report.

    Summary banner, color-coded change lists, cell-level table changes
    with before/after highlighting, inline word diffs for changed
    sections, and the unified diff in a dark code block (delta-inspired).
    """
    esc = html_module.escape
    parts = ["<!DOCTYPE html>", "<html>", "<head>",
             "<meta charset=\"utf-8\">",
             f"<title>Document comparison — {esc(label_a)} vs "
             f"{esc(label_b)}</title>",
             f"<style>{_DIFF_CSS}</style>",
             "</head>", "<body>"]
    parts.append(f"<h1>Document comparison</h1>")
    parts.append(f"<p class=\"meta\">{esc(label_a)} → {esc(label_b)}</p>")
    parts.append(f"<div class=\"summary\">{esc(comparison.summary)}</div>")

    def change_list(title: str, items: list[str], css: str,
                    marker: str) -> None:
        if not items:
            return
        parts.append(f"<h2>{esc(title)}</h2><ul class=\"changes\">")
        for item in items:
            parts.append(f"<li class=\"{css}\">{marker} {esc(item)}</li>")
        parts.append("</ul>")

    change_list("Sections added", comparison.sections_added, "add", "+")
    change_list("Sections removed", comparison.sections_removed, "del", "−")
    change_list("Sections changed", comparison.sections_changed, "chg", "~")
    if comparison.sections_moved:
        parts.append("<h2>Sections moved / renamed</h2><ul class=\"changes\">")
        for move in comparison.sections_moved:
            parts.append(
                f"<li class=\"mov\">⇄ {esc(str(move['from']))} → "
                f"{esc(str(move['to']))} "
                f"({float(move.get('similarity', 0)):.0%} similar)</li>")
        parts.append("</ul>")
    change_list("Tables added", comparison.tables_added, "add", "+")
    change_list("Tables removed", comparison.tables_removed, "del", "−")
    change_list("Tables changed", comparison.tables_changed, "chg", "~")

    cell_changes = comparison.stats.get("table_cell_changes", {})
    for table_name, changes in cell_changes.items():
        if not changes:
            continue
        parts.append(f"<h2>Table: {esc(table_name)}</h2>")
        parts.append("<table class=\"cells\"><tr><th>Row</th><th>Column</th>"
                     "<th>Before</th><th>After</th><th>Change</th></tr>")
        for change in changes:
            ctype = change.get("type", "cell")
            if ctype == "cell":
                parts.append(
                    f"<tr><td>{change.get('row', '')}</td>"
                    f"<td>{esc(str(change.get('column', '')))}</td>"
                    f"<td class=\"before\">{esc(str(change.get('before', '')))}</td>"
                    f"<td class=\"after\">{esc(str(change.get('after', '')))}</td>"
                    f"<td>modified</td></tr>")
            elif ctype == "row_added":
                vals = " | ".join(str(v) for v in change.get("values", []))
                parts.append(
                    f"<tr><td>{change.get('row', '')}</td><td>—</td>"
                    f"<td></td><td class=\"after\">"
                    f"{esc(vals)}</td><td>row added</td></tr>")
            else:
                vals = " | ".join(str(v) for v in change.get("values", []))
                parts.append(
                    f"<tr><td>{change.get('row', '')}</td><td>—</td>"
                    f"<td class=\"before\">{esc(vals)}</td><td></td>"
                    f"<td>row removed</td></tr>")
        parts.append("</table>")

    word_diffs = comparison.stats.get("section_word_diffs", {})
    for section_name, ops in word_diffs.items():
        parts.append(f"<h2>Section: {esc(section_name)}</h2>"
                     "<p class=\"word\">")
        for op, text in ops:
            if op == "delete":
                parts.append(f"<del>{esc(text)}</del>")
            elif op == "insert":
                parts.append(f"<ins>{esc(text)}</ins>")
            else:
                parts.append(esc(text))
        parts.append("</p>")

    if comparison.unified_diff:
        parts.append("<h2>Unified diff</h2><pre class=\"diff\">")
        for line in comparison.unified_diff.splitlines():
            cls = _diff_line_class(line)
            parts.append(f"<span class=\"{cls}\">{esc(line)}</span>"
                         if cls else esc(line))
        parts.append("</pre>")
    parts.extend(["</body>", "</html>"])
    return "\n".join(parts) + "\n"


def render_terminal(comparison: DocumentComparison, *,
                    color: bool = True) -> str:
    """Render a comparison for the terminal (delta-style ANSI colors).

    Pass ``color=False`` for plain output (logs, pipes).
    """
    def paint(text: str, name: str) -> str:
        return f"{_ANSI[name]}{text}{_ANSI['reset']}" if color else text

    lines = [paint("Document comparison", "bold"),
             paint(comparison.summary, "cyan"), ""]
    groups = [
        ("Sections added", comparison.sections_added, "green", "+"),
        ("Sections removed", comparison.sections_removed, "red", "-"),
        ("Sections changed", comparison.sections_changed, "yellow", "~"),
        ("Tables added", comparison.tables_added, "green", "+"),
        ("Tables removed", comparison.tables_removed, "red", "-"),
        ("Tables changed", comparison.tables_changed, "yellow", "~"),
    ]
    for title, items, ansi, marker in groups:
        if not items:
            continue
        lines.append(paint(f"{title}:", "bold"))
        for item in items:
            lines.append(paint(f"  {marker} {item}", ansi))
    if comparison.sections_moved:
        lines.append(paint("Sections moved / renamed:", "bold"))
        for move in comparison.sections_moved:
            lines.append(paint(
                f"  ⇄ {move['from']} → {move['to']}", "blue"))
    cell_changes = comparison.stats.get("table_cell_changes", {})
    for table_name, changes in cell_changes.items():
        if not changes:
            continue
        lines.append(paint(f"Table {table_name}:", "bold"))
        for change in changes:
            ctype = change.get("type", "cell")
            if ctype == "cell":
                lines.append(
                    f"  row {change['row']} {change['column']}: "
                    f"{paint(str(change['before']), 'red')} → "
                    f"{paint(str(change['after']), 'green')}")
            elif ctype == "row_added":
                lines.append(paint(
                    f"  + row {change['row']}: "
                    + " | ".join(str(v) for v in change["values"]), "green"))
            else:
                lines.append(paint(
                    f"  - row {change['row']}: "
                    + " | ".join(str(v) for v in change["values"]), "red"))
    if comparison.unified_diff:
        lines.append("")
        for line in comparison.unified_diff.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                lines.append(paint(line, "green"))
            elif line.startswith("-") and not line.startswith("---"):
                lines.append(paint(line, "red"))
            elif line.startswith("@@"):
                lines.append(paint(line, "cyan"))
            else:
                lines.append(paint(line, "gray") if color else line)
    return "\n".join(lines).rstrip() + "\n"


def render_markdown(comparison: DocumentComparison, *,
                    label_a: str = "before",
                    label_b: str = "after") -> str:
    """Render a comparison as a markdown report (chat/paste friendly)."""
    lines = [f"# Document comparison: {label_a} → {label_b}", "",
             f"**{comparison.summary}**", ""]

    def section(title: str, items: list[str], marker: str) -> None:
        if not items:
            return
        lines.append(f"## {title}")
        lines.extend(f"- {marker} `{item}`" for item in items)
        lines.append("")

    section("Sections added", comparison.sections_added, "+")
    section("Sections removed", comparison.sections_removed, "-")
    section("Sections changed", comparison.sections_changed, "~")
    if comparison.sections_moved:
        lines.append("## Sections moved / renamed")
        for move in comparison.sections_moved:
            lines.append(f"- ⇄ `{move['from']}` → `{move['to']}`")
        lines.append("")
    section("Tables added", comparison.tables_added, "+")
    section("Tables removed", comparison.tables_removed, "-")
    section("Tables changed", comparison.tables_changed, "~")

    cell_changes = comparison.stats.get("table_cell_changes", {})
    for table_name, changes in cell_changes.items():
        if not changes:
            continue
        lines.append(f"## Table `{table_name}`")
        lines.append("| Row | Column | Before | After |")
        lines.append("| --- | --- | --- | --- |")
        for change in changes:
            ctype = change.get("type", "cell")
            if ctype == "cell":
                lines.append(f"| {change['row']} | {change['column']} | "
                             f"{change['before']} | {change['after']} |")
            elif ctype == "row_added":
                vals = " \\| ".join(str(v) for v in change["values"])
                lines.append(f"| {change['row']} | — |  | {vals} | added |")
            else:
                vals = " \\| ".join(str(v) for v in change["values"])
                lines.append(f"| {change['row']} | — | {vals} |  | removed |")
        lines.append("")
    if comparison.unified_diff:
        lines.append("## Unified diff")
        lines.append("```diff")
        lines.append(comparison.unified_diff)
        lines.append("```")
    return "\n".join(lines).rstrip() + "\n"
