"""Document comparison: diff two parsed documents (L4).

:func:`compare_documents` produces a structured
:class:`DocumentComparison` — section-level added/removed/changed sets,
table-level changes, and a unified text diff.  :func:`diff_documents` is
the plain unified-diff shortcut.  Everything is stdlib ``difflib``; no
optional dependencies.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any

from .errors import DocumentError
from .model import Document, full_text

__all__ = ["DocumentComparison", "compare_documents", "diff_documents"]


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
    unified_diff: str = ""
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "text_changed": self.text_changed,
            "sections_added": list(self.sections_added),
            "sections_removed": list(self.sections_removed),
            "sections_changed": list(self.sections_changed),
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


def compare_documents(first: Document, second: Document, *,
                      context: int = 3) -> DocumentComparison:
    """Compare two documents structurally and textually.

    Sections are matched by heading (untitled sections by position);
    tables by name.  Never raises on mismatched shapes — differences are
    reported, not errors.
    """
    if not isinstance(first, Document) or not isinstance(second, Document):
        raise DocumentError("compare_documents needs two Document objects")

    a_sections = {_section_key(i, s.heading): s
                  for i, s in enumerate(first.sections)}
    b_sections = {_section_key(i, s.heading): s
                  for i, s in enumerate(second.sections)}
    sections_added = sorted(set(b_sections) - set(a_sections))
    sections_removed = sorted(set(a_sections) - set(b_sections))
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

    unified = diff_documents(first, second, context=context)
    text_changed = bool(unified)

    bits: list[str] = []
    if sections_added:
        bits.append(f"{len(sections_added)} section(s) added")
    if sections_removed:
        bits.append(f"{len(sections_removed)} section(s) removed")
    if sections_changed:
        bits.append(f"{len(sections_changed)} section(s) changed")
    if tables_added:
        bits.append(f"{len(tables_added)} table(s) added")
    if tables_removed:
        bits.append(f"{len(tables_removed)} table(s) removed")
    if tables_changed:
        bits.append(f"{len(tables_changed)} table(s) changed")
    summary = "; ".join(bits) if bits else "no changes"

    return DocumentComparison(
        summary=summary,
        text_changed=text_changed,
        sections_added=sections_added,
        sections_removed=sections_removed,
        sections_changed=sections_changed,
        tables_added=tables_added,
        tables_removed=tables_removed,
        tables_changed=tables_changed,
        unified_diff=unified,
        stats={
            "sections_added": len(sections_added),
            "sections_removed": len(sections_removed),
            "sections_changed": len(sections_changed),
            "tables_added": len(tables_added),
            "tables_removed": len(tables_removed),
            "tables_changed": len(tables_changed),
            "diff_lines": len(unified.splitlines()) if unified else 0,
            "table_details": table_details,
        },
    )
