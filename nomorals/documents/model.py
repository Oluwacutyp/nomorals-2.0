"""Document model: the universal container the document engine parses into.

Every parser in :mod:`nomorals.documents.parsers` produces a
:class:`Document`; every converter in :mod:`nomorals.documents.convert`
consumes one.  The model is deliberately plain — dataclasses with
``to_dict``/``from_dict`` round-trips so documents can cross process
boundaries as JSON.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from ..core.ids import ulid_now

__all__ = ["Document", "Section", "Table", "full_text", "new_document"]


@dataclass
class Section:
    """One headed chunk of prose. ``level`` is 1..6 (markdown heading depth)."""

    level: int = 1
    heading: str = ""
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level, "heading": self.heading, "text": self.text}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Section:
        try:
            return cls(
                level=int(data.get("level", 1)),
                heading=str(data.get("heading", "")),
                text=str(data.get("text", "")),
            )
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"bad Section payload: {exc}") from exc


@dataclass
class Table:
    """A rectangular data table: header row plus data rows, all strings."""

    name: str = ""
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "headers": list(self.headers),
                "rows": [list(r) for r in self.rows]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Table:
        try:
            return cls(
                name=str(data.get("name", "")),
                headers=[str(h) for h in data.get("headers", [])],
                rows=[[str(c) for c in row] for row in data.get("rows", [])],
            )
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"bad Table payload: {exc}") from exc

    def as_text(self) -> str:
        """The table rendered as plain pipe-separated lines."""
        lines = []
        if self.headers:
            lines.append(" | ".join(self.headers))
        lines.extend(" | ".join(row) for row in self.rows)
        return "\n".join(lines)


@dataclass
class Document:
    """One parsed document, regardless of source format."""

    id: str = ""
    format: str = ""
    title: str = ""
    author: str = ""
    source: str = ""  # file path or URL the document came from
    created_at: float = 0.0
    sections: list[Section] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = ulid_now()
        if not self.created_at:
            self.created_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # asdict already recurses into nested dataclasses; keep it explicit.
        data["sections"] = [s.to_dict() for s in self.sections]
        data["tables"] = [t.to_dict() for t in self.tables]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        try:
            return cls(
                id=str(data.get("id", "") or ulid_now()),
                format=str(data.get("format", "")),
                title=str(data.get("title", "")),
                author=str(data.get("author", "")),
                source=str(data.get("source", "")),
                created_at=float(data.get("created_at", 0.0) or 0.0),
                sections=[Section.from_dict(s) for s in data.get("sections", [])],
                tables=[Table.from_dict(t) for t in data.get("tables", [])],
                metadata=dict(data.get("metadata", {})),
            )
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"bad Document payload: {exc}") from exc


def new_document(*, format: str, title: str = "", author: str = "",
                 source: str = "") -> Document:
    """Build a fresh Document with a ULID id and current timestamp."""
    return Document(id=ulid_now(), format=format, title=title, author=author,
                    source=source, created_at=time.time())


def full_text(doc: Document) -> str:
    """All searchable text of a document: title, headings, prose, tables."""
    parts: list[str] = []
    if doc.title:
        parts.append(doc.title)
    for section in doc.sections:
        if section.heading:
            parts.append(section.heading)
        if section.text:
            parts.append(section.text)
    for table in doc.tables:
        if table.name:
            parts.append(table.name)
        text = table.as_text()
        if text:
            parts.append(text)
    return "\n\n".join(parts)
