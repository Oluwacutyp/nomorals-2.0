"""Document model: the universal container the document engine parses into.

Every parser in :mod:`nomorals.documents.parsers` produces a
:class:`Document`; every converter in :mod:`nomorals.documents.convert`
consumes one.  The model is deliberately plain — dataclasses with
``to_dict``/``from_dict`` round-trips so documents can cross process
boundaries as JSON.

Design notes (mined from docling's ``DoclingDocument``, Unstructured
elements, and LangChain ``Document``):

* :class:`Section` carries ``page`` provenance and a ``kind`` (text, list,
  code, quote, caption, figure) so renderers, chunkers, and citations can
  treat blocks differently instead of flattening everything to prose.
* :class:`Table` carries ``caption``, ``page``, and a ``confidence``
  score (0..1, Camelot-style) plus record/column accessors.
* :class:`Document` offers structure-aware :meth:`chunks` (the
  HierarchicalChunker idea: section boundaries respected, heading context
  inherited), :meth:`content_hash` for dedup, and an :meth:`outline` TOC.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

from ..core.ids import ulid_now

__all__ = ["Document", "Section", "Table", "full_text", "new_document",
           "SECTION_KINDS"]

#: Known section kinds.  Parsers set these; anything else stays "text".
SECTION_KINDS = frozenset({
    "text", "list", "code", "quote", "caption", "figure", "footnote",
})


@dataclass
class Section:
    """One headed chunk of prose. ``level`` is 1..6 (markdown heading depth).

    ``page`` is 1-based source-page provenance (0 = unknown) and ``kind``
    is one of ``text``/``list``/``code``/``quote``/``caption``/``figure``/
    ``footnote`` — renderers and chunkers use it to treat blocks
    differently instead of flattening everything to prose.
    """

    level: int = 1
    heading: str = ""
    text: str = ""
    page: int = 0
    kind: str = "text"

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level, "heading": self.heading,
                "text": self.text, "page": self.page, "kind": self.kind}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Section:
        try:
            kind = str(data.get("kind", "text") or "text")
            return cls(
                level=int(data.get("level", 1)),
                heading=str(data.get("heading", "")),
                text=str(data.get("text", "")),
                page=int(data.get("page", 0) or 0),
                kind=kind if kind in SECTION_KINDS else "text",
            )
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"bad Section payload: {exc}") from exc

    def word_count(self) -> int:
        """Words in heading + text."""
        return len((self.heading + " " + self.text).split())


@dataclass
class Table:
    """A rectangular data table: header row plus data rows, all strings.

    ``caption`` is a human label recovered near the table ("Table 3: …"),
    ``page`` is 1-based source-page provenance (0 = unknown), and
    ``confidence`` is a 0..1 extraction-confidence score (Camelot-style:
    higher means cleaner column alignment).
    """

    name: str = ""
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    caption: str = ""
    page: int = 0
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "headers": list(self.headers),
                "rows": [list(r) for r in self.rows],
                "caption": self.caption, "page": self.page,
                "confidence": self.confidence}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Table:
        try:
            return cls(
                name=str(data.get("name", "")),
                headers=[str(h) for h in data.get("headers", [])],
                rows=[[str(c) for c in row] for row in data.get("rows", [])],
                caption=str(data.get("caption", "") or ""),
                page=int(data.get("page", 0) or 0),
                confidence=float(data.get("confidence", 0.0) or 0.0),
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

    @property
    def n_rows(self) -> int:
        """Number of data rows (excluding the header)."""
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        """Number of columns (header width, or widest row)."""
        if self.headers:
            return len(self.headers)
        return max((len(r) for r in self.rows), default=0)

    def to_records(self) -> list[dict[str, str]]:
        """Rows as dicts keyed by header (pandas ``to_dict('records')``)."""
        headers = self.headers or [f"col_{i + 1}" for i in range(self.n_cols)]
        records = []
        for row in self.rows:
            padded = list(row) + [""] * (len(headers) - len(row))
            records.append({h: padded[i] for i, h in enumerate(headers)})
        return records

    def column(self, name_or_index: str | int) -> list[str]:
        """One column of data cells, by header name or 0-based index."""
        if isinstance(name_or_index, str):
            if name_or_index not in self.headers:
                raise KeyError(f"no column {name_or_index!r} in table "
                               f"{self.name!r}")
            index = self.headers.index(name_or_index)
        else:
            index = int(name_or_index)
            if not 0 <= index < self.n_cols:
                raise IndexError(f"column index {index} out of range "
                                 f"(table has {self.n_cols} columns)")
        return [row[index] if index < len(row) else "" for row in self.rows]

    def stats(self) -> dict[str, dict[str, Any]]:
        """Per-column fill/numeric/uniqueness stats (profiling hook)."""
        headers = self.headers or [f"col_{i + 1}" for i in range(self.n_cols)]
        out: dict[str, dict[str, Any]] = {}
        for i, header in enumerate(headers):
            cells = [row[i] if i < len(row) else "" for row in self.rows]
            filled = [c for c in cells if c.strip()]
            numeric = 0
            for cell in filled:
                try:
                    float(cell.replace(",", "").replace("%", ""))
                    numeric += 1
                except ValueError:
                    pass
            out[header] = {
                "cells": len(cells),
                "filled": len(filled),
                "fill_ratio": round(len(filled) / len(cells), 3) if cells else 0.0,
                "numeric_ratio": round(numeric / len(filled), 3) if filled else 0.0,
                "unique": len(set(filled)),
            }
        return out


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

    # ── document intelligence ────────────────────────────────────────────
    def outline(self) -> list[dict[str, Any]]:
        """Table of contents: [{level, heading, page}] for headed sections."""
        return [
            {"level": s.level, "heading": s.heading, "page": s.page}
            for s in self.sections if s.heading
        ]

    def word_count(self) -> int:
        """Total words across title, sections, and tables."""
        return len(full_text(self).split())

    def reading_time_minutes(self, wpm: int = 200) -> float:
        """Estimated reading time in minutes at ``wpm`` words per minute."""
        if wpm <= 0:
            raise ValueError(f"wpm must be positive, got {wpm}")
        return round(self.word_count() / wpm, 1)

    def find_sections(self, query: str) -> list[Section]:
        """Sections whose heading or text contains ``query``
        (case-insensitive)."""
        needle = (query or "").lower()
        if not needle:
            return []
        return [s for s in self.sections
                if needle in s.heading.lower() or needle in s.text.lower()]

    def iter_blocks(self) -> Iterator[dict[str, Any]]:
        """Reading-order walk over sections then tables (docling
        ``iterate_items`` spirit, stdlib edition)."""
        for i, section in enumerate(self.sections):
            yield {"kind": "section", "index": i, "section": section}
        for i, table in enumerate(self.tables):
            yield {"kind": "table", "index": i, "table": table}

    def content_hash(self) -> str:
        """SHA-256 over normalized full text — stable identity for
        dedup and change detection (DeepHash idea, stdlib edition)."""
        normalized = " ".join(full_text(self).split()).lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def chunks(self, max_words: int = 300) -> list[dict[str, Any]]:
        """Structure-aware chunks for retrieval (docling HierarchicalChunker
        idea): section boundaries are never crossed; each chunk inherits
        its heading path as context; oversized sections split on
        paragraph breaks; undersized siblings merge (``merge_peers``)."""
        if max_words < 20:
            raise ValueError(f"max_words must be >= 20, got {max_words}")
        chunks: list[dict[str, Any]] = []
        heading_path: list[str] = []

        def push(text: str, page: int) -> None:
            text = text.strip()
            if not text:
                return
            context = " > ".join(h for h in heading_path if h)
            if chunks and chunks[-1]["words"] + len(text.split()) <= max_words \
                    and chunks[-1]["page"] == page:
                # merge_peers: fold an undersized sibling into the previous
                # chunk when it still fits.
                chunks[-1]["text"] += "\n\n" + text
                chunks[-1]["words"] = len(chunks[-1]["text"].split())
            else:
                chunks.append({"text": text, "heading_path": context,
                               "page": page,
                               "words": len(text.split())})

        for section in self.sections:
            # Maintain the heading stack: pop while the new heading is not
            # deeper than the stack top.
            while heading_path and section.level <= len(heading_path):
                # The stack depth mirrors heading levels only loosely
                # (documents skip levels); pop one per section and rely on
                # level-1 sections to reset.
                if section.level == 1:
                    heading_path.clear()
                    break
                heading_path.pop()
                break
            if section.heading:
                heading_path.append(section.heading)
                # Keep the path shallow and readable.
                heading_path = heading_path[-4:]
            body = section.text.strip()
            if not body:
                continue
            words = body.split()
            if len(words) <= max_words:
                push(body, section.page)
                continue
            # Oversized section: split on paragraph breaks, then sentences.
            paras: list[str] = []
            for para in body.split("\n"):
                para = para.strip()
                if not para:
                    continue
                if len(para.split()) <= max_words:
                    paras.append(para)
                else:
                    # Hard split long paragraphs on sentence boundaries.
                    buf, count = [], 0
                    for sent in para.split(". "):
                        buf.append(sent)
                        count += len(sent.split())
                        if count >= max_words:
                            paras.append(". ".join(buf).strip())
                            buf, count = [], 0
                    if buf:
                        paras.append(". ".join(buf).strip())
            # Anything still oversized (no sentence boundaries at all)
            # splits on raw word count — chunks must honor max_words.
            final: list[str] = []
            for para in paras:
                words_p = para.split()
                if len(words_p) <= max_words:
                    final.append(para)
                else:
                    for i in range(0, len(words_p), max_words):
                        final.append(" ".join(words_p[i:i + max_words]))
            for para in final:
                push(para, section.page)
        # Tables become their own chunks (never merged into prose).
        for table in self.tables:
            text = table.as_text()
            if text.strip():
                chunks.append({
                    "text": (f"Table: {table.name or table.caption}\n"
                             + text).strip(),
                    "heading_path": "",
                    "page": table.page,
                    "words": len(text.split()),
                })
        return chunks

    def merge(self, other: Document) -> Document:
        """New Document concatenating ``other`` onto this one (sections and
        tables appended; metadata merged, ``other`` winning on conflicts)."""
        if not isinstance(other, Document):
            raise ValueError("merge needs a Document object")
        merged_meta = dict(self.metadata)
        merged_meta.update(other.metadata)
        return Document(
            format=self.format or other.format,
            title=self.title or other.title,
            author=self.author or other.author,
            source=self.source or other.source,
            sections=list(self.sections) + list(other.sections),
            tables=list(self.tables) + list(other.tables),
            metadata=merged_meta,
        )

    def to_json(self, indent: int = 2) -> str:
        """The document as a JSON string (round-trips via
        :meth:`from_dict`)."""
        return json.dumps(self.to_dict(), indent=indent,
                          ensure_ascii=False, default=str)


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
