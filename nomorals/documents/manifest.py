"""Manifest derivation for documents-engine Documents (Doc 16 P1).

Builds the canonical :class:`~nomorals.core.longdoc.LongDocumentManifest`
for a :class:`~nomorals.documents.model.Document`: one segment per
section, then one per table (the model's own reading order, mirroring
``Document.iter_blocks``).  A segment's canonical content is the
canonical JSON of its ``to_dict()`` form — byte-exact, so a segment
hash changes iff the section/table actually changed.

The manifest's ``revision`` is the document's own revision counter
(the P1 revision contract on the model); its provenance ref is
:func:`document_revision_ref`.  A parsed document is complete by
construction — every section and table it has is present — so
completeness is ``complete`` with the segment count as the declared
total.
"""

from __future__ import annotations

import json
from typing import Any

from ..core.longdoc import (
    Completeness,
    LongDocumentManifest,
    RevisionRef,
    SegmentSpec,
    build_manifest,
)
from .model import Document, Section, Table

__all__ = ["manifest_from_document", "document_revision_ref",
           "section_content", "table_content",
           "document_content_resolver"]


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def section_content(section: Section) -> str:
    """The canonical content of one section (what its segment hash
    covers and what :meth:`LongDocumentManifest.verify` recomputes)."""
    return _canonical(section.to_dict())


def table_content(table: Table) -> str:
    """The canonical content of one table (see :func:`section_content`)."""
    return _canonical(table.to_dict())


def document_revision_ref(doc: Document) -> RevisionRef:
    """The provenance ref for this document at its current revision."""
    ref = doc.edition_ref()
    return RevisionRef(source_kind="document",
                       document_id=ref["document_id"],
                       revision=ref["revision"],
                       content_hash=ref["content_hash"])


def manifest_from_document(doc: Document, *,
                           created_by: str = "documents.manifest",
                           ) -> LongDocumentManifest:
    """Derive the manifest of a documents-engine Document."""
    specs: list[SegmentSpec] = []
    for i, section in enumerate(doc.sections):
        specs.append(SegmentSpec(
            segment_id=f"sec-{i + 1}", kind="section",
            title=section.heading,
            content=section_content(section),
            content_ref=f"document:{doc.id}#sec-{i + 1}",
            metadata={"level": section.level, "page": section.page,
                      "kind": section.kind},
        ))
    for i, table in enumerate(doc.tables):
        specs.append(SegmentSpec(
            segment_id=f"tbl-{i + 1}", kind="table",
            title=table.name or table.caption,
            content=table_content(table),
            content_ref=f"document:{doc.id}#tbl-{i + 1}",
            metadata={"page": table.page, "rows": table.n_rows,
                      "cols": table.n_cols},
        ))
    return build_manifest(
        document_id=doc.id, source_kind="document",
        source_ref=doc.source, title=doc.title, author=doc.author,
        revision=doc.revision, segment_specs=specs,
        completeness=Completeness(
            status="complete", declared_total=len(specs),
            present_count=len(specs),
            note="parsed document: every section/table present by "
                 "construction"),
        created_by=created_by,
        metadata={"format": doc.format, "source": doc.source},
    )


def document_content_resolver(doc: Document):
    """Resolve document manifest segments to readable content for
    the generalized reader (Doc 16 P2): a section reads as its
    heading over its text; a table reads as its caption/name over
    the pipe-separated grid (:meth:`Table.as_text`).  The manifest
    hash covers the canonical JSON form; the reader serves the
    human form of the same section/table — same source, two
    projections, neither inventing content."""
    sections = {f"sec-{i + 1}": s for i, s in enumerate(doc.sections)}
    tables = {f"tbl-{i + 1}": t for i, t in enumerate(doc.tables)}

    def resolve(segment) -> str | None:
        if not segment.present:
            return None
        section = sections.get(segment.segment_id)
        if section is not None:
            if section.heading:
                return f"{section.heading}\n\n{section.text}"
            return section.text
        table = tables.get(segment.segment_id)
        if table is not None:
            label = table.caption or table.name
            grid = table.as_text()
            return f"{label}\n\n{grid}" if label else grid
        return None
    return resolve
