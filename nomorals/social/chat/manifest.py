"""Manifest derivation for presentation Docs and raw long text
(Doc 16 P1).

Two LongOutput-shaped producers:

* :func:`manifest_from_doc` — a semantic :class:`~.present.Doc`.
  Segments are the summary (when present), one per section, and the
  overflow tail (when present — it is real content: the document-route
  payload).  A segment's canonical content is the canonical JSON of
  the section's own ``doc_to_json`` payload dict, so the hash covers
  exactly what the persistence form preserves.  A Doc is ephemeral
  and has no id or revision counter of its own, so its manifest id is
  content-addressed (``doc:<hash-prefix>``) and its revision is the
  content hash — stated plainly, never dressed as a counter.
* :func:`manifest_from_text` — the raw string LongOutput routes to a
  document.  One segment holding the whole text: finer blocking is
  the semantic-paging work of a later pass, and inventing block
  boundaries here would make the hash identity lie about the source.
"""

from __future__ import annotations

import json

from ...core.longdoc import (
    Completeness,
    LongDocumentManifest,
    SegmentSpec,
    build_manifest,
    hash_text,
)
from .present import Doc, doc_to_json

__all__ = ["manifest_from_doc", "manifest_from_text"]


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def manifest_from_doc(doc: Doc, *, source_ref: str = "",
                      created_by: str = "social.chat.manifest",
                      ) -> LongDocumentManifest:
    """Derive the manifest of a presentation Doc."""
    payload = json.loads(doc_to_json(doc))
    fingerprint = hash_text(doc_to_json(doc))[:16]
    specs: list[SegmentSpec] = []
    if doc.summary:
        specs.append(SegmentSpec(
            segment_id="summary", kind="summary", title="",
            content=doc.summary,
            content_ref=f"doc:{fingerprint}#summary"))
    for i, raw in enumerate(payload.get("sections", [])):
        section = doc.sections[i]
        specs.append(SegmentSpec(
            segment_id=f"sec-{i + 1}", kind="section",
            title=section.title, content=_canonical(raw),
            content_ref=f"doc:{fingerprint}#sec-{i + 1}",
            metadata={"state": section.state}))
    if doc.overflow_text:
        specs.append(SegmentSpec(
            segment_id="overflow", kind="overflow", title="",
            content=doc.overflow_text,
            content_ref=f"doc:{fingerprint}#overflow"))
    return build_manifest(
        document_id=f"doc:{fingerprint}", source_kind="present-doc",
        source_ref=source_ref, title=doc.title, revision=None,
        segment_specs=specs,
        completeness=Completeness(
            status="complete", declared_total=len(specs),
            present_count=len(specs),
            note="presentation Doc: whole by construction"),
        created_by=created_by,
        metadata={"tone": doc.tone},
    )


def manifest_from_text(text: str, *, title: str = "",
                       source_ref: str = "",
                       created_by: str = "social.chat.manifest",
                       ) -> LongDocumentManifest:
    """Derive the manifest of a raw long text (LongOutput's document
    payload): a single segment holding the full original text."""
    text = text or ""
    fingerprint = hash_text(text)[:16]
    if not title:
        first = next((ln.strip() for ln in text.splitlines()
                      if ln.strip()), "")
        title = first if len(first) <= 100 else ""
    specs = [SegmentSpec(
        segment_id="text", kind="text", title="",
        content=text, content_ref=f"text:{fingerprint}")]
    return build_manifest(
        document_id=f"text:{fingerprint}", source_kind="text",
        source_ref=source_ref, title=title, revision=None,
        segment_specs=specs,
        completeness=Completeness(
            status="complete", declared_total=1, present_count=1,
            note="raw text: the full original, one segment"),
        created_by=created_by,
    )
