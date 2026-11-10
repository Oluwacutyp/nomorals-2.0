"""Long-document manifest + revision contract (Doc 16, Pass 1).

One canonical, addressable, versioned description of a long document —
a followed story, an authored book, a parsed document, a presentation
Doc, a raw long text — that every reading surface (chat reader, HTML
edition, Markdown/PDF/EPUB exports) can derive from and cite.

The manifest is a *derived value*: it is computed from a source's
current content, never stored as a second source of truth, and it is
fully deterministic — the same source state always yields a
byte-identical manifest.  It carries:

* **identity** — a stable ``document_id`` (the source's own id/slug,
  or a content-addressed id when the source has none);
* **revision** — the source's monotonic content revision when it has
  one (the documents engine's :class:`~nomorals.documents.model.Document`
  after this pass), plus a ``content_hash`` over the segment hashes.
  ``revision_ref`` is the single comparable token: two manifests are
  the same revision of the same content iff their refs are equal.
  For sources without a revision counter the content hash *is* the
  revision identity — changed content is a new revision by
  construction, and nothing pretends a counter exists;
* **structure** — ordered segments with stable ids, titles, sizes and
  per-segment SHA-256 hashes over the segment's canonical content.
  The manifest holds content *refs*, not copies: a segment names
  where its canonical content lives (``story:<slug>#ch-3`` …);
* **completeness** — declared total vs present count, with missing
  chapter ranges.  A manifest of chapter 1 of 12 says so; a partial
  source can never pose as the whole work.  Missing content stays
  unavailable — it is declared missing, never invented;
* **provenance** — source kind/ref and the derived-from chain of
  :class:`RevisionRef` records an edition was made against, so a
  derivative can *prove* which revision it came from and staleness
  is detectable, never silent.

Layer: core (L1), stdlib only.  The per-producer derivations live
beside their producers (``documents/manifest.py``,
``social/chat/manifest.py``, ``books/manifest.py``) and build through
:func:`build_manifest`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "MANIFEST_FORMAT", "MANIFEST_VERSION", "GENESIS_REVISION",
    "RevisionRef", "ManifestSegment", "Completeness", "Provenance",
    "PresentationHints", "SegmentSpec", "LongDocumentManifest",
    "build_manifest", "hash_text", "missing_ranges",
]

#: Manifest format name + schema version (bump on breaking changes).
MANIFEST_FORMAT = "nomorals-longdoc-manifest"
MANIFEST_VERSION = 1

#: The revision every content-bearing source starts at.  Legacy
#: records written before the revision contract existed load at the
#: genesis revision — defined, never absent.
GENESIS_REVISION = 1


def hash_text(text: str) -> str:
    """SHA-256 hex of the UTF-8 bytes of ``text`` — the one hash
    primitive every segment and envelope hash in this contract uses.
    Pure function of the bytes: no timestamps, no randomness."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _envelope_hash(title: str,
                   segments: Sequence[ManifestSegment]) -> str:
    """Content identity of a whole manifest: the framing title plus
    the ordered segment identities (id, hash, presence).  Segment
    hashes transitively cover every content byte, so the envelope
    changes iff the title, the segment set/order, or any segment's
    content changes."""
    payload = {
        "title": title,
        "segments": [
            {"segment_id": s.segment_id, "sha256": s.sha256,
             "present": s.present}
            for s in segments
        ],
    }
    return hash_text(json.dumps(payload, sort_keys=True,
                                ensure_ascii=False,
                                separators=(",", ":")))


def missing_ranges(present: set[int], total: int) -> list[list[int]]:
    """Compact ``[start, end]`` ranges of ``1..total`` absent from
    ``present`` — how a story manifest declares chapters it does not
    hold without inventing entries for them."""
    out: list[list[int]] = []
    start: int | None = None
    for n in range(1, max(0, int(total)) + 1):
        if n in present:
            if start is not None:
                out.append([start, n - 1])
                start = None
        elif start is None:
            start = n
    if start is not None:
        out.append([start, int(total)])
    return out


@dataclass(frozen=True)
class RevisionRef:
    """A recorded pointer at one revision of one source — what a
    derivative/edition stores so its provenance is provable.

    ``revision`` is the source's monotonic counter when it has one,
    else ``None`` (counter-less source: the content hash alone is the
    revision identity).
    """

    source_kind: str
    document_id: str
    revision: int | None
    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {"source_kind": self.source_kind,
                "document_id": self.document_id,
                "revision": self.revision,
                "content_hash": self.content_hash}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RevisionRef:
        revision = data.get("revision")
        return cls(
            source_kind=str(data.get("source_kind", "")),
            document_id=str(data.get("document_id", "")),
            revision=(int(revision) if revision is not None else None),
            content_hash=str(data.get("content_hash", "")),
        )

    def is_current(self, *, document_id: str, revision: int | None,
                   content_hash: str) -> bool:
        """True iff the named source still sits at exactly this ref.

        Strict by design: a different document, different content
        bytes, or — when this ref carries a counter — a moved-on
        revision counter all read as NOT current.  A derivative made
        against this ref is then provably stale, never silently so.
        """
        if document_id != self.document_id:
            return False
        if content_hash != self.content_hash:
            return False
        return self.revision is None or revision == self.revision


@dataclass
class ManifestSegment:
    """One ordered, addressable piece of a long document.

    ``segment_id`` is stable within the source's addressing scheme
    (chapter numbers for chaptered sources — stable across revisions;
    positional ``sec-N`` for sectioned sources — stable within the
    manifest's revision, which is the revision the reader pins).
    ``sha256`` is over the segment's canonical content (defined per
    producer by its derivation); an absent segment (declared by the
    source, not held) has ``present=False`` and an empty hash — it is
    declared missing, never fabricated.
    """

    segment_id: str
    kind: str  # "chapter" | "section" | "table" | "summary" | "overflow" | "text"
    title: str = ""
    ordinal: int = 0
    present: bool = True
    char_count: int = 0
    byte_count: int = 0
    sha256: str = ""
    content_ref: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"segment_id": self.segment_id, "kind": self.kind,
                "title": self.title, "ordinal": self.ordinal,
                "present": self.present, "char_count": self.char_count,
                "byte_count": self.byte_count, "sha256": self.sha256,
                "content_ref": self.content_ref,
                "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ManifestSegment:
        return cls(
            segment_id=str(data.get("segment_id", "")),
            kind=str(data.get("kind", "")),
            title=str(data.get("title", "")),
            ordinal=int(data.get("ordinal", 0) or 0),
            present=bool(data.get("present", True)),
            char_count=int(data.get("char_count", 0) or 0),
            byte_count=int(data.get("byte_count", 0) or 0),
            sha256=str(data.get("sha256", "")),
            content_ref=str(data.get("content_ref", "")),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class Completeness:
    """Declared-vs-held accounting for a manifest's source.

    ``status``: ``complete`` (everything the source declares is
    present), ``partial`` (a declared total is known and some of it
    is missing), ``open`` (no declared total — completeness cannot
    be judged, e.g. an ongoing story or a still-growing book).
    """

    status: str = "complete"
    declared_total: int | None = None
    present_count: int = 0
    missing_ranges: list[list[int]] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status,
                "declared_total": self.declared_total,
                "present_count": self.present_count,
                "missing_ranges": [list(r) for r in self.missing_ranges],
                "note": self.note}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Completeness:
        total = data.get("declared_total")
        return cls(
            status=str(data.get("status", "complete")),
            declared_total=(int(total) if total is not None else None),
            present_count=int(data.get("present_count", 0) or 0),
            missing_ranges=[[int(r[0]), int(r[1])]
                            for r in data.get("missing_ranges", []) or []
                            if len(r) == 2],
            note=str(data.get("note", "")),
        )


@dataclass
class Provenance:
    """Where a manifest's source came from and what it derives from.

    ``derived_from`` is the chain of :class:`RevisionRef` records for
    editions/derivatives (empty for a primary source manifest).
    """

    derived_from: list[RevisionRef] = field(default_factory=list)
    created_by: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"derived_from": [r.to_dict() for r in self.derived_from],
                "created_by": self.created_by}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Provenance:
        return cls(
            derived_from=[RevisionRef.from_dict(r)
                          for r in data.get("derived_from", []) or []],
            created_by=str(data.get("created_by", "")),
        )


@dataclass
class PresentationHints:
    """Presentation policy a manifest carries for later passes.

    ``brand_policy`` follows the typography law: ``default`` (the
    Peacethefirst stamp on presented output), ``off`` and ``custom``
    are the owner's preserved alternatives.  Hints only in P1 — no
    surface consumes them yet.
    """

    brand_policy: str = "default"
    theme: str = ""
    locale: str = ""
    direction: str = "auto"  # "auto" | "ltr" | "rtl"

    def to_dict(self) -> dict[str, Any]:
        return {"brand_policy": self.brand_policy, "theme": self.theme,
                "locale": self.locale, "direction": self.direction}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PresentationHints:
        return cls(
            brand_policy=str(data.get("brand_policy", "default")),
            theme=str(data.get("theme", "")),
            locale=str(data.get("locale", "")),
            direction=str(data.get("direction", "auto")),
        )


@dataclass
class SegmentSpec:
    """Adapter input to :func:`build_manifest`: one segment with its
    canonical content (``None`` = declared by the source but not
    held).  Hashing and sizing happen in exactly one place — here in
    the contract — so no producer can drift the rules."""

    segment_id: str
    kind: str
    title: str = ""
    content: str | None = ""
    content_ref: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LongDocumentManifest:
    """The canonical manifest of one long document at one revision."""

    document_id: str
    source_kind: str  # "story" | "forge-book" | "document" | "present-doc" | "text"
    source_ref: str = ""
    title: str = ""
    author: str = ""
    revision: int | None = None
    content_hash: str = ""
    segments: list[ManifestSegment] = field(default_factory=list)
    completeness: Completeness = field(default_factory=Completeness)
    provenance: Provenance = field(default_factory=Provenance)
    presentation: PresentationHints = field(default_factory=PresentationHints)
    metadata: dict[str, Any] = field(default_factory=dict)
    format: str = MANIFEST_FORMAT
    version: int = MANIFEST_VERSION

    @property
    def revision_ref(self) -> str:
        """The single comparable revision token: ``r<n>:<hash>`` when
        the source has a revision counter, else ``h:<hash>`` (the
        hash itself is the revision identity)."""
        if self.revision is not None:
            return f"r{self.revision}:{self.content_hash}"
        return f"h:{self.content_hash}"

    def segment(self, segment_id: str) -> ManifestSegment | None:
        for seg in self.segments:
            if seg.segment_id == segment_id:
                return seg
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"format": self.format, "version": self.version,
                "document_id": self.document_id,
                "source_kind": self.source_kind,
                "source_ref": self.source_ref,
                "title": self.title, "author": self.author,
                "revision": self.revision,
                "content_hash": self.content_hash,
                "segments": [s.to_dict() for s in self.segments],
                "completeness": self.completeness.to_dict(),
                "provenance": self.provenance.to_dict(),
                "presentation": self.presentation.to_dict(),
                "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LongDocumentManifest:
        if str(data.get("format", "")) != MANIFEST_FORMAT:
            raise ValueError(
                f"not a {MANIFEST_FORMAT} payload: "
                f"format={data.get('format')!r}")
        revision = data.get("revision")
        return cls(
            document_id=str(data.get("document_id", "")),
            source_kind=str(data.get("source_kind", "")),
            source_ref=str(data.get("source_ref", "")),
            title=str(data.get("title", "")),
            author=str(data.get("author", "")),
            revision=(int(revision) if revision is not None else None),
            content_hash=str(data.get("content_hash", "")),
            segments=[ManifestSegment.from_dict(s)
                      for s in data.get("segments", []) or []],
            completeness=Completeness.from_dict(
                data.get("completeness", {}) or {}),
            provenance=Provenance.from_dict(data.get("provenance", {}) or {}),
            presentation=PresentationHints.from_dict(
                data.get("presentation", {}) or {}),
            metadata=dict(data.get("metadata", {}) or {}),
            format=str(data.get("format", MANIFEST_FORMAT)),
            version=int(data.get("version", MANIFEST_VERSION) or
                        MANIFEST_VERSION),
        )

    def to_json(self) -> str:
        """Canonical JSON form (sorted keys): the same manifest always
        serializes to the same bytes — this is the round-trip and
        determinism surface the contract tests pin."""
        return json.dumps(self.to_dict(), sort_keys=True,
                          ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> LongDocumentManifest:
        return cls.from_dict(json.loads(text))

    def verify(self, contents: Mapping[str, str]) -> list[str]:
        """Check supplied canonical contents against this manifest.

        ``contents`` maps ``content_ref`` → canonical content text.
        Returns a list of problems (empty = verified): a present
        segment whose content is missing from the map, a segment
        whose recomputed hash differs from the recorded hash (the
        tamper case), a size mismatch, or an envelope mismatch (the
        framing/segment set no longer hashes to ``content_hash``).
        """
        problems: list[str] = []
        for seg in self.segments:
            if not seg.present:
                continue
            content = contents.get(seg.content_ref)
            if content is None:
                problems.append(
                    f"segment {seg.segment_id}: no content supplied "
                    f"for {seg.content_ref}")
                continue
            digest = hash_text(content)
            if digest != seg.sha256:
                problems.append(
                    f"segment {seg.segment_id}: content hash mismatch "
                    f"(manifest {seg.sha256[:12]}…, supplied "
                    f"{digest[:12]}…)")
            elif len(content) != seg.char_count:
                problems.append(
                    f"segment {seg.segment_id}: char count mismatch "
                    f"(manifest {seg.char_count}, supplied "
                    f"{len(content)})")
        if _envelope_hash(self.title, self.segments) != self.content_hash:
            problems.append(
                "envelope: content_hash does not match the title + "
                "segment set (manifest was altered after derivation)")
        return problems


def build_manifest(*, document_id: str, source_kind: str,
                   source_ref: str = "", title: str = "",
                   author: str = "", revision: int | None = None,
                   segment_specs: Sequence[SegmentSpec],
                   completeness: Completeness,
                   derived_from: Sequence[RevisionRef] = (),
                   created_by: str = "",
                   presentation: PresentationHints | None = None,
                   metadata: Mapping[str, Any] | None = None,
                   ) -> LongDocumentManifest:
    """Assemble a manifest from adapter-supplied segment specs.

    The one place segment hashes, sizes, ordinals and the envelope
    ``content_hash`` are computed — adapters supply canonical content
    and addressing; the contract does the rest, identically for every
    producer.
    """
    segments: list[ManifestSegment] = []
    for ordinal, spec in enumerate(segment_specs):
        if spec.content is None:
            segments.append(ManifestSegment(
                segment_id=spec.segment_id, kind=spec.kind,
                title=spec.title, ordinal=ordinal, present=False,
                content_ref=spec.content_ref,
                metadata=dict(spec.metadata)))
            continue
        encoded = spec.content.encode("utf-8")
        segments.append(ManifestSegment(
            segment_id=spec.segment_id, kind=spec.kind,
            title=spec.title, ordinal=ordinal, present=True,
            char_count=len(spec.content), byte_count=len(encoded),
            sha256=hashlib.sha256(encoded).hexdigest(),
            content_ref=spec.content_ref,
            metadata=dict(spec.metadata)))
    return LongDocumentManifest(
        document_id=document_id, source_kind=source_kind,
        source_ref=source_ref, title=title, author=author,
        revision=revision,
        content_hash=_envelope_hash(title, segments),
        segments=segments, completeness=completeness,
        provenance=Provenance(derived_from=list(derived_from),
                              created_by=created_by),
        presentation=presentation or PresentationHints(),
        metadata=dict(metadata or {}),
    )
