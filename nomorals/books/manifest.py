"""Manifest derivation for books: followed stories + forge books
(Doc 16 P1).

* :func:`manifest_from_book` — an authored :class:`~.model.Book`.
  One segment per chapter; a declared chapter whose text is not
  written yet is an *absent* segment (present=False, no hash) — it is
  declared missing, never fabricated.  Segment ids are chapter
  numbers (``ch-<n>``), stable across revisions.
* :func:`manifest_from_story` — a followed story on the
  :class:`~.reader.StoryReader` shelf.  Segments are the chapters
  held in the local library cache.  Derivation is a pure local read:
  it NEVER fetches (``_cached_chapter``/``_cached_numbers`` are the
  disk-truth seams) — a chapter the library does not hold is declared
  through completeness (declared total + missing ranges), not pulled
  from the network and not invented.  A chapter segment's canonical
  content is the chapter body text (``Chapter.text`` — the prose
  paragraphs; the title rides as segment metadata, the same split
  ``StoryReader.chapter_text`` wraps when reading).

Neither source has a monotonic revision counter (a book's
``updated_at`` is a wall clock, not a content version), so both
manifests carry ``revision=None``: the content hash is the revision
identity, per the contract in :mod:`nomorals.core.longdoc`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.longdoc import (
    Completeness,
    LongDocumentManifest,
    PresentationHints,
    SegmentSpec,
    build_manifest,
    missing_ranges,
)

if TYPE_CHECKING:  # annotations only — no runtime import cycle
    from .model import Book
    from .reader import StoryReader

__all__ = ["manifest_from_book", "manifest_from_story"]


def manifest_from_book(book: Book, *,
                       created_by: str = "books.manifest",
                       ) -> LongDocumentManifest:
    """Derive the manifest of an authored forge Book."""
    specs: list[SegmentSpec] = []
    for chapter in book.chapters:
        specs.append(SegmentSpec(
            segment_id=f"ch-{chapter.number}", kind="chapter",
            title=chapter.title,
            content=chapter.text if chapter.text else None,
            content_ref=f"forge:{book.slug}#ch-{chapter.number}",
            metadata={"status": chapter.status,
                      "words": chapter.words},
        ))
    total = len(book.chapters)
    present = sum(1 for s in specs if s.content is not None)
    if total == 0:
        completeness = Completeness(
            status="open", declared_total=0, present_count=0,
            note="no chapters declared yet")
    elif present < total:
        completeness = Completeness(
            status="partial", declared_total=total,
            present_count=present,
            note="declared chapters not yet written are absent "
                 "segments")
    elif book.organic and not book.concluded:
        completeness = Completeness(
            status="open", declared_total=total,
            present_count=present,
            note="every declared chapter is written; the book is "
                 "still growing (organic, not concluded)")
    else:
        completeness = Completeness(
            status="complete", declared_total=total,
            present_count=present, note="every chapter written")
    return build_manifest(
        document_id=f"forge:{book.slug}", source_kind="forge-book",
        source_ref=book.slug, title=book.display_title,
        author=book.author, revision=None, segment_specs=specs,
        completeness=completeness, created_by=created_by,
        presentation=PresentationHints(locale=book.language or ""),
        metadata={"genre": book.genre, "book_status": book.status,
                  "organic": book.organic},
    )


def manifest_from_story(reader: StoryReader, slug: str, *,
                        created_by: str = "books.manifest",
                        ) -> LongDocumentManifest:
    """Derive the manifest of a followed story from its local cache
    only (no network, ever)."""
    story = reader._load(slug)
    specs: list[SegmentSpec] = []
    held: set[int] = set()
    for number in sorted(reader._cached_numbers(story.slug)):
        chapter = reader._cached_chapter(story.slug, number)
        if chapter is None:
            continue  # vanished between listing and read — not held
        held.add(number)
        specs.append(SegmentSpec(
            segment_id=f"ch-{number}", kind="chapter",
            title=chapter.title, content=chapter.text,
            content_ref=f"story:{story.slug}#ch-{number}",
            metadata={"source": chapter.source,
                      "words": chapter.words},
        ))
    declared_total = story.ingest_total or story.total_chapters or None
    if declared_total is None:
        completeness = Completeness(
            status="open", declared_total=None,
            present_count=len(specs),
            note="source chapter total unknown — completeness "
                 "cannot be judged")
    else:
        ranges = missing_ranges(held, declared_total)
        completeness = Completeness(
            status="partial" if ranges else "complete",
            declared_total=declared_total, present_count=len(specs),
            missing_ranges=ranges,
            note=("chapters declared by the source but not held in "
                  "the local library" if ranges else
                  "every declared chapter is held locally"))
    return build_manifest(
        document_id=f"story:{story.slug}", source_kind="story",
        source_ref=story.url, title=story.title,
        author=story.author, revision=None, segment_specs=specs,
        completeness=completeness, created_by=created_by,
        metadata={"source": story.source,
                  "story_status": story.status,
                  "genres": list(story.genres)},
    )
