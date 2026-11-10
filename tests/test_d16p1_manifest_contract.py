"""Doc 16 Pass 1 — the long-document manifest + revision contract.

These tests ARE the contract the later Doc-16 passes read: the
manifest schema and its round-trip, deterministic derivation,
per-segment tamper detection, the Document revision rules
(genesis → seal → bump; drift detectable; edition staleness
provable), and per-producer completeness honesty.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.books.manifest import manifest_from_book, manifest_from_story
from nomorals.books.model import STATUS_PLAN, STATUS_WRITTEN, Book
from nomorals.books.model import Chapter as BookChapter
from nomorals.books.reader import FollowedStory, StoryReader
from nomorals.books.sources import Chapter as SourceChapter
from nomorals.core.longdoc import (
    GENESIS_REVISION,
    Completeness,
    LongDocumentManifest,
    RevisionRef,
    SegmentSpec,
    build_manifest,
)
from nomorals.documents.manifest import (
    document_revision_ref,
    manifest_from_document,
    section_content,
    table_content,
)
from nomorals.documents.model import Document, Table
from nomorals.documents.model import Section as DocSection
from nomorals.social.chat.manifest import manifest_from_doc, manifest_from_text
from nomorals.social.chat.present import KV, Doc, Section


def _document() -> Document:
    return Document(
        id="doc-fixed-id", format="md", title="Doc Title", author="Au",
        source="file.md",
        sections=[DocSection(level=1, heading="H1", text="Body text."),
                  DocSection(level=2, heading="H2", text="More body.")],
        tables=[Table(name="T1", headers=["a"], rows=[["1"]])],
    )


def _book() -> Book:
    return Book(
        topic="T", slug="t-book", title="T Book", author="Au",
        chapters=[
            BookChapter(number=1, title="One", text="Body one.",
                        status=STATUS_WRITTEN),
            BookChapter(number=2, title="Two", text="",
                        status=STATUS_PLAN),
            BookChapter(number=3, title="Three", text="Body three.",
                        status=STATUS_WRITTEN),
        ])


def _present_doc() -> Doc:
    return Doc(
        title="Report", summary="The summary.",
        sections=(Section(title="S1", rows=(KV("k", "v"),),
                          lines=("a line",)),),
        overflow_text="the long tail")


def _reader_with_story(total=10, cached=(1, 2, 3, 5)):
    tmp = tempfile.mkdtemp()
    ctx = SimpleNamespace(db=None,
                          settings=SimpleNamespace(workspace_dir=tmp))
    reader = StoryReader(ctx)
    story = FollowedStory(
        slug="the-test-tale", title="The Test Tale",
        url="https://example.com/tale", source="fake", author="A. Uthor",
        total_chapters=total)
    reader._save(story)
    for n in cached:
        reader._store_chapter(story.slug, SourceChapter(
            number=n, title=f"Chapter {n}",
            url=f"https://example.com/tale/{n}",
            paragraphs=[f"Alpha {n}.", f"Beta {n}."]))
    return reader, story


class RoundTripTests(unittest.TestCase):
    """Schema round-trip per producer shape: manifest → canonical
    JSON → manifest is the identical object, byte-for-byte."""

    def _round_trip(self, manifest):
        text = manifest.to_json()
        back = LongDocumentManifest.from_json(text)
        self.assertEqual(back, manifest)
        self.assertEqual(back.to_json(), text)
        return back

    def test_document_shape(self):
        self._round_trip(manifest_from_document(_document()))

    def test_present_doc_shape(self):
        self._round_trip(manifest_from_doc(_present_doc()))

    def test_text_shape(self):
        self._round_trip(manifest_from_text("line one\nline two"))

    def test_forge_book_shape(self):
        self._round_trip(manifest_from_book(_book()))

    def test_story_shape(self):
        reader, story = _reader_with_story()
        self._round_trip(manifest_from_story(reader, story.slug))

    def test_derived_from_chain_round_trips(self):
        manifest = build_manifest(
            document_id="doc:x", source_kind="document", title="Ed",
            segment_specs=[SegmentSpec(segment_id="sec-1",
                                       kind="section", content="body",
                                       content_ref="document:x#sec-1")],
            completeness=Completeness(),
            derived_from=[RevisionRef(source_kind="document",
                                      document_id="doc:src", revision=3,
                                      content_hash="ab" * 32)],
            created_by="test")
        back = self._round_trip(manifest)
        self.assertEqual(back.provenance.derived_from[0].revision, 3)

    def test_wrong_format_rejected(self):
        with self.assertRaises(ValueError):
            LongDocumentManifest.from_dict({"format": "something-else"})


class DeterminismTests(unittest.TestCase):
    """Same source state → identical manifest, every time."""

    def test_document_derivation_deterministic(self):
        doc = _document()
        self.assertEqual(manifest_from_document(doc).to_json(),
                         manifest_from_document(doc).to_json())

    def test_same_content_same_segment_hashes(self):
        a, b = _document(), _document()
        a.id, b.id = "doc-a", "doc-b"  # identity differs…
        ma, mb = manifest_from_document(a), manifest_from_document(b)
        # …content identity does not.
        self.assertEqual([s.sha256 for s in ma.segments],
                         [s.sha256 for s in mb.segments])
        self.assertEqual(ma.content_hash, mb.content_hash)

    def test_story_derivation_deterministic(self):
        reader, story = _reader_with_story()
        self.assertEqual(manifest_from_story(reader, story.slug).to_json(),
                         manifest_from_story(reader, story.slug).to_json())

    def test_revision_ref_stable_for_same_content(self):
        doc = _document()
        self.assertEqual(manifest_from_document(doc).revision_ref,
                         manifest_from_document(doc).revision_ref)


class TamperTests(unittest.TestCase):
    """Flipping a segment's bytes is detected, per segment; a framing
    change is caught by the envelope; untouched segments verify."""

    def _contents(self, doc):
        contents = {}
        for i, section in enumerate(doc.sections):
            contents[f"document:{doc.id}#sec-{i + 1}"] = \
                section_content(section)
        for i, table in enumerate(doc.tables):
            contents[f"document:{doc.id}#tbl-{i + 1}"] = \
                table_content(table)
        return contents

    def test_clean_contents_verify(self):
        doc = _document()
        manifest = manifest_from_document(doc)
        self.assertEqual(manifest.verify(self._contents(doc)), [])

    def test_flipped_segment_detected_exactly(self):
        doc = _document()
        manifest = manifest_from_document(doc)
        contents = self._contents(doc)
        key = f"document:{doc.id}#sec-2"
        contents[key] = contents[key].replace("More", "M0re")
        problems = manifest.verify(contents)
        self.assertEqual(len(problems), 1)
        self.assertIn("sec-2", problems[0])

    def test_missing_content_is_a_problem(self):
        doc = _document()
        manifest = manifest_from_document(doc)
        problems = manifest.verify({})
        self.assertTrue(any("sec-1" in p for p in problems))

    def test_altered_manifest_envelope_detected(self):
        doc = _document()
        manifest = manifest_from_document(doc)
        manifest.title = "A Different Title"  # post-derivation edit
        problems = manifest.verify(self._contents(doc))
        self.assertTrue(any("envelope" in p for p in problems))

    def test_story_segment_tamper_detected(self):
        reader, story = _reader_with_story(cached=(1, 2))
        manifest = manifest_from_story(reader, story.slug)
        contents = {}
        for n in (1, 2):
            ch = reader._cached_chapter(story.slug, n)
            contents[f"story:{story.slug}#ch-{n}"] = ch.text
        self.assertEqual(manifest.verify(contents), [])
        contents[f"story:{story.slug}#ch-1"] += " injected"
        problems = manifest.verify(contents)
        self.assertEqual(len(problems), 1)
        self.assertIn("ch-1", problems[0])


class DocumentRevisionTests(unittest.TestCase):
    """The revision contract on the documents model."""

    def test_fresh_document_is_genesis_unsealed(self):
        doc = _document()
        self.assertEqual(doc.revision, GENESIS_REVISION)
        self.assertEqual(doc.revision_hash, "")
        self.assertFalse(doc.revision_drifted())

    def test_legacy_payload_loads_at_genesis(self):
        payload = _document().to_dict()
        payload.pop("revision")
        payload.pop("revision_hash")
        doc = Document.from_dict(payload)
        self.assertEqual(doc.revision, GENESIS_REVISION)
        self.assertEqual(doc.revision_hash, "")

    def test_revision_survives_round_trip(self):
        doc = _document()
        doc.refresh_revision()
        doc.sections[0].text = "Changed."
        self.assertTrue(doc.refresh_revision())
        back = Document.from_dict(doc.to_dict())
        self.assertEqual(back.revision, doc.revision)
        self.assertEqual(back.revision_hash, doc.revision_hash)

    def test_first_refresh_seals_without_bumping(self):
        doc = _document()
        self.assertFalse(doc.refresh_revision())
        self.assertEqual(doc.revision, GENESIS_REVISION)
        self.assertTrue(doc.revision_hash)
        self.assertFalse(doc.refresh_revision())  # no change, no bump
        self.assertEqual(doc.revision, GENESIS_REVISION)

    def test_mutation_bumps_and_rescales(self):
        doc = _document()
        doc.refresh_revision()
        doc.sections[1].text = "Edited body."
        self.assertTrue(doc.revision_drifted())  # detectable pre-refresh
        self.assertTrue(doc.refresh_revision())
        self.assertEqual(doc.revision, GENESIS_REVISION + 1)
        self.assertFalse(doc.revision_drifted())

    def test_exact_hash_is_byte_exact_unlike_dedup_hash(self):
        a, b = _document(), _document()
        b.sections[0].text = "body text."  # case-only change
        self.assertEqual(a.content_hash(), b.content_hash())  # lossy
        self.assertNotEqual(a.exact_content_hash(),
                            b.exact_content_hash())  # exact

    def test_metadata_is_not_content(self):
        doc = _document()
        doc.refresh_revision()
        doc.metadata["anything"] = "new"
        self.assertFalse(doc.revision_drifted())
        self.assertFalse(doc.refresh_revision())
        self.assertEqual(doc.revision, GENESIS_REVISION)

    def test_edition_ref_goes_provably_stale(self):
        doc = _document()
        doc.refresh_revision()
        ref = document_revision_ref(doc)
        self.assertTrue(ref.is_current(
            document_id=doc.id, revision=doc.revision,
            content_hash=doc.exact_content_hash()))
        doc.sections[0].text = "Moved on."
        doc.refresh_revision()
        self.assertFalse(ref.is_current(
            document_id=doc.id, revision=doc.revision,
            content_hash=doc.exact_content_hash()))

    def test_unrefreshed_mutation_detected_by_hash(self):
        doc = _document()
        doc.refresh_revision()
        ref = document_revision_ref(doc)
        doc.sections[0].text = "Sneaky edit, no refresh."
        self.assertFalse(ref.is_current(
            document_id=doc.id, revision=doc.revision,
            content_hash=doc.exact_content_hash()))


class ManifestRevisionTests(unittest.TestCase):
    """Revision discipline at the manifest level."""

    def test_document_manifest_carries_the_counter(self):
        doc = _document()
        doc.refresh_revision()
        manifest = manifest_from_document(doc)
        self.assertEqual(manifest.revision, GENESIS_REVISION)
        self.assertTrue(manifest.revision_ref.startswith("r1:"))
        doc.sections[0].text = "New edition of the body."
        doc.refresh_revision()
        moved = manifest_from_document(doc)
        self.assertEqual(moved.revision, GENESIS_REVISION + 1)
        self.assertNotEqual(moved.revision_ref, manifest.revision_ref)

    def test_counterless_sources_use_hash_identity(self):
        book = _book()
        manifest = manifest_from_book(book)
        self.assertIsNone(manifest.revision)
        self.assertTrue(manifest.revision_ref.startswith("h:"))
        book.chapters[0].text = "Rewritten body."
        self.assertNotEqual(manifest_from_book(book).revision_ref,
                            manifest.revision_ref)


class CompletenessTests(unittest.TestCase):
    """Completeness is declared honestly per producer."""

    def test_forge_book_partial_with_absent_segment(self):
        manifest = manifest_from_book(_book())
        comp = manifest.completeness
        self.assertEqual(comp.status, "partial")
        self.assertEqual(comp.declared_total, 3)
        self.assertEqual(comp.present_count, 2)
        absent = manifest.segment("ch-2")
        self.assertFalse(absent.present)
        self.assertEqual(absent.sha256, "")

    def test_forge_book_complete(self):
        book = _book()
        book.chapters[1].text = "Now written."
        book.chapters[1].status = STATUS_WRITTEN
        self.assertEqual(manifest_from_book(book).completeness.status,
                         "complete")

    def test_forge_book_organic_unconcluded_is_open(self):
        book = _book()
        book.organic = True
        book.concluded = False
        for ch in book.chapters:
            ch.text = ch.text or "Written."
            ch.status = STATUS_WRITTEN
        self.assertEqual(manifest_from_book(book).completeness.status,
                         "open")

    def test_story_partial_with_missing_ranges(self):
        reader, story = _reader_with_story(total=10, cached=(1, 2, 3, 5))
        manifest = manifest_from_story(reader, story.slug)
        comp = manifest.completeness
        self.assertEqual(comp.status, "partial")
        self.assertEqual(comp.declared_total, 10)
        self.assertEqual(comp.present_count, 4)
        self.assertEqual(comp.missing_ranges, [[4, 4], [6, 10]])
        self.assertEqual([s.segment_id for s in manifest.segments],
                         ["ch-1", "ch-2", "ch-3", "ch-5"])

    def test_story_unknown_total_is_open(self):
        reader, story = _reader_with_story(total=0, cached=(1,))
        manifest = manifest_from_story(reader, story.slug)
        self.assertEqual(manifest.completeness.status, "open")
        self.assertIsNone(manifest.completeness.declared_total)

    def test_story_complete(self):
        reader, story = _reader_with_story(total=2, cached=(1, 2))
        self.assertEqual(
            manifest_from_story(reader, story.slug).completeness.status,
            "complete")

    def test_document_and_doc_and_text_are_complete(self):
        for manifest in (manifest_from_document(_document()),
                         manifest_from_doc(_present_doc()),
                         manifest_from_text("body")):
            self.assertEqual(manifest.completeness.status, "complete")
            self.assertEqual(manifest.completeness.present_count,
                             len(manifest.segments))


class AddressingTests(unittest.TestCase):
    """Stable ids + content refs, and the present-Doc segment set."""

    def test_document_refs(self):
        manifest = manifest_from_document(_document())
        self.assertEqual(manifest.document_id, "doc-fixed-id")
        self.assertEqual(manifest.segments[0].content_ref,
                         "document:doc-fixed-id#sec-1")
        self.assertEqual(manifest.segments[-1].kind, "table")

    def test_present_doc_segments(self):
        manifest = manifest_from_doc(_present_doc())
        self.assertEqual([s.segment_id for s in manifest.segments],
                         ["summary", "sec-1", "overflow"])
        self.assertEqual(manifest.segments[0].kind, "summary")
        self.assertTrue(manifest.document_id.startswith("doc:"))

    def test_story_refs_carry_chapter_numbers(self):
        reader, story = _reader_with_story(cached=(1, 2))
        manifest = manifest_from_story(reader, story.slug)
        self.assertEqual(manifest.segments[1].content_ref,
                         "story:the-test-tale#ch-2")


if __name__ == "__main__":
    unittest.main()
