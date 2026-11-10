"""Doc 16 Pass 2 — the generalized reader service.

One reader over any P1 manifest (story / forge book / document /
present Doc / raw text): full-content paging whose pages partition
the segment byte for byte, contents that show absent segments as
absent, per-reader positions that survive restarts, revision
pinning with an honest "newer revision" flag instead of a yanked
page, and the callback discipline (a rotated token makes duplicate
and out-of-order taps stale no-ops, never jumps).

Fake adapter only — no network.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.books import reader as reader_mod
from nomorals.books.manifest import (
    book_content_resolver,
    manifest_from_book,
    manifest_from_story,
    story_content_resolver,
)
from nomorals.books.model import Book
from nomorals.books.model import Chapter as BookChapter
from nomorals.books.reader import StoryReader
from nomorals.books.sources import Chapter, StoryMeta
from nomorals.documents.manifest import (
    document_content_resolver,
    manifest_from_document,
)
from nomorals.documents.model import Document, Section
from nomorals.social.chat.manifest import (
    doc_content_resolver,
    manifest_from_doc,
    manifest_from_text,
    text_content_resolver,
)
from nomorals.social.chat.present import KV, Doc
from nomorals.social.chat.present import Section as PresentSection
from nomorals.social.chat.reader_service import (
    ReaderService,
    content_resolver_from_map,
)


def _ctx():
    tmp = tempfile.mkdtemp()
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(db=None, settings=settings, extras={}), tmp


class _Adapter:
    name = "fake"
    verified = True

    def __init__(self, bodies: dict[int, str], total: int = 0) -> None:
        self.bodies = bodies
        self.total = total or max(bodies)

    def novel(self, url: str) -> StoryMeta:
        return StoryMeta(title="The Long Tale", url=url, source=self.name,
                         author="A. Teller", status="Ongoing",
                         total_chapters=self.total)

    def chapter_list(self, novel_url: str, *, limit: int = 0):
        out = [(n, f"Chapter {n}", f"https://fake/ch/{n}")
               for n in range(1, self.total + 1)]
        return out[:limit] if limit else out

    def chapter_url(self, novel_url: str, number: int) -> str:
        return f"https://fake/ch/{number}" if 1 <= number <= self.total else ""

    def fetch_chapter(self, url: str) -> Chapter:
        n = int(url.rsplit("/", 1)[-1])
        body = self.bodies[n]
        return Chapter(
            number=n, title=f"Chapter {n}", url=url,
            paragraphs=body.split("\n\n"),
            prev_url=f"https://fake/ch/{n - 1}" if n > 1 else "",
            next_url=f"https://fake/ch/{n + 1}" if n < self.total else "",
            source=self.name)


_FAST_PROFILE = {"sync_chapters": 3, "list_limit": 50,
                 "pause_s": 0.0, "timeout": 5.0}
_URL = "https://fake/novel/the-long-tale"


def _para(tag: str, chars: int) -> str:
    seed = f"{tag} carries the tale forward, beat after beat. "
    return (seed * (chars // len(seed) + 1))[:chars]


class _StoryCase(unittest.TestCase):
    bodies = {1: _para("Alpha", 7000), 2: _para("Beta", 1500),
              3: _para("Gamma", 900)}
    total = 3

    def setUp(self) -> None:
        self.ctx, self.tmp = _ctx()
        self.adapter = _Adapter(dict(self.bodies), total=self.total)
        self._orig_adapter = reader_mod.adapter_for_url
        reader_mod.adapter_for_url = lambda url: self.adapter
        self._orig_profile = StoryReader._profile
        StoryReader._profile = lambda self: dict(_FAST_PROFILE)
        self.r = StoryReader(self.ctx)
        self.slug = self.r.follow(_URL).slug
        self.store = Path(self.tmp) / "sessions.json"
        self.svc = ReaderService(self.store)

    def tearDown(self) -> None:
        reader_mod.adapter_for_url = self._orig_adapter
        StoryReader._profile = self._orig_profile

    def _manifest(self):
        return manifest_from_story(self.r, self.slug)


class StoryNavigationTests(_StoryCase):
    def test_pages_partition_chapters_and_cross_boundaries(self):
        self.r.read(self.slug, chapter=1)
        self.r.read(self.slug, chapter=2)
        manifest = self._manifest()
        session = self.svc.open("chat:telegram:1", manifest,
                                command_prefix="books go")
        resolver = story_content_resolver(self.r)
        bodies: list[str] = []
        view = self.svc.view(session.session_id, resolver)
        self.assertEqual(view.status, "ok")
        while True:
            bodies.append(view.body)
            if not (view.has_next_page or view.has_next_segment):
                break
            view = self.svc.navigate(session.session_id, resolver, "next")
        joined = "".join(bodies)
        canonical = (self.r._cached_chapter(self.slug, 1).text
                     + self.r._cached_chapter(self.slug, 2).text)
        self.assertEqual(joined, canonical)
        # one more next at the very end: no move, honest note
        end = self.svc.navigate(session.session_id, resolver, "next")
        self.assertIn("end of the document", end.note)
        # prev crosses back over the boundary to chapter 1's LAST
        # page, then walks to its first page and stops there
        back = self.svc.navigate(session.session_id, resolver, "prev")
        self.assertEqual(back.segment_id, "ch-1")
        self.assertEqual(back.page_index, back.page_count - 1)
        self.assertGreater(back.page_index, 0)
        while back.has_prev_page:
            back = self.svc.navigate(session.session_id, resolver, "prev")
        self.assertEqual(back.page_index, 0)
        start = self.svc.navigate(session.session_id, resolver, "prev")
        self.assertIn("start of the document", start.note)
        self.assertEqual(start.page_index, 0)

    def test_open_lands_on_first_present_and_resume_keeps_place(self):
        self.r.read(self.slug, chapter=2)
        manifest = self._manifest()
        session = self.svc.open("chat:telegram:1", manifest)
        resolver = story_content_resolver(self.r)
        self.svc.goto(session.session_id, resolver, segment_id="ch-2")
        again = self.svc.open("chat:telegram:1", self._manifest())
        self.assertEqual(again.segment_id, "ch-2")
        fresh = ReaderService(self.store)  # a restart: new instance
        resumed = fresh.session_for("chat:telegram:1",
                                    f"story:{self.slug}")
        self.assertIsNotNone(resumed)
        self.assertEqual(resumed.segment_id, "ch-2")


class ContentsTruthTests(_StoryCase):
    total = 5
    bodies = {n: _para(f"C{n}", 400) for n in range(1, 6)}

    def test_story_contents_synthesize_unheld_chapters(self):
        self.r.read(self.slug, chapter=1)
        self.r.read(self.slug, chapter=2)
        self.r.read(self.slug, chapter=4)
        view = self.svc.contents_view(self._manifest())
        by_id = {row.segment_id: row for row in view.rows}
        self.assertEqual(
            [by_id[f"ch-{n}"].present for n in range(1, 6)],
            [True, True, False, True, False])
        self.assertEqual(view.declared_total, 5)
        self.assertEqual(view.present_count, 3)
        text = self.svc.render_contents_text(view)
        self.assertIn("3 of 5 held", text)
        self.assertIn("not held", text)

    def test_forge_contents_show_unwritten_chapter_absent(self):
        book = Book(topic="t", slug="forge-tale", title="Forge Tale",
                    chapters=[
                        BookChapter(number=1, title="One",
                                    text="Body one."),
                        BookChapter(number=2, title="Two", text=""),
                        BookChapter(number=3, title="Three",
                                    text="Body three."),
                    ])
        manifest = manifest_from_book(book)
        view = self.svc.contents_view(manifest)
        rows = {row.segment_id: row for row in view.rows}
        self.assertFalse(rows["ch-2"].present)
        self.assertTrue(rows["ch-1"].present)
        session = self.svc.open("chat:telegram:1", manifest)
        resolver = book_content_resolver(book)
        landed = self.svc.goto(session.session_id, resolver,
                               segment_id="ch-2")
        self.assertEqual(landed.status, "unavailable")
        self.assertIn("not held", landed.note)
        # and navigation skips the absent chapter entirely
        first = self.svc.goto(session.session_id, resolver,
                              segment_id="ch-1")
        self.assertEqual(first.status, "ok")
        nxt = self.svc.navigate(session.session_id, resolver, "next")
        self.assertEqual(nxt.segment_id, "ch-3")


class GeneralityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.svc = ReaderService()  # in-memory

    def test_document_manifest_navigates(self):
        doc = Document(title="Field Report", sections=[
            Section(level=1, heading="Intro", text=_para("Intro", 4000)),
            Section(level=1, heading="Findings",
                    text="The findings stand."),
        ])
        doc.refresh_revision()
        manifest = manifest_from_document(doc)
        session = self.svc.open("reader-a", manifest,
                                command_prefix="read go")
        resolver = document_content_resolver(doc)
        view = self.svc.view(session.session_id, resolver)
        self.assertEqual(view.status, "ok")
        self.assertIn("Intro", view.body)
        bodies = [view.body]
        while view.has_next_page or view.has_next_segment:
            view = self.svc.navigate(session.session_id, resolver, "next")
            bodies.append(view.body)
        self.assertIn("The findings stand.", "".join(bodies))
        self.assertEqual(view.segment_id, "sec-2")

    def test_present_doc_manifest_navigates(self):
        doc = Doc(title="Morning Board", summary="The day in one line.",
                  sections=(
                      PresentSection(
                          title="Tasks",
                          rows=(KV("First", "ship the reader"),),
                          lines=("a loose line of prose",)),
                      PresentSection(title="Notes",
                                     lines=(_para("Notes", 3500),)),
                  ))
        manifest = manifest_from_doc(doc)
        session = self.svc.open("reader-a", manifest)
        resolver = doc_content_resolver(doc)
        view = self.svc.view(session.session_id, resolver)
        self.assertEqual(view.body, "The day in one line.")
        seen: list[str] = []
        while True:
            seen.append(view.body)
            if not (view.has_next_page or view.has_next_segment):
                break
            view = self.svc.navigate(session.session_id, resolver, "next")
        joined = "".join(seen)
        self.assertIn("First: ship the reader", joined)
        self.assertIn("a loose line of prose", joined)
        self.assertIn("Notes", joined)

    def test_text_manifest_pages_partition_exactly(self):
        text = "\n\n".join(_para(f"P{i}", 900) for i in range(12))
        manifest = manifest_from_text(text, title="Raw")
        session = self.svc.open("reader-a", manifest, page_size=3000)
        resolver = text_content_resolver(text)
        bodies: list[str] = []
        view = self.svc.view(session.session_id, resolver)
        while True:
            bodies.append(view.body)
            if not view.has_next_page:
                break
            view = self.svc.navigate(session.session_id, resolver, "next")
        self.assertEqual("".join(bodies), text)
        self.assertGreater(len(bodies), 3)


class PositionTests(_StoryCase):
    def test_two_readers_hold_independent_positions(self):
        self.r.read(self.slug, chapter=1)
        self.r.read(self.slug, chapter=2)
        manifest = self._manifest()
        resolver = story_content_resolver(self.r)
        a = self.svc.open("chat:telegram:1", manifest)
        b = self.svc.open("chat:whatsapp:9", manifest)
        self.svc.goto(a.session_id, resolver, segment_id="ch-2")
        self.svc.navigate(b.session_id, resolver, "next")
        self.assertEqual(
            self.svc.session_for("chat:telegram:1",
                                 f"story:{self.slug}").segment_id, "ch-2")
        stayed = self.svc.session_for("chat:whatsapp:9",
                                      f"story:{self.slug}")
        self.assertEqual(stayed.segment_id, "ch-1")
        self.assertGreater(stayed.offset, 0)


class RevisionPinningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.svc = ReaderService(Path(self.tmp) / "sessions.json")
        self.doc = Document(title="Living Doc", sections=[
            Section(level=1, heading="One", text="original body"),
            Section(level=1, heading="Two", text="second body"),
        ])
        self.doc.refresh_revision()

    def _contents_map(self, doc: Document) -> dict[str, str]:
        from nomorals.documents.manifest import section_content, table_content
        out: dict[str, str] = {}
        for i, s in enumerate(doc.sections):
            out[f"document:{doc.id}#sec-{i + 1}"] = section_content(s)
        for i, t in enumerate(doc.tables):
            out[f"document:{doc.id}#tbl-{i + 1}"] = table_content(t)
        return out

    def test_changed_segment_pins_and_announces_then_switch(self):
        manifest_v1 = manifest_from_document(self.doc)
        resolver_v1 = content_resolver_from_map(
            self._contents_map(self.doc))
        session = self.svc.open("reader-a", manifest_v1)
        view = self.svc.view(session.session_id, resolver_v1)
        self.assertIn("original body", view.body)
        # the source moves: section one is rewritten, revision bumps
        self.doc.sections[0].text = "rewritten body"
        self.doc.refresh_revision()
        self.assertEqual(self.doc.revision, 2)
        manifest_v2 = manifest_from_document(self.doc)
        reopened = self.svc.open("reader-a", manifest_v2)
        self.assertTrue(reopened.update_available)
        # the pinned page still serves the pinned revision's bytes
        pinned = self.svc.view(session.session_id, resolver_v1)
        self.assertIn("original body", pinned.body)
        self.assertTrue(pinned.update_available)
        # switching adopts v2, re-anchored on the same segment
        resolver_v2 = content_resolver_from_map(
            self._contents_map(self.doc))
        switched = self.svc.switch_revision(session.session_id,
                                             resolver_v2)
        self.assertEqual(switched.segment_id, "sec-1")
        self.assertIn("rewritten body", switched.body)
        self.assertFalse(switched.update_available)

    def test_unchanged_segment_adopts_growth_silently(self):
        manifest_v1 = manifest_from_document(self.doc)
        session = self.svc.open("reader-a", manifest_v1)
        # growth elsewhere: a third section appears; segment one is
        # byte-identical, so no flag, no yank — but the new segment
        # becomes reachable
        self.doc.sections.append(
            Section(level=1, heading="Three", text="third body"))
        self.doc.refresh_revision()
        manifest_v2 = manifest_from_document(self.doc)
        reopened = self.svc.open("reader-a", manifest_v2)
        self.assertFalse(reopened.update_available)
        self.assertEqual(reopened.segment_id, "sec-1")
        resolver = document_content_resolver(self.doc)
        view = self.svc.goto(session.session_id, resolver,
                             segment_id="sec-3")
        self.assertEqual(view.status, "ok")
        self.assertIn("third body", view.body)


class CallbackDisciplineTests(_StoryCase):
    def test_stale_and_double_taps_never_move_the_reader(self):
        self.r.read(self.slug, chapter=1)
        manifest = self._manifest()
        session = self.svc.open("chat:telegram:1", manifest,
                                command_prefix="books go")
        resolver = story_content_resolver(self.r)
        view = self.svc.view(session.session_id, resolver)
        token = view.token
        # button payload shape + Telegram budget
        rows = view.buttons("books go")
        payloads = [data for row in rows for _label, data in row]
        self.assertIn(f"books go {token} n", payloads)
        for payload in payloads:
            self.assertLessEqual(len(payload.encode()), 47)
        first = self.svc.act(token, "n", resolver)
        self.assertEqual(first.status, "ok")
        moved_to = self.svc.session_for(
            "chat:telegram:1", f"story:{self.slug}").offset
        self.assertGreater(moved_to, 0)
        # double-tap: the same (now rotated-out) token again
        again = self.svc.act(token, "n", resolver)
        self.assertEqual(again.status, "stale")
        self.assertIn("has not moved", again.note)
        self.assertEqual(
            self.svc.session_for("chat:telegram:1",
                                 f"story:{self.slug}").offset, moved_to)
        # an unknown token is unknown, not stale
        unknown = self.svc.act("0123456789ab", "n", resolver)
        self.assertEqual(unknown.status, "unknown")
        # contents + jump through tokens work off the current token
        current = self.svc.view(session.session_id, resolver).token
        contents = self.svc.act(current, "c", resolver)
        self.assertEqual(contents.status, "ok")
        self.assertIsNotNone(contents.contents)
        current = self.svc.view(session.session_id, resolver).token
        jumped = self.svc.act(current, "j1", resolver)
        self.assertEqual(jumped.status, "ok")
        self.assertEqual(jumped.view.segment_id, "ch-1")
        self.assertEqual(jumped.view.page_index, 0)


class HonestAbsenceTests(unittest.TestCase):
    def test_unservable_segment_and_empty_document(self):
        svc = ReaderService()
        manifest = manifest_from_text("hello world", title="Tiny")
        session = svc.open("reader-a", manifest)
        gone = svc.view(session.session_id, lambda seg: None)
        self.assertEqual(gone.status, "unavailable")
        self.assertIn("could not be served", gone.note)
        empty = manifest_from_text("", title="Empty")
        empty_session = svc.open("reader-b", empty)
        view = svc.view(empty_session.session_id,
                        text_content_resolver(""))
        self.assertIn(view.status, ("ok", "empty"))
        # a manifest with no segments at all reads as empty, plainly
        from nomorals.core.longdoc import Completeness, build_manifest
        bare = build_manifest(
            document_id="bare:1", source_kind="text", title="Bare",
            segment_specs=[], completeness=Completeness(status="open"))
        bare_session = svc.open("reader-c", bare)
        bare_view = svc.view(bare_session.session_id, lambda seg: None)
        self.assertEqual(bare_view.status, "empty")
        self.assertIn("no content yet", bare_view.note)


if __name__ == "__main__":
    unittest.main()
