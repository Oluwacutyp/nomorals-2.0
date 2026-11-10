"""Doc 16 Pass 6 — live outputs: the lifecycle battery.

What P2 shipped and this pass completes (evaluation gap 7, doc §5/§6):

* **Draft/finalized labels.**  Two producers record a real
  production state in their segment metadata — a forge chapter's
  ``status`` (``writing`` = a draft whose partial text is already
  served; ``planned`` with no text = declared, not yet written) and
  a presentation section's ``state`` (any state but ``ready`` is
  not final).  P2's reader never surfaced either.  Now the
  contents rows and the page render label a not-final segment from
  that manifest truth only; producers that record no state
  (stories, parsed documents, raw text) get no label, and a
  contradictory record (text present, status still ``planned``)
  gets no invented reconciliation.
* **Switch honesty.**  ``switch_revision`` on a segment that
  VANISHED from the newer revision used to land on the first
  present segment, silently.  It now lands on the nearest
  surviving segment (the follower, else the predecessor) and the
  returned view says where the reader landed and why; a surviving
  segment's switch is noted too.
* **Restart + corruption.**  A new ReaderService over the same
  on-disk store resumes the same pinned page at the same revision,
  update flag intact.  A corrupt store is contained — reading
  restarts empty — and the failure is NAMED in the package failure
  ledger instead of swallowed silently.
* **Removal.**  Content removed under the reader renders
  unavailable with the reason and a recovery path (the contents
  pointer); the session is not corrupted and keeps navigating.

The "new content available" notice itself is render-time (P2):
there is no background trigger anywhere in the shipped code that
re-derives a manifest for an open session — sessions observe a new
revision when a surface next opens the document, and that render
is where the flag speaks.  This suite pins that surfacing
end-to-end; no publish path is touched.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.books.manifest import book_content_resolver, manifest_from_book
from nomorals.books.model import Book
from nomorals.books.model import Chapter as BookChapter
from nomorals.core.longdoc import (
    Completeness,
    SegmentSpec,
    build_manifest,
)
from nomorals.documents.manifest import manifest_from_document
from nomorals.documents.model import Document, Section
from nomorals.social.chat import failures
from nomorals.social.chat.manifest import (
    doc_content_resolver,
    manifest_from_doc,
)
from nomorals.social.chat.present import KV, Doc
from nomorals.social.chat.present import Section as PresentSection
from nomorals.social.chat.reader_service import (
    ReaderService,
    content_resolver_from_map,
)


def _manifest(doc_id: str, bodies: dict[str, str | None],
              titles: dict[str, str] | None = None):
    """A manifest over ordered ``segment_id → body`` (None = absent)."""
    titles = titles or {}
    specs = [
        SegmentSpec(segment_id=sid, kind="chapter",
                    title=titles.get(sid, sid), content=body,
                    content_ref=f"{doc_id}#{sid}")
        for sid, body in bodies.items()
    ]
    return build_manifest(
        document_id=doc_id, source_kind="text", title="Tale",
        segment_specs=specs,
        completeness=Completeness(
            status="complete", declared_total=len(specs),
            present_count=sum(1 for b in bodies.values() if b)))


def _resolver(manifest, bodies: dict[str, str]):
    return content_resolver_from_map(
        {seg.content_ref: bodies[seg.segment_id]
         for seg in manifest.segments if seg.segment_id in bodies})


class ForgeStateLabelTests(unittest.TestCase):
    """(a) A forge book's chapters: written / writing / planned."""

    def setUp(self) -> None:
        self.svc = ReaderService()
        self.book = Book(
            topic="t", slug="live-book", title="Live Book",
            chapters=[
                BookChapter(number=1, title="Finished", text="Done body.",
                            status="written"),
                BookChapter(number=2, title="Drafting",
                            text="Partial body, still growing.",
                            status="writing"),
                BookChapter(number=3, title="Someday", text="",
                            status="planned"),
            ])
        self.manifest = manifest_from_book(self.book)
        self.resolver = book_content_resolver(self.book)

    def test_contents_label_not_final_chapters_only(self):
        view = self.svc.contents_view(self.manifest)
        rows = {r.segment_id: r for r in view.rows}
        self.assertEqual(rows["ch-1"].state, "")
        self.assertEqual(rows["ch-2"].state,
                         "draft — still being written")
        self.assertEqual(rows["ch-3"].state,
                         "planned — not yet written")
        self.assertFalse(rows["ch-3"].present)
        text = self.svc.render_contents_text(view)
        self.assertIn("draft — still being written", text)
        self.assertIn("planned — not yet written", text)
        finished_line = next(ln for ln in text.splitlines()
                             if "Finished" in ln)
        self.assertNotIn("draft", finished_line)
        self.assertNotIn("planned", finished_line)

    def test_page_render_labels_the_draft_being_served(self):
        session = self.svc.open("reader-a", self.manifest)
        view = self.svc.goto(session.session_id, self.resolver,
                             segment_id="ch-2")
        self.assertEqual(view.status, "ok")
        self.assertIn("Partial body", view.body)
        self.assertEqual(view.segment_state,
                         "draft — still being written")
        rendered = self.svc.render_text(view)
        self.assertIn("✎ draft — still being written", rendered)
        done = self.svc.goto(session.session_id, self.resolver,
                             segment_id="ch-1")
        self.assertEqual(done.segment_state, "")
        self.assertNotIn("✎", self.svc.render_text(done))

    def test_present_but_planned_gets_no_invented_label(self):
        # contradictory source data (text exists, never marked
        # written): presence is served, no state is claimed
        book = Book(topic="t", slug="odd-book", title="Odd",
                    chapters=[BookChapter(number=1, title="One",
                                          text="Body one.")])
        manifest = manifest_from_book(book)
        view = self.svc.contents_view(manifest)
        self.assertEqual(view.rows[0].state, "")


class PresentDocStateLabelTests(unittest.TestCase):
    """(a) A presentation Doc's sections: ready vs not-final states."""

    def setUp(self) -> None:
        self.svc = ReaderService()
        self.doc = Doc(title="Morning Board", sections=(
            PresentSection(title="Tasks",
                           rows=(KV("First", "ship it"),)),
            PresentSection(title="Prices", state="empty",
                           state_detail="no feed yet"),
            PresentSection(title="Feed", state="error",
                           state_detail="source down"),
        ))
        self.manifest = manifest_from_doc(self.doc)
        self.resolver = doc_content_resolver(self.doc)

    def test_contents_and_pages_label_non_ready_sections(self):
        view = self.svc.contents_view(self.manifest)
        rows = {r.segment_id: r for r in view.rows}
        self.assertEqual(rows["sec-1"].state, "")
        self.assertEqual(rows["sec-2"].state, "not final — empty")
        self.assertEqual(rows["sec-3"].state, "not final — error")
        session = self.svc.open("reader-a", self.manifest)
        page = self.svc.goto(session.session_id, self.resolver,
                             segment_id="sec-2")
        self.assertEqual(page.segment_state, "not final — empty")
        ready = self.svc.goto(session.session_id, self.resolver,
                              segment_id="sec-1")
        self.assertEqual(ready.segment_state, "")


class UnlabelledProducerTests(unittest.TestCase):
    """(a) Producers recording no state get no label — ever."""

    def test_document_segments_carry_no_state_label(self):
        svc = ReaderService()
        doc = Document(title="Report", sections=[
            Section(level=1, heading="One", text="body one")])
        view = svc.contents_view(manifest_from_document(doc))
        self.assertTrue(all(r.state == "" for r in view.rows))


class SwitchRevisionTests(unittest.TestCase):
    """(c) The explicit switch: nearest surviving + said out loud."""

    def setUp(self) -> None:
        self.svc = ReaderService()
        self.titles = {"ch-1": "Alpha", "ch-2": "Beta",
                       "ch-3": "Gamma", "ch-4": "Delta"}

    def _open_flagged(self, v1_bodies, v2_bodies, at: str):
        v1 = _manifest("doc:tale", v1_bodies, self.titles)
        session = self.svc.open("reader-a", v1)
        resolver1 = _resolver(v1, {k: v for k, v in v1_bodies.items()
                                   if v})
        self.svc.goto(session.session_id, resolver1, segment_id=at)
        v2 = _manifest("doc:tale", v2_bodies, self.titles)
        reopened = self.svc.open("reader-a", v2)
        self.assertTrue(reopened.update_available)
        resolver2 = _resolver(v2, {k: v for k, v in v2_bodies.items()
                                   if v})
        return session, resolver2

    def test_vanished_middle_segment_lands_on_follower_with_note(self):
        session, resolver2 = self._open_flagged(
            {"ch-1": "a", "ch-2": "b", "ch-3": "c", "ch-4": "d"},
            {"ch-1": "a", "ch-2": "b", "ch-4": "d"}, at="ch-3")
        view = self.svc.switch_revision(session.session_id, resolver2)
        self.assertEqual(view.segment_id, "ch-4")
        self.assertEqual(view.body, "d")
        self.assertIn("Gamma", view.note)
        self.assertIn("not in the newer revision", view.note)
        self.assertIn("Delta", view.note)

    def test_vanished_last_segment_lands_on_predecessor(self):
        session, resolver2 = self._open_flagged(
            {"ch-1": "a", "ch-2": "b", "ch-3": "c"},
            {"ch-1": "a", "ch-2": "b"}, at="ch-3")
        view = self.svc.switch_revision(session.session_id, resolver2)
        self.assertEqual(view.segment_id, "ch-2")
        self.assertIn("not in the newer revision", view.note)

    def test_surviving_segment_switch_notes_the_switch(self):
        session, resolver2 = self._open_flagged(
            {"ch-1": "a", "ch-2": "b"}, {"ch-1": "a", "ch-2": "B2"},
            at="ch-2")
        view = self.svc.switch_revision(session.session_id, resolver2)
        self.assertEqual(view.segment_id, "ch-2")
        self.assertEqual(view.body, "B2")
        self.assertIn("switched to the newer revision", view.note)
        self.assertFalse(view.update_available)

    def test_switch_via_u_verb_actuates_the_same_move(self):
        session, resolver2 = self._open_flagged(
            {"ch-1": "a", "ch-2": "b", "ch-3": "c"},
            {"ch-1": "a", "ch-3": "c"}, at="ch-2")
        token = self.svc.view(session.session_id, resolver2).token
        result = self.svc.act(token, "u", resolver2)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.view.segment_id, "ch-3")
        self.assertIn("not in the newer revision", result.view.note)

    def test_vanished_into_empty_revision_stays_honest(self):
        session, resolver2 = self._open_flagged(
            {"ch-1": "a"}, {"ch-1": None}, at="ch-1")
        view = self.svc.switch_revision(session.session_id, resolver2)
        self.assertNotEqual(view.status, "ok")
        self.assertIn("not in the newer revision", view.note)


class RestartTests(unittest.TestCase):
    """§8 lifecycle: restart mid-read over the on-disk store."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.store = Path(self.tmp) / "reader_sessions.json"
        self.bodies = {"ch-1": "alpha " * 900, "ch-2": "beta " * 100}
        self.manifest = _manifest("doc:tale", self.bodies)

    def test_restart_resumes_same_page_same_revision(self):
        svc = ReaderService(self.store)
        session = svc.open("reader-a", self.manifest)
        resolver = _resolver(self.manifest, self.bodies)
        svc.goto(session.session_id, resolver, segment_id="ch-1")
        page2 = svc.navigate(session.session_id, resolver, "next")
        self.assertEqual(page2.page_index, 1)
        revived = ReaderService(self.store)
        back = revived.session_for("reader-a", "doc:tale")
        self.assertIsNotNone(back)
        view = revived.view(back.session_id, resolver)
        self.assertEqual(view.body, page2.body)
        self.assertEqual(view.page_index, 1)
        self.assertEqual(view.revision_ref, page2.revision_ref)
        self.assertEqual(back.segment_id, "ch-1")

    def test_update_flag_and_pin_survive_restart(self):
        svc = ReaderService(self.store)
        svc.open("reader-a", self.manifest)
        changed = dict(self.bodies, **{"ch-1": "rewritten " * 900})
        v2 = _manifest("doc:tale", changed)
        self.assertTrue(svc.open("reader-a", v2).update_available)
        revived = ReaderService(self.store)
        back = revived.session_for("reader-a", "doc:tale")
        self.assertTrue(back.update_available)
        resolver = _resolver(self.manifest, self.bodies)
        view = revived.view(back.session_id, resolver)
        self.assertTrue(view.update_available)
        rendered = revived.render_text(view)
        self.assertIn("newer revision of this document is available",
                      rendered)


class CorruptStoreTests(unittest.TestCase):
    """§8 lifecycle: a corrupt session store is contained + named."""

    def setUp(self) -> None:
        failures.clear_failures()
        self.tmp = tempfile.mkdtemp()
        self.store = Path(self.tmp) / "reader_sessions.json"

    def tearDown(self) -> None:
        failures.clear_failures()

    def test_garbage_store_restarts_empty_and_names_the_failure(self):
        self.store.write_text("this is not json {{{", encoding="utf-8")
        svc = ReaderService(self.store)
        self.assertIsNone(svc.session_for("reader-a", "doc:tale"))
        sites = [f.site for f in failures.recent_failures()]
        self.assertIn("reader_service.load_store", sites)
        # and reading itself still works on the fresh store
        manifest = _manifest("doc:tale", {"ch-1": "a"})
        session = svc.open("reader-a", manifest)
        self.assertEqual(session.segment_id, "ch-1")

    def test_one_bad_row_is_skipped_named_and_others_load(self):
        svc = ReaderService(self.store)
        manifest = _manifest("doc:tale", {"ch-1": "a"})
        svc.open("reader-a", manifest)
        payload = json.loads(self.store.read_text(encoding="utf-8"))
        payload["sessions"].append(
            {"session_id": "broken-row", "offset": "not-a-number"})
        self.store.write_text(json.dumps(payload), encoding="utf-8")
        failures.clear_failures()
        revived = ReaderService(self.store)
        self.assertIsNotNone(revived.session_for("reader-a", "doc:tale"))
        sites = [f.site for f in failures.recent_failures()]
        self.assertIn("reader_service.load_session_row", sites)


class RemovalTests(unittest.TestCase):
    """§8 lifecycle: content removed under the reader."""

    def setUp(self) -> None:
        self.svc = ReaderService()
        self.bodies = {"ch-1": "first body", "ch-2": "second body"}
        self.manifest = _manifest("doc:tale", self.bodies)
        self.contents = {seg.content_ref: self.bodies[seg.segment_id]
                         for seg in self.manifest.segments}
        self.resolver = content_resolver_from_map(self.contents)

    def test_source_removal_renders_unavailable_and_session_recovers(self):
        session = self.svc.open("reader-a", self.manifest)
        view = self.svc.view(session.session_id, self.resolver)
        self.assertEqual(view.status, "ok")
        # the source loses the chapter under the reader
        self.contents[self.manifest.segments[0].content_ref] = None
        gone = self.svc.view(session.session_id, self.resolver)
        self.assertEqual(gone.status, "unavailable")
        self.assertIn("no longer holds it", gone.note)
        rendered = self.svc.render_text(gone)
        self.assertIn("contents:", rendered)  # the recovery path
        # the session is not corrupted: another part still reads
        other = self.svc.goto(session.session_id, self.resolver,
                              segment_id="ch-2")
        self.assertEqual(other.status, "ok")
        self.assertEqual(other.body, "second body")

    def test_cursor_on_a_segment_the_pin_never_held(self):
        session = self.svc.open("reader-a", self.manifest)
        session.segment_id = "ch-99"  # a place the revision lacks
        view = self.svc.view(session.session_id, self.resolver)
        self.assertEqual(view.status, "unavailable")
        self.assertIn("may have been removed", view.note)
        back = self.svc.goto(session.session_id, self.resolver,
                             segment_id="ch-1")
        self.assertEqual(back.status, "ok")


if __name__ == "__main__":
    unittest.main()
