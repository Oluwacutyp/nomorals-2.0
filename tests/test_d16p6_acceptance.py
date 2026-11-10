"""Doc 16 Pass 6 — the acceptance battery: the whole programme,
end to end, over its real seams (doc §8).

Scenario A — a long output is routed once (P4) and every promised
surface reconciles to the same bytes: the reader actions page
through the FULL text (P2 discipline), the raw download is the
original byte for byte (P4/P5), the HTML edition carries the tail
of the text (P3), and the routing decision is on the journal (P4).

Scenario B — a followed story read on a no-pager surface (the
reader-service path /books part rides): a new chapter is ingested
→ the open session adopts the growth silently (the segment under
the reader is byte-identical — P2's pinned rule) and the chapter
is reachable; a CORRECTION to the chapter under the reader → the
session pins its revision, the next render announces the newer
revision, the explicit switch adopts it; exports (P5) track each
revision and their completeness lines stay true throughout.
Named honestly: the story resolver serves the live cache, so a
correction is visible to a resolver read immediately — the pin
this scenario asserts is the contract-level one (the session's
pinned manifest/revision does not move until the switch), which
is exactly what P2's map-resolver pin proves at byte level.

Scenario C — a documents-engine Document edited between editions:
the revision counter moves, both derivatives print their own
revision, and the P1 RevisionRef of the first edition reads
provably NOT current against the edited source (staleness is
detectable, never silent).
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.books.manifest import manifest_from_story, story_content_resolver
from nomorals.books.reader import FollowedStory, StoryReader
from nomorals.books.sources import Chapter
from nomorals.documents.convert import to_html
from nomorals.documents.manifest import (
    document_revision_ref,
    manifest_from_document,
)
from nomorals.documents.model import Document, Section
from nomorals.social.chat.output_router import (
    ROUTE_READER_EDITION,
    OutputRouter,
)
from nomorals.social.chat.reader_service import ReaderService

KEY = "chat:telegram:42"

PROSE = (
    "Quarterly research digest\n\n"
    + ("Finding: the signal held across every session, with volume "
       "confirming the move and no sign of the usual exhaustion "
       "pattern near the highs. " * 12 + "\n\n")
    + ("Risk note: the invalidation sits below the prior swing "
       "low; a close through it ends the thesis and the plan "
       "reverts to observation only. " * 12 + "\n\n")
)
LONG = PROSE * 14  # ~52k chars — past the reader+edition band
TAIL = "reverts to observation only."


def _ctx():
    tmp = tempfile.mkdtemp()
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(db=None, settings=settings, extras={}), tmp


class RoutedOutputScenarioTests(unittest.TestCase):
    """Scenario A: route → read fully → download → edition."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.router = OutputRouter(Path(self.tmp) / "longdoc")

    def test_long_output_reconciles_across_every_surface(self):
        plan = self.router.plan(LONG, platform="telegram",
                                reader_key=KEY)
        self.assertEqual(plan.decision.kind, ROUTE_READER_EDITION)
        svc = self.router.reader
        resolver = self.router.resolver()
        session = svc.session_for(KEY, plan.decision.document_id)
        self.assertIsNotNone(session)
        # read the whole thing through the token actions alone
        view = svc.view(session.session_id, resolver)
        bodies = [view.body]
        while view.has_next_page or view.has_next_segment:
            result = svc.act(view.token, "n", resolver)
            self.assertEqual(result.status, "ok")
            view = result.view
            bodies.append(view.body)
        self.assertEqual("".join(bodies), LONG)
        # the raw download is the original, byte for byte
        path, _name = self.router.file_for(
            plan.decision.document_id, "txt")
        self.assertEqual(path.read_bytes(), LONG.encode("utf-8"))
        # the branded edition carries the tail of the text
        html = self.router.edition_html(plan.decision.document_id)
        self.assertIn(TAIL, html)
        self.assertIn("Powered by peacethefirst", html)
        # and the decision itself is journaled with its reason
        rows = self.router.recent(KEY)
        self.assertTrue(rows)
        self.assertEqual(rows[0]["route"], ROUTE_READER_EDITION)
        self.assertEqual(rows[0]["document_id"],
                         plan.decision.document_id)


CH1 = "Opening chapter. CH1-ORIGINAL-SENTINEL"
CH1_FIXED = "Opening chapter, corrected. CH1-CORRECTED-SENTINEL"
CH2 = "Second chapter. CH2-SENTINEL"
CH3 = "Third chapter, newly arrived. CH3-SENTINEL"


class StoryLifecycleScenarioTests(unittest.TestCase):
    """Scenario B: ingest → growth → correction → switch → export."""

    def setUp(self) -> None:
        self.ctx, self.tmp = _ctx()
        self.reader = StoryReader(self.ctx)
        self.slug = "live-tale"
        story = FollowedStory(
            slug=self.slug, title="The Live Tale",
            url="https://fake/live-tale", source="fake",
            author="A. Teller", status="Ongoing", total_chapters=2,
            ingest_total=2)
        self.reader._save(story)
        self._plant(1, CH1)
        self._plant(2, CH2)
        self.svc = ReaderService(Path(self.tmp) / "reader_sessions.json")
        self.resolver = story_content_resolver(self.reader)
        self.key = "chat:whatsapp:9"  # a no-pager surface's reader

    def _plant(self, number: int, text: str) -> None:
        self.reader._store_chapter(self.slug, Chapter(
            number=number, title=f"Chapter {number}",
            url=f"https://fake/live-tale/{number}",
            paragraphs=[text], source="fake"))

    def _held_text(self, number: int) -> str:
        return self.reader._cached_chapter(self.slug, number).text

    def test_ingest_correction_switch_export(self):
        manifest = manifest_from_story(self.reader, self.slug)
        self.assertEqual(manifest.completeness.status, "complete")
        self.assertEqual(manifest.completeness.present_count, 2)
        session = self.svc.open(self.key, manifest)
        view = self.svc.view(session.session_id, self.resolver)
        self.assertEqual(view.body, self._held_text(1))

        # a new chapter arrives (ingest writes the cache + totals)
        self._plant(3, CH3)
        story = self.reader._load(self.slug)
        story.total_chapters = 3
        story.ingest_total = 3
        self.reader._save(story)
        grown = manifest_from_story(self.reader, self.slug)
        reopened = self.svc.open(self.key, grown)
        # the chapter under the reader is byte-identical: silent
        # adoption, no flag, no yank — but the growth is reachable
        self.assertFalse(reopened.update_available)
        contents = self.svc.contents_view(
            grown, current_segment_id=reopened.segment_id)
        self.assertEqual([r.present for r in contents.rows],
                         [True, True, True])
        third = self.svc.goto(session.session_id, self.resolver,
                              segment_id="ch-3")
        self.assertEqual(third.body, self._held_text(3))
        first_export = self.reader.export_story(self.slug, "html")
        html = Path(first_export["path"]).read_text(encoding="utf-8")
        self.assertIn("Complete edition — 3 of 3", html)
        self.assertIn("CH1-ORIGINAL-SENTINEL", html)
        self.assertIn("CH3-SENTINEL", html)

        # a correction lands on the chapter under the reader
        self.svc.goto(session.session_id, self.resolver,
                      segment_id="ch-1")
        self._plant(1, CH1_FIXED)
        corrected = manifest_from_story(self.reader, self.slug)
        self.assertNotEqual(corrected.revision_ref, grown.revision_ref)
        flagged = self.svc.open(self.key, corrected)
        self.assertTrue(flagged.update_available)
        # the pin holds at the contract level until the switch
        self.assertEqual(flagged.pinned().revision_ref,
                         grown.revision_ref)
        view = self.svc.view(session.session_id, self.resolver)
        self.assertTrue(view.update_available)
        rendered = self.svc.render_text(view)
        self.assertIn("newer revision of this document is available",
                      rendered)
        # the explicit switch adopts the corrected revision
        result = self.svc.act(view.token, "u", self.resolver)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.view.body, self._held_text(1))
        self.assertIn("CH1-CORRECTED-SENTINEL", result.view.body)
        self.assertFalse(result.view.update_available)
        self.assertEqual(
            self.svc.session_for(self.key, corrected.document_id)
            .pinned().revision_ref, corrected.revision_ref)
        # and the export now tracks the corrected revision
        second = self.reader.export_story(self.slug, "html")
        html = Path(second["path"]).read_text(encoding="utf-8")
        self.assertIn("CH1-CORRECTED-SENTINEL", html)
        self.assertNotIn("CH1-ORIGINAL-SENTINEL", html)
        self.assertIn("Complete edition — 3 of 3", html)
        self.assertNotEqual(second["revision_ref"],
                            first_export["revision_ref"])
        self.assertEqual(second["revision_ref"],
                         corrected.revision_ref)


class DocumentRevisionScenarioTests(unittest.TestCase):
    """Scenario C: an edit between editions is provably detectable."""

    def test_edit_moves_revision_and_stales_the_old_ref(self):
        doc = Document(title="Field Notes", sections=[
            Section(level=1, heading="One", text="original body")])
        doc.refresh_revision()  # seal genesis; the counter stays 1
        self.assertEqual(doc.revision, 1)
        ref_before = document_revision_ref(doc)
        manifest_before = manifest_from_document(doc)
        html_before = to_html(doc)
        self.assertIn("Revision 1", html_before)
        # the ref's currency is judged against the source's own
        # revision state (edition_ref: counter + exact content hash)
        state_before = doc.edition_ref()
        self.assertTrue(ref_before.is_current(
            document_id=state_before["document_id"],
            revision=state_before["revision"],
            content_hash=state_before["content_hash"]))

        doc.sections[0].text = "rewritten body"
        self.assertTrue(doc.refresh_revision())
        self.assertEqual(doc.revision, 2)
        manifest_after = manifest_from_document(doc)
        html_after = to_html(doc)
        self.assertIn("Revision 2", html_after)
        self.assertNotEqual(manifest_after.revision_ref,
                            manifest_before.revision_ref)
        # the first edition's ref no longer describes the source
        state_after = doc.edition_ref()
        self.assertFalse(ref_before.is_current(
            document_id=state_after["document_id"],
            revision=state_after["revision"],
            content_hash=state_after["content_hash"]))
        ref_after = document_revision_ref(doc)
        self.assertTrue(ref_after.is_current(
            document_id=state_after["document_id"],
            revision=state_after["revision"],
            content_hash=state_after["content_hash"]))


if __name__ == "__main__":
    unittest.main()
