"""Doc 16 Pass 6 — the §8 quality set: offline / visual / RTL /
accessibility, scoped to what this box's surfaces can prove.

* OFFLINE — the reader, the edition renderer and the story export
  all run with the network seams stubbed dead: the P1/P5 no-fetch
  discipline (manifest derivation is a pure local read; exports
  build from the local cache only) proven under a dead socket,
  not just by code reading.
* VISUAL — the edition carries its design as data: one CSS
  custom-property token set keyed off ``html[data-theme]``, the
  treatment class per source kind, the completeness line and the
  brand per policy (extending P3's pins where they were thin).
* RTL — a manifest whose PresentationHints say ``direction="rtl"``
  renders the edition with ``dir="rtl"`` (P3's renderer consumes
  the hint — verified here), while the chat reader serves the
  segment bytes unchanged: direction is presentation, never a
  content mutation, so an LTR and an RTL manifest of the same
  content read byte-identically.
* ACCESSIBILITY — reader renders are plain text where every
  button's payload also appears as a typed command (no meaning
  carried by position or colour alone; contents rows are
  numbered), and the edition uses semantic landmarks + a heading
  hierarchy (structure asserted, not vibes).

Live visual proof on a real phone/browser remains the owner's
EC2 pull, as with every pass of this programme.
"""
from __future__ import annotations

import os
import socket
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.books.manifest import manifest_from_story, story_content_resolver
from nomorals.books.reader import FollowedStory, StoryReader
from nomorals.core.edition import render_edition_from_map
from nomorals.core.longdoc import (
    Completeness,
    PresentationHints,
    SegmentSpec,
    build_manifest,
)
from nomorals.social.chat.output_router import OutputRouter
from nomorals.social.chat.reader_service import (
    ReaderService,
    content_resolver_from_map,
)


@contextmanager
def _dead_network():
    """Every socket seam refuses: any fetch attempt fails loudly."""
    def _boom(*args, **kwargs):
        raise OSError("network is dead inside this test")

    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(
            socket.socket, "connect", _boom))
        stack.enter_context(mock.patch.object(
            socket.socket, "connect_ex", _boom))
        stack.enter_context(mock.patch("socket.create_connection", _boom))
        stack.enter_context(mock.patch("socket.getaddrinfo", _boom))
        yield


def _ctx():
    tmp = tempfile.mkdtemp()
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(db=None, settings=settings, extras={}), tmp


def _manifest(source_kind="text", direction="auto", locale="",
              bodies=("Alpha body.", "Beta body.")):
    specs = [
        SegmentSpec(segment_id=f"ch-{i + 1}", kind="chapter",
                    title=f"Chapter {i + 1}", content=body,
                    content_ref=f"{source_kind}:x#ch-{i + 1}")
        for i, body in enumerate(bodies)
    ]
    return build_manifest(
        document_id=f"{source_kind}:x", source_kind=source_kind,
        title="The Work", segment_specs=specs,
        completeness=Completeness(status="complete",
                                  declared_total=len(specs),
                                  present_count=len(specs)),
        presentation=PresentationHints(direction=direction,
                                       locale=locale))


def _contents_of(manifest):
    return {seg.content_ref: f"body of {seg.segment_id}"
            for seg in manifest.segments}


class OfflineTests(unittest.TestCase):
    """Reading, rendering and exporting never touch the network."""

    def setUp(self) -> None:
        self.ctx, self.tmp = _ctx()
        self.reader = StoryReader(self.ctx)
        story = FollowedStory(
            slug="offline-tale", title="The Offline Tale",
            url="https://fake/offline-tale", source="fake",
            author="A. Teller", status="Ongoing", total_chapters=2,
            ingest_total=2)
        self.reader._save(story)
        from nomorals.books.sources import Chapter
        for n, text in ((1, "First held chapter body."),
                        (2, "Second held chapter body.")):
            self.reader._store_chapter("offline-tale", Chapter(
                number=n, title=f"Chapter {n}",
                url=f"https://fake/offline-tale/{n}",
                paragraphs=[text], source="fake"))

    def test_story_read_and_export_under_dead_network(self):
        with _dead_network():
            manifest = manifest_from_story(self.reader, "offline-tale")
            svc = ReaderService()
            session = svc.open("reader-a", manifest)
            view = svc.view(session.session_id,
                            story_content_resolver(self.reader))
            self.assertEqual(view.status, "ok")
            self.assertIn("First held chapter body.", view.body)
            result = self.reader.export_story("offline-tale", "html")
            html = Path(result["path"]).read_text(encoding="utf-8")
            self.assertIn("Second held chapter body.", html)
            self.assertIn("Complete edition", html)
            self.assertEqual(result["completeness"], "complete")

    def test_router_and_edition_under_dead_network(self):
        text = ("Research findings\n\n"
                + "The offline route still reads end to end. " * 300)
        with _dead_network():
            router = OutputRouter(Path(self.tmp) / "longdoc")
            plan = router.plan(text, platform="telegram",
                               reader_key="chat:telegram:1")
            self.assertIn(plan.decision.kind, ("reader", "reader+edition"))
            svc = router.reader
            resolver = router.resolver()
            view = svc.view(svc.session_for(
                "chat:telegram:1", plan.decision.document_id).session_id,
                resolver)
            bodies = [view.body]
            while view.has_next_page or view.has_next_segment:
                result = svc.act(view.token, "n", resolver)
                view = result.view
                bodies.append(view.body)
            self.assertEqual("".join(bodies), text)
            html = router.edition_html(plan.decision.document_id)
            self.assertIn("offline route still reads", html)


class VisualTests(unittest.TestCase):
    """The edition's design is present as data, per kind + theme."""

    def test_theme_token_set_and_treatment_per_kind(self):
        for kind, treatment in (("forge-book", "book"),
                                ("story", "book"),
                                ("document", "technical"),
                                ("present-doc", "report"),
                                ("text", "report")):
            manifest = _manifest(source_kind=kind)
            html = render_edition_from_map(manifest, _contents_of(manifest))
            self.assertIn(f'<body class="treatment-{treatment}">', html)
            self.assertIn(":root{--bg:", html)
            self.assertIn('html[data-theme="sepia"]', html)
            self.assertIn('data-theme="light"', html)
            self.assertIn("Complete edition", html)

    def test_sepia_theme_is_baked_in(self):
        manifest = _manifest()
        html = render_edition_from_map(manifest, _contents_of(manifest),
                                       theme="sepia")
        self.assertIn('data-theme="sepia"', html)
        self.assertIn("body of ch-1", html)

    def test_brand_policy_off_removes_the_mark(self):
        manifest = _manifest()
        branded = render_edition_from_map(manifest, _contents_of(manifest))
        self.assertIn("Powered by peacethefirst", branded)
        clean = render_edition_from_map(
            manifest, _contents_of(manifest), brand_policy="off")
        self.assertNotIn("peacethefirst", clean.lower())


class RtlTests(unittest.TestCase):
    """Direction is presentation: edition carries it, bytes don't move."""

    def test_edition_renders_dir_rtl(self):
        manifest = _manifest(direction="rtl", locale="ar")
        html = render_edition_from_map(manifest, _contents_of(manifest))
        self.assertIn('dir="rtl"', html)
        self.assertIn('lang="ar"', html)

    def test_reader_bytes_identical_across_directions(self):
        bodies = ("Alpha body — unchanged.", "Beta body — unchanged.")
        renders = {}
        for direction in ("ltr", "rtl"):
            manifest = _manifest(direction=direction, bodies=bodies)
            svc = ReaderService()
            session = svc.open("reader-a", manifest)
            resolver = content_resolver_from_map(
                {seg.content_ref: bodies[i]
                 for i, seg in enumerate(manifest.segments)})
            view = svc.view(session.session_id, resolver)
            renders[direction] = (view.body, svc.render_text(view))
        self.assertEqual(renders["ltr"][0], bodies[0])
        self.assertEqual(renders["ltr"], renders["rtl"])


class AccessibilityTests(unittest.TestCase):
    """Labelled controls everywhere; semantic structure in editions."""

    def setUp(self) -> None:
        self.bodies = {"ch-1": "alpha " * 900, "ch-2": "beta"}
        specs = [
            SegmentSpec(segment_id=sid, kind="chapter", title=sid,
                        content=body, content_ref=f"text:x#{sid}")
            for sid, body in self.bodies.items()
        ]
        self.manifest = build_manifest(
            document_id="text:x", source_kind="text", title="Tale",
            segment_specs=specs,
            completeness=Completeness(status="complete",
                                      declared_total=2, present_count=2))
        self.svc = ReaderService()
        self.resolver = content_resolver_from_map(
            {seg.content_ref: self.bodies[seg.segment_id]
             for seg in self.manifest.segments})

    def test_every_button_has_a_typed_text_fallback(self):
        session = self.svc.open("reader-a", self.manifest)
        view = self.svc.view(session.session_id, self.resolver)
        rendered = self.svc.render_text(view)
        buttons = [pair for row in view.buttons("read go")
                   for pair in row]
        self.assertTrue(buttons)
        for label, payload in buttons:
            self.assertTrue(label.strip())
            self.assertIn(payload, rendered)

    def test_contents_rows_are_numbered_not_position_only(self):
        view = self.svc.contents_view(
            self.manifest, current_segment_id="ch-1")
        text = self.svc.render_contents_text(view)
        for row in view.rows:
            self.assertIn(f"{row.ordinal}. ", text)

    def test_edition_landmarks_and_heading_hierarchy(self):
        manifest = _manifest(source_kind="forge-book")
        html = render_edition_from_map(manifest, _contents_of(manifest))
        for landmark in ('<header class="cover">', "<h1",
                         '<nav class="toc" aria-label="Edition contents">',
                         "<main>", '<footer class="edition-footer">'):
            self.assertIn(landmark, html)
        self.assertIn('<section class="segment kind-chapter"', html)
        self.assertIn("<h2>Chapter 1</h2>", html)


if __name__ == "__main__":
    unittest.main()
