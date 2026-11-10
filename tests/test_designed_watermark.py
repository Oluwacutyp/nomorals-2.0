"""Public-surface watermark pass — the designed Peacethefirst mark.

Pins:
(a) PDF: a resolved mark draws on EVERY page of BOTH render paths
    (legacy + structured), as page furniture — the watermark op
    block opens each page's content stream (drawn under the body),
    rotated, under an ExtGState alpha; the body text itself is
    untouched and still extracts.  Policy off / no mark → no brand
    anywhere in any page stream.  Custom text honoured; metadata
    stamps unchanged.
(b) Resolution: core.branding is the ONE home — default/custom/
    off/unknown semantics mirror the edition's historical ones, and
    the constants equal the chat typography canonical stamp.
(c) HTML editions: the watermark layer renders under the default
    policy (screen + print CSS), vanishes under off (with every
    other brand string, as the edition contract promises), and
    carries custom text when the policy is custom.
(d) Producers through their real paths: documents convert, books
    forge build (PDF + EPUB title page), reader story export
    (PDF default + off, EPUB title page), deliver_report,
    filesend create_file.
(e) Images: the overlay burns a corner mark; a public-classified
    send through the gateway AND through the expression adapter
    seam delivers a marked copy (original file untouched); a
    personal/unclassified send is byte-identical; maker cards
    carry the runtime's group=public / DM=personal classification
    on the artifact itself.

Fake data only; no network.  Images are synthesized with Pillow.
"""

from __future__ import annotations

import io
import os
import re
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from nomorals.core import branding, edition
from nomorals.core.branding import resolve_brand_mark
from nomorals.core.pdf import (
    PdfError,
    _collect_pages,
    _decompress,
    _parse_objects,
    _stream_bytes,
    read_pdf_text,
    render_pdf,
)
from nomorals.social.chat.typography import DEFAULT_STAMP_TEXT

BRAND = branding.BRAND_TEXT

_BODY = "\n\n".join(
    f"Watermark body paragraph {i} with a readable sentence in it."
    for i in range(140))


def _page_streams(data: bytes) -> list[bytes]:
    """Decompressed content stream per page, in document order."""
    objects = _parse_objects(data)
    pages = _collect_pages(objects, objects[1])
    streams = []
    for page in pages:
        m = re.search(rb"/Contents\s+(\d+)\s+0\s+R", page)
        assert m is not None
        payload = _stream_bytes(objects[int(m.group(1))])
        streams.append(_decompress(payload))
    return streams


def _assert_marked_every_page(case: unittest.TestCase,
                              data: bytes, text: str = BRAND,
                              ) -> list[bytes]:
    streams = _page_streams(data)
    case.assertGreaterEqual(len(streams), 2)
    for stream in streams:
        case.assertTrue(stream.startswith(b"q /GSW gs"),
                        "watermark block must open the page stream")
        case.assertEqual(stream.count(text.encode("latin-1")), 3,
                         "three tiled diagonal instances per page")
        case.assertIn(b"/F1 34 Tf", stream)
    case.assertIn(b"/ExtGState", data)
    case.assertIn(b"/ca 0.15", data)
    return streams


def _assert_unmarked(case: unittest.TestCase, data: bytes) -> None:
    case.assertNotIn(b"peacethefirst", data)
    case.assertNotIn(b"/GSW", data)
    for stream in _page_streams(data):
        case.assertNotIn(b"peacethefirst", stream)


class PdfWatermarkTests(unittest.TestCase):
    def test_legacy_path_every_page_marked(self) -> None:
        data = render_pdf(_BODY, title="Marked",
                          watermark=resolve_brand_mark())
        streams = _assert_marked_every_page(self, data)
        # rotated, not axis-aligned: the Tm carries non-zero b/c
        m = re.search(rb"([\d.]+) ([\d.]+) ([-\d.]+) ([\d.]+) "
                      rb"([\d.]+) ([\d.]+) Tm \(Powered", streams[0])
        self.assertIsNotNone(m)
        self.assertNotEqual(m.group(2), b"0")
        # the mark is furniture: it precedes the first body draw,
        # and the body still extracts whole
        body_at = streams[0].find(b"Watermark body paragraph 0")
        self.assertGreater(body_at, streams[0].find(b"peacethefirst"))
        self.assertIn("Watermark body paragraph 0",
                      read_pdf_text(data))

    def test_structured_path_every_page_marked(self) -> None:
        text = "# Book\n\n## Start\n\n" + _BODY
        data = render_pdf(text, title="Book", headings=True,
                          chapter_break=True, toc=True,
                          metadata={"title": "Book",
                                    "subject": "stamp-check"},
                          watermark=resolve_brand_mark())
        _assert_marked_every_page(self, data)
        # metadata stamps ride unchanged beside the watermark
        self.assertIn(b"stamp-check", data)
        self.assertIn(b"/Producer (nomorals core.pdf)", data)

    def test_no_mark_and_off_policy_are_clean(self) -> None:
        _assert_unmarked(self, render_pdf(_BODY, title="Plain"))
        self.assertIsNone(resolve_brand_mark("off"))
        _assert_unmarked(self, render_pdf(
            _BODY, title="Off",
            watermark=resolve_brand_mark("off")))

    def test_custom_text_and_string_coercion(self) -> None:
        data = render_pdf(_BODY, title="Custom",
                          watermark=resolve_brand_mark(
                              "custom", "ACME HOUSE"))
        streams = _assert_marked_every_page(self, data, "ACME HOUSE")
        for stream in streams:
            self.assertNotIn(b"peacethefirst", stream)
        coerced = render_pdf(_BODY, title="Str", watermark="HOUSE")
        _assert_marked_every_page(self, coerced, "HOUSE")
        with self.assertRaises(PdfError):
            render_pdf(_BODY, title="Bad", watermark=42)


class BrandResolutionTests(unittest.TestCase):
    def test_one_home_one_semantics(self) -> None:
        self.assertEqual(branding.BRAND_TEXT, DEFAULT_STAMP_TEXT)
        self.assertEqual(edition.BRAND_TEXT, branding.BRAND_TEXT)
        self.assertEqual(resolve_brand_mark().text, BRAND)
        self.assertEqual(resolve_brand_mark("default").text, BRAND)
        self.assertEqual(resolve_brand_mark("nonsense").text, BRAND)
        self.assertEqual(resolve_brand_mark("custom", "X").text, "X")
        # custom with no text falls back to the house mark
        self.assertEqual(resolve_brand_mark("custom", "").text, BRAND)
        self.assertIsNone(resolve_brand_mark(" OFF "))


class EditionWatermarkTests(unittest.TestCase):
    def _html(self, **kw) -> str:
        from nomorals.social.chat.manifest import manifest_from_text, text_content_resolver

        md = "# Title\n\nA short body for the edition."
        manifest = manifest_from_text(md, title="T")
        return edition.render_edition(
            manifest, text_content_resolver(md), **kw)

    def test_default_policy_renders_the_layer(self) -> None:
        html = self._html()
        self.assertIn('class="watermark"', html)
        self.assertIn(f"<span>{BRAND}</span>", html)
        self.assertIn(".watermark{position:fixed", html)
        # retained chrome: cover line + footer + print mark
        self.assertIn('class="brand-line"', html)
        self.assertIn('class="print-mark"', html)

    def test_off_policy_removes_every_brand(self) -> None:
        html = self._html(brand_policy="off")
        self.assertNotIn('class="watermark"', html)
        self.assertNotIn("peacethefirst", html)

    def test_custom_text_in_the_layer(self) -> None:
        html = self._html(brand_policy="custom",
                          brand_text="ACME HOUSE")
        self.assertIn("<span>ACME HOUSE</span>", html)

    def test_chromeless_body_carries_no_watermark(self) -> None:
        self.assertNotIn('class="watermark"',
                         self._html(chrome=False))


# ── producers through their real paths ──────────────────────────────


class ProducerPdfTests(unittest.TestCase):
    def test_documents_convert_pdf(self) -> None:
        from nomorals.documents.convert import to_pdf
        from nomorals.documents.model import Document
        from nomorals.documents.model import Section as DocSection

        doc = Document(
            id="doc-wm", format="md", title="Doc Title", author="Au",
            source="file.md",
            sections=[DocSection(level=1, heading="H1",
                                 text=_BODY)])
        _assert_marked_every_page(self, to_pdf(doc))

    def test_filesend_create_file_pdf(self) -> None:
        from nomorals.tools.filesend import create_file

        tmp = tempfile.mkdtemp()
        ctx = SimpleNamespace(
            settings=SimpleNamespace(workspace_dir=tmp), extras={})
        out = create_file(ctx, "wm-report", _BODY, format="pdf",
                          title="WM Report")
        data = Path(out["path"]).read_bytes()
        _assert_marked_every_page(self, data)

    def test_deliver_report_pdf(self) -> None:
        from nomorals.tools import deliver_report as DR

        ctx = SimpleNamespace(
            settings=SimpleNamespace(
                workspace_dir=tempfile.mkdtemp()),
            extras={}, router=None)
        bundle = DR.generate_report(
            ctx, "Q3 markets",
            [("Overview", _BODY), ("Risks", "watch the curve")],
            generated="2026-10-01 12:00")
        data = Path(bundle.pdf_path).read_bytes()
        _assert_marked_every_page(self, data)


def _forge_book(tmp: str):
    from nomorals.books.forge import BookForge
    from nomorals.books.model import STATUS_WRITTEN, Book
    from nomorals.books.model import Chapter as BookChapter

    ctx = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=tmp), extras={})
    forge = BookForge(ctx)
    book = Book(
        topic="T", slug="wm-book", title="WM Book", author="Au",
        chapters=[
            BookChapter(number=1, title="One", text=_BODY,
                        status=STATUS_WRITTEN),
            BookChapter(number=2, title="Two", text=_BODY,
                        status=STATUS_WRITTEN),
        ])
    forge.save(book)
    return forge, book


class BookProducerTests(unittest.TestCase):
    def test_forge_build_pdf_marked(self) -> None:
        tmp = tempfile.mkdtemp()
        forge, _book = _forge_book(tmp)
        result = forge.build("wm-book", formats=["pdf"])
        data = Path(result["pdf"]).read_bytes()
        _assert_marked_every_page(self, data)

    def test_forge_epub_title_page_carries_the_brand(self) -> None:
        from nomorals.books.forge import epub_bytes

        _forge, book = _forge_book(tempfile.mkdtemp())
        data = epub_bytes(book)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            title_pages = [
                zf.read(name).decode("utf-8", "replace")
                for name in zf.namelist()
                if name.endswith(".xhtml")
                and "book-title" in zf.read(name).decode(
                    "utf-8", "replace")]
        self.assertTrue(title_pages, "a title page must exist")
        self.assertIn(BRAND, title_pages[0])


def _plant_story(reader, slug: str = "wm-tale"):
    from nomorals.books.reader import FollowedStory
    from nomorals.books.sources import Chapter

    story = FollowedStory(
        slug=slug, title="The WM Tale", url="https://fake/wm",
        source="fake", author="A. Teller", synopsis="s",
        genres=["fantasy"], status="Ongoing", total_chapters=2,
        ingest_total=2)
    reader._save(story)
    for n in (1, 2):
        reader._store_chapter(slug, Chapter(
            number=n, title=f"Chapter {n}",
            url=f"https://fake/wm/{n}",
            paragraphs=_BODY.split("\n\n"), source="fake"))
    return story


class ReaderProducerTests(unittest.TestCase):
    def _reader(self):
        from nomorals.books.reader import StoryReader

        ctx = SimpleNamespace(
            db=None,
            settings=SimpleNamespace(
                workspace_dir=tempfile.mkdtemp()),
            extras={})
        reader = StoryReader(ctx)
        _plant_story(reader)
        return reader

    def test_export_pdf_marked_by_default(self) -> None:
        reader = self._reader()
        out = reader.export_story("wm-tale", "pdf")
        _assert_marked_every_page(
            self, Path(out["path"]).read_bytes())

    def test_export_pdf_off_policy_is_clean(self) -> None:
        reader = self._reader()
        out = reader.export_story("wm-tale", "pdf",
                                  brand_policy="off")
        data = Path(out["path"]).read_bytes()
        self.assertNotIn(b"peacethefirst", data)
        self.assertNotIn(b"/GSW", data)

    def test_export_epub_title_page_brand_and_off(self) -> None:
        reader = self._reader()
        out = reader.export_story("wm-tale", "epub")
        with zipfile.ZipFile(out["path"]) as zf:
            pages = {name: zf.read(name).decode("utf-8", "replace")
                     for name in zf.namelist()
                     if name.endswith(".xhtml")}
        title = next(v for v in pages.values()
                     if "book-title" in v)
        self.assertIn(BRAND, title)
        out_off = reader.export_story("wm-tale", "epub",
                                      brand_policy="off")
        with zipfile.ZipFile(out_off["path"]) as zf:
            pages_off = {name: zf.read(name).decode("utf-8",
                                                   "replace")
                         for name in zf.namelist()
                         if name.endswith(".xhtml")}
        title_off = next(v for v in pages_off.values()
                         if "book-title" in v)
        self.assertNotIn(BRAND, title_off)


# ── images ──────────────────────────────────────────────────────────


def _png_bytes(width: int = 400, height: int = 300,
               color=(70, 110, 160)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def _corner_and_center(data: bytes):
    from PIL import Image

    img = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = img.size
    corner = [img.getpixel((x, y))
              for x in range(w - 130, w - 5, 12)
              for y in range(h - 40, h - 5, 6)]
    return corner, img.getpixel((w // 2, h // 2))


class ImageOverlayTests(unittest.TestCase):
    def test_burn_marks_the_corner_only(self) -> None:
        from nomorals.social.chat.brandmark import burn_brand_mark

        raw = _png_bytes()
        marked = burn_brand_mark(raw)
        corner_raw, center_raw = _corner_and_center(raw)
        corner_new, center_new = _corner_and_center(marked)
        self.assertNotEqual(corner_raw, corner_new)
        self.assertEqual(center_raw, center_new)

    def test_burn_rejects_non_images(self) -> None:
        from nomorals.social.chat.brandmark import BrandMarkError, burn_brand_mark

        with self.assertRaises(BrandMarkError):
            burn_brand_mark(b"this is not an image")


def _adapter():
    from nomorals.social.chat.base import ChatAdapter

    class Adapter(ChatAdapter):
        name = "telegram-bot"

        def __init__(self) -> None:
            super().__init__()
            self.media: list = []

        def run(self, handler):  # pragma: no cover - unused
            pass

        def send(self, chat, text, **kw):
            from nomorals.social.chat.base import SendResult

            return SendResult(ok=True, platform=self.name,
                              message_id="t1")

        def send_media(self, chat, media, *, caption=""):
            from nomorals.social.chat.base import SendResult

            self.media.append((media.path, caption))
            return SendResult(ok=True, platform=self.name,
                              message_id="m1")

    return Adapter()


class GatewayAudienceTests(unittest.TestCase):
    def _send(self, tmp: str, **kw):
        from nomorals.social.chat.base import ChatRef
        from nomorals.social.chat.gateway import ChatGateway

        path = os.path.join(tmp, "pic.png")
        Path(path).write_bytes(_png_bytes())
        adapter = _adapter()
        gw = ChatGateway({"telegram-bot": adapter})
        chat = ChatRef(platform="telegram-bot", chat_id="111")
        result = gw.send_file("telegram-bot", chat, path,
                              kind="image", **kw)
        self.assertTrue(result.ok)
        return path, adapter

    def test_public_send_delivers_a_marked_copy(self) -> None:
        tmp = tempfile.mkdtemp()
        original, adapter = self._send(tmp, audience="public")
        sent_path = adapter.media[0][0]
        self.assertNotEqual(sent_path, original)
        corner_raw, _ = _corner_and_center(Path(original).read_bytes())
        corner_new, _ = _corner_and_center(Path(sent_path).read_bytes())
        self.assertNotEqual(corner_raw, corner_new)
        # the caller's own file is never mutated
        self.assertEqual(Path(original).read_bytes(), _png_bytes())

    def test_unclassified_and_personal_send_the_original(self) -> None:
        tmp = tempfile.mkdtemp()
        for kw in ({}, {"audience": "personal"}):
            original, adapter = self._send(tmp, **kw)
            self.assertEqual(adapter.media[0][0], original)
            self.assertEqual(Path(original).read_bytes(),
                             _png_bytes())


class ExpressionAudienceTests(unittest.TestCase):
    def _host(self, adapter):
        from nomorals.agents.partner.runtime_expression import RuntimeExpressionMixin

        class Host(RuntimeExpressionMixin):
            def _group_adapter(self, platform):
                return adapter

        return Host()

    def _message(self):
        return SimpleNamespace(
            chat=SimpleNamespace(platform="telegram-bot"))

    def test_adapter_path_burns_public_refs_only(self) -> None:
        from nomorals.social.chat.base import MediaRef

        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "card.png")
        Path(path).write_bytes(_png_bytes())
        adapter = _adapter()
        host = self._host(adapter)
        public = MediaRef(path=path, kind="image",
                          audience="public")
        result = host._expression_send(self._message(), public)
        self.assertTrue(result.ok)
        sent_path = adapter.media[0][0]
        self.assertNotEqual(sent_path, path)
        corner_new, _ = _corner_and_center(
            Path(sent_path).read_bytes())
        corner_raw, _ = _corner_and_center(_png_bytes())
        self.assertNotEqual(corner_raw, corner_new)
        # personal: the very same file goes out, byte-identical
        adapter2 = _adapter()
        host2 = self._host(adapter2)
        personal = MediaRef(path=path, kind="image",
                            audience="personal")
        result2 = host2._expression_send(self._message(), personal)
        self.assertTrue(result2.ok)
        self.assertEqual(adapter2.media[0][0], path)


class MakerClassificationTests(unittest.TestCase):
    def test_card_ref_carries_the_classification(self) -> None:
        from nomorals.social.chat.makers import MakerService

        captured: list = []

        def send(ref, caption):
            captured.append(ref)
            return SimpleNamespace(ok=True)

        svc = MakerService(media_dir=Path(tempfile.mkdtemp()))
        out = svc.card("qcard", "Ada | we ship at dawn",
                       audience="public", send=send)
        self.assertTrue(out.get("sent"))
        self.assertEqual(captured[0].audience, "public")
        # the bytes at rest are the unmarked render — the mark is
        # a property of the send seam, not of the stored card
        corner, _ = _corner_and_center(
            Path(captured[0].path).read_bytes())
        raw = svc.card("qcard", "Ada | we ship at dawn",
                       audience="personal", send=send)
        self.assertTrue(raw.get("sent"))
        self.assertEqual(captured[-1].audience, "personal")
        self.assertEqual(
            Path(captured[0].path).read_bytes(),
            Path(captured[-1].path).read_bytes())

    def test_runtime_classifies_group_public_dm_personal(self) -> None:
        from nomorals.agents.partner.runtime_makers import RuntimeMakersMixin
        from nomorals.social.chat.base import ChatKind
        from nomorals.social.chat.makers import MakerService

        captured: list = []

        class Host(RuntimeMakersMixin):
            def __init__(self, service):
                self._svc = service

            def _makers_service(self):
                return self._svc

            def _expression_send(self, message, ref, caption=""):
                captured.append(ref)
                return SimpleNamespace(ok=True)

        svc = MakerService(media_dir=Path(tempfile.mkdtemp()))
        host = Host(svc)
        for kind, want in ((ChatKind.GROUP, "public"),
                           (ChatKind.DM, "personal")):
            message = SimpleNamespace(
                chat=SimpleNamespace(kind=kind, platform="local",
                                     key="local:room1"),
                sender="Ada", meta={})
            out = host._makers_card("qcard", "Ada | hi", message)
            self.assertEqual(out, "")
            self.assertEqual(captured[-1].audience, want)


if __name__ == "__main__":
    unittest.main()
