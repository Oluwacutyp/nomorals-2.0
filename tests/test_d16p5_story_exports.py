"""Doc 16 Pass 5 — exports through the manifest: the completeness oracle.

The defect this pins shut (evaluation gap 3): a followed story could
be READ fully (P2) but never EXPORTED — there was no file a reader
could keep, and no derivative anywhere carried the manifest truth
(revision + declared completeness) that P1's contract promises.
Now a followed story exports as PDF / EPUB / HTML from its LOCALLY
HELD chapters only, built through its manifest, and every derivative
(story, forge book, document) records the revision_ref it was built
from plus the declared completeness.

The oracle, per format, for a planted story with gaps and a
>12000-char chapter:

- every held chapter's full text is present in the export (EPUB:
  the chapter XHTML holds every paragraph; HTML: the edition holds
  every paragraph; PDF: the repo's own read_pdf_text extraction
  holds every paragraph, in order — whole-string containment across
  a page boundary is impossible by construction of the shipped
  renderer, whose running footer ("— page N — <date>") interleaves
  at page breaks, so the check is per-paragraph plus head/tail
  sentinels, which localizes any loss to the paragraph);
- chapters 4 and 6 are DECLARED missing where they would sit —
  never invented, never silently dropped;
- the revision + completeness stamps are present (front matter in
  owner words; the full revision_ref token in machine metadata —
  OPF meta for EPUB, /Info keywords for PDF);
- a zero-held-chapters story refuses in words, for every format.

Fake data only — chapters are planted straight into the reader's
cache, so no network is touched and no fetch can sneak content in.
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.books.manifest import manifest_from_story
from nomorals.books.reader import FollowedStory, ReaderError, StoryReader
from nomorals.books.sources import Chapter
from nomorals.core.pdf import read_pdf_text


def _ctx():
    tmp = tempfile.mkdtemp()
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(db=None, settings=settings, extras={}), tmp


def _norm(text: str) -> str:
    return " ".join(text.split())


def _para(tag: str, chars: int) -> str:
    seed = f"{tag} unfolds slowly, sentence over sentence. "
    return (seed * (chars // len(seed) + 1))[:chars]


# held chapters of the planted story: 1, 2, 3, 5, 7 of a declared 7 —
# chapters 4 and 6 are the gaps.  Chapter 2 runs past the 12000-char
# read preview and ends in a tail sentinel, so any preview-sourced
# export fails this suite.
CH1 = "\n\n".join([_para("Opening", 900), "CH1-CLOSING-SENTINEL rests here."])
CH2 = "\n\n".join([_para("Second-a", 5200), _para("Second-b", 5400),
                   _para("Second-c", 2600),
                   "The deep tail. TAIL-BEYOND-PREVIEW"])
CH3 = "Third chapter, short and whole. CH3-SENTINEL"
CH5 = "Fifth chapter, after the first gap. CH5-SENTINEL"
CH7 = "Seventh chapter, the last one held. CH7-SENTINEL"
HELD = {1: CH1, 2: CH2, 3: CH3, 5: CH5, 7: CH7}
MISSING = (4, 6)


def _plant(reader: StoryReader, slug: str = "gap-tale",
           held: dict | None = None, total: int = 7,
           declared: bool = True) -> FollowedStory:
    story = FollowedStory(
        slug=slug, title="The Gap Tale", url="https://fake/gap-tale",
        source="fake", author="A. Teller", synopsis="A tale with holes.",
        genres=["fantasy"], status="Ongoing",
        total_chapters=total if declared else 0,
        ingest_total=total if declared else 0)
    reader._save(story)
    for n, text in (HELD if held is None else held).items():
        reader._store_chapter(slug, Chapter(
            number=n, title=f"Chapter {n}",
            url=f"https://fake/gap-tale/{n}",
            paragraphs=text.split("\n\n"), source="fake"))
    return story


def _xhtml_text(raw: str) -> str:
    import html as html_module
    text = re.sub(r"<[^>]+>", " ", raw)
    return _norm(html_module.unescape(text))


class StoryExportOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx, self.tmp = _ctx()
        self.reader = StoryReader(self.ctx)
        _plant(self.reader)
        self.manifest = manifest_from_story(self.reader, "gap-tale")

    # ── EPUB ─────────────────────────────────────────────────────────
    def test_epub_oracle(self):
        self.assertGreater(len(CH2), 12000)
        r = self.reader.export_story("gap-tale", "epub")
        self.assertEqual(r["present"], 5)
        self.assertEqual(r["declared_total"], 7)
        self.assertEqual(r["completeness"], "partial")
        z = zipfile.ZipFile(io.BytesIO(Path(r["path"]).read_bytes()))
        names = z.namelist()
        self.assertEqual(names[0], "mimetype")
        # every declared chapter has a page — held ones carry text,
        # missing ones carry only their declaration
        for n in range(1, 8):
            self.assertIn(f"OEBPS/ch{n:03d}.xhtml", names)
        for n, text in HELD.items():
            page = _xhtml_text(z.read(f"OEBPS/ch{n:03d}.xhtml").decode())
            for para in text.split("\n\n"):
                self.assertIn(_norm(para), page,
                              f"chapter {n} lost a paragraph in the EPUB")
        self.assertIn("TAIL-BEYOND-PREVIEW",
                      _xhtml_text(z.read("OEBPS/ch002.xhtml").decode()))
        for n in MISSING:
            page = _xhtml_text(z.read(f"OEBPS/ch{n:03d}.xhtml").decode())
            self.assertIn("declared missing", page)
            self.assertIn("never invented", page)
            self.assertNotIn("SENTINEL", page)
            self.assertLess(len(page), 400)
        # stamps: the completeness line in owner words on the title
        # page; the full revision_ref + counts in OPF metadata
        title_page = z.read("OEBPS/title.xhtml").decode()
        self.assertIn("Partial edition", title_page)
        self.assertIn("5 of 7 chapters present", title_page)
        self.assertIn("Missing: chapters 4, 6", title_page)
        opf = z.read("OEBPS/content.opf").decode()
        self.assertIn('property="nomorals:revision_ref"', opf)
        self.assertIn(self.manifest.revision_ref, opf)
        self.assertIn('property="nomorals:completeness"', opf)
        self.assertIn("partial 5 of 7", opf)
        self.assertIn("<dc:title>The Gap Tale</dc:title>", opf)
        # the manifest itself verifies against the held content —
        # the export was built from exactly this revision
        self.assertEqual(r["revision_ref"], self.manifest.revision_ref)

    # ── PDF ──────────────────────────────────────────────────────────
    def test_pdf_oracle(self):
        r = self.reader.export_story("gap-tale", "pdf")
        data = Path(r["path"]).read_bytes()
        self.assertTrue(data.startswith(b"%PDF"))
        text = _norm(read_pdf_text(data))
        # every paragraph of every held chapter, in order
        cursor = 0
        for n in sorted(HELD):
            for para in HELD[n].split("\n\n"):
                at = text.find(_norm(para), cursor)
                self.assertGreaterEqual(
                    at, 0, f"chapter {n} paragraph missing from the PDF")
                cursor = at
        self.assertIn("TAIL-BEYOND-PREVIEW", text)
        # the gaps are declared where the chapters would sit
        self.assertIn("Chapter 4 is not in the local library yet", text)
        self.assertIn("Chapter 6 is not in the local library yet", text)
        self.assertIn("declared missing here, never invented", text)
        # stamps: the front-matter line in the body, the /Info
        # metadata carrying subject + the full revision_ref
        self.assertIn("5 of 7 chapters present", text)
        self.assertIn("Missing: chapters 4, 6", text)
        self.assertIn(b"nomorals-longdoc", data)
        self.assertIn(self.manifest.revision_ref.encode(), data)
        self.assertIn(b"Partial edition - 5 of 7 chapters present", data)

    # ── HTML ─────────────────────────────────────────────────────────
    def test_html_oracle(self):
        r = self.reader.export_story("gap-tale", "html")
        html = Path(r["path"]).read_text(encoding="utf-8")
        for n, text in HELD.items():
            for para in text.split("\n\n"):
                self.assertIn(_norm(para), _norm(html),
                              f"chapter {n} lost a paragraph in the HTML")
        self.assertIn("TAIL-BEYOND-PREVIEW", html)
        # the P3 front matter carries the truth: partial, with the
        # missing chapters named — never posed as complete
        self.assertIn("Partial edition", html)
        self.assertIn("5 of 7 chapters present", html)
        self.assertIn("Missing: chapters 4, 6", html)
        self.assertNotIn("Complete edition", html)

    # ── honesty edges ────────────────────────────────────────────────
    def test_zero_held_chapters_refuses_in_words(self):
        _plant(self.reader, slug="empty-tale", held={}, total=3)
        for fmt in ("pdf", "epub", "html"):
            with self.assertRaises(ReaderError) as cm:
                self.reader.export_story("empty-tale", fmt)
            self.assertIn("no chapters in the local library",
                          str(cm.exception))
        exports = (self.reader.story_dir("empty-tale") / "exports")
        self.assertFalse(exports.exists() and any(exports.iterdir()))

    def test_unknown_format_refuses(self):
        with self.assertRaises(ReaderError):
            self.reader.export_story("gap-tale", "docx")

    def test_complete_story_exports_as_complete(self):
        held = {1: "One. DONE1", 2: "Two. DONE2", 3: "Three. DONE3"}
        _plant(self.reader, slug="whole-tale", held=held, total=3)
        r = self.reader.export_story("whole-tale", "epub")
        self.assertEqual(r["completeness"], "complete")
        z = zipfile.ZipFile(io.BytesIO(Path(r["path"]).read_bytes()))
        opf = z.read("OEBPS/content.opf").decode()
        self.assertIn("complete 3 of 3", opf)
        title_page = z.read("OEBPS/title.xhtml").decode()
        self.assertIn("Complete edition", title_page)
        self.assertNotIn("declared missing",
                         _xhtml_text(z.read("OEBPS/ch002.xhtml").decode()))
        pdf = self.reader.export_story("whole-tale", "pdf")
        text = _norm(read_pdf_text(Path(pdf["path"]).read_bytes()))
        self.assertIn("Complete edition", text)
        self.assertNotIn("not in the local library yet", text)

    def test_open_story_exports_what_it_holds(self):
        # the source total is unknown: the export holds exactly the
        # held chapters and says the edition is ongoing — it does
        # not invent a total to look complete against
        _plant(self.reader, slug="open-tale",
               held={1: "First held. OPEN1", 2: "Second held. OPEN2"},
               declared=False)
        r = self.reader.export_story("open-tale", "epub")
        self.assertEqual(r["completeness"], "open")
        self.assertIsNone(r["declared_total"])
        z = zipfile.ZipFile(io.BytesIO(Path(r["path"]).read_bytes()))
        names = [n for n in z.namelist() if "/ch0" in n]
        self.assertEqual(len(names), 2)
        self.assertIn("Ongoing edition",
                      z.read("OEBPS/title.xhtml").decode())


class BooksExportVerbTests(unittest.TestCase):
    """The /books export verb: builds through the same seam and
    delivers the file through the gateway, caption carrying the
    completeness truth."""

    def setUp(self) -> None:
        self.ctx, self.tmp = _ctx()
        self.reader = StoryReader(self.ctx)
        _plant(self.reader)
        from nomorals.agents.partner.runtime_search import RuntimeSearchMixin
        m = RuntimeSearchMixin.__new__(RuntimeSearchMixin)
        m.context = self.ctx
        m._ref_from_key = lambda key: SimpleNamespace(
            platform="whatsapp")
        self.sent: list = {}

        def _send_file(platform, chat, path, caption="", name=""):
            self.sent.update(platform=platform, path=path,
                             caption=caption, name=name)
            return SimpleNamespace(ok=True)

        m.gateway = SimpleNamespace(send_file=_send_file)
        self.m = m

    def test_export_verb_sends_the_stamped_file(self):
        reply = self.m._control_book("export gap-tale epub",
                                     "whatsapp:123")
        self.assertTrue(self.sent["path"].endswith("gap-tale.epub"))
        self.assertTrue(Path(self.sent["path"]).exists())
        self.assertEqual(self.sent["name"], "gap-tale.epub")
        self.assertIn("5 of 7 chapters present", self.sent["caption"])
        self.assertIn("sent above", reply)
        self.assertIn("Missing: chapters 4, 6", reply)

    def test_export_verb_default_format_is_pdf(self):
        self.m._control_book("export gap-tale", "whatsapp:123")
        self.assertTrue(self.sent["path"].endswith("gap-tale.pdf"))

    def test_export_verb_bad_format_is_a_usage_line(self):
        reply = self.m._control_book("export gap-tale docx",
                                     "whatsapp:123")
        self.assertIn("usage:", reply)
        self.assertEqual(self.sent, {})

    def test_export_verb_unknown_story_is_honest(self):
        reply = self.m._control_book("export no-such-story pdf",
                                     "whatsapp:123")
        self.assertIn("export refused", reply)
        self.assertEqual(self.sent, {})


class ForgeDerivativeStampTests(unittest.TestCase):
    """Forge books gain the same stamps (P5): the EPUB declares
    unwritten chapters instead of dropping them, the PDF carries
    the front-matter line + /Info metadata, and the saved
    manuscript.md stays the pure manuscript."""

    def setUp(self) -> None:
        self.ctx, self.tmp = _ctx()
        from nomorals.books.forge import BookForge
        from nomorals.books.model import STATUS_PLAN, STATUS_WRITTEN, Book, Chapter
        self.forge = BookForge(self.ctx)
        book = Book(topic="t", slug="stamp-book", title="Stamp Book",
                    author="Devon", language="en", genre="essay",
                    description="a stamping test")
        book.chapters = [
            Chapter(number=1, title="First",
                    text="Written one, fully present. FORGE1",
                    status=STATUS_WRITTEN),
            Chapter(number=2, title="Second",
                    text="Written two, fully present. FORGE2",
                    status=STATUS_WRITTEN),
            Chapter(number=3, title="Third", text="",
                    status=STATUS_PLAN),
        ]
        self.forge.save(book)
        self.book = book

    def test_forge_epub_declares_and_stamps(self):
        from nomorals.books.forge import epub_bytes
        from nomorals.books.manifest import manifest_from_book
        data = epub_bytes(self.book)
        z = zipfile.ZipFile(io.BytesIO(data))
        names = z.namelist()
        for n in (1, 2, 3):
            self.assertIn(f"OEBPS/ch{n:03d}.xhtml", names)
        third = _xhtml_text(z.read("OEBPS/ch003.xhtml").decode())
        self.assertIn("has not been written yet", third)
        self.assertIn("no text is invented", third)
        first = _xhtml_text(z.read("OEBPS/ch001.xhtml").decode())
        self.assertIn("FORGE1", first)
        opf = z.read("OEBPS/content.opf").decode()
        manifest = manifest_from_book(self.book)
        self.assertIn(manifest.revision_ref, opf)
        self.assertIn("partial 2 of 3", opf)
        self.assertIn("Partial edition",
                      z.read("OEBPS/title.xhtml").decode())

    def test_forge_build_pdf_stamps_and_manuscript_stays_pure(self):
        r = self.forge.build("stamp-book",
                             formats=("pdf", "epub", "html"))
        data = Path(r["pdf"]).read_bytes()
        text = _norm(read_pdf_text(data))
        self.assertIn("FORGE1", text)
        self.assertIn("FORGE2", text)
        self.assertIn("2 of 3 chapters present", text)
        from nomorals.books.manifest import manifest_from_book
        manifest = manifest_from_book(self.forge.load("stamp-book"))
        self.assertIn(manifest.revision_ref.encode(), data)
        manuscript = Path(r["manuscript"]).read_text(encoding="utf-8")
        self.assertNotIn("chapters present", manuscript)
        self.assertTrue(Path(r["epub"]).exists())
        self.assertTrue(Path(r["html"]).exists())


class DocumentDerivativeStampTests(unittest.TestCase):
    """Documents exports carry the revision their P1 contract
    promises: PDF front matter + /Info, EPUB OPF — while
    to_markdown's own output (and its round-trip) stays untouched."""

    def _doc(self):
        from nomorals.documents.model import Section, new_document
        doc = new_document(format="md", title="Quarterly")
        doc.sections = [
            Section(level=1, heading="Intro",
                    text="Introduction body text. DOCSTAMP1", page=1),
            Section(level=1, heading="Figures",
                    text="The figures section body. DOCSTAMP2", page=1),
        ]
        return doc

    def test_to_pdf_carries_revision_stamp(self):
        from nomorals.documents.convert import to_markdown, to_pdf
        from nomorals.documents.manifest import manifest_from_document
        doc = self._doc()
        manifest = manifest_from_document(doc)
        data = to_pdf(doc)
        text = _norm(read_pdf_text(data))
        self.assertIn(f"Revision {doc.revision}", text)
        self.assertIn("Complete edition", text)
        self.assertIn("DOCSTAMP1", text)
        self.assertIn("DOCSTAMP2", text)
        self.assertIn(manifest.revision_ref.encode(), data)
        # the stamp lives only in the PDF copy — to_markdown's own
        # output (and its parse round-trip) is unchanged
        self.assertNotIn("Revision", to_markdown(doc))

    def test_to_epub_carries_revision_stamp_and_round_trips(self):
        from nomorals.documents import parse_bytes
        from nomorals.documents.convert import to_epub
        from nomorals.documents.manifest import manifest_from_document
        doc = self._doc()
        manifest = manifest_from_document(doc)
        data = to_epub(doc)
        z = zipfile.ZipFile(io.BytesIO(data))
        opf = z.read("OEBPS/content.opf").decode()
        self.assertIn('property="nomorals:revision_ref"', opf)
        self.assertIn(manifest.revision_ref, opf)
        self.assertIn("complete 2 of 2", opf)
        back = parse_bytes(data, filename="x.epub")
        self.assertEqual(back.title, "Quarterly")
        self.assertGreaterEqual(len(back.sections), 2)


if __name__ == "__main__":
    unittest.main()
