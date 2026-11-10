"""Doc 16 Pass 3 — branded editions: ONE HTML renderer for every
file edition, carrying the Peacethefirst brand.

Pins:
(a) one manifest per treatment renders completely — every present
    segment's text present in the HTML, in order; absent segments
    declared in the contents and in the completeness line;
(b) brand: default on (cover line + footer + print-only mark),
    brand_policy=off grep-clean, custom text honoured, custom
    without text falls back to the house mark; the core brand
    constant equals the chat typography canonical stamp;
(c) self-containment: no external asset references; every segment
    still present with the script block removed;
(d) escaping: planted markup in source content is inert;
(e) completeness honesty: a partial edition says "N of M" in its
    front matter; revision counters print, hash identities never;
(f) compatibility: forge html_book + documents to_html keep their
    pinned shapes; the themes render; deliver_report's chrome
    (cover/toc/footer + resolving anchors) survives the re-point.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from nomorals.books.forge import html_book
from nomorals.books.manifest import (
    book_content_resolver,
    manifest_from_book,
    manifest_from_story,
    story_content_resolver,
)
from nomorals.books.model import STATUS_PLAN, STATUS_WRITTEN, Book
from nomorals.books.model import Chapter as BookChapter
from nomorals.books.reader import FollowedStory, StoryReader
from nomorals.books.sources import Chapter as SourceChapter
from nomorals.core import edition
from nomorals.core.edition import completeness_line, render_edition
from nomorals.documents.convert import to_html
from nomorals.documents.errors import DocumentError
from nomorals.documents.manifest import manifest_from_document
from nomorals.documents.model import Document, Table
from nomorals.documents.model import Section as DocSection
from nomorals.social.chat.manifest import (
    doc_content_resolver,
    manifest_from_doc,
    manifest_from_text,
    text_content_resolver,
)
from nomorals.social.chat.present import KV, Doc, Section
from nomorals.social.chat.typography import DEFAULT_STAMP_TEXT
from nomorals.tools.deliver_report import render_report_html
from nomorals.tools.filesend import md_to_html


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


_EXTERNAL_ASSET = re.compile(
    r"""(?:src|href)\s*=\s*["']https?://|url\(\s*["']?https?://""", re.I)
_SCRIPT_BLOCK = re.compile(r"<script>.*?</script>", re.S)


class TreatmentRenderTests(unittest.TestCase):
    """(a) every treatment renders its manifest completely."""

    def test_book_treatment_complete_and_ordered(self):
        book = _book()
        out = render_edition(manifest_from_book(book),
                             book_content_resolver(book))
        self.assertIn('<body class="treatment-book">', out)
        self.assertIn("Body one.", out)
        self.assertIn("Body three.", out)
        self.assertLess(out.index("Body one."), out.index("Body three."))
        # the declared-but-unwritten chapter 2 is declared absent
        self.assertIn("not yet available", out)
        self.assertIn("2 of 3 chapters", completeness_line(
            manifest_from_book(book)))
        self.assertIn("Partial edition", out)

    def test_story_treatment_declares_missing_chapters(self):
        reader, story = _reader_with_story()
        manifest = manifest_from_story(reader, story.slug)
        out = render_edition(manifest, story_content_resolver(reader))
        for n in (1, 2, 3, 5):
            self.assertIn(f"Alpha {n}.", out)
            self.assertIn(f"Beta {n}.", out)
        line = completeness_line(manifest)
        self.assertIn("4 of 10 chapters", line)
        self.assertIn("chapters 4, 6–10", line)
        self.assertIn("Partial edition", out)

    def test_technical_treatment_renders_sections_and_table(self):
        from nomorals.documents.convert import _edition_resolver
        doc = _document()
        out = render_edition(manifest_from_document(doc),
                             _edition_resolver(doc))
        self.assertIn("treatment-technical", out)
        self.assertIn("Body text.", out)
        self.assertIn("More body.", out)
        self.assertIn("<table>", out)
        self.assertIn("<th>a</th>", out)
        self.assertIn("<td>1</td>", out)

    def test_report_treatment_present_doc_full(self):
        doc = _present_doc()
        out = render_edition(manifest_from_doc(doc),
                             doc_content_resolver(doc))
        self.assertIn("treatment-report", out)
        self.assertIn("The summary.", out)
        self.assertIn("a line", out)
        self.assertIn("the long tail", out)  # overflow segment rendered

    def test_themes_all_render(self):
        manifest = manifest_from_text("Alpha body.\n\nBeta body.",
                                      title="Notes")
        for theme in ("light", "dark", "sepia", "print", "minimal"):
            out = render_edition(manifest,
                                 text_content_resolver(
                                     "Alpha body.\n\nBeta body."),
                                 theme=theme)
            self.assertIn(f'data-theme="{theme}"', out)
            self.assertIn("Alpha body.", out)
        dark = render_edition(manifest, text_content_resolver("x"),
                              theme="dark")
        self.assertIn("#1a1a2e", dark)


class BrandTests(unittest.TestCase):
    """(b) the brand: default / off / custom, chrome only."""

    def _render(self, **kw):
        manifest = manifest_from_text("Some body text.", title="Notes")
        return render_edition(manifest,
                              text_content_resolver("Some body text."),
                              **kw)

    def test_brand_constant_matches_chat_canon(self):
        self.assertEqual(edition.BRAND_TEXT, DEFAULT_STAMP_TEXT)
        self.assertEqual(edition.BRAND_TEXT, "Powered by peacethefirst")

    def test_default_brand_in_chrome(self):
        out = self._render()
        self.assertIn('<p class="brand-line">Powered by peacethefirst</p>',
                      out)
        self.assertIn("edition-footer", out)
        self.assertIn("print-mark", out)
        self.assertGreaterEqual(out.count("Powered by peacethefirst"), 3)

    def test_brand_off_is_grep_clean(self):
        out = self._render(brand_policy="off")
        self.assertNotIn("peacethefirst", out.lower())
        self.assertNotIn('<p class="brand-line">', out)
        self.assertNotIn('class="print-mark"', out)
        self.assertIn("Some body text.", out)  # content unaffected

    def test_brand_off_via_manifest_policy(self):
        manifest = manifest_from_text("Body.", title="Notes")
        manifest.presentation.brand_policy = "off"
        out = render_edition(manifest, text_content_resolver("Body."))
        self.assertNotIn("peacethefirst", out.lower())

    def test_custom_brand_text(self):
        out = self._render(brand_policy="custom",
                           brand_text="Made by Zed")
        self.assertIn("Made by Zed", out)
        self.assertNotIn("peacethefirst", out.lower())

    def test_custom_without_text_falls_back_to_house_mark(self):
        out = self._render(brand_policy="custom")
        self.assertIn("Powered by peacethefirst", out)

    def test_brand_never_inside_content(self):
        # a code segment stays byte-faithful: no brand is injected
        # into segment content even when the brand is on
        doc = Document(
            id="d", format="md", title="Code Doc", author="",
            source="f.md",
            sections=[DocSection(level=2, heading="Code", kind="code",
                                 text='print("hello")')])
        out = to_html(doc)
        code_block = out.split("<pre><code>", 1)[1].split(
            "</code></pre>", 1)[0]
        # the code survives as itself (standard-escaped) and the
        # brand never lands inside it
        self.assertEqual(code_block, "print(&quot;hello&quot;)")
        self.assertNotIn("peacethefirst", code_block.lower())


class SelfContainmentTests(unittest.TestCase):
    """(c) no external assets; content survives without the script."""

    def test_no_external_asset_references(self):
        book = _book()
        outputs = [
            html_book(book),
            to_html(_document()),
            md_to_html("# T\n\nBody.\n", title="T"),
            render_report_html("Topic", [("S", "body")]),
        ]
        for out in outputs:
            self.assertIsNone(_EXTERNAL_ASSET.search(out), out[:200])

    def test_content_present_with_script_removed(self):
        book = _book()
        out = html_book(book)
        stripped = _SCRIPT_BLOCK.sub("", out)
        self.assertNotIn("<script>", stripped)
        self.assertIn("Body one.", stripped)
        self.assertIn("Body three.", stripped)
        self.assertIn("T Book", stripped)


class EscapingTests(unittest.TestCase):
    """(d) untrusted source markup renders as inert text."""

    def test_planted_script_in_chapter_is_inert(self):
        book = Book(
            topic="T", slug="evil-book", title="Evil", author="",
            chapters=[BookChapter(
                number=1, title="One",
                text='Before <script>alert(1)</script> after.',
                status=STATUS_WRITTEN)])
        out = html_book(book)
        self.assertNotIn("<script>alert(1)</script>", out)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", out)

    def test_planted_markup_in_document_is_inert(self):
        doc = Document(
            id="d", format="md", title="T", author="", source="f.md",
            sections=[DocSection(
                level=1, heading="H",
                text="the <b>quarterly</b> results & more")])
        out = to_html(doc)
        self.assertNotIn("<b>quarterly</b>", out)
        self.assertIn("&lt;b&gt;quarterly&lt;/b&gt;", out)
        self.assertIn("&amp;", out)

    def test_planted_script_in_report_section_title(self):
        out = render_report_html(
            "Topic", [("<script>alert(1)</script>", "body text")])
        self.assertNotIn("<script>alert(1)</script>", out)
        self.assertIn("&lt;script&gt;", out)


class FrontMatterHonestyTests(unittest.TestCase):
    """(e) completeness + revision: counters print, hashes never."""

    def test_partial_edition_states_n_of_m(self):
        out = html_book(_book())
        self.assertIn("Partial edition", out)
        self.assertIn("2 of 3 chapters", out)

    def test_complete_edition_says_complete(self):
        book = Book(
            topic="T", slug="done-book", title="Done", author="",
            chapters=[BookChapter(number=1, title="One", text="Only.",
                                  status=STATUS_WRITTEN)])
        out = html_book(book)
        self.assertIn("Complete edition", out)

    def test_document_revision_counter_prints(self):
        doc = _document()
        manifest = manifest_from_document(doc)
        out = to_html(doc)
        self.assertIsNotNone(manifest.revision)
        self.assertIn(f"Revision {manifest.revision}", out)

    def test_hash_identity_never_prints(self):
        book = _book()
        manifest = manifest_from_book(book)
        out = html_book(book)
        self.assertIsNone(manifest.revision)
        self.assertNotIn("Revision", out.split("<main>")[0])
        self.assertNotIn(manifest.content_hash[:16], out)
        self.assertNotIn(manifest.revision_ref, out)


class CompatibilityTests(unittest.TestCase):
    """(f) the re-pointed producers keep their pinned shapes."""

    def test_forge_html_book_shape(self):
        out = html_book(_book(), theme="dark")
        self.assertIn("<!DOCTYPE html>", out)
        self.assertIn("T Book", out)
        self.assertIn('id="ch1"', out)
        self.assertIn("#1a1a2e", out)

    def test_documents_to_html_shape(self):
        out = to_html(_document())
        self.assertTrue(out.startswith("<!DOCTYPE html>"))
        self.assertIn("<style>", out)
        self.assertIn("Contents", out)
        self.assertIn("<table>", out)

    def test_documents_heading_anchor_and_toc_off(self):
        doc = Document(
            id="d", format="md", title="T", author="", source="f.md",
            sections=[DocSection(level=1, heading="Intro",
                                 text="Hello.")])
        out = to_html(doc)
        self.assertIn('href="#intro"', out)
        self.assertIn("<h1>Intro</h1>", out)
        bare = to_html(doc, toc=False)
        self.assertNotIn("Contents", bare)
        self.assertNotIn('id="intro"', bare)

    def test_documents_unknown_theme_raises(self):
        with self.assertRaises(DocumentError) as ctx:
            to_html(_document(), theme="neon")
        self.assertEqual(ctx.exception.kind, "invalid_argument")

    def test_documents_brand_default_and_off(self):
        self.assertIn("Powered by peacethefirst", to_html(_document()))
        self.assertNotIn("peacethefirst",
                         to_html(_document(),
                                 brand_policy="off").lower())

    def test_md_to_html_renders_markdown_edition(self):
        out = md_to_html("# Head\n\nBody **bold** text.\n", title="Rep")
        self.assertIn("<h1", out)
        self.assertIn("<strong>bold</strong>", out)
        self.assertIn("Powered by peacethefirst", out)

    def test_md_to_html_chrome_free_body(self):
        out = md_to_html("# Head\n\nBody.\n", title="Rep", chrome=False)
        self.assertIn("Body.", out)
        self.assertNotIn('class="doc-title"', out)
        self.assertNotIn("peacethefirst", out.lower())

    def test_render_report_html_chrome_and_anchors(self):
        out = render_report_html(
            "Quarterly", [("Overview", "All good."),
                          ("Risks", "Watch out.")], title="Q")
        self.assertIn('class="report-toc"', out)
        self.assertIn("report-cover", out)
        self.assertIn("report-footer", out)
        self.assertIn('href="#overview"', out)
        self.assertIn('id="overview"', out)  # the href resolves
        self.assertIn('href="#risks"', out)
        self.assertIn('id="risks"', out)
        self.assertIn("Powered by peacethefirst", out)


if __name__ == "__main__":
    unittest.main()
