"""52-websites #10 — Project Gutenberg adapter in the search pool.

The adapter itself predates this lane (books/descriptors.py); what
this suite pins is the #10 wiring:

* the catalogue record carries delivery formats + the public-domain
  licence label (StoryMeta → FollowedStory → read reply);
* a title follow ("get me Middlemarch") reaches Gutenberg through
  search_all and lands a readable book in the reader + its library;
* the tool surface (story_search) reaches Gutenberg too.

All network is stubbed (mocked urlopen) — no live fetches.  The
Gutendex payloads are real-shape (Middlemarch, id 145, George
Eliot); the plain-text fixture is short neutral prose cut into two
chapters, not the book's real text.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.books import sources as book_sources
from nomorals.books.descriptors import GutenbergAdapter
from nomorals.books.reader import StoryReader
from nomorals.books.sources import SourceError

MIDDLEMARCH_BOOK = {
    "id": 145,
    "title": "Middlemarch",
    "authors": [{"name": "Eliot, George",
                 "birth_year": 1819, "death_year": 1880}],
    "subjects": ["Didactic fiction", "England -- Fiction",
                 "Women -- England -- Fiction"],
    "bookshelves": ["Best Books Ever Listings"],
    "languages": ["en"],
    "copyright": False,
    "media_type": "Text",
    "formats": {
        "text/html": "https://www.gutenberg.org/files/145/145-h/145-h.htm",
        "application/epub+zip": "https://www.gutenberg.org/files/145/145.epub",
        "application/x-mobipocket-ebook": "https://www.gutenberg.org/files/145/145.kindle",
        "text/plain; charset=utf-8": "https://www.gutenberg.org/files/145/145.txt",
        "text/plain; charset=us-ascii": "https://www.gutenberg.org/files/145/145-8.txt",
        "application/rdf+xml": "https://www.gutenberg.org/ebooks/145.rdf",
    },
    "download_count": 12345,
}

MIDDLEMARCH_SEARCH = {"count": 1, "results": [MIDDLEMARCH_BOOK]}

#: short neutral fixture prose (NOT the book's real text), cut into
#: two plain-text chapters by library.split_chapters.
MIDDLEMARCH_TEXT = """The Project Gutenberg eBook of Middlemarch, by George Eliot

*** START OF THE PROJECT GUTENBERG EBOOK MIDDLEMARCH ***

CHAPTER I.

The town woke early. The market square filled with carts and
voices, and the morning bell rang over the rooftops.

Miss Brooke walked to the window and looked out at the street
below, where the day was already busy with errands.

CHAPTER II.

Sir James arrived after breakfast with news from the parish,
and the household gathered to hear what he had to say.

The afternoon passed quietly, with letters to write and
visitors to receive before the evening meal.

*** END OF THE PROJECT GUTENBERG EBOOK MIDDLEMARCH ***
"""


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _gutenberg_urlopen(url, timeout=None):
    """Route every Gutenberg/Gutendex URL to a canned payload.

    Anything else raises loudly — an unexpected URL is a test bug,
    not a network call to hide."""
    target = url if isinstance(url, str) else url.full_url
    if "gutendex.com/books?search" in target:
        return _FakeResponse(json.dumps(MIDDLEMARCH_SEARCH).encode())
    if "gutendex.com/books/145" in target:
        return _FakeResponse(json.dumps(MIDDLEMARCH_BOOK).encode())
    if "gutenberg.org/files/145/" in target:
        return _FakeResponse(MIDDLEMARCH_TEXT.encode())
    raise SourceError(f"unexpected Gutenberg URL {target}")


def _ctx():
    tmp = tempfile.TemporaryDirectory()
    ctx = SimpleNamespace(
        db=None, settings=SimpleNamespace(workspace_dir=tmp.name))
    return ctx, tmp


class TestGutenbergFormats(unittest.TestCase):
    def test_format_labels_canonical_order_and_dedupe(self):
        labels = GutenbergAdapter._format_labels({
            "application/pdf": "x",
            "text/plain; charset=us-ascii": "x",
            "text/plain; charset=utf-8": "x",
            "application/epub+zip": "x",
            "text/html": "x",
            "application/x-mobipocket-ebook": "x",
            "application/rdf+xml": "x",
            "image/jpeg": "x",
        })
        self.assertEqual(labels,
                         ["Plain text", "HTML", "EPUB", "Kindle", "PDF"])

    def test_format_labels_empty(self):
        self.assertEqual(GutenbergAdapter._format_labels({}), [])
        self.assertEqual(GutenbergAdapter._format_labels(None), [])

    def test_txt_url_prefers_utf8_then_ascii(self):
        ad = GutenbergAdapter()
        self.assertEqual(
            ad._txt_url({"formats": {
                "text/plain; charset=us-ascii": "http://x/a.txt",
                "text/plain; charset=utf-8": "http://x/u.txt"}}),
            "http://x/u.txt")
        self.assertEqual(
            ad._txt_url({"formats": {
                "text/plain; charset=us-ascii": "http://x/a.txt"}}),
            "http://x/a.txt")
        self.assertEqual(
            ad._txt_url({"formats": {
                "text/plain; charset=iso-8859-1": "http://x/i.txt"}}),
            "http://x/i.txt")

    def test_txt_url_raises_without_plain_text(self):
        ad = GutenbergAdapter()
        with self.assertRaises(SourceError) as ctx:
            ad._txt_url({"formats": {"application/epub+zip": "http://x/e"}})
        self.assertIn("no plain-text format", str(ctx.exception))

    def test_novel_carries_formats_and_licence(self):
        ad = GutenbergAdapter()
        with mock.patch("urllib.request.urlopen",
                        side_effect=_gutenberg_urlopen):
            meta = ad.novel("https://www.gutenberg.org/ebooks/145")
        self.assertEqual(meta.title, "Middlemarch")
        self.assertEqual(meta.author, "Eliot, George")
        self.assertEqual(meta.licence, "Public domain (US)")
        self.assertEqual(meta.formats,
                         ["Plain text", "HTML", "EPUB", "Kindle"])
        self.assertIn("Didactic fiction", meta.synopsis)


class TestGutenbergReaderEndToEnd(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("urllib.request.urlopen",
                             side_effect=_gutenberg_urlopen)
        patcher.start()
        self.addCleanup(patcher.stop)
        # the search pool's dedicated adapters would hit the live
        # network — the pool under test is Gutenberg only
        pool_patcher = mock.patch.object(book_sources, "ADAPTERS", [])
        pool_patcher.start()
        self.addCleanup(pool_patcher.stop)
        book_sources._GUTENBERG_POOL_ADAPTER = None
        self.addCleanup(setattr, book_sources,
                        "_GUTENBERG_POOL_ADAPTER", None)

    def test_follow_by_title_lands_licence_and_formats(self):
        ctx, tmp = _ctx()
        self.addCleanup(tmp.cleanup)
        reader = StoryReader(ctx)
        story = reader.follow("Middlemarch")
        self.assertEqual(story.title, "Middlemarch")
        self.assertEqual(story.source, "gutenberg")
        self.assertEqual(story.licence, "Public domain (US)")
        self.assertEqual(story.formats,
                         ["Plain text", "HTML", "EPUB", "Kindle"])
        # the record on disk carries them (the catalogue record)
        on_disk = json.loads(
            (reader.story_dir(story.slug) / "story.json")
            .read_text(encoding="utf-8"))
        self.assertEqual(on_disk["licence"], "Public domain (US)")
        self.assertEqual(on_disk["formats"],
                         ["Plain text", "HTML", "EPUB", "Kindle"])
        # the shelf lists them
        rows = reader.following()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["licence"], "Public domain (US)")
        self.assertEqual(rows[0]["formats"],
                         ["Plain text", "HTML", "EPUB", "Kindle"])

    def test_read_reply_labels_licence_and_formats(self):
        ctx, tmp = _ctx()
        self.addCleanup(tmp.cleanup)
        reader = StoryReader(ctx)
        story = reader.follow("Middlemarch")
        out = reader.read(story.slug, chapter=1)
        self.assertEqual(out["chapter"], 1)
        self.assertIn("market square", out["text"])
        self.assertEqual(out["licence"], "Public domain (US)")
        self.assertEqual(out["formats"],
                         ["Plain text", "HTML", "EPUB", "Kindle"])
        self.assertEqual(out["source"], "gutenberg")

    def test_ingest_pulls_the_whole_book_into_the_library(self):
        ctx, tmp = _ctx()
        self.addCleanup(tmp.cleanup)
        reader = StoryReader(ctx)
        story = reader.follow("Middlemarch")
        report = reader.ingest(story.slug)
        self.assertEqual(report["ingested"], 2)
        self.assertTrue(report["complete"])
        self.assertEqual(report["failed"], [])
        chapters = sorted(
            (reader.story_dir(story.slug) / "chapters").glob("*.md"))
        self.assertEqual(len(chapters), 2)
        self.assertIn("market square", chapters[0].read_text())
        self.assertIn("Sir James arrived", chapters[1].read_text())

    def test_re_follow_refreshes_metadata_keeps_progress(self):
        ctx, tmp = _ctx()
        self.addCleanup(tmp.cleanup)
        reader = StoryReader(ctx)
        story = reader.follow("Middlemarch")
        reader.set_progress(story.slug, 1)
        again = reader.follow("https://www.gutenberg.org/ebooks/145")
        self.assertEqual(again.slug, story.slug)
        self.assertEqual(again.licence, "Public domain (US)")
        self.assertEqual(again.formats,
                         ["Plain text", "HTML", "EPUB", "Kindle"])
        prog = reader.get_progress(story.slug)
        self.assertEqual(prog["chapter"], 1)


class _CapturingRegistry:
    """A registry double that records the book tools' callables."""

    def __init__(self, context):
        self.context = context
        self.tools = {}
        self.descriptions = {}

    def register(self, name, description="", capability=None,
                 parameters=None, **_kwargs):
        self.descriptions[name] = description

        def deco(fn):
            self.tools[name] = fn
            return fn
        return deco


class TestGutenbergToolSurface(unittest.TestCase):
    def test_story_search_reaches_gutenberg(self):
        ctx, tmp = _ctx()
        self.addCleanup(tmp.cleanup)
        from nomorals.books import tools as book_tools
        reg = _CapturingRegistry(ctx)
        book_tools.register(reg)
        # the description names Gutenberg, so the brain can pick it
        # for a public-domain book request
        self.assertIn("Project Gutenberg",
                      reg.descriptions["story_search"])
        with mock.patch("urllib.request.urlopen",
                        side_effect=_gutenberg_urlopen), \
                mock.patch.object(book_sources, "ADAPTERS", []):
            book_sources._GUTENBERG_POOL_ADAPTER = None
            try:
                out = reg.tools["story_search"]("middlemarch")
            finally:
                book_sources._GUTENBERG_POOL_ADAPTER = None
        self.assertEqual(len(out["hits"]), 1)
        hit = out["hits"][0]
        self.assertEqual(hit["title"], "Middlemarch")
        self.assertEqual(hit["source"], "gutenberg")
        self.assertEqual(hit["url"],
                         "https://www.gutenberg.org/ebooks/145")


if __name__ == "__main__":
    unittest.main()
