"""Universal story reader: descriptors, Gutenberg, blocked honesty,
bible auto-build, voiceprint drift, progress, cache sync, serials.

All network is mocked (FakeFetcher / mocked urlopen) — no live fetches.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.books import cache_sync
from nomorals.books.bible import BibleBuilder
from nomorals.books.descriptors import (
    DescriptorAdapter, GutenbergAdapter, descriptor_by_id,
    descriptor_for_url, list_sources,
)
from nomorals.books.reader import StoryReader
from nomorals.books.serial import SerialManager
from nomorals.books.sources import SourceBlocked, SourceError, adapter_for_url
from nomorals.books.voiceprint import compare, fingerprint
from nomorals.tools.browser import parse_html


def _ctx():
    tmp = tempfile.mkdtemp()
    return SimpleNamespace(db=None, settings=SimpleNamespace(workspace_dir=tmp)), tmp


class _FakeFetcher:
    def __init__(self, pages):
        self.pages = pages

    def get(self, url, **_):
        for key, html in self.pages.items():
            if url.startswith(key):
                return url, parse_html(html), html
        raise SourceError(f"no fixture for {url}")


SH_SEARCH = """<html><body>
<a href="https://www.scribblehub.com/series/123/the-wandering-scribe">The Wandering Scribe</a>
</body></html>"""

SH_NOVEL = """<html><head>
<meta property="og:title" content="The Wandering Scribe"/>
<meta property="og:description" content="A scribe wanders."/>
</head><body>
<div class="toc">
<a href="https://www.scribblehub.com/read/123/chapter-1">Chapter 1: The Road</a>
<a href="https://www.scribblehub.com/read/123/chapter-2">Chapter 2: The River</a>
</div></body></html>"""

SH_CHAPTER_BODY = " ".join(
    ["The road stretched ahead, dusty and long, and the scribe walked it "
     "with ink-stained hands and a pack full of paper."] * 12)
SH_CHAPTER = f"""<html><body><div class="chp_raw">
<p>{SH_CHAPTER_BODY}</p>
<p>He stopped at the river and washed his hands.</p>
</div></body></html>"""


class TestDescriptors(unittest.TestCase):
    def test_registry_legal_first(self):
        sources = list_sources()
        ids = [s["id"] for s in sources]
        self.assertIn("gutenberg", ids)
        self.assertIn("royalroad", ids)
        self.assertIn("scribblehub", ids)
        # every row carries a rights label
        self.assertTrue(all(s["licence"] for s in sources))
        self.assertTrue(all(s["rights_class"] for s in sources))

    def test_descriptor_for_url(self):
        self.assertEqual(descriptor_for_url(
            "https://www.scribblehub.com/series/1/x").id, "scribblehub")
        self.assertEqual(descriptor_for_url(
            "https://www.gutenberg.org/ebooks/84").id, "gutenberg")
        self.assertEqual(descriptor_for_url(
            "https://archiveofourown.org/works/1").id, "ao3")
        self.assertIsNone(descriptor_for_url("https://example.com/x"))

    def test_scribblehub_descriptor_flow(self):
        fetcher = _FakeFetcher({
            "https://www.scribblehub.com/?s=": SH_SEARCH,
            "https://www.scribblehub.com/series/123": SH_NOVEL,
            "https://www.scribblehub.com/read/123/chapter-1": SH_CHAPTER,
        })
        ad = DescriptorAdapter(descriptor_by_id("scribblehub"), fetcher=fetcher)
        hits = ad.search("wandering")
        self.assertEqual(len(hits), 1)
        self.assertIn("Wandering Scribe", hits[0].title)
        meta = ad.novel("https://www.scribblehub.com/series/123/the-wandering-scribe")
        self.assertEqual(meta.title, "The Wandering Scribe")
        self.assertIn("Scribble Hub", meta.licence)
        chapters = ad.chapter_list(meta.url)
        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[0][0], 1)
        ch = ad.fetch_chapter(chapters[0][2])
        self.assertGreater(ch.words, 100)
        self.assertIn("scribe", ch.text.lower())

    def test_wattpad_requires_key_no_network(self):
        def _boom(*_a, **_k):
            raise AssertionError("wattpad must not hit the network without a key")
        ad = DescriptorAdapter(descriptor_by_id("wattpad"),
                               fetcher=_FakeFetcher({}))
        ad.fetcher.get = _boom
        with self.assertRaises(SourceError) as cm:
            ad.search("anything")
        self.assertIn("WATTPAD_API_KEY", str(cm.exception))

    def test_blocked_source_honest(self):
        class _BlockedFetcher:
            def get(self, url, **_):
                raise SourceBlocked("cloudflare challenge")
        ad = DescriptorAdapter(descriptor_by_id("scribblehub"),
                               fetcher=_BlockedFetcher())
        with self.assertRaises(SourceBlocked):
            ad.fetch_chapter("https://www.scribblehub.com/read/1/chapter-1")

    def test_adapter_for_url_routes_descriptors(self):
        ad = adapter_for_url("https://www.gutenberg.org/ebooks/84")
        self.assertIsInstance(ad, GutenbergAdapter)
        ad2 = adapter_for_url("https://www.scribblehub.com/series/1/x")
        self.assertIsInstance(ad2, DescriptorAdapter)


GUT_BOOK = {
    "id": 84, "title": "Frankenstein",
    "authors": [{"name": "Shelley, Mary Wollstonecraft"}],
    "subjects": ["Horror tales"],
    "formats": {"text/plain; charset=utf-8":
                "https://www.gutenberg.org/files/84/84-8.txt"},
}
GUT_TEXT = """*** START OF THE PROJECT GUTENBERG EBOOK FRANKENSTEIN ***

Chapter 1

It was on a dreary night of November that I beheld the accomplishment
of my toils. With an anxiety that almost amounted to agony, I collected
the instruments of life around me, that I might infuse a spark of
being into the lifeless thing that lay at my feet.

Chapter 2

I beheld the stream of my existence flowing onwards. The creature
turned and looked upon me with eyes that spoke of a horror I had
wrought, and I fled from the room in terror.

*** END OF THE PROJECT GUTENBERG EBOOK FRANKENSTEIN ***"""


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
    if isinstance(url, str):
        target = url
    else:
        target = url.full_url
    if "gutendex.com/books?search" in target:
        return _FakeResponse(json.dumps(
            {"results": [GUT_BOOK]}).encode())
    if "gutendex.com/books/84" in target:
        return _FakeResponse(json.dumps(GUT_BOOK).encode())
    if "gutenberg.org/files/84" in target:
        return _FakeResponse(GUT_TEXT.encode())
    raise SourceError(f"unexpected Gutenberg URL {target}")


class TestGutenberg(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("urllib.request.urlopen",
                             side_effect=_gutenberg_urlopen)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.ad = GutenbergAdapter()

    def test_search_novel_chapters(self):
        hits = self.ad.search("frankenstein")
        self.assertEqual(hits[0].title, "Frankenstein")
        self.assertIn("gutenberg.org/ebooks/84", hits[0].url)
        meta = self.ad.novel("https://www.gutenberg.org/ebooks/84")
        self.assertEqual(meta.title, "Frankenstein")
        self.assertIn("Public domain", meta.licence)
        chapters = self.ad.chapter_list(meta.url)
        self.assertGreaterEqual(len(chapters), 1)
        ch = self.ad.fetch_chapter(chapters[0][2])
        self.assertGreater(ch.words, 20)
        self.assertIn("dreary night", ch.text)

    def test_bad_url_honest(self):
        with self.assertRaises(SourceError):
            self.ad.novel("https://www.gutenberg.org/ebooks/notanumber")


VOICE_A = ("The old harbour smelled of salt and tar. Mara stood on the "
           "stone quay and watched the fishing boats come home, their "
           "lanterns swaying in the dark water. She had waited all day "
           "for the tide to turn, and when it did she felt the cold wind "
           "on her face and knew that nothing would be the same again.")
VOICE_B = ("The old harbour smelled of salt and tar. Mara stood on the "
           "stone quay and watched the fishing boats come home, their "
           "lanterns swaying in the dark water. She had waited all day "
           "for the tide to turn, and when it did she felt the cold wind "
           "on her face and knew that nothing would be the same again.")


class TestVoiceprintDrift(unittest.TestCase):
    def test_same_voice_zero_drift(self):
        result = compare(fingerprint(VOICE_A), fingerprint(VOICE_B))
        self.assertEqual(result["drift"], 0.0)
        self.assertFalse(result["blocked"])

    def test_different_voice_drifts_more(self):
        other = ("BUY NOW! LIMITED TIME! CLICK HERE! FREE! FREE! FREE! "
                 "!!! ??? ... *** --- $$$ ### @@@")
        same = compare(fingerprint(VOICE_A), fingerprint(VOICE_B))
        diff = compare(fingerprint(VOICE_A), fingerprint(other))
        self.assertGreater(diff["drift"], same["drift"])


class TestBibleAutoBuild(unittest.TestCase):
    def test_bible_from_chapters(self):
        ctx, _tmp = _ctx()
        builder = BibleBuilder(ctx)
        chapters = [
            (1, "The Road", """
Elena walked the road. "I must reach the tower before nightfall," Elena said.
Tomas had been her companion for three years, ever since the flood.
"The tower keepers forbid crossing the river at night," Tomas said.
Elena counted the stones the way her mother had taught her, and Elena did not like the math.
"I will carry the map," Elena said, "and you carry the lantern, Tomas."
Tomas laughed despite himself. "You always did talk too much, Elena," he said.
"""),
            (2, "The Tower", """
"The tower remembers every name carved into it," said Keeper Aldric.
Elena touched the stone. "I will not let the bell ring twice," she vowed.
Tomas drew his blade. Aldric smiled the smile of a man holding a secret.
"We need to reach the top before dawn," Tomas said.
The old rule held: no fire may be kindled within the tower. It costs the bearer their memory.
Aldric studied Elena for a long moment. "The girl has her mother's eyes," Aldric murmured.
"I must know what the bell promised you, Aldric," Elena said quietly.
"""),
        ]
        bible = builder.build("test-story", "The Tower", chapters)
        self.assertTrue(bible.characters or bible.threads or bible.world_rules)
        self.assertTrue(bible.voice_notes or bible.tone)
        loaded = builder.load("test-story")
        self.assertIsNotNone(loaded)
        updated = builder.update("test-story", [
            (3, "The Bell", "Elena heard the bell ring once. She froze.")])
        self.assertEqual(updated.chapters_digested, 3)


class TestReaderProgressAndCache(unittest.TestCase):
    def test_progress_bookmarks_and_cache_roundtrip(self):
        ctx, _tmp = _ctx()
        reader = StoryReader(ctx)
        # follow a Gutenberg URL with mocked network
        with mock.patch("urllib.request.urlopen",
                        side_effect=_gutenberg_urlopen):
            story = reader.follow("https://www.gutenberg.org/ebooks/84")
        self.assertEqual(story.title, "Frankenstein")
        reader.set_progress(story.slug, 1)
        prog = reader.get_progress(story.slug)
        self.assertEqual(prog["chapter"], 1)
        reader.add_bookmark(story.slug, 1, 0, "start")
        marks = reader.list_bookmarks(story.slug)
        self.assertEqual(len(marks), 1)
        self.assertEqual(marks[0]["label"], "start")
        # cache export → fresh context import
        out = cache_sync.export_cache(
            ctx, story.slug,
            os.path.join(_tmp, "frank-cache.zip"))
        self.assertTrue(os.path.exists(out["path"]))
        self.assertGreater(out["files"], 0)
        ctx2, _tmp2 = _ctx()
        imported = cache_sync.import_cache(ctx2, out["path"])
        self.assertEqual(imported["slug"], story.slug)
        self.assertGreater(imported["imported_files"], 0)
        # traversal safety: a hostile member is skipped, never written
        import zipfile
        evil = os.path.join(_tmp, "evil.zip")
        with zipfile.ZipFile(evil, "w") as zf:
            zf.writestr("story.json",
                        json.dumps({"slug": "evil-story"}))
            zf.writestr("../evil.txt", "pwned")
        res = cache_sync.import_cache(ctx2, evil)
        self.assertEqual(res["skipped_files"], 1)
        self.assertFalse(os.path.exists(
            os.path.join(_tmp2, "evil.txt")))


class TestSerialManager(unittest.TestCase):
    def test_serial_start_and_next_chapter(self):
        ctx, _tmp = _ctx()
        mgr = SerialManager(ctx)
        started = mgr.start_serial(
            "A scribe in a harbour town finds a map that rewrites itself.",
            genre="fantasy",
            for_whom={"name": "Ada", "relationship": "friend",
                      "occasion": "birthday"})
        slug = started["slug"]
        self.assertEqual(started["kind"], "serial")
        self.assertEqual(started["for_whom"]["name"], "Ada")
        nxt = mgr.write_next(slug)
        self.assertGreater(nxt.get("number", 0), 1)
        self.assertGreater(nxt.get("words", 0), 0)
        self.assertIn("drift", nxt)
        status = mgr.status(slug)
        self.assertEqual(status["for_whom"]["name"], "Ada")
        self.assertGreaterEqual(len(status.get("drift_log", [])), 1)

    def test_novel_start(self):
        ctx, _tmp = _ctx()
        mgr = SerialManager(ctx)
        started = mgr.start_novel(
            "A lighthouse keeper keeps a log of ships that never arrive.",
            genre="mystery")
        self.assertEqual(started["kind"], "novel")
        self.assertGreater(started["chapter_1"]["words"], 0)


if __name__ == "__main__":
    unittest.main()
