"""Source adapters: parsing on fixture HTML, no network.

Verifies the selectors mined live on 2026-10-09 against canned markup
shaped exactly like the real pages (freewebnovel.com, novelfull.net,
royalroad.com), plus block detection, the fetch-chain fallback wiring,
and adapter routing.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.books import sources
from nomorals.books.sources import (
    ADAPTERS, FreeWebNovelAdapter, GenericAdapter, NovelFullAdapter,
    PandaNovelAdapter, RoyalRoadAdapter, SourceBlocked, adapter_for_url,
)
from nomorals.tools.browser import parse_html


FWN_CHAPTER = """<html><head><title>Ch 36 | Free Web Novel</title></head><body>
<div class="main"><div class="txt "><div id="article">
<h4>Chapter 36 The Rarity</h4>
<p>Zorian's eyes shot open as pain erupted.</p>
<p>"Good morning, brother!" Kirielle chirped.</p>
<p>Visit freewebnovel.com for the best novel reading experience</p>
</div></div>
<div class="reader-page-nav">
<a href="/novel/genetic-ascension/chapter-35">Prev Chapter</a>
<a href="/novel/genetic-ascension/chapter-37">Next Chapter</a>
</div></div></body></html>"""

FWN_NOVEL = """<html><head>
<meta property="og:title" content="Genetic Ascension"/>
<meta property="og:novel:author" content="Awespec"/>
<meta property="og:novel:genre" content="Fantasy, Action"/>
<meta property="og:novel:status" content="OnGoing"/>
<meta property="og:description" content="The people of Earth have been summoned."/>
</head><body>
<a href="/novel/genetic-ascension/chapter-1">Read first</a>
<a href="/novel/genetic-ascension/chapter-2422">Chapter 2422: The Rarity</a>
<a href="/novel/genetic-ascension/chapter-2421">Chapter 2421: Direction</a>
</body></html>"""

NF_CHAPTER = """<html><head><title>Chapter 1 | NovelFull</title></head><body>
<div class="txt fwn-reader-txt"><div id="chapter-content" class="chapter-c">
<p>Translator: StarveCleric</p>
<p>"Swindler! Great swindler!"</p>
<p>An enraged roar echoed through the hall.</p>
</div></div>
<a href="/library-of-heavens-path/chapter-2-shameless.html">Next Chapter</a>
</body></html>"""

NF_NOVEL = """<html><head>
<meta property="og:title" content="Library of Heaven's Path"/>
<meta property="og:novel:author" content="Heng Sao Tian Xia"/>
<meta property="og:novel:status" content="Completed"/>
</head><body>
<div id="list-chapter" data-total-chapters="2271" data-total-page="57"></div>
<ul><li><a class="con" title="Chapter 1: Swindler"
href="/library-of-heavens-path/chapter-1-swindler.html">
<span class="chapter-text">Chapter 1: Swindler</span></a></li></ul>
</body></html>"""

RR_CHAPTER = """<html><head><title>Ch 1 | Royal Road</title></head><body>
<div class="chapter-inner chapter-content">
<h4>Chapter 001 &nbsp; Good Morning Brother</h4>
<p>Zorian's eyes abruptly shot open.</p>
<p>"Good morning, brother!"</p>
</div></body></html>"""

RR_NOVEL = """<html><head><title>Mother of Learning | Royal Road</title></head>
<body><h1>Mother of Learning</h1>
<table><tr><td><a href="/fiction/21220/mother-of-learning/chapter/301778/1-good-morning-brother">1. Good Morning Brother</a></td></tr>
<tr><td><a href="/fiction/21220/mother-of-learning/chapter/301779/2-in-which-zorian">2. In Which Zorian</a></td></tr>
</table></body></html>"""


class _FakeFetcher:
    """Serves canned pages instead of the network."""

    def __init__(self, pages: dict[str, str]) -> None:
        self.pages = pages

    def get(self, url: str, **_: object):
        for key, html in self.pages.items():
            if url.startswith(key):
                return url, parse_html(html), html
        raise sources.SourceError(f"no fixture for {url}")


class TestFreeWebNovel(unittest.TestCase):
    def setUp(self) -> None:
        self.ad = FreeWebNovelAdapter(fetcher=_FakeFetcher({
            "https://freewebnovel.com/novel/genetic-ascension/chapter-36":
                FWN_CHAPTER,
        }))

    def test_fetch_chapter(self) -> None:
        ch = self.ad.fetch_chapter(
            "https://freewebnovel.com/novel/genetic-ascension/chapter-36")
        self.assertEqual(ch.number, 36)
        self.assertIn("Chapter 36", ch.title)
        # junk injection stripped, prose kept
        self.assertEqual(len(ch.paragraphs), 2)
        self.assertTrue(all("freewebnovel.com for the best" not in p
                            for p in ch.paragraphs))
        self.assertTrue(ch.prev_url.endswith("chapter-35"))
        self.assertTrue(ch.next_url.endswith("chapter-37"))
        self.assertEqual(ch.source, "freewebnovel")

    def test_novel_meta(self) -> None:
        ad = FreeWebNovelAdapter(fetcher=_FakeFetcher({
            "https://freewebnovel.com/novel/genetic-ascension": FWN_NOVEL,
        }))
        meta = ad.novel("https://freewebnovel.com/novel/genetic-ascension")
        self.assertEqual(meta.title, "Genetic Ascension")
        self.assertEqual(meta.author, "Awespec")
        self.assertIn("Fantasy", meta.genres)
        self.assertEqual(meta.status, "OnGoing")
        self.assertEqual(meta.total_chapters, 2422)

    def test_matches(self) -> None:
        self.assertTrue(self.ad.matches(
            "https://freewebnovel.com/novel/x/chapter-1"))
        self.assertFalse(self.ad.matches("https://novelfull.net/x.html"))


class TestNovelFull(unittest.TestCase):
    def test_fetch_chapter(self) -> None:
        ad = NovelFullAdapter(fetcher=_FakeFetcher({
            "https://novelfull.net/library-of-heavens-path/chapter-1-swindler.html":
                NF_CHAPTER,
        }))
        ch = ad.fetch_chapter(
            "https://novelfull.net/library-of-heavens-path/chapter-1-swindler.html")
        self.assertEqual(ch.number, 1)
        self.assertEqual(len(ch.paragraphs), 3)  # translator line kept
        self.assertTrue(ch.next_url.endswith("chapter-2-shameless.html"))
        self.assertEqual(ch.prev_url, "")

    def test_novel_and_chapter_list(self) -> None:
        ad = NovelFullAdapter(fetcher=_FakeFetcher({
            "https://novelfull.net/library-of-heavens-path.html": NF_NOVEL,
        }))
        meta = ad.novel("https://novelfull.net/library-of-heavens-path.html")
        self.assertEqual(meta.total_chapters, 2271)
        self.assertEqual(meta.status, "Completed")
        links = ad._chapter_links(parse_html(NF_NOVEL),
                                  "https://novelfull.net/library-of-heavens-path.html")
        self.assertEqual(links[0][0], 1)
        self.assertIn("Swindler", links[0][1])


class TestRoyalRoad(unittest.TestCase):
    def test_fetch_chapter(self) -> None:
        ad = RoyalRoadAdapter(fetcher=_FakeFetcher({
            "https://www.royalroad.com/fiction/21220/mother-of-learning/chapter/301778/1-good-morning-brother":
                RR_CHAPTER,
            "https://www.royalroad.com/fiction/21220/mother-of-learning":
                RR_NOVEL,
        }))
        ch = ad.fetch_chapter(
            "https://www.royalroad.com/fiction/21220/mother-of-learning"
            "/chapter/301778/1-good-morning-brother")
        self.assertEqual(ch.number, 1)
        self.assertEqual(len(ch.paragraphs), 2)
        # prev/next derived from the fiction's chapter order
        self.assertIn("chapter/301779", ch.next_url)
        self.assertEqual(ch.prev_url, "")

    def test_novel(self) -> None:
        ad = RoyalRoadAdapter(fetcher=_FakeFetcher({
            "https://www.royalroad.com/fiction/21220/mother-of-learning":
                RR_NOVEL,
        }))
        meta = ad.novel(
            "https://www.royalroad.com/fiction/21220/mother-of-learning")
        self.assertEqual(meta.title, "Mother of Learning")
        self.assertEqual(meta.total_chapters, 2)


class TestRoutingAndFallback(unittest.TestCase):
    def test_adapter_for_url(self) -> None:
        self.assertIsInstance(
            adapter_for_url("https://freewebnovel.com/novel/x"),
            FreeWebNovelAdapter)
        self.assertIsInstance(
            adapter_for_url("https://novelfull.net/x.html"),
            NovelFullAdapter)
        self.assertIsInstance(
            adapter_for_url("https://readnovelfull.com/x.html"),
            NovelFullAdapter)
        self.assertIsInstance(
            adapter_for_url("https://www.royalroad.com/fiction/1/x"),
            RoyalRoadAdapter)
        self.assertIsInstance(
            adapter_for_url("https://pandanovel.com/novel/x"),
            PandaNovelAdapter)
        # unknown aggregator → honestly-flagged generic adapter
        generic = adapter_for_url("https://some-obscure-reader.net/novel/x")
        self.assertIsInstance(generic, GenericAdapter)
        self.assertFalse(generic.verified)

    def test_pandanovel_flagged_unverified(self) -> None:
        self.assertFalse(PandaNovelAdapter().verified)
        self.assertTrue(FreeWebNovelAdapter().verified)

    def test_generic_adapter_scores_candidates(self) -> None:
        body = " ".join(["Real prose sentence carrying the story forward."] * 12)
        ad = GenericAdapter(name="test", domains=("example.com",),
                            fetcher=_FakeFetcher({
                                "https://example.com/ch1": f"""<html><body>
<div id="sidebar"><p>menu</p></div>
<div class="chapter-content"><p>{body}</p>
<p>{body}</p></div>
</body></html>""",
                            }))
        ch = ad.fetch_chapter("https://example.com/ch1")
        self.assertEqual(len(ch.paragraphs), 2)
        self.assertIn("Real prose", ch.paragraphs[0])

    def test_block_detection(self) -> None:
        self.assertTrue(sources._looks_blocked(
            "<html><head><title>Just a moment...</title></head>"
            "<body>" + "x" * 900 + "</body></html>"))
        self.assertFalse(sources._looks_blocked(
            "<html><head><title>Chapter 1</title></head><body>"
            + "<p>A full paragraph of genuine chapter prose, long enough "
              "to clear the minimum-body heuristic.</p>" * 40
            + "</body></html>"))

    def test_all_adapters_registered(self) -> None:
        names = {a.name for a in ADAPTERS}
        self.assertTrue({"freewebnovel", "novelfull", "royalroad",
                         "pandanovel"} <= names)


if __name__ == "__main__":
    unittest.main()
