"""GLOBAL coverage contract: sources span the world, never one country.

Owner correction 2026-10-17 ("Not limited to Nigeria"): the multi-source
expansion leaned Nigerian/African.  Nigeria/Africa is ONE region among
many — /play, /news and /search must cover every region: Americas,
Europe, Asia, Africa, Middle East, Oceania.  No live network — all HTTP
is faked.
"""

import unittest
from unittest import mock

from nomorals.agents.news import (
    DEFAULT_CATEGORIES,
    FEED_CATEGORIES,
    FEED_REGIONS,
    NEWS_GLOBAL_REGIONS,
    news_covered_regions,
)
from nomorals.media.sources import (
    GLOBAL_REGIONS,
    MUSIC_SOURCE_REGIONS,
    MUSIC_SEARCH_SOURCES,
    MusicBrainzSource,
    OpenverseSource,
    WikimediaSource,
    music_covered_regions,
)
from nomorals.tools import web as webmod

SIX_REGIONS = {"americas", "europe", "asia", "africa", "middle_east",
               "oceania"}


class PlayGlobalCoverageTests(unittest.TestCase):
    def test_chain_has_sixteen_sources(self):
        self.assertGreaterEqual(len(MUSIC_SEARCH_SOURCES), 16,
                                MUSIC_SEARCH_SOURCES)

    def test_new_global_sources_present(self):
        for name in ("youtube-music-search", "musicbrainz-search",
                     "openverse-search", "wikimedia-search"):
            self.assertIn(name, MUSIC_SEARCH_SOURCES)

    def test_every_chain_entry_has_a_region(self):
        self.assertEqual(set(MUSIC_SOURCE_REGIONS),
                         set(MUSIC_SEARCH_SOURCES))

    def test_play_covers_all_six_regions(self):
        self.assertEqual(set(GLOBAL_REGIONS), SIX_REGIONS)
        self.assertTrue(SIX_REGIONS <= music_covered_regions(),
                        music_covered_regions())

    def test_chain_is_not_nigeria_only(self):
        # Non-African sources must exist — Nigeria/Africa is one region.
        non_african = [s for s in MUSIC_SEARCH_SOURCES
                       if MUSIC_SOURCE_REGIONS[s] != "africa"
                       and MUSIC_SOURCE_REGIONS[s] != "local"]
        self.assertGreaterEqual(len(non_african), 10, non_african)
        for name in ("youtube-music-search", "soundcloud-search",
                     "musicbrainz-search", "openverse-search"):
            self.assertIn(name, non_african)

    def test_musicbrainz_parse(self):
        payload = {"recordings": [
            {"id": "abc-1", "title": "Goodbye",
             "artist-credit": [{"artist": {"name": "Someone"}}]},
            {"id": "abc-2", "title": "Hola",
             "artist-credit": [{"artist": {"name": "Alguien"}}]},
        ]}
        rows = MusicBrainzSource().parse(payload, "goodbye")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].title, "Goodbye")
        self.assertEqual(rows[0].artist, "Someone")
        self.assertFalse(rows[0].downloadable)
        self.assertIn("musicbrainz.org/recording/abc-1", rows[0].url)

    def test_openverse_parse(self):
        payload = {"results": [
            {"title": "Goodbye", "creator": "Someone",
             "url": "https://example.com/goodbye.mp3",
             "foreign_landing_url": "https://example.com/landing"},
        ]}
        rows = OpenverseSource().parse(payload, "goodbye")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].url, "https://example.com/goodbye.mp3")
        self.assertTrue(rows[0].downloadable)

    def test_wikimedia_parse(self):
        payload = {"query": {"pages": {
            "1": {"title": "File:Goodbye.ogg",
                  "imageinfo": [{"url": "https://example.com/g.ogg",
                                 "mime": "audio/ogg"}]},
            "2": {"title": "File:Photo.jpg",
                  "imageinfo": [{"url": "https://example.com/p.jpg",
                                 "mime": "image/jpeg"}]},
        }}}
        rows = WikimediaSource().parse(payload, "goodbye")
        self.assertEqual(len(rows), 1)  # the image is skipped
        self.assertEqual(rows[0].url, "https://example.com/g.ogg")
        self.assertTrue(rows[0].downloadable)

    def test_new_sources_degrade_to_empty_on_failure(self):
        for cls in (MusicBrainzSource, OpenverseSource, WikimediaSource):
            with mock.patch.object(cls, "_get", return_value=""):
                self.assertEqual(cls().search("goodbye"), [])
            with mock.patch.object(cls, "_get",
                                   side_effect=RuntimeError("down")):
                self.assertEqual(cls().search("goodbye"), [])


class NewsGlobalCoverageTests(unittest.TestCase):
    def test_regional_categories_exist(self):
        for cat in ("latam", "middle_east", "oceania", "americas"):
            self.assertIn(cat, FEED_CATEGORIES)
            self.assertTrue(FEED_CATEGORIES[cat], f"{cat} is empty")

    def test_every_category_has_a_region(self):
        self.assertEqual(set(FEED_REGIONS), set(FEED_CATEGORIES))

    def test_default_digest_covers_all_six_regions(self):
        self.assertEqual(set(NEWS_GLOBAL_REGIONS), SIX_REGIONS)
        self.assertTrue(SIX_REGIONS <= news_covered_regions(),
                        news_covered_regions())

    def test_default_categories_include_every_region(self):
        for cat in ("latam", "middle_east", "oceania", "americas",
                    "europe", "asia", "africa", "nigeria"):
            self.assertIn(cat, DEFAULT_CATEGORIES)

    def test_new_regions_have_feeds_in_default(self):
        from nomorals.agents.news import DEFAULT_FEEDS
        urls = [u for _, u in DEFAULT_FEEDS]
        self.assertTrue(any("elpais" in u for u in urls))      # latam
        self.assertTrue(any("arabnews" in u for u in urls))    # middle east
        self.assertTrue(any("smh.com.au" in u for u in urls))  # oceania
        self.assertTrue(any("lanacion" in u for u in urls))    # americas


class SearchGlobalCoverageTests(unittest.TestCase):
    def test_ten_engines(self):
        self.assertGreaterEqual(len(webmod._SEARCH_ENGINES), 10,
                                webmod._SEARCH_ENGINES)
        for name in ("brave", "qwant"):
            self.assertIn(name, webmod._SEARCH_ENGINES)

    def test_no_region_pinning_in_engine_urls(self):
        # GLOBAL: no engine URL may carry a region pin.
        for engine in webmod._SEARCH_ENGINES:
            url = webmod._engine_url(engine, "test")
            for pin in ("cc=", "gl=", "hl=", "region=", "kl="):
                self.assertNotIn(pin, url, f"{engine}: {url}")

    def test_brave_qwant_parse_via_generic(self):
        html = ('<a href="https://example.com/r1">Result One</a>'
                '<a href="https://search.brave.com/x">chrome</a>')
        out = webmod._parse_generic(html)
        urls = [r["url"] for r in out]
        self.assertIn("https://example.com/r1", urls)
        self.assertNotIn("https://search.brave.com/x", urls)


if __name__ == "__main__":
    unittest.main()
