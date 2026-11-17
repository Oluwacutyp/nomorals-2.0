"""MP3.pm / MP3Juice wiring — the owner's own sources, wired correctly.

Failing-first for the music fix: the owner's bac87ee4 classes crash at
runtime (``json`` never imported, ``_UA`` undefined), the ``__all__``
entries name strategy keys instead of classes, the resolver calls a
``_s_blog_search_many`` that does not exist (and lists each strategy
twice), and neither source is in MUSIC_SEARCH_SOURCES / SOURCE_CHAIN /
the candidate sweep. Placement is the chain's own rule (downloadable
sources after the regional blogs, before SoundCloud) — NOT first.
"""

import json
import unittest

from nomorals.media import sources as _sources
from nomorals.media.resolver import SourceResolver
from nomorals.media.sources import (
    BROWSER_UA,
    MUSIC_SEARCH_SOURCES,
    MUSIC_SOURCE_DOMAINS,
    MUSIC_SOURCE_REGIONS,
    SOURCE_CHAIN,
    MP3JuiceSource,
    MP3PMSource,
    SourceCandidate,
)

_RESOLVER_PATH = "nomorals/media/resolver.py"


class Mp3pmSearchTests(unittest.TestCase):
    def test_search_parses_json_response(self):
        payload = json.dumps({"songs": [
            {"url": "https://a-1.mp3.pm/song/7-hello-world/",
             "title": "Hello-World", "artist": "Some Artist"},
        ]})
        src = MP3PMSource()
        src._get = lambda url: payload  # noqa: SLF001 - seam under test
        found = src.search("hello world", limit=5)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].title, "Hello World")
        self.assertEqual(found[0].artist, "Some Artist")
        self.assertEqual(found[0].source, "mp3pm")


class Mp3juicePostTests(unittest.TestCase):
    def test_post_sends_browser_user_agent(self):
        import urllib.request

        seen = {}

        class _FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b"<html></html>"

        real_urlopen = urllib.request.urlopen

        def _capture(req, timeout=None):
            seen["ua"] = req.get_header("User-agent")
            seen["url"] = req.full_url
            return _FakeResp()

        urllib.request.urlopen = _capture
        try:
            MP3JuiceSource()._post("/", {"query": "x"})  # noqa: SLF001
        finally:
            urllib.request.urlopen = real_urlopen
        self.assertEqual(seen["ua"], BROWSER_UA)
        self.assertTrue(seen["url"].endswith("/"))


class ExportsTests(unittest.TestCase):
    def test_every_all_entry_is_a_real_attribute(self):
        missing = [n for n in _sources.__all__
                   if not hasattr(_sources, n)]
        self.assertEqual(missing, [])

    def test_both_classes_exported(self):
        self.assertIn("MP3PMSource", _sources.__all__)
        self.assertIn("MP3JuiceSource", _sources.__all__)


class ChainMembershipTests(unittest.TestCase):
    def test_both_classes_are_instantiable(self):
        # BaseScrapeSource is abstract over search_url / parse_search /
        # extract_download — a class that never implements them cannot
        # be constructed at all, so no chain entry could ever run it.
        pm = MP3PMSource()
        juice = MP3JuiceSource()
        self.assertTrue(pm.search_url("hello").startswith("https://"))
        self.assertEqual(juice.search_url("hello"),
                         "https://service-et-reception.fr/")
        rows = pm.parse_search(json.dumps({"songs": [
            {"url": "https://a-1.mp3.pm/song/7-hello-world/",
             "title": "Hello-World"}]}))
        self.assertEqual(rows, [("Hello World",
                                 "https://a-1.mp3.pm/song/7-hello-world/")])
        self.assertEqual(
            pm.extract_download(
                '<a href="https://x.mp3.pm/f/song.mp3">dl</a>',
                "https://a-1.mp3.pm/song/7-hello-world/"),
            "https://x.mp3.pm/f/song.mp3")
        rows = juice.parse_search(
            '<div data-id="9"></div><h3>Hello <b>World</b></h3>')
        self.assertEqual(rows, [("Hello World",
                                 "https://service-et-reception.fr/?id=9")])

    def test_both_have_a_region(self):
        # test_global_sources pins the REGIONS map == the chain set.
        # Both are global (neither is a regional blog): mp3.pm is a
        # global archive, the MP3Juice clone serves YouTube results.
        self.assertEqual(MUSIC_SOURCE_REGIONS["mp3pm-search"], "global")
        self.assertEqual(MUSIC_SOURCE_REGIONS["mp3juice-search"], "global")

    def test_both_domains_are_registered_seeds(self):
        # test_pass88 pins every chain strategy's domain as a seed in
        # the source registry — otherwise the health loop reports the
        # chain entry as missing/unknown and the job never covers it.
        from nomorals.media.source_registry import _seed

        reg = _seed()
        for strategy in ("mp3pm-search", "mp3juice-search"):
            domain = MUSIC_SOURCE_DOMAINS[strategy]
            desc = reg.for_domain(domain)
            self.assertIsNotNone(desc, f"{strategy} ({domain}) not registered")
            self.assertIn("music-search", desc.kinds)

    def test_both_in_search_sources_and_chain(self):
        self.assertIn("mp3pm-search", MUSIC_SEARCH_SOURCES)
        self.assertIn("mp3juice-search", MUSIC_SEARCH_SOURCES)
        chain_names = [name for name, _ in SOURCE_CHAIN]
        self.assertIn("mp3pm-search", chain_names)
        self.assertIn("mp3juice-search", chain_names)

    def test_order_after_regional_blogs_not_first(self):
        # Owner order 2026-11-17: mdundo moved down, after the big
        # globals — its loose search surfaced irrelevant tracks too
        # early. mp3pm/mp3juice keep their 2026-11-16 placement (after
        # the regional blogs, before SoundCloud).
        self.assertEqual(MUSIC_SEARCH_SOURCES[0], "workspace-scan")
        mdundo = MUSIC_SEARCH_SOURCES.index("mdundo-search")
        self.assertLess(MUSIC_SEARCH_SOURCES.index("mp3pm-search"),
                        mdundo)
        self.assertLess(MUSIC_SEARCH_SOURCES.index("mp3juice-search"),
                        mdundo)
        self.assertLess(MUSIC_SEARCH_SOURCES.index("mp3pm-search"),
                        MUSIC_SEARCH_SOURCES.index("soundcloud-search"))
        self.assertGreater(mdundo,
                           MUSIC_SEARCH_SOURCES.index(
                               "youtube-music-search"))
        self.assertLess(mdundo,
                        MUSIC_SEARCH_SOURCES.index("archiveorg-search"))


class ResolverWiringTests(unittest.TestCase):
    def _resolve_text_span(self):
        with open(_RESOLVER_PATH, encoding="utf-8") as fh:
            text = fh.read()
        start = text.index("def _resolve_text")
        end = text.index("def _s_blog_search", start)
        return text[start:end]

    def test_single_chain_lists_each_strategy_once(self):
        span = self._resolve_text_span()
        self.assertEqual(span.count('"mp3pm-search"'), 1)
        self.assertEqual(span.count('"mp3juice-search"'), 1)

    def test_no_call_to_nonexistent_search_many(self):
        with open(_RESOLVER_PATH, encoding="utf-8") as fh:
            text = fh.read()
        self.assertNotIn("_s_blog_search_many", text)

    def test_single_seam_resolves_via_mp3pm(self):
        real_search = MP3PMSource.search

        def _fake(self, query, limit=5):
            return [SourceCandidate(
                source="mp3pm", title="Hello World",
                artist="Some Artist",
                url="https://a-1.mp3.pm/song/7-hello-world/")]

        MP3PMSource.search = _fake
        try:
            got = SourceResolver._s_blog_search(
                None, "hello world", "MP3PMSource", "mp3pm")
        finally:
            MP3PMSource.search = real_search
        self.assertTrue(got.ok)
        self.assertEqual(got.title, "Hello World")
        self.assertEqual(got.kind, "mp3pm")

    def test_sweep_methods_exist_and_return_candidates(self):
        real_pm = MP3PMSource.search
        real_juice = MP3JuiceSource.search

        def _fake_pm(self, query, limit=5):
            return [SourceCandidate(
                source="mp3pm", title="Hello World",
                artist="Some Artist",
                url="https://a-1.mp3.pm/song/7-hello-world/")]

        def _fake_juice(self, query, limit=5):
            return [SourceCandidate(
                source="mp3juice", title="Hello World",
                artist="Some Artist",
                url="https://service-et-reception.fr/?id=9")]

        MP3PMSource.search = _fake_pm
        MP3JuiceSource.search = _fake_juice
        # _api_source_search_many never touches self, so an unbound
        # instance is enough — the seam is the class lookup, not state.
        resolver = SourceResolver.__new__(SourceResolver)
        try:
            pm = SourceResolver._mp3pm_search_many(resolver, "hello", 5)
            juice = SourceResolver._mp3juice_search_many(resolver, "hello", 5)
        finally:
            MP3PMSource.search = real_pm
            MP3JuiceSource.search = real_juice
        self.assertEqual(len(pm), 1)
        self.assertTrue(pm[0].downloadable)
        self.assertEqual(len(juice), 1)
        self.assertTrue(juice[0].downloadable)


if __name__ == "__main__":
    unittest.main()
