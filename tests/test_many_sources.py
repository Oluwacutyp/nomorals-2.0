"""Many-sources contract: /play, /news, /search must use MANY sources,
never 1-2.  Owner complaint 2026-10-17: every search command used only
one or two sources.  These tests pin the breadth + the visibility +
the continue-on-error chain.  No live network — all HTTP is faked."""

import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.media import resolver as resolver_mod
from nomorals.media.resolver import SourceResolver
from nomorals.media.sources import MUSIC_SEARCH_SOURCES
from nomorals.agents.news import FEED_CATEGORIES, NewsAgent, parse_feed
from nomorals.search.federated import federated_search
from nomorals.search.model import SearchResult
from nomorals.tools import web as webmod


# ── /play: MANY sources ────────────────────────────────────────────────

class PlayManySourcesTests(unittest.TestCase):
    def test_play_text_chain_has_more_than_ten_sources(self):
        self.assertGreater(len(MUSIC_SEARCH_SOURCES), 10,
                           MUSIC_SEARCH_SOURCES)
        self.assertIn("workspace-scan", MUSIC_SEARCH_SOURCES)
        self.assertIn("netnaija-search", MUSIC_SEARCH_SOURCES)
        self.assertIn("notjustok-search", MUSIC_SEARCH_SOURCES)
        self.assertIn("tooxclusive-search", MUSIC_SEARCH_SOURCES)
        self.assertIn("soundcloud-search", MUSIC_SEARCH_SOURCES)
        self.assertIn("bandcamp-search", MUSIC_SEARCH_SOURCES)
        self.assertIn("youtube-search", MUSIC_SEARCH_SOURCES)
        self.assertIn("archiveorg-search", MUSIC_SEARCH_SOURCES)
        self.assertIn("itunes-search", MUSIC_SEARCH_SOURCES)
        self.assertIn("deezer-search", MUSIC_SEARCH_SOURCES)
        self.assertIn("boomplay-search", MUSIC_SEARCH_SOURCES)
        self.assertIn("spotify-search", MUSIC_SEARCH_SOURCES)

    def test_unknown_title_tries_all_sources(self):
        # Every source in the chain is attempted on total failure.
        ctx = SimpleNamespace(settings=SimpleNamespace(workspace_dir="/nonexistent"))
        with mock.patch.object(resolver_mod, "probe",
                               side_effect=RuntimeError("no yt-dlp")), \
             mock.patch("nomorals.media.playback.PlaybackEngine._youtube_search_id",
                        side_effect=RuntimeError("no yt-dlp")), \
             mock.patch("nomorals.media.sources.NetNaijaSource.search", return_value=[]), \
             mock.patch("nomorals.media.sources.NotJustOkSource.search", return_value=[]), \
             mock.patch("nomorals.media.sources.TooXclusiveSource.search", return_value=[]), \
             mock.patch("nomorals.media.sources.BandcampSource.search", return_value=[]), \
             mock.patch("nomorals.media.sources.ArchiveOrgSource.search", return_value=[]), \
             mock.patch("nomorals.media.sources.ITunesSource.search", return_value=[]), \
             mock.patch("nomorals.media.sources.DeezerSource.search", return_value=[]), \
             mock.patch("nomorals.media.sources.BoomplaySource.search", return_value=[]):
            out = SourceResolver(ctx).resolve("xyqz totally unknown 999")
        self.assertFalse(out.ok)
        tried = " ".join(out.attempts)
        for name in MUSIC_SEARCH_SOURCES:
            self.assertIn(name, tried, tried)

    def test_one_source_failing_does_not_end_play_search(self):
        # NetNaija raising must not stop SoundCloud from resolving.
        ctx = SimpleNamespace(settings=SimpleNamespace(workspace_dir="/nonexistent"))
        fake_sc = SimpleNamespace(
            search_tracks=lambda q, limit=1: [{
                "permalink_url": "https://soundcloud.com/a/goodbye",
                "title": "Goodbye", "artist": "Someone",
                "duration_ms": 180000, "policy": "ALLOW"}])
        with mock.patch("nomorals.media.sources.NetNaijaSource.search",
                        side_effect=RuntimeError("netnaija down")), \
             mock.patch("nomorals.media.sources.NotJustOkSource.search",
                        return_value=[]), \
             mock.patch("nomorals.media.sources.TooXclusiveSource.search",
                        return_value=[]):
            out = SourceResolver(ctx, soundcloud=fake_sc).resolve("goodbye")
        self.assertTrue(out.ok)
        self.assertEqual(out.source_name, "SoundCloud search")

    def test_search_candidates_aggregate_many_sources(self):
        ctx = SimpleNamespace(settings=SimpleNamespace(workspace_dir="/nonexistent"))
        cand = SimpleNamespace(url="https://x/y", title="T", artist="A",
                               duration=0.0, direct_url="", extra={})
        with mock.patch("nomorals.media.sources.NetNaijaSource.search",
                        return_value=[cand]), \
             mock.patch("nomorals.media.sources.NotJustOkSource.search",
                        return_value=[]), \
             mock.patch("nomorals.media.sources.TooXclusiveSource.search",
                        return_value=[]), \
             mock.patch("nomorals.media.sources.BandcampSource.search",
                        return_value=[]), \
             mock.patch("nomorals.media.sources.ArchiveOrgSource.search",
                        return_value=[]), \
             mock.patch("nomorals.media.sources.ITunesSource.search",
                        return_value=[]), \
             mock.patch("nomorals.media.sources.DeezerSource.search",
                        return_value=[]), \
             mock.patch("nomorals.media.sources.BoomplaySource.search",
                        return_value=[]), \
             mock.patch.object(SourceResolver, "_soundcloud_search_many",
                               return_value=[]), \
             mock.patch.object(SourceResolver, "_youtube_search_many",
                               return_value=[]):
            out = SourceResolver(ctx).search_candidates("goodbye", limit=8)
        self.assertTrue(any(r.source_name == "NetNaija search" for r in out))

    def test_itunes_deezer_are_preview_only(self):
        it = resolver_mod  # noqa: F841
        from nomorals.media.sources import ITunesSource, DeezerSource
        with mock.patch.object(ITunesSource, "_get", return_value=(
                '{"results": [{"trackName": "Goodbye", "artistName": "A",'
                ' "previewUrl": "https://example.com/p.m4a",'
                ' "trackViewUrl": "https://example.com/t"}]}')):
            rows = ITunesSource().search("goodbye")
        self.assertTrue(rows[0].extra["preview_only"])
        with mock.patch.object(DeezerSource, "_get", return_value=(
                '{"data": [{"title": "Goodbye", "preview": '
                '"https://example.com/p.mp3", "link": "https://example.com/t",'
                ' "artist": {"name": "A"}, "duration": 30}]}')):
            rows = DeezerSource().search("goodbye")
        self.assertTrue(rows[0].extra["preview_only"])


# ── /news: MANY outlets + visible breadth ──────────────────────────────

class NewsManySourcesTests(unittest.TestCase):
    def test_feed_count_is_many(self):
        total = sum(len(v) for v in FEED_CATEGORIES.values())
        self.assertGreaterEqual(total, 60, total)
        self.assertIn("africa", FEED_CATEGORIES)
        self.assertIn("entertainment", FEED_CATEGORIES)
        cats = {c for c, _ in
                [(k, None) for k in FEED_CATEGORIES]}
        for want in ("nigeria", "africa", "world", "tech", "business",
                     "science", "sports", "usa", "europe", "asia", "wires"):
            self.assertIn(want, cats)

    def test_run_reports_sources_tried_and_ok(self):
        ctx = SimpleNamespace(settings=SimpleNamespace(news=SimpleNamespace(
            feeds="", cap=10)))
        agent = NewsAgent(ctx)
        feed_xml = (
            '<?xml version="1.0"?><rss version="2.0"><channel>'
            '<item><title>Same story everywhere</title>'
            '<link>https://example.com/1</link>'
            '<description>Body one.</description></item>'
            '</channel></rss>')
        with mock.patch.object(agent, "_feeds", return_value=[
                ("BBC News", "https://example.com/bbc.xml"),
                ("Punch", "https://example.com/punch.xml"),
                ("Dead Feed", "https://example.com/dead.xml")]), \
             mock.patch.object(agent, "_client") as client:
            def _get(url):
                if "dead" in url:
                    raise RuntimeError("feed down")
                return SimpleNamespace(text=feed_xml)
            client.get.side_effect = _get
            report = agent.run()
        self.assertEqual(report["sources_tried"], 3)
        self.assertIn("BBC News", report["sources_ok"])
        self.assertIn("Punch", report["sources_ok"])
        # Same story on 2 feeds dedupes to ONE item (title dedupe).
        self.assertEqual(report["items"], 1)
        self.assertTrue(any("Dead Feed" in e for e in report["errors"]))

    def test_parse_feed_basics(self):
        items = parse_feed(
            '<?xml version="1.0"?><rss version="2.0"><channel>'
            '<item><title>T</title><link>https://example.com/t</link>'
            '<description>D</description></item></channel></rss>')
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["title"], "T")


# ── /search: MANY engines, aggregated ─────────────────────────────────

class WebSearchManyEnginesTests(unittest.TestCase):
    def test_engine_list_has_more_than_five(self):
        self.assertGreater(len(webmod._SEARCH_ENGINES), 5)
        for want in ("duckduckgo", "bing", "mojeek", "ecosia", "yahoo",
                     "startpage", "yandex"):
            self.assertIn(want, webmod._SEARCH_ENGINES)

    def test_run_engines_aggregates_all_and_continues_on_failure(self):
        def fake_get(url, **kwargs):
            if "duckduckgo.com" in url:
                raise RuntimeError("ddg down")
            if "bing.com" in url:
                return SimpleNamespace(text=(
                    '<li class="b_algo"><h2><a href="https://example.com/b">'
                    'Bing title here</a></h2></li>'))
            if "mojeek.com" in url:
                return SimpleNamespace(text=(
                    '<a class="title" href="https://example.com/m">'
                    'Mojeek title here</a>'))
            return SimpleNamespace(text="<html><body>no links</body></html>")
        client = SimpleNamespace(get=fake_get)
        results, used, failed = webmod._run_engines(client, "goodbye")
        urls = [r["url"] for r in results]
        self.assertIn("https://example.com/b", urls)
        self.assertIn("https://example.com/m", urls)
        self.assertIn("bing", used)
        self.assertIn("mojeek", used)
        self.assertIn("duckduckgo", failed)

    def test_generic_parser_drops_engine_chrome(self):
        html = (
            '<a href="https://www.ecosia.org/search?q=x">Search again</a>'
            '<a href="https://example.com/real">A real result title here</a>'
            '<a href="/settings">Settings page link</a>')
        out = webmod._parse_generic(html)
        self.assertEqual([r["url"] for r in out],
                         ["https://example.com/real"])


# ── federated: one bad source never kills the search ──────────────────

class _Stub:
    def __init__(self, name, hits=None, boom=None):
        self.name = name
        self._hits = hits or []
        self._boom = boom

    def probe(self):
        return None

    def search(self, query, *, limit, since=None, before=None):
        if self._boom:
            raise self._boom
        return self._hits[:limit]


def _hit(source, title):
    return SearchResult(query="q", title=title, snippet="s",
                        source=source, type="web", raw_score=1.0,
                        provenance={"url": f"https://example.com/{title}"},
                        source_id=f"{source}:{title}")


class FederatedContinueTests(unittest.TestCase):
    def test_one_bad_source_other_still_returns(self):
        resp = federated_search(
            "q", adapters={"memory": _Stub("memory", boom=RuntimeError("db gone")),
                           "wisdom": _Stub("wisdom", hits=[_hit("wisdom", "w1")])},
            sources=["memory", "wisdom"], limit=5)
        self.assertIn("memory", resp.sources_skipped)
        self.assertIn("db gone", resp.sources_skipped["memory"])
        self.assertEqual([h.source for h in resp.hits], ["wisdom"])

    def test_all_sources_failing_returns_empty_not_raise(self):
        resp = federated_search(
            "q", adapters={"memory": _Stub("memory", boom=RuntimeError("x"))},
            sources=["memory"], limit=5)
        self.assertEqual(resp.hits, [])
        self.assertIn("memory", resp.sources_skipped)


if __name__ == "__main__":
    unittest.main()
