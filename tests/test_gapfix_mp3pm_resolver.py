"""mp3.pm dead search endpoint + resolver grown-source lambda arity.

Owner live report 2026-11-17 (EC2 log screenshot):

* ``mp3pm: GET failed for https://mp3.pm/api/search?q=tell+me+one+lie:
  HTTP Error 404: Not Found`` — the JSON endpoint the source reads
  does not exist.  The site's real search XHR is ``POST
  /public/api.search.php`` returning a redirect URL, and the results
  page itself (``GET /search?q=``) is server-rendered with full song
  rows — title, artist, direct listen/download URLs.
* ``resolver candidates: <lambda> failed: ... takes from 0 to 1
  positional arguments but 2 were given`` — the grown-source entry in
  ``search_candidates_detailed`` is a 0-1-arg lambda the sweep calls
  with ``(q, limit)``, so every grown source fails the detailed sweep.

All fetches are stubbed — no network.  The row fixture is the real
``<li class="cplayer-sound-item">`` recorded from the live search
page for "tell me one lie" (2026-11-17), verbatim.
"""

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.media.resolver import SourceResolver
from nomorals.media.sources import MP3PMSource, SourceCandidate

# Recorded live row (https://mp3.pm/search?q=tell+me+one+lie).
_ROW_HTML = (
    '<ul class="mp3list">'
'<li class="cplayer-sound-item" data-sound-id="248463781" data-sound-url="https://cs1.mp3.pm/listen/248463781/MDV5SHViSDd6WEY1Y3RtaC8wQWNyc29jNFFwWnNUMURmT0xaR05PRXR2ZlBWNmdXUjZaV0lUQmxjMDNVeUFKZW9KeU85MlFJR04wNEdRdTB2R0VwTlBkV0FIaHhwZm12dStENmt3WS9IQ2UyQ1A2dzl2VUhyQ3ZuTEUvVXFhdVY/Ronis_Goliath_-_Tell_Me_One_Lie_(mp3.pm).mp3" data-download-url="https://cs1.mp3.pm/download/248463781/MDV5SHViSDd6WEY1Y3RtaC8wQWNyc29jNFFwWnNUMURmT0xaR05PRXR2ZlBWNmdXUjZaV0lUQmxjMDNVeUFKZW9KeU85MlFJR04wNEdRdTB2R0VwTlBkV0FIaHhwZm12dStENmt3WS9IQ2UyQ1A2dzl2VUhyQ3ZuTEUvVXFhdVY/Ronis_Goliath_-_Tell_Me_One_Lie_(mp3.pm).mp3" data-share-url="https://mp3.pmhttps://28682549-ronis-goliath.mp3.pm/song/248463781-tell-me-one-lie/">\n\n\t<!-- mp3list-btns -->\n\t\t<div class="mp3list-btns">\n\t\t\t<a href="javascript:void(0);" class="mp3list-btn-play cplayer-ui-play" title="play">(play)</a>\n\t\t\t<a href="javascript:void(0);" class="mp3list-btn-pause cplayer-ui-pause" title="pause">(pause)</a>\n\t\t\t<a href="https://28682549-ronis-goliath.mp3.pm/song/248463781-tell-me-one-lie/" class="mp3list-btn-download cplayer-ui-download" title="download">(download)</a>\n\n\t\t\t<a href="javascript:void(0);" class="mp3list-btn-fb cplayer-ui-share-fb" title="share: facebook">(fb)</a>\n\t\t\t<a href="javascript:void(0);" class="mp3list-btn-vk cplayer-ui-share-vk" title="share: vk">(vk)</a>\n\t\t\t<a href="javascript:void(0);" class="mp3list-btn-tw cplayer-ui-share-tw" title="share: twitter">(tw)</a>\n\t\t</div>\n\t\t<!-- / mp3list-btns -->\n\n\t\t<h4>\n\t\t\t<a href="https://28682549-ronis-goliath.mp3.pm/"><i class="cplayer-data-sound-author">Ronis Goliath</i></a>\n\t\t\t<a href="https://28682549-ronis-goliath.mp3.pm/song/248463781-tell-me-one-lie/"><b class="cplayer-data-sound-title">Tell Me One Lie</b></a>\n\t\t</h4>\n\t\t<em class="cplayer-data-sound-time">02:41</em>\n\t</li>'
    + '</ul>'
)

_EMPTY_HTML = '<ul class="mp3list"></ul>'

_SONG_URL = "https://28682549-ronis-goliath.mp3.pm/song/248463781-tell-me-one-lie/"


class Mp3pmRowTests(unittest.TestCase):
    def test_search_parses_server_rendered_rows(self):
        src = MP3PMSource()
        src._get = lambda url: _ROW_HTML  # noqa: SLF001 - seam under test
        found = src.search("tell me one lie", limit=5)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].title, "Tell Me One Lie")
        self.assertEqual(found[0].artist, "Ronis Goliath")
        self.assertEqual(found[0].source, "mp3pm")
        self.assertEqual(found[0].url, _SONG_URL)
        self.assertTrue(found[0].direct_url.startswith(
            "https://cs1.mp3.pm/listen/248463781/"),
            found[0].direct_url)
        self.assertTrue(found[0].direct_url.endswith(
            "Ronis_Goliath_-_Tell_Me_One_Lie_(mp3.pm).mp3"))
        self.assertEqual(found[0].duration, 161.0)

    def test_search_unknown_query_returns_empty(self):
        src = MP3PMSource()
        src._get = lambda url: _EMPTY_HTML  # noqa: SLF001
        self.assertEqual(src.search("zzqzxqvjwkjw", limit=5), [])

    def test_search_dead_fetch_returns_empty(self):
        src = MP3PMSource()
        src._get = lambda url: ""  # noqa: SLF001 - dead endpoint
        self.assertEqual(src.search("tell me one lie", limit=5), [])

    def test_search_url_points_at_search_page(self):
        self.assertEqual(
            MP3PMSource().search_url("tell me one lie"),
            "https://mp3.pm/search?q=tell+me+one+lie")

    def test_parse_search_parses_rows(self):
        rows = MP3PMSource().parse_search(_ROW_HTML)
        self.assertEqual(rows, [("Tell Me One Lie", _SONG_URL)])

    def test_parse_search_still_parses_json(self):
        rows = MP3PMSource().parse_search(json.dumps({"songs": [
            {"url": "https://a-1.mp3.pm/song/7-hello-world/",
             "title": "Hello-World"}]}))
        self.assertEqual(rows, [("Hello World",
                                 "https://a-1.mp3.pm/song/7-hello-world/")])


# ── grown-source sweep ──────────────────────────────────────────────

_NAMED_MANY = (
    "_netnaija_search_many", "_notjustok_search_many",
    "_tooxclusive_search_many", "_mp3pm_search_many",
    "_mp3juice_search_many", "_soundcloud_search_many",
    "_bandcamp_search_many", "_youtube_search_many",
    "_youtube_music_search_many", "_mdundo_search_many",
    "_archiveorg_search_many", "_jamendo_search_many",
    "_fma_search_many", "_jiosaavn_search_many",
    "_musicbrainz_search_many", "_openverse_search_many",
    "_wikimedia_search_many", "_itunes_search_many",
    "_deezer_search_many",
)


def _ok_fetcher(url):
    return 200, b"<html><body>a real page with real content</body></html>"


class GrownSweepTests(unittest.TestCase):
    def _promoted_resolver(self):
        from nomorals.media.source_registry import _seed

        reg = _seed()
        self.assertTrue(reg.note_candidate(
            "https://freshbeats.example.org/song", via="test"))
        summary = reg.verify_health(fetcher=_ok_fetcher)
        self.assertEqual(summary["promoted"], 1, summary)
        ctx = SimpleNamespace(
            settings=SimpleNamespace(workspace_dir="/nonexistent"))
        return SourceResolver(ctx, source_registry=reg)

    def _run_sweep(self, resolver):
        from nomorals.media import sources as sources_mod

        grown_hit = SourceCandidate(
            source="grown:freshbeats.example.org", title="Fresh",
            url="https://freshbeats.example.org/posts/1")
        patchers = [mock.patch.object(
            SourceResolver, name, return_value=[])
            for name in _NAMED_MANY]
        patchers.append(mock.patch.object(
            sources_mod.DiscoveredBlogSource, "search",
            return_value=[grown_hit]))
        for p in patchers:
            p.start()
        try:
            return resolver.search_candidates_detailed("fresh beats")
        finally:
            for p in reversed(patchers):
                p.stop()

    def test_detailed_sweep_grown_source_serves(self):
        cands, attempts = self._run_sweep(self._promoted_resolver())
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0].kind, "grown:freshbeats.example.org")
        self.assertEqual(cands[0].title, "Fresh")
        self.assertIn("grown:freshbeats.example.org — ok (1)", attempts)

    def test_detailed_sweep_grown_attempt_recorded(self):
        _cands, attempts = self._run_sweep(self._promoted_resolver())
        self.assertTrue(any(
            a.startswith("grown:freshbeats.example.org") for a in attempts),
            attempts)
        self.assertFalse(any("takes from 0 to 1" in a for a in attempts),
                         attempts)


if __name__ == "__main__":
    unittest.main()
