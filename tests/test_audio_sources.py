"""Tests for nomorals/media/sources.py — the /play source strategy chain.

Covers (all with mocked network — the offline suite stays offline):
- AudiomackSource: URL detection, public stream-API URL building,
  graceful search degradation (no keyless endpoint exists).
- NetNaijaSource: WP search parsing, multi-hop download extraction,
  never-raises on garbage/network failure.
- BoomplaySource: metadata search, honest non-downloadable marking.
- BaseScrapeSource helpers: media-link extraction, URL absolutizing.
- Resolver wiring: new strategies in the text/URL chains, pick-list
  candidates, chain ordering and fallthrough.
- Playback download(): new kinds (audiomack/netnaija/boomplay),
  generalized YouTube fallback labels.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nomorals.media.sources import (
    AudiomackSource,
    BaseScrapeSource,
    BoomplaySource,
    FMASource,
    JamendoSource,
    MdundoSource,
    NetNaijaSource,
    SourceCandidate,
    candidate_asset_class,
)

# ── fixtures: mocked HTML ─────────────────────────────────────────────

_WP_SEARCH_HTML = """
<html><body>
<article><h2 class="entry-title">
<a href="https://netnaija.com/music/burna-boy-last-last/">Burna Boy - Last Last</a>
</h2></article>
<article><h2 class="entry-title">
<a href="https://netnaija.com/music/ayra-starr-rush/">Ayra Starr - Rush</a>
</h2></article>
</body></html>
"""

_POST_WITH_DIRECT_MP3 = """
<html><body>
<h1>Burna Boy - Last Last</h1>
<a href="https://dl.netnaija.com/music/last-last.mp3">Download Now (8.2 MB)</a>
</body></html>
"""

_POST_WITH_HOP = """
<html><body>
<h1>Burna Boy - Last Last</h1>
<a href="https://netnaija.com/dl/abc123/">Download</a>
</body></html>
"""

_HOP_PAGE = """
<html><body>
<p>Your download is ready</p>
<a href="https://files.netnaija.com/x/last-last.mp3">Download Now (8.2 MB)</a>
</body></html>
"""

_BOOMPLAY_SEARCH_HTML = """
<html><body>
<div class="result">
<a href="/songs/211634092_last-last">Last Last</a>
<span>Burna Boy</span>
</div>
<div class="result">
<a href="/songs/998877665_rush">Rush</a>
<span>Ayra Starr</span>
</div>
</body></html>
"""

#: SPA /search/music/ payload: songWrap buttons with data-data
#: fields split on @+# → [0]=song id, [4]=title, [5]=artist.
_BOOMPLAY_SONG_WRAP_HTML = """
<html><body>
<button data-id="211634092" data-data="211634092@+#5@+#10039@+#https://x/img.jpg@+#Last Last@+#Burna Boy@+#tok@+#03:42@+#OMS@+#0@+#tok2@+#Album@+#https://x/cover.jpg@+#https://x/artist.jpg@+#1777640@+#1777431@+#0"></button>
<button data-id="998877665" data-data="998877665@+#5@+#10040@+#https://x/img2.jpg@+#Rush@+#Ayra Starr@+#tok@+#03:10@+#OMS@+#0@+#tok2@+#Album@+#https://x/cover2.jpg@+#https://x/artist2.jpg@+#1777641@+#1777432@+#0"></button>
</body></html>
"""


# ── AudiomackSource ───────────────────────────────────────────────────


class TestAudiomackSource:
    def test_handles_track_url(self):
        src = AudiomackSource()
        assert src.handles_url(
            "https://audiomack.com/burnaboy/song/last-last")
        assert src.handles_url(
            "http://www.audiomack.com/ayra-starr/song/rush")

    def test_rejects_non_audiomack(self):
        src = AudiomackSource()
        assert not src.handles_url("https://soundcloud.com/x/y")
        assert not src.handles_url("https://youtube.com/watch?v=abc")
        assert not src.handles_url("")
        assert not src.handles_url(None)

    def test_track_api_url(self):
        src = AudiomackSource()
        url = src.track_api_url("https://audiomack.com/burnaboy/song/last-last")
        assert url == ("https://audiomack.com/api/music/url/song/"
                       "burnaboy/last-last?extended=1")

    def test_track_api_url_bad_input(self):
        src = AudiomackSource()
        assert src.track_api_url("https://example.com/x") == ""
        assert src.track_api_url("") == ""

    def test_search_degrades_gracefully(self):
        # No keyless search endpoint exists (verified 2026-10-09) —
        # search() must return [] and never raise, so the chain moves on.
        src = AudiomackSource()
        assert src.search("burna boy") == []
        assert src.search("") == []

    def test_download_url_is_permalink(self):
        src = AudiomackSource()
        c = SourceCandidate(source="audiomack", title="Last Last",
                            url="https://audiomack.com/burnaboy/song/last-last")
        assert src.download_url(c) == c.url


# ── NetNaijaSource ────────────────────────────────────────────────────


class TestNetNaijaSource:
    def test_search_url(self):
        src = NetNaijaSource()
        url = src.search_url("burna boy")
        assert url.startswith("https://netnaija.com/?s=")
        assert "burna" in url

    def test_parse_search(self):
        src = NetNaijaSource()
        results = src.parse_search(_WP_SEARCH_HTML)
        assert len(results) == 2
        assert results[0][0] == "Burna Boy - Last Last"
        assert results[0][1] == "https://netnaija.com/music/burna-boy-last-last/"
        assert results[1][0] == "Ayra Starr - Rush"

    def test_parse_search_empty(self):
        src = NetNaijaSource()
        assert src.parse_search("") == []
        assert src.parse_search("<html><body>nothing here</body></html>") == []

    def test_extract_direct_mp3(self):
        src = NetNaijaSource()
        url = src.extract_download(
            _POST_WITH_DIRECT_MP3, "https://netnaija.com/music/x/")
        assert url == "https://dl.netnaija.com/music/last-last.mp3"

    def test_extract_multi_hop(self):
        src = NetNaijaSource()
        with patch.object(NetNaijaSource, "_get", return_value=_HOP_PAGE):
            url = src.extract_download(
                _POST_WITH_HOP, "https://netnaija.com/music/x/")
        assert url == "https://files.netnaija.com/x/last-last.mp3"

    def test_extract_no_download(self):
        src = NetNaijaSource()
        assert src.extract_download(
            "<html><body>no links</body></html>",
            "https://netnaija.com/music/x/") == ""

    def test_search_never_raises(self):
        src = NetNaijaSource()
        with patch.object(NetNaijaSource, "_get", return_value=_WP_SEARCH_HTML):
            cands = src.search("burna boy", limit=5)
        assert len(cands) == 2
        assert all(isinstance(c, SourceCandidate) for c in cands)
        assert cands[0].source == "netnaija"

    def test_search_network_failure(self):
        src = NetNaijaSource()
        with patch.object(NetNaijaSource, "_get", return_value=""):
            assert src.search("burna boy") == []

    def test_search_empty_query(self):
        assert NetNaijaSource().search("") == []

    def test_download_url_resolves(self):
        src = NetNaijaSource()
        c = SourceCandidate(source="netnaija", title="Last Last",
                            url="https://netnaija.com/music/x/")
        with patch.object(NetNaijaSource, "_get",
                          return_value=_POST_WITH_DIRECT_MP3):
            assert src.download_url(c) == \
                "https://dl.netnaija.com/music/last-last.mp3"

    def test_download_url_prefers_direct(self):
        src = NetNaijaSource()
        c = SourceCandidate(source="netnaija", title="X",
                            url="https://netnaija.com/music/x/",
                            direct_url="https://cdn.example.com/x.mp3")
        assert src.download_url(c) == "https://cdn.example.com/x.mp3"

    def test_download_url_never_raises(self):
        src = NetNaijaSource()
        c = SourceCandidate(source="netnaija", url="https://netnaija.com/x/")
        with patch.object(NetNaijaSource, "_get",
                          side_effect=Exception("boom")):
            assert src.download_url(c) == ""


# ── BoomplaySource ────────────────────────────────────────────────────


class TestBoomplaySource:
    def test_search_metadata(self):
        # Browser-backed SPA search: songWrap data-data payloads.
        src = BoomplaySource()
        with patch.object(BoomplaySource, "_browser_fetch",
                          return_value=_BOOMPLAY_SONG_WRAP_HTML):
            cands = src.search("burna", limit=5)
        assert len(cands) == 2
        assert cands[0].title == "Last Last"
        assert cands[0].artist == "Burna Boy"
        assert cands[0].url == "https://www.boomplay.com/songs/211634092"
        # honestly non-downloadable
        assert all(c.downloadable is False for c in cands)

    def test_search_legacy_plain_http_fallback(self):
        # Plain-HTTP fallback still parses /songs/ href links when the
        # browser fetch comes back empty.
        src = BoomplaySource()
        with patch.object(BoomplaySource, "_browser_fetch",
                          return_value=""), \
                patch.object(BoomplaySource, "_get",
                             return_value=_BOOMPLAY_SEARCH_HTML):
            cands = src.search("burna", limit=5)
        assert len(cands) == 2
        assert cands[0].title == "Last Last"
        assert cands[0].url == "https://www.boomplay.com/songs/211634092_last-last"

    def test_search_empty(self):
        src = BoomplaySource()
        assert src.search("") == []
        with patch.object(BoomplaySource, "_browser_fetch",
                          return_value=""), \
                patch.object(BoomplaySource, "_get", return_value=""):
            assert src.search("x") == []

    def test_search_never_raises(self):
        src = BoomplaySource()
        with patch.object(BoomplaySource, "_browser_fetch",
                          side_effect=Exception("boom")), \
                patch.object(BoomplaySource, "_get",
                             side_effect=Exception("boom")):
            assert src.search("x") == []

    def test_download_url_empty(self):
        src = BoomplaySource()
        c = SourceCandidate(source="boomplay", title="X",
                            url="https://www.boomplay.com/songs/1_x")
        assert src.download_url(c) == ""


# ── BaseScrapeSource helpers ──────────────────────────────────────────


#: Realistic Jamendo v3.0 tracks payload (envelope + track rows as
#: the API returns them — verified live 2026-11-12 that a no-key call
#: fails with "Invalid Client Id", so the request shape below is the
#: keyed shape the client id unlocks).
_JAMENDO_PAYLOAD = """{"headers": {"status": "success", "code": 0,
"error_message": "", "warnings": "", "results_count": 3},
"results": [
{"id": "1234567", "name": "Summer Breeze", "duration": 214,
"artist_id": "76543", "artist_name": "Coastal Lines",
"album_name": "Salt Air", "album_id": "98765",
"license_ccurl": "https://creativecommons.org/licenses/by/3.0/",
"releasedate": "2021-06-01",
"audio": "https://mp3d.jamendo.com/download/track/1234567/mp32/x.mp3",
"audiodownload": "https://mp3d.jamendo.com/download/track/1234567/mp32/x.mp3",
"shareurl": "https://www.jamendo.com/track/1234567/summer-breeze",
"image": "https://usercontent.jamendo.com?type=album&id=98765&width=300"},
{"id": "1234568", "name": "Night Drive", "duration": 187,
"artist_id": "76544", "artist_name": "Neon Coast",
"album_name": "After Dark", "album_id": "98766",
"license_ccurl": "https://creativecommons.org/licenses/by-nc-sa/3.0/",
"audio": "https://mp3d.jamendo.com/download/track/1234568/mp32/y.mp3",
"shareurl": "https://www.jamendo.com/track/1234568/night-drive"},
{"id": "1234569", "name": "Silent Row", "duration": 150,
"artist_id": "76545", "artist_name": "No Audio",
"album_name": "Blank", "album_id": "98767",
"license_ccurl": "https://creativecommons.org/licenses/by/3.0/",
"shareurl": "https://www.jamendo.com/track/1234569/silent-row"}]}"""


# ── JamendoSource ─────────────────────────────────────────────────────


class TestJamendoSource:
    def test_search_url_with_client_id(self, monkeypatch):
        monkeypatch.setenv("NM_JAMENDO_CLIENT_ID", "abc123")
        url = JamendoSource().search_url("summer breeze")
        assert url.startswith("https://api.jamendo.com/v3.0/tracks/?")
        assert "client_id=abc123" in url
        assert "format=json" in url
        assert "search=summer+breeze" in url

    def test_search_url_without_client_id_omits_param(self, monkeypatch):
        monkeypatch.delenv("NM_JAMENDO_CLIENT_ID", raising=False)
        url = JamendoSource().search_url("summer breeze")
        assert "client_id" not in url
        assert "format=json" in url

    def test_search_without_client_id_returns_empty_without_network(
            self, monkeypatch):
        monkeypatch.delenv("NM_JAMENDO_CLIENT_ID", raising=False)
        with patch.object(JamendoSource, "_get") as get:
            assert JamendoSource().search("summer breeze") == []
            get.assert_not_called()

    def test_search_with_client_id_returns_full_candidates(
            self, monkeypatch):
        monkeypatch.setenv("NM_JAMENDO_CLIENT_ID", "abc123")
        with patch.object(JamendoSource, "_get",
                          return_value=_JAMENDO_PAYLOAD):
            cands = JamendoSource().search("summer breeze")
        assert [c.title for c in cands] == ["Summer Breeze", "Night Drive"]
        first = cands[0]
        assert first.artist == "Coastal Lines"
        assert first.duration == pytest.approx(214.0)
        assert first.direct_url.endswith(".mp3")
        assert first.downloadable is True
        assert candidate_asset_class(first) == "full"
        assert first.extra["page"].startswith(
            "https://www.jamendo.com/track/")
        assert "creativecommons.org" in first.extra["license"]
        # audio-only row falls back to the stream URL
        assert cands[1].direct_url.endswith("y.mp3")
        # the row with no audio field is skipped, never metadata
        assert all(c.title != "Silent Row" for c in cands)

    def test_parse_failed_envelope_returns_empty(self):
        failed = {"headers": {"status": "failed", "code": 5,
                              "error_message": "Jamendo Api Invalid Client "
                              "Id Error: Your credential is not "
                              "authorized..", "warnings": "",
                              "results_count": 0}, "results": []}
        assert JamendoSource().parse(failed, "x") == []
        assert JamendoSource().parse({}, "x") == []
        assert JamendoSource().parse(None, "x") == []

    def test_empty_query_returns_empty(self, monkeypatch):
        monkeypatch.setenv("NM_JAMENDO_CLIENT_ID", "abc123")
        with patch.object(JamendoSource, "_get") as get:
            assert JamendoSource().search("   ") == []
            get.assert_not_called()


#: Realistic FMA search-page rows — titles, artists, albums,
#: durations and /music/ URL shapes copied from the LIVE search page
#: fetched 2026-11-12 (?quicksearch=blue+dot, 2,193 tracks), with
#: the anchor markup as the page renders it.  The JSON API the row
#: used to describe is dead (verified: /api, /api/trackSearch and
#: /recent.json all 404), so the fixture is HTML, not a JSON envelope.
_FMA_SEARCH_HTML = """
<html><body>
<table>
<tr><td><a href="/music/Blue_Dot_Sessions/reflections/ether-ridge/">Ether Ridge</a> by <a href="/music/Blue_Dot_Sessions/">Blue Dot Sessions</a> <a href="/music/Blue_Dot_Sessions/reflections/">Reflections</a> &mdash; 03:10</td></tr>
<tr><td><a href="/music/Blue_Dot_Sessions/the-viscount/etude-1-cherub/"><span>Etude 1 Cherub</span></a> by <a href="/music/Blue_Dot_Sessions/">Blue Dot Sessions</a> <a href="/music/Blue_Dot_Sessions/the-viscount/">The Viscount</a> &mdash; 02:00</td></tr>
<tr><td><a href="https://freemusicarchive.org/music/M_Mucci/days-blur-together/days-blur-together/">Days Blur Together</a> by <a href="/music/M_Mucci/">M. Mucci</a> <a href="/music/M_Mucci/days-blur-together/">Days Blur Together</a> &mdash; 65:36</td></tr>
<tr><td><a href="/music/Blue_Dot_Sessions/reflections/ether-ridge/">Ether Ridge</a> by <a href="/music/Blue_Dot_Sessions/">Blue Dot Sessions</a> <a href="/music/Blue_Dot_Sessions/reflections/">Reflections</a> &mdash; 03:10</td></tr>
<tr><td><a href="/music/Blue_Dot_Sessions/comptroller/even-spread/">Even Spread</a> by <a href="/music/Blue_Dot_Sessions/">Blue Dot Sessions</a> <a href="/music/Blue_Dot_Sessions/comptroller/">Comptroller</a></td></tr>
<tr><td><a href="/search/?quicksearch=blue&amp;page=2">next page</a></td></tr>
</table>
</body></html>
"""


# ── FMASource ─────────────────────────────────────────────────────────


class TestFMASource:
    def test_search_url_shape(self):
        url = FMASource().search_url("blue dot")
        assert url == ("https://freemusicarchive.org/search"
                       "?quicksearch=blue+dot&pageSize=20&sort=track")

    def test_search_parses_live_shaped_rows(self):
        with patch.object(FMASource, "_get",
                          return_value=_FMA_SEARCH_HTML):
            cands = FMASource().search("blue dot")
        assert [c.title for c in cands] == [
            "Ether Ridge", "Etude 1 Cherub", "Days Blur Together",
            "Even Spread"]
        first = cands[0]
        assert first.artist == "Blue Dot Sessions"
        assert first.extra["album"] == "Reflections"
        assert first.duration == pytest.approx(190.0)
        assert first.url == ("https://freemusicarchive.org/music/"
                             "Blue_Dot_Sessions/reflections/ether-ridge/")
        # metadata-class: identified, no audio — never a download claim
        assert first.downloadable is False
        assert candidate_asset_class(first) == "metadata"
        assert "track page" in first.extra["note"]
        # nested markup inside the anchor is stripped
        assert cands[1].title == "Etude 1 Cherub"
        assert cands[1].duration == pytest.approx(120.0)
        # absolute hrefs stay absolute; multi-digit minutes parse
        assert cands[2].url.startswith("https://freemusicarchive.org/")
        assert cands[2].duration == pytest.approx(65 * 60 + 36.0)
        # a row with no duration still yields its track
        assert cands[3].title == "Even Spread"
        assert cands[3].duration == pytest.approx(0.0)

    def test_search_skips_non_track_links_and_dedupes(self):
        with patch.object(FMASource, "_get",
                          return_value=_FMA_SEARCH_HTML):
            cands = FMASource().search("blue dot")
        assert all("/music/" in c.url for c in cands)
        assert len({c.url for c in cands}) == len(cands)

    def test_search_empty_and_garbage(self):
        with patch.object(FMASource, "_get") as get:
            assert FMASource().search("   ") == []
            get.assert_not_called()
        with patch.object(FMASource, "_get",
                          return_value="<html><body>no results</body></html>"):
            assert FMASource().search("zzz") == []
        with patch.object(FMASource, "_get", return_value=""):
            assert FMASource().search("zzz") == []


#: Realistic Mdundo search-page rows — titles, artists, durations
#: and /stream/ URL shapes copied from the LIVE search page fetched
#: 2026-11-12 (play.mdundo.com/search-results?term=diamond+platnumz,
#: 32 songs), with the markup in the ESCAPED form the page actually
#: serves (\/, \&quot;, \u0022) — the parser must decode it before
#: matching.  The /stream/<id> URL is the stable redirector to a
#: signed mdundosound.com MP3 (verified: HTTP 200, audio/mpeg);
#: /dl/ is a page flow, not a direct file.
_MDUNDO_SEARCH_HTML = r"""
<div id=\&quot;songlist-page-search-results-738a0856c16af35c4ac1ad513afd07a5-id--song-1571564\&quot; class=\&quot;md-playlist-song-item\&quot; data-url-play=\&quot;https:\/\/mdundo.com\/stream\/1571564\&quot; data-url-download=\&quot;https:\/\/mdundo.com\/dl\/1571564?pwa=1\&quot; data-duration-seconds=\&quot;233\&quot; data-duration=\&quot;3:53\&quot; data-title=\&quot;Sikomi\&quot; data-artist=\&quot;Diamond Platnumz\&quot; data-artist-url=\&quot;\/artist\/175073\/Diamond-Platnumz\&quot; data-id=\&quot;1571564\&quot; data-is-premium=\&quot;0\&quot;>
<div id=\&quot;songlist-page-search-results-738a0856c16af35c4ac1ad513afd07a5-id--song-1571566\&quot; class=\&quot;md-playlist-song-item\&quot; data-url-play=\&quot;https:\/\/mdundo.com\/stream\/1571566\&quot; data-url-download=\&quot;https:\/\/mdundo.com\/dl\/1571566?pwa=1\&quot; data-duration-seconds=\&quot;234\&quot; data-duration=\&quot;3:54\&quot; data-title=\&quot;Ukimuona\&quot; data-artist=\&quot;Diamond Platnumz\&quot; data-artist-url=\&quot;\/artist\/175073\/Diamond-Platnumz\&quot; data-id=\&quot;1571566\&quot; data-is-premium=\&quot;0\&quot;>
<div id=\&quot;songlist-page-search-results-738a0856c16af35c4ac1ad513afd07a5-id--song-1579085\&quot; class=\&quot;md-playlist-song-item\&quot; data-url-play=\&quot;https:\/\/mdundo.com\/stream\/1579085\&quot; data-url-download=\&quot;https:\/\/mdundo.com\/dl\/1579085?pwa=1\&quot; data-duration-seconds=\&quot;251\&quot; data-duration=\&quot;4:11\&quot; data-title=\&quot;Mbagala\&quot; data-artist=\&quot;Diamond Platnumz\&quot; data-artist-url=\&quot;\/artist\/175073\/Diamond-Platnumz\&quot; data-id=\&quot;1579085\&quot; data-is-premium=\&quot;0\&quot;>
<div id=\&quot;songlist-page-search-results-738a0856c16af35c4ac1ad513afd07a5-id--song-9999999\&quot; class=\&quot;md-playlist-song-item\&quot; data-url-play=\&quot;https:\/\/mdundo.com\/stream\/9999999\&quot; data-url-download=\&quot;https:\/\/mdundo.com\/dl\/9999999?pwa=1\&quot; data-duration-seconds=\&quot;200\&quot; data-duration=\&quot;3:20\&quot; data-title=\&quot;Premium Only\&quot; data-artist=\&quot;Diamond Platnumz\&quot; data-artist-url=\&quot;\/artist\/175073\/Diamond-Platnumz\&quot; data-id=\&quot;9999999\&quot; data-is-premium=\&quot;1\&quot;>
<div id=\&quot;songlist-page-search-results-738a0856c16af35c4ac1ad513afd07a5-id--song-1571564\&quot; class=\&quot;md-playlist-song-item\&quot; data-url-play=\&quot;https:\/\/mdundo.com\/stream\/1571564\&quot; data-url-download=\&quot;https:\/\/mdundo.com\/dl\/1571564?pwa=1\&quot; data-duration-seconds=\&quot;233\&quot; data-duration=\&quot;3:53\&quot; data-title=\&quot;Sikomi\&quot; data-artist=\&quot;Diamond Platnumz\&quot; data-artist-url=\&quot;\/artist\/175073\/Diamond-Platnumz\&quot; data-id=\&quot;1571564\&quot; data-is-premium=\&quot;0\&quot;>
<div id=\&quot;songlist-page-search-results-738a0856c16af35c4ac1ad513afd07a5-id--song-1571521\&quot; class=\&quot;md-playlist-song-item\&quot; data-url-play=\&quot;https:\/\/mdundo.com\/stream\/1571521\&quot; data-url-download=\&quot;https:\/\/mdundo.com\/dl\/1571521?pwa=1\&quot; data-title=\&quot;No Duration Song\&quot; data-artist=\&quot;Diamond Platnumz\&quot; data-artist-url=\&quot;\/artist\/175073\/Diamond-Platnumz\&quot; data-id=\&quot;1571521\&quot; data-is-premium=\&quot;0\&quot;>
<a href=\&quot;/artist\/175073\/Diamond-Platnumz\&quot;>Diamond Platnumz</a>
"""


# ── MdundoSource ──────────────────────────────────────────────────────


class TestMdundoSource:
    def test_search_url_shape(self):
        url = MdundoSource().search_url("diamond platnumz")
        assert url == ("https://play.mdundo.com/search-results"
                       "?term=diamond+platnumz")

    def test_search_parses_live_shaped_rows(self):
        with patch.object(MdundoSource, "_get",
                          return_value=_MDUNDO_SEARCH_HTML):
            cands = MdundoSource().search("diamond platnumz")
        assert [c.title for c in cands] == [
            "Sikomi", "Ukimuona", "Mbagala", "No Duration Song"]
        first = cands[0]
        assert first.artist == "Diamond Platnumz"
        assert first.duration == pytest.approx(233.0)
        assert first.url == "https://mdundo.com/stream/1571564"
        assert first.direct_url == first.url
        # full-track: the /stream/<id> redirector serves a whole MP3
        assert first.downloadable is True
        assert candidate_asset_class(first) == "full"
        assert first.extra["download_page"] == (
            "https://mdundo.com/dl/1571564?pwa=1")
        assert "redirector" in first.extra["note"]
        # duplicate row deduped; premium row skipped, never a hit
        assert len({c.url for c in cands}) == len(cands)
        assert all(c.title != "Premium Only" for c in cands)
        # a row with no duration still yields its track
        assert cands[3].duration == pytest.approx(0.0)

    def test_search_empty_and_garbage(self):
        with patch.object(MdundoSource, "_get") as get:
            assert MdundoSource().search("   ") == []
            get.assert_not_called()
        with patch.object(MdundoSource, "_get",
                          return_value="<html><body>no results</body></html>"):
            assert MdundoSource().search("zzz") == []
        with patch.object(MdundoSource, "_get", return_value=""):
            assert MdundoSource().search("zzz") == []

    def test_unescape_handles_live_escapes(self):
        raw = (r'data-url-play=\&quot;https:\/\/mdundo.com\/stream\/1'
               r'\&quot; data-title=\u0022X\u0022 &gt;')
        norm = MdundoSource._unescape(raw)
        assert norm == ('data-url-play="https://mdundo.com/stream/1"'
                        ' data-title="X" >')


class TestScrapeHelpers:
    def test_direct_media_links(self):
        html = ('<a href="/music/a.mp3">x</a>'
                '<audio src="https://cdn.example.com/b.m4a"></audio>'
                '<a href="/page/">not media</a>')
        links = BaseScrapeSource._direct_media_links(
            html, "https://example.com/")
        assert "https://example.com/music/a.mp3" in links
        assert "https://cdn.example.com/b.m4a" in links
        assert len(links) == 2

    def test_abs(self):
        assert BaseScrapeSource._abs("https://a.com/x/", "/y") == \
            "https://a.com/y"

    def test_clean(self):
        assert BaseScrapeSource._clean("  a &amp; b\n ") == "a & b"


# ── resolver wiring ───────────────────────────────────────────────────


class TestResolverWiring:
    def _resolver(self):
        from nomorals.media.resolver import SourceResolver
        return SourceResolver(context=MagicMock())

    def test_audiomack_url_strategy_rejects_other_urls(self):
        from nomorals.media.resolver import ResolutionError
        r = self._resolver()
        with pytest.raises(ResolutionError):
            r._s_audiomack_url("https://soundcloud.com/x/y")

    def test_audiomack_url_strategy_accepts(self):
        r = self._resolver()
        with patch("nomorals.media.resolver.probe",
                   return_value={"title": "Last Last"}):
            out = r._s_audiomack_url(
                "https://audiomack.com/burnaboy/song/last-last")
        assert out.ok and out.kind == "audiomack"
        assert out.title == "Last Last"

    def test_netnaija_search_no_results(self):
        from nomorals.media.resolver import ResolutionError
        r = self._resolver()
        with patch.object(NetNaijaSource, "search", return_value=[]):
            with pytest.raises(ResolutionError):
                r._s_netnaija_search("zzzznothing")

    def test_netnaija_search_hit(self):
        r = self._resolver()
        cands = [SourceCandidate(source="netnaija", title="Last Last",
                                 url="https://netnaija.com/music/x/")]
        with patch.object(NetNaijaSource, "search", return_value=cands):
            out = r._s_netnaija_search("last last")
        assert out.ok and out.kind == "netnaija"
        assert out.title == "Last Last"

    def test_boomplay_search_marks_non_downloadable(self):
        r = self._resolver()
        cands = [SourceCandidate(source="boomplay", title="Last Last",
                                 url="https://www.boomplay.com/songs/1_x",
                                 downloadable=False)]
        with patch.object(BoomplaySource, "search", return_value=cands):
            out = r._s_boomplay_search("last last")
        assert out.ok and out.kind == "boomplay"
        assert out.downloadable is False

    def test_jamendo_search_hit_is_full_track(self):
        r = self._resolver()
        cands = [SourceCandidate(source="jamendo", title="Summer Breeze",
                                 artist="Coastal Lines",
                                 url="https://mp3d.jamendo.com/download/"
                                     "track/1234567/mp32/x.mp3",
                                 direct_url="https://mp3d.jamendo.com/"
                                            "download/track/1234567/mp32/"
                                            "x.mp3")]
        with patch.object(JamendoSource, "search", return_value=cands):
            out = r._s_blog_search("summer breeze", "JamendoSource",
                                   "jamendo")
        assert out.ok and out.kind == "jamendo"
        assert out.title == "Summer Breeze"
        assert out.path_or_url.endswith(".mp3")
        assert out.downloadable is True

    def test_mdundo_search_hit_is_full_track(self):
        r = self._resolver()
        cands = [SourceCandidate(source="mdundo", title="Sikomi",
                                 artist="Diamond Platnumz",
                                 url="https://mdundo.com/stream/1571564",
                                 direct_url="https://mdundo.com/stream/"
                                            "1571564")]
        with patch.object(MdundoSource, "search", return_value=cands):
            out = r._s_blog_search("sikomi", "MdundoSource", "mdundo")
        assert out.ok and out.kind == "mdundo"
        assert out.title == "Sikomi"
        assert out.path_or_url == "https://mdundo.com/stream/1571564"
        assert out.downloadable is True

    def test_text_chain_places_jamendo_after_archiveorg(self):
        # Global full-track CC catalogue, grouped with the other open
        # archive (archive.org), before the metadata-only block — and
        # before the Openverse aggregator, which already indexes
        # Jamendo: the direct catalogue beats the aggregator.
        import inspect

        from nomorals.media import resolver as res_mod
        src = inspect.getsource(res_mod.SourceResolver._resolve_text)
        ao = src.index("archiveorg-search")
        ja = src.index("jamendo-search")
        bp = src.index("boomplay-search")
        ov = src.index("openverse-search")
        assert ao < ja < bp
        assert ja < ov

    def test_fma_search_hit_is_metadata_only(self):
        # FMA identifies the track and its page — the JSON API is
        # dead, so no audio is claimed.  The chain keeps going for a
        # full track and only surfaces this when nothing downloadable
        # exists anywhere (MusicBrainz shape).
        r = self._resolver()
        cands = [SourceCandidate(
            source="fma", title="Ether Ridge", artist="Blue Dot Sessions",
            url="https://freemusicarchive.org/music/Blue_Dot_Sessions/"
                "reflections/ether-ridge/",
            duration=190.0, downloadable=False)]
        with patch.object(FMASource, "search", return_value=cands):
            out = r._s_metadata_search(
                "ether ridge", "FMASource", "fma",
                "Free Music Archive — track identified, no audio; "
                "download via the track page")
        assert out.ok and out.kind == "fma"
        assert out.title == "Ether Ridge"
        assert out.path_or_url.endswith("/ether-ridge/")
        assert out.downloadable is False

    def test_text_chain_places_fma_after_jamendo(self):
        # The third open archive, grouped with archive.org and
        # Jamendo, before the metadata-only block — and before the
        # Openverse aggregator.
        import inspect

        from nomorals.media import resolver as res_mod
        src = inspect.getsource(res_mod.SourceResolver._resolve_text)
        ja = src.index("jamendo-search")
        fm = src.index("fma-search")
        bp = src.index("boomplay-search")
        ov = src.index("openverse-search")
        assert ja < fm < bp
        assert fm < ov

    def test_text_chain_does_not_claim_soundclick(self):
        # SoundClick is NOT in the chain, deliberately (CENSUS B4):
        # verified live 2026-11-12, the site has no public API and
        # no anonymous server-rendered search — the search URL
        # redirects to a JS-driven discover page whose anonymous
        # HTML carries only promo songs unrelated to the query, so
        # there is no honest source class to wire (parsing the
        # promos as results would return wrong tracks; a JSON-API
        # client would be fake).  Remove this pin only when a
        # live-browser probe pins the framework search and a
        # track-page audio URL, and wire the real shape then.
        from nomorals.media import sources as src_mod

        assert "soundclick-search" not in src_mod.MUSIC_SEARCH_SOURCES
        assert all(name != "soundclick-search"
                   for name, _ in src_mod.SOURCE_CHAIN)
        assert not hasattr(src_mod, "SoundClickSource")

    def test_text_chain_does_not_claim_ghanamusic(self):
        # GhanaMusic is NOT in the chain, deliberately (CENSUS B6):
        # verified 2026-11-12, a Hostinger JS challenge 403s every
        # anonymous fetch (home, wp-json, search), so the chain's
        # plain fetch reads nothing; and a real music post carries
        # zero .mp3 links and zero audio tags — the only listen path
        # is an Apple Music embed plus an ffm.to smart-link to
        # streaming stores — so there is no honest source class to
        # wire (a NetNaija-shaped parser would return editorial
        # articles as song candidates with nothing to download; a
        # JSON-API client would be fake).  Remove this pin only when
        # a live-browser probe passes the challenge and pins a
        # track-page audio URL, and wire the real shape then.
        from nomorals.media import sources as src_mod

        assert "ghanamusic-search" not in src_mod.MUSIC_SEARCH_SOURCES
        assert all(name != "ghanamusic-search"
                   for name, _ in src_mod.SOURCE_CHAIN)
        assert not hasattr(src_mod, "GhanaMusicSource")

    def test_text_chain_does_not_claim_palcomp3(self):
        # Palco MP3 is NOT in the chain, deliberately (CENSUS B7):
        # verified 2026-11-12, an Akamai bot wall 403s every plain
        # fetch from this box (curl AND urllib with full browser
        # headers — home, /graphql/ and search alike), so the chain's
        # plain fetch reads nothing; and no search shape is verified
        # — /busca?q= is 404, /pesquisa/ is an artist page, and the
        # yt-dlp PalcoMP3 extractor (which proves the unauthenticated
        # GraphQL API returning mp3File direct MP3s) carries no search
        # — so there is no honest source class to wire (a JSON client
        # would have to invent a GraphQL search field; an HTML parser
        # would have to invent a search path).  Remove this pin only
        # when a live-browser probe pins the search URL shape,
        # confirms the GraphQL endpoint answers from a browser
        # session, and checks the schema for a search field, and wire
        # the real shape then.
        from nomorals.media import sources as src_mod

        assert "palcomp3-search" not in src_mod.MUSIC_SEARCH_SOURCES
        assert all(name != "palcomp3-search"
                   for name, _ in src_mod.SOURCE_CHAIN)
        assert not hasattr(src_mod, "PalcoMP3Source")

    def test_text_chain_includes_new_sources(self):
        # The chain tries netnaija before the proven globals.
        # Boomplay is BACK in the chain (browser-backed search,
        # verified live 2026-10-18) after archiveorg, before the
        # metadata-only sources.
        from nomorals.media import resolver as res_mod
        import inspect
        src = inspect.getsource(res_mod.SourceResolver._resolve_text)
        assert "netnaija-search" in src
        assert "boomplay-search" in src
        nn = src.index("netnaija-search")
        sc = src.index("soundcloud-search")
        yt = src.index("youtube-search")
        assert nn < sc < yt
        ao = src.index("archiveorg-search")
        bp = src.index("boomplay-search")
        mb = src.index("musicbrainz-search")
        assert ao < bp < mb

    def test_url_chain_tries_audiomack_first(self):
        from nomorals.media import resolver as res_mod
        import inspect
        src = inspect.getsource(res_mod.SourceResolver._resolve_url)
        assert src.index('"audiomack"') < src.index('"yt-dlp"')


# ── playback download() ───────────────────────────────────────────────


def _engine(tmp_path=None, **kwargs):
    """PlaybackEngine with stubbed tools + YouTube methods."""
    from nomorals.media.playback import PlaybackEngine
    ctx = MagicMock()
    tools = MagicMock()

    dl_path = "/tmp/dl.mp3"
    if tmp_path is not None:
        dl_path = str(tmp_path / "dl.mp3")
        open(dl_path, "w").write("fake mp3")

    def fake_call(name, **kw):
        exc = kwargs.get("download_exc")
        if exc is not None:
            raise exc
        res = kwargs.get("download_result")
        if res is not None:
            return res
        return SimpleNamespace(ok=True, value={"path": dl_path})

    tools.call.side_effect = fake_call
    ctx.tools = tools
    eng = PlaybackEngine(ctx)
    eng._youtube_search_id = MagicMock(
        return_value=kwargs.get("yt_search_id", "dQw4w9WgXcQ"))
    eng._youtube_audio = MagicMock(
        return_value=kwargs.get("yt_audio_path", "/tmp/yt.mp3"))
    return eng


class TestPlaybackNewKinds:
    def test_boomplay_honest_refusal(self):
        eng = _engine()
        out = eng.download({"kind": "boomplay", "title": "Last Last",
                            "path": "https://www.boomplay.com/songs/1_x"})
        assert out["ok"] is False
        assert "protected" in out["reason"]

    def test_audiomack_download_via_ytdlp(self, tmp_path):
        eng = _engine(tmp_path)
        out = eng.download({"kind": "audiomack", "title": "Last Last",
                            "path": "https://audiomack.com/b/song/l"})
        assert out["ok"] is True
        assert out["path"] == str(tmp_path / "dl.mp3")

    def test_audiomack_protection_error_falls_back_to_youtube(self):
        eng = _engine()
        out = eng.download({"kind": "audiomack", "title": "Last Last",
                            "artist": "Burna Boy",
                            "path": "https://audiomack.com/b/song/l"},
                           )
        # force the protection error through the mock
        eng2 = _engine(download_exc=Exception("This video is DRM protected"))
        out = eng2.download({"kind": "audiomack", "title": "Last Last",
                             "artist": "Burna Boy",
                             "path": "https://audiomack.com/b/song/l"})
        assert out["ok"] is True
        assert out["path"] == "/tmp/yt.mp3"
        assert "Audiomack" in out["note"]

    def test_netnaija_resolves_direct_then_downloads(self, tmp_path):
        eng = _engine(tmp_path)
        with patch.object(NetNaijaSource, "download_url",
                          return_value="https://cdn.example.com/x.mp3"):
            out = eng.download({"kind": "netnaija", "title": "Last Last",
                                "path": "https://netnaija.com/music/x/"})
        assert out["ok"] is True

    def test_netnaija_no_direct_falls_back_to_youtube(self):
        eng = _engine()
        with patch.object(NetNaijaSource, "download_url", return_value=""):
            out = eng.download({"kind": "netnaija", "title": "Last Last",
                                "artist": "Burna Boy",
                                "path": "https://netnaija.com/music/x/"})
        assert out["ok"] is True
        assert out["path"] == "/tmp/yt.mp3"
        assert "NetNaija" in out["note"]

    def test_netnaija_never_raises(self):
        eng = _engine()
        with patch.object(NetNaijaSource, "download_url",
                          side_effect=Exception("boom")):
            out = eng.download({"kind": "netnaija", "title": "X",
                                "path": "https://netnaija.com/music/x/"})
        assert isinstance(out, dict) and "ok" in out
