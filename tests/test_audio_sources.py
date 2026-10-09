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
    NetNaijaSource,
    SourceCandidate,
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
        src = BoomplaySource()
        with patch.object(BoomplaySource, "_get",
                          return_value=_BOOMPLAY_SEARCH_HTML):
            cands = src.search("burna", limit=5)
        assert len(cands) == 2
        assert cands[0].title == "Last Last"
        assert cands[0].url == "https://www.boomplay.com/songs/211634092_last-last"
        # honestly non-downloadable
        assert all(c.downloadable is False for c in cands)

    def test_search_empty(self):
        src = BoomplaySource()
        assert src.search("") == []
        with patch.object(BoomplaySource, "_get", return_value=""):
            assert src.search("x") == []

    def test_search_never_raises(self):
        src = BoomplaySource()
        with patch.object(BoomplaySource, "_get",
                          side_effect=Exception("boom")):
            assert src.search("x") == []

    def test_download_url_empty(self):
        src = BoomplaySource()
        c = SourceCandidate(source="boomplay", title="X",
                            url="https://www.boomplay.com/songs/1_x")
        assert src.download_url(c) == ""


# ── BaseScrapeSource helpers ──────────────────────────────────────────


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

    def test_text_chain_includes_new_sources(self):
        # The chain tries netnaija before the proven globals and
        # boomplay (metadata-only) after them.
        from nomorals.media import resolver as res_mod
        import inspect
        src = inspect.getsource(res_mod.SourceResolver._resolve_text)
        assert "netnaija-search" in src
        assert "boomplay-search" in src
        nn = src.index("netnaija-search")
        sc = src.index("soundcloud-search")
        yt = src.index("youtube-search")
        bp = src.index("boomplay-search")
        assert nn < sc < yt < bp

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
