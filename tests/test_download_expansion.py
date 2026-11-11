"""Downloads expansion pins — webdrive census build shape B1–B3.

B1  SoundCloud precision: the track payload's ``track_authorization``
    is forwarded on the transcoding call; blocks are STRUCTURED
    (preview / drm / region / login / private) and the class reaches
    the resolver and the chat surfaces; a policy-blocked track that a
    different source serves is DECLARED a substitute in the delivered
    line, and the single-item send path (one specific track was
    asked for) does not substitute at all.
B2  The connector's resolved stream URL is a first-class rung in
    the MediaDownloader ladder, after direct/kit, before proxy.
B3  The rendered-capture stage: a fake tab whose network log carries
    an audio URL gets THAT URL fetched through the tab session; a
    DRM-shaped log yields the DRM reason and no download; the stage
    sits between proxy and the static browser stage.

Everything is faked at the seams — no network, no browser engine.
"""

from __future__ import annotations

import json
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from nomorals.connectors.soundcloud import (
    SoundCloudError,
    block_reason_text,
    classify_block,
)

# ── shared fakes ─────────────────────────────────────────────────────


class FakeResponse:
    def __init__(self, status: int = 200, payload: Any = None,
                 text: str = "") -> None:
        self.status = status
        self._payload = payload
        self._text = text
        self.headers: dict[str, str] = {}

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    @property
    def text(self) -> str:
        if self._text:
            return self._text
        return json.dumps(self._payload)

    def json(self) -> Any:
        if self._payload is not None:
            return self._payload
        return json.loads(self._text)


class FakeHttp:
    """Scripted stand-in for the connector's HttpClient."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.routes: list[tuple[str, str, FakeResponse]] = []

    def route(self, method: str, path: str, response: FakeResponse):
        self.routes.append((method.upper(), path, response))

    def get(self, url: str, **kw: Any) -> FakeResponse:
        params = kw.pop("params", None) or {}
        if params:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{urllib.parse.urlencode(params)}"
        self.calls.append(("GET", url))
        for rm, rp, resp in sorted(self.routes, key=lambda r: -len(r[1])):
            if rm == "GET" and rp in url:
                return resp
        return FakeResponse(404, {"error": "not mocked"})


CID = "a" * 32

TRACK_AUTH = {
    "id": 111, "kind": "track", "title": "Midnight Drive",
    "duration": 180000, "genre": "Synthwave",
    "artwork_url": "https://i1.sndcdn.com/artworks-abc-large.jpg",
    "permalink_url": "https://soundcloud.com/artist/midnight-drive",
    "playback_count": 1000, "likes_count": 50,
    "policy": "ALLOW", "streamable": True,
    "track_authorization": "tok-abc-123",
    "user": {"username": "artist"},
    "media": {"transcodings": [
        {"url": "https://api-v2.soundcloud.com/media/1/stream",
         "format": {"protocol": "progressive",
                    "mime_type": "audio/mpeg"}},
    ]},
}


def _connector(http: FakeHttp):
    from nomorals.accounts.vault import CredentialVault
    from nomorals.connectors.soundcloud import SoundCloudConnector, _reset_client_id_cache
    from nomorals.storage.db import Database

    vault = CredentialVault(Database(":memory:"),
                            master_passphrase="test")
    conn = SoundCloudConnector(vault, http=http)
    _reset_client_id_cache()
    http.route("GET", "https://soundcloud.com", FakeResponse(
        200, text='<html><script src="https://a-v2.sndcdn.com/assets/'
                  '0-abc-3.js"></script></html>'))
    http.route("GET", "https://a-v2.sndcdn.com/assets/0-abc-3.js",
               FakeResponse(200, text=f'webpack{{client_id:"{CID}"}}'))
    return conn


@pytest.fixture(autouse=True)
def _no_env_pin(monkeypatch):
    monkeypatch.delenv("SOUNDCLOUD_CLIENT_ID", raising=False)


# ── B1: track_authorization on the wire ──────────────────────────────


class TestTrackAuthorization:
    def test_token_forwarded_on_transcoding_call(self):
        http = FakeHttp()
        conn = _connector(http)
        http.route(
            "GET", "https://api-v2.soundcloud.com/media/1/stream",
            FakeResponse(200, {"url": "https://cf-media.sndcdn.com/p.mp3"}))
        out = conn.stream_url(TRACK_AUTH)
        assert out["url"].endswith("p.mp3")
        stream_calls = [u for _m, u in http.calls if "/media/" in u]
        assert stream_calls, "transcoding call never happened"
        wire = stream_calls[0]
        assert "track_authorization=tok-abc-123" in wire
        assert f"client_id={CID}" in wire

    def test_no_token_no_param(self):
        http = FakeHttp()
        conn = _connector(http)
        track = {k: v for k, v in TRACK_AUTH.items()
                 if k != "track_authorization"}
        http.route(
            "GET", "https://api-v2.soundcloud.com/media/1/stream",
            FakeResponse(200, {"url": "https://cf-media.sndcdn.com/p.mp3"}))
        conn.stream_url(track)
        stream_calls = [u for _m, u in http.calls if "/media/" in u]
        assert stream_calls
        assert "track_authorization" not in stream_calls[0]

    def test_token_survives_id_lookup(self):
        # stream_url by id: the FULL payload (with its token) is
        # fetched first, and the token rides the transcoding call.
        http = FakeHttp()
        conn = _connector(http)
        http.route("GET", "/tracks/111", FakeResponse(200, TRACK_AUTH))
        http.route(
            "GET", "https://api-v2.soundcloud.com/media/1/stream",
            FakeResponse(200, {"url": "https://cf-media.sndcdn.com/p.mp3"}))
        conn.stream_url(111)
        stream_calls = [u for _m, u in http.calls if "/media/" in u]
        assert any("track_authorization=tok-abc-123" in u
                   for u in stream_calls)


# ── B1: structured block classes ─────────────────────────────────────


class TestBlockClasses:
    def test_classify_block_vocabulary(self):
        assert classify_block(policy="PREVIEW") == "preview"
        assert classify_block(policy="SNIP") == "preview"
        assert classify_block(policy="BLOCK") == "region"
        assert classify_block(status=403) == "login"
        assert classify_block(text="This video is DRM protected") == "drm"
        assert classify_block(
            text="not available in your country") == "region"
        assert classify_block(policy="ALLOW") == ""
        assert block_reason_text("preview")
        assert block_reason_text("") == ""

    def test_preview_policy_error_carries_class(self):
        http = FakeHttp()
        conn = _connector(http)
        track = dict(TRACK_AUTH, policy="PREVIEW",
                     media={"transcodings": []})
        with pytest.raises(SoundCloudError) as ei:
            conn.stream_url(track)
        assert ei.value.reason == "preview"
        assert "preview" in str(ei.value)

    def test_region_policy_error_carries_class(self):
        http = FakeHttp()
        conn = _connector(http)
        track = dict(TRACK_AUTH, policy="BLOCK",
                     media={"transcodings": []})
        with pytest.raises(SoundCloudError) as ei:
            conn.stream_url(track)
        assert ei.value.reason == "region"

    def test_transcoding_refusal_is_login_class(self):
        http = FakeHttp()
        conn = _connector(http)
        http.route(
            "GET", "https://api-v2.soundcloud.com/media/1/stream",
            FakeResponse(403, text="unauthorized"))
        with pytest.raises(SoundCloudError) as ei:
            conn.stream_url(TRACK_AUTH)
        assert ei.value.reason == "login"
        assert ei.value.status_code == 403

    def test_preview_stream_url_is_preview_class(self):
        http = FakeHttp()
        conn = _connector(http)
        http.route(
            "GET", "https://api-v2.soundcloud.com/media/1/stream",
            FakeResponse(200, {
                "url": "https://cf-media.sndcdn.com/preview/xyz.mp3"}))
        with pytest.raises(SoundCloudError) as ei:
            conn.stream_url(TRACK_AUTH)
        assert ei.value.reason == "preview"

    def test_summary_carries_block_reason(self):
        from nomorals.connectors.soundcloud import SoundCloudConnector

        blocked = SoundCloudConnector.summarize_track(
            dict(TRACK_AUTH, policy="SNIP", streamable=False))
        assert blocked["block_reason"] == "preview"
        free = SoundCloudConnector.summarize_track(TRACK_AUTH)
        assert free["block_reason"] == ""


class TestResolverCarriesBlock:
    def test_blocked_track_reaches_resolution_with_class(self):
        from nomorals.media.resolver import SourceResolver

        summary = {
            "id": 5, "title": "Gated Song", "artist": "Ayo Maff",
            "duration_ms": 200000, "genre": "",
            "artwork_url": "",
            "permalink_url": "https://soundcloud.com/ayo/gated",
            "playback_count": 0, "likes_count": 0, "policy": "SNIP",
            "streamable": False, "block_reason": "preview",
            "kind": "track", "_full": None,
        }
        adapter = SimpleNamespace(
            resolve=lambda url: {"kind": "track", "track": summary})
        resolver = SourceResolver(SimpleNamespace(), soundcloud=adapter)
        out = resolver.resolve("https://soundcloud.com/ayo/gated")
        assert out.ok is True            # found…
        assert out.downloadable is False  # …but blocked
        assert out.block_reason == "preview"
        assert out.preview_only is True
        assert out.kind == "soundcloud"


class TestPlaybackSubstitutionFields:
    def _engine(self):
        from nomorals.media.playback import PlaybackEngine

        ctx = mock.MagicMock()
        tools = mock.MagicMock()

        def fake_call(name, **kwargs):
            raise Exception(
                "[soundcloud] 2361175013: This video is DRM protected")

        tools.call = fake_call
        ctx.tools = tools
        engine = PlaybackEngine.__new__(PlaybackEngine)
        engine.context = ctx
        engine._soundcloud_play_url = lambda item: "https://sc/x"  # noqa: E731
        engine._youtube_search_id = lambda query: "dQw4w9WgXcQ"  # noqa: E731
        engine._youtube_audio = lambda item: "/tmp/song.mp3"  # noqa: E731
        return engine

    def test_fallback_result_declares_substitution(self):
        engine = self._engine()
        res = engine.download({
            "kind": "soundcloud", "path": "https://soundcloud.com/a/b",
            "title": "Lifestyle (YA MAN)", "artist": "Ayo Maff"})
        assert res["ok"] is True
        assert res["substituted_from"] == "SoundCloud"
        assert "DRM" in res["substitute_reason"]
        assert "YouTube" in res["note"]


# ── B1: chat surfaces — announced vs never for a specific track ──────


class _Gateway:
    def __init__(self):
        self.files = []

    def send_file(self, platform, target, path, caption="", kind=None):
        self.files.append((platform, target, path, caption))
        return SimpleNamespace(ok=True, platform=platform,
                               message_id="f1")


def _handler():
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    rt = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    rt.context = SimpleNamespace()
    rt.gateway = _Gateway()
    return rt


def _chat():
    chat = mock.Mock()
    chat.platform = "telegram"
    chat.chat_id = "12345"
    chat.key = "telegram:12345"
    return chat


MATCH = ("Lifestyle (YA MAN)", "Ayo Maff")
SC_BLOCKED = {
    "kind": "soundcloud",
    "path": "https://soundcloud.com/ayo/lifestyle",
    "title": "Lifestyle (YA MAN)", "artist": "Ayo Maff",
    "downloadable": False, "block_reason": "preview",
}
YT_ITEM = {
    "kind": "youtube",
    "path": "https://www.youtube.com/watch?v=abcdefghijk",
    "title": "Lifestyle (YA MAN)", "artist": "Ayo Maff",
}


class TestSubstitutionSurfaces:
    def test_prejudged_block_substitution_is_announced(self):
        rt = _handler()
        engine = mock.Mock()
        engine.queue.return_value = []
        engine.download.return_value = {
            "ok": True, "path": "/tmp/yt.mp3",
            "title": "Lifestyle (YA MAN)"}
        out, sent = rt._play_send_with_fallthrough(
            engine, dict(SC_BLOCKED), [dict(YT_ITEM)], [], _chat(),
            match=MATCH, query="lifestyle by Ayo Maff")
        assert sent is True
        assert len(rt.gateway.files) == 1
        # The delivered line still leads with the track…
        assert out.startswith("🎵 Lifestyle (YA MAN)")
        # …and DECLARES the substitution + the specific block.
        assert "substitute" in out
        assert "preview" in out

    def test_same_source_winner_is_not_a_substitution(self):
        rt = _handler()
        engine = mock.Mock()
        engine.queue.return_value = []
        engine.download.return_value = {
            "ok": True, "path": "/tmp/sc.mp3",
            "title": "Lifestyle (YA MAN)"}
        good_sc = dict(SC_BLOCKED)
        del good_sc["downloadable"]
        # A different SoundCloud upload of the same song (a distinct
        # candidate — same kind+path would dedupe to the blocked one).
        good_sc["path"] = "https://soundcloud.com/other/lifestyle-cover"
        out, sent = rt._play_send_with_fallthrough(
            engine, dict(SC_BLOCKED), [good_sc], [], _chat(),
            match=MATCH, query="lifestyle by Ayo Maff")
        assert sent is True
        assert "substitute" not in out

    def test_specific_track_send_never_substitutes(self):
        rt = _handler()
        engine = mock.Mock()
        engine.queue.return_value = []
        engine.download.return_value = {
            "ok": True, "path": "/tmp/yt-lookalike.mp3",
            "title": "Lifestyle (YA MAN)",
            "note": "via YouTube (SoundCloud was blocked)",
            "substituted_from": "SoundCloud",
            "substitute_reason": "This video is DRM protected"}
        item = {"kind": "soundcloud",
                "path": "https://soundcloud.com/ayo/lifestyle",
                "title": "Lifestyle (YA MAN)", "artist": "Ayo Maff"}
        out, sent = rt._play_send_in_chat_result(engine, item, _chat())
        # The exact track was asked for: the lookalike is NOT sent,
        # and the refusal names the block + how to get a copy.
        assert sent is False
        assert rt.gateway.files == []
        assert "blocked" in out
        assert "DRM" in out
        assert "by name" in out


# ── B2: connector stream as a ladder rung ────────────────────────────


class _FakeKit:
    def __init__(self, payload: bytes = b"connector-bytes"):
        self.payload = payload
        self.calls: list[str] = []

    def download(self, url, dest, progress=None):
        self.calls.append(url)
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(self.payload)
        return SimpleNamespace(
            ok=True, path=str(dest), size_bytes=len(self.payload),
            error="", resumed=False, restarted=False)


def _no_descriptor_extractors():
    return SimpleNamespace(best_for=lambda url: None)


def _fail_direct(*a, **k):
    raise RuntimeError("connection reset by peer")


class TestConnectorStage:
    def _downloader(self, tmp, connector):
        from nomorals.media.downloader import MediaDownloader

        return MediaDownloader(
            None, direct_retries=1, soundcloud_connector=connector,
            kit_downloader=_FakeKit(),
            extractors=_no_descriptor_extractors())

    def test_stream_url_is_a_rung_after_direct(self, tmp_path):
        from unittest.mock import patch

        connector = SimpleNamespace(stream_url=lambda url: {
            "url": "https://cf-media.sndcdn.com/prog.mp3",
            "protocol": "progressive", "mime_type": "audio/mpeg",
            "track": {"title": "Midnight Drive",
                      "permalink_url":
                          "https://soundcloud.com/artist/midnight-drive"}})
        dl = self._downloader(tmp_path, connector)
        with patch("nomorals.tools.media.download",
                   side_effect=_fail_direct):
            report = dl.download(
                "https://soundcloud.com/artist/midnight-drive",
                dest_dir=tmp_path, audio_only=True)
        assert report.ok
        assert report.extractor == "soundcloud-connector"
        assert report.title == "Midnight Drive"
        # The kit fetched the RESOLVED STREAM URL, not the page.
        assert dl._kit().calls == ["https://cf-media.sndcdn.com/prog.mp3"]
        # Ladder order: direct was tried first, connector served.
        assert report.tried() == ["direct", "connector"]
        assert Path(report.path).read_bytes() == b"connector-bytes"

    def test_blocked_connector_journals_class_and_falls_through(
            self, tmp_path):
        from unittest.mock import patch

        def blocked(url):
            raise SoundCloudError(
                "no playable stream: SoundCloud blocks this track "
                "in this region", reason="region")

        dl = self._downloader(tmp_path, SimpleNamespace(
            stream_url=blocked))
        order = []

        def browser(report, *a, **k):
            from nomorals.media.downloader import StageLog
            order.append("browser")
            report.stages.append(StageLog(stage="browser",
                                          note="mocked"))
            return False

        with patch("nomorals.tools.media.download",
                   side_effect=_fail_direct), patch.object(dl, "_stage_browser",
                          side_effect=browser):
            report = dl.download(
                "https://soundcloud.com/artist/gated",
                dest_dir=tmp_path, audio_only=True)
        assert report.ok is False
        conn_logs = [s for s in report.stages if s.stage == "connector"]
        assert conn_logs and "region" in conn_logs[0].note
        # The chain kept going after the connector's answer.
        assert order == ["browser"]
        assert report.tried()[-1] == "browser"

    def test_non_soundcloud_url_never_engages_connector(self, tmp_path):
        from unittest.mock import patch

        connector = SimpleNamespace(
            stream_url=mock.Mock(side_effect=AssertionError("engaged")))
        dl = self._downloader(tmp_path, connector)
        with patch("nomorals.tools.media.download",
                   side_effect=_fail_direct), patch.object(dl, "_stage_browser",
                          return_value=False):
            report = dl.download("https://example.com/watch?v=1",
                                 dest_dir=tmp_path)
        assert report.ok is False
        assert "connector" not in report.tried()
        connector.stream_url.assert_not_called()


# ── B3: rendered capture stage ───────────────────────────────────────


class FakeTab:
    """A rendered tab fake: scripted network log + session fetch."""

    def __init__(self, entries, payloads=None):
        self.entries = entries
        self.payloads = payloads or {}
        self.fetched: list[str] = []
        self.clicked: list[str] = []
        self.closed = False

    def network_requests(self, limit=100, failed_only=False):
        return {"requests": list(self.entries), "tab_id": "fake"}

    def click(self, target):
        self.clicked.append(target)
        raise RuntimeError("no such element")

    def wait_for_network_idle(self, timeout=15000):
        return {"ok": True}

    def session_fetch_bytes(self, url):
        self.fetched.append(url)
        if url in self.payloads:
            return self.payloads[url]
        raise RuntimeError(f"HTTP 404 for {url}")

    def close(self):
        self.closed = True


def _resp(url, ctype="", size=0, status=200):
    entry = {"kind": "response", "method": "GET", "url": url,
             "status": status, "ts": 0.0}
    if ctype:
        entry["content_type"] = ctype
    if size:
        entry["content_length"] = size
    return entry


def _rendered_downloader(tmp, tab):
    from nomorals.media.downloader import MediaDownloader

    def _blocked(url):
        raise SoundCloudError("stream needs a signed-in account",
                              reason="login")

    return MediaDownloader(
        None, direct_retries=1,
        rendered_tab_factory=lambda page_url: tab,
        soundcloud_connector=SimpleNamespace(stream_url=_blocked),
        extractors=_no_descriptor_extractors())


class TestRenderedStage:
    def test_captures_audio_url_from_network_log(self, tmp_path):
        from unittest.mock import patch

        media_url = "https://cf-media.sndcdn.com/abc123.mp3"
        tab = FakeTab(
            entries=[
                _resp("https://soundcloud.com/artist/track",
                      "text/html", 5000),
                _resp(media_url, "audio/mpeg", 4_000_000),
            ],
            payloads={media_url: b"captured-audio-bytes"})
        dl = _rendered_downloader(tmp_path, tab)
        with patch("nomorals.tools.media.download",
                   side_effect=_fail_direct):
            report = dl.download(
                "https://soundcloud.com/artist/track",
                dest_dir=tmp_path, audio_only=True)
        assert report.ok
        assert report.extractor == "rendered-capture"
        # THAT url, fetched through the tab's session.
        assert tab.fetched == [media_url]
        assert Path(report.path).read_bytes() == b"captured-audio-bytes"
        assert tab.closed is True
        # Ladder order: rendered served before the browser stage,
        # which never ran.
        assert "browser" not in report.tried()
        assert report.tried()[-1] == "rendered"

    def test_drm_log_yields_drm_reason_and_no_download(self, tmp_path):
        from unittest.mock import patch

        tab = FakeTab(entries=[
            _resp("https://site.test/watch/1", "text/html", 4000),
            {"kind": "request", "method": "POST",
             "url": "https://widevine-license.example.com/acquire",
             "ts": 0.0},
            _resp("https://cdn.test/seg-1.m4s", "video/iso.segment", 900),
        ])
        dl = _rendered_downloader(tmp_path, tab)
        with patch("nomorals.tools.media.download",
                   side_effect=_fail_direct), patch.object(dl, "_stage_browser",
                          return_value=False):
            report = dl.download("https://site.test/watch/1",
                                 dest_dir=tmp_path)
        assert report.ok is False
        assert tab.fetched == []          # nothing was downloaded
        assert "DRM" in report.likely_cause
        rendered_logs = [s for s in report.stages
                         if s.stage == "rendered"]
        assert rendered_logs and "DRM" in rendered_logs[-1].error

    def test_hls_capture_walks_segments_through_session(self, tmp_path):
        from unittest.mock import patch

        playlist_url = "https://cf-hls.sndcdn.com/pl.m3u8"
        playlist = ("#EXTM3U\n#EXT-X-TARGETDURATION:10\n"
                    "#EXTINF:10,\nseg-1.mp3\n#EXTINF:10,\nseg-2.mp3\n"
                    "#EXT-X-ENDLIST\n")
        tab = FakeTab(
            entries=[_resp(playlist_url, "application/x-mpegURL", 300)],
            payloads={
                playlist_url: playlist.encode(),
                "https://cf-hls.sndcdn.com/seg-1.mp3": b"SEG1-",
                "https://cf-hls.sndcdn.com/seg-2.mp3": b"SEG2",
            })
        dl = _rendered_downloader(tmp_path, tab)
        with patch("nomorals.tools.media.download",
                   side_effect=_fail_direct):
            report = dl.download(
                "https://soundcloud.com/artist/hls-track",
                dest_dir=tmp_path, audio_only=True)
        assert report.ok
        assert Path(report.path).read_bytes() == b"SEG1-SEG2"
        assert tab.fetched[0] == playlist_url

    def test_no_media_in_log_falls_through_to_browser(self, tmp_path):
        from unittest.mock import patch

        tab = FakeTab(entries=[
            _resp("https://site.test/page", "text/html", 3000)])
        dl = _rendered_downloader(tmp_path, tab)
        order = []

        def browser(report, *a, **k):
            from nomorals.media.downloader import StageLog
            order.append("browser")
            report.stages.append(StageLog(stage="browser",
                                          note="mocked"))
            return False

        with patch("nomorals.tools.media.download",
                   side_effect=_fail_direct), patch.object(dl, "_stage_browser",
                          side_effect=browser):
            # A short chain timeout bounds the capture poll:
            # the log never gains media, so the stage must give
            # up inside the budget, not hang the chain.
            report = dl.download("https://site.test/page",
                                 dest_dir=tmp_path, timeout=2.0)
        assert report.ok is False
        tried = report.tried()
        assert tried.index("rendered") < tried.index("browser")


class TestCaptureSelection:
    def test_audio_first_and_preview_excluded(self):
        from nomorals.media.downloader import _capture_candidates

        entries = [
            _resp("https://cdn.test/movie.mp4", "video/mp4", 50_000_000),
            _resp("https://cdn.test/song.mp3", "audio/mpeg", 4_000_000),
            _resp("https://cdn.test/preview/clip.mp3", "audio/mpeg", 100),
            _resp("https://cdn.test/pl.m3u8",
                  "application/x-mpegURL", 200),
        ]
        got = _capture_candidates(entries, audio_only=True)
        assert got[0] == "https://cdn.test/song.mp3"
        assert not any("preview" in u for u in got)
        assert "https://cdn.test/pl.m3u8" in got

    def test_drm_evidence_marker(self):
        from nomorals.media.downloader import _drm_evidence

        entries = [{"kind": "request", "method": "POST",
                    "url": "https://lic.example.com/widevine/v1",
                    "ts": 0.0}]
        assert _drm_evidence(entries).endswith("widevine/v1")
        assert _drm_evidence([]) == ""


class TestNetworkLogEvidence:
    def test_response_entries_carry_content_type_and_length(self):
        from unittest.mock import Mock

        from nomorals.browser.service import RenderedTab

        tab = RenderedTab("t1", "s", "state.json", Mock())
        response = SimpleNamespace(
            url="https://cdn.test/song.mp3", status=200,
            request=SimpleNamespace(method="GET"),
            headers={"content-type": "audio/mpeg",
                     "content-length": "4000000"})
        tab._on_response(response)
        entry = tab._network_log[-1]
        assert entry["content_type"] == "audio/mpeg"
        assert entry["content_length"] == 4000000
        assert entry["status"] == 200
