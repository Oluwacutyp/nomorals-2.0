"""SoundCloud blocked audios (audit N3) — the credential mechanism.

Owner evidence (2026-11-11): "all those blocked audios on SoundCloud
how on earth do people do it… In short our downloads need to start
expanding using webs even."  At audit time there was no cookie /
client_id mechanism in ``nomorals/media/*.py`` and SoundCloud URLs
kept 403ing.

What this file pins (all network-free — scripted fake HTTP):

* the connector rediscovers its client_id on 403 as well as 401
  (a rate-limited or revoked id can refuse either way — yt-dlp's
  own SoundCloud extractor retries on both), and never
  rediscovers over an env pin;
* the connector falls back to the browser session's homepage HTML
  when the plain HTTP fetch fails (the web-expansion rung);
* the connector forwards the owner's ``oauth_token`` session
  cookie on api-v2 calls (the cookie SoundCloud's web player sets
  — yt-dlp reads exactly that name from the cookie jar);
* a snipped transcoding is a preview on BOTH protocols (the old
  shape checked only ``"/preview/"`` in the resolved URL, so an
  HLS preview landed as success while a progressive preview
  raised);
* ``media.soundcloud_credentials`` resolves env pin → vault →
  live discovery and persists discoveries in the vault;
* the SoundCloud descriptor names the real session cookie
  (``oauth_token``, not ``auth_token``);
* the downloader arms the connector (client_id + session) before
  the connector stage, and the failure report names the
  SoundCloud block class / action instead of the YouTube help.

No new brain tool: the download already routes through the
existing chain, so the registry stays at 785 tools.
"""

from __future__ import annotations

import json
import os
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors.soundcloud import (
    SoundCloudConnector,
    SoundCloudError,
    _reset_client_id_cache,
)
from nomorals.storage.db import Database

CID_OLD = "a" * 32
CID_NEW = "b" * 32
CID_ENV = "c" * 32

TRACK_FULL = {
    "id": 111, "kind": "track", "title": "Midnight Drive",
    "duration": 180000, "genre": "Synthwave",
    "artwork_url": "https://i1.sndcdn.com/artworks-abc-large.jpg",
    "permalink_url": "https://soundcloud.com/artist/midnight-drive",
    "playback_count": 1000, "likes_count": 50,
    "policy": "ALLOW", "streamable": True,
    "user": {"username": "artist"},
    "media": {"transcodings": [
        {"url": "https://api-v2.soundcloud.com/media/1/stream",
         "preset": "mp3_0_1", "duration": 180000, "snipped": False,
         "format": {"protocol": "progressive",
                    "mime_type": "audio/mpeg"}},
        {"url": "https://api-v2.soundcloud.com/media/1/hls",
         "preset": "mp3_0_1", "duration": 180000, "snipped": False,
         "format": {"protocol": "hls",
                    "mime_type": "application/x-mpegURL"}},
    ]},
}


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
    """Scripted stand-in for HttpClient. No network."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.routes: list[tuple[str, str, FakeResponse]] = []

    def route(self, method: str, path: str, response: FakeResponse) -> None:
        self.routes.append((method.upper(), path, response))

    def _dispatch(self, method: str, url: str, **kw: Any) -> FakeResponse:
        self.calls.append((method.upper(), url, dict(kw.get("headers")
                                                     or {})))
        for rm, rp, resp in sorted(self.routes, key=lambda r: -len(r[1])):
            if rm == method.upper() and rp in url:
                return resp
        return FakeResponse(404, {"error": "not mocked"})

    def get(self, url: str, **kw: Any) -> FakeResponse:
        params = kw.pop("params", None) or {}
        if params:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{urllib.parse.urlencode(params)}"
        return self._dispatch("GET", url, **kw)


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test")


def _connector(http: FakeHttp | None = None, *,
               page_fetcher: Any = None) -> SoundCloudConnector:
    http = http or FakeHttp()
    conn = SoundCloudConnector(_vault(), http=http,
                               page_fetcher=page_fetcher)
    _reset_client_id_cache()
    return conn


def _discovery_routes(http: FakeHttp, cid: str = CID_NEW) -> None:
    """Homepage + bundle routes so a fresh discovery finds *cid*."""
    http.route("GET", "https://soundcloud.com", FakeResponse(
        200, text='<html><script src="https://a-v2.sndcdn.com/assets/'
                  '0-abc-3.js"></script></html>'))
    http.route("GET", "https://a-v2.sndcdn.com/assets/0-abc-3.js",
               FakeResponse(200, text=f'webpack{{client_id:"{cid}"}}'))


@pytest.fixture(autouse=True)
def _no_env_pin(monkeypatch):
    monkeypatch.delenv("SOUNDCLOUD_CLIENT_ID", raising=False)


# ── 403 rediscovery ──────────────────────────────────────────────────


class TestClientId403Retry:
    def test_403_invalidates_and_rediscovers_once(self):
        http = FakeHttp()
        _discovery_routes(http)
        http.route("GET", "/tracks/111?client_id=" + CID_OLD,
                   FakeResponse(403, {"error": "Forbidden"}))
        http.route("GET", "/tracks/111?client_id=" + CID_NEW,
                   FakeResponse(200, TRACK_FULL))
        conn = _connector(http)
        assert conn.set_client_id(CID_OLD)
        got = conn.get_track(111)
        assert got["id"] == 111
        assert any(CID_NEW in url for _m, url, _h in http.calls)

    def test_403_after_refresh_reports_the_refusal(self):
        http = FakeHttp()
        _discovery_routes(http)
        http.route("GET", "/tracks/111?client_id=" + CID_OLD,
                   FakeResponse(403, {"error": "Forbidden"}))
        http.route("GET", "/tracks/111?client_id=" + CID_NEW,
                   FakeResponse(403, {"error": "Forbidden"}))
        conn = _connector(http)
        assert conn.set_client_id(CID_OLD)
        with pytest.raises(SoundCloudError) as ei:
            conn.get_track(111)
        assert ei.value.status_code == 403
        # A 403 that survives a fresh id is the track's own
        # refusal, not a stale credential — the message says so.
        assert "already refreshed once" in str(ei.value)

    def test_env_pin_is_never_rediscovered_over(self):
        http = FakeHttp()
        _discovery_routes(http)
        http.route("GET", "/tracks/111",
                   FakeResponse(403, {"error": "Forbidden"}))
        conn = _connector(http)
        os.environ["SOUNDCLOUD_CLIENT_ID"] = CID_ENV
        try:
            with pytest.raises(SoundCloudError) as ei:
                conn.get_track(111)
        finally:
            os.environ.pop("SOUNDCLOUD_CLIENT_ID", None)
        assert ei.value.status_code == 403
        # No discovery ran: the homepage was never fetched.
        assert not any("soundcloud.com" in url
                       and "/tracks/" not in url
                       for _m, url, _h in http.calls)

    def test_malformed_client_id_is_refused(self):
        conn = _connector()
        assert conn.set_client_id("not-an-id") is False
        assert conn.set_client_id("a" * 31) is False
        assert conn.set_client_id("") is False

    def test_homepage_fetch_failure_uses_page_fetcher(self):
        html = ('<html><script src="https://a-v2.sndcdn.com/assets/'
                '0-abc-3.js"></script></html>')
        http = FakeHttp()
        # The plain HTTP homepage fetch 403s (the bot check that
        # kills discovery on some boxes).
        http.route("GET", "https://soundcloud.com",
                   FakeResponse(403, {"error": "Forbidden"}))
        http.route("GET", "https://a-v2.sndcdn.com/assets/0-abc-3.js",
                   FakeResponse(200, text=f'client_id:"{CID_NEW}"'))
        conn = _connector(http, page_fetcher=lambda: html)
        assert conn._discover_client_id() == CID_NEW

    def test_no_page_fetcher_means_honest_discovery_failure(self):
        http = FakeHttp()
        http.route("GET", "https://soundcloud.com",
                   FakeResponse(403, {"error": "Forbidden"}))
        conn = _connector(http)
        with pytest.raises(SoundCloudError) as ei:
            conn._discover_client_id()
        assert "SOUNDCLOUD_CLIENT_ID" in str(ei.value)


# ── session cookies ──────────────────────────────────────────────────


class TestSessionCookies:
    def test_oauth_token_rides_api_calls(self):
        http = FakeHttp()
        http.route("GET", "/search/tracks",
                   FakeResponse(200, {"collection": []}))
        conn = _connector(http)
        assert conn.set_session_cookies(
            [{"name": "oauth_token", "value": "tok-123",
              "domain": "soundcloud.com"}]) == 1
        os.environ["SOUNDCLOUD_CLIENT_ID"] = CID_OLD
        try:
            conn.search_tracks("x", limit=1)
        finally:
            os.environ.pop("SOUNDCLOUD_CLIENT_ID", None)
        cookies = [h.get("Cookie", "") for _m, _u, h in http.calls]
        assert any("oauth_token=tok-123" in c for c in cookies)

    def test_session_cookies_ride_discovery_fetches(self):
        http = FakeHttp()
        _discovery_routes(http)
        conn = _connector(http)
        conn.set_session_cookies(
            {"oauth_token": "tok-999", "other": "v"})
        conn._discover_client_id()
        cookies = [h.get("Cookie", "") for _m, _u, h in http.calls]
        assert any("oauth_token=tok-999" in c for c in cookies)

    def test_no_session_means_no_cookie_header(self):
        http = FakeHttp()
        http.route("GET", "/search/tracks",
                   FakeResponse(200, {"collection": []}))
        conn = _connector(http)
        os.environ["SOUNDCLOUD_CLIENT_ID"] = CID_OLD
        try:
            conn.search_tracks("x", limit=1)
        finally:
            os.environ.pop("SOUNDCLOUD_CLIENT_ID", None)
        assert all("Cookie" not in h for _m, _u, h in http.calls)


# ── preview honesty ──────────────────────────────────────────────────


class TestPreviewHonesty:
    def test_snipped_hls_raises_preview(self):
        snipped = dict(TRACK_FULL, policy="SNIP",
                       media={"transcodings": [
                           {"url": "https://api-v2.soundcloud.com"
                                   "/media/9/hls",
                            "preset": "mp3_1_0", "duration": 30000,
                            "snipped": True,
                            "format": {"protocol": "hls",
                                       "mime_type": "audio/mpeg"}},
                       ]})
        http = FakeHttp()
        http.route("GET", "/tracks/999", FakeResponse(200, snipped))
        conn = _connector(http)
        os.environ["SOUNDCLOUD_CLIENT_ID"] = CID_OLD
        try:
            with pytest.raises(SoundCloudError) as ei:
                conn.stream_url(999)
        finally:
            os.environ.pop("SOUNDCLOUD_CLIENT_ID", None)
        assert ei.value.reason == "preview"

    def test_snipped_progressive_skipped_for_full_hls(self):
        mixed = dict(TRACK_FULL, policy="ALLOW",
                     media={"transcodings": [
                         {"url": "https://api-v2.soundcloud.com"
                                 "/media/9/stream",
                          "preset": "mp3_1_0", "duration": 30000,
                          "snipped": True,
                          "format": {"protocol": "progressive",
                                     "mime_type": "audio/mpeg"}},
                         {"url": "https://api-v2.soundcloud.com"
                                 "/media/9/hls",
                          "preset": "mp3_0_1", "duration": 180000,
                          "snipped": False,
                          "format": {"protocol": "hls",
                                     "mime_type": "application/x-mpegURL"}},
                     ]})
        http = FakeHttp()
        http.route("GET", "/tracks/998", FakeResponse(200, mixed))
        http.route("GET", "/media/9/hls", FakeResponse(
            200, {"url": "https://cf-hls-media.sndcdn.com/x.m3u8"}))
        conn = _connector(http)
        os.environ["SOUNDCLOUD_CLIENT_ID"] = CID_OLD
        try:
            stream = conn.stream_url(998)
        finally:
            os.environ.pop("SOUNDCLOUD_CLIENT_ID", None)
        assert stream["protocol"] == "hls"
        assert "cf-hls-media" in stream["url"]

    def test_full_track_still_prefers_progressive(self):
        http = FakeHttp()
        http.route("GET", "/tracks/111", FakeResponse(200, TRACK_FULL))
        http.route("GET", "/media/1/stream", FakeResponse(
            200, {"url": "https://cf-media.sndcdn.com/p.mp3"}))
        conn = _connector(http)
        os.environ["SOUNDCLOUD_CLIENT_ID"] = CID_OLD
        try:
            stream = conn.stream_url(111)
        finally:
            os.environ.pop("SOUNDCLOUD_CLIENT_ID", None)
        assert stream["protocol"] == "progressive"
        assert stream["url"].endswith("p.mp3")


# ── credentials module ───────────────────────────────────────────────


class _DictStore:
    def __init__(self) -> None:
        self.data: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.data.get(key)

    def set(self, key: str, entry: Any) -> None:
        self.data[key] = entry


class TestCredentialsModule:
    def test_env_pin_wins(self, monkeypatch):
        from nomorals.media.soundcloud_credentials import (
            resolve_client_id,
        )
        store = _DictStore()
        store.set("soundcloud_client_id:discovered",
                  {"client_id": CID_NEW, "discovered_at": 0})
        fake = SimpleNamespace(client_id=lambda: CID_NEW)
        monkeypatch.setenv("SOUNDCLOUD_CLIENT_ID", CID_ENV)
        got, source = resolve_client_id(connector=fake,
                                        vault_store=store)
        assert (got, source) == (CID_ENV, "env")

    def test_vault_beats_live_discovery(self):
        from nomorals.media.soundcloud_credentials import (
            resolve_client_id,
        )
        store = _DictStore()
        store.set("soundcloud_client_id:discovered",
                  {"client_id": CID_NEW, "discovered_at": 0})
        fake = SimpleNamespace(
            client_id=lambda: (_ for _ in ()).throw(
                AssertionError("discovery must not run")))
        got, source = resolve_client_id(connector=fake,
                                        vault_store=store)
        assert (got, source) == (CID_NEW, "vault")

    def test_discovered_id_is_persisted(self):
        from nomorals.media.soundcloud_credentials import (
            VAULT_KEY,
            resolve_client_id,
        )
        store = _DictStore()
        fake = SimpleNamespace(client_id=lambda: CID_NEW)
        got, source = resolve_client_id(connector=fake,
                                        vault_store=store)
        assert (got, source) == (CID_NEW, "discovered")
        assert store.get(VAULT_KEY)["client_id"] == CID_NEW

    def test_malformed_vault_entry_reads_as_absent(self):
        from nomorals.media.soundcloud_credentials import (
            load_discovered_client_id,
        )
        store = _DictStore()
        store.set("soundcloud_client_id:discovered",
                  {"client_id": "not-an-id"})
        assert load_discovered_client_id(store) == ""

    def test_malformed_id_is_never_stored(self):
        from nomorals.media.soundcloud_credentials import (
            VAULT_KEY,
            store_discovered_client_id,
        )
        store = _DictStore()
        assert store_discovered_client_id(store, "nope") is False
        assert store.get(VAULT_KEY) is None

    def test_unavailable_without_connector_or_vault(self):
        from nomorals.media.soundcloud_credentials import (
            resolve_client_id,
        )
        got, source = resolve_client_id(connector=None,
                                        vault_store=None)
        assert (got, source) == ("", "unavailable")


# ── descriptor ───────────────────────────────────────────────────────


class TestDescriptor:
    def test_session_cookie_is_oauth_token(self):
        from nomorals.media.site_descriptors import (
            SOUNDCLOUD_DESCRIPTOR,
        )
        # oauth_token is the cookie SoundCloud's web player sets
        # and the one yt-dlp's SoundCloud extractor reads from the
        # cookie jar — auth_token is set by no real browser login,
        # so the B5 estate lookup could never find a session.
        assert SOUNDCLOUD_DESCRIPTOR.auth_cookie_names == (
            "oauth_token",)
        assert SOUNDCLOUD_DESCRIPTOR.auth_domains == (
            "soundcloud.com",)


# ── downloader wiring ────────────────────────────────────────────────


class _FakeKit:
    def __init__(self, payload: bytes = b"connector-bytes") -> None:
        self.payload = payload
        self.calls: list[str] = []

    def download(self, url, dest, progress=None):
        self.calls.append(url)
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(self.payload)
        return SimpleNamespace(
            ok=True, path=str(dest), size_bytes=len(self.payload),
            error="", resumed=False, restarted=False)


class _RecordingConnector:
    """Duck-typed SoundCloud connector: records the prep calls."""

    def __init__(self) -> None:
        self.client_ids: list[str] = []
        self.session_counts: list[int] = []
        self.fetcher_set = False
        self.stream_calls: list[str] = []

    def set_page_fetcher(self, fetcher) -> None:
        self.fetcher_set = True

    def set_client_id(self, client_id: str) -> bool:
        self.client_ids.append(client_id)
        return True

    def client_id(self) -> str:
        return CID_NEW

    def set_session_cookies(self, cookies) -> int:
        count = len(list(cookies or ()))
        self.session_counts.append(count)
        return count

    def stream_url(self, url: str) -> dict[str, Any]:
        self.stream_calls.append(url)
        return {
            "url": "https://cf-media.sndcdn.com/prog.mp3",
            "protocol": "progressive", "mime_type": "audio/mpeg",
            "track": {"title": "Midnight Drive",
                      "permalink_url":
                          "https://soundcloud.com/artist/midnight-drive"}}


def _no_descriptor_extractors():
    return SimpleNamespace(best_for=lambda url: None)


def _estate_auth(*, cookies=(), source="estate"):
    return SimpleNamespace(
        resolve=lambda desc, require=False: SimpleNamespace(
            ok=True, cookies=tuple(cookies), source=source,
            status="ok", owner_action=""))


class TestDownloaderPrep:
    def test_connector_armed_before_stream_url(self, tmp_path,
                                               monkeypatch):
        from unittest.mock import patch

        from nomorals.media.downloader import MediaDownloader

        connector = _RecordingConnector()
        monkeypatch.setenv("SOUNDCLOUD_CLIENT_ID", CID_ENV)
        auth = _estate_auth(cookies=(
            {"name": "oauth_token", "value": "tok-1",
             "domain": "soundcloud.com"},))
        dl = MediaDownloader(
            None, direct_retries=1,
            soundcloud_connector=connector,
            kit_downloader=_FakeKit(),
            extractors=_no_descriptor_extractors(),
            site_auth=auth)
        with patch("nomorals.tools.media.download",
                   side_effect=RuntimeError("connection reset")):
            report = dl.download(
                "https://soundcloud.com/artist/midnight-drive",
                dest_dir=tmp_path, audio_only=True)
        assert report.ok
        assert connector.fetcher_set is True
        assert connector.client_ids == [CID_ENV]
        assert connector.session_counts == [1]
        assert connector.stream_calls == [
            "https://soundcloud.com/artist/midnight-drive"]

    def test_prep_failure_never_breaks_the_chain(self, tmp_path,
                                                  monkeypatch):
        from unittest.mock import patch

        from nomorals.media.downloader import MediaDownloader

        class _BreakingConnector(_RecordingConnector):
            def set_client_id(self, client_id: str) -> bool:
                raise RuntimeError("vault locked")

            def set_session_cookies(self, cookies) -> int:
                raise RuntimeError("estate unreadable")

        connector = _BreakingConnector()
        monkeypatch.setenv("SOUNDCLOUD_CLIENT_ID", CID_ENV)
        dl = MediaDownloader(
            None, direct_retries=1,
            soundcloud_connector=connector,
            kit_downloader=_FakeKit(),
            extractors=_no_descriptor_extractors(),
            site_auth=_estate_auth(cookies=(
                {"name": "oauth_token", "value": "tok-1"},)))
        with patch("nomorals.tools.media.download",
                   side_effect=RuntimeError("connection reset")):
            report = dl.download(
                "https://soundcloud.com/artist/midnight-drive",
                dest_dir=tmp_path, audio_only=True)
        assert report.ok


# ── failure reporting ────────────────────────────────────────────────


class TestFailureReporting:
    def _report(self, url, stages):
        from nomorals.media.downloader import (
            DownloadReport,
            MediaDownloader,
        )
        dl = MediaDownloader.__new__(MediaDownloader)
        report = DownloadReport(url=url, stages=list(stages))
        dl._finish_failure(report)
        return report

    def test_soundcloud_bot_block_names_soundcloud_action(self):
        from nomorals.media.downloader import FailureKind, StageLog
        report = self._report(
            "https://soundcloud.com/artist/blocked-track",
            [StageLog(stage="direct",
                      url="https://soundcloud.com/a/b",
                      error_class=FailureKind.BOT_BLOCK.value,
                      error="HTTP Error 403: Forbidden")])
        assert report.failure is FailureKind.BOT_BLOCK
        assert "youtube" not in report.hint.lower()
        assert "soundcloud" in report.hint.lower()
        assert "cookies import" in report.hint

    def test_connector_preview_block_is_the_cause(self):
        from nomorals.media.downloader import FailureKind, StageLog
        report = self._report(
            "https://soundcloud.com/artist/snippet",
            [StageLog(stage="connector",
                      url="https://soundcloud.com/a/b",
                      error_class=FailureKind.SITE_ERROR.value,
                      error='"Snippet" is preview-only on SoundCloud '
                            "(~30s)",
                      note="soundcloud block: preview")])
        assert "preview" in report.likely_cause.lower()
        assert "another source" in report.hint

    def test_connector_login_block_names_the_session_action(self):
        from nomorals.media.downloader import FailureKind, StageLog
        report = self._report(
            "https://soundcloud.com/artist/gated",
            [StageLog(stage="connector",
                      url="https://soundcloud.com/a/b",
                      error_class=FailureKind.SITE_ERROR.value,
                      error="SoundCloud wants a signed-in account",
                      note="soundcloud block: login")])
        assert "signed-in account" in report.likely_cause
        assert "cookies import" in report.hint

    def test_other_sites_keep_the_youtube_help(self):
        from nomorals.media.downloader import FailureKind, StageLog
        report = self._report(
            "https://www.youtube.com/watch?v=abc",
            [StageLog(stage="direct",
                      url="https://www.youtube.com/watch?v=abc",
                      error_class=FailureKind.BOT_BLOCK.value,
                      error="Sign in to confirm you're not a bot")])
        assert "youtube" in report.hint.lower()


# ── registry ─────────────────────────────────────────────────────────


class TestToolCount:
    def test_tool_count_still_806(self):
        from types import SimpleNamespace

        from nomorals.tools.registry import ToolRegistry

        # No new brain tool: the SoundCloud download already routes
        # through the existing download chain, so the registry stays
        # at 787 after the museum ship added museum_library (the
        # lane's base pinned 785; rebumped at N3 ship time).
        registry = ToolRegistry(SimpleNamespace())
        registry.register_builtins()
        assert len(registry._tools) == 806  # D5 (walabel) adds whatsapp_account_label
