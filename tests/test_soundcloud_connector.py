"""SoundCloud connector tests. HTTP is fully mocked — no network."""

from __future__ import annotations

import json
import os
import unittest
import urllib.parse
from typing import Any
from unittest import mock

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors.base import ConnectorError
from nomorals.connectors.registry import get_connector
from nomorals.connectors.soundcloud import (
    SoundCloudConnector,
    SoundCloudError,
    _reset_client_id_cache,
)
from nomorals.storage.db import Database


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test")


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
        self.calls: list[tuple[str, str]] = []
        self.routes: list[tuple[str, str, FakeResponse]] = []

    def route(self, method: str, path: str, response: FakeResponse) -> None:
        self.routes.append((method.upper(), path, response))

    def _dispatch(self, method: str, url: str, **kw: Any) -> FakeResponse:
        self.calls.append((method.upper(), url))
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


CID = "a" * 32

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
         "format": {"protocol": "progressive",
                    "mime_type": "audio/mpeg"}},
        {"url": "https://api-v2.soundcloud.com/media/1/hls",
         "format": {"protocol": "hls",
                    "mime_type": "application/x-mpegURL"}},
    ]},
}

TRACK_HLS_ONLY = dict(TRACK_FULL, id=222, media={"transcodings": [
    {"url": "https://api-v2.soundcloud.com/media/2/hls",
     "format": {"protocol": "hls",
                "mime_type": "application/x-mpegURL"}},
]})

TRACK_UNSTREAMABLE = dict(TRACK_FULL, id=333, policy="BLOCK",
                          media={"transcodings": []})

PLAYLIST_FULL = {
    "id": 777, "kind": "playlist", "title": "Night Mix",
    "track_count": 2, "duration": 360000,
    "artwork_url": "https://i1.sndcdn.com/artworks-pl-large.jpg",
    "permalink_url": "https://soundcloud.com/artist/sets/night-mix",
    "user": {"username": "artist"},
    "tracks": [TRACK_FULL, TRACK_HLS_ONLY],
}


def _soundcloud(http: FakeHttp | None = None, env_cid: str = ""
                ) -> tuple[SoundCloudConnector, FakeHttp]:
    http = http or FakeHttp()
    conn = SoundCloudConnector(_vault(), http=http)
    _reset_client_id_cache()
    return conn, http


def _route_discovery(http: FakeHttp, cid: str = CID) -> None:
    http.route("GET", "https://soundcloud.com", FakeResponse(
        200, text='<html><script src="https://a-v2.sndcdn.com/assets/0-abc-3.js">'
                 '</script></html>'))
    http.route("GET", "https://a-v2.sndcdn.com/assets/0-abc-3.js",
               FakeResponse(200, text=f'webpack{{client_id:"{cid}"}}'))


class RegistryTests(unittest.TestCase):
    def test_registered(self) -> None:
        self.assertIs(get_connector("soundcloud"), SoundCloudConnector)

    def test_metadata(self) -> None:
        self.assertEqual(SoundCloudConnector.id, "soundcloud")
        self.assertIn("none",
                      [m.value for m in SoundCloudConnector.auth_methods])


class DiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_client_id_cache()

    def tearDown(self) -> None:
        _reset_client_id_cache()

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.pop("SOUNDCLOUD_CLIENT_ID", None)
        return env

    def test_env_pin_wins(self) -> None:
        conn, http = _soundcloud()
        with mock.patch.dict(os.environ, {"SOUNDCLOUD_CLIENT_ID": CID},
                             clear=False):
            self.assertEqual(conn._client_id(), CID)
        self.assertEqual(http.calls, [])  # no discovery traffic

    def test_discovery_scrapes_bundle(self) -> None:
        conn, http = _soundcloud()
        _route_discovery(http)
        with mock.patch.dict(os.environ, self._env(), clear=True):
            self.assertEqual(conn._client_id(), CID)
        # second call uses the cache — no more HTTP
        with mock.patch.dict(os.environ, self._env(), clear=True):
            self.assertEqual(conn._client_id(), CID)
        self.assertEqual(len(http.calls), 2)

    def test_discovery_failure_is_clear(self) -> None:
        conn, http = _soundcloud()
        http.route("GET", "https://soundcloud.com",
                   FakeResponse(500, {"error": "boom"}))
        with mock.patch.dict(os.environ, self._env(), clear=True):
            with self.assertRaises(SoundCloudError) as ctx:
                conn._client_id()
        self.assertIn("SOUNDCLOUD_CLIENT_ID", str(ctx.exception))

    def test_discovery_no_bundles_is_clear(self) -> None:
        conn, http = _soundcloud()
        http.route("GET", "https://soundcloud.com",
                   FakeResponse(200, text="<html>no scripts here</html>"))
        with mock.patch.dict(os.environ, self._env(), clear=True):
            with self.assertRaises(SoundCloudError) as ctx:
                conn._client_id()
        self.assertIn("layout changed", str(ctx.exception))


class ConnectTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_client_id_cache()

    def tearDown(self) -> None:
        _reset_client_id_cache()

    def test_connect_live(self) -> None:
        conn, http = _soundcloud()
        _route_discovery(http)
        http.route("GET", "/search/tracks",
                   FakeResponse(200, {"collection": [TRACK_FULL]}))
        env = dict(os.environ)
        env.pop("SOUNDCLOUD_CLIENT_ID", None)
        with mock.patch.dict(os.environ, env, clear=True):
            result = conn.connect()
        self.assertTrue(result.ok)
        self.assertIn("no account", result.account)

    def test_connect_discovery_failure(self) -> None:
        conn, http = _soundcloud()
        http.route("GET", "https://soundcloud.com",
                   FakeResponse(500, {}))
        env = dict(os.environ)
        env.pop("SOUNDCLOUD_CLIENT_ID", None)
        with mock.patch.dict(os.environ, env, clear=True):
            result = conn.connect()
        self.assertFalse(result.ok)
        self.assertIn("SOUNDCLOUD_CLIENT_ID", result.message)

    def test_disconnect_clears_cache(self) -> None:
        conn, http = _soundcloud()
        _route_discovery(http)
        env = dict(os.environ)
        env.pop("SOUNDCLOUD_CLIENT_ID", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(conn._client_id(), CID)
            conn.disconnect()
            # cache cleared → discovery runs again
            self.assertEqual(conn._client_id(), CID)
        self.assertEqual(len(http.calls), 4)

    def test_test_connection_true_false(self) -> None:
        conn, http = _soundcloud()
        _route_discovery(http)
        http.route("GET", "/search/tracks",
                   FakeResponse(200, {"collection": []}))
        env = dict(os.environ)
        env.pop("SOUNDCLOUD_CLIENT_ID", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(conn.test_connection())
        http.routes.clear()
        _route_discovery(http)
        http.route("GET", "/search/tracks", FakeResponse(500, {}))
        _reset_client_id_cache()
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(conn.test_connection())


class SearchTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_client_id_cache()
        self.conn, self.http = _soundcloud()
        _route_discovery(self.http)
        self._env = dict(os.environ)
        self._env.pop("SOUNDCLOUD_CLIENT_ID", None)
        self._patch = mock.patch.dict(os.environ, self._env, clear=True)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.addCleanup(_reset_client_id_cache)

    def test_search_tracks(self) -> None:
        self.http.route("GET", "/search/tracks",
                        FakeResponse(200, {"collection": [TRACK_FULL]}))
        results = self.conn.search_tracks("midnight")
        self.assertEqual(len(results), 1)
        tr = results[0]
        self.assertEqual(tr["title"], "Midnight Drive")
        self.assertEqual(tr["artist"], "artist")
        self.assertEqual(tr["duration_ms"], 180000)
        self.assertEqual(tr["permalink_url"],
                         "https://soundcloud.com/artist/midnight-drive")

    def test_search_empty_query(self) -> None:
        with self.assertRaises(SoundCloudError):
            self.conn.search_tracks("  ")

    def test_search_bad_kind(self) -> None:
        with self.assertRaises(SoundCloudError):
            self.conn.search("tape", "x")

    def test_search_playlists(self) -> None:
        self.http.route("GET", "/search/playlists",
                        FakeResponse(200, {"collection": [PLAYLIST_FULL]}))
        results = self.conn.search_playlists("night")
        self.assertEqual(results[0]["title"], "Night Mix")
        self.assertEqual(results[0]["track_count"], 2)


class ResolveTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_client_id_cache()
        self.conn, self.http = _soundcloud()
        _route_discovery(self.http)
        env = dict(os.environ)
        env.pop("SOUNDCLOUD_CLIENT_ID", None)
        self._patch = mock.patch.dict(os.environ, env, clear=True)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.addCleanup(_reset_client_id_cache)

    def test_resolve_track(self) -> None:
        self.http.route("GET", "/resolve", FakeResponse(200, TRACK_FULL))
        out = self.conn.resolve("https://soundcloud.com/artist/midnight-drive")
        self.assertEqual(out["kind"], "track")
        self.assertEqual(out["track"]["id"], 111)

    def test_resolve_playlist(self) -> None:
        self.http.route("GET", "/resolve", FakeResponse(200, PLAYLIST_FULL))
        out = self.conn.resolve("https://soundcloud.com/artist/sets/night-mix")
        self.assertEqual(out["kind"], "playlist")
        self.assertEqual(out["playlist"]["track_count"], 2)

    def test_resolve_user(self) -> None:
        self.http.route("GET", "/resolve",
                        FakeResponse(200, {"id": 9, "kind": "user",
                                           "username": "artist",
                                           "permalink_url":
                                           "https://soundcloud.com/artist",
                                           "track_count": 40,
                                           "followers_count": 100}))
        out = self.conn.resolve("https://soundcloud.com/artist")
        self.assertEqual(out["kind"], "user")
        self.assertEqual(out["user"]["username"], "artist")

    def test_resolve_not_a_url(self) -> None:
        with self.assertRaises(SoundCloudError):
            self.conn.resolve("just some words")

    def test_resolve_404_is_clear(self) -> None:
        self.http.route("GET", "/resolve", FakeResponse(404, {}))
        with self.assertRaises(SoundCloudError) as ctx:
            self.conn.resolve("https://soundcloud.com/artist/gone")
        self.assertIn("private, deleted", str(ctx.exception))


class StreamTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_client_id_cache()
        self.conn, self.http = _soundcloud()
        _route_discovery(self.http)
        env = dict(os.environ)
        env.pop("SOUNDCLOUD_CLIENT_ID", None)
        self._patch = mock.patch.dict(os.environ, env, clear=True)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.addCleanup(_reset_client_id_cache)

    def test_stream_prefers_progressive(self) -> None:
        self.http.route(
            "GET", "https://api-v2.soundcloud.com/media/1/stream",
            FakeResponse(200, {"url": "https://cf-media.sndcdn.com/prog.mp3"}))
        out = self.conn.stream_url(TRACK_FULL)
        self.assertEqual(out["protocol"], "progressive")
        self.assertEqual(out["url"], "https://cf-media.sndcdn.com/prog.mp3")
        self.assertEqual(out["track"]["title"], "Midnight Drive")

    def test_stream_falls_back_to_hls(self) -> None:
        self.http.route(
            "GET", "https://api-v2.soundcloud.com/media/2/hls",
            FakeResponse(200, {"url": "https://cf-media.sndcdn.com/h.m3u8"}))
        out = self.conn.stream_url(TRACK_HLS_ONLY)
        self.assertEqual(out["protocol"], "hls")
        self.assertTrue(out["url"].endswith(".m3u8"))

    def test_stream_by_id(self) -> None:
        self.http.route("GET", "/tracks/111", FakeResponse(200, TRACK_FULL))
        self.http.route(
            "GET", "https://api-v2.soundcloud.com/media/1/stream",
            FakeResponse(200, {"url": "https://cf-media.sndcdn.com/p.mp3"}))
        out = self.conn.stream_url(111)
        self.assertTrue(out["url"].endswith(".mp3"))

    def test_stream_by_url(self) -> None:
        self.http.route("GET", "/resolve", FakeResponse(200, TRACK_FULL))
        self.http.route("GET", "/tracks/111", FakeResponse(200, TRACK_FULL))
        self.http.route(
            "GET", "https://api-v2.soundcloud.com/media/1/stream",
            FakeResponse(200, {"url": "https://cf-media.sndcdn.com/p.mp3"}))
        out = self.conn.stream_url("https://soundcloud.com/artist/midnight-drive")
        self.assertEqual(out["track"]["id"], 111)

    def test_stream_unstreamable_fails_fast(self) -> None:
        with self.assertRaises(SoundCloudError) as ctx:
            self.conn.stream_url(TRACK_UNSTREAMABLE)
        self.assertIn("no playable stream", str(ctx.exception))

    def test_stream_garbage_target(self) -> None:
        with self.assertRaises(SoundCloudError):
            self.conn.stream_url("not a track")


class PlaylistUserTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_client_id_cache()
        self.conn, self.http = _soundcloud()
        _route_discovery(self.http)
        env = dict(os.environ)
        env.pop("SOUNDCLOUD_CLIENT_ID", None)
        self._patch = mock.patch.dict(os.environ, env, clear=True)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.addCleanup(_reset_client_id_cache)

    def test_playlist_tracks_by_url(self) -> None:
        self.http.route("GET", "/resolve", FakeResponse(200, PLAYLIST_FULL))
        tracks = self.conn.playlist_tracks(
            "https://soundcloud.com/artist/sets/night-mix")
        self.assertEqual(len(tracks), 2)
        self.assertEqual(tracks[0]["title"], "Midnight Drive")
        # playlist payloads carry transcodings → stream_url works directly
        self.assertIsNotNone(tracks[0]["_full"])

    def test_playlist_tracks_by_id(self) -> None:
        self.http.route("GET", "/playlists/777",
                        FakeResponse(200, PLAYLIST_FULL))
        tracks = self.conn.playlist_tracks(777)
        self.assertEqual(len(tracks), 2)

    def test_playlist_tracks_wrong_kind(self) -> None:
        self.http.route("GET", "/resolve", FakeResponse(200, TRACK_FULL))
        with self.assertRaises(SoundCloudError):
            self.conn.playlist_tracks("https://soundcloud.com/artist/x")

    def test_get_playlist_bad_id(self) -> None:
        with self.assertRaises(SoundCloudError):
            self.conn.get_playlist("abc")

    def test_user_tracks_by_url(self) -> None:
        self.http.route("GET", "/resolve",
                        FakeResponse(200, {"id": 9, "kind": "user",
                                           "username": "artist",
                                           "permalink_url":
                                           "https://soundcloud.com/artist",
                                           "track_count": 40,
                                           "followers_count": 5}))
        self.http.route("GET", "/users/9/tracks",
                        FakeResponse(200, {"collection": [TRACK_FULL]}))
        tracks = self.conn.user_tracks("https://soundcloud.com/artist")
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]["artist"], "artist")


class RetryTests(unittest.TestCase):
    def test_401_invalidates_cache_and_rediscovers(self) -> None:
        conn, http = _soundcloud()
        _route_discovery(http, cid="b" * 32)
        # every api-v2 call 401s once; after rediscovery it succeeds
        calls = {"n": 0}

        orig = http._dispatch

        def flaky(method: str, url: str, **kw: Any) -> FakeResponse:
            if "api-v2.soundcloud.com/search" in url and calls["n"] == 0:
                calls["n"] += 1
                return FakeResponse(401, {})
            return orig(method, url, **kw)

        http._dispatch = flaky  # type: ignore[method-assign]
        http.route("GET", "/search/tracks",
                   FakeResponse(200, {"collection": [TRACK_FULL]}))
        env = dict(os.environ)
        env.pop("SOUNDCLOUD_CLIENT_ID", None)
        with mock.patch.dict(os.environ, env, clear=True):
            results = conn.search_tracks("midnight")
        self.assertEqual(results[0]["id"], 111)
        self.assertEqual(calls["n"], 1)

    def test_401_with_pinned_id_fails_fast(self) -> None:
        conn, http = _soundcloud()
        http.route("GET", "/search/tracks", FakeResponse(401, {}))
        with mock.patch.dict(os.environ, {"SOUNDCLOUD_CLIENT_ID": CID},
                             clear=False):
            with self.assertRaises(SoundCloudError) as ctx:
                conn.search_tracks("midnight")
        self.assertIn("revoked", str(ctx.exception))

    def test_429_is_clear(self) -> None:
        conn, http = _soundcloud()
        _route_discovery(http)
        http.route("GET", "/search/tracks", FakeResponse(429, {}))
        env = dict(os.environ)
        env.pop("SOUNDCLOUD_CLIENT_ID", None)
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(SoundCloudError) as ctx:
                conn.search_tracks("midnight")
        self.assertIn("rate limit", str(ctx.exception))


class SummarizeTests(unittest.TestCase):
    def test_summarize_track_missing_fields(self) -> None:
        tr = SoundCloudConnector.summarize_track({"id": 1})
        self.assertEqual(tr["title"], "")
        self.assertEqual(tr["artist"], "")
        self.assertEqual(tr["duration_ms"], 0)

    def test_summarize_playlist(self) -> None:
        pl = SoundCloudConnector.summarize_playlist(PLAYLIST_FULL)
        self.assertEqual(pl["track_count"], 2)
        self.assertEqual(pl["artist"], "artist")

    def test_instructions_mention_keyless(self) -> None:
        conn, _http = _soundcloud()
        self.assertIn("no account", conn.connect_instructions())

    def test_status_shape(self) -> None:
        conn, http = _soundcloud()
        _route_discovery(http)
        http.route("GET", "/search/tracks",
                   FakeResponse(200, {"collection": []}))
        env = dict(os.environ)
        env.pop("SOUNDCLOUD_CLIENT_ID", None)
        with mock.patch.dict(os.environ, env, clear=True):
            status = conn.status()
        self.assertTrue(status.connected)
        self.assertIn("no account", status.account or "")


if __name__ == "__main__":
    unittest.main()
