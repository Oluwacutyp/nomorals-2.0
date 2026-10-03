"""Spotify connector tests. HTTP is fully mocked — no network."""

from __future__ import annotations

import json
import os
import unittest
import urllib.parse
from types import SimpleNamespace
from typing import Any
from unittest import mock

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors.base import ConnectorError
from nomorals.connectors.checkpoints import CheckpointState
from nomorals.connectors.registry import get_connector
from nomorals.connectors.spotify import SpotifyConnector, SpotifyError
from nomorals.storage.db import Database


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test")


class FakeResponse:
    def __init__(self, status: int = 200, payload: Any = None,
                 headers: dict[str, str] | None = None) -> None:
        self.status = status
        self._payload = payload
        self.headers = headers or {}

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    @property
    def text(self) -> str:
        return json.dumps(self._payload)

    def json(self) -> Any:
        return self._payload


class FakeHttp:
    """Scripted stand-in for HttpClient. No network."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any, Any]] = []
        self.routes: list[tuple[str, str, FakeResponse]] = []

    def route(self, method: str, path: str, response: FakeResponse) -> None:
        self.routes.append((method.upper(), path, response))

    def _dispatch(
        self, method: str, url: str, payload: Any = None, **kw: Any
    ) -> FakeResponse:
        self.calls.append((method.upper(), url, payload, kw.get("headers")))
        for rm, rp, resp in sorted(self.routes, key=lambda r: -len(r[1])):
            if rm == method.upper() and rp in url:
                return resp
        return FakeResponse(404, {"error": {"message": "not mocked",
                                           "reason": "NOT_MOCKED"}})

    def get(self, url: str, **kw: Any) -> FakeResponse:
        params = kw.pop("params", None) or {}
        if params:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{urllib.parse.urlencode(params)}"
        return self._dispatch("GET", url, None, **kw)

    def post_json(self, url: str, payload: Any, **kw: Any) -> FakeResponse:
        return self._dispatch("POST", url, payload, **kw)

    def post_form(self, url: str, form: Any, **kw: Any) -> FakeResponse:
        return self._dispatch("POST", url, form, **kw)

    def put_json(self, url: str, payload: Any, **kw: Any) -> FakeResponse:
        return self._dispatch("PUT", url, payload, **kw)

    def request(self, method: str, url: str, **kw: Any) -> FakeResponse:
        return self._dispatch(method, url, kw.get("data"), **kw)


ME = {"id": "user1", "display_name": "Devon",
      "email": "devon@example.com"}

TOKENS = {
    "access_token": "BQD.a",
    "refresh_token": "AQDrefresh",
    "expires_in": 3600,
    "token_type": "Bearer",
}

PLAYLISTS = {
    "items": [{
        "id": "pl1", "name": "Chill", "description": "d", "public": False,
        "tracks": {"total": 12}, "uri": "spotify:playlist:pl1",
        "external_urls": {"spotify": "https://open.spotify.com/pl1"},
    }],
    "total": 1,
}

DEVICES = {"devices": [{
    "id": "dev1", "name": "Phone", "type": "Smartphone",
    "is_active": True, "is_restricted": False,
}]}

NOW_PLAYING = {
    "is_playing": True,
    "progress_ms": 42000,
    "device": {"name": "Phone"},
    "item": {"name": "Song", "uri": "spotify:track:t1",
             "artists": [{"name": "Artist"}],
             "album": {"name": "Album"}},
}


def _spotify(http: FakeHttp | None = None) -> tuple[SpotifyConnector, FakeHttp]:
    http = http or FakeHttp()
    return SpotifyConnector(_vault(), http=http), http


def _connected(http: FakeHttp | None = None) -> tuple[SpotifyConnector, FakeHttp]:
    conn, http = _spotify(http)
    http.route("POST", "accounts.spotify.com/api/token",
               FakeResponse(200, TOKENS))
    http.route("GET", "/v1/me", FakeResponse(200, ME))
    with mock.patch.dict(os.environ, {"SPOTIFY_CLIENT_ID": "cid",
                                      "SPOTIFY_CLIENT_SECRET": "csecret"}):
        result = conn.connect(
            redirect_url="http://127.0.0.1:8888/callback?code=authcode")
    assert result.ok
    return conn, http


def _checkpoint(stage: str, state: Any = CheckpointState.RESOLVED,
                note: str = "", resume_state: dict | None = None):
    return SimpleNamespace(
        id=9, state=state, result_note=note,
        resume_state=resume_state or {},
    )


class RegistryTests(unittest.TestCase):
    def test_registered(self) -> None:
        self.assertIs(get_connector("spotify"), SpotifyConnector)

    def test_metadata(self) -> None:
        self.assertEqual(SpotifyConnector.id, "spotify")
        self.assertIn("oauth2", [m.value for m in SpotifyConnector.auth_methods])


class ConnectTests(unittest.TestCase):
    def test_connect_redirect_url(self) -> None:
        conn, http = _spotify()
        http.route("POST", "accounts.spotify.com/api/token",
                   FakeResponse(200, TOKENS))
        http.route("GET", "/v1/me", FakeResponse(200, ME))
        with mock.patch.dict(os.environ, {"SPOTIFY_CLIENT_ID": "cid",
                                          "SPOTIFY_CLIENT_SECRET": "csecret"}):
            result = conn.connect(
                redirect_url="http://127.0.0.1:8888/callback?code=authcode")
        self.assertTrue(result.ok)
        self.assertEqual(result.account, "Devon")
        cred = conn.vault.get("connector:spotify", "user1")
        self.assertEqual(cred.password, "AQDrefresh")  # refresh vaulted
        self.assertEqual((cred.metadata or {}).get("email"),
                         "devon@example.com")
        # client credentials travel as HTTP Basic on the token call
        _m, _u, _p, headers = http.calls[0]
        self.assertTrue(headers["Authorization"].startswith("Basic "))

    def test_connect_bare_code(self) -> None:
        conn, http = _spotify()
        http.route("POST", "accounts.spotify.com/api/token",
                   FakeResponse(200, TOKENS))
        http.route("GET", "/v1/me", FakeResponse(200, ME))
        with mock.patch.dict(os.environ, {"SPOTIFY_CLIENT_ID": "cid",
                                          "SPOTIFY_CLIENT_SECRET": "csecret"}):
            result = conn.connect(code="authcode")
        self.assertTrue(result.ok)

    def test_connect_bad_code(self) -> None:
        conn, http = _spotify()
        http.route("POST", "accounts.spotify.com/api/token",
                   FakeResponse(400, {"error": "invalid_grant",
                                      "error_description": "bad code"}))
        with mock.patch.dict(os.environ, {"SPOTIFY_CLIENT_ID": "cid",
                                          "SPOTIFY_CLIENT_SECRET": "csecret"}):
            with self.assertRaises(SpotifyError) as ctx:
                conn.connect(code="stale")
            self.assertIn("rejected the token request", str(ctx.exception))
        self.assertIsNone(conn._load_credential())

    def test_connect_without_code_prints_guide(self) -> None:
        conn, _http = _spotify()
        with mock.patch.dict(os.environ, {"SPOTIFY_CLIENT_ID": "cid"}):
            result = conn.connect()
        self.assertFalse(result.ok)
        self.assertIn("redirect_url", result.message)

    def test_connect_no_client_id(self) -> None:
        conn, _http = _spotify()
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ConnectorError) as ctx:
                conn.connect(code="x")
            self.assertIn("no Spotify client id", str(ctx.exception))

    def test_connect_rejects_second_account(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.connect(code="other")
        self.assertIn("already connected", str(ctx.exception))

    def test_disconnect_clears(self) -> None:
        conn, _http = _connected()
        conn.disconnect()
        self.assertIsNone(conn._load_credential())
        conn.disconnect()  # idempotent


class StatusTests(unittest.TestCase):
    def test_status_not_connected(self) -> None:
        conn, _http = _spotify()
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertFalse(conn.test_connection())

    def test_status_connected(self) -> None:
        conn, http = _connected()
        http.route("GET", "/v1/me", FakeResponse(200, ME))
        st = conn.status()
        self.assertTrue(st.connected)
        self.assertEqual(st.account, "Devon")
        self.assertTrue(conn.test_connection())

    def test_status_revoked_grant(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/v1/me", FakeResponse(401, {"error": {}}))
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertIn("rejected", st.detail)
        self.assertFalse(conn.test_connection())


class TokenRefreshTests(unittest.TestCase):
    def test_expired_access_token_refreshes_and_rotates(self) -> None:
        conn, http = _connected()
        cred = conn._load_credential()
        assert cred is not None
        meta = dict(cred.metadata or {})
        meta["access_expires_at"] = 1.0  # force expiry
        conn.vault.store(
            service="connector:spotify", username=cred.username,
            password=cred.password, credential_type="oauth_token",
            tags=["connector", "spotify"], metadata=meta,
        )
        http.routes.clear()
        http.route("POST", "accounts.spotify.com/api/token",
                   FakeResponse(200, {
                       "access_token": "BQD.new",
                       "refresh_token": "AQDrotated",
                       "expires_in": 3600,
                   }))
        http.route("GET", "/v1/me", FakeResponse(200, ME))
        with mock.patch.dict(os.environ,
                             {"SPOTIFY_CLIENT_SECRET": "csecret"}):
            me = conn.get_me()
        self.assertEqual(me["id"], "user1")
        cred2 = conn._load_credential()
        assert cred2 is not None
        # newest refresh token stored (rotation honored)
        self.assertEqual(cred2.password, "AQDrotated")
        self.assertEqual((cred2.metadata or {}).get("access_token"),
                         "BQD.new")

    def test_refresh_failure_fails_fast(self) -> None:
        conn, http = _connected()
        cred = conn._load_credential()
        assert cred is not None
        meta = dict(cred.metadata or {})
        meta["access_expires_at"] = 1.0
        conn.vault.store(
            service="connector:spotify", username=cred.username,
            password=cred.password, credential_type="oauth_token",
            tags=["connector", "spotify"], metadata=meta,
        )
        http.routes.clear()
        http.route("POST", "accounts.spotify.com/api/token",
                   FakeResponse(400, {"error": "invalid_grant"}))
        with mock.patch.dict(os.environ,
                             {"SPOTIFY_CLIENT_SECRET": "csecret"}):
            with self.assertRaises(SpotifyError):
                conn.get_me()


class LibraryTests(unittest.TestCase):
    def test_get_me(self) -> None:
        conn, http = _connected()
        http.route("GET", "/v1/me", FakeResponse(200, ME))
        me = conn.get_me()
        self.assertEqual(me["display_name"], "Devon")

    def test_list_playlists(self) -> None:
        conn, http = _connected()
        http.route("GET", "/v1/me/playlists",
                   FakeResponse(200, PLAYLISTS))
        result = conn.list_playlists(limit=10)
        self.assertEqual(result["total"], 1)
        pl = result["playlists"][0]
        self.assertEqual(pl["id"], "pl1")
        self.assertEqual(pl["name"], "Chill")
        self.assertEqual(pl["tracks_total"], 12)

    def test_search(self) -> None:
        conn, http = _connected()
        http.route("GET", "/v1/search",
                   FakeResponse(200, {"tracks": {"items": []}}))
        result = conn.search("hello", types=["track", "artist"])
        self.assertIn("tracks", result)
        _m, url, _p, _h = http.calls[-1]
        self.assertIn("type=track%2Cartist", url)

    def test_search_empty_query(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.search("")

    def test_search_bad_types(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.search("x", types=["bogus"])

    def test_rate_limit(self) -> None:
        conn, http = _connected()
        http.route("GET", "/v1/me/playlists",
                   FakeResponse(429, {"error": {}},
                                headers={"retry-after": "3"}))
        with self.assertRaises(SpotifyError) as ctx:
            conn.list_playlists()
        self.assertEqual(ctx.exception.status_code, 429)

    def test_not_connected_raises(self) -> None:
        conn, _http = _spotify()
        with self.assertRaises(ConnectorError) as ctx:
            conn.get_me()
        self.assertIn("not connected", str(ctx.exception))


class PlaylistWriteTests(unittest.TestCase):
    def test_create_playlist_confirmed(self) -> None:
        conn, http = _connected()
        http.route("GET", "/v1/me", FakeResponse(200, ME))
        http.route("POST", "/v1/users/user1/playlists",
                   FakeResponse(201, dict(PLAYLISTS["items"][0],
                                          name="New Mix")))
        result = conn.create_playlist("New Mix", description="d",
                                      confirmed=True)
        self.assertEqual(result["name"], "New Mix")
        self.assertEqual(result["id"], "pl1")
        _m, url, payload, _h = http.calls[-1]
        self.assertIn("/v1/users/user1/playlists", url)
        self.assertFalse(payload["public"])

    def test_create_playlist_needs_confirmation(self) -> None:
        conn, http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.create_playlist("New Mix")
        self.assertIn("confirmation", str(ctx.exception))
        self.assertEqual(
            [c for c in http.calls if "playlists" in c[1] and c[0] == "POST"],
            [])

    def test_create_playlist_empty_name(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.create_playlist("  ", confirmed=True)

    def test_add_tracks_confirmed(self) -> None:
        conn, http = _connected()
        http.route("POST", "/v1/playlists/pl1/tracks",
                   FakeResponse(201, {"snapshot_id": "snap1"}))
        result = conn.add_tracks("pl1", ["spotify:track:t1",
                                         "spotify:track:t2"],
                                 confirmed=True)
        self.assertEqual(result["snapshot_id"], "snap1")
        self.assertEqual(result["added"], 2)
        _m, url, payload, _h = http.calls[-1]
        self.assertIn("/v1/playlists/pl1/tracks", url)
        self.assertEqual(payload["uris"],
                         ["spotify:track:t1", "spotify:track:t2"])

    def test_add_tracks_needs_confirmation(self) -> None:
        conn, http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.add_tracks("pl1", ["spotify:track:t1"])
        self.assertIn("confirmation", str(ctx.exception))

    def test_add_tracks_bad_uris(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.add_tracks("pl1", ["not-a-uri"], confirmed=True)
        self.assertIn("invalid track URIs", str(ctx.exception))

    def test_add_tracks_empty(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.add_tracks("pl1", [], confirmed=True)

    def test_add_tracks_negative_position(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.add_tracks("pl1", ["spotify:track:t1"], position=-1,
                            confirmed=True)


class PlaybackTests(unittest.TestCase):
    def test_list_devices(self) -> None:
        conn, http = _connected()
        http.route("GET", "/v1/me/player/devices",
                   FakeResponse(200, DEVICES))
        devices = conn.list_devices()
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0]["name"], "Phone")
        self.assertTrue(devices[0]["is_active"])

    def test_now_playing(self) -> None:
        conn, http = _connected()
        http.route("GET", "/v1/me/player/currently-playing",
                   FakeResponse(200, NOW_PLAYING))
        np = conn.now_playing()
        self.assertTrue(np["playing"])
        self.assertEqual(np["track"], "Song")
        self.assertEqual(np["artists"], ["Artist"])
        self.assertEqual(np["device"], "Phone")

    def test_now_playing_nothing(self) -> None:
        conn, http = _connected()
        http.route("GET", "/v1/me/player/currently-playing",
                   FakeResponse(204, None))
        self.assertEqual(conn.now_playing(), {"playing": False})

    def test_play_active_device(self) -> None:
        conn, http = _connected()
        http.route("GET", "/v1/me/player/devices",
                   FakeResponse(200, DEVICES))
        http.route("PUT", "/v1/me/player/play", FakeResponse(204, None))
        result = conn.play(context_uri="spotify:playlist:pl1")
        self.assertTrue(result["playing"])
        self.assertEqual(result["device_id"], "dev1")
        _m, _u, payload, _h = http.calls[-1]
        self.assertEqual(payload["context_uri"], "spotify:playlist:pl1")

    def test_play_no_active_device_fails_fast(self) -> None:
        conn, http = _connected()
        http.route("GET", "/v1/me/player/devices",
                   FakeResponse(200, {"devices": []}))
        with self.assertRaises(SpotifyError) as ctx:
            conn.play()
        self.assertIn("no active Spotify device", str(ctx.exception))
        self.assertEqual(
            [c for c in http.calls if "me/player/play" in c[1]], [])

    def test_play_no_device_404(self) -> None:
        conn, http = _connected()
        http.route("GET", "/v1/me/player/devices",
                   FakeResponse(200, DEVICES))
        http.route("PUT", "/v1/me/player/play",
                   FakeResponse(404, {"error": {
                       "message": "Player command failed: No active device",
                       "reason": "NO_ACTIVE_DEVICE"}}))
        with self.assertRaises(SpotifyError) as ctx:
            conn.play()
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertIn("NO_ACTIVE_DEVICE", ctx.exception.reason)

    def test_pause(self) -> None:
        conn, http = _connected()
        http.route("PUT", "/v1/me/player/pause", FakeResponse(204, None))
        result = conn.pause()
        self.assertFalse(result["playing"])

    def test_pause_nothing_playing_fails_fast(self) -> None:
        conn, http = _connected()
        http.route("PUT", "/v1/me/player/pause",
                   FakeResponse(404, {"error": {"reason": "NO_ACTIVE_DEVICE"}}))
        with self.assertRaises(SpotifyError) as ctx:
            conn.pause()
        self.assertIn("nothing is playing", str(ctx.exception))


class ResumeCheckpointTests(unittest.TestCase):
    def _route_oauth(self, http: FakeHttp) -> None:
        http.route("POST", "accounts.spotify.com/api/token",
                   FakeResponse(200, TOKENS))
        http.route("GET", "/v1/me", FakeResponse(200, ME))

    def test_resume_oauth_code(self) -> None:
        conn, http = _spotify()
        self._route_oauth(http)
        cp = _checkpoint(
            "oauth_code",
            note="url=http://127.0.0.1:8888/callback?code=abc",
            resume_state={"stage": "oauth_code", "client_id": "cid",
                          "redirect_uri": "http://127.0.0.1:8888/callback",
                          "scopes": ["s1"]},
        )
        with mock.patch.dict(os.environ,
                             {"SPOTIFY_CLIENT_SECRET": "csecret"}):
            result = conn.resume_checkpoint(cp, db=object())
        self.assertTrue(result["connected"])
        self.assertEqual(result["account"], "Devon")

    def test_resume_oauth_no_code(self) -> None:
        conn, _http = _spotify()
        cp = _checkpoint("oauth_code", note="nothing",
                         resume_state={"stage": "oauth_code"})
        with self.assertRaises(ConnectorError):
            conn.resume_checkpoint(cp, db=object())

    def test_resume_create_playlist(self) -> None:
        conn, http = _connected()
        http.route("GET", "/v1/me", FakeResponse(200, ME))
        http.route("POST", "/v1/users/user1/playlists",
                   FakeResponse(201, dict(PLAYLISTS["items"][0],
                                          name="Resumed")))
        cp = _checkpoint(
            "create_playlist",
            resume_state={"stage": "create_playlist", "payload": {
                "name": "Resumed", "description": "", "public": False,
            }},
        )
        result = conn.resume_checkpoint(cp, db=object())
        self.assertEqual(result["name"], "Resumed")

    def test_resume_create_playlist_no_payload(self) -> None:
        conn, _http = _connected()
        cp = _checkpoint("create_playlist",
                         resume_state={"stage": "create_playlist"})
        with self.assertRaises(ConnectorError):
            conn.resume_checkpoint(cp, db=object())

    def test_resume_add_tracks(self) -> None:
        conn, http = _connected()
        http.route("POST", "/v1/playlists/pl1/tracks",
                   FakeResponse(201, {"snapshot_id": "s9"}))
        cp = _checkpoint(
            "add_tracks",
            resume_state={"stage": "add_tracks", "payload": {
                "playlist_id": "pl1", "uris": ["spotify:track:t1"],
            }},
        )
        result = conn.resume_checkpoint(cp, db=object())
        self.assertEqual(result["snapshot_id"], "s9")

    def test_resume_add_tracks_no_payload(self) -> None:
        conn, _http = _connected()
        cp = _checkpoint("add_tracks",
                         resume_state={"stage": "add_tracks"})
        with self.assertRaises(ConnectorError):
            conn.resume_checkpoint(cp, db=object())

    def test_resume_unresolved_checkpoint(self) -> None:
        conn, _http = _spotify()
        cp = _checkpoint("oauth_code", state=CheckpointState.PENDING)
        with self.assertRaises(ConnectorError):
            conn.resume_checkpoint(cp, db=object())

    def test_resume_unknown_stage(self) -> None:
        conn, _http = _spotify()
        cp = _checkpoint("bogus", resume_state={"stage": "bogus"})
        with self.assertRaises(ConnectorError):
            conn.resume_checkpoint(cp, db=object())


class TrackLookupTests(unittest.TestCase):
    TRACK = {
        "id": "t1", "uri": "spotify:track:t1", "name": "Song",
        "artists": [{"name": "Artist"}],
        "album": {"name": "Album"}, "duration_ms": 210000,
        "external_urls": {"spotify": "https://open.spotify.com/track/t1"},
        "explicit": False,
    }

    def test_normalize_uri_passthrough(self) -> None:
        self.assertEqual(
            SpotifyConnector.normalize_uri("spotify:track:abc123"),
            "spotify:track:abc123")
        self.assertEqual(
            SpotifyConnector.normalize_uri("spotify:playlist:pl1"),
            "spotify:playlist:pl1")

    def test_normalize_open_spotify_link(self) -> None:
        self.assertEqual(
            SpotifyConnector.normalize_uri(
                "https://open.spotify.com/track/abc123?si=xyz"),
            "spotify:track:abc123")
        self.assertEqual(
            SpotifyConnector.normalize_uri(
                "https://open.spotify.com/intl-de/album/def456"),
            "spotify:album:def456")

    def test_normalize_rejects_garbage(self) -> None:
        for bad in ("", "not a uri", "spotify:track:",
                    "https://example.com/track/abc",
                    "spotify:bogus:abc"):
            with self.assertRaises(SpotifyError, msg=bad):
                SpotifyConnector.normalize_uri(bad)

    def test_get_track(self) -> None:
        conn, http = _connected()
        http.route("GET", "/v1/tracks/t1",
                   FakeResponse(200, self.TRACK))
        info = conn.get_track("spotify:track:t1")
        self.assertEqual(info["name"], "Song")
        self.assertEqual(info["artists"], ["Artist"])
        self.assertEqual(info["album"], "Album")
        self.assertEqual(info["uri"], "spotify:track:t1")

    def test_get_track_accepts_link(self) -> None:
        conn, http = _connected()
        http.route("GET", "/v1/tracks/t1",
                   FakeResponse(200, self.TRACK))
        info = conn.get_track("https://open.spotify.com/track/t1")
        self.assertEqual(info["id"], "t1")

    def test_get_track_rejects_non_track(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(SpotifyError):
            conn.get_track("spotify:album:abc")

    def test_get_track_not_found(self) -> None:
        conn, http = _connected()
        http.route("GET", "/v1/tracks/nope",
                   FakeResponse(404, {"error": {"message": "not found",
                                               "reason": "NOT_FOUND"}}))
        with self.assertRaises(SpotifyError) as ctx:
            conn.get_track("spotify:track:nope")
        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
