"""YouTube connector tests. HTTP is fully mocked — no network."""

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
from nomorals.connectors.youtube import YouTubeConnector, YouTubeError
from nomorals.storage.db import Database


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test")


class FakeResponse:
    def __init__(self, status: int = 200, payload: Any = None) -> None:
        self.status = status
        self._payload = payload
        self.headers: dict[str, str] = {}

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
        return FakeResponse(404, {"error": {"errors": [{"reason": "notFound"}]}})

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

    def request(self, method: str, url: str, **kw: Any) -> FakeResponse:
        return self._dispatch(method, url, kw.get("data"), **kw)


CHANNEL = {
    "items": [{
        "id": "UC123",
        "snippet": {"title": "Devon Channel"},
        "contentDetails": {"relatedPlaylists": {"uploads": "UU123"}},
    }],
}

PLAYLIST_ITEMS = {
    "items": [{
        "snippet": {
            "title": "Hello world",
            "description": "first video",
            "publishedAt": "2026-01-01T00:00:00Z",
            "thumbnails": {"default": {"url": "https://i.yt/img.jpg"}},
        },
        "contentDetails": {"videoId": "vid1"},
    }],
    "nextPageToken": "",
    "pageInfo": {"totalResults": 1},
}

TOKENS = {
    "access_token": "ya29.a",
    "refresh_token": "1//refresh",
    "expires_in": 3600,
    "token_type": "Bearer",
}


def _youtube(http: FakeHttp | None = None) -> tuple[YouTubeConnector, FakeHttp]:
    http = http or FakeHttp()
    return YouTubeConnector(_vault(), http=http), http


def _connected(http: FakeHttp | None = None) -> tuple[YouTubeConnector, FakeHttp]:
    conn, http = _youtube(http)
    http.route("POST", "oauth2.googleapis.com/token",
               FakeResponse(200, TOKENS))
    http.route("GET", "/youtube/v3/channels", FakeResponse(200, CHANNEL))
    with mock.patch.dict(os.environ, {"GOOGLE_CLIENT_ID": "cid",
                                      "GOOGLE_CLIENT_SECRET": "csecret"}):
        result = conn.connect(code="authcode")
    assert result.ok
    return conn, http


def _checkpoint(stage: str, state: Any = CheckpointState.RESOLVED,
                note: str = "", resume_state: dict | None = None):
    return SimpleNamespace(
        id=7, state=state, result_note=note,
        resume_state=resume_state or {},
    )


class RegistryTests(unittest.TestCase):
    def test_registered(self) -> None:
        self.assertIs(get_connector("youtube"), YouTubeConnector)

    def test_metadata(self) -> None:
        self.assertEqual(YouTubeConnector.id, "youtube")
        self.assertIn("oauth2", [m.value for m in YouTubeConnector.auth_methods])


class ConnectTests(unittest.TestCase):
    def test_connect_code_exchanges_and_stores(self) -> None:
        conn, http = _youtube()
        http.route("POST", "oauth2.googleapis.com/token",
                   FakeResponse(200, TOKENS))
        http.route("GET", "/youtube/v3/channels",
                   FakeResponse(200, CHANNEL))
        with mock.patch.dict(os.environ, {"GOOGLE_CLIENT_ID": "cid",
                                          "GOOGLE_CLIENT_SECRET": "csecret"}):
            result = conn.connect(code="authcode")
        self.assertTrue(result.ok)
        self.assertEqual(result.account, "Devon Channel")
        self.assertIn("youtube.upload", " ".join(result.scopes))
        cred = conn.vault.get("connector:youtube", "Devon Channel")
        self.assertEqual(cred.password, "1//refresh")
        # token exchange went to Google's token endpoint
        form_calls = [c for c in http.calls if "oauth2.googleapis" in c[1]]
        self.assertEqual(form_calls[0][2]["grant_type"], "authorization_code")

    def test_connect_bad_code(self) -> None:
        conn, http = _youtube()
        http.route("POST", "oauth2.googleapis.com/token",
                   FakeResponse(400, {"error": "invalid_grant"}))
        with mock.patch.dict(os.environ, {"GOOGLE_CLIENT_ID": "cid",
                                          "GOOGLE_CLIENT_SECRET": "csecret"}):
            with self.assertRaises(ConnectorError):
                conn.connect(code="stale")
        self.assertIsNone(conn._load_credential())

    def test_connect_no_channel(self) -> None:
        conn, http = _youtube()
        http.route("POST", "oauth2.googleapis.com/token",
                   FakeResponse(200, TOKENS))
        http.route("GET", "/youtube/v3/channels",
                   FakeResponse(200, {"items": []}))
        with mock.patch.dict(os.environ, {"GOOGLE_CLIENT_ID": "cid",
                                          "GOOGLE_CLIENT_SECRET": "csecret"}):
            with self.assertRaises(YouTubeError) as ctx:
                conn.connect(code="authcode")
            self.assertIn("no YouTube channel", str(ctx.exception))

    def test_connect_without_code_prints_guide(self) -> None:
        conn, _http = _youtube()
        with mock.patch.dict(os.environ, {"GOOGLE_CLIENT_ID": "cid"}):
            result = conn.connect()
        self.assertFalse(result.ok)
        self.assertIn("code=<code>", result.message)

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
        conn, _http = _youtube()
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertFalse(conn.test_connection())

    def test_status_connected(self) -> None:
        conn, http = _connected()
        http.route("GET", "/youtube/v3/channels",
                   FakeResponse(200, CHANNEL))
        st = conn.status()
        self.assertTrue(st.connected)
        self.assertEqual(st.account, "Devon Channel")
        self.assertTrue(conn.test_connection())

    def test_status_rejected_token(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/youtube/v3/channels",
                   FakeResponse(401, {"error": {"errors": []}}))
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertIn("rejected", st.detail)
        self.assertFalse(conn.test_connection())


class ApiTests(unittest.TestCase):
    def test_get_channel(self) -> None:
        conn, http = _connected()
        http.route("GET", "/youtube/v3/channels",
                   FakeResponse(200, CHANNEL))
        channel = conn.get_channel()
        self.assertEqual(channel["id"], "UC123")

    def test_list_videos(self) -> None:
        conn, http = _connected()
        http.route("GET", "/youtube/v3/channels",
                   FakeResponse(200, CHANNEL))
        http.route("GET", "/youtube/v3/playlistItems",
                   FakeResponse(200, PLAYLIST_ITEMS))
        result = conn.list_videos(max_results=10)
        self.assertEqual(len(result["videos"]), 1)
        video = result["videos"][0]
        self.assertEqual(video["video_id"], "vid1")
        self.assertEqual(video["title"], "Hello world")
        self.assertEqual(result["total"], 1)
        # uploads playlist id comes from the channel
        _m, url, _p, _h = http.calls[-1]
        self.assertIn("playlistId=UU123", url)

    def test_list_videos_no_uploads_playlist(self) -> None:
        conn, http = _connected()
        no_uploads = {"items": [{
            "id": "UC123", "snippet": {"title": "X"},
            "contentDetails": {"relatedPlaylists": {}},
        }]}
        http.routes.clear()
        http.route("GET", "/youtube/v3/channels",
                   FakeResponse(200, no_uploads))
        with self.assertRaises(YouTubeError) as ctx:
            conn.list_videos()
        self.assertIn("uploads playlist", str(ctx.exception))

    def test_search_videos(self) -> None:
        conn, http = _connected()
        http.route("GET", "/youtube/v3/search",
                   FakeResponse(200, {"items": [{
                       "id": {"videoId": "s1"},
                       "snippet": {"title": "Found",
                                   "channelTitle": "Ch",
                                   "publishedAt": "2026-01-02"},
                   }]}))
        results = conn.search_videos("hello", order="date")
        self.assertEqual(results[0]["video_id"], "s1")
        self.assertEqual(results[0]["channel"], "Ch")

    def test_search_empty_query(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.search_videos("")

    def test_search_bad_order(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.search_videos("x", order="bogus")

    def test_quota_403(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/youtube/v3/playlistItems",
                   FakeResponse(403, {"error": {"errors": [
                       {"reason": "quotaExceeded"}]}}))
        http.route("GET", "/youtube/v3/channels",
                   FakeResponse(200, CHANNEL))
        with self.assertRaises(YouTubeError) as ctx:
            conn.list_videos()
        self.assertIn("quota", str(ctx.exception))

    def test_invalid_json(self) -> None:
        conn, http = _connected()
        resp = FakeResponse(200, None)
        resp.json = mock.Mock(side_effect=ValueError("no json"))  # type: ignore[method-assign]
        http.routes.clear()
        http.route("GET", "/youtube/v3/channels", resp)
        with self.assertRaises(YouTubeError) as ctx:
            conn.get_channel()
        self.assertIn("invalid JSON", str(ctx.exception))

    def test_not_connected_raises(self) -> None:
        conn, _http = _youtube()
        with self.assertRaises(ConnectorError) as ctx:
            conn.get_channel()
        self.assertIn("not connected", str(ctx.exception))


class UploadTests(unittest.TestCase):
    def test_upload_confirmed(self) -> None:
        import tempfile
        conn, http = _connected()
        http.route("POST", "/upload/youtube/v3/videos",
                   FakeResponse(200, {
                       "id": "newvid",
                       "snippet": {"title": "My video"},
                       "status": {"privacyStatus": "unlisted"},
                   }))
        with tempfile.TemporaryDirectory() as tmp:
            src = f"{tmp}/v.mp4"
            with open(src, "wb") as fh:
                fh.write(b"\x00\x01video-bytes")
            result = conn.upload_video(src, "My video",
                                       description="desc",
                                       privacy="unlisted",
                                       confirmed=True)
        self.assertEqual(result["video_id"], "newvid")
        self.assertEqual(result["privacy"], "unlisted")
        _m, url, data, headers = http.calls[-1]
        self.assertIn("uploadType=multipart", url)
        self.assertIn("multipart/related", headers["Content-Type"])
        self.assertIn(b"My video", data)
        self.assertIn(b"\x00\x01video-bytes", data)

    def test_upload_needs_confirmation(self) -> None:
        import tempfile
        conn, http = _connected()
        with tempfile.TemporaryDirectory() as tmp:
            src = f"{tmp}/v.mp4"
            with open(src, "wb") as fh:
                fh.write(b"x")
            with self.assertRaises(ConnectorError) as ctx:
                conn.upload_video(src, "My video")
        self.assertIn("confirmation", str(ctx.exception))
        self.assertEqual(
            [c for c in http.calls if "upload/youtube" in c[1]], [])

    def test_upload_missing_file(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.upload_video("/no/such.mp4", "t", confirmed=True)

    def test_upload_no_title(self) -> None:
        import tempfile
        conn, _http = _connected()
        with tempfile.TemporaryDirectory() as tmp:
            src = f"{tmp}/v.mp4"
            with open(src, "wb") as fh:
                fh.write(b"x")
            with self.assertRaises(ConnectorError):
                conn.upload_video(src, "  ", confirmed=True)

    def test_upload_bad_privacy(self) -> None:
        import tempfile
        conn, _http = _connected()
        with tempfile.TemporaryDirectory() as tmp:
            src = f"{tmp}/v.mp4"
            with open(src, "wb") as fh:
                fh.write(b"x")
            with self.assertRaises(ConnectorError):
                conn.upload_video(src, "t", privacy="friends",
                                  confirmed=True)

    def test_upload_too_big(self) -> None:
        import tempfile
        conn, _http = _connected()
        with tempfile.TemporaryDirectory() as tmp:
            src = f"{tmp}/v.mp4"
            with open(src, "wb") as fh:
                fh.write(b"x" * 100)
            with mock.patch("nomorals.connectors.youtube._MULTIPART_MAX_BYTES",
                            10):
                with self.assertRaises(ConnectorError) as ctx:
                    conn.upload_video(src, "t", confirmed=True)
                self.assertIn("resumable", str(ctx.exception))

    def test_upload_api_error(self) -> None:
        import tempfile
        conn, http = _connected()
        http.route("POST", "/upload/youtube/v3/videos",
                   FakeResponse(400, {"error": {"errors": [
                       {"reason": "invalidCategory"}]}}))
        with tempfile.TemporaryDirectory() as tmp:
            src = f"{tmp}/v.mp4"
            with open(src, "wb") as fh:
                fh.write(b"x")
            with self.assertRaises(YouTubeError) as ctx:
                conn.upload_video(src, "t", confirmed=True)
            self.assertEqual(ctx.exception.status_code, 400)


class ResumeCheckpointTests(unittest.TestCase):
    def test_resume_oauth_code(self) -> None:
        conn, http = _youtube()
        http.route("POST", "oauth2.googleapis.com/token",
                   FakeResponse(200, TOKENS))
        http.route("GET", "/youtube/v3/channels",
                   FakeResponse(200, CHANNEL))
        cp = _checkpoint(
            "oauth_code", note="code=abc123",
            resume_state={"stage": "oauth_code", "client_id": "cid",
                          "scopes": ["s1"]},
        )
        with mock.patch.dict(os.environ,
                             {"GOOGLE_CLIENT_SECRET": "csecret"}):
            result = conn.resume_checkpoint(cp, db=object())
        self.assertTrue(result["connected"])
        self.assertEqual(result["account"], "Devon Channel")

    def test_resume_oauth_no_code(self) -> None:
        conn, _http = _youtube()
        cp = _checkpoint("oauth_code", note="nothing here",
                         resume_state={"stage": "oauth_code"})
        with self.assertRaises(ConnectorError):
            conn.resume_checkpoint(cp, db=object())

    def test_resume_upload_video(self) -> None:
        import tempfile
        conn, http = _connected()
        http.route("POST", "/upload/youtube/v3/videos",
                   FakeResponse(200, {"id": "v2", "snippet": {"title": "T"},
                                      "status": {"privacyStatus": "private"}}))
        with tempfile.TemporaryDirectory() as tmp:
            src = f"{tmp}/v.mp4"
            with open(src, "wb") as fh:
                fh.write(b"x")
            cp = _checkpoint(
                "upload_video",
                resume_state={"stage": "upload_video", "payload": {
                    "file": src, "title": "T", "description": "",
                    "tags": [], "privacy": "private", "category_id": "22",
                }},
            )
            result = conn.resume_checkpoint(cp, db=object())
        self.assertEqual(result["video_id"], "v2")

    def test_resume_upload_no_payload(self) -> None:
        conn, _http = _connected()
        cp = _checkpoint("upload_video",
                         resume_state={"stage": "upload_video"})
        with self.assertRaises(ConnectorError):
            conn.resume_checkpoint(cp, db=object())

    def test_resume_unresolved_checkpoint(self) -> None:
        conn, _http = _youtube()
        cp = _checkpoint("oauth_code", state=CheckpointState.PENDING)
        with self.assertRaises(ConnectorError):
            conn.resume_checkpoint(cp, db=object())

    def test_resume_unknown_stage(self) -> None:
        conn, _http = _youtube()
        cp = _checkpoint("bogus", resume_state={"stage": "bogus"})
        with self.assertRaises(ConnectorError):
            conn.resume_checkpoint(cp, db=object())


if __name__ == "__main__":
    unittest.main()
