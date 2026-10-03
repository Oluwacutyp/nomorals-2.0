"""Instagram connector tests. HTTP is fully mocked — no network."""

from __future__ import annotations

import json
import unittest
import urllib.parse
from typing import Any
from unittest import mock

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors.base import ConnectorError
from nomorals.connectors.instagram import InstagramConnector, InstagramError
from nomorals.connectors.registry import get_connector
from nomorals.storage.db import Database


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test")


class FakeResponse:
    def __init__(self, status: int = 200, payload: Any = None) -> None:
        self.status = status
        self._payload = payload

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
        return FakeResponse(400, {"error": {"message": "not mocked",
                                            "code": 1}})

    def get(self, url: str, **kw: Any) -> FakeResponse:
        params = kw.pop("params", None) or {}
        if params:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{urllib.parse.urlencode(params)}"
        return self._dispatch("GET", url, None, **kw)

    def post_json(self, url: str, payload: Any, **kw: Any) -> FakeResponse:
        params = kw.pop("params", None) or {}
        if params:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{urllib.parse.urlencode(params)}"
        return self._dispatch("POST", url, payload, **kw)

    def request(self, method: str, url: str, **kw: Any) -> FakeResponse:
        return self._dispatch(method, url, None, **kw)


ME = {"id": "1020384", "name": "Devon Owner"}
PAGES = {"data": [{
    "id": "555", "name": "Devon Page",
    "instagram_business_account": {"id": "1784140000",
                                   "username": "devon.ig"},
}]}
PROFILE = {"id": "1784140000", "username": "devon.ig"}


def _ig(http: FakeHttp | None = None) -> tuple[InstagramConnector, FakeHttp]:
    http = http or FakeHttp()
    return InstagramConnector(_vault(), http=http), http


def _connected(
    http: FakeHttp | None = None,
) -> tuple[InstagramConnector, FakeHttp]:
    conn, http = _ig(http)
    http.route("GET", "/me", FakeResponse(200, ME))
    http.route("GET", "/me/accounts", FakeResponse(200, PAGES))
    result = conn.connect(token="EAAtok")
    assert result.ok
    return conn, http


class RegistryTests(unittest.TestCase):
    def test_registered(self) -> None:
        self.assertIs(get_connector("instagram"), InstagramConnector)

    def test_metadata(self) -> None:
        self.assertEqual(InstagramConnector.id, "instagram")
        self.assertIn("oauth2", [m.value for m in
                                 InstagramConnector.auth_methods])


class ConnectTests(unittest.TestCase):
    def test_connect_validates_and_stores(self) -> None:
        conn, http = _ig()
        http.route("GET", "/me", FakeResponse(200, ME))
        http.route("GET", "/me/accounts", FakeResponse(200, PAGES))
        result = conn.connect(token="EAAtok")
        self.assertTrue(result.ok)
        self.assertIn("devon.ig", result.account)
        cred = conn.vault.get("connector:instagram", "@devon.ig")
        self.assertEqual(cred.password, "EAAtok")
        self.assertEqual(cred.metadata["ig_user_id"], "1784140000")
        _m, url, _p, _h = http.calls[0]
        self.assertIn("access_token", url)

    def test_connect_rejects_second_token(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.connect(token="other")
        self.assertIn("already connected", str(ctx.exception))

    def test_connect_empty_token_prints_grant_guide(self) -> None:
        conn, _http = _ig()
        with mock.patch("sys.stdin.isatty", return_value=False):
            with self.assertRaises(ConnectorError) as ctx:
                conn.connect(token="")
        self.assertIn("developers.facebook.com", str(ctx.exception))

    def test_connect_no_business_account_raises(self) -> None:
        conn, http = _ig()
        http.route("GET", "/me", FakeResponse(200, ME))
        http.route("GET", "/me/accounts", FakeResponse(200, {"data": []}))
        with self.assertRaises(ConnectorError) as ctx:
            conn.connect(token="EAAtok")
        self.assertIn("business/creator", str(ctx.exception))
        self.assertIsNone(conn._load_credential())

    def test_connect_rejected_token_fails_fast(self) -> None:
        conn, http = _ig()
        http.route("GET", "/me", FakeResponse(400, {
            "error": {"message": "Invalid OAuth access token.",
                      "code": 190},
        }))
        with self.assertRaises(InstagramError) as ctx:
            conn.connect(token="bad")
        self.assertIn("Invalid OAuth access token", str(ctx.exception))
        self.assertIsNone(conn._load_credential())

    def test_disconnect_clears(self) -> None:
        conn, _http = _connected()
        conn.disconnect()
        self.assertIsNone(conn._load_credential())
        conn.disconnect()  # idempotent


class StatusTests(unittest.TestCase):
    def test_status_not_connected(self) -> None:
        conn, _http = _ig()
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertFalse(conn.test_connection())

    def test_status_connected(self) -> None:
        conn, http = _connected()
        http.route("GET", "/me", FakeResponse(200, ME))
        http.route("GET", "/1784140000", FakeResponse(200, PROFILE))
        st = conn.status()
        self.assertTrue(st.connected)
        self.assertIn("devon.ig", st.account or "")
        self.assertTrue(conn.test_connection())

    def test_status_rejected_token(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/me", FakeResponse(400, {
            "error": {"message": "expired", "code": 190}}))
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertIn("rejected", st.detail)
        self.assertFalse(conn.test_connection())


class ApiTests(unittest.TestCase):
    def test_get_me(self) -> None:
        conn, http = _connected()
        http.route("GET", "/me", FakeResponse(200, ME))
        me = conn.get_me()
        self.assertEqual(me["id"], "1020384")

    def test_get_profile(self) -> None:
        conn, http = _connected()
        http.route("GET", "/1784140000", FakeResponse(200, {
            "id": "1784140000", "username": "devon.ig",
            "account_type": "BUSINESS", "media_count": 42,
        }))
        profile = conn.get_profile()
        self.assertEqual(profile["account_type"], "BUSINESS")

    def test_list_media(self) -> None:
        conn, http = _connected()
        http.route("GET", "/1784140000/media", FakeResponse(200, {
            "data": [{"id": "1", "media_type": "IMAGE"}],
        }))
        media = conn.list_media()
        self.assertEqual(len(media), 1)
        _m, url, _p, _h = http.calls[-1]
        self.assertIn("1784140000/media", url)

    def test_create_media_container(self) -> None:
        conn, http = _connected()
        http.route("POST", "/1784140000/media", FakeResponse(200,
                                                             {"id": "c1"}))
        cid = conn.create_media_container("https://example.com/pic.jpg",
                                          caption="hi")
        self.assertEqual(cid, "c1")
        _m, _u, payload, _h = http.calls[-1]
        self.assertEqual(payload["image_url"], "https://example.com/pic.jpg")

    def test_create_media_container_video(self) -> None:
        conn, http = _connected()
        http.route("POST", "/1784140000/media", FakeResponse(200,
                                                             {"id": "c2"}))
        conn.create_media_container(video_url="https://example.com/v.mp4")
        _m, _u, payload, _h = http.calls[-1]
        self.assertEqual(payload["media_type"], "VIDEO")

    def test_create_media_container_no_url_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.create_media_container()

    def test_create_media_container_both_urls_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.create_media_container(
                "https://a.jpg", video_url="https://b.mp4")

    def test_create_media_container_non_https_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.create_media_container("http://a.jpg")

    def test_publish_media(self) -> None:
        conn, http = _connected()
        http.route("POST", "/media_publish", FakeResponse(200, {"id": "m9"}))
        result = conn.publish_media("c1")
        self.assertEqual(result["id"], "m9")
        _m, _u, payload, _h = http.calls[-1]
        self.assertEqual(payload["creation_id"], "c1")

    def test_publish_media_empty_id_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.publish_media("")

    def test_container_status(self) -> None:
        conn, http = _connected()
        http.route("GET", "/c2", FakeResponse(200,
                                             {"status_code": "FINISHED"}))
        self.assertEqual(conn.container_status("c2"), "FINISHED")

    def test_publish_image_two_step(self) -> None:
        conn, http = _connected()
        http.route("POST", "/1784140000/media?access_token",
                   FakeResponse(200, {"id": "c1"}))
        http.route("POST", "/media_publish?access_token",
                   FakeResponse(200, {"id": "m9"}))
        result = conn.publish_image("https://example.com/pic.jpg",
                                    caption="hi", confirmed=True)
        self.assertEqual(result["id"], "m9")
        methods = [c[0] for c in http.calls[-2:]]
        self.assertEqual(methods, ["POST", "POST"])

    def test_publish_image_needs_confirmation(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.publish_image("https://example.com/pic.jpg")
        self.assertIn("confirmation", str(ctx.exception))

    def test_not_connected_raises(self) -> None:
        conn, _http = _ig()
        with self.assertRaises(ConnectorError) as ctx:
            conn.list_media()
        self.assertIn("not connected", str(ctx.exception))

    def test_rate_limit_error(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/me", FakeResponse(429, {
            "error": {"message": "throttled", "code": 4}}))
        with self.assertRaises(InstagramError) as ctx:
            conn.get_me()
        self.assertEqual(ctx.exception.status_code, 429)


if __name__ == "__main__":
    unittest.main()
