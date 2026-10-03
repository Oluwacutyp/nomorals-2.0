"""LinkedIn connector tests. HTTP is fully mocked — no network."""

from __future__ import annotations

import json
import unittest
import urllib.parse
from typing import Any
from unittest import mock

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors.base import ConnectorError
from nomorals.connectors.linkedin import LinkedInConnector, LinkedInError
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
        return FakeResponse(404, {"message": "not mocked"})

    def get(self, url: str, **kw: Any) -> FakeResponse:
        params = kw.pop("params", None) or {}
        if params:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{urllib.parse.urlencode(params)}"
        return self._dispatch("GET", url, None, **kw)

    def post_json(self, url: str, payload: Any, **kw: Any) -> FakeResponse:
        return self._dispatch("POST", url, payload, **kw)

    def request(self, method: str, url: str, **kw: Any) -> FakeResponse:
        return self._dispatch(method, url, None, **kw)


USERINFO = {"sub": "abc123", "name": "Devon Owner",
            "email": "devon@example.com"}


def _li(http: FakeHttp | None = None) -> tuple[LinkedInConnector, FakeHttp]:
    http = http or FakeHttp()
    return LinkedInConnector(_vault(), http=http), http


def _connected(
    http: FakeHttp | None = None,
) -> tuple[LinkedInConnector, FakeHttp]:
    conn, http = _li(http)
    http.route("GET", "/v2/userinfo", FakeResponse(200, USERINFO))
    result = conn.connect(token="li_tok")
    assert result.ok
    return conn, http


class RegistryTests(unittest.TestCase):
    def test_registered(self) -> None:
        self.assertIs(get_connector("linkedin"), LinkedInConnector)

    def test_metadata(self) -> None:
        self.assertEqual(LinkedInConnector.id, "linkedin")
        self.assertIn("oauth2", [m.value for m in
                                 LinkedInConnector.auth_methods])


class ConnectTests(unittest.TestCase):
    def test_connect_validates_and_stores(self) -> None:
        conn, http = _li()
        http.route("GET", "/v2/userinfo", FakeResponse(200, USERINFO))
        result = conn.connect(token="li_tok")
        self.assertTrue(result.ok)
        self.assertIn("Devon Owner", result.account)
        cred = conn.vault.get("connector:linkedin", "Devon Owner")
        self.assertEqual(cred.password, "li_tok")
        self.assertEqual(cred.metadata["person_urn"], "urn:li:person:abc123")
        _m, _u, _p, headers = http.calls[0]
        self.assertEqual(headers["Authorization"], "Bearer li_tok")

    def test_connect_rejects_second_token(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.connect(token="other")
        self.assertIn("already connected", str(ctx.exception))

    def test_connect_empty_token_prints_grant_guide(self) -> None:
        conn, _http = _li()
        with mock.patch("sys.stdin.isatty", return_value=False):
            with self.assertRaises(ConnectorError) as ctx:
                conn.connect(token="")
        self.assertIn("developer.linkedin.com", str(ctx.exception))

    def test_connect_missing_sub_raises(self) -> None:
        conn, http = _li()
        http.route("GET", "/v2/userinfo", FakeResponse(200, {"name": "x"}))
        with self.assertRaises(LinkedInError) as ctx:
            conn.connect(token="li_tok")
        self.assertIn("openid", str(ctx.exception))
        self.assertIsNone(conn._load_credential())

    def test_connect_rejected_token_fails_fast(self) -> None:
        conn, http = _li()
        http.route("GET", "/v2/userinfo", FakeResponse(401, {
            "status": 401, "serviceErrorCode": 65600,
            "message": "Invalid access token",
        }))
        with self.assertRaises(LinkedInError) as ctx:
            conn.connect(token="bad")
        self.assertIn("rejected", str(ctx.exception))
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIsNone(conn._load_credential())

    def test_disconnect_clears(self) -> None:
        conn, _http = _connected()
        conn.disconnect()
        self.assertIsNone(conn._load_credential())
        conn.disconnect()  # idempotent


class StatusTests(unittest.TestCase):
    def test_status_not_connected(self) -> None:
        conn, _http = _li()
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertFalse(conn.test_connection())

    def test_status_connected(self) -> None:
        conn, http = _connected()
        http.route("GET", "/v2/userinfo", FakeResponse(200, USERINFO))
        st = conn.status()
        self.assertTrue(st.connected)
        self.assertIn("Devon Owner", st.account or "")
        self.assertTrue(conn.test_connection())

    def test_status_rejected_token(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/v2/userinfo", FakeResponse(401, {
            "message": "expired"}))
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertIn("rejected", st.detail)
        self.assertFalse(conn.test_connection())


class ApiTests(unittest.TestCase):
    def test_get_me(self) -> None:
        conn, http = _connected()
        http.route("GET", "/v2/userinfo", FakeResponse(200, USERINFO))
        me = conn.get_me()
        self.assertEqual(me["sub"], "abc123")

    def test_post(self) -> None:
        conn, http = _connected()
        http.route("POST", "/rest/posts", FakeResponse(201, {
            "id": "urn:li:share:999"}))
        result = conn.post("hello linkedin", confirmed=True)
        self.assertEqual(result["id"], "urn:li:share:999")
        _m, _u, payload, _h = http.calls[-1]
        self.assertEqual(payload["author"], "urn:li:person:abc123")
        self.assertEqual(payload["commentary"], "hello linkedin")
        self.assertEqual(payload["visibility"], "PUBLIC")

    def test_post_connections_visibility(self) -> None:
        conn, http = _connected()
        http.route("POST", "/rest/posts", FakeResponse(201, {}))
        conn.post("hi", visibility="CONNECTIONS", confirmed=True)
        _m, _u, payload, _h = http.calls[-1]
        self.assertEqual(payload["visibility"], "CONNECTIONS")

    def test_post_empty_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.post("  ", confirmed=True)

    def test_post_too_long_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.post("x" * 3001, confirmed=True)
        self.assertIn("3000", str(ctx.exception))

    def test_post_bad_visibility_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.post("hi", visibility="FRIENDS", confirmed=True)

    def test_post_needs_confirmation(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.post("hi")
        self.assertIn("confirmation", str(ctx.exception))

    def test_post_403_missing_scope(self) -> None:
        conn, http = _connected()
        http.route("POST", "/rest/posts", FakeResponse(403, {
            "status": 403, "serviceErrorCode": 100,
            "message": "Not enough permissions",
        }))
        with self.assertRaises(LinkedInError) as ctx:
            conn.post("hi", confirmed=True)
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIn("w_member_social", str(ctx.exception))

    def test_get_post(self) -> None:
        conn, http = _connected()
        http.route("GET", "/rest/posts/", FakeResponse(200, {
            "id": "urn:li:share:999", "commentary": "hi"}))
        post = conn.get_post("urn:li:share:999")
        self.assertEqual(post["commentary"], "hi")
        _m, url, _p, _h = http.calls[-1]
        self.assertIn("/rest/posts/urn:li:share:999", url)

    def test_get_post_empty_id_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.get_post("")

    def test_not_connected_raises(self) -> None:
        conn, _http = _li()
        with self.assertRaises(ConnectorError) as ctx:
            conn.get_me()
        self.assertIn("not connected", str(ctx.exception))

    def test_rate_limit_error(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/v2/userinfo", FakeResponse(429, {
            "message": "throttled"}))
        with self.assertRaises(LinkedInError) as ctx:
            conn.get_me()
        self.assertEqual(ctx.exception.status_code, 429)

    def test_invalid_json_error(self) -> None:
        conn, http = _connected()

        class BadJson(FakeResponse):
            def json(self):  # noqa: D102
                raise ValueError("nope")

        http.routes.clear()
        http.route("GET", "/v2/userinfo", BadJson(200, None))
        with self.assertRaises(LinkedInError) as ctx:
            conn.get_me()
        self.assertIn("invalid JSON", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
