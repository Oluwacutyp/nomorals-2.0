"""X (Twitter) connector tests. HTTP is fully mocked — no network."""

from __future__ import annotations

import json
import unittest
import urllib.parse
from typing import Any
from unittest import mock

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors.base import ConnectorError
from nomorals.connectors.registry import get_connector
from nomorals.connectors.x import XConnector, XError
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
        return FakeResponse(404, {"title": "Not Found", "detail": "not mocked"})

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


ME = {"data": {"id": "2244994945", "name": "Devon", "username": "devon_x"}}


def _x(http: FakeHttp | None = None) -> tuple[XConnector, FakeHttp]:
    http = http or FakeHttp()
    return XConnector(_vault(), http=http), http


def _connected(http: FakeHttp | None = None) -> tuple[XConnector, FakeHttp]:
    conn, http = _x(http)
    http.route("GET", "/users/me", FakeResponse(200, ME))
    result = conn.connect(token="bearer123")
    assert result.ok
    return conn, http


class RegistryTests(unittest.TestCase):
    def test_registered(self) -> None:
        self.assertIs(get_connector("x"), XConnector)

    def test_metadata(self) -> None:
        self.assertEqual(XConnector.id, "x")
        self.assertIn("api_key", [m.value for m in XConnector.auth_methods])


class ConnectTests(unittest.TestCase):
    def test_connect_validates_and_stores(self) -> None:
        conn, http = _x()
        http.route("GET", "/users/me", FakeResponse(200, ME))
        result = conn.connect(token="bearer123")
        self.assertTrue(result.ok)
        self.assertIn("devon_x", result.account)
        cred = conn.vault.get("connector:x", "@devon_x")
        self.assertEqual(cred.password, "bearer123")
        _m, _u, _p, headers = http.calls[0]
        self.assertEqual(headers["Authorization"], "Bearer bearer123")

    def test_connect_rejects_second_token(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.connect(token="other")
        self.assertIn("already connected", str(ctx.exception))

    def test_connect_empty_token_raises(self) -> None:
        conn, _http = _x()
        with mock.patch("sys.stdin.isatty", return_value=False):
            with self.assertRaises(ConnectorError):
                conn.connect(token="")

    def test_connect_rejected_token_fails_fast(self) -> None:
        conn, http = _x()
        http.route("GET", "/users/me", FakeResponse(401, {
            "title": "Unauthorized", "detail": "Unauthorized",
            "type": "about:blank", "status": 401,
        }))
        with self.assertRaises(XError) as ctx:
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
        conn, _http = _x()
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertFalse(conn.test_connection())

    def test_status_connected(self) -> None:
        conn, http = _connected()
        http.route("GET", "/users/me", FakeResponse(200, ME))
        st = conn.status()
        self.assertTrue(st.connected)
        self.assertIn("devon_x", st.account or "")
        self.assertTrue(conn.test_connection())

    def test_status_rejected_token(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/users/me", FakeResponse(401, {"title": "x"}))
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertIn("rejected", st.detail)
        self.assertFalse(conn.test_connection())


class ApiTests(unittest.TestCase):
    def test_get_me(self) -> None:
        conn, http = _connected()
        http.route("GET", "/users/me", FakeResponse(200, ME))
        me = conn.get_me()
        self.assertEqual(me["username"], "devon_x")

    def test_post_tweet(self) -> None:
        conn, http = _connected()
        http.route("POST", "/tweets", FakeResponse(200, {
            "data": {"id": "123", "text": "hello"},
        }))
        result = conn.post_tweet("hello", confirmed=True)
        self.assertEqual(result["id"], "123")
        _m, _u, payload, _h = http.calls[-1]
        self.assertEqual(payload["text"], "hello")

    def test_post_tweet_reply(self) -> None:
        conn, http = _connected()
        http.route("POST", "/tweets", FakeResponse(200, {"data": {}}))
        conn.post_tweet("reply", reply_to_tweet_id="999", confirmed=True)
        _m, _u, payload, _h = http.calls[-1]
        self.assertEqual(payload["reply"]["in_reply_to_tweet_id"], "999")

    def test_post_tweet_empty_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.post_tweet("   ", confirmed=True)

    def test_post_tweet_too_long_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.post_tweet("x" * 281, confirmed=True)
        self.assertIn("280", str(ctx.exception))

    def test_post_tweet_needs_confirmation(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.post_tweet("hello")
        self.assertIn("confirmation", str(ctx.exception))

    def test_post_tweet_api_error(self) -> None:
        conn, http = _connected()
        http.route("POST", "/tweets", FakeResponse(403, {
            "title": "Forbidden",
            "detail": "You are not allowed to create a Tweet with duplicate content.",
        }))
        with self.assertRaises(XError) as ctx:
            conn.post_tweet("hello", confirmed=True)
        self.assertEqual(ctx.exception.status_code, 403)

    def test_get_timeline(self) -> None:
        conn, http = _connected()
        http.route("GET", "/users/me", FakeResponse(200, ME))
        http.route("GET", "/tweets", FakeResponse(200, {
            "data": [{"id": "1", "text": "a"}, {"id": "2", "text": "b"}],
            "meta": {"result_count": 2},
        }))
        tweets = conn.get_timeline()
        self.assertEqual(len(tweets), 2)
        _m, url, _p, _h = http.calls[-1]
        self.assertIn("2244994945/tweets", url)

    def test_get_timeline_explicit_user(self) -> None:
        conn, http = _connected()
        http.route("GET", "/tweets", FakeResponse(200, {"data": []}))
        conn.get_timeline(user_id="555")
        _m, url, _p, _h = http.calls[-1]
        self.assertIn("/users/555/tweets", url)

    def test_search_recent(self) -> None:
        conn, http = _connected()
        http.route("GET", "/tweets/search/recent", FakeResponse(200, {
            "data": [{"id": "9", "text": "found"}],
            "meta": {"result_count": 1},
        }))
        results = conn.search_recent("from:devon_x", max_results=50)
        self.assertEqual(results[0]["id"], "9")

    def test_search_recent_empty_query_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.search_recent("")

    def test_not_connected_raises(self) -> None:
        conn, _http = _x()
        with self.assertRaises(ConnectorError) as ctx:
            conn.get_me()
        self.assertIn("not connected", str(ctx.exception))

    def test_rate_limit_error(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/users/me", FakeResponse(429, {
            "title": "Too Many Requests",
        }))
        with self.assertRaises(XError) as ctx:
            conn.get_me()
        self.assertEqual(ctx.exception.status_code, 429)

    def test_errors_shape_error(self) -> None:
        conn, http = _connected()
        http.route("POST", "/tweets", FakeResponse(200, {
            "errors": [{"message": "duplicate", "code": 187}],
        }))
        with self.assertRaises(XError) as ctx:
            conn.post_tweet("hi", confirmed=True)
        self.assertIn("duplicate", str(ctx.exception))

    def test_invalid_json_error(self) -> None:
        conn, http = _connected()

        class BadJson(FakeResponse):
            def json(self):  # noqa: D102
                raise ValueError("nope")

        http.routes.clear()
        http.route("GET", "/users/me", BadJson(200, None))
        with self.assertRaises(XError) as ctx:
            conn.get_me()
        self.assertIn("invalid JSON", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
