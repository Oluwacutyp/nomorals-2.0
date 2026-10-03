"""Slack connector tests. HTTP is fully mocked — no network."""

from __future__ import annotations

import json
import unittest
import urllib.parse
from typing import Any
from unittest import mock

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors.base import ConnectorError
from nomorals.connectors.registry import get_connector
from nomorals.connectors.slack import SlackConnector, SlackError
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
        return FakeResponse(200, {"ok": False, "error": "not_mocked"})

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


AUTH = {"ok": True, "user": "devonbot", "user_id": "U123",
        "team": "DevonHQ", "team_id": "T456"}


def _slack(http: FakeHttp | None = None) -> tuple[SlackConnector, FakeHttp]:
    http = http or FakeHttp()
    return SlackConnector(_vault(), http=http), http


def _connected(
    http: FakeHttp | None = None,
) -> tuple[SlackConnector, FakeHttp]:
    conn, http = _slack(http)
    http.route("POST", "/auth.test", FakeResponse(200, AUTH))
    result = conn.connect(token="xoxb-123")
    assert result.ok
    return conn, http


class RegistryTests(unittest.TestCase):
    def test_registered(self) -> None:
        self.assertIs(get_connector("slack"), SlackConnector)

    def test_metadata(self) -> None:
        self.assertEqual(SlackConnector.id, "slack")
        self.assertIn("api_key", [m.value for m in SlackConnector.auth_methods])


class ConnectTests(unittest.TestCase):
    def test_connect_validates_and_stores(self) -> None:
        conn, http = _slack()
        http.route("POST", "/auth.test", FakeResponse(200, AUTH))
        result = conn.connect(token="xoxb-123")
        self.assertTrue(result.ok)
        self.assertIn("devonbot", result.account)
        cred = conn.vault.get("connector:slack", "@devonbot@DevonHQ")
        self.assertEqual(cred.password, "xoxb-123")
        _m, _u, _p, headers = http.calls[0]
        self.assertEqual(headers["Authorization"], "Bearer xoxb-123")

    def test_connect_rejects_second_token(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.connect(token="other")
        self.assertIn("already connected", str(ctx.exception))

    def test_connect_empty_token_raises(self) -> None:
        conn, _http = _slack()
        with mock.patch("sys.stdin.isatty", return_value=False):
            with self.assertRaises(ConnectorError):
                conn.connect(token="")

    def test_connect_rejected_token_fails_fast(self) -> None:
        conn, http = _slack()
        http.route("POST", "/auth.test", FakeResponse(200, {
            "ok": False, "error": "invalid_auth",
        }))
        with self.assertRaises(SlackError) as ctx:
            conn.connect(token="bad")
        self.assertIn("invalid_auth", str(ctx.exception))
        self.assertIn("invalid or revoked", str(ctx.exception))
        self.assertIsNone(conn._load_credential())

    def test_disconnect_clears(self) -> None:
        conn, _http = _connected()
        conn.disconnect()
        self.assertIsNone(conn._load_credential())
        conn.disconnect()  # idempotent


class StatusTests(unittest.TestCase):
    def test_status_not_connected(self) -> None:
        conn, _http = _slack()
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertFalse(conn.test_connection())

    def test_status_connected(self) -> None:
        conn, http = _connected()
        http.route("POST", "/auth.test", FakeResponse(200, AUTH))
        st = conn.status()
        self.assertTrue(st.connected)
        self.assertIn("devonbot", st.account or "")
        self.assertTrue(conn.test_connection())

    def test_status_rejected_token(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("POST", "/auth.test", FakeResponse(200, {
            "ok": False, "error": "token_revoked"}))
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertIn("rejected", st.detail)
        self.assertFalse(conn.test_connection())


class ApiTests(unittest.TestCase):
    def test_test_auth(self) -> None:
        conn, http = _connected()
        http.route("POST", "/auth.test", FakeResponse(200, AUTH))
        info = conn.test_auth()
        self.assertEqual(info["team_id"], "T456")

    def test_list_channels(self) -> None:
        conn, http = _connected()
        http.route("GET", "/conversations.list", FakeResponse(200, {
            "ok": True,
            "channels": [{"id": "C1", "name": "general"}],
            "response_metadata": {"next_cursor": "abc"},
        }))
        result = conn.list_channels()
        self.assertEqual(result["channels"][0]["name"], "general")
        self.assertEqual(result["next_cursor"], "abc")

    def test_list_channels_cursor_passthrough(self) -> None:
        conn, http = _connected()
        http.route("GET", "/conversations.list", FakeResponse(200, {
            "ok": True, "channels": [],
            "response_metadata": {"next_cursor": ""},
        }))
        conn.list_channels(cursor="prev")
        _m, url, _p, _h = http.calls[-1]
        self.assertIn("cursor=prev", url)

    def test_send_message(self) -> None:
        conn, http = _connected()
        http.route("POST", "/chat.postMessage", FakeResponse(200, {
            "ok": True, "ts": "1234.5", "channel": "C1",
        }))
        result = conn.send_message("C1", "hello", confirmed=True)
        self.assertEqual(result["ts"], "1234.5")
        _m, _u, payload, _h = http.calls[-1]
        self.assertEqual(payload["channel"], "C1")
        self.assertEqual(payload["text"], "hello")

    def test_send_message_empty_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.send_message("C1", "", confirmed=True)

    def test_send_message_no_channel_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.send_message("", "hi", confirmed=True)

    def test_send_message_too_long_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.send_message("C1", "x" * 40001, confirmed=True)
        self.assertIn("40000", str(ctx.exception))

    def test_send_message_needs_confirmation(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.send_message("C1", "hi")
        self.assertIn("confirmation", str(ctx.exception))

    def test_send_message_slack_error(self) -> None:
        conn, http = _connected()
        http.route("POST", "/chat.postMessage", FakeResponse(200, {
            "ok": False, "error": "channel_not_found",
        }))
        with self.assertRaises(SlackError) as ctx:
            conn.send_message("C9", "hi", confirmed=True)
        self.assertEqual(ctx.exception.slack_error, "channel_not_found")
        self.assertIn("invited", str(ctx.exception))

    def test_read_history(self) -> None:
        conn, http = _connected()
        http.route("GET", "/conversations.history", FakeResponse(200, {
            "ok": True,
            "messages": [{"ts": "1", "text": "a"}, {"ts": "2", "text": "b"}],
            "has_more": True,
            "response_metadata": {"next_cursor": "nxt"},
        }))
        result = conn.read_history("C1", limit=2)
        self.assertEqual(len(result["messages"]), 2)
        self.assertTrue(result["has_more"])
        self.assertEqual(result["next_cursor"], "nxt")

    def test_read_history_no_channel_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.read_history("")

    def test_not_connected_raises(self) -> None:
        conn, _http = _slack()
        with self.assertRaises(ConnectorError) as ctx:
            conn.test_auth()
        self.assertIn("not connected", str(ctx.exception))

    def test_rate_limit_error(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("POST", "/auth.test", FakeResponse(429, {
            "ok": False, "error": "ratelimited",
        }))
        with self.assertRaises(SlackError) as ctx:
            conn.test_auth()
        self.assertEqual(ctx.exception.status_code, 429)

    def test_invalid_json_error(self) -> None:
        conn, http = _connected()

        class BadJson(FakeResponse):
            def json(self):  # noqa: D102
                raise ValueError("nope")

        http.routes.clear()
        http.route("POST", "/auth.test", BadJson(200, None))
        with self.assertRaises(SlackError) as ctx:
            conn.test_auth()
        self.assertIn("invalid JSON", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
