"""Telegram connector tests. HTTP is fully mocked — no network."""

from __future__ import annotations

import json
import unittest
import urllib.parse
from typing import Any
from unittest import mock

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors.base import ConnectorError
from nomorals.connectors.registry import get_connector
from nomorals.connectors.telegram import TelegramConnector, TelegramError
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
        return FakeResponse(404, {"ok": False, "description": "not mocked"})

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


ME = {"ok": True, "result": {
    "id": 123456, "is_bot": True, "first_name": "Devon",
    "username": "devon_test_bot", "can_join_groups": True,
    "can_read_all_group_messages": False,
}}


def _telegram(http: FakeHttp | None = None) -> tuple[TelegramConnector, FakeHttp]:
    http = http or FakeHttp()
    return TelegramConnector(_vault(), http=http), http


def _connected(http: FakeHttp | None = None) -> tuple[TelegramConnector, FakeHttp]:
    conn, http = _telegram(http)
    http.route("POST", "/getMe", FakeResponse(200, ME))
    result = conn.connect(token="tok123")
    assert result.ok
    return conn, http


class RegistryTests(unittest.TestCase):
    def test_registered(self) -> None:
        self.assertIs(get_connector("telegram"), TelegramConnector)

    def test_metadata(self) -> None:
        self.assertEqual(TelegramConnector.id, "telegram")
        self.assertIn("api_key", [m.value for m in TelegramConnector.auth_methods])


class ConnectTests(unittest.TestCase):
    def test_connect_validates_and_stores(self) -> None:
        conn, http = _telegram()
        http.route("POST", "/getMe", FakeResponse(200, ME))
        result = conn.connect(token="tok123")
        self.assertTrue(result.ok)
        self.assertIn("devon_test_bot", result.account)
        cred = conn.vault.get("connector:telegram", "@devon_test_bot")
        self.assertEqual(cred.password, "tok123")
        # token travels in the URL path per Bot API design
        _m, url, _p, _h = http.calls[0]
        self.assertIn("api.telegram.org/bot", url)

    def test_connect_rejects_second_token(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.connect(token="other")
        self.assertIn("already connected", str(ctx.exception))

    def test_connect_empty_token_raises(self) -> None:
        conn, _http = _telegram()
        with mock.patch("sys.stdin.isatty", return_value=False):
            with self.assertRaises(ConnectorError):
                conn.connect(token="")

    def test_connect_rejected_token_fails_fast(self) -> None:
        conn, http = _telegram()
        http.route("POST", "/getMe", FakeResponse(401, {
            "ok": False, "error_code": 401,
            "description": "Unauthorized",
        }))
        with self.assertRaises(TelegramError) as ctx:
            conn.connect(token="bad")
        self.assertIn("rejected", str(ctx.exception))
        self.assertIsNone(conn._load_credential())

    def test_connect_api_error_raises(self) -> None:
        conn, http = _telegram()
        http.route("POST", "/getMe", FakeResponse(200, {
            "ok": False, "error_code": 404, "description": "Not Found",
        }))
        with self.assertRaises(TelegramError) as ctx:
            conn.connect(token="tok123")
        self.assertIn("Not Found", str(ctx.exception))

    def test_disconnect_clears(self) -> None:
        conn, _http = _connected()
        conn.disconnect()
        self.assertIsNone(conn._load_credential())
        conn.disconnect()  # idempotent


class StatusTests(unittest.TestCase):
    def test_status_not_connected(self) -> None:
        conn, _http = _telegram()
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertFalse(conn.test_connection())

    def test_status_connected(self) -> None:
        conn, http = _connected()
        http.route("POST", "/getMe", FakeResponse(200, ME))
        st = conn.status()
        self.assertTrue(st.connected)
        self.assertIn("devon_test_bot", st.account or "")
        self.assertTrue(conn.test_connection())

    def test_status_rejected_token(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("POST", "/getMe", FakeResponse(401, {"ok": False}))
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertIn("rejected", st.detail)
        self.assertFalse(conn.test_connection())


class ApiTests(unittest.TestCase):
    def test_get_me(self) -> None:
        conn, http = _connected()
        http.route("POST", "/getMe", FakeResponse(200, ME))
        me = conn.get_me()
        self.assertEqual(me["username"], "devon_test_bot")

    def test_send_message(self) -> None:
        conn, http = _connected()
        http.route("POST", "/sendMessage", FakeResponse(200, {
            "ok": True, "result": {"message_id": 7, "text": "hi"},
        }))
        result = conn.send_message(12345, "hi")
        self.assertEqual(result["message_id"], 7)
        _m, _u, payload, _h = http.calls[-1]
        self.assertEqual(payload["chat_id"], 12345)
        self.assertEqual(payload["text"], "hi")

    def test_send_message_reply_and_parse_mode(self) -> None:
        conn, http = _connected()
        http.route("POST", "/sendMessage", FakeResponse(200, {
            "ok": True, "result": {"message_id": 8},
        }))
        conn.send_message("@channel", "<b>hi</b>", parse_mode="HTML",
                         reply_to_message_id=42,
                         disable_notification=True)
        _m, _u, payload, _h = http.calls[-1]
        self.assertEqual(payload["parse_mode"], "HTML")
        self.assertEqual(payload["reply_to_message_id"], 42)
        self.assertTrue(payload["disable_notification"])

    def test_send_message_empty_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.send_message(1, "")

    def test_send_message_too_long_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.send_message(1, "x" * 4097)
        self.assertIn("4096", str(ctx.exception))

    def test_send_message_bad_parse_mode_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.send_message(1, "hi", parse_mode="BBCode")

    def test_send_message_api_error(self) -> None:
        conn, http = _connected()
        http.route("POST", "/sendMessage", FakeResponse(200, {
            "ok": False, "error_code": 400,
            "description": "Bad Request: chat not found",
        }))
        with self.assertRaises(TelegramError) as ctx:
            conn.send_message(999, "hi")
        self.assertIn("chat not found", str(ctx.exception))

    def test_get_updates(self) -> None:
        conn, http = _connected()
        http.route("POST", "/getUpdates", FakeResponse(200, {
            "ok": True, "result": [
                {"update_id": 1,
                 "message": {"message_id": 1, "text": "hello"}},
            ],
        }))
        updates = conn.get_updates(timeout=10)
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0]["update_id"], 1)

    def test_get_chat(self) -> None:
        conn, http = _connected()
        http.route("POST", "/getChat", FakeResponse(200, {
            "ok": True,
            "result": {"id": -1001, "title": "Ops", "type": "supergroup"},
        }))
        chat = conn.get_chat(-1001)
        self.assertEqual(chat["title"], "Ops")

    def test_leave_chat(self) -> None:
        conn, http = _connected()
        http.route("POST", "/leaveChat", FakeResponse(200, {
            "ok": True, "result": True,
        }))
        self.assertTrue(conn.leave_chat(-1001))

    def test_not_connected_raises(self) -> None:
        conn, _http = _telegram()
        with self.assertRaises(ConnectorError) as ctx:
            conn.send_message(1, "hi")
        self.assertIn("not connected", str(ctx.exception))

    def test_rate_limit_error(self) -> None:
        conn, http = _connected()
        http.route("POST", "/sendMessage", FakeResponse(429, {
            "ok": False, "description": "Too Many Requests",
        }))
        with self.assertRaises(TelegramError) as ctx:
            conn.send_message(1, "hi")
        self.assertEqual(ctx.exception.status_code, 429)


if __name__ == "__main__":
    unittest.main()
