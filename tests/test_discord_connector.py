"""Discord connector tests. HTTP is fully mocked — no network."""

from __future__ import annotations

import json
import unittest
import urllib.parse
from typing import Any
from unittest import mock

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors.base import ConnectorError
from nomorals.connectors.discord import DiscordConnector, DiscordError
from nomorals.connectors.registry import get_connector
from nomorals.storage.db import Database


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test")


class FakeResponse:
    def __init__(
        self,
        status: int = 200,
        payload: Any = None,
        headers: dict[str, str] | None = None,
    ) -> None:
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
        if self._payload is None:
            raise ValueError("no JSON here")
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
        return FakeResponse(404, {"message": "not mocked", "code": 0})

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


ME = {"id": "111", "username": "devonbot", "discriminator": "0",
      "bot": True}


def _discord(http: FakeHttp | None = None) -> tuple[DiscordConnector, FakeHttp]:
    http = http or FakeHttp()
    return DiscordConnector(_vault(), http=http), http


def _connected(http: FakeHttp | None = None) -> tuple[DiscordConnector, FakeHttp]:
    conn, http = _discord(http)
    http.route("GET", "/users/@me", FakeResponse(200, ME))
    result = conn.connect(token="dtok123")
    assert result.ok
    return conn, http


class RegistryTests(unittest.TestCase):
    def test_registered(self) -> None:
        self.assertIs(get_connector("discord"), DiscordConnector)

    def test_metadata(self) -> None:
        self.assertEqual(DiscordConnector.id, "discord")
        self.assertIn("api_key", [m.value for m in DiscordConnector.auth_methods])


class ConnectTests(unittest.TestCase):
    def test_connect_validates_and_stores(self) -> None:
        conn, http = _discord()
        http.route("GET", "/users/@me", FakeResponse(200, ME))
        result = conn.connect(token="dtok123")
        self.assertTrue(result.ok)
        self.assertIn("devonbot", result.account)
        cred = conn.vault.get("connector:discord", "@devonbot")
        self.assertEqual(cred.password, "dtok123")
        # bot token goes in the Authorization header, never the URL
        _m, url, _p, headers = http.calls[0]
        self.assertEqual(headers["Authorization"], "Bot dtok123")
        self.assertNotIn("dtok123", url)

    def test_connect_rejects_second_token(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.connect(token="other")
        self.assertIn("already connected", str(ctx.exception))

    def test_connect_empty_token_raises(self) -> None:
        conn, _http = _discord()
        with mock.patch("sys.stdin.isatty", return_value=False):
            with self.assertRaises(ConnectorError):
                conn.connect(token="")

    def test_connect_rejected_token_fails_fast(self) -> None:
        conn, http = _discord()
        http.route("GET", "/users/@me", FakeResponse(
            401, {"message": "401: Unauthorized", "code": 0}))
        with self.assertRaises(DiscordError) as ctx:
            conn.connect(token="bad")
        self.assertIn("rejected", str(ctx.exception))
        self.assertIsNone(conn._load_credential())

    def test_disconnect_clears(self) -> None:
        conn, _http = _connected()
        conn.disconnect()
        self.assertIsNone(conn._load_credential())
        conn.disconnect()


class StatusTests(unittest.TestCase):
    def test_status_not_connected(self) -> None:
        conn, _http = _discord()
        self.assertFalse(conn.status().connected)
        self.assertFalse(conn.test_connection())

    def test_status_connected(self) -> None:
        conn, http = _connected()
        http.route("GET", "/users/@me", FakeResponse(200, ME))
        st = conn.status()
        self.assertTrue(st.connected)
        self.assertIn("devonbot", st.account or "")
        self.assertTrue(conn.test_connection())

    def test_status_rejected_token(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/users/@me", FakeResponse(401, {"message": "x"}))
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertFalse(conn.test_connection())


class ApiTests(unittest.TestCase):
    def test_get_me(self) -> None:
        conn, http = _connected()
        http.route("GET", "/users/@me", FakeResponse(200, ME))
        me = conn.get_me()
        self.assertEqual(me["id"], "111")

    def test_get_user(self) -> None:
        conn, http = _connected()
        http.route("GET", "/users/222", FakeResponse(200, {
            "id": "222", "username": "someone", "discriminator": "0",
        }))
        user = conn.get_user("222")
        self.assertEqual(user["username"], "someone")

    def test_list_guilds(self) -> None:
        conn, http = _connected()
        http.route("GET", "/users/@me/guilds", FakeResponse(200, [
            {"id": "1", "name": "Builders"},
            {"id": "2", "name": "Ops"},
        ]))
        guilds = conn.list_guilds()
        self.assertEqual([g["name"] for g in guilds], ["Builders", "Ops"])
        _m, url, _p, _h = http.calls[-1]
        self.assertIn("limit=100", url)

    def test_list_channels(self) -> None:
        conn, http = _connected()
        http.route("GET", "/guilds/1/channels", FakeResponse(200, [
            {"id": "10", "name": "general", "type": 0},
            {"id": "11", "name": "voice", "type": 2},
        ]))
        channels = conn.list_channels("1")
        self.assertEqual(len(channels), 2)
        texts = [c for c in channels if c["type"] == 0]
        self.assertEqual(texts[0]["name"], "general")

    def test_get_channel(self) -> None:
        conn, http = _connected()
        http.route("GET", "/channels/10", FakeResponse(200, {
            "id": "10", "name": "general", "type": 0,
        }))
        channel = conn.get_channel("10")
        self.assertEqual(channel["name"], "general")

    def test_send_message(self) -> None:
        conn, http = _connected()
        http.route("POST", "/channels/10/messages", FakeResponse(200, {
            "id": "555", "content": "hello",
        }))
        result = conn.send_message("10", "hello")
        self.assertEqual(result["id"], "555")
        _m, url, payload, _h = http.calls[-1]
        self.assertIn("/channels/10/messages", url)
        self.assertEqual(payload["content"], "hello")

    def test_send_message_empty_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.send_message("10", "")

    def test_send_message_too_long_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.send_message("10", "x" * 2001)
        self.assertIn("2000", str(ctx.exception))

    def test_forbidden_means_missing_permission(self) -> None:
        conn, http = _connected()
        http.route("POST", "/channels/10/messages", FakeResponse(
            403, {"message": "Missing Permissions", "code": 50013}))
        with self.assertRaises(DiscordError) as ctx:
            conn.send_message("10", "hi")
        self.assertIn("permission", str(ctx.exception))
        self.assertEqual(ctx.exception.discord_code, 50013)

    def test_rate_limit_carries_retry_after(self) -> None:
        conn, http = _connected()
        http.route("GET", "/users/@me/guilds", FakeResponse(
            429, {"message": "slow down", "retry_after": 2.5, "code": 0}))
        with self.assertRaises(DiscordError) as ctx:
            conn.list_guilds()
        self.assertEqual(ctx.exception.status_code, 429)
        self.assertAlmostEqual(ctx.exception.retry_after, 2.5)

    def test_not_connected_raises(self) -> None:
        conn, _http = _discord()
        with self.assertRaises(ConnectorError) as ctx:
            conn.get_me()
        self.assertIn("not connected", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
