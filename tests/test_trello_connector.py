"""Trello connector tests. HTTP is fully mocked — no network."""

from __future__ import annotations

import json
import unittest
import urllib.parse
from typing import Any
from unittest import mock

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors.base import ConnectorError
from nomorals.connectors.checkpoints import (
    CheckpointStore,
    HumanCheckpointPending,
)
from nomorals.connectors.registry import get_connector
from nomorals.connectors.trello import TrelloConnector, TrelloError
from nomorals.storage.db import Database


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test")


def _db() -> Database:
    return Database(":memory:")


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


class BadJsonResponse(FakeResponse):
    def json(self) -> Any:
        raise ValueError("not json")


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
        return self._dispatch("GET", url, None, **kw)

    def post_json(self, url: str, payload: Any, **kw: Any) -> FakeResponse:
        return self._dispatch("POST", url, payload, **kw)

    def put_json(self, url: str, payload: Any, **kw: Any) -> FakeResponse:
        return self._dispatch("PUT", url, payload, **kw)

    def post_form(self, url: str, form: Any, **kw: Any) -> FakeResponse:
        return self._dispatch("POST", url, form, **kw)

    def request(self, method: str, url: str, **kw: Any) -> FakeResponse:
        return self._dispatch(method, url, None, **kw)


ME = {"id": "m1", "username": "devon_owner", "fullName": "Devon Owner"}


def _trello(http: FakeHttp | None = None) -> tuple[TrelloConnector, FakeHttp]:
    http = http or FakeHttp()
    return TrelloConnector(_vault(), http=http), http


def _connected(http: FakeHttp | None = None) -> tuple[TrelloConnector, FakeHttp]:
    conn, http = _trello(http)
    http.route("GET", "/1/members/me", FakeResponse(200, dict(ME)))
    result = conn.connect(api_key="key123", token="tok456")
    assert result.ok
    return conn, http


class RegistryTests(unittest.TestCase):
    def test_registered(self) -> None:
        self.assertIs(get_connector("trello"), TrelloConnector)

    def test_metadata(self) -> None:
        self.assertEqual(TrelloConnector.id, "trello")
        self.assertIn("api_key", [m.value for m in TrelloConnector.auth_methods])


class ConnectTests(unittest.TestCase):
    def test_connect_validates_and_stores(self) -> None:
        conn, http = _trello()
        http.route("GET", "/1/members/me", FakeResponse(200, dict(ME)))
        result = conn.connect(api_key="key123", token="tok456")
        self.assertTrue(result.ok)
        self.assertIn("@devon_owner", result.account)
        cred = conn.vault.get("connector:trello", "@devon_owner")
        self.assertEqual(cred.password, "tok456")
        self.assertEqual(cred.metadata["api_key"], "key123")
        # key + token travel as query params per Trello's API design
        _m, url, _p, _h = http.calls[0]
        self.assertIn("api.trello.com/1/members/me", url)
        self.assertIn("key=key123", url)
        self.assertIn("token=tok456", url)

    def test_connect_rejects_second_pair(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.connect(api_key="other", token="other")
        self.assertIn("already connected", str(ctx.exception))

    def test_connect_empty_raises(self) -> None:
        conn, _http = _trello()
        with mock.patch("sys.stdin.isatty", return_value=False):
            with self.assertRaises(ConnectorError):
                conn.connect(api_key="", token="")

    def test_connect_rejected_pair_fails_fast(self) -> None:
        conn, http = _trello()
        http.route("GET", "/1/members/me",
                   FakeResponse(401, "invalid token"))
        with self.assertRaises(TrelloError) as ctx:
            conn.connect(api_key="key123", token="bad")
        self.assertIn("401", str(ctx.exception))
        self.assertIsNone(conn._load_credential())

    def test_disconnect_clears(self) -> None:
        conn, _http = _connected()
        conn.disconnect()
        self.assertIsNone(conn._load_credential())
        conn.disconnect()  # idempotent


class StatusTests(unittest.TestCase):
    def test_status_not_connected(self) -> None:
        conn, _http = _trello()
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertFalse(conn.test_connection())

    def test_status_connected(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/1/members/me", FakeResponse(200, dict(ME)))
        st = conn.status()
        self.assertTrue(st.connected)
        self.assertIn("devon_owner", st.account or "")
        self.assertTrue(conn.test_connection())

    def test_status_rejected_token(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/1/members/me", FakeResponse(401, "bad"))
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertIn("rejected", st.detail)
        self.assertFalse(conn.test_connection())


class ApiTests(unittest.TestCase):
    def test_get_me(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/1/members/me", FakeResponse(200, dict(ME)))
        me = conn.get_me()
        self.assertEqual(me["username"], "devon_owner")

    def test_list_boards(self) -> None:
        conn, http = _connected()
        http.route("GET", "/1/members/me/boards", FakeResponse(200, [
            {"id": "b1", "name": "Sprint", "closed": False},
        ]))
        boards = conn.list_boards()
        self.assertEqual(boards[0]["name"], "Sprint")
        _m, url, _p, _h = http.calls[-1]
        self.assertIn("filter=open", url)

    def test_list_lists(self) -> None:
        conn, http = _connected()
        http.route("GET", "/1/boards/b1/lists", FakeResponse(200, [
            {"id": "l1", "name": "Todo"},
            {"id": "l2", "name": "Doing"},
        ]))
        lists = conn.list_lists("b1")
        self.assertEqual(len(lists), 2)
        self.assertEqual(lists[1]["name"], "Doing")

    def test_list_lists_empty_board_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.list_lists("")

    def test_list_cards(self) -> None:
        conn, http = _connected()
        http.route("GET", "/1/lists/l1/cards", FakeResponse(200, [
            {"id": "c1", "name": "Write docs"},
        ]))
        cards = conn.list_cards("l1")
        self.assertEqual(cards[0]["id"], "c1")

    def test_get_card(self) -> None:
        conn, http = _connected()
        http.route("GET", "/1/cards/c1", FakeResponse(200, {
            "id": "c1", "name": "Write docs", "idList": "l1",
        }))
        card = conn.get_card("c1")
        self.assertEqual(card["idList"], "l1")

    def test_create_card_confirmed(self) -> None:
        conn, http = _connected()
        http.route("POST", "/1/cards", FakeResponse(200, {
            "id": "c9", "name": "Ship it", "url": "https://trello.com/c/c9",
        }))
        result = conn.create_card(
            "l1", "Ship it", description="the release",
            due="2026-10-10T00:00:00Z", confirmed=True,
        )
        self.assertEqual(result["id"], "c9")
        _m, url, _p, _h = http.calls[-1]
        self.assertEqual(_m, "POST")
        self.assertIn("/1/cards", url)
        self.assertIn("idList=l1", url)
        self.assertIn("name=Ship+it", url)
        self.assertIn("due=2026-10-10", url)

    def test_create_card_requires_confirmation(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.create_card("l1", "Ship it")
        self.assertIn("confirmed=True", str(ctx.exception))

    def test_create_card_validates(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.create_card("", "Ship it", confirmed=True)
        with self.assertRaises(ConnectorError):
            conn.create_card("l1", "", confirmed=True)

    def test_create_card_via_human_checkpoint(self) -> None:
        conn, http = _connected()
        db = _db()
        with self.assertRaises(HumanCheckpointPending) as ctx:
            conn.create_card("l1", "Ship it", db=db)
        store = CheckpointStore(db)
        cp = store.get(ctx.exception.checkpoint.id)
        self.assertIn("Ship it", cp.instructions)
        http.route("POST", "/1/cards", FakeResponse(200, {"id": "c10"}))
        store.resolve(cp.id, note="approved")
        result = conn.resume_checkpoint(store.get(cp.id), db=db)
        self.assertEqual(result["id"], "c10")

    def test_move_card_confirmed(self) -> None:
        conn, http = _connected()
        http.route("PUT", "/1/cards/c1", FakeResponse(200, {
            "id": "c1", "idList": "l2",
        }))
        result = conn.move_card("c1", "l2", confirmed=True)
        self.assertEqual(result["idList"], "l2")
        _m, url, _p, _h = http.calls[-1]
        self.assertEqual(_m, "PUT")
        self.assertIn("/1/cards/c1", url)
        self.assertIn("idList=l2", url)

    def test_move_card_requires_confirmation(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.move_card("c1", "l2")
        self.assertIn("confirmed=True", str(ctx.exception))

    def test_move_card_validates(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.move_card("", "l2", confirmed=True)
        with self.assertRaises(ConnectorError):
            conn.move_card("c1", "", confirmed=True)

    def test_move_card_via_human_checkpoint(self) -> None:
        conn, http = _connected()
        db = _db()
        with self.assertRaises(HumanCheckpointPending) as ctx:
            conn.move_card("c1", "l2", db=db)
        store = CheckpointStore(db)
        http.route("PUT", "/1/cards/c1", FakeResponse(200, {"id": "c1"}))
        store.resolve(ctx.exception.checkpoint.id, note="approved")
        result = conn.resume_checkpoint(
            store.get(ctx.exception.checkpoint.id), db=db
        )
        self.assertEqual(result["id"], "c1")

    def test_not_connected_raises(self) -> None:
        conn, _http = _trello()
        with self.assertRaises(ConnectorError) as ctx:
            conn.get_me()
        self.assertIn("not connected", str(ctx.exception))

    def test_token_scrubbed_from_errors(self) -> None:
        conn, http = _connected()
        http.route("GET", "/1/cards/c1", FakeResponse(400, "bad request"))
        with self.assertRaises(TrelloError) as ctx:
            conn.get_card("c1")
        message = str(ctx.exception)
        self.assertIn("<token>", message)
        self.assertNotIn("tok456", message)

    def test_rate_limit_error(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/1/members/me", FakeResponse(429, "slow"))
        with self.assertRaises(TrelloError) as ctx:
            conn.get_me()
        self.assertEqual(ctx.exception.status_code, 429)

    def test_invalid_json_error(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/1/members/me", BadJsonResponse(200, "nope"))
        with self.assertRaises(TrelloError) as ctx:
            conn.get_me()
        self.assertIn("invalid JSON", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
