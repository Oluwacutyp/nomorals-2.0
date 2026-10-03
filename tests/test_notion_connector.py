"""Notion connector tests. HTTP is fully mocked — no network."""

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
from nomorals.connectors.notion import NotionConnector, NotionError
from nomorals.connectors.registry import get_connector
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
        params = kw.pop("params", None) or {}
        if params:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{urllib.parse.urlencode(params)}"
        return self._dispatch("GET", url, None, **kw)

    def post_json(self, url: str, payload: Any, **kw: Any) -> FakeResponse:
        return self._dispatch("POST", url, payload, **kw)

    def put_json(self, url: str, payload: Any, **kw: Any) -> FakeResponse:
        return self._dispatch("PUT", url, payload, **kw)

    def post_form(self, url: str, form: Any, **kw: Any) -> FakeResponse:
        return self._dispatch("POST", url, form, **kw)

    def request(self, method: str, url: str, **kw: Any) -> FakeResponse:
        self.calls.append(
            (method.upper(), url, kw.get("data"), kw.get("headers"))
        )
        for rm, rp, resp in sorted(self.routes, key=lambda r: -len(r[1])):
            if rm == method.upper() and rp in url:
                return resp
        return FakeResponse(404, {"ok": False, "description": "not mocked"})


def _notion(http: FakeHttp | None = None) -> tuple[NotionConnector, FakeHttp]:
    http = http or FakeHttp()
    return NotionConnector(_vault(), http=http), http


def _connected(http: FakeHttp | None = None) -> tuple[NotionConnector, FakeHttp]:
    conn, http = _notion(http)
    http.route("POST", "/search", FakeResponse(200, {"results": []}))
    result = conn.connect(token="secret_test123")
    assert result.ok
    return conn, http


class RegistryTests(unittest.TestCase):
    def test_registered(self) -> None:
        self.assertIs(get_connector("notion"), NotionConnector)

    def test_metadata(self) -> None:
        self.assertEqual(NotionConnector.id, "notion")
        self.assertIn("api_key", [m.value for m in NotionConnector.auth_methods])


class ConnectTests(unittest.TestCase):
    def test_connect_validates_and_stores(self) -> None:
        conn, http = _notion()
        http.route("POST", "/search", FakeResponse(200, {
            "results": [{"id": "abc"}],
        }))
        result = conn.connect(token="secret_test123")
        self.assertTrue(result.ok)
        cred = conn.vault.get("connector:notion", "notion")
        self.assertEqual(cred.password, "secret_test123")
        _m, url, payload, headers = http.calls[0]
        self.assertIn("api.notion.com/v1/search", url)
        self.assertEqual(headers["Authorization"], "Bearer secret_test123")
        self.assertEqual(headers["Notion-Version"], "2022-06-28")
        self.assertEqual(payload["page_size"], 1)

    def test_connect_rejects_second_token(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.connect(token="other")
        self.assertIn("already connected", str(ctx.exception))

    def test_connect_empty_token_raises(self) -> None:
        conn, _http = _notion()
        with mock.patch("sys.stdin.isatty", return_value=False):
            with self.assertRaises(ConnectorError):
                conn.connect(token="")

    def test_connect_rejected_token_fails_fast(self) -> None:
        conn, http = _notion()
        http.route("POST", "/search", FakeResponse(401, {
            "code": "unauthorized", "message": "API token is invalid.",
        }))
        with self.assertRaises(NotionError) as ctx:
            conn.connect(token="bad")
        self.assertIn("401", str(ctx.exception))
        self.assertIsNone(conn._load_credential())

    def test_disconnect_clears(self) -> None:
        conn, _http = _connected()
        conn.disconnect()
        self.assertIsNone(conn._load_credential())
        conn.disconnect()  # idempotent


class StatusTests(unittest.TestCase):
    def test_status_not_connected(self) -> None:
        conn, _http = _notion()
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertFalse(conn.test_connection())

    def test_status_connected(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("POST", "/search", FakeResponse(200, {"results": []}))
        st = conn.status()
        self.assertTrue(st.connected)
        self.assertTrue(conn.test_connection())

    def test_status_rejected_token(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("POST", "/search", FakeResponse(401, {}))
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertIn("rejected", st.detail)
        self.assertFalse(conn.test_connection())


class ApiTests(unittest.TestCase):
    def test_get_page(self) -> None:
        conn, http = _connected()
        http.route("GET", "/pages/abc123", FakeResponse(200, {
            "id": "abc123", "object": "page",
            "properties": {"title": {"title": [{"plain_text": "Hi"}]}},
        }))
        page = conn.get_page("abc-123")  # dashes are stripped
        self.assertEqual(page["id"], "abc123")
        _m, url, _p, _h = http.calls[-1]
        self.assertIn("/pages/abc123", url)

    def test_get_page_empty_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.get_page("")

    def test_get_page_404_names_sharing(self) -> None:
        conn, http = _connected()
        http.route("GET", "/pages/missing", FakeResponse(404, {
            "code": "object_not_found", "message": "Could not find page.",
        }))
        with self.assertRaises(NotionError) as ctx:
            conn.get_page("missing")
        self.assertIn("Connections", str(ctx.exception))

    def test_get_page_content(self) -> None:
        conn, http = _connected()
        http.route("GET", "/blocks/abc123/children", FakeResponse(200, {
            "results": [{"id": "b1", "type": "paragraph"}],
        }))
        blocks = conn.get_page_content("abc123")
        self.assertEqual(blocks[0]["id"], "b1")

    def test_query_database(self) -> None:
        conn, http = _connected()
        http.route("POST", "/databases/db1/query", FakeResponse(200, {
            "results": [{"id": "p1"}], "has_more": False,
            "next_cursor": None,
        }))
        result = conn.query_database(
            "db1",
            filter={"property": "Status", "status": {"equals": "Done"}},
            sorts=[{"property": "Name", "direction": "ascending"}],
        )
        self.assertEqual(len(result["results"]), 1)
        self.assertFalse(result["has_more"])
        _m, url, payload, _h = http.calls[-1]
        self.assertIn("/databases/db1/query", url)
        self.assertEqual(payload["filter"]["property"], "Status")
        self.assertEqual(payload["page_size"], 50)

    def test_query_database_empty_id_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.query_database("")

    def test_list_databases(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("POST", "/search", FakeResponse(200, {
            "results": [{"id": "db1", "object": "database"}],
        }))
        dbs = conn.list_databases(query="road")
        self.assertEqual(dbs[0]["object"], "database")
        _m, _u, payload, _h = http.calls[-1]
        self.assertEqual(
            payload["filter"], {"property": "object", "value": "database"}
        )
        self.assertEqual(payload["query"], "road")

    def test_create_page_in_database_confirmed(self) -> None:
        conn, http = _connected()
        http.route("POST", "/pages", FakeResponse(200, {
            "id": "new-page", "url": "https://notion.so/new-page",
        }))
        result = conn.create_page(
            database_id="db1",
            properties={"Name": {"title": [{"text": {"content": "Task"}}]}},
            confirmed=True,
        )
        self.assertEqual(result["id"], "new-page")
        _m, url, payload, _h = http.calls[-1]
        self.assertIn("api.notion.com/v1/pages", url)
        self.assertEqual(payload["parent"], {"database_id": "db1"})

    def test_create_page_under_parent_page(self) -> None:
        conn, http = _connected()
        http.route("POST", "/pages", FakeResponse(200, {"id": "child"}))
        conn.create_page(
            parent_page_id="parent1", title="Child page",
            children=[{"object": "block", "type": "paragraph",
                       "paragraph": {"rich_text": []}}],
            confirmed=True,
        )
        _m, _u, payload, _h = http.calls[-1]
        self.assertEqual(payload["parent"], {"page_id": "parent1"})
        self.assertEqual(len(payload["children"]), 1)

    def test_create_page_requires_confirmation(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.create_page(database_id="db1", properties={})
        self.assertIn("confirmed=True", str(ctx.exception))

    def test_create_page_validates_parent(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.create_page(
                database_id="db1", parent_page_id="p1",
                properties={}, confirmed=True,
            )
        with self.assertRaises(ConnectorError):
            conn.create_page(properties={}, confirmed=True)

    def test_create_page_via_human_checkpoint(self) -> None:
        conn, http = _connected()
        db = _db()
        with self.assertRaises(HumanCheckpointPending) as ctx:
            conn.create_page(database_id="db1", properties={}, db=db)
        store = CheckpointStore(db)
        cp = store.get(ctx.exception.checkpoint.id)
        self.assertIn("db1", cp.instructions)
        http.route("POST", "/pages", FakeResponse(200, {"id": "cp-page"}))
        store.resolve(cp.id, note="approved")
        result = conn.resume_checkpoint(store.get(cp.id), db=db)
        self.assertEqual(result["id"], "cp-page")

    def test_update_page_confirmed(self) -> None:
        conn, http = _connected()
        http.route("PATCH", "/pages/abc123", FakeResponse(200, {
            "id": "abc123",
        }))
        result = conn.update_page(
            "abc123", {"Status": {"status": {"name": "Done"}}},
            confirmed=True,
        )
        self.assertEqual(result["id"], "abc123")
        _m, url, data, headers = http.calls[-1]
        self.assertEqual(_m, "PATCH")
        self.assertIn("/pages/abc123", url)
        body = json.loads(data.decode("utf-8"))
        self.assertEqual(
            body["properties"]["Status"]["status"]["name"], "Done"
        )
        self.assertEqual(headers["Notion-Version"], "2022-06-28")

    def test_update_page_requires_confirmation(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.update_page("abc123", {"Status": {}})
        self.assertIn("confirmed=True", str(ctx.exception))

    def test_update_page_validates(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.update_page("", {"a": 1}, confirmed=True)
        with self.assertRaises(ConnectorError):
            conn.update_page("abc123", {}, confirmed=True)

    def test_update_page_via_human_checkpoint(self) -> None:
        conn, http = _connected()
        db = _db()
        with self.assertRaises(HumanCheckpointPending) as ctx:
            conn.update_page("abc123", {"Status": {}}, db=db)
        store = CheckpointStore(db)
        http.route("PATCH", "/pages/abc123", FakeResponse(200, {
            "id": "abc123",
        }))
        store.resolve(ctx.exception.checkpoint.id, note="approved")
        result = conn.resume_checkpoint(
            store.get(ctx.exception.checkpoint.id), db=db
        )
        self.assertEqual(result["id"], "abc123")

    def test_not_connected_raises(self) -> None:
        conn, _http = _notion()
        with self.assertRaises(ConnectorError) as ctx:
            conn.get_page("abc")
        self.assertIn("not connected", str(ctx.exception))

    def test_rate_limit_error(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("POST", "/search", FakeResponse(429, {}))
        with self.assertRaises(NotionError) as ctx:
            conn.list_databases()
        self.assertEqual(ctx.exception.status_code, 429)

    def test_api_error_detail(self) -> None:
        conn, http = _connected()
        http.route("POST", "/pages", FakeResponse(400, {
            "code": "validation_error",
            "message": "body failed validation",
        }))
        with self.assertRaises(NotionError) as ctx:
            conn.create_page(database_id="db1", properties={},
                             confirmed=True)
        self.assertIn("validation", str(ctx.exception))
        self.assertEqual(ctx.exception.error_code, "validation_error")

    def test_invalid_json_error(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("POST", "/search", BadJsonResponse(200, "nope"))
        with self.assertRaises(NotionError) as ctx:
            conn.list_databases()
        self.assertIn("invalid JSON", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
