"""Dropbox connector tests. HTTP is fully mocked — no network."""

from __future__ import annotations

import json
import os
import unittest
import urllib.parse
from typing import Any
from unittest import mock

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors.base import ConnectorError
from nomorals.connectors.dropbox import DropboxConnector, DropboxError
from nomorals.connectors.registry import get_connector
from nomorals.storage.db import Database


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test")


class FakeResponse:
    def __init__(
        self,
        status: int = 200,
        payload: Any = None,
        *,
        headers: dict[str, str] | None = None,
        body: bytes = b"",
    ) -> None:
        self.status = status
        self._payload = payload
        self.headers = headers or {}
        self.body = body

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
        return FakeResponse(404, {"error_summary": "not mocked"})

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


ACCOUNT = {
    "account_id": "dbid:AAH4f99T0taONIb-OurWxbNQ6ywGRopQngc",
    "email": "devon@example.com",
    "name": {"display_name": "Devon"},
}

TOKENS = {
    "access_token": "sl.short",
    "refresh_token": "refresh123",
    "expires_in": 14400,
    "token_type": "bearer",
}

FOLDER = {
    "entries": [
        {".tag": "folder", "name": "Docs", "path_lower": "/docs",
         "id": "id:folder1"},
        {".tag": "file", "name": "a.txt", "path_lower": "/a.txt",
         "size": 11, "id": "id:file1", "client_modified": "2026-01-01"},
    ],
    "cursor": "cursor1",
    "has_more": False,
}

UPLOAD_RESULT = {
    "name": "up.txt", "path_lower": "/up.txt", "size": 7, "id": "id:up1",
}


def _dropbox(http: FakeHttp | None = None) -> tuple[DropboxConnector, FakeHttp]:
    http = http or FakeHttp()
    return DropboxConnector(_vault(), http=http), http


def _connected_token(
    http: FakeHttp | None = None,
) -> tuple[DropboxConnector, FakeHttp]:
    conn, http = _dropbox(http)
    http.route("POST", "/2/users/get_current_account",
               FakeResponse(200, ACCOUNT))
    result = conn.connect(token="long-lived-token")
    assert result.ok
    return conn, http


def _connected_oauth(
    http: FakeHttp | None = None,
) -> tuple[DropboxConnector, FakeHttp]:
    conn, http = _dropbox(http)
    http.route("POST", "oauth2/token", FakeResponse(200, TOKENS))
    http.route("POST", "/2/users/get_current_account",
               FakeResponse(200, ACCOUNT))
    result = conn.connect(client_id="appkey", client_secret="appsecret",
                          code="authcode")
    assert result.ok
    return conn, http


class RegistryTests(unittest.TestCase):
    def test_registered(self) -> None:
        self.assertIs(get_connector("dropbox"), DropboxConnector)

    def test_metadata(self) -> None:
        self.assertEqual(DropboxConnector.id, "dropbox")
        methods = [m.value for m in DropboxConnector.auth_methods]
        self.assertIn("api_key", methods)
        self.assertIn("oauth2", methods)


class ConnectTests(unittest.TestCase):
    def test_connect_token_validates_and_stores(self) -> None:
        conn, http = _dropbox()
        http.route("POST", "/2/users/get_current_account",
                   FakeResponse(200, ACCOUNT))
        result = conn.connect(token="long-lived-token")
        self.assertTrue(result.ok)
        self.assertIn("devon@example.com", result.account)
        cred = conn.vault.get("connector:dropbox", ACCOUNT["account_id"])
        self.assertEqual(cred.password, "long-lived-token")
        self.assertEqual((cred.metadata or {}).get("auth"), "api_key")
        _m, url, _p, headers = http.calls[0]
        self.assertIn("api.dropboxapi.com", url)
        self.assertEqual(headers["Authorization"],
                         "Bearer long-lived-token")

    def test_connect_token_rejected(self) -> None:
        conn, http = _dropbox()
        http.route("POST", "/2/users/get_current_account",
                   FakeResponse(401, {"error_summary": "invalid_access_token/"}))
        with self.assertRaises(DropboxError) as ctx:
            conn.connect(token="bad")
        self.assertIn("401", str(ctx.exception))
        self.assertIsNone(conn._load_credential())

    def test_connect_oauth_code(self) -> None:
        conn, http = _dropbox()
        http.route("POST", "oauth2/token", FakeResponse(200, TOKENS))
        http.route("POST", "/2/users/get_current_account",
                   FakeResponse(200, ACCOUNT))
        result = conn.connect(client_id="appkey", client_secret="appsecret",
                              code="authcode")
        self.assertTrue(result.ok)
        cred = conn.vault.get("connector:dropbox", ACCOUNT["account_id"])
        self.assertEqual(cred.password, "refresh123")  # refresh vaulted
        self.assertEqual((cred.metadata or {}).get("auth"), "oauth2")
        self.assertEqual((cred.metadata or {}).get("client_id"), "appkey")

    def test_connect_oauth_bad_code(self) -> None:
        conn, http = _dropbox()
        http.route("POST", "oauth2/token",
                   FakeResponse(400, {"error": "invalid_grant"}))
        with self.assertRaises(DropboxError) as ctx:
            conn.connect(client_id="appkey", client_secret="appsecret",
                         code="stale")
        self.assertIn("rejected the authorization code", str(ctx.exception))
        self.assertIsNone(conn._load_credential())

    def test_connect_neither_prints_guide(self) -> None:
        conn, _http = _dropbox()
        with mock.patch.dict(os.environ, {"DROPBOX_CLIENT_ID": "appkey"}):
            result = conn.connect()
        self.assertFalse(result.ok)
        self.assertIn("oauth2/authorize", result.message)

    def test_connect_rejects_second_credential(self) -> None:
        conn, _http = _connected_token()
        with self.assertRaises(ConnectorError) as ctx:
            conn.connect(token="other")
        self.assertIn("already connected", str(ctx.exception))

    def test_disconnect_clears(self) -> None:
        conn, _http = _connected_token()
        conn.disconnect()
        self.assertIsNone(conn._load_credential())
        conn.disconnect()  # idempotent


class StatusTests(unittest.TestCase):
    def test_status_not_connected(self) -> None:
        conn, _http = _dropbox()
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertFalse(conn.test_connection())

    def test_status_connected(self) -> None:
        conn, http = _connected_token()
        http.route("POST", "/2/users/get_current_account",
                   FakeResponse(200, ACCOUNT))
        st = conn.status()
        self.assertTrue(st.connected)
        self.assertIn("devon@example.com", st.account or "")
        self.assertTrue(conn.test_connection())

    def test_status_rejected_token(self) -> None:
        conn, http = _connected_token()
        http.routes.clear()
        http.route("POST", "/2/users/get_current_account",
                   FakeResponse(401, {"error_summary": "expired_access_token/"}))
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertIn("rejected", st.detail)
        self.assertFalse(conn.test_connection())


class OAuthRefreshTests(unittest.TestCase):
    def test_expired_access_token_refreshes(self) -> None:
        conn, http = _connected_oauth()
        # force the cached access token to look expired
        cred = conn._load_credential()
        assert cred is not None
        meta = dict(cred.metadata or {})
        meta["access_expires_at"] = 1.0
        conn.vault.store(
            service="connector:dropbox", username=cred.username,
            password=cred.password, credential_type="oauth_token",
            tags=["connector", "dropbox"], metadata=meta,
        )
        http.routes.clear()
        http.route("POST", "oauth2/token", FakeResponse(200, {
            "access_token": "sl.new", "expires_in": 14400,
        }))
        http.route("POST", "/2/files/list_folder",
                   FakeResponse(200, FOLDER))
        with mock.patch.dict(os.environ,
                             {"DROPBOX_CLIENT_SECRET": "appsecret"}):
            folders = conn.list_folder()
        self.assertEqual(len(folders["entries"]), 2)
        # the rotated credential kept the old refresh token
        cred2 = conn._load_credential()
        assert cred2 is not None
        self.assertEqual(cred2.password, "refresh123")
        self.assertEqual((cred2.metadata or {}).get("access_token"), "sl.new")


class ApiTests(unittest.TestCase):
    def test_get_current_account(self) -> None:
        conn, http = _connected_token()
        http.route("POST", "/2/users/get_current_account",
                   FakeResponse(200, ACCOUNT))
        account = conn.get_current_account()
        self.assertEqual(account["email"], "devon@example.com")

    def test_list_folder(self) -> None:
        conn, http = _connected_token()
        http.route("POST", "/2/files/list_folder",
                   FakeResponse(200, FOLDER))
        result = conn.list_folder("/docs")
        self.assertEqual(len(result["entries"]), 2)
        self.assertEqual(result["entries"][0]["tag"], "folder")
        self.assertEqual(result["entries"][1]["size"], 11)
        self.assertEqual(result["cursor"], "cursor1")
        self.assertFalse(result["has_more"])
        _m, _u, payload, _h = http.calls[-1]
        self.assertEqual(payload["path"], "/docs")

    def test_list_folder_continue(self) -> None:
        conn, http = _connected_token()
        http.route("POST", "/2/files/list_folder/continue",
                   FakeResponse(200, {"entries": [], "cursor": "c2",
                                      "has_more": False}))
        result = conn.list_folder(cursor="c1")
        self.assertEqual(result["entries"], [])

    def test_get_metadata_empty_path(self) -> None:
        conn, _http = _connected_token()
        with self.assertRaises(ConnectorError):
            conn.get_metadata("")

    def test_download_bytes(self) -> None:
        conn, http = _connected_token()
        http.route("POST", "/2/files/download",
                   FakeResponse(200, None, body=b"file-bytes",
                                headers={"dropbox-api-result":
                                         json.dumps(UPLOAD_RESULT)}))
        result = conn.download("/up.txt")
        self.assertEqual(result["data"], b"file-bytes")
        self.assertEqual(result["metadata"]["name"], "up.txt")
        _m, url, _p, headers = http.calls[-1]
        self.assertIn("content.dropboxapi.com", url)
        arg = json.loads(headers["Dropbox-API-Arg"])
        self.assertEqual(arg["path"], "/up.txt")

    def test_download_to_file(self) -> None:
        import tempfile
        conn, http = _connected_token()
        http.route("POST", "/2/files/download",
                   FakeResponse(200, None, body=b"file-bytes",
                                headers={"dropbox-api-result": "{}"}))
        with tempfile.TemporaryDirectory() as tmp:
            dest = f"{tmp}/dl.txt"
            result = conn.download("/up.txt", dest=dest)
            with open(dest, "rb") as fh:
                self.assertEqual(fh.read(), b"file-bytes")
            self.assertEqual(result["dest"], dest)

    def test_download_conflict(self) -> None:
        conn, http = _connected_token()
        http.route("POST", "/2/files/download",
                   FakeResponse(409, {"error_summary": "path/not_found/"}))
        with self.assertRaises(DropboxError) as ctx:
            conn.download("/missing.txt")
        self.assertIn("not_found", str(ctx.exception))

    def test_download_empty_path(self) -> None:
        conn, _http = _connected_token()
        with self.assertRaises(ConnectorError):
            conn.download("")

    def test_upload_confirmed(self) -> None:
        import tempfile
        conn, http = _connected_token()
        http.route("POST", "/2/files/upload",
                   FakeResponse(200, None,
                                headers={"dropbox-api-result":
                                         json.dumps(UPLOAD_RESULT)}))
        with tempfile.TemporaryDirectory() as tmp:
            src = f"{tmp}/up.txt"
            with open(src, "w") as fh:
                fh.write("payload")
            result = conn.upload(src, "/up.txt", confirmed=True)
        self.assertEqual(result["name"], "up.txt")
        self.assertEqual(result["size"], 7)
        _m, url, data, headers = http.calls[-1]
        self.assertIn("content.dropboxapi.com", url)
        self.assertEqual(data, b"payload")
        arg = json.loads(headers["Dropbox-API-Arg"])
        self.assertEqual(arg["path"], "/up.txt")
        self.assertEqual(headers["Content-Type"], "application/octet-stream")

    def test_upload_needs_confirmation(self) -> None:
        import tempfile
        conn, http = _connected_token()
        with tempfile.TemporaryDirectory() as tmp:
            src = f"{tmp}/up.txt"
            with open(src, "w") as fh:
                fh.write("payload")
            with self.assertRaises(ConnectorError) as ctx:
                conn.upload(src, "/up.txt")
        self.assertIn("confirmation", str(ctx.exception))
        self.assertEqual(
            [c for c in http.calls if "files/upload" in c[1]], [])

    def test_upload_missing_file(self) -> None:
        conn, _http = _connected_token()
        with self.assertRaises(ConnectorError):
            conn.upload("/no/such/file", "/x.txt", confirmed=True)

    def test_upload_bad_path(self) -> None:
        import tempfile
        conn, _http = _connected_token()
        with tempfile.TemporaryDirectory() as tmp:
            src = f"{tmp}/up.txt"
            with open(src, "w") as fh:
                fh.write("x")
            with self.assertRaises(ConnectorError):
                conn.upload(src, "no-leading-slash", confirmed=True)

    def test_upload_bad_mode(self) -> None:
        import tempfile
        conn, _http = _connected_token()
        with tempfile.TemporaryDirectory() as tmp:
            src = f"{tmp}/up.txt"
            with open(src, "w") as fh:
                fh.write("x")
            with self.assertRaises(ConnectorError):
                conn.upload(src, "/up.txt", mode="update", confirmed=True)

    def test_delete_confirmed(self) -> None:
        conn, http = _connected_token()
        http.route("POST", "/2/files/delete_v2",
                   FakeResponse(200, {"metadata": {"name": "old.txt"}}))
        result = conn.delete("/old.txt", confirmed=True)
        self.assertEqual(result["deleted"], "/old.txt")
        _m, _u, payload, _h = http.calls[-1]
        self.assertEqual(payload["path"], "/old.txt")

    def test_delete_needs_confirmation(self) -> None:
        conn, http = _connected_token()
        with self.assertRaises(ConnectorError) as ctx:
            conn.delete("/old.txt")
        self.assertIn("confirmation", str(ctx.exception))
        self.assertEqual(
            [c for c in http.calls if "delete_v2" in c[1]], [])

    def test_delete_empty_path(self) -> None:
        conn, _http = _connected_token()
        with self.assertRaises(ConnectorError):
            conn.delete("", confirmed=True)

    def test_rate_limit(self) -> None:
        conn, http = _connected_token()
        http.route("POST", "/2/files/list_folder",
                   FakeResponse(429, {"error_summary": "too_many_requests/"},
                                headers={"retry-after": "5"}))
        with self.assertRaises(DropboxError) as ctx:
            conn.list_folder()
        self.assertEqual(ctx.exception.status_code, 429)
        self.assertIn("rate limit", str(ctx.exception))

    def test_invalid_json(self) -> None:
        conn, http = _connected_token()
        resp = FakeResponse(200, None)
        resp.json = mock.Mock(side_effect=ValueError("no json"))  # type: ignore[method-assign]
        http.route("POST", "/2/files/list_folder", resp)
        with self.assertRaises(DropboxError) as ctx:
            conn.list_folder()
        self.assertIn("invalid JSON", str(ctx.exception))

    def test_not_connected_raises(self) -> None:
        conn, _http = _dropbox()
        with self.assertRaises(ConnectorError) as ctx:
            conn.list_folder()
        self.assertIn("not connected", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
