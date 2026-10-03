"""Google Drive connector tests. HTTP is fully mocked — no network."""

from __future__ import annotations

import json
import os
import unittest
import urllib.parse
from pathlib import Path
from typing import Any
from unittest import mock

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors.base import ConnectorError
from nomorals.connectors.checkpoints import CheckpointStore
from nomorals.connectors.drive import DriveConnector, DriveError
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
        if self._payload is None:
            raise ValueError("no JSON here")
        return self._payload


class FakeHttp:
    """Scripted stand-in for HttpClient. No network.

    Honors ``stream_to`` (writes canned bytes) and records raw bodies so
    multipart uploads can be asserted on.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any, Any]] = []
        self.routes: list[tuple[str, str, FakeResponse]] = []
        self.download_bytes = b"file-content-bytes"

    def route(self, method: str, path: str, response: FakeResponse) -> None:
        self.routes.append((method.upper(), path, response))

    def _dispatch(
        self, method: str, url: str, payload: Any = None, **kw: Any
    ) -> FakeResponse:
        stream_to = kw.get("stream_to")
        self.calls.append(
            (method.upper(), url, payload, kw.get("headers"), stream_to)
        )
        if stream_to:
            Path(stream_to).write_bytes(self.download_bytes)
        for rm, rp, resp in sorted(self.routes, key=lambda r: -len(r[1])):
            if rm == method.upper() and rp in url:
                return resp
        return FakeResponse(404, {"error": {"message": "not mocked"}})

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


TOKENS = {
    "access_token": "ya29.drive",
    "refresh_token": "1//refd",
    "expires_in": 3600,
    "token_type": "Bearer",
}
ABOUT = {"user": {"emailAddress": "owner@example.com"}}


def _drive(http: FakeHttp | None = None) -> tuple[DriveConnector, FakeHttp]:
    http = http or FakeHttp()
    return DriveConnector(_vault(), http=http), http


def _connected(http: FakeHttp | None = None) -> tuple[DriveConnector, FakeHttp]:
    conn, http = _drive(http)
    http.route("POST", "oauth2.googleapis.com/token",
               FakeResponse(200, dict(TOKENS)))
    http.route("GET", "/drive/v3/about", FakeResponse(200, dict(ABOUT)))
    result = conn.connect(
        client_id="cid123", client_secret="s3cr3t", code="auth-code-1"
    )
    assert result.ok
    return conn, http


class RegistryTests(unittest.TestCase):
    def test_registered(self) -> None:
        self.assertIs(get_connector("gdrive"), DriveConnector)

    def test_metadata(self) -> None:
        self.assertEqual(DriveConnector.id, "gdrive")
        self.assertIn("oauth2", [m.value for m in DriveConnector.auth_methods])


class ConnectTests(unittest.TestCase):
    def test_connect_with_code_vaults_refresh_token(self) -> None:
        conn, _http = _connected()
        self.assertIn("owner@example.com", conn.status().account or "")
        cred = conn.vault.get("connector:gdrive", "owner@example.com")
        self.assertEqual(cred.password, "1//refd")
        self.assertIn("drive.file", cred.metadata["scopes"][0])

    def test_connect_custom_scopes(self) -> None:
        conn, http = _drive()
        http.route("POST", "oauth2.googleapis.com/token",
                   FakeResponse(200, dict(TOKENS)))
        http.route("GET", "/drive/v3/about", FakeResponse(200, dict(ABOUT)))
        with mock.patch("builtins.print"):
            result = conn.connect(
                client_id="c", client_secret="s", code="code",
                scopes=["https://www.googleapis.com/auth/drive"],
            )
        self.assertTrue(result.ok)
        self.assertIn("auth/drive", result.scopes[0])

    def test_connect_rejects_second_account(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.connect(client_id="x", client_secret="y", code="z")
        self.assertIn("already connected", str(ctx.exception))

    def test_connect_without_code_prints_guide(self) -> None:
        conn, _http = _drive()
        with mock.patch("builtins.print") as fake_print:
            result = conn.connect(client_id="cid123", client_secret="s3cr3t")
        self.assertFalse(result.ok)
        printed = " ".join(c.args[0] for c in fake_print.call_args_list)
        self.assertIn("accounts.google.com", printed)

    def test_resume_checkpoint_completes_oauth(self) -> None:
        conn, http = _drive()
        db = _db()
        with mock.patch("builtins.print"):
            conn.connect(client_id="cid123", client_secret="s3cr3t", db=db)
        store = CheckpointStore(db)
        cp = store.list_pending("gdrive")[0]
        http.route("POST", "oauth2.googleapis.com/token",
                   FakeResponse(200, dict(TOKENS)))
        http.route("GET", "/drive/v3/about", FakeResponse(200, dict(ABOUT)))
        store.resolve(cp.id, note="code=auth-code-7")
        result = conn.resume_checkpoint(
            store.get(cp.id), db=db, client_secret="s3cr3t"
        )
        self.assertTrue(result["connected"])
        self.assertEqual(result["account"], "owner@example.com")

    def test_disconnect_clears(self) -> None:
        conn, _http = _connected()
        conn.disconnect()
        self.assertIsNone(conn._load_credential())

    def test_status_and_test_connection(self) -> None:
        conn, http = _connected()
        http.route("GET", "/drive/v3/about", FakeResponse(200, dict(ABOUT)))
        st = conn.status()
        self.assertTrue(st.connected)
        self.assertEqual(st.account, "owner@example.com")
        self.assertTrue(conn.test_connection())
        conn2, _h2 = _drive()
        self.assertFalse(conn2.status().connected)
        self.assertFalse(conn2.test_connection())


class FileTests(unittest.TestCase):
    def test_list_files(self) -> None:
        conn, http = _connected()
        http.route("GET", "/drive/v3/files", FakeResponse(200, {
            "files": [{"id": "f1", "name": "a.txt"}],
            "nextPageToken": "n2",
        }))
        result = conn.list_files(query="name contains 'a'")
        self.assertEqual(len(result["files"]), 1)
        self.assertEqual(result["next_page_token"], "n2")
        _m, url, _p, headers, _s = http.calls[-1]
        self.assertIn("q=name", url)
        self.assertEqual(headers["Authorization"], "Bearer ya29.drive")

    def test_get_file(self) -> None:
        conn, http = _connected()
        http.route("GET", "/drive/v3/files/f1", FakeResponse(200, {
            "id": "f1", "name": "a.txt", "mimeType": "text/plain",
        }))
        meta = conn.get_file("f1")
        self.assertEqual(meta["name"], "a.txt")

    def test_get_file_empty_id_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.get_file("")

    def test_upload_builds_multipart(self) -> None:
        conn, http = _connected()
        http.route("POST", "/upload/drive/v3/files", FakeResponse(200, {
            "id": "new1", "name": "report.txt",
        }))
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "report.txt"
            src.write_text("hello drive")
            meta = conn.upload(str(src), folder_id="fld9")
        self.assertEqual(meta["id"], "new1")
        _m, url, body, headers, _s = http.calls[-1]
        self.assertIn("uploadType=multipart", url)
        self.assertIn("multipart/related", headers["Content-Type"])
        self.assertIn(b"hello drive", body)
        self.assertIn(b"application/json", body)
        self.assertIn(b'"name": "report.txt"', body)
        self.assertIn(b'"parents": ["fld9"]', body)

    def test_upload_missing_file_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.upload("/tmp/definitely-not-here-xyz.txt")
        self.assertIn("not a file", str(ctx.exception))

    def test_download_streams_to_disk(self) -> None:
        conn, http = _connected()
        http.download_bytes = b"binary-payload"
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            dest = str(Path(tmp) / "out.bin")
            result = conn.download("f1", dest)
            self.assertEqual(Path(dest).read_bytes(), b"binary-payload")
        self.assertEqual(result["bytes"], len(b"binary-payload"))
        _m, url, _p, _h, stream_to = http.calls[-1]
        self.assertIn("alt=media", url)
        self.assertTrue(str(stream_to).endswith("out.bin"))

    def test_export_file(self) -> None:
        conn, http = _connected()
        http.route("GET", "/drive/v3/files/doc1/export",
                   FakeResponse(200, {}))
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            dest = str(Path(tmp) / "doc.pdf")
            result = conn.export_file(
                "doc1", dest, mime_type="application/pdf"
            )
            self.assertTrue(Path(dest).exists())
        self.assertEqual(result["file_id"], "doc1")
        _m, url, _p, _h, _s = http.calls[-1]
        self.assertIn("mimeType=application%2Fpdf", url)

    def test_export_needs_mime_type(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.export_file("doc1", "/tmp/x.pdf", mime_type="")

    def test_delete(self) -> None:
        conn, http = _connected()
        http.route("DELETE", "/drive/v3/files/f1", FakeResponse(204, None))
        self.assertTrue(conn.delete("f1"))
        _m, url, _p, _h, _s = http.calls[-1]
        self.assertIn("/drive/v3/files/f1", url)

    def test_trash(self) -> None:
        conn, http = _connected()
        http.route("PATCH", "/drive/v3/files/f1", FakeResponse(200, {
            "id": "f1", "trashed": True,
        }))
        result = conn.trash("f1")
        self.assertTrue(result["trashed"])
        _m, _u, body, headers, _s = http.calls[-1]
        self.assertEqual(headers["Content-Type"], "application/json")

    def test_not_found_error(self) -> None:
        conn, http = _connected()
        http.route("GET", "/drive/v3/files/nope", FakeResponse(404, {
            "error": {"message": "File not found"},
        }))
        with self.assertRaises(DriveError) as ctx:
            conn.get_file("nope")
        self.assertEqual(ctx.exception.status_code, 404)

    def test_not_connected_raises(self) -> None:
        conn, _http = _drive()
        with self.assertRaises(ConnectorError) as ctx:
            conn.list_files()
        self.assertIn("not connected", str(ctx.exception))


class RefreshTests(unittest.TestCase):
    def test_expired_access_token_refreshes(self) -> None:
        conn, http = _connected()
        cred = conn._load_credential()
        conn._store_google_tokens(
            cred.username, "cid123",
            {"refresh_token": cred.password, "access_token": "old",
             "expires_in": -10},
            account="owner@example.com",
            scopes=["s"],
        )
        http.routes.clear()  # drop the connect() token route
        http.route("POST", "oauth2.googleapis.com/token", FakeResponse(
            200, {"access_token": "ya29.fresh", "expires_in": 3600,
                  "token_type": "Bearer"}))
        http.route("GET", "/drive/v3/files", FakeResponse(200, {"files": []}))
        with mock.patch.dict(os.environ, {"GOOGLE_CLIENT_SECRET": "s3cr3t"}):
            conn.list_files()
        _m, _u, _p, headers, _s = http.calls[-1]
        self.assertEqual(headers["Authorization"], "Bearer ya29.fresh")


if __name__ == "__main__":
    unittest.main()
