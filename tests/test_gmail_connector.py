"""Gmail connector tests. HTTP is fully mocked — no network."""

from __future__ import annotations

import base64
import json
import os
import unittest
import urllib.parse
from typing import Any
from unittest import mock

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors.base import ConnectorError
from nomorals.connectors.checkpoints import (
    CheckpointKind,
    CheckpointStore,
    HumanCheckpointPending,
)
from nomorals.connectors.gmail import GmailConnector, GmailError
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
        return self._dispatch(method, url, None, **kw)


TOKENS = {
    "access_token": "ya29.acc",
    "refresh_token": "1//ref",
    "expires_in": 3600,
    "token_type": "Bearer",
}
PROFILE = {"emailAddress": "owner@example.com", "messagesTotal": 42}


def _gmail(http: FakeHttp | None = None) -> tuple[GmailConnector, FakeHttp]:
    http = http or FakeHttp()
    return GmailConnector(_vault(), http=http), http


def _connected(
    http: FakeHttp | None = None,
) -> tuple[GmailConnector, FakeHttp]:
    conn, http = _gmail(http)
    http.route("POST", "oauth2.googleapis.com/token",
               FakeResponse(200, dict(TOKENS)))
    http.route("GET", "/users/me/profile", FakeResponse(200, dict(PROFILE)))
    result = conn.connect(
        client_id="cid123", client_secret="s3cr3t", code="auth-code-1"
    )
    assert result.ok
    return conn, http


class RegistryTests(unittest.TestCase):
    def test_registered(self) -> None:
        self.assertIs(get_connector("gmail"), GmailConnector)

    def test_metadata(self) -> None:
        self.assertEqual(GmailConnector.id, "gmail")
        self.assertIn("oauth2", [m.value for m in GmailConnector.auth_methods])


class ConnectTests(unittest.TestCase):
    def test_connect_with_code_vaults_refresh_token(self) -> None:
        conn, http = _connected()
        cred = conn.vault.get("connector:gmail", "owner@example.com")
        self.assertEqual(cred.password, "1//ref")
        self.assertEqual(cred.credential_type, "oauth_token")
        # access token rides in metadata, never as the vaulted secret
        self.assertEqual(cred.metadata["access_token"], "ya29.acc")
        # the client secret went to the token endpoint, never a URL
        for method, url, payload, _h in http.calls:
            self.assertNotIn("s3cr3t", url)
        form_calls = [c for c in http.calls if "oauth2.googleapis.com" in c[1]]
        self.assertTrue(form_calls)
        self.assertEqual(form_calls[0][2]["grant_type"], "authorization_code")

    def test_connect_rejects_second_account(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.connect(client_id="x", client_secret="y", code="z")
        self.assertIn("already connected", str(ctx.exception))

    def test_connect_without_code_prints_guide(self) -> None:
        conn, _http = _gmail()
        with mock.patch("builtins.print") as fake_print:
            result = conn.connect(client_id="cid123", client_secret="s3cr3t")
        self.assertFalse(result.ok)
        self.assertIn("code", result.message)
        printed = " ".join(c.args[0] for c in fake_print.call_args_list)
        self.assertIn("accounts.google.com", printed)

    def test_connect_without_code_and_db_parks_checkpoint(self) -> None:
        conn, _http = _gmail()
        db = _db()
        with mock.patch("builtins.print"):
            result = conn.connect(
                client_id="cid123", client_secret="s3cr3t", db=db
            )
        self.assertFalse(result.ok)
        self.assertIn("checkpoint", result.message)
        store = CheckpointStore(db)
        pending = store.list_pending("gmail")
        self.assertEqual(len(pending), 1)

    def test_resume_checkpoint_completes_oauth(self) -> None:
        conn, http = _gmail()
        db = _db()
        with mock.patch("builtins.print"):
            conn.connect(client_id="cid123", client_secret="s3cr3t", db=db)
        store = CheckpointStore(db)
        cp = store.list_pending("gmail")[0]
        http.route("POST", "oauth2.googleapis.com/token",
                   FakeResponse(200, dict(TOKENS)))
        http.route("GET", "/users/me/profile",
                   FakeResponse(200, dict(PROFILE)))
        store.resolve(cp.id, note="code=auth-code-9")
        result = conn.resume_checkpoint(
            store.get(cp.id), db=db, client_secret="s3cr3t"
        )
        self.assertTrue(result["connected"])
        self.assertEqual(result["account"], "owner@example.com")

    def test_resume_unresolved_checkpoint_raises(self) -> None:
        conn, _http = _gmail()
        db = _db()
        store = CheckpointStore(db)
        cp = store.create("gmail", CheckpointKind.MANUAL_STEP, "t", "i",
                          resume_state={"stage": "oauth_code"})
        with self.assertRaises(ConnectorError) as ctx:
            conn.resume_checkpoint(cp, db=db)
        self.assertIn("not resolved", str(ctx.exception))

    def test_bad_code_fails_fast(self) -> None:
        conn, http = _gmail()
        http.route("POST", "oauth2.googleapis.com/token", FakeResponse(
            400, {"error": "invalid_grant",
                  "error_description": "bad code"}))
        with self.assertRaises(ConnectorError) as ctx:
            conn.connect(client_id="c", client_secret="s", code="stale")
        self.assertIn("rejected the authorization code", str(ctx.exception))
        self.assertIsNone(conn._load_credential())

    def test_disconnect_clears(self) -> None:
        conn, _http = _connected()
        conn.disconnect()
        self.assertIsNone(conn._load_credential())
        conn.disconnect()


class StatusTests(unittest.TestCase):
    def test_status_not_connected(self) -> None:
        conn, _http = _gmail()
        self.assertFalse(conn.status().connected)
        self.assertFalse(conn.test_connection())

    def test_status_connected(self) -> None:
        conn, http = _connected()
        http.route("GET", "/users/me/profile",
                   FakeResponse(200, dict(PROFILE)))
        st = conn.status()
        self.assertTrue(st.connected)
        self.assertEqual(st.account, "owner@example.com")
        self.assertTrue(conn.test_connection())

    def test_revoked_token_status(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/users/me/profile", FakeResponse(
            401, {"error": {"message": "Invalid Credentials"}}))
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertIn("rejected", st.detail)


class ReadTests(unittest.TestCase):
    def test_list_messages(self) -> None:
        conn, http = _connected()
        http.route("GET", "/users/me/messages", FakeResponse(200, {
            "messages": [{"id": "m1", "threadId": "t1"}],
            "nextPageToken": "tok2", "resultSizeEstimate": 1,
        }))
        result = conn.list_messages(query="from:boss is:unread",
                                   max_results=10)
        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(result["next_page_token"], "tok2")
        _m, url, _p, headers = http.calls[-1]
        self.assertIn("q=from", url)
        self.assertEqual(headers["Authorization"], "Bearer ya29.acc")

    def test_get_message(self) -> None:
        conn, http = _connected()
        http.route("GET", "/users/me/messages/m1", FakeResponse(200, {
            "id": "m1", "snippet": "hello there",
        }))
        msg = conn.get_message("m1", format="metadata")
        self.assertEqual(msg["snippet"], "hello there")
        _m, url, _p, _h = http.calls[-1]
        self.assertIn("format=metadata", url)

    def test_get_message_bad_format_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.get_message("m1", format="xml")

    def test_get_thread(self) -> None:
        conn, http = _connected()
        http.route("GET", "/users/me/threads/t1", FakeResponse(200, {
            "id": "t1", "messages": [{"id": "m1"}],
        }))
        thread = conn.get_thread("t1")
        self.assertEqual(len(thread["messages"]), 1)

    def test_list_labels(self) -> None:
        conn, http = _connected()
        http.route("GET", "/users/me/labels", FakeResponse(200, {
            "labels": [{"id": "INBOX", "name": "INBOX"}],
        }))
        labels = conn.list_labels()
        self.assertEqual(labels[0]["id"], "INBOX")

    def test_insufficient_scope_error(self) -> None:
        conn, http = _connected()
        http.route("GET", "/users/me/messages", FakeResponse(403, {
            "error": {"errors": [{"reason": "insufficientPermissions"}],
                      "message": "Insufficient Permission"},
        }))
        with self.assertRaises(GmailError) as ctx:
            conn.list_messages()
        self.assertIn("scope", str(ctx.exception))
        self.assertEqual(ctx.exception.reason, "insufficientPermissions")

    def test_not_connected_raises(self) -> None:
        conn, _http = _gmail()
        with self.assertRaises(ConnectorError) as ctx:
            conn.list_messages()
        self.assertIn("not connected", str(ctx.exception))


class SendTests(unittest.TestCase):
    def test_send_requires_confirmation(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.send_message("a@b.c", "subj", "body")
        self.assertIn("confirmed=True", str(ctx.exception))

    def test_send_refuses_empty_body(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.send_message("a@b.c", "subj", "", confirmed=True)

    def test_send_confirmed(self) -> None:
        conn, http = _connected()
        http.route("POST", "/users/me/messages/send", FakeResponse(200, {
            "id": "sent1", "threadId": "t9", "labelIds": ["SENT"],
        }))
        result = conn.send_message(
            "a@b.c", "Hello", "body text", confirmed=True
        )
        self.assertEqual(result["id"], "sent1")
        _m, url, payload, _h = http.calls[-1]
        self.assertIn("/users/me/messages/send", url)
        raw = base64.urlsafe_b64decode(payload["raw"]).decode("utf-8")
        self.assertIn("To: a@b.c", raw)
        self.assertIn("Subject: Hello", raw)
        self.assertIn("body text", raw)

    def test_send_via_human_checkpoint(self) -> None:
        conn, http = _connected()
        db = _db()
        with self.assertRaises(HumanCheckpointPending) as ctx:
            conn.send_message("a@b.c", "Hello", "body text", db=db)
        cp_id = ctx.exception.checkpoint.id
        store = CheckpointStore(db)
        cp = store.get(cp_id)
        self.assertIn("a@b.c", cp.instructions)
        self.assertIn("body text", cp.instructions)
        http.route("POST", "/users/me/messages/send", FakeResponse(200, {
            "id": "sent2", "threadId": "t9",
        }))
        store.resolve(cp_id, note="approved")
        result = conn.resume_checkpoint(store.get(cp_id), db=db)
        self.assertEqual(result["id"], "sent2")

    def test_send_checkpoint_unresolved_raises(self) -> None:
        conn, _http = _connected()
        db = _db()
        with self.assertRaises(HumanCheckpointPending) as ctx:
            conn.send_message("a@b.c", "Hello", "body", db=db)
        store = CheckpointStore(db)
        with self.assertRaises(ConnectorError):
            conn.resume_checkpoint(
                store.get(ctx.exception.checkpoint.id), db=db
            )


class RefreshTests(unittest.TestCase):
    def test_expired_access_token_refreshes(self) -> None:
        conn, http = _connected()
        # expire the cached access token
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
            200, {"access_token": "ya29.new", "expires_in": 3600,
                  "token_type": "Bearer"}))
        http.route("GET", "/users/me/messages", FakeResponse(200, {
            "messages": [], "resultSizeEstimate": 0,
        }))
        with mock.patch.dict(os.environ, {"GOOGLE_CLIENT_SECRET": "s3cr3t"}):
            conn.list_messages()
        _m, _u, _p, headers = http.calls[-1]
        self.assertEqual(headers["Authorization"], "Bearer ya29.new")
        # refresh token preserved across rotation
        self.assertEqual(
            conn._load_credential().password, "1//ref"
        )


if __name__ == "__main__":
    unittest.main()
