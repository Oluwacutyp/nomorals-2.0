"""Google Calendar connector tests. HTTP is fully mocked — no network."""

from __future__ import annotations

import io
import json
import os
import unittest
import urllib.parse
from contextlib import redirect_stdout
from typing import Any
from unittest import mock

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors.base import ConnectorError
from nomorals.connectors.checkpoints import (
    CheckpointStore,
    HumanCheckpointPending,
)
from nomorals.connectors.gcalendar import GCalendarConnector, GCalendarError
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
        params = kw.pop("params", None) or {}
        if params:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{urllib.parse.urlencode(params)}"
        return self._dispatch("POST", url, payload, **kw)

    def put_json(self, url: str, payload: Any, **kw: Any) -> FakeResponse:
        params = kw.pop("params", None) or {}
        if params:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{urllib.parse.urlencode(params)}"
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


TOKENS = {
    "access_token": "ya29.acc",
    "refresh_token": "1//ref",
    "expires_in": 3600,
    "token_type": "Bearer",
}
PRIMARY = {"id": "primary", "summary": "owner@example.com"}


def _gcal(http: FakeHttp | None = None) -> tuple[GCalendarConnector, FakeHttp]:
    http = http or FakeHttp()
    return GCalendarConnector(_vault(), http=http), http


def _connected(
    http: FakeHttp | None = None,
) -> tuple[GCalendarConnector, FakeHttp]:
    conn, http = _gcal(http)
    http.route("POST", "oauth2.googleapis.com/token",
               FakeResponse(200, dict(TOKENS)))
    http.route("GET", "/calendar/v3/calendars/primary",
               FakeResponse(200, dict(PRIMARY)))
    with mock.patch.dict(os.environ, {"GOOGLE_CLIENT_SECRET": "shhh"}):
        result = conn.connect(client_id="cid123", code="authcode")
    assert result.ok
    return conn, http


class RegistryTests(unittest.TestCase):
    def test_registered(self) -> None:
        self.assertIs(get_connector("gcalendar"), GCalendarConnector)

    def test_metadata(self) -> None:
        self.assertEqual(GCalendarConnector.id, "gcalendar")
        self.assertIn("oauth2", [m.value for m in GCalendarConnector.auth_methods])


class ConnectTests(unittest.TestCase):
    def test_connect_without_code_prints_guide(self) -> None:
        conn, _http = _gcal()
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = conn.connect(client_id="cid123")
        self.assertFalse(result.ok)
        self.assertIn("accounts.google.com", buf.getvalue())
        self.assertIn("calendar", result.scopes[0])
        self.assertIsNone(conn._load_credential())

    def test_connect_with_code_vaults_tokens(self) -> None:
        conn, http = _gcal()
        http.route("POST", "oauth2.googleapis.com/token",
                   FakeResponse(200, dict(TOKENS)))
        http.route("GET", "/calendar/v3/calendars/primary",
                   FakeResponse(200, dict(PRIMARY)))
        with mock.patch.dict(os.environ, {"GOOGLE_CLIENT_SECRET": "shhh"}):
            result = conn.connect(client_id="cid123", code="authcode")
        self.assertTrue(result.ok)
        self.assertEqual(result.account, "owner@example.com")
        cred = conn.vault.get("connector:gcalendar", "owner@example.com")
        self.assertEqual(cred.password, "1//ref")
        self.assertEqual(cred.credential_type, "oauth_token")
        self.assertEqual(cred.metadata["access_token"], "ya29.acc")
        # client secret only went to the token form, never a URL
        for _m, url, _p, _h in http.calls:
            self.assertNotIn("shhh", url)
        form_calls = [c for c in http.calls
                      if "oauth2.googleapis.com" in c[1]]
        self.assertEqual(
            form_calls[0][2]["grant_type"], "authorization_code"
        )

    def test_connect_rejects_second_account(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.connect(client_id="cid123", code="other")
        self.assertIn("already connected", str(ctx.exception))

    def test_disconnect_clears(self) -> None:
        conn, _http = _connected()
        conn.disconnect()
        self.assertIsNone(conn._load_credential())
        conn.disconnect()  # idempotent


class StatusTests(unittest.TestCase):
    def test_status_not_connected(self) -> None:
        conn, _http = _gcal()
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertFalse(conn.test_connection())

    def test_status_connected(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/calendar/v3/calendars/primary",
                   FakeResponse(200, dict(PRIMARY)))
        st = conn.status()
        self.assertTrue(st.connected)
        self.assertEqual(st.account, "owner@example.com")
        self.assertTrue(conn.test_connection())

    def test_status_rejected_token(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/calendar/v3/calendars/primary",
                   FakeResponse(401, {}))
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertIn("rejected", st.detail)
        self.assertFalse(conn.test_connection())


class ApiTests(unittest.TestCase):
    START = {"dateTime": "2026-10-04T14:00:00+01:00"}
    END = {"dateTime": "2026-10-04T15:00:00+01:00"}

    def test_list_calendars(self) -> None:
        conn, http = _connected()
        http.route("GET", "/users/me/calendarList", FakeResponse(200, {
            "items": [{"id": "primary", "summary": "owner@example.com"}],
        }))
        calendars = conn.list_calendars()
        self.assertEqual(calendars[0]["id"], "primary")

    def test_list_events(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/calendars/primary/events", FakeResponse(200, {
            "items": [{"id": "e1", "summary": "Standup"}],
            "timeZone": "Europe/Lagos",
        }))
        result = conn.list_events(
            time_min="2026-10-04T00:00:00+01:00",
            time_max="2026-10-05T00:00:00+01:00",
            query="standup",
        )
        self.assertEqual(result["events"][0]["summary"], "Standup")
        _m, url, _p, _h = http.calls[-1]
        self.assertIn("/calendars/primary/events", url)
        self.assertIn("timeMin=2026-10-04T00", url)
        self.assertIn("q=standup", url)
        self.assertIn("singleEvents=true", url)

    def test_get_event(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/calendars/primary/events/e1",
                   FakeResponse(200, {"id": "e1", "summary": "Standup"}))
        event = conn.get_event("e1")
        self.assertEqual(event["summary"], "Standup")

    def test_get_event_empty_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.get_event("")

    def test_create_event_confirmed(self) -> None:
        conn, http = _connected()
        http.route("POST", "/calendars/primary/events", FakeResponse(200, {
            "id": "e2", "summary": "Focus block", "htmlLink": "https://x",
        }))
        result = conn.create_event(
            "Focus block", self.START, self.END,
            description="deep work", location="home",
            attendees=["a@example.com"], confirmed=True,
        )
        self.assertEqual(result["id"], "e2")
        _m, url, payload, headers = http.calls[-1]
        self.assertIn("/calendars/primary/events", url)
        self.assertEqual(headers["Authorization"], "Bearer ya29.acc")
        self.assertEqual(payload["summary"], "Focus block")
        self.assertEqual(payload["start"], self.START)
        self.assertEqual(payload["attendees"], [{"email": "a@example.com"}])
        self.assertIn("sendUpdates=all", url)  # guests get notified

    def test_create_event_all_day_no_notify(self) -> None:
        conn, http = _connected()
        http.route("POST", "/calendars/primary/events", FakeResponse(200, {
            "id": "e3",
        }))
        conn.create_event(
            "Holiday", {"date": "2026-12-25"}, {"date": "2026-12-26"},
            confirmed=True,
        )
        _m, url, payload, _h = http.calls[-1]
        self.assertEqual(payload["start"], {"date": "2026-12-25"})
        self.assertNotIn("sendUpdates", url)

    def test_create_event_validates(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.create_event("", self.START, self.END, confirmed=True)
        with self.assertRaises(ConnectorError):
            conn.create_event("x", {"nope": 1}, self.END, confirmed=True)
        with self.assertRaises(ConnectorError):
            conn.create_event("x", self.START, self.END, calendar_id="",
                              confirmed=True)

    def test_create_event_requires_confirmation(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.create_event("Focus", self.START, self.END)
        self.assertIn("confirmed=True", str(ctx.exception))

    def test_create_event_via_human_checkpoint(self) -> None:
        conn, http = _connected()
        db = _db()
        with self.assertRaises(HumanCheckpointPending) as ctx:
            conn.create_event("Focus", self.START, self.END, db=db)
        store = CheckpointStore(db)
        cp = store.get(ctx.exception.checkpoint.id)
        self.assertIn("Focus", cp.instructions)
        http.route("POST", "/calendars/primary/events", FakeResponse(200, {
            "id": "e4",
        }))
        store.resolve(cp.id, note="approved")
        result = conn.resume_checkpoint(store.get(cp.id), db=db)
        self.assertEqual(result["id"], "e4")

    def test_create_event_resume_unresolved_raises(self) -> None:
        conn, _http = _connected()
        db = _db()
        with self.assertRaises(HumanCheckpointPending) as ctx:
            conn.create_event("Focus", self.START, self.END, db=db)
        store = CheckpointStore(db)
        with self.assertRaises(ConnectorError):
            conn.resume_checkpoint(
                store.get(ctx.exception.checkpoint.id), db=db
            )

    def test_update_event_confirmed(self) -> None:
        conn, http = _connected()
        http.route("PATCH", "/calendars/primary/events/e1",
                   FakeResponse(200, {"id": "e1", "location": "Room A"}))
        result = conn.update_event("e1", {"location": "Room A"},
                                   confirmed=True)
        self.assertEqual(result["location"], "Room A")
        _m, url, data, _h = http.calls[-1]
        self.assertEqual(_m, "PATCH")
        self.assertIn("/calendars/primary/events/e1", url)
        body = json.loads(data.decode("utf-8"))
        self.assertEqual(body, {"location": "Room A"})

    def test_update_event_validates(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.update_event("", {"location": "x"}, confirmed=True)
        with self.assertRaises(ConnectorError):
            conn.update_event("e1", {}, confirmed=True)

    def test_update_event_requires_confirmation(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.update_event("e1", {"location": "x"})

    def test_delete_event_confirmed(self) -> None:
        conn, http = _connected()
        http.route("DELETE", "/calendars/primary/events/e1",
                   FakeResponse(204, None))
        result = conn.delete_event("e1", confirmed=True)
        self.assertTrue(result["deleted"])
        _m, url, _p, _h = http.calls[-1]
        self.assertEqual(_m, "DELETE")
        self.assertIn("/calendars/primary/events/e1", url)

    def test_delete_event_requires_confirmation(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.delete_event("e1")
        self.assertIn("confirmed=True", str(ctx.exception))

    def test_delete_event_empty_id_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.delete_event("", confirmed=True)

    def test_not_connected_raises(self) -> None:
        conn, _http = _gcal()
        with self.assertRaises(ConnectorError) as ctx:
            conn.list_events()
        self.assertIn("not connected", str(ctx.exception))

    def test_rate_limit_error(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/calendars/primary/events",
                   FakeResponse(429, {}))
        with self.assertRaises(GCalendarError) as ctx:
            conn.list_events()
        self.assertEqual(ctx.exception.status_code, 429)

    def test_forbidden_names_scope(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("POST", "/calendars/primary/events", FakeResponse(403, {
            "error": {"errors": [{"reason": "insufficientPermissions"}]},
        }))
        with self.assertRaises(GCalendarError) as ctx:
            conn.create_event("x", self.START, self.END, confirmed=True)
        self.assertIn("scope", str(ctx.exception))
        self.assertEqual(ctx.exception.reason, "insufficientPermissions")

    def test_invalid_json_error(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/calendars/primary/events",
                   BadJsonResponse(200, "nope"))
        with self.assertRaises(GCalendarError) as ctx:
            conn.list_events()
        self.assertIn("invalid JSON", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
