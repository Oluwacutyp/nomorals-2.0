"""Twilio connector tests. HTTP is fully mocked — no network."""

from __future__ import annotations

import base64
import json
import unittest
import urllib.parse
from typing import Any
from unittest import mock

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors.base import ConnectorError
from nomorals.connectors.registry import get_connector
from nomorals.connectors.twilio import TwilioConnector, TwilioError
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
        return FakeResponse(404, {"code": 0, "message": "not mocked"})

    def get(self, url: str, **kw: Any) -> FakeResponse:
        params = kw.pop("params", None) or {}
        if params:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{urllib.parse.urlencode(params)}"
        return self._dispatch("GET", url, None, **kw)

    def post_form(self, url: str, form: Any, **kw: Any) -> FakeResponse:
        return self._dispatch("POST", url, form, **kw)

    def post_json(self, url: str, payload: Any, **kw: Any) -> FakeResponse:
        return self._dispatch("POST", url, payload, **kw)

    def request(self, method: str, url: str, **kw: Any) -> FakeResponse:
        return self._dispatch(method, url, None, **kw)


ACCOUNT = {"sid": "AC123", "friendly_name": "Devon Twilio",
           "status": "active"}
NUMBERS = {"incoming_phone_numbers": [
    {"phone_number": "+15550001111", "friendly_name": "main"}]}
MSGS = {"messages": [{"sid": "SM1", "to": "+15550002222",
                      "body": "hi"}]}


def _tw(http: FakeHttp | None = None) -> tuple[TwilioConnector, FakeHttp]:
    http = http or FakeHttp()
    return TwilioConnector(_vault(), http=http), http


def _connected(
    http: FakeHttp | None = None,
) -> tuple[TwilioConnector, FakeHttp]:
    conn, http = _tw(http)
    http.route("GET", "/2010-04-01/Accounts/AC123.json",
               FakeResponse(200, ACCOUNT))
    result = conn.connect(account_sid="AC123", auth_token="tok")
    assert result.ok
    return conn, http


def _basic(sid: str, token: str) -> str:
    return "Basic " + base64.b64encode(f"{sid}:{token}".encode()).decode()


class RegistryTests(unittest.TestCase):
    def test_registered(self) -> None:
        self.assertIs(get_connector("twilio"), TwilioConnector)

    def test_metadata(self) -> None:
        self.assertEqual(TwilioConnector.id, "twilio")
        self.assertIn("basic", [m.value for m in TwilioConnector.auth_methods])


class ConnectTests(unittest.TestCase):
    def test_connect_validates_and_stores(self) -> None:
        conn, http = _tw()
        http.route("GET", "/2010-04-01/Accounts/AC123.json",
                   FakeResponse(200, ACCOUNT))
        result = conn.connect(account_sid="AC123", auth_token="tok")
        self.assertTrue(result.ok)
        self.assertIn("Devon Twilio", result.account)
        cred = conn.vault.get("connector:twilio", "AC123")
        self.assertEqual(cred.username, "AC123")
        self.assertEqual(cred.password, "tok")
        _m, _u, _p, headers = http.calls[0]
        self.assertEqual(headers["Authorization"], _basic("AC123", "tok"))

    def test_connect_rejects_second_account(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.connect(account_sid="AC9", auth_token="x")
        self.assertIn("already connected", str(ctx.exception))

    def test_connect_missing_parts_raise(self) -> None:
        conn, _http = _tw()
        with self.assertRaises(ConnectorError):
            conn.connect(account_sid="AC123", auth_token="")

    def test_connect_rejected_credential_fails_fast(self) -> None:
        conn, http = _tw()
        http.route("GET", "/2010-04-01/Accounts/AC123.json",
                   FakeResponse(401, {
                       "code": 20003, "status": 401,
                       "message": "Authenticate",
                       "more_info": "https://x",
                   }))
        with self.assertRaises(TwilioError) as ctx:
            conn.connect(account_sid="AC123", auth_token="bad")
        self.assertIn("rejected", str(ctx.exception))
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertEqual(ctx.exception.twilio_code, 20003)
        self.assertIsNone(conn._load_credential())

    def test_disconnect_clears(self) -> None:
        conn, _http = _connected()
        conn.disconnect()
        self.assertIsNone(conn._load_credential())
        conn.disconnect()  # idempotent


class StatusTests(unittest.TestCase):
    def test_status_not_connected(self) -> None:
        conn, _http = _tw()
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertFalse(conn.test_connection())

    def test_status_connected(self) -> None:
        conn, http = _connected()
        http.route("GET", "/2010-04-01/Accounts/AC123.json",
                   FakeResponse(200, ACCOUNT))
        st = conn.status()
        self.assertTrue(st.connected)
        self.assertIn("Devon Twilio", st.account or "")
        self.assertTrue(conn.test_connection())

    def test_status_rejected_credential(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/2010-04-01/Accounts/AC123.json",
                   FakeResponse(401, {"message": "bad"}))
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertIn("rejected", st.detail)
        self.assertFalse(conn.test_connection())


class ApiTests(unittest.TestCase):
    def test_get_account(self) -> None:
        conn, http = _connected()
        http.route("GET", "/2010-04-01/Accounts/AC123.json",
                   FakeResponse(200, ACCOUNT))
        account = conn.get_account()
        self.assertEqual(account["status"], "active")

    def test_send_sms(self) -> None:
        conn, http = _connected()
        http.route("GET", "/IncomingPhoneNumbers.json",
                   FakeResponse(200, NUMBERS))
        http.route("POST", "/Messages.json", FakeResponse(201, {
            "sid": "SM9", "status": "queued"}))
        result = conn.send_sms("+15550002222", "hello",
                               confirmed=True)
        self.assertEqual(result["sid"], "SM9")
        _m, _u, form, _h = http.calls[-1]
        self.assertEqual(form["To"], "+15550002222")
        self.assertEqual(form["Body"], "hello")
        self.assertEqual(form["From"], "+15550001111")

    def test_send_sms_explicit_from(self) -> None:
        conn, http = _connected()
        http.route("POST", "/Messages.json", FakeResponse(201, {}))
        conn.send_sms("+15550002222", "hi",
                      from_number="+15550003333", confirmed=True)
        _m, _u, form, _h = http.calls[-1]
        self.assertEqual(form["From"], "+15550003333")

    def test_send_sms_empty_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.send_sms("+1555", "", confirmed=True)

    def test_send_sms_no_to_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.send_sms("", "hi", confirmed=True)

    def test_send_sms_too_long_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.send_sms("+1555", "x" * 1601, confirmed=True)
        self.assertIn("1600", str(ctx.exception))

    def test_send_sms_needs_confirmation(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.send_sms("+1555", "hi")
        self.assertIn("confirmation", str(ctx.exception))

    def test_send_sms_no_owned_number_raises(self) -> None:
        conn, http = _connected()
        http.route("GET", "/IncomingPhoneNumbers.json",
                   FakeResponse(200, {"incoming_phone_numbers": []}))
        with self.assertRaises(ConnectorError) as ctx:
            conn.send_sms("+1555", "hi", confirmed=True)
        self.assertIn("no owned Twilio number", str(ctx.exception))

    def test_send_sms_api_error(self) -> None:
        conn, http = _connected()
        http.route("GET", "/IncomingPhoneNumbers.json",
                   FakeResponse(200, NUMBERS))
        http.route("POST", "/Messages.json", FakeResponse(400, {
            "code": 21211, "status": 400,
            "message": "The 'To' number is not a valid phone number.",
        }))
        with self.assertRaises(TwilioError) as ctx:
            conn.send_sms("bogus", "hi", confirmed=True)
        self.assertEqual(ctx.exception.twilio_code, 21211)

    def test_make_call(self) -> None:
        conn, http = _connected()
        http.route("GET", "/IncomingPhoneNumbers.json",
                   FakeResponse(200, NUMBERS))
        http.route("POST", "/Calls.json", FakeResponse(201, {
            "sid": "CA1", "status": "queued"}))
        result = conn.make_call("+15550002222",
                                "https://example.com/twiml.xml",
                                confirmed=True)
        self.assertEqual(result["sid"], "CA1")
        _m, _u, form, _h = http.calls[-1]
        self.assertEqual(form["Url"], "https://example.com/twiml.xml")

    def test_make_call_no_twiml_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.make_call("+1555", "", confirmed=True)

    def test_make_call_non_https_twiml_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.make_call("+1555", "http://example.com/t.xml",
                           confirmed=True)

    def test_make_call_needs_confirmation(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.make_call("+1555", "https://example.com/t.xml")
        self.assertIn("confirmation", str(ctx.exception))

    def test_list_messages(self) -> None:
        conn, http = _connected()
        http.route("GET", "/Messages.json", FakeResponse(200, MSGS))
        messages = conn.list_messages(limit=5)
        self.assertEqual(messages[0]["sid"], "SM1")
        _m, url, _p, _h = http.calls[-1]
        self.assertIn("PageSize=5", url)

    def test_list_numbers(self) -> None:
        conn, http = _connected()
        http.route("GET", "/IncomingPhoneNumbers.json",
                   FakeResponse(200, NUMBERS))
        numbers = conn.list_numbers()
        self.assertEqual(numbers[0]["phone_number"], "+15550001111")

    def test_not_connected_raises(self) -> None:
        conn, _http = _tw()
        with self.assertRaises(ConnectorError) as ctx:
            conn.list_messages()
        self.assertIn("not connected", str(ctx.exception))

    def test_rate_limit_error(self) -> None:
        conn, http = _connected()
        http.route("GET", "/Messages.json", FakeResponse(429, {
            "message": "throttled"}))
        with self.assertRaises(TwilioError) as ctx:
            conn.list_messages()
        self.assertEqual(ctx.exception.status_code, 429)


if __name__ == "__main__":
    unittest.main()
