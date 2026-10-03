"""Wise connector tests. HTTP is fully mocked — no network."""

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
from nomorals.connectors.wise import WiseConnector, WiseError
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
        return self._dispatch(method, url, None, **kw)


PROFILES = [
    {"id": 111, "type": "personal", "fullName": "Devon Tester"},
    {"id": 222, "type": "business", "name": "Devon Ltd"},
]


def _wise(http: FakeHttp | None = None) -> tuple[WiseConnector, FakeHttp]:
    http = http or FakeHttp()
    return WiseConnector(_vault(), http=http), http


def _connected(http: FakeHttp | None = None) -> tuple[WiseConnector, FakeHttp]:
    conn, http = _wise(http)
    http.route("GET", "/v1/profiles", FakeResponse(200, PROFILES))
    result = conn.connect(token="tok_live_123")
    assert result.ok
    return conn, http


class RegistryTests(unittest.TestCase):
    def test_registered(self) -> None:
        self.assertIs(get_connector("wise"), WiseConnector)

    def test_metadata(self) -> None:
        self.assertEqual(WiseConnector.id, "wise")
        self.assertIn("api_key", [m.value for m in WiseConnector.auth_methods])


class ConnectTests(unittest.TestCase):
    def test_connect_validates_and_stores(self) -> None:
        conn, http = _wise()
        http.route("GET", "/v1/profiles", FakeResponse(200, PROFILES))
        result = conn.connect(token="tok_live_123")
        self.assertTrue(result.ok)
        self.assertIn("2 profile(s)", result.message)
        cred = conn.vault.get("connector:wise", "wise")
        self.assertEqual(cred.password, "tok_live_123")
        self.assertEqual(cred.metadata["base_url"], "https://api.wise.com")
        _m, url, _p, headers = http.calls[0]
        self.assertIn("api.wise.com/v1/profiles", url)
        self.assertEqual(headers["Authorization"], "Bearer tok_live_123")

    def test_connect_sandbox(self) -> None:
        conn, http = _wise()
        http.route("GET", "/v1/profiles", FakeResponse(200, PROFILES))
        result = conn.connect(token="tok_sandbox", sandbox=True)
        self.assertTrue(result.ok)
        self.assertIn("sandbox", result.account)
        cred = conn.vault.get("connector:wise", "wise")
        self.assertEqual(
            cred.metadata["base_url"], "https://api.sandbox.transferwise.tech"
        )
        _m, url, _p, _h = http.calls[0]
        self.assertIn("api.sandbox.transferwise.tech", url)

    def test_connect_rejects_second_token(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.connect(token="other")
        self.assertIn("already connected", str(ctx.exception))

    def test_connect_empty_token_raises(self) -> None:
        conn, _http = _wise()
        with mock.patch("sys.stdin.isatty", return_value=False):
            with self.assertRaises(ConnectorError):
                conn.connect(token="")

    def test_connect_rejected_token_fails_fast(self) -> None:
        conn, http = _wise()
        http.route("GET", "/v1/profiles",
                   FakeResponse(401, {"error": "unauthorized"}))
        with self.assertRaises(WiseError) as ctx:
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
        conn, _http = _wise()
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertFalse(conn.test_connection())

    def test_status_connected(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/v1/profiles", FakeResponse(200, PROFILES))
        st = conn.status()
        self.assertTrue(st.connected)
        self.assertIn("live", st.account or "")
        self.assertTrue(conn.test_connection())

    def test_status_rejected_token(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/v1/profiles", FakeResponse(401, {}))
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertIn("rejected", st.detail)
        self.assertFalse(conn.test_connection())


class ApiTests(unittest.TestCase):
    def test_list_profiles(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/v1/profiles", FakeResponse(200, PROFILES))
        profiles = conn.list_profiles()
        self.assertEqual(len(profiles), 2)
        self.assertEqual(profiles[0]["fullName"], "Devon Tester")

    def test_create_quote_source_amount(self) -> None:
        conn, http = _connected()
        http.route("POST", "/v2/quotes", FakeResponse(200, {
            "id": "quote-uuid-1", "rate": 0.92,
        }))
        quote = conn.create_quote("USD", "EUR", source_amount=100)
        self.assertEqual(quote["id"], "quote-uuid-1")
        _m, url, payload, _h = http.calls[-1]
        self.assertIn("/v2/quotes", url)
        self.assertEqual(payload["sourceCurrency"], "USD")
        self.assertEqual(payload["targetCurrency"], "EUR")
        self.assertEqual(payload["sourceAmount"], 100)
        self.assertNotIn("targetAmount", payload)

    def test_create_quote_target_amount(self) -> None:
        conn, http = _connected()
        http.route("POST", "/v2/quotes", FakeResponse(200, {"id": "q2"}))
        conn.create_quote("usd", "eur", target_amount=85)
        _m, _u, payload, _h = http.calls[-1]
        self.assertEqual(payload["targetAmount"], 85)

    def test_create_quote_validates_amounts(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.create_quote("USD", "EUR", source_amount=100,
                              target_amount=85)
        with self.assertRaises(ConnectorError):
            conn.create_quote("USD", "EUR")
        with self.assertRaises(ConnectorError):
            conn.create_quote("", "EUR", source_amount=100)

    def test_create_authenticated_quote(self) -> None:
        conn, http = _connected()
        http.route("POST", "/v3/profiles/111/quotes", FakeResponse(200, {
            "id": "auth-quote-uuid", "rateExpirationTime": "2026-10-03T00:00:00Z",
        }))
        quote = conn.create_authenticated_quote(
            111, "USD", "EUR", source_amount=100
        )
        self.assertEqual(quote["id"], "auth-quote-uuid")
        _m, url, payload, _h = http.calls[-1]
        self.assertIn("/v3/profiles/111/quotes", url)
        self.assertEqual(payload["sourceAmount"], 100)

    def test_create_authenticated_quote_needs_profile(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.create_authenticated_quote(0, "USD", "EUR",
                                            source_amount=100)

    def test_create_transfer_confirmed(self) -> None:
        conn, http = _connected()
        http.route("POST", "/v1/transfers", FakeResponse(200, {
            "id": 987, "status": "waiting_for_funds",
        }))
        result = conn.create_transfer(
            555, "auth-quote-uuid", reference="rent", confirmed=True
        )
        self.assertEqual(result["id"], 987)
        self.assertEqual(result["status"], "waiting_for_funds")
        _m, url, payload, _h = http.calls[-1]
        self.assertIn("/v1/transfers", url)
        self.assertEqual(payload["targetAccount"], 555)
        self.assertEqual(payload["quoteUuid"], "auth-quote-uuid")
        self.assertTrue(payload["customerTransactionId"])  # idempotency UUID
        self.assertEqual(payload["details"]["reference"], "rent")

    def test_create_transfer_requires_confirmation(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.create_transfer(555, "auth-quote-uuid")
        self.assertIn("confirmed=True", str(ctx.exception))

    def test_create_transfer_validates(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.create_transfer(0, "q", confirmed=True)
        with self.assertRaises(ConnectorError):
            conn.create_transfer(555, "", confirmed=True)

    def test_create_transfer_via_human_checkpoint(self) -> None:
        conn, http = _connected()
        db = _db()
        with self.assertRaises(HumanCheckpointPending) as ctx:
            conn.create_transfer(555, "auth-quote-uuid", db=db)
        store = CheckpointStore(db)
        cp = store.get(ctx.exception.checkpoint.id)
        self.assertIn("555", cp.instructions)
        http.route("POST", "/v1/transfers", FakeResponse(200, {
            "id": 988, "status": "waiting_for_funds",
        }))
        store.resolve(cp.id, note="approved")
        result = conn.resume_checkpoint(store.get(cp.id), db=db)
        self.assertEqual(result["id"], 988)

    def test_create_transfer_resume_unresolved_raises(self) -> None:
        conn, _http = _connected()
        db = _db()
        with self.assertRaises(HumanCheckpointPending) as ctx:
            conn.create_transfer(555, "q", db=db)
        store = CheckpointStore(db)
        with self.assertRaises(ConnectorError):
            conn.resume_checkpoint(
                store.get(ctx.exception.checkpoint.id), db=db
            )

    def test_get_transfer(self) -> None:
        conn, http = _connected()
        http.route("GET", "/v1/transfers/987", FakeResponse(200, {
            "id": 987, "status": "outgoing_payment_sent",
        }))
        transfer = conn.get_transfer(987)
        self.assertEqual(transfer["status"], "outgoing_payment_sent")

    def test_get_transfer_needs_id(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.get_transfer(0)

    def test_list_recipients(self) -> None:
        conn, http = _connected()
        http.route("GET", "/v2/accounts", FakeResponse(200, [
            {"id": 555, "accountHolderName": "Devon"},
        ]))
        recipients = conn.list_recipients(111)
        self.assertEqual(recipients[0]["id"], 555)

    def test_list_recipients_needs_profile(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.list_recipients(0)

    def test_not_connected_raises(self) -> None:
        conn, _http = _wise()
        with self.assertRaises(ConnectorError) as ctx:
            conn.list_profiles()
        self.assertIn("not connected", str(ctx.exception))

    def test_rate_limit_error(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/v1/profiles", FakeResponse(429, {}))
        with self.assertRaises(WiseError) as ctx:
            conn.list_profiles()
        self.assertEqual(ctx.exception.status_code, 429)

    def test_api_error_detail(self) -> None:
        conn, http = _connected()
        http.route("POST", "/v1/transfers", FakeResponse(422, {
            "errors": [{"code": "BALANCE_TOO_LOW", "message": "no funds"}],
        }))
        with self.assertRaises(WiseError) as ctx:
            conn.create_transfer(555, "q", confirmed=True)
        self.assertIn("no funds", str(ctx.exception))
        self.assertEqual(ctx.exception.error_code, "BALANCE_TOO_LOW")

    def test_invalid_json_error(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/v1/profiles", BadJsonResponse(200, "nope"))
        with self.assertRaises(WiseError) as ctx:
            conn.list_profiles()
        self.assertIn("invalid JSON", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
