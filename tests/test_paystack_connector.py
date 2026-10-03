"""Paystack connector tests. HTTP is fully mocked — no network."""

from __future__ import annotations

import hashlib
import hmac
import json
import unittest
import urllib.parse
from typing import Any
from unittest import mock

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors.base import ConnectorError
from nomorals.connectors.checkpoints import CheckpointStore
from nomorals.connectors.paystack import PaystackConnector, PaystackError
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
        return FakeResponse(404, {"status": False,
                                  "message": "not mocked"})

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


def _paystack(http: FakeHttp | None = None) -> tuple[PaystackConnector, FakeHttp]:
    http = http or FakeHttp()
    return PaystackConnector(_vault(), http=http), http


def _connected(
    http: FakeHttp | None = None, *, key: str = "sk_test_abc"
) -> tuple[PaystackConnector, FakeHttp]:
    conn, http = _paystack(http)
    http.route("GET", "/transaction", FakeResponse(200, {
        "status": True, "message": "ok", "data": [], "meta": {},
    }))
    result = conn.connect(secret_key=key)
    assert result.ok
    return conn, http


class RegistryTests(unittest.TestCase):
    def test_registered(self) -> None:
        self.assertIs(get_connector("paystack"), PaystackConnector)

    def test_metadata(self) -> None:
        self.assertEqual(PaystackConnector.id, "paystack")
        self.assertIn("api_key", [m.value for m in PaystackConnector.auth_methods])


class ConnectTests(unittest.TestCase):
    def test_connect_validates_and_stores(self) -> None:
        conn, http = _paystack()
        http.route("GET", "/transaction", FakeResponse(200, {
            "status": True, "message": "ok", "data": [],
        }))
        result = conn.connect(secret_key="sk_test_abc")
        self.assertTrue(result.ok)
        self.assertIn("test", result.account)
        cred = conn.vault.get("connector:paystack", "paystack")
        self.assertEqual(cred.password, "sk_test_abc")
        self.assertEqual(cred.metadata["mode"], "test")
        # secret in the Bearer header, never the URL
        _m, url, _p, headers = http.calls[0]
        self.assertEqual(headers["Authorization"], "Bearer sk_test_abc")
        self.assertNotIn("sk_test_abc", url)

    def test_connect_detects_live_mode(self) -> None:
        conn, http = _connected(key="sk_live_xyz")
        self.assertIn("live", conn.vault.get(
            "connector:paystack", "paystack").metadata["mode"])

    def test_connect_rejects_second_key(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.connect(secret_key="other")
        self.assertIn("already connected", str(ctx.exception))

    def test_connect_empty_key_raises(self) -> None:
        conn, _http = _paystack()
        with mock.patch("sys.stdin.isatty", return_value=False):
            with self.assertRaises(ConnectorError):
                conn.connect(secret_key="")

    def test_connect_rejected_key_fails_fast(self) -> None:
        conn, http = _paystack()
        http.route("GET", "/transaction", FakeResponse(
            401, {"status": False, "message": "Invalid key"}))
        with self.assertRaises(PaystackError) as ctx:
            conn.connect(secret_key="sk_test_bad")
        self.assertIn("rejected", str(ctx.exception))
        self.assertIsNone(conn._load_credential())

    def test_connect_api_refusal_fails_fast(self) -> None:
        conn, http = _paystack()
        http.route("GET", "/transaction", FakeResponse(200, {
            "status": False, "message": "Something went wrong",
        }))
        with self.assertRaises(PaystackError) as ctx:
            conn.connect(secret_key="sk_test_abc")
        self.assertIn("refused", str(ctx.exception))

    def test_disconnect_clears(self) -> None:
        conn, _http = _connected()
        conn.disconnect()
        self.assertIsNone(conn._load_credential())
        conn.disconnect()


class StatusTests(unittest.TestCase):
    def test_status_not_connected(self) -> None:
        conn, _http = _paystack()
        self.assertFalse(conn.status().connected)
        self.assertFalse(conn.test_connection())

    def test_status_connected(self) -> None:
        conn, http = _connected()
        http.route("GET", "/transaction", FakeResponse(200, {
            "status": True, "message": "ok", "data": [],
        }))
        st = conn.status()
        self.assertTrue(st.connected)
        self.assertIn("test", st.account or "")
        self.assertTrue(conn.test_connection())

    def test_status_rejected_key(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/transaction", FakeResponse(401, {
            "status": False, "message": "Invalid key",
        }))
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertFalse(conn.test_connection())


class TransactionTests(unittest.TestCase):
    def test_initialize_transaction(self) -> None:
        conn, http = _connected()
        http.route("POST", "/transaction/initialize", FakeResponse(200, {
            "status": True, "message": "Authorization URL created",
            "data": {
                "authorization_url": "https://checkout.paystack.com/abc",
                "access_code": "abc", "reference": "ref123",
            },
        }))
        result = conn.initialize_transaction(
            "buyer@example.com", 500000,
            callback_url="https://shop.example/cb",
            metadata={"order_id": "7"},
        )
        self.assertEqual(result["reference"], "ref123")
        self.assertIn("checkout.paystack.com",
                      result["authorization_url"])
        _m, url, payload, _h = http.calls[-1]
        self.assertIn("/transaction/initialize", url)
        self.assertEqual(payload["email"], "buyer@example.com")
        self.assertEqual(payload["amount"], 500000)
        self.assertEqual(payload["callback_url"], "https://shop.example/cb")
        self.assertEqual(payload["metadata"], {"order_id": "7"})

    def test_initialize_bad_email_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.initialize_transaction("not-an-email", 1000)

    def test_initialize_bad_amount_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.initialize_transaction("a@b.c", 0)
        with self.assertRaises(ConnectorError):
            conn.initialize_transaction("a@b.c", -50)

    def test_verify_transaction(self) -> None:
        conn, http = _connected()
        http.route("GET", "/transaction/verify/ref123", FakeResponse(200, {
            "status": True, "message": "Verification successful",
            "data": {
                "status": "success", "reference": "ref123",
                "amount": 500000, "currency": "NGN",
                "paid_at": "2026-01-01T00:00:00Z",
                "customer": {"email": "buyer@example.com"},
            },
        }))
        result = conn.verify_transaction("ref123")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["amount_kobo"], 500000)
        self.assertEqual(result["currency"], "NGN")

    def test_verify_empty_reference_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.verify_transaction("")

    def test_get_transaction(self) -> None:
        conn, http = _connected()
        http.route("GET", "/transaction/4242", FakeResponse(200, {
            "status": True, "message": "ok",
            "data": {"id": 4242, "status": "success"},
        }))
        result = conn.get_transaction("4242")
        self.assertEqual(result["id"], 4242)

    def test_list_transactions(self) -> None:
        conn, http = _connected()
        http.routes.clear()  # drop the connect() validation route
        http.route("GET", "/transaction", FakeResponse(200, {
            "status": True, "message": "ok",
            "data": [{"reference": "r1"}, {"reference": "r2"}],
            "meta": {"total": 2, "page": 1},
        }))
        result = conn.list_transactions(per_page=2, status="success")
        self.assertEqual(len(result["transactions"]), 2)
        self.assertEqual(result["meta"]["total"], 2)
        _m, url, _p, _h = http.calls[-1]
        self.assertIn("perPage=2", url)
        self.assertIn("status=success", url)

    def test_list_transactions_bad_status_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.list_transactions(status="maybe")

    def test_not_connected_raises(self) -> None:
        conn, _http = _paystack()
        with self.assertRaises(ConnectorError) as ctx:
            conn.list_transactions()
        self.assertIn("not connected", str(ctx.exception))


class CustomerTests(unittest.TestCase):
    def test_create_customer(self) -> None:
        conn, http = _connected()
        http.route("POST", "/customer", FakeResponse(200, {
            "status": True, "message": "Customer created",
            "data": {"email": "buyer@example.com", "id": 99},
        }))
        result = conn.create_customer(
            "buyer@example.com", first_name="Ada", phone="+2348000000000"
        )
        self.assertEqual(result["id"], 99)
        _m, _u, payload, _h = http.calls[-1]
        self.assertEqual(payload["first_name"], "Ada")
        self.assertEqual(payload["phone"], "+2348000000000")

    def test_create_customer_bad_email_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.create_customer("nope")

    def test_list_customers(self) -> None:
        conn, http = _connected()
        http.route("GET", "/customer", FakeResponse(200, {
            "status": True, "message": "ok",
            "data": [{"email": "a@b.c"}], "meta": {},
        }))
        result = conn.list_customers()
        self.assertEqual(len(result["customers"]), 1)


class ChargeTests(unittest.TestCase):
    def test_charge_requires_confirmation(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.charge_authorization("AUTH_x", "a@b.c", 250000)
        self.assertIn("confirmed=True", str(ctx.exception))

    def test_charge_refuses_bad_amount(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.charge_authorization("AUTH_x", "a@b.c", 0, confirmed=True)

    def test_charge_confirmed(self) -> None:
        conn, http = _connected()
        http.route("POST", "/transaction/charge_authorization",
                   FakeResponse(200, {
                       "status": True, "message": "Charge attempted",
                       "data": {"reference": "ch_1", "status": "success"},
                   }))
        result = conn.charge_authorization(
            "AUTH_x", "a@b.c", 250000, confirmed=True
        )
        self.assertEqual(result["reference"], "ch_1")
        _m, url, payload, _h = http.calls[-1]
        self.assertIn("/transaction/charge_authorization", url)
        self.assertEqual(payload["authorization_code"], "AUTH_x")
        self.assertEqual(payload["amount"], 250000)

    def test_charge_via_human_checkpoint(self) -> None:
        from nomorals.connectors.checkpoints import HumanCheckpointPending

        conn, http = _connected()
        db = _db()
        with self.assertRaises(HumanCheckpointPending) as ctx:
            conn.charge_authorization("AUTH_x", "a@b.c", 250000, db=db)
        cp_id = ctx.exception.checkpoint.id
        store = CheckpointStore(db)
        self.assertIn("250", store.get(cp_id).instructions.replace(",", ""))
        http.route("POST", "/transaction/charge_authorization",
                   FakeResponse(200, {
                       "status": True, "message": "ok",
                       "data": {"reference": "ch_2"},
                   }))
        store.resolve(cp_id, note="approved")
        result = conn.resume_checkpoint(store.get(cp_id), db=db)
        self.assertEqual(result["reference"], "ch_2")

    def test_resume_unresolved_checkpoint_raises(self) -> None:
        from nomorals.connectors.checkpoints import HumanCheckpointPending

        conn, _http = _connected()
        db = _db()
        with self.assertRaises(HumanCheckpointPending) as ctx:
            conn.charge_authorization("AUTH_x", "a@b.c", 250000, db=db)
        store = CheckpointStore(db)
        with self.assertRaises(ConnectorError):
            conn.resume_checkpoint(
                store.get(ctx.exception.checkpoint.id), db=db
            )


class WebhookTests(unittest.TestCase):
    def test_verify_webhook_signature(self) -> None:
        conn, _http = _connected(key="sk_test_webhook")
        body = b'{"event":"charge.success"}'
        sig = hmac.new(b"sk_test_webhook", body, hashlib.sha512).hexdigest()
        self.assertTrue(conn.verify_webhook_signature(body, sig))
        self.assertFalse(conn.verify_webhook_signature(body, "deadbeef"))
        self.assertFalse(conn.verify_webhook_signature(b"tampered", sig))


if __name__ == "__main__":
    unittest.main()
