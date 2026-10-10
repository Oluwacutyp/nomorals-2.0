"""Stripe connector tests. HTTP is fully mocked — no network.

Covers: connect/status/test_connection (live + test mode), balance,
customers, payment intents, payment links, charges, invoices, products /
prices, the money-move confirmation gate (transfer/payout/intent
confirm), and checkpoint resume.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest import mock
from urllib.parse import parse_qsl

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors.base import ConnectorError
from nomorals.connectors.registry import get_connector
from nomorals.connectors.stripe import StripeConnector, StripeError, _flatten
from nomorals.storage.db import Database


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test")


class FakeResponse:
    def __init__(self, status: int = 200, payload: Any = None) -> None:
        self.status = status
        self._payload = payload
        self.headers: dict = {}
        self.url = ""

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def json(self) -> Any:
        return self._payload


class FakeHttp:
    """Scripted stand-in for HttpClient. No network."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any, Any]] = []
        self.routes: list[tuple[str, str, FakeResponse]] = []

    def route(self, method: str, path: str, response: FakeResponse) -> None:
        self.routes.append((method.upper(), path, response))

    def _dispatch(self, method: str, url: str, payload: Any = None,
                  **kw: Any) -> FakeResponse:
        self.calls.append((method.upper(), url, payload, kw.get("headers")))
        for rm, rp, resp in sorted(self.routes, key=lambda r: -len(r[1])):
            if rm == method.upper() and rp in url:
                return resp
        return FakeResponse(200, {"error": {"type": "invalid_request_error",
                                            "code": "not_mocked",
                                            "message": "not mocked"}})

    def get(self, url: str, **kw: Any) -> FakeResponse:
        return self._dispatch("GET", url, None, **kw)

    def post_form(self, url: str, form: Any, **kw: Any) -> FakeResponse:
        return self._dispatch("POST", url, dict(form), **kw)

    def request(self, method: str, url: str, **kw: Any) -> FakeResponse:
        return self._dispatch(method.upper(), url, kw.get("params"), **kw)


def _stripe(http: FakeHttp | None = None) -> tuple[StripeConnector, FakeHttp]:
    http = http or FakeHttp()
    return StripeConnector(_vault(), http=http), http


def _connected(http: FakeHttp | None = None,
               key: str = "sk_test_abc") -> tuple[StripeConnector, FakeHttp]:
    conn, http = _stripe(http)
    http.route("GET", "/v1/account",
               FakeResponse(200, {"id": "acct_1", "email": "o@x.com",
                                  "country": "US"}))
    result = conn.connect(secret_key=key)
    assert result.ok
    return conn, http


class FlattenTest(unittest.TestCase):
    def test_bracket_notation(self):
        flat = _flatten({"line_items": [{"price": "price_1", "quantity": 2}],
                         "amount": 500, "flag": True, "skip": None})
        self.assertEqual(flat["line_items[0][price]"], "price_1")
        self.assertEqual(flat["line_items[0][quantity]"], "2")
        self.assertEqual(flat["amount"], "500")
        self.assertEqual(flat["flag"], "true")
        self.assertNotIn("skip", flat)


class LifecycleTest(unittest.TestCase):
    def test_registered(self):
        self.assertIs(get_connector("stripe"), StripeConnector)

    def test_connect_test_mode(self):
        conn, http = _stripe()
        http.route("GET", "/v1/account",
                   FakeResponse(200, {"id": "acct_1", "email": "o@x.com"}))
        result = conn.connect(secret_key="sk_test_abc")
        self.assertTrue(result.ok)
        self.assertIn("test", result.account)
        self.assertIn("TEST MODE", result.message)

    def test_connect_live_mode(self):
        conn, http = _stripe()
        http.route("GET", "/v1/account",
                   FakeResponse(200, {"id": "acct_1",
                                      "business_name": "Acme"}))
        result = conn.connect(secret_key="sk_live_abc")
        self.assertTrue(result.ok)
        self.assertIn("LIVE", result.message)

    def test_connect_rejects_non_key(self):
        conn, _http = _stripe()
        with self.assertRaises(ConnectorError):
            conn.connect(secret_key="pk_live_whatever")

    def test_connect_twice_refused(self):
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.connect(secret_key="sk_test_other")

    def test_status_not_connected(self):
        conn, _http = _stripe()
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertIn("STRIPE_SECRET_KEY", st.detail)

    def test_status_connected(self):
        conn, http = _connected()
        http.route("GET", "/v1/balance",
                   FakeResponse(200, {"available": [{"amount": 12500,
                                                     "currency": "usd"}],
                                      "pending": []}))
        st = conn.status()
        self.assertTrue(st.connected)
        self.assertIn("125.00 USD", st.detail)
        self.assertIn("test", st.detail)

    def test_status_bad_key_is_honest(self):
        conn, http = _connected()
        http.route("GET", "/v1/balance",
                   FakeResponse(200, {"error": {"type": "authentication_error",
                                               "code": "invalid_api_key",
                                               "message": "bad key"}}))
        st = conn.status()
        self.assertFalse(st.connected)

    def test_api_error_shape(self):
        conn, http = _connected()
        http.route("GET", "/v1/balance",
                   FakeResponse(200, {"error": {"type": "authentication_error",
                                               "code": "invalid_api_key",
                                               "message": "bad key"}}))
        with self.assertRaises(StripeError) as ctx:
            conn.get_balance()
        self.assertEqual(ctx.exception.code, "invalid_api_key")

    def test_disconnect(self):
        conn, _http = _connected()
        conn.disconnect()
        self.assertFalse(conn.status().connected)


class ReadsTest(unittest.TestCase):
    def test_get_balance(self):
        conn, http = _connected()
        http.route("GET", "/v1/balance",
                   FakeResponse(200, {"available": []}))
        self.assertEqual(conn.get_balance(), {"available": []})

    def test_list_customers(self):
        conn, http = _connected()
        http.route("GET", "/v1/customers",
                   FakeResponse(200, {"data": [{"id": "cus_1"}]}))
        out = conn.list_customers(limit=5)
        self.assertEqual(out, [{"id": "cus_1"}])

    def test_create_customer(self):
        conn, http = _connected()
        http.route("POST", "/v1/customers",
                   FakeResponse(200, {"id": "cus_2"}))
        out = conn.create_customer(name="Ada", email="a@x.com")
        self.assertEqual(out["id"], "cus_2")
        _m, _u, form, headers = http.calls[-1]
        self.assertEqual(form["name"], "Ada")
        self.assertTrue(headers["Authorization"].startswith("Bearer sk_test"))

    def test_create_payment_intent(self):
        conn, http = _connected()
        http.route("POST", "/v1/payment_intents",
                   FakeResponse(200, {"id": "pi_1", "status": "requires_confirmation"}))
        out = conn.create_payment_intent(2500, "usd", description="gig pay")
        self.assertEqual(out["id"], "pi_1")
        _m, _u, form, _h = http.calls[-1]
        self.assertEqual(form["amount"], "2500")
        self.assertEqual(form["currency"], "usd")

    def test_create_payment_link_brackets(self):
        conn, http = _connected()
        http.route("POST", "/v1/payment_links",
                   FakeResponse(200, {"id": "plink_1",
                                      "url": "https://pay.stripe/1"}))
        out = conn.create_payment_link(
            [{"price_data": {"currency": "usd", "unit_amount": 500,
                             "product_data": {"name": "gig"}},
              "quantity": 1}])
        self.assertEqual(out["url"], "https://pay.stripe/1")
        _m, _u, form, _h = http.calls[-1]
        self.assertEqual(form["line_items[0][price_data][unit_amount]"], "500")

    def test_list_charges(self):
        conn, http = _connected()
        http.route("GET", "/v1/charges",
                   FakeResponse(200, {"data": [{"id": "ch_1"}]}))
        self.assertEqual(conn.list_charges(), [{"id": "ch_1"}])

    def test_create_product_and_price(self):
        conn, http = _connected()
        http.route("POST", "/v1/products",
                   FakeResponse(200, {"id": "prod_1"}))
        http.route("POST", "/v1/prices",
                   FakeResponse(200, {"id": "price_1"}))
        self.assertEqual(conn.create_product("gig")["id"], "prod_1")
        self.assertEqual(
            conn.create_price("prod_1", 5000, "usd")["id"], "price_1")


class MoneyGateTest(unittest.TestCase):
    def test_transfer_needs_confirmation(self):
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.create_transfer(10000, "usd", "acct_dest")
        self.assertIn("confirmation", str(ctx.exception))

    def test_transfer_confirmed(self):
        conn, http = _connected()
        http.route("POST", "/v1/transfers",
                   FakeResponse(200, {"id": "tr_1"}))
        out = conn.create_transfer(10000, "usd", "acct_dest", confirmed=True)
        self.assertEqual(out["id"], "tr_1")
        _m, _u, form, _h = http.calls[-1]
        self.assertEqual(form["destination"], "acct_dest")

    def test_payout_needs_confirmation(self):
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.create_payout(5000, "usd")

    def test_payout_confirmed(self):
        conn, http = _connected()
        http.route("POST", "/v1/payouts",
                   FakeResponse(200, {"id": "po_1"}))
        out = conn.create_payout(5000, "usd", confirmed=True)
        self.assertEqual(out["id"], "po_1")

    def test_confirm_intent_needs_confirmation(self):
        conn, http = _connected()
        http.route("GET", "/v1/payment_intents/pi_1",
                   FakeResponse(200, {"id": "pi_1", "amount": 2500,
                                      "currency": "usd"}))
        with self.assertRaises(ConnectorError):
            conn.confirm_payment_intent("pi_1")

    def test_confirm_intent_confirmed(self):
        conn, http = _connected()
        http.route("GET", "/v1/payment_intents/pi_1",
                   FakeResponse(200, {"id": "pi_1", "amount": 2500,
                                      "currency": "usd",
                                      "description": "gig"}))
        http.route("POST", "/v1/payment_intents/pi_1/confirm",
                   FakeResponse(200, {"id": "pi_1", "status": "succeeded"}))
        out = conn.confirm_payment_intent("pi_1", confirmed=True)
        self.assertEqual(out["status"], "succeeded")

    def test_resume_checkpoint(self):
        conn, http = _connected()
        http.route("POST", "/v1/payouts",
                   FakeResponse(200, {"id": "po_9"}))
        cp = mock.Mock()
        cp.resume_state = {"stage": "create_payout", "op": "create_payout",
                           "amount": 700, "currency": "usd",
                           "destination": ""}
        out = conn.resume_checkpoint(cp, db=object())
        self.assertEqual(out["id"], "po_9")

    def test_resume_checkpoint_rejects_unknown_stage(self):
        conn, _http = _connected()
        cp = mock.Mock()
        cp.resume_state = {"stage": "nope", "op": "nope"}
        with self.assertRaises(StripeError):
            conn.resume_checkpoint(cp, db=object())


if __name__ == "__main__":
    unittest.main()
