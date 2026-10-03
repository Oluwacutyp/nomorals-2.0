"""Coinbase connector tests. HTTP is fully mocked — no network.

Crypto fixtures: a throwaway P-256 key generated locally with openssl for
these tests only — never a real credential. Its public point (also
printed by openssl at generation time) is asserted against the parser so
the DER decoding is grounded, not self-referential.
"""

from __future__ import annotations

import base64
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
from nomorals.connectors.coinbase import (
    CoinbaseConnector,
    CoinbaseError,
    _build_jwt,
    _ecdsa_sign_p256,
    _ecdsa_verify_p256,
    _load_private_scalar,
    _public_point,
)
from nomorals.connectors.registry import get_connector
from nomorals.storage.db import Database

# THROWAWAY TEST KEY ONLY — generated locally for unit tests.
_TEST_PEM = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIE0y0lC6n65jPG7tTxHU/wczYTV7ToKF/6QjlqL2Q/LIoAoGCCqGSM49
AwEHoUQDQgAEr6sNt9mcJ22ez+jafGDlJJiC0kndohCAtcqlgPwrPxgeNeco3QS1
CtrZdHhOKYb1vc/eODYJUqaSOTGuQxwgrg==
-----END EC PRIVATE KEY-----"""

# Same key, PKCS8 encoding (openssl pkcs8 -topk8 -nocrypt).
_TEST_PKCS8 = """-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgTTLSULqfrmM8bu1P
EdT/BzNhNXtOgoX/pCOWovZD8sihRANCAASvqw232ZwnbZ7P6Np8YOUkmILSSd2i
EIC1yqWA/Cs/GB415yjdBLUK2tl0eE4phvW9z944NglSppI5Ma5DHCCu
-----END PRIVATE KEY-----"""

# openssl's own report of the test key's public point (grounded fixture).
_TEST_PUB_X = 0xAFAB0DB7D99C276D9ECFE8DA7C60E5249882D249DDA21080B5CAA580FC2B3F18
_TEST_PUB_Y = 0x1E35E728DD04B50ADAD974784E2986F5BDCFDE38360952A6923931AE431C20AE

_TEST_KEY_NAME = "organizations/test-org/apiKeys/test-key-id"


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


def _coinbase(http: FakeHttp | None = None) -> tuple[CoinbaseConnector, FakeHttp]:
    http = http or FakeHttp()
    return CoinbaseConnector(_vault(), http=http), http


def _connected(http: FakeHttp | None = None) -> tuple[CoinbaseConnector, FakeHttp]:
    conn, http = _coinbase(http)
    http.route("GET", "/api/v3/brokerage/accounts", FakeResponse(200, {
        "accounts": [{"uuid": "a1", "name": "USD Wallet"}],
    }))
    result = conn.connect(key_name=_TEST_KEY_NAME, private_key=_TEST_PEM)
    assert result.ok
    return conn, http


class RegistryTests(unittest.TestCase):
    def test_registered(self) -> None:
        self.assertIs(get_connector("coinbase"), CoinbaseConnector)

    def test_metadata(self) -> None:
        self.assertEqual(CoinbaseConnector.id, "coinbase")
        self.assertIn("api_key", [m.value for m in CoinbaseConnector.auth_methods])


class CryptoTests(unittest.TestCase):
    def test_parse_sec1_pem_matches_openssl_public_point(self) -> None:
        scalar = _load_private_scalar(_TEST_PEM)
        pub = _public_point(scalar)
        self.assertEqual(pub.x, _TEST_PUB_X)
        self.assertEqual(pub.y, _TEST_PUB_Y)

    def test_parse_pkcs8_pem_gives_same_scalar(self) -> None:
        self.assertEqual(
            _load_private_scalar(_TEST_PKCS8),
            _load_private_scalar(_TEST_PEM),
        )

    def test_parse_rejects_garbage(self) -> None:
        with self.assertRaises(CoinbaseError):
            _load_private_scalar("not a key at all")
        with self.assertRaises(CoinbaseError):
            _load_private_scalar(
                "-----BEGIN EC PRIVATE KEY-----\n!!!\n-----END EC PRIVATE KEY-----"
            )

    def test_sign_verify_roundtrip(self) -> None:
        scalar = _load_private_scalar(_TEST_PEM)
        pub = _public_point(scalar)
        sig = _ecdsa_sign_p256(scalar, b"coinbase test message")
        self.assertEqual(len(sig), 64)
        self.assertTrue(
            _ecdsa_verify_p256(pub.x, pub.y, b"coinbase test message", sig)
        )

    def test_verify_rejects_tampered_message(self) -> None:
        scalar = _load_private_scalar(_TEST_PEM)
        pub = _public_point(scalar)
        sig = _ecdsa_sign_p256(scalar, b"original")
        self.assertFalse(_ecdsa_verify_p256(pub.x, pub.y, b"tampered", sig))
        self.assertFalse(_ecdsa_verify_p256(pub.x, pub.y, b"original", b"\x00" * 64))

    def test_jwt_structure_matches_advanced_trade_scheme(self) -> None:
        token = _build_jwt(
            "GET", "/api/v3/brokerage/accounts", _TEST_KEY_NAME, _TEST_PEM
        )
        header_b64, payload_b64, sig_b64 = token.split(".")

        def decode(part: str) -> Any:
            return json.loads(
                base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))
            )

        header = decode(header_b64)
        self.assertEqual(header["alg"], "ES256")
        self.assertEqual(header["kid"], _TEST_KEY_NAME)
        self.assertTrue(header["nonce"])
        payload = decode(payload_b64)
        self.assertEqual(payload["sub"], _TEST_KEY_NAME)
        self.assertEqual(payload["iss"], "coinbase-cloud")
        self.assertEqual(payload["aud"], ["retail_rest_api_proxy"])
        self.assertEqual(
            payload["uri"], "GET api.coinbase.com/api/v3/brokerage/accounts"
        )
        self.assertEqual(payload["exp"] - payload["nbf"], 120)
        raw_sig = base64.urlsafe_b64decode(sig_b64 + "=" * (-len(sig_b64) % 4))
        self.assertEqual(len(raw_sig), 64)
        # The JWT signature verifies against the key's own public point.
        scalar = _load_private_scalar(_TEST_PEM)
        pub = _public_point(scalar)
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        self.assertTrue(
            _ecdsa_verify_p256(pub.x, pub.y, signing_input, raw_sig)
        )

    def test_build_jwt_rejects_bad_key(self) -> None:
        with self.assertRaises(CoinbaseError):
            _build_jwt("GET", "/x", "name", "garbage")


class ConnectTests(unittest.TestCase):
    def test_connect_validates_and_stores(self) -> None:
        conn, http = _coinbase()
        http.route("GET", "/api/v3/brokerage/accounts", FakeResponse(200, {
            "accounts": [{"uuid": "a1"}, {"uuid": "a2"}],
        }))
        result = conn.connect(key_name=_TEST_KEY_NAME, private_key=_TEST_PEM)
        self.assertTrue(result.ok)
        self.assertIn("2", result.message)
        cred = conn.vault.get("connector:coinbase", "coinbase")
        self.assertEqual(cred.password, _TEST_PEM)
        self.assertEqual(cred.metadata["key_name"], _TEST_KEY_NAME)
        # every request carries a Bearer JWT, never the raw key material
        _m, _u, _p, headers = http.calls[0]
        auth = headers["Authorization"]
        self.assertTrue(auth.startswith("Bearer "))
        self.assertEqual(len(auth.split(" ")[1].split(".")), 3)
        self.assertNotIn(_TEST_PEM, str(http.calls))

    def test_connect_rejects_second_key(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.connect(key_name="other", private_key=_TEST_PEM)
        self.assertIn("already connected", str(ctx.exception))

    def test_connect_empty_raises(self) -> None:
        conn, _http = _coinbase()
        with mock.patch("sys.stdin.isatty", return_value=False):
            with self.assertRaises(ConnectorError):
                conn.connect(key_name="", private_key="")

    def test_connect_rejected_key_fails_fast(self) -> None:
        conn, http = _coinbase()
        http.route("GET", "/api/v3/brokerage/accounts",
                   FakeResponse(401, {"message": "invalid token"}))
        with self.assertRaises(CoinbaseError) as ctx:
            conn.connect(key_name=_TEST_KEY_NAME, private_key=_TEST_PEM)
        self.assertIn("401", str(ctx.exception))
        self.assertIsNone(conn._load_credential())

    def test_connect_bad_pem_fails_fast(self) -> None:
        conn, http = _coinbase()
        http.route("GET", "/api/v3/brokerage/accounts", FakeResponse(200, {
            "accounts": [],
        }))
        with self.assertRaises(CoinbaseError):
            conn.connect(key_name=_TEST_KEY_NAME, private_key="garbage")
        self.assertIsNone(conn._load_credential())

    def test_disconnect_clears(self) -> None:
        conn, _http = _connected()
        conn.disconnect()
        self.assertIsNone(conn._load_credential())
        conn.disconnect()  # idempotent


class StatusTests(unittest.TestCase):
    def test_status_not_connected(self) -> None:
        conn, _http = _coinbase()
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertFalse(conn.test_connection())

    def test_status_connected(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/api/v3/brokerage/accounts", FakeResponse(200, {
            "accounts": [{"uuid": "a1"}],
        }))
        st = conn.status()
        self.assertTrue(st.connected)
        self.assertTrue(conn.test_connection())

    def test_status_rejected_key(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/api/v3/brokerage/accounts",
                   FakeResponse(401, {"message": "bad"}))
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertIn("rejected", st.detail)
        self.assertFalse(conn.test_connection())


class ApiTests(unittest.TestCase):
    def test_list_accounts(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/api/v3/brokerage/accounts", FakeResponse(200, {
            "accounts": [{"uuid": "a1", "name": "USD"}],
        }))
        accounts = conn.list_accounts()
        self.assertEqual(accounts[0]["uuid"], "a1")

    def test_get_product(self) -> None:
        conn, http = _connected()
        http.route("GET", "/api/v3/brokerage/products/BTC-USD",
                   FakeResponse(200, {
                       "product_id": "BTC-USD", "price": "67000",
                   }))
        product = conn.get_product("btc-usd")
        self.assertEqual(product["product_id"], "BTC-USD")

    def test_get_product_empty_raises(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.get_product("")

    def test_place_market_order_confirmed(self) -> None:
        conn, http = _connected()
        http.route("POST", "/api/v3/brokerage/orders", FakeResponse(200, {
            "success": True,
            "success_response": {"order_id": "oid-1"},
        }))
        result = conn.place_order("BTC-USD", "BUY", "MARKET",
                                  quote_size=50, confirmed=True)
        self.assertTrue(result["success"])
        _m, url, payload, headers = http.calls[-1]
        self.assertIn("/api/v3/brokerage/orders", url)
        self.assertTrue(headers["Authorization"].startswith("Bearer "))
        self.assertEqual(payload["product_id"], "BTC-USD")
        self.assertEqual(payload["side"], "BUY")
        cfg = payload["order_configuration"]["market_market_ioc"]
        self.assertEqual(cfg["quote_size"], "50")

    def test_place_limit_order_confirmed(self) -> None:
        conn, http = _connected()
        http.route("POST", "/api/v3/brokerage/orders", FakeResponse(200, {
            "success": True,
            "success_response": {"order_id": "oid-2"},
        }))
        conn.place_order("ETH-USD", "SELL", "LIMIT", size=0.5, price=3000,
                         confirmed=True)
        _m, _u, payload, _h = http.calls[-1]
        cfg = payload["order_configuration"]["limit_limit_gtc"]
        self.assertEqual(cfg["base_size"], "0.5")
        self.assertEqual(cfg["limit_price"], "3000")

    def test_place_order_requires_confirmation(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.place_order("BTC-USD", "BUY", "MARKET", size=0.01)
        self.assertIn("confirmed=True", str(ctx.exception))

    def test_place_order_validates(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.place_order("BTC-USD", "HOLD", "MARKET", size=1,
                             confirmed=True)
        with self.assertRaises(ConnectorError):
            conn.place_order("BTC-USD", "BUY", "NOPE", size=1, confirmed=True)
        with self.assertRaises(ConnectorError):
            conn.place_order("BTC-USD", "BUY", "MARKET", confirmed=True)
        with self.assertRaises(ConnectorError):
            conn.place_order("BTC-USD", "SELL", "MARKET", quote_size=50,
                             confirmed=True)  # quote_size is BUY-only
        with self.assertRaises(ConnectorError):
            conn.place_order("BTC-USD", "BUY", "LIMIT", size=1,
                             confirmed=True)  # no price
        with self.assertRaises(ConnectorError):
            conn.place_order("", "BUY", "MARKET", size=1, confirmed=True)

    def test_place_order_via_human_checkpoint(self) -> None:
        conn, http = _connected()
        db = _db()
        with self.assertRaises(HumanCheckpointPending) as ctx:
            conn.place_order("BTC-USD", "BUY", "MARKET", size=0.01, db=db)
        store = CheckpointStore(db)
        cp = store.get(ctx.exception.checkpoint.id)
        self.assertIn("BTC-USD", cp.instructions)
        http.route("POST", "/api/v3/brokerage/orders", FakeResponse(200, {
            "success": True,
            "success_response": {"order_id": "oid-3"},
        }))
        store.resolve(cp.id, note="approved")
        result = conn.resume_checkpoint(store.get(cp.id), db=db)
        self.assertTrue(result["success"])

    def test_place_order_resume_unresolved_raises(self) -> None:
        conn, _http = _connected()
        db = _db()
        with self.assertRaises(HumanCheckpointPending) as ctx:
            conn.place_order("BTC-USD", "BUY", "MARKET", size=0.01, db=db)
        store = CheckpointStore(db)
        with self.assertRaises(ConnectorError):
            conn.resume_checkpoint(
                store.get(ctx.exception.checkpoint.id), db=db
            )

    def test_cancel_order_confirmed(self) -> None:
        conn, http = _connected()
        http.route("POST", "/api/v3/brokerage/orders/batch_cancel",
                   FakeResponse(200, {
                       "results": [{"order_id": "oid-1", "success": True}],
                   }))
        result = conn.cancel_order("oid-1", confirmed=True)
        self.assertTrue(result["results"][0]["success"])
        _m, url, payload, _h = http.calls[-1]
        self.assertIn("batch_cancel", url)
        self.assertEqual(payload["order_ids"], ["oid-1"])

    def test_cancel_order_requires_confirmation(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.cancel_order("oid-1")
        self.assertIn("confirmed=True", str(ctx.exception))

    def test_cancel_order_needs_id(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.cancel_order("", confirmed=True)

    def test_cancel_order_via_human_checkpoint(self) -> None:
        conn, http = _connected()
        db = _db()
        with self.assertRaises(HumanCheckpointPending) as ctx:
            conn.cancel_order("oid-9", db=db)
        store = CheckpointStore(db)
        http.route("POST", "/api/v3/brokerage/orders/batch_cancel",
                   FakeResponse(200, {"results": []}))
        store.resolve(ctx.exception.checkpoint.id, note="approved")
        result = conn.resume_checkpoint(
            store.get(ctx.exception.checkpoint.id), db=db
        )
        self.assertIn("results", result)

    def test_not_connected_raises(self) -> None:
        conn, _http = _coinbase()
        with self.assertRaises(ConnectorError) as ctx:
            conn.list_accounts()
        self.assertIn("not connected", str(ctx.exception))

    def test_rate_limit_error(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/api/v3/brokerage/accounts",
                   FakeResponse(429, {"message": "slow down"}))
        with self.assertRaises(CoinbaseError) as ctx:
            conn.list_accounts()
        self.assertEqual(ctx.exception.status_code, 429)

    def test_api_error_unpacks_coinbase_errors(self) -> None:
        conn, http = _connected()
        http.route("POST", "/api/v3/brokerage/orders", FakeResponse(400, {
            "errors": [{"error": "INVALID_ORDER", "message": "bad size"}],
        }))
        with self.assertRaises(CoinbaseError) as ctx:
            conn.place_order("BTC-USD", "BUY", "MARKET", size=1,
                             confirmed=True)
        self.assertIn("bad size", str(ctx.exception))
        self.assertEqual(ctx.exception.error_code, "INVALID_ORDER")

    def test_invalid_json_error(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/api/v3/brokerage/accounts",
                   BadJsonResponse(200, "not-json"))
        with self.assertRaises(CoinbaseError) as ctx:
            conn.list_accounts()
        self.assertIn("invalid JSON", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
