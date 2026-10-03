"""Binance connector tests. HTTP is fully mocked — no network."""

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
from nomorals.connectors.binance import (
    TESTNET_BASE,
    BinanceConnector,
    BinanceError,
)
from nomorals.connectors.checkpoints import (
    CheckpointStore,
    HumanCheckpointPending,
)
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
        return FakeResponse(400, {"code": -1, "msg": "not mocked"})

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


ACCOUNT = {
    "canTrade": True,
    "balances": [
        {"asset": "BTC", "free": "0.5", "locked": "0.0"},
        {"asset": "USDT", "free": "1000.0", "locked": "50.0"},
        {"asset": "DUST", "free": "0.0", "locked": "0.0"},
    ],
}


def _binance(http: FakeHttp | None = None) -> tuple[BinanceConnector, FakeHttp]:
    http = http or FakeHttp()
    return BinanceConnector(_vault(), http=http), http


def _connected(
    http: FakeHttp | None = None, *, testnet: bool = False,
    key: str = "APIKEY123", secret: str = "SECRET456",
) -> tuple[BinanceConnector, FakeHttp]:
    conn, http = _binance(http)
    http.route("GET", "/api/v3/account", FakeResponse(200, dict(ACCOUNT)))
    result = conn.connect(api_key=key, api_secret=secret, testnet=testnet)
    assert result.ok
    return conn, http


def _signed_parts(url: str) -> tuple[str, dict[str, str]]:
    """Split a signed Binance URL into (query_string, params incl signature)."""
    qs = url.split("?", 1)[1]
    params = dict(urllib.parse.parse_qsl(qs))
    return qs, params


class RegistryTests(unittest.TestCase):
    def test_registered(self) -> None:
        self.assertIs(get_connector("binance"), BinanceConnector)

    def test_metadata(self) -> None:
        self.assertEqual(BinanceConnector.id, "binance")
        self.assertIn("api_key", [m.value for m in BinanceConnector.auth_methods])


class ConnectTests(unittest.TestCase):
    def test_connect_validates_and_vaults(self) -> None:
        conn, http = _binance()
        http.route("GET", "/api/v3/account", FakeResponse(200, dict(ACCOUNT)))
        result = conn.connect(api_key="APIKEY123", api_secret="SECRET456")
        self.assertTrue(result.ok)
        self.assertIn("live", result.account)
        cred = conn.vault.get("connector:binance", "binance")
        self.assertEqual(cred.password, "SECRET456")
        self.assertEqual(cred.metadata["api_key"], "APIKEY123")
        self.assertTrue(cred.metadata["can_trade"])
        # the API key rides in the header, the secret only in the HMAC
        _m, url, _p, headers = http.calls[0]
        self.assertEqual(headers["X-MBX-APIKEY"], "APIKEY123")
        self.assertNotIn("SECRET456", url)

    def test_connect_signature_is_correct_hmac(self) -> None:
        conn, http = _binance()
        http.route("GET", "/api/v3/account", FakeResponse(200, dict(ACCOUNT)))
        conn.connect(api_key="APIKEY123", api_secret="SECRET456")
        _m, url, _p, _h = http.calls[0]
        qs, params = _signed_parts(url)
        signature = params.pop("signature")
        unsigned = qs.rsplit("&signature=", 1)[0]
        expected = hmac.new(
            b"SECRET456", unsigned.encode(), hashlib.sha256
        ).hexdigest()
        self.assertEqual(signature, expected)
        self.assertIn("timestamp", params)
        self.assertEqual(params["recvWindow"], "5000")

    def test_connect_testnet(self) -> None:
        conn, http = _connected(testnet=True)
        cred = conn.vault.get("connector:binance", "binance")
        self.assertEqual(cred.metadata["base_url"], TESTNET_BASE)
        self.assertTrue(cred.metadata["testnet"])
        _m, url, _p, _h = http.calls[0]
        self.assertTrue(url.startswith(TESTNET_BASE))

    def test_connect_rejects_second_pair(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.connect(api_key="x", api_secret="y")
        self.assertIn("already connected", str(ctx.exception))

    def test_connect_empty_raises(self) -> None:
        conn, _http = _binance()
        with mock.patch("sys.stdin.isatty", return_value=False):
            with self.assertRaises(ConnectorError):
                conn.connect(api_key="", api_secret="")

    def test_connect_rejected_key_fails_fast(self) -> None:
        conn, http = _binance()
        http.route("GET", "/api/v3/account", FakeResponse(
            401, {"code": -2015, "msg": "Invalid API-key"}))
        with self.assertRaises(BinanceError) as ctx:
            conn.connect(api_key="bad", api_secret="bad")
        self.assertIn("rejected", str(ctx.exception))
        self.assertIsNone(conn._load_credential())

    def test_disconnect_clears(self) -> None:
        conn, _http = _connected()
        conn.disconnect()
        self.assertIsNone(conn._load_credential())
        conn.disconnect()


class StatusTests(unittest.TestCase):
    def test_status_not_connected(self) -> None:
        conn, _http = _binance()
        self.assertFalse(conn.status().connected)
        self.assertFalse(conn.test_connection())

    def test_status_connected(self) -> None:
        conn, http = _connected()
        http.route("GET", "/api/v3/account", FakeResponse(200, dict(ACCOUNT)))
        st = conn.status()
        self.assertTrue(st.connected)
        self.assertIn("live", st.account or "")
        self.assertTrue(conn.test_connection())

    def test_status_rejected_key(self) -> None:
        conn, http = _connected()
        http.routes.clear()
        http.route("GET", "/api/v3/account", FakeResponse(
            401, {"code": -2015, "msg": "bad"}))
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertFalse(conn.test_connection())


class MarketTests(unittest.TestCase):
    def test_get_price_single(self) -> None:
        conn, http = _connected()
        http.route("GET", "/api/v3/ticker/price", FakeResponse(200, {
            "symbol": "BTCUSDT", "price": "67000.50",
        }))
        result = conn.get_price("btcusdt")
        self.assertEqual(result["price"], "67000.50")
        _m, url, _p, headers = http.calls[-1]
        self.assertIn("symbol=BTCUSDT", url)
        # public call: no signature, no API key
        self.assertNotIn("signature", url)
        self.assertIsNone(headers)

    def test_get_price_all(self) -> None:
        conn, http = _connected()
        http.route("GET", "/api/v3/ticker/price", FakeResponse(200, [
            {"symbol": "BTCUSDT", "price": "67000.50"},
        ]))
        result = conn.get_price()
        self.assertEqual(len(result), 1)

    def test_get_ticker_24h(self) -> None:
        conn, http = _connected()
        http.route("GET", "/api/v3/ticker/24hr", FakeResponse(200, {
            "symbol": "BTCUSDT", "priceChangePercent": "2.5",
        }))
        result = conn.get_ticker_24h("BTCUSDT")
        self.assertEqual(result["priceChangePercent"], "2.5")


class AccountTests(unittest.TestCase):
    def test_get_balances_nonzero(self) -> None:
        conn, http = _connected()
        http.route("GET", "/api/v3/account", FakeResponse(200, dict(ACCOUNT)))
        balances = conn.get_balances()
        assets = {b["asset"] for b in balances}
        self.assertEqual(assets, {"BTC", "USDT"})
        usdt = next(b for b in balances if b["asset"] == "USDT")
        self.assertEqual(usdt["free"], 1000.0)
        self.assertEqual(usdt["locked"], 50.0)
        self.assertEqual(usdt["total"], 1050.0)

    def test_get_balances_includes_dust_when_asked(self) -> None:
        conn, http = _connected()
        http.route("GET", "/api/v3/account", FakeResponse(200, dict(ACCOUNT)))
        balances = conn.get_balances(nonzero=False)
        self.assertEqual(len(balances), 3)

    def test_list_open_orders(self) -> None:
        conn, http = _connected()
        http.route("GET", "/api/v3/openOrders", FakeResponse(200, [
            {"orderId": 1, "symbol": "BTCUSDT", "side": "BUY"},
        ]))
        orders = conn.list_open_orders("BTCUSDT")
        self.assertEqual(orders[0]["orderId"], 1)
        _m, url, _p, _h = http.calls[-1]
        self.assertIn("symbol=BTCUSDT", url)
        self.assertIn("signature=", url)

    def test_get_order(self) -> None:
        conn, http = _connected()
        http.route("GET", "/api/v3/order", FakeResponse(200, {
            "orderId": 1, "status": "FILLED",
        }))
        result = conn.get_order("BTCUSDT", order_id=1)
        self.assertEqual(result["status"], "FILLED")

    def test_get_order_needs_identifier(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.get_order("BTCUSDT")


class OrderTests(unittest.TestCase):
    def test_place_order_requires_confirmation(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.place_order("BTCUSDT", "BUY", "LIMIT", 0.01, price=60000)
        self.assertIn("confirmed=True", str(ctx.exception))

    def test_place_order_validates(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.place_order("BTCUSDT", "HOLD", "LIMIT", 0.01, price=1,
                             confirmed=True)
        with self.assertRaises(ConnectorError):
            conn.place_order("BTCUSDT", "BUY", "NOPE", 0.01,
                             confirmed=True)
        with self.assertRaises(ConnectorError):
            conn.place_order("BTCUSDT", "BUY", "LIMIT", 0, price=1,
                             confirmed=True)
        with self.assertRaises(ConnectorError):
            conn.place_order("BTCUSDT", "BUY", "LIMIT", 0.01,
                             confirmed=True)  # no price
        with self.assertRaises(ConnectorError):
            conn.place_order("BTCUSDT", "BUY", "STOP_LOSS", 0.01,
                             confirmed=True)  # no stop price

    def test_place_limit_order_confirmed(self) -> None:
        conn, http = _connected()
        http.route("POST", "/api/v3/order", FakeResponse(200, {
            "orderId": 101, "symbol": "BTCUSDT", "status": "NEW",
        }))
        result = conn.place_order(
            "BTCUSDT", "BUY", "LIMIT", 0.01, price=60000, confirmed=True
        )
        self.assertEqual(result["orderId"], 101)
        _m, url, _p, headers = http.calls[-1]
        self.assertIn("/api/v3/order", url)
        _qs, params = _signed_parts(url)
        self.assertEqual(params["symbol"], "BTCUSDT")
        self.assertEqual(params["side"], "BUY")
        self.assertEqual(params["type"], "LIMIT")
        self.assertEqual(params["timeInForce"], "GTC")
        self.assertEqual(headers["X-MBX-APIKEY"], "APIKEY123")

    def test_place_market_order_no_time_in_force(self) -> None:
        conn, http = _connected()
        http.route("POST", "/api/v3/order", FakeResponse(200, {
            "orderId": 102, "status": "FILLED",
        }))
        conn.place_order("BTCUSDT", "SELL", "MARKET", 0.01, confirmed=True)
        _m, url, _p, _h = http.calls[-1]
        _qs, params = _signed_parts(url)
        self.assertNotIn("timeInForce", params)
        self.assertNotIn("price", params)

    def test_place_order_via_human_checkpoint(self) -> None:
        conn, http = _connected()
        db = _db()
        with self.assertRaises(HumanCheckpointPending) as ctx:
            conn.place_order("BTCUSDT", "BUY", "LIMIT", 0.01, price=60000,
                             db=db)
        store = CheckpointStore(db)
        cp = store.get(ctx.exception.checkpoint.id)
        self.assertIn("BTCUSDT", cp.instructions)
        http.route("POST", "/api/v3/order", FakeResponse(200, {
            "orderId": 103, "status": "NEW",
        }))
        store.resolve(cp.id, note="approved")
        result = conn.resume_checkpoint(store.get(cp.id), db=db)
        self.assertEqual(result["orderId"], 103)

    def test_place_order_resume_unresolved_raises(self) -> None:
        conn, _http = _connected()
        db = _db()
        with self.assertRaises(HumanCheckpointPending) as ctx:
            conn.place_order("BTCUSDT", "BUY", "MARKET", 0.01, db=db)
        store = CheckpointStore(db)
        with self.assertRaises(ConnectorError):
            conn.resume_checkpoint(
                store.get(ctx.exception.checkpoint.id), db=db
            )

    def test_cancel_order_requires_confirmation(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.cancel_order("BTCUSDT", order_id=101)
        self.assertIn("confirmed=True", str(ctx.exception))

    def test_cancel_order_confirmed(self) -> None:
        conn, http = _connected()
        http.route("DELETE", "/api/v3/order", FakeResponse(200, {
            "orderId": 101, "status": "CANCELED",
        }))
        result = conn.cancel_order("BTCUSDT", order_id=101, confirmed=True)
        self.assertEqual(result["status"], "CANCELED")
        _m, url, _p, _h = http.calls[-1]
        self.assertIn("orderId=101", url)
        self.assertIn("signature=", url)

    def test_cancel_needs_identifier(self) -> None:
        conn, _http = _connected()
        with self.assertRaises(ConnectorError):
            conn.cancel_order("BTCUSDT", confirmed=True)

    def test_binance_error_detail(self) -> None:
        conn, http = _connected()
        http.route("POST", "/api/v3/order", FakeResponse(400, {
            "code": -2010, "msg": "Account has insufficient balance.",
        }))
        with self.assertRaises(BinanceError) as ctx:
            conn.place_order("BTCUSDT", "BUY", "MARKET", 999,
                             confirmed=True)
        self.assertEqual(ctx.exception.binance_code, -2010)
        self.assertIn("insufficient balance", str(ctx.exception))

    def test_rate_limit_error(self) -> None:
        conn, http = _connected()
        http.routes.clear()  # drop the connect() validation route
        http.route("GET", "/api/v3/account", FakeResponse(
            429, {"code": -1003, "msg": "Too many requests"}))
        with self.assertRaises(BinanceError) as ctx:
            conn.get_balances()
        self.assertEqual(ctx.exception.status_code, 429)


if __name__ == "__main__":
    unittest.main()
