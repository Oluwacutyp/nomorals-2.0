"""Jumia seller connector: offline tests against a scripted HTTP stand-in.

Everything runs against FakeHttp — no real network, no real Jumia
credentials. The fake pins down the contract this connector was built
against (Jumia Vendor Center API, GOP.pdf / GPM.pdf, May 2023):

* auth: POST /token as x-www-form-urlencoded with
  client_id / grant_type=refresh_token / refresh_token
* refresh rotation: each /token response's fresh refresh token is
  re-vaulted
* lazy refresh: calls reuse the cached access token until near expiry;
  a 401 triggers exactly one refresh + retry
* repeated query params: /orders/items?orderId=A&orderId=B,
  /orders/shipment-providers?orderItemId=A&orderItemId=B
* mutating payloads: PUT /orders/cancel {"orderItemIds": [...]},
  POST v2/orders/pack {"packages": [...]}, POST /orders/ready-to-ship
  {"orderItemIds": [...]}, POST /feeds/products/{kind}
* 429 maps to a clear rate-limit error
"""

from __future__ import annotations

import json
import os
import time
import unittest
from typing import Any
from unittest import mock

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors.jumia import (
    JUMIA_API_BASE,
    FEED_KINDS,
    JumiaConnector,
    JumiaError,
)
from nomorals.connectors import ConnectorError
from nomorals.connectors.checkpoints import (
    CheckpointKind,
    CheckpointState,
    CheckpointStore,
    HumanCheckpointPending,
    request_human_action,
)
from nomorals.storage.db import Database


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test")


class FakeResponse:
    def __init__(
        self,
        status: int = 200,
        payload: Any = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self.headers = headers or {}
        self._payload = payload if payload is not None else {}

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    @property
    def text(self) -> str:
        return json.dumps(self._payload)

    def json(self) -> Any:
        return self._payload


class FakeHttp:
    """Scripted stand-in for HttpClient: routes (method, url) -> response."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.routes: dict[tuple[str, str], FakeResponse] = {}
        self.default = FakeResponse(200, {})

    def route(self, method: str, url: str, resp: FakeResponse) -> None:
        self.routes[(method.upper(), url)] = resp

    def _serve(self, method: str, url: str, **kw: Any) -> FakeResponse:
        self.calls.append({"method": method.upper(), "url": url, **kw})
        return self.routes.get((method.upper(), url), self.default)

    def get(self, url: str, **kw: Any) -> FakeResponse:
        return self._serve("GET", url, **kw)

    def post_json(self, url: str, payload: Any, **kw: Any) -> FakeResponse:
        return self._serve("POST", url, payload=payload, **kw)

    def post_form(self, url: str, form: Any, **kw: Any) -> FakeResponse:
        return self._serve("POST", url, form=form, **kw)

    def request(self, method: str, url: str, **kw: Any) -> FakeResponse:
        return self._serve(method, url, **kw)

    def last(self, method: str, url: str) -> dict[str, Any] | None:
        for call in reversed(self.calls):
            if call["method"] == method.upper() and call["url"] == url:
                return call
        return None


_TOKENS = {
    "access_token": "access-123",
    "expires_in": 3600,
    "refresh_token": "refresh-456",
    "refresh_expires_in": 31536000,
    "token_type": "bearer",
}

_SHOPS = {"shops": [{"id": "shop-1", "name": "Devon Demo Store"}]}


def _token_route(http: FakeHttp, tokens: dict[str, Any] | None = None) -> None:
    http.route("POST", f"{JUMIA_API_BASE}/token",
               FakeResponse(200, tokens or dict(_TOKENS)))


def _connect(http: FakeHttp, vault: CredentialVault,
             tokens: dict[str, Any] | None = None) -> JumiaConnector:
    _token_route(http, tokens)
    http.route("GET", f"{JUMIA_API_BASE}/shops", FakeResponse(200, _SHOPS))
    conn = JumiaConnector(vault, http=http)
    conn.connect(client_id="client-1", refresh_token="refresh-0")
    return conn


class JumiaConnectTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vault = _vault()
        self.http = FakeHttp()

    def test_connect_mints_tokens_and_vaults_rotated_refresh(self) -> None:
        conn = _connect(self.http, self.vault)
        token_call = self.http.last("POST", f"{JUMIA_API_BASE}/token")
        self.assertIsNotNone(token_call)
        form = token_call["form"]
        self.assertEqual(form["grant_type"], "refresh_token")
        self.assertEqual(form["client_id"], "client-1")
        self.assertEqual(form["refresh_token"], "refresh-0")
        # The rotated refresh token is what lands in the vault.
        cred = conn._load_credential()
        self.assertIsNotNone(cred)
        assert cred is not None
        self.assertEqual(cred.username, "client-1")
        self.assertEqual(cred.password, "refresh-456")
        self.assertEqual(cred.credential_type, "oauth_token")
        self.assertEqual(cred.metadata["access_token"], "access-123")

    def test_connect_reports_shop_name_as_account(self) -> None:
        _token_route(self.http)
        self.http.route("GET", f"{JUMIA_API_BASE}/shops",
                        FakeResponse(200, _SHOPS))
        conn = JumiaConnector(self.vault, http=self.http)
        result = conn.connect(client_id="client-1", refresh_token="refresh-0")
        self.assertTrue(result.ok)
        self.assertEqual(result.account, "Devon Demo Store")

    def test_connect_falls_back_to_client_id_when_shops_fail(self) -> None:
        _token_route(self.http)
        self.http.route("GET", f"{JUMIA_API_BASE}/shops",
                        FakeResponse(500, {"message": "boom"}))
        conn = JumiaConnector(self.vault, http=self.http)
        result = conn.connect(client_id="client-9", refresh_token="refresh-0")
        self.assertEqual(result.account, "client-9")

    def test_connect_rejects_invalid_grant(self) -> None:
        self.http.route(
            "POST", f"{JUMIA_API_BASE}/token",
            FakeResponse(400, {"error": "invalid_grant",
                               "error_description": "Invalid refresh token"}),
        )
        conn = JumiaConnector(self.vault, http=self.http)
        with self.assertRaises(JumiaError) as ctx:
            conn.connect(client_id="client-1", refresh_token="stale")
        self.assertIn("refresh token", str(ctx.exception).lower())
        self.assertIsNone(conn._load_credential())

    def test_connect_fails_fast_without_credentials_non_tty(self) -> None:
        for var in ("JUMIA_CLIENT_ID", "JUMIA_REFRESH_TOKEN"):
            os.environ.pop(var, None)
        conn = JumiaConnector(self.vault, http=self.http)
        with mock.patch("sys.stdin.isatty", return_value=False):
            with self.assertRaises(ConnectorError):
                conn.connect()

    def test_connect_uses_env_credentials(self) -> None:
        _token_route(self.http)
        self.http.route("GET", f"{JUMIA_API_BASE}/shops",
                        FakeResponse(200, _SHOPS))
        conn = JumiaConnector(self.vault, http=self.http)
        with mock.patch.dict(os.environ, {"JUMIA_CLIENT_ID": "env-client",
                                          "JUMIA_REFRESH_TOKEN": "env-refresh"}):
            result = conn.connect()
        self.assertTrue(result.ok)
        token_call = self.http.last("POST", f"{JUMIA_API_BASE}/token")
        self.assertEqual(token_call["form"]["client_id"], "env-client")

    def test_disconnect_is_idempotent(self) -> None:
        conn = _connect(self.http, self.vault)
        conn.disconnect()
        self.assertIsNone(conn._load_credential())
        conn.disconnect()  # second time is fine
        self.assertFalse(conn.test_connection())

    def test_test_connection_true_when_shops_ok(self) -> None:
        conn = _connect(self.http, self.vault)
        self.assertTrue(conn.test_connection())

    def test_test_connection_false_without_credential(self) -> None:
        conn = JumiaConnector(self.vault, http=self.http)
        self.assertFalse(conn.test_connection())

    def test_status_not_connected(self) -> None:
        conn = JumiaConnector(self.vault, http=self.http)
        status = conn.status()
        self.assertFalse(status.connected)
        self.assertIn("not connected", status.detail)

    def test_status_connected(self) -> None:
        conn = _connect(self.http, self.vault)
        status = conn.status()
        self.assertTrue(status.connected)
        self.assertEqual(status.account, "Devon Demo Store")

    def test_status_false_after_refresh_rejected(self) -> None:
        conn = _connect(self.http, self.vault)
        # Simulate: access token expired AND refresh token dead.
        self.http.route("GET", f"{JUMIA_API_BASE}/shops",
                        FakeResponse(401, {"message": "expired"}))
        self.http.route(
            "POST", f"{JUMIA_API_BASE}/token",
            FakeResponse(400, {"error": "invalid_grant",
                               "error_description": "Invalid refresh token"}),
        )
        status = conn.status()
        self.assertFalse(status.connected)
        self.assertIn("refresh token", status.detail.lower())


class JumiaRefreshTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vault = _vault()
        self.http = FakeHttp()

    def test_access_token_reused_until_near_expiry(self) -> None:
        conn = _connect(self.http, self.vault)
        calls_before = len(self.http.calls)
        conn.list_shops()
        conn.list_shops()
        token_calls = [c for c in self.http.calls[calls_before:]
                       if c["url"].endswith("/token")]
        self.assertEqual(token_calls, [])

    def test_refresh_happens_when_access_expired(self) -> None:
        conn = _connect(self.http, self.vault)
        cred = conn._load_credential()
        assert cred is not None
        # Age the cached access token past expiry.
        cred.metadata["access_expires_at"] = time.time() - 10
        self.vault.store(service="connector:jumia", username=cred.username,
                         password=cred.password,
                         credential_type="oauth_token",
                         tags=["connector", "jumia"], metadata=cred.metadata)
        rotated = dict(_TOKENS, access_token="access-2",
                       refresh_token="refresh-2")
        self.http.route("POST", f"{JUMIA_API_BASE}/token",
                        FakeResponse(200, rotated))
        conn.list_shops()
        cred2 = conn._load_credential()
        assert cred2 is not None
        self.assertEqual(cred2.password, "refresh-2")
        self.assertEqual(cred2.metadata["access_token"], "access-2")

    def test_401_triggers_one_refresh_and_retry(self) -> None:
        conn = _connect(self.http, self.vault)
        # First /shops call 401s; after refresh the retry succeeds.
        seen = {"n": 0}

        orig = conn._raw_api

        def flaky(method, path, payload, token, **kw):
            if path == "/shops" and seen["n"] == 0:
                seen["n"] += 1
                raise JumiaError("expired", status_code=401)
            return orig(method, path, payload, token, **kw)

        rotated = dict(_TOKENS, access_token="access-fresh",
                       refresh_token="refresh-fresh")
        self.http.route("POST", f"{JUMIA_API_BASE}/token",
                        FakeResponse(200, rotated))
        with mock.patch.object(conn, "_raw_api", side_effect=flaky):
            shops = conn.list_shops()
        self.assertEqual(shops[0]["name"], "Devon Demo Store")
        cred = conn._load_credential()
        assert cred is not None
        self.assertEqual(cred.password, "refresh-fresh")

    def test_401_after_refresh_raises_clear_error(self) -> None:
        conn = _connect(self.http, self.vault)

        def always_401(method, path, payload, token, **kw):
            raise JumiaError("nope", status_code=401)

        self.http.route("POST", f"{JUMIA_API_BASE}/token",
                        FakeResponse(200, dict(_TOKENS)))
        with mock.patch.object(conn, "_raw_api", side_effect=always_401):
            with self.assertRaises(JumiaError) as ctx:
                conn.list_shops()
        self.assertIn("refresh token", str(ctx.exception).lower())


class JumiaShopsOrdersTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vault = _vault()
        self.http = FakeHttp()
        self.conn = _connect(self.http, self.vault)

    def _url_of(self, path: str) -> str:
        return f"{JUMIA_API_BASE}{path}"

    def test_list_shops_master_flag(self) -> None:
        self.conn.list_shops(master_shop=True)
        call = self.http.last("GET", self._url_of("/shops-of-master-shop"))
        self.assertIsNotNone(call)
        self.assertIn("Bearer access-123",
                      call["headers"]["Authorization"])

    def test_list_orders_params_and_pagination_shape(self) -> None:
        self.http.route(
            "GET", self._url_of("/orders"),
            FakeResponse(200, {"orders": [{"id": "o1"}],
                               "nextToken": "tok-9", "isLastPage": False}),
        )
        out = self.conn.list_orders(status="PENDING,SHIPPED", country="NG",
                                    size=50)
        self.assertEqual(out["orders"], [{"id": "o1"}])
        self.assertEqual(out["nextToken"], "tok-9")
        self.assertFalse(out["isLastPage"])
        call = self.http.last("GET", self._url_of("/orders"))
        params = call["params"]
        self.assertEqual(params["status"], "PENDING,SHIPPED")
        self.assertEqual(params["country"], "NG")
        self.assertEqual(params["size"], 50)

    def test_list_orders_rejects_unknown_status(self) -> None:
        with self.assertRaises(JumiaError) as ctx:
            self.conn.list_orders(status="BOGUS")
        self.assertIn("unknown order status", str(ctx.exception))

    def test_list_orders_date_filters(self) -> None:
        self.http.route("GET", self._url_of("/orders"),
                        FakeResponse(200, {"orders": []}))
        self.conn.list_orders(created_after="2026-09-01 00:00:00",
                              created_before="2026-09-30 23:59:59")
        call = self.http.last("GET", self._url_of("/orders"))
        self.assertEqual(call["params"]["createdAfter"], "2026-09-01 00:00:00")
        self.assertEqual(call["params"]["createdBefore"],
                         "2026-09-30 23:59:59")

    def test_get_order_items_uses_repeated_orderId(self) -> None:
        self.http.route("GET", self._url_of("/orders/items"),
                        FakeResponse(200, [{"orderId": "a"}]))
        items = self.conn.get_order_items(["a", "b"], status="PENDING")
        self.assertEqual(items, [{"orderId": "a"}])
        call = self.http.last("GET", self._url_of("/orders/items"))
        pairs = call["params"]
        self.assertEqual(pairs.count(("orderId", "a")), 1)
        self.assertEqual(pairs.count(("orderId", "b")), 1)
        self.assertIn(("status", "PENDING"), pairs)

    def test_get_order_items_needs_ids(self) -> None:
        with self.assertRaises(JumiaError):
            self.conn.get_order_items([])

    def test_get_shipment_providers_uses_repeated_orderItemId(self) -> None:
        self.http.route(
            "GET", self._url_of("/orders/shipment-providers"),
            FakeResponse(200, {"orderItems": [{"id": "i1"}]}),
        )
        out = self.conn.get_shipment_providers(["i1", "i2"])
        self.assertEqual(out, [{"id": "i1"}])
        call = self.http.last("GET",
                              self._url_of("/orders/shipment-providers"))
        pairs = call["params"]
        self.assertEqual(pairs.count(("orderItemId", "i1")), 1)
        self.assertEqual(pairs.count(("orderItemId", "i2")), 1)

    def test_429_maps_to_rate_limit_error(self) -> None:
        self.http.route("GET", self._url_of("/orders"),
                        FakeResponse(429, {"message": "slow down"}))
        with self.assertRaises(JumiaError) as ctx:
            self.conn.list_orders()
        self.assertEqual(ctx.exception.status_code, 429)
        self.assertIn("rate limit", str(ctx.exception).lower())


class JumiaFulfilmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vault = _vault()
        self.http = FakeHttp()
        self.conn = _connect(self.http, self.vault)

    def _url_of(self, path: str) -> str:
        return f"{JUMIA_API_BASE}{path}"

    def test_cancel_orders_puts_orderItemIds(self) -> None:
        self.http.route(
            "PUT", self._url_of("/orders/cancel"),
            FakeResponse(200, {"success": {"total": 2},
                               "error": {"total": 0}}),
        )
        out = self.conn.cancel_orders(["i1", "i2"])
        self.assertEqual(out["success"]["total"], 2)
        call = self.http.last("PUT", self._url_of("/orders/cancel"))
        body = json.loads(call["data"].decode("utf-8"))
        self.assertEqual(body, {"orderItemIds": ["i1", "i2"]})
        self.assertIn("application/json", call["headers"]["Content-Type"])

    def test_cancel_orders_needs_ids(self) -> None:
        with self.assertRaises(JumiaError):
            self.conn.cancel_orders([])

    def test_create_package_v2_default(self) -> None:
        self.http.route("POST", self._url_of("v2/orders/pack"),
                        FakeResponse(201, {"success": {"total": 1}}))
        out = self.conn.create_package([
            {"orderItems": ["i1"], "shipmentProviderId": "sp1",
             "trackingCode": "EJ833555375"},
        ])
        self.assertEqual(out["success"]["total"], 1)
        call = self.http.last("POST", self._url_of("v2/orders/pack"))
        self.assertEqual(call["payload"]["packages"][0]["trackingCode"],
                         "EJ833555375")

    def test_create_package_v1_shape(self) -> None:
        self.http.route("POST", self._url_of("/orders/pack"),
                        FakeResponse(200, {"success": {"total": 1}}))
        self.conn.create_package(
            [{"orderItems": ["i1"], "shipmentProviderId": "sp1"}],
            api_version=1,
        )
        call = self.http.last("POST", self._url_of("/orders/pack"))
        self.assertEqual(call["payload"]["orderItems"],
                         [{"id": "i1", "shipmentProviderId": "sp1"}])

    def test_create_package_rejects_bad_version(self) -> None:
        with self.assertRaises(JumiaError):
            self.conn.create_package([{}], api_version=3)

    def test_mark_ready_to_ship(self) -> None:
        self.http.route("POST", self._url_of("/orders/ready-to-ship"),
                        FakeResponse(200, {"success": {"total": 1}}))
        self.conn.mark_ready_to_ship(["i1"])
        call = self.http.last("POST", self._url_of("/orders/ready-to-ship"))
        self.assertEqual(call["payload"], {"orderItemIds": ["i1"]})

    def test_print_labels(self) -> None:
        self.http.route("POST", self._url_of("/orders/print-labels"),
                        FakeResponse(200, {"labels": ["pdf-bytes"]}))
        self.conn.print_labels({"orderItemIds": ["i1"]})
        call = self.http.last("POST", self._url_of("/orders/print-labels"))
        self.assertEqual(call["payload"], {"orderItemIds": ["i1"]})


class JumiaCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vault = _vault()
        self.http = FakeHttp()
        self.conn = _connect(self.http, self.vault)

    def _url_of(self, path: str) -> str:
        return f"{JUMIA_API_BASE}{path}"

    def test_list_products_by_seller_sku(self) -> None:
        self.http.route("GET", self._url_of("/catalog/products"),
                        FakeResponse(200, {"products": [{"sellerSku": "S"}]}))
        out = self.conn.list_products(sellerSku="SKU-001")
        self.assertEqual(out, [{"sellerSku": "S"}])
        call = self.http.last("GET", self._url_of("/catalog/products"))
        self.assertEqual(call["params"], {"sellerSku": "SKU-001"})

    def test_get_brands(self) -> None:
        self.http.route("GET", self._url_of("/catalog/brands"),
                        FakeResponse(200, {"brands": [{"name": "Nike"}]}))
        self.assertEqual(self.conn.get_brands(), [{"name": "Nike"}])

    def test_get_categories(self) -> None:
        self.http.route("GET", self._url_of("/catalog/categories"),
                        FakeResponse(200, {"categories": [{"id": 1}]}))
        self.assertEqual(self.conn.get_categories(), [{"id": 1}])

    def test_get_attribute_set(self) -> None:
        self.http.route("GET", self._url_of("/catalog/attribute-sets/9"),
                        FakeResponse(200, {"id": 9}))
        self.assertEqual(self.conn.get_attribute_set("9"), {"id": 9})
        with self.assertRaises(JumiaError):
            self.conn.get_attribute_set("")

    def test_submit_feed_validates_kind(self) -> None:
        for kind in FEED_KINDS:
            self.http.route("POST", self._url_of(f"/feeds/products/{kind}"),
                            FakeResponse(200, {"feedId": "f1"}))
            out = self.conn.submit_feed(kind, {"rows": []})
            self.assertEqual(out["feedId"], "f1")
        with self.assertRaises(JumiaError):
            self.conn.submit_feed("bogus", {"rows": []})
        with self.assertRaises(JumiaError):
            self.conn.submit_feed("price", {})

    def test_feed_status(self) -> None:
        self.http.route("GET", self._url_of("/feeds/f1"),
                        FakeResponse(200, {"status": "done"}))
        self.assertEqual(self.conn.feed_status("f1"), {"status": "done"})
        with self.assertRaises(JumiaError):
            self.conn.feed_status("")


class JumiaProvisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vault = _vault()
        self.http = FakeHttp()
        self.conn = JumiaConnector(self.vault, http=self.http)

    def test_unknown_kind_raises(self) -> None:
        with self.assertRaises(ConnectorError):
            self.conn.provision("webhook")

    def test_seller_account_needs_db(self) -> None:
        with self.assertRaises(ConnectorError) as ctx:
            self.conn.provision("jumia_seller_account")
        self.assertIn("database", str(ctx.exception).lower())

    def test_seller_account_refuses_while_connected(self) -> None:
        conn = _connect(self.http, self.vault)
        with self.assertRaises(ConnectorError) as ctx:
            conn.provision("jumia_seller_account", db=object())
        self.assertIn("one account per service", str(ctx.exception))

    def test_seller_account_flow_resolves_to_connect_instructions(self) -> None:
        db = Database(":memory:")
        # Non-interactive: the human step pauses with a pending checkpoint.
        with self.assertRaises(HumanCheckpointPending) as ctx:
            request_human_action(
                "jumia", CheckpointKind.MANUAL_STEP,
                "Create your Jumia seller account", "do the steps",
                db=db, resume_state={"stage": "signup"},
            )
        cp = ctx.exception.checkpoint
        self.assertEqual(cp.state, CheckpointState.PENDING)
        CheckpointStore(db).resolve(cp.id)
        resolved = CheckpointStore(db).get(cp.id)
        out = self.conn.resume_checkpoint(resolved, db=db)
        self.assertTrue(out["done"])
        self.assertIn("Self Authorization", out["message"])
        self.assertIn("JUMIA_REFRESH_TOKEN", out["message"])

    def test_resume_unresolved_checkpoint_raises(self) -> None:
        db = Database(":memory:")
        with self.assertRaises(HumanCheckpointPending) as ctx:
            request_human_action(
                "jumia", CheckpointKind.MANUAL_STEP,
                "title", "instructions", db=db,
                resume_state={"stage": "signup"},
            )
        cp = ctx.exception.checkpoint
        with self.assertRaises(ConnectorError):
            self.conn.resume_checkpoint(cp, db=db)

    def test_provision_end_to_end_interactive(self) -> None:
        db = Database(":memory:")
        # Non-interactive: provisioning raises HumanCheckpointPending.
        with self.assertRaises(HumanCheckpointPending):
            self.conn.provision("jumia_seller_account", db=db)
        # Resolve the stored checkpoint and finish the flow.
        store = CheckpointStore(db)
        pending = store.list_pending("jumia")
        self.assertEqual(len(pending), 1)
        store.resolve(pending[0].id)
        out = self.conn.resume_checkpoint(store.get(pending[0].id), db=db)
        self.assertTrue(out["done"])
        self.assertIn("Self Authorization", out["message"])


class JumiaHonestyTests(unittest.TestCase):
    """The connector must stay honest about what Jumia exposes."""

    def setUp(self) -> None:
        self.vault = _vault()
        self.http = FakeHttp()

    def test_description_admits_no_buyer_api(self) -> None:
        desc = JumiaConnector.description.lower()
        self.assertIn("no buyer", desc)

    def test_no_buyer_or_consumer_shopping_methods(self) -> None:
        buyer_words = ("browse", "cart", "checkout", "place_order",
                       "search_products", "buy")
        for name in dir(JumiaConnector):
            self.assertNotIn(name.lower().replace("_", ""), 
                             [w.replace("_", "") for w in buyer_words],
                             f"suspicious buyer-ish method: {name}")

    def test_connect_instructions_names_self_authorization(self) -> None:
        conn = JumiaConnector(self.vault, http=self.http)
        text = conn.connect_instructions()
        self.assertIn("Self Authorization", text)
        self.assertIn("JUMIA_REFRESH_TOKEN", text)

    def test_429_message_mentions_documented_limits(self) -> None:
        conn = _connect(self.http, self.vault)
        self.http.route("GET", f"{JUMIA_API_BASE}/orders",
                        FakeResponse(429, {}))
        with self.assertRaises(JumiaError) as ctx:
            conn.list_orders()
        self.assertIn("200", str(ctx.exception))

    def test_capabilities_documents_seller_only_contract(self) -> None:
        conn = JumiaConnector(self.vault, http=self.http)
        caps = conn.capabilities()
        self.assertEqual(caps["side"], "seller")
        self.assertIn("orders", caps["can"])
        self.assertIn("catalog", caps["can"])
        self.assertIn("feeds", caps["can"])
        # the hard API limitation, stated plainly
        self.assertIn("buyer_product_search", caps["cannot"])
        self.assertIn("no buyer product search api",
                      caps["cannot"]["buyer_product_search"].lower()
                      .replace("-", " "))
        self.assertIn("checkout", caps["cannot"]["buyer_checkout"])


if __name__ == "__main__":
    unittest.main()
