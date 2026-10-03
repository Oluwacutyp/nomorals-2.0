"""Jumia connector — seller-side marketplace operations via the Vendor Center API.

What exists (official and documented):

* **Vendor Center (VC) API** at ``https://vendor-api.jumia.com`` — the
  programmatic interface for Jumia sellers. Two functional interfaces:
  GOP (Global Order Processing: orders/shops/shipment) and GPM (Global
  Product Management: catalog/feeds). Documented by Jumia's
  ``jumia-global.com`` college article (GOP.pdf / GPM.pdf, May 2023).
* Auth is OAuth2, machine-to-machine ("Self Authorization"): the owner
  creates an application in the Vendor Center UI
  (vendorcenter.jumia.com -> Settings -> Manage Applications -> Create
  Application -> Self Authorization) and generates a long-term refresh
  token (1 year). The connector exchanges it at ``POST /token``
  (``grant_type=refresh_token``) for a short-term access token and a
  *fresh* refresh token, which it rotates and re-vaults each refresh.
* Rate limits: 200 requests/minute, 4 requests/second per mastershop
  (HTTP 429 when exceeded).

What does NOT exist (so this connector honestly cannot do):

* **No buyer-side API.** Jumia publishes no public product-search,
  cart, or order-placement API. This connector cannot browse
  jumia.com.ng as a shopper or place orders as a buyer — those flows are
  web/mobile-app only and intentionally undocumented.
* The old per-country Seller Center API is deprecated (no technical
  support; old sellers only) — this connector targets the VC API.
* Consignment management (GPM ``/consignment-order``) is FBJ-shop only
  and is not implemented here yet.

Mutating endpoints (cancel, pack, ready-to-ship, print-labels, feeds)
take their request payloads as caller-supplied dicts whose shapes are
defined by the GOP/GPM PDF documentation — the connector validates the
envelope (required keys) but does not invent schema.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from ..core.logging_setup import get_logger
from .auth import prompt_secret
from .base import (
    AuthMethod,
    Connector,
    ConnectorError,
    ConnectorStatus,
    ConnectResult,
)
from .registry import register_connector

__all__ = ["JumiaConnector", "JumiaError", "JUMIA_API_BASE"]

_log = get_logger(__name__)

#: Vendor Center API base (same host for GOP and GPM interfaces).
JUMIA_API_BASE = "https://vendor-api.jumia.com"

#: Where the owner creates the self-authorization application and mints
#: the refresh token.
VENDOR_CENTER_URL = "https://vendorcenter.jumia.com"

#: Item statuses accepted by /orders and /orders/items (GOP docs).
ORDER_ITEM_STATUSES = (
    "PENDING",
    "SHIPPED",
    "CANCELED",
    "RETURNED",
    "FAILED",
    "DELIVERED",
    "READY_TO_SHIP",
)

#: Feed kinds for POST /feeds/products/{kind} (GPM docs).
FEED_KINDS = ("create", "update", "stock", "price", "status")

#: Seconds before access-token expiry at which we proactively refresh.
_REFRESH_LEEWAY = 300.0


class JumiaError(ConnectorError):
    """A Jumia VC API call failed."""

    def __init__(self, message: str, *, status_code: int = 0) -> None:
        super().__init__(message)
        self.status_code = status_code


@register_connector
class JumiaConnector(Connector):
    """Devon's Jumia adapter: seller shops, orders, shipment, catalog.

    Seller-side only. Requires a Jumia seller account with a
    Self Authorization application in the Vendor Center; connects with
    the application's ``client_id`` plus the long-term refresh token the
    owner generates in the UI.
    """

    id = "jumia"
    name = "Jumia"
    description = (
        "Seller-side Jumia marketplace operations via the official "
        "Vendor Center API: list shops, fetch and process orders "
        "(items, pack, ready-to-ship, print labels, cancel), and manage "
        "the product catalog (brands, categories, products, "
        "price/stock/status feeds). Requires a Jumia seller account — "
        "no buyer APIs exist."
    )
    auth_methods = (AuthMethod.OAUTH2,)
    PROVISIONABLE = ("jumia_seller_account",)

    # ── lifecycle ────────────────────────────────────────────────

    def connect(
        self,
        *,
        client_id: str | None = None,
        refresh_token: str | None = None,
    ) -> ConnectResult:
        """Validate client_id + refresh token, mint tokens, vault-store.

        The refresh token is long-lived (1 year); the connector keeps it
        in the encrypted vault and exchanges it for short-term access
        tokens on demand, rotating it each refresh per Jumia's best
        practice.
        """
        cid = (client_id or "").strip()
        rtoken = (refresh_token or "").strip()
        if not cid and not rtoken:
            print(self.connect_instructions())
        cid = cid or prompt_secret(
            "Jumia VC application client id", env_var="JUMIA_CLIENT_ID"
        )
        rtoken = rtoken or prompt_secret(
            "Jumia VC refresh token", env_var="JUMIA_REFRESH_TOKEN"
        )
        if not cid or not rtoken:
            raise ConnectorError(
                "empty client id or refresh token: nothing to connect with"
            )
        tokens = self._mint_tokens(cid, rtoken)
        access = str(tokens["access_token"])
        if not access:
            raise JumiaError(
                "Jumia did not return an access token for this refresh token"
            )
        account = self._account_name(cid, access)
        self._store_tokens(cid, tokens, account=account)
        creds = self._load_credential()
        _log.info("jumia connected for client %s (%s)", cid, account)
        return ConnectResult(
            ok=True,
            account=account,
            scopes=[],
            message=(
                f"connected to Jumia as {account}. The refresh token lives "
                "in the encrypted vault; regenerate it in the Vendor "
                "Center (Settings -> Manage Applications) if it ever "
                "needs revoking. Note: this connector is seller-side "
                "only — Jumia offers no buyer API."
            ),
            credential_id=creds.id if creds else 0,
        )

    def disconnect(self) -> None:
        self._clear_credential()

    def status(self) -> ConnectorStatus:
        cred = self._load_credential()
        if cred is None:
            return ConnectorStatus(
                connected=False,
                detail="not connected — run `nm connectors connect --name jumia`",
            )
        try:
            shops = self._api("GET", "/shops")
            account = self._account_name_from_shops(shops, cred.username)
        except JumiaError as exc:
            return ConnectorStatus(
                connected=False,
                account=cred.username,
                last_checked=time.time(),
                detail=(
                    f"Jumia rejected the credentials ({exc}): generate a "
                    "fresh refresh token in the Vendor Center UI and "
                    "reconnect"
                ),
            )
        return ConnectorStatus(
            connected=True,
            account=account,
            last_checked=time.time(),
            detail="credentials valid",
        )

    def test_connection(self) -> bool:
        cred = self._load_credential()
        if cred is None:
            return False
        try:
            self._api("GET", "/shops")
            return True
        except ConnectorError:
            return False

    # ── provisioning ───────────────────────────────────────────

    def provision(self, kind: str, **kwargs: Any) -> dict[str, Any]:
        if kind == "jumia_seller_account":
            return self._provision_seller_account(**kwargs)
        raise ConnectorError(
            f"jumia cannot provision {kind!r} "
            f"(provisionable: {', '.join(self.PROVISIONABLE)})"
        )

    # ── guided seller-account creation (human-in-the-loop) ───────

    def _provision_seller_account(
        self,
        *,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Guide the owner through creating their own Jumia seller account.

        Jumia seller signup requires email verification and document
        approval — human steps only. Devon prepares the checklist, then
        pauses so the owner completes the human parts personally. One
        account per service: refuses while a credential is stored.
        """
        from .checkpoints import CheckpointKind

        if db is None:
            raise ConnectorError(
                "jumia_seller_account provisioning needs a database for "
                "checkpoints (pass db=)"
            )
        existing = self._load_credential()
        if existing is not None:
            raise ConnectorError(
                f"jumia is already connected as {existing.username} — "
                "one account per service. Disconnect first to switch."
            )
        cp = self.request_human(
            CheckpointKind.MANUAL_STEP,
            "Create your Jumia seller account",
            "\n".join([
                "Devon cannot sign up for you — Jumia seller registration "
                "needs your own identity. The human part:",
                f"1. Open {VENDOR_CENTER_URL} and click 'Sell on Jumia'.",
                "2. Choose your country (e.g. Nigeria) and verify your "
                "email with the code Jumia sends you.",
                "3. Enter your phone number and set your password.",
                "4. Register as an individual or business — as a business "
                "you'll need your CAC/TIN documents.",
                "5. Add your shop name and shipping zone, agree to the "
                "terms, and submit.",
                "6. Wait for Jumia's review (typically 3-7 business days).",
                "Resolve this checkpoint once your seller account is "
                "approved.",
            ]),
            db=db,
            context=context,
            resume_state={"stage": "signup"},
        )
        return self.resume_checkpoint(cp, db=db, context=context)

    def resume_checkpoint(
        self,
        checkpoint: Any,
        *,
        db: Any,
        context: Any = None,
    ) -> dict[str, Any]:
        """Continue the guided seller flow after the owner acts."""
        from .checkpoints import CheckpointKind, CheckpointState

        if checkpoint.state != CheckpointState.RESOLVED:
            raise ConnectorError(
                f"checkpoint {checkpoint.id} is {checkpoint.state.value}, "
                "not resolved — the owner must complete the human step first"
            )
        stage = (checkpoint.resume_state or {}).get("stage", "")
        if stage == "signup":
            # The owner's resolve IS the attestation: approval happened.
            return {
                "account": "owner's Jumia seller account",
                "done": True,
                "identity": "owner",
                "message": (
                    "Jumia seller account approved. Next step: in the "
                    "Vendor Center go to Settings -> Manage Applications -> "
                    "Create Application -> Self Authorization, generate a "
                    "refresh token, then run "
                    "`nm connectors connect --name jumia` with "
                    "JUMIA_CLIENT_ID and JUMIA_REFRESH_TOKEN set — Devon "
                    "will validate them, vault-store them, and can then "
                    "manage your shops, orders, and catalog."
                ),
            }
        if stage == "verify_email":
            raise ConnectorError(
                "jumia seller provisioning has no verify_email stage"
            )
        raise ConnectorError(
            f"jumia cannot resume checkpoint stage {stage!r}"
        )

    # ── capability map ─────────────────────────────────────────

    def capabilities(self) -> dict[str, Any]:
        """Exactly what this connector can and cannot do, in one place.

        Jumia publishes no buyer-side API at all, so this is the honest
        contract: seller operations via the Vendor Center API, nothing
        shopper-side. Callers (and the owner) should check this instead
        of guessing.
        """
        return {
            "side": "seller",
            "auth": "oauth2 self-authorization (client_id + refresh token, "
                    "minted in the Vendor Center UI)",
            "can": {
                "shops": "list shops in the mastershop (/shops)",
                "orders": "list/filter orders, fetch order items, list "
                          "shipment providers per item",
                "order_processing": "cancel order items, pack packages "
                                    "(v1/v2), mark ready-to-ship, print "
                                    "shipping labels",
                "catalog": "list products, brands, categories, attribute "
                           "sets (GPM)",
                "feeds": "submit product feeds (create/update/stock/price/"
                         "status) and check feed processing status",
            },
            "cannot": {
                "buyer_product_search": "no buyer product-search API "
                                        "exists — Jumia publishes no public "
                                        "search/catalog endpoint for shoppers",
                "buyer_checkout": "no cart or order-placement API — "
                                  "checkout is web/mobile-app only",
                "price_tracking": "no price-history endpoint",
                "consignment": "GPM /consignment-order is FBJ-shop only "
                               "and is not implemented here",
                "token_minting": "refresh tokens can only be generated in "
                                 "the Vendor Center UI, never via the API",
            },
            "notes": (
                "Rate limits: 200 requests/minute, 4 requests/second per "
                "mastershop (HTTP 429). Mutating order endpoints need the "
                "VC - Order Manager role."
            ),
        }

    # ── shops ──────────────────────────────────────────────────

    def list_shops(self, *, master_shop: bool = False) -> list[dict[str, Any]]:
        """Shops in the mastershop (``/shops``) or of the master shop."""
        path = "/shops-of-master-shop" if master_shop else "/shops"
        data = self._api("GET", path)
        shops = data.get("shops", data) if isinstance(data, dict) else data
        return shops if isinstance(shops, list) else [shops]

    # ── orders (GOP) ───────────────────────────────────────────

    def list_orders(
        self,
        *,
        status: str = "",
        country: str = "",
        shop_id: str = "",
        created_after: str = "",
        created_before: str = "",
        updated_after: str = "",
        updated_before: str = "",
        sort: str = "ASC",
        size: int = 100,
    ) -> dict[str, Any]:
        """Orders across countries, paginated (``nextToken`` / ``isLastPage``).

        Documented filters: ``status`` (comma-separated status names, e.g.
        "PENDING,SHIPPED"), ``country`` (codes e.g. "NG,EG"), ``shop_id``.
        Date filters accept one pair only — createdAfter/createdBefore or
        updatedAfter/updatedBefore ("YYYY-MM-DD HH:MM:SS"); with no date
        filters Jumia returns today's orders.
        """
        params: dict[str, Any] = {"sort": sort, "size": size}
        if status:
            names = [s.strip().upper() for s in status.split(",") if s.strip()]
            bad = [s for s in names if s not in ORDER_ITEM_STATUSES]
            if bad:
                raise JumiaError(
                    f"unknown order status(es) {bad}: choose from "
                    f"{', '.join(ORDER_ITEM_STATUSES)}"
                )
            params["status"] = ",".join(names)
        if country:
            params["country"] = ",".join(
                c.strip().upper() for c in country.split(",") if c.strip()
            )
        if shop_id:
            params["shopId"] = shop_id
        if created_after:
            params["createdAfter"] = created_after
        if created_before:
            params["createdBefore"] = created_before
        if updated_after:
            params["updatedAfter"] = updated_after
        if updated_before:
            params["updatedBefore"] = updated_before
        data = self._api("GET", "/orders", params=params)
        return {
            "orders": data.get("orders", []),
            "nextToken": data.get("nextToken"),
            "isLastPage": data.get("isLastPage"),
        }

    def get_order_items(
        self,
        order_ids: list[str],
        *,
        status: str = "",
        shop_id: str = "",
    ) -> list[dict[str, Any]]:
        """Order items for one or more order ids (repeated ``orderId``)."""
        if not order_ids:
            raise JumiaError("get_order_items needs at least one order id")
        params: list[tuple[str, str]] = [
            ("orderId", oid) for oid in order_ids
        ]
        if status:
            params.append(("status", status.strip().upper()))
        if shop_id:
            params.append(("shopId", shop_id))
        data = self._api("GET", "/orders/items", params=params)
        return data if isinstance(data, list) else [data]

    def get_shipment_providers(
        self, order_item_ids: list[str]
    ) -> list[dict[str, Any]]:
        """Available shipment providers per order item id."""
        if not order_item_ids:
            raise JumiaError(
                "get_shipment_providers needs at least one order item id"
            )
        params = [("orderItemId", iid) for iid in order_item_ids]
        data = self._api("GET", "/orders/shipment-providers", params=params)
        items = data.get("orderItems", data) if isinstance(data, dict) else data
        return items if isinstance(items, list) else [items]

    def cancel_orders(self, order_item_ids: list[str]) -> dict[str, Any]:
        """Cancel order items. Requires the VC - Order Manager role.

        Body is ``{"orderItemIds": [...]}`` per the GOP docs. Returns the
        ``success`` / ``error`` breakdown per item.
        """
        if not order_item_ids:
            raise JumiaError("cancel_orders needs at least one order item id")
        return self._api(
            "PUT", "/orders/cancel", payload={"orderItemIds": list(order_item_ids)}
        )

    def create_package(
        self,
        packages: list[dict[str, Any]],
        *,
        api_version: int = 2,
    ) -> dict[str, Any]:
        """Pack order items into packages. Requires VC - Order Manager.

        ``api_version=2`` (default, recommended) posts ``{"packages":
        [{"orderItems": [...], "shipmentProviderId": ..., "trackingCode":
        optional}]}`` to ``v2/orders/pack``; ``api_version=1`` posts
        ``{"orderItems": [{"id": ..., "shipmentProviderId": ...}]}`` to
        ``/orders/pack``. Same-order / same-country / same-shipment-method
        constraints per package apply (GOP docs).
        """
        if api_version not in (1, 2):
            raise JumiaError("api_version must be 1 or 2")
        if not packages:
            raise JumiaError("create_package needs at least one package")
        if api_version == 2:
            return self._api("POST", "v2/orders/pack", payload={"packages": packages})
        order_items: list[dict[str, Any]] = []
        for pkg in packages:
            provider = pkg.get("shipmentProviderId")
            for oid in pkg.get("orderItems", []):
                item: dict[str, Any] = {"id": oid}
                if provider:
                    item["shipmentProviderId"] = provider
                order_items.append(item)
        return self._api("POST", "/orders/pack", payload={"orderItems": order_items})

    def mark_ready_to_ship(self, order_item_ids: list[str]) -> dict[str, Any]:
        """Mark packed order items ready to ship (VC - Order Manager).

        Body is ``{"orderItemIds": [...]}`` per the GOP docs.
        """
        if not order_item_ids:
            raise JumiaError(
                "mark_ready_to_ship needs at least one order item id"
            )
        return self._api(
            "POST",
            "/orders/ready-to-ship",
            payload={"orderItemIds": list(order_item_ids)},
        )

    def print_labels(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Print shipping labels (VC - Order Manager).

        ``payload`` is caller-supplied per the GOP PDF ("Print Labels -
        /orders/print-labels").
        """
        if not payload:
            raise JumiaError("print_labels needs a request payload")
        return self._api("POST", "/orders/print-labels", payload=payload)

    # ── catalog (GPM) ──────────────────────────────────────────

    def list_products(self, **params: Any) -> list[dict[str, Any]]:
        """Products (``GET /catalog/products``).

        Pass catalog filters through, e.g. ``sellerSku="SKU-001"`` to view
        the generated product sid and QC status (per the GPM docs).
        """
        data = self._api("GET", "/catalog/products", params=params or None)
        products = data.get("products", data) if isinstance(data, dict) else data
        return products if isinstance(products, list) else [products]

    def get_brands(self, **params: Any) -> list[dict[str, Any]]:
        """Listing structure: brands (``GET /catalog/brands``)."""
        data = self._api("GET", "/catalog/brands", params=params or None)
        brands = data.get("brands", data) if isinstance(data, dict) else data
        return brands if isinstance(brands, list) else [brands]

    def get_categories(self, **params: Any) -> list[dict[str, Any]]:
        """Listing structure: categories (``GET /catalog/categories``)."""
        data = self._api("GET", "/catalog/categories", params=params or None)
        cats = data.get("categories", data) if isinstance(data, dict) else data
        return cats if isinstance(cats, list) else [cats]

    def get_attribute_set(self, attribute_set_id: str) -> dict[str, Any]:
        """Listing structure: one attribute set's attributes."""
        if not attribute_set_id:
            raise JumiaError("get_attribute_set needs an attribute set id")
        return self._api("GET", f"/catalog/attribute-sets/{attribute_set_id}")

    def submit_feed(
        self, kind: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Submit a product feed (``POST /feeds/products/{kind}``).

        ``kind`` is one of create/update/stock/price/status — covering
        product creation, content updates, and price/inventory/status
        changes. ``payload`` follows the GPM PDF feed schemas and is
        processed asynchronously; track it with :meth:`feed_status`.
        """
        kind = (kind or "").strip().lower()
        if kind not in FEED_KINDS:
            raise JumiaError(
                f"unknown feed kind {kind!r}: choose from "
                f"{', '.join(FEED_KINDS)}"
            )
        if not payload:
            raise JumiaError("submit_feed needs a feed payload")
        return self._api("POST", f"/feeds/products/{kind}", payload=payload)

    def feed_status(self, feed_id: str) -> dict[str, Any]:
        """Feed processing result (``GET /feeds/{feed_id}``)."""
        if not feed_id:
            raise JumiaError("feed_status needs a feed id")
        return self._api("GET", f"/feeds/{feed_id}")

    # ── HTTP plumbing ──────────────────────────────────────────

    def _require_credential(self):
        cred = self._load_credential()
        if cred is None:
            raise ConnectorError(
                "jumia is not connected — run "
                "`nm connectors connect --name jumia` first"
            )
        return cred

    def _mint_tokens(
        self, client_id: str, refresh_token: str
    ) -> dict[str, Any]:
        """Exchange a refresh token for access + fresh refresh token.

        ``POST /token`` as ``application/x-www-form-urlencoded`` with
        ``grant_type=refresh_token`` (GOP docs). ``client_secret`` is only
        required for the Web Application flow; self-authorization apps
        don't receive one.
        """
        try:
            resp = self.http.post_form(
                f"{JUMIA_API_BASE}/token",
                {
                    "client_id": client_id,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
            )
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise JumiaError(f"jumia token request failed: {exc}") from exc
        if resp.status == 400:
            try:
                detail = resp.json().get("error_description", "")
            except Exception:  # noqa: BLE001 - fall back to raw text
                detail = resp.text[:200]
            raise JumiaError(
                "Jumia rejected the refresh token (invalid_grant)"
                + (f": {detail}" if detail else "")
                + " — generate a fresh one in the Vendor Center "
                "(Settings -> Manage Applications) and reconnect",
                status_code=400,
            )
        if not resp.ok:
            raise JumiaError(
                f"jumia /token failed ({resp.status}): {resp.text[:200]}",
                status_code=resp.status,
            )
        return resp.json()

    def _store_tokens(
        self,
        client_id: str,
        tokens: dict[str, Any],
        *,
        account: str = "",
    ) -> None:
        """Vault the (possibly rotated) refresh token + access token cache.

        The refresh token is the credential; the access token rides along
        in encrypted metadata so calls don't re-auth every time.
        """
        expires_in = float(tokens.get("expires_in", 3600) or 3600)
        metadata = {
            "auth": "oauth2-self-auth",
            "account_hint": account,
            "access_token": tokens.get("access_token", ""),
            "access_expires_at": time.time() + expires_in,
            "token_type": tokens.get("token_type", "bearer"),
        }
        self._store_credential(
            client_id,
            str(tokens.get("refresh_token", "")),
            credential_type="oauth_token",
            metadata=metadata,
        )

    def _access_token(self) -> str:
        """A fresh access token, refreshing (and rotating) when due."""
        cred = self._require_credential()
        meta = cred.metadata or {}
        access = str(meta.get("access_token", ""))
        expires_at = float(meta.get("access_expires_at", 0) or 0)
        if access and expires_at - time.time() > _REFRESH_LEEWAY:
            return access
        tokens = self._mint_tokens(cred.username, cred.password)
        access = str(tokens.get("access_token", ""))
        if not access:
            raise JumiaError(
                "Jumia did not return an access token on refresh"
            )
        account = str(meta.get("account_hint", "")) or self._account_name(
            cred.username, access
        )
        self._store_tokens(cred.username, tokens, account=account)
        _log.info("jumia access token refreshed (refresh token rotated)")
        return access

    def _api(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        params: Any = None,
    ) -> Any:
        """One VC API call; a 401 triggers one token refresh + retry."""
        token = self._access_token()
        try:
            return self._raw_api(method, path, payload, token, params=params)
        except JumiaError as exc:
            if exc.status_code != 401:
                raise
            # One retry with a forcibly refreshed token.
            cred = self._require_credential()
            tokens = self._mint_tokens(cred.username, cred.password)
            fresh = str(tokens.get("access_token", ""))
            if not fresh:
                raise JumiaError(
                    "Jumia did not return an access token on refresh"
                ) from exc
            meta = cred.metadata or {}
            self._store_tokens(
                cred.username,
                tokens,
                account=str(meta.get("account_hint", "")) or cred.username,
            )
            try:
                return self._raw_api(method, path, payload, fresh, params=params)
            except JumiaError as retry_exc:
                if retry_exc.status_code == 401:
                    raise JumiaError(
                        "Jumia still rejects the credentials after refresh "
                        "(401) — generate a fresh refresh token in the "
                        "Vendor Center (Settings -> Manage Applications) "
                        "and reconnect",
                        status_code=401,
                    ) from retry_exc
                raise

    def _raw_api(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        token: str,
        *,
        params: Any = None,
    ) -> Any:
        url = f"{JUMIA_API_BASE}{path}"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            if method == "GET":
                resp = self.http.get(url, headers=headers, params=params)
            elif method == "POST":
                resp = self.http.post_json(url, payload or {}, headers=headers)
            elif method == "PUT":
                body = json.dumps(payload or {}, default=str).encode("utf-8")
                put_headers = {
                    "Content-Type": "application/json",
                    **headers,
                }
                resp = self.http.request(
                    "PUT", url, data=body, headers=put_headers
                )
            else:
                raise ConnectorError(f"unsupported method {method}")
        except ConnectorError:
            raise
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise JumiaError(f"jumia request failed: {exc}") from exc
        if resp.status == 429:
            raise JumiaError(
                "Jumia rate limit exceeded (200 requests/minute, "
                "4 requests/second per mastershop) — slow down",
                status_code=429,
            )
        if resp.status in (401, 403):
            raise JumiaError(
                "Jumia rejected the credentials "
                f"({resp.status}): {resp.text[:200]}",
                status_code=resp.status,
            )
        if not resp.ok:
            try:
                detail = resp.json().get("message", "")
            except Exception:  # noqa: BLE001 - fall back to raw text
                detail = resp.text[:200]
            raise JumiaError(
                f"jumia {method} {path} failed ({resp.status}): {detail}",
                status_code=resp.status,
            )
        return resp.json()

    # ── account naming ─────────────────────────────────────────

    def _account_name(self, client_id: str, access_token: str) -> str:
        """Human account label: first shop name, else the client id."""
        try:
            headers = {"Authorization": f"Bearer {access_token}"}
            resp = self.http.get(f"{JUMIA_API_BASE}/shops", headers=headers)
            if resp.ok:
                return self._account_name_from_shops(resp.json(), client_id)
        except Exception:  # noqa: BLE001 - naming is best-effort
            pass
        return client_id

    @staticmethod
    def _account_name_from_shops(shops: Any, fallback: str) -> str:
        if isinstance(shops, dict):
            shops = shops.get("shops", shops)
        if isinstance(shops, list) and shops:
            first = shops[0]
            if isinstance(first, dict):
                name = str(first.get("name") or first.get("shopName") or "")
                if name:
                    return name
        return fallback

    def connect_instructions(self) -> str:
        """The human steps in the Vendor Center UI before connecting."""
        return "\n".join([
            "Jumia connects with a Self Authorization application (the API "
            "cannot mint refresh tokens — only the Vendor Center UI can).",
            f"1. Open {VENDOR_CENTER_URL} and log in to your seller account.",
            "2. Settings -> Manage Applications -> Create Application.",
            "3. Choose 'Self Authorization' (machine-to-machine) and name it.",
            "4. Click the Generate Token icon on the new application and copy",
            "   the refresh token (long-lived: 1 year).",
            "5. Note the application Client Id shown on the same screen.",
            "6. Paste both below (or set JUMIA_CLIENT_ID / JUMIA_REFRESH_TOKEN).",
        ])


# ── repeated-query-param support ──────────────────────────────────
#
# /orders/items takes repeated ``orderId`` params and
# /orders/shipment-providers repeated ``orderItemId`` params. The core
# HttpClient.get accepts ``params`` as a mapping; a list of tuples is
# forwarded to urllib urlencode (doseq-capable), which encodes repeated
# pairs. Nothing to patch here — this comment records the contract the
# tests pin down.
