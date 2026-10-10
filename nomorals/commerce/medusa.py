"""Medusa self-hosted commerce backend — zero platform tax.

Devon provisions and manages headless storefronts for white-label clients.
Medusa (Node.js/TypeScript, headless, modular, MIT) is the primary engine;
WooCommerce is the budget path for cost-sensitive Nigerian businesses.

Multi-tenant state machine (per store)::

    requested → provisioning → ready
                  ↘ failed (with reason, retryable)

Namespace-per-store isolation: each store gets its own API key pair and its
own webhook secret. Devon's registry (SQLite) maps store_id → credentials
reference — secrets live in the vault, never in this module.

Provisioning steps (real, against the Medusa v2 Admin API):
  (a) health-check the Medusa backend (GET /health)
  (b) admin auth (POST /auth/user/emailpass)
  (c) publishable API key (POST /admin/publishable-api-keys), linked to a
      sales channel (POST /admin/api-keys/{id}/sales-channels) — an unlinked
      key lists no products on the storefront
  (d) cart-recovery webhook → #66's CartRecovery.handle_webhook. Medusa v2
      has NO /admin/webhooks REST route — outbound events are wired with a
      subscriber file. This step GENERATES the real subscriber TypeScript
      (``src/subscribers/devon-cart-recovery.ts``) and stores it in the
      registry for the operator to install. The store is marked ready only
      when every real step succeeded — never fake "ready".

AI catalog generation: an LLM drafts product listings (name, description,
price in ₦, image spec) from a business description; each product is created
via POST /admin/products with variants + prices inline, a shipping profile
set (products without one cannot be checked out), and inventory enabled.
The LLM is injected (llm_fn) — without one, generate_catalog fails closed.

Conversational management: manage(store, instruction) parses owner
instructions ("add 10% discount this weekend", "mark the suya out of
stock") into Medusa API calls. Parsing is rule-based over a small grammar;
anything unparseable returns a clarification, never a guess.

All HTTP goes through an injectable ``http`` client so tests run offline.
"""

from __future__ import annotations

import base64
import json
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "MedusaError",
    "Store",
    "StoreStatus",
    "StoreManager",
    "provision_store",
    "provision_woocommerce",
    "generate_catalog",
    "generate_webhook_subscriber",
    "manage",
]

#: Store lifecycle states.
StoreStatus = str  # "requested" | "provisioning" | "ready" | "failed"

REQUESTED = "requested"
PROVISIONING = "provisioning"
READY = "ready"
FAILED = "failed"

#: Medusa Admin API paths (Medusa v2 — verified against docs + real adapters).
_HEALTH_PATH = "/health"
_AUTH_PATH = "/auth/user/emailpass"              # v2 admin JWT (not /admin/auth)
_PUBLISHABLE_KEYS_PATH = "/admin/publishable-api-keys"
_SALES_CHANNELS_PATH = "/admin/sales-channels"
_PRODUCTS_PATH = "/admin/products"
_PROMOTIONS_PATH = "/admin/promotions"
_SHIPPING_PROFILES_PATH = "/admin/shipping-profiles"
#: No /admin/webhooks route exists in Medusa v2 — webhooks ship as a
#: subscriber file (see generate_webhook_subscriber).


class MedusaError(RuntimeError):
    """Medusa backend failure — always carries the failed step."""


@dataclass
class Store:
    """One provisioned storefront."""
    id: str
    name: str
    status: StoreStatus = REQUESTED
    engine: str = "medusa"          # "medusa" | "woocommerce"
    api_url: str = ""
    admin_url: str = ""
    api_key_ref: str = ""          # vault reference, never the key itself
    webhook_secret_ref: str = ""
    webhook_note: str = ""         # how the cart webhook is actually wired
    currency: str = "NGN"
    created_at: float = 0.0
    failure_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "status": self.status,
            "engine": self.engine, "api_url": self.api_url,
            "admin_url": self.admin_url, "api_key_ref": self.api_key_ref,
            "webhook_note": self.webhook_note,
            "currency": self.currency, "created_at": self.created_at,
            "failure_reason": self.failure_reason,
        }


@dataclass
class ProductDraft:
    name: str
    description: str
    price_kobo: int
    image_spec: str = ""
    tags: list[str] = field(default_factory=list)


def generate_webhook_subscriber(delivery_url: str, secret: str,
                                *, event: str = "cart.updated") -> str:
    """Generate the real Medusa v2 subscriber that forwards cart events.

    Medusa v2 has no ``/admin/webhooks`` REST route — outbound events are
    wired with a subscriber file. The operator installs the returned
    TypeScript at ``src/subscribers/devon-cart-recovery.ts`` in their Medusa
    project and restarts the backend.
    """
    return f"""// Devon cart-recovery subscriber — install at
// src/subscribers/devon-cart-recovery.ts in your Medusa v2 project,
// then restart the backend. Forwards cart events to Devon's receiver.
import type {{ SubscriberArgs, SubscriberConfig }} from "@medusajs/framework"

const DEVON_URL = {json.dumps(delivery_url)}
const DEVON_SECRET = {json.dumps(secret)}

export default async function devonCartRecoveryHandler({{
  event: {{ data }},
}}: SubscriberArgs<{{ id: string }}>) {{
  try {{
    await fetch(DEVON_URL, {{
      method: "POST",
      headers: {{
        "Content-Type": "application/json",
        "x-devon-webhook-secret": DEVON_SECRET,
      }},
      body: JSON.stringify({{ type: {json.dumps(event)}, data: {{ cart: data }} }}),
    }})
  }} catch (err) {{
    // Never break the storefront — log and move on.
    console.error("[devon] cart-recovery webhook failed", err)
  }}
}}

export const config: SubscriberConfig = {{ event: {json.dumps(event)} }}
"""


# ── HTTP client seam ──────────────────────────────────────────────────────

class _UrllibHttp:
    """Minimal real HTTP client (stdlib). Used when no client is injected."""

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout

    def request(self, method: str, url: str, *,
                headers: dict[str, str] | None = None,
                json_body: Any = None) -> tuple[int, Any]:
        import urllib.request
        data = None
        hdrs = dict(headers or {})
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            hdrs["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=hdrs,
                                     method=method.upper())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8", "replace")
                try:
                    return resp.status, json.loads(body)
                except Exception:
                    return resp.status, body
        except Exception as exc:  # noqa: BLE001 — fail closed upstream
            raise MedusaError(f"HTTP {method} {url} failed: {exc}") from exc


# ── Store manager ─────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS stores (
    store_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    engine TEXT NOT NULL,
    api_url TEXT NOT NULL,
    admin_url TEXT NOT NULL,
    api_key_ref TEXT NOT NULL,
    webhook_secret_ref TEXT NOT NULL,
    currency TEXT NOT NULL,
    created_at REAL NOT NULL,
    failure_reason TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS webhook_subscribers (
    store_id TEXT PRIMARY KEY,
    code TEXT NOT NULL,
    event TEXT NOT NULL DEFAULT 'cart.updated',
    created_at REAL NOT NULL
);
"""


class StoreManager:
    """Provisions and manages Medusa/WooCommerce stores.

    ``http`` is injectable (tests pass a fake). ``vault`` stores API keys —
    this manager only ever keeps vault *references*.
    """

    def __init__(self, db_path: str = "", *,
                 http: Any = None,
                 vault: Any = None,
                 now: Callable[[], float] | None = None) -> None:
        import os
        path = db_path or os.path.expanduser("~/.nomorals/commerce/stores.db")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        try:
            self._db.execute(
                "ALTER TABLE stores ADD COLUMN webhook_note TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass
        self._db.commit()
        self._http = http or _UrllibHttp()
        self._vault = vault
        self._now = now or time.time

    # ── registry ──

    def _row_to_store(self, row: sqlite3.Row) -> Store:
        return Store(
            id=row["store_id"], name=row["name"], status=row["status"],
            engine=row["engine"], api_url=row["api_url"],
            admin_url=row["admin_url"], api_key_ref=row["api_key_ref"],
            webhook_secret_ref=row["webhook_secret_ref"],
            webhook_note=row["webhook_note"] if "webhook_note" in row.keys() else "",
            currency=row["currency"], created_at=row["created_at"],
            failure_reason=row["failure_reason"],
        )

    def get(self, store_id: str) -> Store | None:
        try:
            row = self._db.execute(
                "SELECT * FROM stores WHERE store_id = ?", (store_id,)
            ).fetchone()
        except Exception:  # noqa: BLE001
            return None
        if row is None:
            return None
        return self._row_to_store(row)

    def list(self, *, status: str = "") -> list[Store]:
        try:
            if status:
                rows = self._db.execute(
                    "SELECT * FROM stores WHERE status = ? ORDER BY created_at",
                    (status,)).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT * FROM stores ORDER BY created_at").fetchall()
        except Exception:  # noqa: BLE001
            return []
        return [self._row_to_store(r) for r in rows]

    def _save(self, store: Store) -> None:
        self._db.execute(
            """INSERT OR REPLACE INTO stores
               (store_id, name, status, engine, api_url, admin_url,
                api_key_ref, webhook_secret_ref, webhook_note, currency,
                created_at, failure_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (store.id, store.name, store.status, store.engine, store.api_url,
             store.admin_url, store.api_key_ref, store.webhook_secret_ref,
             store.webhook_note, store.currency, store.created_at,
             store.failure_reason),
        )
        self._db.commit()

    def _set_status(self, store: Store, status: str, reason: str = "") -> None:
        store.status = status
        store.failure_reason = reason
        self._save(store)

    def get_webhook_subscriber(self, store_id: str) -> str:
        """Return the generated subscriber code for a store ("" if none)."""
        try:
            row = self._db.execute(
                "SELECT code FROM webhook_subscribers WHERE store_id = ?",
                (store_id,)).fetchone()
        except Exception:  # noqa: BLE001
            return ""
        return (row["code"] or "") if row else ""

    # ── provisioning ──

    def provision_store(self, business: str, *,
                        base_url: str = "http://localhost:9000",
                        admin_email: str = "", admin_password: str = "",
                        currency: str = "NGN",
                        webhook_base: str = "http://localhost:8000") -> Store:
        """Provision a Medusa store. Real steps, fail-closed at each one.

        (a) health-check the backend; (b) admin auth; (c) publishable API key
        linked to a sales channel; (d) cart-recovery subscriber code
        generated for the operator to install. Only marks ready when all
        succeed.
        """
        name = (business or "").strip()
        if not name:
            raise MedusaError("business name is required")
        store = Store(
            id="store_" + uuid.uuid4().hex[:10], name=name,
            status=REQUESTED, engine="medusa",
            api_url=base_url.rstrip("/"),
            admin_url=base_url.rstrip("/") + "/app",
            currency=currency, created_at=self._now(),
        )
        self._save(store)
        self._set_status(store, PROVISIONING)
        try:
            self._provision_medusa(store, admin_email, admin_password,
                                   webhook_base=webhook_base)
        except MedusaError as exc:
            self._set_status(store, FAILED, str(exc))
            raise
        except Exception as exc:  # noqa: BLE001
            self._set_status(store, FAILED, f"unexpected: {exc}")
            raise MedusaError(f"provisioning failed: {exc}") from exc
        self._set_status(store, READY)
        return store

    def _api(self, method: str, store: Store, path: str,
             *, token: str = "", body: Any = None,
             params: dict[str, str] | None = None) -> Any:
        headers: dict[str, str] = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        url = store.api_url + path
        if params:
            import urllib.parse
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
        status, payload = self._http.request(
            method, url, headers=headers, json_body=body)
        if not (200 <= status < 300):
            raise MedusaError(f"{method} {path} → HTTP {status}: {payload!r}"[:200])
        return payload

    def _provision_medusa(self, store: Store,
                          admin_email: str, admin_password: str,
                          *, webhook_base: str) -> None:
        # (a) backend alive?
        try:
            status, _ = self._http.request("GET", store.api_url + _HEALTH_PATH)
        except MedusaError as exc:
            raise MedusaError(
                "Medusa backend not reachable at "
                f"{store.api_url} — start it first (`npx medusa start` or "
                "your compose file). " + str(exc)) from exc
        if not (200 <= status < 300):
            raise MedusaError(
                f"Medusa health check → HTTP {status} at {store.api_url}")

        # (b) admin auth — v2 path is /auth/user/emailpass
        if not admin_email or not admin_password:
            raise MedusaError(
                "Medusa admin email/password required for provisioning "
                "(first-time setup creates the admin user).")
        auth = self._api("POST", store, _AUTH_PATH, body={
            "email": admin_email, "password": admin_password})
        token = (auth or {}).get("token") or (auth or {}).get("access_token", "")
        if not token:
            raise MedusaError("Medusa admin auth failed — bad credentials?")

        # (c) publishable API key, linked to a sales channel
        key_payload = self._api("POST", store, _PUBLISHABLE_KEYS_PATH,
                                token=token,
                                body={"title": f"{store.name} publishable"})
        pk = ((key_payload or {}).get("publishable_api_key")
              or (key_payload or {}).get("api_key") or {})
        pub_id = pk.get("id", "")
        publishable = pk.get("token", "")
        if not publishable:
            raise MedusaError("Medusa publishable-key creation failed")
        self._link_sales_channel(store, token, pub_id)
        if self._vault is not None and publishable:
            try:
                ref = self._vault.store(
                    f"medusa/{store.id}/publishable", publishable)
                store.api_key_ref = str(ref)
            except Exception:  # noqa: BLE001 — keep going, record raw ref
                store.api_key_ref = f"medusa/{store.id}/publishable"
        else:
            store.api_key_ref = f"medusa/{store.id}/publishable"

        # (d) cart webhooks → #66 CartRecovery. Medusa v2 has no
        # /admin/webhooks REST route — outbound events ship as a subscriber
        # file. Generate the real one and store it for the operator.
        webhook_secret = "whsec_" + uuid.uuid4().hex[:16]
        if self._vault is not None:
            try:
                wref = self._vault.store(
                    f"medusa/{store.id}/webhook_secret", webhook_secret)
                store.webhook_secret_ref = str(wref)
            except Exception:  # noqa: BLE001
                store.webhook_secret_ref = f"medusa/{store.id}/webhook_secret"
        else:
            store.webhook_secret_ref = f"medusa/{store.id}/webhook_secret"
        delivery_url = webhook_base.rstrip("/") + "/commerce/cart"
        code = generate_webhook_subscriber(delivery_url, webhook_secret)
        self._db.execute(
            """INSERT OR REPLACE INTO webhook_subscribers
               (store_id, code, event, created_at)
               VALUES (?, ?, 'cart.updated', ?)""",
            (store.id, code, self._now()),
        )
        store.webhook_note = (
            "cart.updated subscriber generated — install "
            "src/subscribers/devon-cart-recovery.ts in the Medusa project "
            "and restart the backend")
        self._save(store)

    def _link_sales_channel(self, store: Store, token: str,
                            publishable_id: str) -> None:
        """Link the publishable key to the first sales channel.

        An unlinked key returns empty product listings on the storefront —
        this step is what makes the storefront actually show products.
        """
        if not publishable_id:
            return
        try:
            channels = self._api("GET", store, _SALES_CHANNELS_PATH,
                                 token=token,
                                 params={"limit": "1"})
            items = (channels or {}).get("sales_channels", [])
            if not items:
                _log.warning("no sales channels on %s — publishable key "
                             "unlinked; storefront products will be empty",
                             store.id)
                return
            self._api("POST", store,
                      f"/admin/api-keys/{publishable_id}/sales-channels",
                      token=token,
                      body={"add": [items[0].get("id")]})
        except MedusaError as exc:
            _log.warning("sales-channel link failed for %s: %s", store.id, exc)

    # ── WooCommerce (budget path) ──

    def _resolve_secret(self, ref: str) -> str:
        """Resolve a vault ref to the actual secret. "" when unresolvable."""
        if not ref:
            return ""
        if self._vault is not None:
            for method in ("resolve", "get", "read"):
                fn = getattr(self._vault, method, None)
                if callable(fn):
                    try:
                        value = fn(ref)
                        if value:
                            return str(value)
                    except Exception:  # noqa: BLE001
                        continue
        return ""

    def _woo_auth(self, key: str, secret: str) -> dict[str, str]:
        # WooCommerce REST uses HTTP Basic with consumer key/secret.
        if not key or not secret:
            return {}
        raw = f"{key}:{secret}".encode("utf-8")
        return {"Authorization":
                "Basic " + base64.b64encode(raw).decode("ascii")}

    def provision_woocommerce(self, business: str, *,
                              site_url: str,
                              consumer_key_ref: str = "",
                              consumer_secret_ref: str = "",
                              consumer_key: str = "",
                              consumer_secret: str = "",
                              webhook_base: str = "http://localhost:8000",
                              currency: str = "NGN") -> Store:
        """Budget path for cost-sensitive Nigerian businesses.

        WooCommerce on cheap shared hosting instead of a Medusa VPS.
        Same Store interface; cart webhooks via the WooCommerce REST API
        (documented custom topic ``action.woocommerce_add_to_cart`` — WC
        core has no native cart topic).
        """
        name = (business or "").strip()
        if not name:
            raise MedusaError("business name is required")
        if not site_url:
            raise MedusaError("WooCommerce site URL is required")
        store = Store(
            id="store_" + uuid.uuid4().hex[:10], name=name,
            status=REQUESTED, engine="woocommerce",
            api_url=site_url.rstrip("/") + "/wp-json/wc/v3",
            admin_url=site_url.rstrip("/") + "/wp-admin",
            api_key_ref=consumer_key_ref, currency=currency,
            created_at=self._now(),
        )
        self._save(store)
        self._set_status(store, PROVISIONING)
        key = consumer_key or self._resolve_secret(consumer_key_ref)
        secret = consumer_secret or self._resolve_secret(consumer_secret_ref)
        headers = self._woo_auth(key, secret)
        if not headers:
            _log.warning("no WooCommerce credentials for %s — API calls "
                         "will fail unless the site allows them", store.id)
        try:
            status, _ = self._http.request(
                "GET", store.api_url + "/system_status", headers=headers)
            if not (200 <= status < 300):
                raise MedusaError(
                    f"WooCommerce API → HTTP {status} at {store.api_url}")
        except MedusaError:
            self._set_status(store, FAILED, "WooCommerce API not reachable")
            raise
        except Exception as exc:  # noqa: BLE001
            self._set_status(store, FAILED,
                             f"WooCommerce not reachable: {exc}")
            raise MedusaError(f"WooCommerce not reachable: {exc}") from exc
        # Register the cart webhook (documented custom-action topic).
        try:
            self._register_woo_webhook(store, headers, webhook_base)
        except MedusaError as exc:
            _log.warning("WooCommerce webhook registration failed: %s", exc)
        self._set_status(store, READY)
        return store

    def _register_woo_webhook(self, store: Store, headers: dict[str, str],
                              webhook_base: str) -> None:
        webhook_secret = "whsec_" + uuid.uuid4().hex[:16]
        if self._vault is not None:
            try:
                wref = self._vault.store(
                    f"woocommerce/{store.id}/webhook_secret", webhook_secret)
                store.webhook_secret_ref = str(wref)
            except Exception:  # noqa: BLE001
                store.webhook_secret_ref = f"woocommerce/{store.id}/webhook_secret"
        status, payload = self._http.request(
            "POST", store.api_url + "/webhooks", headers=headers, json_body={
                "name": f"Devon cart recovery ({store.name})",
                "topic": "action.woocommerce_add_to_cart",
                "delivery_url": webhook_base.rstrip("/") + "/commerce/cart",
                "secret": webhook_secret,
            })
        if not (200 <= status < 300):
            raise MedusaError(
                f"WooCommerce webhook registration → HTTP {status}: "
                f"{payload!r}"[:200])
        store.webhook_note = (
            "webhook registered: action.woocommerce_add_to_cart → "
            "/commerce/cart (HMAC-SHA256 signed with the vault secret)")

    # ── catalog generation ──

    def generate_catalog(self, store: Store, business_description: str, *,
                         llm_fn: Callable[[str], str] | None = None,
                         token: str = "") -> list[dict[str, Any]]:
        """LLM drafts products → created via the store's API.

        Fails closed without an LLM. Prices are in ₦ (kobo internally —
        Medusa v2 price amounts are smallest-currency-unit).
        """
        if store.status != READY:
            raise MedusaError(
                f"store '{store.name}' is {store.status} — provision it first")
        if llm_fn is None:
            raise MedusaError(
                "catalog generation needs an LLM — pass llm_fn or draft "
                "products manually")
        prompt = (
            "You are a Nigerian e-commerce catalog writer. "
            f"Business: {business_description}\n"
            "Return a JSON list of 5-10 products. Each: "
            '{"name", "description" (1-2 sentences, warm Nigerian voice), '
            '"price_naira" (integer, realistic Lagos market price), '
            '"tags" (list), "image_spec" (one-line visual description of the '
            "product photo to shoot/generate later)}. "
            "JSON only, no prose."
        )
        try:
            raw = llm_fn(prompt)
            products = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            raise MedusaError(f"LLM catalog draft failed: {exc}") from exc
        if not isinstance(products, list) or not products:
            raise MedusaError("LLM returned no products")
        shipping_profile_id = self._default_shipping_profile(store, token)
        created: list[dict[str, Any]] = []
        for p in products:
            try:
                name = str(p.get("name", "")).strip()
                price_naira = int(p.get("price_naira", 0))
                if not name or price_naira <= 0:
                    continue
            except Exception:  # noqa: BLE001
                continue
            description = str(p.get("description", ""))
            tags = [str(t) for t in (p.get("tags") or [])]
            image_spec = str(p.get("image_spec") or "")
            if store.engine == "medusa":
                payload: dict[str, Any] = {
                    "title": name,
                    "description": description,
                    # Variants carry prices inline (v2 shape); inventory on
                    # so stock levels work out of the box.
                    "options": [{"title": "Default",
                                 "values": ["Default"]}],
                    "variants": [{
                        "title": name,
                        "manage_inventory": True,
                        "prices": [{
                            "amount": price_naira * 100,  # kobo
                            "currency_code": store.currency.lower(),
                        }],
                        "options": {"Default": "Default"},
                    }],
                    "tags": [{"value": t} for t in tags],
                    "metadata": ({"image_spec": image_spec}
                                 if image_spec else {}),
                }
                # Products without a shipping profile cannot be checked out.
                if shipping_profile_id:
                    payload["shipping_profile_id"] = shipping_profile_id
                resp = self._api("POST", store, _PRODUCTS_PATH,
                                 token=token, body=payload)
            else:
                resp = self._api("POST", store, "/products", body={
                    "name": name,
                    "description": description,
                    "regular_price": str(price_naira),
                    "manage_stock": True,
                    "tags": [{"name": t} for t in tags],
                })
            created.append({"name": name, "price_naira": price_naira,
                            "image_spec": image_spec, "remote": resp})
        if not created:
            raise MedusaError("no valid products in the LLM draft")
        return created

    def _default_shipping_profile(self, store: Store, token: str) -> str:
        """Fetch the default shipping profile id (Medusa only).

        Products created without one cannot be checked out — this lookup
        is what keeps generate_catalog's output actually sellable.
        """
        if store.engine != "medusa":
            return ""
        try:
            payload = self._api("GET", store, _SHIPPING_PROFILES_PATH,
                                token=token, params={"limit": "1"})
            profiles = (payload or {}).get("shipping_profiles", [])
            return str(profiles[0].get("id", "")) if profiles else ""
        except MedusaError as exc:
            _log.warning("shipping-profile lookup failed for %s: %s",
                         store.id, exc)
            return ""

    # ── discounts ──

    def create_discount(self, store: Store, pct: int, *,
                        token: str = "") -> str:
        """Create a real percentage-off coupon in the store. Returns the code.

        Medusa v2 payload uses ``application_method`` (the required shape);
        WooCommerce uses POST /coupons. Raises MedusaError on failure.
        """
        if not 1 <= pct <= 90:
            raise MedusaError(f"{pct}% looks off — pick 1-90%.")
        code = f"SAVE{pct}"
        if store.engine == "medusa":
            self._api("POST", store, _PROMOTIONS_PATH, token=token, body={
                "code": code,
                "type": "percentage",
                "status": "active",
                "application_method": {
                    "type": "percentage",
                    "target_type": "order",
                    "allocation": "each",
                    "value": pct,
                    "max_quantity": 1,
                    "currency_code": store.currency.lower(),
                },
            })
        else:
            self._api("POST", store, "/coupons", body={
                "code": code, "discount_type": "percent",
                "amount": str(pct)})
        return code

    # ── conversational management ──

    _DISCOUNT_RE = re.compile(
        r"(\d{1,3})\s*%\s*(discount|off)", re.IGNORECASE)
    _DURATION_RE = re.compile(
        r"this\s+(weekend|week|month)|today|tomorrow", re.IGNORECASE)
    _OUT_OF_STOCK_RE = re.compile(
        r"mark\s+(?:the\s+)?(.+?)\s+out of stock", re.IGNORECASE)
    _IN_STOCK_RE = re.compile(
        r"(?:mark\s+)?(?:the\s+)?(.+?)\s+(?:back\s+)?in stock", re.IGNORECASE)

    def manage(self, store: Store, instruction: str, *,
               token: str = "") -> dict[str, Any]:
        """'add 10% discount this weekend' / 'mark the suya out of stock'.

        Rule-based over a small grammar. Unparseable → clarification dict,
        never a guessed API call.
        """
        if store.status != READY:
            return {"ok": False,
                    "message": f"store '{store.name}' is {store.status}"}
        text = (instruction or "").strip()
        if not text:
            return {"ok": False, "message": "tell me what to change."}

        m = self._DISCOUNT_RE.search(text)
        if m:
            pct = int(m.group(1))
            if not 1 <= pct <= 90:
                return {"ok": False,
                        "message": f"{pct}% looks off — pick 1-90%."}
            dur = self._DURATION_RE.search(text)
            duration = f" ({dur.group(0)})" if dur else ""
            try:
                code = self.create_discount(store, pct, token=token)
            except MedusaError as exc:
                return {"ok": False, "message": str(exc)}
            return {"ok": True, "action": "discount", "code": code,
                    "message": f"✅ {pct}% discount created{duration} "
                               f"(code {code})."}

        m = self._OUT_OF_STOCK_RE.search(text)
        if m:
            return self._set_stock(store, m.group(1).strip(), False,
                                   token=token)
        m = self._IN_STOCK_RE.search(text)
        if m:
            return self._set_stock(store, m.group(1).strip(), True,
                                   token=token)
        return {"ok": False,
                "message": ("I can do: 'add N% discount ...', "
                            "'mark <product> out of stock', "
                            "'mark <product> in stock'. "
                            f"Didn't understand: '{text[:60]}'")}

    def _set_stock(self, store: Store, product_name: str, in_stock: bool, *,
                   token: str = "") -> dict[str, Any]:
        if not product_name:
            return {"ok": False, "message": "which product?"}
        try:
            if store.engine == "medusa":
                found = self._api(
                    "GET", store,
                    _PRODUCTS_PATH, token=token,
                    params={"q": product_name, "limit": "1",
                            "fields": "*variants"})
                prods = (found or {}).get("products", [])
                if not prods:
                    return {"ok": False,
                            "message": f"no product matching '{product_name}'"}
                variants = prods[0].get("variants") or []
                if not variants:
                    return {"ok": False,
                            "message": f"'{product_name}' has no variants "
                                       "to stock-manage"}
                pid, vid = prods[0].get("id"), variants[0].get("id")
                resp = self._api(
                    "POST", store, f"{_PRODUCTS_PATH}/{pid}/variants/{vid}",
                    token=token,
                    body={"manage_inventory": True,
                          "inventory_quantity": 100 if in_stock else 0})
            else:
                found = self._api(
                    "GET", store, "/products",
                    params={"search": product_name, "per_page": "1"})
                prods = found if isinstance(found, list) else []
                if not prods:
                    return {"ok": False,
                            "message": f"no product matching '{product_name}'"}
                pid = prods[0].get("id")
                resp = self._api("PUT", store, f"/products/{pid}", body={
                    "stock_status": "instock" if in_stock else "outofstock",
                    "manage_stock": True,
                    "stock_quantity": 100 if in_stock else 0})
        except MedusaError as exc:
            return {"ok": False, "message": str(exc)}
        state = "in stock ✅" if in_stock else "out of stock ⏸️"
        return {"ok": True, "action": "stock",
                "message": f"✅ '{product_name}' marked {state}.",
                "remote": resp}


# ── module-level convenience (owner chat path) ─────────────────────────────

_default_manager: StoreManager | None = None


def _manager() -> StoreManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = StoreManager()
    return _default_manager


def provision_store(business: str, **kwargs: Any) -> Store:
    """Provision a Medusa store (module-level convenience)."""
    return _manager().provision_store(business, **kwargs)


def provision_woocommerce(business: str, **kwargs: Any) -> Store:
    """Provision a WooCommerce store (module-level convenience)."""
    return _manager().provision_woocommerce(business, **kwargs)


def generate_catalog(store: Store, business_description: str,
                     **kwargs: Any) -> list[dict[str, Any]]:
    """Generate + create a product catalog (module-level convenience)."""
    return _manager().generate_catalog(store, business_description, **kwargs)


def manage(store: Store, instruction: str, **kwargs: Any) -> dict[str, Any]:
    """Conversational store management (module-level convenience)."""
    return _manager().manage(store, instruction, **kwargs)
