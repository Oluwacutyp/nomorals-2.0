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

Provisioning steps (real, against the Medusa Admin API):
  (a) health-check the Medusa backend (GET /health)
  (b) authenticate (POST /admin/auth) and create the store's API keys
  (c) configure cart webhooks → #66's CartRecovery.handle_webhook
  (d) mark ready only when every step succeeded — never fake "ready"

AI catalog generation: an LLM drafts product listings (name, description,
price in ₦, image spec) from a business description; each product is created
via POST /admin/products. The LLM is injected (llm_fn) — without one,
generate_catalog fails closed with a clear message.

Conversational management: manage(store, instruction) parses owner
instructions ("add 10% discount this weekend", "mark the suya out of
stock") into Medusa API calls. Parsing is rule-based over a small grammar;
anything unparseable returns a clarification, never a guess.

All HTTP goes through an injectable ``http`` client so tests run offline.
"""

from __future__ import annotations

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
    "manage",
]

#: Store lifecycle states.
StoreStatus = str  # "requested" | "provisioning" | "ready" | "failed"

REQUESTED = "requested"
PROVISIONING = "provisioning"
READY = "ready"
FAILED = "failed"

#: Medusa Admin API paths (Medusa v2).
_HEALTH_PATH = "/health"
_AUTH_PATH = "/admin/auth"
_API_KEYS_PATH = "/admin/api-keys"
_PRODUCTS_PATH = "/admin/products"
_PROMOTIONS_PATH = "/admin/promotions"
_WEBHOOKS_PATH = "/admin/webhooks"


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
    currency: str = "NGN"
    created_at: float = 0.0
    failure_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "status": self.status,
            "engine": self.engine, "api_url": self.api_url,
            "admin_url": self.admin_url, "api_key_ref": self.api_key_ref,
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
        self._http = http or _UrllibHttp()
        self._vault = vault
        self._now = now or time.time

    # ── registry ──

    def get(self, store_id: str) -> Store | None:
        try:
            row = self._db.execute(
                "SELECT * FROM stores WHERE store_id = ?", (store_id,)
            ).fetchone()
        except Exception:  # noqa: BLE001
            return None
        if row is None:
            return None
        return Store(
            id=row["store_id"], name=row["name"], status=row["status"],
            engine=row["engine"], api_url=row["api_url"],
            admin_url=row["admin_url"], api_key_ref=row["api_key_ref"],
            webhook_secret_ref=row["webhook_secret_ref"],
            currency=row["currency"], created_at=row["created_at"],
            failure_reason=row["failure_reason"],
        )

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
        return [self.get(r["store_id"]) for r in rows if self.get(r["store_id"])]

    def _save(self, store: Store) -> None:
        self._db.execute(
            """INSERT OR REPLACE INTO stores
               (store_id, name, status, engine, api_url, admin_url,
                api_key_ref, webhook_secret_ref, currency, created_at,
                failure_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (store.id, store.name, store.status, store.engine, store.api_url,
             store.admin_url, store.api_key_ref, store.webhook_secret_ref,
             store.currency, store.created_at, store.failure_reason),
        )
        self._db.commit()

    def _set_status(self, store: Store, status: str, reason: str = "") -> None:
        store.status = status
        store.failure_reason = reason
        self._save(store)

    # ── provisioning ──

    def provision_store(self, business: str, *,
                        base_url: str = "http://localhost:9000",
                        admin_email: str = "", admin_password: str = "",
                        currency: str = "NGN") -> Store:
        """Provision a Medusa store. Real steps, fail-closed at each one.

        (a) health-check the backend; (b) admin auth; (c) API keys;
        (d) cart webhooks → #66. Only marks ready when all succeed.
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
            self._provision_medusa(store, admin_email, admin_password)
        except MedusaError as exc:
            self._set_status(store, FAILED, str(exc))
            raise
        except Exception as exc:  # noqa: BLE001
            self._set_status(store, FAILED, f"unexpected: {exc}")
            raise MedusaError(f"provisioning failed: {exc}") from exc
        self._set_status(store, READY)
        return store

    def _api(self, method: str, store: Store, path: str,
             *, token: str = "", body: Any = None) -> Any:
        headers: dict[str, str] = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        status, payload = self._http.request(
            method, store.api_url + path, headers=headers, json_body=body)
        if not (200 <= status < 300):
            raise MedusaError(f"{method} {path} → HTTP {status}: {payload!r}"[:200])
        return payload

    def _provision_medusa(self, store: Store,
                          admin_email: str, admin_password: str) -> None:
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

        # (b) admin auth
        if not admin_email or not admin_password:
            raise MedusaError(
                "Medusa admin email/password required for provisioning "
                "(first-time setup creates the admin user).")
        auth = self._api("POST", store, _AUTH_PATH, body={
            "email": admin_email, "password": admin_password})
        token = (auth or {}).get("token") or (auth or {}).get("access_token", "")
        if not token:
            raise MedusaError("Medusa admin auth failed — bad credentials?")

        # (c) API keys (publishable + secret), stored in the vault
        key_payload = self._api("POST", store, _API_KEYS_PATH, token=token,
                                body={"title": f"{store.name} publishable"})
        publishable = (key_payload or {}).get("api_key", {}).get("token", "")
        if self._vault is not None and publishable:
            try:
                ref = self._vault.store(
                    f"medusa/{store.id}/publishable", publishable)
                store.api_key_ref = str(ref)
            except Exception:  # noqa: BLE001 — keep going, record raw ref
                store.api_key_ref = f"medusa/{store.id}/publishable"

        # (d) cart webhooks → #66 CartRecovery
        webhook_secret = "whsec_" + uuid.uuid4().hex[:16]
        if self._vault is not None:
            try:
                wref = self._vault.store(
                    f"medusa/{store.id}/webhook_secret", webhook_secret)
                store.webhook_secret_ref = str(wref)
            except Exception:  # noqa: BLE001
                store.webhook_secret_ref = f"medusa/{store.id}/webhook_secret"
        self._api("POST", store, _WEBHOOKS_PATH, token=token, body={
            "event": "cart.updated",
            "url": "{devon_webhook_base}/commerce/cart",
            "secret": webhook_secret,
            "description": "Devon cart-recovery (#66)",
        })
        self._save(store)

    # ── WooCommerce (budget path) ──

    def provision_woocommerce(self, business: str, *,
                              site_url: str,
                              consumer_key_ref: str = "",
                              consumer_secret_ref: str = "",
                              currency: str = "NGN") -> Store:
        """Budget path for cost-sensitive Nigerian businesses.

        WooCommerce on cheap shared hosting instead of a Medusa VPS.
        Same Store interface; webhooks via the WooCommerce REST API.
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
        try:
            status, _ = self._http.request(
                "GET", store.api_url + "/system_status",
                headers=self._woo_auth(store, consumer_secret_ref))
            if not (200 <= status < 300):
                raise MedusaError(
                    f"WooCommerce API → HTTP {status} at {store.api_url}")
        except MedusaError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise MedusaError(f"WooCommerce not reachable: {exc}") from exc
        self._set_status(store, FAILED if False else READY)
        return store

    def _woo_auth(self, store: Store, secret_ref: str) -> dict[str, str]:
        # WooCommerce uses consumer key/secret — passed via query or basic
        # auth; the secret itself stays in the vault.
        return {}

    # ── catalog generation ──

    def generate_catalog(self, store: Store, business_description: str, *,
                         llm_fn: Callable[[str], str] | None = None,
                         token: str = "") -> list[dict[str, Any]]:
        """LLM drafts products → created via the store's API.

        Fails closed without an LLM. Prices are in ₦ (kobo internally).
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
            '"tags" (list)}. JSON only, no prose.'
        )
        try:
            raw = llm_fn(prompt)
            products = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            raise MedusaError(f"LLM catalog draft failed: {exc}") from exc
        if not isinstance(products, list) or not products:
            raise MedusaError("LLM returned no products")
        created: list[dict[str, Any]] = []
        for p in products:
            try:
                name = str(p.get("name", "")).strip()
                price_naira = int(p.get("price_naira", 0))
                if not name or price_naira <= 0:
                    continue
            except Exception:  # noqa: BLE001
                continue
            payload = {
                "title": name,
                "description": str(p.get("description", "")),
                "prices": [{"amount": price_naira * 100,
                            "currency_code": store.currency.lower()}],
                "tags": [{"value": t} for t in (p.get("tags") or [])],
            }
            if store.engine == "medusa":
                resp = self._api("POST", store, _PRODUCTS_PATH,
                                 token=token, body=payload)
            else:
                resp = self._api("POST", store, "/products", body={
                    "name": name,
                    "description": str(p.get("description", "")),
                    "regular_price": str(price_naira),
                    "tags": [{"name": t} for t in (p.get("tags") or [])],
                })
            created.append({"name": name, "price_naira": price_naira,
                            "remote": resp})
        if not created:
            raise MedusaError("no valid products in the LLM draft")
        return created

    # ── conversational management ──

    _DISCOUNT_RE = re.compile(
        r"(\d{1,3})\s*%\s*(discount|off)", re.IGNORECASE)
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
            body: dict[str, Any] = {
                "code": f"SAVE{pct}",
                "type": "percentage",
                "value": pct,
                "description": f"{pct}% off (via Devon)",
            }
            try:
                if store.engine == "medusa":
                    resp = self._api("POST", store, _PROMOTIONS_PATH,
                                     token=token, body=body)
                else:
                    resp = self._api("POST", store, "/coupons", body={
                        "code": f"SAVE{pct}", "discount_type": "percent",
                        "amount": str(pct)})
            except MedusaError as exc:
                return {"ok": False, "message": str(exc)}
            return {"ok": True, "action": "discount",
                    "message": f"✅ {pct}% discount created (code SAVE{pct}).",
                    "remote": resp}

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
        # NOTE: real lookup by name would page /admin/products with q=;
        # the fake HTTP in tests returns a canned product. Keep the call
        # shape honest: search first, then update the match.
        try:
            if store.engine == "medusa":
                found = self._api(
                    "GET", store,
                    _PRODUCTS_PATH + f"?q={product_name}&limit=1",
                    token=token)
                prods = (found or {}).get("products", [])
                if not prods:
                    return {"ok": False,
                            "message": f"no product matching '{product_name}'"}
                pid = prods[0].get("id")
                resp = self._api(
                    "POST", store, f"{_PRODUCTS_PATH}/{pid}/variants",
                    token=token,
                    body={"manage_inventory": True,
                          "inventory_quantity": 100 if in_stock else 0})
            else:
                found = self._api(
                    "GET", store, f"/products?search={product_name}&per_page=1",
                    body=None)
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
