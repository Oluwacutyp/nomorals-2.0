"""Tests for nomorals/commerce/medusa.py — all offline (fake HTTP)."""

import json
import tempfile

import pytest

from nomorals.commerce.medusa import (
    FAILED, PROVISIONING, READY, REQUESTED,
    MedusaError, Store, StoreManager,
)


class FakeHttp:
    """Pretends to be a Medusa backend."""

    def __init__(self, *, down: bool = False, bad_auth: bool = False):
        self.down = down
        self.bad_auth = bad_auth
        self.calls: list[tuple[str, str, dict]] = []

    def request(self, method, url, *, headers=None, json_body=None):
        self.calls.append((method, url, json_body or {}))
        if self.down:
            raise MedusaError("connection refused")
        if url.endswith("/health"):
            return 200, {"status": "ok"}
        if url.endswith("/admin/auth"):
            if self.bad_auth:
                return 401, {"message": "bad credentials"}
            return 200, {"token": "tok_admin_123"}
        if url.endswith("/admin/api-keys"):
            return 200, {"api_key": {"token": "pk_live_abc"}}
        if url.endswith("/admin/webhooks"):
            return 200, {"webhook": {"id": "wh_1"}}
        if "/admin/products" in url and method == "POST" and "variants" not in url:
            return 200, {"product": {"id": "prod_1",
                                    "title": (json_body or {}).get("title")}}
        if "/admin/products?q=" in url:
            return 200, {"products": [{"id": "prod_suya",
                                       "title": "Suya Platter"}]}
        if "/variants" in url:
            return 200, {"variant": {"id": "var_1"}}
        if url.endswith("/admin/promotions"):
            return 200, {"promotion": {"id": "promo_1"}}
        return 404, {"message": "not found"}


class FakeVault:
    def __init__(self):
        self.secrets: dict[str, str] = {}

    def store(self, ref, value):
        self.secrets[ref] = value
        return ref


def _mgr(**kw):
    kw.setdefault("db_path", tempfile.mktemp(suffix=".db"))
    kw.setdefault("http", FakeHttp())
    kw.setdefault("vault", FakeVault())
    return StoreManager(**kw)


# ── state machine ─────────────────────────────────────────────────────────

def test_provision_happy_path_reaches_ready():
    mgr = _mgr()
    store = mgr.provision_store("SuyaSpot", admin_email="a@b.c",
                                admin_password="secret")
    assert store.status == READY
    assert store.engine == "medusa"
    assert store.api_url == "http://localhost:9000"
    assert store.admin_url == "http://localhost:9000/app"
    assert store.api_key_ref.startswith("medusa/")
    assert mgr.get(store.id).status == READY


def test_provision_steps_in_order():
    http = FakeHttp()
    mgr = _mgr(http=http)
    mgr.provision_store("SuyaSpot", admin_email="a@b.c",
                        admin_password="secret")
    paths = [c[1] for c in http.calls]
    assert any(p.endswith("/health") for p in paths)
    assert any(p.endswith("/admin/auth") for p in paths)
    assert any(p.endswith("/admin/api-keys") for p in paths)
    assert any(p.endswith("/admin/webhooks") for p in paths)
    # health first, auth second
    first_health = next(i for i, p in enumerate(paths) if p.endswith("/health"))
    first_auth = next(i for i, p in enumerate(paths) if p.endswith("/admin/auth"))
    assert first_health < first_auth


def test_provision_fails_closed_when_backend_down():
    mgr = _mgr(http=FakeHttp(down=True))
    with pytest.raises(MedusaError):
        mgr.provision_store("SuyaSpot", admin_email="a@b.c",
                            admin_password="secret")
    stores = mgr.list()
    assert stores and stores[0].status == FAILED
    assert "not reachable" in stores[0].failure_reason


def test_provision_requires_admin_credentials():
    mgr = _mgr()
    with pytest.raises(MedusaError, match="email/password"):
        mgr.provision_store("SuyaSpot")
    assert mgr.list()[0].status == FAILED


def test_provision_bad_auth_fails():
    mgr = _mgr(http=FakeHttp(bad_auth=True))
    with pytest.raises(MedusaError):
        mgr.provision_store("SuyaSpot", admin_email="a@b.c",
                            admin_password="wrong")
    assert mgr.list()[0].status == FAILED


def test_provision_empty_business_rejected():
    mgr = _mgr()
    with pytest.raises(MedusaError, match="business name"):
        mgr.provision_store("   ")


def test_list_filter_by_status():
    mgr = _mgr()
    mgr.provision_store("A", admin_email="a@b.c", admin_password="s")
    mgr2 = _mgr(http=FakeHttp(down=True))
    # share nothing — separate DBs; just check filtering works
    assert mgr.list(status=READY)
    assert not mgr.list(status=FAILED)


# ── webhook configuration ─────────────────────────────────────────────────

def test_webhook_configured_for_cart_recovery():
    http = FakeHttp()
    mgr = _mgr(http=http)
    mgr.provision_store("SuyaSpot", admin_email="a@b.c",
                        admin_password="secret")
    wh_calls = [c for c in http.calls if c[1].endswith("/admin/webhooks")]
    assert wh_calls
    body = wh_calls[0][2]
    assert body["event"] == "cart.updated"
    assert "cart" in body["url"]


# ── catalog generation ────────────────────────────────────────────────────

def _llm(prompt):
    assert "SuyaSpot" in prompt
    return json.dumps([
        {"name": "Suya Platter", "description": "Smoky beef suya.",
         "price_naira": 5000, "tags": ["grill"]},
        {"name": "Asun", "description": "Peppered goat meat.",
         "price_naira": 4500, "tags": ["spicy"]},
    ])


def test_generate_catalog_creates_products():
    http = FakeHttp()
    mgr = _mgr(http=http)
    store = mgr.provision_store("SuyaSpot", admin_email="a@b.c",
                                admin_password="secret")
    created = mgr.generate_catalog(store, "SuyaSpot — Lagos grill joint",
                                   llm_fn=_llm, token="tok_admin_123")
    assert len(created) == 2
    assert created[0]["name"] == "Suya Platter"
    assert created[0]["price_naira"] == 5000
    product_calls = [c for c in http.calls
                     if c[1].endswith("/admin/products")]
    assert len(product_calls) == 2
    # kobo pricing
    assert product_calls[0][2]["prices"][0]["amount"] == 500_000


def test_generate_catalog_fails_closed_without_llm():
    mgr = _mgr()
    store = mgr.provision_store("SuyaSpot", admin_email="a@b.c",
                                admin_password="secret")
    with pytest.raises(MedusaError, match="needs an LLM"):
        mgr.generate_catalog(store, "desc")


def test_generate_catalog_requires_ready_store():
    mgr = _mgr()
    store = Store(id="x", name="X", status=REQUESTED)
    with pytest.raises(MedusaError, match="provision it first"):
        mgr.generate_catalog(store, "desc", llm_fn=_llm)


# ── conversational management ─────────────────────────────────────────────

def test_manage_discount():
    http = FakeHttp()
    mgr = _mgr(http=http)
    store = mgr.provision_store("SuyaSpot", admin_email="a@b.c",
                                admin_password="secret")
    result = mgr.manage(store, "add 10% discount this weekend",
                        token="tok_admin_123")
    assert result["ok"] is True
    assert "10%" in result["message"]
    promo_calls = [c for c in http.calls
                   if c[1].endswith("/admin/promotions")]
    assert promo_calls and promo_calls[0][2]["value"] == 10


def test_manage_out_of_stock():
    mgr = _mgr()
    store = mgr.provision_store("SuyaSpot", admin_email="a@b.c",
                                admin_password="secret")
    result = mgr.manage(store, "mark the suya out of stock",
                        token="tok_admin_123")
    assert result["ok"] is True
    assert "out of stock" in result["message"]


def test_manage_unparseable_returns_clarification():
    mgr = _mgr()
    store = mgr.provision_store("SuyaSpot", admin_email="a@b.c",
                                admin_password="secret")
    result = mgr.manage(store, "do the thing with the stuff")
    assert result["ok"] is False
    assert "discount" in result["message"]


def test_manage_rejects_absurd_discount():
    mgr = _mgr()
    store = mgr.provision_store("SuyaSpot", admin_email="a@b.c",
                                admin_password="secret")
    result = mgr.manage(store, "add 150% discount now")
    assert result["ok"] is False


def test_manage_non_ready_store():
    mgr = _mgr()
    store = Store(id="x", name="X", status=FAILED)
    result = mgr.manage(store, "add 10% discount")
    assert result["ok"] is False


# ── WooCommerce path ──────────────────────────────────────────────────────

class FakeWooHttp(FakeHttp):
    def request(self, method, url, *, headers=None, json_body=None):
        self.calls.append((method, url, json_body or {}))
        if "/system_status" in url:
            return 200, {"environment": {"version": "8.0"}}
        return 404, {}


def test_provision_woocommerce():
    mgr = _mgr(http=FakeWooHttp())
    store = mgr.provision_woocommerce(
        "MamaPut", site_url="https://mamaput.ng",
        consumer_key_ref="woo/mamaput/key")
    assert store.status == READY
    assert store.engine == "woocommerce"
    assert store.api_url == "https://mamaput.ng/wp-json/wc/v3"


def test_provision_woocommerce_needs_url():
    mgr = _mgr()
    with pytest.raises(MedusaError, match="site URL"):
        mgr.provision_woocommerce("MamaPut", site_url="")


# ── namespace isolation ───────────────────────────────────────────────────

def test_stores_get_distinct_ids_and_key_refs():
    mgr = _mgr()
    a = mgr.provision_store("A", admin_email="a@b.c", admin_password="s")
    b = mgr.provision_store("B", admin_email="a@b.c", admin_password="s")
    assert a.id != b.id
    assert a.api_key_ref != b.api_key_ref
    vault = mgr._vault
    assert f"medusa/{a.id}/publishable" in vault.secrets
    assert f"medusa/{b.id}/publishable" in vault.secrets
