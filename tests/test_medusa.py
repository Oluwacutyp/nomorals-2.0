"""Tests for nomorals/commerce/medusa.py — all offline (fake HTTP).

Endpoints mirror the real Medusa v2 Admin API surface:
- POST /auth/user/emailpass (admin JWT)
- POST /admin/publishable-api-keys + sales-channel link
- POST /admin/products with variants + inline prices + shipping profile
- POST /admin/promotions with application_method
- no /admin/webhooks route — cart webhooks ship as a generated subscriber
"""

import json
import tempfile

import pytest

from nomorals.commerce.medusa import (
    FAILED, PROVISIONING, READY, REQUESTED,
    MedusaError, Store, StoreManager,
    generate_webhook_subscriber,
)


class FakeHttp:
    """Pretends to be a Medusa v2 backend."""

    def __init__(self, *, down: bool = False, bad_auth: bool = False):
        self.down = down
        self.bad_auth = bad_auth
        self.calls: list[tuple[str, str, dict]] = []

    def request(self, method, url, *, headers=None, json_body=None):
        self.calls.append((method, url, json_body or {}, headers or {}))
        if self.down:
            raise MedusaError("connection refused")
        if url.endswith("/health"):
            return 200, {"status": "ok"}
        if url.endswith("/auth/user/emailpass"):
            if self.bad_auth:
                return 401, {"message": "bad credentials"}
            return 200, {"token": "tok_admin_123"}
        if url.endswith("/admin/publishable-api-keys"):
            return 200, {"publishable_api_key": {"id": "ppk_1",
                                                 "token": "pk_live_abc"}}
        if "/admin/sales-channels" in url and method == "GET":
            return 200, {"sales_channels": [{"id": "sc_1"}]}
        if "/sales-channels" in url and method == "POST":
            return 200, {}
        if "/admin/shipping-profiles" in url:
            return 200, {"shipping_profiles": [{"id": "sp_default"}]}
        if ("/admin/products" in url and method == "POST"
                and "/variants" not in url):
            return 200, {"product": {"id": "prod_1",
                                    "title": (json_body or {}).get("title")}}
        if "/admin/products" in url and method == "GET" and "q=" in url:
            return 200, {"products": [{
                "id": "prod_suya", "title": "Suya Platter",
                "variants": [{"id": "var_1"}]}]}
        if "/variants/" in url:
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
    assert any(p.endswith("/auth/user/emailpass") for p in paths)
    assert any(p.endswith("/admin/publishable-api-keys") for p in paths)
    # v2 has no /admin/webhooks REST route — must never be called
    assert not any(p.endswith("/admin/webhooks") for p in paths)
    # health first, auth second
    first_health = next(i for i, p in enumerate(paths) if p.endswith("/health"))
    first_auth = next(i for i, p in enumerate(paths)
                      if p.endswith("/auth/user/emailpass"))
    assert first_health < first_auth


def test_publishable_key_linked_to_sales_channel():
    http = FakeHttp()
    mgr = _mgr(http=http)
    mgr.provision_store("SuyaSpot", admin_email="a@b.c",
                        admin_password="secret")
    link_calls = [c for c in http.calls
                  if "/sales-channels" in c[1] and c[0] == "POST"]
    assert link_calls
    assert link_calls[0][2] == {"add": ["sc_1"]}


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


# ── webhook subscriber (the real v2 mechanism) ────────────────────────────

def test_webhook_subscriber_generated_for_cart_recovery():
    http = FakeHttp()
    mgr = _mgr(http=http)
    store = mgr.provision_store("SuyaSpot", admin_email="a@b.c",
                                admin_password="secret",
                                webhook_base="https://devon.test")
    code = mgr.get_webhook_subscriber(store.id)
    assert code
    assert "cart.updated" in code
    assert "https://devon.test/commerce/cart" in code
    assert "whsec_" in code
    assert "SubscriberConfig" in code
    # the store record explains the install step honestly
    assert "subscriber" in store.webhook_note.lower()


def test_generate_webhook_subscriber_standalone():
    code = generate_webhook_subscriber("https://x.test/hook", "whsec_abc")
    assert 'event: "cart.updated"' in code
    assert "https://x.test/hook" in code
    assert "whsec_abc" in code


# ── catalog generation ────────────────────────────────────────────────────

def _llm(prompt):
    assert "SuyaSpot" in prompt
    return json.dumps([
        {"name": "Suya Platter", "description": "Smoky beef suya.",
         "price_naira": 5000, "tags": ["grill"],
         "image_spec": "Overhead shot of suya on newspaper"},
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
    assert created[0]["image_spec"].startswith("Overhead")
    product_calls = [c for c in http.calls
                     if c[1].endswith("/admin/products")]
    assert len(product_calls) == 2
    body = product_calls[0][2]
    # v2 shape: prices inline on variants, kobo amounts
    assert body["variants"][0]["prices"][0]["amount"] == 500_000
    assert body["variants"][0]["manage_inventory"] is True
    # shipping profile set so the product can actually be checked out
    assert body["shipping_profile_id"] == "sp_default"
    # LLM image spec plumbed into metadata
    assert body["metadata"]["image_spec"].startswith("Overhead")


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
    assert "weekend" in result["message"]
    assert result["code"] == "SAVE10"
    promo_calls = [c for c in http.calls
                   if c[1].endswith("/admin/promotions")]
    assert promo_calls
    app_method = promo_calls[0][2]["application_method"]
    assert app_method["value"] == 10
    assert app_method["type"] == "percentage"
    assert app_method["allocation"] == "each"


def test_create_discount_rejects_absurd_pct():
    mgr = _mgr()
    store = mgr.provision_store("SuyaSpot", admin_email="a@b.c",
                                admin_password="s")
    with pytest.raises(MedusaError, match="1-90"):
        mgr.create_discount(store, 150, token="t")


def test_manage_out_of_stock():
    http = FakeHttp()
    mgr = _mgr(http=http)
    store = mgr.provision_store("SuyaSpot", admin_email="a@b.c",
                                admin_password="secret")
    result = mgr.manage(store, "mark the suya out of stock",
                        token="tok_admin_123")
    assert result["ok"] is True
    assert "out of stock" in result["message"]
    variant_calls = [c for c in http.calls if "/variants/" in c[1]]
    assert variant_calls
    assert variant_calls[0][2]["inventory_quantity"] == 0
    assert variant_calls[0][2]["manage_inventory"] is True


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
        self.calls.append((method, url, json_body or {}, headers or {}))
        if "/system_status" in url:
            return 200, {"environment": {"version": "8.0"}}
        if url.endswith("/webhooks"):
            return 200, {"id": 42, "topic": "action.woocommerce_add_to_cart"}
        return 404, {}


def test_provision_woocommerce():
    http = FakeWooHttp()
    mgr = _mgr(http=http)
    store = mgr.provision_woocommerce(
        "MamaPut", site_url="https://mamaput.ng",
        consumer_key_ref="woo/mamaput/key",
        consumer_key="ck_test", consumer_secret="cs_test")
    assert store.status == READY
    assert store.engine == "woocommerce"
    assert store.api_url == "https://mamaput.ng/wp-json/wc/v3"
    # real basic-auth header on the API check
    sys_calls = [c for c in http.calls if "/system_status" in c[1]]
    assert sys_calls
    assert sys_calls[0][3]["Authorization"].startswith("Basic ")
    # webhook registered with the documented custom cart topic
    wh_calls = [c for c in http.calls if c[1].endswith("/webhooks")]
    assert wh_calls
    assert wh_calls[0][2]["topic"] == "action.woocommerce_add_to_cart"
    assert "cart" in wh_calls[0][2]["delivery_url"]
    assert mgr.get(store.id).status == READY


def test_woo_auth_builds_basic_header():
    mgr = _mgr()
    headers = mgr._woo_auth("ck_1", "cs_2")
    assert headers["Authorization"].startswith("Basic ")
    import base64
    assert base64.b64decode(
        headers["Authorization"][6:]).decode() == "ck_1:cs_2"
    assert mgr._woo_auth("", "") == {}


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
