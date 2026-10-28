"""Pass tail-10 pins: commerce/medusa failures degrade honestly, nothing recorded.

Target: nomorals/commerce/medusa.py — 0 converts / 10 exclusions.
Every site below maps a failure to a genuine degraded value (the
documented miss/unknown/no-vault value of that function) and the stack
proceeds honestly — no fabricated success, no wrong secret, no wrong
action. None of them records, because none of them hides a
user-visible outcome.
Ledger note: the commerce package has NO failures ledger and the target
contains no record_failure call at all — its only observability surface
is module _log. NOT-recording is therefore pinned by snapshotting
_log.method_calls before each trigger and asserting it unchanged after
(tail-8 snapshot method, applied to _log): strictly stronger than
asserting empty, and pollution-proof against other tests in the same
session.
No bare ``except: pass`` anywhere in this file — the scanner counts
test files too; the tests below need no except blocks at all (stubs
raise, the code under test catches).
"""

import json
import tempfile
from unittest import mock

import nomorals.commerce.medusa as med
from nomorals.commerce.medusa import READY, StoreManager, _UrllibHttp


class FakeHttp:
    """Pretends to be a Medusa v2 / Woo backend (offline)."""

    def request(self, method, url, *, headers=None, json_body=None):
        if url.endswith("/health"):
            return 200, {"status": "ok"}
        if url.endswith("/auth/user/emailpass"):
            return 200, {"token": "tok_admin_123"}
        if url.endswith("/admin/publishable-api-keys"):
            return 200, {"publishable_api_key": {"id": "ppk_1", "token": "pk_live_abc"}}
        if "/admin/sales-channels" in url and method == "GET":
            return 200, {"sales_channels": [{"id": "sc_1"}]}
        if "/sales-channels" in url and method == "POST":
            return 200, {}
        if "/admin/shipping-profiles" in url:
            return 200, {"shipping_profiles": [{"id": "sp_default"}]}
        if "/admin/products" in url and method == "POST" and "/variants" not in url:
            return 200, {"product": {"id": "prod_1", "title": (json_body or {}).get("title")}}
        if "/system_status" in url:
            return 200, {"environment": {"version": "8.0"}}
        if url.endswith("/webhooks"):
            return 200, {"id": 42}
        return 404, {"message": "not found"}


class RaisingVault:
    """Vault whose every method raises."""

    def store(self, ref, value):
        raise RuntimeError("vault boom")

    def resolve(self, ref):
        raise RuntimeError("vault boom")

    def get(self, ref):
        raise RuntimeError("vault boom")

    def read(self, ref):
        raise RuntimeError("vault boom")


class RaisingDb:
    """DB whose every execute raises."""

    def execute(self, *a, **k):
        raise RuntimeError("db boom")


def _mgr(**kw):
    kw.setdefault("db_path", tempfile.mktemp(suffix=".db"))
    return StoreManager(**kw)


def _silent(log_mock, before):
    assert log_mock.method_calls == before


def test_193_unparseable_json_returns_raw_body():
    class Resp:
        status = 200

        def read(self):
            return b"not json at all"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with mock.patch.object(med, "_log") as log_mock:
        before = list(log_mock.method_calls)
        with mock.patch("urllib.request.urlopen", return_value=Resp()):
            status, payload = _UrllibHttp().request("GET", "https://example.invalid/x")
        assert (status, payload) == (200, "not json at all")
        _silent(log_mock, before)


def test_244_duplicate_column_pass_is_idempotent_migration():
    path = tempfile.mktemp(suffix=".db")
    with mock.patch.object(med, "_log") as log_mock:
        before = list(log_mock.method_calls)
        StoreManager(db_path=path, http=FakeHttp())
        mgr = StoreManager(db_path=path, http=FakeHttp())
        assert mgr.list() == []
        _silent(log_mock, before)


def test_269_get_db_failure_is_none_miss_value():
    mgr = _mgr(http=FakeHttp())
    mgr._db = RaisingDb()
    with mock.patch.object(med, "_log") as log_mock:
        before = list(log_mock.method_calls)
        assert mgr.get("store_nope") is None
        _silent(log_mock, before)


def test_284_list_db_failure_is_empty():
    mgr = _mgr(http=FakeHttp())
    mgr._db = RaisingDb()
    with mock.patch.object(med, "_log") as log_mock:
        before = list(log_mock.method_calls)
        assert mgr.list() == []
        _silent(log_mock, before)


def test_313_subscriber_db_failure_is_empty():
    mgr = _mgr(http=FakeHttp())
    mgr._db = RaisingDb()
    with mock.patch.object(med, "_log") as log_mock:
        before = list(log_mock.method_calls)
        assert mgr.get_webhook_subscriber("store_nope") == ""
        _silent(log_mock, before)


def test_413_vault_failure_falls_back_to_raw_publishable_ref():
    mgr = _mgr(http=FakeHttp(), vault=RaisingVault())
    with mock.patch.object(med, "_log") as log_mock:
        before = list(log_mock.method_calls)
        store = mgr.provision_store("SuyaSpot", admin_email="a@b.c",
                                    admin_password="pw")
        assert store.status == READY
        assert store.api_key_ref == f"medusa/{store.id}/publishable"
        _silent(log_mock, before)


def test_427_vault_failure_falls_back_to_raw_webhook_ref():
    mgr = _mgr(http=FakeHttp(), vault=RaisingVault())
    with mock.patch.object(med, "_log") as log_mock:
        before = list(log_mock.method_calls)
        store = mgr.provision_store("SuyaSpot", admin_email="a@b.c",
                                    admin_password="pw")
        assert store.status == READY
        assert store.webhook_secret_ref == f"medusa/{store.id}/webhook_secret"
        _silent(log_mock, before)


def test_485_resolve_all_methods_fail_is_empty():
    mgr = _mgr(http=FakeHttp(), vault=RaisingVault())
    with mock.patch.object(med, "_log") as log_mock:
        before = list(log_mock.method_calls)
        assert mgr._resolve_secret("medusa/x/publishable") == ""
        _silent(log_mock, before)


def test_562_woo_vault_failure_falls_back_to_raw_ref():
    mgr = _mgr(http=FakeHttp(), vault=RaisingVault())
    with mock.patch.object(med, "_log") as log_mock:
        before = list(log_mock.method_calls)
        store = mgr.provision_woocommerce("MamaPut", site_url="https://mamaput.ng",
                                          consumer_key="ck_test",
                                          consumer_secret="cs_test")
        assert store.status == READY
        assert store.webhook_secret_ref == f"woocommerce/{store.id}/webhook_secret"
        _silent(log_mock, before)


def test_621_bad_product_skipped_rest_created():
    mgr = _mgr(http=FakeHttp())
    store = mgr.provision_store("SuyaSpot", admin_email="a@b.c",
                                admin_password="pw")

    def _llm(prompt):
        return json.dumps([
            {"name": "Suya Platter", "description": "d", "price_naira": 5000,
             "tags": [], "image_spec": "Overhead shot"},
            {"name": "Bad", "description": "d", "price_naira": "abc",
             "tags": [], "image_spec": "x"},
            {"name": "Zobo", "description": "d", "price_naira": 1000,
             "tags": [], "image_spec": "Overhead jug"},
        ])

    with mock.patch.object(med, "_log") as log_mock:
        before = list(log_mock.method_calls)
        created = mgr.generate_catalog(store, "SuyaSpot", llm_fn=_llm, token="tok")
        assert [c["name"] for c in created] == ["Suya Platter", "Zobo"]
        _silent(log_mock, before)
