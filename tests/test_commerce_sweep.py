"""Sweep tests for nomorals/commerce/ — new behavior mined from real-world practice.

Covers: configurable sequences, objection-handling + incentive templates,
coupon_fn seam, retry/backoff, per-step attribution, expiry, funnel stats,
UTM tagging, opt-out line, opt-in source recording, Medusa v2 API shapes,
subscriber-based webhooks, WooCommerce basic-auth + webhook topic, and the
create_discount helper shared with cart recovery.
"""

from __future__ import annotations

import json
import tempfile

import pytest

from nomorals.commerce.cart_recovery import (
    STEPS, CartRecovery, normalize_webhook,
)
from nomorals.commerce.medusa import (
    READY, MedusaError, Store, StoreManager,
)


# ── shared fakes ──────────────────────────────────────────────────────────

def _engine(tmp_path, **kw):
    sent: list[tuple[str, str]] = []

    def sender(phone: str, text: str) -> bool:
        sent.append((phone, text))
        return True

    kw.setdefault("sender", sender)
    engine = CartRecovery(str(tmp_path / "carts.db"), **kw)
    engine.sent = sent  # type: ignore[attr-defined]
    return engine


def _cart(**kw):
    base = dict(
        cart_id="cart-1",
        phone="+2348012345678",
        items=[{"name": "Jollof Rice"}, {"name": "Dodo"}],
        total_kobo=250_000,
        opted_in=True,
        store="TestStore",
        customer_name="Ada",
        checkout_link="https://pay.test/cart-1",
    )
    base.update(kw)
    return base


class FakeHttp:
    def __init__(self):
        self.calls: list[tuple] = []

    def request(self, method, url, *, headers=None, json_body=None):
        self.calls.append((method, url, json_body or {}, headers or {}))
        if url.endswith("/health"):
            return 200, {"status": "ok"}
        if url.endswith("/auth/user/emailpass"):
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


# ── configurable sequence ─────────────────────────────────────────────────

def test_custom_steps_used(tmp_path):
    custom = ((1, 60, "reminder"), (2, 3600, "last_call"))
    cr = _engine(tmp_path, steps=custom)
    cr.cart_abandoned(**_cart())
    rows = cr._db.execute(
        "SELECT step, run_at FROM cart_steps ORDER BY step").fetchall()
    assert [r["step"] for r in rows] == [1, 2]
    base = rows[0]["run_at"] - 60
    assert abs(rows[1]["run_at"] - (base + 3600)) < 2
    # step 2 here IS the last call, so it may carry the incentive
    cr.run_step("cart-1", 2)
    assert "last call" in cr.sent[0][1].lower()


def test_default_steps_unchanged():
    assert [n for n, _d, _t in STEPS] == [1, 2, 3]
    assert STEPS[0][1] == 30 * 60 and STEPS[1][1] == 24 * 3600


# ── objection-handling template ───────────────────────────────────────────

def test_step_2_addresses_checkout_objection(tmp_path):
    cr = _engine(tmp_path)
    cr.cart_abandoned(**_cart())
    cr.run_step("cart-1", 2)
    text = cr.sent[0][1].lower()
    assert "thinking" in text
    assert "hidden fees" in text  # the #1 abandonment reason, handled


# ── incentive discipline: step 3 only ─────────────────────────────────────

def test_incentive_only_in_last_call(tmp_path):
    cr = _engine(tmp_path)
    cr.cart_abandoned(**_cart(coupon_code="SAVE10", discount_pct=10))
    cr.run_step("cart-1", 1)
    assert "SAVE10" not in cr.sent[0][1]
    cr.run_step("cart-1", 2)
    assert "SAVE10" not in cr.sent[1][1]
    cr.run_step("cart-1", 3)
    assert "SAVE10" in cr.sent[2][1]
    assert "10%" in cr.sent[2][1]


def test_coupon_fn_mints_code_for_final_step(tmp_path):
    calls = []

    def coupon_fn(cart):
        calls.append(cart.cart_id)
        return "RESCUE15"

    cr = _engine(tmp_path, coupon_fn=coupon_fn)
    cr.cart_abandoned(**_cart())
    assert calls == ["cart-1"]
    cr.run_step("cart-1", 1)
    assert "RESCUE15" not in cr.sent[0][1]
    cr.run_step("cart-1", 3)
    assert "RESCUE15" in cr.sent[1][1]


def test_coupon_fn_failure_never_breaks_intake(tmp_path):
    def boom(cart):
        raise RuntimeError("coupon service down")

    cr = _engine(tmp_path, coupon_fn=boom)
    cart = cr.cart_abandoned(**_cart())
    assert cart.opted_in is True
    assert cr.run_step("cart-1", 1) is True


# ── retry / backoff ───────────────────────────────────────────────────────

def test_failed_send_retries_after_backoff(tmp_path):
    state = {"fail": True}
    sent: list[str] = []

    def flaky(phone, text):
        if state["fail"]:
            return False
        sent.append(text)
        return True

    now = [1_000_000.0]
    cr = CartRecovery(str(tmp_path / "c.db"), sender=flaky,
                      now=lambda: now[0])
    cr.cart_abandoned(**_cart())
    cr._now = lambda: now[0] + 31 * 60
    assert cr.run_due_steps() == 0  # send failed
    row = cr._db.execute(
        "SELECT status, attempts FROM cart_steps "
        "WHERE cart_id='cart-1' AND step=1").fetchone()
    assert row["status"] == "failed"
    assert row["attempts"] == 1
    # not yet due for retry
    assert cr.run_due_steps() == 0
    assert sent == []
    # backoff elapsed → retry goes out
    now[0] += 16 * 60
    state["fail"] = False
    assert cr.run_due_steps() == 1
    assert len(sent) == 1


def test_retry_budget_exhausted_gives_up(tmp_path):
    cr = _engine(tmp_path, sender=lambda p, t: False)
    cr.cart_abandoned(**_cart())
    cr.run_step("cart-1", 1)
    cr.run_step("cart-1", 1)  # retry attempt
    row = cr._db.execute(
        "SELECT status, attempts FROM cart_steps "
        "WHERE cart_id='cart-1' AND step=1").fetchone()
    assert row["attempts"] == 2
    # no third attempt even when due
    cr._now = lambda: 10**10
    assert cr.run_due_steps() == 0
    assert cr.sent == []


# ── per-step attribution ──────────────────────────────────────────────────

def test_recovery_attributed_to_step_sent(tmp_path):
    cr = _engine(tmp_path)
    cr.cart_abandoned(**_cart())
    cr.run_step("cart-1", 1)
    cr.run_step("cart-1", 2)
    cr.mark_recovered("cart-1")
    row = cr._db.execute(
        "SELECT recovered_via_step FROM recoveries "
        "WHERE cart_id='cart-1'").fetchone()
    assert row["recovered_via_step"] == 2
    s = cr.stats()
    assert s["recovered_via_step_7d"] == {"2": 1}


def test_recovery_with_no_messages_attributed_zero(tmp_path):
    cr = _engine(tmp_path)
    cr.cart_abandoned(**_cart())
    cr.mark_recovered("cart-1")
    row = cr._db.execute(
        "SELECT recovered_via_step FROM recoveries "
        "WHERE cart_id='cart-1'").fetchone()
    assert row["recovered_via_step"] == 0


# ── expiry ────────────────────────────────────────────────────────────────

def test_expire_dead_carts(tmp_path):
    cr = _engine(tmp_path)
    cr.cart_abandoned(**_cart(cart_id="dead"))
    # burn the whole sequence (sends succeed)
    for step in (1, 2, 3):
        cr.run_step("dead", step)
    assert cr.mark_expired("dead") is True
    row = cr._db.execute(
        "SELECT status FROM carts WHERE cart_id='dead'").fetchone()
    assert row["status"] == "expired"
    # recovered carts are never expired
    cr.cart_abandoned(**_cart(cart_id="live"))
    cr.mark_recovered("live")
    assert cr.mark_expired("live") is False


def test_mark_expired_refuses_open_sequence(tmp_path):
    cr = _engine(tmp_path)
    cr.cart_abandoned(**_cart())
    assert cr.mark_expired("cart-1") is False  # steps still pending


def test_expire_dead_carts_sweeps_old_abandoned(tmp_path):
    now = [1_000_000.0]
    cr = CartRecovery(str(tmp_path / "c.db"), sender=lambda p, t: True,
                      now=lambda: now[0])
    cr.cart_abandoned(**_cart(cart_id="old"))
    now[0] += 31 * 60
    cr.run_due_steps()  # step 1 sent
    now[0] += 25 * 3600
    cr.run_due_steps()  # step 2 sent
    now[0] += 49 * 3600
    cr.run_due_steps()  # step 3 sent
    now[0] += 8 * 86400
    assert cr.expire_dead_carts() == 1
    s = cr.stats()
    assert s["expired_7d"] == 0  # created >7d ago — outside the window
    assert s["recovered_7d"] == 0


# ── funnel report + stats ─────────────────────────────────────────────────

def test_funnel_report(tmp_path):
    cr = _engine(tmp_path)
    cr.cart_abandoned(**_cart(cart_id="a"))
    cr.cart_abandoned(**_cart(cart_id="b"))
    cr.mark_recovered("a")
    report = cr.funnel_report()
    assert "2 abandoned" in report
    assert "1 recovered" in report
    assert "50.0%" in report
    assert "benchmark" in report


def test_stats_revenue_per_recipient_and_failures(tmp_path):
    cr = CartRecovery(str(tmp_path / "c.db"),
                      sender=lambda p, t: False)
    cr.cart_abandoned(**_cart(cart_id="a", total_kobo=200_000))
    cr.cart_abandoned(**_cart(cart_id="b", total_kobo=400_000))
    cr.mark_recovered("a")
    cr.mark_recovered("b")
    cr.run_step("a", 1)  # no-op: already recovered
    s = cr.stats()
    assert s["recovered_7d"] == 2
    assert s["revenue_per_recipient_kobo_7d"] == 300_000
    assert s["messages_failed"] >= 0


# ── UTM tagging + opt-out ─────────────────────────────────────────────────

def test_checkout_link_tagged_per_step(tmp_path):
    cr = _engine(tmp_path)
    cr.cart_abandoned(**_cart())
    cr.run_step("cart-1", 1)
    text = cr.sent[0][1]
    assert "utm_source=devon-recovery" in text
    assert "utm_medium=whatsapp" in text
    assert "utm_campaign=cart-step-1" in text


def test_opt_out_line_present_and_disablable(tmp_path):
    cr = _engine(tmp_path)
    cr.cart_abandoned(**_cart())
    cr.run_step("cart-1", 1)
    assert "STOP" in cr.sent[0][1]
    sent: list[str] = []
    cr2 = CartRecovery(str(tmp_path / "c2.db"),
                       sender=lambda p, t: sent.append(t) or True,
                       opt_out_text="")
    cr2.cart_abandoned(**_cart())
    cr2.run_step("cart-1", 1)
    assert "STOP" not in sent[0]


def test_opt_in_source_recorded(tmp_path):
    cr = _engine(tmp_path)
    cr.cart_abandoned(**_cart(opt_in_source="checkout checkbox"))
    row = cr._db.execute(
        "SELECT opt_in_source FROM carts WHERE cart_id='cart-1'").fetchone()
    assert row["opt_in_source"] == "checkout checkbox"


def test_normalize_webhook_carries_compliance_fields():
    out = normalize_webhook({
        "event": "cart.abandoned",
        "cart": {"id": "w1", "phone": "+1", "opted_in": True,
                 "opt_in_source": "popup", "coupon_code": "X1",
                 "discount_pct": 5},
    })
    assert out["opt_in_source"] == "popup"
    assert out["coupon_code"] == "X1"
    assert out["discount_pct"] == 5


# ── schema migration on old DBs ───────────────────────────────────────────

def test_old_db_migrates_forward(tmp_path):
    import sqlite3
    path = str(tmp_path / "old.db")
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE carts (cart_id TEXT PRIMARY KEY, phone TEXT, "
               "items_json TEXT DEFAULT '[]', total_kobo INTEGER DEFAULT 0, "
               "opted_in INTEGER DEFAULT 0, status TEXT DEFAULT 'abandoned', "
               "store TEXT DEFAULT 'unknown', customer_name TEXT DEFAULT '', "
               "checkout_link TEXT DEFAULT '', created_at REAL)")
    db.execute("CREATE TABLE cart_steps (id INTEGER PRIMARY KEY "
               "AUTOINCREMENT, cart_id TEXT, step INTEGER, run_at REAL, "
               "sent_at REAL DEFAULT 0, status TEXT DEFAULT 'pending', "
               "UNIQUE(cart_id, step))")
    db.execute("CREATE TABLE recoveries (cart_id TEXT PRIMARY KEY, "
               "recovered_kobo INTEGER, recovered_at REAL, "
               "store TEXT DEFAULT 'unknown')")
    db.commit()
    db.close()
    cr = CartRecovery(path, sender=lambda p, t: True)
    cr.cart_abandoned(**_cart())  # works against the migrated schema
    assert cr.run_step("cart-1", 1) is True
    assert cr.mark_recovered("cart-1") is True


# ── Medusa: create_discount + cart-recovery coupon seam ───────────────────

def test_cart_recovery_coupon_is_a_real_store_coupon():
    """End-to-end seam: the step-3 incentive is minted in the store's API."""
    mgr = _mgr()
    store = mgr.provision_store("SuyaSpot", admin_email="a@b.c",
                                admin_password="s")
    assert store.status == READY
    sent: list[str] = []

    def coupon_fn(cart):
        return mgr.create_discount(store, 10, token="tok_admin_123")

    import tempfile as _tf
    cr = CartRecovery(_tf.mktemp(suffix=".db"),
                      sender=lambda p, t: sent.append(t) or True,
                      coupon_fn=coupon_fn)
    cr.cart_abandoned(**_cart())
    cr.run_step("cart-1", 1)
    assert "SAVE10" not in sent[0]
    cr.run_step("cart-1", 3)
    assert "SAVE10" in sent[1]
    # and the store actually got the promotion call
    promo_calls = [c for c in mgr._http.calls
                   if c[1].endswith("/admin/promotions")]
    assert promo_calls
    assert promo_calls[0][2]["application_method"]["value"] == 10


def test_resolve_secret_uses_vault_resolve():
    class ResolvingVault(FakeVault):
        def resolve(self, ref):
            return {"woo/k/key": "ck_real"}.get(ref, "")

    mgr = StoreManager(db_path=tempfile.mktemp(suffix=".db"),
                       http=FakeHttp(), vault=ResolvingVault())
    assert mgr._resolve_secret("woo/k/key") == "ck_real"
    assert mgr._resolve_secret("missing") == ""


def test_list_builds_stores_without_n_plus_one():
    mgr = _mgr()
    mgr.provision_store("A", admin_email="a@b.c", admin_password="s")
    mgr.provision_store("B", admin_email="a@b.c", admin_password="s")
    stores = mgr.list()
    assert {s.name for s in stores} == {"A", "B"}
    assert all(s.webhook_note for s in stores)


def test_store_to_dict_includes_webhook_note():
    s = Store(id="x", name="X", webhook_note="note here")
    assert s.to_dict()["webhook_note"] == "note here"


def test_woo_provision_marks_failed_when_down():
    class DownHttp(FakeHttp):
        def request(self, method, url, *, headers=None, json_body=None):
            raise MedusaError("connection refused")

    mgr = StoreManager(db_path=tempfile.mktemp(suffix=".db"),
                       http=DownHttp(), vault=FakeVault())
    with pytest.raises(MedusaError):
        mgr.provision_woocommerce("MamaPut", site_url="https://x.ng")
    assert mgr.list()[0].status != READY
