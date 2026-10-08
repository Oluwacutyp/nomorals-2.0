"""Offline tests for nomorals/commerce/cart_recovery.py (build-map #66)."""

from __future__ import annotations

import pytest

from nomorals.commerce.cart_recovery import (
    CartRecovery,
    STEPS,
    normalize_webhook,
)


@pytest.fixture()
def cr(tmp_path):
    sent: list[tuple[str, str]] = []

    def sender(phone: str, text: str) -> bool:
        sent.append((phone, text))
        return True

    engine = CartRecovery(str(tmp_path / "carts.db"), sender=sender)
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


# ── opt-in gate ───────────────────────────────────────────────────────────

def test_no_opt_in_no_messages(cr):
    cr.cart_abandoned(**_cart(opted_in=False))
    # advance clock past all steps
    cr._now = lambda: 10**10
    assert cr.run_due_steps() == 0
    assert cr.sent == []


def test_no_opt_in_no_steps_scheduled(cr):
    cr.cart_abandoned(**_cart(opted_in=False))
    rows = cr._db.execute("SELECT COUNT(*) AS n FROM cart_steps").fetchone()
    assert rows["n"] == 0


def test_opt_in_schedules_three_steps(cr):
    cr.cart_abandoned(**_cart())
    rows = cr._db.execute(
        "SELECT step, run_at FROM cart_steps ORDER BY step"
    ).fetchall()
    assert [r["step"] for r in rows] == [1, 2, 3]
    # timing: 30min, 24h, 72h
    delays = [r["run_at"] for r in rows]
    base = rows[0]["run_at"] - 30 * 60
    assert abs(delays[1] - (base + 24 * 3600)) < 2
    assert abs(delays[2] - (base + 72 * 3600)) < 2


# ── 3-step sequence ───────────────────────────────────────────────────────

def test_step_1_sends_reminder(cr):
    cr.cart_abandoned(**_cart())
    cr._now = lambda: 10**10  # way past step 1, before step 2? no — all due
    # run only step 1 manually
    assert cr.run_step("cart-1", 1) is True
    assert len(cr.sent) == 1
    phone, text = cr.sent[0]
    assert phone == "+2348012345678"
    assert "forgot" in text.lower() or "left something" in text.lower()
    assert "Ada" in text
    assert "₦2,500" in text


def test_step_2_and_3_templates(cr):
    cr.cart_abandoned(**_cart())
    cr.run_step("cart-1", 2)
    assert "thinking" in cr.sent[0][1].lower()
    cr.run_step("cart-1", 3)
    assert "last call" in cr.sent[1][1].lower()


def test_max_three_messages(cr):
    cr.cart_abandoned(**_cart())
    for step in (1, 2, 3):
        cr.run_step("cart-1", step)
    assert len(cr.sent) == 3
    # no step 4 exists
    assert cr.run_step("cart-1", 4) is False
    assert len(cr.sent) == 3


def test_run_due_steps_sends_only_due(cr):
    now = [1_000_000.0]
    cr2 = CartRecovery.__new__(CartRecovery)
    # simpler: use cr with injected clock via monkeypatch of _now
    cr.cart_abandoned(**_cart())
    # only step 1 due (30 min passed, 24h not)
    first_run = cr._db.execute(
        "SELECT run_at FROM cart_steps WHERE cart_id='cart-1' AND step=1"
    ).fetchone()["run_at"]
    cr._now = lambda: first_run + 10
    assert cr.run_due_steps() == 1
    assert len(cr.sent) == 1
    # nothing else due yet
    assert cr.run_due_steps() == 0
    assert len(cr.sent) == 1


# ── recovery stops the sequence ───────────────────────────────────────────

def test_recovery_cancels_remaining_steps(cr):
    cr.cart_abandoned(**_cart())
    cr.run_step("cart-1", 1)
    assert cr.mark_recovered("cart-1") is True
    # steps 2/3 must not send
    assert cr.run_step("cart-1", 2) is False
    assert cr.run_step("cart-1", 3) is False
    assert len(cr.sent) == 1


def test_recovery_records_revenue(cr):
    cr.cart_abandoned(**_cart())
    cr.mark_recovered("cart-1")
    row = cr._db.execute(
        "SELECT recovered_kobo FROM recoveries WHERE cart_id='cart-1'"
    ).fetchone()
    assert row["recovered_kobo"] == 250_000


def test_recovery_unknown_cart(cr):
    assert cr.mark_recovered("nope") is False


# ── weekly report ─────────────────────────────────────────────────────────

def test_weekly_report_empty(cr):
    report = cr.weekly_report()
    assert "no carts recovered" in report.lower()


def test_weekly_report_with_recovery(cr):
    cr.cart_abandoned(**_cart(cart_id="c1"))
    cr.cart_abandoned(**_cart(cart_id="c2", total_kobo=2_200_000))
    cr.mark_recovered("c1")
    cr.mark_recovered("c2")
    report = cr.weekly_report()
    assert "₦24,500" in report
    assert "2 carts" in report
    assert "Devon recovered" in report


def test_stats(cr):
    cr.cart_abandoned(**_cart())
    cr.run_step("cart-1", 1)
    cr.mark_recovered("cart-1")
    s = cr.stats()
    assert s["abandoned_7d"] == 1
    assert s["recovered_7d"] == 1
    assert s["recovered_kobo_7d"] == 250_000
    assert s["messages_sent"] == 1


# ── webhook normalization ─────────────────────────────────────────────────

def test_webhook_shopify():
    event = {
        "topic": "carts/update",
        "store": "ShopA",
        "cart": {
            "id": "shop-1",
            "customer_phone": "+2348099999999",
            "line_items": [{"title": "Suya", "quantity": 2}],
            "total_price": 5000,
            "amount_unit": "major",
            "accepts_marketing": True,
            "abandoned_checkout_url": "https://shopa.test/c/1",
            "customer": {"first_name": "Tunde", "last_name": "O"},
        },
    }
    out = normalize_webhook(event)
    assert out is not None
    assert out["cart_id"] == "shop-1"
    assert out["phone"] == "+2348099999999"
    assert out["total_kobo"] == 500_000
    assert out["opted_in"] is True
    assert out["customer_name"] == "Tunde O"
    assert out["store"] == "ShopA"


def test_webhook_medusa():
    event = {
        "type": "cart.updated",
        "data": {
            "cart": {
                "id": "med-1",
                "phone": "+2348077777777",
                "items": [{"name": "Zobo"}],
                "total_kobo": 150_00,
                "opted_in": False,
                "customer_name": "Kemi",
            }
        },
    }
    out = normalize_webhook(event)
    assert out is not None
    assert out["opted_in"] is False
    assert out["total_kobo"] == 150_00


def test_webhook_woocommerce():
    event = {
        "action": "woocommerce_cart_updated",
        "cart": {"token": "woo-9", "total": 0, "items": []},
    }
    out = normalize_webhook(event)
    assert out is not None
    assert out["cart_id"] == "woo-9"


def test_webhook_unknown_ignored():
    assert normalize_webhook({"type": "order.created"}) is None
    assert normalize_webhook({}) is None
    assert normalize_webhook("garbage") is None  # type: ignore[arg-type]


def test_webhook_flows_into_engine(cr):
    event = {
        "event": "cart.abandoned",
        "cart": {
            "id": "web-1",
            "phone": "+2348011111111",
            "items": [{"name": "Akara"}],
            "total_kobo": 50_000,
            "opted_in": True,
        },
    }
    cart = cr.handle_webhook(event)
    assert cart is not None
    assert cart.opted_in is True
    rows = cr._db.execute(
        "SELECT COUNT(*) AS n FROM cart_steps WHERE cart_id='web-1'"
    ).fetchone()
    assert rows["n"] == 3


# ── sender seam ───────────────────────────────────────────────────────────

def test_sender_seam_used(tmp_path):
    """Custom sender (the #68 cost-aware wrapper shape) is honored."""
    calls: list[tuple[str, str]] = []
    cr = CartRecovery(
        str(tmp_path / "c.db"),
        sender=lambda phone, text: calls.append((phone, text)) or True,
    )
    cr.cart_abandoned(**_cart())
    assert cr.run_step("cart-1", 1) is True
    assert calls and calls[0][0] == "+2348012345678"


def test_sender_failure_never_raises(tmp_path):
    def bad_sender(phone: str, text: str) -> bool:
        raise RuntimeError("bridge down")

    cr = CartRecovery(str(tmp_path / "c.db"), sender=bad_sender)
    cr.cart_abandoned(**_cart())
    # sender raising inside run_step → run_step catches? No — sender is
    # called by run_step; run_step wraps in try/except. Verify:
    assert cr.run_step("cart-1", 1) is False


def test_sender_false_marks_failed(tmp_path):
    cr = CartRecovery(
        str(tmp_path / "c.db"), sender=lambda p, t: False
    )
    cr.cart_abandoned(**_cart())
    assert cr.run_step("cart-1", 1) is False
    row = cr._db.execute(
        "SELECT status FROM cart_steps WHERE cart_id='cart-1' AND step=1"
    ).fetchone()
    assert row["status"] == "failed"


# ── never raises ──────────────────────────────────────────────────────────

def test_intake_never_raises(tmp_path):
    cr = CartRecovery(str(tmp_path / "c.db"))
    cart = cr.cart_abandoned(None, None, None, None, None)  # type: ignore[arg-type]
    assert cart.cart_id is None
