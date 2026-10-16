"""Tests for the Commerce area (payments, reconciliation, storefront,
logistics, support, tax, analytics, catalog).

Invariants under test:
- payments never fake success (missing key / no HTTP → PaymentError)
- reconciliation balances (debits == credits, no double-counting)
- tax: Nigeria VAT is exactly 7.5%
- dunning retries follow the 1/3/7-day schedule, then past_due
- analytics sums match the ledger exactly
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from nomorals.commerce import payments as pay_mod
from nomorals.commerce.payments import (
    PaymentError, PaymentGateway, provider_for)
from nomorals.commerce.reconciliation import Reconciler
from nomorals.commerce.storefront import (
    Product, cover_brief, cover_prompt, generate_cover, product_page_data)
from nomorals.commerce.logistics import (
    Address, validate_address, rate_quote, eta, tracking, CARRIERS)
from nomorals.commerce.support import SupportInbox
from nomorals.commerce.tax import compute_tax, jurisdictions
from nomorals.commerce.analytics import (
    revenue_by_day, revenue_by_channel, monthly_client_report,
    period_summary)
from nomorals.commerce.catalog import Catalog, diff_catalogs

FIXED_NOW = 1_800_000_000.0


def _fake_http_ok(method, url, headers, json_body):
    return 200, {"status": "ok", "url": url, "body": json_body}


def _fake_http_fail(method, url, headers, json_body):
    return 402, {"error": "card declined"}


def _gateway(**kw):
    return PaymentGateway(
        keys={"paystack": "sk_test_x", "flutterwave": "flw_test_x",
              "monnify": "mon_test_x", "mpesa": "mp_test_x",
              "stripe": "sk_stripe_x"},
        http=_fake_http_ok, now=lambda: FIXED_NOW, **kw)


# ── payments: never fake success ──────────────────────────────────────────

def test_charge_missing_key_raises():
    gw = PaymentGateway(keys={}, http=_fake_http_ok)
    with pytest.raises(PaymentError, match="secret key missing"):
        gw.charge("paystack", 100_000)


def test_charge_no_http_raises_even_with_key():
    gw = PaymentGateway(keys={"paystack": "sk_test_x"}, http=None)
    with pytest.raises(PaymentError, match="no HTTP client"):
        gw.charge("paystack", 100_000)


def test_charge_gateway_rejection_raises():
    gw = PaymentGateway(keys={"paystack": "sk_test_x"},
                        http=_fake_http_fail, now=lambda: FIXED_NOW)
    with pytest.raises(PaymentError, match="rejected"):
        gw.charge("paystack", 100_000)


def test_charge_negative_amount_raises():
    with pytest.raises(PaymentError):
        _gateway().charge("paystack", -5)


def test_charge_success_records_fee_and_net():
    gw = _gateway()
    p = gw.charge("paystack", 1_000_000, customer="Ada",
                  channel="whatsapp")
    assert p.status == "succeeded"
    # Paystack: 1.5% of 1,000,000 = 15,000 + 10,000 fixed = 25,000 (under cap)
    assert p.fee_minor == 25_000
    assert p.net_minor == 975_000
    assert p.channel == "whatsapp"


def test_fee_caps_apply():
    assert provider_for("paystack").fee_for(100_000_000) == 200_000
    assert provider_for("flutterwave").fee_for(100_000_000) == 200_000
    assert provider_for("monnify").fee_for(100_000_000) == 100_000


def test_unknown_provider_raises():
    with pytest.raises(PaymentError):
        provider_for("nope")


def test_refund_requires_real_http_and_matching_amount():
    gw = _gateway()
    p = gw.charge("paystack", 500_000)
    with pytest.raises(PaymentError):
        gw.refund(p, amount_minor=600_000)
    refund = gw.refund(p, amount_minor=500_000, reason="customer request")
    assert refund.status == "refunded"
    assert refund.amount_minor == -500_000
    assert p.status == "refunded"


def test_webhook_verify_hmac():
    gw = _gateway()
    body = b'{"event": "charge.success"}'
    sig = hmac.new(b"sk_test_x", body, hashlib.sha256).hexdigest()
    assert gw.webhook_verify("paystack", body, sig) is True
    assert gw.webhook_verify("paystack", body, "deadbeef") is False
    assert gw.webhook_verify("paystack", body, "") is False


def test_subscription_and_dunning_schedule():
    gw = _gateway()
    sub = gw.create_subscription("paystack", "Ada", 250_000)
    assert sub.status == "active"
    # Retry 1 → +1 day
    gw.record_failed_charge(sub)
    assert sub.failed_attempts == 1
    assert sub.next_retry_at == FIXED_NOW + 1 * 86400
    # Retry 2 → +3 days
    gw.record_failed_charge(sub)
    assert sub.next_retry_at == FIXED_NOW + 3 * 86400
    # Retry 3 → +7 days
    gw.record_failed_charge(sub)
    assert sub.next_retry_at == FIXED_NOW + 7 * 86400
    # Retry 4 → exhausted → past_due, no more retries
    gw.record_failed_charge(sub)
    assert sub.status == "past_due"
    assert sub.next_retry_at == 0.0
    assert gw.dunning_due() == []


def test_dunning_due_lists_retry_ready():
    times = [FIXED_NOW]
    gw = PaymentGateway(keys={"paystack": "sk_test_x"}, http=_fake_http_ok,
                        now=lambda: times[0])
    sub = gw.create_subscription("paystack", "Ada", 250_000)
    gw.record_failed_charge(sub)  # next retry +1d
    assert gw.dunning_due() == []
    times[0] = FIXED_NOW + 2 * 86400
    assert gw.dunning_due() == [sub]


def test_mpesa_no_subscriptions():
    with pytest.raises(PaymentError, match="does not support subscriptions"):
        _gateway().create_subscription("mpesa", "Kamau", 100_000)


def test_invoice_mark_paid_requires_matching_payment():
    gw = _gateway()
    inv = gw.create_invoice("Ada", 500_000)
    p = gw.charge("paystack", 400_000)
    with pytest.raises(PaymentError, match="does not match"):
        gw.mark_paid(inv, p)
    p2 = gw.charge("paystack", 500_000)
    gw.mark_paid(inv, p2)
    assert inv.status == "paid"
    assert inv.payment_reference == p2.reference


def test_dispute_lifecycle():
    gw = _gateway()
    d = gw.record_dispute("ref_1", 100_000, "fraud")
    assert d.status == "open"
    gw.resolve_dispute(d, "won", "evidence accepted")
    assert d.status == "won"
    with pytest.raises(PaymentError):
        gw.resolve_dispute(d, "won")
    with pytest.raises(PaymentError):
        gw.resolve_dispute(gw.record_dispute("r", 1, "x"), "maybe")


def test_cancel_subscription_idempotent():
    gw = _gateway()
    sub = gw.create_subscription("stripe", "Bob", 1000, currency="USD")
    gw.cancel_subscription(sub)
    assert sub.status == "cancelled"
    gw.cancel_subscription(sub)
    assert sub.status == "cancelled"


# ── reconciliation: balances ──────────────────────────────────────────────

def _paid_pair(gw=None):
    gw = gw or _gateway()
    inv = gw.create_invoice("Ada", 500_000)
    p = gw.charge("paystack", 500_000)
    gw.mark_paid(inv, p)
    return gw, inv, p


def test_reconciliation_balances_and_posts_to_ledger():
    gw, inv, p = _paid_pair()
    rec = Reconciler(now=lambda: FIXED_NOW)
    entry = rec.reconcile(p, inv)
    assert entry.debit_minor == entry.credit_minor == 500_000
    assert rec.balanced is True
    totals = rec.totals()
    assert totals["gross_minor"] == 500_000
    assert totals["fees_minor"] == p.fee_minor
    # Ledger received income + fee rows
    txns = rec.ledger.transactions()
    kinds = [t.kind for t in txns]
    assert "income" in kinds and "spend" in kinds
    assert sum(t.amount_kobo for t in txns if t.kind == "income") == 500_000


def test_reconciliation_rejects_duplicates_and_unpaid():
    gw, inv, p = _paid_pair()
    rec = Reconciler()
    rec.reconcile(p, inv)
    with pytest.raises(ValueError, match="already reconciled"):
        rec.reconcile(p, inv)
    # Unpaid invoice
    inv2 = gw.create_invoice("Bob", 100_000)
    p2 = gw.charge("paystack", 100_000)
    with pytest.raises(ValueError, match="mark it paid"):
        rec.reconcile(p2, inv2)
    # Failed payment
    from nomorals.commerce.payments import Payment
    bad = Payment(id="x", provider="paystack", amount_minor=1,
                  currency="NGN", reference="bad", status="failed")
    with pytest.raises(ValueError, match="cannot reconcile"):
        rec.reconcile(bad, None)


def test_reconciliation_unmatched_payment_flagged_not_dropped():
    gw = _gateway()
    p = gw.charge("flutterwave", 250_000)
    rec = Reconciler()
    entry = rec.reconcile(p, None)  # no invoice
    assert entry.invoice_id == ""
    assert len(rec.unreconciled) == 1
    assert "unreconciled" in rec.unreconciled_report().lower()
    assert rec.balanced is True


def test_reconciliation_refund_reverses():
    gw, inv, p = _paid_pair()
    rec = Reconciler()
    rec.reconcile(p, inv)
    refund = gw.refund(p, amount_minor=500_000)
    entry = rec.reconcile(refund, inv)
    assert entry.kind == "refund"
    assert rec.balanced is True
    assert rec.totals()["refunds_minor"] == 500_000


# ── storefront: honest brief-only ────────────────────────────────────────

def _product():
    return Product(id="p1", name="Smoked Catfish 500g",
                   price_minor=450_000, description="Oak-smoked, Lagos.",
                   tags=["smoked", "fish"], stock=12,
                   brand_voice="warm market stall")


def test_cover_brief_never_claims_image():
    brief = cover_brief(_product())
    assert brief["generated"] is False
    assert brief["image"] is None
    assert "Catfish" in brief["subject"] or "catfish" in brief["subject"].lower()
    assert "watermark" in brief["avoid"]


def test_cover_prompt_is_text():
    prompt = cover_prompt(_product())
    assert isinstance(prompt, str) and "Catfish" in prompt


def test_generate_cover_without_imggen_is_brief_only():
    out = generate_cover(_product())
    assert out["generated"] is False
    assert out["image"] is None
    assert "No image generator" in out["note"]


def test_generate_cover_with_imggen_uses_real_result():
    out = generate_cover(_product(), imggen=lambda prompt: "/tmp/cover.png")
    assert out["generated"] is True
    assert out["image"] == "/tmp/cover.png"


def test_generate_cover_imggen_failure_reported_not_faked():
    def boom(prompt):
        raise RuntimeError("GPU down")
    out = generate_cover(_product(), imggen=boom)
    assert out["generated"] is False
    assert out["image"] is None
    assert "GPU down" in out["error"]


def test_product_page_data_no_invented_reviews():
    page = product_page_data(_product())
    assert page["reviews"] == []
    assert page["in_stock"] is True
    assert page["cover"]["generated"] is False


# ── logistics ─────────────────────────────────────────────────────────────

def _addr(**over):
    base = {"line1": "14 Allen Avenue", "locality": "Ikeja",
            "state": "Lagos", "postal_code": "100001",
            "country": "NG", "phone": "+2348012345678",
            "recipient": "Ada"}
    base.update(over)
    return Address(**base)


def test_address_nigeria_valid():
    result = validate_address(_addr())
    assert result["ok"] is True
    assert result["errors"] == []


def test_address_nigeria_rejects_bad_state_postal_phone():
    bad = validate_address(_addr(state="Narnia", postal_code="123",
                                 phone="555"))
    assert bad["ok"] is False
    assert len(bad["errors"]) == 3
    # "Abuja" normalises to FCT
    assert validate_address(_addr(state="Abuja"))["normalized"].state == "FCT"


def test_address_generic_requires_locality_country():
    assert validate_address(
        Address(line1="1 Main St", locality="Accra",
                country="GH"))["ok"] is True
    assert validate_address(Address(line1="", locality=""))["ok"] is False


def test_rate_quote_is_labelled_estimate():
    quote = rate_quote("gig", origin=_addr(), destination=_addr(),
                       weight_kg=2.0, distance_km=40.0)
    assert quote["estimated"] is True
    assert "offline tariff table" in quote["source"]
    assert quote["fee_minor"] > 0
    # Heavier costs more; unknown carrier raises
    heavier = rate_quote("gig", origin=_addr(), destination=_addr(),
                         weight_kg=5.0, distance_km=40.0)
    assert heavier["fee_minor"] > quote["fee_minor"]
    with pytest.raises(ValueError):
        rate_quote("dhl", origin=_addr(), destination=_addr(), weight_kg=1.0)
    with pytest.raises(ValueError):
        rate_quote("gig", origin=_addr(), destination=_addr(), weight_kg=0)


def test_rate_quote_live_passthrough_not_edited():
    live = {"fee_minor": 123_456, "currency": "NGN"}
    out = rate_quote("kwik", origin=_addr(), destination=_addr(),
                     weight_kg=1.0,
                     live_quote=lambda **kw: dict(live))
    assert out["fee_minor"] == 123_456
    assert out["estimated"] is False


def test_eta_bands():
    assert "hours" in eta("kwik")["eta"]
    assert "days" in eta("gig", distance_km=600)["eta"]
    assert eta("sendbox")["estimated"] is True


def test_tracking_honest_not_available():
    result = tracking("gig", "GIG123456")
    assert result["status"] == "not_available"
    assert "No live tracker" in result["reason"]
    # With a real tracker, the real result comes back
    real = tracking("gig", "GIG123456",
                    tracker=lambda **kw: {"status": "in_transit",
                                          "location": "Lagos hub"})
    assert real["status"] == "in_transit"
    assert real["location"] == "Lagos hub"
    # No waybill → not_available, not invented
    assert tracking("gig", "")["status"] == "not_available"


def test_carrier_descriptors_present():
    assert set(CARRIERS) == {"gig", "kwik", "sendbox"}
    assert all(c.name for c in CARRIERS.values())


# ── support ───────────────────────────────────────────────────────────────

def test_support_ticket_lifecycle_and_sla():
    inbox = SupportInbox(now=lambda: FIXED_NOW)
    t = inbox.open("Ada", "Where is my order?", priority="high",
                   customer_phone="+2348012345678")
    assert t.status == "open"
    assert t.sla_deadline() == FIXED_NOW + 4 * 3600
    inbox.assign(t, "agent-1")
    assert t.status == "assigned" and t.assignee == "agent-1"
    draft = inbox.reply_draft(t, "order_status", reference="ORD-1",
                              status_text="shipped", detail="Arrives tomorrow.")
    assert draft["draft"] is True and draft["sent"] is False
    assert "ORD-1" in draft["text"]
    assert t.first_response_at == FIXED_NOW
    inbox.close(t, "Delivered.")
    assert t.status == "closed" and t.closed_at == FIXED_NOW


def test_support_sla_breach_and_stats():
    times = [FIXED_NOW]
    inbox = SupportInbox(now=lambda: times[0])
    t = inbox.open("Ada", "Urgent!", priority="urgent")
    assert inbox.breached() == []
    times[0] = FIXED_NOW + 2 * 3600  # past the 1h urgent SLA, no reply
    assert inbox.breached() == [t]
    stats = inbox.stats()
    assert stats["total"] == 1 and stats["sla_breached"] == 1
    assert stats["by_channel"] == {"whatsapp": 1}


def test_support_validation():
    inbox = SupportInbox()
    with pytest.raises(ValueError):
        inbox.open("", "")
    with pytest.raises(ValueError):
        inbox.open("Ada", "hi", priority="critical")
    t = inbox.open("Ada", "hi")
    with pytest.raises(ValueError):
        inbox.reply_draft(t, "nonexistent_template")
    inbox.close(t)
    with pytest.raises(ValueError):
        inbox.assign(t, "agent-1")


# ── tax ───────────────────────────────────────────────────────────────────

def test_tax_nigeria_7_5_pct():
    result = compute_tax(1_000_000, "NG")
    assert result["rate_bps"] == 750
    assert result["tax_minor"] == 75_000
    assert result["total_minor"] == 1_075_000
    assert result["tax_name"] == "VAT"


def test_tax_integer_math_no_float_drift():
    # 7.5% of ₦333.33 (33333 kobo) = 2499.975 → floor 2499 kobo
    result = compute_tax(33_333, "NG")
    assert result["tax_minor"] == 2_499
    assert result["tax_minor"] + result["net_minor"] == result["total_minor"]


def test_tax_other_jurisdictions_and_overrides():
    assert compute_tax(1_000_000, "GB")["tax_minor"] == 200_000
    assert compute_tax(1_000_000, "US")["tax_minor"] == 0  # no federal VAT
    overridden = compute_tax(1_000_000, "NG", overrides={"NG": 500})
    assert overridden["tax_minor"] == 50_000
    with pytest.raises(ValueError):
        compute_tax(100, "XX")
    with pytest.raises(ValueError):
        compute_tax(-1, "NG")


def test_jurisdiction_table_descriptors():
    table = jurisdictions()
    assert "NG" in table and table["NG"].tax_name == "VAT"
    assert table["US"].notes  # explains the 0 rate


# ── analytics: sums match ledger ─────────────────────────────────────────

def _ledger_with_txns():
    from nomorals.commerce.reconciliation import InMemoryLedger
    ledger = InMemoryLedger()
    # 2026-10-01 and 2026-10-02 UTC
    import datetime as _dt
    t1 = _dt.datetime(2026, 10, 1, 12, tzinfo=_dt.timezone.utc).timestamp()
    t2 = _dt.datetime(2026, 10, 2, 12, tzinfo=_dt.timezone.utc).timestamp()
    ledger.log(500_000, "sales", "payment a", kind="income", ts=t1)
    ledger.log(300_000, "sales", "payment b", kind="income", ts=t2)
    ledger.log(20_000, "fees", "gateway fee", kind="spend", ts=t2)
    ledger.log(100_000, "sales_refund", "refund", kind="spend", ts=t2)
    return ledger, t1, t2


def test_analytics_sums_match_ledger():
    ledger, t1, t2 = _ledger_with_txns()
    summary = period_summary(ledger, since=t1 - 1, until=t2 + 86400)
    assert summary["gross_minor"] == 800_000
    assert summary["fees_minor"] == 20_000
    assert summary["refunds_minor"] == 100_000
    assert summary["net_minor"] == 680_000
    assert summary["income_txns"] == 2
    daily = revenue_by_day(ledger, since=t1 - 1, until=t2 + 86400)
    assert daily == {"2026-10-01": 500_000, "2026-10-02": 300_000}
    assert sum(daily.values()) == summary["gross_minor"]


def test_analytics_channel_split_only_counts_succeeded():
    from nomorals.commerce.payments import Payment
    payments = [
        Payment(id="1", provider="paystack", amount_minor=500_000,
                currency="NGN", reference="r1", status="succeeded",
                channel="whatsapp", created_at=FIXED_NOW),
        Payment(id="2", provider="paystack", amount_minor=999_000,
                currency="NGN", reference="r2", status="failed",
                channel="whatsapp", created_at=FIXED_NOW),
        Payment(id="3", provider="stripe", amount_minor=100_000,
                currency="USD", reference="r3", status="succeeded",
                channel="online", created_at=FIXED_NOW),
    ]
    split = revenue_by_channel(payments)
    assert split == {"online": 100_000, "whatsapp": 500_000}


def test_monthly_report_answers_with_real_numbers():
    ledger, t1, _t2 = _ledger_with_txns()
    report = monthly_client_report(ledger, 2026, 10, store="Ada's Store")
    assert "₦8,000.00" in report  # gross
    assert "₦6,800.00" in report  # net
    assert "Best day" in report
    assert "how well did i run your business" in report.lower()


def test_monthly_report_empty_month_is_honest():
    from nomorals.commerce.reconciliation import InMemoryLedger
    report = monthly_client_report(InMemoryLedger(), 2026, 11,
                                    store="Ada's Store")
    assert "No recorded transactions" in report
    assert "₦0.00" in report


# ── catalog ───────────────────────────────────────────────────────────────

def _prod(pid, price, name="Item", stock=1):
    return {"id": pid, "name": name, "price_minor": price,
            "currency": "NGN", "description": "", "category": "food",
            "stock": stock, "tags": ["a"]}


def test_catalog_crud_and_feed_export():
    cat = Catalog()
    cat.upsert(_prod("p1", 100_000))
    cat.upsert(_prod("p2", 200_000))
    assert len(cat) == 2
    assert cat.get("p1")["price_minor"] == 100_000
    data = json.loads(cat.feed_export("json"))
    assert len(data) == 2
    csv_text = cat.feed_export("csv")
    assert "p1" in csv_text and "price_minor" in csv_text.splitlines()[0]
    with pytest.raises(ValueError):
        cat.upsert({"name": "no id"})
    with pytest.raises(ValueError):
        cat.feed_export("xml")
    assert cat.remove("p1") is True
    assert cat.remove("p1") is False


def test_catalog_sync_diff_report():
    cat = Catalog([_prod("p1", 100_000), _prod("p2", 200_000),
                   _prod("p3", 300_000)])
    remote = [_prod("p1", 100_000),  # unchanged
              _prod("p2", 250_000),  # changed price
              _prod("p4", 400_000)]  # added; p3 removed
    diff = cat.sync(remote)
    assert diff["summary"] == {"local": 3, "remote": 3, "added": 1,
                               "removed": 1, "changed": 1, "unchanged": 1}
    assert diff["added"][0]["id"] == "p4"
    assert diff["removed"][0]["id"] == "p3"
    assert diff["changed"][0]["id"] == "p2"
    assert diff["changed"][0]["fields"]["price_minor"] == {
        "local": 200_000, "remote": 250_000}
    # Pure-function diff agrees
    assert diff_catalogs(cat.list(), remote)["summary"] == diff["summary"]


def test_catalog_apply_diff_removals_opt_in():
    cat = Catalog([_prod("p1", 100_000), _prod("p2", 200_000)])
    diff = cat.sync([_prod("p2", 250_000), _prod("p3", 300_000)])
    counts = cat.apply_diff(diff)  # removals NOT applied by default
    assert counts == {"added": 1, "changed": 1, "removed": 0}
    assert cat.get("p1") is not None  # still there
    assert cat.get("p2")["price_minor"] == 250_000
    counts2 = cat.apply_diff(cat.sync([_prod("p2", 250_000),
                                       _prod("p3", 300_000)]),
                             apply_removed=True)
    assert counts2["removed"] == 1
    assert cat.get("p1") is None
