"""Offline tests for the WhatsApp cost-awareness layer (#68)."""

from __future__ import annotations

import pytest

from nomorals.social.whatsapp_cost import (
    RATE_MARKETING_KOBO,
    RATE_SERVICE_KOBO,
    CampaignEstimate,
    CostAwareSender,
    CostTracker,
    naira,
    worth_it,
)


def tracker(**kw):
    return CostTracker(":memory:", **kw)


# ── accounting ──────────────────────────────────────────────────────────


def test_rates_are_sane():
    assert RATE_SERVICE_KOBO == 1_400      # ₦14
    assert RATE_MARKETING_KOBO == 8_400    # ₦84


def test_track_records_cost():
    t = tracker()
    assert t.track("+2348000000001", "service") == RATE_SERVICE_KOBO
    assert t.track("+2348000000002", "marketing") == RATE_MARKETING_KOBO
    assert t.spent_week() == RATE_SERVICE_KOBO + RATE_MARKETING_KOBO
    assert t.message_count() == 2


def test_track_unknown_category_raises():
    t = tracker()
    with pytest.raises(ValueError):
        t.track("+2348000000001", "carrier-pigeon")


def test_weekly_spend_format():
    t = tracker()
    t.track("a", "service")
    out = t.weekly_spend()
    assert "₦14" in out and "WhatsApp" in out and "1 message" in out


def test_spend_vs_revenue():
    t = tracker()
    t.track("a", "marketing")  # ₦84
    out = t.spend_vs_revenue(47_000_00)  # ₦47,000 recovered
    assert "₦84" in out and "₦47,000" in out
    assert "return on messaging spend" in out


def test_spend_vs_revenue_no_spend():
    t = tracker()
    out = t.spend_vs_revenue(47_000_00)
    assert "₦0" in out and "₦47,000" in out


# ── budgets ─────────────────────────────────────────────────────────────


def test_no_budget_is_unlimited():
    t = tracker()
    t.track("a", "marketing")  # client="default"
    ok, spent, budget = t.check_budget("nobody")
    assert ok is True and budget == 0 and spent == 0
    ok2, spent2, _ = t.check_budget("default")
    assert ok2 is True and spent2 == RATE_MARKETING_KOBO


def test_hard_budget_blocks():
    t = tracker()
    t.set_budget("shop", RATE_MARKETING_KOBO)  # one message
    t.track("a", "marketing", client="shop")
    ok, spent, budget = t.check_budget("shop")
    assert ok is False and spent == RATE_MARKETING_KOBO


def test_hard_budget_allows_under():
    t = tracker()
    t.set_budget("shop", RATE_MARKETING_KOBO * 5)
    t.track("a", "marketing", client="shop")
    ok, _, _ = t.check_budget("shop")
    assert ok is True


def test_soft_budget_warns_not_blocks():
    t = tracker()
    t.set_budget("shop", RATE_MARKETING_KOBO, hard=False)
    t.track("a", "marketing", client="shop")
    ok, _, _ = t.check_budget("shop")
    assert ok is True  # soft: warn, don't block


def test_budget_warning_threshold():
    t = tracker()
    t.set_budget("shop", RATE_MARKETING_KOBO * 10)
    assert t.budget_warning("shop") == ""
    for i in range(8):  # 80% consumed
        t.track(f"p{i}", "marketing", client="shop")
    warn = t.budget_warning("shop")
    assert "⚠️" in warn and "shop" in warn


def test_budgets_are_per_client():
    t = tracker()
    t.set_budget("a", RATE_MARKETING_KOBO)
    t.track("x", "marketing", client="a")
    ok_a, _, _ = t.check_budget("a")
    ok_b, _, _ = t.check_budget("b")
    assert ok_a is False and ok_b is True


# ── CostAwareSender ─────────────────────────────────────────────────────


def test_sender_tracks_successful_sends():
    t = tracker()
    sent = []
    aware = CostAwareSender(lambda p, x: sent.append((p, x)) or True,
                            tracker=t, client="shop")
    assert aware("+2341", "hello") is True
    assert t.spent_week(client="shop") == RATE_MARKETING_KOBO  # default category


def test_sender_does_not_track_failed_sends():
    t = tracker()
    aware = CostAwareSender(lambda p, x: False, tracker=t)
    assert aware("+2341", "hello") is False
    assert t.spent_week() == 0


def test_sender_blocks_on_hard_budget():
    t = tracker()
    t.set_budget("shop", RATE_SERVICE_KOBO)
    calls = []
    aware = CostAwareSender(lambda p, x: calls.append(1) or True,
                            tracker=t, client="shop", category="service")
    assert aware("+2341", "one") is True
    assert aware("+2341", "two") is False  # budget spent → blocked
    assert len(calls) == 1  # wrapped sender never saw the second


def test_sender_never_raises():
    def boom(p, x):
        raise RuntimeError("bridge exploded")
    aware = CostAwareSender(boom, tracker=tracker())
    assert aware("+2341", "hi") is False


def test_estimate_campaign_text():
    aware = CostAwareSender(lambda p, x: True, tracker=tracker())
    est = aware.estimate_campaign(100, "marketing")
    assert isinstance(est, CampaignEstimate)
    assert est.total_kobo == 100 * RATE_MARKETING_KOBO
    assert est.text == (
        "this campaign will cost ₦8,400 for 100 customers "
        "(marketing, ₦84 each)"
    )


def test_estimate_flags_budget_breach():
    t = tracker()
    t.set_budget("shop", 1_000_00)  # ₦1,000
    aware = CostAwareSender(lambda p, x: True, tracker=t, client="shop")
    est = aware.estimate_campaign(100, "marketing")  # ₦8,400
    assert est.within_budget is False


def test_campaign_approval_fail_closed():
    aware = CostAwareSender(lambda p, x: True, tracker=tracker())
    est = aware.estimate_campaign(10)
    assert aware.request_campaign_approval(est) is False  # no fn → no send
    assert aware.request_campaign_approval(est, lambda e: True) is True
    assert aware.request_campaign_approval(est, lambda e: False) is False


# ── batching ────────────────────────────────────────────────────────────


def test_batch_combines_per_recipient():
    t = tracker()
    sent = []
    aware = CostAwareSender(lambda p, x: sent.append((p, x)) or True,
                            tracker=t, category="service")
    aware.queue("+2341", "order shipped")
    aware.queue("+2341", "tracking: XYZ")
    aware.queue("+2342", "order shipped")
    assert aware.queued == 3
    assert aware.flush() == 2  # two recipients → two messages
    assert aware.queued == 0
    by_phone = {p: x for p, x in sent}
    assert "order shipped" in by_phone["+2341"]
    assert "tracking: XYZ" in by_phone["+2341"]
    # two messages tracked, not three
    assert t.message_count() == 2


def test_batch_respects_budget():
    t = tracker()
    t.set_budget("default", RATE_SERVICE_KOBO)  # one message
    sent = []
    aware = CostAwareSender(lambda p, x: sent.append(1) or True,
                            tracker=t, category="service")
    aware.queue("+2341", "a")
    aware.queue("+2342", "b")
    assert aware.flush() == 1
    assert len(sent) == 1


# ── worth_it ────────────────────────────────────────────────────────────


def test_worth_it_cart_recovery():
    # ₦84 marketing message chasing a ₦47,000 cart at 30% recovery
    assert worth_it(47_000_00 * 0.30, RATE_MARKETING_KOBO) is True


def test_worth_it_not_worth_it():
    # ₦100 expected vs ₦84 cost at 2× ROI → 100 < 168 → not worth it
    assert worth_it(10_000, RATE_MARKETING_KOBO) is False


def test_worth_it_custom_roi():
    assert worth_it(10_000, 8_400, min_roi=1.0) is True
    assert worth_it(10_000, 8_400, min_roi=2.0) is False


def test_worth_it_zero_cost():
    assert worth_it(1, 0) is True
    assert worth_it(0, 0) is False


# ── #66 integration ─────────────────────────────────────────────────────


def test_wraps_cart_recovery_sender():
    from nomorals.commerce.cart_recovery import CartRecovery

    clock = [1_000_000.0]
    underlying = []
    t = tracker()
    aware = CostAwareSender(
        lambda p, x: underlying.append((p, x)) or True,
        tracker=t, client="suyaspot", category="marketing",
    )
    cr = CartRecovery(db_path=":memory:", sender=aware,
                      now=lambda: clock[0])
    cr.cart_abandoned("c1", "+2348000000001", [{"name": "Suya"}],
                      47_000_00, opted_in=True, store="SuyaSpot")
    # advance past all three steps
    clock[0] += 30 * 60 + 1
    cr.run_due_steps()
    clock[0] += 24 * 3600
    cr.run_due_steps()
    clock[0] += 48 * 3600
    cr.run_due_steps()

    assert len(underlying) == 3  # all three messages went out
    # cost-awareness tracked all three as marketing
    assert t.message_count(client="suyaspot") == 3
    assert t.spent_week(client="suyaspot") == 3 * RATE_MARKETING_KOBO
    # and the combined transparency report works
    cr.mark_recovered("c1")
    report = t.spend_vs_revenue(47_000_00, client="suyaspot")
    assert "₦252" in report and "₦47,000" in report  # 3 × ₦84 = ₦252


def test_budget_blocks_cart_recovery_sequence():
    from nomorals.commerce.cart_recovery import CartRecovery

    clock = [1_000_000.0]
    underlying = []
    t = tracker()
    t.set_budget("suyaspot", RATE_MARKETING_KOBO)  # exactly one message
    aware = CostAwareSender(
        lambda p, x: underlying.append((p, x)) or True,
        tracker=t, client="suyaspot", category="marketing",
    )
    cr = CartRecovery(db_path=":memory:", sender=aware,
                      now=lambda: clock[0])
    cr.cart_abandoned("c1", "+2348000000001", [{"name": "Suya"}],
                      47_000_00, opted_in=True)
    clock[0] += 30 * 60 + 1
    cr.run_due_steps()
    clock[0] += 24 * 3600
    cr.run_due_steps()
    # budget exhausted after the first message: sequence halted, no crash
    assert len(underlying) == 1
    assert t.spent_week(client="suyaspot") == RATE_MARKETING_KOBO
