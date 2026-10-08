"""Build-map #75: budget-in-first-reply + points-vs-cash + multi-origin.

All offline. Tests the display rules and their wiring into
#70 (Duffel offers), #71 (price alerts), #72 (itineraries).
"""

from __future__ import annotations

import pytest

from nomorals.travel.display import (
    format_with_budget,
    points_vs_cash,
    multi_origin_search,
    format_multi_origin,
    LoyaltyProgram,
    enrich_offers_text,
    enrich_alert_text,
    enrich_itinerary_text,
)
from nomorals.travel.watchers import PriceAlert, format_alert


def _prog(name="Miles", balance=100_000, award=65_000, cash=8_000_000):
    return LoyaltyProgram(name=name, balance=balance,
                          award_points=award, award_cash_kobo=cash)


# ── format_with_budget ───────────────────────────────────────────────────


def test_total_always_first():
    out = format_with_budget("line1\nline2", 45_000_000)
    assert out.startswith("💰 Total: ₦450,000")
    assert "line1" in out


def test_within_budget():
    out = format_with_budget("x", 45_000_000, budget_kobo=50_000_000)
    assert "within your ₦500k budget" in out


def test_over_budget():
    out = format_with_budget("x", 55_000_000, budget_kobo=50_000_000)
    assert "over budget by ₦50,000" in out


def test_no_budget_no_line():
    out = format_with_budget("x", 45_000_000)
    assert "budget" not in out.lower() or "Total" in out
    assert "over budget" not in out


# ── points_vs_cash ───────────────────────────────────────────────────────


def test_points_vs_cash_format():
    out = points_vs_cash(45_000_000, [_prog()])
    assert "₦450k cash" in out
    assert "65,000 Miles" in out
    assert "₦80k" in out


def test_insufficient_balance_skipped():
    p = _prog(balance=10_000, award=65_000)
    assert points_vs_cash(45_000_000, [p]) == ""


def test_no_programs():
    assert points_vs_cash(45_000_000, []) == ""


def test_multiple_programs():
    out = points_vs_cash(45_000_000, [_prog("Miles"), _prog("Points")])
    assert "Miles" in out and "Points" in out


# ── multi_origin_search ──────────────────────────────────────────────────


def _search_fn(origin, destination):
    prices = {"LOS": "450000.00", "ABV": "380000.00"}
    if origin not in prices:
        return []
    from types import SimpleNamespace
    return [SimpleNamespace(total_amount=prices[origin],
                            total_currency="NGN")]


def test_multi_origin_cheapest():
    r = multi_origin_search(["LOS", "ABV"], "LHR", _search_fn)
    assert r["cheapest"]["origin"] == "ABV"
    assert r["cheapest"]["kobo"] == 38_000_000


def test_multi_origin_failure_tolerated():
    def bad(origin, dest):
        raise RuntimeError("down")
    r = multi_origin_search(["LOS", "XXX"], "LHR", bad)
    assert r["cheapest"] is None
    assert r["checked"][0]["status"] == "failed"


def test_format_multi_origin():
    r = multi_origin_search(["LOS", "ABV"], "LHR", _search_fn)
    out = format_multi_origin(r, "LOS")
    assert "also checked ABV: ₦380k" in out
    assert "cheapest is ABV" in out


# ── wiring: #70 Duffel ──────────────────────────────────────────────────


def test_duffel_format_itinerary_budget():
    from nomorals.connectors.duffel import DuffelConnector, Offer
    c = DuffelConnector.__new__(DuffelConnector)
    offer = Offer(id="o", total_amount="450000.00", total_currency="NGN",
                  airline="Air Peace", origin="LOS", destination="LHR")
    out = c.format_itinerary(offer, budget_kobo=50_000_000,
                             programs=[_prog()])
    assert "within your budget" in out
    assert "🎖️" in out


def test_duffel_format_itinerary_over_budget():
    from nomorals.connectors.duffel import DuffelConnector, Offer
    c = DuffelConnector.__new__(DuffelConnector)
    offer = Offer(id="o", total_amount="550000.00", total_currency="NGN",
                  origin="LOS", destination="LHR")
    out = c.format_itinerary(offer, budget_kobo=50_000_000)
    assert "over budget by ₦50,000" in out


def test_duffel_format_itinerary_backward_compat():
    from nomorals.connectors.duffel import DuffelConnector, Offer
    c = DuffelConnector.__new__(DuffelConnector)
    offer = Offer(id="o", total_amount="450000.00", total_currency="NGN",
                  origin="LOS", destination="LHR")
    out = c.format_itinerary(offer)
    assert "Total:" in out
    assert "budget" not in out.lower()


# ── wiring: #71 alerts ──────────────────────────────────────────────────


def test_alert_with_budget():
    a = PriceAlert(watch_id="w", route="LOS→LHR",
                   previous_kobo=45_000_000, current_kobo=37_500_000,
                   savings_pct=17.0)
    out = format_alert(a, budget_kobo=40_000_000)
    assert "dropped" in out
    assert "within your ₦400k budget" in out


def test_alert_over_budget():
    a = PriceAlert(watch_id="w", route="LOS→LHR",
                   previous_kobo=45_000_000, current_kobo=37_500_000,
                   savings_pct=17.0)
    out = format_alert(a, budget_kobo=30_000_000)
    assert "over budget" in out


def test_alert_points():
    a = PriceAlert(watch_id="w", route="LOS→LHR",
                   previous_kobo=45_000_000, current_kobo=37_500_000,
                   savings_pct=17.0)
    out = format_alert(a, programs=[_prog()])
    assert "🎖️" in out


def test_alert_backward_compat():
    a = PriceAlert(watch_id="w", route="LOS→LHR",
                   previous_kobo=45_000_000, current_kobo=37_500_000,
                   savings_pct=17.0)
    out = format_alert(a)
    assert "dropped" in out
    assert "budget" not in out.lower()


# ── wiring: #72 itinerary ────────────────────────────────────────────────


def test_itinerary_summary_with_cost(tmp_path):
    from nomorals.travel.itinerary import ItineraryBuilder
    b = ItineraryBuilder(db_path=str(tmp_path / "t.db"))
    trip = b.ingest_email("BA075 LOS to LHR PNR: ABC123")
    out = b.summary(trip.id, total_kobo=45_000_000,
                    budget_kobo=50_000_000, programs=[_prog()])
    assert "💰 Total: ₦450,000" in out
    assert "within budget" in out
    assert "🎖️" in out


def test_itinerary_summary_no_cost_backward_compat(tmp_path):
    from nomorals.travel.itinerary import ItineraryBuilder
    b = ItineraryBuilder(db_path=str(tmp_path / "t.db"))
    trip = b.ingest_email("BA075 LOS to LHR PNR: ABC123")
    out = b.summary(trip.id)
    assert "💰" not in out


# ── enrich helpers ─────────────────────────────────────────────────────


def test_enrich_offers_text():
    from types import SimpleNamespace
    offers = [SimpleNamespace(total_amount="450000.00",
                              total_currency="NGN")]
    out = enrich_offers_text(["1. Air Peace 08:00"], offers,
                             budget_kobo=50_000_000,
                             programs=[_prog()])
    assert out.startswith("💰 Total: ₦450,000")
    assert "within your" in out
    assert "🎖️" in out


def test_enrich_alert_text():
    out = enrich_alert_text("✈️ dropped!", 37_500_000,
                            budget_kobo=40_000_000,
                            programs=[_prog()])
    assert "within your" in out
    assert "🎖️" in out


def test_enrich_itinerary_text():
    out = enrich_itinerary_text("🧳 trip", 45_000_000,
                                budget_kobo=40_000_000)
    assert out.startswith("💰 Total: ₦450,000")
    assert "over budget by ₦50,000" in out
