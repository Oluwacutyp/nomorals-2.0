"""Sweep tests: travel module upgrades (mined → built 2026-10-10).

watchers: price history, Hopper-style forecast, rise/cooldown triggers.
itinerary: airline resolution, TripIt merging, timeline, conflicts, ICS, packing.
display: CPP valuation, best redemption, themes.
whitelabel: templates, escalation, onboarding, analytics.
groups: split modes, payments, nudges, overview, settle-guard.
"""

from __future__ import annotations

import time

import pytest

from nomorals.travel import watchers as W
from nomorals.travel import itinerary as I
from nomorals.travel import display as D
from nomorals.travel import whitelabel as WL
from nomorals.travel import groups as G


# ── helpers ───────────────────────────────────────────────────────────────

class _Duffel:
    def __init__(self, prices):
        self.prices = prices  # (o, d, date) -> [(kobo, cur, offer)]

    def search_offers(self, o, d, date):
        out = []
        for kobo, cur, offer in self.prices.get((o, d, date), []):
            out.append(type("O", (), {
                "total_amount": str(kobo / 100),
                "total_currency": cur,
                "raw": {},
            })())
        return out


def _watcher(tmp_path, prices):
    w = W.PriceWatcher(db_path=str(tmp_path / "w.db"),
                       duffel=_Duffel(prices))
    return w


# ── watchers: forecast ────────────────────────────────────────────────────

def _pts(kobos):
    return [W.PricePoint(date="2026-01-01", amount_kobo=k) for k in kobos]


def test_forecast_buy_at_low():
    fc = W.forecast(_pts([500_000, 480_000, 460_000, 440_000, 420_000]))
    assert fc["verdict"] == W.VERDICT_BUY
    assert fc["confidence"] > 0
    assert fc["reasons"]


def test_forecast_wait_when_rising():
    fc = W.forecast(_pts([400_000, 420_000, 440_000, 470_000, 500_000]))
    assert fc["verdict"] == W.VERDICT_WAIT


def test_forecast_watch_needs_history():
    fc = W.forecast(_pts([400_000]))
    assert fc["verdict"] == W.VERDICT_WATCH
    assert fc["confidence"] == 0.0


def test_trend_stats():
    s = W.trend_stats(_pts([400_000, 500_000, 450_000]))
    assert s["low"] == 400_000 and s["high"] == 500_000
    assert s["avg"] == pytest.approx(450_000)
    assert W.trend_stats([]) == {}


def test_verdict_banner():
    b = W.verdict_banner({"verdict": W.VERDICT_BUY, "confidence": 0.8,
                          "reasons": ["at the low"]})
    assert "BUY NOW" in b and "at the low" in b


# ── watchers: history + rise/cooldown ─────────────────────────────────────

def test_history_recorded_on_check(tmp_path):
    w = _watcher(tmp_path, {("LOS", "LHR", "2026-12-01"):
                            [(450_000, "NGN", None)]})
    watch = w.watch("LOS", "LHR", "2026-12-01")
    w.check(watch)
    w.check(watch)
    assert len(w.history(watch.id)) == 2


def test_rise_alert_fires(tmp_path):
    w = _watcher(tmp_path, {("LOS", "LHR", "2026-12-01"):
                            [(450_000, "NGN", None)]})
    watch = w.watch("LOS", "LHR", "2026-12-01", notify_on="rise",
                    threshold_pct=10.0)
    assert w.check(watch) is None  # baseline
    w._duffel.prices[("LOS", "LHR", "2026-12-01")] = [(520_000, "NGN", None)]
    alert = w.check(watch)
    assert alert is not None
    assert alert.kind == "rise"
    assert "climbing" in alert.text


def test_rise_cooldown_blocks_spam(tmp_path):
    w = _watcher(tmp_path, {("LOS", "LHR", "2026-12-01"):
                            [(450_000, "NGN", None)]})
    watch = w.watch("LOS", "LHR", "2026-12-01", notify_on="rise",
                    threshold_pct=5.0, cooldown_hours=24.0)
    w.check(watch)
    w._duffel.prices[("LOS", "LHR", "2026-12-01")] = [(500_000, "NGN", None)]
    assert w.check(watch) is not None
    w._duffel.prices[("LOS", "LHR", "2026-12-01")] = [(560_000, "NGN", None)]
    # new high but within cooldown → silent
    assert w.check(watch) is None


def test_parse_watch_request_modes():
    d = W.parse_watch_request("track LOS LHR 2026-12-01 rise 15%")
    assert d["notify_on"] == "rise"
    d = W.parse_watch_request("track LOS LHR 2026-12-01 under 400k")
    assert d["notify_on"] == "drop"
    d = W.parse_watch_request("track LOS LHR 2026-12-01 any move")
    assert d["notify_on"] == "any"


def test_format_watches(tmp_path):
    w = _watcher(tmp_path, {})
    assert "No active" in w.format_watches()
    w.watch("LOS", "LHR", "2026-12-01", target_kobo=40_000_000)
    out = w.format_watches()
    assert "LOS→LHR" in out and "target" in out


# ── itinerary ─────────────────────────────────────────────────────────────

def test_airline_for_flight_number():
    assert I.airline_for_flight_number("BA075") == "British Airways"
    assert I.airline_for_flight_number("VK102") == "ValueJet"
    assert I.airline_for_flight_number("XX999") == ""
    assert I.airline_for_flight_number("") == ""


def test_parse_fills_airline():
    p = I.parse_confirmation("Flight BA075 LOS to LHR departs 22:45 PNR ABC123")
    assert p["flights"] and p["flights"][0]["airline"] == "British Airways"


def _builder(tmp_path):
    return I.ItineraryBuilder(db_path=str(tmp_path / "trips.db"),
                              vault_dir=str(tmp_path / "vault"))


def test_merge_trips(tmp_path):
    b = _builder(tmp_path)
    t1 = b.ingest_email("Flight BA075 LOS to LHR on 2026-12-01. Departs: 22:45. "
                        "PNR AAA111", auto_merge=False)
    t2 = b.ingest_email("Hotel: Transcorp Hilton. Check-in: 2026-12-02. "
                        "Check-out: 2026-12-05", auto_merge=False)
    assert t1.id != t2.id
    merged = b.merge_trips([t1.id, t2.id])
    assert merged is not None
    assert len(merged.flights) == 1 and len(merged.hotels) == 1
    assert b.get_trip(t2.id) is None


def test_auto_merge_on_ingest(tmp_path):
    b = _builder(tmp_path)
    t1 = b.ingest_email("Flight BA075 LOS to LHR on 2026-12-01. Departs: 22:45. "
                        "PNR AAA111")
    t2 = b.ingest_email("Hotel: Transcorp Hilton. Check-in: 2026-12-02. "
                        "Check-out: 2026-12-05. Booking reference: AAA111")
    assert t2.id == t1.id  # same PNR → folded into the same trip
    assert len(t2.hotels) == 1


def test_no_merge_different_pnr_but_suggested(tmp_path):
    b = _builder(tmp_path)
    t1 = b.ingest_email("Flight BA075 LOS to LHR on 2026-12-01. Departs: 22:45. "
                        "PNR AAA111")
    t2 = b.ingest_email("Hotel: Transcorp Hilton. Check-in: 2026-12-02. "
                        "Check-out: 2026-12-05. Confirmation: HTL-99812")
    assert t2.id != t1.id  # different PNRs → stays separate
    parsed = I.parse_confirmation(
        "Hotel: Transcorp Hilton. Check-in: 2026-12-02. "
        "Check-out: 2026-12-05. Confirmation: HTL-99812")
    suggested = b.find_merge_candidate(parsed, exclude_id=t2.id)
    assert suggested is not None and suggested.id == t1.id


def test_timeline_and_status(tmp_path):
    b = _builder(tmp_path)
    t = b.ingest_email("Flight BA075 LOS to LHR on 2030-01-05. Departs: 22:45. "
                       "arrives 2030-01-06 04:10 PNR AAA111")
    assert I.trip_status(t) == "upcoming"
    evs = I.trip_timeline(t)
    assert evs and evs[0]["when"] <= evs[-1]["when"]
    assert any("in " in e["countdown"] or "tomorrow" in e["countdown"]
               for e in evs if e["countdown"])
    out = b.timeline(t.id)
    assert "travel timeline" in out


def test_detect_conflicts():
    trip = I.Trip(id="x", flights=[
        I.Flight(flight_number="A1", departs="2026-12-01T10:00",
                 arrives="2026-12-01T12:00"),
        I.Flight(flight_number="A2", departs="2026-12-01T11:00",
                 arrives="2026-12-01T13:00"),
    ])
    warns = I.detect_conflicts(trip)
    assert any("overlap" in w for w in warns)


def test_export_ics(tmp_path):
    b = _builder(tmp_path)
    t = b.ingest_email("Flight BA075 LOS to LHR on 2026-12-01. Departs: 22:45. "
                       "Arrives: 04:10. PNR AAA111")
    path = b.export_ics(t.id)
    assert path and path.endswith(".ics")
    text = open(path).read()
    assert "BEGIN:VCALENDAR" in text and "BA075" in text


def test_packing_list():
    trip = I.Trip(id="x", flights=[
        I.Flight(flight_number="BA075", origin="LOS", destination="LHR")],
        hotels=[I.HotelStay(name="H", check_in="2026-12-01",
                             check_out="2026-12-04")])
    items = I.packing_list(trip)
    names = [i for i, _ in items]
    assert any("passport" in n for n in names)
    assert any("3 night" in n for n in names)


def test_summary_has_status_and_conflicts(tmp_path):
    b = _builder(tmp_path)
    t = b.ingest_email("Flight BA075 LOS to LHR on 2030-01-05. Departs: 22:45. "
                       "PNR AAA111")
    out = b.summary(t.id)
    assert "🗓️" in out  # upcoming status emoji
    assert "British Airways" in out


# ── display ───────────────────────────────────────────────────────────────

def test_cpp_math():
    # ₦450k cash, 65k points + ₦80k fees → (450000-80000)/65000 = 5.69¢
    assert D.cpp(45_000_000, 65_000, 8_000_000) == pytest.approx(569.23, abs=0.01)
    assert D.cpp(45_000_000, 0) == 0.0


def test_rate_redemption():
    assert D.rate_redemption(600) == "great"
    assert D.rate_redemption(350) == "good"
    assert D.rate_redemption(200) == "fair"
    assert D.rate_redemption(50) == "poor"


def test_best_redemption():
    ps = [D.LoyaltyProgram(name="A", balance=100_000,
                           award_points=65_000, award_cash_kobo=8_000_000),
          D.LoyaltyProgram(name="B", balance=100_000,
                           award_points=30_000, award_cash_kobo=8_000_000)]
    best = D.best_redemption(45_000_000, ps)
    assert best["program"] == "B"  # higher CPP
    assert "warning" in best
    assert D.best_redemption(45_000_000, []) == {}


def test_bank_partners():
    assert "Amex Membership Rewards" in D.bank_partners("Flying Blue")


def test_themes():
    rich, compact, minimal = (D.themed("rich"), D.themed("compact"),
                              D.themed("minimal"))
    assert rich.header("x").startswith("✈️")
    assert compact.header("x").startswith("»")
    assert D.themed("bogus").name == "rich"
    assert minimal.price(45_000_000) == "450000"


def test_points_vs_cash_shows_cpp():
    out = D.points_vs_cash(45_000_000, [D.LoyaltyProgram(
        name="Miles", balance=100_000, award_points=65_000,
        award_cash_kobo=8_000_000)])
    assert "kobo/pt" in out  # CPP now attached


# ── whitelabel ────────────────────────────────────────────────────────────

def test_template_registry():
    assert set(WL.TEMPLATES) >= {"airline", "hotel", "tours", "car"}
    tpl = WL.template_for("hotel", "Eko Hotel")
    assert tpl["name"] == "Eko Hotel Concierge"
    assert len(tpl["flows"]) >= 3
    assert WL.template_for("bogus")["name"].endswith("(VIKI template)")


def test_route_to_human():
    yes, reason = WL.route_to_human("I want a refund, this is a scam")
    assert yes and reason
    no, _ = WL.route_to_human("what time does check-in open?")
    assert not no


def test_client_onboarding_and_analytics(tmp_path):
    db = str(tmp_path / "c.db")
    store = WL.TravelClientStore(db)
    c = store.create("Test Air", channels=["whatsapp", "instagram"],
                     languages=["en", "pcm"])
    assert c.channels == ["whatsapp", "instagram"]
    WL.log_client_event(c.id, WL.EVENT_QUERY, {"q": "flights"}, db_path=db)
    WL.log_client_event(c.id, WL.EVENT_BOOKING, db_path=db)
    WL.log_client_event(c.id, WL.EVENT_KNOWLEDGE_HIT, db_path=db)
    out = WL.client_analytics(c.id, db_path=db)
    assert "queries: 1" in out and "bookings: 1" in out
    assert "grounding: 100%" in out
    onb = WL.client_onboarding(c.id, db_path=db)
    assert "onboarding" in onb and "⬜" in onb  # knowledge empty


def test_list_knowledge_empty(tmp_path):
    assert WL.list_knowledge("nope_nonexistent") == []


# ── groups ────────────────────────────────────────────────────────────────

def test_compute_shares_exact_sum():
    assert sum(G.compute_shares(10_000, ["a", "b", "c"])) == 10_000
    assert G.compute_shares(10_001, ["a", "b", "c"]) == [3334, 3334, 3333]
    with pytest.raises(ValueError):
        G.compute_shares(10_000, ["a", "b"], split="percent",
                         shares=[50, 40])
    with pytest.raises(ValueError):
        G.compute_shares(10_000, ["a", "b"], split="exact",
                         shares=[6000, 3000])


def _gstore(tmp_path):
    return G.GroupTripStore("testgroup", db_path=str(tmp_path / "g.db"))


def test_percent_split_expense(tmp_path):
    s = _gstore(tmp_path)
    t = s.create_trip("Lagos", ["Ada", "Mama"], created_by="Ada")
    exp = s.add_expense(t.id, "Ada", 10_000, ["Ada", "Mama"], "dinner",
                        split="percent", shares=[70, 30])
    assert exp is not None
    assert exp.shares_kobo == [7000, 3000]
    bal = s.balances(t.id)
    assert bal["Ada"] == 3000 and bal["Mama"] == -3000


def test_record_payment_reduces_debt(tmp_path):
    s = _gstore(tmp_path)
    t = s.create_trip("Lagos", ["Ada", "Mama"], created_by="Ada")
    s.add_expense(t.id, "Ada", 10_000, ["Ada", "Mama"], "dinner")
    assert s.balances(t.id)["Mama"] == -5000
    pay = s.record_payment(t.id, "Mama", "Ada", 2000)
    assert pay is not None
    assert s.balances(t.id)["Mama"] == -3000
    assert s.balances(t.id)["Ada"] == 3000
    # full settle → zero
    s.record_payment(t.id, "Mama", "Ada", 3000)
    assert s.settlement(t.id) == []


def test_leave_settle_guard(tmp_path):
    s = _gstore(tmp_path)
    t = s.create_trip("Lagos", ["Ada", "Mama"], created_by="Ada")
    s.add_expense(t.id, "Ada", 10_000, ["Ada", "Mama"], "dinner")
    ok, msg = s.leave_trip(t.id, "Mama")
    assert not ok and "settle up" in msg
    s.record_payment(t.id, "Mama", "Ada", 5000)
    ok, _ = s.leave_trip(t.id, "Mama")
    assert ok


def test_poll_nudges(tmp_path):
    s = _gstore(tmp_path)
    t = s.create_trip("Lagos", ["Ada", "Mama", "Zoe"], created_by="Ada")
    poll = s.create_poll(t.id, "Where?", ["Beach", "City"],
                         deadline=time.time() + 3600, created_by="Ada")
    assert poll in s.polls_closing_soon(t.id)
    s.vote(poll.id, "Ada", "Beach")
    assert s.poll_nonvoters(poll.id) == ["Mama", "Zoe"]


def test_categories_and_overview(tmp_path):
    s = _gstore(tmp_path)
    t = s.create_trip("Lagos", ["Ada", "Mama"], created_by="Ada",
                      destination="Lekki")
    s.set_trip_details(t.id, start_date="2026-12-01", end_date="2026-12-05")
    s.add_expense(t.id, "Ada", 5000, ["Ada", "Mama"], "suya",
                  category="food")
    s.add_expense(t.id, "Mama", 20000, ["Ada", "Mama"], "hotel",
                  category="stay")
    cats = s.category_totals(t.id)
    assert cats == {"stay": 20000, "food": 5000}
    s.set_budget(t.id, "Ada", 50000)
    out = s.trip_overview(t.id)
    assert "Lekki" in out and "█" in out and "#food" in out


def test_member_totals(tmp_path):
    s = _gstore(tmp_path)
    t = s.create_trip("Lagos", ["Ada", "Mama"], created_by="Ada")
    s.add_expense(t.id, "Ada", 10_000, ["Ada", "Mama"], "dinner")
    totals = s.member_totals(t.id)
    assert totals["Ada"]["paid"] == 10_000
    assert totals["Mama"]["spent"] == 5000
    assert totals["Mama"]["balance"] == -5000


def test_control_gtrip_pay_and_overview(tmp_path):
    s = _gstore(tmp_path)
    chat = type("C", (), {"kind": "group", "platform": "tg",
                          "id": "1"})()
    ctx = type("X", (), {"settings": type(
        "S", (), {"community_dir": str(tmp_path)})()})()
    out = G.control_gtrip("new Lagos trip", context=ctx, chat=chat,
                          sender="Ada")
    assert "Group trip created" in out
    out = G.control_gtrip("expense 10k dinner for Ada,Mama #food",
                          context=ctx, chat=chat, sender="Ada")
    assert "Logged" in out and "#food" in out
    out = G.control_gtrip("pay Ada 5000", context=ctx, chat=chat,
                          sender="Mama")
    assert "paid Ada" in out
    out = G.control_gtrip("overview", context=ctx, chat=chat, sender="Ada")
    assert "per person" in out


def test_control_gtrip_percent_split(tmp_path):
    chat = type("C", (), {"kind": "group", "platform": "tg",
                          "id": "2"})()
    ctx = type("X", (), {"settings": type(
        "S", (), {"community_dir": str(tmp_path)})()})()
    G.control_gtrip("new Trip", context=ctx, chat=chat, sender="Ada")
    out = G.control_gtrip("expense 10k dinner for Ada,Mama split 70/30",
                          context=ctx, chat=chat, sender="Ada")
    assert "₦7,000" in out and "₦3,000" in out
