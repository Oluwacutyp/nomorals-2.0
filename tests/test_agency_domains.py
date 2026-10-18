"""Failing-first: domain loop completion + cross-domain lineage (audit §6-7)."""
import pytest

from nomorals.agency.domains import (
    LEAD_TRANSITIONS, advance_lead, score_gig, evaluate_deal,
    classify_weather_item, cluster_news, GIG_POST_SUBMISSION,
)
from nomorals.agency.initiatives import Initiative, InitiativeStore
from nomorals.storage.db import Database


def test_lead_lifecycle_forbids_skipping_to_paid():
    with pytest.raises(ValueError):
        advance_lead("discovered", "paid")
    assert advance_lead("discovered", "qualified") == "qualified"
    assert advance_lead("proposal", "won") == "won"
    assert advance_lead("won", "delivery") == "delivery"
    assert advance_lead("delivery", "paid") == "paid"


def test_gig_scoring_preserves_uncertainty_not_advertised_payout():
    scored = score_gig(advertised_payout=500.0, win_probability=0.2,
                       costs=50.0, hours=10.0, time_to_payment_days=30)
    assert scored["expected_net"] == pytest.approx(500 * 0.2 - 50)
    assert scored["expected_net"] < scored["advertised_payout"]
    assert scored["uncertainty"] == "advertised payout is not expected income"


def test_gig_post_submission_states_exist():
    assert GIG_POST_SUBMISSION[:3] == ["submitted", "scoped", "delivering"]


def test_deal_discount_on_unnecessary_item_is_not_savings():
    verdict = evaluate_deal(needed=False, discount_pct=60.0,
                            price=100.0, landed_price=140.0,
                            historical_low=120.0)
    assert verdict["is_saving"] is False
    assert verdict["action"] == "skip"


def test_deal_real_saving_requires_landed_below_history():
    verdict = evaluate_deal(needed=True, discount_pct=10.0, price=100.0,
                            landed_price=90.0, historical_low=120.0)
    assert verdict["is_saving"] is True


def test_weather_heuristic_is_never_an_official_warning():
    item = classify_weather_item(kind="forecast_heuristic", region="NG",
                                 text="heavy rain likely")
    assert item["category"] == "forecast"
    assert item["label"].startswith("forecast") and "NOT an official" in item["label"]
    nws = classify_weather_item(kind="nws_alert", region="US",
                                text="flash flood warning")
    assert nws["category"] == "official_warning"
    # NWS is US-only: an NWS-labelled item for Nigeria is dishonest.
    with pytest.raises(ValueError):
        classify_weather_item(kind="nws_alert", region="NG", text="x")


def test_news_clustering_merges_duplicates():
    items = [{"title": "Gold Hits Record High!", "url": "https://a.example/1"},
             {"title": "gold hits record high", "url": "https://b.example/2"},
             {"title": "Local team wins derby", "url": "https://c.example/3"}]
    clusters = cluster_news(items)
    assert len(clusters) == 2
    assert len(clusters[0]["items"]) == 2


def test_cross_domain_lineage_links_research_to_outcome():
    store = InitiativeStore(Database(":memory:"))
    parent = store.create(Initiative(
        outcome="Research MT5 tick API", kind="exploration",
        objective="trading engine", evidence="gap",
        next_action="research", next_review_at=9999999999.0))
    child = store.create(Initiative(
        outcome="Build tick trigger demo", kind="evidence_opportunity",
        objective="trading engine", evidence="research finding",
        next_action="build demo", next_review_at=9999999999.0,
        parent_id=parent.id))
    assert store.lineage(child.id)[0].id == parent.id
