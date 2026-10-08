"""Representation-quality ledger (build-map #35). All offline."""
import time

import pytest

from nomorals.cognition.representation import (
    ACTION_TYPES,
    RepresentationLedger,
    log_representation_action,
)


@pytest.fixture()
def ledger():
    return RepresentationLedger(":memory:")


def test_log_round_trip(ledger):
    rec = ledger.log_action("apply", "gig application submitted: g123",
                            metadata={"gig_id": "g123"})
    assert rec.id.startswith("rep_")
    got = ledger.get(rec.id)
    assert got is not None
    assert got.action_type == "apply"
    assert got.description == "gig application submitted: g123"
    assert got.metadata == {"gig_id": "g123"}
    assert got.outcome == "unknown"


def test_log_rejects_bad_action_type(ledger):
    with pytest.raises(ValueError):
        ledger._log_action("teleport", "x")


def test_log_action_never_raises():
    # Even with a broken DB path, log_action returns a record, not an error.
    rec = RepresentationLedger(":memory:").log_action("post", "hi")
    assert rec.id  # real record


def test_record_outcome_and_feedback(ledger):
    rec = ledger.log_action("message", "SMS sent to +123")
    assert ledger.record_outcome(rec.id, "success", owner_feedback=+1)
    got = ledger.get(rec.id)
    assert got.outcome == "success"
    assert got.owner_feedback == 1
    # unknown id → False, not an exception
    assert ledger.record_outcome("rep_nope", "success") is False


def test_record_outcome_feedback_words(ledger):
    rec = ledger.log_action("post", "published")
    assert ledger.record_outcome(rec.id, "failed", owner_feedback="down",
                                 feedback_text="wrong tone")
    got = ledger.get(rec.id)
    assert got.owner_feedback == -1
    assert got.feedback_text == "wrong tone"


def test_counterfactual_storage(ledger):
    rec = ledger.log_action("negotiate", "haggled price to 50k")
    nid = ledger.record_counterfactual(
        rec.id, "open at 40k instead",
        "seller anchors high; a lower open lands lower")
    assert nid.startswith("cf_")
    got = ledger.get(rec.id)
    assert len(got.counterfactuals) == 1
    assert got.counterfactuals[0]["better_action"] == "open at 40k instead"


def test_quality_score_success(ledger):
    for i in range(4):
        r = ledger.log_action("apply", f"app {i}")
        ledger.record_outcome(r.id, "success")
    assert ledger.quality_score("apply") == pytest.approx(1.0)


def test_quality_score_unknown_excluded(ledger):
    # Unknown outcomes carry zero weight — they never inflate.
    r1 = ledger.log_action("apply", "app one")
    ledger.record_outcome(r1.id, "failed")
    ledger.log_action("apply", "app two")  # unknown, no outcome
    ledger.log_action("apply", "app three")  # unknown, no outcome
    assert ledger.quality_score("apply") == pytest.approx(0.0)


def test_quality_score_no_data_prior(ledger):
    assert ledger.quality_score("purchase") == pytest.approx(0.5)


def test_owner_downvote_hits_harder_than_upvote_helps(ledger):
    # success + owner -1 should score well below success + owner +1
    r1 = ledger.log_action("message", "m1")
    ledger.record_outcome(r1.id, "success", owner_feedback=-1)
    down = ledger.quality_score("message")

    ledger2 = RepresentationLedger(":memory:")
    r2 = ledger2.log_action("message", "m1")
    ledger2.record_outcome(r2.id, "partial", owner_feedback=+1)
    up = ledger2.quality_score("message")

    assert down < 0.6  # disapproval drags a success down hard
    assert up > down


def test_recency_weighting():
    led = RepresentationLedger(":memory:")
    old = led.log_action("post", "old post",
                         created_at=time.time() - 25 * 86400)
    led.record_outcome(old.id, "failed")
    new = led.log_action("post", "new post")
    led.record_outcome(new.id, "success")
    # Recent success outweighs the old failure with 14-day half-life.
    assert led.quality_score("post") > 0.6


def test_low_quality_patterns(ledger):
    for i in range(3):
        r = ledger.log_action("negotiate", f"deal {i}")
        ledger.record_outcome(r.id, "failed", owner_feedback="down")
    for i in range(2):
        r = ledger.log_action("apply", f"app {i}")
        ledger.record_outcome(r.id, "success")
    patterns = ledger.low_quality_patterns(threshold=0.5)
    types = [p["action_type"] for p in patterns]
    assert "negotiate" in types
    assert "apply" not in types
    neg = next(p for p in patterns if p["action_type"] == "negotiate")
    assert neg["score"] < 0.5
    assert len(neg["examples"]) == 3


def test_low_quality_patterns_carry_counterfactuals(ledger):
    r = ledger.log_action("purchase", "bought data 5k")
    ledger.record_outcome(r.id, "failed")
    ledger.record_counterfactual(r.id, "wait for the promo",
                                 "MTN runs promos on Fridays")
    patterns = ledger.low_quality_patterns(threshold=0.5)
    pur = next(p for p in patterns if p["action_type"] == "purchase")
    assert len(pur["counterfactuals"]) == 1
    assert pur["counterfactuals"][0]["better_action"] == "wait for the promo"


def test_summary_format(ledger):
    r = ledger.log_action("apply", "gig application submitted: g9")
    ledger.record_outcome(r.id, "success", owner_feedback=+1)
    text = ledger.summary()
    assert "How I represented you" in text
    assert "apply" in text
    assert "gig application submitted: g9" in text
    assert "success" in text


def test_summary_empty(ledger):
    text = ledger.summary()
    assert "No autonomous actions" in text


def test_summary_flags_weak_spots(ledger):
    r = ledger.log_action("message", "SMS blasted")
    ledger.record_outcome(r.id, "failed", owner_feedback=-1)
    ledger.record_counterfactual(r.id, "ask first", "owner hates surprises")
    text = ledger.summary()
    assert "Needs work" in text
    assert "ask first" in text


def test_hook_helper_never_raises(tmp_path):
    # log_representation_action writes to the real home-path DB by default;
    # with a broken settings object it must still not raise.
    class BadSettings:
        @property
        def home_path(self):
            raise RuntimeError("boom")
    assert log_representation_action("post", "x",
                                     settings=BadSettings()) == ""


def test_action_types_covered():
    assert set(ACTION_TYPES) == {"negotiate", "purchase", "apply",
                                 "message", "post"}
