"""Sweep tests for nomorals/cognition upgrades — all offline, tmp SQLite.

Covers: Wilson scoring, exception-class extraction, fingerprint rules,
recommend()/rank() objectives, trend, export/import, cluster lifecycle,
triage/regressed, lesson cards, representation distill/trend/
inconsistencies/feedback-rate.
"""

import json
import time

import pytest

from nomorals.cognition import (
    FailureKB,
    TrajectoryStore,
    cluster_key_for,
    exception_class_of,
    wilson_interval,
    wilson_lower,
)
from nomorals.cognition.failure_kb import guidance_for
from nomorals.cognition.representation import RepresentationLedger


# ── helpers ──────────────────────────────────────────────────────────────

def make_store(tmp_path, name="cog.db"):
    return TrajectoryStore(db=str(tmp_path / name))


def fail(store, task_kind, error, n, model_id="m1", fclass=""):
    for _ in range(n):
        store.record(task_kind=task_kind, capability="general",
                     model_id=model_id, success=False, error=error,
                     failure_class=fclass)


def win(store, task_kind, model_id, n, cost=0.0, latency_s=0.0):
    for _ in range(n):
        store.record(task_kind=task_kind, capability="general",
                     model_id=model_id, success=True,
                     cost=cost, latency_s=latency_s)


def lose(store, task_kind, model_id, n):
    for _ in range(n):
        store.record(task_kind=task_kind, capability="general",
                     model_id=model_id, success=False,
                     error="Err: boom")


# ── wilson ───────────────────────────────────────────────────────────────

def test_wilson_lower_prefers_confidence_over_luck():
    # 1/1 has mean 1.0; 80/100 has mean 0.8 — honest ranking flips them.
    assert wilson_lower(1, 1) < wilson_lower(80, 100)


def test_wilson_bounds_sane_and_empty_prior():
    lo, hi = wilson_interval(7, 10)
    assert 0.0 <= lo <= 0.7 <= hi <= 1.0
    assert wilson_lower(0, 0) == pytest.approx(0.5)
    # known reference value from the Evan Miller gist
    assert wilson_lower(9, 10) == pytest.approx(0.5958436, rel=1e-4)


def test_exception_class_of():
    assert exception_class_of("TimeoutError: upstream timeout") == "TimeoutError"
    assert exception_class_of("a.b.MyException: x") == "a.b.MyException"
    assert exception_class_of("plain message, no class") == ""
    assert exception_class_of("note: lowercase prefix") == ""
    assert exception_class_of("") == ""
    assert exception_class_of("line one\nValueError: second") == ""  # first line wins


# ── fingerprint rules ────────────────────────────────────────────────────

def test_fingerprint_rule_merges_clusters(tmp_path):
    store = make_store(tmp_path)
    fail(store, "chat", "RateLimitError: req 111", 2)
    fail(store, "chat", "HTTPError 429: too many", 3)
    before = store.failure_clusters("chat")
    assert len(before) == 2  # distinct signatures, distinct clusters
    store.add_fingerprint_rule("substring", "429", "rate_limited", priority=1)
    store.add_fingerprint_rule("class", "RateLimitError", "rate_limited",
                               priority=2)
    merged = store.failure_clusters("chat")
    assert len(merged) == 1
    assert merged[0]["error_signature"] == "rate_limited"


def test_fingerprint_rule_regex_and_removal(tmp_path):
    store = make_store(tmp_path)
    rid = store.add_fingerprint_rule("regex", r"db-\w+-\d+", "db-down")
    fail(store, "sync", "ConnectionError: db-prod-1 refused", 2)
    clusters = store.failure_clusters("sync")
    assert clusters[0]["error_signature"] == "db-down"
    assert clusters[0]["raw_signature"] != "db-down"
    assert store.remove_fingerprint_rule(rid)
    assert not store.remove_fingerprint_rule("fpr_nope")
    assert store.failure_clusters("sync")[0]["error_signature"] != "db-down"


def test_bad_fingerprint_rule_rejected(tmp_path):
    store = make_store(tmp_path)
    with pytest.raises(ValueError):
        store.add_fingerprint_rule("regex", r"(unclosed", "x")
    with pytest.raises(ValueError):
        store.add_fingerprint_rule("bogus", "x", "y")


def test_cluster_carries_class_firstseen_blastradius(tmp_path):
    store = make_store(tmp_path)
    fail(store, "chat", "TimeoutError: slow", 2, model_id="m-a")
    fail(store, "chat", "TimeoutError: slow", 1, model_id="m-b")
    c = store.failure_clusters("chat")[0]
    assert c["exception_class"] == "TimeoutError"
    assert c["first_seen"] <= c["last_seen"]
    assert c["models_affected"] == ["m-a", "m-b"]
    assert c["count"] == 3


# ── recommend / rank objectives ──────────────────────────────────────────

def test_recommend_mean_picks_best(tmp_path):
    store = make_store(tmp_path)
    win(store, "chat", "good", 9)
    lose(store, "chat", "good", 1)
    lose(store, "chat", "bad", 9)
    win(store, "chat", "bad", 1)
    best, details = store.recommend("chat", "general", ["bad", "good"],
                                    strategy="mean")
    assert best == "good"
    assert details["candidates"]["good"]["rate"] > 0.8


def test_recommend_wilson_not_fooled_by_luck(tmp_path):
    store = make_store(tmp_path)
    win(store, "chat", "lucky", 1)
    for _ in range(80):
        store.record(task_kind="chat", capability="general",
                     model_id="steady", success=True)
    for _ in range(20):
        store.record(task_kind="chat", capability="general",
                     model_id="steady", success=False)
    best, _ = store.recommend("chat", "general", ["lucky", "steady"],
                              strategy="wilson")
    assert best == "steady"


def test_recommend_thompson_seeded_deterministic(tmp_path):
    store = make_store(tmp_path)
    win(store, "chat", "a", 6)
    lose(store, "chat", "a", 4)
    win(store, "chat", "b", 5)
    lose(store, "chat", "b", 5)
    first, _ = store.recommend("chat", "general", ["a", "b"],
                               strategy="thompson", seed=42)
    second, _ = store.recommend("chat", "general", ["a", "b"],
                                strategy="thompson", seed=42)
    assert first == second


def test_recommend_ucb1_explores_untried_first(tmp_path):
    store = make_store(tmp_path)
    win(store, "chat", "known", 50)
    best, _ = store.recommend("chat", "general", ["known", "newbie"],
                              strategy="ucb1")
    assert best == "newbie"  # unexplored arm scores +inf


def test_recommend_cost_prefers_cheap_reliable(tmp_path):
    store = make_store(tmp_path)
    win(store, "chat", "pricey", 10, cost=1.0)
    win(store, "chat", "cheap", 9, cost=0.001)
    lose(store, "chat", "cheap", 1)
    best, details = store.recommend("chat", "general", ["pricey", "cheap"],
                                    strategy="cost")
    assert best == "cheap"
    assert details["candidates"]["cheap"]["avg_cost"] < 0.01


def test_rank_efficiency_and_speed_objectives(tmp_path):
    store = make_store(tmp_path)
    win(store, "chat", "pricey", 10, cost=2.0, latency_s=1.0)
    win(store, "chat", "cheap", 10, cost=0.01, latency_s=1.0)
    win(store, "chat", "slow", 10, cost=0.01, latency_s=120.0)
    assert store.rank("chat", "general", ["pricey", "cheap"],
                      objective="efficiency")[0] == "cheap"
    assert store.rank("chat", "general", ["slow", "cheap"],
                      objective="speed")[0] == "cheap"
    # legacy default unchanged: tie on rate → name order
    assert store.rank("chat", "general", ["pricey", "cheap"]) == [
        "cheap", "pricey"]
    with pytest.raises(ValueError):
        store.rank("chat", "general", ["a"], objective="bogus")
    with pytest.raises(ValueError):
        store.recommend("chat", "general", [], strategy="mean")


def test_slice_stats_and_trend(tmp_path):
    store = make_store(tmp_path)
    win(store, "chat", "m1", 7)
    lose(store, "chat", "m1", 3)
    st = store.slice_stats("chat", "general", model_id="m1")
    assert st["n"] == 10 and st["successes"] == 7
    assert st["wilson_lower"] <= st["rate"] <= st["wilson_upper"]
    tr = store.trend("chat", "general", model_id="m1")
    assert tr["direction"] in ("improving", "declining", "stable")
    assert tr["recent_n"] + tr["prior_n"] == 10
    # declining case: oldest succeed, recent fail
    store3 = make_store(tmp_path, "c3.db")
    now = time.time()
    for i in range(8):
        store3._add(task_kind="chat", capability="general", model_id="m",
                    success=(i < 4),  # i=0..3 oldest success
                    created_at=now - (8 - i) * 86400)
    tr3 = store3.trend("chat", "general", model_id="m", window_days=16.0)
    assert tr3["direction"] == "declining"


# ── export / import ──────────────────────────────────────────────────────

def test_trajectory_export_import_roundtrip(tmp_path):
    store = make_store(tmp_path)
    win(store, "chat", "m1", 3)
    fail(store, "chat", "Err: x", 2, model_id="m2")
    store.add_fingerprint_rule("substring", "Err", "err-group")
    path = tmp_path / "cog-export.json"
    info = store.export_json(path)
    assert info["trajectories"] == 5 and info["rules"] == 1

    store2 = make_store(tmp_path, "fresh.db")
    got = store2.import_json(path)
    assert got == {"trajectories": 5, "rules": 1}
    assert store2.success_rate("chat", "general", model_id="m1") > 0.99
    assert len(store2.failure_clusters("chat")) == 1
    assert store2.failure_clusters("chat")[0]["error_signature"] == "err-group"
    (tmp_path / "bogus.json").write_text(json.dumps({"kind": "nope"}))
    with pytest.raises(ValueError):
        store2.import_json(tmp_path / "bogus.json")


# ── FailureKB lifecycle ──────────────────────────────────────────────────

def make_kb(tmp_path):
    store = make_store(tmp_path)
    return FailureKB(store=store), store


def test_note_dedup_and_search(tmp_path):
    kb, store = make_kb(tmp_path)
    fail(store, "chat", "Boom", 3)
    key = cluster_key_for("chat", "Boom")
    first = kb.note(key, "restart the box")
    second = kb.note(key, "restart the box")
    assert first == second
    hits = kb.search_notes("restart")
    assert len(hits) == 1 and hits[0]["note_text"] == "restart the box"
    assert kb.search_notes("") == []
    assert kb.search_notes("nothing matches this") == []


def test_resolve_reopen_state(tmp_path):
    kb, store = make_kb(tmp_path)
    fail(store, "chat", "Boom", 3)
    key = cluster_key_for("chat", "Boom")
    assert kb.state(key) == "new"
    kb.resolve(key, "deployed fix in v2")
    assert kb.state(key) == "resolved"
    assert kb.reopen(key)
    assert kb.state(key) == "new"
    assert not kb.reopen(key)  # nothing to reopen


def test_triage_excludes_resolved_and_orders_by_priority(tmp_path):
    kb, store = make_kb(tmp_path)
    fail(store, "chat", "Ancient: x", 10)  # big but will be resolved
    fail(store, "chat", "Fresh: y", 3)
    ancient_key = cluster_key_for("chat", "Ancient: x")
    kb.resolve(ancient_key, "fixed long ago")
    triage = kb.triage()
    keys = [c["cluster_key"] for c in triage]
    assert ancient_key not in keys
    assert cluster_key_for("chat", "Fresh: y") in keys
    assert all(c["state"] == "new" for c in triage)
    assert all("priority" in c for c in triage)


def test_regressed_detects_comeback(tmp_path):
    kb, store = make_kb(tmp_path)
    fail(store, "chat", "Flaky: z", 3)
    key = cluster_key_for("chat", "Flaky: z")
    kb.resolve(key, "thought we fixed it")
    assert kb.regressed() == []
    fail(store, "chat", "Flaky: z", 2)  # it came back
    reg = kb.regressed()
    assert len(reg) == 1
    assert reg[0]["cluster_key"] == key
    assert reg[0]["post_resolution_count"] == 2
    assert "thought we fixed it" in reg[0]["resolution_text"]


def test_lessons_and_report(tmp_path):
    kb, store = make_kb(tmp_path)
    fail(store, "chat", "TimeoutError: slow poke", 4,
         fclass="timeout")
    kb.note_for_error("chat", "TimeoutError: slow poke",
                      "bump the timeout to 60s")
    cards = kb.lessons()
    assert len(cards) == 1
    card = cards[0]
    assert card["exception_class"] == "TimeoutError"
    assert card["failure_class"] == "timeout"
    assert "bump the timeout" in card["lesson_text"]
    assert "backoff" in card["guidance"]  # timeout heuristic hint
    assert card["notes"] == ["bump the timeout to 60s"]
    report = kb.report()
    assert "Failure triage" in report
    assert "TimeoutError: slow poke" in report
    assert "bump the timeout" in report


def test_guidance_for():
    assert "jitter" in guidance_for("rate_limited")
    assert "credentials" in guidance_for("auth_expired")  # keyword match
    assert guidance_for("mystery_class_xyz") == ""


def test_lookup_carries_state_and_priority(tmp_path):
    kb, store = make_kb(tmp_path)
    fail(store, "chat", "Boom", 3)
    clusters = kb.lookup("chat")
    assert clusters[0]["state"] == "new"
    assert clusters[0]["priority"] > 0


# ── RepresentationLedger additions ───────────────────────────────────────

@pytest.fixture()
def ledger():
    return RepresentationLedger(":memory:")


def _log(ledger, atype, desc, outcome, fb=None, fb_text="", when=None):
    rec = ledger.log_action(atype, desc,
                            created_at=when or time.time())
    ledger.record_outcome(rec.id, outcome, owner_feedback=fb,
                          feedback_text=fb_text)
    return rec


def test_distill_builds_lesson_cards(ledger):
    for i in range(3):
        r = _log(ledger, "post", f"long rambly post {i}", "failed",
                 fb=-1, fb_text="too long")
        ledger.record_counterfactual(r.id, "keep posts under 40 words",
                                     "owner prefers short posts")
    _log(ledger, "apply", "clean application", "success", fb=+1)
    cards = ledger.distill(threshold=0.7)
    assert len(cards) == 1
    card = cards[0]
    assert card["action_type"] == "post"
    assert "keep posts under 40 words" in card["lesson_text"]
    assert "too long" in card["lesson_text"]
    assert card["counterfactuals"] == [
        "keep posts under 40 words — owner prefers short posts"]
    # explicit action_type bypasses the threshold
    assert ledger.distill("apply")[0]["action_type"] == "apply"
    # no weak types → no cards
    assert RepresentationLedger(":memory:").distill() == []


def test_trend_detects_decline(ledger):
    now = time.time()
    for i in range(4):  # older: successes, outside the 14d recent half
        _log(ledger, "post", f"old {i}", "success",
             when=now - 20 * 86400 - i * 3600)
    for i in range(4):  # recent: failures
        _log(ledger, "post", f"new {i}", "failed",
             when=now - i * 3600)
    tr = ledger.trend("post", window_days=28.0)
    assert tr["direction"] == "declining"
    assert tr["recent_score"] < tr["prior_score"]


def test_feedback_rate(ledger):
    _log(ledger, "post", "a", "success", fb=+1)
    _log(ledger, "post", "b", "success", fb=-1)
    ledger.log_action("post", "c")
    ledger.log_action("post", "d")
    assert ledger.feedback_rate("post") == pytest.approx(0.5)
    assert ledger.feedback_rate("negotiate") == 0.0


def test_inconsistencies_flagged(ledger):
    r = _log(ledger, "message", "sms sent", "success", fb=-1,
             fb_text="wrong number!")
    found = ledger.inconsistencies()
    assert len(found) == 1
    assert found[0]["id"] == r.id
    assert "disapproved" in found[0]["note"]
    # consistent rows are not flagged
    _log(ledger, "message", "ok sms", "success", fb=+1)
    assert len(ledger.inconsistencies()) == 1


def test_representation_export_import(tmp_path, ledger):
    r = _log(ledger, "apply", "gig app", "partial", fb=+1)
    ledger.record_counterfactual(r.id, "attach portfolio", "clients ask")
    path = tmp_path / "rep-export.json"
    info = ledger.export_json(path)
    assert info["actions"] == 1 and info["counterfactuals"] == 1
    ledger2 = RepresentationLedger(":memory:")
    got = ledger2.import_json(path)
    assert got == {"actions": 1, "counterfactuals": 1}
    assert ledger2.quality_score("apply") == pytest.approx(
        ledger.quality_score("apply"))
    assert len(ledger2.get(r.id).counterfactuals) == 1
    (tmp_path / "bogus.json").write_text(json.dumps({"kind": "nope"}))
    with pytest.raises(ValueError):
        ledger2.import_json(tmp_path / "bogus.json")


def test_summary_shows_trend_and_feedback(ledger):
    now = time.time()
    for i in range(3):  # older successes: inside 14d window, outside 7d half
        _log(ledger, "post", f"old {i}", "success",
             when=now - 10 * 86400 - i * 3600)
    for i in range(3):  # recent failures
        _log(ledger, "post", f"p{i}", "failed",
             when=now - i * 3600)
    text = ledger.summary()
    assert "Quality by type" in text
    assert "↘" in text  # declining trend arrow
    assert "Owner feedback" in text
    assert "Needs work" in text
