"""Tests for nomorals.cognition.trajectories — fully offline, tmp SQLite."""

import random
import time

import pytest

from nomorals.cognition import TrajectoryStore
from nomorals.cognition.trajectories import (
    cluster_key_for,
    normalize_error,
)


def make_store(tmp_path):
    return TrajectoryStore(db=str(tmp_path / "cog.db"))


def test_empty_store_returns_prior():
    store = TrajectoryStore()
    assert store.success_rate("chat", "general", model_id="m1") == 0.5
    assert store.success_rate("x", "y") == 0.5
    assert store.rank("chat", "general", ["b", "a"], kind="model") == ["a", "b"]
    assert store.failure_clusters() == []


def test_all_success_rate_near_one(tmp_path):
    store = make_store(tmp_path)
    for _ in range(10):
        store.record(
            task_kind="chat", capability="general",
            model_id="m1", success=True,
        )
    assert store.success_rate("chat", "general", model_id="m1") > 0.999


def test_all_failure_rate_near_zero(tmp_path):
    store = make_store(tmp_path)
    for _ in range(10):
        store.record(
            task_kind="chat", capability="general",
            model_id="m1", success=False, error="boom",
        )
    assert store.success_rate("chat", "general", model_id="m1") < 0.001


def test_mixed_rate(tmp_path):
    store = make_store(tmp_path)
    for _ in range(6):
        store.record(task_kind="chat", capability="general",
                     model_id="m1", success=True)
    for _ in range(4):
        store.record(task_kind="chat", capability="general",
                     model_id="m1", success=False)
    assert store.success_rate("chat", "general", model_id="m1") == pytest.approx(
        0.6, abs=0.05
    )


def test_slice_isolation_by_model_and_kind(tmp_path):
    store = make_store(tmp_path)
    store.record(task_kind="chat", capability="general", model_id="m1",
                 success=True)
    store.record(task_kind="chat", capability="general", model_id="m2",
                 success=False)
    assert store.success_rate("chat", "general", model_id="m1") > 0.9
    assert store.success_rate("chat", "general", model_id="m2") < 0.1
    # wildcard model_id aggregates both → ~0.5
    assert store.success_rate("chat", "general") == pytest.approx(0.5, abs=0.05)
    # different task_kind is a different slice
    assert store.success_rate("code", "general", model_id="m1") == 0.5


def test_recency_weighting_old_failures_decay(tmp_path):
    """Old failures should weigh far less than recent successes."""
    store = make_store(tmp_path)
    day = 86_400.0
    now = time.time()
    old = now - 60 * day  # 60 days ago; half-life 14d → weight ≈ 2^-4.3
    for _ in range(10):
        store._add(
            task_kind="chat", capability="general", model_id="m1",
            success=False, error="stale bug", created_at=old,
        )
    for _ in range(5):
        store.record(task_kind="chat", capability="general", model_id="m1",
                     success=True)
    rate = store.success_rate("chat", "general", model_id="m1",
                              half_life_days=14.0)
    # Without decay it would be 5/15 ≈ 0.33; decayed old failures → ≈ 0.9+.
    assert rate > 0.85


def test_recent_outcome_matters_more(tmp_path):
    """One recent success outweighs many old failures with short half-life."""
    store = make_store(tmp_path)
    day = 86_400.0
    now = time.time()
    for _ in range(20):
        store._add(
            task_kind="chat", capability="general", model_id="m1",
            success=False, created_at=now - 30 * day,
        )
    store._add(
        task_kind="chat", capability="general", model_id="m1",
        success=True, created_at=now,
    )
    assert store.success_rate("chat", "general", model_id="m1",
                              half_life_days=1.0) > 0.9


def test_rank_orders_by_simulated_history(tmp_path):
    """Model A 90% success, model B 40% over 50 trajectories → A first."""
    store = make_store(tmp_path)
    rng = random.Random(42)
    for _ in range(50):
        store.record(
            task_kind="search", capability="web",
            model_id="model-a", success=rng.random() < 0.90,
        )
        store.record(
            task_kind="search", capability="web",
            model_id="model-b", success=rng.random() < 0.40,
        )
    ranked = store.rank("search", "web", ["model-b", "model-a"], kind="model")
    assert ranked[0] == "model-a"
    assert ranked[1] == "model-b"


def test_rank_kind_skill(tmp_path):
    store = make_store(tmp_path)
    for _ in range(5):
        store.record(task_kind="chat", capability="general", skill_id="s-good",
                     success=True)
    for _ in range(5):
        store.record(task_kind="chat", capability="general", skill_id="s-bad",
                     success=False)
    assert store.rank("chat", "general", ["s-bad", "s-good"],
                      kind="skill") == ["s-good", "s-bad"]


def test_rank_deterministic_on_ties(tmp_path):
    store = make_store(tmp_path)
    cands = ["zeta", "alpha", "mid"]
    first = store.rank("chat", "general", cands, kind="model")
    assert first == sorted(cands)
    assert store.rank("chat", "general", cands, kind="model") == first
    # measured candidates sort above unknowns; unknown ties by name
    for _ in range(4):
        store.record(task_kind="chat", capability="general", model_id="alpha",
                     success=True)
    second = store.rank("chat", "general", cands, kind="model")
    assert second == ["alpha", "mid", "zeta"]


def test_failure_clusters_group_identical(tmp_path):
    store = make_store(tmp_path)
    for _ in range(4):
        store.record(task_kind="chat", capability="general", model_id="m1",
                     success=False, error="TimeoutError: upstream timeout")
    store.record(task_kind="chat", capability="general", model_id="m1",
                 success=True)
    clusters = store.failure_clusters("chat")
    assert len(clusters) == 1
    c = clusters[0]
    assert c["task_kind"] == "chat"
    assert c["count"] == 4
    assert c["error_signature"] == "TimeoutError: upstream timeout"
    assert c["last_seen"] > 0
    assert len(c["example_ids"]) == 4


def test_failure_clusters_normalize_volatile_text(tmp_path):
    store = make_store(tmp_path)
    store.record(task_kind="code", capability="python", model_id="m1",
                 success=False,
                 error="ValueError: bad widget 0x7f8a1234 at 2026-09-01T12:34:56")
    store.record(task_kind="code", capability="python", model_id="m1",
                 success=False,
                 error="ValueError: bad widget 0x9b00cdef at 2026-10-02T08:11:00Z")
    store.record(task_kind="code", capability="python", model_id="m1",
                 success=False, error="RuntimeError: oom killed")
    clusters = store.failure_clusters("code")
    assert len(clusters) == 2
    by_sig = {c["error_signature"]: c["count"] for c in clusters}
    assert by_sig.get("ValueError: bad widget <addr> at <ts>") == 2
    assert by_sig.get("RuntimeError: oom killed") == 1


def test_failure_clusters_limit_and_all_kinds(tmp_path):
    store = make_store(tmp_path)
    for i in range(3):
        store.record(task_kind="chat", capability="g", model_id="m",
                     success=False, error=f"Err{i}")
    clusters = store.failure_clusters(limit=2)
    assert len(clusters) == 2
    assert len(store.failure_clusters(None)) == 3


def test_normalize_error_multiline_takes_first_stable_line(tmp_path):
    assert normalize_error(
        "\n\nTraceback (most recent call last):\n"
        "ValueError: bad input 0xdeadbeef"
    ) == "Traceback (most recent call last):"
    assert normalize_error("") == "(no error message)"
    assert normalize_error("   \n  ") == "(no error message)"


def test_cluster_key_for_is_stable(tmp_path):
    assert cluster_key_for("chat", "boom") == "chat::boom"
    assert cluster_key_for("chat", "boom") == cluster_key_for("chat", "boom")
