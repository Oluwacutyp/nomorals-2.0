"""Tests for nomorals.preferences — cross-domain taste, descriptors as data."""

import pytest

from nomorals.preferences import (
    DOMAIN_DESCRIPTORS,
    PreferenceStore,
    get_adapter,
    learn_from_representation,
    record_signal,
    register_adapter,
    reset_adapters,
    suggest,
)
from nomorals.preferences.adapters import DescriptorAdapter


@pytest.fixture(autouse=True)
def _reset_adapters():
    reset_adapters()
    yield
    reset_adapters()


@pytest.fixture()
def store():
    return PreferenceStore()


def test_descriptors_are_axes_not_taste():
    # Every required domain exists with axes; no weights / taste baked in.
    for domain in ("media", "music", "food", "travel", "games", "books", "news"):
        assert domain in DOMAIN_DESCRIPTORS
        assert DOMAIN_DESCRIPTORS[domain]["axes"]
    assert set(DOMAIN_DESCRIPTORS["music"]["axes"]) >= {
        "energy", "tempo", "language", "era"}
    assert set(DOMAIN_DESCRIPTORS["food"]["axes"]) >= {
        "spice", "cuisine", "diet"}
    fresh = PreferenceStore()
    assert fresh.get_weights("music") == {}
    assert fresh.get_weight("music", "energy", "high") == 0.0


def test_adapters_are_descriptor_passthroughs():
    adapter = get_adapter("music")
    assert isinstance(adapter, DescriptorAdapter)
    assert adapter.describe()["axes"] == DOMAIN_DESCRIPTORS["music"]["axes"]
    extracted = adapter.extract({"energy": "high", "tempo": "fast",
                                 "unrelated": "ignored"})
    assert extracted == {"energy": ["high"], "tempo": ["fast"]}
    # Nested-under-domain and attributes shapes also work.
    assert adapter.extract({"music": {"energy": "low"}}) == {"energy": ["low"]}
    assert adapter.extract({"attributes": {"era": "90s"}}) == {"era": ["90s"]}
    # Unknown domain -> no adapter, honest None.
    assert get_adapter("nope") is None


def test_register_adapter():
    class Custom:
        domain = "music"

        def describe(self):
            return {"axes": ["energy"], "description": "custom"}

        def extract(self, item):
            return {"energy": [str(item.get("energy", "")).lower()]} if item.get("energy") else {}

    register_adapter("music", Custom())
    assert get_adapter("music").describe()["axes"] == ["energy"]
    with pytest.raises(TypeError):
        register_adapter("music", object())


def test_approve_disapprove_shifts_ranking(store):
    candidates = [
        {"id": "low", "energy": "low", "tempo": "slow"},
        {"id": "high", "energy": "high", "tempo": "fast"},
    ]
    # Fresh store: no taste, scores all 0, input order preserved.
    ranked = suggest(store, "music", candidates)
    assert [row["item"]["id"] for row in ranked] == ["low", "high"]
    assert all(row["score"] == 0.0 for row in ranked)
    # Approve the high-energy item -> it outranks the low one.
    record_signal(store, "music", {"energy": "high", "tempo": "fast"},
                  approve=True)
    ranked = suggest(store, "music", candidates)
    assert ranked[0]["item"]["id"] == "high"
    assert ranked[0]["score"] > ranked[1]["score"]


def test_disapprove_moves_more_than_approve():
    approved = PreferenceStore()
    record_signal(approved, "music", {"energy": "high"}, approve=True)
    disapproved = PreferenceStore()
    record_signal(disapproved, "music", {"energy": "high"}, approve=False)
    assert approved.get_weight("music", "energy", "high") == pytest.approx(1.0)
    assert disapproved.get_weight("music", "energy", "high") == pytest.approx(-1.5)
    assert abs(disapproved.get_weight("music", "energy", "high")) > abs(
        approved.get_weight("music", "energy", "high"))


def test_suggest_never_invents_candidates(store):
    candidates = [{"id": "a", "energy": "high"}, {"id": "b", "energy": "low"}]
    record_signal(store, "music", {"energy": "high"}, approve=True)
    ranked = suggest(store, "music", candidates)
    assert len(ranked) == len(candidates)
    assert {row["item"]["id"] for row in ranked} == {"a", "b"}
    assert all("score" in row for row in ranked)
    # Empty input -> empty output, nothing invented.
    assert suggest(store, "music", []) == []


def test_unknown_domain_honest_empty(store):
    assert suggest(store, "nope", [{"id": "x"}]) == []
    assert store.get_weights("nope") == {}
    result = record_signal(store, "nope", {"energy": "high"}, approve=True)
    assert result["recorded"] is False
    assert store.get_weights("nope") == {}


def test_learn_from_representation(store):
    entries = [
        {"domain": "music", "owner_feedback": 1,
         "item": {"energy": "high", "tempo": "fast"}},
        {"domain": "music", "owner_feedback": -1,
         "item": {"energy": "low"}},
        {"domain": "music", "owner_feedback": None,
         "item": {"energy": "high"}},  # no feedback -> skipped
        {"domain": "nope", "owner_feedback": 1,
         "item": {"energy": "high"}},  # unknown domain -> skipped
        {"action": "food", "owner_feedback": 1,
         "metadata": {"domain": "food", "spice": "hot"}},
    ]
    result = learn_from_representation(store, entries)
    assert result["recorded"] == 3
    assert result["skipped"] == 2
    assert store.get_weight("music", "energy", "high") == pytest.approx(1.0)
    assert store.get_weight("music", "energy", "low") == pytest.approx(-1.5)
    assert store.get_weight("food", "spice", "hot") == pytest.approx(1.0)
    # RepresentationLedger-style objects (ActionRecord-like) also work.
    class _Rec:
        action_type = "books"
        owner_feedback = -1
        metadata = {"genre": "horror"}

    result = learn_from_representation(store, [_Rec()])
    assert result["recorded"] == 1
    assert store.get_weight("books", "genre", "horror") == pytest.approx(-1.5)


def test_json_backend_persists(tmp_path):
    path = str(tmp_path / "prefs.json")
    store = PreferenceStore(path)
    record_signal(store, "food", {"spice": "hot", "cuisine": "thai"},
                  approve=True)
    reopened = PreferenceStore(path)
    assert reopened.get_weight("food", "spice", "hot") == pytest.approx(1.0)
    assert reopened.get_weight("food", "cuisine", "thai") == pytest.approx(1.0)


def test_cross_domain_weights_are_independent(store):
    record_signal(store, "music", {"energy": "high"}, approve=True)
    # Same axis name in another domain is untouched: taste is per-domain.
    assert store.get_weight("books", "energy", "high") == 0.0
    assert store.get_weights("books") == {}
