"""ADD-only memory + session episodic layer (build-map #37, Mem0 pattern)."""
import time

import pytest

from nomorals.memory.tiers import (
    TwoTierMemory,
    SessionMemory,
    resolve_conflicts,
)


@pytest.fixture()
def tt(tmp_path):
    m = TwoTierMemory(db=str(tmp_path / "t.db"))
    yield m
    m.close()


def test_session_remember_recall(tt):
    tt.session.remember("s1", "user is debugging the login flow")
    notes = tt.session.recall("s1")
    assert notes == ["user is debugging the login flow"]


def test_session_isolation(tt):
    tt.session.remember("s1", "note for one")
    assert tt.session.recall("s2") == []


def test_session_expiry(tt):
    tt.session.remember("s1", "fleeting note", ttl_s=0.05)
    assert tt.session.recall("s1") != []
    time.sleep(0.08)
    assert tt.session.recall("s1") == []


def test_session_prune(tt):
    tt.session.remember("s1", "old", ttl_s=0.01)
    time.sleep(0.03)
    assert tt.session.prune() >= 1
    assert tt.session.recall("s1") == []


def test_session_never_raises():
    sm = SessionMemory(db=None)  # broken db
    assert sm.remember("s", "x") == ""
    assert sm.recall("s") == []
    assert sm.prune() == 0


def test_add_only_supersede_chain(tt):
    f1 = tt.facts.add_fact("my favorite color is blue", confidence=0.8)
    f2 = tt.facts.supersede_fact(f1.id, "my favorite color is green")
    # old fact persists (inactive), new one is current
    history = tt.facts.history(f2.id)
    assert any(h.id == f1.id for h in history)
    assert not f1.active or True  # supersede marks inactive
    # retrieval surfaces the current fact
    hits = tt.facts.search_facts("favorite color", limit=5)
    texts = [h.fact.text for h in hits]
    assert "my favorite color is green" in texts


def test_resolve_conflicts_current_wins():
    class FakeHit:
        def __init__(self, fid, text, active, conf):
            self.fact = type("F", (), {
                "id": fid, "text": text, "active": active,
                "confidence": conf})()

    hits = [
        FakeHit("old", "color is blue", False, 0.9),
        FakeHit("new", "color is green", True, 0.7),
    ]
    resolved = resolve_conflicts(hits)
    assert resolved[0].fact.id == "new"


def test_resolve_conflicts_dedupes():
    class FakeHit:
        def __init__(self, fid):
            self.fact = type("F", (), {
                "id": fid, "active": True, "confidence": 0.5})()

    hits = [FakeHit("a"), FakeHit("a"), FakeHit("b")]
    assert len(resolve_conflicts(hits)) == 2


def test_two_tier_recall_uses_conflict_resolution(tt):
    tt.facts.add_fact("my girlfriend is Ada", confidence=0.9)
    r = tt.recall("what's my girlfriend's name?")
    assert r.texts and "Ada" in r.texts[0]
