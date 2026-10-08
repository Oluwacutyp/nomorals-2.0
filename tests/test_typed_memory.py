"""Typed memory objects + hybrid retrieval (build-map #39). All offline."""
import json
import warnings

import pytest

from nomorals.memory.types import (
    ENTITY_TYPES,
    TypedEntity,
    entity_home,
    load_entity,
    new_entity,
    parse_person_page,
    save_entity,
)
from nomorals.memory.hybrid import (
    Hit,
    HybridMemoryIndex,
    detect_entity_type,
    hybrid_search,
    rrf_fuse,
    type_aware_search,
)
from nomorals.memory.tiers import TwoTierMemory


# ── schemas ────────────────────────────────────────────────────────────────

def test_all_six_types_present():
    assert set(ENTITY_TYPES) == {
        "person", "project", "place", "commitment", "preference", "habit"}


def test_valid_entity_passes():
    e = TypedEntity(etype="person", id="p1",
                    fields={"name": "Ada", "relationship": "girlfriend"})
    assert e.fields["name"] == "Ada"


def test_missing_required_raises():
    with pytest.raises(ValueError, match="requires field"):
        TypedEntity(etype="person", id="p1", fields={"notes": "no name"})


def test_unknown_field_warns_not_crashes():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        e = TypedEntity(etype="person", id="p1",
                        fields={"name": "Ada", "zzz": "kept"})
    assert any("unknown field" in str(x.message) for x in w)
    assert e.fields["zzz"] == "kept"  # kept, not dropped


def test_unknown_type_raises():
    with pytest.raises(ValueError, match="unknown entity type"):
        TypedEntity(etype="spaceship", id="s1", fields={})


def test_new_entity_fresh_id():
    a = new_entity("preference", {"topic": "briefings", "value": "morning"})
    b = new_entity("preference", {"topic": "briefings", "value": "morning"})
    assert a.id != b.id
    assert a.etype == "preference"


# ── person page seed ─────────────────────────────────────────────────────────

def test_parse_person_page(tmp_path):
    page = tmp_path / "ada.md"
    page.write_text(
        "---\n"
        "display_name: Ada\n"
        "summary: The user's girlfriend.\n"
        "rank: 1\n"
        "updated: 2026-10-03\n"
        "---\n\n"
        "# Ada\n\n## Facts\n- She is great.\n",
        encoding="utf-8")
    e = parse_person_page(page)
    assert e.etype == "person"
    assert e.fields["name"] == "Ada"
    assert e.fields["notes"] == "The user's girlfriend."
    assert e.fields["last_contact"] == "2026-10-03"


def test_parse_person_page_no_frontmatter(tmp_path):
    page = tmp_path / "bob.md"
    page.write_text("# Bob\n\nJust a guy.\n", encoding="utf-8")
    e = parse_person_page(page)
    assert e.fields["name"] == "Bob"  # stem fallback
    assert "Just a guy" in e.fields["notes"]


def test_parse_real_girlfriend_page():
    import pathlib
    p = pathlib.Path.home() / "memory" / "people" / "girlfriend.md"
    if not p.is_file():
        pytest.skip("no real people page in this env")
    e = parse_person_page(p)
    assert e.etype == "person"
    assert e.fields["name"]  # never empty


# ── sidecar store ────────────────────────────────────────────────────────────

def test_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("nomorals.memory.types.entity_home",
                        lambda settings=None: tmp_path)
    e = new_entity("commitment",
                   {"title": "call mom", "status": "open"})
    path = save_entity(e)
    assert path.is_file()
    back = load_entity(e.id, etype="commitment")
    assert back is not None
    assert back.fields["title"] == "call mom"


def test_load_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr("nomorals.memory.types.entity_home",
                        lambda settings=None: tmp_path)
    assert load_entity("nope") is None


# ── RRF ──────────────────────────────────────────────────────────────────────

def test_rrf_known_ranks():
    # vector: [a, b]; bm25: [b, a]. k=60.
    fused = rrf_fuse(["a", "b"], ["b", "a"])
    ids = [u for u, _, _, _ in fused]
    # b: 1/61 + 1/60 > a: 1/60 + 1/61 — tie actually; check determinism
    assert set(ids) == {"a", "b"}
    scores = {u: s for u, s, _, _ in fused}
    assert abs(scores["a"] - (1 / 60 + 1 / 61)) < 1e-9
    assert abs(scores["b"] - (1 / 61 + 1 / 60)) < 1e-9
    # tie → id order
    assert ids == ["a", "b"]


def test_rrf_single_signal_wins():
    fused = rrf_fuse(["x"], ["y"])
    assert fused[0][0] == "x"  # rank 0 in vector: 1/60 > 1/60... tie → id
    # both 1/60 → tie broken on id: x < y
    assert [u for u, _, _, _ in fused] == ["x", "y"]


def test_rrf_both_present_beats_single():
    fused = rrf_fuse(["a"], ["a", "b"])
    # a: 1/60 + 1/60 = 0.0333 > b: 1/61
    assert fused[0][0] == "a"
    assert fused[0][1] > fused[1][1]


def test_rrf_ranks_reported():
    fused = rrf_fuse(["a", "b"], ["b"])
    by_id = {u: (vr, br) for u, _, vr, br in fused}
    assert by_id["a"] == (0, None)
    assert by_id["b"] == (1, 0)


# ── hybrid_search ────────────────────────────────────────────────────────────

def test_hybrid_merges_and_dedupes():
    v = lambda q, n: [("a", "alpha text"), ("b", "beta text")]
    b = lambda q, n: [("b", "beta text"), ("c", "gamma text")]
    hits = hybrid_search("test", vector_fn=v, bm25_fn=b, limit=10)
    ids = [h.id for h in hits]
    assert ids == ["b", "a", "c"]  # b in both wins
    assert hits[0].vector_rank == 1
    assert hits[0].bm25_rank == 0


def test_hybrid_dead_signal_degrades():
    def boom(q, n):
        raise RuntimeError("vector down")
    b = lambda q, n: [("c", "gamma")]
    hits = hybrid_search("test", vector_fn=boom, bm25_fn=b, limit=10)
    assert [h.id for h in hits] == ["c"]


def test_hybrid_both_dead_returns_empty():
    def boom(q, n):
        raise RuntimeError("down")
    assert hybrid_search("x", vector_fn=boom, bm25_fn=boom) == []


def test_hybrid_limit():
    v = lambda q, n: [(str(i), f"t{i}") for i in range(20)]
    b = lambda q, n: []
    assert len(hybrid_search("x", vector_fn=v, bm25_fn=b, limit=5)) == 5


# ── type detection ───────────────────────────────────────────────────────────

def test_detect_person():
    assert detect_entity_type("find the person I met at the conference") == \
        "person"


def test_detect_commitment():
    assert detect_entity_type("what commitments do I have this week") == \
        "commitment"


def test_detect_ambiguous_none():
    assert detect_entity_type("what did we talk about") is None


def test_type_aware_filters():
    hits = [Hit(id="p1", text="Ada"), Hit(id="c1", text="call mom")]
    entities = {"p1": "person", "c1": "commitment"}
    out = type_aware_search("find the person I met", hits, entities)
    assert [h.id for h in out] == ["p1"]


def test_type_aware_passthrough_when_ambiguous():
    hits = [Hit(id="p1", text="Ada"), Hit(id="c1", text="call mom")]
    out = type_aware_search("what did we talk about", hits, {"p1": "person"})
    assert len(out) == 2


# ── HybridMemoryIndex ────────────────────────────────────────────────────────

def test_hybrid_index_search():
    idx = HybridMemoryIndex()
    idx.index_unit("u1", "fact", "my girlfriend is Ada")
    idx.index_unit("u2", "fact", "the sky is blue")
    hits = idx.bm25_search("girlfriend Ada")
    assert hits and hits[0][0] == "u1"


# ── tiers wiring ─────────────────────────────────────────────────────────────

@pytest.fixture()
def tt(tmp_path):
    m = TwoTierMemory(db=str(tmp_path / "t.db"))
    yield m
    m.close()


def test_recall_default_unchanged(tt):
    # default path has no hybrid_hits — behavior unchanged
    tt.facts.add_fact("my girlfriend is Ada", confidence=0.9)
    r = tt.recall("girlfriend name")
    assert r.hybrid_hits == []
    assert r.texts


def test_recall_hybrid_opt_in(tt):
    tt.facts.add_fact("my girlfriend is Ada", confidence=0.9)
    tt.observe("Ada had an operation last week.")
    r = tt.recall("Ada girlfriend", hybrid=True)
    assert r.hybrid_hits, "hybrid should fuse both lanes"
    # RRF scores visible for debugging
    assert all(h.rrf_score > 0 for h in r.hybrid_hits)


def test_resolve_id_fact(tt):
    f = tt.facts.add_fact("my dog is Rex", confidence=0.8)
    got = tt.resolve_id(f.id)
    assert got is not None
    assert got["kind"] == "fact"
    assert got["text"] == "my dog is Rex"
    assert got["active"] is True


def test_resolve_id_event(tt):
    ids = tt.events.record_event("We discussed the launch plan today.")
    assert ids
    got = tt.resolve_id(ids[0])
    assert got is not None
    assert got["kind"] == "event"


def test_resolve_id_session(tt):
    nid = tt.session.remember("s9", "debugging auth")
    got = tt.resolve_id(nid)
    assert got is not None
    assert got["kind"] == "session"
    assert got["session_id"] == "s9"


def test_resolve_id_missing(tt):
    assert tt.resolve_id("nope") is None
    assert tt.resolve_id("") is None
