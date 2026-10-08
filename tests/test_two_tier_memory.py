"""Two-tier memory architecture (Khoj pattern) — build-map #36.

All offline. The critical invariant under test: the existing MemoryManager
is untouched — the two-tier layer is purely additive.
"""
import threading
import time

import pytest

from nomorals.memory.tiers import (
    EventStore,
    FactStore,
    MemoryUpdates,
    TwoTierMemory,
    chunk_text,
    distill_facts,
)
from nomorals.storage.db import Database


@pytest.fixture
def db():
    return Database(":memory:")


@pytest.fixture
def mem(db):
    m = TwoTierMemory(db=db)
    yield m
    m.close()


# ── chunking ─────────────────────────────────────────────────────────────

def test_chunking_respects_sentence_boundaries():
    text = "First sentence here. Second sentence here. " * 100
    chunks = chunk_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)
    # No chunk ends mid-sentence (each ends with sentence punctuation,
    # unless it was a single over-long sentence).
    for c in chunks:
        assert c.rstrip()[-1] in ".!?…", f"mid-sentence split: {c[-40:]!r}"


def test_chunking_short_text_single_chunk():
    assert chunk_text("Hello world.") == ["Hello world."]
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunking_long_sentence_clause_split():
    long_sent = "word, " * 400 + "end."
    chunks = chunk_text(long_sent)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)


# ── EventStore ───────────────────────────────────────────────────────────

def test_event_record_and_search(db):
    store = EventStore(db)
    ids = store.record_event("user: we talked about Lagos traffic\ndevon: yes, it was bad")
    assert len(ids) == 1
    assert store.count() == 1
    hits = store.search_events("Lagos traffic")
    assert hits, "should find the recorded event"
    assert "Lagos" in hits[0].text


def test_event_search_since_filter(db):
    store = EventStore(db)
    store.record_event("old event about cats", ts=1000.0)
    store.record_event("new event about cats", ts=2000.0)
    hits = store.search_events("cats", since=1500.0, limit=10)
    assert all(h.ts >= 1500.0 for h in hits)
    assert any("new event" in h.text for h in hits)


def test_event_record_empty_noop(db):
    store = EventStore(db)
    assert store.record_event("") == []
    assert store.record_event("   ") == []
    assert store.count() == 0
    assert store.search_events("") == []


# ── FactStore ────────────────────────────────────────────────────────────

def test_fact_add_and_search(db):
    store = FactStore(db)
    fact = store.add_fact("my girlfriend is named Ada", confidence=0.95)
    assert fact.id
    assert fact.active
    hits = store.search_facts("what is my girlfriend's name?")
    assert hits, "should find the fact"
    assert hits[0].fact.text == "my girlfriend is named Ada"


def test_fact_add_empty_rejected(db):
    store = FactStore(db)
    with pytest.raises(ValueError):
        store.add_fact("")
    with pytest.raises(ValueError):
        store.add_fact("   ")


def test_supersede_chain_auditable(db):
    store = FactStore(db)
    v1 = store.add_fact("my girlfriend is named Ada", confidence=0.9)
    v2 = store.supersede_fact(v1.id, "my ex-girlfriend was named Ada", confidence=0.85)
    assert v2.supersedes == v1.id
    assert not store.get_fact(v1.id).active
    assert store.get_fact(v2.id).active
    # Old version still retrievable — history is auditable.
    history = store.history(v2.id)
    assert [f.id for f in history] == [v1.id, v2.id]
    # Superseded facts don't surface in search.
    hits = store.search_facts("girlfriend named", limit=10)
    texts = [h.fact.text for h in hits]
    assert "my girlfriend is named Ada" not in texts


def test_supersede_unknown_or_inactive_rejected(db):
    store = FactStore(db)
    with pytest.raises(ValueError, match="unknown fact"):
        store.supersede_fact("nope", "new text")
    v1 = store.add_fact("fact one")
    v2 = store.supersede_fact(v1.id, "fact two")
    with pytest.raises(ValueError, match="already superseded"):
        store.supersede_fact(v1.id, "fact three — fork refused")


def test_fact_confidence_clamped(db):
    store = FactStore(db)
    f = store.add_fact("x", confidence=5.0)
    assert f.confidence == 1.0
    f2 = store.add_fact("y", confidence=-1.0)
    assert f2.confidence == 0.0


# ── distill_facts (Muninn) ───────────────────────────────────────────────

def _llm(returning):
    def fn(prompt):
        assert "EXCHANGE" in prompt
        return returning
    return fn


def test_distill_create_and_supersede():
    updates = distill_facts(
        "user: actually her name is Adaeze not Ada",
        llm_fn=_llm('{"create": [{"text": "her name is Adaeze", "confidence": 0.9}], '
                    '"supersede": [{"old_id": "f1", "text": "her name is Adaeze", '
                    '"confidence": 0.9}]}'))
    assert len(updates.create) == 1
    assert updates.create[0]["text"] == "her name is Adaeze"
    assert len(updates.supersede) == 1
    assert updates.supersede[0]["old_id"] == "f1"


def test_distill_no_llm_fn_empty():
    assert distill_facts("my name is Bob", llm_fn=None).empty()


def test_distill_empty_exchange_empty():
    assert distill_facts("", llm_fn=_llm('{"create":[]}')).empty()
    assert distill_facts("   ", llm_fn=_llm('{"create":[]}')).empty()


def test_distill_garbage_json_empty():
    updates = distill_facts("hello", llm_fn=_llm("not json at all {{{"))
    assert updates.empty()


def test_distill_llm_raises_empty():
    def boom(prompt):
        raise RuntimeError("model down")
    assert distill_facts("hello", llm_fn=boom).empty()


def test_distill_low_confidence_dropped():
    updates = distill_facts(
        "maybe I like tea?",
        llm_fn=_llm('{"create": [{"text": "I like tea", "confidence": 0.4}]}'))
    assert updates.empty(), "below the 0.6 creation bar"


def test_distill_code_fence_tolerated():
    updates = distill_facts(
        "my dog is Rex",
        llm_fn=_llm('```json\n{"create": [{"text": "my dog is Rex", '
                    '"confidence": 0.95}]}\n```'))
    assert len(updates.create) == 1


def test_distill_malformed_items_skipped():
    updates = distill_facts(
        "x",
        llm_fn=_llm('{"create": [{"text": ""}, "notadict", '
                    '{"text": "good fact", "confidence": 0.8}], '
                    '"supersede": [{"old_id": "", "text": "y"}]}'))
    assert len(updates.create) == 1
    assert updates.create[0]["text"] == "good fact"
    assert updates.supersede == []


# ── TwoTierMemory facade ─────────────────────────────────────────────────

def test_observe_records_event_sync(mem):
    ids = mem.observe("user: hi\ndevon: hello", distill=False)
    assert len(ids) == 1
    assert mem.stats["observed"] == 1


def test_observe_distill_async_nonblocking(mem):
    barrier = threading.Event()

    def slow_llm(prompt):
        barrier.wait(timeout=5)
        return '{"create": [{"text": "slow fact", "confidence": 0.9}]}'

    mem.llm_fn = slow_llm
    started = time.perf_counter()
    mem.observe("user: testing async", distill=True)
    elapsed = time.perf_counter() - started
    assert elapsed < 2.0, "observe must not block on the LLM"
    barrier.set()  # let the daemon thread finish


def test_observe_never_raises(mem):
    mem.observe("", distill=True)
    mem.observe(None, distill=True)  # type: ignore[arg-type]


def test_apply_updates_batch(mem):
    updates = MemoryUpdates(
        create=[{"text": "fact A", "confidence": 0.9}],
        supersede=[])
    done = mem.apply_updates(updates)
    assert done == {"created": 1, "superseded": 0, "skipped": 0}
    f = mem.facts.add_fact("fact B")
    done2 = mem.apply_updates(MemoryUpdates(
        supersede=[{"old_id": f.id, "text": "fact B v2", "confidence": 0.8}]))
    assert done2["superseded"] == 1


def test_recall_facts_first_for_direct_question(mem):
    mem.facts.add_fact("my girlfriend is named Ada", confidence=0.95)
    mem.events.record_event("we discussed the weather at length yesterday")
    result = mem.recall("what's my girlfriend's name?")
    assert result.facts_first
    assert result.facts, "fact should be found"
    assert result.texts[0] == "my girlfriend is named Ada"


def test_recall_events_lead_for_temporal_question(mem):
    mem.facts.add_fact("my girlfriend is named Ada", confidence=0.95)
    mem.events.record_event("we discussed Lagos traffic patterns")
    result = mem.recall("what did we discuss last tuesday?")
    assert not result.facts_first, "temporal questions lead with events"


def test_recall_empty_query_safe(mem):
    result = mem.recall("")
    assert result.texts == []


def test_recall_to_dict(mem):
    mem.facts.add_fact("I like morning briefings", confidence=0.8)
    d = mem.recall("do I like morning briefings?").to_dict()
    assert d["query"]
    assert "facts" in d and "events" in d


# ── the inviolable constraint: MemoryManager untouched ───────────────────

def test_memory_manager_still_works_untouched():
    """The existing memory path is not modified by this module — import it
    and exercise remember/recall to prove the contract holds."""
    from unittest.mock import MagicMock
    from nomorals.memory.manager import MemoryManager

    ctx = MagicMock()
    ctx.settings = None
    db = Database(":memory:")
    db.migrate()
    ctx.db = db
    mgr = MemoryManager(ctx)
    rid = mgr.remember("the sky is blue", kind="fact", importance=0.9)
    assert rid
    result = mgr.recall("what color is the sky")
    assert result.records, "existing recall still works"
    assert any("sky" in r.content for r in result.records)
