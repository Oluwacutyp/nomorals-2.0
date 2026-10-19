"""PASS #12: memory + knowledge store — verified bugs, failing-first.

1. remember() auto-fired the DESTRUCTIVE consolidate() once the store
   hit consolidate_every records — silently deleting near-duplicate
   episodes and low-value memories with no explicit owner command,
   contradicting the documented "explicit, human-invoked use only" rule.
2. KnowledgeStore.search() served EXPIRED entries as fresh, and
   prune_expired() had zero production callers.
3. KnowledgeStore.deposit() silently dropped duplicate sightings —
   corroboration from a second source was lost instead of recorded.
"""
import time

import pytest

from nomorals.agency.knowledge import KnowledgeStore
from nomorals.memory.base import MemoryKind
from nomorals.storage.db import Database


@pytest.fixture()
def mem():
    import shutil
    import tempfile

    from nomorals.agents.context import build_context
    from nomorals.core.config import Settings
    from nomorals.memory.embeddings import Embedder

    home = tempfile.mkdtemp(prefix="nm-mem12-")
    settings = Settings(home=home)
    context = build_context(settings)
    context.__enter__()
    m = context.memory
    m.embedder = Embedder(provider="hashing", dimensions=512)
    m.consolidate_every = 10 ** 9  # disabled unless a test opts in
    yield m
    context.__exit__(None, None, None)
    shutil.rmtree(home, ignore_errors=True)


def _remember_episodes(m, n, text="the owner likes strong coffee"):
    ids = []
    for _ in range(n):
        ids.append(m.remember(text, kind=MemoryKind.EPISODE,
                              importance=0.9, index=False))
    return ids


def test_remember_never_auto_deletes(mem):
    # The destructive consolidate() (simhash dedupe + forget_below) must
    # never run on its own: the owner's rule is "forget only on explicit
    # command". Additive distillation is the only automatic path.
    # (consolidate() has a min_episodes=5 floor, so this needs 6.)
    mem.consolidate_every = 5
    ids = _remember_episodes(mem, 6)  # 6th remember crosses the threshold
    assert len(ids) == 6
    assert all(ids)
    # Nothing deleted: every episode still retrievable. The additive path
    # may ADD a distilled fact (count 7), but it must never remove.
    for rid in ids:
        assert mem.get(rid) is not None, \
            f"remember() auto-deleted record {rid} via destructive consolidate()"
    assert mem.repo.count() >= 6


def test_remember_still_distills_additively(mem):
    # The automatic path must still distil durable facts — additively.
    mem.consolidate_every = 5
    _remember_episodes(mem, 6, text="episode about the morning walk routine")
    facts = [r for r in mem.db.query("SELECT * FROM memories")
             if r["kind"] == MemoryKind.FACT and "distilled_from" in str(r.get("metadata") or "")]
    assert mem.repo.count() >= 6, "additive path must not delete episodes"
    assert facts, "additive consolidation should have distilled a fact"


def test_knowledge_search_excludes_expired():
    db = Database(":memory:")
    ks = KnowledgeStore(db)
    ks.deposit("research", "the reactor coolant threshold is 400C",
               source="manual-a", ttl_s=3600)
    # already-expired at deposit time
    ks.deposit("research", "the old pump model is discontinued",
               source="manual-b", ttl_s=1, now=time.time() - 10)
    hits = ks.search("pump discontinued")
    assert all("discontinued" not in h["text"] for h in hits), \
        "search() served an expired knowledge entry as fresh"
    hits = ks.search("reactor coolant")
    assert len(hits) == 1, "unexpired entry must still be found"


def test_knowledge_dedupe_records_corroboration():
    db = Database(":memory:")
    ks = KnowledgeStore(db)
    assert ks.deposit("research", "the reactor coolant threshold is 400C",
                      source="manual-a") is True
    assert ks.deposit("news", "the reactor coolant threshold is 400C",
                      source="feed-b") is False  # duplicate: not stored twice
    assert ks.count() == 1, "duplicate must not create a second row"
    row = db.query_one("SELECT * FROM agency_knowledge")
    assert int(row["sightings"]) == 2, \
        "duplicate sighting must increment the corroboration count"
    sources = str(row.get("sources") or "")
    assert "manual-a" in sources and "feed-b" in sources, \
        f"both provenances must be recorded, got: {sources!r}"
    assert float(row.get("last_seen") or 0) > 0


def test_knowledge_prune_expired_removes_only_expired():
    db = Database(":memory:")
    ks = KnowledgeStore(db)
    ks.deposit("research", "permanent fact about alloys", source="s")
    ks.deposit("research", "temporary outage notice", source="s",
               ttl_s=1, now=time.time() - 10)
    assert ks.count() == 2
    dropped = ks.prune_expired()
    assert dropped == 1
    assert ks.count() == 1
    assert ks.search("outage") == []
    assert len(ks.search("alloys")) == 1


def _seed_identical_episodes(m, n=6, text="the owner takes the night bus"):
    from nomorals.memory.base import MemoryKind

    ids = []
    for _ in range(n):
        rid = m.remember(text, kind=MemoryKind.EPISODE, importance=0.9,
                         index=False)
        assert rid
        ids.append(rid)
    return ids


def test_role_agents_never_destructively_consolidate(mem):
    # MemoryAgent (default action) and ReflectionAgent both consolidated
    # destructively on task input — deleting near-duplicate episodes with
    # no explicit owner command. Both must use the additive path now.
    from types import SimpleNamespace

    from nomorals.agents.roles import MemoryAgent, ReflectionAgent
    from nomorals.core.tasks import Task

    ids = _seed_identical_episodes(mem)
    ctx = SimpleNamespace(memory=mem, tools=None)

    out = MemoryAgent(context=ctx).work(Task(name="t", payload={}))
    assert out.get("error") is None, out

    out = ReflectionAgent(context=ctx).work(Task(name="t", payload={}))
    assert out.get("error") is None, out

    for rid in ids:
        assert mem.get(rid) is not None, \
            f"role-agent consolidate deleted record {rid}"
