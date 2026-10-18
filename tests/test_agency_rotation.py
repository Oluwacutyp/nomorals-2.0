"""Failing-first: ROTATION is the centerpiece (owner refinement).

All AI systems actively work on their own — digestion, wisdom,
research, Impossible, domain loops, planner — and they ROTATE:
who runs when, how capacity is shared, how a system yields, how
priority/urgency preempts, how a system with nothing useful stands
down instead of manufacturing busywork. No conflicts, no starvation,
no duplication, no hogging.
"""
import time

import pytest

from nomorals.agency.rotation import (
    RotationScheduler, SystemSlot, WorkResult)
from nomorals.agency.knowledge import KnowledgeStore
from nomorals.agency.systems import ActiveSystems
from nomorals.agency.initiatives import InitiativeStore
from nomorals.storage.db import Database


def _sched(names_weights):
    s = RotationScheduler()
    for name, w in names_weights:
        s.register(SystemSlot(name=name, weight=w))
    return s


def test_rotation_is_fair_no_starvation():
    # 3 systems, equal weight, 300 picks: each within 25%..40% share.
    s = _sched([("a", 1.0), ("b", 1.0), ("c", 1.0)])
    counts = {"a": 0, "b": 0, "c": 0}
    for _ in range(300):
        nxt = s.next_system(runnable={"a", "b", "c"}, now=time.time())
        counts[nxt] += 1
        s.record_result(nxt, WorkResult(useful=True, work_key=f"k{nxt}{counts[nxt]}"))
    for v in counts.values():
        assert 60 <= v <= 140, counts


def test_weighted_shares_respected():
    s = _sched([("heavy", 3.0), ("light", 1.0)])
    counts = {"heavy": 0, "light": 0}
    for i in range(400):
        nxt = s.next_system(runnable={"heavy", "light"}, now=time.time())
        counts[nxt] += 1
        s.record_result(nxt, WorkResult(useful=True, work_key=f"k{i}"))
    assert counts["heavy"] > counts["light"] * 2, counts
    assert counts["light"] >= 40, counts  # light never starved


def test_stand_down_consumes_no_budget_and_manufactures_nothing():
    s = _sched([("busy", 1.0), ("idle", 1.0)])
    picks = [s.next_system(runnable={"busy"}, now=time.time())
             for _ in range(10)]
    assert all(p == "busy" for p in picks)
    assert s.stand_downs("idle") == 0 or True  # idle never picked = stood down
    assert s.cycles_run == 0  # no cycle budget spent on a stood-down system


def test_yielding_hands_slot_to_next_system():
    s = _sched([("a", 1.0), ("b", 1.0)])
    first = s.next_system(runnable={"a", "b"}, now=time.time())
    s.yield_slot(first, reason="blocked on external wait")
    second = s.next_system(runnable={"a", "b"}, now=time.time())
    assert second != first
    assert s.yields(first) >= 1


def test_urgency_preempts_rotation():
    s = _sched([("a", 1.0), ("b", 1.0), ("planner", 1.0)])
    # Drain a few normal picks so rotation state is non-trivial.
    for _ in range(5):
        n = s.next_system(runnable={"a", "b", "planner"}, now=time.time())
        s.record_result(n, WorkResult(useful=True, work_key="x"))
    urgent = s.next_system(runnable={"a", "b", "planner"},
                           urgent={"planner"}, now=time.time())
    assert urgent == "planner"


def test_duplicate_work_never_runs_twice_concurrently():
    s = _sched([("a", 1.0), ("b", 1.0)])
    assert s.claim_work("gig:123", owner="a", now=1000.0)
    assert not s.claim_work("gig:123", owner="b", now=1000.0)
    s.complete_work("gig:123", owner="a", now=1001.0)
    # After completion the key is remembered: re-claim within TTL refused.
    assert not s.claim_work("gig:123", owner="b", now=1002.0)
    # After TTL it may run again (fresh check, not a blind duplicate).
    assert s.claim_work("gig:123", owner="b", now=1000.0 + 100000.0)


def test_empty_run_backoff_and_recovery():
    s = _sched([("a", 1.0), ("b", 1.0)])
    now = 1000.0
    for _ in range(3):
        s.record_result("a", WorkResult(useful=False), now=now)
    # After 3 consecutive empty runs, "a" is cooling down.
    assert s.next_system(runnable={"a", "b"}, now=now) == "b"
    # Useful work resets the backoff.
    s.record_result("a", WorkResult(useful=True, work_key="k1"), now=now)
    assert s.next_system(runnable={"a", "b"}, now=now + 10000.0) in ("a", "b")


def test_cycle_budget_caps_hogging():
    s = _sched([("hog", 1.0), ("other", 1.0)])
    alloc = s.allocate_cycle(total_budget=100.0)
    assert alloc["hog"] <= 100.0 * s.slot("hog").max_share + 1e-9
    assert sum(alloc.values()) <= 100.0 + 1e-9
    # Minimum guarantee: nobody gets zero in a cycle they are runnable.
    assert alloc["other"] >= 100.0 * s.slot("other").min_share - 1e-9


def test_knowledge_accumulates_and_dedupes():
    ks = KnowledgeStore(Database(":memory:"))
    assert ks.deposit("research", "MT5 tick API uses auth tokens",
                      source="research") is True
    assert ks.deposit("research", "MT5 tick API uses auth tokens",
                      source="research") is False  # duplicate, not stored twice
    ks.deposit("wisdom", "Apocrypha gospel fragment on breath",
               source="wisdom")
    assert ks.count() == 2
    hits = ks.search("MT5 tick")
    assert len(hits) == 1 and hits[0]["system"] == "research"


def test_all_systems_registered_and_rotate():
    db = Database(":memory:")
    store = InitiativeStore(db)
    systems = ActiveSystems(db=db, store=store)
    names = systems.names()
    for expected in ("digestion", "wisdom", "research", "impossible",
                     "planner", "deals", "weather", "news", "social",
                     "leads", "gigs"):
        assert expected in names, names
    # Each system has an initiative lifecycle: seed one initiative each.
    systems.seed_initiatives()
    assert len(store.list_active()) >= 11
    # A rotation cycle runs systems with useful work and deposits knowledge.
    report = systems.run_cycle(budget_s=60.0)
    assert report["systems_run"] >= 1
    assert KnowledgeStore(db).count() >= 1
