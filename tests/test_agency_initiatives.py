"""Failing-first tests: initiative lifecycle (audit build step 2).

Invariant under test: every nonterminal initiative has an executable
next action, a timed wake-up, or a specific blocker.
"""
import time

import pytest

from nomorals.agency.initiatives import Initiative, InitiativeStore
from nomorals.agency.deliberation import DeliberationLoop
from nomorals.storage.db import Database


def _store():
    return InitiativeStore(Database(":memory:"))


def _init(**kw):
    base = dict(
        outcome="Ship demo", kind="explicit_goal", objective="portfolio",
        evidence="owner asked for demo", next_action="build demo",
        next_review_at=time.time() + 3600,
    )
    base.update(kw)
    return Initiative(**base)


def test_invariant_rejects_initiative_with_no_next_step():
    store = _store()
    with pytest.raises(ValueError):
        store.create(_init(next_action="", next_review_at=0, blocker=""))


def test_dedupe_merges_instead_of_multiplying():
    store = _store()
    a = store.create(_init(dedupe_key="gig:123"))
    b = store.create(_init(dedupe_key="gig:123", evidence="second sighting"))
    assert a.id == b.id
    assert len(store.list_active()) == 1
    assert "second sighting" in store.get(a.id).evidence


def test_lease_claim_prevents_second_entrypoint():
    store = _store()
    it = store.create(_init())
    assert store.claim(it.id, owner="executor-a", lease_s=60)
    assert not store.claim(it.id, owner="executor-b", lease_s=60)
    store.release(it.id, owner="executor-a")
    assert store.claim(it.id, owner="executor-b", lease_s=60)


def test_boot_reconcile_recovers_expired_lease_and_reopens_stale():
    store = _store()
    it = store.create(_init())
    assert store.claim(it.id, owner="dead-worker", lease_s=60)
    # Simulate a crashed worker: force the lease into the past.
    store.db.execute(
        "UPDATE agency_initiatives SET lease_until=? WHERE id=?",
        (time.time() - 1, it.id))
    recovered = store.boot_reconcile(now=time.time())
    assert recovered["leases_recovered"] == 1
    assert store.get(it.id).lease_owner is None


def test_complete_records_stop_reason_and_verification():
    store = _store()
    it = store.create(_init())
    store.complete(it.id, stop_reason="delivered",
                   verification={"receipt": "acceptance-msg-1"})
    done = store.get(it.id)
    assert done.status == "completed"
    assert done.stop_reason == "delivered"
    assert done.verification["receipt"] == "acceptance-msg-1"


def test_deliberation_ages_waiting_and_never_starves_critical():
    store = _store()
    now = time.time()
    store.create(_init(outcome="critical overdue", kind="standing",
                       priority=1.0, next_review_at=now - 7200,
                       next_action="fix it"))
    store.create(_init(outcome="fresh research", kind="exploration",
                       priority=0.9, next_review_at=now + 3600,
                       next_action="read more"))
    loop = DeliberationLoop(store)
    chosen = loop.choose(now=now, capacity=1)
    assert len(chosen) == 1
    assert chosen[0].outcome == "critical overdue"
