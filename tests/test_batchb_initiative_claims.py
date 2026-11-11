"""Batch B — F03: initiative claims are atomic at the storage layer
and mutations are fenced to the lease generation.

Failing-first evidence: on base, ``claim`` returns a bare bool from
a read-check-save sequence (two connections can both win), and
``advance``/``complete`` accept no lease identity at all.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

from nomorals.agency.initiatives import (
    Initiative,
    InitiativeStore,
    StaleInitiativeLease,
)
from nomorals.agency.systems import ActiveSystems
from nomorals.storage.db import Database


def _store(tmp_path, name: str = "i.db") -> InitiativeStore:
    return InitiativeStore(Database(str(tmp_path / name)))


def _init(**kw) -> Initiative:
    base = dict(outcome="do the thing", kind="explicit_goal",
                next_action="start")
    base.update(kw)
    return Initiative(**base)


def test_claim_returns_the_lease_token_and_only_one_owner(tmp_path):
    store = _store(tmp_path)
    it = store.create(_init())
    token = store.claim(it.id, owner="executor-a", lease_s=60)
    assert isinstance(token, str) and token
    assert store.claim(it.id, owner="executor-b", lease_s=60) is None
    assert store.release(it.id, owner="executor-a") is True
    token_b = store.claim(it.id, owner="executor-b", lease_s=60)
    assert isinstance(token_b, str) and token_b
    assert token_b != token


def test_concurrent_claims_across_connections_have_exactly_one_winner(
        tmp_path):
    path = str(tmp_path / "race.db")
    seed = InitiativeStore(Database(path))
    it = seed.create(_init())
    barrier = threading.Barrier(8)
    wins: list[str] = []

    def racer(i: int) -> None:
        store = InitiativeStore(Database(path))
        barrier.wait(timeout=10)
        token = store.claim(it.id, owner=f"racer-{i}", lease_s=60)
        if token:
            wins.append(token)

    threads = [threading.Thread(target=racer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert len(wins) == 1, f"expected exactly one winner, got {len(wins)}"


def test_stale_owner_cannot_advance_or_complete(tmp_path):
    store = _store(tmp_path)
    it = store.create(_init())
    token_a = store.claim(it.id, owner="a", lease_s=60)
    assert token_a
    store.db.execute(
        "UPDATE agency_initiatives SET lease_until = ? WHERE id = ?",
        (time.time() - 1, it.id))
    token_b = store.claim(it.id, owner="b", lease_s=60)
    assert token_b and token_b != token_a
    with pytest.raises(StaleInitiativeLease):
        store.advance(it.id, next_action="a moves", lease_token=token_a)
    with pytest.raises(StaleInitiativeLease):
        store.complete(it.id, stop_reason="a finishes",
                       lease_token=token_a)
    with pytest.raises(StaleInitiativeLease):
        store.advance(it.id, next_action="anonymous move")
    moved = store.advance(it.id, next_action="b moves",
                          lease_token=token_b)
    assert moved.next_action == "b moves"
    done = store.complete(it.id, stop_reason="b finishes",
                          lease_token=token_b)
    assert done.status == "completed"


def test_unleased_mutations_keep_working(tmp_path):
    store = _store(tmp_path)
    it = store.create(_init())
    moved = store.advance(it.id, next_action="direct work")
    assert moved.next_action == "direct work"
    done = store.complete(it.id, stop_reason="delivered",
                          verification={"receipt": "r-1"})
    assert done.status == "completed"
    assert done.verification["receipt"] == "r-1"


def test_executor_claims_the_trigger_initiative_it_works(tmp_path):
    db = Database(os.path.join(str(tmp_path), "t.db"))
    store = InitiativeStore(db)
    systems = ActiveSystems(db=db, store=store)
    it = store.create(Initiative(
        outcome="lead trigger", objective="leads",
        kind="evidence_opportunity", dedupe_key="lead:quiet:1",
        next_action="work it"))
    systems._work_trigger(it, time.time())
    kinds = [e["kind"] for e in store.events(it.id)]
    assert "claimed" in kinds, f"executor never claimed the lease: {kinds}"
    assert "released" in kinds
    after = store.get(it.id)
    assert after is not None and after.lease_owner is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
