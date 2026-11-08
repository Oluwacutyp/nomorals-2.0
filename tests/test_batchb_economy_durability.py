"""Batch B — economy durability (doc 3 findings B–D, owner-adopted):
persistence failure propagates, multi-leg operations commit
atomically, writers are serialized across instances, and receipts
are idempotent under an operation id.

Failing-first evidence: on base, ``_save_json`` swallows OSError
(award reports success with a zero stored balance), transfers
append each leg in a separate save (a crash between legs leaves a
partial move), two instances lose each other's awards, and no
operation accepts an ``op_id``.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest import mock

import pytest

from nomorals.community.group_economy import (
    GroupEconomy,
    GroupEconomyError,
    PrizePool,
)


def _funded(tmp_path) -> GroupEconomy:
    econ = GroupEconomy(tmp_path / "econ")
    econ.verify_member("g", "alice")
    econ.verify_member("g", "bob")
    econ.award("g", "alice", 100, "seed")
    return econ


def _fail_replace():
    return mock.patch.object(Path, "replace",
                             side_effect=OSError("disk full"))


def _flaky_replace(fail_after: int):
    real_replace = Path.replace
    state = {"n": 0}

    def flaky(self, target):
        state["n"] += 1
        if state["n"] > fail_after:
            raise OSError("crash between legs")
        return real_replace(self, target)

    return mock.patch.object(Path, "replace", flaky)


def test_failed_save_propagates_and_no_award_is_reported(tmp_path):
    econ = _funded(tmp_path)
    before = econ.balance("g", "alice")
    with _fail_replace():
        with pytest.raises(GroupEconomyError):
            econ.award("g", "alice", 10, "more")
    assert econ.balance("g", "alice") == before


def test_prize_save_failure_propagates(tmp_path):
    pool = PrizePool(tmp_path / "econ")
    with _fail_replace():
        with pytest.raises(GroupEconomyError):
            pool.create_prize("g", "pizza night")


def test_transfer_commits_both_legs_or_neither(tmp_path):
    econ = _funded(tmp_path)
    with _flaky_replace(fail_after=1):
        try:
            econ.transfer("g", "alice", "bob", 40)
        except GroupEconomyError:
            pass
    outcome = (econ.balance("g", "alice"), econ.balance("g", "bob"))
    assert outcome in ((60, 40), (100, 0)), (
        f"partial transfer persisted: {outcome}")


def test_taxed_transfer_is_one_commit(tmp_path):
    econ = _funded(tmp_path)
    with _flaky_replace(fail_after=1):
        try:
            econ.transfer_taxed("g", "alice", "bob", 50, 5)
        except GroupEconomyError:
            pass
    outcome = (econ.balance("g", "alice"), econ.balance("g", "bob"),
               econ.treasury_balance("g"))
    assert outcome in ((50, 45, 5), (100, 0, 0)), (
        f"partial taxed transfer persisted: {outcome}")


def test_concurrent_instances_never_lose_an_award(tmp_path):
    econ_a = GroupEconomy(tmp_path / "econ")
    econ_b = GroupEconomy(tmp_path / "econ")
    econ_a.verify_member("g", "alice")
    econ_a.verify_member("g", "bob")
    gate = threading.Event()
    entered = threading.Event()
    real_save = GroupEconomy._save_json

    def blocking_save(self, group_id, name, data):
        if name == "ledger" and self is econ_a:
            entered.set()
            gate.wait(timeout=10)
        return real_save(self, group_id, name, data)

    with mock.patch.object(GroupEconomy, "_save_json", blocking_save):
        ta = threading.Thread(
            target=lambda: econ_a.award("g", "alice", 10, "a"))
        ta.start()
        assert entered.wait(timeout=10)
        tb = threading.Thread(
            target=lambda: econ_b.award("g", "bob", 20, "b"))
        tb.start()
        time.sleep(0.5)
        gate.set()
        ta.join(timeout=10)
        tb.join(timeout=10)
    assert econ_a.balance("g", "alice") == 10
    assert econ_a.balance("g", "bob") == 20, (
        "a concurrent award was lost to a stale snapshot overwrite")
    assert len(econ_a._ledger("g")) == 2


def test_transfer_retry_with_op_id_moves_money_once(tmp_path):
    econ = _funded(tmp_path)
    assert econ.transfer("g", "alice", "bob", 30, op_id="pay-1") is True
    assert econ.transfer("g", "alice", "bob", 30, op_id="pay-1") is True
    assert (econ.balance("g", "alice"), econ.balance("g", "bob")) == (70, 30)
    entries = [e for e in econ._ledger("g") if e.op_id == "pay-1"]
    assert len(entries) == 2
    assert econ.transfer("g", "alice", "bob", 5, op_id="pay-2") is True
    assert econ.balance("g", "bob") == 35


def test_award_retry_with_op_id_returns_the_original_entry(tmp_path):
    econ = GroupEconomy(tmp_path / "econ")
    econ.verify_member("g", "alice")
    first = econ.award("g", "alice", 25, "daily",
                       op_id="daily-2026-11-08:alice")
    second = econ.award("g", "alice", 25, "daily",
                        op_id="daily-2026-11-08:alice")
    assert first is not None and second is not None
    assert second.id == first.id
    assert econ.balance("g", "alice") == 25


def test_bank_round_trip_stays_balanced(tmp_path):
    econ = _funded(tmp_path)
    assert econ.bank_deposit("g", "alice", 40, op_id="dep-1") is True
    assert econ.bank_deposit("g", "alice", 40, op_id="dep-1") is True
    assert econ.balance("g", "alice") == 60
    assert econ.bank_balance("g", "alice") == 40
    assert econ.bank_withdraw("g", "alice", 15) is True
    assert econ.balance("g", "alice") == 75
    assert econ.bank_balance("g", "alice") == 25


def test_quest_claim_retry_after_partial_failure_pays_once(tmp_path):
    econ = GroupEconomy(tmp_path / "econ")
    econ.verify_member("g", "alice")
    for _ in range(3):
        econ.record_activity("g", "alice", "helpful_answer")
    real_replace = Path.replace

    def flaky(self, target):
        if str(target).endswith("quest_claims.json"):
            raise OSError("crash after the award committed")
        return real_replace(self, target)

    with mock.patch.object(Path, "replace", flaky):
        with pytest.raises(GroupEconomyError):
            econ.claim_quest("g", "alice", "helpful_3")
    entry = econ.claim_quest("g", "alice", "helpful_3")
    assert entry is not None
    awards = [e for e in econ._ledger("g") if e.ref == "quest:helpful_3"]
    assert len(awards) == 1, "the retry paid the quest reward twice"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
