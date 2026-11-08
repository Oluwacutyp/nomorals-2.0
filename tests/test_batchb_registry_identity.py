"""Batch B — F01 (idempotency publish-order + durable claims) and
F08 (typed results at the tool boundary).

Failing-first evidence: on base bab2041/2e645ff these tests fail —
the reservation is released before the record is stored (duplicate
execution in the gap), a second registry instance sharing the db
executes a claimed key again, ``result_protocol``/``OpResult`` do
not exist, and a real dict-protocol tool's ``{"ok": False}`` return
is wrapped and memoized as success.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from nomorals.storage.db import Database
from nomorals.tools.registry import OpResult, ToolRegistry


def _reg_with_db(tmp_path):
    db = Database(str(tmp_path / "reg.db"))
    return ToolRegistry(SimpleNamespace(db=db)), db


# ── F01: publish before release ─────────────────────────────────────


def test_record_is_published_before_the_reservation_is_released(tmp_path):
    reg, _db = _reg_with_db(tmp_path)
    calls: list[int] = []

    def effect(x: int = 0) -> str:
        calls.append(x)
        return f"v-{x}"

    reg.register("effect", effect, capability="fs.write")
    entered = threading.Event()
    unblock = threading.Event()
    real_store = reg._idempotency_store

    def blocking_store(rec):
        entered.set()
        assert unblock.wait(timeout=10)
        return real_store(rec)

    reg._idempotency_store = blocking_store
    results: dict[str, object] = {}

    def call(tag: str) -> None:
        results[tag] = reg.call("effect", actor="alice",
                                idempotency_key="k", x=5)

    t1 = threading.Thread(target=call, args=("first",))
    t1.start()
    assert entered.wait(timeout=10)
    t2 = threading.Thread(target=call, args=("second",))
    t2.start()
    time.sleep(0.5)
    assert calls == [5], (
        "a duplicate executed while the owner was between its effect "
        "and its record — the reservation was released before the "
        "record was published")
    unblock.set()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert calls == [5]
    assert results["first"].ok and results["second"].ok
    assert results["second"].value == "v-5"


def test_second_registry_instance_cannot_duplicate_a_claimed_key(tmp_path):
    db = Database(str(tmp_path / "reg.db"))
    ctx = SimpleNamespace(db=db)
    reg_a, reg_b = ToolRegistry(ctx), ToolRegistry(ctx)
    calls: list[int] = []
    started = threading.Event()
    finish = threading.Event()

    def slow(x: int = 0) -> str:
        calls.append(x)
        started.set()
        assert finish.wait(timeout=10)
        return "done"

    for reg in (reg_a, reg_b):
        reg.register("slow", slow, capability="fs.write")
    out: dict[str, object] = {}
    ta = threading.Thread(
        target=lambda: out.setdefault(
            "a", reg_a.call("slow", actor="alice",
                            idempotency_key="k2", x=1)))
    ta.start()
    assert started.wait(timeout=10)
    tb = threading.Thread(
        target=lambda: out.setdefault(
            "b", reg_b.call("slow", actor="alice",
                            idempotency_key="k2", x=1)))
    tb.start()
    time.sleep(0.5)
    assert calls == [1], "a second process/instance executed a claimed key"
    finish.set()
    ta.join(timeout=10)
    tb.join(timeout=10)
    assert calls == [1]
    assert out["b"].ok and out["b"].value == "done"


def _seed_claim(db, *, key: str, claimed_at: float) -> None:
    db.execute(
        "CREATE TABLE IF NOT EXISTS idempotency_claims ("
        "tool TEXT NOT NULL, key TEXT NOT NULL, owner_token TEXT NOT NULL,"
        " actor TEXT NOT NULL, args_digest TEXT NOT NULL,"
        " tool_version TEXT NOT NULL, claimed_at REAL NOT NULL,"
        " PRIMARY KEY (tool, key))")
    db.execute(
        "INSERT INTO idempotency_claims (tool, key, owner_token, actor,"
        " args_digest, tool_version, claimed_at) VALUES (?,?,?,?,?,?,?)",
        ("effect", key, "dead-owner", "alice",
         ToolRegistry._args_digest({"x": 1}), "", claimed_at))


def test_stale_claim_is_reported_uncertain_never_reexecuted(tmp_path):
    reg, db = _reg_with_db(tmp_path)
    calls: list[int] = []

    def effect(x: int = 0) -> str:
        calls.append(x)
        return "ran"

    reg.register("effect", effect, capability="fs.write")
    _seed_claim(db, key="k3", claimed_at=time.time() - 100_000)
    out = reg.call("effect", actor="alice", idempotency_key="k3", x=1)
    assert not out.ok
    assert out.error.code == "tool.idempotency_uncertain"
    assert calls == [], "a crashed owner's key was silently re-executed"


def test_fresh_foreign_claim_waits_then_reports_uncertain(tmp_path):
    reg, db = _reg_with_db(tmp_path)
    calls: list[int] = []

    def effect(x: int = 0) -> str:
        calls.append(x)
        return "ran"

    reg.register("effect", effect, capability="fs.write")
    _seed_claim(db, key="k4", claimed_at=time.time())
    reg._idempotency_wait_s = 0.3
    started = time.time()
    out = reg.call("effect", actor="alice", idempotency_key="k4", x=1)
    assert time.time() - started < 10
    assert not out.ok
    assert out.error.code == "tool.idempotency_uncertain"
    assert calls == [], "a waiter became a duplicate executor"


def test_failed_call_frees_the_key_for_a_retry(tmp_path):
    reg, _db = _reg_with_db(tmp_path)
    calls: list[int] = []

    def flaky(x: int = 0) -> str:
        calls.append(x)
        if len(calls) == 1:
            raise RuntimeError("died mid-run")
        return "recovered"

    reg.register("flaky", flaky, capability="fs.write")
    first = reg.call("flaky", actor="alice", idempotency_key="k5", x=1)
    assert not first.ok
    second = reg.call("flaky", actor="alice", idempotency_key="k5", x=1)
    assert second.ok and second.value == "recovered"
    assert calls == [1, 1]


# ── F08: typed results at the boundary ──────────────────────────────


def test_dict_protocol_failure_is_typed_and_not_memoized():
    reg = ToolRegistry(None)
    calls: list[int] = []

    def flaky() -> dict:
        calls.append(1)
        if len(calls) == 1:
            return {"ok": False, "error": "synthetic transient",
                    "retryable": True}
        return {"ok": True, "value": 7}

    reg.register("flaky", flaky, capability="net.out",
                 result_protocol="dict_ok")
    first = reg.call("flaky", actor="a", idempotency_key="dk")
    assert not first.ok, "a domain failure was wrapped as success"
    assert first.error.retryable is True
    assert "synthetic transient" in str(first.error)
    second = reg.call("flaky", actor="a", idempotency_key="dk")
    assert second.ok and second.value["value"] == 7
    assert len(calls) == 2, "the failure was memoized — the retry never ran"
    third = reg.call("flaky", actor="a", idempotency_key="dk")
    assert third.ok and len(calls) == 2, "the success was not memoized"


def test_opresult_partial_is_typed_and_not_a_terminal_success():
    reg = ToolRegistry(None)
    calls: list[int] = []

    def partial() -> OpResult:
        calls.append(1)
        return OpResult.partial({"rows": 3}, error="2 of 5 sources answered")

    reg.register("partial", partial, capability="net.out")
    first = reg.call("partial", actor="a", idempotency_key="pk")
    assert first.ok
    assert isinstance(first.value, OpResult)
    assert first.value.status == "partial"
    assert first.value.error == "2 of 5 sources answered"
    second = reg.call("partial", actor="a", idempotency_key="pk")
    assert second.ok
    assert len(calls) == 2, "a partial result was memoized as terminal success"


def test_opresult_failed_and_uncertain_become_typed_errors():
    reg = ToolRegistry(None)

    def failed() -> OpResult:
        return OpResult.failed("permanent: no such dataset", retryable=False)

    def uncertain() -> OpResult:
        return OpResult.uncertain("the provider may have applied the change")

    reg.register("failed", failed, capability="net.out")
    reg.register("uncertain", uncertain, capability="net.out")
    out_f = reg.call("failed", actor="a")
    assert not out_f.ok
    assert "no such dataset" in str(out_f.error)
    assert out_f.error.retryable is False
    out_u = reg.call("uncertain", actor="a")
    assert not out_u.ok
    assert out_u.error.code == "tool.result_uncertain"
    assert out_u.error.retryable is False


def test_ok_field_is_plain_data_without_the_protocol():
    reg = ToolRegistry(None)
    reg.register("data", lambda: {"ok": False, "error": "just a data row"},
                 capability="fs.read")
    out = reg.call("data", actor="a", idempotency_key="nk")
    assert out.ok
    assert out.value == {"ok": False, "error": "just a data row"}


def test_real_accounts_tool_failure_is_not_a_success():
    import nomorals.accounts.temp_mail as temp_mail
    from nomorals.tools import accounts as accounts_tools

    reg = ToolRegistry(None)
    accounts_tools.register(reg)

    def boom(providers=None):
        raise RuntimeError("provider down")

    original = temp_mail.grab_address_cascade
    temp_mail.grab_address_cascade = boom
    try:
        out = reg.call("temp_email", actor="a", idempotency_key="tk")
    finally:
        temp_mail.grab_address_cascade = original
    assert not out.ok, "a real tool's {'ok': False} was reported as success"
    assert "provider down" in str(out.error)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
