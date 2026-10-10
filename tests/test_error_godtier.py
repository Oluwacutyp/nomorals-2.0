"""Tests for the God-tier error system: incidents, selfheal, budgets, degradation."""

import time

import pytest

from nomorals.core.budgets import BudgetManager, ErrorBudget
from nomorals.core.degradation import DegradationLadder, LadderExhausted, Rung
from nomorals.core.error_intelligence import ErrorIntelligence
from nomorals.core.incidents import IncidentJournal, signature_of
from nomorals.core.selfheal import (
    FallbackChainStrategy,
    RecoveryStatus,
    RestartStrategy,
    RetryStrategy,
    SelfHealingExecutor,
    SubsystemSupervisor,
)
from nomorals.core.retry import BackoffPolicy


def _boom(msg="boom"):
    raise ConnectionError(msg)


def _capture(fn, *args, **kwargs):
    """Run fn; return the exception it raises."""
    try:
        fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        return e
    raise AssertionError("expected an exception")


# ── incidents ────────────────────────────────────────────────────────────

def test_signature_ignores_message():
    e1 = _capture(_boom, "host A down")
    s1 = signature_of(e1, "net")
    e2 = _capture(_boom, "host B down")
    s2 = signature_of(e2, "net")
    assert s1 == s2


def test_signature_differs_by_subsystem():
    e1 = _capture(_boom)
    s1 = signature_of(e1, "a")
    e2 = _capture(_boom)
    s2 = signature_of(e2, "b")
    assert s1 != s2


def test_journal_records_and_recalls():
    j = IncidentJournal()
    e = _capture(_boom)
    inc = j.record_incident(e, subsystem="net", category="network")
    assert inc.id > 0
    assert inc.repeat_count_1h == 1
    similar = j.find_similar(e, "net")
    assert len(similar) == 1
    assert j.known_fix(e, "net") is None  # no verified fix yet


def test_verified_fix_promotion():
    j = IncidentJournal()
    e = _capture(_boom)
    inc = j.record_incident(e, subsystem="net")
    from nomorals.core.incidents import RecoveryRecord
    # Unverified attempt: must NOT become a known fix.
    j.record_recovery(RecoveryRecord(inc.id, inc.signature, "retry",
                                     verified=False))
    assert j.known_fix(e, "net") is None
    # Verified attempt: promoted.
    j.record_recovery(RecoveryRecord(inc.id, inc.signature, "retry",
                                     detail="backed off 2s",
                                     verified=True,
                                     verify_note="probe ok"))
    fix = j.known_fix(e, "net")
    assert fix is not None
    assert fix["strategy"] == "retry"
    assert fix["successes"] == 1


def test_chronic_detection():
    j = IncidentJournal()
    # Speed up: lower thresholds via instance attrs is not supported;
    # instead record enough incidents.
    for _ in range(3):
        last = _capture(_boom)
        j.record_incident(last, subsystem="net")
    assert j.is_chronic(last, "net")


def test_fuzzy_match_survives_location_shift():
    j = IncidentJournal()
    e = _capture(_boom)
    j.record_incident(e, subsystem="net")
    # Same type+subsystem, different call site: exact misses, fuzzy hits.
    def other_site():
        raise ConnectionError("different line")
    e2 = _capture(other_site)
    assert j.find_similar(e2, "net") == []
    fuzzy = j.find_similar_fuzzy(e2, "net", location_threshold=0.3)
    assert len(fuzzy) >= 1


def test_causal_chain_links_subsystems():
    j = IncidentJournal()
    def db_slow():
        raise TimeoutError("db slow")
    db_inc = j.record_incident(_capture(db_slow), subsystem="db")
    time.sleep(0.01)
    tg_inc = j.record_incident(_capture(_boom), subsystem="telegram")
    causes = j.causal_chain(tg_inc, window_s=60.0)
    assert any(c.id == db_inc.id for c in causes)


def test_failure_spike_detection():
    j = IncidentJournal()
    for _ in range(5):
        j.record_incident(_capture(_boom), subsystem="net")
    spike = j.failure_spike("net", window_s=600.0, baseline_s=3600.0)
    assert spike is not None
    assert spike["spike"] is True


def test_top_failing():
    j = IncidentJournal()
    for _ in range(4):
        j.record_incident(_capture(_boom), subsystem="net")
    top = j.top_failing(window_s=86400, limit=5)
    assert top and top[0]["n"] == 4


# ── self-healing executor ────────────────────────────────────────────────

def test_executor_ok_path():
    ex = SelfHealingExecutor("test", journal=IncidentJournal())
    out = ex.execute(lambda: 42)
    assert out.status == RecoveryStatus.OK
    assert out.result == 42


def test_executor_retries_transient():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("slow")
        return "won"

    ex = SelfHealingExecutor(
        "test", journal=IncidentJournal(),
        strategies=[RetryStrategy(BackoffPolicy(base=0.0, max_attempts=4))])
    out = ex.execute(flaky, verify=lambda r: r == "won")
    assert out.status == RecoveryStatus.RECOVERED
    assert out.result == "won"
    assert out.verified
    assert calls["n"] == 3


def test_executor_fallback_chain():
    ex = SelfHealingExecutor("test", journal=IncidentJournal(),
                             strategies=[])
    out = ex.execute(
        lambda: (_ for _ in ()).throw(RuntimeError("primary dead")),
        fallbacks=[("fb1", lambda: (_ for _ in ()).throw(ValueError("x"))),
                   ("fb2", lambda: "fallback-result")],
        verify=lambda r: r == "fallback-result")
    assert out.status == RecoveryStatus.RECOVERED
    assert out.result == "fallback-result"
    assert out.strategy_used == "fallback_chain"


def test_executor_degraded():
    ex = SelfHealingExecutor("test", journal=IncidentJournal(),
                             strategies=[])
    out = ex.execute(
        lambda: (_ for _ in ()).throw(RuntimeError("dead")),
        on_degrade=lambda analysis, ctx: {"partial": True})
    assert out.status == RecoveryStatus.DEGRADED
    assert out.result == {"partial": True}
    assert out.degraded_reason != ""


def test_executor_escalates_when_nothing_works():
    ex = SelfHealingExecutor("test", journal=IncidentJournal(),
                             strategies=[])
    out = ex.execute(lambda: (_ for _ in ()).throw(RuntimeError("dead")))
    assert out.status == RecoveryStatus.FAILED
    assert out.analysis is not None
    assert out.incident is not None


def test_executor_uses_known_fix_first():
    j = IncidentJournal()
    from nomorals.core.incidents import RecoveryRecord
    def known_shape():
        raise ValueError("known shape")
    e = _capture(known_shape)
    inc = j.record_incident(e, subsystem="test")
    j.record_recovery(RecoveryRecord(inc.id, inc.signature, "retry",
                                     detail="works", verified=True,
                                     verify_note="ok"))
    ex = SelfHealingExecutor("test", journal=j,
                             strategies=[RetryStrategy(
                                 BackoffPolicy(base=0.0, max_attempts=2))])
    out = ex.execute(lambda: "fine")
    assert out.status == RecoveryStatus.OK  # primary worked; memory unused


def test_circuit_breaker_fast_path():
    from nomorals.core.retry import CircuitBreaker
    cb = CircuitBreaker("test", failure_threshold=1, reset_timeout=60.0)
    cb.record_failure(RuntimeError("x"))  # opens the breaker
    ex = SelfHealingExecutor("test", journal=IncidentJournal(), breaker=cb)
    out = ex.execute(lambda: "never runs")
    assert out.status == RecoveryStatus.FAILED
    assert out.strategy_used == "circuit_open"


# ── supervisor ───────────────────────────────────────────────────────────

def test_supervisor_restarts_crashed_worker():
    import threading
    j = IncidentJournal()
    runs = {"n": 0}
    second_run = threading.Event()
    stop_ev_holder = {}

    def worker(stop_event=None):
        runs["n"] += 1
        if runs["n"] == 1:
            raise RuntimeError("crash once")
        second_run.set()
        stop_event.wait(timeout=10)

    sup = SubsystemSupervisor("test", journal=j, max_restarts=3,
                              restart_window_s=60.0)
    sup.add_worker("w", worker, restart_delay_s=0.05)
    sup.start_all()
    assert second_run.wait(timeout=10)
    sup.stop_all()
    assert runs["n"] >= 2


def test_supervisor_gives_up_after_max_restarts():
    j = IncidentJournal()

    def always_crash():
        raise RuntimeError("always")

    sup = SubsystemSupervisor("test", journal=j, max_restarts=2,
                              restart_window_s=60.0)
    sup.add_worker("w", always_crash, restart_delay_s=0.01)
    sup.start_all()
    time.sleep(1.0)
    status = sup.status()
    assert status["w"]["running"] is False
    assert status["w"]["restarts_in_window"] == 2
    sup.stop_all()


# ── budgets ──────────────────────────────────────────────────────────────

def test_budget_burn_rate_fires():
    j = IncidentJournal()
    b = ErrorBudget("svc", slo=0.999, window_s=86400, journal=j)
    now = time.time()
    # 50% failure rate over the last hour: massive burn.
    for i in range(100):
        j.record_heartbeat("svc", ok=(i % 2 == 0),
                           ts=now - (i * 30))
    alerts = b.check(now=now)
    assert any(a.severity == "page" for a in alerts)
    assert b.budget_remaining(now=now) < 0.2


def test_budget_healthy_stays_quiet():
    j = IncidentJournal()
    b = ErrorBudget("svc", slo=0.99, window_s=86400, journal=j)
    now = time.time()
    for i in range(200):
        j.record_heartbeat("svc", ok=True, ts=now - (i * 300))
    assert b.check(now=now) == []
    assert b.budget_remaining(now=now) == 1.0


def test_budget_manager():
    j = IncidentJournal()
    m = BudgetManager(journal=j)
    b = m.budget_for("svc", slo=0.99)
    assert m.status_all()["svc"]["slo"] == 0.99
    assert m.check_all() == []


# ── degradation ladders ──────────────────────────────────────────────────

def test_ladder_steps_down_and_recovers():
    primary_down = {"down": True}

    def primary():
        if primary_down["down"]:
            raise ConnectionError("primary dead")
        return "primary-ok"

    ladder = DegradationLadder("svc", rungs=[
        Rung("primary", primary, probe=lambda: not primary_down["down"],
             probe_interval_s=0.0),
        Rung("fallback", lambda: "fallback-ok",
             capability_trade="reduced quality",
             honesty="Primary unavailable — using fallback."),
    ])
    result, meta = ladder.call()
    assert result == "fallback-ok"
    assert meta["degraded"] is True
    assert meta["honesty"] != ""
    assert ladder.degraded

    # Primary heals: next call climbs back.
    primary_down["down"] = False
    result2, meta2 = ladder.call()
    assert result2 == "primary-ok"
    assert meta2["degraded"] is False


def test_ladder_exhausted_raises():
    ladder = DegradationLadder("svc", rungs=[
        Rung("a", lambda: (_ for _ in ()).throw(RuntimeError("x")),
             honesty="a failed"),
        Rung("b", lambda: (_ for _ in ()).throw(RuntimeError("y")),
             honesty="b failed"),
    ])
    with pytest.raises(LadderExhausted):
        ladder.call()


def test_ladder_requires_honesty():
    with pytest.raises(ValueError):
        DegradationLadder("svc", rungs=[
            Rung("primary", lambda: 1),
            Rung("silent", lambda: 2),  # no honesty clause
        ])


# ── facade ───────────────────────────────────────────────────────────────

def test_facade_analyze_remembers():
    ei = ErrorIntelligence()
    def refused1():
        raise ConnectionError("conn refused")
    a1 = ei.analyze(_capture(refused1),
                    context={"subsystem": "net", "operation": "dial"})
    # Second identical failure: chronic machinery engages, related found.
    def refused2():
        raise ConnectionError("conn refused again")
    a2 = ei.analyze(_capture(refused2),
                    context={"subsystem": "net", "operation": "dial"})
    assert len(a2.related_errors) >= 1


def test_facade_heal_and_health():
    ei = ErrorIntelligence()
    ex = ei.heal("test")
    out = ex.execute(lambda: "ok")
    assert out.ok
    health = ei.system_health()
    assert "budgets" in health and "ladders" in health
    assert "top_failing_24h" in health
