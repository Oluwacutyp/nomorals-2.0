"""Failure pins for nomorals/core/selfheal.py (tail-112).

NOTE: this module is core/selfheal.py — NOT core/self_heal.py
(the underscore module, pinned separately in tail-72).

One pin per census site. All sites adjudicated EXCLUSION:

- 311 executor failure path: diagnose() crash -> doctor = {}. The
  callee's own contract is "never raises" (the guard is a second
  belt), and doctor flows only into the escalation record — the
  outcome is byte-identical with diagnose working or raising
  (pinned both ways): incident recorded, FAILED outcome returned.
  The guard itself records nothing about the diagnose failure.
- 495 _fn_accepts_stop: an uninspectable callable -> False. A
  capability probe: what cannot be proven to accept a stop_event
  is treated as non-cooperative (the conservative direction); the
  worker still runs under supervision either way.
- 635 watchdog loop: an is_alive() probe that raises counts as
  death — the noqa prints it ("a broken probe is a failure").
  Fail-closed AND recorded: an incident is journalled, the death
  is error-logged, and restart budget is consumed exactly as for
  a clean death (pinned: budget exhaustion stops the watch).
"""

import logging
import time
from unittest import mock

from nomorals.core import selfheal as sh
from nomorals.core.incidents import IncidentJournal


def _boom_fn():
    raise ValueError("primary boom")


# -------------------------------------------------------------------- 311
def test_diagnose_crash_still_escalates_with_incident(monkeypatch):
    monkeypatch.setattr(sh, "diagnose",
                        mock.Mock(side_effect=RuntimeError("doctor boom")))
    records = []

    class _Cap(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Cap(level=logging.DEBUG)
    sh._log.addHandler(handler)
    try:
        executor = sh.SelfHealingExecutor(
            "probe", journal=IncidentJournal(), strategies=[])
        out = executor.execute(_boom_fn)
    finally:
        sh._log.removeHandler(handler)
    assert out.status == sh.RecoveryStatus.FAILED
    assert out.incident is not None
    # The diagnose guard itself leaves no trace (escalation logs fire,
    # but nothing reports the doctor failure).
    assert not any("diagnose" in m or "doctor" in m for m in records)


def test_diagnose_healthy_control_same_outcome_shape(monkeypatch):
    monkeypatch.setattr(sh, "diagnose", lambda exc: {"cause": "x"})
    executor = sh.SelfHealingExecutor(
        "probe", journal=IncidentJournal(), strategies=[])
    out = executor.execute(_boom_fn)
    assert out.status == sh.RecoveryStatus.FAILED
    assert out.incident is not None


def test_execute_primary_success_control():
    executor = sh.SelfHealingExecutor(
        "probe", journal=IncidentJournal(), strategies=[])
    out = executor.execute(lambda: 42)
    assert out.status == sh.RecoveryStatus.OK
    assert out.result == 42


# -------------------------------------------------------------------- 495
class _Uninspectable:
    @property
    def __signature__(self):
        raise ValueError("no signature")

    def __call__(self, stop_event=None):
        raise AssertionError("never called by the probe")


def test_fn_accepts_stop_uninspectable_is_false():
    assert sh.SubsystemSupervisor._fn_accepts_stop(_Uninspectable()) is False


def test_fn_accepts_stop_controls():
    def with_stop(stop_event=None):
        raise AssertionError("never called by the probe")

    def plain(x):
        raise AssertionError("never called by the probe")

    assert sh.SubsystemSupervisor._fn_accepts_stop(with_stop) is True
    assert sh.SubsystemSupervisor._fn_accepts_stop(plain) is False


# -------------------------------------------------------------------- 635
class _FakeJournal:
    def __init__(self):
        self.incidents = []

    def record_incident(self, exc, subsystem=None, category=None,
                        severity=None, context=None):
        self.incidents.append((subsystem, category))
        return object()


def test_watchdog_broken_probe_counts_as_death_and_gives_up():
    journal = _FakeJournal()
    restarts = []
    supervisor = sh.SubsystemSupervisor(
        "probe", journal=journal, max_restarts=2, restart_window_s=60.0)

    def bad_alive():
        raise RuntimeError("probe boom")

    supervisor.watch("w1", bad_alive, lambda: restarts.append(1),
                     check_interval_s=0.02)
    try:
        deadline = time.time() + 5
        while time.time() < deadline:
            state = supervisor.status().get("w1", {})
            if state.get("running") is False and len(restarts) >= 2:
                break
            time.sleep(0.05)
        time.sleep(0.15)  # let any in-flight iteration settle
        # Each death is recorded; the budget (2 restarts) is consumed
        # exactly as for clean deaths, then the watch gives up.
        assert len(journal.incidents) == 3
        assert all(cat == "supervisor" for _sub, cat in journal.incidents)
        assert len(restarts) == 2
        assert supervisor.status()["w1"]["running"] is False
    finally:
        supervisor.stop_all(timeout_s=2.0)


def test_watchdog_healthy_probe_control():
    journal = _FakeJournal()
    restarts = []
    supervisor = sh.SubsystemSupervisor(
        "probe", journal=journal, max_restarts=2)
    supervisor.watch("ok", lambda: True, lambda: restarts.append(1),
                     check_interval_s=0.02)
    try:
        time.sleep(0.25)
        assert journal.incidents == []
        assert restarts == []
    finally:
        supervisor.stop_all(timeout_s=2.0)
