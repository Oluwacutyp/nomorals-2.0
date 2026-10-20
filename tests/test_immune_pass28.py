"""Pass #28 — immune system: verified-bug regression tests.

Bugs pinned here (all reproduced on pristine HEAD 0de32d34):

1. The adaptive gate ENFORCED on a sample sitting exactly AT the p95
   of a constant baseline — i.e. on the most normal sample the baseline
   has ever seen. The gate's contract is "over BOTH p95 thresholds";
   ``>=`` arrests normal behaviour (the autoimmune failure class).
2. ``AdaptiveBaseline.record`` dropped the legacy ``growth`` key that
   ``police._exceeds_cancer`` and the gate itself both accept as an
   alias for ``process_growth`` — those samples recorded growth 0.0,
   and ``thresholds`` then called the baseline "ready" with a vacuous
   growth leg (p95 0.0), so a zero-growth sample was enforced.
3. ``BehaviourDriftMonitor.add_day`` appended unconditionally, so
   three adds of the SAME calendar date counted as 3 days and produced
   a verdict from a single day — violating "one metrics dict per day"
   and "needs 3 full days before any verdict".
4. ``/immune status`` (and ``/immune fever status``) read the fever's
   ``active`` attribute, which only ``end()`` clears — an expired
   fever reported ACTIVE forever.
"""

from __future__ import annotations

import time

import pytest

from nomorals.immune.baseline import (
    AdaptiveBaseline,
    adaptive_consult_then_enforce,
)
from nomorals.immune.chat import control_immune, get_immune_state, reset_immune_state
from nomorals.immune.drift import BehaviourDriftMonitor


def _constant_baseline(n=40, spawned=10.0, growth=1.0):
    base = AdaptiveBaseline()
    for _ in range(n):
        base.record({"processes_spawned": spawned, "process_growth": growth})
    return base


# ── bug 1: at-threshold is not over-threshold ─────────────────────

def test_gate_does_not_enforce_sample_exactly_at_p95():
    base = _constant_baseline()
    assert base.thresholds().status == "ready"
    enforced = []
    result = adaptive_consult_then_enforce(
        {"process_id": "p", "processes_spawned": 10, "process_growth": 1.0},
        base, is_system_process=True,
        enforce_fn=lambda s: enforced.append(s))
    assert result["action"] == "report_only"
    assert enforced == []


def test_gate_still_enforces_sample_strictly_over_p95():
    base = _constant_baseline()
    enforced = []
    result = adaptive_consult_then_enforce(
        {"process_id": "p", "processes_spawned": 50, "process_growth": 5},
        base, is_system_process=True,
        enforce_fn=lambda s: enforced.append(s))
    assert result["action"] == "enforce"
    assert enforced


# ── bug 2: growth alias dropped + half-degenerate baseline ────────

def test_record_accepts_legacy_growth_alias():
    base = AdaptiveBaseline()
    for _ in range(35):
        base.record({"processes_spawned": 10.0, "growth": 2.0})
    th = base.thresholds()
    assert th.status == "ready"
    assert th.growth_p95 == pytest.approx(2.0)


def test_half_degenerate_baseline_is_insufficient():
    # spawned leg healthy, growth leg has only ever seen zero: the
    # baseline cannot say what "too much growth" is -> report-only.
    base = AdaptiveBaseline()
    for _ in range(35):
        base.record({"processes_spawned": 10.0, "process_growth": 0.0})
    th = base.thresholds()
    assert th.status == "insufficient"
    assert th.growth_p95 == pytest.approx(0.0)
    enforced = []
    result = adaptive_consult_then_enforce(
        {"process_id": "p", "processes_spawned": 9999, "process_growth": 0.0},
        base, is_system_process=True,
        enforce_fn=lambda s: enforced.append(s))
    assert result["action"] == "report_only"
    assert enforced == []


# ── pins: the <30 report-only rule + no stale threshold cache ────

def test_enforcement_stays_silent_at_29_samples():
    base = _constant_baseline(n=29)
    assert base.thresholds().status == "insufficient"
    enforced = []
    result = adaptive_consult_then_enforce(
        {"process_id": "p", "processes_spawned": 99999, "process_growth": 999},
        base, is_system_process=True,
        enforce_fn=lambda s: enforced.append(s))
    assert result["action"] == "report_only"
    assert enforced == []


def test_thresholds_recomputed_from_live_samples():
    base = _constant_baseline(n=30, spawned=10.0, growth=1.0)
    first = base.thresholds()
    for _ in range(30):
        base.record({"processes_spawned": 100.0, "process_growth": 4.0})
    second = base.thresholds()
    assert second.spawned_p95 > first.spawned_p95
    assert second.growth_p95 > first.growth_p95


# ── bug 3: one metrics dict per day ───────────────────────────────

def test_drift_same_date_counts_as_one_day():
    mon = BehaviourDriftMonitor()
    day = {"tokens": 100, "processes_spawned": 2, "grants_held": 1, "errors": 0}
    mon.add_day("2026-10-20", day)
    mon.add_day("2026-10-20", {**day, "grants_held": 2})
    report = mon.add_day("2026-10-20", {**day, "grants_held": 3})
    assert report.days_considered == 1
    assert report.verdict == "insufficient"


def test_drift_same_date_replaces_metrics():
    mon = BehaviourDriftMonitor()
    base = {"tokens": 100, "processes_spawned": 2, "grants_held": 1, "errors": 0}
    mon.add_day("d1", base)
    mon.add_day("d2", base)
    # Re-adding d2 with corrected (calm) numbers must replace, not append:
    # three recorded entries but only two distinct days -> insufficient.
    report = mon.add_day("d2", {**base, "tokens": 100})
    assert report.days_considered == 2
    assert report.verdict == "insufficient"
    report = mon.add_day("d3", base)
    assert report.days_considered == 3
    assert report.verdict == "ok"


# ── bug 4: expired fever must not report ACTIVE ───────────────────

def test_fever_status_not_active_after_bound_passed():
    reset_immune_state("pass28")
    try:
        state = get_immune_state("pass28")
        state.fever.start(
            "pass28 check", 10,
            announce_fn=state.fever_announcements.append,
            now=time.time() - 100,
        )
        assert state.fever.is_active() is False  # bound long past
        out = control_immune("status", chat_key="pass28")
        assert "fever: ACTIVE" not in out
        out = control_immune("fever status", chat_key="pass28")
        assert out == "fever: off."
    finally:
        reset_immune_state("pass28")
