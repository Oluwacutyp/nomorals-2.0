"""Cognition area (six): thinking, calibration, ledger extension,
failure loop, action registry, selfheal facade.

Covers the invariants from the build brief:
- ledger records every autonomous action class
- disapproval (-1) weighs more than approval (+1)
- unknown outcomes never inflate a score
- calibration improves with data
- failure loop never proposes without sandbox / never auto-applies
- both selfheal modules still importable (and the facade unifies them)
"""

import pytest

from nomorals.cognition.action_registry import (
    ACTION_CLASSES,
    ACTION_CLASS_TO_LEDGER_TYPE,
    normalize_action_class,
    record_all,
    record_all_classes,
)
from nomorals.cognition.calibration import CalibrationStore
from nomorals.cognition.failure_loop import (
    FailureLoop,
    apply_proposal,
    build_proposal,
    cluster_failures,
    propose_fix,
    sandbox_check,
)
from nomorals.cognition.failure_kb import FailureKB
from nomorals.cognition.ledger_ext import (
    ExtendedRepresentationLedger,
    normalize_stakes,
)
from nomorals.cognition.representation import RepresentationLedger
from nomorals.cognition.thinking import (
    ThinkingStore,
    make_plan,
    record_act,
    reflect_on,
)
from nomorals.cognition.trajectories import TrajectoryStore


# ── thinking: plan -> act -> reflect -> remember ─────────────────────

def test_thinking_plan_act_reflect_remember_flow():
    store = ThinkingStore()
    plan = store.new_plan(
        "Draft and send a status update",
        ["Draft update", "Send update"],
        task_kind="comms", capability="message", model_id="model-a",
        predicted_confidence=0.8,
    )
    assert plan.status == "planned"
    store.act(plan.id, 0, "message", True, outcome="drafted")
    store.act(plan.id, 1, "send", False, outcome="failed", error="TimeoutError: gateway")
    reflection = store.reflect(plan.id)
    assert reflection.worked == ["Draft update"]
    assert reflection.failed == ["Send update"]
    assert reflection.success_rate == pytest.approx(0.5)
    assert reflection.lessons and reflection.next_steps
    result = store.remember(plan.id)
    assert result["trajectories_written"] == 2
    # One success + one failure -> "done", not "failed".
    assert store.plan(plan.id).status == "done"
    # Trajectories actually landed in the store (1 failure cluster).
    clusters = store.trajectory_store.failure_clusters("comms")
    assert len(clusters) == 1
    assert "TimeoutError" in clusters[0]["error_signature"]


def test_thinking_pure_functions_reject_bad_input():
    plan = make_plan("G", ["only step"])
    with pytest.raises(ValueError):
        make_plan("", ["x"])
    with pytest.raises(ValueError):
        make_plan("G", [])
    with pytest.raises(ValueError):
        record_act(plan, 5, "send", True)  # step_index out of range
    empty = reflect_on(plan, [])
    assert empty.success_rate == pytest.approx(0.5)
    assert empty.total == 0


def test_thinking_remember_all_failed_marks_failed():
    store = ThinkingStore()
    plan = store.new_plan("Doomed", ["Step one"], task_kind="t", capability="c")
    store.act(plan.id, 0, "send", False, error="boom")
    store.remember(plan.id)
    assert store.plan(plan.id).status == "failed"


# ── calibration ───────────────────────────────────────────────────────

def test_calibration_no_data_returns_prior():
    cal = CalibrationStore()
    assert cal.calibrated_confidence("task", "model") == pytest.approx(0.5)
    assert cal.brier_score("task", "model") is None
    assert "No calibration data" in cal.summary()


def test_calibration_overconfident_model_gets_corrected():
    cal = CalibrationStore()
    # Model says 0.9 every time, right exactly half the time.
    for i in range(20):
        cal.record("summarize", "model-x", 0.9, success=(i % 2 == 0))
    assert cal.empirical_rate("summarize", "model-x") == pytest.approx(0.5)
    assert cal.calibration_gap("summarize", "model-x") == pytest.approx(0.4)
    corrected = cal.calibrated_confidence("summarize", "model-x", 0.9)
    assert corrected < 0.9
    assert corrected == pytest.approx(0.58, abs=0.02)


def test_calibration_improves_with_data():
    cal = CalibrationStore()
    for i in range(20):
        cal.record("summarize", "model-x", 0.9, success=(i % 2 == 0))
    raw = cal.brier_score("summarize", "model-x")
    calibrated = cal.calibrated_brier("summarize", "model-x")
    assert raw is not None and calibrated is not None
    assert calibrated < raw  # correction helps on overconfident data
    # With no data at all, calibrated confidence stays the prior.
    fresh = CalibrationStore()
    assert fresh.calibrated_confidence("t", "m", 0.9) == pytest.approx(0.9)
    # Sparse data barely moves the prediction (shrinkage).
    sparse = CalibrationStore()
    sparse.record("t", "m", 0.9, success=False)
    assert sparse.calibrated_confidence("t", "m", 0.9) >= 0.75  # 1 obs barely moves it
    # Summary feeds the owner real numbers.
    assert "overconfident" in cal.summary()
    assert "summarize" in cal.summary("summarize", "model-x")


def test_calibration_rejects_bad_confidence():
    cal = CalibrationStore()
    with pytest.raises(ValueError):
        cal.record("t", "m", 1.5, True)


# ── ledger extension ──────────────────────────────────────────────────

def test_ledger_records_every_autonomous_action_class():
    ext = ExtendedRepresentationLedger()
    recs = record_all_classes(ext)
    assert len(recs) == len(ACTION_CLASSES)
    assert all(getattr(r, "id", "") for r in recs)
    # Every class is stored with its extended name in metadata.
    seen = {r.metadata["action_class"] for r in recs}
    assert seen == set(ACTION_CLASSES)
    # And every class maps onto a valid base ledger type.
    assert set(ACTION_CLASS_TO_LEDGER_TYPE) == set(ACTION_CLASSES)
    assert set(ACTION_CLASS_TO_LEDGER_TYPE.values()) <= {
        "negotiate", "purchase", "apply", "message", "post"}


def test_ledger_record_all_batch_and_unknown_class():
    ext = ExtendedRepresentationLedger()
    recs = record_all(ext, [
        {"action_class": "negotiate", "outcome": "success",
         "owner_feedback": 1, "counterparty": "Acme",
         "description": "Negotiated rate"},
        {"action_class": "buy", "outcome": "success",
         "stakes": "high", "counterparty": "Acme"},
    ])
    assert len(recs) == 2
    assert recs[0].metadata["counterparty"] == "Acme"
    assert recs[1].metadata["stakes"] == "high"
    with pytest.raises(ValueError):
        record_all(ext, [{"action_class": "teleport"}])


def test_ledger_disapproval_weights_more_than_approval():
    # One success + approval vs one success + disapproval, same stakes.
    approved = ExtendedRepresentationLedger()
    approved.record_autonomous_action(
        "post", outcome="success", owner_feedback=1)
    disapproved = ExtendedRepresentationLedger()
    disapproved.record_autonomous_action(
        "post", outcome="success", owner_feedback=-1)
    a = approved.stakes_weighted_quality_score("post")
    d = disapproved.stakes_weighted_quality_score("post")
    assert d < a
    # Disapproval drags a success below the 0.5 prior; approval keeps it at 1.0.
    assert d == pytest.approx(0.5)
    assert a == pytest.approx(1.0)
    # Distance from the prior is larger for disapproval on a partial:
    # partial+approval = 0.75 (+0.25), partial+disapproval = 0.0 (-0.5).
    p_approved = ExtendedRepresentationLedger()
    p_approved.record_autonomous_action("post", outcome="partial", owner_feedback=1)
    p_disapproved = ExtendedRepresentationLedger()
    p_disapproved.record_autonomous_action("post", outcome="partial", owner_feedback=-1)
    assert p_approved.stakes_weighted_quality_score("post") == pytest.approx(0.75)
    assert p_disapproved.stakes_weighted_quality_score("post") == pytest.approx(0.0)


def test_ledger_unknown_does_not_inflate():
    ext = ExtendedRepresentationLedger()
    ext.record_autonomous_action("post", outcome="unknown")
    ext.record_autonomous_action("post", outcome="unknown")
    # No known data -> prior, not an inflated score.
    assert ext.stakes_weighted_quality_score("post") == pytest.approx(0.5)
    assert ext.ledger.quality_score("post") == pytest.approx(0.5)
    # One unknown + one genuine failure -> score reflects the failure only.
    ext.record_autonomous_action("post", outcome="failed")
    assert ext.stakes_weighted_quality_score("post") == pytest.approx(0.0)


def test_ledger_stakes_weighting_and_counterparty_memory():
    ext = ExtendedRepresentationLedger()
    # Low-stakes success + high-stakes failure: stakes-weighted score
    # must be below the unweighted mean (0.5) because the failure weighs more.
    ext.record_autonomous_action("buy", stakes="low", outcome="success",
                                 counterparty="Acme")
    ext.record_autonomous_action("buy", stakes="critical", outcome="failed",
                                 counterparty="Acme")
    weighted = ext.stakes_weighted_quality_score("buy")
    assert weighted < 0.5
    assert weighted == pytest.approx(0.25 / 1.25)  # (0.25*1 + 1.0*0)/1.25
    cp = ext.counterparty_summary("Acme")
    assert cp["actions"] == 2
    assert cp["by_action_class"] == {"buy": 2}
    assert ext.counterparties() == ["Acme"]
    # Counterparty memory survives a re-wrap of the same ledger.
    rewrapped = ExtendedRepresentationLedger(ledger=ext.ledger)
    assert rewrapped.counterparty_summary("Acme")["actions"] == 2
    assert normalize_stakes("high") == ("high", 0.75)
    assert normalize_stakes(0.9)[0] == "critical"
    with pytest.raises(ValueError):
        normalize_stakes("astronomical")


def test_ledger_prediction_is_not_permission():
    ext = ExtendedRepresentationLedger()
    ext.record_autonomous_action("post", outcome="failed", owner_feedback=-1)
    prediction = ext.predict_outcome("post", stakes="critical", counterparty="Nobody")
    assert prediction["permission_required"] is False
    assert prediction["grants_permission"] is False
    assert prediction["denies_permission"] is False
    # Recording is never blocked by a low prediction.
    rec = ext.record_autonomous_action(
        "post", stakes="critical", outcome="success", counterparty="Nobody",
        predicted_confidence=prediction["predicted_quality"])
    assert rec.id
    # Summary includes stakes + counterparty sections.
    summary = ext.summary()
    assert "Stakes-weighted" in summary
    assert "Counterparty memory" in summary


def test_ledger_ext_wraps_base_ledger_without_duplicating():
    base = RepresentationLedger(":memory:")
    ext = ExtendedRepresentationLedger(ledger=base)
    assert ext.ledger is base
    # Base-ledger API forwards through __getattr__.
    ext.record_autonomous_action("send", outcome="success")
    assert base.quality_score("message") == pytest.approx(1.0)
    assert ext.quality_score("message") == pytest.approx(1.0)


# ── failure loop ──────────────────────────────────────────────────────

def _clustered_loop() -> FailureLoop:
    loop = FailureLoop()
    for _ in range(3):
        loop.store.record(task_kind="fetch", capability="web", model_id="m1",
                          success=False, error="TimeoutError: gateway timed out")
    return loop


def test_failure_loop_cluster_fix_sandbox_proposal():
    loop = _clustered_loop()
    clusters = loop.cluster_failures()
    assert len(clusters) == 1
    assert clusters[0]["count"] == 3
    proposal = loop.build_proposal(clusters[0])
    assert proposal["applied"] is False
    assert proposal["sandbox_required"] is True
    assert proposal["requires_owner_approval"] is True
    assert proposal["fix_kind"] == "retry_with_backoff"
    assert proposal["suggested_fix"]
    # Module-level helpers agree.
    assert cluster_failures(loop.kb) == loop.cluster_failures()
    assert propose_fix(clusters[0])["applied"] is False


def test_failure_loop_never_proposes_without_sandbox():
    loop = _clustered_loop()
    cluster = loop.cluster_failures()[0]
    proposal = loop.build_proposal(cluster)
    # Approval alone, without any sandbox pass, must NOT apply.
    refused = loop.apply_proposal(proposal, approve=True)
    assert refused["applied"] is False
    assert refused["status"] == "sandbox_required"
    # No approval at all must NOT apply, even with a sandbox pass.
    sandbox = loop.sandbox_check(proposal, check_fn=lambda _p: True)
    assert sandbox["passed"] is True
    passed_proposal = loop.build_proposal(cluster, sandbox_result=sandbox)
    assert passed_proposal["sandbox_passed"] is True
    assert passed_proposal["applied"] is False  # build never applies
    unapproved = loop.apply_proposal(passed_proposal, approve=False)
    assert unapproved["applied"] is False
    assert unapproved["status"] == "awaiting_approval"
    # Approval + sandbox pass is the ONLY path to applied.
    applied = loop.apply_proposal(passed_proposal, approve=True)
    assert applied["applied"] is True
    assert applied["status"] == "applied"
    # The input proposal is never mutated.
    assert passed_proposal["applied"] is False


def test_failure_loop_sandbox_failure_blocks_apply():
    loop = _clustered_loop()
    cluster = loop.cluster_failures()[0]
    proposal = loop.build_proposal(cluster)
    # A failing custom check blocks even approved application.
    bad = loop.sandbox_check(proposal, check_fn=lambda _p: False)
    assert bad["passed"] is False
    blocked = loop.apply_proposal(proposal, approve=True, sandbox_result=bad)
    assert blocked["applied"] is False
    # A dangerous fix text fails the static scan.
    evil = dict(proposal, suggested_fix="please run rm -rf / to fix it")
    assert sandbox_check(evil)["passed"] is False
    # Uncompilable code text fails the sandbox.
    assert sandbox_check(proposal, code_text="def broken(:")["passed"] is False
    # Compilable code text passes (it is never executed).
    assert sandbox_check(proposal, code_text="x = 1\n")["passed"] is True
    # A check_fn that raises fails closed.
    def _boom(_p):
        raise RuntimeError("kaboom")
    assert sandbox_check(proposal, check_fn=_boom)["passed"] is False


def test_failure_loop_report_and_unknown_cluster():
    loop = FailureLoop()
    assert "nothing to propose" in loop.report().lower()
    with pytest.raises(ValueError):
        propose_fix({})
    assert cluster_failures(None) == []


# ── action registry ───────────────────────────────────────────────────

def test_action_registry_normalization_and_aliases():
    assert normalize_action_class("  BUY ") == "buy"
    assert normalize_action_class("dm") == "message"
    assert normalize_action_class("email") == "message"
    with pytest.raises(ValueError):
        normalize_action_class("teleport")
    # The brief's example classes are all present.
    for cls in ("negotiate", "buy", "apply", "post", "send",
                "schedule", "publish", "trade"):
        assert cls in ACTION_CLASSES


# ── selfheal facade: both originals untouched, both importable ───────

def test_selfheal_both_modules_still_importable():
    import nomorals.core.self_heal as self_heal
    import nomorals.core.selfheal as selfheal

    assert selfheal.SelfHealingExecutor is not None
    assert selfheal.RecoveryOutcome is not None
    assert self_heal.heal_one is not None
    assert self_heal.FIX_STRATEGIES is not None


def test_selfheal_facade_reexports_both():
    from nomorals.core import selfheal_facade as facade

    desc = facade.describe()
    assert desc["overlap"] == []
    assert "SelfHealingExecutor" in desc["selfheal"]
    assert "heal_one" in desc["self_heal"]
    assert facade.SelfHealingExecutor is not None
    assert facade.heal_one is not None
    assert facade.FIX_STRATEGIES is not None
    assert "heal_one" in facade.healing_names()
    assert "SelfHealingExecutor" in facade.healing_names()
    # Facade exposes the original modules themselves.
    assert facade.selfheal_module.__name__.endswith("selfheal")
    assert facade.self_heal_module.__name__.endswith("self_heal")
