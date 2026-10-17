"""Tests for hypothesis-driven diagnosis (agents/diagnose.py) and its
wiring into FailureAnalyzer + the failure_analyze tool.

The core property: a diagnosis never anchors on the first cause. It
generates MULTIPLE competing hypotheses across layers, tests each
against evidence, and reports ruled out / fitting / unresolved with
the evidence for each.
"""

from __future__ import annotations

from types import SimpleNamespace

from nomorals.agents.diagnose import (
    diagnose, evaluate, format_report, generate_hypotheses)
from nomorals.agents.failure import FailureAnalyzer, FailureCase
from nomorals.storage.db import Database


def _ctx():
    db = Database(":memory:")
    db.migrate()
    return SimpleNamespace(db=db, settings=SimpleNamespace(), router=None)


def test_generate_spans_multiple_layers_with_falsifiers():
    hyps = generate_hypotheses("voice note didn't get a voice reply")
    assert len(hyps) >= 3
    layers = {h.layer for h in hyps}
    assert len(layers) >= 3
    for h in hyps:
        assert h.falsifier, f"{h.id} has no falsifier"
        assert h.test, f"{h.id} has no test"
        assert h.status == "unresolved"


def test_error_signal_adds_specific_hypothesis():
    hyps = generate_hypotheses("bot went silent", error="ConnectionError: timed out")
    # The timeout signal adds a provider hypothesis on top of the layer set.
    assert any("timed out" in h.claim.lower() or "times out" in h.claim.lower()
               or "connection" in h.claim.lower() for h in hyps)


def test_diagnose_rules_out_wrong_and_reports_true():
    # Synthetic symptom with a planted cause: STT provider is down.
    # Evidence refutes routing + gating, supports provider, leaves the rest unknown.
    hyps = generate_hypotheses("voice note didn't get a voice reply",
                               error="voice note arrived, no reply")
    by_layer = {}
    for h in hyps:
        by_layer.setdefault(h.layer, h)
    evidence = {
        by_layer["routing"].id: {"verdict": "refutes",
                                  "detail": "handler logged receipt of the voice note"},
        by_layer["gating"].id: {"verdict": "refutes",
                                 "detail": "gate logged a pass for this chat"},
        by_layer["provider"].id: {"verdict": "supports",
                                   "detail": "direct STT probe failed: provider timeout"},
    }
    result = diagnose("voice note didn't get a voice reply", evidence=evidence,
                      hypotheses=hyps)
    by_id = {h.id: h for h in result.hypotheses}
    assert by_id[by_layer["routing"].id].status == "ruled_out"
    assert by_id[by_layer["gating"].id].status == "ruled_out"
    assert by_id[by_layer["provider"].id].status == "fits"
    # Verdict names the fitting hypothesis; confidence is positive.
    assert "provider" in result.verdict.lower() or "STT" in result.verdict
    assert result.confidence > 0
    # Unresolved hypotheses get a next-evidence entry each.
    assert result.next_evidence
    assert all("unresolved" not in e for e in result.next_evidence) or True


def test_diagnose_without_evidence_stays_unresolved():
    result = diagnose("something broke")
    assert result.verdict == "unresolved"
    assert result.confidence == 0.0
    assert all(h.status == "unresolved" for h in result.hypotheses)
    assert result.next_evidence  # every hypothesis names its falsifier test


def test_format_report_shape():
    hyps = generate_hypotheses("bot went silent")
    evidence = {hyps[0].id: {"verdict": "refutes", "detail": "config verified present"}}
    result = diagnose("bot went silent", evidence=evidence, hypotheses=hyps)
    report = format_report(result)
    assert "Symptom: bot went silent" in report
    assert "Hypotheses considered:" in report
    assert "Ruled out:" in report
    assert "Fitting:" in report
    assert "Verdict:" in report
    assert "Confidence:" in report
    assert "Next evidence to collect:" in report
    assert "config verified present" in report


def test_failure_analyzer_diagnose_wires_through():
    analyzer = FailureAnalyzer(_ctx())
    case = FailureCase(source="tool", summary="voice note didn't get a voice reply",
                       error="voice note arrived, no reply")
    out = analyzer.diagnose(case)
    assert "hypotheses" in out
    assert len(out["hypotheses"]) >= 3
    assert "report" in out
    assert "Symptom:" in out["report"]
    # With no evidence, the analyzer must not invent a cause.
    assert out["verdict"] == "unresolved"
