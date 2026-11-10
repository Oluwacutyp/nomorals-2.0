"""Doc 24 PG2 — HonestyLedger promotion enforcement.

PG1 (core/proof_grade.py) built the grade scale; PG2 makes the
impossible ledger USE it. Before this pass, ``record`` /
``record_attempt(new_status=...)`` stored any status string a caller
passed — a claim could be relabelled ``measured`` by fiat and every
reader believed the label. These pins cover:

* the closed vocabulary (out-of-vocabulary statuses/evidence kinds
  are refused at write, nothing stored);
* evidence-gated promotion (a status is recorded only as high as
  the attached evidence earns; the request survives as a CLAIM on
  the record's audit trail — requested / earned / applied / grade);
* intermediate earning (evidence below the requested floor earns
  the highest rung it supports, not a binary pass/fail);
* readers surfacing the EARNED status (list filter, core.status()),
  including over legacy rows written before enforcement existed.

Offline; every ledger lives under tmp_path.
"""

from __future__ import annotations

import json

import pytest

from nomorals.core.proof_grade import (
    EVIDENCE_KIND_GRADES,
    EvidenceRef,
    grade_for_impossible_status,
)
from nomorals.impossible import ImpossibleMadePossible
from nomorals.impossible.charter import STATUSES
from nomorals.impossible.ledger import (
    _KIND_GRADES,
    _STATUS_FLOORS,
    HonestyLedger,
    LedgerStatusError,
    earned_status_of,
)

FALSIFIER = {"falsifier": "Ice still melts before hour 48 in a 35C room."}


@pytest.fixture()
def ledger(tmp_path):
    return HonestyLedger(tmp_path / "ledger.json")


def _born(ledger, status="hypothesis", **kwargs):
    kwargs.setdefault("hypothesis", dict(FALSIFIER))
    return ledger.record("a cooler that keeps drinks cold for 48 hours",
                         status=status, **kwargs)


# ── the mirrors cannot drift from PG1's real tables ────────────────


class TestMirrors:
    def test_kind_grades_mirror_proof_grade(self):
        expected = {kind: int(grade)
                    for kind, grade in EVIDENCE_KIND_GRADES.items()}
        assert expected == _KIND_GRADES

    def test_status_floors_mirror_proof_grade(self):
        expected = {status: int(grade_for_impossible_status(status))
                    for status in STATUSES}
        assert expected == _STATUS_FLOORS

    def test_lifecycle_is_the_charter_vocabulary(self):
        assert STATUSES == ("idea", "hypothesis", "tested",
                            "prototype", "built", "measured")


# ── the closed vocabulary refuses at write ─────────────────────────


class TestClosedVocabulary:
    def test_record_refuses_out_of_vocabulary_status(self, ledger):
        with pytest.raises(LedgerStatusError, match="not a claim status"):
            ledger.record("anything", status="proven")
        assert ledger.list() == []

    def test_record_attempt_refuses_and_stores_nothing(self, ledger):
        entry = _born(ledger)
        with pytest.raises(LedgerStatusError, match="closed lifecycle"):
            ledger.record_attempt(entry["id"], outcome="ran",
                                  evidence="bench log", lesson="n/a",
                                  new_status="verified")
        after = ledger.get(entry["id"])
        assert after["attempts"] == []
        assert after["status"] == "hypothesis"

    def test_unknown_evidence_kind_refused(self, ledger):
        entry = _born(ledger)
        with pytest.raises(LedgerStatusError, match="unknown evidence kind"):
            ledger.record_attempt(
                entry["id"], outcome="ran", evidence="x", lesson="y",
                new_status="tested",
                evidence_refs=[{"kind": "vibes", "ref": "chat:1"}])
        assert ledger.get(entry["id"])["attempts"] == []

    def test_empty_evidence_ref_refused(self, ledger):
        with pytest.raises(LedgerStatusError, match="must name where"):
            ledger.record("anything", status="built",
                          evidence_refs=[{"kind": "artifact_pass",
                                          "ref": "  "}])


# ── the gate: requested vs earned ──────────────────────────────────


class TestEvidenceGate:
    def test_born_measured_without_evidence_earns_hypothesis(self, ledger):
        entry = _born(ledger, status="measured")
        assert entry["status"] == "hypothesis"
        assert entry["requested_status"] == "measured"
        (audit,) = entry["status_audit"]
        assert audit["action"] == "record"
        assert audit["requested"] == "measured"
        assert audit["earned"] == "hypothesis"
        assert audit["applied"] == "hypothesis"
        # The G1 comes from the record's own formalization — and the
        # intrinsic ref is ON the audit, not an invisible exception.
        assert audit["grade"] == "G1"
        assert audit["evidence"] == [{
            "kind": "logical_check",
            "ref": f"claim:{entry['id']}:hypothesis"}]

    def test_born_without_formalization_earns_idea(self, ledger):
        entry = ledger.record("a passing thought")
        assert entry["status"] == "idea"
        assert entry["requested_status"] == "hypothesis"
        assert entry["status_audit"][0]["grade"] == "G0"

    def test_born_built_with_artifact_evidence_is_not_demoted(self, ledger):
        entry = _born(ledger, status="built", evidence_refs=[
            {"kind": "artifact_pass", "ref": "artifact:cooler-v1"}])
        assert entry["status"] == "built"
        assert entry["status_audit"][0]["earned"] == "built"
        assert entry["status_audit"][0]["grade"] == "G5"

    def test_promotion_without_evidence_stays_put_and_is_audited(
            self, ledger):
        entry = _born(ledger)
        out = ledger.record_attempt(entry["id"], outcome="attempted",
                                    evidence="wrote a spec", lesson="n/a",
                                    new_status="built")
        assert out["status"] == "hypothesis"
        assert out["requested_status"] == "built"
        audit = out["status_audit"][-1]
        assert audit["action"] == "attempt"
        assert audit["requested"] == "built"
        assert audit["earned"] == "idea"  # no refs -> G0
        assert audit["applied"] == "hypothesis"  # never demoted either
        assert audit["grade"] == "G0"
        attempt = out["attempts"][-1]
        assert attempt["requested_status"] == "built"
        assert attempt["applied_status"] == "hypothesis"

    def test_promotion_with_qualifying_evidence_applies(self, ledger):
        entry = _born(ledger)
        out = ledger.record_attempt(
            entry["id"], outcome="ran the killing experiment",
            evidence="coupon test rig", lesson="held 40h",
            new_status="tested",
            evidence_refs=[EvidenceRef("simulation_check", "exp:rig-1")])
        assert out["status"] == "tested"
        assert out["status_audit"][-1]["grade"] == "G3"
        # The real EvidenceRef was normalized onto the attempt record.
        assert out["attempts"][-1]["evidence_refs"] == [
            {"kind": "simulation_check", "ref": "exp:rig-1"}]

    def test_intermediate_earning_is_not_binary(self, ledger):
        entry = _born(ledger)
        out = ledger.record_attempt(
            entry["id"], outcome="simulated", evidence="model output",
            lesson="rungs are real", new_status="built",
            evidence_refs=[{"kind": "simulation_check", "ref": "sim:7"}])
        # Asked for built (G5 floor); a simulation check (G3) earns
        # tested — the highest rung its evidence supports.
        assert out["status"] == "tested"
        audit = out["status_audit"][-1]
        assert audit["requested"] == "built"
        assert audit["earned"] == "tested"
        assert audit["applied"] == "tested"

    def test_measured_needs_monitoring_grade_evidence(self, ledger):
        entry = _born(ledger, status="built", evidence_refs=[
            {"kind": "artifact_pass", "ref": "artifact:cooler-v1"}])
        out = ledger.record_attempt(
            entry["id"], outcome="deployed", evidence="ran a week",
            lesson="n/a", new_status="measured",
            evidence_refs=[{"kind": "artifact_pass", "ref": "artifact:v1"}])
        # G5 evidence against a G6 floor: stays built, audit shows why.
        assert out["status"] == "built"
        assert out["status_audit"][-1]["earned"] == "built"
        out = ledger.record_attempt(
            entry["id"], outcome="monitored", evidence="receipts",
            lesson="n/a", new_status="measured",
            evidence_refs=[{"kind": "monitoring_receipt",
                            "ref": "receipt:week-1"}])
        assert out["status"] == "measured"
        assert out["status_audit"][-1]["grade"] == "G6"

    def test_explicit_step_down_applies(self, ledger):
        entry = _born(ledger)
        ledger.record_attempt(
            entry["id"], outcome="ran", evidence="rig", lesson="ok",
            new_status="tested",
            evidence_refs=[{"kind": "simulation_check", "ref": "exp:1"}])
        out = ledger.record_attempt(entry["id"], outcome="regression",
                                    evidence="rig re-run", lesson="redo",
                                    new_status="hypothesis")
        assert out["status"] == "hypothesis"
        assert out["status_audit"][-1]["applied"] == "hypothesis"

    def test_plain_attempt_without_new_status_is_unchanged(self, ledger):
        entry = _born(ledger)
        out = ledger.record_attempt(entry["id"], outcome="failed",
                                    evidence="coupon test",
                                    lesson="Bitumen softens at 60C")
        assert out["status"] == "hypothesis"
        assert len(out["status_audit"]) == 1  # only the birth event
        assert out["attempts"][-1]["evidence_refs"] == []
        assert "requested_status" not in out["attempts"][-1]


# ── readers surface the earned status ──────────────────────────────


def _write_raw_ledger(path, claims):
    path.write_text(json.dumps({"claims": claims}), encoding="utf-8")


def _legacy_entry(**overrides):
    entry = {
        "id": "legacy000001", "claim": "an old claim",
        "formulation": {}, "classification": {}, "prior_art_summary": "",
        "transfer_methods": [], "gate": {"verdict": None, "why": None,
                                         "failed": []},
        "hypothesis": {}, "reframed": {}, "status": "measured",
        "falsifier": "", "attempts": [], "conflicts": [],
        "created_at": 0.0, "updated_at": 0.0,
        "checked_by": "impossible-gate", "what_would_change_my_mind": "",
    }
    entry.update(overrides)
    return entry


class TestReaders:
    def test_list_filters_on_earned_status(self, ledger):
        demoted = _born(ledger, status="measured")
        honest = _born(ledger, status="built", evidence_refs=[
            {"kind": "artifact_pass", "ref": "artifact:x"}])
        assert ledger.list(status="measured") == []
        assert {c["id"] for c in ledger.list(status="built")} == {
            honest["id"]}
        assert {c["id"] for c in ledger.list(status="hypothesis")} == {
            demoted["id"]}

    def test_legacy_out_of_vocabulary_row_reads_unevidenced(self, tmp_path):
        path = tmp_path / "ledger.json"
        _write_raw_ledger(path, [_legacy_entry(status="proven!!!")])
        ledger = HonestyLedger(path)
        entry = ledger.get("legacy000001")
        assert entry is not None  # reads without crashing
        assert earned_status_of(entry) == "idea"  # grades unevidenced
        assert ledger.list(status="idea")[0]["id"] == "legacy000001"
        assert ledger.list(status="measured") == []

    def test_legacy_fiat_measured_without_evidence_is_not_inherited(
            self, tmp_path):
        path = tmp_path / "ledger.json"
        _write_raw_ledger(path, [_legacy_entry()])
        ledger = HonestyLedger(path)
        assert earned_status_of(ledger.get("legacy000001")) == "idea"

    def test_legacy_hypothesis_with_falsifier_keeps_its_standing(
            self, tmp_path):
        path = tmp_path / "ledger.json"
        _write_raw_ledger(path, [_legacy_entry(
            status="hypothesis",
            falsifier="Ice still melts before hour 48.")])
        ledger = HonestyLedger(path)
        assert earned_status_of(ledger.get("legacy000001")) == "hypothesis"

    def test_core_status_surfaces_earned_and_names_the_claim(self, tmp_path):
        path = tmp_path / "ledger.json"
        _write_raw_ledger(path, [_legacy_entry(
            falsifier="Ice still melts before hour 48.")])
        imp = ImpossibleMadePossible(ledger_path=path,
                                     queue_path=tmp_path / "queue.json",
                                     spec_dir=tmp_path / "specs")
        out = imp.status("legacy000001")
        assert out["status"] == "hypothesis"  # earned, from the falsifier
        assert out["recorded_status"] == "measured"  # the claim, as a claim
        assert out["status_unverified"] is True

    def test_production_submit_born_at_hypothesis_with_audit(self, tmp_path):
        imp = ImpossibleMadePossible(ledger_path=tmp_path / "ledger.json",
                                     queue_path=tmp_path / "queue.json",
                                     spec_dir=tmp_path / "specs")
        dossier = imp.submit(
            "a house in Lagos that stays cool without grid power")
        entry = imp.ledger.get(dossier["claim_id"])
        assert entry["status"] == "hypothesis"
        assert entry["requested_status"] == "hypothesis"
        (audit,) = entry["status_audit"]
        assert audit["applied"] == "hypothesis"
        assert audit["grade"] == "G1"
        out = imp.status(dossier["claim_id"])
        assert out["status"] == "hypothesis"
        assert "recorded_status" not in out
