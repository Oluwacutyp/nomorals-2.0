"""Doc 24 Pass PG1 — the G0–G6 proof-grade vocabulary in core.

These tests ARE the pass's contract: the ladder is the doc's
(definitions pinned verbatim), a grade is a derived value of the
strongest evidence attached (the no-silent-upgrade rule is a tested
refusal on every construction path, including deserialization), and
the mapping tables from the four shipped "evidence" objects are
pinned row by row — including the rows that map to nothing, with
their reasons. The mirrored source vocabularies are pinned against
the real modules' values.

Shadow only: nothing here asserts any production surface emits a
grade — PG2/PG3 do that.
"""
from __future__ import annotations

import dataclasses
import inspect
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.core import proof_grade
from nomorals.core.proof_grade import (
    DELEGATION_KIND_MAPPINGS,
    DELEGATION_KNOWN_KINDS,
    DELEGATION_UNMAPPED_REASON,
    EVIDENCE_KIND_GRADES,
    GRADE_TABLE,
    IMPOSSIBLE_STATUS_MAPPINGS,
    IMPOSSIBLE_STATUSES,
    STEWARD_RESOLUTION_EVIDENCE,
    STEWARD_UNKNOWN_VERDICT,
    STEWARD_VERDICT_MAPPINGS,
    STEWARD_VERDICTS,
    WS4_CLAIM_KINDS,
    WS4_KIND_MAPPINGS,
    ClaimGrade,
    EvidenceRef,
    ProofGrade,
    ProofGradeError,
    grade_for_delegation_kind,
    grade_for_impossible_status,
    grade_for_steward_verdict,
    grade_for_ws4_kind,
    grade_of,
)


def ev(kind: str, ref: str = "ref-1") -> EvidenceRef:
    return EvidenceRef(kind=kind, ref=ref)


class ScaleTests(unittest.TestCase):
    def test_seven_rungs_ordered(self):
        grades = [ProofGrade.G0, ProofGrade.G1, ProofGrade.G2,
                  ProofGrade.G3, ProofGrade.G4, ProofGrade.G5,
                  ProofGrade.G6]
        self.assertEqual(sorted(grades), grades)
        self.assertEqual(max(grades), ProofGrade.G6)
        self.assertEqual(min(grades), ProofGrade.G0)
        self.assertTrue(ProofGrade.G0 < ProofGrade.G1 < ProofGrade.G6)

    def test_table_covers_every_grade(self):
        self.assertEqual(set(GRADE_TABLE), set(ProofGrade))

    def test_definitions_are_the_docs_verbatim(self):
        # The doc's ladder table cells, quoted — not paraphrased.
        expected = {
            ProofGrade.G0: ("raw generative thought", "hypothesis only"),
            ProofGrade.G1: ("logical/unit/schema check",
                            "inexpensive filter"),
            ProofGrade.G2: ("symbolic/numeric derivation",
                            "quantitative claim"),
            ProofGrade.G3: ("simulation/property/adversarial check",
                            "model-dependent claim"),
            ProofGrade.G4: ("primary empirical source / "
                            "independent replication", "real-world claim"),
            ProofGrade.G5: ("built artifact/experiment/operation passes",
                            "practical proof"),
            ProofGrade.G6: ("ongoing monitoring/receipt", "durable proof"),
        }
        for grade, (evidence, use) in expected.items():
            self.assertEqual(GRADE_TABLE[grade].evidence, evidence)
            self.assertEqual(GRADE_TABLE[grade].use, use)

    def test_earning_kinds_round_trip(self):
        # Each rung's earning kind is the kind that evidences it.
        for grade, info in GRADE_TABLE.items():
            self.assertEqual(EVIDENCE_KIND_GRADES[info.earning_kind], grade)
        self.assertEqual(set(EVIDENCE_KIND_GRADES.values()), set(ProofGrade))

    def test_parse(self):
        self.assertIs(ProofGrade.parse(ProofGrade.G3), ProofGrade.G3)
        self.assertIs(ProofGrade.parse(3), ProofGrade.G3)
        self.assertIs(ProofGrade.parse("G3"), ProofGrade.G3)
        self.assertIs(ProofGrade.parse("g0"), ProofGrade.G0)
        for junk in ("G7", "high", "", 7, -1, None, True, 2.5):
            with self.assertRaises(ProofGradeError, msg=repr(junk)):
                ProofGrade.parse(junk)


class GradeOfTests(unittest.TestCase):
    def test_no_evidence_is_g0(self):
        self.assertEqual(grade_of([]), ProofGrade.G0)
        self.assertEqual(grade_of(()), ProofGrade.G0)

    def test_each_kind_earns_exactly_its_grade(self):
        for kind, grade in EVIDENCE_KIND_GRADES.items():
            self.assertEqual(grade_of([ev(kind)]), grade, kind)

    def test_mixed_evidence_takes_the_strongest(self):
        items = [ev("derivation"), ev("logical_check"),
                 ev("artifact_pass", "art-9"), ev("derivation", "calc-2")]
        self.assertEqual(grade_of(items), ProofGrade.G5)
        self.assertEqual(grade_of(list(reversed(items))), ProofGrade.G5)

    def test_generative_provenance_alone_is_still_g0(self):
        self.assertEqual(grade_of([ev("generative_thought")]),
                         ProofGrade.G0)

    def test_bare_shapes_are_not_evidence(self):
        for junk in ("derivation", {"kind": "derivation", "ref": "x"}, 5):
            with self.assertRaises(ProofGradeError, msg=repr(junk)):
                grade_of([junk])


class EvidenceRefTests(unittest.TestCase):
    def test_unknown_kind_refused(self):
        with self.assertRaises(ProofGradeError):
            ev("vibes")

    def test_empty_ref_refused(self):
        for bad in ("", "   "):
            with self.assertRaises(ProofGradeError):
                ev("derivation", bad)

    def test_dict_round_trip(self):
        item = ev("empirical_source", "case-receipt-42")
        self.assertEqual(EvidenceRef.from_dict(item.to_dict()), item)

    def test_from_dict_refuses_malformed(self):
        for bad in (None, "x", {"kind": "derivation"},
                    {"kind": "nope", "ref": "x"},
                    {"kind": "derivation", "ref": ""}):
            with self.assertRaises(ProofGradeError, msg=repr(bad)):
                EvidenceRef.from_dict(bad)


class ClaimGradeTests(unittest.TestCase):
    def test_grade_is_derived_not_a_field(self):
        # The fiat path must not exist: the record stores evidence only.
        self.assertEqual([f.name for f in dataclasses.fields(ClaimGrade)],
                         ["evidence"])
        record = ClaimGrade(evidence=(ev("derivation"),))
        self.assertEqual(record.grade, ProofGrade.G2)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            record.grade = ProofGrade.G6  # type: ignore[misc]

    def test_empty_claim_is_g0(self):
        self.assertEqual(ClaimGrade().grade, ProofGrade.G0)

    def test_asserting_at_the_earned_grade(self):
        record = ClaimGrade.asserting(
            ProofGrade.G5, [ev("artifact_pass", "build-7")])
        self.assertEqual(record.grade, ProofGrade.G5)

    def test_asserting_above_the_evidence_is_refused(self):
        # The no-silent-upgrade rule: a G5 demand on G2 evidence.
        with self.assertRaises(ProofGradeError):
            ClaimGrade.asserting(ProofGrade.G5, [ev("derivation")])
        with self.assertRaises(ProofGradeError):
            ClaimGrade.asserting("G6", [])

    def test_asserting_below_the_evidence_carries_the_derived(self):
        # A demand is a floor check, never a cap: the evidence governs.
        record = ClaimGrade.asserting(
            ProofGrade.G1, [ev("empirical_source", "src-3")])
        self.assertEqual(record.grade, ProofGrade.G4)

    def test_meets_and_require(self):
        record = ClaimGrade(evidence=(ev("simulation_check", "sim-1"),))
        self.assertTrue(record.meets(ProofGrade.G3))
        self.assertTrue(record.meets("G1"))
        self.assertFalse(record.meets(ProofGrade.G4))
        record.require(ProofGrade.G3)
        with self.assertRaises(ProofGradeError):
            record.require(ProofGrade.G6)


class SerializationTests(unittest.TestCase):
    def test_round_trip_preserves_grade_and_evidence(self):
        record = ClaimGrade(evidence=(
            ev("empirical_source", "src-1"),
            ev("monitoring_receipt", "journal-9"),
        ))
        payload = record.to_dict()
        self.assertEqual(payload["grade"], "G6")
        back = ClaimGrade.from_dict(payload)
        self.assertEqual(back, record)
        self.assertEqual(back.grade, ProofGrade.G6)

    def test_stored_fiat_grade_is_refused(self):
        # A stored G5 whose attached evidence earns G2: refused.
        with self.assertRaises(ProofGradeError):
            ClaimGrade.from_dict({
                "grade": "G5",
                "evidence": [{"kind": "derivation", "ref": "calc-1"}],
            })

    def test_stored_grade_with_no_evidence_is_refused(self):
        with self.assertRaises(ProofGradeError):
            ClaimGrade.from_dict({"grade": "G5", "evidence": []})
        with self.assertRaises(ProofGradeError):
            ClaimGrade.from_dict({"grade": "G1"})

    def test_understated_stored_grade_reads_back_derived(self):
        # Evidence governs: an under-claim is corrected by the
        # evidence attached, never trusted as the standing.
        back = ClaimGrade.from_dict({
            "grade": "G1",
            "evidence": [{"kind": "empirical_source", "ref": "src-1"}],
        })
        self.assertEqual(back.grade, ProofGrade.G4)

    def test_malformed_payloads_refused(self):
        for bad in (None, "G5", {"grade": "GX", "evidence": []},
                    {"evidence": "not-a-list"},
                    {"evidence": [{"kind": "nope", "ref": "x"}]}):
            with self.assertRaises(ProofGradeError, msg=repr(bad)):
                ClaimGrade.from_dict(bad)


class DelegationMappingTests(unittest.TestCase):
    def test_known_kinds_have_exactly_one_row_each(self):
        self.assertEqual(set(DELEGATION_KIND_MAPPINGS),
                         set(DELEGATION_KNOWN_KINDS))
        for row in DELEGATION_KIND_MAPPINGS.values():
            self.assertTrue(row.reason.strip())

    def test_domain_evidence_maps_to_g4(self):
        # Concrete row: read-back from the domain's primary store.
        self.assertEqual(grade_for_delegation_kind("domain-evidence"),
                         ProofGrade.G4)

    def test_verdict_maps_to_no_grade_with_reason(self):
        self.assertIsNone(grade_for_delegation_kind("verdict"))
        reason = DELEGATION_KIND_MAPPINGS["verdict"].reason
        self.assertIn("judgment", reason)

    def test_open_vocabulary_maps_to_none_never_raises(self):
        # The manifests' prose duties are the open vocabulary in
        # production: unlisted, therefore unmapped — not guessed.
        for kind in ("test run", "build output", "deploy receipt",
                     "some future kind"):
            self.assertIsNone(grade_for_delegation_kind(kind), kind)
        self.assertIn("open vocabulary", DELEGATION_UNMAPPED_REASON)

    def test_known_kinds_pin_against_the_steward_source(self):
        # The pair lives as a literal in outcome_steward's contract
        # creation; if that literal changes, this pin must fail.
        from nomorals.agents import outcome_steward
        source = inspect.getsource(outcome_steward)
        self.assertIn('required_evidence=("domain-evidence", "verdict")',
                      source)
        self.assertEqual(DELEGATION_KNOWN_KINDS,
                         ("domain-evidence", "verdict"))


class StewardMappingTests(unittest.TestCase):
    def test_mirror_matches_the_real_vocabulary(self):
        from nomorals.agents.outcome_cases import (
            RESOLVABLE_VERDICTS,
            UNKNOWN_VERDICT,
        )
        self.assertEqual(STEWARD_VERDICTS, RESOLVABLE_VERDICTS)
        self.assertEqual(STEWARD_UNKNOWN_VERDICT, UNKNOWN_VERDICT)

    def test_every_verdict_has_exactly_one_row(self):
        expected = set(STEWARD_VERDICTS) | {STEWARD_UNKNOWN_VERDICT}
        self.assertEqual(set(STEWARD_VERDICT_MAPPINGS), expected)

    def test_no_verdict_maps_to_a_grade(self):
        # The honest table: outcomes are not proof grades.
        for verdict, row in STEWARD_VERDICT_MAPPINGS.items():
            self.assertIsNone(row.grade, verdict)
            self.assertIsNone(grade_for_steward_verdict(verdict), verdict)
            self.assertTrue(row.reason.strip(), verdict)

    def test_unknown_verdict_row_names_the_refusal(self):
        reason = STEWARD_VERDICT_MAPPINGS[STEWARD_UNKNOWN_VERDICT].reason
        self.assertIn("asserts nothing", reason)

    def test_resolution_evidence_relation_is_recorded(self):
        self.assertIn("evidence receipt", STEWARD_RESOLUTION_EVIDENCE)

    def test_unknown_verdict_string_refused(self):
        with self.assertRaises(ProofGradeError):
            grade_for_steward_verdict("kinda-worked")


class WS4MappingTests(unittest.TestCase):
    def test_mirror_matches_the_real_vocabulary(self):
        from nomorals.media.explore.trends import CLAIM_KINDS
        self.assertEqual(WS4_CLAIM_KINDS, CLAIM_KINDS)
        self.assertEqual(set(WS4_KIND_MAPPINGS), set(CLAIM_KINDS))

    def test_kind_grades(self):
        # One concrete example per row: the kind IS its bar.
        self.assertEqual(grade_for_ws4_kind("trend"), ProofGrade.G4)
        self.assertEqual(grade_for_ws4_kind("spike"), ProofGrade.G2)
        self.assertEqual(grade_for_ws4_kind("unverified"), ProofGrade.G1)
        for row in WS4_KIND_MAPPINGS.values():
            self.assertTrue(row.reason.strip())

    def test_unknown_kind_refused(self):
        with self.assertRaises(ProofGradeError):
            grade_for_ws4_kind("viral")


class ImpossibleMappingTests(unittest.TestCase):
    def test_mirror_matches_the_real_vocabulary(self):
        from nomorals.impossible.charter import STATUSES
        self.assertEqual(IMPOSSIBLE_STATUSES, STATUSES)
        self.assertEqual(set(IMPOSSIBLE_STATUS_MAPPINGS), set(STATUSES))

    def test_status_grades(self):
        expected = {
            "idea": ProofGrade.G0,
            "hypothesis": ProofGrade.G1,
            "tested": ProofGrade.G3,
            "prototype": ProofGrade.G5,
            "built": ProofGrade.G5,
            "measured": ProofGrade.G6,
        }
        for status, grade in expected.items():
            self.assertEqual(grade_for_impossible_status(status), grade,
                             status)
            self.assertTrue(
                IMPOSSIBLE_STATUS_MAPPINGS[status].reason.strip(), status)

    def test_unknown_status_refused(self):
        with self.assertRaises(ProofGradeError):
            grade_for_impossible_status("vibes-based")


class AdoptionRuleTests(unittest.TestCase):
    def test_the_adoption_rule_is_recorded_in_the_module(self):
        doc = " ".join((proof_grade.__doc__ or "").split())
        self.assertIn("No level silently upgrades itself", doc)
        self.assertIn("Doc 18", doc)          # the ticket consumes this
        self.assertIn("L-levels", doc)        # docs 22/23 map or strike
        self.assertIn("ONE grade scale", doc)


if __name__ == "__main__":
    unittest.main()
