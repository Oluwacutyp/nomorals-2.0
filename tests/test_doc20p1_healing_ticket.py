"""Doc 20 P1 — the healing-ticket projection: pins.

Offline: every record is planted through the organs' REAL
writers/stores — IncidentJournal (record_incident /
record_recovery), the chat module's real per-chat immune state,
Thymus + AntibodyStore (propose/accept/add), Police (its real
respond() rungs), CaseFile via police.build_case,
AdaptiveBaseline.record() — so the projection is proven against
the store shapes that actually ship, not invented dicts.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nomorals.agents.healing_ticket import (  # noqa: E402
    HEALING_FIELDS,
    healing_ticket_coverage,
    healing_ticket_for,
)
from nomorals.core.incidents import (  # noqa: E402
    IncidentJournal,
    RecoveryRecord,
)
from nomorals.immune.antibodies import AntibodyStore  # noqa: E402
from nomorals.immune.baseline import AdaptiveBaseline  # noqa: E402
from nomorals.immune.chat import get_immune_state, reset_immune_state  # noqa: E402
from nomorals.immune.thymus import (  # noqa: E402
    SelfCorpus,
    SelfExemplar,
    Thymus,
)


def _raise_value_error() -> None:
    try:
        raise ValueError("kaboom for the ticket")
    except ValueError:
        raise


class HealingTicketBase(unittest.TestCase):
    CHAT_KEY = "doc20p1-test-chat"

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="doc20p1-")
        self.journal_path = str(Path(self.tmp) / "incidents.sqlite3")
        self.journal = IncidentJournal(self.journal_path)
        self.state = get_immune_state(self.CHAT_KEY)

    def tearDown(self) -> None:
        reset_immune_state(self.CHAT_KEY)

    # ── real-writer planters ──────────────────────────────────

    def plant_incident(self, subsystem: str = "immune-test") -> int:
        try:
            _raise_value_error()
        except ValueError as exc:
            incident = self.journal.record_incident(
                exc, subsystem=subsystem, severity="high",
                context={"origin": "doc20p1-test"})
        return incident.id

    def plant_recovery(self, incident_id: int, *, verified: bool,
                       strategy: str = "restart_component") -> int:
        return self.journal.record_recovery(RecoveryRecord(
            incident_id=incident_id,
            signature="sig-doc20p1",
            strategy=strategy,
            detail="restarted the failing component",
            verified=verified,
            verify_note=("re-ran the failing call: ok" if verified
                         else "verify step never confirmed"),
        ))

    def plant_antibody(self, *, status: str = "active") -> dict:
        thymus = self.state.thymus
        thymus.corpus = SelfCorpus(exemplars=[SelfExemplar(
            text="a benign self pattern for selection")])
        proposed = thymus.propose(
            pattern=r"invoice\s+paid",
            pathogen_class="shady_invoice",
            reason="planted by the ticket suite")
        accepted = thymus.accept(proposed)
        record = self.state.store.add(accepted)
        if status == "retired":
            retired = thymus.retire(accepted, "suite retirement")
            record = self.state.store.update(retired)
        return record

    def plant_disarm_receipt(self, *, revoke_fn=None) -> object:
        police = self.state.police
        if revoke_fn is not None:
            police.revoke_fn = revoke_fn
        receipt = police.disarm("grant-1")
        self.state._record(receipt)
        return receipt

    def plant_case_receipt(self, *, wired: bool) -> tuple[object, object | None]:
        police = self.state.police
        if wired:
            from nomorals.immune.casefile import build_case as build

            def _casefile_fn(event, timeline_fn=None):
                # The chat module's own construction (immune/chat.py):
                # build a CaseFile from the event and file it on the
                # state.
                case = build({"text": str(event)})
                self.state.casefiles.append(case)
                return case

            police.casefile_fn = _casefile_fn
        receipt = police.build_case(
            {"text": "suspicious attachment", "target": "artifact-9"})
        self.state._record(receipt)
        case = self.state.casefiles[-1] if self.state.casefiles else None
        return receipt, case

    def assert_partition(self, ticket: dict) -> None:
        sourced = set(ticket["sources"])
        absent = set(ticket["absent"])
        self.assertEqual(sourced | absent, set(HEALING_FIELDS),
                         f"partition broken on {ticket['ref']}")
        self.assertFalse(sourced & absent,
                         f"field both sourced and absent on {ticket['ref']}")
        for name in sourced:
            self.assertIsNotNone(ticket[name],
                                 f"sourced field {name} is None")


class IncidentTicketTests(HealingTicketBase):

    def test_incident_ticket_partition_and_fields(self) -> None:
        iid = self.plant_incident()
        ticket = healing_ticket_for(f"incident:{iid}",
                                    journal=self.journal)
        self.assertTrue(ticket["found"])
        self.assertEqual(ticket["kind"], "incident")
        self.assertEqual(ticket["purpose"]["exc_type"], "ValueError")
        self.assertEqual(ticket["purpose"]["subsystem"], "immune-test")
        self.assert_partition(ticket)
        # An incident asserts no effect: claim + actors + steward
        # are the named absences.
        for name in ("actors", "claim", "steward"):
            self.assertIn(name, ticket["absent"])
        self.assertEqual(ticket["lifecycle"]["standing"], "open")

    def test_incident_standing_recovered_and_heal_related(self) -> None:
        iid = self.plant_incident()
        self.plant_recovery(iid, verified=True)
        ticket = healing_ticket_for(
            f"incident:{iid}", journal=self.journal,
            journal_db=self.journal_path)
        self.assertEqual(ticket["lifecycle"]["standing"], "recovered")
        self.assertTrue(any(ref.startswith("heal:")
                            for ref in ticket["related_refs"]))

    def test_incident_ticket_via_db_only(self) -> None:
        iid = self.plant_incident()
        ticket = healing_ticket_for(f"incident:{iid}",
                                    journal_db=self.journal_path)
        self.assertTrue(ticket["found"])
        self.assertEqual(ticket["evidence"]["repeat_count_24h"], 1)
        self.assert_partition(ticket)

    def test_missing_incident_is_typed_not_found(self) -> None:
        result = healing_ticket_for("incident:99999",
                                    journal=self.journal)
        self.assertFalse(result["found"])
        self.assertIn("no incident record", result["reason"])

    def test_incident_without_handles_is_typed_not_found(self) -> None:
        result = healing_ticket_for("incident:1")
        self.assertFalse(result["found"])
        self.assertIn("journal", result["reason"])


class HealingRunTicketTests(HealingTicketBase):

    def test_verified_recovery_grades_g5(self) -> None:
        iid = self.plant_incident()
        rid = self.plant_recovery(iid, verified=True)
        ticket = healing_ticket_for(f"heal:{rid}",
                                    journal_db=self.journal_path)
        self.assertTrue(ticket["found"])
        self.assert_partition(ticket)
        self.assertEqual(ticket["status"], "verified")
        self.assertEqual(ticket["claim"]["grade"], "G5")
        self.assertEqual(ticket["claim"]["evidence"][0]["kind"],
                         "artifact_pass")
        # The journal promoted the verified fix (Mined rule 3).
        self.assertTrue(ticket["evidence"]["promoted_fix"])
        self.assertEqual(ticket["evidence"]["verify_note"],
                         "re-ran the failing call: ok")
        self.assertIn(f"incident:{iid}", ticket["related_refs"])
        self.assertIn("steward", ticket["absent"])

    def test_unverified_recovery_grades_g0(self) -> None:
        iid = self.plant_incident()
        rid = self.plant_recovery(iid, verified=False)
        ticket = healing_ticket_for(f"heal:{rid}",
                                    journal_db=self.journal_path)
        self.assertEqual(ticket["status"], "unverified")
        self.assertEqual(ticket["claim"]["grade"], "G0")
        self.assertEqual(ticket["claim"]["evidence"], [])
        self.assertIn("UNVERIFIED", ticket["claim"]["grade_reason"])
        self.assertFalse(ticket["evidence"]["verified"])

    def test_evidence_backed_grades_higher_than_evidence_free(self) -> None:
        """The machinery discriminates: same envelope, same claim
        shape — the record whose own journal entry evidences the
        effect grades strictly above the one that doesn't."""
        iid = self.plant_incident()
        verified = healing_ticket_for(
            f"heal:{self.plant_recovery(iid, verified=True)}",
            journal_db=self.journal_path)
        unverified = healing_ticket_for(
            f"heal:{self.plant_recovery(iid, verified=False)}",
            journal_db=self.journal_path)
        self.assertGreater(verified["claim"]["grade"],
                           unverified["claim"]["grade"])

    def test_heal_reads_via_journal_instance_alone(self) -> None:
        """A file-backed journal instance is enough — the
        projection finds the journal's own database file."""
        iid = self.plant_incident()
        rid = self.plant_recovery(iid, verified=True)
        ticket = healing_ticket_for(f"heal:{rid}", journal=self.journal)
        self.assertTrue(ticket["found"])

    def test_heal_with_memory_only_journal_is_typed_not_found(self) -> None:
        memory_journal = IncidentJournal(":memory:")
        result = healing_ticket_for("heal:1", journal=memory_journal)
        self.assertFalse(result["found"])
        self.assertIn("database", result["reason"])


class PoliceReceiptTicketTests(HealingTicketBase):

    def test_disarm_claim_grades_on_cited_evidence_not_wording(self) -> None:
        """THE disarm pin (doc 20's sharpest finding), after P2:
        with no revoke function wired, the receipt itself now
        records that NO revocation was performed — so the ticket
        carries no claim at all. (In P1 the same unwired disarm
        worded a bare "revoked" claim and graded G0; P2 fixed the
        receipt semantics, and a legacy receipt of that shape
        still grades G0 — pinned in the P2 suite.)"""
        receipt = self.plant_disarm_receipt()
        self.assertEqual(
            receipt.reason,
            "no revocation performed — revoke function not wired; "
            "grant grant-1 left in force")
        ticket = healing_ticket_for(
            f"receipt:{receipt.id}",
            immune_states={self.CHAT_KEY: self.state})
        self.assertTrue(ticket["found"])
        self.assert_partition(ticket)
        self.assertIn("claim", ticket["absent"])
        self.assertIn("NO effect was performed",
                      ticket["absent"]["claim"])
        self.assertIn("revoke_fn was not wired",
                      ticket["absent"]["claim"])
        self.assertIn("evidence", ticket["absent"])
        self.assertEqual(ticket["actors"]["actor"], "immune.police")
        self.assertEqual(ticket["purpose"]["rung_name"], "Disarm")

    def test_disarm_wired_and_recorded_grades_on_the_outcome(self) -> None:
        """With a working revoke_fn, P2's receipt records the call
        and its return — the claim now grades on that outcome
        record (an operation pass), strictly above the G0 a bare
        unrecorded claim earns."""
        receipt = self.plant_disarm_receipt(
            revoke_fn=lambda grant_id: True)
        ticket = healing_ticket_for(
            f"receipt:{receipt.id}",
            immune_states={self.CHAT_KEY: self.state})
        self.assertEqual(ticket["claim"]["text"],
                         "capability grant grant-1 revoked")
        self.assertEqual(ticket["claim"]["grade"], "G5")
        self.assertEqual(ticket["claim"]["evidence"],
                         [{"kind": "artifact_pass",
                           "ref": f"receipt:{receipt.id}"}])
        self.assertIn("PERFORMED", ticket["claim"]["grade_reason"])

    def test_casefile_receipt_with_built_case_grades_g5(self) -> None:
        receipt, case = self.plant_case_receipt(wired=True)
        self.assertIsNotNone(case)
        ticket = healing_ticket_for(
            f"receipt:{receipt.id}",
            immune_states={self.CHAT_KEY: self.state})
        self.assertEqual(ticket["claim"]["grade"], "G5")
        self.assertEqual(ticket["claim"]["evidence"][0]["ref"],
                         f"case:{case.id}")
        self.assertIn(f"case:{case.id}", ticket["related_refs"])
        self.assertEqual(ticket["evidence"]["cited_refs"],
                         [f"case:{case.id}"])

    def test_casefile_receipt_without_case_has_no_claim(self) -> None:
        """After P2, an unwired build_case records that no case
        file was built — the ticket carries no claim. (A legacy
        receipt wording the bare claim with no case id still
        grades G0 — pinned in the P2 suite.)"""
        receipt, case = self.plant_case_receipt(wired=False)
        self.assertIsNone(case)
        ticket = healing_ticket_for(
            f"receipt:{receipt.id}",
            immune_states={self.CHAT_KEY: self.state})
        self.assertIn("claim", ticket["absent"])
        self.assertIn("casefile_fn was not wired",
                      ticket["absent"]["claim"])

    def test_report_only_receipt_has_no_claim(self) -> None:
        receipt = self.state.police.respond(
            {"owner_directed": True, "target": "owner-work"})
        self.state._record(receipt)
        self.assertEqual(receipt.action, "report_only")
        ticket = healing_ticket_for(
            f"receipt:{receipt.id}",
            immune_states={self.CHAT_KEY: self.state})
        self.assertIn("claim", ticket["absent"])
        self.assertIn("NO effect", ticket["absent"]["claim"])
        self.assert_partition(ticket)


class CaseAndAntibodyTicketTests(HealingTicketBase):

    def test_case_ticket(self) -> None:
        receipt, case = self.plant_case_receipt(wired=True)
        case.add_evidence("the suspicious bytes")
        case.seal()
        ticket = healing_ticket_for(
            f"case:{case.id}",
            immune_states={self.CHAT_KEY: self.state})
        self.assertTrue(ticket["found"])
        self.assert_partition(ticket)
        self.assertEqual(ticket["status"], "sealed")
        self.assertEqual(ticket["evidence"]["evidence_items"], 1)
        self.assertIn("claim", ticket["absent"])
        self.assertIn(f"receipt:{receipt.id}", ticket["related_refs"])

    def test_case_needs_supplied_state(self) -> None:
        result = healing_ticket_for("case:whatever")
        self.assertFalse(result["found"])
        self.assertIn("immune state", result["reason"])

    def test_antibody_ticket(self) -> None:
        record = self.plant_antibody()
        ticket = healing_ticket_for(
            f"antibody:{record['id']}",
            immune_states={self.CHAT_KEY: self.state})
        self.assertTrue(ticket["found"])
        self.assert_partition(ticket)
        self.assertEqual(ticket["lifecycle"]["status"], "active")
        self.assertEqual(ticket["lifecycle"]["version"], 1)
        # The match claim cites no evidence artifact: G0, reason
        # named — a rationale is not evidence.
        self.assertEqual(ticket["claim"]["grade"], "G0")
        self.assertIn("not recorded on the record",
                      ticket["claim"]["grade_reason"])
        self.assertEqual(ticket["evidence"]["versions_in_history"], 0)

    def test_antibody_retired_lifecycle(self) -> None:
        record = self.plant_antibody(status="retired")
        ticket = healing_ticket_for(
            f"antibody:{record['id']}",
            immune_states={self.CHAT_KEY: self.state})
        self.assertEqual(ticket["lifecycle"]["status"], "retired")
        self.assertEqual(ticket["lifecycle"]["retire_reason"],
                         "suite retirement")

    def test_antibody_via_standalone_store(self) -> None:
        store = AntibodyStore(path=str(Path(self.tmp) / "ab.json"))
        thymus = Thymus(SelfCorpus(exemplars=[SelfExemplar(
            text="ordinary chat, nothing to see")]))
        accepted = thymus.accept(thymus.propose(
            r"phish\s+link", "phishing", reason="standalone store"))
        record = store.add(accepted)
        ticket = healing_ticket_for(f"antibody:{record['id']}",
                                    antibody_stores=[store])
        self.assertTrue(ticket["found"])
        self.assertEqual(ticket["purpose"]["pathogen_class"],
                         "phishing")


class ImmuneStateTicketTests(HealingTicketBase):

    def test_state_ticket_with_fever_drift_and_baseline(self) -> None:
        self.plant_antibody()
        self.state.fever.start("suite fever", 600,
                               announce_fn=self.state
                               .fever_announcements.append)
        for day in range(3):
            self.state.drift.add_day(
                f"2026-11-0{day + 1}",
                {"tokens": 100 + day * 900, "tool_calls": 50,
                 "errors": 10, "processes_spawned": 5})
        baseline = AdaptiveBaseline()
        for i in range(30):
            baseline.record({"processes_spawned": (i % 5) + 1,
                             "process_growth": 100.0 + i,
                             "tokens": 1000, "errors": 0})
        ticket = healing_ticket_for(
            f"state:{self.CHAT_KEY}",
            immune_states={self.CHAT_KEY: self.state},
            baseline=baseline)
        self.assertTrue(ticket["found"])
        self.assert_partition(ticket)
        self.assertEqual(ticket["status"], "fever")
        self.assertTrue(ticket["lifecycle"]["fever_active"])
        self.assertEqual(ticket["lifecycle"]["drift_days"], 3)
        self.assertEqual(ticket["liveness"]["antibodies"], 1)
        self.assertEqual(ticket["liveness"]["active_antibodies"], 1)
        self.assertEqual(ticket["evidence"]["baseline"]["status"],
                         "ready")
        self.assertEqual(ticket["evidence"]["baseline"]["samples"], 30)

    def test_state_ticket_names_unread_baseline(self) -> None:
        ticket = healing_ticket_for(
            "state:default",
            immune_states={"default": self.state})
        self.assertTrue(ticket["found"])
        self.assertIn("baseline_unread", ticket["detail"])
        self.assertIn("ImmuneMonitor", ticket["detail"]["baseline_unread"])


class CoverageTests(HealingTicketBase):

    def test_coverage_over_planted_population(self) -> None:
        iid = self.plant_incident()
        self.plant_recovery(iid, verified=True)
        self.plant_recovery(iid, verified=False)
        self.plant_antibody()
        self.plant_disarm_receipt()
        _receipt, case = self.plant_case_receipt(wired=True)
        self.assertIsNotNone(case)
        coverage = healing_ticket_coverage(
            journal=self.journal, journal_db=self.journal_path,
            immune_states={self.CHAT_KEY: self.state})
        self.assertEqual(coverage["by_source"].get("incident"), 1)
        self.assertEqual(coverage["by_source"].get("healing_run"), 2)
        self.assertEqual(coverage["by_source"].get("antibody"), 1)
        self.assertEqual(coverage["by_source"].get("immune_case"), 1)
        self.assertEqual(coverage["by_source"].get("police_receipt"), 2)
        self.assertEqual(coverage["by_source"].get("immune_state"), 1)
        self.assertEqual(coverage["operations"], 8)
        # Every planted kind keeps actors and/or steward absent,
        # so NO ticket is complete — the count is never inflated.
        self.assertEqual(coverage["complete_tickets"], 0)
        self.assertEqual(coverage["fields_absent"]["steward"], 8)
        # Claim grades discriminate across the population. After
        # doc 20 P2 the planted (unwired) disarm records that no
        # effect was performed, so it carries NO claim and drops
        # out of the tally entirely (was G0 in P1, when the receipt
        # still worded a bare "revoked" claim).
        self.assertEqual(coverage["claim_grades"].get("G5"), 2)
        self.assertEqual(coverage["claim_grades"].get("G0"), 2)
        self.assertEqual(coverage["unread"], [])
        self.assertTrue(any("complete tickets" in line
                            for line in coverage["summary"]))
        for entry in coverage["operations_detail"]:
            self.assertIn("ref", entry)

    def test_coverage_names_unread_sources(self) -> None:
        coverage = healing_ticket_coverage()
        self.assertEqual(coverage["operations"], 0)
        self.assertEqual(coverage["complete_tickets"], 0)
        self.assertTrue(any("incident journal" in note
                            for note in coverage["unread"]))
        self.assertTrue(any("immune states" in note
                            for note in coverage["unread"]))


class DegradationTests(HealingTicketBase):

    def test_bad_ref_shape_is_typed_not_found(self) -> None:
        result = healing_ticket_for("nonsense")
        self.assertFalse(result["found"])
        result = healing_ticket_for("wat:1")
        self.assertFalse(result["found"])

    def test_missing_journal_file_reads_as_no_records(self) -> None:
        missing = str(Path(self.tmp) / "nope.sqlite3")
        result = healing_ticket_for("incident:1", journal_db=missing)
        self.assertFalse(result["found"])
        self.assertIn("no incident record", result["reason"])
        # Observation created nothing.
        self.assertFalse(Path(missing).exists())

    def test_missing_tables_degrade(self) -> None:
        empty = str(Path(self.tmp) / "empty.sqlite3")
        Path(empty).write_bytes(b"")
        result = healing_ticket_for("heal:1", journal_db=empty)
        self.assertFalse(result["found"])
        self.assertIn("no recovery record", result["reason"])

    def test_steward_absent_on_every_kind(self) -> None:
        iid = self.plant_incident()
        rid = self.plant_recovery(iid, verified=True)
        record = self.plant_antibody()
        receipt = self.plant_disarm_receipt()
        states = {self.CHAT_KEY: self.state}
        tickets = [
            healing_ticket_for(f"incident:{iid}", journal=self.journal,
                               journal_db=self.journal_path),
            healing_ticket_for(f"heal:{rid}",
                               journal_db=self.journal_path),
            healing_ticket_for(f"antibody:{record['id']}",
                               immune_states=states),
            healing_ticket_for(f"receipt:{receipt.id}",
                               immune_states=states),
            healing_ticket_for(f"state:{self.CHAT_KEY}",
                               immune_states=states),
        ]
        for ticket in tickets:
            self.assertTrue(ticket["found"], ticket)
            self.assertIn("steward", ticket["absent"])
            self.assertIn("never fabricated",
                          ticket["absent"]["steward"])


if __name__ == "__main__":
    unittest.main()
