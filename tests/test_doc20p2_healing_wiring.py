"""Doc 20 P2 — executor wiring + receipt honesty.

Two defects, fixed in this order of care:

A. RECEIPT SEMANTICS. Police receipts claimed effects that never
   happened (an unwired disarm worded "capability grant revoked").
   Now every effect rung records the performing function's outcome
   on the receipt (details.effect / details.effect_outcomes) and
   the reason text states the true outcome — success wording only
   for a performed effect with its result on the record. The P1
   healing ticket projects the new truth: not-performed/failed
   receipts carry NO claim; performed ones grade on the recorded
   outcome; legacy (pre-P2) receipts, which carry no effect
   record, still grade on cited evidence alone — a bare legacy
   success claim earns G0.

B. EXECUTOR WIRING. The SelfHealingExecutor had zero production
   callers. Tool calls — the production path every tool failure
   already funnels through (ToolRegistry._call_inner) — now route
   through ErrorSystem.healer_for() when an error system is wired:
   incident journaled, recovery attempted for the classes the
   executor handles, verified recoveries promoted by the journal.
   Unwired, the call is byte-for-byte the old plain call. A healer
   failure can never take the tool call down (fail-open).
"""

from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from nomorals.agents.healing_ticket import healing_ticket_for
from nomorals.core.budgets import BudgetManager
from nomorals.core.degradation import LadderManager
from nomorals.core.error_system import ErrorSystem, set_error_system
from nomorals.core.incidents import IncidentJournal
from nomorals.core.proof_grade import ProofGrade
from nomorals.immune.chat import control_immune, get_immune_state, reset_immune_state
from nomorals.immune.police import Police, Receipt, consult_then_enforce
from nomorals.tools.registry import ToolRegistry

CHAT_KEY = "p2-suite"


# ── A1: police receipt semantics, at the organ ─────────────────────


class PoliceReceiptHonestyTests(unittest.TestCase):

    def test_disarm_unwired_states_not_performed(self) -> None:
        receipt = Police().disarm("grant-1")
        self.assertEqual(receipt.details["effect"], "not_performed")
        self.assertEqual(receipt.details["effect_outcomes"],
                         [{"fn": "revoke_fn", "wired": False}])
        self.assertIn("no revocation performed", receipt.reason)
        self.assertIn("not wired", receipt.reason)
        # Never the bare success claim.
        self.assertNotIn("capability grant grant-1 revoked",
                         receipt.reason)

    def test_disarm_wired_records_the_outcome(self) -> None:
        calls: list[str] = []

        def revoke(grant_id: str) -> str:
            calls.append(grant_id)
            return "revoked-ok"

        receipt = Police(revoke_fn=revoke).disarm("grant-1")
        self.assertEqual(calls, ["grant-1"])
        self.assertEqual(receipt.reason,
                         "capability grant grant-1 revoked")
        self.assertEqual(receipt.details["effect"], "performed")
        outcome = receipt.details["effect_outcomes"][0]
        self.assertEqual(outcome["fn"], "revoke_fn")
        self.assertTrue(outcome["wired"])
        self.assertTrue(outcome["ok"])
        self.assertEqual(outcome["result"], "revoked-ok")

    def test_disarm_wired_but_raising_records_the_failure(self) -> None:
        def revoke(grant_id: str) -> None:
            raise RuntimeError("policy store down")

        receipt = Police(revoke_fn=revoke).disarm("grant-1")
        self.assertEqual(receipt.details["effect"], "failed")
        self.assertIn("failed", receipt.reason)
        self.assertIn("RuntimeError", receipt.reason)
        self.assertNotIn("capability grant grant-1 revoked",
                         receipt.reason)
        outcome = receipt.details["effect_outcomes"][0]
        self.assertFalse(outcome["ok"])
        self.assertIn("policy store down", outcome["error"])

    def test_quarantine_unwired_leaves_artifact_unmarked(self) -> None:
        artifact = {"id": "art-1", "text": "blob"}
        receipt = Police().quarantine(artifact)
        self.assertEqual(receipt.details["effect"], "not_performed")
        self.assertNotIn("quarantined", artifact)
        self.assertIn("not moved to vault", receipt.reason)
        self.assertIn("no vault function wired", receipt.reason)

    def test_quarantine_wired_vaults_and_marks(self) -> None:
        vault: list[object] = []
        artifact = {"id": "art-1", "text": "blob"}
        receipt = Police(vault_fn=vault.append).quarantine(artifact)
        self.assertEqual(vault, [artifact])
        self.assertTrue(artifact["quarantined"])
        self.assertEqual(receipt.details["effect"], "performed")
        self.assertEqual(receipt.reason,
                         "artifact moved to vault, never deleted")

    def test_quarantine_raising_vault_records_failure(self) -> None:
        def vault(artifact: object) -> None:
            raise OSError("vault read-only")

        artifact = {"id": "art-1"}
        receipt = Police(vault_fn=vault).quarantine(artifact)
        self.assertEqual(receipt.details["effect"], "failed")
        self.assertNotIn("quarantined", artifact)
        self.assertIn("OSError", receipt.reason)

    def test_arrest_both_wired_performs_both(self) -> None:
        killed: list[str] = []
        cancelled: list[str] = []
        receipt = Police(kill_fn=killed.append,
                         cancel_fn=cancelled.append).arrest("proc-9")
        self.assertEqual(killed, ["proc-9"])
        self.assertEqual(cancelled, ["proc-9"])
        self.assertEqual(receipt.details["effect"], "performed")
        self.assertIn("process proc-9 killed", receipt.reason)
        self.assertIn("mission cancelled", receipt.reason)

    def test_arrest_partial_names_the_unwired_half(self) -> None:
        killed: list[str] = []
        receipt = Police(kill_fn=killed.append).arrest("proc-9")
        self.assertEqual(receipt.details["effect"], "partial")
        self.assertIn("process proc-9 killed", receipt.reason)
        self.assertIn("mission not cancelled", receipt.reason)
        self.assertIn("no cancel function wired", receipt.reason)

    def test_arrest_unwired_performs_nothing(self) -> None:
        receipt = Police().arrest("proc-9")
        self.assertEqual(receipt.details["effect"], "not_performed")
        self.assertIn("not killed", receipt.reason)
        self.assertIn("not cancelled", receipt.reason)

    def test_build_case_unwired_builds_nothing(self) -> None:
        receipt = Police().build_case({"target": "artifact-9"})
        self.assertEqual(receipt.details["effect"], "not_performed")
        self.assertIsNone(receipt.details["case_id"])
        self.assertIn("no case file built", receipt.reason)

    def test_build_case_wired_records_the_case_id(self) -> None:
        class _Case:
            id = "case-7"

        receipt = Police(
            casefile_fn=lambda event, timeline_fn=None: _Case(),
        ).build_case({"target": "artifact-9"})
        self.assertEqual(receipt.details["effect"], "performed")
        self.assertEqual(receipt.details["case_id"], "case-7")
        self.assertEqual(receipt.reason, "case file built")

    def test_build_case_wired_without_id_says_so(self) -> None:
        receipt = Police(
            casefile_fn=lambda event, timeline_fn=None: object(),
        ).build_case({"target": "artifact-9"})
        self.assertEqual(receipt.details["effect"], "performed")
        self.assertIsNone(receipt.details["case_id"])
        self.assertIn("no case id", receipt.reason)
        self.assertNotEqual(receipt.reason, "case file built")

    def test_notify_batched_with_digest_records_delivery(self) -> None:
        digested: list[object] = []
        police = Police(
            presence_fn=lambda event: {"interruptible": False},
            digest_fn=digested.append)
        receipt = police.notify({"target": "t1"})
        self.assertEqual(receipt.action, "notify_batched")
        self.assertEqual(len(digested), 1)
        self.assertEqual(receipt.details["delivered"], "digest")
        self.assertEqual(receipt.details["effect"], "performed")

    def test_notify_batched_without_digest_delivers_nothing(self) -> None:
        police = Police(
            presence_fn=lambda event: {"interruptible": False})
        receipt = police.notify({"target": "t1"})
        self.assertEqual(receipt.action, "notify_batched")
        self.assertEqual(receipt.details["delivered"], "none")
        self.assertEqual(receipt.details["effect"], "not_performed")
        self.assertIn("nothing delivered", receipt.reason)

    def test_notify_immediate_is_a_decision_not_a_delivery(self) -> None:
        receipt = Police().notify({"target": "t1"})
        self.assertEqual(receipt.action, "notify")
        self.assertEqual(receipt.details["effect"], "not_performed")
        self.assertEqual(receipt.details["delivered"],
                         "decision_immediate")
        self.assertIn("performs no delivery itself", receipt.reason)

    def test_enforce_unwired_takes_no_effect(self) -> None:
        receipt = consult_then_enforce(
            {"processes_spawned": 500, "process_id": "p1"},
            is_system_process=True, owner_directed=False)
        self.assertEqual(receipt.action, "enforce")
        self.assertFalse(receipt.details["enforced"])
        self.assertEqual(receipt.details["effect"], "not_performed")
        self.assertIn("no enforce function wired", receipt.reason)
        self.assertIn("no effect taken", receipt.reason)

    def test_enforce_wired_records_the_effect(self) -> None:
        enforced: list[object] = []
        receipt = consult_then_enforce(
            {"processes_spawned": 500, "process_id": "p1"},
            is_system_process=True, owner_directed=False,
            enforce_fn=enforced.append)
        self.assertEqual(len(enforced), 1)
        self.assertTrue(receipt.details["enforced"])
        self.assertEqual(receipt.details["effect"], "performed")

    def test_respond_dispatch_disarm_unwired_is_honest(self) -> None:
        receipt = Police().respond(
            {"grant_id": "g1", "target": "g1"}, rung=4)
        self.assertEqual(receipt.action, "disarm")
        self.assertEqual(receipt.details["effect"], "not_performed")
        self.assertIn("no revocation performed", receipt.reason)


# ── A2: the P1 projection over new + legacy receipts ───────────────


class ProjectionOverReceiptsTests(unittest.TestCase):

    def setUp(self) -> None:
        reset_immune_state(CHAT_KEY)
        self.state = get_immune_state(CHAT_KEY)

    def tearDown(self) -> None:
        reset_immune_state(CHAT_KEY)

    def ticket_for(self, receipt: object) -> dict:
        self.state._record(receipt)
        return healing_ticket_for(
            f"receipt:{receipt.id}",
            immune_states={CHAT_KEY: self.state})

    def test_legacy_bare_disarm_still_grades_g0(self) -> None:
        """A pre-P2 receipt — success wording, no effect record —
        grades on cited evidence alone: none attached, G0. The
        record is all the ticket may grade."""
        legacy = Receipt(
            rung=4, action="disarm", target="grant-1",
            reason="capability grant grant-1 revoked",
            details={"grant_id": "grant-1"})
        ticket = self.ticket_for(legacy)
        self.assertEqual(ticket["claim"]["grade"], "G0")
        self.assertEqual(ticket["claim"]["evidence"], [])
        self.assertIn("revoke function was wired",
                      ticket["claim"]["grade_reason"])

    def test_legacy_bare_casefile_still_grades_g0(self) -> None:
        legacy = Receipt(
            rung=6, action="casefile", target="artifact-9",
            reason="case file built", details={"case_id": None})
        ticket = self.ticket_for(legacy)
        self.assertEqual(ticket["claim"]["grade"], "G0")
        self.assertIn("no case id", ticket["claim"]["grade_reason"])

    def test_recorded_claim_grades_strictly_above_legacy(self) -> None:
        """The discrimination property: a performed-and-recorded
        disarm grades strictly above a bare legacy claim of the
        same words."""
        recorded = self.ticket_for(
            Police(revoke_fn=lambda grant_id: True).disarm("grant-1"))
        legacy = self.ticket_for(Receipt(
            rung=4, action="disarm", target="grant-1",
            reason="capability grant grant-1 revoked",
            details={"grant_id": "grant-1"}))
        self.assertGreater(ProofGrade[recorded["claim"]["grade"]],
                           ProofGrade[legacy["claim"]["grade"]])
        self.assertEqual(recorded["claim"]["grade"], "G5")

    def test_partial_arrest_projects_a_partial_claim(self) -> None:
        receipt = Police(kill_fn=lambda pid: None).arrest("proc-9")
        ticket = self.ticket_for(receipt)
        self.assertIn("PARTIAL", ticket["claim"]["grade_reason"])
        self.assertEqual(ticket["claim"]["grade"], "G5")
        self.assertIn("not cancelled", ticket["claim"]["text"])

    def test_failed_disarm_projects_no_claim(self) -> None:
        def revoke(grant_id: str) -> None:
            raise RuntimeError("policy store down")

        ticket = self.ticket_for(Police(revoke_fn=revoke).disarm("g"))
        self.assertIn("claim", ticket["absent"])
        self.assertIn("FAILED", ticket["absent"]["claim"])
        self.assertIn("RuntimeError", ticket["absent"]["claim"])

    def test_notify_decision_projects_no_claim(self) -> None:
        ticket = self.ticket_for(Police().notify({"target": "t1"}))
        self.assertIn("claim", ticket["absent"])
        self.assertIn("NO effect was performed",
                      ticket["absent"]["claim"])


# ── A3: the owner surface follows the receipt ──────────────────────


class ChatSurfaceHonestyTests(unittest.TestCase):

    def tearDown(self) -> None:
        reset_immune_state("p2-chat")

    def test_chat_disarm_reply_never_claims_an_unwired_revoke(self) -> None:
        out = control_immune("disarm grant-9", chat_key="p2-chat")
        self.assertIn("NOT performed", out)
        self.assertIn("no revocation performed", out)
        self.assertNotIn("disarmed: grant grant-9 revoked", out)


# ── B: executor wiring through the production tool path ────────────


class _FlakyError(ConnectionError):
    """A transient failure whose retry hint keeps tests fast:
    the shipped BackoffPolicy honors retry_after first."""
    retry_after = 0.0


class ExecutorWiringTests(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.journal_path = str(Path(self.tmp.name) / "journal.db")
        self.journal = IncidentJournal(path=self.journal_path)
        self.es = ErrorSystem(
            journal=self.journal,
            budgets=BudgetManager(journal=self.journal),
            ladders=LadderManager())
        self.registry = ToolRegistry()
        self.addCleanup(set_error_system, None)

    def tearDown(self) -> None:
        set_error_system(None)

    # journal reads (the journal P1 reads — its own file)

    def _rows(self, sql: str) -> list[tuple]:
        conn = sqlite3.connect(f"file:{self.journal_path}?mode=ro",
                               uri=True)
        try:
            return conn.execute(sql).fetchall()
        finally:
            conn.close()

    def _incidents(self, subsystem: str) -> int:
        return self._rows(
            "SELECT COUNT(*) FROM incidents WHERE subsystem = "
            f"'{subsystem}'")[0][0]

    def _recoveries(self) -> list[tuple]:
        return self._rows(
            "SELECT strategy, verified FROM recoveries ORDER BY id")

    def test_unwired_registry_is_the_plain_call(self) -> None:
        set_error_system(None)
        calls: list[int] = []

        def flaky() -> str:
            calls.append(1)
            if len(calls) == 1:
                raise _FlakyError("transient")
            return "ok-result"

        self.registry.register("flaky", flaky)
        outcome = self.registry.call("flaky")
        # No error system: no retry, no healing — the first
        # failure is the result, exactly as before P2.
        self.assertFalse(outcome.ok)
        self.assertEqual(len(calls), 1)

    def test_flaky_tool_recovers_end_to_end(self) -> None:
        set_error_system(self.es)
        calls: list[int] = []

        def flaky() -> str:
            calls.append(1)
            if len(calls) == 1:
                raise _FlakyError("transient network blip")
            return "ok-result"

        self.registry.register("flaky", flaky)
        outcome = self.registry.call("flaky")
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.value, "ok-result")
        self.assertEqual(len(calls), 2)
        # Incident journaled, recovery recorded VERIFIED, and the
        # journal's own promotion made it a known fix.
        self.assertEqual(self._incidents("tools/flaky"), 1)
        self.assertEqual(self._recoveries(), [("retry", 1)])
        fixes = self._rows("SELECT strategy FROM verified_fixes")
        self.assertEqual(fixes, [("retry",)])

    def test_permanent_failure_journals_an_unverified_recovery(self) -> None:
        set_error_system(self.es)
        calls: list[int] = []

        def broken() -> str:
            calls.append(1)
            raise _FlakyError("still down")

        self.registry.register("broken", broken)
        outcome = self.registry.call("broken")
        # The caller sees the tool's own failure, unchanged.
        self.assertFalse(outcome.ok)
        self.assertIn("still down", outcome.error.message)
        self.assertGreater(len(calls), 1)  # the ladder climbed
        self.assertEqual(self._incidents("tools/broken"), 1)
        self.assertEqual(self._recoveries(), [("retry", 0)])
        self.assertEqual(self._rows("SELECT * FROM verified_fixes"),
                         [])

    def test_unhandled_class_journals_no_recovery(self) -> None:
        """A failure class the executor does not handle (a plain
        programming error, not retryable) records the incident —
        and NO recovery. Nothing is fabricated."""
        set_error_system(self.es)
        calls: list[int] = []

        def buggy() -> str:
            calls.append(1)
            raise ValueError("bad input shape")

        self.registry.register("buggy", buggy)
        outcome = self.registry.call("buggy")
        self.assertFalse(outcome.ok)
        self.assertIn("bad input shape", outcome.error.message)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self._incidents("tools/buggy"), 1)
        self.assertEqual(self._recoveries(), [])

    def test_healer_machinery_failure_fails_open(self) -> None:
        """If healer_for itself raises, the tool call proceeds as
        the plain call — success and failure alike."""
        set_error_system(self.es)

        def boom(subsystem: str) -> object:
            raise RuntimeError("healer broke")

        self.es.healer_for = boom  # type: ignore[method-assign]
        self.registry.register("fine", lambda: "ok-result")
        outcome = self.registry.call("fine")
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.value, "ok-result")

        def broken() -> str:
            raise ValueError("the tool's own failure")

        self.registry.register("broken2", broken)
        outcome = self.registry.call("broken2")
        self.assertFalse(outcome.ok)
        self.assertIn("the tool's own failure", outcome.error.message)

    def test_healer_raising_after_the_call_keeps_its_result(self) -> None:
        """If the executor dies AFTER the callable produced a
        result, the caller still gets that result."""
        set_error_system(self.es)

        class _DyingHealer:
            def execute(self, fn, **kwargs):  # noqa: ANN001, ANN202
                fn()  # the tool ran and produced its result
                raise RuntimeError("executor died afterwards")

        self.es.healer_for = lambda subsystem: _DyingHealer()  # type: ignore[method-assign]
        self.registry.register("fine", lambda: "ok-result")
        outcome = self.registry.call("fine")
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.value, "ok-result")


if __name__ == "__main__":
    unittest.main()
