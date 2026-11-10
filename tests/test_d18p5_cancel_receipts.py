"""Doc 18 P5 — cancellation receipts, delegation's cancel
terminal, and the help protocol.

Pins: (a) one receipt per organ with its mapping — browser
unhonored → receipt unhonored; coremind queued-cancel → confirmed
with "never ran"; a job-flag cancel → requested (before P5 the
flag was the whole record and NO receipt existed anywhere; the
facade receipt is now the durable record of the request); (b) an
unmapped native state refuses the receipt but the cancel still
executes; a receipt failure never blocks or undoes a cancel;
(c) delegation cancel: a live contract cancels terminal + its
claims disposition follows the shipped append-only ledger rule
(open challenges/unverified evidence remain, named in the
cancellation claim) · a stale-generation cancel is refused ·
completed/waived contracts refuse with their state named · a
late completion after cancel cannot move the contract; (d) help:
a request lands open → an answer lands on the record → the
ticket shows both states; (e) ticket flips: contract cancel is
filled from delegation (no longer absent), and a facade receipt
rides in a ticket's cancel field when one exists for the ref.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace

from nomorals.agents.coremind import CoreMind, Intent
from nomorals.agents.delegation import (
    DelegationError,
    DelegationStore,
    StaleGenerationError,
)
from nomorals.agents.help_protocol import (
    HelpError,
    HelpStore,
    help_requests_for_ref,
    request_help,
)
from nomorals.agents.run_ticket import ticket_for
from nomorals.browser.jobs import (
    BrowserJobManager,
    BrowserRunJob,
    BrowserRunRecordStore,
)
from nomorals.core.cancellation import (
    CANCEL_MAPPINGS,
    RECEIPT_STATES,
    CancelReceiptStore,
    UnmappedCancelState,
    cancel_with_receipt,
    map_cancel_state,
)
from nomorals.core.errors import NotFound
from nomorals.core.jobs import JobRegistry
from nomorals.media_edit.jobs import JobManager, MediaEditError
from nomorals.missions.mission import Mission, MissionStore
from nomorals.storage.db import Database


def _file_db(test: unittest.TestCase) -> Database:
    tmp = tempfile.TemporaryDirectory(
        dir=os.environ.get("NM_TEST_TMPDIR") or None)
    test.addCleanup(tmp.cleanup)
    db = Database(os.path.join(tmp.name, "p5.db"))
    db.migrate()
    return db


def _contract(store: DelegationStore, **kw):
    base = dict(goal="do the job", worker="worker-wren",
                verifier="verifier-amy", witness="witness-jo",
                required_evidence=("test run",))
    base.update(kw)
    return store.create_contract(**base)


# ── coremind fixture (the P4 suite's shape) ──────────────────────


class _Settings:
    def __init__(self, root, mind=None):
        self._root = root
        self.mind = mind

    def resolve(self, key):
        return os.path.join(self._root, key)


class _Context:
    def __init__(self, root, mind=None):
        self.settings = _Settings(root, mind)
        self.extras = {}
        self.memory = None
        self.router = None
        self.db = None


def _mind_cfg(max_inflight=1, timeout=0.05, overdue=0):
    return SimpleNamespace(max_inflight=max_inflight,
                           inflight_acquire_timeout_s=timeout,
                           queue_overdue_s=overdue)


def _make_mind(test, cfg=None):
    tmp = tempfile.TemporaryDirectory(
        dir=os.environ.get("NM_TEST_TMPDIR") or None)
    test.addCleanup(tmp.cleanup)
    return CoreMind(_Context(tmp.name, cfg))


def _intent(target):
    return Intent(kind="research", confidence=1.0, target=target,
                  route="research_swarm")


def _wait_for(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


# ── delegation: the facade + the cancel terminal ─────────────────


class DelegationCancelTests(unittest.TestCase):
    def setUp(self):
        self.db = _file_db(self)
        self.store = DelegationStore(self.db)
        self.receipts = CancelReceiptStore(self.db)

    def test_live_contract_cancels_terminal_with_receipt(self):
        contract = _contract(self.store)
        evidence = self.store.submit_evidence(
            contract.id, kind="test run", claim="tests pass",
            producer="worker-wren")
        challenge = self.store.challenge(
            contract.id, challenger="verifier-amy",
            claim="the run looks staged")
        out = cancel_with_receipt(
            self.db, f"contract:{contract.id}", requested_by="owner",
            reason="superseded", delegation=self.store)
        self.assertIsNone(out["receipt_error"])
        receipt = out["receipt"]
        self.assertEqual(receipt["receipt_state"], "confirmed")
        self.assertEqual(receipt["native_state"], "cancelled")
        self.assertEqual(receipt["organ"], "delegation")
        self.assertEqual(receipt["requested_by"], "owner")
        fresh = self.store.get_contract(contract.id)
        self.assertEqual(fresh.status, "cancelled")
        self.assertTrue(fresh.terminal)
        self.assertGreater(fresh.completed_at, 0)
        # The cancellation claim names the claims left open, and
        # the append-only rule holds: nothing was rewritten.
        detail = receipt["detail"]
        self.assertIn("cancelled by owner: superseded", detail)
        self.assertIn("1 skeptic challenge(s)", detail)
        self.assertIn("1 unverified evidence claim(s)", detail)
        claims = {c.id: c for c in self.store.claims(contract.id)}
        self.assertEqual(claims[challenge.id].verdict, "uncertain")
        self.assertEqual(claims[evidence.id].verdict, "unverified")
        kinds = [c.kind for c in self.store.claims(contract.id)]
        self.assertIn("cancellation", kinds)
        # A cancelled contract no longer counts as open.
        self.assertEqual(self.store.open_contracts(), [])

    def test_stale_generation_cancel_refused_and_receipted(self):
        contract = _contract(self.store)
        with self.assertRaises(StaleGenerationError):
            cancel_with_receipt(
                self.db, f"contract:{contract.id}",
                requested_by="owner",
                generation=contract.generation + 5,
                delegation=self.store)
        receipt = self.receipts.latest_for_ref(
            f"contract:{contract.id}")
        self.assertIsNotNone(receipt)
        self.assertEqual(receipt["receipt_state"], "refused")
        self.assertEqual(receipt["native_state"], "stale_generation")
        self.assertEqual(self.store.get_contract(contract.id).status,
                         "open")

    def test_completed_and_waived_contracts_refuse_named(self):
        done = _contract(self.store)
        claim = self.store.submit_evidence(
            done.id, kind="test run", claim="tests pass",
            producer="worker-wren")
        self.store.verify_claim(claim.id, validator="verifier-amy",
                                verdict="supported")
        self.store.sign_off(done.id, verifier="verifier-amy")
        decision = self.store.complete(done.id, by="verifier-amy")
        self.assertTrue(decision.ok)
        with self.assertRaises(DelegationError):
            cancel_with_receipt(self.db, f"contract:{done.id}",
                                delegation=self.store)
        receipt = self.receipts.latest_for_ref(f"contract:{done.id}")
        self.assertEqual(receipt["receipt_state"], "refused")
        self.assertEqual(receipt["native_state"], "completed")

        waived = _contract(self.store)
        self.store.waive(waived.id, by="verifier-amy",
                         reason="not needed")
        with self.assertRaises(DelegationError):
            cancel_with_receipt(self.db, f"contract:{waived.id}",
                                delegation=self.store)
        receipt = self.receipts.latest_for_ref(
            f"contract:{waived.id}")
        self.assertEqual(receipt["receipt_state"], "refused")
        self.assertEqual(receipt["native_state"], "waived")

    def test_unknown_contract_receipts_not_found(self):
        with self.assertRaises(DelegationError):
            cancel_with_receipt(self.db, "contract:nope-123",
                                delegation=self.store)
        receipt = self.receipts.latest_for_ref("contract:nope-123")
        self.assertEqual(receipt["receipt_state"], "not_found")

    def test_late_completion_cannot_move_a_cancelled_contract(self):
        contract = _contract(self.store)
        cancel_with_receipt(self.db, f"contract:{contract.id}",
                            requested_by="owner",
                            delegation=self.store)
        decision = self.store.complete(contract.id, by="worker-wren")
        self.assertFalse(decision.ok)
        self.assertIn("cancelled", decision.reasons[0])
        self.assertIn("terminal", decision.reasons[0])
        self.assertEqual(
            self.store.get_contract(contract.id).status, "cancelled")

    def test_second_cancel_of_cancelled_contract_receipts_confirmed(
            self):
        contract = _contract(self.store)
        cancel_with_receipt(self.db, f"contract:{contract.id}",
                            delegation=self.store)
        with self.assertRaises(DelegationError):
            cancel_with_receipt(self.db, f"contract:{contract.id}",
                                delegation=self.store)
        rows = self.receipts.for_ref(f"contract:{contract.id}")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1]["receipt_state"], "confirmed")
        self.assertEqual(rows[-1]["native_state"], "cancelled")


# ── browser / media / jobs / coremind / mission receipts ─────────


class BrowserFacadeTests(unittest.TestCase):
    def test_unhonored_run_receipts_requested_then_unhonored(self):
        db = _file_db(self)
        mgr = BrowserJobManager(db=db)
        release = threading.Event()
        self.addCleanup(release.set)

        def runner(job):
            release.wait(5)
            return {"stopped_reason": "goal met", "steps": 4}

        job, _ = mgr.start("https://example.com/z", "ignore cancels",
                           runner)
        out = cancel_with_receipt(db, f"browser:{job.job_id}",
                                  browser_manager=mgr)
        self.assertEqual(out["receipt"]["receipt_state"], "requested")
        release.set()
        self.assertTrue(job.wait(5))
        self.assertTrue(mgr.flush_records())
        row = BrowserRunRecordStore(db).get(job.job_id)
        self.assertEqual(row["cancel_state"], "unhonored")
        # A second cancel reads the record's final word.
        out2 = cancel_with_receipt(db, f"browser:{job.job_id}",
                                   browser_manager=mgr)
        self.assertEqual(out2["receipt"]["receipt_state"],
                         "unhonored")
        self.assertEqual(out2["receipt"]["native_state"],
                         "already_terminal:unhonored")

    def test_queued_cancel_receipt_and_unknown_run(self):
        db = _file_db(self)
        mgr = BrowserJobManager(db=db)
        job = BrowserRunJob(job_id="br-p5queued",
                            url="https://example.com",
                            goal="never runs")
        job._on_change = lambda: mgr._record_job(job)
        mgr._jobs[job.job_id] = job
        mgr._record_job(job)
        out = cancel_with_receipt(db, f"browser:{job.job_id}",
                                  browser_manager=mgr)
        self.assertEqual(out["receipt"]["receipt_state"], "requested")
        # The manager's own run pre-check confirms the queued
        # cancel: the runner is never invoked (the P1 flow).
        called = []
        mgr._run(job, lambda j: called.append(True) or {})
        self.assertEqual(called, [])
        self.assertEqual(job.status, "cancelled")
        self.assertTrue(mgr.flush_records())
        row = BrowserRunRecordStore(db).get(job.job_id)
        self.assertEqual(row["cancel_state"], "confirmed")
        out = cancel_with_receipt(db, "browser:br-ghost",
                                  browser_manager=mgr)
        self.assertEqual(out["receipt"]["receipt_state"], "not_found")


class MediaFacadeTests(unittest.TestCase):
    def test_queued_confirmed_running_requested_unknown_raises(self):
        db = _file_db(self)
        mgr = JobManager(db=db)
        gate = threading.Event()
        self.addCleanup(gate.set)

        def blocker(progress):
            while not gate.is_set():
                progress(0.5)
                time.sleep(0.02)
            return {"output": "/m/blocker.mp4"}

        first = mgr.submit("video", "blocker", blocker)
        second = mgr.submit("video", "queued one",
                            lambda progress: {"output": "/m/t.mp4"})
        deadline = time.time() + 5
        while mgr.status(first)["status"] != "running":
            self.assertLess(time.time(), deadline)
            time.sleep(0.02)
        out = cancel_with_receipt(db, f"render:{second}",
                                  media_manager=mgr)
        self.assertEqual(out["receipt"]["receipt_state"], "confirmed")
        self.assertEqual(out["receipt"]["native_state"],
                         "cancelled_queued")
        out = cancel_with_receipt(db, f"render:{first}",
                                  media_manager=mgr)
        self.assertEqual(out["receipt"]["receipt_state"], "requested")
        # The blocker stops at its next progress report — the
        # gate stays shut until it has landed cancelled.
        info = mgr.wait(first, timeout=10)
        self.assertEqual(info["status"], "cancelled")
        gate.set()
        with self.assertRaises(MediaEditError):
            cancel_with_receipt(db, "render:mj-ghost",
                                media_manager=mgr)
        receipt = CancelReceiptStore(db).latest_for_ref(
            "render:mj-ghost")
        self.assertEqual(receipt["receipt_state"], "not_found")


class JobsFacadeTests(unittest.TestCase):
    def test_flag_requested_terminal_refused_unknown_not_found(self):
        db = _file_db(self)
        registry = JobRegistry()
        gate = threading.Event()
        self.addCleanup(gate.set)
        job = registry.submit(lambda: gate.wait(30), tool="fetch")
        out = cancel_with_receipt(db, f"job:{job.id}",
                                  registry=registry)
        # Before P5 this flag was the whole record and no receipt
        # existed; the facade receipt is now the durable record.
        self.assertEqual(out["receipt"]["receipt_state"], "requested")
        self.assertEqual(out["receipt"]["native_state"], "requested")
        gate.set()
        self.assertTrue(_wait_for(
            lambda: registry.poll(job.id).status == "done"))
        out = cancel_with_receipt(db, f"job:{job.id}",
                                  registry=registry)
        self.assertEqual(out["receipt"]["receipt_state"], "refused")
        out = cancel_with_receipt(db, "job:job-ghost",
                                  registry=registry)
        self.assertEqual(out["receipt"]["receipt_state"], "not_found")


class CoremindFacadeTests(unittest.TestCase):
    def test_queued_confirmed_active_refused_unknown_not_found(self):
        db = _file_db(self)
        mind = _make_mind(self, _mind_cfg(max_inflight=1))
        gate = threading.Event()
        self.addCleanup(gate.set)

        def job_a():
            gate.wait(timeout=5)
            return "A done"

        id_a = mind._new_job(_intent("alpha"))
        mind._send_async("k:console", job_a, id_a, "started A",
                         kind="research")
        self.assertTrue(_wait_for(lambda: mind._inflight_now == 1))
        id_b = mind._new_job(_intent("bravo"))
        mind._send_async("k:console", lambda: "B done", id_b,
                         "started B", kind="research")
        self.assertEqual(mind.job_record(id_b)["status"], "queued")

        out = cancel_with_receipt(db, f"mind:{id_b}", coremind=mind)
        self.assertEqual(out["receipt"]["receipt_state"], "confirmed")
        self.assertIn("never run", out["receipt"]["detail"])
        self.assertEqual(mind.job_record(id_b)["status"], "cancelled")

        out = cancel_with_receipt(db, f"mind:{id_a}", coremind=mind)
        self.assertEqual(out["receipt"]["receipt_state"], "refused")
        self.assertEqual(out["receipt"]["native_state"], "active")

        out = cancel_with_receipt(db, "mind:ghost", coremind=mind)
        self.assertEqual(out["receipt"]["receipt_state"], "not_found")
        # Drain before the temp root goes away: job A's thread
        # writes its state file as it finishes (the P4 discipline).
        gate.set()
        self.assertTrue(_wait_for(
            lambda: mind.job_record(id_a)["status"] == "done"
            and mind._inflight_now == 0))


class MissionFacadeTests(unittest.TestCase):
    def test_pending_confirmed_done_refused_unknown_not_found(self):
        db = _file_db(self)
        store = MissionStore(db)
        mission = store.create(Mission(goal="survey", name="survey"))
        out = cancel_with_receipt(db, f"mission:{mission.id}",
                                  requested_by="owner",
                                  missions=store)
        self.assertEqual(out["receipt"]["receipt_state"], "confirmed")
        self.assertEqual(store.get(mission.id).status, "cancelled")

        done = store.create(Mission(goal="finished", name="fin"))
        store.set_status(done.id, "done")
        out = cancel_with_receipt(db, f"mission:{done.id}",
                                  missions=store)
        self.assertEqual(out["receipt"]["receipt_state"], "refused")
        # A finished mission's record is history — never rewritten.
        self.assertEqual(store.get(done.id).status, "done")

        with self.assertRaises(NotFound):
            cancel_with_receipt(db, "mission:ghost", missions=store)
        receipt = CancelReceiptStore(db).latest_for_ref(
            "mission:ghost")
        self.assertEqual(receipt["receipt_state"], "not_found")


# ── facade discipline: fail-closed semantics, fail-open cancel ───


class FacadeDisciplineTests(unittest.TestCase):
    def test_mapping_tables_are_complete_and_closed(self):
        for organ, table in CANCEL_MAPPINGS.items():
            self.assertTrue(table, organ)
            for word, state in table.items():
                self.assertIn(state, RECEIPT_STATES, (organ, word))
                self.assertEqual(map_cancel_state(organ, word), state)
        with self.assertRaises(UnmappedCancelState):
            map_cancel_state("media", "exploded")
        with self.assertRaises(UnmappedCancelState):
            map_cancel_state("no-such-organ", "cancelled")

    def test_unmapped_native_state_refuses_receipt_cancel_still_runs(
            self):
        db = _file_db(self)
        calls = []

        class _ExplodingManager:
            def cancel(self, job_id):
                calls.append(job_id)
                return {"id": job_id, "status": "exploded",
                        "started_at": 1.0}

        out = cancel_with_receipt(db, "render:mj-x",
                                  media_manager=_ExplodingManager())
        self.assertEqual(calls, ["mj-x"])  # the cancel executed
        self.assertIsNone(out["receipt"])
        self.assertIn("exploded", out["receipt_error"])
        self.assertEqual(out["native"]["status"], "exploded")
        self.assertIsNone(CancelReceiptStore(db).latest_for_ref(
            "render:mj-x"))

    def test_receipt_failure_never_blocks_the_cancel(self):
        db = _file_db(self)
        mgr = BrowserJobManager(db=db)
        job, _ = mgr.start("https://example.com/q", "quick",
                           lambda j: {"stopped_reason": "met"})
        self.assertTrue(job.wait(5))
        # db=None: the cancel executes through the organ and its
        # native answer comes back; only the receipt is missing,
        # with the reason named.
        out = cancel_with_receipt(None, f"browser:{job.job_id}",
                                  browser_manager=mgr)
        self.assertEqual(out["native"]["state"], "already_terminal")
        self.assertIsNone(out["receipt"])
        self.assertIn("no database handle", out["receipt_error"])

    def test_missing_handle_cancels_nothing_and_says_so(self):
        db = _file_db(self)
        store = DelegationStore(db)
        contract = _contract(store)
        out = cancel_with_receipt(db, f"contract:{contract.id}")
        self.assertIsNone(out["receipt"])
        self.assertIn("no delegation handle", out["receipt_error"])
        self.assertEqual(store.get_contract(contract.id).status,
                         "open")


# ── the help protocol ────────────────────────────────────────────


class HelpProtocolTests(unittest.TestCase):
    def setUp(self):
        self.db = _file_db(self)
        self.help = HelpStore(self.db)

    def test_request_answer_lifecycle(self):
        record = self.help.request_help(
            "contract:c-1", requester="worker-wren",
            kind="needs_input", detail="which corpus wins?",
            helper="owner")
        self.assertEqual(record["state"], "open")
        self.assertEqual(record["kind"], "needs_input")
        answered = self.help.answer_help(
            record["id"], by="owner", answer="the 2026 corpus")
        self.assertEqual(answered["state"], "answered")
        self.assertEqual(answered["answer"], "the 2026 corpus")
        self.assertEqual(answered["answered_by"], "owner")
        self.assertGreater(answered["answered_at"], 0)
        with self.assertRaises(HelpError):
            self.help.answer_help(record["id"], by="owner",
                                  answer="again")

    def test_withdraw_is_the_requesters_alone(self):
        record = request_help(self.db, "mission:m-1",
                              requester="runner", kind="blocked",
                              detail="waiting on the corpus job")
        with self.assertRaises(HelpError):
            self.help.withdraw_help(record["id"], by="someone-else")
        withdrawn = self.help.withdraw_help(record["id"],
                                            by="runner")
        self.assertEqual(withdrawn["state"], "withdrawn")
        rows = help_requests_for_ref(self.db, "mission:m-1")
        self.assertEqual([r["state"] for r in rows], ["withdrawn"])

    def test_invalid_kind_and_ref_are_caller_bugs(self):
        with self.assertRaises(ValueError):
            self.help.request_help("contract:c-1", requester="w",
                                   kind="whatever", detail="x")
        with self.assertRaises(ValueError):
            self.help.request_help("no-colon", requester="w",
                                   kind="blocked", detail="x")
        with self.assertRaises(HelpError):
            self.help.answer_help("help-ghost", by="o", answer="x")


# ── ticket flips ─────────────────────────────────────────────────


class TicketFlipTests(unittest.TestCase):
    def test_contract_cancel_filled_and_receipt_rides(self):
        db = _file_db(self)
        store = DelegationStore(db)
        contract = _contract(store)
        ticket = ticket_for(f"contract:{contract.id}", db)
        self.assertNotIn("cancel", ticket["absent"])
        self.assertEqual(ticket["cancel"]["state"], "none")
        self.assertEqual(ticket["sources"]["cancel"], "delegation")
        cancel_with_receipt(db, f"contract:{contract.id}",
                            requested_by="owner", delegation=store)
        ticket = ticket_for(f"contract:{contract.id}", db)
        self.assertEqual(ticket["cancel"]["state"], "cancelled")
        receipt = ticket["cancel"]["receipt"]
        self.assertEqual(receipt["state"], "confirmed")
        self.assertEqual(receipt["organ"], "delegation")
        self.assertEqual(receipt["requested_by"], "owner")

    def test_browser_receipt_rides_the_ticket(self):
        db = _file_db(self)
        mgr = BrowserJobManager(db=db)
        release = threading.Event()
        self.addCleanup(release.set)

        def runner(job):
            release.wait(5)
            return {"stopped_reason": "goal met", "steps": 1}

        job, _ = mgr.start("https://example.com/t", "ticket ride",
                           runner)
        cancel_with_receipt(db, f"browser:{job.job_id}",
                            browser_manager=mgr)
        self.assertTrue(mgr.flush_records())
        ticket = ticket_for(f"browser:{job.job_id}", db)
        self.assertEqual(ticket["cancel"]["state"], "requested")
        self.assertEqual(ticket["cancel"]["receipt"]["state"],
                         "requested")
        release.set()
        self.assertTrue(job.wait(5))

    def test_ticket_shows_help_open_then_answered(self):
        db = _file_db(self)
        store = DelegationStore(db)
        contract = _contract(store)
        ref = f"contract:{contract.id}"
        record = request_help(db, ref, requester="worker-wren",
                              kind="needs_review",
                              detail="sanity-check the plan")
        ticket = ticket_for(ref, db)
        shown = ticket["detail"]["help_requests"]
        self.assertEqual(len(shown), 1)
        self.assertEqual(shown[0]["state"], "open")
        self.assertEqual(shown[0]["kind"], "needs_review")
        HelpStore(db).answer_help(record["id"], by="verifier-amy",
                                  answer="plan holds")
        ticket = ticket_for(ref, db)
        shown = ticket["detail"]["help_requests"]
        self.assertEqual(shown[0]["state"], "answered")
        self.assertEqual(shown[0]["answer"], "plan holds")
        self.assertEqual(shown[0]["answered_by"], "verifier-amy")


if __name__ == "__main__":
    unittest.main()
