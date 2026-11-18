"""Lane 74: /grill-me + /plan cancel.

Failing-first: on pristine be1ca4e6 there is no ``PlanStore.cancel``,
no ``DelegationManager.cancel_run``, no ``nomorals.agents.grill``,
and no ``grill-me`` control verb — every test here fails at
collection or on a missing attribute until the build lands.
"""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents.delegation import DelegationError, DelegationStore
from nomorals.agents.delegation_manager import DelegationManager
from nomorals.agents.plan.errors import PlanError
from nomorals.agents.plan.store import PlanStore
from nomorals.commitments.ledger import CommitmentLedger
from nomorals.storage.db import Database

NOW = 1_790_000_000.0  # fixed clock for stable age strings


# ── delegation fakes (same shape as test_delegation_manager) ────────

def _db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


class _FakeAgent:
    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def __call__(self, role, input_dict):
        with self._lock:
            self.calls.append({"role": role,
                               "input": dict(input_dict or {})})
            index = len(self.calls) - 1
        behavior = self.script[min(index, len(self.script) - 1)]
        kind = behavior[0]
        if kind == "ok":
            return {"ok": True, "result": behavior[1]}
        if kind == "fail":
            return {"ok": False, "error": behavior[1]}
        if kind == "block":
            behavior[1].wait(timeout=30)
            return {"ok": True, "result": {"unblocked": True}}
        raise AssertionError(f"unknown behavior {behavior!r}")


class _FakeNotifier:
    def __init__(self):
        self.published: list[dict] = []

    def __call__(self, kind, title, body, **_kw):
        self.published.append({"kind": kind, "title": title,
                               "body": body})
        return {"delivered": True, "delivery_state": "sent"}


def _manager(db, fake, notifier, **kw):
    store = DelegationStore(db)
    return DelegationManager(store, db=db, run_agent=fake,
                             notifier=notifier, **kw)


# ── plan-store helpers ──────────────────────────────────────────────

def _saved_plan(store: PlanStore, task: str = "fix the retry loop",
                runs=None) -> dict:
    return store.save_plan(
        task=task, survey={}, plan={}, runs=list(runs or []),
        unassigned=[], plan_error="", chat_key="")


def _completed_run() -> dict:
    return {"id": "dr-deadbeef0001", "role": "tester",
            "status": "completed"}


class PlanCancelTests(unittest.TestCase):
    def test_cancel_running_plan_marks_cancelled_and_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            store = PlanStore(
                path=home / "data" / "plans.json",
                plans_dir=home / "plans")
            record = _saved_plan(store)
            self.assertEqual(record["state"], "running")
            outcome = store.cancel(record["id"], by="owner")
            self.assertFalse(outcome["already_cancelled"])
            self.assertEqual(outcome["record"]["state"], "cancelled")
            self.assertIsNotNone(
                outcome["record"].get("cancelled_at"))
            # Durable: a fresh store over the same file still reads it.
            reopened = PlanStore(
                path=home / "data" / "plans.json",
                plans_dir=home / "plans")
            self.assertEqual(
                reopened.get_plan(record["id"])["state"], "cancelled")
            rows = reopened.list_plans()
            self.assertEqual([r["state"] for r in rows], ["cancelled"])

    def test_cancel_twice_is_honest(self):
        store = PlanStore(path=None, plans_dir=None)
        record = _saved_plan(store)
        first = store.cancel(record["id"], by="owner")
        self.assertFalse(first["already_cancelled"])
        second = store.cancel(record["id"], by="owner")
        self.assertTrue(second["already_cancelled"])
        self.assertEqual(second["record"]["state"], "cancelled")

    def test_cancel_unknown_plan_raises(self):
        store = PlanStore(path=None, plans_dir=None)
        with self.assertRaises(PlanError) as caught:
            store.cancel("pl-doesnotexist", by="owner")
        self.assertIn("pl-doesnotexist", str(caught.exception))

    def test_cancel_done_plan_refused(self):
        store = PlanStore(path=None, plans_dir=None)
        record = _saved_plan(store, runs=[_completed_run()])
        done = store.refresh_state(
            record["id"],
            lambda run_id: SimpleNamespace(status="completed"))
        self.assertEqual(done["state"], "done")
        with self.assertRaises(PlanError) as caught:
            store.cancel(record["id"], by="owner")
        self.assertIn("already done", str(caught.exception))

    def test_derive_keeps_cancelled_despite_live_runs(self):
        from nomorals.agents.plan.store import derive_plan_state

        store = PlanStore(path=None, plans_dir=None)
        record = _saved_plan(store)
        store.cancel(record["id"], by="owner")
        cancelled = store.get_plan(record["id"])
        # Even with a live run behind it, a cancelled plan stays
        # cancelled — refresh must not resurrect it as running/failed.
        live = SimpleNamespace(status="running")
        self.assertEqual(
            derive_plan_state(cancelled, lambda _rid: live),
            "cancelled")
        refreshed = store.refresh_state(
            record["id"], lambda _rid: live)
        self.assertEqual(refreshed["state"], "cancelled")

    def test_cancel_stops_live_runs_through_manager(self):
        db = _db()
        event = threading.Event()
        fake = _FakeAgent([("block", event)])
        manager = _manager(db, fake, _FakeNotifier())
        try:
            run = manager.spawn(role="tester",
                                goal="cover the new retry path")
            self.assertIn(manager.get_run(run.id).status,
                          ("running", "steering"))
            store = PlanStore(path=None, plans_dir=None)
            record = _saved_plan(
                store, runs=[{"id": run.id, "role": "tester",
                              "status": "running"}])
            outcome = store.cancel(record["id"], by="owner",
                                   manager=manager)
            self.assertEqual(outcome["runs_cancelled"], [run.id])
            self.assertEqual(outcome["runs_terminal"], [])
            self.assertEqual(manager.get_run(run.id).status,
                             "cancelled")
            self.assertEqual(outcome["record"]["state"], "cancelled")
        finally:
            event.set()
            manager.shutdown(wait=False)


class DelegationCancelRunTests(unittest.TestCase):
    def test_cancel_run_claims_cancelled_and_fails_contract(self):
        db = _db()
        event = threading.Event()
        fake = _FakeAgent([("block", event)])
        manager = _manager(db, fake, _FakeNotifier())
        try:
            run = manager.spawn(role="tester",
                                goal="cover the new retry path")
            outcome = manager.cancel_run(run.id, by="owner")
            self.assertTrue(outcome["cancelled"])
            self.assertEqual(outcome["status"], "cancelled")
            live = manager.get_run(run.id)
            self.assertEqual(live.status, "cancelled")
            self.assertIn("owner", live.error)
            contract = DelegationStore(db).get_contract(
                run.contract_id)
            self.assertIsNotNone(contract)
            self.assertEqual(contract.status, "failed")
        finally:
            event.set()
            manager.shutdown(wait=False)

    def test_cancel_run_terminal_is_noop(self):
        db = _db()
        fake = _FakeAgent([("ok", {"done": True})])
        manager = _manager(db, fake, _FakeNotifier())
        try:
            run = manager.spawn(role="tester",
                                goal="cover the new retry path")
            done = manager.wait(run.id, timeout=10)
            self.assertEqual(done.status, "completed")
            outcome = manager.cancel_run(run.id, by="owner")
            self.assertFalse(outcome["cancelled"])
            self.assertEqual(outcome["status"], "completed")
            self.assertEqual(manager.get_run(run.id).status,
                             "completed")
        finally:
            manager.shutdown(wait=False)

    def test_cancel_run_unknown_raises(self):
        db = _db()
        manager = _manager(db, _FakeAgent([]), _FakeNotifier())
        try:
            with self.assertRaises(DelegationError) as caught:
                manager.cancel_run("dr-doesnotexist", by="owner")
            self.assertIn("dr-doesnotexist", str(caught.exception))
        finally:
            manager.shutdown(wait=False)


# ── grill ──────────────────────────────────────────────────────────

def _grill_stores(now: float = NOW):
    """An in-memory plan store + commitment ledger with real dirt."""
    plans = PlanStore(path=None, plans_dir=None)
    old = _saved_plan(plans, task="fix the whatsapp bridge retry path")
    # Age the running plan 12 days by rewriting its created stamp.
    plans._plans[old["id"]]["created_at"] = now - 12 * 86400
    failed = _saved_plan(plans, task="migrate the old tracker")
    plans._plans[failed["id"]]["state"] = "failed"
    plans._plans[failed["id"]]["plan_error"] = "survey unavailable"
    done = _saved_plan(plans, task="ship the dunning driver")
    plans._plans[done["id"]]["state"] = "done"
    ledger = CommitmentLedger(path=None)
    overdue = ledger.add(
        what="send the deck to Alex", promisee="Alex",
        deadline=now - 3 * 86400, ts=now - 10 * 86400)
    ledger.add(what="call mum back", promisee="Mum",
               deadline=now + 3600, ts=now - 86400)
    kept = ledger.add(what="pay the cleaner", promisee="Cleaner",
                      deadline=now - 86400, ts=now - 2 * 86400)
    ledger.mark_kept(kept.id, "paid on time")
    broken = ledger.add(what="finish the invoice", promisee="Sam",
                        deadline=now - 2 * 86400, ts=now - 5 * 86400)
    ledger.mark_broken(broken.id, "never sent")
    return plans, ledger, overdue


class GrillEvidenceTests(unittest.TestCase):
    def test_grill_names_real_plans_and_commitments(self):
        from nomorals.agents.grill import gather_grill_evidence, render_grill

        plans, ledger, _overdue = _grill_stores()
        evidence = gather_grill_evidence(
            plans=plans, commitments=ledger, now=NOW)
        self.assertEqual(evidence["plans"]["total"], 3)
        self.assertEqual(evidence["plans"]["running"], 1)
        self.assertEqual(evidence["plans"]["failed"], 1)
        self.assertEqual(evidence["plans"]["done"], 1)
        self.assertEqual(
            evidence["plans"]["oldest_running"]["task"],
            "fix the whatsapp bridge retry path")
        self.assertEqual(
            len(evidence["commitments"]["overdue"]), 1)
        self.assertEqual(
            evidence["commitments"]["overdue"][0]["what"],
            "send the deck to Alex")
        self.assertEqual(
            evidence["commitments"]["reliability"]["kept"], 1)
        self.assertEqual(
            evidence["commitments"]["reliability"]["broken"], 1)
        text = render_grill(evidence)
        self.assertIn("fix the whatsapp bridge retry path", text)
        self.assertIn("send the deck to Alex", text)
        self.assertIn("12 days", text)
        self.assertIn("3 days", text)
        # The kept rate is real arithmetic, not a canned line.
        self.assertIn("1 of 2", text)

    def test_grill_empty_is_honest(self):
        from nomorals.agents.grill import gather_grill_evidence, render_grill

        evidence = gather_grill_evidence(
            plans=PlanStore(path=None, plans_dir=None),
            commitments=CommitmentLedger(path=None), now=NOW)
        text = render_grill(evidence)
        self.assertIn("nothing to grill", text.lower())
        self.assertNotIn("12 days", text)
        self.assertNotIn("overdue", text.lower())

    def test_grill_section_filter(self):
        from nomorals.agents.grill import gather_grill_evidence, render_grill

        plans, ledger, _overdue = _grill_stores()
        evidence = gather_grill_evidence(
            plans=plans, commitments=ledger, now=NOW)
        plans_only = render_grill(evidence, section="plans")
        self.assertIn("fix the whatsapp bridge", plans_only)
        self.assertNotIn("send the deck", plans_only)
        commits_only = render_grill(evidence, section="commitments")
        self.assertIn("send the deck", commits_only)
        self.assertNotIn("fix the whatsapp bridge", commits_only)
        with self.assertRaises(ValueError):
            render_grill(evidence, section="nope")


class GrillRegistrationTests(unittest.TestCase):
    def test_grill_me_registered_owner_only(self):
        from nomorals.social.chat.control import COMMAND_DETAILS, CONTROL_COMMANDS, PUBLIC_COMMANDS
        from nomorals.social.chat.personal_commands import PERSONAL_OWNER_COMMANDS

        self.assertIn("grill-me", CONTROL_COMMANDS)
        self.assertEqual(CONTROL_COMMANDS["grill-me"], (0, None))
        self.assertIn("grill-me", COMMAND_DETAILS)
        self.assertIn("/grill-me",
                      COMMAND_DETAILS["grill-me"]["usage"])
        # Owner-only: neither public catalog nor the pinned 48.
        self.assertNotIn("grill-me", PUBLIC_COMMANDS)
        self.assertNotIn("grill-me", PERSONAL_OWNER_COMMANDS)
        self.assertEqual(len(PERSONAL_OWNER_COMMANDS), 48)

    def test_plan_usage_mentions_cancel(self):
        from nomorals.social.chat.control import COMMAND_DETAILS

        self.assertIn("cancel",
                      COMMAND_DETAILS["plan"]["usage"])
        self.assertIn("/plan cancel <id>",
                      COMMAND_DETAILS["plan"]["usage"])


if __name__ == "__main__":
    unittest.main()
