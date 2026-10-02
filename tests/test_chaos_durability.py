"""Chaos durability: inject faults, assert the guards catch them.

Style mirrors ``tests/test_fault_injection.py`` (scripted faults, real
subsystems, honest assertions).  Every injection below is paired with a
guard that fires — a chaos test where nothing checks the damage is just
vandalism.

* scheduler/queue — dropped and duplicated items vs ``reconcile`` /
  ``detect_duplicates``
* policy — a flipped grant vs the audit-trail ``verify_decision``
* artifact store — a corrupted blob write vs ``ArtifactStore.verify``
* mission DAG — a task dying mid-run vs the executor's state machine +
  ``TaskGraph.audit``
"""

import shutil
import tempfile
import threading
import unittest
from pathlib import Path

from nomorals.agents.runtime import HybridExecutor
from nomorals.core.policy import CapabilitySet, Policy, PolicyDecision
from nomorals.core.tasks import (
    Task,
    TaskGraph,
    TaskKind,
    TaskState,
)
from nomorals.storage.artifacts import ArtifactStore
from nomorals.storage.blob import BlobStore
from nomorals.storage.db import Database
from nomorals.storage.queue import WorkQueue


def make_db(test=None):
    tmp = tempfile.mkdtemp(prefix="chaos-")
    if test is not None:
        test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    db = Database(str(Path(tmp) / "chaos.db"))
    db.migrate()
    if test is not None:
        test.addCleanup(db.close)
    return db, tmp


class TestQueueChaos(unittest.TestCase):
    """(a) scheduler/queue: drop and duplicate a queued item."""

    def setUp(self):
        self.db, self.tmp = make_db(self)
        self.queue = WorkQueue(self.db, backoff_base=0.0, backoff_cap=0.0)

    def test_dropped_item_caught_by_reconcile(self):
        receipts = [
            self.queue.enqueue("jobs", {"n": 1}),
            self.queue.enqueue("jobs", {"n": 2}),
            self.queue.enqueue("jobs", {"n": 3}),
        ]
        # The fault: a queued row vanishes (lost message, crashed broker).
        self.db.execute("DELETE FROM work_queue WHERE id = ?", (receipts[1],))

        missing = self.queue.reconcile(receipts)

        self.assertEqual(missing, [receipts[1]],
                         "reconcile must name the dropped job")
        # And the survivors still process normally.
        jobs = self.queue.lease("jobs", worker="w1", batch=10)
        self.assertEqual({j.id for j in jobs}, {receipts[0], receipts[2]})

    def test_reconcile_clean_when_nothing_dropped(self):
        receipts = [self.queue.enqueue("jobs", {"n": i}) for i in range(3)]
        self.assertEqual(self.queue.reconcile(receipts), [])

    def test_duplicate_delivery_caught_by_detector(self):
        payload = {"url": "https://example.com/once"}
        first = self.queue.enqueue("fetch", payload)
        # The fault: the same work enqueued twice (at-least-once redelivery
        # without idempotency).
        second = self.queue.enqueue("fetch", payload)

        dups = self.queue.detect_duplicates("fetch")

        self.assertEqual(len(dups), 1)
        self.assertEqual(set(dups[0]["ids"]), {first, second})
        # Completed jobs are history, not duplicates.
        self.queue.complete(first)
        self.queue.complete(second)
        self.assertEqual(self.queue.detect_duplicates("fetch"), [])


class TestPolicyChaos(unittest.TestCase):
    """(b) policy decisions: a flipped grant vs the audit trail."""

    def setUp(self):
        self.policy = Policy(default_grant=CapabilitySet.all())

    def _flipped_check(self, policy):
        """Wrap Policy.check to invert the verdict after it is recorded."""
        original = Policy.check

        def flipped(self, capability, **kwargs):
            decision = original(self, capability, **kwargs)
            return PolicyDecision(
                allowed=not decision.allowed,
                reason=decision.reason,
                capability=decision.capability,
                actor=decision.actor,
                needs_confirmation=decision.needs_confirmation,
                audit_id=decision.audit_id,
            )

        Policy.check = flipped
        self.addCleanup(setattr, Policy, "check", original)

    def test_flipped_grant_caught_by_audit_verification(self):
        clean = self.policy.check("fs.read", actor="tester")
        self.assertTrue(self.policy.verify_decision(clean),
                        "an honest decision verifies")

        self._flipped_check(self.policy)
        tampered = self.policy.check("fs.read", actor="tester")

        self.assertFalse(tampered.allowed,
                         "the injection flipped the grant to deny")
        self.assertFalse(self.policy.verify_decision(tampered),
                         "guard must catch the flipped decision")

    def test_flipped_deny_caught_too(self):
        strict = Policy(default_grant=CapabilitySet.none())
        self._flipped_check(strict)
        tampered = strict.check("fs.delete", actor="mallory")
        # An honest evaluation denies; the flip claims a grant.
        self.assertTrue(tampered.allowed)
        self.assertFalse(strict.verify_decision(tampered))

    def test_decision_without_audit_id_does_not_verify(self):
        forged = PolicyDecision(allowed=True, capability="fs.read",
                                actor="mallory")
        self.assertFalse(self.policy.verify_decision(forged))


class TestArtifactChaos(unittest.TestCase):
    """(c) artifact store: a failed blob write vs read-back verification."""

    def setUp(self):
        self.db, self.tmp = make_db(self)
        self.blobs = BlobStore(self.db, Path(self.tmp) / "blobs")
        self.store = ArtifactStore(self.db, self.blobs)

    def test_clean_write_verifies(self):
        art = self.store.put_text("hello golden", creator="test")
        self.assertTrue(self.store.verify(art.id))
        self.assertEqual(self.store.read_text(art.id), "hello golden")

    def test_corrupted_blob_write_caught_by_verify(self):
        original_put = self.blobs.put_bytes

        def corrupt_write(data, **kwargs):
            info = original_put(data, **kwargs)
            # The fault: bytes on disk no longer match the recorded hash
            # (torn write, disk corruption, truncated stream).
            target = self.blobs.path_for(info.sha256, info.compressed)
            target.write_bytes(b"\x00" * 16)
            return info

        self.blobs.put_bytes = corrupt_write
        try:
            art = self.store.put_text("important bytes", creator="test")
        finally:
            self.blobs.put_bytes = original_put

        self.assertFalse(self.store.verify(art.id),
                         "guard must catch the corrupted blob write")

    def test_verify_unknown_artifact_is_false(self):
        self.assertFalse(self.store.verify("no-such-artifact"))


def _io_task(name, fn, **kw):
    return Task(name=name, fn=fn, kind=TaskKind.IO, **kw)


class TestMissionDagChaos(unittest.TestCase):
    """(d) mission DAG execution: a task that dies mid-run."""

    def test_mid_run_death_fails_task_and_skips_dependents(self):
        side_effects = []

        def die_mid_run():
            side_effects.append("partial")  # torn side effect, then death
            raise RuntimeError("worker died mid-run")

        graph = TaskGraph(name="mission-dag")
        plan = graph.add(_io_task("plan", lambda: "plan-ok"))
        execute = graph.add(_io_task("execute", die_mid_run), depends_on=[plan])
        verify = graph.add(_io_task("verify", lambda: "verify-ok"),
                           depends_on=[execute])

        with HybridExecutor(threads=2, use_processes=False) as ex:
            report = ex.run(graph)

        # Existing guard: the executor's state machine.
        self.assertEqual(graph.tasks[execute.id].state, TaskState.FAILED)
        self.assertIn("died mid-run", graph.tasks[execute.id].error)
        self.assertEqual(graph.tasks[verify.id].state, TaskState.SKIPPED)
        self.assertEqual(graph.tasks[plan.id].state, TaskState.DONE)
        self.assertEqual(report.failed, 1)
        self.assertEqual(report.skipped, 1)
        # New guard: the run's bookkeeping is honest — no silent corruption.
        self.assertEqual(graph.audit(), [])
        self.assertEqual(side_effects, ["partial"])

    def test_idempotent_retry_recovers_after_mid_run_death(self):
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("died mid-run")
            return "recovered"

        graph = TaskGraph(name="retry-dag")
        task = _io_task("flaky", flaky, retries=1)
        graph.add(task)
        with HybridExecutor(threads=2, use_processes=False) as ex:
            ex.run(graph)

        self.assertEqual(task.state, TaskState.DONE)
        self.assertEqual(task.result, "recovered")
        self.assertEqual(graph.audit(), [])

    def test_audit_catches_tampered_bookkeeping(self):
        # Prove the guard is not ornamental: hand-corrupt a graph and it fires.
        graph = TaskGraph(name="tampered")
        a = graph.add(_io_task("a", lambda: 1))
        b = graph.add(_io_task("b", lambda: 2), depends_on=[a])
        with HybridExecutor(threads=2, use_processes=False) as ex:
            ex.run(graph)
        self.assertEqual(graph.audit(), [])

        # Tamper: a DONE task with no result (silent data loss).
        graph.tasks[a.id].result = None
        problems = graph.audit()
        self.assertTrue(any("no result" in p for p in problems), problems)

        # Tamper: a task marked DONE despite a failed dependency.
        graph.tasks[a.id].mark_failed("boom")
        graph.tasks[b.id].state = TaskState.DONE  # lie
        problems = graph.audit()
        self.assertTrue(any("failed dependency" in p for p in problems),
                        problems)


if __name__ == "__main__":
    unittest.main()
