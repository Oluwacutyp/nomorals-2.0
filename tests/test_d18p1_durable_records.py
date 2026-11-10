"""Doc 18 P1 — durable run records for the two in-memory job paths.

Pins: (a) a browser run's record is written at start + terminal and
survives a manager restart (new instance, same store) in its last
recorded state; (b) a record stuck at running, aged, is discovered
by the steward's browser adapter as unknown-outcome, an aged queued
record as overdue; (c) cancel distinguishes requested from
confirmed — a queued run's cancel lands confirmed before its runner
is ever invoked, a cooperative runner's stop lands confirmed, and a
run that ends without honoring the request is recorded UNHONORED,
never as a clean cancel; (d) records never carry credentials
(userinfo / secret query values are redacted at the boundary);
(e) media jobs get the same treatment via media_job_records +
the render adapter; (f) a record-write failure never changes the
run's behavior (fail-open, failure recorded); (g) adapter_health
flips for exactly browser + render — transfers + artifacts stay
unavailable with their reasons.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace

from nomorals.agents.outcome_cases import OutcomeCaseStore
from nomorals.agents.outcome_steward import (
    BrowserAdapter,
    OutcomeSteward,
    RenderAdapter,
    adapter_health,
)
from nomorals.browser.jobs import (
    BrowserJobManager,
    BrowserRunJob,
    BrowserRunRecordStore,
)
from nomorals.core.jobs import JobRegistry
from nomorals.media_edit.jobs import JobManager, MediaJobRecordStore
from nomorals.storage.db import Database

OLD = time.time() - 7200.0


def _file_db(test: unittest.TestCase) -> Database:
    tmp = tempfile.TemporaryDirectory(
        dir=os.environ.get("NM_TEST_TMPDIR") or None)
    test.addCleanup(tmp.cleanup)
    db = Database(os.path.join(tmp.name, "records.db"))
    db.migrate()
    return db


def _browser_snapshot(run_id: str, status: str, *,
                      updated_at: float) -> dict:
    now = time.time()
    return {"run_id": run_id, "url": "https://example.com/a",
            "goal": "check the thing", "status": status, "steps": 2,
            "stopped_reason": "", "final_url": "https://example.com/a",
            "wall_outcome": "no wall", "error": "",
            "cancel_state": "", "cancel_requested_at": 0.0,
            "created_at": now - 7200.0, "started_at": now - 7200.0,
            "finished_at": 0.0, "updated_at": updated_at}


def _media_snapshot(job_id: str, status: str, *,
                    updated_at: float) -> dict:
    now = time.time()
    return {"job_id": job_id, "kind": "video", "label": "trim intro",
            "backend": "", "input_ref": "/media/in.mp4",
            "output_ref": "", "status": status, "error": "",
            "cancel_requested_at": 0.0, "created_at": now - 7200.0,
            "started_at": now - 7200.0, "finished_at": 0.0,
            "updated_at": updated_at}


class BrowserRecordTests(unittest.TestCase):
    def test_record_written_and_survives_manager_restart(self):
        db = _file_db(self)
        mgr = BrowserJobManager(db=db)
        job, dup = mgr.start("https://example.com/x", "find the price",
                             lambda j: {"steps": 3,
                                        "stopped_reason": "goal met",
                                        "final_url":
                                            "https://example.com/x"})
        self.assertFalse(dup)
        self.assertTrue(job.wait(5))
        self.assertTrue(mgr.flush_records())
        store = BrowserRunRecordStore(db)
        row = store.get(job.job_id)
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "done")
        self.assertEqual(row["steps"], 3)
        self.assertEqual(row["goal"], "find the price")
        # A NEW manager over the same store (a restart): the run's
        # last recorded state is still fully readable, and the
        # steward's collect reads it as succeeded.
        BrowserJobManager(db=db)
        case = SimpleNamespace(detail={"run_id": job.job_id})
        items = BrowserAdapter(db).collect(case, time.time())
        self.assertEqual(items[0].establishes, "succeeded")

    def test_crashed_run_discovered_by_steward_adapter(self):
        db = _file_db(self)
        store = BrowserRunRecordStore(db)
        store.record(_browser_snapshot("br-stuck1", "running",
                                       updated_at=OLD))
        store.record(_browser_snapshot("br-stuck2", "queued",
                                       updated_at=OLD))
        store.record(_browser_snapshot("br-fresh", "running",
                                       updated_at=time.time()))
        found = {(c.subject_key, c.reason)
                 for c in BrowserAdapter(db).discover(time.time())}
        self.assertIn(("browser:br-stuck1", "unknown-outcome"), found)
        self.assertIn(("browser:br-stuck2", "overdue"), found)
        self.assertNotIn(("browser:br-fresh", "unknown-outcome"), found)
        self.assertNotIn(("browser:br-fresh", "overdue"), found)

    def test_cancel_queued_lands_confirmed_before_runner_runs(self):
        db = _file_db(self)
        mgr = BrowserJobManager(db=db)
        job = BrowserRunJob(job_id="br-queued1",
                            url="https://example.com",
                            goal="never runs")
        job._on_change = lambda: mgr._record_job(job)
        mgr._jobs[job.job_id] = job
        mgr._record_job(job)
        result = mgr.cancel(job.job_id)
        self.assertEqual(result["state"], "requested")
        called = []
        mgr._run(job, lambda j: called.append(True) or {})
        self.assertEqual(called, [])  # runner never invoked
        self.assertEqual(job.status, "cancelled")
        self.assertTrue(mgr.flush_records())
        row = BrowserRunRecordStore(db).get(job.job_id)
        self.assertEqual(row["status"], "cancelled")
        self.assertEqual(row["cancel_state"], "confirmed")

    def test_cancel_cooperative_runner_lands_confirmed(self):
        db = _file_db(self)
        mgr = BrowserJobManager(db=db)

        def runner(job):
            while not job.cancel_requested():
                time.sleep(0.01)
            return {"stopped_reason": "cancelled: owner request",
                    "steps": 1}

        job, _ = mgr.start("https://example.com/y", "wait for it",
                           runner)
        result = mgr.cancel(job.job_id)
        self.assertEqual(result["state"], "requested")
        self.assertTrue(job.wait(5))
        self.assertEqual(job.status, "cancelled")
        self.assertTrue(mgr.flush_records())
        row = BrowserRunRecordStore(db).get(job.job_id)
        self.assertEqual(row["status"], "cancelled")
        self.assertEqual(row["cancel_state"], "confirmed")

    def test_cancel_unhonored_reads_unhonored(self):
        db = _file_db(self)
        mgr = BrowserJobManager(db=db)
        release = threading.Event()

        def runner(job):
            release.wait(5)
            return {"stopped_reason": "goal met", "steps": 4}

        job, _ = mgr.start("https://example.com/z", "ignore cancels",
                           runner)
        result = mgr.cancel(job.job_id)
        self.assertEqual(result["state"], "requested")
        self.assertTrue(mgr.flush_records())
        row = BrowserRunRecordStore(db).get(job.job_id)
        self.assertEqual(row["cancel_state"], "requested")
        release.set()
        self.assertTrue(job.wait(5))
        self.assertTrue(mgr.flush_records())
        row = BrowserRunRecordStore(db).get(job.job_id)
        self.assertEqual(row["status"], "done")
        self.assertEqual(row["cancel_state"], "unhonored")
        # ...and the steward's evidence says so, in words.
        case = SimpleNamespace(detail={"run_id": job.job_id})
        items = BrowserAdapter(db).collect(case, time.time())
        self.assertIn("without honoring it", items[0].summary)

    def test_cancel_unknown_and_terminal(self):
        mgr = BrowserJobManager(db=_file_db(self))
        self.assertEqual(mgr.cancel("br-nope")["state"], "not_found")
        job, _ = mgr.start("https://example.com/w", "quick",
                           lambda j: {"stopped_reason": "met"})
        self.assertTrue(job.wait(5))
        result = mgr.cancel(job.job_id)
        self.assertEqual(result["state"], "already_terminal")
        self.assertEqual(result["status"], "done")

    def test_record_redacts_credentials(self):
        db = _file_db(self)
        mgr = BrowserJobManager(db=db)
        job, _ = mgr.start(
            "https://user:pw123@example.com/path"
            "?token=abc123&q=gold&api_key=k9",
            "redaction check", lambda j: {"stopped_reason": "met"})
        self.assertTrue(job.wait(5))
        self.assertTrue(mgr.flush_records())
        row = BrowserRunRecordStore(db).get(job.job_id)
        self.assertNotIn("pw123", row["url"])
        self.assertNotIn("abc123", row["url"])
        self.assertNotIn("k9", row["url"].split("?")[-1])
        self.assertIn("example.com", row["url"])
        self.assertIn("q=gold", row["url"])

    def test_record_write_failure_never_gates_the_run(self):
        class _BrokenDb:
            def executescript(self, script):
                raise RuntimeError("disk gone")

            def execute(self, *a, **kw):
                raise RuntimeError("disk gone")

        mgr = BrowserJobManager(db=_BrokenDb())
        self.assertIsNone(mgr._records)  # store build failed open
        job, _ = mgr.start("https://example.com/v", "still runs",
                           lambda j: {"stopped_reason": "met"})
        self.assertTrue(job.wait(5))
        self.assertEqual(job.status, "done")

        # Mid-run failure: a store that raises on every write.
        class _BrokenStore:
            def record(self, snapshot):
                raise RuntimeError("write failed")

        mgr2 = BrowserJobManager(db=_file_db(self))
        mgr2._records = _BrokenStore()
        job2, _ = mgr2.start("https://example.com/u", "also runs",
                             lambda j: {"stopped_reason": "met"})
        self.assertTrue(job2.wait(5))
        self.assertEqual(job2.status, "done")


class MediaRecordTests(unittest.TestCase):
    def test_record_lifecycle_and_restart(self):
        db = _file_db(self)
        mgr = JobManager(db=db)
        job_id = mgr.submit("video", "trim the intro",
                            lambda progress: {"output": "/m/out.mp4"},
                            input_ref="/m/in.mp4")
        info = mgr.wait(job_id, timeout=10)
        self.assertEqual(info["status"], "done")
        row = MediaJobRecordStore(db).get(job_id)
        self.assertEqual(row["status"], "done")
        self.assertEqual(row["output_ref"], "/m/out.mp4")
        self.assertEqual(row["label"], "trim the intro")
        # Restart: a new manager over the same store reads the same
        # last recorded state; the steward collects succeeded.
        JobManager(db=db)
        case = SimpleNamespace(detail={"job_id": job_id})
        items = RenderAdapter(db).collect(case, time.time())
        self.assertEqual(items[0].establishes, "succeeded")

    def test_cancel_queued_and_running_records(self):
        db = _file_db(self)
        mgr = JobManager(db=db)
        gate = threading.Event()

        def blocker(progress):
            while not gate.is_set():
                progress(0.5)
                time.sleep(0.02)
            return {"output": "/m/blocker.mp4"}

        first = mgr.submit("video", "blocker", blocker)
        second = mgr.submit("video", "queued one",
                            lambda progress: {"output": "/m/two.mp4"})
        # Wait until the first is actually running.
        deadline = time.time() + 5
        while mgr.status(first)["status"] != "running":
            self.assertLess(time.time(), deadline)
            time.sleep(0.02)
        mgr.cancel(second)  # queued -> cancelled, confirmed at once
        row = MediaJobRecordStore(db).get(second)
        self.assertEqual(row["status"], "cancelled")
        mgr.cancel(first)  # running -> request lands on the record
        row = MediaJobRecordStore(db).get(first)
        self.assertEqual(row["status"], "running")
        self.assertGreater(row["cancel_requested_at"], 0.0)
        # The blocker notices at its next progress report and stops.
        info = mgr.wait(first, timeout=10)
        self.assertEqual(info["status"], "cancelled")
        row = MediaJobRecordStore(db).get(first)
        self.assertEqual(row["status"], "cancelled")
        gate.set()

    def test_stuck_records_discovered_by_render_adapter(self):
        db = _file_db(self)
        store = MediaJobRecordStore(db)
        store.record(_media_snapshot("mj-stuck1", "running",
                                     updated_at=OLD))
        store.record(_media_snapshot("mj-stuck2", "queued",
                                     updated_at=OLD))
        store.record(_media_snapshot("mj-fresh", "running",
                                     updated_at=time.time()))
        found = {(c.subject_key, c.reason)
                 for c in RenderAdapter(db).discover(time.time())}
        self.assertIn(("render:mj-stuck1", "unknown-outcome"), found)
        self.assertIn(("render:mj-stuck2", "overdue"), found)
        self.assertNotIn(("render:mj-fresh", "unknown-outcome"), found)

    def test_record_write_failure_never_gates_the_job(self):
        class _BrokenStore:
            def record(self, snapshot):
                raise RuntimeError("write failed")

        mgr = JobManager(db=_file_db(self))
        mgr._records = _BrokenStore()
        job_id = mgr.submit("video", "still runs",
                            lambda progress: {"output": "/m/x.mp4"})
        info = mgr.wait(job_id, timeout=10)
        self.assertEqual(info["status"], "done")


class StewardFlipTests(unittest.TestCase):
    def test_adapter_health_flips_exactly_two_domains(self):
        health = {row["domain"]: row for row in adapter_health()}
        self.assertEqual(len(health), 9)
        self.assertEqual(health["browser"]["status"], "working")
        self.assertIn("browser_run_records", health["browser"]["note"])
        self.assertEqual(health["render"]["status"], "working")
        self.assertIn("media_job_records", health["render"]["note"])
        for domain in ("transfers", "artifacts"):
            self.assertEqual(health[domain]["status"], "unavailable",
                             domain)
            self.assertTrue(health[domain]["note"], domain)

    def test_sweep_opens_cases_from_the_new_records(self):
        db = _file_db(self)
        BrowserRunRecordStore(db).record(
            _browser_snapshot("br-orphan", "running", updated_at=OLD))
        MediaJobRecordStore(db).record(
            _media_snapshot("mj-orphan", "running", updated_at=OLD))
        steward = OutcomeSteward(db, registry=JobRegistry())
        steward.sweep()
        rows = db.query(
            "SELECT domain, reason FROM outcome_cases"
            " WHERE domain IN ('browser', 'render')")
        found = {(r["domain"], r["reason"]) for r in rows}
        self.assertIn(("browser", "unknown-outcome"), found)
        self.assertIn(("render", "unknown-outcome"), found)
        # Cases render subjects without internal ids.
        cases = OutcomeCaseStore(db)
        for case in cases.list_cases():
            if case.domain in ("browser", "render"):
                self.assertNotIn(case.subject_key, case.subject)


if __name__ == "__main__":
    unittest.main()
