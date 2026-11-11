"""Doc 18 P1 remainder — the webnav loop polls cancel_requested().

The P1 job manager shipped the cooperative contract ("runners poll
``job.cancel_requested()`` and stop themselves"), and its suite
pinned that contract against hand-written fake runners — but the
PRODUCTION runner, the wired ``AgentLoop`` behind ``/browser run``,
had no cancel seam at all: a cancel requested mid-run was recorded
``requested`` and the loop drove pages to its own end, landing the
record ``done`` / ``unhonored`` every time.

Pins here:

* the loop polls at its step boundaries (before the first fetch,
  at the top of every step, and once more as it closes out) —
  steps after a cancel never execute, proven against the fake
  driver's action log;
* a cancel landing between the final step and the record close
  still reads cancelled, at BOTH close-out layers: the loop's own
  re-check, and ``finish()``'s conversion for poll-armed results;
* a runner that never polls keeps the shipped ``unhonored``
  reading (the P1 distinction survives);
* no-cancel runs are byte-identical in steps, trace and result
  shape to the unobserved loop;
* a poll that raises degrades fail-open: the failure is recorded,
  the poll disarms, the healthy run completes untouched;
* the /runs stop path (Fix 5: ``cancel_with_receipt`` with the
  browser manager) drives this same signal end-to-end.
"""

from __future__ import annotations

import os
import tempfile
import threading
import unittest

from nomorals.browser.agent import Action, AgentLoop, Decision, Page
from nomorals.browser.failures import clear_failures, recent_failures
from nomorals.browser.jobs import (
    BrowserJobManager,
    BrowserRunRecordStore,
)
from nomorals.core.cancellation import cancel_with_receipt, receipts_for_ref
from nomorals.storage.db import Database

START = "https://nav.example/0"


def _file_db(test: unittest.TestCase) -> Database:
    tmp = tempfile.TemporaryDirectory(
        dir=os.environ.get("NM_TEST_TMPDIR") or None)
    test.addCleanup(tmp.cleanup)
    db = Database(os.path.join(tmp.name, "records.db"))
    db.migrate()
    return db


class ChainDriver:
    """Fake driver seam: a chain of pages, each carrying one
    benign ``next`` element. Every act() is logged — the log is
    how the pins prove which steps executed."""

    def __init__(self, length: int = 6) -> None:
        self.pages: dict[str, Page] = {}
        for i in range(length):
            url = f"https://nav.example/{i}"
            elements = ([{"id": "next", "label": "next page",
                          "url": f"https://nav.example/{i + 1}"}]
                        if i + 1 < length else [])
            self.pages[url] = Page(url=url, title=f"P{i}",
                                   elements=elements)
        self.acts: list[dict] = []
        self.fetches: list[str] = []

    def fetch(self, url: str) -> Page:
        self.fetches.append(url)
        return self.pages.get(url, Page(url=url, title="missing"))

    def act(self, action: Action, page: Page) -> Page:
        self.acts.append(action.to_dict())
        if action.kind == "goto":
            return self.fetch(action.target)
        for el in page.elements:
            if el.get("id") == action.target and el.get("url"):
                return self.fetch(str(el["url"]))
        return page


def _always_next(obs) -> Decision | None:
    for el in obs.page.elements:
        if el.get("id") == "next":
            return Decision(
                Action(kind="click", target="next",
                       label="next page"),
                reason="test decider")
    return None


def _loop(driver: ChainDriver, **kwargs) -> AgentLoop:
    kwargs.setdefault("decide_llm", _always_next)
    kwargs.setdefault("autonomy", "act_silent")
    return AgentLoop(driver, **kwargs)


class LoopPollTests(unittest.TestCase):
    def test_no_hook_run_is_byte_identical(self):
        driver = ChainDriver(4)
        out = _loop(driver, max_steps=10).run(START, goal="walk")
        self.assertEqual(set(out), {"goal", "steps",
                                    "stopped_reason", "final_url",
                                    "trace"})
        self.assertEqual(out["steps"], 3)
        self.assertEqual(out["stopped_reason"], "nothing safe to do")
        self.assertEqual(out["final_url"], "https://nav.example/3")
        self.assertEqual(len(driver.acts), 3)
        self.assertEqual([t["step"] for t in out["trace"]], [1, 2, 3])

    def test_quiet_hook_changes_nothing_but_the_marker(self):
        driver = ChainDriver(4)
        out = _loop(driver, max_steps=10,
                    cancel_requested=lambda: False).run(
                        START, goal="walk")
        self.assertEqual(out["steps"], 3)
        self.assertEqual(out["stopped_reason"], "nothing safe to do")
        self.assertEqual(out["final_url"], "https://nav.example/3")
        self.assertEqual(len(driver.acts), 3)
        self.assertEqual([t["step"] for t in out["trace"]], [1, 2, 3])
        self.assertTrue(out["cancel_poll_armed"])

    def test_cancel_before_start_drives_nothing(self):
        driver = ChainDriver(4)
        out = _loop(driver, cancel_requested=lambda: True).run(
            START, goal="walk")
        self.assertEqual(out["steps"], 0)
        self.assertEqual(out["stopped_reason"],
                         "cancelled: owner request")
        self.assertEqual(out["final_url"], START)
        self.assertEqual(driver.fetches, [])
        self.assertEqual(driver.acts, [])

    def test_cancel_mid_run_stops_at_the_next_boundary(self):
        driver = ChainDriver(6)
        flag = threading.Event()

        def decider(obs):
            if obs.step == 2:
                flag.set()  # the request lands during step 2
            return _always_next(obs)

        out = _loop(driver, max_steps=30, decide_llm=decider,
                    cancel_requested=flag.is_set).run(
                        START, goal="walk")
        # Step 2's action was already in flight when the request
        # landed and completed; step 3 was never driven.
        self.assertEqual(len(driver.acts), 2)
        self.assertEqual(out["steps"], 2)
        self.assertEqual(out["stopped_reason"],
                         "cancelled: owner request")
        self.assertEqual(out["final_url"], "https://nav.example/2")

    def test_cancel_after_final_step_lands_via_loop_recheck(self):
        # max_steps=1: the only step's act sets the flag, the loop
        # exhausts its budget, and the close-out re-check — not a
        # step boundary — is what can observe the request.
        driver = ChainDriver(6)
        flag = threading.Event()

        def decider(obs):
            flag.set()
            return _always_next(obs)

        out = _loop(driver, max_steps=1, decide_llm=decider,
                    cancel_requested=flag.is_set).run(
                        START, goal="walk")
        self.assertEqual(len(driver.acts), 1)
        self.assertEqual(out["steps"], 1)
        self.assertTrue(out["stopped_reason"].startswith("cancelled"))
        self.assertIn("step budget exhausted", out["stopped_reason"])

    def test_poll_failure_degrades_without_killing_the_run(self):
        clear_failures()
        calls: list[int] = []

        def broken_poll() -> bool:
            calls.append(1)
            raise RuntimeError("cancel store unreadable")

        driver = ChainDriver(3)
        out = _loop(driver, max_steps=10,
                    cancel_requested=broken_poll).run(
                        START, goal="walk")
        # The healthy run completed exactly as an unobserved one.
        self.assertEqual(out["steps"], 2)
        self.assertEqual(out["stopped_reason"], "nothing safe to do")
        self.assertEqual(len(driver.acts), 2)
        # The poll disarmed after its first failure (no per-step
        # spam) and the failure is on the browser ledger.
        self.assertEqual(len(calls), 1)
        sites = [f.site for f in recent_failures()]
        self.assertIn("browser.agent.cancel_poll", sites)


class _BlockingDecider:
    """A decider that parks inside step 2 until released — the
    deterministic mid-flight window for manager-level cancels."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, obs):
        if obs.step == 2:
            self.entered.set()
            self.release.wait(10)
        return _always_next(obs)


def _production_runner(driver_box: list, decider=None):
    """The runner exactly as runtime_reach wires it: a real
    AgentLoop over the job, polling the job's own cancel hook."""
    def runner(job):
        driver = ChainDriver(6)
        driver_box.append(driver)
        loop = AgentLoop(
            driver, decide_llm=decider or _always_next,
            autonomy="act_silent", max_steps=30,
            on_progress=job.note_progress,
            cancel_requested=job.cancel_requested)
        return loop.run(START, goal="walk")
    return runner


class ManagerCancelTests(unittest.TestCase):
    def test_manager_cancel_stops_the_real_loop(self):
        db = _file_db(self)
        mgr = BrowserJobManager(db=db)
        decider = _BlockingDecider()
        box: list = []
        job, _ = mgr.start(START, "walk",
                           _production_runner(box, decider))
        self.assertTrue(decider.entered.wait(5))
        result = mgr.cancel(job.job_id)
        self.assertEqual(result["state"], "requested")
        decider.release.set()
        self.assertTrue(job.wait(5))
        self.assertEqual(job.status, "cancelled")
        self.assertEqual(job.stopped_reason, "cancelled: owner request")
        # The step in flight completed; no later step was driven.
        self.assertEqual(len(box[0].acts), 2)
        self.assertTrue(mgr.flush_records())
        row = BrowserRunRecordStore(db).get(job.job_id)
        self.assertEqual(row["status"], "cancelled")
        self.assertEqual(row["cancel_state"], "confirmed")
        self.assertEqual(row["stopped_reason"],
                         "cancelled: owner request")

    def test_cancel_in_the_close_sliver_lands_cancelled(self):
        # A poll-armed runner whose final poll just missed the
        # request: it returns a natural stop while the request is
        # already standing, and finish()'s close-out re-check is
        # the layer that must land the record cancelled.
        db = _file_db(self)
        mgr = BrowserJobManager(db=db)
        release = threading.Event()

        def runner(job):
            release.wait(10)
            return {"goal": "walk", "steps": 3,
                    "stopped_reason": "nothing safe to do",
                    "final_url": "https://nav.example/3",
                    "trace": [], "cancel_poll_armed": True}

        job, _ = mgr.start(START, "walk", runner)
        result = mgr.cancel(job.job_id)
        self.assertEqual(result["state"], "requested")
        release.set()
        self.assertTrue(job.wait(5))
        self.assertEqual(job.status, "cancelled")
        self.assertEqual(job.cancel_state, "confirmed")
        self.assertTrue(job.stopped_reason.startswith("cancelled"))
        self.assertIn("nothing safe to do", job.stopped_reason)
        self.assertTrue(mgr.flush_records())
        row = BrowserRunRecordStore(db).get(job.job_id)
        self.assertEqual(row["status"], "cancelled")
        self.assertEqual(row["cancel_state"], "confirmed")

    def test_unarmed_runner_keeps_the_unhonored_reading(self):
        # The control for the pin above: identical timing, but the
        # runner never polled (no marker) — the shipped P1
        # distinction survives untouched.
        db = _file_db(self)
        mgr = BrowserJobManager(db=db)
        release = threading.Event()

        def runner(job):
            release.wait(10)
            return {"goal": "walk", "steps": 3,
                    "stopped_reason": "nothing safe to do",
                    "final_url": "https://nav.example/3",
                    "trace": []}

        job, _ = mgr.start(START, "walk", runner)
        mgr.cancel(job.job_id)
        release.set()
        self.assertTrue(job.wait(5))
        self.assertEqual(job.status, "done")
        self.assertEqual(job.cancel_state, "unhonored")
        self.assertTrue(mgr.flush_records())
        row = BrowserRunRecordStore(db).get(job.job_id)
        self.assertEqual(row["status"], "done")
        self.assertEqual(row["cancel_state"], "unhonored")

    def test_runs_stop_facade_drives_the_same_signal(self):
        # Fix 5's /runs stop is cancel_with_receipt over the live
        # manager — the same request the loop now honors.
        db = _file_db(self)
        mgr = BrowserJobManager(db=db)
        decider = _BlockingDecider()
        box: list = []
        job, _ = mgr.start(START, "walk",
                           _production_runner(box, decider))
        self.assertTrue(decider.entered.wait(5))
        outcome = cancel_with_receipt(
            db, f"browser:{job.job_id}", requested_by="owner",
            reason="stopped from /runs", browser_manager=mgr)
        self.assertEqual(outcome["organ"], "browser")
        self.assertEqual(outcome["native_state"], "requested")
        self.assertIsNotNone(outcome["receipt"])
        self.assertEqual(outcome["receipt"]["receipt_state"],
                         "requested")
        decider.release.set()
        self.assertTrue(job.wait(5))
        self.assertEqual(job.status, "cancelled")
        self.assertTrue(mgr.flush_records())
        row = BrowserRunRecordStore(db).get(job.job_id)
        self.assertEqual(row["cancel_state"], "confirmed")
        receipts = receipts_for_ref(db, f"browser:{job.job_id}")
        self.assertEqual(len(receipts), 1)
        self.assertEqual(receipts[0]["requested_by"], "owner")


if __name__ == "__main__":
    unittest.main()
