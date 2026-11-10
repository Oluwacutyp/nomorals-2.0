"""Doc 18 P4 — admission, capacity, queue-with-record at coremind.

Pins: (a) saturation QUEUES with a durable record — an accepted job
is never marked failed for arriving during load (the pre-P4 shed);
the record shows queued → active → terminal with timestamps, why
it waits, and its position; (b) the queue drains FIFO on the
completion event (handoff), no polling; (c) cancel-while-queued
lands typed on the record and the job never runs, while running /
finished / unknown jobs answer with their real state; (d) restart
semantics: rows still queued at load read ``lost`` with the reason
— never silently vanished, never replayed — while ``active`` rows
keep their pre-P4 meaning; (e) jobs that start immediately are
untouched (no queue fields, outcomes identical to the pre-P4
execution layer); (f) the admission limit derives from the shipped
profile table (max_parallel) when config does not set it, and an
explicit config value still wins; (g) a queued job past the named
overdue bound surfaces as overdue on its record and its doc-18
ticket — and still runs when a slot frees; (h) the ticket
projection reads coremind jobs via ``mind:`` refs with the
field-partition invariant intact.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.coremind import (
    MAX_INFLIGHT_DEFAULT,
    QUEUE_OVERDUE_DEFAULT_S,
    CoreMind,
    Intent,
)
from nomorals.agents.run_ticket import (
    TICKET_FIELDS,
    ticket_coverage,
    ticket_for,
)
from nomorals.core.jobs import JobRegistry


class _Settings:
    """Settings whose resolve() lands under a temp root, so the
    mind's durable state file is real (the restart test needs it).
    ``mind`` is the settings.mind section (None = unconfigured)."""

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


def _make_mind(test, cfg=None, root=None):
    if root is None:
        tmp = tempfile.TemporaryDirectory(
            dir=os.environ.get("NM_TEST_TMPDIR") or None)
        test.addCleanup(tmp.cleanup)
        root = tmp.name
    return CoreMind(_Context(root, cfg)), root


def _intent(target):
    return Intent(kind="research", confidence=1.0, target=target,
                  route="research_swarm")


def _wait_for(pred, timeout=5.0):
    # monotonic, never the wall clock: deadlines must not move
    # when the clock does (and the repo's clock discipline agrees).
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


def _blocking_job(gate, body):
    def job():
        gate.wait(timeout=5)
        return body
    return job


class SaturationQueueTest(unittest.TestCase):
    def test_queued_jobs_drain_fifo_with_full_records(self):
        mind, _ = _make_mind(self, _mind_cfg(max_inflight=1))
        gate = threading.Event()
        order: list[str] = []

        def job_a():
            order.append("A")
            gate.wait(timeout=5)
            return "A done"

        def make(name):
            def job():
                order.append(name)
                return f"{name} done"
            return job

        id_a = mind._new_job(_intent("alpha"))
        mind._send_async("k:console", job_a, id_a, "started A",
                         kind="research")
        self.assertTrue(_wait_for(lambda: mind._inflight_now == 1))

        id_b = mind._new_job(_intent("bravo"))
        reply_b = mind._send_async("k:console", make("B"), id_b,
                                   "started B", kind="research")
        id_c = mind._new_job(_intent("charlie"))
        reply_c = mind._send_async("k:console", make("C"), id_c,
                                   "started C", kind="research")

        self.assertIn("queued at position 1", reply_b)
        self.assertIn("queued at position 2", reply_c)
        row_b = mind.job_record(id_b)
        row_c = mind.job_record(id_c)
        self.assertEqual(row_b["status"], "queued")
        self.assertEqual(row_b["queue_position"], 1)
        self.assertEqual(row_c["queue_position"], 2)
        self.assertTrue(row_b["queue_reason"])
        self.assertGreater(row_b["queued_at"], 0)
        self.assertNotIn("finished", row_b)
        # The whole point: nothing was shed, nothing reads failed.
        self.assertEqual(mind._send_shed, 0)
        self.assertEqual(mind._send_queued, 2)
        self.assertEqual(mind.queue_state()["order"], [id_b, id_c])

        gate.set()
        self.assertTrue(_wait_for(
            lambda: order == ["A", "B", "C"]
            and mind.job_record(id_c)["status"] == "done"))
        row_b = mind.job_record(id_b)
        self.assertEqual(row_b["status"], "done")
        self.assertEqual(row_b["note"], "B done")
        self.assertGreaterEqual(row_b["started_at"], row_b["queued_at"])
        self.assertIn("waited_s", row_b)
        self.assertEqual(mind.job_record(id_a)["note"], "A done")
        self.assertTrue(_wait_for(lambda: mind._inflight_now == 0))
        self.assertEqual(mind.queue_state()["depth"], 0)
        self.assertEqual(mind._send_started, 3)

    def test_cancel_while_queued_lands_on_the_record(self):
        mind, _ = _make_mind(self, _mind_cfg(max_inflight=1))
        gate = threading.Event()
        ran = threading.Event()

        id_a = mind._new_job(_intent("alpha"))
        mind._send_async("k:console", _blocking_job(gate, "A done"),
                         id_a, "started A", kind="research")
        self.assertTrue(_wait_for(lambda: mind._inflight_now == 1))

        def job_b():
            ran.set()
            return "B done"

        id_b = mind._new_job(_intent("bravo"))
        mind._send_async("k:console", job_b, id_b, "started B",
                         kind="research")
        self.assertEqual(mind.job_record(id_b)["status"], "queued")

        result = mind.cancel_job(id_b)
        self.assertTrue(result["cancelled"])
        self.assertEqual(result["state"], "cancelled")
        row_b = mind.job_record(id_b)
        self.assertEqual(row_b["status"], "cancelled")
        self.assertIn("never ran", row_b["note"])
        self.assertGreater(row_b["finished"], 0)
        self.assertEqual(mind.queue_state()["depth"], 0)

        running = mind.cancel_job(id_a)
        self.assertFalse(running["cancelled"])
        self.assertEqual(running["state"], "active")

        gate.set()
        self.assertTrue(_wait_for(
            lambda: mind.job_record(id_a)["status"] == "done"))
        finished = mind.cancel_job(id_a)
        self.assertFalse(finished["cancelled"])
        self.assertEqual(finished["state"], "done")
        unknown = mind.cancel_job("nope-nope")
        self.assertFalse(unknown["cancelled"])
        self.assertEqual(unknown["state"], "not_found")
        # The cancelled job never ran, even after the slot freed.
        self.assertFalse(ran.is_set())
        self.assertEqual(mind._send_started, 1)


class ImmediateStartTest(unittest.TestCase):
    def test_fast_path_outcomes_are_unchanged(self):
        mind, _ = _make_mind(
            self, _mind_cfg(max_inflight=2, timeout=1.0))

        def check(job, want_status, want_note_part):
            job_id = mind._new_job(_intent("fast"))
            mind._send_async("k:console", job, job_id, "started",
                             kind="research")
            self.assertTrue(_wait_for(
                lambda: mind.job_record(job_id)["status"]
                in ("done", "failed")))
            row = mind.job_record(job_id)
            self.assertEqual(row["status"], want_status)
            self.assertIn(want_note_part, row["note"])
            # No queue detour touched the record.
            self.assertNotIn("queued_at", row)
            return row

        check(lambda: "✅ all good", "done", "✅ all good")
        # The execution layer's standing convention: an organ job
        # that fails returns ❌ text instead of raising, and the
        # terminal state follows that return (doc 12's full
        # inference question is NOT this pass — pinned as-is).
        check(lambda: "❌ organ failed", "failed", "❌ organ failed")

        def boom():
            raise RuntimeError("boom-went-the-organ")

        check(boom, "failed", "boom-went-the-organ")
        self.assertEqual(mind._send_queued, 0)
        self.assertEqual(mind._send_shed, 0)


class RestartRecoveryTest(unittest.TestCase):
    def test_queued_rows_read_lost_after_restart(self):
        cfg = _mind_cfg(max_inflight=1)
        mind1, root = _make_mind(self, cfg)
        gate = threading.Event()
        id_a = mind1._new_job(_intent("alpha"))
        mind1._send_async("k:console",
                          _blocking_job(gate, "A done"), id_a,
                          "started A", kind="research")
        self.assertTrue(_wait_for(lambda: mind1._inflight_now == 1))
        id_b = mind1._new_job(_intent("bravo"))
        mind1._send_async("k:console", lambda: "B done", id_b,
                          "started B", kind="research")
        self.assertEqual(mind1.job_record(id_b)["status"], "queued")

        # A new process on the same state dir: the queued row closes
        # as lost (named, on the record, persisted) — the running
        # row keeps its pre-P4 meaning.
        mind2, _ = _make_mind(self, cfg, root=root)
        row_b = mind2.job_record(id_b)
        self.assertEqual(row_b["status"], "lost")
        self.assertTrue(row_b["note"].startswith("lost:"))
        self.assertGreater(row_b["finished"], 0)
        self.assertEqual(mind2.job_record(id_a)["status"], "active")
        on_disk = json.loads(
            (Path(root) / "data" / "coremind" / "state.json")
            .read_text())
        disk_b = [j for j in on_disk["jobs"] if j["id"] == id_b]
        self.assertEqual(disk_b[0]["status"], "lost")

        gate.set()
        self.assertTrue(_wait_for(lambda: mind1._inflight_now == 0))


class CapacityProfileTest(unittest.TestCase):
    def test_limit_follows_the_profile_table(self):
        for profile, want in (("termux", 4), ("laptop", 8),
                              ("workstation", 16)):
            with mock.patch.dict(os.environ,
                                 {"NM_PROFILE": profile}):
                mind, _ = _make_mind(self, None)
                self.assertEqual(mind._max_inflight, want,
                                 f"profile {profile}")

    def test_explicit_config_still_wins_over_profile(self):
        with mock.patch.dict(os.environ, {"NM_PROFILE": "termux"}):
            mind, _ = _make_mind(self, _mind_cfg(max_inflight=3))
            self.assertEqual(mind._max_inflight, 3)

    def test_shipped_settings_defaults_derive_from_profile(self):
        # The production config path: Settings.mind is a MindSettings
        # whose max_inflight defaults to 0 (unset) — the mind must
        # derive from the profile table, not read a second hard-coded
        # default out of the dataclass.
        from nomorals.core.config import MindSettings

        cfg = MindSettings()
        self.assertEqual(cfg.max_inflight, 0)
        self.assertEqual(cfg.queue_overdue_s, 0.0)
        with mock.patch.dict(os.environ, {"NM_PROFILE": "termux"}):
            mind, _ = _make_mind(self, cfg)
            self.assertEqual(mind._max_inflight, 4)
            self.assertEqual(mind._queue_overdue_s,
                             QUEUE_OVERDUE_DEFAULT_S)

    def test_named_fallback_constants_stand(self):
        self.assertEqual(MAX_INFLIGHT_DEFAULT, 8)
        self.assertEqual(QUEUE_OVERDUE_DEFAULT_S, 600.0)
        mind, _ = _make_mind(self, _mind_cfg(max_inflight=1))
        self.assertEqual(mind._queue_overdue_s,
                         QUEUE_OVERDUE_DEFAULT_S)


class OverdueAndTicketTest(unittest.TestCase):
    def _assert_partition(self, ticket):
        for name in TICKET_FIELDS:
            in_sources = name in ticket["sources"]
            in_absent = name in ticket["absent"]
            self.assertNotEqual(
                in_sources, in_absent,
                f"{name}: exactly one of sources/absent")
            if in_absent:
                self.assertIsNone(ticket[name])
                self.assertTrue(ticket["absent"][name])

    def test_overdue_surfaces_and_the_job_still_runs(self):
        mind, _ = _make_mind(
            self, _mind_cfg(max_inflight=1, overdue=0.2))
        gate = threading.Event()
        id_a = mind._new_job(_intent("alpha"))
        mind._send_async("k:console", _blocking_job(gate, "A done"),
                         id_a, "started A", kind="research")
        self.assertTrue(_wait_for(lambda: mind._inflight_now == 1))
        id_b = mind._new_job(_intent("bravo"))
        mind._send_async("k:console", lambda: "B done", id_b,
                         "started B", kind="research")
        self.assertEqual(mind.job_record(id_b)["status"], "queued")

        time.sleep(0.4)
        ticket = ticket_for(f"mind:{id_b}", coremind=mind)
        self.assertTrue(ticket["found"])
        self.assertEqual(ticket["kind"], "coremind_job")
        self.assertEqual(ticket["status"], "queued")
        live = ticket["liveness"]
        self.assertTrue(live["overdue"])
        self.assertGreaterEqual(live["wait_s"], 0.2)
        self.assertEqual(live["queue_position"], 1)
        self.assertEqual(live["queue_depth"], 1)
        self.assertEqual(ticket["sources"]["liveness"], "coremind")
        self._assert_partition(ticket)
        # Overdue is a report, not a death.
        self.assertEqual(mind.job_record(id_b)["status"], "queued")
        gate.set()
        self.assertTrue(_wait_for(
            lambda: mind.job_record(id_b)["status"] == "done"))

    def test_mind_ticket_shapes_and_not_found(self):
        mind, _ = _make_mind(self, _mind_cfg(max_inflight=1))
        job_id = mind._new_job(_intent("write the report"))
        mind._send_async("k:console", lambda: "✅ report written",
                         job_id, "started", kind="research")
        self.assertTrue(_wait_for(
            lambda: mind.job_record(job_id)["status"] == "done"))

        ticket = ticket_for(f"mind:{job_id}", coremind=mind)
        self.assertTrue(ticket["found"])
        self.assertEqual(ticket["purpose"], "write the report")
        self.assertEqual(ticket["sources"]["purpose"], "coremind")
        self.assertEqual(ticket["status"], "done")
        # Without a db the leases field stays absent, with the
        # P2/P3 reason naming the unread store.
        self.assertIn("leases", ticket["absent"])
        self._assert_partition(ticket)

        missing = ticket_for("mind:missing1", coremind=mind)
        self.assertFalse(missing["found"])
        no_mind = ticket_for(f"mind:{job_id}")
        self.assertFalse(no_mind["found"])
        self.assertIn("CoreMind", no_mind["reason"])

    def test_coverage_counts_coremind_jobs(self):
        mind, _ = _make_mind(self, _mind_cfg(max_inflight=1))
        for target in ("one", "two"):
            job_id = mind._new_job(_intent(target))
            mind._send_async("k:console", lambda: "✅ ok", job_id,
                             "started", kind="research")
        self.assertTrue(_wait_for(
            lambda: all(r["status"] == "done"
                        for r in mind.job_records())))
        coverage = ticket_coverage(None, registry=JobRegistry(),
                                   coremind=mind)
        self.assertEqual(coverage["by_source"]["coremind_job"], 2)
        self.assertEqual(coverage["operations"], 2)
        joined = "\n".join(coverage["summary"])
        self.assertIn("coremind jobs 2", joined)
        refs = {row["ref"] for row in coverage["operations_detail"]}
        self.assertTrue(all(r.startswith("mind:") for r in refs))


if __name__ == "__main__":
    unittest.main()
