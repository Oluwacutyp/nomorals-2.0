"""Doc 18 P3 — the named-resource lease authority.

Pins: (a) acquire grants exclusively and ``check`` answers the
fencing question definitively; (b) contention is a typed refusal
naming the current holder + expiry — never a wait, never a steal;
(c) the same holder re-acquiring is idempotent (renew is the
extension verb); (d) renew extends only at the live generation;
(e) the doc §6 fencing story end-to-end — a lapsed holder whose
resource was re-acquired is refused ``stale_generation`` on BOTH
renew and release, and the successor's lease is untouched;
(f) release frees the resource for an immediate acquire at the
NEXT generation (generations are never reused); (g) expiry is
computed at read — no sweep runs anywhere in these tests;
(h) concurrent acquires grant exactly once; (i) resource names
are an open vocabulary seeded by the doc §2A list, not a closed
enum; (j) the P2 ticket projection reads the authority — a
holder's ticket carries its leases cited to core.leases, a
lease-less ticket reads present-but-empty (NOT absent), a no-db
ticket keeps leases absent with the reason, and coverage reports
held counts instead of the pre-P3 absence.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest

from nomorals.agents.delegation import DelegationStore
from nomorals.agents.run_ticket import ticket_coverage, ticket_for
from nomorals.browser.jobs import BrowserRunRecordStore
from nomorals.core.jobs import JobRegistry
from nomorals.core.leases import (
    KNOWN_RESOURCES,
    LeaseStore,
    live_lease_for_resource,
    live_leases_for_holder,
)
from nomorals.media_edit.jobs import MediaJobRecordStore
from nomorals.missions.mission import Mission, MissionStore
from nomorals.storage.db import Database

T0 = 1_700_000_000.0


def _file_db(test: unittest.TestCase, *, migrate: bool = False) -> Database:
    tmp = tempfile.TemporaryDirectory(
        dir=os.environ.get("NM_TEST_TMPDIR") or None)
    test.addCleanup(tmp.cleanup)
    db = Database(os.path.join(tmp.name, "leases.db"))
    if migrate:
        db.migrate()
    return db


class LeaseGrantTests(unittest.TestCase):

    def setUp(self) -> None:
        self.db = _file_db(self)
        self.store = LeaseStore(self.db)

    def test_acquire_grants_and_check_answers(self):
        result = self.store.acquire(
            "browser-profile", "browser:run-1", 600, now=T0)
        self.assertTrue(result["granted"])
        lease = result["lease"]
        self.assertEqual(lease["resource"], "browser-profile")
        self.assertEqual(lease["holder_ref"], "browser:run-1")
        self.assertEqual(lease["generation"], 1)
        self.assertEqual(lease["state"], "held")
        self.assertEqual(lease["acquired_at"], T0)
        self.assertEqual(lease["expires_at"], T0 + 600)
        # The fencing question, answered definitively.
        self.assertTrue(self.store.check(
            "browser-profile", "browser:run-1", 1, now=T0))
        self.assertFalse(self.store.check(
            "browser-profile", "browser:run-2", 1, now=T0))
        self.assertFalse(self.store.check(
            "browser-profile", "browser:run-1", 2, now=T0))
        self.assertEqual(
            self.store.current("browser-profile", now=T0)["lease_id"],
            lease["lease_id"])
        self.assertEqual(self.store.get(lease["lease_id"], now=T0),
                         lease)
        self.assertEqual(
            self.store.leases_for_resource("browser-profile", now=T0),
            [lease])
        self.assertEqual(
            self.store.leases_held_by("browser:run-1", now=T0),
            [lease])

    def test_contention_names_holder_and_expiry(self):
        first = self.store.acquire("gpu", "job:a", 300, now=T0)["lease"]
        result = self.store.acquire("gpu", "job:b", 300, now=T0 + 5)
        self.assertFalse(result["granted"])
        self.assertEqual(result["reason"], "held")
        self.assertEqual(result["holder_ref"], "job:a")
        self.assertEqual(result["lease_id"], first["lease_id"])
        self.assertEqual(result["generation"], 1)
        self.assertEqual(result["expires_at"], T0 + 300)
        # The refusal disturbed nothing.
        self.assertTrue(self.store.check("gpu", "job:a", 1, now=T0 + 5))
        self.assertEqual(
            self.store.current("gpu", now=T0 + 5)["holder_ref"],
            "job:a")

    def test_same_holder_reacquire_is_idempotent(self):
        first = self.store.acquire("file", "mission:m1", 600, now=T0)
        again = self.store.acquire("file", "mission:m1", 600,
                                    now=T0 + 10)
        self.assertTrue(again["granted"])
        self.assertTrue(again["already_held"])
        self.assertEqual(again["lease"]["lease_id"],
                         first["lease"]["lease_id"])
        self.assertEqual(again["lease"]["generation"], 1)
        # Re-acquire does NOT extend — renew is the extension verb.
        self.assertEqual(again["lease"]["expires_at"], T0 + 600)

    def test_acquire_validates_inputs(self):
        with self.assertRaises(ValueError):
            self.store.acquire("", "job:a", 60, now=T0)
        with self.assertRaises(ValueError):
            self.store.acquire("gpu", "", 60, now=T0)
        with self.assertRaises(ValueError):
            self.store.acquire("gpu", "job:a", 0, now=T0)

    def test_open_vocabulary_seeded_by_the_doc_list(self):
        self.assertEqual(
            set(KNOWN_RESOURCES),
            {"browser-profile", "file", "account-grant", "tool",
             "gpu", "paid-quota"})
        # Not a closed enum: an unnamed resource leases fine.
        result = self.store.acquire("some-new-resource", "job:a",
                                    60, now=T0)
        self.assertTrue(result["granted"])


class LeaseRenewReleaseTests(unittest.TestCase):

    def setUp(self) -> None:
        self.db = _file_db(self)
        self.store = LeaseStore(self.db)

    def test_renew_extends_only_at_the_live_generation(self):
        lease = self.store.acquire("gpu", "job:a", 600,
                                   now=T0)["lease"]
        renewed = self.store.renew(lease["lease_id"], 1, now=T0 + 100)
        self.assertTrue(renewed["renewed"])
        # Default TTL is the lease's own: now + 600.
        self.assertEqual(renewed["lease"]["expires_at"], T0 + 700)
        explicit = self.store.renew(lease["lease_id"], 1, ttl_s=60,
                                    now=T0 + 200)
        self.assertEqual(explicit["lease"]["expires_at"], T0 + 260)
        refused = self.store.renew(lease["lease_id"], 99, now=T0 + 200)
        self.assertFalse(refused["renewed"])
        self.assertEqual(refused["reason"], "stale_generation")
        self.assertEqual(refused["current_generation"], 1)
        missing = self.store.renew("lease-nope", 1, now=T0)
        self.assertEqual(missing["reason"], "not_found")
        # The refusals wrote nothing.
        self.assertEqual(
            self.store.current("gpu", now=T0 + 200)["expires_at"],
            T0 + 260)

    def test_release_frees_for_immediate_acquire(self):
        lease = self.store.acquire("file", "job:a", 600,
                                   now=T0)["lease"]
        released = self.store.release(lease["lease_id"], 1, now=T0 + 5)
        self.assertTrue(released["released"])
        self.assertEqual(released["lease"]["state"], "released")
        self.assertEqual(released["lease"]["released_at"], T0 + 5)
        self.assertIsNone(self.store.current("file", now=T0 + 5))
        nxt = self.store.acquire("file", "job:b", 600, now=T0 + 5)
        self.assertTrue(nxt["granted"])
        self.assertEqual(nxt["lease"]["generation"], 2)
        # A second release of the ended lease is a typed refusal.
        again = self.store.release(lease["lease_id"], 2, now=T0 + 6)
        self.assertFalse(again["released"])
        self.assertEqual(again["reason"], "released")

    def test_generations_never_reused(self):
        for expected in (1, 2, 3):
            lease = self.store.acquire("gpu", "job:a", 600,
                                       now=T0)["lease"]
            self.assertEqual(lease["generation"], expected)
            self.store.release(lease["lease_id"], expected, now=T0)


class LeaseFencingTests(unittest.TestCase):
    """The doc §6 fixture: two tickets fight one lease."""

    def setUp(self) -> None:
        self.db = _file_db(self)
        self.store = LeaseStore(self.db)

    def test_interrupted_holder_is_fenced_out(self):
        first = self.store.acquire("browser-profile", "browser:a",
                                   30, now=T0)["lease"]
        # The rush cannot steal an unexpired lease.
        rush = self.store.acquire("browser-profile", "browser:b",
                                  30, now=T0 + 10)
        self.assertFalse(rush["granted"])
        # A's lease lapses; B acquires at the next generation.
        second = self.store.acquire("browser-profile", "browser:b",
                                    30, now=T0 + 31)
        self.assertTrue(second["granted"])
        self.assertEqual(second["lease"]["generation"], 2)
        # A's late renew AND release are refused as stale, and
        # neither touches B's lease.
        renew = self.store.renew(first["lease_id"], 1, now=T0 + 32)
        self.assertFalse(renew["renewed"])
        self.assertEqual(renew["reason"], "stale_generation")
        self.assertEqual(renew["current_generation"], 2)
        release = self.store.release(first["lease_id"], 1, now=T0 + 32)
        self.assertFalse(release["released"])
        self.assertEqual(release["reason"], "stale_generation")
        self.assertTrue(self.store.check(
            "browser-profile", "browser:b", 2, now=T0 + 32))
        self.assertFalse(self.store.check(
            "browser-profile", "browser:a", 1, now=T0 + 32))
        # The record shows what happened to A's lease.
        self.assertEqual(
            self.store.get(first["lease_id"], now=T0 + 32)["state"],
            "expired")
        history = self.store.leases_for_resource(
            "browser-profile", include_history=True, now=T0 + 32)
        self.assertEqual([row["generation"] for row in history],
                         [2, 1])

    def test_expiry_is_computed_never_swept(self):
        lease = self.store.acquire("paid-quota", "job:a", 5,
                                   now=T0)["lease"]
        later = T0 + 6
        # No sweep call exists in this test: reads compute expiry.
        self.assertIsNone(self.store.current("paid-quota", now=later))
        self.assertEqual(
            self.store.leases_held_by("job:a", now=later), [])
        self.assertFalse(self.store.check(
            "paid-quota", "job:a", 1, now=later))
        self.assertEqual(
            self.store.get(lease["lease_id"], now=later)["state"],
            "expired")
        self.assertIsNone(live_lease_for_resource(
            self.db, "paid-quota", now=later))
        self.assertEqual(live_leases_for_holder(
            self.db, "job:a", now=later), [])
        # A late renew of a lapsed-but-unsuperseded lease reads
        # expired (its generation is still current — staleness
        # would be a lie).
        renew = self.store.renew(lease["lease_id"], 1, now=later)
        self.assertFalse(renew["renewed"])
        self.assertEqual(renew["reason"], "expired")
        release = self.store.release(lease["lease_id"], 1, now=later)
        self.assertFalse(release["released"])
        self.assertEqual(release["reason"], "expired")


class LeaseConcurrencyTests(unittest.TestCase):

    def test_concurrent_acquire_grants_exactly_once(self):
        db = _file_db(self)
        store = LeaseStore(db)
        barrier = threading.Barrier(8)
        results: list[dict] = []
        lock = threading.Lock()

        def worker(index: int) -> None:
            barrier.wait(timeout=10)
            result = store.acquire("gpu", f"job:{index}", 600)
            with lock:
                results.append(result)

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        granted = [r for r in results if r["granted"]]
        refused = [r for r in results if not r["granted"]]
        self.assertEqual(len(granted), 1)
        self.assertEqual(len(refused), 7)
        winner = granted[0]["lease"]
        self.assertEqual(winner["generation"], 1)
        self.assertEqual(store.current("gpu")["holder_ref"],
                         winner["holder_ref"])
        # Exactly one row is in the recorded held state.
        held = db.query(
            "SELECT COUNT(*) AS n FROM resource_leases"
            " WHERE state = 'held'")
        self.assertEqual(held[0]["n"], 1)


def _browser_snapshot(run_id: str) -> dict:
    now = time.time()
    return {"run_id": run_id, "url": "https://example.com/a",
            "goal": "check the thing", "status": "running",
            "steps": 1, "stopped_reason": "",
            "final_url": "https://example.com/a",
            "wall_outcome": "no wall", "error": "",
            "cancel_state": "", "cancel_requested_at": 0.0,
            "created_at": now, "started_at": now,
            "finished_at": 0.0, "updated_at": now}


def _media_snapshot(job_id: str) -> dict:
    now = time.time()
    return {"job_id": job_id, "kind": "video", "label": "trim",
            "backend": "ffmpeg", "input_ref": "/media/in.mp4",
            "output_ref": "", "status": "queued", "error": "",
            "cancel_requested_at": 0.0, "created_at": now,
            "started_at": now, "finished_at": 0.0,
            "updated_at": now}


class TicketLeaseIntegrationTests(unittest.TestCase):

    def setUp(self) -> None:
        self.db = _file_db(self, migrate=True)
        self.leases = LeaseStore(self.db)

    def test_browser_ticket_carries_its_leases(self):
        BrowserRunRecordStore(self.db).record(
            _browser_snapshot("run-x"))
        granted = self.leases.acquire(
            "browser-profile", "browser:run-x", 3600)["lease"]
        ticket = ticket_for("browser:run-x", self.db)
        self.assertTrue(ticket["found"])
        self.assertNotIn("leases", ticket["absent"])
        self.assertEqual(ticket["sources"]["leases"], "core.leases")
        self.assertEqual(len(ticket["leases"]), 1)
        entry = ticket["leases"][0]
        self.assertEqual(entry["lease_id"], granted["lease_id"])
        self.assertEqual(entry["resource"], "browser-profile")
        self.assertEqual(entry["generation"], 1)
        self.assertEqual(entry["expires_at"], granted["expires_at"])

    def test_leaseless_ticket_is_present_but_empty(self):
        store = DelegationStore(self.db)
        contract = store.create_contract(
            goal="plain work", worker="w", verifier="v")
        ticket = ticket_for(f"contract:{contract.id}", self.db)
        # Present-but-empty is a FACT about the store now — the
        # field is cited, not absent (the pre-P3 shape).
        self.assertEqual(ticket["leases"], [])
        self.assertEqual(ticket["sources"]["leases"], "core.leases")
        self.assertNotIn("leases", ticket["absent"])

    def test_expired_lease_does_not_ride_the_ticket(self):
        BrowserRunRecordStore(self.db).record(
            _browser_snapshot("run-old"))
        self.leases.acquire("browser-profile", "browser:run-old",
                            60, now=time.time() - 7200.0)
        ticket = ticket_for("browser:run-old", self.db)
        self.assertEqual(ticket["leases"], [])

    def test_no_db_ticket_keeps_leases_absent(self):
        registry = JobRegistry()
        gate = threading.Event()
        self.addCleanup(gate.set)
        job = registry.submit(lambda: gate.wait(30), tool="fetch_page")
        ticket = ticket_for(f"job:{job.id}", registry=registry)
        self.assertTrue(ticket["found"])
        self.assertIsNone(ticket["leases"])
        self.assertIn("core.leases", ticket["absent"]["leases"])
        self.assertIn("no database handle",
                      ticket["absent"]["leases"])

    def test_coverage_reports_held_counts(self):
        DelegationStore(self.db).create_contract(
            goal="c", worker="w", verifier="v")
        MissionStore(self.db).create(Mission(goal="m", name="m"))
        BrowserRunRecordStore(self.db).record(
            _browser_snapshot("run-c"))
        MediaJobRecordStore(self.db).record(
            _media_snapshot("job-c"))
        self.leases.acquire("browser-profile", "browser:run-c", 3600)
        registry = JobRegistry()
        gate = threading.Event()
        self.addCleanup(gate.set)
        registry.submit(lambda: gate.wait(30), tool="fetch_page")
        coverage = ticket_coverage(self.db, registry=registry)
        self.assertEqual(coverage["operations"], 5)
        self.assertNotIn("leases", coverage["fields_absent"])
        self.assertEqual(coverage["leases_held"], 1)
        self.assertEqual(coverage["lease_holders"], 1)
        joined = "\n".join(coverage["summary"])
        self.assertIn("leases: 1 held by 1 of 5 operations", joined)


if __name__ == "__main__":
    unittest.main()
