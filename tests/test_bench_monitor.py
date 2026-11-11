"""Multi-cursor bench, Lanes 0+1 — contracts + Task Monitor claiming.

Pins: (a) task registration refuses an unfenceable spec (no
write-set, no acceptance, unknown dep, duplicate key); (b) two
cursors racing one write-set get exactly one claim and the loser
is refused NAMING the holder (sequential + threaded race); (c)
disjoint write-sets claim concurrently, and subtree-glob claims
fence the files under them; (d) a dependency-blocked task is
unclaimable until its dep is PROMOTED (submitted is not enough);
(e) heartbeat renews at the claim generation and keeps the claim
alive past its original expiry; (f) silence expires the claim,
the watchdog requeues the task, a second cursor re-claims at the
NEXT generation, and the zombie's late submit is refused
``stale_generation`` by the lease store; (g) claims are durable
— a fresh monitor over a fresh Database on the same file sees
and honours them (restart survival); (h) a recorded dead pid is
a dead claim however fresh the lease; (i) a live claim with no
progress earns exactly one ``stalled`` journal event; (j) the
read view hides an unpromoted candidate from peers and reveals
it at promotion; (k) releases free the write-set immediately at
the next generation; (l) the journal records every transition
with actor/generation/reason.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import unittest

from nomorals.agents.bench import BenchMonitor, BenchStore, TaskSpec
from nomorals.agents.bench.model import BenchError
from nomorals.agents.delegation import DelegationStore
from nomorals.storage.db import Database

T0 = 1_700_000_000.0


def _db(test: unittest.TestCase) -> Database:
    tmp = tempfile.TemporaryDirectory(
        dir=os.environ.get("NM_TEST_TMPDIR") or None)
    test.addCleanup(tmp.cleanup)
    return Database(os.path.join(tmp.name, "bench.db"))


def _spec(write_set, *, deps=(), instructions="do the thing",
          acceptance=("tests pass",)):
    return TaskSpec(instructions=instructions,
                    acceptance=tuple(acceptance),
                    write_set=tuple(write_set), deps=tuple(deps))


class _BenchCase(unittest.TestCase):

    def setUp(self) -> None:
        self.db = _db(self)
        self.monitor = BenchMonitor(self.db, lease_ttl_s=10.0,
                                    stall_after_s=5.0)
        self.run = self.monitor.open_run("goal", "rev-base", now=T0)
        self.run_id = self.run["id"]

    def register(self, key, write_set, **kw):
        return self.monitor.register_task(
            self.run_id, key, _spec(write_set, **kw), now=T0)


class RegistrationTests(_BenchCase):

    def test_spec_without_write_set_is_refused(self):
        with self.assertRaises(BenchError):
            self.monitor.register_task(
                self.run_id, "x",
                TaskSpec(instructions="i", acceptance=("a",)), now=T0)

    def test_spec_without_acceptance_is_refused(self):
        with self.assertRaises(BenchError):
            self.monitor.register_task(
                self.run_id, "x",
                TaskSpec(instructions="i", write_set=("a.py",)),
                now=T0)

    def test_unknown_dep_is_refused(self):
        with self.assertRaises(BenchError):
            self.register("x", ("a.py",), deps=("ghost",))

    def test_duplicate_key_is_refused(self):
        self.register("x", ("a.py",))
        with self.assertRaises(BenchError):
            self.register("x", ("b.py",))

    def test_registered_task_is_pending_and_journaled(self):
        task = self.register("x", ("a.py",))
        self.assertEqual(task["state"], "pending")
        kinds = [e["kind"] for e in
                 self.monitor.store.events(self.run_id)]
        self.assertEqual(kinds, ["run_opened", "task_registered"])


class ClaimContentionTests(_BenchCase):

    def test_second_cursor_is_refused_naming_the_holder(self):
        self.register("a", ("nomorals/agents/swarm.py",))
        first = self.monitor.claim(self.run_id, "a", "cursor-1",
                                   now=T0)
        self.assertTrue(first["claimed"])
        second = self.monitor.claim(self.run_id, "a", "cursor-2",
                                    now=T0 + 1)
        self.assertFalse(second["claimed"])
        self.assertEqual(second["reason"], "not_claimable")
        self.assertEqual(second["holder"], "cursor-1")
        refusals = self.monitor.store.events(
            self.run_id, kind="claim_refused")
        self.assertEqual(len(refusals), 1)
        self.assertEqual(refusals[0]["actor"], "cursor-2")

    def test_overlapping_write_set_on_another_task_is_refused(self):
        self.register("a", ("nomorals/agents/swarm.py",))
        self.register("b", ("nomorals/agents/swarm.py",
                            "nomorals/agents/other.py"))
        self.assertTrue(self.monitor.claim(
            self.run_id, "a", "cursor-1", now=T0)["claimed"])
        refused = self.monitor.claim(self.run_id, "b", "cursor-2",
                                     now=T0)
        self.assertFalse(refused["claimed"])
        self.assertEqual(refused["reason"], "write_set_held")
        self.assertEqual(refused["holder"], "cursor-1")
        self.assertEqual(refused["holder_task"], "a")
        self.assertEqual(refused["files"],
                         ["nomorals/agents/swarm.py"])
        self.assertEqual(refused["expires_at"], T0 + 10.0)
        # nothing half-granted: task b holds no leases at all
        leases = self.monitor.leases.leases_held_by("cursor-2",
                                                    now=T0)
        self.assertEqual(leases, [])

    def test_subtree_glob_fences_files_under_it(self):
        self.register("wide", ("nomorals/agents/**",))
        self.register("narrow", ("nomorals/agents/swarm.py",))
        self.assertTrue(self.monitor.claim(
            self.run_id, "wide", "cursor-1", now=T0)["claimed"])
        refused = self.monitor.claim(self.run_id, "narrow",
                                     "cursor-2", now=T0)
        self.assertFalse(refused["claimed"])
        self.assertEqual(refused["reason"], "write_set_held")

    def test_disjoint_write_sets_claim_concurrently(self):
        self.register("a", ("nomorals/agents/swarm.py",))
        self.register("b", ("nomorals/agents/coding.py",))
        one = self.monitor.claim(self.run_id, "a", "cursor-1",
                                 now=T0)
        two = self.monitor.claim(self.run_id, "b", "cursor-2",
                                 now=T0)
        self.assertTrue(one["claimed"])
        self.assertTrue(two["claimed"])
        self.assertEqual(one["generation"], 1)
        self.assertEqual(two["generation"], 1)

    def test_threaded_race_grants_exactly_once(self):
        self.register("a", ("nomorals/agents/swarm.py",))
        barrier = threading.Barrier(2)
        results: dict[str, dict] = {}

        def race(cursor: str) -> None:
            barrier.wait(timeout=10)
            results[cursor] = self.monitor.claim(
                self.run_id, "a", cursor, now=T0)

        threads = [threading.Thread(target=race, args=(c,))
                   for c in ("cursor-1", "cursor-2")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        granted = [c for c, r in results.items() if r["claimed"]]
        self.assertEqual(len(granted), 1)
        task = self.monitor.store.get_task(self.run_id, "a")
        self.assertEqual(task["holder"], granted[0])
        held = self.monitor.leases.leases_held_by(granted[0], now=T0)
        self.assertEqual(len(held), 2)  # task lease + file lease

    def test_same_cursor_reclaim_is_idempotent(self):
        self.register("a", ("a.py",))
        first = self.monitor.claim(self.run_id, "a", "cursor-1",
                                   now=T0)
        again = self.monitor.claim(self.run_id, "a", "cursor-1",
                                   now=T0 + 1)
        self.assertTrue(again["claimed"])
        self.assertTrue(again["already_held"])
        self.assertEqual(again["generation"], first["generation"])

    def test_claim_records_a_delegation_contract(self):
        self.register("a", ("a.py",))
        claim = self.monitor.claim(self.run_id, "a", "cursor-1",
                                   now=T0)
        contract = DelegationStore(self.db).get_contract(
            claim["contract_id"])
        self.assertIsNotNone(contract)
        self.assertEqual(contract.worker, "cursor-1")
        self.assertEqual(contract.verifier, "bench-verifier")
        self.assertEqual(contract.integration_owner, "bench-monitor")
        self.assertNotEqual(contract.worker, contract.verifier)


class DependencyTests(_BenchCase):

    def _promote(self, key: str, revision: str) -> None:
        """Drive one task through the Lane 2 seam at the store
        level: submit, then the promotion transition the
        integration queue will perform."""
        store = self.monitor.store
        task = store.get_task(self.run_id, key)
        moved = store.transition(task["id"], ["submitted"],
                                 "promoted", now=T0 + 3)
        self.assertTrue(moved)
        store.note_promoted_revision(self.run_id, revision)

    def test_dep_must_be_promoted_not_merely_submitted(self):
        self.register("base", ("base.py",))
        self.register("next", ("next.py",), deps=("base",))
        refused = self.monitor.claim(self.run_id, "next",
                                     "cursor-2", now=T0)
        self.assertFalse(refused["claimed"])
        self.assertEqual(refused["reason"], "deps_unpromoted")
        self.assertEqual(refused["deps"], ["base"])
        claim = self.monitor.claim(self.run_id, "base", "cursor-1",
                                   now=T0)
        self.assertTrue(claim["claimed"])
        submitted = self.monitor.submit(
            self.run_id, "base", "cursor-1", claim["generation"],
            candidate_ref="cand-1", now=T0 + 1)
        self.assertTrue(submitted["submitted"])
        still = self.monitor.claim(self.run_id, "next", "cursor-2",
                                   now=T0 + 2)
        self.assertFalse(still["claimed"])
        self.assertEqual(still["reason"], "deps_unpromoted")
        self.assertEqual(self.monitor.ready_tasks(self.run_id), [])
        self._promote("base", "rev-2")
        ready = self.monitor.ready_tasks(self.run_id)
        self.assertEqual([t["key"] for t in ready], ["next"])
        granted = self.monitor.claim(self.run_id, "next",
                                     "cursor-2", now=T0 + 4)
        self.assertTrue(granted["claimed"])
        self.assertEqual(granted["base_revision"], "rev-2")


class HeartbeatAndExpiryTests(_BenchCase):

    def test_heartbeat_keeps_the_claim_alive(self):
        self.register("a", ("a.py",))
        claim = self.monitor.claim(self.run_id, "a", "cursor-1",
                                   now=T0)
        beat = self.monitor.heartbeat(
            self.run_id, "a", "cursor-1", claim["generation"],
            progress=True, now=T0 + 8)
        self.assertTrue(beat["renewed"])
        self.assertEqual(beat["expires_at"], T0 + 18)
        # past the ORIGINAL expiry, the renewed claim still holds
        refused = self.monitor.claim(self.run_id, "a", "cursor-2",
                                     now=T0 + 12)
        self.assertFalse(refused["claimed"])
        outcome = self.monitor.sweep(now=T0 + 12)
        self.assertEqual(outcome["expired"], [])

    def test_wrong_generation_heartbeat_is_refused(self):
        self.register("a", ("a.py",))
        claim = self.monitor.claim(self.run_id, "a", "cursor-1",
                                   now=T0)
        beat = self.monitor.heartbeat(
            self.run_id, "a", "cursor-1", claim["generation"] + 9,
            now=T0 + 1)
        self.assertFalse(beat["renewed"])
        self.assertEqual(beat["reason"], "stale_generation")

    def test_silence_expires_requeues_and_fences_the_zombie(self):
        self.register("a", ("a.py",))
        claim = self.monitor.claim(self.run_id, "a", "cursor-1",
                                   now=T0)
        outcome = self.monitor.sweep(now=T0 + 30)
        self.assertEqual(outcome["expired"], ["a"])
        task = self.monitor.store.get_task(self.run_id, "a")
        self.assertEqual(task["state"], "requeued")
        self.assertEqual(task["holder"], "")
        kinds = [e["kind"] for e in self.monitor.store.events(
            self.run_id, task_id=task["id"])]
        self.assertIn("claim_expired", kinds)
        self.assertIn("task_requeued", kinds)
        # a second cursor re-claims at the NEXT generation
        second = self.monitor.claim(self.run_id, "a", "cursor-2",
                                    now=T0 + 31)
        self.assertTrue(second["claimed"])
        self.assertEqual(second["generation"],
                         claim["generation"] + 1)
        # the zombie's late submit is refused by the store fence
        late = self.monitor.submit(
            self.run_id, "a", "cursor-1", claim["generation"],
            candidate_ref="zombie-work", now=T0 + 32)
        self.assertFalse(late["submitted"])
        self.assertEqual(late["reason"], "stale_generation")
        # and its zombie heartbeat likewise
        beat = self.monitor.heartbeat(
            self.run_id, "a", "cursor-1", claim["generation"],
            now=T0 + 32)
        self.assertFalse(beat["renewed"])

    def test_lapsed_claim_is_expired_inline_by_a_new_claim(self):
        self.register("a", ("a.py",))
        self.monitor.claim(self.run_id, "a", "cursor-1", now=T0)
        # no sweep runs; the lease lapses and a new claim at a
        # later clock expires the dead claim inline
        second = self.monitor.claim(self.run_id, "a", "cursor-2",
                                    now=T0 + 30)
        self.assertTrue(second["claimed"])
        self.assertEqual(second["generation"], 2)
        kinds = [e["kind"] for e in self.monitor.store.events(
            self.run_id)]
        self.assertIn("claim_expired", kinds)

    def test_dead_pid_is_a_dead_claim_however_fresh_the_lease(self):
        proc = subprocess.Popen(["true"])
        proc.wait(timeout=10)
        dead_pid = proc.pid
        self.register("a", ("a.py",))
        self.monitor.claim(self.run_id, "a", "cursor-1",
                           pid=dead_pid, now=T0)
        outcome = self.monitor.sweep(now=T0 + 1)
        self.assertEqual(outcome["expired"], ["a"])
        events = self.monitor.store.events(
            self.run_id, kind="claim_expired")
        self.assertEqual(events[0]["reason"], "pid_dead")

    def test_stalled_event_fires_once_per_episode(self):
        self.register("a", ("a.py",))
        claim = self.monitor.claim(self.run_id, "a", "cursor-1",
                                   now=T0)
        # heartbeats arrive (claim stays live) but no progress
        self.monitor.heartbeat(self.run_id, "a", "cursor-1",
                               claim["generation"], now=T0 + 4)
        self.monitor.heartbeat(self.run_id, "a", "cursor-1",
                               claim["generation"], now=T0 + 8)
        outcome = self.monitor.sweep(now=T0 + 9)
        self.assertEqual(outcome["stalled"], ["a"])
        outcome = self.monitor.sweep(now=T0 + 10)
        self.assertEqual(outcome["stalled"], [])
        stalled = self.monitor.store.events(
            self.run_id, kind="stalled")
        self.assertEqual(len(stalled), 1)
        self.assertEqual(stalled[0]["reason"], "no_artifact_change")


class ReleaseAndViewTests(_BenchCase):

    def test_release_frees_the_write_set_at_the_next_generation(self):
        self.register("a", ("a.py",))
        claim = self.monitor.claim(self.run_id, "a", "cursor-1",
                                   now=T0)
        released = self.monitor.release(
            self.run_id, "a", "cursor-1", claim["generation"],
            reason="yielding", now=T0 + 1)
        self.assertTrue(released["released"])
        task = self.monitor.store.get_task(self.run_id, "a")
        self.assertEqual(task["state"], "pending")
        second = self.monitor.claim(self.run_id, "a", "cursor-2",
                                    now=T0 + 2)
        self.assertTrue(second["claimed"])
        self.assertEqual(second["generation"],
                         claim["generation"] + 1)

    def test_peer_view_hides_unpromoted_candidates(self):
        self.register("a", ("a.py",))
        claim = self.monitor.claim(self.run_id, "a", "cursor-1",
                                   now=T0)
        self.monitor.submit(self.run_id, "a", "cursor-1",
                            claim["generation"],
                            candidate_ref="cand-secret", now=T0 + 1)
        store = self.monitor.store
        peer = store.peer_view(self.run_id, for_cursor="cursor-2")
        self.assertEqual(peer[0]["candidate_ref"], "")
        self.assertFalse(peer[0]["candidate_visible"])
        own = store.peer_view(self.run_id, for_cursor="cursor-1")
        self.assertEqual(own[0]["candidate_ref"], "cand-secret")
        queue = self.monitor.promotion_candidates(self.run_id)
        self.assertEqual([t["key"] for t in queue], ["a"])
        # promotion (Lane 2's transition, driven at the seam)
        task = store.get_task(self.run_id, "a")
        store.transition(task["id"], ["submitted"], "promoted",
                         now=T0 + 2)
        store.note_promoted_revision(self.run_id, "rev-2")
        peer = store.peer_view(self.run_id, for_cursor="cursor-2")
        self.assertEqual(peer[0]["candidate_ref"], "cand-secret")
        self.assertTrue(peer[0]["candidate_visible"])
        self.assertEqual(peer[0]["promoted_revision"], "rev-2")


class DurabilityTests(unittest.TestCase):

    def test_claim_survives_a_monitor_restart(self):
        tmp = tempfile.TemporaryDirectory(
            dir=os.environ.get("NM_TEST_TMPDIR") or None)
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "bench.db")
        first_db = Database(path)
        first = BenchMonitor(first_db, lease_ttl_s=10.0)
        run = first.open_run("goal", "rev-base", now=T0)
        first.register_task(run["id"], "a", _spec(("a.py",)),
                            now=T0)
        claim = first.claim(run["id"], "a", "cursor-1", now=T0)
        self.assertTrue(claim["claimed"])
        first_db.close()
        # a brand-new monitor over a brand-new Database connection
        second_db = Database(path)
        second = BenchMonitor(second_db, lease_ttl_s=10.0)
        task = second.store.get_task(run["id"], "a")
        self.assertEqual(task["state"], "claimed")
        self.assertEqual(task["holder"], "cursor-1")
        refused = second.claim(run["id"], "a", "cursor-2",
                               now=T0 + 1)
        self.assertFalse(refused["claimed"])
        self.assertEqual(refused["holder"], "cursor-1")
        beat = second.heartbeat(run["id"], "a", "cursor-1",
                                claim["generation"], now=T0 + 2)
        self.assertTrue(beat["renewed"])
        second_db.close()


class StoreContractTests(unittest.TestCase):

    def test_verdict_rows_round_trip(self):
        db = _db(self)
        store = BenchStore(db)
        run = store.create_run("goal", "rev-base", now=T0)
        task = store.insert_task(
            run["id"], "a",
            TaskSpec(instructions="i", acceptance=("a",),
                     write_set=("a.py",)), now=T0)
        row = store.record_verdict(
            run["id"], task["id"], candidate_ref="cand-1",
            verifier="bench-verifier", verdict="contradicted",
            evidence_ref="log-1", now=T0 + 1)
        self.assertEqual(row["verdict"], "contradicted")
        verdicts = store.verdicts(task["id"])
        self.assertEqual(len(verdicts), 1)
        self.assertEqual(verdicts[0]["evidence_ref"], "log-1")
        with self.assertRaises(BenchError):
            store.record_verdict(
                run["id"], task["id"], candidate_ref="c",
                verifier="v", verdict="maybe", now=T0)


if __name__ == "__main__":
    unittest.main()
