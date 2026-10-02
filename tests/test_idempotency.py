"""Wave J — idempotency: keys, dedupe, retry-after-failure, concurrent
collapse, create-mission-once, and the mission runner's idempotent step path.

Unit tier: fully offline. Real IdempotencyStore on an in-memory Database;
the MissionRunner is driven with a stub context and a stubbed step executor
so no LLM or agent is ever constructed.
"""

import threading
import os
import time
import unittest
from types import SimpleNamespace

from nomorals.missions import (
    IdempotencyStore,
    MissionRunner,
    MissionStore,
    StepOutcome,
    create_mission_once,
    dedupe,
    idempotency_key,
    mission_idempotency_key,
    step_idempotency_key,
)
from nomorals.missions.idempotency import DedupeTimeout
from nomorals.storage.db import Database


def make_db():
    db = Database(":memory:")
    db.migrate()
    return db


def make_store(db):
    return IdempotencyStore(db)


class KeyTests(unittest.TestCase):
    def test_stable_and_scoped(self):
        k1 = idempotency_key("mission_step", {"step": "s", "goal": "g"},
                             scope="m1")
        k2 = idempotency_key("mission_step", {"goal": "g", "step": "s"},
                             scope="m1")
        self.assertEqual(k1, k2)  # key order of params must not matter
        self.assertTrue(k1.startswith("idem_"))

    def test_params_and_scope_change_the_key(self):
        base = idempotency_key("k", {"a": 1}, scope="s")
        self.assertNotEqual(base, idempotency_key("k", {"a": 2}, scope="s"))
        self.assertNotEqual(base, idempotency_key("k", {"a": 1}, scope="other"))
        self.assertNotEqual(base, idempotency_key("other", {"a": 1},
                                                  scope="s"))

    def test_step_key_uses_mission_scope(self):
        step = SimpleNamespace(name="fetch", goal="fetch data", role="io")
        k1 = step_idempotency_key("m1", step)
        k2 = step_idempotency_key("m2", step)
        self.assertNotEqual(k1, k2)  # same step, different mission → re-runs
        self.assertEqual(k1, step_idempotency_key("m1", step))


class DedupeTests(unittest.TestCase):
    def setUp(self):
        self.db = make_db()
        self.addCleanup(self.db.close)
        self.store = make_store(self.db)

    def test_executes_once_on_duplicate(self):
        calls = []

        def fn():
            calls.append(1)
            return {"n": 42}

        key = idempotency_key("t", {"x": 1})
        first = dedupe(self.store, key, fn)
        second = dedupe(self.store, key, fn)
        self.assertEqual(len(calls), 1)
        self.assertTrue(first.executed)
        self.assertFalse(second.executed)
        self.assertEqual(first.value, {"n": 42})
        self.assertEqual(second.value, {"n": 42})
        self.assertEqual(second.status, "completed")

    def test_failed_key_allows_retry(self):
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("boom")
            return "recovered"

        key = idempotency_key("t", {"flaky": True})
        with self.assertRaises(RuntimeError):
            dedupe(self.store, key, flaky)
        self.assertEqual(self.store.status(key), "failed")
        result = dedupe(self.store, key, flaky)
        self.assertTrue(result.executed)
        self.assertEqual(result.value, "recovered")
        self.assertEqual(self.store.status(key), "completed")
        self.assertEqual(len(attempts), 2)

    def test_unsuccessful_value_is_returned_but_retried(self):
        calls = []

        def attempt():
            calls.append(1)
            return {"ok": len(calls) > 1, "payload": "p"}

        key = idempotency_key("step", {"s": 1})
        first = dedupe(self.store, key, attempt,
                       succeeded=lambda v: bool(v["ok"]))
        self.assertTrue(first.executed)
        self.assertEqual(first.status, "failed")  # recorded failed…
        self.assertFalse(first.value["ok"])  # …but the value is returned
        second = dedupe(self.store, key, attempt,
                        succeeded=lambda v: bool(v["ok"]))
        self.assertTrue(second.executed)  # retried, not replayed
        self.assertTrue(second.value["ok"])
        self.assertEqual(len(calls), 2)

    def test_concurrent_duplicates_collapse_to_one_execution(self):
        # File-backed DB: Database hands each thread its own connection, and
        # with :memory: those are *separate* databases — the shared table
        # only exists across threads on a real file, like production.
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        self.addCleanup(lambda: os.unlink(tmp.name) if os.path.exists(tmp.name)
                        else None)
        db = Database(tmp.name)
        db.migrate()
        self.addCleanup(db.close)
        store = make_store(db)

        calls = []
        lock = threading.Lock()
        gate = threading.Barrier(4)

        def slow():
            with lock:
                calls.append(1)
            time.sleep(0.4)
            return {"shared": "result"}

        key = idempotency_key("t", {"race": True})
        outcomes = []
        errors = []

        def worker():
            gate.wait()
            try:
                outcomes.append(dedupe(store, key, slow))
            except Exception as exc:  # noqa: BLE001 - surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        self.assertTrue(all(not t.is_alive() for t in threads))
        self.assertEqual(errors, [])
        self.assertEqual(len(calls), 1)  # exactly one execution
        self.assertEqual(len(outcomes), 4)
        for outcome in outcomes:
            self.assertEqual(outcome.value, {"shared": "result"})
        self.assertEqual(sum(1 for o in outcomes if o.executed), 1)

    def test_stale_running_claim_is_stolen(self):
        key = idempotency_key("t", {"stale": True})
        now = time.time()
        self.db.execute(
            "INSERT INTO idempotency_keys (key, status, result_json, error,"
            " owner, created_at, updated_at)"
            " VALUES (?, 'running', '{}', '', 'dead-owner', ?, ?)",
            (key, now - 10_000, now - 10_000),
        )
        calls = []
        result = dedupe(self.store, key, lambda: calls.append(1) or "fresh",
                        stale_after=60.0)
        self.assertTrue(result.executed)
        self.assertEqual(result.value, "fresh")
        self.assertEqual(len(calls), 1)

    def test_fresh_running_claim_blocks_until_resolved(self):
        key = idempotency_key("t", {"foreign": True})
        now = time.time()
        self.db.execute(
            "INSERT INTO idempotency_keys (key, status, result_json, error,"
            " owner, created_at, updated_at)"
            " VALUES (?, 'running', '{}', '', 'other-process', ?, ?)",
            (key, now, now),
        )
        # Nobody will resolve it and stealing is disabled by a huge
        # stale_after: the waiter must time out, not hang forever.
        with self.assertRaises(DedupeTimeout):
            dedupe(self.store, key, lambda: "never",
                   stale_after=10_000.0, wait_timeout=0.3,
                   poll_interval=0.05)

    def test_clear_forgets_the_key(self):
        key = idempotency_key("t", {"c": 1})
        dedupe(self.store, key, lambda: 1)
        self.assertTrue(self.store.clear(key))
        self.assertIsNone(self.store.status(key))
        self.assertFalse(self.store.clear(key))


class CreateMissionOnceTests(unittest.TestCase):
    def setUp(self):
        self.db = make_db()
        self.addCleanup(self.db.close)
        self.missions = MissionStore(self.db)
        self.idem = make_store(self.db)

    def test_duplicate_creation_returns_the_original(self):
        key = mission_idempotency_key("write the report", scope="chat-1")
        first, created_first = create_mission_once(
            self.missions, self.idem, key, "write the report")
        second, created_second = create_mission_once(
            self.missions, self.idem, key, "write the report")
        self.assertTrue(created_first)
        self.assertFalse(created_second)
        self.assertEqual(first.id, second.id)
        self.assertEqual(len(self.missions.list()), 1)

    def test_different_key_creates_a_new_mission(self):
        key1 = mission_idempotency_key("goal one", scope="s")
        key2 = mission_idempotency_key("goal two", scope="s")
        m1, _ = create_mission_once(self.missions, self.idem, key1, "goal one")
        m2, _ = create_mission_once(self.missions, self.idem, key2, "goal two")
        self.assertNotEqual(m1.id, m2.id)


def _fake_step(name="fetch"):
    return SimpleNamespace(name=name, goal=f"goal of {name}", role="io")


class RunnerIdempotencyTests(unittest.TestCase):
    """The retry path: a completed step is replayed, a failed step retries."""

    def setUp(self):
        self.db = make_db()
        self.addCleanup(self.db.close)
        self.missions = MissionStore(self.db)
        self.idem = make_store(self.db)

    def _runner(self, **kwargs):
        ctx = SimpleNamespace(db=self.db, memory=None)
        return MissionRunner(ctx, store=self.missions, milestones=False,
                             **kwargs)

    def test_completed_step_not_reexecuted(self):
        runner = self._runner(idempotency=self.idem)
        calls = []

        def fake_run(mission, step, started):
            calls.append(step.name)
            return StepOutcome(step=step.name, ok=True,
                               payload={"data": "fetched"})

        runner._run_step_agent = fake_run
        mission = self.missions.create_new("do the thing")
        first = runner._execute_step(mission, _fake_step())
        # Simulate the crash-retry path: the step ran, the checkpoint never
        # landed, the runner drives the same step again.
        second = runner._execute_step(mission, _fake_step())
        self.assertEqual(calls, ["fetch"])  # the agent ran exactly once
        self.assertEqual(first.payload, {"data": "fetched"})
        self.assertEqual(second.payload, {"data": "fetched"})
        self.assertTrue(second.ok)

    def test_failed_step_retries(self):
        runner = self._runner(idempotency=self.idem)
        calls = []

        def fake_run(mission, step, started):
            calls.append(step.name)
            ok = len(calls) > 1
            return StepOutcome(step=step.name, ok=ok,
                               detail="" if ok else "flaky provider")

        runner._run_step_agent = fake_run
        mission = self.missions.create_new("do the thing")
        first = runner._execute_step(mission, _fake_step())
        second = runner._execute_step(mission, _fake_step())
        self.assertFalse(first.ok)
        self.assertTrue(second.ok)
        self.assertEqual(calls, ["fetch", "fetch"])  # retried, not replayed

    def test_distinct_missions_do_not_share_step_keys(self):
        runner = self._runner(idempotency=self.idem)
        calls = []
        runner._run_step_agent = lambda m, s, t: calls.append(
            (m.id, s.name)) or StepOutcome(step=s.name, ok=True)
        m1 = self.missions.create_new("first")
        m2 = self.missions.create_new("second")
        runner._execute_step(m1, _fake_step())
        runner._execute_step(m2, _fake_step())
        self.assertEqual(len(calls), 2)  # same step name, new mission → runs

    def test_no_idempotency_store_executes_every_time(self):
        runner = self._runner()  # default: idempotency=None
        self.assertIsNone(runner.idempotency)
        calls = []
        runner._run_step_agent = lambda m, s, t: calls.append(1) or StepOutcome(
            step=s.name, ok=True)
        mission = self.missions.create_new("do the thing")
        runner._execute_step(mission, _fake_step())
        runner._execute_step(mission, _fake_step())
        self.assertEqual(len(calls), 2)  # historical always-execute path


if __name__ == "__main__":
    unittest.main()
