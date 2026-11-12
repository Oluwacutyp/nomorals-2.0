"""Single-writer actor for SQLite (2026-11-12): workers hand writes to a
queue, ONE dedicated writer thread owns the SQLite write connection and
passes the queued writes into SQLite serially; readers keep reading
directly via WAL thread-local connections.

The owner's design, in his words: "have them write somewhere and have a
way it gets passed into sqlite". SQLite allows exactly one writer at a
time — that is the format, not our code — so the ceiling on concurrent
workers was the write lock. The actor removes the lock spinning, the busy
retries and the starvation (fair FIFO); it does NOT parallelize
transactions (one transaction at a time on one connection is inherent).

Durability contract (pinned here + in db.py): the queue is a HANDOFF, not
storage. A write is acknowledged only after the writer applied it (and,
for a transaction, after COMMIT returned), so a write the caller was told
succeeded is durable to exactly the degree SQLite's configured
synchronous mode makes it. A write still sitting in the queue when the
process dies was never acknowledged and is lost; that is the loss window,
bounded by the caller's wait.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from nomorals.core.errors import StorageError
from nomorals.storage.db import Database

_TEST_TMPDIR = os.environ.get("NM_TEST_TMPDIR") or None


def _mem_db() -> Database:
    return Database(":memory:")


class _FileDbTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(dir=_TEST_TMPDIR)
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "test.db"

    def make_db(self) -> Database:
        db = Database(self.path)
        self.addCleanup(db.close)
        return db


class ConcurrentWriterTests(_FileDbTest):
    def test_eight_threads_write_without_retries_or_errors(self):
        db = self.make_db()
        db.execute("CREATE TABLE w (n INTEGER)")
        errors: list[BaseException] = []

        def worker(k: int) -> None:
            try:
                for i in range(100):
                    db.insert("w", {"n": k * 1000 + i})
            except BaseException as exc:  # noqa: BLE001 - collected, asserted
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(k,), daemon=True)
                   for k in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        self.assertEqual(errors, [])
        self.assertTrue(all(not t.is_alive() for t in threads))
        self.assertEqual(
            db.query("SELECT COUNT(*) c FROM w")[0]["c"], 800)
        # The actor serializes on one connection: no busy retries, no
        # lock failures — the whole point of the queue.
        self.assertEqual(db.stats.get("retries", 0), 0)
        self.assertEqual(db.stats.get("errors", 0), 0)

    def test_concurrent_transactions_all_commit(self):
        db = self.make_db()
        db.execute("CREATE TABLE t (a INTEGER)")
        errors: list[BaseException] = []

        def worker(k: int) -> None:
            try:
                for _ in range(20):
                    with db.transaction():
                        db.execute("INSERT INTO t VALUES (?)", (k,))
            except BaseException as exc:  # noqa: BLE001 - collected
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(k,), daemon=True)
                   for k in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        self.assertEqual(errors, [])
        self.assertEqual(
            db.query("SELECT COUNT(*) c FROM t")[0]["c"], 80)

    def test_memory_db_is_shared_across_threads(self):
        # Plain ":memory:" used to be per-connection: a reader thread
        # silently read a DIFFERENT empty database. The shared-cache URI
        # makes every connection of one Database see the same database.
        db = _mem_db()
        self.addCleanup(db.close)
        db.execute("CREATE TABLE t (a INTEGER)")
        db.insert("t", {"a": 7})
        seen: list[list[dict]] = []

        def reader() -> None:
            seen.append(db.query("SELECT a FROM t"))

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        t.join(timeout=10)
        self.assertEqual(seen, [[{"a": 7}]])


class RoutedCursorTests(unittest.TestCase):
    def test_routed_cursor_surface(self):
        db = _mem_db()
        self.addCleanup(db.close)
        db.execute("CREATE TABLE t (a INTEGER)")
        cur = db.execute("INSERT INTO t VALUES (5)")
        self.assertEqual(int(cur.lastrowid or 0), 1)
        cur = db.execute("UPDATE t SET a = 6 WHERE a = 5")
        self.assertEqual(cur.rowcount, 1)
        with db.transaction():
            db.execute("INSERT INTO t VALUES (6)")
            db.execute("INSERT INTO t VALUES (7)")
            cur = db.execute("SELECT a FROM t ORDER BY a")
            # The routed cursor returns sqlite3.Row, exactly like the
            # direct path's cursor — query() is the dict-converting layer.
            self.assertEqual(dict(cur.fetchone()), {"a": 6})
            # Rows are 6, 6, 7 (the updated row + the two inserts).
            self.assertEqual([dict(r) for r in cur.fetchmany(1)], [{"a": 6}])
            self.assertEqual([dict(r) for r in cur.fetchall()], [{"a": 7}])
            cur2 = db.execute("SELECT a FROM t ORDER BY a")
            self.assertEqual([dict(r) for r in cur2],
                             [{"a": 6}, {"a": 6}, {"a": 7}])
            with self.assertRaises(StorageError):
                cur2.execute("SELECT 1")
        self.assertEqual(
            db.query("SELECT COUNT(*) c FROM t")[0]["c"], 3)

    def test_executemany_and_executescript_routed(self):
        db = _mem_db()
        self.addCleanup(db.close)
        db.execute("CREATE TABLE t (a INTEGER)")
        cur = db.executemany("INSERT INTO t VALUES (?)", [(1,), (2,), (3,)])
        self.assertEqual(cur.rowcount, 3)
        db.executescript("CREATE TABLE u (b TEXT); INSERT INTO u VALUES ('x');")
        self.assertEqual(db.query("SELECT b FROM u"), [{"b": "x"}])
        self.assertEqual(
            db.query("SELECT COUNT(*) c FROM t")[0]["c"], 3)


class ManualTransactionTests(unittest.TestCase):
    def test_manual_begin_rollback(self):
        db = _mem_db()
        self.addCleanup(db.close)
        db.execute("CREATE TABLE t (a INTEGER)")
        self.assertFalse(db.connection_in_transaction)
        db.execute("BEGIN")
        self.assertTrue(db.connection_in_transaction)
        db.execute("INSERT INTO t VALUES (9)")
        self.assertEqual(db.query("SELECT a FROM t"), [{"a": 9}])
        db.execute("ROLLBACK")
        self.assertFalse(db.connection_in_transaction)
        self.assertEqual(db.query("SELECT COUNT(*) c FROM t")[0]["c"], 0)

    def test_double_begin_and_stray_commit_raise(self):
        db = _mem_db()
        self.addCleanup(db.close)
        db.execute("CREATE TABLE t (a INTEGER)")
        db.execute("BEGIN")
        try:
            with self.assertRaises(StorageError):
                db.execute("BEGIN")
        finally:
            db.execute("ROLLBACK")
        with self.assertRaises(StorageError):
            db.execute("COMMIT")


class BoundedAcquireTests(unittest.TestCase):
    def test_second_transaction_fails_fast_naming_holder(self):
        db = _mem_db()
        self.addCleanup(db.close)
        entered = threading.Event()
        release = threading.Event()

        def holder() -> None:
            with db.transaction():
                entered.set()
                release.wait(30)

        t = threading.Thread(target=holder, name="writer-holder",
                             daemon=True)
        t.start()
        self.assertTrue(entered.wait(10))
        try:
            with mock.patch(
                    "nomorals.storage.db._WRITE_LOCK_TIMEOUT_S", 0.5):
                t0 = time.perf_counter()
                with self.assertRaises(StorageError) as cm, db.transaction():
                    pass
                waited = time.perf_counter() - t0
            self.assertLess(waited, 10)
            self.assertIn("writer-holder", str(cm.exception))
            self.assertIn("held by thread", str(cm.exception))
        finally:
            release.set()
            t.join(timeout=10)
        with db.transaction():
            db.execute("CREATE TABLE t (a INTEGER)")
            db.execute("INSERT INTO t VALUES (1)")
        self.assertEqual(db.query("SELECT a FROM t"), [{"a": 1}])

    def test_lock_state_names_owner_thread_and_writer(self):
        db = _mem_db()
        self.addCleanup(db.close)
        entered = threading.Event()
        release = threading.Event()

        def holder() -> None:
            with db.transaction():
                entered.set()
                release.wait(30)

        t = threading.Thread(target=holder, name="writer-state",
                             daemon=True)
        t.start()
        self.assertTrue(entered.wait(10))
        try:
            state = db.lock_state()
            holder_state = state["write_lock_holder"]
            self.assertIsNotNone(holder_state)
            # "thread" stays the OWNER thread's name (the dbstall pin);
            # the writer thread is an extra field, not a replacement.
            self.assertEqual(holder_state["thread"], "writer-state")
            self.assertEqual(holder_state["writer_thread"], "nm-db-writer")
            self.assertGreaterEqual(holder_state["held_s"], 0)
            self.assertEqual(state["open_transactions"], 1)
        finally:
            release.set()
            t.join(timeout=10)
        self.assertIsNone(db.lock_state()["write_lock_holder"])


class InterruptActorTests(unittest.TestCase):
    def test_interrupt_aborts_stuck_fetchall_through_actor(self):
        # The stuck fetchall runs on the WRITER connection (inside a
        # routed transaction), so interrupt_thread(worker) must map the
        # worker to the actor's current job and interrupt the writer.
        db = _mem_db()
        self.addCleanup(db.close)
        db.execute("CREATE TABLE t (a INTEGER)")
        for i in range(100):
            db.execute("INSERT INTO t VALUES (?)", (i,))
        started = threading.Event()
        outcome: list[str] = []

        def worker() -> None:
            try:
                with db.transaction():
                    started.set()
                    db.query(
                        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL "
                        "SELECT x+1 FROM c LIMIT 30000000) "
                        "SELECT count(*) FROM c, t")
            except BaseException as exc:  # noqa: BLE001 - outcome asserted
                outcome.append(str(exc))

        t = threading.Thread(target=worker, name="writer-stuck",
                             daemon=True)
        t.start()
        self.assertTrue(started.wait(10))
        time.sleep(0.5)  # let the fetchall get deep into its stepping loop
        self.assertTrue(db.interrupt_thread(t.ident))
        t.join(timeout=10)
        self.assertFalse(t.is_alive())
        self.assertTrue(any("interrupted" in o.lower() for o in outcome),
                        f"expected 'interrupted' in {outcome}")
        self.assertIsNone(db.lock_state()["write_lock_holder"])
        with db.transaction():
            db.execute("INSERT INTO t VALUES (999)")
        self.assertEqual(
            db.query("SELECT COUNT(*) c FROM t WHERE a = 999")[0]["c"], 1)

    def test_interrupt_unknown_thread_is_false(self):
        db = _mem_db()
        self.addCleanup(db.close)
        self.assertFalse(db.interrupt_thread(999999999))


class WriterDeathTests(_FileDbTest):
    def _kill_actor(self, db: Database) -> None:
        # Kill the loop for real: the next _next_job() call raises
        # inside _run, which marks the actor dead and fails waiters.
        # The trigger job is a SELECT submitted straight to the actor
        # (no visible side effect); it unblocks the loop's queue.get(),
        # and the FOLLOWING _next_job() call is the patched one.
        from nomorals.storage.db import _WriteJob

        actor = db._ensure_writer()
        actor._next_job = lambda: (_ for _ in ()).throw(  # type: ignore[method-assign]
            RuntimeError("simulated writer death"))
        # The trigger must be APPLIED (not deferred to the pending queue)
        # so the loop calls _next_job() again and hits the patched one:
        # while a transaction is open only its owner's jobs apply, so
        # the trigger carries the transaction owner's ident.
        owner = actor.txn_owner
        if owner is None:
            owner = threading.get_ident()
        job = _WriteJob("stmt", "SELECT 1", (), owner,
                        "test", actor.generation)
        actor.enqueue(job)
        self.assertTrue(job.event.wait(5))
        for _ in range(100):
            if not actor.is_alive():
                break
            time.sleep(0.05)
        self.assertFalse(actor.is_alive())

    def test_dead_writer_fails_loud_and_restart_recovers(self):
        db = self.make_db()
        db.execute("CREATE TABLE t (a INTEGER)")
        db.insert("t", {"a": 1})
        self.assertFalse(db.restart_writer_actor())  # alive: nothing to do
        self._kill_actor(db)
        with self.assertRaises(StorageError) as cm:
            db.insert("t", {"a": 2})
        self.assertIn("writer actor is dead", str(cm.exception))
        self.assertTrue(db.restart_writer_actor())
        db.insert("t", {"a": 2})
        self.assertEqual(
            db.query("SELECT COUNT(*) c FROM t")[0]["c"], 2)

    def test_queued_waiter_gets_the_death_error(self):
        db = self.make_db()
        db.execute("CREATE TABLE t (a INTEGER)")
        entered = threading.Event()
        release = threading.Event()

        def holder() -> None:
            with db.transaction():
                entered.set()
                release.wait(30)

        t = threading.Thread(target=holder, name="writer-death-holder",
                             daemon=True)
        t.start()
        self.assertTrue(entered.wait(10))
        outcome: list[str] = []

        def waiter() -> None:
            try:
                with db.transaction():
                    pass
            except BaseException as exc:  # noqa: BLE001 - outcome asserted
                outcome.append(str(exc))

        w = threading.Thread(target=waiter, daemon=True)
        w.start()
        time.sleep(0.5)  # let the waiter's begin reach the pending queue
        try:
            self._kill_actor(db)
        finally:
            release.set()
            t.join(timeout=10)
        w.join(timeout=10)
        self.assertFalse(w.is_alive())
        self.assertTrue(
            any("writer actor died" in o for o in outcome),
            f"expected death error in {outcome}")

    def test_restart_mid_transaction_is_loud_not_silent(self):
        db = self.make_db()
        db.execute("CREATE TABLE t (a INTEGER)")
        entered = threading.Event()
        outcome: list[str] = []

        def holder() -> None:
            try:
                with db.transaction():
                    db.execute("INSERT INTO t VALUES (1)")
                    entered.set()  # the first row has LANDED on the writer
                    time.sleep(2.0)  # the restart lands mid-transaction
                    # The next routed call must fail LOUDLY: the transaction
                    # is gone, and continuing outside one would be silent
                    # de-transactionalization.
                    try:
                        db.execute("INSERT INTO t VALUES (2)")
                    except StorageError as exc:
                        outcome.append(str(exc))
                    else:
                        outcome.append("NO ERROR — silently de-transactionalized")
            except BaseException as exc:  # noqa: BLE001 - outcome asserted
                outcome.append(f"UNEXPECTED {exc!r}")

        t = threading.Thread(target=holder, daemon=True)
        t.start()
        self.assertTrue(entered.wait(10))
        time.sleep(0.3)
        self._kill_actor(db)
        self.assertTrue(db.restart_writer_actor())
        t.join(timeout=10)
        self.assertFalse(t.is_alive())
        self.assertTrue(
            any("transaction was lost" in o for o in outcome),
            f"expected 'transaction was lost' in {outcome}")
        # The lost transaction's row never landed; a fresh transaction
        # works normally afterwards.
        self.assertEqual(
            db.query("SELECT COUNT(*) c FROM t")[0]["c"], 0)
        with db.transaction():
            db.execute("INSERT INTO t VALUES (3)")
        self.assertEqual(
            db.query("SELECT COUNT(*) c FROM t")[0]["c"], 1)


class SlowWorkOutsideTransactionTests(unittest.TestCase):
    def test_slow_embed_does_not_hold_the_writer(self):
        # The call-site fix: MemoryManager.remember embeds BEFORE the
        # transaction, so a 1s embed must not serialize a concurrent
        # writer. (On the old shape — embed inside the transaction —
        # the concurrent transaction waited the full second.)
        from types import SimpleNamespace

        from nomorals.memory.manager import MemoryManager

        db = _mem_db()
        self.addCleanup(db.close)
        db.migrate()

        class SlowEmbedder:
            def embed(self, text: str) -> list[float]:
                time.sleep(1.0)
                return [0.0, 1.0]

            def embed_many(self, texts: list[str]) -> list[list[float]]:
                return [self.embed(t) for t in texts]

        ctx = SimpleNamespace(db=db, settings=None, router=None)
        mgr = MemoryManager(ctx, embedder=SlowEmbedder(),
                            vector_backend="legacy")
        done: list[str] = []
        errors: list[BaseException] = []

        def storer() -> None:
            try:
                mgr.remember("slow memory", index=True)
                done.append("stored")
            except BaseException as exc:  # noqa: BLE001 - asserted below
                errors.append(exc)

        t = threading.Thread(target=storer, daemon=True)
        t0 = time.perf_counter()
        t.start()
        # While the embed is still running (no transaction open yet),
        # a concurrent transaction must complete promptly.
        time.sleep(0.2)
        with db.transaction():
            db.execute("CREATE TABLE IF NOT EXISTS probe (a INTEGER)")
            db.execute("INSERT INTO probe VALUES (1)")
        waited = time.perf_counter() - t0
        self.assertLess(waited, 0.9)
        t.join(timeout=10)
        self.assertEqual(errors, [])
        self.assertEqual(done, ["stored"])
        self.assertEqual(
            db.query("SELECT COUNT(*) c FROM probe")[0]["c"], 1)


class CloseTests(unittest.TestCase):
    def test_close_then_write_raises(self):
        db = _mem_db()
        db.execute("CREATE TABLE t (a INTEGER)")
        db.close()
        with self.assertRaises(StorageError):
            db.execute("INSERT INTO t VALUES (1)")


if __name__ == "__main__":
    unittest.main()
