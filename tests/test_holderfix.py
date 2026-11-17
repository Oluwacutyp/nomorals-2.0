"""Holder-fix (2026-11-17): a DB write-lock wedge must be recoverable in-process.

Live evidence: thread ``chat-telegram`` held the write lock 1658s -> 1703s;
every writer failed after 30s. Trace:
``~/workspace/holderfix-verify/HOLDERFIX_TRACE.md``. Two proven structural
holes, both fixed in ``nomorals/storage/db.py``:

(a) ``Database.transaction()`` closed only on ``except Exception`` / ``else``
    — a ``BaseException`` escaping the body skipped both close branches
    and the transaction stayed open on the writer actor forever.
(b) The writer actor never checked whether the transaction's owner thread
    was still alive — a dead owner's transaction stayed open forever and
    every later writer was deferred until its 30s timeout.

The (b) tests forge nothing: the dead owner is a real thread that issued
a real manual ``BEGIN`` and then exited without closing. The one
actor-level test (loop-top branch) is the exception, and its docstring
says why: the loop-top window is a race in production.
"""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from nomorals.core.errors import StorageError
from nomorals.storage.db import Database, _WriteJob


def _mem_db() -> Database:
    return Database(":memory:")


class BaseExceptionRollbackTests(unittest.TestCase):
    def test_keyboard_interrupt_in_transaction_rolls_back(self):
        db = _mem_db()
        self.addCleanup(db.close)
        with db.transaction():
            db.execute("CREATE TABLE t (a INTEGER)")
        seen: dict[str, BaseException | None] = {"exc": None}

        def worker() -> None:
            try:
                with db.transaction():
                    db.execute("INSERT INTO t VALUES (1)")
                    raise KeyboardInterrupt("boom")
            except BaseException as exc:  # noqa: BLE001 - the test records it
                seen["exc"] = exc

        t = threading.Thread(target=worker, name="holderfix-ki",
                             daemon=True)
        t.start()
        t.join(timeout=10)
        self.assertFalse(t.is_alive())
        self.assertIsInstance(seen["exc"], KeyboardInterrupt)
        # lock_state() never blocks, so it is the first witness: on the
        # unfixed code the holder is still named here.
        state = db.lock_state()
        self.assertIsNone(state["write_lock_holder"])
        self.assertEqual(state["open_transactions"], 0)
        # The insert was rolled back, and a later write succeeds
        # promptly instead of timing out on the wedge. (Reads on a
        # :memory: database route through the actor, so the read below
        # would itself hang on a wedged base — keep it under the
        # patched timeout too.)
        with mock.patch(
                "nomorals.storage.db._WRITE_LOCK_TIMEOUT_S", 0.5):
            with db.transaction():
                db.execute("INSERT INTO t VALUES (2)")
            self.assertEqual(db.query("SELECT a FROM t"), [{"a": 2}])

    def test_keyboard_interrupt_in_nested_transaction_rolls_back_to_savepoint(self):
        db = _mem_db()
        self.addCleanup(db.close)
        with db.transaction():
            db.execute("CREATE TABLE t (a INTEGER)")
        with db.transaction():
            db.execute("INSERT INTO t VALUES (1)")
            try:
                with db.transaction():
                    db.execute("INSERT INTO t VALUES (2)")
                    raise KeyboardInterrupt("boom")
            except KeyboardInterrupt:
                pass  # outer body survives; only the inner work is lost
        # lock_state() never blocks, so it is the first witness: on the
        # unfixed code the outer transaction is still open here (the
        # KeyboardInterrupt skipped both close branches).
        state = db.lock_state()
        self.assertIsNone(state["write_lock_holder"])
        self.assertEqual(state["open_transactions"], 0)
        with mock.patch(
                "nomorals.storage.db._WRITE_LOCK_TIMEOUT_S", 0.5):
            self.assertEqual(db.query("SELECT a FROM t"), [{"a": 1}])

    def test_keyboard_interrupt_in_readonly_direct_transaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ro.sqlite"
            setup = Database(str(path))
            try:
                with setup.transaction():
                    setup.execute("CREATE TABLE t (a INTEGER)")
            finally:
                setup.close()
            db = Database(str(path), readonly=True)
            self.addCleanup(db.close)
            self.assertFalse(db.connection_in_transaction)
            with self.assertRaises(KeyboardInterrupt), db.transaction():
                raise KeyboardInterrupt("boom")
            self.assertFalse(db.connection_in_transaction)


class DeadOwnerReapTests(unittest.TestCase):
    def _dead_owner_db(self) -> Database:
        """A db whose writer holds an open transaction owned by a dead thread.

        Real machinery throughout: a real thread issues a real manual
        BEGIN and then exits without COMMIT/ROLLBACK.
        """

        db = _mem_db()
        with db.transaction():
            db.execute("CREATE TABLE t (a INTEGER)")
        entered = threading.Event()

        def opener() -> None:
            db.execute("BEGIN IMMEDIATE")
            entered.set()

        t = threading.Thread(target=opener, name="holderfix-dead-owner",
                             daemon=True)
        t.start()
        self.assertTrue(entered.wait(10))
        t.join(timeout=10)
        self.assertFalse(t.is_alive())
        state = db.lock_state()
        self.assertIsNotNone(state["write_lock_holder"])
        self.assertEqual(state["write_lock_holder"]["thread"],
                         "holderfix-dead-owner")
        return db

    def test_dead_owner_transaction_is_reaped(self):
        db = self._dead_owner_db()
        self.addCleanup(db.close)
        # The next write from another thread succeeds promptly: the
        # actor reaps the dead owner instead of deferring to the 30s
        # timeout. On the unfixed code this raises StorageError.
        with mock.patch(
                "nomorals.storage.db._WRITE_LOCK_TIMEOUT_S", 0.5):
            db.execute("INSERT INTO t VALUES (1)")
        self.assertEqual(db.query("SELECT a FROM t"), [{"a": 1}])
        state = db.lock_state()
        self.assertIsNone(state["write_lock_holder"])
        self.assertEqual(state["open_transactions"], 0)

    def test_live_owner_transaction_is_not_reaped(self):
        db = _mem_db()
        self.addCleanup(db.close)
        with db.transaction():
            db.execute("CREATE TABLE t (a INTEGER)")
        entered = threading.Event()
        release = threading.Event()

        def holder() -> None:
            with db.transaction():
                db.execute("INSERT INTO t VALUES (1)")
                entered.set()
                release.wait(30)

        t = threading.Thread(target=holder, name="holderfix-live-owner",
                             daemon=True)
        t.start()
        self.assertTrue(entered.wait(10))
        try:
            # A live owner is never rolled back by the reaper: the
            # write times out naming the holder, and the transaction
            # is still open afterwards.
            with mock.patch(
                    "nomorals.storage.db._WRITE_LOCK_TIMEOUT_S", 0.5), \
                    self.assertRaises(StorageError) as cm, db.transaction():
                pass
            self.assertIn("holderfix-live-owner", str(cm.exception))
            state = db.lock_state()
            self.assertIsNotNone(state["write_lock_holder"])
            self.assertEqual(state["write_lock_holder"]["thread"],
                             "holderfix-live-owner")
        finally:
            release.set()
            t.join(timeout=10)
        # The live owner's work commits cleanly once released.
        self.assertEqual(db.query("SELECT a FROM t"), [{"a": 1}])
        self.assertIsNone(db.lock_state()["write_lock_holder"])

    def test_dead_owner_pending_victim_drains(self):
        db = _mem_db()
        self.addCleanup(db.close)
        with db.transaction():
            db.execute("CREATE TABLE t (a INTEGER)")
        entered = threading.Event()
        release = threading.Event()
        victim_done = threading.Event()
        victim_error: dict[str, BaseException | None] = {"exc": None}

        def holder() -> None:
            # Manual BEGIN, not `with db.transaction()`: when this
            # thread returns, the transaction stays open with a dead
            # owner — that is exactly the wedge (b) must clear. (A
            # `with` body would close on exit, and fix (a) would roll
            # a BaseException back; neither exercises the reaper.)
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT INTO t VALUES (0)")
            entered.set()
            release.wait(30)

        t = threading.Thread(target=holder, name="holderfix-pending-owner",
                             daemon=True)
        t.start()
        self.assertTrue(entered.wait(10))

        def victim() -> None:
            try:
                db.execute("INSERT INTO t VALUES (1)")
            except BaseException as exc:  # noqa: BLE001 - recorded
                victim_error["exc"] = exc
            finally:
                victim_done.set()

        # The victim arrives while the owner is alive, so its job is
        # deferred to the pending deque. Then the owner thread exits
        # with the transaction still open. The next arrival reaps the
        # dead owner (rolling back its INSERT 0) and the pending
        # victim drains.
        v = threading.Thread(target=victim, name="holderfix-victim",
                             daemon=True)
        with mock.patch(
                "nomorals.storage.db._WRITE_LOCK_TIMEOUT_S", 5.0):
            v.start()
            time.sleep(0.3)  # let the victim's job reach the pending deque
            release.set()
            t.join(timeout=10)
            self.assertFalse(t.is_alive())
            db.execute("INSERT INTO t VALUES (2)")
            self.assertTrue(victim_done.wait(10))
        self.assertIsNone(victim_error["exc"])
        self.assertEqual(
            [row["a"] for row in db.query("SELECT a FROM t ORDER BY a")],
            [1, 2])
        self.assertIsNone(db.lock_state()["write_lock_holder"])

    def test_reap_at_loop_top_drains_pending_without_new_arrival(self):
        """Actor-level: the loop-top reap needs no new arrival.

        The loop-top window (owner dies between ``_apply`` and the next
        ``_next_job`` call, while a victim already waits in pending)
        is a race in production and cannot be hit deterministically
        through the public API, so this test drives the actor directly:
        a real dead owner (real BEGIN + real thread exit) and a real
        ``_WriteJob`` placed in the actor's pending deque. The ROLLBACK
        runs on the writer connection, which is ``check_same_thread``
        False, so driving ``_next_job`` from the test thread is sound.
        """

        db = self._dead_owner_db()
        self.addCleanup(db.close)
        actor = db._writer_actor
        assert actor is not None
        job = _WriteJob("stmt", "INSERT INTO t VALUES (5)", (),
                        threading.get_ident(), "holderfix-loop-top",
                        actor.generation)
        with actor._state_lock:
            actor._pending.append(job)
        # Drive _next_job in a thread: on the unfixed code it blocks
        # forever in queue.get() (nothing reaps the dead owner), so
        # join with a timeout and fail fast instead of hanging the run.
        outcome: dict[str, object] = {}

        def drive() -> None:
            outcome["job"] = actor._next_job()

        driver = threading.Thread(target=drive, name="holderfix-driver",
                                  daemon=True)
        driver.start()
        driver.join(timeout=5.0)
        self.assertFalse(driver.is_alive())
        self.assertIs(outcome.get("job"), job)
        self.assertIsNone(actor.txn_owner)
        actor._apply(outcome["job"])  # type: ignore[arg-type]
        self.assertEqual(db.query("SELECT a FROM t"), [{"a": 5}])
        self.assertIsNone(db.lock_state()["write_lock_holder"])


class WatchdogReapTests(unittest.TestCase):
    """Fix (c): a transaction held past the limit is rolled back even
    when its owner is alive — the 2026-11-17 wedge's owner,
    ``chat-telegram``, was alive and still dispatching inbound 2800s
    in, so no liveness check could ever have freed it.

    The watchdog constants are patched small here (1.0s / 0.2s) the
    same way the dbstall suite patches ``_WRITE_LOCK_TIMEOUT_S``: both
    are read at call time, so the patch shortens the wait.
    """

    def _patch_watchdog(self):
        return mock.patch.multiple(
            "nomorals.storage.db",
            _STALE_TXN_REAP_S=1.0,
            _WATCHDOG_POLL_S=0.2,
        )

    def test_live_stale_transaction_is_reaped_by_watchdog(self):
        db = _mem_db()
        self.addCleanup(db.close)
        with db.transaction():
            db.execute("CREATE TABLE t (a INTEGER)")
        entered = threading.Event()
        outcome: list[str] = []

        def holder() -> None:
            try:
                with db.transaction():
                    db.execute("INSERT INTO t VALUES (1)")
                    entered.set()
                    time.sleep(3.0)  # past the patched 1s deadline
                    try:
                        db.execute("INSERT INTO t VALUES (2)")
                    except StorageError as exc:
                        outcome.append(str(exc))
                    else:
                        outcome.append(
                            "NO ERROR — silently de-transactionalized")
            except BaseException as exc:  # noqa: BLE001 - outcome asserted
                outcome.append(f"UNEXPECTED {exc!r}")

        with self._patch_watchdog():
            t = threading.Thread(target=holder, name="holderfix-watchdog",
                                 daemon=True)
            t.start()
            self.assertTrue(entered.wait(10))
            t.join(timeout=10)
        self.assertFalse(t.is_alive())
        # Two loud failures, both naming the watchdog: the next routed
        # call inside the body, and the transaction() exit (the owner
        # caught the first and would otherwise have exited normally,
        # pretending the commit landed).
        self.assertEqual(len(outcome), 2, f"outcome was {outcome}")
        for entry in outcome:
            self.assertIn("rolled back by the stale-transaction watchdog",
                          entry)
            self.assertIn("transaction was lost", entry)
        # The lost transaction's row never landed; a fresh transaction
        # works normally afterwards.
        self.assertEqual(db.query("SELECT COUNT(*) c FROM t")[0]["c"], 0)
        with db.transaction():
            db.execute("INSERT INTO t VALUES (3)")
        self.assertEqual(db.query("SELECT a FROM t"), [{"a": 3}])
        self.assertIsNone(db.lock_state()["write_lock_holder"])

    def test_watchdog_rollback_is_loud_at_transaction_exit(self):
        db = _mem_db()
        self.addCleanup(db.close)
        with db.transaction():
            db.execute("CREATE TABLE t (a INTEGER)")
        entered = threading.Event()
        outcome: list[str] = []

        def holder() -> None:
            try:
                with db.transaction():
                    db.execute("INSERT INTO t VALUES (1)")
                    entered.set()
                    # Past the patched 1s deadline, and NO further
                    # routed calls: the only loud failure available is
                    # the transaction() exit check.
                    time.sleep(3.0)
            except BaseException as exc:  # noqa: BLE001 - outcome asserted
                outcome.append(str(exc))

        with self._patch_watchdog():
            t = threading.Thread(target=holder, name="holderfix-watchdog-exit",
                                 daemon=True)
            t.start()
            self.assertTrue(entered.wait(10))
            t.join(timeout=10)
        self.assertFalse(t.is_alive())
        self.assertEqual(len(outcome), 1, f"outcome was {outcome}")
        self.assertIn("rolled back by the stale-transaction watchdog",
                      outcome[0])
        self.assertIn("transaction was lost", outcome[0])
        self.assertEqual(db.query("SELECT COUNT(*) c FROM t")[0]["c"], 0)

    def test_short_transaction_is_not_reaped(self):
        db = _mem_db()
        self.addCleanup(db.close)
        with db.transaction():
            db.execute("CREATE TABLE t (a INTEGER)")
        with self._patch_watchdog(), db.transaction():
            db.execute("INSERT INTO t VALUES (1)")
            time.sleep(0.3)  # inside the patched 1s deadline
        self.assertEqual(db.query("SELECT a FROM t"), [{"a": 1}])
        self.assertIsNone(db.lock_state()["write_lock_holder"])


if __name__ == "__main__":
    unittest.main()
