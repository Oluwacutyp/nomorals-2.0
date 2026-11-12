"""SQLite access layer.

Why SQLite, and why this shape
------------------------------
SQLite is not a compromise here. It is the only database that (a) survives
``pkg install sqlite`` on a phone with no daemon, (b) is a single file you can
back up atomically, and (c) handles hundreds of GB with WAL enabled. The
:class:`Database` wrapper exists to make three things correct that are easy to get
wrong:

1. **Connections are thread-local.** SQLite connections are not shareable across
   threads by default. A pool of thread-local connections gives real parallel
   reads under WAL without a connection-passing ceremony.
2. **One writer at a time, explicitly.** ``BEGIN IMMEDIATE`` takes the write lock
   up front, so a writer fails fast on contention instead of deadlocking halfway
   through a statement batch.
3. **Savepoints for nesting.** ``transaction()`` is re-entrant: an inner call
   becomes a ``SAVEPOINT`` so composed repository methods behave correctly.
4. **One writer *thread* (the actor).** SQLite allows exactly one writer at a
   time — that is the format, not this code — so concurrent workers used to
   serialize on a process-wide lock and fail at a 30s bound. Instead, every
   write is handed to a queue (:class:`_WriterActor`): one dedicated thread
   owns the one write connection and passes the queued writes into SQLite
   serially, while readers keep reading directly through WAL thread-local
   connections. A write is acknowledged only after the writer applied it
   (and, for a transaction, after COMMIT returned), so a write the caller
   was told succeeded is durable to exactly the degree SQLite's configured
   ``synchronous`` mode makes it; a write still sitting in the queue when
   the process dies was never acknowledged and is lost — that is the loss
   window, and it is bounded by the caller's wait. No acknowledged write is
   ever lost to the queue.
"""

from __future__ import annotations

import collections
import itertools
import os
import queue
import random
import re
import sqlite3
import threading
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from ..core.errors import ConstraintViolation, NotFound, StorageError
from ..core.logging_setup import get_logger
from ..core.style import active_theme, header, kv_lines, paint, styled_table
from .failures import record_failure

__all__ = ["AsyncDatabase", "Database", "Row", "split_sql", "transaction", "open_database"]

_log = get_logger(__name__)

Row = sqlite3.Row

#: How many total attempts a statement gets when SQLite reports the database
#: as locked/busy before the error is surfaced to the caller.
_LOCK_RETRY_ATTEMPTS = 6
#: Base delay (seconds) for exponential backoff between lock retries.
_LOCK_RETRY_BASE_DELAY_S = 0.05
#: Upper bound (seconds) for a single backoff sleep.
_LOCK_RETRY_MAX_DELAY_S = 2.0
#: Bound (seconds) for acquiring the process-wide write lock in
#: :meth:`Database.transaction`. The lock used to be acquired
#: unboundedly: a worker abandoned mid-transaction held it forever
#: and every later transaction() in the process blocked forever
#: (reproduced: a holder sleeping inside a transaction blocked a
#: second transaction() past a 60s wall). SQLite's own BEGIN IMMEDIATE
#: already serializes writers at the file level with a bounded retry,
#: so the process lock is a fast-path optimization, not correctness —
#: it must fail fast with the holder named instead of hanging the
#: process.
_WRITE_LOCK_TIMEOUT_S = 30.0


def _lock_retry_delay(attempt: int) -> float:
    """Exponential backoff with +/-50% jitter for ``database is locked`` retries.

    ``attempt`` is the 1-based retry number (not the initial try).
    """
    delay = min(_LOCK_RETRY_MAX_DELAY_S, _LOCK_RETRY_BASE_DELAY_S * (2 ** (attempt - 1)))
    return delay * (0.5 + random.random())


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


class Database:
    """Thread-safe SQLite database with migrations, WAL, and metrics.

        db = Database("data/nomorals.db")
        db.migrate()
        with db.transaction():
            db.execute("INSERT INTO t (a) VALUES (?)", (1,))
        rows = db.query("SELECT * FROM t WHERE a > ?", (0,))
    """

    def __init__(
        self,
        path: str | os.PathLike[str] = ":memory:",
        *,
        wal: bool = True,
        busy_timeout_ms: int = 5000,
        synchronous: str = "NORMAL",
        foreign_keys: bool = True,
        cache_size_kb: int = 64_000,
        timeout: float = 30.0,
        readonly: bool = False,
        wal_autocheckpoint: int = 1000,
        slow_query_threshold_s: float | None = 5.0,
        journal_size_limit: int = 64 * 1024 * 1024,
    ) -> None:
        # NB: compare via str() — a caller may pass Path(":memory:") (e.g.
        # open_database() wraps its argument in Path first), and
        # Path(":memory:") == ":memory:" is False (Path never equals str).
        # The old comparison therefore treated an in-memory database as
        # file-backed with the literal path ":memory:", and the migration
        # lock then created a file named ":memory:.migrate.lock" — a name
        # containing ':' that Windows cannot check out, which broke EVERY
        # Windows clone of the repo (checkout failed, scripts/ missing).
        self.path = None if str(path) == ":memory:" else Path(path)
        self.wal = wal and self.path is not None
        self.busy_timeout_ms = busy_timeout_ms
        self.synchronous = synchronous
        self.foreign_keys = foreign_keys
        self.cache_size_kb = cache_size_kb
        self.readonly = readonly
        self._timeout = timeout
        self.wal_autocheckpoint = wal_autocheckpoint
        #: Log a warning (and bump the slow_queries stat) for any statement
        #: slower than this. None disables. Query observability: a bot that
        #: runs for weeks accrues slow queries silently without this.
        self.slow_query_threshold_s = slow_query_threshold_s
        #: Cap on the -wal sidecar in bytes. When the WAL rewinds, pages above
        #: this limit are returned to the filesystem instead of being held
        #: for reuse — the dev.to "WAL never shrinks" fix. 0 disables.
        self.journal_size_limit = journal_size_limit

        self._local = threading.local()
        self._write_lock = threading.RLock()
        #: Which thread holds _write_lock (ident/name/since), if any. With
        #: the writer actor this record is owned by the actor's transaction
        #: state (see _WriterActor._sync_txn_state); the direct path (readonly
        #: databases) still sets it itself.
        self._write_lock_holder: dict[str, Any] | None = None
        #: Thread ident -> that thread's live connection (for
        #: interrupt_thread(): a stuck SQLite step on one thread is
        #: aborted from another via conn.interrupt()).
        self._by_thread: dict[int, sqlite3.Connection] = {}
        self._all_connections: list[sqlite3.Connection] = []
        self._all_lock = threading.Lock()
        #: Guards ``stats``: execute()/transaction() run on every worker
        #: thread and ``+=`` on a dict value is a read-modify-write.
        self._stats_lock = threading.Lock()
        self._closed = False
        self._depth = threading.local()
        #: Per-thread actor generation of the transaction that thread owns
        #: (set by transaction() after a successful begin). A restart
        #: between begin and the next routed call is loud, not silent:
        #: the transaction is gone and the caller is told so.
        self._txn_gen = threading.local()
        #: The writer actor (None until the first routed write; readonly
        #: databases never get one). Guarded by _writer_lock.
        self._writer_actor: _WriterActor | None = None
        self._writer_conn: sqlite3.Connection | None = None
        self._writer_lock = threading.Lock()
        self._writer_generation = 0
        #: Shared-cache URI for :memory: databases: one per Database
        #: instance, fixed at construction (NOT id(self) — ids are reused
        #: after GC). Every connection of this instance (readers + writer
        #: actor) shares one in-memory database; two Database instances
        #: never share. Plain ":memory:" is per-connection.
        self._memory_uri = f"file:nomorals-mem-{next(_MEMORY_DB_COUNTER)}?mode=memory&cache=shared"

        self.stats = {
            "queries": 0,
            "writes": 0,
            "transactions": 0,
            "retries": 0,
            "errors": 0,
            "slow_queries": 0,
            "query_seconds": 0.0,
        }

        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        # Open eagerly so configuration errors surface at construction.
        self._connection()

    # ── connection management ────────────────────────────────────────────────
    def _connection(self) -> sqlite3.Connection:
        if self._closed:
            raise StorageError("database is closed")
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        target = self._memory_uri if self.path is None else str(self.path)
        if self.readonly and self.path is not None:
            uri = f"file:{target}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=self._timeout)
        else:
            # Ensure the parent directory exists (fresh installs).
            if self.path is not None:
                Path(target).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                target,
                uri=self.path is None,  # :memory: uses the shared-cache URI
                timeout=self._timeout,
                isolation_level=None,  # autocommit; we drive transactions ourselves
                check_same_thread=False,
            )
        conn.row_factory = sqlite3.Row
        self._configure(conn)
        self._local.conn = conn
        with self._all_lock:
            self._all_connections.append(conn)
            self._by_thread[threading.get_ident()] = conn
        return conn

    def _configure(self, conn: sqlite3.Connection) -> None:
        pragmas = [
            ("busy_timeout", self.busy_timeout_ms),
            ("foreign_keys", "ON" if self.foreign_keys else "OFF"),
        ]
        if not self.readonly:
            pragmas += [
                ("journal_mode", "WAL" if self.wal else "DELETE"),
                ("synchronous", self.synchronous),
                ("cache_size", -self.cache_size_kb),
                ("temp_store", "MEMORY"),
                ("mmap_size", 256 * 1024 * 1024),
                ("journal_size_limit", self.journal_size_limit),
            ]
            if self.wal:
                pragmas.append(("wal_autocheckpoint", self.wal_autocheckpoint))
        for name, value in pragmas:
            try:
                conn.execute(f"PRAGMA {name}={value}")
            except sqlite3.Error as exc:  # pragma: no cover - platform dependent
                _log.debug("PRAGMA %s=%s failed: %s", name, value, exc)
        # WAL is attempted, not assumed: on filesystems that can't hold the
        # -shm sidecar (network shares, some FUSE mounts) SQLite silently
        # keeps the old journal mode. A WARN here beats a mysterious
        # "database is locked" three weeks later.
        if self.wal and not self.readonly:
            try:
                actual = conn.execute("PRAGMA journal_mode").fetchone()[0]
            except sqlite3.Error:  # pragma: no cover - introspection only
                actual = "unknown"
            if str(actual).lower() != "wal":
                _log.warning(
                    "requested WAL journal mode but database is in %r mode; "
                    "concurrent readers will block writers",
                    actual,
                )

    def close(self) -> None:
        """Close every connection opened by any thread."""
        self._closed = True
        actor = self._writer_actor
        if actor is not None:
            actor.stop()
            self._writer_actor = None
            self._writer_conn = None
        with self._all_lock:
            connections, self._all_connections = self._all_connections, []
            self._by_thread.clear()
        for conn in connections:
            try:
                conn.close()
            except sqlite3.Error as e:  # pragma: no cover
                _log.debug("connection close failed during teardown: %s", e)
        self._local.conn = None

    def release_thread(self) -> None:
        """Close and forget the calling thread's connection.

        Connections are per-thread (``threading.local``) and registered in
        ``_all_connections`` so ``close()`` can tear everything down. When a
        *one-shot* worker thread dies (typing keepalive, delayed reply,
        book build, …) its thread-local entry is dropped but the registered
        connection object — and its file descriptor — lives on until
        process exit. One leaked fd per short-lived thread adds up on a
        bot that runs for weeks, so those threads call this in a ``finally``
        block. Long-lived pool threads must NOT call it; the next
        ``_connection()`` lazily reopens, so a stray call is harmless but
        pointless.
        """
        conn = getattr(self._local, "conn", None)
        self._local.conn = None
        if conn is None:
            return
        with self._all_lock:
            if conn in self._all_connections:
                self._all_connections.remove(conn)
            self._by_thread.pop(threading.get_ident(), None)
        try:
            conn.close()
        except sqlite3.Error:  # pragma: no cover - close is best-effort here
            _log.debug("thread connection close failed", exc_info=True)

    def interrupt_thread(self, ident: int) -> bool:
        """Abort a pending SQLite step on another thread's connection.

        ``sqlite3.Connection.interrupt()`` is the one safe cross-thread
        prod: a worker stuck inside a long stepping loop (fetchall over
        a huge result) aborts at the next step with
        ``OperationalError("interrupted")`` — measured 0.19s — and its
        transaction() finally-block then rolls back and releases the
        write lock, so the abandoned thread unwinds instead of
        lingering with the lock held. It does NOT abort a step blocked
        inside a user-defined function (SQLite only checks the flag
        between VDBE steps) — that limitation is real and named here.
        Never raises; False when the thread has no live connection.

        With the writer actor the stuck step usually runs on the WRITER
        connection, not the worker's own: when *ident* owns the job the
        writer is currently applying, the writer connection is
        interrupted (and the owner's queued jobs are cancelled so their
        waits unwind immediately). A worker's own connection is
        interrupted first — a stuck read on it is the worker's own step.
        An idle writer connection is NOT interrupted: a stale interrupt
        flag would abort the next innocent writer's statement.
        """
        try:
            interrupted = False
            with self._all_lock:
                conn = self._by_thread.get(int(ident))
            if conn is not None:
                conn.interrupt()
                interrupted = True
            actor = self._writer_actor
            if actor is not None:
                if actor.current_job_owner() == int(ident):
                    writer_conn = self._writer_conn
                    if writer_conn is not None and writer_conn is not conn:
                        writer_conn.interrupt()
                        interrupted = True
                if actor.cancel_pending(int(ident)) > 0:
                    interrupted = True
            return interrupted
        except Exception:  # noqa: BLE001 - teardown is best-effort
            _log.debug("interrupt_thread failed", exc_info=True)
            return False

    def lock_state(self) -> dict[str, Any]:
        """Snapshot of the DB lock state for a stuck-call report.

        Names the write-lock holder (thread + how long held), the
        journal mode actually achieved, the WAL sidecar size, and how
        many connections hold an open transaction — the facts that
        decide whether a stuck job is lock-blocked, WAL-starved, or
        elsewhere. Never raises; best-effort reads only.
        """
        holder: dict[str, Any] | None = None
        with self._all_lock:
            raw = self._write_lock_holder
            if raw is not None:
                holder = {
                    "thread": str(raw.get("name") or raw.get("ident") or "?"),
                    "ident": raw.get("ident"),
                    "held_s": round(time.monotonic() - float(
                        raw.get("since") or time.monotonic()), 1),
                }
            open_txns = 0
            for conn in list(self._all_connections):
                try:
                    if conn.in_transaction:
                        open_txns += 1
                except Exception:  # noqa: BLE001 - closed conn
                    pass
            n_conns = len(self._all_connections)
        # With the writer actor the holder is the transaction's OWNER
        # thread (its name stays in "thread", as before); the writer
        # thread + the statement it is applying are extra fields.
        actor = self._writer_actor
        if holder is not None and actor is not None:
            work = actor.current_work()
            holder["writer_thread"] = "nm-db-writer"
            holder["writer_ident"] = actor.thread_ident
            holder["current_sql"] = work.get("sql", "") if work else ""
        try:
            journal = self.journal_mode()
        except Exception:  # noqa: BLE001
            journal = "unknown"
        try:
            wal_bytes = self.wal_size_bytes()
        except Exception:  # noqa: BLE001
            wal_bytes = -1
        return {
            "write_lock_holder": holder,
            "journal_mode": journal,
            "wal_bytes": wal_bytes,
            "connections": n_conns,
            "open_transactions": open_txns,
        }

    # ── writer actor ─────────────────────────────────────────────────────
    def _writer_connection(self) -> sqlite3.Connection:
        """The writer actor's own connection, created on the actor thread.

        Registered like any other connection (so close() and interrupt()
        see it), and kept in ``_writer_conn`` so the actor can apply jobs
        on it. Called once, from the actor thread, before its loop starts.
        """

        conn = self._connection()
        self._writer_conn = conn
        return conn

    def _start_writer_actor(self) -> _WriterActor:
        """Create + start a fresh actor. Caller must hold _writer_lock."""

        self._writer_generation += 1
        actor = _WriterActor(self, generation=self._writer_generation)
        self._writer_actor = actor
        actor.start()
        return actor

    def _ensure_writer(self) -> _WriterActor:
        """The live writer actor, starting one lazily on first write.

        Readonly databases never get one (None is returned by the
        callers' readonly check, not here). A DEAD actor raises loudly
        naming it — it is never silently restarted; recovery is the
        operator's explicit restart_writer_actor().
        """

        actor = self._writer_actor
        if actor is not None and actor.is_alive():
            return actor
        with self._writer_lock:
            actor = self._writer_actor
            if actor is not None and actor.is_alive():
                return actor
            if actor is not None:
                raise StorageError(
                    "database writer actor is dead"
                    + (f" ({actor.dead_error})" if actor.dead_error else "")
                    + " — call restart_writer_actor()")
            if self._closed:
                raise StorageError("database is closed")
            return self._start_writer_actor()

    def restart_writer_actor(self) -> bool:
        """Restart a DEAD writer actor (explicit operator recovery).

        Closes the old writer connection (rolling back any dangling
        transaction it held and releasing the file lock) and starts a
        fresh actor on a new connection. Returns False when the current
        actor is alive (nothing to restart). Any transaction a caller
        had open across the restart is GONE — its next routed call
        raises StorageError naming the lost transaction instead of
        silently continuing outside one. Raises StorageError when the
        database is closed.
        """

        if self._closed:
            raise StorageError("database is closed")
        with self._writer_lock:
            actor = self._writer_actor
            if actor is not None and actor.is_alive():
                return False
            old = self._writer_conn
            self._writer_conn = None
            if old is not None:
                with self._all_lock:
                    if old in self._all_connections:
                        self._all_connections.remove(old)
                    for ident, conn in list(self._by_thread.items()):
                        if conn is old:
                            self._by_thread.pop(ident, None)
                try:
                    old.close()
                except sqlite3.Error as exc:
                    _log.debug("old writer connection close failed: %s", exc)
            self._start_writer_actor()
            return True

    def _in_routed_txn(self) -> bool:
        """True when this thread owns a transaction on the writer actor.

        Covers both shapes: a transaction() block (thread-local depth)
        and a manual BEGIN issued via execute() (the actor's txn_owner
        is this thread). While True, every statement — reads included —
        routes to the writer connection, so the transaction sees its own
        writes.
        """

        if getattr(self._depth, "value", 0) > 0:
            return True
        actor = self._writer_actor
        return actor is not None and actor.txn_owner == threading.get_ident()

    def _check_txn_generation(self) -> None:
        """Fail loudly if the writer actor was restarted mid-transaction."""

        if getattr(self._depth, "value", 0) <= 0:
            return
        actor = self._writer_actor
        gen = getattr(self._txn_gen, "value", None)
        if actor is not None and gen is not None and gen != actor.generation:
            raise StorageError(
                "writer actor was restarted; the transaction was lost")

    def _holder_timeout_error(self, what: str) -> StorageError:
        """The bounded-acquire error, naming the holder like c5bebb95."""

        actor = self._writer_actor
        work = actor.current_work() if actor is not None else None
        if work is not None:
            who = work.get("thread") or work.get("ident") or "?"
            held = float(work.get("held_s") or 0.0)
            return StorageError(
                f"database write lock held by thread {who} for {held:.1f}s "
                f"(>{_WRITE_LOCK_TIMEOUT_S:g}s wait) — a previous job may "
                f"have been abandoned mid-transaction")
        return StorageError(
            f"database write for {what} timed out after "
            f"{_WRITE_LOCK_TIMEOUT_S:g}s (writer actor did not respond)")

    def _convert_execute_error(self, exc: BaseException) -> BaseException:
        """Map a raw error delivered by the actor onto the public types."""

        if isinstance(exc, (StorageError, ConstraintViolation)):
            return exc
        if isinstance(exc, sqlite3.IntegrityError):
            return ConstraintViolation(str(exc))
        if isinstance(exc, sqlite3.OperationalError):
            return StorageError(str(exc), retryable=_is_retryable_sqlite(exc))
        if isinstance(exc, sqlite3.Error):
            return StorageError(str(exc))
        return exc

    def _enqueue_job(self, kind: str, sql: str, params: Any) -> _WriteJob:
        """Build + enqueue a job for this thread on the live actor."""

        actor = self._ensure_writer()
        job = _WriteJob(kind, sql, params, threading.get_ident(),
                        threading.current_thread().name, actor.generation)
        actor.enqueue(job)
        return job

    def _wait_job(self, job: _WriteJob, *, what: str,
                  timeout: float | None = None) -> Any:
        """Wait for a job, converting its outcome (or a timeout) loudly.

        ``timeout`` defaults to the module constant, resolved at CALL
        time (never as a def-time default) so a patched
        ``_WRITE_LOCK_TIMEOUT_S`` — the dbstall suite patches it to 0.5
        — actually shortens the wait.
        """

        if timeout is None:
            timeout = _WRITE_LOCK_TIMEOUT_S
        if not job.event.wait(timeout):
            job.cancelled = True
            raise self._holder_timeout_error(what)
        if job.error is not None:
            raise self._convert_execute_error(job.error)
        return job.result

    def _routed_execute(self, sql: str, params: Any = ()) -> _RoutedCursor:
        self._check_txn_generation()
        job = self._enqueue_job("stmt", sql, params)
        return self._wait_job(job, what=f"statement {sql[:60]!r}")

    def _routed_executemany(self, sql: str, seq: Any) -> _RoutedCursor:
        self._check_txn_generation()
        rows = list(seq)
        job = self._enqueue_job("executemany", sql, rows)
        return self._wait_job(job, what=f"statement {sql[:60]!r}")

    def _routed_script(self, script: str) -> None:
        self._check_txn_generation()
        job = self._enqueue_job("script", script, ())
        self._wait_job(job, what="script")

    def _routed_txn_control(self, sql: str) -> _RoutedCursor:
        """BEGIN/COMMIT/ROLLBACK/SAVEPOINT/RELEASE issued via execute()."""

        word = _sql_first_word(sql)
        if word == "BEGIN":
            self._check_txn_generation()
            job = self._enqueue_job("begin", sql, ())
            return self._wait_job(job, what="transaction begin")
        if word in ("COMMIT", "ROLLBACK", "END"):
            job = self._enqueue_job(word.lower() if word != "END" else "commit",
                                    sql, ())
            result = self._wait_job(job, what=f"transaction {word.lower()}")
            return result
        self._check_txn_generation()
        job = self._enqueue_job("stmt", sql, ())
        return self._wait_job(job, what=f"statement {sql[:60]!r}")

    def _actor_begin(self, *, immediate: bool) -> None:
        """Open a transaction on the writer actor (bounded wait)."""

        sql = "BEGIN IMMEDIATE" if immediate else "BEGIN"
        job = self._enqueue_job("begin", sql, ())
        self._wait_job(job, what="transaction begin")

    def _actor_close(self, kind: str) -> None:
        """Close a transaction on the writer actor; best-effort like _safe.

        Waits in _WRITE_LOCK_TIMEOUT_S slices (3 max) with loud logging,
        then gives up — the transaction stays open on the writer and the
        holder naming + the next begin's failure surface it. Never raises
        out of a transaction() exit path. A close job is never cancelled:
        a close that never applies wedges the actor forever.
        """

        actor = self._writer_actor
        if actor is None or not actor.is_alive():
            return
        try:
            job = _WriteJob(kind, kind.upper(), (),
                            threading.get_ident(), threading.current_thread().name,
                            actor.generation)
            actor.enqueue(job)
        except StorageError as exc:
            _log.warning("transaction %s not enqueued: %s", kind, exc)
            return
        for _ in range(3):
            if job.event.wait(_WRITE_LOCK_TIMEOUT_S):
                if job.error is not None:
                    _log.warning("transaction %s failed: %s", kind, job.error)
                return
            _log.warning("transaction %s close wait exceeded %gs (writer busy?)",
                         kind, _WRITE_LOCK_TIMEOUT_S)
        _log.warning(
            "transaction %s close abandoned after %gs; the writer actor "
            "still holds the transaction", kind, 3 * _WRITE_LOCK_TIMEOUT_S)

    def __enter__(self) -> "Database":
        return self
    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def is_memory(self) -> bool:
        return self.path is None

    def connection_count(self) -> int:
        with self._all_lock:
            return len(self._all_connections)

    def _bump_stat(self, key: str, amount: float = 1) -> None:
        """Thread-safe stats increment (see ``_stats_lock``)."""
        with self._stats_lock:
            self.stats[key] = self.stats.get(key, 0) + amount

    # ── execution ────────────────────────────────────────────────────────────
    def _with_lock_retry(self, fn: Callable[[], Any]) -> Any:
        """Run ``fn``; retry when SQLite reports the database as locked/busy.

        SQLite's own ``busy_timeout`` usually absorbs contention, but a second
        process, a second :class:`Database` on the same file, or a long
        checkpoint can hold the write lock past the timeout. Retrying here
        (bounded, with backoff + jitter) turns those transient collisions into
        a short pause instead of a failed write.
        """
        attempt = 0
        while True:
            try:
                return fn()
            except sqlite3.OperationalError as exc:
                if not _is_retryable_sqlite(exc) or attempt + 1 >= _LOCK_RETRY_ATTEMPTS:
                    raise
                attempt += 1
                self._bump_stat("retries")
                _log.debug("database is locked, retry %d/%d", attempt, _LOCK_RETRY_ATTEMPTS - 1)
                time.sleep(_lock_retry_delay(attempt))

    def execute(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> sqlite3.Cursor | _RoutedCursor:
        """Run a single statement. Returns the cursor (use ``.lastrowid``/``.rowcount``).

        Writes — and every statement inside a transaction, or on a
        :memory: database — route to the writer actor and return a
        :class:`_RoutedCursor` over the materialized rows; plain reads on
        a file-backed database run directly on the caller's WAL connection.
        """
        if not self.readonly and (
                self._in_routed_txn() or self.path is None or _is_write_sql(sql)):
            if _is_txn_control_sql(sql):
                return self._routed_txn_control(sql)
            return self._routed_execute(sql, params)
        started = time.perf_counter()
        try:
            cursor = self._with_lock_retry(lambda: self._connection().execute(sql, params))
        except sqlite3.IntegrityError as exc:
            self._bump_stat("errors")
            raise ConstraintViolation(str(exc)) from exc
        except sqlite3.OperationalError as exc:
            self._bump_stat("errors")
            raise StorageError(str(exc), retryable=_is_retryable_sqlite(exc)) from exc
        except sqlite3.Error as exc:
            self._bump_stat("errors")
            raise StorageError(str(exc)) from exc
        finally:
            elapsed = time.perf_counter() - started
            self._bump_stat("queries")
            self._bump_stat("query_seconds", elapsed)
            self._note_slow(sql, elapsed)
        if sql.lstrip()[:6].upper() in {"INSERT", "UPDATE", "DELETE", "REPLAC"}:
            self._bump_stat("writes")
        return cursor

    def _note_slow(self, sql: str, elapsed: float) -> None:
        """Record and warn on statements slower than the configured threshold."""
        threshold = self.slow_query_threshold_s
        if threshold is None or elapsed < threshold:
            return
        self._bump_stat("slow_queries")
        preview = " ".join(sql.split())[:160]
        _log.warning("slow query (%.2fs > %.2fs): %s", elapsed, threshold, preview)

    def executemany(self, sql: str, seq: Iterable[Sequence[Any]]) -> sqlite3.Cursor | _RoutedCursor:
        if not self.readonly:
            return self._routed_executemany(sql, seq)
        started = time.perf_counter()
        try:
            cursor = self._with_lock_retry(lambda: self._connection().executemany(sql, list(seq)))
        except sqlite3.IntegrityError as exc:
            self._bump_stat("errors")
            raise ConstraintViolation(str(exc)) from exc
        except sqlite3.OperationalError as exc:
            self._bump_stat("errors")
            raise StorageError(str(exc), retryable=_is_retryable_sqlite(exc)) from exc
        except sqlite3.Error as exc:
            self._bump_stat("errors")
            raise StorageError(str(exc)) from exc
        finally:
            elapsed = time.perf_counter() - started
            self._bump_stat("queries")
            self._bump_stat("query_seconds", elapsed)
            self._note_slow(sql, elapsed)
        self._bump_stat("writes")
        return cursor

    def executescript(self, script: str) -> None:
        """Run a multi-statement script OUTSIDE any transaction.

        Warning: :meth:`sqlite3.Connection.executescript` issues an implicit COMMIT
        before running. Never call this inside :meth:`transaction` — use
        :meth:`execute_statements` instead, which keeps atomicity.
        """
        if not self.readonly:
            self._routed_script(script)
            return
        try:
            self._connection().executescript(script)
        except sqlite3.Error as exc:
            self._bump_stat("errors")
            raise StorageError(str(exc)) from exc

    def execute_statements(self, script: str) -> int:
        """Run every statement in ``script`` individually, in the current transaction.

        This is the transaction-safe counterpart to :meth:`executescript`, and the
        only correct way to apply DDL migrations: a failure partway through leaves
        the surrounding transaction intact so the caller can roll the whole thing
        back. Returns the number of statements executed.
        """
        count = 0
        for statement in split_sql(script):
            self.execute(statement)
            count += 1
        return count

    def query(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> list[dict[str, Any]]:
        """Return every row as a dict."""
        return [_row_to_dict(row) for row in self.query_rows(sql, params)]

    def query_rows(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> list[sqlite3.Row]:
        cursor = self.execute(sql, params)
        try:
            return cursor.fetchall()
        finally:
            cursor.close()

    def query_one(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> dict[str, Any] | None:
        cursor = self.execute(sql, params)
        try:
            row = cursor.fetchone()
        finally:
            cursor.close()
        return _row_to_dict(row) if row is not None else None

    def scalar(self, sql: str, params: Sequence[Any] | dict[str, Any] = (), default: Any = None) -> Any:
        """Return the first column of the first row, or ``default``.

        ``default`` is used both when there is no row *and* when the value is
        SQL NULL. That is deliberate: nearly every call site does
        ``int(db.scalar(...))``, and handing those a None turns a missing value
        into a TypeError somewhere far from here. Use :meth:`query_one` when you
        genuinely need to distinguish NULL from absent.
        """
        row = self.query_one(sql, params)
        if row is None:
            return default
        value = next(iter(row.values()), None)
        return default if value is None else value

    def insert(self, table: str, values: dict[str, Any]) -> int:
        """Insert one row, returning the new rowid."""
        columns = list(values)
        quoted = ", ".join('"' + c + '"' for c in columns)
        placeholders = ", ".join("?" for _ in columns)
        sql = f'INSERT INTO "{table}" ({quoted}) VALUES ({placeholders})'
        cursor = self.execute(sql, [values[c] for c in columns])
        return int(cursor.lastrowid or 0)

    def update(self, table: str, values: dict[str, Any], where: str, params: Sequence[Any] = ()) -> int:
        """Update rows; returns the number of rows changed."""
        if not values:
            return 0
        assignments = ", ".join(f'"{c}" = ?' for c in values)
        sql = f'UPDATE "{table}" SET {assignments} WHERE {where}'
        cursor = self.execute(sql, [*values.values(), *params])
        return cursor.rowcount

    def delete(self, table: str, where: str, params: Sequence[Any] = ()) -> int:
        cursor = self.execute(f'DELETE FROM "{table}" WHERE {where}', params)
        return cursor.rowcount

    # ── transactions ─────────────────────────────────────────────────────────
    @contextmanager
    def transaction(self, *, immediate: bool = True) -> Iterator["Database"]:
        """Re-entrant transaction. Nested calls become SAVEPOINTs.

        ``immediate=True`` acquires the write lock at BEGIN, so two writers fail
        fast rather than deadlocking after doing half their work. Writes go
        through the writer actor (see the module docstring): the begin waits
        on the actor's queue with the same 30s bound, and the holder named
        on timeout is the transaction's owner thread.
        """
        depth = getattr(self._depth, "value", 0)
        if depth > 0:
            savepoint = f"sp_{depth}"
            self.execute(f"SAVEPOINT {savepoint}")
            self._depth.value = depth + 1
            try:
                yield self
            except Exception:
                self.execute(f"ROLLBACK TO {savepoint}")
                self.execute(f"RELEASE {savepoint}")
                self._depth.value = depth
                raise
            else:
                self.execute(f"RELEASE {savepoint}")
                self._depth.value = depth
            return

        if self.readonly:
            with self._transaction_direct(immediate=immediate) as handle:
                yield handle
            return

        self._actor_begin(immediate=immediate)
        self._depth.value = 1
        actor = self._writer_actor
        self._txn_gen.value = actor.generation if actor is not None else None
        self._bump_stat("transactions")
        try:
            yield self
        except Exception:
            self._actor_close("rollback")
            raise
        else:
            self._actor_close("commit")
        finally:
            self._depth.value = 0
            self._txn_gen.value = None

    @contextmanager
    def _transaction_direct(self, *, immediate: bool = True) -> Iterator[Database]:
        """The pre-actor transaction body (RLock + own connection).

        Used only by readonly databases, which have no writer actor. The
        RLock serializes writers the old way; the bounded acquire and the
        holder naming are unchanged.
        """
        if not self._write_lock.acquire(timeout=_WRITE_LOCK_TIMEOUT_S):
            # Bounded, with the holder named: an abandoned worker
            # mid-transaction used to hold this lock forever and every
            # later transaction() in the process blocked forever with
            # it. Fail fast instead — the caller (a job) fails
            # honestly and the alert names who holds the lock.
            holder = self._write_lock_holder
            if holder is not None:
                held = time.monotonic() - float(
                    holder.get("since") or time.monotonic())
                who = str(holder.get("name") or holder.get("ident") or "?")
                raise StorageError(
                    f"database write lock held by thread {who} for "
                    f"{held:.1f}s (>{_WRITE_LOCK_TIMEOUT_S:g}s wait) — "
                    "a previous job may have been abandoned "
                    "mid-transaction")
            raise StorageError(
                "database write lock not acquired in "
                f"{_WRITE_LOCK_TIMEOUT_S:g}s")
        with self._all_lock:
            self._write_lock_holder = {
                "ident": threading.get_ident(),
                "name": threading.current_thread().name,
                "since": time.monotonic(),
            }
        try:
            self.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            self._depth.value = 1
            self._bump_stat("transactions")
            try:
                yield self
            except Exception:
                self._safe("ROLLBACK")
                raise
            else:
                self._safe("COMMIT")
            finally:
                self._depth.value = 0
        finally:
            with self._all_lock:
                if self._write_lock_holder is not None and \
                        self._write_lock_holder.get("ident") == \
                        threading.get_ident():
                    self._write_lock_holder = None
            self._write_lock.release()

    def _safe(self, sql: str) -> None:
        # A COMMIT that hits SQLITE_BUSY leaves the transaction open, so a
        # blind retry is the correct action (not a fresh BEGIN).
        #
        # If this thread owns a transaction that is dangling on the writer
        # actor (its COMMIT was swallowed, or it was opened by a manual
        # BEGIN), the COMMIT/ROLLBACK belongs on the writer connection,
        # not this thread's own connection — route it there. The actor's
        # own close path calls _safe from the actor thread, where the
        # owner check below is False, so it runs directly on the writer
        # connection as before. Never raises.
        try:
            actor = self._writer_actor
            word = _sql_first_word(sql)
            if (actor is not None and actor.is_alive()
                    and word in ("COMMIT", "ROLLBACK")
                    and actor.txn_owner == threading.get_ident()
                    and threading.get_ident() != actor.thread_ident):
                job = _WriteJob(word.lower(), sql, (),
                                threading.get_ident(),
                                threading.current_thread().name,
                                actor.generation)
                try:
                    actor.enqueue(job)
                except StorageError as exc:
                    _log.warning("%s not enqueued: %s", sql, exc)
                    return
                if not job.event.wait(_WRITE_LOCK_TIMEOUT_S):
                    _log.warning("%s close wait exceeded %gs", sql,
                                 _WRITE_LOCK_TIMEOUT_S)
                elif job.error is not None:
                    _log.warning("%s failed: %s", sql, job.error)
                return
        except Exception:  # noqa: BLE001 - _safe never raises; direct path below
            _log.debug("_safe actor routing failed; using direct path", exc_info=True)
        try:
            self._with_lock_retry(lambda: self._connection().execute(sql))
        except sqlite3.Error as exc:  # pragma: no cover - rollback of a broken txn
            _log.warning("%s failed: %s", sql, exc)

    @property
    def in_transaction(self) -> bool:
        return getattr(self._depth, "value", 0) > 0

    @property
    def connection_in_transaction(self) -> bool:
        """True while the SQLite connection itself holds an open transaction.

        Unlike :attr:`in_transaction` — the logical nesting depth, which the
        transaction manager resets as it unwinds — this reads SQLite's own
        state, so a COMMIT that failed and was swallowed by :meth:`_safe`
        (``SQLITE_FULL`` on a full disk is the case that matters) still shows
        up here as a dangling transaction whose work never landed. The
        migration runner checks it to prove a commit actually committed.
        With the writer actor the transaction lives on the writer
        connection, so a routed transaction owned by this thread counts
        here too — the runner's check must see it.
        """
        try:
            if self._in_routed_txn():
                return True
            return bool(self._connection().in_transaction)
        except Exception:  # noqa: BLE001 - closed/unopened db has no open txn
            return False

    # ── maintenance ──────────────────────────────────────────────────────────
    def pragma(self, name: str, value: Any | None = None) -> Any:
        """Read (or set) a PRAGMA. Read form returns the scalar value."""
        if value is None:
            return self.scalar(f"PRAGMA {name}")
        self.execute(f"PRAGMA {name}={value}")
        return self.scalar(f"PRAGMA {name}")

    def journal_mode(self) -> str:
        """The journal mode SQLite actually achieved for this database."""
        return str(self.scalar("PRAGMA journal_mode", default="unknown"))

    def optimize(self, mask: int = 0x10002) -> None:
        """Run ``PRAGMA optimize`` so the query planner sees fresh statistics.

        sqlite.org's recipe: run after every schema change / CREATE INDEX, and
        periodically on long-lived connections. :meth:`migrate` calls this
        automatically when it applies anything.
        """
        try:
            self.execute(f"PRAGMA optimize={mask}")
        except sqlite3.Error as exc:  # pragma: no cover - platform dependent
            _log.debug("PRAGMA optimize failed: %s", exc)

    def wal_size_bytes(self) -> int:
        """Current size of the -wal sidecar (0 when not in WAL or no sidecar)."""
        if self.path is None:
            return 0
        sidecar = self.path.with_name(self.path.name + "-wal")
        try:
            return sidecar.stat().st_size
        except OSError:
            return 0

    def backup_to(self, dest: str | os.PathLike[str], *, verify: bool = True) -> Path:
        """Copy this database to ``dest`` with the online backup API.

        Safe to call while the database is live and being written to —
        unlike ``shutil.copy``, which can capture a torn WAL state. When
        ``verify`` is true the copy gets an ``integrity_check`` before this
        returns; a failed verification deletes the copy and raises.
        """
        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        source = self._connection()
        target = sqlite3.connect(str(dest_path))
        try:
            source.backup(target)
        finally:
            target.close()
        if verify:
            probe = Database(dest_path, readonly=True)
            try:
                result = probe.integrity_check()
            finally:
                probe.close()
            if result.strip().lower() != "ok":
                try:
                    dest_path.unlink()
                except OSError as e:  # pragma: no cover - best effort
                    record_failure('db.Database.backup_to', e)
                raise StorageError(f"backup to {dest_path} failed integrity_check: {result}")
        return dest_path

    def checkpoint(self, mode: str = "TRUNCATE") -> tuple[int, int]:
        """Force a WAL checkpoint. Returns (busy, log_pages)."""
        if not self.wal:
            return (0, 0)
        row = self.query_one(f"PRAGMA wal_checkpoint({mode})")
        if row is None:
            return (0, 0)
        values = list(row.values())
        return (int(values[1] or 0), int(values[2] or 0))

    def wal_health(self) -> dict[str, Any]:
        """WAL health snapshot for the maintenance tick.

        A long-running read transaction pins the checkpointer: ``log_pages``
        keeps growing while ``checkpointed_pages`` stalls, and the -wal file
        never rewinds. A scheduler tick that calls this and alerts when
        ``lag_pages`` grows monotonically catches the leak weeks before disk
        pressure does. (dev.to "Why your SQLite WAL file never shrinks".)
        """
        health: dict[str, Any] = {
            "journal_mode": self.journal_mode(),
            "wal_size_bytes": self.wal_size_bytes(),
            "busy": 0,
            "log_pages": 0,
            "checkpointed_pages": 0,
            "lag_pages": 0,
        }
        if not self.wal:
            return health
        try:
            row = self.query_one("PRAGMA wal_checkpoint(PASSIVE)")
            if row is not None:
                values = list(row.values())
                health["busy"] = int(values[0] or 0)
                health["log_pages"] = int(values[1] or 0)
                health["checkpointed_pages"] = int(values[2] or 0)
                health["lag_pages"] = max(
                    0, health["log_pages"] - health["checkpointed_pages"]
                )
        except sqlite3.Error as e:  # pragma: no cover - introspection only
            record_failure('db.Database.wal_health', e)
        return health

    def checkpoint_maintenance(self) -> dict[str, Any]:
        """Idle-time WAL maintenance: TRUNCATE checkpoint with busy retry.

        Call from a quiet-period scheduler tick (not the write hot path):
        TRUNCATE rewinds the -wal file to zero bytes when no reader pins it.
        Returns the checkpoint outcome; ``busy=True`` means a reader held the
        log — retry on the next tick rather than forcing it.
        """
        before = self.wal_size_bytes()
        busy = log_pages = 0
        for _ in range(3):
            busy, log_pages = self.checkpoint("TRUNCATE")
            if not busy:
                break
            time.sleep(0.5)
        after = self.wal_size_bytes()
        return {
            "busy": bool(busy),
            "log_pages": log_pages,
            "wal_bytes_before": before,
            "wal_bytes_after": after,
            "reclaimed_bytes": max(0, before - after),
        }

    def explain(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> list[dict[str, Any]]:
        """``EXPLAIN QUERY PLAN`` rows as dicts — makes slow queries actionable."""
        cursor = self.execute("EXPLAIN QUERY PLAN " + sql, params)
        try:
            rows = cursor.fetchall()
        finally:
            cursor.close()
        return [
            {
                "id": r[0],
                "parent": r[1],
                "notused": r[2],
                "detail": r[3],
            }
            for r in rows
        ]

    def foreign_key_check(self) -> list[dict[str, Any]]:
        """Rows violating foreign keys (empty = clean). Run after stress."""
        cursor = self.execute("PRAGMA foreign_key_check")
        try:
            rows = cursor.fetchall()
        finally:
            cursor.close()
        return [
            {"table": r[0], "rowid": r[1], "fk_table": r[2], "fk_index": r[3]}
            for r in rows
        ]

    def vacuum(self) -> None:
        self.execute("VACUUM")

    def integrity_check(self) -> str:
        return str(self.scalar("PRAGMA integrity_check", default="unknown"))

    def table_exists(self, name: str) -> bool:
        return (
            self.scalar(
                "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
                (name,),
            )
            is not None
        )

    def tables(self) -> list[str]:
        rows = self.query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        return [r["name"] for r in rows]

    def table_info(self, name: str) -> list[dict[str, Any]]:
        return self.query(f'PRAGMA table_info("{name}")')

    def row_count(self, name: str) -> int:
        return int(self.scalar(f'SELECT COUNT(*) FROM "{name}"', default=0))

    def migrate(self) -> "MigrationSummary":
        """Apply all pending migrations, then refresh the query planner."""
        from .migrations import MIGRATIONS
        from .schema import MigrationRunner

        runner = MigrationRunner(self)
        applied = runner.apply_all(MIGRATIONS)
        if applied:
            self.optimize()
        return MigrationSummary(applied=applied, version=runner.current_version())

    def stats_snapshot(self) -> dict[str, Any]:
        # Note: tables() issues a query, so the snapshot itself bumps the
        # "queries" counter by one. The stats merge is taken under
        # _stats_lock for a consistent read.
        info = {
            "path": str(self.path) if self.path else ":memory:",
            "connections": self.connection_count(),
            "tables": len(self.tables()),
            "journal_mode": self.journal_mode(),
            "wal_size_bytes": self.wal_size_bytes(),
        }
        try:
            info["page_count"] = int(self.scalar("PRAGMA page_count", default=0))
            info["freelist_count"] = int(self.scalar("PRAGMA freelist_count", default=0))
            info["page_size"] = int(self.scalar("PRAGMA page_size", default=0))
        except sqlite3.Error as e:  # pragma: no cover - introspection only
            record_failure('db.Database.stats_snapshot', e)
        if self.path is not None and self.path.exists():
            info["size_bytes"] = self.path.stat().st_size
        with self._stats_lock:
            return {**self.stats, **info}

    def format_stats(self, theme: Any = None) -> str:
        """Human-readable database overview through the shared style layer."""
        theme = theme or active_theme()
        snap = self.stats_snapshot()
        wal = self.wal_health() if not self.is_memory else {}
        lines = [
            header("database", theme=theme),
            *kv_lines(
                {
                    "path": snap.get("path"),
                    "journal": snap.get("journal_mode"),
                    "tables": snap.get("tables"),
                    "connections": snap.get("connections"),
                    "size": _fmt_bytes(snap.get("size_bytes", 0)),
                    "wal": _fmt_bytes(wal.get("wal_size_bytes", 0)) if wal else "n/a",
                    "queries": snap.get("queries"),
                    "writes": snap.get("writes"),
                    "transactions": snap.get("transactions"),
                    "retries": snap.get("retries"),
                    "errors": paint(str(snap.get("errors")), "error", theme=theme)
                    if snap.get("errors")
                    else "0",
                    "slow queries": snap.get("slow_queries"),
                },
                theme=theme,
            ),
        ]
        if wal and wal.get("lag_pages"):
            lines.append(
                paint(
                    f"⚠ WAL lag: {wal['lag_pages']} pages uncheckpointed "
                    f"({wal['log_pages']} log / {wal['checkpointed_pages']} done) — "
                    "a pinned reader may be blocking the checkpointer",
                    "warn",
                    theme=theme,
                )
            )
        return "\n".join(lines)


def _fmt_bytes(value: Any) -> str:
    try:
        num = float(value or 0)
    except (TypeError, ValueError):
        return str(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024 or unit == "TB":
            return f"{num:.1f}{unit}" if unit != "B" else f"{int(num)}B"
        num /= 1024
    return f"{num:.1f}TB"  # pragma: no cover


class AsyncDatabase:
    """Async offload wrapper around :class:`Database`.

    The stdlib ``sqlite3`` API is blocking; calling it inside ``async def``
    stalls the event loop for the whole query. This wrapper routes every call
    through a dedicated single-thread executor (the loom/bateau84 skill's
    prescription, pinned to one thread so even ``:memory:`` databases keep a
    single thread-local connection) so async handlers stay responsive. The
    underlying :class:`Database` is thread-safe by design (thread-local
    connections, single-writer lock), so this is safe — not a workaround.

        adb = AsyncDatabase(db)
        rows = await adb.query("SELECT * FROM memories LIMIT 10")
        async with adb.transaction():
            await adb.execute("INSERT INTO t (a) VALUES (?)", (1,))
    """

    def __init__(self, db: Database) -> None:
        import concurrent.futures

        self._db = db
        #: One thread for all DB work: calls serialize here (SQLite is
        #: single-writer anyway) and share one thread-local connection —
        #: this is what makes ``:memory:`` databases work under async.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="nm-asyncdb",
        )

    @property
    def db(self) -> Database:
        return self._db

    def close(self) -> None:
        """Shut down the backing executor (the Database itself is untouched)."""
        self._executor.shutdown(wait=True)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        attr = getattr(self._db, name)
        if not callable(attr) or name in {"transaction"}:
            return attr

        import asyncio
        import functools

        @functools.wraps(attr)
        async def _offloaded(*args: Any, **kwargs: Any) -> Any:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor, functools.partial(attr, *args, **kwargs)
            )

        return _offloaded

    def transaction(self, **kwargs: Any) -> "_AsyncTransaction":
        return _AsyncTransaction(self, **kwargs)


class _AsyncTransaction:
    """``async with adb.transaction():`` — offloads the sync context manager."""

    def __init__(self, adb: AsyncDatabase, **kwargs: Any) -> None:
        self._adb = adb
        self._kwargs = kwargs
        self._cm: Any = None

    async def __aenter__(self) -> AsyncDatabase:
        import asyncio

        adb, kwargs = self._adb, self._kwargs
        loop = asyncio.get_running_loop()

        def _enter() -> Any:
            self._cm = adb.db.transaction(**kwargs)
            return self._cm.__enter__()

        await loop.run_in_executor(adb._executor, _enter)
        return adb

    async def __aexit__(self, *exc: Any) -> Any:
        import asyncio

        cm, self._cm = self._cm, None
        if cm is None:
            return False
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._adb._executor, cm.__exit__, *exc)


class MigrationSummary:
    """Result of :meth:`Database.migrate`."""

    __slots__ = ("applied", "version")

    def __init__(self, applied: list[str], version: int) -> None:
        self.applied = applied
        self.version = version

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"MigrationSummary(applied={self.applied!r}, version={self.version})"


def _is_retryable_sqlite(exc: sqlite3.Error) -> bool:
    text = str(exc).lower()
    return "locked" in text or "busy" in text


#: Shared-cache in-memory databases: one URI per Database instance so every
#: connection of that instance (reader threads + the writer actor) sees the
#: SAME database. Plain ":memory:" is per-connection, which made a
#: multi-thread :memory: use silently read a different empty database.
_MEMORY_DB_COUNTER = itertools.count(1)

#: First keywords that make a statement a write (routed to the writer actor).
_WRITE_FIRST = frozenset({
    "INSERT", "UPDATE", "DELETE", "REPLACE", "CREATE", "DROP", "ALTER",
    "VACUUM", "ANALYZE", "REINDEX", "ATTACH", "DETACH",
})

#: Transaction-control keywords. Issued via execute() they open/close a
#: transaction on the writer connection owned by the calling thread (the
#: migration runner and its tests use manual BEGIN/ROLLBACK); transaction()
#: itself uses the same jobs internally.
_TXN_CONTROL = frozenset({"BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE", "END"})


def _sql_first_word(sql: str) -> str:
    """First SQL keyword of *sql*, ignoring leading whitespace/comments."""

    text = sql.lstrip()
    while text.startswith("--"):
        newline = text.find("\n")
        if newline < 0:
            return ""
        text = text[newline + 1:].lstrip()
    while text.startswith("/*"):
        end = text.find("*/", 2)
        if end < 0:
            return ""
        text = text[end + 2:].lstrip()
    match = re.match(r"[A-Za-z]+", text)
    return match.group(0).upper() if match else ""


def _is_write_sql(sql: str) -> bool:
    """True when *sql* writes (so it belongs on the writer actor)."""

    word = _sql_first_word(sql)
    if not word or word == "EXPLAIN":
        return False
    if word == "WITH":
        return bool(re.search(r"\b(?:INSERT|UPDATE|DELETE|REPLACE)\b", sql, re.IGNORECASE))
    if word in _WRITE_FIRST:
        return True
    if word == "PRAGMA":
        # wal_checkpoint is maintenance (safe + meaningful from any
        # connection; the maintenance tick must not queue behind writes).
        if re.match(r"(?i)pragma\s+wal_checkpoint\b", sql.lstrip()):
            return False
        # The set form (PRAGMA x=y) writes; the read form does not.
        return "=" in sql
    return False


def _is_txn_control_sql(sql: str) -> bool:
    """True when *sql* opens/closes a transaction (BEGIN/COMMIT/...)."""

    return _sql_first_word(sql) in _TXN_CONTROL


class _RoutedCursor:
    """Cursor-shaped view over rows materialized on the writer thread.

    A live sqlite3.Cursor cannot cross threads, so a routed execute()
    fetches on the writer actor and hands the caller this shim: the same
    fetchone/fetchall/fetchmany/close/lastrowid/rowcount/description
    surface the direct path returns. Iterating it iterates the materialized
    rows. It cannot be re-executed (call db.execute() again instead).
    """

    __slots__ = ("_rows", "_pos", "lastrowid", "rowcount", "description")

    def __init__(self, rows: Sequence[Any], lastrowid: Any, rowcount: int,
                 description: Any) -> None:
        self._rows = list(rows)
        self._pos = 0
        self.lastrowid = lastrowid
        self.rowcount = rowcount
        self.description = description

    def fetchall(self) -> list[Any]:
        rest = self._rows[self._pos:]
        self._pos = len(self._rows)
        return rest

    def fetchone(self) -> Any:
        if self._pos >= len(self._rows):
            return None
        row = self._rows[self._pos]
        self._pos += 1
        return row

    def fetchmany(self, size: int | None = None) -> list[Any]:
        if size is None:
            size = 1
        end = min(len(self._rows), self._pos + size)
        chunk = self._rows[self._pos:end]
        self._pos = end
        return chunk

    def __iter__(self) -> Iterator[Any]:
        return iter(self._rows[self._pos:])

    def close(self) -> None:
        return None

    def execute(self, *args: Any, **kwargs: Any) -> None:
        raise StorageError("a routed cursor cannot be re-executed; call db.execute() again")


class _WriteJob:
    """One unit of work for the writer actor: a SQL statement (or a
    transaction boundary) handed over by a caller thread, applied on the
    writer connection, with the outcome delivered back to the waiter."""

    __slots__ = ("kind", "sql", "params", "owner", "owner_name", "event",
                 "result", "error", "cancelled", "enqueued_at", "generation")

    def __init__(self, kind: str, sql: str, params: Any, owner: int,
                 owner_name: str, generation: int) -> None:
        self.kind = kind
        self.sql = sql
        self.params = params
        self.owner = owner
        self.owner_name = owner_name
        self.event = threading.Event()
        self.result: Any = None
        self.error: BaseException | None = None
        self.cancelled = False
        self.enqueued_at = time.monotonic()
        self.generation = generation


class _WriterActor:
    """The single-writer actor: one dedicated thread owns the one SQLite
    write connection and applies queued writes serially.

    Durability contract (stated here, not implied): the queue is a HANDOFF,
    not storage. A write is acknowledged to its caller only after the
    writer applied it (and, for a transaction, after COMMIT returned) — so
    a write the caller was told succeeded is durable to exactly the degree
    SQLite's configured ``synchronous`` mode makes it (``NORMAL`` by
    default). A write still sitting in the queue when the process dies was
    never acknowledged and is lost; that is the loss window, and it is
    bounded by the caller's wait (a waiter that times out is told the write
    failed). No acknowledged write is ever lost to the queue.

    Failure contract: the actor loop catches per-job errors and keeps
    going; a loop-level exception marks the actor dead, fails every pending
    waiter loudly, and exits. Enqueueing on a dead actor raises
    StorageError naming the dead writer. Recovery is explicit —
    :meth:`Database.restart_writer_actor` — never a silent restart.
    """

    def __init__(self, db: Database, generation: int) -> None:
        self._db = db
        self._queue: queue.Queue[_WriteJob] = queue.Queue()
        self._pending: collections.deque[_WriteJob] = collections.deque()
        self._thread = threading.Thread(target=self._run, name="nm-db-writer", daemon=True)
        self._state_lock = threading.Lock()
        self._txn_owner: int | None = None
        self._txn_owner_name: str = ""
        self._txn_since: float = 0.0
        self._current: tuple[_WriteJob, float] | None = None
        self._generation = generation
        self._dead = False
        self._dead_error: str | None = None
        self._started = threading.Event()

    # -- lifecycle ------------------------------------------------------
    def start(self) -> None:
        self._thread.start()
        if not self._started.wait(10.0):
            raise StorageError("database writer actor did not start")

    def is_alive(self) -> bool:
        return self._thread.is_alive() and not self._dead

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def thread_ident(self) -> int | None:
        return self._thread.ident

    @property
    def dead_error(self) -> str | None:
        return self._dead_error

    def enqueue(self, job: _WriteJob) -> None:
        if self._dead:
            raise StorageError(
                "database writer actor is dead"
                + (f" ({self._dead_error})" if self._dead_error else "")
                + " — call restart_writer_actor()")
        if job.generation != self._generation:
            raise StorageError("writer actor was restarted; the transaction was lost")
        self._queue.put(job)

    def stop(self) -> None:
        """Ask the actor to exit (drain-fail pending waiters) and join it."""

        with suppress(Exception):  # close must never raise
            self._queue.put(_WriteJob("shutdown", "", (), 0, "", self._generation))
        self._thread.join(5.0)

    # -- state readers (for lock_state / interrupt / routing) -----------
    @property
    def txn_owner(self) -> int | None:
        with self._state_lock:
            return self._txn_owner

    def current_work(self) -> dict[str, Any] | None:
        """Who/what the writer is doing now (holder naming)."""

        now = time.monotonic()
        with self._state_lock:
            if self._txn_owner is not None:
                return {
                    "thread": self._txn_owner_name,
                    "ident": self._txn_owner,
                    "held_s": max(0.0, now - self._txn_since),
                    "kind": "transaction",
                    "sql": self._current[0].sql[:120] if self._current else "",
                }
            if self._current is not None:
                job, started = self._current
                return {
                    "thread": job.owner_name,
                    "ident": job.owner,
                    "held_s": max(0.0, now - started),
                    "kind": job.kind,
                    "sql": job.sql[:120],
                }
        return None

    def current_job_owner(self) -> int | None:
        """Owner of the job the writer is applying RIGHT NOW, else None.

        Distinct from current_work(): a transaction can be open while the
        writer is idle between statements. Only a job in flight has a
        SQLite step to abort — interrupting an idle writer connection
        would leave a stale flag that aborts the NEXT innocent writer's
        statement.
        """

        with self._state_lock:
            if self._current is None:
                return None
            return self._current[0].owner

    def cancel_pending(self, owner: int) -> int:
        """Drop + fail every queued job of *owner* (interrupt path)."""

        dropped = 0
        with self._state_lock:
            kept: collections.deque[_WriteJob] = collections.deque()
            while self._pending:
                job = self._pending.popleft()
                if job.owner == owner:
                    job.error = sqlite3.OperationalError("interrupted")
                    job.event.set()
                    dropped += 1
                else:
                    kept.append(job)
            self._pending = kept
        return dropped

    # -- the loop -------------------------------------------------------
    def _run(self) -> None:
        try:
            self._db._writer_connection()
            self._started.set()
            while True:
                job = self._next_job()
                if job is None:  # shutdown
                    break
                if job.cancelled:
                    continue
                self._apply(job)
        except Exception as exc:  # noqa: BLE001 - a dead writer must be loud
            self._dead = True
            self._dead_error = str(exc)
            self._started.set()
            self._fail_all(StorageError(f"database writer actor died: {exc}"))
            _log.error("writer actor died: %s", exc, exc_info=True)

    def _next_job(self) -> _WriteJob | None:
        """Next job to apply, preserving FIFO and transaction ownership.

        While a transaction is open only its owner's jobs apply; everyone
        else's waits in the pending deque and drains first when the
        transaction closes. Shutdown is checked before the owner check.
        """

        while True:
            with self._state_lock:
                if self._pending and self._txn_owner is None:
                    job = self._pending.popleft()
                    return job
            job = self._queue.get()
            if job.kind == "shutdown":
                self._fail_all(StorageError("database is closed"))
                return None
            if job.cancelled:
                continue
            with self._state_lock:
                if self._txn_owner is not None and job.owner != self._txn_owner:
                    self._pending.append(job)
                    continue
                if job.kind == "script" and self._txn_owner is not None:
                    self._pending.append(job)
                    continue
            return job

    def _fail_all(self, error: BaseException) -> None:
        with self._state_lock:
            pending, self._pending = self._pending, collections.deque()
        for job in pending:
            job.error = error
            job.event.set()
        while True:
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                break
            job.error = error
            job.event.set()

    def _apply(self, job: _WriteJob) -> None:
        db = self._db
        # Resolve the writer connection per job (not cached from startup):
        # _connection() is the seam the retry tests patch, and it returns
        # the actor thread's cached connection normally — so this is the
        # same object every time unless the seam was deliberately replaced.
        try:
            conn = db._writer_connection()
        except Exception as exc:  # noqa: BLE001 - delivered to the waiter
            job.error = exc
            job.event.set()
            return
        with self._state_lock:
            self._current = (job, time.monotonic())
        try:
            if job.kind == "begin":
                if conn.in_transaction:
                    raise StorageError("cannot start a transaction within a transaction")
                db._with_lock_retry(lambda: conn.execute(job.sql))
            elif job.kind in ("commit", "rollback"):
                if not conn.in_transaction:
                    raise StorageError(f"cannot {job.kind} - no transaction is active")
                # _safe is sabotage-aware and runs on the actor thread, so
                # _connection() IS the writer connection here.
                db._safe(job.sql)
            elif job.kind == "script":
                conn.executescript(job.sql)
            else:  # stmt / executemany
                started = time.perf_counter()
                in_txn = conn.in_transaction
                if job.kind == "executemany":
                    if in_txn:
                        cur = conn.executemany(job.sql, job.params)
                    else:
                        cur = db._with_lock_retry(lambda: conn.executemany(job.sql, job.params))
                elif in_txn:
                    cur = conn.execute(job.sql, job.params)
                else:
                    cur = db._with_lock_retry(lambda: conn.execute(job.sql, job.params))
                rows = cur.fetchall() if cur.description is not None else []
                job.result = _RoutedCursor(rows, cur.lastrowid, cur.rowcount, cur.description)
                elapsed = time.perf_counter() - started
                db._bump_stat("queries")
                db._bump_stat("query_seconds", elapsed)
                if job.sql.lstrip()[:6].upper() in {"INSERT", "UPDATE", "DELETE", "REPLAC"}:
                    db._bump_stat("writes")
                db._note_slow(job.sql, elapsed)
        except Exception as exc:  # noqa: BLE001 - delivered to the waiter
            db._bump_stat("errors")
            if (isinstance(exc, sqlite3.OperationalError)
                    and "interrupted" in str(exc).lower()
                    and conn.in_transaction):
                # An interrupted statement inside a transaction must not
                # leave the transaction dangling (the caller may be
                # abandoned) — roll it back now so the queue never wedges.
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error as exc:
                    _log.debug("poison rollback failed: %s", exc)
            job.error = exc
        finally:
            with self._state_lock:
                self._current = None
            self._sync_txn_state(job)
            job.event.set()

    def _sync_txn_state(self, job: _WriteJob) -> None:
        """Trust SQLite's state, never the actor's assumption.

        After every job, adopt the owner when a transaction opened (covers
        a manual BEGIN issued via execute()) and clear it when the
        transaction closed (covers COMMIT/ROLLBACK as statements and
        implicit commits). The holder record follows the same sync.
        """

        db = self._db
        conn = db._writer_conn
        if conn is None:
            return
        try:
            in_txn = bool(conn.in_transaction)
        except sqlite3.Error:
            in_txn = False
        with self._state_lock:
            if in_txn and self._txn_owner is None:
                self._txn_owner = job.owner
                self._txn_owner_name = job.owner_name
                self._txn_since = time.monotonic()
                with db._all_lock:
                    db._write_lock_holder = {
                        "ident": job.owner,
                        "name": job.owner_name,
                        "since": self._txn_since,
                    }
            elif not in_txn and self._txn_owner is not None:
                self._txn_owner = None
                self._txn_owner_name = ""
                with db._all_lock:
                    db._write_lock_holder = None


def split_sql(script: str) -> list[str]:
    """Split a SQL script into individual statements.

    Character-scanning rather than regex or line-based: a ``;`` inside a string
    literal is not a terminator, ``--`` and ``/* */`` comments are stripped, and
    several statements on one line are separated correctly.

    Known limitation: ``CREATE TRIGGER``/``CREATE VIEW ... BEGIN ... END`` bodies
    contain bare semicolons and are not split correctly. The schema does not use
    them; add :func:`sqlite3.complete_statement` bracketing here if that changes.
    """
    statements: list[str] = []
    buffer: list[str] = []
    i, length = 0, len(script)
    in_single = in_double = False

    while i < length:
        ch = script[i]

        if in_single:
            buffer.append(ch)
            if ch == "'":
                if i + 1 < length and script[i + 1] == "'":  # escaped quote
                    buffer.append("'")
                    i += 2
                    continue
                in_single = False
            i += 1
            continue

        if in_double:
            buffer.append(ch)
            if ch == '"':
                in_double = False
            i += 1
            continue

        if ch == "-" and script.startswith("--", i):
            newline = script.find("\n", i)
            i = length if newline < 0 else newline
            continue

        if ch == "/" and script.startswith("/*", i):
            end = script.find("*/", i + 2)
            i = length if end < 0 else end + 2
            continue

        if ch == "'":
            in_single = True
            buffer.append(ch)
            i += 1
            continue

        if ch == '"':
            in_double = True
            buffer.append(ch)
            i += 1
            continue

        if ch == ";":
            statement = "".join(buffer).strip()
            if statement:
                statements.append(statement)
            buffer = []
            i += 1
            continue

        buffer.append(ch)
        i += 1

    tail = "".join(buffer).strip()
    if tail:
        statements.append(tail)
    return statements


@contextmanager
def transaction(db: Database, *, immediate: bool = True) -> Iterator[Database]:
    """Functional form of :meth:`Database.transaction`."""
    with db.transaction(immediate=immediate) as handle:
        yield handle


def require_row(row: dict[str, Any] | None, what: str = "row") -> dict[str, Any]:
    """Unwrap a ``query_one`` result or raise :class:`NotFound`."""
    if row is None:
        raise NotFound(f"{what} not found")
    return row


def release_thread_connection(db: "Database | None") -> None:
    """Best-effort per-thread DB connection release for one-shot workers.

    No-op when the thread never touched the DB or ``db`` is None — safe to
    call unconditionally in a thread's ``finally`` block. One line per
    spawn site; see :meth:`Database.release_thread` for why.
    """
    try:
        if db is not None:
            db.release_thread()
    except Exception as e:  # never break thread teardown
        record_failure('db.release_thread_connection', e)


_CORRUPT_MARKERS = (
    "file is not a database",
    "database disk image is malformed",
    "file is corrupted",
    "is encrypted or is not a database",
    "not a database",
)


def _looks_corrupt(exc: BaseException) -> bool:
    """True when ``exc`` smells like a corrupt/unreadable SQLite file rather
    than a transient failure (locked/busy) or a schema problem."""
    text = str(exc).lower()
    return any(marker in text for marker in _CORRUPT_MARKERS)


def open_database(path: str | os.PathLike[str], **kwargs: Any) -> tuple["Database", bool, str | None]:
    """Open ``path`` and run migrations, quarantining a corrupt file instead
    of dying.

    A phone bot that refuses to boot because its SQLite file got clobbered
    (killed mid-checkpoint, dying flash) is a brick until the owner SSHes in
    and deletes it by hand — and deleting it loses everything. So: on a
    *corruption* signature the file (plus WAL/journal sidecars) is moved
    aside to ``<name>.corrupt-<timestamp>`` and a fresh database is created.
    The owner's data survives in the quarantine for later forensics.

    Returns ``(database, recovered, backup_path)`` — ``recovered`` is True
    only when a corrupt file was quarantined. Non-corruption errors raise
    unchanged.
    """
    from pathlib import Path as _Path

    target = _Path(path)
    database: Database | None = None
    corrupt_error: Exception | None = None
    try:
        database = Database(target, **kwargs)
        database.migrate()
        return database, False, None
    except Exception as exc:  # noqa: BLE001 - inspect before deciding
        if target.name == ":memory:" or not _looks_corrupt(exc):
            raise
        corrupt_error = exc  # the except-block name is deleted after the block
    # Corrupt: quarantine the file and its sidecars, then start fresh.
    # NOTE: the quarantine runs BEFORE database.close() — closing the last
    # connection to a WAL database makes SQLite delete the -wal/-shm
    # sidecars, which would destroy the very files we're preserving.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = target.with_name(f"{target.name}.corrupt-{stamp}")
    moved: list[str] = []
    for candidate in (target, target.with_name(target.name + "-wal"),
                      target.with_name(target.name + "-shm"),
                      target.with_name(target.name + "-journal")):
        if candidate.exists():
            try:
                dest = backup.parent / (candidate.name + ".corrupt-" + stamp)
                candidate.replace(dest)
                moved.append(str(dest))
            except OSError as move_exc:
                _log.warning("could not quarantine %s: %s", candidate, move_exc)
    if database is not None:
        try:
            database.close()
        except Exception as e:  # best-effort
            record_failure('db.open_database', e)
    _log.error(
        "database file %s is corrupt (%s); quarantined %d file(s), starting "
        "fresh. The quarantined copy keeps the old data for forensics.",
        target, corrupt_error, len(moved),
    )
    fresh = Database(target, **kwargs)
    fresh.migrate()
    return fresh, True, str(backup)
