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
"""

from __future__ import annotations

import os
import random
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from ..core.errors import ConstraintViolation, NotFound, StorageError
from ..core.logging_setup import get_logger
from ..core.style import active_theme, header, kv_lines, paint, styled_table

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
        self.path = Path(path) if path != ":memory:" else None
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
        self._all_connections: list[sqlite3.Connection] = []
        self._all_lock = threading.Lock()
        #: Guards ``stats``: execute()/transaction() run on every worker
        #: thread and ``+=`` on a dict value is a read-modify-write.
        self._stats_lock = threading.Lock()
        self._closed = False
        self._depth = threading.local()

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
        target = ":memory:" if self.path is None else str(self.path)
        if self.readonly and self.path is not None:
            uri = f"file:{target}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=self._timeout)
        else:
            # Ensure the parent directory exists (fresh installs).
            if self.path is not None:
                Path(target).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                target,
                timeout=self._timeout,
                isolation_level=None,  # autocommit; we drive transactions ourselves
                check_same_thread=False,
            )
        conn.row_factory = sqlite3.Row
        self._configure(conn)
        self._local.conn = conn
        with self._all_lock:
            self._all_connections.append(conn)
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
        with self._all_lock:
            connections, self._all_connections = self._all_connections, []
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
        try:
            conn.close()
        except sqlite3.Error:  # pragma: no cover - close is best-effort here
            _log.debug("thread connection close failed", exc_info=True)

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

    def execute(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> sqlite3.Cursor:
        """Run a single statement. Returns the cursor (use ``.lastrowid``/``.rowcount``)."""
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

    def executemany(self, sql: str, seq: Iterable[Sequence[Any]]) -> sqlite3.Cursor:
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
        fast rather than deadlocking after doing half their work.
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

        self._write_lock.acquire()
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
            self._write_lock.release()

    def _safe(self, sql: str) -> None:
        # A COMMIT that hits SQLITE_BUSY leaves the transaction open, so a
        # blind retry is the correct action (not a fresh BEGIN).
        try:
            self._with_lock_retry(lambda: self._connection().execute(sql))
        except sqlite3.Error as exc:  # pragma: no cover - rollback of a broken txn
            _log.warning("%s failed: %s", sql, exc)

    @property
    def in_transaction(self) -> bool:
        return getattr(self._depth, "value", 0) > 0

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
                except OSError:  # pragma: no cover - best effort
                    pass
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
        except sqlite3.Error:  # pragma: no cover - introspection only
            pass
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
        except sqlite3.Error:  # pragma: no cover - introspection only
            pass
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
    except Exception:  # noqa: BLE001 - never break thread teardown
        pass


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
        except Exception:  # noqa: BLE001 - best-effort
            pass
    _log.error(
        "database file %s is corrupt (%s); quarantined %d file(s), starting "
        "fresh. The quarantined copy keeps the old data for forensics.",
        target, corrupt_error, len(moved),
    )
    fresh = Database(target, **kwargs)
    fresh.migrate()
    return fresh, True, str(backup)
