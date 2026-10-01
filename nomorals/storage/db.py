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
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from ..core.errors import ConstraintViolation, NotFound, StorageError
from ..core.logging_setup import get_logger

__all__ = ["Database", "Row", "split_sql", "transaction"]

_log = get_logger(__name__)

Row = sqlite3.Row


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
    ) -> None:
        self.path = Path(path) if path != ":memory:" else None
        self.wal = wal and self.path is not None
        self.busy_timeout_ms = busy_timeout_ms
        self.synchronous = synchronous
        self.foreign_keys = foreign_keys
        self.cache_size_kb = cache_size_kb
        self.readonly = readonly
        self._timeout = timeout

        self._local = threading.local()
        self._write_lock = threading.RLock()
        self._all_connections: list[sqlite3.Connection] = []
        self._all_lock = threading.Lock()
        self._closed = False
        self._depth = threading.local()

        self.stats = {
            "queries": 0,
            "writes": 0,
            "transactions": 0,
            "retries": 0,
            "errors": 0,
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
            ]
        for name, value in pragmas:
            try:
                conn.execute(f"PRAGMA {name}={value}")
            except sqlite3.Error as exc:  # pragma: no cover - platform dependent
                _log.debug("PRAGMA %s=%s failed: %s", name, value, exc)

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

    # ── execution ────────────────────────────────────────────────────────────
    def execute(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> sqlite3.Cursor:
        """Run a single statement. Returns the cursor (use ``.lastrowid``/``.rowcount``)."""
        started = time.perf_counter()
        try:
            cursor = self._connection().execute(sql, params)
        except sqlite3.IntegrityError as exc:
            self.stats["errors"] += 1
            raise ConstraintViolation(str(exc)) from exc
        except sqlite3.OperationalError as exc:
            self.stats["errors"] += 1
            raise StorageError(str(exc), retryable=_is_retryable_sqlite(exc)) from exc
        except sqlite3.Error as exc:
            self.stats["errors"] += 1
            raise StorageError(str(exc)) from exc
        finally:
            self.stats["queries"] += 1
            self.stats["query_seconds"] += time.perf_counter() - started
        if sql.lstrip()[:6].upper() in {"INSERT", "UPDATE", "DELETE", "REPLAC"}:
            self.stats["writes"] += 1
        return cursor

    def executemany(self, sql: str, seq: Iterable[Sequence[Any]]) -> sqlite3.Cursor:
        started = time.perf_counter()
        try:
            cursor = self._connection().executemany(sql, list(seq))
        except sqlite3.IntegrityError as exc:
            self.stats["errors"] += 1
            raise ConstraintViolation(str(exc)) from exc
        except sqlite3.Error as exc:
            self.stats["errors"] += 1
            raise StorageError(str(exc)) from exc
        finally:
            self.stats["queries"] += 1
            self.stats["query_seconds"] += time.perf_counter() - started
        self.stats["writes"] += 1
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
            self.stats["errors"] += 1
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
            self.stats["transactions"] += 1
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
        try:
            self._connection().execute(sql)
        except sqlite3.Error as exc:  # pragma: no cover - rollback of a broken txn
            _log.warning("%s failed: %s", sql, exc)

    @property
    def in_transaction(self) -> bool:
        return getattr(self._depth, "value", 0) > 0

    # ── maintenance ──────────────────────────────────────────────────────────
    def checkpoint(self, mode: str = "TRUNCATE") -> tuple[int, int]:
        """Force a WAL checkpoint. Returns (busy, log_pages)."""
        if not self.wal:
            return (0, 0)
        row = self.query_one(f"PRAGMA wal_checkpoint({mode})")
        if row is None:
            return (0, 0)
        values = list(row.values())
        return (int(values[1] or 0), int(values[2] or 0))

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
        """Apply all pending migrations."""
        from .migrations import MIGRATIONS
        from .schema import MigrationRunner

        runner = MigrationRunner(self)
        applied = runner.apply_all(MIGRATIONS)
        return MigrationSummary(applied=applied, version=runner.current_version())

    def stats_snapshot(self) -> dict[str, Any]:
        info = {
            "path": str(self.path) if self.path else ":memory:",
            "connections": self.connection_count(),
            "tables": len(self.tables()),
        }
        if self.path is not None and self.path.exists():
            info["size_bytes"] = self.path.stat().st_size
        return {**self.stats, **info}


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
