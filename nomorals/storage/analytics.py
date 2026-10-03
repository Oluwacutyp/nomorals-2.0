"""Analytical query backends for the storage layer.

The operational store is SQLite: right for OLTP, thread-local connections, WAL,
single-file backup. It is the wrong engine for *analytical* questions — "memory
growth per week", "task completion rate by month", "p95 tool-call latency by
tool" — where a columnar vectorized engine scans millions of rows without
breaking a sweat.

This module adds a backend abstraction *for analytics only*:

* :class:`AnalyticsBackend` — the protocol (``load_table`` / ``query`` /
  ``table_names`` / ``clear``).
* :class:`DuckDBAnalytics` — DuckDB, in-process, columnar. Optional dependency
  (``pip install duckdb``); the best free analytical engine available, MIT
  licensed, zero-config, no server. Tables are synced from the SQLite
  :class:`~nomorals.storage.db.Database` on demand — the SQLite file stays the
  system of record.
* :class:`SQLiteAnalytics` — passthrough to the live database. No new
  dependency, always available, correct for small data; slower on big scans.

Use :func:`open_analytics` to get the best available backend automatically.
Nothing here changes existing behavior: when DuckDB is not installed, every
call path degrades to SQLite and keeps working.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from ..compat import load_optional
from ..core.errors import StorageError
from ..core.logging_setup import get_logger
from .db import Database

__all__ = [
    "AnalyticsBackend",
    "DuckDBAnalytics",
    "SQLiteAnalytics",
    "open_analytics",
    "table_row_counts",
]

_log = get_logger(__name__)

_DUCKDB_INSTALL_HINT = (
    "DuckDB is not installed. Install it with `pip install duckdb` "
    "(free, MIT licensed, no server) or use prefer='sqlite'."
)

#: How many rows to move per batch when syncing a SQLite table into DuckDB.
_SYNC_BATCH = 50_000


class AnalyticsBackend(Protocol):
    """Analytical SQL over storage data. Sync-then-query; SQLite stays primary."""

    name: str

    def load_table(
        self,
        db: Database,
        table: str,
        *,
        where: str = "",
        params: Sequence[Any] = (),
    ) -> int:
        """Sync ``table`` (optionally filtered) from ``db``; return rows loaded."""
        ...

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        """Run read-only analytical SQL; return rows as dicts."""
        ...

    def table_names(self) -> list[str]:
        """Names of tables currently loaded in the analytical engine."""
        ...

    def clear(self) -> None:
        """Drop every loaded table."""
        ...


def _duckdb_type(declared: str) -> str:
    """Map a SQLite declared column type to a DuckDB type."""
    upper = (declared or "").upper()
    for token, mapped in (
        ("BIGINT", "BIGINT"),
        ("INT", "BIGINT"),
        ("DOUBLE", "DOUBLE"),
        ("FLOAT", "DOUBLE"),
        ("REAL", "DOUBLE"),
        ("NUMERIC", "DOUBLE"),
        ("DECIMAL", "DOUBLE"),
        ("BOOLEAN", "BOOLEAN"),
        ("BOOL", "BOOLEAN"),
        ("BLOB", "BLOB"),
        ("DATE", "DATE"),
        ("TIMESTAMP", "TIMESTAMP"),
        ("DATETIME", "TIMESTAMP"),
    ):
        if token in upper:
            return mapped
    return "VARCHAR"


class DuckDBAnalytics:
    """Columnar analytical queries over synced SQLite tables (DuckDB, MIT).

    DuckDB runs in-process — no server, no config — and executes analytical
    SQL (GROUP BY / window functions / approximate quantiles) vectorized,
    typically 10-100x faster than SQLite on full-table scans. The SQLite
    database remains the system of record; call :meth:`load_table` (or
    :meth:`refresh`) to re-sync before running reports.
    """

    name = "duckdb"

    def __init__(self) -> None:
        duckdb = load_optional("duckdb")
        if duckdb is None:
            raise StorageError(_DUCKDB_INSTALL_HINT)
        self._duckdb = duckdb
        self._conn = duckdb.connect(":memory:")
        self._lock = threading.RLock()
        self._loaded: dict[str, int] = {}
        self.stats = {"tables_loaded": 0, "rows_loaded": 0, "queries": 0}

    # ── sync ───────────────────────────────────────────────────────────────
    def load_table(
        self,
        db: Database,
        table: str,
        *,
        where: str = "",
        params: Sequence[Any] = (),
    ) -> int:
        """Copy ``table`` from the SQLite ``db`` into DuckDB. Returns row count."""
        columns = db.table_info(table)
        if not columns:
            raise StorageError(f"table {table!r} does not exist in the source database")
        names = [c["name"] for c in columns]
        types = [_duckdb_type(str(c["type"])) for c in columns]
        ddl = ", ".join(f'"{n}" {t}' for n, t in zip(names, types, strict=True))

        quoted_cols = ", ".join(f'"{n}"' for n in names)
        sql = f'SELECT {quoted_cols} FROM "{table}"'
        if where:
            sql += f" WHERE {where}"

        placeholders = ", ".join("?" for _ in names)
        insert_sql = f'INSERT INTO "{table}" VALUES ({placeholders})'

        total = 0
        with self._lock:
            self._conn.execute(f'DROP TABLE IF EXISTS "{table}"')
            self._conn.execute(f'CREATE TABLE "{table}" ({ddl})')
            cursor = db.execute(sql, params)
            try:
                while True:
                    batch = cursor.fetchmany(_SYNC_BATCH)
                    if not batch:
                        break
                    rows = [tuple(row) for row in batch]
                    self._conn.executemany(insert_sql, rows)
                    total += len(rows)
            finally:
                cursor.close()
            self._loaded[table] = total
        self.stats["tables_loaded"] += 1
        self.stats["rows_loaded"] += total
        _log.debug("analytics: synced %s (%d rows) into DuckDB", table, total)
        return total

    def refresh(self, db: Database, table: str) -> int:
        """Re-sync ``table`` (drop + reload)."""
        return self.load_table(db, table)

    def sync_tables(self, db: Database, tables: Sequence[str]) -> dict[str, int]:
        """Sync several tables; returns ``{table: rows}``."""
        return {table: self.load_table(db, table) for table in tables}

    # ── query ──────────────────────────────────────────────────────────────
    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        lowered = sql.lstrip().upper()
        if not lowered.startswith("SELECT") and not lowered.startswith("WITH"):
            raise StorageError("analytics backend is read-only: only SELECT/WITH allowed")
        with self._lock:
            # NOTE: duckdb's Connection.execute() returns the connection
            # itself as a cursor-like object — do NOT close it, or the whole
            # in-memory database goes away.
            cursor = self._conn.execute(sql, list(params))
            names = [d[0] for d in cursor.description or []]
            rows = cursor.fetchall()
        self.stats["queries"] += 1
        return [dict(zip(names, row, strict=True)) for row in rows]

    def table_names(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' ORDER BY table_name"
            ).fetchall()
        return [r[0] for r in rows]

    def clear(self) -> None:
        with self._lock:
            for table in self.table_names():
                self._conn.execute(f'DROP TABLE IF EXISTS "{table}"')
            self._loaded.clear()

    # ── parquet interchange ────────────────────────────────────────────────
    def to_parquet(self, table: str, path: str | Path) -> Path:
        """Export a loaded table to Parquet (DuckDB-native columnar file)."""
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._conn.execute(
                f"COPY (SELECT * FROM \"{table}\") TO '{target}' (FORMAT PARQUET)"
            )
        return target

    def from_parquet(self, path: str | Path, table: str) -> int:
        """Load a Parquet file as an analytical table."""
        source = Path(path).expanduser()
        if not source.is_file():
            raise StorageError(f"parquet file not found: {source}")
        with self._lock:
            self._conn.execute(
                f'CREATE OR REPLACE TABLE "{table}" AS '
                f"SELECT * FROM read_parquet('{source}')"
            )
            count = self._conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            self._loaded[table] = int(count)
        return int(count)

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception as exc:  # noqa: BLE001 - teardown must not raise
                _log.debug("duckdb close failed: %s", exc)

    def stats_snapshot(self) -> dict[str, Any]:
        return {**self.stats, "tables": self.table_names()}


class SQLiteAnalytics:
    """Analytical queries run directly against the SQLite database.

    Zero new dependencies; correct for small-to-medium data. This is the
    automatic fallback when DuckDB is not installed, so analytical code paths
    keep working everywhere.
    """

    name = "sqlite"

    def __init__(self, db: Database) -> None:
        self.db = db
        self.stats = {"queries": 0}

    def load_table(
        self,
        db: Database,
        table: str,
        *,
        where: str = "",
        params: Sequence[Any] = (),
    ) -> int:
        # Nothing to sync: the data already lives here. Validate the table
        # exists and return its (filtered) row count so callers get the same
        # contract as the DuckDB backend.
        sql = f'SELECT COUNT(*) FROM "{table}"'
        if where:
            sql += f" WHERE {where}"
        try:
            return int(db.scalar(sql, params, default=0))
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(f"cannot load table {table!r}: {exc}") from exc

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        lowered = sql.lstrip().upper()
        if not lowered.startswith("SELECT") and not lowered.startswith("WITH"):
            raise StorageError("analytics backend is read-only: only SELECT/WITH allowed")
        rows = self.db.query(sql, params)
        self.stats["queries"] += 1
        return rows

    def table_names(self) -> list[str]:
        return self.db.tables()

    def clear(self) -> None:
        # No synced copies exist; nothing to drop.
        return None


def open_analytics(db: Database, *, prefer: str = "auto") -> AnalyticsBackend:
    """Return the best available analytical backend.

    ``prefer="auto"`` (default) returns DuckDB when installed, otherwise the
    SQLite passthrough. ``prefer="duckdb"`` raises :class:`StorageError` with
    an install hint when DuckDB is missing; ``prefer="sqlite"`` always works.
    """
    if prefer == "duckdb":
        return DuckDBAnalytics()
    if prefer == "sqlite":
        return SQLiteAnalytics(db)
    if prefer != "auto":
        raise StorageError(f"unknown analytics backend preference: {prefer!r}")
    if load_optional("duckdb") is not None:
        try:
            return DuckDBAnalytics()
        except StorageError:
            _log.debug("duckdb present but unusable; falling back to sqlite analytics")
    return SQLiteAnalytics(db)


def table_row_counts(engine: AnalyticsBackend) -> dict[str, int]:
    """Row count per table through any analytical backend (schema-agnostic)."""
    counts: dict[str, int] = {}
    for table in engine.table_names():
        row = engine.query(f'SELECT COUNT(*) AS n FROM "{table}"')
        counts[table] = int(row[0]["n"]) if row else 0
    return counts
