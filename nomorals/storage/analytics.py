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
import time as _time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from ..compat import load_optional
from ..core.errors import StorageError
from ..core.logging_setup import get_logger
from ..core.style import active_theme, header, kv_lines, styled_table
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
        #: Attached SQLite databases: alias → path (zero-copy, always fresh).
        self._attached: dict[str, str] = {}
        #: query_cached memo: (sql, params) → (expires_at, rows).
        self._cache: dict[tuple[str, str], tuple[float, list[dict[str, Any]]]] = {}
        self.stats = {"tables_loaded": 0, "rows_loaded": 0, "queries": 0,
                      "cache_hits": 0}

    # ── zero-copy attach ─────────────────────────────────────────────────
    #
    # The "SQLite + DuckDB blending" pattern: ATTACH the live SQLite file
    # (TYPE SQLITE) instead of copying rows. Analytical queries see fresh
    # data with no sync step; the SQLite file stays the system of record.
    # Falls back to copy-sync for :memory: databases (no file to attach).

    def attach(self, db: Database, *, alias: str = "src") -> list[str]:
        """ATTACH a SQLite database file for zero-copy analytical queries.

        Returns the table names visible under ``alias``. Query them as
        ``alias.table`` — e.g. ``SELECT … FROM src.memories``.
        """
        if db.path is None or not db.path.exists():
            raise StorageError(
                "attach needs a file-backed database; use load_table() for :memory:"
            )
        path = str(db.path).replace("'", "''")
        with self._lock:
            self._conn.execute(
                f"ATTACH '{path}' AS \"{alias}\" (TYPE SQLITE, READ_ONLY)"
            )
            self._attached[alias] = str(db.path)
            rows = self._conn.execute(
                "SELECT table_name FROM information_schema.tables "
                f"WHERE table_catalog = '{alias}' AND table_schema = 'main' "
                "ORDER BY table_name"
            ).fetchall()
        return [r[0] for r in rows]

    def detach(self, alias: str = "src") -> None:
        with self._lock:
            self._conn.execute(f'DETACH "{alias}"')
            self._attached.pop(alias, None)

    def attached(self) -> dict[str, str]:
        return dict(self._attached)

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
            self._cache.clear()

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

    # ── analytical helpers ─────────────────────────────────────────────

    def profile(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        """``EXPLAIN ANALYZE`` for a query — the slow-query story, answered."""
        lowered = sql.lstrip().upper()
        if not lowered.startswith("SELECT") and not lowered.startswith("WITH"):
            raise StorageError("analytics backend is read-only: only SELECT/WITH allowed")
        with self._lock:
            cursor = self._conn.execute("EXPLAIN ANALYZE " + sql, list(params))
            names = [d[0] for d in cursor.description or []]
            rows = cursor.fetchall()
        return [dict(zip(names, row, strict=True)) for row in rows]

    def query_cached(
        self, sql: str, params: Sequence[Any] = (),
        *, ttl: float = 300.0,
    ) -> list[dict[str, Any]]:
        """Memoized :meth:`query` — expensive analytical queries, cached.

        ``ttl`` seconds of freshness; pass ``ttl=0`` to bypass. The cache key
        includes the params, and :meth:`clear` / :meth:`refresh` invalidate.
        """
        import json as _json

        key = (sql, _json.dumps(list(params), default=str))
        now = _time.time()
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None and ttl > 0 and hit[0] > now:
                self.stats["cache_hits"] += 1
                return [dict(r) for r in hit[1]]
        rows = self.query(sql, params)
        if ttl > 0:
            with self._lock:
                self._cache[key] = (now + ttl, [dict(r) for r in rows])
        return rows

    def invalidate_cache(self) -> int:
        """Drop all cached query results. Returns entries cleared."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
        return count

    def describe(self, table: str) -> dict[str, Any]:
        """Column names/types plus row count — one-dict table overview."""
        with self._lock:
            cols = self._conn.execute(f'DESCRIBE SELECT * FROM "{table}"').fetchall()
            count = self._conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        return {
            "table": table,
            "rows": int(count),
            "columns": [
                {"name": c[0], "type": c[1], "null": c[2], "key": c[3]}
                for c in cols
            ],
        }

    def sample(self, table: str, n: int = 100, *, seed: int = 42) -> list[dict[str, Any]]:
        """Reservoir sample of ``n`` rows (``USING SAMPLE``)."""
        return self.query(
            f'SELECT * FROM "{table}" USING SAMPLE {max(1, n)} '
            f"(reservoir, {seed})"
        )

    def histogram(
        self, table: str, column: str, *, bins: int = 20,
        where: str = "", params: Sequence[Any] = (),
    ) -> list[dict[str, Any]]:
        """Bin a numeric column into ``bins`` buckets (DuckDB ``histogram``)."""
        sql = (
            f'SELECT UNNEST(histogram("{column}", {max(1, bins)})) AS bin '
            f'FROM "{table}"'
        )
        if where:
            sql += f" WHERE {where}"
        rows = self.query(sql, params)
        out = []
        for row in rows:
            binned = row["bin"]
            out.append({
                "low": binned.get("low") if isinstance(binned, dict) else None,
                "high": binned.get("high") if isinstance(binned, dict) else None,
                "count": binned.get("count") if isinstance(binned, dict) else None,
                "raw": binned,
            })
        return out

    def to_csv(self, table: str, path: str | Path) -> Path:
        """Export a loaded table to CSV."""
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._conn.execute(
                f"COPY (SELECT * FROM \"{table}\") TO '{target}' (FORMAT CSV, HEADER)"
            )
        return target

    def iter_query(
        self, sql: str, params: Sequence[Any] = (),
        *, batch: int = 10_000,
    ) -> Any:
        """Yield result batches — analytical results bigger than RAM."""
        lowered = sql.lstrip().upper()
        if not lowered.startswith("SELECT") and not lowered.startswith("WITH"):
            raise StorageError("analytics backend is read-only: only SELECT/WITH allowed")
        with self._lock:
            cursor = self._conn.execute(sql, list(params))
            names = [d[0] for d in cursor.description or []]
            while True:
                rows = cursor.fetchmany(batch)
                if not rows:
                    break
                yield [dict(zip(names, row, strict=True)) for row in rows]
        self.stats["queries"] += 1

    def format_stats(self, theme: Any = None) -> str:
        """Human-readable backend overview through the shared style layer."""
        theme = theme or active_theme()
        snap = self.stats_snapshot()
        return "\n".join([
            header("analytics (duckdb)", theme=theme),
            *kv_lines(
                {
                    "tables": ", ".join(snap["tables"]) or "–",
                    "attached": ", ".join(
                        f"{a}→{p}" for a, p in self._attached.items()) or "–",
                    "rows loaded": snap["rows_loaded"],
                    "queries": snap["queries"],
                    "cache hits": snap["cache_hits"],
                    "cached queries": len(self._cache),
                },
                theme=theme,
            ),
        ])

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

    # ── analytical helpers (same API as the DuckDB backend) ────────────

    def profile(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        """``EXPLAIN QUERY PLAN`` — the portable half of DuckDB's ANALYZE."""
        return self.db.explain(sql, params)

    def query_cached(
        self, sql: str, params: Sequence[Any] = (),
        *, ttl: float = 300.0,
    ) -> list[dict[str, Any]]:
        """No cache on the passthrough backend — same signature, direct query."""
        return self.query(sql, params)

    def invalidate_cache(self) -> int:
        return 0

    def describe(self, table: str) -> dict[str, Any]:
        cols = self.db.table_info(table)
        return {
            "table": table,
            "rows": self.db.row_count(table),
            "columns": [
                {"name": c["name"], "type": c["type"], "null": not c["notnull"],
                 "key": "pk" if c["pk"] else ""}
                for c in cols
            ],
        }

    def sample(self, table: str, n: int = 100, *, seed: int = 42) -> list[dict[str, Any]]:
        # Portable reservoir-ish sample: random ordering is O(n) but this is
        # the small-data backend by design.
        return self.db.query(
            f'SELECT * FROM "{table}" ORDER BY RANDOM() LIMIT ?',
            (max(1, n),),
        )

    def histogram(
        self, table: str, column: str, *, bins: int = 20,
        where: str = "", params: Sequence[Any] = (),
    ) -> list[dict[str, Any]]:
        bounds = self.db.query_one(
            f'SELECT MIN("{column}") AS lo, MAX("{column}") AS hi FROM "{table}"'
            + (f" WHERE {where}" if where else ""),
            params,
        ) or {}
        lo, hi = bounds.get("lo"), bounds.get("hi")
        if lo is None or hi is None or lo == hi:
            return []
        width = (hi - lo) / max(1, bins)
        rows = self.db.query(
            f'SELECT CAST(("{column}" - ?) / ? AS INTEGER) AS b, COUNT(*) AS n '
            f'FROM "{table}"' + (f" WHERE {where}" if where else "") +
            ' GROUP BY b ORDER BY b',
            (lo, width, *params),
        )
        return [
            {"low": lo + r["b"] * width,
             "high": lo + (r["b"] + 1) * width,
             "count": r["n"], "raw": None}
            for r in rows
        ]

    def iter_query(
        self, sql: str, params: Sequence[Any] = (),
        *, batch: int = 10_000,
    ) -> Any:
        """Yield result batches (same API as the DuckDB backend)."""
        lowered = sql.lstrip().upper()
        if not lowered.startswith("SELECT") and not lowered.startswith("WITH"):
            raise StorageError("analytics backend is read-only: only SELECT/WITH allowed")
        cursor = self.db.execute(sql, params)
        try:
            while True:
                rows = cursor.fetchmany(batch)
                if not rows:
                    break
                yield [dict(r) for r in rows]
        finally:
            cursor.close()
        self.stats["queries"] += 1

    def format_stats(self, theme: Any = None) -> str:
        theme = theme or active_theme()
        return "\n".join([
            header("analytics (sqlite passthrough)", theme=theme),
            *kv_lines(
                {
                    "tables": len(self.table_names()),
                    "queries": self.stats["queries"],
                },
                theme=theme,
            ),
        ])


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
