"""``nm analytics`` — analytical queries over the live store.

Exposes ``storage/analytics.py`` (``open_analytics``: DuckDB when
installed, the SQLite passthrough otherwise) on the CLI, next to
``nm db`` — the analytical half ``nm db query`` cannot do: DuckDB's
columnar engine for GROUP BY / window functions over full-table scans,
no forced LIMIT truncation, and backend-aware reads (on a Mongo install
a seam-served table is synced from the seam, never silently read from
the SQLite fossil — the passthrough refuses those queries loudly).

    nm analytics status [--json]     which backend is live + its stats
    nm analytics tables [--json]     every live table with row counts
    nm analytics query "<sql>" [--limit N] [--json]
                                     read-only analytical SQL (SELECT/WITH)
    nm analytics describe <table> [--json]
    nm analytics sample <table> [--rows N] [--json]
    nm analytics histogram <table> <column> [--bins N] [--json]
    nm analytics profile "<sql>" [--json]
                                     EXPLAIN ANALYZE (DuckDB) / QUERY PLAN

``query`` / ``describe`` / ``sample`` / ``histogram`` / ``profile`` sync
the SQL's referenced tables into DuckDB first (``load_table``); on the
SQLite passthrough they run directly against the live database. A
missing database, a non-SELECT statement, or an unknown table is an
honest refusal (exit 1), never invented data. Empty results print
"0 rows".
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from ...storage.analytics import analytics_source_rows, open_analytics

#: FROM/JOIN table references in analytical SQL — the same shape as
#: ``storage/analytics.py::_SQL_TABLE_REF_RE`` (which stays private):
#: only used to decide which tables to sync into DuckDB before a query,
#: never to rewrite anyone's SQL.
_TABLE_REF_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+\"?([A-Za-z_][A-Za-z0-9_]*)\"?", re.IGNORECASE)

_DEFAULT_LIMIT = 100
_MAX_LIMIT = 1000


def _usage() -> int:
    print("usage: nm analytics status [--json]\\n"
          "       nm analytics tables [--json]\\n"
          "       nm analytics query \"<select sql>\" [--limit N] [--json]\\n"
          "       nm analytics describe <table> [--json]\\n"
          "       nm analytics sample <table> [--rows N] [--json]\\n"
          "       nm analytics histogram <table> <column> [--bins N] [--json]\\n"
          "       nm analytics profile \"<select sql>\" [--json]",
          file=sys.stderr)
    return 2


def _cmd_analytics(args: Any, context: Any) -> int:
    """Route ``nm analytics <verb>``."""
    words = list(getattr(args, "task", None) or [])
    if not words:
        return _usage()
    verb = words[0].lower()
    try:
        if verb == "status":
            return _analytics_status(args, context)
        if verb == "tables":
            return _analytics_tables(args, context)
        if verb == "query":
            return _analytics_query(args, context, words[1:])
        if verb == "describe":
            return _analytics_describe(args, context, words[1:])
        if verb == "sample":
            return _analytics_sample(args, context, words[1:])
        if verb == "histogram":
            return _analytics_histogram(args, context, words[1:])
        if verb == "profile":
            return _analytics_profile(args, context, words[1:])
    except Exception as exc:  # noqa: BLE001 - fail fast with the real error
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"unknown analytics verb: {verb}", file=sys.stderr)
    return 2


def _as_json(args: Any) -> bool:
    return bool(getattr(args, "json", False))


def _db(context: Any) -> Any:
    db = getattr(context, "db", None)
    if db is None:
        raise RuntimeError("no database attached to this context")
    return db


def _refs(sql: str) -> list[str]:
    """Table names a query touches, in first-seen order, deduplicated."""
    seen: list[str] = []
    for match in _TABLE_REF_RE.finditer(sql or ""):
        name = match.group(1)
        if name not in seen:
            seen.append(name)
    return seen


def _ensure_loaded(engine: Any, db: Any, tables: list[str]) -> None:
    """Sync ``tables`` into the analytical engine when it keeps copies.

    DuckDB syncs from the live source (the seam for seam-served tables,
    SQLite otherwise); the SQLite passthrough reads the live database
    directly, so there is nothing to sync.
    """
    if getattr(engine, "name", "") != "duckdb":
        return
    for table in tables:
        engine.load_table(db, table)


def _limit(args: Any) -> int:
    try:
        wanted = int(getattr(args, "limit", 0) or _DEFAULT_LIMIT)
    except (TypeError, ValueError):
        wanted = _DEFAULT_LIMIT
    return max(1, min(wanted, _MAX_LIMIT))


def _print_rows(rows: list[dict[str, Any]], args: Any, *,
               title: str = "") -> int:
    total = len(rows)
    shown = rows[:_limit(args)]
    if _as_json(args):
        print(json.dumps({"rows": total, "showing": len(shown),
                          "data": shown}, indent=2, default=str))
        return 0
    if title:
        print(f"{title} ({total} rows"
              + (f", showing {len(shown)}" if len(shown) < total else "")
              + "):")
    elif total == 0:
        print("0 rows")
        return 0
    else:
        print(f"{total} rows"
              + (f" (showing {len(shown)})" if len(shown) < total else "")
              + ":")
    for row in shown:
        print("  " + ", ".join(f"{k}={v}" for k, v in row.items()))
    return 0


def _analytics_status(args: Any, context: Any) -> int:
    engine = open_analytics(_db(context))
    if _as_json(args):
        print(json.dumps({
            "backend": engine.name,
            "tables": engine.table_names(),
            "stats": dict(getattr(engine, "stats", {})),
        }, indent=2, default=str))
        return 0
    print(engine.format_stats())
    return 0


def _analytics_tables(args: Any, context: Any) -> int:
    db = _db(context)
    rows = []
    for name in db.tables():
        try:
            seam = analytics_source_rows(db, name)
            count = len(seam) if seam is not None else db.row_count(name)
        except Exception:  # noqa: BLE001 - one bad table must not kill the listing
            count = -1
        rows.append({"name": name, "rows": count})
    rows.sort(key=lambda r: -r["rows"])
    if _as_json(args):
        print(json.dumps({"count": len(rows), "tables": rows},
                         indent=2, default=str))
        return 0
    print(f"tables ({len(rows)}):")
    for row in rows[:100]:
        count = "?" if row["rows"] < 0 else str(row["rows"])
        print(f"  {row['name']} — {count} rows")
    return 0


def _analytics_query(args: Any, context: Any, rest: list[str]) -> int:
    if not rest:
        print("usage: nm analytics query \"<select sql>\" [--limit N]",
              file=sys.stderr)
        return 2
    sql = " ".join(rest)
    db = _db(context)
    engine = open_analytics(db)
    _ensure_loaded(engine, db, _refs(sql))
    return _print_rows(engine.query(sql), args)


def _analytics_describe(args: Any, context: Any, rest: list[str]) -> int:
    if not rest:
        print("usage: nm analytics describe <table>", file=sys.stderr)
        return 2
    table = rest[0]
    db = _db(context)
    engine = open_analytics(db)
    _ensure_loaded(engine, db, [table])
    info = engine.describe(table)
    if _as_json(args):
        print(json.dumps(info, indent=2, default=str))
        return 0
    print(f"{info['table']} — {info['rows']} rows")
    for col in info["columns"]:
        print(f"  {col['name']} ({col['type']})")
    return 0


def _analytics_sample(args: Any, context: Any, rest: list[str]) -> int:
    if not rest:
        print("usage: nm analytics sample <table> [--rows N]",
              file=sys.stderr)
        return 2
    table = rest[0]
    try:
        n = max(1, int(getattr(args, "rows", 0) or 100))
    except (TypeError, ValueError):
        n = 100
    db = _db(context)
    engine = open_analytics(db)
    _ensure_loaded(engine, db, [table])
    rows = engine.sample(table, n)
    return _print_rows(rows, args, title=f"sample of {table}")


def _analytics_histogram(args: Any, context: Any, rest: list[str]) -> int:
    if len(rest) < 2:
        print("usage: nm analytics histogram <table> <column> [--bins N]",
              file=sys.stderr)
        return 2
    table, column = rest[0], rest[1]
    try:
        bins = max(1, int(getattr(args, "bins", 0) or 20))
    except (TypeError, ValueError):
        bins = 20
    db = _db(context)
    engine = open_analytics(db)
    _ensure_loaded(engine, db, [table])
    rows = engine.histogram(table, column, bins=bins)
    return _print_rows(rows, args, title=f"histogram of {table}.{column}")


def _analytics_profile(args: Any, context: Any, rest: list[str]) -> int:
    if not rest:
        print("usage: nm analytics profile \"<select sql>\"",
              file=sys.stderr)
        return 2
    sql = " ".join(rest)
    db = _db(context)
    engine = open_analytics(db)
    _ensure_loaded(engine, db, _refs(sql))
    return _print_rows(engine.profile(sql), args, title="profile")
