"""``nm db`` — CLI mirror of the ``/db`` chat command.

Inspects the state database through the same ``db_tables / db_schema /
db_query / db_counts`` tools the chat command calls: tables, schema,
row counts, and read-only SELECT queries (a LIMIT is forced automatically).

    nm db tables [--json]          list every table with row counts
    nm db schema <table> [--json]  columns of one table
    nm db query "SELECT ..."       read-only SQL query (SELECT only)
    nm db counts [--top N]         the largest tables, for orientation
"""

from __future__ import annotations

import json
import sys
from typing import Any


def _usage() -> int:
    print("usage: nm db tables [--json] [--pattern STR]\n"
          "       nm db schema <table> [--json]\n"
          "       nm db query <select sql>\n"
          "       nm db counts [--top N]",
          file=sys.stderr)
    return 2


def _cmd_db(args: Any, context: Any) -> int:
    """Route ``nm db <verb>``."""
    words = list(getattr(args, "task", None) or [])
    if not words:
        return _usage()
    verb = words[0].lower()
    try:
        if verb == "tables":
            return _db_tables(args, context, words[1:])
        if verb == "schema":
            return _db_schema(args, context, words[1:])
        if verb == "query":
            return _db_query(args, context, words[1:])
        if verb == "counts":
            return _db_counts(args, context)
    except Exception as exc:  # noqa: BLE001 - fail fast with the real error
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"unknown db verb: {verb}", file=sys.stderr)
    return 2


def _as_json(args: Any) -> bool:
    return bool(getattr(args, "json", False))


def _call(context: Any, tool: str, **kwargs: Any) -> Any:
    outcome = context.tools.call(tool, **kwargs)
    if not outcome.ok:
        raise RuntimeError(getattr(outcome.error, "message", outcome.error))
    return outcome.value


def _db_tables(args: Any, context: Any, rest: list[str]) -> int:
    pattern = " ".join(rest) if rest else ""
    value = _call(context, "db_tables", pattern=pattern)
    if _as_json(args):
        print(json.dumps(value, indent=2, default=str))
        return 0
    print(f"tables ({value['count']}):")
    for table in value["tables"][:100]:
        print(f"  {table['name']} — {table['rows']} rows")
    return 0


def _db_schema(args: Any, context: Any, rest: list[str]) -> int:
    if not rest:
        print("usage: nm db schema <table>", file=sys.stderr)
        return 2
    value = _call(context, "db_schema", table=rest[0].strip())
    if _as_json(args):
        print(json.dumps(value, indent=2, default=str))
        return 0
    print(f"{value['table']} ({value['rows']} rows):")
    for col in value["columns"]:
        flags = "".join(flag for cond, flag in (
            (col["pk"], "PK"), (col["not_null"], "NN"),
        ) if cond)
        default = f" default={col['default']}" if col.get("default") is not None else ""
        print(f"  {col['name']} {col['type']} {flags}{default}")
    return 0


def _db_query(args: Any, context: Any, rest: list[str]) -> int:
    if not rest:
        print("usage: nm db query <select sql>", file=sys.stderr)
        return 2
    value = _call(context, "db_query", sql=" ".join(rest).strip())
    if _as_json(args):
        print(json.dumps(value, indent=2, default=str))
        return 0
    rows = value["data"][:20]
    if not rows:
        print(f"{value['sql']}\n(no rows)")
        return 0
    cols = value["columns"]
    print(" ".join(cols))
    for row in rows:
        print(" | ".join(str(row.get(c, ""))[:40] for c in cols))
    if len(rows) < value["rows"]:
        print(f"… {value['rows'] - len(rows)} more (limit forced ≤ 200)")
    return 0


def _db_counts(args: Any, context: Any) -> int:
    top = getattr(args, "top", 0) or 12
    value = _call(context, "db_counts", top=top)
    if _as_json(args):
        print(json.dumps(value, indent=2, default=str))
        return 0
    print("largest tables:")
    for table in value["tables"]:
        print(f"  {table['rows']:>8,}  {table['name']}")
    return 0
