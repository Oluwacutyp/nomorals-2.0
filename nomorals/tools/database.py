"""Database tools: structured storage, queried from the agent plane.

The whole system is SQLite-backed (migrations, WAL, FTS). These tools give
the agents — and the owner via /db — first-class access to that structured
storage instead of scraping flat files:

* ``db_tables``  — every table with its row count
* ``db_schema``  — columns of one table (type, not-null, default, pk)
* ``db_query``   — SELECT-only, LIMIT-forced (the same rule Devon enforces)
* ``db_counts``  — the big tables first, for a quick orientation shot

Writes stay with the owning subsystems (memory manager, research agent, …);
these tools are the *read* plane, gated behind ``db.read``.
"""

from __future__ import annotations

import re
from typing import Any

from ..core.errors import ToolError, ValidationError
from ..core.policy import Capability

__all__ = ["is_select_only", "register"]

_MAX_LIMIT = 5000  # it's the owner's own state DB — deep queries are legal
_ROW_LIMIT_DEFAULT = 25


def is_select_only(sql: str) -> bool:
    """True when ``sql`` is a single read statement (SELECT / WITH ... SELECT)."""
    s = re.sub(r"\s+", " ", (sql or "").strip()).rstrip(";").strip()
    if not s:
        return False
    if ";" in s:
        return False  # one statement only
    head = s.split(None, 1)[0].lower()
    if head == "select":
        return True
    if head == "with":
        return bool(re.search(r"\bselect\b", s, re.IGNORECASE)) and not re.search(
            r"\b(insert|update|delete|drop|alter|create|replace|vacuum|attach)\b",
            s, re.IGNORECASE,
        )
    if head in {"pragma", "explain"}:
        return head == "pragma" and re.match(r"(?i)^pragma\s+(table_info|table_xinfo|index_list)\(", s)
    return False


def _force_limit(sql: str, limit: int) -> str:
    s = re.sub(r"\s+", " ", (sql or "").strip()).rstrip(";").strip()
    if re.search(r"(?i)\blimit\s+\d+\s*$", s):
        m = re.search(r"(?i)\blimit\s+(\d+)$", s)
        existing = int(m.group(1))
        if existing > limit:
            s = re.sub(r"(?i)\blimit\s+\d+$", f"limit {limit}", s)
        return s
    return f"{s} limit {limit}"


def _rows_to_dicts(rows: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(dict(row))
    return out


def register(registry: Any) -> None:
    """Attach the read-only database tools to a registry."""
    context = registry.context
    db = getattr(context, "db", None) if context is not None else None

    @registry.register(
        "db_tables",
        description="List every table in the state database with its row count.",
        capability=Capability.DB_READ,
        parameters={"pattern": "str (optional) — substring filter on table name"},
    )
    def db_tables(pattern: str = "") -> dict[str, Any]:
        if db is None:
            raise ToolError("no database attached to this context")
        rows = db.query("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
        out = []
        for row in rows:
            name = row["name"]
            if pattern and pattern.lower() not in name.lower():
                continue
            try:
                count = int(db.scalar(f"SELECT COUNT(*) FROM {name}", default=0) or 0)
            except Exception:  # noqa: BLE001 - a broken table still gets listed
                count = -1
            out.append({"name": name, "rows": count})
        return {"count": len(out), "tables": out}

    @registry.register(
        "db_schema",
        description="Columns of one table: name, type, not-null, default, primary key.",
        capability=Capability.DB_READ,
        parameters={"table": "str — table name"},
    )
    def db_schema(table: str) -> dict[str, Any]:
        if db is None:
            raise ToolError("no database attached to this context")
        table = (table or "").strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
            raise ValidationError(f"bad table name: {table!r}")
        exists = db.scalar(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ?", (table,)
        )
        if not exists:
            raise ToolError(f"no such table: {table}")
        rows = db.query(f"PRAGMA table_info({table})")
        columns = [
            {
                "name": r["name"],
                "type": r["type"] or "",
                "not_null": bool(r["notnull"]),
                "default": r["dflt_value"],
                "pk": bool(r["pk"]),
            }
            for r in rows
        ]
        try:
            row_count = int(db.scalar(f"SELECT COUNT(*) FROM {table}", default=0) or 0)
        except Exception:  # noqa: BLE001
            row_count = -1
        return {"table": table, "rows": row_count, "columns": columns}

    @registry.register(
        "db_query",
        description=(
            "Run a read-only SQL query against the state database. "
            "SELECT (or WITH ... SELECT) only; a LIMIT is forced automatically."
        ),
        capability=Capability.DB_READ,
        parameters={
            "sql": "str — a single SELECT statement",
            "limit": "int (optional, default 25, max 200)",
        },
    )
    def db_query(sql: str, *, limit: int = _ROW_LIMIT_DEFAULT) -> dict[str, Any]:
        if db is None:
            raise ToolError("no database attached to this context")
        if not is_select_only(sql):
            raise ToolError("only single read statements (SELECT / WITH ... SELECT) are allowed")
        limit = max(1, min(int(limit or _ROW_LIMIT_DEFAULT), _MAX_LIMIT))
        safe_sql = _force_limit(sql, limit)
        rows = _rows_to_dicts(db.query(safe_sql))
        cols = list(rows[0].keys()) if rows else []
        return {"sql": safe_sql, "rows": len(rows), "columns": cols, "data": rows}

    @registry.register(
        "db_counts",
        description="The largest tables in the state database, for orientation.",
        capability=Capability.DB_READ,
        parameters={"top": "int (optional, default 12)"},
    )
    def db_counts(top: int = 12) -> dict[str, Any]:
        if db is None:
            raise ToolError("no database attached to this context")
        top = max(1, min(int(top or 12), 50))
        rows = db.query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        sized = []
        for row in rows:
            name = row["name"]
            try:
                count = int(db.scalar(f"SELECT COUNT(*) FROM {name}", default=0) or 0)
            except Exception:  # noqa: BLE001
                continue
            sized.append({"name": name, "rows": count})
        sized.sort(key=lambda t: -t["rows"])
        return {"count": len(sized), "tables": sized[:top]}
