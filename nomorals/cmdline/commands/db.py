"""``nm db`` — CLI mirror of the ``/db`` chat command.

Inspects the state database through the same ``db_tables / db_schema /
db_query / db_counts`` tools the chat command calls: tables, schema,
row counts, and read-only SELECT queries (a LIMIT is forced automatically).

    nm db tables [--json]          list every table with row counts
    nm db schema <table> [--json]  columns of one table
    nm db query "SELECT ..."       read-only SQL query (SELECT only)
    nm db counts [--top N]         the largest tables, for orientation

Plus the Mongo program's operator verbs (P2) — maintenance actions run
against the install's own context, the same shape as ``nm backup``:

    nm db carry [--dry-run] [--collections a,b] [--json]
        carry the central store's tables into the configured Mongo
        target (storage.mongo_uri / mongo_db) through the document
        seam; idempotent + resumable, progress journaled in the target
    nm db parity [--status] [--json]
        run the parity gate (fresh re-read of both stores; the verdict
        is recorded) — or, with --status, print the latest record, the
        exact dict the P6 cutover will gate on
"""

from __future__ import annotations

import json
import sys
from typing import Any


def _usage() -> int:
    print("usage: nm db tables [--json] [--pattern STR]\n"
          "       nm db schema <table> [--json]\n"
          "       nm db query <select sql>\n"
          "       nm db counts [--top N]\n"
          "       nm db carry [--dry-run] [--collections a,b] [--json]\n"
          "       nm db parity [--status] [--json]",
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
        if verb == "carry":
            return _db_carry(args, context, words[1:])
        if verb == "parity":
            return _db_parity(args, context, words[1:])
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


def _mongo_target(context: Any, *, require_live: bool = True) -> Any:
    """The configured Mongo target as a DocumentStore (never the seam's
    fallback logic — a carry/gate target must be Mongo or fail loudly)."""
    from ...storage.docstore import DEFAULT_MONGO_URI, MongoDocumentStore

    storage = context.settings.storage
    store = MongoDocumentStore(
        storage.mongo_uri or DEFAULT_MONGO_URI,
        storage.mongo_db or "nomorals")
    if require_live:
        health = store.health()
        if not health.ok:
            raise RuntimeError(
                f"mongo target is {health.state}: {health.detail}")
    return store


def _db_carry(args: Any, context: Any, rest: list[str]) -> int:
    from ...storage.carry import carry

    dry_run = "--dry-run" in rest
    collections: list[str] | None = None
    for i, word in enumerate(rest):
        if word == "--collections" and i + 1 < len(rest):
            collections = [c.strip() for c in rest[i + 1].split(",") if c.strip()]
    target = _mongo_target(context, require_live=not dry_run)

    def progress(event: dict[str, Any]) -> None:
        if event.get("event") == "batch":
            print(f"  {event['collection']}: {event['written']}"
                  f"/{event['source_count']} rows", flush=True)
        elif event.get("event") in {"done", "verified_skip"}:
            print(f"  {event['collection']}: {event['event']}"
                  f" ({event.get('written', event.get('count', 0))} written)",
                  flush=True)

    report = carry(context.db, target, dry_run=dry_run,
                   collections=collections, progress=progress)
    if _as_json(args):
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        moved = sum(c.written for c in report.collections)
        verb = "would move" if dry_run else "moved"
        print(f"carry {'(dry run) ' if dry_run else ''}complete: "
              f"{len(report.collections)} collections, {verb} {moved} rows, "
              f"{len(report.excluded)} tables excluded by the mapping "
              f"(see storage/carry.py EXCLUDED_TABLES)")
        for entry in report.collections:
            if entry.status == "diverged":
                print(f"  DIVERGED {entry.collection}: target checksum "
                      f"!= source after carry")
    return 0 if report.ok else 1


def _db_parity(args: Any, context: Any, rest: list[str]) -> int:
    from ...storage.carry import parity_gate_status, run_parity_gate

    if "--status" in rest:
        status = parity_gate_status(context.db)
        if _as_json(args):
            print(json.dumps(status, indent=2, default=str))
        elif status["at"] is None:
            print("parity: no gate has run yet — `nm db parity` runs one")
        else:
            verdict = "PASS" if status["passed"] else "FAIL"
            print(f"parity: {verdict} at {status['at']} "
                  f"(mapping {status['mapping_version']}, "
                  f"{len(status['divergences'])} divergent collections)")
            for div in status["divergences"]:
                print(f"  {div['collection']}: source={div['source_count']} "
                      f"target={div['target_count']} "
                      f"ids={div['divergent_ids'][:5]}")
        return 0
    target = _mongo_target(context)
    gate = run_parity_gate(context.db, target)
    if _as_json(args):
        print(json.dumps(gate, indent=2, default=str))
    else:
        verdict = "PASS" if gate["passed"] else "FAIL"
        print(f"parity gate: {verdict} — {len(gate['collections'])} "
              f"collections compared, record stored")
        for entry in gate["collections"]:
            if not entry["ok"]:
                print(f"  {entry['collection']}: source={entry['source_count']} "
                      f"target={entry['target_count']} "
                      f"only_source={entry['only_in_source'][:5]} "
                      f"only_target={entry['only_in_target'][:5]} "
                      f"differing={entry['differing'][:5]}")
    return 0 if gate["passed"] else 1


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
