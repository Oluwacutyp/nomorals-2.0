"""The carry: SQLite → Mongo data movement + the parity gate (Mongo P2).

Why this exists
---------------
The Mongo program (owner directive 2026-11-08: "Database to mongodb and
God tier", plan of record in the mongo census) ports Devon's persistence
domain by domain onto the document-store seam
(:mod:`nomorals.storage.docstore`). Before any install may *cut over* —
before Mongo becomes the store Devon reads — the live data must be carried
across and parity must be **proven**, not assumed. This module is that
proof machinery:

* :func:`inventory` + :data:`EXCLUDED_TABLES` — the row↔document mapping,
  owned here and nowhere else. Every table in the central store gets a
  disposition: mapped (table → same-named collection) or excluded *by
  name, with the reason recorded*. Nothing is silently skipped.
* :func:`carry` — the exporter. Reads the live SQLite store (never writes
  to it), writes documents through the DocumentStore seam (``put`` is the
  seam's sanctioned write path; there is no bulk verb in the protocol, so
  batching lives on the read/progress side). Idempotent (re-running
  converges, including deleting target documents whose source rows are
  gone), resumable (per-collection progress — counts + checksums — is
  journaled in the target's ``_carry_state`` collection; a re-run skips
  collections whose recorded source checksum still matches), and supports
  a dry-run that moves nothing, not even progress state.
* :func:`run_parity_gate` + :func:`parity_gate_status` — the gate. An
  independent pass re-reads BOTH stores fresh and compares every mapped
  collection with :func:`nomorals.storage.parity.check_document_parity`
  (counts + ordered canonical checksums + divergent ids) plus sampled
  field-level equality. The verdict is persisted as a durable record in
  the operator's SQLite database (``carry_parity_records``, a
  runtime-created table — migrations stay frozen). ``parity_gate_status``
  is the exact check the P6 cutover will call.

Scope (per the census phases): the CENTRAL store only. Sidecar files are
P5; the queue/scheduler coordination residue and FTS are P4 — those
tables are excluded here *by name with reasons*, not forgotten.

The mapping (v1, byte-parity first — census §5.6)
-------------------------------------------------
* Table → collection of the same name; the primary key becomes ``_id``:
  a single-column PK stringifies (``5`` and ``"5"`` are the same id), a
  composite PK joins its parts with U+001F, a PK-less table falls back to
  ``rowid:<n>`` (the schema has no WITHOUT ROWID tables; rowids are
  stable for the lifetime of a quiesced source file).
* Column values carry verbatim: INTEGER → int, REAL → float (epoch
  timestamps stay numeric), TEXT → str — JSON-as-TEXT columns stay
  strings in v1; promoting them to subdocuments is a later, separately
  verified pass. BLOBs become ``{"$blob_b64": <base64>}`` — a tagged,
  reversible, JSON-safe form, so the canonical parity form *is* the
  stored form. NULL → None.
* The DocumentStore protocol reserves ``_id`` and the managed
  ``expires_at`` field. A source column named ``_id`` is carried as
  ``sql__id``; one named ``expires_at`` as ``sql_expires_at`` (its value
  is data, not the store's TTL) — except where a per-table transform
  says otherwise.
* Per-table transforms (declared in :data:`TRANSFORMS`): ``kv_store``
  rows become exactly the documents the P1 seam port writes
  (``{_id: key, value, kind, updated_at}`` with the row's own
  ``expires_at`` as the managed expiry) — one collection, one shape;
  a raw row dump into ``kv_store`` would corrupt the live KV view.

The carry owns its target collections while it runs: run it against a
quiesced source (the census's rule — a copy of the live file first) and
a target no one else is writing.
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger
from .docstore import _matches, mask_mongo_uri
from .failures import record_failure
from .parity import check_document_parity, collection_checksum

__all__ = [
    "MAPPING_VERSION",
    "STATE_COLLECTION",
    "RECORDS_TABLE",
    "EXCLUDED_TABLES",
    "TableMapping",
    "CarryCollectionResult",
    "CarryReport",
    "RowDocumentView",
    "inventory",
    "mapped_tables",
    "row_to_document",
    "carry",
    "run_parity_gate",
    "parity_gate_status",
]

_log = get_logger(__name__)

#: Bump when any mapping rule changes; stamped into progress + records.
MAPPING_VERSION = "carry-v1"

#: Target-side collection holding per-collection carry progress.
STATE_COLLECTION = "_carry_state"

#: Runtime-created table (operator's SQLite db) holding parity records.
RECORDS_TABLE = "carry_parity_records"

#: Tables the carry deliberately does NOT move, each with its reason.
#: Anything in the live store that is neither mapped nor named here (or
#: classified as FTS residue at inventory time) fails the inventory pin
#: in tests — an unmapped table is a bug report, never a silent skip.
EXCLUDED_TABLES: dict[str, str] = {
    "schema_migrations": (
        "sqlite-rung bookkeeping: the migration ledger belongs to the "
        "SQLite schema machinery; Mongo collections derive from "
        "migrations.py read-only and keep no ledger of their own"),
    "schema_migrations_repeatable": (
        "sqlite-rung bookkeeping (repeatable-migration ledger), same "
        "reason as schema_migrations"),
    "sqlite_sequence": (
        "sqlite-internal AUTOINCREMENT counter; Mongo _ids carry the "
        "identity it tracked"),
    "work_queue": (
        "coordination residue: atomic claim/lease semantics port in P4 "
        "(findOneAndUpdate), not bulk carry — carrying rows would move "
        "lease state, not data"),
    "schedule_jobs": (
        "coordination residue: scheduler claim/lease port is P4, same "
        "reason as work_queue"),
    "schedule_runs": (
        "scheduler run log of the coordination residue — moves with "
        "schedule_jobs in P4"),
    "docstore_docs": (
        "the document seam's own SQLite-rung table: its Mongo "
        "counterpart is written by the seam itself (shadow parity), "
        "not by row carry"),
    RECORDS_TABLE: (
        "the P2 parity gate's own bookkeeping; records live with the "
        "operator's SQLite database by design"),
}

#: Exclusion reason for FTS5 virtual tables and their shadow tables.
FTS_EXCLUSION_REASON = (
    "FTS residue: full-text search stays on the SQLite rung (P4) — "
    "there is no self-hosted Mongo FTS5/BM25 parity")

#: DocumentStore-managed field names a source column may not occupy.
_RESERVED = {"_id": "sql__id", "expires_at": "sql_expires_at"}

_ID_SEP = "\x1f"


# ── the mapping ──────────────────────────────────────────────────────────────

@dataclass
class Column:
    name: str
    type: str = ""
    pk: int = 0  # 0 = not in PK; otherwise 1-based position in the PK


@dataclass
class TableMapping:
    """One table's disposition: mapped (with its shape) or excluded."""

    table: str
    collection: str = ""
    columns: list[Column] = field(default_factory=list)
    excluded_reason: str = ""

    @property
    def mapped(self) -> bool:
        return not self.excluded_reason

    @property
    def pk_columns(self) -> list[str]:
        return [c.name for c in sorted(
            (c for c in self.columns if c.pk), key=lambda c: c.pk)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "collection": self.collection if self.mapped else None,
            "mapped": self.mapped,
            "pk": self.pk_columns,
            "columns": [c.name for c in self.columns],
            "excluded_reason": self.excluded_reason or None,
        }


def _live_tables(db: Any) -> list[dict[str, Any]]:
    rows = db.query(
        "SELECT name, type, sql FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' OR name = 'sqlite_sequence' "
        "ORDER BY name")
    return [dict(r) for r in rows]


def _columns(db: Any, table: str) -> list[Column]:
    rows = db.query(f'PRAGMA table_info("{table}")')
    return [
        Column(name=r["name"], type=r["type"] or "", pk=int(r["pk"] or 0))
        for r in rows
    ]


def inventory(db: Any) -> list[TableMapping]:
    """Every table/view in the central store, with its disposition.

    Derived from the LIVE store (``sqlite_master`` + ``PRAGMA``), so the
    mapping can never drift from the schema the migrations actually
    built — including runtime-created tables, which get the default
    disposition (mapped) unless named in :data:`EXCLUDED_TABLES` or
    classified as FTS residue here.
    """
    entries = _live_tables(db)
    virtual = {
        e["name"] for e in entries
        if e["type"] == "table" and e["sql"]
        and "using fts" in e["sql"].lower().replace("  ", " ")
    }
    out: list[TableMapping] = []
    for entry in entries:
        name, kind = entry["name"], entry["type"]
        if kind == "view":
            out.append(TableMapping(
                table=name,
                excluded_reason="a view: derived at read time, holds no "
                                "data of its own to carry"))
            continue
        if kind != "table":
            continue
        if name in EXCLUDED_TABLES:
            out.append(TableMapping(table=name,
                                    excluded_reason=EXCLUDED_TABLES[name]))
            continue
        if name in virtual or any(name.startswith(v + "_") for v in virtual):
            out.append(TableMapping(table=name,
                                    excluded_reason=FTS_EXCLUSION_REASON))
            continue
        out.append(TableMapping(table=name, collection=name,
                                columns=_columns(db, name)))
    return out


def mapped_tables(db: Any) -> list[TableMapping]:
    return [m for m in inventory(db) if m.mapped]


# ── row → document ───────────────────────────────────────────────────────────

def encode_value(value: Any) -> Any:
    """One SQLite value → its canonical document value (module docstring)."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"$blob_b64": base64.b64encode(bytes(value)).decode("ascii")}
    return str(value)


def _doc_id(mapping: TableMapping, row: Mapping[str, Any]) -> str:
    pk = mapping.pk_columns
    if len(pk) == 1:
        return str(row[pk[0]])
    if pk:
        return _ID_SEP.join(str(row[c]) for c in pk)
    return f"rowid:{row['__carry_rowid']}"


def _generic_row(mapping: TableMapping, row: Mapping[str, Any]):
    doc_id = _doc_id(mapping, row)
    pk = set(mapping.pk_columns)
    doc: dict[str, Any] = {}
    for col in mapping.columns:
        if col.name in pk:
            continue
        doc[_RESERVED.get(col.name, col.name)] = encode_value(row[col.name])
    return doc_id, doc, None


def _kv_store_row(mapping: TableMapping, row: Mapping[str, Any]):
    """kv_store rows → the exact documents the P1 seam port writes."""
    doc = {
        "value": row["value"],
        "kind": row["kind"],
        "updated_at": row["updated_at"],
    }
    expires = row.get("expires_at")
    return str(row["key"]), doc, (float(expires) if expires is not None else None)


#: Per-table row transforms: ``(mapping, row) -> (doc_id, fields, expires_at)``.
TRANSFORMS: dict[str, Callable[..., tuple[str, dict[str, Any], float | None]]] = {
    "kv_store": _kv_store_row,
}


def row_to_document(mapping: TableMapping,
                    row: Mapping[str, Any]) -> tuple[str, dict[str, Any], float | None]:
    """Map one source row to ``(doc_id, fields, managed_expires_at)``.

    Accepts any row mapping (``sqlite3.Row`` included); transforms always
    receive a plain dict.
    """
    transform = TRANSFORMS.get(mapping.table, _generic_row)
    return transform(mapping, dict(row))


def _select_sql(mapping: TableMapping) -> str:
    cols = ", ".join(f'"{c.name}"' for c in mapping.columns)
    return f'SELECT rowid AS __carry_rowid, {cols} FROM "{mapping.table}"'


def _iter_rows(db: Any, mapping: TableMapping, batch_size: int):
    """Stream a table's rows in rowid order, ``batch_size`` at a time."""
    sql = _select_sql(mapping)
    last = 0
    while True:
        rows = db.query(
            f"{sql} WHERE rowid > ? ORDER BY rowid LIMIT ?",
            (last, batch_size))
        if not rows:
            return
        for row in rows:
            yield dict(row)
        last = int(rows[-1]["__carry_rowid"])


class RowDocumentView:
    """A read-only DocumentStore shape over the SQLite source, via the mapping.

    This is what lets the parity gate reuse
    :func:`nomorals.storage.parity.check_document_parity` unchanged: the
    source presents the exact documents the carry wrote (same ids, same
    fields, same managed ``expires_at``), so the canonical comparison is
    apples-to-apples by construction. Only the read side of the protocol
    is implemented — the view is a proof instrument, never a write path.
    """

    name = "sqlite-source"
    capabilities = {"read_view": True}

    def __init__(self, db: Any) -> None:
        self._db = db
        self._by_collection = {m.collection: m for m in mapped_tables(db)}

    def _docs(self, collection: str) -> list[dict[str, Any]]:
        mapping = self._by_collection[collection]
        docs = []
        for row in _iter_rows(self._db, mapping, 1000):
            doc_id, fields, expires_at = row_to_document(mapping, row)
            doc = dict(fields)
            doc["_id"] = doc_id
            doc["expires_at"] = expires_at
            docs.append(doc)
        docs.sort(key=lambda d: d["_id"])
        return docs

    def collections(self) -> list[str]:
        return sorted(self._by_collection)

    def get(self, collection: str, doc_id: str, *,
            include_expired: bool = False) -> dict[str, Any] | None:
        for doc in self._docs(collection):
            if doc["_id"] == doc_id:
                if include_expired or doc.get("expires_at") is None \
                        or float(doc["expires_at"]) > time.time():
                    return doc
                return None
        return None

    def find(self, collection: str, filter: Mapping[str, Any] | None = None, *,
             include_expired: bool = False, after_id: str | None = None,
             limit: int | None = None) -> list[dict[str, Any]]:
        now = time.time()
        out: list[dict[str, Any]] = []
        for doc in self._docs(collection):
            if after_id is not None and doc["_id"] <= after_id:
                continue
            if _matches(doc, filter, now, include_expired=include_expired):
                out.append(doc)
                if limit is not None and len(out) >= limit:
                    break
        return out

    def count(self, collection: str, filter: Mapping[str, Any] | None = None, *,
              include_expired: bool = False) -> int:
        return len(self.find(collection, filter, include_expired=include_expired))


# ── the carry ────────────────────────────────────────────────────────────────

@dataclass
class CarryCollectionResult:
    collection: str
    status: str  # "carried" | "verified_skip" | "dry_run" | "diverged"
    source_count: int = 0
    written: int = 0
    deleted: int = 0
    source_checksum: str = ""
    target_checksum: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection": self.collection,
            "status": self.status,
            "source_count": self.source_count,
            "written": self.written,
            "deleted": self.deleted,
            "source_checksum": self.source_checksum,
            "target_checksum": self.target_checksum,
        }


@dataclass
class CarryReport:
    dry_run: bool
    mapping_version: str
    collections: list[CarryCollectionResult] = field(default_factory=list)
    excluded: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.status != "diverged" for c in self.collections)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "mapping_version": self.mapping_version,
            "ok": self.ok,
            "collections": [c.to_dict() for c in self.collections],
            "excluded": self.excluded,
        }


def _state_doc(target: Any, collection: str) -> dict[str, Any] | None:
    try:
        return target.get(STATE_COLLECTION, collection, include_expired=True)
    except Exception:  # noqa: BLE001 - a target without state just re-carries
        return None


def carry(db: Any, target: Any, *, dry_run: bool = False,
          batch_size: int = 500,
          collections: list[str] | None = None,
          progress: Callable[[dict[str, Any]], None] | None = None,
          ) -> CarryReport:
    """Carry every mapped table from ``db`` into ``target`` (a DocumentStore).

    Never writes to ``db``. With ``dry_run`` nothing is written anywhere —
    the report carries the counts + source checksums that *would* move.
    ``progress`` (optional) receives one event dict per batch and per
    finished collection. A collection whose target-side progress record
    still matches the fresh source checksum is verified and skipped;
    anything else is (re)carried — ``put`` upserts + pruning of
    source-deleted documents make the re-run converge.
    """
    inv = inventory(db)
    excluded = [m.to_dict() for m in inv if not m.mapped]
    report = CarryReport(dry_run=dry_run, mapping_version=MAPPING_VERSION,
                         excluded=excluded)
    view = RowDocumentView(db)
    wanted = set(collections) if collections else None
    for mapping in inv:
        if not mapping.mapped:
            continue
        if wanted is not None and mapping.collection not in wanted:
            continue
        report.collections.append(_carry_one(
            db, target, view, mapping, batch_size=batch_size,
            dry_run=dry_run, progress=progress))
    return report


def _carry_one(db: Any, target: Any, view: RowDocumentView,
               mapping: TableMapping, *, batch_size: int, dry_run: bool,
               progress: Callable[[dict[str, Any]], None] | None,
               ) -> CarryCollectionResult:
    collection = mapping.collection
    source_count, source_checksum = collection_checksum(view, collection)
    result = CarryCollectionResult(
        collection=collection, status="carried",
        source_count=source_count, source_checksum=source_checksum)

    state = None if dry_run else _state_doc(target, collection)
    if state and state.get("status") == "done" \
            and state.get("mapping_version") == MAPPING_VERSION \
            and state.get("source_checksum") == source_checksum:
        result.status = "verified_skip"
        result.target_checksum = str(state.get("target_checksum", ""))
        if progress:
            progress({"collection": collection, "event": "verified_skip",
                      "count": source_count})
        return result
    if dry_run:
        result.status = "dry_run"
        return result

    seen: set[str] = set()
    written = 0
    for row in _iter_rows(db, mapping, batch_size):
        doc_id, fields, expires_at = row_to_document(mapping, row)
        target.put(collection, doc_id, fields, expires_at=expires_at)
        seen.add(doc_id)
        written += 1
        if progress and written % batch_size == 0:
            progress({"collection": collection, "event": "batch",
                      "written": written, "source_count": source_count})

    # Converge: documents whose source rows are gone must not linger.
    deleted = 0
    target_ids = [d["_id"] for d in target.find(
        collection, include_expired=True)]
    for stale in sorted(set(target_ids) - seen):
        if target.delete(collection, stale):
            deleted += 1
    result.written, result.deleted = written, deleted

    _, target_checksum = collection_checksum(target, collection)
    result.target_checksum = target_checksum
    status = "carried" if target_checksum == source_checksum else "diverged"
    result.status = status
    target.put(STATE_COLLECTION, collection, {
        "collection": collection,
        "status": "done" if status == "carried" else "diverged",
        "source_count": source_count,
        "source_checksum": source_checksum,
        "target_checksum": target_checksum,
        "mapping_version": MAPPING_VERSION,
        "at": time.time(),
    })
    if status == "diverged":
        record_failure(
            "storage.carry",
            f"carry of {collection} diverged: source checksum "
            f"{source_checksum[:12]} vs target {target_checksum[:12]}")
    if progress:
        progress({"collection": collection, "event": "done",
                  "written": written, "deleted": deleted, "status": status})
    return result


# ── the parity gate ──────────────────────────────────────────────────────────

def _ensure_records_table(db: Any) -> None:
    db.execute(
        f"CREATE TABLE IF NOT EXISTS {RECORDS_TABLE} ("
        "id TEXT PRIMARY KEY, at REAL NOT NULL, mapping_version TEXT NOT NULL, "
        "source TEXT NOT NULL, target TEXT NOT NULL, passed INTEGER NOT NULL, "
        "report TEXT NOT NULL)")


def _source_identity(db: Any) -> str:
    path = getattr(db, "path", None)
    return f"sqlite:{path}" if path is not None else "sqlite::memory:"


def _target_identity(target: Any) -> str:
    name = getattr(target, "name", type(target).__name__)
    db_name = getattr(target, "_database_name", None)
    uri = getattr(target, "_uri", None)
    if db_name and uri:
        return f"{name}:{mask_mongo_uri(str(uri))} db={db_name}"
    return str(name)


def _field_level_sample(view: RowDocumentView, target: Any, collection: str,
                        ids: list[str], *, sample: int) -> list[dict[str, Any]]:
    """Field-by-field comparison on a deterministic sample of documents."""
    if not ids or sample <= 0:
        return []
    step = max(1, len(ids) // sample)
    picked = ids[::step][:sample]
    mismatches: list[dict[str, Any]] = []
    for doc_id in picked:
        a = view.get(collection, doc_id, include_expired=True)
        b = target.get(collection, doc_id, include_expired=True)
        if a is None or b is None:
            mismatches.append({"id": doc_id,
                               "fields": ["<presence>"],
                               "only_in": "source" if b is None else "target"})
            continue
        keys = sorted(set(a) | set(b))
        differing = [k for k in keys if a.get(k) != b.get(k)]
        if differing:
            mismatches.append({"id": doc_id, "fields": differing[:20]})
    return mismatches


def run_parity_gate(db: Any, target: Any, *, sample: int = 25,
                    record: bool = True,
                    limit: int = 2_000_000) -> dict[str, Any]:
    """Re-read both stores fresh and prove (or refute) parity, per collection.

    PASS requires, for every mapped collection: equal counts, equal
    ordered canonical checksums, no divergent ids from
    :func:`check_document_parity`, and no field-level mismatch in the
    sample. The verdict is persisted (``record=True``) as a durable
    parity record and divergences are journaled to the failure ledger.
    """
    view = RowDocumentView(db)
    per_collection: list[dict[str, Any]] = []
    passed = True
    for mapping in mapped_tables(db):
        collection = mapping.collection
        report = check_document_parity(view, target, collection, limit=limit)
        entry = report.to_dict()
        entry["source_count"] = entry.pop("count_a")
        entry["target_count"] = entry.pop("count_b")
        entry["source_checksum"] = entry.pop("checksum_a")
        entry["target_checksum"] = entry.pop("checksum_b")
        entry["only_in_source"] = entry.pop("only_in_a")
        entry["only_in_target"] = entry.pop("only_in_b")
        ids = sorted({d["_id"] for d in view.find(
            collection, include_expired=True, limit=limit)}) if sample > 0 else []
        entry["field_mismatches"] = _field_level_sample(
            view, target, collection, ids, sample=sample)
        entry["ok"] = bool(report.ok and not entry["field_mismatches"])
        if not entry["ok"]:
            passed = False
        per_collection.append(entry)

    gate = {
        "passed": passed,
        "at": time.time(),
        "mapping_version": MAPPING_VERSION,
        "source": _source_identity(db),
        "target": _target_identity(target),
        "collections": per_collection,
    }
    if record:
        _ensure_records_table(db)
        db.insert(RECORDS_TABLE, {
            "id": uuid.uuid4().hex,
            "at": gate["at"],
            "mapping_version": MAPPING_VERSION,
            "source": gate["source"],
            "target": gate["target"],
            "passed": 1 if passed else 0,
            "report": json.dumps(gate, default=str),
        })
    if not passed:
        bad = [c["collection"] for c in per_collection if not c["ok"]]
        record_failure(
            "storage.carry.parity_gate",
            f"parity gate FAILED for {bad}: see {RECORDS_TABLE} record "
            f"at {gate['at']}")
        _log.warning("parity gate failed: %s", bad)
    return gate


def parity_gate_status(db: Any) -> dict[str, Any]:
    """The cutover-readiness hook (P6 calls exactly this).

    Returns the latest parity record as ``{passed, at, mapping_version,
    divergences}`` — ``passed`` is False with ``at=None`` when no gate
    has ever run: no record is not a pass.
    """
    try:
        row = db.query_one(
            f"SELECT * FROM {RECORDS_TABLE} ORDER BY at DESC LIMIT 1")
    except Exception:  # noqa: BLE001 - table may not exist before any gate run
        row = None
    if row is None:
        return {"passed": False, "at": None,
                "mapping_version": MAPPING_VERSION,
                "divergences": [], "reason": "no parity gate has run"}
    gate = json.loads(row["report"])
    divergences: list[dict[str, Any]] = []
    for entry in gate.get("collections", []):
        if entry.get("ok"):
            continue
        ids = (list(entry.get("only_in_source", []))
               + list(entry.get("only_in_target", []))
               + list(entry.get("differing", [])))[:50]
        divergences.append({
            "collection": entry["collection"],
            "source_count": entry.get("source_count"),
            "target_count": entry.get("target_count"),
            "divergent_ids": ids,
            "field_mismatch_ids": [m["id"] for m in
                                   entry.get("field_mismatches", [])][:50],
        })
    return {
        "passed": bool(row["passed"]),
        "at": float(row["at"]),
        "mapping_version": row["mapping_version"],
        "source": row["source"],
        "target": row["target"],
        "divergences": divergences,
    }
