"""The document-store seam: one protocol, swappable backends (Mongo program P0).

Why this exists
---------------
Devon's persistence grew as raw SQL against one central SQLite file plus a
federation of sidecar files. The Mongo program (owner directive 2026-11-08:
"Database to mongodb and God tier") ports domain by domain onto a document
store — but no call site may ever speak to a driver directly. This module is
the seam every port goes through, in the house shape of
:func:`~nomorals.storage.analytics.open_analytics` and
:func:`~nomorals.storage.s3blob.open_blob_store`:

* :class:`DocumentStore` — the protocol: collections of JSON documents with
  ``get/put/delete/find/update_fields/inc/cas/replace/take`` over dict
  filters, store-managed expiry, and a health surface.
* :class:`SQLiteDocumentStore` — the offline rung. Wraps the existing
  :class:`~nomorals.storage.db.Database`; collections live in a
  runtime-created ``docstore_docs`` table (the same runtime-table precedent
  as the queue/FTS/vector stores — no migration is touched, and
  ``migrations.py`` stays frozen).
* :class:`MongoDocumentStore` — the primary-where-reachable rung, over
  ``pymongo`` (an optional extra, like DuckDB for analytics). One Mongo
  collection per document collection; documents are stored verbatim plus
  the managed ``_id`` / ``expires_at`` fields.
* :class:`ShadowDocumentStore` — dual-write wrapper (Pass 1 shadow mode):
  writes land on both rungs, reads come from the primary, divergences are
  recorded — never merged. Two writable primaries would fork the data;
  this wrapper has exactly one read primary by construction.

Backend selection (operator contract)
-------------------------------------
Pinned per install by config, never auto-flapped per boot:

* ``storage.backend`` — ``"sqlite"`` (default) or ``"mongo"``
  (env ``NM_DB_BACKEND``).
* ``storage.mongo_uri`` — e.g. ``mongodb://127.0.0.1:27017/?replicaSet=rs0``
  (env ``NM_MONGO_URI``). Empty means the local default
  ``mongodb://127.0.0.1:27017``.
* ``storage.mongo_db`` — Mongo database name (env ``NM_MONGO_DB``,
  default ``"nomorals"``).
* ``storage.mongo_shadow`` — ``true`` dual-writes Mongo + SQLite while
  Mongo is configured (env ``NM_MONGO_SHADOW``).

Boot behavior when Mongo is configured but unusable follows the
``open_database`` precedent — degrade loudly, never brick, never silently
split-brain: resolution falls back to the SQLite rung, the reason is
journaled to the failure ledger, and :func:`storage_backend_status` reports
the Mongo state as ``driver_missing`` / ``unreachable`` / ``broken`` —
visibly distinct from ``off`` (not configured). Profile rule: on
Termux/Android-class platforms (low-memory, ``storage_backend="sqlite"``)
Mongo is honored only when an explicit ``mongo_uri`` is configured; the
bare ``backend="mongo"`` flip stays on SQLite there, with the reason named
``profile_default``.

Expiry semantics are identical on both rungs by construction: lazy on read
(an expired document is absent) plus :meth:`DocumentStore.sweep_expired`.
Mongo TTL indexes are deliberately *not* used — they would make expiry
server-clock-driven and unobservable to the parity checker; the sweeper is
the one expiry mechanism, exactly as ``KVStore.delete_expired`` is for the
SQL rung.

Concurrency guarantees, honestly: single-document operations are atomic on
both rungs (SQLite via ``BEGIN IMMEDIATE`` transactions, Mongo via
single-document atomic operators). Multi-document transactions exist only
on the SQLite rung (and on Mongo only against a replica set, which the
protocol does not rely on); each store advertises this in ``capabilities``
instead of pretending.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..compat import load_optional
from ..core.errors import StorageError
from ..core.logging_setup import get_logger
from .failures import record_failure

__all__ = [
    "BackendHealth",
    "DocumentStore",
    "MongoDocumentStore",
    "SQLiteDocumentStore",
    "ShadowDocumentStore",
    "StoreResolution",
    "attach_document_store",
    "mask_mongo_uri",
    "open_document_store",
    "resolve_document_store",
    "storage_backend_status",
    "DEFAULT_MONGO_URI",
    "MONGO_INSTALL_HINT",
]

_log = get_logger(__name__)

DEFAULT_MONGO_URI = "mongodb://127.0.0.1:27017"

MONGO_INSTALL_HINT = (
    "pymongo is not installed. Install it with `pip install nomorals[mongo]` "
    "(or `pip install pymongo`) to enable the MongoDB storage backend; "
    "until then Devon serves the SQLite rung."
)

_COLLECTION_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*$")
_FILTER_OPS = {"$gt", "$gte", "$lt", "$lte", "$ne", "$in", "$exists"}
_RANGE_SENTINEL = "\uffff"


def _check_collection(collection: str) -> str:
    if not collection or not _COLLECTION_RE.match(collection) or collection.startswith("system."):
        raise StorageError(f"invalid document collection name: {collection!r}")
    return collection


def _is_expired(expires_at: Any, now: float) -> bool:
    return expires_at is not None and float(expires_at) <= now


def _live(doc: Mapping[str, Any] | None, now: float) -> bool:
    return doc is not None and not _is_expired(doc.get("expires_at"), now)


def _prefix_range(prefix: str) -> dict[str, Any] | None:
    """An ``_id`` range filter matching exactly the keys under ``prefix``."""
    if not prefix:
        return None
    return {"_id": {"$gte": prefix, "$lt": prefix + _RANGE_SENTINEL}}


def _matches(doc: Mapping[str, Any], filter: Mapping[str, Any] | None, now: float,
             *, include_expired: bool) -> bool:
    """In-Python filter evaluation (SQLite rung; also the semantic reference)."""
    if not include_expired and not _live(doc, now):
        return False
    if not filter:
        return True
    for key, cond in filter.items():
        value = doc.get(key)
        if isinstance(cond, Mapping):
            for op, operand in cond.items():
                if op == "$gt" and not (value is not None and value > operand):
                    return False
                if op == "$gte" and not (value is not None and value >= operand):
                    return False
                if op == "$lt" and not (value is not None and value < operand):
                    return False
                if op == "$lte" and not (value is not None and value <= operand):
                    return False
                if op == "$ne" and value == operand:
                    return False
                if op == "$in" and value not in operand:
                    return False
                if op == "$exists" and ((key in doc) != bool(operand)):
                    return False
        elif value != cond:
            return False
    return True


def mask_mongo_uri(uri: str) -> str:
    """Redact any password in a Mongo URI before it reaches a log/status."""
    if "@" not in uri:
        return uri
    scheme, _, rest = uri.partition("://")
    creds, _, host = rest.partition("@")
    if ":" in creds:
        creds = creds.split(":", 1)[0] + ":***"
    return f"{scheme}://{creds}@{host}"


@dataclass
class BackendHealth:
    """One backend's liveness, stated so OFF is never confused with BROKEN."""

    backend: str  # "sqlite" | "mongo"
    state: str  # "live" | "off" | "driver_missing" | "unreachable" | "broken"
    detail: str = ""
    checked_at: float = field(default_factory=time.time)

    @property
    def ok(self) -> bool:
        return self.state == "live"

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "state": self.state,
            "detail": self.detail,
            "checked_at": self.checked_at,
        }


class DocumentStore(Protocol):
    """Collections of JSON documents behind one backend-neutral contract.

    Documents are plain dicts. The store manages two fields: ``_id`` (the
    document id, always present on reads) and ``expires_at`` (epoch float
    or ``None`` for immortal). Callers pass expiry explicitly as an
    absolute ``expires_at`` on writes — the store never invents one.
    """

    name: str
    capabilities: dict[str, bool]

    def get(self, collection: str, doc_id: str, *,
            include_expired: bool = False) -> dict[str, Any] | None: ...

    def put(self, collection: str, doc_id: str, doc: Mapping[str, Any], *,
            expires_at: float | None = None) -> None: ...

    def update_fields(self, collection: str, doc_id: str,
                      fields: Mapping[str, Any], *, upsert: bool = False) -> bool: ...

    def inc(self, collection: str, doc_id: str, field: str,
            delta: float = 1.0) -> float: ...

    def cas(self, collection: str, doc_id: str,
            expected: Mapping[str, Any] | None,
            new_doc: Mapping[str, Any], *,
            expires_at: float | None = None) -> bool: ...

    def replace(self, collection: str, doc_id: str, doc: Mapping[str, Any], *,
                expires_at: float | None = None) -> dict[str, Any] | None: ...

    def take(self, collection: str, doc_id: str) -> dict[str, Any] | None: ...

    def delete(self, collection: str, doc_id: str) -> bool: ...

    def set_expiry(self, collection: str, doc_id: str,
                   expires_at: float | None) -> bool: ...

    def find(self, collection: str, filter: Mapping[str, Any] | None = None, *,
             include_expired: bool = False, after_id: str | None = None,
             limit: int | None = None) -> list[dict[str, Any]]: ...

    def count(self, collection: str, filter: Mapping[str, Any] | None = None, *,
              include_expired: bool = False) -> int: ...

    def delete_many(self, collection: str,
                    filter: Mapping[str, Any] | None = None) -> int: ...

    def sweep_expired(self, collection: str | None = None,
                      prefix: str = "") -> int: ...

    def collections(self) -> list[str]: ...

    def health(self) -> BackendHealth: ...

    def close(self) -> None: ...


def _clean_doc(collection: str, doc_id: str, doc: Mapping[str, Any],
               expires_at: float | None) -> dict[str, Any]:
    if "_id" in doc and doc["_id"] != doc_id:
        raise StorageError(
            f"document _id {doc['_id']!r} does not match id {doc_id!r} "
            f"for collection {collection!r}")
    out = dict(doc)
    out["_id"] = doc_id
    out["expires_at"] = expires_at
    return out


class SQLiteDocumentStore:
    """DocumentStore over the existing SQLite :class:`Database` (offline rung).

    Collections share one runtime-created table — ``docstore_docs`` with the
    document as a JSON body — created with ``CREATE TABLE IF NOT EXISTS`` at
    first use, the same runtime-table precedent as the queue and vector
    stores. Filters are evaluated against the JSON body in Python: correct
    on every field, and honest about it — this rung is the fallback, sized
    for correctness, not for analytical scans.
    """

    name = "sqlite"
    capabilities = {
        "multi_doc_transactions": True,
        "atomic_single_doc": True,
        "server_side_filters": False,
    }
    TABLE = "docstore_docs"

    def __init__(self, db: Any, *, now_fn: Callable[[], float] | None = None) -> None:
        self._db = db
        self._now = now_fn or time.time
        self._lock = threading.RLock()
        db.execute(
            f"CREATE TABLE IF NOT EXISTS {self.TABLE} ("
            "collection TEXT NOT NULL, id TEXT NOT NULL, doc TEXT NOT NULL, "
            "expires_at REAL, PRIMARY KEY (collection, id))"
        )
        db.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self.TABLE}_expiry "
            f"ON {self.TABLE} (collection, expires_at)"
        )

    # ── internals ────────────────────────────────────────────────────────────
    def _tx(self):
        tx = getattr(self._db, "transaction", None)
        if tx is None:
            from contextlib import nullcontext
            return nullcontext()
        return tx()

    def _read_row(self, collection: str, doc_id: str) -> dict[str, Any] | None:
        row = self._db.query_one(
            f"SELECT doc FROM {self.TABLE} WHERE collection = ? AND id = ?",
            (collection, doc_id),
        )
        return json.loads(row["doc"]) if row else None

    def _write_row(self, collection: str, doc: dict[str, Any]) -> None:
        self._db.execute(
            f"INSERT INTO {self.TABLE} (collection, id, doc, expires_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(collection, id) DO UPDATE SET "
            "doc = excluded.doc, expires_at = excluded.expires_at",
            (collection, doc["_id"], json.dumps(doc), doc.get("expires_at")),
        )

    # ── protocol ─────────────────────────────────────────────────────────────
    def get(self, collection, doc_id, *, include_expired=False):
        _check_collection(collection)
        doc = self._read_row(collection, doc_id)
        if doc is None or (not include_expired and not _live(doc, self._now())):
            return None
        return doc

    def put(self, collection, doc_id, doc, *, expires_at=None):
        _check_collection(collection)
        with self._tx():
            self._write_row(collection, _clean_doc(collection, doc_id, doc, expires_at))

    def update_fields(self, collection, doc_id, fields, *, upsert=False):
        _check_collection(collection)
        if "_id" in fields:
            raise StorageError("update_fields may not set _id")
        with self._tx():
            current = self._read_row(collection, doc_id)
            if not _live(current, self._now()):
                if not upsert:
                    return False
                current = {"_id": doc_id, "expires_at": None}
            current.update(fields)
            self._write_row(collection, current)
        return True

    def inc(self, collection, doc_id, field, delta=1.0):
        _check_collection(collection)
        with self._tx():
            current = self._read_row(collection, doc_id)
            if not _live(current, self._now()):
                current = {"_id": doc_id, "expires_at": None}
            value = current.get(field, 0)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise StorageError(
                    f"inc on non-numeric field {field!r} in {collection}/{doc_id}")
            new_value = float(value) + float(delta)
            current[field] = int(new_value) if new_value.is_integer() else new_value
            self._write_row(collection, current)
        return float(new_value)

    def cas(self, collection, doc_id, expected, new_doc, *, expires_at=None):
        _check_collection(collection)
        with self._tx():
            current = self._read_row(collection, doc_id)
            live = _live(current, self._now())
            if expected is None:
                if live:
                    return False
            else:
                if not live or current is None:
                    return False
                if any(current.get(k) != v for k, v in expected.items()):
                    return False
            self._write_row(collection, _clean_doc(collection, doc_id, new_doc, expires_at))
        return True

    def replace(self, collection, doc_id, doc, *, expires_at=None):
        _check_collection(collection)
        with self._tx():
            current = self._read_row(collection, doc_id)
            previous = current if _live(current, self._now()) else None
            self._write_row(collection, _clean_doc(collection, doc_id, doc, expires_at))
        return previous

    def take(self, collection, doc_id):
        _check_collection(collection)
        with self._tx():
            current = self._read_row(collection, doc_id)
            if current is None:
                return None
            self._db.execute(
                f"DELETE FROM {self.TABLE} WHERE collection = ? AND id = ?",
                (collection, doc_id),
            )
        return current if _live(current, self._now()) else None

    def delete(self, collection, doc_id):
        _check_collection(collection)
        changed = self._db.delete(
            self.TABLE, "collection = ? AND id = ?", (collection, doc_id))
        return changed > 0

    def set_expiry(self, collection, doc_id, expires_at):
        _check_collection(collection)
        with self._tx():
            current = self._read_row(collection, doc_id)
            if current is None:
                return False
            current["expires_at"] = expires_at
            self._write_row(collection, current)
        return True

    def find(self, collection, filter=None, *, include_expired=False,
             after_id=None, limit=None):
        _check_collection(collection)
        rows = self._db.query(
            f"SELECT doc FROM {self.TABLE} WHERE collection = ? ORDER BY id",
            (collection,),
        )
        now = self._now()
        out: list[dict[str, Any]] = []
        for row in rows:
            doc = json.loads(row["doc"])
            if after_id is not None and doc["_id"] <= after_id:
                continue
            if _matches(doc, filter, now, include_expired=include_expired):
                out.append(doc)
                if limit is not None and len(out) >= limit:
                    break
        return out

    def count(self, collection, filter=None, *, include_expired=False):
        return len(self.find(collection, filter, include_expired=include_expired))

    def delete_many(self, collection, filter=None):
        _check_collection(collection)
        now = self._now()
        with self._tx():
            rows = self._db.query(
                f"SELECT id, doc FROM {self.TABLE} WHERE collection = ?",
                (collection,),
            )
            victims = [
                r["id"] for r in rows
                if _matches(json.loads(r["doc"]), filter, now, include_expired=True)
            ]
            for doc_id in victims:
                self._db.execute(
                    f"DELETE FROM {self.TABLE} WHERE collection = ? AND id = ?",
                    (collection, doc_id),
                )
        return len(victims)

    def sweep_expired(self, collection=None, prefix=""):
        now = self._now()
        if collection is not None:
            _check_collection(collection)
            where = "collection = ? AND expires_at IS NOT NULL AND expires_at <= ?"
            args: tuple = (collection, now)
        else:
            where = "expires_at IS NOT NULL AND expires_at <= ?"
            args = (now,)
        if prefix:
            where += " AND id >= ? AND id < ?"
            args = (*args, prefix, prefix + _RANGE_SENTINEL)
        cursor = self._db.execute(f"DELETE FROM {self.TABLE} WHERE {where}", args)
        return cursor.rowcount or 0

    def collections(self):
        rows = self._db.query(
            f"SELECT DISTINCT collection FROM {self.TABLE} ORDER BY collection")
        return [r["collection"] for r in rows]

    def health(self):
        try:
            self._db.scalar(f"SELECT COUNT(*) FROM {self.TABLE}", default=0)
            return BackendHealth("sqlite", "live", "docstore_docs reachable")
        except Exception as exc:  # noqa: BLE001 - health must never raise
            return BackendHealth("sqlite", "broken", f"{type(exc).__name__}: {exc}")

    def close(self):
        pass  # The Database owns its connections; nothing store-local to close.


class MongoDocumentStore:
    """DocumentStore over a real MongoDB server via ``pymongo`` (optional extra).

    The client is created lazily and pinged on :meth:`health` / first use,
    with a short server-selection timeout so a dead endpoint degrades in
    seconds, never hangs boot. Every driver error is re-raised as
    :class:`StorageError` — callers never see pymongo types.
    """

    name = "mongo"
    capabilities = {
        "multi_doc_transactions": False,
        "atomic_single_doc": True,
        "server_side_filters": True,
    }

    def __init__(self, uri: str = DEFAULT_MONGO_URI, database: str = "nomorals", *,
                 now_fn: Callable[[], float] | None = None,
                 server_selection_timeout_ms: int = 2000,
                 client: Any = None) -> None:
        self._uri = uri or DEFAULT_MONGO_URI
        self._database_name = database or "nomorals"
        self._now = now_fn or time.time
        self._timeout_ms = server_selection_timeout_ms
        self._client = client
        self._lock = threading.RLock()

    # ── driver plumbing ──────────────────────────────────────────────────────
    def _mongo(self):
        module = load_optional("pymongo")
        if module is None:
            raise StorageError(MONGO_INSTALL_HINT)
        return module

    def _get_client(self):
        if self._client is None:
            pymongo = self._mongo()
            self._client = pymongo.MongoClient(
                self._uri,
                serverSelectionTimeoutMS=self._timeout_ms,
                connectTimeoutMS=self._timeout_ms,
            )
        return self._client

    def _coll(self, collection: str):
        _check_collection(collection)
        return self._get_client()[self._database_name][collection]

    def _run(self, fn: Callable[[], Any]) -> Any:
        pymongo = self._mongo()
        try:
            return fn()
        except StorageError:
            raise
        except pymongo.errors.PyMongoError as exc:
            raise StorageError(f"mongo operation failed: {exc}") from exc

    def _live_filter(self) -> dict[str, Any]:
        return {"$or": [{"expires_at": None}, {"expires_at": {"$gt": self._now()}}]}

    def _translate(self, filter: Mapping[str, Any] | None, *,
                     include_expired: bool) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, cond in (filter or {}).items():
            if isinstance(cond, Mapping):
                unknown = set(cond) - _FILTER_OPS
                if unknown:
                    raise StorageError(f"unsupported filter operators: {sorted(unknown)}")
                out[key] = dict(cond)
            else:
                out[key] = cond
        if not include_expired:
            out = {"$and": [out, self._live_filter()]} if out else self._live_filter()
        return out

    # ── protocol ─────────────────────────────────────────────────────────────
    def get(self, collection, doc_id, *, include_expired=False):
        filt: dict[str, Any] = {"_id": doc_id}
        if not include_expired:
            filt = {"$and": [filt, self._live_filter()]}
        doc = self._run(lambda: self._coll(collection).find_one(filt))
        return dict(doc) if doc else None

    def put(self, collection, doc_id, doc, *, expires_at=None):
        full = _clean_doc(collection, doc_id, doc, expires_at)
        self._run(lambda: self._coll(collection).replace_one(
            {"_id": doc_id}, full, upsert=True))

    def update_fields(self, collection, doc_id, fields, *, upsert=False):
        if "_id" in fields:
            raise StorageError("update_fields may not set _id")
        coll = self._coll(collection)
        filt = {"$and": [{"_id": doc_id}, self._live_filter()]}
        result = self._run(lambda: coll.update_one(filt, {"$set": dict(fields)}))
        if result.matched_count:
            return True
        if not upsert:
            return False
        full = {"_id": doc_id, "expires_at": None, **dict(fields)}
        self._run(lambda: coll.replace_one({"_id": doc_id}, full, upsert=True))
        return True

    def inc(self, collection, doc_id, field, delta=1.0):
        coll = self._coll(collection)
        filt = {"$and": [{"_id": doc_id}, self._live_filter()]}
        doc = self._run(lambda: coll.find_one_and_update(
            filt, {"$inc": {field: float(delta)}},
            return_document=self._mongo().ReturnDocument.AFTER))
        if doc is not None:
            return float(doc[field])
        # Absent or expired: start fresh (an expired doc must not resurrect
        # with its old value or expiry).
        fresh = {"_id": doc_id, "expires_at": None, field: float(delta)}
        self._run(lambda: coll.replace_one({"_id": doc_id}, fresh, upsert=True))
        return float(delta)

    def cas(self, collection, doc_id, expected, new_doc, *, expires_at=None):
        coll = self._coll(collection)
        full = _clean_doc(collection, doc_id, new_doc, expires_at)
        pymongo = self._mongo()
        if expected is None:
            try:
                coll.insert_one(dict(full))
                return True
            except pymongo.errors.DuplicateKeyError:
                pass
            except pymongo.errors.PyMongoError as exc:
                raise StorageError(f"mongo operation failed: {exc}") from exc
            # A doc exists: the swap is allowed only when it is expired, and
            # only against the exact expiry we observed (a racing writer
            # that refreshed it wins, and this CAS honestly fails).
            current = coll.find_one({"_id": doc_id})
            if current is None or not _is_expired(current.get("expires_at"), self._now()):
                return False
            result = self._run(lambda: coll.replace_one(
                {"_id": doc_id, "expires_at": current.get("expires_at")}, full))
            return result.matched_count > 0
        filt: dict[str, Any] = {"_id": doc_id, **dict(expected)}
        filt = {"$and": [filt, self._live_filter()]}
        result = self._run(lambda: coll.replace_one(filt, full))
        return result.matched_count > 0

    def replace(self, collection, doc_id, doc, *, expires_at=None):
        coll = self._coll(collection)
        full = _clean_doc(collection, doc_id, doc, expires_at)
        pymongo = self._mongo()
        filt = {"$and": [{"_id": doc_id}, self._live_filter()]}
        previous = self._run(lambda: coll.find_one_and_replace(
            filt, full, return_document=pymongo.ReturnDocument.BEFORE))
        if previous is not None:
            return dict(previous)
        # Absent or expired: overwrite whatever is there; report no live
        # predecessor either way.
        self._run(lambda: coll.replace_one({"_id": doc_id}, full, upsert=True))
        return None

    def take(self, collection, doc_id):
        coll = self._coll(collection)
        doc = self._run(lambda: coll.find_one_and_delete({"_id": doc_id}))
        if doc is None or not _live(doc, self._now()):
            return None
        return dict(doc)

    def delete(self, collection, doc_id):
        result = self._run(lambda: self._coll(collection).delete_one({"_id": doc_id}))
        return result.deleted_count > 0

    def set_expiry(self, collection, doc_id, expires_at):
        result = self._run(lambda: self._coll(collection).update_one(
            {"_id": doc_id}, {"$set": {"expires_at": expires_at}}))
        return result.matched_count > 0

    def find(self, collection, filter=None, *, include_expired=False,
             after_id=None, limit=None):
        filt = self._translate(filter, include_expired=include_expired)
        if after_id is not None:
            range_cond = {"_id": {"$gt": after_id}}
            filt = {"$and": [filt, range_cond]} if filt else range_cond
        cursor = self._coll(collection).find(filt).sort("_id", 1)
        if limit is not None:
            cursor = cursor.limit(limit)
        return [dict(d) for d in self._run(lambda: list(cursor))]

    def count(self, collection, filter=None, *, include_expired=False):
        filt = self._translate(filter, include_expired=include_expired)
        return int(self._run(lambda: self._coll(collection).count_documents(filt)))

    def delete_many(self, collection, filter=None):
        filt = self._translate(filter, include_expired=True)
        result = self._run(lambda: self._coll(collection).delete_many(filt))
        return int(result.deleted_count)

    def sweep_expired(self, collection=None, prefix=""):
        filt: dict[str, Any] = {
            "expires_at": {"$ne": None, "$lte": self._now()}}
        range_cond = _prefix_range(prefix)
        if range_cond:
            filt.update(range_cond)
        names = [collection] if collection else self.collections()
        total = 0
        for name in names:
            result = self._run(lambda n=name: self._coll(n).delete_many(filt))
            total += int(result.deleted_count)
        return total

    def collections(self):
        return sorted(self._run(
            lambda: self._get_client()[self._database_name].list_collection_names()))

    def health(self):
        try:
            self._get_client().admin.command("ping")
            return BackendHealth(
                "mongo", "live",
                f"reachable at {mask_mongo_uri(self._uri)} db={self._database_name}")
        except StorageError as exc:
            return BackendHealth("mongo", "driver_missing", str(exc))
        except Exception as exc:  # noqa: BLE001 - health must never raise
            return BackendHealth(
                "mongo", "unreachable",
                f"{mask_mongo_uri(self._uri)}: {type(exc).__name__}: {exc}")

    def close(self):
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None


class ShadowDocumentStore:
    """Dual-write wrapper: one read primary, both rungs written (Pass 1).

    Writes apply to the primary first (its result is authoritative), then
    to the secondary; a secondary failure is journaled as a divergence and
    never raised — the primary already holds the truth. Reads come from the
    primary only. :meth:`parity` compares the two rungs on demand so the
    shadow can be *proven*, not assumed (see
    :mod:`nomorals.storage.parity`).
    """

    name = "shadow"
    capabilities = {
        "multi_doc_transactions": False,
        "atomic_single_doc": True,
        "server_side_filters": False,
    }

    def __init__(self, primary: DocumentStore, secondary: DocumentStore) -> None:
        self.primary = primary
        self.secondary = secondary
        self._divergences = 0
        self._lock = threading.RLock()

    @property
    def divergences(self) -> int:
        return self._divergences

    def _shadow(self, op: str, fn: Callable[[], Any]) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - the shadow must not break the write
            with self._lock:
                self._divergences += 1
            record_failure("storage.docstore.shadow", f"{op}: {exc}")
            _log.warning("shadow write failed (%s): %s", op, exc)

    # ── writes: primary authoritative, secondary shadowed ───────────────────
    def put(self, collection, doc_id, doc, *, expires_at=None):
        self.primary.put(collection, doc_id, doc, expires_at=expires_at)
        self._shadow("put", lambda: self.secondary.put(
            collection, doc_id, doc, expires_at=expires_at))

    def delete(self, collection, doc_id):
        result = self.primary.delete(collection, doc_id)
        self._shadow("delete", lambda: self.secondary.delete(collection, doc_id))
        return result

    def update_fields(self, collection, doc_id, fields, *, upsert=False):
        result = self.primary.update_fields(collection, doc_id, fields, upsert=upsert)
        self._shadow("update_fields", lambda: self.secondary.update_fields(
            collection, doc_id, fields, upsert=upsert))
        return result

    def inc(self, collection, doc_id, field, delta=1.0):
        value = self.primary.inc(collection, doc_id, field, delta)
        self._shadow("inc", lambda: self.secondary.inc(collection, doc_id, field, delta))
        return value

    def cas(self, collection, doc_id, expected, new_doc, *, expires_at=None):
        result = self.primary.cas(collection, doc_id, expected, new_doc,
                                  expires_at=expires_at)
        self._shadow("cas", lambda: self.secondary.cas(
            collection, doc_id, expected, new_doc, expires_at=expires_at))
        return result

    def replace(self, collection, doc_id, doc, *, expires_at=None):
        previous = self.primary.replace(collection, doc_id, doc, expires_at=expires_at)
        self._shadow("replace", lambda: self.secondary.replace(
            collection, doc_id, doc, expires_at=expires_at))
        return previous

    def take(self, collection, doc_id):
        doc = self.primary.take(collection, doc_id)
        self._shadow("take", lambda: self.secondary.take(collection, doc_id))
        return doc

    def set_expiry(self, collection, doc_id, expires_at):
        result = self.primary.set_expiry(collection, doc_id, expires_at)
        self._shadow("set_expiry", lambda: self.secondary.set_expiry(
            collection, doc_id, expires_at))
        return result

    def delete_many(self, collection, filter=None):
        result = self.primary.delete_many(collection, filter)
        self._shadow("delete_many", lambda: self.secondary.delete_many(collection, filter))
        return result

    def sweep_expired(self, collection=None, prefix=""):
        result = self.primary.sweep_expired(collection, prefix)
        self._shadow("sweep", lambda: self.secondary.sweep_expired(collection, prefix))
        return result

    # ── reads: primary only ──────────────────────────────────────────────────
    def get(self, collection, doc_id, *, include_expired=False):
        return self.primary.get(collection, doc_id, include_expired=include_expired)

    def find(self, collection, filter=None, *, include_expired=False,
             after_id=None, limit=None):
        return self.primary.find(collection, filter, include_expired=include_expired,
                                 after_id=after_id, limit=limit)

    def count(self, collection, filter=None, *, include_expired=False):
        return self.primary.count(collection, filter, include_expired=include_expired)

    def collections(self):
        return self.primary.collections()

    def health(self):
        health = self.primary.health()
        return BackendHealth(
            health.backend, health.state,
            f"primary={self.primary.name} secondary={self.secondary.name} "
            f"divergences={self._divergences}; {health.detail}")

    def parity(self, collection: str = "kv_store"):
        from .parity import check_document_parity
        return check_document_parity(self.primary, self.secondary, collection)

    def close(self):
        self.primary.close()
        self.secondary.close()


@dataclass
class StoreResolution:
    """The outcome of backend selection: what was asked, what is serving."""

    configured: str  # "sqlite" | "mongo" (after the profile rule)
    active: str  # the backend actually serving reads/writes
    store: Any  # a DocumentStore for `active` (Shadow wrapper when shadowing)
    mongo_health: BackendHealth
    sqlite_health: BackendHealth
    degraded: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "configured_backend": self.configured,
            "active_backend": self.active,
            "degraded": self.degraded,
            "reason": self.reason,
            "mongo": self.mongo_health.to_dict(),
            "sqlite": self.sqlite_health.to_dict(),
        }


def _platform_caps(platform: Any = None) -> Any:
    if platform is not None:
        return getattr(platform, "capabilities", platform)
    try:
        from ..core.platform import detect_platform
        return detect_platform().capabilities
    except Exception:  # noqa: BLE001 - resolution must never brick on detection
        return None


def resolve_document_store(settings: Any, db: Any = None, *,
                           platform: Any = None) -> StoreResolution:
    """Select the document-store backend for ``settings`` (never raises).

    ``db`` is the SQLite :class:`Database` the SQLite rung wraps; it is
    required for the SQLite implementation and for a degraded fallback.
    When Mongo is configured and healthy, the returned store serves Mongo
    and ``db`` is untouched by the store itself.
    """
    storage = settings.storage
    configured = str(getattr(storage, "backend", "sqlite") or "sqlite").lower()
    if configured not in {"sqlite", "mongo"}:
        configured = "sqlite"
    uri = str(getattr(storage, "mongo_uri", "") or "")
    reason = ""

    # Profile rule: mobile-class platforms (Termux/Android) stay on SQLite
    # unless an explicit Mongo URI is configured — a bare backend flip on a
    # phone must not strand its store on an endpoint it cannot host/reach.
    if configured == "mongo" and not uri:
        caps = _platform_caps(platform)
        if caps is not None and getattr(caps, "low_memory", False) \
                and getattr(caps, "storage_backend", "sqlite") == "sqlite":
            configured = "sqlite"
            reason = "profile_default: mobile-class profile keeps sqlite without an explicit mongo_uri"

    sqlite_health = BackendHealth("sqlite", "off", "no database handle provided")
    sqlite_store: SQLiteDocumentStore | None = None
    if db is not None:
        try:
            sqlite_store = SQLiteDocumentStore(db)
            sqlite_health = sqlite_store.health()
        except Exception as exc:  # noqa: BLE001 - resolution reports, never raises
            sqlite_health = BackendHealth("sqlite", "broken", f"{type(exc).__name__}: {exc}")

    if configured == "sqlite":
        return StoreResolution(
            configured="sqlite", active="sqlite", store=sqlite_store,
            mongo_health=BackendHealth("mongo", "off", "backend is sqlite"),
            sqlite_health=sqlite_health, reason=reason)

    mongo = MongoDocumentStore(
        uri or DEFAULT_MONGO_URI,
        str(getattr(storage, "mongo_db", "nomorals") or "nomorals"),
    )
    mongo_health = mongo.health()
    if mongo_health.ok:
        store: Any = mongo
        if getattr(storage, "mongo_shadow", False) and sqlite_store is not None:
            store = ShadowDocumentStore(mongo, sqlite_store)
        return StoreResolution(
            configured="mongo", active="mongo", store=store,
            mongo_health=mongo_health, sqlite_health=sqlite_health, reason=reason)

    # Configured-but-unusable Mongo: degrade to the SQLite rung, loudly.
    degraded_reason = (
        f"mongo configured but {mongo_health.state}: {mongo_health.detail}; "
        "serving the sqlite rung")
    record_failure("storage.docstore.resolve", degraded_reason)
    _log.warning("document store degraded: %s", degraded_reason)
    return StoreResolution(
        configured="mongo", active="sqlite", store=sqlite_store,
        mongo_health=mongo_health, sqlite_health=sqlite_health,
        degraded=True, reason=degraded_reason)


def open_document_store(settings: Any, db: Any = None, *,
                        platform: Any = None) -> DocumentStore:
    """Return the active :class:`DocumentStore` for ``settings``.

    Modeled on :func:`open_analytics`: the caller gets the best configured
    backend and never handles driver errors itself. Raises
    :class:`StorageError` only when no rung can serve (no ``db`` for the
    SQLite rung and Mongo unusable).
    """
    resolution = resolve_document_store(settings, db, platform=platform)
    if resolution.store is None:
        raise StorageError(
            "no document store available: sqlite rung needs a Database handle "
            f"and mongo is {resolution.mongo_health.state} "
            f"({resolution.mongo_health.detail})")
    return resolution.store


def attach_document_store(db: Any, settings: Any, *,
                          platform: Any = None) -> StoreResolution:
    """Resolve the backend at boot and attach it to the central ``Database``.

    Sets ``db.store_resolution`` always (observability) and
    ``db.document_store`` only when a non-SQLite rung is active — KVStore
    picks the attached store up from there, so its existing consumers route
    to Mongo with no call-site changes, while the SQLite default keeps the
    native ``kv_store`` table path byte-for-byte.
    """
    resolution = resolve_document_store(settings, db, platform=platform)
    db.store_resolution = resolution
    db.document_store = (
        resolution.store if resolution.active != "sqlite" else None)
    if resolution.degraded:
        _log.warning("storage backend degraded at boot: %s", resolution.reason)
    return resolution


def storage_backend_status(settings: Any, db: Any = None, *,
                           platform: Any = None) -> dict[str, Any]:
    """JSON-able backend health surface (doctor/status consumers).

    ``configured`` vs ``active`` vs per-backend ``state`` keeps the four
    situations an operator must tell apart — sqlite by choice, mongo live,
    mongo off, mongo broken — mutually unmistakable.
    """
    resolution = resolve_document_store(settings, db, platform=platform)
    caps = _platform_caps(platform)
    # The pinned set (Mongo P4): domains that live on the SQLite rung
    # BY DESIGN whatever the active backend — a queryable fact on the
    # status surface, not tribal knowledge. Lazy import: carry imports
    # this module, so the registry is read at call time, never at
    # import time.
    from .carry import pinned_domains

    return {
        **resolution.to_dict(),
        "profile_storage_default": getattr(caps, "storage_backend", None),
        "mongo_uri": mask_mongo_uri(
            str(getattr(settings.storage, "mongo_uri", "") or DEFAULT_MONGO_URI)),
        "mongo_shadow": bool(getattr(settings.storage, "mongo_shadow", False)),
        "pinned_domains": pinned_domains(),
    }
