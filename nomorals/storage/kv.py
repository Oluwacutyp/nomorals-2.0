"""First-class key/value store over the ``kv_store`` table — or Mongo.

Why this exists
---------------
Thirty-plus modules hand-roll their own ``_kv_get``/``_kv_set`` helpers with
inconsistent upsert shapes (some omit ``kind``, some use ``INSERT`` where
``INSERT OR REPLACE`` was meant, none support expiry). This module is the one
canonical layer:

* **Namespaces** — ``KVStore(db, namespace="arena")`` scopes every key to
  ``arena:<key>``. The empty namespace is the global scope and keeps every
  existing raw-SQL key working unchanged.
* **TTL / expiry** — ``set(..., ttl=seconds)`` stores ``expires_at`` (migration
  86). Reads filter expired rows lazily; :meth:`delete_expired` is the sweeper
  a scheduler tick calls.
* **Types** — JSON encoding for arbitrary values; typed getters for the
  scalar hot path (str/int/float/bool) with no JSON overhead on read.
* **Atomicity** — compare-and-set, counters, and multi-set run inside the
  database transaction, so concurrent workers can't interleave.
* **Backwards compatible** — raw SQL written by older callers keeps working;
  ``expires_at`` is NULLABLE so rows written before migration 86 read as
  immortal.

Backends (Mongo program P1)
---------------------------
Every operation routes through one of two persistence backends; the public
API above is identical on both:

* **Native SQL** (default) — the statements this module has always run
  against the ``kv_store`` table of the given :class:`Database`. This is
  the offline rung and the behavior every existing caller and test pins.
* **DocumentStore seam** — when constructed with ``store=`` (a
  :class:`~nomorals.storage.docstore.DocumentStore`), or when the given
  database carries an attached store from
  :func:`~nomorals.storage.docstore.attach_document_store`, entries live
  in the seam's ``kv_store`` collection instead — MongoDB when configured,
  with no call-site changes. :meth:`KVStore.from_settings` performs the
  config-driven resolution explicitly.

The seam backend's atomicity is per-entry: CAS/replace/take/inc are single
-document atomic operations there, and composed operations (hash writes,
rename) hold a process-local lock rather than a cross-process transaction.
That difference is stated, not hidden — see ``DocumentStore.capabilities``.

The ``Database`` drives the transactions here; this class never opens its own
connections.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import nullcontext
from typing import Any, Callable, Iterable, Mapping, NamedTuple

from ..core.errors import StorageError
from ..core.logging_setup import get_logger
from ..core.style import active_theme, header, kv_lines, styled_table

__all__ = ["KVStore", "now"]

_log = get_logger(__name__)

#: Separator between namespace and key. Existing call sites already use the
#: ``prefix:key`` convention (``opsselfheal:``, ``opsalert:``), so namespaced
#: keys stay greppable and consistent.
_SEP = ":"

#: Hard cap on a scan page so a runaway ``keys()`` can't OOM a phone.
_MAX_SCAN = 100_000

#: The DocumentStore collection KV entries live in on the seam backend.
_DOC_COLLECTION = "kv_store"

#: Sentinel distinguishing "no guard" from a guard on the raw value None-ish.
_ABSENT = object()


def now() -> float:
    return time.time()


class _Entry(NamedTuple):
    raw: str
    kind: str
    expires_at: float | None
    updated_at: float


def _decode(raw: str) -> Any:
    """JSON-decode a stored raw value; legacy plain strings pass through."""
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def _live_clause(alias: str = "") -> str:
    """SQL fragment excluding expired rows (lazy expiry)."""
    col = f"{alias}expires_at" if not alias else f"{alias}.expires_at"
    return f"({col} IS NULL OR {col} > ?)"


_UPSERT_SQL = (
    "INSERT INTO kv_store (key, value, kind, updated_at, expires_at) "
    "VALUES (?, ?, ?, ?, ?) "
    "ON CONFLICT(key) DO UPDATE SET "
    "value = excluded.value, kind = excluded.kind, "
    "updated_at = excluded.updated_at, expires_at = excluded.expires_at"
)


class _SqlKvBackend:
    """The native rung: KVStore's original SQL against ``kv_store``."""

    def __init__(self, db: Any) -> None:
        self._db = db

    def transaction(self):
        tx = getattr(self._db, "transaction", None)
        return tx() if tx is not None else nullcontext()

    def get_entry(self, full: str, now: float, *,
                  include_expired: bool = False) -> _Entry | None:
        if include_expired:
            row = self._db.query_one(
                "SELECT value, kind, updated_at, expires_at FROM kv_store "
                "WHERE key = ?", (full,))
        else:
            row = self._db.query_one(
                "SELECT value, kind, updated_at, expires_at FROM kv_store "
                f"WHERE key = ? AND {_live_clause()}", (full, now))
        if row is None:
            return None
        return _Entry(row["value"], row["kind"], row["expires_at"], row["updated_at"])

    def get_entries(self, fulls: list[str], now: float) -> dict[str, _Entry]:
        if not fulls:
            return {}
        placeholders = ", ".join("?" for _ in fulls)
        rows = self._db.query(
            "SELECT key, value, kind, updated_at, expires_at FROM kv_store "
            f"WHERE key IN ({placeholders}) AND {_live_clause()}",
            (*fulls, now),
        )
        return {
            r["key"]: _Entry(r["value"], r["kind"], r["expires_at"], r["updated_at"])
            for r in rows
        }

    def put_entry(self, full: str, entry: _Entry) -> None:
        self._db.execute(
            _UPSERT_SQL,
            (full, entry.raw, entry.kind, entry.updated_at, entry.expires_at),
        )

    def put_entries(self, rows: list[tuple[str, _Entry]]) -> None:
        self._db.executemany(
            _UPSERT_SQL,
            [(full, e.raw, e.kind, e.updated_at, e.expires_at) for full, e in rows],
        )

    def delete_entry(self, full: str) -> bool:
        return self._db.delete("kv_store", "key = ?", (full,)) > 0

    def delete_prefix(self, scoped: str) -> int:
        return self._db.delete("kv_store", "key LIKE ? ESCAPE '\\'", (scoped + "%",))

    def clear_all(self) -> int:
        return self._db.execute("DELETE FROM kv_store").rowcount

    def sweep(self, scoped: str, now: float) -> int:
        if scoped:
            cursor = self._db.execute(
                "DELETE FROM kv_store WHERE key LIKE ? ESCAPE '\\' "
                "AND expires_at IS NOT NULL AND expires_at <= ?",
                (scoped + "%", now),
            )
        else:
            cursor = self._db.execute(
                "DELETE FROM kv_store "
                "WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            )
        return cursor.rowcount

    def set_expiry(self, full: str, expires_at: float | None, now: float) -> bool:
        changed = self._db.execute(
            "UPDATE kv_store SET expires_at = ?, updated_at = ? WHERE key = ?",
            (expires_at, now, full),
        ).rowcount
        return changed > 0

    def cas_entry(self, full: str, guard: Any, entry: _Entry, now: float) -> bool:
        # The caller already validated the decoded value inside the same
        # transaction; the guard re-checks the raw bytes so a racing write
        # between read and write fails the swap instead of being clobbered.
        current = self.get_entry(full, now)
        if guard is _ABSENT:
            if current is not None:
                return False
        elif current is None or current.raw != guard:
            return False
        self.put_entry(full, entry)
        return True

    def replace_entry(self, full: str, entry: _Entry, now: float) -> _Entry | None:
        old = self.get_entry(full, now)
        self.put_entry(full, entry)
        return old

    def take_entry(self, full: str, now: float) -> _Entry | None:
        old = self.get_entry(full, now)
        if old is not None:
            self.delete_entry(full)
        return old

    def incr_entry(self, full: str, delta: float, now: float) -> float:
        current_entry = self.get_entry(full, now)
        current = 0.0
        if current_entry is not None:
            decoded = _decode(current_entry.raw)
            try:
                current = float(decoded)
            except (TypeError, ValueError):
                current = 0.0
        new_value = current + delta
        stored: Any = int(new_value) if float(new_value).is_integer() else new_value
        self._db.execute(
            "INSERT INTO kv_store (key, value, kind, updated_at) "
            "VALUES (?, ?, 'json', ?) "
            "ON CONFLICT(key) DO UPDATE SET "
            "value = excluded.value, kind = excluded.kind, "
            "updated_at = excluded.updated_at, expires_at = NULL",
            (full, json.dumps(stored), now),
        )
        return new_value

    def list_entries(self, scoped: str, now: float, *, after: str | None = None,
                     limit: int | None = None,
                     include_expired: bool = False) -> list[tuple[str, _Entry]]:
        sql = ("SELECT key, value, kind, updated_at, expires_at FROM kv_store "
               "WHERE key LIKE ? ESCAPE '\\'")
        params: list[Any] = [scoped + "%"]
        if after is not None:
            sql += " AND key > ?"
            params.append(after)
        if not include_expired:
            sql += f" AND {_live_clause()}"
            params.append(now)
        sql += " ORDER BY key"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._db.query(sql, tuple(params))
        return [
            (r["key"], _Entry(r["value"], r["kind"], r["expires_at"], r["updated_at"]))
            for r in rows
        ]

    def count_prefix(self, scoped: str, now: float) -> int:
        return int(
            self._db.scalar(
                "SELECT COUNT(*) FROM kv_store "
                f"WHERE key LIKE ? ESCAPE '\\' AND {_live_clause()}",
                (scoped + "%", now),
                default=0,
            )
        )

    def namespaces(self, now: float) -> list[str]:
        rows = self._db.query(
            f"SELECT DISTINCT substr(key, 1, instr(key, '{_SEP}') - 1) AS ns "
            f"FROM kv_store WHERE instr(key, '{_SEP}') > 0 "
            f"AND {_live_clause()} ORDER BY ns",
            (now,),
        )
        return [r["ns"] for r in rows if r["ns"]]

    def stats(self, scoped: str) -> tuple[int, int, Any, Any]:
        if scoped:
            where = "WHERE key LIKE ? ESCAPE '\\'"
            args: tuple = (scoped + "%",)
        else:
            where = ""
            args = ()
        row = self._db.query_one(
            "SELECT COUNT(*) AS n, "
            "SUM(CASE WHEN expires_at IS NOT NULL THEN 1 ELSE 0 END) AS with_ttl, "
            "MIN(updated_at) AS oldest, MAX(updated_at) AS newest "
            f"FROM kv_store {where}",
            args,
        ) or {}
        return (int(row.get("n") or 0), int(row.get("with_ttl") or 0),
                row.get("oldest"), row.get("newest"))

    def memory(self, scoped: str) -> tuple[int, int]:
        if scoped:
            where = "WHERE key LIKE ? ESCAPE '\\'"
            args: tuple = (scoped + "%",)
        else:
            where = ""
            args = ()
        row = self._db.query_one(
            "SELECT COUNT(*) AS n, COALESCE(SUM(LENGTH(key) + LENGTH(value)), 0) AS bytes "
            f"FROM kv_store {where}",
            args,
        ) or {}
        return int(row.get("n") or 0), int(row.get("bytes") or 0)


class _DocKvBackend:
    """KV over a :class:`DocumentStore` (the Mongo seam rung).

    Entries are documents in the ``kv_store`` collection:
    ``{_id, value, kind, updated_at}`` plus the store-managed
    ``expires_at``. Single-document protocol operations carry the atomic
    operations (CAS, replace, take, inc-with-CAS-retry); composed
    operations serialize on a process-local lock — the seam advertises
    ``multi_doc_transactions=False`` for Mongo rather than hiding that.
    """

    def __init__(self, store: Any, now_fn: Callable[[], float]) -> None:
        self._store = store
        self._now = now_fn
        self._lock = threading.RLock()

    def transaction(self):
        return _LockContext(self._lock)

    @staticmethod
    def _range(prefix: str) -> dict[str, Any] | None:
        if not prefix:
            return None
        return {"_id": {"$gte": prefix, "$lt": prefix + "\uffff"}}

    @staticmethod
    def _to_entry(doc: Mapping[str, Any] | None) -> _Entry | None:
        if doc is None:
            return None
        return _Entry(doc["value"], doc.get("kind", "text"),
                      doc.get("expires_at"), float(doc.get("updated_at", 0.0)))

    def get_entry(self, full: str, now: float, *,
                  include_expired: bool = False) -> _Entry | None:
        return self._to_entry(
            self._store.get(_DOC_COLLECTION, full, include_expired=include_expired))

    def get_entries(self, fulls: list[str], now: float) -> dict[str, _Entry]:
        out: dict[str, _Entry] = {}
        for full in fulls:
            entry = self.get_entry(full, now)
            if entry is not None:
                out[full] = entry
        return out

    def put_entry(self, full: str, entry: _Entry) -> None:
        self._store.put(
            _DOC_COLLECTION, full,
            {"value": entry.raw, "kind": entry.kind, "updated_at": entry.updated_at},
            expires_at=entry.expires_at,
        )

    def put_entries(self, rows: list[tuple[str, _Entry]]) -> None:
        for full, entry in rows:
            self.put_entry(full, entry)

    def delete_entry(self, full: str) -> bool:
        return self._store.delete(_DOC_COLLECTION, full)

    def delete_prefix(self, scoped: str) -> int:
        return self._store.delete_many(_DOC_COLLECTION, self._range(scoped))

    def clear_all(self) -> int:
        return self._store.delete_many(_DOC_COLLECTION, None)

    def sweep(self, scoped: str, now: float) -> int:
        return self._store.sweep_expired(_DOC_COLLECTION, prefix=scoped)

    def set_expiry(self, full: str, expires_at: float | None, now: float) -> bool:
        doc = self._store.get(_DOC_COLLECTION, full, include_expired=True)
        if doc is None:
            return False
        doc = {k: v for k, v in doc.items() if k not in {"_id", "expires_at"}}
        doc["updated_at"] = now
        self._store.put(_DOC_COLLECTION, full, doc, expires_at=expires_at)
        return True

    def cas_entry(self, full: str, guard: Any, entry: _Entry, now: float) -> bool:
        doc = {"value": entry.raw, "kind": entry.kind,
               "updated_at": entry.updated_at}
        if guard is _ABSENT:
            return self._store.cas(
                _DOC_COLLECTION, full, None, doc, expires_at=entry.expires_at)
        return self._store.cas(
            _DOC_COLLECTION, full, {"value": guard}, doc,
            expires_at=entry.expires_at)

    def replace_entry(self, full: str, entry: _Entry, now: float) -> _Entry | None:
        previous = self._store.replace(
            _DOC_COLLECTION, full,
            {"value": entry.raw, "kind": entry.kind,
             "updated_at": entry.updated_at},
            expires_at=entry.expires_at,
        )
        return self._to_entry(previous)

    def take_entry(self, full: str, now: float) -> _Entry | None:
        return self._to_entry(self._store.take(_DOC_COLLECTION, full))

    def incr_entry(self, full: str, delta: float, now: float) -> float:
        # Optimistic CAS loop: read, compute, swap guarded on the exact raw
        # value read. Bounded retries — contention beyond that is an error
        # the caller should see, not a silent wrong total.
        for _ in range(16):
            current_entry = self.get_entry(full, now)
            current = 0.0
            if current_entry is not None:
                decoded = _decode(current_entry.raw)
                try:
                    current = float(decoded)
                except (TypeError, ValueError):
                    current = 0.0
            new_value = current + delta
            stored: Any = int(new_value) if float(new_value).is_integer() else new_value
            entry = _Entry(json.dumps(stored), "json", None, now)
            guard: Any = _ABSENT if current_entry is None else current_entry.raw
            if self.cas_entry(full, guard, entry, now):
                return new_value
        raise StorageError(f"kv incr contention on {full!r}: CAS retries exhausted")

    def list_entries(self, scoped: str, now: float, *, after: str | None = None,
                     limit: int | None = None,
                     include_expired: bool = False) -> list[tuple[str, _Entry]]:
        docs = self._store.find(
            _DOC_COLLECTION, self._range(scoped),
            include_expired=include_expired, after_id=after, limit=limit)
        out = []
        for doc in docs:
            entry = self._to_entry(doc)
            if entry is not None:
                out.append((doc["_id"], entry))
        return out

    def count_prefix(self, scoped: str, now: float) -> int:
        return self._store.count(_DOC_COLLECTION, self._range(scoped))

    def namespaces(self, now: float) -> list[str]:
        docs = self._store.find(_DOC_COLLECTION, limit=_MAX_SCAN)
        found = set()
        for doc in docs:
            key = doc["_id"]
            if _SEP in key:
                found.add(key.split(_SEP, 1)[0])
        return sorted(ns for ns in found if ns)

    def stats(self, scoped: str) -> tuple[int, int, Any, Any]:
        docs = self._store.find(
            _DOC_COLLECTION, self._range(scoped), include_expired=True)
        with_ttl = sum(1 for d in docs if d.get("expires_at") is not None)
        stamps = [float(d.get("updated_at", 0.0)) for d in docs]
        return (len(docs), with_ttl,
                min(stamps) if stamps else None,
                max(stamps) if stamps else None)

    def memory(self, scoped: str) -> tuple[int, int]:
        docs = self._store.find(
            _DOC_COLLECTION, self._range(scoped), include_expired=True)
        total = sum(len(d["_id"]) + len(str(d.get("value", ""))) for d in docs)
        return len(docs), total


class _LockContext:
    """Context manager acquiring a lock (the seam backend's 'transaction')."""

    def __init__(self, lock: threading.RLock) -> None:
        self._lock = lock

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, *exc: Any) -> bool:
        self._lock.release()
        return False


class KVStore:
    """Namespaced, TTL-capable key/value access over ``kv_store``.

        kv = KVStore(db)                        # global scope, SQL rung
        arena = kv.namespace("arena")           # scoped to "arena:..."
        arena.set("custom_topics", ["a", "b"], ttl=3600)
        arena.get("custom_topics", default=[])
        arena.incr("plays_today")

    Pass a :class:`~nomorals.storage.docstore.DocumentStore` as ``store=``
    (or attach one to the database at boot) to serve the same API from the
    document-store seam — MongoDB when configured. See the module docstring.
    """

    TABLE = "kv_store"

    def __init__(
        self,
        db: Any = None,
        *,
        namespace: str = "",
        now_fn: Callable[[], float] | None = None,
        store: Any = None,
    ) -> None:
        if store is None and db is not None:
            # Boot-attached seam (attach_document_store): present only when
            # a non-SQLite rung is active, so the default path is untouched.
            store = getattr(db, "document_store", None)
        if db is None and store is None:
            raise ValueError("KVStore needs a Database or a DocumentStore")
        self._db = db
        self._store = store
        # Nested namespaces (``a:b``) are allowed: ``namespaced()`` builds them
        # and every operation keys off the full ``<ns>:`` prefix, so nesting
        # stays unambiguous for prefix scans and deletes.
        self._ns = namespace.strip().strip(_SEP)
        self._now = now_fn or now
        self._backend: Any = (
            _DocKvBackend(store, self._now) if store is not None
            else _SqlKvBackend(db)
        )
        #: The backend resolution when built via :meth:`from_settings`.
        self.store_resolution: Any = None

    @classmethod
    def from_settings(cls, db: Any, settings: Any, *, namespace: str = "",
                      now_fn: Callable[[], float] | None = None,
                      platform: Any = None) -> KVStore:
        """Build the KV store the install's config selects.

        Resolves the document-store backend (see
        :func:`~nomorals.storage.docstore.resolve_document_store`): Mongo
        when configured + reachable, otherwise the native SQL rung —
        including the degraded case, where resolution has already journaled
        *why* Mongo is not serving. The resolution rides along on
        ``kv.store_resolution`` for status surfaces.
        """
        from .docstore import resolve_document_store

        resolution = resolve_document_store(settings, db, platform=platform)
        if resolution.active == "sqlite":
            kv = cls(db, namespace=namespace, now_fn=now_fn)
        else:
            kv = cls(db, namespace=namespace, now_fn=now_fn,
                     store=resolution.store)
        kv.store_resolution = resolution
        return kv

    # ── scoping ──────────────────────────────────────────────────────────────
    @property
    def namespace(self) -> str:
        return self._ns

    @property
    def backend_name(self) -> str:
        """``"mongo"``/``"sqlite"`` seam name, or ``"sql"`` for the native rung."""
        return self._store.name if self._store is not None else "sql"

    def namespaced(self, namespace: str) -> KVStore:
        """Return a store scoped to ``<self.ns>:<namespace>`` (nested)."""
        child = namespace.strip().strip(_SEP)
        if _SEP in child:
            raise ValueError(f"namespace may not contain {_SEP!r}: {child!r}")
        full = f"{self._ns}{_SEP}{child}" if self._ns else child
        return KVStore(self._db, namespace=full, now_fn=self._now,
                       store=self._store)

    namespace_of = namespaced  # alias: kv.namespace_of("x") reads naturally

    def _full(self, key: str) -> str:
        if not key:
            raise ValueError("kv key must not be empty")
        return f"{self._ns}{_SEP}{key}" if self._ns else key

    def _prefix(self) -> str:
        return f"{self._ns}{_SEP}" if self._ns else ""

    def _scoped(self, prefix: str) -> str:
        """Namespace-scoped key or prefix; empty ``prefix`` = whole namespace."""
        return self._full(prefix) if prefix else self._prefix()

    # ── reads ────────────────────────────────────────────────────────────────
    def get(self, key: str, default: Any = None) -> Any:
        """Return the JSON-decoded value, ``default`` when missing/expired."""
        raw = self.get_raw(key)
        if raw is None:
            return default
        return _decode(raw)

    def get_raw(self, key: str) -> str | None:
        entry = self._backend.get_entry(self._full(key), self._now())
        return entry.raw if entry else None

    def get_str(self, key: str, default: str = "") -> str:
        raw = self.get_raw(key)
        return raw if raw is not None else default

    def get_int(self, key: str, default: int = 0) -> int:
        return self._coerce_num(key, int, default)

    def get_float(self, key: str, default: float = 0.0) -> float:
        return self._coerce_num(key, float, default)

    def get_bool(self, key: str, default: bool = False) -> bool:
        raw = self.get_raw(key)
        if raw is None:
            return default
        value = _decode(raw)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _coerce_num(self, key: str, kind: type, default: Any) -> Any:
        raw = self.get_raw(key)
        if raw is None:
            return default
        value = _decode(raw)
        try:
            return kind(value)  # type: ignore[call-arg]
        except (ValueError, TypeError):
            return default

    def get_many(self, keys: Iterable[str]) -> dict[str, Any]:
        """Batch read; missing/expired keys are absent from the result."""
        keys = list(dict.fromkeys(keys))
        if not keys:
            return {}
        full = [self._full(k) for k in keys]
        entries = self._backend.get_entries(full, self._now())
        back = {f: k for f, k in zip(full, keys)}
        return {back[f]: _decode(e.raw) for f, e in entries.items()}

    def exists(self, key: str) -> bool:
        return self._backend.get_entry(self._full(key), self._now()) is not None

    def ttl(self, key: str) -> float | None:
        """Seconds until expiry; ``None`` = immortal or missing.

        Returns 0.0 for a key that is already expired but not yet swept.
        """
        entry = self._backend.get_entry(
            self._full(key), self._now(), include_expired=True)
        if entry is None or entry.expires_at is None:
            return None
        return max(0.0, float(entry.expires_at) - self._now())

    # ── writes ───────────────────────────────────────────────────────────────
    def _tx(self):
        return self._backend.transaction()

    def _upsert(
        self,
        key: str,
        value: str,
        kind: str,
        ttl: float | None,
    ) -> None:
        full = self._full(key)
        if ttl is not None and ttl <= 0:
            self.delete(key)
            return
        expires_at = (self._now() + ttl) if ttl is not None else None
        with self._tx():
            self._backend.put_entry(
                full, _Entry(value, kind, expires_at, self._now()))

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store any JSON-serializable value (or a plain string as-is)."""
        if isinstance(value, str):
            # Strings are stored verbatim so legacy ``SELECT value`` readers
            # and typed getters see exactly what was written.
            self._upsert(key, value, "text", ttl)
        else:
            self._upsert(key, json.dumps(value), "json", ttl)

    def set_raw(self, key: str, value: str, kind: str = "text",
                ttl: float | None = None) -> None:
        self._upsert(key, value, kind, ttl)

    def set_many(self, mapping: Mapping[str, Any], ttl: float | None = None) -> int:
        """Batch write inside one transaction. Returns keys written."""
        if not mapping:
            return 0
        if ttl is not None and ttl <= 0:
            for key in mapping:
                self.delete(key)
            return 0
        expires_at = (self._now() + ttl) if ttl is not None else None
        stamp = self._now()
        rows = []
        for key, value in mapping.items():
            if isinstance(value, str):
                rows.append((self._full(key), _Entry(value, "text", expires_at, stamp)))
            else:
                rows.append((self._full(key),
                             _Entry(json.dumps(value), "json", expires_at, stamp)))
        with self._tx():
            self._backend.put_entries(rows)
        return len(rows)

    def delete(self, key: str) -> bool:
        return self._backend.delete_entry(self._full(key))

    def delete_prefix(self, prefix: str) -> int:
        """Delete every key under ``<ns>:<prefix>``. Returns rows removed."""
        return self._backend.delete_prefix(self._scoped(prefix))

    def clear_namespace(self) -> int:
        """Delete every key in this namespace (global scope: everything)."""
        if not self._ns:
            return self._backend.clear_all()
        return self.delete_prefix("")

    def delete_expired(self) -> int:
        """Sweep expired rows. Returns rows removed."""
        return self._backend.sweep(self._prefix(), self._now())

    # ── expiry management ────────────────────────────────────────────────────
    def expire(self, key: str, ttl: float | None) -> bool:
        """Set/refresh the TTL of an existing key. ``None`` clears expiry."""
        expires_at = (self._now() + ttl) if ttl is not None and ttl > 0 else None
        if ttl is not None and ttl <= 0:
            return self.delete(key)
        with self._tx():
            if self._backend.get_entry(self._full(key), self._now()) is None:
                return False
            return self._backend.set_expiry(
                self._full(key), expires_at, self._now())

    def persist(self, key: str) -> bool:
        """Remove expiry from a key, making it immortal."""
        return self.expire(key, None)

    # ── redis-grade ergonomics ─────────────────────────────────────────────
    #
    # The patterns below are the Redis/Couchbase surface this store's callers
    # keep re-implementing by hand: sliding expiry, set-if-absent, atomic
    # get-and-replace, hash fields. One canonical implementation each.

    def touch(self, key: str, ttl: float | None) -> bool:
        """Extend (or set) a key's TTL without reading its value.

        Sliding expiry: a session key touched on every request never expires
        while the session is active. ``ttl=None`` persists (immortal).
        Returns False when the key is missing/expired.
        """
        return self.expire(key, ttl)

    def get_and_touch(self, key: str, ttl: float | None,
                      default: Any = None) -> Any:
        """Read a key and refresh its TTL in one call (Couchbase ``getAndTouch``)."""
        value = self.get(key, default)
        if value is not default or self.exists(key):
            self.expire(key, ttl)
        return value

    def setnx(self, key: str, value: Any, ttl: float | None = None) -> bool:
        """Set ``key`` only if absent (Redis ``SET NX``). Returns True if set."""
        return self.compare_and_set(key, None, value, ttl=ttl)

    def getset(self, key: str, value: Any, ttl: float | None = None) -> Any:
        """Atomically replace ``key``'s value, returning the old one (or None)."""
        raw = value if isinstance(value, str) else json.dumps(value)
        kind = "text" if isinstance(value, str) else "json"
        expires_at = (self._now() + ttl) if ttl is not None else None
        with self._tx():
            old = self._backend.replace_entry(
                self._full(key), _Entry(raw, kind, expires_at, self._now()),
                self._now())
        return _decode(old.raw) if old is not None else None

    def getdel(self, key: str) -> Any:
        """Atomically read and delete ``key`` (Redis ``GETDEL``)."""
        with self._tx():
            entry = self._backend.take_entry(self._full(key), self._now())
        return _decode(entry.raw) if entry is not None else None

    def pop(self, key: str, default: Any = None) -> Any:
        """``getdel`` with a default."""
        with self._tx():
            if not self.exists(key):
                return default
            return self.getdel(key)

    def append(self, key: str, suffix: str, ttl: float | None = None) -> int:
        """Append ``suffix`` to a text value; returns the new length.

        Missing keys start from ``""``. Non-string values are JSON-encoded
        first so the append is still well-defined.
        """
        with self._tx():
            raw = self.get_raw(key)
            if raw is None:
                current = ""
            else:
                decoded = _decode(raw)
                current = decoded if isinstance(decoded, str) else raw
            new_value = current + suffix
            self.set(key, new_value, ttl=ttl)
        return len(new_value)

    def rename(self, src: str, dst: str) -> bool:
        """Rename ``src`` to ``dst`` (Redis ``RENAME``), keeping TTL. False if missing."""
        with self._tx():
            entry = self._backend.get_entry(self._full(src), self._now())
            if entry is None:
                return False
            ttl = None
            if entry.expires_at is not None:
                ttl = max(0.0, float(entry.expires_at) - self._now())
            self.set_raw(dst, entry.raw, kind=entry.kind or "text", ttl=ttl)
            self.delete(src)
        return True

    def get_or_set(self, key: str, factory: Callable[[], Any],
                   ttl: float | None = None) -> Any:
        """Cache-aside: return the live value, or compute/store/return it.

        ``factory`` runs at most once per miss for the calling thread's
        transaction; concurrent threads may both compute (documented, like
        most cache-aside helpers) — the last writer wins, atomically.
        """
        value = self.get(key, default=None)
        if value is not None or self.exists(key):
            return value
        computed = factory()
        # Another thread may have populated it while we computed; setnx keeps
        # the first writer's value and we return what's actually stored.
        self.setnx(key, computed, ttl=ttl)
        return self.get(key, default=computed)

    def scan_cursor(self, prefix: str = "", *, cursor: str = "",
                    count: int = 100) -> tuple[str, list[tuple[str, Any]]]:
        """Bounded cursor pagination (Redis ``SCAN`` semantics).

        Returns ``(next_cursor, [(key, value)])``; ``next_cursor == ""`` means
        the scan is complete. Keys are ordered, so the cursor is simply the
        last key returned.
        """
        count = min(max(1, count), _MAX_SCAN)
        after = (self._prefix() + cursor) if cursor else None
        rows = self._backend.list_entries(
            self._scoped(prefix), self._now(), after=after, limit=count + 1)
        n = len(self._prefix())
        out = [(full[n:], _decode(entry.raw)) for full, entry in rows[:count]]
        if len(rows) > count:
            next_cursor = rows[count - 1][0][n:]
        else:
            next_cursor = ""
        return next_cursor, out

    # ── hashes ─────────────────────────────────────────────────────────────
    #
    # Redis-hash semantics over a single JSON-object row: ``user:42`` holds
    # ``{"name": ..., "plan": ...}`` and fields are read/written atomically
    # without deserializing the whole object at the call site.

    def _hget_obj(self, key: str) -> dict[str, Any] | None:
        value = self.get(key)
        if value is None:
            return None
        if not isinstance(value, dict):
            raise TypeError(f"kv hash operation on non-hash key {key!r}")
        return value

    def hset(self, key: str, field: str, value: Any,
             ttl: float | None = None) -> int:
        """Set one hash field. Returns 1 if the field is new, 0 if updated."""
        with self._tx():
            obj = self._hget_obj(key) or {}
            is_new = field not in obj
            obj[field] = value
            if ttl is not None:
                self.set(key, obj, ttl=ttl)
            else:
                # Preserve any existing TTL.
                entry = self._backend.get_entry(
                    self._full(key), self._now(), include_expired=True)
                keep = None
                if entry is not None and entry.expires_at is not None:
                    keep = max(0.0, float(entry.expires_at) - self._now())
                self.set(key, obj, ttl=keep)
        return 1 if is_new else 0

    def hget(self, key: str, field: str, default: Any = None) -> Any:
        obj = self._hget_obj(key)
        return obj.get(field, default) if obj else default

    def hgetall(self, key: str) -> dict[str, Any]:
        return self._hget_obj(key) or {}

    def hdel(self, key: str, *fields: str) -> int:
        """Delete hash fields. Returns the number removed."""
        with self._tx():
            obj = self._hget_obj(key)
            if not obj:
                return 0
            removed = sum(1 for f in fields if obj.pop(f, None) is not None)
            if obj:
                self.set(key, obj)
            else:
                self.delete(key)
        return removed

    def hincrby(self, key: str, field: str, delta: float = 1) -> float:
        """Atomically increment a hash field (Redis ``HINCRBY``)."""
        with self._tx():
            obj = self._hget_obj(key) or {}
            current = obj.get(field, 0)
            try:
                new_value = float(current) + delta
            except (TypeError, ValueError):
                raise TypeError(
                    f"hash field {key!r}.{field!r} is not numeric")
            obj[field] = (int(new_value) if new_value.is_integer()
                          else new_value)
            self.set(key, obj)
        return obj[field]

    def hkeys(self, key: str) -> list[str]:
        return list((self._hget_obj(key) or {}).keys())

    def hlen(self, key: str) -> int:
        return len(self._hget_obj(key) or {})

    # ── atomic ops ───────────────────────────────────────────────────────────
    def compare_and_set(self, key: str, expected: Any, new: Any,
                        ttl: float | None = None) -> bool:
        """Atomically set ``key`` to ``new`` iff its current value == ``expected``.

        ``expected=None`` means "only set when the key is absent". Returns
        True when the swap happened. The decoded comparison happens against
        the live entry; the backend then guards the write on the exact raw
        bytes read, so a racing writer fails the swap on every backend.
        """
        full = self._full(key)
        expires_at = (self._now() + ttl) if ttl is not None else None
        new_raw = new if isinstance(new, str) else json.dumps(new)
        new_kind = "text" if isinstance(new, str) else "json"
        entry = _Entry(new_raw, new_kind, expires_at, self._now())
        with self._tx():
            current = self._backend.get_entry(full, self._now())
            if expected is None:
                if current is not None:
                    return False
                guard: Any = _ABSENT
            else:
                if current is None:
                    return False
                if _decode(current.raw) != expected:
                    return False
                guard = current.raw
            return self._backend.cas_entry(full, guard, entry, self._now())

    def incr(self, key: str, delta: float = 1) -> float:
        """Atomically add ``delta`` to a numeric key (created at 0 if absent).

        Tolerates values written as plain strings by legacy callers.
        """
        with self._tx():
            return self._backend.incr_entry(
                self._full(key), float(delta), self._now())

    def decr(self, key: str, delta: float = 1) -> float:
        return self.incr(key, -delta)

    # ── inspection ───────────────────────────────────────────────────────────
    def keys(self, prefix: str = "", limit: int = 1000) -> list[str]:
        """Live keys under this namespace starting with ``prefix`` (unprefixed)."""
        limit = min(max(1, limit), _MAX_SCAN)
        rows = self._backend.list_entries(
            self._scoped(prefix), self._now(), limit=limit)
        n = len(self._prefix())
        return [full[n:] for full, _ in rows]

    def scan(self, prefix: str = "", limit: int = 1000) -> list[tuple[str, Any]]:
        """``(key, decoded value)`` pairs under this namespace (unprefixed keys)."""
        limit = min(max(1, limit), _MAX_SCAN)
        rows = self._backend.list_entries(
            self._scoped(prefix), self._now(), limit=limit)
        n = len(self._prefix())
        return [(full[n:], _decode(entry.raw)) for full, entry in rows]

    def count(self, prefix: str = "") -> int:
        return self._backend.count_prefix(self._scoped(prefix), self._now())

    def namespaces(self) -> list[str]:
        """Top-level namespaces present in the store (global scope only)."""
        return self._backend.namespaces(self._now())

    def stats(self) -> dict[str, Any]:
        """Size/age overview of this namespace."""
        n, with_ttl, oldest, newest = self._backend.stats(self._prefix())
        return {
            "namespace": self._ns or "<global>",
            "keys": n,
            "with_ttl": with_ttl,
            "oldest_updated_at": oldest,
            "newest_updated_at": newest,
        }

    def memory_usage(self) -> dict[str, Any]:
        """Rough size estimate of this namespace (bytes of stored values)."""
        n, total = self._backend.memory(self._prefix())
        return {
            "namespace": self._ns or "<global>",
            "keys": n,
            "bytes": total,
        }

    def format_stats(self, theme: Any = None) -> str:
        """Human-readable namespace overview through the shared style layer."""
        theme = theme or active_theme()
        stats = self.stats()
        mem = self.memory_usage()
        rows = [
            (stats["namespace"], str(stats["keys"]), str(stats["with_ttl"]),
             _kv_bytes(mem["bytes"])),
        ]
        return "\n".join([
            header("kv store", theme=theme),
            styled_table(
                ["namespace", "keys", "with ttl", "size"],
                rows, theme=theme,
            ),
        ])

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"KVStore(namespace={self._ns!r}, backend={self.backend_name!r})"


def _kv_bytes(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return f"{num:.1f}{unit}" if unit != "B" else f"{num}B"
        num /= 1024
    return f"{num:.1f}GB"  # pragma: no cover
