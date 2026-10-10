"""First-class key/value store over the ``kv_store`` table.

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

The ``Database`` drives the transactions here; this class never opens its own
connections.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Iterable, Mapping

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


def now() -> float:
    return time.time()


class KVStore:
    """Namespaced, TTL-capable key/value access over ``kv_store``.

        kv = KVStore(db)                        # global scope
        arena = kv.namespace("arena")           # scoped to "arena:..."
        arena.set("custom_topics", ["a", "b"], ttl=3600)
        arena.get("custom_topics", default=[])
        arena.incr("plays_today")
    """

    TABLE = "kv_store"

    def __init__(
        self,
        db: Any,
        *,
        namespace: str = "",
        now_fn: Callable[[], float] | None = None,
    ) -> None:
        self._db = db
        # Nested namespaces (``a:b``) are allowed: ``namespaced()`` builds them
        # and every operation keys off the full ``<ns>:`` prefix, so nesting
        # stays unambiguous for prefix scans and deletes.
        self._ns = namespace.strip().strip(_SEP)
        self._now = now_fn or now

    # ── scoping ──────────────────────────────────────────────────────────────
    @property
    def namespace(self) -> str:
        return self._ns

    def namespaced(self, namespace: str) -> "KVStore":
        """Return a store scoped to ``<self.ns>:<namespace>`` (nested)."""
        child = namespace.strip().strip(_SEP)
        if _SEP in child:
            raise ValueError(f"namespace may not contain {_SEP!r}: {child!r}")
        full = f"{self._ns}{_SEP}{child}" if self._ns else child
        return KVStore(self._db, namespace=full, now_fn=self._now)

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
    def _live_clause(self, alias: str = "") -> str:
        """SQL fragment excluding expired rows (lazy expiry)."""
        col = f"{alias}expires_at" if not alias else f"{alias}.expires_at"
        return f"({col} IS NULL OR {col} > ?)"

    def get(self, key: str, default: Any = None) -> Any:
        """Return the JSON-decoded value, ``default`` when missing/expired."""
        raw = self.get_raw(key)
        if raw is None:
            return default
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            # Rows written by legacy raw-SQL callers as plain strings.
            return raw

    def get_raw(self, key: str) -> str | None:
        row = self._db.query_one(
            f"SELECT value FROM {self.TABLE} WHERE key = ? AND {self._live_clause()}",
            (self._full(key), self._now()),
        )
        return row["value"] if row else None

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
        try:
            value = json.loads(raw)
        except (ValueError, TypeError):
            value = raw
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _coerce_num(self, key: str, kind: type, default: Any) -> Any:
        raw = self.get_raw(key)
        if raw is None:
            return default
        try:
            value = json.loads(raw)
        except (ValueError, TypeError):
            value = raw
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
        placeholders = ", ".join("?" for _ in full)
        rows = self._db.query(
            f"SELECT key, value FROM {self.TABLE} "
            f"WHERE key IN ({placeholders}) AND {self._live_clause()}",
            (*full, self._now()),
        )
        back = {f: k for f, k in zip(full, keys)}
        out: dict[str, Any] = {}
        for row in rows:
            try:
                out[back[row["key"]]] = json.loads(row["value"])
            except (ValueError, TypeError):
                out[back[row["key"]]] = row["value"]
        return out

    def exists(self, key: str) -> bool:
        return (
            self._db.scalar(
                f"SELECT 1 FROM {self.TABLE} WHERE key = ? AND {self._live_clause()}",
                (self._full(key), self._now()),
            )
            is not None
        )

    def ttl(self, key: str) -> float | None:
        """Seconds until expiry; ``None`` = immortal or missing.

        Returns 0.0 for a key that is already expired but not yet swept.
        """
        row = self._db.query_one(
            f"SELECT expires_at FROM {self.TABLE} WHERE key = ?",
            (self._full(key),),
        )
        if row is None or row["expires_at"] is None:
            return None
        return max(0.0, float(row["expires_at"]) - self._now())

    # ── writes ───────────────────────────────────────────────────────────────
    def _tx(self):
        """Transaction context, or a no-op for DBs without transaction support."""
        tx = getattr(self._db, "transaction", None)
        if tx is None:
            from contextlib import nullcontext
            return nullcontext()
        return tx()

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
            self._db.execute(
                f"INSERT INTO {self.TABLE} (key, value, kind, updated_at, expires_at) "
                f"VALUES (?, ?, ?, ?, ?) "
                f"ON CONFLICT(key) DO UPDATE SET "
                f"value = excluded.value, kind = excluded.kind, "
                f"updated_at = excluded.updated_at, expires_at = excluded.expires_at",
                (full, value, kind, self._now(), expires_at),
            )

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
                rows.append((self._full(key), value, "text", stamp, expires_at))
            else:
                rows.append((self._full(key), json.dumps(value), "json", stamp, expires_at))
        with self._tx():
            self._db.executemany(
                f"INSERT INTO {self.TABLE} (key, value, kind, updated_at, expires_at) "
                f"VALUES (?, ?, ?, ?, ?) "
                f"ON CONFLICT(key) DO UPDATE SET "
                f"value = excluded.value, kind = excluded.kind, "
                f"updated_at = excluded.updated_at, expires_at = excluded.expires_at",
                rows,
            )
        return len(rows)

    def delete(self, key: str) -> bool:
        changed = self._db.delete(self.TABLE, "key = ?", (self._full(key),))
        return changed > 0

    def delete_prefix(self, prefix: str) -> int:
        """Delete every key under ``<ns>:<prefix>``. Returns rows removed."""
        like = self._scoped(prefix) + "%"
        return self._db.delete(self.TABLE, "key LIKE ? ESCAPE '\\'", (like,))

    def clear_namespace(self) -> int:
        """Delete every key in this namespace (global scope: everything)."""
        if not self._ns:
            return self._db.execute(f"DELETE FROM {self.TABLE}").rowcount
        return self.delete_prefix("")

    def delete_expired(self) -> int:
        """Sweep expired rows. Returns rows removed."""
        if self._ns:
            cursor = self._db.execute(
                f"DELETE FROM {self.TABLE} WHERE key LIKE ? ESCAPE '\\' "
                f"AND expires_at IS NOT NULL AND expires_at <= ?",
                (self._prefix() + "%", self._now()),
            )
        else:
            cursor = self._db.execute(
                f"DELETE FROM {self.TABLE} "
                f"WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (self._now(),),
            )
        return cursor.rowcount

    # ── expiry management ────────────────────────────────────────────────────
    def expire(self, key: str, ttl: float | None) -> bool:
        """Set/refresh the TTL of an existing key. ``None`` clears expiry."""
        expires_at = (self._now() + ttl) if ttl is not None and ttl > 0 else None
        if ttl is not None and ttl <= 0:
            return self.delete(key)
        with self._tx():
            row = self._db.query_one(
                f"SELECT key FROM {self.TABLE} WHERE key = ? AND {self._live_clause()}",
                (self._full(key), self._now()),
            )
            if row is None:
                return False
            self._db.execute(
                f"UPDATE {self.TABLE} SET expires_at = ?, updated_at = ? WHERE key = ?",
                (expires_at, self._now(), self._full(key)),
            )
        return True

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
        with self._tx():
            old = self.get(key)
            self.set(key, value, ttl=ttl)
        return old

    def getdel(self, key: str) -> Any:
        """Atomically read and delete ``key`` (Redis ``GETDEL``)."""
        with self._tx():
            value = self.get(key)
            self.delete(key)
        return value

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
                try:
                    decoded = json.loads(raw)
                    current = decoded if isinstance(decoded, str) else raw
                except (ValueError, TypeError):
                    current = raw
            new_value = current + suffix
            self.set(key, new_value, ttl=ttl)
        return len(new_value)

    def rename(self, src: str, dst: str) -> bool:
        """Rename ``src`` to ``dst`` (Redis ``RENAME``), keeping TTL. False if missing."""
        with self._tx():
            row = self._db.query_one(
                f"SELECT value, kind, expires_at FROM {self.TABLE} "
                f"WHERE key = ? AND {self._live_clause()}",
                (self._full(src), self._now()),
            )
            if row is None:
                return False
            ttl = None
            if row["expires_at"] is not None:
                ttl = max(0.0, float(row["expires_at"]) - self._now())
            self.set_raw(dst, row["value"], kind=row["kind"] or "text",
                         ttl=ttl)
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
        like = self._scoped(prefix) + "%"
        params: list[Any] = [like]
        key_filter = ""
        if cursor:
            key_filter = "AND key > ?"
            params.append(self._prefix() + cursor)
        params.extend([self._now(), count + 1])  # live-clause, then LIMIT
        rows = self._db.query(
            f"SELECT key, value FROM {self.TABLE} "
            f"WHERE key LIKE ? ESCAPE '\\' {key_filter} "
            f"AND {self._live_clause()} ORDER BY key LIMIT ?",
            tuple(params),
        )
        n = len(self._prefix())
        out: list[tuple[str, Any]] = []
        for row in rows[:count]:
            try:
                value: Any = json.loads(row["value"])
            except (ValueError, TypeError):
                value = row["value"]
            out.append((row["key"][n:], value))
        if len(rows) > count:
            next_cursor = rows[count - 1]["key"][n:]
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
                row = self._db.query_one(
                    f"SELECT expires_at FROM {self.TABLE} WHERE key = ?",
                    (self._full(key),),
                )
                keep = None
                if row and row["expires_at"] is not None:
                    keep = max(0.0, float(row["expires_at"]) - self._now())
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
        True when the swap happened.
        """
        full = self._full(key)
        expires_at = (self._now() + ttl) if ttl is not None else None
        new_raw = new if isinstance(new, str) else json.dumps(new)
        new_kind = "text" if isinstance(new, str) else "json"
        with self._tx():
            current = self.get_raw(key)
            if expected is None:
                if current is not None:
                    return False
            else:
                if current is None:
                    return False
                try:
                    cur_val: Any = json.loads(current)
                except (ValueError, TypeError):
                    cur_val = current
                if cur_val != expected:
                    return False
            self._db.execute(
                f"INSERT INTO {self.TABLE} (key, value, kind, updated_at, expires_at) "
                f"VALUES (?, ?, ?, ?, ?) "
                f"ON CONFLICT(key) DO UPDATE SET "
                f"value = excluded.value, kind = excluded.kind, "
                f"updated_at = excluded.updated_at, expires_at = excluded.expires_at",
                (full, new_raw, new_kind, self._now(), expires_at),
            )
        return True

    def incr(self, key: str, delta: float = 1) -> float:
        """Atomically add ``delta`` to a numeric key (created at 0 if absent).

        Tolerates values written as plain strings by legacy callers.
        """
        full = self._full(key)
        with self._tx():
            current = self.get_float(key, 0.0)
            new_value = current + delta
            # Store integers without a decimal point when exact.
            stored: Any = int(new_value) if float(new_value).is_integer() else new_value
            self._db.execute(
                f"INSERT INTO {self.TABLE} (key, value, kind, updated_at) "
                f"VALUES (?, ?, 'json', ?) "
                f"ON CONFLICT(key) DO UPDATE SET "
                f"value = excluded.value, kind = excluded.kind, "
                f"updated_at = excluded.updated_at, expires_at = NULL",
                (full, json.dumps(stored), self._now()),
            )
        return new_value

    def decr(self, key: str, delta: float = 1) -> float:
        return self.incr(key, -delta)

    # ── inspection ───────────────────────────────────────────────────────────
    def keys(self, prefix: str = "", limit: int = 1000) -> list[str]:
        """Live keys under this namespace starting with ``prefix`` (unprefixed)."""
        limit = min(max(1, limit), _MAX_SCAN)
        like = self._scoped(prefix) + "%"
        rows = self._db.query(
            f"SELECT key FROM {self.TABLE} WHERE key LIKE ? ESCAPE '\\' "
            f"AND {self._live_clause()} ORDER BY key LIMIT ?",
            (like, self._now(), limit),
        )
        n = len(self._prefix())
        return [r["key"][n:] for r in rows]

    def scan(self, prefix: str = "", limit: int = 1000) -> list[tuple[str, Any]]:
        """``(key, decoded value)`` pairs under this namespace (unprefixed keys)."""
        limit = min(max(1, limit), _MAX_SCAN)
        like = self._scoped(prefix) + "%"
        rows = self._db.query(
            f"SELECT key, value FROM {self.TABLE} WHERE key LIKE ? ESCAPE '\\' "
            f"AND {self._live_clause()} ORDER BY key LIMIT ?",
            (like, self._now(), limit),
        )
        n = len(self._prefix())
        out = []
        for row in rows:
            try:
                value: Any = json.loads(row["value"])
            except (ValueError, TypeError):
                value = row["value"]
            out.append((row["key"][n:], value))
        return out

    def count(self, prefix: str = "") -> int:
        like = self._scoped(prefix) + "%"
        return int(
            self._db.scalar(
                f"SELECT COUNT(*) FROM {self.TABLE} "
                f"WHERE key LIKE ? ESCAPE '\\' AND {self._live_clause()}",
                (like, self._now()),
                default=0,
            )
        )

    def namespaces(self) -> list[str]:
        """Top-level namespaces present in the store (global scope only)."""
        rows = self._db.query(
            f"SELECT DISTINCT substr(key, 1, instr(key, '{_SEP}') - 1) AS ns "
            f"FROM {self.TABLE} WHERE instr(key, '{_SEP}') > 0 "
            f"AND {self._live_clause()} ORDER BY ns",
            (self._now(),),
        )
        return [r["ns"] for r in rows if r["ns"]]

    def stats(self) -> dict[str, Any]:
        """Size/age overview of this namespace."""
        prefix = self._prefix()
        if prefix:
            where = "WHERE key LIKE ? ESCAPE '\\'"
            args: tuple = (prefix + "%",)
        else:
            where = ""
            args = ()
        row = self._db.query_one(
            f"SELECT COUNT(*) AS n, "
            f"SUM(CASE WHEN expires_at IS NOT NULL THEN 1 ELSE 0 END) AS with_ttl, "
            f"MIN(updated_at) AS oldest, MAX(updated_at) AS newest "
            f"FROM {self.TABLE} {where}",
            args,
        ) or {}
        return {
            "namespace": self._ns or "<global>",
            "keys": int(row.get("n") or 0),
            "with_ttl": int(row.get("with_ttl") or 0),
            "oldest_updated_at": row.get("oldest"),
            "newest_updated_at": row.get("newest"),
        }

    def memory_usage(self) -> dict[str, Any]:
        """Rough size estimate of this namespace (bytes of stored values)."""
        prefix = self._prefix()
        if prefix:
            where = "WHERE key LIKE ? ESCAPE '\\'"
            args: tuple = (prefix + "%",)
        else:
            where = ""
            args = ()
        row = self._db.query_one(
            f"SELECT COUNT(*) AS n, COALESCE(SUM(LENGTH(key) + LENGTH(value)), 0) AS bytes "
            f"FROM {self.TABLE} {where}",
            args,
        ) or {}
        return {
            "namespace": self._ns or "<global>",
            "keys": int(row.get("n") or 0),
            "bytes": int(row.get("bytes") or 0),
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
        return f"KVStore(namespace={self._ns!r})"


def _kv_bytes(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return f"{num:.1f}{unit}" if unit != "B" else f"{num}B"
        num /= 1024
    return f"{num:.1f}GB"  # pragma: no cover
