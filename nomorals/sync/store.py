"""Versioned sync store: the durable unit of replication.

Every record carries a **Hybrid Logical Clock** ``(hlc_ts, hlc_count)`` plus
``device_id``. Deletes are tombstones, not removals, so deletes replicate.
Values merge **per field**: each field of the JSON value carries its own HLC
in the ``clocks`` fat column, so two devices editing *different* fields of
the same record both win — only true same-field conflicts fall back to
last-write-wins.

Replication uses a monotonic per-store sequence number (``seq``), assigned
on every local write — including writes applied from a peer. The engine's
push/pull cursors are seqs, not timestamps, so a backdated write (explicit
old ``updated_at``, phone/cloud clock skew) can never slip past a cursor
unseen. ``updated_at`` stays as the human wall-clock display field; the HLC
is the conflict-resolution key.

Why HLC instead of wall clock: phone clocks drift and users change them by
hand. A device 9 hours behind would lose *every* conflict under wall-clock
LWW even when its edit happened last. The HLC's ``receive`` rule pulls a
lagging device's clock forward on every remote write, so "later" means
causally later, not "later according to one device's wall clock".
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from ..core.logging_setup import get_logger
from ..storage.db import Database

__all__ = ["SyncRecord", "SyncStore", "ensure_column"]

_log = get_logger(__name__)

TABLE = "sync_records"

#: Default tombstone grace period: 90 days. Deleting a tombstone is only
#: safe once every peer has pulled it; without coordinated GC the honest
#: trade is a conservative time-based expiry. A device offline longer than
#: the grace period can resurrect a GC'd key on its next push — monitor
#: your oldest unsynced replica before shortening this.
DEFAULT_TOMBSTONE_GRACE_S = 90 * 86400


def ensure_column(db: Database, table: str, column: str, ddl: str) -> None:
    """Add ``column`` to ``table`` if missing.

    Tolerates losing a creation race with another thread/process building
    the same schema: if the ALTER fails, the column is re-checked and the
    error only propagates when the column is still genuinely absent.
    """
    cols = {r["name"] for r in db.table_info(table)}
    if column in cols:
        return
    try:
        db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
    except Exception:  # noqa: BLE001 - re-checked below; real errors re-raise
        cols = {r["name"] for r in db.table_info(table)}
        if column not in cols:
            raise
        _log.debug("column %s.%s appeared via a concurrent migration",
                   table, column)


def _norm_clocks(raw: Any) -> dict[str, dict[str, Any]]:
    """Normalize the per-field clock map from wire/DB form."""
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        if not isinstance(v, dict):
            continue
        out[str(k)] = {
            "ts": float(v.get("ts") or 0.0),
            "c": int(v.get("c") or 0),
            "d": str(v.get("d") or ""),
            "tomb": bool(v.get("tomb")),
        }
    return out


def _clock_key(ts: float, count: int, device_id: str) -> tuple[float, int, str]:
    return (ts, count, device_id)


@dataclass
class SyncRecord:
    key: str
    value: dict[str, Any]
    updated_at: float
    device_id: str
    deleted: bool = False
    # Local replication cursor, assigned by SyncStore on write. Not part of
    # the LWW identity — two records are "the same" when (hlc, device_id,
    # value, deleted, clocks) match, regardless of seq.
    seq: int = 0
    # Hybrid Logical Clock: (hlc_ts, hlc_count). The conflict-resolution
    # key. (0.0, 0) means "pre-HLC row" — see wins() for the legacy path.
    hlc_ts: float = 0.0
    hlc_count: int = 0
    # Per-field clocks: field -> {"ts", "c", "d", "tomb"}. Drives the
    # field-level merge; "tomb" marks a field deleted by put().
    clocks: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.clocks = _norm_clocks(self.clocks)

    def clock_key(self) -> tuple[float, int, str]:
        """Total order key for conflict resolution."""
        return _clock_key(self.hlc_ts, self.hlc_count, self.device_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": dict(self.value),
            "updated_at": self.updated_at,
            "device_id": self.device_id,
            "deleted": self.deleted,
            "seq": self.seq,
            "hlc_ts": self.hlc_ts,
            "hlc_count": self.hlc_count,
            "clocks": {k: dict(v) for k, v in self.clocks.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SyncRecord":
        """Rebuild from :meth:`to_dict` output (hub HTTP wire format).

        ``seq`` is informational here: the receiver's store assigns its own
        seq on write, and LWW identity ignores it. Missing HLC/clock
        fields default to the pre-HLC shape so old peers interoperate.
        """
        return cls(
            key=str(data.get("key") or ""),
            value=dict(data.get("value") or {}),
            updated_at=float(data.get("updated_at") or 0.0),
            device_id=str(data.get("device_id") or ""),
            deleted=bool(data.get("deleted")),
            seq=int(data.get("seq") or 0),
            hlc_ts=float(data.get("hlc_ts") or 0.0),
            hlc_count=int(data.get("hlc_count") or 0),
            clocks=_norm_clocks(data.get("clocks")),
        )

    @staticmethod
    def wins(a: "SyncRecord", b: "SyncRecord") -> "SyncRecord":
        """Last-write-wins; ties break deterministically by device_id.

        Compares the HLC ``(hlc_ts, hlc_count, device_id)``. When *either*
        record predates HLC (clock ``(0.0, 0)``) the comparison falls back
        to the legacy ``(updated_at, device_id)`` rule — HLC ordering is
        only meaningful when both sides speak HLC, and this keeps old
        peers interoperable instead of silently losing every conflict.
        """
        if (a.hlc_ts, a.hlc_count) == (0.0, 0) or \
                (b.hlc_ts, b.hlc_count) == (0.0, 0):
            if (a.updated_at, a.device_id) >= (b.updated_at, b.device_id):
                return a
            return b
        if a.clock_key() >= b.clock_key():
            return a
        return b


class SyncStore:
    """SQLite-backed versioned KV store with HLC + per-field LWW merge."""

    def __init__(self, db: Database, device_id: str) -> None:
        if not device_id:
            raise ValueError("device_id is required")
        self.db = db
        self.device_id = device_id
        self._lock = threading.Lock()
        self._subscribers: list[Callable[[SyncRecord], None]] = []
        self._ensure_schema()
        self._load_clock()

    # ── schema ─────────────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        self.db.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL DEFAULT '{{}}',
                updated_at REAL NOT NULL DEFAULT 0,
                device_id TEXT NOT NULL DEFAULT '',
                deleted INTEGER NOT NULL DEFAULT 0,
                seq INTEGER NOT NULL DEFAULT 0,
                hlc_ts REAL NOT NULL DEFAULT 0,
                hlc_count INTEGER NOT NULL DEFAULT 0,
                clocks TEXT NOT NULL DEFAULT '{{}}'
            )"""
        )
        self.db.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_updated ON {TABLE}(updated_at)"
        )
        # Migrations for pre-HLC / pre-seq databases.
        self._ensure_seq_column()
        ensure_column(self.db, TABLE, "hlc_ts", "REAL NOT NULL DEFAULT 0")
        ensure_column(self.db, TABLE, "hlc_count", "INTEGER NOT NULL DEFAULT 0")
        ensure_column(self.db, TABLE, "clocks", "TEXT NOT NULL DEFAULT '{}'")
        self._backfill_hlc()
        self.db.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_seq ON {TABLE}(seq)"
        )

    def _ensure_seq_column(self) -> None:
        """Migration for pre-seq databases: add the column, then backfill.

        Backfill order is (updated_at, rowid) continuing from the current
        max seq, so rows that predate the seq era keep a sensible order and
        no two rows share a seq.
        """
        ensure_column(self.db, TABLE, "seq", "INTEGER NOT NULL DEFAULT 0")
        with self.db.transaction():
            pending = self.db.query(
                f"SELECT rowid AS rid FROM {TABLE} "
                "WHERE seq = 0 ORDER BY updated_at, rowid"
            )
            if not pending:
                return
            start = int(self.db.scalar(
                f"SELECT COALESCE(MAX(seq), 0) FROM {TABLE}", default=0))
            for i, row in enumerate(pending, start=1):
                self.db.execute(
                    f"UPDATE {TABLE} SET seq = ? WHERE rowid = ?",
                    (start + i, row["rid"]),
                )
            _log.info("sync store: backfilled seq for %d rows", len(pending))

    def _backfill_hlc(self) -> None:
        """Migration for pre-HLC rows: seed the clock from ``updated_at``.

        Every pre-HLC row gets ``hlc_ts = updated_at, hlc_count = 0`` and a
        per-field clock for each value field at the record's own HLC, so
        the field-level merge has something to compare against.
        """
        with self.db.transaction():
            rows = self.db.query(
                f"SELECT key, value, updated_at, device_id, hlc_ts "
                f"FROM {TABLE} WHERE hlc_ts = 0 AND hlc_count = 0"
            )
            for r in rows:
                try:
                    value = json.loads(r["value"] or "{}")
                except Exception:  # noqa: BLE001 - corrupt value; keep going
                    value = {}
                clocks = {
                    str(k): {"ts": float(r["updated_at"] or 0.0), "c": 0,
                             "d": str(r["device_id"] or ""), "tomb": False}
                    for k in (value if isinstance(value, dict) else {})
                }
                self.db.execute(
                    f"UPDATE {TABLE} SET hlc_ts = ?, hlc_count = 0, "
                    f"clocks = ? WHERE key = ?",
                    (float(r["updated_at"] or 0.0), json.dumps(clocks),
                     r["key"]),
                )
            if rows:
                _log.info("sync store: backfilled HLC for %d rows", len(rows))

    def _load_clock(self) -> None:
        """Seed the in-memory HLC from the newest row clock on disk."""
        with self._lock:
            ts = float(self.db.scalar(
                f"SELECT COALESCE(MAX(hlc_ts), 0) FROM {TABLE}", default=0.0))
            count = int(self.db.scalar(
                f"SELECT COALESCE(MAX(hlc_count), 0) FROM {TABLE} "
                "WHERE hlc_ts = ?", (ts,), default=0))
            self._hlc_ts = ts
            self._hlc_count = count

    # ── hybrid logical clock ───────────────────────────────────────────

    def _tick(self, now: float | None = None) -> tuple[float, int]:
        """HLC send rule: advance the clock for a local write."""
        now = time.time() if now is None else now
        with self._lock:
            if now > self._hlc_ts:
                self._hlc_ts, self._hlc_count = now, 0
            else:
                self._hlc_count += 1
            return self._hlc_ts, self._hlc_count

    def _receive(self, remote_ts: float, remote_count: int,
                 now: float | None = None) -> tuple[float, int]:
        """HLC receive rule: fold a remote clock into ours.

        A lagging device's clock jumps forward past any clock it has seen,
        so its *next* write is causally "later" than everything observed —
        this is what makes LWW correct under phone clock skew.
        """
        now = time.time() if now is None else now
        with self._lock:
            lts, lc = self._hlc_ts, self._hlc_count
            ts = max(lts, remote_ts, now)
            if ts == lts == remote_ts:
                count = max(lc, remote_count) + 1
            elif ts == lts:
                count = lc + 1
            elif ts == remote_ts:
                count = remote_count + 1
            else:
                count = 0
            self._hlc_ts, self._hlc_count = ts, count
            return ts, count

    def clock(self) -> tuple[float, int]:
        """Current HLC ``(ts, count)`` (for diagnostics)."""
        with self._lock:
            return self._hlc_ts, self._hlc_count

    # ── subscriptions ────────────────────────────────────────────────

    def subscribe(self, fn: Callable[[SyncRecord], None]) -> None:
        """Call ``fn(record)`` after every applied write (put/delete/apply).

        Powers live/continuous sync: the engine re-syncs (debounced) on
        change instead of polling blindly. Fail-open — a raising
        subscriber is logged, never breaks the write.
        """
        with self._lock:
            if fn not in self._subscribers:
                self._subscribers.append(fn)

    def unsubscribe(self, fn: Callable[[SyncRecord], None]) -> None:
        with self._lock:
            if fn in self._subscribers:
                self._subscribers.remove(fn)

    def _notify(self, rec: SyncRecord) -> None:
        with self._lock:
            subs = list(self._subscribers)
        for fn in subs:
            try:
                fn(rec)
            except Exception:  # noqa: BLE001 - fail-open telemetry
                _log.debug("sync subscriber failed for key %s", rec.key,
                           exc_info=True)

    # ── writes ───────────────────────────────────────────────────────

    def put(
        self,
        key: str,
        value: dict[str, Any],
        *,
        updated_at: float | None = None,
    ) -> SyncRecord:
        """Write a value. Fields removed since the previous write are
        tombstoned (not silently kept), so field deletes replicate."""
        if not key:
            raise ValueError("key is required")
        if not isinstance(value, dict):
            raise ValueError("value must be a dict")
        now = time.time()
        hlc_ts, hlc_count = self._tick(now)
        previous = self.get_record(key)
        prev_fields = set(previous.value) if previous else set()
        clocks: dict[str, dict[str, Any]] = {}
        for fname in value:
            clocks[fname] = {"ts": hlc_ts, "c": hlc_count,
                             "d": self.device_id, "tomb": False}
        for fname in prev_fields - set(value):
            # Field delete: tombstone wins merges until a newer write
            # revives the field.
            clocks[fname] = {"ts": hlc_ts, "c": hlc_count,
                             "d": self.device_id, "tomb": True}
        rec = SyncRecord(
            key=key,
            value=dict(value),
            updated_at=updated_at if updated_at is not None else now,
            device_id=self.device_id,
            deleted=False,
            hlc_ts=hlc_ts,
            hlc_count=hlc_count,
            clocks=clocks,
        )
        self._upsert(rec)
        self._notify(rec)
        return rec

    def put_many(self, mapping: dict[str, dict[str, Any]]) -> list[SyncRecord]:
        """Batch :meth:`put`. One HLC tick per key (causal order kept)."""
        return [self.put(k, v) for k, v in mapping.items()]

    def delete(self, key: str) -> SyncRecord:
        """Write a tombstone. The key disappears from reads but replicates."""
        if not key:
            raise ValueError("key is required")
        hlc_ts, hlc_count = self._tick()
        rec = SyncRecord(
            key=key, value={}, updated_at=time.time(),
            device_id=self.device_id, deleted=True,
            hlc_ts=hlc_ts, hlc_count=hlc_count, clocks={},
        )
        self._upsert(rec)
        self._notify(rec)
        return rec

    # ── reads ────────────────────────────────────────────────────────

    def get(self, key: str) -> SyncRecord | None:
        row = self.db.query_one(f"SELECT * FROM {TABLE} WHERE key=?", (key,))
        if not row or row["deleted"]:
            return None
        return self._row_to_record(row)

    def get_record(self, key: str) -> SyncRecord | None:
        """Like get(), but returns tombstones too (for replication)."""
        row = self.db.query_one(f"SELECT * FROM {TABLE} WHERE key=?", (key,))
        return self._row_to_record(row) if row else None

    def list_changed_since(self, since: float) -> list[SyncRecord]:
        rows = self.db.query(
            f"SELECT * FROM {TABLE} WHERE updated_at > ? ORDER BY updated_at",
            (since,),
        )
        return [self._row_to_record(r) for r in rows]

    def list_since_seq(self, seq: int,
                       limit: int | None = None) -> list[SyncRecord]:
        """Records written after replication cursor ``seq``, in seq order.

        This is the cursor the sync engine uses. Unlike
        :meth:`list_changed_since` it cannot miss a backdated write
        (explicit old ``updated_at``, phone/cloud clock skew): the cursor
        is a monotonic local sequence, not a timestamp.
        """
        sql = f"SELECT * FROM {TABLE} WHERE seq > ? ORDER BY seq"
        params: tuple[Any, ...] = (seq,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (seq, max(1, int(limit)))
        rows = self.db.query(sql, params)
        return [self._row_to_record(r) for r in rows]

    def count_since_seq(self, seq: int) -> int:
        return int(self.db.scalar(
            f"SELECT COUNT(*) FROM {TABLE} WHERE seq > ?", (seq,),
            default=0))

    def max_seq(self) -> int:
        return int(self.db.scalar(
            f"SELECT COALESCE(MAX(seq), 0) FROM {TABLE}", default=0))

    def keys(self) -> list[str]:
        rows = self.db.query(
            f"SELECT key FROM {TABLE} WHERE deleted=0 ORDER BY key"
        )
        return [r["key"] for r in rows]

    def count(self) -> int:
        rows = self.db.query(
            f"SELECT COUNT(*) AS n FROM {TABLE} WHERE deleted=0"
        )
        return int(rows[0]["n"]) if rows else 0

    def tombstone_count(self) -> int:
        rows = self.db.query(
            f"SELECT COUNT(*) AS n FROM {TABLE} WHERE deleted=1"
        )
        return int(rows[0]["n"]) if rows else 0

    # ── merge ────────────────────────────────────────────────────────

    def apply(self, rec: SyncRecord) -> bool:
        """Merge a remote record. Returns True if it changed local state.

        Same-field conflicts resolve by HLC last-write-wins; *different*
        fields merge, so concurrent edits to different fields of one
        record both survive. Deletes win ties (a tombstone at the same
        clock as an edit keeps the key dead — safer than resurrection).
        """
        if not rec.key:
            raise ValueError("record key is required")
        # Fold the remote clock in first: even a no-op write tells us the
        # peer has seen this far, and our next local write must sort after.
        self._receive(rec.hlc_ts, rec.hlc_count)
        current = self.get_record(rec.key)
        if current is None:
            self._upsert(rec)
            self._notify(rec)
            return True
        # Legacy-shaped record (no HLC, no field clocks — e.g. an old
        # peer or a hand-built record): whole-record LWW exactly like
        # the pre-HLC store, then stamp a synthetic HLC seeded from
        # updated_at so future comparisons are HLC-meaningful.
        if (rec.hlc_ts, rec.hlc_count) == (0.0, 0) and not rec.clocks:
            if SyncRecord.wins(current, rec) is not rec:
                return False
            rec = replace(
                rec,
                hlc_ts=rec.updated_at if rec.updated_at > 0 else time.time(),
                hlc_count=0,
            )
            self._upsert(rec)
            self._notify(rec)
            return True
        # Identical record (e.g., our own push echoed back): no-op.
        if _records_equal(current, rec):
            return False
        merged = _merge_records(current, rec)
        if _records_equal(current, merged):
            return False
        # The merged record sorts after both inputs.
        winner = SyncRecord.wins(current, rec)
        merged.hlc_ts, merged.hlc_count = winner.hlc_ts, winner.hlc_count
        self._upsert(merged)
        self._notify(merged)
        return True

    # ── tombstone garbage collection ─────────────────────────────────

    def gc_tombstones(
        self,
        older_than_s: float = DEFAULT_TOMBSTONE_GRACE_S,
        *,
        apply: bool = False,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Reclaim tombstones older than ``older_than_s``.

        Dry-run by default (``apply=False``): reports what *would* be
        deleted without touching anything. Pass ``apply=True`` to really
        delete.

        Safety note: a tombstone deleted here can never be pulled by a
        peer that hasn't synced past it — such a peer keeps its live copy
        and will resurrect the key on its next push. Only GC when every
        peer syncs more often than the grace period (default 90 days).
        """
        now = time.time() if now is None else now
        cutoff = now - older_than_s
        total = self.tombstone_count()
        rows = self.db.query(
            f"SELECT key FROM {TABLE} WHERE deleted=1 AND updated_at < ?",
            (cutoff,),
        )
        deleted = 0
        if apply and rows:
            with self.db.transaction():
                self.db.executemany(
                    f"DELETE FROM {TABLE} WHERE key = ?",
                    [(r["key"],) for r in rows],
                )
            deleted = len(rows)
            _log.info("sync store: GC'd %d tombstones (cutoff %.0f)",
                      deleted, cutoff)
        return {
            "tombstones_total": total,
            "eligible": len(rows),
            "deleted": deleted,
            "dry_run": not apply,
            "older_than_s": older_than_s,
            "cutoff": cutoff,
        }

    # ── divergence detection (anti-entropy aid) ──────────────────────

    def digest(self) -> str:
        """SHA-256 over the whole keyspace — cheap convergence check.

        Two stores that exchanged all writes have equal digests. Exchange
        digests with a peer first; only run the expensive :meth:`diff_keys`
        when they differ.
        """
        h = hashlib.sha256()
        rows = self.db.query(
            f"SELECT key, hlc_ts, hlc_count, device_id, deleted, value "
            f"FROM {TABLE} ORDER BY key"
        )
        for r in rows:
            h.update(
                f"{r['key']}\x00{r['hlc_ts']:.6f}\x00{r['hlc_count']}"
                f"\x00{r['device_id']}\x00{int(r['deleted'])}\x00".encode()
            )
            h.update(hashlib.sha256(
                (r["value"] or "{}").encode()).digest())
        return h.hexdigest()

    def diff_keys(self, other: "SyncStore") -> dict[str, list[str]]:
        """Compare keyspaces with another store.

        Returns ``{"only_in_self", "only_in_other", "different"}`` — keys
        missing on one side, or present on both with different
        (clock, value, deleted) state. O(n) on both sides; use
        :meth:`digest` as the cheap pre-check.
        """
        def snapshot(store: "SyncStore") -> dict[str, tuple]:
            rows = store.db.query(
                f"SELECT key, hlc_ts, hlc_count, device_id, deleted, value "
                f"FROM {TABLE}"
            )
            snap = {}
            for r in rows:
                vhash = hashlib.sha256(
                    json.dumps(json.loads(r["value"] or "{}"),
                               sort_keys=True).encode()
                ).hexdigest()
                snap[r["key"]] = (round(float(r["hlc_ts"]), 6),
                                  int(r["hlc_count"]), r["device_id"],
                                  int(r["deleted"]), vhash)
            return snap

        mine, theirs = snapshot(self), snapshot(other)
        only_in_self = sorted(k for k in mine if k not in theirs)
        only_in_other = sorted(k for k in theirs if k not in mine)
        different = sorted(k for k in mine
                           if k in theirs and mine[k] != theirs[k])
        return {
            "only_in_self": only_in_self,
            "only_in_other": only_in_other,
            "different": different,
        }

    # ── internals ────────────────────────────────────────────────────

    def _upsert(self, rec: SyncRecord) -> None:
        # Every local write — own put/delete or a peer's record applied here
        # — advances the replication cursor. Pulled records MUST bump the
        # local seq too: that is what lets a hub fan records out to a third
        # device. The write and the seq assignment are one transaction, so
        # concurrent writers can never share or skip a seq.
        with self.db.transaction():
            nxt = int(self.db.scalar(
                f"SELECT COALESCE(MAX(seq), 0) FROM {TABLE}", default=0)) + 1
            self.db.execute(
                f"""INSERT INTO {TABLE}
                        (key, value, updated_at, device_id, deleted, seq,
                         hlc_ts, hlc_count, clocks)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value=excluded.value,
                        updated_at=excluded.updated_at,
                        device_id=excluded.device_id,
                        deleted=excluded.deleted,
                        seq=excluded.seq,
                        hlc_ts=excluded.hlc_ts,
                        hlc_count=excluded.hlc_count,
                        clocks=excluded.clocks""",
                (
                    rec.key,
                    json.dumps(rec.value, default=str),
                    rec.updated_at,
                    rec.device_id,
                    1 if rec.deleted else 0,
                    nxt,
                    rec.hlc_ts,
                    rec.hlc_count,
                    json.dumps(rec.clocks, default=str),
                ),
            )
        rec.seq = nxt

    @staticmethod
    def _row_to_record(row: Any) -> SyncRecord:
        try:
            clocks = json.loads(row.get("clocks") or "{}")
        except Exception:  # noqa: BLE001 - corrupt clocks; treat as none
            clocks = {}
        return SyncRecord(
            key=row["key"],
            value=json.loads(row["value"] or "{}"),
            updated_at=row["updated_at"],
            device_id=row["device_id"],
            deleted=bool(row["deleted"]),
            seq=int(row.get("seq") or 0),
            hlc_ts=float(row.get("hlc_ts") or 0.0),
            hlc_count=int(row.get("hlc_count") or 0),
            clocks=_norm_clocks(clocks),
        )


# ── merge helpers (module-level: testable without a store) ───────────────


def _records_equal(a: SyncRecord, b: SyncRecord) -> bool:
    return (
        a.value == b.value
        and a.deleted == b.deleted
        and (a.hlc_ts, a.hlc_count, a.device_id)
        == (b.hlc_ts, b.hlc_count, b.device_id)
        and a.clocks == b.clocks
    )


def _field_clock(rec: SyncRecord, fname: str) -> tuple[float, int, str] | None:
    c = rec.clocks.get(fname)
    if c is None:
        return None
    return _clock_key(c["ts"], c["c"], c["d"])


def _max_field_clock(rec: SyncRecord) -> tuple[float, int, str]:
    """Newest clock touching this record: any field clock, else its HLC."""
    best = rec.clock_key()
    for c in rec.clocks.values():
        key = _clock_key(c["ts"], c["c"], c["d"])
        if key > best:
            best = key
    return best


def _merge_records(local: SyncRecord, remote: SyncRecord) -> SyncRecord:
    """Field-level LWW merge of two versions of one key.

    - Both live: per-field winner by ``(hlc_ts, hlc_count, device_id)``.
      Field tombstones (from :meth:`SyncStore.put`) delete fields whose
      tombstone clock is newest.
    - One side deleted: the tombstone wins when its clock is >= the live
      side's newest field clock (ties favor delete — safer than
      resurrection); otherwise the key resurrects with merged fields.
    - Both deleted: the newer tombstone wins.
    """
    if local.deleted and remote.deleted:
        winner = SyncRecord.wins(local, remote)
        return SyncRecord(
            key=local.key, value={}, updated_at=winner.updated_at,
            device_id=winner.device_id, deleted=True,
            hlc_ts=winner.hlc_ts, hlc_count=winner.hlc_count, clocks={},
        )

    if local.deleted != remote.deleted:
        gone = local if local.deleted else remote
        live = remote if local.deleted else local
        if gone.clock_key() >= _max_field_clock(live):
            return SyncRecord(
                key=local.key, value={}, updated_at=gone.updated_at,
                device_id=gone.device_id, deleted=True,
                hlc_ts=gone.hlc_ts, hlc_count=gone.hlc_count, clocks={},
            )
        # Resurrect: the live side's fields, merged against nothing
        # (the tombstone carries no fields). apply() stamps the winning
        # record clock afterwards.
        if local.hlc_ts >= remote.hlc_ts:
            hlc_ts, hlc_count = local.hlc_ts, local.hlc_count
        else:
            hlc_ts, hlc_count = remote.hlc_ts, remote.hlc_count
        return SyncRecord(
            key=local.key, value=dict(live.value),
            updated_at=max(local.updated_at, remote.updated_at),
            device_id=live.device_id, deleted=False,
            hlc_ts=hlc_ts, hlc_count=hlc_count,
            clocks=dict(live.clocks),
        )

    # Both live: field-level merge.
    merged_value: dict[str, Any] = dict(local.value)
    merged_clocks: dict[str, dict[str, Any]] = dict(local.clocks)
    fields = set(local.value) | set(remote.value) | set(local.clocks) | set(
        remote.clocks)
    for fname in fields:
        lc = _field_clock(local, fname)
        rc = _field_clock(remote, fname)
        if rc is not None and (lc is None or rc > lc):
            clock = remote.clocks[fname]
            if clock["tomb"]:
                merged_value.pop(fname, None)
            else:
                merged_value[fname] = remote.value[fname]
            merged_clocks[fname] = dict(clock)
        elif lc is not None and local.clocks[fname]["tomb"]:
            merged_value.pop(fname, None)
    winner = SyncRecord.wins(local, remote)
    return SyncRecord(
        key=local.key,
        value=merged_value,
        updated_at=max(local.updated_at, remote.updated_at),
        device_id=winner.device_id,
        deleted=False,
        hlc_ts=winner.hlc_ts,
        hlc_count=winner.hlc_count,
        clocks=merged_clocks,
    )
