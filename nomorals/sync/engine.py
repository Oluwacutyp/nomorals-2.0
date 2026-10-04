"""Sync engine: push/pull replication between stores.

A device pushes its changes to a peer and pulls the peer's changes,
merging with last-write-wins. Progress is tracked per peer as monotonic
sequence cursors (see :mod:`.store`), so sync is incremental and can never
miss a backdated write. The peer is usually the cloud hub; devices never
need to talk to each other directly.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger
from ..storage.db import Database
from .store import SyncRecord, SyncStore

__all__ = ["SyncPeer", "LocalPeer", "SyncEngine", "SyncResult"]

_log = get_logger(__name__)


def _emit(topic: str, data: dict[str, Any]) -> None:
    """Publish a telemetry event. Best-effort: a broken bus or subscriber
    must never break a sync run (fail-open telemetry, fail-closed
    function)."""
    try:
        global_bus.publish(Event(topic=topic, data=data, source=__name__))
    except Exception:  # noqa: BLE001 - telemetry is fail-open
        _log.debug("event %s failed", topic, exc_info=True)

PROGRESS_TABLE = "sync_progress"


class SyncPeer(ABC):
    """What the engine needs from the other side."""

    @abstractmethod
    def push_records(self, records: list[SyncRecord]) -> int:
        """Accept records. Returns count applied."""
        ...

    @abstractmethod
    def fetch_since(self, since: float) -> list[SyncRecord]:
        """Return records changed after ``since`` (timestamp cursor)."""
        ...

    def fetch_since_seq(self, seq: int) -> list[SyncRecord]:
        """Return records written after replication cursor ``seq``.

        Peers backed by a :class:`SyncStore` override this. The default
        raises :exc:`NotImplementedError` and the engine falls back to
        :meth:`fetch_since` — a legacy peer keeps working, it just can't
        see backdated writes.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support seq cursors")


class LocalPeer(SyncPeer):
    """A peer backed by another SyncStore (same machine, or tests)."""

    def __init__(self, store: SyncStore) -> None:
        self.store = store

    def push_records(self, records: list[SyncRecord]) -> int:
        n = 0
        for rec in records:
            if self.store.apply(rec):
                n += 1
        return n

    def fetch_since(self, since: float) -> list[SyncRecord]:
        return self.store.list_changed_since(since)

    def fetch_since_seq(self, seq: int) -> list[SyncRecord]:
        return self.store.list_since_seq(seq)


@dataclass
class SyncResult:
    pushed: int
    pulled: int
    conflicts_resolved: int
    duration_s: float

    def to_dict(self) -> dict:
        return {
            "pushed": self.pushed,
            "pulled": self.pulled,
            "conflicts_resolved": self.conflicts_resolved,
            "duration_s": round(self.duration_s, 3),
        }


class SyncEngine:
    """Incremental two-way sync between a local store and a peer."""

    def __init__(self, db: Database, store: SyncStore) -> None:
        self.db = db
        self.store = store
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.db.execute(
            f"""CREATE TABLE IF NOT EXISTS {PROGRESS_TABLE} (
                peer_id TEXT PRIMARY KEY,
                last_push REAL NOT NULL DEFAULT 0,
                last_pull REAL NOT NULL DEFAULT 0,
                push_seq INTEGER NOT NULL DEFAULT 0,
                pull_seq INTEGER NOT NULL DEFAULT 0
            )"""
        )
        self._ensure_seq_columns()

    def _ensure_seq_columns(self) -> None:
        """Migration: pre-seq progress rows carry timestamp cursors.

        The push cursor is translated once into a seq cursor (max local seq
        at or before the old timestamp mark) — same seq space, exact. The
        pull cursor lives in the *peer's* seq space, which isn't visible
        here, so it resets to 0: the first post-upgrade pull re-fetches from
        the peer's start, and every already-seen record is an idempotent
        no-op on apply. Rows already carrying a seq cursor are untouched.
        """
        from .store import TABLE as STORE_TABLE, ensure_column

        for col in ("push_seq", "pull_seq"):
            ensure_column(self.db, PROGRESS_TABLE, col,
                          "INTEGER NOT NULL DEFAULT 0")
        with self.db.transaction():
            stale = self.db.query(
                f"""SELECT peer_id, last_push, push_seq
                    FROM {PROGRESS_TABLE}
                    WHERE push_seq = 0 AND last_push > 0"""
            )
            for row in stale:
                push_seq = int(self.db.scalar(
                    f"SELECT COALESCE(MAX(seq), 0) FROM {STORE_TABLE} "
                    "WHERE updated_at <= ?",
                    (row["last_push"],), default=0))
                self.db.execute(
                    f"""UPDATE {PROGRESS_TABLE}
                        SET push_seq = ?
                        WHERE peer_id = ?""",
                    (push_seq, row["peer_id"]),
                )
            if stale:
                _log.info("sync: migrated %d push cursors to seq",
                          len(stale))

    def _progress(self, peer_id: str) -> tuple[int, int, float, float]:
        """(push_seq, pull_seq, last_push_ts, last_pull_ts) for a peer."""
        row = self.db.query_one(
            f"SELECT * FROM {PROGRESS_TABLE} WHERE peer_id=?", (peer_id,)
        )
        if not row:
            return 0, 0, 0.0, 0.0
        return (
            int(row.get("push_seq") or 0),
            int(row.get("pull_seq") or 0),
            float(row.get("last_push") or 0.0),
            float(row.get("last_pull") or 0.0),
        )

    def _save_progress(
        self,
        peer_id: str,
        push_seq: int,
        pull_seq: int,
        last_push_ts: float = 0.0,
        last_pull_ts: float = 0.0,
    ) -> None:
        self.db.execute(
            f"""INSERT INTO {PROGRESS_TABLE}
                    (peer_id, last_push, last_pull, push_seq, pull_seq)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(peer_id) DO UPDATE SET
                    last_push=excluded.last_push,
                    last_pull=excluded.last_pull,
                    push_seq=excluded.push_seq,
                    pull_seq=excluded.pull_seq""",
            (peer_id, last_push_ts, last_pull_ts, push_seq, pull_seq),
        )

    def sync(self, peer: SyncPeer, peer_id: str = "hub") -> SyncResult:
        """Push local changes, then pull remote changes. Returns counts.

        The push cursor is always the local seq. The pull cursor is the
        peer's seq when the peer supports it, else the legacy timestamp
        cursor via :meth:`SyncPeer.fetch_since`.
        """
        if not peer_id:
            raise ValueError("peer_id is required")
        started = time.time()
        push_seq, pull_seq, last_push_ts, last_pull_ts = self._progress(peer_id)

        # Push: our writes since the push cursor (seq — never misses a
        # backdated write).
        outgoing = self.store.list_since_seq(push_seq)
        pushed = peer.push_records(outgoing) if outgoing else 0
        new_push_seq = max([r.seq for r in outgoing], default=push_seq)
        # The timestamp marks are informational from here on; seqs are the
        # cursors. Keep them fresh for humans reading the table.
        new_push_ts = max(
            [r.updated_at for r in outgoing], default=last_push_ts)

        # Pull: their changes since the pull cursor; merge locally.
        try:
            incoming = peer.fetch_since_seq(pull_seq)
        except NotImplementedError:
            incoming = peer.fetch_since(last_pull_ts)
            new_pull_seq = pull_seq
            new_pull_ts = max(
                [r.updated_at for r in incoming], default=last_pull_ts)
        else:
            new_pull_seq = max([r.seq for r in incoming], default=pull_seq)
            new_pull_ts = last_pull_ts
        pulled = 0
        conflicts = 0
        for rec in incoming:
            before = self.store.get_record(rec.key)
            if self.store.apply(rec):
                pulled += 1
                if before is not None:
                    conflicts += 1

        self._save_progress(peer_id, new_push_seq, new_pull_seq,
                            new_push_ts, new_pull_ts)
        result = SyncResult(
            pushed=pushed,
            pulled=pulled,
            conflicts_resolved=conflicts,
            duration_s=time.time() - started,
        )
        _log.info(
            "sync with %s: pushed=%d pulled=%d conflicts=%d",
            peer_id, pushed, pulled, conflicts,
        )
        _emit("sync.completed", {
            "peer_id": peer_id,
            "pushed": pushed,
            "pulled": pulled,
            "conflicts_resolved": conflicts,
            "duration_s": result.duration_s,
        })
        return result

    def status(self, peer_id: str = "hub") -> dict:
        push_seq, pull_seq, last_push_ts, last_pull_ts = self._progress(peer_id)
        return {
            "peer_id": peer_id,
            "device_id": self.store.device_id,
            "local_keys": self.store.count(),
            "pending_push": len(self.store.list_since_seq(push_seq)),
            "push_seq": push_seq,
            "pull_seq": pull_seq,
            "last_push": last_push_ts,
            "last_pull": last_pull_ts,
        }
