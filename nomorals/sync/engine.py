"""Sync engine: push/pull replication between stores.

A device pushes its changes to a peer and pulls the peer's changes,
merging with last-write-wins. Progress is tracked per peer so sync is
incremental. The peer is usually the cloud hub; devices never need to
talk to each other directly.
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
        """Return records changed after ``since``."""
        ...


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
                last_pull REAL NOT NULL DEFAULT 0
            )"""
        )

    def _progress(self, peer_id: str) -> tuple[float, float]:
        row = self.db.query_one(
            f"SELECT * FROM {PROGRESS_TABLE} WHERE peer_id=?", (peer_id,)
        )
        if not row:
            return 0.0, 0.0
        return float(row["last_push"]), float(row["last_pull"])

    def _save_progress(
        self, peer_id: str, last_push: float, last_pull: float
    ) -> None:
        self.db.execute(
            f"""INSERT INTO {PROGRESS_TABLE} (peer_id, last_push, last_pull)
                VALUES (?, ?, ?)
                ON CONFLICT(peer_id) DO UPDATE SET
                    last_push=excluded.last_push,
                    last_pull=excluded.last_pull""",
            (peer_id, last_push, last_pull),
        )

    def sync(self, peer: SyncPeer, peer_id: str = "hub") -> SyncResult:
        """Push local changes, then pull remote changes. Returns counts."""
        if not peer_id:
            raise ValueError("peer_id is required")
        started = time.time()
        last_push, last_pull = self._progress(peer_id)

        # Push: our changes since last push.
        outgoing = self.store.list_changed_since(last_push)
        pushed = peer.push_records(outgoing) if outgoing else 0
        new_push_mark = max(
            [r.updated_at for r in outgoing], default=last_push
        )

        # Pull: their changes since last pull; merge locally.
        incoming = peer.fetch_since(last_pull)
        pulled = 0
        conflicts = 0
        for rec in incoming:
            before = self.store.get_record(rec.key)
            if self.store.apply(rec):
                pulled += 1
                if before is not None:
                    conflicts += 1
        new_pull_mark = max(
            [r.updated_at for r in incoming], default=last_pull
        )

        self._save_progress(peer_id, new_push_mark, new_pull_mark)
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
        last_push, last_pull = self._progress(peer_id)
        return {
            "peer_id": peer_id,
            "device_id": self.store.device_id,
            "local_keys": self.store.count(),
            "pending_push": len(self.store.list_changed_since(last_push)),
            "last_push": last_push,
            "last_pull": last_pull,
        }
