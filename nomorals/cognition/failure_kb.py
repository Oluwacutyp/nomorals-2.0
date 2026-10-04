"""Operator-facing knowledge base over trajectory failure clusters.

:class:`FailureKB` is thin by design: clustering lives in
:mod:`nomorals.cognition.trajectories`; this module adds human
operator notes on top of the clusters and surfaces the recurring
ones (the "fix me" list).

Layer rule: L3 — only :mod:`nomorals.core` (layer 1) and
:mod:`nomorals.storage` (layer 2) are imported.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from ..core.ids import new_short_id
from ..storage.db import Database
from .trajectories import TrajectoryStore

_log = logging.getLogger(__name__)

__all__ = ["FailureKB"]

_NOTES_DDL = """
CREATE TABLE IF NOT EXISTS cog_cluster_notes (
    id          TEXT PRIMARY KEY,
    cluster_key TEXT NOT NULL DEFAULT '',
    note_text   TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cog_notes_cluster
    ON cog_cluster_notes(cluster_key, created_at);
"""

_REPEATED_THRESHOLD = 3


class FailureKB:
    """Attach operator notes to failure clusters and surface recurring ones."""

    def __init__(
        self,
        store: TrajectoryStore | None = None,
        db: Database | str | Path | None = None,
    ) -> None:
        if store is not None and db is not None:
            raise ValueError(
                "pass either store or db, not both — they would point at "
                "different databases and notes would land in the wrong one"
            )
        self.store = store if store is not None else TrajectoryStore(db)
        self.db = self.store.db
        self.db.executescript(_NOTES_DDL)
        self._lock = threading.RLock()

    # ── notes ────────────────────────────────────────────────────────────
    def note(self, cluster_key: str, note_text: str) -> str:
        """Attach an operator note to a failure cluster.

        ``cluster_key`` is the ``task_kind::error_signature`` key (see
        :func:`nomorals.cognition.trajectories.cluster_key_for`).  Returns
        the note id.
        """
        note_id = new_short_id("note_")
        with self._lock:
            self.db.execute(
                "INSERT INTO cog_cluster_notes (id, cluster_key, note_text,"
                " created_at) VALUES (?, ?, ?, ?)",
                (
                    note_id,
                    cluster_key or "",
                    note_text or "",
                    time.time(),
                ),
            )
        _log.info("failure-kb note %s on %s", note_id, cluster_key)
        return note_id

    def note_for_error(self, task_kind: str, error: str,
                       note_text: str) -> str:
        """Attach a note given a raw error string (signature normalized).

        Broker convenience: pass the raw ``error`` exactly as recorded;
        the cluster key is derived with the same normalization used by
        :meth:`TrajectoryStore.failure_clusters`, so the note lands on
        the right cluster.
        """
        from .trajectories import cluster_key_for, normalize_error

        return self.note(
            cluster_key_for(task_kind, normalize_error(error)), note_text
        )

    def _notes_for(self, cluster_key: str) -> list[dict]:
        rows = self.db.query(
            "SELECT id, note_text, created_at FROM cog_cluster_notes"
            " WHERE cluster_key = ? ORDER BY created_at DESC",
            (cluster_key or "",),
        )
        return [
            {
                "id": row["id"],
                "note_text": row["note_text"],
                "created_at": float(row["created_at"]),
            }
            for row in rows
        ]

    # ── lookups ──────────────────────────────────────────────────────────
    def lookup(self, task_kind: str) -> list[dict]:
        """Failure clusters for ``task_kind``, each with attached notes.

        Each dict carries the store's cluster fields (``task_kind``,
        ``error_signature``, ``count``, ``last_seen``, ``example_ids``)
        plus ``cluster_key`` and ``notes`` (newest first).  Empty when
        the store has no failures for the kind.
        """
        from .trajectories import cluster_key_for

        clusters = self.store.failure_clusters(task_kind, limit=10_000)
        for cluster in clusters:
            key = cluster_key_for(
                cluster["task_kind"], cluster["error_signature"]
            )
            cluster["cluster_key"] = key
            cluster["notes"] = self._notes_for(key)
        return clusters

    def repeated(self, limit: int = 10) -> list[dict]:
        """Clusters with ``count >= 3`` — the "fix me" list.

        Ordered most-frequent first, truncated to ``limit``.  Notes are
        included via :meth:`lookup` semantics.
        """
        from .trajectories import cluster_key_for

        clusters = self.store.failure_clusters(None, limit=10_000)
        out = []
        for cluster in clusters:
            if cluster["count"] < _REPEATED_THRESHOLD:
                continue
            key = cluster_key_for(
                cluster["task_kind"], cluster["error_signature"]
            )
            cluster["cluster_key"] = key
            cluster["notes"] = self._notes_for(key)
            out.append(cluster)
            if len(out) >= max(0, int(limit)):
                break
        return out
