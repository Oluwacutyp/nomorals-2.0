"""Offline mode — MORE_TO_ADD gap 6. Built as FALLBACK, online-first stays primary.

When the device is connected, everything runs online and this
package is inert. When connectivity drops:

* ``queue`` — offline actions are queued (persisted to disk, so they
  survive a restart) instead of failing or being silently dropped.
* ``sync`` — on reconnect, queued actions are merged in FIFO order;
  an action whose base version no longer matches current state is
  flagged as a CONFLICT and never silently clobbers newer state.
* ``matrix`` — the degraded-capability matrix: every feature's
  offline level (``full`` / ``degraded`` / ``unavailable``) as a
  data table, so callers degrade honestly instead of guessing.

No background syncing lives here: the host calls ``sync()`` when it
knows it is back online. This package never phones home, never
retries on its own, and never pretends an offline action succeeded.
"""

from __future__ import annotations

from . import matrix, queue, sync
from .matrix import OFFLINE_MATRIX, features_at, matrix_table, offline_level
from .queue import OfflineAction, OfflineQueue, default_offline_root
from .sync import Conflict, SyncEngine, SyncResult

__all__ = [
    "matrix", "queue", "sync",
    "OFFLINE_MATRIX", "features_at", "matrix_table", "offline_level",
    "OfflineAction", "OfflineQueue", "default_offline_root",
    "Conflict", "SyncEngine", "SyncResult",
]
