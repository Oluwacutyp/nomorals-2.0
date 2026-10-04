"""Multi-device sync: last-write-wins state replication.

Layer L5. Each device keeps a :class:`SyncStore` (versioned key-value with
tombstones). Devices exchange records with a hub; conflicts resolve by
``(updated_at, device_id)`` — deterministic, no clocks required beyond
monotonicity per device.
"""

from __future__ import annotations

from .engine import LocalPeer, SyncEngine, SyncPeer
from .errors import SyncError
from .http_peer import HttpSyncPeer
from .store import SyncRecord, SyncStore

__all__ = ["LocalPeer", "SyncEngine", "SyncPeer", "SyncError", "SyncRecord",
           "SyncStore", "HttpSyncPeer"]
