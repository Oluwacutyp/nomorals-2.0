"""Multi-device sync: HLC-ordered, per-field last-write-wins replication.

Layer L5. Each device keeps a :class:`SyncStore` (versioned key-value with
tombstones). Devices exchange records with a hub; conflicts resolve by
hybrid-logical-clock ``(hlc_ts, hlc_count, device_id)`` — deterministic and
correct under phone clock skew — with per-field merging so concurrent edits
to different fields of one record both survive.
"""

from __future__ import annotations

from .engine import AutoSync, LocalPeer, SyncEngine, SyncPeer, SyncPreview, SyncResult
from .errors import SyncAuthError, SyncConnectionError, SyncError, SyncHubError
from .http_peer import HttpSyncPeer
from .store import SyncRecord, SyncStore

__all__ = [
    "AutoSync", "LocalPeer", "SyncEngine", "SyncPeer", "SyncPreview",
    "SyncResult", "SyncError", "SyncAuthError", "SyncConnectionError",
    "SyncHubError", "SyncRecord", "SyncStore", "HttpSyncPeer",
]

# Tier 3: sync namespaces — typed replicated state with visible conflicts.
from .namespaces import Conflict, Namespace, NamespaceSet, merge_namespaces

__all__ += ["Conflict", "Namespace", "NamespaceSet", "merge_namespaces"]
