"""Mesh node registry: identity, heartbeat, presence.

Each device (phone, cloud worker, laptop) registers as a node and sends
periodic heartbeats. Presence = nodes whose last heartbeat is fresh.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..storage.db import Database
from .errors import NodeUnknown

__all__ = ["MeshNode", "NodeRegistry"]

_log = get_logger(__name__)

TABLE = "mesh_nodes"
DEFAULT_STALE_AFTER = 120.0  # seconds without heartbeat → not present


@dataclass
class MeshNode:
    node_id: str
    name: str
    platform: str  # termux | linux | darwin | windows | android
    capabilities: list[str] = field(default_factory=list)
    last_seen: float = 0.0
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "platform": self.platform,
            "capabilities": list(self.capabilities),
            "last_seen": self.last_seen,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MeshNode":
        """Rebuild from :meth:`to_dict` output (hub HTTP wire format)."""
        return cls(
            node_id=str(data.get("node_id") or ""),
            name=str(data.get("name") or ""),
            platform=str(data.get("platform") or ""),
            capabilities=[str(c) for c in (data.get("capabilities") or [])],
            last_seen=float(data.get("last_seen") or 0.0),
            created_at=float(data.get("created_at") or 0.0),
        )

    @property
    def age(self) -> float:
        """Seconds since the last heartbeat."""
        return time.time() - self.last_seen


class NodeRegistry:
    """Durable registry of mesh nodes."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.db.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                node_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                platform TEXT NOT NULL DEFAULT '',
                capabilities TEXT NOT NULL DEFAULT '[]',
                last_seen REAL NOT NULL DEFAULT 0,
                created_at REAL NOT NULL DEFAULT 0
            )"""
        )

    def register(
        self,
        name: str,
        platform: str = "",
        capabilities: list[str] | None = None,
        node_id: str | None = None,
    ) -> MeshNode:
        """Register a node (idempotent by node_id). Returns the node."""
        nid = node_id or new_id()
        now = time.time()
        existing = self.get(nid)
        if existing is not None:
            # Re-registration refreshes metadata + heartbeat.
            self.db.execute(
                f"""UPDATE {TABLE}
                    SET name=?, platform=?, capabilities=?, last_seen=?
                    WHERE node_id=?""",
                (name, platform, json.dumps(capabilities or []), now, nid),
            )
            existing.name = name
            existing.platform = platform
            existing.capabilities = capabilities or []
            existing.last_seen = now
            return existing
        self.db.insert(
            TABLE,
            {
                "node_id": nid,
                "name": name,
                "platform": platform,
                "capabilities": json.dumps(capabilities or []),
                "last_seen": now,
                "created_at": now,
            },
        )
        _log.info("mesh node registered: %s (%s)", nid, name)
        return MeshNode(
            node_id=nid,
            name=name,
            platform=platform,
            capabilities=capabilities or [],
            last_seen=now,
            created_at=now,
        )

    def heartbeat(self, node_id: str) -> None:
        """Record a heartbeat. Raises NodeUnknown for unregistered nodes."""
        now = time.time()
        cur = self.db.execute(
            f"UPDATE {TABLE} SET last_seen=? WHERE node_id=?", (now, node_id)
        )
        if cur.rowcount == 0:
            raise NodeUnknown(f"node not registered: {node_id}")

    def get(self, node_id: str) -> MeshNode | None:
        row = self.db.query_one(
            f"SELECT * FROM {TABLE} WHERE node_id=?", (node_id,)
        )
        return self._row_to_node(row) if row else None

    def list_active(self, stale_after: float = DEFAULT_STALE_AFTER) -> list[MeshNode]:
        """Nodes with a heartbeat newer than ``stale_after`` seconds."""
        cutoff = time.time() - stale_after
        rows = self.db.query(
            f"SELECT * FROM {TABLE} WHERE last_seen >= ? ORDER BY last_seen DESC",
            (cutoff,),
        )
        return [self._row_to_node(r) for r in rows]

    def list_all(self) -> list[MeshNode]:
        rows = self.db.query(f"SELECT * FROM {TABLE} ORDER BY last_seen DESC")
        return [self._row_to_node(r) for r in rows]

    def prune(self, stale_after: float = DEFAULT_STALE_AFTER * 10) -> int:
        """Delete nodes silent for a long time. Returns rows removed."""
        cutoff = time.time() - stale_after
        cur = self.db.execute(f"DELETE FROM {TABLE} WHERE last_seen < ?", (cutoff,))
        return cur.rowcount

    @staticmethod
    def _row_to_node(row: Any) -> MeshNode:
        caps = row["capabilities"]
        return MeshNode(
            node_id=row["node_id"],
            name=row["name"],
            platform=row["platform"],
            capabilities=json.loads(caps) if caps else [],
            last_seen=row["last_seen"],
            created_at=row["created_at"],
        )
