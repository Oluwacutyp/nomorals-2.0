"""Mesh node registry: identity, heartbeat, presence.

Each device (phone, cloud worker, laptop) registers as a node and sends
periodic heartbeats. Presence = nodes whose last heartbeat is fresh.

Presence is three-state, Serf/SWIM-inspired:

- **ready** — heartbeat within ``stale_after``.
- **suspect** — missed heartbeats, but not gone long enough to prune.
  A suspect node can still refute the suspicion by heartbeating.
- **gone** — silent past the prune horizon; eligible for deletion.

Nodes also carry Kubernetes-style ``labels`` (``{"gpu": "true",
"region": "eu"}``) so dispatch can select nodes by selector instead of
addressing them by id — the same idea as ``nodeSelector``. ``info`` is
the cheap NodeStatus-lite payload a node may attach to its heartbeat
(load, version, queue depth); it is informational only and never used
for correctness.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.events import Event, global_bus
from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..storage.db import Database
from .errors import NodeUnknown

__all__ = ["MeshNode", "NodeRegistry", "format_nodes_table"]

_log = get_logger(__name__)

TABLE = "mesh_nodes"
DEFAULT_STALE_AFTER = 120.0  # seconds without heartbeat → suspect, not present
DEFAULT_PRUNE_AFTER = 1200.0  # seconds without heartbeat → gone, prunable


def _emit(topic: str, data: dict[str, Any]) -> None:
    try:
        global_bus.publish(Event(topic=topic, data=data, source=__name__))
    except Exception:  # noqa: BLE001 - telemetry is fail-open
        _log.debug("event %s failed", topic, exc_info=True)


def _parse_json_map(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


@dataclass
class MeshNode:
    node_id: str
    name: str
    platform: str  # termux | linux | darwin | windows | android
    capabilities: list[str] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)
    last_seen: float = 0.0
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "platform": self.platform,
            "capabilities": list(self.capabilities),
            "labels": dict(self.labels),
            "info": dict(self.info),
            "last_seen": self.last_seen,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MeshNode":
        """Rebuild from :meth:`to_dict` output (hub HTTP wire format)."""
        labels = data.get("labels") or {}
        return cls(
            node_id=str(data.get("node_id") or ""),
            name=str(data.get("name") or ""),
            platform=str(data.get("platform") or ""),
            capabilities=[str(c) for c in (data.get("capabilities") or [])],
            labels={str(k): str(v) for k, v in dict(labels).items()},
            info=dict(data.get("info") or {}),
            last_seen=float(data.get("last_seen") or 0.0),
            created_at=float(data.get("created_at") or 0.0),
        )

    @property
    def age(self) -> float:
        """Seconds since the last heartbeat."""
        return time.time() - self.last_seen

    def is_fresh(self, stale_after: float = DEFAULT_STALE_AFTER) -> bool:
        """True while the node counts as present."""
        return self.age <= stale_after

    def matches(
        self,
        capabilities: list[str] | None = None,
        labels: dict[str, str] | None = None,
    ) -> bool:
        """Selector match: every requested capability present, every
        requested label equal (Kubernetes nodeSelector semantics)."""
        if capabilities and not all(c in self.capabilities for c in capabilities):
            return False
        if labels and not all(
            str(self.labels.get(k)) == str(v) for k, v in labels.items()
        ):
            return False
        return True

    def presence(self, stale_after: float = DEFAULT_STALE_AFTER,
                 prune_after: float = DEFAULT_PRUNE_AFTER) -> str:
        """ready | suspect | gone — the three-state presence model."""
        if self.age <= stale_after:
            return "ready"
        if self.age <= prune_after:
            return "suspect"
        return "gone"

    @staticmethod
    def _human_age(seconds: float) -> str:
        if seconds < 1:
            return "just now"
        if seconds < 60:
            return f"{int(seconds)}s ago"
        if seconds < 3600:
            return f"{int(seconds // 60)}m ago"
        if seconds < 86400:
            return f"{int(seconds // 3600)}h ago"
        return f"{int(seconds // 86400)}d ago"

    def describe(self) -> str:
        """One-line human summary for CLI/status output."""
        glyph = {"ready": "●", "suspect": "◐", "gone": "○"}[self.presence()]
        caps = ",".join(self.capabilities) if self.capabilities else "—"
        labels = (
            " ".join(f"{k}={v}" for k, v in sorted(self.labels.items()))
            or "—"
        )
        return (
            f"{glyph} {self.name} [{self.node_id[:8]}] "
            f"({self.platform or '?'}) · caps: {caps} · "
            f"labels: {labels} · seen {self._human_age(self.age)}"
        )


def format_nodes_table(nodes: list[MeshNode]) -> str:
    """God-tier presence table for CLI/dashboard rendering."""
    if not nodes:
        return "no mesh nodes registered"
    glyph = {"ready": "●", "suspect": "◐", "gone": "○"}
    rows = []
    for n in nodes:
        state = n.presence()
        rows.append((
            f"{glyph[state]} {state}",
            n.name,
            n.node_id[:8],
            n.platform or "—",
            ",".join(n.capabilities) or "—",
            " ".join(f"{k}={v}" for k, v in sorted(n.labels.items())) or "—",
            MeshNode._human_age(n.age),
        ))
    headers = ("state", "name", "id", "platform", "capabilities", "labels", "seen")
    widths = [
        max(len(str(row[i])) for row in [headers, *rows])
        for i in range(len(headers))
    ]
    lines = [
        "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)),
        "  ".join("─" * w for w in widths),
    ]
    lines.extend(
        "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
        for row in rows
    )
    return "\n".join(lines)


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
        # Later columns: add idempotently for databases created before
        # labels/info existed.
        existing = {r["name"] for r in self.db.table_info(TABLE)}
        if "labels" not in existing:
            self.db.execute(f"ALTER TABLE {TABLE} ADD COLUMN labels TEXT NOT NULL DEFAULT '{{}}'")
        if "info" not in existing:
            self.db.execute(f"ALTER TABLE {TABLE} ADD COLUMN info TEXT NOT NULL DEFAULT '{{}}'")

    def register(
        self,
        name: str,
        platform: str = "",
        capabilities: list[str] | None = None,
        node_id: str | None = None,
        labels: dict[str, str] | None = None,
    ) -> MeshNode:
        """Register a node (idempotent by node_id). Returns the node."""
        nid = node_id or new_id()
        now = time.time()
        existing = self.get(nid)
        labels_json = json.dumps({str(k): str(v) for k, v in (labels or {}).items()})
        if existing is not None:
            # Re-registration refreshes metadata + heartbeat.
            self.db.execute(
                f"""UPDATE {TABLE}
                    SET name=?, platform=?, capabilities=?, labels=?,
                        last_seen=?
                    WHERE node_id=?""",
                (name, platform, json.dumps(capabilities or []),
                 labels_json, now, nid),
            )
            existing.name = name
            existing.platform = platform
            existing.capabilities = capabilities or []
            existing.labels = dict(labels or {})
            existing.last_seen = now
            _emit("mesh.node.reregistered", {"node_id": nid, "name": name})
            return existing
        self.db.insert(
            TABLE,
            {
                "node_id": nid,
                "name": name,
                "platform": platform,
                "capabilities": json.dumps(capabilities or []),
                "labels": labels_json,
                "info": "{}",
                "last_seen": now,
                "created_at": now,
            },
        )
        _log.info("mesh node registered: %s (%s)", nid, name)
        _emit("mesh.node.joined", {"node_id": nid, "name": name,
                                   "platform": platform})
        return MeshNode(
            node_id=nid,
            name=name,
            platform=platform,
            capabilities=capabilities or [],
            labels=dict(labels or {}),
            last_seen=now,
            created_at=now,
        )

    def heartbeat(self, node_id: str, info: dict[str, Any] | None = None) -> None:
        """Record a heartbeat. ``info`` is the NodeStatus-lite payload
        (load, version, queue depth...) stored verbatim. Raises
        NodeUnknown for unregistered nodes."""
        now = time.time()
        if info is None:
            cur = self.db.execute(
                f"UPDATE {TABLE} SET last_seen=? WHERE node_id=?", (now, node_id)
            )
        else:
            cur = self.db.execute(
                f"UPDATE {TABLE} SET last_seen=?, info=? WHERE node_id=?",
                (now, json.dumps(info, default=str), node_id),
            )
        if cur.rowcount == 0:
            raise NodeUnknown(f"node not registered: {node_id}")

    def deregister(self, node_id: str) -> MeshNode:
        """Graceful leave: remove a node immediately instead of waiting
        for the prune horizon. Returns the removed node."""
        node = self.get(node_id)
        if node is None:
            raise NodeUnknown(f"node not registered: {node_id}")
        self.db.execute(f"DELETE FROM {TABLE} WHERE node_id=?", (node_id,))
        _log.info("mesh node deregistered: %s (%s)", node_id, node.name)
        _emit("mesh.node.left", {"node_id": node_id, "name": node.name})
        return node

    def update_labels(self, node_id: str, labels: dict[str, str]) -> MeshNode:
        """Replace a node's labels. Raises NodeUnknown when missing."""
        node = self.get(node_id)
        if node is None:
            raise NodeUnknown(f"node not registered: {node_id}")
        node.labels = {str(k): str(v) for k, v in labels.items()}
        self.db.execute(
            f"UPDATE {TABLE} SET labels=? WHERE node_id=?",
            (json.dumps(node.labels), node_id),
        )
        return node

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

    def list_suspect(
        self,
        stale_after: float = DEFAULT_STALE_AFTER,
        prune_after: float = DEFAULT_PRUNE_AFTER,
    ) -> list[MeshNode]:
        """Nodes in the suspicion window: missed heartbeats long enough to
        stop being *ready*, but not long enough to be *gone*. A suspect
        node refutes the suspicion simply by heartbeating."""
        now = time.time()
        rows = self.db.query(
            f"""SELECT * FROM {TABLE}
                WHERE last_seen < ? AND last_seen >= ?
                ORDER BY last_seen DESC""",
            (now - stale_after, now - prune_after),
        )
        return [self._row_to_node(r) for r in rows]

    def select(
        self,
        capabilities: list[str] | None = None,
        labels: dict[str, str] | None = None,
        *,
        limit: int | None = None,
        stale_after: float = DEFAULT_STALE_AFTER,
    ) -> list[MeshNode]:
        """Capability-aware node selection (Kubernetes nodeSelector style).

        Only *ready* nodes are considered. Results prefer the freshest
        heartbeat first. Raises nothing when empty — the caller decides
        whether empty means broadcast, wait, or fail.
        """
        candidates = [
            n for n in self.list_active(stale_after=stale_after)
            if n.matches(capabilities=capabilities, labels=labels)
        ]
        if limit is not None:
            candidates = candidates[: max(0, limit)]
        return candidates

    def prune(self, stale_after: float = DEFAULT_PRUNE_AFTER) -> int:
        """Delete nodes silent for a long time. Returns rows removed.

        Default horizon is the *gone* threshold, not the *stale*
        threshold: pruning must not delete suspect nodes that may still
        refute.
        """
        cutoff = time.time() - stale_after
        cur = self.db.execute(f"DELETE FROM {TABLE} WHERE last_seen < ?", (cutoff,))
        removed = cur.rowcount
        if removed:
            _emit("mesh.node.pruned", {"removed": removed})
        return removed

    @staticmethod
    def _row_to_node(row: Any) -> MeshNode:
        caps = row["capabilities"]
        return MeshNode(
            node_id=row["node_id"],
            name=row["name"],
            platform=row["platform"],
            capabilities=json.loads(caps) if caps else [],
            labels={str(k): str(v) for k, v in _parse_json_map(row.get("labels")).items()},
            info=_parse_json_map(row.get("info")),
            last_seen=row["last_seen"],
            created_at=row["created_at"],
        )
