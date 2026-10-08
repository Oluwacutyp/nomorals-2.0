"""Live world-graph planning substrate (o9 knowledge-graph pattern).

Build-map #89 — the highest-leverage architectural steal in wave 12.

A live graph of the owner's world: people, projects, commitments,
assets, schedules, deadlines as nodes; ``depends_on``, ``blocks``,
``owned_by``, ``due`` as edges.  Disruptions propagate automatically:
"flight cancelled → all dependent meetings/events flagged".

This module PROJECTS memory/people/goals/scheduler state into a graph —
it does not replace memory.  ``sync_from_memory()`` reads; it never
writes back.

First consumer: scheduler + proactivity — "your 2pm depends on a Lagos
flight that just delayed".

Profile gating: in-memory SQLite on termux (no disk churn on the phone),
persistent file on laptop/workstation.

Every public method never raises.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "NodeType",
    "EdgeType",
    "GraphNode",
    "GraphEdge",
    "WorldGraph",
    "control_graph",
    "disruption_alerts",
    "GRAPH_DISCLAIMER",
]

GRAPH_DISCLAIMER = (
    "The world graph is a planning projection — it reflects what you told "
    "Devon, not ground truth. Verify consequential disruptions yourself."
)

NODE_TYPES = ("person", "project", "commitment", "asset", "schedule", "deadline")
EDGE_TYPES = ("depends_on", "blocks", "owned_by", "due")


class NodeType(str):
    """Node type names (plain strings; use the NODE_TYPES tuple)."""


class EdgeType(str):
    """Edge type names (plain strings; use the EDGE_TYPES tuple)."""


@dataclass
class GraphNode:
    node_id: str = ""
    type: str = ""          # person | project | commitment | asset | schedule | deadline
    label: str = ""
    attrs: dict[str, Any] = field(default_factory=dict)  # due_ts, location, note, ...
    disrupted: bool = False
    external_ref: str = ""  # e.g. "memory:<record_id>" — for idempotent projection
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class GraphEdge:
    from_id: str = ""
    to_id: str = ""
    type: str = ""          # depends_on | blocks | owned_by | due
    created_at: float = 0.0


def _default_db_path() -> str:
    home = os.path.expanduser("~")
    return os.path.join(home, ".nomorals", "planning", "world_graph.db")


def _is_termux() -> bool:
    try:
        import sys
        prefix = (getattr(sys, "prefix", "") or "").lower()
        return "termux" in prefix or "com.termux" in os.environ.get("HOME", "")
    except Exception:  # noqa: BLE001
        return False


class WorldGraph:
    """The live graph of the owner's world. Never raises."""

    def __init__(self, db_path: str = "", *, profile: str = "") -> None:
        self._in_memory = (profile == "termux") or (not profile and _is_termux())
        path = ":memory:" if self._in_memory else (db_path or _default_db_path())
        self._db: sqlite3.Connection | None = None
        try:
            if not self._in_memory:
                os.makedirs(os.path.dirname(path), exist_ok=True)
            self._db = sqlite3.connect(path, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS graph_nodes (
                       node_id TEXT PRIMARY KEY, type TEXT, label TEXT,
                       attrs TEXT, disrupted INTEGER DEFAULT 0,
                       external_ref TEXT, created_at REAL, updated_at REAL)"""
            )
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS graph_edges (
                       from_id TEXT, to_id TEXT, type TEXT,
                       created_at REAL,
                       PRIMARY KEY (from_id, to_id, type))"""
            )
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS gn_type ON graph_nodes(type)"
            )
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS ge_to ON graph_edges(to_id)"
            )
            self._db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS gn_ext ON graph_nodes(external_ref)"
            )
            self._db.commit()
        except Exception:  # noqa: BLE001 — a bad DB path is an empty graph
            _log.warning("world_graph: db unavailable, running empty", exc_info=True)
            self._db = None

    # ── nodes ─────────────────────────────────────────────────────────

    def add_node(self, type: str, label: str,
                 attrs: dict[str, Any] | None = None,
                 external_ref: str = "") -> GraphNode | None:
        """Add a node. Never raises; returns None on bad input/db."""
        try:
            type = (type or "").strip().lower()
            label = (label or "").strip()[:200]
            if type not in NODE_TYPES or not label:
                return None
            if self._db is None:
                return None
            if external_ref:
                row = self._db.execute(
                    "SELECT * FROM graph_nodes WHERE external_ref = ?",
                    (external_ref,)).fetchone()
                if row:
                    return self._row_to_node(row)
            node = GraphNode(
                node_id="wg_" + uuid.uuid4().hex[:10],
                type=type, label=label,
                attrs=dict(attrs or {}),
                external_ref=external_ref or "",
                created_at=time.time(), updated_at=time.time(),
            )
            self._db.execute(
                "INSERT INTO graph_nodes VALUES (?,?,?,?,?,?,?,?)",
                (node.node_id, node.type, node.label,
                 json.dumps(node.attrs), 0,
                 external_ref if external_ref else None,
                 node.created_at, node.updated_at))
            self._db.commit()
            return node
        except Exception:  # noqa: BLE001
            _log.debug("add_node failed", exc_info=True)
            return None

    def get(self, node_id: str) -> GraphNode | None:
        try:
            if self._db is None:
                return None
            row = self._db.execute(
                "SELECT * FROM graph_nodes WHERE node_id = ?",
                (node_id,)).fetchone()
            return self._row_to_node(row) if row else None
        except Exception:  # noqa: BLE001
            return None

    def list_nodes(self, type: str = "") -> list[GraphNode]:
        try:
            if self._db is None:
                return []
            if type:
                rows = self._db.execute(
                    "SELECT * FROM graph_nodes WHERE type = ? ORDER BY label",
                    (type,)).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT * FROM graph_nodes ORDER BY type, label").fetchall()
            return [self._row_to_node(r) for r in rows]
        except Exception:  # noqa: BLE001
            return []

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its edges. Never raises."""
        try:
            if self._db is None:
                return False
            self._db.execute("DELETE FROM graph_edges WHERE from_id = ? OR to_id = ?",
                             (node_id, node_id))
            cur = self._db.execute("DELETE FROM graph_nodes WHERE node_id = ?",
                                   (node_id,))
            self._db.commit()
            return cur.rowcount > 0
        except Exception:  # noqa: BLE001
            return False

    # ── edges ─────────────────────────────────────────────────────────

    def add_edge(self, from_id: str, to_id: str, type: str) -> bool:
        """Add an edge. ``A depends_on B``: A is affected if B is disrupted.
        ``A blocks B``: if A is disrupted, B is affected. Never raises."""
        try:
            type = (type or "").strip().lower()
            if type not in EDGE_TYPES or not from_id or not to_id:
                return False
            if from_id == to_id:
                return False
            if self._db is None:
                return False
            if self.get(from_id) is None or self.get(to_id) is None:
                return False
            self._db.execute(
                "INSERT OR IGNORE INTO graph_edges VALUES (?,?,?,?)",
                (from_id, to_id, type, time.time()))
            self._db.commit()
            return True
        except Exception:  # noqa: BLE001
            return False

    def edges(self, node_id: str = "") -> list[GraphEdge]:
        try:
            if self._db is None:
                return []
            if node_id:
                rows = self._db.execute(
                    "SELECT * FROM graph_edges WHERE from_id = ? OR to_id = ?",
                    (node_id, node_id)).fetchall()
            else:
                rows = self._db.execute("SELECT * FROM graph_edges").fetchall()
            return [GraphEdge(from_id=r["from_id"], to_id=r["to_id"],
                              type=r["type"], created_at=r["created_at"] or 0.0)
                    for r in rows]
        except Exception:  # noqa: BLE001
            return []

    # ── queries ───────────────────────────────────────────────────────

    def dependents(self, node_id: str) -> list[GraphNode]:
        """Nodes that would be affected if this node is disrupted."""
        try:
            affected_ids = self._propagate({node_id}) - {node_id}
            out = []
            for nid in affected_ids:
                n = self.get(nid)
                if n:
                    out.append(n)
            return sorted(out, key=lambda n: n.label)
        except Exception:  # noqa: BLE001
            return []

    def dependencies(self, node_id: str) -> list[GraphNode]:
        """Nodes this node depends on (direct depends_on targets)."""
        try:
            ids = {e.to_id for e in self.edges(node_id)
                   if e.from_id == node_id and e.type == "depends_on"}
            out = [self.get(i) for i in ids]
            return sorted([n for n in out if n], key=lambda n: n.label)
        except Exception:  # noqa: BLE001
            return []

    def what_breaks_if(self, node_id: str) -> dict[str, Any]:
        """Summary of the blast radius — without marking anything."""
        try:
            node = self.get(node_id)
            if node is None:
                return {"ok": False, "reason": "node not found"}
            affected = self.dependents(node_id)
            return {
                "ok": True,
                "node": node.label,
                "count": len(affected),
                "affected": [{"id": n.node_id, "type": n.type, "label": n.label}
                             for n in affected],
            }
        except Exception:  # noqa: BLE001
            return {"ok": False, "reason": "query failed"}

    def critical_path(self) -> list[GraphNode]:
        """Longest depends_on chain among deadline/schedule/commitment nodes.

        The commitments that hold everything else up.
        """
        try:
            nodes = [n for n in self.list_nodes()
                     if n.type in ("deadline", "schedule", "commitment")]
            ids = {n.node_id for n in nodes}
            # longest path in a DAG via memoised DFS; cycles break by visited set
            deps: dict[str, list[str]] = {}
            for n in nodes:
                deps[n.node_id] = [e.to_id for e in self.edges(n.node_id)
                                   if e.from_id == n.node_id
                                   and e.type == "depends_on"
                                   and e.to_id in ids]

            memo: dict[str, list[str]] = {}

            def longest(nid: str, seen: frozenset) -> list[str]:
                if nid in memo:
                    return memo[nid]
                best: list[str] = [nid]
                for d in deps.get(nid, []):
                    if d in seen:
                        continue
                    cand = [nid] + longest(d, seen | {nid})
                    if len(cand) > len(best):
                        best = cand
                memo[nid] = best
                return best

            best_path: list[str] = []
            for n in nodes:
                p = longest(n.node_id, frozenset())
                if len(p) > len(best_path):
                    best_path = p
            by_id = {n.node_id: n for n in nodes}
            return [by_id[i] for i in best_path if i in by_id]
        except Exception:  # noqa: BLE001
            return []

    # ── disruption ────────────────────────────────────────────────────

    def _propagate(self, seeds: set[str]) -> set[str]:
        """BFS: seeds → everything affected via depends_on / blocks."""
        try:
            affected = set(seeds)
            frontier = list(seeds)
            all_edges = self.edges()
            while frontier:
                cur = frontier.pop()
                for e in all_edges:
                    nxt = None
                    # A depends_on B, B disrupted → A affected
                    if e.type == "depends_on" and e.to_id == cur:
                        nxt = e.from_id
                    # A blocks B, A disrupted → B affected
                    elif e.type == "blocks" and e.from_id == cur:
                        nxt = e.to_id
                    if nxt and nxt not in affected:
                        affected.add(nxt)
                        frontier.append(nxt)
            return affected
        except Exception:  # noqa: BLE001
            return set(seeds)

    def mark_disrupted(self, node_id: str,
                       *, note: str = "") -> list[GraphNode]:
        """Mark a node disrupted and propagate. Returns affected nodes.

        "Flight cancelled → all dependent meetings/events flagged."
        """
        try:
            node = self.get(node_id)
            if node is None or self._db is None:
                return []
            affected_ids = self._propagate({node_id})
            now = time.time()
            for nid in affected_ids:
                attrs = dict(self.get(nid).attrs) if self.get(nid) else {}
                if note:
                    attrs["disruption_note"] = (note or "")[:300]
                self._db.execute(
                    "UPDATE graph_nodes SET disrupted = 1, attrs = ?, updated_at = ? "
                    "WHERE node_id = ?",
                    (json.dumps(attrs), now, nid))
            self._db.commit()
            out = []
            for nid in affected_ids:
                n = self.get(nid)
                if n:
                    out.append(n)
            return sorted(out, key=lambda n: n.label)
        except Exception:  # noqa: BLE001
            _log.debug("mark_disrupted failed", exc_info=True)
            return []

    def clear_disruption(self, node_id: str) -> bool:
        """Clear the disrupted flag on one node. Never raises."""
        try:
            if self._db is None:
                return False
            cur = self._db.execute(
                "UPDATE graph_nodes SET disrupted = 0, updated_at = ? WHERE node_id = ?",
                (time.time(), node_id))
            self._db.commit()
            return cur.rowcount > 0
        except Exception:  # noqa: BLE001
            return False

    def disrupted(self) -> list[GraphNode]:
        """All currently disrupted nodes."""
        try:
            return [n for n in self.list_nodes() if n.disrupted]
        except Exception:  # noqa: BLE001
            return []

    # ── projection (read-only from memory/scheduler) ──────────────────

    def sync_from_memory(self, memory: Any = None) -> int:
        """Project memory facts → graph nodes. Reads only; never writes to
        memory. Idempotent via ``external_ref = "memory:<record_id>"``.

        ``memory`` may be a MemoryManager-like object with ``recall(query,
        limit=)``; when omitted, a default manager is attempted. Returns the
        number of nodes upserted. Never raises.
        """
        try:
            mgr = memory
            if mgr is None:
                try:
                    from ..memory.manager import MemoryManager
                    mgr = MemoryManager()
                except Exception:  # noqa: BLE001
                    return 0
            recall = getattr(mgr, "recall", None)
            if not callable(recall):
                return 0
            result = recall("upcoming commitments deadlines people projects flights",
                            limit=60)
            records = getattr(result, "records", None) or []
            count = 0
            for rec in records:
                try:
                    node = self._record_to_node(rec)
                    if node:
                        count += 1
                except Exception:  # noqa: BLE001
                    continue
            return count
        except Exception:  # noqa: BLE001
            _log.debug("sync_from_memory failed", exc_info=True)
            return 0

    def _record_to_node(self, rec: Any) -> GraphNode | None:
        """Map one memory record → a graph node (or None to skip)."""
        try:
            text = str(getattr(rec, "text", "") or getattr(rec, "content", "") or "")
            if not text.strip():
                return None
            kind = str(getattr(rec, "kind", "") or "").lower()
            tags = str(getattr(rec, "tags", "") or "").lower()
            blob = f"{kind} {tags}"
            if any(k in blob for k in ("deadline", "due", "expir")):
                ntype = "deadline"
            elif any(k in blob for k in ("flight", "meeting", "event", "appointment",
                                         "trip", "schedule", "calendar")):
                ntype = "schedule"
            elif any(k in blob for k in ("goal", "project", "mission", "plan")):
                ntype = "project"
            elif any(k in blob for k in ("person", "people", "contact", "family")):
                ntype = "person"
            elif any(k in blob for k in ("asset", "account", "subscription",
                                         "contract", "device")):
                ntype = "asset"
            else:
                ntype = "commitment"
            label = text.strip().split("\n")[0][:140]
            rec_id = str(getattr(rec, "record_id", "") or getattr(rec, "id", "") or "")
            attrs = {
                "source": "memory",
                "kind": kind or "",
                "tags": tags or "",
            }
            created = getattr(rec, "created_at", None)
            if created:
                try:
                    attrs["memory_created"] = float(created)
                except (TypeError, ValueError):
                    pass
            return self.add_node(ntype, label, attrs,
                                 external_ref=f"memory:{rec_id}" if rec_id else "")
        except Exception:  # noqa: BLE001
            return None

    def sync_from_tasks(self, tasks: list[dict[str, Any]] | None) -> int:
        """Project scheduled tasks → schedule nodes. ``tasks`` are plain dicts
        with at least ``label``/``action`` and optional ``run_at``/``task_id``.
        Idempotent via ``external_ref = "task:<id>"``. Never raises."""
        try:
            count = 0
            for t in tasks or []:
                label = str(t.get("label") or t.get("action") or "").strip()[:140]
                if not label:
                    continue
                tid = str(t.get("task_id") or t.get("id") or "")
                attrs = {"source": "scheduler"}
                for k in ("run_at", "next_run", "cron_expr", "status", "action"):
                    if t.get(k) is not None:
                        attrs[k] = t[k]
                node = self.add_node("schedule", label, attrs,
                                     external_ref=f"task:{tid}" if tid else "")
                if node:
                    count += 1
            return count
        except Exception:  # noqa: BLE001
            return 0

    # ── helpers ───────────────────────────────────────────────────────

    def _row_to_node(self, row: sqlite3.Row) -> GraphNode:
        try:
            attrs = json.loads(row["attrs"] or "{}")
        except (TypeError, ValueError):
            attrs = {}
        return GraphNode(
            node_id=row["node_id"], type=row["type"] or "",
            label=row["label"] or "", attrs=attrs,
            disrupted=bool(row["disrupted"]),
            external_ref=row["external_ref"] or "",
            created_at=row["created_at"] or 0.0,
            updated_at=row["updated_at"] or 0.0)

    def summary(self) -> str:
        """One-line world status."""
        try:
            nodes = self.list_nodes()
            edges = self.edges()
            dis = [n for n in nodes if n.disrupted]
            return (f"🌐 world graph: {len(nodes)} nodes, {len(edges)} edges"
                    + (f", ⚠️ {len(dis)} disrupted" if dis else ""))
        except Exception:  # noqa: BLE001
            return "🌐 world graph: unavailable"


def disruption_alerts(graph: WorldGraph,
                       tasks: list[dict[str, Any]] | None = None) -> list[str]:
    """First consumer seam: scheduler + proactivity.

    Matches disrupted nodes against upcoming task labels and returns plain
    alert strings like "your 2pm depends on a Lagos flight that just
    delayed". ``tasks`` are plain dicts (label/action/run_at). Never raises.
    """
    try:
        alerts: list[str] = []
        dis = graph.disrupted()
        if not dis:
            return []
        for task in tasks or []:
            label = str(task.get("label") or task.get("action") or "").lower()
            if not label:
                continue
            for node in dis:
                words = [w for w in node.label.lower().split() if len(w) > 3]
                if any(w in label for w in words):
                    run = task.get("run_at") or task.get("next_run")
                    when = ""
                    if run:
                        try:
                            import datetime as _dt
                            when = _dt.datetime.fromtimestamp(float(run)).strftime("%H:%M")
                        except (TypeError, ValueError):
                            when = ""
                    alerts.append(
                        f"⚠️ heads-up{(' — your ' + when) if when else ''}: "
                        f"'{task.get('label') or task.get('action')}' depends on "
                        f"'{node.label}', which is disrupted"
                        + (f" ({node.attrs.get('disruption_note')})"
                           if node.attrs.get("disruption_note") else "")
                        + ".")
                    break
        return alerts
    except Exception:  # noqa: BLE001
        return []


# ── chat ─────────────────────────────────────────────────────────────

def _usage() -> str:
    return (
        "🌐 /graph — live world-graph planning substrate\n"
        "  add <type> <label>          — node (person|project|commitment|asset|schedule|deadline)\n"
        "  link <from-id> <to-id> <type> — edge (depends_on|blocks|owned_by|due)\n"
        "  disrupt <node-id> [note]    — mark disrupted, propagate to dependents\n"
        "  clear <node-id>             — clear a disruption\n"
        "  show [node-id]              — summary, or one node + dependents\n"
        "  breaks <node-id>            — what breaks if this node fails\n"
        "  path                        — critical path (longest dependency chain)\n"
        "  sync                        — project memory + tasks into the graph"
    )


def control_graph(tail: str, context: Any = None, chat: Any = None,
                  graph: WorldGraph | None = None) -> str:
    """``/graph`` chat control (owner-only at dispatch). Never raises."""
    try:
        g = graph or WorldGraph()
        parts = (tail or "").strip().split(None, 1)
        if not parts:
            return g.summary() + "\n" + _usage() + "\n" + GRAPH_DISCLAIMER
        cmd, rest = parts[0].lower(), (parts[1] if len(parts) > 1 else "")

        if cmd == "add":
            bits = rest.split(None, 1)
            if len(bits) < 2:
                return "usage: /graph add <type> <label>"
            node = g.add_node(bits[0], bits[1])
            if node is None:
                return f"couldn't add that node — type must be one of: {', '.join(NODE_TYPES)}"
            return f"✅ added [{node.type}] {node.label}\n`{node.node_id}`"

        if cmd == "link":
            bits = rest.split()
            if len(bits) < 3:
                return "usage: /graph link <from-id> <to-id> <type>"
            ok = g.add_edge(bits[0], bits[1], bits[2])
            if not ok:
                return (f"couldn't link — check both ids exist and type is one of: "
                        f"{', '.join(EDGE_TYPES)}")
            frm = g.get(bits[0])
            to = g.get(bits[1])
            return (f"✅ linked: '{frm.label if frm else bits[0]}' "
                    f"{bits[2].lower()} '{to.label if to else bits[1]}'")

        if cmd == "disrupt":
            bits = rest.split(None, 1)
            if not bits:
                return "usage: /graph disrupt <node-id> [note]"
            affected = g.mark_disrupted(bits[0], note=bits[1] if len(bits) > 1 else "")
            if not affected:
                return "node not found."
            lines = [f"⚠️ disrupted: {n.label} [{n.type}]" for n in affected]
            return "\n".join(lines) + f"\n{len(affected)} node(s) affected.\n" + GRAPH_DISCLAIMER

        if cmd == "clear":
            if not rest.strip():
                return "usage: /graph clear <node-id>"
            return ("✅ disruption cleared."
                    if g.clear_disruption(rest.strip())
                    else "node not found.")

        if cmd == "show":
            if rest.strip():
                node = g.get(rest.strip())
                if node is None:
                    return "node not found."
                deps = g.dependencies(node.node_id)
                affected = g.dependents(node.node_id)
                out = [f"🧩 [{node.type}] {node.label} `{'⚠️ disrupted' if node.disrupted else 'ok'}`"]
                if node.attrs.get("disruption_note"):
                    out.append(f"   note: {node.attrs['disruption_note']}")
                if deps:
                    out.append("   depends on: " + ", ".join(d.label for d in deps))
                if affected:
                    out.append("   affects: " + ", ".join(n.label for n in affected))
                return "\n".join(out)
            return g.summary()

        if cmd == "breaks":
            if not rest.strip():
                return "usage: /graph breaks <node-id>"
            r = g.what_breaks_if(rest.strip())
            if not r.get("ok"):
                return "node not found."
            lines = [f"💥 if '{r['node']}' fails → {r['count']} node(s) affected:"]
            lines += [f"   • [{a['type']}] {a['label']}" for a in r["affected"][:15]]
            return "\n".join(lines) + "\n" + GRAPH_DISCLAIMER

        if cmd == "path":
            path = g.critical_path()
            if not path:
                return "no dependency chain found yet — add nodes and depends_on links."
            return ("🛤️ critical path:\n" +
                    "\n".join(f"   {i+1}. [{n.type}] {n.label}"
                              for i, n in enumerate(path)))

        if cmd == "sync":
            count = g.sync_from_memory()
            return f"🔄 projected {count} node(s) from memory into the graph (read-only).\n" + GRAPH_DISCLAIMER

        return _usage()
    except Exception as e:  # noqa: BLE001
        return f"graph hiccup: {e}"
