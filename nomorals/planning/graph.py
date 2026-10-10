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
        self._in_memory = (db_path == ":memory:") or (profile == "termux") or (
            not profile and _is_termux())
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

    # ── structural analysis (mined: Neo4j GDS centrality, supply-chain
    # articulation analysis, workflow-intelligence path queries) ───────────

    def _undirected_adj(self) -> dict[str, set[str]]:
        """Undirected adjacency over dependency edges (depends_on + blocks)."""
        adj: dict[str, set[str]] = {}
        try:
            for e in self.edges():
                if e.type not in ("depends_on", "blocks"):
                    continue
                adj.setdefault(e.from_id, set()).add(e.to_id)
                adj.setdefault(e.to_id, set()).add(e.from_id)
        except Exception:  # noqa: BLE001
            pass
        return adj

    def _directed_adj(self) -> dict[str, set[str]]:
        """Directed adjacency: u -> v means u must come before v
        (i.e. v depends_on u; u blocks v ⇒ u before v)."""
        adj: dict[str, set[str]] = {}
        try:
            for e in self.edges():
                if e.type == "depends_on":
                    adj.setdefault(e.to_id, set()).add(e.from_id)
                elif e.type == "blocks":
                    adj.setdefault(e.from_id, set()).add(e.to_id)
        except Exception:  # noqa: BLE001
            pass
        return adj

    def weak_links(self, limit: int = 8) -> list[dict[str, Any]]:
        """Load-bearing nodes: whose failure fragments the world.

        Betweenness centrality (Brandes, on the undirected dependency
        projection) + fan-in/fan-out. "What should I protect?" — the
        proactive complement to ``dependents()``. Never raises.
        """
        try:
            adj = self._undirected_adj()
            nodes = [n.node_id for n in self.list_nodes() if n.node_id in adj]
            if not nodes:
                return []
            # Brandes' algorithm (unweighted).
            btw: dict[str, float] = {v: 0.0 for v in nodes}
            for s in nodes:
                stack: list[str] = []
                pred: dict[str, list[str]] = {v: [] for v in nodes}
                sigma: dict[str, float] = dict.fromkeys(nodes, 0.0)
                sigma[s] = 1.0
                dist: dict[str, int] = dict.fromkeys(nodes, -1)
                dist[s] = 0
                queue = [s]
                while queue:
                    v = queue.pop(0)
                    stack.append(v)
                    for w in adj.get(v, ()):
                        if w not in dist:
                            continue
                        if dist[w] < 0:
                            queue.append(w)
                            dist[w] = dist[v] + 1
                        if dist[w] == dist[v] + 1:
                            sigma[w] += sigma[v]
                            pred[w].append(v)
                delta: dict[str, float] = dict.fromkeys(nodes, 0.0)
                while stack:
                    w = stack.pop()
                    for v in pred[w]:
                        if sigma[w] > 0:
                            delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
                    if w != s:
                        btw[w] += delta[w]
            n = len(nodes)
            # Raw Brandes here counts ordered pairs, so the undirected
            # normalizer is (n-1)(n-2) — scores stay in [0, 1].
            norm = 1.0 / max(1.0, (n - 1) * (n - 2))
            directed = self._directed_adj()
            by_id = {nd.node_id: nd for nd in self.list_nodes()}
            ranked = []
            for nid in nodes:
                nd = by_id.get(nid)
                fan_out = len(directed.get(nid, ()))
                fan_in = sum(1 for outs in directed.values() if nid in outs)
                ranked.append({
                    "id": nid,
                    "label": nd.label if nd else nid,
                    "type": nd.type if nd else "",
                    "betweenness": round(btw[nid] * norm, 3),
                    "fan_in": fan_in,
                    "fan_out": fan_out,
                    "dependents": len(self._propagate({nid}) - {nid}),
                })
            ranked.sort(key=lambda r: (r["betweenness"], r["dependents"],
                                       r["fan_in"] + r["fan_out"]),
                        reverse=True)
            return ranked[:max(1, int(limit or 8))]
        except Exception:  # noqa: BLE001
            _log.debug("weak_links failed", exc_info=True)
            return []

    def find_cycles(self) -> list[list[str]]:
        """Explicit dependency cycles (canonical, deduped).

        ``critical_path()`` and ``schedule_order()`` silently route around
        cycles; this reports them so they can be fixed. Never raises.
        """
        try:
            adj = self._directed_adj()
            WHITE, GRAY, BLACK = 0, 1, 2
            color: dict[str, int] = {}
            found: list[list[str]] = []
            seen: set[tuple[str, ...]] = set()

            def visit(start: str) -> None:
                stack: list[tuple[str, list[str]]] = [(start, [start])]
                color[start] = GRAY
                while stack:
                    v, path = stack[-1]
                    advanced = False
                    for w in sorted(adj.get(v, ())):
                        c = color.get(w, WHITE)
                        if c == WHITE:
                            color[w] = GRAY
                            stack.append((w, path + [w]))
                            advanced = True
                            break
                        elif c == GRAY and w in path:
                            cyc = path[path.index(w):]
                            # canonical rotation: start at smallest id
                            i = cyc.index(min(cyc))
                            canon = tuple(cyc[i:] + cyc[:i])
                            if canon not in seen:
                                seen.add(canon)
                                found.append(list(canon))
                    if not advanced:
                        color[v] = BLACK
                        stack.pop()

            for nid in sorted(adj):
                if color.get(nid, WHITE) == WHITE:
                    visit(nid)
            return found
        except Exception:  # noqa: BLE001
            _log.debug("find_cycles failed", exc_info=True)
            return []

    def path_between(self, from_id: str, to_id: str) -> list[GraphNode]:
        """Shortest dependency path between two nodes ("how is X connected
        to Y"). BFS over the undirected dependency projection. Never raises.
        """
        try:
            if not from_id or not to_id:
                return []
            adj = self._undirected_adj()
            if from_id not in adj or to_id not in adj:
                # endpoints may exist without dependency edges
                if from_id == to_id and self.get(from_id):
                    return [self.get(from_id)]  # type: ignore[list-item]
                return []
            prev: dict[str, str] = {from_id: ""}
            queue = [from_id]
            while queue:
                v = queue.pop(0)
                if v == to_id:
                    break
                for w in adj.get(v, ()):
                    if w not in prev:
                        prev[w] = v
                        queue.append(w)
            if to_id not in prev:
                return []
            ids: list[str] = []
            cur = to_id
            while cur:
                ids.append(cur)
                cur = prev[cur]
            ids.reverse()
            by_id = {n.node_id: n for n in self.list_nodes()}
            return [by_id[i] for i in ids if i in by_id]
        except Exception:  # noqa: BLE001
            return []

    def neighborhood(self, node_id: str, depth: int = 2) -> list[GraphNode]:
        """k-hop subgraph around a node (dependency edges, undirected).
        Focused views without dumping the whole graph. Never raises."""
        try:
            adj = self._undirected_adj()
            depth = max(0, min(4, int(depth or 0)))
            seen = {node_id}
            frontier = [node_id]
            for _ in range(depth):
                nxt: list[str] = []
                for v in frontier:
                    for w in adj.get(v, ()):
                        if w not in seen:
                            seen.add(w)
                            nxt.append(w)
                frontier = nxt
            by_id = {n.node_id: n for n in self.list_nodes()}
            return sorted((by_id[i] for i in seen if i in by_id),
                          key=lambda n: n.label)
        except Exception:  # noqa: BLE001
            return []

    def schedule_waves(self) -> list[list[GraphNode]]:
        """Topological *layers*: tasks at the same depth have no dependencies
        between them and can run in parallel (the Neo4j maximal-distance
        insight). Cyclic leftovers go in a final "cyclic" wave, reported
        honestly. Never raises.
        """
        try:
            nodes = self.list_nodes()
            by_id = {n.node_id: n for n in nodes}
            indeg: dict[str, int] = {n.node_id: 0 for n in nodes}
            before: dict[str, list[str]] = {n.node_id: [] for n in nodes}
            for e in self.edges():
                if (e.type == "depends_on" and e.from_id in by_id
                        and e.to_id in by_id and e.from_id != e.to_id):
                    before[e.to_id].append(e.from_id)
                    indeg[e.from_id] += 1

            def due_key(n: GraphNode) -> float:
                due = n.attrs.get("due_ts") or n.attrs.get("due")
                try:
                    return float(due) if due else float("inf")
                except (TypeError, ValueError):
                    return float("inf")

            waves: list[list[GraphNode]] = []
            ready = sorted((nid for nid, d in indeg.items() if d == 0),
                           key=lambda nid: due_key(by_id[nid]))
            while ready:
                wave = [by_id[nid] for nid in ready]
                waves.append(wave)
                nxt_ready: list[str] = []
                for nid in ready:
                    for m in before[nid]:
                        indeg[m] -= 1
                        if indeg[m] == 0:
                            nxt_ready.append(m)
                ready = sorted(nxt_ready,
                               key=lambda nid: due_key(by_id[nid]))
            placed = {n.node_id for w in waves for n in w}
            leftover = sorted((n for n in nodes if n.node_id not in placed),
                              key=due_key)
            if leftover:
                waves.append(leftover)  # the cyclic wave
            return waves
        except Exception:  # noqa: BLE001
            _log.debug("schedule_waves failed", exc_info=True)
            return []

    def simulate_disruption(self, node_id: str) -> dict[str, Any]:
        """Dry-run disruption: layered blast-radius tree WITHOUT writing
        state. ``what_breaks_if()`` gives a flat summary; this shows the
        shape (depth layers) so the cascade is legible. Never raises.
        """
        try:
            node = self.get(node_id)
            if node is None:
                return {"ok": False, "reason": "node not found"}
            # BFS in propagation order, recording depth.
            depth: dict[str, int] = {node_id: 0}
            frontier = [node_id]
            all_edges = self.edges()
            while frontier:
                cur = frontier.pop(0)
                for e in all_edges:
                    nxt = None
                    if e.type == "depends_on" and e.to_id == cur:
                        nxt = e.from_id
                    elif e.type == "blocks" and e.from_id == cur:
                        nxt = e.to_id
                    if nxt and nxt not in depth:
                        depth[nxt] = depth[cur] + 1
                        frontier.append(nxt)
            by_id = {n.node_id: n for n in self.list_nodes()}
            layers: dict[int, list[str]] = {}
            for nid, d in depth.items():
                if nid == node_id:
                    continue
                layers.setdefault(d, []).append(
                    by_id[nid].label if nid in by_id else nid)
            return {
                "ok": True,
                "node": node.label,
                "total_affected": len(depth) - 1,
                "max_depth": max(depth.values()) if depth else 0,
                "layers": {str(k): sorted(v)
                           for k, v in sorted(layers.items())},
            }
        except Exception:  # noqa: BLE001
            _log.debug("simulate_disruption failed", exc_info=True)
            return {"ok": False, "reason": "simulation failed"}

    def to_mermaid(self, node_ids: list[str] | None = None) -> str:
        """``graph LR`` export for chat/GodConsole rendering.

        Disrupted nodes get the ⚠️ marker; edge labels show the relation.
        Never raises.
        """
        try:
            nodes = self.list_nodes()
            if node_ids:
                want = set(node_ids)
                nodes = [n for n in nodes if n.node_id in want]
            keep = {n.node_id for n in nodes}
            by_id = {n.node_id: n for n in nodes}

            def esc(label: str) -> str:
                return (label or "").replace('"', "'").replace("\n", " ")[:60]

            def nid(n: GraphNode) -> str:
                return "n_" + n.node_id.replace("-", "_")[:14]

            lines = ["graph LR"]
            for n in nodes:
                mark = " ⚠️" if n.disrupted else ""
                lines.append(f'    {nid(n)}["{esc(n.label)}{mark}<br/><i>{n.type}</i>"]')
            seen_edges = 0
            for e in self.edges():
                if e.from_id not in keep or e.to_id not in keep:
                    continue
                a, b = by_id[e.from_id], by_id[e.to_id]
                lines.append(f'    {nid(a)} -->|"{e.type}"| {nid(b)}')
                seen_edges += 1
                if seen_edges > 120:
                    lines.append("    %% …edge cap reached")
                    break
            if len(nodes) > 60:
                lines.append("    %% …node cap reached")
            return "\n".join(lines)
        except Exception:  # noqa: BLE001
            return "graph LR\n    %% unavailable"

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

    def schedule_order(self) -> list[GraphNode]:
        """Topological order honoring ``depends_on`` edges, with due-date
        urgency breaking ties: among the ready nodes, the one due soonest
        goes first.  "What should I do first?"

        Cycle-safe (Kahn's algorithm on a DAG projection; nodes caught in
        a cycle are appended last, due-date order).  Never raises.
        """
        try:
            nodes = self.list_nodes()
            by_id = {n.node_id: n for n in nodes}
            # edge a depends_on b  =>  b must come before a
            indeg: dict[str, int] = {n.node_id: 0 for n in nodes}
            before: dict[str, list[str]] = {n.node_id: [] for n in nodes}
            for e in self.edges():
                if (e.type == "depends_on" and e.from_id in by_id
                        and e.to_id in by_id and e.from_id != e.to_id):
                    before[e.to_id].append(e.from_id)
                    indeg[e.from_id] += 1

            def due_key(n: GraphNode) -> float:
                due = n.attrs.get("due_ts") or n.attrs.get("due")
                try:
                    return float(due) if due else float("inf")
                except (TypeError, ValueError):
                    return float("inf")

            ready = sorted((nid for nid, d in indeg.items() if d == 0),
                           key=lambda nid: due_key(by_id[nid]))
            out: list[GraphNode] = []
            while ready:
                nid = ready.pop(0)
                out.append(by_id[nid])
                for nxt in before[nid]:
                    indeg[nxt] -= 1
                    if indeg[nxt] == 0:
                        ready.append(nxt)
                ready.sort(key=lambda x: due_key(by_id[x]))
            # cycle leftovers: due-date order, appended (honest, not dropped)
            if len(out) < len(nodes):
                seen = {n.node_id for n in out}
                leftover = sorted((n for n in nodes if n.node_id not in seen),
                                  key=due_key)
                out.extend(leftover)
            return out
        except Exception:  # noqa: BLE001
            _log.debug("schedule_order failed", exc_info=True)
            return []

    def critical_path_pert(self) -> dict[str, Any]:
        """Duration-aware critical path (PERT).

        Nodes may carry PERT three-point estimates in attrs:
        ``optimistic`` / ``most_likely`` / ``pessimistic`` (minutes).
        Expected time TE = (O + 4M + P) / 6, sigma = (P - O) / 6.  Nodes
        without estimates default to 0 expected time (honest: unknown
        work isn't counted as free, it's counted as *unestimated*).

        Returns {"ok", "path", "expected_minutes", "std_minutes",
        "unestimated"}.  Never raises.
        """
        try:
            import math as _math
            from .estimates import pert

            nodes = [n for n in self.list_nodes()
                     if n.type in ("deadline", "schedule", "commitment")]
            ids = {n.node_id for n in nodes}
            by_id = {n.node_id: n for n in nodes}

            def te_sigma(n: GraphNode) -> tuple[float, float]:
                a = n.attrs
                if all(k in a for k in ("optimistic", "most_likely",
                                       "pessimistic")):
                    try:
                        te, sigma = pert(a["optimistic"], a["most_likely"],
                                         a["pessimistic"])
                        return max(0.0, te), max(0.0, sigma)
                    except Exception:  # noqa: BLE001
                        pass
                return 0.0, 0.0

            deps: dict[str, list[str]] = {}
            for n in nodes:
                deps[n.node_id] = [e.to_id for e in self.edges(n.node_id)
                                   if e.from_id == n.node_id
                                   and e.type == "depends_on"
                                   and e.to_id in ids]

            memo: dict[str, tuple[list[str], float, float]] = {}

            def longest(nid: str, seen: frozenset
                        ) -> tuple[list[str], float, float]:
                # -> (path, sum of TE, sum of sigma^2)
                if nid in memo:
                    return memo[nid]
                own_te, own_sigma = te_sigma(by_id[nid])
                best: tuple[list[str], float, float] = (
                    [nid], own_te, own_sigma ** 2)
                for d in deps.get(nid, []):
                    if d in seen:
                        continue
                    sub_path, sub_te, sub_var = longest(d, seen | {nid})
                    cand = ([nid] + sub_path, own_te + sub_te,
                            own_sigma ** 2 + sub_var)
                    if cand[1] > best[1]:
                        best = cand
                memo[nid] = best
                return best

            best_path: list[str] = []
            best_te = best_var = 0.0
            for n in nodes:
                p, te, var = longest(n.node_id, frozenset())
                if te > best_te:
                    best_path, best_te, best_var = p, te, var

            unestimated = sum(1 for nid in best_path
                              if te_sigma(by_id[nid])[0] <= 0)
            return {
                "ok": True,
                "path": [by_id[i].label for i in best_path if i in by_id],
                "node_ids": best_path,
                "expected_minutes": round(best_te, 1),
                "std_minutes": round(_math.sqrt(best_var), 1),
                "unestimated": unestimated,
            }
        except Exception:  # noqa: BLE001
            _log.debug("critical_path_pert failed", exc_info=True)
            return {"ok": False, "reason": "query failed"}

    def project_goal(self, goal: Any) -> int:
        """Project a GoalSystem goal + its steps into the graph.

        The goal becomes a ``project`` node; each step becomes a
        ``commitment`` node with ``depends_on`` edges chaining consecutive
        steps (step N+1 depends_on step N), plus ``owned_by`` edges to the
        goal.  Step nodes carry ``due_ts`` when the goal has a deadline.
        Idempotent via ``external_ref`` ("goal:<id>" / "goalstep:<id>").
        Disruption alerts now cover goals: "your deploy goal depends on a
        flight that just delayed".  Never raises; returns nodes upserted.
        """
        try:
            goal_id = str(getattr(goal, "id", "") or "")
            title = str(getattr(goal, "title", "") or "").strip()
            if not goal_id or not title:
                return 0
            deadline = float(getattr(goal, "deadline", 0) or 0)
            count = 0
            goal_node = self.add_node(
                "project", title,
                {"source": "goals", "status": getattr(goal, "status", ""),
                 "progress": getattr(goal, "progress", 0.0),
                 **({"due_ts": deadline} if deadline else {})},
                external_ref=f"goal:{goal_id}")
            if goal_node is None:
                return 0
            count += 1
            prev_step_id = ""
            for s in list(getattr(goal, "steps", []) or []):
                desc = str(getattr(s, "description", "") or "").strip()[:140]
                if not desc:
                    continue
                step_id = str(getattr(s, "id", "") or "")
                node = self.add_node(
                    "commitment", desc,
                    {"source": "goals", "goal_id": goal_id,
                     "step_status": getattr(s, "status", ""),
                     **({"due_ts": deadline} if deadline else {})},
                    external_ref=f"goalstep:{step_id}" if step_id else "")
                if node is None:
                    continue
                count += 1
                self.add_edge(node.node_id, goal_node.node_id, "owned_by")
                if prev_step_id:
                    self.add_edge(node.node_id, prev_step_id, "depends_on")
                prev_step_id = node.node_id
            return count
        except Exception:  # noqa: BLE001
            _log.debug("project_goal failed", exc_info=True)
            return 0

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
        "  weak                        — load-bearing nodes to protect\n"
        "  cycles                      — report dependency cycles\n"
        "  hood <node-id> [depth]      — k-hop neighborhood view\n"
        "  between <a-id> <b-id>       — shortest dependency path A→B\n"
        "  waves                       — parallel execution waves\n"
        "  sim <node-id>               — dry-run disruption (no state change)\n"
        "  map [node-id]               — mermaid diagram of the graph\n"
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

        if cmd == "weak":
            links = g.weak_links()
            if not links:
                return "not enough dependency links yet to rank load-bearing nodes."
            lines = ["🧱 load-bearing nodes (protect these first):"]
            for i, w in enumerate(links, 1):
                lines.append(
                    f"   {i}. {w['label']} [{w['type']}] — centrality {w['betweenness']}, "
                    f"{w['dependents']} downstream, fan {w['fan_in']}→{w['fan_out']}")
            return "\n".join(lines) + "\n" + GRAPH_DISCLAIMER

        if cmd == "cycles":
            cycles = g.find_cycles()
            if not cycles:
                return "✅ no dependency cycles found."
            by_id = {n.node_id: n for n in g.list_nodes()}
            lines = [f"🔁 {len(cycles)} dependency cycle(s):"]
            for cyc in cycles[:10]:
                labels = [by_id[i].label if i in by_id else i for i in cyc]
                lines.append("   • " + " → ".join(labels) + f" → {labels[0]}")
            return "\n".join(lines) + "\n" + GRAPH_DISCLAIMER

        if cmd == "hood":
            bits = rest.split()
            if not bits:
                return "usage: /graph hood <node-id> [depth]"
            depth = 2
            if len(bits) > 1:
                try:
                    depth = int(bits[1])
                except (TypeError, ValueError):
                    depth = 2
            hood = g.neighborhood(bits[0], depth)
            if not hood:
                return "node not found."
            lines = [f"🧩 neighborhood (depth {depth}):"]
            lines += [f"   • [{n.type}] {n.label}"
                      + (" ⚠️" if n.disrupted else "") for n in hood[:25]]
            return "\n".join(lines) + "\n" + GRAPH_DISCLAIMER

        if cmd == "between":
            bits = rest.split()
            if len(bits) < 2:
                return "usage: /graph between <a-id> <b-id>"
            path = g.path_between(bits[0], bits[1])
            if not path:
                return "no dependency path connects those two."
            return ("🔗 " + " → ".join(n.label for n in path)
                    + f"\n{len(path) - 1} hop(s).\n" + GRAPH_DISCLAIMER)

        if cmd == "waves":
            waves = g.schedule_waves()
            if not waves:
                return "nothing scheduled yet."
            lines = ["🌊 parallel waves (same wave = no dependencies between them):"]
            for i, wave in enumerate(waves, 1):
                tag = " ⚠️ cyclic" if i == len(waves) and g.find_cycles() else ""
                lines.append(f"   wave {i}{tag}: " +
                             ", ".join(n.label for n in wave[:8])
                             + (" …" if len(wave) > 8 else ""))
            return "\n".join(lines) + "\n" + GRAPH_DISCLAIMER

        if cmd == "sim":
            if not rest.strip():
                return "usage: /graph sim <node-id>"
            r = g.simulate_disruption(rest.strip())
            if not r.get("ok"):
                return "node not found."
            lines = [f"🔬 dry run — if '{r['node']}' failed "
                     f"({r['total_affected']} affected, depth {r['max_depth']}):"]
            for depth, labels in r["layers"].items():
                lines.append(f"   depth {depth}: " + ", ".join(labels[:8])
                             + (" …" if len(labels) > 8 else ""))
            return ("\n".join(lines)
                    + "\n(no state changed — this was a simulation.)\n"
                    + GRAPH_DISCLAIMER)

        if cmd == "map":
            node = g.get(rest.strip()) if rest.strip() else None
            ids = ([n.node_id for n in g.neighborhood(node.node_id, 2)]
                   if node else None)
            return ("```mermaid\n" + g.to_mermaid(ids) + "\n```\n"
                    + GRAPH_DISCLAIMER)

        if cmd == "sync":
            count = g.sync_from_memory()
            return f"🔄 projected {count} node(s) from memory into the graph (read-only).\n" + GRAPH_DISCLAIMER

        return _usage()
    except Exception as e:  # noqa: BLE001
        return f"graph hiccup: {e}"
