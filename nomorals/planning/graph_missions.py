"""Graph missions: missions planned on a planning graph, not a step list.

A classic mission plan is a flat step list — when step 7 fails, the
mission reports "failed at step 7" and stops. A graph mission treats the
plan as a graph: nodes are ``goal`` / ``task`` / ``alternative``, edges
are ``depends`` / ``alternative``. When a task node fails, the executor
follows its ``alternative`` edges and tries another way to reach the
same goal. The outcome is "found another way" (with the reroute
recorded), not "failed at step N".

Three views of one plan, kept in sync by construction:

- ``plan_authoring`` — a plain dict spec -> :class:`PlanGraph`
  (the authoring view: what a human / planner writes).
- ``compiled`` — a :class:`PlanGraph` -> :class:`CompiledPlan`
  (the executable view: topological order + cycle report).
- persisted — :class:`PlanGraph` -> JSON and back
  (:meth:`PlanGraph.to_json` / :meth:`PlanGraph.from_json`,
  :meth:`PlanGraph.save` / :meth:`PlanGraph.load`).

Execution is dependency-driven: a node runs only after every node it
``depends`` on has succeeded. A failed node is retried via its
alternatives (depth-first, cycle-safe, visited set) before the mission
is declared failed. Every public function never raises.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "PlanNode",
    "PlanEdge",
    "PlanGraph",
    "CompiledPlan",
    "ExecutionResult",
    "plan_authoring",
    "compiled",
    "execute_graph",
]

NODE_KINDS = ("goal", "task", "alternative")
EDGE_TYPES = ("depends", "alternative")


@dataclass
class PlanNode:
    """One node of a mission plan graph."""

    node_id: str = ""
    kind: str = "task"  # goal | task | alternative
    label: str = ""
    action: str = ""  # key into the executor's action registry
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "kind": self.kind,
                "label": self.label, "action": self.action,
                "attrs": dict(self.attrs)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanNode":
        data = data or {}
        return cls(node_id=str(data.get("node_id") or ""),
                   kind=str(data.get("kind") or "task"),
                   label=str(data.get("label") or ""),
                   action=str(data.get("action") or ""),
                   attrs=dict(data.get("attrs") or {}))


@dataclass
class PlanEdge:
    """One edge: ``from_id depends alternative`` semantics.

    ``depends``: ``from_id`` runs only after ``to_id`` succeeds.
    ``alternative``: ``to_id`` is another way to achieve ``from_id`` —
    tried when ``from_id`` fails.
    """

    from_id: str = ""
    to_id: str = ""
    type: str = "depends"  # depends | alternative

    def to_dict(self) -> dict[str, Any]:
        return {"from_id": self.from_id, "to_id": self.to_id,
                "type": self.type}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanEdge":
        data = data or {}
        return cls(from_id=str(data.get("from_id") or ""),
                   to_id=str(data.get("to_id") or ""),
                   type=str(data.get("type") or "depends"))


@dataclass
class ExecutionResult:
    """What :func:`execute_graph` did. Never raises on construction."""

    ok: bool = False
    completed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    reroutes: list[dict[str, str]] = field(default_factory=list)
    message: str = ""
    attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "completed": list(self.completed),
                "failed": list(self.failed),
                "reroutes": [dict(r) for r in self.reroutes],
                "message": self.message, "attempts": self.attempts}


@dataclass
class CompiledPlan:
    """The executable view of a :class:`PlanGraph`. Never raises."""

    order: list[str] = field(default_factory=list)
    cycles: list[list[str]] = field(default_factory=list)
    unreachable: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.cycles

    def to_dict(self) -> dict[str, Any]:
        return {"order": list(self.order),
                "cycles": [list(c) for c in self.cycles],
                "unreachable": list(self.unreachable), "ok": self.ok}


class PlanGraph:
    """A mission plan as a graph. In-memory; JSON-persistable."""

    def __init__(self, *, goal: str = "") -> None:
        self.goal = (goal or "").strip()
        self.nodes: dict[str, PlanNode] = {}
        self.edges: list[PlanEdge] = []
        self.created_at: float = time.time()

    # ── authoring ───────────────────────────────────────────────

    def add_node(self, kind: str, label: str, *, action: str = "",
                 node_id: str = "",
                 attrs: dict[str, Any] | None = None) -> PlanNode | None:
        """Add a node. Returns None on bad input (never raises)."""
        try:
            kind = (kind or "").strip().lower()
            label = (label or "").strip()
            if kind not in NODE_KINDS or not label:
                return None
            node = PlanNode(
                node_id=node_id or "pg_" + uuid.uuid4().hex[:10],
                kind=kind, label=label[:300],
                action=(action or "").strip(),
                attrs=dict(attrs or {}))
            self.nodes[node.node_id] = node
            return node
        except Exception:  # noqa: BLE001
            _log.debug("graph_missions add_node failed", exc_info=True)
            return None

    def add_edge(self, from_id: str, to_id: str, type: str) -> bool:
        """Add an edge. ``depends`` / ``alternative`` only; both ends
        must exist; no self-edges; duplicates ignored. Never raises."""
        try:
            type = (type or "").strip().lower()
            if type not in EDGE_TYPES:
                return False
            if not from_id or not to_id or from_id == to_id:
                return False
            if from_id not in self.nodes or to_id not in self.nodes:
                return False
            if any(e.from_id == from_id and e.to_id == to_id
                   and e.type == type for e in self.edges):
                return True
            self.edges.append(PlanEdge(from_id, to_id, type))
            return True
        except Exception:  # noqa: BLE001
            return False

    def dependencies(self, node_id: str) -> list[str]:
        """Node ids ``node_id`` depends on (``depends`` edges out)."""
        return [e.to_id for e in self.edges
                if e.from_id == node_id and e.type == "depends"]

    def alternatives(self, node_id: str) -> list[str]:
        """Node ids that are another way to achieve ``node_id``."""
        return [e.to_id for e in self.edges
                if e.from_id == node_id and e.type == "alternative"]

    def dependents(self, node_id: str) -> list[str]:
        """Node ids that depend on ``node_id``."""
        return [e.from_id for e in self.edges
                if e.to_id == node_id and e.type == "depends"]

    # ── compiled view ───────────────────────────────────────────

    def compile(self) -> CompiledPlan:
        """Topological order over ``depends`` edges + cycle report.

        Cycle-safe (Kahn's algorithm); nodes caught in a cycle are
        reported in ``cycles`` and appended to ``order`` last (honest,
        not dropped) so execution can still attempt them.
        """
        try:
            indeg: dict[str, int] = {nid: 0 for nid in self.nodes}
            before: dict[str, list[str]] = {nid: [] for nid in self.nodes}
            for e in self.edges:
                if e.type != "depends":
                    continue
                # from depends on to  =>  to comes before from
                before[e.to_id].append(e.from_id)
                indeg[e.from_id] += 1
            ready = sorted(nid for nid, d in indeg.items() if d == 0)
            order: list[str] = []
            while ready:
                nid = ready.pop(0)
                order.append(nid)
                for nxt in before[nid]:
                    indeg[nxt] -= 1
                    if indeg[nxt] == 0:
                        ready.append(nxt)
                ready.sort()
            leftover = [nid for nid in self.nodes if nid not in set(order)]
            cycles = [sorted(leftover)] if leftover else []
            order.extend(sorted(leftover))
            # A node is unreachable when it is not the goal, nothing
            # depends on it, and it is not an alternative of anything —
            # i.e. no path from it to the goal through depends/alternative
            # edges. (Goal nodes themselves are always reachable.)
            goal_ids = {n.node_id for n in self.nodes.values()
                        if n.kind == "goal"}
            reachable_to_goal: set[str] = set(goal_ids)
            changed = True
            while changed:
                changed = False
                for e in self.edges:
                    if e.to_id in reachable_to_goal and e.from_id not in reachable_to_goal:
                        reachable_to_goal.add(e.from_id)
                        changed = True
            unreachable = sorted(nid for nid in self.nodes
                                 if nid not in reachable_to_goal)
            return CompiledPlan(order=order, cycles=cycles,
                                unreachable=unreachable)
        except Exception:  # noqa: BLE001
            _log.debug("graph_missions compile failed", exc_info=True)
            return CompiledPlan(order=sorted(self.nodes),
                                cycles=[sorted(self.nodes)])

    # ── persistence ─────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {"goal": self.goal, "created_at": self.created_at,
                "nodes": [n.to_dict() for n in self.nodes.values()],
                "edges": [e.to_dict() for e in self.edges]}

    def to_json(self) -> str:
        try:
            return json.dumps(self.to_dict(), sort_keys=True)
        except (TypeError, ValueError):
            return json.dumps({"goal": self.goal, "nodes": [],
                               "edges": [], "created_at": self.created_at})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanGraph":
        graph = cls(goal=str((data or {}).get("goal") or ""))
        try:
            graph.created_at = float((data or {}).get("created_at") or 0.0)
        except (TypeError, ValueError):
            graph.created_at = 0.0
        for raw in (data or {}).get("nodes") or []:
            if isinstance(raw, dict):
                node = PlanNode.from_dict(raw)
                if node.node_id and node.kind in NODE_KINDS:
                    graph.nodes[node.node_id] = node
        for raw in (data or {}).get("edges") or []:
            if isinstance(raw, dict):
                edge = PlanEdge.from_dict(raw)
                if (edge.type in EDGE_TYPES and edge.from_id in graph.nodes
                        and edge.to_id in graph.nodes
                        and edge.from_id != edge.to_id):
                    graph.edges.append(edge)
        return graph

    @classmethod
    def from_json(cls, text: str) -> "PlanGraph":
        try:
            data = json.loads(text or "{}")
            if not isinstance(data, dict):
                return cls()
            return cls.from_dict(data)
        except (json.JSONDecodeError, TypeError, ValueError):
            return cls()

    def save(self, path: str) -> bool:
        """Write JSON to ``path``. Never raises."""
        try:
            if not path:
                return False
            parent = os.path.dirname(os.path.abspath(path))
            os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.to_json())
            return True
        except OSError:
            _log.debug("graph_missions save failed", exc_info=True)
            return False

    @classmethod
    def load(cls, path: str) -> "PlanGraph | None":
        """Read JSON from ``path``. None when missing/unreadable."""
        try:
            with open(path, encoding="utf-8") as fh:
                return cls.from_json(fh.read())
        except OSError:
            return None


def plan_authoring(spec: dict[str, Any] | None) -> PlanGraph:
    """Authoring view: a plain dict spec -> :class:`PlanGraph`.

    Spec shape::

        {"goal": "ship the report",
         "tasks": [
           {"id": "collect", "label": "collect facts",
            "action": "collect", "depends_on": [],
            "alternatives": [
              {"id": "collect-cache", "label": "use cached facts",
               "action": "collect_cache"}]},
           {"id": "draft", "label": "draft report", "action": "draft",
            "depends_on": ["collect"]},
         ]}

    Also accepts the raw persisted shape (``{"nodes": [...],
    "edges": [...]}``). Unknown / malformed entries are skipped, never
    raised. Never raises.
    """
    try:
        spec = spec or {}
        if "nodes" in spec or "edges" in spec:
            return PlanGraph.from_dict(spec)
        graph = PlanGraph(goal=str(spec.get("goal") or ""))
        goal_node = graph.add_node("goal", graph.goal or "mission")
        if goal_node is None:
            return graph
        id_map: dict[str, str] = {}
        tasks = spec.get("tasks") or []
        if not isinstance(tasks, list):
            tasks = []
        # First pass: task nodes + their alternatives.
        for raw in tasks:
            if not isinstance(raw, dict):
                continue
            label = str(raw.get("label") or raw.get("name") or "").strip()
            if not label:
                continue
            node = graph.add_node(
                "task", label,
                action=str(raw.get("action") or ""),
                node_id=str(raw.get("id") or "") or "",
                attrs={k: v for k, v in raw.items()
                       if k not in ("id", "label", "name", "action",
                                    "depends_on", "alternatives")})
            if node is None:
                continue
            id_map[str(raw.get("id") or node.node_id)] = node.node_id
            graph.add_edge(node.node_id, goal_node.node_id, "depends")
            for alt in raw.get("alternatives") or []:
                if not isinstance(alt, dict):
                    continue
                alt_label = str(alt.get("label") or alt.get("name")
                                or "").strip()
                if not alt_label:
                    continue
                alt_node = graph.add_node(
                    "alternative", alt_label,
                    action=str(alt.get("action") or ""),
                    node_id=str(alt.get("id") or "") or "",
                    attrs={k: v for k, v in alt.items()
                           if k not in ("id", "label", "name", "action",
                                        "depends_on", "alternatives")})
                if alt_node is None:
                    continue
                id_map[str(alt.get("id") or alt_node.node_id)] = alt_node.node_id
                graph.add_edge(node.node_id, alt_node.node_id, "alternative")
        # Second pass: depends_on edges (ids may forward-reference).
        for raw in tasks:
            if not isinstance(raw, dict):
                continue
            src = id_map.get(str(raw.get("id") or ""))
            if not src:
                continue
            for dep in raw.get("depends_on") or []:
                dst = id_map.get(str(dep))
                if dst and dst != src:
                    graph.add_edge(src, dst, "depends")
        return graph
    except Exception:  # noqa: BLE001
        _log.debug("plan_authoring failed", exc_info=True)
        return PlanGraph()


def compiled(graph: PlanGraph | None) -> CompiledPlan:
    """Compiled view: a :class:`PlanGraph` -> executable order."""
    if graph is None:
        return CompiledPlan()
    return graph.compile()


def execute_graph(
    graph: PlanGraph | None,
    actions: dict[str, Callable[..., Any]] | None = None,
    *,
    context: dict[str, Any] | None = None,
) -> ExecutionResult:
    """Execute a plan graph.

    ``actions`` maps a node's ``action`` key to a callable
    ``(node, context) -> bool | None``; a callable returning falsy or
    raising counts as failure. Nodes with no registered action count
    as succeeded (planning-only nodes) — the graph, not the registry,
    decides structure.

    On a node failure the executor follows ``alternative`` edges
    (depth-first, cycle-safe) and tries each alternative in order;
    an alternative that succeeds satisfies the failed node. The
    result message says "found another way" when any reroute saved
    the mission, and names the failed node only when no alternative
    worked. Never raises.
    """
    result = ExecutionResult()
    try:
        if graph is None or not graph.nodes:
            result.message = "nothing to execute — empty plan graph."
            return result
        actions = actions or {}
        ctx = dict(context or {})
        plan = graph.compile()
        succeeded: set[str] = set()
        attempted: set[str] = set()

        def run_node(node_id: str, seen: frozenset) -> bool:
            """Run one node; on failure try its alternatives."""
            nonlocal result
            if node_id in succeeded:
                return True
            if node_id in seen:
                return False
            node = graph.nodes.get(node_id)
            if node is None:
                return False
            result.attempts += 1
            attempted.add(node_id)
            fn = actions.get(node.action) if node.action else None
            ok = True
            if callable(fn):
                try:
                    ok = bool(fn(node, ctx))
                except Exception:  # noqa: BLE001 — failure, not crash
                    ok = False
            if ok:
                succeeded.add(node_id)
                result.completed.append(node.label)
                return True
            # Failure: follow alternative edges before giving up.
            for alt_id in graph.alternatives(node_id):
                if alt_id in seen:
                    continue
                alt = graph.nodes.get(alt_id)
                if alt is None:
                    continue
                result.reroutes.append(
                    {"failed": node.label,
                     "alternative": alt.label})
                if run_node(alt_id, seen | {node_id}):
                    succeeded.add(node_id)
                    result.completed.append(
                        f"{node.label} (via {alt.label})")
                    return True
            result.failed.append(node.label)
            return False

        overall = True
        for node_id in plan.order:
            node = graph.nodes.get(node_id)
            if node is None or node.kind == "goal":
                continue
            # Alternatives are only entered via reroute, never directly.
            if node.kind == "alternative":
                continue
            if node_id in succeeded:
                continue
            # A task runs only after its dependencies succeeded (or
            # were satisfied via an alternative).
            deps_ok = all(d in succeeded
                          or graph.nodes.get(d) is not None
                          and graph.nodes[d].kind == "goal"
                          for d in graph.dependencies(node_id)
                          if graph.nodes.get(d) is not None
                          and graph.nodes[d].kind != "alternative")
            if not deps_ok:
                # Dependency failed with no working alternative: this
                # node cannot run — it fails without being attempted.
                if node.label not in result.failed:
                    result.failed.append(node.label)
                overall = False
                continue
            if not run_node(node_id, frozenset()):
                overall = False
        result.ok = overall and not result.failed
        if result.ok and result.reroutes:
            via = "; ".join(
                f"'{r['failed']}' → '{r['alternative']}'"
                for r in result.reroutes)
            result.message = (
                f"found another way — rerouted: {via}. "
                f"{len(result.completed)} node(s) completed.")
        elif result.ok:
            result.message = (
                f"mission complete — {len(result.completed)} node(s) "
                "completed, no reroute needed.")
        else:
            failed = ", ".join(result.failed) if result.failed else "unknown"
            if result.reroutes:
                result.message = (
                    f"could not complete — tried alternatives but still "
                    f"blocked at: {failed}.")
            else:
                result.message = f"could not complete — blocked at: {failed}."
        return result
    except Exception:  # noqa: BLE001
        _log.debug("execute_graph failed", exc_info=True)
        result.message = "execution hit a snag — no nodes completed."
        return result
