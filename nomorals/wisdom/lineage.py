"""Lineage graph — teacher → student, text → commentary,
translation chains.

Wisdom gap fix: the timeline knew *when* things happened but not *who
taught whom* or *which text comments on which*. This graph is the
missing structure: nodes (person / text / translation / commentary /
school) and typed edges (teacher_of, student_of, commentary_on,
translation_of, influenced, derived_from).

Honesty rules:
- Edges require both endpoint nodes to exist and a ``source`` string —
  an unsourced lineage claim is rejected, not stored.
- ``from_timeline`` derives only ``tradition_sequence`` edges (earlier
  event → later event in the same tradition), labeled as *derived*,
  never presented as teacher-student claims.
- No dates are invented: nodes carry dates only when the caller
  supplies them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["LineageNode", "LineageEdge", "LineageGraph", "RELATIONS"]

RELATIONS = ("teacher_of", "student_of", "commentary_on",
             "translation_of", "influenced", "derived_from",
             "tradition_sequence")


@dataclass
class LineageNode:
    id: str
    kind: str  # person | text | translation | commentary | school
    name: str
    tradition: str = ""
    dates: str = ""  # free text, only when known — never invented
    notes: str = ""
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LineageEdge:
    from_id: str
    to_id: str
    relation: str
    source: str  # required: where this claim comes from

    def validate(self) -> None:
        if self.relation not in RELATIONS:
            raise ValueError(
                f"unknown relation {self.relation!r}; known: {RELATIONS}")
        if not self.source:
            raise ValueError(
                "lineage edge needs a source — unsourced lineage is "
                "rejected, not stored")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LineageGraph:
    """Typed lineage graph with chain queries."""

    def __init__(self) -> None:
        self.nodes: dict[str, LineageNode] = {}
        self.edges: list[LineageEdge] = []

    # ── mutation ─────────────────────────────────────────────────────
    def add_node(self, node: LineageNode) -> LineageNode:
        if not node.id or not node.name:
            raise ValueError("lineage node needs id + name")
        if node.kind not in ("person", "text", "translation",
                              "commentary", "school"):
            raise ValueError(f"unknown node kind {node.kind!r}")
        if node.id in self.nodes:
            raise ValueError(f"node {node.id!r} already exists")
        self.nodes[node.id] = node
        return node

    def add_edge(self, edge: LineageEdge) -> LineageEdge:
        edge.validate()
        if edge.from_id not in self.nodes:
            raise ValueError(f"unknown from-node {edge.from_id!r}")
        if edge.to_id not in self.nodes:
            raise ValueError(f"unknown to-node {edge.to_id!r}")
        if edge in self.edges:
            raise ValueError("duplicate edge")
        self.edges.append(edge)
        return edge

    # ── queries ──────────────────────────────────────────────────────
    def teachers_of(self, person_id: str) -> list[LineageNode]:
        return [self.nodes[e.from_id] for e in self.edges
                if e.to_id == person_id and e.relation == "teacher_of"]

    def students_of(self, person_id: str) -> list[LineageNode]:
        return [self.nodes[e.to_id] for e in self.edges
                if e.from_id == person_id and e.relation == "teacher_of"]

    def commentaries_on(self, text_id: str) -> list[LineageNode]:
        return [self.nodes[e.from_id] for e in self.edges
                if e.to_id == text_id and e.relation == "commentary_on"]

    def translations_of(self, text_id: str) -> list[LineageNode]:
        """Translation chain for a text: translators + their sources."""
        return [self.nodes[e.from_id] for e in self.edges
                if e.to_id == text_id and e.relation == "translation_of"]

    def translation_chain(self, text_id: str) -> list[dict[str, Any]]:
        """The full translation chain: text → translations → their
        translations (recursive), each edge with its source."""
        out: list[dict[str, Any]] = []
        seen: set[str] = set()

        def _walk(node_id: str, depth: int) -> None:
            for e in self.edges:
                if e.to_id == node_id and e.relation == "translation_of":
                    child = self.nodes[e.from_id]
                    if child.id in seen:
                        continue
                    seen.add(child.id)
                    out.append({"depth": depth, "node": child.to_dict(),
                                "edge": e.to_dict()})
                    _walk(child.id, depth + 1)
        _walk(text_id, 0)
        return out

    def ancestors(self, node_id: str, *,
                  relations: tuple[str, ...] = ("teacher_of", "influenced")
                  ) -> list[LineageNode]:
        """Transitive predecessors (teachers of teachers …)."""
        out: list[LineageNode] = []
        seen: set[str] = {node_id}
        frontier = [node_id]
        while frontier:
            nxt: list[str] = []
            for e in self.edges:
                if e.to_id in frontier and e.relation in relations \
                        and e.from_id not in seen:
                    seen.add(e.from_id)
                    out.append(self.nodes[e.from_id])
                    nxt.append(e.from_id)
            frontier = nxt
        return out

    def descendants(self, node_id: str, *,
                    relations: tuple[str, ...] = ("teacher_of", "influenced")
                    ) -> list[LineageNode]:
        out: list[LineageNode] = []
        seen: set[str] = {node_id}
        frontier = [node_id]
        while frontier:
            nxt: list[str] = []
            for e in self.edges:
                if e.from_id in frontier and e.relation in relations \
                        and e.to_id not in seen:
                    seen.add(e.to_id)
                    out.append(self.nodes[e.to_id])
                    nxt.append(e.to_id)
            frontier = nxt
        return out

    def lineage_for(self, query: str) -> list[dict[str, Any]]:
        """Everything connected to nodes whose name matches ``query``
        (case-insensitive substring): node + its edges + neighbours."""
        query = (query or "").strip().lower()
        if not query:
            raise ValueError("lineage_for needs a non-empty query")
        matched = [n for n in self.nodes.values()
                   if query in n.name.lower()]
        out = []
        for node in matched:
            neighbours = []
            for e in self.edges:
                if e.from_id == node.id:
                    neighbours.append({"edge": e.to_dict(),
                                       "other": self.nodes[e.to_id].to_dict()})
                elif e.to_id == node.id:
                    neighbours.append({"edge": e.to_dict(),
                                       "other": self.nodes[e.from_id].to_dict()})
            out.append({"node": node.to_dict(), "neighbours": neighbours})
        return out

    @classmethod
    def from_timeline(cls, events: list[dict[str, Any]]) -> "LineageGraph":
        """Build a graph from timeline events.

        Nodes = events (kind=text). Edges = ``tradition_sequence``
        between consecutive events of the same tradition, sourced as
        'derived from timeline ordering'. These are NOT teacher-student
        claims — the relation name says so.
        """
        graph = cls()
        by_tradition: dict[str, list[dict[str, Any]]] = {}
        for i, event in enumerate(events):
            node_id = f"event-{i}"
            graph.add_node(LineageNode(
                id=node_id, kind="text",
                name=str(event.get("title", f"event {i}")),
                tradition=str(event.get("tradition", "")),
                dates=f"{event.get('start')}–{event.get('end')}",
                notes=str(event.get("summary", "")),
                sources=list(event.get("sources", []))))
            by_tradition.setdefault(str(event.get("tradition", "")),
                                    []).append((i, node_id))
        for tradition, items in by_tradition.items():
            ordered = sorted(items)
            for (i_prev, prev), (i_cur, cur) in zip(ordered, ordered[1:]):
                graph.add_edge(LineageEdge(
                    from_id=prev, to_id=cur,
                    relation="tradition_sequence",
                    source="derived from timeline ordering "
                           f"(events {i_prev} → {i_cur}, tradition "
                           f"{tradition!r})"))
        return graph

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": [n.to_dict() for n in self.nodes.values()],
                "edges": [e.to_dict() for e in self.edges]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LineageGraph":
        graph = cls()
        for raw in d.get("nodes", []):
            graph.add_node(LineageNode(**raw))
        for raw in d.get("edges", []):
            graph.add_edge(LineageEdge(**raw))
        return graph
