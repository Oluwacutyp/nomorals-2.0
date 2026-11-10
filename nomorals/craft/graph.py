"""CraftGraph — a read-computed projection (L5), docs 22+23 P1.

Doc 22 §1's chain — Domain → Occupation → Role → Competency →
Task → Tool → Deliverable → Standard → Verifier — does not exist
in Devon as a structure, while almost everything it would index
*does*: the goals registry's domains, the specialist manifests'
roles, delegation contracts with named role-holders, run tickets
over the durable run records, versioned artifacts, and the skill
filebook (packs included).  This module is the graph as a
**projection over those shipped records** — the doc-15/18
doctrine: computed on read, every time, from the stores of record.
It is never a store itself; nothing is written here, and a node
that cannot resolve back to its source record does not belong in
the graph (the manifests consistency pattern: the graph can never
drift into describing an imaginary workforce).

Sources, each read fail-open (a source that cannot be read
contributes no nodes and a note in ``source_notes`` — an honest
gap, never a crash and never invented nodes):

* the goals registry → ``domain`` nodes;
* the taxonomy seed (``craft/seed.py``, doc 22 §1's branches) →
  seed ``domain``/``occupation``/``role`` nodes — the seed file is
  their record;
* the specialist manifests → ``role`` nodes, ``binds`` edges to
  the roster actors they name;
* the skill filebook → ``skill`` and ``pack`` nodes (a pack is a
  skill doc the schema in ``skills/packs.py`` reads as a pack);
* delegation contracts → ``contract`` nodes, role edges to their
  holders, a ``governed_by`` edge to the manifest class;
* the run-ticket projection → ``ticket`` nodes for every operation
  it covers (contracts, missions, jobs, browser runs, media jobs),
  with ``projected_as`` edges from their contracts;
* the artifact store → ``artifact`` nodes, ``derived_from`` edges
  along provenance.

The query surface is :class:`CraftGraph` itself: nodes by kind,
edges in and out, neighbours.  Stage C's composer composes cells
against this graph; nothing here composes anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..agents.delegation import CONTRACT_ROLES, DelegationStore
from ..agents.manifests import MANIFESTS
from ..agents.run_ticket import ticket_coverage, ticket_for
from ..goals.classes import list_classes
from ..skills.filebook import FileSkillLibrary
from ..skills.packs import is_pack
from ..storage.artifacts import Artifact
from .seed import SEED_DOMAINS

__all__ = [
    "NODE_KINDS",
    "GraphNode",
    "GraphEdge",
    "CraftGraph",
    "project_graph",
]

#: The node vocabulary: doc 22 §1's chain first, then the record
#: kinds the projection actually indexes.  A kind with no nodes is
#: vocabulary, not a claim — kinds are populated only by records.
NODE_KINDS: tuple[str, ...] = (
    "domain", "occupation", "role", "competency", "task", "tool",
    "deliverable", "standard", "verifier",
    "contract", "ticket", "artifact", "skill", "pack", "actor",
)


@dataclass
class GraphNode:
    """One node: a key, a kind, a human label, and the **source**
    naming the record it resolves to (a store, a registry, a file).
    ``detail`` carries only facts that record itself states."""

    key: str
    kind: str
    label: str
    source: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "kind": self.kind,
                "label": self.label, "source": self.source,
                "detail": dict(self.detail)}


@dataclass
class GraphEdge:
    """One directed edge between two node keys.  Edges exist only
    between nodes the projection actually holds — no dangling
    references to records that were not read."""

    kind: str
    src: str
    dst: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "src": self.src, "dst": self.dst,
                "detail": dict(self.detail)}


class CraftGraph:
    """An immutable snapshot of one projection run."""

    def __init__(self, nodes: tuple[GraphNode, ...],
                 edges: tuple[GraphEdge, ...],
                 source_notes: tuple[str, ...] = ()) -> None:
        self.nodes = nodes
        self.edges = edges
        self.source_notes = source_notes
        self._by_key = {node.key: node for node in nodes}

    def node(self, key: str) -> GraphNode | None:
        return self._by_key.get(key)

    def by_kind(self, kind: str) -> list[GraphNode]:
        return [node for node in self.nodes if node.kind == kind]

    def edges_from(self, key: str) -> list[GraphEdge]:
        return [edge for edge in self.edges if edge.src == key]

    def edges_to(self, key: str) -> list[GraphEdge]:
        return [edge for edge in self.edges if edge.dst == key]

    def neighbours(self, key: str) -> list[GraphNode]:
        keys = {edge.dst for edge in self.edges_from(key)}
        keys |= {edge.src for edge in self.edges_to(key)}
        return [self._by_key[k] for k in sorted(keys)
                if k in self._by_key]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for node in self.nodes:
            out[node.kind] = out.get(node.kind, 0) + 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": [node.to_dict() for node in self.nodes],
                "edges": [edge.to_dict() for edge in self.edges],
                "counts": self.counts(),
                "source_notes": list(self.source_notes)}


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    return _SLUG_RE.sub("-", str(text or "").lower()).strip("-")


class _Builder:
    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []
        self._edge_keys: set[tuple[str, str, str]] = set()
        self.notes: list[str] = []

    def add_node(self, key: str, kind: str, label: str, source: str,
                 detail: dict[str, Any] | None = None) -> GraphNode:
        existing = self.nodes.get(key)
        if existing is not None:
            return existing
        node = GraphNode(key=key, kind=kind, label=label,
                         source=source, detail=dict(detail or {}))
        self.nodes[key] = node
        return node

    def add_edge(self, kind: str, src: str, dst: str,
                 detail: dict[str, Any] | None = None) -> None:
        if src not in self.nodes or dst not in self.nodes:
            return
        marker = (kind, src, dst)
        if marker in self._edge_keys:
            return
        self._edge_keys.add(marker)
        self.edges.append(GraphEdge(kind=kind, src=src, dst=dst,
                                    detail=dict(detail or {})))

    def build(self) -> CraftGraph:
        return CraftGraph(tuple(self.nodes.values()),
                          tuple(self.edges), tuple(self.notes))


def _project_static(builder: _Builder,
                    library: FileSkillLibrary | None) -> None:
    """Registry + seed + manifest + filebook sources (no db)."""
    for cls in list_classes():
        builder.add_node(
            f"domain:{cls.id}", "domain", cls.title,
            "goals/classes registry",
            {"progress_model": cls.progress_model,
             "tool_hints": list(cls.tool_hints)})
    for domain in SEED_DOMAINS:
        domain_key = f"domain:seed-{domain['id']}"
        builder.add_node(domain_key, "domain", domain["title"],
                         "craft/seed taxonomy",
                         {"seed": True})
        for occupation in domain["occupations"]:
            occ_key = (f"occupation:seed-{domain['id']}/"
                       f"{occupation['id']}")
            builder.add_node(occ_key, "occupation",
                             occupation["title"],
                             "craft/seed taxonomy", {"seed": True})
            builder.add_edge("contains", domain_key, occ_key)
            for role_name in occupation["roles"]:
                role_key = f"role:seed-{_slug(role_name)}"
                builder.add_node(role_key, "role", role_name,
                                 "craft/seed taxonomy", {"seed": True})
                builder.add_edge("names_role", occ_key, role_key)
    for m in MANIFESTS:
        role_key = f"role:{m.class_id}"
        builder.add_node(
            role_key, "role", m.title, "agents/manifests",
            {"authority": m.authority, "heavy": m.heavy,
             "evidence_duties": list(m.evidence_duties),
             "responsibilities": m.responsibilities})
        bindings = m.bindings()
        for surface, names in bindings.items():
            for name in names:
                actor_key = f"actor:{name}"
                builder.add_node(actor_key, "actor", name,
                                 "agents/manifests bindings",
                                 {"surface": surface})
                builder.add_edge("binds", role_key, actor_key,
                                 {"surface": surface})
    if library is not None:
        for doc in library.docs():
            pack = is_pack(doc)
            kind = "pack" if pack else "skill"
            local = (f"draft:{doc.id}" if doc.is_draft else doc.id)
            key = f"{kind}:{local}"
            builder.add_node(
                key, kind, doc.title, "skills/filebook",
                {"origin": doc.origin, "version": doc.version,
                 "draft": doc.is_draft, "domain": doc.domain,
                 "layer": doc.layer})
            if pack and doc.domain:
                builder.add_edge("about_domain", key,
                                 f"domain:{doc.domain}")
            if doc.revises:
                builder.add_edge("revises", key,
                                 f"skill:{doc.revises}")


def _project_contracts(builder: _Builder, db: Any,
                       limit: int) -> None:
    store = DelegationStore(db)
    for contract in store.recent_contracts(limit=limit):
        key = f"contract:{contract.id}"
        builder.add_node(
            key, "contract", contract.title or contract.goal,
            "agents/delegation contracts",
            {"status": contract.status,
             "specialist_class": contract.specialist_class,
             "consequential": contract.consequential,
             "claims_recorded": len(store.claims(contract.id))})
        for role in CONTRACT_ROLES:
            holder = contract.role_holder(role)
            if not holder:
                continue
            actor_key = f"actor:{holder}"
            builder.add_node(actor_key, "actor", holder,
                             "agents/delegation role holders", {})
            builder.add_edge(role, key, actor_key)
        if contract.specialist_class:
            builder.add_edge("governed_by", key,
                             f"role:{contract.specialist_class}")


def _project_tickets(builder: _Builder, db: Any, limit: int) -> None:
    coverage = ticket_coverage(db, limit=limit)
    for row in coverage.get("operations_detail") or []:
        ref = str(row.get("ref") or "")
        if not ref:
            continue
        ticket = ticket_for(ref, db)
        if not ticket.get("found"):
            continue
        key = f"ticket:{ref}"
        builder.add_node(
            key, "ticket", str(ticket.get("label") or ref),
            "agents/run_ticket projection",
            {"kind": ticket.get("kind") or row.get("kind") or "",
             "status": ticket.get("status") or "",
             "absent_fields": list(row.get("absent_fields") or [])})
        if ref.startswith("contract:"):
            builder.add_edge("projected_as",
                             f"contract:{ref.split(':', 1)[1]}", key)


def _project_artifacts(builder: _Builder, db: Any, limit: int) -> None:
    rows = db.query(
        "SELECT * FROM artifacts ORDER BY created_at DESC LIMIT ?",
        (int(limit),))
    artifacts = []
    for row in rows or []:
        artifact = Artifact.from_row(dict(row))
        artifacts.append(artifact)
        builder.add_node(
            f"artifact:{artifact.id}", "artifact",
            f"{artifact.type} by {artifact.creator or 'unknown'}",
            "storage/artifacts",
            {"type": artifact.type, "creator": artifact.creator,
             "verification_state":
                 artifact.provenance.verification_state})
    for artifact in artifacts:
        key = f"artifact:{artifact.id}"
        for source_id in artifact.provenance.derived_from:
            builder.add_edge("derived_from", key,
                             f"artifact:{source_id}")
        if artifact.mission_id:
            builder.add_edge("produced_for", key,
                             f"ticket:mission:{artifact.mission_id}")


def project_graph(db: Any = None, *,
                  library: FileSkillLibrary | None = None,
                  limit: int = 50) -> CraftGraph:
    """Compute the CraftGraph from the shipped records, now.

    ``db`` is the shared Database the record stores live on; with
    no db the graph carries only the static sources (registries,
    seed, manifests, and the filebook when ``library`` is given).
    Empty stores project to an empty record layer — never an error.
    """
    builder = _Builder()
    _project_static(builder, library)
    if db is not None:
        for name, source in (("contracts", _project_contracts),
                             ("tickets", _project_tickets),
                             ("artifacts", _project_artifacts)):
            try:
                source(builder, db, limit)
            except Exception as exc:  # noqa: BLE001 - one unreadable
                # source contributes nothing, and says so
                builder.notes.append(
                    f"{name}: unavailable "
                    f"({type(exc).__name__}: {exc})")
    return builder.build()
