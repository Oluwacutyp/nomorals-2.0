"""Knowledge graph memory — structured, relational, reason-over-stored-knowledge.

Where the flat ``memories`` table stores text, the knowledge graph stores
*entities* (nodes) and the *typed relations* between them (edges), giving
the system strong relationship mapping, long-term recall, and the ability
to *reason over* stored knowledge:

  * ``upsert_node(label, type, properties)`` — add/update an entity
  * ``link(src, dst, relation, weight)``     — add a typed relation
  * ``neighbors(node, depth)``               — local graph traversal
  * ``path(src, dst)``                       — find the relation path
    between two entities (reasoning over stored knowledge)
  * ``infer()``                              — derive implied edges via
    transitive relations (knows/friend/part_of), with a confidence
  * ``recall(query)`` / ``context_for(query)`` — pull relevant sub-graph
    context to inject into an agent prompt

It is a general-purpose memory backend (distinct from the OSINT identity
graph, which is domain-specific): people, concepts, events, facts, places,
organizations, skills, and goals all live in the same graph.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["KnowledgeGraph", "GraphNode", "GraphEdge",
           "knowledge_context", "register"]

# Relations that compose transitively for inference: A R B, B R C => A R C.
_TRANSITIVE = {"knows", "friend_of", "part_of", "works_at", "member_of",
               "caused", "located_in"}
_NODE_TYPES = {"person", "concept", "event", "fact", "place",
               "organization", "skill", "goal", "entity",
               # research-digest claim graph (2026-10-01): research claims,
               # their domains, and cited sources are first-class node types
               # so upsert_node stops coercing them to generic "entity".
               "claim", "domain", "source"}


@dataclass
class GraphNode:
    id: str
    label: str
    type: str = "entity"
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    last_access: float = 0.0
    access_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label, "type": self.type,
                "properties": self.properties, "access_count": self.access_count}

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "GraphNode":
        try:
            props = json.loads(row.get("properties") or "{}")
        except Exception:  # noqa: BLE001
            props = {}
        return cls(id=row["id"], label=row["label"],
                   type=row.get("type", "entity"), properties=props,
                   created_at=float(row.get("created_at", 0)),
                   last_access=float(row.get("last_access", 0)),
                   access_count=int(row.get("access_count", 0)))


@dataclass
class GraphEdge:
    id: str
    src: str
    dst: str
    relation: str
    weight: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "src": self.src, "dst": self.dst,
                "relation": self.relation, "weight": self.weight}

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "GraphEdge":
        try:
            props = json.loads(row.get("properties") or "{}")
        except Exception:  # noqa: BLE001
            props = {}
        return cls(id=row["id"], src=row["src"], dst=row["dst"],
                   relation=row.get("relation", ""),
                   weight=float(row.get("weight", 1.0)), properties=props,
                   created_at=float(row.get("created_at", 0)))


class KnowledgeGraph:
    def __init__(self, db: Any) -> None:
        self.db = db

    # ── write ───────────────────────────────────────────────────────────────
    def upsert_node(self, label: str, *, type: str = "entity",
                    properties: dict[str, Any] | None = None) -> GraphNode:
        label = (label or "").strip() or "unknown"
        ntype = type if type in _NODE_TYPES else "entity"
        now = time.time()
        props = properties or {}
        row = self.db.query_one(
            "SELECT * FROM kg_nodes WHERE label=? AND type=?", (label, ntype))
        if row:
            node_id = row["id"]
            merged = {**GraphNode.from_row(row).properties, **props}
            self.db.execute(
                "UPDATE kg_nodes SET properties=?, last_access=?, "
                "access_count=access_count+1 WHERE id=?",
                (json.dumps(merged, default=str)[:4000], now, node_id))
        else:
            node_id = new_short_id("kg")
            self.db.execute(
                "INSERT INTO kg_nodes (id, label, type, properties, "
                "created_at, last_access) VALUES (?,?,?,?,?,?)",
                (node_id, label, ntype, json.dumps(props, default=str)[:4000],
                 now, now))
        node = self.get_node(node_id)
        return node or GraphNode(id=node_id, label=label, type=ntype,
                                 created_at=now, last_access=now)

    def link(self, src: str, dst: str, relation: str, *,
             weight: float = 1.0,
             properties: dict[str, Any] | None = None,
             auto_nodes: bool = True) -> Optional[GraphEdge]:
        """Add a typed edge. ``auto_nodes`` resolves bare labels to nodes
        (creating them as generic entities if needed) so callers can link by
        label without managing ids."""
        src_id = self._resolve(src, auto_nodes)
        dst_id = self._resolve(dst, auto_nodes)
        if not src_id or not dst_id:
            return None
        relation = (relation or "").strip() or "related_to"
        now = time.time()
        row = self.db.query_one(
            "SELECT id FROM kg_edges WHERE src=? AND dst=? AND relation=?",
            (src_id, dst_id, relation))
        if row:
            self.db.execute(
                "UPDATE kg_edges SET weight=?, properties=? WHERE id=?",
                (weight, json.dumps(properties or {}, default=str)[:2000],
                 row["id"]))
            return self.get_edge(row["id"])
        edge_id = new_short_id("kge")
        self.db.execute(
            "INSERT INTO kg_edges (id, src, dst, relation, weight, properties,"
            " created_at) VALUES (?,?,?,?,?,?,?)",
            (edge_id, src_id, dst_id, relation, weight,
             json.dumps(properties or {}, default=str)[:2000], now))
        return self.get_edge(edge_id)

    def _resolve(self, ref: str, auto_nodes: bool) -> Optional[str]:
        """Accept a node id or a label; resolve to a node id."""
        ref = (ref or "").strip()
        if not ref:
            return None
        row = self.db.query_one("SELECT id FROM kg_nodes WHERE id=?", (ref,))
        if row:
            return row["id"]
        row = self.db.query_one(
            "SELECT id FROM kg_nodes WHERE label=? ORDER BY access_count DESC"
            " LIMIT 1", (ref,))
        if row:
            return row["id"]
        if auto_nodes:
            return self.upsert_node(ref, type="entity").id
        return None

    # ── read / traverse ─────────────────────────────────────────────────────
    def get_node(self, node_id: str) -> GraphNode | None:
        row = self.db.query_one("SELECT * FROM kg_nodes WHERE id=?", (node_id,))
        return GraphNode.from_row(row) if row else None

    def get_edge(self, edge_id: str) -> GraphEdge | None:
        row = self.db.query_one("SELECT * FROM kg_edges WHERE id=?", (edge_id,))
        return GraphEdge.from_row(row) if row else None

    def find_node(self, label: str, *, type: str = "") -> GraphNode | None:
        if type:
            row = self.db.query_one("SELECT * FROM kg_nodes WHERE label=? AND "
                                    "type=?", (label, type))
        else:
            row = self.db.query_one(
                "SELECT * FROM kg_nodes WHERE label=? ORDER BY access_count "
                "DESC LIMIT 1", (label,))
        return GraphNode.from_row(row) if row else None

    def neighbors(self, ref: str, *, depth: int = 1,
                  relation: str = "") -> dict[str, Any]:
        """BFS to ``depth`` hops from a node (by id or label). Returns the
        visited nodes and the edges connecting them."""
        start = self._resolve(ref, auto_nodes=False)
        if not start:
            return {"nodes": [], "edges": []}
        seen = {start}
        frontier = [start]
        edges: list[GraphEdge] = []
        depth = max(1, min(int(depth or 1), 4))
        for _ in range(depth):
            next_frontier: list[str] = []
            for node_id in frontier:
                for e in self._edges_touching(node_id, relation):
                    other = e.dst if e.src == node_id else e.src
                    edges.append(e)
                    if other not in seen:
                        seen.add(other)
                        next_frontier.append(other)
            frontier = next_frontier
            if not frontier:
                break
        nodes = [self.get_node(nid) for nid in seen]
        return {"nodes": [n.to_dict() for n in nodes if n],
                "edges": [e.to_dict() for e in edges]}

    def _edges_touching(self, node_id: str, relation: str) -> list[GraphEdge]:
        if relation:
            rows = self.db.query(
                "SELECT * FROM kg_edges WHERE (src=? OR dst=?) AND relation=?",
                (node_id, node_id, relation))
        else:
            rows = self.db.query(
                "SELECT * FROM kg_edges WHERE src=? OR dst=?",
                (node_id, node_id))
        return [GraphEdge.from_row(r) for r in rows]

    def query(self, subject: str = "", relation: str = "",
              obj: str = "") -> list[GraphEdge]:
        """Triple lookup: filter edges by subject / relation / object (any
        subset; blank = wildcard)."""
        clauses, params = [], []
        if subject:
            clauses.append("s.label = ?"); params.append(subject)
        if relation:
            clauses.append("e.relation = ?"); params.append(relation)
        if obj:
            clauses.append("d.label = ?"); params.append(obj)
        sql = ("SELECT e.* FROM kg_edges e JOIN kg_nodes s ON e.src=s.id "
               "JOIN kg_nodes d ON e.dst=d.id")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return [GraphEdge.from_row(r) for r in self.db.query(sql, params)]

    def path(self, src: str, dst: str, *, max_depth: int = 4) -> list[GraphEdge]:
        """BFS for the shortest relation path between two nodes (reasoning
        over stored knowledge). Empty list if no path within max_depth."""
        start = self._resolve(src, auto_nodes=False)
        goal = self._resolve(dst, auto_nodes=False)
        if not start or not goal or start == goal:
            return []
        from collections import deque
        q = deque([(start, [start])])
        seen = {start}
        while q:
            node_id, trail = q.popleft()
            if len(trail) - 1 > max_depth:
                continue
            for e in self._edges_touching(node_id, ""):
                other = e.dst if e.src == node_id else e.src
                if other == goal:
                    # reconstruct the edge trail
                    return self._reconstruct(trail + [other])
                if other not in seen:
                    seen.add(other)
                    q.append((other, trail + [other]))
        return []

    def _reconstruct(self, node_trail: list[str]) -> list[GraphEdge]:
        edges = []
        for a, b in zip(node_trail, node_trail[1:]):
            row = self.db.query_one(
                "SELECT * FROM kg_edges WHERE src=? AND dst=?", (a, b))
            if not row:
                row = self.db.query_one(
                    "SELECT * FROM kg_edges WHERE src=? AND dst=?", (b, a))
            if row:
                edges.append(GraphEdge.from_row(row))
        return edges

    # ── inference ───────────────────────────────────────────────────────────
    def infer(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Derive implied edges via transitive relations: for A R B and
        B R C (R transitive) with no direct A R C edge, emit a suggested
        edge with confidence = weight(A-R-B) * weight(B-R-C) * 0.8."""
        derived: list[dict[str, Any]] = []
        rows = self.db.query(
            "SELECT * FROM kg_edges WHERE relation IN (%s)"
            % ",".join("?" * len(_TRANSITIVE)), tuple(_TRANSITIVE))
        by_rel: dict[str, list[GraphEdge]] = {}
        for r in rows:
            e = GraphEdge.from_row(r)
            by_rel.setdefault(e.relation, []).append(e)
        for rel, edges in by_rel.items():
            if rel not in _TRANSITIVE:
                continue
            outgoing = {e.src: e for e in edges}
            for e2 in edges:
                # e1: X -rel-> e2.src, e2: e2.src -rel-> e2.dst
                e1 = outgoing.get(e2.src)
                if e1 is None or e1 is e2 or e1.src == e2.dst:
                    continue
                cand = self.db.query_one(
                    "SELECT 1 FROM kg_edges WHERE src=? AND dst=? AND "
                    "relation=?", (e1.src, e2.dst, rel))
                if cand:
                    continue
                conf = e1.weight * e2.weight * 0.8
                derived.append({
                    "src": e1.src, "dst": e2.dst, "relation": rel,
                    "confidence": round(conf, 3),
                    "via": e2.src,
                })
                if len(derived) >= limit:
                    return derived
        return derived

    def commit_inferences(self) -> int:
        """Persist derived edges (from ``infer``) into the graph with a
        derived flag. Returns how many were committed."""
        n = 0
        for d in self.infer():
            if self._resolve(d["src"], auto_nodes=False) and \
                    self._resolve(d["dst"], auto_nodes=False):
                self.link(d["src"], d["dst"], d["relation"],
                          weight=d["confidence"],
                          properties={"derived": True, "via": d["via"]})
                n += 1
        return n

    # ── memory hygiene: consolidation, decay, structure ─────────────────────
    def consolidate(self, *, dry_run: bool = False) -> dict[str, Any]:
        """Merge duplicate nodes whose labels match case/whitespace-insensively
        (same type).  Keeps the oldest node, unions properties (survivor wins
        on conflicts), rewires every edge touching a duplicate onto the
        survivor, and drops self-edges created by the merge.

        Returns ``{"merged": n, "duplicates": [...], "dry_run": bool}``.
        """
        out: dict[str, Any] = {"merged": 0, "duplicates": [], "dry_run": dry_run}
        rows = self.db.query("SELECT * FROM kg_nodes ORDER BY created_at, id")
        seen: dict[tuple[str, str], str] = {}
        for row in rows:
            node = GraphNode.from_row(row)
            key = (node.type, " ".join(node.label.lower().split()))
            if not key[1]:
                continue
            keep = seen.get(key)
            if keep is None:
                seen[key] = node.id
                continue
            # node is a duplicate of `keep`
            out["duplicates"].append({"drop": node.id, "into": keep,
                                      "label": node.label})
            if dry_run:
                continue
            survivor = self.get_node(keep)
            # property union: survivor wins
            merged_props = {**node.properties,
                            **(survivor.properties if survivor else {})}
            # rewiring: every edge pointing at the duplicate now points at
            # the survivor (in either direction); self-edges vanish
            edges = self.db.query("SELECT * FROM kg_edges WHERE src=? OR dst=?",
                                  (node.id, node.id))
            for e_row in edges:
                e = GraphEdge.from_row(e_row)
                if e.src == keep or e.dst == keep:
                    self.db.execute("DELETE FROM kg_edges WHERE id=?",
                                    (e.id,))
                    continue
                new_src = keep if e.src == node.id else e.src
                new_dst = keep if e.dst == node.id else e.dst
                self.db.execute("DELETE FROM kg_edges WHERE id=?", (e.id,))
                self.link(new_src, new_dst, e.relation, weight=e.weight,
                          properties=e.properties)
            self.db.execute("DELETE FROM kg_nodes WHERE id=?", (node.id,))
            self.db.execute(
                "UPDATE kg_nodes SET properties=? WHERE id=?",
                (json.dumps(merged_props, default=str)[:4000], keep))
            out["merged"] += 1
        return out

    def decay(self, *, per_day: float = 0.01, floor: float = 0.1,
              stale_days: float = 90.0) -> dict[str, Any]:
        """Time-based memory decay.  Every node carries a ``confidence``
        property (default 1.0); it decays ``per_day`` per day of inactivity,
        floored at ``floor``.  Nodes not accessed in ``stale_days`` get
        ``stale=True`` so consumers can downweight or purge them.

        Returns ``{"updated": n, "stale": n, "purged_candidates": n}``.
        """
        now = time.time()
        rows = self.db.query("SELECT * FROM kg_nodes")
        updated = stale = 0
        for row in rows:
            node = GraphNode.from_row(row)
            conf = node.properties.get("confidence", 1.0)
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                conf = 1.0
            idle_days = max(0.0, (now - node.last_access) / 86400.0)
            new_conf = max(floor, conf * (1.0 - per_day) ** idle_days)
            props = dict(node.properties)
            if idle_days >= stale_days:
                if not props.get("stale"):
                    props["stale"] = True
                    stale += 1
            props["confidence"] = round(new_conf, 4)
            self.db.execute("UPDATE kg_nodes SET properties=? WHERE id=?",
                            (json.dumps(props, default=str)[:4000], node.id))
            updated += 1
        return {"updated": updated, "stale": stale,
                "purged_candidates": stale}

    def communities(self, *, iterations: int = 8,
                    max_nodes: int = 1500,
                    limit: int = 10) -> list[dict[str, Any]]:
        """Deterministic label-propagation community detection.

        Nodes seed with their own id as label; each round a node adopts the
        most frequent neighbor label (ties broken by smallest label, so the
        result is stable).  Returns the top ``limit`` communities by size:
        ``{"label": str, "size": n, "members": [ids...], "types": {...}}``.
        """
        rows = self.db.query(
            "SELECT id FROM kg_nodes ORDER BY id LIMIT ?", (max_nodes,))
        ids = [r["id"] for r in rows]
        if not ids:
            return []
        adj: dict[str, list[str]] = {i: [] for i in ids}
        for r in self.db.query("SELECT src, dst FROM kg_edges"):
            if r["src"] in adj and r["dst"] in adj:
                adj[r["src"]].append(r["dst"])
                adj[r["dst"]].append(r["src"])
        label: dict[str, str] = {i: i for i in ids}
        order = sorted(ids)
        for _ in range(iterations):
            changed = False
            for node in order:
                neigh = adj[node]
                if not neigh:
                    continue
                counts: dict[str, int] = {}
                for n in neigh:
                    counts[label[n]] = counts.get(label[n], 0) + 1
                best = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
                if best != label[node]:
                    label[node] = best
                    changed = True
            if not changed:
                break
        groups: dict[str, list[str]] = {}
        for node, lab in label.items():
            groups.setdefault(lab, []).append(node)
        by_size = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
        out: list[dict[str, Any]] = []
        for members in by_size[:limit]:
            if len(members) < 2:
                continue
            types: dict[str, int] = {}
            for m in members:
                row = self.db.query_one("SELECT type FROM kg_nodes WHERE id=?",
                                        (m,))
                if row:
                    types[row["type"]] = types.get(row["type"], 0) + 1
            out.append({"label": members[0][:12], "size": len(members),
                        "members": sorted(members)[:50], "types": types})
        return out

    def top(self, *, ntype: str = "", n: int = 10) -> list[dict[str, Any]]:
        """Highest-degree nodes (optionally filtered by type) — the hubs of
        what the system actually knows.  Each entry: node dict + degree."""
        deg: dict[str, int] = {}
        for r in self.db.query("SELECT src, dst FROM kg_edges"):
            deg[r["src"]] = deg.get(r["src"], 0) + 1
            deg[r["dst"]] = deg.get(r["dst"], 0) + 1
        q = ("SELECT * FROM kg_nodes"
             + (" WHERE type=?" if ntype else "")
             + " ORDER BY id")
        rows = self.db.query(q, (ntype,) if ntype else ())
        scored = sorted(((deg.get(r["id"], 0), r) for r in rows),
                        key=lambda x: (-x[0], x[1]["label"]))
        return [{**GraphNode.from_row(r).to_dict(), "degree": d}
                for d, r in scored[:max(1, n)]]

    # ── recall / context ────────────────────────────────────────────────────
    def recall(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        """Find nodes whose label/properties overlap the query, and return
        each with its immediate neighborhood as context."""
        q = (query or "").lower()
        qtok = set(w for w in q.split() if len(w) >= 3)
        rows = self.db.query("SELECT * FROM kg_nodes ORDER BY access_count DESC"
                             " LIMIT 400")
        scored = []
        for r in rows:
            node = GraphNode.from_row(r)
            text = (node.label + " " + json.dumps(node.properties, default=str)
                    ).lower()
            overlap = len(qtok & set(text.split())) / len(qtok) if qtok else 0
            if overlap > 0:
                scored.append((overlap * (1 + min(node.access_count, 10) / 10),
                               node))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for _score, node in scored[:limit]:
            nb = self.neighbors(node.id, depth=1)
            out.append({**node.to_dict(), "edges": nb["edges"]})
        return out

    def context_for(self, query: str, *, limit: int = 4) -> str:
        """A compact sub-graph context block for injection into a prompt."""
        recalled = self.recall(query, limit=limit)
        if not recalled:
            return ""
        lines = [f"Known from the knowledge graph about '{query[:60]}':"]
        for node in recalled:
            edges = node.get("edges", [])
            rels = ", ".join(
                f"{e['relation']}->" if e["src"] == node["id"]
                else f"<-{e['relation']}" for e in edges[:4])
            lines.append(f"  - {node['label']} [{node['type']}]"
                         + (f" ({rels})" if rels else ""))
        return "\n".join(lines)

    def stats(self) -> dict[str, Any]:
        nodes = self.db.query_one("SELECT COUNT(*) AS n FROM kg_nodes")["n"]
        edges = self.db.query_one("SELECT COUNT(*) AS n FROM kg_edges")["n"]
        by_type = {r["type"]: r["n"] for r in self.db.query(
            "SELECT type, COUNT(*) AS n FROM kg_nodes GROUP BY type")}
        return {"nodes": nodes, "edges": edges, "by_type": by_type}

    # ── text curation ───────────────────────────────────────────────────────
    #: (regex, kind) — ordered; first match class wins per occurrence
    _EXTRACTORS: tuple[tuple[str, str], ...] = (
        (r"[a-f0-9]{64}", "sha256 digest"),
        (r"\b[a-f0-9]{40}\b", "sha1 digest"),
        (r"\b[a-f0-9]{32}\b", "md5 digest"),
        (r"\b[0-9a-f]{24,44}\b(?=.{0,80}(base32|base58))", "encoded string"),
        (r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "ip address"),
        (r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", "email"),
        (r"https?://[^\s\"'<>]+", "url"),
        (r"\b(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b",
         "domain"),
        (r"\b[A-Z][a-z]+(?:[A-Z][a-z0-9]+){1,}\b", "named symbol"),
        (r"[A-Za-z0-9_\-]{2,64}=\S{6,120}", "token assignment"),
    )

    def curate_from_text(
        self,
        text: str,
        *,
        source: str = "text",
        link_rels: dict[str, str] | None = None,
        max_items: int = 40,
    ) -> dict[str, Any]:
        """Extract entities from free text into the graph.

        Heuristic but real: known digests (with plaintext when the
        known-hash table hits), URLs, IPs, emails, domains, quoted
        secrets, CamelCase symbols, and ``key=value`` tokens each become
        typed nodes, linked from a ``source`` node with the relation
        ``mentions`` (or a per-kind override via ``link_rels``).

        Returns ``{"added_nodes": n, "added_links": m, "items": [...]}``.
        Never raises — curation is an enhancement, not a dependency.
        """
        out: dict[str, Any] = {"added_nodes": 0, "added_links": 0,
                               "items": [], "source": source}
        text = (text or "")[:200_000]
        if not text.strip():
            return out
        try:
            src_node = self.upsert_node(
                source or "text", type="entity",
                properties={"curated": True})
        except Exception:  # noqa: BLE001
            return out

        seen: set[str] = set()
        items: list[tuple[str, str, dict[str, Any]]] = []
        for pattern, kind in self._EXTRACTORS:
            for m in re.finditer(pattern, text):
                item = m.group(0).strip(".,;:!?\"'()[]{}")
                if not item or item.lower() in seen:
                    continue
                seen.add(item.lower())
                props: dict[str, Any] = {"kind": kind,
                                         "source": source}
                # known-hash enrichment
                if "digest" in kind:
                    try:
                        from ..core.decoder import (identify_hash,
                                                    known_hash_lookup,
                                                    known_hash_lookup_chained)

                        cands = identify_hash(item)
                        props["algorithm_candidates"] = cands[:4]
                        # full chain: built-in table, then digests we've
                        # solved before (learned store)
                        known = known_hash_lookup_chained(self.db, item)
                        if known:
                            props["known_plaintext"] = known["plaintext"]
                            props["algorithm"] = known["algorithm"]
                            props["known_source"] = ("built-in" if
                                                     known_hash_lookup(item)
                                                     else "learned")
                    except Exception:  # noqa: BLE001
                        pass
                items.append((item, kind, props))
                if len(items) >= max_items:
                    break
            if len(items) >= max_items:
                break
        if not items:
            return out

        for item, kind, props in items:
            rel = (link_rels or {}).get(kind, "mentions")
            try:
                if "digest" in kind and "known_plaintext" in props:
                    plain = props.pop("known_plaintext")
                    self.upsert_node(f"secret:{plain}", type="fact",
                                     properties={"kind": "plaintext",
                                                 "source": source})
                    node = self.upsert_node(item, type="fact",
                                            properties=props)
                    self.link(node.id, f"secret:{plain}", "decodes_to",
                              weight=2.0)
                    self.link(src_node.id, node.id, "contains", weight=1.5)
                else:
                    node = self.upsert_node(item, type="entity",
                                            properties=props)
                    self.link(src_node.id, node.id, rel, weight=1.0)
                out["added_nodes"] += 1
                out["added_links"] += 1
                out["items"].append({"item": item, "kind": kind,
                                     "relation": rel})
            except Exception:  # noqa: BLE001 — one bad item, not the whole run
                continue
        return out


# ── registry ───────────────────────────────────────────────────────────────


def knowledge_context(context: Any, query: str, *,
                      limit: int = 4) -> str:
    """Knowledge-graph context for *query* — a prompt-ready block, or "".

    This is the universal accessor for "what does the system already know
    about this?": the reasoning pre-flight, the orchestrator, the partner,
    and any agent call this and prepend the result. It is a cheap local
    query and never raises — an empty graph, a missing table, or a bad
    config all yield "" so callers can concatenate unconditionally.
    """
    try:
        enabled = str(getattr(context.settings, "reasoning_knowledge", "on")
                      or "on").strip().lower()
        if enabled == "off":
            return ""
        query = (query or "").strip()
        if len(query) < 8:
            return ""
        graph = KnowledgeGraph(context.db)
        if not graph.stats().get("nodes"):
            return ""
        return graph.context_for(query, limit=limit)
    except Exception:  # noqa: BLE001 — knowledge is an enhancement, never a crash
        return ""


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "kg",
        description=(
            "Knowledge graph memory: entities + typed relations with "
            "traversal, path-finding, inference, and memory hygiene. "
            "action=add (node) | link | neighbors | query | path | recall | "
            "infer | commit_inferences | context | curate (extract entities "
            "from free text) | consolidate (merge duplicate labels) | decay "
            "(time-based confidence decay) | communities (label propagation) "
            "| top (hubs) | stats."
        ),
        capability="memory.write",
        parameters={
            "action": "str — add|link|neighbors|query|path|recall|infer|commit_inferences|context|curate|stats",
            "label": "str — node label (add/neighbors/query)",
            "type": "str — node type",
            "properties": "str — JSON object (add)",
            "src": "str — source node (link/path/query)",
            "dst": "str — destination node (link/path)",
            "relation": "str — typed relation (link/query)",
            "weight": "float (str) — edge weight",
            "query": "str — for recall/context",
            "depth": "int — for neighbors/path",
            "limit": "int",
        },
    )
    def kg(
        action: str = "stats", *, label: str = "", type: str = "",
        properties: str = "", src: str = "", dst: str = "", relation: str = "",
        weight: str = "1.0", query: str = "", depth: str = "1", limit: str = "5",
    ) -> dict[str, Any]:
        graph = KnowledgeGraph(context.db)
        action = (action or "stats").strip().lower()
        try:
            w = float(weight)
        except ValueError:
            w = 1.0
        try:
            d = int(depth or 1)
        except ValueError:
            d = 1
        try:
            n = int(limit or 5)
        except ValueError:
            n = 5
        try:
            props = json.loads(properties) if properties else {}
        except Exception:  # noqa: BLE001
            props = {}
        if action == "add":
            node = graph.upsert_node(label, type=type, properties=props)
            return {"ok": True, "node": node.to_dict()}
        if action == "link":
            e = graph.link(src, dst, relation, weight=w)
            return {"ok": e is not None, "edge": e.to_dict() if e else None}
        if action == "neighbors":
            return graph.neighbors(label or src, depth=d, relation=relation)
        if action == "query":
            edges = graph.query(subject=label or src, relation=relation,
                                obj=dst)
            return {"edges": [e.to_dict() for e in edges]}
        if action == "path":
            edges = graph.path(src, dst, max_depth=max(1, d))
            return {"path": [e.to_dict() for e in edges],
                    "found": bool(edges)}
        if action == "recall":
            return {"results": graph.recall(query, limit=n)}
        if action == "infer":
            return {"inferences": graph.infer(limit=n)}
        if action == "commit_inferences":
            return {"committed": graph.commit_inferences()}
        if action == "context":
            return {"context": graph.context_for(query, limit=n)}
        if action == "curate":
            # Extract entities from free text (decode output, research,
            # news, transcripts) into the graph. `query` carries the text,
            # `label` the source label.
            return {"curated": graph.curate_from_text(
                query or label, source=label or "curated-text")}
        if action == "consolidate":
            return {"ok": True,
                    **graph.consolidate(dry_run=(relation == "dry"))}
        if action == "decay":
            return {"ok": True, **graph.decay()}
        if action == "communities":
            return {"communities": graph.communities(limit=max(1, n))}
        if action == "top":
            return {"top": graph.top(ntype=type or "", n=max(1, n))}
        return graph.stats()
