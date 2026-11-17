"""Paper graph — the Connected Papers core (52-websites #36).

One seed paper in → the web of papers around it. The similarity
is a map, not a citation tree: relatedness is computed from
co-citation (later papers cite A and B together) and bibliographic
coupling (A and B cite many of the same sources), counted over the
seed's neighbourhood — pure counting over reference sets, fully
deterministic. Companion views: **Prior Works** (older papers most
cited *by the graph's papers* — the field's foundations) and
**Derivative Works** (newer papers citing many graph papers —
where the field went). Clusters name the schools; the reading
order says what to read next and why.

The honest boundary, named in every answer: the candidate pool
rides open citation data (OpenAlex — the seed's references plus
the papers citing the seed, each with its own reference list),
not the full Semantic Scholar graph. A graph over N papers never
masquerades as a complete field map — the pool size is stated
next to every claim. A seed with no paper, or a pool too thin to
draw, is refused honestly; no paper, citation or cluster is ever
invented.

Layering: this organ is L4 (research/). It builds on the evidence
organ's fetchers (:mod:`nomorals.research.extraction`) and never
imports the scene organ (L5) — the diagram it hands out is plain
data; the chat handler renders it.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .extraction import (
    Paper,
    _default_fetch_json,
    _default_home,
    _reconstruct_openalex_abstract,
)

_log = get_logger("nomorals.research.paper_graph")

UA = "nomorals-research/1.0 (paper-graph; contact: owner)"
TIMEOUT = 15
GRAPH_FILENAME = "paper_graphs.json"
GRAPH_DIRNAME = "paper_graphs"

#: seed references fetched per graph (quick depth).
QUICK_REF_LIMIT = 50
#: seed references fetched per graph (deep depth).
DEEP_REF_LIMIT = 100
#: papers citing the seed fetched per graph (quick depth).
QUICK_CITING_LIMIT = 25
#: papers citing the seed fetched per graph (deep depth).
DEEP_CITING_LIMIT = 50
#: nodes in the drawn graph (quick depth) — the reference shows ~40.
QUICK_NODE_LIMIT = 40
#: nodes in the drawn graph (deep depth).
DEEP_NODE_LIMIT = 60
#: edges kept in the drawn graph (top by raw score).
MAX_EDGES = 80
#: a raw (coupling + co-citation) at or above this joins a cluster.
CLUSTER_MIN_RAW = 2

DEPTH_LIMITS: dict[str, dict[str, int]] = {
    "quick": {"refs": QUICK_REF_LIMIT, "citing": QUICK_CITING_LIMIT,
              "nodes": QUICK_NODE_LIMIT},
    "deep": {"refs": DEEP_REF_LIMIT, "citing": DEEP_CITING_LIMIT,
             "nodes": DEEP_NODE_LIMIT},
}

#: OpenAlex work ids are full URLs; the short form is the W-number.
_OPENALEX_ID = re.compile(r"(W\d+)$")
_DOI_LIKE = re.compile(r"10\.\d{4,9}/\S+")

#: node colours in the diagram, by year relative to the seed.
COLOUR_SEED = "#d4a017"
COLOUR_PRIOR = "#4a90d9"
COLOUR_DERIVATIVE = "#7fb069"
COLOUR_SAME_YEAR = "#9a9a9a"


# ── helpers ───────────────────────────────────────────────────────────────

def _short_id(openalex_url: str) -> str:
    """https://openalex.org/W123456 → W123456; anything else passes through."""
    match = _OPENALEX_ID.search((openalex_url or "").strip())
    return match.group(1) if match else (openalex_url or "").strip()


def _openalex_url(short: str) -> str:
    return f"https://openalex.org/{short}"


def _looks_like_doi(query: str) -> bool:
    return bool(_DOI_LIKE.search((query or "").strip()))


def _short_title(title: str, limit: int = 48) -> str:
    title = " ".join((title or "").split())
    if len(title) <= limit:
        return title or "untitled"
    return title[: limit - 1].rstrip() + "…"


def _norm_doi(doi: str) -> str:
    doi = (doi or "").strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.lower().startswith(prefix):
            doi = doi[len(prefix):]
            break
    return doi


# ── shapes ────────────────────────────────────────────────────────────────

@dataclass
class PaperNode:
    """One paper in the graph: the report's PaperNode shape."""

    paper_id: str
    title: str
    year: int | None = None
    citation_count: int | None = None
    doi: str = ""
    url: str = ""
    source: str = ""

    @classmethod
    def from_paper(cls, paper: Paper) -> PaperNode:
        return cls(paper_id=paper.id, title=paper.title, year=paper.year,
                   citation_count=paper.citations, doi=paper.doi,
                   url=paper.url, source=paper.source)

    def to_dict(self) -> dict[str, Any]:
        return {"paper_id": self.paper_id, "title": self.title,
                "year": self.year, "citation_count": self.citation_count,
                "doi": self.doi, "url": self.url, "source": self.source}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaperNode:
        return cls(paper_id=data.get("paper_id", ""),
                   title=data.get("title", ""),
                   year=data.get("year"),
                   citation_count=data.get("citation_count"),
                   doi=data.get("doi", ""), url=data.get("url", ""),
                   source=data.get("source", ""))


@dataclass
class SimilarityEdge:
    """One similarity edge: co-citation + coupling, counted, not guessed."""

    a: str
    b: str
    co_citation: int
    coupling: int
    score: float = 0.0  # normalised to [0, 1] against the graph's max raw

    @property
    def raw(self) -> int:
        return self.co_citation + self.coupling

    def to_dict(self) -> dict[str, Any]:
        return {"a": self.a, "b": self.b,
                "co_citation": self.co_citation,
                "coupling": self.coupling, "score": self.score}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimilarityEdge:
        return cls(a=data.get("a", ""), b=data.get("b", ""),
                   co_citation=int(data.get("co_citation", 0)),
                   coupling=int(data.get("coupling", 0)),
                   score=float(data.get("score", 0.0)))


@dataclass
class GraphCluster:
    """One school in the graph: papers bound by similarity edges."""

    id: str
    name: str
    member_ids: list[str] = field(default_factory=list)
    top_paper_id: str = ""
    year_min: int | None = None
    year_max: int | None = None

    @property
    def size(self) -> int:
        return len(self.member_ids)

    @property
    def stands_alone(self) -> bool:
        return self.size == 1

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name,
                "member_ids": list(self.member_ids),
                "top_paper_id": self.top_paper_id,
                "year_min": self.year_min, "year_max": self.year_max}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphCluster:
        return cls(id=data.get("id", ""), name=data.get("name", ""),
                   member_ids=list(data.get("member_ids") or []),
                   top_paper_id=data.get("top_paper_id", ""),
                   year_min=data.get("year_min"),
                   year_max=data.get("year_max"))


@dataclass
class ReadingStep:
    """One step of the reading order: the paper and why it comes next."""

    paper_id: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"paper_id": self.paper_id, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReadingStep:
        return cls(paper_id=data.get("paper_id", ""),
                   reason=data.get("reason", ""))


@dataclass
class PaperGraph:
    """The graph around one seed: the report's PaperGraph shape."""

    seed_id: str
    nodes: list[PaperNode] = field(default_factory=list)
    edges: list[SimilarityEdge] = field(default_factory=list)
    prior_works: list[dict[str, Any]] = field(default_factory=list)
    derivative_works: list[dict[str, Any]] = field(default_factory=list)
    pool_size: int = 0
    clusters: list[GraphCluster] = field(default_factory=list)
    reading: list[ReadingStep] = field(default_factory=list)
    seed_title: str = ""

    @property
    def origins(self) -> list[str]:
        return [self.seed_id]

    def node_by_id(self, paper_id: str) -> PaperNode | None:
        for node in self.nodes:
            if node.paper_id == paper_id:
                return node
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"seed_id": self.seed_id, "seed_title": self.seed_title,
                "nodes": [n.to_dict() for n in self.nodes],
                "edges": [e.to_dict() for e in self.edges],
                "prior_works": self.prior_works,
                "derivative_works": self.derivative_works,
                "pool_size": self.pool_size,
                "clusters": [c.to_dict() for c in self.clusters],
                "reading": [s.to_dict() for s in self.reading]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaperGraph:
        return cls(
            seed_id=data.get("seed_id", ""),
            seed_title=data.get("seed_title", ""),
            nodes=[PaperNode.from_dict(n)
                   for n in data.get("nodes") or []],
            edges=[SimilarityEdge.from_dict(e)
                   for e in data.get("edges") or []],
            prior_works=list(data.get("prior_works") or []),
            derivative_works=list(data.get("derivative_works") or []),
            pool_size=int(data.get("pool_size", 0)),
            clusters=[GraphCluster.from_dict(c)
                      for c in data.get("clusters") or []],
            reading=[ReadingStep.from_dict(s)
                     for s in data.get("reading") or []])


# ── similarity math (pure counting over reference sets) ───────────────────

def coupling(refs_a: set[str], refs_b: set[str]) -> int:
    """Bibliographic coupling: how many sources A and B both cite."""
    return len(refs_a & refs_b)


def co_citation(a: str, b: str,
                refsets: dict[str, set[str]]) -> int:
    """Co-citation: how many papers cite A and B together."""
    return sum(1 for refs in refsets.values()
               if a in refs and b in refs)


def similarity_edges(ids: list[str],
                     refsets: dict[str, set[str]]) -> list[SimilarityEdge]:
    """Every pair's co-citation + coupling, normalised to [0, 1].

    Deterministic: pairs in sorted-id order, ties broken by id, and
    the normaliser is the graph's own max raw score — the same
    corpus always yields the same edges.
    """
    ordered = sorted(set(ids))
    edges: list[SimilarityEdge] = []
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            cc = co_citation(a, b, refsets)
            cp = coupling(refsets.get(a, set()), refsets.get(b, set()))
            if cc == 0 and cp == 0:
                continue
            edges.append(SimilarityEdge(a=a, b=b, co_citation=cc,
                                        coupling=cp))
    top = max((e.raw for e in edges), default=0)
    for edge in edges:
        edge.score = (edge.raw / top) if top else 0.0
    edges.sort(key=lambda e: (-e.raw, e.a, e.b))
    return edges


def clusters_of(nodes: list[PaperNode], edges: list[SimilarityEdge],
                *, min_raw: int = CLUSTER_MIN_RAW) -> list[GraphCluster]:
    """Schools in the graph: connected components over strong edges.

    A paper with no strong edge stands alone — named as such, never
    forced into a school. A cluster is named after its most-cited
    member, with its year span. Deterministic: components in
    largest-first order, ties by top citation count.
    """
    by_id = {n.paper_id: n for n in nodes}
    parent: dict[str, str] = {n.paper_id: n.paper_id for n in nodes}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for edge in edges:
        if edge.raw < min_raw:
            continue
        if edge.a not in parent or edge.b not in parent:
            continue
        ra, rb = find(edge.a), find(edge.b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups: dict[str, list[str]] = {}
    for node in nodes:
        groups.setdefault(find(node.paper_id), []).append(node.paper_id)
    clusters: list[GraphCluster] = []
    for index, members in enumerate(sorted(
            groups.values(),
            key=lambda ms: (-len(ms),
                            -max(by_id[m].citation_count or 0
                                 for m in ms),
                            sorted(ms)[0]))):
        members = sorted(members)
        top = max(members, key=lambda m: (by_id[m].citation_count or 0,
                                          by_id[m].title))
        years = [by_id[m].year for m in members if by_id[m].year]
        if len(members) == 1:
            name = f"stands alone — {_short_title(by_id[members[0]].title)}"
        else:
            span = (f"{min(years)}–{max(years)}" if len(years) > 1
                    else (str(years[0]) if years else "n.d."))
            name = (f"around “{_short_title(by_id[top].title, 36)}” "
                    f"({span}, {len(members)} papers)")
        clusters.append(GraphCluster(
            id=f"cluster-{index + 1}", name=name, member_ids=members,
            top_paper_id=top,
            year_min=min(years) if years else None,
            year_max=max(years) if years else None))
    return clusters


def reading_order(graph: PaperGraph) -> list[ReadingStep]:
    """What to read next, and why — seed, foundations, similar, follow-ups.

    Each paper appears once, in the order that teaches the field:
    the seed first, then the prior works (most cited by the graph
    first), then the papers most similar to the seed, then the
    derivative works (most graph papers cited first). A paper
    already placed keeps its earliest, strongest reason.
    """
    by_id = {n.paper_id: n for n in graph.nodes}
    seed = graph.node_by_id(graph.seed_id)
    steps: list[ReadingStep] = []
    seen: set[str] = set()

    def add(paper_id: str, reason: str) -> None:
        if paper_id in seen or paper_id not in by_id:
            return
        seen.add(paper_id)
        steps.append(ReadingStep(paper_id=paper_id, reason=reason))

    if seed is not None:
        add(seed.paper_id,
            "start here — the seed paper this map is drawn around")
    for item in graph.prior_works:
        pid = str(item.get("paper_id", ""))
        count = int(item.get("in_graph_citations", 0))
        add(pid, f"foundation — cited by {count} paper(s) in this graph")
    seed_edges = sorted(
        (e for e in graph.edges
         if graph.seed_id in (e.a, e.b)),
        key=lambda e: (-e.raw, e.a, e.b))
    for edge in seed_edges:
        other = edge.b if edge.a == graph.seed_id else edge.a
        add(other,
            f"most similar to the seed (co-citation {edge.co_citation}, "
            f"coupling {edge.coupling})")
    for item in graph.derivative_works:
        pid = str(item.get("paper_id", ""))
        count = int(item.get("graph_out_degree", 0))
        add(pid, f"where the field went — cites {count} graph paper(s)")
    for node in sorted(graph.nodes,
                       key=lambda n: (-(n.citation_count or 0), n.title)):
        add(node.paper_id, "in the neighbourhood")
    return steps


# ── retrieval (open citation data, keyless, injectable, never raises) ─────

def _work_to_paper(work: dict[str, Any]) -> Paper:
    # The paper's id is its OpenAlex id — the reference sets are
    # keyed by OpenAlex URLs, so a DOI id here would never match
    # them and the seed would fall out of its own graph.
    paper = Paper(
        title=(work.get("title") or "").strip(),
        year=work.get("publication_year"),
        doi=_norm_doi(work.get("doi") or ""),
        url=work.get("doi") or work.get("id") or "",
        source="OpenAlex",
        abstract=_reconstruct_openalex_abstract(
            work.get("abstract_inverted_index") or {}),
        citations=work.get("cited_by_count"))
    if work.get("id"):
        paper.id = str(work["id"])
    return paper


def _fetch_openalex_search(query: str, fetch_json: Any) -> list[dict[str, Any]]:
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(
        {"search": query, "per-page": 5})
    try:
        data = fetch_json(url, timeout=TIMEOUT)
    except Exception as exc:  # noqa: BLE001 — a source never kills the graph
        _log.warning("openalex search failed: %s", exc)
        return []
    if not isinstance(data, dict):
        return []
    return [w for w in (data.get("results") or []) if isinstance(w, dict)]


def _fetch_openalex_work_by_doi(doi: str, fetch_json: Any) -> dict[str, Any] | None:
    url = "https://api.openalex.org/works/" + urllib.parse.quote(
        f"doi:{_norm_doi(doi)}", safe=":/")
    try:
        data = fetch_json(url, timeout=TIMEOUT)
    except Exception as exc:  # noqa: BLE001 — a source never kills the graph
        _log.warning("openalex doi fetch failed: %s", exc)
        return None
    return data if isinstance(data, dict) and data.get("id") else None


def _fetch_openalex_works(ids: list[str],
                          fetch_json: Any) -> list[dict[str, Any]]:
    """Batch-fetch OpenAlex work records, 25 ids per call."""
    out: list[dict[str, Any]] = []
    shorts = [_short_id(i) for i in ids if _short_id(i)]
    for start in range(0, len(shorts), 25):
        chunk = shorts[start:start + 25]
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(
            {"filter": "openalex:" + "|".join(chunk), "per-page": 50})
        try:
            data = fetch_json(url, timeout=TIMEOUT)
        except Exception as exc:  # noqa: BLE001 — a source never kills it
            _log.warning("openalex batch fetch failed: %s", exc)
            continue
        if isinstance(data, dict):
            out.extend(w for w in (data.get("results") or [])
                       if isinstance(w, dict))
    return out


def _fetch_citing(seed_short: str, limit: int,
                  fetch_json: Any) -> list[dict[str, Any]]:
    url = ("https://api.openalex.org/works?filter=cites:"
           + seed_short + "&per-page="
           + str(max(1, min(limit, 50)))
           + "&sort=cited_by_count:desc")
    try:
        data = fetch_json(url, timeout=TIMEOUT)
    except Exception as exc:  # noqa: BLE001 — a source never kills the graph
        _log.warning("openalex citing fetch failed: %s", exc)
        return []
    if not isinstance(data, dict):
        return []
    return [w for w in (data.get("results") or []) if isinstance(w, dict)]


def _fetch_semantic_scholar_search(query: str,
                                   fetch_json: Any) -> list[dict[str, Any]]:
    url = ("https://api.semanticscholar.org/graph/v1/paper/search?"
           + urllib.parse.urlencode(
               {"query": query, "limit": 5,
                "fields": "title,abstract,year,citationCount,"
                          "externalIds,url,authors"}))
    try:
        data = fetch_json(url, timeout=TIMEOUT)
    except Exception as exc:  # noqa: BLE001 — a source never kills the graph
        _log.warning("semantic scholar search failed: %s", exc)
        return []
    if not isinstance(data, dict):
        return []
    return [w for w in (data.get("data") or []) if isinstance(w, dict)]


def resolve_seed(query: str, *,
                 fetch_json: Any = None) -> tuple[Paper | None, str, list[str]]:
    """Find the seed paper: OpenAlex search, DOI, then Semantic Scholar.

    Returns (paper, openalex_id, notes). A paper found only through
    Semantic Scholar is resolved back to OpenAlex by DOI when it has
    one — the graph itself is always built from OpenAlex reference
    lists, one citation source, never a mixed seam.
    """
    fetch_json = fetch_json or _default_fetch_json
    query = (query or "").strip()
    notes: list[str] = []
    if not query:
        return None, "", ["no seed given — hand me a paper title or DOI"]
    if _looks_like_doi(query):
        work = _fetch_openalex_work_by_doi(query, fetch_json)
        if work is not None:
            paper = _work_to_paper(work)
            notes.append("OpenAlex: seed resolved by DOI")
            return paper, _openalex_url(_short_id(str(work.get("id", "")))), notes
        notes.append("OpenAlex: no work for that DOI")
    works = _fetch_openalex_search(query, fetch_json)
    if works:
        work = works[0]
        paper = _work_to_paper(work)
        notes.append(f"OpenAlex: seed resolved by search "
                     f"({len(works)} candidate(s))")
        return paper, _openalex_url(_short_id(str(work.get("id", "")))), notes
    notes.append("OpenAlex: no paper for that title")
    items = _fetch_semantic_scholar_search(query, fetch_json)
    for item in items:
        ext = item.get("externalIds") or {}
        doi = _norm_doi(ext.get("DOI") or "")
        if doi:
            work = _fetch_openalex_work_by_doi(doi, fetch_json)
            if work is not None:
                paper = _work_to_paper(work)
                notes.append("Semantic Scholar: seed found, resolved "
                             "to OpenAlex by DOI")
                return paper, _openalex_url(
                    _short_id(str(work.get("id", "")))), notes
        paper = Paper(
            title=(item.get("title") or "").strip(),
            authors=[(a or {}).get("name", "")
                     for a in (item.get("authors") or [])],
            year=item.get("year"), doi=doi,
            url=item.get("url") or "", source="Semantic Scholar",
            abstract=(item.get("abstract") or "").strip(),
            citations=item.get("citationCount"))
        notes.append("Semantic Scholar: seed found, but it has no "
                     "OpenAlex record — the graph needs OpenAlex "
                     "reference lists")
        return paper, "", notes
    notes.append("Semantic Scholar: no paper for that title either")
    return None, "", notes


# ── durable graph store ───────────────────────────────────────────────────

class PaperGraphStore:
    """Durable paper graphs: <home>/data/paper_graphs.json.

    Atomic writes (temp file + replace). A graph holds the seed, the
    nodes, the edges, the prior/derivative views, the clusters and
    the reading order — so a graph can be re-read, re-rendered and
    re-drawn across restarts.
    """

    def __init__(self, path: str | Path = "") -> None:
        self.path = str(path) if path else str(
            _default_home(None) / "data" / GRAPH_FILENAME)
        self._data: dict[str, Any] = {"graphs": {}}
        self._load()

    def _load(self) -> None:
        try:
            raw = Path(self.path).read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict) and isinstance(
                    data.get("graphs"), dict):
                self._data = data
        except (OSError, ValueError):
            self._data = {"graphs": {}}

    def _save(self) -> None:
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2),
                       encoding="utf-8")
        tmp.replace(path)

    def save(self, review: dict[str, Any]) -> None:
        self._data["graphs"][review["id"]] = review
        self._save()

    def get(self, review_id: str) -> dict[str, Any] | None:
        return self._data["graphs"].get(review_id)

    def list(self) -> list[dict[str, Any]]:
        """Graph metadata, newest first."""
        out = [{
            "id": r.get("id", ""), "seed": r.get("seed_title", ""),
            "created_at": r.get("created_at", 0.0),
            "nodes": len((r.get("graph") or {}).get("nodes") or []),
        } for r in self._data["graphs"].values()]
        out.sort(key=lambda r: r["created_at"], reverse=True)
        return out

    def latest(self) -> dict[str, Any] | None:
        listed = self.list()
        if not listed:
            return None
        return self.get(listed[0]["id"])


def open_graph_store(home: str | Path | None = None) -> PaperGraphStore:
    """Open (creating if needed) the durable graph store."""
    return PaperGraphStore(_default_home(home) / "data" / GRAPH_FILENAME)


# ── the graph flow ────────────────────────────────────────────────────────

def build_graph(home: str | Path | None, seed_query: str, *,
                depth: str = "quick",
                store: PaperGraphStore | None = None,
                fetch_json: Any = None) -> dict[str, Any]:
    """Resolve the seed → fetch the neighbourhood → count the graph.

    Never raises on a dead network or an unknown seed: an unmapped
    seed returns ok False with the reason named, and the miss is
    still recorded so it is visible.
    """
    seed_query = (seed_query or "").strip()
    if depth not in DEPTH_LIMITS:
        raise ValueError(
            f"unknown graph depth “{depth}” — quick|deep")
    store = store or open_graph_store(home)
    review_id = ("pg" + hashlib.sha1(
        f"{seed_query}|{time.time()}".encode()).hexdigest()[:10])
    if not seed_query:
        review = {"id": review_id, "seed": "", "seed_title": "",
                  "created_at": time.time(), "graph": None,
                  "error": "give me a paper — a title or a DOI"}
        store.save(review)
        return {"ok": False, "seed": seed_query, "review_id": review_id,
                "graph": None,
                "error": "give me a paper — a title or a DOI",
                "reply": "give me a paper — a title or a DOI, e.g. "
                         "/papergraph 10.1038/nature12373"}
    seed_paper, seed_short, notes = resolve_seed(
        seed_query, fetch_json=fetch_json)
    if seed_paper is None or not seed_short:
        reason = ("no paper found for "
                  f"“{seed_query}” in the open corpus (OpenAlex, "
                  "Semantic Scholar)")
        if seed_paper is not None:
            reason = (f"“{seed_paper.title}” has no OpenAlex record, "
                      "so its reference lists cannot be drawn")
        review = {"id": review_id, "seed": seed_query,
                  "seed_title": seed_paper.title if seed_paper else "",
                  "created_at": time.time(), "graph": None,
                  "error": reason, "sources": notes}
        store.save(review)
        return {"ok": False, "seed": seed_query, "review_id": review_id,
                "graph": None, "error": reason,
                "reply": reason + " — nothing to draw from, so nothing "
                         "invented. Try the DOI.",
                "sources": notes}
    limits = DEPTH_LIMITS[depth]
    fetch_json = fetch_json or _default_fetch_json
    seed_id = _openalex_url(_short_id(seed_short))
    seed_work_refs: list[str] = []
    fetched = _fetch_openalex_works([seed_id], fetch_json)
    if fetched:
        seed_work_refs = [str(r) for r in
                          (fetched[0].get("referenced_works") or [])]
    ref_ids = seed_work_refs[:limits["refs"]]
    citing = _fetch_citing(seed_short, limits["citing"], fetch_json)
    citing_ids = [_short_id(str(w.get("id", ""))) for w in citing
                  if _short_id(str(w.get("id", "")))]
    pool_ids = list(dict.fromkeys(ref_ids + citing_ids))
    pool_records = _fetch_openalex_works(pool_ids, fetch_json)
    by_short: dict[str, dict[str, Any]] = {}
    for work in pool_records:
        by_short[_short_id(str(work.get("id", "")))] = work
    # The seed's own record (when fetched) joins the reference sets
    # so co-citation counts the seed as a citer of its references.
    refsets: dict[str, set[str]] = {seed_id: set(ref_ids)}
    nodes: list[PaperNode] = [PaperNode.from_paper(seed_paper)]
    seen_nodes: set[str] = {seed_id}
    for pid in pool_ids:
        work = by_short.get(_short_id(pid))
        if work is None:
            notes.append(f"OpenAlex: no record for {pid} — left out "
                         "of the graph, never faked")
            continue
        full = _openalex_url(_short_id(pid))
        refs = {str(r) for r in (work.get("referenced_works") or [])}
        refsets[full] = refs
        if full in seen_nodes:
            continue
        seen_nodes.add(full)
        nodes.append(PaperNode.from_paper(_work_to_paper(work)))
    # Rank the pool by similarity to the seed, then citation count,
    # and keep the most related — the drawn graph is the ~40 most
    # related papers, the way theirs is.
    node_ids = [n.paper_id for n in nodes]
    all_edges = similarity_edges(node_ids, refsets)
    seed_rank = {e.b if e.a == seed_id else e.a: e.raw
                 for e in all_edges if seed_id in (e.a, e.b)}
    seed_node = nodes[0]
    rest = sorted(nodes[1:],
                  key=lambda n: (-seed_rank.get(n.paper_id, 0),
                                 -(n.citation_count or 0), n.title))
    kept = [seed_node] + rest[:limits["nodes"] - 1]
    kept_ids = {n.paper_id for n in kept}
    kept_refsets = {pid: refs for pid, refs in refsets.items()
                    if pid in kept_ids}
    edges = similarity_edges(sorted(kept_ids), kept_refsets)
    edges = [e for e in edges if e.raw >= 1][:MAX_EDGES]
    graph = PaperGraph(
        seed_id=seed_id, seed_title=seed_paper.title, nodes=kept,
        edges=edges, pool_size=len(pool_ids))
    _fill_views(graph, refsets, seed_id)
    graph.clusters = clusters_of(kept, edges)
    graph.reading = reading_order(graph)
    review = {"id": review_id, "seed": seed_query,
              "seed_title": seed_paper.title,
              "created_at": time.time(), "graph": graph.to_dict(),
              "sources": notes}
    store.save(review)
    out: dict[str, Any] = {
        "ok": True, "seed": seed_query, "review_id": review_id,
        "graph": graph.to_dict(), "sources": notes,
        "diagram": graph_diagram(graph)}
    out["reply"] = format_graph_reply(out)
    return out


def _fill_views(graph: PaperGraph, refsets: dict[str, set[str]],
                seed_id: str) -> None:
    """Prior works (foundations) and derivative works (follow-ups).

    Prior: older than the seed, ranked by how many graph papers
    cite them. Derivative: newer than the seed, ranked by how many
    graph papers they cite. Both are pure counts over the graph's
    own reference sets — never asserted, always counted.
    """
    seed = graph.node_by_id(seed_id)
    seed_year = seed.year if seed else None
    graph_ids = {n.paper_id for n in graph.nodes}
    in_graph: dict[str, int] = {}
    for node in graph.nodes:
        for ref in refsets.get(node.paper_id, set()):
            if ref in graph_ids:
                in_graph[ref] = in_graph.get(ref, 0) + 1
    priors: list[dict[str, Any]] = []
    derivatives: list[dict[str, Any]] = []
    for node in graph.nodes:
        if node.paper_id == seed_id:
            continue
        out_degree = len(refsets.get(node.paper_id, set()) & graph_ids)
        if seed_year is not None and node.year is not None:
            if node.year < seed_year and in_graph.get(node.paper_id, 0):
                priors.append({"paper_id": node.paper_id,
                               "title": node.title, "year": node.year,
                               "in_graph_citations":
                                   in_graph[node.paper_id]})
            elif node.year > seed_year and out_degree:
                derivatives.append({"paper_id": node.paper_id,
                                    "title": node.title,
                                    "year": node.year,
                                    "graph_out_degree": out_degree})
        elif node.year is None and out_degree:
            # No year to place it: a newer-looking paper that cites
            # the graph is a follow-up, never a foundation.
            derivatives.append({"paper_id": node.paper_id,
                                "title": node.title, "year": node.year,
                                "graph_out_degree": out_degree})
    priors.sort(key=lambda p: (-p["in_graph_citations"],
                               p["year"] or 0, p["title"]))
    derivatives.sort(key=lambda p: (-p["graph_out_degree"],
                                    -(p["year"] or 0), p["title"]))
    graph.prior_works = priors
    graph.derivative_works = derivatives


# ── diagram (plain data — the scene organ renders it) ─────────────────────

def graph_diagram(graph: PaperGraph) -> dict[str, Any]:
    """The graph as a scene-organ diagram spec (plain data, no import).

    Nodes are sized by citation count in the label and coloured by
    year relative to the seed (gold seed, blue older, green newer,
    grey undated); edges are labelled with their co-citation and
    coupling counts. The chat handler renders this through the
    scene organ; the tool and CLI return it as data.
    """
    seed = graph.node_by_id(graph.seed_id)
    seed_year = seed.year if seed else None
    nodes: list[dict[str, Any]] = []
    for index, node in enumerate(graph.nodes):
        if node.paper_id == graph.seed_id:
            colour, shape = COLOUR_SEED, "box"
        elif seed_year is not None and node.year is not None \
                and node.year < seed_year:
            colour, shape = COLOUR_PRIOR, "circle"
        elif seed_year is not None and node.year is not None \
                and node.year > seed_year:
            colour, shape = COLOUR_DERIVATIVE, "circle"
        else:
            colour, shape = COLOUR_SAME_YEAR, "circle"
        label = _short_title(node.title, 30)
        if node.citation_count:
            label += f" ({node.citation_count})"
        nodes.append({"id": f"p{index}", "label": label,
                      "shape": shape, "color": colour})
    id_map = {node.paper_id: f"p{index}"
              for index, node in enumerate(graph.nodes)}
    edges: list[dict[str, Any]] = []
    for edge in graph.edges:
        if edge.a not in id_map or edge.b not in id_map:
            continue
        edges.append({"from": id_map[edge.a], "to": id_map[edge.b],
                      "label": f"co {edge.co_citation} · cp "
                               f"{edge.coupling}"})
    return {"kind": "diagram",
            "title": f"Papers around “{_short_title(graph.seed_title, 40)}”",
            "nodes": nodes, "edges": edges}


# ── rendering ─────────────────────────────────────────────────────────────

def _n_papers(n: int) -> str:
    return "1 paper" if n == 1 else f"{n} papers"


def format_graph_reply(answer: dict[str, Any]) -> str:
    """Render a graph answer (or a stored review) as plain chat text."""
    if not answer.get("ok", True):
        return answer.get("reply") or answer.get("error") or "no answer"
    graph_data = answer.get("graph") or {}
    graph = graph_data if isinstance(graph_data, PaperGraph) \
        else PaperGraph.from_dict(graph_data)
    sources = answer.get("sources") or []
    by_id = {n.paper_id: n for n in graph.nodes}
    seed = graph.node_by_id(graph.seed_id)
    lines = [
        f"Papers around “{graph.seed_title}” — "
        f"{_n_papers(len(graph.nodes))} in the graph, drawn from a "
        f"pool of {graph.pool_size} candidate(s) (OpenAlex). A graph "
        f"over {len(graph.nodes)} papers is a neighbourhood, not the "
        f"whole field.",
    ]
    if sources:
        lines.append("Sources: " + "; ".join(sources) + ".")
    if seed is not None:
        lines.append(
            f"Seed: {seed.title} ({seed.year or 'n.d.'}"
            + (f", {seed.citation_count} citations" if seed.citation_count
               else "") + ").")
    seed_edges = sorted(
        (e for e in graph.edges if graph.seed_id in (e.a, e.b)),
        key=lambda e: (-e.raw, e.a, e.b))
    if seed_edges:
        lines.append("Closest to the seed:")
        for edge in seed_edges[:8]:
            other_id = edge.b if edge.a == graph.seed_id else edge.a
            other = by_id.get(other_id)
            if other is None:
                continue
            lines.append(
                f"  [{edge.score:.2f}] {other.title} "
                f"({other.year or 'n.d.'}, co-citation "
                f"{edge.co_citation}, coupling {edge.coupling})")
    for cluster in graph.clusters:
        if cluster.stands_alone:
            lines.append(f"Cluster: {cluster.name}.")
            continue
        members = ", ".join(
            _short_title((by_id[m].title if m in by_id else m), 24)
            for m in cluster.member_ids[:5])
        lines.append(f"Cluster: {cluster.name} — {members}.")
    if graph.prior_works:
        lines.append("Prior works (the foundations):")
        for item in graph.prior_works[:5]:
            lines.append(
                f"  {item.get('title', '')} "
                f"({item.get('year') or 'n.d.'}) — cited by "
                f"{item.get('in_graph_citations', 0)} graph paper(s)")
    if graph.derivative_works:
        lines.append("Derivative works (where the field went):")
        for item in graph.derivative_works[:5]:
            lines.append(
                f"  {item.get('title', '')} "
                f"({item.get('year') or 'n.d.'}) — cites "
                f"{item.get('graph_out_degree', 0)} graph paper(s)")
    if graph.reading:
        lines.append("Read next:")
        for index, step in enumerate(graph.reading[:8], 1):
            node = by_id.get(step.paper_id)
            title = node.title if node else step.paper_id
            lines.append(f"  {index}. {_short_title(title, 44)} — "
                         f"{step.reason}")
    lines.append("Corpus: open citation data only (OpenAlex, with "
                 "Semantic Scholar only to resolve a seed). No paper, "
                 "citation or cluster is invented — a thin pool says "
                 "so in the pool size above.")
    return "\n".join(lines)
