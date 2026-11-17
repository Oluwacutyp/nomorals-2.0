"""Paper-graph tool: ``paper_graph`` — the citation neighbourhood.

Devon hands over a paper and gets the web of papers around it:
what it cites, what cites it, the papers close to it by shared
citations — counted from open citation data (OpenAlex), with the
clusters named and a reading order that says what to read next
and why. The tool is a thin surface over
:mod:`nomorals.research.paper_graph`, the Connected Papers core
beside the evidence and consensus organs — one retrieval path,
no second corpus. The diagram comes back as a plain spec (data);
the caller renders it through the scene organ — this tool never
imports visuals. A seed with no paper is refused honestly, never
invented.
"""

from __future__ import annotations

from typing import Any

from ..core.policy import Capability

__all__ = ["register"]


def _home_from_context(context: Any) -> str | None:
    settings = getattr(context, "settings", None) if context else None
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(
        settings, "home", None)
    return str(home) if home else None


def register(registry: Any) -> None:
    """Attach the ``paper_graph`` tool to a registry."""
    context = registry.context

    @registry.register(
        "paper_graph",
        description=(
            "Paper graph — the citation neighbourhood around a paper. "
            "Give it a paper (title or DOI) and get the web of papers "
            "around it from open citation data (OpenAlex): what it "
            "cites, what cites it, and the papers close to it by "
            "shared citations (co-citation + bibliographic coupling, "
            "pure counting, deterministic), with the clusters named "
            "(which papers form a school, which stand apart) and a "
            "reading order that says what to read next and why. "
            "action=<map|list|show|graph>; map takes "
            "query=<title or DOI> and depth=<quick|deep, default "
            "quick>; show/graph take review_id=<id from a previous "
            "map>. graph returns the diagram as plain spec data for "
            "the caller to render. A seed with no paper is refused "
            "honestly — no paper, citation or cluster is ever "
            "invented. Returns ok, review_id, graph, diagram, "
            "sources, or error."
        ),
        capability=Capability.NET_OUT,
    )
    def paper_graph_tool(
        query: str = "",
        action: str = "map",
        review_id: str = "",
        depth: str = "quick",
    ) -> dict[str, Any]:
        from ..research.paper_graph import (
            PaperGraph,
            build_graph,
            format_graph_reply,
            graph_diagram,
            open_graph_store,
        )

        home = _home_from_context(context)
        store = open_graph_store(home)
        act = (action or "map").strip().lower()
        if act == "list":
            return {"ok": True, "reviews": store.list()}
        if act in {"show", "graph"}:
            if not (review_id or "").strip():
                raise ValueError(
                    "paper_graph show/graph needs review_id=<id>")
            review = store.get(review_id.strip())
            if review is None:
                return {"ok": False, "review_id": review_id.strip(),
                        "error": f"no paper graph “{review_id.strip()}”"}
            if act == "show":
                shown = dict(review)
                shown["ok"] = True
                shown["reply"] = format_graph_reply(
                    {**review, "ok": True})
                return shown
            graph = PaperGraph.from_dict(review.get("graph") or {})
            return {"ok": True, "review_id": review_id.strip(),
                    "seed": review.get("seed", ""),
                    "seed_title": review.get("seed_title", ""),
                    "diagram": graph_diagram(graph),
                    "reply": (f"graph for “{graph.seed_title}” — "
                              f"{len(graph.nodes)} papers, "
                              f"{len(graph.edges)} edges; the diagram "
                              "spec is in the diagram field, render it "
                              "through the scene organ.")}
        if act != "map":
            raise ValueError(
                f"unknown paper_graph action “{act}” — map|list|show|graph")
        if not (query or "").strip():
            raise ValueError(
                "paper_graph map needs query=<title or DOI>")
        return build_graph(home, query.strip(), store=store,
                           depth=(depth or "quick").strip().lower())
