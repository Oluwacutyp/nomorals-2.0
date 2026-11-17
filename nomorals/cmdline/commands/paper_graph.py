"""``nm papergraph`` — the paper graph around a paper.

    nm papergraph <title or DOI> [--deep] [--json]
    nm papergraph list [--json]
    nm papergraph show <id> [--json]
    nm papergraph graph <id> [--json]

The same paper-graph organ chat /papergraph uses: the seed paper
is resolved against the open scholarly corpus (OpenAlex, with a
Semantic Scholar fallback), the citation neighbourhood is counted
from reference sets (co-citation + bibliographic coupling, pure
counting, deterministic), and the answer names the clusters and
the reading order — what to read next and why. ``graph`` renders
the diagram to a PNG under <home>/visuals. A seed with no paper
prints the honest reason and exits 1 — never an invented graph.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ...research.paper_graph import (
    PaperGraph,
    build_graph,
    format_graph_reply,
    graph_diagram,
    open_graph_store,
)

__all__ = ["_cmd_paper_graph"]


def _home(context: Any) -> str | None:
    settings = getattr(context, "settings", None) if context else None
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(
        settings, "home", None)
    return str(home) if home else None


def _print(answer: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(answer, indent=2, default=str))
    else:
        print(answer.get("reply") or answer.get("error") or "")


def _cmd_paper_graph(args: argparse.Namespace, context: Any) -> int:
    home = _home(context)
    store = open_graph_store(home)
    words = list(getattr(args, "paper_graph_query", None) or [])
    as_json = bool(getattr(args, "json", False))
    deep = bool(getattr(args, "deep", False))
    depth = "deep" if deep else "quick"
    verb = (words[0].lower() if words else "")
    if verb == "list":
        reviews = store.list()
        if as_json:
            print(json.dumps({"ok": True, "reviews": reviews}, indent=2,
                             default=str))
        elif not reviews:
            print("no paper graphs yet — map one with: "
                  "nm papergraph <title or DOI>")
        else:
            for review in reviews:
                print(f"{review['id']} — {review['seed']} "
                      f"({review['nodes']} papers)")
        return 0
    if verb in {"show", "graph"}:
        if len(words) < 2:
            print(f"papergraph: give me a graph id — "
                  f"nm papergraph {verb} <id>", file=sys.stderr)
            return 2
        review = store.get(words[1])
        if review is None:
            print(f"no paper graph “{words[1]}”", file=sys.stderr)
            return 1
        if verb == "show":
            _print({**review, "ok": True,
                    "reply": format_graph_reply(
                        {**review, "ok": True})}, as_json)
            return 0
        from ...visuals.render import render_png

        graph = PaperGraph.from_dict(review.get("graph") or {})
        visuals_dir = (Path(home) if home else
                       Path("~/.nomorals").expanduser()) / "visuals"
        visuals_dir.mkdir(parents=True, exist_ok=True)
        name = f"paper-graph-{review['id']}.png"
        try:
            path = render_png(graph_diagram(graph), visuals_dir / name)
        except Exception as exc:  # noqa: BLE001 — clean CLI failure
            print(f"papergraph: render failed: {exc}", file=sys.stderr)
            return 1
        if as_json:
            print(json.dumps({"ok": True, "review_id": review["id"],
                              "path": str(path)}, indent=2))
        else:
            print(f"paper graph rendered to {path}")
        return 0
    query = " ".join(words).strip()
    if not query:
        print("papergraph: give me a paper — a title or a DOI, e.g. "
              "nm papergraph 10.1038/nature12373", file=sys.stderr)
        return 2
    answer = build_graph(home, query, store=store, depth=depth)
    if not answer.get("ok"):
        print(answer.get("error") or "no paper found",
              file=sys.stderr)
        return 1
    _print(answer, as_json)
    return 0
