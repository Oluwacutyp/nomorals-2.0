"""``nm graph`` — queryable code graph over a repo.

Builds (incrementally) and queries the durable code graph:
files, modules, classes, functions, resolved call/import/inherit
edges — the Graphify shape, built by Devon herself with zero LLM
calls. See nomorals/tools/code_graph.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _cmd_graph(args: argparse.Namespace, context: Any) -> int:
    """Route `nm graph` to build / status / who-calls / calls /
    where / owner / search."""
    from ...tools.code_graph import CodeGraph

    root = str(Path(args.root).expanduser().resolve())
    action = args.graph_action
    graph = CodeGraph(root)
    try:
        if action == "build":
            stats = graph.build(force=bool(args.force))
            return _emit(args, stats.to_dict(), _render_build(stats))
        if action == "status":
            return _emit(args, graph.status(), _render_status(graph))
        if action == "who-calls":
            rows = graph.who_calls(args.name, limit=args.limit)
            return _emit(args, {"target": args.name, "callers": rows},
                         _render_rows(f"who calls {args.name}", rows,
                                      ("caller", "file", "line")))
        if action == "calls":
            rows = graph.calls(args.name, limit=args.limit)
            return _emit(args, {"target": args.name, "callees": rows},
                         _render_rows(f"what {args.name} calls", rows,
                                      ("callee", "file", "line")))
        if action == "where":
            rows = graph.where_defined(args.name, limit=args.limit)
            return _emit(args, {"target": args.name,
                                "definitions": rows},
                         _render_rows(f"where {args.name} is defined",
                                      rows, ("qualified", "kind", "file",
                                             "line")))
        if action == "owner":
            info = graph.owner(args.name)
            return _emit(args, info, _render_owner(info))
        if action == "search":
            rows = graph.search(args.query, limit=args.limit)
            return _emit(args, {"query": args.query, "results": rows},
                         _render_rows(f"search: {args.query}", rows,
                                      ("qualified", "kind", "file",
                                       "line")))
        print(f"unknown graph action: {action}", file=sys.stderr)
        return 2
    finally:
        graph.close()


def _emit(args: argparse.Namespace, payload: dict[str, Any],
          text: str) -> int:
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(text)
    return 0


def _render_build(stats: Any) -> str:
    d = stats.to_dict()
    return (f"graph built ({d['parser']} parser): "
            f"{d['files_total']} files, {d['parsed']} parsed, "
            f"{d['skipped']} skipped, {d['removed']} removed — "
            f"{d['nodes']} nodes, {d['edges']} edges "
            f"in {d['seconds']}s")


def _render_status(graph: Any) -> str:
    st = graph.status()
    return (f"code graph for {st['root']}\n"
            f"  store: {st['store']}\n"
            f"  parser: {st['parser'] or '(never built)'}\n"
            f"  files: {st['files']}  nodes: {st['nodes']}  "
            f"edges: {st['edges']}")


def _render_rows(title: str, rows: list[dict[str, Any]],
                keys: tuple[str, ...]) -> str:
    if not rows:
        return f"{title}: (none)"
    lines = [f"{title} ({len(rows)}):"]
    for row in rows:
        lines.append("  " + "  ".join(
            str(row.get(key, "")) for key in keys))
    return "\n".join(lines)


def _render_owner(info: dict[str, Any]) -> str:
    lines = [f"{info['path']} — module {info['module'] or '(none)'}",
             f"  owner: {info['owner']}"]
    for sym in info["symbols"]:
        lines.append(f"  {sym['qualified']} ({sym['kind']})"
                     f" line {sym['line']}")
    if not info["symbols"]:
        lines.append("  (no symbols)")
    return "\n".join(lines)
