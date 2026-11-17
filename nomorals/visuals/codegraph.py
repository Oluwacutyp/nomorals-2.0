"""Coding agents render the survey as a diagram.

Devon already surveys call chains before building — the survey
becomes a diagram instead of a text wall. The graph comes from the
real repo through the durable code graph (no LLM call anywhere in
construction or query): nodes are the symbol asked about, its
callers, and its callees; edges are the resolved call edges.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..tools.code_graph import CodeGraph
from .spec import SceneSpecError, diagram_from_parts

__all__ = ["callgraph_scene"]


def callgraph_scene(root: str | Path, symbol: str, *,
                    depth: int = 1,
                    limit: int = 20,
                    db_path: str | Path | None = None
                    ) -> dict[str, Any]:
    """A symbol in a real repo → a validated call-graph diagram spec.

    ``root`` is the repo root, ``symbol`` a qualified name
    (``module.func`` or ``Class.method``). Callers sit on the left,
    callees on the right, the symbol in the middle. Raises
    :class:`SceneSpecError` when the symbol is not defined anywhere
    in the repo — an unknown symbol draws nothing, never a guess.
    """
    if depth != 1:
        raise SceneSpecError(
            "scene spec: call-graph depth is 1 in this rung — callers "
            "and callees of one symbol")
    graph = CodeGraph(root, db_path=db_path)
    try:
        graph.build(provision=False)
        defined = graph.where_defined(symbol, limit=1)
        callers = graph.who_calls(symbol, limit=limit)
        callees = graph.calls(symbol, limit=limit)
    finally:
        graph.close()
    if not defined and not callers and not callees:
        raise SceneSpecError(
            f"scene spec: symbol {symbol!r} is not defined in this repo")
    nodes: list[dict[str, Any]] = [
        {"id": symbol, "label": symbol, "shape": "circle"},
    ]
    edges: list[dict[str, Any]] = []
    seen: set[str] = {symbol}
    for row in callers:
        caller = str(row.get("caller") or "")
        if not caller or caller in seen:
            continue
        seen.add(caller)
        nodes.append({"id": caller, "label": caller})
        edges.append({"from": caller, "to": symbol,
                      "label": str(row.get("file") or "")})
    for row in callees:
        callee = str(row.get("callee") or "")
        if not callee or callee in seen:
            continue
        seen.add(callee)
        nodes.append({"id": callee, "label": callee})
        edges.append({"from": symbol, "to": callee,
                      "label": str(row.get("file") or "")})
    return diagram_from_parts(f"call graph: {symbol}", nodes, edges)
