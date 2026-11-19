"""VectorCraft: open, read, edit, render and save real SVG vector
art (VectorCraft port of storytold/vectorcraft — an open-source
vector art tool in pure Rust).

A thin surface over :mod:`nomorals.media.vector` — the organ owns
the SVG codec, the document model, the boolean engine and the
rasterizer; the tool only routes arguments and shapes the reply.
Pure local work, stdlib + PIL, no upload, no retention. Writes
output files, so FS_WRITE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.policy import Capability
from ..media.vector import (
    Affine,
    VectorDocument,
    VectorEditor,
    render_document,
)
from ..media.vector import (
    run_parity as run_vectorcraft_parity,
)
from ..media.vector import (
    verify_attribution as verify_vectorcraft_attribution,
)

__all__ = ["register"]

_ACTIONS = ("inspect", "create", "add", "style", "transform",
            "group", "boolean", "render", "parity", "attribution")


def _split_ids(value: str | list[str] | None) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [part.strip() for part in value.split(",") if part.strip()]


def register(registry: Any) -> None:
    """Attach the vectorcraft family to a registry."""
    @registry.register(
        "vectorcraft",
        description=(
            "Open, read, edit, render and save real SVG vector art, "
            "local and uncapped — inspect layers, shapes, paths, "
            "gradients and text; add rectangles, circles, ellipses, "
            "lines, polygons, paths and text; restyle fills, strokes "
            "and opacity; transform, group and ungroup; boolean "
            "union, intersection, difference and xor on shapes; "
            "render to a real PNG. Edits preserve the untouched "
            "nodes. VectorCraft actions (vector document model + "
            "fidelity-preserving edits + rasterizer): "
            "inspect=<read the document inventory, sources=[path]>; "
            "create=<blank document, output=<path>, width=<w>, "
            "height=<h>, title=<t>>; add=<add a shape, "
            "kind=<rect|circle|ellipse|line|polygon|path|text>, "
            "x=<n>, y=<n>, width=<n>, height=<n>, r=<n>, text=<t>, "
            "d=<path data>, fill=<color>>; style=<restyle a node, "
            "node=<id>, fill=<color>, stroke=<color>, "
            "stroke_width=<n>, opacity=<0..1>>; transform=<move a "
            "node, node=<id>, dx=<n>, dy=<n>, rotate=<deg>, "
            "scale=<n>>; group=<group nodes, nodes=<id1,id2>, "
            "name=<n>>; boolean=<boolean op, op=<union|intersection|"
            "difference|xor>, a=<id>, b=<id>>; render=<render to PNG, "
            "sources=[path], output=<path.png>, scale=<n>>; "
            "parity=<run the machine-checked parity TOML>; "
            "attribution=<verify the SHA-256 attribution DB>. "
            "action=<inspect|create|add|style|transform|group|"
            "boolean|render|parity|attribution>; sources=<list of "
            ".svg paths>; output=<path to write>."
        ),
        capability=Capability.FS_WRITE,
    )
    def vectorcraft_tool(action: str = "inspect",
                         sources: list[str] | None = None,
                         output: str = "", kind: str = "rect",
                         x: float = 0.0, y: float = 0.0,
                         width: float = 0.0, height: float = 0.0,
                         r: float = 0.0, text: str = "",
                         d: str = "", fill: str = "",
                         stroke: str = "", stroke_width: float = 0.0,
                         opacity: float = -1.0, node: str = "",
                         dx: float = 0.0, dy: float = 0.0,
                         rotate: float = 0.0, scale: float = 0.0,
                         nodes: str | list[str] | None = None,
                         name: str = "", op: str = "union",
                         a: str = "", b: str = "",
                         title: str = ""
                         ) -> dict[str, Any]:
        action = (action or "inspect").lower()
        if action not in _ACTIONS:
            raise ValueError(f"unknown vectorcraft action {action!r}; "
                             f"use one of {', '.join(_ACTIONS)}")
        sources = list(sources or [])
        if action == "inspect":
            if not sources:
                raise ValueError("vectorcraft inspect needs a source "
                                 ".svg")
            summary = VectorDocument.from_bytes(
                Path(sources[0]).read_bytes()).to_dict()
            summary["reply"] = (
                f"{sources[0]}: .svg, {summary['nodes']} node(s), "
                f"{len(summary['layers'])} layer(s), "
                f"{len(summary['gradients'])} gradient(s).")
            return summary
        if action == "parity":
            result = run_vectorcraft_parity()
            failed = [c["id"] for c in result["cases"]
                      if c["status"] == "fail"]
            result["reply"] = (
                f"vectorcraft parity: {result['passed']} passed, "
                f"{result['failed']} failed"
                + (f" ({', '.join(failed)})" if failed else "") + ".")
            return result
        if action == "attribution":
            result = verify_vectorcraft_attribution()
            result["reply"] = (
                f"vectorcraft attribution: {result['checked']} "
                f"verified, {result['failed']} failed, "
                f"{result['skipped']} skipped (clone not on disk).")
            return result
        if action == "create":
            if not output:
                raise ValueError("vectorcraft create needs output=<path>")
            editor = VectorEditor.create(
                width or 800.0, height or 600.0, title=title)
            saved = editor.save()
            out_path = Path(output).expanduser()
            out_path.write_bytes(saved)
            return {"ok": True, "output": str(out_path),
                    "nodes": 0, "size_after": len(saved),
                    "reply": f"created blank .svg at {out_path}."}
        if action == "render":
            if not sources:
                raise ValueError("vectorcraft render needs a source "
                                 ".svg")
            if not output:
                raise ValueError("vectorcraft render needs output=<path>")
            doc = VectorDocument.from_bytes(
                Path(sources[0]).read_bytes())
            png = render_document(doc, scale=scale or 1.0)
            out_path = Path(output).expanduser()
            out_path.write_bytes(png)
            return {"ok": True, "output": str(out_path),
                    "size_after": len(png),
                    "reply": f"rendered {sources[0]} to {out_path} "
                             f"({len(png)} bytes)."}
        if not sources:
            raise ValueError(f"vectorcraft {action} needs a source .svg")
        if not output:
            raise ValueError(f"vectorcraft {action} needs output=<path>")
        original = Path(sources[0]).read_bytes()
        editor = VectorEditor.from_bytes(original)
        if action == "add":
            kind = (kind or "rect").lower()
            if kind == "rect":
                node_id = editor.add_rect(x, y, width, height,
                                          fill=fill or "#000000")
            elif kind == "circle":
                node_id = editor.add_circle(x, y, r,
                                            fill=fill or "#000000")
            elif kind == "ellipse":
                node_id = editor.add_ellipse(x, y, width, height,
                                             fill=fill or "#000000")
            elif kind == "line":
                node_id = editor.add_line(x, y, width, height,
                                          stroke=stroke or "#000000")
            elif kind == "polygon":
                node_id = editor.add_polygon(
                    [(x, y), (x + width, y),
                     (x + width / 2, y + height)],
                    fill=fill or "#000000")
            elif kind == "path":
                if not d:
                    raise ValueError("vectorcraft add path needs d=<data>")
                node_id = editor.add_path(d, fill=fill or "#000000")
            elif kind == "text":
                node_id = editor.add_text(x, y, text,
                                          fill=fill or "#000000")
            else:
                raise ValueError(
                    f"vectorcraft add kind {kind!r}; use rect, circle, "
                    "ellipse, line, polygon, path or text")
            detail = f"added {kind} {node_id}"
        elif action == "style":
            if not node:
                raise ValueError("vectorcraft style needs node=<id>")
            if fill:
                editor.set_fill(node, fill)
            if stroke:
                editor.set_stroke(
                    node, stroke,
                    width=stroke_width if stroke_width > 0 else None)
            if opacity >= 0.0:
                editor.set_opacity(node, opacity)
            detail = f"restyled node {node}"
            node_id = node
        elif action == "transform":
            if not node:
                raise ValueError("vectorcraft transform needs node=<id>")
            affine = Affine.identity()
            if rotate:
                affine = Affine.rotate(rotate)
            if scale:
                affine = Affine.scale(scale) * affine
            affine = Affine.translate(dx, dy) * affine
            editor.transform(node, affine)
            detail = f"transformed node {node}"
            node_id = node
        elif action == "group":
            ids = _split_ids(nodes)
            if len(ids) < 2:
                raise ValueError(
                    "vectorcraft group needs nodes=<id1,id2>")
            node_id = editor.group(ids, name=name or "Group")
            detail = f"grouped {len(ids)} nodes into {node_id}"
        elif action == "boolean":
            if not a or not b:
                raise ValueError(
                    "vectorcraft boolean needs a=<id> and b=<id>")
            result_ids = editor.apply_boolean(op, a, b)
            detail = (f"boolean {op} on {a} and {b}: "
                      f"{len(result_ids)} result node(s)")
            node_id = result_ids[0] if result_ids else ""
        else:  # pragma: no cover - guarded by _ACTIONS above
            raise ValueError(f"unknown vectorcraft action {action!r}")
        saved = editor.save()
        out_path = Path(output).expanduser()
        out_path.write_bytes(saved)
        result: dict[str, Any] = {
            "ok": True, "output": str(out_path),
            "nodes": len(editor.document.all_nodes()),
            "size_before": len(original),
            "size_after": len(saved),
            "reply": f"{detail}; saved to {out_path}."}
        if action in ("add", "group", "boolean"):
            result["node"] = node_id
        return result
