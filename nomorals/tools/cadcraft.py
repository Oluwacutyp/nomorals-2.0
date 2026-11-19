"""CadCraft: sketch, constrain, solve, draft and render real CAD
drawings (CadCraft port of storytold/cadcraft — an open-source
clean-room CAD application in Rust).

A thin surface over :mod:`nomorals.media.cad` — the organ owns the
``.cadcraft`` codec, the document model, the constraint solver, the
typed command machines, the drafting operations and the renderer;
the tool only routes arguments and shapes the reply. Pure local
work, stdlib + PIL, no upload, no retention. Writes output files,
so FS_WRITE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.policy import Capability
from ..media.cad import (
    CadDocument,
    CadEditor,
    CadError,
    GeomRef,
    render_document,
)
from ..media.cad import (
    run_parity as run_cad_parity,
)
from ..media.cad import (
    verify_attribution as verify_cad_attribution,
)
from ..media.cad.codec import (
    export_dxf,
    import_drawing,
    parse_document,
    to_document_bytes,
)
from ..media.cad.commands import CommandSession
from ..media.cad.solver import dof_report, solve

__all__ = ["register"]

_ACTIONS = ("inspect", "create", "add", "move", "rotate", "scale",
            "constrain", "solve", "dimension", "draft", "render",
            "import", "export", "parity", "attribution")


def _split_ids(value: str | list[str] | None) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [part.strip() for part in value.split(",") if part.strip()]


def _refs(value: str | list[str] | None) -> list[GeomRef]:
    out: list[GeomRef] = []
    for part in _split_ids(value):
        handle, _, sub = part.partition(":")
        out.append(GeomRef(handle, sub or "whole"))
    return out


def _load(sources: list[str]) -> tuple[CadDocument, Path]:
    if not sources:
        raise ValueError("cadcraft needs a source file")
    path = Path(sources[0])
    if not path.is_file():
        raise ValueError(f"cadcraft source not found: {path}")
    if path.suffix.lower() in (".dxf", ".dwg"):
        return import_drawing(path.name, path.read_bytes()), path
    return parse_document(path.read_bytes()), path


def _save(doc: CadDocument, sources: list[str],
          output: str) -> Path:
    target = Path(output) if output else Path(sources[0])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(to_document_bytes(doc))
    return target


def register(registry: Any) -> None:
    """Attach the cadcraft family to a registry."""
    @registry.register(
        "cadcraft",
        description=(
            "Sketch, constrain, solve, draft and render real CAD "
            "drawings, local and uncapped — inspect sketches, "
            "entities, layers and constraints; add lines, arcs, "
            "circles, ellipses, splines, points, polylines and text; "
            "move, rotate and scale entities; apply constraints and "
            "solve the sketch (degrees of freedom reported honestly); "
            "add measured dimensions; draft with trim, extend, "
            "offset, fillet, chamfer, mirror and array; render the "
            "drawing to a real PNG; import and export DXF. CadCraft "
            "actions (CAD document model + constraint solver + "
            "drafting + renderer): inspect=<read the drawing "
            "inventory, sources=[path]>; create=<blank drawing, "
            "output=<path>, title=<t>>; add=<add an entity, "
            "kind=<line|circle|arc|rectangle|polyline|text>, x=<n>, "
            "y=<n>, width=<n>, height=<n>, radius=<n>, text=<t>>; "
            "move=<move entities, nodes=<id1,id2>, dx=<n>, dy=<n>>; "
            "rotate=<rotate entities, nodes=<ids>, x=<cx>, y=<cy>, "
            "angle=<degrees>>; scale=<scale entities, nodes=<ids>, "
            "x=<cx>, y=<cy>, factor=<n>>; constrain=<add a "
            "constraint, kind=<coincident|horizontal|vertical|"
            "parallel|perpendicular|tangent|equal|distance_aligned|"
            "distance_horizontal|distance_vertical|angular|radius|"
            "diameter|fix>, nodes=<id:sub,id:sub>, value=<n>>; "
            "solve=<solve the sketch's constraints>; dimension=<add "
            "a measured dimension, kind=<aligned|angular|radius|"
            "diameter>, nodes=<id:sub,id:sub>, x=<n>, y=<n>>; "
            "draft=<run one typed command, command=<line|circle|"
            "arc|trim|extend|offset|fillet|chamfer|mirror|array|"
            "move|rotate|scale|erase>, params=<dict>>; render=<render "
            "to PNG, sources=[path], output=<path.png>, width=<w>, "
            "height=<h>>; import=<import a DXF drawing, "
            "sources=[path.dxf], output=<path.cadcraft>>; export=<"
            "export to DXF, sources=[path], output=<path.dxf>>; "
            "parity=<run the machine-checked parity TOML>; "
            "attribution=<verify the SHA-256 attribution DB>. "
            "action=<inspect|create|add|move|rotate|scale|constrain|"
            "solve|dimension|draft|render|import|export|parity|"
            "attribution>; sources=<list of .cadcraft/.dxf paths>; "
            "output=<path to write>."
        ),
        capability=Capability.FS_WRITE,
    )
    def cadcraft_tool(action: str = "inspect",
                      sources: list[str] | None = None,
                      output: str = "", kind: str = "line",
                      x: float = 0.0, y: float = 0.0,
                      width: float = 0.0, height: float = 0.0,
                      radius: float = 0.0, text: str = "",
                      node: str = "",
                      nodes: str | list[str] | None = None,
                      dx: float = 0.0, dy: float = 0.0,
                      angle: float = 0.0, factor: float = 1.0,
                      value: float | None = None,
                      command: str = "",
                      params: dict | None = None,
                      title: str = "",
                      ) -> dict[str, Any]:
        action = (action or "inspect").lower()
        if action not in _ACTIONS:
            raise ValueError(f"unknown cadcraft action {action!r}; "
                             f"use one of {', '.join(_ACTIONS)}")
        sources = list(sources or [])
        if action == "parity":
            result = run_cad_parity()
            failed = result["failed"]
            result["reply"] = (
                f"cadcraft parity: {result['passed']} passed, "
                f"{len(failed)} failed"
                + (f" ({', '.join(failed)})" if failed else "") + ".")
            return result
        if action == "attribution":
            result = verify_cad_attribution()
            failed = result["failed"]
            result["reply"] = (
                f"cadcraft attribution: {result['checked']} verified, "
                f"{len(failed)} failed, {len(result['skipped'])} skipped, "
                f"{result['noted']} noted.")
            return result
        if action == "create":
            doc = CadDocument.blank(title=title or "Untitled")
            if not output:
                raise ValueError("cadcraft create needs output=<path>")
            target = _save(doc, sources, output)
            return {"output": str(target), "entities": 0,
                    "reply": f"created blank drawing at {target}."}
        if action == "import":
            if not sources:
                raise ValueError("cadcraft import needs sources=[path.dxf]")
            path = Path(sources[0])
            if not path.is_file():
                raise ValueError(f"cadcraft source not found: {path}")
            try:
                doc = import_drawing(path.name, path.read_bytes())
            except CadError as exc:
                raise ValueError(str(exc)) from exc
            if not output:
                raise ValueError("cadcraft import needs output=<path>")
            target = _save(doc, sources, output)
            return {"output": str(target),
                    "entities": doc.entity_count(),
                    "reply": (f"imported {path} as {target} "
                              f"({doc.entity_count()} entities).")}
        doc, path = _load(sources)
        editor = CadEditor(doc)
        if action == "inspect":
            bounds = doc.bounds()
            return {"title": doc.title, "entities": doc.entity_count(),
                    "constraints": doc.constraint_count(),
                    "layers": [layer.name for layer in doc.layers],
                    "bounds": list(bounds) if bounds else None,
                    "dof": dof_report(doc),
                    "reply": (f"{path}: CAD drawing, "
                              f"{doc.entity_count()} entities, "
                              f"{doc.constraint_count()} constraints, "
                              f"{len(doc.layers)} layers.")}
        if action == "solve":
            try:
                report = solve(doc)
            except CadError as exc:
                raise ValueError(str(exc)) from exc
            target = _save(doc, sources, output)
            report["output"] = str(target)
            report["reply"] = (
                f"solved {path}: converged={report['converged']}, "
                f"status={report['status']}, dof={report['dof']}, "
                f"max residual={report['max_residual']:.2e}.")
            return report
        if action == "render":
            if not output:
                raise ValueError("cadcraft render needs output=<path>")
            png = render_document(doc, width=800, height=600)
            target = Path(output)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(png)
            return {"output": str(target), "entities": doc.entity_count(),
                    "reply": f"rendered {path} to {target}."}
        if action == "add":
            if kind == "line":
                item = editor.add_line((x, y), (x + width, y + height))
            elif kind == "circle":
                item = editor.add_circle((x, y), radius)
            elif kind == "arc":
                item = editor.add_arc((x, y), radius, 0.0, 90.0)
            elif kind == "rectangle":
                item = editor.add_polyline(
                    [(x, y), (x + width, y), (x + width, y + height),
                     (x, y + height)], closed=True)
            elif kind == "polyline":
                item = editor.add_polyline([(x, y),
                                            (x + width, y + height)])
            elif kind == "text":
                item = editor.add_text((x, y), text or "Text",
                                       height or 2.5)
            else:
                raise ValueError(f"unknown cadcraft add kind {kind!r}")
            target = _save(doc, sources, output)
            return {"item": item, "output": str(target),
                    "reply": f"added {kind} {item} to {target}."}
        if action == "move":
            ids = _split_ids(nodes) or ([node] if node else [])
            if not ids:
                raise ValueError("cadcraft move needs nodes=<ids>")
            editor.move(ids, dx, dy)
            target = _save(doc, sources, output)
            return {"items": ids, "output": str(target),
                    "reply": f"moved {len(ids)} entities in {target}."}
        if action == "rotate":
            ids = _split_ids(nodes) or ([node] if node else [])
            if not ids:
                raise ValueError("cadcraft rotate needs nodes=<ids>")
            editor.rotate(ids, (x, y), angle)
            target = _save(doc, sources, output)
            return {"items": ids, "output": str(target),
                    "reply": f"rotated {len(ids)} entities in {target}."}
        if action == "scale":
            ids = _split_ids(nodes) or ([node] if node else [])
            if not ids:
                raise ValueError("cadcraft scale needs nodes=<ids>")
            editor.scale(ids, (x, y), factor)
            target = _save(doc, sources, output)
            return {"items": ids, "output": str(target),
                    "reply": f"scaled {len(ids)} entities in {target}."}
        if action == "constrain":
            refs = _refs(nodes)
            if not refs:
                raise ValueError("cadcraft constrain needs nodes=<refs>")
            try:
                cid = editor.constrain(kind, refs, value)
            except CadError as exc:
                raise ValueError(str(exc)) from exc
            target = _save(doc, sources, output)
            return {"constraint": cid, "output": str(target),
                    "reply": f"added {kind} constraint {cid} in {target}."}
        if action == "dimension":
            refs = _refs(nodes)
            if not refs:
                raise ValueError("cadcraft dimension needs nodes=<refs>")
            try:
                dim = editor.add_dimension(kind, refs, (x, y))
            except CadError as exc:
                raise ValueError(str(exc)) from exc
            target = _save(doc, sources, output)
            return {"dimension": dim.id, "value": dim.value,
                    "output": str(target),
                    "reply": (f"added {kind} dimension {dim.id} "
                              f"({dim.value:g}) in {target}.")}
        if action == "draft":
            if not command:
                raise ValueError("cadcraft draft needs command=<name>")
            session = CommandSession(doc)
            try:
                result = session.execute(command, params or {})
            except CadError as exc:
                raise ValueError(str(exc)) from exc
            target = _save(doc, sources, output)
            result["output"] = str(target)
            result["reply"] = (f"ran {command} in {target}: "
                               f"created {result['created']}.")
            return result
        if action == "export":
            if not output:
                raise ValueError("cadcraft export needs output=<path>")
            target = Path(output)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(export_dxf(doc))
            return {"output": str(target), "entities": doc.entity_count(),
                    "reply": f"exported {path} to DXF at {target}."}
        raise ValueError(f"unknown cadcraft action {action!r}")
