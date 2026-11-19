"""MaterialCraft: design, cook, preview and export node-based PBR
materials (MaterialCraft port of storytold/materialcraft — an
open-source clean-room node-based material designer in Rust).

A thin surface over :mod:`nomorals.media.material` — the organ owns
the ``.mcraft`` codec, the document model, the node library, the
topological cook + content-addressed cache, the PBR preview
renderer and the preset library; the tool only routes arguments
and shapes the reply. Pure local work, stdlib + numpy + PIL, no
upload, no retention. Writes output files, so FS_WRITE.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from ..core.policy import Capability
from ..media.material import (
    MaterialDocument,
    MaterialEditor,
    MaterialError,
    build_preset,
    list_presets,
    search,
)
from ..media.material import (
    run_parity as run_material_parity,
)
from ..media.material import (
    verify_attribution as verify_material_attribution,
)
from ..media.material.eval import CookCache, cook
from ..media.material.pbr import render_preview

__all__ = ["register"]

_ACTIONS = ("inspect", "create", "add", "connect", "set", "cook",
            "render", "library", "preset", "export", "import",
            "parity", "attribution")

_CACHE = CookCache()


def _load(sources: list[str]) -> tuple[MaterialDocument, Path]:
    if not sources:
        raise ValueError("materialcraft needs a source file")
    path = Path(sources[0])
    if not path.is_file():
        raise ValueError(f"materialcraft source not found: {path}")
    try:
        return MaterialDocument.parse(path.read_bytes()), path
    except MaterialError as exc:
        raise ValueError(str(exc)) from exc


def _save(doc: MaterialDocument, sources: list[str],
          output: str) -> Path:
    target = Path(output) if output else Path(sources[0])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(doc.to_bytes())
    return target


def _parse_params(params: dict | None) -> dict:
    return dict(params or {})


def register(registry: Any) -> None:
    """Attach the materialcraft family to a registry."""
    @registry.register(
        "materialcraft",
        description=(
            "Design, cook, preview and export node-based PBR materials, "
            "local and uncapped — inspect materials, node graphs and "
            "channels; add nodes from the 26-node library (Perlin and "
            "cellular noise, Voronoi, bricks, tiles, gradients, blend, "
            "levels, blur, HSL, warp, normal-from-height); wire nodes "
            "into PBR channels (albedo, roughness, metallic, normal, "
            "height, ao, emissive); cook through the content-addressed "
            "cache (unchanged graphs return cached maps, edits re-cook "
            "only downstream); render a GGX material-ball preview; use "
            "the preset library (metal, wood, stone, fabric, glass, "
            "brick); export portable .mcmat materials. MaterialCraft "
            "actions (node-graph material designer + cook cache + PBR "
            "preview): inspect=<read the material inventory, "
            "sources=[path]>; create=<blank material, output=<path>, "
            "title=<t>>; add=<add a node, kind=<def id>, node=<id>, "
            "x=<n>, y=<n>, params=<dict>>; connect=<wire two nodes, "
            "node=<from>, nodes=<to>, value=<from-port>, "
            "command=<to-port>>; set=<set a parameter or output, "
            "node=<id>, kind=<param|output>, value=<v>, "
            "params=<dict>>; cook=<cook the graph, sources=[path], "
            "output=<path.png prefix>>; render=<render the PBR "
            "preview, sources=[path], output=<path.png>>; library=<"
            "list/search presets, value=<query>>; preset=<build a "
            "preset material, value=<id>, output=<path>>; export=<"
            "export a portable .mcmat, sources=[path], "
            "output=<path.mcmat>>; import=<import a .mcmat, "
            "sources=[path.mcmat], output=<path.mcraft>>; parity=<run "
            "the machine-checked parity TOML>; attribution=<verify the "
            "SHA-256 attribution DB>. action=<inspect|create|add|"
            "connect|set|cook|render|library|preset|export|import|"
            "parity|attribution>; sources=<list of .mcraft paths>; "
            "output=<path to write>."
        ),
        capability=Capability.FS_WRITE,
    )
    def materialcraft_tool(action: str = "inspect",
                           sources: list[str] | None = None,
                           output: str = "", kind: str = "",
                           node: str = "", nodes: str | list[str] | None = None,
                           x: float = 0.0, y: float = 0.0,
                           value: Any = None,
                           command: str = "",
                           params: dict | None = None,
                           title: str = "",
                           ) -> dict[str, Any]:
        action = (action or "inspect").lower()
        if action not in _ACTIONS:
            raise ValueError(f"unknown materialcraft action {action!r}; "
                             f"use one of {', '.join(_ACTIONS)}")
        sources = list(sources or [])
        if action == "parity":
            result = run_material_parity()
            result["reply"] = (
                f"materialcraft parity: {result['passed']} passed, "
                f"{result['failed']} failed.")
            return result
        if action == "attribution":
            result = verify_material_attribution()
            result["reply"] = (
                f"materialcraft attribution: {result['verified']} verified, "
                f"{result['failed']} failed, {result['skipped']} skipped, "
                f"{result['noted']} noted.")
            return result
        if action == "library":
            query = str(value or "")
            presets = search(query) if query else list_presets()
            return {"presets": presets,
                    "reply": f"{len(presets)} presets"
                             + (f" matching {query!r}." if query else ".")}
        if action == "preset":
            preset_id = str(value or "")
            if not preset_id:
                raise ValueError("materialcraft preset needs value=<id>")
            graph = build_preset(preset_id)
            doc = MaterialDocument(name=graph.name, graphs=[graph])
            doc.mark_dirty()
            if not output:
                raise ValueError("materialcraft preset needs output=<path>")
            target = _save(doc, sources, output)
            return {"preset": preset_id, "nodes": len(graph.nodes),
                    "channels": sorted(graph.outputs.keys()),
                    "output": str(target),
                    "reply": (f"built preset {preset_id} "
                              f"({len(graph.nodes)} nodes) at {target}.")}
        if action == "create":
            doc = MaterialDocument.blank(title or "Untitled")
            if not output:
                raise ValueError("materialcraft create needs output=<path>")
            target = _save(doc, sources, output)
            return {"output": str(target), "nodes": 0,
                    "reply": f"created blank material at {target}."}
        if action == "import":
            if not sources:
                raise ValueError("materialcraft import needs sources=[path.mcmat]")
            path = Path(sources[0])
            if not path.is_file():
                raise ValueError(f"materialcraft source not found: {path}")
            try:
                doc, manifest = MaterialDocument.import_mcmat(path.read_bytes())
            except MaterialError as exc:
                raise ValueError(str(exc)) from exc
            if not output:
                raise ValueError("materialcraft import needs output=<path>")
            target = _save(doc, sources, output)
            return {"output": str(target), "channels": manifest.get("channels", []),
                    "reply": (f"imported {path} as {target} "
                              f"(channels: {', '.join(manifest.get('channels', []))}).")}
        doc, path = _load(sources)
        editor = MaterialEditor(doc)
        if action == "inspect":
            graph = doc.graph
            return {"title": doc.name, "nodes": len(graph.nodes),
                    "connections": len(graph.connections),
                    "outputs": sorted(graph.outputs.keys()),
                    "size": graph.size,
                    "fidelity": doc.fidelity_report(),
                    "reply": (f"{path}: material {doc.name!r}, "
                              f"{len(graph.nodes)} nodes, "
                              f"{len(graph.connections)} wires, "
                              f"channels: {', '.join(sorted(graph.outputs.keys())) or 'none'}.")}
        if action == "add":
            if not kind:
                raise ValueError("materialcraft add needs kind=<node def id>")
            try:
                with editor.edit():
                    item = editor.add_node(kind, node or None, x, y,
                                           _parse_params(params))
            except MaterialError as exc:
                raise ValueError(str(exc)) from exc
            target = _save(doc, sources, output)
            return {"node": item.id, "def": kind, "output": str(target),
                    "reply": f"added {kind} node {item.id} to {target}."}
        if action == "connect":
            if not node or not nodes:
                raise ValueError("materialcraft connect needs node=<from> "
                                 "and nodes=<to>")
            dst = nodes[0] if isinstance(nodes, list) else str(nodes)
            try:
                with editor.edit():
                    editor.connect(node, dst,
                                   str(value or "output"),
                                   command or "input")
            except MaterialError as exc:
                raise ValueError(str(exc)) from exc
            target = _save(doc, sources, output)
            return {"from": node, "to": dst, "output": str(target),
                    "reply": f"wired {node} -> {dst} in {target}."}
        if action == "set":
            if not node:
                raise ValueError("materialcraft set needs node=<id>")
            try:
                with editor.edit():
                    if kind == "output":
                        editor.set_output(str(value or ""), node)
                    elif params:
                        for key, val in _parse_params(params).items():
                            editor.set_param(node, key, val)
                    elif kind:
                        editor.set_param(node, kind, value)
                    else:
                        raise ValueError("materialcraft set needs kind=<param> "
                                         "or params=<dict>")
            except MaterialError as exc:
                raise ValueError(str(exc)) from exc
            target = _save(doc, sources, output)
            return {"node": node, "output": str(target),
                    "reply": f"set {node} in {target}."}
        if action == "cook":
            try:
                cooked = cook(doc.graph, _CACHE)
            except MaterialError as exc:
                raise ValueError(str(exc)) from exc
            result: dict[str, Any] = {
                "channels": sorted(cooked.outputs.keys()),
                "nodes_cooked": cooked.nodes_cooked,
                "cache_hits": cooked.cache_hits,
                "errors": cooked.errors,
                "size": list(cooked.size),
                "reply": (f"cooked {path}: channels "
                          f"{', '.join(sorted(cooked.outputs.keys())) or 'none'}, "
                          f"{cooked.nodes_cooked} nodes cooked, "
                          f"{cooked.cache_hits} cache hits."),
            }
            if output:
                written = []
                for channel, image in cooked.outputs.items():
                    target = Path(f"{output}.{channel}.png")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(_png_bytes(image))
                    written.append(str(target))
                result["maps"] = written
            return result
        if action == "render":
            if not output:
                raise ValueError("materialcraft render needs output=<path>")
            try:
                cooked = cook(doc.graph, _CACHE)
            except MaterialError as exc:
                raise ValueError(str(exc)) from exc
            image, report = render_preview(cooked.outputs)
            target = Path(output)
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target)
            return {"output": str(target), "channels": report["channels_used"],
                    "not_rendered": report["not_rendered"],
                    "reply": (f"rendered preview of {path} to {target} "
                              f"(channels: {', '.join(report['channels_used'])}; "
                              f"not rendered: {', '.join(report['not_rendered'])}).")}
        if action == "export":
            if not output:
                raise ValueError("materialcraft export needs output=<path>")
            try:
                cooked = cook(doc.graph, _CACHE)
            except MaterialError as exc:
                raise ValueError(str(exc)) from exc
            maps = {channel: _png_bytes(image)
                    for channel, image in cooked.outputs.items()}
            target = Path(output)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(doc.export_mcmat(maps))
            return {"output": str(target), "channels": sorted(maps.keys()),
                    "reply": (f"exported {path} to portable material "
                              f"{target} ({', '.join(sorted(maps.keys()))}).")}
        raise ValueError(f"unknown materialcraft action {action!r}")


def _png_bytes(image) -> bytes:
    buf = io.BytesIO()
    image.to_8bit().save(buf, format="PNG")
    return buf.getvalue()
