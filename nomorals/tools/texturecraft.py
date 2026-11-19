"""TextureCraft: open, read, edit, paint, tile and render real
.tcraft texture projects — plus procedural generation, mipmaps,
normal-map derivation and UV padding (TextureCraft port of
storytold/texturecraft — an open-source 3D texture painter).

A thin surface over :mod:`nomorals.media.textures` — the organ owns
the tile model, the noise, the blend modes, the patterns, the
compositor, the brush engine, the mipmaps, the normal derivation and
the .tcraft codec; the tool only routes arguments and shapes the
reply. Pure local work, stdlib + numpy + PIL only, no upload, no
retention. Writes output files, so FS_WRITE.
"""

from __future__ import annotations

from typing import Any

from ..core.policy import Capability
from ..media.textures import TextureDocument, run_parity, verify_attribution
from ..media.textures.compose import composite
from ..media.textures.edit import TextureEditor
from ..media.textures.mipmaps import mipmap_chain
from ..media.textures.render import save_preview

__all__ = ["register"]

_ACTIONS = ("inspect", "create", "add", "paint", "fill", "pattern",
            "blend", "compose", "render", "mipmap", "normal", "dilate",
            "filter", "resize", "undo", "parity", "attribution")


def register(registry: Any) -> None:
    """Attach the texturecraft family to a registry."""
    @registry.register(
        "texturecraft",
        description=(
            "Open, read, edit, paint, tile and render real .tcraft "
            "texture projects, local and uncapped — inspect texture "
            "sets, layers and tiles; create blank textures; add paint, "
            "fill, group and instance layers; paint strokes and fills; "
            "generate procedural textures (clouds, bricks, wood, "
            "voronoi and 19 more, seeded and deterministic); blend "
            "layers with the 32 W3C blend modes; composite channels; "
            "render PNG previews at any mipmap level; build mipmap "
            "chains and sample them (nearest/bilinear/trilinear); "
            "derive normal maps from height with real Sobel gradients; "
            "pad UV seams by BFS dilation. Edits preserve the "
            "untouched tiles byte-for-byte. TextureCraft actions "
            "(sparse 64x64 COW tiles + procedural patterns + layer "
            "compositor): inspect=<read the project inventory, "
            "path=<.tcraft>>; create=<blank texture, out=<path>, "
            "title=<t>, width=<px>, height=<px>>; add=<add a layer, "
            "kind=<paint|fill|group|instance>, name=<n>, set=<set>, "
            "fill=<uniform|procedural>, pattern=<pattern>, seed=<n>>; "
            "paint=<paint a stroke or rectangle, layer=<id>, "
            "channel=<base_color|...>, path=<\"x,y;x,y\">, "
            "color=<\"r,g,b,a\">, size=<px>>; fill=<fill a layer, "
            "layer=<id>, channel=<c>, color=<\"r,g,b,a\">>; pattern=<fill "
            "a layer with a procedural pattern, layer=<id>, "
            "channel=<c>, pattern=<kind>, seed=<n>, scale=<n>>; blend="
            "<set a layer's blend mode/opacity, layer=<id>, "
            "channel=<c>, mode=<multiply|...>, opacity=<0-1>>; compose="
            "<composite a channel in memory, path=<.tcraft>, set=<set>, "
            "channel=<c>>; render=<render a PNG preview, "
            "path=<.tcraft>, set=<set>, channel=<c>, out=<png>, "
            "level=<mip>>; mipmap=<build the mipmap chain, "
            "path=<.tcraft>, set=<set>, channel=<c>>; normal=<derive a "
            "normal map from height, path=<.tcraft>, set=<set>, "
            "strength=<n>, out=<.tcraft>>; dilate=<pad UV seams, "
            "path=<.tcraft>, set=<set>, channel=<c>, max_dist=<n>>; "
            "filter=<apply a filter to a layer, layer=<id>, "
            "channel=<c>, kind=<blur|invert|levels|...>>; resize=<resize "
            "a set, set=<set>, width=<px>, height=<px>>; undo=<undo the "
            "last edit>; parity=<run the machine-checked parity TOML>; "
            "attribution=<verify the SHA-256 attribution DB>. action="
            "<inspect|create|add|paint|fill|pattern|blend|compose|"
            "render|mipmap|normal|dilate|filter|resize|undo|parity|"
            "attribution>; path=<.tcraft path>; out=<path to write>."
        ),
        capability=Capability.FS_WRITE,
    )
    def texturecraft_tool(action: str = "inspect",
                         path: str = "", out: str = "",
                         title: str = "", width: int = 128,
                         height: int = 128, set: str = "Body",
                         kind: str = "paint", name: str = "",
                         layer: int = 0, channel: str = "base_color",
                         fill: str = "", pattern: str = "",
                         seed: int = 1, scale: float = 4.0,
                         mode: str = "normal", opacity: float = 1.0,
                         color: str = "", path_points: str = "",
                         size: float = 16.0, level: int = 0,
                         max_dist: int = 8, strength: float = 1.0,
                         value: float = 0.0,
                         ) -> dict[str, Any]:
        action = (action or "inspect").lower()
        if action not in _ACTIONS:
            raise ValueError(f"unknown texturecraft action {action!r}; use "
                             f"one of {', '.join(_ACTIONS)}")
        if action == "parity":
            result = run_parity()
            failed_ids = list(result["failed"])
            return {"ok": True, "passed": result["passed"],
                    "failed": len(failed_ids),
                    "failed_ids": failed_ids,
                    "reply": f"texturecraft parity: {result['passed']} "
                             f"passed, {len(failed_ids)} failed"
                             + (f" ({', '.join(failed_ids)})" if failed_ids
                                else "") + "."}
        if action == "attribution":
            result = verify_attribution()
            return {"ok": True, "verified": result["verified"],
                    "failed": len(result["failed"]),
                    "skipped": len(result["skipped"]),
                    "failed_items": result["failed"],
                    "skipped_items": result["skipped"],
                    "reply": f"texturecraft attribution: "
                             f"{result['verified']} verified, "
                             f"{len(result['failed'])} failed, "
                             f"{len(result['skipped'])} skipped."}
        if action == "create":
            if not out:
                raise ValueError("create needs out=<path>")
            doc = TextureDocument.create_blank(title or "Untitled",
                                               width, height)
            doc.save(out)
            return {"ok": True, "path": out, "title": doc.title,
                    "width": width, "height": height,
                    "reply": f"created blank texture {width}x{height} "
                             f"at {out}."}
        if not path:
            raise ValueError(f"{action} needs path=<.tcraft>")
        doc = TextureDocument.open(path)
        if action == "inspect":
            return {"ok": True, **doc.inspect(),
                    "reply": f"{doc.title}: {len(doc.sets)} set(s), "
                             f"{sum(len(s.layers) for s in doc.sets)} "
                             f"layer(s)."}
        ed = TextureEditor(doc)
        if action == "add":
            fill_def = None
            if kind == "fill":
                if fill == "uniform":
                    fill_def = {"kind": "uniform",
                                "color": _parse_color(color, (65535, 65535,
                                                             65535, 65535))}
                elif fill == "procedural":
                    fill_def = {"kind": "procedural", "pattern": pattern,
                                "seed": seed, "scale": scale}
                else:
                    raise ValueError(
                        "add a fill layer with fill=<uniform|procedural>")
            layer_id = ed.add_layer(set, name or "Layer", kind=kind,
                                    fill=fill_def)
            doc.save(path)
            return {"ok": True, "layer_id": layer_id,
                    "reply": f"added {kind} layer {layer_id} "
                             f"({name or 'Layer'}) to {set}."}
        if action in ("paint", "fill", "pattern", "blend", "filter"):
            if layer <= 0:
                raise ValueError(f"{action} needs layer=<id>")
            if action == "paint":
                color_vals = _parse_color(
                    color, (65535, 65535, 65535, 65535))
                if path_points:
                    pts = _parse_points(path_points)
                    ed.paint_stroke(layer, channel, pts, color_vals,
                                    size=size)
                else:
                    ed.paint_rect(layer, channel, 0, 0,
                                  doc.width, doc.height, color_vals)
            elif action == "fill":
                ed.fill_layer(layer, channel,
                              _parse_color(color, (65535, 65535, 65535, 65535)))
            elif action == "pattern":
                ed.pattern_fill(layer, channel, pattern, seed=seed,
                                scale=scale)
            elif action == "blend":
                ed.set_blend(set, layer, channel, mode, opacity=opacity)
            elif action == "filter":
                ed.apply_filter_to_layer(set, layer, channel,
                                         (kind, value))
            doc.save(path)
            return {"ok": True, "layer_id": layer,
                    "reply": f"{action} applied to layer {layer} "
                             f"({channel})."}
        if action == "compose":
            img = composite(doc, set, channel)
            return {"ok": True, "channel": channel,
                    "allocated_tiles": img.allocated_count(),
                    "width": img.width, "height": img.height,
                    "reply": f"composited {channel} in {set}: "
                             f"{img.allocated_count()} allocated tile(s)."}
        if action == "render":
            if not out:
                raise ValueError("render needs out=<png path>")
            report = save_preview(doc, set, channel, out, level=level)
            return {"ok": True, **report,
                    "reply": f"rendered {channel} at mip level {level} "
                             f"to {out} ({report['width']}x"
                             f"{report['height']})."}
        if action == "mipmap":
            img = composite(doc, set, channel)
            chain = mipmap_chain(img)
            return {"ok": True, "channel": channel,
                    "levels": len(chain),
                    "sizes": [[c.width, c.height] for c in chain],
                    "reply": f"mipmap chain for {channel}: "
                             f"{len(chain)} levels, "
                             f"{chain[0].width}x{chain[0].height} down to "
                             f"{chain[-1].width}x{chain[-1].height}."}
        if action == "normal":
            report = ed.derive_normal(set, strength=strength)
            if out:
                doc.save(out)
            else:
                doc.save(path)
            return {"ok": True, **report,
                    "reply": f"derived a normal map from height "
                             f"(layer {report['layer_id']})."}
        if action == "dilate":
            report = ed.dilate_channel(set, channel, max_dist=max_dist)
            doc.save(path)
            return {"ok": True, **report,
                    "reply": f"padded {report['padded_texels']} texel(s) "
                             f"in {channel} (layer "
                             f"{report['layer_id']})."}
        if action == "resize":
            ed.resize(set, width, height)
            doc.save(path)
            return {"ok": True, "width": width, "height": height,
                    "reply": f"resized {set} to {width}x{height}."}
        if action == "undo":
            done = ed.undo()
            if done:
                doc.save(path)
            return {"ok": done, "undone": done,
                    "reply": "undid the last edit."
                             if done else "nothing to undo."}
        raise ValueError(f"unhandled texturecraft action {action!r}")


def _parse_color(text: str, default: tuple[int, ...]) -> list[int]:
    if not text:
        return list(default)
    parts = [int(p.strip()) for p in text.split(",") if p.strip() != ""]
    if not parts:
        return list(default)
    return parts


def _parse_points(text: str) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        x, y = chunk.split(",")
        pts.append((float(x), float(y)))
    return pts
