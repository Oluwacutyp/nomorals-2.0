"""DesignCraft: open, read, edit, lay out and render real page
designs (DesignCraft port of storytold/designcraft — an
open-source page and layout design editor in pure Rust).

A thin surface over :mod:`nomorals.media.layout` — the organ owns
the ``.designcraft`` codec, the document model, the flex engine,
the composer, the renderer and the editor; the tool only routes
arguments and shapes the reply. Pure local work, stdlib + PIL, no
upload, no retention. Writes output files, so FS_WRITE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.policy import Capability
from ..media.layout import (
    Fill,
    LayoutDocument,
    LayoutEditor,
    LayoutError,
    Stroke,
    render_page,
)
from ..media.layout import (
    run_parity as run_layout_parity,
)
from ..media.layout import (
    verify_attribution as verify_layout_attribution,
)
from ..media.layout.codec import parse_document, to_document_bytes
from ..media.layout.flex import layout_all

__all__ = ["register"]

_ACTIONS = ("inspect", "create", "add", "style", "move", "resize",
            "group", "layout", "render", "guides", "parity",
            "attribution")


def _split_ids(value: str | list[str] | None) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [part.strip() for part in value.split(",") if part.strip()]


def _load(sources: list[str]) -> tuple[LayoutDocument, Path]:
    if not sources:
        raise ValueError("designcraft needs a source .designcraft file")
    path = Path(sources[0])
    if not path.is_file():
        raise ValueError(f"designcraft source not found: {path}")
    return parse_document(path.read_bytes()), path


def _save(doc: LayoutDocument, sources: list[str],
          output: str) -> Path:
    target = Path(output) if output else Path(sources[0])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(to_document_bytes(doc))
    return target


def register(registry: Any) -> None:
    """Attach the designcraft family to a registry."""
    @registry.register(
        "designcraft",
        description=(
            "Open, read, edit, lay out and render real page designs, "
            "local and uncapped — inspect spreads, pages, items, "
            "stories and styles; add rectangles, ovals and text; "
            "restyle fills and strokes; move, resize, group and "
            "align items; run the flex layout engine; snap to "
            "guides; render pages to real PNGs. Edits preserve the "
            "untouched items. DesignCraft actions (page/layout "
            "document model + fidelity-preserving edits + flex + "
            "renderer): inspect=<read the document inventory, "
            "sources=[path]>; create=<blank document, output=<path>, "
            "width=<w>, height=<h>, title=<t>>; add=<add a shape, "
            "kind=<rect|oval|text>, page=<id>, x=<n>, y=<n>, "
            "width=<n>, height=<n>, text=<t>, fill=<swatch>>; "
            "style=<restyle an item, node=<id>, fill=<swatch>, "
            "stroke=<color>, opacity=<0..1>>; move=<move an item, "
            "node=<id>, x=<n>, y=<n>>; resize=<resize an item, "
            "node=<id>, width=<n>, height=<n>>; group=<group items, "
            "nodes=<id1,id2>, name=<n>>; layout=<run the flex "
            "engine on the document>; render=<render to PNG, "
            "sources=[path], output=<path.png>, page=<id>, "
            "scale=<n>>; guides=<add a guide, page=<id>, "
            "axis=<vertical|horizontal>, position=<n>>; parity=<run "
            "the machine-checked parity TOML>; attribution=<verify "
            "the SHA-256 attribution DB>. action=<inspect|create|add|"
            "style|move|resize|group|layout|render|guides|parity|"
            "attribution>; sources=<list of .designcraft paths>; "
            "output=<path to write>."
        ),
        capability=Capability.FS_WRITE,
    )
    def designcraft_tool(action: str = "inspect",
                         sources: list[str] | None = None,
                         output: str = "", page: str = "",
                         kind: str = "rect", x: float = 0.0,
                         y: float = 0.0, width: float = 0.0,
                         height: float = 0.0, text: str = "",
                         fill: str = "", stroke: str = "",
                         opacity: float = -1.0, node: str = "",
                         nodes: str | list[str] | None = None,
                         name: str = "", axis: str = "vertical",
                         position: float = 0.0, title: str = "",
                         scale: float = 1.0
                         ) -> dict[str, Any]:
        action = (action or "inspect").lower()
        if action not in _ACTIONS:
            raise ValueError(f"unknown designcraft action {action!r}; "
                             f"use one of {', '.join(_ACTIONS)}")
        sources = list(sources or [])
        if action == "inspect":
            doc, path = _load(sources)
            return {"title": doc.title, "pages": doc.page_count(),
                    "items": doc.item_count(),
                    "stories": doc.story_count(),
                    "layers": [layer.name for layer in doc.layers],
                    "swatches": [s.name for s in doc.swatches],
                    "spreads": len(doc.spreads),
                    "reply": (f"{path}: .designcraft, "
                              f"{doc.page_count()} page(s), "
                              f"{doc.item_count()} item(s), "
                              f"{doc.story_count()} stories.")}
        if action == "create":
            doc = LayoutDocument.blank(title=title or "Untitled")
            if width and height:
                doc.spreads[0].pages[0].width = float(width)
                doc.spreads[0].pages[0].height = float(height)
            if not output:
                raise ValueError("designcraft create needs output=<path>")
            target = _save(doc, sources, output)
            return {"output": str(target), "pages": 1, "items": 0,
                    "reply": f"created blank layout at {target}."}
        if action == "parity":
            result = run_layout_parity()
            failed = result["failed"]
            result["reply"] = (
                f"designcraft parity: {result['passed']} passed, "
                f"{len(failed)} failed"
                + (f" ({', '.join(failed)})" if failed else "") + ".")
            return result
        if action == "attribution":
            result = verify_layout_attribution()
            failed = result["failed"]
            result["reply"] = (
                f"designcraft attribution: {result['checked']} verified, "
                f"{len(failed)} failed, {len(result['skipped'])} skipped, "
                f"{result['noted']} noted.")
            return result
        doc, path = _load(sources)
        editor = LayoutEditor(doc)
        if action == "render":
            page_id = page or doc.spreads[0].pages[0].id
            if not output:
                raise ValueError("designcraft render needs output=<path>")
            png = render_page(doc, page_id, scale=scale or 1.0)
            target = Path(output)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(png)
            return {"output": str(target), "pages": doc.page_count(),
                    "reply": f"rendered page {page_id} to {target}."}
        if action == "add":
            page_id = page or doc.spreads[0].pages[0].id
            if kind == "text":
                item = editor.add_text(page_id, x, y, width, height,
                                       text or "Text",
                                       name=name or "text")
            elif kind == "oval":
                item = editor.add_oval(page_id, x, y, width, height,
                                       fill=Fill(swatch=fill or "None"),
                                       name=name or "oval")
            else:
                item = editor.add_rect(page_id, x, y, width, height,
                                       fill=Fill(swatch=fill or "None"),
                                       name=name or "rectangle")
            target = _save(doc, sources, output)
            return {"item": item.id, "output": str(target),
                    "reply": f"added {kind} {item.id} to {target}."}
        if action == "style":
            if not node:
                raise ValueError("designcraft style needs node=<id>")
            editor.restyle(node,
                           fill=Fill(swatch=fill) if fill else None,
                           stroke=(Stroke(color=stroke, width=1.0)
                                   if stroke else None),
                           opacity=opacity if opacity >= 0 else None)
            target = _save(doc, sources, output)
            return {"item": node, "output": str(target),
                    "reply": f"restyled {node} in {target}."}
        if action == "move":
            if not node:
                raise ValueError("designcraft move needs node=<id>")
            editor.move(node, x, y)
            target = _save(doc, sources, output)
            return {"item": node, "output": str(target),
                    "reply": f"moved {node} to ({x:g}, {y:g}) in {target}."}
        if action == "resize":
            if not node:
                raise ValueError("designcraft resize needs node=<id>")
            editor.resize(node, width, height)
            target = _save(doc, sources, output)
            return {"item": node, "output": str(target),
                    "reply": f"resized {node} to {width:g}x{height:g} in "
                             f"{target}."}
        if action == "group":
            ids = _split_ids(nodes)
            if len(ids) < 2:
                raise ValueError("designcraft group needs nodes=<id1,id2>")
            group = editor.group(ids, name=name or "group")
            target = _save(doc, sources, output)
            return {"item": group.id, "output": str(target),
                    "reply": f"grouped {len(ids)} items as {group.id} in "
                             f"{target}."}
        if action == "layout":
            layout_all(doc.all_items())
            target = _save(doc, sources, output)
            return {"output": str(target),
                    "reply": f"ran the flex layout on {target}."}
        if action == "guides":
            page_id = page or doc.spreads[0].pages[0].id
            try:
                guide = editor.add_guide(page_id, axis, position)
            except LayoutError as exc:
                raise ValueError(str(exc)) from exc
            target = _save(doc, sources, output)
            return {"guide": {"axis": guide.axis,
                              "position": guide.position},
                    "output": str(target),
                    "reply": f"added {axis} guide at {position:g} in "
                             f"{target}."}
        raise ValueError(f"unknown designcraft action {action!r}")
