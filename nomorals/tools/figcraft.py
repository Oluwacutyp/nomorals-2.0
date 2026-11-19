"""FigCraft: open, read, edit, prototype, render and save real
.figcraft interface designs — plus live two-editor collaboration
(FigCraft port of storytold/figcraft — a clean-room interface
design and prototyping app).

A thin surface over :mod:`nomorals.media.design` — the organ
owns the .figcraft package, the document model, the auto-layout
engine, the component system, the prototype flows, the
collaborative editing core and the frame renderer; the tool only
routes arguments and shapes the reply. Pure local work, stdlib +
PIL only, no upload, no retention. Writes output files, so
FS_WRITE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.policy import Capability
from ..media.design import (
    DesignDocument,
    DesignEditor,
    add_interaction,
    flows,
    render_frame,
    resolve_instance,
    run_parity,
    verify_attribution,
)
from ..media.design.session import SessionClient, SessionServer

__all__ = ["register"]

_ACTIONS = ("inspect", "create", "add", "set", "style", "text",
            "component", "instance", "prototype", "render", "collab",
            "parity", "attribution")


def register(registry: Any) -> None:
    """Attach the figcraft family to a registry."""
    @registry.register(
        "figcraft",
        description=(
            "Open, read, edit, prototype, render and save real "
            ".figcraft interface designs, local and uncapped — "
            "inspect frames, components, variants, instances, "
            "auto-layout, styles, tokens and prototype flows; add "
            "frames, rectangles and text, restyle fills and tokens, "
            "wire prototype interactions, resolve component "
            "instances, render a frame to PNG, or run a live "
            "two-editor collaboration session that merges "
            "concurrently. Edits preserve the untouched nodes "
            "byte-for-byte. FigCraft actions (design model + "
            "fidelity-preserving edits + frame renderer + collab "
            "core): inspect=<read the design inventory, "
            "sources=[path]>; create=<blank design, output=<path>, "
            "title=<t>>; add=<add a node, kind=<rect|text|frame>, "
            "parent=<id>, name=<n>, x=<px>, y=<px>, width=<px>, "
            "height=<px>, fill=<#hex>, text=<t>>; set=<set a text "
            "node's characters or a node's fill, node=<id>, "
            "text=<t>, fill=<#hex>>; style=<restyle a node, "
            "node=<id>, fill=<#hex>, opacity=<0-1>, name=<n>, "
            "corner_radius=<px>>; text=<set text, node=<id>, "
            "text=<t>>; component=<add a component or resolve an "
            "instance, kind=<component|instance|resolve>, "
            "parent=<id>, source=<id>, node=<id>>; instance=<add an "
            "instance of a component, parent=<id>, source=<id>>; "
            "prototype=<wire an interaction or list flows, "
            "kind=<interaction|flows>, node=<id>, trigger=<on_click>, "
            "action=<navigate>, destination=<id>, page=<id>>; "
            "render=<render a frame to PNG, node=<id>, output=<path>>; "
            "collab=<run a two-editor merge simulation on a design, "
            "sources=[path]>; parity=<run the machine-checked parity "
            "TOML>; attribution=<verify the SHA-256 attribution DB>. "
            "action=<inspect|create|add|set|style|text|component|"
            "instance|prototype|render|collab|parity|attribution>; "
            "sources=<list of .figcraft paths>; output=<path to write>."
        ),
        capability=Capability.FS_WRITE,
    )
    def figcraft_tool(action: str = "inspect",
                      sources: list[str] | None = None,
                      output: str = "", text: str = "",
                      node: int = 0, parent: int = 0, source: int = 0,
                      page: int = 0, kind: str = "",
                      name: str = "", x: float = 0.0, y: float = 0.0,
                      width: float = 0.0, height: float = 0.0,
                      fill: str = "", title: str = "",
                      trigger: str = "on-click", interaction: str = "navigate",
                      destination: int = 0, opacity: float = -1.0,
                      corner_radius: float = -1.0,
                      path: str = "", out: str = "",
                      ) -> dict[str, Any]:
        action = (action or "inspect").lower()
        if action not in _ACTIONS:
            raise ValueError(f"unknown figcraft action {action!r}; use "
                             f"one of {', '.join(_ACTIONS)}")
        sources = list(sources or [])
        if path and not sources:
            sources = [path]
        if out and not output:
            output = out
        if action == "parity":
            result = run_parity()
            failed_ids = list(result["failed"])
            return {"ok": True, "passed": result["passed"],
                    "failed": len(failed_ids),
                    "failed_ids": failed_ids,
                    "reply": f"figcraft parity: {result['passed']} "
                             f"passed, {len(failed_ids)} failed"
                             + (f" ({', '.join(failed_ids)})" if failed_ids else "")
                             + "."}
        if action == "attribution":
            result = verify_attribution()
            return {"ok": True, "verified": result["verified"],
                    "failed": len(result["failed"]),
                    "skipped": len(result["skipped"]),
                    "failed_items": result["failed"],
                    "skipped_items": result["skipped"],
                    "reply": f"figcraft attribution: {result['verified']} "
                             f"verified, {len(result['failed'])} failed, "
                             f"{len(result['skipped'])} skipped."}
        if action == "create":
            if not output:
                raise ValueError("figcraft create needs output=<path>")
            doc = DesignDocument.blank(title or "Untitled")
            saved = doc.to_bytes()
            out_path = Path(output).expanduser()
            out_path.write_bytes(saved)
            return {"ok": True, "output": str(out_path),
                    "nodes": len(doc.nodes), "size_after": len(saved),
                    "reply": f"created blank .figcraft at {out_path}."}
        if action == "collab":
            if not sources:
                raise ValueError("figcraft collab needs a source .figcraft")
            doc = DesignDocument.from_bytes(Path(sources[0]).read_bytes())
            server = SessionServer(doc)
            alice = SessionClient(server, "alice")
            bob = SessionClient(server, "bob")
            alice.connect()
            bob.connect()
            text_nodes = [nid for nid, n in doc.nodes.items()
                          if n.get("type") == "text"]
            rect_nodes = [nid for nid, n in doc.nodes.items()
                          if n.get("type") == "rectangle"]
            if text_nodes:
                alice.set_text(text_nodes[0], "Alice text")
            if rect_nodes:
                bob.set_fill(rect_nodes[0], "#ff0000")
            alice.receive()
            bob.receive()
            alice.receive()
            bob.receive()
            assert alice.doc is not None and bob.doc is not None
            converged = alice.doc.to_dict() == bob.doc.to_dict()
            return {"ok": True, "converged": converged,
                    "alice_nodes": len(alice.doc.nodes),
                    "bob_nodes": len(bob.doc.nodes),
                    "reply": "two-editor simulation: "
                             + ("both editors converged."
                                if converged else
                                "editors DIVERGED — merge is broken.")}
        if action == "inspect":
            if not sources:
                raise ValueError("figcraft inspect needs a source .figcraft")
            doc = DesignDocument.from_bytes(Path(sources[0]).read_bytes())
            summary = doc.to_dict()
            kinds: dict[str, int] = {}
            for n in doc.nodes.values():
                kinds[str(n.get("type"))] = kinds.get(str(n.get("type")), 0) + 1
            summary["kinds"] = kinds
            summary["reply"] = (
                f"{sources[0]}: .figcraft, {len(doc.nodes)} node(s), "
                f"{len(doc.pages)} page(s), {len(doc.images)} image(s), "
                f"{len(doc.styles)} style(s).")
            return summary
        if action == "render":
            if not sources:
                raise ValueError("figcraft render needs a source .figcraft")
            if not output:
                raise ValueError("figcraft render needs output=<path>")
            doc = DesignDocument.from_bytes(Path(sources[0]).read_bytes())
            png, info = render_frame(doc, node)
            out_path = Path(output).expanduser()
            out_path.write_bytes(png)
            return {"ok": True, "output": str(out_path),
                    "width": info["width"], "height": info["height"],
                    "unrendered": info["unrendered"],
                    "reply": f"rendered frame {node} to {out_path} "
                             f"({info['width']}x{info['height']})."}
        if not sources:
            raise ValueError(f"figcraft {action} needs a source .figcraft")
        if not output:
            raise ValueError(f"figcraft {action} needs output=<path>")
        original = Path(sources[0]).read_bytes()
        doc = DesignDocument.from_bytes(original)
        editor = DesignEditor(doc)
        if action == "add":
            kind = (kind or "rect").lower()
            if kind == "rect":
                nid = editor.add_rect(parent, name or "Rectangle", x, y,
                                      width, height, fill or "#4472c4")
            elif kind == "text":
                nid = editor.add_text(parent, name or "Text", text, x, y,
                                      width or 200.0)
            elif kind == "frame":
                nid = editor.add_frame(parent, name or "Frame", x, y,
                                       width, height,
                                       fill or None)
            else:
                raise ValueError(
                    f"figcraft add kind {kind!r}; use rect, text or frame")
            detail = f"added {kind} {nid} to parent {parent}"
        elif action in ("set", "text"):
            if text:
                editor.set_text(node, text)
                detail = f"set text of node {node} to {text!r}"
            elif fill:
                editor.set_fill(node, fill)
                detail = f"set fill of node {node} to {fill}"
            else:
                raise ValueError(
                    "figcraft set needs text=<t> or fill=<#hex>")
        elif action == "style":
            kwargs: dict[str, Any] = {}
            if fill:
                kwargs["fill"] = fill
            if opacity >= 0.0:
                kwargs["opacity"] = opacity
            if name:
                kwargs["name"] = name
            if corner_radius >= 0.0:
                kwargs["corner_radius"] = corner_radius
            if not kwargs:
                raise ValueError(
                    "figcraft style needs fill, opacity, name or "
                    "corner_radius")
            editor.restyle(node, **kwargs)
            detail = f"restyled node {node}"
        elif action == "component":
            kind = (kind or "component").lower()
            if kind == "component":
                nid = editor.add_component(parent, name or "Component",
                                           width, height)
                detail = f"added component {nid} to parent {parent}"
            elif kind == "instance":
                nid = editor.add_instance(parent, source)
                detail = f"added instance {nid} of component {source}"
            elif kind == "resolve":
                resolved = resolve_instance(doc, node)
                return {"ok": True, "exact": resolved["exact"],
                        "component": resolved["component"],
                        "root": resolved["tree"].get("name"),
                        "reply": f"instance {node} resolves to "
                                 f"{resolved['tree'].get('name')!r} "
                                 f"(exact={resolved['exact']})."}
            else:
                raise ValueError(
                    f"figcraft component kind {kind!r}; use component, "
                    f"instance or resolve")
        elif action == "instance":
            nid = editor.add_instance(parent, source)
            detail = f"added instance {nid} of component {source}"
        elif action == "prototype":
            kind = (kind or "interaction").lower()
            if kind == "interaction":
                entry = add_interaction(doc, node, trigger, interaction,
                                        destination or None)
                detail = (f"wired {trigger} on node {node} to "
                          f"{interaction} -> {destination}")
                _ = entry
            elif kind == "flows":
                result = flows(doc, page)
                return {"ok": True, "flows": result,
                        "reply": f"page {page} has flows from {result}."}
            else:
                raise ValueError(
                    f"figcraft prototype kind {kind!r}; use interaction "
                    f"or flows")
        else:  # pragma: no cover - guarded by _ACTIONS above
            raise ValueError(f"unknown figcraft action {action!r}")
        saved = editor.to_bytes()
        out_path = Path(output).expanduser()
        out_path.write_bytes(saved)
        report = editor.fidelity_report(original)
        return {"ok": True, "output": str(out_path),
                "nodes": len(doc.nodes),
                "size_before": len(original), "size_after": len(saved),
                "changed": report["changed"],
                "reply": f"{detail}; saved to {out_path}."}
