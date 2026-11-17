"""Board tool: the whiteboard organ for the brain.

Devon draws on a board the way she draws a scene — she composes
shapes, binds them with arrows, moves them, undoes, and renders the
result to a PNG or a live page she can keep drawing on. One tool,
action-dispatched, over :mod:`nomorals.media.edit_engine.board`:
the organ owns the model, the store, the render and the page; this
module is only the surface.

``action=create`` makes a board (optionally from a diagram spec —
``diagram`` carries the scene organ's diagram dict, nodes + edges).
``add``/``move``/``update``/``remove``/``bind``/``unbind`` edit it,
``undo``/``redo`` step the op history, ``show``/``list``/``delete``
read and manage it, ``render`` writes the PNG, ``page`` writes the
editable live page. Boards persist under ``<home>/boards/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.policy import Capability
from ..media.edit_engine.board import (
    BoardDocument,
    BoardError,
    BoardStore,
    board_from_diagram,
    render_board,
    write_board_page,
)

__all__ = ["register"]


def _home_of(context: Any) -> str | None:
    settings = (getattr(context, "settings", None)
                if context is not None else None)
    home = (getattr(settings, "home", None)
            if settings is not None else None)
    return str(home) if home else None


def _store(context: Any) -> BoardStore:
    return BoardStore(home=_home_of(context))


def _load(store: BoardStore, board_id: str) -> BoardDocument:
    ref = (board_id or "").strip()
    if not ref:
        raise ValueError("board needs a board id")
    found = store.matches(ref)
    if not found:
        raise ValueError(f"no board {ref!r}")
    if len(found) > 1:
        raise ValueError(
            f"board {ref!r} is ambiguous: {len(found)} boards match "
            f"({', '.join(f[:8] for f in found)}) — use the full id")
    doc = store.load(found[0])
    if doc is None:
        raise ValueError(f"board {ref!r} could not be read")
    return doc


def _summary(doc: BoardDocument) -> dict[str, Any]:
    return {
        "id": doc.id,
        "title": doc.title,
        "theme": doc.theme,
        "shapes": len(doc.shapes),
        "bindings": len(doc.bindings),
        "version": doc.version,
        "hash": doc.doc_hash[:12],
        "shape_list": [s.to_dict() for s in doc.shapes],
        "binding_list": [b.to_dict() for b in doc.bindings],
    }


def register(registry: Any) -> None:
    """Attach the ``board`` tool."""
    context = registry.context

    @registry.register(
        "board",
        description=(
            "The whiteboard: create and edit a board of shapes, "
            "arrows, text, sticky notes and freehand strokes, and "
            "render it to a PNG or a live editable page. "
            "action=create makes a board (title, theme=clean|sketch, "
            "or diagram=<diagram spec dict> to build it from nodes + "
            "edges). action=add adds a shape (shape dict with id/type/"
            "x/y/w/h/props). action=move moves a shape (shape_id, x, "
            "y). action=update changes a shape (shape_id, changes "
            "dict — props merge). action=remove deletes a shape (its "
            "bindings drop with it). action=bind connects two shapes "
            "with an arrow (from_shape, to_shape, label). "
            "action=unbind drops a binding (binding_id). "
            "action=undo|redo steps the op history. action=show "
            "reports a board; action=list lists boards; action=delete "
            "removes one. action=render writes the PNG (theme, size); "
            "action=page writes the editable live page. Boards persist "
            "under <home>/boards/."
        ),
        capability=Capability.MEDIA,
    )
    def board_tool(action: str = "create", board_id: str = "",
                   title: str = "", theme: str = "",
                   diagram: dict | None = None,
                   shape: dict | None = None, shape_id: str = "",
                   x: float | None = None, y: float | None = None,
                   changes: dict | None = None,
                   from_shape: str = "", to_shape: str = "",
                   label: str = "", binding_id: str = "",
                   size: str = "", out: str = "") -> dict[str, Any]:
        store = _store(context)
        verb = (action or "create").strip().lower()
        if verb == "create":
            use_theme = (theme or "clean").strip().lower()
            if diagram is not None:
                doc = board_from_diagram(diagram, title=title,
                                         theme=use_theme)
            else:
                doc = BoardDocument.create(title, theme=use_theme)
            store.save(doc)
            result = _summary(doc)
            result["ok"] = True
            result["action"] = "create"
            return result
        if verb == "list":
            return {"ok": True, "action": "list",
                    "boards": store.list_boards()}
        if verb == "add":
            if not isinstance(shape, dict) or not shape:
                raise ValueError("board add needs a shape dict")
            doc = _load(store, board_id)
            created = doc.add_shape(shape)
            store.save(doc)
            return {"ok": True, "action": "add", "board_id": doc.id,
                    "shape": created.to_dict(),
                    "shapes": len(doc.shapes)}
        if verb == "move":
            if x is None or y is None:
                raise ValueError("board move needs x and y")
            doc = _load(store, board_id)
            moved = doc.update_shape(shape_id, x=float(x), y=float(y))
            store.save(doc)
            return {"ok": True, "action": "move", "board_id": doc.id,
                    "x": moved.x, "y": moved.y}
        if verb == "update":
            if not isinstance(changes, dict) or not changes:
                raise ValueError("board update needs a changes dict")
            doc = _load(store, board_id)
            updated = doc.update_shape(shape_id, **changes)
            store.save(doc)
            return {"ok": True, "action": "update",
                    "board_id": doc.id,
                    "shape": updated.to_dict()}
        if verb == "remove":
            doc = _load(store, board_id)
            result = doc.remove_shape(shape_id)
            store.save(doc)
            return {"ok": True, "action": "remove",
                    "board_id": doc.id,
                    "removed_bindings": result["removed_bindings"],
                    "shapes": len(doc.shapes)}
        if verb == "bind":
            doc = _load(store, board_id)
            binding = doc.bind(from_shape, to_shape, label=label)
            store.save(doc)
            return {"ok": True, "action": "bind", "board_id": doc.id,
                    "binding": binding.to_dict(),
                    "bindings": len(doc.bindings)}
        if verb == "unbind":
            doc = _load(store, board_id)
            doc.unbind(binding_id)
            store.save(doc)
            return {"ok": True, "action": "unbind",
                    "board_id": doc.id,
                    "bindings": len(doc.bindings)}
        if verb == "undo":
            doc = _load(store, board_id)
            try:
                doc.undo()
            except BoardError as exc:
                raise ValueError(str(exc)) from None
            store.save(doc)
            return {"ok": True, "action": "undo", "board_id": doc.id,
                    "shapes": len(doc.shapes),
                    "can_undo": doc.can_undo,
                    "can_redo": doc.can_redo}
        if verb == "redo":
            doc = _load(store, board_id)
            try:
                doc.redo()
            except BoardError as exc:
                raise ValueError(str(exc)) from None
            store.save(doc)
            return {"ok": True, "action": "redo", "board_id": doc.id,
                    "shapes": len(doc.shapes),
                    "can_undo": doc.can_undo,
                    "can_redo": doc.can_redo}
        if verb == "show":
            doc = _load(store, board_id)
            result = _summary(doc)
            result["ok"] = True
            result["action"] = "show"
            return result
        if verb == "delete":
            doc = _load(store, board_id)
            store.delete(doc.id)
            return {"ok": True, "action": "delete",
                    "board_id": doc.id}
        if verb == "render":
            doc = _load(store, board_id)
            dims = (1280, 720)
            if size:
                try:
                    wide, tall = size.lower().split("x")
                    dims = (int(wide), int(tall))
                except ValueError:
                    raise ValueError(
                        "board render size must be WxH, got "
                        f"{size!r}") from None
            out_dir = None
            if out:
                out_path = Path(out)
                if out_path.suffix.lower() == ".png":
                    record = render_board(
                        doc, theme=theme or None, size=dims,
                        path=str(out_path))
                else:
                    out_dir = str(out_path)
                    record = render_board(
                        doc, theme=theme or None, size=dims,
                        out_dir=out_dir)
            else:
                home = _home_of(context)
                out_dir = (str(Path(home) / "boards")
                           if home else None)
                record = render_board(doc, theme=theme or None,
                                      size=dims, out_dir=out_dir)
            record["ok"] = True
            record["action"] = "render"
            record["board_id"] = doc.id
            return record
        if verb == "page":
            doc = _load(store, board_id)
            home = _home_of(context)
            out_dir = (str(Path(home) / "boards")
                       if home else None)
            target = Path(out_dir) / f"board-{doc.id[:8]}.html" \
                if out_dir else Path(f"board-{doc.id[:8]}.html")
            path = write_board_page(doc, target, theme=theme or None)
            return {"ok": True, "action": "page", "board_id": doc.id,
                    "path": path}
        raise ValueError(f"unknown board action {action!r}")
