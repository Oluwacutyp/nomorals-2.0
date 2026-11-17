"""``nm board`` — the whiteboard organ on the command line.

    nm board new <title> [--theme clean|sketch] [--diagram-file spec.json]
    nm board add <board> <type> <label> [--x N] [--y N] [--w N] [--h N] [--color #hex] [--points "x,y;x,y"]
    nm board move <board> <shape> <x> <y>
    nm board remove <board> <shape>
    nm board bind <board> <from> <to> [--label text]
    nm board unbind <board> <binding>
    nm board undo|redo <board>
    nm board show <board> [--json]
    nm board list [--json]
    nm board delete <board>
    nm board render <board> [--theme clean|sketch] [--out path.png] [--size WxH]
    nm board page <board> [--out path.html]

Boards and shapes take an id or a unique id prefix. Nothing here
raises a traceback — failures print cleanly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _err(msg: str) -> int:
    print(f"board: {msg}", file=sys.stderr)
    return 1


def _ok(msg: str) -> int:
    print(msg)
    return 0


def _home_path(context: Any) -> Path | None:
    settings = getattr(context, "settings", None)
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(
        settings, "home", None)
    return Path(home) if home else None


def _store(context: Any):
    from ...media.edit_engine.board import BoardStore

    return BoardStore(home=_home_path(context))


def _load_board(store, ref: str):
    found = store.matches(ref or "")
    if not found:
        return None, f"no board {ref!r} — try `nm board list`"
    if len(found) > 1:
        return None, (f"board {ref!r} matches {len(found)} boards — "
                      "use the full id")
    doc = store.load(found[0])
    if doc is None:
        return None, f"board {ref!r} could not be read"
    return doc, None


def _find_shape(doc, ref: str):
    if doc.shape(ref) is not None:
        return doc.shape(ref), None
    hits = [s for s in doc.shapes if s.id.startswith(ref)]
    if len(hits) == 1:
        return hits[0], None
    if not hits:
        return None, f"no shape {ref!r} on this board"
    return None, (f"shape {ref!r} matches {len(hits)} shapes — "
                  "use the full id")


def _cmd_new(args: argparse.Namespace, context) -> int:
    from ...media.edit_engine.board import (
        BoardDocument,
        board_from_diagram,
    )

    theme = (getattr(args, "theme", "") or "clean").lower()
    if theme not in ("clean", "sketch"):
        return _err("theme must be clean or sketch")
    try:
        diagram_file = getattr(args, "diagram_file", "") or ""
        if diagram_file:
            spec = json.loads(Path(diagram_file).read_text(
                encoding="utf-8"))
            doc = board_from_diagram(spec, title=args.title,
                                     theme=theme)
        else:
            doc = BoardDocument.create(args.title, theme=theme)
    except Exception as exc:  # noqa: BLE001 - CLI surfaces as text
        return _err(str(exc))
    store = _store(context)
    store.save(doc)
    return _ok(f"board {doc.id[:8]} — {doc.title or '(untitled)'} "
               f"({theme}, {len(doc.shapes)} shape(s))")


def _cmd_add(args: argparse.Namespace, context) -> int:
    from ...media.edit_engine.board import SHAPE_TYPES

    shape_type = (getattr(args, "shape_type", "") or "").lower()
    if shape_type not in SHAPE_TYPES:
        return _err(f"shape type must be one of {list(SHAPE_TYPES)}")
    store = _store(context)
    doc, error = _load_board(store, getattr(args, "board_ref", ""))
    if doc is None:
        return _err(error or "board not found")
    shape: dict[str, Any] = {"type": shape_type}
    for key in ("x", "y", "w", "h"):
        value = getattr(args, key, None)
        if value is not None:
            shape[key] = float(value)
    props: dict[str, Any] = {}
    label = getattr(args, "label", "") or ""
    if shape_type in ("rect", "ellipse", "text", "sticky"):
        props["text"] = label
    elif shape_type == "arrow":
        props["label"] = label
    color = getattr(args, "color", "") or ""
    if color:
        props["color"] = color
    points = getattr(args, "points", "") or ""
    if points:
        try:
            props["points"] = [
                [float(part) for part in pair.split(",")]
                for pair in points.split(";")]
        except ValueError:
            return _err("points must look like \"x,y;x,y\"")
    if props:
        shape["props"] = props
    try:
        created = doc.add_shape(shape)
    except Exception as exc:  # noqa: BLE001 - CLI surfaces as text
        return _err(str(exc))
    store.save(doc)
    return _ok(f"added {created.type} {created.id[:8]} "
               f"({created.x:.0f}, {created.y:.0f}) to board "
               f"{doc.id[:8]}")


def _cmd_move(args: argparse.Namespace, context) -> int:
    store = _store(context)
    doc, error = _load_board(store, getattr(args, "board_ref", ""))
    if doc is None:
        return _err(error or "board not found")
    shape, error = _find_shape(doc, getattr(args, "shape_ref", ""))
    if shape is None:
        return _err(error or "shape not found")
    try:
        doc.update_shape(shape.id, x=float(args.x), y=float(args.y))
    except Exception as exc:  # noqa: BLE001 - CLI surfaces as text
        return _err(str(exc))
    store.save(doc)
    return _ok(f"moved {shape.id[:8]} to ({args.x:.0f}, {args.y:.0f})")


def _cmd_remove(args: argparse.Namespace, context) -> int:
    store = _store(context)
    doc, error = _load_board(store, getattr(args, "board_ref", ""))
    if doc is None:
        return _err(error or "board not found")
    shape, error = _find_shape(doc, getattr(args, "shape_ref", ""))
    if shape is None:
        return _err(error or "shape not found")
    try:
        result = doc.remove_shape(shape.id)
    except Exception as exc:  # noqa: BLE001 - CLI surfaces as text
        return _err(str(exc))
    store.save(doc)
    dropped = result.get("removed_bindings") or []
    return _ok(f"removed {shape.id[:8]}"
               + (f" — {len(dropped)} binding(s) dropped"
                  if dropped else ""))


def _cmd_bind(args: argparse.Namespace, context) -> int:
    store = _store(context)
    doc, error = _load_board(store, getattr(args, "board_ref", ""))
    if doc is None:
        return _err(error or "board not found")
    src, error = _find_shape(doc, getattr(args, "from_ref", ""))
    if src is None:
        return _err(error or "shape not found")
    dst, error = _find_shape(doc, getattr(args, "to_ref", ""))
    if dst is None:
        return _err(error or "shape not found")
    try:
        binding = doc.bind(src.id, dst.id,
                           label=getattr(args, "label", "") or "")
    except Exception as exc:  # noqa: BLE001 - CLI surfaces as text
        return _err(str(exc))
    store.save(doc)
    return _ok(f"bound {src.id[:8]} → {dst.id[:8]} "
               f"(arrow {binding.arrow_shape[:8]})")


def _cmd_unbind(args: argparse.Namespace, context) -> int:
    store = _store(context)
    doc, error = _load_board(store, getattr(args, "board_ref", ""))
    if doc is None:
        return _err(error or "board not found")
    ref = getattr(args, "binding_ref", "")
    binding = doc.binding(ref)
    if binding is None:
        hits = [b for b in doc.bindings if b.id.startswith(ref)]
        if len(hits) == 1:
            binding = hits[0]
        else:
            return _err(f"no binding {ref!r} on this board")
    try:
        doc.unbind(binding.id)
    except Exception as exc:  # noqa: BLE001 - CLI surfaces as text
        return _err(str(exc))
    store.save(doc)
    return _ok(f"unbound {binding.id[:8]} (the arrow stays)")


def _cmd_undo_redo(args: argparse.Namespace, context,
                   *, redo: bool = False) -> int:
    store = _store(context)
    doc, error = _load_board(store, getattr(args, "board_ref", ""))
    if doc is None:
        return _err(error or "board not found")
    try:
        if redo:
            doc.redo()
        else:
            doc.undo()
    except Exception as exc:  # noqa: BLE001 - CLI surfaces as text
        return _err(str(exc))
    store.save(doc)
    verb = "redid" if redo else "undid"
    return _ok(f"{verb} — {len(doc.shapes)} shape(s) now")


def _cmd_show(args: argparse.Namespace, context) -> int:
    store = _store(context)
    doc, error = _load_board(store, getattr(args, "board_ref", ""))
    if doc is None:
        return _err(error or "board not found")
    if getattr(args, "json", False):
        print(json.dumps(doc.to_dict(), indent=2))
        return 0
    lines = [f"board {doc.id[:8]} — {doc.title or '(untitled)'} "
             f"({doc.theme}, v{doc.version}):"]
    for shape in doc.shapes:
        label = (shape.props.get("text", "")
                 or shape.props.get("label", ""))
        lines.append(
            f"  {shape.id[:8]}  {shape.type} "
            f"({shape.x:.0f}, {shape.y:.0f}) {shape.w:.0f}x{shape.h:.0f}"
            + (f" — {label}" if label else ""))
    for binding in doc.bindings:
        lines.append(
            f"  ↳ {binding.arrow_shape[:8]} binds "
            f"{binding.from_shape[:8]} → {binding.to_shape[:8]}"
            + (f" — {binding.label}" if binding.label else ""))
    if not doc.shapes:
        lines.append("  (empty — add a shape to draw on it)")
    return _ok("\n".join(lines))


def _cmd_list(args: argparse.Namespace, context) -> int:
    store = _store(context)
    rows = store.list_boards()
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        return _ok("no boards yet — `nm board new <title>` makes one.")
    lines = ["boards:"]
    for row in rows:
        lines.append(
            f"  {row['id'][:8]}  {row['title'] or '(untitled)'} — "
            f"{row['shapes']} shape(s), {row['bindings']} binding(s), "
            f"{row['theme']}")
    return _ok("\n".join(lines))


def _cmd_delete(args: argparse.Namespace, context) -> int:
    store = _store(context)
    doc, error = _load_board(store, getattr(args, "board_ref", ""))
    if doc is None:
        return _err(error or "board not found")
    store.delete(doc.id)
    return _ok(f"deleted board {doc.id[:8]}")


def _cmd_render(args: argparse.Namespace, context) -> int:
    from ...media.edit_engine.board import render_board

    store = _store(context)
    doc, error = _load_board(store, getattr(args, "board_ref", ""))
    if doc is None:
        return _err(error or "board not found")
    theme = (getattr(args, "theme", "") or "").lower()
    if theme and theme not in ("clean", "sketch"):
        return _err("theme must be clean or sketch")
    dims = (1280, 720)
    size = getattr(args, "size", "") or ""
    if size:
        try:
            wide, tall = size.lower().split("x")
            dims = (int(wide), int(tall))
        except ValueError:
            return _err(f"size must be WxH, got {size!r}")
    out = getattr(args, "out", "") or ""
    try:
        if out and Path(out).suffix.lower() == ".png":
            record = render_board(doc, theme=theme or None,
                                  size=dims, path=out)
        else:
            home = _home_path(context)
            out_dir = (str(Path(home) / "boards")
                       if home else out or None)
            record = render_board(doc, theme=theme or None,
                                  size=dims, out_dir=out_dir)
    except Exception as exc:  # noqa: BLE001 - CLI surfaces as text
        return _err(str(exc))
    if getattr(args, "json", False):
        print(json.dumps(record, indent=2))
        return 0
    return _ok(f"board rendered — {record['path']} "
               f"({record['shapes']} shape(s), {record['theme']} theme, "
               f"hash {record['render_hash'][:12]})")


def _cmd_page(args: argparse.Namespace, context) -> int:
    from ...media.edit_engine.board import write_board_page

    store = _store(context)
    doc, error = _load_board(store, getattr(args, "board_ref", ""))
    if doc is None:
        return _err(error or "board not found")
    out = getattr(args, "out", "") or ""
    home = _home_path(context)
    if not out:
        out_dir = Path(home) / "boards" if home else Path.cwd()
        out = str(out_dir / f"board-{doc.id[:8]}.html")
    try:
        path = write_board_page(doc, out)
    except Exception as exc:  # noqa: BLE001 - CLI surfaces as text
        return _err(str(exc))
    return _ok(f"live board page — {path}")


def cmd_board(args: argparse.Namespace, context) -> int:
    """``nm board <verb> ...`` — dispatch."""
    action = getattr(args, "board_action", "")
    if action == "new":
        return _cmd_new(args, context)
    if action == "add":
        return _cmd_add(args, context)
    if action == "move":
        return _cmd_move(args, context)
    if action == "remove":
        return _cmd_remove(args, context)
    if action == "bind":
        return _cmd_bind(args, context)
    if action == "unbind":
        return _cmd_unbind(args, context)
    if action == "undo":
        return _cmd_undo_redo(args, context)
    if action == "redo":
        return _cmd_undo_redo(args, context, redo=True)
    if action == "show":
        return _cmd_show(args, context)
    if action == "list":
        return _cmd_list(args, context)
    if action == "delete":
        return _cmd_delete(args, context)
    if action == "render":
        return _cmd_render(args, context)
    if action == "page":
        return _cmd_page(args, context)
    return _err("usage: nm board new|add|move|remove|bind|unbind|"
                "undo|redo|show|list|delete|render|page ...")
