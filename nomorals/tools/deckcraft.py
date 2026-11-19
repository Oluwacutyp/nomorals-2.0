"""DeckCraft: open, read, edit, render and save real .pptx files
(DeckCraft port of storytold/deckcraft — a clean-room presentation
app).

A thin surface over :mod:`nomorals.documents.decks` — the organ
owns the OOXML package, the document model, the fidelity-preserving
editor and the slide renderer; the tool only routes arguments and
shapes the reply. Pure local work, stdlib + PIL only, no upload, no
retention. Writes output files, so FS_WRITE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.policy import Capability
from ..documents.decks import (
    DeckDocument,
    DeckEditor,
)
from ..documents.decks import (
    render_slide as render_deck_slide,
)
from ..documents.decks import (
    run_parity as run_deckcraft_parity,
)
from ..documents.decks import (
    verify_attribution as verify_deckcraft_attribution,
)

__all__ = ["register"]

_ACTIONS = ("inspect", "create", "replace", "style", "add", "table",
            "image", "notes", "render", "reorder", "remove",
            "metadata", "parity", "attribution")


def _split_list(value: str | list[str] | None) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [part.strip() for part in value.split(",") if part.strip()]


def _split_rows(value: str | list[Any] | None) -> list[list[str]]:
    if not value:
        return []
    if isinstance(value, list):
        return [[str(cell) for cell in row] for row in value]
    rows: list[list[str]] = []
    for row in value.split(";"):
        rows.append([cell.strip() for cell in row.split("|")])
    return rows


def register(registry: Any) -> None:
    """Attach the deckcraft family to a registry."""
    @registry.register(
        "deckcraft",
        description=(
            "Open, read, edit, render and save real .pptx "
            "presentations, local and uncapped — inspect slides, "
            "shapes, tables, charts, images, notes and transitions; "
            "replace text, restyle runs, add slides, text boxes and "
            "tables, embed pictures, set speaker notes and "
            "transitions, reorder or remove slides, set document "
            "metadata, render a slide to PNG, or create a blank deck. "
            "Edits preserve the untouched parts byte-for-byte. "
            "DeckCraft actions (presentation model + "
            "fidelity-preserving edits + slide renderer): "
            "inspect=<read the deck inventory, sources=[path]>; "
            "create=<blank deck, output=<path>, title=<t>, "
            "author=<a>>; replace=<replace text, text=<old>, "
            "new_text=<new>, first_only=<true|false>>; style=<restyle, "
            "slide=<n>, shape=<n>, para=<n>, run=<n>, bold=<true|false>, "
            "italic=<true|false>, color=<hex>, size=<half-points>>; "
            "add=<add a slide or text box, kind=<slide|textbox>, "
            "slide=<n>, text=<t>, layout=<content|title>>; "
            "table=<append a table to a slide, slide=<n>, "
            "headers=<a,b>, rows=<1|2;3|4>>; image=<embed a picture, "
            "slide=<n>, sources=[image path]>; notes=<set speaker "
            "notes, slide=<n>, text=<t>>; render=<render a slide to "
            "PNG, slide=<n>, output=<path>, width=<px>>; "
            "reorder=<move a slide, slide=<n>, to=<n>>; "
            "remove=<remove a slide, slide=<n>>; metadata=<set "
            "title/author, title=<t>, author=<a>>; parity=<run the "
            "machine-checked parity TOML>; attribution=<verify the "
            "SHA-256 attribution DB>. action=<inspect|create|replace|"
            "style|add|table|image|notes|render|reorder|remove|"
            "metadata|parity|attribution>; sources=<list of .pptx "
            "paths>; output=<path to write>."
        ),
        capability=Capability.FS_WRITE,
    )
    def deckcraft_tool(action: str = "inspect",
                       sources: list[str] | None = None,
                       output: str = "", text: str = "",
                       new_text: str = "", first_only: bool = False,
                       slide: int = 0, shape: int = 0, para: int = 0,
                       run: int = 0, bold: bool | None = None,
                       italic: bool | None = None,
                       color: str = "", size: int = 0,
                       kind: str = "slide", layout: str = "content",
                       headers: str | list[str] | None = None,
                       rows: str | list[Any] | None = None,
                       title: str = "", author: str = "",
                       to: int = 0, width: int = 960
                       ) -> dict[str, Any]:
        action = (action or "inspect").lower()
        if action not in _ACTIONS:
            raise ValueError(f"unknown deckcraft action {action!r}; use "
                             f"one of {', '.join(_ACTIONS)}")
        sources = list(sources or [])
        if action == "inspect":
            if not sources:
                raise ValueError("deckcraft inspect needs a source .pptx")
            summary = DeckDocument.from_bytes(
                Path(sources[0]).read_bytes()).to_dict()
            summary["reply"] = (
                f"{sources[0]}: .pptx, "
                f"{len(summary['slides'])} slide(s), "
                f"{sum(len(s['shapes']) for s in summary['slides'])} "
                f"shape(s), {len(summary['images'])} image(s), "
                f"{len(summary['layouts'])} layout(s).")
            return summary
        if action == "parity":
            result = run_deckcraft_parity()
            failed = [c["id"] for c in result["cases"]
                      if c["status"] == "fail"]
            result["reply"] = (
                f"deckcraft parity: {result['passed']} passed, "
                f"{result['failed']} failed"
                + (f" ({', '.join(failed)})" if failed else "") + ".")
            return result
        if action == "attribution":
            result = verify_deckcraft_attribution()
            result["reply"] = (
                f"deckcraft attribution: {result['checked']} verified, "
                f"{result['failed']} failed, "
                f"{result['skipped']} skipped (clone not on disk).")
            return result
        if action == "create":
            if not output:
                raise ValueError("deckcraft create needs output=<path>")
            editor = DeckEditor.create(title=title or None,
                                       author=author or None)
            saved = editor.save()
            out_path = Path(output).expanduser()
            out_path.write_bytes(saved)
            return {"ok": True, "output": str(out_path),
                    "slides": len(editor.document.slides),
                    "size_after": len(saved),
                    "reply": f"created blank .pptx at {out_path}."}
        if action == "render":
            if not sources:
                raise ValueError("deckcraft render needs a source .pptx")
            if not output:
                raise ValueError("deckcraft render needs output=<path>")
            doc = DeckDocument.from_bytes(Path(sources[0]).read_bytes())
            result = render_deck_slide(doc, slide, width=width)
            out_path = Path(output).expanduser()
            out_path.write_bytes(result["png"])
            return {"ok": True, "output": str(out_path),
                    "width": result["width"],
                    "height": result["height"],
                    "unrendered": result["unrendered"],
                    "reply": f"rendered slide {slide} to {out_path} "
                             f"({result['width']}x{result['height']})."}
        if not sources:
            raise ValueError(f"deckcraft {action} needs a source .pptx")
        if not output:
            raise ValueError(f"deckcraft {action} needs output=<path>")
        original = Path(sources[0]).read_bytes()
        editor = DeckEditor.from_bytes(original)
        if action == "replace":
            if not text:
                raise ValueError("deckcraft replace needs text=<old>")
            replaced = editor.replace_text(text, new_text,
                                           first_only=first_only)
            detail = f"replaced {replaced} occurrence(s) of " \
                     f"{text!r} with {new_text!r}"
        elif action == "style":
            editor.set_run_format(
                slide, shape, para, run, bold=bold, italic=italic,
                color=color or None, size_half_points=size or None)
            detail = (f"run {run} of paragraph {para} on shape "
                      f"{shape} of slide {slide} restyled")
        elif action == "add":
            kind = (kind or "slide").lower()
            if kind == "slide":
                index = editor.add_slide(layout=layout, title=text
                                         or None)
                detail = f"added slide {index} from layout {layout!r}"
            elif kind == "textbox":
                shape_id = editor.add_text_box(slide, text)
                detail = f"added text box {shape_id} to slide {slide}"
            else:
                raise ValueError(
                    f"deckcraft add kind {kind!r}; use slide or "
                    f"textbox")
        elif action == "table":
            head = _split_list(headers)
            body = _split_rows(rows)
            if not head and not body:
                raise ValueError(
                    "deckcraft table needs headers=<a,b> and/or "
                    "rows=<1|2;3|4>")
            shape_id = editor.add_table(slide, head, body)
            detail = (f"added table {shape_id} ({len(head)} column(s), "
                      f"{len(body)} row(s)) to slide {slide}")
        elif action == "image":
            if len(sources) < 2:
                raise ValueError(
                    "deckcraft image needs sources=[deck.pptx, "
                    "image.png]")
            image_bytes = Path(sources[1]).read_bytes()
            shape_id = editor.add_image(slide, image_bytes)
            detail = f"embedded picture {shape_id} on slide {slide}"
        elif action == "notes":
            editor.set_notes(slide, text)
            detail = f"speaker notes set on slide {slide}"
        elif action == "reorder":
            editor.move_slide(slide, to)
            detail = f"moved slide {slide} to position {to}"
        elif action == "remove":
            editor.remove_slide(slide)
            detail = f"removed slide {slide}"
        elif action == "metadata":
            fields: dict[str, str] = {}
            if title:
                fields["title"] = title
            if author:
                fields["author"] = author
            if not fields:
                raise ValueError(
                    "deckcraft metadata needs title=<t> and/or "
                    "author=<a>")
            editor.set_metadata(**fields)
            detail = "metadata set: " + ", ".join(
                f"{key}={value!r}" for key, value in fields.items())
        else:  # pragma: no cover - guarded by _ACTIONS above
            raise ValueError(f"unknown deckcraft action {action!r}")
        saved = editor.save()
        out_path = Path(output).expanduser()
        out_path.write_bytes(saved)
        result: dict[str, Any] = {
            "ok": True, "output": str(out_path),
            "slides": len(editor.document.slides),
            "size_before": len(original),
            "size_after": len(saved),
            "reply": f"{detail}; saved to {out_path}."}
        if action == "replace":
            result["replacements"] = replaced
        return result
