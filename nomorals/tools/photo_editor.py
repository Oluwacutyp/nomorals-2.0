"""Photo-editor tool: ``photo_editor`` — the brain's answer to
"open this photo" / "edit this photo".

One tool, action-dispatched, over
:mod:`nomorals.media.photo_editor`; the organ owns the document,
the PSD format and the store — this module is only the surface.

``action=open`` opens a PSD file (``path``) into the Studio
store and reports its layers. ``action=edit`` applies one ladder
edit (``op`` = crop / enhance / develop, ``params``) to the open
document. ``action=undo`` / ``action=redo`` step the history.
``action=show`` reports a document; ``action=list`` lists the open
ones; ``action=save`` writes the document back as PSD (``path``,
or the store copy when omitted); ``action=render`` writes the
current composite PNG; ``action=page`` writes the Studio page
(plain HTML, no scripts) and reports its path.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..core.policy import Capability
from ..media.photo_editor.document import (
    PhotoDocument,
    PhotoEditorError,
)
from ..media.photo_editor.page import photo_page_html
from ..media.photo_editor.store import PhotoStore

__all__ = ["register", "parse_photo_request"]


_PSD_RE = re.compile(r"([^\s\"']+\.psd)", re.IGNORECASE)


def parse_photo_request(
        text: str, known: list[tuple[str, str]]
        ) -> tuple[str, str] | None:
    """Natural phrasing -> (verb, ref) for the photo editor, or None.

    Matches only when the text names a ``.psd`` path (``open``) or
    an already-open document by id or title (``show`` / ``edit``) —
    a bare "open this photo" with nothing to open it from is not a
    match, so the hook never steals an unrelated question.
    """
    low = (text or "").lower()
    if "photo" not in low and "psd" not in low:
        return None
    match = _PSD_RE.search(text or "")
    if match is not None and ("open" in low or "edit" in low):
        return ("open", match.group(1))
    for ident, title in known:
        for name in (title, ident):
            if name and name.lower() in low:
                if "edit" in low:
                    return ("edit", ident)
                if "show" in low or "open" in low:
                    return ("show", ident)
    return None


def _home_of(context: Any) -> str | None:
    settings = (getattr(context, "settings", None)
                if context is not None else None)
    home = (getattr(settings, "home", None)
            if settings is not None else None)
    return str(home) if home else None


def _store(context: Any) -> PhotoStore:
    return PhotoStore(home=_home_of(context))


def _load(store: PhotoStore, doc_id: str
          ) -> tuple[PhotoDocument, str]:
    ref = (doc_id or "").strip()
    if not ref:
        raise ValueError("photo_editor needs a document id")
    found = store.matches(ref)
    if not found:
        raise ValueError(f"no open photo {ref!r}")
    if len(found) > 1:
        raise ValueError(
            f"photo {ref!r} is ambiguous: {len(found)} documents "
            f"match ({', '.join(f[:8] for f in found)}) — use the "
            "full id")
    doc = store.load(found[0])
    if doc is None:
        raise ValueError(f"photo {ref!r} could not be read")
    return doc, found[0]


def _summary(doc: PhotoDocument, doc_id: str) -> dict[str, Any]:
    return {
        "id": doc_id,
        "title": doc.title,
        "size": [doc.width, doc.height],
        "layers": len(doc.layers),
        "pixel_layers": sum(
            1 for layer in doc.layers if layer.kind == "pixels"),
        "layer_list": [layer.to_dict() for layer in doc.layers],
        "history": list(doc.history),
        "notes": list(doc.notes),
    }


def register(registry: Any) -> None:
    """Attach the ``photo_editor`` tool."""
    context = registry.context

    @registry.register(
        "photo_editor",
        description=(
            "Devon Studio's native photo editor (the PhotoCraft "
            "port): open a PSD file as a layered document, edit it "
            "through the image ladder, and save it back as PSD. "
            "action=open opens a PSD (path) into the Studio store. "
            "action=edit applies one edit (op=crop with "
            "box=[l,t,r,b], op=enhance with brightness/contrast/"
            "sharpness/color, op=develop with the legacy sliders "
            "brightness/contrast/exposure/shadows/highlights/"
            "saturation or the full LightCraft controls as dotted "
            "keys — light.exposure, wb.temp/tint, curve.master, "
            "mixer.red.hue, grading.shadows.hue, geometry.rotate, "
            "profile.id — or a whole settings dict under "
            "'settings'; layer=<index> picks the layer, default "
            "the topmost visible one). "
            "action=undo|redo steps the op history. action=show "
            "reports a document; action=list lists open documents. "
            "action=save writes the document as PSD (path, or the "
            "store copy when omitted). action=render writes the "
            "current composite PNG. action=page writes the Studio "
            "photo page (plain HTML). Documents persist under "
            "<home>/data/photo_editor/."
        ),
        capability=Capability.MEDIA,
    )
    def photo_editor_tool(action: str = "open", doc_id: str = "",
                          path: str = "", op: str = "",
                          params: dict | None = None,
                          out: str = "") -> dict[str, Any]:
        store = _store(context)
        verb = (action or "open").strip().lower()
        if verb == "open":
            src = (path or "").strip()
            if not src:
                raise ValueError(
                    "photo_editor open needs a PSD path")
            doc = PhotoDocument.open_psd(src)
            ident = store.save(doc)
            result = _summary(doc, ident)
            result["ok"] = True
            return result
        if verb == "list":
            docs = store.list_docs()
            return {"ok": True, "documents": docs,
                    "count": len(docs)}
        doc, ident = _load(store, doc_id)
        if verb == "show":
            result = _summary(doc, ident)
            result["ok"] = True
            return result
        if verb == "edit":
            name = (op or "").strip().lower()
            if not name:
                raise ValueError(
                    "photo_editor edit needs op=crop|enhance|"
                    "develop")
            clean = dict(params or {})
            if "box" in clean and isinstance(clean["box"], list):
                clean["box"] = tuple(clean["box"])
            try:
                doc.apply_edit(name, **clean)
            except PhotoEditorError as exc:
                raise ValueError(str(exc)) from exc
            store.save(doc, doc_id=ident)
            result = _summary(doc, ident)
            result["ok"] = True
            result["edit"] = name
            return result
        if verb == "undo":
            entry = doc.undo()
            if entry is None:
                return {"ok": True, "undone": None,
                        "note": "nothing to undo"}
            store.save(doc, doc_id=ident)
            result = _summary(doc, ident)
            result["ok"] = True
            result["undone"] = entry
            return result
        if verb == "redo":
            entry = doc.redo()
            if entry is None:
                return {"ok": True, "redone": None,
                        "note": "nothing to redo"}
            store.save(doc, doc_id=ident)
            result = _summary(doc, ident)
            result["ok"] = True
            result["redone"] = entry
            return result
        if verb == "save":
            target = (path or "").strip()
            if target:
                saved = doc.save_psd(target)
            else:
                store.save(doc, doc_id=ident)
                saved = str(store.psd_path(ident))
            result = _summary(doc, ident)
            result["ok"] = True
            result["saved"] = saved
            return result
        if verb == "render":
            target = Path(out or "").expanduser() \
                if out else None
            if target is None:
                raise ValueError(
                    "photo_editor render needs out=<png path>")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(doc.render_png_bytes())
            result = _summary(doc, ident)
            result["ok"] = True
            result["rendered"] = str(target)
            return result
        if verb == "page":
            target = Path(out or "").expanduser() \
                if out else None
            if target is None:
                raise ValueError(
                    "photo_editor page needs out=<html path>")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                photo_page_html(doc, doc_id=ident),
                encoding="utf-8")
            result = _summary(doc, ident)
            result["ok"] = True
            result["page"] = str(target)
            return result
        raise ValueError(f"unknown photo_editor action {verb!r}")
