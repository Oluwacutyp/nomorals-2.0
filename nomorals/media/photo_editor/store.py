"""Durable photo sessions — the BoardStore shape, for photos.

One directory per open document under ``<home>/data/photo_editor/``:

* ``doc.psd`` — the document itself, as PSD (the current pixels +
  layers; a save writes the whole document, so a load never has to
  replay the journal through a ladder that may have changed).
* ``meta.json`` — id, title, size, layer list, the op journal
  (audit, not replay), and timestamps.

``matches(ref)`` is prefix matching over ids, like BoardStore:
an ambiguous prefix names its matches instead of guessing.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from ...core.ids import new_id
from .document import PhotoDocument, PhotoEditorError

__all__ = ["PhotoStore", "default_photo_dir"]


def default_photo_dir(home: str | None = None) -> str:
    from ...core.failure_store import home_dir as _devon_home_dir

    base = home or _devon_home_dir()
    return str(Path(base) / "data" / "photo_editor")


class PhotoStore:
    """Open photo documents, durable under a home directory."""

    def __init__(self, home: str | None = None,
                 root: str | None = None) -> None:
        self.root = Path(root or default_photo_dir(home))
        self.root.mkdir(parents=True, exist_ok=True)

    # -- paths ---------------------------------------------------------
    def _dir(self, doc_id: str) -> Path:
        return self.root / doc_id

    def _psd_path(self, doc_id: str) -> Path:
        return self._dir(doc_id) / "doc.psd"

    def _meta_path(self, doc_id: str) -> Path:
        return self._dir(doc_id) / "meta.json"

    def psd_path(self, doc_id: str) -> Path:
        """The stored PSD for a document id."""
        return self._psd_path(doc_id)

    # -- write -----------------------------------------------------------
    def save(self, doc: PhotoDocument, title: str = "",
             doc_id: str = "") -> str:
        """Persist a document; return its id (minted when new)."""
        ident = (doc_id or "").strip() or new_id()
        folder = self._dir(ident)
        folder.mkdir(parents=True, exist_ok=True)
        doc.save_psd(self._psd_path(ident))
        meta = {
            "id": ident,
            "title": title or doc.title or "untitled",
            "width": doc.width,
            "height": doc.height,
            "layers": [layer.to_dict() for layer in doc.layers],
            "history": list(doc.history),
            "notes": list(doc.notes),
            "updated_at": time.time(),
        }
        tmp = self._meta_path(ident).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        os.replace(tmp, self._meta_path(ident))
        return ident

    # -- read ------------------------------------------------------------
    def matches(self, ref: str) -> list[str]:
        """Ids starting with ``ref`` (prefix match, like boards)."""
        needle = (ref or "").strip()
        if not needle:
            return []
        found = []
        for entry in sorted(self.root.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name.startswith(needle) \
                    and self._psd_path(entry.name).exists():
                found.append(entry.name)
        return found

    def load(self, ref: str) -> PhotoDocument | None:
        """Load by id or unambiguous prefix; None when missing,
        an error when ambiguous."""
        found = self.matches(ref)
        if not found:
            if self._psd_path(ref).exists():
                found = [ref]
            else:
                return None
        if len(found) > 1:
            raise PhotoEditorError(
                f"photo {ref!r} is ambiguous: {len(found)} "
                f"documents match "
                f"({', '.join(f[:8] for f in found)}) — use the "
                "full id")
        doc_id = found[0]
        doc = PhotoDocument.open_psd(self._psd_path(doc_id))
        meta = self.meta(doc_id)
        if meta is not None:
            doc.title = meta.get("title") or doc.title
            doc.history = list(meta.get("history") or [])
            doc.notes = list(meta.get("notes") or [])
        return doc

    def meta(self, doc_id: str) -> dict | None:
        path = self._meta_path(doc_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def list_docs(self) -> list[dict]:
        out = []
        for entry in sorted(self.root.iterdir()):
            if not entry.is_dir():
                continue
            meta = self.meta(entry.name)
            if meta is not None:
                out.append(meta)
        return out

    def delete(self, ref: str) -> bool:
        found = self.matches(ref)
        if not found:
            return False
        if len(found) > 1:
            raise PhotoEditorError(
                f"photo {ref!r} is ambiguous: {len(found)} "
                "documents match — use the full id")
        folder = self._dir(found[0])
        for child in folder.iterdir():
            child.unlink()
        folder.rmdir()
        return True
