"""Per-user sticker packs (DESIGN §21): the stealer's other
half.

``/steal`` has always claimed a replied sticker into HER
library (``expression.py``) — the room-facing collection her
moments draw from.  A pack is the thief's own copy: the
stickers a person has stolen, keyed by the §29 person
identity (canonical platform + the platform's own id, the
``group_gangs.person_key`` discipline — never display names,
never fuzzy merging), persisted as one JSON document beside
the expression library with the same store discipline
(atomic tmp+replace saves; a corrupt file is quarantined,
never silently read as an empty pack).

Capture is honest about platforms:

* **Bytes in hand** (the adapter downloaded the sticker):
  WebP stays WebP, WebM stays WebM — those ARE her sticker
  formats (Telegram static/WhatsApp, Telegram video).
  Another raster image is normalized to a real WebP through
  the conversion organ.  The bytes are COPIED into the pack's
  own directory under the media store — a pack owns its
  media; adapter download caches are not a home.
* **Bytes out of reach**: the item is stored BY REFERENCE —
  the platform file_id is the pack entry, re-sends go by
  handle, and the entry says so.  Nothing pretends a capture
  happened that didn't.
* **.tgs (Lottie)**: Telegram's animated-sticker JSON rides
  as a file_id or a gzip download named ``.webp``.  It is
  stored by reference with the reason named — Devon holds
  the handle, not a render she cannot honestly produce.

A stolen sticker enters the GROUP's expression pool only
through her library (``/steal``'s existing capture, and the
admin-enabled auto-collect §21 names is a pipeline hook,
not a pack behavior) — packs are personal by construction.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

from ...core.ids import new_short_id
from ...core.logging_setup import get_logger

_log = get_logger("nomorals.social.sticker_packs")

__all__ = [
    "PackError", "StickerPackStore", "capture_sticker",
    "MAX_PACK_ITEMS",
]

#: A pack is a working collection, not an archive: full is a
#: plain refusal, never a silent eviction of someone's media.
MAX_PACK_ITEMS = 200


class PackError(RuntimeError):
    """A steal that could not honestly land in the pack."""


def _sniff(head: bytes) -> str:
    """Format from magic bytes: "webp" | "webm" | "tgs"
    (gzip — the .tgs container) | "" when unrecognized."""
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    if head[:2] == b"\x1f\x8b":
        return "tgs"
    return ""


class StickerPackStore:
    """All packs, one JSON document: ``{person_key: [item]}``.

    An item: ``{id, format, path, file_ids, by_reference,
    note, name, origin, sha, uses, created}`` — ``path`` is
    the pack-owned copy ("" for by-reference items),
    ``file_ids`` the platform handles captured at steal time.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.recovered = False
        self.data: dict[str, Any] = {"packs": {}, "meta": {}}
        self.load()

    # -- persistence (ExpressionLibrary discipline) --------
    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) \
                    or not isinstance(raw.get("packs"), dict):
                raise ValueError("pack root is not an object")
            self.data = raw
        except (OSError, ValueError) as exc:
            quarantine = self.path.with_suffix(
                f".corrupt-{int(time.time())}")
            try:
                self.path.replace(quarantine)
            except OSError:
                pass
            _log.warning(
                "sticker packs at %s unreadable (%s); "
                "quarantined to %s — starting empty",
                self.path, exc, quarantine)
            self.recovered = True
            self.data = {"packs": {}, "meta": {}}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1),
                       encoding="utf-8")
        tmp.replace(self.path)

    # -- pack operations ------------------------------------
    def pack(self, person: str) -> list[dict[str, Any]]:
        return list((self.data.get("packs") or {}).get(person)
                    or [])

    def __len__(self) -> int:
        return sum(len(v) for v in
                   (self.data.get("packs") or {}).values())

    def get(self, person: str,
            index: int) -> dict[str, Any] | None:
        """One item by its 1-based pack position (the number
        ``/sticker pack`` lists and accepts)."""
        items = self.pack(person)
        if 1 <= int(index) <= len(items):
            return items[int(index) - 1]
        return None

    def find(self, person: str, *, sha: str = "",
             file_ids: dict[str, str] | None = None
             ) -> dict[str, Any] | None:
        """The existing twin, if the pack already holds this
        media (same bytes, or a shared platform handle) — a
        pack never holds the same sticker twice."""
        file_ids = {str(k): str(v) for k, v in
                    (file_ids or {}).items() if v}
        for item in self.pack(person):
            if sha and item.get("sha") == sha:
                return item
            held = item.get("file_ids") or {}
            if any(held.get(k) == v for k, v in file_ids.items()):
                return item
        return None

    def add(self, person: str, item: dict[str, Any]) -> dict[str, Any]:
        packs = self.data.setdefault("packs", {})
        items = packs.setdefault(person, [])
        if len(items) >= MAX_PACK_ITEMS:
            raise PackError(
                f"your sticker pack is full ({MAX_PACK_ITEMS}) "
                f"— it's a working collection, not an archive")
        items.append(item)
        self.save()
        return item

    def note_use(self, person: str, item_id: str) -> None:
        for item in self.pack(person):
            if item.get("id") == item_id:
                item["uses"] = int(item.get("uses") or 0) + 1
                self.save()
                return


def _reference_item(*, file_ids: dict[str, str], note: str,
                    name: str, origin: str) -> dict[str, Any]:
    return {
        "id": "sp_" + new_short_id(), "format": "reference",
        "path": "", "file_ids": file_ids, "by_reference": True,
        "note": note, "name": str(name or "stolen sticker"),
        "origin": str(origin or ""), "sha": "", "uses": 0,
        "created": time.time(),
    }


def capture_sticker(store: StickerPackStore, person: str, *,
                    path: str = "", kind: str = "",
                    file_ids: dict[str, str] | None = None,
                    platform: str = "", media_dir: Path | str = "",
                    origin: str = "", name: str = ""
                    ) -> dict[str, Any]:
    """Land one stolen sticker in ``person``'s pack.

    ``path``/``kind``/``file_ids`` are the surface's resolved
    replied media (the /steal seam).  Returns the pack item —
    the existing twin when the pack already holds this media.
    Raises PackError when there is honestly nothing to store.
    """
    file_ids = {str(k): str(v) for k, v in
                (file_ids or {}).items() if v}
    src = Path(path) if path else None
    if src is not None and not src.exists():
        src = None
    if src is None:
        if file_ids:
            twin = store.find(person, file_ids=file_ids)
            if twin is not None:
                return twin
            return store.add(person, _reference_item(
                file_ids=file_ids,
                note=("held by platform reference — this "
                      "surface couldn't hand me the bytes, so "
                      "re-sends go by the platform's own handle"),
                name=name, origin=origin))
        raise PackError(
            "there's nothing to steal here — on this platform "
            "path I couldn't reach the sticker's bytes or its "
            "handle.")

    raw = src.read_bytes()
    if not raw:
        raise PackError("that sticker download came back empty "
                        "— nothing to steal.")
    sha = hashlib.sha256(raw).hexdigest()
    fmt = _sniff(raw[:16]) \
        or src.suffix.lower().lstrip(".")
    if fmt == "tgs":
        twin = store.find(person, sha=sha, file_ids=file_ids)
        if twin is not None:
            return twin
        return store.add(person, _reference_item(
            file_ids=file_ids,
            note=("a .tgs (Lottie) sticker — held by platform "
                  "reference; I don't render Lottie, so the "
                  "handle is the honest copy"),
            name=name, origin=origin))
    twin = store.find(person, sha=sha, file_ids=file_ids)
    if twin is not None:
        return twin
    if len(store.pack(person)) >= MAX_PACK_ITEMS:
        raise PackError(
            f"your sticker pack is full ({MAX_PACK_ITEMS}) "
            f"— it's a working collection, not an archive")

    person_dir = Path(media_dir) / "packs" / hashlib.sha256(
        person.encode()).hexdigest()[:16]
    person_dir.mkdir(parents=True, exist_ok=True)
    item_id = "sp_" + new_short_id()
    if fmt in ("webp", "webm"):
        dest = person_dir / f"{item_id}.{fmt}"
        shutil.copyfile(src, dest)
        note = ""
    else:
        # Another raster shape (png/jpg photo-sticker): the
        # conversion organ normalizes it to a real WebP.
        from . import media_convert

        try:
            made = media_convert.image_to_sticker(
                src, out_dir=person_dir)
        except media_convert.ConversionError as exc:
            raise PackError(
                f"that media isn't a sticker I can keep: "
                f"{exc}") from exc
        dest = Path(made)
        final = person_dir / f"{item_id}.webp"
        if dest != final:
            shutil.move(str(dest), str(final))
            dest = final
        fmt = "webp"
        note = "normalized to WebP from the original image"
    item = {
        "id": item_id, "format": fmt, "path": str(dest),
        "file_ids": file_ids, "by_reference": False,
        "note": note, "name": str(name or "stolen sticker"),
        "origin": str(origin or ""), "sha": sha, "uses": 0,
        "created": time.time(),
    }
    return store.add(person, item)
