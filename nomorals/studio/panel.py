"""The declarative panel a Studio plug-in contributes to a page.

Upstream (``craft-extensions``) ships plug-ins as ``plugin.wasm``
with no files, network or clock in version 1. Devon's Studio pages
are plain HTML under a ``style-src 'none'`` CSP, so a wasm module
cannot run in them — the Studio plug-in is declarative instead: a
``panel.json`` naming the page and slot it fills, a title, and
blocks of text, labelled values and links. The host renders the
blocks with every byte escaped; the panel carries no code and asks
for no permissions (the manifest enforces the empty list).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .text import Lines, validate_text

__all__ = ["KNOWN_PAGES", "KNOWN_SLOTS", "MAX_PANEL_BYTES", "PanelError",
           "PanelFile"]

MAX_PANEL_BYTES = 64 * 1024
SCHEMA = 1

#: The Studio pages a panel may attach to, and the slots on each.
#: The photo page is the first page with slots; the board and the
#: extensions board gain theirs when a plug-in asks for them.
KNOWN_PAGES: dict[str, tuple[str, ...]] = {
    "photo": ("above", "below"),
}

KNOWN_SLOTS = ("above", "below")

_MAX_TEXT = 2000
_MAX_BLOCKS = 32
_MAX_LABEL = 64


class PanelError(ValueError):
    """A panel file was refused, with the reason as the message."""


@dataclass(frozen=True)
class PanelFile:
    """A parsed ``panel.json``."""

    schema: int
    page: str
    slot: str
    title: str
    blocks: tuple

    @classmethod
    def parse(cls, data: bytes) -> PanelFile:
        """Parse and check a panel file, or raise PanelError."""
        if len(data) > MAX_PANEL_BYTES:
            raise PanelError("the panel file is too large")
        try:
            raw = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise PanelError("the panel file cannot be read") from None
        if not isinstance(raw, dict):
            raise PanelError("the panel file cannot be read")
        schema = raw.get("schema")
        if schema != SCHEMA:
            raise PanelError(
                "the panel uses a format version "
                f"({schema}) this app does not know")
        page = raw.get("page", "")
        slot = raw.get("slot", "")
        slots = KNOWN_PAGES.get(str(page))
        if slots is None:
            raise PanelError(
                "the panel is for a Studio page this app does not "
                f"have: {page}")
        if slot not in slots:
            raise PanelError(
                f"the panel asks for a slot ({slot}) the {page} "
                "page does not have")
        title = raw.get("title", "")
        err = validate_text(str(title), 64, Lines.ONE)
        if err is not None:
            raise PanelError(f"the panel title is not valid: {err.value}")
        blocks_raw = raw.get("blocks") or []
        if not isinstance(blocks_raw, list) or len(blocks_raw) > _MAX_BLOCKS:
            raise PanelError("the panel file cannot be read")
        blocks = tuple(_parse_block(item) for item in blocks_raw)
        return cls(schema=SCHEMA, page=str(page), slot=str(slot),
                   title=str(title), blocks=blocks)


def _parse_block(raw: object) -> dict:
    if not isinstance(raw, dict):
        raise PanelError("the panel file cannot be read")
    kind = raw.get("kind", "")
    if kind == "text":
        text = raw.get("text", "")
        err = validate_text(str(text), _MAX_TEXT, Lines.MANY)
        if err is not None:
            raise PanelError(f"the panel text is not valid: {err.value}")
        return {"kind": "text", "text": str(text)}
    if kind == "kv":
        label = raw.get("label", "")
        value = raw.get("value", "")
        for part, name in ((label, "label"), (value, "value")):
            err = validate_text(str(part), _MAX_LABEL, Lines.ONE)
            if err is not None:
                raise PanelError(
                    f"the panel {name} is not valid: {err.value}")
        return {"kind": "kv", "label": str(label), "value": str(value)}
    if kind == "link":
        label = raw.get("label", "")
        href = raw.get("href", "")
        err = validate_text(str(label), _MAX_LABEL, Lines.ONE)
        if err is not None:
            raise PanelError(
                f"the panel link label is not valid: {err.value}")
        if not isinstance(href, str) or not href.startswith("https://") \
                or len(href) > 200 or " " in href:
            raise PanelError(
                "the panel link must be an https address")
        return {"kind": "link", "label": str(label), "href": href}
    raise PanelError(f"the panel has a block kind this app does not "
                     f"know: {kind}")
