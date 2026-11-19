"""FigCraft document model — real ``.figcraft`` files (lane 105).

The file shape is upstream's own (storytold/figcraft, read from
``examples/ridgeline-app.figcraft`` at survey time): a ZIP holding
``document.json`` (``{"format": "figcraft", "version": 1, ...}``)
plus a ``thumbnail.png``. Nodes are keyed by string id in the JSON
and carry the upstream fields — ``type``, ``parent``, ``children``,
``transform`` [a b c d tx ty], ``width``/``height``, ``fills``,
``strokes``, ``corners``, ``clip``, ``layout`` / ``child_layout``,
text ``style`` + ``characters``, component ``source``/``props``/
``overrides``, ``bound`` variable links, ``effects``, ``guides``,
``interactions``.

Devon's port is its own Python following that design; no upstream
code is copied. Bounds: 64 MB package / 32 MB part / JSON depth
100 / 500k nodes — the same discipline as the word organ.

Node ids follow the upstream collab allocator: an id's high 32
bits are its client id space (``client_id << 32 | counter``);
ids in existing files occupy space 0. :meth:`DesignDocument.alloc_id`
hands out fresh ids in a caller's space, continuing above every
id of that space already in the document.
"""

from __future__ import annotations

import io
import json
import zipfile
from copy import deepcopy
from typing import Any

from .errors import DesignError

__all__ = ["DesignDocument", "MAX_PACKAGE_BYTES", "MAX_PART_BYTES"]

MAX_PACKAGE_BYTES = 64 * 1024 * 1024
MAX_PART_BYTES = 32 * 1024 * 1024
MAX_NODES = 500_000
MAX_JSON_DEPTH = 100

_DOCUMENT_JSON = "document.json"
_THUMBNAIL = "thumbnail.png"

_NODE_TYPES = frozenset({
    "page", "section", "frame", "group", "rectangle", "ellipse",
    "star", "text", "vector", "component", "componentSet",
    "instance", "image", "line",
})


def _canonical(value: Any) -> bytes:
    """Canonical JSON bytes for a document dict (sorted keys)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _check_depth(value: Any, depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise DesignError("document JSON is nested too deeply",
                          kind="bad_format")
    if isinstance(value, dict):
        for child in value.values():
            _check_depth(child, depth + 1)
    elif isinstance(value, list):
        for child in value:
            _check_depth(child, depth + 1)


def _color(r: float, g: float, b: float, a: float = 1.0) -> dict[str, float]:
    return {"r": r, "g": g, "b": b, "a": a}


class DesignDocument:
    """A ``.figcraft`` design document: pages, nodes, styles, variables."""

    def __init__(self, data: dict[str, Any]):
        self._data = data
        self._counters: dict[int, int] = {}
        # Verbatim input bytes: an unedited document writes back
        # byte-identical. Any mutation marks the document dirty and
        # the next write re-serialises deterministically instead.
        self._raw: bytes | None = None
        self._dirty = True

    def mark_dirty(self) -> None:
        """Record that the document changed since it was read."""
        self._dirty = True
        self._raw = None

    # -- construction -------------------------------------------------
    @classmethod
    def from_bytes(cls, raw: bytes) -> DesignDocument:
        if len(raw) > MAX_PACKAGE_BYTES:
            raise DesignError(
                f"package is {len(raw)} bytes, over the "
                f"{MAX_PACKAGE_BYTES} byte cap", kind="bad_file")
        try:
            archive = zipfile.ZipFile(io.BytesIO(raw))
        except zipfile.BadZipFile as exc:
            raise DesignError(f"not a .figcraft zip archive: {exc}",
                              kind="bad_file") from exc
        try:
            names = archive.namelist()
        except zipfile.BadZipFile as exc:
            raise DesignError(f"unreadable .figcraft archive: {exc}",
                              kind="bad_file") from exc
        if _DOCUMENT_JSON not in names:
            raise DesignError("archive has no document.json",
                              kind="bad_format")
        try:
            info = archive.getinfo(_DOCUMENT_JSON)
        except KeyError as exc:  # pragma: no cover - namelist said so
            raise DesignError("archive has no document.json",
                              kind="bad_format") from exc
        if info.file_size > MAX_PART_BYTES:
            raise DesignError("document.json is over the 32 MB part cap",
                              kind="bad_format")
        try:
            text = archive.read(_DOCUMENT_JSON)
        except (zipfile.BadZipFile, OSError) as exc:
            raise DesignError(f"cannot read document.json: {exc}",
                              kind="bad_file") from exc
        try:
            data = json.loads(text.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise DesignError(f"document.json is not valid JSON: {exc}",
                              kind="bad_format") from exc
        doc = cls._from_dict(data)
        doc._thumbnail = None
        if _THUMBNAIL in names:
            try:
                doc._thumbnail = archive.read(_THUMBNAIL)
            except (zipfile.BadZipFile, OSError):
                doc._thumbnail = None
        doc._raw_images = {}
        for name in names:
            if name.startswith("images/") and name != "images/":
                try:
                    doc._raw_images[name] = archive.read(name)
                except (zipfile.BadZipFile, OSError):
                    continue
        doc._raw = raw
        doc._dirty = False
        return doc

    @classmethod
    def _from_dict(cls, data: Any) -> DesignDocument:
        if not isinstance(data, dict):
            raise DesignError("document.json is not a JSON object",
                              kind="bad_format")
        _check_depth(data)
        if data.get("format") != "figcraft":
            raise DesignError(
                f"not a figcraft document (format={data.get('format')!r})",
                kind="bad_format")
        if data.get("version") != 1:
            raise DesignError(
                f"unsupported figcraft version {data.get('version')!r}",
                kind="unsupported")
        nodes = data.get("nodes")
        if not isinstance(nodes, dict):
            raise DesignError("document has no nodes object",
                              kind="bad_format")
        if len(nodes) > MAX_NODES:
            raise DesignError(f"document has {len(nodes)} nodes, over the "
                              f"{MAX_NODES} cap", kind="bad_format")
        pages = data.get("pages")
        if not isinstance(pages, list) or not pages:
            raise DesignError("document has no pages", kind="bad_format")
        doc = cls(data)
        doc._thumbnail = None
        doc._raw_images = {}
        return doc

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DesignDocument:
        return cls._from_dict(deepcopy(data))

    @classmethod
    def blank(cls, name: str = "Untitled") -> DesignDocument:
        page = {"id": 1, "name": "Page 1", "type": "page", "parent": None,
                "background": _color(0.11764706, 0.11764706, 0.11764706),
                "flows": [],
                "prototype": {"device": "None",
                              "background": _color(0.11764706, 0.11764706,
                                                  0.11764706),
                              "rotation": "portrait"},
                "guides": [], "children": [], "width": 0.0, "height": 0.0}
        data = {"format": "figcraft", "version": 1, "name": name,
                "pages": [1], "current_page": 1,
                "nodes": {"1": page}, "next_id": 2,
                "styles": [], "variables": {"collections": [], "variables": []},
                "comments": [], "document_colors": [], "imageIndex": []}
        doc = cls(data)
        doc._thumbnail = None
        doc._raw_images = {}
        return doc

    # -- accessors ----------------------------------------------------
    @property
    def format(self) -> str:
        return str(self._data.get("format"))

    @property
    def version(self) -> int:
        return int(self._data.get("version", 0))

    @property
    def name(self) -> str:
        return str(self._data.get("name", ""))

    @name.setter
    def name(self, value: str) -> None:
        self._data["name"] = value
        self.mark_dirty()

    @property
    def pages(self) -> list[int]:
        return [int(p) for p in self._data.get("pages", [])]

    @property
    def current_page(self) -> int:
        return int(self._data.get("current_page", self.pages[0]))

    @property
    def nodes(self) -> dict[int, dict[str, Any]]:
        return {int(k): v for k, v in self._data.get("nodes", {}).items()}

    @property
    def next_id(self) -> int:
        return int(self._data.get("next_id", 1))

    @property
    def variables(self) -> dict[str, Any]:
        return self._data.setdefault("variables",
                                     {"collections": [], "variables": []})

    @property
    def styles(self) -> list[Any]:
        return self._data.setdefault("styles", [])

    @property
    def comments(self) -> list[Any]:
        return self._data.setdefault("comments", [])

    @property
    def thumbnail(self) -> bytes | None:
        return self._thumbnail

    @property
    def images(self) -> dict[str, bytes]:
        return dict(self._raw_images)

    def node(self, nid: int) -> dict[str, Any]:
        nodes = self._data.get("nodes", {})
        try:
            return nodes[str(int(nid))]
        except KeyError as exc:
            raise DesignError(f"no node {nid}", kind="not_found") from exc

    def has_node(self, nid: int) -> bool:
        return str(int(nid)) in self._data.get("nodes", {})

    def children(self, nid: int) -> list[dict[str, Any]]:
        parent = self.node(nid)
        out = []
        for child_id in parent.get("children", []):
            try:
                out.append(self.node(int(child_id)))
            except DesignError:
                continue
        return out

    def parent_id(self, nid: int) -> int | None:
        parent = self.node(nid).get("parent")
        return None if parent is None else int(parent)

    def alloc_id(self, client_id: int = 0) -> int:
        """Hand out a fresh id in ``client_id``'s space (high 32 bits).

        Continues above every id of that space already in the
        document, so two editors never mint the same id.
        """
        space = int(client_id) << 32
        highest = self._counters.get(int(client_id), -1)
        for raw in self._data.get("nodes", {}):
            nid = int(raw)
            if (nid >> 32) == int(client_id) and (nid & 0xFFFFFFFF) > highest:
                highest = nid & 0xFFFFFFFF
        counter = highest + 1
        self._counters[int(client_id)] = counter
        fresh = space | counter
        if fresh >= int(self._data.get("next_id", 0)):
            self._data["next_id"] = fresh + 1
            self.mark_dirty()
        return fresh

    # -- serialisation ------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self._data)

    def to_bytes(self) -> bytes:
        if not self._dirty and self._raw is not None:
            return self._raw
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for filename, payload in [
                    (_DOCUMENT_JSON, _canonical(self._data))] + (
                    [(_THUMBNAIL, self._thumbnail)]
                    if self._thumbnail is not None else []) + [
                    (name, self._raw_images[name])
                    for name in sorted(self._raw_images)]:
                info = zipfile.ZipInfo(filename, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(info, payload)
        return buf.getvalue()

    def put_image(self, key: str, data: bytes) -> None:
        """Store an image part (``images/<key>``) and list it."""
        name = f"images/{key}"
        self._raw_images[name] = data
        index = self._data.setdefault("imageIndex", [])
        if key not in index:
            index.append(key)
        self.mark_dirty()

    # -- fidelity -----------------------------------------------------
    def fidelity_report(self, reference: bytes,
                          touched: set[int] | frozenset[int] | None = None
                          ) -> dict[str, Any]:
        """Compare this document against ``reference`` bytes.

        Returns node counts, the changed/added/removed ids, how many
        nodes changed that the edit did NOT touch (``touched`` names
        the ids the edit intentionally changed; 0 for a clean edit),
        and whether the thumbnail and image parts survived
        byte-identical.
        """
        try:
            ref = DesignDocument.from_bytes(reference)
        except DesignError as exc:
            raise DesignError(f"cannot read reference document: {exc}",
                              kind="bad_format") from exc
        ref_nodes = {int(k): v for k, v in ref._data.get("nodes", {}).items()}
        mine = self.nodes
        changed: list[int] = []
        for nid, node in mine.items():
            ref_node = ref_nodes.get(nid)
            if ref_node is None:
                continue
            # A parent's children list grows when a node is added
            # under it — that is reported in ``added``/``removed``,
            # not as a content change of the parent.
            mine_core = {k: v for k, v in node.items() if k != "children"}
            ref_core = {k: v for k, v in ref_node.items() if k != "children"}
            if _canonical(mine_core) != _canonical(ref_core):
                changed.append(nid)
        added = sorted(set(mine) - set(ref_nodes))
        removed = sorted(set(ref_nodes) - set(mine))
        touched = set(touched or ())
        untouched = [nid for nid in changed if nid not in touched]
        return {
            "nodes": len(mine),
            "changed_ids": sorted(changed),
            "added": added,
            "removed": removed,
            "untouched_changed": len(untouched),
            "parts_preserved": (
                (self._thumbnail is None) == (ref._thumbnail is None)
                and (self._thumbnail is None
                     or self._thumbnail == ref._thumbnail)
                and self._raw_images == ref._raw_images),
        }
