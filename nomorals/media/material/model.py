"""Material document model — Package → Graph → Node / Connection.

MaterialCraft port of upstream's ``crates/core`` (``graph.rs``,
``value.rs``, ``package.rs``) and ``crates/formats`` (``.mcraft``
packages, ``.mcmat`` portable materials): a material is a node
graph whose outputs feed named PBR channels; parameters live only
where they differ from the catalog defaults, so files stay small
and new defaults reach old files.

Fidelity discipline follows the cad/vector organs: an unedited
document writes back byte-identical (the raw input bytes are kept
until the first mutation); any mutation marks the document dirty
and the next write re-serialises deterministically with a fixed
ZIP epoch. ``fidelity_report`` names exactly which nodes changed.
"""

from __future__ import annotations

import copy
import io
import json
import zipfile
from dataclasses import dataclass, field

from .errors import MaterialError, MaterialErrorKind

FORMAT_VERSION = 1
MAX_PACKAGE_BYTES = 64 * 1024 * 1024
MAX_PART_BYTES = 32 * 1024 * 1024
MAX_JSON_DEPTH = 100
MAX_NODES = 500_000

CHANNELS = ("albedo", "roughness", "metallic", "normal", "height",
            "ao", "emissive")

_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


def _json_depth(value: object, depth: int = 0) -> int:
    if depth > MAX_JSON_DEPTH:
        raise MaterialError(MaterialErrorKind.bad_format,
                            "JSON nesting too deep")
    if isinstance(value, dict):
        return max([depth] + [_json_depth(v, depth + 1)
                              for v in value.values()])
    if isinstance(value, list):
        return max([depth] + [_json_depth(v, depth + 1) for v in value])
    return depth


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


@dataclass
class BaseParams:
    """Parameters every node carries (upstream's base parameters)."""

    size_mode: str = "relative_to_input"
    size: list[int] = field(default_factory=lambda: [0, 0])
    format: str = "inherit"
    seed: int = 0
    tiling: list[bool] = field(default_factory=lambda: [True, True])

    def to_dict(self) -> dict:
        out: dict = {}
        if self.size_mode != "relative_to_input":
            out["sizeMode"] = self.size_mode
        if self.size != [0, 0]:
            out["size"] = list(self.size)
        if self.format != "inherit":
            out["format"] = self.format
        if self.seed != 0:
            out["seed"] = self.seed
        if self.tiling != [True, True]:
            out["tiling"] = list(self.tiling)
        return out

    @classmethod
    def from_dict(cls, data: dict) -> BaseParams:
        if not isinstance(data, dict):
            raise MaterialError(MaterialErrorKind.bad_format,
                                "base params must be an object")
        return cls(
            size_mode=str(data.get("sizeMode", "relative_to_input")),
            size=[int(v) for v in data.get("size", [0, 0])],
            format=str(data.get("format", "inherit")),
            seed=int(data.get("seed", 0)),
            tiling=[bool(v) for v in data.get("tiling", [True, True])],
        )


@dataclass
class Node:
    """A node instance: a catalog definition with parameter overrides."""

    id: str
    def_: str
    label: str = ""
    pos: list[float] = field(default_factory=lambda: [0.0, 0.0])
    params: dict = field(default_factory=dict)
    bindings: dict = field(default_factory=dict)
    base: BaseParams = field(default_factory=BaseParams)
    disabled: bool = False

    def to_dict(self) -> dict:
        # deepcopy: the dict must not alias live state — snapshots
        # (undo, fidelity) are taken from to_dict, and an aliased
        # params dict would silently follow every later edit.
        out: dict = {"id": self.id, "def": self.def_,
                     "pos": [float(self.pos[0]), float(self.pos[1])]}
        if self.label:
            out["label"] = self.label
        if self.params:
            out["params"] = copy.deepcopy(self.params)
        if self.bindings:
            out["bindings"] = copy.deepcopy(self.bindings)
        base = self.base.to_dict()
        if base:
            out["base"] = base
        if self.disabled:
            out["disabled"] = True
        return out

    @classmethod
    def from_dict(cls, data: dict) -> Node:
        if not isinstance(data, dict):
            raise MaterialError(MaterialErrorKind.bad_format,
                                "node must be an object")
        node_id = data.get("id")
        def_ = data.get("def")
        if not isinstance(node_id, str) or not node_id:
            raise MaterialError(MaterialErrorKind.bad_format,
                                "node without an id")
        if not isinstance(def_, str) or not def_:
            raise MaterialError(MaterialErrorKind.bad_format,
                                f"node {node_id} without a def")
        return cls(
            id=node_id,
            def_=def_,
            label=str(data.get("label", "")),
            pos=[float(v) for v in data.get("pos", [0.0, 0.0])],
            params=dict(data.get("params", {})),
            bindings=dict(data.get("bindings", {})),
            base=BaseParams.from_dict(data.get("base", {})),
            disabled=bool(data.get("disabled", False)),
        )


@dataclass
class Connection:
    """A wire from one node's output port to another's input port."""

    from_node: str
    from_port: str
    to_node: str
    to_port: str

    def to_dict(self) -> dict:
        return {"from": [self.from_node, self.from_port],
                "to": [self.to_node, self.to_port]}

    @classmethod
    def from_dict(cls, data: dict) -> Connection:
        if not isinstance(data, dict):
            raise MaterialError(MaterialErrorKind.bad_format,
                                "connection must be an object")
        src = data.get("from", [])
        dst = data.get("to", [])
        if len(src) != 2 or len(dst) != 2:
            raise MaterialError(MaterialErrorKind.bad_format,
                                "connection needs [node, port] ends")
        return cls(str(src[0]), str(src[1]), str(dst[0]), str(dst[1]))


@dataclass
class Graph:
    """One node graph: nodes, wires and the PBR channel outputs."""

    id: str = "main"
    name: str = "Material"
    size: int = 256
    seed: int = 0
    nodes: dict[str, Node] = field(default_factory=dict)
    connections: list[Connection] = field(default_factory=list)
    outputs: dict[str, dict] = field(default_factory=dict)
    exposed: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "size": self.size,
            "seed": self.seed,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "connections": [c.to_dict() for c in self.connections],
            "outputs": copy.deepcopy(self.outputs),
            "exposed": copy.deepcopy(self.exposed),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Graph:
        if not isinstance(data, dict):
            raise MaterialError(MaterialErrorKind.bad_format,
                                "graph must be an object")
        graph = cls(
            id=str(data.get("id", "main")),
            name=str(data.get("name", "Material")),
            size=int(data.get("size", 256)),
            seed=int(data.get("seed", 0)),
            outputs=dict(data.get("outputs", {})),
            exposed=dict(data.get("exposed", {})),
        )
        nodes = data.get("nodes", [])
        if not isinstance(nodes, list) or len(nodes) > MAX_NODES:
            raise MaterialError(MaterialErrorKind.bad_format,
                                "graph nodes must be a list under the cap")
        for raw in nodes:
            node = Node.from_dict(raw)
            if node.id in graph.nodes:
                raise MaterialError(MaterialErrorKind.bad_format,
                                    f"duplicate node id {node.id}")
            graph.nodes[node.id] = node
        for raw in data.get("connections", []):
            graph.connections.append(Connection.from_dict(raw))
        return graph

    def node_ids(self) -> list[str]:
        return list(self.nodes.keys())


class MaterialDocument:
    """A material package: one or more graphs, the first is the main one."""

    def __init__(self, name: str = "Untitled",
                 graphs: list[Graph] | None = None) -> None:
        self.name = name
        self.graphs: list[Graph] = graphs or [Graph()]
        self.meta: dict = {}
        self._raw: bytes | None = None
        self._snapshot: dict | None = None
        self._dirty = False

    @property
    def graph(self) -> Graph:
        return self.graphs[0]

    def to_dict(self) -> dict:
        return {
            "format": "mcraft",
            "version": FORMAT_VERSION,
            "name": self.name,
            "graphs": [g.to_dict() for g in self.graphs],
            "meta": copy.deepcopy(self.meta),
        }

    def to_bytes(self) -> bytes:
        """Serialise to a deterministic ``.mcraft`` zip.

        Unedited documents return their original bytes verbatim.
        """
        if not self._dirty and self._raw is not None:
            return self._raw
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, payload in (
                    ("package.json", canonical_json(self.to_dict())),
                    ("meta.json", canonical_json({"format": "mcraft",
                                                  "version": FORMAT_VERSION}))):
                info = zipfile.ZipInfo(name, date_time=_ZIP_EPOCH)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                zf.writestr(info, payload)
        return buf.getvalue()

    @classmethod
    def parse(cls, data: bytes) -> MaterialDocument:
        """Parse ``.mcraft`` bytes (zip: package.json + meta.json)."""
        if len(data) > MAX_PACKAGE_BYTES:
            raise MaterialError(MaterialErrorKind.too_large,
                                "package exceeds the 64MB cap")
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as exc:
            raise MaterialError(MaterialErrorKind.bad_format,
                                f"not a .mcraft zip: {exc}") from exc
        names = zf.namelist()
        if "package.json" not in names:
            raise MaterialError(MaterialErrorKind.bad_format,
                                ".mcraft package needs package.json")
        raw = zf.read("package.json")
        if len(raw) > MAX_PART_BYTES:
            raise MaterialError(MaterialErrorKind.too_large,
                                "package.json exceeds the 32MB cap")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise MaterialError(MaterialErrorKind.bad_format,
                                f"package.json is not valid JSON: {exc}") from exc
        _json_depth(payload)
        if not isinstance(payload, dict) or payload.get("format") != "mcraft":
            raise MaterialError(MaterialErrorKind.bad_format,
                                "not a MaterialCraft package")
        version = payload.get("version")
        if version != FORMAT_VERSION:
            raise MaterialError(MaterialErrorKind.unsupported,
                                f"unsupported package version {version!r}")
        graphs = payload.get("graphs")
        if not isinstance(graphs, list) or not graphs:
            raise MaterialError(MaterialErrorKind.bad_format,
                                "package needs at least one graph")
        doc = cls(name=str(payload.get("name", "Untitled")),
                  graphs=[Graph.from_dict(g) for g in graphs])
        doc.meta = dict(payload.get("meta", {}))
        doc._raw = data
        doc._snapshot = {g.id: {n.id: n.to_dict()
                                for n in g.nodes.values()}
                         for g in doc.graphs}
        return doc

    @classmethod
    def blank(cls, name: str = "Untitled") -> MaterialDocument:
        doc = cls(name=name)
        doc._dirty = True
        doc._snapshot = {"main": {}}
        return doc

    def mark_dirty(self) -> None:
        self._dirty = True
        self._raw = None

    def fidelity_report(self) -> dict:
        """Which nodes changed vs the document as opened (or created)."""
        current = {g.id: {n.id: n.to_dict() for n in g.nodes.values()}
                   for g in self.graphs}
        snapshot = self._snapshot or {}
        changed: list[str] = []
        added: list[str] = []
        removed: list[str] = []
        for gid, nodes in current.items():
            before = snapshot.get(gid, {})
            for nid, node in nodes.items():
                if nid not in before:
                    added.append(nid)
                elif before[nid] != node:
                    changed.append(nid)
            for nid in before:
                if nid not in nodes:
                    removed.append(nid)
        for gid in snapshot:
            if gid not in current:
                removed.extend(snapshot[gid].keys())
        return {
            "changed": sorted(changed),
            "added": sorted(added),
            "removed": sorted(removed),
            "untouched_changed": 0,
            "dirty": self._dirty,
        }

    # -- .mcmat portable materials (upstream formats/material.rs) --

    def export_mcmat(self, maps: dict[str, bytes],
                     exposed: dict | None = None) -> bytes:
        """Export cooked maps + exposed inputs + source package.

        A ``.mcmat`` zip holds one PNG per PBR channel, a manifest
        with the exposed inputs and presets, and the source package
        so the material can be re-cooked by the reader.
        """
        buf = io.BytesIO()
        manifest = {
            "format": "mcmat",
            "version": 1,
            "name": self.name,
            "channels": sorted(maps.keys()),
            "exposed": exposed or {},
            "source": "source.mcraft",
        }
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, payload in (
                    ("manifest.json", canonical_json(manifest)),
                    ("source.mcraft", self.to_bytes())):
                info = zipfile.ZipInfo(name, date_time=_ZIP_EPOCH)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                zf.writestr(info, payload)
            for channel, png in maps.items():
                info = zipfile.ZipInfo(f"maps/{channel}.png",
                                       date_time=_ZIP_EPOCH)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                zf.writestr(info, png)
        return buf.getvalue()

    @classmethod
    def import_mcmat(cls, data: bytes) -> tuple[MaterialDocument, dict]:
        """Import a ``.mcmat``: rebuilds the source package, returns
        (document, manifest). Map PNGs are not re-cooked on import;
        the source package carries the graph."""
        if len(data) > MAX_PACKAGE_BYTES:
            raise MaterialError(MaterialErrorKind.too_large,
                                ".mcmat exceeds the 64MB cap")
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as exc:
            raise MaterialError(MaterialErrorKind.bad_format,
                                f"not a .mcmat zip: {exc}") from exc
        if "manifest.json" not in zf.namelist():
            raise MaterialError(MaterialErrorKind.bad_format,
                                ".mcmat needs manifest.json")
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        if manifest.get("format") != "mcmat":
            raise MaterialError(MaterialErrorKind.bad_format,
                                "not a portable material")
        if "source.mcraft" not in zf.namelist():
            raise MaterialError(MaterialErrorKind.bad_format,
                                ".mcmat needs source.mcraft")
        return cls.parse(zf.read("source.mcraft")), manifest
