"""FigCraft editor — fidelity-preserving edits (lane 105).

Every edit goes through the document model and records the ids
it touched, so :meth:`DesignEditor.fidelity_report` can prove
that only the intended nodes changed. Adding a node mints a
fresh id in space 0 (single-user editing); session clients pass
their own client-space id explicitly so two editors never mint
the same id (see ``session.py``).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .components import detach_instance
from .errors import DesignError
from .model import DesignDocument
from .proto import add_interaction

__all__ = ["DesignEditor"]


def _hex_to_color(hex_color: str) -> dict[str, float]:
    text = hex_color.lstrip("#")
    if len(text) != 6:
        raise DesignError(f"bad hex colour {hex_color!r}",
                          kind="invalid_argument")
    try:
        r, g, b = (int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError as exc:
        raise DesignError(f"bad hex colour {hex_color!r}",
                          kind="invalid_argument") from exc
    return {"r": r, "g": g, "b": b, "a": 1.0}


def _solid(hex_color: str) -> dict[str, Any]:
    return {"type": "solid", "color": _hex_to_color(hex_color),
            "opacity": 1.0, "visible": True, "blend": "normal"}


class DesignEditor:
    """Mutating edits over a :class:`DesignDocument`."""

    def __init__(self, doc: DesignDocument):
        self.doc = doc
        self.touched: set[int] = set()
        self.added: list[int] = []

    # -- helpers ------------------------------------------------------
    def _mint(self, node_id: int | None = None) -> int:
        if node_id is not None:
            if self.doc.has_node(node_id):
                raise DesignError(f"node {node_id} already exists",
                                  kind="invalid_argument")
            return int(node_id)
        return self.doc.alloc_id()

    def _attach(self, parent_id: int, node: dict[str, Any]) -> int:
        self.doc.mark_dirty()
        parent = self.doc.node(parent_id)
        nodes = self.doc._data.setdefault("nodes", {})
        nodes[str(node["id"])] = node
        children = list(parent.get("children") or [])
        children.append(int(node["id"]))
        parent["children"] = children
        self.touched.add(parent_id)
        self.added.append(int(node["id"]))
        return int(node["id"])

    @staticmethod
    def _base(nid: int, name: str, ntype: str, parent: int,
              x: float, y: float, width: float, height: float) -> dict[str, Any]:
        return {"id": nid, "name": name, "type": ntype, "parent": parent,
                "transform": [1.0, 0.0, 0.0, 1.0, float(x), float(y)],
                "width": float(width), "height": float(height)}

    # -- creation -----------------------------------------------------
    def add_page(self, name: str = "Page") -> int:
        self.doc.mark_dirty()
        nid = self.doc.alloc_id()
        page = {"id": nid, "name": name, "type": "page", "parent": None,
                "background": {"r": 0.11764706, "g": 0.11764706,
                               "b": 0.11764706, "a": 1.0},
                "flows": [],
                "prototype": {"device": "None",
                              "background": {"r": 0.11764706, "g": 0.11764706,
                                             "b": 0.11764706, "a": 1.0},
                              "rotation": "portrait"},
                "guides": [], "children": [], "width": 0.0, "height": 0.0}
        self.doc._data.setdefault("nodes", {})[str(nid)] = page
        pages = self.doc._data.setdefault("pages", [])
        pages.append(nid)
        self.added.append(nid)
        return nid

    def add_frame(self, parent_id: int, name: str, x: float, y: float,
                  width: float, height: float, fill: str | None = None,
                  node_id: int | None = None) -> int:
        nid = self._mint(node_id)
        node = self._base(nid, name, "frame", parent_id, x, y, width, height)
        node["children"] = []
        node["clip"] = True
        if fill is not None:
            node["fills"] = [_solid(fill)]
        return self._attach(parent_id, node)

    def add_rect(self, parent_id: int, name: str, x: float, y: float,
                 width: float, height: float, fill: str,
                 node_id: int | None = None) -> int:
        nid = self._mint(node_id)
        node = self._base(nid, name, "rectangle", parent_id, x, y, width, height)
        node["fills"] = [_solid(fill)]
        return self._attach(parent_id, node)

    def add_text(self, parent_id: int, name: str, text: str, x: float, y: float,
                 width: float, font_size: float = 16.0,
                 fill: str = "#111111", node_id: int | None = None) -> int:
        nid = self._mint(node_id)
        node = self._base(nid, name, "text", parent_id, x, y, width,
                          font_size * 1.4)
        node["characters"] = text
        node["style"] = {"font_family": "Inter", "font_style": "Regular",
                         "font_size": float(font_size),
                         "line_height": {"unit": "percent", "value": 140.0},
                         "letter_spacing": {"unit": "percent", "value": 0.0},
                         "decoration": "none", "case": "original",
                         "fills": None, "link": None}
        node["fills"] = [_solid(fill)]
        return self._attach(parent_id, node)

    def add_component(self, parent_id: int, name: str, width: float,
                      height: float, node_id: int | None = None) -> int:
        nid = self._mint(node_id)
        node = self._base(nid, name, "component", parent_id, 0.0, 0.0,
                          width, height)
        node["children"] = []
        node["props"] = {}
        node["description"] = ""
        return self._attach(parent_id, node)

    def add_component_set(self, parent_id: int, name: str,
                          node_id: int | None = None) -> int:
        nid = self._mint(node_id)
        node = self._base(nid, name, "componentSet", parent_id, 0.0, 0.0,
                          0.0, 0.0)
        node["children"] = []
        node["description"] = ""
        return self._attach(parent_id, node)

    def add_instance(self, parent_id: int, source_id: int,
                     props: dict[str, Any] | None = None,
                     node_id: int | None = None) -> int:
        source = self.doc.node(source_id)
        if source.get("type") not in ("component", "componentSet"):
            raise DesignError(f"node {source_id} is not a component source",
                              kind="invalid_argument")
        nid = self._mint(node_id)
        node = self._base(nid, source.get("name", "Instance") + " instance",
                          "instance", parent_id, 0.0, 0.0,
                          float(source.get("width") or 0.0),
                          float(source.get("height") or 0.0))
        node["component"] = source_id
        node["source"] = source_id
        node["props"] = dict(props or {})
        node["overrides"] = []
        return self._attach(parent_id, node)

    # -- mutation -----------------------------------------------------
    def set_text(self, node_id: int, text: str) -> None:
        self.doc.mark_dirty()
        node = self.doc.node(node_id)
        if node.get("type") != "text":
            raise DesignError(f"node {node_id} is not a text node",
                              kind="invalid_argument")
        node["characters"] = text
        self.touched.add(node_id)

    def set_fill(self, node_id: int, hex_color: str) -> None:
        self.doc.mark_dirty()
        node = self.doc.node(node_id)
        node["fills"] = [_solid(hex_color)]
        self.touched.add(node_id)

    def set_size(self, node_id: int, width: float, height: float) -> None:
        self.doc.mark_dirty()
        node = self.doc.node(node_id)
        node["width"] = float(width)
        node["height"] = float(height)
        self.touched.add(node_id)

    def move(self, node_id: int, x: float, y: float) -> None:
        self.doc.mark_dirty()
        node = self.doc.node(node_id)
        transform = list(node.get("transform") or [1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        transform[4] = float(x)
        transform[5] = float(y)
        node["transform"] = transform
        self.touched.add(node_id)

    def restyle(self, node_id: int, **kwargs: Any) -> None:
        self.doc.mark_dirty()
        node = self.doc.node(node_id)
        if "fill" in kwargs:
            node["fills"] = [_solid(str(kwargs["fill"]))]
        if "opacity" in kwargs:
            node["opacity"] = float(kwargs["opacity"])
        if "name" in kwargs:
            node["name"] = str(kwargs["name"])
        if "corner_radius" in kwargs:
            radius = float(kwargs["corner_radius"])
            node["corners"] = {"radii": [radius] * 4, "smoothing": 0.0}
        self.touched.add(node_id)

    def set_variable(self, variable_id: str, mode_id: str,
                     value: dict[str, Any]) -> None:
        self.doc.mark_dirty()
        variables = self.doc.variables.get("variables", [])
        for variable in variables:
            if str(variable.get("id")) == variable_id:
                variable.setdefault("values", {})[mode_id] = deepcopy(value)
                self.touched.add(-1)
                return
        raise DesignError(f"no variable {variable_id}", kind="not_found")

    def wire_interaction(self, node_id: int, trigger: str, action: str,
                         destination: int | None,
                         transition: str = "instant") -> dict[str, Any]:
        self.doc.mark_dirty()
        entry = add_interaction(self.doc, node_id, trigger, action,
                                destination, transition)
        self.touched.add(node_id)
        return entry

    def remove(self, node_id: int) -> None:
        self.doc.mark_dirty()
        node = self.doc.node(node_id)
        parent_id = node.get("parent")
        nodes = self.doc._data.get("nodes", {})

        def collect(nid: int) -> list[int]:
            out = [nid]
            for child in list(nodes.get(str(nid), {}).get("children") or []):
                out.extend(collect(int(child)))
            return out

        for victim in collect(node_id):
            nodes.pop(str(victim), None)
        if parent_id is not None:
            parent = self.doc.node(int(parent_id))
            parent["children"] = [c for c in parent.get("children", [])
                                  if int(c) != node_id]
            self.touched.add(int(parent_id))

    def detach(self, instance_id: int) -> int:
        self.doc.mark_dirty()
        root = detach_instance(self.doc, instance_id)
        self.touched.add(root)
        return root

    # -- output -------------------------------------------------------
    def to_bytes(self) -> bytes:
        return self.doc.to_bytes()

    def fidelity_report(self, reference: bytes) -> dict[str, Any]:
        return self.doc.fidelity_report(reference, touched=self.touched)
