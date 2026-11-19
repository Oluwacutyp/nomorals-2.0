"""Material editor — validated edits with snapshot undo.

MaterialCraft port of upstream's ``crates/engine`` command model:
every edit goes through a typed method that validates first, and
edits run inside a snapshot — the document is cloned before the
edit and restored on any failure, so a failed edit never leaves a
half-changed graph (upstream's ``Session::edit``). ``edit_merged``
folds repeated edits with the same key (a slider drag) into one
undo step.
"""

from __future__ import annotations

from contextlib import contextmanager

from .errors import MaterialError, MaterialErrorKind
from .model import CHANNELS, Connection, Graph, MaterialDocument, Node
from .nodes import by_id


class MaterialEditor:
    """Edit a material document, atomically, with undo."""

    def __init__(self, doc: MaterialDocument) -> None:
        self.doc = doc
        self._undo: list[dict] = []
        self._redo: list[dict] = []
        self._touched: set[str] = set()

    @property
    def graph(self) -> Graph:
        return self.doc.graph

    def _snapshot(self) -> dict:
        return self.doc.to_dict()

    def _restore(self, snapshot: dict) -> None:
        graphs = [Graph.from_dict(g) for g in snapshot["graphs"]]
        self.doc.graphs = graphs
        self.doc.name = snapshot.get("name", self.doc.name)
        self.doc.mark_dirty()

    @contextmanager
    def edit(self, merge_key: str | None = None):
        """Run edits atomically; restore the snapshot on any failure."""
        before = self._snapshot()
        try:
            yield self
        except Exception:
            self._restore(before)
            raise
        self.doc.mark_dirty()
        if merge_key is None or not self._undo or self._undo[-1][0] != merge_key:
            self._undo.append((merge_key, before))
        self._redo.clear()

    def edit_merged(self, merge_key: str):
        """An edit context that folds into the previous step with the
        same key (a slider drag is one undo step)."""
        return self.edit(merge_key=merge_key)

    def undo(self) -> bool:
        if not self._undo:
            return False
        _key, snapshot = self._undo.pop()
        self._redo.append((None, self._snapshot()))
        self._restore(snapshot)
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        _key, snapshot = self._redo.pop()
        self._undo.append((None, self._snapshot()))
        self._restore(snapshot)
        return True

    # -- node ops ------------------------------------------------------

    def add_node(self, def_: str, node_id: str | None = None,
                 x: float = 0.0, y: float = 0.0,
                 params: dict | None = None) -> Node:
        definition = by_id(def_)
        if definition is None:
            raise MaterialError(MaterialErrorKind.not_found,
                                f"unknown node def {def_!r}")
        node_id = node_id or self._next_id()
        if node_id in self.graph.nodes:
            raise MaterialError(MaterialErrorKind.invalid_argument,
                                f"node id {node_id!r} already exists")
        node = Node(id=node_id, def_=def_, pos=[float(x), float(y)],
                    params=dict(params or {}))
        self.graph.nodes[node_id] = node
        self._touched.add(node_id)
        self.doc.mark_dirty()
        return node

    def _next_id(self) -> str:
        i = 1
        while f"n{i}" in self.graph.nodes:
            i += 1
        return f"n{i}"

    def remove_node(self, node_id: str) -> None:
        if node_id not in self.graph.nodes:
            raise MaterialError(MaterialErrorKind.not_found,
                                f"no node {node_id!r}")
        del self.graph.nodes[node_id]
        self.graph.connections = [
            c for c in self.graph.connections
            if c.from_node != node_id and c.to_node != node_id]
        for channel, target in list(self.graph.outputs.items()):
            if target.get("node") == node_id:
                del self.graph.outputs[channel]
        self._touched.add(node_id)
        self.doc.mark_dirty()

    def move_node(self, node_id: str, x: float, y: float) -> None:
        node = self._node(node_id)
        node.pos = [float(x), float(y)]
        self._touched.add(node_id)
        self.doc.mark_dirty()

    def set_param(self, node_id: str, param: str, value: object) -> None:
        node = self._node(node_id)
        definition = by_id(node.def_)
        if definition is None:
            raise MaterialError(MaterialErrorKind.not_found,
                                f"unknown node def {node.def_!r}")
        param_def = definition.param(param)
        if param_def is None:
            raise MaterialError(MaterialErrorKind.invalid_argument,
                                f"node {node.def_!r} has no parameter {param!r}")
        node.params[param] = value
        self._touched.add(node_id)
        self.doc.mark_dirty()

    def set_label(self, node_id: str, label: str) -> None:
        node = self._node(node_id)
        node.label = str(label)
        self._touched.add(node_id)
        self.doc.mark_dirty()

    def connect(self, from_node: str, to_node: str,
                from_port: str = "output", to_port: str = "input") -> None:
        src = self._node(from_node)
        dst = self._node(to_node)
        src_def = by_id(src.def_)
        dst_def = by_id(dst.def_)
        if src_def is None or dst_def is None:
            raise MaterialError(MaterialErrorKind.not_found,
                                "unknown node def on a wire end")
        if not any(p.id == from_port for p in src_def.outputs):
            raise MaterialError(MaterialErrorKind.invalid_argument,
                                f"{src.def_!r} has no output port {from_port!r}")
        dst_port_def = next((p for p in dst_def.inputs if p.id == to_port), None)
        if dst_port_def is None:
            raise MaterialError(MaterialErrorKind.invalid_argument,
                                f"{dst.def_!r} has no input port {to_port!r}")
        src_port_def = next(p for p in src_def.outputs if p.id == from_port)
        if (dst_port_def.kind != "any" and src_port_def.kind != "any"
                and dst_port_def.kind != src_port_def.kind):
            raise MaterialError(
                MaterialErrorKind.invalid_argument,
                f"cannot wire {src_port_def.kind} output to "
                f"{dst_port_def.kind} input")
        for existing in self.graph.connections:
            if (existing.from_node == from_node
                    and existing.from_port == from_port
                    and existing.to_node == to_node
                    and existing.to_port == to_port):
                return  # already wired
        self.graph.connections.append(
            Connection(from_node, from_port, to_node, to_port))
        self._touched.add(to_node)
        self.doc.mark_dirty()

    def disconnect(self, from_node: str, to_node: str,
                   from_port: str = "output", to_port: str = "input") -> None:
        before = len(self.graph.connections)
        self.graph.connections = [
            c for c in self.graph.connections
            if not (c.from_node == from_node and c.from_port == from_port
                    and c.to_node == to_node and c.to_port == to_port)]
        if len(self.graph.connections) == before:
            raise MaterialError(MaterialErrorKind.not_found,
                                "no such connection")
        self._touched.add(to_node)
        self.doc.mark_dirty()

    def set_output(self, channel: str, node_id: str,
                   port: str = "output") -> None:
        if channel not in CHANNELS:
            raise MaterialError(MaterialErrorKind.invalid_argument,
                                f"unknown channel {channel!r}")
        self._node(node_id)
        self.graph.outputs[channel] = {"node": node_id, "port": port}
        self._touched.add(node_id)
        self.doc.mark_dirty()

    def clear_output(self, channel: str) -> None:
        self.graph.outputs.pop(channel, None)
        self.doc.mark_dirty()

    def set_size(self, size: int) -> None:
        if not 1 <= int(size) <= 8192:
            raise MaterialError(MaterialErrorKind.invalid_argument,
                                "graph size must be 1..8192")
        self.graph.size = int(size)
        self.doc.mark_dirty()

    def set_seed(self, seed: int) -> None:
        self.graph.seed = int(seed)
        self.doc.mark_dirty()

    def _node(self, node_id: str) -> Node:
        node = self.graph.nodes.get(node_id)
        if node is None:
            raise MaterialError(MaterialErrorKind.not_found,
                                f"no node {node_id!r}")
        return node

    def touched(self) -> list[str]:
        return sorted(self._touched)

    def fidelity_report(self) -> dict:
        report = self.doc.fidelity_report()
        report["touched"] = self.touched()
        return report
