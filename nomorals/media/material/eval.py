"""Graph cooking — topological order + content-addressed cache.

MaterialCraft port of upstream's ``crates/eval``: ``cook`` walks a
graph in dependency order. For each node it hashes the definition
id, the resolved parameters, the size, the seed, tiling, the keys of
its input images and the resource bytes: equal keys hit the cache,
so an edit re-cooks only the nodes downstream of it, and undo finds
earlier results again. A failing node cooks to a magenta image and
records its error; nodes after it still cook. A cycle is refused
with the cycle path named.

Sizes follow upstream: generators take the graph ("parent") size,
filters take their main input's size; both can be offset by powers
of two through the node's base size, and set absolutely. While
editing, outputs are capped at the preview size (512 px by
default), with pixel-based parameters scaled by ``w / 256``.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field

from .errors import MaterialError, MaterialErrorKind
from .image import FloatImage
from .model import Graph, Node
from .nodes import CookCtx, by_id, resolve_params

PREVIEW_CAP = 512
MAX_SIZE = 8192
CACHE_MAX_ENTRIES = 64


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, default=str)


def hash_of(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


class CookCache:
    """A bounded LRU cache keyed by content hash, with honest stats."""

    def __init__(self, max_entries: int = CACHE_MAX_ENTRIES) -> None:
        self.max_entries = max_entries
        self._entries: OrderedDict[str, FloatImage] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> FloatImage | None:
        image = self._entries.get(key)
        if image is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return image

    def put(self, key: str, image: FloatImage) -> None:
        self._entries[key] = image
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self.evictions += 1

    def __len__(self) -> int:
        return len(self._entries)

    def stats(self) -> dict:
        return {"entries": len(self._entries), "hits": self.hits,
                "misses": self.misses, "evictions": self.evictions}


@dataclass
class Cooked:
    """Everything a cook produced."""

    outputs: dict[str, FloatImage] = field(default_factory=dict)
    per_node: dict[str, dict] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    nodes_cooked: int = 0
    cache_hits: int = 0
    size: tuple[int, int] = (256, 256)


def _find_cycle(graph: Graph) -> list[str] | None:
    """Depth-first cycle search; returns the cycle path when found."""
    adjacency: dict[str, list[str]] = {nid: [] for nid in graph.nodes}
    for conn in graph.connections:
        if conn.from_node in adjacency and conn.to_node in adjacency:
            adjacency[conn.from_node].append(conn.to_node)
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(node_id: str) -> list[str] | None:
        state[node_id] = 1
        stack.append(node_id)
        for nxt in adjacency.get(node_id, []):
            if state.get(nxt, 0) == 1:
                at = stack.index(nxt)
                return stack[at:] + [nxt]
            if state.get(nxt, 0) == 0:
                found = visit(nxt)
                if found is not None:
                    return found
        stack.pop()
        state[node_id] = 2
        return None

    for node_id in graph.nodes:
        if state.get(node_id, 0) == 0:
            found = visit(node_id)
            if found is not None:
                return found
    return None


def topological_order(graph: Graph) -> list[str]:
    """Dependency order (Kahn); a cycle raises naming the cycle."""
    cycle = _find_cycle(graph)
    if cycle is not None:
        raise MaterialError(
            MaterialErrorKind.cycle,
            "graph has a cycle: " + " -> ".join(cycle))
    in_degree: dict[str, int] = {nid: 0 for nid in graph.nodes}
    downstream: dict[str, list[str]] = {nid: [] for nid in graph.nodes}
    for conn in graph.connections:
        if conn.from_node in in_degree and conn.to_node in in_degree:
            in_degree[conn.to_node] += 1
            downstream[conn.from_node].append(conn.to_node)
    ready = sorted(nid for nid, deg in in_degree.items() if deg == 0)
    order: list[str] = []
    while ready:
        node_id = ready.pop(0)
        order.append(node_id)
        for nxt in sorted(downstream[node_id]):
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                ready.append(nxt)
        ready.sort()
    if len(order) != len(graph.nodes):
        raise MaterialError(MaterialErrorKind.cycle,
                            "graph has a cycle (order incomplete)")
    return order


def _resolve_size(graph: Graph, node: Node,
                  inputs: dict[str, FloatImage | None],
                  definition: object) -> tuple[int, int]:
    """Upstream's size rule: generators take the parent size, filters
    take their main input's size; base.size offsets by powers of 2."""
    mode = node.base.size_mode
    if mode == "absolute":
        size = (1 << node.base.size[0], 1 << node.base.size[1])
    elif mode == "relative_to_parent":
        size = (graph.size, graph.size)
    else:
        # main input = first connected input image, else parent size
        main = None
        for image in inputs.values():
            if image is not None:
                main = image
                break
        size = (main.w, main.h) if main is not None else (graph.size, graph.size)
    offset = node.base.size
    w = size[0] * (1 << offset[0]) if offset[0] >= 0 else size[0] // (1 << -offset[0])
    h = size[1] * (1 << offset[1]) if offset[1] >= 0 else size[1] // (1 << -offset[1])
    return (max(1, min(MAX_SIZE, w)), max(1, min(MAX_SIZE, h)))


def _is_generator(definition_id: str) -> bool:
    return definition_id in (
        "uniform_color", "input_color", "input_grayscale", "bitmap",
        "perlin_noise", "cellular_noise", "voronoi", "brick_generator",
        "tile_generator", "checker", "gradient_linear", "gradient_radial",
        "shape")


def cook(graph: Graph, cache: CookCache | None = None,
         size: int | None = None, preview: bool = True) -> Cooked:
    """Cook a graph: topological order, cached per node.

    ``size`` overrides the graph size; ``preview`` caps outputs at
    512 px (upstream's preview cap). Returns every channel output,
    per-node images with their cache keys and hit flags, errors per
    node, and the cook counts.
    """
    if cache is None:
        # NB: never `cache or CookCache()` — an empty cache defines
        # __len__ and is falsy, which would silently discard the
        # caller's cache on every cook.
        cache = CookCache()
    base_size = size or graph.size
    if preview:
        base_size = min(base_size, PREVIEW_CAP)
    base_size = max(1, min(MAX_SIZE, base_size))
    graph_size = (base_size, base_size)
    order = topological_order(graph)
    cooked = Cooked(size=graph_size)
    images: dict[str, FloatImage] = {}
    keys: dict[str, str] = {}
    hits_before = cache.hits
    cooked_count = 0
    for node_id in order:
        node = graph.nodes[node_id]
        definition = by_id(node.def_)
        if definition is None:
            cooked.errors[node_id] = f"unknown node def {node.def_!r}"
            images[node_id] = FloatImage.magenta(*graph_size)
            keys[node_id] = hash_of(("missing", node.def_, node_id))
            cooked.per_node[node_id] = {"key": keys[node_id],
                                        "cached": False, "error": True}
            continue
        inputs: dict[str, FloatImage | None] = {}
        input_keys: list[str | None] = []
        for conn in graph.connections:
            if conn.to_node != node_id:
                continue
            image = images.get(conn.from_node)
            inputs[conn.to_port] = image
            input_keys.append(keys.get(conn.from_node))
        # ordered input list matching the definition's ports
        ordered: list[FloatImage | None] = []
        for port in definition.inputs:
            ordered.append(inputs.get(port.id))
        if _is_generator(node.def_):
            node_size = graph_size
        else:
            node_size = _resolve_size(graph, node, inputs, definition)
        if preview:
            node_size = (min(node_size[0], PREVIEW_CAP),
                         min(node_size[1], PREVIEW_CAP))
        if node.disabled and ordered and ordered[0] is not None:
            images[node_id] = ordered[0]
            keys[node_id] = hash_of(("disabled", node.def_, node_id,
                                     keys.get("passthrough")))
            cooked.per_node[node_id] = {"key": keys[node_id],
                                        "cached": False, "disabled": True}
            continue
        params = resolve_params(node.params, definition)
        seed = graph.seed + node.base.seed
        key = hash_of({
            "def": node.def_,
            "params": params,
            "size": node_size,
            "seed": seed,
            "tiling": node.base.tiling,
            "inputs": input_keys,
        })
        cached = cache.get(key)
        if cached is not None:
            images[node_id] = cached
            keys[node_id] = key
            cooked.per_node[node_id] = {"key": key, "cached": True}
            continue
        ctx = CookCtx(size=node_size, seed=seed,
                      tiling=(bool(node.base.tiling[0]),
                              bool(node.base.tiling[1])),
                      inputs=ordered, params=params)
        try:
            image = definition.cook(ctx)
        except MaterialError as exc:
            cooked.errors[node_id] = str(exc)
            image = FloatImage.magenta(*node_size)
        except Exception as exc:  # noqa: BLE001 — a cook must never crash the graph
            cooked.errors[node_id] = f"cook failed: {exc}"
            image = FloatImage.magenta(*node_size)
        cache.put(key, image)
        images[node_id] = image
        keys[node_id] = key
        cooked.per_node[node_id] = {"key": key, "cached": False}
        cooked_count += 1
    cooked.nodes_cooked = cooked_count
    cooked.cache_hits = cache.hits - hits_before
    # channel outputs: every "output" node feeds its channel
    for node_id in order:
        node = graph.nodes[node_id]
        if node.def_ != "output":
            continue
        definition = by_id("output")
        params = resolve_params(node.params, definition) if definition else {}
        channel = str(params.get("channel", "albedo"))
        image = images.get(node_id)
        if image is not None:
            cooked.outputs[channel] = image
    return cooked
