"""Material library — named presets, each a real node graph.

MaterialCraft port of upstream's ``crates/engine`` sample materials
and graph templates: every preset is a graph built from the same
catalog nodes the owner can edit, not a baked image. Each preset is
cooked and previewed by the parity runner, and searchable by name,
tag and PBR channel.
"""

from __future__ import annotations

from .model import Connection, Graph, Node

_BROWN_LIGHT = [0.55, 0.35, 0.2, 1.0]
_BROWN_DARK = [0.25, 0.14, 0.07, 1.0]
_BRICK_LIGHT = [0.62, 0.28, 0.22, 1.0]
_BRICK_DARK = [0.45, 0.18, 0.14, 1.0]


def _node(node_id: str, def_: str, x: float, y: float,
          params: dict | None = None, label: str = "") -> Node:
    return Node(id=node_id, def_=def_, label=label, pos=[x, y],
                params=params or {})


def _out(graph: Graph, channel: str, node_id: str) -> None:
    graph.outputs[channel] = {"node": node_id, "port": "output"}


def _wire(graph: Graph, src: str, dst: str,
          src_port: str = "output", dst_port: str = "input") -> None:
    graph.connections.append(Connection(src, src_port, dst, dst_port))


def _wire2(graph: Graph, src: str, dst: str, dst_port: str) -> None:
    graph.connections.append(Connection(src, "output", dst, dst_port))


def metal() -> Graph:
    """Brushed metal: stretched Perlin grain drives roughness."""
    graph = Graph(id="main", name="Brushed Metal", size=256, seed=7)
    graph.nodes["albedo"] = _node("albedo", "uniform_color", 40, 40,
                                  {"color": [0.62, 0.63, 0.65, 1.0]})
    graph.nodes["grain"] = _node("grain", "perlin_noise", 40, 140,
                                 {"scale": 2.0, "octaves": 3})
    graph.nodes["rough"] = _node("rough", "levels", 220, 140,
                                 {"in_black": 0.2, "in_white": 0.8})
    graph.nodes["metal"] = _node("metal", "input_grayscale", 40, 240,
                                 {"value": 1.0})
    graph.nodes["flat"] = _node("flat", "input_grayscale", 220, 240,
                                {"value": 0.5})
    graph.nodes["nrm"] = _node("nrm", "height_to_normal", 400, 140,
                               {"strength": 0.3})
    graph.nodes["o_albedo"] = _node("o_albedo", "output", 580, 40,
                                    {"channel": "albedo"})
    graph.nodes["o_rough"] = _node("o_rough", "output", 580, 140,
                                   {"channel": "roughness"})
    graph.nodes["o_metal"] = _node("o_metal", "output", 580, 240,
                                   {"channel": "metallic"})
    graph.nodes["o_nrm"] = _node("o_nrm", "output", 580, 340,
                                 {"channel": "normal"})
    _wire(graph, "grain", "rough")
    _wire(graph, "grain", "nrm")
    _wire(graph, "albedo", "o_albedo")
    _wire(graph, "rough", "o_rough")
    _wire(graph, "metal", "o_metal")
    _wire(graph, "nrm", "o_nrm")
    for channel, node in (("albedo", "o_albedo"), ("roughness", "o_rough"),
                          ("metallic", "o_metal"), ("normal", "o_nrm")):
        _out(graph, channel, node)
    return graph


def wood() -> Graph:
    """Wood planks: brick courses warped by grain, brown gradient."""
    graph = Graph(id="main", name="Wood Planks", size=256, seed=11)
    graph.nodes["planks"] = _node("planks", "brick_generator", 40, 40,
                                  {"brick_width": 128.0, "brick_height": 32.0,
                                   "mortar": 2.0, "offset": 0.5,
                                   "color_a": [0.5, 0.32, 0.18, 1.0],
                                   "color_b": [0.45, 0.28, 0.15, 1.0],
                                   "mortar_color": [0.2, 0.12, 0.06, 1.0]})
    graph.nodes["grain"] = _node("grain", "perlin_noise", 40, 140,
                                 {"scale": 3.0, "octaves": 4})
    graph.nodes["warp"] = _node("warp", "warp", 220, 40,
                                {"intensity": 0.08, "scale": 3.0})
    graph.nodes["tint"] = _node("tint", "gradient_map", 400, 40,
                                {"stops": [[0.0] + _BROWN_DARK,
                                           [1.0] + _BROWN_LIGHT]})
    graph.nodes["rough"] = _node("rough", "levels", 400, 140,
                                 {"in_black": 0.3, "in_white": 0.9})
    graph.nodes["nrm"] = _node("nrm", "height_to_normal", 400, 240,
                               {"strength": 0.8})
    graph.nodes["o_albedo"] = _node("o_albedo", "output", 580, 40,
                                    {"channel": "albedo"})
    graph.nodes["o_rough"] = _node("o_rough", "output", 580, 140,
                                   {"channel": "roughness"})
    graph.nodes["o_nrm"] = _node("o_nrm", "output", 580, 240,
                                 {"channel": "normal"})
    _wire(graph, "planks", "warp")
    _wire(graph, "warp", "tint")
    _wire(graph, "grain", "rough")
    _wire(graph, "grain", "nrm")
    _wire(graph, "tint", "o_albedo")
    _wire(graph, "rough", "o_rough")
    _wire(graph, "nrm", "o_nrm")
    for channel, node in (("albedo", "o_albedo"), ("roughness", "o_rough"),
                          ("normal", "o_nrm")):
        _out(graph, channel, node)
    return graph


def stone() -> Graph:
    """Stone: Voronoi plates, cracks as height, gray albedo."""
    graph = Graph(id="main", name="Stone", size=256, seed=23)
    graph.nodes["plates"] = _node("plates", "voronoi", 40, 40,
                                  {"scale": 5.0, "edge_width": 0.08})
    graph.nodes["cracks"] = _node("cracks", "cellular_noise", 40, 140,
                                  {"scale": 5.0, "mode": "f2-f1"})
    graph.nodes["albedo"] = _node("albedo", "levels", 220, 40,
                                  {"in_black": 0.1, "in_white": 0.9})
    graph.nodes["nrm"] = _node("nrm", "height_to_normal", 220, 140,
                               {"strength": 1.5})
    graph.nodes["rough"] = _node("rough", "roughness_from_height", 400, 140,
                                 {"base": 0.85, "variation": 0.3})
    graph.nodes["ao"] = _node("ao", "ao_from_height", 400, 240,
                              {"strength": 0.6})
    graph.nodes["o_albedo"] = _node("o_albedo", "output", 580, 40,
                                    {"channel": "albedo"})
    graph.nodes["o_rough"] = _node("o_rough", "output", 580, 140,
                                   {"channel": "roughness"})
    graph.nodes["o_nrm"] = _node("o_nrm", "output", 580, 240,
                                 {"channel": "normal"})
    graph.nodes["o_ao"] = _node("o_ao", "output", 580, 340,
                                {"channel": "ao"})
    graph.nodes["o_height"] = _node("o_height", "output", 580, 440,
                                    {"channel": "height"})
    _wire(graph, "plates", "albedo")
    _wire(graph, "cracks", "nrm")
    _wire(graph, "cracks", "rough")
    _wire(graph, "cracks", "ao")
    _wire(graph, "cracks", "o_height")
    _wire(graph, "albedo", "o_albedo")
    _wire(graph, "rough", "o_rough")
    _wire(graph, "nrm", "o_nrm")
    _wire(graph, "ao", "o_ao")
    for channel, node in (("albedo", "o_albedo"), ("roughness", "o_rough"),
                          ("normal", "o_nrm"), ("ao", "o_ao"),
                          ("height", "o_height")):
        _out(graph, channel, node)
    return graph


def fabric() -> Graph:
    """Fabric weave: a tile grid with thread noise, very rough."""
    graph = Graph(id="main", name="Fabric Weave", size=256, seed=31)
    graph.nodes["weave"] = _node("weave", "tile_generator", 40, 40,
                                 {"rows": 16, "cols": 16, "gap": 1.0,
                                  "color_a": [0.45, 0.3, 0.55, 1.0],
                                  "color_b": [0.4, 0.26, 0.5, 1.0],
                                  "gap_color": [0.15, 0.1, 0.2, 1.0]})
    graph.nodes["thread"] = _node("thread", "perlin_noise", 40, 140,
                                  {"scale": 16.0, "octaves": 2})
    graph.nodes["blend"] = _node("blend", "blend", 220, 40,
                                 {"mode": "multiply", "opacity": 0.35})
    graph.nodes["rough"] = _node("rough", "input_grayscale", 220, 140,
                                 {"value": 0.95})
    graph.nodes["o_albedo"] = _node("o_albedo", "output", 580, 40,
                                    {"channel": "albedo"})
    graph.nodes["o_rough"] = _node("o_rough", "output", 580, 140,
                                   {"channel": "roughness"})
    _wire2(graph, "weave", "blend", "base")
    _wire2(graph, "thread", "blend", "blend")
    _wire(graph, "blend", "o_albedo")
    _wire(graph, "rough", "o_rough")
    for channel, node in (("albedo", "o_albedo"), ("roughness", "o_rough")):
        _out(graph, channel, node)
    return graph


def glass() -> Graph:
    """Glass: smooth and flat. No transmission — the renderer does
    not do refraction, and the preset says so honestly."""
    graph = Graph(id="main", name="Glass", size=256, seed=5)
    graph.nodes["albedo"] = _node("albedo", "uniform_color", 40, 40,
                                  {"color": [0.75, 0.82, 0.85, 1.0]})
    graph.nodes["rough"] = _node("rough", "input_grayscale", 40, 140,
                                 {"value": 0.05})
    graph.nodes["metal"] = _node("metal", "input_grayscale", 40, 240,
                                 {"value": 0.0})
    graph.nodes["o_albedo"] = _node("o_albedo", "output", 580, 40,
                                    {"channel": "albedo"})
    graph.nodes["o_rough"] = _node("o_rough", "output", 580, 140,
                                   {"channel": "roughness"})
    graph.nodes["o_metal"] = _node("o_metal", "output", 580, 240,
                                   {"channel": "metallic"})
    _wire(graph, "albedo", "o_albedo")
    _wire(graph, "rough", "o_rough")
    _wire(graph, "metal", "o_metal")
    for channel, node in (("albedo", "o_albedo"), ("roughness", "o_rough"),
                          ("metallic", "o_metal")):
        _out(graph, channel, node)
    return graph


def brick() -> Graph:
    """Red brick wall: brick courses, mortar, height and normal."""
    graph = Graph(id="main", name="Red Brick", size=256, seed=17)
    graph.nodes["courses"] = _node("courses", "brick_generator", 40, 40,
                                   {"brick_width": 42.0, "brick_height": 16.0,
                                    "mortar": 3.0, "offset": 0.5,
                                    "color_a": _BRICK_LIGHT,
                                    "color_b": _BRICK_DARK,
                                    "mortar_color": [0.78, 0.76, 0.72, 1.0]})
    graph.nodes["grain"] = _node("grain", "perlin_noise", 40, 140,
                                 {"scale": 8.0, "octaves": 3})
    graph.nodes["tint"] = _node("tint", "blend", 220, 40,
                                {"mode": "multiply", "opacity": 0.25})
    graph.nodes["height"] = _node("height", "levels", 220, 140,
                                  {"in_black": 0.0, "in_white": 1.0})
    graph.nodes["nrm"] = _node("nrm", "height_to_normal", 400, 140,
                               {"strength": 1.2})
    graph.nodes["rough"] = _node("rough", "input_grayscale", 400, 240,
                                 {"value": 0.85})
    graph.nodes["o_albedo"] = _node("o_albedo", "output", 580, 40,
                                    {"channel": "albedo"})
    graph.nodes["o_rough"] = _node("o_rough", "output", 580, 140,
                                   {"channel": "roughness"})
    graph.nodes["o_nrm"] = _node("o_nrm", "output", 580, 240,
                                 {"channel": "normal"})
    graph.nodes["o_height"] = _node("o_height", "output", 580, 340,
                                    {"channel": "height"})
    _wire2(graph, "courses", "tint", "base")
    _wire2(graph, "grain", "tint", "blend")
    _wire(graph, "courses", "height")
    _wire(graph, "height", "nrm")
    _wire(graph, "tint", "o_albedo")
    _wire(graph, "rough", "o_rough")
    _wire(graph, "nrm", "o_nrm")
    _wire(graph, "height", "o_height")
    for channel, node in (("albedo", "o_albedo"), ("roughness", "o_rough"),
                          ("normal", "o_nrm"), ("height", "o_height")):
        _out(graph, channel, node)
    return graph


PRESETS: dict[str, dict] = {
    "metal": {"name": "Brushed Metal", "tags": ["metal", "brushed", "pbr"],
              "channels": ["albedo", "roughness", "metallic", "normal"],
              "build": metal},
    "wood": {"name": "Wood Planks", "tags": ["wood", "planks", "grain"],
             "channels": ["albedo", "roughness", "normal"],
             "build": wood},
    "stone": {"name": "Stone", "tags": ["stone", "rock", "plates"],
              "channels": ["albedo", "roughness", "normal", "ao", "height"],
              "build": stone},
    "fabric": {"name": "Fabric Weave", "tags": ["fabric", "cloth", "weave"],
               "channels": ["albedo", "roughness"],
               "build": fabric},
    "glass": {"name": "Glass", "tags": ["glass", "smooth", "transparent"],
              "channels": ["albedo", "roughness", "metallic"],
              "build": glass,
              "note": "no transmission: the renderer does not refract"},
    "brick": {"name": "Red Brick", "tags": ["brick", "wall", "masonry"],
              "channels": ["albedo", "roughness", "normal", "height"],
              "build": brick},
}


def list_presets() -> list[dict]:
    """Every preset: id, name, tags, channels."""
    return [{"id": preset_id, "name": info["name"],
             "tags": info["tags"], "channels": info["channels"],
             "note": info.get("note", "")}
            for preset_id, info in PRESETS.items()]


def search(query: str) -> list[dict]:
    """Search presets by name, tag or channel (case-insensitive)."""
    needle = query.strip().lower()
    out = []
    for preset in list_presets():
        hay = " ".join([preset["name"]] + preset["tags"]
                       + preset["channels"]).lower()
        if not needle or needle in hay:
            out.append(preset)
    return out


def build_preset(preset_id: str) -> Graph:
    info = PRESETS.get(preset_id)
    if info is None:
        from .errors import MaterialError, MaterialErrorKind
        raise MaterialError(MaterialErrorKind.not_found,
                            f"unknown preset {preset_id!r}")
    return info["build"]()
