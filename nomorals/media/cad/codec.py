"""The ``.cadcraft`` package codec + DXF import/export.

The native format follows the layout organ's zip shape (lane
104): ``document.json`` + ``meta.json``, canonical JSON, fixed
zip timestamps, deterministic bytes, so an unedited document
writes back byte-identical. Bounds mirror the word organ's
OOXML package caps.

DXF import reads the real corpus drawings (Autodesk's published
DXF Reference, group-code pairs): HEADER, TABLES/LAYER, BLOCKS,
ENTITIES — LINE, CIRCLE, ARC, ELLIPSE, SPLINE, POINT,
LWPOLYLINE, POLYLINE, TEXT, MTEXT, SOLID (as a filled closed
polyline), DIMENSION, INSERT. Unknown entity types are kept as
opaque records in the document ``extra`` so a round trip never
silently drops them; the loss is reported, never hidden. DWG is
a binary format with no stdlib reader — it is refused honestly
with kind ``unsupported`` naming the format, never faked.
"""

from __future__ import annotations

import io
import json
import math
import zipfile

from .errors import CadError
from .model import (
    Arc,
    Block,
    CadDocument,
    CadPoint,
    CadText,
    Circle,
    Dimension,
    Ellipse,
    Insert,
    Layer,
    Line,
    Polyline,
    Spline,
    _fmt,
)

__all__ = [
    "MAX_JSON_DEPTH", "MAX_JSON_ELEMENTS", "MAX_PACKAGE_BYTES",
    "MAX_PART_BYTES", "export_dxf", "import_drawing", "import_dxf",
    "parse_document", "read_package", "to_document_bytes",
    "write_package",
]

MAX_PACKAGE_BYTES = 64 * 1024 * 1024
MAX_PART_BYTES = 32 * 1024 * 1024
MAX_JSON_DEPTH = 100
MAX_JSON_ELEMENTS = 500_000

_FORMAT_VERSION = 1
_DATE = (1980, 1, 1, 0, 0, 0)


def _canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _bounded_json_loads(data: bytes):
    """Parse JSON with a depth/element budget so a hostile file
    cannot blow up the parser (same habit as the word organ's
    bounded XML)."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CadError(f"document.json is not UTF-8: {exc}",
                       kind="parse") from exc
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CadError(f"document.json is not valid JSON: {exc}",
                       kind="parse") from exc
    count = [0]

    def walk(node, depth: int) -> None:
        if depth > MAX_JSON_DEPTH:
            raise CadError("document.json nests too deep",
                           kind="parse")
        count[0] += 1
        if count[0] > MAX_JSON_ELEMENTS:
            raise CadError("document.json has too many elements",
                           kind="parse")
        if isinstance(node, dict):
            for value in node.values():
                walk(value, depth + 1)
        elif isinstance(node, list):
            for value in node:
                walk(value, depth + 1)

    walk(obj, 0)
    return obj


def read_package(data: bytes) -> dict[str, bytes]:
    """Read a ``.cadcraft`` zip into {part name: bytes}."""
    if len(data) > MAX_PACKAGE_BYTES:
        raise CadError(
            f"package is {len(data)} bytes (max {MAX_PACKAGE_BYTES})",
            kind="parse")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise CadError(f"not a .cadcraft package: {exc}",
                       kind="parse") from exc
    parts: dict[str, bytes] = {}
    try:
        for info in archive.infolist():
            if info.is_dir():
                continue
            if info.file_size > MAX_PART_BYTES:
                raise CadError(f"part {info.filename} is too large",
                               kind="parse")
            parts[info.filename] = archive.read(info.filename)
    except CadError:
        raise
    except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
        raise CadError(f"cannot read .cadcraft package: {exc}",
                       kind="parse") from exc
    return parts


def write_package(parts: dict[str, bytes]) -> bytes:
    """Write parts deterministically (fixed timestamps, sorted)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(parts):
            info = zipfile.ZipInfo(name, date_time=_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, parts[name])
    return buf.getvalue()


def parse_document(data: bytes) -> CadDocument:
    parts = read_package(data)
    raw = parts.get("document.json")
    if raw is None:
        raise CadError("package has no document.json", kind="parse")
    doc = CadDocument.from_dict(_bounded_json_loads(raw))
    meta = parts.get("meta.json")
    if meta is not None:
        doc.extra.setdefault("_meta", _bounded_json_loads(meta))
    return doc


def to_document_bytes(doc: CadDocument) -> bytes:
    data = dict(doc.to_dict())
    meta = data.pop("_meta", None)
    parts = {"document.json": _canonical_json(data)}
    if meta is not None:
        parts["meta.json"] = _canonical_json(meta)
    return write_package(parts)


# ── DXF ───────────────────────────────────────────────────────────

def _dxf_pairs(data: bytes) -> list[tuple[int, str]]:
    """Parse DXF group-code pairs (code line, value line)."""
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover - defensive
        raise CadError(f"DXF is not readable text: {exc}",
                       kind="parse") from exc
    lines = text.splitlines()
    pairs: list[tuple[int, str]] = []
    i = 0
    while i + 1 < len(lines):
        try:
            code = int(lines[i].strip())
        except ValueError:
            i += 1
            continue
        pairs.append((code, lines[i + 1].rstrip("\n")))
        i += 2
    if not pairs:
        raise CadError("DXF has no group-code pairs", kind="parse")
    return pairs


def _group_map(pairs: list[tuple[int, str]]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for code, value in pairs:
        out.setdefault(code, []).append(value)
    return out


def _f(groups: dict[int, list[str]], code: int,
       default: float = 0.0) -> float:
    try:
        return float(groups[code][0])
    except (KeyError, IndexError, ValueError):
        return default


def _s(groups: dict[int, list[str]], code: int,
       default: str = "") -> str:
    try:
        return groups[code][0]
    except (KeyError, IndexError):
        return default


def _xy(groups: dict[int, list[str]], xcode: int = 10,
        ycode: int = 20) -> tuple[float, float]:
    return (_f(groups, xcode), _f(groups, ycode))


def _section_entities(pairs: list[tuple[int, str]],
                      name: str) -> list[list[tuple[int, str]]]:
    """Entity group-lists inside the named SECTION."""
    out: list[list[tuple[int, str]]] = []
    in_section = False
    current: list[tuple[int, str]] | None = None
    i = 0
    while i < len(pairs):
        code, value = pairs[i]
        if code == 0 and value == "SECTION":
            in_section = False
            current = None
        elif code == 2 and value == name and not in_section:
            in_section = True
        elif code == 0 and value == "ENDSEC":
            # Flush the section's last entity before leaving it.
            if current is not None:
                out.append(current)
            in_section = False
            current = None
        elif in_section and code == 0:
            if current is not None:
                out.append(current)
            current = [(code, value)]
        elif in_section and current is not None:
            current.append((code, value))
        i += 1
    if current is not None:
        out.append(current)
    return out


def _entity_from_groups(doc: CadDocument, kind: str,
                        groups: dict[int, list[str]],
                        extra_groups: list[tuple[int, str]]) -> object | None:
    """Build one entity from its DXF group codes. Returns None
    for entity types the model does not carry (the caller keeps
    them as opaque records)."""
    layer = _s(groups, 8, "0")
    color = None
    try:
        color_code = int(_s(groups, 62, ""))
        color = _ACI_COLORS.get(color_code)
    except ValueError:
        color = None
    # Mint the id only for entities the model keeps: an unmodelled
    # type (HATCH, …) must not consume a number, or the ids of the
    # entities after it shift on every round trip.
    if kind not in ("LINE", "CIRCLE", "ARC", "ELLIPSE", "POINT",
                    "TEXT", "MTEXT", "SOLID", "LWPOLYLINE", "SPLINE",
                    "DIMENSION", "INSERT"):
        return None
    # Degenerate radii are rejected here too, before the id is
    # minted, for the same stability reason.
    if kind in ("CIRCLE", "ARC") and _f(groups, 40) <= 0:
        return None
    if kind == "ELLIPSE" and (_f(groups, 40, 1.0) <= 0
                              or math.hypot(*_xy(groups, 11, 21)) <= 0):
        return None
    eid = doc.new_id()
    if kind == "LINE":
        return Line(id=eid, a=_xy(groups, 10, 20),
                    b=_xy(groups, 11, 21), layer=layer, color=color)
    if kind == "CIRCLE":
        radius = _f(groups, 40)
        if radius <= 0:
            return None
        return Circle(id=eid, center=_xy(groups, 10, 20),
                      radius=radius, layer=layer, color=color)
    if kind == "ARC":
        radius = _f(groups, 40)
        if radius <= 0:
            return None
        return Arc(id=eid, center=_xy(groups, 10, 20), radius=radius,
                   start=math.radians(_f(groups, 50)),
                   end=math.radians(_f(groups, 51)),
                   layer=layer, color=color)
    if kind == "ELLIPSE":
        ratio = _f(groups, 40, 1.0)
        if ratio <= 0:
            return None
        major = _xy(groups, 11, 21)
        if math.hypot(*major) <= 0:
            return None
        return Ellipse(id=eid, center=_xy(groups, 10, 20),
                       major=major, ratio=ratio,
                       start=_f(groups, 41, 0.0),
                       end=_f(groups, 42, math.tau),
                       layer=layer, color=color)
    if kind == "POINT":
        return CadPoint(id=eid, p=_xy(groups, 10, 20), layer=layer,
                        color=color)
    if kind in ("TEXT", "MTEXT"):
        text = _s(groups, 1, "")
        # MTEXT paragraph breaks are "\P"; TEXT never carries a
        # literal newline — normalise both to "\n" on read.
        text = text.replace("\\P", "\n")
        return CadText(id=eid, position=_xy(groups, 10, 20),
                       text=text, height=_f(groups, 40, 2.5),
                       rotation=_f(groups, 50, 0.0), layer=layer,
                       color=color)
    if kind == "SOLID":
        corners = [_xy(groups, 10, 20), _xy(groups, 11, 21),
                   _xy(groups, 12, 22), _xy(groups, 13, 23)]
        return Polyline(id=eid, vertices=corners, closed=True,
                        fill=True, layer=layer, color=color)
    if kind == "LWPOLYLINE":
        count = int(_f(groups, 90, 0))
        xs = groups.get(10, [])
        ys = groups.get(20, [])
        # Pair by index up to the shorter list: a malformed DXF
        # can carry mismatched coordinate lists, and the declared
        # count below truncates what survives.
        verts = [(float(xs[i]), float(ys[i]))
                 for i in range(min(len(xs), len(ys)))]
        if count:
            verts = verts[:count]
        flags = int(_f(groups, 70, 0))
        return Polyline(id=eid, vertices=verts, closed=bool(flags & 1),
                        layer=layer, color=color)
    if kind == "SPLINE":
        fit_count = int(_f(groups, 74, 0))
        fit_xs = groups.get(11, [])
        fit_ys = groups.get(21, [])
        if fit_count and fit_xs:
            pts = [(float(fit_xs[i]), float(fit_ys[i]))
                   for i in range(min(len(fit_xs), len(fit_ys)))
                   ][:fit_count]
            fit = True
        else:
            ctrl_xs = groups.get(10, [])
            ctrl_ys = groups.get(20, [])
            pts = [(float(ctrl_xs[i]), float(ctrl_ys[i]))
                   for i in range(min(len(ctrl_xs), len(ctrl_ys)))]
            fit = False
        if len(pts) < 2:
            return None
        flags = int(_f(groups, 70, 0))
        return Spline(id=eid, points=pts, closed=bool(flags & 1),
                      fit=fit, layer=layer, color=color)
    if kind == "DIMENSION":
        dim_type = int(_f(groups, 70, 0)) & 7
        dim_kind = {0: "aligned", 1: "aligned", 2: "angular",
                    3: "diameter", 4: "radius",
                    5: "angular"}.get(dim_type, "aligned")
        return Dimension(id=eid, kind=dim_kind, refs=[],
                         value=_f(groups, 42, 0.0),
                         position=_xy(groups, 11, 21),
                         text=_s(groups, 1, ""), layer=layer,
                         color=color)
    if kind == "INSERT":
        return Insert(id=eid, block=_s(groups, 2, ""),
                      position=_xy(groups, 10, 20),
                      rotation=math.radians(_f(groups, 50, 0.0)),
                      scale=(_f(groups, 41, 1.0), _f(groups, 42, 1.0)),
                      layer=layer, color=color)
    return None


_ACI_COLORS = {1: "#ff0000", 2: "#ffff00", 3: "#00ff00",
               4: "#00ffff", 5: "#0000ff", 6: "#ff00ff",
               7: "#ffffff", 8: "#808080", 9: "#c0c0c0"}


def import_dxf(data: bytes) -> CadDocument:
    """Import a DXF drawing into a CadDocument.

    Opaque entity types (hatches, leaders, xrefs, …) are kept in
    ``doc.extra["dxf_unmodelled"]`` as ``{type, count}`` rows so
    the import never silently drops geometry — the count is
    reported, and the entities are not modelled.
    """
    pairs = _dxf_pairs(data)
    doc = CadDocument.blank(title="DXF import")
    # Layers from the TABLES section.
    table_entities = _section_entities(pairs, "TABLES")
    # Blank() already carries layer "0"; seed from the live list
    # so the table's own "0" row is not appended a second time.
    seen_layers: set[str] = {layer.name for layer in doc.layers}
    for entity in table_entities:
        groups = _group_map(entity)
        if entity[0][1] == "LAYER":
            name = _s(groups, 2, "0")
            if name and name not in seen_layers \
                    and doc.layer(name) is None:
                seen_layers.add(name)
                doc.layers.append(Layer(name=name))
    if "0" not in seen_layers:
        doc.layers.insert(0, Layer(name="0"))
    # Blocks from the BLOCKS section.
    block_entities = _section_entities(pairs, "BLOCKS")
    current_block: Block | None = None
    for entity in block_entities:
        kind = entity[0][1]
        groups = _group_map(entity)
        if kind == "BLOCK":
            current_block = Block(name=_s(groups, 2, ""))
            doc.blocks[current_block.name] = current_block
        elif kind == "ENDBLK":
            current_block = None
        elif current_block is not None:
            built = _entity_from_groups(doc, kind, groups, entity)
            if built is not None:
                current_block.entities.append(built)
    # Model-space entities.
    unmodelled: dict[str, int] = {}
    for entity in _section_entities(pairs, "ENTITIES"):
        kind = entity[0][1]
        if kind in ("ENDSEC", "ENDBLK", "SEQEND", "VERTEX"):
            continue
        if kind == "POLYLINE":
            # Vertices follow as VERTEX records until SEQEND.
            continue
        groups = _group_map(entity)
        built = _entity_from_groups(doc, kind, groups, entity)
        if built is None:
            if kind not in ("ATTRIB",):
                unmodelled[kind] = unmodelled.get(kind, 0) + 1
            continue
        doc.entities[built.id] = built
    # POLYLINE entities with their VERTEX children.
    _import_polylines(doc, pairs)
    if unmodelled:
        doc.extra["dxf_unmodelled"] = [
            {"type": k, "count": v}
            for k, v in sorted(unmodelled.items())]
    return doc


def _import_polylines(doc: CadDocument,
                      pairs: list[tuple[int, str]]) -> None:
    """Second pass: POLYLINE + VERTEX + SEQEND groups in ENTITIES."""
    entities = _section_entities(pairs, "ENTITIES")
    i = 0
    while i < len(entities):
        entity = entities[i]
        if entity[0][1] != "POLYLINE":
            i += 1
            continue
        groups = _group_map(entity)
        verts: list[tuple[float, float]] = []
        j = i + 1
        while j < len(entities) and entities[j][0][1] == "VERTEX":
            vgroups = _group_map(entities[j])
            verts.append(_xy(vgroups, 10, 20))
            j += 1
        if len(verts) >= 2:
            flags = int(_f(groups, 70, 0))
            poly = Polyline(id=doc.new_id(), vertices=verts,
                            closed=bool(flags & 1),
                            layer=_s(groups, 8, "0"))
            doc.entities[poly.id] = poly
        i = j


def import_drawing(filename: str, data: bytes) -> CadDocument:
    """Import by file extension: ``.dxf`` (parsed), ``.cadcraft``
    (native), ``.dwg`` refused honestly."""
    lower = filename.lower()
    if lower.endswith(".dxf"):
        return import_dxf(data)
    if lower.endswith(".cadcraft"):
        return parse_document(data)
    if lower.endswith(".dwg"):
        raise CadError(
            "DWG import needs the binary DWG reader, which is not "
            "ported in this build — export the drawing as DXF first",
            kind="unsupported")
    raise CadError(f"unknown drawing format for {filename!r}",
                   kind="unsupported")


def export_dxf(doc: CadDocument) -> bytes:
    """Write a minimal DXF (R12 shape) for the modelled entities."""
    lines: list[str] = []

    def pair(code: int, value: str) -> None:
        lines.append(f"{code:>3}")
        lines.append(str(value))

    def section(name: str) -> None:
        pair(0, "SECTION")
        pair(2, name)

    def endsec() -> None:
        pair(0, "ENDSEC")

    section("HEADER")
    pair(9, "$ACADVER")
    pair(1, "AC1032")
    pair(9, "$INSUNITS")
    pair(70, "4")
    endsec()
    section("TABLES")
    pair(0, "TABLE")
    pair(2, "LAYER")
    pair(70, str(len(doc.layers)))
    for layer in doc.layers:
        pair(0, "LAYER")
        pair(2, layer.name)
        pair(70, "0")
        pair(62, "7")
        pair(6, "Continuous")
    pair(0, "ENDTAB")
    endsec()
    if doc.blocks:
        section("BLOCKS")
        for block in doc.blocks.values():
            pair(0, "BLOCK")
            pair(8, "0")
            pair(2, block.name)
            pair(70, "0")
            pair(10, "0.0")
            pair(20, "0.0")
            for entity in block.entities:
                _write_entity(lines, pair, entity)
            pair(0, "ENDBLK")
        endsec()
    section("ENTITIES")
    for entity in doc.entities.values():
        _write_entity(lines, pair, entity)
    endsec()
    pair(0, "EOF")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_entity(lines: list[str], pair, entity) -> None:
    kind = type(entity).__name__
    if kind == "Line":
        pair(0, "LINE")
        pair(8, entity.layer)
        pair(10, _fmt(entity.a[0]))
        pair(20, _fmt(entity.a[1]))
        pair(11, _fmt(entity.b[0]))
        pair(21, _fmt(entity.b[1]))
    elif kind == "Circle":
        pair(0, "CIRCLE")
        pair(8, entity.layer)
        pair(10, _fmt(entity.center[0]))
        pair(20, _fmt(entity.center[1]))
        pair(40, _fmt(entity.radius))
    elif kind == "Arc":
        pair(0, "ARC")
        pair(8, entity.layer)
        pair(10, _fmt(entity.center[0]))
        pair(20, _fmt(entity.center[1]))
        pair(40, _fmt(entity.radius))
        pair(50, _fmt(math.degrees(entity.start)))
        pair(51, _fmt(math.degrees(entity.end)))
    elif kind == "Ellipse":
        pair(0, "ELLIPSE")
        pair(8, entity.layer)
        pair(10, _fmt(entity.center[0]))
        pair(20, _fmt(entity.center[1]))
        pair(11, _fmt(entity.major[0]))
        pair(21, _fmt(entity.major[1]))
        pair(40, _fmt(entity.ratio))
        pair(41, _fmt(entity.start))
        pair(42, _fmt(entity.end))
    elif kind == "CadPoint":
        pair(0, "POINT")
        pair(8, entity.layer)
        pair(10, _fmt(entity.p[0]))
        pair(20, _fmt(entity.p[1]))
    elif kind == "Polyline":
        if entity.fill and len(entity.vertices) == 4:
            pair(0, "SOLID")
            pair(8, entity.layer)
            for idx, code in enumerate((10, 11, 12, 13)):
                pair(code, _fmt(entity.vertices[idx][0]))
                pair(code + 10, _fmt(entity.vertices[idx][1]))
        else:
            pair(0, "LWPOLYLINE")
            pair(8, entity.layer)
            pair(90, str(len(entity.vertices)))
            pair(70, "1" if entity.closed else "0")
            for v in entity.vertices:
                pair(10, _fmt(v[0]))
                pair(20, _fmt(v[1]))
    elif kind == "Spline":
        pair(0, "SPLINE")
        pair(8, entity.layer)
        pair(70, "1" if entity.closed else "0")
        pair(71, "3")
        pair(72, str(len(entity.points) + 4))
        pair(73, str(len(entity.points)))
        pair(74, str(len(entity.points)) if entity.fit else "0")
        for p in entity.points:
            pair(10, _fmt(p[0]))
            pair(20, _fmt(p[1]))
        if entity.fit:
            for p in entity.points:
                pair(11, _fmt(p[0]))
                pair(21, _fmt(p[1]))
    elif kind == "CadText":
        pair(0, "TEXT")
        pair(8, entity.layer)
        pair(10, _fmt(entity.position[0]))
        pair(20, _fmt(entity.position[1]))
        pair(40, _fmt(entity.height))
        # A literal newline inside a group value would shift every
        # following pair; DXF writes it as the "\P" break.
        pair(1, entity.text.replace("\n", "\\P"))
        pair(50, _fmt(entity.rotation))
    elif kind == "Dimension":
        pair(0, "DIMENSION")
        pair(8, entity.layer)
        pair(10, _fmt(entity.position[0]))
        pair(20, _fmt(entity.position[1]))
        pair(11, _fmt(entity.position[0]))
        pair(21, _fmt(entity.position[1]))
        dim_type = {"aligned": 1, "linear": 0, "angular": 2,
                    "diameter": 3, "radius": 4}.get(entity.kind, 1)
        pair(70, str(dim_type))
        pair(42, _fmt(entity.value))
        pair(1, entity.text)
    elif kind == "Insert":
        pair(0, "INSERT")
        pair(8, entity.layer)
        pair(2, entity.block)
        pair(10, _fmt(entity.position[0]))
        pair(20, _fmt(entity.position[1]))
        pair(50, _fmt(math.degrees(entity.rotation)))
        pair(41, _fmt(entity.scale[0]))
        pair(42, _fmt(entity.scale[1]))
    # Unknown kinds are not written: the export names what it
    # writes, and import keeps unmodelled types in extra.
