"""Devon Atlas — the polygon-fill pass (closes M7's remainder).

M7 shipped overlay polygons as OUTLINES ONLY with an explicit
note: the MapOverlay seam had no fill.  This pass closes that
remainder:

* the seam carries regions (``OverlayPolygon``: outer ring +
  hole rings) and the vector renderer paints them FILLED under
  their ring stroke — fill color, fill opacity and ring stroke
  are the style document's overlay paint
  (``style.OverlayPaint``), never hardcoded in the renderer;
* the fill rule is the renderer's own even-odd scanline,
  applied per polygon: holes empty, and the parts of a
  multi-polygon never cancel each other;
* overlay features carry holes whole from the GeoJSON gate
  (the old "holes are not carried" loss is gone), through the
  store's serialization;
* a polygon whose geometry cannot be filled honestly — a
  self-intersecting or degenerate ring, a hole outside its
  outer ring — degrades to its closed outline with the defect
  named (conversion note + receipt detail), never a crash and
  never a silent wrong fill;
* labels still draw AFTER the overlay (the text backend is
  handed the filled buffer) with the style's halo — the style
  data carries the contrast answer;
* a render with NO overlay is byte-identical to the render
  before this pass (pinned by sha256 against a hash verified
  byte-for-byte against the pre-pass code on base 578f2ded).

Fixture pins use a synthetic z2 pack (one water tile) built
in-test; the real-pack acceptance lives in the pass's verify
run over pack-a-nigeria, not in this hermetic suite.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from nomorals.maps.coords import (
    BBox,
    lat_to_y_float,
    lon_to_x_float,
    tile_x_to_lon,
    tile_y_to_lat,
)
from nomorals.maps.manifest import (
    build_manifest,
    load_manifest,
    tile_relpath,
    write_manifest,
)
from nomorals.maps.overlays import (
    Overlay,
    OverlayFeature,
    OverlayStore,
    geojson_to_features,
    overlay_to_map_overlay,
    validate_features,
)
from nomorals.maps.pngcodec import decode_png
from nomorals.maps.style import (
    DEFAULT_STYLE,
    KIND_FILL,
    KIND_LABEL,
    MapStyle,
    OverlayPaint,
    StyleError,
    StyleLayer,
    parse_color,
    style_hash,
)
from nomorals.maps.vector_render import (
    MapOverlay,
    OverlayPolygon,
    VectorRenderer,
    polygon_defects,
)

ATTRIBUTION = "© Fill Fixture contributors"
ZOOM = 2
TILE_X, TILE_Y = 1, 1
W = H = 256
BBOX = BBox(west=tile_x_to_lon(TILE_X, ZOOM),
            south=tile_y_to_lat(TILE_Y + 1, ZOOM),
            east=tile_x_to_lon(TILE_X + 1, ZOOM),
            north=tile_y_to_lat(TILE_Y, ZOOM))

WATER = parse_color("#17648f")
FILL = (255, 0, 0)
STROKE = (255, 255, 255)

# the overlay square used across the suite: (lat, lon) corners
SQUARE = ((20.0, -70.0), (20.0, -30.0), (40.0, -30.0),
          (40.0, -70.0))
HOLE = ((27.0, -56.0), (27.0, -44.0), (33.0, -44.0),
        (33.0, -56.0))
BOWTIE = ((20.0, -70.0), (40.0, -30.0), (20.0, -30.0),
          (40.0, -70.0))


def _px(lat: float, lon: float) -> tuple[int, int]:
    """The renderer's projection of (lat, lon) into the output."""
    x = (lon_to_x_float(lon, ZOOM) - float(TILE_X)) * W
    y = (lat_to_y_float(lat, ZOOM) - float(TILE_Y)) * H
    return int(x), int(y)


def _pixel(rgb: bytes, x: int, y: int) -> tuple[int, int, int]:
    at = (y * W + x) * 3
    return tuple(rgb[at:at + 3])


def _blend(rgb: tuple[int, int, int], under: tuple[int, int, int],
           opacity: float) -> tuple[int, int, int]:
    keep = 1.0 - opacity
    return tuple(int(rgb[i] * opacity + under[i] * keep + 0.5)
                 for i in range(3))


# ── a tiny real protobuf encoder (the M3 fixture idiom) ──────


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        if n:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _fv(no: int, value: int) -> bytes:
    return _varint((no << 3) | 0) + _varint(value)


def _fb(no: int, body: bytes) -> bytes:
    return _varint((no << 3) | 2) + _varint(len(body)) + body


def _zz(n: int) -> int:
    return (n << 1) if n >= 0 else ((-n << 1) - 1)


def _packed(nums) -> bytes:
    return b"".join(_varint(int(n)) for n in nums)


def _cmd(command: int, count: int) -> int:
    return (count << 3) | command


def _geometry(*segments, close: bool = False) -> list[int]:
    out: list[int] = []
    cx = cy = 0
    for command, points in segments:
        out.append(_cmd(command, len(points)))
        for x, y in points:
            out.append(_zz(x - cx))
            out.append(_zz(y - cy))
            cx, cy = x, y
    if close:
        out.append(_cmd(7, 1))
    return out


def _feature(*, tags=(), geom_type=0, geometry=()) -> bytes:
    body = b""
    if tags:
        body += _fb(2, _packed(tags))
    body += _fv(3, geom_type)
    if geometry:
        body += _fb(4, _packed(geometry))
    return body


def _layer(name: str, features, keys=(), values=(), extent=4096,
           version=2) -> bytes:
    body = b""
    body += _fb(1, name.encode("utf-8"))
    for feat in features:
        body += _fb(2, feat)
    for key in keys:
        body += _fb(3, key.encode("utf-8"))
    for value in values:
        body += _fb(4, value)
    if extent != 4096:
        body += _fv(5, extent)
    body += _fv(15, version)
    return body


def _tile(*layers) -> bytes:
    return b"".join(_fb(3, layer) for layer in layers)


def _fixture_tile() -> bytes:
    """One z2 tile: water over the whole extent + a named place
    point at the overlay square's centre (30, -50)."""
    water = _layer(
        "water",
        [_feature(geom_type=3,
                  geometry=_geometry(
                      (1, [(0, 0)]),
                      (2, [(4096, 0), (4096, 4096), (0, 4096)]),
                      close=True))])
    cx, cy = _px(30.0, -50.0)
    place = _layer(
        "place",
        [_feature(tags=(0, 0), geom_type=1,
                  geometry=_geometry(
                      (1, [(cx * 16, cy * 16)])))],
        keys=("name",),
        values=(_fb(1, b"Fill Town"),))
    return _tile(water, place)


def _make_pack(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    tile_path = root / tile_relpath(ZOOM, TILE_X, TILE_Y, "mvt")
    tile_path.parent.mkdir(parents=True, exist_ok=True)
    tile_path.write_bytes(_fixture_tile())
    manifest = build_manifest(
        root, pack_id="fixture-fill", name="Fixture Fill",
        bbox=BBox(west=-180.0, south=-85.0, east=180.0, north=85.0),
        zoom_min=ZOOM, zoom_max=ZOOM, attribution=ATTRIBUTION,
        built_at="2026-11-10T00:00:00+00:00", tile_format="mvt")
    write_manifest(root, manifest)
    return load_manifest(root)


def _style(**paint) -> MapStyle:
    base = {"fill": "#ff0000", "fill_opacity": 1.0,
            "stroke": "#ffffff", "stroke_width": 2.0}
    base.update(paint)
    return MapStyle(
        name="fill-test", background="#000000", no_data="#1a1a20",
        layers=(
            StyleLayer("water", KIND_FILL, fill="#17648f"),
            StyleLayer("place", KIND_LABEL, text_field="name",
                       text_size=12.0, text_color="#ffffff",
                       halo_color="#000000"),
        ),
        overlay=OverlayPaint(**base))


class _CapturingBackend:
    """The text seam, test-double: records the buffer it was
    handed (labels composite over whatever is beneath) and
    returns it unchanged."""

    def __init__(self) -> None:
        self.seen: bytes | None = None
        self.labels: tuple = ()

    def available(self) -> bool:
        return True

    def draw_labels(self, width, height, rgb, labels):
        self.seen = bytes(rgb)
        self.labels = labels
        return bytes(rgb)


@pytest.fixture()
def pack(tmp_path):
    root = tmp_path / "pack"
    manifest = _make_pack(root)
    return root, manifest


def _render(pack, overlay=None, style=None, backend=None):
    root, manifest = pack
    renderer = VectorRenderer(
        root, manifest, style=style or _style(),
        text_backend=backend or _CapturingBackend())
    result = renderer.render(BBOX, ZOOM, size=(W, H),
                             overlay=overlay)
    assert result.image is not None, result.receipt.detail
    _w, _h, rgb = decode_png(result.image)
    return result, rgb


# ── the fill itself ──────────────────────────────────────────


class TestPolygonFill:
    def test_square_fills_interior_edge_and_outside(self, pack):
        overlay = MapOverlay(polygons=(
            OverlayPolygon(rings=(SQUARE,), label="plot"),))
        result, rgb = _render(pack, overlay)
        # centre + a point just inside the south edge: the fill
        assert _pixel(rgb, *_px(30.0, -50.0)) == FILL
        assert _pixel(rgb, *_px(22.0, -50.0)) == FILL
        # on the west boundary: the ring stroke, over the fill
        assert _pixel(rgb, *_px(30.0, -70.0)) == STROKE
        # outside the polygon: the water beneath, untouched
        assert _pixel(rgb, *_px(30.0, -20.0)) == WATER
        # the receipt counts the polygon, no degradation note
        assert "1 polygon(s)" in result.receipt.detail
        assert "outline only" not in result.receipt.detail

    def test_hole_stays_unfilled(self, pack):
        overlay = MapOverlay(polygons=(
            OverlayPolygon(rings=(SQUARE, HOLE), label="plot"),))
        _result, rgb = _render(pack, overlay)
        # the hole's centre shows the water beneath
        assert _pixel(rgb, *_px(30.0, -50.0)) == WATER
        # inside the outer ring, outside the hole: filled
        assert _pixel(rgb, *_px(30.0, -65.0)) == FILL
        # the hole's own boundary is stroked too
        assert _pixel(rgb, *_px(30.0, -56.0)) == STROKE

    def test_two_polygons_fill_independently(self, pack):
        outer = ((15.0, -75.0), (15.0, -25.0), (45.0, -25.0),
                 (45.0, -75.0))
        inner = ((25.0, -55.0), (25.0, -45.0), (35.0, -45.0),
                 (35.0, -55.0))
        overlay = MapOverlay(polygons=(
            OverlayPolygon(rings=(outer,), label="outer"),
            OverlayPolygon(rings=(inner,), label="inner")))
        _result, rgb = _render(pack, overlay)
        # the inner polygon's centre is FILLED — each polygon
        # fills in its own even-odd pass, so a polygon inside
        # another never cancels the interior out
        assert _pixel(rgb, *_px(30.0, -50.0)) == FILL
        assert _pixel(rgb, *_px(20.0, -30.0)) == FILL

    def test_markers_paint_over_the_fill(self, pack):
        overlay = MapOverlay(
            polygons=(OverlayPolygon(rings=(SQUARE,),
                                     label="plot"),),
            markers=((30.0, -50.0),))
        _result, rgb = _render(pack, overlay)
        cx, cy = _px(30.0, -50.0)
        # the marker's ink centre sits over the fill
        assert _pixel(rgb, cx, cy) == (0, 0, 0)
        # its gold ring (5px out) sits over the fill too
        assert _pixel(rgb, cx + 5, cy) == (232, 179, 75)

    def test_opacity_blends_over_what_is_beneath(self, pack):
        style = _style(fill_opacity=0.5)
        overlay = MapOverlay(polygons=(
            OverlayPolygon(rings=(SQUARE,), label="plot"),))
        _result, rgb = _render(pack, overlay, style=style)
        assert _pixel(rgb, *_px(30.0, -50.0)) == _blend(
            FILL, WATER, 0.5)

    def test_default_style_paint_is_translucent_gold(self, pack):
        overlay = MapOverlay(polygons=(
            OverlayPolygon(rings=(SQUARE,), label="plot"),))
        _result, rgb = _render(pack, overlay, style=DEFAULT_STYLE)
        gold = parse_color(DEFAULT_STYLE.overlay.fill)
        expected = _blend(gold, WATER,
                          DEFAULT_STYLE.overlay.fill_opacity)
        centre = _pixel(rgb, *_px(30.0, -50.0))
        assert centre == expected
        assert centre != gold and centre != WATER

    def test_style_data_change_changes_the_fill(self, pack):
        overlay = MapOverlay(polygons=(
            OverlayPolygon(rings=(SQUARE,), label="plot"),))
        _r1, rgb_red = _render(pack, overlay)
        _r2, rgb_green = _render(
            pack, overlay, style=_style(fill="#00ff00"))
        assert _pixel(rgb_red, *_px(30.0, -50.0)) == (255, 0, 0)
        assert _pixel(rgb_green, *_px(30.0, -50.0)) == (0, 255, 0)

    def test_labels_compose_over_the_fill(self, pack):
        backend = _CapturingBackend()
        overlay = MapOverlay(polygons=(
            OverlayPolygon(rings=(SQUARE,), label="plot"),))
        _result, _rgb = _render(pack, overlay, backend=backend)
        # the text backend was handed the FILLED buffer: labels
        # draw over the fill, not under it
        assert backend.seen is not None
        assert _pixel(backend.seen, *_px(30.0, -50.0)) == FILL
        # and the label spec carries the style's own colors:
        # near-white text, the style's halo as the contrast
        specs = [s for s in backend.labels if s.text == "Fill Town"]
        assert len(specs) == 1
        assert specs[0].color == (255, 255, 255)
        assert specs[0].halo == (0, 0, 0)

    def test_no_overlay_render_is_byte_identical(self, pack):
        result, _rgb = _render(pack, None)
        # sha256 of the pre-pass render of this exact fixture,
        # verified byte-for-byte against base 578f2ded (the
        # fill pass touches no code path a no-overlay render
        # walks: fills default to the overwrite path, the
        # overlay painter never runs)
        assert hashlib.sha256(result.image).hexdigest() == (
            "e0b7d84a90e3c4f986f26bcccd8dd5d4845c5e0e649bd9ec"
            "883e75a7c5f7e5c5")


# ── degradation: invalid geometry → outline + named note ─────


class TestDegradation:
    def test_bowtie_degrades_at_the_renderer(self, pack):
        overlay = MapOverlay(polygons=(
            OverlayPolygon(rings=(BOWTIE,), label="twisted"),))
        result, rgb = _render(pack, overlay)
        # not filled: the centre (the bowtie's crossing point)
        # carries the outline stroke, not the fill
        assert _pixel(rgb, *_px(30.0, -50.0)) != FILL
        assert "outline only" in result.receipt.detail
        assert "crosses itself" in result.receipt.detail
        assert "'twisted'" in result.receipt.detail

    def test_bowtie_degrades_at_conversion(self):
        overlay = Overlay("ov-t", "x", 0.0, (
            OverlayFeature("polygon", BOWTIE, "twisted"),))
        converted, notes = overlay_to_map_overlay(overlay)
        assert converted.polygons == ()
        assert len(converted.polylines) == 1
        ring = converted.polylines[0]
        assert ring[0] == ring[-1]  # the outline is closed
        assert len(notes) == 1
        assert "outline only" in notes[0]
        assert "crosses itself" in notes[0]
        assert "'twisted'" in notes[0]

    def test_hole_outside_degrades_at_conversion(self):
        far_hole = ((50.0, -56.0), (50.0, -44.0), (55.0, -44.0),
                    (55.0, -56.0))
        overlay = Overlay("ov-t", "x", 0.0, (
            OverlayFeature("polygon", SQUARE, "plot",
                           holes=(far_hole,)),))
        converted, notes = overlay_to_map_overlay(overlay)
        assert converted.polygons == ()
        assert len(notes) == 1
        assert "hole lies outside" in notes[0]

    def test_defects_named_by_the_checker(self):
        assert polygon_defects((SQUARE,)) == ()
        assert polygon_defects((SQUARE + SQUARE[:1],)) == ()
        assert "crosses itself" in polygon_defects((BOWTIE,))[0]
        collinear = ((20.0, -70.0), (20.0, -50.0), (20.0, -30.0))
        assert "zero area" in polygon_defects((collinear,))[0]
        assert "degenerate" in polygon_defects(
            (((20.0, -70.0), (20.0, -70.0), (30.0, -30.0)),))[0]
        assert polygon_defects((SQUARE, HOLE)) == ()


# ── the GeoJSON path: gate → features → seam → PNG ───────────


class TestGeoJsonEndToEnd:
    def _doc(self):
        return {"type": "FeatureCollection", "features": [
            {"type": "Feature",
             "geometry": {"type": "Polygon", "coordinates": [
                 [[-70, 20], [-30, 20], [-30, 40], [-70, 40],
                  [-70, 20]],
                 [[-56, 27], [-44, 27], [-44, 33], [-56, 33],
                  [-56, 27]]]},
             "properties": {"name": "catchment"}},
            {"type": "Feature",
             "geometry": {"type": "MultiPolygon", "coordinates": [
                 [[[-80, 20], [-60, 20], [-60, 30], [-80, 30],
                   [-80, 20]]],
                 [[[-40, 35], [-20, 35], [-20, 45], [-40, 45],
                   [-40, 35]]]]},
             "properties": {"name": "wards"}},
        ]}

    def test_geojson_polygons_fill_with_holes(self, pack):
        features, _notes = geojson_to_features(self._doc())
        polygons = [f for f in features if f.kind == "polygon"]
        assert len(polygons) == 3
        assert polygons[0].holes and len(polygons[0].holes) == 1
        overlay = Overlay("ov-gj", "gj", 0.0, tuple(features))
        converted, notes = overlay_to_map_overlay(overlay)
        assert len(converted.polygons) == 3
        assert notes == ()
        _result, rgb = _render(pack, MapOverlay(
            polygons=converted.polygons))
        # the holed polygon: hole centre unfilled, body filled
        assert _pixel(rgb, *_px(30.0, -50.0)) == WATER
        assert _pixel(rgb, *_px(25.0, -65.0)) == FILL
        # both MultiPolygon parts filled
        assert _pixel(rgb, *_px(25.0, -68.0)) == FILL
        assert _pixel(rgb, *_px(40.0, -28.0)) == FILL

    def test_store_round_trip_carries_holes(self, tmp_path):
        store = OverlayStore(home=str(tmp_path))
        store.create("catchment", [
            OverlayFeature("polygon", SQUARE, "plot",
                           holes=(HOLE,))])
        back = store.get("catchment")
        assert back is not None
        assert back.features[0].holes == (HOLE,)
        # and a hole-less feature's stored shape is unchanged
        store.create("plain", [OverlayFeature("marker",
                                              ((7.0, 5.0),), "m")])
        plain = store.get("plain")
        assert plain is not None
        assert "holes" not in plain.features[0].to_dict()

    def test_hole_validation_is_named(self):
        short_hole = OverlayFeature(
            "polygon", SQUARE, "plot",
            holes=(((27.0, -56.0), (27.0, -44.0)),))
        errors = validate_features([short_hole])
        assert any("hole" in e and "at least 3" in e
                   for e in errors)
        stray = OverlayFeature("polyline", SQUARE[:2], "line",
                               holes=(HOLE,))
        errors = validate_features([stray])
        assert any("holes belong" in e for e in errors)


# ── the style document ───────────────────────────────────────


class TestOverlayPaintData:
    def test_paint_validation_is_named(self):
        with pytest.raises(StyleError):
            OverlayPaint(fill="not-a-color")
        with pytest.raises(StyleError):
            OverlayPaint(fill_opacity=1.5)
        with pytest.raises(StyleError):
            OverlayPaint(stroke_width=-1.0)

    def test_style_round_trip_keeps_the_paint(self):
        clone = MapStyle.from_dict(DEFAULT_STYLE.to_dict())
        assert clone.overlay == DEFAULT_STYLE.overlay
        assert style_hash(clone) == style_hash(DEFAULT_STYLE)

    def test_old_document_without_overlay_gets_defaults(self):
        doc = DEFAULT_STYLE.to_dict()
        del doc["overlay"]
        style = MapStyle.from_dict(doc)
        assert style.overlay == OverlayPaint()

    def test_seam_structural_contracts(self):
        with pytest.raises(StyleError):
            OverlayPolygon(rings=())
        with pytest.raises(StyleError):
            OverlayPolygon(rings=(((1.0, 2.0), (3.0, 4.0)),))
        # the pre-fill construction shapes still work untouched
        overlay = MapOverlay(polylines=(((1.0, 2.0), (3.0, 4.0)),),
                             markers=((1.0, 2.0),))
        assert overlay.polygons == ()
