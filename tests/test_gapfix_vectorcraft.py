"""Lane 101 — VectorCraft port: an editable vector-art document model
over real SVG bytes (stdlib XML + PIL raster), boolean path ops,
machine-checked parity TOML, SHA-256 attribution DB, and the
``vectorcraft`` brain tool.

Failing-first: every test here fails on the base (no vector organ)
and passes only when the real engine exists. Fixtures are REAL SVG
files vendored under tests/fixtures/vectorcraft/ — hand-written by an
independent generator (never by the organ under test). A missing
fixture is a failure, never a silent skip.
"""

from __future__ import annotations

import io
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from nomorals.media.vector import (
    Affine,
    VectorDocument,
    VectorEditor,
    VectorError,
    load_attribution,
    load_parity,
    render_document,
    run_parity,
    verify_attribution,
)
from nomorals.media.vector import (
    Path as VectorPath,
)
from nomorals.media.vector.geom import (
    arc_to_cubic,
    parse_path_data,
    parse_transform,
)
from nomorals.media.vector.pathops import boolean, polygon_area
from nomorals.tools.registry import ToolRegistry

FIXTURES = Path(__file__).parent / "fixtures" / "vectorcraft"
REPO_ROOT = Path(__file__).parent.parent
PARITY_TOML = REPO_ROOT / "nomorals" / "media" / "vector-parity.toml"
ATTRIBUTION_TOML = REPO_ROOT / "nomorals" / "media" \
    / "vector-attribution.toml"

ALL_FIXTURES = (
    "simple.svg", "styled.svg", "paths.svg", "layers.svg",
    "empty.svg", "broken.svg", "unknown.svg", "bool.svg",
    "text.svg", "css.svg",
)


def _read(name: str) -> bytes:
    path = FIXTURES / name
    if not path.exists():
        raise AssertionError(f"fixture missing: {path} (never a skip)")
    return path.read_bytes()


def _doc(name: str) -> VectorDocument:
    return VectorDocument.from_bytes(_read(name))


class TestFixturesPresent(unittest.TestCase):
    def test_all_fixtures_exist(self) -> None:
        for name in ALL_FIXTURES:
            self.assertTrue((FIXTURES / name).exists(), name)


class TestGeom(unittest.TestCase):
    def test_path_data_all_commands(self) -> None:
        path = parse_path_data(
            "M 0 0 L 10 0 H 20 V 10 C 30 20, 40 0, 50 10 "
            "S 70 20, 80 10 Q 90 0, 100 10 T 120 10 "
            "A 10 10 0 0 1 140 10 Z")
        kinds = [seg.kind for seg in path.segments]
        self.assertEqual(kinds, ["M", "L", "L", "L", "C", "C", "Q",
                                 "Q", "A", "Z"])
        self.assertEqual(path.to_d().count("M"), 1)

    def test_path_data_relative_commands(self) -> None:
        path = parse_path_data("m 10 10 h 20 v 20 l -5 -5 z")
        # flatten() returns a list of contours, not a flat point
        # list: [0] is the first (only) contour.
        pts = path.flatten(0.5)[0]
        self.assertEqual(pts[0], (10.0, 10.0))
        self.assertEqual(pts[1], (30.0, 10.0))
        self.assertEqual(pts[2], (30.0, 30.0))

    def test_cubic_flatten_endpoints_and_bounds(self) -> None:
        path = parse_path_data("M 0 0 C 0 100, 100 100, 100 0")
        # flatten() returns a list of contours: [0] is the first
        # contour, [0][0] its first point.
        pts = path.flatten(0.5)[0]
        self.assertEqual(pts[0], (0.0, 0.0))
        self.assertAlmostEqual(pts[-1][0], 100.0, places=6)
        self.assertAlmostEqual(pts[-1][1], 0.0, places=6)
        lo, hi = path.bounds()
        self.assertAlmostEqual(hi[1], 75.0, delta=0.5)
        self.assertGreater(path.length(), 100.0)

    def test_arc_to_cubic_endpoints(self) -> None:
        curves = arc_to_cubic((0.0, 0.0), 50.0, 50.0, 0.0, 0, 1,
                              (100.0, 0.0))
        self.assertTrue(curves)
        self.assertAlmostEqual(curves[0][0][0], 0.0, places=6)
        self.assertAlmostEqual(curves[-1][-1][0], 100.0, places=6)
        self.assertAlmostEqual(curves[-1][-1][1], 0.0, places=6)

    def test_affine_compose_and_apply(self) -> None:
        move = Affine.translate(10.0, 20.0)
        spin = Affine.rotate(90.0)
        both = move * spin
        self.assertAlmostEqual(both.apply((1.0, 0.0))[0], 10.0,
                               places=6)
        self.assertAlmostEqual(both.apply((1.0, 0.0))[1], 21.0,
                               places=6)
        self.assertAlmostEqual(Affine.scale(2.0, 3.0).apply((4.0, 5.0)),
                               (8.0, 15.0))

    def test_parse_transform(self) -> None:
        mat = parse_transform("translate(10,20) rotate(90) scale(2)")
        x, y = mat.apply((1.0, 0.0))
        self.assertAlmostEqual(x, 10.0, places=6)
        self.assertAlmostEqual(y, 22.0, places=6)
        self.assertEqual(parse_transform("matrix(1 0 0 1 5 6)").apply(
            (0.0, 0.0)), (5.0, 6.0))

    def test_simplify_removes_collinear(self) -> None:
        path = parse_path_data("M 0 0 L 10 0 L 20 0 L 30 0 L 30 10")
        simple = path.simplify(0.1)
        # flatten() returns [contour]: the contour itself holds
        # the three surviving points (start, corner, end).
        pts = simple.flatten(0.5)[0]
        self.assertEqual(len(pts), 3)

    def test_reverse_flips_winding(self) -> None:
        path = parse_path_data("M 0 0 L 10 0 L 10 10 L 0 10 Z")
        area = polygon_area(path)
        rev = path.reverse()
        self.assertAlmostEqual(polygon_area(rev), -area, places=6)


class TestSvgModel(unittest.TestCase):
    def test_simple_parse(self) -> None:
        doc = _doc("simple.svg")
        self.assertEqual(doc.title, "Simple Fixture")
        self.assertEqual((doc.width, doc.height), (400.0, 300.0))
        nodes = doc.all_nodes()
        self.assertEqual(len(nodes), 3)
        rect = doc.find_node(nodes[0].node_id)
        self.assertEqual(rect.kind, "rect")
        self.assertEqual(rect.fill.color, "#ff0000")
        self.assertEqual(rect.stroke.color, "#000000")
        circle = [n for n in nodes if n.kind == "circle"][0]
        self.assertAlmostEqual(circle.opacity, 0.8)

    def test_styled_gradients(self) -> None:
        doc = _doc("styled.svg")
        self.assertEqual(len(doc.defs.gradients), 2)
        linear = doc.defs.gradients["grad1"]
        self.assertEqual(linear.kind, "linear")
        self.assertEqual(len(linear.stops), 2)
        self.assertEqual(linear.stops[0].color, "#ff0000")
        radial = doc.defs.gradients["grad2"]
        self.assertEqual(radial.kind, "radial")
        nodes = doc.all_nodes()
        self.assertEqual(len(nodes), 4)
        ellipse = [n for n in nodes if n.kind == "ellipse"][0]
        self.assertEqual(ellipse.stroke.dash, (6.0, 3.0))

    def test_paths_text_and_textpath(self) -> None:
        doc = _doc("paths.svg")
        nodes = doc.all_nodes()
        self.assertEqual(len(nodes), 7)
        texts = [n for n in nodes if n.kind == "text"]
        self.assertEqual(len(texts), 2)
        self.assertEqual(texts[0].text, "hello vector")
        self.assertEqual(texts[1].text_path, "curve")
        arc_path = [n for n in nodes if n.kind == "path"][2]
        # The arc path is a full circle built from two sweep-0
        # arcs, which wind clockwise in y-down coordinates, so the
        # SIGNED area is negative; the magnitude is pi * 40^2.
        self.assertAlmostEqual(abs(polygon_area(arc_path.path)),
                               math.pi * 40.0 * 40.0, delta=400.0)

    def test_layers_and_use(self) -> None:
        doc = _doc("layers.svg")
        self.assertEqual(len(doc.layers), 2)
        self.assertEqual(doc.layers[0].name, "Background")
        self.assertEqual(doc.layers[1].name, "Foreground")
        nodes = doc.all_nodes()
        self.assertEqual(len(nodes), 4)
        stamps = [n for n in nodes
                  if n.fill.color == "#123456"]
        self.assertEqual(len(stamps), 2)

    def test_css_class_styles(self) -> None:
        doc = _doc("css.svg")
        nodes = doc.all_nodes()
        self.assertEqual(nodes[0].fill.color, "#ff0000")
        self.assertEqual(nodes[0].stroke.color, "#000000")
        self.assertEqual(nodes[1].fill.color, "#0000ff")
        self.assertAlmostEqual(nodes[1].opacity, 0.5)

    def test_unknown_elements_preserved(self) -> None:
        doc = _doc("unknown.svg")
        self.assertEqual(len(doc.unknown), 2)
        data = doc.to_bytes()
        self.assertIn(b"custom:widget", data)
        self.assertIn(b"foreignObject", data)
        self.assertIn(b"html inside", data)
        again = VectorDocument.from_bytes(data)
        self.assertEqual(len(again.unknown), 2)

    def test_empty_and_broken(self) -> None:
        doc = _doc("empty.svg")
        self.assertEqual(doc.all_nodes(), [])
        self.assertEqual((doc.width, doc.height), (100.0, 100.0))
        with self.assertRaises(VectorError) as caught:
            _doc("broken.svg")
        self.assertEqual(caught.exception.kind, "parse")

    def test_round_trip_canonical(self) -> None:
        for name in ("simple.svg", "styled.svg", "paths.svg",
                     "layers.svg", "text.svg", "css.svg"):
            doc = _doc(name)
            again = VectorDocument.from_bytes(doc.to_bytes())
            report = doc.fidelity_report(again)
            self.assertEqual(report["untouched_changed"], 0, name)
            self.assertEqual(report["nodes_before"],
                             report["nodes_after"], name)

    def test_viewbox_preserved(self) -> None:
        doc = _doc("simple.svg")
        self.assertEqual(doc.view_box, (0.0, 0.0, 400.0, 300.0))


class TestEditor(unittest.TestCase):
    def test_add_rect_and_verbatim_clean_bytes(self) -> None:
        original = _read("simple.svg")
        editor = VectorEditor.from_bytes(original)
        self.assertEqual(editor.to_bytes(), original)
        editor.add_rect(200.0, 200.0, 50.0, 40.0, fill="#123456")
        self.assertEqual(len(editor.document.all_nodes()), 4)
        saved = editor.save()
        again = VectorDocument.from_bytes(saved)
        self.assertEqual(len(again.all_nodes()), 4)

    def test_set_fill_and_style(self) -> None:
        editor = VectorEditor.from_bytes(_read("simple.svg"))
        node = editor.document.all_nodes()[0]
        editor.set_fill(node.node_id, "#00ff00")
        editor.set_stroke(node.node_id, "#112233", width=4.0)
        editor.set_opacity(node.node_id, 0.5)
        saved = VectorDocument.from_bytes(editor.save())
        changed = saved.find_node(node.node_id)
        self.assertEqual(changed.fill.color, "#00ff00")
        self.assertEqual(changed.stroke.color, "#112233")
        self.assertEqual(changed.stroke.width, 4.0)
        self.assertAlmostEqual(changed.opacity, 0.5)

    def test_transform_moves_bounds(self) -> None:
        editor = VectorEditor.from_bytes(_read("simple.svg"))
        node = editor.document.all_nodes()[0]
        before = node.bounds()
        editor.transform(node.node_id, Affine.translate(100.0, 0.0))
        after = VectorDocument.from_bytes(
            editor.save()).find_node(node.node_id).bounds()
        self.assertAlmostEqual(after[0][0] - before[0][0], 100.0,
                               places=6)

    def test_group_ungroup(self) -> None:
        editor = VectorEditor.from_bytes(_read("simple.svg"))
        ids = [n.node_id for n in editor.document.all_nodes()[:2]]
        group_id = editor.group(ids, name="pair")
        doc = VectorDocument.from_bytes(editor.save())
        group = doc.find_node(group_id)
        self.assertEqual(group.kind, "group")
        self.assertEqual(len(group.children), 2)
        editor.ungroup(group_id)
        doc = VectorDocument.from_bytes(editor.save())
        self.assertEqual(len(doc.all_nodes()), 3)
        self.assertIsNone(doc.find_node(group_id))

    def test_remove_and_create_blank(self) -> None:
        editor = VectorEditor.from_bytes(_read("simple.svg"))
        node = editor.document.all_nodes()[0]
        editor.remove(node.node_id)
        self.assertEqual(len(editor.document.all_nodes()), 2)
        blank = VectorEditor.create(640.0, 480.0, title="Blank")
        blank.add_circle(320.0, 240.0, 50.0, fill="#ff0000")
        doc = VectorDocument.from_bytes(blank.save())
        self.assertEqual(doc.title, "Blank")
        self.assertEqual(len(doc.all_nodes()), 1)

    def test_fidelity_untouched_nodes(self) -> None:
        before = _doc("styled.svg")
        editor = VectorEditor.from_bytes(_read("styled.svg"))
        touched = editor.document.all_nodes()[0].node_id
        editor.set_fill(touched, "#123456")
        after = VectorDocument.from_bytes(editor.save())
        report = before.fidelity_report(after, touched={touched})
        self.assertEqual(report["untouched_changed"], 0)
        self.assertEqual(report["changed_ids"], 1)

    def test_add_text_and_path(self) -> None:
        editor = VectorEditor.create(200.0, 200.0)
        editor.add_text(10.0, 50.0, "hello", font_size=20.0)
        editor.add_path("M 10 100 L 190 100 L 100 180 Z",
                        fill="#00ff00")
        doc = VectorDocument.from_bytes(editor.save())
        self.assertEqual(len(doc.all_nodes()), 2)
        self.assertEqual(doc.all_nodes()[0].text, "hello")


class TestPathops(unittest.TestCase):
    def _rect_path(self, x: float, y: float, w: float, h: float
                   ) -> VectorPath:
        return parse_path_data(
            f"M {x} {y} L {x + w} {y} L {x + w} {y + h} "
            f"L {x} {y + h} Z")

    def test_union_area(self) -> None:
        a = self._rect_path(20, 20, 120, 120)
        b = self._rect_path(80, 80, 120, 120)
        out = boolean("union", a, b)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(abs(polygon_area(out[0])),
                               25200.0, delta=1.0)

    def test_intersection_area(self) -> None:
        a = self._rect_path(20, 20, 120, 120)
        b = self._rect_path(80, 80, 120, 120)
        out = boolean("intersection", a, b)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(abs(polygon_area(out[0])),
                               3600.0, delta=1.0)

    def test_difference_area(self) -> None:
        a = self._rect_path(20, 20, 120, 120)
        b = self._rect_path(80, 80, 120, 120)
        out = boolean("difference", a, b)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(abs(polygon_area(out[0])),
                               10800.0, delta=1.0)

    def test_xor_area(self) -> None:
        a = self._rect_path(20, 20, 120, 120)
        b = self._rect_path(80, 80, 120, 120)
        out = boolean("xor", a, b)
        self.assertAlmostEqual(
            sum(abs(polygon_area(p)) for p in out),
            21600.0, delta=1.0)

    def test_disjoint_union_and_contained_difference(self) -> None:
        a = self._rect_path(0, 0, 10, 10)
        b = self._rect_path(50, 50, 10, 10)
        out = boolean("union", a, b)
        self.assertAlmostEqual(
            sum(abs(polygon_area(p)) for p in out),
            200.0, delta=1.0)
        self.assertEqual(boolean("intersection", a, b), [])
        big = self._rect_path(0, 0, 100, 100)
        small = self._rect_path(20, 20, 10, 10)
        out = boolean("difference", big, small)
        self.assertAlmostEqual(
            sum(abs(polygon_area(p)) for p in out),
            9900.0, delta=1.0)

    def test_editor_boolean(self) -> None:
        editor = VectorEditor.from_bytes(_read("bool.svg"))
        nodes = editor.document.all_nodes()
        a = [n for n in nodes if n.node_id == "a"][0]
        b = [n for n in nodes if n.node_id == "b"][0]
        editor.apply_boolean("union", a.node_id, b.node_id)
        doc = VectorDocument.from_bytes(editor.save())
        self.assertEqual(len(doc.all_nodes()), 2)
        result = [n for n in doc.all_nodes() if n.kind == "path"][0]
        self.assertAlmostEqual(
            abs(polygon_area(result.path)),
            25200.0, delta=1.0)


class TestRender(unittest.TestCase):
    def test_render_png_pixels(self) -> None:
        png = render_document(_doc("simple.svg"))
        self.assertTrue(png.startswith(b"\x89PNG"))
        image = Image.open(io.BytesIO(png))
        self.assertEqual(image.size, (400, 300))
        self.assertEqual(image.mode, "RGBA")
        self.assertEqual(image.getpixel((30, 30)), (255, 0, 0, 255))
        self.assertEqual(image.getpixel((250, 100)), (0, 0, 255, 204))
        self.assertEqual(image.getpixel((390, 290)), (0, 0, 0, 0))

    def test_render_gradient_direction(self) -> None:
        png = render_document(_doc("styled.svg"))
        image = Image.open(io.BytesIO(png))
        left = image.getpixel((15, 15))
        right = image.getpixel((105, 15))
        self.assertGreater(left[0], left[2])
        self.assertGreater(right[2], right[0])

    def test_render_scale_and_text(self) -> None:
        png = render_document(_doc("text.svg"), scale=2.0)
        image = Image.open(io.BytesIO(png))
        self.assertEqual(image.size, (600, 300))
        self.assertTrue(any(
            image.getpixel((x, y))[3] > 0
            for x in range(10, 200) for y in range(20, 45)))

    def test_render_text_on_path(self) -> None:
        png = render_document(_doc("paths.svg"))
        image = Image.open(io.BytesIO(png))
        self.assertTrue(any(
            image.getpixel((x, y))[3] > 0
            for x in range(20, 380) for y in range(180, 260)))


class TestParityAttribution(unittest.TestCase):
    def test_parity_passes(self) -> None:
        result = run_parity(PARITY_TOML)
        self.assertEqual(result["failed"], 0, str(result))
        self.assertEqual(result["passed"], 10)
        self.assertEqual(len(result["cases"]), 10)

    def test_parity_missing_fixture_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            toml = Path(tmp) / "vector-parity.toml"
            toml.write_text(
                "schema_version = 1\n\n"
                "[[cases]]\n"
                "id = \"gone\"\n"
                "fixture = \"no-such-file.svg\"\n"
                "ops = [\"inspect\"]\n"
                "[cases.expect]\n"
                "nodes = 1\n", encoding="utf-8")
            result = run_parity(path=toml)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["cases"][0]["status"], "fail")

    def test_attribution_verifies(self) -> None:
        result = verify_attribution(ATTRIBUTION_TOML)
        self.assertEqual(result["failed"], 0, str(result))
        self.assertGreaterEqual(result["checked"], 15)
        self.assertEqual(result["skipped"], 0)

    def test_parity_and_attribution_toml_load(self) -> None:
        data = load_parity(PARITY_TOML)
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(len(data["cases"]), 10)
        attrs = load_attribution(ATTRIBUTION_TOML)
        self.assertTrue(attrs["assets"])


class TestTool(unittest.TestCase):
    def _registry(self) -> ToolRegistry:
        registry = ToolRegistry(context=SimpleNamespace(
            settings=SimpleNamespace(), home=Path(tempfile.mkdtemp())))
        registry.register_builtins()
        return registry

    def _call(self, registry: ToolRegistry, **kwargs) -> dict:
        outcome = registry.call("vectorcraft", **kwargs)
        if hasattr(outcome, "value"):
            return outcome.value
        return outcome

    def test_vectorcraft_registered(self) -> None:
        registry = self._registry()
        spec = registry.get("vectorcraft")
        self.assertIsNotNone(spec)
        self.assertIn("inspect", spec.description)
        # 721 after the vectorcraft port, 722 after the gridcraft
        # port (lane 102) added the gridcraft tool, 723 after the
        # deckcraft port (lane 103) added the deckcraft tool.
        self.assertEqual(len(registry._tools), 723)

    def test_tool_inspect_and_render(self) -> None:
        registry = self._registry()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.png"
            value = self._call(registry, action="inspect", sources=[
                str(FIXTURES / "simple.svg")])
            self.assertEqual(value["nodes"], 3)
            self.assertIn("reply", value)
            rendered = self._call(registry, action="render", sources=[
                str(FIXTURES / "simple.svg")], output=str(out))
            self.assertTrue(rendered["ok"])
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 100)

    def test_tool_unknown_action(self) -> None:
        registry = self._registry()
        outcome = registry.call("vectorcraft", action="nope")
        self.assertFalse(outcome.ok)
        self.assertIn("unknown", str(outcome.error).lower())


if __name__ == "__main__":
    unittest.main()
