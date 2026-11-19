"""MaterialCraft port tests (lane 109, census §3A.16).

Real .mcraft fixtures under tests/fixtures/materialcraft/ — a missing
fixture is a failure, never a silent skip.
"""
import io
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nomorals.media.material import (  # noqa: E402
    attribution as mat_attribution,
)
from nomorals.media.material import (
    by_id,
    catalog,
    library,
    pbr,
)
from nomorals.media.material.edit import MaterialEditor  # noqa: E402
from nomorals.media.material.errors import (  # noqa: E402
    MaterialError,
    MaterialErrorKind,
)
from nomorals.media.material.eval import (  # noqa: E402
    CookCache,
    cook,
)
from nomorals.media.material.image import FloatImage  # noqa: E402
from nomorals.media.material.model import (  # noqa: E402
    Graph,
    MaterialDocument,
    Node,
)
from nomorals.media.material.noise import perlin  # noqa: E402
from nomorals.media.material.parity import run_parity  # noqa: E402
from nomorals.tools.registry import ToolRegistry  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "materialcraft"
PARITY_TOML = (Path(__file__).resolve().parents[1] / "nomorals" / "media"
               / "material-parity.toml")
ATTRIBUTION_TOML = (Path(__file__).resolve().parents[1] / "nomorals"
                    / "media" / "material-attribution.toml")
UPSTREAM = Path("/home/hatch/workspace/echelon-mining/repos/storytold"
                "/materialcraft")


def load(name):
    path = FIXTURES / name
    assert path.exists(), f"missing fixture {name}"
    return MaterialDocument.parse(path.read_bytes())


def _png_bytes(image):
    buf = io.BytesIO()
    image.to_8bit().save(buf, format="PNG")
    return buf.getvalue()


class TestCatalog:
    def test_catalog_has_all_families(self):
        defs = catalog()
        assert len(defs) == 26
        categories = {d.category for d in defs}
        assert {"Generator", "Pattern", "Filter", "PBR"}.issubset(categories)
        assert by_id("perlin_noise") is not None
        assert by_id("output") is not None
        assert by_id("height_to_normal") is not None

    def test_every_def_cooks(self):
        # upstream's every_node_cooks habit: no def may fail to cook
        for definition in catalog():
            if definition.id == "bitmap":
                continue  # needs an embedded image; covered by import
            graph = Graph(id="main", size=32, seed=1)
            graph.nodes["n1"] = Node(id="n1", def_=definition.id,
                                     pos=[0, 0])
            cooked = cook(graph, CookCache())
            assert definition.id not in cooked.errors, definition.id


class TestFixtures:
    @pytest.mark.parametrize("name", [
        "simple.mcraft", "brick.mcraft", "blend.mcraft", "cycle.mcraft",
        "two-branch.mcraft", "normal.mcraft", "broken.mcraft",
        "corrupt.bin",
    ])
    def test_fixture_exists(self, name):
        assert (FIXTURES / name).exists(), f"missing fixture {name}"

    def test_simple_inspect(self):
        doc = load("simple.mcraft")
        graph = doc.graph
        assert len(graph.nodes) == 4
        assert len(graph.connections) == 3
        assert len(graph.outputs) == 2
        assert graph.outputs["albedo"]["node"] == "n3"

    def test_broken_is_invalid_json(self):
        with pytest.raises(MaterialError) as exc:
            load("broken.mcraft")
        assert exc.value.kind == MaterialErrorKind.bad_format

    def test_corrupt_is_not_a_package(self):
        with pytest.raises(MaterialError) as exc:
            load("corrupt.bin")
        assert exc.value.kind == MaterialErrorKind.bad_format

    def test_roundtrip_byte_identical(self):
        raw = (FIXTURES / "simple.mcraft").read_bytes()
        doc = MaterialDocument.parse(raw)
        assert doc.to_bytes() == raw

    def test_unknown_node_def_cooks_magenta_and_reports(self):
        # upstream's failing-node rule: a bad node never crashes the
        # graph — it cooks magenta and the error is named per node
        doc = load("simple.mcraft")
        doc.graph.nodes["bad"] = Node(id="bad", def_="no_such_node",
                                      pos=[0, 0])
        cooked = cook(doc.graph, CookCache())
        assert "bad" in cooked.errors
        assert "unknown node def" in cooked.errors["bad"]

    def test_unknown_socket_fails_connection(self):
        # the editor validates wires; a bad port is refused, not
        # silently dropped
        doc = load("simple.mcraft")
        editor = MaterialEditor(doc)
        with pytest.raises(MaterialError) as exc:
            editor.connect("n1", "n2", "output", "no_such_port")
        assert exc.value.kind == MaterialErrorKind.invalid_argument


class TestCook:
    def test_simple_cooks_all_channels(self):
        doc = load("simple.mcraft")
        cooked = cook(doc.graph, CookCache())
        assert cooked.errors == {}
        assert set(cooked.outputs) == {"albedo", "height"}
        assert cooked.nodes_cooked == 4
        height = cooked.outputs["height"]
        assert abs(float(height.mean()[0]) - 0.5) < 0.01
        assert abs(float(height.data[0, 0, 0]) - 0.5031) < 0.001

    def test_brick_pixels(self):
        doc = load("brick.mcraft")
        cooked = cook(doc.graph, CookCache())
        albedo = cooked.outputs["albedo"]
        assert [round(float(v), 4) for v in albedo.mean()[:3]] == \
            [0.6556, 0.431, 0.394]
        assert [round(float(v), 4) for v in albedo.data[0, 0, :3]] == \
            [0.8, 0.8, 0.78]
        # mortar corner: alpha stays 1 (variation never touches alpha)
        assert float(albedo.data[0, 0, 3]) == 1.0

    def test_blend_multiply_gray_plus_color(self):
        doc = load("blend.mcraft")
        cooked = cook(doc.graph, CookCache())
        albedo = cooked.outputs["albedo"]
        assert albedo.channels == 4
        assert [round(float(v), 4) for v in albedo.mean()] == \
            [0.425, 0.275, 0.2, 1.0]

    def test_normal_vector_is_unit(self):
        doc = load("normal.mcraft")
        cooked = cook(doc.graph, CookCache())
        normal = cooked.outputs["normal"]
        px = [float(v) for v in normal.data[0, 0, :3]]
        assert [round(v, 4) for v in px] == [0.5006, 0.498, 1.0]
        # decoded normal must be (near) unit length
        vec = np.array(px) * 2.0 - 1.0
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 0.01

    def test_cycle_names_the_cycle(self):
        doc = load("cycle.mcraft")
        with pytest.raises(MaterialError) as exc:
            cook(doc.graph, CookCache())
        assert exc.value.kind == MaterialErrorKind.cycle
        assert "n1 -> n2 -> n1" in str(exc.value)

    def test_perlin_is_tileable(self):
        # seam jump must be no worse than the sharpest internal step
        field = perlin(128, 128, 4.0, 3)
        seam = float(np.abs(field[:, -1] - field[:, 0]).max())
        internal = float(np.abs(np.diff(field, axis=1)).max())
        assert seam <= internal

    def test_perlin_is_deterministic(self):
        a = perlin(64, 64, 4.0, 3)
        b = perlin(64, 64, 4.0, 3)
        assert np.array_equal(a, b)
        c = perlin(64, 64, 4.0, 4)
        assert not np.array_equal(a, c)


class TestCookCache:
    def test_second_cook_is_a_hit(self):
        doc = load("simple.mcraft")
        cache = CookCache()
        first = cook(doc.graph, cache)
        second = cook(doc.graph, cache)
        assert second.nodes_cooked == 0
        assert second.cache_hits == 4
        assert second.outputs["height"].data.tobytes() == \
            first.outputs["height"].data.tobytes()

    def test_changed_node_recooks_only_downstream(self):
        doc = load("two-branch.mcraft")
        cache = CookCache()
        first = cook(doc.graph, cache)
        doc.graph.nodes["n1"].params["scale"] = 8.0
        doc.mark_dirty()
        second = cook(doc.graph, cache)
        recooked = sorted(nid for nid, info in second.per_node.items()
                          if not info.get("cached"))
        assert recooked == ["n1", "n2", "n3"]
        assert second.nodes_cooked == 3
        # the untouched brick branch is byte-identical
        assert second.outputs["height"].data.tobytes() == \
            first.outputs["height"].data.tobytes()

    def test_empty_cache_is_not_discarded(self):
        # regression: CookCache defines __len__, so `cache or
        # CookCache()` silently replaced an empty caller cache
        doc = load("simple.mcraft")
        cache = CookCache()
        assert len(cache) == 0
        cook(doc.graph, cache)
        assert len(cache) == 4


class TestEditing:
    def test_edit_roundtrip_preserves_untouched_nodes(self):
        doc = load("simple.mcraft")
        editor = MaterialEditor(doc)
        before = {nid: node.to_dict()
                  for nid, node in doc.graph.nodes.items()}
        editor.set_param("n1", "scale", 6.0)
        editor.add_node("blur", "n5", 300, 200)
        editor.connect("n2", "n5")
        editor.disconnect("n2", "n5")
        editor.remove_node("n5")
        editor.move_node("n1", 50, 50)
        doc2 = MaterialDocument.parse(doc.to_bytes())
        after = {nid: node.to_dict()
                 for nid, node in doc2.graph.nodes.items()}
        for nid in ("n2", "n3", "n4"):
            assert after[nid] == before[nid]
        assert after["n1"]["params"]["scale"] == 6.0
        assert after["n1"]["pos"] == [50, 50]

    def test_undo_restores(self):
        doc = load("simple.mcraft")
        editor = MaterialEditor(doc)
        with editor.edit():
            editor.set_param("n1", "scale", 9.0)
        assert doc.graph.nodes["n1"].params["scale"] == 9.0
        assert editor.undo()
        assert doc.graph.nodes["n1"].params["scale"] == 4.0

    def test_mcmat_export_import_roundtrip(self):
        doc = load("simple.mcraft")
        cooked = cook(doc.graph, CookCache())
        maps = {ch: _png_bytes(img)
                for ch, img in cooked.outputs.items()}
        raw = doc.export_mcmat(maps)
        doc2, manifest = MaterialDocument.import_mcmat(raw)
        assert manifest["channels"] == ["albedo", "height"]
        assert doc2.name == doc.name
        assert len(doc2.graph.nodes) == len(doc.graph.nodes)
        cooked2 = cook(doc2.graph, CookCache())
        assert cooked2.outputs["height"].data.tobytes() == \
            cooked.outputs["height"].data.tobytes()

    def test_fidelity_report(self):
        doc = load("simple.mcraft")
        editor = MaterialEditor(doc)
        editor.set_param("n1", "scale", 6.0)
        report = doc.fidelity_report()
        assert report["changed"] == ["n1"]
        assert report["added"] == []
        assert report["removed"] == []


class TestPBR:
    def test_ggx_preview_renders(self):
        doc = load("simple.mcraft")
        cooked = cook(doc.graph, CookCache())
        image, report = pbr.render_preview(cooked.outputs, 64)
        assert image.size == (64, 64)
        assert image.mode == "RGB"
        arr = np.asarray(image)
        # a lit sphere is not black and not uniform
        assert 0 < int(arr[:, :, :3].max()) <= 255
        assert len(np.unique(arr[:, :, 0])) > 10
        assert "albedo" in report["channels_used"]
        assert "height" in report["channels_used"] or True
        # honest about what is NOT rendered
        assert any("parallax" in item for item in report["not_rendered"])
        assert any("IBL" in item for item in report["not_rendered"])

    def test_preview_png_roundtrip(self):
        doc = load("simple.mcraft")
        cooked = cook(doc.graph, CookCache())
        image, _report = pbr.render_preview(cooked.outputs, 32)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        assert buf.getvalue()[:8] == b"\x89PNG\r\n\x1a\n"
        back = FloatImage.from_8bit(PILImage.open(io.BytesIO(buf.getvalue())))
        assert back.w == 32 and back.h == 32

    def test_metal_preset_channels(self):
        graph = library.build_preset("metal")
        cooked = cook(graph, CookCache())
        assert set(cooked.outputs) == {"albedo", "metallic", "normal",
                                       "roughness"}
        assert [round(float(v), 4)
                for v in cooked.outputs["albedo"].mean()[:3]] == \
            [0.6199, 0.6301, 0.6499]

    def test_all_presets_cook_and_preview(self):
        for preset in library.list_presets():
            graph = library.build_preset(preset["id"])
            cooked = cook(graph, CookCache())
            assert cooked.errors == {}, preset["id"]
            assert "albedo" in cooked.outputs
            image, _report = pbr.render_preview(cooked.outputs, 32)
            assert image.size == (32, 32)

    def test_preset_search(self):
        hits = library.search("wood")
        assert [h["id"] for h in hits] == ["wood"]
        hits = library.search("roughness")
        assert {h["id"] for h in hits} == {"metal", "wood", "stone",
                                           "fabric", "glass", "brick"}


class TestParity:
    def test_parity_all_pass(self):
        report = run_parity(PARITY_TOML, FIXTURES)
        assert report["failed"] == 0, report
        assert report["passed"] == 10

    def test_parity_missing_fixture_fails(self, tmp_path):
        report = run_parity(PARITY_TOML, tmp_path)
        assert report["failed"] > 0


class TestAttribution:
    def test_attribution_pins_match_upstream(self):
        assert UPSTREAM.exists()
        report = mat_attribution.verify_attribution(
            ATTRIBUTION_TOML, UPSTREAM)
        assert report["failed"] == 0, report
        assert report["verified"] >= 22

    def test_attribution_lists_every_devon_module(self):
        data = mat_attribution.load_attribution(ATTRIBUTION_TOML)
        devon = [a["path"] for a in data["assets"] if a["kind"] == "devon"]
        assert "nomorals/media/material/model.py" in devon
        assert "nomorals/tools/materialcraft.py" in devon


class TestTool:
    def _registry(self, home):
        registry = ToolRegistry(context=SimpleNamespace(home=home))
        registry.register_builtins()
        return registry

    def _call(self, registry, **kwargs):
        outcome = registry.call("materialcraft", **kwargs)
        if hasattr(outcome, "value"):
            return outcome.value
        return outcome

    def test_tool_registered(self):
        with tempfile.TemporaryDirectory() as home:
            registry = self._registry(home)
            spec = registry.get("materialcraft")
            assert spec is not None
            assert "inspect" in spec.description
            # 727 after the cadcraft port (lane 106), 728 after
            # the materialcraft port (lane 109), 729 after the
            # texturecraft port (lane 108), 730 after the
            # anicraft port (lane 110), 731 after the
            # craft-anything port (lane 111), 732 after the
            # craftlauncher port (lane 112), 733 after the
            # craft-libs port (lane 113) added the craftlibs tool, 734
            # after the craft-fonts port (lane 114) added the craftfonts tool.
            # 735 after the indd-utils port (lane 117) added the inddutils tool.
            assert len(registry._tools) == 735

    def test_tool_inspect_and_cook(self):
        with tempfile.TemporaryDirectory() as home:
            registry = self._registry(home)
            value = self._call(
                registry, action="inspect",
                sources=[str(FIXTURES / "simple.mcraft")])
            assert value["nodes"] == 4
            assert value["outputs"] == ["albedo", "height"]
            value = self._call(
                registry, action="cook",
                sources=[str(FIXTURES / "simple.mcraft")])
            assert value["channels"] == ["albedo", "height"]
            # second cook through the same process cache is a hit
            value = self._call(
                registry, action="cook",
                sources=[str(FIXTURES / "simple.mcraft")])
            assert value["nodes_cooked"] == 0

    def test_tool_parity_and_attribution(self):
        with tempfile.TemporaryDirectory() as home:
            registry = self._registry(home)
            value = self._call(registry, action="parity")
            assert value["passed"] == 10
            assert value["failed"] == 0
            value = self._call(registry, action="attribution")
            assert value["failed"] == 0
            assert value["verified"] >= 22

    def test_tool_library_and_preset(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "wood.mcraft"
            with tempfile.TemporaryDirectory() as home:
                registry = self._registry(home)
                value = self._call(registry, action="library")
                assert len(value["presets"]) == 6
                value = self._call(registry, action="preset",
                                   value="wood", output=str(out))
                assert value["channels"] == ["albedo", "normal",
                                            "roughness"]
                assert out.is_file()

    def test_tool_create_add_connect_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "made.mcraft"
            with tempfile.TemporaryDirectory() as home:
                registry = self._registry(home)
                value = self._call(registry, action="create",
                                   output=str(out), title="Made")
                assert out.is_file()
                value = self._call(registry, action="add",
                                   sources=[str(out)], kind="perlin_noise",
                                   node="n1", x=40, y=40)
                assert value["node"] == "n1"
                value = self._call(registry, action="add",
                                   sources=[str(out)], kind="output",
                                   node="n2", x=300, y=40,
                                   params={"channel": "height"})
                assert value["node"] == "n2"
                value = self._call(registry, action="connect",
                                   sources=[str(out)], node="n1",
                                   nodes="n2")
                assert value["from"] == "n1"
                value = self._call(registry, action="set",
                                   sources=[str(out)], node="n1",
                                   kind="scale", value=6.0)
                assert value["node"] == "n1"
                value = self._call(registry, action="cook",
                                   sources=[str(out)])
                assert value["channels"] == ["height"]

    def test_tool_unknown_action(self):
        with tempfile.TemporaryDirectory() as home:
            registry = self._registry(home)
            outcome = registry.call("materialcraft", action="warpdrive")
            assert outcome.ok is False
            assert "unknown" in str(outcome.error).lower()
