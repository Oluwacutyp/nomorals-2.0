"""EffectCraft port — Devon's own motion graphics compositor (lane 93).

Failing-first: the ``nomorals.media.effects`` organ does not exist on
base. Compositions, layers, keyframes, a real raster effect registry
(ported from EffectCraft's effect families), safe value expressions
(the ``expr`` crate's surface, without its JS runtime — per
PARITY.md that runtime was deliberately not ported), and a 3D camera
that also registers as a scene3d rung, completing the motion_studio
camera3d group that refuses with no rung registered.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from nomorals.media.edit_engine import scene3d
from nomorals.media.effects import camera as fx_camera
from nomorals.media.effects import effects as fx_effects
from nomorals.media.effects import expressions as fx_expr
from nomorals.media.effects.composition import (
    Composition,
    Layer,
    render_composition,
    render_composition_video,
)
from nomorals.media.effects.keyframes import PropertyTrack


def _arr(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float64)


class CompositionTest(unittest.TestCase):
    def test_solid_layer_color(self) -> None:
        comp = Composition(160, 90, fps=12.0, duration=1.0)
        comp.add_layer(Layer("solid", color=(10, 200, 30)))
        img = render_composition(comp, 0.0)
        self.assertEqual(img.size, (160, 90))
        px = _arr(img)[45, 80]
        self.assertTrue((px == np.array([10, 200, 30])).all())

    def test_top_layer_opacity_blends(self) -> None:
        comp = Composition(160, 90, fps=12.0, duration=1.0)
        comp.add_layer(Layer("solid", color=(200, 0, 0)))
        comp.add_layer(Layer("solid", color=(0, 0, 200), opacity=0.5))
        px = _arr(render_composition(comp, 0.0))[45, 80]
        self.assertAlmostEqual(px[0], 100.0, delta=2)
        self.assertAlmostEqual(px[2], 100.0, delta=2)

    def test_keyframed_transform_moves_layer(self) -> None:
        comp = Composition(160, 90, fps=12.0, duration=2.0)
        layer = Layer("solid", color=(255, 255, 0), size=(20, 20))
        layer.transform["position_x"] = PropertyTrack("position_x")
        layer.transform["position_x"].add(0.0, 10.0)
        layer.transform["position_x"].add(2.0, 130.0)
        comp.add_layer(layer)
        early = _arr(render_composition(comp, 0.0))
        late = _arr(render_composition(comp, 2.0))
        self.assertFalse(np.array_equal(early, late))
        # the yellow square sits left early, right late
        self.assertGreater(early[45, 15, 0], 200)
        self.assertGreater(late[45, 145, 0], 200)

    def test_text_layer_renders(self) -> None:
        comp = Composition(160, 90, fps=12.0, duration=1.0)
        comp.add_layer(Layer("text", text="Hi", color=(255, 255, 255),
                             size=(160, 90)))
        img = render_composition(comp, 0.0)
        self.assertFalse(np.array_equal(_arr(img), np.zeros((90, 160, 3))))

    def test_blend_multiply_darkens(self) -> None:
        comp = Composition(160, 90, fps=12.0, duration=1.0)
        comp.add_layer(Layer("solid", color=(200, 200, 200)))
        comp.add_layer(Layer("solid", color=(100, 100, 100),
                             blend="multiply"))
        px = _arr(render_composition(comp, 0.0))[45, 80]
        self.assertLess(px[0], 100)

    def test_render_video_is_real_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            comp = Composition(160, 90, fps=12.0, duration=1.0)
            comp.add_layer(Layer("solid", color=(30, 120, 200)))
            out = Path(td) / "comp.mp4"
            report = render_composition_video(comp, out, workdir=Path(td))
            self.assertTrue(report["ok"], report)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 1000)


class KeyframeTest(unittest.TestCase):
    def test_sample_midpoint_smooth(self) -> None:
        track = PropertyTrack("opacity")
        track.add(0.0, 0.0)
        track.add(2.0, 100.0)
        self.assertAlmostEqual(track.sample(1.0), 50.0, places=1)
        self.assertAlmostEqual(track.sample(0.0), 0.0)
        self.assertAlmostEqual(track.sample(2.0), 100.0)

    def test_sample_clamps_outside(self) -> None:
        track = PropertyTrack("x")
        track.add(1.0, 5.0)
        track.add(2.0, 9.0)
        self.assertEqual(track.sample(0.0), 5.0)
        self.assertEqual(track.sample(9.0), 9.0)

    def test_expression_drives_value(self) -> None:
        track = PropertyTrack("x")
        track.set_expression("time * 10")
        self.assertAlmostEqual(track.sample(1.5), 15.0)


class ExpressionTest(unittest.TestCase):
    def test_arithmetic_and_time(self) -> None:
        self.assertAlmostEqual(fx_expr.evaluate("time * 2 + 1", {"time": 3.0}), 7.0)
        self.assertAlmostEqual(fx_expr.evaluate("sin(time)", {"time": 0.0}), 0.0)

    def test_wiggle_is_deterministic(self) -> None:
        ctx = {"time": 1.25}
        a = fx_expr.evaluate("wiggle(2, 10)", ctx)
        b = fx_expr.evaluate("wiggle(2, 10)", ctx)
        self.assertEqual(a, b)
        self.assertTrue(-10.0 <= a <= 10.0)

    def test_unsafe_expression_refused(self) -> None:
        with self.assertRaises(fx_expr.ExpressionError):
            fx_expr.evaluate("__import__('os').system('id')", {"time": 0.0})
        with self.assertRaises(fx_expr.ExpressionError):
            fx_expr.evaluate("open('/etc/passwd').read()", {"time": 0.0})
        with self.assertRaises(fx_expr.ExpressionError):
            fx_expr.evaluate("no_such_fn(time)", {"time": 0.0})


class RasterEffectTest(unittest.TestCase):
    def _img(self) -> Image.Image:
        arr = np.zeros((90, 160, 3), dtype=np.uint8)
        arr[:, :80] = (200, 30, 30)
        arr[:, 80:] = (30, 30, 200)
        return Image.fromarray(arr)

    def test_registry_lists_real_effects(self) -> None:
        names = fx_effects.list_effects()
        self.assertGreaterEqual(len(names), 15)
        for name in ("gaussian_blur", "brightness", "vignette",
                     "chroma_key", "drop_shadow"):
            self.assertIn(name, names)

    def test_every_effect_transforms_and_is_deterministic(self) -> None:
        for name in fx_effects.list_effects():
            with self.subTest(effect=name):
                out1 = fx_effects.apply_effect(name, self._img(), {})
                out2 = fx_effects.apply_effect(name, self._img(), {})
                self.assertEqual(out1.size, self._img().size)
                self.assertTrue(
                    np.array_equal(np.asarray(out1), np.asarray(out2)),
                    f"{name} is not deterministic")

    def test_brightness_really_brightens(self) -> None:
        out = fx_effects.apply_effect("brightness", self._img(),
                                      {"amount": 1.5})
        self.assertGreater(float(_arr(out).mean()),
                           float(_arr(self._img()).mean()))

    def test_unknown_effect_and_bad_params_raise(self) -> None:
        with self.assertRaises(fx_effects.EffectError):
            fx_effects.apply_effect("no_such_effect", self._img(), {})
        with self.assertRaises(fx_effects.EffectError):
            fx_effects.apply_effect("gaussian_blur", self._img(),
                                    {"radius": -5.0})


class CameraTest(unittest.TestCase):
    def test_perspective_moves_corners(self) -> None:
        img = Image.new("RGB", (160, 90), (40, 40, 40))
        cam = fx_camera.Camera3D(position=(0.0, 0.0, 500.0),
                                 rotation=(0.0, 25.0, 0.0), fov_y=40.0)
        out = fx_camera.apply_camera(img, cam)
        self.assertEqual(out.size, img.size)
        self.assertFalse(np.array_equal(np.asarray(out), np.asarray(img)))

    def test_camera_rung_registers_in_scene3d(self) -> None:
        scene3d.reset_renderers()
        try:
            fx_camera.register_camera_rung()
            self.assertIn("effects-perspective-camera",
                          scene3d.registered_renderers())
            probe = fx_camera.PerspectiveCameraRenderer().probe()
            self.assertTrue(probe.available)
        finally:
            scene3d.reset_renderers()


class EffectToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _registry(self):
        from nomorals.tools.registry import ToolRegistry
        reg = ToolRegistry(SimpleNamespace(
            settings=SimpleNamespace(home_path=self.tmp)))
        reg.register_builtins()
        return reg

    def test_effectcraft_tool_lists_effects(self) -> None:
        reg = self._registry()
        tool = reg._tools["effectcraft"].fn
        listed = tool(action="list_effects")
        self.assertTrue(listed["ok"])
        self.assertGreaterEqual(len(listed["effects"]), 15)

    def test_effectcraft_tool_renders_composition(self) -> None:
        reg = self._registry()
        tool = reg._tools["effectcraft"].fn
        created = tool(action="create", name="comp1", width=160, height=90,
                       fps=12, duration=1.0)
        self.assertTrue(created["ok"], created)
        added = tool(action="add_layer", name="comp1", kind="solid",
                     color=[30, 120, 200])
        self.assertTrue(added["ok"], added)
        rendered = tool(action="render", name="comp1",
                        out=str(self.tmp / "comp1.mp4"))
        self.assertTrue(rendered["ok"], rendered)
        self.assertTrue((self.tmp / "comp1.mp4").exists())


if __name__ == "__main__":
    unittest.main()
