"""LayerStack tests: Photoshop-style layer stack for media_edit."""

import json
import unittest
from pathlib import Path

from PIL import Image

from nomorals.media_edit import layers
from nomorals.media_edit.images import MediaEditError
from nomorals.media_edit.layers import LayerStack


def _img(color, size=(32, 24)):
    return Image.new("RGB", size, color)


class TestBase(unittest.TestCase):
    def test_blank_canvas_base(self):
        st = LayerStack((64, 48))
        self.assertEqual(st.canvas_size, (64, 48))
        out = st.flatten()
        self.assertEqual(out.size, (64, 48))
        self.assertEqual(out.getpixel((0, 0))[:3], (255, 255, 255))

    def test_blank_canvas_custom_bg(self):
        st = LayerStack((16, 16), bg="red")
        out = st.flatten()
        self.assertEqual(out.getpixel((0, 0))[:3], (255, 0, 0))

    def test_pil_image_base(self):
        st = LayerStack(_img("blue"))
        out = st.flatten()
        self.assertEqual(out.getpixel((3, 3))[:3], (0, 0, 255))

    def test_path_base(self):
        p = Path("/tmp/layers_test_base.png")
        _img("green").save(p)
        try:
            st = LayerStack(p)
            out = st.flatten()
            self.assertEqual(out.getpixel((3, 3))[:3], (0, 128, 0))
        finally:
            p.unlink(missing_ok=True)

    def test_bad_base_rejected(self):
        with self.assertRaises(MediaEditError):
            LayerStack(42)
        with self.assertRaises(MediaEditError):
            LayerStack("/nonexistent/xyz.png")


class TestAddEdit(unittest.TestCase):
    def setUp(self):
        self.st = LayerStack((64, 48))

    def test_add_image_returns_id(self):
        lid = self.st.add_image(_img("red"))
        self.assertEqual(len(self.st), 1)
        info = self.st.layer_info()[0]
        self.assertEqual(info["id"], lid)
        self.assertEqual(info["type"], "image")
        self.assertEqual(info["size"], [32, 24])

    def test_add_text_shape_solid(self):
        t = self.st.add_text("hi", color="black")
        s = self.st.add_shape("rect", position=(4, 4), size=(10, 10))
        f = self.st.add_solid("blue")
        types = [l["type"] for l in self.st.layer_info()]
        self.assertEqual(types, ["text", "shape", "solid"])
        self.assertEqual({t, s, f}, {l["id"] for l in self.st.layer_info()})

    def test_move_rename(self):
        lid = self.st.add_shape("rect", position=(0, 0), size=(8, 8))
        self.st.move(lid, 20, 10)
        self.assertEqual(self.st.layer_info()[0]["position"], [20, 10])
        self.st.rename(lid, "my rect")
        self.assertEqual(self.st.layer_info()[0]["name"], "my rect")
        with self.assertRaises(MediaEditError):
            self.st.move(lid, "a", 1)
        with self.assertRaises(MediaEditError):
            self.st.rename(lid, "  ")

    def test_remove(self):
        a = self.st.add_solid("red")
        b = self.st.add_solid("blue")
        removed = self.st.remove(a)
        self.assertEqual(removed["id"], a)
        self.assertEqual(len(self.st), 1)
        self.assertEqual(self.st.layer_info()[0]["id"], b)
        with self.assertRaises(MediaEditError):
            self.st.remove(a)

    def test_reorder(self):
        a = self.st.add_solid("red", name="a")
        b = self.st.add_solid("blue", name="b")
        c = self.st.add_solid("green", name="c")
        idx = self.st.reorder(a, 2)
        self.assertEqual(idx, 2)
        self.assertEqual([l["id"] for l in self.st.layer_info()],
                         [b, c, a])
        # clamping
        idx = self.st.reorder(b, 99)
        self.assertEqual(idx, 2)
        self.assertEqual(self.st.layer_info()[2]["id"], b)

    def test_move_up_down(self):
        a = self.st.add_solid("red")
        b = self.st.add_solid("blue")
        self.assertEqual(self.st.move_up(a), 1)
        self.assertEqual([l["id"] for l in self.st.layer_info()], [b, a])
        self.assertEqual(self.st.move_up(a), 1)  # already top: no-op
        self.assertEqual(self.st.move_down(a), 0)
        self.assertEqual([l["id"] for l in self.st.layer_info()], [a, b])
        self.assertEqual(self.st.move_down(a), 0)  # already bottom: no-op

    def test_unknown_id_errors(self):
        for fn in (lambda: self.st.set_opacity("nope", 0.5),
                   lambda: self.st.set_blend("nope", "multiply"),
                   lambda: self.st.set_visible("nope", False),
                   lambda: self.st.move("nope", 1, 1),
                   lambda: self.st.rename("nope", "x"),
                   lambda: self.st.remove("nope"),
                   lambda: self.st.reorder("nope", 0),
                   lambda: self.st.move_up("nope")):
            with self.assertRaises(MediaEditError):
                fn()


class TestOpacityBlend(unittest.TestCase):
    def setUp(self):
        self.st = LayerStack((64, 48))
        self.lid = self.st.add_image(_img("red"))

    def test_opacity_clamps(self):
        self.st.set_opacity(self.lid, 2.0)
        self.assertEqual(self.st.layer_info()[0]["opacity"], 1.0)
        self.st.set_opacity(self.lid, -3.0)
        self.assertEqual(self.st.layer_info()[0]["opacity"], 0.0)

    def test_opacity_non_numeric_rejected(self):
        with self.assertRaises(MediaEditError):
            self.st.set_opacity(self.lid, "half")
        with self.assertRaises(MediaEditError):
            self.st.set_opacity(self.lid, float("nan"))

    def test_invalid_blend_rejected(self):
        with self.assertRaises(MediaEditError):
            LayerStack((8, 8)).add_solid("red", blend="photon")
        with self.assertRaises(MediaEditError):
            self.st.set_blend(self.lid, "photon")

    def test_all_blend_modes_accepted(self):
        from nomorals.media_edit.studio import BLEND_MODES
        for mode in BLEND_MODES:
            self.st.set_blend(self.lid, mode)

    def test_blend_modes_change_pixels(self):
        # mid-gray base; opaque red layer: normal vs multiply must differ
        a = LayerStack(_img((128, 128, 128), (32, 24)))
        la = a.add_image(_img("red", (32, 24)), blend="normal")
        b = LayerStack(_img((128, 128, 128), (32, 24)))
        lb = b.add_image(_img("red", (32, 24)), blend="multiply")
        pa = a.flatten().getpixel((4, 4))[:3]
        pb = b.flatten().getpixel((4, 4))[:3]
        self.assertEqual(pa, (255, 0, 0))
        self.assertNotEqual(pa, pb)
        self.assertEqual(pb, (128, 0, 0))
        a.set_blend(la, "multiply")
        self.assertEqual(a.flatten().getpixel((4, 4))[:3], pb)

    def test_opacity_affects_pixels(self):
        full = self.st.flatten().getpixel((4, 4))[:3]
        self.st.set_opacity(self.lid, 0.5)
        half = self.st.flatten().getpixel((4, 4))[:3]
        self.assertNotEqual(full, half)
        # 50% red over white ≈ pinkish, not full red
        self.assertTrue(half[0] > 200 and half[1] > 100)

    def test_visibility_respected(self):
        self.st.add_solid("blue")
        self.st.set_visible(self.st.layer_info()[-1]["id"], False)
        out = self.st.flatten()
        self.assertEqual(out.getpixel((4, 4))[:3], (255, 0, 0))  # red layer
        self.st.set_visible(self.st.layer_info()[-1]["id"], True)
        out = self.st.flatten()
        self.assertEqual(out.getpixel((4, 4))[:3], (0, 0, 255))  # blue covers


class TestRendering(unittest.TestCase):
    def test_text_layer_renders_non_blank(self):
        st = LayerStack((128, 64))
        st.add_text("Hello Devon", font_size=32, color="black")
        out = st.flatten()
        px = out.load()
        dark = sum(1 for x in range(out.width) for y in range(out.height)
                   if sum(px[x, y][:3]) < 200)
        self.assertGreater(dark, 50)

    def test_shape_layer_renders(self):
        st = LayerStack((64, 48))
        st.add_shape("ellipse", position=(8, 8), size=(24, 16), fill="red")
        out = st.flatten()
        self.assertEqual(out.getpixel((20, 16))[:3], (255, 0, 0))

    def test_solid_layer_renders(self):
        st = LayerStack((64, 48))
        st.add_solid("green")
        out = st.flatten()
        self.assertEqual(out.getpixel((30, 20))[:3], (0, 128, 0))

    def test_position_anchor_and_move(self):
        st = LayerStack((64, 48))
        lid = st.add_text("hi", font_size=24, color="black",
                          position="bottom-right")
        info = st.layer_info()[0]
        self.assertEqual(info["position"], "bottom-right")
        out = st.flatten()
        dark = sum(1 for x in range(out.width) for y in range(out.height)
                   if sum(out.load()[x, y][:3]) < 200)
        self.assertGreater(dark, 10)
        st.move(lid, 0, 0)
        self.assertEqual(st.layer_info()[0]["position"], [0, 0])

    def test_bad_position_rejected(self):
        st = LayerStack((8, 8))
        with self.assertRaises(MediaEditError):
            st.add_text("x", position="middle-earth")
        with self.assertRaises(MediaEditError):
            st.add_image(_img("red"), position=(1, 2, 3))

    def test_bad_shape_rejected(self):
        st = LayerStack((8, 8))
        with self.assertRaises(MediaEditError):
            st.add_shape("hexagon")
        with self.assertRaises(MediaEditError):
            st.add_text("")

    def test_bad_color_rejected(self):
        st = LayerStack((8, 8))
        with self.assertRaises(MediaEditError):
            st.add_solid("not-a-color")
        with self.assertRaises(MediaEditError):
            st.add_text("x", color="nope")


class TestSerialization(unittest.TestCase):
    def _rich_stack(self, tmp):
        img_path = Path(tmp) / "red.png"
        _img("red", (16, 12)).save(img_path)
        st = LayerStack((64, 48))
        st.add_image(str(img_path), name="pic", opacity=0.8,
                     blend="multiply", position=(4, 4))
        st.add_text("Hello", font_size=28, color="black", name="title")
        st.add_shape("rect", position=(40, 30), size=(12, 8), fill="blue",
                     name="box")
        st.add_solid("green", opacity=0.25, name="tint")
        st.set_visible(st.layer_info()[0]["id"], False)
        return st

    def test_round_trip(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            st = self._rich_stack(tmp)
            d = st.to_dict()
            # JSON-serializable
            raw = json.dumps(d)
            st2 = LayerStack.from_dict(json.loads(raw))
            self.assertEqual(len(st2), 4)
            # ids, order, params preserved
            i1 = [l["id"] for l in st.layer_info()]
            i2 = [l["id"] for l in st2.layer_info()]
            self.assertEqual(i1, i2)
            for a, b in zip(st.layer_info(), st2.layer_info()):
                self.assertEqual(a["opacity"], b["opacity"])
                self.assertEqual(a["blend"], b["blend"])
                self.assertEqual(a["visible"], b["visible"])
                self.assertEqual(a["type"], b["type"])
            # pixel-identical renders
            from PIL import ImageChops
            diff = ImageChops.difference(st.flatten(), st2.flatten())
            self.assertIsNone(diff.getbbox())

    def test_in_memory_image_not_serializable(self):
        st = LayerStack((32, 32))
        st.add_image(_img("red"))
        with self.assertRaises(MediaEditError) as ctx:
            st.to_dict()
        self.assertIn("in-memory", str(ctx.exception))

    def test_in_memory_base_not_serializable(self):
        st = LayerStack(_img("red"))
        st.add_solid("blue")
        with self.assertRaises(MediaEditError):
            st.to_dict()

    def test_path_base_round_trip(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "base.png"
            _img("white", (32, 24)).save(p)
            st = LayerStack(p)
            st.add_solid("red")
            st2 = LayerStack.from_dict(json.loads(json.dumps(st.to_dict())))
            self.assertEqual(st2.flatten().getpixel((2, 2))[:3], (255, 0, 0))

    def test_bad_version_rejected(self):
        with self.assertRaises(MediaEditError):
            LayerStack.from_dict({"version": 999, "base": {}, "layers": []})


if __name__ == "__main__":
    unittest.main()
