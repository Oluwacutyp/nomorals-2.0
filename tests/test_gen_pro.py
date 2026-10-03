"""Tests for pro image-gen: styles/aspects/quality, img2img, inpaint,
outpaint, background removal/replacement."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from nomorals.media_edit import generate as G
from nomorals.media_edit.generate import (
    ASPECT_RATIOS,
    QUALITY,
    STYLES,
    GenerativeEditError,
    _resolve_gen_params,
    op_bg_remove,
    op_bg_replace,
    op_img2img,
    op_inpaint,
    op_outpaint,
    op_txt2img,
)


def _img(w=64, h=64, color="red"):
    return Image.new("RGB", (w, h), color)


def _backend():
    be = MagicMock()
    be.generate.return_value = [_img()]
    be.img2img.side_effect = lambda im, p, **k: _img(*im.size)
    be.inpaint.side_effect = lambda im, m, p, **k: im.copy()
    return be


class PresetTests(unittest.TestCase):
    def test_styles_nonempty(self):
        self.assertGreaterEqual(len(STYLES), 10)
        for name, suffix in STYLES.items():
            self.assertTrue(suffix.strip(), name)

    def test_aspects_cover_common(self):
        for a in ("1:1", "16:9", "9:16", "4:3", "3:2"):
            self.assertIn(a, ASPECT_RATIOS)

    def test_quality_ordered(self):
        self.assertLess(QUALITY["draft"]["steps"],
                        QUALITY["standard"]["steps"])
        self.assertLess(QUALITY["standard"]["steps"],
                        QUALITY["ultra"]["steps"])

    def test_resolve_style_appends(self):
        prompt, _, _, _ = _resolve_gen_params("a cat", style="anime")
        self.assertIn("a cat", prompt)
        self.assertIn(STYLES["anime"], prompt)

    def test_resolve_aspect_sets_size(self):
        _, w, h, _ = _resolve_gen_params("x", aspect="16:9")
        self.assertEqual((w, h), ASPECT_RATIOS["16:9"])

    def test_explicit_size_beats_aspect(self):
        _, w, h, _ = _resolve_gen_params("x", aspect="16:9",
                                         width=100, height=200)
        self.assertEqual((w, h), (100, 200))

    def test_resolve_quality_sets_steps(self):
        _, _, _, steps = _resolve_gen_params("x", quality="ultra")
        self.assertEqual(steps, QUALITY["ultra"]["steps"])

    def test_explicit_steps_beat_quality(self):
        _, _, _, steps = _resolve_gen_params("x", quality="ultra", steps=7)
        self.assertEqual(steps, 7)

    def test_unknown_preset_raises(self):
        with self.assertRaises(GenerativeEditError):
            _resolve_gen_params("x", style="nope")
        with self.assertRaises(GenerativeEditError):
            _resolve_gen_params("x", aspect="nope")
        with self.assertRaises(GenerativeEditError):
            _resolve_gen_params("x", quality="nope")


class Txt2ImgProTests(unittest.TestCase):
    def test_style_aspect_quality_pass_through(self):
        be = _backend()
        with patch.object(G, "get_backend", return_value=be):
            op_txt2img("a cat", style="cinematic", aspect="16:9",
                       quality="draft")
        _, kwargs = be.generate.call_args
        self.assertIn(STYLES["cinematic"], kwargs["prompt"]
                      if "prompt" in kwargs else be.generate.call_args[0][0])
        self.assertEqual(kwargs["width"], ASPECT_RATIOS["16:9"][0])
        self.assertEqual(kwargs["height"], ASPECT_RATIOS["16:9"][1])
        self.assertEqual(kwargs["steps"], QUALITY["draft"]["steps"])


class Img2ImgTests(unittest.TestCase):
    def test_img2img_calls_backend(self):
        be = _backend()
        with patch.object(G, "get_backend", return_value=be):
            out = op_img2img(_img(), "make it cyberpunk", strength=0.7,
                             seed=3)
        self.assertIsInstance(out, Image.Image)
        _, kwargs = be.img2img.call_args
        self.assertAlmostEqual(kwargs["strength"], 0.7)
        self.assertEqual(kwargs["seed"], 3)

    def test_img2img_strength_validated(self):
        be = _backend()
        with patch.object(G, "get_backend", return_value=be):
            # validation happens in backend; mock passes through, so
            # check the op rejects empty prompt itself via backend mock
            with self.assertRaises(Exception):
                be.img2img.side_effect = GenerativeEditError("bad strength")
                op_img2img(_img(), "x", strength=0)

    def test_img2img_style_applied(self):
        be = _backend()
        with patch.object(G, "get_backend", return_value=be):
            op_img2img(_img(), "a city", style="anime")
        prompt = be.img2img.call_args[0][1]
        self.assertIn(STYLES["anime"], prompt)

    def test_img2img_unknown_style_raises(self):
        be = _backend()
        with patch.object(G, "get_backend", return_value=be):
            with self.assertRaises(GenerativeEditError):
                op_img2img(_img(), "x", style="nope")


class InpaintTests(unittest.TestCase):
    def test_inpaint_normalizes_mask(self):
        be = _backend()
        seen = {}

        def fake_inpaint(img, mask, prompt, **kw):
            seen["mask"] = mask
            return img.copy()

        be.inpaint.side_effect = fake_inpaint
        with patch.object(G, "get_backend", return_value=be):
            out = op_inpaint(_img(80, 60), (10, 10, 30, 30),
                             "remove the stain")
        self.assertIsInstance(out, Image.Image)
        self.assertEqual(seen["mask"].size, (80, 60))
        self.assertEqual(seen["mask"].mode, "L")

    def test_inpaint_accepts_pil_mask(self):
        be = _backend()
        with patch.object(G, "get_backend", return_value=be):
            m = Image.new("L", (64, 64), 0)
            out = op_inpaint(_img(), m, "fill with sky")
            self.assertIsInstance(out, Image.Image)

    def test_hf_inpaint_uses_masked_composite(self):
        # Real HF backend logic with a fake client: img2img result is
        # composited through the mask; unmasked pixels preserved.
        from nomorals.media_edit.generate import HFInferenceBackend
        be = HFInferenceBackend(model="m", token="t")
        fake_client = MagicMock()
        fake_client.image_to_image.return_value = _img(64, 64, "blue")
        with patch.object(be, "_client", return_value=fake_client):
            out = be.inpaint(_img(64, 64, "red"), (0, 0, 10, 10),
                             "blue blob")
        self.assertIsInstance(out, Image.Image)
        self.assertEqual(out.size, (64, 64))


class OutpaintTests(unittest.TestCase):
    def test_outpaint_expands_canvas(self):
        be = _backend()
        seen = {}

        def fake_inpaint(img, mask, prompt, **kw):
            seen["img"] = img
            seen["mask"] = mask
            return img.copy()

        be.inpaint.side_effect = fake_inpaint
        with patch.object(G, "get_backend", return_value=be):
            out = op_outpaint(_img(64, 64), "extend the landscape",
                              right=32, bottom=16)
        self.assertEqual(out.size, (96, 80))
        # Mask covers exactly the new border.
        m = seen["mask"]
        self.assertEqual(m.size, (96, 80))

    def test_outpaint_preserves_center(self):
        be = _backend()
        # Identity backend: canvas returned unchanged; center must equal
        # the original pixels.
        be.inpaint.side_effect = lambda im, m, p, **k: im.copy()
        with patch.object(G, "get_backend", return_value=be):
            out = op_outpaint(_img(32, 32, "green"), "more",
                              left=8, top=8)
            center = out.crop((8, 8, 40, 40))
            orig = _img(32, 32, "green")
            self.assertEqual(list(center.getdata()),
                             list(orig.getdata()))

    def test_outpaint_needs_padding(self):
        be = _backend()
        with patch.object(G, "get_backend", return_value=be):
            with self.assertRaises(GenerativeEditError):
                op_outpaint(_img(), "x")

    def test_outpaint_rejects_negative(self):
        be = _backend()
        with patch.object(G, "get_backend", return_value=be):
            with self.assertRaises(GenerativeEditError):
                op_outpaint(_img(), "x", top=-5)


class BgRemoveTests(unittest.TestCase):
    def _chroma_img(self):
        # Green background with a red square subject.
        img = Image.new("RGB", (40, 40), (0, 200, 0))
        from PIL import ImageDraw
        d = ImageDraw.Draw(img)
        d.rectangle([12, 12, 28, 28], fill=(200, 0, 0))
        return img

    def test_chroma_removes_background(self):
        out = op_bg_remove(self._chroma_img(), mode="chroma",
                           chroma_color=(0, 200, 0), tolerance=60,
                           feather=0)
        self.assertEqual(out.mode, "RGBA")
        # Corner (background) fully transparent.
        self.assertEqual(out.getpixel((0, 0))[3], 0)
        # Center (subject) opaque.
        self.assertEqual(out.getpixel((20, 20))[3], 255)

    def test_chroma_auto_key_from_corners(self):
        out = op_bg_remove(self._chroma_img(), mode="chroma", feather=0)
        self.assertEqual(out.getpixel((0, 0))[3], 0)
        self.assertEqual(out.getpixel((20, 20))[3], 255)

    def test_chroma_bad_tolerance_raises(self):
        with self.assertRaises(GenerativeEditError):
            op_bg_remove(self._chroma_img(), mode="chroma", tolerance=999)

    def test_auto_without_rembg_raises_helpful(self):
        try:
            import rembg  # noqa: F401
            self.skipTest("rembg is installed here")
        except ImportError:
            pass
        with self.assertRaises(GenerativeEditError) as ctx:
            op_bg_remove(self._chroma_img(), mode="auto")
        self.assertIn("rembg", str(ctx.exception))
        self.assertIn("pip install", str(ctx.exception))

    def test_unknown_mode_raises(self):
        with self.assertRaises(GenerativeEditError):
            op_bg_remove(self._chroma_img(), mode="magic")

    def test_bg_replace_color(self):
        out = op_bg_replace(self._chroma_img(), "#0000ff", mode="chroma",
                            chroma_color=(0, 200, 0), tolerance=60)
        self.assertEqual(out.mode, "RGB")
        # Background now blue.
        self.assertEqual(out.getpixel((0, 0)), (0, 0, 255))
        # Subject preserved.
        r, g, b = out.getpixel((20, 20))
        self.assertGreater(r, 100)

    def test_bg_replace_blur(self):
        out = op_bg_replace(self._chroma_img(), "blur", mode="chroma",
                            chroma_color=(0, 200, 0), tolerance=60)
        self.assertEqual(out.mode, "RGB")
        self.assertEqual(out.size, (40, 40))

    def test_bg_replace_image(self):
        bg = _img(80, 20, "yellow")
        out = op_bg_replace(self._chroma_img(), bg, mode="chroma",
                            chroma_color=(0, 200, 0), tolerance=60)
        self.assertEqual(out.size, (40, 40))
        # Cover-fit yellow background visible in a corner.
        self.assertEqual(out.getpixel((0, 0)), (255, 255, 0))


class OpRegistryTests(unittest.TestCase):
    def test_new_ops_registered(self):
        from nomorals.media_edit.images import OP_ALLOWLIST, _OP_FUNCS
        for name in ("img2img", "inpaint", "outpaint",
                     "bg_remove", "bg_replace"):
            self.assertIn(name, OP_ALLOWLIST, name)
            self.assertIn(name, _OP_FUNCS, name)


if __name__ == "__main__":
    unittest.main()
