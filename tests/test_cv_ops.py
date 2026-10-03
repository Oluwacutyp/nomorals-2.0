"""Tests for nomorals.media_edit.cv_ops (OpenCV + scikit-image backends).

The backends are declared dependencies of the ``media-edit`` extra, so they
are imported directly. These tests exercise the real implementations;
``FallbackTests`` forces each backend off (via ``C._BACKENDS``) to cover the
Pillow fallbacks, and ``ZeroDependencyTests`` blocks cv2, scikit-image AND
numpy at once to prove the standing rule: every op works with zero
third-party dependencies.
"""

import unittest

import numpy as np
from PIL import Image, ImageDraw

from nomorals.media_edit import cv_ops as C
from nomorals.media_edit import images as IM
from nomorals.media_edit.images import MediaEditError, apply_chain


def _noisy_rgb(seed: int = 7, size: int = 64) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(
        rng.integers(0, 256, (size, size, 3)).astype("uint8"), "RGB")


def _shapes(size: int = 128) -> Image.Image:
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([30, 30, 98, 98], fill="red")
    d.ellipse([10, 10, 40, 40], fill="blue")
    return img


def _box_mask(size: tuple[int, int], box: tuple[int, int, int, int]):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rectangle(list(box), fill=255)
    return m


import contextlib


@contextlib.contextmanager
def _force_backends(**overrides):
    """Force backend selection by patching ``C._BACKENDS`` (test helper)."""
    saved = dict(C._BACKENDS)
    C._BACKENDS.clear()
    C._BACKENDS.update(overrides)
    try:
        yield
    finally:
        C._BACKENDS.clear()
        C._BACKENDS.update(saved)


class DenoiseTests(unittest.TestCase):
    def test_nlmeans_returns_pil_same_size(self):
        out = C.denoise(_noisy_rgb())
        self.assertIsInstance(out, Image.Image)
        self.assertEqual(out.size, (64, 64))
        self.assertEqual(out.mode, "RGB")

    def test_nlmeans_reduces_noise_variance(self):
        img = _noisy_rgb()
        before = np.asarray(img).astype(float).var()
        after = np.asarray(C.denoise(img, strength=15)).astype(float).var()
        self.assertLess(after, before)

    def test_bilateral_method(self):
        out = C.denoise(_noisy_rgb(), strength=8, method="bilateral")
        self.assertIsInstance(out, Image.Image)
        self.assertEqual(out.size, (64, 64))

    def test_preserves_alpha(self):
        img = _noisy_rgb().convert("RGBA")
        alpha = Image.new("L", img.size, 128)
        img.putalpha(alpha)
        out = C.denoise(img)
        self.assertEqual(out.mode, "RGBA")
        self.assertEqual(list(out.getchannel("A").getdata()),
                         list(alpha.getdata()))

    def test_bad_method(self):
        with self.assertRaises(MediaEditError):
            C.denoise(_noisy_rgb(), method="nope")

    def test_bad_strength(self):
        with self.assertRaises(MediaEditError):
            C.denoise(_noisy_rgb(), strength=0)


class EdgeDetectTests(unittest.TestCase):
    def test_returns_grayscale_edge_map(self):
        out = C.edge_detect(_shapes())
        self.assertIsInstance(out, Image.Image)
        self.assertEqual(out.mode, "L")
        self.assertEqual(out.size, (128, 128))

    def test_finds_edges(self):
        out = C.edge_detect(_shapes())
        self.assertGreater(np.asarray(out).sum(), 0)

    def test_blank_image_has_no_edges(self):
        out = C.edge_detect(Image.new("RGB", (64, 64), "gray"))
        self.assertEqual(np.asarray(out).sum(), 0)


class InpaintTests(unittest.TestCase):
    def test_inpaint_box_string(self):
        img = Image.new("RGB", (64, 64), "navy")
        out = C.inpaint_cv(img, "20,20,44,44")
        self.assertIsInstance(out, Image.Image)
        self.assertEqual(out.size, (64, 64))
        # inpainted region should blend toward the navy background
        region = np.asarray(out)[20:45, 20:45].astype(float)
        self.assertLess(np.abs(region - [0, 0, 128]).mean(), 40)

    def test_inpaint_pil_mask(self):
        img = Image.new("RGB", (64, 64), "green")
        out = C.inpaint_cv(img, _box_mask((64, 64), (20, 20, 44, 44)),
                           method="ns")
        self.assertEqual(out.size, (64, 64))

    def test_empty_mask_fails(self):
        with self.assertRaises(MediaEditError):
            C.inpaint_cv(_noisy_rgb(), Image.new("L", (64, 64), 0))

    def test_box_outside_image_fails(self):
        with self.assertRaises(MediaEditError):
            C.inpaint_cv(_noisy_rgb(), "100,100,200,200")

    def test_bad_method(self):
        with self.assertRaises(MediaEditError):
            C.inpaint_cv(_noisy_rgb(), "1,1,4,4", method="nope")


class SeamlessCloneTests(unittest.TestCase):
    def test_clone_center(self):
        fg = _shapes(64).crop((20, 20, 60, 60))
        bg = Image.new("RGB", (200, 200), "green")
        out = C.seamless_clone(fg, bg, position="center")
        self.assertIsInstance(out, Image.Image)
        self.assertEqual(out.size, (200, 200))

    def test_clone_explicit_position(self):
        # textured patch (flat color would vanish under Poisson blending,
        # which is correct gradient-domain behavior)
        fg = _shapes(64).crop((16, 16, 56, 56))
        bg = Image.new("RGB", (200, 200), "blue")
        out = C.seamless_clone(fg, bg, position=(100, 100))
        self.assertEqual(out.size, (200, 200))
        center = np.asarray(out)[90:110, 90:110].astype(float)
        corner = np.asarray(out)[0:20, 0:20].astype(float)
        # pasted region differs from the untouched blue background
        self.assertGreater(np.abs(center - corner).mean(), 5)

    def test_clone_with_mask(self):
        fg = Image.new("RGB", (40, 40), "red")
        bg = Image.new("RGB", (200, 200), "blue")
        mask = Image.new("L", (40, 40), 0)
        ImageDraw.Draw(mask).ellipse([5, 5, 35, 35], fill=255)
        out = C.seamless_clone(fg, bg, mask=mask)
        self.assertEqual(out.size, (200, 200))

    def test_clone_uses_alpha_as_mask(self):
        fg = Image.new("RGBA", (40, 40), (255, 0, 0, 0))
        ImageDraw.Draw(fg).ellipse([5, 5, 35, 35], fill=(255, 0, 0, 255))
        bg = Image.new("RGB", (200, 200), "blue")
        out = C.seamless_clone(fg, bg)
        self.assertEqual(out.size, (200, 200))

    def test_empty_mask_fails(self):
        with self.assertRaises(MediaEditError):
            C.seamless_clone(_shapes(32), Image.new("RGB", (100, 100)),
                             mask=Image.new("L", (32, 32), 0))

    def test_bad_position(self):
        with self.assertRaises(MediaEditError):
            C.seamless_clone(_shapes(32), Image.new("RGB", (100, 100)),
                             position="top-left")


class SharpenTests(unittest.TestCase):
    def test_zero_amount_is_noop(self):
        img = _shapes()
        out = C.sharpen_advanced(img, amount=0)
        self.assertEqual(list(out.getdata()), list(img.getdata()))

    def test_sharpen_increases_local_contrast(self):
        img = _shapes()
        base = np.asarray(img).astype(float)
        sharp = np.asarray(C.sharpen_advanced(img, amount=2.0)).astype(float)
        # variance of the laplacian-ish difference grows with sharpening
        self.assertGreater(np.abs(sharp - base).mean(), 0.05)

    def test_bad_params(self):
        with self.assertRaises(MediaEditError):
            C.sharpen_advanced(_shapes(), amount=-1)
        with self.assertRaises(MediaEditError):
            C.sharpen_advanced(_shapes(), sigma=0)


class StylizeTests(unittest.TestCase):
    def test_cartoonize(self):
        out = C.cartoonize(_shapes())
        self.assertIsInstance(out, Image.Image)
        self.assertEqual(out.size, (128, 128))
        self.assertEqual(out.mode, "RGB")

    def test_pencil_sketch(self):
        out = C.pencil_sketch(_shapes())
        self.assertEqual(out.mode, "L")
        self.assertEqual(out.size, (128, 128))
        # sketch of a mostly-white image should be mostly bright
        self.assertGreater(np.asarray(out).mean(), 180)

    def test_sketch_bad_sigma(self):
        with self.assertRaises(MediaEditError):
            C.pencil_sketch(_shapes(), blur_sigma=0)


class PerspectiveTests(unittest.TestCase):
    def test_explicit_corners(self):
        out = C.perspective_transform(
            _shapes(), corners=[(10, 10), (118, 10), (118, 118), (10, 118)])
        self.assertEqual(out.size, (108, 108))

    def test_corners_string(self):
        out = C.perspective_transform(
            _shapes(), corners="10,10;118,10;118,118;10,118")
        self.assertEqual(out.size, (108, 108))

    def test_explicit_output_size(self):
        out = C.perspective_transform(
            _shapes(), corners=[(10, 10), (118, 10), (118, 118), (10, 118)],
            width=50, height=60)
        self.assertEqual(out.size, (50, 60))

    def test_bad_corners(self):
        with self.assertRaises(MediaEditError):
            C.perspective_transform(_shapes(), corners=[(1, 2), (3, 4)])
        with self.assertRaises(MediaEditError):
            C.perspective_transform(_shapes(), corners="1,2;3,4")

    def test_auto_detect_on_blank_fails_fast(self):
        with self.assertRaises(MediaEditError):
            C.perspective_transform(Image.new("RGB", (64, 64), "gray"))

    def test_auto_detect_on_quad(self):
        # white quadrilateral on black -> auto-detect should find 4 corners
        img = Image.new("RGB", (200, 200), "black")
        ImageDraw.Draw(img).polygon([(30, 40), (170, 30), (180, 170),
                                     (20, 160)], fill="white")
        out = C.perspective_transform(img)
        self.assertIsInstance(out, Image.Image)
        w, h = out.size
        self.assertTrue(100 < w < 200 and 100 < h < 200)


class GrabcutTests(unittest.TestCase):
    def _subject(self):
        img = Image.new("RGB", (128, 128), "skyblue")
        ImageDraw.Draw(img).ellipse([39, 39, 89, 89], fill="darkred")
        return img

    def test_transparent_cutout(self):
        out = C.grabcut_segment(self._subject(), rect=(30, 30, 98, 98),
                                iterations=2)
        self.assertEqual(out.mode, "RGBA")
        alpha = np.asarray(out.getchannel("A"))
        self.assertTrue((alpha == 0).any())   # some background removed
        self.assertTrue((alpha == 255).any())  # subject kept

    def test_blur_background(self):
        out = C.grabcut_segment(self._subject(), rect=(30, 30, 98, 98),
                                iterations=2, background="blur")
        self.assertEqual(out.mode, "RGB")
        self.assertEqual(out.size, (128, 128))

    def test_white_background(self):
        out = C.grabcut_segment(self._subject(), rect=(30, 30, 98, 98),
                                iterations=2, background="white")
        self.assertEqual(out.mode, "RGB")

    def test_rect_string(self):
        out = C.grabcut_segment(self._subject(), rect="30,30,98,98",
                                iterations=2)
        self.assertEqual(out.mode, "RGBA")

    def test_bad_background(self):
        with self.assertRaises(MediaEditError):
            C.grabcut_segment(self._subject(), background="checkerboard")

    def test_rect_outside_image(self):
        with self.assertRaises(MediaEditError):
            C.grabcut_segment(self._subject(), rect=(0, 0, 500, 500))

    def test_bad_rect_string(self):
        with self.assertRaises(MediaEditError):
            C.grabcut_segment(self._subject(), rect="not-a-box")


class RescaleTests(unittest.TestCase):
    def test_upscale_size(self):
        out = C.rescale_ski(_shapes(), scale=1.5)
        self.assertEqual(out.size, (192, 192))

    def test_downscale(self):
        out = C.rescale_ski(_shapes(), scale=0.5, order=1)
        self.assertEqual(out.size, (64, 64))

    def test_preserves_alpha(self):
        img = _shapes().convert("RGBA")
        out = C.rescale_ski(img, scale=2.0)
        self.assertEqual(out.mode, "RGBA")
        self.assertEqual(out.size, (256, 256))

    def test_bad_scale(self):
        with self.assertRaises(MediaEditError):
            C.rescale_ski(_shapes(), scale=0)

    def test_bad_order(self):
        with self.assertRaises(MediaEditError):
            C.rescale_ski(_shapes(), order=9)


class ExposureTests(unittest.TestCase):
    def test_gamma_brightens(self):
        dark = Image.new("RGB", (32, 32), (40, 40, 40))
        out = C.exposure_adjust(dark, mode="gamma", gamma=0.5)
        self.assertGreater(np.asarray(out).mean(), 40)

    def test_gamma_darkens(self):
        bright = Image.new("RGB", (32, 32), (200, 200, 200))
        out = C.exposure_adjust(bright, mode="gamma", gamma=2.0)
        self.assertLess(np.asarray(out).mean(), 200)

    def test_log_mode(self):
        out = C.exposure_adjust(_shapes(), mode="log")
        self.assertEqual(out.size, (128, 128))

    def test_adaptive_mode(self):
        out = C.exposure_adjust(_noisy_rgb(3, 48))
        self.assertEqual(out.size, (48, 48))

    def test_bad_mode(self):
        with self.assertRaises(MediaEditError):
            C.exposure_adjust(_shapes(), mode="vintage")

    def test_bad_gamma(self):
        with self.assertRaises(MediaEditError):
            C.exposure_adjust(_shapes(), mode="gamma", gamma=0)


class MatchHistogramTests(unittest.TestCase):
    def test_match_shifts_colors(self):
        src = Image.new("RGB", (48, 48), (200, 100, 50))
        ref = Image.new("RGB", (48, 48), (20, 60, 200))
        out = C.match_histograms(src, ref)
        self.assertEqual(out.size, (48, 48))
        mean = np.asarray(out).mean(axis=(0, 1))
        # blue channel should now dominate like the reference
        self.assertGreater(mean[2], mean[0])

    def test_accepts_path(self):
        # reference as a file path
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "ref.png")
            Image.new("RGB", (32, 32), "teal").save(p)
            out = C.match_histograms(Image.new("RGB", (32, 32), "olive"),
                                     p)
            self.assertEqual(out.size, (32, 32))

    def test_bad_reference(self):
        with self.assertRaises(MediaEditError):
            C.match_histograms(_shapes(32), 12345)


class AdaptiveThresholdTests(unittest.TestCase):
    def test_sauvola_binary(self):
        out = C.adaptive_threshold(_shapes())
        self.assertEqual(out.mode, "L")
        vals = set(np.unique(np.asarray(out)).tolist())
        self.assertTrue(vals <= {0, 255})

    def test_otsu(self):
        out = C.adaptive_threshold(_shapes(), method="otsu")
        self.assertEqual(out.mode, "L")

    def test_niblack(self):
        out = C.adaptive_threshold(_shapes(), method="niblack")
        self.assertEqual(out.mode, "L")

    def test_bad_method(self):
        with self.assertRaises(MediaEditError):
            C.adaptive_threshold(_shapes(), method="bradley")


class ParseHelpersTests(unittest.TestCase):
    def test_parse_box(self):
        self.assertEqual(C.parse_box("10,20,30,40"), (10, 20, 30, 40))
        self.assertIsNone(C.parse_box("not a box"))
        self.assertIsNone(C.parse_box(123))

    def test_parse_corners(self):
        pts = C.parse_corners("10,10;118,10;118,118;10,118")
        self.assertEqual(pts, [(10, 10), (118, 10), (118, 118), (10, 118)])
        self.assertIsNone(C.parse_corners("10,10;20,20"))
        self.assertIsNone(C.parse_corners("nope"))


class RegistrationTests(unittest.TestCase):
    def test_all_ops_registered(self):
        for name in C.CV_IMAGE_OPS:
            self.assertIn(name, IM.OP_ALLOWLIST, name)
            self.assertIn(name, IM._OP_FUNCS, name)

    def test_expected_op_names(self):
        self.assertEqual(set(C.CV_IMAGE_OPS), {
            "denoise", "edge_detect", "inpaint_cv", "seamless_clone",
            "sharpen", "cartoonize", "pencil_sketch", "perspective",
            "grabcut", "rescale", "exposure", "match_histogram",
            "adaptive_threshold", "orb_features", "kmeans", "deblur",
            "superpixels", "white_balance", "auto_levels",
            "color_transfer", "panorama", "retouch_smooth", "tone_map",
            "detail_enhance", "stylize", "clarity", "shadow_highlight",
            "dehaze", "seam_carve", "tilt_shift", "selective_color",
            "split_tone", "curves", "cube_lut", "replace_background"})

    def test_backend_status(self):
        st = C.backend_status()
        self.assertTrue(st["opencv"])
        self.assertTrue(st["scikit_image"])

    def test_op_chain_end_to_end(self):
        out = apply_chain(_shapes(), [
            {"op": "denoise", "strength": 5, "method": "bilateral"},
            {"op": "sharpen", "amount": 0.8},
            {"op": "exposure", "mode": "adaptive"},
        ])
        self.assertIsInstance(out, Image.Image)
        self.assertEqual(out.size, (128, 128))


class FallbackTests(unittest.TestCase):
    """Multi-backend: with cv2/skimage forced off, Pillow fallbacks kick in.

    Backends resolve through ``C._BACKENDS``; patching that dict forces
    the fallback paths without uninstalling anything.
    """

    def _force(self, **overrides):
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            saved = dict(C._BACKENDS)
            C._BACKENDS.clear()
            C._BACKENDS.update(overrides)
            try:
                yield
            finally:
                C._BACKENDS.clear()
                C._BACKENDS.update(saved)
        return _ctx()

    def _pillow_only(self):
        return self._force(cv2=None, skimage=None, numpy=None)

    def _no_cv2(self):
        return self._force(cv2=None)

    def test_denoise_pillow_fallback(self):
        with self._pillow_only():
            out = C.denoise(_noisy_rgb())
            self.assertIsInstance(out, Image.Image)
            self.assertEqual(out.size, (64, 64))

    def test_denoise_skimage_fallback(self):
        with self._force(cv2=None, skimage=_real_skimage(),
                         numpy=_real_numpy()):
            out = C.denoise(_noisy_rgb(), strength=8)
            self.assertEqual(out.size, (64, 64))

    def test_edge_detect_pillow_fallback(self):
        with self._pillow_only():
            out = C.edge_detect(_shapes())
            self.assertEqual(out.mode, "L")
            self.assertEqual(out.size, (128, 128))

    def test_edge_detect_skimage_fallback(self):
        with self._force(cv2=None, skimage=_real_skimage(),
                         numpy=_real_numpy()):
            out = C.edge_detect(_shapes())
            self.assertEqual(out.mode, "L")
            self.assertGreater(np.asarray(out).sum(), 0)

    def test_sharpen_pillow_fallback(self):
        with self._pillow_only():
            out = C.sharpen_advanced(_shapes(), amount=1.0)
            self.assertEqual(out.size, (128, 128))

    def test_sharpen_skimage_fallback(self):
        with self._force(cv2=None, skimage=_real_skimage(),
                         numpy=_real_numpy()):
            out = C.sharpen_advanced(_shapes(), amount=1.0)
            self.assertEqual(out.size, (128, 128))

    def test_cartoonize_pillow_fallback(self):
        with self._pillow_only():
            out = C.cartoonize(_shapes())
            self.assertEqual(out.size, (128, 128))

    def test_sketch_numpy_fallback(self):
        with self._force(cv2=None, numpy=_real_numpy()):
            out = C.pencil_sketch(_shapes())
            self.assertEqual(out.mode, "L")

    def test_sketch_pillow_fallback(self):
        with self._pillow_only():
            out = C.pencil_sketch(_shapes())
            self.assertEqual(out.mode, "L")
            self.assertEqual(out.size, (128, 128))

    def test_clone_pillow_fallback(self):
        with self._force(cv2=None, numpy=_real_numpy()):
            fg = _shapes(64).crop((16, 16, 56, 56))
            bg = Image.new("RGB", (200, 200), "blue")
            out = C.seamless_clone(fg, bg, position="center")
            self.assertEqual(out.size, (200, 200))

    def test_rescale_cv2_fallback(self):
        with self._force(skimage=None):
            out = C.rescale_ski(_shapes(), scale=1.5)
            self.assertEqual(out.size, (192, 192))

    def test_rescale_pillow_fallback(self):
        with self._pillow_only():
            out = C.rescale_ski(_shapes(), scale=2.0)
            self.assertEqual(out.size, (256, 256))

    def test_exposure_cv2_fallback(self):
        with self._force(skimage=None):
            out = C.exposure_adjust(_shapes(), mode="adaptive")
            self.assertEqual(out.size, (128, 128))

    def test_exposure_pillow_fallback(self):
        with self._pillow_only():
            dark = Image.new("RGB", (32, 32), (40, 40, 40))
            out = C.exposure_adjust(dark, mode="gamma", gamma=0.5)
            self.assertGreater(np.asarray(out).mean(), 40)
            out2 = C.exposure_adjust(_shapes(), mode="adaptive")
            self.assertEqual(out2.size, (128, 128))

    def test_match_hist_numpy_fallback(self):
        with self._force(skimage=None, numpy=_real_numpy()):
            src = Image.new("RGB", (48, 48), (200, 100, 50))
            ref = Image.new("RGB", (48, 48), (20, 60, 200))
            out = C.match_histograms(src, ref)
            mean = np.asarray(out).mean(axis=(0, 1))
            self.assertGreater(mean[2], mean[0])

    def test_threshold_cv2_fallback(self):
        with self._force(skimage=None):
            out = C.adaptive_threshold(_shapes(), method="sauvola")
            self.assertEqual(out.mode, "L")

    def test_threshold_numpy_fallback(self):
        with self._force(skimage=None, cv2=None, numpy=_real_numpy()):
            out = C.adaptive_threshold(_shapes(), method="otsu")
            vals = set(np.unique(np.asarray(out)).tolist())
            self.assertTrue(vals <= {0, 255})

    def test_perspective_pillow_fallback(self):
        with self._force(cv2=None, numpy=_real_numpy()):
            out = C.perspective_transform(
                _shapes(), corners="10,10;118,10;118,118;10,118")
            self.assertEqual(out.size, (108, 108))

    def test_inpaint_skimage_fallback(self):
        with self._force(cv2=None, numpy=_real_numpy(),
                         skimage=_real_skimage()):
            img = Image.new("RGB", (64, 64), "navy")
            out = C.inpaint_cv(img, "20,20,44,44")
            self.assertEqual(out.size, (64, 64))

    def test_inpaint_pillow_fallback(self):
        with self._pillow_only():
            img = Image.new("RGB", (48, 48), "navy")
            out = C.inpaint_cv(img, "20,20,28,28")
            self.assertIsInstance(out, Image.Image)
            self.assertEqual(out.size, (48, 48))
            # damaged box was filled with surrounding color, not left black
            self.assertEqual(out.getpixel((24, 24)), (0, 0, 128))

    def test_grabcut_pillow_fallback(self):
        with self._pillow_only():
            out = C.grabcut_segment(_shapes(), rect=(20, 20, 110, 110))
            self.assertEqual(out.mode, "RGBA")
            self.assertEqual(out.size, (128, 128))
            alpha = out.getchannel("A")
            # feathered box: solid inside, transparent outside
            self.assertEqual(alpha.getpixel((64, 64)), 255)
            self.assertEqual(alpha.getpixel((5, 5)), 0)

    def test_perspective_auto_no_cv2_raises_hint(self):
        with self._force(cv2=None):
            with self.assertRaises(MediaEditError) as cm:
                C.perspective_transform(_shapes())
            self.assertIn("OpenCV", str(cm.exception))

    def test_backend_status_reflects_forced_backends(self):
        with self._pillow_only():
            st = C.backend_status()
            self.assertFalse(st["opencv"])
            self.assertFalse(st["scikit_image"])
            self.assertFalse(st["numpy"])
            self.assertTrue(st["pillow"])
            # standing rule: every op has a working backend, none is "none"
            for op, backend in st["selected"].items():
                self.assertNotEqual(
                    backend, "none",
                    f"{op} has no working backend with zero third-party deps")


class ZeroDependencyTests(unittest.TestCase):
    """Standing rule: EVERY op works with zero third-party dependencies.

    Blocks cv2, scikit-image AND numpy at once, then runs the full op
    matrix plus correctness spot-checks on the trickiest pure-Pillow /
    stdlib fallbacks (panorama stitching, perspective solve, inpainting,
    dehazing, seam carving, .cube LUTs, local thresholding).
    """

    def _zero(self):
        return _force_backends(cv2=None, skimage=None, numpy=None)

    def _identity_cube(self, size=4):
        lines = ["TITLE identity", f"LUT_3D_SIZE {size}"]
        for b in range(size):
            for g in range(size):
                for r in range(size):
                    lines.append(
                        f"{r / (size - 1)} {g / (size - 1)} "
                        f"{b / (size - 1)}")
        return "\n".join(lines)

    def test_every_op_runs_with_zero_deps(self):
        img = _noisy_rgb(seed=11, size=64)
        mask = _box_mask((64, 64), (20, 20, 40, 40))
        bg = _noisy_rgb(seed=12, size=80)
        ref = _noisy_rgb(seed=13, size=64)
        frame_a = img.crop((0, 0, 40, 64))
        frame_b = img.crop((24, 0, 64, 64))
        cube = self._identity_cube()
        calls = [
            ("denoise", lambda: C.denoise(img)),
            ("edge_detect", lambda: C.edge_detect(img)),
            ("inpaint_cv", lambda: C.inpaint_cv(img, mask)),
            ("seamless_clone", lambda: C.seamless_clone(img, bg)),
            ("seamless_clone/mono",
             lambda: C.seamless_clone(img, bg, mix="monochrome")),
            ("sharpen", lambda: C.sharpen_advanced(img)),
            ("cartoonize", lambda: C.cartoonize(img)),
            ("pencil_sketch", lambda: C.pencil_sketch(img)),
            ("perspective", lambda: C.perspective_transform(
                img, corners=[(4, 4), (60, 2), (62, 60), (2, 62)])),
            ("grabcut", lambda: C.grabcut_segment(img, rect=(8, 8, 56, 56))),
            ("rescale", lambda: C.rescale_ski(img, 0.5)),
            ("exposure", lambda: C.exposure_adjust(img, mode="gamma")),
            ("match_histogram", lambda: C.match_histograms(img, ref)),
            ("adaptive_threshold",
             lambda: C.adaptive_threshold(img, method="sauvola")),
            ("orb_features", lambda: C.orb_features(img, n=20)),
            ("kmeans", lambda: C.kmeans_quantize(img, k=4)),
            ("deblur", lambda: C.deblur(img)),
            ("superpixels", lambda: C.superpixels(img, n_segments=12)),
            ("white_balance", lambda: C.white_balance(img)),
            ("auto_levels", lambda: C.auto_levels(img)),
            ("color_transfer", lambda: C.color_transfer(img, ref)),
            ("panorama", lambda: C.panorama(frame_a, [frame_b])),
            ("retouch_smooth", lambda: C.retouch_smooth(img)),
            ("tone_map", lambda: C.tone_map(img)),
            ("detail_enhance", lambda: C.detail_enhance(img)),
            ("stylize", lambda: C.stylize(img)),
            ("clarity", lambda: C.clarity(img)),
            ("shadow_highlight", lambda: C.shadow_highlight(img)),
            ("dehaze", lambda: C.dehaze(img)),
            ("seam_carve", lambda: C.seam_carve(img, width=52, height=50)),
            ("tilt_shift", lambda: C.tilt_shift(img)),
            ("selective_color", lambda: C.selective_color(img)),
            ("split_tone", lambda: C.split_tone(img)),
            ("apply_curves", lambda: C.apply_curves(img)),
            ("cube_lut", lambda: C.apply_cube_lut(img, cube)),
            ("replace_background",
             lambda: C.replace_background(img, bg, rect=(8, 8, 56, 56))),
        ]
        with self._zero():
            for name, fn in calls:
                with self.subTest(op=name):
                    out = fn()
                    self.assertIsInstance(out, Image.Image, name)
            with self.subTest(op="orb_count"):
                self.assertIsInstance(C.orb_count(img, n=20), int)

    def test_panorama_zero_deps_is_pixel_exact(self):
        rng = np.random.default_rng(21)
        big = Image.fromarray(
            rng.integers(0, 256, (72, 120, 3)).astype("uint8"), "RGB")
        a = big.crop((0, 0, 60, 72))
        b = big.crop((36, 0, 96, 72))
        with self._zero():
            pano = C.panorama(a, [b])
        self.assertEqual(pano.size, (96, 72))
        # non-blended margins must match the source frames exactly
        self.assertEqual(
            list(pano.crop((0, 0, 20, 72)).getdata()),
            list(big.crop((0, 0, 20, 72)).getdata()))
        self.assertEqual(
            list(pano.crop((76, 0, 96, 72)).getdata()),
            list(big.crop((76, 0, 96, 72)).getdata()))

    def test_perspective_pure_python_matches_numpy(self):
        src = [(4.0, 4.0), (60.0, 2.0), (62.0, 60.0), (2.0, 62.0)]
        dst = [(0.0, 0.0), (63.0, 0.0), (63.0, 63.0), (0.0, 63.0)]
        py_coeffs = C._perspective_coeffs_pillow(src, dst)
        np_coeffs = C._perspective_coeffs(src, dst, np)
        for a, b_ in zip(py_coeffs, np_coeffs):
            self.assertAlmostEqual(a, b_, places=6)

    def test_perspective_pure_python_matches_numpy_warp(self):
        # end-to-end: identical pixels whether coefficients come from the
        # pure-Python solver or the numpy solver
        img = _shapes(64)
        src = [(4.0, 4.0), (60.0, 2.0), (62.0, 60.0), (2.0, 62.0)]
        out_py = C._perspective_pillow(img, src, 60, 60, None)
        out_np = C._perspective_pillow(img, src, 60, 60, np)
        diff = max(abs(a - b)
                   for pa, pb in zip(out_py.getdata(), out_np.getdata())
                   for a, b in zip(pa, pb))
        self.assertEqual(diff, 0)

    def test_perspective_zero_deps_geometry(self):
        img = _shapes(64)
        with self._zero():
            out = C.perspective_transform(
                img, corners=[(0, 0), (64, 0), (64, 64), (0, 64)])
        # output size follows the corner geometry; mode preserved
        self.assertEqual(out.size, (64, 64))
        self.assertEqual(out.mode, "RGB")

    def test_inpaint_pillow_fills_from_surroundings(self):
        img = Image.new("RGB", (48, 48), (200, 30, 30))
        mask = _box_mask((48, 48), (20, 20, 28, 28))
        with self._zero():
            out = C.inpaint_cv(img, mask)
        # solid surround -> filled region matches the surround
        self.assertEqual(out.getpixel((24, 24)), (200, 30, 30))

    def test_dehaze_pillow_adds_contrast(self):
        # synthetic haze: blend a scene 50% toward white
        scene = _noisy_rgb(seed=31, size=48)
        hazy = Image.blend(scene, Image.new("RGB", (48, 48), "white"), 0.5)
        with self._zero():
            out = C.dehaze(hazy)
        self.assertEqual(out.size, (48, 48))
        before = hazy.convert("L").getextrema()
        after = out.convert("L").getextrema()
        self.assertGreater(after[1] - after[0], before[1] - before[0])

    def test_seam_carve_python_shrinks(self):
        img = _shapes(64)
        with self._zero():
            out = C.seam_carve(img, width=52, height=56)
        self.assertEqual(out.size, (52, 56))

    def test_cube_lut_python_identity(self):
        img = _noisy_rgb(seed=41, size=32)
        with self._zero():
            out = C.apply_cube_lut(img, self._identity_cube())
        diff = [abs(a - b)
                for px_a, px_b in zip(img.getdata(), out.getdata())
                for a, b in zip(px_a, px_b)]
        self.assertLess(max(diff), 3)

    def test_adaptive_threshold_pillow_is_binary(self):
        with self._zero():
            for method in ("otsu", "sauvola", "niblack"):
                out = C.adaptive_threshold(_shapes(64), method=method)
                self.assertEqual(out.mode, "L")
                self.assertTrue(
                    set(out.getdata()) <= {0, 255}, method)

    def test_match_hist_pillow_moves_distribution(self):
        src = Image.new("RGB", (48, 48), (200, 100, 50))
        ref = Image.new("RGB", (48, 48), (20, 60, 200))
        with self._zero():
            out = C.match_histograms(src, ref)
        r, g, b = out.split()
        self.assertGreater(sum(b.getdata()), sum(r.getdata()))

    def test_clarity_pillow_boosts_edges(self):
        # midtone edges (no clipping) so the boost is visible
        img = Image.new("RGB", (64, 64), (128, 128, 128))
        d = ImageDraw.Draw(img)
        d.rectangle([20, 20, 44, 44], fill=(100, 100, 100))
        d.rectangle([44, 44, 60, 60], fill=(160, 160, 160))
        with self._zero():
            out = C.clarity(img, amount=0.8, radius=8)
        self.assertEqual(out.size, img.size)
        self.assertNotEqual(list(out.getdata()), list(img.getdata()))

    def test_retouch_pillow_reduces_noise(self):
        noisy = _noisy_rgb(seed=51, size=48)
        with self._zero():
            out = C.retouch_smooth(noisy, radius=4.0, amount=0.8)
        self.assertEqual(out.size, noisy.size)
        import PIL.ImageChops as IC
        # smoothed image differs from pure noise
        self.assertIsNotNone(IC.difference(noisy, out).getbbox())

    def test_orb_pillow_marks_corners(self):
        with self._zero():
            out = C.orb_features(_shapes(96), n=15)
            count = C.orb_count(_shapes(96), n=15)
        self.assertEqual(out.size, (96, 96))
        self.assertGreaterEqual(count, 0)
        self.assertLessEqual(count, 15)

    def test_superpixels_grid_zero_deps(self):
        with self._zero():
            out = C.superpixels(_shapes(64), n_segments=16)
        self.assertEqual(out.size, (64, 64))
        # grid lines drawn in red
        reds = sum(1 for px in out.getdata() if px == (255, 0, 0))
        self.assertGreater(reds, 0)

    def test_stylize_tonemap_detail_deblur_zero_deps(self):
        img = _shapes(64)
        with self._zero():
            for fn in (C.stylize, C.tone_map, C.detail_enhance, C.deblur):
                out = fn(img)
                self.assertIsInstance(out, Image.Image)
                self.assertEqual(out.size, (64, 64))


def _real_skimage():
    import skimage
    return skimage


def _real_numpy():
    return np


if __name__ == "__main__":
    unittest.main()


class FullCapabilityTests(unittest.TestCase):
    """The extended op set: detection, deconvolution, segmentation, color."""

    def _force(self, **overrides):
        return _force_backends(**overrides)

    def _textured(self, size: int = 128):
        rng = np.random.default_rng(11)
        img = Image.fromarray(
            rng.integers(0, 256, (size, size, 3)).astype("uint8"), "RGB")
        d = ImageDraw.Draw(img)
        d.rectangle([20, 20, 60, 60], fill="red")
        d.ellipse([70, 70, 110, 110], fill="blue")
        return img

    # -- ORB features -------------------------------------------------
    def test_orb_features(self):
        out = C.orb_features(self._textured(), n=200)
        self.assertIsInstance(out, Image.Image)
        self.assertEqual(out.size, (128, 128))

    def test_orb_count(self):
        n = C.orb_count(self._textured(), n=200)
        self.assertIsInstance(n, int)
        self.assertGreater(n, 0)

    def test_orb_bad_n(self):
        with self.assertRaises(MediaEditError):
            C.orb_features(self._textured(), n=0)

    def test_orb_no_cv2_falls_back(self):
        with self._force(cv2=None):
            n = C.orb_count(self._textured())
            self.assertIsInstance(n, int)
            out = C.orb_features(self._textured(), n=50)
            self.assertIsInstance(out, Image.Image)

    # -- kmeans -------------------------------------------------------
    def test_kmeans_quantize(self):
        out = C.kmeans_quantize(self._textured(), k=4)
        uniq = np.unique(np.asarray(out).reshape(-1, 3), axis=0)
        self.assertLessEqual(len(uniq), 4)

    def test_kmeans_pillow_fallback(self):
        with self._force(cv2=None):
            out = C.kmeans_quantize(self._textured(), k=4)
            uniq = np.unique(np.asarray(out).reshape(-1, 3), axis=0)
            self.assertLessEqual(len(uniq), 4)

    def test_kmeans_bad_k(self):
        with self.assertRaises(MediaEditError):
            C.kmeans_quantize(self._textured(), k=1)
        with self.assertRaises(MediaEditError):
            C.kmeans_quantize(self._textured(), k=300)

    # -- deblur -------------------------------------------------------
    def test_deblur_wiener(self):
        out = C.deblur(self._textured(48), psf_size=5, method="wiener")
        self.assertEqual(out.size, (48, 48))

    def test_deblur_richardson_lucy(self):
        out = C.deblur(self._textured(32), psf_size=3,
                       method="richardson_lucy", balance=0.05)
        self.assertEqual(out.size, (32, 32))

    def test_deblur_bad_method(self):
        with self.assertRaises(MediaEditError):
            C.deblur(self._textured(32), method="nope")

    def test_deblur_no_skimage_falls_back(self):
        with self._force(skimage=None):
            out = C.deblur(self._textured(32))
            self.assertEqual(out.size, (32, 32))

    # -- superpixels --------------------------------------------------
    def test_superpixels_overlay(self):
        out = C.superpixels(self._textured(96), n_segments=25)
        self.assertEqual(out.size, (96, 96))
        self.assertEqual(out.mode, "RGB")

    def test_superpixels_mean_color(self):
        out = C.superpixels(self._textured(64), n_segments=16,
                            overlay=False)
        self.assertEqual(out.size, (64, 64))

    def test_superpixels_bad_n(self):
        with self.assertRaises(MediaEditError):
            C.superpixels(self._textured(32), n_segments=1)

    def test_superpixels_no_skimage_falls_back(self):
        with self._force(skimage=None):
            out = C.superpixels(self._textured(32), n_segments=8)
            self.assertEqual(out.size, (32, 32))

    # -- white balance ------------------------------------------------
    def test_white_balance_grayworld(self):
        cast = Image.new("RGB", (64, 64), (200, 100, 100))
        out = C.white_balance(cast, method="grayworld")
        means = np.asarray(out).mean(axis=(0, 1))
        self.assertLess(max(means) - min(means), 5)

    def test_white_balance_maxwhite(self):
        out = C.white_balance(Image.new("RGB", (32, 32), (100, 150, 200)),
                              method="maxwhite")
        self.assertEqual(out.size, (32, 32))

    def test_white_balance_bad_method(self):
        with self.assertRaises(MediaEditError):
            C.white_balance(self._textured(16), method="vintage")

    def test_white_balance_needs_no_numpy(self):
        # pure-Pillow path: works with every backend forced off
        with self._force(cv2=None, skimage=None, numpy=None):
            out = C.white_balance(Image.new("RGB", (32, 32), (200, 100, 50)))
            self.assertEqual(out.size, (32, 32))

    # -- auto levels --------------------------------------------------
    def test_auto_levels_stretches(self):
        grad = Image.new("L", (64, 1))
        grad.putdata([100 + int(20 * (x / 63)) for x in range(64)])
        flat = grad.resize((64, 64)).convert("RGB")
        before = np.asarray(flat).astype(float).std()
        after = np.asarray(C.auto_levels(flat)).astype(float).std()
        self.assertGreater(after, before * 2)

    def test_auto_levels_bad_cutoff(self):
        with self.assertRaises(MediaEditError):
            C.auto_levels(self._textured(16), cutoff=60)

    # -- color transfer -----------------------------------------------
    def test_color_transfer(self):
        src = Image.new("RGB", (48, 48), (200, 100, 50))
        ref = Image.new("RGB", (48, 48), (20, 60, 200))
        out = C.color_transfer(src, ref)
        self.assertEqual(out.size, (48, 48))
        mean = np.asarray(out).mean(axis=(0, 1))
        self.assertGreater(mean[2], mean[0])

    def test_color_transfer_pillow_fallback(self):
        with self._force(cv2=None):
            src = Image.new("RGB", (48, 48), (200, 100, 50))
            ref = Image.new("RGB", (48, 48), (20, 60, 200))
            out = C.color_transfer(src, ref)
            self.assertEqual(out.size, (48, 48))

    # -- panorama -----------------------------------------------------
    def _overlap_pair(self):
        rng = np.random.default_rng(5)
        base = Image.fromarray(
            rng.integers(0, 256, (160, 320, 3)).astype("uint8"), "RGB")
        d = ImageDraw.Draw(base)
        d.rectangle([40, 40, 120, 120], fill="red")
        d.ellipse([180, 40, 280, 120], fill="blue")
        return base.crop((0, 0, 220, 160)), base.crop((100, 0, 320, 160))

    def test_panorama(self):
        a, b = self._overlap_pair()
        out = C.panorama(a, images=[b])
        self.assertIsInstance(out, Image.Image)
        w, _ = out.size
        self.assertGreater(w, 220)  # wider than either input

    def test_panorama_needs_two(self):
        with self.assertRaises(MediaEditError):
            C.panorama(self._textured(64), images=[])

    def test_panorama_bad_mode(self):
        a, b = self._overlap_pair()
        with self.assertRaises(MediaEditError):
            C.panorama(a, images=[b], mode="cylinder")

    def test_panorama_no_cv2_falls_back(self):
        a, b = self._overlap_pair()
        with self._force(cv2=None):
            out = C.panorama(a, images=[b])
            self.assertIsInstance(out, Image.Image)
            self.assertGreaterEqual(out.width, a.width)

    def test_panorama_op_chain(self):
        a, b = self._overlap_pair()
        out = apply_chain(a, [{"op": "panorama", "images": [b]}])
        self.assertIsInstance(out, Image.Image)
        self.assertGreater(out.size[0], 220)

    # -- backend status covers everything ------------------------------
    def test_backend_status_all_ops(self):
        st = C.backend_status()
        self.assertEqual(len(st["selected"]), 35)
        self.assertEqual(st["selected"]["white_balance"], "pillow")
        self.assertEqual(st["selected"]["auto_levels"], "pillow")
        self.assertEqual(st["selected"]["tilt_shift"], "pillow")


class ProBatchTests(unittest.TestCase):
    """Pro retouch / color / compositing ops (free path vs paid editors)."""

    def _force(self, **overrides):
        return _force_backends(**overrides)

    def _photo(self, w: int = 160, h: int = 120, seed: int = 21):
        rng = np.random.default_rng(seed)
        img = Image.fromarray(
            rng.integers(0, 256, (h, w, 3)).astype("uint8"), "RGB")
        d = ImageDraw.Draw(img)
        d.rectangle([30, 30, 90, 90], fill=(200, 60, 60))
        return img

    def _cube(self, size: int = 2) -> str:
        lines = ["TITLE \"test\"", f"LUT_3D_SIZE {size}"]
        for b in range(size):
            for g in range(size):
                for r in range(size):
                    lines.append(f"{r/(size-1)} {g/(size-1)} {b/(size-1)}")
        return "\n".join(lines) + "\n"

    # -- retouch_smooth ------------------------------------------------
    def test_retouch_smooth(self):
        out = C.retouch_smooth(self._photo(), radius=6, amount=0.7)
        self.assertEqual(out.size, (160, 120))
        # smoothing reduces variance vs the noisy source
        self.assertLess(np.asarray(out).astype(float).var(),
                        np.asarray(self._photo()).astype(float).var())

    def test_retouch_smooth_pillow_fallback(self):
        with self._force(cv2=None, numpy=_real_numpy()):
            out = C.retouch_smooth(self._photo(), radius=4)
            self.assertEqual(out.size, (160, 120))

    def test_retouch_smooth_bad_params(self):
        with self.assertRaises(MediaEditError):
            C.retouch_smooth(self._photo(), amount=2.0)
        with self.assertRaises(MediaEditError):
            C.retouch_smooth(self._photo(), radius=0)

    # -- tone_map -------------------------------------------------------
    def test_tone_map(self):
        for method in ("mantiuk", "drago", "reinhard"):
            out = C.tone_map(self._photo(), method=method)
            self.assertEqual(out.size, (160, 120), method)

    def test_tone_map_bad_method(self):
        with self.assertRaises(MediaEditError):
            C.tone_map(self._photo(), method="nope")

    def test_tone_map_no_cv2_falls_back(self):
        with self._force(cv2=None):
            out = C.tone_map(self._photo())
            self.assertEqual(out.size, (160, 120))

    # -- detail_enhance / stylize ---------------------------------------
    def test_detail_enhance(self):
        out = C.detail_enhance(self._photo())
        self.assertEqual(out.size, (160, 120))

    def test_detail_enhance_no_cv2_falls_back(self):
        with self._force(cv2=None):
            out = C.detail_enhance(self._photo())
            self.assertEqual(out.size, (160, 120))

    def test_stylize(self):
        out = C.stylize(self._photo())
        self.assertEqual(out.size, (160, 120))

    # -- clarity ----------------------------------------------------------
    def test_clarity_boosts_local_contrast(self):
        img = self._photo()
        base = np.asarray(img).astype(float)
        out = np.asarray(C.clarity(img, amount=0.8)).astype(float)
        self.assertGreater(np.abs(out - base).mean(), 0.1)

    def test_clarity_negative_softens(self):
        out = C.clarity(self._photo(), amount=-0.5)
        self.assertEqual(out.size, (160, 120))

    def test_clarity_bad_amount(self):
        with self.assertRaises(MediaEditError):
            C.clarity(self._photo(), amount=1.5)

    # -- shadow_highlight -------------------------------------------------
    def test_shadow_highlight_lifts_shadows(self):
        dark = Image.new("RGB", (48, 48), (30, 30, 30))
        out = C.shadow_highlight(dark, shadows=0.8, highlights=0.0)
        self.assertGreater(np.asarray(out).mean(), 30)

    def test_shadow_highlight_pillow_fallback(self):
        with self._force(cv2=None, skimage=None, numpy=None):
            out = C.shadow_highlight(Image.new("RGB", (32, 32), (40, 40, 40)))
            self.assertEqual(out.size, (32, 32))

    def test_shadow_highlight_bad_params(self):
        with self.assertRaises(MediaEditError):
            C.shadow_highlight(self._photo(), shadows=2.0)

    # -- dehaze -------------------------------------------------------------
    def test_dehaze(self):
        # hazy = blended toward white
        img = self._photo()
        hazy = Image.blend(img, Image.new("RGB", img.size, "white"), 0.4)
        out = C.dehaze(hazy)
        self.assertEqual(out.size, img.size)
        # dehazed should be darker / more saturated than hazy input
        self.assertLess(np.asarray(out).astype(float).mean(),
                        np.asarray(hazy).astype(float).mean())

    def test_dehaze_bad_omega(self):
        with self.assertRaises(MediaEditError):
            C.dehaze(self._photo(), omega=0)

    # -- seam_carve -----------------------------------------------------------
    def test_seam_carve_width(self):
        out = C.seam_carve(self._photo(), width=120)
        self.assertEqual(out.size, (120, 120))

    def test_seam_carve_height(self):
        out = C.seam_carve(self._photo(), height=90)
        self.assertEqual(out.size, (160, 90))

    def test_seam_carve_preserves_subject(self):
        # red square should survive better than naive scaling would allow
        img = Image.new("RGB", (120, 60), "skyblue")
        ImageDraw.Draw(img).rectangle([45, 15, 75, 45], fill="red")
        out = C.seam_carve(img, width=80)
        px = np.asarray(out)
        red = (px[:, :, 0] > 150) & (px[:, :, 1] < 100)
        self.assertGreater(red.sum(), 50)

    def test_seam_carve_no_grow(self):
        with self.assertRaises(MediaEditError):
            C.seam_carve(self._photo(), width=999)

    # -- tilt_shift -------------------------------------------------------------
    def test_tilt_shift(self):
        out = C.tilt_shift(self._photo())
        self.assertEqual(out.size, (160, 120))
        # edges should be blurrier than the center band
        arr = np.asarray(out).astype(float)
        edge_var = np.abs(np.diff(arr[:10], axis=0)).mean()
        mid_var = np.abs(np.diff(arr[55:65], axis=0)).mean()
        self.assertLess(edge_var, mid_var)

    def test_tilt_shift_bad_params(self):
        with self.assertRaises(MediaEditError):
            C.tilt_shift(self._photo(), focus_center=2.0)

    # -- selective_color ----------------------------------------------------------
    def test_selective_color(self):
        img = Image.new("RGB", (64, 64), "blue")
        ImageDraw.Draw(img).rectangle([20, 20, 44, 44], fill="red")
        out = C.selective_color(img, hue=0, hue_width=40)
        px = np.asarray(out).astype(float)
        # red square stays saturated, blue background goes gray
        sq = px[28:36, 28:36]
        bg = px[0:8, 0:8]
        self.assertGreater(sq[:, :, 0].mean() - sq[:, :, 2].mean(), 40)
        self.assertLess(abs(bg[:, :, 0].mean() - bg[:, :, 2].mean()), 30)

    def test_selective_color_bad_hue(self):
        with self.assertRaises(MediaEditError):
            C.selective_color(self._photo(), hue=400)

    # -- split_tone -----------------------------------------------------------------
    def test_split_tone(self):
        grad = Image.new("L", (64, 64))
        grad.putdata([int(255 * (x / 63)) for y in range(64)
                      for x in range(64)])
        img = grad.convert("RGB")
        out = C.split_tone(img, shadows=(0, 80, 160),
                           highlights=(220, 180, 120), strength=0.8)
        px = np.asarray(out).astype(float)
        left = px[:, :16]    # shadows -> bluish
        right = px[:, 48:]   # highlights -> warm
        self.assertGreater(left[:, :, 2].mean(), left[:, :, 0].mean())
        self.assertGreater(right[:, :, 0].mean(), right[:, :, 2].mean())

    def test_split_tone_pillow_fallback(self):
        with self._force(cv2=None, skimage=None, numpy=None):
            out = C.split_tone(self._photo())
            self.assertEqual(out.size, (160, 120))

    def test_split_tone_bad_strength(self):
        with self.assertRaises(MediaEditError):
            C.split_tone(self._photo(), strength=2.0)

    # -- apply_curves ---------------------------------------------------------------
    def test_apply_curves_identity(self):
        img = self._photo()
        out = C.apply_curves(img, [(0, 0), (255, 255)])
        self.assertEqual(list(out.getdata()), list(img.getdata()))

    def test_apply_curves_brighten(self):
        img = Image.new("RGB", (16, 16), (100, 100, 100))
        out = C.apply_curves(img, [(0, 0), (128, 180), (255, 255)])
        self.assertGreater(np.asarray(out).mean(), 100)

    def test_apply_curves_channel(self):
        img = Image.new("RGB", (16, 16), (100, 100, 100))
        out = C.apply_curves(img, [(0, 0), (255, 200)], channel="r")
        px = np.asarray(out)
        self.assertTrue((px[:, :, 0] < px[:, :, 1]).all())

    def test_apply_curves_bad(self):
        with self.assertRaises(MediaEditError):
            C.apply_curves(self._photo(), [(0, 0), (999, 999)])
        with self.assertRaises(MediaEditError):
            C.apply_curves(self._photo(), channel="cmyk")

    # -- apply_cube_lut -----------------------------------------------------------------
    def test_apply_cube_lut_identity(self):
        img = self._photo(64, 48)
        out = C.apply_cube_lut(img, self._cube(4))
        diff = np.abs(np.asarray(out).astype(int)
                      - np.asarray(img).astype(int)).mean()
        self.assertLess(diff, 3.0)

    def test_apply_cube_lut_warm(self):
        # LUT that warms: map blue down
        lines = ["LUT_3D_SIZE 2"]
        for b in (0.0, 1.0):
            for g in (0.0, 1.0):
                for r in (0.0, 1.0):
                    lines.append(f"{r} {g} {b * 0.5}")
        img = Image.new("RGB", (32, 32), (100, 100, 200))
        out = C.apply_cube_lut(img, "\n".join(lines) + "\n")
        px = np.asarray(out).astype(float)
        self.assertLess(px[:, :, 2].mean(), 200)

    def test_apply_cube_lut_bad(self):
        with self.assertRaises(MediaEditError):
            C.apply_cube_lut(self._photo(32, 32), "not a lut at all")

    def test_apply_cube_lut_no_numpy_falls_back(self):
        with self._force(numpy=None):
            out = C.apply_cube_lut(self._photo(16, 16), self._cube())
            self.assertEqual(out.size, (16, 16))

    # -- replace_background ---------------------------------------------------------------
    def test_replace_background(self):
        img = Image.new("RGB", (128, 128), "skyblue")
        ImageDraw.Draw(img).ellipse([39, 39, 89, 89], fill="darkred")
        bg = Image.new("RGB", (64, 64), "navy")
        out = C.replace_background(img, bg, rect=(30, 30, 98, 98))
        self.assertEqual(out.size, (128, 128))
        # corners should now be navy-ish (new background showed through)
        px = np.asarray(out).astype(float)
        corner = px[0:10, 0:10]
        self.assertGreater(corner[:, :, 2].mean(), 60)

    def test_replace_background_no_cv2_falls_back(self):
        with self._force(cv2=None):
            out = C.replace_background(self._photo(64, 48),
                                       Image.new("RGB", (64, 48)),
                                       rect=(8, 8, 56, 40))
            self.assertEqual(out.size, (64, 48))

    # -- op chain integration -----------------------------------------------------------------
    def test_pro_chain_end_to_end(self):
        out = apply_chain(self._photo(), [
            {"op": "white_balance"},
            {"op": "shadow_highlight", "shadows": 0.3, "highlights": 0.2},
            {"op": "clarity", "amount": 0.4},
            {"op": "curves", "points": [[0, 0], [255, 255]]},
        ])
        self.assertIsInstance(out, Image.Image)
        self.assertEqual(out.size, (160, 120))
