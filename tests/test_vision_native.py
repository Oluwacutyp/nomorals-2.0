"""Native vision tests: Devon's own eyes, no model, no network.

Covers nomorals/vision/native.py (perceptual hashing, color analysis,
quality metrics, EXIF, native compare, template matching, graceful
unavailability) and the strategy-chain upgrades in nomorals/tools/vision.py
(read_text strategies, template locate, offline compare, analyze,
capabilities).

All images are synthesized with Pillow — no fixtures, no network.
"""

from __future__ import annotations

import io
import unittest
from types import SimpleNamespace
from typing import Any

from PIL import Image, ImageDraw

from nomorals.core.errors import ToolError
from nomorals.tools import vision as V
from nomorals.vision import native as N


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _solid(color: tuple[int, int, int] = (255, 0, 0),
           size: tuple[int, int] = (64, 48)) -> bytes:
    return _png(Image.new("RGB", size, color))


def _scene() -> tuple[bytes, bytes]:
    """Big image + an exact template cropped from it."""
    big = Image.new("RGB", (320, 240), (20, 20, 20))
    draw = ImageDraw.Draw(big)
    draw.rectangle([100, 60, 140, 100], fill=(30, 144, 255))
    draw.rectangle([105, 65, 115, 75], fill=(255, 255, 0))
    template = big.crop((100, 60, 140, 100))
    return _png(big), _png(template)


def _context(router: Any = None) -> SimpleNamespace:
    import tempfile

    settings = SimpleNamespace(
        partner=SimpleNamespace(vision=SimpleNamespace(
            allow_screenshot=False, max_dimension=1568,
            max_image_bytes=25 * 1024 * 1024, log_calls=False,
            enabled=True, ocr_binary="", ocr_language="eng")),
        workspace_dir=tempfile.mkdtemp())
    return SimpleNamespace(settings=settings, router=router,
                           extras={"attachments": []})


class HashTests(unittest.TestCase):
    def test_dhash_deterministic(self) -> None:
        data = _solid()
        self.assertEqual(N.phash(data), N.phash(data))
        self.assertRegex(N.phash(data), r"^[0-9a-f]{16}$")

    def test_ahash_differs_from_dhash_shape(self) -> None:
        self.assertRegex(N.phash(_solid(), "ahash"), r"^[0-9a-f]{16}$")

    def test_distance_zero_for_identical(self) -> None:
        h = N.phash(_solid())
        self.assertEqual(N.hash_distance(h, h), 0)

    def test_distance_large_for_different(self) -> None:
        # solid red vs solid blue: dhash of flat images is 0 for both —
        # use textured images so the hashes actually differ
        a = _png(Image.effect_noise((32, 32), 64).convert("RGB"))
        b = _png(Image.effect_noise((32, 32), 200).convert("RGB"))
        dist = N.hash_distance(N.phash(a), N.phash(b))
        self.assertGreater(dist, 0)

    def test_distance_rejects_garbage(self) -> None:
        with self.assertRaises(ToolError):
            N.hash_distance("not-a-hash", "00" * 8)


class ColorTests(unittest.TestCase):
    def test_solid_red_dominant(self) -> None:
        colors = N.color_analysis(_solid((255, 0, 0)))
        top = colors["dominant"][0]
        self.assertEqual(top["hex"], "#ff0000")
        self.assertAlmostEqual(top["share"], 1.0)
        self.assertEqual(colors["method"], "pil-mediancut-quantize")

    def test_brightness_contrast_saturation_ranges(self) -> None:
        colors = N.color_analysis(_solid((255, 0, 0)))
        self.assertGreater(colors["brightness"], 0)
        self.assertLess(colors["brightness"], 255)
        self.assertGreaterEqual(colors["contrast"], 0)
        self.assertGreater(colors["saturation_pct"], 50)


class QualityTests(unittest.TestCase):
    def test_quality_shape(self) -> None:
        q = N.quality_metrics(_solid())
        self.assertIn(q["sharpness_label"], ("sharp", "normal", "soft", "blurry"))
        self.assertGreaterEqual(q["entropy_bits"], 0)
        self.assertLessEqual(q["entropy_bits"], 8)

    def test_textured_sharper_than_flat(self) -> None:
        flat = N.quality_metrics(_solid())["sharpness"]
        noisy = N.quality_metrics(
            _png(Image.effect_noise((64, 48), 128).convert("RGB")))["sharpness"]
        self.assertGreater(noisy, flat)


class ExifTests(unittest.TestCase):
    def test_no_exif_reported(self) -> None:
        exif = N.exif_data(_solid())
        self.assertFalse(exif["present"])


class CompareNativeTests(unittest.TestCase):
    def test_identical(self) -> None:
        data = _solid()
        result = N.compare_native(data, data)
        self.assertTrue(result["identical"])
        self.assertEqual(result["changed_fraction"], 0.0)
        self.assertIsNone(result["changed_bbox_1000"])

    def test_changed_region_found(self) -> None:
        base = Image.new("RGB", (100, 100), (10, 10, 10))
        mod = base.copy()
        ImageDraw.Draw(mod).rectangle([20, 30, 60, 70], fill=(250, 250, 250))
        result = N.compare_native(_png(base), _png(mod))
        self.assertFalse(result["identical"])
        self.assertGreater(result["changed_fraction"], 0)
        bbox = result["changed_bbox_1000"]
        self.assertIsNotNone(bbox)
        # changed rect at px (20,30)-(60,70) → 0-1000 coords
        assert bbox is not None
        self.assertLessEqual(abs(bbox["x"] - 200), 60)
        self.assertLessEqual(abs(bbox["y"] - 300), 60)

    def test_bad_bytes_raise(self) -> None:
        with self.assertRaises(ToolError):
            N.compare_native(b"not an image", _solid())


class TemplateLocateTests(unittest.TestCase):
    def test_exact_template_found(self) -> None:
        big, template = _scene()
        result = N.template_locate(big, template)
        self.assertTrue(result["found"])
        self.assertAlmostEqual(result["score"], 1.0, places=3)
        self.assertEqual(result["method"], "template-match-ncc")
        # template at px (100,60) size 40x40 in 320x240
        self.assertLessEqual(abs(result["x"] - 312), 5)
        self.assertLessEqual(abs(result["y"] - 250), 5)

    def test_missing_template_not_found(self) -> None:
        big, _ = _scene()
        other = Image.new("RGB", (40, 40), (200, 30, 30))
        draw = ImageDraw.Draw(other)
        draw.ellipse([5, 5, 35, 35], fill=(10, 200, 10))
        result = N.template_locate(big, _png(other))
        self.assertFalse(result["found"])
        self.assertIn("not guessing", result["note"])

    def test_flat_template_rejected(self) -> None:
        big, _ = _scene()
        with self.assertRaises(ToolError):
            N.template_locate(big, _solid((128, 128, 128), (40, 40)))


class UnavailableBackendTests(unittest.TestCase):
    def test_faces_without_opencv(self) -> None:
        if N._backend("cv2") is not None:
            self.skipTest("cv2 installed — graceful path not exercised")
        with self.assertRaises(N.NativeUnavailable) as ctx:
            N.detect_faces(_solid())
        self.assertIn("opencv", str(ctx.exception).lower())
        self.assertEqual(ctx.exception.code, "vision.native_unavailable")

    def test_qr_without_pyzbar(self) -> None:
        if N._backend("pyzbar") is not None:
            self.skipTest("pyzbar installed")
        with self.assertRaises(N.NativeUnavailable):
            N.read_qr(_solid())

    def test_layout_without_tesseract(self) -> None:
        if N._tesseract_binary() is not None:
            self.skipTest("tesseract installed")
        with self.assertRaises(N.NativeUnavailable) as ctx:
            N.document_layout(_solid())
        self.assertIn("tesseract", str(ctx.exception).lower())

    def test_analyze_reports_unavailable_backends(self) -> None:
        report = N.analyze(_solid())
        self.assertTrue(report["colors"]["available"])
        self.assertTrue(report["hashes"]["available"])
        for section in ("faces", "qr"):
            sec = report[section]
            if N._backend("cv2" if section == "faces" else "pyzbar") is None:
                self.assertFalse(sec["available"])
                self.assertIn("why", sec)


class CapabilitiesTests(unittest.TestCase):
    def test_report_shape(self) -> None:
        caps = N.capabilities()
        self.assertIn("profile", caps)
        self.assertIn("backends", caps)
        self.assertIn("native", caps)
        self.assertIn("needs_model", caps)
        # honest about what still needs a model
        for key in ("describe", "semantic_compare", "word_locate",
                    "face_identity"):
            self.assertIn(key, caps["needs_model"])
        # metadata/colors/hashes/compare always available with PIL
        for key in ("metadata", "colors", "hashes", "compare"):
            self.assertTrue(caps["native"][key]["available"])

    def test_missing_backends_have_install_hints(self) -> None:
        caps = N.capabilities()
        for name, info in caps["native"].items():
            if not info["available"]:
                self.assertIn("why", info)


class ReadTextStrategyTests(unittest.TestCase):
    def test_native_strategy_without_tesseract_raises_clearly(self) -> None:
        if N._tesseract_binary() is not None:
            self.skipTest("tesseract installed")
        ctx = _context()
        with self.assertRaises(N.NativeUnavailable):
            V.read_text(ctx, _solid(), strategy="native")

    def test_bad_strategy_rejected(self) -> None:
        with self.assertRaises(ToolError):
            V.read_text(_context(), _solid(), strategy="bogus")

    def test_auto_chains_to_model_when_no_tesseract(self) -> None:
        if N._tesseract_binary() is not None:
            self.skipTest("tesseract installed")
        from nomorals.llm.base import LLMProvider, LLMResponse
        from nomorals.llm.router import LLMRouter

        class Stub(LLMProvider):
            name = "stub"

            @property
            def capabilities(self) -> set[str]:
                return {"vision"}

            def chat(self, messages, params=None, **kw):
                return LLMResponse(text="", model="stub")

            def describe_image(self, image, prompt="", params=None, **kw):
                return LLMResponse(text="HELLO", model="stub-vlm",
                                   provider="stub")

        router = LLMRouter()
        router.add(Stub(), primary=True, name="stub")
        result = V.read_text(_context(router), _solid(), strategy="auto")
        self.assertEqual(result["text"], "HELLO")
        self.assertEqual(result["method"], "vlm")


class LocateTemplateTests(unittest.TestCase):
    def test_template_locate_needs_no_router(self) -> None:
        big, template = _scene()
        result = V.locate(_context(), big, "the blue square", template=template)
        self.assertTrue(result["found"])
        self.assertEqual(result["method"], "template-match-ncc")
        self.assertAlmostEqual(result["score"], 1.0, places=3)


class CompareToolTests(unittest.TestCase):
    def test_offline_compare_is_native_and_honest(self) -> None:
        base = Image.new("RGB", (100, 100), (10, 10, 10))
        mod = base.copy()
        ImageDraw.Draw(mod).rectangle([20, 30, 60, 70], fill=(250, 250, 250))
        result = V.compare(_context(), _png(base), _png(mod))
        self.assertEqual(result["method"], "native")
        self.assertIn("native", result)
        self.assertGreater(result["native"]["changed_fraction"], 0)
        # no fake error-string description
        self.assertNotIn("[vision unavailable", result["description"])

    def test_compare_with_router_adds_semantic(self) -> None:
        from nomorals.llm.base import LLMProvider, LLMResponse
        from nomorals.llm.router import LLMRouter

        class Stub(LLMProvider):
            name = "stub"

            @property
            def capabilities(self) -> set[str]:
                return {"vision"}

            def chat(self, messages, params=None, **kw):
                return LLMResponse(text="", model="stub")

            def describe_image(self, image, prompt="", params=None, **kw):
                return LLMResponse(text="the right side changed", model="s",
                                   provider="stub")

        router = LLMRouter()
        router.add(Stub(), primary=True, name="stub")
        result = V.compare(_context(router), _solid(), _solid())
        self.assertEqual(result["method"], "side-by-side")
        self.assertIn("native", result)
        self.assertTrue(result["native"]["identical"])
        self.assertIn("changed", result["description"])


class AnalyzeToolTests(unittest.TestCase):
    def test_analyze_tool(self) -> None:
        result = V.analyze(_context(), _solid((0, 128, 255)))
        self.assertEqual(result["metadata"]["format"], "png")
        self.assertEqual(result["colors"]["dominant"][0]["hex"], "#0080ff")
        self.assertIn("seconds", result)

    def test_capabilities_tool(self) -> None:
        result = V.vision_capabilities(_context())
        self.assertIn("native", result)
        self.assertFalse(result["router_vision"])

    def test_new_tools_registered(self) -> None:
        from nomorals.tools.registry import ToolRegistry

        reg = ToolRegistry()
        V.register(reg)
        for name in ("vision_analyze", "vision_layout", "vision_capabilities"):
            self.assertIn(name, reg.names(), name)


if __name__ == "__main__":
    unittest.main()
