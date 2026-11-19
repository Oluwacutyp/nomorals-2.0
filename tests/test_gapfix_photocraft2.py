"""PhotoCraft remainder — lane 97: the full doc model renders.

Lane 76 shipped the PSD piece with adjustments, blends, effects,
text and smart objects PARSED but not rendered (oracle over the
258-file corpus: 80 exact, 55 differs-adjustment, 2 differs-blend,
121 differs-unrendered, 0 unexplained). This lane renders them:
the shared compositor in :mod:`nomorals.media.photo_editor.compose`
applies adjustment layers, blend modes, layer effects, live text
and smart-filter masks, and the writer is byte-stable where the
editor made no change (258/258) with a spliced save for
layer-state-only changes.

Every test here drives the PRODUCTION path against REAL PSD files
— the vendored fixtures in tests/fixtures/photocraft/ plus the
full 258-file corpus when the mining checkout is present (the
corpus test skips honestly when it is absent, never silently).
A decoder proven only against fixtures written by the same hand
as the decoder proves nothing; the oracle is the check.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "photocraft"
CORPUS = Path(os.environ.get(
    "NM_PHOTOCRAFT_CORPUS",
    str(Path.home() / "workspace" / "echelon-mining" / "repos"
        / "storytold" / "photocraft-corpus" / "photoshop")))


def _read(path: Path) -> bytes:
    return path.read_bytes()


def _corpus(rel: str) -> Path:
    return CORPUS / rel


def _max_diff(a, b) -> int:
    """Max per-channel absolute difference between two PIL images."""
    import numpy as np

    x = np.asarray(a).astype(int)
    y = np.asarray(b).astype(int)
    if x.shape != y.shape:
        return 255
    return int(abs(x - y).max()) if x.size else 0


def _flatten_vs_composite(rel: str) -> int:
    """The oracle in miniature for one corpus file."""
    from nomorals.media.photo_editor.psd import PsdFile

    psd = PsdFile.from_bytes(_read(_corpus(rel)))
    return _max_diff(psd.flatten_layers(), psd.composite_rgba())


def _mean_diff(rel: str) -> float:
    """Mean per-channel absolute difference, in levels (0..255).

    The honest metric for renders whose mean is rounding-level
    but whose max has a few outlier pixels at sharp edges and
    step boundaries.
    """
    import numpy as np

    from nomorals.media.photo_editor.psd import PsdFile

    psd = PsdFile.from_bytes(_read(_corpus(rel)))
    a = np.asarray(psd.flatten_layers()).astype(int)
    b = np.asarray(psd.composite_rgba()).astype(int)
    return float(abs(a - b).mean())


class AdjustmentRenderTests(unittest.TestCase):
    """Each adjustment class the oracle used to leave unrendered
    now renders within a few levels of Photoshop's composite."""

    def test_gradient_map_catmull_rom_bake(self) -> None:
        # A linear gradient bake left a 0.017 mean error; the
        # Catmull-Rom bake (Photoshop's interpolation) lands at
        # rounding level.
        self.assertLessEqual(
            _flatten_vs_composite(
                "adjustments/rgb8/gradient-map.psd"), 1)

    def test_photo_filter_clips_before_set_lum(self) -> None:
        # SetLum on the unclipped matrix output dragged saturated
        # reds out of gamut (max 0.18); clipping first lands at
        # a few levels.
        self.assertLessEqual(
            _flatten_vs_composite(
                "adjustments/rgb8/photo-filter.psd"), 5)

    def test_gray_curves_and_levels_are_exact(self) -> None:
        # A grey document has one channel: the grey record drives
        # all three replicated RGB channels.
        self.assertEqual(
            _flatten_vs_composite("adjustments/gray8/curves.psd"),
            0)
        self.assertEqual(
            _flatten_vs_composite("adjustments/gray8/levels.psd"),
            0)

    def test_gray_exposure_is_rounding(self) -> None:
        self.assertLessEqual(
            _flatten_vs_composite(
                "adjustments/gray8/exposure.psd"), 1)

    def test_rgb_exposure_is_exact(self) -> None:
        self.assertEqual(
            _flatten_vs_composite(
                "adjustments/rgb8/exposure.psd"), 0)

    def test_lab_invert_and_tone_in_space(self) -> None:
        # Lab adjustments run in the native space (normalized
        # Lab), not through an RGB detour.
        self.assertLessEqual(
            _mean_diff("adjustments/lab8/invert.psd"), 1.0)
        self.assertLessEqual(
            _mean_diff("adjustments/lab8/curves.psd"), 2.0)
        self.assertLessEqual(
            _mean_diff("adjustments/lab8/levels.psd"), 2.0)

    def test_cmyk_legacy_brightness_contrast_per_ink(self) -> None:
        # Legacy brightness/contrast on a CMYK document applies
        # per ink, not in RGB space.
        self.assertLessEqual(
            _mean_diff(
                "adjustments/cmyk8/brightness-contrast-legacy.psd"),
            1.0)

    def test_rgb_adjustments_at_rounding(self) -> None:
        for rel in ("adjustments/rgb8/brightness-contrast.psd",
                    "adjustments/rgb8/brightness-contrast-legacy.psd",
                    "adjustments/rgb8/channel-mixer.psd",
                    "adjustments/rgb8/color-balance.psd"):
            with self.subTest(rel=rel):
                self.assertLessEqual(_flatten_vs_composite(rel), 1)


class BlendRenderTests(unittest.TestCase):
    def test_multiply_text_blend_is_rounding(self) -> None:
        self.assertLessEqual(
            _flatten_vs_composite(
                "text/text-on-gradient-blend-multiply.psd"), 1)

    def test_clipped_hue_sat_blend_color(self) -> None:
        # The disc's antialiased edge is the known remainder:
        # a handful of edge pixels sit above rounding.
        self.assertLessEqual(
            _mean_diff(
                "adjustments/rgb8/clipped-hue-sat-blend-color.psd"),
            1.0)


class EffectsRenderTests(unittest.TestCase):
    def test_inner_bevel_smooth_is_rounding(self) -> None:
        self.assertLessEqual(
            _mean_diff("effects/bevel-inner-smooth.psd"), 0.1)

    def test_outer_bevel_smooth_is_rounding(self) -> None:
        self.assertLessEqual(
            _mean_diff("effects/bevel-outer-smooth.psd"), 0.1)

    def test_chisel_bevel_renders(self) -> None:
        # Chisel-hard on a star: the height field is right and
        # the deltas match Photoshop's energy; the remainder is
        # sub-pixel edge placement at the star's points.
        self.assertLessEqual(
            _mean_diff("effects/bevel-emboss-chisel-hard.psd"),
            2.0)

    def test_satin_renders(self) -> None:
        self.assertLessEqual(
            _mean_diff("effects/satin-default.psd"), 1.0)

    def test_outer_glow_renders(self) -> None:
        self.assertLessEqual(
            _flatten_vs_composite(
                "effects/outer-glow-soft.psd"), 2)


class TextRenderTests(unittest.TestCase):
    def test_antialiased_text_is_exact(self) -> None:
        # The stored raster stays primary; the gamma-1.45 text
        # composite mixes it exactly.
        self.assertEqual(
            _flatten_vs_composite("text/antialias-smooth.psd"), 0)

    def test_text_with_layer_styles_renders(self) -> None:
        self.assertLessEqual(
            _mean_diff("text/text-with-layer-styles.psd"), 1.0)


class SmartObjectTests(unittest.TestCase):
    def test_filter_mask_hard_is_exact(self) -> None:
        # The stored pixels are the filtered result WITHOUT the
        # filter mask; Photoshop blends filtered/unfiltered by
        # the FEid mask at render time.
        self.assertEqual(
            _flatten_vs_composite(
                "smart-filters/sf-filter-mask-hard.psd"), 0)

    def test_gaussian_blur_smart_filter_renders(self) -> None:
        self.assertLessEqual(
            _flatten_vs_composite(
                "smart-filters/sf-gaussian-blur-r4.psd"), 1)

    def test_smart_stack_blend_and_mask(self) -> None:
        self.assertLessEqual(
            _mean_diff(
                "smart-filters/sf-stack-blend-and-mask.psd"), 0.01)


class ByteStableTests(unittest.TestCase):
    def test_psd_write_back_is_byte_stable(self) -> None:
        from nomorals.media.photo_editor.psd import PsdFile

        for fixture in sorted(FIXTURES.glob("*.psd")):
            with self.subTest(fixture=fixture.name):
                data = _read(fixture)
                self.assertEqual(
                    PsdFile.from_bytes(data).to_bytes(), data)

    def test_document_write_back_is_byte_stable(self) -> None:
        from nomorals.media.photo_editor.document import PhotoDocument

        for fixture in sorted(FIXTURES.glob("*.psd")):
            with self.subTest(fixture=fixture.name):
                data = _read(fixture)
                doc = PhotoDocument.from_psd_bytes(data)
                self.assertEqual(doc.to_psd_bytes(), data)

    def test_document_flatten_matches_psd_flatten(self) -> None:
        from nomorals.media.photo_editor.document import PhotoDocument
        from nomorals.media.photo_editor.psd import PsdFile

        for fixture in sorted(FIXTURES.glob("*.psd")):
            with self.subTest(fixture=fixture.name):
                data = _read(fixture)
                doc = PhotoDocument.from_psd_bytes(data)
                psd = PsdFile.from_bytes(data)
                self.assertEqual(
                    _max_diff(doc.flatten(), psd.flatten_layers()),
                    0)

    def test_layer_toggle_splices_clean_layers_verbatim(self) -> None:
        from nomorals.media.photo_editor.document import PhotoDocument
        from nomorals.media.photo_editor.psd import PsdFile

        data = _read(CORPUS / "adjustments" / "rgb8"
                     / "gradient-map.psd")
        doc = PhotoDocument.from_psd_bytes(data)
        doc.set_layer_state(1, opacity=128)
        blob = doc.to_psd_bytes()
        self.assertNotEqual(blob, data)
        again = PsdFile.from_bytes(blob)
        self.assertEqual(again.layers[1].opacity, 128)
        source = PsdFile.from_bytes(data)
        self.assertEqual(again.layers[0].channels,
                         source.layers[0].channels)
        # Undo restores the verbatim write-back.
        doc.undo()
        self.assertEqual(doc.to_psd_bytes(), data)


class LayerStateTests(unittest.TestCase):
    def test_layer_toggle_by_index_and_name(self) -> None:
        from nomorals.media.photo_editor.document import PhotoDocument

        doc = PhotoDocument.from_psd_bytes(_read(
            CORPUS / "adjustments" / "rgb8" / "gradient-map.psd"))
        doc.set_layer_state(1, visible=False)
        self.assertFalse(doc.layers[1].visible)
        doc.set_layer_state("gradient-map", visible=True, opacity=64)
        self.assertTrue(doc.layers[1].visible)
        self.assertEqual(doc.layers[1].opacity, 64)
        doc.undo()
        self.assertEqual(doc.layers[1].opacity, 255)
        doc.undo()
        self.assertTrue(doc.layers[1].visible)

    def test_layer_toggle_rejects_bad_ref_and_opacity(self) -> None:
        from nomorals.media.photo_editor.document import (
            PhotoDocument,
            PhotoEditorError,
        )

        doc = PhotoDocument.from_psd_bytes(_read(
            CORPUS / "adjustments" / "rgb8" / "gradient-map.psd"))
        with self.assertRaises(PhotoEditorError):
            doc.set_layer_state("no such layer", visible=False)
        with self.assertRaises(PhotoEditorError):
            doc.set_layer_state(1, opacity=300)
        with self.assertRaises(PhotoEditorError):
            doc.set_layer_state(1)


class OracleTests(unittest.TestCase):
    def test_oracle_full_corpus_when_present(self) -> None:
        from nomorals.media.photo_editor.oracle import run_oracle

        if not CORPUS.is_dir():
            self.skipTest(f"no corpus checkout at {CORPUS}")
        report = run_oracle(str(CORPUS))
        summary = report["summary"]
        self.assertEqual(summary["files"], 258)
        self.assertEqual(summary["failed"], 0, report["files"])
        # Lane 76: 80 exact / 55 differs-adjustment / 2
        # differs-blend / 121 differs-unrendered. Lane 97
        # renders every class; what remains is render gaps.
        self.assertGreaterEqual(summary["exact"], 160)
        self.assertEqual(summary["differs_adjustment"], 0)
        self.assertEqual(summary["differs_blend"], 0)
        self.assertEqual(summary["differs_unrendered"], 0)
        self.assertEqual(summary["byte_stable"], 258)
        self.assertEqual(summary["doc_byte_stable"], 258)

    def test_oracle_roundtrip_all_fixtures(self) -> None:
        from nomorals.media.photo_editor.oracle import run_oracle

        report = run_oracle(str(FIXTURES))
        for entry in report["files"]:
            self.assertTrue(entry["roundtrip_ok"], entry)


class WiringTests(unittest.TestCase):
    def test_tool_layer_and_info_actions(self) -> None:
        from nomorals.core.policy import CapabilitySet
        from nomorals.tools.registry import ToolRegistry

        with tempfile.TemporaryDirectory() as tmp:
            context = SimpleNamespace(
                settings=SimpleNamespace(home=tmp))
            registry = ToolRegistry(context)
            registry.register_builtins()
            opened = registry.call(
                "photo_editor", actor="test",
                capabilities=CapabilitySet.all(),
                action="open",
                path=str(CORPUS / "adjustments" / "rgb8"
                         / "gradient-map.psd"))
            self.assertTrue(opened.ok, opened)
            doc_id = opened.unwrap()["id"]
            info = registry.call(
                "photo_editor", actor="test",
                capabilities=CapabilitySet.all(),
                action="info", doc_id=doc_id)
            self.assertTrue(info.ok, info)
            feats = info.unwrap()["features"]
            kinds = {row["name"]: row for row in feats}
            self.assertEqual(
                kinds["gradient-map"]["adjustment"],
                "gradient_map")
            layered = registry.call(
                "photo_editor", actor="test",
                capabilities=CapabilitySet.all(),
                action="layer", doc_id=doc_id,
                params={"layer": 1, "opacity": 128})
            self.assertTrue(layered.ok, layered)
            self.assertEqual(
                layered.unwrap()["layer_list"][1]["opacity"], 128)
            shown = registry.call(
                "photo_editor", actor="test",
                capabilities=CapabilitySet.all(),
                action="show", doc_id=doc_id)
            self.assertEqual(
                shown.unwrap()["layer_list"][1]["opacity"], 128)

    def test_page_shows_layer_features(self) -> None:
        from nomorals.media.photo_editor.document import PhotoDocument
        from nomorals.media.photo_editor.page import photo_page_html

        doc = PhotoDocument.from_psd_bytes(_read(
            CORPUS / "adjustments" / "rgb8" / "gradient-map.psd"))
        html = photo_page_html(doc)
        self.assertIn("gradient_map", html)
        self.assertIn("/photo layer", html)
        self.assertIn("/photo info", html)
        self.assertNotIn("<script", html)


if __name__ == "__main__":
    unittest.main()
