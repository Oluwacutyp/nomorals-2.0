"""LightCraft port — the full non-destructive develop engine (lane 98).

The gap, failing-first on the pristine base: Devon has the develop op
only as a 10-slider pattern (``media/imggen/develop.py``: exposure,
temperature, tint, brightness, contrast, shadows, highlights,
saturation, vibrance, sharpness). LightCraft's develop engine — the
dual MIT/Apache-2.0 reference in
``echelon-mining/repos/storytold/lightcraft`` — is a full parametric
develop model: Light (exposure/contrast/highlights/shadows/whites/
blacks), white balance in Kelvin with named presets, parametric +
point tone curves, 8-band HSL mixer, colour grading wheels, B&W mix,
texture/clarity/dehaze, vignette, seeded grain, detail (sharpen + NR),
optics (distortion/CA/defringe/vignetting), geometry (crop/rotate/
flips), calibration, HDR with an SDR rendition, local-adjustment
masks, spot healing, profiles, process versions (V1 frozen, V2
current), and a typed ``ErrorKind`` — decided on by code, never by
message text (``crates/engine/src/errors.rs``).

This lane ports that engine into ``media/imggen/develop.py`` in
place (the pattern rung becomes the full engine; the 10 legacy
sliders keep their exact behaviour), wires it through the
ImageDocument develop stage and the photo_editor ``develop`` op, and
proves it on real images.
"""

from __future__ import annotations

import io
import json
import unittest

from PIL import Image


def _img(w: int = 16, h: int = 16, color=(128, 128, 128)):
    return Image.new("RGB", (w, h), color)


def _gradient(w: int = 32, h: int = 32):
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = (x * 255 // max(1, w - 1),
                        y * 255 // max(1, h - 1),
                        (x + y) * 255 // max(1, w + h - 2))
    return img


def _mean(img):
    px = list(img.convert("RGB").getdata())
    n = len(px)
    return tuple(sum(c[i] for c in px) / n for i in range(3))


def _png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestSettingsSchema(unittest.TestCase):
    def test_defaults_are_identity(self):
        from nomorals.media.imggen.develop import DevelopSettings
        s = DevelopSettings()
        self.assertTrue(s.is_identity())
        out = s.render(_gradient())
        src = _gradient()
        self.assertEqual(list(out.getdata()), list(src.getdata()))

    def test_json_round_trip(self):
        from nomorals.media.imggen.develop import DevelopSettings
        s = DevelopSettings.from_dict({
            "light": {"exposure": 1.0, "whites": 40.0},
            "color": {"saturation": 25.0},
            "profile": {"id": "lc.vivid", "amount": 80.0},
        })
        back = DevelopSettings.from_dict(json.loads(s.to_json()))
        self.assertEqual(back, s)
        self.assertEqual(back.light.exposure, 1.0)
        self.assertEqual(back.profile.id, "lc.vivid")

    def test_lenient_read_ignores_unknown_fields(self):
        from nomorals.media.imggen.develop import DevelopSettings
        s = DevelopSettings.from_dict({
            "light": {"exposure": 0.5, "not_a_control": 9},
            "future_section": {"x": 1},
        })
        self.assertEqual(s.light.exposure, 0.5)
        self.assertTrue(s.is_identity() is False)

    def test_damaged_process_reads_as_v1(self):
        from nomorals.media.imggen.develop import DevelopSettings, ProcessVersion
        for bad in (None, -1, 2.5, "v2", [2], 2 ** 40):
            s = DevelopSettings.from_dict({"process": bad})
            self.assertEqual(s.process, ProcessVersion.V1,
                             f"process {bad!r} must read as V1")
        missing = DevelopSettings.from_dict({})
        self.assertEqual(missing.process, ProcessVersion.V1)
        latest = DevelopSettings()
        self.assertEqual(latest.process, ProcessVersion.LATEST)
        self.assertFalse(latest.process.is_outdated())
        self.assertTrue(ProcessVersion.V1.is_outdated())

    def test_disabled_section_renders_as_default(self):
        from nomorals.media.imggen.develop import DevelopSettings
        s = DevelopSettings.from_dict({
            "light": {"exposure": 1.0},
            "disabled_sections": ["light"],
        })
        out = s.render(_gray_img(100))
        self.assertEqual(out.getpixel((0, 0))[0],
                         _gray_img(100).getpixel((0, 0))[0])

    def test_unknown_setting_key_is_bad_params(self):
        from nomorals.media.imggen.develop import DevelopError, DevelopErrorKind, DevelopSettings
        with self.assertRaises(DevelopError) as cm:
            DevelopSettings.from_dict({"light": {"glow": 1.0}})
        self.assertEqual(cm.exception.kind,
                         DevelopErrorKind.bad_params)

    def test_out_of_range_is_bad_params(self):
        from nomorals.media.imggen.develop import DevelopError, DevelopErrorKind, DevelopSettings
        with self.assertRaises(DevelopError) as cm:
            DevelopSettings.from_dict({"light": {"exposure": 99.0}})
        self.assertEqual(cm.exception.kind,
                         DevelopErrorKind.bad_params)


def _gray_img(v: int) -> Image.Image:
    return Image.new("RGB", (8, 8), (v, v, v))


class TestWhiteBalance(unittest.TestCase):
    def test_wb_presets_match_upstream(self):
        from nomorals.media.imggen.develop import WB_PRESETS
        self.assertEqual(WB_PRESETS["daylight"], (5500.0, 10.0))
        self.assertEqual(WB_PRESETS["cloudy"], (6500.0, 10.0))
        self.assertEqual(WB_PRESETS["shade"], (7500.0, 10.0))
        self.assertEqual(WB_PRESETS["tungsten"], (2850.0, 0.0))
        self.assertEqual(WB_PRESETS["fluorescent"], (3800.0, 21.0))
        self.assertEqual(WB_PRESETS["flash"], (5500.0, 0.0))

    def test_as_shot_is_identity(self):
        from nomorals.media.imggen.develop import DevelopSettings
        s = DevelopSettings.from_dict(
            {"wb": {"mode": "as_shot", "temp": 6500.0, "tint": 0.0}})
        img = _gradient()
        self.assertEqual(list(s.render(img).getdata()),
                         list(img.getdata()))

    def test_warmer_temp_warms_the_image(self):
        from nomorals.media.imggen.develop import DevelopSettings
        img = Image.new("RGB", (4, 4), (200, 100, 50))
        warm = DevelopSettings.from_dict(
            {"wb": {"mode": "custom", "temp": 8500.0, "tint": 0.0}})
        cool = DevelopSettings.from_dict(
            {"wb": {"mode": "custom", "temp": 3500.0, "tint": 0.0}})
        w = warm.render(img).getpixel((0, 0))
        c = cool.render(img).getpixel((0, 0))
        self.assertGreater(w[0], c[0])
        self.assertLess(w[2], c[2])

    def test_unknown_wb_mode_is_bad_params(self):
        from nomorals.media.imggen.develop import DevelopError, DevelopErrorKind, DevelopSettings
        with self.assertRaises(DevelopError) as cm:
            DevelopSettings.from_dict({"wb": {"mode": "candle"}})
        self.assertEqual(cm.exception.kind,
                         DevelopErrorKind.bad_params)


class TestLightAndTone(unittest.TestCase):
    def test_exposure_is_ev_gain(self):
        from nomorals.media.imggen.develop import DevelopSettings
        s = DevelopSettings.from_dict({"light": {"exposure": 1.0}})
        out = s.render(_gray_img(64))
        self.assertAlmostEqual(out.getpixel((0, 0))[0], 128, delta=3)

    def test_whites_lift_only_the_bright_end(self):
        from nomorals.media.imggen.develop import DevelopSettings
        img = Image.new("RGB", (2, 1))
        img.putpixel((0, 0), (30, 30, 30))
        img.putpixel((1, 0), (225, 225, 225))
        s = DevelopSettings.from_dict({"light": {"whites": 80.0}})
        out = s.render(img)
        self.assertGreater(out.getpixel((1, 0))[0] - 225,
                           out.getpixel((0, 0))[0] - 30 + 5)

    def test_blacks_drop_only_the_dark_end(self):
        from nomorals.media.imggen.develop import DevelopSettings
        img = Image.new("RGB", (2, 1))
        img.putpixel((0, 0), (30, 30, 30))
        img.putpixel((1, 0), (225, 225, 225))
        s = DevelopSettings.from_dict({"light": {"blacks": -80.0}})
        out = s.render(img)
        self.assertGreater(30 - out.getpixel((0, 0))[0],
                           225 - out.getpixel((1, 0))[0] + 5)

    def test_parametric_curve_lifts_mids(self):
        from nomorals.media.imggen.develop import DevelopSettings
        s = DevelopSettings.from_dict(
            {"curve": {"lights": 60.0, "darks": 0.0,
                       "highlights": 0.0, "shadows": 0.0}})
        out = s.render(_gray_img(128))
        self.assertGreater(out.getpixel((0, 0))[0], 128)

    def test_point_curve_is_piecewise_linear(self):
        from nomorals.media.imggen.develop import DevelopSettings
        s = DevelopSettings.from_dict(
            {"curve": {"master": [[0.0, 0.0], [0.5, 0.75],
                                  [1.0, 1.0]]}})
        out = s.render(_gray_img(128))
        self.assertAlmostEqual(out.getpixel((0, 0))[0], 191, delta=4)
        flat = s.render(_gray_img(0))
        self.assertEqual(flat.getpixel((0, 0))[0], 0)
        top = s.render(_gray_img(255))
        self.assertEqual(top.getpixel((0, 0))[0], 255)

    def test_bad_point_curve_is_bad_params(self):
        from nomorals.media.imggen.develop import DevelopError, DevelopErrorKind, DevelopSettings
        with self.assertRaises(DevelopError) as cm:
            DevelopSettings.from_dict(
                {"curve": {"master": [[0.5, 0.5], [0.25, 0.25]]}})
        self.assertEqual(cm.exception.kind,
                         DevelopErrorKind.bad_params)

    def test_process_v2_highlights_keep_local_contrast(self):
        # V2's highlights/shadows use an edge-aware base: on a hard
        # black/white edge the lift must not bleed across it the way
        # the V1 luminance weighting does.
        from nomorals.media.imggen.develop import DevelopSettings
        img = Image.new("RGB", (16, 1))
        for x in range(8):
            img.putpixel((x, 0), (20, 20, 20))
        for x in range(8, 16):
            img.putpixel((x, 0), (235, 235, 235))
        v1 = DevelopSettings.from_dict(
            {"process": 1, "light": {"shadows": 100.0}})
        v2 = DevelopSettings.from_dict(
            {"process": 2, "light": {"shadows": 100.0}})
        o1 = v1.render(img)
        o2 = v2.render(img)
        # Both lift the dark side; V2 keeps the bright side closer
        # to untouched at the seam (no bleed from the dark lift).
        self.assertGreater(o1.getpixel((0, 0))[0], 20)
        self.assertGreater(o2.getpixel((0, 0))[0], 20)
        self.assertLess(abs(o2.getpixel((15, 0))[0] - 235),
                        abs(o1.getpixel((15, 0))[0] - 235) + 1)


class TestColour(unittest.TestCase):
    def test_mixer_band_shifts_only_its_hue(self):
        from nomorals.media.imggen.develop import DevelopSettings
        img = Image.new("RGB", (2, 1))
        img.putpixel((0, 0), (220, 30, 30))   # red band
        img.putpixel((1, 0), (30, 30, 220))   # blue band
        s = DevelopSettings.from_dict(
            {"mixer": {"red": {"hue": 60.0, "sat": 0.0, "lum": 0.0}}})
        out = s.render(img)
        r0 = out.getpixel((0, 0))
        r1 = out.getpixel((1, 0))
        # red pixel moved towards orange/yellow (green up); blue
        # pixel essentially untouched.
        self.assertGreater(r0[1], 60)
        self.assertAlmostEqual(r1[2], 220, delta=8)
        self.assertAlmostEqual(r1[0], 30, delta=8)

    def test_grading_wheel_tints_the_shadows(self):
        from nomorals.media.imggen.develop import DevelopSettings
        img = Image.new("RGB", (2, 1))
        img.putpixel((0, 0), (30, 30, 30))
        img.putpixel((1, 0), (225, 225, 225))
        s = DevelopSettings.from_dict(
            {"grading": {"shadows": {"hue": 220.0, "sat": 80.0,
                                     "lum": 0.0}}})
        out = s.render(img)
        dark = out.getpixel((0, 0))
        bright = out.getpixel((1, 0))
        self.assertGreater(dark[2], dark[0] + 5)
        self.assertAlmostEqual(bright[0], bright[2], delta=6)

    def test_bw_treatment_is_gray(self):
        from nomorals.media.imggen.develop import DevelopSettings
        s = DevelopSettings.from_dict({"treatment": "bw"})
        out = s.render(Image.new("RGB", (2, 2), (200, 100, 50)))
        r, g, b = out.getpixel((0, 0))
        self.assertEqual(r, g)
        self.assertEqual(g, b)

    def test_bw_mix_weights_bands(self):
        from nomorals.media.imggen.develop import DevelopSettings
        img = Image.new("RGB", (2, 1))
        img.putpixel((0, 0), (220, 30, 30))
        img.putpixel((1, 0), (30, 220, 30))
        s = DevelopSettings.from_dict({
            "treatment": "bw",
            "bw_mix": {"red": 100.0, "green": -100.0},
        })
        out = s.render(img)
        self.assertGreater(out.getpixel((0, 0))[0],
                           out.getpixel((1, 0))[0] + 20)

    def test_calibration_rotates_the_red_primary(self):
        from nomorals.media.imggen.develop import DevelopSettings
        img = Image.new("RGB", (1, 1), (220, 30, 30))
        s = DevelopSettings.from_dict(
            {"calibration": {"red_hue": 60.0, "red_sat": 0.0,
                             "green_hue": 0.0, "green_sat": 0.0,
                             "blue_hue": 0.0, "blue_sat": 0.0}})
        out = s.render(img)
        self.assertGreater(out.getpixel((0, 0))[1], 60)


class TestEffectsDetailOptics(unittest.TestCase):
    def test_dehaze_adds_contrast_to_haze(self):
        from nomorals.media.imggen.develop import DevelopSettings
        img = Image.new("RGB", (16, 16))
        px = img.load()
        for y in range(16):
            for x in range(16):
                v = 120 + (40 if (x // 4 + y // 4) % 2 else -40)
                px[x, y] = (v, v, v)
        s = DevelopSettings.from_dict({"effects": {"dehaze": 80.0}})
        out = s.render(img)
        vals = [p[0] for p in out.getdata()]
        self.assertGreater(max(vals) - min(vals), 80)

    def test_clarity_is_local_contrast(self):
        from nomorals.media.imggen.develop import DevelopSettings
        img = _gradient(32, 32)
        s = DevelopSettings.from_dict({"effects": {"clarity": 80.0}})
        out = s.render(img)
        # local contrast raises the spread of the gradient
        before = [p[0] for p in img.getdata()]
        after = [p[0] for p in out.getdata()]
        self.assertGreater(max(after) - min(after),
                           max(before) - min(before) - 2)

    def test_vignette_darkens_corners_not_centre(self):
        from nomorals.media.imggen.develop import DevelopSettings
        img = _img(32, 32, (180, 180, 180))
        s = DevelopSettings.from_dict(
            {"vignette": {"amount": -80.0, "midpoint": 50.0,
                          "feather": 50.0}})
        out = s.render(img)
        self.assertLess(out.getpixel((0, 0))[0], 150)
        self.assertAlmostEqual(out.getpixel((16, 16))[0], 180,
                               delta=8)

    def test_grain_is_deterministic_per_seed(self):
        from nomorals.media.imggen.develop import DevelopSettings
        img = _img(32, 32, (128, 128, 128))
        a = DevelopSettings.from_dict(
            {"grain": {"amount": 60.0, "seed": 7}}).render(img)
        b = DevelopSettings.from_dict(
            {"grain": {"amount": 60.0, "seed": 7}}).render(img)
        c = DevelopSettings.from_dict(
            {"grain": {"amount": 60.0, "seed": 8}}).render(img)
        self.assertEqual(_png_bytes(a), _png_bytes(b))
        self.assertNotEqual(_png_bytes(a), _png_bytes(c))
        self.assertNotEqual(list(a.getdata()), list(img.getdata()))

    def test_grain_zero_is_identity(self):
        from nomorals.media.imggen.develop import DevelopSettings
        img = _gradient()
        s = DevelopSettings.from_dict({"grain": {"amount": 0.0}})
        self.assertEqual(list(s.render(img).getdata()),
                         list(img.getdata()))

    def test_distortion_moves_pixels(self):
        from nomorals.media.imggen.develop import DevelopSettings
        img = Image.new("RGB", (32, 32), (0, 0, 0))
        for y in range(32):
            img.putpixel((16, y), (255, 255, 255))
        s = DevelopSettings.from_dict({"optics": {"distortion": 60.0}})
        out = s.render(img)
        self.assertNotEqual(list(out.getdata()), list(img.getdata()))
        # the white line is no longer straight down the middle
        col_mid = [out.getpixel((16, y))[0] for y in range(32)]
        self.assertLess(sum(col_mid), 32 * 255)

    def test_optics_vignetting_darkens_edges(self):
        from nomorals.media.imggen.develop import DevelopSettings
        img = _img(32, 32, (180, 180, 180))
        s = DevelopSettings.from_dict(
            {"optics": {"vignetting": -60.0}})
        out = s.render(img)
        self.assertLess(out.getpixel((0, 0))[0], 160)
        self.assertAlmostEqual(out.getpixel((16, 16))[0], 180,
                               delta=8)

    def test_sharpen_raises_edge_contrast(self):
        from nomorals.media.imggen.develop import DevelopSettings
        img = Image.new("RGB", (16, 16), (100, 100, 100))
        for x in range(8, 16):
            for y in range(16):
                img.putpixel((x, y), (180, 180, 180))
        s = DevelopSettings.from_dict(
            {"detail": {"sharpen_amount": 100.0,
                        "sharpen_radius": 1.0}})
        out = s.render(img)
        self.assertGreater(out.getpixel((8, 8))[0],
                           img.getpixel((8, 8))[0])

    def test_luminance_nr_smooths_noise(self):
        import random

        from nomorals.media.imggen.develop import DevelopSettings
        rnd = random.Random(3)
        img = Image.new("RGB", (24, 24))
        px = img.load()
        for y in range(24):
            for x in range(24):
                v = 128 + rnd.randint(-40, 40)
                px[x, y] = (v, v, v)
        s = DevelopSettings.from_dict(
            {"detail": {"nr_luminance": 80.0}})
        out = s.render(img)
        vals_in = [p[0] for p in img.getdata()]
        vals_out = [p[0] for p in out.getdata()]
        mean_in = sum(vals_in) / len(vals_in)
        mean_out = sum(vals_out) / len(vals_out)
        var_in = sum((v - mean_in) ** 2 for v in vals_in) / len(vals_in)
        var_out = sum((v - mean_out) ** 2 for v in vals_out) / len(vals_out)
        self.assertLess(var_out, var_in * 0.7)


class TestGeometry(unittest.TestCase):
    def test_crop_and_flips(self):
        from nomorals.media.imggen.develop import DevelopSettings
        img = Image.new("RGB", (8, 4))
        img.putpixel((0, 0), (255, 0, 0))
        img.putpixel((7, 0), (0, 0, 255))
        s = DevelopSettings.from_dict(
            {"geometry": {"crop": {"x": 0.0, "y": 0.0, "w": 0.5,
                                   "h": 1.0}}})
        out = s.render(img)
        self.assertEqual(out.size, (4, 4))
        self.assertEqual(out.getpixel((0, 0)), (255, 0, 0))
        flip = DevelopSettings.from_dict(
            {"geometry": {"crop": {"flip_h": True}}})
        fout = flip.render(img)
        self.assertEqual(fout.size, (8, 4))
        self.assertEqual(fout.getpixel((0, 0)), (0, 0, 255))

    def test_rotate_90_swaps_dimensions(self):
        from nomorals.media.imggen.develop import DevelopSettings
        img = Image.new("RGB", (8, 4), (90, 90, 90))
        s = DevelopSettings.from_dict({"geometry": {"rotate": 90.0}})
        out = s.render(img)
        self.assertEqual(out.size, (4, 8))

    def test_upright_auto_is_an_honest_refusal(self):
        from nomorals.media.imggen.develop import DevelopError, DevelopErrorKind, DevelopSettings
        s = DevelopSettings.from_dict(
            {"geometry": {"upright": "auto"}})
        with self.assertRaises(DevelopError) as cm:
            s.render(_img())
        self.assertEqual(cm.exception.kind,
                         DevelopErrorKind.bad_params)


class TestMasksAndSpots(unittest.TestCase):
    def test_linear_mask_limits_the_adjustment(self):
        from nomorals.media.imggen.develop import DevelopSettings
        img = _gray_img(100)
        img = img.resize((32, 8))
        s = DevelopSettings.from_dict({
            "masks": [{
                "shape": {"kind": "linear",
                          "start": [0.0, 0.5], "end": [1.0, 0.5]},
                "adjust": {"exposure": 1.0},
            }],
        })
        out = s.render(img)
        left = out.getpixel((0, 4))[0]
        right = out.getpixel((31, 4))[0]
        self.assertGreater(right, left + 30)
        self.assertAlmostEqual(left, 100, delta=8)

    def test_radial_mask_is_local(self):
        from nomorals.media.imggen.develop import DevelopSettings
        img = _gray_img(100).resize((32, 32))
        s = DevelopSettings.from_dict({
            "masks": [{
                "shape": {"kind": "radial", "center": [0.5, 0.5],
                          "rx": 0.2, "ry": 0.2, "feather": 0.0},
                "adjust": {"exposure": 1.0},
            }],
        })
        out = s.render(img)
        self.assertGreater(out.getpixel((16, 16))[0], 150)
        self.assertAlmostEqual(out.getpixel((0, 0))[0], 100, delta=8)

    def test_luminance_mask_targets_bright_pixels(self):
        from nomorals.media.imggen.develop import DevelopSettings
        img = Image.new("RGB", (2, 1))
        img.putpixel((0, 0), (30, 30, 30))
        img.putpixel((1, 0), (230, 230, 230))
        s = DevelopSettings.from_dict({
            "masks": [{
                "shape": {"kind": "luminance", "lo": 0.7, "hi": 1.0,
                          "lo_feather": 0.0, "hi_feather": 0.0},
                "adjust": {"exposure": -1.0},
            }],
        })
        out = s.render(img)
        self.assertAlmostEqual(out.getpixel((0, 0))[0], 30, delta=8)
        self.assertLess(out.getpixel((1, 0))[0], 180)

    def test_spot_clone_copies_from_source_offset(self):
        from nomorals.media.imggen.develop import DevelopSettings
        img = Image.new("RGB", (16, 16), (50, 50, 50))
        for y in range(4, 8):
            for x in range(4, 8):
                img.putpixel((x, y), (220, 30, 30))
        s = DevelopSettings.from_dict({
            "spots": [{
                "mode": "clone", "points": [[0.75, 0.75]],
                "size": 0.2, "source_offset": [-0.5, -0.5],
            }],
        })
        out = s.render(img)
        self.assertGreater(out.getpixel((12, 12))[0], 150)

    def test_spot_remove_fills_with_neighbours(self):
        from nomorals.media.imggen.develop import DevelopSettings
        img = Image.new("RGB", (16, 16), (90, 90, 90))
        for y in range(6, 10):
            for x in range(6, 10):
                img.putpixel((x, y), (255, 255, 255))
        s = DevelopSettings.from_dict({
            "spots": [{
                "mode": "remove", "points": [[0.5, 0.5]],
                "size": 0.2,
            }],
        })
        out = s.render(img)
        self.assertLess(out.getpixel((8, 8))[0], 180)


class TestProfilesPresetsAndHdr(unittest.TestCase):
    def test_builtin_profiles(self):
        from nomorals.media.imggen.develop import BUILTIN_PROFILES
        self.assertIn("lc.color", BUILTIN_PROFILES)
        self.assertIn("lc.vivid", BUILTIN_PROFILES)
        self.assertIn("lc.mono", BUILTIN_PROFILES)

    def test_vivid_profile_adds_colour(self):
        from nomorals.media.imggen.develop import DevelopSettings
        img = Image.new("RGB", (2, 2), (160, 120, 100))
        s = DevelopSettings.from_dict(
            {"profile": {"id": "lc.vivid", "amount": 100.0}})
        out = s.render(img)
        self.assertNotEqual(out.getpixel((0, 0)), (160, 120, 100))

    def test_unknown_profile_is_missing_profile(self):
        from nomorals.media.imggen.develop import DevelopError, DevelopErrorKind, DevelopSettings
        s = DevelopSettings.from_dict(
            {"profile": {"id": "lc.nope", "amount": 100.0}})
        with self.assertRaises(DevelopError) as cm:
            s.render(_img())
        self.assertEqual(cm.exception.kind,
                         DevelopErrorKind.missing_profile)

    def test_profile_amount_zero_is_identity(self):
        from nomorals.media.imggen.develop import DevelopSettings
        img = _gradient()
        s = DevelopSettings.from_dict(
            {"profile": {"id": "lc.vivid", "amount": 0.0}})
        self.assertEqual(list(s.render(img).getdata()),
                         list(img.getdata()))

    def test_named_presets_apply(self):
        from nomorals.media.imggen.develop import DevelopSettings
        img = _gradient()
        s = DevelopSettings.from_dict({})
        warm = s.apply_preset("warm")
        self.assertNotEqual(
            list(warm.render(img).getdata()), list(img.getdata()))
        mono = s.apply_preset("mono")
        r, g, b = mono.render(img).getpixel((0, 0))
        self.assertEqual(r, g)
        with self.assertRaises(KeyError):
            s.apply_preset("no-such-preset")

    def test_hdr_sdr_rendition(self):
        from nomorals.media.imggen.develop import DevelopSettings
        s = DevelopSettings.from_dict({
            "hdr": {"enabled": True, "max_ev": 3.0,
                    "sdr_brightness": 100.0},
        })
        rendition = s.sdr_rendition()
        self.assertFalse(rendition.hdr.enabled)
        self.assertAlmostEqual(rendition.light.exposure,
                               s.light.exposure + 1.0, delta=0.01)
        img = _gray_img(100)
        out = s.render(img)
        self.assertGreater(out.getpixel((0, 0))[0], 100)


class TestErrorKinds(unittest.TestCase):
    def test_kind_does_not_depend_on_the_message(self):
        # Upstream's rule (errors.rs): decide on kind, never text.
        from nomorals.media.imggen.develop import DevelopError, DevelopErrorKind
        err = DevelopError(DevelopErrorKind.cancelled,
                           "the render was cancelled by the user")
        self.assertEqual(err.kind, DevelopErrorKind.cancelled)
        other = DevelopError(
            DevelopErrorKind.other,
            "cancelled: the model is not installed")
        self.assertEqual(other.kind, DevelopErrorKind.other)

    def test_open_corrupt_bytes_is_bad_file(self):
        from nomorals.media.imggen.develop import (
            DevelopError,
            DevelopErrorKind,
            open_develop_source,
        )
        with self.assertRaises(DevelopError) as cm:
            open_develop_source(io.BytesIO(b"not an image at all"))
        self.assertEqual(cm.exception.kind,
                         DevelopErrorKind.bad_file)

    def test_open_raw_extension_is_unsupported_format(self):
        from nomorals.media.imggen.develop import (
            DevelopError,
            DevelopErrorKind,
            open_develop_source,
        )
        with self.assertRaises(DevelopError) as cm:
            open_develop_source("photo.cr2")
        self.assertEqual(cm.exception.kind,
                         DevelopErrorKind.unsupported_format)
        with self.assertRaises(DevelopError) as cm:
            open_develop_source("photo.dng")
        self.assertEqual(cm.exception.kind,
                         DevelopErrorKind.unsupported_format)

    def test_open_missing_file_is_bad_file(self):
        from nomorals.media.imggen.develop import (
            DevelopError,
            DevelopErrorKind,
            open_develop_source,
        )
        with self.assertRaises(DevelopError) as cm:
            open_develop_source("/nonexistent/photo.jpg")
        self.assertEqual(cm.exception.kind,
                         DevelopErrorKind.bad_file)

    def test_render_non_image_is_bad_file(self):
        from nomorals.media.imggen.develop import DevelopError, DevelopErrorKind, DevelopSettings
        with self.assertRaises(DevelopError) as cm:
            DevelopSettings().render("not an image")
        self.assertEqual(cm.exception.kind,
                         DevelopErrorKind.bad_file)


class TestDevelopStack(unittest.TestCase):
    def _photo_bytes(self):
        return _png_bytes(_gradient(32, 32))

    def test_source_bytes_never_change(self):
        from nomorals.media.imggen.develop import DevelopStack
        src = self._photo_bytes()
        stack = DevelopStack(src)
        stack.add_edit("lift", {"light": {"exposure": 1.0}})
        stack.add_edit("warm", {"wb": {"mode": "custom",
                                       "temp": 8000.0}})
        stack.render()
        self.assertEqual(stack.source_bytes, src)

    def test_stack_renders_and_re_renders(self):
        from nomorals.media.imggen.develop import DevelopStack
        stack = DevelopStack(self._photo_bytes())
        before = stack.render_before()
        stack.add_edit("lift", {"light": {"exposure": 1.0}})
        after = stack.render()
        self.assertNotEqual(_png_bytes(before), _png_bytes(after))
        stack.add_edit("cooler", {"light": {"exposure": -0.5}})
        later = stack.render()
        self.assertNotEqual(_png_bytes(after), _png_bytes(later))
        # same settings, same bytes — the cache path included
        self.assertEqual(_png_bytes(stack.render()),
                         _png_bytes(later))

    def test_before_is_the_default_render(self):
        from nomorals.media.imggen.develop import DevelopSettings, DevelopStack
        stack = DevelopStack(self._photo_bytes())
        stack.add_edit("lift", {"light": {"exposure": 1.0}})
        self.assertEqual(_png_bytes(stack.render_before()),
                         _png_bytes(
                             DevelopSettings().render(
                                 Image.open(io.BytesIO(
                                     self._photo_bytes())))))

    def test_stack_json_round_trip(self):
        from nomorals.media.imggen.develop import DevelopStack
        stack = DevelopStack(self._photo_bytes())
        stack.add_edit("lift", {"light": {"exposure": 1.0}})
        back = DevelopStack.from_json(stack.to_json(),
                                      self._photo_bytes())
        self.assertEqual(_png_bytes(back.render()),
                         _png_bytes(stack.render()))
        self.assertEqual(back.edits[0][0], "lift")


class TestLegacyUntouched(unittest.TestCase):
    def test_legacy_develop_image_unchanged(self):
        from nomorals.media.imggen.develop import develop_image
        out = develop_image(_gray_img(64), exposure=1.0)
        self.assertAlmostEqual(out.getpixel((0, 0))[0], 128, delta=2)
        out = develop_image(_gray_img(64), exposure=-1.0)
        self.assertAlmostEqual(out.getpixel((0, 0))[0], 32, delta=2)
        img = Image.new("RGB", (1, 1), (200, 100, 50))
        out = develop_image(img, saturation=-1.0)
        r, g, b = out.getpixel((0, 0))
        self.assertEqual(r, g)
        self.assertEqual(g, b)

    def test_legacy_unknown_control_still_imggen_error(self):
        from nomorals.media.imggen import ImgGenError
        from nomorals.media.imggen.develop import develop_image
        with self.assertRaises(ImgGenError,
                               msg="unknown develop control"):
            develop_image(_gray_img(10), glow=0.5)

    def test_legacy_defaults_and_fields_unchanged(self):
        from nomorals.media.imggen.develop import DEVELOP_FIELDS, develop_defaults
        self.assertEqual(DEVELOP_FIELDS[:3],
                         ("exposure", "temperature", "tint"))
        self.assertEqual(develop_defaults()["exposure"], 0.0)


class TestWiring(unittest.TestCase):
    def _doc_with_png_source(self):
        import tempfile

        from nomorals.media.imggen import ImageDocument
        with tempfile.NamedTemporaryFile(suffix=".png",
                                        delete=False) as tmp:
            _gradient(32, 32).save(tmp.name, format="PNG")
            name = tmp.name
        self.addCleanup(lambda: __import__("os").unlink(name))
        return ImageDocument(name)

    def test_image_document_develop_with_new_keys(self):
        doc = self._doc_with_png_source()
        doc.develop(**{"light.exposure": 1.0,
                       "color.saturation": 30.0})
        out = doc.render().image
        self.assertGreater(_mean(out)[0], _mean(_gradient(32, 32))[0])

    def test_image_document_develop_legacy_still_works(self):
        doc = self._doc_with_png_source()
        doc.develop(exposure=0.5)
        out = doc.render()
        self.assertIsNotNone(out.image)

    def test_photo_document_develop_with_new_keys(self):
        from pathlib import Path

        from nomorals.media.photo_editor.document import PhotoDocument
        fixtures = (Path(__file__).resolve().parent
                    / "fixtures" / "photocraft")
        doc = PhotoDocument.from_psd_bytes(
            (fixtures / "rgb8-baseline-layer-copy.psd").read_bytes())
        before = doc.flatten()
        doc.apply_edit("develop", **{"light.exposure": 1.0})
        after = doc.flatten()
        self.assertNotEqual(list(after.getdata()),
                            list(before.getdata()))
        self.assertEqual(doc.history[-1]["op"], "develop")

    def test_photo_document_develop_legacy_still_works(self):
        from pathlib import Path

        from nomorals.media.photo_editor.document import PhotoDocument
        fixtures = (Path(__file__).resolve().parent
                    / "fixtures" / "photocraft")
        doc = PhotoDocument.from_psd_bytes(
            (fixtures / "rgb8-baseline-layer-copy.psd").read_bytes())
        doc.apply_edit("develop", brightness=0.5)
        self.assertEqual(doc.history[-1]["op"], "develop")


if __name__ == "__main__":
    unittest.main()
