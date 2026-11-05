"""Dedicated suite for the rebuilt color engine (§1.1c).

Pins, with independently computed expectations:
- .cube parse/generate/apply round-trips (lattice order verified
  against ffmpeg's lut3d: red varies fastest);
- ASC-CDL math against hand-computed values from the standard
  formula (SOP + Rec.709-luma saturation);
- scopes against synthetic frames with known content — a scope that
  cannot tell a red frame from a green one fails here;
- identity grades are bit-identical;
- missing rungs / bad inputs return named reasons, never fakes.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

from nomorals.media.studio_color import (
    CDL, CubeLUT, StudioColorError, apply_cdl, apply_lut, bake_cdl_cube_file,
    bake_grade_lut, cdf_match_tables, channel_tables, color_wheels,
    cdl_from_wheels, evaluate_curve, generate_cube, histogram_data,
    load_cube, match_shot, measure_frame, parse_cube, parse_curve_points,
    rgb_parade_data, scopes, scopes_for_media, vectorscope_centroid,
    vectorscope_data, waveform_data, _ffmpeg, _grab_frame,
)

FFMPEG = _ffmpeg()
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not installed")


def _solid(h: int, w: int, rgb: tuple[float, float, float]) -> np.ndarray:
    a = np.zeros((h, w, 3))
    a[..., 0], a[..., 1], a[..., 2] = rgb
    return a


# ── .cube parsing ──────────────────────────────────────────────────

class TestCubeParse:
    def test_parse_minimal_and_sample_lattice(self):
        # size-2 identity written by hand in .cube row order
        # (red fastest): index = r + 2g + 4b.
        rows = []
        for b in (0.0, 1.0):
            for g in (0.0, 1.0):
                for r in (0.0, 1.0):
                    rows.append(f"{r} {g} {b}")
        lut = parse_cube('TITLE "hand"\nLUT_3D_SIZE 2\n' + "\n".join(rows))
        assert lut.size == 2 and lut.title == "hand"
        out = lut.sample(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
        assert np.allclose(out, [[1, 0, 0], [0, 0, 1]], atol=1e-6)

    def test_row_order_red_fastest(self):
        # Only row index 1 is white. ffmpeg lut3d (verified against
        # the real binary) reads that as lattice (r=1, g=0, b=0).
        rows = ["0 0 0", "1 1 1"] + ["0 0 0"] * 6
        lut = parse_cube("LUT_3D_SIZE 2\n" + "\n".join(rows))
        assert np.allclose(lut.sample(np.array([[1.0, 0.0, 0.0]])),
                           [[1, 1, 1]], atol=1e-6)
        assert np.allclose(lut.sample(np.array([[0.0, 0.0, 1.0]])),
                           [[0, 0, 0]], atol=1e-6)

    def test_comments_and_domain(self):
        text = ("# a comment\nLUT_3D_SIZE 2\nDOMAIN_MIN 0 0 0\n"
                "DOMAIN_MAX 0.5 0.5 0.5\n"
                + "\n".join(
                    f"{r} {g} {b}"
                    for b in (0.0, 1.0) for g in (0.0, 1.0)
                    for r in (0.0, 1.0)))
        lut = parse_cube(text)
        assert lut.domain_max == (0.5, 0.5, 0.5)
        # input 0.25 is the middle of the declared domain → the
        # identity lattice returns the normalized coordinate 0.5.
        out = lut.sample(np.array([[0.25, 0.25, 0.25]]))
        assert np.allclose(out, [[0.5, 0.5, 0.5]], atol=1e-6)

    def test_1d_plus_3d(self):
        # 1D stage inverts (2 entries: 0→1, 1→0), 3D stage identity.
        text = ("LUT_1D_SIZE 2\nLUT_3D_SIZE 2\n"
                "1 1 1\n0 0 0\n"
                + "\n".join(
                    f"{r} {g} {b}"
                    for b in (0.0, 1.0) for g in (0.0, 1.0)
                    for r in (0.0, 1.0)))
        lut = parse_cube(text)
        out = lut.sample(np.array([[0.25, 0.5, 0.75]]))
        assert np.allclose(out, [[0.75, 0.5, 0.25]], atol=1e-6)

    def test_row_count_mismatch_is_named(self):
        with pytest.raises(StudioColorError, match="row count mismatch"):
            parse_cube("LUT_3D_SIZE 2\n0 0 0\n1 1 1\n")

    def test_unknown_keyword_is_named(self):
        with pytest.raises(StudioColorError, match="unknown .cube keyword"):
            parse_cube("LUT_3D_SIZE 2\nWIBBLE 3\n" + "0 0 0\n" * 8)

    def test_no_size_is_named(self):
        with pytest.raises(StudioColorError, match="no LUT_3D_SIZE"):
            parse_cube("0 0 0\n1 1 1\n")

    def test_bad_size_is_named(self):
        with pytest.raises(StudioColorError, match="out of range"):
            parse_cube("LUT_3D_SIZE 1\n0 0 0\n")

    def test_bad_row_is_named(self):
        with pytest.raises(StudioColorError, match="exactly 3 values"):
            parse_cube("LUT_3D_SIZE 2\n" + "0 0\n" + "0 0 0\n" * 7)


class TestCubeRoundTrip:
    def test_generate_serialize_parse_apply(self):
        def warm(r, g, b):
            return (np.clip(r * 1.1, 0, 1), g ** 0.9, np.clip(b * 0.9, 0, 1))
        gen = generate_cube(17, warm, title="warmth")
        back = parse_cube(gen.to_cube_text())
        assert back.size == 17 and back.title == "warmth"
        # known pixel deltas at lattice points (exact by construction)
        px = np.array([[0.5, 0.5, 0.5], [0.25, 0.75, 0.125]])
        expect = np.stack([np.clip(px[:, 0] * 1.1, 0, 1), px[:, 1] ** 0.9,
                           np.clip(px[:, 2] * 0.9, 0, 1)], axis=-1)
        assert np.allclose(back.apply(px), expect, atol=2e-3)
        # and mid-lattice points stay close (trilinear of a smooth fn)
        mid = np.array([[0.53, 0.31, 0.77]])
        expect_mid = np.stack([np.clip(mid[:, 0] * 1.1, 0, 1),
                               mid[:, 1] ** 0.9,
                               np.clip(mid[:, 2] * 0.9, 0, 1)], axis=-1)
        assert np.allclose(back.apply(mid), expect_mid, atol=0.02)

    def test_load_write_file(self, tmp_path):
        lut = CubeLUT.identity(9)
        p = lut.write(tmp_path / "id.cube")
        back = load_cube(p)
        assert back.size == 9
        rng = np.random.default_rng(7)
        a = rng.random((4, 4, 3))
        assert np.allclose(back.apply(a), a, atol=1e-6)

    def test_identity_uint8_bit_identical(self):
        lut = CubeLUT.identity(17)
        ramp = np.arange(256, dtype=np.uint8)
        img = np.stack([ramp, ramp[::-1], ramp], axis=-1).reshape(1, 256, 3)
        assert np.array_equal(lut.apply_uint8(img), img)
        rng = np.random.default_rng(11)
        noisy = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        assert np.array_equal(lut.apply_uint8(noisy), noisy)

    def test_apply_to_image(self):
        from PIL import Image
        img = Image.new("RGB", (4, 4), (200, 100, 50))
        swap = generate_cube(9, lambda r, g, b: (b, g, r))
        out = swap.apply_to_image(img)
        assert out.getpixel((0, 0)) == (50, 100, 200)

    def test_from_tables(self):
        tables = [[(255 - i) / 255.0 for i in range(256)]] * 3
        lut = CubeLUT.from_tables(tables, size=17)
        out = lut.apply(np.array([[0.25, 0.5, 1.0]]))
        assert np.allclose(out, [[0.75, 0.5, 0.0]], atol=2e-3)


# ── ASC-CDL ────────────────────────────────────────────────────────

class TestCDLMath:
    def test_sop_hand_computed(self):
        # r: 0.25*2 + 0.1 = 0.6 · g: 0.4*1 + 0 = 0.4 · b: 0.8*0.5 = 0.4
        cdl = CDL(slope=(2, 1, 0.5), offset=(0.1, 0, 0), power=(1, 1, 1))
        out = cdl.apply_array(np.array([[[0.25, 0.4, 0.8]]]))[0, 0]
        assert np.allclose(out, [0.6, 0.4, 0.4], atol=1e-12)

    def test_power_hand_computed(self):
        cdl = CDL(power=(2, 0.5, 1))
        out = cdl.apply_array(np.array([[[0.5, 0.25, 0.3]]]))[0, 0]
        assert np.allclose(out, [0.25, 0.5, 0.3], atol=1e-12)

    def test_combined_against_scalar_math(self):
        # (0.5*1.2 + 0.05) ** 1.1, computed independently via math.pow
        cdl = CDL(slope=(1.2, 1.2, 1.2), offset=(0.05, 0.05, 0.05),
                  power=(1.1, 1.1, 1.1))
        out = cdl.apply_array(np.array([[[0.5, 0.5, 0.5]]]))[0, 0]
        assert np.allclose(out, [math.pow(0.65, 1.1)] * 3, atol=1e-12)

    def test_negative_base_clamps_to_zero(self):
        cdl = CDL(offset=(-0.3, -0.3, -0.3), power=(2, 2, 2))
        out = cdl.apply_array(np.array([[[0.1, 0.1, 0.1]]]))[0, 0]
        assert np.allclose(out, [0, 0, 0], atol=1e-12)

    def test_output_clamps_to_one(self):
        cdl = CDL(slope=(4, 4, 4))
        out = cdl.apply_array(np.array([[[0.9, 0.9, 0.9]]]))[0, 0]
        assert np.allclose(out, [1, 1, 1], atol=1e-12)

    def test_saturation_zero_is_rec709_luma(self):
        # pure red SOP (1,0,0) → luma 0.2126 in every channel
        out = CDL(saturation=0.0).apply_array(
            np.array([[[1.0, 0.0, 0.0]]]))[0, 0]
        assert np.allclose(out, [0.2126, 0.2126, 0.2126], atol=1e-12)

    def test_saturation_half_hand_computed(self):
        # luma = 0.2126; out = luma + 0.5*(sop - luma)
        out = CDL(saturation=0.5).apply_array(
            np.array([[[1.0, 0.0, 0.0]]]))[0, 0]
        assert np.allclose(out, [0.6063, 0.1063, 0.1063], atol=1e-9)

    def test_identity_is_bit_identical(self):
        rng = np.random.default_rng(5)
        a = rng.random((16, 16, 3))
        assert np.array_equal(CDL().apply_array(a), a)
        assert CDL().is_identity()

    def test_power_must_be_positive(self):
        with pytest.raises(StudioColorError, match="power must be > 0"):
            CDL(power=(1, 0, 1))

    def test_wheels_mapping(self):
        cdl = cdl_from_wheels((0.1, 0, 0), (2, 1, 1), (1.5, 1, 1))
        assert cdl.offset == (0.1, 0.0, 0.0)
        assert cdl.power == (2.0, 1.0, 1.0)
        assert cdl.slope == (1.5, 1.0, 1.0)

    def test_dict_round_trip(self):
        cdl = CDL(slope=(1.1, 1, 0.9), offset=(0.01, 0, -0.02),
                  power=(1.2, 1, 1), saturation=0.8)
        assert CDL.from_dict(cdl.to_dict()) == cdl


# ── curves + baking ────────────────────────────────────────────────

class TestCurvesAndBake:
    def test_parse_points(self):
        assert parse_curve_points("0/0 0.5/0.6 1/1") == [
            (0.0, 0.0), (0.5, 0.6), (1.0, 1.0)]
        with pytest.raises(StudioColorError, match="bad curve point"):
            parse_curve_points("0-0 1-1")

    def test_identity_curve(self):
        xs = np.linspace(0, 1, 9)
        assert np.allclose(evaluate_curve([(0, 0), (1, 1)], xs), xs,
                           atol=1e-9)

    def test_scurve_fixed_points(self):
        pts = [(0, 0), (0.25, 0.1), (0.5, 0.5), (0.75, 0.9), (1, 1)]
        out = evaluate_curve(pts, np.array([0.0, 0.5, 1.0]))
        assert np.allclose(out, [0.0, 0.5, 1.0], atol=1e-9)
        # S-shape: below the line before midpoint, above after
        q = evaluate_curve(pts, np.array([0.25, 0.75]))
        assert q[0] < 0.25 and q[1] > 0.75

    def test_bake_cdl_lut_matches_cdl(self):
        cdl = CDL(slope=(1.2, 1.0, 0.8), offset=(0.02, 0.0, 0.01),
                  power=(1.1, 1.0, 0.9), saturation=0.9)
        lut = bake_grade_lut(17, cdl=cdl)
        # lattice points (multiples of 1/16) reproduce the CDL tightly
        pts = np.array([[0.5, 0.25, 0.75], [0.125, 0.625, 0.375]])
        assert np.allclose(lut.apply(pts), cdl.apply_array(pts), atol=2e-3)

    def test_bake_curves(self):
        lut = bake_grade_lut(17, curves={"master": "0/0 0.5/0.6 1/1"})
        out = lut.apply(np.array([[0.5, 0.5, 0.5]]))
        assert np.allclose(out, [[0.6, 0.6, 0.6]], atol=2e-3)

    def test_bake_identity_bit_identical(self):
        lut = bake_grade_lut(17, cdl=CDL())
        rng = np.random.default_rng(9)
        img = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
        assert np.array_equal(lut.apply_uint8(img), img)

    def test_bake_cdl_cube_file_content_addressed(self, tmp_path):
        cdl = CDL(slope=(1.3, 1.0, 1.0))
        p1 = bake_cdl_cube_file(cdl, cache_dir=tmp_path)
        p2 = bake_cdl_cube_file(cdl, cache_dir=tmp_path)
        assert p1 == p2 and p1.exists()
        back = load_cube(p1)
        assert np.allclose(back.apply(np.array([[0.5, 0.5, 0.5]])),
                           cdl.apply_array(np.array([[0.5, 0.5, 0.5]])),
                           atol=2e-3)
        other = bake_cdl_cube_file(CDL(slope=(1.4, 1.0, 1.0)),
                                   cache_dir=tmp_path)
        assert other != p1

    def test_channel_tables(self):
        tabs = channel_tables(lambda c, x: x * 0.5 if c == 0 else x)
        assert len(tabs) == 768
        assert tabs[0] == 0 and tabs[255] == 128  # 1.0*0.5 → 127.5 → 128
        assert tabs[256] == 0 and tabs[256 + 255] == 255
        assert tabs[512 + 128] == 128


# ── scopes on synthetic frames ─────────────────────────────────────

class TestScopes:
    def test_histogram_distinguishes_red_from_green(self):
        red = _solid(8, 8, (1, 0, 0))
        green = _solid(8, 8, (0, 1, 0))
        hr, hg = histogram_data(red), histogram_data(green)
        assert hr["r"][255] == 64 and hr["g"][255] == 0
        assert hg["g"][255] == 64 and hg["r"][255] == 0
        assert hr["luma"][76] == 64  # 0.299*255 ≈ 76

    def test_histogram_exact_counts(self):
        a = np.zeros((2, 2, 3))
        a[0, 0] = (1, 1, 1)
        h = histogram_data(a)
        assert h["r"][0] == 3 and h["r"][255] == 1
        assert sum(h["r"]) == 4

    def test_waveform_columns(self):
        a = np.zeros((4, 8, 3))
        a[:, 4:] = 1.0  # right half white
        w = waveform_data(a, width=8)
        assert w[0][0] == 4 and w[255][7] == 4
        assert w[255][0] == 0 and w[0][7] == 0
        assert sum(sum(row) for row in w) == 32

    def test_rgb_parade_channels_separate(self):
        p = rgb_parade_data(_solid(4, 4, (1, 0, 0)), width=4)
        assert p["r"][255][0] == 4 and p["g"][255][0] == 0
        assert p["g"][0][0] == 4 and p["b"][0][0] == 4

    def test_vectorscope_centroids_hand_values(self):
        # BT.601: Cb=(B−Y)/1.772, Cr=(R−Y)/1.402
        cb, cr = vectorscope_centroid(_solid(4, 4, (1, 0, 0)))
        assert abs(cb - (-0.299 / 1.772)) < 1e-9
        assert abs(cr - (0.701 / 1.402)) < 1e-9
        cb, cr = vectorscope_centroid(_solid(4, 4, (0, 1, 0)))
        assert abs(cb - (-0.587 / 1.772)) < 1e-9
        assert abs(cr - (-0.587 / 1.402)) < 1e-9
        cb, cr = vectorscope_centroid(_solid(4, 4, (0, 0, 1)))
        assert abs(cb - (0.886 / 1.772)) < 1e-9
        assert abs(cr - (-0.114 / 1.402)) < 1e-9
        cb, cr = vectorscope_centroid(_solid(4, 4, (0.5, 0.5, 0.5)))
        assert abs(cb) < 1e-9 and abs(cr) < 1e-9

    def test_vectorscope_grid_lands_on_red_cell(self):
        grid = vectorscope_data(_solid(4, 4, (1, 0, 0)), size=128)
        total = sum(sum(row) for row in grid)
        assert total == 16
        # red: Cb≈−0.1687 → col (0.3313*128)=42; Cr=0.5 → top row 0
        assert grid[0][42] == 16

    def test_measure_frame(self):
        m = measure_frame(_solid(4, 4, (1, 0, 0)))
        assert m["channels"]["r"]["mean"] == 1.0
        assert abs(m["luma"]["mean"] - 0.299) < 1e-9
        assert m["clipped_white_pct"] == 0.0
        m2 = measure_frame(_solid(4, 4, (1, 1, 1)))
        assert m2["clipped_white_pct"] == 100.0
        assert m2["mean_saturation"] == 0.0

    def test_scopes_bundle(self):
        s = scopes(_solid(8, 8, (0.2, 0.4, 0.6)))
        assert s["ok"] is True
        assert set(s) >= {"measurements", "histogram", "waveform",
                          "rgb_parade", "vectorscope"}

    def test_scopes_bad_shape_is_named(self):
        s = scopes(np.zeros((8, 8)))
        assert s["ok"] is False and "RGB frame" in s["reason"]

    def test_scopes_for_media_missing_file(self):
        r = scopes_for_media("/nonexistent/definitely.mp4")
        assert r["ok"] is False and "reason" in r

    def test_scopes_for_media_still_image(self, tmp_path):
        from PIL import Image
        p = tmp_path / "red.png"
        Image.new("RGB", (8, 8), (255, 0, 0)).save(p)
        r = scopes_for_media(p)
        assert r["ok"] is True
        assert r["histogram"]["r"][255] == 64
        cb, cr = r["measurements"]["mean_chroma"]["cb"], \
            r["measurements"]["mean_chroma"]["cr"]
        assert abs(cb - (-0.299 / 1.772)) < 1e-6 and abs(cr - 0.5) < 1e-6


# ── convergence: media_edit stills share the table builder ─────────

class TestStillsConvergence:
    def test_grade_lut_matches_legacy_formula(self):
        from nomorals.media_edit.studio import _grade_lut

        def legacy(lift, gamma, gain):
            out = []
            for c in range(3):
                ga = gamma[c] if gamma[c] > 0 else 1.0
                for i in range(256):
                    base = (i / 255.0) * gain[c] + lift[c]
                    v = base ** (1.0 / ga) if base > 0 else 0.0
                    out.append(max(0, min(255, int(round(v * 255)))))
            return out

        for params in [
            ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
            ((0.05, -0.02, 0.0), (1.2, 0.9, 1.0), (1.1, 1.0, 0.95)),
            ((-0.2, 0.1, 0.0), (0.0, 2.0, 1.0), (0.8, 1.3, 1.0)),
        ]:
            assert _grade_lut(*params) == legacy(*params)

    def test_op_grade_neutral_is_unchanged(self):
        from PIL import Image
        from nomorals.media_edit.studio import op_grade
        img = Image.new("RGB", (4, 4), (120, 80, 40))
        out = op_grade(img)
        assert list(out.getdata()) == list(img.getdata())


# ── canonical chain wiring (edit_engine) ───────────────────────────

class TestCanonicalChain:
    def test_named_grade_chain_unchanged(self):
        from nomorals.media.edit_engine.effects import color_grade_chain
        assert color_grade_chain("warm") == \
            "eq=saturation=1.15,colorbalance=rs=0.08:gs=0.03:bs=-0.06"

    def test_cdl_effect_bakes_lut3d_chain(self, tmp_path, monkeypatch):
        from nomorals.media.edit_engine.effects import _fx_color_grade
        monkeypatch.setenv("NM_DEVON_LUT_CACHE", str(tmp_path))
        ctx = {"w": 16, "h": 16, "fps": 30, "duration": 1.0, "seed": 1,
               "beats": []}
        chain = _fx_color_grade(
            {"cdl": {"slope": [1.5, 1.0, 1.0], "power": [1.0, 1.0, 1.0]}},
            ctx)
        assert chain.startswith("lut3d='")
        path = Path(chain[len("lut3d='"):-1])
        assert path.exists()
        lut = load_cube(path)
        out = lut.apply(np.array([[0.5, 0.5, 0.5]]))
        assert np.allclose(out, [[0.75, 0.5, 0.5]], atol=2e-3)

    def test_cdl_chain_explicit_cache_dir(self, tmp_path):
        from nomorals.media.edit_engine.effects import (
            color_grade_cdl_chain)
        chain = color_grade_cdl_chain(
            {"slope": [2.0, 1.0, 1.0]}, cache_dir=str(tmp_path))
        path = Path(chain[len("lut3d='"):-1])
        assert path.parent == tmp_path and path.exists()

    def test_bad_cdl_raises_media_edit_error(self):
        from nomorals.media.edit_engine.effects import (
            color_grade_cdl_chain)
        from nomorals.media_edit.videos import MediaEditError
        with pytest.raises(MediaEditError, match="bad cdl grade"):
            color_grade_cdl_chain({"power": [0.0, 1.0, 1.0]})


# ── media-level ops (ffmpeg e2e on tiny generated media) ───────────

def _make_video(path: Path, rgb: tuple[int, int, int],
                seconds: float = 2.0) -> Path:
    hexcol = "0x%02x%02x%02x" % rgb
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-f", "lavfi",
         "-i", f"color=c={hexcol}:s=16x16:r=5:d={seconds}",
         "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@needs_ffmpeg
class TestMediaOps:
    def test_apply_lut_validates_cube(self, tmp_path):
        bad = tmp_path / "bad.cube"
        bad.write_text("LUT_3D_SIZE 2\n0 0 0\n")
        src = _make_video(tmp_path / "in.mp4", (128, 64, 32))
        r = apply_lut(src, bad, out=str(tmp_path / "o.mp4"))
        assert r["ok"] is False and "invalid .cube" in r["reason"]

    def test_apply_lut_missing(self, tmp_path):
        r = apply_lut("/nonexistent.mp4", "/nonexistent.cube")
        assert r["ok"] is False

    def test_apply_generated_lut_swaps_channels(self, tmp_path):
        cube = tmp_path / "swap.cube"
        generate_cube(17, lambda r, g, b: (b, g, r), path=cube)
        src = _make_video(tmp_path / "in.mp4", (200, 60, 30))
        out = tmp_path / "out.mp4"
        r = apply_lut(src, cube, out=str(out))
        assert r["ok"] is True and r["lut_info"]["size"] == 17
        arr, reason = _grab_frame(str(out), at=0.05)
        assert arr is not None, reason
        mean = arr.reshape(-1, 3).mean(axis=0)
        # (200,60,30) → (30,60,200), modulo codec quantization
        assert abs(mean[0] - 30 / 255) < 0.06
        assert abs(mean[2] - 200 / 255) < 0.06

    def test_color_wheels_are_cdl_now(self, tmp_path):
        src = _make_video(tmp_path / "in.mp4", (128, 128, 128))
        out = tmp_path / "out.mp4"
        r = color_wheels(src, gain=(1.5, 1.0, 1.0), out=str(out))
        assert r["ok"] is True and r["engine"] == "asc-cdl"
        assert r["cdl"]["slope"] == [1.5, 1.0, 1.0]
        arr, _ = _grab_frame(str(out), at=0.05)
        mean = arr.reshape(-1, 3).mean(axis=0)
        assert abs(mean[0] - 192 / 255) < 0.05  # 128 * 1.5
        assert abs(mean[1] - 128 / 255) < 0.04

    def test_color_wheels_bad_gamma_is_named(self, tmp_path):
        src = _make_video(tmp_path / "in.mp4", (128, 128, 128))
        r = color_wheels(src, gamma=(0.0, 1.0, 1.0),
                         out=str(tmp_path / "o.mp4"))
        assert r["ok"] is False and "power must be > 0" in r["reason"]

    def test_apply_cdl_saturation(self, tmp_path):
        src = _make_video(tmp_path / "in.mp4", (200, 60, 30))
        out = tmp_path / "out.mp4"
        r = apply_cdl(src, saturation=0.0, out=str(out))
        assert r["ok"] is True
        arr, _ = _grab_frame(str(out), at=0.05)
        px = arr.reshape(-1, 3).mean(axis=0)
        # desaturated → channels ≈ equal (luma of the source color)
        assert abs(px[0] - px[1]) < 0.05 and abs(px[1] - px[2]) < 0.05

    def test_match_shot_converges(self, tmp_path):
        src = _make_video(tmp_path / "src.mp4", (200, 60, 40))
        ref = _make_video(tmp_path / "ref.mp4", (40, 60, 200))
        out = tmp_path / "out.mp4"
        r = match_shot(src, ref, out=str(out))
        assert r["ok"] is True and r["method"] == "histogram-transfer"
        arr, _ = _grab_frame(str(out), at=0.05)
        mean = arr.reshape(-1, 3).mean(axis=0) * 255
        before = np.array([200.0, 60.0, 40.0])
        target = np.array([40.0, 60.0, 200.0])
        assert np.linalg.norm(mean - target) < \
            0.5 * np.linalg.norm(before - target)

    def test_cdf_tables_converge_on_arrays(self):
        rng = np.random.default_rng(4)
        src = rng.random((32, 32, 3)) * 0.4            # dark frame
        ref = 0.6 + rng.random((32, 32, 3)) * 0.4      # bright frame
        tables = cdf_match_tables(src, ref)
        lut = CubeLUT.from_tables(tables, size=32)
        matched = lut.apply(src)
        d_before = abs(src.mean() - ref.mean())
        d_after = abs(matched.mean() - ref.mean())
        assert d_after < 0.25 * d_before

    def test_scopes_tool_registered_and_runs(self, tmp_path):
        from PIL import Image
        from nomorals.tools.registry import ToolRegistry

        class FakeContext:
            workspace = str(tmp_path)

        reg = ToolRegistry(context=FakeContext()).register_builtins()
        names = {s.get("name") for s in reg.schemas()}
        assert "studio_scopes" in names
        p = tmp_path / "red.png"
        Image.new("RGB", (8, 8), (255, 0, 0)).save(p)
        out = reg.call("studio_scopes", path=str(p))
        assert out.ok and out.value["ok"] is True
        assert out.value["histogram"]["r"][255] == 64
