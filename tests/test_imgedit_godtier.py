"""Phase 8B tests: CPU inpaint routing, auto-mask, outpaint quality,
strength auto-select. All torch-free (the CPU tiers are the point)."""

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from nomorals.media.imggen import ImgGenError
from nomorals.media.imggen import edit
from nomorals.media.imggen import masks


def _gradient_square(size=96):
    a = np.zeros((size, size, 3), np.uint8)
    a[..., 0] = np.arange(size)[None, :].repeat(size, 0) * 2
    a[..., 1] = np.arange(size)[:, None].repeat(size, 1) * 2
    a[..., 2] = 128
    img = Image.fromarray(a)
    ImageDraw.Draw(img).rectangle([30, 30, 60, 60], fill=(255, 0, 0))
    return img


def _rect_mask(size=96, box=(32, 32, 58, 58)):
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rectangle(box, fill=255)
    return m


# ---------------------------------------------------------------------------
# suggest_strength
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("desc,lo,hi", [
    ("touch up the scratch", 0.2, 0.3),
    ("remove the watermark", 0.2, 0.3),
    ("recolor the car blue", 0.35, 0.45),
    ("replace the dog with a cat", 0.45, 0.55),
    ("change the background to a beach", 0.5, 0.6),
    ("in the style of anime", 0.65, 0.75),
    ("redraw it completely", 0.75, 0.85),
    ("make it better", 0.5, 0.6),  # no signal → balanced default
])
def test_suggest_strength_mapping(desc, lo, hi):
    s, why = edit.suggest_strength(desc)
    assert lo <= s <= hi, (desc, s)
    assert isinstance(why, str) and why
    assert 0.0 < s < 1.0


def test_img2img_rejects_bad_strength_string():
    # string validation fires before the torch gate (fail fast)
    with pytest.raises(ImgGenError):
        edit.img2img(object(), _gradient_square(), "x", strength="bogus")
    with pytest.raises(ImgGenError):
        edit.inpaint_cpu(_gradient_square(), _rect_mask(),
                         method="bogus")


# ---------------------------------------------------------------------------
# CPU inpaint routing
# ---------------------------------------------------------------------------

def test_inpaint_cpu_numpy_fills_and_preserves():
    img = _gradient_square()
    mask = _rect_mask()
    out, label = edit.inpaint_cpu(img, mask, method="numpy")
    assert "non-neural" in label  # honest tier label
    oa, ia = np.asarray(out).astype(int), np.asarray(img).astype(int)
    # unmasked corner identical
    assert np.array_equal(oa[5, 5], ia[5, 5])
    # masked region filled with *something* (not left black/empty)
    hole = oa[40:52, 40:52]
    assert hole.mean() > 10
    # the hole sits inside the red square, whose unmasked ring is the
    # nearest known content → the fill must continue the red, not the
    # distant gradient
    assert hole[..., 0].mean() > 200
    assert hole[..., 1].mean() < 60


def test_inpaint_cpu_empty_mask_is_noop():
    img = _gradient_square()
    out, label = edit.inpaint_cpu(img, Image.new("L", (96, 96), 0),
                                  method="numpy")
    assert np.array_equal(np.asarray(out), np.asarray(img))
    assert "nothing masked" in label


def test_inpaint_auto_without_pipeline_uses_cpu_honestly():
    img = _gradient_square()
    out, label = edit.inpaint_auto(img, _rect_mask(), pipeline=None)
    assert "non-neural" in label
    assert "neural-blended" not in label
    assert out.size == img.size


def test_neural_paths_fail_honestly_without_torch():
    img = _gradient_square()
    with pytest.raises(ImgGenError, match="[Tt]orch"):
        edit.img2img(object(), img, "x")
    with pytest.raises(ImgGenError, match="[Tt]orch"):
        edit.inpaint(object(), img, _rect_mask(), "x")
    with pytest.raises(ImgGenError, match="[Tt]orch"):
        edit.outpaint(object(), img, "x", right=8)


# ---------------------------------------------------------------------------
# auto-mask
# ---------------------------------------------------------------------------

def test_box_mask_feather_edges():
    m = masks.box_mask((64, 64), (10, 10, 30, 30), feather_radius=6)
    a = np.asarray(m)
    assert a[20, 20] > 200              # clearly inside
    assert a[0, 0] < 8                  # deep outside (blur halo only)
    edge_vals = set(np.unique(a[8:12, 18:22]).tolist())
    assert any(0 < v < 255 for v in edge_vals), "feather must soften edges"
    with pytest.raises(ImgGenError):
        masks.box_mask((64, 64), (10, 10, 10, 10))


def test_auto_mask_box_tier():
    res = masks.auto_mask(_gradient_square(), box=(30, 30, 61, 61))
    assert res.tier == "box+feather"
    a = np.asarray(res.mask)
    assert a[45, 45] > 200 and a[0, 0] == 0


def test_auto_mask_flood_covers_solid_region():
    img = Image.new("RGB", (64, 64), (10, 10, 10))
    ImageDraw.Draw(img).rectangle([20, 20, 40, 40], fill=(200, 50, 50))
    res = masks.auto_mask(img, point=(30, 30), tier="flood")
    assert res.tier == "flood"
    a = np.asarray(res.mask) > 127
    assert a[30, 30] and not a[5, 5]
    # stays inside the solid square (no leak into the dark bg)
    assert a.sum() < 30 * 30


def test_auto_mask_point_outside_raises():
    with pytest.raises(ImgGenError):
        masks.auto_mask(_gradient_square(), point=(999, 999),
                        tier="flood")


def test_auto_mask_text_without_locator_raises_honestly():
    with pytest.raises(ImgGenError, match="box_predictor"):
        masks.auto_mask(_gradient_square(), text="the red car")


def test_auto_mask_text_with_locator_uses_box_tier():
    res = masks.auto_mask(
        _gradient_square(), text="the red square",
        box_predictor=lambda img, text: (30, 30, 61, 61))
    assert res.tier == "box+feather"


def test_sam_status_is_honest():
    st = masks.sam_status()
    assert set(st) >= {"package", "checkpoint", "usable", "reason"}
    assert st["usable"] == (st["package"] is not None
                            and st["checkpoint"] is not None)


# ---------------------------------------------------------------------------
# outpaint quality
# ---------------------------------------------------------------------------

def test_edge_extend_fill_size_and_continuity():
    img = _gradient_square()
    ext = edit.edge_extend_fill(img, right=24, bottom=16)
    assert ext.size == (120, 112)
    # the original pixels are untouched by the fill
    assert np.array_equal(np.asarray(ext)[:96, :96],
                          np.asarray(img))


def test_pyramid_blend_identity():
    rng = np.random.default_rng(3)
    a = Image.fromarray(rng.integers(0, 256, (48, 48, 3), np.uint8))
    b = Image.fromarray(rng.integers(0, 256, (48, 48, 3), np.uint8))
    white = Image.new("L", (48, 48), 255)
    black = Image.new("L", (48, 48), 0)
    assert np.array_equal(np.asarray(edit.pyramid_blend(a, b, white)),
                          np.asarray(a))
    assert np.array_equal(np.asarray(edit.pyramid_blend(a, b, black)),
                          np.asarray(b))


def test_pyramid_blend_beats_naive_seam():
    img = _gradient_square()
    ext = edit.edge_extend_fill(img, right=24)
    # naive: hard paste of a flat fill at the seam
    naive = Image.new("RGB", (120, 96), (0, 0, 0))
    naive.paste(img, (0, 0))
    naive.paste(Image.new("RGB", (24, 96), (200, 200, 200)), (96, 0))
    naive_seam = edit.seam_metric(naive, 96)
    keep = Image.new("L", (120, 96), 255)
    keep.paste(Image.new("L", (96, 96), 0), (0, 0))
    keep = keep.filter(ImageFilter.GaussianBlur(8))
    blended = edit.pyramid_blend(ext, naive, keep, levels=4)
    assert edit.seam_metric(blended, 96) < naive_seam


def test_outpaint_cpu_size_label_and_seam():
    img = _gradient_square()
    out, label = edit.outpaint_cpu(img, right=24, bottom=24)
    assert out.size == (120, 120)
    assert "non-neural" in label
    # seam nearly invisible: ratio close to 1
    assert edit.seam_metric(out, 96) < 2.0
    # deep interior: high-frequency detail preserved; only a faint
    # low-frequency tone blend from the pyramid (inherent to
    # multi-resolution blending, mean drift < 2 levels)
    oa, ia = np.asarray(out).astype(int), np.asarray(img).astype(int)
    drift = np.abs(oa[40:56, 40:56] - ia[40:56, 40:56]).mean()
    assert drift < 2.0, drift


def test_seam_metric_sane_on_clean_image():
    img = _gradient_square()
    m = edit.seam_metric(img, 48)
    assert 0.5 < m < 3.0  # no seam → near baseline


def test_feather_mask_edges():
    m = Image.new("L", (32, 32), 0)
    ImageDraw.Draw(m).rectangle([8, 8, 24, 24], fill=255)
    f = edit.feather_mask(m, radius=4)
    a = np.asarray(f)
    assert a[16, 16] > 200 and a[0, 0] == 0
    assert any(0 < v < 255 for v in np.unique(a))
    # radius 0 keeps it binary
    b = np.asarray(edit.feather_mask(m, radius=0))
    assert set(np.unique(b).tolist()) <= {0, 255}


# ---------------------------------------------------------------------------
# ControlNet-style conditioning interface (torch-free parts)
# ---------------------------------------------------------------------------

def test_control_condition_validation():
    from nomorals.media.imggen.pipeline import ControlCondition
    c = ControlCondition("canny", Image.new("L", (8, 8), 0))
    assert c.kind == "canny" and c.strength == 1.0
    with pytest.raises(ImgGenError):
        ControlCondition("scribble", None)
    with pytest.raises(ImgGenError):
        ControlCondition("depth", None, strength=3.0)


def test_extract_canny_finds_real_edges():
    from nomorals.media.imggen.pipeline import extract_canny
    img = Image.new("RGB", (64, 64), (20, 20, 20))
    ImageDraw.Draw(img).rectangle([16, 16, 48, 48],
                                  fill=(220, 220, 220))
    e = extract_canny(img)
    assert e.size == (64, 64) and e.mode == "L"
    ea = np.asarray(e)
    n = (ea > 127).sum()
    assert 60 < n < 400           # ~perimeter, not noise, not empty
    assert ea[16, 16] == 255      # corner is an edge
    assert ea[32, 32] == 0        # solid center is not
    assert ea[4, 4] == 0          # solid background is not


def test_extract_canny_blank_is_empty():
    from nomorals.media.imggen.pipeline import extract_canny
    e = extract_canny(Image.new("RGB", (32, 32), (50, 50, 50)))
    assert (np.asarray(e) > 127).sum() == 0


def test_extract_depth_pose_fail_honestly():
    from nomorals.media.imggen.pipeline import extract_depth, extract_pose
    img = Image.new("RGB", (16, 16), (0, 0, 0))
    with pytest.raises(ImgGenError, match="depth"):
        extract_depth(img)
    with pytest.raises(ImgGenError, match="pose"):
        extract_pose(img)


def test_attach_controlnet_validates():
    from unittest.mock import MagicMock

    from nomorals.media.imggen.pipeline import (
        ControlCondition, NativePipeline)
    # can't build a real pipeline without torch — exercise the
    # registration logic on an uninitialized instance
    pipe = NativePipeline.__new__(NativePipeline)
    pipe.controlnets = {}
    with pytest.raises(ImgGenError):
        pipe.attach_controlnet("scribble", MagicMock())
    with pytest.raises(ImgGenError):
        pipe.attach_controlnet("canny", object())  # not callable
    fake = MagicMock()
    pipe.attach_controlnet("canny", fake)
    assert pipe.controlnets["canny"] is fake
    # conditioning without an attached net fails honestly
    pipe.controlnets = {}
    with pytest.raises(ImgGenError, match="no controlnet attached"):
        pipe._check_conditioning(
            [ControlCondition("canny", Image.new("L", (8, 8), 0))])
    # ...and passes once attached
    pipe.attach_controlnet("canny", fake)
    out = pipe._check_conditioning(
        [ControlCondition("canny", Image.new("L", (8, 8), 0))])
    assert len(out) == 1
    with pytest.raises(ImgGenError):
        pipe._check_conditioning(["not-a-condition"])


def test_attach_diffusers_controlnet_needs_diffusers():
    from unittest.mock import MagicMock

    from nomorals.media.imggen.pipeline import NativePipeline
    pipe = NativePipeline.__new__(NativePipeline)
    pipe.controlnets = {}
    pipe.device = "cpu"
    try:
        import diffusers  # noqa: F401
        has_diffusers = True
    except ImportError:
        has_diffusers = False
    if not has_diffusers:
        with pytest.raises(ImgGenError, match="diffusers"):
            pipe.attach_diffusers_controlnet("canny", MagicMock())
    else:
        pipe.attach_diffusers_controlnet("canny", MagicMock())
        assert "canny" in pipe.controlnets


# ---------------------------------------------------------------------------
# creativity slider
# ---------------------------------------------------------------------------

def test_creativity_to_strength_mapping():
    from nomorals.media.imggen.upscale import creativity_to_strength
    assert creativity_to_strength(0.0) == pytest.approx(0.12)
    assert creativity_to_strength(1.0) == pytest.approx(0.62)
    mid = creativity_to_strength(0.5)
    assert 0.12 < mid < 0.62
    assert creativity_to_strength(0.25) < creativity_to_strength(0.75)
    with pytest.raises(ImgGenError):
        creativity_to_strength(1.5)
    with pytest.raises(ImgGenError):
        creativity_to_strength(-0.1)


def test_upscale_classical_still_fine():
    from nomorals.media.imggen.upscale import upscale_classical
    img = Image.new("RGB", (32, 32), (90, 110, 130))
    out = upscale_classical(img, scale=2.0)
    assert out.size == (64, 64)
    with pytest.raises(ImgGenError):
        upscale_classical(img, scale=0)
