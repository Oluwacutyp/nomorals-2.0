"""52-websites #16 Unscreen — video background removal effect.

The effect composes the bgremove subject mask (anchor frame) +
mask_track (translation tracking) + an ffmpeg composite (subject
over a replacement colour or image), recorded in effects_applied
like every Studio effect. The fixtures are synthetic clips whose
ground truth is exact by construction: a textured red square
moving on a solid blue field.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from nomorals.media.edit_engine.effects import (
    effect_properties,
    effect_rung,
    list_effects,
    validate_effect_params,
)
from nomorals.media.edit_engine.render import apply_effect, render_timeline
from nomorals.media.edit_engine.timeline import Clip, Timeline, Track
from nomorals.media.edit_engine.video_four import (
    bg_color,
    remove_video_background,
    remove_video_background_params,
)
from nomorals.media.imggen import ImgGenError
from nomorals.media.imggen.masks import MaskRef
from nomorals.media_edit.images import MediaEditError

FFMPEG = "/usr/bin/ffmpeg"
W, H, FPS, N = 64, 64, 12, 12


def _frames(tmp: Path, *, cut: bool = False, vanish: bool = False,
            blank: bool = False) -> list[Image.Image]:
    """Synthetic clip frames: a textured red square on blue.

    The square starts centred (so the bgremove centre seed lands
    on it) and translates +1px/frame. ``cut`` swaps the scene
    mid-clip (green field, yellow square); ``vanish`` drops the
    subject after frame 0; ``blank`` is a featureless blue field.
    """
    out: list[Image.Image] = []
    for i in range(N):
        if blank:
            out.append(Image.new("RGB", (W, H), (0, 0, 255)))
            continue
        if cut and i >= N // 2:
            # a different world entirely (like the tracker's own
            # scene_cut corpus fixture): white field, black square
            # at a different position — the red/blue template must
            # not match across the cut, and the frame difference
            # must clear scene-intel's 0.4 scene floor.
            im = Image.new("RGB", (W, H), (255, 255, 255))
            d = ImageDraw.Draw(im)
            d.rectangle([32, 30, 53, 51], fill=(0, 0, 0))
            d.rectangle([36, 34, 38, 36], fill=(255, 255, 255))
            d.rectangle([44, 42, 46, 44], fill=(255, 255, 255))
            out.append(im)
            continue
        if vanish and i > 0:
            out.append(Image.new("RGB", (W, H), (0, 0, 255)))
            continue
        x0 = 10 + i
        im = Image.new("RGB", (W, H), (0, 0, 255))
        d = ImageDraw.Draw(im)
        d.rectangle([x0, 18, x0 + 27, 45], fill=(255, 0, 0))
        for dx, dy in ((4, 4), (16, 8), (8, 16), (18, 18)):
            d.rectangle([x0 + dx, 18 + dy, x0 + dx + 2, 18 + dy + 2],
                        fill=(255, 255, 255))
        out.append(im)
    return out


def _clip(tmp: Path, name: str = "clip.mp4", **kw) -> Path:
    """Write the synthetic frames out as an mp4 (12 fps, 1 s)."""
    d = tmp / name.replace(".mp4", "")
    d.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(_frames(tmp, **kw)):
        im.save(str(d / f"f{i:05d}.png"))
    out = tmp / name
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-framerate", str(FPS),
         "-i", str(d / "f%05d.png"), "-c:v", "libx264",
         "-pix_fmt", "yuv420p", str(out)],
        check=True)
    return out


def _frame_at(path: Path, t: float, tmp: Path) -> Image.Image:
    """The rendered frame at ``t`` seconds, as an RGB image."""
    png = tmp / f"probe_{t}.png"
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-ss", f"{t:.3f}", "-i",
         str(path), "-frames:v", "1", str(png)],
        check=True)
    with Image.open(png) as im:
        return im.convert("RGB").copy()


def _is_red(px: tuple[int, int, int]) -> bool:
    r, g, b = px
    return r > 150 and g < 110 and b < 110


def _is_green(px: tuple[int, int, int]) -> bool:
    r, g, b = px
    return g > 150 and r < 110 and b < 110


# ── the mask rung ──────────────────────────────────────────────

def test_maskref_subject_resolves_to_the_subject(tmp_path):
    anchor = _frames(tmp_path)[0]
    res = MaskRef(kind="subject", tier="auto").resolve(anchor)
    assert res.tier, "the resolving rung must be named"
    assert res.detail, "the resolving rung must explain itself"
    mask = res.mask.convert("L")
    # the square (28x28 of 64x64) is the mask; the blue field is not
    px = mask.load()
    assert px[24, 31] > 127, "square interior must be masked in"
    assert px[2, 2] < 128, "blue corner must be masked out"
    assert px[60, 60] < 128, "blue corner must be masked out"
    coverage = sum(1 for x in range(W) for y in range(H)
                   if px[x, y] > 127) / (W * H)
    assert 0.10 < coverage < 0.40, coverage


def test_maskref_subject_unknown_kind_raises():
    with pytest.raises(ImgGenError, match="subject"):
        MaskRef.from_dict({"kind": "nope"})


# ── the registry ───────────────────────────────────────────────

def test_effect_is_registered_and_honest():
    assert "remove_video_background" in list_effects()
    props = effect_properties()["effects"]["remove_video_background"]
    assert props["rung"] == "edit_engine.render._chain_bg_replace"
    assert sorted(props["params"]) == [
        "background", "background_image", "segmentation"]
    assert effect_rung("remove_video_background") == \
        "edit_engine.render._chain_bg_replace"
    validate_effect_params("remove_video_background",
                           {"background": "#0000ff",
                            "segmentation": "tracked"})
    with pytest.raises(MediaEditError, match="segmentation"):
        validate_effect_params("remove_video_background",
                               {"segmentation": "sideways"})


def test_bg_color_names_and_hex():
    assert bg_color("") == "#00ff00"
    assert bg_color("blue") == "#0000ff"
    assert bg_color("#A0B0C0") == "#a0b0c0"
    with pytest.raises(MediaEditError, match="background colour"):
        bg_color("not-a-colour")


def test_descriptor_helper_carries_subject_mask_and_shots(tmp_path):
    clip = _clip(tmp_path)
    out = remove_video_background(Clip(path=str(clip)),
                                  background="blue")
    assert len(out.effects) == 1
    fx = out.effects[0]
    assert fx.name == "remove_video_background"
    assert fx.params["mask"]["kind"] == "subject"
    assert fx.params["mask_track"] is True
    assert fx.params["background"] == "#0000ff"
    assert fx.params["segmentation"] == "auto"
    # one scene, no cut inside the window
    assert fx.params.get("_shots", {}).get("spans_cut") is False


# ── the render: tracked path ───────────────────────────────────

def test_tracked_clip_replaces_background_keeps_subject(tmp_path):
    clip = _clip(tmp_path)
    res = apply_effect(
        clip, "remove_video_background", fps=FPS,
        **remove_video_background_params(str(clip),
                                         background="#00ff00"))
    out = Path(res["output"])
    assert out.exists()
    assert res["effects_applied"], "the record must exist"
    rec = res["effects_applied"][0]
    assert rec["effect"] == "remove_video_background"
    assert rec["rung"] == "edit_engine.render._chain_bg_replace"
    assert rec["background"] == {"kind": "color", "value": "#00ff00"}
    assert rec["mask"]["rung"], "the mask rung must be named"
    assert rec["mask"]["tracking"]["status"] == "tracked", rec
    assert rec["mask_path"] == "tracked", rec
    # frame 6: square at x=16..43 — centre still red, corner green
    mid = _frame_at(out, 0.5, tmp_path)
    assert _is_red(mid.getpixel((29, 31))), "subject must survive"
    assert _is_green(mid.getpixel((2, 2))), "background must go green"
    assert _is_green(mid.getpixel((60, 60))), "background must go green"
    # the first and last frames agree: the mask followed the subject
    first = _frame_at(out, 0.05, tmp_path)
    assert _is_red(first.getpixel((24, 31)))
    assert _is_green(first.getpixel((60, 2)))


def test_image_background_behind_subject(tmp_path):
    clip = _clip(tmp_path)
    bg = tmp_path / "bg.png"
    Image.new("RGB", (W, H), (255, 255, 0)).save(str(bg))
    res = apply_effect(
        clip, "remove_video_background", fps=FPS,
        **remove_video_background_params(
            str(clip), background_image=str(bg)))
    rec = res["effects_applied"][0]
    assert rec["background"] == {"kind": "image", "value": str(bg)}
    assert rec["mask"]["tracking"]["status"] == "tracked", rec
    mid = _frame_at(Path(res["output"]), 0.5, tmp_path)
    px = mid.getpixel((2, 2))
    assert px[0] > 150 and px[1] > 150 and px[2] < 110, px
    assert _is_red(mid.getpixel((29, 31)))


def test_windowed_effect_only_inside_its_window(tmp_path):
    clip = _clip(tmp_path)
    params = remove_video_background_params(str(clip))
    res = apply_effect(clip, "remove_video_background", at=(0.25, 0.75),
                       fps=FPS, **params)
    rec = res["effects_applied"][0]
    assert rec["window"] == [0.25, 0.75]
    out = Path(res["output"])
    before = _frame_at(out, 0.1, tmp_path)
    assert not _is_green(before.getpixel((2, 2))), \
        "before the window the background must stay blue"
    inside = _frame_at(out, 0.5, tmp_path)
    assert _is_green(inside.getpixel((2, 2)))
    assert _is_red(inside.getpixel((29, 31)))


def test_per_frame_path_segments_every_frame(tmp_path):
    clip = _clip(tmp_path)
    res = apply_effect(
        clip, "remove_video_background", fps=FPS,
        **remove_video_background_params(str(clip),
                                         segmentation="per-frame"))
    rec = res["effects_applied"][0]
    assert rec["mask_path"] == "per-frame segmentation", rec
    assert rec["mask"]["tracking"]["status"] == "per-frame", rec
    mid = _frame_at(Path(res["output"]), 0.5, tmp_path)
    assert _is_red(mid.getpixel((29, 31)))
    assert _is_green(mid.getpixel((2, 2)))


# ── the record when tracking cannot follow ─────────────────────

def test_cut_clip_reports_partial_and_names_the_shots(tmp_path):
    clip = _clip(tmp_path, name="cut.mp4", cut=True)
    clip = remove_video_background(Clip(path=str(clip)))
    assert clip.effects[0].params["_shots"]["spans_cut"] is True
    cuts = clip.effects[0].params["_shots"]["cuts"]
    assert cuts and abs(cuts[0] - 0.5) < 0.1, cuts
    res = apply_effect(clip.path, "remove_video_background", fps=FPS,
                       **clip.effects[0].params)
    rec = res["effects_applied"][0]
    assert rec["mask"]["tracking"]["status"] == "partial", rec
    assert rec["mask_path"] == "partially tracked", rec
    assert rec["shots"]["spans_cut"] is True


def test_vanishing_subject_falls_back_to_per_frame(tmp_path):
    clip = _clip(tmp_path, name="vanish.mp4", vanish=True)
    res = apply_effect(
        clip, "remove_video_background", fps=FPS,
        **remove_video_background_params(str(clip)))
    rec = res["effects_applied"][0]
    # the anchor tracks nothing once the subject is gone; auto
    # degrades to per-frame segmentation and says so
    assert rec["mask_path"] == "per-frame segmentation", rec
    assert rec["mask"]["tracking"]["status"] == "per-frame", rec


def test_featureless_clip_holds_and_names_texture(tmp_path):
    clip = _clip(tmp_path, name="blank.mp4", blank=True)
    res = apply_effect(
        clip, "remove_video_background", fps=FPS,
        **remove_video_background_params(str(clip),
                                         segmentation="tracked"))
    rec = res["effects_applied"][0]
    assert rec["mask"]["tracking"]["status"] == "held", rec
    assert "texture" in rec["mask"]["tracking"]["detail"], rec
    assert rec["mask_path"] == "anchor-and-hold", rec


def test_still_image_holds_with_the_reason(tmp_path):
    still = tmp_path / "still.png"
    _frames(tmp_path)[0].save(str(still))
    res = apply_effect(
        still, "remove_video_background", still_duration=1.0,
        **remove_video_background_params(str(still)))
    rec = res["effects_applied"][0]
    assert rec["mask"]["tracking"]["status"] == "held", rec
    assert "still image" in rec["mask"]["tracking"]["detail"], rec
    assert rec["mask_path"] == "anchor-and-hold", rec


# ── the refusals ───────────────────────────────────────────────

def test_effect_without_a_mask_refuses_by_name(tmp_path):
    clip = _clip(tmp_path)
    with pytest.raises(MediaEditError, match="subject mask"):
        apply_effect(clip, "remove_video_background",
                     background="#00ff00")


def test_bad_background_colour_refuses_by_name(tmp_path):
    clip = _clip(tmp_path)
    params = remove_video_background_params(str(clip))
    params["background"] = "banana"
    with pytest.raises(MediaEditError, match="#RRGGBB"):
        apply_effect(clip, "remove_video_background", **params)


def test_missing_background_image_refuses_by_name(tmp_path):
    clip = _clip(tmp_path)
    with pytest.raises(MediaEditError, match="no such background image"):
        apply_effect(
            clip, "remove_video_background",
            **remove_video_background_params(
                str(clip), background_image=str(tmp_path / "nope.png")))


# ── the Studio flow: timeline render + brain tool ──────────────

def test_timeline_render_carries_the_effect_record(tmp_path):
    clip = _clip(tmp_path)
    clip = remove_video_background(Clip(path=str(clip)),
                                   background="blue")
    tl = Timeline(video=Track(clips=[clip]), width=W, height=H, fps=FPS)
    out = tmp_path / "tl.mp4"
    res = render_timeline(tl, out=str(out), keep_clip_audio=False)
    assert out.exists()
    assert res["effects_applied"], "the timeline record must exist"
    rec = res["effects_applied"][0]
    assert rec["effect"] == "remove_video_background"
    assert rec["mask"]["tracking"]["status"] == "tracked", rec
    mid = _frame_at(out, 0.5, tmp_path)
    assert _is_red(mid.getpixel((29, 31)))
    px = mid.getpixel((2, 2))
    assert px[2] > 150 and px[0] < 110, px  # blue replacement


def test_studio_tool_renders_and_returns_the_record(tmp_path):
    from nomorals.tools import studio as studio_mod

    seen: dict[str, object] = {}

    class _Ctx:
        def __init__(self, workspace):
            self.workspace = workspace

    class _Reg:
        def __init__(self, ctx):
            self.context = ctx

        def register(self, name, *, description="", capability=None):
            def deco(fn):
                seen[name] = fn
                return fn
            return deco

    studio_mod.register(_Reg(_Ctx(tmp_path)))
    fn = seen["studio_remove_background"]
    clip = _clip(tmp_path)
    res = fn(str(clip), background="green")
    out = Path(res["output"])
    assert out.exists()
    assert res["effects_applied"], "the tool must return the record"
    rec = res["effects_applied"][0]
    assert rec["effect"] == "remove_video_background"
    assert rec["mask"]["tracking"]["status"] == "tracked", rec
    mid = _frame_at(out, 0.5, tmp_path)
    assert _is_red(mid.getpixel((29, 31)))
    assert _is_green(mid.getpixel((2, 2)))
