"""The three Wave-5 effects remainders, adjudicated.

1. **Mask tracking across frames** — a masked effect whose
   ``mask_track`` is set follows its subject through the clip
   (the patch moves, the grade moves with it), and a mask that
   cannot be followed holds at the anchor with the record saying
   so. Corpus-level tracker quality is pinned separately in
   ``tests/test_mask_track_corpus.py``.
2. **Animated CDL components** — the ten ``cdl_*`` scalar paths
   bind keyframe tracks through the ``sampled`` family; several
   components on one clip discretize JOINTLY into single
   sub-effects (one CDL bake per window, never sequential
   re-grades), and a mid-animation frame measures between its
   endpoints in the expected direction, against values predicted
   by the canonical CDL math.
3. **Regional velocity_ramp** — stays refused, and the refusal is
   now *proven*, not asserted: the velocity rung rewrites the
   stream timeline (measured duration change), so after any ramp
   the effected branch's frame at an output instant carries a
   different source instant than the base it would composite over
   (measured) — the mask composite pairs frames on one shared
   timeline, and no rung resamples a retimed branch back onto it.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from nomorals.media.edit_engine import (
    Clip,
    Effect,
    Keyframe,
    KeyframeTrack,
    Timeline,
    Track,
)
from nomorals.media.edit_engine.effects import _ctx, _effect_filter
from nomorals.media.edit_engine.keyframes import discretize
from nomorals.media.edit_engine.render import (
    _chain_effect,
    apply_effect,
    render_timeline,
)
from nomorals.media.studio_color import CDL
from nomorals.media_edit.videos import MediaEditError, run_ffmpeg, video_probe
from tests.edit_engine_golden import color_source, diff_stats, frame_stats

NEEDS_FFMPEG = pytest.mark.skipif(
    os.system("ffmpeg -hide_banner -version > /dev/null 2>&1") != 0,
    reason="ffmpeg not installed",
)

W, H, FPS = 320, 240, 30


@pytest.fixture()
def workdir(tmp_path):
    d = tmp_path / "rem"
    d.mkdir()
    return d


def _frame(video: str, t: float, workdir: Path):
    from PIL import Image
    png = Path(workdir) / f"f_{t:.3f}.png"
    run_ffmpeg(["-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
                "-y", str(png)])
    return Image.open(png).convert("RGB")


def _region(img, box) -> tuple[float, float, float]:
    crop = img.crop(box)
    data = list(crop.getdata())
    n = len(data)
    return tuple(sum(c[i] for c in data) / n for i in range(3))


def _spread(rgb) -> float:
    return max(rgb) - min(rgb)


# ---------------------------------------------------------------------------
# animated CDL — composition rules at the builder
# ---------------------------------------------------------------------------

class TestCDLCompose:
    def test_components_compose_over_identity(self):
        from nomorals.media.edit_engine.effects import compose_cdl
        out = compose_cdl({"cdl_slope_r": 2.0})
        assert out["slope"] == [2.0, 1.0, 1.0]
        assert out["saturation"] == 1.0

    def test_components_compose_over_cdl_dict(self):
        from nomorals.media.edit_engine.effects import compose_cdl
        out = compose_cdl({"cdl": {"offset": [0.1, 0.0, 0.0]},
                           "cdl_saturation": 0.5})
        assert out["offset"] == [0.1, 0.0, 0.0]
        assert out["saturation"] == 0.5

    def test_grade_and_components_conflict_raises(self):
        ctx = _ctx(W, H, FPS, 1.0, 0)
        with pytest.raises(MediaEditError) as ei:
            _effect_filter("color_grade",
                           {"grade": "warm", "cdl_slope_r": 2.0}, ctx)
        assert "two different decisions" in str(ei.value)

    def test_power_zero_refused_by_name_at_the_bake(self):
        ctx = _ctx(W, H, FPS, 1.0, 0)
        with pytest.raises(MediaEditError) as ei:
            _effect_filter("color_grade", {"cdl_power_g": 0.0}, ctx)
        assert "power must be > 0" in str(ei.value)

    def test_static_component_pixel_matches_cdl_math(self, workdir):
        src = color_source(workdir / "solid.mp4", "0x78603C",
                           w=W, h=H, seconds=0.5)
        out = workdir / "graded.mp4"
        apply_effect(src, "color_grade", out=str(out),
                     cdl_slope_r=1.6, cdl_saturation=0.6)
        got = _region(_frame(str(out), 0.2, workdir), (0, 0, W, H))
        want = CDL(slope=(1.6, 1.0, 1.0), saturation=0.6).apply_array(
            np.array([[[120 / 255, 96 / 255, 60 / 255]]]))[0, 0] * 255
        for g, w_ in zip(got, want, strict=True):
            assert abs(g - w_) <= 10.0, (got, want)


# ---------------------------------------------------------------------------
# animated CDL — the render: joint discretization, measured mid-frame
# ---------------------------------------------------------------------------

@NEEDS_FFMPEG
class TestAnimatedCDLRender:
    BASE = {"cdl": {"slope": [1, 1, 1], "offset": [0, 0, 0],
                    "power": [1, 1, 1], "saturation": 1.0}}
    def _tracks(self):
        slope = KeyframeTrack(
            clip_id="c0", effect="color_grade", param="cdl_slope_r",
            keys=[Keyframe(0.0, 1.0, "linear"),
                  Keyframe(1.0, 1.8, "linear")])
        sat = KeyframeTrack(
            clip_id="c0", effect="color_grade", param="cdl_saturation",
            keys=[Keyframe(0.0, 1.0, "linear"),
                  Keyframe(1.0, 0.2, "linear")])
        return slope, sat

    def _predicted(self, tracks, t, src_rgb):
        """CDL output for the window containing t (the renderer's
        own sampling rule: each track at the window midpoint),
        computed by the canonical CDL math on the source color as
        actually decoded (the lavfi hex → yuv roundtrip shifts it
        a few counts; the oracle measures it instead of assuming)."""
        vals = {}
        for tr in tracks:
            win = next((a, b) for a, b, _v in discretize(tr, fps=FPS)
                       if a <= t < b)
            vals[tr.param] = tr.sample((win[0] + win[1]) / 2.0)
        cdl = CDL(slope=(vals["cdl_slope_r"], 1.0, 1.0),
                  saturation=vals["cdl_saturation"])
        arr = np.array([[tuple(c / 255 for c in src_rgb)]])
        return tuple(float(v) * 255 for v in cdl.apply_array(arr)[0, 0])

    def _render(self, workdir, tracks, name="out.mp4"):
        src = color_source(workdir / "solid.mp4", "0x78603C",
                           w=W, h=H, seconds=1.0)
        clip = Clip(path=src, id="c0",
                    effects=(Effect(name="color_grade",
                                    params=dict(self.BASE)),))
        tl = Timeline(video=Track(kind="video", clips=[clip]),
                      keyframes=list(tracks),
                      width=W, height=H, fps=FPS)
        out = workdir / name
        res = render_timeline(tl, out=out, workdir=workdir / "wd")
        return str(out), res

    def test_mid_frame_grade_is_between_endpoints(self, workdir):
        tracks = self._tracks()
        out, res = self._render(workdir, tracks)
        entries = [e for e in res["effects_applied"]
                   if e["kind"] == "keyframe-sampled"]
        assert {e["source"] for e in entries} == {
            "keyframe:cdl_slope_r", "keyframe:cdl_saturation"}
        # one joint discretization: both components report the
        # same window count (they were baked together, per window)
        assert len({e["windows"] for e in entries}) == 1
        assert entries[0]["windows"] >= 4
        src_rgb = _region(
            _frame(str(workdir / "solid.mp4"), 0.01, workdir),
            (0, 0, W, H))
        times = (0.01, 0.5, 0.96)
        got = [_region(_frame(out, t, workdir), (0, 0, W, H))
               for t in times]
        for t, g in zip(times, got, strict=True):
            want = self._predicted(tracks, t, src_rgb)
            for gv, wv in zip(g, want, strict=True):
                assert abs(gv - wv) <= 12.0, (t, g, want)
        # the expected directions, on the measured frames. (Not
        # red: slope_r raises red while saturation pulls it back
        # to luma — the two components genuinely fight there, and
        # the prediction check above is what pins red. Blue is
        # the clean readout: its slope never moves, so blue rides
        # luma up as saturation falls — strictly rising, with the
        # mid frame strictly between its endpoints; and the end
        # state's spread is far below the start's, saturation's
        # signature.)
        assert got[0][2] < got[1][2] < got[2][2]
        assert _spread(got[2]) < _spread(got[0]) - 15
        assert abs(got[1][2] - got[0][2]) > 5
        assert abs(got[1][2] - got[2][2]) > 5

    def test_joint_two_paths_one_result(self, workdir):
        """The renderer's joint binding vs a caller-built windowed
        assembly of the same union windows (the 1.2c oracle)."""
        tracks = self._tracks()
        via_engine, _res = self._render(workdir, tracks, "engine.mp4")
        src = color_source(workdir / "solid2.mp4", "0x78603C",
                           w=W, h=H, seconds=1.0)
        bounds = sorted({x for tr in tracks
                         for a, b, _v in discretize(tr, fps=FPS)
                         for x in (a, b)})
        parts: list[str] = []
        ctx = _ctx(W, H, FPS, 1.0, 0)
        cur = "0:v"
        for i, (a, b) in enumerate(zip(bounds, bounds[1:],
                                       strict=False)):
            mid = (a + b) / 2.0
            params = dict(self.BASE)
            for tr in tracks:
                params[tr.param] = round(tr.sample(mid), 4)
            cur = _chain_effect(
                parts, cur,
                Effect(name="color_grade", params=params, at=(a, b)),
                ctx, f"w{i}")
        oracle = workdir / "oracle.mp4"
        run_ffmpeg(["-i", src, "-filter_complex", ";".join(parts),
                    "-map", f"[{cur}]", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(oracle)])
        for t in (0.25, 0.75):
            a_stats = frame_stats(via_engine, t, workdir=workdir)
            b_stats = frame_stats(str(oracle), t, workdir=workdir)
            assert diff_stats(a_stats, b_stats, tol=8.0) == [], f"t={t}"

    def test_single_component_track(self, workdir):
        _slope, sat = self._tracks()
        out, res = self._render(workdir, [sat], "single.mp4")
        entries = [e for e in res["effects_applied"]
                   if e["kind"] == "keyframe-sampled"]
        assert len(entries) == 1
        assert entries[0]["source"] == "keyframe:cdl_saturation"
        early = _region(_frame(out, 0.05, workdir), (0, 0, W, H))
        late = _region(_frame(out, 0.95, workdir), (0, 0, W, H))
        assert _spread(early) > _spread(late) + 10

    def test_two_tracks_one_param_is_ambiguous(self, workdir):
        _slope, sat = self._tracks()
        dup = KeyframeTrack(
            clip_id="c0", effect="color_grade", param="cdl_saturation",
            keys=[Keyframe(0.0, 0.5), Keyframe(1.0, 0.9)])
        with pytest.raises(MediaEditError) as ei:
            self._render(workdir, [sat, dup], "dup.mp4")
        assert "ambiguous animation" in str(ei.value)


# ---------------------------------------------------------------------------
# mask tracking — the render follows the subject; held says so
# ---------------------------------------------------------------------------

def _moving_patch_video(workdir: Path, name: str = "moving.mp4") -> str:
    """Textured red patch drifting (3, 2) px/frame over a static
    blue-noise field, 24 frames at 30 fps, near-lossless encode."""
    from PIL import Image
    rng = np.random.default_rng(41)
    bg = np.stack([
        rng.integers(20, 90, size=(H, W)),
        rng.integers(20, 90, size=(H, W)),
        rng.integers(150, 256, size=(H, W))], axis=-1).astype(np.uint8)
    patch = np.stack([
        rng.integers(150, 256, size=(80, 110)),
        rng.integers(20, 90, size=(80, 110)),
        rng.integers(20, 90, size=(80, 110))], axis=-1).astype(np.uint8)
    seq = workdir / "patchseq"
    seq.mkdir()
    for i in range(24):
        f = bg.copy()
        x, y = 40 + 3 * i, 60 + 2 * i
        f[y:y + 80, x:x + 110] = patch
        Image.fromarray(f).save(str(seq / f"f{i:03d}.png"))
    out = workdir / name
    run_ffmpeg(["-framerate", "30", "-i", str(seq / "f%03d.png"),
                "-c:v", "libx264", "-crf", "6", "-pix_fmt", "yuv420p",
                "-y", str(out)])
    return str(out)


PATCH_BOX = {"kind": "box", "box": [40, 60, 150, 140], "feather": 0}
# sampling rects at the final frame: the patch's home then, and
# the ground it started on (both clear of the other rect)
FOLLOWED_RECT = (156, 130, 212, 178)
ORIGIN_RECT = (48, 68, 100, 132)


@NEEDS_FFMPEG
class TestMaskTrackingRender:
    def _render(self, workdir, *, track: bool, name: str):
        src = _moving_patch_video(workdir)
        params = {"grade": "noir", "mask": dict(PATCH_BOX)}
        if track:
            params["mask_track"] = True
        clip = Clip(path=src, id="m0",
                    effects=(Effect(name="color_grade", params=params),))
        tl = Timeline(video=Track(kind="video", clips=[clip]),
                      width=W, height=H, fps=FPS)
        out = workdir / name
        res = render_timeline(tl, out=out,
                              workdir=workdir / f"wd_{name}")
        return str(out), res

    def test_tracked_mask_follows_the_patch(self, workdir):
        out, res = self._render(workdir, track=True, name="trk.mp4")
        entry = res["effects_applied"][0]
        assert entry["mask"]["tracking"]["status"] == "tracked"
        assert "tracked" in entry["mask"]["anchor"]
        img = _frame(out, 0.76, workdir)
        # the grade sits on the patch's final home…
        assert _spread(_region(img, FOLLOWED_RECT)) < 30
        # …and NOT on the ground the patch started on
        assert _spread(_region(img, ORIGIN_RECT)) > 45

    def test_held_control_stays_on_the_anchor(self, workdir):
        out, res = self._render(workdir, track=False, name="held.mp4")
        entry = res["effects_applied"][0]
        assert "tracking" not in entry["mask"]
        img = _frame(out, 0.76, workdir)
        # mirror image of the tracked render: the grade stayed
        assert _spread(_region(img, FOLLOWED_RECT)) > 45
        assert _spread(_region(img, ORIGIN_RECT)) < 30

    def test_untextured_mask_holds_and_records_why(self, workdir):
        src = color_source(workdir / "flat.mp4", "0xC82828",
                           w=W, h=H, seconds=0.5)
        clip = Clip(path=src, id="f0", effects=(Effect(
            name="color_grade",
            params={"grade": "noir",
                    "mask": {"kind": "box",
                             "box": [10, 10, 150, 230], "feather": 0},
                    "mask_track": True}),))
        tl = Timeline(video=Track(kind="video", clips=[clip]),
                      width=W, height=H, fps=FPS)
        out = workdir / "flat_out.mp4"
        res = render_timeline(tl, out=out, workdir=workdir / "wd")
        entry = res["effects_applied"][0]
        tracking = entry["mask"]["tracking"]
        assert tracking["status"] == "held"
        assert "texture" in tracking["detail"]
        # the held mask still applies, at the anchor, as before
        img = _frame(str(out), 0.2, workdir)
        assert _spread(_region(img, (30, 30, 130, 210))) < 30

    def test_apply_effect_tracked_mask(self, workdir):
        src = _moving_patch_video(workdir)
        out = workdir / "single.mp4"
        res = apply_effect(src, "color_grade", out=str(out),
                           grade="noir", mask=dict(PATCH_BOX),
                           mask_track=True)
        entry = res["effects_applied"][0]
        assert entry["mask"]["tracking"]["status"] == "tracked"
        img = _frame(str(out), 0.76, workdir)
        assert _spread(_region(img, FOLLOWED_RECT)) < 30


# ---------------------------------------------------------------------------
# regional velocity_ramp — the refusal, proven
# ---------------------------------------------------------------------------

def _banded_video(workdir: Path) -> str:
    """Four half-second color bands: red, green, blue, yellow."""
    from PIL import Image
    bands = [(220, 30, 30), (30, 200, 30), (30, 30, 220),
             (220, 200, 30)]
    seq = workdir / "bandseq"
    seq.mkdir()
    for i in range(60):
        img = Image.new("RGB", (W, H), bands[min(i // 15, 3)])
        img.save(str(seq / f"f{i:03d}.png"))
    out = workdir / "bands.mp4"
    run_ffmpeg(["-framerate", "30", "-i", str(seq / "f%03d.png"),
                "-c:v", "libx264", "-crf", "6", "-pix_fmt", "yuv420p",
                "-y", str(out)])
    return str(out)


@NEEDS_FFMPEG
class TestRegionalVelocityBlocker:
    def test_velocity_rewrites_the_timeline(self, workdir):
        src = _banded_video(workdir)
        out = workdir / "ramped.mp4"
        apply_effect(src, "velocity_ramp", out=str(out),
                     speed=2.0, at=(0.0, 1.0))
        dur = float(video_probe(str(out))["duration"])
        # a one-second 2x window inside a 2 s clip: the stream
        # itself is 1.5 s — velocity does not tint frames, it
        # re-times the whole stream
        assert 1.35 < dur < 1.65

    def test_ramped_branch_carries_a_different_instant(self, workdir):
        src = _banded_video(workdir)
        out = workdir / "ramped2.mp4"
        apply_effect(src, "velocity_ramp", out=str(out),
                     speed=2.0, at=(0.0, 1.0))
        # at output t=1.25 the BASE timeline shows source 1.25
        # (blue band); the velocity branch shows source 1.75
        # (yellow band) — after the window, branch frame k is
        # simply not base frame k, which is exactly what the
        # shared-timeline mask composite cannot pair.
        base = _region(_frame(src, 1.25, workdir), (0, 0, W, H))
        branch = _region(_frame(str(out), 1.25, workdir), (0, 0, W, H))
        assert base[2] > 150 and base[0] < 90          # blue
        assert branch[0] > 150 and branch[2] < 90      # yellow

    def test_masked_velocity_still_refuses_by_name(self, workdir):
        src = _banded_video(workdir)
        clip = Clip(path=src, id="v0", effects=(Effect(
            name="velocity_ramp",
            params={"speed": 2.0, "mask": dict(PATCH_BOX)}),))
        tl = Timeline(video=Track(kind="video", clips=[clip]),
                      width=W, height=H, fps=FPS)
        with pytest.raises(MediaEditError) as ei:
            render_timeline(tl, out=workdir / "no.mp4",
                            workdir=workdir / "wd")
        assert "regional velocity ramp" in str(ei.value)
