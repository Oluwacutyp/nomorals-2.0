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
3. **Regional velocity_ramp** — the refusal was *proven* first
   (the velocity rung rewrites the stream timeline, measured, so
   a retimed branch can never pair with the shared-timeline mask
   composite), then the missing rung was built: the region
   time-remap resampler (`_chain_velocity_resampled`) stamps
   every source frame with the output instant that shows it, so
   the composite framesync resamples the retimed branch back
   onto the base timeline — the region follows the ramp integral
   inside the window, absorbs the offset by hold (fast) or cut
   (slow) at the window edge, and is back on the base clock
   after it. The landing is measured below, not asserted.
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
# regional velocity_ramp — why the rung had to exist (the proof),
# then the resampler landing, measured
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

    def test_masked_velocity_bad_speed_refuses_by_name(self, workdir):
        # the resampler shares the structural chain's ramp math,
        # refusals included: a non-positive speed is still named
        src = _banded_video(workdir)
        clip = Clip(path=src, id="v0", effects=(Effect(
            name="velocity_ramp",
            params={"speed": 0.0, "mask": dict(PATCH_BOX)}),))
        tl = Timeline(video=Track(kind="video", clips=[clip]),
                      width=W, height=H, fps=FPS)
        with pytest.raises(MediaEditError) as ei:
            render_timeline(tl, out=workdir / "no.mp4",
                            workdir=workdir / "wd")
        assert "speed must be > 0" in str(ei.value)


# ---------------------------------------------------------------------------
# regional velocity_ramp — the resampler, measured
# ---------------------------------------------------------------------------

#: ten 0.2 s bands (6 frames each — frame-aligned at 30 fps),
#: far apart in RGB so a decoded frame classifies by nearest
#: centroid even after two lossy encodes.
BANDS10 = [(220, 30, 30), (30, 200, 30), (30, 30, 220),
           (220, 200, 30), (220, 30, 220), (30, 220, 220),
           (240, 240, 240), (240, 140, 30), (140, 30, 240),
           (240, 140, 160)]
LEFT_HALF_MASK = {"kind": "box", "box": [0, 0, W // 2, H],
                  "feather": 0}
IN_RECT = (40, 100, 120, 140)     # inside the mask (left half)
OUT_RECT = (220, 100, 300, 140)   # outside it


def _band_index(rgb) -> int:
    return min(range(len(BANDS10)),
               key=lambda i: sum((a - b) ** 2 for a, b in
                                  zip(rgb, BANDS10[i], strict=True)))


def _banded10_video(workdir: Path, name: str = "bands10.mp4",
                    *, audio: bool = False) -> str:
    """The 10-band clock: source time is readable from any frame.

    With ``audio``, band k also carries a sine tone at
    300 + 100k Hz, so the audio timeline is readable the same
    way — the (d) oracle: picture retimes regionally, sound must
    stay on the base clock.
    """
    from PIL import Image
    seq = workdir / f"seq_{name}"
    seq.mkdir()
    for i in range(60):
        img = Image.new("RGB", (W, H), BANDS10[min(i // 6, 9)])
        img.save(str(seq / f"f{i:03d}.png"))
    out = workdir / name
    args = ["-framerate", "30", "-i", str(seq / "f%03d.png")]
    if audio:
        for k in range(10):
            args += ["-f", "lavfi", "-i",
                     f"sine=frequency={300 + 100 * k}"
                     f":sample_rate=48000:duration=0.2"]
        tones = "".join(f"[{k + 1}:a]" for k in range(10))
        args += ["-filter_complex",
                 f"{tones}concat=n=10:v=0:a=1[tones]",
                 "-map", "0:v", "-map", "[tones]",
                 "-c:a", "aac", "-shortest"]
    args += ["-c:v", "libx264", "-crf", "6", "-pix_fmt", "yuv420p",
             "-y", str(out)]
    run_ffmpeg(args)
    return str(out)


def _frame_count(video: str) -> int:
    import subprocess

    from nomorals.media_edit.videos import ffprobe_path
    out = subprocess.run(
        [ffprobe_path(), "-v", "error", "-count_frames",
         "-select_streams", "v:0", "-show_entries",
         "stream=nb_read_frames", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True).stdout
    return int(out.strip())


def _bands_at(video: str, t: float, workdir: Path) -> tuple[int, int]:
    img = _frame(video, t, workdir)
    return (_band_index(_region(img, IN_RECT)),
            _band_index(_region(img, OUT_RECT)))


def _render_banded(workdir: Path, fx: Effect, *, audio=False,
                   name="out.mp4", src_name="bands10.mp4"):
    src = _banded10_video(workdir, src_name, audio=audio)
    clip = Clip(path=src, id="v0", effects=(fx,))
    tl = Timeline(video=Track(kind="video", clips=[clip]),
                  width=W, height=H, fps=FPS)
    out = workdir / name
    res = render_timeline(tl, out=out, workdir=workdir / f"wd_{name}",
                          keep_clip_audio=audio)
    return src, str(out), res


@NEEDS_FFMPEG
class TestRegionalVelocityResampled:
    def test_fast_ramp_follows_integral_holds_and_rejoins(
            self, workdir):
        fx = Effect(name="velocity_ramp",
                    params={"speed": 2.0, "mask": dict(LEFT_HALF_MASK)},
                    at=(0.6, 1.4))
        src, out, res = _render_banded(workdir, fx)
        entry = [e for e in res["effects_applied"]
                 if e["effect"] == "velocity_ramp"][0]
        assert entry["kind"] == "structural-resampled"
        assert entry["mask"]["rung"] == "box+feather"
        # (a) before the window, output frame k pairs region and
        # base at the same source instant — both read band 1
        assert _bands_at(out, 0.3, workdir) == (1, 1)
        # (b) inside the window the region shows σ(t) = 0.6+2(t−0.6):
        # at t=0.8, σ=1.0 → band 5, while the base reads band 4
        assert _bands_at(out, 0.8, workdir) == (5, 4)
        # the hold: σ reached 1.4 at t*=1.0; at t=1.1 the region
        # still shows the window's last frame (band 6) while the
        # base has only reached band 5
        assert _bands_at(out, 1.1, workdir) == (6, 5)
        # (a) after the window the region is back on the base
        # clock — the desync the blocker proved (the structural
        # branch reads source 2.1 here, i.e. past the last band)
        # is gone: both read band 8
        assert _bands_at(out, 1.7, workdir) == (8, 8)
        # (c) the composite is exactly the base's length
        assert _frame_count(out) == _frame_count(src) == 60
        assert abs(float(video_probe(out)["duration"])
                   - float(video_probe(src)["duration"])) < 0.001

    def test_slow_ramp_cuts_and_rejoins(self, workdir):
        fx = Effect(name="velocity_ramp",
                    params={"speed": 0.5, "mask": dict(LEFT_HALF_MASK)},
                    at=(0.0, 1.0))
        _src, out, _res = _render_banded(workdir, fx,
                                         name="slow.mp4")
        # inside: σ(t) = 0.5t — at t=0.7 the region reads band 1
        # (source 0.35) while the base reads band 3
        assert _bands_at(out, 0.7, workdir) == (1, 3)
        # late in the window: σ=0.475 → band 2, base band 4
        assert _bands_at(out, 0.95, workdir) == (2, 4)
        # after the edge the unplayed tail (source 0.5–1.0) is
        # cut, inside the region only: both read band 6
        assert _bands_at(out, 1.3, workdir) == (6, 6)
        assert _frame_count(out) == 60

    def test_multiple_ramps_each_rejoin(self, workdir):
        fx = Effect(name="velocity_ramp",
                    params={"ramps": [(0.3, 0.9, 2.0),
                                      (1.2, 1.8, 0.5)],
                            "mask": dict(LEFT_HALF_MASK)})
        _src, out, _res = _render_banded(workdir, fx,
                                         name="multi.mp4")
        # fast window: σ(0.5) = 0.3+2·0.2 = 0.7 → band 3, base 2
        assert _bands_at(out, 0.5, workdir) == (3, 2)
        # its hold (t* = 0.6): at t=0.7 the region sits on the
        # window's last frame (band 4), the base is at band 3
        assert _bands_at(out, 0.7, workdir) == (4, 3)
        # between windows both clocks agree again (band 5)
        assert _bands_at(out, 1.0, workdir) == (5, 5)
        # slow window: σ(1.4) = 1.2+0.5·0.2 = 1.3 → band 6, base 7
        assert _bands_at(out, 1.4, workdir) == (6, 7)
        # and after its cut, rejoined (band 9 both)
        assert _bands_at(out, 1.9, workdir) == (9, 9)
        assert _frame_count(out) == 60

    def test_apply_effect_surface_records_the_rung(self, workdir):
        src = _banded10_video(workdir)
        out = workdir / "single.mp4"
        res = apply_effect(src, "velocity_ramp", out=str(out),
                           speed=2.0, at=(0.6, 1.4),
                           mask=dict(LEFT_HALF_MASK))
        entry = res["effects_applied"][0]
        assert entry["kind"] == "structural-resampled"
        assert _bands_at(str(out), 0.8, workdir) == (5, 4)
        assert _bands_at(str(out), 1.7, workdir) == (8, 8)
        assert _frame_count(str(out)) == 60

    def test_identity_speed_masked_is_a_plain_region(self, workdir):
        fx = Effect(name="velocity_ramp",
                    params={"speed": 1.0, "mask": dict(LEFT_HALF_MASK)},
                    at=(0.6, 1.4))
        _src, out, _res = _render_banded(workdir, fx,
                                         name="ident.mp4")
        assert _bands_at(out, 0.8, workdir) == (4, 4)
        assert _frame_count(out) == 60

    def test_audio_stays_on_the_base_timeline(self, workdir):
        # (d) the clip's sound is never routed through the video
        # resampler: every 0.2 s band of the output audio carries
        # the tone the BASE clock assigns to it — including the
        # bands inside the ramp window, where the picture region
        # is hearing source time σ(t) instead.
        fx = Effect(name="velocity_ramp",
                    params={"speed": 2.0, "mask": dict(LEFT_HALF_MASK)},
                    at=(0.6, 1.4))
        _src, out, _res = _render_banded(workdir, fx, audio=True,
                                         name="av.mp4",
                                         src_name="bands_av.mp4")
        assert _bands_at(out, 0.8, workdir) == (5, 4)  # video retimed
        wav = workdir / "out.wav"
        run_ffmpeg(["-i", out, "-vn", "-ac", "1", "-ar", "48000",
                    "-c:a", "pcm_s16le", "-y", str(wav)])
        import wave
        with wave.open(str(wav)) as w:
            raw = np.frombuffer(w.readframes(w.getnframes()),
                                dtype=np.int16).astype(np.float64)
        assert len(raw) >= 2 * 48000 - 4800  # ~the full 2 s
        for k in range(10):
            seg = raw[int((0.2 * k + 0.05) * 48000):
                      int((0.2 * k + 0.15) * 48000)]
            spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
            freq = float(np.argmax(spec)) * 48000 / len(seg)
            assert abs(freq - (300 + 100 * k)) <= 30, (k, freq)
