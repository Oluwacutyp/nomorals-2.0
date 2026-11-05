"""Echelon Wave 5 — the effects port (plan §1.2b/§1.2c property
model, §1.1c(d) grades as Document state, §1.1d mask carrier).

Proves: the property registry is total and enforced at the chain
funnel (undeclared params, kind/range/enum violations raise naming
the property path); keyframes bind through the registry — the new
``sampled`` family renders frame-exact windows and matches a
caller-built windowed assembly (two paths, one result); regional
effects apply inside the MaskRef mask and not outside it, with the
mask rung recorded; a Document's look_lock grades the render
through the canonical chain with applied/carried honestly split;
every effect application lands in ``effects_applied`` with its
producing rung; deferred effects refuse by name.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from nomorals.media.edit_engine import (
    DEFERRED_EFFECTS,
    EFFECTS,
    EFFECT_PROPERTIES,
    Clip,
    DocTrack,
    Document,
    Effect,
    Keyframe,
    KeyframeTrack,
    OverlayTrack,
    Timeline,
    Track,
)
from nomorals.media.edit_engine.effects import (
    _ctx,
    _effect_filter,
    effect_properties,
    effect_property_path,
    keyframable_family,
    keyframable_params,
    validate_effect_params,
)
from nomorals.media.edit_engine.keyframes import classify_track, discretize
from nomorals.media.edit_engine.render import (
    _chain_effect,
    apply_effect,
    render_document,
    render_timeline,
)
from nomorals.media_edit.videos import MediaEditError, run_ffmpeg

from tests.edit_engine_golden import (
    color_source,
    diff_stats,
    frame_stats,
    load_goldens,
    pattern_source,
)

NEEDS_FFMPEG = pytest.mark.skipif(
    os.system("ffmpeg -hide_banner -version > /dev/null 2>&1") != 0,
    reason="ffmpeg not installed",
)

W, H, FPS = 320, 180, 30


@pytest.fixture()
def workdir(tmp_path):
    d = tmp_path / "fx"
    d.mkdir()
    return d


def _twotone(workdir: Path, name: str = "twotone.png") -> str:
    """Left half red, right half blue — the regional oracle source."""
    from PIL import Image
    img = Image.new("RGB", (W, H))
    px = img.load()
    for x in range(W):
        for y in range(H):
            px[x, y] = (200, 30, 30) if x < W // 2 else (30, 30, 200)
    p = workdir / name
    img.save(p)
    return str(p)


def _region_means(video: str, t: float, workdir: Path) -> dict:
    """Mean RGB of the left/right thirds of one sampled frame."""
    from PIL import Image
    png = Path(workdir) / f"region_{t:.2f}.png"
    run_ffmpeg(["-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
                "-y", str(png)])
    img = Image.open(png).convert("RGB")
    w, h = img.size

    def mean(box):
        crop = img.crop(box)
        data = list(crop.getdata())
        n = len(data)
        return tuple(sum(c[i] for c in data) / n for i in range(3))

    return {"left": mean((10, 10, w // 2 - 10, h - 10)),
            "right": mean((w // 2 + 10, 10, w - 10, h - 10))}


def _spread(rgb) -> float:
    return max(rgb) - min(rgb)


LEFT_BOX_MASK = {"kind": "box", "box": [0, 0, W // 2, H], "feather": 0}


# ---------------------------------------------------------------------------
# the property registry
# ---------------------------------------------------------------------------

class TestPropertyRegistry:
    def test_registry_is_total_over_builders(self):
        assert set(EFFECT_PROPERTIES) == set(EFFECTS) | {"velocity_ramp"}

    def test_every_declared_default_validates(self):
        for name, params in EFFECT_PROPERTIES.items():
            validate_effect_params(name, {})
            for p, spec in params.items():
                if spec.default is not None:
                    validate_effect_params(name, {p: spec.default})

    def test_undeclared_param_raises_naming_the_path(self):
        with pytest.raises(MediaEditError) as ei:
            validate_effect_params("blur", {"radus": 5.0})
        assert effect_property_path("blur", "radus") in str(ei.value)

    def test_range_and_kind_and_enum_violations_raise(self):
        with pytest.raises(MediaEditError):
            validate_effect_params("blur", {"radius": 500.0})
        with pytest.raises(MediaEditError):
            validate_effect_params("blur", {"radius": "wide"})
        with pytest.raises(MediaEditError):
            validate_effect_params("ken_burns",
                                   {"direction": "sideways"})
        with pytest.raises(MediaEditError):
            validate_effect_params("opacity", {"alpha": None})
        with pytest.raises(MediaEditError):
            validate_effect_params("rgb_split", {"animate": "yes"})

    def test_reserved_and_internal_keys_pass(self):
        validate_effect_params(
            "blur", {"radius": 4.0, "mask": dict(LEFT_BOX_MASK),
                     "_mask_png": "/x.png", "_source": "look_lock"})

    def test_registry_as_data_is_json_safe(self):
        data = effect_properties()
        assert json.loads(json.dumps(data)) == data
        blur = data["effects"]["blur"]["params"]["radius"]
        assert blur["path"] == "effects/blur/radius"
        assert blur["keyframable"] == "sampled"
        assert data["deferred"]["camera3d"]

    def test_keyframable_derivation_keeps_wave12_declarations(self):
        kf = keyframable_params()
        for name, param in (("zoom", "z"), ("ken_burns", "zoom"),
                            ("punch_zoom", "zmax"), ("opacity", "alpha"),
                            ("position", "x"), ("position", "y")):
            assert param in kf[name]
        # the Wave-5 sampled declarations
        for name, param in (("blur", "radius"), ("grain", "strength"),
                            ("rgb_split", "shift"), ("shake", "amplitude"),
                            ("flash", "level"), ("ken_burns", "pan_x"),
                            ("beat_pulse", "strength")):
            assert param in kf[name]
            assert keyframable_family(name, param) == "sampled"
        # named grades and the cdl dict are cuts, not curves —
        # but the CDL's scalar components are keyframable paths
        # (the §1.2b remainder, closed by the registry extension):
        # exactly the ten cdl_* components, all sampled family.
        cg = kf["color_grade"]
        assert "grade" not in cg and "cdl" not in cg
        assert set(cg) == {
            "cdl_slope_r", "cdl_slope_g", "cdl_slope_b",
            "cdl_offset_r", "cdl_offset_g", "cdl_offset_b",
            "cdl_power_r", "cdl_power_g", "cdl_power_b",
            "cdl_saturation"}
        for param in cg:
            assert keyframable_family("color_grade", param) == "sampled"


# ---------------------------------------------------------------------------
# keyframe binding through the registry
# ---------------------------------------------------------------------------

class TestKeyframeBinding:
    def test_families(self):
        def fam(effect, param):
            t = KeyframeTrack(clip_id="c", effect=effect, param=param,
                              keys=[Keyframe(0.0, 1.0)])
            return classify_track(t)
        assert fam("zoom", "z") == "expression"
        assert fam("punch_zoom", "zmax") == "expression"
        assert fam("opacity", "alpha") == "opacity"
        assert fam("position", "x") == "position"
        assert fam("blur", "radius") == "sampled"
        assert fam("grain", "strength") == "sampled"
        assert fam("ken_burns", "pan_y") == "sampled"

    def test_undeclared_target_raises_listing_declared(self):
        t = KeyframeTrack(clip_id="c", effect="color_grade",
                          param="grade", keys=[Keyframe(0.0, 1.0)])
        with pytest.raises(ValueError) as ei:
            classify_track(t)
        assert "not keyframable" in str(ei.value)
        assert "blur.radius" in str(ei.value)

    def test_frame_exact_windows(self):
        track = KeyframeTrack(
            clip_id="c", effect="blur", param="radius",
            keys=[Keyframe(0.0, 0.0, "linear"),
                  Keyframe(10 / 30, 6.0, "linear"),
                  Keyframe(1.0, 12.0, "linear")])
        wins = discretize(track, fps=FPS)
        assert len(wins) > 4
        for a, b, _v in wins:
            assert abs(a * FPS - round(a * FPS)) < 1e-9
            assert abs(b * FPS - round(b * FPS)) < 1e-9
        assert wins[0][0] == 0.0 and abs(wins[-1][1] - 1.0) < 1e-9
        # each window's value is the curve at its middle frame
        for a, b, v in wins:
            mid = (a + b) / 2.0
            assert v == pytest.approx(track.sample(mid), abs=1e-9)


# ---------------------------------------------------------------------------
# validation + deferral at the chain funnel
# ---------------------------------------------------------------------------

class TestFunnel:
    def test_registry_enforced_in_effect_filter(self):
        ctx = _ctx(W, H, FPS, 1.0, 0)
        with pytest.raises(MediaEditError):
            _effect_filter("blur", {"radius": "wide"}, ctx)
        with pytest.raises(MediaEditError):
            _effect_filter("blur", {"radus": 3.0}, ctx)
        assert "boxblur" in _effect_filter("blur", {"radius": 3.0}, ctx)

    def test_deferred_effect_refuses_by_name(self):
        ctx = _ctx(W, H, FPS, 1.0, 0)
        with pytest.raises(MediaEditError) as ei:
            _effect_filter("camera3d", {}, ctx)
        assert "deferred" in str(ei.value)
        assert "§1.5e" in str(ei.value)

    def test_deferred_keyframe_target_refuses_by_name(self):
        t = KeyframeTrack(clip_id="c", effect="camera3d", param="x",
                          keys=[Keyframe(0.0, 1.0)])
        with pytest.raises(ValueError) as ei:
            classify_track(t)
        assert "deferred" in str(ei.value)

    def test_apply_effect_deferred_and_unknown(self, workdir):
        src = workdir / "dummy.mp4"
        src.write_bytes(b"x")
        with pytest.raises(MediaEditError) as ei:
            apply_effect(src, "camera3d")
        assert "deferred" in str(ei.value)
        with pytest.raises(MediaEditError) as ei2:
            apply_effect(src, "no_such_effect")
        assert "unknown effect" in str(ei2.value)
        assert "camera3d" in DEFERRED_EFFECTS


# ---------------------------------------------------------------------------
# sampled keyframes render
# ---------------------------------------------------------------------------

@NEEDS_FFMPEG
class TestSampledRender:
    def test_two_paths_one_result(self, workdir):
        """The renderer's sampled binding vs a caller-built windowed
        assembly through the same chain builder (the 1.2c oracle
        habit): sampled frames must agree."""
        src = pattern_source(workdir / "pat.mp4", w=W, h=H, seconds=1.0)
        track = KeyframeTrack(
            clip_id="s0", effect="blur", param="radius",
            keys=[Keyframe(0.0, 0.5, "linear"),
                  Keyframe(1.0, 12.0, "linear")])
        tl = Timeline(
            video=Track(kind="video", clips=[Clip(path=src, id="s0")]),
            keyframes=[track], width=W, height=H, fps=FPS)
        via_engine = workdir / "engine.mp4"
        res = render_timeline(tl, out=via_engine,
                              workdir=workdir / "wd_engine")
        assert res["keyframes_applied"][0]["family"] == "sampled"
        entry = res["effects_applied"][0]
        assert entry["kind"] == "keyframe-sampled"
        assert entry["source"] == "keyframe:radius"
        assert entry["rung"] == "edit_engine.effects._fx_blur"
        assert entry["windows"] >= 4

        # the standalone path: windowed static sub-effects, chained
        # by hand through _chain_effect (what apply_effect drives)
        parts: list[str] = []
        ctx = _ctx(W, H, FPS, 1.0, 0)
        cur = "0:v"
        for i, (a, b, v) in enumerate(discretize(track, fps=FPS)):
            cur = _chain_effect(
                parts, cur,
                Effect(name="blur", params={"radius": round(v, 4)},
                       at=(a, b)),
                ctx, f"w{i}")
        oracle = workdir / "oracle.mp4"
        run_ffmpeg(["-i", src, "-filter_complex", ";".join(parts),
                    "-map", f"[{cur}]", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(oracle)])
        for t in (0.25, 0.75):
            a_stats = frame_stats(via_engine, t, workdir=workdir)
            b_stats = frame_stats(oracle, t, workdir=workdir)
            assert diff_stats(a_stats, b_stats, tol=8.0) == [], f"t={t}"

    def test_sampled_consumes_the_static_base(self, workdir):
        src = color_source(workdir / "c.mp4", "red", w=W, h=H)
        clip = Clip(path=src, id="c0",
                    effects=(Effect(name="blur",
                                    params={"radius": 3.0}),))
        track = KeyframeTrack(
            clip_id="c0", effect="blur", param="radius",
            keys=[Keyframe(0.0, 2.0, "linear"),
                  Keyframe(1.0, 9.0, "linear")])
        tl = Timeline(video=Track(kind="video", clips=[clip]),
                      keyframes=[track], width=W, height=H, fps=FPS)
        res = render_timeline(tl, out=workdir / "out.mp4",
                              workdir=workdir / "wd")
        blur_entries = [e for e in res["effects_applied"]
                        if e["effect"] == "blur"]
        assert len(blur_entries) == 1
        assert blur_entries[0]["kind"] == "keyframe-sampled"

    def test_two_statics_are_an_ambiguity_error(self, workdir):
        src = color_source(workdir / "c.mp4", "red", w=W, h=H)
        clip = Clip(path=src, id="c0",
                    effects=(Effect(name="blur", params={"radius": 2.0}),
                             Effect(name="blur", params={"radius": 4.0})))
        track = KeyframeTrack(
            clip_id="c0", effect="blur", param="radius",
            keys=[Keyframe(0.0, 2.0), Keyframe(1.0, 9.0)])
        tl = Timeline(video=Track(kind="video", clips=[clip]),
                      keyframes=[track], width=W, height=H, fps=FPS)
        with pytest.raises(MediaEditError) as ei:
            render_timeline(tl, out=workdir / "out.mp4",
                            workdir=workdir / "wd")
        assert "ambiguous base" in str(ei.value)

    def test_sampled_on_overlay_track(self, workdir):
        base = color_source(workdir / "base.mp4", "black", w=W, h=H)
        over = pattern_source(workdir / "over.mp4", w=W, h=H,
                              seconds=1.0)
        track = KeyframeTrack(
            clip_id="o0", effect="grain", param="strength",
            keys=[Keyframe(0.0, 2.0, "linear"),
                  Keyframe(1.0, 30.0, "linear")])
        tl = Timeline(
            video=Track(kind="video", clips=[Clip(path=base, id="b0")]),
            overlays=[OverlayTrack(
                kind="video", label="fxlayer",
                clips=[Clip(path=over, id="o0", at=0.0, duration=1.0)])],
            keyframes=[track], width=W, height=H, fps=FPS)
        res = render_timeline(tl, out=workdir / "out.mp4",
                              workdir=workdir / "wd")
        grain = [e for e in res["effects_applied"]
                 if e["effect"] == "grain"]
        assert grain and grain[0]["kind"] == "keyframe-sampled"
        assert grain[0]["track"] == "fxlayer"


# ---------------------------------------------------------------------------
# regional (masked) effects
# ---------------------------------------------------------------------------

@NEEDS_FFMPEG
class TestRegionalEffects:
    def _doc(self, effects, *, kind="image"):
        return Document(
            tracks=[DocTrack(kind=kind, clips=[Clip(
                path=self._src, duration=1.0, effects=effects)])],
            output={"width": W, "height": H, "fps": FPS,
                    "profile": "pc"},
            title="regional")

    @pytest.fixture(autouse=True)
    def _setup(self, workdir):
        self._src = _twotone(workdir)
        self._wd = workdir

    def test_masked_grade_applies_inside_only(self):
        fx = (Effect(name="color_grade",
                     params={"grade": "noir", "mask": LEFT_BOX_MASK}),)
        out = self._wd / "masked.mp4"
        res = render_document(self._doc(fx), out=out,
                              workdir=self._wd / "wd_m")
        regions = _region_means(out, 0.5, self._wd)
        # inside the mask: noir — saturation gone
        assert _spread(regions["left"]) < 25
        # outside: the blue survives untouched
        assert regions["right"][2] - regions["right"][0] > 80
        entry = res["effects_applied"][0]
        assert entry["mask"]["rung"] == "box+feather"
        assert entry["effect"] == "color_grade"

        # the control: no mask → the grade covers the right side too
        plain = (Effect(name="color_grade", params={"grade": "noir"}),)
        out2 = self._wd / "plain.mp4"
        render_document(self._doc(plain), out=out2,
                        workdir=self._wd / "wd_p")
        regions2 = _region_means(out2, 0.5, self._wd)
        assert _spread(regions2["right"]) < 25

    def test_file_mask_rung_is_recorded(self):
        from PIL import Image
        m = Image.new("L", (W, H), 0)
        m.paste(255, (0, 0, W // 2, H))
        mpath = self._wd / "mask.png"
        m.save(mpath)
        fx = (Effect(name="color_grade", params={
            "grade": "noir",
            "mask": {"kind": "file", "path": str(mpath)}}),)
        res = render_document(self._doc(fx), out=self._wd / "f.mp4",
                              workdir=self._wd / "wd_f")
        assert res["effects_applied"][0]["mask"]["rung"] == "file"

    def test_masked_effect_on_video_clip(self):
        # a real video input (the anchor is a grabbed frame)
        vid = self._wd / "twotone.mp4"
        run_ffmpeg(["-loop", "1", "-framerate", str(FPS), "-t", "1",
                    "-i", self._src, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(vid)])
        fx = (Effect(name="color_grade",
                     params={"grade": "noir", "mask": LEFT_BOX_MASK}),)
        doc = Document(
            tracks=[DocTrack(kind="video", clips=[Clip(
                path=str(vid), duration=1.0, effects=fx)])],
            output={"width": W, "height": H, "fps": FPS,
                    "profile": "pc"})
        out = self._wd / "vid.mp4"
        render_document(doc, out=out, workdir=self._wd / "wd_v")
        regions = _region_means(out, 0.5, self._wd)
        assert _spread(regions["left"]) < 25
        assert regions["right"][2] - regions["right"][0] > 80

    def test_windowed_masked_effect(self):
        fx = (Effect(name="color_grade",
                     params={"grade": "noir", "mask": LEFT_BOX_MASK},
                     at=(0.25, 0.75)),)
        res = render_document(self._doc(fx), out=self._wd / "w.mp4",
                              workdir=self._wd / "wd_w")
        entry = res["effects_applied"][0]
        assert entry["window"] == [0.25, 0.75]
        assert entry["mask"]["rung"] == "box+feather"

    def test_masked_effect_on_overlay_track(self):
        # the overlay renderer's own mask wiring (inputs after the
        # canvas + clip inputs) — a masked grade on an overlay clip
        base = color_source(self._wd / "base.mp4", "black", w=W, h=H)
        vid = self._wd / "ov.mp4"
        run_ffmpeg(["-loop", "1", "-framerate", str(FPS), "-t", "1",
                    "-i", self._src, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(vid)])
        fx = (Effect(name="color_grade",
                     params={"grade": "noir", "mask": LEFT_BOX_MASK}),)
        tl = Timeline(
            video=Track(kind="video", clips=[Clip(path=base, id="b0")]),
            overlays=[OverlayTrack(
                kind="video", label="wash",
                clips=[Clip(path=str(vid), id="o0", at=0.0,
                            duration=1.0, effects=fx)])],
            width=W, height=H, fps=FPS)
        out = self._wd / "ov_out.mp4"
        res = render_timeline(tl, out=out, workdir=self._wd / "wd_ov")
        regions = _region_means(out, 0.5, self._wd)
        assert _spread(regions["left"]) < 25
        assert regions["right"][2] - regions["right"][0] > 80
        entry = [e for e in res["effects_applied"]
                 if e["effect"] == "color_grade"][0]
        assert entry["mask"]["rung"] == "box+feather"
        assert entry["track"] == "wash"

    def test_unresolvable_mask_raises_before_render(self):
        fx = (Effect(name="blur", params={
            "radius": 4.0,
            "mask": {"kind": "auto", "text": "the red half"}}),)
        with pytest.raises(MediaEditError) as ei:
            render_document(self._doc(fx), out=self._wd / "x.mp4",
                            workdir=self._wd / "wd_x")
        assert "mask did not resolve" in str(ei.value)

    def test_masked_velocity_ramp_refuses(self):
        fx = (Effect(name="velocity_ramp",
                     params={"speed": 2.0, "mask": LEFT_BOX_MASK}),)
        with pytest.raises(MediaEditError) as ei:
            render_document(self._doc(fx), out=self._wd / "v.mp4",
                            workdir=self._wd / "wd_vr")
        assert "structural" in str(ei.value)

    def test_mask_survives_document_round_trip(self):
        doc = self._doc((Effect(name="blur", params={
            "radius": 4.0, "mask": LEFT_BOX_MASK}),))
        back = Document.from_dict(doc.to_dict())
        params = back.tracks[0].clips[0].effects[0].params
        assert params["mask"] == LEFT_BOX_MASK


# ---------------------------------------------------------------------------
# look_lock — grades as Document state
# ---------------------------------------------------------------------------

@NEEDS_FFMPEG
class TestLookLock:
    def _doc(self, src, look_lock=None):
        return Document(
            tracks=[DocTrack(kind="video",
                             clips=[Clip(path=src, id="v0")])],
            look_lock=look_lock,
            output={"width": W, "height": H, "fps": FPS,
                    "profile": "pc"})

    def test_grade_look_applies_and_reports(self, workdir):
        src = color_source(workdir / "red.mp4", "red", w=W, h=H)
        lock = {"name": "house", "grade": "noir", "grain": 0.0,
                "palette": [[1.0, 0.0, 0.0]], "lens": "35mm"}
        out = workdir / "look.mp4"
        res = render_document(self._doc(src, lock), out=out,
                              workdir=workdir / "wd")
        stats = frame_stats(out, 0.5, workdir=workdir)
        assert _spread(stats["mean"]) < 30  # red, desaturated
        manifest = res["document"]["look_lock"]
        assert manifest["applied"] == ["color_grade(noir)"]
        assert "palette" in manifest["carried"]
        assert "lens" in manifest["carried"]
        stages = [r["stage"] for r in res["rungs"]]
        assert "look_lock" in stages
        look_entries = [e for e in res["effects_applied"]
                        if e["source"] == "look_lock"]
        assert look_entries and \
            look_entries[0]["effect"] == "color_grade"

        control = workdir / "control.mp4"
        render_document(self._doc(src), out=control,
                        workdir=workdir / "wd_c")
        cstats = frame_stats(control, 0.5, workdir=workdir)
        assert cstats["mean"][0] - cstats["mean"][2] > 100

    def test_cdl_look_consumes_the_canonical_chain(self, workdir):
        src = color_source(workdir / "gray.mp4", "gray", w=W, h=H)
        lock = {"cdl": {"slope": [1.6, 1.0, 1.0],
                        "offset": [0.0, 0.0, 0.0],
                        "power": [1.0, 1.0, 1.0], "saturation": 1.0}}
        out = workdir / "cdl.mp4"
        res = render_document(self._doc(src, lock), out=out,
                              workdir=workdir / "wd")
        assert res["document"]["look_lock"]["applied"] == \
            ["color_grade(cdl)"]
        stats = frame_stats(out, 0.5, workdir=workdir)
        control = workdir / "cdl_control.mp4"
        render_document(self._doc(src), out=control,
                        workdir=workdir / "wd_c")
        cstats = frame_stats(control, 0.5, workdir=workdir)
        assert stats["mean"][0] > cstats["mean"][0] + 10

    def test_carried_only_look_renders_unchanged_and_says_so(
            self, workdir):
        src = color_source(workdir / "red.mp4", "red", w=W, h=H)
        lock = {"name": "meta only", "palette": [[0.0, 0.0, 1.0]]}
        out = workdir / "meta.mp4"
        res = render_document(self._doc(src, lock), out=out,
                              workdir=workdir / "wd")
        assert res["document"]["look_lock"]["applied"] == []
        note = [r for r in res["rungs"] if r["stage"] == "look_lock"]
        assert note and "applied: none" in note[0]["note"]
        stats = frame_stats(out, 0.5, workdir=workdir)
        assert stats["mean"][0] - stats["mean"][2] > 100


# ---------------------------------------------------------------------------
# the honesty record
# ---------------------------------------------------------------------------

@NEEDS_FFMPEG
class TestEffectsRecord:
    def test_static_effects_record_names_rungs(self, workdir):
        src = color_source(workdir / "c.mp4", "red", w=W, h=H)
        clip = Clip(path=src, id="c0", effects=(
            Effect(name="blur", params={"radius": 3.0}),
            Effect(name="grain", params={"strength": 12.0})))
        tl = Timeline(video=Track(kind="video", clips=[clip]),
                      width=W, height=H, fps=FPS)
        res = render_timeline(tl, out=workdir / "out.mp4",
                              workdir=workdir / "wd")
        by_name = {e["effect"]: e for e in res["effects_applied"]}
        assert by_name["blur"]["rung"] == "edit_engine.effects._fx_blur"
        assert by_name["grain"]["rung"] == \
            "edit_engine.effects._fx_grain"
        assert by_name["blur"]["source"] == "clip"
        assert by_name["blur"]["kind"] == "builder"
        stages = [r["stage"] for r in res["rungs"]]
        assert "effects" in stages

    def test_velocity_record_is_structural(self, workdir):
        src = color_source(workdir / "c.mp4", "red", w=W, h=H,
                           seconds=2.0)
        clip = Clip(path=src, id="c0", effects=(
            Effect(name="velocity_ramp", params={"speed": 2.0}),))
        tl = Timeline(video=Track(kind="video", clips=[clip]),
                      width=W, height=H, fps=FPS)
        res = render_timeline(tl, out=workdir / "out.mp4",
                              workdir=workdir / "wd")
        entries = [e for e in res["effects_applied"]
                   if e["effect"] == "velocity_ramp"]
        assert entries and entries[0]["kind"] == "structural"
        assert entries[0]["rung"] == \
            "edit_engine.render._chain_velocity"
        assert Path(res["output"]).exists()


# ---------------------------------------------------------------------------
# golden + determinism for the effects fixture
# ---------------------------------------------------------------------------

def build_masked_grade(workdir: Path) -> Path:
    """The Wave-5 golden fixture: two-tone still, noir in the left
    box mask only."""
    workdir.mkdir(parents=True, exist_ok=True)
    src = _twotone(workdir)
    doc = Document(
        tracks=[DocTrack(kind="image", clips=[Clip(
            path=src, duration=1.0,
            effects=(Effect(name="color_grade", params={
                "grade": "noir", "mask": LEFT_BOX_MASK}),))])],
        output={"width": W, "height": H, "fps": FPS, "profile": "pc"},
        title="masked grade golden")
    out = workdir / "masked_grade.mp4"
    render_document(doc, out=out, workdir=workdir / "render")
    return out


@NEEDS_FFMPEG
class TestGolden:
    def test_masked_grade_matches_golden(self, workdir):
        out = build_masked_grade(workdir)
        stats = frame_stats(out, 0.5, workdir=workdir)
        golden = load_goldens()["masked_grade"]["stats"]
        assert diff_stats(stats, golden) == []

    def test_masked_grade_determinism(self, workdir):
        a = build_masked_grade(workdir / "run_a")
        b = build_masked_grade(workdir / "run_b")
        sa = frame_stats(a, 0.5, workdir=workdir)
        sb = frame_stats(b, 0.5, workdir=workdir)
        assert diff_stats(sa, sb, tol=0.5) == []
        ha = hashlib.sha256(Path(a).read_bytes()).hexdigest()
        hb = hashlib.sha256(Path(b).read_bytes()).hexdigest()
        assert ha == hb


# ---------------------------------------------------------------------------
# surfaces: the tool + the parity record
# ---------------------------------------------------------------------------

class TestSurfaces:
    def test_studio_tool_serves_the_registry(self):
        from nomorals.tools.registry import ToolRegistry

        class FakeContext:
            workspace = "/tmp/studio_fx_ws"

        reg = ToolRegistry(context=FakeContext()).register_builtins()
        out = reg.call("studio_effect_properties")
        assert out.ok and out.value["ok"] is True
        radius = out.value["effects"]["blur"]["params"]["radius"]
        assert radius["keyframable"] == "sampled"
        one = reg.call("studio_effect_properties", effect="blur")
        assert one.value["effect"] == "blur"
        deferred = reg.call("studio_effect_properties",
                            effect="camera3d")
        assert "deferred" in deferred.value
        bad = reg.call("studio_effect_properties", effect="nope")
        assert bad.value["ok"] is False

    def test_parity_record_carries_the_effects_rows(self):
        parity = (Path(__file__).parent.parent / "nomorals" / "media"
                  / "PARITY.md").read_text(encoding="utf-8")
        assert "## Effects — EffectCraft" in parity
        for needle in ("EFFECT_PROPERTIES", "sampled", "MaskRef",
                       "look_lock", "effects_applied"):
            assert needle in parity, needle
