"""Tests for the Devon Studio rebuild: automation, keyframes, color, tools."""

import os

import pytest

from nomorals.media.edit_engine.keyframes import (
    KeyframeTrack,
    fade_in,
    fade_out,
    list_easings,
    track_to_filter,
    zoom_ramp,
)


class TestKeyframes:
    def test_sample_before_first(self):
        t = KeyframeTrack(clip_id="c", effect="e", param="p")
        t.add(1.0, 10.0).add(3.0, 30.0)
        assert t.sample(0.0) == 10.0

    def test_sample_after_last(self):
        t = KeyframeTrack(clip_id="c", effect="e", param="p")
        t.add(1.0, 10.0).add(3.0, 30.0)
        assert t.sample(99.0) == 30.0

    def test_sample_midpoint_linear(self):
        t = KeyframeTrack(clip_id="c", effect="e", param="p")
        t.add(0.0, 0.0, easing="linear").add(2.0, 10.0, easing="linear")
        assert t.sample(1.0) == pytest.approx(5.0)

    def test_sample_midpoint_smooth(self):
        t = KeyframeTrack(clip_id="c", effect="e", param="p")
        t.add(0.0, 0.0, easing="smooth").add(2.0, 10.0, easing="smooth")
        # smoothstep(0.5) = 0.5
        assert t.sample(1.0) == pytest.approx(5.0)

    def test_smooth_eases_ends(self):
        t = KeyframeTrack(clip_id="c", effect="e", param="p")
        t.add(0.0, 0.0, easing="smooth").add(4.0, 8.0, easing="smooth")
        # smoothstep rises slowly at the start
        assert t.sample(1.0) < 2.0  # linear would be 2.0

    def test_empty_track_raises(self):
        t = KeyframeTrack(clip_id="c", effect="e", param="p")
        with pytest.raises(ValueError):
            t.sample(1.0)

    def test_bad_easing_raises(self):
        t = KeyframeTrack(clip_id="c", effect="e", param="p")
        with pytest.raises(ValueError):
            t.add(0.0, 1.0, easing="nope")

    def test_roundtrip(self):
        t = zoom_ramp("clip1", 0.0, 2.0)
        d = t.to_dict()
        t2 = KeyframeTrack.from_dict(d)
        assert t2.sample(1.0) == pytest.approx(t.sample(1.0))

    def test_fade_in(self):
        t = fade_in("c", 2.0)
        assert t.sample(0.0) == 0.0
        assert t.sample(2.0) == 1.0

    def test_fade_out(self):
        t = fade_out("c", 5.0, 1.0)
        assert t.sample(5.0) == 1.0
        assert t.sample(6.0) == 0.0

    def test_ffmpeg_expr_single_key(self):
        t = KeyframeTrack(clip_id="c", effect="e", param="p")
        t.add(0.0, 1.5)
        expr = track_to_filter(t, "zoompan=z='{v}'")
        assert "1.500000" in expr

    def test_ffmpeg_expr_multi_key(self):
        t = KeyframeTrack(clip_id="c", effect="e", param="p")
        t.add(0.0, 1.0, easing="linear").add(2.0, 2.0, easing="linear")
        expr = track_to_filter(t, "eq=brightness={v}")
        assert "if(lt(t," in expr

    def test_list_easings(self):
        easings = list_easings()
        assert "linear" in easings and "smooth" in easings


class TestStudioToolsRegistered:
    def test_studio_tools_in_registry(self):
        from nomorals.tools.registry import ToolRegistry

        class FakeContext:
            workspace = "/tmp/studio_test_ws"

        reg = ToolRegistry(context=FakeContext()).register_builtins()
        expected = [
            "studio_cut_silences", "studio_detect_scenes", "studio_split_scenes",
            "studio_reframe", "studio_rough_cut", "studio_batch",
            "studio_grade_lut", "studio_grade_curves", "studio_grade_wheels",
            "studio_auto_grade", "studio_match_shot", "studio_look",
            "studio_keyframe", "studio_fade", "studio_zoom_ramp",
            "studio_timeline_new", "studio_render_timeline",
        ]
        schemas = reg.schemas()
        names = {s.get("name") for s in schemas}
        missing = [e for e in expected if e not in names]
        assert not missing, f"missing studio tools: {missing}"

    def test_timeline_new(self):
        from nomorals.tools.registry import ToolRegistry

        class FakeContext:
            workspace = "/tmp/studio_test_ws"

        reg = ToolRegistry(context=FakeContext()).register_builtins()
        out = reg.call("studio_timeline_new", name="test")
        assert out.ok
        assert out.value["timeline"]["name"] == "test"
        assert out.value["timeline"]["video"]["kind"] == "video"
        assert "text_layers" in out.value["timeline"]

    def test_keyframe_tool_bad_key(self):
        from nomorals.tools.registry import ToolRegistry

        class FakeContext:
            workspace = "/tmp/studio_test_ws"

        reg = ToolRegistry(context=FakeContext()).register_builtins()
        out = reg.call("studio_keyframe", clip_id="c", effect="e",
                       param="p", keys=["badkey"])
        assert out.ok
        assert out.value["ok"] is False

    def test_look_unknown(self):
        from nomorals.tools.registry import ToolRegistry

        class FakeContext:
            workspace = "/tmp/studio_test_ws"

        reg = ToolRegistry(context=FakeContext()).register_builtins()
        out = reg.call("studio_look", path="/nonexistent.mp4", look="nope")
        assert out.ok
        assert out.value["ok"] is False


class TestColorLooks:
    def test_list_looks(self):
        from nomorals.media.studio_color import list_looks, LOOKS
        assert set(list_looks()) == set(LOOKS)
        # engines stay style-agnostic: looks are data, not code
        assert all(isinstance(v, dict) for v in LOOKS.values())

    def test_apply_lut_missing(self):
        from nomorals.media.studio_color import apply_lut
        r = apply_lut("/nonexistent.mp4", "/nonexistent.cube")
        assert r["ok"] is False

    def test_apply_curves_no_curves(self):
        from nomorals.media.studio_color import apply_curves
        r = apply_curves("/nonexistent.mp4")
        assert r["ok"] is False

    def test_list_luts_empty(self):
        from nomorals.media.studio_color import list_luts
        # no assertion on content; just must not crash
        assert isinstance(list_luts(search_dirs=["/nonexistent"]), list)


class TestAutomationHelpers:
    def test_batch_unknown_op(self):
        from nomorals.media.studio_automation import batch_process
        r = batch_process([], op="nope")
        assert r["ok"] is False

    def test_reframe_bad_aspect(self):
        from nomorals.media.studio_automation import auto_reframe
        r = auto_reframe("/nonexistent.mp4", aspect="bad")
        assert r["ok"] is False
