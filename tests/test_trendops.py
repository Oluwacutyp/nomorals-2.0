"""Tests for TrendOps — trend-driven video editing.

Proves: format validation + JSON round-trips, expiry enforcement
(a dead trend never applies), the roadster seed, the ramp/beat
segment plan, a real ffmpeg render of a trend cut, fail-closed
beats/missing-footage behaviour, capture deriving a format from a
real clip, propose ranking, and chat/tool registration.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import pytest

from nomorals.media.edit_engine import EditSpec
from nomorals.media.trendops import FormatLibrary, TrendFormat, TrendOps
from nomorals.media_edit.videos import MediaEditError, video_probe

NEEDS_FFMPEG = pytest.mark.skipif(
    os.system("ffmpeg -hide_banner -version > /dev/null 2>&1") != 0,
    reason="ffmpeg not installed",
)


@pytest.fixture()
def root(tmp_path):
    return tmp_path / "trendops"


@pytest.fixture()
def ops(root):
    return TrendOps(root=root)


@pytest.fixture()
def workdir(tmp_path):
    d = tmp_path / "footage"
    d.mkdir()
    return d


def _mp4(workdir, name="clip.mp4", color="red", seconds=4.0, w=320, h=568):
    from nomorals.media_edit.videos import run_ffmpeg

    p = workdir / name
    run_ffmpeg(["-f", "lavfi", "-i",
                f"color=c={color}:s={w}x{h}:d={seconds}:r=30",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)],
               duration=seconds)
    return p


def _test_format(**overrides):
    data = {
        "slug": "testcut",
        "name": "Test cut",
        "ramp_curve": [[0.0, 0.5, 0.5], [0.5, 1.0, 1.0]],
        "cut_rule": "beats",
        "canvas": {"width": 320, "height": 568, "fps": 15},
        "expires_at": time.time() + 86400.0,
    }
    data.update(overrides)
    return TrendFormat.from_dict(data)


# ---------------------------------------------------------------------------
# format validation
# ---------------------------------------------------------------------------

class TestFormatValidation:
    def test_bad_slug(self):
        with pytest.raises(MediaEditError):
            TrendFormat.from_dict({"slug": "Bad Slug!"})

    def test_ramp_overlap(self):
        with pytest.raises(MediaEditError):
            _test_format(ramp_curve=[[0.0, 0.6, 1.0], [0.5, 1.0, 2.0]])

    def test_ramp_speed_range(self):
        with pytest.raises(MediaEditError):
            _test_format(ramp_curve=[[0.0, 1.0, 0.0]])
        with pytest.raises(MediaEditError):
            _test_format(ramp_curve=[[0.0, 1.0, 17.0]])

    def test_bad_effect(self):
        with pytest.raises(MediaEditError):
            _test_format(effect_cycle=["not_an_effect"])

    def test_bad_caption_style(self):
        with pytest.raises(MediaEditError):
            _test_format(caption={"style": "comic_sans"})

    def test_unknown_base_style(self):
        with pytest.raises(MediaEditError):
            _test_format(base_style="nope")

    def test_bad_transition(self):
        with pytest.raises(MediaEditError):
            _test_format(transitions="teleport")

    def test_bad_cut_rule(self):
        with pytest.raises(MediaEditError):
            _test_format(cut_rule="vibes")

    def test_json_round_trip(self):
        fmt = _test_format()
        back = TrendFormat.from_dict(json.loads(json.dumps(fmt.to_dict())))
        assert back.slug == fmt.slug
        assert back.ramp_curve == fmt.ramp_curve
        assert back.canvas == fmt.canvas


# ---------------------------------------------------------------------------
# library: seed, expiry, prune
# ---------------------------------------------------------------------------

class TestFormatLibrary:
    def test_seeds_roadster(self, root):
        lib = FormatLibrary(root)
        assert "roadster" in lib.list()
        fmt = lib.get("roadster")
        assert fmt.cut_rule == "beats"
        assert fmt.ramp_curve[0][2] == pytest.approx(0.5)
        assert fmt.is_live()

    def test_save_and_get(self, root):
        lib = FormatLibrary(root)
        lib.save(_test_format())
        assert lib.get("testcut").name == "Test cut"

    def test_expired_never_applies(self, root):
        lib = FormatLibrary(root)
        lib.save(_test_format(slug="dead",
                              expires_at=time.time() - 10.0))
        assert "dead" not in lib.list()
        assert "dead" in lib.list(live_only=False)
        with pytest.raises(MediaEditError, match="expired"):
            lib.get("dead")

    def test_prune(self, root):
        lib = FormatLibrary(root)
        lib.save(_test_format(slug="dead",
                              expires_at=time.time() - 10.0))
        assert lib.prune() == ["dead"]
        assert "dead" not in lib.list(live_only=False)

    def test_unknown_slug(self, root):
        lib = FormatLibrary(root)
        with pytest.raises(MediaEditError, match="unknown trend format"):
            lib.get("nope")

    def test_bump_uses(self, root):
        lib = FormatLibrary(root)
        lib.save(_test_format())
        lib.bump_uses("testcut")
        assert lib.get("testcut").uses == 1


# ---------------------------------------------------------------------------
# spec building: ramp + beat segment plan
# ---------------------------------------------------------------------------

class TestBuildSpec:
    def test_piece_speeds_follow_ramp(self, workdir):
        clip = _mp4(workdir)
        fmt = _test_format()
        spec = fmt.build_spec([str(clip)], beats=[1.0, 2.0, 3.0])
        assert isinstance(spec, EditSpec)
        assert len(spec.segments) == 4
        assert [c.speed for c in spec.clips] == [
            pytest.approx(0.5), pytest.approx(0.5),
            pytest.approx(1.0), pytest.approx(1.0)]
        assert spec.cut_points == [1.0, 2.0, 3.0]
        assert spec.width == 320 and spec.height == 568

    def test_beats_rule_without_beats_fails_closed(self, workdir):
        clip = _mp4(workdir)
        fmt = _test_format()
        with pytest.raises(MediaEditError, match="no beat grid"):
            fmt.build_spec([str(clip)])

    def test_missing_footage_fails_closed(self):
        fmt = _test_format()
        with pytest.raises(MediaEditError, match="no such footage"):
            fmt.build_spec(["/nope/missing.mp4"], beats=[1.0])

    def test_interval_grid(self, workdir):
        clip = _mp4(workdir)
        fmt = _test_format(cut_rule="interval", interval_s=1.0)
        spec = fmt.build_spec([str(clip)])
        assert spec.cut_points == [1.0, 2.0, 3.0]
        assert len(spec.segments) == 4


# ---------------------------------------------------------------------------
# apply: real render
# ---------------------------------------------------------------------------

@NEEDS_FFMPEG
class TestApplyRenders:
    def test_apply_renders_real_mp4(self, ops, workdir):
        clip = _mp4(workdir)
        ops.add_format(_test_format())
        result = ops.apply_format(str(clip), "testcut",
                                  beats=[1.0, 2.0, 3.0])
        assert result["ok"] is True
        out = Path(result["output"])
        assert out.exists() and out.stat().st_size > 0
        info = video_probe(out)
        assert info["width"] == 320 and info["height"] == 568
        assert abs(info["duration"] - 4.0) < 0.4
        assert result["segments"] == 4
        assert result["beats_used"] == 3
        assert result["review"]["ok"] is True
        assert result["review"]["qa"]["status"] == "qa_unavailable"
        assert ops.get_format("testcut").uses == 1

    def test_apply_with_caption(self, ops, workdir):
        clip = _mp4(workdir)
        ops.add_format(_test_format())
        result = ops.apply_format(str(clip), "testcut", beats=[1.0, 2.0],
                                  caption="Me and bro in 2030")
        assert result["ok"] is True
        assert Path(result["output"]).exists()

    def test_apply_beats_rule_without_beats_fails_closed(self, ops, workdir):
        clip = _mp4(workdir)
        ops.add_format(_test_format())
        with pytest.raises(MediaEditError, match="no beat grid"):
            ops.apply_format(str(clip), "testcut")

    def test_apply_missing_footage_fails_closed(self, ops):
        ops.add_format(_test_format())
        with pytest.raises(MediaEditError, match="no such footage"):
            ops.apply_format("/nope/missing.mp4", "testcut",
                             beats=[1.0])

    def test_apply_expired_format_fails_closed(self, ops, workdir):
        clip = _mp4(workdir)
        ops.add_format(_test_format(slug="dead",
                                    expires_at=time.time() - 10.0))
        with pytest.raises(MediaEditError, match="expired"):
            ops.apply_format(str(clip), "dead", beats=[1.0])


# ---------------------------------------------------------------------------
# capture + propose
# ---------------------------------------------------------------------------

class _FakeReport:
    def __init__(self, ok, path, title=""):
        self.ok = ok
        self.path = path
        self.title = title

    def summary(self):
        return "fake failure"


class _FakeDownloader:
    def __init__(self, clip):
        self.clip = clip

    def download(self, url, dest_dir=None, **_ignored):
        dest = Path(dest_dir) / "captured.mp4"
        shutil.copyfile(self.clip, dest)
        return _FakeReport(True, str(dest), title="trend clip")


class TestCaptureAndPropose:
    def test_capture_derives_format(self, ops, workdir):
        clip = _mp4(workdir)
        ops._downloader = _FakeDownloader(clip)
        got = ops.capture_trend("https://tiktok.example/@x/video/1",
                                name="caught", niche="gta")
        assert got["ok"] is True
        assert got["slug"] == "caught"
        assert got["measured"]["duration"] == pytest.approx(4.0, abs=0.2)
        # testsrc has no audio: no beats measured → interval fallback, stated
        assert got["format"]["cut_rule"] == "interval"
        assert "caught" in ops.library.list()

    def test_capture_download_failure_fails_closed(self, ops):
        class _Fail:
            def download(self, url, dest_dir=None, **_ignored):
                return _FakeReport(False, "")

        ops._downloader = _Fail()
        with pytest.raises(MediaEditError, match="could not catch"):
            ops.capture_trend("https://tiktok.example/@x/video/1",
                              name="nope")

    def test_propose_ranks_by_expiry(self, ops):
        now = time.time()
        ops.add_format(_test_format(slug="soon",
                                    expires_at=now + 3600.0))
        ops.add_format(_test_format(slug="later",
                                    expires_at=now + 30 * 86400.0))
        rows = ops.propose()
        assert [r["slug"] for r in rows[:2]] == ["soon", "later"]
        assert rows[0]["apply"].startswith("/trendops apply soon")

    def test_propose_empty(self, ops):
        # fresh library seeds roadster, so prune it away first
        ops.library.prune()
        ops.remove_format("roadster")
        assert ops.propose() == []


# ---------------------------------------------------------------------------
# registration: tool + chat dispatch
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_tool_registered(self):
        from nomorals.tools.media import register as register_media
        from nomorals.tools.registry import ToolRegistry

        reg = ToolRegistry()
        register_media(reg)
        assert "trendops" in reg.names()
        spec = reg.get("trendops")
        assert spec is not None
        assert "apply" in spec.description

    def test_runtime_dispatch_mentions_trendops(self):
        src = Path("nomorals/agents/partner/runtime.py").read_text()
        assert '"trendops"' in src and "_control_trendops" in src
        src2 = Path("nomorals/agents/partner/runtime_media.py").read_text()
        assert "def _control_trendops" in src2

    def test_chat_registry_has_trendops(self):
        from nomorals.social.chat import control as control_mod

        found = False
        for name in dir(control_mod):
            obj = getattr(control_mod, name)
            if isinstance(obj, dict) and "trendops" in obj:
                found = True
                break
        assert found, "trendops missing from control.py registries"
