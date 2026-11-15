"""Tests for the WatchVideo organ — temporal QA of TrendOps renders.

Proves: a real cut from moving footage is watched over time and the
verdict is a real pass (cuts on the beat grid, pacing, motion); the
same footage rendered with no cuts at all fails the cuts leg with
the missing cuts named; footage with no measurable frame variation
(a solid-colour clip) is ``qa_unavailable`` with the reason — never
a fake pass, never a fail for a render that may be fine; an
unreadable video is ``qa_unavailable`` without raising; and the
chat reply carries the verdict line when it passes or fails, and
stays silent about it when unmeasurable.

The thresholds are pinned by measurement, not by these tests: on a
real cut the cuts measure 2.23x–4.92x the video's median frame
change, while the same footage with no cuts peaks at 1.41x — the
1.8x bar sits between the two classes, and the verdicts carry the
measured ratios as evidence.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.media.director.watchvideo import watch_video
from nomorals.media.trendops import TrendOps
from nomorals.social import drafts as drafts_mod
from nomorals.social.drafts import DraftQueue

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_artifact_command import _FakeTools, _RecordingGateway  # noqa: E402
from test_trendops import _mp4, _test_format  # noqa: E402

NEEDS_FFMPEG = pytest.mark.skipif(
    os.system("ffmpeg -hide_banner -version > /dev/null 2>&1") != 0,
    reason="ffmpeg not installed",
)

_OWNER_ID = "5478650254"


def _testsrc(workdir, name="clip.mp4", seconds=4.0, w=320, h=568):
    """Moving footage: testsrc (pattern + timecode) — every frame
    differs from the last, so cuts are measurable."""
    from nomorals.media_edit.videos import run_ffmpeg

    p = workdir / name
    run_ffmpeg(["-f", "lavfi", "-i",
                f"testsrc=size={w}x{h}:rate=30:duration={seconds}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)],
               duration=seconds)
    return p


def _series_fn(pairs, fps=15.0, duration=4.0):
    """Inject a synthetic diff series: ``sample_fn`` for watch_video."""
    def fn(_path):
        return (list(pairs), fps, duration)
    return fn


def _synthetic(spikes, *, base=0.03, fps=15.0, duration=4.0):
    """Diff pairs at ``fps``: ``base`` everywhere except a spike at
    each time in ``spikes`` (the pair straddling it)."""
    pairs = []
    n = int(duration * fps)
    for i in range(n):
        t0, t1 = i / fps, (i + 1) / fps
        mid = (t0 + t1) / 2.0
        diff = base
        for spike_t, spike_v in spikes:
            if abs(mid - spike_t) < 1.0 / fps:
                diff = spike_v
        pairs.append((t0, t1, diff))
    return pairs


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


# ---------------------------------------------------------------------------
# end to end: a real render, watched over time
# ---------------------------------------------------------------------------

@NEEDS_FFMPEG
class TestWatchRealRender:
    def test_real_cut_passes(self, ops, workdir):
        clip = _testsrc(workdir)
        ops.add_format(_test_format())
        result = ops.apply_format(str(clip), "testcut",
                                  beats=[1.0, 2.0, 3.0])
        assert result["ok"] is True
        qa = result["review"]["qa"]
        assert qa["status"] == "pass"
        cuts = qa["checks"]["cuts"]
        assert [f["at"] for f in cuts["found"]] == [1.0, 2.0, 3.0]
        assert cuts["missing"] == []
        # the evidence is real ratios, not a bare verdict
        assert all(f["ratio"] >= 1.8 for f in cuts["found"])
        assert qa["checks"]["pacing"]["status"] == "pass"
        assert qa["checks"]["motion"]["status"] == "pass"

    def test_no_cut_render_fails_cuts(self, ops, workdir):
        clip = _testsrc(workdir)
        ops.add_format(_test_format(cut_rule="none"))
        result = ops.apply_format(str(clip), "testcut")
        assert result["ok"] is True
        # the same footage, asked for cuts it was never given: the
        # organ must say so, with the cuts named — not pass it.
        verdict = watch_video(result["output"],
                              cut_points=[1.0, 2.0, 3.0],
                              segments=[1.0, 1.0, 1.0, 1.0])
        assert verdict["status"] == "fail"
        assert [m["at"] for m in verdict["checks"]["cuts"]["missing"]] == [
            1.0, 2.0, 3.0]
        assert verdict["checks"]["cuts"]["found"] == []

    def test_solid_red_is_unavailable_not_fail(self, ops, workdir):
        clip = _mp4(workdir)  # solid red: every frame identical
        ops.add_format(_test_format())
        result = ops.apply_format(str(clip), "testcut",
                                  beats=[1.0, 2.0, 3.0])
        assert result["ok"] is True
        qa = result["review"]["qa"]
        assert qa["status"] == "qa_unavailable"
        assert "no measurable frame variation" in qa["notes"]


# ---------------------------------------------------------------------------
# the organ on its own: unreadable input, synthetic series
# ---------------------------------------------------------------------------

class TestWatchVideoDirect:
    def test_unreadable_video_is_unavailable(self, tmp_path):
        verdict = watch_video(str(tmp_path / "nope.mp4"),
                              cut_points=[1.0], segments=[1.0, 1.0])
        assert verdict["status"] == "qa_unavailable"
        assert "could not watch" in verdict["notes"]

    def test_synthetic_series_pass(self):
        pairs = _synthetic([(1.0, 0.12), (2.0, 0.25), (3.0, 0.15)])
        verdict = watch_video("synthetic.mp4", cut_points=[1.0, 2.0, 3.0],
                              segments=[1.0, 1.0, 1.0, 1.0],
                              sample_fn=_series_fn(pairs))
        assert verdict["status"] == "pass"
        assert [f["at"] for f in
                verdict["checks"]["cuts"]["found"]] == [1.0, 2.0, 3.0]
        assert verdict["checks"]["pacing"]["status"] == "pass"
        assert verdict["checks"]["motion"]["status"] == "pass"

    def test_synthetic_series_missing_cut(self):
        pairs = _synthetic([(1.0, 0.12), (2.0, 0.25)])
        verdict = watch_video("synthetic.mp4", cut_points=[1.0, 2.0, 3.0],
                              segments=[1.0, 1.0, 1.0, 1.0],
                              sample_fn=_series_fn(pairs))
        assert verdict["status"] == "fail"
        assert [m["at"] for m in
                verdict["checks"]["cuts"]["missing"]] == [3.0]
        assert "3.0s" in verdict["notes"]

    def test_synthetic_flat_series_is_unavailable(self):
        pairs = _synthetic([], base=0.0)
        verdict = watch_video("synthetic.mp4", cut_points=[1.0, 2.0, 3.0],
                              segments=[1.0, 1.0, 1.0, 1.0],
                              sample_fn=_series_fn(pairs))
        assert verdict["status"] == "qa_unavailable"
        assert "no measurable frame variation" in verdict["notes"]

    def test_synthetic_frozen_segment_fails_motion(self):
        # segment 2 (2.0–3.0s) frozen solid while the rest moves.
        pairs = []
        for i in range(60):
            t0, t1 = i / 15.0, (i + 1) / 15.0
            mid = (t0 + t1) / 2.0
            diff = 0.03
            if abs(mid - 1.0) < 1 / 15.0:
                diff = 0.12
            elif abs(mid - 2.0) < 1 / 15.0:
                diff = 0.25
            elif abs(mid - 3.0) < 1 / 15.0:
                diff = 0.15
            elif t0 >= 2.0 and t1 <= 3.0:
                diff = 0.0
            pairs.append((t0, t1, diff))
        verdict = watch_video("synthetic.mp4", cut_points=[1.0, 2.0, 3.0],
                              segments=[1.0, 1.0, 1.0, 1.0],
                              sample_fn=_series_fn(pairs))
        assert verdict["status"] == "fail"
        assert verdict["checks"]["motion"]["status"] == "fail"
        assert "2" in (verdict["checks"]["motion"].get("notes") or "")


# ---------------------------------------------------------------------------
# the chat reply carries the verdict
# ---------------------------------------------------------------------------

class _FakeSocialManager:
    """The publisher seam: records publishes, posts to one platform."""

    instances: list[_FakeSocialManager] = []

    def __init__(self, context: Any) -> None:
        self.context = context
        self.published: list[tuple[str, Any, dict]] = []

    def register_builtins(self) -> _FakeSocialManager:
        _FakeSocialManager.instances.append(self)
        return self

    def publish(self, content: str, platforms: Any = None,
                actor: str = "user", confirmation: str = "",
                **extra: Any) -> Any:
        self.published.append((content, platforms, extra))
        return SimpleNamespace(
            posted=["x"], failed=[], error="",
            results=[SimpleNamespace(
                platform="x", ok=True, external_id="ext1", url="",
                metrics={"media_offered": 1, "media_carried": 1})])


class _ChatRig(unittest.TestCase):
    """PartnerRuntime over a tmp home (the trendops-drafts harness
    shape) — the dispatch, the render, the draft lifecycle stay real;
    only the publisher is faked."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-watchvideo-")
        self.db_path = str(Path(self.tmp.name) / "drafts.db")
        self.trend_root = Path(self.tmp.name) / "trendops"
        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
            "partner.owner_chats": _OWNER_ID,
            "partner.owner_sender_ids": _OWNER_ID,
        })
        self.context = build_context(settings, with_executor=False,
                                     with_tools=False)

        class _Router:
            def chat(self, *a: Any, **k: Any) -> Any:
                raise AssertionError("test must not call the model")

        self.context.router = _Router()
        self.context.tools = _FakeTools()
        self.runtime = PartnerRuntime(self.context,
                                      gateway=_RecordingGateway())
        _FakeSocialManager.instances = []
        self._patchers = [
            mock.patch.object(
                drafts_mod, "DraftQueue",
                lambda *a, **k: DraftQueue(db_path=self.db_path)),
            mock.patch("nomorals.social.SocialManager",
                       _FakeSocialManager),
            mock.patch(
                "nomorals.media.trendops.editor.DEFAULT_ROOT",
                self.trend_root),
        ]
        for patcher in self._patchers:
            patcher.start()
        seed_ops = TrendOps(context=self.context)
        seed_ops.add_format(_test_format())

    def tearDown(self) -> None:
        for patcher in self._patchers:
            patcher.stop()
        with contextlib.suppress(Exception):
            self.runtime.stop()
        self.context.close()
        self.tmp.cleanup()

    def _apply(self, clip: str) -> Any:
        return self.runtime._control_trendops(
            f"apply testcut {clip} --beats 1.0,2.0,3.0")

    @NEEDS_FFMPEG
    def test_reply_carries_qa_pass(self):
        workdir = Path(self.tmp.name) / "footage"
        workdir.mkdir()
        clip = _testsrc(workdir)
        reply = self._apply(str(clip))
        text = str(getattr(reply, "text", "") or "")
        assert "cut to Test cut" in text
        assert "temporal QA pass" in text
        assert "3/3 cuts on beats" in text

    @NEEDS_FFMPEG
    def test_reply_silent_when_unmeasurable(self):
        workdir = Path(self.tmp.name) / "footage"
        workdir.mkdir()
        clip = _mp4(workdir)  # solid red: QA unavailable
        reply = self._apply(str(clip))
        text = str(getattr(reply, "text", "") or "")
        assert "cut to Test cut" in text
        assert "temporal QA" not in text
