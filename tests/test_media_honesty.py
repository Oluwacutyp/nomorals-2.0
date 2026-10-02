"""Honest-status tests for media/edit pipelines (wave D).

Every op that degrades, clamps, or skips must say so explicitly:

* ``make_gif`` reports the clamped fps/width it actually used;
* ``transcode`` reports the real codecs and the effective CRF (VP9
  remaps x264-style CRF values);
* off-canvas text/composite layers warn instead of silently drawing
  nothing;
* ``_dispatch_video`` sandboxes every path it hands to ffmpeg
  (concat sources, subtitle files), rejecting out-of-workspace paths
  with a real error instead of failing deep inside ffmpeg.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from nomorals.media_edit import videos
from nomorals.media_edit.studio import composite_layers, op_text_layer


def _make_clip(path: Path, duration: int = 4) -> Path:
    if not shutil.which("ffmpeg"):
        raise unittest.SkipTest("ffmpeg not available")
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc=duration={duration}:size=320x240:rate=30",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(path), "-y"],
        capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"fixture clip failed: {proc.stderr[:300]}")
    return path


class GifHonestyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="honest_gif_"))
        self.addCleanup(shutil.rmtree, self.root, True)
        _make_clip(self.root / "clip.mp4")

    def test_make_gif_reports_clamped_fps_and_width(self) -> None:
        res = videos.make_gif(self.root / "clip.mp4", fps=60, width=4000,
                              duration=1)
        # 60fps/4000px were clamped to the 20fps / 800px sanity caps —
        # the result must report what it actually did.
        self.assertEqual(res["fps"], 20)
        self.assertEqual(res["width"], 800)
        self.assertTrue(Path(res["output"]).exists())

    def test_make_gif_keeps_sane_values(self) -> None:
        res = videos.make_gif(self.root / "clip.mp4", fps=10, width=320,
                              duration=1)
        self.assertEqual(res["fps"], 10)
        self.assertEqual(res["width"], 320)


class TranscodeHonestyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="honest_tc_"))
        self.addCleanup(shutil.rmtree, self.root, True)
        _make_clip(self.root / "clip.mp4")

    def test_transcode_reports_codecs_and_effective_crf(self) -> None:
        res = videos.transcode(self.root / "clip.mp4", ext=".webm", crf=23)
        self.assertEqual(res["video_codec"], "libvpx-vp9")
        self.assertEqual(res["audio_codec"], "libopus")
        # VP9's CRF scale differs: 23 (x264-style) -> 30 actually used
        self.assertEqual(res["crf"], 30)
        self.assertTrue(Path(res["output"]).exists())

    def test_transcode_mp4_keeps_x264_crf(self) -> None:
        res = videos.transcode(self.root / "clip.mp4", ext=".mp4", crf=25)
        self.assertEqual(res["video_codec"], "libx264")
        self.assertEqual(res["audio_codec"], "aac")
        self.assertEqual(res["crf"], 25)


class OffCanvasHonestyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = Image.new("RGB", (200, 200), (10, 20, 30))

    def test_text_layer_offcanvas_warns_and_returns_base(self) -> None:
        with self.assertLogs("nomorals.media_edit.studio",
                             level="WARNING") as logs:
            out = op_text_layer(self.base, "hi", position=(-500, -500))
        self.assertTrue(any("off-canvas" in m for m in logs.output))
        # pixels unchanged: nothing was drawn
        self.assertEqual(list(out.getdata()), list(self.base.getdata()))

    def test_composite_offcanvas_layer_warns(self) -> None:
        with self.assertLogs("nomorals.media_edit.studio",
                             level="WARNING") as logs:
            out = composite_layers(self.base, [
                {"type": "shape", "shape": "rect",
                 "box": [500, 500, 600, 600], "fill": "red"},
            ])
        self.assertTrue(any("off-canvas" in m for m in logs.output))
        self.assertEqual(list(out.convert("RGB").getdata()),
                         list(self.base.getdata()))


class DispatchSandboxTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="honest_sb_"))
        self.addCleanup(shutil.rmtree, self.root, True)
        (self.root / "a.mp4").write_bytes(b"fake")
        (self.root / "b.mp4").write_bytes(b"fake")
        self.context = SimpleNamespace(
            settings=SimpleNamespace(workspace_dir=str(self.root)),
            db=None, router=None)

    def test_concat_rejects_out_of_workspace_source(self) -> None:
        from nomorals.tools.media_edit import _dispatch_video
        from nomorals.core.errors import NoMoralsError
        with self.assertRaises(NoMoralsError):
            _dispatch_video(
                {"video_op": "concat", "sources": ["/etc/passwd"]},
                self.root / "a.mp4", context=self.context)

    def test_burn_subtitles_rejects_out_of_workspace_sub(self) -> None:
        from nomorals.tools.media_edit import _dispatch_video
        from nomorals.core.errors import NoMoralsError
        with self.assertRaises(NoMoralsError):
            _dispatch_video(
                {"video_op": "burn_subtitles",
                 "subtitles": "/etc/passwd"},
                self.root / "a.mp4", context=self.context)


if __name__ == "__main__":
    unittest.main()
