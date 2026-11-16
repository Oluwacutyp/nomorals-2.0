"""Video watcher: Devon truly sees and hears a video.

Organ tests use a REAL local video file (generated with ffmpeg:
animated test pattern + a sine tone) and injected fake seeing/hearing
rungs — the organ's own work (probe, sampling, stitching, honest
degradation, rendering) is what's pinned; the vision model and the
STT backend are not available in tests, and the fakes stand in for
them at the exact seam the production rungs use. The handler tests
patch the real rung entry points (tools.vision.extract /
tools.audio.stt) so the full chat path runs against the real organ.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.core.errors import ModelError, ToolError
from nomorals.media.video_watch import (
    render_watch,
    watch_capabilities,
    watch_video,
)

TMP_ROOT = os.environ.get("NM_TEST_TMPDIR") or None


def _make_video(path: str, *, duration: int = 6, audio: bool = True) -> str:
    """Generate a small real mp4: animated test pattern (+ tone)."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i",
           f"testsrc=size=320x240:rate=10:duration={duration}"]
    if audio:
        cmd += ["-f", "lavfi", "-i",
                f"sine=frequency=440:duration={duration}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest", path]
    else:
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # No libx264 in this build — mpeg4 is always there.
        cmd = [c if c != "libx264" else "mpeg4" for c in cmd]
        proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"ffmpeg failed: {proc.stderr[:400]}"
    assert Path(path).exists() and Path(path).stat().st_size > 0
    return path


def _fake_describe(seen_prompts: list | None = None):
    def describe(data: bytes, prompt: str) -> dict:
        if seen_prompts is not None:
            seen_prompts.append(prompt)
        return {"summary": "a test pattern with a moving cross",
                "key_text": "",
                "notable_elements": ["test pattern", "moving cross"],
                "scene_type": "other"}
    return describe


def _fake_transcribe(text: str = "hello from the video"):
    def transcribe(path: str) -> dict:
        assert Path(path).exists(), f"audio track missing: {path}"
        return {"text": text, "provider": "fake-stt", "seconds": 0.1}
    return transcribe


class WatchVideoTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(
            prefix="videowatch-test-", dir=TMP_ROOT)
        self.addCleanup(self._tmp.cleanup)
        self.video = _make_video(str(Path(self._tmp.name) / "clip.mp4"))
        self.silent = _make_video(
            str(Path(self._tmp.name) / "silent.mp4"), audio=False)

    def test_watch_real_video_with_fake_rungs(self):
        result = watch_video(
            self.video, frames=6,
            describe_fn=_fake_describe(),
            transcribe_fn=_fake_transcribe())
        self.assertEqual(result["status"], "watched")
        self.assertEqual(result["frames_sampled"], 6)
        self.assertEqual(len(result["frames"]), 6)
        self.assertEqual(result["transcript"], "hello from the video")
        self.assertEqual(result["transcript_provider"], "fake-stt")
        self.assertAlmostEqual(result["duration"], 6.0, delta=0.5)
        self.assertIn("test pattern", result["summary"])
        self.assertIn("hello from the video", result["summary"])

    def test_frame_timestamps_are_spaced(self):
        result = watch_video(
            self.video, frames=6,
            describe_fn=_fake_describe(),
            transcribe_fn=_fake_transcribe())
        stamps = [f["t"] for f in result["frames"]]
        self.assertEqual(len(stamps), 6)
        self.assertAlmostEqual(stamps[0], 0.0, delta=0.01)
        for earlier, later in zip(stamps, stamps[1:], strict=False):
            self.assertLess(earlier, later)
        self.assertLess(stamps[-1], result["duration"])

    def test_silent_video_is_watched_not_failed(self):
        result = watch_video(
            self.silent, frames=4,
            describe_fn=_fake_describe(),
            transcribe_fn=_fake_transcribe())
        self.assertEqual(result["status"], "watched")
        self.assertEqual(result["transcript"], "")
        self.assertTrue(any("silent" in limit for limit in result["limits"]))

    def test_vision_down_still_hears(self):
        def blind(_data: bytes, _prompt: str) -> dict:
            raise ModelError("no vision-capable model provider is configured")

        result = watch_video(
            self.video, frames=4,
            describe_fn=blind,
            transcribe_fn=_fake_transcribe())
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["transcript"], "hello from the video")
        self.assertTrue(any("could not see" in limit
                            for limit in result["limits"]))

    def test_both_rungs_down_is_unavailable(self):
        def blind(_data: bytes, _prompt: str) -> dict:
            raise ModelError("no vision-capable model provider is configured")

        def deaf(_path: str) -> dict:
            raise ToolError("no STT backend available")

        result = watch_video(
            self.video, frames=4,
            describe_fn=blind, transcribe_fn=deaf)
        self.assertEqual(result["status"], "unavailable")
        joined = "; ".join(result["limits"])
        self.assertIn("could not see", joined)
        self.assertIn("could not hear", joined)

    def test_missing_file_is_unavailable_not_a_crash(self):
        result = watch_video(
            str(Path(self._tmp.name) / "nope.mp4"),
            describe_fn=_fake_describe(),
            transcribe_fn=_fake_transcribe())
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("could not read", result["limits"][0])

    def test_question_is_threaded_into_frame_prompts(self):
        prompts: list[str] = []
        result = watch_video(
            self.video, frames=3, question="is anyone dancing?",
            describe_fn=_fake_describe(prompts),
            transcribe_fn=_fake_transcribe())
        self.assertEqual(result["question"], "is anyone dancing?")
        self.assertTrue(prompts)
        for prompt in prompts:
            self.assertIn("is anyone dancing?", prompt)

    def test_render_watch_shape(self):
        result = watch_video(
            self.video, frames=3,
            describe_fn=_fake_describe(),
            transcribe_fn=_fake_transcribe())
        text = render_watch(result)
        self.assertIn("6.0s video", text)
        self.assertIn("[0.0s]", text)
        self.assertIn("test pattern", text)
        self.assertIn("hello from the video", text)
        self.assertNotIn("⚠️", text)  # nothing to warn about when fully watched

    def test_render_unavailable_names_the_wall(self):
        result = watch_video(
            str(Path(self._tmp.name) / "nope.mp4"),
            describe_fn=_fake_describe(),
            transcribe_fn=_fake_transcribe())
        text = render_watch(result)
        self.assertIn("could not watch", text)
        self.assertIn("could not read", text)

    def test_capabilities_report(self):
        caps = watch_capabilities(context=SimpleNamespace(router=None))
        self.assertTrue(caps["ffmpeg"])  # ffmpeg is on this box
        self.assertFalse(caps["vision"])  # no router on a bare context
        self.assertIn("vision", " ".join(caps["notes"]).lower())


def _bare_runtime():
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.storage.db import Database

    rt = PartnerRuntime.__new__(PartnerRuntime)
    rt.context = SimpleNamespace(db=Database(":memory:"))
    rt._owner_chats = {"telegram:1"}
    # Same harness shape as tests/test_caption_vision_voices.py: the
    # send path returns the text instead of touching a gateway.
    rt._send_long_checked = lambda platform, chat, text, *a, **k: text
    return rt


def _owner_message(media=None):
    return SimpleNamespace(
        chat=SimpleNamespace(key="telegram:1", platform="telegram"),
        text="/video watch clip.mp4",
        meta={"is_owner": True},
        media=media or [],
    )


def _stranger_message():
    return SimpleNamespace(
        chat=SimpleNamespace(key="telegram:999", platform="telegram"),
        text="/video watch clip.mp4",
        meta={},
        media=[],
    )


class WatchCommandTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(
            prefix="videowatch-cmd-", dir=TMP_ROOT)
        self.addCleanup(self._tmp.cleanup)
        self.video = _make_video(str(Path(self._tmp.name) / "clip.mp4"))
        self.rt = _bare_runtime()

    def test_video_watch_attached_video_owner(self):
        media = [SimpleNamespace(kind="video", path=self.video,
                                 name="clip.mp4", mime="video/mp4")]
        with mock.patch("nomorals.tools.vision.extract",
                        return_value={"summary": "a test pattern",
                                      "key_text": "",
                                      "notable_elements": [],
                                      "scene_type": "other"}), \
            mock.patch("nomorals.tools.audio.stt",
                       return_value={"text": "spoken words",
                                     "provider": "fake", "seconds": 0.1}):
            out = self.rt._control_video(
                "watch", chat_key="telegram:1",
                message=_owner_message(media))
        self.assertIn("6.0s video", out)
        self.assertIn("test pattern", out)
        self.assertIn("spoken words", out)

    def test_video_watch_path_owner(self):
        with mock.patch("nomorals.tools.vision.extract",
                        return_value={"summary": "a test pattern",
                                      "key_text": "",
                                      "notable_elements": [],
                                      "scene_type": "other"}), \
            mock.patch("nomorals.tools.audio.stt",
                       return_value={"text": "spoken words",
                                     "provider": "fake", "seconds": 0.1}):
            out = self.rt._control_video(
                f"watch {self.video} what is happening?",
                chat_key="telegram:1",
                message=_owner_message())
        self.assertIn("test pattern", out)
        self.assertIn("what is happening?", out)

    def test_video_watch_no_rungs_is_honest(self):
        # No vision router on the bare context, no STT backend here:
        # the reply names both walls instead of inventing a description.
        out = self.rt._control_video(
            f"watch {self.video}", chat_key="telegram:1",
            message=_owner_message())
        self.assertIn("could not watch", out)
        self.assertIn("could not see", out)
        self.assertIn("could not hear", out)

    def test_video_watch_stranger_refused(self):
        out = self.rt._control_video(
            f"watch {self.video}", chat_key="telegram:999",
            message=_stranger_message())
        self.assertEqual(out, "that one's just for the owner.")

    def test_video_watch_info(self):
        out = self.rt._control_video(
            "watch info", chat_key="telegram:1",
            message=_owner_message())
        self.assertIn("ffmpeg", out.lower())
        self.assertIn("vision", out.lower())

    def test_video_watch_usage(self):
        out = self.rt._control_video(
            "watch", chat_key="telegram:1", message=_owner_message())
        self.assertIn("usage: /video watch", out)

    def test_video_search_still_works(self):
        # "watch" as a search term must not hijack plain /video search:
        # /video watchlist ... still searches (no crash, no watch reply).
        out = self.rt._control_video("watchlist lofi")
        self.assertNotIn("could not watch", out)


class WatchDocsTests(unittest.TestCase):
    def test_video_details_document_watch(self):
        from nomorals.social.chat.control import COMMAND_DETAILS, help_text

        self.assertIn("watch", COMMAND_DETAILS["video"]["usage"])
        self.assertIn("/video watch", help_text())


if __name__ == "__main__":
    unittest.main()
