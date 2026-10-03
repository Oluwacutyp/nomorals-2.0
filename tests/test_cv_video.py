"""Tests for nomorals.media_edit.cv_video (multi-backend frame-level video)
and the videos.py OpenCV-backed enhancements.

Fixtures are tiny synthetic videos written with cv2.VideoWriter, so the
suite runs in seconds with no network and no real footage. Backend
selection is exercised explicitly: auto primaries, forced backends, and
degradation when cv2 is unavailable.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import shutil
import sys
import tempfile
import unittest
import unittest.mock

import cv2
import numpy as np

from nomorals.media_edit import cv_video as CV
from nomorals.media_edit import videos as V
from nomorals.media_edit.images import MediaEditError

W, H, FPS = 160, 120, 30
FFMPEG_FILTERS = ["grayscale", "blur", "sharpen", "invert", "sepia",
                  "vignette", "edges", "emboss", "pixelate",
                  "warm", "cool"]
OPENCV_FIRST_FILTERS = ["sketch", "cartoonize"]


def _write_video(path, frames):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, fourcc, FPS, (w, h))
    assert vw.isOpened(), f"writer failed for {path}"
    for f in frames:
        vw.write(f)
    vw.release()


def _gradient_video(path, n=60):
    frames = []
    for i in range(n):
        v = int(255 * i / max(1, n - 1))
        frames.append(np.full((H, W, 3), (v, 255 - v, 128), np.uint8))
    _write_video(path, frames)
    return path


def _scene_change_video(path, n=60):
    # first half solid red (BGR), second half solid blue
    frames = [np.full((H, W, 3), (0, 0, 255), np.uint8)] * (n // 2)
    frames += [np.full((H, W, 3), (255, 0, 0), np.uint8)] * (n - n // 2)
    _write_video(path, frames)
    return path


def _shaky_video(path, n=60):
    # white square drifting right with sinusoidal jitter on black
    frames = []
    for i in range(n):
        img = np.zeros((H, W, 3), np.uint8)
        x = 20 + i + int(6 * np.sin(i * 0.7))
        cv2.rectangle(img, (x, 40), (x + 30, 70), (255, 255, 255), -1)
        frames.append(img)
    _write_video(path, frames)
    return path


def _count_frames(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return n


@contextlib.contextmanager
def _no_cv2():
    """Simulate an install without OpenCV (ffmpeg still present)."""
    old = sys.modules.get("cv2", "ABSENT")
    sys.modules["cv2"] = None
    reloaded = importlib.reload(CV)
    try:
        yield reloaded
    finally:
        if old == "ABSENT":
            del sys.modules["cv2"]
        else:
            sys.modules["cv2"] = old
        importlib.reload(CV)


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cv-video-test-")
        self.vid = _gradient_video(os.path.join(self.tmp, "grad.mp4"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class BackendContractTests(unittest.TestCase):
    def test_module_imports_without_cv2(self):
        # No artificial gate: importing cv_video never requires cv2.
        with _no_cv2() as mod:
            self.assertFalse(mod.cv2_available())
            self.assertFalse(mod._have_cv2())

    def test_cv2_last_resort_hint(self):
        with _no_cv2() as mod:
            with self.assertRaises(ImportError) as ctx:
                mod._cv2()
            self.assertIn("pip install nomorals[media-edit]",
                          str(ctx.exception))

    def test_auto_degrades_to_ffmpeg(self):
        # cv2 missing + ffmpeg present -> auto still works via ffmpeg.
        tmp = tempfile.mkdtemp(prefix="cv-degrade-")
        try:
            vid = _gradient_video(os.path.join(tmp, "g.mp4"))
            with _no_cv2() as mod:
                res = mod.extract_frames(vid, count=3, out_dir=tmp)
                self.assertEqual(res["backend"], "ffmpeg")
                self.assertEqual(res["count"], 3)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_no_backend_at_all(self):
        with _no_cv2() as mod:
            with unittest.mock.patch.object(
                    mod, "_ffmpeg_bin", return_value=None):
                with self.assertRaises(MediaEditError) as ctx:
                    mod.extract_frames("/tmp/x.mp4", count=1)
                self.assertIn("pip install nomorals[media-edit]",
                              str(ctx.exception))

    def test_invalid_backend_rejected(self):
        tmp = tempfile.mkdtemp(prefix="cv-badbe-")
        try:
            vid = _gradient_video(os.path.join(tmp, "g.mp4"))
            for fn, kw in [
                (CV.extract_frames, {"count": 2}),
                (CV.apply_filter_to_video, {"filter_name": "blur"}),
                (CV.create_timelapse, {}),
                (CV.stabilize_basic, {}),
                (CV.frame_diff_highlights, {}),
            ]:
                with self.assertRaises(MediaEditError, msg=fn.__name__):
                    fn(vid, backend="cuda", out_dir=tmp, **kw)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class ExtractFramesTests(_Base):
    def test_auto_primary_is_opencv(self):
        res = CV.extract_frames(self.vid, count=6, out_dir=self.tmp)
        self.assertEqual(res["backend"], "opencv")
        self.assertTrue(res["exact_frames"])
        self.assertEqual(res["count"], 6)
        self.assertEqual(len(res["frames"]), 6)
        for f in res["frames"]:
            self.assertTrue(os.path.exists(f))
        self.assertIn("t0.00s", res["frames"][0])
        self.assertIn(f"t{(59 / FPS):.2f}s", res["frames"][-1])

    def test_forced_ffmpeg(self):
        res = CV.extract_frames(self.vid, count=6, out_dir=self.tmp,
                                backend="ffmpeg")
        self.assertEqual(res["backend"], "ffmpeg")
        self.assertFalse(res["exact_frames"])
        self.assertEqual(res["count"], 6)

    def test_forced_opencv(self):
        res = CV.extract_frames(self.vid, count=4, out_dir=self.tmp,
                                backend="opencv")
        self.assertEqual(res["backend"], "opencv")
        self.assertTrue(res["exact_frames"])

    def test_interval_mode(self):
        res = CV.extract_frames(self.vid, interval=1.0, out_dir=self.tmp)
        self.assertEqual(res["count"], 2)  # 60 frames @30fps, every 30

    def test_timestamps_mode(self):
        res = CV.extract_frames(self.vid, timestamps=["0.5", 1.0],
                                out_dir=self.tmp, width=80)
        self.assertEqual(res["count"], 2)
        img = cv2.imread(res["frames"][0])
        self.assertEqual(img.shape[1], 80)

    def test_exactly_one_selector(self):
        with self.assertRaises(MediaEditError):
            CV.extract_frames(self.vid, out_dir=self.tmp)
        with self.assertRaises(MediaEditError):
            CV.extract_frames(self.vid, count=3, interval=1.0,
                              out_dir=self.tmp)

    def test_bad_inputs(self):
        with self.assertRaises(MediaEditError):
            CV.extract_frames(os.path.join(self.tmp, "nope.mp4"), count=2)
        with self.assertRaises(MediaEditError):
            CV.extract_frames(self.vid, count=0, out_dir=self.tmp)
        with self.assertRaises(MediaEditError):
            CV.extract_frames(self.vid, count=2, out_dir=self.tmp,
                              fmt="bmp", backend="opencv")
        with self.assertRaises(MediaEditError):
            CV.extract_frames(self.vid, timestamps=[9999],
                              out_dir=self.tmp, backend="opencv")

    def test_progress_cb(self):
        seen = []
        CV.extract_frames(self.vid, count=3, out_dir=self.tmp,
                          progress_cb=seen.append)
        self.assertTrue(seen)
        self.assertEqual(seen[-1], 1.0)


class GrabFrameTests(_Base):
    def test_exact_frame_color(self):
        path = _scene_change_video(os.path.join(self.tmp, "sc.mp4"))
        frame, actual = CV.grab_frame_at(path, 0.5)  # red half
        self.assertEqual(frame.shape, (H, W, 3))
        self.assertAlmostEqual(actual, 0.5, places=1)
        mean = frame.mean(axis=(0, 1))
        self.assertGreater(mean[2], 150)  # red channel dominant
        self.assertLess(mean[0], 100)
        frame2, _ = CV.grab_frame_at(path, 1.5)  # blue half
        mean2 = frame2.mean(axis=(0, 1))
        self.assertGreater(mean2[0], 150)
        self.assertLess(mean2[2], 100)

    def test_past_end(self):
        with self.assertRaises(MediaEditError):
            CV.grab_frame_at(self.vid, 9999)


class FilterTests(_Base):
    def _assert_gray(self, path):
        frame, _ = CV.grab_frame_at(path, 0.5)
        b, g, r = (int(v) for v in frame[10, 10])
        self.assertLess(abs(b - g), 12)
        self.assertLess(abs(g - r), 12)

    def test_auto_routes_ffmpeg_filters(self):
        for name in FFMPEG_FILTERS:
            res = CV.apply_filter_to_video(
                self.vid, filter_name=name, out_dir=self.tmp,
                suffix=f"auto-{name}")
            self.assertEqual(res["backend"], "ffmpeg", name)
            self.assertFalse(res["audio_dropped"], name)
            self.assertTrue(os.path.exists(res["output"]), name)
            self.assertEqual(_count_frames(res["output"]), 60, name)
        self._assert_gray(
            CV.apply_filter_to_video(
                self.vid, filter_name="grayscale", out_dir=self.tmp,
                suffix="auto-gray")["output"])

    def test_auto_routes_opencv_first_filters(self):
        for name in OPENCV_FIRST_FILTERS:
            res = CV.apply_filter_to_video(
                self.vid, filter_name=name, out_dir=self.tmp,
                suffix=f"auto-{name}")
            self.assertEqual(res["backend"], "opencv", name)
            self.assertTrue(res["audio_dropped"], name)
            self.assertTrue(os.path.exists(res["output"]), name)

    def test_forced_opencv_all_filters(self):
        for name in sorted(CV.FILTERS):
            res = CV.apply_filter_to_video(
                self.vid, filter_name=name, out_dir=self.tmp,
                suffix=f"cv-{name}", backend="opencv")
            self.assertEqual(res["backend"], "opencv", name)
            self.assertEqual(res["frames"], 60, name)

    def test_forced_ffmpeg_approx_filters(self):
        # sketch/cartoonize are opencv-first, but ffmpeg approximations
        # exist — forced ffmpeg must succeed, not fail.
        for name in OPENCV_FIRST_FILTERS:
            res = CV.apply_filter_to_video(
                self.vid, filter_name=name, out_dir=self.tmp,
                suffix=f"ff-{name}", backend="ffmpeg")
            self.assertEqual(res["backend"], "ffmpeg", name)
            self.assertFalse(res["audio_dropped"], name)
            self.assertTrue(os.path.exists(res["output"]), name)
            self.assertEqual(_count_frames(res["output"]), 60, name)

    def test_unknown_filter(self):
        with self.assertRaises(MediaEditError) as ctx:
            CV.apply_filter_to_video(self.vid, filter_name="nope",
                                     out_dir=self.tmp)
        self.assertIn("grayscale", str(ctx.exception))

    def test_strength_bounds(self):
        with self.assertRaises(MediaEditError):
            CV.apply_filter_to_video(self.vid, filter_name="blur",
                                     strength=2.5, out_dir=self.tmp)
        with self.assertRaises(MediaEditError):
            CV.apply_filter_to_video(self.vid, filter_name="blur",
                                     strength=-0.1, out_dir=self.tmp)


class TimelapseTests(_Base):
    def test_auto_primary_is_ffmpeg_and_keeps_audio(self):
        res = CV.create_timelapse(self.vid, factor=2, out_dir=self.tmp)
        self.assertEqual(res["backend"], "ffmpeg")
        self.assertFalse(res["audio_dropped"])
        # setpts retimes by timestamps, so the frame count is time-based
        # (±2), not frame-exact like the OpenCV path.
        self.assertTrue(28 <= _count_frames(res["output"]) <= 32)
        self.assertAlmostEqual(res["duration_out"], 1.0, places=1)

    def test_forced_opencv(self):
        res = CV.create_timelapse(self.vid, factor=2, out_dir=self.tmp,
                                  backend="opencv")
        self.assertEqual(res["backend"], "opencv")
        self.assertTrue(res["audio_dropped"])
        self.assertEqual(res["frames_in"], 60)
        self.assertEqual(res["frames_out"], 30)

    def test_factor_floor(self):
        with self.assertRaises(MediaEditError):
            CV.create_timelapse(self.vid, factor=1, out_dir=self.tmp)


class StabilizeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cv-stab-test-")
        self.vid = _shaky_video(os.path.join(self.tmp, "shaky.mp4"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _vidstab(self):
        return CV._ffmpeg_has_filter("vidstabdetect", "vidstabtransform")

    def test_auto_picks_best_available(self):
        res = CV.stabilize_basic(self.vid, smoothing_radius=5,
                                 out_dir=self.tmp)
        self.assertTrue(os.path.exists(res["output"]))
        if self._vidstab():
            self.assertEqual(res["backend"], "ffmpeg")
            self.assertEqual(res["method"], "vidstab-two-pass")
            self.assertFalse(res["audio_dropped"])
        else:
            self.assertEqual(res["backend"], "opencv")
            self.assertEqual(res["method"], "feature-tracking")
            self.assertEqual(res["frames"], 60)

    def test_forced_opencv(self):
        res = CV.stabilize_basic(self.vid, smoothing_radius=5,
                                 out_dir=self.tmp, backend="opencv")
        self.assertEqual(res["backend"], "opencv")
        self.assertEqual(res["method"], "feature-tracking")
        self.assertEqual(res["frames"], 60)
        self.assertEqual(_count_frames(res["output"]), 60)
        crop = res["crop"]
        self.assertGreaterEqual(crop["x"], 0)
        self.assertGreaterEqual(crop["y"], 0)
        self.assertTrue(res["audio_dropped"])

    def test_forced_ffmpeg(self):
        if not self._vidstab():
            with self.assertRaises(MediaEditError) as ctx:
                CV.stabilize_basic(self.vid, smoothing_radius=5,
                                   out_dir=self.tmp, backend="ffmpeg")
            self.assertIn("vidstab", str(ctx.exception))
            return
        res = CV.stabilize_basic(self.vid, smoothing_radius=5,
                                 out_dir=self.tmp, backend="ffmpeg")
        self.assertEqual(res["backend"], "ffmpeg")
        self.assertTrue(os.path.exists(res["output"]))

    def test_opencv_no_crop(self):
        res = CV.stabilize_basic(self.vid, smoothing_radius=5,
                                 crop_borders=False, out_dir=self.tmp,
                                 backend="opencv")
        frame, _ = CV.grab_frame_at(res["output"], 0)
        self.assertEqual(frame.shape[1], W)
        self.assertEqual(frame.shape[0], H)

    def test_bad_radius(self):
        with self.assertRaises(MediaEditError):
            CV.stabilize_basic(self.vid, smoothing_radius=0,
                               out_dir=self.tmp)


class HighlightsTests(_Base):
    def test_detects_scene_change(self):
        path = _scene_change_video(os.path.join(self.tmp, "sc.mp4"), n=90)
        res = CV.frame_diff_highlights(path, threshold=0.02, min_gap=0.5,
                                       out_dir=self.tmp)
        self.assertEqual(res["backend"], "opencv")
        self.assertGreaterEqual(res["count"], 1)
        # the cut happens at frame 45 -> t=1.5s
        ts = [e["t"] for e in res["events"]]
        self.assertTrue(any(abs(t - 1.5) < 0.2 for t in ts), ts)
        for ev in res["events"]:
            self.assertTrue(os.path.exists(ev["thumb"]))
            self.assertGreaterEqual(ev["score"], 0.02)

    def test_min_gap_dedupes(self):
        path = _scene_change_video(os.path.join(self.tmp, "sc.mp4"), n=90)
        res = CV.frame_diff_highlights(path, threshold=0.02,
                                       min_gap=999, out_dir=self.tmp)
        self.assertLessEqual(res["count"], 1)

    def test_ffmpeg_backend_honestly_unsupported(self):
        with self.assertRaises(MediaEditError) as ctx:
            CV.frame_diff_highlights(self.vid, out_dir=self.tmp,
                                     backend="ffmpeg")
        self.assertIn("select", str(ctx.exception))

    def test_bad_params(self):
        with self.assertRaises(MediaEditError):
            CV.frame_diff_highlights(self.vid, threshold=0,
                                     out_dir=self.tmp)
        with self.assertRaises(MediaEditError):
            CV.frame_diff_highlights(self.vid, threshold=1.5,
                                     out_dir=self.tmp)
        with self.assertRaises(MediaEditError):
            CV.frame_diff_highlights(self.vid, min_gap=-1,
                                     out_dir=self.tmp)


class VideosEnhancementTests(_Base):
    def test_thumbnail_auto_is_opencv_exact(self):
        res = V.frame_accurate_thumbnail(self.vid, timestamp=1.0,
                                         width=80, out_dir=self.tmp)
        self.assertEqual(res["backend"], "opencv")
        self.assertTrue(res["exact"])
        self.assertTrue(os.path.exists(res["output"]))
        self.assertAlmostEqual(res["actual_t"], 1.0, places=1)
        img = cv2.imread(res["output"])
        self.assertEqual(img.shape[1], 80)

    def test_thumbnail_ffmpeg_fallback(self):
        res = V.frame_accurate_thumbnail(self.vid, timestamp=1.0,
                                         width=80, out_dir=self.tmp,
                                         backend="ffmpeg")
        self.assertEqual(res["backend"], "ffmpeg")
        self.assertFalse(res["exact"])
        self.assertTrue(os.path.exists(res["output"]))

    def test_thumbnail_missing_file(self):
        with self.assertRaises(MediaEditError):
            V.frame_accurate_thumbnail(os.path.join(self.tmp, "nope.mp4"))

    def test_preview_grid_auto_is_opencv_labeled(self):
        res = V.make_preview_grid(self.vid, cols=2, rows=2,
                                  cell_width=80, out_dir=self.tmp)
        self.assertEqual(res["backend"], "opencv")
        self.assertTrue(res["labeled"])
        self.assertTrue(os.path.exists(res["output"]))
        self.assertEqual(res["cells"], 4)
        img = cv2.imread(res["output"])
        self.assertEqual(img.shape[1], 160)  # 2 cols x 80
        self.assertEqual(img.shape[0], 120)  # 2 rows x 60

    def test_preview_grid_ffmpeg_fallback(self):
        res = V.make_preview_grid(self.vid, cols=2, rows=2,
                                  cell_width=80, out_dir=self.tmp,
                                  backend="ffmpeg")
        self.assertEqual(res["backend"], "ffmpeg")
        self.assertFalse(res["labeled"])
        self.assertTrue(os.path.exists(res["output"]))

    def test_preview_grid_bad_dims(self):
        with self.assertRaises(MediaEditError):
            V.make_preview_grid(self.vid, cols=0, out_dir=self.tmp)


class SlowMotionTests(_Base):
    def test_auto_primary_is_ffmpeg(self):
        res = CV.slow_motion(self.vid, factor=2, out_dir=self.tmp)
        self.assertEqual(res["backend"], "ffmpeg")
        self.assertEqual(res["method"], "minterpolate-mci")
        self.assertFalse(res["audio_dropped"])
        # 60 frames @30fps -> 120 frames @60fps, same 2s stretched to 4s
        self.assertTrue(115 <= _count_frames(res["output"]) <= 125)
        self.assertAlmostEqual(res["duration_out"], 4.0, delta=0.3)

    def test_opencv_blend(self):
        vid = _gradient_video(os.path.join(self.tmp, "s.mp4"), n=30)
        res = CV.slow_motion(vid, factor=2, method="blend",
                             out_dir=self.tmp, backend="opencv")
        self.assertEqual(res["backend"], "opencv")
        self.assertEqual(res["method"], "blend")
        self.assertTrue(res["audio_dropped"])
        self.assertEqual(res["frames_out"], 59)  # 30 + 29 mids
        self.assertEqual(_count_frames(res["output"]), 59)

    def test_opencv_flow(self):
        vid = _gradient_video(os.path.join(self.tmp, "f.mp4"), n=12)
        res = CV.slow_motion(vid, factor=2, method="flow",
                             out_dir=self.tmp, backend="opencv")
        self.assertEqual(res["method"], "flow")
        self.assertEqual(res["frames_out"], 23)
        self.assertTrue(os.path.exists(res["output"]))

    def test_bad_args(self):
        with self.assertRaises(MediaEditError):
            CV.slow_motion(self.vid, factor=1, out_dir=self.tmp)
        with self.assertRaises(MediaEditError):
            CV.slow_motion(self.vid, method="warp", out_dir=self.tmp)


class ReverseTests(_Base):
    def test_auto_primary_is_ffmpeg(self):
        res = CV.reverse_video(self.vid, out_dir=self.tmp)
        self.assertEqual(res["backend"], "ffmpeg")
        self.assertFalse(res["audio_dropped"])
        self.assertEqual(_count_frames(res["output"]), 60)
        # first frame of reversed ~= last frame of original (v=255)
        frame, _ = CV.grab_frame_at(res["output"], 0)
        mean = frame.mean(axis=(0, 1))
        self.assertGreater(mean[0], 150)  # B channel high (v=255)

    def test_opencv_fallback(self):
        res = CV.reverse_video(self.vid, out_dir=self.tmp,
                               backend="opencv")
        self.assertEqual(res["backend"], "opencv")
        self.assertTrue(res["audio_dropped"])
        self.assertEqual(res["frames"], 60)
        frame, _ = CV.grab_frame_at(res["output"], 0)
        mean = frame.mean(axis=(0, 1))
        self.assertGreater(mean[0], 150)


class BoomerangTests(_Base):
    def test_boomerang(self):
        res = CV.boomerang(self.vid, start=0, duration=1,
                           out_dir=self.tmp)
        self.assertEqual(res["backend"], "ffmpeg")
        self.assertTrue(os.path.exists(res["output"]))
        # 1s slice @30fps forward + 1s reversed = 60 frames
        self.assertTrue(55 <= _count_frames(res["output"]) <= 65)
        self.assertEqual(res["loops"], 2)

    def test_bad_duration(self):
        with self.assertRaises(MediaEditError):
            CV.boomerang(self.vid, duration=0, out_dir=self.tmp)

    def test_opencv_honestly_unsupported(self):
        with self.assertRaises(MediaEditError) as ctx:
            CV.boomerang(self.vid, out_dir=self.tmp, backend="opencv")
        self.assertIn("mux", str(ctx.exception).lower())


class BlurryTests(_Base):
    def _mixed_video(self):
        rng = np.random.default_rng(7)
        frames = []
        for i in range(60):
            noise = (rng.random((H, W, 3)) * 255).astype(np.uint8)
            if i >= 30:
                noise = cv2.GaussianBlur(noise, (31, 31), 0)
            frames.append(noise)
        path = os.path.join(self.tmp, "mixed.mp4")
        _write_video(path, frames)
        return path

    def test_finds_blurry_second_half(self):
        path = self._mixed_video()
        res = CV.find_blurry_frames(path, threshold=100.0,
                                    out_dir=self.tmp)
        self.assertEqual(res["backend"], "opencv")
        self.assertGreater(res["count"], 0)
        # all detections should be in the blurred half (t >= 1.0s)
        self.assertTrue(all(e["t"] >= 0.9 for e in res["events"]),
                        [e["t"] for e in res["events"]][:5])
        self.assertGreaterEqual(res["blurriest"]["t"], 0.9)
        for ev in res["events"][:3]:
            self.assertTrue(os.path.exists(ev["thumb"]))

    def test_bad_threshold(self):
        with self.assertRaises(MediaEditError):
            CV.find_blurry_frames(self.vid, threshold=0, out_dir=self.tmp)


class HeatmapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cv-heat-test-")
        self.vid = _shaky_video(os.path.join(self.tmp, "shaky.mp4"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_heatmap(self):
        res = CV.motion_heatmap(self.vid, out_dir=self.tmp)
        self.assertEqual(res["backend"], "opencv")
        self.assertTrue(os.path.exists(res["heatmap"]))
        img = cv2.imread(res["heatmap"])
        self.assertEqual(img.shape[1], W)
        self.assertEqual(img.shape[0], H)
        self.assertGreater(res["peak_motion"]["score"], 0)
        self.assertGreater(res["peak_motion"]["t"], 0)

    def test_bad_decay(self):
        with self.assertRaises(MediaEditError):
            CV.motion_heatmap(self.vid, decay=1.5, out_dir=self.tmp)


class SplitTests(_Base):
    def _three_scene_video(self):
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        frames = []
        for c in colors:
            frames += [np.full((H, W, 3), c, np.uint8)] * 30
        path = os.path.join(self.tmp, "scenes.mp4")
        _write_video(path, frames)
        return path

    def test_split_opencv_detect(self):
        path = self._three_scene_video()
        res = CV.split_on_scenes(path, threshold=0.02, min_gap=0.3,
                                 out_dir=self.tmp, backend="opencv")
        self.assertEqual(res["backend"], "opencv")
        self.assertEqual(res["count"], 3)
        for sc in res["scenes"]:
            self.assertTrue(os.path.exists(sc["output"]))
            self.assertTrue(25 <= _count_frames(sc["output"]) <= 35)
        # scenes tile the timeline without gaps
        self.assertAlmostEqual(res["scenes"][0]["start"], 0.0, places=1)
        self.assertAlmostEqual(res["scenes"][-1]["end"], 3.0, delta=0.3)

    def test_split_ffmpeg_detect(self):
        path = self._three_scene_video()
        res = CV.split_on_scenes(path, threshold=0.3, min_gap=0.3,
                                 out_dir=self.tmp, backend="ffmpeg")
        self.assertEqual(res["backend"], "ffmpeg")
        self.assertGreaterEqual(res["count"], 2)
        for sc in res["scenes"]:
            self.assertTrue(os.path.exists(sc["output"]))

    def test_split_auto(self):
        path = self._three_scene_video()
        res = CV.split_on_scenes(path, threshold=0.02, min_gap=0.3,
                                 out_dir=self.tmp)
        self.assertEqual(res["backend"], "opencv")
        self.assertEqual(res["count"], 3)

    def test_bad_threshold(self):
        with self.assertRaises(MediaEditError):
            CV.split_on_scenes(self.vid, threshold=0, out_dir=self.tmp)


class KenBurnsTests(_Base):
    def _img(self):
        from PIL import Image
        path = os.path.join(self.tmp, "still.png")
        Image.new("RGB", (320, 240), (30, 120, 200)).save(path)
        return path

    def test_kenburns(self):
        res = CV.kenburns(self._img(), duration=2, fps=15,
                          width=320, height=240, out_dir=self.tmp)
        self.assertEqual(res["backend"], "ffmpeg")
        self.assertTrue(os.path.exists(res["output"]))
        n = _count_frames(res["output"])
        self.assertTrue(25 <= n <= 35, n)  # 2s @ 15fps

    def test_kenburns_zoom_out_pan(self):
        res = CV.kenburns(self._img(), duration=1, zoom_from=1.5,
                          zoom_to=1.0, pan="right", fps=10,
                          width=320, height=240, out_dir=self.tmp)
        self.assertTrue(os.path.exists(res["output"]))

    def test_bad_args(self):
        with self.assertRaises(MediaEditError):
            CV.kenburns(self._img(), duration=0, out_dir=self.tmp)
        with self.assertRaises(MediaEditError):
            CV.kenburns(self._img(), pan="diagonal", out_dir=self.tmp)
        with self.assertRaises(MediaEditError):
            CV.kenburns(os.path.join(self.tmp, "nope.png"),
                        out_dir=self.tmp)

    def test_opencv_honestly_unsupported(self):
        with self.assertRaises(MediaEditError):
            CV.kenburns(self._img(), out_dir=self.tmp, backend="opencv")


class SlideshowTests(_Base):
    def _imgs(self, n=3):
        from PIL import Image
        paths = []
        for i in range(n):
            pth = os.path.join(self.tmp, f"ss{i}.png")
            Image.new("RGB", (320, 240),
                      (50 * i, 100, 200 - 40 * i)).save(pth)
            paths.append(pth)
        return paths

    def test_slideshow(self):
        res = CV.slideshow(self._imgs(), duration_each=1,
                           transition="fade", transition_duration=0.2,
                           fps=15, width=320, height=240, out_dir=self.tmp)
        self.assertEqual(res["backend"], "ffmpeg")
        self.assertTrue(os.path.exists(res["output"]))
        # 3x1s - 2x0.2s = 2.6s @15fps ≈ 39 frames
        n = _count_frames(res["output"])
        self.assertTrue(30 <= n <= 45, n)
        self.assertAlmostEqual(res["duration"], 2.6, delta=0.3)

    def test_single_image(self):
        res = CV.slideshow(self._imgs(1), duration_each=1, fps=10,
                           width=320, height=240, out_dir=self.tmp)
        self.assertTrue(os.path.exists(res["output"]))

    def test_bad_args(self):
        with self.assertRaises(MediaEditError):
            CV.slideshow([], out_dir=self.tmp)
        with self.assertRaises(MediaEditError):
            CV.slideshow(self._imgs(), transition_duration=5,
                         out_dir=self.tmp)


class ChromaKeyTests(_Base):
    def _green_video(self):
        # green screen with a red square "subject"
        frames = []
        for i in range(30):
            img = np.full((H, W, 3), (0, 255, 0), np.uint8)
            cv2.rectangle(img, (50, 40), (110, 80), (0, 0, 255), -1)
            frames.append(img)
        path = os.path.join(self.tmp, "green.mp4")
        _write_video(path, frames)
        return path

    def _bg_image(self):
        from PIL import Image
        pth = os.path.join(self.tmp, "bg.png")
        Image.new("RGB", (W, H), (200, 200, 50)).save(pth)
        return pth

    def test_chroma_key(self):
        res = CV.chroma_key(self._green_video(), self._bg_image(),
                            out_dir=self.tmp)
        self.assertEqual(res["backend"], "ffmpeg")
        self.assertTrue(os.path.exists(res["output"]))
        # background shows through: corners should be yellowish now
        frame, _ = CV.grab_frame_at(res["output"], 0.5)
        corner = frame[5, 5].mean()
        self.assertGreater(frame[5, 5][1], 100)  # green channel from bg

    def test_bad_color_params(self):
        with self.assertRaises(MediaEditError):
            CV.chroma_key(self._green_video(), self._bg_image(),
                          similarity=2.0, out_dir=self.tmp)


class PipTests(_Base):
    def test_pip(self):
        res = CV.pip(self.vid, self.vid, position="top-left",
                     scale=0.3, out_dir=self.tmp)
        self.assertEqual(res["backend"], "ffmpeg")
        self.assertTrue(os.path.exists(res["output"]))
        self.assertEqual(_count_frames(res["output"]), 60)

    def test_bad_args(self):
        with self.assertRaises(MediaEditError):
            CV.pip(self.vid, self.vid, position="middle",
                   out_dir=self.tmp)
        with self.assertRaises(MediaEditError):
            CV.pip(self.vid, self.vid, scale=2.0, out_dir=self.tmp)


class FreezeTests(_Base):
    def test_freeze(self):
        res = CV.freeze_frame(self.vid, timestamp=1.0, duration=2,
                              out_dir=self.tmp)
        self.assertTrue(os.path.exists(res["output"]))
        n = _count_frames(res["output"])
        self.assertTrue(55 <= n <= 65, n)  # 2s @30fps
        # all frames identical (it's a freeze)
        f1, _ = CV.grab_frame_at(res["output"], 0.2)
        f2, _ = CV.grab_frame_at(res["output"], 1.5)
        self.assertEqual(int(np.abs(f1.astype(int) - f2.astype(int)).sum()),
                         0)

    def test_bad_duration(self):
        with self.assertRaises(MediaEditError):
            CV.freeze_frame(self.vid, duration=0, out_dir=self.tmp)


class DenoiseTests(_Base):
    def test_ffmpeg_primary(self):
        res = CV.denoise_video(self.vid, strength=1.0, out_dir=self.tmp)
        self.assertEqual(res["backend"], "ffmpeg")
        self.assertFalse(res["audio_dropped"])
        self.assertTrue(os.path.exists(res["output"]))
        self.assertEqual(_count_frames(res["output"]), 60)

    def test_opencv_fallback(self):
        vid = _gradient_video(os.path.join(self.tmp, "dn.mp4"), n=15)
        res = CV.denoise_video(vid, strength=0.5, out_dir=self.tmp,
                               backend="opencv")
        self.assertEqual(res["backend"], "opencv")
        self.assertTrue(res["audio_dropped"])
        self.assertEqual(res["frames"], 15)

    def test_bad_strength(self):
        with self.assertRaises(MediaEditError):
            CV.denoise_video(self.vid, strength=3, out_dir=self.tmp)


class TimelapseDeflickerTests(_Base):
    def test_deflicker(self):
        res = CV.create_timelapse(self.vid, factor=2, deflicker=True,
                                  out_dir=self.tmp)
        self.assertEqual(res["backend"], "ffmpeg")
        self.assertTrue(res["deflicker"])
        self.assertTrue(os.path.exists(res["output"]))

    def test_deflicker_needs_ffmpeg(self):
        with self.assertRaises(MediaEditError):
            CV.create_timelapse(self.vid, factor=2, deflicker=True,
                                out_dir=self.tmp, backend="opencv")


class CliWiringTests(_Base):
    def _parse(self, argv):
        from nomorals.cmdline.parser import _parser
        return _parser().parse_args(argv)

    def test_parser_accepts_new_subcommands(self):
        for argv, action in [
            (["studio", "vframes", "f.mp4", "--count", "3"], "vframes"),
            (["studio", "vfilter", "f.mp4", "edges"], "vfilter"),
            (["studio", "vtimelapse", "f.mp4", "--factor", "8"],
             "vtimelapse"),
            (["studio", "vstabilize", "f.mp4"], "vstabilize"),
            (["studio", "vhighlights", "f.mp4"], "vhighlights"),
            (["studio", "vthumb", "f.mp4", "1:30"], "vthumb"),
            (["studio", "vpreview", "f.mp4", "--cols", "2"], "vpreview"),
            (["studio", "vslowmo", "f.mp4", "--factor", "4"], "vslowmo"),
            (["studio", "vreverse", "f.mp4"], "vreverse"),
            (["studio", "vboomerang", "f.mp4", "--duration", "1"],
             "vboomerang"),
            (["studio", "vblurry", "f.mp4"], "vblurry"),
            (["studio", "vheatmap", "f.mp4"], "vheatmap"),
            (["studio", "vsplit", "f.mp4"], "vsplit"),
            (["studio", "vkenburns", "i.png"], "vkenburns"),
            (["studio", "vslideshow", "a.png", "b.png"], "vslideshow"),
            (["studio", "vchroma", "f.mp4", "bg.png"], "vchroma"),
            (["studio", "vpip", "a.mp4", "b.mp4"], "vpip"),
            (["studio", "vfreeze", "f.mp4", "1.5"], "vfreeze"),
            (["studio", "vdenoise", "f.mp4"], "vdenoise"),
        ]:
            args = self._parse(argv)
            self.assertEqual(args.studio_action, action, argv)
            self.assertEqual(args.backend, "auto", argv)

    def test_parser_backend_flag(self):
        args = self._parse(["studio", "vfilter", "f.mp4", "blur",
                            "--backend", "ffmpeg"])
        self.assertEqual(args.backend, "ffmpeg")

    def test_handler_vframes(self):
        from nomorals.cmdline.commands.media import _cmd_studio_cv_video
        args = argparse.Namespace(studio_action="vframes", file=self.vid,
                                  timestamps=None, interval=None, count=3,
                                  width=None, out_dir=self.tmp,
                                  backend="auto")
        self.assertEqual(_cmd_studio_cv_video(args, False), 0)

    def test_handler_vfilter_json(self):
        from nomorals.cmdline.commands.media import _cmd_studio_cv_video
        args = argparse.Namespace(studio_action="vfilter", file=self.vid,
                                  filter="invert", strength=1.0,
                                  out_dir=self.tmp, backend="auto")
        self.assertEqual(_cmd_studio_cv_video(args, True), 0)

    def test_handler_reports_clean_error(self):
        from nomorals.cmdline.commands.media import _cmd_studio_cv_video
        args = argparse.Namespace(studio_action="vthumb",
                                  file=os.path.join(self.tmp, "nope.mp4"),
                                  timestamp=0, width=640,
                                  out_dir=self.tmp, backend="auto")
        self.assertEqual(_cmd_studio_cv_video(args, False), 1)

    def test_handler_vreverse(self):
        from nomorals.cmdline.commands.media import _cmd_studio_cv_video
        args = argparse.Namespace(studio_action="vreverse", file=self.vid,
                                  out_dir=self.tmp, backend="auto")
        self.assertEqual(_cmd_studio_cv_video(args, False), 0)

    def test_handler_vsplit_json(self):
        from nomorals.cmdline.commands.media import _cmd_studio_cv_video
        args = argparse.Namespace(studio_action="vsplit", file=self.vid,
                                  threshold=0.02, min_gap=0.5, min_scene=0.5,
                                  out_dir=self.tmp, backend="auto")
        self.assertEqual(_cmd_studio_cv_video(args, True), 0)


class PythonFallbackTests(_Base):
    """Zero-mandatory-dependency tier: cv2 absent, stdlib + ffmpeg only."""

    def _no_cv2_mod(self):
        return unittest.mock.patch.object(CV, "_have_cv2",
                                          return_value=False)

    def test_highlights_python(self):
        vid = _scene_change_video(os.path.join(self.tmp, "cut.mp4"))
        with self._no_cv2_mod():
            res = CV.frame_diff_highlights(
                vid, threshold=0.03, min_gap=0.2, out_dir=self.tmp,
                backend="auto")
        self.assertEqual(res["backend"], "python")
        self.assertEqual(res["count"], 1)
        ev = res["events"][0]
        self.assertAlmostEqual(ev["t"], 1.0, places=1)
        self.assertGreater(ev["score"], 0.03)
        self.assertTrue(os.path.exists(ev["thumb"]))

    def test_blurry_python(self):
        with self._no_cv2_mod():
            res = CV.find_blurry_frames(
                self.vid, out_dir=self.tmp, backend="auto")
        self.assertEqual(res["backend"], "python")
        self.assertEqual(res["metric"], "gradient-energy")
        self.assertEqual(res["frames_scanned"], 60)
        self.assertIn("blurriest", res)

    def test_heatmap_python(self):
        vid = _scene_change_video(os.path.join(self.tmp, "cut.mp4"))
        with self._no_cv2_mod():
            res = CV.motion_heatmap(vid, out_dir=self.tmp, backend="auto")
        self.assertEqual(res["backend"], "python")
        with open(res["heatmap"], "rb") as fh:
            self.assertEqual(fh.read(8), b"\x89PNG\r\n\x1a\n")
        self.assertAlmostEqual(res["peak_motion"]["t"], 1.0, places=1)

    def test_slowmo_python(self):
        with self._no_cv2_mod():
            res = CV.slow_motion(self.vid, factor=2, out_dir=self.tmp,
                                 backend="python")
        self.assertEqual(res["backend"], "python")
        self.assertEqual(res["method"], "blend")
        self.assertEqual(res["frames"], 60 + 59)  # n + (n-1) blends
        self.assertTrue(res["audio_dropped"])
        self.assertTrue(os.path.exists(res["output"]))
        self.assertEqual(_count_frames(res["output"]), 119)

    def test_python_needs_ffmpeg_binary(self):
        with self._no_cv2_mod(), \
                unittest.mock.patch.object(CV, "_ffmpeg_bin",
                                           return_value=None):
            with self.assertRaises(MediaEditError) as ctx:
                CV.frame_diff_highlights(
                    self.vid, out_dir=self.tmp, backend="auto")
            self.assertIn("ffmpeg", str(ctx.exception).lower())


class DeshakeTests(_Base):
    def test_deshake_tier(self):
        # libvidstab hidden -> built-in deshake, still ffmpeg, audio kept
        real = CV._ffmpeg_has_filter

        def fake(*names):
            if set(names) == {"vidstabdetect", "vidstabtransform"}:
                return False
            return real(*names)

        with unittest.mock.patch.object(CV, "_ffmpeg_has_filter",
                                        side_effect=fake):
            res = CV.stabilize_basic(self.vid, out_dir=self.tmp,
                                     backend="auto")
        self.assertEqual(res["backend"], "ffmpeg")
        self.assertEqual(res["method"], "deshake")
        self.assertFalse(res["audio_dropped"])
        self.assertTrue(os.path.exists(res["output"]))


class ResolveBackendTests(unittest.TestCase):
    def test_auto_picks_python_without_cv2(self):
        with unittest.mock.patch.object(CV, "_have_cv2",
                                        return_value=False):
            self.assertEqual(
                CV._resolve_backend("auto", ("opencv", "python")), "python")

    def test_explicit_python(self):
        self.assertEqual(CV._resolve_backend("python", ("python",)),
                         "python")

    def test_explicit_python_without_ffmpeg_fails(self):
        with unittest.mock.patch.object(CV, "_ffmpeg_bin",
                                        return_value=None), \
                self.assertRaises(MediaEditError):
            CV._resolve_backend("python", ("python",))

    def test_invalid_backend_name(self):
        with self.assertRaises(MediaEditError):
            CV._resolve_backend("nope", ("opencv", "python"))

    def test_backend_choices_include_python(self):
        self.assertIn("python", CV._BACKENDS)


class PngWriterTests(unittest.TestCase):
    def test_write_png_roundtrip(self):
        tmp = tempfile.mkdtemp(prefix="png-test-")
        try:
            path = os.path.join(tmp, "t.png")
            w, h = 8, 6
            rgb = bytes(c for y in range(h) for x in range(w)
                        for c in (x * 32 % 256, y * 40 % 256, 128))
            CV._write_png(path, w, h, rgb)
            with open(path, "rb") as fh:
                data = fh.read()
            self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))
            # IHDR carries width/height big-endian right after the header
            self.assertEqual(int.from_bytes(data[16:20], "big"), w)
            self.assertEqual(int.from_bytes(data[20:24], "big"), h)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
