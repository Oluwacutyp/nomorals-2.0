"""Media editing tests (Prompt 15): universal image + video editing.

Covers:
- image engine ops (transform/convert/enhance/annotate/composite/probe)
- deterministic intent parsing (20 intents) + dry-run + ambiguous fallback
- tool layer: sandboxing, never-overwrite, EXIF strip, mocked vision locate
- video background jobs: trim (stream-copy), audio extraction, probing
- graceful degradation: missing ffmpeg, corrupt/oversized inputs
- `nm media` CLI surface
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image

from nomorals.media_edit import images
from nomorals.media_edit.images import MediaEditError
from nomorals.media_edit import intent as intent_mod
from nomorals.media_edit.intent import (
    AmbiguousInstructionError, parse_instruction)
from nomorals.media_edit import videos
from nomorals.tools.registry import ToolRegistry


def _make_image(path, size=(1600, 1200), color=(60, 120, 200)):
    img = Image.new("RGB", size, color)
    img.save(path, quality=90)
    return path


def _make_clip(path, duration=10, size="320x240"):
    if not shutil.which("ffmpeg"):
        raise unittest.SkipTest("ffmpeg not available")
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc=duration={duration}:size={size}:rate=30",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(path), "-y"],
        capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"fixture clip failed: {proc.stderr[:300]}")
    return path


class ImageEngineTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="mimg_"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.src = _make_image(self.root / "photo.jpg")

    def test_probe(self):
        info = images.image_probe(self.src)
        self.assertEqual((info["width"], info["height"]), (1600, 1200))
        self.assertEqual(info["format"], "JPEG")
        self.assertEqual(info["kind"], "image")

    def test_resize_modes(self):
        img = images.load_image(self.src)
        try:
            fit = images.op_resize(img, 800, 800, mode="fit")
            self.assertEqual(fit.size, (800, 600))
            exact = images.op_resize(img, 800, 800, mode="exact")
            self.assertEqual(exact.size, (800, 800))
            fill = images.op_resize(img, 800, 800, mode="fill")
            self.assertEqual(fill.size, (800, 800))
        finally:
            img.close()

    def test_crop_aspect_anchors(self):
        img = images.load_image(self.src)
        try:
            for anchor in ("center", "top", "bottom", "left", "right", "smart"):
                out = images.op_crop(img, aspect="1:1", anchor=anchor)
                self.assertEqual(out.size, (1200, 1200), anchor)
            wide = images.op_crop(img, aspect="16:9", anchor="center")
            self.assertEqual(wide.size, (1600, 900))
        finally:
            img.close()

    def test_enhance_grayscale(self):
        img = images.load_image(self.src)
        try:
            out = images.op_enhance(img, grayscale=True)
            self.assertEqual(out.mode, "L")
            bright = images.op_enhance(img, brightness=1.5)
            self.assertGreater(
                sum(bright.convert("L").getdata()),
                sum(img.convert("L").getdata()))
        finally:
            img.close()

    def test_annotate(self):
        img = images.load_image(self.src)
        try:
            t = images.op_annotate_text(img, "hello", position="top")
            self.assertEqual(t.size, img.size)
            for shape in ("rectangle", "circle", "arrow"):
                s = images.op_annotate_shape(img, shape)
                self.assertEqual(s.size, img.size)
        finally:
            img.close()

    def test_composites(self):
        a = images.load_image(self.src)
        b = images.load_image(self.src)
        try:
            stack = images.op_stack([a, b], direction="horizontal")
            self.assertEqual(stack.size, (3200, 1200))
            grid = images.op_grid([a, b, a, b], cols=2, cell=100)
            self.assertEqual(grid.size, (204, 204))
            meme = images.op_meme(a, top="top text", bottom="bottom text")
            self.assertEqual(meme.size, a.size)
        finally:
            a.close()
            b.close()

    def test_edit_image_never_overwrites_and_strips_exif(self):
        before = self.src.stat().st_mtime_ns
        content_before = self.src.read_bytes()
        res = images.edit_image(
            self.src, [{"op": "resize", "width": 100, "height": 100,
                        "mode": "exact"}],
            out_dir=self.root / "edited", suffix="small")
        out = Path(res["output"])
        self.assertTrue(out.exists())
        self.assertNotEqual(out.resolve(), self.src.resolve())
        # original byte-identical
        self.assertEqual(self.src.read_bytes(), content_before)
        self.assertEqual(self.src.stat().st_mtime_ns, before)
        info = images.image_probe(out)
        self.assertEqual((info["width"], info["height"]), (100, 100))
        self.assertEqual(info["exif"]["tag_count"], 0)

    def test_convert_op(self):
        res = images.edit_image(
            self.src, [{"op": "convert", "format": "WEBP"}],
            out_dir=self.root / "edited", suffix="webpd")
        self.assertTrue(str(res["output"]).endswith(".webp"))
        self.assertEqual(images.image_probe(res["output"])["format"], "WEBP")

    def test_validate_ops_rejects_unknown(self):
        with self.assertRaises(MediaEditError):
            images.validate_ops([{"op": "nuke"}])
        with self.assertRaises(MediaEditError):
            images.validate_ops([])

    def test_batch_edit(self):
        _make_image(self.root / "b.jpg", color=(10, 20, 30))
        results = images.batch_edit(
            self.root,
            [{"op": "thumbnail", "size": 64}],
            pattern="*.jpg", out_dir=self.root / "edited")
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIn("output", r)

    def test_watermark(self):
        logo = _make_image(self.root / "logo.png", size=(200, 100),
                           color=(255, 255, 0))
        res = images.edit_image(
            self.src, [{"op": "watermark", "logo": str(logo)}],
            out_dir=self.root / "edited", suffix="wmed")
        self.assertTrue(Path(res["output"]).exists())


class IntentTests(unittest.TestCase):
    def test_image_intents(self):
        cases = [
            ("make it square for instagram", "image",
             ["crop", "resize"]),
            ("resize to 1920x1080", "image", ["resize"]),
            ("resize to 1080", "image", ["resize"]),
            ("convert to webp", "image", ["convert"]),
            ("watermark with logo.png", "image", ["watermark"]),
            ("rotate 90", "image", ["rotate"]),
            ("rotate left", "image", ["rotate"]),
            ("grayscale", "image", ["enhance"]),
            ("brighten", "image", ["enhance"]),
            ("circle the login button", "image", ["annotate_shape"]),
            ('add text "hello world"', "image", ["annotate_text"]),
            ("make a thumbnail", "image", ["thumbnail"]),
            ("flip horizontal", "image", ["flip"]),
            ("sharpen", "image", ["enhance"]),
            ("crop to 16:9", "image", ["crop"]),
            ("meme top: hi bottom: bye", "image", ["meme"]),
            ("auto contrast", "image", ["enhance"]),
        ]
        for text, kind, ops in cases:
            with self.subTest(text=text):
                parsed = parse_instruction(text, kind=kind)
                self.assertEqual(parsed.kind, "image")
                self.assertEqual([o["op"] for o in parsed.ops], ops)

    def test_video_intents(self):
        cases = [
            ("trim the first 30 seconds",
             {"video_op": "trim", "start": 0, "end": 30.0}),
            ("cut the first 10s", {"video_op": "trim", "start": 0,
                                   "end": 10.0}),
            ("trim 0:30-1:00", {"video_op": "trim", "start": "0:30",
                                "end": "1:00"}),
            ("extract the audio", {"video_op": "extract_audio"}),
            ("make a gif", {"video_op": "make_gif"}),
            ("extract 5 thumbnails", {"video_op": "extract_frames",
                                      "count": 5}),
            ("convert to mp4", {"video_op": "transcode", "ext": ".mp4"}),
            ("resize to 720p", {"video_op": "transcode", "height": 720}),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                parsed = parse_instruction(text, kind="video")
                self.assertEqual(parsed.kind, "video")
                for k, v in expected.items():
                    self.assertEqual(parsed.action.get(k), v, text)

    def test_square_instagram_is_1080(self):
        parsed = parse_instruction("make it square for instagram",
                                   kind="image")
        resize = [o for o in parsed.ops if o["op"] == "resize"][0]
        self.assertEqual((resize["width"], resize["height"]), (1080, 1080))

    def test_ambiguous_raises_with_hints(self):
        with self.assertRaises(AmbiguousInstructionError) as ctx:
            parse_instruction("do something magical", kind="image")
        self.assertIn("square", str(ctx.exception))

    def test_dry_run_describe(self):
        parsed = parse_instruction("trim the first 30 seconds", kind="video")
        text = intent_mod.describe_plan(parsed)
        self.assertIn("trim", text)
        self.assertIn("background job", text)


class ToolLayerTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="mtool_"))
        self.addCleanup(shutil.rmtree, self.root, True)
        _make_image(self.root / "photo.jpg")
        context = SimpleNamespace(
            settings=SimpleNamespace(workspace_dir=str(self.root)),
            db=None, router=None)
        self.registry = ToolRegistry(context)
        self.registry.register_builtins()

    def _call(self, name, **kwargs):
        outcome = self.registry.call(name, **kwargs)
        self.assertTrue(outcome.ok, f"{name} failed: {outcome.error}")
        return outcome.value

    def test_tools_registered(self):
        names = set(self.registry.names())
        for tool in ("media_edit", "media_edit_video", "media_edit_probe",
                     "media_job_status", "media_jobs", "media_convert"):
            self.assertIn(tool, names)

    def test_media_edit_square_instagram(self):
        res = self._call("media_edit", image_path="photo.jpg",
                         instruction="make it square for instagram")
        out = Path(res["output"])
        self.assertTrue(out.exists())
        info = images.image_probe(out)
        self.assertEqual((info["width"], info["height"]), (1080, 1080))
        self.assertEqual(info["exif"]["tag_count"], 0)
        # original untouched
        orig = images.image_probe(self.root / "photo.jpg")
        self.assertEqual((orig["width"], orig["height"]), (1600, 1200))

    def test_media_edit_dry_run(self):
        res = self._call("media_edit", image_path="photo.jpg",
                         instruction="rotate 90", dry_run=True)
        self.assertTrue(res["dry_run"])
        self.assertIn("rotate", res["plan"])
        self.assertFalse((self.root / "edited").exists())

    def test_media_edit_explicit_ops(self):
        res = self._call(
            "media_edit", image_path="photo.jpg",
            ops=[{"op": "flip", "direction": "vertical"}],
            suffix="flipped")
        self.assertTrue(Path(res["output"]).exists())

    def test_circle_the_thing_with_mocked_vision(self):
        from nomorals.tools import media_edit as me_tool
        me_tool.register_locator(lambda path, desc: (10, 20, 100, 120)
                                 if desc == "login button" else None)
        self.addCleanup(me_tool.register_locator, None)
        res = self._call("media_edit", image_path="photo.jpg",
                         instruction="circle the login button")
        self.assertTrue(Path(res["output"]).exists())

    def test_circle_without_locator_fails_cleanly(self):
        from nomorals.tools import media_edit as me_tool
        me_tool.register_locator(None)
        outcome = self.registry.call(
            "media_edit", image_path="photo.jpg",
            instruction="circle the login button")
        self.assertFalse(outcome.ok)
        self.assertIn("locator", str(outcome.error).lower())

    def test_sandbox_rejects_escape(self):
        outcome = self.registry.call(
            "media_edit", image_path="/etc/passwd",
            instruction="grayscale")
        self.assertFalse(outcome.ok)

    def test_probe_image(self):
        info = self._call("media_edit_probe", path="photo.jpg")
        self.assertEqual(info["kind"], "image")
        self.assertEqual((info["width"], info["height"]), (1600, 1200))

    def test_convert_image(self):
        res = self._call("media_convert", path="photo.jpg", format="webp")
        self.assertTrue(str(res["output"]).endswith(".webp"))

    def test_missing_ffmpeg_video_fails_image_ok(self):
        # video tools fail with the clear ffmpeg error...
        (self.root / "clip.mp4").write_bytes(b"\x00" * 4096)
        with mock.patch.object(videos, "ffmpeg_path",
                               side_effect=MediaEditError(
                                   videos._FFMPEG_HINT)):
            outcome = self.registry.call(
                "media_edit_video", video_path="clip.mp4",
                instruction="trim the first 3 seconds")
            self.assertTrue(outcome.ok)  # enqueue works; job itself fails
            job_id = outcome.value["job_id"]
            from nomorals.media_edit.jobs import get_manager
            info = get_manager().wait(job_id, timeout=60)
            self.assertEqual(info["status"], "failed")
            self.assertIn("ffmpeg is not installed", info["error"])
        # ...image tools are unaffected
        res = self._call("media_edit", image_path="photo.jpg",
                         instruction="grayscale")
        self.assertTrue(Path(res["output"]).exists())

    def test_oversized_image_rejected(self):
        from nomorals.tools import media_edit as me_tool
        with mock.patch.object(me_tool, "MAX_IMAGE_BYTES", 10):
            outcome = self.registry.call(
                "media_edit", image_path="photo.jpg",
                instruction="grayscale")
            self.assertFalse(outcome.ok)
            self.assertIn("over the", str(outcome.error))


class VideoJobTests(unittest.TestCase):
    def setUp(self):
        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg not available")
        self.root = Path(tempfile.mkdtemp(prefix="mvid_"))
        self.addCleanup(shutil.rmtree, self.root, True)
        _make_clip(self.root / "clip.mp4")
        context = SimpleNamespace(
            settings=SimpleNamespace(workspace_dir=str(self.root)),
            db=None, router=None)
        self.registry = ToolRegistry(context)
        self.registry.register_builtins()

    def _call(self, name, **kwargs):
        outcome = self.registry.call(name, **kwargs)
        self.assertTrue(outcome.ok, f"{name} failed: {outcome.error}")
        return outcome.value

    def test_trim_background_job_stream_copy(self):
        started = time.perf_counter()
        res = self._call("media_edit_video", video_path="clip.mp4",
                         instruction="trim the first 3 seconds")
        job_id = res["job_id"]
        self.assertEqual(res["status"], "queued")
        # poll like a client would
        info = self._call("media_job_status", job_id=job_id)
        self.assertIn(info["status"], ("queued", "running", "done"))
        from nomorals.media_edit.jobs import get_manager
        done = get_manager().wait(job_id, timeout=120)
        elapsed = time.perf_counter() - started
        self.assertEqual(done["status"], "done", done.get("error"))
        self.assertEqual(done["result"]["mode"], "stream-copy")
        probe = videos.video_probe(done["output_ref"])
        self.assertAlmostEqual(probe["duration"], 3.0, delta=0.6)
        self.assertLess(elapsed, 60, "stream copy should be fast")
        # original untouched
        self.assertAlmostEqual(
            videos.video_probe(self.root / "clip.mp4")["duration"],
            10.0, delta=0.5)

    def test_extract_audio(self):
        res = self._call("media_edit_video", video_path="clip.mp4",
                         instruction="extract the audio")
        from nomorals.media_edit.jobs import get_manager
        done = get_manager().wait(res["job_id"], timeout=120)
        self.assertEqual(done["status"], "done", done.get("error"))
        out = Path(done["output_ref"])
        self.assertTrue(out.exists() and out.stat().st_size > 0)
        self.assertTrue(str(out).endswith(".mp3"))

    def test_video_probe(self):
        info = self._call("media_edit_probe", path="clip.mp4")
        self.assertEqual(info["kind"], "video")
        self.assertEqual((info["width"], info["height"]), (320, 240))
        self.assertEqual(info["video_codec"], "h264")
        self.assertAlmostEqual(info["duration"], 10.0, delta=0.5)

    def test_corrupt_video_fails_job_cleanly(self):
        (self.root / "junk.mp4").write_bytes(os.urandom(2048))
        res = self._call("media_edit_video", video_path="junk.mp4",
                         instruction="trim the first 3 seconds")
        from nomorals.media_edit.jobs import get_manager
        done = get_manager().wait(res["job_id"], timeout=120)
        self.assertEqual(done["status"], "failed")
        self.assertTrue(done["error"])

    def test_jobs_list(self):
        res = self._call("media_edit_video", video_path="clip.mp4",
                         instruction="extract the audio")
        from nomorals.media_edit.jobs import get_manager
        # wait so tearDown doesn't delete the fixture mid-job
        done = get_manager().wait(res["job_id"], timeout=120)
        self.assertEqual(done["status"], "done", done.get("error"))
        jobs = self._call("media_jobs", limit=5)
        self.assertTrue(any(j["kind"] == "video" for j in jobs))

    def test_convert_video_queues_job(self):
        res = self._call("media_convert", path="clip.mp4", format="webm")
        self.assertIn("job_id", res)
        from nomorals.media_edit.jobs import get_manager
        done = get_manager().wait(res["job_id"], timeout=180)
        self.assertEqual(done["status"], "done", done.get("error"))
        self.assertTrue(str(done["output_ref"]).endswith(".webm"))


class CLITests(unittest.TestCase):
    """Exercise `nm media` through the real CLI entry point."""

    def setUp(self):
        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg not available")
        self.home = Path(tempfile.mkdtemp(prefix="mcli_"))
        self.addCleanup(shutil.rmtree, self.home, True)
        ws = self.home / ".nomorals" / "workspace"
        ws.mkdir(parents=True)
        _make_image(ws / "photo.jpg")
        _make_clip(ws / "clip.mp4", duration=6)
        self.env = dict(os.environ, HOME=str(self.home),
                        PYTHONPATH="/home/hatch/workspace/devon")

    def _nm(self, *args):
        proc = subprocess.run(
            [sys.executable, "-m", "nomorals", "media", *args],
            capture_output=True, text=True, timeout=180,
            cwd="/home/hatch/workspace/devon", env=self.env)
        return proc

    def test_cli_probe_image(self):
        proc = self._nm("probe", "photo.jpg")
        self.assertEqual(proc.returncode, 0, proc.stderr[-500:])
        self.assertIn("1600x1200", proc.stdout)

    def test_cli_edit_image(self):
        proc = self._nm("edit", "photo.jpg", "make it square for instagram")
        self.assertEqual(proc.returncode, 0, proc.stderr[-500:])
        self.assertIn("wrote", proc.stdout)
        out = list((self.home / ".nomorals" / "workspace"
                    / "edited").glob("photo-*.jpeg"))
        self.assertTrue(out, "edited artifact missing")
        info = images.image_probe(out[0])
        self.assertEqual((info["width"], info["height"]), (1080, 1080))

    def test_cli_edit_dry_run(self):
        proc = self._nm("edit", "photo.jpg", "rotate 90", "--dry-run")
        self.assertEqual(proc.returncode, 0, proc.stderr[-500:])
        self.assertIn("rotate", proc.stdout)

    def test_cli_video_trim_wait(self):
        proc = self._nm("edit", "clip.mp4", "trim the first 2 seconds",
                        "--wait")
        self.assertEqual(proc.returncode, 0, proc.stderr[-800:])
        self.assertIn("done", proc.stdout)


if __name__ == "__main__":
    unittest.main()
