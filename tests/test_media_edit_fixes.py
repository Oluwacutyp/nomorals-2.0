"""Regression tests for the media_edit correctness sweep (Oct 2026).

Covers every behavior implemented/fixed in this sweep:

- ``parse_time``: 'M:SS' is minutes:seconds (was: parsed as hours),
  malformed timestamps fail fast.
- ``compile_video``: speed ramps accept ``start=None``/``end=None``;
  ``timeline_trim`` accepts "MM:SS" strings; still-image segments and
  overlay inputs are reported in ``loop_inputs``; the graph always ends
  in a canonical ``[vout]`` label (``gif-preview`` renders again).
- ``render_video_compiled``: still-image inputs are passed to ffmpeg
  with ``-loop 1`` (slideshows used to starve for frames).
- ``_segment_durations``: non-positive segment speed fails fast.
- ``annotate_shape(locate=...)``: fails fast with a clear error when no
  object locator is wired; works with a registered locator; ``box`` +
  ``locate`` together is an error.
- ``WatermarkSpec.apply`` / ``op_watermark(spec)``.
- ``burn_subtitles``: filter-special chars in the subtitle path escaped.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image, ImageDraw

from nomorals.media_edit import images as images_mod
from nomorals.media_edit.images import (
    MediaEditError,
    WatermarkSpec,
    clear_object_locators,
    register_object_locator,
)
from nomorals.media_edit import studio as studio_mod
from nomorals.media_edit.studio import (
    build_template,
    compile_video,
    render_video_compiled,
)
from nomorals.media_edit import videos as videos_mod
from nomorals.media_edit.videos import parse_time

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _make_image(path, size=(320, 240), color=(60, 120, 200)):
    img = Image.new("RGB", size, color)
    d = ImageDraw.Draw(img)
    d.ellipse([size[0] * 0.6, size[1] * 0.6,
               size[0] * 0.9, size[1] * 0.9], fill=(230, 230, 240))
    img.save(path)
    return Path(path)


class ParseTimeTest(unittest.TestCase):
    def test_colon_forms(self):
        self.assertEqual(parse_time("1:30"), 90.0)
        self.assertEqual(parse_time("0:30"), 30.0)
        self.assertEqual(parse_time("1:00"), 60.0)
        self.assertEqual(parse_time("0:01:30.5"), 90.5)
        self.assertEqual(parse_time("2:00:00"), 7200.0)

    def test_bare_and_suffixed(self):
        self.assertEqual(parse_time("90"), 90.0)
        self.assertEqual(parse_time(90), 90.0)
        self.assertEqual(parse_time("1.5"), 1.5)
        self.assertEqual(parse_time("90s"), 90.0)
        self.assertEqual(parse_time("2m"), 120.0)
        self.assertEqual(parse_time("1500ms"), 1.5)
        # long minutes form stays lenient and predictable
        self.assertEqual(parse_time("90:30"), 90 * 60 + 30)

    def test_rejects_malformed(self):
        for bad in ("1:75", "abc", "1:2:3:4", "-5", "12:34:56:78", ""):
            with self.assertRaises(MediaEditError, msg=bad):
                parse_time(bad)


class CompileVideoFixesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mef-"))
        self.still = _make_image(self.tmp / "still.jpg")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _compile(self, **kw):
        kw.setdefault("work_dir", self.tmp / "work")
        return compile_video(
            [{"path": str(self.still), "kind": "image",
              "duration": 90.0}], **kw)

    def test_speed_ramp_none_bounds(self):
        c = self._compile(
            speed_ramps=[{"factor": 2.0, "start": None, "end": None}])
        self.assertAlmostEqual(c["duration"], 45.0)

    def test_speed_ramp_string_bounds(self):
        c = self._compile(
            speed_ramps=[{"factor": 2.0, "start": "0:30", "end": "1:00"}])
        # 30s @2x = 15s + the remaining 60s
        self.assertAlmostEqual(c["duration"], 75.0)

    def test_timeline_trim_mmss_strings(self):
        c = self._compile(timeline_trim=("0:30", "1:00"))
        self.assertAlmostEqual(c["duration"], 30.0)

    def test_timeline_trim_numeric_still_works(self):
        c = self._compile(timeline_trim=(10, 40))
        self.assertAlmostEqual(c["duration"], 30.0)

    def test_still_segment_loop_inputs(self):
        c = self._compile()
        self.assertEqual(c["loop_inputs"], {0: 90.0})

    def test_canonical_vout_label(self):
        c = self._compile()
        self.assertEqual(c["v_label"], "[vout]")
        self.assertIn("[vout]", c["filter_complex"])
        # maps reference the canonical label, not a preset-specific one
        self.assertIn("[vout]", c["maps"])

    def test_gif_preview_graph_is_self_consistent(self):
        c = self._compile(export="gif-preview")
        self.assertEqual(c["v_label"], "[vout]")
        # every label referenced by the gif path exists in the graph
        self.assertIn(c["v_label"], c["filter_complex"])

    def test_slideshow_template_loops_stills(self):
        still2 = _make_image(self.tmp / "still2.jpg")
        st = build_template("slideshow",
                            images=[str(self.still), str(still2)],
                            duration_each=2.0)
        parts = st._collect_video_ops()
        c = compile_video(
            parts["segments"], transitions=parts["transitions"],
            title=parts["title"], lower_thirds=parts["lower_thirds"],
            export=parts["export"], work_dir=self.tmp / "work2")
        self.assertEqual(c["loop_inputs"][0], 2.0)
        self.assertEqual(c["loop_inputs"][1], 2.0)

    def test_title_and_lower_third_loop_full_timeline(self):
        c = compile_video(
            [{"path": str(self.still), "kind": "image", "duration": 2.0}],
            title={"text": "Hi", "duration": 2.0},
            lower_thirds=[{"text": "Name", "start": 0.5, "duration": 1.0}],
            work_dir=self.tmp / "work3")
        # inputs: [still, title.png, lt0.png]; overlays loop the timeline
        self.assertEqual(c["loop_inputs"][0], 2.0)
        self.assertAlmostEqual(c["loop_inputs"][1], c["duration"])
        self.assertAlmostEqual(c["loop_inputs"][2], c["duration"])

    def test_segment_speed_zero_rejected(self):
        with self.assertRaises(MediaEditError):
            studio_mod._segment_durations(
                [{"path": str(self.still), "kind": "image", "duration": 2.0,
                  "speed": 0}])

    def test_segment_speed_negative_rejected(self):
        with self.assertRaises(MediaEditError):
            studio_mod._segment_durations(
                [{"path": str(self.still), "kind": "image", "duration": 2.0,
                  "speed": -1.5}])


class RenderLoopInputsTest(unittest.TestCase):
    """render_video_compiled must pass -loop for still-image inputs."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mef-render-"))
        self.still = _make_image(self.tmp / "still.jpg")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _compiled(self):
        return compile_video(
            [{"path": str(self.still), "kind": "image", "duration": 2.0}],
            export="gif-preview", work_dir=self.tmp / "work")

    def test_loop_flags_in_ffmpeg_args(self):
        compiled = self._compiled()
        seen = {}

        def fake_run(args, **kw):
            seen["args"] = list(args)
            out = Path(args[-1])
            out.touch()
            return {"seconds": 0.1}

        with mock.patch.object(videos_mod, "run_ffmpeg",
                               side_effect=fake_run):
            render_video_compiled(compiled, self.tmp / "o.gif", timeout=30)
        args = seen["args"]
        idx = args.index(str(self.still))
        # "-loop 1 -framerate 30 -t <dur> -i <still>"
        self.assertEqual(args[idx - 7:idx + 1],
                         ["-loop", "1", "-framerate", "30", "-t", "2.0",
                          "-i", str(self.still)])

    def test_gif_filter_maps_defined_labels(self):
        compiled = self._compiled()
        seen = {}

        def fake_run(args, **kw):
            seen["args"] = list(args)
            Path(args[-1]).touch()
            return {"seconds": 0.1}

        with mock.patch.object(videos_mod, "run_ffmpeg",
                               side_effect=fake_run):
            render_video_compiled(compiled, self.tmp / "o.gif", timeout=30)
        args = seen["args"]
        fc = args[args.index("-filter_complex") + 1]
        # the gif branch must reference the canonical labels, never [vpre]
        self.assertIn("[vout]", fc)
        self.assertNotIn("[vpre]", fc)
        self.assertIn("anullsink", fc)
        self.assertIn("[gout]", args)


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg not installed")
class EndToEndRenderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mef-e2e-"))
        self.s1 = _make_image(self.tmp / "s1.jpg")
        self.s2 = _make_image(self.tmp / "s2.jpg")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_gif_preview_renders(self):
        compiled = compile_video(
            [{"path": str(self.s1), "kind": "image", "duration": 1.5}],
            export="gif-preview", work_dir=self.tmp / "w")
        out = self.tmp / "preview.gif"
        r = render_video_compiled(compiled, out, timeout=120)
        self.assertTrue(out.exists())
        self.assertGreater(r["bytes"], 0)
        self.assertEqual(r["preset"], "gif-preview")
        self.assertEqual(out.read_bytes()[:6], b"GIF89a")

    def test_slideshow_of_stills_has_full_duration(self):
        st = build_template("slideshow",
                            images=[str(self.s1), str(self.s2)],
                            duration_each=1.0, transition="fade",
                            transition_duration=0.5)
        parts = st._collect_video_ops()
        compiled = compile_video(
            parts["segments"], transitions=parts["transitions"],
            export=parts["export"], work_dir=self.tmp / "w2")
        out = self.tmp / "show.mp4"
        render_video_compiled(compiled, out, timeout=180)
        info = videos_mod.video_probe(out)
        # 2x1.0s stills minus one 0.5s xfade overlap = 1.5s
        self.assertAlmostEqual(info["duration"], 1.5, delta=0.4)
        self.assertEqual((info["width"], info["height"]), (1280, 720))


class LocateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mef-locate-"))
        self.img = Image.new("RGB", (200, 150), (40, 80, 160))
        clear_object_locators()

    def tearDown(self):
        clear_object_locators()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_no_locator_fails_fast(self):
        with self.assertRaises(MediaEditError) as ctx:
            images_mod.apply_chain(
                self.img,
                [{"op": "annotate_shape", "shape": "circle",
                  "locate": "the cat"}])
        self.assertIn("locator", str(ctx.exception))

    def test_registered_locator_resolves(self):
        register_object_locator(lambda img, q: (10, 20, 60, 80))
        out = images_mod.apply_chain(
            self.img,
            [{"op": "annotate_shape", "shape": "circle",
              "locate": "the cat"}])
        self.assertEqual(out.size, self.img.size)

    def test_locator_returning_none_fails_fast(self):
        register_object_locator(lambda img, q: None)
        with self.assertRaises(MediaEditError):
            images_mod.apply_chain(
                self.img,
                [{"op": "annotate_shape", "shape": "circle",
                  "locate": "nothing here"}])

    def test_box_and_locate_conflict(self):
        register_object_locator(lambda img, q: (10, 20, 60, 80))
        with self.assertRaises(MediaEditError):
            images_mod.op_annotate_shape(
                self.img, "circle", box=(1, 1, 5, 5), locate="cat")


class WatermarkSpecTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mef-wm-"))
        self.img = Image.new("RGB", (100, 80), (30, 60, 120))
        self.logo_path = self.tmp / "logo.png"
        Image.new("RGBA", (40, 20), (255, 0, 0, 255)).save(self.logo_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_spec_apply_draws_logo(self):
        spec = WatermarkSpec(logo_path=str(self.logo_path))
        out = spec.apply(self.img)
        # 100x80 image, scale 0.15 -> 15x8 logo at bottom-right margin 20:
        # occupies x in [65, 80), y in [52, 60)
        r, g, b = out.convert("RGB").getpixel((72, 56))
        self.assertGreater(r, 150)

    def test_op_accepts_spec(self):
        spec = WatermarkSpec(logo_path=str(self.logo_path),
                             position="top-left")
        out = images_mod.op_watermark(self.img, spec)
        # 15x8 logo at (20, 20)
        r, g, b = out.convert("RGB").getpixel((27, 24))
        self.assertGreater(r, 150)


class BurnSubtitlesEscapeTest(unittest.TestCase):
    def test_special_chars_escaped(self):
        tmp = Path(tempfile.mkdtemp(prefix="mef-sub-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        src = tmp / "v.mp4"
        src.touch()
        sub = tmp / "a:b,c.srt"
        sub.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n")
        seen = {}

        def fake_run(args, **kw):
            seen["args"] = list(args)
            Path(args[-1]).touch()
            return {"seconds": 0.1}

        with mock.patch.object(videos_mod, "run_ffmpeg",
                               side_effect=fake_run), \
             mock.patch.object(videos_mod, "video_probe",
                               return_value={"duration": 2.0}), \
             mock.patch.object(videos_mod, "has_libass",
                               return_value=True):
            videos_mod.burn_subtitles(src, sub, out_dir=str(tmp))
        args = seen["args"]
        vf = args[args.index("-vf") + 1]
        self.assertTrue(vf.startswith("subtitles="))
        # ':' and ',' in the filename must be backslash-escaped for the
        # filter parser
        self.assertIn("\\:", vf)
        self.assertIn("\\,", vf)


if __name__ == "__main__":
    unittest.main()
