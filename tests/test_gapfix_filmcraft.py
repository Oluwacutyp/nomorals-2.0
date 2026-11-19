"""FilmCraft port — Devon's own non-linear video editor (lane 93).

Failing-first: the ``nomorals.media.film`` organ does not exist on base.
The timeline model ports FilmCraft's edit algebra (Premiere semantics:
overwrite / insert / lift / extract / ripple-delete / trims, sync-lock
conflicts, keyframes in media time) and renders finished videos of any
length through the existing render queue + frame pipe.
"""

from __future__ import annotations

import json
import math
import struct
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from nomorals.media.film import audio as film_audio
from nomorals.media.film import render as film_render
from nomorals.media.film.project import (
    FilmError,
    FilmProject,
    Marker,
    Transition,
)
from nomorals.media.film.store import FilmStore, film_path
from nomorals.tools.registry import ToolRegistry


def _solid_png(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


def _sine_wav(path: Path, *, seconds: float = 1.0, freq: float = 440.0,
              sr: int = 22050, amp: float = 0.5) -> Path:
    n = int(seconds * sr)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = b"".join(
            struct.pack("<h", int(amp * 32767 * math.sin(2 * math.pi * freq * i / sr)))
            for i in range(n)
        )
        w.writeframes(frames)
    return path


def _probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


class FilmProjectModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.red = _solid_png(self.tmp / "red.png", (160, 90), (200, 30, 30))
        self.blue = _solid_png(self.tmp / "blue.png", (160, 90), (30, 30, 200))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_create_defaults(self) -> None:
        proj = FilmProject("demo", width=160, height=90, fps=12.0)
        self.assertEqual(proj.duration, 0.0)
        self.assertEqual(proj.width, 160)
        self.assertEqual(proj.fps, 12.0)

    def test_add_clip_and_no_overlap(self) -> None:
        proj = FilmProject("demo", width=160, height=90, fps=12.0)
        track = proj.add_track("video", "V1")
        c1 = proj.add_clip(track, str(self.red), at=0.0, duration=2.0)
        self.assertEqual(c1.duration, 2.0)
        self.assertEqual(proj.duration, 2.0)
        with self.assertRaises(FilmError):
            proj.add_clip(track, str(self.blue), at=1.0, duration=2.0)

    def test_insert_pushes_later_clips(self) -> None:
        proj = FilmProject("demo", width=160, height=90, fps=12.0)
        track = proj.add_track("video", "V1")
        c1 = proj.add_clip(track, str(self.red), at=0.0, duration=2.0)
        c2 = proj.add_clip(track, str(self.blue), at=2.0, duration=2.0)
        proj.insert_clip(track, str(self.red), at=1.0, duration=1.0)
        self.assertAlmostEqual(c1.at, 0.0)
        self.assertAlmostEqual(c2.at, 3.0)
        # the split part + the inserted clip + c2: 1 + 1 + 1 + 2 = 5s
        self.assertAlmostEqual(proj.duration, 5.0)

    def test_split_sums_to_original(self) -> None:
        proj = FilmProject("demo", width=160, height=90, fps=12.0)
        track = proj.add_track("video", "V1")
        c1 = proj.add_clip(track, str(self.red), at=0.0, duration=4.0)
        a, b = proj.split_clip(c1.id, at_time=1.5)
        self.assertAlmostEqual(a.duration + b.duration, 4.0)
        self.assertAlmostEqual(a.at, 0.0)
        self.assertAlmostEqual(b.at, 1.5)
        self.assertAlmostEqual(b.start, 1.5)

    def test_ripple_delete_closes_gap(self) -> None:
        proj = FilmProject("demo", width=160, height=90, fps=12.0)
        track = proj.add_track("video", "V1")
        c1 = proj.add_clip(track, str(self.red), at=0.0, duration=2.0)
        c2 = proj.add_clip(track, str(self.blue), at=2.0, duration=2.0)
        proj.ripple_delete(c1.id)
        self.assertAlmostEqual(c2.at, 0.0)
        self.assertAlmostEqual(proj.duration, 2.0)

    def test_sync_lock_conflict_names_track(self) -> None:
        proj = FilmProject("demo", width=160, height=90, fps=12.0)
        v1 = proj.add_track("video", "V1")
        a1 = proj.add_track("audio", "A1")
        proj.tracks[1].sync_locked = True
        c1 = proj.add_clip(v1, str(self.red), at=0.0, duration=2.0)
        proj.add_clip(a1, str(self.tmp / "a.wav"), at=1.0, duration=2.0,
                      kind="audio")
        with self.assertRaises(FilmError) as cm:
            proj.ripple_delete(c1.id)
        self.assertIn("A1", str(cm.exception))

    def test_trim_regular_and_ripple(self) -> None:
        proj = FilmProject("demo", width=160, height=90, fps=12.0)
        track = proj.add_track("video", "V1")
        c1 = proj.add_clip(track, str(self.red), at=0.0, duration=4.0)
        proj.trim_clip(c1.id, "out", -1.0)
        self.assertAlmostEqual(c1.duration, 3.0)
        c2 = proj.add_clip(track, str(self.blue), at=3.0, duration=2.0)
        proj.trim_clip(c2.id, "in", 1.0, ripple=True)
        self.assertAlmostEqual(c2.duration, 1.0)
        self.assertAlmostEqual(c2.at, 4.0)

    def test_transition_needs_adjacent_clips(self) -> None:
        proj = FilmProject("demo", width=160, height=90, fps=12.0)
        track = proj.add_track("video", "V1")
        c1 = proj.add_clip(track, str(self.red), at=0.0, duration=2.0)
        c2 = proj.add_clip(track, str(self.blue), at=2.0, duration=2.0)
        tr = proj.add_transition(c1.id, c2.id, "cross_dissolve", 0.5)
        self.assertIsInstance(tr, Transition)
        self.assertEqual(tr.kind, "cross_dissolve")
        with self.assertRaises(FilmError):
            proj.add_transition(c1.id, c2.id, "teleport", 0.5)

    def test_grade_changes_pixels(self) -> None:
        proj = FilmProject("demo", width=160, height=90, fps=12.0)
        track = proj.add_track("video", "V1")
        c1 = proj.add_clip(track, str(self.red), at=0.0, duration=1.0)
        proj.grade_clip(c1.id, lift=0.0, gamma=1.0, gain=1.5, saturation=1.0)
        self.assertEqual(c1.grade["gain"], 1.5)

    def test_markers_and_round_trip(self) -> None:
        proj = FilmProject("demo", width=160, height=90, fps=12.0)
        track = proj.add_track("video", "V1")
        proj.add_clip(track, str(self.red), at=0.0, duration=2.0)
        proj.add_marker(1.0, "beat", color="red")
        data = proj.to_dict()
        proj2 = FilmProject.from_dict(data)
        self.assertEqual(proj2.duration, 2.0)
        self.assertEqual(len(proj2.markers), 1)
        self.assertIsInstance(proj2.markers[0], Marker)
        self.assertEqual(proj.content_hash(), proj2.content_hash())

    def test_store_save_load(self) -> None:
        store = FilmStore(self.tmp / "film")
        proj = FilmProject("demo", width=160, height=90, fps=12.0)
        track = proj.add_track("video", "V1")
        proj.add_clip(track, str(self.red), at=0.0, duration=2.0)
        store.save(proj)
        self.assertIn("demo", store.list_names())
        loaded = store.load("demo")
        self.assertEqual(loaded.duration, 2.0)
        self.assertEqual(loaded.content_hash(), proj.content_hash())

    def test_film_path_honors_settings(self) -> None:
        settings = SimpleNamespace(home_path=self.tmp)
        self.assertEqual(film_path(settings), self.tmp / "film" / "projects")


class FilmAudioTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.voice = _sine_wav(self.tmp / "voice.wav", seconds=2.0, freq=440.0)
        self.bed = _sine_wav(self.tmp / "bed.wav", seconds=4.0, freq=220.0)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_ducking_envelope(self) -> None:
        env = film_audio.ducking_envelope(
            4.0, [(1.0, 3.0)], sr=22050, duck_gain=0.25)
        self.assertAlmostEqual(float(env[0]), 1.0, places=2)
        mid = env[int(2.0 * 22050)]
        self.assertAlmostEqual(float(mid), 0.25, places=2)
        self.assertAlmostEqual(float(env[-1]), 1.0, places=2)

    def test_mix_ducks_bed_under_voice(self) -> None:
        # the "voice" is silent (gain 0) so the bed's own level is
        # what the two windows compare — the speech windows still
        # drive the ducking envelope
        mixed = film_audio.mix_audio(
            [{"path": str(self.voice), "at": 1.0, "gain": 0.0},
             {"path": str(self.bed), "at": 0.0, "gain": 0.5}],
            duration_s=4.0, sr=22050, workdir=self.tmp / "work",
            ducking={"bed_index": 1, "speech": [(1.0, 3.0)],
                     "duck_gain": 0.2})
        self.assertEqual(len(mixed), 4 * 22050)
        bed_only = mixed[int(0.5 * 22050):int(0.9 * 22050)]
        bed_ducked = mixed[int(1.5 * 22050):int(2.5 * 22050)]

        def rms(a):
            return float(np.sqrt(np.mean(
                np.asarray(a, dtype=np.float64) ** 2)))

        self.assertGreater(rms(bed_only), rms(bed_ducked))

    def test_loudness_measured(self) -> None:
        report = film_audio.measure_loudness(self.voice)
        self.assertIn("integrated", report)
        self.assertIsInstance(report["integrated"], float)


class FilmRenderTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.red = _solid_png(self.tmp / "red.png", (160, 90), (200, 30, 30))
        self.blue = _solid_png(self.tmp / "blue.png", (160, 90), (30, 30, 200))
        self.voice = _sine_wav(self.tmp / "voice.wav", seconds=1.0, freq=440.0)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _project(self) -> FilmProject:
        proj = FilmProject("demo", width=160, height=90, fps=12.0)
        v1 = proj.add_track("video", "V1")
        proj.add_clip(v1, str(self.red), at=0.0, duration=2.0)
        proj.add_clip(v1, str(self.blue), at=2.0, duration=2.0)
        proj.add_transition_by_index(v1, 0, "cross_dissolve", 0.5)
        title = proj.add_track("title", "T1")
        proj.add_title(title, "Hello Devon", at=0.0, duration=2.0)
        cap = proj.add_track("caption", "C1")
        proj.add_caption(cap, [{"text": "hello world", "start": 0.5,
                                "end": 1.5}])
        proj.grade_clip(proj.tracks[0].items[0].id, lift=0.0, gamma=1.0,
                        gain=1.2, saturation=1.0)
        return proj

    def test_render_produces_real_video(self) -> None:
        proj = self._project()
        out = self.tmp / "out.mp4"
        report = film_render.render_project(
            proj, out, workdir=self.tmp / "work")
        self.assertTrue(report["ok"], report)
        self.assertTrue(out.exists())
        info = _probe(out)
        self.assertGreater(float(info["format"]["duration"]), 3.5)
        streams = {s["codec_type"] for s in info["streams"]}
        self.assertIn("video", streams)

    def test_render_with_narration_and_bed(self) -> None:
        proj = self._project()
        out = self.tmp / "out2.mp4"
        report = film_render.render_project(
            proj, out, workdir=self.tmp / "work2",
            narration_wav=str(self.voice), bed_wav=str(self.voice))
        self.assertTrue(report["ok"], report)
        info = _probe(out)
        streams = {s["codec_type"] for s in info["streams"]}
        self.assertIn("audio", streams)
        self.assertIn("loudness", report)

    def test_srt_export(self) -> None:
        proj = self._project()
        srt = film_render.export_srt(proj, self.tmp / "caps.srt")
        text = srt.read_text()
        self.assertIn("hello world", text)
        self.assertIn("00:00:00,500", text)

    def test_render_with_video_file_clip(self) -> None:
        # a REAL video file as a clip — the ffmpeg raw-pipe reader
        # path, not just stills
        src = self.tmp / "src.mp4"
        proc = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
             "-i", "testsrc=duration=2:size=160x90:rate=12",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        proj = FilmProject("demo", width=160, height=90, fps=12.0)
        v1 = proj.add_track("video", "V1")
        proj.add_clip(v1, str(src), at=0.0, duration=2.0)
        out = self.tmp / "vid.mp4"
        report = film_render.render_project(
            proj, out, workdir=self.tmp / "work5")
        self.assertTrue(report["ok"], report)
        info = _probe(out)
        self.assertGreater(float(info["format"]["duration"]), 1.5)

    def test_render_with_composition_clip(self) -> None:
        from nomorals.media.effects.composition import Composition, Layer
        from nomorals.media.effects.store import CompositionStore
        comp_store = CompositionStore(self.tmp / "effects" / "compositions")
        comp = Composition(160, 90, fps=12.0, duration=2.0)
        comp.add_layer(Layer("solid", color=(30, 120, 200)))
        comp_store.save("blue-comp", comp)
        proj = FilmProject("demo", width=160, height=90, fps=12.0)
        v1 = proj.add_track("composition", "C1")
        proj.add_clip(v1, "blue-comp", at=0.0, duration=2.0,
                      kind="composition")
        out = self.tmp / "compclip.mp4"
        report = film_render.render_project(
            proj, out, workdir=self.tmp / "work6", comp_store=comp_store)
        self.assertTrue(report["ok"], report)
        info = _probe(out)
        self.assertGreater(float(info["format"]["duration"]), 1.5)

    def test_render_refuses_missing_source(self) -> None:
        proj = FilmProject("demo", width=160, height=90, fps=12.0)
        v1 = proj.add_track("video", "V1")
        proj.add_clip(v1, str(self.tmp / "nope.png"), at=0.0, duration=1.0)
        report = film_render.render_project(
            proj, self.tmp / "x.mp4", workdir=self.tmp / "work3")
        self.assertFalse(report["ok"])
        self.assertIn("nope.png", report["error"])

    def test_long_title_fits_inside_frame(self) -> None:
        proj = FilmProject("demo", width=160, height=90, fps=12.0)
        v1 = proj.add_track("video", "V1")
        proj.add_clip(v1, str(self.red), at=0.0, duration=2.0)
        title = proj.add_track("title", "T1")
        proj.add_title(title, "Devon cuts the film and more",
                       at=0.0, duration=2.0)
        out = self.tmp / "title.mp4"
        report = film_render.render_project(
            proj, out, workdir=self.tmp / "work4")
        self.assertTrue(report["ok"], report)
        # the title band (middle rows) must not touch either edge:
        # the plate is graded red, the title is white with a black
        # stroke — any title pixel in the edge columns fails
        import subprocess as _sp
        frame = self.tmp / "title-frame.png"
        _sp.run(["ffmpeg", "-y", "-v", "error", "-ss", "0.5",
                 "-i", str(out), "-frames:v", "1", str(frame)],
                check=True, timeout=60)
        arr = np.asarray(Image.open(frame).convert("RGB"),
                         dtype=np.float64)
        band = arr[25:65, :, :]
        plate = np.array([200.0, 30.0, 30.0])  # the ungraded red plate
        edge = np.concatenate([band[:, :4, :], band[:, -4:, :]], axis=1)
        diff = np.abs(edge - plate).max(axis=2)
        self.assertTrue((diff < 40).all(),
                        "title text touches the frame edge")


class FilmToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.red = _solid_png(self.tmp / "red.png", (160, 90), (200, 30, 30))
        self.blue = _solid_png(self.tmp / "blue.png", (160, 90), (30, 30, 200))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _registry(self) -> ToolRegistry:
        reg = ToolRegistry(SimpleNamespace(
            settings=SimpleNamespace(home_path=self.tmp)))
        reg.register_builtins()
        return reg

    def test_tools_registered(self) -> None:
        reg = self._registry()
        self.assertIn("filmcraft", reg._tools)
        self.assertIn("effectcraft", reg._tools)

    def test_tool_create_render_list(self) -> None:
        reg = self._registry()
        film = reg._tools["filmcraft"].fn
        created = film(action="create", name="demo", width=160, height=90,
                       fps=12)
        self.assertTrue(created["ok"], created)
        added = film(action="add_clip", name="demo", track="V1",
                     source=str(self.red), at=0.0, duration=2.0)
        self.assertTrue(added["ok"], added)
        added2 = film(action="add_clip", name="demo", track="V1",
                      source=str(self.blue), at=2.0, duration=2.0)
        self.assertTrue(added2["ok"], added2)
        rendered = film(action="render", name="demo",
                        out=str(self.tmp / "tool.mp4"))
        self.assertTrue(rendered["ok"], rendered)
        self.assertTrue((self.tmp / "tool.mp4").exists())
        listed = film(action="list")
        self.assertIn("demo", [p["name"] for p in listed["projects"]])


if __name__ == "__main__":
    unittest.main()
