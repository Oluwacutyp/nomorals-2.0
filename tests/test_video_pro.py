"""Tests for pro video ops: transition, effect, speed_ramp, ducking, mix_audio.

run_ffmpeg and video_probe are mocked (established pattern from
tests/test_edit_new.py); assertions check the exact filter strings the
ops build.
"""
from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


@contextlib.contextmanager
def _op_env(probe_ret=None):
    """Two real (empty) source files + a pre-created output file, with
    run_ffmpeg, video_probe and _out patched."""
    from nomorals.media_edit import videos as V
    tmp = tempfile.mkdtemp(prefix="videopro-")
    src = os.path.join(tmp, "a.mp4")
    src2 = os.path.join(tmp, "b.mp4")
    music = os.path.join(tmp, "music.mp3")
    for f in (src, src2, music):
        with open(f, "wb") as fh:
            fh.write(b"x" * 64)
    outp = os.path.join(tmp, "out.mp4")
    with open(outp, "wb") as fh:
        fh.write(b"x" * 128)
    probe = probe_ret or {"duration": 10.0, "width": 640, "height": 480,
                          "fps": 30.0, "streams": []}
    with patch.object(V, "run_ffmpeg") as run, \
         patch.object(V, "video_probe", MagicMock(return_value=probe)), \
         patch.object(V, "_out", return_value=Path(outp)):
        yield {"src": src, "src2": src2, "music": music,
               "out": outp, "run": run, "V": V, "tmp": tmp}
    shutil.rmtree(tmp, ignore_errors=True)


def _fc(args):
    """Pull the -filter_complex value out of a run_ffmpeg arg list."""
    return args[args.index("-filter_complex") + 1]


def _vf(args):
    return args[args.index("-vf") + 1]


class TransitionTests(unittest.TestCase):
    def test_xfade_builds_filter(self):
        with _op_env() as env:
            V, run = env["V"], env["run"]
            res = V.transition(env["src"], env["src2"],
                               kind="xfade", duration=1.0, transition="fade")
            self.assertEqual(res["kind"], "xfade")
            self.assertEqual(res["transition"], "fade")
            self.assertEqual(res["duration"], 19.0)  # 10 + 10 - 1
            fc = _fc(run.call_args[0][0])
            self.assertIn("xfade=transition=fade:duration=1.0:offset=9.0", fc)

    def test_xfade_slideleft_and_dissolve(self):
        with _op_env() as env:
            V, run = env["V"], env["run"]
            V.transition(env["src"], env["src2"], transition="slideleft")
            self.assertIn("xfade=transition=slideleft",
                          _fc(run.call_args[0][0]))
            V.transition(env["src"], env["src2"], transition="dissolve")
            self.assertIn("xfade=transition=dissolve",
                          _fc(run.call_args[0][0]))

    def test_xfade_unknown_transition_raises(self):
        with _op_env() as env:
            with self.assertRaises(Exception) as ctx:
                env["V"].transition(env["src"], env["src2"],
                                    transition="melt")
            self.assertIn("fade", str(ctx.exception))  # lists valid ones
            env["run"].assert_not_called()

    def test_xfade_acrossfades_audio(self):
        probe = {"duration": 10.0, "width": 640, "height": 480, "fps": 30.0,
                 "streams": [{"type": "video"}, {"type": "audio"}]}
        with _op_env(probe) as env:
            env["V"].transition(env["src"], env["src2"], duration=2.0)
            fc = _fc(env["run"].call_args[0][0])
            self.assertIn("acrossfade=d=2.0", fc)

    def test_fadeblack_builds_filters(self):
        with _op_env() as env:
            V, run = env["V"], env["run"]
            res = V.transition(env["src"], env["src2"],
                               kind="fadeblack", duration=1.5)
            self.assertEqual(res["kind"], "fadeblack")
            self.assertEqual(res["duration"], 20.0)  # 10 + 10, no overlap
            fc = _fc(run.call_args[0][0])
            self.assertIn("fade=t=out:st=8.5:d=1.5", fc)
            self.assertIn("fade=t=in:st=0:d=1.5", fc)
            self.assertIn("concat=n=2:v=1:a=0", fc)

    def test_rejects_long_duration(self):
        with _op_env() as env:
            with self.assertRaises(Exception):
                env["V"].transition(env["src"], env["src2"], duration=10.0)
            with self.assertRaises(Exception):
                env["V"].transition(env["src"], env["src2"], duration=25.0)
            env["run"].assert_not_called()

    def test_rejects_zero_duration_and_bad_kind(self):
        with _op_env() as env:
            V = env["V"]
            with self.assertRaises(Exception):
                V.transition(env["src"], env["src2"], duration=0)
            with self.assertRaises(Exception):
                V.transition(env["src"], env["src2"], kind="wipe")
            with self.assertRaises(Exception):
                V.transition("/nonexistent-a.mp4", env["src2"])

    def test_rejects_duration_longer_than_shorter_clip(self):
        probe_short = {"duration": 3.0, "streams": []}
        with _op_env(probe_short) as env:
            # duration 5 < both-10 default but >= the 3s clip
            with self.assertRaises(Exception):
                env["V"].transition(env["src"], env["src2"], duration=5.0)


class EffectTests(unittest.TestCase):
    EXPECTED = {"grayscale", "sepia", "vignette", "sharpen",
                "denoise", "vintage", "invert"}

    def test_all_presets_distinct_nonempty(self):
        with _op_env() as env:
            V, run = env["V"], env["run"]
            seen = set()
            for preset in sorted(self.EXPECTED):
                V.effect(env["src"], preset)
                vf = _vf(run.call_args[0][0])
                self.assertTrue(vf, preset)
                seen.add(vf)
            self.assertEqual(len(seen), len(self.EXPECTED),
                             "every preset must map to a distinct -vf")

    def test_effect_result_and_spot_checks(self):
        with _op_env() as env:
            V, run = env["V"], env["run"]
            res = V.effect(env["src"], "sepia")
            self.assertEqual(res["preset"], "sepia")
            self.assertIn("colorchannelmixer", res["filter"])
            V.effect(env["src"], "denoise")
            self.assertIn("hqdn3d", _vf(run.call_args[0][0]))
            V.effect(env["src"], "vintage")
            vf = _vf(run.call_args[0][0])
            self.assertIn("curves=vintage", vf)
            self.assertIn("colorbalance", vf)
            V.effect(env["src"], "invert")
            self.assertEqual(_vf(run.call_args[0][0]), "negate")

    def test_unknown_preset_raises_with_list(self):
        with _op_env() as env:
            with self.assertRaises(Exception) as ctx:
                env["V"].effect(env["src"], "cyberpunk")
            msg = str(ctx.exception)
            for name in ("grayscale", "sepia", "vintage"):
                self.assertIn(name, msg)
            env["run"].assert_not_called()

    def test_effect_missing_source_raises(self):
        with _op_env() as env:
            with self.assertRaises(Exception):
                env["V"].effect("/nonexistent.mp4", "grayscale")


class SpeedRampTests(unittest.TestCase):
    def test_bad_segments_rejected(self):
        with _op_env() as env:
            V = env["V"]
            # overlap
            with self.assertRaises(Exception):
                V.speed_ramp(env["src"], [(0, 6, 1.0), (5, 10, 2.0)])
            # out of range
            with self.assertRaises(Exception):
                V.speed_ramp(env["src"], [(0, 11, 1.0)])
            # zero / negative factor
            with self.assertRaises(Exception):
                V.speed_ramp(env["src"], [(0, 5, 0.0), (5, 10, 1.0)])
            with self.assertRaises(Exception):
                V.speed_ramp(env["src"], [(0, 5, -2.0), (5, 10, 1.0)])
            # gap in the middle
            with self.assertRaises(Exception):
                V.speed_ramp(env["src"], [(0, 4, 1.0), (5, 10, 1.0)])
            # gap at start / end
            with self.assertRaises(Exception):
                V.speed_ramp(env["src"], [(1, 10, 1.0)])
            with self.assertRaises(Exception):
                V.speed_ramp(env["src"], [(0, 9, 1.0)])
            # unsorted
            with self.assertRaises(Exception):
                V.speed_ramp(env["src"], [(5, 10, 1.0), (0, 5, 1.0)])
            # inverted segment
            with self.assertRaises(Exception):
                V.speed_ramp(env["src"], [(5, 5, 1.0)])
            env["run"].assert_not_called()

    def test_honest_trim_speed_concat_path(self):
        with _op_env() as env:
            V = env["V"]
            segs = [(0, 4, 0.5), (4, 6, 2.0), (6, 10, 1.0)]
            trim_outs = [f"{env['tmp']}/seg{i:02d}.mp4" for i in range(3)]
            sped_outs = [f"{env['tmp']}/seg{i:02d}r.mp4" for i in range(3)]
            with patch.object(V, "trim",
                              side_effect=[{"output": o} for o in trim_outs]
                              ) as trim, \
                 patch.object(V, "speed",
                              side_effect=[{"output": o} for o in sped_outs]
                              ) as speed, \
                 patch.object(V, "concat",
                              return_value={"output": env["out"]}
                              ) as concat:
                res = V.speed_ramp(env["src"], segs)
                self.assertEqual(trim.call_count, 3)
                self.assertEqual(speed.call_count, 3)
                # trim got the raw segment bounds...
                self.assertEqual(trim.call_args_list[1][1]["start"], 4)
                self.assertEqual(trim.call_args_list[1][1]["end"], 6)
                # ...and speed got the segment's factor
                self.assertEqual(speed.call_args_list[0][0][1], 0.5)
                self.assertEqual(speed.call_args_list[1][0][1], 2.0)
                self.assertEqual(speed.call_args_list[2][0][1], 1.0)
                # concat joined the sped parts
                concat.assert_called_once()
                self.assertEqual(concat.call_args[0][0], sped_outs)
                self.assertEqual(res["output"], env["out"])
                self.assertEqual(len(res["segments"]), 3)


class DuckingTests(unittest.TestCase):
    PROBE = {"duration": 10.0, "width": 640, "height": 480, "fps": 30.0,
             "streams": [{"type": "video"}, {"type": "audio"}]}

    def test_ducking_builds_sidechaincompress(self):
        with _op_env(self.PROBE) as env:
            V, run = env["V"], env["run"]
            res = V.ducking(env["src"], env["music"])
            self.assertEqual(res["music_db"], -14.0)
            fc = _fc(run.call_args[0][0])
            self.assertIn("sidechaincompress", fc)
            self.assertIn("asplit", fc)
            self.assertIn("volume=-14.0dB", fc)
            args = run.call_args[0][0]
            self.assertIn(env["music"], args)
            self.assertIn("0:v", args)

    def test_ducking_missing_music_raises(self):
        with _op_env(self.PROBE) as env:
            with self.assertRaises(Exception):
                env["V"].ducking(env["src"], "/nonexistent-track.mp3")
            env["run"].assert_not_called()

    def test_ducking_needs_video_audio(self):
        with _op_env({"duration": 10.0, "streams": [{"type": "video"}]}
                     ) as env:
            with self.assertRaises(Exception):
                env["V"].ducking(env["src"], env["music"])


class MixAudioTests(unittest.TestCase):
    PROBE = {"duration": 10.0, "width": 640, "height": 480, "fps": 30.0,
             "streams": [{"type": "video"}, {"type": "audio"}]}

    def test_mix_under_uses_amix(self):
        with _op_env(self.PROBE) as env:
            V, run = env["V"], env["run"]
            res = V.mix_audio(env["src"], env["music"], volume=0.5)
            self.assertFalse(res["replace"])
            fc = _fc(run.call_args[0][0])
            self.assertIn("amix=inputs=2", fc)
            self.assertIn("volume=0.5", fc)

    def test_replace_maps_new_track(self):
        with _op_env(self.PROBE) as env:
            V, run = env["V"], env["run"]
            res = V.mix_audio(env["src"], env["music"], replace=True)
            self.assertTrue(res["replace"])
            fc = _fc(run.call_args[0][0])
            self.assertNotIn("amix", fc)
            self.assertIn("atrim=0:10.0", fc)
            args = run.call_args[0][0]
            self.assertIn("[1:a]", fc)  # the new track feeds [aout]
            self.assertIn("[aout]", args)

    def test_mix_audio_validates(self):
        with _op_env(self.PROBE) as env:
            V = env["V"]
            with self.assertRaises(Exception):
                V.mix_audio(env["src"], "/nonexistent.mp3")
            with self.assertRaises(Exception):
                V.mix_audio(env["src"], env["music"], volume=-1)
            env["run"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
