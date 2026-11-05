"""Tests for voice model-file ingest validation (Wave 3b).

The trust boundary: files are sniffed from magic bytes, structurally
parsed, and hashed BEFORE anything ingests them; malformed files are
refused with a coded ``ModelFileError`` (``code="voice.model_file"``,
``details["reason"]`` from the closed taxonomy) — never a crash,
never a silent bad load. Fixtures are generated in the test (real
WAVs written with the stdlib wave module, truncations/corruptions
derived from them), in a disk-backed temp dir.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomorals.core.errors import NoMoralsError
from nomorals.voice.model_files import (
    IngestPolicy,
    ModelFileError,
    sniff_format,
    validate_model_file,
    validate_reference_audio,
)
from nomorals.voice.tts import VoiceLibrary


def _write_wav(path: str, seconds: float = 2.0, rate: int = 16000) -> str:
    frames = int(seconds * rate)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(struct.pack("<%dh" % frames,
                                    *([100] * frames)))
    return path


class _TmpCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def path(self, name: str) -> str:
        return os.path.join(self.tmp, name)


class TestSniff(unittest.TestCase):
    def test_magic_bytes_not_names(self):
        self.assertEqual(sniff_format(b"RIFF\x00\x00\x00\x00WAVE"),
                         "wav")
        self.assertEqual(sniff_format(b"fLaC\x00\x00"), "flac")
        self.assertEqual(sniff_format(b"OggS\x00\x02"), "ogg")
        self.assertEqual(sniff_format(b"ID3\x03\x00"), "mp3")
        self.assertEqual(sniff_format(b"\x00\x00\x00\x18ftypisom"),
                         "video")
        self.assertEqual(sniff_format(b"\x1a\x45\xdf\xa3\x93"), "video")
        self.assertEqual(
            sniff_format(b"\x08\x07\x12\x03abc", suffix=".onnx"),
            "onnx")
        # a bare 0x08 blob without the model suffix stays unknown
        self.assertEqual(sniff_format(b"\x08\x07\x12\x03abc"),
                         "unknown")
        self.assertEqual(sniff_format(b"plain text here"), "unknown")


class TestReferenceAudio(_TmpCase):
    def test_valid_wav_measured(self):
        path = _write_wav(self.path("clip.wav"), seconds=2.0)
        info = validate_reference_audio(path)
        self.assertEqual(info.kind, "reference_audio")
        self.assertEqual(info.format, "wav")
        self.assertEqual(info.sample_rate, 16000)
        self.assertEqual(info.channels, 1)
        self.assertAlmostEqual(info.seconds, 2.0, places=2)
        self.assertEqual(len(info.sha256), 64)
        self.assertEqual(info.size_bytes, os.path.getsize(path))

    def test_missing_and_empty(self):
        with self.assertRaises(ModelFileError) as ctx:
            validate_reference_audio(self.path("nope.wav"))
        self.assertEqual(ctx.exception.details["reason"],
                         "missing_file")
        empty = self.path("empty.wav")
        open(empty, "wb").close()
        with self.assertRaises(ModelFileError) as ctx:
            validate_reference_audio(empty)
        self.assertEqual(ctx.exception.details["reason"], "empty_file")

    def test_video_file_refused_at_the_door(self):
        path = self.path("sneaky.wav")      # video bytes, .wav name
        with open(path, "wb") as fh:
            fh.write(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64)
        with self.assertRaises(ModelFileError) as ctx:
            validate_reference_audio(path)
        self.assertEqual(ctx.exception.details["reason"],
                         "unsupported_format")
        self.assertEqual(ctx.exception.code, "voice.model_file")
        self.assertIsInstance(ctx.exception, NoMoralsError)
        self.assertIsInstance(ctx.exception, ValueError)

    def test_truncated_payload_refused(self):
        path = _write_wav(self.path("cut.wav"), seconds=2.0)
        with open(path, "r+b") as fh:
            fh.truncate(os.path.getsize(path) // 2)
        with self.assertRaises(ModelFileError) as ctx:
            validate_reference_audio(path)
        self.assertEqual(ctx.exception.details["reason"],
                         "truncated_payload")

    def test_corrupt_header_refused(self):
        path = self.path("garbage.wav")
        with open(path, "wb") as fh:
            fh.write(b"RIFF\x24\x00\x00\x00WAVE" + b"\x00" * 8)
        with self.assertRaises(ModelFileError) as ctx:
            validate_reference_audio(path)
        self.assertEqual(ctx.exception.details["reason"],
                         "corrupt_header")

    def test_duration_bounds_and_size_ceiling(self):
        short = _write_wav(self.path("short.wav"), seconds=0.2)
        # Default policy: duration is measured and recorded, not a
        # refusal — clip length is a quality axis (probe warnings,
        # voice-print verdicts), not the structural gate's business.
        self.assertAlmostEqual(
            validate_reference_audio(short).seconds, 0.2, places=2)
        # A caller-set duration bound is enforced as data.
        bounded = IngestPolicy(min_seconds=1.0, max_seconds=900.0)
        with self.assertRaises(ModelFileError) as ctx:
            validate_reference_audio(short, bounded)
        self.assertEqual(ctx.exception.details["reason"],
                         "duration_out_of_range")
        good = _write_wav(self.path("good.wav"), seconds=2.0)
        tiny = IngestPolicy(max_audio_bytes=1024)
        with self.assertRaises(ModelFileError) as ctx:
            validate_reference_audio(good, tiny)
        self.assertEqual(ctx.exception.details["reason"], "too_large")

    def test_unparseable_formats_pass_unmeasured(self):
        path = self.path("note.ogg")
        with open(path, "wb") as fh:
            fh.write(b"OggS" + b"\x00" * 128)
        info = validate_reference_audio(path)
        self.assertEqual(info.format, "ogg")
        self.assertIsNone(info.seconds)     # unmeasured, not invented


class TestModelFile(_TmpCase):
    def _onnx(self, name: str = "voice.onnx",
              payload: bytes = b"\x08\x07" + b"\x12" * 64) -> str:
        path = self.path(name)
        with open(path, "wb") as fh:
            fh.write(payload)
        return path

    def test_valid_model_measured(self):
        path = self._onnx()
        with open(path + ".json", "w", encoding="utf-8") as fh:
            json.dump({"audio": {"sample_rate": 22050}}, fh)
        info = validate_model_file(path)
        self.assertEqual(info.kind, "onnx_model")
        self.assertEqual(info.format, "onnx")
        self.assertEqual(info.sample_rate, 22050)
        self.assertEqual(len(info.sha256), 64)

    def test_absent_companion_is_recorded_not_refused(self):
        info = validate_model_file(self._onnx())
        self.assertEqual(info.sample_rate, 0)

    def test_non_model_refused(self):
        wav = _write_wav(self.path("clip.wav"))
        os.rename(wav, self.path("model.onnx"))
        with self.assertRaises(ModelFileError) as ctx:
            validate_model_file(self.path("model.onnx"))
        self.assertEqual(ctx.exception.details["reason"],
                         "unsupported_format")

    def test_corrupt_companion_refused(self):
        path = self._onnx()
        with open(path + ".json", "w", encoding="utf-8") as fh:
            fh.write("{not json")
        with self.assertRaises(ModelFileError) as ctx:
            validate_model_file(path)
        self.assertEqual(ctx.exception.details["reason"],
                         "corrupt_metadata")
        with open(path + ".json", "w", encoding="utf-8") as fh:
            json.dump({"audio": {"sample_rate": -5}}, fh)
        with self.assertRaises(ModelFileError) as ctx:
            validate_model_file(path)
        self.assertEqual(ctx.exception.details["reason"],
                         "corrupt_metadata")


class TestIngestWiring(_TmpCase):
    """The gate at the real ingest points: VoiceLibrary + Piper."""

    def test_upload_records_ingest_and_copies(self):
        src = _write_wav(self.path("src.wav"), seconds=2.0)
        lib = VoiceLibrary(os.path.join(self.tmp, "store"))
        profile = lib.upload_voice("mine", src)
        self.assertTrue(os.path.isfile(profile.reference_audio_path))
        self.assertEqual(profile.ingest["format"], "wav")
        self.assertEqual(profile.ingest["sample_rate"], 16000)
        self.assertEqual(len(profile.ingest["sha256"]), 64)
        # the record persists through the index round-trip
        lib2 = VoiceLibrary(os.path.join(self.tmp, "store"))
        self.assertEqual(lib2.get("mine").ingest["format"], "wav")

    def test_upload_refuses_video_before_copy(self):
        src = self.path("evil.wav")
        with open(src, "wb") as fh:
            fh.write(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64)
        store = os.path.join(self.tmp, "store")
        lib = VoiceLibrary(store)
        with self.assertRaises(ModelFileError) as ctx:
            lib.upload_voice("evil", src)
        self.assertEqual(ctx.exception.details["reason"],
                         "unsupported_format")
        # nothing entered managed storage
        self.assertIsNone(lib.get("evil"))
        self.assertFalse(os.path.exists(os.path.join(store,
                                                     "evil.wav")))

    def test_add_sample_refuses_truncated(self):
        src = _write_wav(self.path("src.wav"), seconds=2.0)
        lib = VoiceLibrary(os.path.join(self.tmp, "store"))
        lib.upload_voice("mine", src)
        bad = _write_wav(self.path("bad.wav"), seconds=2.0)
        with open(bad, "r+b") as fh:
            fh.truncate(64)
        with self.assertRaises(ModelFileError):
            lib.add_sample("mine", bad)
        self.assertEqual(lib.get("mine").extra_samples, [])


if __name__ == "__main__":
    unittest.main()
