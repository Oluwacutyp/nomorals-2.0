"""Tests for the stdlib-only `system` TTS backend (nomorals/voice/tts.py).

The OS's own speech service — `say` on macOS, `espeak-ng`/`espeak` on
Linux, PowerShell System.Speech on Windows — driven with nothing but
the standard library. Fake binaries (Python scripts using stdlib
aifc/wave) stand in for the real ones; sys.platform and PATH are
patched per test.
"""

from __future__ import annotations

import math
import os
import shutil
import stat
import struct
import sys
import tempfile
import unittest
import wave
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomorals.voice import tts as tts_mod
from nomorals.voice.tts import (
    SystemTTSBackend,
    UniversalTTS,
    _aiff_to_wav_bytes,
    _resample_linear,
    available_backends,
)


def _float_to_extended80(value: float) -> bytes:
    """float → IEEE 754 80-bit extended (for building test AIFFs)."""
    import math

    if value == 0:
        return b"\x00" * 10
    sign = 0x8000 if value < 0 else 0
    value = abs(value)
    exp = math.floor(math.log2(value)) + 16383
    mant = int(value / (2.0 ** (exp - 16383 - 63)))
    return struct.pack(">H", sign | exp) + mant.to_bytes(8, "big")


def _aiff_bytes(sr: int = 22050, seconds: float = 0.2,
                little: bool = False) -> bytes:
    """Minimal valid AIFF (big-endian) or AIFF-C/sowt (little-endian)."""
    import io

    n = int(sr * seconds)
    fmt = "<h" if little else ">h"
    frames = b"".join(
        struct.pack(fmt, int(10000 * math.sin(2 * math.pi * 440 * t / sr)))
        for t in range(n))
    comm_body = (struct.pack(">h", 1) + struct.pack(">i", n)
                 + struct.pack(">h", 16) + _float_to_extended80(float(sr)))
    if little:
        form = b"AIFC"
        # compressionType 'sowt' + Pascal-string compressionName (even-sized)
        comm_body += b"sowt" + bytes([4]) + b"sowt" + b"\x00"
    else:
        form = b"AIFF"
    comm = b"COMM" + struct.pack(">I", len(comm_body)) + comm_body
    ssnd_body = struct.pack(">I", 0) + struct.pack(">I", 0) + frames
    if len(ssnd_body) & 1:
        ssnd_body += b"\x00"
    ssnd = b"SSND" + struct.pack(">I", len(ssnd_body)) + ssnd_body
    body = form + comm + ssnd
    return b"FORM" + struct.pack(">I", len(body)) + body


FAKE_SAY = """\
#!/usr/bin/env python3
import math, os, struct, sys
args = sys.argv[1:]
out = None; voice = None; text_parts = []
i = 0
while i < len(args):
    if args[i] == "-o": out = args[i + 1]; i += 2
    elif args[i] == "-v": voice = args[i + 1]; i += 2
    else: text_parts.append(args[i]); i += 1
sidecar = os.environ.get("FAKE_SAY_SIDECAR")
if sidecar:
    with open(sidecar, "a") as f:
        f.write("voice=%s text=%s\\n" % (voice, " ".join(text_parts)))
sr = 22050
n = int(sr * 0.2)
# AIFF-C/sowt like the real `say`: little-endian frames
frames = b"".join(struct.pack("<h", int(10000 * math.sin(2 * math.pi * 440 * t / sr))) for t in range(n))
exp = 16397  # 22050 Hz in 80-bit extended
mant = 0xAC44000000000000
rate80 = struct.pack(">H", exp) + mant.to_bytes(8, "big")
comm_body = struct.pack(">h", 1) + struct.pack(">i", n) + struct.pack(">h", 16) + rate80 + b"sowt" + bytes([4]) + b"sowt" + b"\\x00"
comm = b"COMM" + struct.pack(">I", len(comm_body)) + comm_body
ssnd_body = struct.pack(">I", 0) + struct.pack(">I", 0) + frames
ssnd = b"SSND" + struct.pack(">I", len(ssnd_body)) + ssnd_body
body = b"AIFC" + comm + ssnd
with open(out, "wb") as f:
    f.write(b"FORM" + struct.pack(">I", len(body)) + body)
"""

FAKE_ESPEAK = """\
#!/usr/bin/env python3
import io, math, os, struct, sys, wave
args = sys.argv[1:]
sidecar = os.environ.get("FAKE_ESPEAK_SIDECAR")
if sidecar:
    with open(sidecar, "a") as f:
        f.write(" ".join(args) + "\\n")
sr = 22050
n = int(sr * 0.2)
buf = io.BytesIO()
with wave.open(buf, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes(b"".join(struct.pack("<h", int(10000 * math.sin(2 * math.pi * 440 * t / sr))) for t in range(n)))
sys.stdout.buffer.write(buf.getvalue())
"""


class TestSystemTTS(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="systts-")
        self.bin = os.path.join(self.tmp, "bin")
        os.makedirs(self.bin)
        self._old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = self.bin + os.pathsep + self._old_path

    def tearDown(self):
        os.environ["PATH"] = self._old_path
        for var in ("FAKE_SAY_SIDECAR", "FAKE_ESPEAK_SIDECAR"):
            os.environ.pop(var, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _install(self, name: str, body: str) -> str:
        path = os.path.join(self.bin, name)
        with open(path, "w") as fh:
            fh.write(body)
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR
                 | stat.S_IXGRP | stat.S_IXOTH)
        return path

    # -- detection ------------------------------------------------------
    def test_detect_say_on_darwin(self):
        self._install("say", FAKE_SAY)
        with mock.patch.object(sys, "platform", "darwin"):
            kind, exe = SystemTTSBackend.detect()
        self.assertEqual(kind, "say")
        self.assertTrue(exe.endswith(os.sep + "say"))
        with mock.patch.object(sys, "platform", "darwin"):
            self.assertTrue(SystemTTSBackend.available())

    def test_detect_espeak_on_linux(self):
        self._install("espeak-ng", FAKE_ESPEAK)
        with mock.patch.object(sys, "platform", "linux"):
            kind, exe = SystemTTSBackend.detect()
        self.assertEqual(kind, "espeak")
        self.assertIn("espeak-ng", exe)

    def test_detect_prefers_espeak_ng_over_espeak(self):
        self._install("espeak", FAKE_ESPEAK)
        ng = self._install("espeak-ng", FAKE_ESPEAK)
        with mock.patch.object(sys, "platform", "linux"):
            kind, exe = SystemTTSBackend.detect()
        self.assertEqual(exe, ng)

    def test_detect_none_when_no_service(self):
        with mock.patch.object(sys, "platform", "linux"):
            self.assertEqual(SystemTTSBackend.detect(), (None, None))
            self.assertFalse(SystemTTSBackend.available())

    def test_detect_unknown_platform(self):
        self._install("say", FAKE_SAY)
        with mock.patch.object(sys, "platform", "plan9"):
            self.assertFalse(SystemTTSBackend.available())

    # -- synthesis ------------------------------------------------------
    def test_synthesize_via_say(self):
        self._install("say", FAKE_SAY)
        with mock.patch.object(sys, "platform", "darwin"):
            backend = SystemTTSBackend()
        self.assertEqual(backend.name, "system")
        samples = backend.synthesize("hello world", None)
        self.assertEqual(len(samples), int(22050 * 0.2))
        self.assertTrue(all(-1.0 <= s <= 1.0 for s in samples[:100]))

    def test_synthesize_via_espeak(self):
        self._install("espeak-ng", FAKE_ESPEAK)
        with mock.patch.object(sys, "platform", "linux"):
            backend = SystemTTSBackend()
        samples = backend.synthesize("hello world", None)
        self.assertEqual(len(samples), int(22050 * 0.2))

    def test_say_receives_voice_name(self):
        self._install("say", FAKE_SAY)
        sidecar = os.path.join(self.tmp, "say.log")
        os.environ["FAKE_SAY_SIDECAR"] = sidecar
        from nomorals.voice.tts import VoiceProfile

        voice = VoiceProfile(name="v", preset_id="Samantha",
                             reference_audio_path="",
                             prompt_text="", language="en")
        with mock.patch.object(sys, "platform", "darwin"):
            SystemTTSBackend().synthesize("hi", voice)
        with open(sidecar) as fh:
            logged = fh.read()
        self.assertIn("voice=Samantha", logged)
        self.assertIn("text=hi", logged)

    def test_espeak_receives_lang_flag(self):
        self._install("espeak-ng", FAKE_ESPEAK)
        sidecar = os.path.join(self.tmp, "espeak.log")
        os.environ["FAKE_ESPEAK_SIDECAR"] = sidecar
        with mock.patch.object(sys, "platform", "linux"):
            SystemTTSBackend(lang="de").synthesize("hallo", None)
        with open(sidecar) as fh:
            logged = fh.read()
        self.assertIn("-v de", logged)
        self.assertIn("--stdout", logged)

    def test_synthesize_empty_text_returns_empty(self):
        self._install("espeak-ng", FAKE_ESPEAK)
        with mock.patch.object(sys, "platform", "linux"):
            self.assertEqual(SystemTTSBackend().synthesize("   ", None),
                             [])

    def test_synthesize_no_service_raises_helpful(self):
        with mock.patch.object(sys, "platform", "linux"):
            backend = SystemTTSBackend()
        with self.assertRaises(RuntimeError) as ctx:
            backend.synthesize("hello", None)
        self.assertIn("espeak-ng", str(ctx.exception))

    # -- stdlib helpers ---------------------------------------------------
    def test_resample_linear_identity(self):
        samples = [0.1, 0.2, 0.3]
        self.assertEqual(_resample_linear(samples, 22050, 22050), samples)

    def test_resample_linear_changes_length(self):
        samples = [0.0] * 16000
        out = _resample_linear(samples, 16000, 22050)
        self.assertEqual(len(out), 22050)

    def test_aiff_to_wav_roundtrip(self):
        from nomorals.voice.tts import HFEndpointBackend

        blob = _aiff_to_wav_bytes(_aiff_bytes())
        samples, rate = HFEndpointBackend._decode_wav(blob)
        self.assertEqual(rate, 22050)
        self.assertEqual(len(samples), int(22050 * 0.2))
        # sine peak survives the big-endian swap
        self.assertGreater(max(abs(s) for s in samples), 0.2)

    def test_aiffc_sowt_roundtrip(self):
        # what macOS `say` really writes: AIFF-C, little-endian frames
        from nomorals.voice.tts import HFEndpointBackend

        blob = _aiff_to_wav_bytes(_aiff_bytes(little=True))
        samples, rate = HFEndpointBackend._decode_wav(blob)
        self.assertEqual(rate, 22050)
        self.assertEqual(len(samples), int(22050 * 0.2))
        self.assertGreater(max(abs(s) for s in samples), 0.2)

    def test_extended80_known_value(self):
        from nomorals.voice.tts import _extended80_to_float

        self.assertAlmostEqual(
            _extended80_to_float(_float_to_extended80(22050.0)), 22050.0)
        self.assertAlmostEqual(
            _extended80_to_float(_float_to_extended80(44100.0)), 44100.0)
        self.assertEqual(_extended80_to_float(b"\x00" * 10), 0.0)

    def test_aiff_rejects_garbage(self):
        with self.assertRaises(ValueError):
            _aiff_to_wav_bytes(b"definitely not aiff")
        with self.assertRaises(ValueError):
            _aiff_to_wav_bytes(b"FORM" + b"\x00" * 8 + b"NOPE"
                               + b"\x00" * 40)

    # -- registry / auto chain -------------------------------------------
    def test_specs_cover_system(self):
        self.assertIn("system", tts_mod._BACKENDS)
        self.assertIn("system", tts_mod._BACKEND_SPECS)

    def test_available_backends_puts_system_last(self):
        self._install("espeak-ng", FAKE_ESPEAK)
        with mock.patch.object(sys, "platform", "linux"):
            found = available_backends()
        self.assertIn("system", found)
        self.assertEqual(found[-1], "system")

    def test_available_backends_omits_system_without_service(self):
        with mock.patch.object(sys, "platform", "linux"):
            self.assertNotIn("system", available_backends())

    def test_explicit_system_backend_missing_service_raises(self):
        with mock.patch.object(sys, "platform", "linux"):
            engine = UniversalTTS(backend="system",
                                  voices_dir=os.path.join(self.tmp, "v"))
            with self.assertRaises(RuntimeError) as ctx:
                engine.speak("hello")
        self.assertIn("espeak-ng", str(ctx.exception))

    def test_speak_end_to_end_with_fake_espeak(self):
        self._install("espeak-ng", FAKE_ESPEAK)
        out = os.path.join(self.tmp, "out.wav")
        with mock.patch.object(sys, "platform", "linux"):
            engine = UniversalTTS(backend="system",
                                  voices_dir=os.path.join(self.tmp, "v"))
            result = engine.speak("hello there", out_path=out)
        self.assertEqual(result["backend"], "system")
        self.assertEqual(result["sample_rate"], 22050)
        self.assertTrue(os.path.isfile(out))
        with wave.open(out, "rb") as wav:
            self.assertEqual(wav.getframerate(), 22050)
            self.assertGreater(wav.getnframes(), 0)

    def test_perform_end_to_end_with_fake_espeak(self):
        self._install("espeak-ng", FAKE_ESPEAK)
        out = os.path.join(self.tmp, "perf.wav")
        with mock.patch.object(sys, "platform", "linux"):
            engine = UniversalTTS(backend="system",
                                  voices_dir=os.path.join(self.tmp, "v"))
            result = engine.perform("Well hello! [laugh] Great news.",
                                    out_path=out, seed=1)
        self.assertEqual(result["backend"], "system")
        self.assertIn("script", result)
        self.assertTrue(os.path.isfile(out))


if __name__ == "__main__":
    unittest.main()
