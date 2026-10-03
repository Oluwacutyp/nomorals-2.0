"""Tests for UniversalSTT (nomorals/voice/stt.py).

faster-whisper (primary), NVIDIA Parakeet via onnx-asr, whisper.cpp via
pywhispercpp, classic whisper, and the HF serverless fallback — one
interface. No real models: every third-party module is stubbed.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import types
import unittest
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomorals.voice import stt as stt_mod
from nomorals.voice.stt import (
    UniversalSTT,
    _STT_BACKENDS,
    _STT_SPECS,
    available_stt_backends,
    make_session_stt,
)


def _wav(path: str) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)


def _stub(name: str, test: "TestUniversalSTT | None" = None,
          **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    if test is not None:
        test._stubbed.append(name)
    return mod


class TestUniversalSTT(unittest.TestCase):
    def setUp(self):
        self._stubbed: list[str] = []
        self.tmp = tempfile.mkdtemp(prefix="stt-")
        self.audio = os.path.join(self.tmp, "clip.wav")
        _wav(self.audio)

    def tearDown(self):
        for name in self._stubbed:
            sys.modules.pop(name, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    # -- faster-whisper -------------------------------------------------
    def _stub_faster_whisper(self, captured: dict):
        seg = types.SimpleNamespace(start=0.0, end=1.2, text=" hello")
        info = types.SimpleNamespace(language="en")

        def transcribe(path, **kwargs):
            captured.update(path=path, kwargs=kwargs)
            return [seg], info

        cls = type("WhisperModel", (), {
            "__init__": lambda self, model_id, device=None,
            compute_type=None: captured.update(
                model_id=model_id, device=device,
                compute_type=compute_type),
            "transcribe": staticmethod(transcribe),
        })
        _stub("faster_whisper", self, WhisperModel=cls)

    def test_faster_whisper_wiring(self):
        captured: dict = {}
        self._stub_faster_whisper(captured)
        stt = UniversalSTT(backend="faster-whisper")
        result = stt.transcribe(self.audio, language="en")
        self.assertEqual(result["text"], "hello")
        self.assertEqual(result["backend"], "faster-whisper")
        self.assertEqual(result["segments"][0]["end"], 1.2)
        self.assertEqual(captured["model_id"], "large-v3-turbo")
        self.assertEqual(captured["path"], self.audio)

    def test_faster_whisper_missing_package(self):
        from unittest import mock

        with mock.patch.object(stt_mod, "_spec", return_value=True):
            with self.assertRaises(RuntimeError) as ctx:
                UniversalSTT(backend="faster-whisper").transcribe(self.audio)
        self.assertIn("pip install faster-whisper", str(ctx.exception))

    # -- parakeet -------------------------------------------------------
    def _stub_onnx_asr(self, captured: dict):
        def load_model(model_id, quantization=None):
            captured.update(model_id=model_id, quantization=quantization)

            class _M:
                def recognize(self, path):
                    captured["path"] = path
                    return "  punctuated text!  "

            return _M()

        _stub("onnx_asr", self, load_model=load_model)

    def test_parakeet_wiring(self):
        captured: dict = {}
        self._stub_onnx_asr(captured)
        stt = UniversalSTT(backend="parakeet")
        result = stt.transcribe(self.audio, language="en")
        self.assertEqual(result["text"], "punctuated text!")
        self.assertEqual(result["backend"], "parakeet")
        self.assertEqual(captured["model_id"], "nemo-parakeet-tdt-0.6b-v3")

    def test_parakeet_missing_package(self):
        from unittest import mock

        with mock.patch.object(stt_mod, "_spec", return_value=True):
            with self.assertRaises(RuntimeError) as ctx:
                UniversalSTT(backend="parakeet").transcribe(self.audio)
        self.assertIn("pip install onnx-asr", str(ctx.exception))

    # -- whisper.cpp ----------------------------------------------------
    def test_whispercpp_wiring(self):
        seg = types.SimpleNamespace(t0=0, t1=150, text=" hi there")

        class _Model:
            def __init__(self, model_id):
                self.model_id = model_id

            def transcribe(self, path, language=None):
                self.last = (path, language)
                return [seg]

        fake_model = _Model("x")
        pkg = _stub("pywhispercpp", self)
        model_mod = _stub("pywhispercpp.model", self, Model=lambda m: fake_model)
        pkg.model = model_mod
        stt = UniversalSTT(backend="whisper.cpp")
        result = stt.transcribe(self.audio, language="en")
        self.assertEqual(result["text"], "hi there")
        self.assertEqual(result["backend"], "whisper.cpp")
        self.assertEqual(fake_model.last, (self.audio, "en"))

    # -- classic whisper -------------------------------------------------
    def test_whisper_wiring(self):
        def load_model(model_id):
            class _M:
                def transcribe(self, path, language=None):
                    return {"text": " classic text ",
                            "language": language or "en",
                            "segments": [{"start": 0.0, "end": 0.5,
                                          "text": "classic text"}]}

            return _M()

        _stub("whisper", self, load_model=load_model)
        stt = UniversalSTT(backend="whisper")
        result = stt.transcribe(self.audio, language="en")
        self.assertEqual(result["text"], "classic text")
        self.assertEqual(result["backend"], "whisper")
        self.assertEqual(len(result["segments"]), 1)

    # -- hf-stt ----------------------------------------------------------
    def test_hf_stt_wiring(self):
        captured: dict = {}

        class _Client:
            def __init__(self, model=None, token=None):
                captured.update(model=model, token=token)

            def automatic_speech_recognition(self, blob):
                captured["bytes"] = len(blob)
                return {"text": " cloud words "}

        _stub("huggingface_hub", self, InferenceClient=_Client)
        stt = UniversalSTT(backend="hf-stt")
        result = stt.transcribe(self.audio, language="en")
        self.assertEqual(result["text"], "cloud words")
        self.assertEqual(result["backend"], "hf-stt")
        self.assertTrue(captured["bytes"] > 0)

    # -- engine behavior --------------------------------------------------
    def test_auto_prefers_faster_whisper(self):
        captured: dict = {}
        self._stub_faster_whisper(captured)
        self._stub_onnx_asr({})
        stt = UniversalSTT(backend="auto")
        self.assertEqual(stt.backend_name, "faster-whisper")
        self.assertEqual(stt.transcribe(self.audio)["backend"],
                         "faster-whisper")

    def test_backend_name_before_load(self):
        stt = UniversalSTT(backend="parakeet")
        self.assertEqual(stt.backend_name, "parakeet")

    def test_unknown_backend(self):
        with self.assertRaises(RuntimeError):
            UniversalSTT(backend="nope").transcribe(self.audio)

    def test_no_backends_installed(self):
        with self.assertRaises(RuntimeError) as ctx:
            UniversalSTT(backend="auto").transcribe(self.audio)
        self.assertIn("no STT backend installed", str(ctx.exception))

    def test_missing_audio_file(self):
        captured: dict = {}
        self._stub_faster_whisper(captured)
        stt = UniversalSTT(backend="faster-whisper")
        with self.assertRaises(FileNotFoundError):
            stt.transcribe("/no/such/file.wav")

    def test_specs_cover_all_backends(self):
        for name in _STT_BACKENDS:
            self.assertIn(name, _STT_SPECS, name)

    def test_available_stt_order(self):
        captured: dict = {}
        self._stub_faster_whisper(captured)
        self._stub_onnx_asr({})
        found = available_stt_backends()
        self.assertEqual(found[0], "faster-whisper")
        self.assertIn("parakeet", found)

    # -- session adapter ---------------------------------------------------
    def test_make_session_stt_returns_text(self):
        captured: dict = {}
        self._stub_faster_whisper(captured)
        fn = make_session_stt(UniversalSTT(backend="faster-whisper"))
        self.assertEqual(fn(self.audio), "hello")
        self.assertFalse(fn.supports_partial)

    def test_make_session_stt_errors_degrade_to_empty(self):
        class _Broken:
            def transcribe(self, path, language="en"):
                raise RuntimeError("mic exploded")

        fn = make_session_stt(_Broken())
        self.assertEqual(fn(self.audio), "")

    def test_make_session_stt_accepts_plain_result(self):
        class _Plain:
            def transcribe(self, path, language="en"):
                return "plain string"

        fn = make_session_stt(_Plain())
        self.assertEqual(fn(self.audio), "plain string")


    # -- session wiring ---------------------------------------------------
    def test_make_local_stt_wires_into_session(self):
        from nomorals.voice.session import make_local_stt

        captured: dict = {}
        self._stub_faster_whisper(captured)
        fn = make_local_stt(backend="faster-whisper")
        self.assertEqual(fn(self.audio), "hello")
        self.assertFalse(fn.supports_partial)

    def test_make_local_stt_exposed_on_package(self):
        import nomorals.voice as voice_pkg
        from nomorals.voice.session import make_local_stt

        self.assertIs(voice_pkg.make_local_stt, make_local_stt)
        self.assertIn("make_local_stt", voice_pkg.__all__)


if __name__ == "__main__":
    unittest.main()
