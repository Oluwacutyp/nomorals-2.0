"""TTS auto-poisoning: a backend that constructs fine but fails at
synthesize time must not silence the engine (lane 86).

Regression: installing ``huggingface_hub`` (which faster-whisper
requires) makes the cloud ``hf-endpoint`` backend "available";
auto scores it above piper by quality (4 vs 3) and its synthesize
fails with a bare ``StopIteration`` (no provider can serve
``fishaudio/s2-pro`` in this env). The auto cascade only fell
through on CONSTRUCTION failures, so ``speak()`` raised and
narration went silent even though piper was installed and working.

The invariant pinned here: a backend that cannot serve must never
win auto-selection, and a synthesize failure on the chosen backend
falls through to the next working backend instead of going silent —
with the dead backend's failure named in the result.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

import nomorals.voice.tts as tts_mod
from nomorals.voice.tts import UniversalTTS


def _stub_module(name: str, **attrs: object) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


class _DeadHFClient:
    """huggingface_hub.InferenceClient that can serve nothing.

    Mirrors the real failure in this env: serverless routing finds no
    provider for the model and ``text_to_speech`` raises a bare
    ``StopIteration`` (empty message).
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def text_to_speech(self, text: str) -> bytes:
        raise StopIteration()


class _FakePiper:
    """A working local backend — the fall-through target."""

    name = "piper"
    sample_rate = 22050
    supports_streaming = False

    def __init__(self) -> None:
        pass

    def synthesize(self, text: str, voice: object, *,
                   instruct: str = "") -> list[float]:
        return [0.0] * 2205


class _DeadPiper(_FakePiper):
    """A local backend that also fails at synthesize time."""

    def synthesize(self, text: str, voice: object, *,
                   instruct: str = "") -> list[float]:
        raise RuntimeError("piper voice file missing")


class AutoPoisoningTest(unittest.TestCase):
    def setUp(self) -> None:
        self.voices_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.voices_dir, ignore_errors=True)
        self.out = os.path.join(self.voices_dir, "out.wav")
        memo = getattr(tts_mod, "_FAILED_BACKENDS", None)
        if memo is not None:
            memo.clear()
        _stub_module("huggingface_hub", InferenceClient=_DeadHFClient)
        # Hermetic segment cache: SegmentCache persists renders to
        # ~/.nomorals/voice_cache/segments/ keyed by (text, voice,
        # backend) — a fake render cached by one test would let a later
        # test "speak" without ever calling synthesize (and a fake
        # silence entry would poison real speakers of the same text).
        # Point the cache at this test's tmp dir.
        import nomorals.voice.efficiency as eff_mod

        self._cache_patch = mock.patch.object(
            eff_mod, "_cache_dir",
            return_value=os.path.join(self.voices_dir, "cache"))
        self._cache_patch.start()
        self.addCleanup(self._cache_patch.stop)

    def tearDown(self) -> None:
        sys.modules.pop("huggingface_hub", None)
        memo = getattr(tts_mod, "_FAILED_BACKENDS", None)
        if memo is not None:
            memo.clear()

    def _engine(self, backend: str = "auto") -> UniversalTTS:
        return UniversalTTS(backend=backend, voices_dir=self.voices_dir)

    def _patched_world(self) -> mock._patch:
        """hf-endpoint + piper installed, hf-endpoint ordered first."""
        backends = mock.patch.dict(
            tts_mod._BACKENDS,
            {"hf-endpoint": tts_mod.HFEndpointBackend,
             "piper": _FakePiper})
        avail = mock.patch.object(
            tts_mod, "available_backends",
            return_value=["hf-endpoint", "piper"])
        return backends, avail

    def test_auto_falls_through_dead_hf_endpoint_to_piper(self) -> None:
        backends, avail = self._patched_world()
        with backends, avail:
            engine = self._engine("auto")
            self.assertEqual(
                engine._ordered_backends("private", "file")[0],
                "hf-endpoint")
            res = engine.speak("Hello world", out_path=self.out)
        self.assertEqual(res["backend"], "piper")
        self.assertTrue(os.path.isfile(self.out))
        fallbacks = res.get("fallbacks") or []
        self.assertTrue(
            any("hf-endpoint" in f for f in fallbacks),
            f"dead backend not named in fallbacks: {fallbacks!r}")
        self.assertTrue(
            any("StopIteration" in f for f in fallbacks),
            f"real failure not named in fallbacks: {fallbacks!r}")

    def test_explicit_hf_endpoint_still_raises(self) -> None:
        # An explicitly requested backend fails loudly — the fix is
        # auto fall-through, not silencing a named choice.
        backends, avail = self._patched_world()
        with backends, avail:
            engine = self._engine("hf-endpoint")
            with self.assertRaises(StopIteration):
                engine.speak("Hello world", out_path=self.out)

    def test_all_backends_fail_names_every_failure(self) -> None:
        backends = mock.patch.dict(
            tts_mod._BACKENDS,
            {"hf-endpoint": tts_mod.HFEndpointBackend,
             "piper": _DeadPiper})
        avail = mock.patch.object(
            tts_mod, "available_backends",
            return_value=["hf-endpoint", "piper"])
        with backends, avail:
            engine = self._engine("auto")
            with self.assertRaises(RuntimeError) as ctx:
                engine.speak("Hello world", out_path=self.out)
        msg = str(ctx.exception)
        self.assertIn("hf-endpoint", msg)
        self.assertIn("piper", msg)

    def test_failed_backend_demoted_on_next_engine(self) -> None:
        # Narration builds a fresh engine per segment — a dead backend
        # must not win auto-selection again on the next engine.
        backends, avail = self._patched_world()
        with backends, avail:
            engine = self._engine("auto")
            res = engine.speak("Hello world", out_path=self.out)
            self.assertEqual(res["backend"], "piper")
            fresh = self._engine("auto")
            ordered = fresh._ordered_backends("private", "file")
        self.assertEqual(ordered[0], "piper")
        self.assertIn("hf-endpoint", ordered)

    def test_failed_backend_memo_expires(self) -> None:
        import time as _time

        backends, avail = self._patched_world()
        with backends, avail:
            engine = self._engine("auto")
            engine.speak("Hello world", out_path=self.out)
            memo = tts_mod._FAILED_BACKENDS
            self.assertIn("hf-endpoint", memo)
            memo["hf-endpoint"] = (
                _time.monotonic() - tts_mod._SYNTH_FAILURE_TTL_S - 1.0)
            fresh = self._engine("auto")
            ordered = fresh._ordered_backends("private", "file")
        self.assertEqual(ordered[0], "hf-endpoint")

    @unittest.skipUnless(
        importlib.util.find_spec("huggingface_hub") is not None,
        "needs huggingface_hub importable (fw-pypath)")
    @unittest.skipUnless(
        os.environ.get("PIPER_VOICE")
        and os.path.isfile(os.environ.get("PIPER_VOICE", "")),
        "needs real piper voice (PIPER_VOICE)")
    def test_real_auto_speaks_through_piper_when_hf_endpoint_dead(self) -> None:
        # Lane 84's reproduction, live: faster-whisper's
        # huggingface_hub on the path makes hf-endpoint "available"
        # and its synthesize fails — the voice must still speak.
        engine = self._engine("auto")
        res = engine.speak(
            "Hello world, this is a test.", out_path=self.out)
        self.assertEqual(res["backend"], "piper")
        self.assertTrue(os.path.isfile(self.out))
        fallbacks = res.get("fallbacks") or []
        self.assertTrue(
            any("hf-endpoint" in f for f in fallbacks),
            f"dead backend not named in fallbacks: {fallbacks!r}")


if __name__ == "__main__":
    unittest.main()
