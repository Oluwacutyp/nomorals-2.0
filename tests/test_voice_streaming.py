"""Tests for the TTS infrastructure upgrades (nomorals/voice/tts.py).

- True backend cascade: auto falls through a broken backend to the next
  working one, and only raises when every candidate failed (naming each).
- Smart selection: select_backends() quality-first for file rendering,
  latency-first (+streaming-capable first) for live voice; public
  audience structurally excludes non-commercial backends.
- Streaming: speak_stream() yields chunks in order for native streaming
  backends (Chatterbox Turbo, Orpheus), sentence-chunks for the rest,
  and NEVER RAISES — failures surface as {"ok": False} dicts.
- Voice profiles: NM_VOICE_PRIVATE / NM_VOICE_PUBLIC defaults, and
  structural enforcement — an XTTS-cloned voice may not serve public.
- fetch_piper_voice(): verified rhasspy/piper-voices repo layout.

No real models are loaded: stub ``chatterbox`` / ``huggingface_hub``
packages and fake backends stand in.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
import wave
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import nomorals.voice.tts as tts_mod
from nomorals.voice.tts import (
    BACKEND_CAPABILITIES,
    UniversalTTS,
    VoiceProfile,
    _split_sentences,
    select_backends,
)


# --------------------------------------------------------------------------
# fakes


class _FailBackend:
    """Explodes on construction — the cascade must skip it."""

    name = "fakefail"
    sample_rate = 24000
    supports_native_tags = False
    supports_streaming = False

    def __init__(self) -> None:
        raise RuntimeError("fakefail is broken on purpose")


class _PlainBackend:
    """Whole-utterance only — the engine sentence-chunks it."""

    name = "fakeplain"
    sample_rate = 16000
    supports_native_tags = False
    supports_streaming = False

    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(self, text, voice, *, instruct=""):
        self.calls.append(text)
        return [0.0] * 160


class _StreamBackend:
    """Native streaming backend."""

    name = "fakestream"
    sample_rate = 24000
    supports_native_tags = False
    supports_streaming = True

    def synthesize(self, text, voice, *, instruct=""):
        return [0.0] * 240

    def synthesize_stream(self, text, voice=None, *, instruct=""):
        for word in text.split():
            yield [0.1] * 24


class _FlakyBackend:
    """Streams one chunk, then dies — speak_stream must not raise."""

    name = "fakeflaky"
    sample_rate = 24000
    supports_native_tags = False
    supports_streaming = True

    def synthesize_stream(self, text, voice=None, *, instruct=""):
        yield [0.1] * 24
        raise RuntimeError("mid-stream explosion")


def _engine_with(backend) -> UniversalTTS:
    eng = UniversalTTS(backend="bark")
    eng._impl = backend
    eng._loaded = True
    return eng


def _make_wav(path: str) -> str:
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)
    return path


# --------------------------------------------------------------------------
# cascade


class CascadeTest(unittest.TestCase):
    def _backends(self):
        return {"fakefail": _FailBackend, "fakeplain": _PlainBackend,
                "fakestream": _StreamBackend}

    def test_auto_falls_through_broken_backend(self):
        with patch("nomorals.voice.tts.available_backends",
                   return_value=["fakefail", "fakeplain"]), \
             patch.dict("nomorals.voice.tts._BACKENDS", self._backends()):
            eng = UniversalTTS(backend="auto")
            impl = eng._load_backend()
            self.assertIsInstance(impl, _PlainBackend)

    def test_auto_all_fail_names_every_failure(self):
        with patch("nomorals.voice.tts.available_backends",
                   return_value=["fakefail"]), \
             patch.dict("nomorals.voice.tts._BACKENDS", self._backends()):
            eng = UniversalTTS(backend="auto")
            with self.assertRaises(RuntimeError) as ctx:
                eng._load_backend()
            msg = str(ctx.exception)
            self.assertIn("fakefail", msg)
            self.assertIn("broken on purpose", msg)
            self.assertIn("pip install", msg)

    def test_auto_nothing_installed_helpful(self):
        with patch("nomorals.voice.tts.available_backends", return_value=[]):
            eng = UniversalTTS(backend="auto")
            with self.assertRaises(RuntimeError) as ctx:
                eng._load_backend()
            self.assertIn("no TTS backend usable", str(ctx.exception))

    def test_explicit_backend_single_shot_error(self):
        # spec maps to an importable module so the install check passes
        # and the constructor's own error surfaces
        with patch.dict("nomorals.voice.tts._BACKENDS", self._backends()), \
             patch.dict("nomorals.voice.tts._BACKEND_SPECS",
                        {"fakefail": "sys"}):
            eng = UniversalTTS(backend="fakefail")
            with self.assertRaises(RuntimeError) as ctx:
                eng._load_backend()
            self.assertIn("broken on purpose", str(ctx.exception))

    def test_loaded_backend_cached(self):
        eng = _engine_with(_PlainBackend())
        self.assertIs(eng._load_backend(), eng._impl)


# --------------------------------------------------------------------------
# smart selection


class SelectBackendsTest(unittest.TestCase):
    FAKE_AVAIL = ["bark", "piper", "chatterbox", "system", "xtts"]

    def _select(self, **kw):
        with patch("nomorals.voice.tts.available_backends",
                   return_value=list(self.FAKE_AVAIL)):
            return select_backends(**kw)

    def test_file_is_quality_first(self):
        got = self._select(purpose="file", audience="private")
        # chatterbox (q5) beats piper (q3); piper (lat1) beats bark (lat5)
        self.assertEqual(got[0], "chatterbox")
        self.assertLess(got.index("piper"), got.index("bark"))
        self.assertEqual(got[-1], "system")  # q1 dead last

    def test_live_is_latency_first_streaming_first(self):
        got = self._select(purpose="live", audience="private")
        # chatterbox streams natively → first despite lat2
        self.assertEqual(got[0], "chatterbox")
        # piper (lat1) before bark (lat5)
        self.assertLess(got.index("piper"), got.index("bark"))

    def test_public_excludes_xtts_structurally(self):
        got = self._select(purpose="file", audience="public")
        self.assertNotIn("xtts", got)
        self.assertIn("chatterbox", got)

    def test_private_keeps_xtts_first(self):
        eng = UniversalTTS(audience="private")
        with patch("nomorals.voice.tts.available_backends",
                   return_value=list(self.FAKE_AVAIL)):
            ordered = eng._ordered_backends("private", "file")
        self.assertEqual(ordered[0], "xtts")

    def test_invalid_purpose_rejected(self):
        with self.assertRaises(ValueError):
            select_backends(purpose="carrier-pigeon")

    def test_invalid_audience_rejected(self):
        with self.assertRaises(ValueError):
            select_backends(audience="everyone")

    def test_capabilities_cover_registered_backends(self):
        for name in tts_mod._BACKENDS:
            self.assertIn(name, BACKEND_CAPABILITIES, name)
            cap = BACKEND_CAPABILITIES[name]
            self.assertIn("quality", cap)
            self.assertIn("latency", cap)
            self.assertIn("streams", cap)


# --------------------------------------------------------------------------
# streaming


class SpeakStreamTest(unittest.TestCase):
    def test_native_streaming_yields_ordered_chunks(self):
        backend = _StreamBackend()
        eng = _engine_with(backend)
        chunks = list(eng.speak_stream("hello brave new world"))
        self.assertEqual(len(chunks), 4)
        for i, chunk in enumerate(chunks, start=1):
            self.assertTrue(chunk["ok"])
            self.assertEqual(chunk["chunk"], i)
            self.assertEqual(chunk["backend"], "fakestream")
            self.assertEqual(chunk["sample_rate"], 24000)
            self.assertEqual(len(chunk["samples"]), 24)

    def test_sentence_fallback_one_call_per_sentence(self):
        backend = _PlainBackend()
        eng = _engine_with(backend)
        chunks = list(eng.speak_stream("Hello world. How are you? Fine!"))
        self.assertEqual(len(chunks), 3)
        self.assertEqual(backend.calls,
                         ["Hello world.", "How are you?", "Fine!"])
        for chunk in chunks:
            self.assertTrue(chunk["ok"])
            self.assertEqual(chunk["sample_rate"], 16000)

    def test_never_raises_no_backends(self):
        with patch("nomorals.voice.tts.available_backends", return_value=[]):
            eng = UniversalTTS(backend="auto")
            out = list(eng.speak_stream("hello"))
        self.assertEqual(len(out), 1)
        self.assertFalse(out[0]["ok"])
        self.assertIn("no TTS backend usable", out[0]["reason"])

    def test_never_raises_mid_stream_failure(self):
        eng = _engine_with(_FlakyBackend())
        out = list(eng.speak_stream("hello world"))
        self.assertEqual(len(out), 2)
        self.assertTrue(out[0]["ok"])
        self.assertFalse(out[1]["ok"])
        self.assertIn("mid-stream explosion", out[1]["reason"])

    def test_empty_text_is_honest_dict(self):
        eng = _engine_with(_StreamBackend())
        out = list(eng.speak_stream("   "))
        self.assertEqual(len(out), 1)
        self.assertFalse(out[0]["ok"])
        self.assertIn("empty", out[0]["reason"])

    def test_audience_override_flows_through(self):
        backend = _StreamBackend()
        eng = _engine_with(backend)
        chunks = list(eng.speak_stream("hi", audience="private"))
        self.assertTrue(chunks[0]["ok"])

    def test_split_sentences_never_raises(self):
        self.assertEqual(_split_sentences(""), [])
        self.assertEqual(_split_sentences("no punctuation"), ["no punctuation"])
        self.assertEqual(len(_split_sentences("a. b! c? d… e\nf")), 6)


# --------------------------------------------------------------------------
# voice profiles: owner-private vs public, enforced


class VoiceProfileEnforcementTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(
            self.tmp, ignore_errors=True))
        for key in ("NM_VOICE_PRIVATE", "NM_VOICE_PUBLIC"):
            self.addCleanup(os.environ.pop, key, None)
            os.environ.pop(key, None)

    def _engine(self, **kw):
        return UniversalTTS(voices_dir=os.path.join(self.tmp, "voices"),
                            **kw)

    def test_voice_for_reads_env(self):
        os.environ["NM_VOICE_PRIVATE"] = "owner-voice"
        os.environ["NM_VOICE_PUBLIC"] = "assistant-voice"
        eng = self._engine()
        self.assertEqual(eng.voice_for("private"), "owner-voice")
        self.assertEqual(eng.voice_for("public"), "assistant-voice")
        self.assertEqual(eng.voice_for(), "owner-voice")  # engine default

    def test_voice_for_unset_is_empty(self):
        self.assertEqual(self._engine().voice_for("private"), "")

    def test_voice_for_rejects_bad_audience(self):
        with self.assertRaises(ValueError):
            self._engine().voice_for("everyone")

    def _register_xtts_voice(self, eng) -> str:
        wav = _make_wav(os.path.join(self.tmp, "me.wav"))
        profile = eng.voices.upload_voice("me", wav, backend="xtts")
        return profile.name

    def test_public_refuses_xtts_cloned_voice(self):
        eng = self._engine(audience="public")
        name = self._register_xtts_voice(eng)
        with self.assertRaises(RuntimeError) as ctx:
            eng._resolve_voice(name, "public")
        self.assertIn("private", str(ctx.exception))

    def test_private_allows_xtts_cloned_voice(self):
        eng = self._engine(audience="private")
        name = self._register_xtts_voice(eng)
        voice = eng._resolve_voice(name, "private")
        self.assertIsInstance(voice, VoiceProfile)
        self.assertEqual(voice.name, name)

    def test_public_allows_mit_cloned_voice(self):
        eng = self._engine(audience="public")
        wav = _make_wav(os.path.join(self.tmp, "pub.wav"))
        eng.voices.upload_voice("assistant", wav, backend="chatterbox")
        voice = eng._resolve_voice("assistant", "public")
        self.assertEqual(voice.name, "assistant")

    def test_voice_backend_survives_library_roundtrip(self):
        eng = self._engine()
        wav = _make_wav(os.path.join(self.tmp, "rt.wav"))
        eng.voices.upload_voice("rt", wav, backend="chatterbox")
        fresh = UniversalTTS(voices_dir=os.path.join(self.tmp, "voices"))
        self.assertEqual(fresh.voices.get("rt").backend, "chatterbox")

    def test_env_default_voice_enforced_for_public(self):
        eng = self._engine(audience="public")
        name = self._register_xtts_voice(eng)
        os.environ["NM_VOICE_PUBLIC"] = name
        with self.assertRaises(RuntimeError):
            eng._resolve_voice(None, "public")


# --------------------------------------------------------------------------
# chatterbox streaming refactor


def _stub_chatterbox(with_stream: bool):
    stubbed = ["chatterbox", "chatterbox.mtl_tts", "chatterbox.tts_turbo"]

    class _FakeModel:
        sr = 24000

        def generate(self, text, **kwargs):
            return [0.2] * 48

        if with_stream:
            def generate_stream(self, text, **kwargs):
                yield [0.1] * 24, {"latency": 0.1}
                yield [0.1] * 24, {"latency": 0.2}

    def make_turbo(device=None, nano=False):
        return _FakeModel()

    pkg = types.ModuleType("chatterbox")
    mtl = types.ModuleType("chatterbox.mtl_tts")
    turbo = types.ModuleType("chatterbox.tts_turbo")
    turbo.ChatterboxTurboTTS = types.SimpleNamespace(
        from_pretrained=staticmethod(make_turbo))
    pkg.mtl_tts = mtl
    pkg.tts_turbo = turbo
    for name, mod in (("chatterbox", pkg), ("chatterbox.mtl_tts", mtl),
                      ("chatterbox.tts_turbo", turbo)):
        sys.modules[name] = mod
    return stubbed


class ChatterboxStreamingTest(unittest.TestCase):
    def setUp(self):
        os.environ["CHATTERBOX_DEVICE"] = "cpu"
        os.environ["CHATTERBOX_VARIANT"] = "turbo"
        self.addCleanup(os.environ.pop, "CHATTERBOX_DEVICE", None)
        self.addCleanup(os.environ.pop, "CHATTERBOX_VARIANT", None)

    def _backend(self, with_stream: bool):
        stubbed = _stub_chatterbox(with_stream)
        self.addCleanup(lambda: [sys.modules.pop(n, None)
                                 for n in stubbed])
        from nomorals.voice.tts import ChatterboxBackend

        return ChatterboxBackend()

    def test_synthesize_stream_native(self):
        backend = self._backend(with_stream=True)
        self.assertTrue(backend.supports_streaming)
        chunks = list(backend.synthesize_stream("hello world", None))
        self.assertEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertEqual(len(list(chunk)), 24)

    def test_synthesize_stream_falls_back_without_api(self):
        backend = self._backend(with_stream=False)
        chunks = list(backend.synthesize_stream("hello world", None))
        self.assertEqual(len(chunks), 1)  # one-shot fallback chunk
        self.assertEqual(len(list(chunks[0])), 48)

    def test_synthesize_unchanged(self):
        backend = self._backend(with_stream=True)
        out = backend.synthesize("hello", None)
        self.assertEqual(len(list(out)), 48)


# --------------------------------------------------------------------------
# fetch_piper_voice


class FetchPiperVoiceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(
            self.tmp, ignore_errors=True))
        mod = types.ModuleType("huggingface_hub")
        self.downloaded: list[tuple[str, str]] = []

        def hf_hub_download(repo_id, filename, local_dir):
            self.downloaded.append((repo_id, filename))
            # mimic the real layout: local_dir/<filename-with-subdirs>
            dest = os.path.join(local_dir, filename)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as fh:
                fh.write(b"fake")
            return dest

        mod.hf_hub_download = hf_hub_download
        sys.modules["huggingface_hub"] = mod
        self.addCleanup(sys.modules.pop, "huggingface_hub", None)

    def test_downloads_onnx_and_json_from_verified_layout(self):
        from nomorals.voice.fetch import fetch_piper_voice

        path = fetch_piper_voice("en_US-lessac-medium", dest_dir=self.tmp)
        self.assertEqual(path, os.path.join(self.tmp,
                                            "en_US-lessac-medium.onnx"))
        self.assertTrue(os.path.isfile(path))
        self.assertTrue(os.path.isfile(
            os.path.join(self.tmp, "en_US-lessac-medium.onnx.json")))
        repos = {r for r, _ in self.downloaded}
        self.assertEqual(repos, {"rhasspy/piper-voices"})
        files = sorted(f for _, f in self.downloaded)
        self.assertEqual(files, [
            "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
            "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        ])

    def test_malformed_voice_id_rejected(self):
        from nomorals.voice.fetch import _piper_voice_files

        with self.assertRaises(ValueError):
            _piper_voice_files("x")
        with self.assertRaises(ValueError):
            _piper_voice_files("")
        with self.assertRaises(ValueError):
            _piper_voice_files("en_US--medium")
        # shape-valid ids map to the verified repo layout
        onnx, js = _piper_voice_files("en_US-lessac-medium")
        self.assertEqual(onnx, "en/en_US/lessac/medium/en_US-lessac-medium.onnx")
        self.assertEqual(js,
                         "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json")

    def test_unknown_voice_fails_loudly_at_download(self):
        mod = sys.modules["huggingface_hub"]

        def _missing(repo_id, filename, local_dir):
            raise FileNotFoundError(filename)

        mod.hf_hub_download = _missing
        from nomorals.voice.fetch import fetch_piper_voice

        with self.assertRaises(RuntimeError) as ctx:
            fetch_piper_voice("en_US-nosuchvoice-medium", dest_dir=self.tmp)
        self.assertIn("not found", str(ctx.exception))

    def test_missing_hf_hub_is_helpful(self):
        sys.modules.pop("huggingface_hub", None)
        from nomorals.voice.fetch import fetch_piper_voice

        with self.assertRaises(RuntimeError) as ctx:
            fetch_piper_voice("en_US-lessac-medium", dest_dir=self.tmp)
        self.assertIn("huggingface_hub", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
