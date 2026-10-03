"""God-tier voice engine tests: tuner, effects, Fish/HF rendering,
HF Inference Endpoint backend (stubbed), the persistent voice catalogue,
and the /voice chat-command surface.

No weights, no network, no GPU: the HF backend is tested against a
stubbed ``huggingface_hub`` module, and synthesis is never invoked.
"""

from __future__ import annotations

import io
import math
import os
import struct
import sys
import tempfile
import types
import unittest
import wave

from nomorals.social.chat.control import parse_control
from nomorals.voice.director import (
    EFFECT_PRESETS,
    PerformanceTuner,
    detect_intent,
    direct,
    normalize_for_speech,
    render_bark,
    render_cosyvoice,
    render_fish,
    render_plain,
)
from nomorals.voice.tts import HFEndpointBackend, write_wav


def _wav_bytes(samples: list[float], rate: int = 24000,
               width: int = 2, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        if width == 1:
            frames = bytes(max(0, min(255, int(s * 128 + 128)))
                           for s in samples)
            if channels == 2:
                frames = b"".join(bytes([b, b]) for b in frames)
            w.writeframes(frames)
        else:
            fmt = "<%dh" % (len(samples) * channels)
            vals = []
            for s in samples:
                v = int(max(-1.0, min(1.0, s)) * 32767)
                vals.extend([v] * channels)
            w.writeframes(struct.pack(fmt, *vals))
    return buf.getvalue()


class TunerNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.t = PerformanceTuner()

    def test_currency(self) -> None:
        self.assertIn("forty-two point five zero dollars",
                      self.t.normalize("It costs $42.50"))
        self.assertEqual("fifty euros", self.t.normalize("€50"))
        self.assertEqual("twenty pounds", self.t.normalize("£20"))
        self.assertEqual("five hundred naira", self.t.normalize("₦500"))

    def test_numbers_ordinals_years(self) -> None:
        self.assertEqual("first place", self.t.normalize("1st place"))
        self.assertEqual("third", self.t.normalize("3rd"))
        self.assertEqual("twenty twenty-six", self.t.normalize("2026"))
        self.assertEqual("three point one four", self.t.normalize("3.14"))
        self.assertEqual("one hundred percent", self.t.normalize("100%"))

    def test_times_abbreviations_social(self) -> None:
        self.assertEqual("at three thirty P M", self.t.normalize("at 3:30pm"))
        self.assertEqual("Doctor Smith", self.t.normalize("Dr. Smith"))
        self.assertEqual("hashtag tag at user", self.t.normalize("#tag @user"))

    def test_normalize_for_speech_matches_tuner(self) -> None:
        text = "Meet at 3:30pm, it costs $5. 2nd street."
        self.assertEqual(normalize_for_speech(text), self.t.normalize(text))


class IntentDetectionTests(unittest.TestCase):
    def test_kinds(self) -> None:
        cases = {
            "How are you?": "question",
            "Wow, amazing!": "exclaim",
            "Hello there": "greeting",
            "I'm so sorry": "apology",
            "haha that's funny": "laugh",
            "do it now please": "command",
            "once upon a time in a dark forest": "story",
            "STOP SHOUTING AT ME RIGHT NOW": "shout",
            "The sky is blue.": "statement",
        }
        for text, kind in cases.items():
            with self.subTest(text=text):
                self.assertEqual(kind, detect_intent(text).kind)

    def test_emotion_hints(self) -> None:
        self.assertEqual("happy", detect_intent("Hello there").emotion)
        self.assertEqual("empathetic", detect_intent("I'm so sorry").emotion)
        self.assertEqual("excited", detect_intent("Wow, amazing!").emotion)


class EffectRoutingTests(unittest.TestCase):
    def test_named_effect_wins(self) -> None:
        tuner = PerformanceTuner(effect="dramatic_whisper")
        routed = tuner.route(detect_intent("Hello there"))
        self.assertEqual(list(EFFECT_PRESETS["dramatic_whisper"]), routed)
        self.assertIn("whispering", routed)

    def test_unknown_effect_falls_back_to_intent(self) -> None:
        tuner = PerformanceTuner(effect="no_such_preset")
        routed = tuner.route(detect_intent("How are you?"))
        self.assertTrue(routed)  # intent-based fallback, never empty

    def test_presets_cover_many_styles(self) -> None:
        self.assertGreaterEqual(len(EFFECT_PRESETS), 10)
        for name in ("dramatic_whisper", "bedtime_story", "hype",
                     "standup", "villain", "sarcastic_bite"):
            self.assertIn(name, EFFECT_PRESETS)

    def test_seed_determinism(self) -> None:
        text = "Hello! How are you? Wait... really?"
        a = direct(text, seed=11, intensity=4, mood="happy")
        b = direct(text, seed=11, intensity=4, mood="happy")
        self.assertEqual(a.text, b.text)
        self.assertEqual(a.cues, b.cues)

    def test_different_seeds_may_differ(self) -> None:
        # not guaranteed, but with intensity 5 over a long text the dice
        # should find *something* different across seeds
        text = ("Wow! " * 20).strip()
        seen = {direct(text, seed=s, intensity=5).text for s in range(6)}
        self.assertGreater(len(seen), 1)

    def test_intensity_zero_is_plain(self) -> None:
        script = direct("Hello! How are you?", intensity=0, seed=1)
        self.assertNotIn("[laugh]", script.text)
        self.assertIn("Hello", script.text)


class RenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.script = direct("Wait... what? [pause:800] Really!",
                             seed=3, intensity=4)

    def test_render_fish_keeps_freeform_tags(self) -> None:
        out = render_fish(self.script)
        # the director's direction tags survive as free-form fish tags
        self.assertIn("[confident]", out)
        self.assertIn("[curious]", out)
        self.assertRegex(out, r"\[[a-z][a-z ]+\]")

    def test_render_fish_pauses_become_fish_tags(self) -> None:
        out = render_fish(self.script)
        self.assertIn("[long pause]", out)   # the 800ms pause
        self.assertNotIn("[pause:800]", out)

    def test_render_plain_strips_tags_gives_pause_points(self) -> None:
        text, pause_points = render_plain(self.script)
        self.assertNotIn("[happy]", text)
        self.assertNotIn("[pause:800]", text)
        self.assertTrue(pause_points)  # (char_idx, ms) splices for XTTS/Kokoro
        self.assertTrue(all(ms > 0 for _, ms in pause_points))

    def test_render_cosyvoice_gives_instruct(self) -> None:
        text, instruct = render_cosyvoice(self.script)
        self.assertIsInstance(text, str)
        self.assertIsInstance(instruct, str)

    def test_render_bark_native(self) -> None:
        out = render_bark(self.script)
        self.assertIsInstance(out, str)
        self.assertTrue(out.strip())


class FakeInferenceClient:
    """Stub for huggingface_hub.InferenceClient."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.init_kwargs = kwargs
        self.last_post: dict | None = None

    def text_to_speech(self, text: str) -> bytes:
        self.last_text = text
        samples = [math.sin(2 * math.pi * 440 * i / 24000) * 0.4
                   for i in range(1200)]
        return _wav_bytes(samples, 24000)

    def post(self, json: dict | None = None, **kwargs: object) -> bytes:
        self.last_post = json
        samples = [0.1] * 800
        return _wav_bytes(samples, 16000)


def _install_hf_stub() -> None:
    mod = types.ModuleType("huggingface_hub")
    mod.InferenceClient = FakeInferenceClient  # type: ignore[attr-defined]
    sys.modules["huggingface_hub"] = mod


class HFEndpointBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        _install_hf_stub()

    def tearDown(self) -> None:
        sys.modules.pop("huggingface_hub", None)

    def test_decode_wav_16bit_roundtrip(self) -> None:
        samples = [math.sin(i / 20) * 0.5 for i in range(1000)]
        blob = _wav_bytes(samples, 24000)
        mono, rate = HFEndpointBackend._decode_wav(blob)
        self.assertEqual(24000, rate)
        self.assertEqual(len(samples), len(mono))
        self.assertLess(max(abs(a - b) for a, b in zip(mono, samples)), 1e-3)

    def test_decode_wav_8bit_unsigned(self) -> None:
        blob = _wav_bytes([-1.0, -0.5, 0.0, 0.5], 8000, width=1)
        mono, rate = HFEndpointBackend._decode_wav(blob)
        self.assertEqual(8000, rate)
        for got, want in zip(mono, [-1.0, -0.5, 0.0, 0.5]):
            self.assertAlmostEqual(want, got, places=2)

    def test_decode_wav_stereo_mixes_to_mono(self) -> None:
        # one true stereo frame: L=+0.5, R=-0.5 → mono 0.0
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(struct.pack("<2h", int(0.5 * 32767),
                                      int(-0.5 * 32767)))
        mono, rate = HFEndpointBackend._decode_wav(buf.getvalue())
        self.assertEqual(16000, rate)
        self.assertEqual(1, len(mono))
        self.assertAlmostEqual(0.0, mono[0], places=2)

    def test_decode_wav_rejects_garbage(self) -> None:
        with self.assertRaises(Exception):
            HFEndpointBackend._decode_wav(b"not a wav at all")

    def test_serverless_path(self) -> None:
        backend = HFEndpointBackend(model="fishaudio/s2-pro")
        self.assertTrue(backend._is_fish())
        self.assertTrue(backend.supports_native_tags)
        out = backend.synthesize("[happy] hello", None)
        self.assertEqual(1200, len(out))
        self.assertEqual(24000, backend.sample_rate)

    def test_non_fish_model_is_plain(self) -> None:
        backend = HFEndpointBackend(model="some/other-tts")
        self.assertFalse(backend._is_fish())
        self.assertFalse(backend.supports_native_tags)

    def test_dedicated_endpoint_posts_inputs(self) -> None:
        backend = HFEndpointBackend(model="custom/cozy",
                                    endpoint_url="http://fake.endpoint")
        client_holder: dict = {}

        orig_client = backend._client

        def spy_client() -> FakeInferenceClient:
            c = orig_client()
            client_holder["c"] = c
            return c

        backend._client = spy_client  # type: ignore[method-assign]
        out = backend.synthesize("hello there", None, instruct="be happy")
        posted = client_holder["c"].last_post
        self.assertEqual("hello there", posted["inputs"])
        self.assertEqual("be happy", posted["parameters"]["instruct"])
        self.assertEqual(800, len(out))
        self.assertEqual(16000, backend.sample_rate)

    def test_clear_error_on_non_audio_response(self) -> None:
        backend = HFEndpointBackend(model="x")

        class JsonClient(FakeInferenceClient):
            def post(self, json: dict | None = None,
                     **kwargs: object) -> bytes:
                return b'{"error": "model loading"}'

        backend._client = lambda: JsonClient()  # type: ignore[method-assign]
        backend.endpoint_url = "http://fake"
        with self.assertRaisesRegex(ValueError, "did not return WAV audio"):
            backend.synthesize("hi", None)

    def test_empty_response_rejected(self) -> None:
        backend = HFEndpointBackend(model="x")

        class EmptyClient(FakeInferenceClient):
            def text_to_speech(self, text: str) -> bytes:
                return b""

        backend._client = lambda: EmptyClient()  # type: ignore[method-assign]
        with self.assertRaisesRegex(ValueError, "no audio bytes"):
            backend.synthesize("hi", None)


class VoiceCatalogueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["NOMORALS_VOICES_DIR"] = self.tmp.name
        from nomorals.voice.catalogue import default_catalogue
        self.cat = default_catalogue()

        # a tiny reference clip
        self.ref = os.path.join(self.tmp.name, "ref.wav")
        samples = [math.sin(2 * math.pi * 220 * i / 16000) * 0.3
                   for i in range(1600)]
        write_wav(self.ref, samples, 16000)

    def tearDown(self) -> None:
        os.environ.pop("NOMORALS_VOICES_DIR", None)
        self.tmp.cleanup()

    def test_clone_registers_profile_and_catalogue(self) -> None:
        from nomorals.voice import catalogue as catmod
        voice = self.cat.clone("narrator", self.ref,
                               transcript="hello world")
        self.assertEqual("narrator", voice.name)
        profile = self.cat.library.get("narrator")
        self.assertIsNotNone(profile)
        self.assertEqual("hello world", profile.prompt_text)
        self.assertTrue(os.path.exists(
            os.path.join(self.tmp.name, "narrator.wav")))

    def test_clone_without_consent_synthesizes(self) -> None:
        # No consent gate: cloning works, audit-logged.
        _install_hf_stub()
        self.addCleanup(sys.modules.pop, "huggingface_hub", None)
        from nomorals.voice.tts import VoiceProfile
        voice = self.cat.clone("noconsent", self.ref)
        profile = self.cat.library.get("noconsent")
        self.assertIsNotNone(profile)
        backend = HFEndpointBackend(model="x")
        out = backend.synthesize("hello", profile)
        self.assertTrue(len(out) > 0)

    def test_invalid_name_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.cat.clone("../evil", self.ref)

    def test_active_and_per_chat_switching(self) -> None:
        self.cat.clone("a", self.ref)
        self.cat.clone("b", self.ref)
        self.cat.set_active("a")
        self.assertEqual("a", self.cat.active_for_chat("chat:x").name)
        self.cat.set_chat_voice("chat:x", "b")
        self.assertEqual("b", self.cat.active_for_chat("chat:x").name)
        # other chats unaffected
        self.assertEqual("a", self.cat.active_for_chat("chat:y").name)
        self.assertTrue(self.cat.clear_chat_voice("chat:x"))
        self.assertEqual("a", self.cat.active_for_chat("chat:x").name)

    def test_unknown_voice_rejected(self) -> None:
        with self.assertRaises(KeyError):
            self.cat.set_active("ghost")
        with self.assertRaises(KeyError):
            self.cat.set_chat_voice("chat:x", "ghost")

    def test_persistence_roundtrip(self) -> None:
        from nomorals.voice.catalogue import default_catalogue
        self.cat.clone("keeper", self.ref,
                       description="kept")
        self.cat.set_active("keeper")
        self.cat.set_chat_voice("chat:z", "keeper")
        cat2 = default_catalogue()
        self.assertEqual("keeper", cat2.active)
        self.assertEqual("keeper", cat2.chat_overrides.get("chat:z"))
        self.assertEqual("kept", cat2.get("keeper").description)

    def test_remove_cleans_overrides(self) -> None:
        self.cat.clone("temp", self.ref)
        self.cat.set_chat_voice("chat:q", "temp")
        self.assertTrue(self.cat.remove("temp"))
        self.assertNotIn("chat:q", self.cat.chat_overrides)
        self.assertFalse(self.cat.remove("temp"))

    def test_resolve_backend_and_profile(self) -> None:
        self.cat.clone("v1", self.ref,
                       backend="fish")
        self.cat.set_active("v1")
        profile, backend = self.cat.resolve("any-chat")
        self.assertEqual("v1", profile)
        self.assertEqual("fish", backend)


class VoiceCommandParsingTests(unittest.TestCase):
    def test_voice_parses(self) -> None:
        cmd = parse_control("/voice use narrator")
        self.assertEqual("voice", cmd.kind)
        self.assertEqual("use", cmd.arg)
        self.assertEqual("use narrator", cmd.tail)

    def test_voice_subcommands_parse(self) -> None:
        for tail in ("list", "clone myvoice", "say hello there",
                     "transcript v some words", "describe v nice",
                     "rm v", "backend v fish"):
            with self.subTest(tail=tail):
                cmd = parse_control(f"/voice {tail}")
                self.assertEqual("voice", cmd.kind)


class ControlVoiceDispatchTests(unittest.TestCase):
    """Exercise PartnerRuntime._control_voice with a stub context."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["NOMORALS_VOICES_DIR"] = self.tmp.name
        self.ref = os.path.join(self.tmp.name, "ref.wav")
        write_wav(self.ref,
                  [math.sin(i / 10) * 0.3 for i in range(1600)], 16000)
        from nomorals.agents.partner_runtime import PartnerRuntime
        from types import SimpleNamespace
        self.rt = PartnerRuntime.__new__(PartnerRuntime)
        self.rt.context = SimpleNamespace(extras={}, tools=None)

    def tearDown(self) -> None:
        os.environ.pop("NOMORALS_VOICES_DIR", None)
        self.tmp.cleanup()

    def test_list_empty(self) -> None:
        reply = self.rt._control_voice("list", "chat:x")
        self.assertIn("no voices", reply)

    def test_use_unknown(self) -> None:
        reply = self.rt._control_voice("use ghost", "chat:x")
        self.assertIn("unknown voice", reply)

    def test_list_shows_active_and_chat_override(self) -> None:
        from nomorals.voice.catalogue import default_catalogue
        cat = default_catalogue()
        cat.clone("n1", self.ref)
        cat.clone("n2", self.ref)
        self.rt._control_voice("use n2", "chat:x")
        reply = self.rt._control_voice("list", "chat:x")
        self.assertIn("n1", reply)
        self.assertIn("n2", reply)
        self.assertIn("[this chat]", reply)

    def test_rm_flow(self) -> None:
        from nomorals.voice.catalogue import default_catalogue
        default_catalogue().clone("bye", self.ref)
        self.assertIn("removed", self.rt._control_voice("rm bye", "chat:x"))
        self.assertIn("unknown voice",
                      self.rt._control_voice("rm bye", "chat:x"))

    def test_say_without_voices_fails_cleanly(self) -> None:
        reply = self.rt._control_voice("say hello", "chat:x")
        self.assertIn("failed", reply.lower())

    def test_usage_on_bare_command(self) -> None:
        reply = self.rt._control_voice("", "chat:x")
        # bare /voice with no voices → the list help text
        self.assertIn("/voice clone", reply)


if __name__ == "__main__":
    unittest.main()
