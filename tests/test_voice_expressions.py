"""Expression-vocabulary tests for the god-tier voice pass.

Covers: the expanded canonical markup (30 bursts, 53 emotions,
delivery styles, new fillers), the Dia + Orpheus renderers, the
onomatopoeia fallback for plain backends, stutter counts, the mood
burst tables, multi-sample cloning, reference-audio probing, the new
backend plumbing (stubbed — no weights, no network, no GPU), and the
latent ``_load_backend`` KeyError fix for ``hf-endpoint``.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import types
import unittest

from nomorals.voice import director as D
from nomorals.voice.director import (
    CANONICAL_BURSTS,
    CANONICAL_DELIVERY,
    CANONICAL_EMOTIONS,
    CANONICAL_FILLERS,
    ONOMATOPOEIA,
    PerformanceTuner,
    direct,
    render_bark,
    render_cosyvoice,
    render_dia,
    render_fish,
    render_for,
    render_orpheus,
    render_plain,
)
from nomorals.voice.tts import (
    TagProcessor,
    UniversalTTS,
    VoiceLibrary,
    VoiceProfile,
    _BACKEND_SPECS,
    _BACKENDS,
    available_backends,
    probe_reference_audio,
    write_wav,
)


class VocabularyTest(unittest.TestCase):
    def test_every_burst_matches_burst_regex(self):
        for burst in CANONICAL_BURSTS:
            m = D._BURST_RE.fullmatch(f"[{burst}]")
            self.assertIsNotNone(m, burst)
            self.assertEqual(m.group(1).lower(), burst)

    def test_every_emotion_and_delivery_matches_emotion_regex(self):
        for emo in CANONICAL_EMOTIONS + CANONICAL_DELIVERY:
            m = D._EMOTION_RE.fullmatch(f"[{emo}]")
            self.assertIsNotNone(m, emo)

    def test_burst_count_covers_the_multiverse(self):
        self.assertGreaterEqual(len(CANONICAL_BURSTS), 30)
        for want in ("laugh", "bellylaugh", "nervouslaugh", "chuckle",
                     "giggle", "sneeze", "sniffle", "groan", "scream",
                     "whistle", "hum", "mumble", "whimper", "pant",
                     "beep", "clap", "applause", "snore"):
            self.assertIn(want, CANONICAL_BURSTS)

    def test_emotion_count(self):
        self.assertGreaterEqual(len(CANONICAL_EMOTIONS), 50)
        for want in ("nostalgic", "terrified", "ecstatic", "deadpan",
                     "smug", "wistful", "triumphant", "desperate",
                     "panicked", "disgusted", "suspicious", "amused",
                     "relieved", "hesitant", "skeptical"):
            self.assertIn(want, CANONICAL_EMOTIONS)

    def test_new_fillers(self):
        for filler in ("like", "well", "yknow", "right", "so"):
            self.assertIn(filler, CANONICAL_FILLERS)
            self.assertIsNotNone(D._BURST_RE.fullmatch(f"[{filler}]"))

    def test_onomatopoeia_covers_every_burst(self):
        for burst in CANONICAL_BURSTS:
            self.assertIn(burst, ONOMATOPOEIA, burst)


class DiaRendererTest(unittest.TestCase):
    def test_native_paren_tags(self):
        out = render_dia("[laugh] ha [sneeze] achoo [cough]")
        self.assertIn("(laughs)", out)
        self.assertIn("(sneezes)", out)
        self.assertIn("(coughs)", out)

    def test_emotions_become_parens(self):
        out = render_dia("[happy] hello [whisper] secret")
        self.assertIn("(happy)", out)
        self.assertIn("(whisper)", out)

    def test_speaker_prefix_added(self):
        self.assertTrue(render_dia("hello").startswith("[S1]"))
        self.assertTrue(render_dia("[S2] hi").startswith("[S2]"))

    def test_strong_becomes_caps(self):
        self.assertIn("REALLY", render_dia("it is <strong>really</strong>"))

    def test_stutter_stays_textual(self):
        self.assertIn("r-really", render_dia("[stutter]really"))

    def test_pause_becomes_ellipsis(self):
        self.assertIn("...", render_dia("wait [pause:400] ok"))


class OrpheusRendererTest(unittest.TestCase):
    def test_native_angle_tags(self):
        out = render_orpheus("[laugh] ha [sigh] oh [cough] [groan]")
        self.assertIn("<laugh>", out)
        self.assertIn("<sigh>", out)
        self.assertIn("<cough>", out)
        self.assertIn("<groan>", out)

    def test_non_native_bursts_fall_back_to_onomatopoeia(self):
        out = render_orpheus("[scream] [beep] [whistle]")
        self.assertIn("aaah!", out)
        self.assertIn("*beep*", out)
        self.assertIn("*whistles*", out)

    def test_sneeze_maps_to_native_sniffle(self):
        self.assertIn("<sniffle>", render_orpheus("[sneeze]"))

    def test_emotions_become_angle_tags(self):
        out = render_orpheus("[happy] yay")
        self.assertIn("<happy>", out)

    def test_stutter_counts(self):
        self.assertIn("r-r-really", render_orpheus("[stutter:2]really"))
        self.assertIn("r-really", render_orpheus("[stutter]really"))


class PlainRendererTest(unittest.TestCase):
    def test_default_still_drops_bursts(self):
        text, pauses = render_plain("[sneeze] achoo [pause:300] [um] ok")
        self.assertNotIn("sneeze", text)
        self.assertIn("um,", text)
        self.assertEqual(pauses, [(8, 300)])

    def test_speak_bursts_uses_onomatopoeia(self):
        text, _pauses = render_plain("[sneeze] [cough] [whistle]",
                                     speak_bursts=True)
        self.assertIn("achoo!", text)
        self.assertIn("*cough*", text)
        self.assertIn("*whistles*", text)

    def test_speak_bursts_drops_empty_onomatopoeia(self):
        text, _ = render_plain("[breath] hello", speak_bursts=True)
        self.assertEqual(text, "hello")


class OtherRenderersNewBurstsTest(unittest.TestCase):
    def test_fish_new_bursts(self):
        out = render_fish("[scream] [bellylaugh] [sneeze] [whistle]")
        self.assertIn("[scream]", out)
        self.assertIn("[belly laughing]", out)
        self.assertIn("[sneeze]", out)
        self.assertIn("[whistle]", out)

    def test_bark_new_bursts(self):
        out = render_bark("[sneeze] [groan] [whistle]")
        self.assertIn("[sneezes]", out)
        self.assertIn("[groans]", out)
        self.assertIn("[whistles]", out)

    def test_cosyvoice_new_bursts(self):
        text, _instruct = render_cosyvoice("[sneeze] [scream] [yawn]")
        self.assertIn("achoo!", text)
        self.assertIn("aaah!", text)

    def test_render_for_routing(self):
        text, extra = render_for("dia", "[laugh] hi")
        self.assertIn("(laughs)", text)
        self.assertIsNone(extra)
        text, extra = render_for("orpheus", "[laugh] hi")
        self.assertIn("<laugh>", text)
        self.assertIsNone(extra)


class StutterTest(unittest.TestCase):
    def test_counts(self):
        self.assertIn("r-r-r-really",
                      D._apply_stutter("[stutter:3]really"))
        self.assertIn("w-word", D._apply_stutter("[stutter]word"))

    def test_count_clamped(self):
        out = D._apply_stutter("[stutter:99]really")
        self.assertEqual(out.count("r-"), 4)  # clamped to 4

    def test_punctuation_kept(self):
        self.assertTrue(
            D._apply_stutter('[stutter]"hello').startswith('"h-hello'))


class MoodBurstTableTest(unittest.TestCase):
    def test_new_moods_have_burst_tables(self):
        for mood in ("sick", "sleepy", "hysterical", "playful"):
            self.assertIn(mood, D._MOOD_PROFILES)
            self.assertTrue(D._MOOD_PROFILES[mood]["bursts"])
        self.assertEqual(D._MOOD_PROFILES["sick"]["bursts"]["cough"], 0.35)

    def test_burst_table_roll_is_deterministic(self):
        tuner = PerformanceTuner(mood="neutral", intensity=5, seed=1)
        tuner.profile = {"laugh": 0.0, "chuckle": 0.0, "breath": 0.0,
                         "sigh": 0.0, "filler": 0.0, "stutter": 0.0,
                         "rate": None, "bursts": {"cough": 1.0}}
        intent = tuner.detect("hello there friend")
        shaped, cues = tuner.shape("hello there friend", intent, [], 0, 1)
        self.assertIn("[cough]", shaped)
        self.assertIn("cough", cues)

    def test_seed_reproducibility_with_new_moods(self):
        a = direct("I feel awful today", mood="sick",
                   intensity=5, seed=42).text
        b = direct("I feel awful today", mood="sick",
                   intensity=5, seed=42).text
        self.assertEqual(a, b)

    def test_new_fillers_can_fire(self):
        tuner = PerformanceTuner(mood="nervous", intensity=5, seed=3)
        seen = set()
        for i in range(30):
            t = PerformanceTuner(mood="nervous", intensity=5, seed=i)
            script = t.tune("are you sure about this?")
            seen.update(script.cues)
        self.assertIn("filler", seen)


class LegacyTagProcessorTest(unittest.TestCase):
    def test_new_bursts_parse_as_sounds(self):
        tp = TagProcessor()
        segs = tp.parse("[sneeze] achoo [bellylaugh]")
        sounds = [s for s in segs if s.tags == ["_sound_"]]
        self.assertEqual(len(sounds), 2)

    def test_to_bark_format_maps_canonical(self):
        tp = TagProcessor()
        segs = tp.parse("[sneeze] achoo [bellylaugh]")
        out = tp.to_bark_format(segs)
        self.assertIn("[sneezes]", out)
        self.assertIn("[laughs]", out)

    def test_legacy_tags_still_pass_through(self):
        tp = TagProcessor()
        segs = tp.parse("[laughs] ha [clears_throat]")
        out = tp.to_bark_format(segs)
        self.assertIn("[laughs]", out)
        self.assertIn("[clears_throat]", out)

    def test_new_emotion_tags_parse(self):
        tp = TagProcessor()
        segs = tp.parse("[terrified] run [ecstatic] yay")
        self.assertEqual(segs[0].tags, ["terrified"])


def _stub_module(name: str, **attrs: object) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


class BackendPlumbingTest(unittest.TestCase):
    def tearDown(self):
        for name in ("dia", "dia.model", "orpheus_tts", "huggingface_hub",
                     "TTS", "TTS.api"):
            sys.modules.pop(name, None)

    def test_dia_missing_raises_helpful_error(self):
        sys.modules.pop("dia", None)
        sys.modules.pop("dia.model", None)
        from nomorals.voice.tts import DiaBackend
        with self.assertRaises(RuntimeError) as ctx:
            DiaBackend()
        self.assertIn("pip install", str(ctx.exception))

    def test_orpheus_missing_raises_helpful_error(self):
        sys.modules.pop("orpheus_tts", None)
        from nomorals.voice.tts import OrpheusBackend
        with self.assertRaises(RuntimeError) as ctx:
            OrpheusBackend()
        self.assertIn("orpheus-speech", str(ctx.exception))

    def test_orpheus_no_consent_gate(self):
        # Consent gate removed per audit: cloning works, audit-logged.
        _stub_module(
            "orpheus_tts",
            OrpheusModel=lambda model_name: types.SimpleNamespace(
                generate_speech=lambda prompt, voice: [b"\x00\x01"]))
        from nomorals.voice.tts import OrpheusBackend
        backend = OrpheusBackend()
        voice = VoiceProfile(name="v", reference_audio_path="/tmp/x.wav")
        # Should NOT raise — no consent gate anymore
        backend.synthesize("<laugh> hi", voice)

    def test_orpheus_synthesize_decodes_pcm(self):
        import struct as _struct

        pcm = _struct.pack("<4h", 0, 16384, -16384, 32767)
        _stub_module(
            "orpheus_tts",
            OrpheusModel=lambda model_name: types.SimpleNamespace(
                generate_speech=lambda prompt, voice: [pcm]))
        from nomorals.voice.tts import OrpheusBackend
        backend = OrpheusBackend()
        out = backend.synthesize("<laugh> hi", None)
        self.assertEqual(len(out), 4)
        self.assertAlmostEqual(out[1], 0.5, places=4)
        self.assertAlmostEqual(out[2], -0.5, places=4)

    def test_hf_endpoint_loads_without_keyerror(self):
        # Regression: "hf-endpoint" was in _BACKENDS but missing from the
        # old inline spec map, so _load_backend raised KeyError.
        _stub_module("huggingface_hub",
                     InferenceClient=object)
        from unittest import mock

        import nomorals.voice.tts as tts_mod
        from nomorals.voice.tts import HFEndpointBackend
        voices_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, voices_dir, ignore_errors=True)
        engine = UniversalTTS(backend="hf-endpoint",
                              voices_dir=voices_dir)
        with mock.patch.object(tts_mod, "_spec", return_value=True):
            backend = engine._load_backend()
        self.assertIsInstance(backend, HFEndpointBackend)

    def test_backend_specs_cover_all_backends(self):
        for name in _BACKENDS:
            self.assertIn(name, _BACKEND_SPECS, name)

    def test_available_backends_sees_stubbed_modules(self):
        from unittest import mock

        import nomorals.voice.tts as tts_mod
        _stub_module("orpheus_tts")
        _stub_module("dia")
        with mock.patch.object(tts_mod, "_spec",
                               side_effect=lambda n: n in ("orpheus_tts",
                                                           "dia")):
            found = available_backends()
        self.assertIn("orpheus", found)
        self.assertIn("dia", found)


class MultiSampleCloningTest(unittest.TestCase):
    def tearDown(self):
        for name in ("TTS", "TTS.api"):
            sys.modules.pop(name, None)

    def _stub_xtts(self, captured: dict):
        fake_api = types.SimpleNamespace(
            TTS=lambda model_id: types.SimpleNamespace(
                tts=lambda text, speaker_wav, language: (
                    captured.update(speaker_wav=speaker_wav) or [b""])))
        _stub_module("TTS.api", TTS=fake_api.TTS)
        _stub_module("TTS", api=sys.modules["TTS.api"])

    def test_single_sample_passes_string(self):
        captured: dict = {}
        self._stub_xtts(captured)
        from nomorals.voice.tts import XTTSBackend
        backend = XTTSBackend()
        voice = VoiceProfile(name="v", reference_audio_path="/tmp/a.wav")
        backend.synthesize("hi", voice)
        self.assertEqual(captured["speaker_wav"], "/tmp/a.wav")

    def test_multi_sample_passes_list(self):
        captured: dict = {}
        self._stub_xtts(captured)
        from nomorals.voice.tts import XTTSBackend
        backend = XTTSBackend()
        voice = VoiceProfile(name="v", reference_audio_path="/tmp/a.wav",
                             extra_samples=["/tmp/b.wav", "/tmp/c.wav"])
        backend.synthesize("hi", voice)
        self.assertEqual(captured["speaker_wav"],
                         ["/tmp/a.wav", "/tmp/b.wav", "/tmp/c.wav"])

    def test_reference_audios_property(self):
        voice = VoiceProfile(name="v")
        self.assertEqual(voice.reference_audios, [])
        voice = VoiceProfile(name="v", reference_audio_path="/tmp/a.wav",
                             extra_samples=["/tmp/b.wav"])
        self.assertEqual(voice.reference_audios,
                         ["/tmp/a.wav", "/tmp/b.wav"])


class VoiceLibrarySamplesTest(unittest.TestCase):
    def test_add_sample_and_transcript_roundtrip(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        lib = VoiceLibrary(tmp)
        src = os.path.join(tmp, "src.wav")
        write_wav(src, [0.0] * 240, 24000)
        lib.upload_voice("me", src)
        lib.add_sample("me", src)
        lib.set_transcript("me", "hello world")
        # reload from disk
        lib2 = VoiceLibrary(tmp)
        profile = lib2.get("me")
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(len(profile.extra_samples), 1)
        self.assertTrue(profile.extra_samples[0].endswith("_sample1.wav"))
        self.assertEqual(profile.prompt_text, "hello world")
        self.assertEqual(len(profile.reference_audios), 2)

    def test_add_sample_unknown_voice(self):
        voices_tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, voices_tmp, ignore_errors=True)
        lib = VoiceLibrary(voices_tmp)
        with self.assertRaises(KeyError):
            lib.add_sample("ghost", "/tmp/x.wav")


class ProbeReferenceAudioTest(unittest.TestCase):
    def test_good_clip(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        path = os.path.join(tmp, "good.wav")
        write_wav(path, [0.1] * (24000 * 10), 24000)
        info = probe_reference_audio(path)
        self.assertTrue(info["ok"])
        self.assertAlmostEqual(info["seconds"], 10.0, places=1)
        self.assertEqual(info["sample_rate"], 24000)
        self.assertEqual(info["warnings"], [])

    def test_short_clip_warns(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        path = os.path.join(tmp, "short.wav")
        write_wav(path, [0.1] * 24000, 24000)
        info = probe_reference_audio(path)
        self.assertTrue(info["ok"])
        self.assertTrue(any("short" in w for w in info["warnings"]))

    def test_missing_file(self):
        info = probe_reference_audio("/nonexistent/nope.wav")
        self.assertFalse(info["ok"])
        self.assertTrue(info["warnings"])


class FetchRegistryTest(unittest.TestCase):
    def test_new_models_registered(self):
        from nomorals.voice.fetch import MODEL_REGISTRY, fetch_model
        for name in ("orpheus", "dia", "qwen3-tts"):
            self.assertIn(name, MODEL_REGISTRY)
        self.assertEqual(
            MODEL_REGISTRY["orpheus"]["hf_repo"],
            "canopylabs/orpheus-3b-0.1-ft")
        self.assertEqual(
            MODEL_REGISTRY["dia"]["hf_repo"], "nari-labs/Dia-1.6B-0626")

    def test_empty_repo_raises_helpful_error(self):
        from nomorals.voice.fetch import fetch_model
        with self.assertRaises(ValueError) as ctx:
            fetch_model("qwen3-tts")
        self.assertIn("--repo", str(ctx.exception))

    def test_unknown_model(self):
        from nomorals.voice.fetch import fetch_model
        with self.assertRaises(ValueError):
            fetch_model("nope-not-a-model")


if __name__ == "__main__":
    unittest.main()
