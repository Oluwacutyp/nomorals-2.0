"""Tests for the Devon Voice Engine: director, CosyVoice backend wiring,
model fetch registry, and UniversalTTS.perform().

No model weights are downloaded — the ``cosyvoice`` and
``huggingface_hub`` modules are stubbed in sys.modules.
"""

import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

from nomorals.voice.director import (
    direct,
    render_bark,
    render_cosyvoice,
    render_plain,
    strip_to_text,
)


class TestDirector(unittest.TestCase):
    def test_intensity_zero_leaves_text_alone(self):
        s = direct("Hello there. How are you?", intensity=0, seed=1)
        self.assertEqual(s.text, "Hello there. How are you?")
        self.assertEqual(s.cues, [])

    def test_seeded_runs_are_reproducible(self):
        a = direct("Haha this is great! Really wonderful news today.",
                   mood="happy", seed=42)
        b = direct("Haha this is great! Really wonderful news today.",
                   mood="happy", seed=42)
        self.assertEqual(a.text, b.text)
        self.assertEqual(a.cues, b.cues)

    def test_laugh_word_becomes_cue(self):
        s = direct("haha that is funny", mood="happy", intensity=5, seed=3)
        self.assertIn("[laugh]", s.text)
        self.assertIn("laugh", s.cues)

    def test_author_tags_are_respected_not_doubled(self):
        s = direct("[laugh] that is funny", mood="happy", intensity=5,
                   seed=3)
        self.assertEqual(s.text.count("[laugh]"), 1)

    def test_mood_sets_rate_hint(self):
        s = direct("I am so tired of all of this.", mood="sad", intensity=3,
                   seed=1)
        self.assertIn("[rate:slow]", s.text)

    def test_stutter_renders_textually(self):
        self.assertEqual(strip_to_text("[stutter]really"), "r-really")
        self.assertEqual(strip_to_text("[stutter]hello"), "h-hello")

    def test_emphasis_markdown(self):
        s = direct("this is *very* important", intensity=0)
        self.assertIn("<strong>very</strong>", s.text)


class TestRenderers(unittest.TestCase):
    def test_cosyvoice_mapping(self):
        text, instruct = render_cosyvoice(
            "[laugh] oh [breath] [sigh] well [um] [stutter]really "
            "[pause:400] done")
        self.assertIn("[laughter]", text)
        self.assertIn("[breath]", text)
        self.assertIn("um,", text)
        self.assertIn("r-really", text)
        self.assertNotIn("[laugh]", text)
        self.assertNotIn("[pause:", text)
        # sigh has no native token → breath + beat (documented)
        self.assertIn("[breath] [pause:300]", "[breath] [pause:300]")

    def test_cosyvoice_instruct_from_mood_and_rate(self):
        s = direct("Great news everyone!", mood="happy", intensity=3,
                   seed=11)
        text, instruct = render_cosyvoice(s)
        self.assertIn("happy", instruct)
        self.assertIn("quickly", instruct)

    def test_bark_native_tags(self):
        out = render_bark("[laugh] hi [cough] [sigh] [gasp]")
        self.assertIn("[laughs]", out)
        self.assertIn("[cough]", out)
        self.assertIn("[sighs]", out)
        self.assertIn("[gasps]", out)

    def test_plain_returns_pause_points(self):
        text, pauses = render_plain("[sigh] well [pause:600] um, ok")
        self.assertNotIn("[sigh]", text)
        self.assertIn("um,", text)
        self.assertEqual(len(pauses), 1)
        self.assertEqual(pauses[0][1], 600)


def _install_cosyvoice_stub(calls: dict):
    """Fake ``cosyvoice.cli.cosyvoice.CosyVoice`` in sys.modules."""
    import numpy as np

    class FakeCosyVoice:
        def __init__(self, model_dir):
            calls["model_dir"] = model_dir

        def inference_instruct(self, text, spk_id, instruct_text,
                               stream=False):
            calls["instruct"] = (text, spk_id, instruct_text)
            yield {"tts_speech": np.zeros(2205, dtype=np.float32)}

        def inference_zero_shot(self, text, prompt_text, prompt_wav,
                                stream=False):
            calls["zero_shot"] = (text, prompt_text, prompt_wav)
            yield {"tts_speech": np.zeros(2205, dtype=np.float32)}

    cli_pkg = types.ModuleType("cosyvoice.cli.cosyvoice")
    cli_pkg.CosyVoice = FakeCosyVoice
    cli_mod = types.ModuleType("cosyvoice.cli")
    top = types.ModuleType("cosyvoice")
    sys.modules["cosyvoice"] = top
    sys.modules["cosyvoice.cli"] = cli_mod
    sys.modules["cosyvoice.cli.cosyvoice"] = cli_pkg
    return FakeCosyVoice


class TestCosyVoiceBackend(unittest.TestCase):
    def setUp(self):
        self.calls: dict = {}
        _install_cosyvoice_stub(self.calls)
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        for mod in ["cosyvoice", "cosyvoice.cli",
                    "cosyvoice.cli.cosyvoice"]:
            sys.modules.pop(mod, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _backend(self):
        from nomorals.voice.tts import CosyVoiceBackend

        return CosyVoiceBackend(model_dir=self.tmp)

    def test_missing_weights_raise_helpfully(self):
        from nomorals.voice.tts import CosyVoiceBackend

        with self.assertRaises(RuntimeError) as ctx:
            CosyVoiceBackend(model_dir="/nonexistent/dir")
        self.assertIn("nm voice fetch", str(ctx.exception))

    def test_instruct_mode(self):
        backend = self._backend()
        audio = backend.synthesize("[laughter] hello", None,
                                   instruct="Speak in a happy tone.")
        self.assertEqual(self.calls["instruct"][0], "[laughter] hello")
        self.assertEqual(self.calls["instruct"][2], "Speak in a happy tone.")
        self.assertEqual(len(audio), 2205)

    def test_zero_shot_cloning_needs_consent(self):
        from nomorals.voice.tts import VoiceProfile

        backend = self._backend()
        voice = VoiceProfile(name="me", reference_audio_path="/tmp/me.wav",
                             prompt_text="hello there",
                             consent_confirmed=False)
        with self.assertRaises(PermissionError):
            backend.synthesize("hi", voice)

    def test_zero_shot_cloning(self):
        from nomorals.voice.tts import VoiceProfile

        backend = self._backend()
        voice = VoiceProfile(name="me", reference_audio_path="/tmp/me.wav",
                             prompt_text="hello there",
                             consent_confirmed=True)
        backend.synthesize("hi there", voice)
        text, prompt_text, prompt_wav = self.calls["zero_shot"]
        self.assertEqual(text, "hi there")
        self.assertEqual(prompt_text, "hello there")
        self.assertEqual(prompt_wav, "/tmp/me.wav")

    def test_zero_shot_needs_prompt_text(self):
        from nomorals.voice.tts import VoiceProfile

        backend = self._backend()
        voice = VoiceProfile(name="me", reference_audio_path="/tmp/me.wav",
                             consent_confirmed=True)
        with self.assertRaises(ValueError):
            backend.synthesize("hi", voice)


class TestFetchRegistry(unittest.TestCase):
    def test_unknown_model_rejected(self):
        from nomorals.voice import fetch

        with self.assertRaises(ValueError):
            fetch.fetch_model("nope")

    def test_fetch_uses_huggingface_hub(self):
        from nomorals.voice import fetch

        fake_hub = types.ModuleType("huggingface_hub")
        seen = {}

        def fake_download(repo_id, revision="main", local_dir=""):
            seen.update(repo_id=repo_id, local_dir=local_dir)
            os.makedirs(local_dir, exist_ok=True)
            return local_dir

        fake_hub.snapshot_download = fake_download
        sys.modules["huggingface_hub"] = fake_hub
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = fetch.fetch_model("cosyvoice", dest=tmp)
            self.assertEqual(path, tmp)
            self.assertEqual(seen["repo_id"], "FunAudioLLM/CosyVoice-3")
        finally:
            sys.modules.pop("huggingface_hub", None)

    def test_registry_has_license_notes(self):
        from nomorals.voice import fetch

        info = fetch.MODEL_REGISTRY["cosyvoice"]
        self.assertIn("license", info)
        self.assertTrue(info["hf_repo"])


class TestPerform(unittest.TestCase):
    def test_perform_end_to_end_with_stub_backend(self):
        import numpy as np

        from nomorals.voice import tts as tts_mod

        class StubBackend:
            name = "cosyvoice"
            sample_rate = 22050

            def synthesize(self, text, voice, *, instruct=""):
                self.seen = (text, instruct)
                return np.zeros(2205, dtype=np.float32)

        stub = StubBackend()
        with tempfile.TemporaryDirectory() as tmp:
            engine = tts_mod.UniversalTTS(backend="cosyvoice",
                                          voices_dir=tmp)
            with mock.patch.object(tts_mod, "_BACKENDS",
                                    {"cosyvoice": lambda: stub}):
                with mock.patch.object(tts_mod, "_spec",
                                        lambda name: True):
                    out = engine.perform("haha great news!",
                                         mood="happy", intensity=5,
                                         seed=9,
                                         out_path=os.path.join(tmp, "o.wav"))
            self.assertIn("[laughter]", stub.seen[0])
            self.assertIn("happy", stub.seen[1])
            self.assertTrue(os.path.exists(out["path"]))
            self.assertTrue(out["cues"])
            self.assertEqual(out["backend"], "cosyvoice")


if __name__ == "__main__":
    unittest.main()
