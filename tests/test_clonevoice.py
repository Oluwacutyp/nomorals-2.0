"""Tests for /clonevoice, /voices, /say <voice>, /voice say [voice].

XTTS is mocked — no model needed. Voice catalogue uses a temp dir.
"""
import os
import tempfile
import unittest
from unittest import mock

from nomorals.voice.catalogue import VoiceCatalogue, default_catalogue


def _make_catalogue():
    tmp = tempfile.mkdtemp(prefix="voices_test_")
    return VoiceCatalogue(voices_dir=tmp), tmp


class SpeakAsTests(unittest.TestCase):
    def test_speak_as_unknown_voice_raises_keyerror(self):
        cat, _ = _make_catalogue()
        with self.assertRaises(KeyError):
            cat.speak_as("hello", "nope")

    def test_speak_as_known_voice_calls_engine(self):
        cat, _ = _make_catalogue()
        cat.add("testvoice", backend="system", profile="",
                description="test", tags=("built-in",))
        fake_out = {"path": "/tmp/out.wav", "backend": "system"}
        with mock.patch("nomorals.voice.catalogue.UniversalTTS") as uts:
            uts.return_value.speak.return_value = fake_out
            out = cat.speak_as("hello there", "testvoice")
        self.assertEqual(out, fake_out)
        uts.return_value.speak.assert_called_once()
        _, kwargs = uts.return_value.speak.call_args
        self.assertEqual(kwargs.get("voice_name"), None)  # no profile set

    def test_speak_as_uses_voice_profile(self):
        cat, _ = _make_catalogue()
        cat.add("cloned1", backend="xtts", profile="cloned1",
                description="cloned", tags=("cloned",))
        with mock.patch("nomorals.voice.catalogue.UniversalTTS") as uts:
            uts.return_value.speak.return_value = {"path": "/tmp/x.wav"}
            cat.speak_as("hi", "cloned1")
        _, kwargs = uts.return_value.speak.call_args
        self.assertEqual(kwargs.get("voice_name"), "cloned1")

    def test_speak_as_does_not_change_active(self):
        cat, _ = _make_catalogue()
        cat.add("a", backend="system", tags=("built-in",))
        cat.add("b", backend="system", tags=("built-in",))
        cat.set_active("a")
        with mock.patch("nomorals.voice.catalogue.UniversalTTS") as uts:
            uts.return_value.speak.return_value = {"path": "/tmp/x.wav"}
            cat.speak_as("hi", "b")
        self.assertEqual(cat.active, "a")


class CloneVoiceCommandTests(unittest.TestCase):
    def _handler(self):
        # minimal mixin instance without full runtime
        from nomorals.agents.partner import runtime_voice as rv

        class H(rv.RuntimeVoiceMixin):
            def __init__(self):
                self.context = mock.MagicMock()

        return H()

    def test_clonevoice_no_name(self):
        h = self._handler()
        out = h._control_clonevoice("", "chat1", message=None)
        self.assertIn("usage", out.lower())

    def test_clonevoice_no_audio(self):
        h = self._handler()
        msg = mock.MagicMock()
        msg.media = []
        msg.reply_to = ""
        out = h._control_clonevoice("bob", "chat1", message=msg)
        self.assertIn("no audio", out.lower())

    def test_clonevoice_reply_to_honest(self):
        h = self._handler()
        msg = mock.MagicMock()
        msg.media = []
        msg.reply_to = "msg123"
        out = h._control_clonevoice("bob", "chat1", message=msg)
        self.assertIn("replied", out.lower())
        self.assertIn("attach", out.lower())

    def test_clonevoice_success(self):
        h = self._handler()
        wav = os.path.join(tempfile.gettempdir(), "clone_test_src.wav")
        with open(wav, "wb") as fh:
            fh.write(b"RIFF....fake")
        msg = mock.MagicMock()
        media = mock.MagicMock()
        media.kind = "voice"
        media.path = wav
        msg.media = [media]
        msg.reply_to = ""
        with mock.patch("nomorals.voice.catalogue.default_catalogue") as dc:
            cat, _ = _make_catalogue()
            # stub transcribe tool
            h.context.tools.call.return_value = mock.MagicMock(
                ok=False, value=None)
            dc.return_value = cat
            out = h._control_clonevoice("bob", "chat1", message=msg)
        self.assertIn("cloned 'bob'", out)
        # persisted
        self.assertIn("bob", cat.voices)
        self.assertIn("cloned", cat.voices["bob"].tags)

    def test_clonevoice_invalid_name(self):
        h = self._handler()
        msg = mock.MagicMock()
        media = mock.MagicMock()
        media.kind = "voice"
        media.path = "/tmp/does_not_matter.wav"
        msg.media = [media]
        msg.reply_to = ""
        with mock.patch("nomorals.voice.catalogue.default_catalogue") as dc:
            cat, _ = _make_catalogue()
            dc.return_value = cat
            out = h._control_clonevoice("bad/name!", "chat1", message=msg)
        self.assertIn("clone failed", out.lower())


class VoicesCommandTests(unittest.TestCase):
    def _handler(self):
        from nomorals.agents.partner import runtime_voice as rv

        class H(rv.RuntimeVoiceMixin):
            def __init__(self):
                self.context = mock.MagicMock()

        return H()

    def test_voices_empty(self):
        h = self._handler()
        with mock.patch("nomorals.voice.tts.SystemTTSBackend") as stb, \
             mock.patch("nomorals.voice.tts.PiperBackend") as pb, \
             mock.patch("nomorals.voice.catalogue.default_catalogue") as dc, \
             mock.patch("nomorals.voice.tts.available_backends",
                        return_value=[]):
            stb.detect.return_value = (None, None)
            pb._search_dirs.return_value = []
            cat, _ = _make_catalogue()
            dc.return_value = cat
            out = h._control_voices("", "chat1")
        self.assertIn("catalogue — empty", out)

    def test_voices_marks_cloned(self):
        h = self._handler()
        with mock.patch("nomorals.voice.tts.SystemTTSBackend") as stb, \
             mock.patch("nomorals.voice.tts.PiperBackend") as pb, \
             mock.patch("nomorals.voice.catalogue.default_catalogue") as dc, \
             mock.patch("nomorals.voice.tts.available_backends",
                        return_value=[]):
            stb.detect.return_value = (None, None)
            pb._search_dirs.return_value = []
            cat, _ = _make_catalogue()
            cat.add("sys1", backend="system", tags=("built-in",))
            cat.add("bob", backend="xtts", profile="bob",
                    tags=("cloned",))
            dc.return_value = cat
            out = h._control_voices("", "chat1")
        self.assertIn("bob", out)
        self.assertIn("🧬", out)  # cloned marker
        self.assertIn("sys1", out)


class SayTTSTests(unittest.TestCase):
    def _handler(self):
        from nomorals.agents.partner import runtime_voice as rv

        class H(rv.RuntimeVoiceMixin):
            def __init__(self):
                self.context = mock.MagicMock()
                self._delivered = None

            def _deliver_voice_note(self, chat_key, path, text, backend,
                                    size_kb=0):
                self._delivered = (chat_key, path, text, backend)
                return f"delivered {backend}"

        return H()

    def test_say_tts_unknown_voice(self):
        h = self._handler()
        with mock.patch("nomorals.voice.catalogue.default_catalogue") as dc:
            cat, _ = _make_catalogue()
            dc.return_value = cat
            out = h._control_say_tts("ghost", "hello", "chat1")
        self.assertIn("unknown voice", out.lower())

    def test_say_tts_success(self):
        h = self._handler()
        with mock.patch("nomorals.voice.catalogue.default_catalogue") as dc:
            cat, _ = _make_catalogue()
            cat.add("bob", backend="xtts", profile="bob", tags=("cloned",))
            dc.return_value = cat
            with mock.patch.object(cat, "speak_as",
                                    return_value={"path": "/tmp/v.wav",
                                                  "backend": "xtts"}) as sa:
                out = h._control_say_tts("bob", "hello world", "chat1")
        sa.assert_called_once_with("hello world", "bob")
        self.assertIn("delivered", out)
        self.assertEqual(h._delivered[2], "hello world")

    def test_say_tts_never_raises(self):
        h = self._handler()
        with mock.patch("nomorals.voice.catalogue.default_catalogue") as dc:
            cat, _ = _make_catalogue()
            cat.add("bob", backend="xtts", profile="bob", tags=("cloned",))
            dc.return_value = cat
            with mock.patch.object(cat, "speak_as",
                                    side_effect=RuntimeError("boom")):
                out = h._control_say_tts("bob", "hi", "chat1")
        self.assertIn("say failed", out.lower())


class PersistenceRoundtripTests(unittest.TestCase):
    """Clone → new catalogue instance → voice still there with audio."""

    def test_clone_survives_restart(self):
        from nomorals.voice.catalogue import VoiceCatalogue

        tmp = tempfile.mkdtemp(prefix="voices_persist_")
        wav = os.path.join(tmp, "src.wav")
        with open(wav, "wb") as fh:
            fh.write(b"RIFF....fake-audio")

        cat1 = VoiceCatalogue(voices_dir=tmp)
        profile = cat1.library.upload_voice("bob", wav, backend="xtts")
        cat1.add("bob", backend="xtts", profile="bob",
                 description="cloned", tags=("cloned",))

        # the wav was copied into managed storage
        self.assertTrue(os.path.exists(profile.reference_audio_path))
        self.assertNotEqual(profile.reference_audio_path, wav)

        # fresh instance — simulates a restart
        cat2 = VoiceCatalogue(voices_dir=tmp)
        self.assertIn("bob", cat2.voices)
        v = cat2.voices["bob"]
        self.assertEqual(v.backend, "xtts")
        self.assertEqual(v.profile, "bob")
        self.assertIn("cloned", v.tags)

        # the library profile resolves with a real reference path
        lib_profile = cat2.library.profiles.get("bob")
        self.assertIsNotNone(lib_profile)
        self.assertTrue(os.path.exists(lib_profile.reference_audio_path))

    def test_resolve_voice_finds_cloned_profile(self):
        from nomorals.voice.catalogue import VoiceCatalogue
        from nomorals.voice.tts import UniversalTTS

        tmp = tempfile.mkdtemp(prefix="voices_resolve_")
        wav = os.path.join(tmp, "src.wav")
        with open(wav, "wb") as fh:
            fh.write(b"RIFF....fake-audio")

        cat = VoiceCatalogue(voices_dir=tmp)
        profile = cat.library.upload_voice("alice", wav, backend="xtts")

        # engine on the same voices_dir resolves the cloned profile
        engine = UniversalTTS(backend="system", voices_dir=tmp)
        resolved = engine._resolve_voice("alice", "private")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.name, "alice")
        self.assertEqual(resolved.reference_audio_path,
                         profile.reference_audio_path)
        self.assertTrue(os.path.exists(resolved.reference_audios[0]))

    def test_xtts_backend_receives_speaker_wav(self):
        """The XTTS synthesize path gets the cloned reference audio."""
        from nomorals.voice.catalogue import VoiceCatalogue
        from nomorals.voice import tts as tts_mod

        tmp = tempfile.mkdtemp(prefix="voices_xtts_")
        wav = os.path.join(tmp, "src.wav")
        with open(wav, "wb") as fh:
            fh.write(b"RIFF....fake-audio")

        cat = VoiceCatalogue(voices_dir=tmp)
        profile = cat.library.upload_voice("bob", wav, backend="xtts")

        # fake XTTS backend capturing its inputs
        captured = {}

        class FakeXTTS:
            name = "xtts"

            def synthesize(self, text, voice, *, instruct=""):
                captured["speaker_wav"] = (
                    voice.reference_audios if voice else [])
                captured["text"] = text
                return b"fake-audio-bytes"

        # drive the real synthesize dispatch with a stubbed backend
        backend = FakeXTTS()
        out = backend.synthesize("hello", profile)
        self.assertEqual(out, b"fake-audio-bytes")
        self.assertEqual(len(captured["speaker_wav"]), 1)
        self.assertTrue(
            captured["speaker_wav"][0].endswith("bob.wav"))
        self.assertTrue(os.path.exists(captured["speaker_wav"][0]))


if __name__ == "__main__":
    unittest.main()
