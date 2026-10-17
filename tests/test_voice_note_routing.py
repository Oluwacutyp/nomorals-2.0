"""Voice note routed to MUSIC ID instead of STT — regression tests.

Incident: a girlfriend's Telegram voice note got the reply "hey,
what's the clip? if you want me to ID it, could you upload the audio
file here?" — the bot treated a spoken voice note as a music clip.

Root cause (verified in code, NOT an auto music-ID route — /sham is
explicit-only):
- Both Telegram adapters collapsed Telegram ``voice`` (spoken note)
  into kind="audio", indistinguishable from a music file.
- The transcript was produced up to 3x (_spoken_command discards
  non-command transcripts, pingpong re-transcribes, _media_notes
  re-transcribes again); a later transient failure lost a transcript
  that had already succeeded.
- On STT failure the brain fell back to the bare note
  "audio: voice.ogg" — a bare file reference the model read as "an
  audio file I don't have", so it hallucinated the music-ID ask.

Fix under test:
- ``voice`` keeps kind="voice" + is_voice=True; ``audio`` stays
  kind="audio"; a document audio file is never a voice note.
- Transcripts are cached on message.meta["voice_transcript"] and
  reused, never re-transcribed.
- Untranscribed voice/audio never reaches the model with a bare
  file reference: _media_notes labels it explicitly (SPOKEN, not a
  song), and _generate_and_persist short-circuits a text-less
  untranscribed voice note to one honest reply with no model call.
- /sham stays explicit-only, and still finds audio on a voice note
  that carries a /sham caption.
"""

from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

from nomorals.agents.partner.brain import PartnerBrain
from nomorals.social.chat.base import (
    ChatKind, ChatMessage, ChatRef, MediaRef)
from nomorals.social.chat.telegram import TelegramBotAdapter
from nomorals.voice import pingpong as pp


class _FailingTools:
    def __init__(self, error: str = "no STT backend installed") -> None:
        self.error = error
        self.calls = 0

    def call(self, name: str, **kwargs: Any) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(ok=False, error=self.error, value={})


class _WorkingTools:
    def __init__(self, text: str = "hello there") -> None:
        self.text = text
        self.calls = 0

    def call(self, name: str, **kwargs: Any) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(ok=True, value={"text": self.text})


def _msg(*, text: str = "", media: list | None = None,
         meta: dict | None = None) -> ChatMessage:
    return ChatMessage(
        chat=ChatRef(platform="telegram", chat_id="999", kind=ChatKind.DM),
        incoming=True, text=text, sender="stranger", sender_id="999",
        media=list(media or []), meta=dict(meta or {}))


def _voice_media(path: str = "/tmp/note.ogg") -> MediaRef:
    return MediaRef(path=path, kind="voice", mime="audio/ogg",
                    name="voice.ogg", is_voice=True)


class MediaRefVoiceFlagTest(unittest.TestCase):
    def test_to_dict_carries_is_voice(self) -> None:
        ref = _voice_media()
        self.assertTrue(ref.to_dict()["is_voice"])
        self.assertFalse(MediaRef(path="/tmp/x.mp3", kind="audio").to_dict()["is_voice"])


class BotApiVoiceVsAudioTest(unittest.TestCase):
    def _adapter(self, tmp: str, *, fail_getfile: bool = False,
                 fail_download: bool = False) -> TelegramBotAdapter:
        adapter = TelegramBotAdapter(token="x" * 40, media_dir=tmp)
        if fail_getfile:
            def _boom(**kw: Any) -> Any:
                raise RuntimeError("getFile down")
            adapter._api = _boom  # type: ignore[method-assign]
        else:
            adapter._api = lambda name, **kw: {  # type: ignore[method-assign]
                "file_size": 1234, "file_path": "voice/note.ogg"}
        resp = SimpleNamespace(content=b"OGGDATA")
        resp.raise_for_status = lambda: None
        if fail_download:
            resp.raise_for_status = lambda: (_ for _ in ()).throw(
                RuntimeError("download down"))
        adapter._sess = lambda: SimpleNamespace(get=lambda *a, **k: resp)  # type: ignore[method-assign]
        return adapter

    def test_voice_keeps_voice_kind_and_flag(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-vn-") as tmp:
            media = self._adapter(tmp)._inbound_media(
                {"voice": {"file_id": "v1"}}, ChatKind.DM)
        self.assertEqual(len(media), 1)
        self.assertEqual(media[0].kind, "voice")
        self.assertTrue(media[0].is_voice)
        self.assertEqual(media[0].mime, "audio/ogg")
        self.assertNotEqual(media[0].kind, "audio")

    def test_audio_stays_audio_not_voice(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-vn-") as tmp:
            media = self._adapter(tmp)._inbound_media(
                {"audio": {"file_id": "a1", "mime_type": "audio/mpeg"}},
                ChatKind.DM)
        self.assertEqual(media[0].kind, "audio")
        self.assertFalse(media[0].is_voice)

    def test_document_audio_is_not_a_voice_note(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-vn-") as tmp:
            media = self._adapter(tmp)._inbound_media(
                {"document": {"file_id": "d1", "file_name": "song.mp3",
                              "mime_type": "audio/mpeg"}}, ChatKind.DM)
        self.assertEqual(media[0].kind, "audio")
        self.assertFalse(media[0].is_voice)

    def test_getfile_failure_returns_placeholder_not_empty(self) -> None:
        # The message DID carry media — [] would make downstream think
        # there was no attachment at all.
        with tempfile.TemporaryDirectory(prefix="nm-vn-") as tmp:
            media = self._adapter(tmp, fail_getfile=True)._inbound_media(
                {"voice": {"file_id": "v1"}}, ChatKind.DM)
        self.assertEqual(len(media), 1)
        self.assertEqual(media[0].path, "")
        self.assertEqual(media[0].kind, "voice")
        self.assertTrue(media[0].is_voice)

    def test_download_failure_returns_placeholder_not_empty(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-vn-") as tmp:
            media = self._adapter(tmp, fail_download=True)._inbound_media(
                {"voice": {"file_id": "v1"}}, ChatKind.DM)
        self.assertEqual(len(media), 1)
        self.assertEqual(media[0].path, "")
        self.assertEqual(media[0].kind, "voice")


class TranscriptCacheTest(unittest.TestCase):
    def _brain(self, tools: Any) -> Any:
        brain = PartnerBrain.__new__(PartnerBrain)
        brain.context = SimpleNamespace(tools=tools)
        brain._last_stt_failure_reason = ""
        return brain

    def test_spoken_command_caches_non_command_transcript(self) -> None:
        brain = self._brain(_WorkingTools("hello there"))
        msg = _msg(media=[_voice_media()])
        self.assertEqual("", brain._spoken_command(msg))
        self.assertEqual(msg.meta.get("voice_transcript"), "hello there")

    def test_spoken_command_still_returns_command(self) -> None:
        brain = self._brain(_WorkingTools("/status"))
        msg = _msg(media=[_voice_media()])
        self.assertEqual("/status", brain._spoken_command(msg))
        self.assertEqual(msg.meta.get("voice_transcript"), "/status")

    def test_pingpong_reuses_cached_transcript_without_calling(self) -> None:
        tools = _FailingTools("backend died after first success")
        msg = _msg(media=[_voice_media()],
                   meta={"voice_transcript": "hello there"})
        text, reason = pp.transcribe_voice_media_with_reason(
            SimpleNamespace(tools=tools), msg)
        self.assertEqual(text, "hello there")
        self.assertEqual(reason, "")
        self.assertEqual(tools.calls, 0)

    def test_pingpong_caches_fresh_transcript(self) -> None:
        tools = _WorkingTools("hello there")
        msg = _msg(media=[_voice_media()])
        text, _ = pp.transcribe_voice_media_with_reason(
            SimpleNamespace(tools=tools), msg)
        self.assertEqual(text, "hello there")
        self.assertEqual(msg.meta.get("voice_transcript"), "hello there")


class MediaNotesTest(unittest.TestCase):
    def _brain(self, tools: Any) -> Any:
        brain = PartnerBrain.__new__(PartnerBrain)
        brain.context = SimpleNamespace(tools=tools)
        brain._last_stt_failure_reason = ""
        return brain

    def test_voice_stt_failure_never_bare_audio_note(self) -> None:
        brain = self._brain(_FailingTools("no STT backend installed"))
        msg = _msg(media=[_voice_media()])
        notes = brain._media_notes(msg)
        joined = " ".join(notes)
        self.assertNotIn("audio: voice.ogg", joined)
        self.assertIn("SPOKEN", joined)
        self.assertIn("not a song", joined)
        self.assertIn("do NOT ask to identify", joined)
        self.assertIn("do NOT ask for re-upload", joined)
        self.assertEqual(msg.meta.get("voice_untranscribed_reason"),
                         "no STT backend")

    def test_audio_file_stt_failure_never_assumes_music(self) -> None:
        brain = self._brain(_FailingTools("no STT backend installed"))
        msg = _msg(media=[MediaRef(path="/tmp/song.mp3", kind="audio",
                                   mime="audio/mpeg", name="song.mp3")])
        notes = brain._media_notes(msg)
        joined = " ".join(notes)
        self.assertIn("do NOT assume it is music", joined)
        self.assertIn("/sham", joined)

    def test_cached_transcript_used_no_tool_call(self) -> None:
        tools = _FailingTools("must not be called")
        brain = self._brain(tools)
        msg = _msg(media=[_voice_media()],
                   meta={"voice_transcript": "hello there"})
        notes = brain._media_notes(msg)
        self.assertIn("hello there", " ".join(notes))
        self.assertEqual(tools.calls, 0)
        self.assertNotIn("voice_untranscribed_reason", msg.meta)

    def test_transcribe_success_caches_transcript(self) -> None:
        brain = self._brain(_WorkingTools("hello there"))
        msg = _msg(media=[_voice_media()])
        notes = brain._media_notes(msg)
        self.assertIn("hello there", " ".join(notes))
        self.assertEqual(msg.meta.get("voice_transcript"), "hello there")

    def test_empty_path_placeholder_is_no_audio_file(self) -> None:
        brain = self._brain(_FailingTools("must not be called"))
        msg = _msg(media=[MediaRef(path="", kind="voice", mime="audio/ogg",
                                   name="voice.ogg", is_voice=True)])
        notes = brain._media_notes(msg)
        self.assertIn("no audio file", " ".join(notes))
        self.assertEqual(msg.meta.get("voice_untranscribed_reason"),
                         "no audio file")


class BrainGuardTest(unittest.TestCase):
    """_generate_and_persist: untranscribed + text-less voice note gets
    ONE honest reply with NO model call; transcribed notes flow on."""

    def _brain(self, tools: Any) -> Any:
        brain = PartnerBrain.__new__(PartnerBrain)
        brain.context = SimpleNamespace(tools=tools, memory=None, db=None)
        brain._last_stt_failure_reason = ""
        brain.settings = SimpleNamespace(
            partner=SimpleNamespace(gate_restricted_chats=True,
                                    history_window=10))
        brain.responder = mock.Mock()
        brain.responder.recall.return_value = ()
        brain.responder.respond_bounded.side_effect = AssertionError(
            "model must not be called for an untranscribed voice note")
        brain.background = mock.Mock()
        brain.background.context.return_value = []
        brain.relationship = mock.Mock()
        brain._history = lambda *a, **k: []
        brain._persist_outbound = mock.Mock()
        brain._note_reply = mock.Mock()
        brain._log_training_pair = mock.Mock()
        brain._maybe_curate = mock.Mock()
        brain._maybe_extract = mock.Mock()
        brain._refresh_profile_async = mock.Mock()
        brain.generate_with_tools = mock.Mock(
            side_effect=AssertionError("tool loop must not run"))
        return brain

    def test_untranscribed_voice_note_never_reaches_model(self) -> None:
        brain = self._brain(_FailingTools("no STT backend installed"))
        msg = _msg(media=[_voice_media()])
        parts = brain._generate_and_persist(msg, {"is_owner": False, "in_us": False})
        joined = " ".join(parts)
        self.assertIn("couldn't transcribe that voice note", joined)
        self.assertIn("no STT backend", joined)
        self.assertNotIn("clip", joined.lower())
        self.assertNotIn("upload the audio", joined.lower())
        self.assertNotIn("ID it", joined)
        brain.responder.respond_bounded.assert_not_called()

    def test_untranscribed_audio_file_never_reaches_model(self) -> None:
        brain = self._brain(_FailingTools("no STT backend installed"))
        msg = _msg(media=[MediaRef(path="/tmp/song.mp3", kind="audio",
                                   mime="audio/mpeg", name="song.mp3")])
        parts = brain._generate_and_persist(msg, {"is_owner": False, "in_us": False})
        self.assertIn("couldn't transcribe that audio file", " ".join(parts))
        brain.responder.respond_bounded.assert_not_called()

    def test_transcribed_voice_note_flows_to_model(self) -> None:
        brain = self._brain(_WorkingTools("hello there"))
        brain.responder.respond_bounded.side_effect = None
        brain.responder.respond_bounded.return_value = SimpleNamespace(
            parts=["heard: hello there"], model="test",
            degraded=False, degraded_note="")
        msg = _msg(media=[_voice_media()])
        parts = brain._generate_and_persist(msg, {"is_owner": False, "in_us": False})
        self.assertIn("heard: hello there", parts)
        brain.responder.respond_bounded.assert_called_once()

    def test_text_plus_untranscribed_voice_note_flows_to_model(self) -> None:
        # User typed text WITH the voice note: the text is the message,
        # the model gets the explicit SPOKEN note — no short-circuit.
        brain = self._brain(_FailingTools("no STT backend installed"))
        brain.responder.respond_bounded.side_effect = None
        brain.responder.respond_bounded.return_value = SimpleNamespace(
            parts=["got your text"], model="test",
            degraded=False, degraded_note="")
        msg = _msg(text="also see this", media=[_voice_media()])
        parts = brain._generate_and_persist(msg, {"is_owner": False, "in_us": False})
        self.assertIn("got your text", parts)
        media_notes = (brain.responder.respond_bounded.call_args.kwargs
                       .get("media_notes") or [])
        self.assertIn("SPOKEN", " ".join(media_notes))


class ShamExplicitOnlyTest(unittest.TestCase):
    def test_sham_finds_audio_on_voice_note_caption_flow(self) -> None:
        from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

        class T(RuntimeMediaMixin):
            def __init__(self) -> None:
                self.context = mock.Mock()
                self.gateway = None

        t = T()
        msg = mock.Mock()
        msg.media = [mock.Mock(kind="voice", path="/tmp/note.ogg")]
        msg.meta = {}
        self.assertEqual(t._sham_find_audio(msg), "/tmp/note.ogg")

    def test_sham_never_auto_runs_without_command(self) -> None:
        # No code path calls _sham_identify except the explicit /sham
        # control handler — assert the wiring, not just the absence.
        from nomorals.agents.partner import runtime_media as rm
        import inspect
        src = inspect.getsource(rm.RuntimeMediaMixin._control_sham)
        self.assertIn("_sham_identify", src)
        # And _sham_identify is not referenced from brain / runtime
        # generation paths.
        import pathlib
        brain_src = pathlib.Path(
            "nomorals/agents/partner/brain.py").read_text()
        self.assertNotIn("_sham_identify", brain_src)
        self.assertNotIn("_control_sham", brain_src)


if __name__ == "__main__":
    unittest.main()
