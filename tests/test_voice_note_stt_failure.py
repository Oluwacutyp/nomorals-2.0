"""Voice-note STT failure: honest reply in normal chats, silence in
games-only, plus the games-only voice-note help/status text and the
vision-flag audio preservation (Gap 3).

Covers:
- (A) /chatmode help + status/set outputs state that games-only
  silences voice notes (even after transcription).
- (B) a voice note whose transcription fails gets ONE honest
  "couldn't transcribe" reply in a normal chat, silence in a
  games-only chat, and the normal flow when transcription succeeds.
- Gap 3: vision-off strips photos but preserves voice notes so STT
  still runs.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.partner import chat_modes
from nomorals.social.chat.base import (
    ChatKind, ChatMessage, ChatRef, MediaRef)
from nomorals.social.chat.control import COMMAND_DETAILS, detailed_help
from nomorals.voice import pingpong as pp

VOICE_NOTE_STATEMENT = (
    "including voice notes (even after transcription)")


def _msg(text: str = "", *, chat_key: str = "telegram:999",
         owner: bool = False, media: list | None = None) -> ChatMessage:
    platform, _, chat_id = chat_key.partition(":")
    return ChatMessage(
        chat=ChatRef(platform=platform or "telegram",
                     chat_id=chat_id or "999", kind=ChatKind.DM),
        incoming=True, text=text,
        sender="owner" if owner else "stranger",
        sender_id="5478650254" if owner else "999",
        meta={"is_owner": True} if owner else {},
        media=list(media or []),
    )


def _voice_msg() -> ChatMessage:
    return _msg(media=[MediaRef(path="/tmp/note.ogg", kind="voice",
                                mime="audio/ogg", name="note.ogg")])


class _RecordingGateway:
    adapters: dict = {}

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def status(self) -> dict:
        return {}

    def stop(self, *a: Any, **k: Any) -> None:
        pass

    def set_rate_limit(self, *a: Any, **k: Any) -> None:
        pass

    def typing(self, *a: Any, **k: Any) -> None:
        pass

    def send(self, platform: str, chat: Any, text: str,
             **kwargs: Any) -> SimpleNamespace:
        self.sent.append((platform, text))
        return SimpleNamespace(ok=True, error="")


class _FailingTools:
    """Transcribe always fails the way a missing backend does."""

    def __init__(self, error: str = "no STT backend installed") -> None:
        self.error = error
        self.calls: list[str] = []

    def call(self, name: str, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(name)
        return SimpleNamespace(ok=False, error=self.error, value={})


class _WorkingTools:
    """Transcribe always succeeds."""

    def __init__(self, text: str = "hello there") -> None:
        self.text = text

    def call(self, name: str, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(ok=True, value={"text": self.text})


class ChatmodeVoiceNoteTextTest(unittest.TestCase):
    """(A) help + status outputs state the games-only voice-note rule."""

    def test_help_states_voice_notes_silenced(self) -> None:
        self.assertIn(VOICE_NOTE_STATEMENT,
                      COMMAND_DETAILS["chatmode"]["what"])
        self.assertIn(VOICE_NOTE_STATEMENT, detailed_help("chatmode"))

    def test_status_and_set_state_voice_notes_silenced(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="nm-chatmode-vn-")
        self.addCleanup(tmp.cleanup)
        modes_path = os.path.join(tmp.name, "chat_modes.json")
        old = os.environ.get("NOMORALS_CHAT_MODES_PATH")
        os.environ["NOMORALS_CHAT_MODES_PATH"] = modes_path
        self.addCleanup(
            lambda: (os.environ.pop("NOMORALS_CHAT_MODES_PATH", None)
                     if old is None else
                     os.environ.__setitem__("NOMORALS_CHAT_MODES_PATH", old)))
        settings = load_settings(overrides={
            "home": tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
        })
        context = build_context(settings, with_executor=False,
                                with_tools=False)

        class _Router:
            def chat(self, *a: Any, **k: Any) -> Any:
                raise AssertionError("test must not call the model")

        context.router = _Router()
        runtime = PartnerRuntime(context, gateway=_RecordingGateway())
        self.addCleanup(runtime.stop)
        self.addCleanup(context.close)
        out = runtime.handle_control(
            "/chatmode games-only", "telegram:999",
            message=_msg("/chatmode games-only", owner=True))
        self.assertIn(VOICE_NOTE_STATEMENT, out)
        out = runtime.handle_control(
            "/chatmode status", "telegram:999",
            message=_msg("/chatmode status", owner=True))
        self.assertIn(VOICE_NOTE_STATEMENT, out)
        self.assertIn("games-only", out)


class TranscribeWithReasonTest(unittest.TestCase):
    def _media_msg(self, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            media=[SimpleNamespace(kind="voice", path="/tmp/note.ogg",
                                   mime="audio/ogg")],
            meta={}, **kwargs)

    def test_success_returns_text_no_reason(self) -> None:
        ctx = SimpleNamespace(tools=_WorkingTools("hello devon"))
        text, reason = pp.transcribe_voice_media_with_reason(
            ctx, self._media_msg())
        self.assertEqual(text, "hello devon")
        self.assertEqual(reason, "")

    def test_backend_failure_reason(self) -> None:
        ctx = SimpleNamespace(tools=_FailingTools("no STT backend installed"))
        msg = self._media_msg()
        text, reason = pp.transcribe_voice_media_with_reason(ctx, msg)
        self.assertEqual(text, "")
        self.assertEqual(reason, "no STT backend")
        self.assertEqual(msg.meta.get("stt_failure_reason"), "no STT backend")

    def test_api_error_reason(self) -> None:
        ctx = SimpleNamespace(tools=_FailingTools("Groq API error 500"))
        text, reason = pp.transcribe_voice_media_with_reason(
            ctx, self._media_msg())
        self.assertEqual(reason, "STT API error")

    def test_exception_reason(self) -> None:
        class _Boom:
            def call(self, *a: Any, **k: Any) -> Any:
                raise RuntimeError("Groq timeout")

        text, reason = pp.transcribe_voice_media_with_reason(
            SimpleNamespace(tools=_Boom()), self._media_msg())
        self.assertEqual(text, "")
        self.assertEqual(reason, "STT API error")

    def test_empty_transcript_reason(self) -> None:
        ctx = SimpleNamespace(tools=_WorkingTools(""))
        text, reason = pp.transcribe_voice_media_with_reason(
            ctx, self._media_msg())
        self.assertEqual(text, "")
        self.assertEqual(reason, "empty transcript")

    def test_no_media_no_reason(self) -> None:
        ctx = SimpleNamespace(tools=_FailingTools())
        text, reason = pp.transcribe_voice_media_with_reason(
            ctx, SimpleNamespace(media=[], meta={}))
        self.assertEqual((text, reason), ("", ""))

    def test_legacy_wrapper_still_returns_text_only(self) -> None:
        ctx = SimpleNamespace(tools=_WorkingTools("hi"))
        self.assertEqual("hi", pp.transcribe_voice_media(ctx, self._media_msg()))
        ctx = SimpleNamespace(tools=_FailingTools())
        self.assertEqual(
            "", pp.transcribe_voice_media(ctx, self._media_msg()))


class BrainTranscribeWithReasonTest(unittest.TestCase):
    def _brain(self, tools: Any) -> Any:
        from nomorals.agents.partner_runtime import PartnerBrain

        brain = PartnerBrain.__new__(PartnerBrain)
        brain.context = SimpleNamespace(tools=tools)
        brain._last_stt_failure_reason = ""
        return brain

    def test_success(self) -> None:
        brain = self._brain(_WorkingTools("hello"))
        note, reason = brain._transcribe_media_note_with_reason(
            SimpleNamespace(path="/tmp/note.ogg"))
        self.assertIn("hello", note)
        self.assertEqual(reason, "")

    def test_failure_reason(self) -> None:
        brain = self._brain(_FailingTools("no STT backend installed"))
        note, reason = brain._transcribe_media_note_with_reason(
            SimpleNamespace(path="/tmp/note.ogg"))
        self.assertEqual(note, "")
        self.assertEqual(reason, "no STT backend")

    def test_plain_wrapper_unchanged(self) -> None:
        brain = self._brain(_WorkingTools("hello"))
        self.assertIn(
            "hello",
            brain._transcribe_media_note(SimpleNamespace(path="/tmp/note.ogg")))
        brain = self._brain(_FailingTools())
        self.assertEqual(
            "", brain._transcribe_media_note(SimpleNamespace(path="/tmp/note.ogg")))

    def test_spoken_command_records_reason_on_meta(self) -> None:
        brain = self._brain(_FailingTools("Groq API error"))
        msg = _voice_msg()
        self.assertEqual("", brain._spoken_command(msg))
        self.assertEqual(msg.meta.get("stt_failure_reason"), "STT API error")


class SttFailureProcessTest(unittest.TestCase):
    """(B) _process: honest reply / games-only silence / success flow."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-stt-")
        self.modes_path = os.path.join(self.tmp.name, "chat_modes.json")
        self._old = os.environ.get("NOMORALS_CHAT_MODES_PATH")
        os.environ["NOMORALS_CHAT_MODES_PATH"] = self.modes_path
        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
        })
        self.context = build_context(settings, with_executor=False,
                                     with_tools=False)

        class _Router:
            def chat(self, *a: Any, **k: Any) -> Any:
                raise AssertionError("test must not call the model")

        self.context.router = _Router()
        self.runtime = PartnerRuntime(self.context,
                                      gateway=_RecordingGateway())
        self.gateway = self.runtime.gateway
        assert isinstance(self.gateway, _RecordingGateway)

    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception:
            pass
        self.context.close()
        if self._old is None:
            os.environ.pop("NOMORALS_CHAT_MODES_PATH", None)
        else:
            os.environ["NOMORALS_CHAT_MODES_PATH"] = self._old
        self.tmp.cleanup()

    def _process(self, msg: ChatMessage) -> None:
        with mock.patch.object(
                self.runtime, "_route_game_move", return_value=None), \
                mock.patch.object(
                    self.runtime.mind, "handle", return_value=None):
            self.runtime._process(msg)

    def test_failure_in_normal_chat_gets_one_honest_reply(self) -> None:
        self.context.tools = _FailingTools("no STT backend installed")
        self._process(_voice_msg())
        self.assertEqual(len(self.gateway.sent), 1)
        self.assertIn("couldn't transcribe that voice note",
                      self.gateway.sent[0][1])
        self.assertIn("no STT backend", self.gateway.sent[0][1])

    def test_failure_in_games_only_chat_is_silent(self) -> None:
        chat_modes.set_chat_mode("telegram:999", "games_only")
        self.context.tools = _FailingTools("no STT backend installed")
        self._process(_voice_msg())
        self.assertEqual(self.gateway.sent, [])

    def test_success_uses_transcript_no_honest_reply(self) -> None:
        self.context.tools = _WorkingTools("hello there")
        seen: list[ChatMessage] = []

        def _handle(message: ChatMessage) -> SimpleNamespace:
            seen.append(message)
            return SimpleNamespace(parts=[], presence=SimpleNamespace(
                delay_seconds=0))

        with mock.patch.object(self.runtime.brain, "handle_message",
                               side_effect=_handle):
            self._process(_voice_msg())
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].text, "hello there")
        self.assertFalse(any("couldn't transcribe" in t
                             for _, t in self.gateway.sent))

    def test_plain_text_never_gets_stt_reply(self) -> None:
        self.context.tools = _FailingTools()
        seen: list[ChatMessage] = []

        def _handle(message: ChatMessage) -> SimpleNamespace:
            seen.append(message)
            return SimpleNamespace(parts=[], presence=SimpleNamespace(
                delay_seconds=0))

        with mock.patch.object(self.runtime.brain, "handle_message",
                               side_effect=_handle):
            self._process(_msg("just typed text"))
        self.assertEqual(len(seen), 1)
        self.assertEqual(self.gateway.sent, [])

    def test_vision_off_preserves_voice_note(self) -> None:
        from nomorals.agents.features import FeatureRegistry

        FeatureRegistry(self.context.db).set("vision", False)
        self.context.tools = _FailingTools("no STT backend installed")
        self._process(_voice_msg())
        # Audio survived the vision-off strip, STT ran and failed
        # honestly — instead of the note vanishing with the media.
        self.assertEqual(len(self.gateway.sent), 1)
        self.assertIn("couldn't transcribe", self.gateway.sent[0][1])

    def test_vision_off_strips_photo(self) -> None:
        from nomorals.agents.features import FeatureRegistry

        FeatureRegistry(self.context.db).set("vision", False)
        self.context.tools = _WorkingTools("hello")
        seen: list[ChatMessage] = []

        def _handle(message: ChatMessage) -> SimpleNamespace:
            seen.append(message)
            return SimpleNamespace(parts=[], presence=SimpleNamespace(
                delay_seconds=0))

        photo = _msg(media=[MediaRef(path="/tmp/photo.jpg", kind="image",
                                     mime="image/jpeg", name="photo.jpg")])
        with mock.patch.object(self.runtime.brain, "handle_message",
                               side_effect=_handle):
            self._process(photo)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].media, [])


if __name__ == "__main__":
    unittest.main()
