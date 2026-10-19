"""PASS #10: voice stack — verified bugs, failing-first.

Bug 1 (per-path transcript cache): ``meta["voice_transcript"]`` was a
single string per message, but ``_spoken_command`` loops over ALL media
and overwrote it — with two voice notes in one message, note 1 was
labeled with note 2's transcript. Transcripts are now cached per media
path (``meta["voice_transcripts"][path]``), legacy single key kept.

Bug 2 (STT reason noise): ``short_stt_reason("")`` returned
``"STT API error: str"`` and ``short_stt_reason(None)`` returned
``"STT API error: NoneType"`` — the never-empty rule named the class,
but for non-exceptions the class name is noise. Non-exceptions with no
text now stay the bare generic; empty-message EXCEPTIONS still name
their class (never-empty rule holds).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from nomorals.agents.partner.brain import PartnerBrain
from nomorals.core.errors import short_stt_reason
from nomorals.social.chat.base import (
    ChatKind, ChatMessage, ChatRef, MediaRef)
from nomorals.voice import pingpong as pp


class _Tools:
    """STT stub: different transcript per path."""

    def __init__(self) -> None:
        self.calls: list = []

    def call(self, name: str, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs.get("path"))
        path = kwargs.get("path", "")
        text = ("FIRST note words" if "note1" in path
                else "SECOND note words")
        return SimpleNamespace(ok=True, value={"text": text})


def _brain(tools) -> PartnerBrain:
    brain = PartnerBrain.__new__(PartnerBrain)
    brain.context = SimpleNamespace(tools=tools)
    brain._last_stt_failure_reason = ""
    return brain


def _msg(media, meta=None) -> ChatMessage:
    return ChatMessage(
        chat=ChatRef(platform="telegram", chat_id="999", kind=ChatKind.DM),
        incoming=True, text="", sender="s", sender_id="999",
        media=list(media), meta=dict(meta or {}))


def _voice(path: str) -> MediaRef:
    return MediaRef(path=path, kind="voice", mime="audio/ogg",
                    name="voice.ogg", is_voice=True)


class PerPathTranscriptCacheTest(unittest.TestCase):
    def test_two_voice_notes_keep_their_own_transcripts(self) -> None:
        # Failed pre-fix: both notes got "SECOND note words".
        tools = _Tools()
        brain = _brain(tools)
        msg = _msg([_voice("/tmp/note1.ogg"), _voice("/tmp/note2.ogg")])
        brain._spoken_command(msg)  # pipeline stage 1: transcribes + caches
        notes = brain._media_notes(msg)  # stage 2: reuses cache
        self.assertEqual(2, len(notes))
        self.assertIn("FIRST note words", notes[0])
        self.assertNotIn("SECOND", notes[0])
        self.assertIn("SECOND note words", notes[1])
        self.assertNotIn("FIRST", notes[1])
        # No double transcription: each path called exactly once.
        self.assertEqual(["/tmp/note1.ogg", "/tmp/note2.ogg"], tools.calls)

    def test_single_note_legacy_key_still_set(self) -> None:
        tools = _Tools()
        brain = _brain(tools)
        msg = _msg([_voice("/tmp/note1.ogg")])
        brain._spoken_command(msg)
        # Legacy single-key readers (truthiness checks) keep working.
        self.assertEqual("FIRST note words",
                         msg.meta.get("voice_transcript"))
        # ... and the per-path entry exists too.
        self.assertEqual("FIRST note words",
                         msg.meta["voice_transcripts"]["/tmp/note1.ogg"])

    def test_pingpong_reuses_per_path_cache(self) -> None:
        tools = _Tools()
        ctx = SimpleNamespace(tools=tools)
        msg = _msg([_voice("/tmp/note1.ogg"), _voice("/tmp/note2.ogg")])
        # Simulate an earlier stage having cached only note1.
        pp.cache_transcript(msg.meta, "/tmp/note1.ogg", "FIRST note words")
        text, reason = pp.transcribe_voice_media_with_reason(ctx, msg)
        self.assertEqual("FIRST note words", text)
        self.assertEqual("", reason)
        # note1 served from cache — no STT call for it.
        self.assertNotIn("/tmp/note1.ogg", tools.calls)


class SttReasonEmptyInputTest(unittest.TestCase):
    def test_empty_string_stays_bare(self) -> None:
        # Failed pre-fix: "STT API error: str".
        self.assertEqual("STT API error", short_stt_reason(""))
        self.assertEqual("STT API error", pp._short_stt_reason(""))

    def test_none_stays_bare(self) -> None:
        # Failed pre-fix: "STT API error: NoneType".
        self.assertEqual("STT API error", short_stt_reason(None))

    def test_empty_exception_still_names_class(self) -> None:
        # Pin: the never-empty rule holds for real exceptions.
        self.assertEqual("STT API error: ValueError",
                         short_stt_reason(ValueError()))
        self.assertEqual("STT API error: ValueError",
                         pp._short_stt_reason(ValueError()))

    def test_twins_agree_on_empty_inputs(self) -> None:
        from nomorals.agents.partner.brain import (
            _short_stt_reason as brain_reason)
        for case in ("", None, ValueError(), "Groq timeout"):
            self.assertEqual(pp._short_stt_reason(case),
                             brain_reason(case))


if __name__ == "__main__":
    unittest.main()
