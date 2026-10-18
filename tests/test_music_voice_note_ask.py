"""Music voice note → ASK "find or create" (owner-only).

Covers the owner decision (final): voice notes WORK (transcribe +
reply) — normal voice-note handling is unchanged. Music in a voice
note → Devon ASKS "Find this music (like Shazam) or create a new
song from it?" with two inline buttons. The ask is OWNER-ONLY:
non-owner DMs get no ask, no buttons — Devon decides (music as
context, reply channel her choice).

* detection is conservative: only a clip the native analyzer scores
  as clearly music counts; speech / uncertain / silence / blips
  never get the ask and fall through to the honest STT reply.
* owner music note → ask with BOTH buttons (callbacks routable in
  CONTROL_COMMANDS, inside the 47-byte budget); no brain turn.
* tapping Find → music-ID runs on the stashed clip; tapping Create
  → production runs; either tap consumes the pending note.
* non-owner DM music note → no ask, no buttons; the brain gets the
  note as music context and decides its own reply.
* speech voice notes → never asked, normal transcribe + reply.
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
from nomorals.social.chat.control import (
    COMMAND_DETAILS, CONTROL_COMMANDS, GAME_COMMANDS, LIST_GROUPS,
    PUBLIC_COMMANDS, detailed_help, list_catalog, parse_control)
from nomorals.voice import music_note as mn
from nomorals.voice import pingpong as pp  # noqa: F401  (keeps parity with sibling suites)


def _analysis(*, ok: bool = True, score: float = 0.8,
              seconds: float = 12.0, silence: float = 0.1,
              bpm: float = 120.0) -> SimpleNamespace:
    return SimpleNamespace(
        ok=ok, music_score=score, seconds=seconds,
        silence_ratio=silence, tempo_bpm=bpm, tempo_confidence=0.9,
        key="C", key_confidence=0.8, rms_db=-18.0)


class DetectMusicTest(unittest.TestCase):
    def test_clear_music_counts(self) -> None:
        res = mn.detect_music("/tmp/clip.ogg",
                              analyze=lambda p: _analysis())
        self.assertTrue(res["is_music"])
        self.assertIn("12s of music", res["heard"])
        self.assertIn("120 BPM", res["heard"])
        self.assertAlmostEqual(res["score"], 0.8)

    def test_speech_never_music(self) -> None:
        res = mn.detect_music("/tmp/clip.ogg",
                              analyze=lambda p: _analysis(score=0.2))
        self.assertFalse(res["is_music"])
        self.assertEqual(res["heard"], "")

    def test_uncertain_mixed_never_music(self) -> None:
        # 0.4–0.6 is the analyzer's "mixed audio" band: uncertain
        # means speech handling, never the ask.
        for score in (0.41, 0.5, 0.59):
            res = mn.detect_music("/tmp/clip.ogg",
                                  analyze=lambda p, s=score: _analysis(score=s))
            self.assertFalse(res["is_music"], f"score {score}")

    def test_mostly_silence_never_music(self) -> None:
        res = mn.detect_music("/tmp/clip.ogg",
                              analyze=lambda p: _analysis(silence=0.8))
        self.assertFalse(res["is_music"])

    def test_too_short_blip_never_music(self) -> None:
        res = mn.detect_music("/tmp/clip.ogg",
                              analyze=lambda p: _analysis(seconds=0.4))
        self.assertFalse(res["is_music"])

    def test_analyze_failure_never_music_never_raises(self) -> None:
        res = mn.detect_music("/tmp/clip.ogg",
                              analyze=lambda p: _analysis(ok=False))
        self.assertFalse(res["is_music"])
        res = mn.detect_music("/tmp/clip.ogg",
                              analyze=lambda p: (_ for _ in ()).throw(
                                  RuntimeError("dsp down")))
        self.assertFalse(res["is_music"])
        res = mn.detect_music("")
        self.assertFalse(res["is_music"])

    def test_heard_line_format(self) -> None:
        res = mn.detect_music("/tmp/clip.ogg",
                              analyze=lambda p: _analysis(
                                  seconds=7.0, bpm=0.0))
        self.assertIn("7s of music", res["heard"])


class AskButtonsTest(unittest.TestCase):
    def _all_callbacks_valid(self, kb: Any) -> None:
        for row in kb or []:
            for _label, data in row:
                kind = (data or "").lstrip("/").split(None, 1)[0].lower()
                self.assertIn(kind, CONTROL_COMMANDS,
                              f"dead button callback: {data!r}")
                self.assertLessEqual(len(data.encode("utf-8")), 47)

    def test_adaptive_music_ask_has_both_buttons(self) -> None:
        from nomorals.social.chat.adaptive_buttons import adaptive_buttons_for

        kb = adaptive_buttons_for(output_kind="music-ask")
        self.assertIsNotNone(kb)
        flat = {label: data for row in (kb or []) for label, data in row}
        self.assertEqual(flat["🔍 Find this music"], "musicnote find")
        self.assertEqual(flat["🎵 Create a song"], "musicnote create")
        self.assertEqual(len(flat), 2)
        self._all_callbacks_valid(kb)

    def test_ask_buttons_match_adaptive(self) -> None:
        from nomorals.social.chat.adaptive_buttons import adaptive_buttons_for

        self.assertEqual(mn.ask_buttons(),
                         adaptive_buttons_for(output_kind="music-ask"))

    def test_ask_text_is_the_question(self) -> None:
        self.assertIn("Find this music", mn.ASK_TEXT)
        self.assertIn("create a new song", mn.ASK_TEXT)
        self.assertIn("🎵", mn.ASK_TEXT)


class PendingStoreTest(unittest.TestCase):
    def test_take_consumes(self) -> None:
        store = mn.PendingMusicNote()
        store.remember("telegram:1", "/tmp/a.ogg", "12s of music")
        first = store.take("telegram:1")
        self.assertEqual(first["path"], "/tmp/a.ogg")
        self.assertIsNone(store.take("telegram:1"))

    def test_peek_does_not_consume(self) -> None:
        store = mn.PendingMusicNote()
        store.remember("telegram:1", "/tmp/a.ogg")
        self.assertIsNotNone(store.peek("telegram:1"))
        self.assertIsNotNone(store.take("telegram:1"))

    def test_expired_entry_is_gone(self) -> None:
        store = mn.PendingMusicNote()
        store.remember("telegram:1", "/tmp/a.ogg")
        store._by_chat["telegram:1"]["ts"] -= mn.PENDING_TTL_S + 1
        self.assertIsNone(store.take("telegram:1"))
        self.assertIsNone(store.peek("telegram:1"))

    def test_new_note_replaces_old(self) -> None:
        store = mn.PendingMusicNote()
        store.remember("telegram:1", "/tmp/a.ogg")
        store.remember("telegram:1", "/tmp/b.ogg")
        self.assertEqual(store.take("telegram:1")["path"], "/tmp/b.ogg")

    def test_unknown_chat_is_none(self) -> None:
        store = mn.PendingMusicNote()
        self.assertIsNone(store.take("telegram:nope"))


# ── _process hook ────────────────────────────────────────────────────

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


def _voice_msg(*, owner: bool = False) -> ChatMessage:
    return _msg(owner=owner, media=[MediaRef(
        path="/tmp/note.ogg", kind="voice",
        mime="audio/ogg", name="note.ogg")])


class _RecordingGateway:
    adapters: dict = {}

    def __init__(self) -> None:
        self.sent: list[tuple[str, str, dict]] = []

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
        self.sent.append((platform, text, dict(kwargs)))
        return SimpleNamespace(ok=True, error="")


class _EmptyTranscriptTools:
    """STT succeeds with an empty transcript — what music yields."""

    def __init__(self) -> None:
        self.calls = 0

    def call(self, name: str, **kwargs: Any) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(ok=True, value={"text": ""})


class _WorkingTools:
    """STT succeeds with speech."""

    def __init__(self, text: str = "hello there") -> None:
        self.text = text

    def call(self, name: str, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(ok=True, value={"text": self.text})


class _FailingTools:
    """No STT backend."""

    def __init__(self, error: str = "no STT backend installed") -> None:
        self.error = error

    def call(self, name: str, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(ok=False, error=self.error, value={})


class MusicHookProcessTest(unittest.TestCase):
    """_process: owner music → ask; non-owner music → context, no ask;
    speech → never asked."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-music-ask-")
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
        mn.PENDING.clear()
        self.addCleanup(mn.PENDING.clear)

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

    def _music(self) -> dict[str, Any]:
        return {"is_music": True, "heard": "12s of music · ~120 BPM",
                "score": 0.8, "seconds": 12.0}

    def test_owner_music_note_gets_ask_with_both_buttons(self) -> None:
        self.context.tools = _EmptyTranscriptTools()
        with mock.patch.object(
                self.runtime.brain, "handle_message",
                side_effect=AssertionError("brain must not run on the ask")):
            with mock.patch(
                    "nomorals.voice.music_note.detect_music",
                    return_value=self._music()):
                self._process(_voice_msg(owner=True))
        self.assertEqual(len(self.gateway.sent), 1)
        _platform, text, kwargs = self.gateway.sent[0]
        self.assertEqual(text, mn.ASK_TEXT)
        buttons = kwargs.get("buttons")
        self.assertIsNotNone(buttons)
        flat = {label: data for row in (buttons or [])
                for label, data in row}
        self.assertEqual(flat["🔍 Find this music"], "musicnote find")
        self.assertEqual(flat["🎵 Create a song"], "musicnote create")
        # Pending clip stashed for the tap.
        pending = mn.PENDING.peek("telegram:999")
        self.assertIsNotNone(pending)
        self.assertEqual(pending["path"], "/tmp/note.ogg")

    def test_non_owner_music_note_gets_no_ask_devon_decides(self) -> None:
        self.context.tools = _EmptyTranscriptTools()
        seen: list[ChatMessage] = []

        def _handle(message: ChatMessage) -> SimpleNamespace:
            seen.append(message)
            return SimpleNamespace(parts=[], presence=SimpleNamespace(
                delay_seconds=0))

        with mock.patch.object(self.runtime.brain, "handle_message",
                               side_effect=_handle):
            with mock.patch(
                    "nomorals.voice.music_note.detect_music",
                    return_value=self._music()):
                self._process(_voice_msg(owner=False))
        # No ask sent, no buttons anywhere, no STT-failure reply.
        self.assertEqual(self.gateway.sent, [])
        self.assertEqual(len(seen), 1)
        # The brain gets the note as music context — and decides.
        self.assertIn("music playing", seen[0].text)
        self.assertIn("120 BPM", seen[0].text)
        self.assertEqual(seen[0].meta.get("music_note", {}).get("path"),
                         "/tmp/note.ogg")
        # No pending entry: a non-owner never gets buttons to tap.
        self.assertIsNone(mn.PENDING.peek("telegram:999"))

    def test_speech_voice_note_never_asked(self) -> None:
        # Speech transcribes: normal path, transcript becomes the text.
        self.context.tools = _WorkingTools("hello there")
        seen: list[ChatMessage] = []

        def _handle(message: ChatMessage) -> SimpleNamespace:
            seen.append(message)
            return SimpleNamespace(parts=[], presence=SimpleNamespace(
                delay_seconds=0))

        with mock.patch.object(self.runtime.brain, "handle_message",
                               side_effect=_handle):
            with mock.patch(
                    "nomorals.voice.music_note.detect_music",
                    side_effect=AssertionError(
                        "detection must not run on transcribed speech")):
                self._process(_voice_msg(owner=True))
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].text, "hello there")
        self.assertEqual(self.gateway.sent, [])
        self.assertIsNone(mn.PENDING.peek("telegram:999"))

    def test_untranscribed_speech_falls_through_to_honest_reply(self) -> None:
        # STT failed AND the clip is speech (low score): no ask —
        # the existing honest "couldn't transcribe" reply runs.
        self.context.tools = _FailingTools("no STT backend installed")
        with mock.patch(
                "nomorals.voice.music_note.detect_music",
                return_value={"is_music": False, "heard": "",
                              "score": 0.2, "seconds": 5.0}):
            self._process(_voice_msg(owner=True))
        self.assertEqual(len(self.gateway.sent), 1)
        self.assertIn("couldn't transcribe that voice note",
                      self.gateway.sent[0][1])
        self.assertNotIn("Find this music", self.gateway.sent[0][1])
        self.assertIsNone(mn.PENDING.peek("telegram:999"))

    def test_music_note_in_games_only_chat_is_silent(self) -> None:
        chat_modes.set_chat_mode("telegram:999", "games_only")
        self.context.tools = _EmptyTranscriptTools()
        with mock.patch(
                "nomorals.voice.music_note.detect_music",
                return_value=self._music()):
            self._process(_voice_msg(owner=True))
        self.assertEqual(self.gateway.sent, [])
        self.assertIsNone(mn.PENDING.peek("telegram:999"))


# ── /musicnote dispatch ──────────────────────────────────────────────

class MusicnoteControlTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-musicnote-")
        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": "telegram",
            "partner.owner_chats": "5478650254",
            "partner.owner_sender_ids": "5478650254",
        })
        self.context = build_context(settings, with_executor=False,
                                     with_tools=False)
        from nomorals.social.chat.gateway import ChatGateway

        class _FakeAdapter:
            name = "telegram"

            def __init__(self) -> None:
                self.sent: list[str] = []

        self.runtime = PartnerRuntime(
            self.context,
            gateway=ChatGateway({"telegram": _FakeAdapter()},
                                db=self.context.db))
        mn.PENDING.clear()
        self.addCleanup(mn.PENDING.clear)

    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception:
            pass
        self.context.close()
        self.tmp.cleanup()

    def test_registered_owner_only(self) -> None:
        self.assertEqual(CONTROL_COMMANDS["musicnote"], (1, 1))
        self.assertNotIn("musicnote", PUBLIC_COMMANDS)
        self.assertNotIn("musicnote", GAME_COMMANDS)
        cmd = parse_control("/musicnote find")
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd.kind, "musicnote")
        self.assertEqual(cmd.tail, "find")

    def test_help_page_and_catalog(self) -> None:
        page = COMMAND_DETAILS["musicnote"]
        self.assertIn("Find this music", page["what"])
        self.assertIn("/musicnote find | create", page["usage"])
        self.assertIn("try:", detailed_help("musicnote"))
        self.assertIn("/musicnote", list_catalog())
        grouped = [k for _t, kinds in LIST_GROUPS for k in kinds]
        self.assertEqual(grouped.count("musicnote"), 1)

    def test_find_runs_music_id_on_stashed_clip(self) -> None:
        mn.PENDING.remember("telegram:999", "/tmp/note.ogg", "12s of music")
        seen: list[Any] = []

        def _fake_sham(message: Any) -> str:
            seen.append(message)
            return "🎵 Artist — Title"

        self.runtime._sham_identify = _fake_sham  # type: ignore[method-assign]
        out = self.runtime.handle_control(
            "/musicnote find", "telegram:999",
            message=_msg("/musicnote find", owner=True))
        self.assertEqual(out, "🎵 Artist — Title")
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].media[0].path, "/tmp/note.ogg")
        self.assertEqual(seen[0].media[0].kind, "voice")
        # The tap consumed the pending note.
        self.assertIsNone(mn.PENDING.peek("telegram:999"))

    def test_create_runs_production_from_stashed_clip(self) -> None:
        mn.PENDING.remember("telegram:999", "/tmp/note.ogg", "12s of music")
        seen: list[tuple] = []

        def _fake_produce(path: str, heard: str, chat_key: str) -> str:
            seen.append((path, heard, chat_key))
            return "🎛 produced from your voice note: New Song"

        self.runtime._musicnote_produce = _fake_produce  # type: ignore[method-assign]
        out = self.runtime.handle_control(
            "/musicnote create", "telegram:999",
            message=_msg("/musicnote create", owner=True))
        self.assertIn("produced from your voice note", out)
        self.assertEqual(seen, [("/tmp/note.ogg", "12s of music",
                                  "telegram:999")])
        self.assertIsNone(mn.PENDING.peek("telegram:999"))

    def test_no_pending_is_honest_not_a_crash(self) -> None:
        out = self.runtime.handle_control(
            "/musicnote find", "telegram:999",
            message=_msg("/musicnote find", owner=True))
        self.assertIn("no music voice note waiting", out)
        out = self.runtime.handle_control(
            "/musicnote create", "telegram:999",
            message=_msg("/musicnote create", owner=True))
        self.assertIn("no music voice note waiting", out)

    def test_bad_action_is_usage(self) -> None:
        out = self.runtime.handle_control(
            "/musicnote dance", "telegram:999",
            message=_msg("/musicnote dance", owner=True))
        self.assertIn("usage: /musicnote find | create", out)

    def test_bare_command_is_arg_error_not_crash(self) -> None:
        out = self.runtime.handle_control(
            "/musicnote", "telegram:999",
            message=_msg("/musicnote", owner=True))
        self.assertIn("needs 1 argument", out)

    def test_non_owner_denied(self) -> None:
        mn.PENDING.remember("telegram:999", "/tmp/note.ogg", "12s of music")
        out = self.runtime.handle_control(
            "/musicnote find", "telegram:999",
            message=_msg("/musicnote find", owner=False))
        self.assertEqual(out, "that one's just for the owner.")
        # Denial does not consume the pending note.
        self.assertIsNotNone(mn.PENDING.peek("telegram:999"))


# ── brain: music context note ────────────────────────────────────────

class BrainMusicNoteTest(unittest.TestCase):
    def _brain(self, tools: Any) -> Any:
        from nomorals.agents.partner.brain import PartnerBrain

        brain = PartnerBrain.__new__(PartnerBrain)
        brain.context = SimpleNamespace(tools=tools, memory=None, db=None)
        brain._last_stt_failure_reason = ""
        brain.settings = SimpleNamespace(
            partner=SimpleNamespace(gate_restricted_chats=True,
                                    history_window=10))
        brain.responder = mock.Mock()
        brain.responder.recall.return_value = ()
        brain.responder.respond_bounded.side_effect = AssertionError(
            "model must not be called")
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

    def test_music_note_becomes_context_not_spoken(self) -> None:
        tools = _FailingTools("must not be called — music is not re-transcribed")
        brain = self._brain(tools)
        msg = _voice_msg(owner=False)
        msg.meta["music_note"] = {"path": "/tmp/note.ogg",
                                  "heard": "12s of music · ~120 BPM"}
        notes = brain._media_notes(msg)
        joined = " ".join(notes)
        self.assertIn("music playing", joined)
        self.assertIn("120 BPM", joined)
        self.assertNotIn("SPOKEN", joined)
        self.assertNotIn("voice_untranscribed_reason", msg.meta)

    def test_music_note_for_other_path_falls_through(self) -> None:
        brain = self._brain(_FailingTools("no STT backend installed"))
        msg = _voice_msg(owner=False)
        msg.meta["music_note"] = {"path": "/tmp/other.ogg", "heard": "x"}
        notes = brain._media_notes(msg)
        self.assertIn("SPOKEN", " ".join(notes))

    def test_music_note_flows_to_model_no_guard(self) -> None:
        brain = self._brain(_FailingTools("must not be called"))
        brain.responder.respond_bounded.side_effect = None
        brain.responder.respond_bounded.return_value = SimpleNamespace(
            parts=["nice track"], model="test",
            degraded=False, degraded_note="")
        msg = _voice_msg(owner=False)
        msg.meta["music_note"] = {"path": "/tmp/note.ogg",
                                  "heard": "12s of music"}
        parts = brain._generate_and_persist(
            msg, {"is_owner": False, "in_us": False})
        self.assertIn("nice track", parts)
        media_notes = (brain.responder.respond_bounded.call_args.kwargs
                       .get("media_notes") or [])
        self.assertIn("music playing", " ".join(media_notes))


if __name__ == "__main__":
    unittest.main()
