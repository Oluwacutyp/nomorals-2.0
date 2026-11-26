"""Tests for the /mix chat surface (nomorals/media/mix_chat.py).

Drives ``control_mix`` directly (fake queue DB + fake sender + real
WAV fixtures) and probes the real PartnerRuntime dispatch once, so
a registration gap shows up here instead of in a live chat.
"""

from __future__ import annotations

import json
import math
import struct
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from types import SimpleNamespace

from nomorals.media.mix_chat import control_mix
from nomorals.media.mix_surface import (
    MixDescriptor,
    load_descriptor,
    share_descriptor,
)

SR = 22050


def _click_track(bpm: float, seconds: float = 5.0) -> array:
    n = int(SR * seconds)
    out = array("d", [0.0]) * n
    beat_n = int(SR * 60.0 / bpm)
    for start in range(0, n, beat_n):
        for i in range(min(2205, n - start)):
            t = i / SR
            out[start + i] += math.exp(-t * 60.0) * math.sin(
                2 * math.pi * 180.0 * t)
    return out


def _write_wav(path: str, samples: array) -> str:
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(struct.pack(f"<{len(samples)}h",
                                  *(int(s * 20000) for s in samples)))
    return path


class FakeDb:
    """Minimal stand-in for context.db: query_rows over fixed rows."""

    def __init__(self, rows):
        self._rows = rows

    def query_rows(self, sql, params=()):
        return list(self._rows)


class FakeSender:
    def __init__(self):
        self.audio: list[tuple] = []
        self.voice: list[tuple] = []

    def send_audio(self, chat, path, **kwargs):
        self.audio.append((str(chat), path))
        return "sent-audio"

    def send_voice(self, chat, path, **kwargs):
        self.voice.append((str(chat), path))
        return "sent-voice"


def _ctx(rows):
    return SimpleNamespace(db=FakeDb(rows))


def _rows(tmp):
    p1 = _write_wav(str(tmp / "one.wav"), _click_track(120.0))
    p2 = _write_wav(str(tmp / "two.wav"), _click_track(122.0))
    return [
        {"path": p1, "title": "Track One", "artist": "A",
         "album": "", "duration": 5.0, "kind": "file"},
        {"path": p2, "title": "Track Two", "artist": "B",
         "album": "", "duration": 5.0, "kind": "file"},
    ]


def _missing_rows():
    return [
        {"path": "/no/such/track-one.wav", "title": "Ghost One",
         "artist": "", "album": "", "duration": 180.0, "kind": "file"},
        {"path": "/no/such/track-two.wav", "title": "Ghost Two",
         "artist": "", "album": "", "duration": 200.0, "kind": "file"},
    ]


class MixChatTests(unittest.TestCase):
    def test_bare_mix_shows_menu(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = control_mix("", chat_key="telegram:1", context=_ctx([]),
                              workdir=tmp)
            self.assertIn("/mix new", out)
            self.assertIn("/mix send", out)
            self.assertNotIn("current mix", out)

    def test_new_from_queue_and_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = _rows(Path(tmp))
            out = control_mix("new late night drive", chat_key="telegram:1",
                              context=_ctx(rows), workdir=tmp)
            self.assertIn("late night drive", out)
            self.assertIn("Track One", out)
            self.assertIn("Track Two", out)
            state = Path(tmp) / "telegram_1.json"
            self.assertTrue(state.exists())
            loaded = load_descriptor(state.read_text(encoding="utf-8"))
            self.assertEqual(loaded.title, "late night drive")
            self.assertEqual(len(loaded.tracks), 2)
            self.assertEqual(loaded.id, "late-night-drive")
            # Card is stable: same fingerprint on re-read.
            card = control_mix("card", chat_key="telegram:1",
                               context=_ctx(rows), workdir=tmp)
            self.assertIn(loaded.fingerprint[:12], card)

    def test_new_empty_queue_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = control_mix("new empty", chat_key="telegram:1",
                              context=_ctx([]), workdir=tmp)
            self.assertIn("queue is empty", out)
            self.assertFalse((Path(tmp) / "telegram_1.json").exists())

    def test_new_replaces_previous_mix(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = _rows(Path(tmp))
            control_mix("new first", chat_key="telegram:1",
                        context=_ctx(rows), workdir=tmp)
            out = control_mix("new second", chat_key="telegram:1",
                              context=_ctx(rows), workdir=tmp)
            self.assertIn("replaced the previous mix", out)
            self.assertIn("second", out)

    def test_new_without_context_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = control_mix("new noctx", chat_key="telegram:1",
                              context=None, workdir=tmp)
            self.assertIn("queue is empty", out)

    def test_add_by_index_title_and_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = _rows(Path(tmp))
            ctx = _ctx(rows)
            control_mix("new base", chat_key="telegram:1",
                        context=ctx, workdir=tmp)
            out = control_mix("add 2", chat_key="telegram:1",
                              context=ctx, workdir=tmp)
            self.assertIn("added #3", out)
            self.assertIn("Track Two", out)
            out = control_mix("add track one", chat_key="telegram:1",
                              context=ctx, workdir=tmp)
            self.assertIn("added #4", out)
            extra = _write_wav(str(Path(tmp) / "extra.wav"),
                               _click_track(100.0))
            out = control_mix(f"add {extra}", chat_key="telegram:1",
                              context=ctx, workdir=tmp)
            self.assertIn("added #5", out)
            self.assertIn("extra", out)

    def test_add_unknown_and_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = _rows(Path(tmp))
            ctx = _ctx(rows)
            control_mix("new base", chat_key="telegram:1",
                        context=ctx, workdir=tmp)
            out = control_mix("add nope-not-there", chat_key="telegram:1",
                              context=ctx, workdir=tmp)
            self.assertIn("no_such_track", out)
            out = control_mix("add track", chat_key="telegram:1",
                              context=ctx, workdir=tmp)
            self.assertIn("ambiguous", out)
            self.assertIn("Track One", out)
            self.assertIn("Track Two", out)
            out = control_mix("add 99", chat_key="telegram:1",
                              context=ctx, workdir=tmp)
            self.assertIn("no_such_track", out)

    def test_add_without_mix(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = control_mix("add 1", chat_key="telegram:1",
                              context=_ctx(_rows(Path(tmp))), workdir=tmp)
            self.assertIn("no mix yet", out)

    def test_order_moves_track(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = _rows(Path(tmp))
            ctx = _ctx(rows)
            control_mix("new base", chat_key="telegram:1",
                        context=ctx, workdir=tmp)
            out = control_mix("order 2 1", chat_key="telegram:1",
                              context=ctx, workdir=tmp)
            self.assertIn("moved track 2 to position 1", out)
            state = load_descriptor(
                (Path(tmp) / "telegram_1.json").read_text(encoding="utf-8"))
            self.assertEqual(state.tracks[0].title, "Track Two")
            self.assertEqual(state.version, 2)
            out = control_mix("order 1 9", chat_key="telegram:1",
                              context=ctx, workdir=tmp)
            self.assertIn("out of range", out)
            out = control_mix("order x y", chat_key="telegram:1",
                              context=ctx, workdir=tmp)
            self.assertIn("usage: /mix order", out)

    def test_curve_set_and_bad_curve(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = _rows(Path(tmp))
            ctx = _ctx(rows)
            control_mix("new base", chat_key="telegram:1",
                        context=ctx, workdir=tmp)
            out = control_mix("curve wave", chat_key="telegram:1",
                              context=ctx, workdir=tmp)
            self.assertIn("energy curve: wave", out)
            self.assertIn("Energy: wave", out)
            out = control_mix("curve sideways", chat_key="telegram:1",
                              context=ctx, workdir=tmp)
            self.assertIn("unknown curve", out)
            self.assertIn("late_peak", out)

    def test_render_plan_only_names_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = _ctx(_missing_rows())
            control_mix("new ghosts", chat_key="telegram:1",
                        context=ctx, workdir=tmp)
            out = control_mix("render", chat_key="telegram:1",
                              context=ctx, workdir=tmp)
            self.assertIn("plan only", out)
            self.assertIn("Ghost One", out)
            self.assertIn("Ghost Two", out)

    def test_render_real_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = _rows(Path(tmp))
            ctx = _ctx(rows)
            control_mix("new real", chat_key="telegram:1",
                        context=ctx, workdir=tmp)
            out = control_mix("render", chat_key="telegram:1",
                              context=ctx, workdir=tmp)
            self.assertIn("rendered:", out)
            wav = Path(tmp) / "telegram_1" / "real.wav"
            self.assertTrue(wav.exists())
            with wave.open(str(wav), "rb") as f:
                self.assertEqual(f.getnchannels(), 1)
                self.assertGreater(f.getnframes(), 0)

    def test_send_audio_and_voice(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = _rows(Path(tmp))
            ctx = _ctx(rows)
            sender = FakeSender()
            control_mix("new real", chat_key="telegram:1",
                        context=ctx, workdir=tmp)
            out = control_mix("send telegram:999", chat_key="telegram:1",
                              context=ctx, sender=sender, workdir=tmp)
            self.assertIn("sent as audio to telegram:999", out)
            self.assertEqual(len(sender.audio), 1)
            self.assertEqual(sender.audio[0][0], "telegram:999")
            self.assertTrue(sender.audio[0][1].endswith(".wav"))
            out = control_mix("send telegram:999 --voice",
                              chat_key="telegram:1", context=ctx,
                              sender=sender, workdir=tmp)
            self.assertIn("sent as voice note to telegram:999", out)
            self.assertEqual(len(sender.voice), 1)

    def test_send_plan_only_and_bad_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = _ctx(_missing_rows())
            sender = FakeSender()
            control_mix("new ghosts", chat_key="telegram:1",
                        context=ctx, workdir=tmp)
            out = control_mix("send telegram:999", chat_key="telegram:1",
                              context=ctx, sender=sender, workdir=tmp)
            self.assertIn("nothing to send", out)
            self.assertEqual(sender.audio, [])
            out = control_mix("send notachat", chat_key="telegram:1",
                              context=ctx, sender=sender, workdir=tmp)
            self.assertIn("bad chat", out)
            out = control_mix("send", chat_key="telegram:1",
                              context=ctx, sender=sender, workdir=tmp)
            self.assertIn("usage: /mix send", out)

    def test_send_without_sender(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = _rows(Path(tmp))
            ctx = _ctx(rows)
            control_mix("new real", chat_key="telegram:1",
                        context=ctx, workdir=tmp)
            out = control_mix("send telegram:999", chat_key="telegram:1",
                              context=ctx, sender=None, workdir=tmp)
            self.assertIn("no sender", out)

    def test_share_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = _rows(Path(tmp))
            ctx = _ctx(rows)
            control_mix("new base", chat_key="telegram:1",
                        context=ctx, workdir=tmp)
            control_mix("curve linear", chat_key="telegram:1",
                        context=ctx, workdir=tmp)
            payload = control_mix("share", chat_key="telegram:1",
                                  context=ctx, workdir=tmp)
            data = json.loads(payload)
            self.assertEqual(data["title"], "base")
            self.assertEqual(len(data["tracks"]), 2)
            with tempfile.TemporaryDirectory() as tmp2:
                out = control_mix(f"load {payload}", chat_key="telegram:2",
                                  context=ctx, workdir=tmp2)
                self.assertIn("loaded mix", out)
                self.assertIn("base", out)
                self.assertIn("Energy: linear", out)

    def test_load_garbage_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = _rows(Path(tmp))
            ctx = _ctx(rows)
            control_mix("new base", chat_key="telegram:1",
                        context=ctx, workdir=tmp)
            out = control_mix("load {not json", chat_key="telegram:1",
                              context=ctx, workdir=tmp)
            self.assertIn("not a mix descriptor", out)
            out = control_mix("load", chat_key="telegram:1",
                              context=ctx, workdir=tmp)
            self.assertIn("usage: /mix load", out)

    def test_unknown_action_shows_menu(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = _rows(Path(tmp))
            ctx = _ctx(rows)
            control_mix("new base", chat_key="telegram:1",
                        context=ctx, workdir=tmp)
            out = control_mix("dance", chat_key="telegram:1",
                              context=ctx, workdir=tmp)
            self.assertIn("unknown /mix action 'dance'", out)
            self.assertIn("/mix new", out)

    def test_corrupt_state_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "telegram_1.json").write_text(
                "{corrupt", encoding="utf-8")
            out = control_mix("card", chat_key="telegram:1",
                              context=_ctx([]), workdir=tmp)
            self.assertIn("corrupt", out)

    def test_share_load_descriptor_roundtrip_unit(self):
        d = MixDescriptor(id="t", title="T")
        self.assertEqual(load_descriptor(share_descriptor(d)).title, "T")


class MixRuntimeDispatchTests(unittest.TestCase):
    def test_mix_dispatches_through_real_runtime(self):
        from nomorals.agents.context import build_context
        from nomorals.agents.partner_runtime import PartnerRuntime
        from nomorals.core.config import load_settings
        from nomorals.social.chat.base import (
            ChatAdapter,
            ChatRef,
            SendResult,
        )
        from nomorals.social.chat.gateway import ChatGateway

        class _Adapter(ChatAdapter):
            def __init__(self, name):
                super().__init__(media_dir="/tmp/nm-test-media")
                self.name = name
                self.sent: list[str] = []

            def run(self, handler):
                pass

            def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
                     buttons=None, **kwargs) -> SendResult:
                self.sent.append(text)
                return SendResult(ok=True, platform=self.name,
                                  message_id=f"m{len(self.sent)}")

        with tempfile.TemporaryDirectory(prefix="nm-test-mix-dispatch-") as home:
            settings = load_settings(overrides={
                "home": home,
                "partner.platforms": "telegram",
                "partner.owner_chats": "5478650254",
                "partner.owner_sender_ids": "5478650254",
            })
            context = build_context(settings, with_executor=False,
                                    with_tools=False)
            adapter = _Adapter("telegram")
            gateway = ChatGateway({"telegram": adapter}, db=context.db)
            runtime = PartnerRuntime(context, gateway=gateway)
            try:
                out = runtime.handle_control("/mix", "telegram:5478650254")
                self.assertIn("/mix new", out)
                self.assertNotIn("unknown command", out)
                out = runtime.handle_control(
                    "/mix new runtime-check", "telegram:5478650254")
                self.assertIn("queue is empty", out)
            finally:
                context.close()


if __name__ == "__main__":
    unittest.main()
