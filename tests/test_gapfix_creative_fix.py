"""Creative-fix pins (owner complaint 2026-11-19).

The mock-chat form shipped 2026-11-11 as 4233a296 but the
owner's mornings never changed: the Motivation class kept
``card`` as its default, and the chat screens staged text
only — no real image ever landed as the photo message, no
real voice note ever landed as the voice message, and the
screen recordings were reachable only from /mock demos,
never from the classes themselves.  These tests pin the
fix:

- the motivation default is ROTATE (card/chat alternate
  by Lagos day + window; an explicit card pref still
  means the card always);
- a run under the default really delivers the chat form
  on a chat slot, and a chat render that fails still
  falls back to the card with the failure named;
- a real generated image lands as the photo message
  (the drawn record says "real", never "placeholder")
  and a real voice note lands with its TRUE duration and
  waveform measured from its own samples;
- /qcard screen, /quote chat, /meme2 screen and
  /fakemsg screen each render AND deliver end to end;
- /qcard screen record and /quote chat record deliver
  a real playable mp4, and a recording that can't encode
  falls back to the still screen with the failure named.
"""

from __future__ import annotations

import math
import random
import shutil
import struct
import tempfile
import time
import unittest
import wave
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from PIL import Image

from nomorals.social.chat import mock_formats
from nomorals.social.chat.fun import FunService, LocalQuotesProvider
from nomorals.social.chat.makers import MakerService, ProviderHealth

LAGOS = ZoneInfo("Africa/Lagos")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
MP4_MAGIC = b"ftyp"


class _Send:
    def __init__(self, ok=True):
        self.calls = []
        self.ok = ok

    def __call__(self, ref, caption=""):
        self.calls.append((ref, caption))
        return SimpleNamespace(ok=self.ok, error="")


def _red_image(size: int = 200) -> Image.Image:
    """A real generated image (a red gradient, no network)."""
    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            px[x, y] = (200, 40 + x // 4, 60 + y // 4)
    return img


def _png_bytes(size: int = 200) -> bytes:
    import io

    buf = io.BytesIO()
    _red_image(size).save(buf, format="PNG")
    return buf.getvalue()


def _wav_file(tmp_path, seconds: float = 2.5) -> Path:
    """A real voice-note file: 16 kHz mono, first half loud,
    second half quiet — the waveform bars must differ."""
    rate = 16000
    frames = int(rate * seconds)
    path = Path(tmp_path) / "voice.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        for i in range(frames):
            amp = 0.9 if i < frames // 2 else 0.2
            val = int(amp * 32767
                      * math.sin(2 * math.pi * 440 * i / rate))
            wf.writeframes(struct.pack("<h", val))
    return path


def _drawn_elements(drawn: dict) -> list[dict]:
    return list(drawn.get("elements") or [])


def _media_entries(drawn: dict) -> list[dict]:
    return [e for e in _drawn_elements(drawn)
            if e.get("kind") in ("image", "voice")]


# ── the media hand-off: real media in, real media drawn ──────

def test_real_image_lands_as_the_photo_message(tmp_path):
    screen = mock_formats.quote_screen(
        "the way is the goal", "Meditations",
        image=_red_image())
    out = mock_formats.render_screen_image(
        screen, media_dir=str(tmp_path),
        media={mock_formats.IMAGE_MESSAGE_ID: _red_image()})
    assert out["ok"], out
    assert out["data"][:8] == PNG_MAGIC
    media = _media_entries(out["drawn"])
    assert len(media) == 1, out["drawn"]
    assert media[0]["media_source"] == "real", media
    assert media[0]["kind"] == "image"


def test_real_voice_note_lands_with_its_true_duration(tmp_path):
    wav = _wav_file(tmp_path, seconds=2.5)
    screen = mock_formats.motivation_screen(
        "carry this: the obstacle is the way forward, always.",
        theme="discipline", voice=str(wav))
    out = mock_formats.render_screen_image(
        screen, media_dir=str(tmp_path))
    assert out["ok"], out
    media = _media_entries(out["drawn"])
    assert len(media) == 1, out["drawn"]
    assert media[0]["kind"] == "voice"
    assert media[0]["media_source"] == "real", media
    assert abs(media[0]["duration_s"] - 2.5) < 0.05, media
    bars = media[0]["bars"]
    assert len(bars) == 26
    assert max(bars[:13]) > max(bars[13:]), bars


def test_no_media_means_no_media_bubble_at_all():
    screen = mock_formats.motivation_screen(
        "carry this: the obstacle is the way forward, always.",
        theme="discipline")
    assert not [m for m in screen.messages
                if m.kind in ("image", "voice")]
    screen = mock_formats.quote_screen("the way", "Someone")
    assert not [m for m in screen.messages
                if m.kind in ("image", "voice")]


def test_image_bytes_ride_the_hand_off_not_a_path(tmp_path):
    screen = mock_formats.entries_screen(
        [{"name": "Ada", "text": "we ship at dawn"}],
        image=_red_image())
    out = mock_formats.render_screen_image(
        screen, media_dir=str(tmp_path),
        media={mock_formats.IMAGE_MESSAGE_ID: _png_bytes()})
    assert out["ok"], out
    media = _media_entries(out["drawn"])
    assert len(media) == 1
    assert media[0]["media_source"] == "real", media


# ── the on-demand forms: render + deliver, end to end ───────

def _quote_service(tmp_path, **kw):
    kw.setdefault("media_dir", tmp_path / "fun_media")
    kw.setdefault("chains", {"quote": [LocalQuotesProvider(
        rng=random.Random(3))]})
    return FunService(**kw)


def _maker_service(tmp_path, **kw):
    return MakerService(media_dir=tmp_path / "maker_media",
                        health=ProviderHealth(), **kw)


def test_qcard_screen_renders_and_delivers(tmp_path):
    send = _Send()
    out = _maker_service(tmp_path).card(
        "qcard", "screen Ada | we ship at dawn", send=send)
    assert out["sent"] is True, out
    assert out["format"] == "screen"
    ref, _caption = send.calls[0]
    assert Path(ref.path).read_bytes()[:8] == PNG_MAGIC


def test_fakemsg_screen_renders_and_delivers(tmp_path):
    send = _Send()
    out = _maker_service(tmp_path).card(
        "fakemsg", "screen Ada | the build is green",
        send=send)
    assert out["sent"] is True, out
    assert out["format"] == "screen"


def test_meme2_screen_renders_and_delivers(tmp_path):
    send = _Send()
    out = _maker_service(tmp_path).card(
        "meme2", "screen Ada | the deploy worked on my machine",
        send=send)
    assert out["sent"] is True, out
    assert out["format"] == "screen"


def test_quote_chat_renders_and_delivers(tmp_path):
    send = _Send()
    out = _quote_service(tmp_path).quote_card(
        format="screen", send=send,
        backdrop_fn=lambda seed="": _red_image())
    assert out["sent"] is True, out
    assert out["format"] == "screen"
    ref, _caption = send.calls[0]
    assert Path(ref.path).read_bytes()[:8] == PNG_MAGIC


def test_quote_chat_carries_the_backdrop_as_its_photo(
        tmp_path, monkeypatch):
    from nomorals.social.chat import mock_formats as mf
    from nomorals.social.chat.mock_screens.service import MockScreenService

    seen = {}
    original_render = MockScreenService.render

    def _capture(self, screen, **kw):
        seen["media"] = kw.get("media")
        return original_render(self, screen, **kw)

    monkeypatch.setattr(MockScreenService, "render", _capture)
    send = _Send()
    out = _quote_service(tmp_path).quote_card(
        format="screen", send=send,
        backdrop_fn=lambda seed="": _red_image())
    assert out["sent"] is True, out
    assert seen["media"] is not None
    assert mf.IMAGE_MESSAGE_ID in seen["media"]


def test_qcard_screen_record_delivers_a_real_mp4(tmp_path):
    send = _Send()
    out = _maker_service(
        tmp_path,
        bytes_fetcher=lambda url: _png_bytes()).card(
            "qcard", "screen record Ada | we ship at dawn",
            send=send)
    assert out["sent"] is True, out
    assert out["format"] == "record", out
    ref, _caption = send.calls[0]
    data = Path(ref.path).read_bytes()
    assert MP4_MAGIC in data[:32], data[:32]
    assert out.get("duration_s", 0) > 0, out


def test_quote_chat_record_delivers_a_real_mp4(tmp_path):
    send = _Send()
    out = _quote_service(tmp_path).quote_card(
        format="record", send=send,
        backdrop_fn=lambda seed="": _red_image())
    assert out["sent"] is True, out
    assert out["format"] == "record", out
    ref, _caption = send.calls[0]
    data = Path(ref.path).read_bytes()
    assert MP4_MAGIC in data[:32], data[:32]
    assert out.get("duration_s", 0) > 0, out


def test_record_failure_falls_back_to_the_still_named(
        tmp_path, monkeypatch):
    class _DeadRecorder:
        def __init__(self, *, media_dir=""):
            self._dir = media_dir

        def render(self, screen, **kw):
            # The real service class, from its own module —
            # the package attribute is the patched stub.
            from nomorals.social.chat.mock_screens.service import MockScreenService as _Svc

            return _Svc(media_dir=self._dir).render(screen, **kw)

        def record_chat(self, screen, **kw):
            return {"sent": False,
                    "reason": "the encoder is down"}

    monkeypatch.setattr(
        "nomorals.social.chat.mock_screens.MockScreenService",
        _DeadRecorder)
    send = _Send()
    out = _maker_service(tmp_path).card(
        "qcard", "screen record Ada | we ship at dawn",
        send=send)
    assert out["sent"] is True, out
    assert out["format"] == "screen", out
    assert "the encoder is down" in out["record_error"], out


def test_quote_record_word_parses_from_the_tail():
    from nomorals.agents.partner.runtime_fun import RuntimeFunMixin

    seen = {}

    class _StubService:
        def quote_card(self, arg, *, format="", send):
            seen.update({"arg": arg, "format": format})
            return {"sent": True}

    runtime = RuntimeFunMixin()
    runtime._fun_service_obj = _StubService()
    message = SimpleNamespace()
    assert runtime._fun_quote("chat record dedication",
                              message) == ""
    assert seen == {"arg": "dedication", "format": "record"}


# ── the Motivation class: rotate is the default ──────────────

def _chat_morning_ts() -> float:
    """A near-future 07:05 Lagos instant whose rotate slot
    is the chat form (odd day ordinal, morning)."""
    base = datetime.now(LAGOS).date()
    for k in range(7, 21):
        day = base + timedelta(days=k)
        if day.toordinal() % 2 == 1:
            return datetime(day.year, day.month, day.day, 7, 5,
                            tzinfo=LAGOS).timestamp()
    raise AssertionError("no chat-slot morning in range")


def _card_morning_ts() -> float:
    base = datetime.now(LAGOS).date()
    for k in range(7, 21):
        day = base + timedelta(days=k)
        if day.toordinal() % 2 == 0:
            return datetime(day.year, day.month, day.day, 7, 5,
                            tzinfo=LAGOS).timestamp()
    raise AssertionError("no card-slot morning in range")


class _MotivationHarness(unittest.TestCase):
    """The m1 harness shape: real Database + ledger, stubbed
    producers/brain/notifier, per-test tmp home."""

    def _ctx(self, **over):
        from nomorals.storage import Database

        tmp = tempfile.mkdtemp(prefix="creativefix-mot-")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        db = Database(str(Path(tmp) / "test.db"))
        db.migrate()
        partner = SimpleNamespace(
            owner_chats="telegram-bot:111", proactive_enabled=True,
            proactive_briefing=True, proactive_watchers=True,
            quiet_start=22, quiet_end=8, timezone="UTC")
        settings = SimpleNamespace(
            workspace_dir=tmp, partner=partner,
            resolve=lambda rel: str(Path(tmp) / rel))
        ctx = SimpleNamespace(db=db, settings=settings, extras={})
        for key, val in over.items():
            setattr(ctx, key, val)
        return ctx

    def _wire(self, ctx):
        from nomorals.agents import motivation as mot
        from nomorals.agents import teachings as teach
        from nomorals.llm import brain as brain_mod

        item = {
            "id": "teach_meditations_1", "kind": "teaching",
            "title": "Meditations", "body": "", "theme": "discipline",
            "tradition": "", "weight": 1.0,
            "source": {"type": "wisdom", "ref": "meditations",
                       "label": "Meditations — WisdomKeeper",
                       "passage": "The impediment to action advances "
                                  "action; what stands in the way "
                                  "becomes the way."},
            "created_at": time.time(), "served": [],
        }
        self.addCleanup(setattr, teach, "library_candidates",
                        teach.library_candidates)
        self.addCleanup(setattr, teach, "net_candidates",
                        teach.net_candidates)
        self.addCleanup(setattr, teach, "memory_candidates",
                        teach.memory_candidates)
        teach.library_candidates = lambda c, themes: [item]
        teach.net_candidates = lambda c, themes: []
        teach.memory_candidates = lambda c: []
        self.addCleanup(setattr, brain_mod, "brain_for",
                        brain_mod.brain_for)
        words = " ".join(["word"] * 150)
        brain_mod.brain_for = lambda c: SimpleNamespace(
            chat=lambda messages: SimpleNamespace(text=words))

        class _Notifier:
            def __init__(self):
                self.calls = []

            def publish(self, kind, title, body="", **kw):
                self.calls.append({"kind": kind, "title": title,
                                   "body": body, **kw})
                return {"delivered": True, "persisted_only": False}

        ctx.notifier = _Notifier()
        return mot


class MotivationRotateDefaultTests(_MotivationHarness):
    def test_default_delivers_the_chat_form_on_a_chat_slot(self):
        ctx = self._ctx()
        mot = self._wire(ctx)
        self.addCleanup(setattr, mot, "_safe_backdrop",
                        mot._safe_backdrop)
        mot._safe_backdrop = lambda piece: _red_image()
        res = mot.run_motivation(ctx, "morning",
                                 now=_chat_morning_ts(),
                                 rng=random.Random(3))
        assert res["delivered"], res
        assert res["format"] == "chat", res
        assert res["card_rendered"] is True
        assert res["screen_media"] == {"image": True,
                                        "voice": False}, res
        screen = (Path(ctx.settings.workspace_dir)
                  / "motivation_cards" / "motivation-morning-chat.png")
        assert screen.read_bytes()[:8] == PNG_MAGIC
        marker = mot._read_marker(ctx)
        assert marker["morning"]["format"] == "chat"

    def test_default_delivers_the_card_form_on_a_card_slot(self):
        ctx = self._ctx()
        mot = self._wire(ctx)
        self.addCleanup(setattr, mot, "_safe_backdrop",
                        mot._safe_backdrop)
        mot._safe_backdrop = lambda piece: _red_image()
        res = mot.run_motivation(ctx, "morning",
                                 now=_card_morning_ts(),
                                 rng=random.Random(3))
        assert res["delivered"], res
        assert res["format"] == "card", res
        card = (Path(ctx.settings.workspace_dir)
                / "motivation_cards" / "motivation-morning.png")
        assert card.read_bytes()[:8] == PNG_MAGIC

    def test_default_chat_failure_falls_back_to_the_card_named(self):
        ctx = self._ctx()
        mot = self._wire(ctx)
        original = mot._render_piece_screen

        def _boom(*a, **kw):
            raise RuntimeError("organ gone")

        mot._render_piece_screen = _boom
        self.addCleanup(setattr, mot, "_render_piece_screen",
                        original)
        res = mot.run_motivation(ctx, "morning",
                                 now=_chat_morning_ts(),
                                 rng=random.Random(3))
        assert res["delivered"], res
        assert res["format"] == "card", res
        assert "organ gone" in res["format_error"]

    def test_voice_note_lands_inside_the_chat_screen(self):
        ctx = self._ctx()
        mot = self._wire(ctx)
        ctx.settings.motivation = {"voice": True}
        tmp = tempfile.mkdtemp(prefix="creativefix-voice-")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        wav = _wav_file(tmp, seconds=2.5)
        self.addCleanup(setattr, mot, "_safe_backdrop",
                        mot._safe_backdrop)
        self.addCleanup(setattr, mot, "_make_voice_ogg",
                        mot._make_voice_ogg)
        mot._safe_backdrop = lambda piece: _red_image()
        mot._make_voice_ogg = lambda c, text: str(wav)
        seen = {}

        import nomorals.social.chat.mock_formats as mf

        original_render = mf.render_screen_image

        def _capture(screen, **kw):
            seen["screen"] = screen
            seen["media"] = kw.get("media")
            return original_render(screen, **kw)

        mf.render_screen_image = _capture
        self.addCleanup(setattr, mf, "render_screen_image",
                        original_render)
        res = mot.run_motivation(ctx, "morning",
                                 now=_chat_morning_ts(),
                                 rng=random.Random(3))
        assert res["delivered"], res
        assert res["format"] == "chat", res
        assert res["screen_media"] == {"image": True,
                                        "voice": True}, res
        voice_msgs = [m for m in seen["screen"].messages
                      if m.kind == "voice"]
        assert len(voice_msgs) == 1, seen["screen"].messages
        assert voice_msgs[0].media == str(wav)
        assert mf.IMAGE_MESSAGE_ID in (seen["media"] or {})
