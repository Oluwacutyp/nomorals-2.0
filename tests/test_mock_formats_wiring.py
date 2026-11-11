"""Mock screens as a content FORMAT — the creative wiring pins.

The mock-screens organ is a format rung on the classes that
already produce content, not a standalone toy:

- ``social.chat.mock_formats`` stages class material as a
  conversation (registry-driven: platforms are read from the
  organ's LIVE design registry, so a family the organ gains
  is selectable with no new wiring);
- /quote gains the screen format (``format="screen"`` /
  ``/quote chat``), with the bigquote card as the named
  fallback when a screen can't render;
- the makers card surface gains the same rung through its
  own selection grammar (``/qcard screen …``, ``/meme2
  screen …``);
- the Motivation class gains the chat format — explicit
  per-run pick, the class pref, and the rotate mode — with
  the other format as fallback and the delivered format
  named in the run result, the marker and the ledger.
"""

from __future__ import annotations

import random
import shutil
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from nomorals.social.chat import mock_formats
from nomorals.social.chat.fun import FunService, LocalQuotesProvider
from nomorals.social.chat.makers import MakerService, ProviderHealth
from nomorals.social.chat.mock_screens import designs

LAGOS = ZoneInfo("Africa/Lagos")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class _Send:
    def __init__(self, ok=True):
        self.calls = []
        self.ok = ok

    def __call__(self, ref, caption=""):
        self.calls.append((ref, caption))
        return SimpleNamespace(ok=self.ok, error="")


# ── the staging builders ─────────────────────────────────────

def test_platforms_come_from_the_live_registry():
    assert mock_formats.chat_platforms() == tuple(designs.CHAT_DESIGNS)
    assert "whatsapp" in mock_formats.chat_platforms()


def test_unknown_platform_is_a_named_refusal():
    with pytest.raises(ValueError) as err:
        mock_formats.require_platform("myspace")
    assert "whatsapp" in str(err.value)


def test_a_new_registry_family_is_selectable_without_new_wiring(
        monkeypatch):
    monkeypatch.setitem(designs.CHAT_DESIGNS, "testgram",
                        designs.CHAT_DESIGNS["whatsapp"])
    assert "testgram" in mock_formats.chat_platforms()
    screen = mock_formats.quote_screen(
        "Stay hungry.", "Steve Jobs", platform="testgram")
    out = mock_formats.render_screen_image(screen)
    assert out["ok"], out
    assert out["data"][:8] == PNG_MAGIC


def test_quote_screen_stages_the_author_as_the_sender():
    screen = mock_formats.quote_screen(
        "Be here now.", "Ram Dass")
    assert screen.title == "Ram Dass"
    senders = {m.sender for m in screen.messages if m.kind == "text"}
    assert "me" in senders  # the owner's answer makes it an exchange
    quote_msg = [m for m in screen.messages
                 if "Be here now." in m.text]
    assert quote_msg and quote_msg[0].reactions == ["❤️"]


def test_motivation_screen_is_a_two_voice_exchange():
    piece = " ".join(["word"] * 150)
    screen = mock_formats.motivation_screen(
        piece, theme="discipline", window="morning")
    assert screen.title == "Devon"
    texts = [m for m in screen.messages if m.kind == "text"]
    assert texts[0].sender == "me"
    assert "discipline" in texts[0].text
    devon = [m for m in texts if m.sender == "devon"]
    assert len(devon) >= 2  # a long piece is a run of bubbles
    assert devon[-1].reactions == ["🙏"]
    assert texts[-1].sender == "me"  # the owner receives it, closes
    joined = " ".join(m.text for m in devon)
    assert joined == piece  # every word survives the staging


def test_split_bubbles_packs_evenly_and_bounds_the_run():
    text = ". ".join(f"Sentence number {i} carries its weight"
                     for i in range(30)) + "."
    bubbles = mock_formats.split_bubbles(text)
    assert 2 <= len(bubbles) <= 4
    assert all(bubbles)
    assert " ".join(bubbles) == text


# ── /quote: the screen rung + the named fallback ─────────────

def _quote_service(tmp_path, **kw):
    kw.setdefault("media_dir", tmp_path / "fun_media")
    kw.setdefault("chains", {"quote": [LocalQuotesProvider(
        rng=random.Random(3))]})
    return FunService(**kw)


def test_quote_screen_format_sends_a_png(tmp_path):
    send = _Send()
    out = _quote_service(tmp_path).quote_card(format="screen",
                                              send=send)
    assert out["sent"] is True, out
    assert out["format"] == "screen"
    ref, caption = send.calls[0]
    assert Path(ref.path).read_bytes()[:8] == PNG_MAGIC
    assert "—" in caption  # the author rides the caption


def test_quote_default_format_result_shape_unchanged(tmp_path):
    out = _quote_service(tmp_path).quote_card(
        send=_Send(), backdrop_fn=lambda seed="": None)
    assert out["sent"] is True, out
    assert "format" not in out and "screen_error" not in out


def test_quote_screen_failure_falls_back_to_the_card(
        tmp_path, monkeypatch):
    class _DeadScreens:
        def __init__(self, *, media_dir=""):
            pass

        def render(self, screen, **kw):
            return {"sent": False, "reason": "no renderer here"}

    monkeypatch.setattr(
        "nomorals.social.chat.mock_screens.MockScreenService",
        _DeadScreens)
    out = _quote_service(tmp_path).quote_card(
        format="screen", send=_Send(),
        backdrop_fn=lambda seed="": None)
    assert out["sent"] is True, out
    assert out["format"] == "card"
    assert out["screen_error"] == "no renderer here"


def test_fun_quote_names_the_format_from_the_tail():
    from nomorals.agents.partner.runtime_fun import RuntimeFunMixin

    seen = {}

    class _StubService:
        def quote_card(self, arg, *, format="", send):
            seen.update({"arg": arg, "format": format})
            return {"sent": True}

    runtime = RuntimeFunMixin()
    runtime._fun_service_obj = _StubService()
    message = SimpleNamespace()
    assert runtime._fun_quote("chat dedication", message) == ""
    assert seen == {"arg": "dedication", "format": "screen"}
    assert runtime._fun_quote("card dedication", message) == ""
    assert seen == {"arg": "dedication", "format": "card"}
    assert runtime._fun_quote("dedication", message) == ""
    assert seen == {"arg": "dedication", "format": ""}


# ── the makers surface: screen as a style word ───────────────

def _maker_service(tmp_path):
    return MakerService(media_dir=tmp_path / "maker_media",
                        health=ProviderHealth())


def test_qcard_screen_word_renders_a_screen(tmp_path):
    send = _Send()
    out = _maker_service(tmp_path).card(
        "qcard", "screen Ada | we ship at dawn", send=send)
    assert out["sent"] is True, out
    assert out["style"] == "screen" and out["format"] == "screen"
    ref, _caption = send.calls[0]
    assert Path(ref.path).read_bytes()[:8] == PNG_MAGIC


def test_meme2_screen_word_carries_the_joke_as_a_chat(tmp_path):
    send = _Send()
    out = _maker_service(tmp_path).card(
        "meme2", "screen Ada | the deploy worked on my machine",
        send=send)
    assert out["sent"] is True, out
    assert out["style"] == "screen"


def test_makers_screen_failure_falls_back_to_the_card(
        tmp_path, monkeypatch):
    class _DeadScreens:
        def __init__(self, *, media_dir=""):
            pass

        def render(self, screen, **kw):
            return {"sent": False, "reason": "organ is down"}

    monkeypatch.setattr(
        "nomorals.social.chat.mock_screens.MockScreenService",
        _DeadScreens)
    send = _Send()
    out = _maker_service(tmp_path).card(
        "qcard", "screen Ada | we ship at dawn", send=send)
    assert out["sent"] is True, out
    assert out["style"] == "quote" and out["format"] == "card"
    assert out["screen_error"] == "organ is down"


def test_makers_screen_word_with_no_words_is_a_usage_line(tmp_path):
    out = _maker_service(tmp_path).card("qcard", "screen",
                                        send=_Send())
    assert out["sent"] is False
    assert "usage" in out["reason"]


# ── the Motivation class: format selection + fallback ────────

def _future_ts(hour, minute=0, *, days=7) -> float:
    day = datetime.now(LAGOS).date() + timedelta(days=days)
    return datetime(day.year, day.month, day.day, hour, minute,
                    tzinfo=LAGOS).timestamp()


class _MotivationHarness(unittest.TestCase):
    """The m1 harness shape: real Database + ledger, stubbed
    producers/brain/notifier, per-test tmp home."""

    def _ctx(self, **over):
        from nomorals.storage import Database

        tmp = tempfile.mkdtemp(prefix="mockwire-mot-")
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


class MotivationFormatTests(_MotivationHarness):
    def test_resolve_format_pref_and_rotation(self):
        from nomorals.agents import motivation as mot

        assert mot._resolve_format({}, "morning", "",
                                   day_ordinal=100) == "card"
        assert mot._resolve_format({"format": "chat"}, "morning",
                                   "", day_ordinal=100) == "chat"
        assert mot._resolve_format({"format": "chat"}, "morning",
                                   "card", day_ordinal=100) == "card"
        assert mot._resolve_format({"format": "rotate"}, "morning",
                                   "", day_ordinal=100) == "card"
        assert mot._resolve_format({"format": "rotate"}, "evening",
                                   "", day_ordinal=100) == "chat"
        assert mot._resolve_format({"format": "rotate"}, "morning",
                                   "", day_ordinal=101) == "chat"

    def test_run_with_chat_format_renders_the_screen(self):
        ctx = self._ctx()
        mot = self._wire(ctx)
        res = mot.run_motivation(ctx, "morning", format="chat",
                                 now=_future_ts(7, 5),
                                 rng=random.Random(3))
        assert res["delivered"], res
        assert res["format"] == "chat", res
        assert res["card_rendered"] is True
        screen = (Path(ctx.settings.workspace_dir)
                  / "motivation_cards" / "motivation-morning-chat.png")
        assert screen.read_bytes()[:8] == PNG_MAGIC
        marker = mot._read_marker(ctx)
        assert marker["morning"]["format"] == "chat"

    def test_chat_failure_falls_back_to_the_card(self):
        ctx = self._ctx()
        mot = self._wire(ctx)
        original = mot._render_piece_screen

        def _boom(*a, **kw):
            raise RuntimeError("organ gone")

        mot._render_piece_screen = _boom
        self.addCleanup(setattr, mot, "_render_piece_screen",
                        original)
        res = mot.run_motivation(ctx, "morning", format="chat",
                                 now=_future_ts(7, 5),
                                 rng=random.Random(3))
        assert res["delivered"], res
        assert res["format"] == "card", res
        assert "organ gone" in res["format_error"]
        card = (Path(ctx.settings.workspace_dir)
                / "motivation_cards" / "motivation-morning.png")
        assert card.read_bytes()[:8] == PNG_MAGIC

    def test_tool_now_accepts_the_format_argument(self):
        ctx = self._ctx()
        mot = self._wire(ctx)
        captured = {}

        class _Registry:
            context = ctx

            def register(self, name, description="", capability=None,
                         parameters=None):
                def deco(fn):
                    captured[name] = fn
                    return fn
                return deco

        mot.register(_Registry())
        out = captured["motivation"](action="now", window="morning",
                                     format="chat")
        assert out["delivered"], out
        assert out["format"] == "chat"
