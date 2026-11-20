"""The creative family, whole: rotate default + platform varieties.

The owner's two orders, both read from his own words in the
main chat (not a summary):

- 2026-11-11: "Also should be able to use them for the
  motivation class. Like you know creative way. Same for the
  quotes and memes and quote card and many more" — the mock
  screen was shipped as an opt-in everywhere and his phone
  never changed.  The fix: the whole family defaults to
  ROTATE (card/chat alternate by Lagos day + window, the
  same one rule the Motivation class uses).
- 2026-11-10: "The /quite command and the likes are all the
  same stuff designs should be varieties Facebook x tiktok
  reddit Instagram mastodon tinder POF and more including
  there comment sections." — /quote rendered ONE design
  (bigquote) every day.  The fix: the card side cycles the
  platform registry, comment sections nesting where the
  platform nests.

Explicit per-run words still win (card / chat / screen /
record / a named style); a screen that can't render falls
back to the card with the failure named.  /meme stays out:
it fetches a picture, it has no card/screen machinery to
rotate.  Single-style aliases (fakettc, nokiamsg, fakeff)
stay out too: their identity IS the style.
"""

from __future__ import annotations

import io
import random
from pathlib import Path
from types import SimpleNamespace

from nomorals.social.chat import makers as makers_mod
from nomorals.social.chat.fun import FunService, LocalQuotesProvider
from nomorals.social.chat.makers import (
    MakerService,
    ProviderHealth,
    platform_for_slot,
    resolve_creative_format,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PLATFORMS = ("facebook", "x", "tiktok", "reddit",
             "instagram", "mastodon", "tinder", "pof")


class _Send:
    def __init__(self, ok=True):
        self.calls = []
        self.ok = ok

    def __call__(self, ref, caption=""):
        self.calls.append((ref, caption))
        return SimpleNamespace(ok=self.ok, error="")


def _quote_service(tmp_path, **kw):
    kw.setdefault("media_dir", tmp_path / "fun_media")
    kw.setdefault("chains", {"quote": [LocalQuotesProvider(
        rng=random.Random(3))]})
    return FunService(**kw)


def _maker_service(tmp_path, **kw):
    kw.setdefault("media_dir", tmp_path / "maker_media")
    kw.setdefault("health", ProviderHealth())
    return MakerService(**kw)


def _pin_slot(monkeypatch, day_ordinal, window):
    """Freeze the Lagos slot the family rotates on."""
    monkeypatch.setattr(
        makers_mod, "lagos_day_window",
        lambda *, now=None: (day_ordinal, window))


def _png_bytes():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 30, 30)).save(buf, "PNG")
    return buf.getvalue()


# ── the one rule ─────────────────────────────────────────────

def test_resolve_creative_format_explicit_and_pref_win():
    # Explicit per-run pick beats everything; a pref beats
    # the rotation; with neither, the slot decides.
    assert resolve_creative_format(
        "chat", day_ordinal=100, window="morning") == "chat"
    assert resolve_creative_format(
        "screen", day_ordinal=100, window="morning") == "chat"
    assert resolve_creative_format(
        "card", day_ordinal=101, window="morning") == "card"
    assert resolve_creative_format(
        "", pref="card", day_ordinal=101,
        window="morning") == "card"
    assert resolve_creative_format(
        "", pref="chat", day_ordinal=100,
        window="morning") == "chat"
    assert resolve_creative_format(
        "", pref="rotate", day_ordinal=100,
        window="morning") == "card"
    assert resolve_creative_format(
        "", pref="rotate", day_ordinal=100,
        window="evening") == "chat"
    assert resolve_creative_format(
        "card", pref="chat", day_ordinal=100,
        window="morning") == "card"


def test_resolve_creative_format_rotates_by_day_and_window():
    # The same slot math the Motivation class pins:
    # morning of an even day is the card, morning of an odd
    # day the chat, and the evening is the neighbour.
    assert resolve_creative_format(
        "", day_ordinal=100, window="morning") == "card"
    assert resolve_creative_format(
        "", day_ordinal=101, window="morning") == "chat"
    assert resolve_creative_format(
        "", day_ordinal=100, window="evening") == "chat"
    assert resolve_creative_format(
        "", day_ordinal=101, window="evening") == "card"


def test_platform_for_slot_cycles_the_whole_registry():
    # Eight consecutive CARD days (even ordinals, morning)
    # walk all eight platforms, in registry order from the
    # pair the cycle starts on — and the cycle wraps.
    seen = [platform_for_slot(day, "morning")
            for day in range(100, 115, 2)]
    assert seen == ["tiktok", "reddit", "instagram",
                    "mastodon", "tinder", "pof",
                    "facebook", "x"]
    assert len(set(seen)) == 8
    assert set(seen) == set(PLATFORMS)
    assert platform_for_slot(116, "morning") == "tiktok"
    # Morning and evening of one day share the platform —
    # the format is what differs between them.
    assert platform_for_slot(100, "evening") == "tiktok"


def test_motivation_resolve_delegates_to_the_one_rule():
    from nomorals.agents import motivation as mot

    # The Motivation class's pins, through the shared
    # implementation — if this drifts, the family has two
    # rules again.
    assert mot._resolve_format({}, "morning", "",
                               day_ordinal=100) == "card"
    assert mot._resolve_format({}, "morning", "",
                               day_ordinal=101) == "chat"
    assert mot._resolve_format({}, "evening", "",
                               day_ordinal=100) == "chat"
    assert mot._resolve_format({"format": "card"}, "morning",
                               "", day_ordinal=101) == "card"


# ── /quote: the default rotates, the card varies ─────────────

def test_plain_quote_card_slot_renders_the_platform(
        tmp_path, monkeypatch):
    _pin_slot(monkeypatch, 100, "morning")
    out = _quote_service(tmp_path).quote_card(
        send=_Send(), backdrop_fn=lambda seed="": None)
    assert out["sent"] is True, out
    assert out["drawn"]["style"] == "tiktok", out
    assert "format" not in out and "screen_error" not in out


def test_plain_quote_chat_slot_stages_the_screen(
        tmp_path, monkeypatch):
    _pin_slot(monkeypatch, 101, "morning")
    send = _Send()
    out = _quote_service(tmp_path).quote_card(
        send=send, backdrop_fn=lambda seed="": None)
    assert out["sent"] is True, out
    assert out["format"] == "screen", out
    ref, _caption = send.calls[0]
    assert Path(ref.path).read_bytes()[:8] == PNG_MAGIC


def test_plain_quote_card_days_land_on_distinct_platforms(
        tmp_path, monkeypatch):
    styles = []
    for day in range(100, 115, 2):  # card days only — odd
        # days stage the screen, not a card
        _pin_slot(monkeypatch, day, "morning")
        out = _quote_service(tmp_path).quote_card(
            send=_Send(), backdrop_fn=lambda seed="": None)
        assert out["sent"] is True, out
        styles.append(out["drawn"]["style"])
    assert styles == ["tiktok", "reddit", "instagram",
                      "mastodon", "tinder", "pof",
                      "facebook", "x"]
    assert len(set(styles)) == 8


def test_plain_quote_evening_is_the_mornings_neighbour(
        tmp_path, monkeypatch):
    _pin_slot(monkeypatch, 100, "evening")
    out = _quote_service(tmp_path).quote_card(
        send=_Send(), backdrop_fn=lambda seed="": None)
    assert out["sent"] is True, out
    assert out["format"] == "screen", out


def test_quote_explicit_card_word_names_the_card_side(
        tmp_path, monkeypatch):
    # "card" forces the card format even on a chat slot —
    # and the card is still the platform of the slot.
    _pin_slot(monkeypatch, 101, "morning")
    out = _quote_service(tmp_path).quote_card(
        format="card", send=_Send(),
        backdrop_fn=lambda seed="": None)
    assert out["sent"] is True, out
    assert out["drawn"]["style"] == "tiktok", out
    assert "format" not in out


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
    assert runtime._fun_quote("record dedication", message) == ""
    assert seen == {"arg": "dedication", "format": "record"}


# ── /qcard + /fakemsg + /meme2: the same default ────────────

def test_plain_qcard_card_slot_renders_the_platform(
        tmp_path, monkeypatch):
    _pin_slot(monkeypatch, 100, "morning")
    send = _Send()
    out = _maker_service(tmp_path).card(
        "qcard", "Ada | we ship at dawn", send=send)
    assert out["sent"] is True, out
    assert out["style"] == "tiktok", out
    assert "format" not in out
    ref, caption = send.calls[0]
    assert Path(ref.path).read_bytes()[:8] == PNG_MAGIC
    assert "TikTok" in caption


def test_plain_qcard_chat_slot_stages_the_screen(
        tmp_path, monkeypatch):
    _pin_slot(monkeypatch, 101, "morning")
    out = _maker_service(tmp_path).card(
        "qcard", "Ada | we ship at dawn", send=_Send())
    assert out["sent"] is True, out
    assert out["style"] == "screen" and out["format"] == "screen"


def test_plain_qcard_comments_nest_on_the_platform(
        tmp_path, monkeypatch):
    _pin_slot(monkeypatch, 100, "morning")
    out = _maker_service(tmp_path).card(
        "qcard",
        "Ada | hello world\n> Bola | nice one\n"
        ">> Ada | thanks — nested",
        send=_Send())
    assert out["sent"] is True, out
    assert out["style"] == "tiktok", out
    drawn = out["drawn"]
    assert drawn["names"] == ["Ada", "Bola", "Ada"]
    assert len(drawn["texts"]) == 3
    assert len(drawn["regions"]["comments"]) == 2
    assert drawn["fingerprint"]["comment_indent"] == 0


def test_plain_qcard_comments_nest_with_the_platforms_indent(
        tmp_path, monkeypatch):
    # Facebook indents its comment thread; the same words on
    # a Facebook slot carry that indent in the drawn record.
    _pin_slot(monkeypatch, 112, "morning")
    out = _maker_service(tmp_path).card(
        "qcard",
        "Ada | hello world\n> Bola | nice one",
        send=_Send())
    assert out["sent"] is True, out
    assert out["style"] == "facebook", out
    assert out["drawn"]["fingerprint"]["comment_indent"] == 44
    assert len(out["drawn"]["regions"]["comments"]) == 1


def test_plain_fakemsg_renders_the_platform(
        tmp_path, monkeypatch):
    _pin_slot(monkeypatch, 100, "morning")
    out = _maker_service(tmp_path).card(
        "fakemsg", "Ada | we ship at dawn", send=_Send())
    assert out["sent"] is True, out
    assert out["style"] == "tiktok", out


def test_plain_meme2_rotates_but_keeps_its_meme_card(
        tmp_path, monkeypatch):
    _pin_slot(monkeypatch, 100, "morning")
    out = _maker_service(tmp_path).card(
        "meme2", "Ada | the deploy worked on my machine",
        send=_Send())
    assert out["sent"] is True, out
    assert out["style"] == "meme", out
    _pin_slot(monkeypatch, 101, "morning")
    out = _maker_service(tmp_path).card(
        "meme2", "Ada | the deploy worked on my machine",
        send=_Send())
    assert out["sent"] is True, out
    assert out["format"] == "screen", out


def test_named_style_wins_over_the_rotation(
        tmp_path, monkeypatch):
    # A named style is that platform's card, always — even
    # on a chat slot, where the rotation would stage a
    # screen.
    _pin_slot(monkeypatch, 101, "morning")
    out = _maker_service(tmp_path).card(
        "qcard", "facebook Ada | we ship at dawn", send=_Send())
    assert out["sent"] is True, out
    assert out["style"] == "facebook", out
    assert "format" not in out
    out = _maker_service(tmp_path).card(
        "qcard", "ttc Ada | we ship at dawn", send=_Send())
    assert out["sent"] is True, out
    assert out["style"] == "ttc", out


def test_card_word_forces_the_card_side(
        tmp_path, monkeypatch):
    _pin_slot(monkeypatch, 101, "morning")
    out = _maker_service(tmp_path).card(
        "qcard", "card Ada | we ship at dawn", send=_Send())
    assert out["sent"] is True, out
    assert out["style"] == "tiktok", out
    assert "format" not in out


def test_record_word_alone_records_the_chat(
        tmp_path, monkeypatch):
    out = _maker_service(
        tmp_path,
        bytes_fetcher=lambda url: _png_bytes()).card(
            "qcard", "record Ada | we ship at dawn",
            send=_Send())
    assert out["sent"] is True, out
    assert out["format"] == "record", out


def test_rotate_screen_failure_falls_back_to_the_platform_card(
        tmp_path, monkeypatch):
    class _DeadScreens:
        def __init__(self, *, media_dir=""):
            pass

        def render(self, screen, **kw):
            return {"sent": False, "reason": "organ is down"}

    monkeypatch.setattr(
        "nomorals.social.chat.mock_screens.MockScreenService",
        _DeadScreens)
    _pin_slot(monkeypatch, 101, "morning")
    send = _Send()
    out = _maker_service(tmp_path).card(
        "qcard", "Ada | we ship at dawn", send=send)
    assert out["sent"] is True, out
    assert out["format"] == "card", out
    assert out["style"] == "tiktok", out
    assert out["screen_error"] == "organ is down"


# ── the deliberate outs ──────────────────────────────────────

def test_single_style_aliases_do_not_rotate(
        tmp_path, monkeypatch):
    # fakettc IS the ttc skin and nokiamsg IS the chat skin
    # — rotating them would destroy what the command is.
    _pin_slot(monkeypatch, 101, "morning")
    out = _maker_service(tmp_path).card(
        "fakettc", "Ada | we ship at dawn", send=_Send())
    assert out["sent"] is True, out
    assert out["style"] == "ttc", out
    out = _maker_service(tmp_path).card(
        "nokiamsg", "Ada | we ship at dawn", send=_Send())
    assert out["sent"] is True, out
    assert out["style"] == "chat", out


def test_meme_stays_a_fetched_picture(tmp_path):
    # /meme has no card/screen machinery to rotate: it
    # fetches a picture and attaches it.  It stays out of
    # the family, honestly — the fetch path returns an image
    # with no format at all.
    class _MemeProvider:
        name = "meme-stub"

        def image(self, arg):
            return {"image_url": "https://example.test/meme.png",
                    "caption": "a meme"}

    service = FunService(
        media_dir=tmp_path / "fun_media",
        chains={"meme": [_MemeProvider()]},
        bytes_fetcher=lambda url: _png_bytes())
    send = _Send()
    out = service.image("meme", "", send=send)
    assert out == {"sent": True, "source": "meme-stub"}, out
    ref, _caption = send.calls[0]
    assert ref.kind == "image"
    assert Path(ref.path).read_bytes()[:8] == PNG_MAGIC
